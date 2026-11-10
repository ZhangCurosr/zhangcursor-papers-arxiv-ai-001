# Can LLM Agents Organize Their Own Memory?

Zichuan Fu<sup>1</sup>\*, Junjia Qi<sup>1</sup>\*, Wenlin Zhang<sup>1</sup>, Yimin Deng<sup>1</sup>, Shirong Wang<sup>1</sup>, Jinyang Li<sup>1</sup>, Xian Wu<sup>2</sup>, Xiangyu Zhao<sup>1†</sup>

<sup>1</sup>City University of Hong Kong, <sup>2</sup>Tencent Jarvis Lab

zc.fu@my.cityu.edu.hk, junjiaqi2-c@my.cityu.edu.hk, wl.z@my.cityu.edu.hk shirowang6-c@cityu.edu.hk, jinyali2-c@my.cityu.edu.hk kevinxwu@tencent.com, xianzhao@cityu.edu.hk

## Abstract

In most LLM agents, weights are fixed after deployment, so new information must be retained in external memory. Existing approaches often rely on predefined records and fixed organizational rules, which can not readily ac commodate evolving tasks and interactions. This raises the question: can LLM agents organize their own memory using ordinary documents? We introduce self-organized memory (SOMEM), a framework that stores memory in linked documents. The agent decides what to summarize, how to group related information, and when to revise existing topics, and every summary it produces stays linked to the original interactions as evidence. The runtime validates updates and automatically derives temporal, relational, and recent-memory views from the organized content. At query time, the agent chooses what to search and read, following document links and inspecting original evidence as needed. On LoCoMo, BEAM-100K, and LongMemEval-S, SOMEM outperforms the strongest baseline by 2.66, 3.10, and 2.00 percentage points, respectively. The constructed memory can be reused across models, and additional reasoning effort can further improve answer quality. These findings demonstrate the feasibility of agents organiz ing and using their own persistent memory through ordinary documents. Code is available at: https://github.com/Qi202/SoMem.

## 1 Introduction

During deployment, LLM agents encounter new information, including user preferences, task progress, and changes in the environment. This information is not incorporated into their fixed model parameters. External memory preserves information across interactions and supplies relevant records as context for later requests (Zhong et al.,

2024; Packer et al., 2023). This allows an agent to personalize its responses, continue earlier activities, and use background knowledge acquired through experience (Zhong et al., 2024; Park et al., 2023). As conversations extend across sessions, such context can include facts and events that are absent from the current request but necessary for interpreting it (Maharana et al., 2024; Wu et al., 2025). This persistent context makes information from earlier interactions available alongside the model’s pretrained knowledge.

Designing agent memory requires deciding what to save, how to organize and update it, and how to access it for a task (Du, 2026). These decisions are connected: the information retained during construction determines the evidence available during retrieval and reading (Wu et al., 2025). Complete histories preserve detail but require the agent to locate relevant exchanges; compact records reduce the amount of text to inspect but may omit qualifications or context needed later (Wu et al., 2025; Fang et al., 2026). New interactions can also revise an earlier fact, connect separate events, or change the period during which a statement holds (Wu et al., 2025; Li et al., 2026). A memory system must therefore support both concise access to accumulated information and recovery of the evidence needed to interpret changes.

Existing methods address these requirements through different representations and operations. MemoryBank and Mem0 extract and update compact records, while Mem0 also offers an entity– relation graph (Zhong et al., 2024; Chhikara et al., 2025). A-Mem generates attributed notes and revises their associations as new memories arrive; LightMem and TiMem consolidate information through processing stages or temporal hierarchies (Xu et al., 2025; Fang et al., 2026; Li et al., 2026). Model-controlled memory operations are also established in MemGPT and the vectorbacked semantic file interface of LSFS (Packer et al., 2023; Shi et al., 2025). Building on such model-controlled operations, we study how editable semantic organization, preserved original evidence, and automatically derived access views can maintain consistency as memory changes. The agent revises the semantic content, while the runtime validates evidence references and reconstructs the access views from accepted edits. The central question is therefore how an agent can revise its memory organization while preserving evidence and consistent access.

![](images/db8c1950278f71bec0efc45edd9ed1335d93ea0740d7eb0b6f201191eff9aac5.jpg)  
Figure 1: Overview of SOMEM, illustrated with a meeting-venue update. The agent revises Topic documents from Source records, and the runtime derives access views from the revised documents. At query time, the agent searches and reads the organized memory and its Source evidence to answer the question. The tools shown are the combined search and read-only shell interface; the main experiments use the equivalent split interface (Appendix A).

We propose self-organized memory (SOMEM), in which the model controls semantic organization and the runtime maintains evidence and access consistency. First, the agent chooses how to group and revise information in Topic documents, which reference immutable Source records and from which a small Core is promoted for persistent context. Second, the runtime applies the same validation checks to every revision. Once a revision is accepted, it rebuilds the access views from the updated content and metadata. This allows semantic grouping to change without requiring the model to maintain separate versions for each access structure. Third, the agent selects further searches and reads from its observations, returning to original records when organized content is insufficient. The same agent can write and read memory, and other models can reuse the resulting workspace through the same interface.

Our contributions are as follows:

• We formulate self-organized memory as editable semantic content that retains links to its supporting evidence. Let the model decide how that content is grouped, while the runtime maintains the access structures that serve retrieval.

• We develop a reorganization procedure that validates edits and reconstructs access views, coupled with sequential inspection of the evidence needed to answer a question.

• We demonstrate the effectiveness of SOMEM through comparisons with baselines on LoCoMo, BEAM-100K, and LongMemEval-S, using a shared GPT-6 Luna Reader. We also show that SOMEM works across multiple backbones and that the constructed memory can be reused across models.

## 2 Method

SOMEM lets an agent organize persistent memory through editable Markdown documents (Figure 1). The agent uses Source records to construct and revise Topic documents, from which a small Core is promoted for persistent context. Deterministic code derives Graph, Timeline, and Recent views from Topic. The agent retrieves evidence using search and document access tools. Implementation details are provided in the Appendix A.

## 2.1 Self-Organizing Memory Documents

Information about the same person, activity, or task can accumulate across many interactions, and later observations may change how earlier information should be grouped. Predefined memory schemas constrain how this grouping can change (Xu et al., 2025), while storing information as isolated records can leave related evidence scattered across separate items (Ji et al., 2026). We therefore let the agent organize memory in editable, linked documents whose subjects and internal structure can evolve with the content.

At update t, the system archives new interaction records $x _ { t }$ as Source records, preserving their original content for later reference. The agent uses Source and existing memory to update Topic and Core:

$$
\begin{array} { c } { S _ { t } = \mathrm { A p p e n d } ( S _ { t - 1 } , x _ { t } ) , } \\ { ( \mathcal { T } _ { t } , \mathcal { C } _ { t } ) = \mathrm { O r g a n i z e } ( S _ { t } , \mathcal { T } _ { t - 1 } , \mathcal { C } _ { t - 1 } ) . } \end{array}\tag{1}
$$

Here, $S _ { t }$ contains the archived Source records, and $\mathcal { T } _ { t }$ is the Topic document collection. Core, denoted by $\mathcal { C } _ { t } ~ \in ~ \mathcal { T } _ { t }$ , is a size-bounded Markdown file within this collection. The agent promotes longterm preferences, standing requirements, and frequently needed background from Topic documents into Core. Core is placed in the initial context of every query. The agent reads relevant Source records and existing documents to decide what to retain and where it belongs. It can create or revise paragraphs and documents, adjust headings, split broad topics, and regroup related information. Edits are staged before acceptance. The runtime checks Source immutability, document structure, and the resolution of evidence references and links, then installs accepted edits and rebuilds the derived views; rejected edits preserve the prior memory state. Periodic organization revisits changed Topic files, while an optional final pass reorganizes the complete Topic collection.

Figure 2 illustrates this structure with two employment events in a single Topic paragraph. Each event has its own time annotation and Source references. A link within the paragraph points to a specific paragraph in the related project document. The agent determines each Markdown document’s topic and internal structure. Within each document, it uses headings to define subtopics and groups related information into paragraphs. It also links specific paragraphs across documents through references. A paragraph is a memory unit; unit i and

## (a) Editable Topic document

![](images/1091ce59f000999cdb84962c6f1673e84396fa2d38d98a140c168678650cc9be.jpg)  
Figure 2: An example Topic document. Each statement is annotated with its event time and supporting Source records, and paragraph references link related Topic documents across the workspace.

its evidence annotations are represented as:

$$
\begin{array} { r l } & { u _ { i } = ( b _ { i } , c _ { i } , E _ { i } , L _ { i } ) , } \\ & { E _ { i } = \{ ( s _ { i j } , \tau _ { i j } , R _ { i j } ) \} _ { j = 1 } ^ { n _ { i } } . } \end{array}\tag{2}
$$

Here, $b _ { i }$ is a stable paragraph identifier, $c _ { i }$ is the paragraph text, and $L _ { i }$ contains the identifiers of referenced memory units. Each of the $n _ { i }$ annotations associates a statement span $s _ { i j }$ in $c _ { i }$ with an event time $\tau _ { i j }$ and supporting references $R _ { i j }$ to Source records in $S _ { t }$ . In Markdown, a footnote marker follows the corresponding statement; its definition records the time and source links, and a trailing paragraph identifier makes the unit addressable from other documents. Event times preserve the temporal precision of the supporting Source records: year, month, or day.

## 2.2 Automatic Memory View Derivation

Questions may access memory by relation, time, or recency (Wu et al., 2025), which a single topical organization does not directly support. Maintaining a separate representation for each access pattern, however, requires propagating every revision across them and increases update overhead (Fang et al., 2026). In SOMEM, the runtime derives access views from the Topic documents maintained by the agent, using paragraph references, event times, and the creation order of memory units:

$$
\begin{array} { r l } & { \mathcal { G } _ { t } = \mathrm { G r a p h } ( \mathcal { T } _ { t } ) , } \\ & { \mathcal { L } _ { t } = \mathrm { T i m e l i n e } ( \mathcal { T } _ { t } ) , } \\ & { \mathcal { R } _ { t } = \mathrm { R e c e n t } _ { K } ( \mathcal { T } _ { t } ) . } \end{array}\tag{3}
$$

Here, $\mathcal { G } _ { t } , \mathcal { L } _ { t }$ , and $\mathcal { R } _ { t }$ denote Graph, Timeline, and Recent, respectively. These operations construct the views without requiring the model to write each representation separately.

Graph. Each Topic memory unit is a node identified by $b _ { i } ;$ a reference $b _ { j } \in L _ { i }$ defines a directed edge from unit i to unit $j$ . The system parses these references into a Relations index that records outgoing links and backlinks for each node. The agent can therefore access both the memories cited by a paragraph and those that cite it. Graph thus uses the existing Topic structure without a separate relation extraction step.

Timeline. For each evidence annotation, the system groups its statement $s _ { i j }$ into a Markdown file according to the event time $\tau _ { i j }$ , retaining its Source references $R _ { i j }$ . File paths preserve the original temporal precision: years, months, and days correspond to timeline/YYYY.md, timeline/YYYY/MM. md, and timeline/YYYY/MM/DD.md, respectively. Each file begins with a level-one heading specifying the corresponding time, followed by individual event entries. Each entry contains the event statement, a reference to its original Topic paragraph, and references to the supporting Source messages.

Recent. The system records the creation order of each memory unit. Recent $, K$ uses this order to select the K most recently created memory units from Topic. The resulting view retains each unit’s paragraph content, Topic location, and source references. Recent updates automatically with memory writes and is included in the initial context of each query.

## 2.3 Multi-View Memory Retrieval

Relevant evidence depends on the question: a topic match may miss temporal constraints, while a compact fact may omit needed context (Wu et al., 2025). Such evidence is often identified only after related memory has been read, so a single retrieval before answering may be insufficient. SOMEM provides read-only operations for content search, file listing, and document reading:

$$
\mathcal { F } = \{ \mathrm { S e a r c h } , \mathrm { L i s t } , \mathrm { R e a d } \} .
$$

Content search. The agent retrieves Topic and Source passages with BM25 (Robertson and Zaragoza, 2009) lexical matching or dense semantic search, choosing the mode for each query. Optional path and date-range filters restrict the search to relevant documents or periods. Each result provides a passage and its location, allowing the agent to inspect the surrounding context.

Document and view access. File-listing and reading operations let the agent locate documents and inspect their contents, including selected line ranges. The agent can inspect Topic sections, read Timeline files for relevant dates, or consult Graph links and backlinks to locate related paragraphs. Source references in Topic and Timeline allow it to check statements against the original interactions. These operations can be combined according to the question, without a prescribed retrieval sequence.

Given a question $q ,$ the agent receives Core, Recent, and a file inventory as its initial memory context. The question date is also provided when available. It uses these tools to retrieve relevant memory and generate an answer supported by the evidence:

$$
\begin{array} { r } { \mathcal { M } _ { t } = ( S _ { t } , \mathcal { T } _ { t } , \mathcal { C } _ { t } , \mathcal { G } _ { t } , \mathcal { L } _ { t } , \mathcal { R } _ { t } ) , } \\ { y = \mathrm { A g e n t } ( q ; \mathcal { M } _ { t } , \mathcal { F } ) . \qquad } \end{array}\tag{4}
$$

Here, $\mathcal { M } _ { t }$ comprises the documents and views constructed above, $\mathcal { F }$ denotes the available read-only access operations, and $y$ is the generated answer. The documents and views remain available for further inspection through the same tool interface.

## 3 Experiments

We compare memory methods in their full configurations on LoCoMo, BEAM-100K, and LongMemEval-S, then examine component ablations, cross-model memory reuse, and Reader effort.

## 3.1 Experimental Setup

Datasets. We evaluate SOMEM on three longterm memory benchmarks: LoCoMo (Maharana et al., 2024), LongMemEval-S (Wu et al., 2025), and BEAM-100K (Tavakoli et al., 2025). Lo-CoMo tests single-hop (SH), multi-hop (MH), open-domain (OD), and temporal (TR) reasoning over multi-session histories; LongMemEval-S covers knowledge update (KU), multi-session reasoning (MS), single-session assistant (SSA), preference (SSP), and user (SSU) information, and temporal reasoning (TR); BEAM-100K covers ten memory capabilities: information extraction (IE), multi-session reasoning (MR), temporal reasoning (TR), knowledge update (KU), contradiction resolution (CR), event ordering (EO), instruction following (IF), preference following (PF), summarization (SUM), and abstention (ABS). Table 1 summarizes the evaluation and analysis-subset sizes. The main method comparisons use the complete evaluation sets; component ablations use the full LoCoMo set, and Reader-effort analyses use fixed subsets. Subsets are stratified by conversation and question type and selected independently of model outputs and scores. Within each analysis, all conditions use the same subsets.

Table 1: Statistics of the evaluation datasets, including full evaluation sets and Reader-effort subsets. Component ablations use the full LoCoMo set.
<table><tr><td>Dataset</td><td>Task scope</td><td>Full / subset</td></tr><tr><td>LoCoMo</td><td>4 QA types</td><td>1,540 / 100</td></tr><tr><td>LongMemEval-S</td><td>6 QA types</td><td>500 / -</td></tr><tr><td>BEAM-100K</td><td>10 capabilities</td><td>400 / 200</td></tr></table>

Evaluation Metrics. We use GPT-4o-mini as the judge with temperature 0 to evaluate complete, unmodified answers. For LoCoMo and LongMemEval-S, the judge assesses correctness based on the question, reference answer, and taskspecific requirements. We report LLM-judge (LJ) accuracy by question type and overall. Overall accuracy is the proportion of correctly answered questions across the evaluation set. For BEAM-100K, each rubric criterion receives a score of 0, 0.5, or 1; we average criteria within each question and then average over questions to obtain the overall nugget score. We also report the pass rate, counting questions with a score of at least 0.5 as passes. Evaluation protocols and configurations are provided in Appendix B.

Implementation Details. We compare SOMEM with representative memory methods: A-Mem (Xu et al., 2025), Mem0 (Chhikara et al., 2025), MemoryOS (Kang et al., 2025), Nemori (Ma et al., 2026), LightMem (Fang et al., 2026), and E-mem (Wang et al., 2026a). In the primary comparisons, GPT-6 Luna serves as both the memory-construction model (Writer) and the question-answering model (Reader) for every memory method, while each method retains its own memory organization and retrieval pipeline; no Reader uses reasoning effort. Full Context uses the same Reader and answers directly from the complete original history, without truncation or retrieval tools. We additionally report SOMEM with other backbones, each serving as both Writer and Reader; these rows are excluded from the ranking. Bold and underline mark the best and second-best distinct values among methods with the shared Reader, including ties. The crossmodel and Reader-effort analyses further use GPT-5.6 Luna, Gemini 3 Flash, DeepSeek V4 Flash, Qwen 3.8 Flash, GPT-6.1 Sol, and GLM-5.3 Flash. Additional implementation details and evaluation protocols are provided in Appendices A and B.

Table 2: LoCoMo LJ accuracy (%) on all 1,540 category 1–4 questions.
<table><tr><td>Method</td><td>SH</td><td>MH</td><td>OD</td><td>TR</td><td>Overall</td></tr><tr><td>Full Context</td><td>94.17</td><td>80.85</td><td>45.83</td><td>72.59</td><td>84.22</td></tr><tr><td>A-Mem</td><td>62.78</td><td>55.67</td><td>41.67</td><td>62.31</td><td>60.06</td></tr><tr><td>Mem0</td><td>83.83</td><td>76.95</td><td>53.12</td><td>82.24</td><td>80.32</td></tr><tr><td>MemoryOS</td><td>60.05</td><td>56.03</td><td>43.75</td><td>50.16</td><td>56.23</td></tr><tr><td>Nemori</td><td>79.79</td><td>67.73</td><td>43.75</td><td>66.36</td><td>72.53</td></tr><tr><td>LightMem</td><td>82.88</td><td>70.21</td><td>52.08</td><td>79.75</td><td>77.99</td></tr><tr><td>E-mem</td><td>92.75</td><td>85.11</td><td>58.33</td><td>78.82</td><td>86.30</td></tr><tr><td>SOMEM</td><td>94.53</td><td>83.33</td><td>59.38</td><td>88.16</td><td>88.96</td></tr><tr><td> $\mathrm { { \bf S 0 M E M ^ { \dag } } }$ </td><td>95.96</td><td>93.26</td><td>65.62</td><td>91.90</td><td>92.73</td></tr><tr><td> ${ \bf S } { \bf O } { \bf M } { \bf E } { \bf M } ^ { \ddagger }$ </td><td>95.36</td><td>91.49</td><td>72.92</td><td>90.97</td><td>92.34</td></tr><tr><td> $\mathrm { S o M E M ^ { \ S } }$ </td><td>94.89</td><td>91.13</td><td>63.54</td><td>89.72</td><td>91.17</td></tr></table>

Writer and Reader: <sup>†</sup> DeepSeek V4 Flash; <sup>‡</sup> GPT-5.6 Luna; <sup>§</sup> Gemini 3 Flash. These rows are excluded from the ranking; all other rows use GPT-6 Luna.

## 3.2 Main Results

Across the benchmarks (Tables 2–4), SOMEM achieves the highest overall scores among the methods evaluated with the shared GPT-6 Luna Reader. The comparisons include direct full-context answering. In Table 2, its advantages in single-hop recall and temporal reasoning highlight the effectiveness of locating specific facts and retaining their context. Topic organization makes related information accessible, while links to Source allow the agent to retrieve the precise details needed to answer these questions. Results with additional backbones also demonstrate the applicability of this representation across models.

Table 3 shows strengths in temporal reasoning, event ordering, and instruction following over long histories. These tasks require connecting events or requirements scattered across interactions and interpreting them in context. Organizing related content by topic and linking it to time-stamped original evidence supports this process: relevant statements can be located together and checked against the history in which they occurred. This provides a coherent basis for relating distant events and recovering instructions whose meaning depends on earlier exchanges.

Table 3: BEAM-100K results (%) on 400 questions. Capability columns report mean nugget scores; the final columns report overall nugget score and pass rate.
<table><tr><td>Method</td><td>IE</td><td>MR</td><td>TR</td><td>KU</td><td>CR</td><td>EO</td><td>IF</td><td>PF</td><td>SUM</td><td>ABS</td><td>Nugget</td><td>Pass</td></tr><tr><td>Full Context</td><td>82.86</td><td>50.25</td><td>41.25</td><td>50.00</td><td>25.63</td><td>36.95</td><td>56.25</td><td>84.38</td><td>49.14</td><td>32.50</td><td>50.92</td><td>55.25</td></tr><tr><td>A-Mem</td><td>76.82</td><td>54.30</td><td>41.25</td><td>46.25</td><td>23.44</td><td>39.78</td><td>66.88</td><td>86.25</td><td>40.11</td><td>30.00</td><td>50.51</td><td>55.75</td></tr><tr><td>Mem0</td><td>62.29</td><td>50.50</td><td>28.75</td><td>58.75</td><td>30.00</td><td>43.99</td><td>68.12</td><td>86.88</td><td>35.70</td><td>25.00</td><td>49.00</td><td>53.25</td></tr><tr><td>MemoryOS</td><td>60.94</td><td>60.49</td><td>34.38</td><td>39.38</td><td>26.88</td><td>32.93</td><td>63.75</td><td>82.08</td><td>36.24</td><td>42.50</td><td>47.96</td><td>50.75</td></tr><tr><td>Nemori</td><td>63.96</td><td>48.59</td><td>24.38</td><td>46.25</td><td>24.06</td><td>41.41</td><td>73.75</td><td>85.00</td><td>44.72</td><td>27.50</td><td>47.96</td><td>51.50</td></tr><tr><td>LightMem</td><td>62.29</td><td>52.88</td><td>31.87</td><td>42.50</td><td>30.31</td><td>29.72</td><td>61.25</td><td>77.50</td><td>18.37</td><td>28.75</td><td>43.54</td><td>47.00</td></tr><tr><td>E-mem</td><td>59.58</td><td>45.36</td><td>30.00</td><td>31.25</td><td>51.56</td><td>41.96</td><td>54.37</td><td>76.25</td><td>25.77</td><td>17.50</td><td>43.36</td><td>49.00</td></tr><tr><td>SOMEM</td><td>73.70</td><td>51.53</td><td>53.75</td><td>48.75</td><td>36.56</td><td>46.84</td><td>73.75</td><td>85.62</td><td>45.94</td><td>23.75</td><td>54.02</td><td>61.50</td></tr></table>

In Table 4, SOMEM is particularly effective at recalling specific information from both user and assistant interactions. Preserving both sides of the original dialogue keeps details omitted from topic summaries available for later inspection and helps answer questions that require precise historical content. The observed strengths highlight the value of retaining detailed evidence alongside organized summaries for long-term conversational recall.

## 3.3 Ablation Study

Table 5 compares Full with component removals on all LoCoMo evaluation questions. We freeze memory and fix the Reader (GPT-6 Luna without reasoning effort) and query settings, changing only query-time access; w/o Topic restricts the Reader to Source records. Additional settings, context measurements, and paired results are provided in Appendix C.1.

Removing either Topic or Source reduces overall accuracy, with paired confidence intervals excluding zero in both cases (Table 9), supporting the complementary roles of organized content and original evidence. Topic groups dispersed facts into thematic entry points, and Source preserves the original details and context needed to verify them. Removing Source produces the larger decline, indicating that organized content complements rather than replaces the original records.

The remaining components mainly shift accuracy across question types: removing Core or any single derived view changes overall accuracy by at most 1.30 points, and the paired intervals include zero (Table 9). Core is a small, bounded subset of Topic that holds long-term preferences, standing requirements, and frequently needed background, and that the Reader receives at the start of every query, much like a system prompt. LoCoMo contains no question type dedicated to preferences or standing instructions, so removing Core leaves overall accuracy unchanged; its role is to carry persistent constraints rather than to supply factual evidence. Removing Recent lowers single-hop accuracy from 94.53 to 92.51 while raising temporal accuracy from 88.16 to 89.10. Recent preloads entries selected by memory creation order, providing readily available factual cues for direct recall. Removing this preload substantially shortens the input to the final answer-producing call while leaving the average call count similar (Table 8). The Reader can therefore continue to inspect Topic and Source with a more compact input, consistent with the different evidence-selection demands of temporal questions. Removing Timeline or Graph each lowers temporal accuracy to 86.92, consistent with time-ordered entries and cross-document links both helping to relate events. Together, these results indicate that Topic and Source are the main contributors to overall accuracy. The derived views provide access paths for specific question types, while Core carries persistent constraints that LoCoMo rarely tests. The effects of the derived views and Core on overall accuracy remain within sampling uncertainty.

## 3.4 Impact of Backbone Models

We evaluate Writer–Reader combinations on Lo-CoMo, using GPT-5.6 Luna, Gemini 3 Flash, and DeepSeek V4 Flash as Writers and these models together with Qwen 3.8 Flash as Readers. We compare memories built by different Writers with a fixed Reader and different Readers on the same memory.

As shown in Table 6, memory built by GPT-5.6 Luna achieves higher overall accuracy with most Readers, and its accuracy averaged across

Table 4: LongMemEval-S LJ accuracy (%) on all 500 questions.
<table><tr><td>Method</td><td>KU</td><td>MS</td><td>SSA</td><td>SSP</td><td>SSU</td><td>TR</td><td>Overall</td></tr><tr><td>Full Context</td><td>92.31</td><td>51.13</td><td>96.43</td><td>53.33</td><td>90.00</td><td>42.11</td><td>65.80</td></tr><tr><td>A-Mem</td><td>75.64</td><td>49.62</td><td>96.43</td><td>30.00</td><td>92.86</td><td>45.86</td><td>62.80</td></tr><tr><td>Mem0</td><td>91.03</td><td>60.90</td><td>76.79</td><td>73.33</td><td>91.43</td><td>62.41</td><td>72.80</td></tr><tr><td>MemoryOS</td><td>73.08</td><td>36.09</td><td>87.50</td><td>66.67</td><td>72.86</td><td>30.08</td><td>53.00</td></tr><tr><td>Nemori</td><td>79.49</td><td>51.88</td><td>82.14</td><td>33.33</td><td>92.86</td><td>60.15</td><td>66.40</td></tr><tr><td>LightMem</td><td>91.67</td><td>75.56</td><td>25.00</td><td>86.67</td><td>92.86</td><td>71.80</td><td>74.50</td></tr><tr><td>E-mem</td><td>87.18</td><td>46.62</td><td>94.64</td><td>26.67</td><td>85.71</td><td>56.39</td><td>65.20</td></tr><tr><td>SOMEM</td><td>77.56</td><td>58.65</td><td>99.11</td><td>73.33</td><td>98.57</td><td>73.31</td><td>76.50</td></tr><tr><td>SOMEM†</td><td>91.03</td><td>83.46</td><td>100.00</td><td>76.67</td><td>100.00</td><td>90.23</td><td>90.20</td></tr></table>

Writer and Reader: <sup>†</sup> DeepSeek V4 Flash. This row is excluded from the ranking; all other rows use GPT-6 Luna.

Table 5: LoCoMo component ablations on all 1,540 questions (LJ accuracy, %). Bold marks the Full reference, not the best score.
<table><tr><td>Condition</td><td>SH</td><td>MH</td><td>OD</td><td>TR</td><td>Overall</td></tr><tr><td>Full</td><td>94.53</td><td>83.33</td><td>59.38</td><td>88.16</td><td>88.96</td></tr><tr><td>w/o Topic</td><td>92.51</td><td>76.95</td><td>57.29</td><td>85.67</td><td>86.04</td></tr><tr><td>w/o Core</td><td>93.10</td><td>86.52</td><td>58.33</td><td>89.41</td><td>88.96</td></tr><tr><td>w/o Source</td><td>83.35</td><td>82.98</td><td>52.08</td><td>78.50</td><td>80.32</td></tr><tr><td>w/o Recent</td><td>92.51</td><td>81.91</td><td>57.29</td><td>89.10</td><td>87.66</td></tr><tr><td>w/o Timeline</td><td>93.22</td><td>85.46</td><td>53.12</td><td>86.92</td><td>87.99</td></tr><tr><td>w/o Graph</td><td>92.87</td><td>84.04</td><td>61.46</td><td>86.92</td><td>88.05</td></tr></table>

Readers is highest for every task category. The usefulness of the constructed memory therefore extends beyond the model that produced it: other Readers can use the same documents and evidence references without rebuilding memory.

Reader choice also changes the task profile of the same memory. DeepSeek V4 Flash achieves higher multi-hop accuracy than GPT-5.6 Luna and Gemini 3 Flash across the evaluated memories, whereas Qwen 3.8 Flash consistently achieves the highest open-domain accuracy. Multi-hop questions require combining recorded facts, while opendomain questions require inferences from conversational context; these distinct requirements make the task breakdown more informative than a single overall Reader ranking. Qwen 3.8 Flash’s lower single-hop accuracy also explains why its opendomain advantage does not consistently yield the highest overall score: single-hop questions account for a much larger share of LoCoMo.

These results support choosing Writers and Readers separately according to task requirements. A constructed workspace can serve Readers with different task profiles, allowing the answering model to change while the stored memory is reused.

## 3.5 Scaling

We vary the Reader’s requested reasoning effort while reusing constructed memory and fixed question sets within each sweep. We measure actual usage as the mean cumulative input and output tokens across Reader calls per question, excluding memory construction and evaluation.

Increasing reasoning effort generally improves answer quality, with more sustained gains on BEAM’s longer histories (Figure 3). These tasks require locating clues, following up on original records, and integrating evidence distributed across the history, leaving more scope for additional reasoning to support evidence acquisition and synthesis. By allowing the Reader to continue searching and reading as needed, SOMEM provides a way to use additional inference resources over existing memory.

The gains nevertheless exhibit saturation and occasional reversals. On LoCoMo, some Readers continue to improve at higher effort levels, whereas others plateau or decline at the highest tested setting. Additional computation therefore does not necessarily translate into more effective evidence use: once the available information is sufficient, further reasoning may offer limited benefit. The appropriate budget depends on both the Reader and the task, rather than always favoring the highest effort level.

Higher reasoning effort need not increase total token usage either. On BEAM-100K, increasing GPT-6 Luna’s requested reasoning effort from High to xHigh improves answer quality while reducing searches, document reads, Reader calls, and cumulative token usage. This pattern is consistent with more targeted query and reading decisions reducing repeated lookups and the repeated input of accumulated context across calls. Thus, stronger reasoning may not only improve answer quality but also reduce token consumption through more efficient memory access, potentially lowering usage costs.

Table 6: LoCoMo accuracy (%) for Writer–Reader combinations on all 1,540 questions. Each Writer’s frozen memory is reused by all Readers. Bold marks the best Reader for each Writer and task, including ties.
<table><tr><td>Writer</td><td>Reader</td><td>SH</td><td>MH</td><td>OD</td><td>TR</td><td>Overall</td></tr><tr><td rowspan="4">GPT-5.6 Luna</td><td>GPT-5.6 Luna</td><td>95.36</td><td>91.49</td><td>72.92</td><td>90.97</td><td>92.34</td></tr><tr><td>Gemini 3 Flash</td><td>96.67</td><td>91.49</td><td>67.71</td><td>91.28</td><td>92.79</td></tr><tr><td>DeepSeek V4 Flash</td><td>96.79</td><td>92.91</td><td>69.79</td><td>90.97</td><td>93.18</td></tr><tr><td>Qwen 3.8 Flash</td><td>95.01</td><td>93.97</td><td>73.96</td><td>93.77</td><td>93.25</td></tr><tr><td rowspan="4">Gemini 3 Flash</td><td>GPT-5.6 Luna</td><td>96.08</td><td>91.13</td><td>64.58</td><td>90.03</td><td>91.95</td></tr><tr><td>Gemini 3 Flash</td><td>94.89</td><td>91.13</td><td>63.54</td><td>89.72</td><td>91.17</td></tr><tr><td>DeepSeek V4 Flash</td><td>96.79</td><td>94.68</td><td>60.42</td><td>91.90</td><td>93.12</td></tr><tr><td>Qwen 3.8 Flash</td><td>94.65</td><td>90.07</td><td>66.67</td><td>90.65</td><td>91.23</td></tr><tr><td rowspan="4">DeepSeek V4 Flash</td><td>GPT-5.6 Luna</td><td>95.24</td><td>91.49</td><td>69.79</td><td>90.65</td><td>92.01</td></tr><tr><td>Gemini 3 Flash</td><td>96.91</td><td>91.84</td><td>68.75</td><td>90.34</td><td>92.86</td></tr><tr><td>DeepSeek V4 Flash</td><td>95.96</td><td>93.26</td><td>65.62</td><td>91.90</td><td>92.73</td></tr><tr><td>Qwen 3.8 Flash</td><td>92.39</td><td>89.01</td><td>70.83</td><td>90.97</td><td>90.13</td></tr></table>

![](images/9936c34937bdc3f6227e11781ed798f0c797f248ad173da7716c4d1a02dcf2c8.jpg)  
Figure 3: Reader scaling on fixed LoCoMo (100 questions) and BEAM-100K (200) subsets. The x-axis is mean cumulative Reader input and output tokens per question, including cached and repeated input. LoCoMo reports LJ accuracy; BEAM reports mean nugget score (×100). Colors denote models and point labels denote requested effort.

## 4 Related Work

## 4.1 Memory Representation and Organization

Long-term memory systems differ in what they store from an interaction and how they connect it to earlier information. MemoryBank (Zhong et al., 2024) and Mem0 (Chhikara et al., 2025) retain compact records, and LangMem (LangChain Team, 2025) provides tools with which an agent adds, updates, deletes, and searches memories, complemented by background extraction and consolidation. Nemori (Ma et al., 2026) decides what to retain by how poorly an experience is predicted, and A-Mem (Xu et al., 2025) adds attributes and associative links to notes. MemMachine (Wang et al., 2026b) instead stores raw conversational episodes with limited LLM-based extraction and retrieves them together with their surrounding dialogue. LightMem (Fang et al., 2026) separates memoryprocessing stages to reduce online cost. TiMem (Li et al., 2026) consolidates information across temporal levels, and Mnemis (Tang et al., 2026) combines similarity retrieval with top-down selection over hierarchical graphs. Zep (Rasmussen et al., 2025) maintains a temporal knowledge graph. MemoryOS (Kang et al., 2025) moves memory through short-, mid-, and long-term tiers, and MemOS (Li et al., 2025) manages memory as an operatingsystem resource. Infini Memory (Ji et al., 2026) maintains topic documents through buffered writing and consolidation; ByteRover (Nguyen et al., 2026) uses an LLM-curated context tree with explicit relations and provenance. SOMEM focuses on the consistency of revision: the model edits semantic content, original interactions remain accessible, and the runtime reconstructs access views after accepting a semantic edit.

## 4.2 Retrieval and Agent-Controlled Access

Lexical and embedding retrieval select evidence according to its similarity to a query. Iterative approaches interleave retrieval with reasoning: IR-CoT (Trivedi et al., 2023) retrieves with each reasoning step, ReAct (Yao et al., 2023) alternates reasoning traces with tool actions, and Self-RAG (Asai et al., 2024) decides when to retrieve and critiques the retrieved passages. Progressive observation also appears beyond text: GUI-Lens (Fu et al., 2026) lets a general-purpose VLM choose the region and scale of each successive view for GUI grounding. For agent memory, E-mem (Wang et al., 2026a) reconstructs evidence within selected episodic contexts before answering, and MIRIX (Wang and Chen, 2025) assigns six memory types to dedicated manager agents coordinated by a meta manager. MemGPT (Packer et al., 2023) exposes memory operations to the LLM, LSFS (Shi et al., 2025) studies model-operated file access, and DeltaMem (Zhang et al., 2026) trains a single agent with reinforcement learning to manage persona memory end to end. These methods establish that memory use can include further operations after the first retrieval result. LLM-as-Code (Qi et al., 2026) argues that deterministic control flow belongs in program code, with the LLM invoked only for reasoning or generation; SOMEM applies a similar division to memory, leaving semantic organization to the model while code validates edits and derives access views. SOMEM combines continuing access with agentorganized documents and resolvable source references.

## 5 Conclusion

Through SOMEM, we demonstrate the feasibility of LLM agents organizing their own persistent memory. The agent uses ordinary Markdown documents to decide how information is summarized, grouped, and revised, while the runtime preserves source evidence and maintains consistent access structures. Experiments on three memory benchmarks demonstrate the effectiveness of this approach, and Lo-CoMo component analyses clarify the roles of original evidence and organized Topic documents. The resulting memory can be reused by different Readers and supports further improvements through additional reasoning effort, with lower token consumption in some settings. These findings show that LLM agents can both read external memory and shape how it is organized and evolves, providing a reusable knowledge base for subsequent tasks.

## 6 Limitations

Full-dataset comparisons on three benchmarks and component ablations on LoCoMo primarily evaluate memory-based question answering over conversation histories; they do not fully cover sustained personalization or task execution during long-term interaction. Reasoning-effort analyses use fixed subsets, and their trends warrant validation on larger question sets and across repeated runs. Evaluation relies on LLM judges without systematic human calibration and may be affected by judge biases.

Component ablations restrict access to frozen memories rather than testing how memory would be reconstructed or reorganized after removing a component; each condition is a single run, and the overall effects of Core and individual derived views on LoCoMo fall within sampling uncertainty. Reasoning-effort settings also differ in implementation across models and providers, so their labels do not represent equivalent compute budgets; crossmodel comparisons reflect performance under the respective configurations.

Memory construction, maintenance, and multiround reading incur additional overhead, whose trade-off with answer quality depends on the model, task, and access strategy. The runtime checks evidence references and structural consistency but cannot guarantee the semantic correctness or completeness of memory content. Detecting and correcting omissions or errors during continued updates therefore remains an important direction for further study. Practical deployment additionally requires access controls, retention policies, and isolated tool execution for the stored conversations.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In International Conference on Learning Representations.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Pengfei Du. 2026. Memory for autonomous LLM agents: Mechanisms, evaluation, and emerging frontiers. arXiv preprint arXiv:2603.07670.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2026. LightMem: Lightweight and efficient memory-augmented generation. In Proceedings ofthe 14th International Conference on Learning Representations.

Zichuan Fu, Shirong Wang, Wenlin Zhang, Guojing Li, Yimin Deng, Jingtong Gao, Junjia Qi, Hanyu Yan, Yefeng Zheng, Xiaopeng Li, Wanyu Wang, Xian Wu, and Xiangyu Zhao. 2026. GUI-Lens: Enabling precise GUI grounding for off-the-shelf VLMs via coarse-to-fine cropping. Preprint, arXiv:2608.03270.

Suozhao Ji, Baodong Wu, Zehao Wang, Lei Xia, Qingping Li, Ruisong Wang, Wenbo Ding, Zhenhua Zhu, Boxun Li, Guohao Dai, and Yu Wang. 2026. Infini memory: Maintainable topic documents for long-term LLM agent memory. arXiv preprint arXiv:2606.10677.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory OS of AI agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25961–25970.

LangChain Team. 2025. LangMem: Composable longterm memory for LangChain agents. https://github. com/langchain-ai/langmem. Open-source software.

Kai Li, Xuanqing Yu, Ziyi Ni, Yi Zeng, Yao Xu, Zheqing Zhang, Xin Li, Jitao Sang, Xiaogang Duan, Xuelei Wang, Chengbao Liu, and Jie Tan. 2026. TiMem: Temporal-hierarchical memory consolidation for long-horizon conversational agents. arXiv preprint arXiv:2601.02845.

Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin Niu, Hanyu Wang, Jiawei Yang, Chen Tang, et al. 2025. MemOS: A memory OS for AI system. arXiv preprint arXiv:2507.03724.

Wenquan Ma, Jiayan Nan, and WenLong Wu. 2026. What deserves memory: Adaptive memory distillation for LLM agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 34789– 34812. Association for Computational Linguistics.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL).

Andy Nguyen, Danh Doan, Hoang Pham, Bao Ha, Dat Pham, Linh Nguyen, Hieu Nguyen, Thien Nguyen, Cuong Do, Phat Nguyen, and Toan Nguyen. 2026. ByteRover: Agent-native memory through LLM-curated hierarchical context. arXiv preprint arXiv:2604.01599.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology (UIST).

Junjia Qi, Zichuan Fu, Jingtong Gao, Wenlin Zhang, Hanyu Yan, Xian Wu, and Xiangyu Zhao. 2026. LLM-as-Code: Agentic programming for agent harness. arXiv preprint arXiv:2606.15874.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Zeru Shi, Kai Mei, Mingyu Jin, Yongye Su, Chaoji Zuo, Wenyue Hua, Wujiang Xu, Yujie Ren, Zirui Liu, Mengnan Du, Dong Deng, and Yongfeng Zhang. 2025. From commands to prompts: LLM-based semantic file system for AIOS. In Proceedings ofthe 13th International Conference on Learning Representations (ICLR).

Zihao Tang, Xin Yu, Ziyu Xiao, Zengxuan Wen, Zelin Li, Jiaxi Zhou, Hualei Wang, Haohua Wang, Haizhen Huang, Weiwei Deng, Feng Sun, and Qi Zhang. 2026. Mnemis: Dual-route retrieval on hierarchical graphs for long-term LLM memory. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23914–23928.

Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, and J. Ross Mitchell. 2025. Beyond a million tokens: Benchmarking and enhancing long-term memory in LLMs. arXiv preprint arXiv:2510.27246.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving retrieval with chain-of-thought reasoning for knowledgeintensive multi-step questions. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10014–10037.

Kaixiang Wang, Yidan Lin, et al. 2026a. E-mem: Multiagent based episodic context reconstruction for LLM agent memory. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research.

Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, and Charles Fan. 2026b. Memmachine: A ground-truth-preserving memory system for personalized AI agents. arXiv preprint arXiv:2604.04853.

Yu Wang and Xi Chen. 2025. MIRIX: Multi-agent memory system for LLM-based agents. arXiv preprint arXiv:2507.07957.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In Proceedings of the 13th International Conference on Learning Representations (ICLR).

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, volume 38.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Qi Zhang, Shen Huang, Chu Liu, Shouqing Yang, Junbo Zhao, Haobo Wang, and Pengjun Xie. 2026. DeltaMem: Towards agentic memory management via reinforcement learning. arXiv preprint arXiv:2604.01560.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence.

## A Implementation Details

Validated memory updates. Free-form editing by an LLM can silently break the structure that later queries rely on, for example by dropping an evidence reference or pointing a link to a missing paragraph. SOMEM therefore stages every edit and installs it only after the runtime checks Source immutability, Topic paragraph structure, the resolution of source references and links, temporal values, and allowed paths; Core additionally passes reference and capacity checks. Accepted edits are normalized, relative links affected by moves are repaired, and Graph, Timeline, Recent, and the file inventory are rebuilt, while a rejected edit leaves the previous state unchanged. These checks preserve document structure and resolvable evidence links during reorganization; factual interpretation remains the responsibility of the model.

Incremental organization. New interactions are archived as Source before the Writer integrates their information into existing Topic and Core content. Periodic organization revisits Topic files changed since the preceding pass, allowing the agent to revise grouping and document structure as related information accumulates; an optional final pass covers the complete Topic tree. Both integration and reorganization use the validation procedure above, so the runtime reconstructs access views from the accepted content. An optional post-write probe asks a factual question about a new interaction and triggers targeted repair when the memory cannot answer it.

Query interface. At query time, the agent receives Core, Recent, the file inventory, and the question date when available, and can list and read files, run exact-text, BM25, or embedding search, and inspect Source records directly. No fixed order of operations is imposed, so the agent can enter through a Topic document, a Timeline date, or a search hit, and then follow links and source references as needed. The split search interface exposes bm25\_search and embedding\_search, together with list\_memory\_files and read\_memory\_file; the latter accepts line offsets and limits. A combined search interface exposes lexical and dense modes through memory\_search. All main experiments use the split interface. The read-only shell interface supports file listing (ls, find), literal search (rg, grep), and reading (cat, sed). These interfaces implement the search, listing, and reading operations described in Section 2.3. Figure 4 illustrates one update and the corresponding evidence inspection.

## B Evaluation Details

LoCoMo. In Table 2, all memory methods in the primary block use GPT-6 Luna for construction and answering, and all configurations are evaluated on the 1,540 category 1–4 questions with the same GPT-4o-mini judge; SOMEM uses agent-controlled search with at most 20 QA turns. The primary SOMEM result and its component ablations use GPT-6 Luna as Reader on the same frozen memory. Additional backbone configurations use the named model for both Writer and Reader with their own construction and query settings.

BEAM-100K. The seven memory methods in Table 3 use GPT-6 Luna for construction and answering on the same 400 questions over 20 conversations. A-Mem retrieves the top 10 linked notes, and Mem0 and LightMem retrieve the top 20 memories with all-MiniLM-L6-v2 embeddings; MemoryOS, Nemori, and E-mem use their released retrieval pipelines; SOMEM uses agent-controlled search with at most 50 QA turns.

LongMemEval-S. All methods in Table 4 are evaluated on the complete 500-question set. The primary block uses GPT-6 Luna for construction and answering, while the additional SOMEM row uses DeepSeek V4 Flash for both stages; SOMEM uses agent-controlled search with at most 50 QA turns.

Evaluation protocols. All primary evaluations score complete saved answers with GPT-4o-mini at temperature 0. LoCoMo uses the judging prompt in Figure 5 (Fang et al., 2026), with COR-RECT/WRONG JSON labels over category 1–4 questions. Overall accuracy is micro-averaged over questions. LongMemEval-S uses question-typespecific judging prompts: multi-session and singlesession-assistant questions use a generic exactmatch template, while the remaining types and abstention cases use task-specific templates. The judge uses top-p = 0.95 and returns a yes/no correctness decision. BEAM uses rubric-nugget scoring for all capabilities, including event ordering. Each rubric criterion is judged against the question and complete answer. Scores are mapped to 0, 0.5, or 1 using thresholds of 0.25 and 0.75, averaged within each question, and then averaged equally over questions. Pass rates use a questionlevel threshold of 0.5. The judge response limit is 300 tokens. Failed requests and invalid LoCoMo labels are unresolved evaluations, not incorrect answers.

![](images/17cfdb277210fdc1a0f17e0bf9526a5b9d913ff43c28eb247009cc8e1ae0e4c8.jpg)  
Figure 4: An example of a memory update and evidence inspection. A venue change is recorded as a revision of the existing Topic paragraph that cites both the original plan and the relocation notice in Source.

Table 7: Controlled settings for the LoCoMo component ablations. The frozen memory, questions, retrieval prompt, search settings, and scoring protocol are shared across conditions.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Questions</td><td>All 1,540 LoCoMo category 1–4 ques- tions</td></tr><tr><td>Memory</td><td>Frozen memory shared across condi- tions</td></tr><tr><td>Reader</td><td>GPT-6 Luna, no reasoning effort</td></tr><tr><td>Query turn limit</td><td>20</td></tr><tr><td>Search interface</td><td>Split BM25 and embedding search</td></tr><tr><td>Variable Judge</td><td>Query-time component access GPT-4o-mini, temperature 0</td></tr></table>

## C Supplementary Results

## C.1 LoCoMo Component Ablation Details

Table 5 reports all four question categories for the component ablations on the complete 1,540- question LoCoMo evaluation set. The Full configuration uses the same answers and scores as the primary SOMEM row in Table 2. The w/o Graph condition removes explicit association links and the Relations index while retaining factual content and Source references.

Controlled settings. Table 7 records the frozen memory source and query settings. All conditions reuse the same memory without reconstruction and change only query-time component access.

Context and access. Table 8 reports the input to the final answer-producing request and the total number of Reader calls per question. Removing Recent reduces final input length while leaving average call counts close to Full, supporting the compact-input interpretation in the main text.

Table 8: Mean final-answer input and Reader calls for the LoCoMo component ablations on all 1,540 questions.
<table><tr><td>Condition</td><td>Input (K)</td><td>Calls</td></tr><tr><td>Full</td><td>18.57</td><td>4.85</td></tr><tr><td>w/o Topic</td><td>4.67</td><td>4.31</td></tr><tr><td>w/o Core</td><td>17.33</td><td>4.81</td></tr><tr><td>w/o Source</td><td>17.01</td><td>3.91</td></tr><tr><td>w/o Recent</td><td>8.28</td><td>4.77</td></tr><tr><td>w/o Timeline</td><td>18.00</td><td>4.92</td></tr><tr><td>w/o Graph</td><td>18.56</td><td>4.80</td></tr></table>

Input is the token count of the final answer-producing request, including instructions, tool definitions, and accumulated dialogue; K denotes 1,000 tokens. Calls is the number of Reader requests over the entire question. Input measures final-request length, not cumulative usage.

Paired question-level changes. Table 9 compares each condition with its matched Full control using the original saved answers and scores on identical questions. The paired counts distinguish net changes from opposing question-level changes: equal overall accuracy can conceal different sets of correct answers. We compute paired percentile bootstrap intervals by resampling conversations to preserve dependence among questions from the same history. These intervals quantify sampling uncertainty conditional on the saved runs.

## C.2 Memory Access Comparison

Settings and paired outcomes. This comparison uses 100 questions stratified over ten LoCoMo conversations and four question types, with a shared GPT-5.6 Luna Reader. Full Context supplies the original conversation; BM25 and Dense retrieve

![](images/c7290a909f53c13566de43a0e0e0fcd435dff331de79b18f4bd7d60d673ca745.jpg)  
Figure 5: Prompt used for LLM-as-a-Judge evaluation on LoCoMo. The placeholders are filled with the question, the gold answer, and the generated answer of each evaluated item.

Table 9: Paired accuracy changes relative to Full on all 1,540 LoCoMo questions. Positive differences favor the ablated condition.
<table><tr><td>Condition</td><td>∆(pp)</td><td>W→R</td><td>R→W</td><td>95% CI (pp)</td></tr><tr><td>w/o Topic</td><td>-2.92</td><td>65</td><td>110</td><td>[−4.96, −0.88]</td></tr><tr><td>w/o Core</td><td>+0.00</td><td>52</td><td>52</td><td>[−1.16, +1.09]</td></tr><tr><td>w/o Source</td><td>-8.64</td><td>28</td><td>161</td><td>[-11.67, -6.11]</td></tr><tr><td>w/o Recent</td><td>-1.30</td><td>44</td><td>64</td><td>[−2.98, +0.30]</td></tr><tr><td>w/o Timeline</td><td>-0.97</td><td>48</td><td>63</td><td>[−2.54, +0.52]</td></tr><tr><td>w/o Graph</td><td>-0.91</td><td>50</td><td>64</td><td>[−2.65, +0.85]</td></tr></table>

W→R and R→W count wrong-to-right and right-to-wrong changes. Intervals use 10,000 paired percentile bootstrap samples (seed 20261031), resampling the ten conversations. They describe sampling uncertainty for the saved runs, not repeatedrun variability; no multiple-comparison correction is applied.

Source evidence once; E-mem reconstructs routed episodic evidence; and LightMem retrieves extracted facts. The masking conditions restrict access to the same frozen SOMEM memory. Table 10 compares answer accuracy, and Table 11 reports paired outcomes on identical questions. The paired results show that organized, iterative access recovers answers missed by one-shot lexical and dense retrieval. Removing Source also loses answers that organized summaries alone do not recover, supporting the use of original records alongside Topic documents.

Table 10: Comparison of memory access methods on LoCoMo with a shared GPT-5.6 Luna Reader. We report LJ accuracy (%) on the analysis subset. <sup>†</sup> denotes our local adaptation, and the best results are highlighted in bold.
<table><tr><td>Method</td><td>SH</td><td>MH</td><td>OD</td><td>TR</td><td>Overall</td></tr><tr><td>Full Context</td><td>90.91</td><td>83.33</td><td>83.33</td><td>90.48</td><td>89.00</td></tr><tr><td>BM25</td><td>76.36</td><td>50.00</td><td>16.67</td><td>57.14</td><td>64.00</td></tr><tr><td>Dense</td><td>54.55</td><td>33.33</td><td>50.00</td><td>61.90</td><td>52.00</td></tr><tr><td>E-mem†</td><td>85.45</td><td>72.22</td><td>50.00</td><td>85.71</td><td>81.00</td></tr><tr><td>LightMem†</td><td>80.00</td><td>61.11</td><td>50.00</td><td>80.95</td><td>75.00</td></tr><tr><td>SOMEM</td><td>98.18</td><td>77.78</td><td>83.33</td><td>100.00</td><td>94.00</td></tr></table>

## C.3 Additional BEAM Results

Pass rates by capability. Table 12 complements the nugget scores in Table 3 with per-capability pass rates for all methods; SOMEM has the highest overall pass rate and the highest or tied-highest pass rate on temporal reasoning, instruction following, preference following, and summarization.

Table 13: Results of SOMEM on BEAM-1M with GPT-6.1 Sol as the Reader under two judge models. We report pass rate and mean nugget score (%).
<table><tr><td>Judge</td><td>Pass</td><td>Nugget</td></tr><tr><td>GPT-4o-mini</td><td>69.29</td><td>63.24</td></tr><tr><td>GPT-6 Sol</td><td>84.14</td><td>73.86</td></tr></table>

Table 11: Question-level paired comparison on the LoCoMo analysis subset between SOMEM and each access method (top) or masking condition (bottom). We report the number of questions in each outcome and the comparatorminus-SOMEM accuracy difference in percentage points with its 95% bootstrap confidence interval.
<table><tr><td>Comparator</td><td>Both correct</td><td>Only comparator</td><td>Only SoMEM</td><td>Both wrong</td><td>Difference</td><td>95% CI</td></tr><tr><td>Full Context</td><td>88</td><td>1</td><td>6</td><td>5</td><td>-5</td><td>[−10,0]</td></tr><tr><td>BM25</td><td>64</td><td>0</td><td>30</td><td>6</td><td>-30</td><td>[−40, -20]</td></tr><tr><td>Dense</td><td>52</td><td>0</td><td>42</td><td>6</td><td>-42</td><td>[−54, -32]</td></tr><tr><td>E-mem (adapted)</td><td>81</td><td>0</td><td>13</td><td>6</td><td>-13</td><td>[−19, −8]</td></tr><tr><td>LightMem (adapted)</td><td>74</td><td>1</td><td>20</td><td>5</td><td>-19</td><td>[−31, −9]</td></tr><tr><td>w/o Core</td><td>92</td><td>1</td><td>2</td><td>5</td><td>-1</td><td>[−4, 2]</td></tr><tr><td>w/o Source</td><td>80</td><td>2</td><td>14</td><td>4</td><td>-12</td><td>[−18,−5]</td></tr></table>

Table 12: Pass rate (%) on BEAM-100K across memory capabilities.
<table><tr><td>Method</td><td>IE</td><td>MR</td><td>TR</td><td>KU</td><td>CR</td><td>EO</td><td>IF</td><td>PF</td><td>SUM</td><td>ABS</td><td>Overall</td></tr><tr><td>Full Context</td><td>87.50</td><td>52.50</td><td>62.50</td><td>50.00</td><td>20.00</td><td>37.50</td><td>65.00</td><td>92.50</td><td>52.50</td><td>32.50</td><td>55.25</td></tr><tr><td>A-Mem</td><td>85.00</td><td>62.50</td><td>65.00</td><td>52.50</td><td>12.50</td><td>42.50</td><td>72.50</td><td>95.00</td><td>40.00</td><td>30.00</td><td>55.75</td></tr><tr><td>Mem0</td><td>72.50</td><td>55.00</td><td>47.50</td><td>62.50</td><td>22.50</td><td>50.00</td><td>75.00</td><td>95.00</td><td>27.50</td><td>25.00</td><td>53.25</td></tr><tr><td>MemoryOS</td><td>65.00</td><td>65.00</td><td>50.00</td><td>42.50</td><td>17.50</td><td>30.00</td><td>72.50</td><td>90.00</td><td>32.50</td><td>42.50</td><td>50.75</td></tr><tr><td>Nemori</td><td>67.50</td><td>57.50</td><td>37.50</td><td>47.50</td><td>15.00</td><td>45.00</td><td>80.00</td><td>92.50</td><td>45.00</td><td>27.50</td><td>51.50</td></tr><tr><td>LightMem</td><td>65.00</td><td>57.50</td><td>47.50</td><td>45.00</td><td>27.50</td><td>25.00</td><td>72.50</td><td>85.00</td><td>15.00</td><td>30.00</td><td>47.00</td></tr><tr><td>E-mem</td><td>60.00</td><td>50.00</td><td>42.50</td><td>32.50</td><td>67.50</td><td>52.50</td><td>62.50</td><td>90.00</td><td>15.00</td><td>17.50</td><td>49.00</td></tr><tr><td>SOMEM</td><td>80.00</td><td>57.50</td><td>77.50</td><td>52.50</td><td>40.00</td><td>47.50</td><td>80.00</td><td>95.00</td><td>60.00</td><td>25.00</td><td>61.50</td></tr></table>

Longer histories on BEAM-1M. On BEAM-1M, whose histories are roughly ten times longer than BEAM-100K, SOMEM with GPT-6.1 Sol reaches a 69.29% pass rate under the GPT-4o-mini judge and 84.14% under the GPT-6 Sol judge (Table 13). This experiment supplements the main benchmark with longer histories. Its Reader configuration differs from the primary BEAM-100K comparison, so it documents performance at this scale rather than isolating the effect of history length.

## D Scaling and Resource Usage

## D.1 Reader Scaling Details

Tables 14 and 15 give the numerical values plotted in Figure 3, on the same 100 LoCoMo and 200 BEAM-100K questions. Input and output are summed over the Reader calls contributing to each adopted answer and then averaged over questions. Input includes cached, cache-creation, and repeated context; reported reasoning tokens are included in output once. Construction, judging, abandoned attempts, and separate auxiliary session requests are excluded. These are token-usage measurements, not monetary charges or the length of the final prompt.

Within each sweep, memory and questions are reused. Effort labels denote the requested reasoning settings.

## D.2 Construction and Query Resources

The following resource measurements supplement the Reader-only scaling analysis. Construction is counted once, whereas query resources accumulate over answered questions.

Cost is concentrated in reusable input context. On the full LoCoMo benchmark, SOMEM reaches 88.96% LJ accuracy at a total reference cost of \$229.73, compared with \$6.75 for Mem0 (80.32%), \$6.46 for LightMem (77.99%), and \$35.19 for A-Mem (60.06%) (Figure 6). Of this cost, \$118.85 is one-time construction, and querying costs about \$0.07 per question. Input context dominates the usage, and 66.3% of input tokens are cache hits across the recorded calls; charging cached input at 10% of the input price lowers the total to \$96.33. The cross-model reuse experiment in Section 3.4 shows how a construction cost can be shared across subsequent queries and Readers. In separate Reader runs, querying averages 5.2–7.4 model calls and about 20 seconds per question (Table 16).

Table 14: LoCoMo scaling details corresponding to Figure 3. Score is LJ accuracy (×100). Input, output, and total tokens are means per question in thousands.
<table><tr><td>Reader</td><td>Effort</td><td>Score</td><td>Input (K/Q)</td><td>Output (K/Q)</td><td>Total (K/Q)</td></tr><tr><td>GPT-6 Luna</td><td>None</td><td>93.00</td><td>76.609</td><td>0.210</td><td>76.819</td></tr><tr><td>GPT-6 Luna</td><td>Low</td><td>91.00</td><td>64.998</td><td>0.184</td><td>65.182</td></tr><tr><td>GPT-6 Luna</td><td>Medium</td><td>91.00</td><td>68.924</td><td>0.195</td><td>69.118</td></tr><tr><td>GPT-6 Luna</td><td>High</td><td>93.00</td><td>79.253</td><td>0.247</td><td>79.500</td></tr><tr><td>GPT-6 Luna</td><td>xHigh</td><td>91.00</td><td>95.935</td><td>0.470</td><td>96.405</td></tr><tr><td>GPT-6.1 Sol</td><td>Low</td><td>93.00</td><td>55.897</td><td>0.155</td><td>56.052</td></tr><tr><td>GPT-6.1 Sol</td><td>Medium</td><td>92.00</td><td>62.888</td><td>0.170</td><td>63.058</td></tr><tr><td>GPT-6.1 Sol</td><td>High</td><td>93.00</td><td>75.701</td><td>0.217</td><td>75.918</td></tr><tr><td>GPT-6.1 Sol</td><td>xHigh</td><td>95.00</td><td>98.204</td><td>0.424</td><td>98.628</td></tr><tr><td>Qwen 3.8 Flash</td><td>Low</td><td>92.00</td><td>54.760</td><td>0.384</td><td>55.144</td></tr><tr><td>Qwen 3.8 Flash</td><td>Medium</td><td>94.00</td><td>71.242</td><td>0.539</td><td>71.781</td></tr><tr><td>Qwen 3.8 Flash</td><td>xHigh</td><td>94.00</td><td>84.926</td><td>0.699</td><td>85.626</td></tr><tr><td>GLM-5.3 Flash</td><td>Low</td><td>93.00</td><td>48.503</td><td>0.236</td><td>48.739</td></tr><tr><td>GLM-5.3 Flash</td><td>High</td><td>94.00</td><td>50.010</td><td>0.303</td><td>50.313</td></tr><tr><td>GLM-5.3 Flash</td><td>Max</td><td>94.00</td><td>60.480</td><td>0.623</td><td>61.103</td></tr></table>

Table 15: BEAM-100K scaling details corresponding to Figure 3. Score is mean nugget score (×100). Input, output, and total tokens are means per question in thousands.
<table><tr><td>Reader</td><td>Effort</td><td>Score</td><td>Input (K/Q)</td><td>Output (K/Q)</td><td>Total (K/Q)</td></tr><tr><td>GPT-6 Luna</td><td>None</td><td>57.85</td><td>115.989</td><td>0.557</td><td>116.547</td></tr><tr><td>GPT-6 Luna</td><td>Low</td><td>56.82</td><td>47.621</td><td>0.468</td><td>48.089</td></tr><tr><td>GPT-6 Luna</td><td>Medium</td><td>57.60</td><td>75.804</td><td>1.015</td><td>76.820</td></tr><tr><td>GPT-6 Luna</td><td>High</td><td>59.07</td><td>110.884</td><td>1.675</td><td>112.558</td></tr><tr><td>GPT-6 Luna</td><td>xHigh</td><td>61.79</td><td>102.774</td><td>1.637</td><td>104.411</td></tr><tr><td>GPT-6.1 Sol</td><td>Low</td><td>63.98</td><td>38.223</td><td>0.490</td><td>38.713</td></tr><tr><td>GPT-6.1 Sol</td><td>Medium</td><td>66.35</td><td>44.427</td><td>0.572</td><td>45.000</td></tr><tr><td>GPT-6.1 Sol</td><td>High</td><td>67.25</td><td>63.551</td><td>0.818</td><td>64.369</td></tr><tr><td>GPT-6.1 Sol</td><td>xHigh</td><td>70.42</td><td>101.481</td><td>1.849</td><td>103.330</td></tr><tr><td>Qwen 3.8 Flash</td><td>None</td><td>70.83</td><td>202.039</td><td>2.307</td><td>204.346</td></tr><tr><td>Qwen 3.8 Flash</td><td>Low</td><td>66.68</td><td>194.662</td><td>2.845</td><td>197.507</td></tr><tr><td>Qwen 3.8 Flash</td><td>Medium</td><td>69.60</td><td>157.804</td><td>2.429</td><td>160.233</td></tr><tr><td>Qwen 3.8 Flash</td><td>xHigh</td><td>70.35</td><td>207.606</td><td>2.684</td><td>210.290</td></tr><tr><td>GLM-5.3 Flash</td><td>Low</td><td>63.93</td><td>20.188</td><td>0.527</td><td>20.715</td></tr><tr><td>GLM-5.3 Flash</td><td>High</td><td>68.82</td><td>25.975</td><td>0.791</td><td>26.766</td></tr><tr><td>GLM-5.3 Flash</td><td>Max</td><td>70.43</td><td>46.652</td><td>1.999</td><td>48.651</td></tr></table>

(a) Efectiveness and cost  
![](images/01373c5aa7e7b40771dac85b24ddfbdaffa26718da6c48d321b618ee2d22cf9a.jpg)

(b) Input / output cost  
![](images/afca93f9338f36dac4c342c5225e66f02a6b4d3a8d7b870293e22377e68ce9c6.jpg)  
Figure 6: Accuracy and cost of memory methods on the full LoCoMo benchmark with GPT-6 Luna. (a) LJ accuracy versus total cost on a logarithmic axis. (b) Cost contributions of input and output tokens, priced at \$1/\$6 per million.

Table 16: Query cost of two Readers on the full Lo-CoMo benchmark: LJ accuracy, model calls per question (Calls/Q), and mean end-to-end latency per question.
<table><tr><td>Reader</td><td>Acc. (%)</td><td>Calls/Q</td><td>Latency (s)</td></tr><tr><td>GPT-5.6 Luna</td><td>92.34</td><td>5.21</td><td>21.59</td></tr><tr><td>DeepSeek V4 Flash</td><td>93.18</td><td>7.40</td><td>18.79</td></tr></table>