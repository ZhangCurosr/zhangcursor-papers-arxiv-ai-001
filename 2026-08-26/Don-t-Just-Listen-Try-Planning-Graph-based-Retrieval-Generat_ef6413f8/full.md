# Don’t Just Listen, Try Planning: Graph-based Retrieval-Generation Agent for Long-form Audio Meeting Understanding

Quanwei Tang<sup>1</sup>, Dong Zhang<sup>1,2\*\*</sup>, Shoushan Li<sup>1</sup>, and Guodong Zhou<sup>1</sup>

<sup>1</sup>School of Computer Science & Technology, NLP Lab, Soochow University, China <sup>2</sup>Jiangsu Key Lab of Language Computing, Suzhou.

dzhang@suda.edu.cn

## Abstract

While long-form audio meeting understanding (LAMU) is garnering growing attention, taskspecific question answering (QA) datasets remain scarce. Existing speech QA paradigms and state-of-the-art Speech LLMs suffer from acoustic information loss and poor long-term context memory. To address these issues, we construct the LongAudioQA dataset and propose the GRGA model, which models heteroge neous audio features into a multi-dimensional graph and leverages agent planning for retrieval and answer generation. GitHub for data and code.

## 1 Introduction

Long-form audio meeting understanding (LAMU) has attracted significant attention in speech processing. However, previous studies have only focused on transcription recognition for long-form multiparty meetings (Yu et al., 2022a,b; Jain et al., 2024). Consequently, there is a lack of dedicated question answering (QA) datasets for long audio meetings, despite the critical importance of this task. To address this gap, we construct the LongAudioQA dataset for LAMU. Distinct from short-form conversation QA, it is designed to capture three core dimensions of long-form audio meeting content: complex semantics, multi-speaker interactions, and quite long timestamps.

Although existing Speech LLMs (KimiTeam et al., 2025; Xu et al., 2025a) have demonstrated strong performance across various tasks, they prioritize textual context by ASR over speech context (You et al., 2022; Lin et al., 2024). Specifically, these models typically map the input speech to the textual context, which inevitably leads to the loss of valuable acoustic information. For example, regarding the query “sudden loud voice at the 19-minute mark” in Figure 1, the inability to access voice volume prevents natural understanding and appropriate response (as the result of Text LLM in the Figure). We denote this phenomenon as Acoustic Missing. Additionally, prior QA on speech conversations has been exclusively centered on shortform audio clips or dialogues (< 30s) (Johnson et al., 2024; Zhao et al., 2025; Hu et al., 2026). When applied to extremely long audio meetings, these methods fail to capture long-term dependencies. Let’s return to the query in Figure 1, we not only need to locate the content of the 19-minute timestamp, but also to trace the answer back to the description of 1:30 (as our GRGA). Without an explicit design for long-range semantic understanding, even RAG approaches (Tang et al., 2025; Li et al., 2025; Wang et al., 2024) fail to identify the correct rationale (as Text RAG). We designate this phenomenon as Context Fragmentation.

![](images/fd4588b10800b3c4f9fbd853ae813e474f3d4790f94ec936723ef5f2ca6efdb9.jpg)  
Figure 1: Existing Speech LLMs and our GRGA for long-form audio QA. By indexing conversations across Semantic, Speaker, and Timestamp dimensions, our model enables precise reasoning.

To address these challenges, we propose the Graph-based Retrieval-Generation Agent (GRGA) model. Specifically, we first model heterogeneous features from the audio, including not only acoustic information (e.g., voice and tone) but also many speaker attributes (e.g., role: project manager, gender: male), into a unified multi-dimensional graph structure. Then, we leverage agent planning to retrieve question-relevant clues from this multidimensional graph and generate the final answer. In summary, we contribute:

• We design and construct the novel dataset LongAudioQA for LAMU.

• We propose GRGA to handle both acoustic missing and context fragmentation.

• We conduct both automatic and human evaluation on three datasets of our LongAudioQA.

## 2 Dataset Construction

Existing speech QA benchmarks (Zhifei et al., 2025) predominantly focus on short-context span extraction or simple intent classification. They often treat the dialogue as a flat sequence of text, neglecting the intricate graph-like structure of meeting interactions (e.g., speaker turns, crossreferences, and temporal dynamics). Consequently, current models are rarely tested on their ability to perform multi-hop reasoning or temporal grounding over long-form recordings. For instance, answering a question like “How did the speaker’s attitude change after the 30-minute discussion?” requires a model to not only localize information but also aggregate evidence across distinct time slices and model the causal dependencies between nodes. Therefore, we propose our LongAudioQA.

## 2.1 Data Selection and Collection

We mainly construct the speech question-answer pairs from the following raw dataset:

AliMeeting. AliMeeting (Yu et al., 2022a,b) is a large-scale Mandarin speech dataset tailored for multi-speaker meeting ASR and SD. A distinguishing feature of this corpus is the simultaneous recording of overlapping far-field audio and individual near-field references. This setup is designed to address the “who said what when” problem, testing the model’s ability to handle speaker overlap and diarization in complex settings.

AMI Meeting. The AMI Conference Corpus (Jain et al., 2024) consists of meeting audio recordings captured using far-field microphones, primarily capturing interactions among non-native English speakers. It is characterized by acoustic complexity, presenting significant challenges commonly encountered in real conference settings, such as background noise and reverberation issues.

DailyTalk. In contrast to the aforementioned conference-centric corpora, DailyTalk (Lee et al., 2022) focuses on high-quality open-domain dyadic conversations. This corpus comprises 2,541 dialogues. DailyTalk provides clean acoustic environment with an emphasis on conversational fluency and prosodic features. We concatenated 20 clips to create inputs of intermediate length (<10 minutes). While shorter than full meetings, this duration serves to ensure our method performs well on longer audio segments while maintaining effectiveness on shorter ones.

## 2.2 Question Definition and Data Annotation

To bridge the gap, we constructed a novel dataset designed to expand the boundaries of long-form meeting understanding. Unlike existing research that relies solely on factual retrieval, our dataset introduces a diversified question taxonomy, encompassing factual, inferential, temporal, and acousticaware as shown in Table 1). This design enables us to systematically evaluate models’ ability to transition from explicit pattern matching to advanced semantic reasoning within complex interaction graphs. Following methodologies for automated data annotation using LLMs (Lian et al., 2025a,b), we employ a hybrid strategy combining large language model-driven automated annotation with human-assisted verification. Specifically, this approach leverages large language models to generate questions from specific local segments of raw conference data based on predefined query types and examples. The models then search conference content for answers and extract detailed evidence. This process is iterative: each batch of generated question-answer pairs undergoes human verification to ensure quality before advancing to the next iteration. While the generation process leverages an “oracle” mechanism to access localized evidence, the core challenge of this study lies in retrieving answers from global, long-context conference data without prior knowledge of relevant segments. This evaluates the model’s ability to overcome context fragmentation.

## 2.3 Dataset Quality Control

Given that automated generation inevitably introduces noise, we enforce a rigorous verification protocol to ensure data reliability. Expert annotators inspect candidate samples to discard hallucinated questions (where answers are absent), rewrite ambiguous references, and validate logic depth (hop count) to ensure complexity balance. Specifically, we require that the generated timestamp evidence yields an Intersection-over-Union (IoU) of > 0.9 with the ground truth. This process results in a high-quality dataset with a human inter-annotator agreement rate of $\kappa = 0 . 9 1$ (Cohen’s Kappa). Finally, the duration and sample statistics of our LongAudioQA, along with the question distribution, are summarized in Table 2 and Figure 2.

<table><tr><td>Category</td><td>Core Competency</td><td>Subtasks &amp; Example Queries</td></tr><tr><td>Factual</td><td>Explicit Retrieval</td><td>• Entity Retrieval: &quot;Who mentioned the project code Alpha&#x27;?&quot; • Attribute Association: &quot;What is the budget proposed by the Manager?&quot; • Keyword Locating: Matching specific text spans.</td></tr><tr><td>Inferential</td><td>Multi-hop Reasoning</td><td>• Causal Inference: Linking a problem raised early with the final decision. • Coreference Resolution: Identifying what “that plan&quot; refers to. • Stance Analysis: Tracking how a speaker&#x27;s attitude evolves.</td></tr><tr><td>Temporal</td><td>Time Awareness</td><td>• Absolute Localization: &quot;What topic was discussed at the 30-min mark?&quot; • Relative Sequencing: &quot;What was discussed after ‘market research&#x27;?&quot; • Frequency Stats: Counting term occurrences in a window.</td></tr><tr><td>Summarization</td><td>Aggregation</td><td>• Topic Summarization: Synthesizing consensus on “backend architecture&quot;. • Speaker Profiling: Summarizing a participant&#x27;s main contributions.</td></tr><tr><td>Acoustic-Aware</td><td>Multi-modal Alignment</td><td>• Emotion &amp; Intensity: “Who seemed most agitated when discussing the budget?&quot; • Cross-modal Localization &amp; Causal: “&quot;What is the reason for the person&#x27;s</td></tr></table>

Table 1: Taxonomy of Meeting Questions in our LongAudioQA. Categories are distinguished by background colors: Factual , Inferential , Temporal , Summarization , and Acoustic-Aware . The Acoustic-Aware category uniquely requires grounding textual semantics with paralinguistic acoustic signals.

<table><tr><td>Dataset</td><td>Dur (h)</td><td>Avg. Dur (s)</td><td>Avg. Turns</td><td>#Q&amp;A</td><td># Dial.</td></tr><tr><td>AliMeeting</td><td>14.91</td><td>1,935</td><td>815</td><td>3,013</td><td>28</td></tr><tr><td>AMI</td><td>18.24</td><td>1,930</td><td>757</td><td>3,243</td><td>34</td></tr><tr><td>DailyTalk</td><td>21.59</td><td>610</td><td>186</td><td>10,200</td><td>128</td></tr></table>

Table 2: Statistical information among DailyTalk, AMI, and AliMeeting datasets. Dur: Duration, Avg: Average.

![](images/3e985dee70bbd96a99b53e5fb4456f9d9f561f34ad667bae641a5759e7c3c00d.jpg)  
Figure 2: Question Distribution Across Corpora

## 3 Methodology

The core challenge in long-form audio QA lies in handling composite queries that involve semantic, temporal, and speaker relationships. Existing methods often rely on “one-shot” retrieval, which fails to capture the intricate dependencies in meeting data. To address this, we propose a novel Graph-based Retrieval-Generation Agent (GRGA) that mimics the cognitive process of a human expert: it does not merely search, but rather plans and reflects.

Cognitive Inspiration. To design an effective agent, we draw inspiration from the cognitive strategies human experts employ to navigate complex meetings. We observe that resolving composite queries typically adheres to a “Search-Reason-Verify” cognitive loop, which adapts dynamically to the query type. For instance, in factual tasks (e.g., verifying budget details), humans execute a targeted keyword scan followed by contextual verification—a mechanism we emulate through retrieval tools. Conversely, inferential tasks (e.g., discerning the rationale behind a rejection) necessitates maintaining working memory while traversing causal chains. Similarly, humans naturally construct a mental timeline to address temporal queries, whereas summarization involves synthesizing fragmented details into high-level concepts. Consequently, we architect our agent to replicate these cognitive behaviors by incorporating a Planner for logical orchestration, a Synthesizer module for holistic information aggregation, and a Self-Reflector for answer verification.

![](images/e6696211b96711c4a60dbaee8bcc6d70bb6797a08751a6553efde7555efce142.jpg)  
Figure 3: The overall architecture of our graph-based retrieval-generation agent GRGA. The framework models longform audio as a Multi-dimensional Graph (bottom-left) to capture semantic, temporal, and speaker dependencies. Our GRGA Planning Process consists of: Query Decomposition, Planning, Execution, Synthesis, and Reflection.

## 3.1 Problem Formulation

We formulate the task as a Partially Observable Markov Decision Process (POMDP) (Lauri et al., 2023a), defined by the tuple M = $\langle S , \mathcal { A } , \mathcal { T } , \mathcal { R } , \Omega , \mathcal { O } , \gamma \rangle$ . The pre-constructed multidimensional graph G serves as the environment with which the agent interacts.

Belief State $\left( \boldsymbol { b } _ { t } \right)$ & Policy (π): The belief state $b _ { t }$ , represented by the query Q and interaction history $H _ { t }$ , serves as the sufficient statistic for decision-making. We leverage a pre-trained LLM as the policy $\pi ( a _ { t } \mid b _ { t } )$ . Implemented via the Query Decomposer and Execution Planner, the policy utilizes in-context reasoning to determine the next optimal action without parameter updates.

Action (A) & Observation (Ω): The action space A comprises topological retrieval operations. The Tool Executor executes $a _ { t }$ on G, yielding a partial observation $o _ { t } \in \Omega$ (e.g., specific acoustic cues or semantic nodes), which updates $b _ { t }$

Reward (R): The Self-Reflector acts as a verifier, providing a heuristic reward $r _ { t }$ . It evaluates whether $o _ { t }$ aligns with the reasoning chain, prompting the planner to prune incorrect paths or terminate generation.

Since the model is training-free, our objective is to approximate the optimal trajectory $\tau ^ { * }$ = $\{ b _ { 0 } , a _ { 0 } , \ldots , a _ { T } \}$ that maximizes the reliability of the final answer. The process terminates when the

Answer Synthesizer determines that sufficient evidence is gathered $( r _ { t } >$ threshold) or the maximum step limit is reached. Figure 3 illustrates our framework.

## 3.2 Audio-to-Graph Indexing

## 3.2.1 Acoustic Semantic Alignment

To initialize the graph nodes, we transform the continuous audio stream into discrete, semantically meaningful units with precise temporal boundaries.

VAD and Transcription. We employ FSMN-VAD to segment audio A into clips, filtering silence. These clips are transcribed by a high-fidelity ASR model to obtain text T. We apply CTCbased Forced Alignment: for each token $w _ { i } .$ , we perform constrained Viterbi decoding to align text with acoustic features, yielding exact boundaries $[ t _ { s t a r t } , t _ { e n d } ]$

Speaker Diarization. Identifying “who spoke” is insufficient; understanding “who they are” (e.g., role, stance) is key to reasoning. We propose a multimodal profiling mechanism. We extract speaker embeddings using ERes2NetV2 and perform incremental clustering. If the cosine similarity between a new embedding and existing clusters is below a threshold $\eta = 0 . 8$ , a new speaker ID is created.

Semantic Role Generation. A raw ID (e.g., Speaker\_0) lacks semantic context. We construct a Speaker Profile $\mathcal { P } _ { k }$ for each unique speaker $S _ { k }$ We aggregate all utterances belonging to $S _ { k }$ and prompt an LLM to distill attributes:

$$
\mathcal { P } _ { k } = \mathtt { L L M } ( \mathtt { C o n c a t } ( \{ v _ { i } . \mathtt { t e x t } | v _ { i } . \mathtt { s p k } = S _ { k } \} ) )\tag{1}
$$

The output is a structured profile, e.g., {Role: Project Manager, Gender: Male, Stance: Conservative}. This profile is stored as a node attribute.

## 3.2.2 Multi-dimensional Graph Modeling

Beyond the raw meeting transcripts, we also incorporate the acoustic features (e.g., speaker identity, start/end timestamps, and corresponding speech) and explicitly model the dialogue flow as a heterogeneous graph $\mathcal { G } = ( \nu , \mathcal { E } )$

Nodes (V): Each utterance is treated as a fundamental node $v _ { i } ,$ enriched with attributes including text transcripts, speaker identity, start/end timestamps, and corresponding speech, etc.

Edges (E): To support the diverse reasoning tasks, we construct multiple edge types:

(1) Temporal Edges $( e _ { t e m p } )$ connecting adjacent utterances $( v _ { i } \to v _ { i + 1 } )$ ;

(2) Reply-To Edges $( e _ { r e p l y } ) { : }$ Captures conversational turns. We establish a directed edge $v _ { i }  v _ { j }$ if $v _ { j }$ occurs within a dynamic window $\Delta t < 5 s$ after $v _ { i }$ and speakers are different;

(3) SameSpeaker Edges $( e _ { s p k } )$ Connects nodes $v _ { i } , v _ { j }$ where spk $( v _ { i } ) = { \mathfrak { s p k } } ( v _ { j } )$ . This enables the agent to aggregate scattered opinions of a specific person;

(4) Entity Edges $( e _ { e n t } )$ derived from entity keyword co-occurrence or discourse relations.

(5) Semantic $E d g e s \left( e _ { s e m } \right)$ To solve context forgetting, we employ an LLM to detect coreference chains. $\operatorname { I f } v _ { j }$ contains pronouns (e.g., “that idea”) referring to an entity in $v _ { i } .$ , we create a semantic link $v _ { j } \xrightarrow { s e m } v _ { i }$

## 3.3 GRGA Planning Process

Belief Initialization: Query Decomposition. Given a user query Q, the belief state is initialized by projecting the unstructured query into a structured constraint space C. We define the decomposition function $f _ { \mathrm { d e c } } : \mathcal { Q }  \mathcal { C }$ as:

$$
\mathcal { C } = f _ { \mathrm { d e c } } ( Q ) = \{ c _ { e } , c _ { c } , c _ { t } , c _ { m } \}\tag{2}
$$

where $c _ { e } , c _ { c } , c _ { t } , c _ { m }$ denote constraints on entities, concepts, time, and metadata, respectively. The initial belief is set as $b _ { 0 } = H _ { 0 } = \{ Q , { \mathcal { C } } \}$

Policy Network: Execution Planning. The Execution Planner approximates the policy π<sub>θ</sub> parameterized by an In-Context Learning (ICL ) (Dong et al., 2024) guided LLM. At step t, it generates a multi-step plan $\mathcal { P } _ { t }$ conditioned on the current belief $b _ { t }$

$$
\mathcal { P } _ { t } \sim \pi _ { \theta } ( \cdot \mid b _ { t } )\tag{3}
$$

The plan $\mathcal { P } _ { t }$ is a sequence of atomic operations derived from the Action Space A (see Table 11):

$$
\mathcal { P } _ { t } = [ o p _ { 1 } , o p _ { 2 } , \hdots , o p _ { k } ] , \quad o p _ { i } \in \mathcal { A }\tag{4}
$$

This formulation enables long-horizon planning, allowing the agent to chain logical operations (e.g., Search → Filter) before interacting with the environment.

Environment Interaction: Tool Execution. The Execution Engine functions as the transition interface. It executes the plan $\mathcal { P } _ { t }$ on the graph $\mathcal { G }$ to obtain an observation $o _ { t } \mathbf { : }$

$$
o _ { t } = \operatorname { E x e c } ( \mathcal P _ { t } , \mathcal G ) = \{ s _ { 1 } , s _ { 2 } , \dots , s _ { n } \}\tag{5}
$$

where each segment $\begin{array} { l c l } { s _ { i } } & { = } & { \left( \mathrm { t e x t } _ { i } , \mathrm { t i m e } _ { i } , \mathrm { s p k } _ { i } \right) } \end{array}$ Upon receiving $o _ { t } ,$ the agent updates its belief state via a deterministic transition function ψ:

$$
b _ { t + 1 } = \psi ( b _ { t } , \mathcal { P } _ { t } , o _ { t } ) = b _ { t } \oplus \{ \mathcal { P } _ { t } , o _ { t } \}\tag{6}
$$

Reward Estimation: Synthesis & Reflection. To guide the reasoning trajectory refinement, we employ a two-stage mechanism as reward function.

1) Answer Synthesis. The synthesizer generates a candidate answer $\hat { A } _ { t }$ and citations $\mathrm { C i t e } _ { t }$ based on the accumulated evidence in $b _ { t + 1 } \colon$

$$
( \hat { A } _ { t } , \mathbf { C i t e } _ { t } ) = f _ { \mathrm { s y n } } ( Q , b _ { t + 1 } )\tag{7}
$$

2) Reflection as Sparse Reward. The Reflector evaluates the logical entailment between the evidence $E \subset b _ { t + 1 }$ and the answer $\hat { A } _ { t }$ , assigning a verification score $s _ { \mathrm { v e r } } \in [ 0 , 1 ]$

$$
s _ { \mathrm { v e r } } = f _ { \mathrm { r e f } } ( Q , \hat { A } _ { t } , E )\tag{8}
$$

The reward $r _ { t }$ is defined as a threshold function:

$$
r _ { t } = \left\{ { \begin{array} { l l l } { 1 } & { { \mathrm { i f ~ } } s _ { \mathrm { v e r } } \geq \tau } & { ( { \mathrm { S u c c e s s , T e r m i n a t e } } ) } \\ { - \beta } & { { \mathrm { i f ~ } } s _ { \mathrm { v e r } } < \tau } & { ( { \mathrm { F a i l u r e , R e \mathrm { - } p l a n } } ) } \end{array} } \right.\tag{9}
$$

If $r _ { t } < 0$ , the negative feedback $f b _ { t }$ (critique) is injected into the belief state: $b _ { t + 1 }  b _ { t + 1 } \cup \{ f b _ { t } \}$ prompting the policy π to generate a corrective plan $\mathcal { P } _ { t + 1 }$ in the next iteration.

## 4 Experimentation

To validate the effectiveness of our GRGA, we conduct extensive experiments on our LongAudioQA dataset.

<table><tr><td rowspan="2">Method</td><td colspan="5">AliMeeting</td><td colspan="5">AMI Meeting</td><td colspan="5">DailyTalk</td></tr><tr><td>Fact.</td><td>Infer.</td><td>Temp.</td><td>Summ.</td><td>Acou.</td><td>Fact.</td><td>Infer.</td><td>Temp.</td><td>Summ.</td><td>Acou.</td><td>Fact.</td><td>Infer.</td><td>Temp.</td><td>Summ.</td><td>Acou.</td></tr><tr><td>Speech as Context</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-Omni</td><td>29.31</td><td>27.33</td><td>15.81</td><td>19.06</td><td>16.50</td><td>20.96</td><td>26.38</td><td>15.12</td><td>16.67</td><td>12.15</td><td>34.26</td><td>38.43</td><td>2.43</td><td>2.11</td><td>9.39</td></tr><tr><td>Audio Flamingo 3</td><td>35.70</td><td>35.23</td><td>13.19</td><td>18.09</td><td>32.89</td><td>16.04</td><td>27.94</td><td>12.62</td><td>10.24</td><td>17.06</td><td>65.13</td><td>61.31</td><td>3.83</td><td>2.44</td><td>40.09</td></tr><tr><td>MiMo-Audio</td><td>54.50</td><td>54.50</td><td>16.92</td><td>39.91</td><td>25.15</td><td>52.75</td><td>64.74</td><td>35.34</td><td>38.30</td><td>17.17</td><td>81.57</td><td>78.65</td><td>5.58</td><td>3.46</td><td>31.44</td></tr><tr><td>Transcription as Context</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-Omni</td><td>47.66</td><td>63.53</td><td>38.34</td><td>32.65</td><td>11.70</td><td>53.55</td><td>46.66</td><td>37.32</td><td>43.63</td><td>14.13</td><td>80.11</td><td>77.14</td><td>62.71</td><td>15.71</td><td>18.27</td></tr><tr><td>Audio Flamingo 3</td><td>32.12</td><td>45.16</td><td>28.23</td><td>21.06</td><td>9.69</td><td>48.52</td><td>40.21</td><td>22.12</td><td>29.16</td><td>12.36</td><td>76.43</td><td>75.26</td><td>58.32</td><td>13.42</td><td>15.85</td></tr><tr><td>MiMo-Audio</td><td>48.02</td><td>64.62</td><td>38.29</td><td>29.43</td><td>15.72</td><td>53.62</td><td>44.10</td><td>37.14</td><td>41.49</td><td>17.94</td><td>80.67</td><td>76.16</td><td>62.46</td><td>15.35</td><td>17.52</td></tr><tr><td>Both Speech and Transcription as Context with RAG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BGE-M3</td><td>43.24</td><td>29.68</td><td>31.56</td><td>15.43</td><td>18.45</td><td>32.5624.56</td><td></td><td>23.78</td><td>25.36</td><td>25.34</td><td>46.72</td><td>36.31</td><td>32.46</td><td>34.79</td><td>26.21</td></tr><tr><td>CLASP</td><td>25.37</td><td>19.54</td><td>18.92</td><td>14.76</td><td>16.32</td><td>24.12</td><td>20.31</td><td>22.76</td><td>11.60</td><td>14.79</td><td>30.58</td><td>33.42</td><td>27.34</td><td>13.52</td><td>11.81</td></tr><tr><td>GRGA(ours)</td><td>57.31</td><td>66.45</td><td>39.48</td><td>44.29</td><td>39.91</td><td>59.46</td><td>65.29</td><td>38.56</td><td>48.46</td><td>35.68</td><td>85.25</td><td>79.06</td><td>63.58</td><td>61.10</td><td>52.32</td></tr></table>

Table 3: Main results on accuracy (%). We compare our proposed method against End-to-End Speech LLM and standard RAG baselines across three datasets. Bold indicates the best performance. Fact.: Factual, Infer.: Inferential, Temp.: Temporal, Summ.: Summarization, Acou.: Acoustic reasoning.

## 4.1 Baselines and Implementation

First, we compare three recent most competitive multi-modal LLMs: Qwen3-Omni (Xu et al., 2025a), Audio Flamingo 3 (Goel et al., 2025), and MiMo-Audio (Xiaomi, 2025), which are considered as SOTAs for speech or spoken text QA. Therefore, we implement them in two forms: 1) use whole meeting Speech as Context and 2) use whole meeting Transcription as Context.

Second, we investigate two tailored RAG approaches for audio QA: BGE-M3 (Chen et al., 2024a) retrieving meeting text through a query, then merging the retrieved text and corresponding audio into the model (TextRAG) and CLASP (Abootorabi and Asgari, 2025), which are also considered as the SOTAs for RAG-based audio QA, retrieving meeting audio through a query, then merging the retrieved audio and corresponding text into the model (AudioRAG). They cast both meeting Speech and Transcription as Context. Notably, both of them adopt the same LLMs as ours to implement. More details of our GRGA and above baselines can refer to Appendix G.

## 4.2 Evaluation Metric

To rigorously assess the reasoning capabilities of our model, we report Semantic Accuracy across all datasets. Traditional n-gram metrics (e.g., BLEU, Exact Match) often fail to capture the true validity of generated responses, as they penalize correct answers that differ in lexical surface forms from the ground truth. Drawing upon recent methodologies in complex reasoning evaluation (Mishra et al., 2025; Shui et al., 2023), we move beyond rigid string matching and establish a robust, automated evaluation pipeline that prioritizes semantic equivalence and logical soundness.

Specifically, we employ a high-capability LLM (LLM-as-a-Judge) to approximate human-level judgment. Formally, let $\mathcal { D } = \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N }$ denote the evaluation dataset, where $q _ { i }$ is the query and $a _ { i }$ is the ground-truth answer. Let $\hat { a } _ { i }$ represent the model-generated response. We define a semantic judgment function $\mathcal { I } .$ , parameterized by an external expert model (e.g., GPT-OSS-120B (OpenAI et al., 2025)):

$$
s _ { i } = \mathcal { I } ( q _ { i } , a _ { i } , \hat { a } _ { i } ) \in \{ 0 , 1 \} ,\tag{10}
$$

where $s _ { i } = 1$ if and only if J determines that $\hat { a } _ { i }$ entails the same semantic information as $a _ { i }$ , and 0 otherwise. The judge is prompted to disregard stylistic differences and focus solely on factual consistency and the correctness of reasoning. Finally, the Semantic Accuracy is computed as the expectation of correct judgments:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } \times 1 0 0 \% .\tag{11}
$$

This approach ensures a fair comparison by validating whether the model successfully retrieves and reasons over the core knowledge, regardless of its output phrasing.

## 4.3 Main Results

Table 3 shows the performance comparison of all baselines of our GRGA on our proposed three datasets for Long-form speech meeting understanding. From this table, we can see:

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">WER (↓)</td><td colspan="2">DER (↓)</td></tr><tr><td>Collar = 0 s</td><td>Collar = 0.25 s</td></tr><tr><td>DailyTalk</td><td>0.85</td><td>6.42</td><td>3.41</td></tr><tr><td>AliMeeting-far</td><td>20.38</td><td>16.80</td><td>13.10</td></tr><tr><td>AMI-sdm</td><td>18.84</td><td>17.60</td><td>14.90</td></tr></table>

Table 4: Error analysis on benchmark datasets. We report Word Error Rate (WER) and Diarization Error Rate (DER) under strict (0 s) and standard (0.25 s) collar (tolerance ) settings. All metrics are reported in percentage (%), and ↓ denotes that lower values are better.

![](images/32a0a6055b3d02998c518b483246e84e5118463f44151ecd4a718725ce1b5bcd.jpg)  
Figure 4: Human evaluation results. Our method shows significant improvements, particularly in Groundedness across complex meeting scenarios.

Overcoming Context Limitations. End-to-End Speech LLMs (e.g., Qwen3-Omni) degrade sharply on long-form datasets, with AudioFlamingo3 dropping from ∼ 65% on DailyTalk to ∼ 16% on AMI. This confirms that fixed context windows hinder reasoning in long-form audio. Conversely, our GRGA maintains robust performance, outperforming the strongest baseline (MiMo-Audio) on AMI, validating the scalability of our graph-based retrieval beyond context limits.

Planning vs. Naive Retrieval. Comparisons with RAG baselines highlight the failure of “oneshot” retrieval. Text RAG, while effective for factual questions, struggles significantly with inferential questions (24.6% vs. ours 65.3% on AMI). This proves that vector similarity alone cannot capture multi-hop dependencies. Our Query Planner bridges this gap by decomposing queries into logical chains. Additionally, the poor performance of Audio RAG (∼ 20%) indicates that raw acoustic retrieval is too noisy, justifying our use of a structured graph intermediate.

## 4.4 Analysis and Discussion

Human Evaluation. As illustrated in Figure 4, our method consistently outperforms Qwen3-Omni across all datasets and metrics. Notably, the most substantial gap appears in Groundedness (e.g., +1.27 on AllMeeting). While the baseline generates fluent text (high Coherence), it frequently lacks evidentiary support in multi-party dialogues. Our superior grounding directly translates to higher Correctness, confirming that precise temporal anchoring effectively reduces factual errors.

![](images/6a3358117ae9b9e32414162aaad368b9cade353526335ad8e0b9efac02f6da3e.jpg)  
Figure 5: Citation Precision-Recall Trade-off. Results are reported in % with a ±2s tolerance.

Upstream Performance Analysis. Table 4 highlights the noise disparity across datasets. DailyTalk serves as a clean baseline with minimal errors (WER 0.85%). Conversely, AMI and AliMeeting present high noise rates, with WERs ∼ 20% and DERs > 13%. Despite these significant upstream errors in transcription and diarization, our GRGA maintains high QA accuracy (Table 3), demonstrating the robustness of our iterative planning and reflection mechanisms against noisy inputs.

Evidence Recall & Precision Analysis. To assess the model’s ability to locate supporting evidence, we report the Precision (P) and Recall (R) of generated citations (timestamps) against ground truth spans (with a ±2s tolerance) in Figure 5 and Table 7. Our GRGA significantly outperforms baselines across all datasets. Specifically, compared to standard Text RAG, our method improves Precision by +19.3% on AMI and +6.7% on AliMeeting. This indicates that our Query Planner effectively filters out irrelevant noise, while the Reflection mechanism ensures that retrieved segments are strictly relevant to the answer, minimizing hallucinated citations common in naive retrieval approaches.

Ablation Study To validate the necessity of each component in our GRGA, we conduct an ablation study on the AMI dataset as a typical example in Table 5. From this table, we can see that removing any module results in varying degrees of performance degradation for our GRGA. Especially, the results of w/o Semantic Search demonstrate that our approach can indeed solve the semantic understanding problem in long-form speech meeting scenario. The removal of Graph Traversal indicates the importance of our designed multi-dimensional graph. More analysis can be found in Appendix 6.

<table><tr><td rowspan="2">Ablation Settings</td><td colspan="2">Overall</td><td colspan="5">AMI</td></tr><tr><td>Avg.</td><td>∆</td><td>Fact. Infer.</td><td>Temp.</td><td></td><td>Summ.</td><td>Acou.</td></tr><tr><td>Ours (Full)</td><td>49.49</td><td>0</td><td>59.46</td><td>65.29</td><td>38.56</td><td>48.46</td><td>35.68</td></tr><tr><td colspan="8">Validation of Query Decomposer</td></tr><tr><td>w/o Query Planner Validation of Query Planner</td><td>46.39</td><td>-3.10</td><td>|57.63 59.35</td><td></td><td>36.79</td><td>45.73</td><td>32.43</td></tr><tr><td>w/o Query Planner Validation of Reflection</td><td>44.68</td><td>-4.80</td><td>56.12 56.35</td><td></td><td>36.41</td><td>43.18</td><td>31.37</td></tr><tr><td>w/o Reflection</td><td>39.64</td><td>-9.84</td><td>51.32</td><td>49.52</td><td>31.10</td><td>39.83</td><td>26.45</td></tr><tr><td colspan="8">Validation of Action Space</td></tr><tr><td>w/o Tool: Filtering</td><td>43.69</td><td>-5.79</td><td>57.81</td><td>62.27</td><td>35.36</td><td>41.45</td><td>28.58</td></tr><tr><td>w/o Tool: Semantic Search</td><td>16.28</td><td>-33.21</td><td>28.9616.38</td><td></td><td>12.21</td><td>13.69</td><td>10.15</td></tr><tr><td>w/o Tool: Graph Traversal</td><td>38.21-11.27</td><td></td><td>45.57 38.62</td><td></td><td>32.31</td><td>41.25</td><td>33.31</td></tr><tr><td>w/o Tool: Audio Access</td><td></td><td>34.71-14.78</td><td>47.52</td><td>249.12</td><td>33.13</td><td>31.71</td><td>12.06</td></tr></table>

Table 5: Ablation study on the AMI dataset. We report the accuracy (%) drop when specific components are removed. ∆: Performance degradation compared to the full framework.

Step Analysis can be found in Appendix C. Case Study can be found in Appendix D. Noise Sensitivity Analysis can be found in Appendix E.

## 5 Related Work

Speech and Meeting Question Answering. Early research in speech QA primarily focused on extracting answer spans from short, single-speaker speech segments, such as Spoken SQuAD (Li et al., 2018) and HeySQuAD (Wu et al., 2023). They normally focus on answering ranking (Hu et al., 2026), instead of generative QA. While recent datasets like AMI (Jain et al., 2024) and AliMeeting (Yu et al., 2022a) provide rich multi-speaker meeting resources, they are predominantly used for ASR and diarization tasks rather than complex reasoning. Existing QA models on these datasets often treat transcripts as flat text sequences, neglecting the intricate temporal and interpersonal dependencies inherent in meetings. In contrast, our work targets long-form meeting comprehension, requiring models to navigate graph-structured dialogues involving multiple speakers and temporal dynamics.

Large Speech Language Models. Recent advancements in multi-modal LLMs have enabled direct processing of audio inputs. However, current Speech LLMs face significant limitations in context length. For instance, Kimi-Audio (KimiTeam et al., 2025) is optimized for short clips (< 30s) and suffers from truncation issues with longer inputs. Even state-of-the-art models like AudioFlamingo3 (Goel et al., 2025) are typically constrained to context of approximately 10 minutes. This bottleneck renders them unsuitable for long-form meeting analysis in the absence of external retrieval mechanisms.

Retrieval-Augmented Generation (RAG). RAG has emerged as a standard paradigm for grounding LLMs in external knowledge (Lewis et al., 2020). Traditional “Retrieve-then-Generate” approaches rely on dense vector similarity to retrieve relevant contexts in a single pass. However, these methods struggle with multi-hop reasoning, where the evidence is fragmented or requires logical deduction steps not captured by semantic similarity alone (Liu et al., 2025; Tang et al., 2025). Recent works have explored recursive retrieval or chain-of-thought prompting to address these limitations. Our framework advances this paradigm by formalizing retrieval as an iterative planning process, specifically tailored to handle the noisy and unstructured nature of ASR transcripts.

LLM Agents and Tool Using. The emergence of LLMs has catalyzed the development of autonomous agents capable of using tools to solve complex tasks, exemplified by frameworks like ReAct (Yao et al., 2023) and Toolformer (Schick et al., 2023). While these agents demonstrate proficiency in open-domain tasks (e.g., web browsing, math) (Zhang et al., 2025), their application to structured audio understanding remains underexplored. We bridge this gap by defining a specialized Action Space for meeting analysis (e.g., time\_range\_search, hybrid\_search) and incorporating a Reflection mechanism modeled as a POMDP (Lauri et al., 2023b), enabling the agent to self-correct in partially observable audio environments.

## 6 Conclusion

We present GRGA, an agentic framework that addresses the acoustic missing and context forgetting issues in long-form speech understanding. By structuring audio into a multimodal heterogeneous graph and formulating QA as a POMDP, we enable an agent to explicitly plan, navigate, and reason over complex interactions. Extensive experiments on our proposed LongAudioQA benchmarks demonstrate that our GRGA significantly outperforms both end-to-end Speech-LLMs and RAG-based SOTAs.

## Acknowledgements

This work was supported by Jiangsu Province Frontier Program Project (BF2025036), and Hong Kong RGC grant GRF #15611021. They are also with Jiangsu Key Lab of Language Computing, Suzhou.

## Limitations

Our work has three main limitations.

(1) Error Propagation: Since graph construction relies on upstream ASR and diarization, severe acoustic noise or speaker overlap may introduce artifacts into the reasoning graph.

(2) Inference Latency: The iterative agentic workflow (planning and reflection) incurs higher computational cost than single-turn RAG, limiting real-time deployment.

(3) Domain Specificity: Our evaluation focuses on structured meetings; generalizing to unstructured domains like movies or vlogs remains future work.

## Ethics Statement

Dataset Sourcing and Compliance. Our dataset is constructed based on three existing public corpora: AliMeeting (Yu et al., 2022a), AMI Meeting Corpus (Jain et al., 2024), and DailyTalk (Lee et al., 2022). We strictly adhere to the original licensing terms of these datasets (e.g., Creative Commons BY-NC-SA 4.0 and Apache License 2.0). We have reviewed the source data to ensure that no Personally Identifiable Information (PII) beyond what was originally consented to by the participants is exposed. For the AMI corpus, we utilize the specific split intended for academic research involving simulated scenarios, minimizing privacy risks associated with real-world private meetings.

Human Annotation and Fair Compensation. We employ highly qualified graduate students as annotators. We ensured that all participants were compensated at a rate significantly above the local minimum wage (approximately \$5 per hour), respecting fair labor standards. We also implemented strict protocols to protect annotators from exposure to any potentially harmful or offensive content, although the source datasets are generally free of such material.

Mitigation of LLM Bias and Hallucination. We acknowledge that utilizing Large Language Models (LLMs) for data generation may introduce inherent biases or hallucinations. To mitigate this, our human-in-the-loop pipeline strictly enforces factual consistency checks. We explicitly instructed annotators to discard questions that reinforce stereotypes or rely on hallucinated events not present in the audio. The high inter-annotator agreement (κ = 0.95) suggests that our verification process effectively filters out low-quality or biased generations.

Potential Societal Impact. The technology proposed in this paper aims to enhance productivity and accessibility by structuring long-form audio. However, we recognize the potential dual-use risk regarding unauthorized surveillance or privacy intrusion in workplace settings. We strongly advocate that the deployment of such meeting analysis tools must be accompanied by transparent consent from all recorded parties and robust data encryption measures. This dataset is released solely for research purposes to advance the field of interpretable audio understanding.

## References

Mohammad Mahdi Abootorabi and Ehsaneddin Asgari. 2025. Clasp: Contrastive language-speech pretraining for multilingual multimodal information retrieval. In Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part IV, page 10–20, Berlin, Heidelberg. Springer-Verlag.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Preprint, arXiv:2402.03216.

Yafeng Chen, Siqi Zheng, Hui Wang, Luyao Cheng, Qian Chen, Shiliang Zhang, and Junjie Li. 2024b. Eres2netv2: Boosting short-duration speaker verification performance with computational efficiency. Preprint, arXiv:2406.02167.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, and 3416 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A survey on in-context learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128, Miami, Florida, USA. Association for Computational Linguistics.

Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, and Bryan Catanzaro. 2025. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models. In NeurIPS 2025 : Annual Conference on Neural Information Processing Systems. NeurIPS.

Jiliang Hu, Zuchao Li, Baoyuan Qi, Liu Guoming, and Ping Wang. 2026. End-to-end contrastive languagespeech pretraining model for long-form spoken question answering. In AAAI 2026.

Aditya Jain, Fagner Cunha, Michael James Bunsen, Juan Sebastián Cañas, Léonard Pasi, Nathan Pinoy, Flemming Helsing, JoAnne Russo, Marc Botham, Michael Sabourin, Jonathan Fréchette, Alexandre Anctil, Yacksecari Lopez, Eduardo Navarro, Filonila Perez Pimentel, Ana Cecilia Zamora, José Alejandro Ramirez Silva, Jonathan Gagnon, Tom August, and 9 others. 2024. Insect identification in the wild: The ami dataset. Preprint, arXiv:2406.12452.

Alexander Johnson, Peter Plantinga, Pheobe Sun, Swaroop Gadiyaram, Abenezer Girma, and Ahmad Emami. 2024. Efficient SQA from long audio contexts: A policy-driven approach. In 25th Annual Conference of the International Speech Communication Association, Interspeech 2024, Kos, Greece, September 1-5, 2024.

KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, and 21 others. 2025. Kimiaudio technical report. Preprint, arXiv:2504.18425.

Mikko Lauri, David Hsu, and Joni Pajarinen. 2023a. Partially observable markov decision processes in robotics: A survey. IEEE Transactions on Robotics, 39(1):21–40.

Mikko Lauri, David Hsu, and Joni Pajarinen. 2023b. Partially observable markov decision processes in robotics: A survey. IEEE Transactions on Robotics, 39(1):21–40.

Keon Lee, Kyumin Park, and Daeyoung Kim. 2022. Dailytalk: Spoken dialogue dataset for conversational text-to-speech. Preprint, arXiv:2207.01063.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Chia-Hsuan Li, Szu-Lin Wu, Chi-Liang Liu, and Hung yi Lee. 2018. Spoken squad: A study of mitigating the impact of speech recognition errors on listening comprehension. Preprint, arXiv:1804.00320.

Qiwei Li, Teng Xiao, Zuchao Li, Ping Wang, Mengjia Shen, and Hai Zhao. 2025. Dialogue-rag: Enhancing retrieval for llms via node-linking utterance rewriting. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 24423–24438.

Zheng Lian, Haoyu Chen, Lan Chen, Haiyang Sun, Licai Sun, Yong Ren, Zebang Cheng, Bin Liu, Rui Liu, Xiaojiang Peng, Jiangyan Yi, and Jianhua Tao. 2025a. Affectgpt: A new dataset, model, and benchmark for emotion understanding with multimodal large language models. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net.

Zheng Lian, Haiyang Sun, Licai Sun, Haoyu Chen, Lan Chen, Hao Gu, Zhuofan Wen, Shun Chen, Siyuan Zhang, Hailiang Yao, Bin Liu, Rui Liu, Shan Liang, Ya Li, Jiangyan Yi, and Jianhua Tao. 2025b. OV-MER: towards open-vocabulary multimodal emotion

recognition. In Forty-second International Confer ence on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net.

Chyi-Jiunn Lin, Guan-Ting Lin, Yung-Sung Chuang, Wei-Lun Wu, Shang-Wen Li, Abdelrahman Mohamed, Hung-Yi Lee, and Lin-Shan Lee. 2024. Speechdpr: End-to-end spoken passage retrieval for open-domain spoken question answering. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic of Korea, April 14-19, 2024, pages 12476–12480. IEEE.

Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and Wentao Zhang. 2025. HopRAG: Multi-hop reasoning for logic-aware retrievalaugmented generation. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 1897–1913, Vienna, Austria. Association for Computational Linguistics.

Venkatesh Mishra, Bimsara Pathiraja, Mihir Parmar, Sat Chidananda, Jayanth Srinivasa, Gaowen Liu, Ali Payani, and Chitta Baral. 2025. Investigating the shortcomings of LLMs in step-by-step legal reasoning. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 7795–7826, Albuquerque, New Mexico. Association for Computational Linguistics.

OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, and 108 others. 2025. gpt-oss-120b. Preprint, arXiv:2508.10925.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. Preprint, arXiv:2212.04356.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, pages 68539–68551. Curran Associates, Inc.

Ruihao Shui, Yixin Cao, Xiang Wang, and Tat-Seng Chua. 2023. A comprehensive evaluation of large language models on legal judgment prediction. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7337–7348, Singapore. Association for Computational Linguistics.

Quanwei Tang, Sophia Yat Mei Lee, Junshuang Wu, Dong Zhang, Shoushan Li, Erik Cambria, and Guodong Zhou. 2025. A comprehensive graph framework for question answering with mode-seeking preference alignment. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21504– 21523, Vienna, Austria. Association for Computational Linguistics.

Mingqiu Wang, Izhak Shafran, Hagen Soltau, Wei Han, Yuan Cao, Dian Yu, and Laurent El Shafey. 2024. Retrieval augmented end-to-end spoken dialog models. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic ofKorea, April 14-19, 2024, pages 12056– 12060. IEEE.

Yijing Wu, SaiKrishna Rallabandi, Ravisutha Srinivasamurthy, Parag Pravin Dakle, Alolika Gon, and Preethi Raghavan. 2023. Heysquad: A spoken question answering dataset. Preprint, arXiv:2304.13689.

LLM-Core-Team Xiaomi. 2025. Mimo-audio: Audio language models are few-shot learners.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, and 19 others. 2025a. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765.

Kai-Tuo Xu, Feng-Long Xie, Xu Tang, and Yao Hu. 2025b. Fireredasr: Open-source industrialgrade mandarin speech recognition models from encoder-decoder to llm integration. Preprint, arXiv:2501.14350.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Chenyu You, Nuo Chen, Fenglin Liu, Shen Ge, Xian Wu, and Yuexian Zou. 2022. End-to-end spoken conversational question answering: Task, dataset and model. In Findings ofthe Associationfor Computational Linguistics: NAACL 2022, pages 1219–1232, Seattle, United States. Association for Computational Linguistics.

Fan Yu, Shiliang Zhang, Yihui Fu, Lei Xie, Siqi Zheng, Zhihao Du, Weilong Huang, Pengcheng Guo, Zhijie Yan, Bin Ma, Xin Xu, and Hui Bu. 2022a. M2MeT: The ICASSP 2022 multi-channel multi-party meeting transcription challenge. In Proc. ICASSP. IEEE.

Fan Yu, Shiliang Zhang, Pengcheng Guo, Yihui Fu, Zhihao Du, Siqi Zheng, Weilong Huang, Lei Xie, Zheng-Hua Tan, DeLiang Wang, Yanmin Qian, Kong Aik Lee, Zhijie Yan, Bin Ma, Xin Xu, and Hui Bu. 2022b. Summary on the ICASSP 2022 multi-channel multiparty meeting transcription grand challenge. In Proc. ICASSP. IEEE.

Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, and Xiangyu Zhao. 2025. Process vs. outcome reward: Which is better for agentic RAG reinforcement learning. NIPS 2025.

Zihan Zhao, Yiyang Jiang, Heyang Liu, Yu Wang, and Yanfeng Wang. 2025. Librisqa: A novel dataset and framework for spoken question answering with large language models. IEEE Trans. Artif. Intell., 6(11):2884–2895.

Xie Zhifei, Mingbao Lin, Zihang Liu, Pengcheng Wu, Shuicheng Yan, and Chunyan Miao. 2025. Audioreasoner: Improving reasoning capability in large audio language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 23840–23862, Suzhou, China. Association for Computational Linguistics.

## A More Analysis in Ablation Study

<table><tr><td rowspan="2">Ablation Settings</td><td colspan="2">Overall</td><td colspan="4">AMI</td></tr><tr><td>Avg.</td><td>∆ Fact.</td><td>Infer.</td><td>Temp.</td><td>Summ.</td><td>Acou.</td></tr><tr><td>Ours (Full)</td><td>49.49</td><td></td><td>59.46 65.29</td><td>38.56</td><td>48.46</td><td>35.68</td></tr><tr><td colspan="7">Validation of Query Decomposer</td></tr><tr><td>w/o Query Planner Validation of Query Planner</td><td>46.39</td><td>-3.10</td><td>57.63 59.35</td><td>36.79</td><td>45.73</td><td>32.43</td></tr><tr><td>w/o Query Planner Validation of Reflection</td><td>44.68</td><td>-4.80</td><td>56.12 56.35</td><td>36.41</td><td>43.18</td><td>31.37</td></tr><tr><td>w/o Reflection</td><td>39.64</td><td>-9.84</td><td>51.3249.52</td><td>31.10</td><td>39.83</td><td>26.45</td></tr><tr><td colspan="7">Validation of Action Space</td></tr><tr><td>w/o Tool: Filtering</td><td>43.69</td><td>-5.79</td><td>57.8162.27</td><td>35.36</td><td>41.45</td><td>28.58</td></tr><tr><td>w/o Tool: Semantic Search</td><td>16.28-33.21</td><td></td><td>28.96 16.38</td><td>12.21</td><td>13.69</td><td>10.15</td></tr><tr><td>w/o Tool: Graph Traversal</td><td>38.21-11.27</td><td></td><td>45.57 38.62</td><td>32.31</td><td>41.25</td><td>33.31</td></tr><tr><td>w/o Tool: Audio Access</td><td>34.71-14.78</td><td></td><td>47.5249.12</td><td>33.13</td><td>31.71</td><td>12.06</td></tr></table>

Table 6: Ablation study on the AMI dataset. We report the accuracy (%) drop when specific components are removed. ∆: Performance degradation compared to the full framework.

To validate the necessity of each component in AudioGraph, we conduct an ablation study on the AMI dataset (Table 6).

Impact of Cognitive Modules. Removing the Query Planner leads to a significant drop of 4.80%, particularly in inferential tasks (-8.94%). This confirms that complex queries (e.g., multi-hop reasoning) cannot be solved by single-step retrieval; explicit planning is essential for decomposing intents. Most notably, the removal of the Reflection mechanism causes a sharp decline of 9.84%. Without the “verify” loop, the agent is prone to hallucination, accepting the first retrieved chunk even if it is irrelevant. This validates our hypothesis that a POMDP-style feedback loop is critical for robustness.

Impact of Graph Tools. The Graph Traversal tool proves indispensable $( \Delta \ : = \ : - 1 1 . 2 7 \% )$ When disabled, the agent degrades to a flat-text searcher, failing to aggregate scattered information via speaker edges $( e _ { s p k } )$ or temporal edges $( e _ { t e m p } )$ . Furthermore, removing Audio Access severely impacts acoustic-aware questions (Accuracy $3 5 . 6 8 \% \to 1 2 . 0 6 \%$ , demonstrating that text transcripts alone are insufficient for capturing paralinguistic cues like emotion or speaker overlap. Finally, Semantic Search serves as the foundational entry point; its removal collapses the system $( \Delta = - 3 3 . 2 1 \% )$ , as the agent loses the ability to locate initial evidence nodes.

<table><tr><td rowspan="2">Method</td><td colspan="2">DailyTalk</td><td colspan="2">AMI</td><td colspan="2">AliMeeting</td></tr><tr><td>P</td><td>R</td><td>P</td><td>R</td><td>P</td><td>R</td></tr><tr><td>Qwen3-Omni</td><td>32.6</td><td>24.6</td><td>25.7</td><td>27.3</td><td>29.3</td><td>28.2</td></tr><tr><td>+ Text RAG</td><td>67.3</td><td>69.1</td><td>56.9</td><td>65.2</td><td>72.4</td><td>68.3</td></tr><tr><td>+ Audio RAG</td><td>61.6</td><td>63.6</td><td>45.2</td><td>42.7</td><td>57.3</td><td>49.6</td></tr><tr><td>+ Ours</td><td>84.2</td><td>77.2</td><td>76.2</td><td>69.7</td><td>79.1</td><td>74.1</td></tr></table>

Table 7: Citation accuracy comparison. Results are reported in % with a ±2s tolerance. P: Precision, R: Recall.

## B Evidence Precision Recall Analysis

Table 7 compares the evidence citation accuracy across three datasets, demonstrating that our method consistently outperforms both the vanilla model and RAG baselines.

## C Step Analysis

![](images/65a164cfdbe3e454069da9cf87a21de6b9a94adba3af95d9961f22b86c9c387a.jpg)  
Figure 6: Average reasoning steps across different question types. The agent exhibits adaptive computation: solving simple Factual queries requires minimal steps, while complex Inferential and Temporal queries trigger deeper reasoning chains. The stacked colors illustrate the distribution of tool usage, showing heavy reliance on Graph Traversal for multi-hop reasoning.

Complexity-Aware Reasoning. The results demonstrate a clear correlation between question difficulty and planning depth. For Factual queries (e.g., “What is the budget?”), the agent adopts a “shortcut” strategy, typically resolving the intent in just 1.8 steps using primarily Semantic Search. This indicates high efficiency. In contrast, Inferential and Temporal queries trigger significantly longer trajectories (avg. >4 steps). This confirms that the agent is actively performing multi-hop reasoning—iteratively traversing e<sub>temp</sub> or e<sub>spk</sub> edges to aggregate scattered evidence, rather than relying on a single retrieval pass.

Tool Usage Distribution. The stacked breakdown reveals task-specific tool preferences. Summarization tasks show a dominant usage of Filter tools (Green) to isolate specific time ranges or speakers. Crucially, Acoustic-aware questions exhibit a high frequency of Audio Access (Red) calls in the final steps, validating that the agent correctly learns to “listen” to the raw audio only when textual transcripts are insufficient (e.g., verifying emotion).

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Human Ratings (1-5 Scale)</td><td rowspan="2">Avg.</td><td rowspan="2">IAA</td></tr><tr><td>Corr.</td><td>Grd.</td><td>Coh.</td></tr><tr><td rowspan="2">AMI</td><td>Qwen3-Omni Ours</td><td>1.79</td><td>2.12</td><td>3.76</td><td>2.56</td><td>0.78</td></tr><tr><td></td><td>2.25</td><td>3.25</td><td>4.35</td><td>3.28 2.83</td><td></td></tr><tr><td>AllMeeting</td><td>Qwen3-Omni Ours</td><td>1.93 2.36</td><td>2.24 3.51</td><td>4.32 4.37</td><td>3.41</td><td>0.75</td></tr><tr><td>DailyTalk</td><td>Qwen3-Omni Ours</td><td>3.43 4.11</td><td>3.72 4.23</td><td>4.12 4.37</td><td>3.76 4.24</td><td>0.82</td></tr><tr><td>Overall</td><td>Average Qwen3-Omni Average Ours Improvement (%)</td><td>2.38 2.91 +0.52</td><td>2.69 3.66 +0.97</td><td>4.07 4.36 +0.30</td><td>=</td><td></td></tr></table>

Table 8: Human evaluation results across different datasets. Models are assessed on three dimensions: Correctness (Corr.), Groundedness (Grd.), and Coherence (Coh.) using a 1-5 scale. We also report average scores (Avg.) and Inter-Annotator Agreement (IAA).

## D Human Evaluation.

To validate real-world performance, we conducted a blind evaluation on 150 queries randomly sampled across five question types from AMI, AliMeeting, and DailyTalk. Three graduate student annotators assessed the responses (IAA=0.78 on avg.). Results in Table 8 shows:

The “Fluency vs. Factualit” Gap. While the baseline (Qwen3-Omni) maintains high Coherence (4.07), its low Groundedness score (2.69) indicates a tendency to generate fluent but hallucinated content. In contrast, our method achieves a massive +0.97 improvement in Groundedness. This confirms that our approach does not merely generate text but accurately anchors answers to precise timestamps and speakers, effectively mitigating hallucinations.

Grounding Drives Correctness. There is a clear positive correlation between groundedness and correctness. By explicitly locating evidence, our method achieves a +0.52 gain in Correctness. This demonstrates that superior localization capabilities directly translate to more factually accurate answers, particularly for reasoning-heavy questions.

Robustness in Complex Scenarios. The performance gap is most pronounced on challenging datasets like AMI and AliMeeting (noisy, multiparty interactions) compared to the cleaner DailyTalk. Our method maintains distinct advantages in these “hard” settings (e.g., +1.13 Groundedness on AMI), proving its robustness where standard omni-models struggle with speaker attribution and temporal reasoning.

![](images/376e8ac5912fc2c70a0bf1d970c982a5055d24e971d8f50779434415086b6b6d.jpg)

![](images/17aa50a4b18cd568942fa6ec3910a66b91299975428c7f24ec97dcc24cf9991c.jpg)

![](images/c2095701a8a2c48d9e804728fd10d6661095b1ab21a2c32054cf1e796fb9e065.jpg)  
Figure 7: Noise Sensitivity Analysis. QA Accuracy comparison under varying upstream error rates. The rightmost panel highlights the widening performance gap (∆ Accuracy), showing that GRGA’s advantage grows in noisier environments.

## E Noise Sensitivity Analysis

To evaluate the robustness of our proposed GRGA against upstream errors, we conducted a sensitivity analysis by simulating varying levels of Word Error Rate (WER) and Diarization Error Rate (DER). We compare our model with the baseline Text RAG system across a 3 × 3 grid of noise conditions.

We define three noise levels for both ASR and Diarization components:

Low (Oracle): We use Ground Truth (GT) transcripts and speaker labels to simulate an ideal scenario.

Mid (Standard): We utilize the outputs from our standard pipeline (as described in Section H), representing a realistic deployment scenario (DER ≈ 17.6%, WER ≈ 18.8%).

High (Simulated Noise): For High WER, we utilize a lower-performance lightweight ASR model (openai/whisper-small (Radford et al., 2022)) to generate transcripts with high error rates (WER ≈ 33.6%). For High DER, we simulated noise by randomly shuffling a subset of speaker labels within a meeting while preserving timestamp boundaries. This represents a worst-case scenario where speaker identity information is highly unreliable (DER ≈ 52.4%).

Figure 7 visualizes the impact of noise, revealing distinct behaviors:

Graceful Degradation. Text RAG suffers catastrophic drops as noise increases, falling from 54.47% to 24.67% in the High/High setting. This underscores the fragility of standard RAG when explicit text or speaker cues are noisy. Conversely, GRGA demonstrates graceful degradation, retaining 39.54% accuracy even under the harshest conditions.

The “Noise Buffer” Effect. Notably, the performance gap between GRGA and the baseline widens from +5.37% (Low/Low) to +14.87% (High/High). This indicates that our graph structure acts as an effective noise buffer. By leveraging topological connectivity and temporal constraints, the agent can compensate for corrupted signals, reducing the system’s reliance on perfect transcription and diarization.

## F Case Study

Table 9 presents a case study illustrating our method’s self-correction capability, where the agent initially generates a low-confidence response but, triggered by the reflection module, re-plans to retrieve specific evidence (e.g., concrete complaints about deadlines), ultimately producing a highly grounded and accurate answer.

## G Experimental Setting Details

## G.1 Baselines

Qwen3-Omni. Qwen3-Omni (Xu et al., 2025a) is a state-of-the-art (SOTA) end-to-end omni-modal model that processes interleaved inputs (e.g., text and audio) and generates text responses. It supports speech inputs of up to 40 minutes. The model adopts a Mixture-of-Experts (MoE) architecture with 30B parameters, of which 3B are activated per inference. Notably, it achieves performance comparable to top-tier proprietary models (e.g., Gemini 2.5 Pro (Comanici et al., 2025)) across a wide range of audio understanding benchmarks.

<table><tr><td>Stage</td><td>Module / Action</td><td>Execution Details &amp; Internal State</td></tr><tr><td colspan="3">User Query: &quot;What is the team&#x27;s general attitude to the project deadline and workload?&quot;</td></tr><tr><td rowspan="4">Phase 1</td><td>Intent Analysis</td><td>Entities: [team&#x27;, project deadline&#x27;, ‘workload&#x27;] Concept: [general attitude&#x27;]</td></tr><tr><td>Plan Execution</td><td>1. Hybrid Search: query=“team project deadline workload&quot; (→ 10 hits) 2. Traverse Relations: Expand context (→ 64 nodes found).</td></tr><tr><td>Draft Answer</td><td>&quot;The team generally views the deadline as tight.. members feeling urgency.&quot; (Generic summary, lacks specific details)</td></tr><tr><td>Reflection (Critic)</td><td>Assessment: XConfidence: 2/5 Reasoning: “Supported is False. Need reverse confirmation for tight’ and ‘work- intensive’ claims.&quot; ↔ Decision: Trigger RE-PLAN.</td></tr><tr><td rowspan="5">Phase 2</td><td>Re-Plan Strategy</td><td>New Goal: Validate sentiment strength (e.g., stress, specific complaints). Tool: hybrid_search(query=“project deadline focused effort&quot;, alpha=0.6)</td></tr><tr><td>Evidence Retrieval</td><td>The new search strategy retrieves highly specific emotional utterances: • utt180: “..only got six days to do the topic segmentation&quot; • utt225: &quot;shall we change the deadlines?&quot; • utt349: “that&#x27;s a hassle too it&#x27;s a lot of work&quot;</td></tr><tr><td>Final Response</td><td>• ut t553: “i wanna do it superficially&quot; &quot;The team&#x27;s attitude indicates the project deadline is perceived as tight... One member states they only have six days [utt180]. Another asks shall we change</td></tr><tr><td></td><td>the deadlines?’ [utt225]. Another calls it ‘a hassle’ [utt349]... expresses a preference to handle it superficially’ [utt553].&quot;</td></tr><tr><td>Final Reflection</td><td>Assessment: √Confidence: 5/5 Status: Supported.</td></tr></table>

Table 9: Case Study on Self-Correction. Initially, the model retrieved broad context but generated a low-confidence (2/5), generic answer. The Reflection module detected the lack of specific evidence and triggered a re-planning step focusing on “focused effort.” This allowed the model to retrieve concrete complaints (e.g., “six days,” “hassle,” “superficially”), resulting in a highly grounded (5/5) final response.

AudioFlamingo3. We select Audio Flamingo 3 (Goel et al., 2025) as a SOTA audio-language model. Built upon Whisper encoder and 7B LLM, it employs an “on-demand thinking” mechanism to facilitate chain-of-thought reasoning. Crucially, its architecture processes inputs via 30-second windows with a strict context cap of 10 minutes. This baseline serves to quantify the performance degradation of context-constrained models when applied to long meeting scenarios.

MiMo-Audio. MiMo-Audio (Xiaomi, 2025) is a SOTA end-to-end audio language model. It combines a 1.2B audio tokenizer with a 7B LLM. The model supports a 128k context window and downsamples audio tokens to 6.25 Hz, enabling efficient processing of long sequences. Crucially, its instruction-tuning stage incorporates “thinking” mechanisms, allowing chain-of-thought reasoning directly over audio inputs. Comparing against MiMo-Audio helps assess whether our graph-based structural reasoning provides advantages beyond large-scale end-to-end pretraining.

Text RAG Baseline. We implement a standard dense retrieval baseline using BGE-M3 (Chen et al., 2024a) embeddings. Meeting transcripts are segmented into utterances with prepended metadata (e.g., [time] SPK: text). Given a query q, we retrieve the top-k (k = 10, λ = 0.25) most relevant segments based on cosine similarity, get text result and corresponding audio clips. Crucially, to maintain discourse coherence, retrieved segments are temporally reordered before being fed into the Speech-LLM. This baseline represents the conventional approach to handling long meetings without graph-based structural reasoning.

Audio RAG Baseline. We construct a crossmodal retrieval baseline using CLASP (Abootorabi and Asgari, 2025), which aligns audio and text in a shared semantic space. The long audio is segmented into clips based on speaker diarization timestamps. Given a text query q, we retrieve the top-k most relevant audio clips via cosine similarity between the query embedding and CLASPencoded audio embeddings. These retrieved raw audio clips, along with their speaker metadata, are then fed directly into the Speech-LLM (same backbone as ours) to generate the response. This baseline evaluates the efficacy of retrieving raw acoustic features versus our proposed graph-based semantic navigation.

## G.2 Evaluation Metric: Semantic Accuracy

To rigorously assess the reasoning capabilities of our model, we report Semantic Accuracy across all datasets. Traditional n-gram metrics (e.g., BLEU, Exact Match) often fail to capture the true validity of generated responses, as they penalize correct answers that differ in lexical surface forms from the ground truth. Drawing upon recent methodologies in complex reasoning evaluation (Mishra et al., 2025; Shui et al., 2023), we move beyond rigid string matching and establish a robust, automated evaluation pipeline that prioritizes semantic equivalence and logical soundness.

Specifically, we employ a Strong LLM (LLMas-a-Judge) to approximate human-level judgment. Formally, let $\mathcal { D } = \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N }$ denote the evaluation dataset, where $q _ { i }$ is the query and $a _ { i }$ is the ground-truth answer. Let $\hat { a } _ { i }$ represent the modelgenerated response. We define a semantic indicator function $\mathbb { I } ( \cdot )$ parameterized by an external judge $\mathcal { I }$ (e.g., GPT-OSS-120B (OpenAI et al., 2025)):

$$
s _ { i } = \mathcal { I } ( q _ { i } , a _ { i } , \hat { a } _ { i } ) \in \{ 0 , 1 \} ,\tag{12}
$$

where $s _ { i } = 1$ if and only if $\mathcal { I }$ determines that $\hat { a } _ { i }$ entails the same semantic information as $a _ { i } .$ , and 0 otherwise. The judge is prompted to ignore stylistic differences and focus strictly on factual consistency and reasoning correctness. Finally, the Semantic Accuracy is computed as the expectation of correct judgments:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } \times 1 0 0 \% .\tag{13}
$$

This approach ensures a fair comparison by validating whether the model successfully retrieves and reasons over the core knowledge, regardless of its output phrasing.

## H Implementation Details

All the methods that did not explicitly state the model used Qwen3-Omni (Xu et al., 2025a).

Graph Construction Setup. For the initial audio processing, we utilize the FireRedASR-AED (Xu et al., 2025b) model for Automatic Speech Recognition (ASR) to transcribe the raw audio meeting data. To obtain precise word-level timestamps $( \mathrm { i } . \mathrm { e } . , t _ { s t a r t }$ and $t _ { e n d } )$ , we employ the Montreal Forced Aligner (MFA).

Speaker Verification: We use a pre-trained ERes2NetV2 (Chen et al., 2024b) model to extract speaker embeddings. The cosine similarity threshold for speaker clustering is set to $\eta = 0 . 8$ based on validation set performance. Edge Construction: The temporal window for establishing “Reply-To” edges is set to $\Delta t = 5$ seconds. Attribute Extraction: We employ Qwen3-Omni (via vllm API) to extract speaker profiles and resolve coreference chains during the node enrichment phase.

Agent Planning Configuration The core planning mechanism of GRGA depends on specific hyper-parameters governing the explorationexploitation trade-off: Backbone Model: The Policy Network $\pi _ { \theta }$ is parameterized by Qwen3-Omni, operated with a temperature of 0 to ensure deterministic reasoning paths. Reward Function: As defined in Eq. (9), the self-reflection verification threshold is set to τ = 4 (1-5 Scale). The penalty factor for failed verification is set to $\beta \ : = \ : 0 . 5$ Search Constraints: The maximum depth for the reasoning tree is limited to $K _ { m a x } = 5$ steps. If the agent fails to reach a verified conclusion within these steps, the process terminates and returns the current best candidate.

All experiments were conducted on a server cluster equipped with 8 × Ascend 910B (64GB) GPUs.

## H.1 Text RAG Baseline.

To benchmark our method against a standard textbased retrieval pipeline, we implement a dense Retrieval-Augmented Generation (RAG) baseline over meeting transcripts.

Context construction. We segment the meeting transcript into individual utterances and treat each utterance as one retrieval unit. Each unit is formatted with temporal and speaker metadata:

$$
u _ { i } = [ \mathsf { s t a r t } _ { i } ~ \mathsf { - } ~ \mathsf { e n d } _ { i } ] ~ \mathsf { S P E A K E R \_ I D : } ~ \mathsf { t e x t } _ { i } .
$$

Dense retrieval. We use BGE-M3 (Chen et al., 2024a) as the embedding backbone. Given a query q, we compute embeddings for the query and each utterance:

$$
\mathbf { e } _ { q } = f _ { \mathrm { B G E } } ( q ) \in \mathbb { R } ^ { 1 0 2 4 } , \qquad \mathbf { e } _ { i } = f _ { \mathrm { B G E } } ( u _ { i } ) \in \mathbb { R } ^ { 1 0 2 4 } .
$$

We apply L2 normalization to enable cosine-based scoring:

$$
\hat { \mathbf { e } } = \frac { \mathbf { e } } { \| \mathbf { e } \| _ { 2 } } , \qquad s _ { i } = \cos ( \hat { \mathbf { e } } _ { q } , \hat { \mathbf { e } } _ { i } ) = \hat { \mathbf { e } } _ { q } ^ { \top } \hat { \mathbf { e } } _ { i } .
$$

We rank all utterances by $s _ { i }$ and retrieve the top-k segments $( k = 1 0 )$ that satisfy a minimum similarity threshold $\lambda = 0 . 2 5$

$$
{ \mathcal { C } } _ { \mathrm { s i m } } = \mathrm { T o p K } \left( \{ u _ { i } \mid s _ { i } \geq \lambda \} , k \right) .
$$

Temporal reordering. To preserve discourse coherence, we reorder the retrieved utterances by their start time before forming the final context:

$$
\mathcal { C } = \mathrm { S o r t B y T i m e } ( \mathcal { C } _ { \mathrm { s i m } } ) .
$$

Generation. We concatenate the reordered segments as context $c = { \mathrm { C o n c a t } } ( { \mathcal { C } } )$ and prompt the generator to answer strictly based on the retrieved context. We use the same backbone LLM as in our main method for a fair comparison:

$$
y = \operatorname { L L M } ( q , c ) .
$$

## H.2 Audio RAG Baseline.

Given a text query q and a long audio recording A, we build a retrieval-augmented generation (RAG) baseline based on CLASP (Abootorabi and Asgari, 2025), a multilingual audio-text representation model that maps raw speech into a shared 768-dimensional semantic space.

## Audio segmenttation.

We segment the long audio A into K shorter clips $\{ a _ { k } \} _ { k = 1 } ^ { K }$ , by speaker diarization.

Embedding computation. We compute the query embedding and the audio clip embeddings using CLASP:

$$
\begin{array} { r } { \mathbf { e } _ { q } = f _ { \mathrm { t e x t } } ( q ) \in \mathbb { R } ^ { 7 6 8 } , } \end{array}\tag{14}
$$

$$
\mathbf { e } _ { k } = f _ { \mathrm { a u d i o } } ( a _ { k } ) \in \mathbb { R } ^ { 7 6 8 } , \quad k = 1 , \dots , K ,\tag{15}
$$

where $f _ { \mathrm { a u d i o } } ( \cdot )$ encodes speech (e.g., via HuBERT and spectrogram encoders with a fusion module), and $f _ { \mathrm { t e x t } } ( \cdot )$ encodes text into the same representation space.

Retrieval. We rank audio clips by cosine similarity and select the most relevant clip (or top-M clips):

$$
s _ { k } = \cos ( \mathbf { e } _ { q } , \mathbf { e } _ { k } ) = { \frac { \mathbf { e } _ { q } ^ { \top } \mathbf { e } _ { k } } { \| \mathbf { e } _ { q } \| _ { 2 } \| \mathbf { e } _ { k } \| _ { 2 } } } ,\tag{16}
$$

$$
k ^ { \star } = \arg \operatorname* { m a x } _ { k \in \{ 1 , \ldots , K \} } s _ { k } .\tag{17}
$$

Optionally, we retrieve $\boldsymbol { K } = \mathrm { T o p } \mathbf { M } ( \{ s _ { k } \} _ { k = 1 } ^ { K } )$ for multi-evidence prompting.

Generation. We feed the query and retrieved evidence to a Speech LLM to generate the final response:

$$
y = \mathrm { L L M } \big ( q , \{ a _ { k } \} _ { k \in \mathcal { K } } \big ) ,\tag{18}
$$

where the evidence is provided, contains retrieval speaker diarization and corresponding audio clips.

## I Multi-dimensional Meeting Database Construction Algorithm

Complexity. Let M be the number of VAD clips, and let $F _ { m }$ be the number of acoustic frames in clip m. Let $N _ { w }$ be the total number of word tokens after forced alignment, $N _ { s }$ the total number of diarization segments, and $N _ { u }$ the number of utterances.

ASR inference and CTC forced alignment are linear in the input length, costing $\textstyle { \mathcal { O } } { \bigl ( } \sum _ { m = 1 } ^ { M } F _ { m } { \bigr ) }$ time. Speaker diarization requires embedding extraction over frames plus clustering over segments; in practice this is $\mathcal { O } \big ( \textstyle \sum _ { m = 1 } ^ { M } F _ { m } \big )$ for feature extraction, and an additional clustering cost by AHC is $\mathcal { O } ( N _ { s } ^ { 2 } )$ ). Speaker assignment via overlap matching costs $\mathcal { O } ( N _ { w } + N _ { s } )$ using the twopointer implementation in Alg. 1 (a naive all-pairs matching would be $\mathcal { O } ( N _ { w } N _ { s } ) )$ . Building utterances from word streams is $\mathcal { O } ( N _ { w } )$ , and adding temporal edges in the meeting graph is $\mathcal { O } ( N _ { u } )$

The space complexity is dominated by storing word- and utterance-level annotations, i.e., $\mathcal { O } ( N _ { w } + N _ { u } )$ , plus the storage overhead of the text and vector indices for utterance retrieval.

```prolog
Algorithm 1: Multi-dimensional Meeting Database Construction
Input: Long meeting audio X; session id sid; hyper-parameters
θ = {max_clip_len, vad_pad, $\beta _ { \mathrm { s i l } } , \alpha _ { \mathrm { o v l p } } , K \}$
Output: Structured database D; meeting graph $G = ( V , E )$ ; indices I
1 X ← PREPROCESS(X) ; $/ /$ resample/mono/normalize
2 C ← FSMN_VAD(X, vad_pad, max_clip_len) ; // clips with absolute spans
3 Initialize tables CLIPS, WORDS, SPKSEGS, UTTERANCES;
4 Initialize graph $G = ( V , E )$ ;
5 foreach clip $c \in { \mathcal { C } }$ do // process each VAD clip
6 $\mathrm { C L I P S } \gets \mathrm { C L I P S } \cup \{ ( s i d , c . i d , c . t _ { s } , c . t _ { e } , c . w a v ) \}$
7 $( T , \mathbf { P } ) \gets \mathrm { A S R } ( c . w a v )$ // transcript and CTC posteriors
8 W ← CTC_FORCEDALIGN(T, P);
; $/ / \mathcal { W } = \{ ( w _ { i } , a _ { i } , b _ { i } , p _ { i } ) \}$ word-level timestamps
9 $\begin{array} { r } { S \gets \mathrm { D I A R I Z E } \big ( c . w a v \big ) } \end{array}$ $~ / / ~ { \cal S } = \{ ( s p k _ { j } , s _ { j } , e _ { j } , e m b _ { j } , c o n f _ { j } ) \}$
10 S ← CLUSTERANDSMOOTH(S) ; // merge/smooth speaker segments
11 $\mathrm { S P K S E G S }  \mathrm { S P K S E G S } \cup \{ ( s i d , c . i d , S ) \} ;$
12 W ←f ASSIGNSPEAKERTOWORDSFAST(W, S, α<sub>ovlp</sub>);
13 $\mathrm { W o R D S }  \mathrm { W o R D S } \cup \{ ( s i d , c . i d , \widetilde { \mathcal { W } } ) \} ;$
14 $\mathcal { U } \gets \mathbf { B U I L D U T T E R A N C E S } ( \widetilde { \mathcal { W } } , \beta _ { \mathrm { s i l } } )$ ; // group into utterances
15 foreach utterance u ∈ U do
16 UTTERANCES ← UTTERANCES
∪{(sid, u.id, u.spk, u.t<sub>s</sub>, u.t<sub>e</sub>, u.text, u.audio_ref, u.conf)};
17 $V \gets V \cup \{ u . i d \}$ ; // utterance nodes in G
18 E ← E ∪ ADDTEMPORALEDGES(UTTERANCES) ; // time adjacency
19 E ← E ∪ ADDSPEAKEREDGES(UTTERANCES) ; // same-speaker links (optional)
20 E ← E ∪ ADDENTITYORTOPICEDGES(UTTERANCES) ; // optional NER/topic links
21 I ← BUILDBM25INDEX(UTTERANCES.TEXT);
22 I<sub>vec</sub> ← BUILDVECTORINDEX(UTTERANCES, K);
23 $\mathcal { T }  \{ \mathcal { T } _ { b m 2 5 } , \mathcal { T } _ { v e c } \} ;$
24 D ← {CLIPS, WORDS, SPKSEGS, UTTERANCES, I};
25 return D, G, I;
```

## J Retrieval-Generation Agent Algorithm

```latex
Algorithm 2: GRGA with Formal Notation
Input: $\overline { { Q \in \mathcal { Q } , \mathcal { G } = ( V , E , X ) } }$
Output: $\hat { A } \in { \mathcal A }$ with $C i t e \subseteq V \times T$
// Decomposition: $f _ { d e c } : \mathcal { Q }  \mathcal { C }$
1 $\mathcal { C } = f _ { d e c } ( Q ) = \{ c _ { e } , c _ { c } , c _ { t } , c _ { m } \}$
2 $b _ { 0 } = H _ { 0 } = \{ Q , { \mathcal { C } } \} ; \quad k \gets 0$
// POMDP Loop
3 while $k < K _ { m a x }$ do
// Policy: $\pi _ { \theta } : B \to \Delta ( { \mathcal { A } } ^ { * } )$
4 $\mathcal { P } _ { k } \sim \pi _ { \theta } ( \cdot \mid b _ { k } )$ where $b _ { k } = b _ { 0 } \cup H _ { k }$
5 if $\mathcal { P } _ { k } = \perp$ then
6 return ⊥
$/ /$ Transition: $T : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$
7 $o _ { k } = \bigoplus _ { o p \in \mathcal { P } _ { k } }$ Exec(op, G) where $o _ { k } \in \mathcal { O }$
$/ /$ Belief Update: $\psi : B \times A \times \mathcal { O } \to B$
8 $b _ { k + 1 } = \psi ( b _ { k } , \mathcal { P } _ { k } , o _ { k } ) = b _ { k } \cup \{ \mathcal { P } _ { k } , o _ { k } \}$
// Synthesis: $f _ { s y n } : \mathcal { Q } \times \mathcal { B }  \mathcal { A } \times \mathcal { C }$
9 $( \hat { A } _ { k } , C i t e _ { k } ) = f _ { s y n } ( Q , b _ { k + 1 } )$
// Reward: $R : { \mathcal { S } } \times { \mathcal { A } } $ R
10 $s _ { v e r } = f _ { r e f } ( Q , \hat { A } _ { k } , o _ { k } ) \in [ 0 , 1 ]$
$r _ { k } = { \left\{ \begin{array} { l l } { 1 } \\ { - \beta } \end{array} \right. }$ if $s _ { v e r } \geq \tau$
11
otherwise
12 if $r _ { k } > 0$ then
13 return $( \hat { A } _ { k } , C i t e _ { k } )$
14 else
15 $H _ { k + 1 } = H _ { k } \cup \{ ( \mathcal { P } _ { k } , o _ { k } , \hat { A } _ { k } , r _ { k } ) \}$ $k \gets k + 1$
16 return ⊥
```

<table><tr><td>Tool</td><td>Time Complexity</td><td>Description</td></tr><tr><td>Filter</td><td> $O ( | V | + | E | )$ </td><td>Metadata filter</td></tr><tr><td>Search</td><td> $O ( | V | \cdot d _ { \mathrm { e m b e d } } + k \log k )$ </td><td>Vector retrieval + sorting</td></tr><tr><td>GraphTraversal</td><td> $O ( \bar { d } \cdot \mathrm { d e p t h } )$ </td><td>BFS (: avg degree)</td></tr><tr><td>Temporal</td><td>O(1)</td><td>Indexed temporal query</td></tr><tr><td>AudioAccess</td><td>O(1)</td><td>Get audio clip</td></tr></table>

Table 10: Time Complexity of Graph Operations

## J.1 Complexity Analysis

We provide a comprehensive theoretical analysis of the time and space complexity of the proposed GRGA algorithm. We examine each component’s computational cost and derive the overall complexity bounds.

## J.1.1 Time Complexity Analysis

The total time complexity of GRGA is determined by the iterative POMDP loop and its constituent operations:

$$
T _ { \mathrm { t o t a l } } = T _ { \mathrm { d e c } } + \sum _ { k = 0 } ^ { K _ { \mathrm { m a x } } } \left( T _ { \mathrm { p l a n } } ^ { ( k ) } + T _ { \mathrm { e x e c } } ^ { ( k ) } + T _ { \mathrm { s y n } } ^ { ( k ) } + T _ { \mathrm { r e f } } ^ { ( k ) } \right)\tag{19}
$$

where $K _ { \mathrm { m a x } }$ is the maximum number of iterations.

Query Decomposition. The decomposition function $f _ { \mathrm { d e c } } : \mathcal { Q }  \mathcal { C }$ involves a single LLM inference:

$$
T _ { \mathrm { d e c } } = { \cal O } ( | Q | \cdot d _ { \mathrm { m o d e l } } )\tag{20}
$$

where |Q| denotes the query length in tokens and $d _ { \mathrm { m o d e l } }$ is the model dimension. This is a one-time operation outside the main loop.

Execution Planning. At iteration k, the policy network $\pi _ { \theta }$ generates a plan conditioned on the belief state $b _ { k }$ :

$$
T _ { \mathrm { p l a n } } ^ { ( k ) } = O ( \left| b _ { k } \right| \cdot d _ { \mathrm { m o d e l } } + L _ { \mathrm { p l a n } } \cdot \left| \mathcal { A } \right| )\tag{21}
$$

where:

$| b _ { k } | = O ( | Q | + | { \mathcal { C } } | + k \cdot | o _ { \mathrm { a v g } } | )$ grows linearly with iteration count

$L _ { \mathrm { p l a n } }$ is the average plan length

$| { \cal A } |$ is the action space size

Tool Execution. The execution engine applies $L _ { \mathrm { p l a n } }$ operations on the meeting graph $\mathcal { G } = ( V , E )$ The complexity depends on the tool type:

The worst-case execution time is:

$$
T _ { \mathrm { e x e c } } ^ { ( k ) } = O ( L _ { \mathrm { p l a n } } \cdot \operatorname* { m a x } \{ | V | \cdot d _ { \mathrm { e m b e d } } , | V | \log | V | \} )\tag{22}
$$

Answer Synthesis. The synthesizer $f _ { \mathrm { s y n } }$ processes accumulated evidence:

$$
T _ { \mathrm { s y n } } ^ { ( k ) } = O \left( \left. \bigcup _ { i = 0 } ^ { k } o _ { i } \right. \cdot d _ { \mathrm { m o d e l } } + L _ { \mathrm { a n s w e r } } \cdot d _ { \mathrm { m o d e l } } \right)\tag{23}
$$

where $L _ { \mathrm { a n s w e r } }$ is the generated answer length. Note that $\left| \bigcup _ { i = 0 } ^ { k } o _ { i } \right| = O ( k \cdot | o _ { \mathrm { a v g } } | )$ grows with iterations.

Reflection. The reflector $f _ { \mathrm { r e f } }$ evaluates logical entailment:

$$
T _ { \mathrm { r e f } } ^ { ( k ) } = O \left( ( | Q | + | \hat { A } _ { k } | + | o _ { k } | ) \cdot d _ { \mathrm { m o d e l } } \right)\tag{24}
$$

Aggregate Complexity

$$
\begin{array} { r } { T _ { \mathrm { t o t a l } } = O \bigg ( K _ { \mathrm { m a x } } \cdot \big [ ( | Q | + k \cdot | o _ { \mathrm { a v g } } | ) \cdot d _ { \mathrm { m o d e l } } } \\ { + L _ { \mathrm { p l a n } } \cdot | V | \cdot d _ { \mathrm { e m b e d } } \big ] \bigg ) } \end{array}\tag{25}
$$

Under typical conditions where $| V | \gg | Q |$ and $d _ { \mathrm { e m b e d } } \approx d _ { \mathrm { m o d e l } }$ , this simplifies to:

$$
T _ { \mathrm { t o t a l } } = O ( K _ { \operatorname* { m a x } } \cdot | V | \cdot d _ { \operatorname { m o d e l } } )\tag{26}
$$

In practice, $K _ { \operatorname* { m a x } } \in [ 1 , 5 ]$ and early stopping (when $s _ { \mathrm { v e r } } \geq \tau )$ significantly reduces the average iteration count $\bar { K } < K _ { \operatorname* { m a x } }$

## J.2 Space Complexity Analysis

The space requirements consist of three main components:

$$
S _ { \mathrm { t o t a l } } = S _ { \mathrm { g r a p h } }\tag{27}
$$

Graph Storage. The meeting graph requires storage for structure and embeddings:

$$
S _ { \mathrm { g r a p h } } = O ( | V | + | E | + | V | \cdot d _ { \mathrm { e m b e d } } )\tag{28}
$$

For a typical 30 minutes meeting:

$| V | \approx$ 900 nodes

• d<sub>embed</sub> = 1024 (BGE-M3)

• Storage: ∼2MB (structure) + ∼3.5MB (embeddings)

## K Tools Details

The table 11 shows the definitions of the atomic tools in our Action Space (A).

<table><tr><td>Category Tool Signature</td><td></td><td>Description &amp; Purpose</td></tr><tr><td rowspan="2">Retrieval</td><td>keyword_search(query)</td><td>BM25-based search for precise entity/term lookup.</td></tr><tr><td>semantic_search(query)</td><td>Dense vector retrieval for abstract semantic con- cepts.</td></tr><tr><td rowspan="2">Filtering</td><td>hybrid_search(query, α = 0.6)</td><td>Weighted combination of keyword and semantic scores to optimize recall.</td></tr><tr><td> $\mathsf { f i l t e \_ t i m e \_ r a n g e } ( t _ { s t a r t } , t _ { e n d } )$ </td><td>Retrieves utterances strictly within a specified time window.</td></tr><tr><td>Traversal</td><td>filte_speaker(nodes, spk_id) traverse_relations(nodes,</td><td>Filters a set of candidate nodes by speaker iden- tity. Walks along graph edges (e.g., Next, Reply-To) to</td></tr><tr><td>Audio</td><td>depth=k) audio_segment</td><td>trace dialogue threads for multi-hop reasoning. Retrieves raw waveform data to ground text in</td></tr><tr><td></td><td> $( t _ { s t a r t } , t _ { e n d } )$ </td><td>acoustic signals (e.g., for emotion detection).</td></tr></table>

Table 11: The definitions of the atomic tools in our Action Space (A). The Query Planner invokes these tools to interact with the meeting graph and retrieve evidence.

## L QA Examples

Header. The box title Example # (QA\_TYPE, Key=#) indicates the example index, its question type (e.g., Factual ), and a unique identifier Key used to retrieve the corresponding record in the released JSON files.

Evidence section (lower part). Below the dashed line, we display the verbatim evidence excerpt(s) for the referenced utterance IDs. Each excerpt is formatted as: Evidence [utt\_id]: [t\_start – t\_end] SPEAKER\_#: transcript. Here, [t\_start – t\_end] denotes the absolute timestamps (in seconds) within the meeting, and SPEAKER\_# is the diarization-based speaker label. This layout makes the supervision explicitly grounded: readers can directly verify that the answer is entailed by the cited utterance(s), and systems can be evaluated on both answer correctness and evidence retrieval.

Example 1 ( Factual , Key=1)

Answer: Speaker 4 has written code to read and remove the XML, but is struggling to store the data (e.g., into a vector) or display it on the screen using Java.

Rationale: Speaker 4 explicitly states their current coding roadblock regarding Java vectors and display.

Evidence [513]: [1083.43 - 1100.41] SPEAKER\_4: i’ve been trying to g write something to read the x. m. l. and get rid of it and i can get rid of it but i’m having trouble putting it anywhere else so it will come up on the screen for the moment i haven’t managed to put it into a vector or whatever in java to play with it.

## Example 2 ( Inferential , Key=26)

<table><tr><td>Question: How did the group decide to resolve the issue of having too many pop-up windows? Answer: The group discussed using tabs (similar to Mozilla) or toggle buttons within a single window to switch between views like the full transcription and the summary, rather than opening separate windows. Evidence (utterance IDs): [265, 266, 269, 273, 304, 314]</td></tr><tr><td>Rationale: Multiple speakers contribute to the idea of using tabs/toggles to manage content within one window frame to avoid clutter.</td></tr><tr><td>Evidence [265]: [657.56 - 663.43] SPEAKER_2: so maybe you can just like choose the s same</td></tr><tr><td>window for transcription and summary. Evidence [266] : [661.99 - 664.73] SPEAKER_1: hmm so like have a tab there.</td></tr><tr><td>Evidence [269] : [665.32 - 669.42] SPEAKER_2: yeah yeah tabs are nice. Evidence [273] : [669.91 - 671.05] SPEAKER_4: mozilla style.</td></tr><tr><td>Evidence [304] : [697.23 - 703.29] SPEAKER_2: yeah uh change the contents of the same window like from transcription to summary.</td></tr><tr><td>Evidence [314] : [719.03 - 725.56] SPEAKER_2: no no it could be like transcription summary like two buttons and you just press on which ever you want.</td></tr></table>

Example 3 ( Summarization , Key=53)

Question: What are the key interface design decisions made during this meeting?

Answer: The team decided to: (1) use buttons instead of right-click menus for speaker characterisation, (2) use tabs or buttons to toggle between transcription and summary in the same window rather than separate windows, and (3) implement right-click menus for topics with options to view all meetings containing that topic.

Rationale: These utterances capture the major UI/UX consensus points reached through discussion.

Evidence [76]: [207.74 - 220.21] SPEAKER\_2: i guess a button button makes a bit more sense ’cause otherwise you don’t really know that oh what if i right click now what happens then it’s like more if it’s visual.

Evidence [84]: [225.08 - 232.04] SPEAKER\_4: it’s more idiot proof isn’t it it’s got a button.

Evidence [88]: [229.16 - 231.24] SPEAKER\_3: that’s true yeah it’s more intuitive really isn’t it.

Evidence [158]: [399.55 - 402.78] SPEAKER\_3: so we could do that in a similar way do it right click as well.

Evidence [167]: [411.28 - 424.13] SPEAKER\_3: so we have basically two options of of browsing the meetings is by either um searching and opening individual observations and when then we have the interlinking by right click basically.

Evidence [266]: [661.99 - 664.73] SPEAKER\_1: hmm so like have a tab there.

Evidence [269]: [665.32 - 669.42] SPEAKER\_2: yeah yeah tabs are nice.

Evidence [314]: [719.03 - 725.56] SPEAKER\_2: no no it could be like transcription summary like two buttons and you just press on which ever you want.

## Example 4 ( Temporal , Key=60)

<table><tr><td>Question: At what point does the team shift from discussing GUI to discussing the interim prototype? Answer: Around 810–824 seconds (approximately 13–14 minutes into the meeting). Evidence (utterance IDs): [361, 363, 365] Rationale: The topic pivot occurs when Speaker 3 asks what prototype they should aim for.</td></tr><tr><td>Evidence [361]: [816.58 - 822.38] SPEAKER_3: and finally the prototype he spoke about what kind</td></tr><tr><td>of prototype could we produce. Evidence [363]: [824.27 - 828.76] SPEAKER_3: because i&#x27;m i&#x27;m just you know i go into the lab and</td></tr><tr><td>i say right what am i gonna change today. Evidence [365]: [828.76 - 836.48] SPEAKER_3: you know and it kind of just it just develops i&#x27;m</td></tr><tr><td>not aiming for anything do we wanna aim for something.</td></tr></table>

## Example 5 ( Acoustic , Key=80)

Question: Why did Speaker 3 express sadness when saying “well i hope so” at approximately 24 seconds?

Answer: Speaker 3 sounded sad when expressing hope that others had done some work, likely in response to Speaker 4’s question “has anybody done anything” and Speaker 1’s admission “not a lot no”, suggesting disappointment about the team’s progress.

Evidence (utterance IDs): [13, 11, 12]

Rationale: This combines the sad emotion label from node 13 with the contextual text from surrounding nodes to explain the cause.

Evidence [11]: [ 19.07 - 23.69] SPEAKER\_4: has anybody done anything.   
Evidence [12]: [ 21.47 - 23.29] SPEAKER\_1: not a lot no.   
Evidence [13]: [ 24.17 - 25.22] SPEAKER\_3: well i hope so.

![](images/b32084905dfde7746b2ead0fa047c06b98325025f78a63641667d8f35caf3234.jpg)

## � Prediction Data (Pred)

� 查看完整数据

Basic Info

Question

Key: 36

Type:

inferential

Answer

They will mention it as future work in the conclusion of their final report, explaining they didn't have enough time.

Evidence Nodes

Nodes: 137, 140, 149, 152, 153

Reasoning

Speaker 2 suggests 'a good future work thing isn't it stick in the conclusion of the final report' and the team agrees to put it under 'changes since the initial specification'.

![](images/15bd68b581fefb7ccbbd37ac90ca7d885fcf7c9e0aa765b2fb39ba77adbb6133.jpg)

## Ground Truth (GT)

## � 查看完整数据

Basic Info

Question

Key: 36

Type:

What will the team do with the removed audio processing goal?

Answer

They will mention it as future work in the conclusion of their final report, explaining they didn't have enough time.

Evidence Nodes

Nodes: 137, 140, 149, 152, 153

Reasoning

Speaker 2 suggests 'a good future work thing isn't it stick in the conclusion of the final report' and the team agrees to put it under 'changes since the initial specification'.

![](images/9d00a41c05c05a2e3d730a7a5e1d5a0cc439e39f2e85b2d5f94edc25e03c6db3.jpg)

## � Evidence Nodes (Evidence Nodes)

![](images/243f023b1c4a4c3dbc9fd8ebfc0dc91cea072576ea338bd8264fee7b5e63929d.jpg)

## � Pred Evidence Nodes

![](images/a9791193db470f02e129b37f112c9aebecc7939c69c381190a3fef3ef9bcfa08.jpg)

Node List: [137, 140, 149, 152, 153]

## ✅ GT Evidence Nodes

Node List: [137, 140, 149, 152, 153]

<table><tr><td rowspan=1 colspan=1>Turn ID</td><td rowspan=1 colspan=1>Start</td><td rowspan=1 colspan=1>End</td><td rowspan=1 colspan=1>Speaker</td><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>Turn ID</td><td rowspan=1 colspan=1>Start</td><td rowspan=1 colspan=1>End</td><td rowspan=1 colspan=1>Speaker</td><td rowspan=1 colspan=1>Text</td></tr><tr><td rowspan=1 colspan=1>137</td><td rowspan=1 colspan=1>357.38</td><td rowspan=1 colspan=1>364.72</td><td rowspan=1 colspan=1>SPK_2</td><td rowspan=1 colspan=1>it&#x27;s a good thing to have to saį</td><td rowspan=1 colspan=1>137</td><td rowspan=1 colspan=1>357.38</td><td rowspan=1 colspan=1>364.72</td><td rowspan=1 colspan=1>SPK_2</td><td rowspan=1 colspan=1>it&#x27;s a good thing to have to say</td></tr><tr><td rowspan=1 colspan=1>140</td><td rowspan=1 colspan=1>364.72</td><td rowspan=1 colspan=1>368.42</td><td rowspan=1 colspan=1>SPK_2</td><td rowspan=1 colspan=1>should if we had more time th</td><td rowspan=1 colspan=1>140</td><td rowspan=1 colspan=1>364.72</td><td rowspan=1 colspan=1>368.42</td><td rowspan=1 colspan=1>SPK_2</td><td rowspan=1 colspan=1>should if we had more time th</td></tr><tr><td rowspan=1 colspan=1>149</td><td rowspan=1 colspan=1>379.35</td><td rowspan=1 colspan=1>382.66</td><td rowspan=1 colspan=1>SPK_3</td><td rowspan=1 colspan=1>but put it under changes since</td><td rowspan=1 colspan=1>149</td><td rowspan=1 colspan=1>379.35</td><td rowspan=1 colspan=1>382.66</td><td rowspan=1 colspan=1>SPK_3</td><td rowspan=1 colspan=1>but put it under changes since</td></tr><tr><td rowspan=1 colspan=1>152</td><td rowspan=1 colspan=1>382.66</td><td rowspan=1 colspan=1>389.01</td><td rowspan=1 colspan=1>SPK_3</td><td rowspan=1 colspan=1>and say you know we didn&#x27;t w</td><td rowspan=1 colspan=1>152</td><td rowspan=1 colspan=1>382.66</td><td rowspan=1 colspan=1>389.01</td><td rowspan=1 colspan=1>SPK_3</td><td rowspan=1 colspan=1>and say you know we didn&#x27;t w</td></tr><tr><td rowspan=1 colspan=1>153</td><td rowspan=1 colspan=1>385.04</td><td rowspan=1 colspan=1>391.36</td><td rowspan=1 colspan=1>SPK_2</td><td rowspan=1 colspan=1>we don&#x27;t think there&#x27;s enough</td><td rowspan=1 colspan=1>153</td><td rowspan=1 colspan=1>385.04</td><td rowspan=1 colspan=1>391.36</td><td rowspan=1 colspan=1>SPK_2</td><td rowspan=1 colspan=1>we don&#x27;t think there&#x27;s enough</td></tr></table>

Figure 8: Data Display

0:00 / 35:44

## � Manual Annotation

## Confidence Score

Confidence Score

1 - Very Uncertain ❌⚠️

2 - Uncertain

3 - Somewhat Confident �

4 - Confident ✅

5 - Very Certain ⭐

## � Scoring Guide

1: Answer likely unsupported or contradicted by evidence

2: Only small part of answer supported or major gaps exist

3: About half of answer supported with notable uncertainties

4: Most claims supported with only minor doubts

5: Every claim directly and fully supported by evidence

Current Score

1/5

Reasoning ((Optional))

Please explain your scoring rationale...

## Additional Annotations

## Groundedness

Groundedness

1 - Completely Inaccurate ❌

Has Contradiction

2 - Most of it is inaccurate

3 - Partially Accurate �

4 - Fairly Accurate ✅

1: Evidence nodes have no relation to answer

5 - Completely Accurate ⭐

## � Scoring Guide

2: Only few evidence items relevant

� Save Annotation

3: About half evidence supports answer

4: Most evidence supports answer

5: All evidence precisely supports answer

Groundedness

![](images/4b88c8b4cd2d99f0ec8560ab26d0a895e710c48d105abde325885b704d504622.jpg)

Missing Evidence

Skip

## Coherence

## Coherence

1 - Completely Incoherent ❌

2 - Mostly Incoherent

3 - Partially Coherent �

4 - Fairly Coherent ✅

5 - Completely Coherent ⭐

## � Scoring Guide

1: Logic confused or selfcontradictory

2: Obvious logical jumps

Not annotated

3: Some logical gaps

4: Logic basically smooth

5: Logic clear and rigorous

## Coherence

![](images/0e2d49a211bb9d839afe084e2cb915be47ee5c904f0c98464cdb0cdaf02ce84a.jpg)

Irrelevant Information

Previous

Next

Progress: 1 / 1

Annotated

⚠️Figure 9: Manual Evaluation

## M Manual Evaluation Tool

## N Data Construction Prompt

## Prompt for Acoustic-Aware QA Generation

```markdown
# Role
You are an expert AI assistant specializing in multimodal speech dataset construction. Your task is
to generate high-quality Acoustic-Aware QA pairs based on the provided audio and speech transcription
and metadata.
# Input Data format
An audio clip.
A list of Nodes. Each Node contains:
‘turn_id‘: [‘timestamp start‘-‘timestamp end‘] Speaker ‘speaker‘: ‘text‘ (Emotion: ‘emotion‘)
(Volume: ‘Volume‘)
# Task Definition: Acoustic-Aware QA
You must generate questions that cannot be answered by reading the text alone. The answer must
require checking the Acoustic Features (specifically the ‘emotion‘ and ‘timestamp‘ fields).
## Critical Requirement 1: Focus on High Arousal Emotions
You must prioritize questions regarding strong emotions such as Anger, Happiness, Excitement, or
Sadness.
### How to handle "Neutral" data:
If the input data contains only "Neutral" labels:
• Do: Ask Verification Questions checking for the presence of strong emotions.
– Good Example: "Did Speaker 3 sound angry when asking about the annual meeting?"
– Good Answer: "No. According to the audio data, Speaker 3 maintained a neutral tone."
• Don’t: Ask passive descriptive questions like "What was the emotion?".
## Critical Requirement 2: NO Confidence Scores
• Strictly Forbidden: Do NOT mention the numeric confidence scores (e.g., ‘0.99‘, ‘1.0‘) in the
Question or the Answer.
• Simply state the emotion label as a fact (e.g., "The speaker was angry").
## Allowed Question Patterns
1. Emotion-Content Attribution (Why is he angry?):
"Why did spk3 sound angry at the 15th minute?" (Combine Emotion label + Text analysis).
2. Specific Time Identification:
"Who sounded the most excited around 10 seconds?"
3. Emotion Verification (For Neutral Data):
"When Speaker 1 said [Text], did they express happiness?"
"Did the speaker sound furious or annoyed during the discussion?"
## NOT Allowed Question Patterns
• Wrong Questions:
Q: Why did Speaker 3 sound sad when saying ’i think the search works as well’ around 45 seconds?
A: Speaker 3 did not sound sad at that moment.
# Output Format (Strict JSON)
Output a valid JSON List. Ensure ‘evidence_nodes‘ is a list of integers. Using ’ instead of " for
citation
```json
{{
"question": "The specific 'question' string.",
"answer": "The brief 'answer' derived from the text.",
"evidence_nodes": [10, 12], // Must be a list of turn_ids
"reasoning": "Brief explanation of the logic."
}}
# Input:
{input_data}
# Output:
```

Prompt for Graph-based QA Generation   
# Role   
You are an expert data annotator specialized in Graph-based Audio Understanding and Conversational   
AI. Your goal is to construct a massive, high-quality Graph RAG dataset from raw ASR transcripts.   
# Input Data Structure   
A list of Nodes. Attributes:   
‘turn\_id‘: [‘timestamp start‘-‘timestamp end‘] Speaker ‘speaker‘: ‘text‘   
# Mission   
Your mission is to exhaustively mine the provided transcript for every possible piece of information   
and convert it into a Question-Answer (QA) pair. Do not stop at just one question per category.   
Aim to generate as many valid, distinct QA pairs as the data supports (e.g., 10-20 pairs for a   
medium-length segment).   
# Critical Guidelines (The "Don’t"s)   
1. Filter Noise: Do NOT generate Factual questions about "YEAH", "OKAY", "MM-HMM" (backchannels).   
• Bad: "Who said ’Yeah’?"   
• Good (Inferential): Use "Yeah" nodes only as evidence for "Agreement" or "Consensus".   
2. Avoid Hallucination: Every answer must be strictly supported by the ‘evidence\_nodes‘.   
3. No Generic Questions: Avoid vague questions like "What happened?". Be specific: "What software   
bug did Speaker A mention?"   
# Question Generation Strategy (How to generate MANY questions)   
To maximize the number of QAs, apply these specific strategies:   
## Strategy A: Entity-Centric Mining (Factual)   
Scan for every entity in the text. Generate a question for each.   
• Entities: Project names, specific numbers, technical terms, people’s names, locations, file   
paths.   
• Trigger: "I see a phone number ’04555’." -> QA: "What is the phone number mentioned?"   
## Strategy B: Logic & Linkage (Inferential)   
Look for connections between distant or adjacent nodes.   
• Cause & Effect: Node X proposes something -> Node Y rejects it. -> QA: "Why was the proposal in   
Node X rejected?"   
• Clarification: Node A asks a question -> Node B answers it. -> QA: "How did Speaker B respond   
to A’s inquiry about [Topic]?"   
• Sentiment: Look for emotional words or emphatic language. -> QA: "Which speaker expressed   
frustration about the file system?"   
## Strategy C: Time Anchors (Temporal)   
• Absolute: "What topic was introduced exactly at the 10-minute mark?"   
• Relative: "What was discussed immediately before the discussion about the budget?"   
• Duration: "How long did the debate about ’UI Design’ last?" (Calculate from start/end timestamps   
of the cluster).   
## Strategy D: Scene Understanding (Summarization)   
• Topic Segmentation: Identify where the topic shifts. -> QA: "Summarize the main points discussed   
regarding [Topic Name]."   
• Speaker Role: "Based on the segment, what seems to be the role of Speaker 0?" (e.g., Manager,   
Technical Lead).   
# Output Format (Strict JSON)   
Output a valid JSON List. Ensure ‘evidence\_nodes‘ is a list of integers.   
\`json   
[   
{{   
"type": "factual | inferential | temporal | summarization",   
"question": "The specific question string.",   
"answer": "The precise answer derived from the text.",   
"evidence\_nodes": [10, 12], // Must be a list of turn\_ids   
"reasoning": "Brief explanation of the logic."   
}}   
# Input:   
{input\_data}   
# Output: