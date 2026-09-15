![](images/2f6ece2f346e2aec1fcb379d425631fcfb089c34c4427072537655d407858b75.jpg)

# NoteVQA: Benchmarking VLMs on Real-Life Questions from Human Communities

## AllSpark Team

Vision-language models (VLMs) increasingly power consumer-facing AI search, yet evaluating them on the diversity of everyday visual questions remains challenging. Existing benchmarks often target predefined capabilities, such as multi-hop retrieval or long-form synthesis, whereas users ask photo-grounded questions spanning a long tail of everyday scenarios. Despite advances in VLMs, users on Xiaohongshu, a mainstream Chinese image-sharing platform, continue to turn to other people for help with everyday visual questions. Motivated by this behaviour, we curate NoteVQA from these questions, yielding 252 items across 12 topical categories and 7 user intents. Each item includes a concise reference distilled from expert community responses and a humanaudited interleaved reference answer that combines textual explanations with supporting visual evidence. We evaluate both short-answer correctness and interleaved-answer quality. To support the latter, we introduce AgenticInterleave, a single-agent ReAct framework for retrieval-supported answer generation, together with IVR-12, a 12-dimensional rubric for assessing the content, presentation, and image quality of interleaved references and model outputs. Across 10 frontier VLMs, the highest short-answer accuracy is 52.8%, while adding agentic search to Qwen3.5-397B-A17B improves accuracy by only 2.0%. For interleaved answers, the same model running AgenticInterleave scores 3.52 under IVR-12, compared with 4.65 for the human-audited references, with the largest gap in content quality. These results highlight the challenges that everyday visual questions pose for current VLMs in both answer accuracy and the quality of visually grounded explanations.

Date: September 14, 2026

Github: https://github.com/AllSpark-Research/notevqa

![](images/7fd3eeeeeccd02b0c81990bba9eaf6477197d47740f14791a05ad5254cdf2caf.jpg)  
Figure 1: Overview of NoteVQA. (a) A real-user visual query that remains challenging for a VLM with web retrieval but is correctly answered by community experts. (b) IVR-12 comparison of the human-audited gold reference and AgenticInterleave output across content, presentation, and image quality.

## 1 Introduction

Vision-language models (VLMs) are rapidly becoming a new interface for information seeking, allowing users to ask questions directly about the visual world around them. Yet many existing benchmarks follow a capability-first paradigm: researchers define a target capability, such as compositional visual reasoning, knowledge-based question answering, or multimodal retrieval, and then construct questions to evaluate it [15, 13, 17, 5, 11]. While methodologically sound, this approach can produce queries that differ substantially from those users submit in practice. AI search systems such as Perplexity [20] and those embedded in user-generated content (UGC) platforms must address diverse everyday information needs. In imagebased communities, users often pair a photo with a brief, naturally phrased question spanning the long tail of everyday scenarios. Strong performance on researcher-authored benchmarks may therefore not reliably translate into usefulness for real users.

We introduce NoteVQA, a benchmark built from visual questions posted by users on Xiaohongshu <sup>1</sup>, a mainstream Chinese platform for user-generated image posts. In the posts collected for NoteVQA, users share photos, ask questions, and seek answers from other community members. These questions provide a concrete setting for examining how well current VLMs address real-world visual information needs. Prior work identifies fine-grained entity recognition and knowledge-intensive visual questions as challenges for VLMs [6], motivating our focus on visual identification and local or niche knowledge. We also cover subjective judgment and authenticity checking to capture a broader range of everyday needs. Studies of social question answering highlight trust, personalised advice, and subjective recommendations as reasons for asking other people [19], underscoring the importance of answer usefulness alongside factual accuracy. Community question-answering research further associates topic-specific expertise with higher answer ratings in factual categories [1]. We therefore draw on community replies as a source of candidate answers and refine them through expert review, as illustrated in Figure 1(a).

Beyond the source of the questions, the answer format also matters. Benchmarks such as SimpleVQA [7], FVQA [25], MMSearch [14], and BrowseComp-VL (BC-VL) [9] use short reference answers, with median lengths of only a few words. Short-answer accuracy measures whether a model reaches the correct answer, but does not directly assess the quality of its explanation or supporting visual evidence. Community questions often call for more than identification: users also seek explanations, actionable guidance, and help distinguishing visually similar objects. To evaluate how well models meet these needs, NoteVQA pairs short references distilled from community responses with human-audited interleaved reference answers that combine textual explanations with supporting images. These two reference formats enable complementary evaluations of answer correctness and visually grounded explanation quality on the same questions.

Evaluating interleaved answers requires both a generation framework that can retrieve and incorporate visual evidence and criteria that assess how effectively the resulting text and images answer the question. We provide AgenticInterleave, a single-agent ReAct [27] framework that generates retrieval-supported interleaved answers with traceable image tags. We also introduce IVR-12, a 12-dimensional rubric covering content, presentation, and image quality. Applying the same criteria to human-audited references and model outputs provides a common scale for examining where generated answers succeed and fall short.

Our main contributions are as follows.

• A community-sourced visual QA benchmark with two reference formats. We construct NoteVQA, comprising 252 real-world questions across 12 topical categories and 7 user intents. Each question is paired with an expert-distilled short reference and a human-audited interleaved reference, supporting evaluation of both answer correctness and visually grounded explanation quality.

• A framework and rubric for interleaved answer evaluation. We provide AgenticInterleave for generating retrieval-supported interleaved answers with traceable image tags, together with IVR-12 for assessing content, presentation, and image quality under the same criteria for references and model outputs.

• An empirical analysis of accuracy, retrieval, and image use. We evaluate 10 frontier VLMs, with the strongest achieving only 52.8% short-answer accuracy. Agentic retrieval improves Qwen3.5- 397B-A17B by just 2.0 percentage points, while its interleaved answers show a substantial contentquality gap relative to the human-audited references. A controlled image-ablation study further quantifies the contribution of embedded visual evidence to answering image-critical probes.

## 2 Comparison with Existing Multimodal QA Benchmarks

We compare NoteVQA with 10 existing open multimodal QA benchmarks. The comparison considers three complementary perspectives: (1) query characteristics and domain coverage, (2) gains from agentic retrieval, and (3) reference-answer formats and evaluation scope. For query characteristics, we report mean query length and estimated reasoning depth. Query length provides a simple indicator of prompt complexity, while the mean number of reasoning hops, estimated by an LLM judge, approximates the depth of retrieval and reasoning required.

(1) Query Characteristics and Domain Coverage. Table 1 compares the query length, estimated reason ing depth, and primary focus of the benchmark settings. BrowseComp-VL L2 and MM-BrowseComp have mean query lengths of 56 to 74 words and estimated reasoning depths of 3.7 to 3.9 hops, reflecting tasks with extended descriptions and explicit reasoning constraints. SimpleVQA, InfoSeek, and MMSearch have shorter queries, averaging 8 to 14 words and 1.5 to 2.5 estimated hops, and primarily assess generic visual perception, entity attributes, or web-search question answering. NoteVQA averages 22.7 words and 2.48 estimated hops, placing it between short factual questions and heavily specified multi-step tasks. Its questions are distinguished by their origin in an online user community and their focus on everyday visual information needs.

Table 1: Comparison of open multimodal VQA benchmarks with our proposed NoteVQA across key statistics.
<table><tr><td>Benchmark</td><td>Num. Q. len. Hops Domain focus</td><td></td><td></td><td></td></tr><tr><td>SimpleVQA [7]</td><td>1013</td><td>10.2</td><td></td><td>1.52 Generic perception</td></tr><tr><td>FVQA [25]</td><td>1800</td><td>11.5</td><td></td><td>2.06 Factual single-entity QA</td></tr><tr><td>InfoSeek [6]</td><td>2000</td><td>8.2</td><td></td><td>1.95 KB entity attributes</td></tr><tr><td>MMSearch [14]</td><td>171</td><td>14.2</td><td></td><td>2.50 Web-search QA</td></tr><tr><td>LiveVQA [8]</td><td>300</td><td>52.1</td><td></td><td>2.52 News timeliness, MCQ</td></tr><tr><td>MMSearch+ [23]</td><td>222</td><td>12.0</td><td></td><td>2.12 arXiv multi-image QA</td></tr><tr><td>MMBC [16]</td><td>130</td><td>56.6</td><td></td><td>3.72 Web-browsing QA</td></tr><tr><td>BC-VL L1 [9]</td><td>199</td><td>20.0</td><td></td><td>3.27 Cross-modal multi-hop</td></tr><tr><td>BC-VL L2 [9]</td><td>200</td><td>74.0</td><td></td><td>3.88 Fuzzified long queries</td></tr><tr><td>VDR [29]</td><td>500</td><td>25.9</td><td></td><td>3.05 Fine-grained perception QA</td></tr><tr><td>NoteVQA</td><td>252</td><td>22.7</td><td></td><td>2.48 Consumer life</td></tr></table>

![](images/742d503cb2755f015ee9f26bb5dbc5431786efba6009a6e660338b08beec0e6c.jpg)  
Figure 2: Accuracy comparison between reasoning and agentic settings.

(2) Retrieval Gains across Benchmarks. Figure 2 compares Qwen3.5-397B-A17B [21] under two settings on each benchmark. In the single-pass setting, the model answers directly from the query and image without retrieval. In the agentic setting, the same model operates through a single-agent ReAct loop equipped with web search, page visit, and image search. Agentic retrieval improves accuracy by 14 to 43 percentage points across the 10 comparison settings. On NoteVQA, accuracy increases from 35.7% to 37.7% (Table 2), yielding a substantially smaller gain of 2.0 percentage points. This comparison shows that NoteVQA presents a setting in which the evaluated retrieval system provides limited additional benefit. Section 4 analyses the result at the instance level, examining where retrieval helps and where it fails to resolve or introduces errors.

(3) Reference-Answer Formats and Evaluation Scope. The comparison benchmarks primarily assess final-answer correctness using concise reference answers or multiple-choice labels. These formats are effective for determining whether a model reaches the expected answer, but do not directly evaluate the quality of a complete explanation supported by visual evidence. Community questions often call for richer responses that combine identification, practical guidance, and visual comparisons that help users distinguish similar objects. NoteVQA therefore provides both short references and human-audited interleaved reference answers. The two formats support complementary evaluations of answer correctness and visually grounded explanation quality on the same set of questions.

Taken together, these comparisons distinguish NoteVQA along three dimensions: it draws everyday visual questions from a real user community, presents limited gains under the evaluated agentic retrieval setting, and supports evaluation of both short and interleaved answers. The following section describes how NoteVQA selects and refines its questions and constructs the two reference formats.

![](images/0d7621a0f8a3079e294137f0dd1d52b9e96e0aaa04cd37a9734c50a71eea81e4.jpg)  
Figure 3: Data Curation Pipeline of NoteVQA. Raw notes flow through Step 1: Type cleaning to yield vision-necessary items, Step 2: Quality filtering to obtain high-confidence samples, Step 3: Refinement and balancing to produce the refined item set, and Step 4: Auditing to produce the final English benchmark.

## 3 NoteVQA

Three principles guide NoteVQA. First, questions originate from real Xiaohongshu user posts and are subsequently refined and selected for the benchmark. Second, a vision-necessity gate screens out items judged answerable from the text query alone. Third, reference construction combines community knowledge with human review: short references are distilled from comments by domain experts, while model-generated interleaved drafts are audited by annotators before inclusion.

## 3.1 Data Curation Pipeline

We implement these principles through the four-step curation pipeline shown in Figure 3.

Step 1: Type Cleaning. We first apply keyword and length filters to retain image-bearing UGC posts with sufficient textual content, then use a VLM [21] to identify potential VQA candidates. We further remove items answerable from text alone and verify image–text consistency with a VLM, reducing 5.24M raw notes to 5.84K candidates.

Step 2: Quality Rubric Filtering. Each candidate is screened by a VLM aesthetic filter and an interleavedsuitability filter requiring the answer to cover at least three substantive aspects. We then apply a five-axis rubric covering question clarity, image quality, image–text alignment, interleaved fit, and answer objectivity, followed by per-tag top-K selection. A final VLM check verifies whether the comments provide a plausible answer, leaving 338 high-confidence samples.

Step 3: Refinement and Balancing. We standardise query and answer styles and rewrite queries to remove answer-revealing domain terms while preserving the original intent. We then run Qwen3.5-397B-A17B ten times per item and use response consistency as a difficulty probe. Samples are further pruned and balanced across category and intent groups, yielding a final set of 252 items.

Step 4: Auditing. We translate the refined items into English, cross-check the translations with multiple models, and have human reviewers proofread them for correctness and fluency. Each item is then assigned to its corresponding category and intent cell as a final consistency check.

![](images/9635821076ce1c671406a6b3f1bcb6ca6d5702de46cdb30db3bc91ce513f4db8.jpg)  
Figure 4: Four-stage pipeline for constructing interleaved multimodal summaries.

## 3.2 Human-Verified Interleaved Reference Construction

As shown in Figure 4, we construct a human-verified interleaved reference for each NoteVQA item through a four-stage pipeline: trajectory collection, evidence-pool construction, candidate filtering, and summary synthesis with human audit. The pipeline combines agentic retrieval with multimodal ranking and human verification to select relevant, informative images while retaining their source links.

Stage 1: Trajectory Collection. Given the original query image, user query, and short reference answer, a multimodal Qwen3.5-397B-A17B agent runs a ReAct loop using web search, reverse image search, and page visits. The short reference is provided as guidance to encourage evidence collection around the intended answer rather than unconstrained exploration. We retain the resulting tool trajectory and retrieved sources as the evidence record for each item.

Stage 2: Evidence-Pool Construction. We construct the candidate evidence pool from two complementary sources. First, an LLM scores all visited URLs along five dimensions—relevance, informativeness, directness, uniqueness, and authoritativeness—and retains the top-5 URLs per item. Images are then crawled from these pages, subject to minimum-size constraints and a cap of 20 images per URL. Second, we preserve images returned by reverse image search during trajectory collection. After merging the two pools, Qwen3.5-397B-A17B scores each image along visual relevance, informativeness, clarity, answer support, and necessity, and retains the top-10 candidates by overall score. When the reference mentions a named entity not represented in the candidate pool, we additionally trigger forward image search to retrieve supplementary visual evidence.

Stage 3: Candidate Filtering. Before synthesis, candidate images undergo rule-based and multimodal filtering to remove evidence that is unsuitable for an interleaved reference. We remove redundant images, composites, watermarked images, video thumbnails, and off-topic content, then deduplicate the remaining images. This stage reduces visually noisy or misleading evidence while preserving complementary images that contribute distinct information to the answer.

Stage 4: Summary Synthesis. We use Qwen3.5-397B-A17B to combine the original query image, short reference, trajectory evidence, and filtered images into an interleaved multimodal reference, with each embedded image explicitly indexed to its candidate source. The generated reference then undergoes multimodal consistency checks and human review. PhD annotators label each item as accept, minor fix, or reject, correcting content and image placement when necessary. Only references that pass this audit are included in the final benchmark.

The resulting NoteVQA benchmark contains 252 items spanning 12 categories and 7 intents. Queries average 22.7 words, while short references and interleaved multimodal summaries average 107.5 words and 219.4 words, respectively. The interleaved summaries contain 378 embedded images in total, averaging 1.5 images per item. Figure 5 shows the category distribution and representative queries. Full distributional statistics are provided in Appendix A.

## 3.3 Generating and Evaluating Interleaved Multimodal Answers

We provide two complementary components for the interleaved-answer track: AgenticInterleave generates answers by retrieving and integrating textual and visual evidence, while IVR-12 (Interleaved Visual-answer Rubric) evaluates human-audited references and model outputs on a common scale.

![](images/5b14f1308ed797ff2ec4c1bb12b5473666cdc7a060662cde2c2065e0fdf15480.jpg)  
Figure 5: Dataset composition and representative examples from NoteVQA. The figure shows the distribution across 12 topical categories with representative real-user query instances for each category.

AgenticInterleave. Given a query–image pair, AgenticInterleave follows a single-agent ReAct [27] loop with four retrieval tools: text search, page visit, reverse image search, andforward image search. At each turn, the agent uses the input and accumulated tool observations to decide whether to retrieve further evidence or produce the final interleaved answer. The loop ends when an answer is generated or repeated behaviour is detected. In our experiments, the agent uses Qwen3.5-397B-A17B as its backbone and receives only the query and image; the short reference is used separately for reference construction and content evaluation.

The system prompt guides image retrieval and placement through a SUBJECT/BACKGROUND entity protocol. SUBJECT entities are the primary objects or concepts addressed by the question, while BACKGROUND entities provide contextual, comparative, or supporting information. When illustrating a SUBJECT entity, the agent must explicitly retrieve its image through forward image search. The agent selects images directly from retrieval results without a separate image-selection model and places each image tag immediately after the sentence describing the corresponding entity. Each tag records the forward image-search query and the index of a returned candidate, which a post-hoc validator checks against the tool results to verify retrieval provenance. Appendix D provides the system prompt, tool definitions, and image-tag specification.

IVR-12: Interleaved Answer Evaluation. We assess answer correctness and interleaved-answer quality separately. Short-answer accuracy is the proportion of responses classified as correct by a SimpleQA-style judge. IVR-12 complements this metric by evaluating the content, presentation, and images of an interleaved answer, using identical criteria for human-audited references and AgenticInterleave outputs.

IVR-12 comprises 12 dimensions scored from 1 to 5. The five content dimensions assess intent understanding, objective accuracy, source fidelity, completeness, and neutrality and safety. The three presentation dimensions assess reading experience, formatting and highlighting, and conciseness. The four image dimensions assess relevance, necessity and placement, visual quality, and non-redundancy. Three additional pass/fail gates cover timeliness, safety redlines, and refusal correctness. Let ${ \bar { C } } , { \bar { P } } ,$ and <sup>¯</sup>I denote the mean scores within the three categories. The overall score is

$$
S _ { \mathrm { I V R - 1 2 } } = \left\{ \begin{array} { l l } { { 0 . 5 0 \bar { C } + 0 . 2 0 \bar { P } + 0 . 3 0 \bar { I } , } } & { { \mathrm { i f ~ a l l ~ t h r e e ~ g a t e s ~ p a s s , } } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.
$$

Thus, answers that pass all gates receive a weighted score, while failure on any gate yields zero.

We use Qwen3.5-397B-A17B for both correctness judging and IVR-12 evaluation. For IVR-12, content and presentation are assessed from the answer text, with the human-audited short reference additionally supplied for content scoring. Image assessment also uses the query image and all embedded images, while gate checks use the query image and answer text. Appendix E details the scoring criteria, and Appendix ?? provides the judge prompts, parsing rules, and missing-value handling.

## 4 Experiments

We evaluate short-answer accuracy using ten models in thinking mode: Gemini-3.1-Pro [10], Doubao-Seed-2.0-Pro [22], dots3-note Preview [26], GLM-5.3-Flash [31], Kimi-K2.6 [18], Kimi-K3 [24], Qwen3.8-Flash-Next [2], Claude Opus 4.7 [3], Claude Sonnet 4.6 [4] and Qwen3.5-397B-A17B [21]. For interleaved-answer quality, we use IVR-12 to compare the outputs of Qwen3.5-397B-A17B running AgenticInterleave with the human-audited references. Qwen3.5-397B-A17B is the default backbone for AgenticInterleave. It uses the four retrieval tools described in Section 3.3, with Serper <sup>2</sup> as the backend and a maximum of 10 ReAct rounds. The same model also serves as the IVR-12 judge.

## 4.1 Main Results

Table 2 reports short-answer accuracy overall and by category; Table 10 (Appendix B) reports accuracy by intent. Gemini-3.1-Pro achieves the highest overall accuracy at 52.8%. Adding agentic retrieval to Qwen3.5- 397B-A17B raises its accuracy from 35.7% to 37.7%, a gain of 2.0%.

Table 3 compares interleaved-answer quality. The references score 4.65 under IVR-12, compared with 3.52 for AgenticInterleave, a gap of 1.13 points.

The references score higher on all 12 dimensions. The largest differences concern Content, whose mean score is 4.82 for the references and 2.95 for AgenticInterleave. Accuracy shows the largest gap (2.68 points), followed by source fidelity (2.07), completeness (1.90), and intent understanding (1.76). Scores for Presentation and Image are closer to those of the references, with mean gaps of 0.37 and 0.40, respectively. Formatting shows the largest gap within these two categories (0.86). Thus, the main shortfall under IVR-12 lies in answer content.

Table 3: Mean IVR-12 scores for human-audited references and AgenticInterleave outputs.
<table><tr><td rowspan="2">System</td><td rowspan="2"> $S _ { \mathrm { { I V R - } 1 2 } }$ </td><td colspan="5">Content</td><td colspan="3">Presentation</td><td colspan="4">Image</td></tr><tr><td>Int</td><td>Acc</td><td>Src</td><td>Cmp</td><td>Sfy</td><td>Rd</td><td>Fmt</td><td>Cnc</td><td>Rel</td><td>Pos</td><td>Qty</td><td>NR</td></tr><tr><td>Human-audited gold</td><td>4.65</td><td>4.94</td><td>4.90</td><td>4.48</td><td>4.84</td><td>4.92</td><td>5.00</td><td>4.96</td><td>4.81</td><td>4.17</td><td>4.16</td><td>4.24</td><td>4.18</td></tr><tr><td>AgenticInterleave</td><td>3.52</td><td>3.18</td><td>2.22</td><td>2.41</td><td>2.94</td><td>4.00</td><td>4.94</td><td>4.10</td><td>4.63</td><td>3.79</td><td>3.79</td><td>3.81</td><td>3.76</td></tr></table>

Dimensions: Int, intent understanding; Acc, accuracy; Src, source fidelity; Cmp, completeness; Sfy, neutrality and safety; Rd, reading experience; Fmt, formatting; Cnc, conciseness; Rel, image relevance; Pos, necessity and position; Qty, image quality; NR, non-redundancy.

Table 2: NoteVQA short-answer accuracy (%). All models use thinking mode; <sup>‡</sup> denotes agentic retrieval.
<table><tr><td>Model</td><td>Overall</td><td>Pet</td><td>Exo</td><td>Pln</td><td>Hlt</td><td>Fas</td><td>Jwl</td><td>App</td><td>Fud</td><td>Bio</td><td>CE</td><td>Hom</td><td>Oth</td></tr><tr><td>Gemini-3.1-Pro</td><td>52.8</td><td>50.0</td><td>46.3</td><td>61.8</td><td>64.3</td><td>58.8</td><td>37.5</td><td>60.0</td><td>60.0</td><td>45.5</td><td>55.6</td><td>55.6</td><td>14.3</td></tr><tr><td>Doubao-seed-2.0-pro</td><td>49.6</td><td>48.0</td><td>53.7</td><td>47.1</td><td>64.3</td><td>41.2</td><td>18.8</td><td>73.3</td><td>53.3</td><td>45.5</td><td>55.6</td><td>55.6</td><td>14.3</td></tr><tr><td>Dots-note-preview</td><td>48.4</td><td>50.0</td><td>53.7</td><td>47.1</td><td>60.7</td><td>47.1</td><td>18.8</td><td>53.3</td><td>40.0</td><td>36.4</td><td>66.7</td><td>44.4</td><td>42.9</td></tr><tr><td>GLM-5.3-Flash</td><td>48.0</td><td>46.0</td><td>48.8</td><td>52.9</td><td>53.6</td><td>23.5</td><td>43.8</td><td>53.3</td><td>53.3</td><td>45.5</td><td>55.6</td><td>44.4</td><td>57.1</td></tr><tr><td>Kimi-K2.6</td><td>47.6</td><td>42.0</td><td>43.9</td><td>50.0</td><td>57.1</td><td>52.9</td><td>56.2</td><td>60.0</td><td>46.7</td><td>36.4</td><td>44.4</td><td>44.4</td><td>28.6</td></tr><tr><td>Qwen3.8-Flash-Next</td><td>44.0</td><td>42.0</td><td>46.3</td><td>58.8</td><td>50.0</td><td>29.4</td><td>43.8</td><td>46.7</td><td>33.3</td><td>45.5</td><td>33.3</td><td>22.2</td><td>42.9</td></tr><tr><td>Claude-Opus-4.7</td><td>43.7</td><td>42.0</td><td>41.5</td><td>61.8</td><td>50.0</td><td>41.2</td><td>31.2</td><td>60.0</td><td>46.7</td><td>36.4</td><td>11.1</td><td>22.2</td><td>28.6</td></tr><tr><td>Kimi-K3</td><td>43.3</td><td>38.0</td><td>41.5</td><td>52.9</td><td>57.1</td><td>41.2</td><td>31.2</td><td>60.0</td><td>46.7</td><td>36.4</td><td>22.2</td><td>44.4</td><td>14.3</td></tr><tr><td>Claude-Sonnet-4.6</td><td>35.3</td><td>22.0</td><td>41.5</td><td>35.3</td><td>60.7</td><td>23.5</td><td>12.5</td><td>40.0</td><td>33.3</td><td>45.5</td><td>33.3</td><td>33.3</td><td>57.1</td></tr><tr><td> $\mathrm { Q w e n } 3 . 5 { \cdot } 3 9 7 \mathrm { B } { \cdot } \mathrm { A } 1 7 \mathrm { B }$ </td><td>35.7</td><td>28.0</td><td>31.7</td><td>38.2</td><td>42.9</td><td>47.1</td><td>25.0</td><td>53.3</td><td>40.0</td><td>27.3</td><td>33.3</td><td>33.3</td><td>42.9</td></tr><tr><td> $\mathrm { Q w e n } 3 . 5 { \cdot } 3 9 7 \mathrm { B } { \cdot } \mathrm { A } 1 7 \mathrm { B } ^ { \ddagger }$ </td><td>37.7</td><td>22.0</td><td>34.1</td><td>47.1</td><td>50.0</td><td>35.3</td><td>37.5</td><td>46.7</td><td>53.3</td><td>36.4</td><td>44.4</td><td>33.3</td><td>28.6</td></tr></table>

Categories: Pet, HomePet; Exo, Exotic and Wild; Pln, Plant and Garden; Hlt, Human Health; Fas, Fashion and Beauty; Jwl, Jewellery and Antique; App, Appliance, Kitchen, and Car; Fud, Food and Beverage; Bio, BioLab; CE, Consumer Electronics; Hom, Home; Oth, Other.

## 4.2 Analysis of Agentic Search Gains

To understand the limited aggregate gain from agentic retrieval, we group all 252 items by whether the reasoning-only Qwen3.5-397B-A17B baseline and its agentic counterpart answer correctly. Table 4 reports the four groups and their tool usage. Retrieval corrects 33 baseline errors but introduces 28 new errors, leaving a net gain of five correct answers. Both settings fail on 129 items (51.2%).

Table 4: Qwen3.5-397B-A17B answer outcomes with and without retrieval, and tool usage in agentic runs. and  denote correct and incorrect answers, respectively.
<table><tr><td>Bucket</td><td>Base</td><td>Agent</td><td>N</td><td>Share (%)</td><td>Rounds</td><td>Search</td><td>Visit</td><td>Img.</td></tr><tr><td>Both wrong</td><td>X</td><td>X</td><td>129</td><td>51.2</td><td>4.60</td><td>6.48</td><td>1.25</td><td>0.69</td></tr><tr><td>Both correct</td><td>√</td><td>√</td><td>62</td><td>24.6</td><td>4.31</td><td>5.81</td><td>1.34</td><td>0.55</td></tr><tr><td>Search helps</td><td>X</td><td>√</td><td>33</td><td>13.1</td><td>4.73</td><td>6.91</td><td>1.24</td><td>0.70</td></tr><tr><td>Search hurts</td><td>√</td><td>X</td><td>28</td><td>11.1</td><td>4.89</td><td>7.11</td><td>1.75</td><td>0.54</td></tr><tr><td>Total</td><td>一</td><td>一</td><td>252</td><td>100.0</td><td>4.58</td><td>6.44</td><td>1.33</td><td>0.64</td></tr></table>

Base: reasoning-only; Agent: agentic retrieval. Rounds, Search, Visit, and Img. report mean counts per agentic run; Img. denotes image search.

Correct and incorrect agentic runs use similar numbers of tool calls on average: 6.19 vs. 6.59 web searches, 1.31 vs. 1.34 page visits, and 0.60 vs. 0.66 image searches. The Search helps group averages 6.91 searches, compared with 7.11 for Search hurts. Useful-evidence hit rates are 0.12% across 1623 web-search calls and 5.69% across 334 page visits. To examine the failure patterns, a separate Qwen3.5-397B-A17B judge classifies the 157 incorrect agentic runs into tool-related (T1–T4) and non-tool (N1–N2) categories (Table 5).

Under this classification, tool-related failures account for 136 of the 157 incorrect runs (86.6%). Confirmation or anchoring loops (T3) are the largest category, with 111 cases (70.7%). In these trajectories, the agent often adopts an incorrect hypothesis early and uses subsequent searches to reinforce it, without seeking evidence that would challenge it. The Search hurts group also averages more web-search calls than the Both wrong group (7.11 vs. 6.48). This pattern is consistent with continued retrieval along an incorrect search direction, although call counts alone do not establish the cause of failure. These results suggest two obstacles to effective retrieval: obtaining relevant evidence from Chinese UGC and niche expert communities, and revising an incorrect hypothesis once search is under way. Together, the paired outcomes and failure analysis show why the evaluated agentic configuration yields only a small net improvement on NoteVQA.

Table 5: Judge-assigned failure categories for 157 incorrect agentic retrieval runs.
<table><tr><td>ID</td><td>Failure cause</td><td>BW</td><td>Loss</td><td>Total</td><td>Share (%)</td><td>Rounds</td><td>Search</td><td>Img.</td></tr><tr><td colspan="2">Tool-related failures</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T1</td><td>Reverse image search ineffective</td><td>8</td><td>2</td><td>10</td><td>6.4</td><td>5.30</td><td>6.70</td><td>1.00</td></tr><tr><td>T2</td><td>Insufficient text-search coverage</td><td>9</td><td>2</td><td>11</td><td>7.0</td><td>6.36</td><td>7.27</td><td>0.91</td></tr><tr><td>T3</td><td>Confirmation / anchoring loop</td><td>90</td><td>21</td><td>111</td><td>70.7</td><td>4.56</td><td>6.96</td><td>0.64</td></tr><tr><td>T4</td><td>Cross-lingual query dilution</td><td>3</td><td>1</td><td>4</td><td>2.5</td><td>4.50</td><td>7.00</td><td>0.75</td></tr><tr><td colspan="2">Non-tool failures</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>N1</td><td>Subjective reference or strict judging</td><td>13</td><td>2</td><td>15</td><td>9.6</td><td>4.13</td><td>4.47</td><td>0.53</td></tr><tr><td>N2</td><td>Other reasoning or execution errors</td><td>6</td><td>0</td><td>6</td><td>3.8</td><td>3.67</td><td>3.33</td><td>0.33</td></tr><tr><td>Total</td><td></td><td>129</td><td>28</td><td>157</td><td>100.0</td><td></td><td></td><td></td></tr></table>

BW: both wrong; Loss: Search hurts. Rounds, Search, and Img. are mean counts per run; Img. denotes image search.

## 4.3 Image Ablation of Interleaved References

We assess the contribution of embedded images using probes derived from the 205 references that contain at least one image. A multimodal generator creates four probes per item: two image-critical probes, designed to require information from an image, and two text-answerable probes, designed to be answerable from the paragraph alone. A positive-control filter removes probes that the reader cannot answer with the full interleaved input, retaining 367 image-critical and 375 text-answerable probes.

The reader, Qwen3.5-397B-A17B-FP8, answers each probe with the same paragraph text under three conditions: image tags removed $( y _ { \mathrm { t e x t } } )$ , tags replaced by [image N] placeholders $( y _ { \mathrm { p h } } )$ , or images displayed inline $( y _ { \mathrm { f u l l } } )$ . The reader receives neither the original query image nor the probe reference answer. A textonly judge grades each response. We report 95% confidence intervals from 2000 bootstrap resamples.

Table 6: Reader accuracy on retained probes under image ablation (%; 95% bootstrap CIs).  
Table 7: Image-critical accuracy by intent (%). $\Delta _ { \mathrm { f - t } } \mathrm { : }$ full minus text (%).
<table><tr><td>Probe type</td><td>Cond.</td><td>Hits/N</td><td>Acc %</td><td>95% CI</td></tr><tr><td>Image-crit.</td><td>Ytext</td><td>76/367</td><td>20.7</td><td>[16.6, 25.1]</td></tr><tr><td>Image-crit.</td><td> $y _ { \mathrm { p h } }$ </td><td>115/367</td><td>31.3</td><td>[26.7, 36.2]</td></tr><tr><td>Image-crit.</td><td>Yfull</td><td>356/367</td><td>97.0</td><td>[95.4, 98.6]</td></tr><tr><td>Text-answ.</td><td>Ytext</td><td>369/375</td><td>98.4</td><td>[97.1, 99.5]</td></tr><tr><td>Text-answ.</td><td> $y _ { \mathrm { p h } }$ </td><td>367/375</td><td>97.9</td><td>[96.3, 99.2]</td></tr><tr><td>Text-answ.</td><td>Yfull</td><td>368/375</td><td>98.1</td><td>[96.5, 99.5]</td></tr></table>

<table><tr><td>Intent</td><td>N Ytext</td><td>Yph</td><td>Yfull</td><td> $\Delta _ { \mathrm { f - t } }$ </td></tr><tr><td>Authenticity</td><td>66 13.6</td><td>19.7</td><td>98.5</td><td>+84.8</td></tr><tr><td>Diagnosis</td><td>65 30.8</td><td>36.9</td><td>100.0</td><td>+69.2</td></tr><tr><td>Identification</td><td>53 22.6</td><td>30.2</td><td>98.1</td><td>+75.5</td></tr><tr><td>Object cond.</td><td>35</td><td>8.6 28.6</td><td>88.6</td><td>+80.0</td></tr><tr><td>Physiology</td><td>57</td><td>28.1 40.4</td><td>94.7</td><td>+66.7</td></tr><tr><td>Procedural</td><td>49</td><td>12.2 26.5</td><td>98.0</td><td>+85.7</td></tr><tr><td>Spoilage</td><td>42</td><td>23.8 38.1</td><td>97.6</td><td>+73.8</td></tr><tr><td>Overall</td><td>367</td><td>20.7 31.3</td><td>97.0</td><td>+76.3</td></tr></table>

Table 6 shows that image-critical accuracy rises from 20.7% with text alone to 97.0% with the full interleaved input, a gain of 76.3%. Placeholder markers raise accuracy to 31.3%, a gain of 10.6% over text alone; displaying the images adds a further 65.7%. Text-answerable accuracy remains between 97.9% and 98.4%, varying by at most 0.5% across conditions.

The full-minus-text accuracy gain is positive across all seven intents, ranging from +66.7 to +85.7% (Table 7). In the 281 probes corrected by $y _ { \mathrm { f u l l } } ,$ , the text-only responses report missing information that the images subsequently provide. These results indicate that embedded images supply useful information beyond the paragraph text on the selected image-critical probes.

## 5 Related Work

Multimodal QA benchmarks. Existing multimodal QA benchmarks evaluate factual knowledge, visual understanding, and information seeking through answer-level correctness. SimpleVQA [7] and FVQA [25] focus on factual visual question answering, while LiveVQA [8] examines the acquisition and use of up-todate visual knowledge. MM-BrowseComp [16], MMSearch [14], and MMSearch-Plus [23] extend evaluation to multimodal browsing and retrieval, with MMSearch-Plus emphasizing fine-grained visual cues and provenance verification. BC-VL [9] and VDR-Bench [29] further construct challenging visual search tasks through entity fuzzification and knowledge-graph-based query expansion, respectively, reducing opportunities for text-only shortcuts or simple image matching. These benchmarks advance the evaluation of visual reasoning and search, but their answer-level metrics do not directly assess the quality of a complete explanation supported by embedded images. NoteVQA complements this line of work with community-sourced visual questions and paired short and interleaved references, enabling both correctness and explanationquality evaluation.

Multimodal deep research. Recent work extends multimodal evaluation from factual answers to longform research reports. MMDR-Bench [12] comprises 140 expert-crafted tasks across 21 domains and evaluates multimodal understanding, citation grounding, and report synthesis using image–text task bundles. MiroEval [28] contains 100 tasks, including 30 multimodal tasks, constructed from real user query patterns and web trends; its evaluation covers synthesis quality, factuality, and the research process. Complementing these benchmarks, PTAH [30] coordinates planning, research, writing, and verification agents to generate interleaved reports, with additional evaluation of image content and multimodal presentation quality. These efforts primarily target extended research workflows and report generation. NoteVQA focuses on everyday visual questions posted in online communities and provides human-audited interleaved references alongside short answers, connecting factual QA evaluation with the quality of visually supported responses.

Rubric design and model-based evaluation. Multimodal report evaluation increasingly considers both textual claims and their visual support. MMDR-Bench [12] evaluates visual evidence fidelity and the consistency between image-referenced claims and their associated visual content. PTAH [30] assesses image content and rendered presentation through dimensions including cross-modal alignment, information complementarity, and density-legibility balance. Building on these directions, IVR-12 evaluates interleaved answers through 12 dimensions grouped into content, presentation, and image quality. Its image dimensions separately assess relevance, necessity and placement, visual quality, and non-redundancy, while its content and presentation dimensions capture whether an answer is accurate, useful, and readable. We apply the same rubric to human-audited references and model outputs, providing a common scale for comparing interleaved-answer quality on real-world visual questions.

## 6 Conclusion

We introduce NoteVQA, a benchmark of 252 community-sourced visual questions that connects real user needs with evaluations of answer correctness and visually grounded explanation quality. Each question is paired with a short reference and a human-audited interleaved reference, complemented by AgenticInterleave for answer generation and IVR-12 for evaluation. Across 10 frontier VLMs, the highest short-answer accuracy is only 52.8%, highlighting the substantial challenge posed by everyday visual questions. Agentic retrieval improves Qwen3.5-397B-A17B by just 2.0 percentage points, while its interleaved answers score 3.52 under IVR-12 compared with 4.65 for the references, with the largest gap in content quality. In the image-ablation study, including embedded images raises reader accuracy on image-critical probes from 20.7% to 97.0%, while performance on text-answerable controls remains stable. This demonstrates that the images contribute information absent from the accompanying text. Together, these findings highlight the need for VLMs that can both answer real-world visual questions accurately and communicate the supporting evidence effectively. We will release the benchmark, evaluation prompts, model outputs, and code for the generation framework and data construction pipeline.

## Contributors

Haonan Jiang, Guojian Zhan, Jiancong Xie, Shijun Wan, Dongiia Zhao, Cheng Chen, Yahui Liu<sup>a</sup>, Yao Hu, Chuan Mu<sup>a</sup>

## References

[1] Lada A. Adamic, Jun Zhang, Eytan Bakshy, and Mark S. Ackerman. Knowledge sharing and Yahoo Answers: Everyone knows something. In Proceedings of the 17th International Conference on World Wide Web, pp. 665–674. Association for Computing Machinery, 2008. URL https://hdl.handle.net/2027. 42/58015.

[2] Alibaba Tongyi Qwen. Qwen3.8-flash-next: A new architecture, towards ultimate cost-efficiency. https://qwen.ai/blog?id=qwen3.8-flash-next. Accessed: 2026-08-26.

[3] Anthropic. Introducing claude opus 4.7. https://www.anthropic.com/news/claude-opus-4-7, . Accessed: 2026-08-16.

[4] Anthropic. Introducing claude sonnet 4.6. https://www.anthropic.com/news/claude-sonnet-4-6, . Accessed: 2026-08-20.

[5] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[6] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 14948– 14968. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.925. URL https://aclanthology.org/2023.emnlp-main.925/.

[7] Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[8] Mingyang Fu, Yuyang Peng, Benlin Liu, Yao Wan, and Dongping Chen. Livevqa: Live visual knowledge seeking. arXiv preprint arXiv:2504.05288, 2025.

[9] Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Kuan Li, Yida Zhao, et al. Webwatcher: Breaking new frontiers of vision-language deep research agent. In International Conference on Learning Representations (ICLR), 2026.

[10] Google. Gemini 3.1 pro. https://blog.google/innovation-and-ai/models-and-research/ gemini-models/gemini-3-1-pro/. Accessed: 2026-08-10.

[11] Hang He, Chuhuai Yue, Chengqi Dong, Chengcheng Wan, Ting Su, Haiying Sun, Jiajun Chai, Xiaohan Wang, and Guojun Yin. Vistahop: Benchmarking multi-hop visual reasoning for visual deepsearch. arXiv preprint arXiv:2606.03273, 2026.

[12] Peizhou Huang, Zixuan Zhong, Zhongwei Wan, Donghao Zhou, Samiul Alam, Xin Wang, Zexin Li, Zhihao Dou, Li Zhu, Jing Xiong, et al. Mmdeepresearch-bench: A benchmark for multimodal deep research agents. arXiv preprint arXiv:2601.12346, 2026.

[13] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[14] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Chaoyou Fu, Guanglu Song, et al. Mmsearch: Benchmarking the potential of large models as multimodal search engines. arXiv preprint arXiv:2409.12959, 2024.

[15] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[16] Shilong Li, Xingyuan Bu, Wenjie Wang, Jiaheng Liu, Jun Dong, Haoyang He, Hao Lu, Haozhe Zhang, Chenchen Jing, Zhen Li, et al. Mm-browsecomp: A comprehensive benchmark for multimodal browsing agents. arXiv preprint arXiv:2508.13186, 2025.

[17] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[18] Moonshot AI. Kimi k2.6: Advancing open-source coding. https://www.kimi.com/blog/kimi-k2-6. Accessed: 2026-08-10.

[19] Meredith Ringel Morris, Jaime Teevan, and Katrina Panovich. What do people ask their social networks, and why? a survey study of status message Q&A behavior. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, 2010. URL https://people.csail.mit.edu/teevan/work/publications/papers/chi10-social.pdf.

[20] Perplexity AI. Perplexity. https://www.perplexity.ai/. Accessed: 2026-08-10.

[21] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.

[22] Bytedance Seed. Seed2. 0 model card: Towards intelligence frontier for real-world complexity. arXiv preprint arXiv:2607.00248, 2026.

[23] Xijia Tao, Teng Yihua, Xinxing Su, Xinyu Fu, Jihao Wu, Chaofan Tao, Ziru Liu, Haoli Bai, Rui Liu, and Lingpeng Kong. Mmsearch-plus: Benchmarking provenance-aware search for multimodal browsing agents. In International Conference on Learning Representations (ICLR), 2026.

[24] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

[25] Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. Fvqa: Fact-based visual question answering. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 40 (10):2413–2427, 2017.

[26] Xiaohongshu. dots3-note preview: A small but mighty step toward long-horizon agency in real life. https://studio.dots.ai/dots/dots3-en.html. Accessed: 2026-08-13.

[27] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[28] Fangda Ye, Yuxin Hu, Pengxiang Zhu, Yibo Li, Ziqi Jin, Yao Xiao, Yibo Wang, Lei Wang, Zhen Zhang, Lu Wang, et al. Miroeval: Benchmarking multimodal deep research agents in process and outcome. arXiv preprint arXiv:2603.28407, 2026.

[29] Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, et al. Vision-deepresearch benchmark: Rethinking visual and textual search for multimodal large language models. arXiv preprint arXiv:2602.02185, 2026.

[30] Chenghao Zhang, Guanting Dong, Yufan Liu, Tong Zhao, Xiaoxi Li, and Zhicheng Dou. Towards verifiable multimodal deep research: A multi-agent harness for interleaved report generation. arXiv preprint arXiv:2605.29861, 2026.

[31] Zhipu AI. Glm-5.3-flash: Frontier intelligence, flash cost. https://z.ai/blog/glm-5.3-flash. Accessed: 2026-08-26.

## A Full Benchmark Statistics

Table 8: Full statistics of NoteVQA on the Stage-11 human-reviewed set.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of items N</td><td>252</td></tr><tr><td>Categories</td><td>12</td></tr><tr><td>Intents</td><td>7</td></tr><tr><td>Query length in words, mean ± std, median</td><td> $2 2 . 7 \pm 1 1 . 0 , 2 2 . 0$ </td></tr><tr><td>Short-answer length in words, mean ± std, median</td><td> $1 0 7 . 5 \pm 4 0 . 2 ,$  102.5</td></tr><tr><td>Interleaved answer length in words, mean ± std, median Total embedded images</td><td> $2 1 9 . 4 \pm 4 7 . 8 , 2 1 5 . 0$ </td></tr><tr><td>Images per item, mean ± std, median</td><td>378</td></tr><tr><td>Embed-count distribution over 0, 1, 2, 3, and 4+ images</td><td> $1 . 5 \pm 1 . 1 , 1 . 0$   $4 7 / 8 9 / 7 4 / 3 2 / 1 0$ </td></tr></table>

Table 9: Category and intent distribution matrix. A dot marks an empty cell.
<table><tr><td>Category Intent</td><td>Diagnosis</td><td>Authenticity</td><td>Physiology</td><td>Identification</td><td>Procedural</td><td>Object Condition</td><td>Spoilage</td><td>Total</td></tr><tr><td>HomePet</td><td>17</td><td>3</td><td>16</td><td>4</td><td>3</td><td>3</td><td>4</td><td>50</td></tr><tr><td>Exotic</td><td>6</td><td>3</td><td>9</td><td>19</td><td>2</td><td>1</td><td>1</td><td>41</td></tr><tr><td>Plant</td><td>8</td><td>3</td><td>6</td><td>6</td><td>10</td><td></td><td>1</td><td>34</td></tr><tr><td>Health</td><td>14</td><td>5</td><td>8</td><td></td><td>1</td><td></td><td></td><td>28</td></tr><tr><td>Fashion</td><td>2</td><td>8</td><td></td><td></td><td>1</td><td>6</td><td></td><td>17</td></tr><tr><td>Jewelry</td><td></td><td>12</td><td></td><td>1</td><td></td><td>2</td><td>1</td><td>16</td></tr><tr><td>Food</td><td></td><td></td><td></td><td>1</td><td></td><td></td><td>14</td><td>15</td></tr><tr><td>Appliance</td><td></td><td>3</td><td></td><td></td><td>4</td><td>8</td><td>•</td><td>15</td></tr><tr><td>BioLab</td><td>3</td><td></td><td>4</td><td></td><td>1</td><td>1</td><td>2</td><td>11</td></tr><tr><td>Home</td><td></td><td>3</td><td></td><td>1</td><td>1</td><td>3</td><td>1</td><td>9</td></tr><tr><td>CE</td><td></td><td>4</td><td></td><td></td><td>3</td><td>2</td><td></td><td>9</td></tr><tr><td>Other</td><td></td><td></td><td></td><td>4</td><td>3</td><td></td><td></td><td>7</td></tr><tr><td>Total</td><td>50</td><td>44</td><td>43</td><td>36</td><td>29</td><td>26</td><td>24</td><td>252</td></tr></table>

## B Intent Accuracy Full Grid

Table 10: Shortanswer accuracy across 7 intents. All models evaluated in thinking mode. Rows marked ∗ use the fourtool agentic retrieval suite of Section 3.3; others are singlepass notool.
<table><tr><td>Model</td><td>Overall</td><td>Diagnosis</td><td>Authenticity</td><td>Physiology</td><td>Identification</td><td>Procedural</td><td>Object Condition</td><td>Spoilage</td></tr><tr><td>Gemini-3.1-Pro</td><td>52.8</td><td>60.0</td><td>38.6</td><td>67.4</td><td>27.8</td><td>65.5</td><td>57.7</td><td>54.2</td></tr><tr><td>Doubao-seed-2.0-pro</td><td>49.6</td><td>66.0</td><td>29.5</td><td>65.1</td><td>30.6</td><td>62.1</td><td>50.0</td><td>37.5</td></tr><tr><td>Dots-note-preview</td><td>48.4</td><td>54.0</td><td>31.8</td><td>67.4</td><td>33.3</td><td>72.4</td><td>42.3</td><td>33.3</td></tr><tr><td>GLM-5.3-Flash</td><td>48.0</td><td>52.0</td><td>27.3</td><td>55.8</td><td>36.1</td><td>86.2</td><td>30.8</td><td>54.2</td></tr><tr><td>Kimi-K2.6</td><td>47.6</td><td>52.0</td><td>36.4</td><td>51.2</td><td>33.3</td><td>62.1</td><td>57.7</td><td>45.8</td></tr><tr><td>Qwen3.8-Flash</td><td>44.0</td><td>60.0</td><td>31.8</td><td>55.8</td><td>44.4</td><td>51.7</td><td>23.1</td><td>25.0</td></tr><tr><td>Claude-4.7-Opus</td><td>43.7</td><td>58.0</td><td>29.5</td><td>48.8</td><td>25.0</td><td>62.1</td><td>42.3</td><td>37.5</td></tr><tr><td>Kimi-K3</td><td>43.3</td><td>56.0</td><td>27.3</td><td>46.5</td><td>25.0</td><td>55.2</td><td>50.0</td><td>45.8</td></tr><tr><td>Claude-4.6-Sonnet</td><td>35.3</td><td>44.0</td><td>20.5</td><td>34.9</td><td>27.8</td><td>48.3</td><td>34.6</td><td>41.7</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>35.7</td><td>40.0</td><td>27.3</td><td>37.2</td><td>25.0</td><td>51.7</td><td>30.8</td><td>41.7</td></tr><tr><td>Qwen3.5-397B-A17B*</td><td>37.7</td><td>46.0</td><td>31.8</td><td>32.6</td><td>27.8</td><td>37.9</td><td>38.5</td><td>54.2</td></tr></table>

Excluding the agentic row, Doubao-seed-2.0-pro leads Diagnosis at 66.0%, Gemini-3.1-Pro leads Authenticity at 38.6%, Qwen3.8-Flash leads Identification at 44.4%, and GLM-5.3-Flash leads Procedural at 86.2%. Gemini-3.1-Pro and Dots-note-preview tie for the highest Physiology accuracy at 67.4%; Gemini-3.1-Pro and Kimi-K2.6 tie for Object Condition at 57.7%; and Gemini-3.1-Pro and GLM-5.3-Flash tie for Spoilage at 54.2%. Across all reported configurations, Authenticity is the only intent whose best accuracy remains below 40%, at 38.6%.

## C Interleaved Multimodal Summary Construction Pipeline Details

This appendix details the pipeline behind the human-verified interleaved multimodal summary described in Section 3.2 and Figure 4: how the per-item image evidence pool is built from an agent trajectory, and how that pool is then synthesised into an interleaved answer and cleaned. For each stage we report the prompts, thresholds, and model-call parameters.

## C.1 Evidence Pool Construction

Contributing URL Ranking. A rule-based pre-pass processes each agent episode and emits one entry per tool result, covering three tool types: search results, image search results with the returned image URL preserved, and visited pages where each visited URL becomes its own entry split by the boundary pattern “The useful information in url for user goal . . . as follows:”. The complete URL list for a task, aggregated across all three tool types, is then packaged into a single text-only prompt whose fields are the question, the reference answer, and one [ID i] type=t | URL: u n c block per URL. Qwen scores each URL on relevance, completeness, clarity, answer-support, and necessity; the five scores are summed and the top-5 URLs per item are retained.

URL Contribution Scoring   
You are evaluating which web pages contributed most to correctly answering a research question.   
\*\*Question\*\*: {query}   
\*\*Expected Answer\*\*: {answer}   
Below are all web pages collected during research, each with an ID, type, URL, and content:   
{links\_block}   
---   
Score each page on 5 dimensions (1-5) and select the TOP {top\_k} most useful pages.   
Return ONLY valid JSON with the top {top\_k} page IDs. Be concise in the reason field.   
Output format:   
{   
"<id>": {   
"reason": "<one-sentence explanation>",   
"relevance\_score": <1-5>,   
"completeness\_score": <1-5>,   
"clarity\_score": <1-5>,   
"answer\_support\_score": <1-5>,   
"necessity\_score": <1-5>   
},   
}   
Scoring criteria:   
- relevance\_score (1-5): Is the page content directly relevant to the question?   
一 completeness\_score (1-5): Does the page contain effective information for answering?   
clarity\_score (1-5): Is the content clear, well-structured, and readable?   
answer\_support\_score (1-5): Does the page support or verify the expected answer?   
necessity\_score (1-5): Is the page essential (not just supplementary)?

Web-Page Image Crawling. Each of the top-5 URLs is fetched through Serper’s scrape endpoint with includeMarkdown=true; the returned og:image is prepended to the markdown so hero images are not lost. Candidate image URLs are then harvested with the union of the three regexes in Table 11, and any URL matching the case-insensitive ad/tracker blocklist in Table 12 is dropped before download. We attempt to download up to 60 candidate images per URL; we retain the first 20 whose shorter side is at least 100 pixels (MIN\_PX= 100, checked via PIL.Image.size). The download timeout per image is 10 s with a browser User-Agent and concurrency 32. Because the previous stage rarely selects an image\_search URL — its textual content is short — a parallel path additionally downloads every image\_url returned by image\_search calls (subject to the same 100 px floor and blocklist) into a separately-tracked image\_search\_pool, ensuring visually valuable results from reverse search are not discarded.

Table 11: Image-URL extraction patterns used on the scraped markdown. The three regexes are applied independently and their captured URL sets are unioned.  
Source Regex (URL captured in group 1)   
Markdown !\[[^\]]\*\]\((https?://[^\s)]+)\)   
HTML <img> <img[^>]+src=["\’]?(https?://[^\s"\’><]+) (case-insensitive)   
CSS background background(?:-image)?\s\*:\s\*url\(["\’]?(https?://[^\s"\’)<]+) (case-insensitive)

Table 12: Ad/tracker/icon blocklist. A candidate image URL is dropped before download if it matches any of these substrings or path patterns (single case-insensitive regex; alternation shown here row-wise for readability).  
Category Substring / pattern   
Ads /ads?/, /banner, advertisement, doubleclick, googletagmanager   
Tracking pixels /pixel[./], /tracking, /beacon, 1x1, analytics, \.gif\?, facebook.com/tr   
Avatars / logos gravatar.com, /icon[s]?/, /logo.(png|svg|ico)\$, /favicon   
Chrome /sprite, /placeholder

```markdown
Candidate-Image Scoring
You are an expert evaluator assessing the multi-dimensional contribution of images to answering
questions.
## Input Information
Two images are provided above in order:
1. **Query Image** (first image): the original image attached to the question.
2. **Web Image** (second image): an image crawled from a contributing web page.
**Question:** {question}
**Expected Answer:** {answer}
## Evaluation Task
Evaluate how the **Web Image** (second image) contributes to answering the question, taking the
Query Image as context.
1. **Visual Relevance** (1-5): Is the Web Image directly related to the question?
2. **Information Content** (1-5): Does the Web Image contain useful information to answer?
3. **Clarity** (1-5): Is the Web Image clear, readable, and well-presented?
4. **Answer Substantiation** (1-5): Does it help verify or clarify the expected answer?
5. **Necessity** (1-5): Is the Web Image essential or highly helpful vs optional?
## Output Format (JSON only)
Return ONLY valid JSON with no additional text:
{
"visual_relevance": {"score": <1-5>, "reason": "<brief explanation>"},
"information_content": {"score": <1-5>, "reason": "<brief explanation>"},
"clarity": {"score": <1-5>, "reason": "<brief explanation>"},
"answer_substantiation": {"score": <1-5>, "reason": "<brief explanation>"},
"necessity": {"score": <1-5>, "reason": "<brief explanation>"},
"overall_score": <1-5>
}
```

Candidate-Image Scoring. Every image in best\_link[\*].images image\_search\_pool is scored by a two-image VLM call: the original query image is passed first, then the candidate. The two pools are then merged and stable-sorted by overall\_score in descending order; the first 10 are retained, with their provenance recorded via a source tag ( {best\_link, image\_search\_pool}) so the two paths remain distinguishable downstream.

## C.2 Interleaved Multimodal Summary Synthesis

Multimodal Summary Generation. A single multimodal Qwen call receives, in order, the query image, up to ten candidate images (indexed Image 1..N), and a text block containing the question, the reference answer, a formatted trajectory (each step’s observation is capped at 3000 characters), and the scores for each candidate laid out as in Table 13. Tags emitted by the model must match the grammar <image>{"candidate\_images": N}</image>; after generation, a regex extracts every referenced index and candidate\_images is filtered down to those actually used. If no candidate image survived scoring for a given item, the generator receives a placeholder.

MMSummary Generator (System Prompt)   
You are an expert answer writer producing image-text interleaved (MMSummary) answers.   
You will receive:   
- a user QUESTION (with its query image, shown first),   
- the CORRECT reference ANSWER,   
- BACKGROUND research findings (tool call trajectory: search/visit/image\_search results),   
- a set of CANDIDATE IMAGES (each shown in order with an index Image 1, Image 2, ...).   
Write a rich, detailed English answer that satisfies the following:   
Style rules:   
- Lead directly with the answer -- do NOT begin with "Based on..." / "According to..." / "The   
research shows...".   
- Expand with specific facts, names, numbers, and context drawn from the background research.   
- Write with authority, as if you already know the topic well.   
Image embedding rules:   
- Embed candidate images ONLY when they genuinely illustrate the neighbouring sentence.   
- Prefer images with higher overall\_score and stronger visual relevance to the sentence you just   
wrote.   
- Insert tags in exactly this JSON format: <image>{"candidate\_images": N}</image>   
where N is the 1-based Image index shown to you. Place the tag on its own line immediately   
after the sentence it illustrates.   
- Do NOT invent image indices -- only use indices from the candidate images actually provided.   
- If NO candidate image is a good fit, write a pure text answer without any <image> tags.   
- Typical density: 1-4 image tags for a 200-400 word answer. Do not force insertion.   
Return ONLY the answer text (with any embedded image tags). No preamble, no metadata.

MMSummary Generator (User Prompt Template)   
## Question (Query image is the FIRST image shown above)   
{question}   
## Correct Reference Answer   
{answer}   
## Background Research (agent trajectory)   
{trajectory\_text}   
## Candidate Images (shown AFTER the query image, in the order Image 1, Image 2, ...)   
{format\_candidate\_scores(candidate\_images)}   
Now write the image-text interleaved MMSummary.

Table 13: Per-candidate score block passed to the generator (one such block for each of the up-to-10 images).  
Line Content   
1 Image {idx} (source={src}) reference tag: <image>{"candidate\_images": {idx}}</image>   
2 visual\_relevance: {score} – {reason}   
information\_content: {score} – {reason}   
clarity: {score} – {reason}   
answer\_substantiation: {score} – {reason}   
6 necessity: {score} – {reason}   
7 overall\_score: {score}

Five-Rule Filter and Cross-Image Deduplication. Each embedded tag is re-examined by a two-image VLM call (query image + candidate) that decides whether to remove the candidate image using five rules — REDUNDANT-WITH-ORIGINAL, COMPOSITE/COLLAGE, GLOBAL-WATERMARK, VIDEO-THUMBNAIL (a playbutton overlay), and OFF-TOPIC. A 200-character window around the tag is passed as context (truncated to 300 characters); an API failure defaults to keep and an image that cannot be loaded defaults to remove. Images that survive the five-rule filter are then compared pairwise against every previously retained image via a single-question VLM probe that returns {"similar": true} to trigger deduplication, so near-duplicates within a summary collapse to one representative image.

Five-Rule Image Filter   
You are evaluating a single candidate image for quality in a multimodal summary.   
Images above:   
- Image 1: Original query image (the question's reference image)   
- Image 2: Candidate image being evaluated   
Question: {query}   
Expected Answer: {answer}   
Candidate image context in summary: "{context}"   
Evaluate whether this candidate image should be REMOVED. Remove it if ANY of these apply:   
1. REDUNDANT WITH ORIGINAL: Shows the same subject, person, or composition as the original query   
image with no meaningful new visual information -- even if not pixel-identical.   
2. COMPOSITE/COLLAGE: Contains multiple unrelated subjects or scenes stitched together in one   
frame (e.g., movie scenes side by side, a photo grid, a collage).   
3. GLOBAL WATERMARK: Has a prominent watermark, logo, or text overlay covering a significant   
portion of the image (agency watermarks, stock photo stamps, etc.).   
4. VIDEO THUMBNAIL: Shows a video play button overlaid on the image, indicating it is a video   
screenshot/thumbnail rather than a standalone photo.   
5. OFF-TOPIC: Clearly irrelevant to the candidate's context in the mmsummary.   
Return ONLY valid JSON:   
{"remove": true, "reason": "<one sentence>"} or {"remove": false}   
Cross-Image Deduplication Probe   
Are these two images very similar -- showing essentially the same subject, person, or scene with   
only minor differences in angle, crop, or lighting? Return ONLY: {"similar": true} or {"   
similar": false}

## D AgenticInterleave System Prompt and Image-Tag Format

Tools. All four tools are asynchronous, retried three times on failure, and file-cached. search executes batched textual web search via Serper’s Google backend and returns the top-10 results per query. visit batch-fetches up to five web pages and invokes an extraction LLM that returns goal-relevant content as JSON. image\_search performs reverse image search via Serper’s Google Lens endpoint on the userprovided image and is limited to a single call per task, since reverse search is costly and easily disrupted by noise. text\_to\_image\_search is a forward text-to-image tool that queries Serper’s Google Images endpoint and returns a ranked list of candidates that the agent references by index in the final answer.

Image-tag format. The agent embeds each retrieved image in the final answer with a single-line tag placed at the end of the sentence that mentions the corresponding entity. Each tag records the exact search query that produced the image pool and the 1-based index of the chosen candidate, so that the tag can be resolved back to a specific tool call at scoring time:

<image>{"query": "exact\_search\_query ", "idx": N}</image>

The system prompt forbids emitting a tag whose query does not match a preceding text\_to\_image\_search call, and forbids an idx outside the range returned by that call. These two invariants are checked by a posthoc validator.

```markdown
AgenticInterleave System Prompt – Part I: Workflow and Core Rules
You are a multimodal deep research agent. Given a user question (which may include one or more
images), conduct thorough searches across various information sources, perform step-by-step
reasoning, and give accurate and concise answers.
## Workflow
Each turn, your response must follow this exact structure:
1. <think>...</think> - Analyze the question and any images, interpret tool results received so
far, and decide what to do next.
2. Then exactly one of:
- <tool_call>...</tool_call> - Call one tool if you need more information.
- <answer>...</answer> - Provide your final answer when you have enough information.
## Core Bu1es
- You must always begin with <think> reasoning.
- Call only ONE tool per turn.
- For questions about specific people, places, objects, artworks, or films, use search or
image_search first to gather information.
- Use visit to read full webpages when search results are insufficient.
If searches fail repeatedly, answer based on your knowledge and the provided images.
```

AgenticInterleave System Prompt – Part II: Image Embedding Rules   
## Image Embedding   
For any visual entity that is a central subject or key content in your answer, search for and   
embed its image.   
Principle: A visual entity has information gain if the reader benefits from seeing what it looks   
like to better understand, recognize, or contextualize your answer. Conversely, if the   
entity is purely functional or background information, it does not need an image.   
Mandatory Workflow:   
1. In the <think> section, identify every visual entity that will appear in the planned answer,   
and classify each as SUBJECT (central to the answer) or BACKGROUND (incidental reference).   
2. For each SUBJECT entity, ask whether it is a main topic or key component of the answer. If   
yes, it has information gain and MUST trigger a text\_to\_image\_search call.   
3. In the <answer> section, embed each retrieved image after the sentence that describes the   
corresponding SUBJECT entity, using the JSON tag <image>{"query": "EXACT\_SEARCH\_QUERY", "idx   
": IMAGE\_NUMBER}</image>.   
Placement Rules:   
Image tags go AFTER the sentence about the entity, never mid-sentence.   
- Each SUBJECT entity gets exactly one image.   
- The idx must correspond to an image actually returned by the tool response (1..5).   
- The query string in the tag must match the query used in the preceding text\_to\_image\_search   
call verbatim.   
MANDATORY: No image tag may appear in the answer without a preceding matching   
text\_to\_image\_search call.

```jsonl
AgenticInterleave System Prompt – Part III: Tools
## Tools
You are provided with the following tools within <tools></tools> XML tags:
<tools>
{"name": "search", "description": "Perform Google web text searches and return the top results.
Accepts multiple queries in a single call.", "parameters": {"query": {"type": "array", "
items": {"type": "string"}, "minItems": 1}}}
{"name": "image_search", "description": "Perform a reverse image search on the question image to
identify entities, retrieve similar images, and find related web pages. This tool should
only be used once.", "parameters": {"goal": {"type": "string"}}}
{"name": "text_to_image_search", "description": "Search for images based on text queries. Use
this to find visual representations of entities, concepts, or subjects mentioned in your
answer.", "parameters": {"query": {"type": "string"}}}
{"name": "visit", "description": "Visit one or more webpages and return a summary of their
content.", "parameters": {"url": {"type": "array", "items": {"type": "string"}}, "goal": {"
type": "string"}}}
</tools>
For each function call, return a JSON object within <tool_call></tool_call> tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
Current date: {today}
```  
Example trajectory. In Figure 6 – Figure 10, we show five consecutive parts of a single AgenticInterleave trajectory produced by Qwen3.5-397B-A17B for a mushroom query from NoteVQA. Together, the figures present the query image and user question, the reasoning and tool interactions, and the final interleaved answer. The query appears in the first part; subsequent parts continue the same trajectory.

![](images/e6579a4e3b42e2f1ad16fda78689df835e7c5ced86fdbcc4cef7e723d3a8518e.jpg)  
Figure 6: AgenticInterleave mushroom-query trajectory, part 1 of 5.

![](images/11b6089fee41fa10bde188ca91580498affed7278cc408ae95f1f69285b41bbb.jpg)  
Figure 7: AgenticInterleave mushroom-query trajectory, part 2 of 5.

![](images/8dae9aa788f95be349a24dee3ab0d8a9fba0224d4b3c3e3f617ddcde787bdf74.jpg)  
Figure 8: AgenticInterleave mushroom-query trajectory, part 3 of 5.

![](images/73a3e735c4e16b50c179329dadeded85abb956d171f39286c09286eaf211c166.jpg)  
Figure 9: AgenticInterleave mushroom-query trajectory, part 4 of 5.

![](images/934b7cd7b9ccbc3c4b22ac9c4fca1de2155af0620c43ba46b9ef0fca5aa9963b.jpg)  
Figure 10: AgenticInterleave mushroom-query trajectory, part 5 of 5.

## E IVR-12 Grader Prompts and Aggregation Logic

This appendix specifies the four judge prompts and the per-item aggregation rule used by the IVR-12 grader when scoring both model-produced interleaved answers and the human-audited gold under the same protocol. Each item triggers four sequential judge calls at temperature 0.0; the specific judge models are those introduced in Section 4. The content and presentation calls are text-only and see the answer with all <image> tags stripped. The image call is multimodal and receives, in order, the query image followed by every embedded image in the order they appear in the answer; the answer text is passed with tags retained so the judge can assess image placement. The gates call is multimodal and receives the query image alongside the stripped answer text. Every prompt asks for a single JSON object as output; the parser tolerates fenced code blocks and trailing text and drops the item only on unrecoverable parse failure.

Per-category means are the arithmetic mean of numeric values in [1, 5]; out-of-range or non-numeric fields are dropped rather than defaulted to a mid-scale value. If the answer contains zero embedded images the image judge is instructed to return 1 on all four dimensions, so the image mean cannot be quietly inflated by an empty answer. The IVR score is

$$
S _ { \mathrm { I V R - 1 2 } } = \left\{ \begin{array} { l l } { { 0 . 5 0 \bar { C } + 0 . 2 0 \bar { P } + 0 . 3 0 \bar { I } , } } & { { \mathrm { i f ~ a l l ~ t h r e e ~ g a t e s ~ p a s s , } } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.
$$

if every gate returns PASS, and 0 otherwise. Missing gate keys are treated as PASS: only an explicit FAIL zeroes the score, which keeps the grader robust to occasional truncated JSON without silently zeroing wellformed answers.

IVR-12 Content Judge Prompt (5 dimensions, text-only)   
You are evaluating a multimodal answer against a reference on 5 CONTENT dimensions.   
Score each dimension on a 1-5 scale where:   
5 = excellent, 4 = good, 3 = acceptable, 2 = weak, 1 = poor.   
Dimensions:   
1. intent : Does the answer address the actual user intent behind the question?   
2. accuracy : Are the factual claims correct and consistent with the reference?   
3. source\_fidelity : Are claims traceable to plausible sources (not hallucinated details)?   
4. completeness : Does the answer cover the key aspects a user needs to act on?   
5. neutrality\_safety: Is the tone neutral, and are safety-relevant claims appropriately hedged?   
QUESTION:   
{query}   
REFERENCE ANSWER (human-verified gold):   
{gold\_short}   
ANSWER TO EVALUATE (text only, image tags stripped):   
{answer\_text}   
Return ONLY valid JSON of the form:   
{"intent": <int>, "accuracy": <int>, "source\_fidelity": <int>, "completeness": <int>, "   
neutrality\_safety": <int>, "reason": "<one short sentence>"}

IVR-12 Presentation Judge Prompt (3 dimensions, text-only)   
You are evaluating a multimodal answer on 3 PRESENTATION dimensions.   
Score each dimension on a 1-5 scale where:   
5 = excellent, 4 = good, 3 = acceptable, 2 = weak, 1 = poor.   
Dimensions:   
1. reading\_experience: Does the answer read fluently and coherently?   
2. formatting : Are paragraph breaks, lists, bolding, and image placement used   
appropriately?   
3. conciseness : Is the answer appropriately concise (not padded, not too terse)?   
QUESTION:   
{query}   
ANSWER TO EVALUATE:   
{answer\_text}   
Return ONLY valid JSON of the form:   
{"reading\_experience": <int>, "formatting": <int>, "conciseness": <int>, "reason": "<one short   
sentence>"}

## IVR-12 Image Judge Prompt (4 dimensions, multimodal)

You are evaluating a multimodal answer on 4 IMAGE dimensions.   
Above are, in order:   
- Image 1: the ORIGINAL query image (what the user asked about)   
Images 2..N: the EMBEDDED images in the answer, in order of appearance   
Score each dimension on a 1-5 scale (5 = excellent):   
1. relevance : Do the embedded images depict the entities/situations they are meant to   
illustrate?   
2. necessity\_position: Are the images placed where they add information gain (SUBJECT entities,   
not decoration)?   
3. quality : Are the images visually clear, well-cropped, and not obvious low-quality   
scrapes?   
4. non\_redundancy : Do the images each add distinct visual information (no near-duplicates)?   
If the answer has ZERO embedded images, score all 4 dimensions as 1 (no visual value delivered).   
QUESTION:   
{query}   
ANSWER TO EVALUATE (with <image> tags marking positions):   
{answer\_full}   
Return ONLY valid JSON of the form:   
{"relevance": <int>, "necessity\_position": <int>, "quality": <int>, "non\_redundancy": <int>, "   
n\_embedded": <int>, "reason": "<one short sentence>"}

IVR-12 Gates Judge Prompt (3 critical PASS/FAIL gates, multimodal)   
You are evaluating a multimodal answer on 3 CRITICAL PASS/FAIL gates.   
Above is the ORIGINAL query image.   
Gates (each PASS or FAIL):   
1. timeliness : Does the answer avoid claims that are obviously stale or contradicted by   
present-day facts?   
(FAIL only if the answer makes a clearly outdated claim relevant to the   
query.)   
2. safety\_redlines : Does the answer avoid dangerous advice (e.g., unsafe medical procedures,   
illegal activity, harmful chemical mixing)?   
3. refusal\_correctness: Does the answer NOT refuse an answerable question? (FAIL means the model   
wrongly refused.)   
(If the question genuinely warrants refusal, e.g. asks for illegal help,   
refusing is correct -> PASS.)   
QUESTION:   
{query}   
ANSWER TO EVALUATE:   
{answer\_text}   
Return ONLY valid JSON of the form:   
{"timeliness": <PASS or FAIL>, "safety\_redlines": <PASS or FAIL>, "refusal\_correctness": <PASS   
or FAIL>, "reason": "<one short sentence>"}

## F Additional Interleaved Reference Answer Cases

As shown in Figure 11 – Figure 13, we present six human-reviewed interleaved reference answers drawn from different category and intent combinations. Each figure contains two cases side by side. Each case includes its own query image, question, and interleaved reference answer.

![](images/31b0e44d21ff48345a30aa95d8643c6908d87a81b26b3e7bd033c754739a7e98.jpg)  
Figure 11: Case Gallery 1: Two human-reviewed interleaved reference answers.

![](images/c881c522521d8988377eb354692a9810f4873c322ea7fe96f62130ddf2e68c94.jpg)  
The elder insists the baby has flat feet. But I remember that before 3 years old it's all pseudo. Does this count? Please have a look if you know.

Answer:   
The flat appearance of your baby's feet in the photo is completely normal and is known as physiological or "pseudo" flat feet, not a true deformity. In infants and toddlers under 3 years old, the arch of the foot is typically hidden by a thick, protective fat pad that fills the space where the arch will eventually form.   
Additionally, the ligaments and muscles in the feet are still developing and are naturally more flexible at this age, which contributes to the flat look when standing or bearing weight. This is a standard part of growth; as your child becomes more active, walks, and runs, that fat pad gradually dissipates, and the arch muscles strengthen.

![](images/d5c59d175bb9172c88ce6dcb53412ddb3ceb65ef1dce8df01d3048c83bbc7228.jpg)  
Found these two kinds of small bugs on several pots of plants. How to distinguish which is friendly and which is enemy?

Most children develop a visible arch naturally between the ages of 3 and 6.

![](images/9a7ce42756dc3110eced3428476b0785afcd3581f6bbbb3221f13017fcf7f18a.jpg)  
Answer: The bright red speck on your basil stem is a beneficial predator known as the Whirligig Mite (\*Anystis baccarum\*), a valuable ally in your garden that actively hunts harmful pests.

![](images/84f503063f77a2597b9888c22db2fed2bcb88581c7f3bf3bea72f80094ae9012.jpg)  
Unlike the destructive spider mites that plague basil, this friendly mite is significantly larger, measuring between 1.0 and 1.5 mm, making it clearly visible to the naked eye as a distinct red dot.

![](images/2552998964e68fc2dd8f543137ac610b836a8527455262c54a943021b4319da9.jpg)  
Figure 12: Case Gallery 2: Two human-reviewed interleaved reference answers.

You can distinguish it from the enemy by its behavior and size; the \*Anystis baccarum\* moves with incredible speed in erratic, zigzagging patterns, whereas harmful spider mites are microscopic (around 0.4 mm), move slowly, and often cluster together under fine silk webbing.

![](images/95cb1cd94ae7781dfca22499a2bd5a477e0ac0fa0ba81ba8fd007a910f884fa4.jpg)  
Figure 13: Case Gallery 3: Two human-reviewed interleaved reference answers.