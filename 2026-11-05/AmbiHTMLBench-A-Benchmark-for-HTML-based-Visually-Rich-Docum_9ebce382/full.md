# AmbiHTMLBench: A Benchmark for HTML-based Visually Rich Document parsing

Boyu Niu<sup>1,2,3</sup> Yingjie Xiong<sup>1,2</sup> Shurui Li<sup>1,2</sup> Linhan Pang<sup>1,2</sup> Yuntong Zhang<sup>1,2</sup> Han Wang<sup>3</sup> Hao Wu<sup>3</sup> Kai Zuo<sup>3</sup> Xu Tang<sup>3</sup> Xuanhe Zhou<sup>1,2∗</sup> Fan Wu<sup>1,2</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Theseus Labs <sup>3</sup>Xiaohongshu  Homepage  GitHub / Hugging Face

## Abstract

Traditional document parsing systems use Markdown as their output representation, which is efective for linear text but does not fully preserve fine-grained spatial structure, table geometry, visual hierarchy, or rendering semantics. We show that HTML is advantageous for heterogeneous and layout-sensitive documents whose representations must remain editable, inspectable, and reusable, and introduce AmbiHTMLBench to systematically study this setting. In this benchmark, given a real-world document, a model must recover executable HTML that preserves its content, organization, and visual appearance. The benchmark contains 1,115 diverse pages with structured annotations and evaluates predictions in two diferent ways: The objective protocol aligns ground-truth and predicted rendered blocks and measures table fidelity, page-level text error, and a two-dimensional layout consistency; a complementary reference-anchored protocol measures perceived fidelity by aligning LLM judges with human preferences. Across ten primary multimodal systems, the best objective score is 0.8092, while the subjective winner is diferent, demonstrating that automatic decomposition and human-perceived fidelity capture complementary properties. We further fine-tune a small-parameter expert model and find that task-specific adaptation yields good performance on document-to-HTML reconstruction task. These results position AmbiHTMLBench as a diagnostic testbed for building smarter and more logical document parsing systems.

![](images/76c2af9987b6658300408585db57be5d99680a29841b915ac0adb6a5020ba7f1.jpg)  
Figure 1 From document images to reusable HTML. Complete pages from realistic genres (top) are reconstructed as HTML and rendered in browser (middle), more cases are shown in appendix C. The bottom row lists potential uses of the generated content.

## 1 Introduction

![](images/28f90cf3270776ca364ff04c08f0a32f2dc1712b3de0bfe4d1e83355633546fe.jpg)  
Figure 2 AmbiHTMLBench focuses on three level of fidelity in HTML based document parsing, with both objective and reference-anchored subjective protocols.

Motivation. Documents are not merely sequences of words: headings, tables, figures, typography, and spatial arrangement jointly determine how information is interpreted. Recent multimodal systems can extract and reason over document images [1–3], but document agents require a stronger capability. They must be able to store evidence, locate the source of an answer, modify a document, and reuse the same representation across retrieval, reasoning, and interaction tasks [4]. Understanding a page once is therefore insuficient; agents need a persistent and actionable representation of the page itself.

Existing representations expose only part of this information. Plain text or Markdown supports eficient reading but discards hierarchy, spatial relations, and visual emphasis. JSON-like schemas can encode selected semantics, yet are usually designed for one downstream task and rarely preserve presentation. Raster images retain appearance but provide neither editable semantic units nor stable machine-addressable references. These limitations force downstream agents to repeatedly infer structure from pixels, creating a gap between document perception and document interaction.

HTML provides a natural representation for closing this gap. Its Document Object Model (DOM) expresses hierarchical and addressable relationships among headings, paragraphs, tables, figures, and captions, allowing humans and agents to locate and manipulate individual elements [5, 6]. At the same time, HTML and CSS provide an executable rendering mechanism for alignment, spacing, typography, positioning, and table geometry. Semantic elements and their visual realization remain connected rather than being separated into an image and an unrelated text transcript. A faithful document-to-HTML system should thus produce an artifact that is simultaneously readable, editable, inspectable, and renderable.

We also obtain direct evidence for this interface property in two small pilots. On six field groups from a UKSwap form, HTML/CSS yields 62.03% pair-joint accuracy for field-to-slot binding, compared with 44.44% for geometry-matched non-HTML records. On PubTabNet tables, HTML-based evidence summaries make row, column, and value nodes explicitly addressable, enabling more reliable correction of an erroneous row assignment than linearized Markdown. These pilots are not part of the benchmark leaderboard; they motivate evaluating whether models can reconstruct the representation that agents can subsequently inspect and act upon.

![](images/d26c12f94bd32e0f0fa2a14d362e333ac15aa9f295cad57f2fbc703ca214a40d.jpg)  
Figure 3 Pilot evidence that HTML representations support spatial binding and table verification.

The missing capability. Despite progress in OCR and document parsing, existing benchmarks largely measure isolated components such as text, tables, formulas, and reading order [7–9]. Screenshot-to-code benchmarks study visual reconstruction, but primarily target webpages with relatively regular DOM-oriented layouts. Neither setting directly tests whether a multimodal model can recover a heterogeneous real-world document as complete, executable HTML. This task is dificult because fidelity has several interacting dimensions: a prediction may contain the correct words but destroy columns or table geometry; it may preserve layout while hallucinating or omitting content; or it may encode a valid DOM whose CSS fails to reproduce the visual hierarchy that guides human reading.

These failure modes also motivate complementary evaluation. Content fidelity should measure whether text, numbers, formulas, and table cells are preserved. Structural fidelity should measure whether blocks and their relationships are organized as in the source. Visual or perceived fidelity should account for the fact that humans may tolerate harmless line-break diferences but strongly penalize lost emphasis, displaced media, or broken table structure. A single metric cannot reliably expose all of these cases, so a useful benchmark should report decomposed automatic signals together with a reference-anchored perceptual measure.

Our benchmark. We present AmbiHTMLBench, a benchmark of 1,115 diverse pages with expert annotations. We render each generated HTML page and align its semantic blocks with the ground truth to measure page-level text error, table fidelity, and two-dimensional layout consistency. We additionally use human-calibrated VLM judgments to estimate perceived fidelity under a separate subjective protocol. This design lets us ask not only whether a model obtains a high score, but also which reconstruction capability is responsible for success or failure.

Our contributions are threefold: (1) a diverse document-to-HTML benchmark covering heterogeneous page types, content, and layouts; (2) a reproducible multi-dimensional evaluation protocol that separates content, structure, rendering, and perceived fidelity; and (3) an empirical study of ten multimodal systems and task-specific fine-tuning. The objective and subjective leaderboards select diferent winners, and adaptation improves smaller models, showing that AmbiHTMLBench is useful both as a diagnostic evaluation and as a training target for agent-ready document representations.

## 2 Related Work

Document understanding has progressed from OCR and layout analysis to multimodal models that jointly encode text, appearance, and spatial position, including LayoutLM variants, DocFormer, LiLT, Donut, DocLLM, and mPLUG-DocOwl [10–17]. Recent agent frameworks add retrieval and tool use, but generally reason over extracted content rather than reconstructing complete documents [18, 19].

Visual generation work such as Pix2Struct, AnyDoc, WebSight, and Design2Code maps screenshots to executable markup, primarily for webpages [20–23]. Conversely, OmniDocBench, PureDocBench, and ParseBench assess document parsing and component-level fidelity [7–9]. AmbiHTMLBench bridges these directions by evaluating full document-to-HTML reconstruction, with content, layout, and perceived fidelity measured separately.

## 3 Benchmark Construction

![](images/f8a5912a1bddcc24140d86d7fad6d8ff4788e7bd0d122a4c78c99ffae9356835.jpg)  
Figure 4 The overall AmbiHTMLBench Construction process.

We collected heterogeneous document images and applied multi-stage quality control to preserve diversity in document type, style, and layout while reducing near-duplicates.

## 3.1 Data Collection

Sources span four categories: institutional PDFs (e.g., reports and datasheets), web-native documentation, newspapers and magazines, and posters or other strongly designed pages. Together they cover conventional and nonstandard layouts, dense text, tables, figures, and mixed media.

## 3.2 Filtering and Quality Control

After data collection, there are above 50,000 pages remaining. Before producing the final benchmark subset, we applied several quality control steps. First, we use MinerU[24] to conduct basic document parsing, and (1) Remove samples whose layout parsing result is missing or unusable. (2) Remove samples with less than 50 words. This step ensures that every retained sample has both a clear layout structure and abundant text content.

Next, based on MinerU layout parsing result, we provide a layout view of the documents by fill the bounding boxes with diferent colors according to their categories, and then use a layout-aware clustering to reduce redundancy. All candidate samples are clustered into ten groups using KMeans over the features obtained after feature extraction of this layout view using ResNet-50. This process groups documents primarily by page organization rather than by raw appearance or textual content.

Within each cluster, we applied an iterative diversity-preserving filtering procedure. At each iteration, we retained one candidate and removed nearby pages according to layout-feature similarity, visual similarity, estimated layout dificulty, and text volume. This procedure was repeated until the retained set reached the desired diversity level.

This procedure balances diversity and quality by limiting dominant layout families, removing near-duplicates, and retaining informative pages across dificulty levels.

After filtering 50,000+ candidates to 4,703 pages, expert review removed overly simple, truncated, overlapping, or specially edited samples, yielding 1,115 benchmark pages.

## 3.3 Dataset Statistics

Experts annotated all 1,115 pages at page and block levels. Page-level labels cover 13 element types, layout (headings, columns, lists), text appearance (blur, vertical writing, handwriting), source type, and language. Block annotations record type and text; formulas use LaTeX and tables use HTML. Pages average 2,329 characters and 26 blocks, and 56.1% exceed 1,000 characters.

## 4 Evaluation Protocol

## 4.1 Objective Evaluation

We render each prediction in headless Chromium at a fixed viewport and extract visible text, lists, tables, figures, formulas, and normalized bounding boxes. Ground-truth annotations are converted to the same block representation, enabling source-independent comparison of content and geometry.

Each object is normalized as

$$
o = ( k , t , b ) ,
$$

where k is its category, t its textual or structural content, and $\boldsymbol { b } = ( x , y , w , h )$ its page-normalized bounding box.

## 4.1.1 Block Matching Algorithm

Tables are matched first as table objects. Regular text, formulas, and cells from unmatched tables then share a text pool, giving text credit to linearized tables while reserving structural credit for preserved table objects.

Table candidates combine content, structure, text, and geometry scores and are assigned one-to-one by thresholded Hungarian matching.

Because HTML may split or merge annotated text blocks, partial text similarity defines a bipartite candidate graph; connected components support one-to-many alignments, with residuals retained as insertions or

deletions. The resulting shared alignment feeds both page-level text scoring and layout analysis. Details are in Appendix A.

## 4.1.2 Content consistency

For matched tables, we report normalized row/cell edit scores: TEDS includes cell text, while TEDS-S measures structure only (an approximation, not full DOM tree edit):

$$
S _ { \mathrm { t a b l e } } = 1 - { \frac { d _ { \mathrm { r o w / c e l l } } } { \operatorname* { m a x } ( | T _ { g } | , | T _ { p } | , 1 ) } } .
$$

Scores are reported on the 263 pages containing tables.

Page-level text fidelity concatenates aligned groups, including one-sided residuals, and computes normalized character edit distance:

$$
D _ { \mathrm { t e x t } } = \frac { \mathrm { L e v } ( T _ { g } , T _ { p } ) } { \operatorname* { m a x } ( | T _ { g } | , | T _ { p } | , 1 ) } .
$$

## 4.1.3 Layout consistency

Layout uses matched tables and text groups as nodes, compares normalized center and size, and evaluates corresponding edges in a ground-truth 3-nearest-neighbor graph. Empty-sided or unsupported text matches do not receive layout credit.

For an edge, similarity combines direction, relative displacement vector, and length:

$$
s _ { \mathrm { e d g e } } = 0 . 4 s _ { \mathrm { d i r } } + 0 . 3 s _ { \mathrm { l e n } } + 0 . 3 s _ { \mathrm { v e c } } .
$$

The page-level layout score is

$$
S _ { \mathrm { l a y o u t } } = 0 . 5 5 \overline { { s } } _ { \mathrm { e d g e } } + 0 . 3 0 \overline { { s } } _ { \mathrm { n o d e } } + 0 . 1 5 F _ { 1 } ^ { \mathrm { c o v e r a g e } } .
$$

The score retains two-dimensional direction and displacement; coverage penalizes omitted reference content.

## 4.2 Subjective Evaluation

Objective metrics cannot fully capture how content and layout errors interact in perception, so we add reference-anchored pairwise judgments.

Each trial presents a reference page and two reconstructions and asks which is more faithful. Figure 5 summarizes the protocol.

![](images/0dcc8c91fedf5ab495684e6d4a5c941154601579fcf23913b0f3c84bd355e6f2.jpg)  
Figure 5 Human calibration and deployment of the frozen preference proxy.

Judgement protocol and human reference. Judges choose A, B, or tie and provide confidence and rationale. Photograph and chart interiors are neutralized while presence, geometry, and placement remain visible.

Diagnostic axes cover global layout, text presentation, and element-specific properties (tables, formulas, algorithms, vertical text, and list hierarchy); scores range from −2 to 2 and do not afect the overall ranking. Thirty-three participants completed 7,920 pairwise judgments (7,722 main annotations and 198 hidden repeats; repeats are excluded from calibration; see Appendix B.1). We encode A, tie, and B as 1, 0.5, and 0; the mean valid vote is

$$
h _ { i } = \frac { n _ { i , A } + 0 . 5 n _ { i , T } } { n _ { i , A } + n _ { i , T } + n _ { i , B } } .
$$

$n _ { i , A } , n _ { i , T }$ , and $n _ { i , B }$ are valid vote counts.

Frozen proxy and deployment. Five VLM judges provide canonicalized votes $x _ { i , j }$ . We fit and freeze a linear proxy on human-calibrated comparisons:

$$
\hat { p } _ { i } ( A ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \sum _ { j = 1 } ^ { 5 } \beta _ { j } x _ { i , j } \right) ,\tag{1}
$$

Coeficients are estimated by ordinary least squares on human labels only. A centered Bradley–Terry model uses $\hat { p } _ { i } ( A )$ as a fractional outcome; page-block bootstrap estimates ranking uncertainty. The schedule includes 1,872 human-labelled and 8,128 proxy-scored comparisons. Prompts, audits, coeficients, and implementation details appear in Appendix B.

## 5 Experiments

We organize the experiments around two complementary leaderboards. The objective leaderboard decomposes reconstruction quality into content and layout fidelity, whereas the subjective leaderboard captures perceived fidelity after these errors interact.

## 5.1 Experimental Setup

We evaluate fifteen multimodal models on the same 1,115-page benchmark set. Each model receives the page image under its standard image-to-HTML prompting interface, and only successfully rendered outputs are scored. Objective scores are computed from the fixed protocol in Section 4.1; the subjective leaderboard uses the frozen preference proxy in Section 4.2. We merge the final reruns of previously identified low-scoring pages before producing the reported leaderboard. Rendering, matching thresholds, and aggregation are fixed before comparing models; implementation details are provided in Appendix A.

## 5.2 Objective Leaderboard

Gemini 3.1 Pro Preview leads the objective ranking, primarily because it has the lowest page-level text error and the strongest table score while maintaining a competitive layout score. Kimi-K3 and Claude Opus 5 form the next tier: Kimi-K3 is slightly stronger on text, whereas Claude Opus 5 is slightly stronger on layout and table structure. Qwen3.7-Plus ranks fourth by balancing moderate text error with stable table and layout scores. GPT-5.6-Sol obtains the highest layout score, but its larger text error and weaker table score place it below the content-leading systems. Grok-4.5 is separated from the remaining models on every axis, especially text, table fidelity, and layout. These results motivate reporting the decomposition rather than interpreting a single score as a complete description of reconstruction quality.

Table 1 Objective leaderboard on 1,115 pages for fifteen primary systems. Text ED is lower-is-better; all other columns are higher-is-better. The weighted score uses table weight 0.3 and text and layout weights 1.0. TEDS and TEDS-S are averaged over the 263 table-bearing pages.
<table><tr><td>Rank</td><td>Model</td><td>Text ED↓</td><td>TEDS↑</td><td>TEDS-S↑</td><td>Layout↑</td><td>Obj. score↑</td></tr><tr><td>1</td><td>Gemini 3.1 Pro Preview</td><td>0.1754</td><td>0.6518</td><td>0.6650</td><td>0.8410</td><td>0.8092</td></tr><tr><td>2</td><td>Claude Opus 5.5</td><td>0.1994</td><td>0.6433</td><td>0.6553</td><td>0.8574</td><td>0.8048</td></tr><tr><td>3</td><td>Gemini 3.8 Flash</td><td>0.2070</td><td>0.6415</td><td>0.6530</td><td>0.8288</td><td>0.7888</td></tr><tr><td>4</td><td>Kimi-K3</td><td>0.2399</td><td>0.6253</td><td>0.6389</td><td>0.8307</td><td>0.7732</td></tr><tr><td>5</td><td>Claude Opus 5</td><td>0.2537</td><td>0.6290</td><td>0.6465</td><td>0.8387</td><td>0.7712</td></tr><tr><td>6</td><td>Qwen 3.8 max</td><td>0.2470</td><td>0.6095</td><td>0.6235</td><td>0.8198</td><td>0.7633</td></tr><tr><td>7</td><td>GPT 6.1 sol</td><td>0.3082</td><td>0.6277</td><td>0.6390</td><td>0.8654</td><td>0.7589</td></tr><tr><td>8</td><td>GPT 6 astra</td><td>0.3133</td><td>0.6188</td><td>0.6324</td><td>0.8654</td><td>0.7555</td></tr><tr><td>9</td><td>Qwen3.7-Plus</td><td>0.2659</td><td>0.6113</td><td>0.6250</td><td>0.8088</td><td>0.7505</td></tr><tr><td>10</td><td>GPT 5.6 Sol</td><td>0.3055</td><td>0.5967</td><td>0.6103</td><td>0.8425</td><td>0.7461</td></tr><tr><td>11</td><td>GPT 5.6 Terra</td><td>0.2998</td><td>0.5953</td><td>0.6131</td><td>0.8335</td><td>0.7445</td></tr><tr><td>12</td><td>Claude Sonnet 5</td><td>0.2935</td><td>0.6042</td><td>0.6218</td><td>0.8218</td><td>0.7433</td></tr><tr><td>13</td><td>ERNIE 5.0</td><td>0.2871</td><td>0.5982</td><td>0.6137</td><td>0.8120</td><td>0.7410</td></tr><tr><td>14</td><td>GLM 5V Turbo</td><td>0.3081</td><td>0.6121</td><td>0.6316</td><td>0.8103</td><td>0.7330</td></tr><tr><td>15</td><td>Grok 4.5</td><td>0.6754</td><td>0.2874</td><td>0.3024</td><td>0.3950</td><td>0.3504</td></tr></table>

## 5.3 Task-Specific Fine-tuning

To test whether AmbiHTMLBench can also serve as a training signal, we fine-tune Qwen3.5-9B and Qwen3.5-4B with transcription-style distillation. We use 1,230 accepted examples from 1,475 images in the PureDocBench real\_degraded track, with no overlap with the benchmark pages. Gemini 3.1 Pro Preview produces HTML using the same transcription instruction as inference; outputs are retained using a block-recall filter (83.4% acceptance), with best-of-four resampling for initially low-scoring pages. This construction favors faithful transcription of the observed page rather than authored or normalized HTML.

Both model sizes use the same one-epoch LoRA recipe: rank and alpha 16, all linear language-model layers trainable, frozen vision and alignment modules, learning rate 10<sup>−5</sup> with cosine decay, and a maximum sequence length of 21,504. Checkpoints are first screened with inexpensive coverage and output-discipline diagnostics and then selected by the full objective evaluator. Table 2 reports the best checkpoints on all 1,115 pages.

Distillation improves the 9B and 4B bases by 4.41 and 1.48 weighted points, respectively, with gains on every 9B axis and on text and table fidelity for 4B. The smaller adapted model also exceeds the unadapted 9B base (0.6934 versus 0.6623). A matched-budget 27B ablation isolates the target style: transcription-style targets reach 0.7373 from a 0.7320 base, whereas cleaned, normalized, and original authored-HTML targets reach 0.7102, 0.7141, and 0.6850. Thus, the alignment between the supervision format and the reconstruction task, rather than surface cleanliness alone, determines the direction of the SFT gain. A subsequent DAPO-style RL stage adds only 0.16 and 0.27 points over the best 9B and 4B SFT checkpoints, respectively, which is within the observed evaluation variability; we therefore use SFT as the primary adapted-model result.

## 5.4 Reliability of Subjective Evaluation

Human reliability. Valid A/B/TIE preferences cover 98.85% of the 7,722 main annotations and provide a reference for all 1,872 comparisons. Exact inter-rater agreement is 62.30% (95% CI: 60.79–63.82), while within-rater agreement on 198 hidden-repeat pairs is 73.23%. Despite these disagreements, removing any one of the 33 participants leaves the ten-model human ranking unchanged (Spearman = 1.0; maximum rank movement = 0). This supports stability to participant removal, not unanimity on individual comparisons.

Table 2 Task-specific fine-tuning results on AmbiHTMLBench. All reported metrics are higher-is-better; “Text” is the normalized text score and the weighted score is the primary objective metric.
<table><tr><td>Model</td><td>Method</td><td>Weighted↑</td><td>Text↑</td><td>TEDS↑</td><td>Layout↑</td></tr><tr><td>Qwen3.5-9B</td><td>Base</td><td>0.6623</td><td>0.6321</td><td>0.5440</td><td>0.7281</td></tr><tr><td>Qwen3.5-9B</td><td>Distill SFT</td><td>0.7064</td><td>0.6761</td><td>0.5827</td><td>0.7737</td></tr><tr><td>Qwen3.5-4B</td><td>Base</td><td>0.6786</td><td>0.6323</td><td>0.5382</td><td>0.7670</td></tr><tr><td>Qwen3.5-4B</td><td>Distill SFT</td><td>0.6934</td><td>0.6556</td><td>0.5722</td><td>0.7675</td></tr></table>

Human–machine alignment. We next test whether the human-calibrated preference proxy recovers the human reference. We fit weights on seven annotation blocks and predict the held-out eighth, rotating through all eight blocks. Predictions cover all 1,872 comparisons, yielding Brier 0.0778 (95% CI: 0.0722–0.0836) and MAE 0.2115 (0.2031–0.2200). The resulting ten-model ranking has Spearman correlation 0.915 (0.867–0.952) with the human ranking. These internal estimates use four-vote means per judge, not the single votes used in deployment; they do not independently validate the expanded leaderboard. Page-bootstrap intervals hold these predictions fixed. Appendix B reports the full agreement measures, fitting protocol, and committee audit, including its pilot-to-holdout reversal.

## 5.5 Subjective Leaderboard

The subjective leaderboard measures perceived fidelity, which depends on the interaction among content, structure, and rendering errors. We therefore report it separately from the objective leaderboard rather than treating human preference as another automatic metric.

Table 3 Final subjective leaderboard from the frozen preference proxy. BT scores are centered; soft preference averages predicted win probability over scheduled appearances. Intervals use page-block bootstrap. The 8,128 additional comparisons are proxy-scored rather than independently human-labelled.
<table><tr><td rowspan="2">Rank</td><td rowspan="2">Model</td><td colspan="3">BT score (95% CI)</td><td rowspan="2">Soft pref. ↑</td><td rowspan="2">Rank CI</td></tr><tr><td>Score</td><td>Low</td><td>High</td></tr><tr><td>1</td><td>GPT 5.6 Sol</td><td>154.70</td><td>140.59</td><td>168.82</td><td>71.64%</td><td>[1, 1]</td></tr><tr><td>2</td><td>GPT 5.6 Terra</td><td>102.87</td><td>90.18</td><td>116.25</td><td>64.76 %</td><td>[2, 3]</td></tr><tr><td>3</td><td>Gemini 3.1 Pro Preview</td><td>100.86</td><td>89.88</td><td>112.74</td><td>64.49 %</td><td>[2, 3]</td></tr><tr><td>4</td><td>Qwen3.7 Plus</td><td>56.04</td><td>45.63</td><td>66.45</td><td>58.12 %</td><td>[4, 4]</td></tr><tr><td>5</td><td>Kimi K3</td><td>9.62</td><td>-2.16</td><td>21.71</td><td>51.32 %</td><td>[5, 6]</td></tr><tr><td>6</td><td>Claude Opus 5</td><td>-2.47</td><td>-14.01</td><td>8.96</td><td>49.51%</td><td>[5, 6]</td></tr><tr><td>7</td><td>Grok 4.5</td><td>-55.49</td><td>-68.68</td><td>-43.23</td><td>41.77%</td><td>[7,8]</td></tr><tr><td>8</td><td>Claude Sonnet 5</td><td>-58.17</td><td>-68.74</td><td>-47.74</td><td>41.38 %</td><td>[7,8]</td></tr><tr><td>9</td><td>GLM 5V Turbo</td><td>-117.28-129.38-106.41</td><td></td><td></td><td>33.12 %</td><td>[9,9]</td></tr><tr><td>10</td><td>ERNIE 5.0</td><td>-190.68-203.33 -178.59</td><td></td><td></td><td>23.88 %</td><td>[10,10]</td></tr></table>

GPT-5.6-Sol ranks first in every bootstrap sample, indicating a stable advantage under the subjective protocol. The remaining systems form several overlapping tiers, especially Terra–Gemini, Kimi-K3–Opus, and Grok–Sonnet. The leaderboard therefore supports robust separation between the strongest systems and the lower-performing group, but does not justify fine-grained claims within unresolved tiers. Because the additional comparisons lack human labels, this table represents a proxy-based perceived-fidelity ranking rather than 10,000 independently human-validated comparisons.

## 5.6 Comparing the Two Leaderboards

The rankings are moderately correlated (Spearman ρ = 0.588; Kendall τ = 0.378). GPT-5.6-Sol ranks first subjectively but fifth objectively, whereas Gemini ranks first objectively and third subjectively.

![](images/bdd446e9e87faa181bc247cae47bc72f7cd90845fdd229229022020ae6bde797.jpg)  
Figure 6 Page-level reversal on page 0309: objective metrics favor Opus (A), whereas recorded subjective judgments favor Sol (B). Insets contrast faithful wording with preserved visual emphasis.

On page 0309 (Figure 6), Opus wins both applicable objective metrics, but the same pair’s recorded subjective proxy favors Sol (0.8399; four of five VLM votes). Sol preserves red emphasis while changing technological content to comprehensive content; Opus preserves the wording but loses the emphasis. This content error coexists with stronger perceived presentation, illustrating complementary evaluation signals.

## 6 Conclusion

We introduced AmbiHTMLBench, a 1,115-page benchmark for reconstructing heterogeneous documents as executable HTML. Its annotations and evaluation protocol treat content, structure, and rendered appearance as related but distinct requirements. The objective leaderboard shows that the best model depends on the axis being measured, while the subjective study demonstrates that perceived fidelity cannot be recovered from any single automatic component.

## References

[1] Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang, W. Che, M. Zhang, and L. Zhou, “LayoutLMv2: Multi-modal pre-training for visually-rich document understanding,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, 2021, pp. 2579–2591. [Online]. Available: https://aclanthology.org/2021.acl-long.201/

[2] D. Wang, N. Raman, M. Sibue, Z. Ma, P. Babkin, S. Kaur, Y. Pei, A. Nourbakhsh, and X. Liu, “DocLLM: A layout-aware generative language model for multimodal document understanding,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2024, pp. 8529–8548. [Online]. Available: https://aclanthology.org/2024.acl-long.463/

[3] A. Hu, H. Xu, J. Ye, M. Yan, L. Zhang, B. Zhang, J. Zhang, Q. Jin, F. Huang, and J. Zhou, “mPLUG-DocOwl 1.5: Unified structure learning for OCR-free document understanding,” in Findings of the Association for Computational Linguistics: EMNLP 2024. Association for Computational Linguistics, 2024, pp. 3096–3120. [Online]. Available: https://aclanthology.org/2024.findings-emnlp.175/

[4] S. Han, P. Xia, R. Zhang, T. Sun, Y. Li, H. Zhu, and H. Yao, “Mdocagent: A multi-modal multi-agent framework for document understanding,” arXiv preprint arXiv:2503.13964, 2025.

[5] J. Li, Y. Xu, L. Cui, and F. Wei, “MarkupLM: Pre-training of text and markup language for visually rich document understanding,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2022, pp. 6078–6087. [Online]. Available: https://aclanthology.org/2022.acl-long.420/

[6] X. Deng, P. Shiralkar, C. Lockard, B. Huang, and H. Sun, “DOM-LM: Learning generalizable representations for HTML documents,” arXiv preprint arXiv:2201.10608, 2022. [Online]. Available: https://arxiv.org/abs/2201.10608

[7] L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao, J. Shi, F. Wu, P. Chu, M. Liu, Z. Li, C. Xu, B. Zhang, B. Shi, Z. Tu, and C. He, “OmniDocBench: Benchmarking diverse PDF document parsing with comprehensive annotations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 24 838–24 848. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2025/html/Ouyang\_OmniDocBench\_Benchmarking\_ Diverse\_PDF\_Document\_Parsing\_with\_Comprehensive\_Annotations\_CVPR\_2025\_paper.html

[8] B. Zhang, S. G. Acosta, P. Carlson, S. Bron, P.-L. Doulcet, and S. Suo, “ParseBench: A document parsing benchmark for AI agents,” arXiv preprint arXiv:2604.08538, 2026. [Online]. Available: https://arxiv.org/abs/2604.08538

[9] Z. Li, Z. Ma, J. Chen, J. Zhang, Z. Su, Y. Zhang, Z. Yu, R. Liu, X. Lv, B. Li, J. Gao, Z. Zhang, C. Yuan, B. Li, and W. Hu, “How far is document parsing from solved? PureDocBench: A source-traceable benchmark across clean, degraded, and real-world settings,” arXiv preprint arXiv:2605.07492, 2026. [Online]. Available: https://arxiv.org/abs/2605.07492

[10] Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, and M. Zhou, “Layoutlm: Pre-training of text and layout for document image understanding,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, ser. KDD ’20. ACM, Aug. 2020, p. 1192–1200. [Online]. Available: http://dx.doi.org/10.1145/3394486.3403172

[11] Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang, W. Che, M. Zhang, and L. Zhou, “Layoutlmv2: Multi-modal pre-training for visually-rich document understanding,” 2022. [Online]. Available: https://arxiv.org/abs/2012.14740

[12] Y. Huang, T. Lv, L. Cui, Y. Lu, and F. Wei, “Layoutlmv3: Pre-training for document ai with unified text and image masking,” 2022. [Online]. Available: https://arxiv.org/abs/2204.08387

[13] S. Appalaraju, B. Jasani, B. U. Kota, Y. Xie, and R. Manmatha, “Docformer: End-to-end transformer for document understanding,” 2021. [Online]. Available: https://arxiv.org/abs/2106.11539

[14] J. Wang, L. Jin, and K. Ding, “Lilt: A simple yet efective language-independent layout transformer for structured document understanding,” 2022. [Online]. Available: https://arxiv.org/abs/2202.13669

[15] G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, and S. Park, “Ocr-free document understanding transformer,” 2022. [Online]. Available: https://arxiv.org/abs/2111.15664

[16] D. Wang, N. Raman, M. Sibue, Z. Ma, P. Babkin, S. Kaur, Y. Pei, A. Nourbakhsh, and X. Liu, “Docllm: A layout-aware generative language model for multimodal document understanding,” 2023. [Online]. Available: https://arxiv.org/abs/2401.00908

[17] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu, C. Li, J. Tian, Q. Qi, J. Zhang, and F. Huang, “mplug-docowl: Modularized multimodal large language model for document understanding,” 2023. [Online]. Available: https://arxiv.org/abs/2307.02499

[18] S. Han, P. Xia, R. Zhang, T. Sun, Y. Li, H. Zhu, and H. Yao, “Mdocagent: A multi-modal multi-agent framework for document understanding,” 2025. [Online]. Available: https://arxiv.org/abs/2503.13964

[19] D. Yang, A. Simoulin, X. Qian, X. Liu, Y. Cao, Z. Teng, and G. Yang, “Docagent: A multi-agent system for automated code documentation generation,” 2025. [Online]. Available: https://arxiv.org/abs/2504.08725

[20] K. Lee, M. Joshi, I. R. Turc, H. Hu, F. Liu, J. M. Eisenschlos, U. Khandelwal, P. Shaw, M.-W. Chang, and K. Toutanova, “Pix2Struct: Screenshot parsing as pretraining for visual language understanding,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 18 893–18 912. [Online]. Available: https://proceedings.mlr.press/v202/lee23g.html

[21] J. Lin, W. Zhu, V. I. Morariu, and C. Tensmeyer, “AnyDoc: Enhancing document generation via large-scale HTML/CSS data synthesis and height-aware reinforcement optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/papers/Lin\_AnyDoc\_Enhancing\_Document\_ Generation\_via\_Large-Scale\_HTMLCSS\_Data\_Synthesis\_and\_CVPR\_2026\_paper.pdf

[22] H. Laurençon, L. Tronchon, and V. Sanh, “Unlocking the conversion of web screenshots into html code with the websight dataset,” 2024. [Online]. Available: https://arxiv.org/abs/2403.09029

[23] C. Si, Y. Zhang, R. Li, Z. Yang, R. Liu, and D. Yang, “Design2Code: Benchmarking multimodal code generation for automated front-end engineering,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Albuquerque, New Mexico: Association for Computational Linguistics, 2025, pp. 3956–3974. [Online]. Available: https://aclanthology.org/2025.naacl-long.199/

[24] J. Niu, Z. Liu, Z. Gu, B. Wang, L. Ouyang, Z. Zhao, T. Chu, T. He, F. Wu, Q. Zhang, Z. Jin, G. Liang, R. Zhang, W. Zhang, Y. Qu, Z. Ren, Y. Sun, Y. Zheng, D. Ma, Z. Tang, B. Niu, Z. Miao, H. Dong, S. Qian, J. Zhang, J. Chen, F. Wang, X. Zhao, L. Wei, W. Li, S. Wang, R. Xu, Y. Cao, L. Chen, Q. Wu, H. Gu, L. Lu, K. Wang, D. Lin, G. Shen, X. Zhou, L. Zhang, Y. Zang, X. Dong, J. Wang, B. Zhang, L. Bai, P. Chu, W. Li, J. Wu, L. Wu, Z. Li, G. Wang, Z. Tu, C. Xu, K. Chen, Y. Qiao, B. Zhou, D. Lin, W. Zhang, and C. He, “Mineru2.5: A decoupled vision-language model for eficient high-resolution document parsing,” 2025. [Online]. Available: https://arxiv.org/abs/2509.22186

## A Objective Evaluation Details

This appendix records implementation choices that are fixed for reproducibility but are not central to the paper’s conceptual contribution.

## A.1 Rendered block extraction and normalization

Each predicted HTML file is rendered in headless Chromium at the benchmark viewport. The evaluator traverses visible DOM content and records the semantic type, visible text, raw table HTML when available, LaTeX-like markup when exposed by the rendered DOM, and the rendered box. A predicted table is kept as one table object; its cells are not independently promoted to page-level blocks unless the table remains unmatched and is transferred to the text pool. Elements whose class attribute contains placeholder are excluded from matching. Invisible text, transparent text, and extremely small text are also ignored. This prevents accessibility or scene-description placeholders from contributing text that is not visibly rendered as document content.

Ground-truth boxes and predicted boxes are mapped to the common page coordinate system. For each object, the outermost coverage box is used. The overlap used by matching is

$$
s _ { \mathrm { o v } } ( g , p ) = \frac { | b _ { g } \cap b _ { p } | } { \operatorname* { m i n } ( | b _ { g } | , | b _ { p } | ) }
$$

with zero returned for disjoint boxes. Thus, the denominator is not the union area: a small block fully covered by a larger block can obtain an overlap of one. This is useful for detecting splits and merges, but the final text and layout scores still penalize missing or excessive content.

## A.2 Table assignment and TEDS approximation

For every ground-truth/predicted table pair, the evaluator computes TEDS, TEDS-S, partial text similarity, and the smaller-area overlap. The candidate score is

$$
0 . 4 5 \mathrm { T E D S } + 0 . 2 5 \mathrm { T E D S - S } + 0 . 2 0 s _ { \mathrm { p a r t i a l } } + 0 . 1 0 s _ { \mathrm { o v } } .
$$

Pairs below the table candidate threshold 0.40 are unavailable to the assignment. The remaining score matrix is solved with a Hungarian one-to-one assignment, followed by threshold validation. This is a global assignment over table candidates, not a greedy sort.

The table parser converts HTML tr/td/th elements into ordered rows and cells. The implemented score is a normalized row/cell sequence edit approximation rather than a general-purpose DOM tree edit distance. TEDS retains cell strings; TEDS-S replaces them with structural cell tokens. Both are reported in [0, 1], with one denoting exact agreement. Cell substitutions are weighted by the normalized character-level text distance, so a one-character diference has a smaller cost than a substantially diferent cell. Row, insertion, deletion, and cell-structure diferences are also represented. Pages without a ground-truth table are omitted from the table means.

## A.3 Text normalization and grouping

Text matching uses a conservative normalization. Whitespace and dollar delimiters are removed, while punctuation, mathematical symbols, CJK characters, Japanese, Korean, and other Unicode letters are retained. A ground-truth $\xi$ is interpreted as a wildcard character during edit-distance computation. Placeholder elements are removed before this stage. Formula objects from both sides are converted to text objects by removing dollar delimiters, and no separate formula CDM score is computed in the current objective leaderboard.

For every ground-truth and predicted text pair, the evaluator computes partial\_ratio after the above normalization. A bipartite edge is created when

$$
s _ { \mathrm { p a r t i a l } } \geq 0 . 6 6 .
$$

Unlike earlier overlap-driven variants, this candidate graph is text-led: geometry is used later for layout scoring rather than for admitting text edges. Connected components of the bipartite graph form local candidate regions. Singleton blocks without edges are kept as residual insertions or deletions, so they still afect page text distance but do not create layout nodes.

For each connected component, the evaluator constructs several deterministic order hypotheses: visual reading order, horizontal order, reversed reading order, and a graph-guided order that places a block near its strongest partners on the other side. It chooses the pair of orders with the lowest concatenated component edit distance. For components whose total number of blocks is small enough, the evaluator additionally tries a bounded local reorder pass over the recovered groups.

Given fixed ground-truth and predicted orders, the evaluator computes the final groups with a monotone segmentation dynamic program. Each transition may consume a short ground-truth segment, a short predicted segment, or both. Two-sided transitions are supported by the bipartite edges in the component, while one-sided transitions represent insertions or deletions and receive full edit cost. The recovered groups therefore support one-to-one, one-to-many, many-to-one, and many-to-many matches while preventing unsupported distant text from being grouped solely to reduce local cost.

After all components and residual blocks are processed, the evaluator performs one page-level ordering pass over the recovered groups and computes a single normalized edit distance:

$$
D _ { \mathrm { t e x t } } = \frac { \mathrm { L e v } ( T _ { g } , T _ { p } ) } { \operatorname* { m a x } ( | T _ { g } | , | T _ { p } | , 1 ) } .
$$

This max denominator makes the score symmetric with respect to which side contains more characters. A lower score is better. The same recovered groups, excluding groups with an empty side, are passed to the layout scorer.

## A.4 Layout graph similarity

The layout graph is built only from matched objects. A matched table contributes one node. A text group contributes one node whose box is the axis-aligned envelope of all GT or predicted boxes in that group. Text groups with an empty GT side or an empty prediction side are not inserted into the graph. When non-text anchors such as matched tables are available, very weak text groups whose normalized text distance exceeds 0.75 are also excluded from graph construction. Node similarity combines normalized center displacement with width and height agreement. For the ground-truth nodes, the evaluator computes the $k = 3$ nearest neighbors by Euclidean center distance. The same node identities are used to read the corresponding predicted relations, so the graph comparison does not require a second text matching step.

For an edge $i  j ,$ direction similarity compares the normalized two-dimensional vectors, length similarity compares the relative Euclidean distances, and vector similarity compares the normalized displacement after prediction. The edge score is

$$
s _ { \mathrm { e d g e } } = 0 . 4 s _ { \mathrm { d i r } } + 0 . 3 s _ { \mathrm { l e n } } + 0 . 3 s _ { \mathrm { v e c } } .
$$

The final score is

$$
S _ { \mathrm { l a y o u t } } = 0 . 5 5 \overline { { s } } _ { \mathrm { e d g e } } + 0 . 3 0 \overline { { s } } _ { \mathrm { n o d e } } + 0 . 1 5 F _ { 1 } ^ { \mathrm { c o v e r a g e } } .
$$

This construction preserves two-dimensional information: a relation that has the correct distance but points in a diferent direction receives a lower score. A group with many source blocks still contributes one graph node because its envelope represents the layout region that was matched. The coverage term is computed from the matched graph nodes relative to the objects eligible for layout scoring, so pages that recover only a small subset of meaningful matched objects receive lower layout scores even when the recovered subset is geometrically accurate.

## A.5 Leaderboard aggregation

The reported table means use only the 263 pages with ground-truth tables. Text error and layout similarity use all 1,115 pages. We convert text error to quality by $1 - D _ { \mathrm { t e x t } }$ and use

$$
S _ { \mathrm { o b j } } = \frac { 0 . 3 \overline { { \mathrm { T E D S } } } + \overline { { ( 1 - D _ { \mathrm { t e x t } } ) } } + \overline { { S _ { \mathrm { l a y o u t } } } } } { 2 . 3 } .
$$

TEDS-S is reported as a diagnostic and is not an additional term in the weighted aggregate, avoiding double-counting table structure. The final leaderboard incorporates the rerun outputs for the previously identified low-scoring pages before aggregation.

## B Subjective Evaluation Supplement

## B.1 Human study and preference targets

The human study comprises eight disjoint blocks of 234 physical comparisons. Blocks 1–5 have five participants each, block 6 has four, and blocks 7–8 have two, for 33 participants in total. Each participant answers 234 main trials and six hidden repeats. Thus, the 7,920 responses comprise 7,722 main annotations and 198 repeated annotations; repeats are used to assess short-term consistency and are excluded from the calibration targets. Of the main annotations, 7,633 contain a valid A, B, or TIE preference and 89 are CANNOT\_JUDGE, giving 98.85% annotation coverage. All 1,872 physical comparisons retain a human target.

For comparison $i , n _ { i , A } , n _ { i , T }$ , and $n _ { i , B }$ count valid votes after mapping the displayed candidates to their canonical identities. Their encoded mean $h _ { i }$ measures preference for canonical A. A tie contributes half a preference point to each candidate. For example, three A votes, one tie, and one B vote give $h _ { i } = 0 . 7$ . A value of 0.5 can arise from unanimous ties or opposing $\mathrm { A } / \mathrm { B }$ votes, so it does not by itself measure agreement. We concatenate the eight 234-entry vectors to obtain the 1,872-entry human reference; we do not average unrelated comparisons across blocks.

The reported pairwise inter-rater agreement uses 13,265 valid within-comparison rating pairs. Exact agreement is 62.30% (95% CI: 60.79–63.82). Fleiss-type κ is 0.357 (0.334–0.379), and ordinal Krippendorf-type α is 0.459 (0.433–0.484). These quantify agreement, not error against an objective ground truth. Among the 1,872 comparisons, 39.85% are unanimous among valid raters and 82.00% have a unique plurality; a plurality need not exceed half the valid votes. Hidden-repeat agreement is 145/198 (73.23%). Removing each of the 33 participants in turn leaves all ten human ranks unchanged. These checks address diferent sources of variation; they do not establish that individual human judgments are noise-free. Block content and participant composition are partly confounded, particularly for the two-rater blocks.

## B.2 Judge inputs, axes, and prompt

Each VLM receives the reference image followed by the two candidate images, with model identities omitted. The reference is the visual standard. Media neutralization removes photograph/chart interior content while retaining presence, geometry, placement, and caption relationships. Text is judged by visible presentation and substantial omissions or garbling rather than character-by-character OCR correctness. Table 4 gives the active comparison counts in the human set; multiple axes may occur in one comparison.

The following is the shared English prompt, with line wrapping adjusted for readability:

You are an independent visual reconstruction judge for AmbiHTMLBench.

You receive exactly three page images in this order: REFERENCE, CANDIDATE A, CANDIDATE B. The Reference is the sole visual standard. Determine which candidate more faithfully and completely reconstructs it as a human viewer would.

Images and charts: do not judge their internal semantic content. Judge only presence or absence, geometry, size, aspect ratio, position, alignment, and relationships to captions and neighboring regions.

Table 4 Diagnostic axes in the 1,872-comparison human set. Counts denote comparisons, not distinct pages. The rarest axes support case analysis but not precise population comparisons.
<table><tr><td>Activation</td><td>Axis</td><td>n Visual criteria</td><td></td></tr><tr><td>Fixed</td><td>Global layout</td><td>1,872</td><td>Regions, columns, spacing, grouping, read- ing order, proportions.</td></tr><tr><td>Fixed</td><td>Text presentation 1,872</td><td></td><td>Wrapping, typography, density, alignment, hierarchy, legibility.</td></tr><tr><td>Conditional Table</td><td></td><td>497</td><td>Visible row/column organization and cell alignment.</td></tr><tr><td>Conditional Formula</td><td></td><td>61</td><td>Fractions, scripts, radicals, matrices, and mathematical alignment.</td></tr><tr><td>Conditional Algorithm</td><td></td><td>12</td><td>Block organization, indentation, and step hierarchy.</td></tr><tr><td>Conditional Vertical text</td><td></td><td>90</td><td>Vertical flow, orientation, and reading or- ganization.</td></tr><tr><td>Conditional List hierarchy</td><td></td><td>9</td><td>Nesting, indentation, markers, and group- ing.</td></tr></table>

Text: judge visible presentation, not character-by-character OCR correctness. Consider missing or extra text regions, large garbling, wrapping, line count, density, typography, size, weight, color, emphasis, spacing, alignment, hierarchy and legibility. Tiny character or punctuation diferences that are dificult to notice should have little or no efect.

GLOBAL\_LAYOUT\_FIDELITY covers page-level spatial organization, regions and columns, alignment, adjacency, spacing, grouping, nesting, reading order, proportions, aspect ratios, whitespace and visual balance.

TEXT\_VISUAL\_PRESENTATION covers the visible presentation of text across the page.

Adaptive axes are present only when the corresponding structure exists in the Reference. Use each supplied English description and bbox\_1000 target only to locate the Reference structure. The candidate structure may move; judge its corresponding visible organization rather than absolute coordinates.

Score every active axis: 2 = A clearly better; 1 = A slightly better; 0 = no stable perceptible diference;   
−1 = B slightly better; −2 = B clearly better. Overall winner must be A, B, or TIE. It is a holistic visual judgment, not a mechanical sum of axis scores.   
Confidence 1–5 means expected reproducibility, not magnitude of diference.

Return strict JSON only:

{"winner":"A|B|TIE", "confidence":1,

"reason":"brief English reason, <=80 words",

"axes":[{"id":"AXIS\_ID", "score":0}]}

The axes array must exactly match the supplied active axes in membership and order. Do not mention model identities or infer which model produced a candidate.

An ACTIVE\_AXES JSON array is appended to this prompt for each comparison. Every entry contains its axis ID, global/adaptive scope, English description, and any reference targets with their role and bbox\_1000 coordinates. This supplies location guidance without requiring the candidate structure to occupy the same coordinates. The response validator checks the overall label, confidence range, axis membership/order, and integer scores. Swapped displays are mapped back by exchanging A/B and negating axis scores; TIE is unchanged. Confidence and free-text rationales do not enter the preference proxy.

## B.3 Preference-proxy fitting and internal validation

The five judges are Claude Opus 5, Claude Sonnet 5, Gemini 3.1 Pro Preview, GPT-5.6-Sol, and GPT-5.6-Terra. Each evaluates every human comparison in two orders and two repetitions, yielding 37,440 completed outputs. Let $v _ { i j o r } \in \{ 0 , 0 . 5 , 1 \}$ be judge $j ^ { \cdot } \mathrm { s }$ canonicalized overall vote under order o and repeat r. The training feature is

$$
\bar { x } _ { i j } = \frac { 1 } { 4 } \sum _ { o = 1 } ^ { 2 } \sum _ { r = 1 } ^ { 2 } v _ { i j o r } , \qquad \widehat { \beta } = \arg \operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { 5 } } \sum _ { i = 1 } ^ { 1 8 7 2 } ( h _ { i } - \bar { \mathbf { x } } _ { i } ^ { \mathsf { T } } \beta ) ^ { 2 } .
$$

We use ordinary least squares, giving every physical comparison equal weight. The fit has no intercept, ridge penalty, nonnegativity constraint, or sum-to-one constraint. The resulting coeficients happen to be positive; they are not normalized. Table 5 records the frozen full-data coeficients. Special-axis scores do not enter this fit. In the ten-parameter representation, the other five coeficients are zero.

Table 5 Frozen preference-proxy coeficients. The deployment applies clipping to the weighted sum; the coeficients are not normalized.
<table><tr><td>Judge</td><td>Coefficient  $\beta _ { j }$ </td></tr><tr><td>Claude Opus 5</td><td>0.09045204815695966</td></tr><tr><td>Claude Sonnet 5</td><td>0.10401091278902799</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>0.37576678499354577</td></tr><tr><td>GPT-5.6-Sol</td><td>0.13742077090225210</td></tr><tr><td>GPT-5.6-Terra</td><td>0.16012498207363500</td></tr></table>

For internal evaluation, each of the eight annotation blocks is held out in turn: 1,638 comparisons fit the coeficients and 234 receive predictions. The identifiers PAPER-01 through PAPER-08 denote these annotation blocks, not source research articles. The 1,872 out-of-fold predictions yield equal-case Brier/MSE $\begin{array} { r } { N ^ { - 1 } \sum _ { i } ( \hat { p } _ { i } - h _ { i } ) ^ { 2 } = 0 . 0 7 7 8 ~ ( 9 5 \% ~ \mathrm { C I } ; ~ 0 . 0 7 2 2 - 0 . 0 8 3 6 ) } \end{array}$ and MAE 0.2115 (0.2031–0.2200), with 100% prediction coverage. Spearman 0.915 (0.867–0.952) compares the ten-model proxy and human rankings; it is not a case-level correlation. The full-data coeficients are used for deployment, not to compute these out-of-fold alignment estimates.

Deployment substitutes one canonicalized vote for each four-vote mean. The internal estimates therefore do not directly validate single-pass deployment on new comparisons. Also, $\begin{array} { r } { \sum _ { j } \beta _ { j } \simeq 0 . 8 6 7 8 \colon } \end{array}$ re-encoding all inputs as $1 - x _ { j }$ does not generally give $1 - \hat { p } ( A )$ . Canonical identities are consequently fixed throughout scoring. This asymmetry is a limitation of the frozen estimator, not corrected post hoc.

## B.4 Discrete committee audit and cost ablations

The historical committee audit uses the same repeated evidence but a diferent aggregation rule. A judge is resolved when at least three of its four canonicalized overall labels agree. Three judges then use two-of-three voting. Unresolved comparisons receive $p ( A ) = 0 . 5$ when computing Brier and remain in its denominator. Resolved coverage is reported separately. The audit weights each comparison by its number of valid human votes divided by five, whereas the preference proxy gives each comparison equal weight; their reported losses should not be interpreted as a controlled comparison difering only in aggregation.

Eligible committees contain one judge from each vendor family and achieve at least 75% coverage. The pilot uses 320 comparisons; the remaining 1,552 form the holdout. Selection minimizes abstention-aware Brier, breaking ties by resolved-case Brier and then coverage. Table 6 reports all four eligible cross-family committees.

The pilot winner v1 falls to fourth on holdout. The paired holdout Brier diference (v1 minus $\mathrm { v 2 ) }$ is +0.0053 $( 9 5 \% \mathrm { C I } { : - 0 . 0 0 0 1 } \mathrm { t o } + 0 . 0 1 0 6 )$ . Among 10,000 bootstrap resamples, v2 ranks first 76.81% of the time. The interval includes zero, and v2 was selected after inspecting holdout; this is selection evidence rather than independent confirmation of v2. Furthermore, formula, algorithm, list, and vertical-text comparisons were concentrated in the pilot and are absent from holdout. On the full human set, descriptive macro Brier over the three special axes with at least 30 comparisons favors v1 (0.1268 versus 0.1324). Thus, the overall ordering does not establish universal superiority across structures. Gemini is present in every eligible committee, and family-specific preferences remain a potential confound.

Table 6 Historical cross-family committee audit. Pilot $n = 3 2 0 ;$ holdout $n = 1 , 5 5 2 .$ . Coverage is the fraction resolved. O, S, G, L, and T denote Opus, Sonnet, Gemini, Sol, and Terra. These discrete committees are separate from the deployed preference proxy.
<table><tr><td colspan="4">Pilot</td><td colspan="3">Holdout</td></tr><tr><td>Committee</td><td>Brier</td><td>Coverage</td><td>Rank</td><td>Brier</td><td>Coverage</td><td>Rank</td></tr><tr><td> $\overline { { \mathrm { S + G + T \ ( v { 1 } ) } } }$ </td><td>0.1258</td><td>77.81%</td><td>1</td><td>0.1293</td><td>78.99%</td><td>4</td></tr><tr><td> $_ { \mathrm { O + G + T } }$ </td><td>0.1302</td><td>78.44%</td><td>2</td><td>0.1263</td><td>79.90%</td><td>2</td></tr><tr><td> $_ \mathrm { S + G + L }$ </td><td>0.1369</td><td>79.06%</td><td>3</td><td>0.1267</td><td>79.45%</td><td>3</td></tr><tr><td> $\mathrm { O + G + L \ ( v 2 ) }$ </td><td>0.1404</td><td>79.06%</td><td>4</td><td>0.1240</td><td>80.35%</td><td>1</td></tr></table>

A separate nested ablation tests all 31 nonempty judge subsets using a learned blend of overall and axis evidence. It fits nonnegative, sum-to-one weights with ridge shrinkage toward equal weights; subset selection uses only inner folds, and loss is weighted by valid human votes. With 100% coverage on 1,872 comparisons, three judges yield Brier 0.0653 (0.0609–0.0699); two judges yield 0.0663 (0.0616–0.0713). The paired two minus-three diference is +0.0010 (−0.0002 to +0.0021), and the one-standard-error rule selects Gemini+Sol in all eight outer folds. Five judges are worse than three by $+ 0 . 0 0 2 5 \ ( + 0 . 0 0 0 9 \ \mathrm { t o \ + 0 . 0 0 4 1 } )$ . Intervals use 10,000 page-block resamples. These results concern an axis-aware model with diferent weighting and fitting, so they neither establish five judges as the minimum nor replace the frozen overall-only regression estimator. With no prespecified noninferiority margin, overlapping intervals do not prove equivalence.

## B.5 Deployment schedule and candidate versions

The fixed schedule contains 10,000 comparisons across 1,113 eligible pages (seed 20260831). Each of ten models appears 2,000 times: 35 model pairs occur 222 times and ten occur 223 times. It retains the 1,872 human-study page/model pairs and adds 8,128 pairs. Pages and document structures are not sampled equally.

Each comparison receives one output per judge in a single display order, totaling 50,000 outputs. Historical directions depend on completed-result availability, never on predicted winners; reuse may therefore preserve latency-related selection efects. New requests use the primary order. The initial deployment reuses 11,380 outputs and obtains 38,620 new ones. The final table has five outputs per comparison, with no missing or duplicated entries.

The reported leaderboard incorporates the subsequent candidate synchronization. Of 58 rerun page/model outputs touched by the schedule, 54 pass the existing visible-text and render-geometry checks; four retain their earlier candidates. The replacements afect 98 comparisons: 490 judge outputs are refreshed and 49,510 retained. Pair identities, quotas, orientations, and coeficients remain fixed. All ten point ranks are unchanged, although scores change; Sol moves from 154.34 to 154.70. This checks version sensitivity, not independent performance.

Twenty-one afected comparisons come from the human-study schedule; their earlier labels do not validate the replacement candidates. The final table therefore contains 1,851 unchanged human-study comparisons, 21 updated comparisons without new human labels, and 8,128 unlabeled additions. The original calibration set is unchanged. Neither its overlap with deployment nor the updated table supplies external human validation.

The initial deployment’s final 22 failures comprise 14 request-size failures and eight timeouts. Eighteen repaired calls use high-quality JPEG reference copies; one further reduces the long edge from 6,320 to 6,000 pixels (5.1%), preserving aspect ratio, to meet the visual-patch limit. Original references remain unchanged. These repairs are separate from candidate replacements.

## B.6 Ranking and uncertainty

For comparison i between models $a _ { i }$ and $b _ { i } ,$ the proxy outcome contributes $\hat { p } _ { i } ( A )$ preference points to $a _ { i }$ and $1 - \hat { p } _ { i } ( A )$ to $b _ { i }$ . The soft-preference column is the mean of these points over a model’s 2,000 scheduled appearances. It is not accuracy against human labels.

Bradley–Terry fitting estimates centered log-abilities $\theta _ { m }$ using

$$
q _ { i } = \sigma \big ( \theta _ { a _ { i } } - \theta _ { b _ { i } } \big ) , \qquad \operatorname* { m a x } _ { \theta } \sum _ { i } \left[ \hat { p } _ { i } ( A ) \log q _ { i } + \big ( 1 - \hat { p } _ { i } ( A ) \big ) \log ( 1 - q _ { i } ) \right] - \frac { 1 0 ^ { - 3 } } { 2 } \sum _ { m } \theta _ { m } ^ { 2 } .
$$

The small $\ell _ { 2 }$ term stabilizes the ranking fit; it is not a penalty on the OLS coeficients. Centering sets the mean ability to zero. Reported scores are $4 0 0 \theta _ { m } / \log ( 1 0 )$ , an Elo-like relative scale. A negative score indicates below-average estimated ability within this comparison pool, not failed reconstruction.

For the deployment leaderboard, 2,000 resamples draw 1,113 page blocks with replacement (seed 20260831). Each selected page carries all of its scheduled comparisons. The Bradley–Terry model is refitted for every resample; 2.5th and 97.5th percentiles form the score and rank intervals. These are marginal intervals and do not constitute simultaneous tests of every model pair. “First in every bootstrap sample” describes the observed resampling frequency, not certainty under all sources of error.

Proxy alignment uses 5,000 page-block resamples stratified by annotation block (seed 20260830), with fixed out-of-fold predictions; the committee audit uses 10,000 similarly stratified resamples. Within each stratum, comparisons from the same page remain together. Bootstrap fixes proxy coeficients and predictions, omitting renewed judge sampling, regression fitting/selection, candidate generation/cleaning, and systematic bias. Alignment is internal to the human study; the expanded leaderboard provides no full-scale human Accuracy, Recall, or F1.

## C Case Study

The reason for performing poorly on AmbiHTMLBench could vary. Some typical cases are listed below.

![](images/1508a4e82fd6d11ba455fe5a3fca43a3a4b86edd5285b0c1feb71ba3128a5437.jpg)  
Figure 7 Text occlusion. Claude Opus 5.5 keeps the section heading between the two tables. In GPT-6.1 Sol, the heading overlaps the preceding table, disrupting the separation between sections. This makes the document dificult to read visually in terms of layout.

![](images/8dc486835bbe7a921e0d06f655d0971c9308764f4498191893982e4676391deb.jpg)  
Figure 8 Table structure degradation. GPT-6.1 Sol aligns the final column header with the table body. In Claude Opus 5.5, the header extends beyond the right edge of the body, leaving the column misaligned, and the header of the second-to-last column has not been properly recognized.

![](images/25329fd606d4c32eb7f795b0593f98f2d5a5be79f07a9cd09e59a6cd5afd597b.jpg)  
Figure 9 Text reproduction error. Claude Sonnet 5.5 substitutes a diferent character; The distinctive color and doodle style of social media notes have also been eliminated. In this case, GPT-6.1 Sol towers over.

![](images/4618c6b0d00fae7197e89aca3209904bc469eb367622804b7e958614c9be5e3e.jpg)  
Figure 10 Information loss. Gemini 3.8 Flash retains the tuition and examination information in the highlighted row. Claude Sonnet 5.5 leaves both corresponding cells empty.

![](images/8520d49432fe93de6afb98a7d47ddb0b97a00651812270ce525d74c0f4b0aa88.jpg)  
Figure 11 Formula rendering error. Claude Opus 5.5 renders the highlighted calculation as two-dimensional fractions. Qwen3.8-Max displays raw formula commands in several lines, obscuring the mathematical structure.

![](images/a63f9cb8d3113e2862846047ab8abd493fa6f2bd8eff358daf16e954e5ae24c8.jpg)  
Figure 12 Code formatting error. Gemini 3.8 Flash retains the line breaks and indentation of the reference code block. Qwen3.8-Max merges separate statements and loses indentation, making the code structure harder to read.