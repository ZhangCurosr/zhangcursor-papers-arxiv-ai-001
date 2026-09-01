# CM2: Multimodal Cultural Reasoning via an Integrated Multi-Agent Framework

Qi Li<sup>1,2,⋆</sup>, Zhaojie Kang<sup>2,3,⋆</sup>, Yingjie He<sup>2</sup>, Zheng Lin<sup>2</sup>, Hao Zhang<sup>2</sup>, Guangxin Wu<sup>2</sup>, Yan Gong<sup>2</sup>, Rong Fu<sup>2</sup>, and Jianyuan Ni<sup>4</sup>

<sup>1</sup> Lanzhou University, Lanzhou, China <sup>2</sup> Peking University, Beijing, China Taiyuan University of Technology, Taiyuan, China 4 Independent Researcher liq23@lzu.edu.cn, kangzj980@outlook.com

Abstract. Multimodal Large Language Models (MLLMs) have shown remarkable success in STEM domains, where progress is often driven by vertical, step-by-step deduction under relatively stable symbol systems. Their horizontal, interdisciplinary cultural reasoning, however, remains underexplored. We propose CM2, a multi-agent framework grounded in the cognitive pathway of human cultural interpretation. CM2 integrates multimodal perception, retrieval-augmented generation, networked reasoning, gated fusion, and reward-driven feedback. Experiments on CM2D across multiple MLLM backbones show consistent gains over CoT and typical reasoning paradigms; ablations validate each module’s contribution, and conflict analyses confirm genuine cross-modal arbitration.

Keywords: MLLMs · Multi-Agent Framework · Cultural Reasoning

## 1 Introduction

Multimodal Large Language Models (MLLMs) [1] have rapidly evolved into general-purpose assistants capable of reasoning over mixed visual–textual inputs. Their strongest gains are observed in structured STEM benchmarks [6], which typically emphasize vertical reasoning: domain-internal, step-by-step deduction under relatively stable symbolic rules and well-defined objectives.

In contrast, cultural understanding tasks—such as interpreting artworks, artifacts, or historically situated imagery—require a fundamentally diferent regime. Building on the humanities and social sciences (HSS) perspective articulated in HSSBench [7], we characterize this regime as horizontal reasoning: integrating heterogeneous evidence across modalities and disciplines while resolving ambiguity among multiple plausible interpretations [19]. Cultural reasoning is inherently context-sensitive: the same motif may convey distinct meanings across periods, styles, or traditions, making evidence attribution and conflict resolution central rather than optional.

![](images/29535c74d256891e223b80aafab4d20572906083d2edc8706f723255a41bb6b2.jpg)  
Fig. 1. Overview of CM2. The framework orchestrates multimodal perception (MP), retrieval-augmented generation (RAG), networked reasoning (NR), gated fusion and synthesis (GFS), and reward-driven feedback (RDF) for iterative cultural reasoning.

Despite its importance, current cultural evaluations remain limited. Many tasks reduce interpretation to simplified QA or recognition-style decisions [3], under-testing interdisciplinary integration and ambiguity handling. As a result, MLLMs often produce fluent yet weakly grounded explanations, over-trust superficial visual cues, or generate culturally plausible but unsupported narratives [20].

To address these challenges, we propose CM2, an integrated multi-agent framework for structured cultural reasoning, grounded in the cognitive pathway of human aesthetic appreciation [8]. CM2 produces complementary evidence from perception, retrieval, and external verification, arbitrates cross-modal conflicts through gated fusion, and iteratively refines outputs under cultural reasoning criteria.

Complementing the framework, we construct CM2D, a curated multimodal cultural dataset probing horizontal reasoning rather than surface recognition, built via a hybrid expert–agent pipeline with 442 expert-verified test samples and 3,715 retrieval entries.

## Our contributions are:

– Framework. We introduce CM2, a five-module framework that explicitly targets horizontal, cross-modal cultural reasoning through parallel evidence generation, conflict-aware fusion, and feedback-driven refinement.

– Dataset. We develop CM2D, a curated multimodal cultural dataset constructed via an expert–agent pipeline, emphasizing context-sensitive interpretation and multimodal dependency.

– Evaluation. Experiments show consistent gains over CoT and typical reasoning baselines across multiple backbone MLLMs; ablations and conflict analyses confirm the gains stem from genuine cross-modal arbitration.

## 2 Related Work

## 2.1 MLLMs and Multimodal Reasoning

Recent MLLMs have achieved strong performance in multimodal instruction following and reasoning [1]. Benchmarks such as multimodal math and STEM reasoning [6] have driven progress in visual encoders, scaling, and prompting. However, these settings predominantly test vertical deduction under fixed symbol systems. Cultural interpretation instead requires integrating heterogeneous cues, handling ambiguity, and reasoning under context shifts. This gap motivates specialized frameworks that go beyond generic prompting or retrieval [4,5,9,13, 15, 22].

## 2.2 Cultural and HSS-Oriented Evaluation

Cultural evaluation remains comparatively underdeveloped. Existing eforts often simplify cultural tasks into direct QA or recognition-like decisions [3], which may miss interpretive nuance. HSSBench [7] provides a complementary diagnostic perspective by systematically characterizing how large models struggle with humanities and social sciences tasks: shallow symbolic associations, ungrounded narratives, and weak evidence reconciliation. While HSSBench is primarily language-centric, its failure modes become more severe in multimodal cultural settings, where evidence can be distributed across style, iconography, and historical context. CM2 is designed as a direct response: producing structured evidence, resolving conflicts, and refining outputs under explicit cultural criteria.

## 3 Method

## 3.1 Cognitive Foundations of Horizontal Reasoning

We characterize cultural interpretation as horizontal reasoning: unlike the vertical, monotonic deduction of STEM, it demands actively integrating and arbitrating heterogeneous evidence that often conflicts across modalities and contexts. This regime mirrors human aesthetic cognition. Leder and Nadal’s [8] five-stage model describes interpretation not as linear deduction but as a dynamic cycle of perceptual analysis, schema activation, classification, cognitive mastering (i.e., conflict arbitration), and evaluation. Drawing on this architecture, CM2 maps each cognitive stage to a dedicated module (Figure 1): multimodal perception (MP) extracts stylistic cues; retrieval-augmented generation (RAG) activates cultural schemas; networked reasoning (NR) verifies uncertain claims externally;

gated fusion (GFS) arbitrates conflicting evidence via source-specific weighting; and reward-driven feedback (RDF) evaluates and refines the output under culturally grounded criteria.

## 3.2 Task Formulation

We formalize a task as $\mathcal { T } = ( I , Q , O , A _ { \mathrm { g t } } )$ , where I is an image, $Q$ a query, $O =$ $\big \{ o _ { 1 } , \dotsc , o _ { K } \big \}$ a candidate option set, and $A _ { \mathrm { { g t } } }$ is the ground-truth answer. CM2 generates three evidence streams, namely perceptual $\left( E _ { \mathrm { m p } } \right)$ , retrieved $( E _ { \mathrm { r a g } } )$ and verified external $( E _ { \mathrm { n r } } )$ , which are fused and refined into a final prediction:

$$
( A ^ { * } , R ^ { * } ) = \mathrm { R D F } \big ( \mathrm { G F S } ( E _ { \mathrm { m p } } , E _ { \mathrm { r a g } } , E _ { \mathrm { n r } } ) , \mathcal { T } \big ) .\tag{1}
$$

## 3.3 Multimodal Perception (MP)

MP uses the backbone VLM to extract culturally salient perceptual cues $E _ { \mathrm { m p } } =$ $\{ D _ { \mathrm { m p } } , \mathcal { Q } A _ { \mathrm { m p } } , S _ { \mathrm { m p } } \} \colon D _ { \mathrm { m p } }$ describes style, composition, and culturally relevant visual attributes $( \mathrm { e . g . }$ , brushwork, color palette, spatial organization); $\mathcal { Q } \mathcal { A } _ { \mathrm { m p } }$ is a structured $\mathrm { Q A }$ block probing motif identity, likely period, and region of origin; and $S _ { \mathrm { m p } }$ is a one-sentence summary of the task-relevant semantic focus.

## 3.4 Retrieval-Augmented Generation (RAG)

RAG anchors interpretation by retrieving culturally relevant precedents from a curated knowledge base $( \mathrm { K B _ { r a g } , 3 , 7 1 5 }$ image–text entries; Sec. 4). Visual similarity $\mathrm { { s i m } _ { \mathrm { { v i s } } } }$ uses CLIP embeddings and textual similarity sim $\mathrm { _ { t e x t } }$ uses a sentencetransformer encoder, both L2-normalized cosine similarities FAISS-indexed; they are combined with $\alpha { = } 0 . 5$ as:

$$
\begin{array} { r } { \mathrm { s i m } ( \mathcal { T } , \mathcal { T } _ { j } ) = \alpha \cdot \mathrm { s i m } _ { \mathrm { v i s } } + ( 1 - \alpha ) \cdot \mathrm { s i m } _ { \mathrm { t e x t } } , } \end{array}\tag{2}
$$

and the top- $K { = } 5$ entries are concatenated as structured context for downstream fusion.

## 3.5 Networked Reasoning (NR)

When symbolic claims from MP and RAG remain uncertain, NR performs iterative external verification via web search [20]. At each round t, the MLLM converts the unresolved claim into a query $\bar { q ^ { ( t ) } }$ , issues it to a search $\mathrm { A P I }$ , and accumulates the top-r=5 snippets into a bufer $h ^ { ( t ) }$ (initialized $h ^ { ( 0 ) } = \varnothing )$

$$
q ^ { ( t ) } = f _ { \mathrm { q u e r y } } ( \mathcal { T } , h ^ { ( t - 1 ) } ) ,\tag{3}
$$

$$
h ^ { ( t ) } = \mathrm { c o n c a t } \big ( h ^ { ( t - 1 ) } , \mathrm { S e a r c h } ( q ^ { ( t ) } ) \big ) .\tag{4}
$$

The loop runs for at most $T _ { \mathrm { m a x } } { = } 3$ iterations, and all accumulated evidence is summarized into a verification report $E _ { \mathrm { n r } } = f _ { \mathrm { s u m m } } ( h ^ { ( T ) } )$ .

## 3.6 Gated Fusion & Synthesis (GFS)

Because the three sources may conflict, GFS performs relevance-weighted arbitration rather than naive aggregation [21]. For each source $x \in \{ \mathrm { m p , r a g , n r } \}$ the backbone VLM scores the evidence on direct relevance, factual accuracy, and usefulness for discriminating options (each 1–10), averaged into a raw score $g _ { x } .$ Scores are normalized by softmax and used to weight the source embeddings $\mathbf { e } _ { x } \colon$

$$
\tilde { g } _ { x } = \frac { \exp ( g _ { x } / \tau _ { g } ) } { \sum _ { x ^ { \prime } } \exp ( g _ { x ^ { \prime } } / \tau _ { g } ) } ,\tag{5}
$$

$$
\mathbf { e } _ { \mathrm { f u s e d } } = \sum _ { x } \tilde { g } _ { x } \mathbf { e } _ { x } ,\tag{6}
$$

with $\tau _ { g } { = } 0 . 7$ controlling the sharpness of weighting. A synthesis prompt $f _ { \mathrm { s y n t h } }$ then generates the fused answer $A _ { \mathrm { g f s } }$ and rationale $R _ { \mathrm { g f s } }$ from the three evidence texts (with their gate weights) and the task, so that culturally grounded evidence can override superficial cues when conflicts arise.

## 3.7 Reward-Driven Feedback (RDF)

RDF evaluates the fused output against six criteria—evidence integration (multiple sources cited and reconciled), visual grounding (anchored in specific visual details), logical flow, option diferentiation (each option distinctly assessed), conclusion strength, and completeness (required output structure)—each scored 1– 10 by the backbone VLM and averaged into s¯ [16, 18]. If $\bar { s } < \tau { = } 0 . 6$ , the VLM summarizes the weakest dimensions into targeted feedback, injected into the next NR and GFS iteration; after at most two rounds, the answer with the highest s¯ is selected $( A ^ { * } = \arg \operatorname* { m a x } _ { t } \bar { s } ^ { ( t ) } ,$ ). Unlike generic self-refinement, RDF pinpoints which cultural dimension is weak.

## 4 Dataset

We introduce CM2D, a multimodal cultural dataset designed to evaluate horizontal and context-sensitive reasoning beyond surface recognition. CM2D contains 4,157 image–question pairs: 442 expert-verified test samples and 3,715 retrieval entries $\left( \mathrm { K B } _ { \mathrm { r a g } } \right)$ . Each instance is a 4-way multiple-choice question with one correct answer and three culturally plausible distractors, covering fine-grained reasoning skills such as symbolism, style, period/region attribution, medium conventions, and contextual interpretation.

Taxonomy and Visual Diversity. To counter Eurocentric bias and ensure diverse visual representation, CM2D is structured into four coarse categories and thirteen fine-grained taxonomies (Figure 2). This categorization is principally based on the primary cultural function and social medium of each artifact. Acknowledging the inherent intersectionality of the humanities (e.g., a religious fresco serving as both fine art and spiritual iconography), we assign the most salient label to each instance to facilitate structured evaluation. The four core categories are:

(i) Visual & Fine Arts: High-aesthetic value creations including Asian Fine Arts, Western Fine Arts, and Court & Elite Art.

(ii) Material Culture & Applied Arts: Objects with physical spatial and utilitarian presence, covering Everyday Material Culture, Decorative Crafts, and Civic & Architectural Heritage.

(iii) Spiritual & Indigenous Traditions: Items reflecting religious or grassroots beliefs, including Religious Iconography, Vernacular Art, and Indigenous Traditions.

![](images/9f9ace029b5a686f0f42c885fc660fc26e09beac5aaae6a9732d19e10c5e2bfe.jpg)

(iv) Cultural Practices & Performance: Dynamic human-centric traditions, spanning Regional Customs, Intangible Heritage, Performing Arts, and Global Culture.

Fig. 2. Taxonomy of CM2D: four coarse categories (inner ring) and thirteen fine-grained sub-categories (outer ring).

Hybrid Construction Pipeline. Data are sourced via a rigorous hybrid expert– agent pipeline to ensure both diversity and depth. This pipeline operates in three phases. Phase I: Multimodal Aggregation. Based on expert-provided seeds, the agent retrieves authoritative materials (e.g., verified museum repositories, opensource textbooks) to gather high-information-density text and images. Phase II: Semantic Drafting. Specialized LLM agents parse the textual background to extract key symbolic attributes and dynamically generate culturally plausible distractors. Phase III: Dual-Tier Validation. All drafted items undergo automated redundancy checks followed by meticulous human expert review, where experts refine skill tags and approve the final answer keys.

Multimodal dependency check. To discourage unimodal shortcuts, we apply screening under two counterfactual settings: (i) text-only, removing the image while keeping (Q, O); and (ii) image-only, keeping I while replacing Q with a generic prompt and shufling O [11]. Items that remain trivially solvable under either setting are revised (e.g., tightening distractors or removing leaked cues) or discarded. A final expert pass confirms that decisive evidence requires cross-modal grounding rather than pure memorization or object spotting.

Dataset Composition and Quality Assurance. We intentionally prioritize data quality and expert-verified density over sheer scale. Within the 3,715 $\mathrm { K B } _ { \mathrm { r a g } }$ entries, approximately 40% are directly expert-authored, while 60% are agentdrafted and subsequently expert-verified. Conversely, to ensure the highest level of academic rigor and eliminate noisy evaluations, the 442-sample test set is 100% human-curated and cross-validated by independent experts. During the validation phase, any candidate question lacking genuine cultural ambiguity was discarded.

Table 1. Accuracy (%) on the CM2D test set. The best result in each row is bold; the best baseline (excluding CM2) is underlined.
<table><tr><td colspan="6">Model Baseline (CoT) Self-Refine CoT-SC Agent Debate CM2</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>23.98</td><td>37.56</td><td>38.91</td><td>39.59</td><td>44.34</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>29.41</td><td>35.52</td><td>35.97</td><td>33.26</td><td>46.83</td></tr><tr><td>llava-onevision-7b</td><td>23.53</td><td>34.40</td><td>40.50</td><td>43.21</td><td>50.68</td></tr><tr><td>Qwen2.5-VL-7B (SFT)</td><td>38.46</td><td></td><td></td><td></td><td>47.06</td></tr><tr><td>GPT4.1-mini</td><td>41.63</td><td>49.10</td><td>54.30</td><td>54.07</td><td>55.20</td></tr></table>

## 5 Experiments

We evaluate CM2 on CM2D and report 4-way multiple-choice accuracy on the 442-sample expert-verified test set, using Qwen2.5-VL-3B-Instruct, Qwen2.5- VL-7B-Instruct [1], llava-onevision-7b [10], and GPT4.1-mini [14] as backbones. As a reference, Baseline performs single-pass Chain-of-Thought (CoT) prompting [19] with greedy decoding and no retrieval, external search, or iterative refinement. Beyond CoT, we further compare three typical reasoning paradigms under the same backbone and identical test set: Self-Refine [12] (two critiqueand-revise rounds), CoT-SC [17] (N=5 chains with temperature 0.7 and majority voting), and Agent Debate [2] (two proposer–critique rounds). CM2 uses the same backbone across all modules with deterministic decoding (temperature 0) and the hyperparameters in Sec. 3; prompts and budgets are held fixed across backbones for comparability.

## 5.1 Main Results

Table 1 reports the main accuracy results on CM2D. CM2 yields consistent improvements across all backbones, suggesting that horizontal cultural reasoning benefits from explicit evidence integration and conflict arbitration rather than scaling alone. Notably, llava-onevision-7b more than doubles accuracy (23.53% → 50.68%), while the stronger GPT4.1-mini still improves (41.63% → 55.20%), indicating that gains are systematic rather than limited to weaker models.

Crucially, CM2 also maintains a significant edge over typical reasoning paradigms. Taking llava-onevision-7b as an example, CoT-SC and Agent Debate achieve 40.50% and 43.21% respectively, notably lagging behind CM2’s 50.68%. This validates our core hypothesis: generic multi-hop reasoning or unguided debate often struggles with horizontal cultural tasks, as they lack the specific mechanisms to resolve explicit cross-modal conflicts via external grounding. Across backbones, CM2 maintains a clear margin over the best baseline (4.75–10.86pp), narrowing to 0.90pp on the strongest backbone GPT4.1-mini.

## 5.2 Ablation Studies

To understand which components drive the improvements, we ablate each module while keeping all other components unchanged. Table 2 shows that every module provides complementary value. MP contributes fine-grained stylistic and compositional cues; RAG provides precedent grounding for iconography and culturally conventional symbols; NR reduces historically plausible but unsupported narratives; GFS is crucial when evidence sources disagree; and RDF improves targeted correction when the initial synthesis is weak.

Table 2. Ablation results (%) on the CM2D test set. The drop from removing each module quantifies its contribution; the largest drop per column is underlined.
<table><tr><td>Ablation</td><td>Qwen2.5-VL-7B</td><td>llava-onevision-7b Qwen2.5-VL-3B</td></tr><tr><td>CM2 (full)</td><td>46.83</td><td>50.68 44.34</td></tr><tr><td>w/o MP</td><td>37.78</td><td>44.12 42.99</td></tr><tr><td>w/o RAG</td><td>40.50</td><td>39.59 39.37</td></tr><tr><td>w/o NR</td><td>42.99</td><td>42.53 40.27</td></tr><tr><td>w/o GFS</td><td>40.27</td><td>39.82 43.21</td></tr><tr><td>w/o RDF</td><td>45.02</td><td>44.57 43.21</td></tr></table>

![](images/b484837aa2c04f9f769b1bf182f1196fbb696ef8e2320c4ebc506c4a8c18dc1e.jpg)  
Fig. 3. Accuracy on the Conflict Subset:CM2’s gated arbitration substantially outperforms Majority Voting, CoT-SC, and Agent Debate.

## 5.3 Fine-Grained Evaluation under Evidence Conflict

While the ablation study (Sec. 5) confirms the necessity of each module, this section investigates the behavioral patterns of CM2 when faced with cross-modal evidence conflict. By comparing the independent predictions of MP, RAG, and NR before fusion, we dynamically partition the test set into two mutually exclusive subsets: the Consensus Subset (where all three sources agree) and the Conflict Subset (where at least one source disagrees). We analyze these subsets on two representative backbones (Qwen2.5-VL-7B and llava-onevision-7b).

Consensus as a strong baseline. Before addressing conflicts, we note that on the complementary Consensus Subset (45.7%–46.2% of samples), CM2 already leads across both backbones (+4.9–11.4pp over the best baseline). This confirms that the enrichment from MP and RAG provides consistent, foundational benefits even when evidence is perfectly aligned.

<table><tr><td colspan="2">Case: Lotus Form (CM2D)</td></tr><tr><td></td><td>Type: Symbolic &amp; stylistic Question: What characteristics are shown in the form of the lotus flow- ers in the artwork? Options: A. Vibrant vitality and growth B. The withered state of autumn X C. A sense of etherealness filled with Eastern philosophy X D. The desolation of winter X</td></tr><tr><td colspan="2">Baseline (CoT): B X Reasoning: The lotus flowers and leaves point downwards, painted with heavy, dark ink strokes. This drooping shape indicates decay and au- tumn&#x27;s withered state. Thus, it depicts the end of a life cycle rather than growth. ◆ Self-Refine: C X Reasoning: [Critique] The initial thought is too literal. [Refine- ment] Eastern ink paintings prioritize philosophical emptiness over physical seasons; drooping lines express etherealness. CoT-SC (N=5): B X Reasoning: B(3), C(1), D(1). Despite temperature sam- pling, all paths are misled by the superficial cue of bent stems, yielding a confident but incorrect consensus on autumnal decay. ◆ Agent Debate: B X Reasoning: Agent A argues dark ink shows autumn decay (B); Agent B argues barren background suggests winter desolation (D). Without external cultural anchors, they default to superficial decay cues and choose B.</td></tr><tr><td colspan="2">CM2 (Ours): A√ Arbitrated Reasoning Process: MP: Stems show upward tension despite downward-facing leaves. RAG: Retrieved critiques define this “drooping but unbroken&quot; style as inner vigor. NR: External sources confirm the lotus symbolizes resilience in this artist&#x27;s context. GFS: Drooping is an expressive technique, not literal decay. Grounded evidence overrides visual bias, confirming vitality.</td></tr></table>

Fig. 4. Qualitative case: Baselines overfit visual cues or hallucinate stereotypes, while CM2 arbitrates evidence to recover vitality (A). Reasoning texts are human-edited summaries, not verbatim model outputs.

Conflict is pervasive and challenging. The core arbitration challenge resides in the Conflict Subset, which constitutes the majority of test samples (54.3% for Qwen2.5-VL-7B-Instruct and 53.8% for llava-onevision-7b), confirming that evidence-source disagreement is pervasive in cultural tasks. On this challenging subset, the single-pass CoT baseline scores only 31.7% (Qwen) and 27.3% (llava). A simple Majority Vote among the pre-fusion agents improves this to 41.2% and 43.3%, while external baselines (CoT-SC: 40.4%, 46.2%; Agent Debate: 38.8%, 44.5%) ofer only incremental gains. In contrast, CM2’s relevanceweighted gated fusion reaches 50.0% and 53.4%—a substantial +18.3–26.1pp over the CoT baseline and +7.2–9.9pp over the best competing method (Figure 3). This demonstrates that passive aggregation, sampling, and debate each leave a substantial gap: only explicit retrieval grounding with conflict-aware gating efectively resolves cross-modal evidence conflict.

## 5.4 Qualitative Case Study

Figure 4 presents a qualitative case study that instantiates common HSS-style failure modes. While typical reasoning paradigms (CoT, Self-Refine, CoT-SC, Agent Debate) either overfit to superficial visual cues (drooping implies decay) or hallucinate generic stereotypes (Eastern emptiness), CM2 efectively integrates perceptual style with retrieved cultural knowledge to accurately derive the underlying symbolic meaning (resilience and vitality).

## 6 Conclusion

We presented CM2, a multi-agent framework that targets horizontal cultural reasoning through parallel evidence generation, retrieval grounding, external verification, gated conflict arbitration, and reward-driven refinement. We introduced CM2D, a curated multimodal dataset emphasizing context-sensitive interpretation with multimodal dependency. Across multiple backbones, CM2 improves substantially over CoT and typical reasoning paradigms (Self-Refine, CoT-SC, Agent Debate), with ablations confirming each module’s contribution and conflict analyses validating genuine cross-modal arbitration.

## References

1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., et al.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

2. Du, Y., Li, S., Torralba, A., Tenenbaum, J.B., Mordatch, I.: Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325 (2023)

3. Gao, J., Xuan, R., Kang, Z., Liao, D., Huang, W., Huang, Z., Xu, Y., Qin, B., He, Z., Yang, X., et al.: Laobench: A large-scale multidimensional lao benchmark for large language models. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 23727– 23743 (2026)

4. Jiang, E.H., Li, L., Sun, R., Liang, X., Li, Y., Wu, Y., Luo, H., Li, H., Zhang, Z., Kang, Z., et al.: Agent q-mix: Selecting the right action for llm multi-agent systems through reinforcement learning. arXiv preprint arXiv:2604.00344 (2026)

5. Kang, Z., Gong, J., Chen, Q., Zhang, H., Liu, J., Fu, R., Feng, Z., Wang, Y., Fong, S., Zhou, K.: Multimodal multi-agent empowered legal judgment prediction. In: ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 12202–12206. IEEE (2026)

6. Kang, Z., Gong, J., Hu, W., Yin, S., Jiang, K., Fang, Z., He, Y., Meng, C., Fu, R., Chen, D., et al.: Quanteval: A benchmark for financial quantitative tasks in large language models. arXiv preprint arXiv:2601.08689 (2026)

7. Kang, Z., Gong, J., Yan, J., Xia, W., Wang, Y., Cheng, Z., Cao, W., Wang, Z., Feng, Z., Ding, H., et al.: Hssbench: Benchmarking humanities and social sciences ability for multimodal large language models. In: International Conference on Learning Representations. vol. 2026, pp. 74664–74719 (2026)

8. Leder, H., Nadal, M.: Ten years of a model of aesthetic appreciation and aesthetic judgments: The aesthetic episode—developments and challenges in empirical aesthetics. British Journal of Psychology 105(4), 443–464 (2014)

9. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., tau Yih, W., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented generation for knowledge-intensive nlp tasks (2021), https://arxiv.org/abs/2005.11401

10. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., Li, C.: LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024), https://arxiv.org/abs/2408.03326

11. Li, K., Shi, Z., Hu, M.J., Jin, Y.X., Yan, X., He, F., Wu, X.: Consistent focus: Mitigating permutation bias in large language models through attention weight averaging. Expert Systems with Applications 294, 128726 (2025)

12. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegrefe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B.P., Hermann, K., Welleck, S., Yazdanbakhsh, A., Clark, P.: Self-refine: Iterative refinement with self-feedback (2023), https://arxiv.org/abs/2303.17651

13. Meng, C., Feng, P., Fu, R., Lee, H.L., Du, X., Kang, Z., Zhang, Z., Zhou, W., Ouyang, C., Gan, Z.: Group cognition learning: Making everything better through governed two-stage agents collaboration. arXiv preprint arXiv:2605.00370 (2026)

14. OpenAI: Introducing gpt-4.1 in the api. https://openai.com/index/gpt-4-1/ (Apr 2025), [Online; accessed 17-Sep-2025]

15. Qian, J., Kang, Z.: " penny wise, pixel foolish": Bypassing price constraints in multimodal agents via visual adversarial perturbations. In: Findings of the Association for Computational Linguistics: ACL 2026. pp. 16059–16073 (2026)

16. Shi, Q., Kang, Z., Zhou, Y., Weng, D., Wu, Y.: Spader: Step-wise peer advantage with diversity-aware exploration rewards for multi-answer question answering. arXiv preprint arXiv:2606.00593 (2026)

17. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2023)

18. Wang, Y., Gao, S., Liu, J., Jiang, S., Haoxiang, X., Zhang, X., Kang, Z., Wang, Y., Liu, Z.: Beyond n-grams: A hierarchical reward learning framework for clinicallyaware medical report generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 33719–33727 (2026)

19. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., Zhou, D.: Chain-of-thought prompting elicits reasoning in large language models (2023), https://arxiv.org/abs/2201.11903

20. Yu, T., Yang, Y., Chai, S., Jinshuai, Z., Jin, H., Wang, H., Zhang, M., Luo, Z., Long, Y., Chen, X., et al.: When seeing is not believing–a benchmark for searchgrounded video misinformation detection. arXiv preprint arXiv:2606.04098 (2026)

21. Zheng, H., Shi, Z., Yi, P.: Medcoact: confidence-aware multi-agent collaboration for complete clinical decision. In: 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). pp. 4525–4528. IEEE (2025)

22. Zheng, L., Zhang, J., Chen, C., Wang, C., Li, H., Li, Y., Mao, Y., Yan, S., Song, Z., Feng, Z., et al.: What should i cite? a rag benchmark for academic citation prediction. In: Proceedings of the ACM Web Conference 2026. pp. 1852–1863 (2026)