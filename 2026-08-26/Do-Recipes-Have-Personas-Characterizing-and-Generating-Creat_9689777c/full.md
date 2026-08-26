# Do Recipes Have Personas? Characterizing and Generating Creator Style in Attributed Procedural Graphs

Lei Jiang

Microsoft, Redmond WA 98052, USA

lionelchange@hotmail.com

Abstract. While large language models (LLMs) possess vast zero-shot procedural knowledge, their tendency to produce homogenized logic often obscures the unique, idiosyncratic execution processes of individual human creators. In this paper, we investigate the computational discovery of procedural personas from unstructured data. To achieve this, we introduce ViralRecipesTrans, a new dataset of procedurally aligned execution flow graphs extracted from popular culinary video transcripts and explicitly mapped to specific creators. We formulate procedural stylometry as a graph learning and process discovery task, revealing a fundamental duality: while traditional lexical classifiers overfit via semantic leakage, discrete topological metrics successfully capture the rigid physical constraints of a creator’s workflow. Building upon this characterization, we extend our framework into a novel generative task—predicting a creator’s exact structural execution graph for unseen dishes. We expose a fundamental dichotomy in style generation between global macro-planning and local structural execution. Our results demonstrate that few-shot LLMs dominate semantic assignment but sufer from persistent macroplanning deficits, whereas our structured two-stage model achieves superior topological control via rigid Markovian priors. Together, an ensemble approach to procedural generation combines the strengths from both sides, dynamically synthesizing global semantic reasoning with localized topological footprints to automate the discovery and generation of personalized workflows.

Keywords: procedural text understanding · large language models · computational stylometry · persona modeling · graph structure learning

## 1 Introduction

Procedural text is one of the most fundamental mechanisms for transmitting complex human knowledge, making it a longstanding testbed for natural language understanding. Historically, culinary recipes were a highly common domain in Natural Language Processing (NLP) research, serving as the primary medium to evaluate event tracking, state-change modeling, and structured sequence generation [22,4,20,6]. While recipe generation tasks are usually measured by linguistic quality with metrics like BLEU [24], strictly text-centric recipe datasets like [3] also provided the structural scale necessary for digging into the culinary aspect with the cooking steps decomposed and compared against each other.

This capability, however, exposes a profound contradiction in the perceived problem space. Anthropologically, canonical culinary knowledge is relatively bounded, comprising roughly 20,000 established global dishes <sup>1</sup>. From a pure knowledge-extraction perspective, this finite space superficially appears "solved" by LLMs, where a structured representation is reliably parsed from text in tools [21,12]. Yet, we observe a paradoxical, daily explosion of viral recipe videos and endless procedural variants for the exact same dishes. If the semantics of cooking is a solved problem, what drives this continuously expanding, highly engaged domain? The answer lies entirely in the human creator. While LLMs default to generating logically sound, homogenized outputs [2], they smooth over the very essence of human variance: the idiosyncratic structural footprints of a specific author’s workflow. It is this procedural stylometry—the latent personas governing how a task is physically approached, parallelized, and paced—that constitutes the true signal amidst the noise, a signal which can no longer be ignored if we aim to fully characterize human authorship.

In summary, culinary recipes possess several unique linguistic and structural properties that make them an ideal testbed for advancing computational stylometry given the duality of text and topology as recipes are not merely sequential text; they are fundamentally executable Directed Acyclic Graphs (DAGs). Also, the real-world execution of any single dish exhibits massive structural variation across diferent creators on top of the underlying causal physical and chemical constraints, making recipe data a unique environment for intra-class stylometry.

To investigate these fingerprints on both linguistics and structure, we introduce the ViralRecipesTrans (VRT) dataset<sup>2</sup> in Section 3, a novel corpus (along with procedurally aligned DAGs extracted) from the transcripts of viral culinary video creators. After a rigorous empirical analysis of procedural stylometry in Section 4, we consider the interesting generative task "creating a viral recipe with the given dish name and ingredients", where the dichotomy between global semantic macro-planning and local procedural execution is conceptualized in Section 5. Specifically, the following core research questions are addressed:

RQ1 (Lexical vs Topology in Procedural Personas): How are authorspecific stylometric signatures distributed across textual expression and structural topology, and to what extent can analyzing this duality overcome the overfitting of purely lexical classification?

RQ2 (Persona-Conditioned Graph Generation): How efectively can a generative framework predict the complete procedural graph of a recipe—including task ordering and structural ellipsis—conditioned solely on the target dish, base ingredients, and the creator’s characterized persona?

RQ3 (Global vs. Local Structural Modeling): How do the inherent trade-ofs between global semantic reasoning and local Markovian constraints manifest across diferent generative architectures, and can an adaptive ensemble framework successfully synthesize these complementary forces to maximize operational fidelity?

## 2 Related Work

## 2.1 Computational Stylometry and Persona Modeling

Historically, computational stylometry has relied on surface-level lexical features, such as N-grams [5] and TF-IDF [11], to determine authorship [29]. Because these discriminative methods severely overfit to vocabulary in constrained domains, recent work has explored structural signatures, utilizing neural models on linguistic dependency trees to withstand cross-topic variance [9]. Concurrently, persona modeling has evolved into a generative task. While models trained on conversational datasets frame personas as explicit semantic facts [8], text style transfer frameworks largely treat persona as a surface-level lexical overlay, modifying tone or vocabulary rather than underlying logic [10]. Even within the culinary domain, existing personalization eforts focus on modeling consumer semantic preferences, such as generating recipes with user-favored ingredients [19]. Our work departs from these paradigms. Because culinary procedures artificially compress semantic variance through physical logic, we introduce procedural stylometry. Rather than classifying explicit traits or generating lexical substitutions, we investigate the latent workflow signatures of creators, formulating persona modeling as a challenge of structural style transfer over execution graphs.

## 2.2 Controlled Generation and Style Transfer

The advent of Retrieval-Augmented Generation (RAG) [15] has increasingly positioned LLM as parametric knowledge bases [26], where models generate zeroshot contextual frameworks prior to reasoning [32]. While this paradigm excels at procedural extraction, enforcing specific topological structures during generation remains a persistent challenge. Existing frameworks [28,27,14] for controllable generation typically employ neural logic constraints [17] to ensure the output adheres to valid physical rules. However, these methods treat structural control purely as a mechanism for objective accuracy—ensuring a generated recipe is executable. In contrast, our work introduces subjective structural style transfer. Within NLP, structural style transfer has recently emerged via benchmarks like StylePTB [18] and methodologies such as optimal transport [23], requiring models to manipulate compositional ordering while preserving semantic intent [1]. Yet, adapting LLMs for this task is dificult; without explicit architectural intervention, zero-shot LLMs frequently collapse into structural copying, failing to induce the target style [13]. Our framework unites these two isolated research tracks.

## 3 The ViralRecipesTrans (VRT) Dataset: Construction and Statistics

To empirically investigate the phenomenon of procedural stylometry, the VRT dataset comprises video transcripts mapped to execution DAGs across culinary creators. While there exist real-world limitations on the total number of cooking influencers with qualified video formats (with valid transcripts and Englishlanguage narratives), we argue that it contains adequate persona information as the results in Sections 4 and 6 suggest. This dataset also presents a contrast to the standardized procedural text found in traditional recipe corpora.

## 3.1 Account Discovery and Filtering

To construct the dataset, we implemented an automated discovery pipeline using the YouTube Data API. Starting from layered seed queries (e.g., general cooking → food-type → cuisine-specific), the pipeline identified candidate channels and applied a rigorous multi-stage filter to ensure data density and procedural diversity:

Volume and Reach: Channels must possess a minimum of 10,000 subscribers and at least 50 uploaded videos.

– Format Constraints: At least 30% of recent uploads must be long-form content (greater than 3 minutes in duration).

– Repertoire Diversity: An LLM-based dish detection pass on video titles was required to identify ≥ 30 unique dishes per account, ensuring recipe diversity.

– Transcript Quality: Three randomly sampled videos per channel were required to yield English transcripts with suficient word counts and successfully compile into valid procedural graphs.

The combined dataset spans 97 unique accounts (with 3 overlapping between long and short formats), ranging from niche home cooks (14K subscribers) to major culinary channels (7.7M subscribers), totaling more than 5,000 videos.

## 3.2 Transcript Collection

For each qualifying account, we fetched English transcripts for up to 50 recipe videos per format, ranked by total view count. Transcripts were obtained via the youtube-transcript-api. To ensure pipeline stability and prevent connection hangs, requests were throttled with a 20–40 second delay and a 90-second signalbased timeout.

Each transcript was quality-checked by building a flow action graph via GPT-5.4 and verifying ≥2 cooking actions and ≥2 distinct ingredients. Low-quality transcripts were discarded automatically.

## 3.3 Descriptive Statistics

The VRT dataset establishes a robust environment for studying procedural style transfer. The global action vocabulary extracted from the transcripts comprises 570 unique cooking verbs and exhibits a heavy long-tailed distribution. The top five most frequent actions (add, mix, cook, stir, cut) account for 37% of all action tokens, while 47% of the per-account actions are hapax legomena (verbs used only once by that specific creator).

Detailed corpus and per-recipe statistics are provided in Table 1.

Table 1. VRT per-recipe textual and topological statistics
<table><tr><td>Metric</td><td>Long-form Short-form Overall</td><td></td></tr><tr><td>Cooking steps (mean ± std)</td><td>14.8 ± 10.6  $\overline { { 9 . 3 \pm 4 . 8 } }$ </td><td> $\overline { { 1 2 . 2 \pm 8 . 9 } }$ </td></tr><tr><td>Unique Ingredients (mean ± std)</td><td> $1 3 . 2 \pm 8 . 0$   $1 0 . 0 \pm 5 . 2$ </td><td> $1 1 . 7 \pm 7 . 0$ </td></tr><tr><td>Transcript length (words, mean)</td><td>1800 176</td><td>1028</td></tr><tr><td>Transcript length (words, median) 1310</td><td>153</td><td>299</td></tr><tr><td>Global action vocabulary</td><td>548 499</td><td>675</td></tr><tr><td>Action vocab per account (mean) 106</td><td>89</td><td>97</td></tr><tr><td>Hapax ratio (per-account mean)</td><td>40% 46%</td><td>43%</td></tr><tr><td>Auto-generated captions</td><td>65% 96%</td><td>80%</td></tr></table>

## 4 Detecting and Characterizing Recipe Personas

Before evaluating fine-grained authorship, we first establish that procedural topology captures the macro-level constraints of the medium. If structural graphs genuinely reflect operational behavior, they should natively diferentiate between foundational procedural archetypes: traditional written text (from food.com [19] and RecipeNLG [3]), narrative long-form and compressed short-form videos.

## 4.1 Archetype-Level Structural Classification

To evaluate macro-level procedural topologies, we formally define the extraction of structural archetypes. Let a recipe be represented as an attributed DAG $\mathcal { G } = ( \nu , \mathcal { E } , \mathcal { X } )$ . We define a purely topological projection function $\phi : \mathcal { G }  \mathbb { R } ^ { d }$ which strips the semantic matrix X and extracts d discrete graph metrics $( \mathrm { e . g . }$ depth-to-width ratio, branch rate, maximum layer depth). We then formulate a classification function $f _ { \theta } : \mathbb { R } ^ { d }  \mathcal { A }$ , mapping the topology to a discrete set of medium-constrained archetypes $\mathcal { A } = \{ a _ { \mathrm { t e x t } } , a _ { \mathrm { l o n g } } , a _ { \mathrm { s h o r t } } \}$ . Using a random forest classifier, we evaluate whether $\phi ( \mathcal G )$ contains suficient signal to isolate these archetypes. As detailed in Table 2, the model successfully separates the topological spaces, achieving an overall Macro F1 score of 0.783.

Unsurprisingly, the classifier perfectly isolates the traditional written archetype $\left( a _ { \mathrm { t e x t } } \right)$ from both video formats. Feature importance analysis reveals that this boundary is heavily governed by the graph’s depth and branching behavior, exposing a distinct cinematographic decomposition pattern. Written recipes exhibit shallow, highly parallelized graphs. Conversely, video-derived workflows are structurally elongated; creators deliberately fracture standard procedures into sequential, deep micro-actions to sustain visual narration, resulting in a massive increase in ingredient reuse (branch rate).

Table 2. Classification performance for the 3-class procedural archetype task. The perfect isolation of the written archetype confirms that cinematographic workflows possess a fundamentally distinct structural topology.
<table><tr><td>Archetype Class (A) Precision Recall F1-Score</td></tr><tr><td>Written Text  $\left( a _ { \mathrm { t e x t } } \right)$  0.999</td><td>0.999 0.999</td></tr><tr><td>Viral Long-Form  $\left( a _ { \mathrm { l o n g } } \right)$ </td><td>0.702 0.613 0.654</td></tr><tr><td>Viral Short-Form  $\left( a _ { \mathrm { s h o r t } } \right)$  0.656</td><td>0.740 0.696</td></tr><tr><td>Overall Metric</td><td>Score</td></tr><tr><td>Accuracy</td><td>0.784</td></tr><tr><td>Macro F1</td><td>0.783</td></tr><tr><td>Top Discriminative Features (φ)</td><td></td></tr><tr><td>1. Depth-to-Width Ratio (0.152)</td><td></td></tr><tr><td>2. Average Out-Degree (0.129)</td><td>3. Ingredient Branch Rate (0.128)</td></tr></table>

Furthermore, the topological projection captures the inherent constraints of short-form media. While short-form graphs $\left( a _ { \mathrm { s h o r t } } \right)$ share the deep, sequential framework of their long-form counterparts, they exhibit a heavy compression archetype. This compression alters not just the graph size, but the operational behavior itself, as short-form workflows drastically prune complex preparatory techniques in favor of linear, "dump-and-bake" assembly sequences. This robust structural separation confirms that procedural DAGs successfully capture the overarching physical constraints of a medium.

## 4.2 The Illusion of Lexical Stylometry

Before isolating a creator’s structural workflow, we first establish the baseline capabilities of traditional linguistic stylometry on procedural transcripts. Using a 50-class authorship attribution (classification) task, we evaluated a purely lexical model (character N-gram TF-IDF [29]) and a handcrafted stylometric model incorporating psycholinguistic features. Linguistic stylometry achieves remarkably high numerical performance. The combined model (with stratified 5-fold crossvalidation) achieves a Top-1 Accuracy of 0.9006, while the purely lexical TF-IDF model peaks at 0.9357, identifying several creators with perfect precision.

While our analysis uses raw transcripts rather than text distortion [30], which isolates personal writing style for better control, high discriminative accuracy still does not establish that the model has captured procedural style. Lexical features conflate repertoire (what a creator cooks) with procedure (how they cook it). This distinction is decisive for our generative objective in Section 5: there, the dish and its ingredients are supplied as input, so any signal derived from niche-specific vocabulary is unavailable by construction. Only structural signal transfers to unseen dishes.

![](images/ac3505326735709763b2881e7c3e31eb4c805edf3b1465f3ac9aa931bed77631.jpg)

![](images/77f74cba330f0b4b4c8548f8ffae05673731649099d3912e169a45de63cd2466.jpg)

![](images/6439131f7c671bcb34c42413eb03fd382bccaa1db4db8c36f4f4a5e5ad34fb99.jpg)  
Fig. 1. Scalability results as the number of classes increases

## 4.3 The Scalability of Procedural Personas

Before evaluating the scalability of individual personas, we must ensure our classifier is learning genuine operational behavior rather than memorizing domainspecific vocabulary. As demonstrated previously, lexical baselines artificially inflate performance by exploiting "lexical leakage"—memorizing the specific ingredients or linguistic quirks associated with a creator’s preferred cuisine. To rigorously eliminate this leakage, we strip our graphical representation down to a strict, 19-dimensional pure topological feature space, discarding all semantic node labels, action bigram transitions, and specific verb categories. Focusing on structural persona, we formulate a scalable authorship attribution task, varying the class space from K = 2 up to $K = 5 0$ creators, utilizing 5 randomized sampling trials per K to prevent dish-overfitting.

The results, visualized in Figure 1, reveal a fundamental divergence between rigid boundary classification and latent signal strength.

Sensitivity to Feature Dimensionality As the classification space scales, absolute Top-1 Accuracy and Macro F1 naturally decay. However, this downward trajectory is an artifact of the classification mechanism itself, not a degradation of the underlying persona signal. Cooking is governed by strict physical logic; as we crowd the feature space with 50 diferent workflows, the shared baseline mechanics (e.g., all chefs must apply heat after preparation) cause the rigid discriminative hyperplanes to overlap and shatter. Yet, when evaluated against a random baseline (1/K), the model’s relative predictive power actually accelerates as complexity increases. The growing delta between the empirical accuracy curve and the random baseline confirms that the model is isolating a genuine, persistent structural footprint rather than exploiting low-resource noise.

The Probabilistic Nature of Persona The true nature of this structural footprint is exposed by the Area Under the ROC Curve (AUC). As shown in the rightmost panel of Figure 1, while hard classification boundaries fail, the AUC remains remarkably flat and stable—hovering near 0.78 even at $K = 5 0$ . This stability proves that the procedural persona operates as a probabilistic ranking signal rather than a strict categorical boundary. Even when the model cannot draw a perfect geometric boundary around a creator’s specific style, it retains a deep enough mathematical understanding of their workflow to consistently rank them near the very top of a massive candidate distribution.

This empirical divergence formally dictates our subsequent methodology. Because human procedural variance exists as a robust distributional preference rather than a static classification boundary, discriminative classifiers ultimately hit a representational ceiling. To fully characterize, capture, and replicate the idiosyncratic routing of a specific creator, we also need to move forward onto generative modeling.

## 5 Modeling Procedural Variance for Persona Generation

Having established that procedural personas exist as latent structural variances rather than strict discriminative boundaries, we pivot to the generative challenge. To prove that we can successfully isolate and model a creator’s workflow, we must be able to autonomously reconstruct how that specific creator would physically approach an unseen recipe.

## 5.1 Task Formulation

We formally define the persona-conditioned generation task as a graph prediction problem. Let a canonical recipe be represented by its semantic dish name D and a set of raw ingredients I. Given a target creator persona $C ,$ our objective is to generate an execution graph $\hat { \mathcal G } _ { C } = ( \nu , \mathcal { E } )$ that replicates the exact physical workflow the creator would employ. Unlike standard conditional text generation (e.g., "rewrite this recipe in the style of C"), this task does not require predicting the creator’s spoken dialogue or vocabulary. Instead, the model must predict the sequence of action states and the precise routing of ingredients into those states. A successful generation $\hat { \mathcal { G } } _ { C }$ must reflect the creator’s distinct topological signature—matching their typical depth, branching patterns, and ingredient usage—while maintaining the physical constraints of the dish D.

Figure 2 shows the workflow of the generative task along with LLM’s role: in the preprocessing of the extraction of flow action graphs, LLM has proven to be robust. Meanwhile, with all the features deterministically constructed, LLM could help with world knowledge for an unknown dish.

## 5.2 Evaluation Paradigm: Physics over Text

Traditional text-generation tasks in the culinary domain heavily rely on linguistic overlap metrics like BLEU [24] or ROUGE [16]: for transcript-style text, its representational power is not as strong as shown in Section 4. To evaluate whether a generated graph $\hat { \mathcal G }$ accurately captures the creator’s persona, we abandon text-overlap metrics in favor of topological alignments against the creator’s LLM-extracted execution graph $\mathcal { G }$ as a given target (which has a high reliability). We introduce two primary metrics inspired by Dice similarity coeficient:

![](images/3c05a6b849c34c1cf00dda1b64bba587c977431e5c8b8684f3c3f49880cabd9f.jpg)  
Fig. 2. The workflow of the persona-controlled recipe generation task

Ingredient F1 (IF1): Creators frequently alter standard recipes by omitting steps or adding signature components. To isolate the model’s semantic assignment capabilities from its topological planning, we measure IF1 strictly on the subset of steps where the predicted physical action aligns with the ground truth. Let $S _ { a l i g n }$ be the set of steps where $\hat { a } _ { t } = a _ { t }$ . We define IF1 as the average harmonic mean of precision and recall between the predicted ingredient subset $\hat { W } _ { t }$ and the actual ingredient subset $W _ { t }$ for all aligned steps:

$$
\mathrm { I F 1 } = \frac { 1 } { | S _ { a l i g n } | } \sum _ { t \in S _ { a l i g n } } \frac { 2 \cdot | \hat { W } _ { t } \cap W _ { t } | } { | \hat { W } _ { t } | + | W _ { t } | }
$$

Normalized Edge F1 (nEF1): This is our primary metric for operational fidelity. It evaluates whether the model routed the correct ingredient into the correct physical state. We extract the set of directed edges $E ,$ , where each edge is a tuple $( a , w )$ representing an ingredient w being subjected to action a. To ensure we are measuring physics rather than vocabulary (lexical leakage), we apply a normalization function norm(a) that maps synonymous verbs $\mathrm { ( e . g . , \Omega " f r y " }$ "sauté", "sear") to a canonical physical state ("cook"). The Normalized Edge F1 is calculated as the overlap of these state-ingredient tuples between the predicted and actual graphs:

$$
\mathrm { n E F 1 } = \frac { 2 \cdot | \hat { E } _ { n o r m } \cap E _ { n o r m } | } { | \hat { E } _ { n o r m } | + | E _ { n o r m } | }
$$

By requiring the model to match the exact $( a , w )$ tuples, nEF1 heavily penalizes structural hallucinations. A high nEF1 guarantees that the model has not just mimicked the creator’s tone, but successfully replicated their distinct operational footprint, making it a very relevant metric in culinary rigor.

Macro-Complexity and Sequence Alignments: In addition to ingredient and edge mapping, we evaluate the macroscopic length prediction using Step Error, defined as the absolute normalized diference between the predicted and actual number of procedural steps: Step $\begin{array} { r } { \mathrm { E r r o r } = \frac { | \hat { N } - N | } { N } } \end{array}$ . Also, to measure pure action selection and ordering independent of ingredients, we track the Action Jaccard Index and the Longest Common Subsequence Ratio between the predicted and actual action sequences, where empirical analysis shows they correlate heavily with nEF1.

## 5.3 The Spectrum of LLM-Assisted Generative Baselines

To establish a robust methodology for procedural graph generation, we evaluate a series of generative paradigms, moving from macroscopic semantic prompting to microscopic topological unrolling. Naturally, when generating an unseen dish, the world knowledge needed is where LLM can help.

Zero-Shot Generation: The most naive baseline treats persona replication as a standard text-based style-transfer task. Given the dish D and ingredients $I ,$ an LLM is prompted zero-shot to output the execution graph in the target creator’s style.

Few-Shot In-Context Learning: a few-shot paradigm, providing the LLM with a retrieved subset of the creator’s historical execution graphs as in-context exemplars, along with a retrieved formal recipe in text.

Microscopic Unrolling (Step-by-Step Prediction): To mitigate macroscopic planning failures, we decompose the graph generation into a strictly stepby-step autoregressive process. That is, the system is tasked only with predicting the next sequential action $a _ { t }$ and its associated ingredient subset $w _ { t } ,$ strictly conditioned on the executed state history. Within this step-by-step framework, we isolate two distinct predictive engines: Autoregressive LLM: This engine relies on semantic few-shot prompting to predict the next logical step; and correspondingly, N-Gram Guided Topology abandons semantic reasoning for the action-prediction phase. It utilizes the creator’s pure statistical action transition matrix to predict the next physical state $P ( a _ { t } | a _ { t - 1 } )$

Also, as diferent generators show a variance on dishes, an ensemble method is also implemented with a selector trained over a subset of features and based on other models’ performance in training.

## 5.4 Structured Two-Stage Generation: Disentangling Skeleton and Semantics

While the LLM-driven approaches provides strong semantic reasoning, it relies entirely on black-box, token-intensive API calls at every step. We also introduce a fully transparent, LLM-free (at inference) generative algorithm: Structured Two-Stage (S2S) Generation [7,31].

This approach mathematically disentangles the generation of the structural topology (the skeleton) from the ingredient assignment (the semantics), allowing us to apply learned creator-specific statistical priors at both levels.

Stage 1: Topological Skeleton Generation (Action Planning) The first stage generates the sequence of action states $A = \left( a _ { 1 } , a _ { 2 } , \dotsc , a _ { N } \right)$ . Rather than relying on an LLM to blindly guess the length of the workflow, we deterministically predict the macro-complexity N. The step count is calculated via a learned heuristic that fuses the creator’s historical mean step count, their ratio of viral-steps to formal-steps, and their steps-per-ingredient density.

With N established, we employ Beam Search to decode the optimal action sequence. At each step t, a candidate action $a _ { t }$ is evaluated using a composite logarithmic objective function that dynamically fuses the creator’s physical habits with the semantic requirements of the dish:

$$
\begin{array} { r } { \mathrm { S c o r e } ( a _ { t } ) = \alpha \log P _ { t r a n s } + \beta \log P _ { s e m } + } \end{array}
$$

$$
\gamma \log P _ { p o s } + \delta \log P _ { u n i } - \lambda ( A )
$$

Where $P _ { t r a n s } ( a _ { t } | a _ { t - 1 } )$ : The creator’s learned bigram transition prior (the physics); $P _ { s e m } ( a _ { t } | \mathcal { F } _ { t } , I )$ : The semantic alignment prior, derived from mapping the formal recipe’s actions $\mathcal { F }$ to the creator’s viral vocabulary via historical ingredient overlap; $P _ { p o s } ( a _ { t } | \phi _ { t } ) ;$ : A positional prior measuring the creator’s likelihood to use $a _ { t }$ in the current phase $\phi _ { t } \in \mathrm  \{ e a r l y $ , mid, late}; and $\lambda ( A )$ : A structural penalty applied to highly repetitive, stagnant action loops.

Stage 2: Semantic Flesh (Ingredient Assignment) Once the skeleton A is generated, the algorithm deterministically routes the available raw ingredients I to the predicted nodes. Instead of relying on LLM semantic reasoning, the assignment is driven by a learned bipartite mapping. An ingredient w is routed to action $a _ { t }$ if it maximizes the creator’s historical ingredient-action prior $P ( \boldsymbol { a } _ { t } | \boldsymbol { w } )$ and exhibits high lexical overlap with the aligned formal step. To preserve the creator’s topological depth-to-width ratio, we mathematically bound the maximum fan-in (ingredients per step).

## 5.5 Performance Comparisons

An analysis of the generative baselines in Table 3 reveals a stark contrast in where diferent architectures excel. When evaluating semantic assignment, the few-shot LLM significantly outperforms the LLM-free baselines, securing the highest Ingredient F1 (IF1) on both long (0.344) and short (0.385) video formats compared to S2S (0.294 and 0.348, respectively). Conversely, when evaluating macro-structural planning, S2S is the definitive leader. By strictly enforcing topological sequences, it achieves the lowest Step Error across both long (0.350)

and short (0.332) videos, cleanly beating the few-shot baseline (0.373 and 0.351). Despite these opposing strengths, both models achieve highly competitive overall operational fidelity; the few-shot model reaches an nEF1 of 0.474 (long) and 0.478 (short), while S2S reaches 0.456 (long) and 0.455 (short). Notably, intermediate microscopic approaches fail to dominate in any metric, because forcing generation into isolated step predictions without planning actually fractures the LLM’s overarching contextual logic. Substituting a diferent creator’s priors while holding the dish and ingredients fixed costs S2S 0.06 nEF1 on both formats $( p < 1 0 ^ { - 5 4 } )$ , confirming that this fidelity derives from persona conditioning rather than generic culinary structure.

These metrics demonstrate that both models reach comparable operational fidelity through entirely divergent, yet complementary, mechanisms. The fewshot baseline leverages its massive parametric memory to dominate semantic assignment, seamlessly routing complex ingredients into the graph. However, it remains constrained by the inherent macro-planning deficits of autoregressive generation, resulting in elevated Step Errors. In contrast, S2S model relies on explicit Markovian priors [25] to perfectly control the sequence topology, securing rigid structural adherence while naturally sacrificing the zero-shot semantic agility of an LLM. This empirical trade-of proves that no singular paradigm can simultaneously maximize semantic nuance and structural rigidity. Because capturing a complete procedural persona requires both global semantic awareness and local topological control, the complementary strengths directly motivate an ensemble framework, which utilizes a learned selector to dynamically route generation to a certain model.

Table 3. Performance comparison over the stylish recipe generation task
<table><tr><td></td><td colspan="3">long videos</td><td colspan="3">short videos</td></tr><tr><td>Item</td><td>IF1</td><td></td><td>nEF1 StepErr</td><td>IF1</td><td></td><td>nEF1 StepErr</td></tr><tr><td>Zero-shot LLM</td><td>0.223</td><td>0.338</td><td>0.481</td><td>0.261</td><td>0.341</td><td>0.458</td></tr><tr><td>Few-shot LLM</td><td>0.344</td><td>0.474</td><td>0.373</td><td>0.385</td><td>0.478</td><td>0.351</td></tr><tr><td>Autoregressive</td><td>0.318</td><td>0.473</td><td>0.446</td><td>0.353</td><td>0.446</td><td>0.410</td></tr><tr><td>N-gram sequence</td><td>0.303</td><td>0.452</td><td>0.549</td><td>0.363</td><td>0.446</td><td>0.513</td></tr><tr><td>Two-stage</td><td>0.294</td><td>0.456</td><td>0.350</td><td>0.348</td><td>0.455</td><td>0.332</td></tr><tr><td>Ensemble with selector 0.328 0.485</td><td></td><td></td><td>0.357</td><td>0.378</td><td>0.479</td><td>0.352</td></tr></table>

Table 4. Ablation study on LLM backbone replacement: comparison of generative performance across proprietary frontier and open-weight models.
<table><tr><td></td><td colspan="3">Long Videos</td><td colspan="3">Short Videos</td></tr><tr><td>Model</td><td>IF1</td><td></td><td>nEF1 StepErr ↓</td><td>IF1</td><td></td><td>nEF1 StepErr ↓</td></tr><tr><td>GPT-5.4</td><td></td><td>0.396 0.556</td><td>0.384</td><td>0.413</td><td>0.532</td><td>0.317</td></tr><tr><td>GPT-4.1-mini</td><td>0.393</td><td>0.504</td><td>0.376</td><td>0.481</td><td>0.446</td><td>0.269</td></tr><tr><td>Qwen3.6-27B</td><td>0.369</td><td>0.510</td><td>0.409</td><td></td><td>0.4100.476</td><td>0.248</td></tr></table>

## 6 Ablation Study

LLM backbone selection: in Table 4, substituting the core LLM engine across both proprietary and open-weight architectures proves that raw parameter scaling doesn’t resolve the macro-planning deficit. A larger model is not necessarily better at adhering to rigid procedural topologies.

Context window saturation in few-shot prompting: To evaluate the few-shot baseline’s reliance on in-context style exemplars, the number of shots (k) is scaled in order to isolate how much of the creator’s procedural footprint is absorbed dynamically versus relying on foundational knowledge. While increasing exemplars yields steady improvements in semantic assignment (peak performance at $k = 9 )$ , injecting too many complex procedural graphs causes context dilution (starting from $k = 1 1 )$ ), overwhelming LLM’s ability to efectively attend to key stylistic signals.

## 7 Conclusion and Future Work

In this work, we challenge the prevailing NLP paradigm for procedural style transfer by demonstrating that traditional linguistic stylometry is inadequate in persona-controlled recipe generation: considering procedural style as a topological footprint, encoding a creator’s persona in the physical shape of their execution graph rather than their vocabulary, we introduced the ViralRecipes-Trans dataset along with metrics to measure operational fidelity, alongside a novel structured two-stage generation algorithm. Between this model’s macroplanning superiority and LLM’s micro-semantic agility from few-shot scenarios, the trade-of is explored with insight on an ensemble.

Future work will bridge this gap by routing toward true neuro-symbolic fusion with interoperability as well as a post-hoc calibrator, which empirically validates physical constraints and repairs potential violations. Further, while this framework was built in the culinary domain, the concept of "topological stylometry" is broadly applicable in other procedural domains for its graph-theoretic extraction, moving toward generalized, persona-aware instructional intelligence.

## References

1. Ekin Akyürek, Afra Amini, Jacob Andreas, and Yoon Kim: Compositional Generalisation with Structured Reordering and Fertility Layers. on Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL), pp. 2157–2171 (2023).

2. Hritik Bansal, Ran Levy, and Yonatan Bisk: Order-based pre-training strategies for procedural text understanding. on Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), pp. 645–652 (2024).

3. Michał Bien, Michał Gilski, Maciej Maciejewska, Wojciech Taisner, Dawid Wisniewski, and Agnieszka Lawrynowicz: RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation. on Proceedings of the International Conference on Natural Language Generation, pp. 22–28 (2020).

4. Antoine Bosselut, Omer Levy, Ari Holtzman, Corin Ennis, Dieter Fox, and Yejin Choi: Simulating Action Dynamics with Neural Process Networks. on Proceedings of the International Conference on Learning Representations (2018).

5. Peter F. Brown, Vincent J. Della Pietra, Peter V. deSouza, Jenifer C. Lai, and Robert L. Mercer: Class-Based n-gram Models of Natural Language. on Computational Linguistics, 18(4), pp. 467–480 (1992).

6. Lucia Donatelli, Theresa Schmidt, Debanjali Biswas, Arne Köhn, Fangzhou Zhai, and Alexander Koller: Aligning Actions Across Recipe Graphs. on Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6930–6942 (2021).

7. Angela Fan, Mike Lewis, and Yann Dauphin: Hierarchical Neural Story Generation. on Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pp. 889–898 (2018).

8. Jia-Chen Gu, Zhen-Hua Ling, Quan Liu, Zhigang Chen, and Xiaodan Zhu: Detecting speaker personas from conversational texts. on Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 1126–1136 (2021).

9. Faezeh Jafariakinabad, Sansiri Tarnpradab, and Kien A. Hua: Syntactic recurrent neural network for authorship attribution. on Proceedings of the IEEE International Conference on Semantic Computing (ICSC), pp. 251–258 (2019).

10. Di Jin, Zhijing Jin, Zhiting Hu, Vlas Vovk, and Liwei Wang: Deep learning for text style transfer: A survey. on Computational Linguistics, 48(1):155–205 (2022).

11. Karen Spärck Jones: A Statistical Interpretation of Term Specificity and Its Application in Retrieval. on Journal of Documentation, 28(1), pp. 11–21 (1972).

12. Leon Kopitar, Leon Bedrač, Larissa J. Strath, Jiang Bian, and Gregor Stiglic: Improving Personalized Meal Planning with Large Language Models: Identifying and Decomposing Compound Ingredients. Nutrients, 17(9), 1492 (2025).

13. Wen Lai, Viktor Hangya, and Alexander Fraser: Style-Specific Neurons for Steering LLMs in Text Style Transfer. on Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5410–5425 (2024).

14. Guillaume Lample, Sandeep Subramanian, Eric Smith, Ludovic Denoyer, Marc’Aurelio Ranzato, and Y-Lan Boureau: Multiple-Attribute Text Rewriting. on Proceedings of the International Conference on Learning Representations (ICLR) (2019).

15. Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. on Advances in Neural Information Processing Systems, 33, pp. 9459–9474 (2020).

16. Chin-Yew Lin: ROUGE: A Package for Automatic Evaluation of Summaries. on Text Summarization Branches Out, pp. 74–81 (2004).

17. Ximing Lu, Peter West, Rowan Zellers, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi: NeuroLogic Decoding: (Un)directed Generation with Neural Logic Constraints. on Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4287–4299 (2021).

18. Yiwei Lyu, Paul Pu Liang, Hai Pham, Eduard Hovy, Barnabás Póczos, Ruslan Salakhutdinov, and Louis-Philippe Morency: StylePTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer. on Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), pp. 2116–2138 (2021).

19. Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, and Julian McAuley: Generating personalized recipes from historical user preferences. on Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5976–5982 (2019).

20. Javier Marin, Aritro Biswas, Ferda Ofli, Nicholas Hynes, Amaia Salvador, Yusuf Aytar, Ingmar Weber, and Antonio Torralba: Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images. on IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1):187–203 (2019).

21. Angelos Mavrogiannis, Christoforos Mavrogiannis, and Yiannis Aloimonos: Cook2LTL: Translating Cooking Recipes to LTL Formulae using Large Language Models. on Proceedings of IEEE International Conference on Robotics and Automation (ICRA), 17679–17686 (2024).

22. Shinsuke Mori, Hirokuni Maeta, Tetsuro Sasada, Koichiro Yoshino, Yoko Yamakata, and Tetsuji Itoh: Flow Graph Corpus from Recipe Texts. on Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC), 2370–2377 (2014).

23. Nasim Nouri: Text Style Transfer via Optimal Transport. on Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), pp. 2532–2541 (2022).

24. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu: Bleu: a Method for Automatic Evaluation of Machine Translation. on Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pp. 311–318 (2002).

25. Guillaume Perez et al.: Markov Constraint as Large Language Model Surrogate. on Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), pp. 1844–1852 (2024).

26. Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel: Language Models as Knowledge Bases? on Proceedings of the Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (IJCNLP), pp. 2463–2473 (2019).

27. Sudha Rao and Joel Tetreault: Dear Sir or Madam, May I Introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer. on Proceedings of the Conference of North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), pp. 129–140 (2018).

28. Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola: Style Transfer from Non-Parallel Text by Cross-Alignment. on Advances in Neural Information Processing Systems (NeurIPS), 30, pp. 6833–6844 (2017).

29. Efstathios Stamatatos: A survey of modern authorship attribution methods. on Journal of the American Society for Information Science and Technology, 60(3), pp. 538–556 (2009).

30. Efstathios Stamatatos: Authorship Attribution Using Text Distortion. on Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL) (2017).

31. Lili Yao, Nanyun Peng, Ralph Weischedel, Kevin Knight, Dongyan Zhao, and Rui Yan: Plan-and-Write: Towards Better Automatic Storytelling. on Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 33(01), pp. 7378–7385 (2019).

32. Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang: Generate rather than retrieve: Large language models are strong context generators. on The Eleventh International Conference on Learning Representations (ICLR) (2023).