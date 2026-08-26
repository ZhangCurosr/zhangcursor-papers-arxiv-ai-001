# RAGSentinel: Certifiable Geometric Consensus for Robust Retrieval-Augmented Generation

Yueyang Quan<sup>1</sup> Anjun Gao<sup>2</sup> Yufei Xia<sup>2</sup>\* Minghong Fang<sup>2</sup> Zhuqing Liu<sup>1</sup> <sup>1</sup>University of North Texas, <sup>2</sup>University of Louisville

## Abstract

Retrieval-augmented generation (RAG) improves the factuality of large language models by grounding responses in external documents, but it also exposes a critical security vulnerability: adversarial documents injected into the knowledge database can enter the context window and steer the model toward targeted incorrect answers. Existing post-retrieval defenses rely on instruction following, parametric knowledge, or text-level consistency, all of which can be imitated or optimized against by adaptive attackers. We propose RAGSentinel, a training-free, label-free defense for blackbox RAG systems. RAGSentinel uses a surrogate encoder to measure query-conditioned hidden-state shifts induced by retrieved documents, removes shared topic directions, and filters poisoned documents as geometric outliers from a robust majority consensus. We prove that, under an honest-majority assumption and a representation-level separation condition, RAGSentinel exactly recovers a poison-free majority-sized context. Experiments across three question-answering datasets, three LLM families, and multiple poisoning attacks show that RAGSentinel consistently achieves low attack success rates while preserving competitive accuracy and remaining effective against adaptive attacks with full pipeline knowledge.

## 1 Introduction

Large language models (LLMs) (Brown et al., 2020; Achiam et al., 2023) have demonstrated remarkable capabilities across a wide range of knowledge-intensive tasks, yet their reliance on static parametric knowledge limits their ability to stay current and factually grounded (Lewis et al., 2020). Retrieval-augmented generation (RAG) (Karpukhin et al., 2020; Chen et al., 2024b) addresses this by coupling LLMs with an external knowledge database at inference time, enabling factually accurate responses without retraining. In a typical RAG pipeline, a retriever selects the top-k documents from the knowledge database, which are concatenated with the user query and passed to the LLM as context.

Recent studies have demonstrated that RAG systems are susceptible to knowledge poisoning attacks. In this threat model, an adversary inserts malicious documents into the knowledge database prior to inference. These documents are carefully constructed so that they achieve high retrieval ranking for a specific target query while embedding an incorrect answer (Zou et al., 2025; Zhong et al., 2023; Zhang et al., 2026a, 2025b). The attack requires no access to the deployed LLM, no query modification, and no inference-time intervention, only the ability to write to the database. Because adversarial documents are optimized to be retrievalrelevant and factually plausible, they are indistinguishable from benign documents in query logs and model outputs, leaving existing pipelines with no natural point of interception (Zhang et al., 2026c; Greshake et al., 2023).

Existing post-retrieval defenses fall into three families, each undermined by a structural limitation. Instruction-based methods prompt the LLM to resist suspicious context (Xiang et al., 2024; Wei et al., 2025; Asai et al., 2024; Gao et al., 2023), but rely on the model’s own judgment to detect contradictions, precisely the capability that plausible adversarial documents are designed to defeat. Knowledge integration techniques aim to resolve discrepancies between retrieved evidence and the LLM’s stored parametric knowledge (Wang et al., 2025; Zhou et al., 2025; Wang et al., 2023b; Jeong et al., 2024). However, their effectiveness depends on the model possessing reliable domainspecific knowledge, making them less applicable when such knowledge is limited or unavailable. In contrast, consistency-based approaches assess the agreement among retrieved documents (Deng et al., 2025; Jiang et al., 2023b; Cheng et al., 2025). Yet, this assumption can be exploited by sophisticated adversaries, who may generate mutually consistent poisoned documents that closely resemble the benign corpus while collectively promoting an incorrect answer. All three families share a common blind spot: each operates on signals that an attacker can directly optimize against, token outputs, parametric knowledge conflicts, or cross-document text overlap. What an attacker cannot easily control, however, is the hidden-state shift that a document induces in an encoder when it asserts a false claim: asserting a factually wrong answer leaves a geometric trace in representation space that text-level optimization cannot erase, because the encoder’s semantic geometry lies outside the attacker’s control. This signal is one that all three families leave entirely untapped.

We make the following observation: when each retrieved document is appended to the query and passed through a surrogate encoder, the resulting hidden-state shift encodes that document’s factual stance. After removing the shared querytopic component, benign documents induce residual shifts that cluster near a common direction, while poisoned documents are geometric outliers in this same space. This structure holds even under retrieval-optimized adversarial documents: asserting a false factual claim leaves a geometric trace in representation space that text-level optimization cannot erase. This signal requires no interaction with the deployed LLM, no labeled poison examples, and no model retraining.

We propose RAGSentinel, a training-free, labelfree post-retrieval defense that exploits this geometric structure to identify and remove poisoned documents before the deployed LLM is invoked. The defender has text-in, text-out access to the deployed LLM and a separately chosen surrogate encoder for hidden-state extraction, with no knowledge of which documents are poisoned or what the correct answer is. RAGSentinel scores documents by their deviation from a robust majority consensus via the geometric median, and selects a trusted context using a query-adaptive filtering radius, passing it to the LLM in a single zero-shot call with exactly $k + 1$ surrogate forward passes.

Theoretically, we prove that under an honestmajority assumption and a geometric separation condition, RAGSentinel exactly recovers a poisonfree majority-sized context (Theorem 1). Empirically, we evaluate on Natural Questions, HotpotQA, and MS-MARCO, across three LLM families and three poisoning attacks. RAGSentinel consistently achieves low attack success rates while maintaining accuracy comparable to Vanilla RAG, incurring negligible computational overhead, and retaining its defensive advantage under three adaptive attacks with full pipeline knowledge.

Our contributions are as follows:

• We identify poisoned documents as geometric outliers in a surrogate encoder’s residual space, persistent even under retrieval-optimized adversarial construction.

• We propose RAGSentinel, a training-free, blackbox-compatible defense that filters poisoned documents by their geometric deviation from the benign majority consensus.

• We prove a certifiable filtering guarantee (Theorem 1) and corroborate it with extensive experiments across datasets, models, attacks, and adaptive adversaries.

## 2 Background and related work

We summarize the most relevant prior work here and defer a detailed discussion to Appendix A. Key notation is summarized in Table 5 (Appendix).

Background on retrieval-augmented generation (RAG): In a typical RAG pipeline, given a user query $q ,$ a retriever selects the top-k documents $\{ p _ { i } \} _ { i = 1 } ^ { k }$ (i indexes each retrieved document) from an external knowledge database and passes them with $q$ to an LLM for generation.

Knowledge poisoning attacks to RAG: Knowledge poisoning attacks inject adversarial documents crafted to rank highly for target queries while encoding wrong answers (Zhong et al., 2023; Zou et al., 2025). Unlike inference-time prompt injection (Greshake et al., 2023; Perez and Ribeiro, 2022), they require only write access to the database, making them undetectable via query logs or model outputs. Attacks range from optimizationbased methods that jointly maximize retrieval relevance and answer manipulation (Wallace et al., 2019) to gradient-free generative attacks producing fluent adversarial documents without token-level artifacts (Zhang et al., 2026c).

Post-retrieval defenses: Defenses fall into three families, each operating on signals the attacker can directly optimize against. Instruction-based methods (Xiang et al., 2024; Wei et al., 2025; Asai et al., 2024; Gao et al., 2023; Yu et al., 2024) resist noisy context via prompting or self-synthesized rationales, but delegate conflict detection to the very LLM adversarial documents are crafted to deceive; RobustRAG avoids this by isolating documents yet discards cross-document consensus entirely. Knowledge consolidation methods (Wang et al., 2025; Zhou et al., 2025; Wang et al., 2023b; Jeong et al., 2024) reconcile retrieved evidence with parametric knowledge, but break down when that knowledge is absent or when poisoned documents align with parametric priors. Consistencybased methods (Deng et al., 2025; Jiang et al., 2023b) filter via cross-document agreement, but a capable attacker can produce documents textually consistent with the benign majority while encoding a contradictory answer. Several recent studies (Gao et al., 2026b; Zhang et al., 2025a, 2026b; Gao et al., 2026a) have also explored post-attack forensic attribution, which seeks to trace successful attacks back to their underlying root causes. Such forensic analysis complements existing defense mechanisms by providing insights after an attack has occurred. However, our work focuses on attack mitigation rather than post-attack investigation and attribution; therefore, we leave forensic analysis outside the scope of this paper.

Knowledge foundation of RAGSentinel: RAGSentinel builds on two untapped lines of work. Byzantine-robust aggregation (Blanchard et al., 2017; Chen et al., 2017) shows the geometric median provably resists adversarial corruption under an honest-majority assumption; we transfer this guarantee from federated gradient aggregation to hidden-state consensus filtering. Representation probing (Burns et al., 2022; Azaria and Mitchell, 2023) shows LLM hidden states encode factual stance that surface text does not reveal; we operationalize this via an independent surrogate encoder, keeping filtering outside the deployed LLM and the attacker’s optimization surface.

## 3 Threat model

Attacker’s goals, capabilities, and knowledge: We consider a knowledge poisoning attacker who injects adversarial documents into the knowledge database to steer the RAG system toward a targeted wrong answer (Zou et al., 2025). The attacker controls only the injected documents: they cannot modify the user query, the retriever, the LLM parameters, or any post-retrieval defense. We assume the attacker knows the target query, the retrieval model family, and the database domain, and is aware that a post-retrieval defense may be present.

Defender’s knowledge and goal: The defender has black-box access to the deployed LLM $f _ { \theta }$ and full access to a separately chosen surrogate encoder $f _ { \varphi } ,$ used solely for hidden-state extraction. This assumption is mild in practice: $f _ { \varphi }$ can be any off-theshelf open-source encoder chosen independently of $f _ { \theta } ,$ , and we show in Section 6 that RAGSentinel’s defense is not sensitive to this choice. Given the user query and the top-k retrieved documents, the defender must identify and remove poisoned documents, without poison labels, ground-truth answers, or model retraining, under the sole structural assumption that poisoned documents are strictly fewer than half of the retrieved set. Remark that the attacker has no access to the surrogate encoder $f _ { \varphi } ,$ its induced representation shifts, or the filtering decisions made during defense. For adaptive attack evaluation, we further consider a stronger attacker who knows the full RAGSentinel algorithmic pipeline and designs poisoned documents to target specific filtering components, while still lacking query or gradient access to the surrogate encoder $f _ { \varphi }$ , its realized hidden-state shifts, and the defense-time filtering decisions.

## 4 Our method

We propose RAGSentinel, a training-free, labelfree post-retrieval defense against knowledge poisoning attacks. The key observation motivating RAGSentinel is illustrated in Figure 1: when each retrieved document is appended to the query, the surrogate encoder’s hidden-state shift encodes the document’s factual stance. Benign documents induce shifts that cluster near a common direction, while poisoned documents are geometric outliers in this residual space (Figure 3 in the Appendix shows additional examples on NQ and HotpotQA). As shown in Figure 4 (Appendix), RAGSentinel exploits this structure under the honest-majority assumption in Section 3 across three phases, shift extraction and preprocessing (Section 4.1), consensus scoring (Section 4.2), and adaptive filtering and context construction (Section 4.3), to identify and remove poisoned documents without labeled examples, model retraining, or white-box access to $f _ { \theta }$ The full pipeline is in Algorithm 1 (Appendix).

![](images/831981bf0cf8901eb85097188247b146c1fb56f99d84dd2ffea558ae75cba559.jpg)  
Figure 1: Poisoned documents are geometric outliers in the topic-removed residual space on the MS-MARCO query “what group sang stairway to heaven?” with ten retrieved documents including two injected poisoned documents. Each marker represents one document: ● benign, ◆ poisoned, ★ geometric median anchor c. Benign documents cluster near a common direction while the two poisoned documents are clear outliers.

## 4.1 Shift extraction and preprocessing

To score documents by their geometric consistency, RAGSentinel needs a representation of each document’s factual influence on the surrogate encoder’s internal state. Naively using the surrogate’s hidden state for the full question-document input conflates three confounds: the query-topic component shared by all retrieved documents, high-dimensional noise from unresponsive dimensions, and outlier shift magnitudes that distort geometric center estimates. The preprocessing pipeline below addresses each in turn, yielding for each document a residual vector that isolates its specific factual influence.

Prompt construction and shift extraction: Both the query-only and document-conditioned inputs are cast into a prompt template so that the pooled representation reflects the surrogate’s about-toanswer state rather than a generic query embedding; full templates are provided in Appendix G. Let $\phi _ { \varphi } ( \cdot )$ denote the hidden-state encoder induced by $f _ { \varphi } ,$ implemented as the representation at the final token of the last hidden layer. The queryonly prompt yields baseline $h _ { 0 } ~ = ~ \phi _ { \varphi } ( q )$ and each document-conditioned prompt yields $h _ { i } \ =$ $\phi _ { \varphi } ( q , p _ { i } )$ , where i represents each of the top-k retrieved documents; identical pooling applied to both ensures direct comparability despite differing sequence lengths. The document-induced hiddenstate shift is $\delta _ { i } = h _ { i } - h _ { 0 }$

Dynamic active subspace selection: LLM hidden states are extremely high-dimensional (often $d > 4 0 9 6 )$ , yet most dimensions respond minimally to any document for a given query, contributing only noise. RAGSentinel dynamically selects a query-specific active subspace of responsive dimensions. For each dimension $\ell ,$ we measure peak responsiveness $u _ { \ell } = \operatorname* { m a x } _ { i } | \delta _ { i } ^ { ( \ell ) } |$ and retain $\mathcal { V } = \left\{ \ell : u _ { \ell } \geq \mathrm { M e d i a n } ( \{ u _ { \ell } \} ) + \mathrm { \tilde { M A D } } ( \{ u _ { \ell } \} ) \right\}$ where the median absolute deviation is represented as $\mathrm { M A D } ( \{ u _ { \ell } \} ) = \mathrm { M e d i a n } ( | u _ { \ell } - \mathrm { M e d i a n } ( \{ u _ { \ell } \} ) | )$ Each shift is projected as $\tilde { \delta } _ { i } = \delta _ { i } [ \gamma ]$ . If |V| falls below a minimum threshold, RAGSentinel falls back to the $\textstyle \mathrm { t o p - } \lceil d / k \rceil$ dimensions ranked by $u _ { \ell } .$

Adaptive norm clipping: Individual documents may produce shifts with anomalously large norms that distort the geometric center estimated in the consensus phase. RAGSentinel clips each projected shift to a query-adaptive bound $B _ { \mathrm { a d p } } ~ =$ Median $( \{ b _ { i } \} ) + \mathrm { M A D } ( \{ b _ { i } \} )$ ), where $b _ { i } = \| \tilde { \delta } _ { i } \| _ { 2 } ,$ which captures the typical norm level without being inflated by the outliers being clipped: $\bar { \delta } _ { i } =$ $\begin{array} { r } { \tilde { \delta } _ { i } \cdot \operatorname* { m i n } \Bigl ( 1 , ~ \frac { \tilde { B } _ { \mathrm { a d p } } } { \| \tilde { \delta } _ { i } \| _ { 2 } + \epsilon } \Bigr ) } \end{array}$

Topic-direction removal: All k documents share a broad query-topic component in their shifts reflecting retrieval relevance rather than factual content; retaining it would make all documents appear mutually consistent and obscure the factual differences the defense must detect. RAGSentinel removes this shared direction by mean-centering: $\begin{array} { r } { z _ { i } = \bar { \delta } _ { i } - \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \bar { \delta } _ { j } } \end{array}$ . Since the preceding normclipping step limits the influence of any individual document, the arithmetic mean provides a simple estimate of the shared topic component. This step is used only to remove the common topic direction; robust estimation of the benign consensus is performed subsequently using the geometric median.

## 4.2 Consensus scoring

With topic-removed residuals $\left\{ z _ { i } \right\}$ , RAGSentinel assigns each document a scalar consensus distance $d _ { i }$ measuring deviation from the benign majority. The majority direction must be estimated from a set containing up to $\lfloor ( k - 1 ) / 2 \rfloor$ poisoned documents, requiring a robust estimator. Moreover, a global estimate alone can be fooled by a poisoned document whose residual points roughly toward the majority direction. RAGSentinel therefore combines a robust global anchor with a local consistency term that checks whether each document is mutually supported by its nearest neighbors.

Global anchor via geometric median: RAGSentinel estimates the majority consensus direction as the geometric median: $c =$ ${ \mathrm { G e o M e d } } ( \{ z _ { i } \} _ { i = 1 } ^ { k } ) = { \mathrm { a r g } } { \mathrm { m i n } } _ { x } \sum _ { i = 1 } ^ { k } \| x - z _ { i } \| _ { 2 }$ Unlike the arithmetic mean, the geometric median remains close to the true majority center as long as fewer than half the inputs are corrupted, making it well-suited to the honest-majority setting. We confirm this advantage empirically in Section 6.2. The global anchor distance is $d _ { i } ^ { \mathrm { a n c h o r } } = 1 - \cos ( z _ { i } , ~ c )$

Local majority-consensus distance: While the global anchor detects broad deviations from the majority direction, it can miss a poisoned document that aligns with the anchor yet is not locally supported by any benign cluster. RAGSentinel therefore supplements it with a local consistency term: for each document $p _ { i }$ , we identify its $m =$ max $( 1 , \lceil k / 2 \rceil - 1 )$ nearest neighbors $\mathcal { N } _ { i } ^ { ( m ) }$ under pairwise cosine distance $D _ { i j } = 1 - \cos ( z _ { i } , z _ { j } )$ and define $\begin{array} { r } { d _ { i } ^ { \mathrm { l o c a l } } = \frac { 1 } { m } \sum _ { j \in { \mathcal N } _ { i } ^ { ( m ) } } { D } _ { i j } } \end{array}$

Adaptive signal combination: The two signals are combined with an adaptive weight that reflects their relative discriminative power:

$$
d _ { i } = \left( 1 - \lambda \right) d _ { i } ^ { \mathrm { a n c h o r } } + \lambda d _ { i } ^ { \mathrm { l o c a l } } ,\tag{1}
$$

where $\begin{array} { r } { \lambda = 1 - \frac { \mathrm { S t d } ( \{ d _ { i } ^ { \mathrm { a n c h o r } } \} ) } { \mathrm { S t d } ( \{ d _ { i } ^ { \mathrm { a n c h o r } } \} ) + \mathrm { S t d } ( \{ d _ { i } ^ { \mathrm { l o c a l } } \} ) + \epsilon } } \end{array}$ . When anchor distances exhibit high spread, the global signal dominates; when they are compressed relative to local distances λ approaches 1 and the local signal takes over. All quantities are derived from the retrieved set of each individual query, making RAGSentinel fully parameter-free.

## 4.3 Adaptive filtering and context construction

Given consensus distances $\{ d _ { i } \}$ , RAGSentinel derives a query-adaptive filtering radius rather than a fixed threshold, since the scale of $d _ { i }$ varies substantially across queries depending on the topical homogeneity of the retrieved set.

Adaptive majority-radius filtering: Let R denote the $\lceil k / 2 \rceil$ -th smallest value in $\{ d _ { i } \}$ , the smallest radius covering at least half the retrieved documents. RAGSentinel adjusts R by the normalized dispersion of the distance distribution:

$$
R _ { \mathrm { a d p } } = \left( 1 + \frac { 1 } { 1 + \sigma _ { d } } \right) R ,\tag{2}
$$

where $\begin{array} { r } { \sigma _ { d } = \frac { \mathrm { M A D } ( \{ d _ { i } \} ) } { \mathrm { M e d i a n } ( \{ d _ { i } \} ) + \epsilon } } \end{array}$ . Expanding toward 2R when distances are compact (tight benign cluster) and contracting toward R when they are dispersed (less structured distribution). The surviving set is $S = \{ i : d _ { i } \leq R _ { \mathrm { a d p } } \}$ . If S is empty, RAGSentinel retains the single document with smallest $d _ { i }$ to prevent empty-context generation failures.

Context construction and generation: Documents in S are sorted by $d _ { i }$ ascending and the top $\lceil k / 2 \rceil$ are selected, ensuring the final context contains at most a majority-sized subset. The selected documents are concatenated in consensus-ranked order and inserted into a standard RAG prompt template (Appendix G). The resulting trusted context ${ \mathcal { C } } ^ { \star }$ is passed to the black-box LLM in a single zero-shot call: $\hat { y } = f _ { \boldsymbol \theta } ( \boldsymbol { q } , \mathcal { C } ^ { \star } )$ . The LLM in RAG is invoked only at this final step, preserving strict black-box compatibility. The full pipeline requires exactly k+1 surrogate forward passes and a single call to $f _ { \boldsymbol { \theta } } ;$ all geometric operations add negligible overhead in practice.

## 5 Theoretical analysis

We analyze when RAGSentinel can exactly separate poisoned documents from a benign majority in the hidden-state residual space. Let $\mathcal { T } _ { b }$ and $\mathcal { T } _ { p }$ denote the benign and poisoned index sets in the top-k retrieved documents, with $| \mathcal { T } _ { p } | = k ^ { \prime } < k / 2$ The analysis is stated on the topic-removed residuals $\{ z _ { i } \} _ { i = 1 } ^ { k }$ produced by the surrogate encoder $f _ { \varphi }$ in the preprocessing steps of Section 4.1. All geometric arguments operate solely on these residuals; the deployed black-box LLM $f _ { \theta }$ does not enter the analysis.

## 5.1 Assumptions

Assumption 1 (Honest majority) The poisoned documents are strictly outnumbered in the retrieved set, i.e., $| \mathcal { T } _ { p } | = k ^ { \prime } < k / 2$

Assumption 2 (Benign consensus) Let $\begin{array} { r l } { \mu _ { b } } & { { } = } \end{array}$ $\frac { 1 } { \left. \mathcal { T } _ { b } \right. } \sum _ { i \in \mathcal { T } _ { b } } z _ { i }$ be the benign residual centroid. There exists $r _ { 0 } \quad > \quad 0$ such that $\begin{array} { r l } { \| \mu _ { b } \| _ { 2 } } & { { } \geq } \end{array}$ $r _ { 0 }$ . Moreover, there exist constants $\Delta _ { \mu } ^ { \mathrm { c o s } } , \ \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } }$ and $\Delta _ { E }$ such that max $_ { i \in \mathcal { T } _ { b } } \big ( 1 - \cos ( z _ { i } , \mu _ { b } ) \big ) \ \leq$ $\Delta _ { \mu } ^ { \mathrm { c o s } }$ , max $\begin{array} { r l r } { \mathrm { ~ _ { \it { i , j } \in \mathcal { I } _ { b } \left( 1 ~ - ~ \cos ( z _ { i } , z _ { j } ) \right) } ~ } } & { { } \leq } & { \Delta _ { \mathrm { p a i r } } ^ { \cos } , } \end{array}$ max $\tau _ { i , j \in \mathcal { T } _ { b } } \| z _ { i } - z _ { j } \| _ { 2 } \leq \Delta _ { E }$

Assumption 3 (Poison-to-benign separation) There exists $\gamma \ > \ 0$ such that every poisoned residual is separated from every benign residual in cosine distance: min $\underset { i \in \mathcal { T } _ { b } } { \ i _ { j \in \mathcal { T } _ { p } } } \big ( 1 - \cos ( z _ { j } , z _ { i } ) \big ) \ge \gamma$

Remark 1 Assumption 2 rules out the degenerate case where benign residuals cancel out after topic-direction removal. Assumption 3 is a representation-level condition on the residuals $\left\{ z _ { i } \right\}$ extracted by the surrogate encoder $f _ { \varphi } .$ it does not claim that everyfactually wrong document must be separated in $f _ { \varphi }$ ’s hidden-state space, but formalizes the regime in which surrogate-based consensus filtering is expected to succeed.

## 5.2 Theoretical guarantee

The following quantities are derived in Appendix B from geometric-median stability, anchor-distance bounds, and local-neighborhood bounds; we introduce them here to state the filtering condition compactly. Let m = max(1, ⌈k/2⌉ − 1) be the neighborhood size used in $d _ { i } ^ { \mathrm { l o c a l } }$ = $\begin{array} { r } { \frac { 1 } { m } \sum _ { j \in \mathcal { N } _ { i } ^ { ( m ) } } D _ { i j } } \end{array}$ , and define $\begin{array} { r } { \eta _ { c } = \frac { k } { k - 2 k ^ { \prime } } \dot { \Delta } _ { E } } \end{array}$ . Define $\begin{array} { r c l } { U _ { \mathrm { a n c } } } & { = } & { \sqrt { 2 \Delta _ { \mu } ^ { \mathrm { c o s } } } + \frac { 2 \eta _ { c } } { r _ { 0 } - \eta _ { c } } } \end{array}$ and $L _ { \mathrm { a n c } } \quad = \quad$ $\begin{array} { r } { \frac { 1 } { 2 } \left( \sqrt { 2 \gamma } - \sqrt { 2 U _ { \mathrm { a n c } } } \right) _ { + } ^ { 2 } } \end{array}$ , where $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ Here $U _ { \mathrm { a n c } }$ upper bounds the anchor distance of benign documents, while $L _ { \mathrm { a n c } }$ lower bounds the anchor distance of poisoned documents. The resulting global-local score gap is $G _ { \mathrm { f u l l } } = ( 1 - \lambda ) ( L _ { \mathrm { a n c } } -$ $\begin{array} { r } { U _ { \mathrm { a n c } } ) + \lambda \left( \frac { m - k ^ { \prime } + 1 } { m } \gamma - \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } \right) } \end{array}$ , and the benign score upper bound is $D _ { b } ^ { + } = ( 1 - \lambda ) U _ { \mathrm { a n c } } + \lambda \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } .$ Theorem 1 (Certifiable filtering) Suppose Assumptions 1-3 hold. $\begin{array} { r } { I f r _ { 0 } > \eta _ { c } , \gamma > \Delta _ { \mathrm { p a i r } } ^ { \cos } , } \end{array}$ and $\begin{array} { r } { G _ { \mathrm { f u l l } } > \frac { D _ { b } ^ { + } } { 1 + \sigma _ { d } } } \end{array}$ , then the trusted context C<sup>⋆</sup> produced by RAGSentinel satisfies

$$
\mathcal { C } ^ { \star } \cap \mathcal { T } _ { p } = \emptyset , \quad | \mathcal { C } ^ { \star } | = \lceil k / 2 \rceil , \quad \mathcal { C } ^ { \star } \subseteq \mathcal { T } _ { b } .\tag{3}
$$

Remark 2 Theorem 1 gives an explicit geometric condition under which RAGSentinel exactlyfilters poisoned documents. The term $G _ { \mathrm { f u l l } }$ lower bounds the consensus-score gap between poisoned and benign documents, while $D _ { b } ^ { + } / ( 1 + \sigma _ { d } )$ upper bounds the additional expansion introduced when the majority radius R is enlarged to $R _ { \mathrm { a d p } } .$ Therefore, the condition in Theorem 1 ensures that the adaptive filtering boundary remains below the closest poisoned document while still retaining a majoritysized benign context.

## 6 Experiments

## 6.1 Experimental setup

Datasets, evaluation metrics, and RAG settings: We evaluate on three open-domain question-answering benchmarks: Natural Questions (NQ) (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), and MS-MARCO (Bajaj et al., 2016). Following (Xiang et al., 2024; Zou et al., 2025), we sample 100 queries randomly from each dataset for experiments. We use two metrics: utility accuracy (ACC), measuring whether the defense preserves normal RAG utility on benign queries, and attack success rate (ASR), measuring the fraction of queries for which the model produces the attacker’s target wrong answer. A strong defense achieves high ACC and low ASR simultaneously. For all experiments, we use Contriever (Izacard et al., 2021) as the retriever and set the number of retrieved documents to $k = 1 0$ by default. We report the licenses of all models and datasets we used in Appendix C.

Baselines and attacks: We compare RAGSentinel against five representative defenses spanning three major post-retrieval defense paradigms: instruction-based methods that guide the LLM to resist poisoned context (RobustRAG (Xiang et al., 2024), InstructRAG (Wei et al., 2025)); knowledge consolidation methods that reconcile retrieved doc uments through iterative self-reflection or trustwor thiness scoring (AstuteRAG (Wang et al., 2025), TrustRAG (Zhou et al., 2025)); and consistency based filtering methods that detect and remove adversarial documents via cross-document consis tency checking (CrAM (Deng et al., 2025)). We also include Vanilla RAG (no defense) as a ref erence. Detailed descriptions of all baselines are provided in Appendix D. We evaluate all methods under three poisoning attacks from two categories. Optimization-based attacks craft adversarial documents by jointly optimizing for retrieval relevance and answer manipulation, including black-box PoisonedRAG (Zou et al., 2025) and Prompt Injection Attack (PIA) (Zhong et al., 2023; Greshake et al., 2023). Generative attacks produce adver sarial documents via prompted generation without gradient-based optimization, including Adversarial Decoding (AD) (Zhang et al., 2026c). Unless otherwise specified, we inject $k ^ { \prime } = 1$ poisoned document into the top-k retrieved set, keeping the honest-majority assumption $( | I _ { p } | < k / 2 )$ satisfied. Implementation details: We use BGE-M3 (Chen et al., 2024a) as the surrogate encoder $f _ { \varphi }$ by default, chosen independently of the deployed LLM and used solely for hidden-state shift extraction. We ablate the surrogate encoder choice in Table 11 and the retriever choice in Table 10. Unless oth erwise specified, all experiments are conducted across three LLMs: Mistral-7B (Jiang et al., 2023a), Llama-3.1-8B (Grattafiori et al., 2024), and Qwen-2.5-7B (Team, 2024). All results are averaged over three independent runs with different random seeds on three NVIDIA L40S GPUs (48GB).

## 6.2 Experimental results

RAGSentinel achieves strong overall defense performance: Table 1 reports ACC and ASR of all defense methods under three poisoning attacks across three datasets and three LLMs. Overall, RAGSentinel delivers the strongest defense performance across the evaluated settings, achieving the lowest ASR in most cases and remaining close to the best-performing baseline in the remaining cases, while maintaining competitive accuracy un der no-attack conditions. First, under no attack, RAGSentinel maintains competitive accuracy overall, for instance, on MS-MARCO with Mistral-7B, RAGSentinel achieves 0.82 ACC, matching Vanilla RAG exactly, confirming that adaptive filtering does not aggressively discard benign documents under normal retrieval conditions. Second, under PoisonedRAG, RAGSentinel substantially reduces ASR across all settings: on NQ with Llama-3.1-8B, RAGSentinel achieves an ASR of 0.03, compared to 0.38 for Vanilla RAG and 0.06 for the next-best baseline RobustRAG. Third, RAGSentinel generalizes consistently across attack types: under PIA and AD, RAGSentinel continues to attain the lowest ASR in most settings, maintaining ASR at most 0.11 across all three LLMs and datasets, while competing methods such as InstructRAG and CrAM frequently exceed 0.30 ASR, demonstrating that the hidden-state consensus signal remains discriminative regardless of how adversarial documents are constructed. We provide a mechanism-level analysis of why each baseline fails in Appendix E.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="4">NQ</td><td colspan="4">HotpotQA</td><td colspan="4">MS-MARCO</td></tr><tr><td>No attack ACC</td><td>PoisonedRAG ACC/ASR</td><td>PIA ACC/ASR</td><td>AD ACC/ASR</td><td>No attack ACC</td><td>PoisonedRAG ACC/ASR</td><td>PIA ACC/ASR</td><td>AD ACC/ASR</td><td>No attack ACC</td><td>PoisonedRAG ACC/ASR</td><td>PIA ACC/ASR</td><td>AD ACC/ASR</td></tr><tr><td rowspan="7">Mistral-7B</td><td>Vanilla RAG</td><td>0.71</td><td>0.53/0.40</td><td>0.58/0.29</td><td>0.49/0.45</td><td>0.73</td><td>0.38/0.60</td><td>0.42/0.51</td><td>0.40/0.52</td><td>0.82</td><td>0.59/0.34</td><td>0.66/0.24</td><td>0.66/0.29</td></tr><tr><td>RobustRAG</td><td>0.64</td><td>0.69/0.11</td><td>0.68/0.09</td><td>0.69/0.11</td><td>0.57</td><td>0.57/0.30</td><td>0.56/0.31</td><td>0.59/0.26</td><td>0.87</td><td>0.87/0.03</td><td>0.85/0.03</td><td>0.86/0.04</td></tr><tr><td>InstructRAG</td><td>0.72</td><td>0.54/0.40</td><td>0.57/0.29</td><td>0.62/0.29</td><td>0.71</td><td>0.43/0.52</td><td>0.42/0.52</td><td>0.56/0.30</td><td>0.82</td><td>0.61/0.32</td><td>0.70/0.23</td><td>0.70/0.21</td></tr><tr><td>AstuteRAG</td><td>0.69</td><td>0.70/0.10</td><td>0.69/0.03</td><td>0.67/0.12</td><td>0.68</td><td>0.67/0.17</td><td>0.64/0.15</td><td>0.63/0.24</td><td>0.85</td><td>0.81/0.12</td><td>0.85/0.08</td><td>0.83/0.11</td></tr><tr><td>TrustRAG</td><td>0.76</td><td>0.71/0.18</td><td>0.72/0.07</td><td>0.73/0.10</td><td>0.68</td><td>0.68/0.17</td><td>0.71/0.09</td><td>0.70/0.18</td><td>0.77</td><td>0.80/0.14</td><td>0.79/0.12</td><td>0.77/0.17</td></tr><tr><td>CrAM</td><td>0.70</td><td>0.48/0.39</td><td>0.51/0.24</td><td>0.51/0.35</td><td>0.70</td><td>0.34/0.57</td><td>0.38/0.50</td><td>0.36/0.56</td><td>0.76</td><td>0.51/0.38</td><td>0.65/0.19</td><td>0.62/0.26</td></tr><tr><td>RAGSentinel</td><td>0.69</td><td>0.63/0.04</td><td>0.63/0.05</td><td>0.61/0.04</td><td>0.68</td><td>0.63/0.09</td><td>0.64/0.06</td><td>0.63/0.09</td><td>0.82</td><td>0.77/0.05</td><td>0.71/0.04</td><td>0.78/0.04</td></tr><tr><td rowspan="7">Llama-3.1-8B</td><td>Vanilla RAG</td><td>0.78</td><td>0.56/0.38</td><td>0.56/0.36</td><td>0.38/0.57</td><td>0.76</td><td>0.43/0.51</td><td>0.43/0.53</td><td>0.29/0.65</td><td>0.86</td><td>0.60/0.32</td><td>0.66/0.24</td><td>0.42/0.52</td></tr><tr><td>RobustRAG</td><td>0.67</td><td>0.68/0.06</td><td>0.64/0.07</td><td>0.68/0.07</td><td>0.55</td><td>0.64/0.23</td><td>0.65/0.20</td><td>0.68/0.19</td><td>0.79</td><td>0.80/0.06</td><td>0.78/0.07</td><td>0.79/0.06</td></tr><tr><td>InstructRAG</td><td>0.86</td><td>0.66/0.31</td><td>0.70/0.25</td><td>0.65/0.25</td><td>0.79</td><td>0.51/0.43</td><td>0.53/0.44</td><td>0.62/0.34</td><td>0.87</td><td>0.73/0.24</td><td>0.75/0.18</td><td>0.76/0.21</td></tr><tr><td>AstuteRAG</td><td>0.80</td><td>0.71/0.14</td><td>0.77/0.06</td><td>0.70/0.19</td><td>0.71</td><td>0.69/0.19</td><td>0.68/0.17</td><td>0.61/0.28</td><td>0.87</td><td>0.82/0.14</td><td>0.86/0.08</td><td>0.78/0.19</td></tr><tr><td>TrustRAG</td><td>0.81</td><td>0.80/0.09</td><td>0.82/0.07</td><td>0.83/0.07</td><td>0.71</td><td>0.67/0.19</td><td>0.71/0.14</td><td>0.64/0.17</td><td>0.89</td><td>0.89/0.08</td><td>0.87/0.10</td><td>0.86/0.11</td></tr><tr><td>CrAM</td><td>0.75</td><td>0.64/0.29</td><td>0.62/0.28</td><td>0.57/0.33</td><td>0.76</td><td>0.61/0.38</td><td>0.56/0.37</td><td>0.48/0.44</td><td>0.84</td><td>0.70/0.25</td><td>0.75/0.14</td><td>0.64/0.29</td></tr><tr><td>RAGSentinel</td><td>0.71</td><td>0.68/0.03</td><td>0.64/0.04</td><td>0.70/0.05</td><td>0.61</td><td>0.60/0.06</td><td>0.64/0.07</td><td>0.60/0.09</td><td>0.72</td><td>0.71/0.05</td><td>0.73/0.08</td><td>0.72/0.04</td></tr><tr><td rowspan="7">Qwen-2.5-7B</td><td>Vanilla RAG</td><td>0.69</td><td>0.54/0.39</td><td>0.50/0.33</td><td>0.44/0.51</td><td>0.71</td><td>0.35/0.61</td><td>0.46/0.49</td><td>0.30/0.62</td><td>0.77</td><td>0.52/0.34</td><td>0.64/0.24</td><td>0.51/0.45</td></tr><tr><td>RobustRAG</td><td>0.63</td><td>0.58/0.18</td><td>0.57/0.17</td><td>0.57/0.18</td><td>0.55</td><td>0.54/0.34</td><td>0.49/0.40</td><td>0.55/0.33</td><td>0.76</td><td>0.72/0.13</td><td>0.72/0.13</td><td>0.75/0.09</td></tr><tr><td>InstructRAG</td><td>0.72</td><td>0.57/0.38</td><td>0.56/0.31</td><td>0.57/0.40</td><td>0.71</td><td>0.51/0.42</td><td>0.51/0.45</td><td>0.53/0.37</td><td>0.81</td><td>0.53/0.39</td><td>0.67/0.20</td><td>0.64/0.30</td></tr><tr><td>AstuteRAG</td><td>0.70</td><td>0.62/0.15</td><td>0.66/0.05</td><td>0.66/0.12</td><td>0.68</td><td>0.57/0.23</td><td>0.60/0.18</td><td>0.56/0.21</td><td>0.75</td><td>0.63/0.21</td><td>0.71/0.07</td><td>0.61/0.26</td></tr><tr><td>TrustRAG</td><td>0.68</td><td>0.70/0.18</td><td>0.66/0.12</td><td>0.63/0.22</td><td>0.69</td><td>0.61/0.24</td><td>0.65/0.13</td><td>0.62/0.18</td><td>0.70</td><td>0.67/0.20</td><td>0.72/0.11</td><td>0.63/0.27</td></tr><tr><td>CrAM</td><td>0.65</td><td>0.58/0.27</td><td>0.62/0.20</td><td>0.65/0.26</td><td>0.65</td><td>0.56/0.30</td><td>0.45/0.42</td><td>0.48/0.39</td><td>0.75</td><td>0.70/0.19</td><td>0.75/0.10</td><td>0.74/0.14</td></tr><tr><td>RAGSentinel</td><td>0.63</td><td>0.65/0.06</td><td>0.62/0.04</td><td>0.62/0.06</td><td>0.69</td><td>0.64/0.11</td><td>0.63/0.10</td><td>0.59/0.07</td><td>0.74</td><td>0.73/0.09</td><td>0.70/0.09</td><td>0.69/0.09</td></tr></table>

Table 1: ACC(↑) and ASR(↓) of all defense methods under PoisonedRAG, PIA, and AD attacks across three datasets and three LLMs.

Impact of retrieved documents k: Table 2 reports the effect of the number of retrieved documents k, varied from 8 to 18, on RAGSentinel under PoisonedRAG with $k ^ { \prime } = 1$ , using Mistral-7B as the LLM. Overall, RAGSentinel maintains consistently low ASR across all values of k, demonstrating its robustness to the retrieval set size. Notably, as k increases, ASR tends to decrease further, for instance, on NQ, ASR drops from 0.05 at k = 8 to 0.01 at k = 16, which is expected since a larger retrieved set provides a stronger benign majority, making the honest-majority assumption easier to satisfy and the consensus anchor more stable.

<table><tr><td>k</td><td>NQ ACC/ASR</td><td>HotpotQA ACČ/ASR</td><td>MS-MARCO ACC/ASR</td></tr><tr><td>8</td><td>0.60/0.05</td><td>0.60/0.08</td><td>0.69/0.10</td></tr><tr><td>10</td><td>0.63/0.04</td><td>0.63/0.09</td><td>0.77/0.05</td></tr><tr><td>12</td><td>0.61/0.03</td><td>0.66/0.07</td><td>0.74/0.05</td></tr><tr><td>14</td><td>0.66/0.02</td><td>0.61/0.06</td><td>0.73/0.07</td></tr><tr><td>16</td><td>0.68/0.01</td><td>0.60/0.05</td><td>0.75/0.07</td></tr><tr><td>18</td><td>0.71/0.03</td><td>0.63/0.02</td><td>0.75/0.08</td></tr></table>

Table 2: Impact of the number of retrieved documents k on RAGSentinel under PoisonedRAG.

![](images/f91d4e114c786d00b6a6ef34cb24efb9901716605c5b9e03390660632c000560.jpg)  
Figure 2: Impact of the number of poisoned documents on RAGSentinel under PoisonedRAG.

Impact of poisoned document k<sup>′</sup>: Figure 2 shows ACC and ASR of RAGSentinel as the number of poisoned documents increases from 1 to 5 under PoisonedRAG, using Mistral-7B as the LLM. RAGSentinel degrades gracefully within its theoretical operating regime: with up to 2 poisoned documents, ASR remains below 0.14 while ACC stays above 0.56 across all datasets. The crossover where ASR surpasses ACC occurs only when the number of poisoned documents approaches the majority threshold (≥ 4), at which point ACC drops sharply (e.g., to 0.31 on NQ with 5 poisoned documents), consistent with the honest-majority assumption $( | I _ { p } | < k / 2 )$ being nearly violated.

Computational overhead of RAGSentinel: Table 3 reports average inference time per query.

<table><tr><td>Method</td><td>NQ</td><td>HotpotQA</td><td>MS-MARCO</td></tr><tr><td>Vanilla RAG</td><td>0.49</td><td>0.49</td><td>0.48</td></tr><tr><td>RobustRAG</td><td>3.15</td><td>3.62</td><td>2.44</td></tr><tr><td>InstructRAG</td><td>0.89</td><td>0.77</td><td>0.96</td></tr><tr><td>AstuteRAG</td><td>2.29</td><td>2.13</td><td>2.96</td></tr><tr><td>TrustRAG</td><td>2.80</td><td>2.60</td><td>2.54</td></tr><tr><td>CrAM</td><td>1.22</td><td>1.22</td><td>1.06</td></tr><tr><td>RAGSentinel</td><td>0.99</td><td>1.16</td><td>0.85</td></tr></table>

Table 3: Average inference time per query (seconds/query).

RAGSentinel incurs modest overhead over Vanilla RAG, while remaining 3× faster than RobustRAG and 2× faster than AstuteRAG, confirming negligible cost relative to the defense gain achieved.

Document filtering quality: Table 6 (Appendix) reports the detection performance of RAGSentinel, where a document $p _ { i }$ is predicted as poisoned if $d _ { i } > R _ { \mathrm { a d p } }$ and benign otherwise. We measure detection accuracy (DACC), false positive rate (FPR), the fraction of benign documents incorrectly flagged as poisoned, and false negative rate (FNR), the fraction of poisoned documents missed by the filter. RAGSentinel achieves consistently high DACC with low FPR and FNR across all settings. FNR closely tracks ASR in Table 1, validating that system-level attack success is primarily determined by document-level filtering failures. The low FPR further explains the negligible accuracy degradation under no attack in Table 1.

Robustness of RAGSentinel under adaptive attacks: We further evaluate RAGSentinel under three adaptive attacks that assume full knowledge of its algorithmic pipeline. All three extend PoisonedRAG with a post-hoc candidate selection step targeting different filtering components: Anchor mimicry minimizes $d ^ { \mathrm { a n c h o r } }$ to evade the geometric median anchor; Norm boundary keeps shift norm within $\boldsymbol { B } _ { \mathrm { a d p } }$ to preserve adversarial signal after clipping; Subspace camouflage minimizes projection onto the active subspace to bypass consensus scoring. Full design details are provided in Appendix F. As shown in Table 7 (Appendix) and Table 1, adaptive attacks do raise RAGSentinel’s ASR compared to standard PoisonedRAG, for instance, from 0.08 up to 0.11 on HotpotQA, confirming that targeted evasion of individual components is effective. Nevertheless, RAGSentinel delivers the strongest defense performance across all settings and attacks, demonstrating its defensive advantage even under full pipeline knowledge.

Case studies: To illustrate how RAGSentinel operates in practice, we present three case studies in Appendix H (each with $k ^ { \prime } = 2 )$ , one per dataset, each showing the full filtering pipeline: per-document consensus distances $\{ d _ { i } \}$ , the adaptive radius $R _ { \mathrm { a d p } }$ the resulting trusted context ${ \mathcal { C } } ^ { \star }$ , and the final generated answer. Across all three cases, the poisoned documents consistently receive the highest consensus distances and are placed well beyond $R _ { \mathrm { a d p } }$ , while the surviving benign documents directly ground the correct answer.

## 7 Discussion

Component-wise analysis of RAGSentinel: Table 8 (Appendix) ablates each component of RAGSentinel across three datasets and three attacks. Every component contributes positively: replacing the geometric median with arithmetic mean causes the largest ASR increase, confirming that robustness to adversarial outliers in anchor estimation is critical. Removing topic-direction removal, active subspace selection, adaptive norm clipping, or the adaptive radius each leads to consistent ASR degradation, validating that all five components are complementary and jointly responsible for RAGSentinel’s defense performance.

Impact of topic-centering estimator: We compare arithmetic mean-centering with coordinatewise median, geometric median, an oracle benignonly mean, and no topic removal. As shown in Table 9 (Appendix), robust centering provides only marginal improvements over arithmetic meancentering, whereas removing topic removal leads to substantially higher ASR. These results indicate that adaptive norm clipping sufficiently limits the practical bias introduced by poisoned documents, and that removing the shared topic component is substantially more important than the choice of centering estimator.

Impact of retrieval model: Table 10 (Appendix) reports RAGSentinel’s performance under three retrievers (Contriever, Contriever-MS (Izacard et al., 2021), and ANCE (Xiong et al., 2020)) across three LLMs and datasets. RAGSentinel maintains consistently low ASR (below 0.11) and stable ACC across all retriever choices, confirming that RAGSentinel generalizes well regardless of how the top-k documents are retrieved.

Impact of pooling strategy: We evaluate the effect of hidden-state pooling under PoisonedRAG using three surrogate encoders across NQ, HotpotQA, and MS-MARCO. For BGE-M3, we compare CLS, mean, and last-token pooling, while for E5-Mistral-7B and Phi-3.5-mini, we compare mean and lasttoken pooling. As shown in Table 4, RAGSentinel remains effective under alternative pooling strategies, with ASR at most 0.14, indicating that its robustness is not specific to the pooling choices.

<table><tr><td>Surrogate</td><td>Pooling</td><td>NQ</td><td>HotpotQA</td><td>MS-MARCO</td></tr><tr><td rowspan="3">BGE-M3</td><td>CLS</td><td>0.61/0.04</td><td>0.62/0.11</td><td>0.75/0.08</td></tr><tr><td>Mean</td><td>0.60/0.09</td><td>0.58/0.14</td><td>0.73/0.10</td></tr><tr><td>Last token</td><td>0.63/0.04</td><td>0.63/0.09</td><td>0.77/0.05</td></tr><tr><td rowspan="2">E5-Mistral-7B</td><td>Mean</td><td>0.61/0.06</td><td>0.60/0.08</td><td>0.75/0.07</td></tr><tr><td>Last token</td><td>0.64/0.03</td><td>0.64/0.04</td><td>0.78/0.04</td></tr><tr><td rowspan="2">Phi-3.5-mini</td><td>Mean</td><td>0.58/0.08</td><td>0.57/0.09</td><td>0.70/0.12</td></tr><tr><td>Last token</td><td>0.61/0.05</td><td>0.60/0.05</td><td>0.73/0.09</td></tr></table>

Table 4: Impact of pooling strategy across surrogate encoders under PoisonedRAG.

Impact of surrogate encoder: Table 11 (Appendix) reports RAGSentinel’s performance under four surrogate encoder choices, BGE-M3, E5- mistral-7b (Wang et al., 2023a), Phi-3.5-mini (Microsoft, 2024), and Mistral-7B, with Mistral-7B as the deployed LLM throughout. RAGSentinel achieves consistently low ASR across all surrogate models, with ASR below 0.09 in nearly all settings, demonstrating that the defense is not sensitive to the choice of surrogate encoder.

Performance of RAGSentinel under mixed attacks: Table 12 (Appendix) evaluates all methods under three mixed attack settings (PoisonedRAG+PIA, PoisonedRAG+AD, PIA+AD), where two attacks are simultaneously active with 1 poisoned document each. RAGSentinel consistently achieves the lowest ASR across all mixed settings, LLMs, and datasets, with ASR remaining below 0.12 throughout. In contrast, methods that struggle under single attacks degrade further under mixed attacks, for instance, Vanilla RAG reaches ASR of 0.80 on HotpotQA with Mistral-7B under PoisonedRAG+AD, and CrAM similarly collapses to 0.86, demonstrating that RAGSentinel’s consensus-based filtering is robust to simultaneous multi-attack poisoning without any modification.

## 8 Conclusion

We present RAGSentinel, a training-free, labelfree post-retrieval defense that identifies poisoned documents as geometric outliers in a surrogate encoder’s hidden-state residual space, with a certifiable filtering guarantee under an honest-majority assumption. Empirically, RAGSentinel achieves low attack success rates across three datasets, three LLM families, and three attack types, including adaptive attackers with full pipeline knowledge. Future work will extend the defense to higher poison ratios and stronger surrogate encoder assumptions.

## 9 Limitations

RAGSentinel currently assumes the surrogate encoder is inaccessible to the attacker. While our adaptive attack evaluation shows RAGSentinel remains effective when the attacker approximates the surrogate, the stronger setting where the attacker has direct query access to $f _ { \varphi }$ remains an open problem. Additionally, RAGSentinel inherits the honest-majority assumption standard in Byzantinerobust aggregation; extending the defense to higher poison ratios is a natural direction for future work.

## 10 Ethical considerations

RAGSentinel is intended to improve the factual reliability of RAG systems, with clear benefits in high-stakes domains such as medical question answering and legal research. We acknowledge a dual-use concern: our adaptive attack strategies are necessary for rigorous evaluation but could inform more effective adversarial construction; we partially mitigate this by showing RAGSentinel retains its advantage under full pipeline knowledge. Incorrect filtering decisions may cause downstream users to receive confidently wrong answers without visible indication of failure, a risk amplified for users from low-resource language communities whose queries may yield less well-structured residual geometries. Practitioners should deploy RAGSentinel as one layer within a broader pipeline including corpus access control and ongoing monitoring, and should not treat it as a complete solution where the honest-majority assumption may be violated. As a training-free method requiring only k + 1 surrogate forward passes, RAGSentinel introduces negligible computational overhead and no significant environmental cost. All datasets and models are used solely for non-commercial academic research consistent with their respective licenses; the artifacts introduced here are intended exclusively for defensive security research and should not be repurposed for offensive adversarial document construction.

## Acknowledgments

This work was supported by the National Artificial Intelligence Research Resource (NAIRR) Pilot under Award Nos. 250513 and 260142 and by the Texas Higher Education Coordinating Board (THECB) Minority Health Research and Education Grant Program (MHGP) under Award No. 1383.3421.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Maya Anderson, Guy Amit, and Abigail Goldsteen. 2024. Is my data in your retrieval database? membership inference attacks against retrieval augmented generation. arXiv preprint arXiv:2405.20446.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pages 9112–9141.

Amos Azaria and Tom Mitchell. 2023. The internal state of an llm knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976.

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, and 1 others. 2016. Ms marco: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268.

Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. 2017. Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems, 30.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2022. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Preprint, arXiv:2402.03216.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024b. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 17754–17762.

Yudong Chen, Lili Su, and Jiaming Xu. 2017. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1–25.

Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, and Minghong Fang. 2025. Secure retrieval-augmented generation against poisoning attacks. In IEEE International Conference on Big Data.

Boyi Deng, Wenjie Wang, Fengbin Zhu, Qifan Wang, and Fuli Feng. 2025. Cram: Credibility-aware attention modification in llms for combating misinformation in rag. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23760– 23768.

Ilias Diakonikolas, Gautam Kamath, Daniel Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. 2019. Robust estimators in high-dimensions without the computational intractability. SIAM Journal on Computing, 48(2):742–864.

Anjun Gao, Yueyang Quan, Zhuqing Liu, and Minghong Fang. 2026a. Beware what you autocomplete: Forensic attribution of backdoored code completions. In Conference on Language Modeling (COLM).

Anjun Gao, Yueyang Quan, Yufei Xia, Zhuqing Liu, and Minghong Fang. 2026b. Patcher: Post-hoc patching of backdoored large language models. In USENIX Security Symposium.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and 1 others. 2023. Rarr: Researching and revising what language models say, using language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16477–16508.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM workshop on artificial intelligence and security, pages 79–90.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. 2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7036–7050.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and 1 others. 2023a. Mistral 7b. arXiv preprint arXiv:2310.06825.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023b. Active retrieval augmented generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, pages 7969–7992.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: a benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 7:453–466.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. 2018. A simple unified framework for detecting outof-distribution samples and adversarial attacks. Advances in neural information processing systems, 31.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459– 9474.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530.

Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang. 2025. Generating is believing: Membership inference attacks against retrieval-augmented generation. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings ofthe 61st annual meeting of the associationfor computational linguistics (volume 1: Long papers), pages 9802–9822.

Microsoft. 2024. Phi-3.5-mini-instruct. https://huggingface.co/microsoft/Phi-3. 5-mini-instruct. Hugging Face model card.

Fábio Perez and Ian Ribeiro. 2022. Ignore previous prompt: Attack techniques for language models. arXiv preprint arXiv:2211.09527.

Krishna Pillutla, Sham M Kakade, and Zaid Harchaoui. 2022. Robust aggregation for federated learning. IEEE Transactions on Signal Processing, 70:1142– 1154.

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. 2022. Towards total recall in industrial anomaly detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14318– 14328.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, pages 31210–31227. PMLR.

Qwen Team. 2024. Qwen2.5: A party of foundation models.

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. Universal adversarial triggers for attacking and analyzing nlp. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 2153–2162.

Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. 2025. Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 30553–30571.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023a. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368.

Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023b. Self-knowledge guided retrieval augmentation for large language models. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 10303–10315.

Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. Instructrag: Instructing retrieval-augmented generation via self-synthesized rationales. In International Conference on Learning Representations, volume 2025, pages 82731–82754.

Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, and Prateek Mittal. 2024. Certifiably robust rag against retrieval corruption. arXiv preprint arXiv:2405.15556.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808.

language models. In 34th USENIX Security Symposium (USENIX Security 25), pages 3827–3844.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380.

Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong Yu. 2024. Chain-of-note: Enhancing robustness in retrieval-augmented language models. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 14672–14685.

Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu, and Minghong Fang. 2026a. Practical poisoning attacks against retrieval-augmented generation. In ACM Symposium on Access Control Models and Technologies (SACMAT).

Baolei Zhang, Haoran Xin, Yuxi Chen, Zhuqing Liu, Biao Yi, Tong Li, Lihai Nie, Zheli Liu, and Minghong Fang. 2026b. Who taught the lie? responsibility attribution for poisoned knowledge in retrievalaugmented generation. In IEEE Symposium on Security and Privacy.

Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, and Zheli Liu. 2025a. Traceback of poisoning attacks to retrieval-augmented generation. In The Web Conference.

Baolei Zhang, Haoran Xin, Jiatong Li, Dongzhe Zhang, Minghong Fang, Zhuqing Liu, Lihai Nie, and Zheli Liu. 2025b. Benchmarking poisoning attacks against retrieval-augmented generation. arXiv preprint arXiv:2505.18543.

Collin Zhang, Tingwei Zhang, and Vitaly Shmatikov. 2026c. Adversarial decoding: Generating readable documents for adversarial objectives. In Findings of the Association for Computational Linguistics: EACL 2026, pages 2053–2068.

Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. 2023. Poisoning retrieval corpora by injecting adversarial passages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13764–13775.

Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and Emine Yilmaz. 2025. Trustrag: enhancing robustness and trustworthiness in retrieval-augmented generation. arXiv preprint arXiv:2501.00879.

Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. {PoisonedRAG}: Knowledge corruption attacks to {Retrieval-Augmented} generation of large

<table><tr><td>Notation Definition</td><td></td></tr><tr><td> $q$ </td><td>User query</td></tr><tr><td> $\{ p _ { i } \} _ { i = 1 } ^ { k }$ </td><td>Set of k retrieved documents</td></tr><tr><td> $k , k ^ { \prime }$ </td><td>Number of retrieved documents; number of injected poisoned documents</td></tr><tr><td> $f _ { \varphi } , f _ { \theta }$ </td><td>Surrogate encoder for hidden-state extraction; deployed black-box LLM</td></tr><tr><td> $\delta _ { i }$ </td><td>Hidden-state shift induced by document  $p _ { i }$ </td></tr><tr><td> $z _ { i }$ </td><td>Residual vector after adaptive norm clipping and topic-direction removal</td></tr><tr><td> $c$ </td><td>Geometric median anchor estimating the benign majority consensus direction</td></tr><tr><td> $d _ { i }$ </td><td>Consensus distance of document  $p _ { i }$ </td></tr><tr><td> $C ^ { \star }$ </td><td>Trusted context passed to  $f _ { \theta }$  for answer generation</td></tr></table>

Table 5: Key notation used in the paper.

Algorithm 1 RAGSentinel   
Require: User query q; top-k retrieved documents $\{ p _ { i } \} _ { i = 1 } ^ { k }$ ; surrogate encoder $f _ { \varphi }$ with hidden-state   
encoder $\phi _ { \varphi } ;$ black-box LLM $f _ { \theta }$   
Ensure: Trusted context ${ \mathcal { C } } ^ { \star } ;$ ; answer $\hat { y }$   
1: // Phase 1: Shift extraction and preprocessing   
2: h<sub>0</sub> $ \phi _ { \varphi } ( q )$   
3: for $i = 1 , \dots ,$ k do   
4: $h _ { i }  \phi _ { \varphi } ( q , p _ { i } ) ; \delta _ { i }  h _ { i } - h _ { 0 }$   
5: end for   
6: Select active subspace V and project $\tilde { \delta } _ { i } \gets \delta _ { i } [ \gamma ]$   
7: Clip shift norms: $\bar { \delta } _ { i }  \tilde { \delta } _ { i }$ · min $\left( 1 , B _ { \mathrm { a d p } } / ( \lVert \tilde { \delta } _ { i } \rVert _ { 2 } + \epsilon ) \right)$   
8: Remove topic direction: $\begin{array} { r } { z _ { i } \gets \bar { \delta } _ { i } - \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \bar { \delta } _ { j } } \end{array}$   
9: // Phase 2: Consensus scoring   
10: c ← GeoMed $( \{ z _ { i } \} _ { i = 1 } ^ { k } ) ;$ d<sup>anchor</sup> $ 1 - \cos ( z _ { i } , c )$   
11: m $\begin{array} { r } {  \operatorname* { m a x } ( 1 , \lceil k / 2 \rceil - 1 ) ; d _ { i } ^ { \mathrm { l o c a l } }  \frac { 1 } { m } \sum _ { j \in \mathcal { N } _ { i } ^ { ( m ) } } D _ { i j } } \end{array}$   
12: Compute adaptive weight λ and combine: ${ d _ { i } } ^ { \circ } \gets \left( 1 - \lambda \right) d _ { i } ^ { \mathrm { a n c h o r } } + \lambda d _ { i } ^ { \mathrm { l o c a l } }$ ▷ Eq. (1)   
13: // Phase 3: Adaptive filtering and context construction   
14: Compute adaptive radius $R _ { \mathrm { a d p } }$ from $\{ d _ { i } \}$ ▷ Eq. (2)   
15: $S \gets \{ i : d _ { i } \leq R _ { \mathrm { a d p } } \}$ ; if $S = \emptyset$ then $S \gets$ {arg min<sub>i</sub> $d _ { i } \}$   
16: $\mathcal { C } ^ { \star } \gets \mathrm { t o p } { - } \lceil k / 2 \rceil$ entries of $S$ sorted by $d _ { i }$ ascending   
17: $\hat { y } \gets f _ { \theta } ( q , \mathcal { C } ^ { \star } )$   
18: return $\mathcal { C } ^ { \star } , \hat { y }$

![](images/af9662b6e50ca3d8f0fda888f46cb9a83813eb86d8c9f196bdbc104aba7ae323.jpg)

![](images/5b720a05a7d9e64aa8967c18dadf0144115cd1d0e6757701fe8abf29c41f5a07.jpg)  
(b) HotpotQA  
Figure 3: Poisoned documents are geometric outliers in the topic-removed residual space with ten retrieved documents including two injected poisoned documents. NQ query: “who played miss wheeler in carry on teacher”; HotpotQA query: “The mass killing that took place at Oakland, California on April 2, 2012 was less deadly than the one that took place on October 1, 2015 in which state?” ● benign, ◆ poisoned, ★ geometric median anchor c.

![](images/359f8ec06c6419be1821a7c6918d4946cffd9a5d4606182f34c11f38766fcbdc.jpg)  
Figure 4: Overview of the RAGSentinel pipeline. An attacker injects poisoned documents into the knowledge database (left). The surrogate encoder encodes each query-document pair and extracts hidden-state shifts (Phase 1). Consensus scoring identifies poisoned documents as geometric outliers via the global anchor (Phase 2). Adaptive filtering keeps a trusted context C<sup>⋆</sup> (Phase 3), which is passed to the black-box LLM for answer generation.

## A Extended related work

This appendix expands on Section 2 with detailed descriptions of attack mechanisms, the three defense families, and the structural limitations that motivate RAGSentinel.

## A.1 Knowledge poisoning attacks

Knowledge poisoning attacks inject adversarial documents crafted to rank highly for target queries while encoding wrong answers (Zhong et al., 2023; Zou et al., 2025; Greshake et al., 2023). Optimization-based attacks jointly optimize retrieval relevance and answer manipulation via adversarial triggers (Zou et al., 2025; Zhong et al., 2023; Wallace et al., 2019); generative attacks (Zhang et al., 2026c) instead produce fluent adversarial documents via prompted generation, eliminating token-level artifacts. Beyond answer manipulation, indirect prompt injection (Greshake et al., 2023; Perez and Ribeiro, 2022) hijacks LLM behavior through instructions embedded in retrieved documents, and white-box membership inference attacks exploit retrieval scores to infer database membership (Anderson et al., 2024; Li et al., 2025). Because poisoned documents are designed to appear topically relevant, fluent, and factually plausible, any defense operating solely on surface-level text signals operates on exactly the space the attacker controls.

## A.2 Post-retrieval defenses

Instruction-based methods: RobustRAG (Xiang et al., 2024) isolates each document, aggregates responses via keyword or decoding-based algorithms, and admits certifiable guarantees, but discards the cross-document consensus signal and incurs k-fold LLM calls, precisely the signal RAGSentinel exploits. InstructRAG (Wei et al., 2025) prompts the LLM to self-synthesize rationales for denoising, usable as in-context demonstrations or fine-tuning data. Self-RAG (Asai et al., 2024) introduces reflection tokens for dynamic per-document critique; Chain-of-Note (Yu et al., 2024) generates reading notes per document to improve robustness to irrelevant context (Shi et al., 2023). All instructionbased methods delegate conflict detection to the same LLM that adversarial documents are crafted to deceive: a plausible poisoned document can produce a confident but wrong rationale indistinguishable from a benign one. RAGSentinel avoids this by making its filtering decision entirely in the surrogate encoder’s representation space, before the LLM is invoked.

Knowledge consolidation methods: AstuteRAG (Wang et al., 2025) adaptively elicits parametric knowledge, iteratively consolidates it with retrieved documents in a source-aware manner, and filters by assessed reliability, requiring no training and supporting black-box LLMs. TrustRAG (Zhou et al., 2025) first clusters embeddings via K-means to remove attack patterns, then applies cosine similarity, ROUGE, and LLM self-assessment to detect remaining malicious documents; however, the clustering stage breaks down under mixed attacks where simultaneously injected documents do not form a coherent cluster. Earlier methods similarly leverage parametric knowledge to decide when to trust external documents (Wang et al., 2023b; Jeong et al., 2024; Mallen et al., 2023). All consolidation methods assume parametric knowledge serves as a reliable arbiter, failing when that knowledge is absent or when poisoned documents are crafted to align with parametric priors (Wang et al., 2025). RAGSentinel makes no use of parametric knowledge and is therefore immune to both failure modes.

Consistency-based methods: CrAM (Deng et al., 2025) identifies influential attention heads via causal tracing and attenuates their weights for lowcredibility tokens, but requires white-box access incompatible with our threat model and leaves adversarial content in context where it can still influence generation through other heads. Active retrieval (Jiang et al., 2023b) dynamically adjusts retrieval based on cross-document consistency signals. Both assume adversarial documents are textlevel inconsistent with the benign majority, an assumption a capable attacker defeats by crafting textually consistent documents that encode a contradictory answer. RAGSentinel sidesteps this by operating in the hidden-state residual space, where text-level consistency can be mimicked but geometric stance cannot be erased.

## A.3 Geometric filtering foundations

All three defense families share a common blind spot that motivates RAGSentinel’s geometric approach: the hidden-state geometry of an independent encoder lies outside the attacker’s optimization surface, unlike token outputs, parametric knowledge conflicts, or surface-level text overlap. Byzantine-robust aggregation: Beyond the honest-majority guarantee noted in Section 2, the geometric median has been extended to highdimensional settings (Pillutla et al., 2022) and robust mean estimation (Diakonikolas et al., 2019). The key transfer replaces gradient vectors with hidden-state shift vectors and worker identities with retrieved document indices; the honest-majority condition translates directly from the federated to the retrieval setting.

Representation-based probing: Prior work probes the deployed LLM’s internal states to detect likely false outputs (Azaria and Mitchell, 2023; Burns et al., 2022; Li et al., 2023). RAGSentinel differs in two respects: it uses an independent surrogate encoder, preventing adversarial pressure from transferring to the probing signal; and it applies the signal to retrieval filtering rather than outputlevel truthfulness detection, intercepting poisoned documents before the LLM is invoked.

Outlier detection in embedding space: Existing methods (Roth et al., 2022; Lee et al., 2018) model a static distribution over document representations and flag deviations query-independently. RAGSentinel instead conditions the residual space on the query via topic-direction removal and active subspace selection, so the relevant geometry reflects factual stance toward the specific query rather than generic document similarity.

## B Proof of Theorem 1

We provide the full proof of Theorem 1. The proof first derives upper and lower bounds on the anchor and local consensus distances for benign and poisoned documents, and then shows that the adaptive majority-radius rule preserves the resulting separation.

Throughout the proof, define

$$
D _ { b } = \operatorname* { m a x } _ { i \in \mathscr { T } _ { b } } d _ { i } , \quad D _ { p } = \operatorname* { m i n } _ { j \in \mathscr { T } _ { p } } d _ { j } ,\tag{4}
$$

where $D _ { b }$ is the largest benign consensus distance and $D _ { p }$ is the smallest poisoned consensus distance. Recall that R is the $\lceil k / 2 \rceil$ -th smallest value among $\{ d _ { i } \} _ { i = 1 } ^ { k }$

## B.1 Role of the proof quantities

We first explain the quantities used in Theorem 1. The neighborhood size $m = \operatorname* { m a x } ( 1 , \lceil k / 2 \rceil - 1 )$ is inherited from the local-consensus score in $d _ { i } ^ { \mathrm { l o c a l } } =$ $\begin{array} { r } { \frac { 1 } { m } \sum _ { j \in \mathcal { N } _ { i } ^ { ( m ) } } D _ { i j } } \end{array}$ . This choice ensures that, under the honest-majority condition $k ^ { \prime } < k / 2$ , the mneighborhood of any poisoned document contains at least $m - k ^ { \prime } + 1$ benign documents.

The quantity

$$
\eta _ { c } = \frac { k } { k - 2 k ^ { \prime } } \Delta _ { E }\tag{5}
$$

comes from the robustness of the geometric median. Since the benign residuals have Euclidean diameter at most $\Delta _ { E }$ and poisoned residuals are fewer than half of the retrieved set, Lemma 3 shows that the geometric-median anchor c is displaced from the benign centroid $\mu _ { b }$ by at most $\eta _ { c }$

The condition $r _ { 0 } > \eta _ { c }$ ensures that the anchor remains non-degenerate. Indeed, Assumption 2 gives $\left\| \mu _ { b } \right\| _ { 2 } \ge r _ { 0 }$ , while Lemma 3 gives $\| c -$ $\mu _ { b } \| _ { 2 } \le \eta _ { c }$ . Therefore,

$$
\| c \| _ { 2 } \geq \| \mu _ { b } \| _ { 2 } - \| c - \mu _ { b } \| _ { 2 } \geq r _ { 0 } - \eta _ { c } > 0 .\tag{6}
$$

This positive lower bound is needed because the anchor distance is defined through cosine similarity.

The quantity

$$
U _ { \mathrm { a n c } } = \sqrt { 2 \Delta _ { \mu } ^ { \mathrm { c o s } } } + \frac { 2 \eta _ { c } } { r _ { 0 } - \eta _ { c } }\tag{7}
$$

upper bounds the anchor distance of benign documents. The first term controls the angular deviation between a benign residual $z _ { i }$ and the benign centroid $\mu _ { b } .$ , while the second term controls the displacement between $\mu _ { b }$ and the geometric-median anchor c after normalization.

The quantity

$$
L _ { \mathrm { a n c } } = { \frac { 1 } { 2 } } \left( { \sqrt { 2 \gamma } } - { \sqrt { 2 U _ { \mathrm { a n c } } } } \right) _ { + } ^ { 2 }\tag{8}
$$

lower bounds the anchor distance of poisoned documents. It follows from the reverse triangle inequality in normalized residual space: a poisoned residual is at least $\sqrt { 2 \gamma }$ away from every benign residual, while a benign residual is at most $\sqrt { 2 U _ { \mathrm { a n c } } }$ away from the anchor.

The local term contributes the separation

$$
{ \frac { m - k ^ { \prime } + 1 } { m } } \gamma - \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } .\tag{9}
$$

The first term lower bounds the local distance of poisoned documents because their mneighborhood contains at least $m - k ^ { \prime } + 1$ benign documents. The second term upper bounds the local distance of benign documents by the benign pairwise angular spread.

Combining the anchor gap and the local gap with the adaptive weight λ gives $G _ { \mathrm { f u l l } } .$ which lower bounds the full consensus-score gap $D _ { p } - D _ { b }$ . Similarly, $D _ { b } ^ { + }$ upper bounds the largest benign consensus distance $D _ { b }$ . The lemmas below make these statements precise.

## B.2 Auxiliary inequalities

Lemma 1 (Cosine-distance identity) For any nonzero vectors u and v,

$$
1 - \cos ( u , v ) = { \frac { 1 } { 2 } } \left\| { \frac { u } { \| u \| _ { 2 } } } - { \frac { v } { \| v \| _ { 2 } } } \right\| _ { 2 } ^ { 2 } .\tag{10}
$$

proof: Let $\hat { u } = u / \lVert u \rVert _ { 2 }$ and $\boldsymbol { \hat { v } } = \boldsymbol { v } / \| \boldsymbol { v } \| _ { 2 }$ . Since $\| \hat { u } \| _ { 2 } = \| \hat { v } \| _ { 2 } = 1$ , we have

$$
\begin{array} { r l } & { \| \hat { \boldsymbol u } - \hat { \boldsymbol v } \| _ { 2 } ^ { 2 } = \| \hat { \boldsymbol u } \| _ { 2 } ^ { 2 } + \| \hat { \boldsymbol v } \| _ { 2 } ^ { 2 } - 2 \hat { \boldsymbol u } ^ { \top } \hat { \boldsymbol v } } \\ & { \qquad = 2 - 2 \cos ( { \boldsymbol u } , { \boldsymbol v } ) = 2 ( 1 - \cos ( { \boldsymbol u } , { \boldsymbol v } ) ) . } \end{array}\tag{11}
$$

Dividing both sides by 2 proves the claim.

## Lemma 2 (Normalization Lipschitz bound)

For any nonzero vectors u and v satisfying min $( \| u \| _ { 2 } , \| v \| _ { 2 } ) \geq \rho > 0$ , we have

$$
\left. { \frac { u } { \| u \| _ { 2 } } } - { \frac { v } { \| v \| _ { 2 } } } \right. _ { 2 } \leq { \frac { 2 \| u - v \| _ { 2 } } { \rho } } .\tag{12}
$$

proof: Let $\alpha _ { u } = \| u \| _ { 2 }$ and $\alpha _ { v } = \| v \| _ { 2 }$ . Then

$$
\begin{array} { r l r } {  { \| \frac { u } { \alpha _ { u } } - \frac { v } { \alpha _ { v } } \| _ { 2 } = \| \frac { u - v } { \alpha _ { u } } + v ( \frac { 1 } { \alpha _ { u } } - \frac { 1 } { \alpha _ { v } } ) \| _ { 2 } } } \\ & { } & { \leq \frac { \| u - v \| _ { 2 } } { \alpha _ { u } } + \| v \| _ { 2 } | \frac { 1 } { \alpha _ { u } } - \frac { 1 } { \alpha _ { v } } | } \\ & { } & { = \frac { \| u - v \| _ { 2 } } { \alpha _ { u } } + \alpha _ { v } \frac { | \alpha _ { v } - \alpha _ { u } | } { \alpha _ { u } \alpha _ { v } } } \\ & { } & { = \frac { \| u - v \| _ { 2 } } { \alpha _ { u } } + \frac { | \alpha _ { v } - \alpha _ { u } | } { \alpha _ { u } } . \quad ( 1 3 \sqrt { \alpha _ { v } + \alpha _ { v } } ) } \end{array}
$$

By the reverse triangle inequality, $| \alpha _ { v } - \alpha _ { u } | \ \leq$ $\lVert u - v \rVert _ { 2 }$ . Since $\alpha _ { u } \geq \rho ,$ , we obtain

$$
\left. { \frac { u } { \| u \| _ { 2 } } } - { \frac { v } { \| v \| _ { 2 } } } \right. _ { 2 } \leq { \frac { 2 \| u - v \| _ { 2 } } { \rho } } .\tag{14}
$$

This proves the lemma.

## B.3 Geometric-median stability

Lemma 3 (Geometric-median stability) Let

$$
\mu _ { b } = \frac { 1 } { | \mathcal { T } _ { b } | } \sum _ { i \in \mathcal { T } _ { b } } z _ { i }\tag{15}
$$

and let

$$
\Delta _ { E } = \operatorname* { m a x } _ { i , j \in \mathcal { T } _ { b } } \Vert z _ { i } - z _ { j } \Vert _ { 2 } .\tag{16}
$$

$I f k ^ { \prime } < k / 2$ and $c = \operatorname { G e o M e d } ( \{ z _ { i } \} _ { i = 1 } ^ { k } )$ , then

$$
\| c - \mu _ { b } \| _ { 2 } \leq \eta _ { c } = \frac { k } { k - 2 k ^ { \prime } } \Delta _ { E } .\tag{17}
$$

proof: Let

$$
F ( x ) = \sum _ { l = 1 } ^ { k } \| x - z _ { l } \| _ { 2 } .\tag{18}
$$

Since c is a geometric median, c minimizes F. Let $v = c - \mu _ { b }$ and $t = \| v \| _ { 2 }$

First, for any benign $i \in \mathcal { Z } _ { b }$

$$
\begin{array} { r l r } {  { \| \mu _ { b } - z _ { i } \| _ { 2 } = \| \frac { 1 } { | { \mathcal T } _ { b } | } \sum _ { l \in { \mathcal T } _ { b } } ( z _ { l } - z _ { i } ) \| _ { 2 } } } \\ & { } & { \leq \frac { 1 } { | { \mathcal T } _ { b } | } \sum _ { l \in { \mathcal T } _ { b } } \| z _ { l } - z _ { i } \| _ { 2 } \leq \Delta _ { E } . } \end{array}\tag{19}
$$

If $t \le \Delta _ { E }$ , then Eq. (17) holds because $k / ( k -$ $2 k ^ { \prime } ) > 1$ . Hence it remains to consider the case $t > \Delta _ { E }$

If $c = z _ { i }$ for some benign $i \in \mathcal { Z } _ { b }$ , then

$$
t = \| z _ { i } - \mu _ { b } \| _ { 2 } \leq \Delta _ { E } ,\tag{20}
$$

contradicting $t > \Delta _ { E }$ . Therefore, under $t > \Delta _ { E } ,$ c cannot coincide with a benign point. It may coincide with a poisoned point, so we use the subgradient optimality condition for $F .$

There exist vectors $g _ { l }$ such that

$$
0 = \sum _ { l = 1 } ^ { k } g _ { l } ,\tag{21}
$$

where

$$
g _ { l } = \frac { c - z _ { l } } { \| c - z _ { l } \| _ { 2 } } \quad \mathrm { i f } c \neq z _ { l } ; \quad \| g _ { l } \| _ { 2 } \leq 1 \quad \mathrm { i f } c = z _ { l }\tag{22}
$$

Since c does not coincide with any benign point, for every $i \in \mathcal { Z } _ { b }$ ,

$$
g _ { i } = { \frac { c - z _ { i } } { \| c - z _ { i } \| _ { 2 } } } .\tag{23}
$$

Taking the inner product of Eq. (21) with v gives

$$
\sum _ { i \in \mathcal { T } _ { b } } \frac { \boldsymbol { v } ^ { \top } ( \boldsymbol { c } - \boldsymbol { z } _ { i } ) } { \Vert \boldsymbol { c } - \boldsymbol { z } _ { i } \Vert _ { 2 } } = - \sum _ { j \in \mathcal { T } _ { p } } \boldsymbol { v } ^ { \top } \boldsymbol { g } _ { j } .\tag{24}
$$

For any benign i, write

$$
c - z _ { i } = v + ( \mu _ { b } - z _ { i } ) .\tag{25}
$$

Using Eq. (19) and Cauchy-Schwarz,

$$
\begin{array} { r l } & { v ^ { \top } ( c - z _ { i } ) = \| v \| _ { 2 } ^ { 2 } + v ^ { \top } ( \mu _ { b } - z _ { i } ) } \\ & { ~ \geq t ^ { 2 } - t \Delta _ { E } = t ( t - \Delta _ { E } ) . } \end{array}\tag{26}
$$

Moreover, by the triangle inequality,

$$
\| c - z _ { i } \| _ { 2 } \leq \| c - \mu _ { b } \| _ { 2 } + \| \mu _ { b } - z _ { i } \| _ { 2 } \leq t + \Delta _ { E } .\tag{27}
$$

Since $t > \Delta _ { E }$ , the numerator in Eq. (26) is positive. Thus

$$
\frac { v ^ { \top } ( c - z _ { i } ) } { \| c - z _ { i } \| _ { 2 } } \geq t \frac { t - \Delta _ { E } } { t + \Delta _ { E } } .\tag{28}
$$

Summing over all benign indices yields

$$
\sum _ { i \in \mathcal { I } _ { b } } \frac { v ^ { \top } ( c - z _ { i } ) } { \Vert c - z _ { i } \Vert _ { 2 } } \geq ( k - k ^ { \prime } ) t \frac { t - \Delta _ { E } } { t + \Delta _ { E } } .\tag{29}
$$

For the poisoned terms, Eq. (22) gives $\| g _ { j } \| _ { 2 } \leq 1$ for every $j \in \mathcal { I } _ { p }$ . Hence

$$
\left| \sum _ { j \in \mathcal { T } _ { p } } \boldsymbol { v } ^ { \top } \boldsymbol { g } _ { j } \right| \leq \sum _ { j \in \mathcal { T } _ { p } } \| \boldsymbol { v } \| _ { 2 } \| g _ { j } \| _ { 2 } \leq k ^ { \prime } t .\tag{30}
$$

Combining Eqs. (24), (29), and (30), we obtain

$$
( k - k ^ { \prime } ) t \frac { t - \Delta _ { E } } { t + \Delta _ { E } } \leq k ^ { \prime } t .\tag{31}
$$

Since $t > 0$ , dividing by t and rearranging gives

$$
\begin{array} { c } { { ( k - k ^ { \prime } ) ( t - \Delta _ { E } ) \leq k ^ { \prime } ( t + \Delta _ { E } ) , } } \\ { { ( k - 2 k ^ { \prime } ) t \leq k \Delta _ { E } . } } \end{array}\tag{32}
$$

Because $k ^ { \prime } < k / 2$ , we have $k - 2 k ^ { \prime } > 0$ , and therefore

$$
t \leq \frac { k } { k - 2 k ^ { \prime } } \Delta _ { E } = \eta _ { c } .\tag{33}
$$

This proves the lemma.

## B.4 Anchor-distance bounds

Lemma 4 (Benign anchor-distance upper bound) Under Assumptions 1 and 2, $i f r _ { 0 } > \eta _ { c } ,$ , then

$$
d _ { i } ^ { \mathrm { a n c h o r } } \leq U _ { \mathrm { a n c } } , \quad \forall i \in  { \mathbb { Z } } _ { b } .\tag{34}
$$

proof: By Lemma 3,

$$
\| c - \mu _ { b } \| _ { 2 } \leq \eta _ { c } .\tag{35}
$$

Since $\| \mu _ { b } \| _ { 2 } \ge r _ { 0 }$ by Assumption 2, the reverse triangle inequality gives

$$
\| c \| _ { 2 } \geq \| \mu _ { b } \| _ { 2 } - \| c - \mu _ { b } \| _ { 2 } \geq r _ { 0 } - \eta _ { c } > 0 .\tag{36}
$$

Fix any $i \in \mathcal { Z } _ { b }$ . By Lemma 1,

$$
d _ { i } ^ { \mathrm { a n c h o r } } = 1 - \cos ( z _ { i } , c ) = \frac 1 2 \left. \frac { z _ { i } } { \lVert z _ { i } \rVert _ { 2 } } - \frac { c } { \lVert c \rVert _ { 2 } } \right. _ { 2 } ^ { 2 } .\tag{37}
$$

The distance between two unit vectors is at most 2, and ${ \textstyle { \frac { 1 } { 2 } } } x ^ { 2 } \leq x$ for all $x \in [ 0 , 2 ]$ . Therefore

$$
d _ { i } ^ { \mathrm { a n c h o r } } \leq \left\| { \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } } - { \frac { c } { \| c \| _ { 2 } } } \right\| _ { 2 } .\tag{38}
$$

By the triangle inequality,

$$
\begin{array} { r } { \left\| \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } - \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } \leq \left\| \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } - \frac { \mu _ { b } } { \| \mu _ { b } \| _ { 2 } } \right\| _ { 2 } } \\ { + \left\| \frac { \mu _ { b } } { \| \mu _ { b } \| _ { 2 } } - \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } . } \end{array}\tag{39}
$$

For the first term, Assumption 2 and Lemma 1 give

$$
\begin{array} { r } { \left\| \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } - \frac { \mu _ { b } } { \| \mu _ { b } \| _ { 2 } } \right\| _ { 2 } = \sqrt { 2 ( 1 - \cos ( z _ { i } , \mu _ { b } ) ) } } \\ { \leq \sqrt { 2 \Delta _ { \mu } ^ { \mathrm { c o s } } } . \qquad ( } \end{array}\tag{40}
$$

For the second term, Lemma 2, Eq. (35), and the lower bounds $\lVert \mu _ { b } \rVert _ { 2 } \ge r _ { 0 }$ and $\| c \| _ { 2 } \ge r _ { 0 } - \eta _ { c }$ imply

$$
\left\| \frac { \mu _ { b } } { \| \mu _ { b } \| _ { 2 } } - \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } \leq \frac { 2 \| \mu _ { b } - c \| _ { 2 } } { r _ { 0 } - \eta _ { c } } \leq \frac { 2 \eta _ { c } } { r _ { 0 } - \eta _ { c } } .\tag{41}
$$

Combining Eqs. (38)-(41) yields

$$
d _ { i } ^ { \mathrm { a n c h o r } } \leq \sqrt { 2 \Delta _ { \mu } ^ { \mathrm { c o s } } } + \frac { 2 \eta _ { c } } { r _ { 0 } - \eta _ { c } } = U _ { \mathrm { a n c } } .\tag{42}
$$

This proves the lemma.

Lemma 5 (Poison anchor lower bound) Under Assumption 3 and the conclusion ofLemma 4,

$$
d _ { j } ^ { \mathrm { a n c h o r } } \geq L _ { \mathrm { a n c } } , \quad \forall j \in \mathbb { Z } _ { p } .\tag{43}
$$

proof: Fix any $j \in \mathcal { I } _ { p }$ and any $i \in \mathcal { Z } _ { b }$ . By Assumption 3 and Lemma 1,

$$
\begin{array} { r l r } {  { \| \frac { z _ { j } } { \| z _ { j } \| _ { 2 } } - \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } \| _ { 2 } = \sqrt { 2 ( 1 - \cos ( z _ { j } , z _ { i } ) ) } } } \\ & { } & { \geq \sqrt { 2 \gamma } . \qquad ( } \end{array}\tag{44}
$$

By Lemma 4,

$$
d _ { i } ^ { \mathrm { a n c h o r } } = \frac { 1 } { 2 } \left\| \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } - \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } ^ { 2 } \leq U _ { \mathrm { a n c } } ,\tag{45}
$$

and therefore

$$
\left. \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } - \frac { c } { \| c \| _ { 2 } } \right. _ { 2 } \leq \sqrt { 2 U _ { \mathrm { a n c } } } .\tag{46}
$$

Using the reverse triangle inequality,

$$
\begin{array} { r l r } & { \left\| \displaystyle \frac { z _ { j } } { \| z _ { j } \| _ { 2 } } - \displaystyle \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } } & \\ & { \geq \left\| \displaystyle \frac { z _ { j } } { \| z _ { j } \| _ { 2 } } - \displaystyle \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } \right\| _ { 2 } - \left\| \displaystyle \frac { z _ { i } } { \| z _ { i } \| _ { 2 } } - \displaystyle \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } } & \\ & { \geq \sqrt { 2 \gamma } - \sqrt { 2 U _ { \mathrm { a n c } } } . } & { ( \lambda } \end{array}\tag{47}
$$

Since the left-hand side is nonnegative,

$$
\left\| { \frac { z _ { j } } { \| z _ { j } \| _ { 2 } } } - { \frac { c } { \| c \| _ { 2 } } } \right\| _ { 2 } \geq \left( { \sqrt { 2 \gamma } } - { \sqrt { 2 U _ { \mathrm { a n c } } } } \right) _ { + } .\tag{48}
$$

Applying Lemma 1 again gives

$$
\begin{array} { l } { \displaystyle { d _ { j } ^ { \mathrm { a n c h o r } } = \frac { 1 } { 2 } \left\| \frac { z _ { j } } { \| z _ { j } \| _ { 2 } } - \frac { c } { \| c \| _ { 2 } } \right\| _ { 2 } ^ { 2 } } } \\ { \displaystyle { \phantom { \frac { 1 } { 2 } } \geq \frac { 1 } { 2 } \left( \sqrt { 2 \gamma } - \sqrt { 2 U _ { \mathrm { a n c } } } \right) _ { + } ^ { 2 } = L _ { \mathrm { a n c } } } . } \end{array}\tag{49}
$$

This proves the lemma.

## B.5 Local-distance bounds

Lemma 6 (Local-distance bounds) Suppose Assumptions 1, 2, and 3 hold. $I f \gamma > \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } }$ , then

$$
d _ { i } ^ { \mathrm { l o c a l } } \leq \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } , \quad \forall i \in \mathcal { T } _ { b } ,\tag{50}
$$

and

$$
d _ { j } ^ { \mathrm { l o c a l } } \geq \frac { m - k ^ { \prime } + 1 } { m } \gamma , \quad \forall j \in \mathcal { T } _ { p } .\tag{51}
$$

proof: First fix a benign document $i \in \mathcal { T } _ { b }$ . Since $k ^ { \prime } < k / 2$ , the number of benign documents other than i is $k - k ^ { \prime } - 1$ . For the nontrivial case $k ^ { \prime } \ge 1$ we have $k \geq 3$ , and

$$
k - k ^ { \prime } - 1 \geq \lceil k / 2 \rceil - 1 = m .\tag{52}
$$

If $k ^ { \prime } = 0$ , there is no poisoned document and the poison-exclusion part of the theorem is immediate; the following argument is used for the nontrivial poisoned case. Thus each benign document has at least m benign neighbours available.

For any benign pair $i , l \ \in \ \mathcal { T } _ { b } .$ Assumption 2 gives

$$
1 - \cos ( z _ { i } , z _ { l } ) \leq \Delta _ { \mathrm { p a i r } } ^ { \cos } .\tag{53}
$$

For any poisoned $j \in \mathcal { T } _ { p }$ and benign $i \in \mathcal { T } _ { b }$ , Assumption 3 gives

$$
1 - \cos ( z _ { j } , z _ { i } ) \geq \gamma > \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } .\tag{54}
$$

Thus the m nearest neighbours of a benign residual can be chosen among benign residuals, and all corresponding distances are at most $\Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } }$ . Therefore,

$$
d _ { i } ^ { \mathrm { l o c a l } } = \frac { 1 } { m } \sum _ { l \in \mathcal { N } _ { i } ^ { ( m ) } } \bigl ( 1 - \cos ( z _ { i } , z _ { l } ) \bigr ) \leq \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } .\tag{55}
$$

Now fix a poisoned document $j \in \mathcal { T } _ { p }$ . Among the $k - 1$ documents other than $j ,$ , at most $k ^ { \prime } - 1$ are poisoned. Since $k ^ { \prime } < k / 2$ and $k ^ { \prime }$ is an integer,

$$
k ^ { \prime } \leq \left\lfloor { \frac { k - 1 } { 2 } } \right\rfloor = \lceil k / 2 \rceil - 1 \leq m .\tag{56}
$$

Therefore, any m-neighbour set of $j$ contains at least $m - ( k ^ { \prime } - 1 ) = m - k ^ { \prime } + 1$ benign residuals. Each benign neighbour contributes at least γ to the pairwise cosine distance by Assumption 3, while all cosine distances are nonnegative. Hence,

$$
\begin{array} { l } { d _ { j } ^ { \mathrm { l o c a l } } = \displaystyle \frac { 1 } { m } \sum _ { l \in  { \mathcal N } _ { j } ^ { ( m ) } } \big ( 1 - \cos ( z _ { j } , z _ { l } ) \big ) } \\ { \geq \displaystyle \frac { m - k ^ { \prime } + 1 } { m } \gamma . } \end{array}\tag{57}
$$

This proves the lemma.

## B.6 Proof of Theorem 1

We first upper bound the largest benign consensus distance. For any $i \in \mathcal { Z } _ { b }$ , Lemmas 4 and 6 imply

$$
\begin{array} { r l } & { d _ { i } = ( 1 - \lambda ) d _ { i } ^ { \mathrm { a n c h o r } } + \lambda d _ { i } ^ { \mathrm { l o c a l } } } \\ & { \phantom { = } \leq ( 1 - \lambda ) U _ { \mathrm { a n c } } + \lambda \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } = D _ { b } ^ { + } . } \end{array}\tag{58}
$$

Taking the maximum over $i \in \mathcal { Z } _ { b }$ gives

$$
D _ { b } \leq D _ { b } ^ { + } .\tag{59}
$$

Next, for any $j \in \mathcal { I } _ { p }$ , Lemmas 5 and 6 imply

$$
\begin{array} { l } { { d _ { j } = ( 1 - \lambda ) d _ { j } ^ { \mathrm { a n c h o r } } + \lambda d _ { j } ^ { \mathrm { l o c a l } } } } \\ { { \mathrm { \textcircled { 1 } } \mathrm { \textcircled { - \lambda } } \mathrm { L a n c } + \lambda \frac { m - k ^ { \prime } + 1 } { m } \gamma . } } \end{array}\tag{60}
$$

Taking the minimum over $j \in \mathcal { I } _ { p }$ and subtracting the benign upper bound gives

$$
\begin{array} { r l } & { D _ { p } - D _ { b } \geq ( 1 - \lambda ) ( L _ { \mathrm { a n c } } - U _ { \mathrm { a n c } } ) } \\ & { \qquad + \lambda \left( \displaystyle \frac { m - k ^ { \prime } + 1 } { m } \gamma - \Delta _ { \mathrm { p a i r } } ^ { \mathrm { c o s } } \right) } \\ & { \qquad = G _ { \mathrm { f u l l } } . } \end{array}\tag{61}
$$

By the condition of Theorem 1,

$$
G _ { \mathrm { f u l l } } > \frac { D _ { b } ^ { + } } { 1 + \sigma _ { d } } .\tag{62}
$$

Combining Eqs. (59), (61), and (62) gives

$$
D _ { p } - D _ { b } > \frac { D _ { b } } { 1 + \sigma _ { d } } .\tag{63}
$$

Let R be the $\lceil k / 2 \rceil$ -th smallest value among $\{ d _ { i } \} _ { i = 1 } ^ { k }$ . Since $| \mathcal { T } _ { p } | = k ^ { \prime } < k / 2$ , the number of benign documents satisfies $| \mathcal { T } _ { b } | = k - k ^ { \prime } \ge \lceil k / 2 \rceil$ All benign documents have consensus distance at most $D _ { b }$ , so at least $\lceil k / 2 \rceil$ entries of $\{ d _ { i } \} _ { i = 1 } ^ { k }$ are at most $D _ { b }$ . Therefore,

$$
R \leq D _ { b } .\tag{64}
$$

By Eq. (2), the adaptive radius is

$$
R _ { \mathrm { a d p } } = \left( 1 + \frac { 1 } { 1 + \sigma _ { d } } \right) R .\tag{65}
$$

Using Eq. (64),

$$
\begin{array} { c } { { R _ { \mathrm { a d p } } \leq \left( 1 + \displaystyle \frac { 1 } { 1 + \sigma _ { d } } \right) D _ { b } } } \\ { { = D _ { b } + \displaystyle \frac { D _ { b } } { 1 + \sigma _ { d } } } } \\ { { < D _ { p } , } } \end{array}\tag{66}
$$

where the last inequality follows from Eq. (63). Let

$$
S = \{ i : d _ { i } \leq R _ { \mathrm { a d p } } \}\tag{67}
$$

be the surviving set before final context construction. Since every poisoned $j \in \mathcal { I } _ { p }$ satisfies $d _ { j } \geq$ $D _ { p } > R _ { \mathrm { a d p } }$ , no poisoned document survives:

$$
S \cap \mathbb { Z } _ { p } = \emptyset .\tag{68}
$$

Thus

$$
S \subseteq { \mathbb { Z } } _ { b } .\tag{69}
$$

It remains to show that the final context has size $\lceil k / 2 \rceil$ . Since $\sigma _ { d } \geq 0$ , Eq. (65) implies

$$
R _ { \mathrm { a d p } } \geq R .\tag{70}
$$

Because R is the $\lceil k / 2 \rceil$ -th smallest value, at least $\lceil k / 2 \rceil$ indices satisfy $d _ { i } \leq R .$ By Eq. (70), those indices also satisfy $d _ { i } \leq R _ { \mathrm { a d p } } .$ , and hence

$$
\begin{array} { r } { | S | \ge \lceil k / 2 \rceil . } \end{array}\tag{71}
$$

Algorithm 1 constructs ${ \mathcal { C } } ^ { \star }$ by selecting the top min $( | S | , \lceil k / 2 \rceil )$ entries of $S$ sorted by ascending consensus distance. Since $| S | \ge \lceil k / 2 \rceil$ , we have

$$
| { \mathcal { C } } ^ { \star } | = \lceil k / 2 \rceil .\tag{72}
$$

Moreover, since ${ \mathcal { C } } ^ { \star } \subseteq S$ and $S \subseteq { \mathcal { T } } _ { b }$ , we obtain

$$
\begin{array} { r } { \mathcal { C } ^ { \star } \subseteq \mathcal { T } _ { b } , \quad \mathcal { C } ^ { \star } \cap \mathcal { T } _ { p } = \emptyset . } \end{array}\tag{73}
$$

Combining Eqs. (72) and (73) proves

$$
\mathcal { C } ^ { \star } \cap \mathcal { T } _ { p } = \emptyset , \quad | \mathcal { C } ^ { \star } | = \lceil k / 2 \rceil , \quad \mathcal { C } ^ { \star } \subseteq \mathcal { T } _ { b } .\tag{74}
$$

This completes the proof of Theorem 1.

## C Licenses of models and datasets

Datasets: We evaluate on three publicly available benchmarks. Natural Questions (NQ) (Kwiatkowski et al., 2019) is released under the CC BY-SA 3.0 license. HotpotQA (Yang et al., 2018) is released under the CC BY-SA 4.0 license. MS-MARCO (Bajaj et al., 2016) is released by Microsoft under the MS-MARCO Dataset Terms of Use for non-commercial research purposes. All three datasets are used solely for research evaluation consistent with their respective terms.

Models: Contriever and Contriever-MS (Izacard et al., 2021) are released under the CC BY-NC 4.0 license. ANCE (Xiong et al., 2020) is released under the MIT license. BGE-M3 (Chen et al., 2024a) is released under the MIT license. E5- mistral-7b (Wang et al., 2023a) is released under the MIT license. Phi-3.5-mini (Microsoft, 2024) is released by Microsoft under the MIT license. Mistral-7B (Jiang et al., 2023a) is released under the Apache 2.0 license. Llama-3.1-8B (Grattafiori et al., 2024) is released by Meta under the Llama 3.1 Community License, permitting research use. Qwen-2.5-7B (Team, 2024) is released under the Apache 2.0 license. All models are used in accordance with their respective licenses for research purposes only.

## D Details of baselines

RobustRAG (Xiang et al., 2024): It is an instruction-based defense against retrieval corruption attacks. It adopts an isolate-then-aggregate strategy: each retrieved document is independently fed to the LLM to obtain an isolated response, and these responses are then securely aggregated via keyword-based or decoding-based algorithms to produce the final answer. This design provides certifiable robustness guarantees, formally proving that accurate responses are always returned even when the attacker has full knowledge of the defense and injects a bounded number of malicious documents.

InstructRAG (Wei et al., 2025): It is an instruction-based defense that addresses noisy retrieval by making the denoising process explicit. Rather than directly predicting answers from potentially noisy documents, InstructRAG prompts the LLM to generate self-synthesized rationales that analyze each retrieved document and articulate how the ground-truth answer is derived. These rationales can serve either as in-context learning demonstrations or as supervised fine-tuning data, enabling the model to explicitly learn to denoise retrieved contents without additional human annotation.

AstuteRAG (Wang et al., 2025): It is a knowledge consolidation method designed to resolve conflicts between the LLM’s internal parametric knowledge and externally retrieved documents. AstuteRAG operates in three steps: it first adaptively elicits the LLM’s internal knowledge, then iteratively consolidates internal and external knowledge in a sourceaware manner by combining consistent information and identifying conflicting information, and finally generates answers according to assessed information reliability. It requires no model training and is compatible with black-box LLMs.

TrustRAG (Zhou et al., 2025): It is a plug-andplay, training-free knowledge consolidation defense against poisoning attacks. TrustRAG employs a two-stage mechanism: the first stage applies K-means clustering over semantic embeddings of retrieved documents to identify and remove surfaced attack patterns; the second stage leverages cosine similarity and ROUGE metrics together with an LLM self-assessment process to detect remaining malicious documents and resolve inconsistencies between external content and the model’s internal knowledge.

CrAM (Deng et al., 2025): It is a consistencybased, plug-and-play method for credibility-aware RAG. CrAM first identifies influential attention heads by extending causal tracing to estimate each head’s contribution to generating incorrect answers over a small calibration set. At inference time, it scales down the attention weights of retrieved document tokens proportionally to their normalized credibility scores in the identified heads, reducing the influence of low-credibility documents without any model fine-tuning or additional inference calls.

## E Why baselines fail: a mechanism-level analysis

The failure modes in Table 1 reflect structural limitations in each baseline’s design. Instructionbased methods (RobustRAG, InstructRAG) either process documents in isolation, discarding the cross-document consensus signal that could override a poisoned document, or rely on the LLM’s own judgment to identify contradictions, which fails precisely when adversarial documents are crafted to sound plausible. Knowledge consolidation methods (AstuteRAG, TrustRAG) depend on the LLM’s parametric knowledge to adjudicate conflicts: when that knowledge is weak or absent, the poisoned document wins; TrustRAG’s K-means clustering additionally breaks under mixed attacks (Table 12), where two simultaneously injected documents from different LLMs do not form a coherent cluster. CrAM’s attention-weight attenuation requires white-box causal tracing to identify relevant heads, incompatible with our black-box threat model, and, even when approximated, leaves adversarial content in the context where it can still influence generation through other heads, explaining its near-Vanilla-RAG ASR in many settings. RAGSentinel avoids all of these failure modes by making its filtering decision entirely in the surrogate encoder’s representation space, before the LLM is invoked, with no dependence on the LLM’s parametric knowledge, white-box access, or document-level isolation.

## F Adaptive attack designs

All three adaptive attacks extend PoisonedRAG (Zou et al., 2025) with a post-hoc candidate selection stage. For each target query, PoisonedRAG first generates a pool of $N = 5 0$ adversarial candidates, each containing the target wrong answer and crafted to achieve high retrieval relevance. The adaptive attacker then simulates RAGSentinel’s preprocessing using BGE-M3 as a surrogate encoder and selects the single candidate most likely to evade RAGSentinel’s filtering. All three attackers have full knowledge of RAGSentinel’s algorithmic steps but do not know which surrogate encoder the defender employs; BGE-M3 is used as an approximation of the defender’s preprocessing.

Anchor mimicry: it targets the geometric median anchor computation in Phase 2. The attacker simulates RAGSentinel’s full preprocessing pipelineshift extraction, active subspace projection, adaptive norm clipping, and topic-direction removalusing BGE-M3 to obtain topic-removed residuals $\left\{ z _ { i } \right\}$ . It then estimates the geometric median anchor c over the residuals of the benign retrieved documents and computes $d ^ { \mathrm { a n c h o r } }$ for each adversarial candidate. The candidate minimizing $d ^ { i }$ anchor is selected, as a small anchor distance indicates that the document’s residual shift aligns with the benign consensus direction, allowing it to impersonate a consensus-supporting document and evade

the global anchor signal.

Norm boundary: it targets the adaptive norm clipping step in Phase 1. Documents whose projected shift norm exceeds $B _ { \mathrm { a d p } } = \mathrm { M e d i a n } ( \{ b _ { i } \} ) +$ $\mathrm { M A D } ( \{ b _ { i } \} )$ are rescaled, attenuating their adversarial signal. The attacker estimates $\boldsymbol { B } _ { \mathrm { a d p } }$ from the benign retrieved documents using BGE-M3 and selects the candidate satisfying $b _ { i } \leq B _ { \mathrm { a d p } }$ while maximizing $b _ { i }$ within this constraint, ensuring the adversarial signal is preserved at full strength without triggering rescaling. Note that $\boldsymbol { B } _ { \mathrm { a d p } }$ is computed dynamically from the full retrieved set including the poisoned document itself; the attacker’s estimation from benign documents alone introduces a slight approximation.

Subspace camouflage: it targets the active subspace selection step in Phase 1. The active dimensions $\nu$ are chosen based on the peak responsiveness of each dimension across the retrieved documents, and all subsequent consensus scoring operates exclusively within this subspace. The attacker estimates V using BGE-M3 on the benign retrieved documents and computes the projection energy of each adversarial candidate onto $\nu$ as $\mathrm { p r o j } _ { i } = \| \delta _ { i } [ \mathcal { V } ] \| _ { 2 } / \| \delta _ { i } \| _ { 2 }$ The candidate minimizing proj is selected, as a small projection onto the active subspace renders the document’s residual shift nearly invisible to both the anchor distance and local consistency computations, allowing it to pass the adaptive filtering with an artificially low consensus distance.

## G Prompt template for RAGSentinel

We describe the three prompt templates used in RAGSentinel. The first two are passed exclusively to the surrogate encoder $f _ { \varphi }$ during the filtering phase and are never seen by the black-box LLM $f _ { \theta }$ . The third is the generation prompt passed to $f _ { \theta }$ after trusted context construction.

Query-only prompt: Used to obtain the baseline hidden state $h _ { 0 } = \phi _ { \varphi } ( q )$ from the surrogate encoder.

Answer the following question.   
Question: [q]   
Answer:

Document-conditioned prompt: Used to obtain the document-conditioned hidden state $h _ { i }$ = $\phi _ { \varphi } ( q , p _ { i } )$ for each retrieved document $p _ { i }$ . Applying the same QA template structure to both prompts ensures that $h _ { 0 }$ and $h _ { i }$ are extracted under identical pooling conditions and are thus directly comparable.

Prompt 2 (Document-conditioned)   
Answer the following question given the   
information in the context.   
Context: $[ p _ { i } ]$   
Question: [q]   
Answer:

RAG generation prompt: After adaptive filtering, the trusted context ${ \mathcal { C } } ^ { \star }$ is assembled by concatenating the surviving documents in ascending order of consensus distance $d _ { i }$ and inserted into the following zero-shot prompt, which is passed to the black-box LLM $f _ { \theta }$ in a single call.

## Prompt 3 (RAG Generation)

You are a helpful assistant answering with retrieved evidence that may contain noise. Use the context when it is relevant and factual, but do not copy claims that are isolated, misleading, or inconsistent with reliable knowledge. Answer the exact question asked; prefer direct evidence over related background or eventual outcomes. Only output a short and concise answer.

Context: $[ { \mathcal { C } } ^ { \star } ]$

Question: [q]

Answer:

## H Case studies

![](images/e355fea2a925721a0ba4541377a47357f8c8723f0491221b7f1261261b815ae8.jpg)  
Case Study 1: NQ under PoisonedRAG with $k ^ { \prime } = 2$ poisoned documents. Both injected documents falsely attribute the cardiac and vasomotor centers to the cerebellum. RAGSentinel assigns them consensus distances ≈2.6× beyond $R _ { \mathrm { a d p } }$ and removes both, while the five kept benign documents directly support the correct answer.

![](images/77943ed0938f1b2767d6a100965c4ea84b1201de0de20212f4b51d6fade19b90.jpg)  
Case Study 2: HotpotQA under PoisonedRAG with k<sup>′</sup> = 2 poisoned documents. Both injected documents falsely attribute the episode to FOX. RAGSentinel assigns them consensus distances ≈1.4× beyond $R _ { \mathrm { a d p } }$ and removes both. Three benign documents whose content is irrelevant to the query (docs 4, 7, 8) are further excluded by the top- $\lceil k / 2 \rceil$ selection step, leaving a focused context that correctly identifies NBC.

![](images/9c5a5bf33c6bfd35a42f93e9ea7b4a2aceea925b1726060d4e261d5299e23eac.jpg)  
Case Study 3: MS-MARCO under PoisonedRAG with $k ^ { \prime } = 2$ poisoned documents. Both injected documents falsely claim the IRA withdrawal age is 45. The benign cluster is unusually tight $( d _ { i } \leq 0 . 6 9 )$ , amplifying the separation of the two poisoned documents $( 1 . 7 \times - 2 . 2 \times$ beyond $R _ { \mathrm { a d p } } )$ . All five documents in ${ \mathcal { C } } ^ { \star }$ explicitly cite the 59 ½ rule, directly grounding the correct answer.

<table><tr><td rowspan="2">Attack</td><td colspan="3">NQ</td><td colspan="3">HotpotQA</td><td colspan="3">MS-MARCO</td></tr><tr><td>DACC↑</td><td>FPR↓</td><td>FNR↓</td><td>DACC↑</td><td>FPR↓</td><td>FNR↓</td><td>DACC↑</td><td>FPR↓</td><td>FNR↓</td></tr><tr><td>PoisonedRAG</td><td>0.941</td><td>0.06</td><td>0.05</td><td>0.963</td><td>0.03</td><td>0.10</td><td>0.940</td><td>0.06</td><td>0.06</td></tr><tr><td>PIA</td><td>0.957</td><td>0.04</td><td>0.07</td><td>0.975</td><td>0.02</td><td>0.07</td><td>0.968</td><td>0.03</td><td>0.05</td></tr><tr><td>AD</td><td>0.968</td><td>0.03</td><td>0.05</td><td>0.981</td><td>0.01</td><td>0.10</td><td>0.977</td><td>0.02</td><td>0.05</td></tr></table>

Table 6: Document filtering performance of RAGSentinel (DACC↑, FPR↓, FNR↓) under three poisoning attacks across three datasets on Mistral-7B.

<table><tr><td rowspan="2">Method</td><td colspan="3">NQ</td><td colspan="3">HotpotQA</td><td colspan="3">MS-MARCO</td></tr><tr><td>Anchor mimicry ACC/ASR</td><td>Norm boundary ACC/ASR</td><td>Subspace camouflage ACC/ASR</td><td>Anchor mimicry ACC/ASR</td><td>Norm boundary ACC/ASR</td><td>Subspace camouflage ACC/ASR</td><td>Anchor mimicry ACC/ASR</td><td>Norm boundary ACC/ASR</td><td>Subspace camouflage ACC/ASR</td></tr><tr><td>Vanilla RAG</td><td>0.54/0.38</td><td>0.56/0.37</td><td>0.49/0.43</td><td>0.45/0.52</td><td>0.48/0.52</td><td>0.44/0.53</td><td>0.59/0.34</td><td>0.58/0.33</td><td>0.60/0.31</td></tr><tr><td>RobustRAG</td><td>0.66/0.15</td><td>0.68/0.12</td><td>0.69/0.12</td><td>0.57/0.30</td><td>0.58/0.28</td><td>0.56/0.30</td><td>0.87/0.03</td><td>0.86/0.04</td><td>0.87/0.03</td></tr><tr><td>InstructRAG</td><td>0.55/0.39</td><td>0.56/0.36</td><td>0.54/0.38</td><td>0.44/0.49</td><td>0.45/0.49</td><td>0.42/0.50</td><td>0.59/0.35</td><td>0.56/0.36</td><td>0.59/0.32</td></tr><tr><td>AstuteRAG</td><td>0.70/0.08</td><td>0.69/0.09</td><td>0.70/0.12</td><td>0.66/0.17</td><td>0.67/0.16</td><td>0.67/0.16</td><td>0.80/0.12</td><td>0.80/0.12</td><td>0.80/0.11</td></tr><tr><td>TrustRAG</td><td>0.72/0.12</td><td>0.67/0.16</td><td>0.70/0.12</td><td>0.68/0.17</td><td>0.68/0.18</td><td>0.69/0.14</td><td>0.75/0.21</td><td>0.76/0.16</td><td>0.79/0.18</td></tr><tr><td>CrAM</td><td>0.43/0.45</td><td>0.50/0.36</td><td>0.43/0.45</td><td>0.41/0.51</td><td>0.38/0.54 0.64/0.11</td><td>0.39/0.53 0.65/0.08</td><td>0.54/0.33</td><td>0.56/0.30 0.72/0.11</td><td>0.55/0.36</td></tr><tr><td>RAGSentinel</td><td>0.59/0.05</td><td>0.63/0.03</td><td>0.59/0.06</td><td>0.59/0.11</td><td></td><td></td><td>0.78/0.07</td><td></td><td>0.81/0.07</td></tr></table>

Table 7: Performance comparison under three adaptive attacks on NQ, HotpotQA, and MS-MARCO.

<table><tr><td rowspan="2">Variant</td><td colspan="4">NQ</td><td colspan="4">HotpotQA</td><td colspan="4">MS-MARCO</td></tr><tr><td>No attack ACC</td><td>PoisonedRAG ACC/ASR</td><td>PIA ACC/ASR</td><td>AD ACC/ASR</td><td>No attack ACC</td><td>PoisonedRAG ACC/ASR</td><td>PIA ACC/ASR</td><td>AD ACC/ASR</td><td>No attack ACC</td><td>PoisonedRAG ACC/ASR</td><td>PIA ACC/ASR</td><td>AD ACC/ASR</td></tr><tr><td>w/ Arithmetic Mean</td><td>0.68</td><td>0.55/0.08</td><td>0.52/0.13</td><td>0.58/0.09</td><td>0.63</td><td>0.57/0.09</td><td>0.59/0.13</td><td>0.66/0.12</td><td>0.80</td><td>0.82/0.13</td><td>0.69/0.11</td><td>0.77/0.13</td></tr><tr><td>w/o Topic-Direction Removal</td><td>0.62</td><td>0.58/0.09</td><td>0.63/0.11</td><td>0.58/0.09</td><td>0.61</td><td>0.55/0.12</td><td>0.57/0.12</td><td>0.60/0.09</td><td>0.80</td><td>0.79/0.09</td><td>0.68/0.11</td><td>0.77/0.09</td></tr><tr><td>w/o Active Subspace Selection</td><td>0.63</td><td>0.53/0.12</td><td>0.54/0.09</td><td>0.52/0.10</td><td>0.63</td><td>0.63/0.09</td><td>0.53/0.12</td><td>0.54/0.15</td><td>0.80</td><td>0.74/0.08</td><td>0.70/0.15</td><td>0.74/0.08</td></tr><tr><td>w/o Adaptive Norm Clipping</td><td>0.65</td><td>0.61/0.07</td><td>0.63/0.10</td><td>0.58/0.07</td><td>0.67</td><td>0.54/0.17</td><td>0.56/0.09</td><td>0.57/0.11</td><td>0.82</td><td>0.76/0.10</td><td>0.70/0.14</td><td>0.72/0.14</td></tr><tr><td>w/ Fixed Radius</td><td>0.63</td><td>0.59/0.11</td><td>0.63/0.10</td><td>0.61/0.08</td><td>0.66</td><td>0.58/0.09</td><td>0.54/0.10</td><td>0.57/0.11</td><td>0.82</td><td>0.71/0.11</td><td>0.69/0.09</td><td>0.80/0.09</td></tr><tr><td>RAGSentinel</td><td>0.69</td><td>0.63/0.04</td><td>0.63/0.05</td><td>0.61/0.04</td><td>0.68</td><td>0.63/0.09</td><td>0.64/0.06</td><td>0.63/0.09</td><td>0.82</td><td>0.77/0.05</td><td>0.71/0.04</td><td>0.78/0.04</td></tr></table>

Table 8: Ablation study of RAGSentinel components across three datasets.

<table><tr><td>Centering method</td><td> $k ^ { \prime } = 1$ </td><td> $k ^ { \prime } = 2$ </td><td> $k ^ { \prime } = 3$ </td><td> $k ^ { \prime } = 4$ </td></tr><tr><td>No topic removal</td><td>0.64/0.10</td><td>0.56/0.19</td><td>0.46/0.31</td><td>0.37/0.47</td></tr><tr><td>Coordinate-wise median</td><td>0.67/0.06</td><td>0.61/0.11</td><td>0.53/0.21</td><td>0.44/0.35</td></tr><tr><td>Geometric median</td><td>0.67/0.05</td><td>0.62/0.11</td><td>0.54/0.20</td><td>0.45/0.34</td></tr><tr><td>Arithmetic mean</td><td>0.68/0.06</td><td>0.61/0.12</td><td>0.53/0.22</td><td>0.44/0.36</td></tr><tr><td>Oracle benign mean</td><td>0.69/0.05</td><td>0.62/0.10</td><td>0.55/0.19</td><td>0.46/0.33</td></tr></table>

Table 9: Impact of topic-centering estimators under PoisonedRAG as the number of poisoned documents increases. Results are averaged across NQ, HotpotQA, and MS-MARCO.
<table><tr><td rowspan="2">Retriever</td><td rowspan="2">Backbone LLM</td><td>NQ</td><td>HotpotQA</td><td>MS-MARCO</td></tr><tr><td>ACC/ASR</td><td>ACC/ASR</td><td>ACC/ASR</td></tr><tr><td rowspan="3">Contriever</td><td>Mistral-7B</td><td>0.63/0.04</td><td>0.63/0.09</td><td>0.77/0.05</td></tr><tr><td>Llama-3.1-8B</td><td>0.68/0.03</td><td>0.60/0.06</td><td>0.71/0.05</td></tr><tr><td>Qwen-2.5-7B</td><td>0.65/0.06</td><td>0.64/0.11</td><td>0.73/0.09</td></tr><tr><td rowspan="3">Contriever-MS</td><td>Mistral-7B</td><td>0.65/0.05</td><td>0.66/0.07</td><td>0.76/0.07</td></tr><tr><td>Llama-3.1-8B</td><td>0.73/0.06</td><td>0.61/0.05</td><td>0.76/0.08</td></tr><tr><td>Qwen-2.5-7B</td><td>0.65/0.04</td><td>0.59/0.06</td><td>0.79/0.06</td></tr><tr><td rowspan="3">ANCE</td><td>Mistral-7B</td><td>0.69/0.06</td><td>0.63/0.07</td><td>0.80/0.04</td></tr><tr><td>Llama-3.1-8B</td><td>0.79/0.03</td><td>0.59/0.08</td><td>0.75/0.08</td></tr><tr><td>Qwen-2.5-7B</td><td>0.60/0.08</td><td>0.58/0.11</td><td>0.74/0.06</td></tr></table>

Table 10: Effect of retrieval model on RAGSentinel across three datasets under PoisonedRAG.

<table><tr><td rowspan="3">Surrogate Model</td><td colspan="4">NQ</td><td colspan="4">HotpotQA</td><td colspan="4">MS-MARCO</td></tr><tr><td>No attack</td><td>PoisonedRAG</td><td>PIA</td><td>AD</td><td>No attack</td><td>PoisonedRAG</td><td>PIA</td><td>AD</td><td>No attack</td><td>PoisonedRAG</td><td>PIA</td><td>AD</td></tr><tr><td>ACC</td><td>ACC/ASR</td><td>ACC/ASR</td><td>ACC/ASR</td><td>ACC</td><td>ACC/ASR</td><td>ACC/ASR</td><td>ACC/ASR</td><td>ACC</td><td>ACC/ASR</td><td>ACC/ASR</td><td>ACC/ASR</td></tr><tr><td>BGE-M3</td><td>0.69</td><td>0.63/0.04</td><td>0.63/0.05</td><td>0.61/0.04</td><td>0.68</td><td>0.63/0.09</td><td>0.64/0.06</td><td>0.63/0.09</td><td>0.82</td><td>0.77/0.05</td><td>0.71/0.04</td><td>0.78/0.04</td></tr><tr><td>E5-mistral-7b</td><td>0.66</td><td>0.64/0.03</td><td>0.66/0.03</td><td>0.67/0.03</td><td>0.64</td><td>0.64/0.04</td><td>0.64/0.07</td><td>0.63/0.05</td><td>0.78</td><td>0.78/0.04</td><td>0.74/0.05</td><td>0.76/0.05</td></tr><tr><td>Phi-3.5-mini</td><td>0.64</td><td>0.61/0.05</td><td>0.64/0.06</td><td>0.64/0.03</td><td>0.65</td><td>0.60/0.05</td><td>0.57/0.05</td><td>0.59/0.05</td><td>0.81</td><td>0.73/0.09</td><td>0.80/0.09</td><td>0.76/0.08</td></tr><tr><td>Mistral-7B</td><td>0.67</td><td>0.64/0.01</td><td>0.63/0.01</td><td>0.64/0.02</td><td>0.65</td><td>0.61/0.08</td><td>0.63/0.06</td><td>0.65/0.09</td><td>0.73</td><td>0.72/0.03</td><td>0.73/0.04</td><td>0.72/0.06</td></tr></table>

Table 11: Impact of surrogate encoder choice on RAGSentinel across three datasets.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">NQ PoisonedRAG+AD</td><td colspan="3">HotpotQA PoisonedRAG+AD</td><td colspan="3">MS-MARCO PoisonedRAG+AD</td></tr><tr><td>PoisonedRAG+PIA ACC/ASR</td><td>ACC/ASR</td><td>PIA+AD ACC/ASR</td><td>PoisonedRAG+PIA ACC/ASR</td><td>ACC/ASR</td><td>PIA+AD ACC/ASR</td><td>PoisonedRAG+PIA ACC/ASR</td><td>ACC/ASR</td><td>PIA+AD ACC/ASR</td></tr><tr><td rowspan="8">Mistral-7B</td><td>Vanilla RAG</td><td>0.49/0.46</td><td>0.37/0.60</td><td>0.35/0.62</td><td>0.35/0.65</td><td>0.18/0.80</td><td>0.24/0.75</td><td>0.43/0.47</td><td>0.42/0.53</td><td>0.43/0.52</td></tr><tr><td>RobustRAG</td><td>0.59/0.25</td><td>0.60/0.26</td><td>0.60/0.27</td><td>0.46/0.44</td><td>0.50/0.42</td><td>0.53/0.38</td><td>0.82/0.09</td><td>0.83/0.09</td><td>0.81/0.10</td></tr><tr><td>InstructRAG</td><td>0.47/0.48</td><td>0.44/0.50</td><td>0.43/0.52</td><td>0.40/0.55</td><td>0.37/0.60</td><td>0.36/0.62</td><td>0.56/0.38</td><td>0.46/0.49</td><td>0.56/0.39</td></tr><tr><td>AstuteRAG</td><td>0.71/0.15</td><td>0.65/0.25</td><td>0.65/0.20</td><td>0.61/0.27</td><td>0.52/0.42</td><td>0.53/0.40</td><td>0.78/0.15</td><td>0.68/0.29</td><td>0.71/0.19</td></tr><tr><td>TrustRAG</td><td>0.69/0.17</td><td>0.73/0.13</td><td>0.69/0.11</td><td>0.66/0.17</td><td>0.69/0.09</td><td>0.68/0.15</td><td>0.77/0.13</td><td>0.77/0.14</td><td>0.78/0.17</td></tr><tr><td>CrAM</td><td>0.30/0.64</td><td>0.26/0.72</td><td>0.24/0.68</td><td>0.26/0.72</td><td>0.14/0.86</td><td>0.19/0.79</td><td>0.36/0.56</td><td>0.35/0.62</td><td>0.44/0.46</td></tr><tr><td>RAGSentinel</td><td>0.64/0.11</td><td>0.64/0.08</td><td>0.60/0.07</td><td>0.63/0.06</td><td>0.64/0.06</td><td>0.60/0.11</td><td>0.77/0.06</td><td>0.79/0.05</td><td>0.70/0.07</td></tr><tr><td>Vanilla RAG</td><td>0.54/0.39</td><td>0.31/0.65</td><td>0.29/0.69</td><td>0.44/0.52</td><td>0.31/0.62</td><td>0.33/0.63</td><td>0.52/0.39</td><td>0.42/0.55</td><td>0.36/0.60</td></tr><tr><td rowspan="8">Llama-3.1-8B</td><td>RobustRAG</td><td>0.63/0.20</td><td>0.61/0.20</td><td>0.64/0.20</td><td>0.66/0.24</td><td>0.67/0.22</td><td>0.66/0.22</td><td>0.75/0.10</td><td>0.80/0.07</td><td>0.77/0.08</td></tr><tr><td>InstructRAG</td><td>0.64/0.32</td><td>0.62/0.35</td><td>0.56/0.42</td><td>0.57/0.41</td><td>0.51/0.47</td><td>0.48/0.49</td><td>0.67/0.31</td><td>0.66/0.32</td><td>0.63/0.35</td></tr><tr><td>AstuteRAG</td><td>0.73/0.13</td><td>0.64/0.26</td><td>0.62/0.29</td><td>0.65/0.24</td><td>0.59/0.31</td><td>0.56/0.34</td><td>0.82/0.12</td><td>0.72/0.23</td><td>0.75/0.22</td></tr><tr><td>TrustRAG</td><td>0.76/0.18</td><td>0.74/0.14</td><td>0.71/0.15</td><td>0.67/0.14</td><td>0.67/0.10</td><td>0.71/0.15</td><td>0.88/0.09</td><td>0.88/0.08</td><td>0.88/0.10</td></tr><tr><td>CrAM</td><td>0.60/0.36</td><td>0.52/0.47</td><td>0.45/0.53</td><td>0.54/0.45</td><td>0.43/0.55</td><td>0.42/0.56</td><td>0.65/0.31</td><td>0.55/0.40</td><td>0.56/0.38</td></tr><tr><td>RAGSentinel</td><td>0.62/0.09</td><td>0.64/0.09</td><td>0.67/0.08</td><td>0.65/0.04</td><td>0.63/0.06</td><td>0.68/0.12</td><td>0.75/0.04</td><td>0.79/0.05</td><td>0.79/0.03</td></tr><tr><td>Vanilla RAG</td><td>0.46/0.49</td><td>0.26/0.71</td><td>0.33/0.66</td><td>0.35/0.61</td><td>0.16/0.82</td><td>0.23/0.75</td><td></td><td></td><td></td></tr><tr><td>RobustRAG</td><td>0.45/0.38</td><td>0.46/0.36</td><td>0.46/0.38</td><td></td><td></td><td></td><td>0.42/0.46 0.64/0.22</td><td>0.27/0.69</td><td>0.30/0.68</td></tr><tr><td rowspan="6">Qwen-2.5-7B</td><td></td><td>0.48/0.47</td><td></td><td></td><td>0.43/0.51</td><td>0.46/0.46</td><td>0.42/0.52</td><td></td><td>0.66/0.22</td><td>0.63/0.23</td></tr><tr><td>InstructRAG</td><td>0.61/0.18</td><td>0.41/0.58 0.57/0.29</td><td>0.41/0.58</td><td>0.39/0.59</td><td>0.33/0.66</td><td>0.31/0.64</td><td>0.49/0.44 0.62/0.22</td><td>0.51/0.47</td><td>0.48/0.48</td></tr><tr><td>AstuteRAG</td><td>0.59/0.19</td><td>0.69/0.11</td><td>0.59/0.23 0.63/0.23</td><td>0.54/0.27 0.63/0.15</td><td>0.49/0.39 0.64/0.15</td><td>0.46/0.41</td><td>0.70/0.15</td><td>0.46/0.44 0.70/0.15</td><td>0.46/0.43</td></tr><tr><td>TrustRAG</td><td>0.41/0.55</td><td>0.43/0.52</td><td>0.47/0.43</td><td>0.30/0.61</td><td>0.33/0.64</td><td>0.65/0.14 0.39/0.50</td><td>0.54/0.41</td><td>0.61/0.35</td><td>0.64/0.23</td></tr><tr><td>CrAM</td><td>0.59/0.08</td><td>0.61/0.08</td><td>0.58/0.09</td><td>0.58/0.07</td><td>0.68/0.09</td><td>0.60/0.11</td><td>0.62/0.10</td><td>0.61/0.07</td><td>0.61/0.28</td></tr><tr><td>RAGSentinel</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.64/0.09</td></tr></table>

Table 12: Defense performance under mixed attack settings (PoisonedRAG+PIA, PoisonedRAG+AD, PIA+AD) across three datasets and three models $\left( k ^ { \prime } = 2 \right)$