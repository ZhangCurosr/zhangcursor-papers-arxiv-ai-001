# GIST: GIBBS-TARGETED HIERARCHICAL DATA SELEC-TION FOR OFFLINE RLVR

Xuliang Huang<sup>1</sup>, Tianxiang Zhao<sup>1</sup>, Chengwei Qin<sup>1,2</sup>, Yao Shu<sup>1,2∗</sup> <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup>The Hong Kong University of Science and Technology

## ABSTRACT

Offline reinforcement learning with verifiable rewards (RLVR) trains on a fixed pool of sampled responses and typically supervises every token of every response, although responses and tokens are not equally informative. We study which ones to supervise through the Gibbs-optimal policy of KL-regularized RLVR, which offline training targets, and measure the mismatch between a policy and this target with a bidirectional KL divergence. At the reference policy, this mismatch factorizes exactly and separates the two outcomes: correct responses contribute mainly through the forward term, concentrated on their low-probability tokens, and incorrect responses through the reverse term, concentrated on the responses the policy generates often and on their high-probability tokens. Based on this analysis, we propose GIST (Gibbs-Targeted Hierarchical Data Selection), a two-level selection method that approximates these principles with signals available before training. GIST uses the sampling temperature to select correct responses on-policy and incorrect responses near the policy’s mode, and it uses the initial policy’s token probabilities to supervise only the tokens where the mismatch lies. Applied to two offline training objectives, GIST raises the average accuracy over GSM8K, MATH-500, and MBPP for all three models under both objectives. In a controlled comparison on Qwen3-0.6B/GSM8K, it outperforms seven selection baselines that choose from the same candidates while supervising 66.6% fewer tokens than training on all candidates of the same prompts.

![](images/95baac9e9978afe10c4c78f1c7a3a59d97504bf643008494c52736b1db9b37e9.jpg)  
Figure 1: Test Strict accuracy versus supervised tokens on Qwen3-0.6B/GSM8K. All methods use BOLT on the same candidate pool. GIST reaches the highest accuracy while supervising 66.6% fewer tokens than training on all candidates and 31.5% fewer than random selection.

## 1 INTRODUCTION

Offline reinforcement learning with verifiable rewards (RLVR) trains a language model on a fixed pool of responses whose outcomes are checked automatically (Lambert et al., 2024; Shao et al., 2024; DeepSeek-AI, 2025; Wang et al., 2024b). Common approaches supervise every token of the responses they retain, although responses and tokens are not equally informative. Deciding which responses to keep and which of their tokens to supervise is therefore a central design choice in offline RLVR.

Existing methods address parts of this choice: rejection sampling keeps verified correct responses $( Z { \mathsf { e } } \cdot$ likman et al., 2022; Yuan et al., 2023), reward-weighted objectives such as BOLT weight responses by their reward (Peng et al., 2019; Shu et al., 2026), and other methods subsample rollouts (Li et al., 2026; Xu et al., 2025) or select tokens by entropy or probability (Wang et al., 2025a; Liu et al., 2026). They leave open how to combine response selection with outcome-dependent token supervision, although whether a token deserves supervision depends on both the outcome and how the response was sampled. A low-probability token in a high-temperature sample may be noise, but in a verified correct response it is more likely a valid step that the policy under-weights, while failures sampled at low temperature reveal errors near the policy’s mode.

We study selection through the Gibbs-optimal policy of KL-regularized RLVR, the target that offline training fits, and measure the mismatch between the policy and this target with a bidirectional KL divergence and a KL-curvature score that bounds it from above and below. The forward term governs correct responses, whose informative instances are under-represented and whose informative tokens have low probability. The reverse term governs incorrect responses, whose informative instances are over-represented and whose informative tokens have high probability. At the reference policy, where offline selection takes place, the mismatch factorizes exactly.

We then propose GIST (Gibbs-Targeted Hierarchical Data Selection), a two-level selection method that approximates these principles with three quantities available at data-construction time: the sampling temperature, the verifier outcome, and the initial policy’s token probabilities. At the response level, GIST takes correct responses from on-policy samples and incorrect responses from near-mode samples, and it keeps the prompts that supply both. At the token level, it supervises the low-probability tokens of correct responses and the high-probability tokens of incorrect ones.

Paired with two offline objectives, BOLT and SFT, GIST improves accuracy on GSM8K, MATH-500, and MBPP in nearly all settings (29 of 30) across three models from two families, and it raises the average accuracy in every case. On Qwen3-0.6B/GSM8K, where all methods choose from the same candidates, GIST outperforms seven selection baselines while supervising 66.6% fewer tokens and taking 49.9% fewer optimizer steps than training on all candidates of the same prompts (Figure 1).

Our contributions are as follows. (i) We derive selection principles for offline RLVR from the bidirectional mismatch to the Gibbs-optimal target, separating the roles of correct and incorrect responses at the response and token levels (Section 3). (ii) We propose GIST, a two-level selection method that approximates these principles from signals observed during data construction (Section 4). (iii) Across three models and two offline objectives, GIST improves nearly all benchmark results, and in a controlled comparison it outperforms seven selectors while supervising 66.6% fewer tokens (Section 5).

## 2 PRELIMINARIES

KL-regularized RLVR. For a prompt x and response $y ,$ a verifier assigns a binary reward $r ( x , y ) \in$ $\{ 0 , 1 \}$ , with $r = 1$ for correct and $r = 0$ for incorrect responses. Throughout, the reference policy $\pi _ { \mathrm { r e f } }$ is the initial policy $\pi _ { 0 }$ , and KL-regularized RLVR (Ouyang et al., 2022; Lambert et al., 2024) maximizes

$$
J _ { x } ( \pi ) = \mathbb { E } _ { y \sim \pi ( \cdot | x ) } { \big [ } r ( x , y ) { \big ] } - \beta \operatorname { K L } ( \pi ( \cdot \mid x ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid x ) { \big ) } ,\tag{1}
$$

where $\beta > 0$ controls the strength of regularization.

Proposition 2.1 (Gibbs target). For a fixed prompt x, the unique maximizer of Eq. (1) is

$$
\pi ^ { \star } ( y \mid x ) = \frac { \pi _ { \mathrm { r e f } } ( y \mid x ) e ^ { r ( x , y ) / \beta } } { Z ( x ) } , \qquad Z ( x ) = \sum _ { y ^ { \prime } } \pi _ { \mathrm { r e f } } ( y ^ { \prime } \mid x ) e ^ { r ( x , y ^ { \prime } ) / \beta } ,\tag{2}
$$

and for any policy π with finite KL divergence to $\pi _ { \mathrm { r e f } }$

$$
J _ { x } ( \pi ^ { \star } ) - J _ { x } ( \pi ) = \beta \operatorname { K L } \bigl ( \pi ( \cdot \mid x ) \parallel \pi ^ { \star } ( \cdot \mid x ) \bigr ) .\tag{3}
$$

The target reweights the reference policy by the exponentiated reward (Peters et al., 2010; Korbak et al., 2022; Rafailov et al., 2023), and Eq. (3) identifies the objective gap with the reverse $\operatorname { K L } \tan \pi ^ { \star }$

Offline fitting of the Gibbs target. Offline training fits $\pi ^ { \star }$ using a fixed pool of responses drawn from a rollout distribution $\mu ,$ with $\mu ( \boldsymbol { y } \mid \boldsymbol { x } ) > 0$ wherever $\pi ^ { \star } ( y \mid x ) > 0$ . Rewriting the forward KL as an expectation under µ gives

$$
\mathbb { E } _ { x } \big [ \mathrm { K L } ( \pi ^ { \star } | | \pi _ { \theta } ) \big ] = \mathbb { E } _ { x } \mathbb { E } _ { y \sim \mu } \big [ - \rho ( x , y ) \log \pi _ { \theta } ( y \mid x ) \big ] + C , \qquad \rho ( x , y ) = \frac { \pi _ { \mathrm { r e f } } ( y \mid x ) } { \mu ( y \mid x ) } \frac { e ^ { r ( x , y ) / \beta } } { Z ( x ) } ,\tag{4}
$$

where C does not depend on θ. When $\mu = \pi _ { \mathrm { r e f } }$ , the weight reduces to $e ^ { r ( x , y ) / \beta } / Z ( x )$ , and Eq. (4) becomes the reference-sampled weighted-likelihood objective of BOLT (Shu et al., 2026). This objective fits the target in the forward direction, whereas the objective gap in Eq. (3) is a reverse KL. Its weights also separate the two outcome classes: an incorrect response receives $e ^ { - 1 / \beta }$ times the weight of a correct response to the same prompt. Section 3 uses both directions to locate the mismatch, and Section 4.3 describes how the weights are computed in training.

Token-level view. Let $\boldsymbol { s } _ { t } = \left( x , y _ { < t } \right)$ denote the prefix at step $t ,$ and let $Z ( s )$ be the expectation of $e ^ { r / \beta }$ over reference continuations of s, so that $Z ( x )$ is the normalizer in Eq. (2) and $Z ( s ) = e ^ { V ( s ) / \beta }$ for the soft value V (Ziebart, 2010; Haarnoja et al., 2017; Rafailov et al., 2024). The Gibbs target then factorizes over tokens as

$$
\pi ^ { \star } ( a \mid s ) = \pi _ { \mathrm { r e f } } ( a \mid s ) \frac { Z ( s a ) } { Z ( s ) } ,\tag{5}
$$

so a token is more likely under the target than under the reference policy exactly when it raises the soft value. For two distributions p and q over responses with conditionals $p _ { t } = p ( \cdot \mid s _ { t } )$ and $q _ { t } = q ( \cdot \mid s _ { t } )$ , the chain rule for KL divergence (Cover & Thomas, 2006) gives

$$
\mathrm { K L } ( p \| q ) = \sum _ { t } \mathbb { E } _ { s _ { t } \sim p } \big [ \mathrm { K L } ( p _ { t } \| q _ { t } ) \big ] .\tag{6}
$$

Throughout, $\varphi ( u ) = u \log u - u + 1 \geq 0$ , so that $\begin{array} { r } { \mathrm { K L } ( p \| q ) = \sum _ { y } q _ { y } \varphi ( p _ { y } / q _ { y } ) } \end{array}$

## 3 A KL-CURVATURE VIEW OF DATA SELECTION

To decide which responses and tokens to supervise, we ask where the policy differs most from the Gibbs target of Section 2, first for whole responses and then for individual tokens (proofs in Appendix A).

## 3.1 BIDIRECTIONAL MISMATCH AND CURVATURE SCORE

Fix a prompt x, and write $p _ { y } = \pi ^ { \star } ( y \mid x )$ and $q _ { y } = \pi _ { \theta } ( y \mid x )$ . For $\lambda \geq 0$ , define

$$
D _ { \lambda } ( p , q ) = \mathrm { K L } ( p \| q ) + \lambda \mathrm { K L } ( q \| p ) , \qquad U _ { y } ^ { \lambda } = ( p _ { y } - q _ { y } ) ^ { 2 } \left( \frac { 1 } { q _ { y } } + \frac { \lambda } { p _ { y } } \right) .\tag{7}
$$

The forward term is the target-fitting loss of Section $2 ;$ it measures how much the policy underrepresents target behavior (Minka, 2005). The reverse term, scaled by $\beta ,$ is the RLVR objective gap (Proposition 2.1); it measures how much policy mass lies on responses that the target down-weights. The score $U _ { y } ^ { \lambda }$ weights the squared probability gap by the local curvature of both KL terms (Csiszar,´ 1967), so a gap of a given size counts more when it occurs at small probabilities.

Theorem 3.1 (KL-curvature bounds). Let p, q be positive on a common support with $\gamma \leq p _ { y } / q _ { y } \leq \Gamma$ where $0 < \gamma \leq 1 \leq \Gamma$ . Then $c _ { - } U ^ { \lambda } \leq D _ { \lambda } \leq c _ { + } U ^ { \lambda }$ , where $\begin{array} { r } { U ^ { \lambda } = \sum _ { y } \bar { U } _ { y } ^ { \lambda } , c _ { - } = \frac { 1 } { 2 } \operatorname* { m i n } \{ \Gamma ^ { - 1 } , \gamma \} } \end{array}$ and $\begin{array} { r } { c _ { + } = \frac { 1 } { 2 } \operatorname* { m a x } \{ \gamma ^ { - 1 } , \Gamma \} } \end{array}$ . As the density ratios converge uniformly to one, $\begin{array} { r } { D _ { \lambda } = \frac { 1 } { 2 } U ^ { \lambda } + o ( U ^ { \lambda } ) } \end{array}$ .

Up to constants, the score thus bounds the mismatch from above and below, and it is exact to second order near the target. The curvature factor also explains why a verifier label alone does not determine a response’s value: the same probability gap counts more where the policy or the target places little mass. The coefficient λ weights the two directions and is used only in the analysis.

## 3.2 BUDGETED RESPONSE SELECTION

To select under a budget, we attribute the mismatch to individual responses through the contribution $K _ { y } ^ { \lambda } = q _ { y } \varphi ( p _ { y } / q _ { y } ) + \bar { \lambda } p _ { y } \varphi ( q _ { y } / p _ { y } )$ , which is nonnegative and satisfies $\begin{array} { r } { D _ { \lambda } = \sum _ { y } ^ { \mathbf { \bar { \alpha } } } K _ { y } ^ { \lambda } } \end{array}$ (Section 2).

Corollary 3.2 (Budgeted selection guarantee). Let $S _ { U }$ contain the B candidates with the largest $U _ { y } ^ { \lambda }$ , and let $S ^ { \star }$ maximize $\textstyle \sum _ { y \in S } K _ { y } ^ { \lambda }$ over subsets of size B. Under the conditions of Theorem 3.1, $\begin{array} { r } { \sum _ { y \in S _ { U } } K _ { y } ^ { \lambda } \geq ( c _ { - } / c _ { + } ) \sum _ { y \in S ^ { \star } } K _ { y } ^ { \lambda } } \end{array}$

Selecting by the curvature score therefore captures a constant fraction of the largest attainable mismatch, similar to coreset guarantees (Sener & Savarese, 2018; Mirzasoleiman et al., 2020), and this fraction approaches one as the policy approaches the target. Note that the guarantee is about the mismatch captured at a fixed policy; it does not concern the outcome of training on the selected set.

With binary rewards, the target is $p _ { y } = \pi _ { \mathrm { r e f } } ( y \mid x ) e ^ { r ( x , y ) / \beta } / Z ( x )$ , and the score behaves differently on the two classes.

Corollary 3.3 (Informative correct and incorrect responses). For fixed $\lambda > 0 \colon ( i ) i f r = 1$ and $q _ { y } \ll p _ { y } ,$ , then $U _ { y } ^ { \lambda } \sim p _ { y } ^ { 2 } / q _ { y }$ and the forward term dominates; (ii) $i f r = 0$ and $q _ { y } \gg p _ { y } ,$ , then $U _ { y } ^ { \lambda } \sim \lambda q _ { y } ^ { 2 } / p _ { y }$ and the reverse term dominates. At comparable reference probabilities, (i) favors under-represented correct responses and (ii)favors over-represented incorrect responses.

Under-represented correct responses drive the forward (coverage) error, and over-represented incorrect ones drive the reverse error, which Proposition 2.1 identifies with the RLVR objective gap. The two directions describe the same mismatch: mass moved onto under-represented correct responses must come from other responses, and the reverse term measures how much of it sits on over-represented errors. Offline selection evaluates the mismatch at $q = \pi _ { 0 } = \pi _ { \mathrm { r e f } }$ , where both classes have an exact form.

Proposition 3.4 (Exact mismatch at the reference policy). $A t q = \pi _ { \mathrm { r e f } } , p _ { y } / q _ { y } = \kappa _ { r } ( x ) : = e ^ { r / \beta } / Z ( x )$ and $K _ { y } ^ { \lambda } = q _ { y } h _ { \lambda } ( \kappa _ { r } ( x ) )$ , where $h _ { \lambda } ( \kappa ) = \kappa$ log $\kappa - \kappa + 1 + \lambda { \left( \kappa - 1 - \log \kappa \right) }$ . With $P ( x )$ the reference success probability, $\kappa _ { 1 } \approx 1 / P ( x ) > 1$ and $\kappa _ { 0 } \approx e ^ { - 1 / \beta } / P ( x ) < 1$

Hence every correct response is under-represented and every incorrect response is over-represented, as in Corollary 3.3. Within a class, $K _ { y } ^ { \lambda } \propto q _ { y }$ , so frequently generated incorrect responses contribute most. For correct responses, the under-representation is uniform at the response level and localized at the token level (Section 3.3).

Remark 3.5 (The curvature score at the reference policy). At $q = \pi _ { \mathrm { r e f } } , \mathrm { E q . } ( 7 )$ gives

$$
U _ { y } ^ { \lambda } = q _ { y } ( \kappa _ { r } - 1 ) ^ { 2 } \Big ( 1 + \frac { \lambda } { \kappa _ { r } } \Big ) ,\tag{8}
$$

so $K _ { y } ^ { \lambda } / U _ { y } ^ { \lambda }$ is constant within a class and Corollary 3.2 is exact there; across classes, Proposition 3.4 gives the exact weights (Appendix A.3).

## 3.3 TOKEN-LEVEL MISMATCH

Applying the chain rule of Eq. (6) to both terms of $D _ { \lambda }$ decomposes the mismatch over prefixes, with the forward part weighted by $p ( s _ { t } )$ and the reverse part by $q ( s _ { t } )$ . For an observed token, we define its conditional score $\bar { U } _ { t } ^ { \lambda }$ and contribution $K _ { t } ^ { \lambda }$ as in $\mathrm { E q . } ( 7 )$ and Section 3.2, with $p _ { t } , q _ { t }$ evaluated at that token. Theorem 3.1 and Corollary 3.2 also hold for these conditional quantities at each prefix, and the two regimes again separate.

Corollary 3.6 (Informative tokens). For fixed $\lambda > 0$ and comparable target conditionals across positions: (i) in a correct response, low-probability tokens $( q _ { t } \ll p _ { t } )$ carry the forward mismatch, with $U _ { t } ^ { \lambda } \sim p _ { t } ^ { 2 } / q _ { t } ; ( i i )$ in an incorrect response, high-probability tokens $( q _ { t } \gg p _ { t } )$ carry the reverse mismatch, with $\bar { U } _ { t } ^ { \lambda } \sim \lambda q _ { t } ^ { 2 } / p _ { t }$

Proposition 3.7 (Conditional token-level factorization). $A t \pi _ { \theta } = \pi _ { \mathrm { r e f } }$ , with $q _ { t } = \pi _ { \mathrm { r e f } } ( y _ { t } \mid s _ { t } )$ and $\rho _ { t } = Z ( s _ { t } y _ { t } ) / Z ( s _ { t } )$ , we have $\begin{array} { r } { K _ { t } ^ { \lambda } = q _ { t } h _ { \lambda } ( \rho _ { t } ) , \prod _ { t } \rho _ { t } = \kappa _ { r ( x , y ) } ( x ) } \end{array}$ , and $\rho _ { t } \leq 1 / q _ { t }$

By Eq. $( 5 ) , \rho _ { t }$ is the token’s soft-value ratio; GIST ranks positions by these conditional quantities and does not estimate the prefix weights (Appendix A). The bound $\rho _ { t } \leq 1 / q _ { t }$ supports Corollary 3.6(i) at the reference policy even without the comparability condition: a large soft-value gain requires a low $q _ { t } ,$ , although a low probability alone does not imply a large gain. For an incorrect response, log $Z$ decreases by $\log ( 1 / \kappa _ { 0 } ) \approx 1 / \overset { \cdot } { \beta } + \log P ( x )$ in total, and for a given decrease, $K _ { t } ^ { \lambda }$ grows with $q _ { t }$ , as in (ii). Because $h _ { \lambda }$ is increasing on $\lbrack 1 , \infty )$ , the largest conditional contribution attainable at a position with $\rho _ { t } \geq 1 \ \mathrm { i s }$

Table 1: Selection principles implied by the bidirectional mismatch.
<table><tr><td>Class</td><td>Dominant term</td><td>Response level</td><td>Token level</td></tr><tr><td>Correct</td><td>forward (coverage)</td><td>under-represented</td><td>low-probability tokens</td></tr><tr><td>Incorrect</td><td>reverse (over-commitment)</td><td>over-represented</td><td>high-probability tokens</td></tr></table>

![](images/23bae701e0bc2213a88cadbc3be324688ac772f7526df2e920ea83d87f79f4bd.jpg)  
Figure 2: Overview of GIST (schematic). (a) Rollouts on-policy $( \tau _ { h } = 1 )$ and near the mode (τ ). (b) Correct responses are taken from $\tau _ { h }$ and incorrect ones from $\tau _ { l } .$ (c) The frozen initial policy scores tokens to mask both classes. (d) Selected responses and masks are passed to an offline training objective.

$$
\begin{array} { r } { q _ { t } h _ { \lambda } ( 1 / q _ { t } ) = \log \frac { 1 } { q _ { t } } - 1 + q _ { t } + \lambda \big ( 1 - q _ { t } - q _ { t } \log \frac { 1 } { q _ { t } } \big ) , } \end{array}\tag{9}
$$

which decreases in $q _ { t }$ . The high-probability tokens of a correct response therefore account for little of its mismatch.

Remark 3.8 (Where $\beta$ enters selection). The within-class ranking by $q _ { y } ,$ , the bound $\rho _ { t } \leq 1 / q _ { t }$ , and the signs of the soft-value totals hold for every $\beta > 0 .$ , so the four principles in Table 1 do not depend on $\beta .$ Only the balance between the two classes depends on $\beta$ and $\lambda ,$ through the ratio of their total contributions,

$$
R _ { \lambda } ( x ) = \frac { \left( 1 - P ( x ) \right) h _ { \lambda } ( \kappa _ { 0 } ) } { P ( x ) h _ { \lambda } ( \kappa _ { 1 } ) } .\tag{10}
$$

At $P ( x ) = 0 . 5$ and $\beta = 0 . 1 , R _ { \lambda }$ is 2.6 for $\lambda = 0$ and 13.4 for $\lambda = 1 ;$ at $\beta = 0 . 5 .$ , it is 1.8 and 2.5. Since the analysis does not fix the class balance, GIST sets it through the number of retained responses in each class.

Table 1 summarizes the four principles, each of which follows from the curvature of the two KL terms.

## 4 GIST: GIBBS-TARGETED HIERARCHICAL DATA SELECTION

The principles in Table 1 depend on target probabilities and soft values, which are not available during data construction. Instead of estimating these quantities, GIST uses three observable signals: the sampling temperature, the verifier outcome, and the initial-policy token probability. With these signals, it selects responses and then tokens within them, and it keeps the prompts that supply both outcomes (Figure 2). All selections are made once, before training.

## 4.1 PROMPT AND RESPONSE SELECTION

For each prompt $x ,$ the initial policy $\pi _ { 0 } = \pi _ { \mathrm { r e f } }$ generates $N _ { h } = 1 6$ responses at temperature $\tau _ { h } = 1 . 0$ and $N _ { l } = 1 6 \mathrm { a t } \tau _ { l } = 0 . 3$ , without top-p (Holtzman et al., 2020) or top-k truncation. A verifier assigns

$r ( x , y ) = 1$ to correct and $r ( x , y ) = 0$ to incorrect responses. GIST takes correct responses from the on-policy pool and incorrect responses from the near-mode pool; we call this assignment temperature pairing:

$$
\mathcal { Y } _ { + } ( x ) = \{ y \in \mathcal { Y } _ { h } ( x ) : r ( x , y ) = 1 \} , \qquad \mathcal { Y } _ { - } ( x ) = \{ y \in \mathcal { Y } _ { l } ( x ) : r ( x , y ) = 0 \} .\tag{11}
$$

Within each class, the contribution is proportional to $q _ { y }$ (Proposition 3.4), so sampling at τ concentrates on the incorrect responses that contribute most. For correct responses, we choose $\tau _ { h } = 1$ with the token level in mind: on-policy responses keep the low-probability steps that the mask selects (Section 5.4), and they need no temperature correction in Eq. (4). The subsequent selection and masking, however, make the training loss an approximation to that objective. A prompt is $e l i g i b l e$ if $| \mathcal { D } _ { + } ( \overline { { x } } ) | \geq B _ { + }$ and $| \mathcal { V } _ { - } ( x ) | \geq B _ { - }$ . The quotas are a practical choice and do not follow from the analysis. They guarantee samples of both classes and bound the weights of correct responses; the cost is that the hardest prompts are excluded. For each eligible prompt, GIST samples $B _ { + } = 4$ correct and $B _ { - } = 4$ incorrect responses uniformly without replacement, treating duplicates as distinct instances. The analysis does not fix the balance between the classes (Remark 3.8), and we use equal quotas.

## 4.2 TOKEN SELECTION

For a retained response y, let $q _ { t } = \pi _ { 0 } ( y _ { t } \mid x , y _ { < t } )$ and let q¯ be its mean over the response positions, excluding EOS. The masks are

$$
m _ { t } ^ { + } = { \bf 1 } \left[ q _ { t } < \bar { q } \right] \mathrm { f o r c o r r e c t r e s p o n s e s } ,
$$

$$
m _ { t } ^ { - } = \mathbf { 1 } \left[ q _ { t } > \bar { q } \right] \mathrm { f o r i n c o r r e c t r e s p o n s e s } .\tag{12}
$$

These masks implement the token-level principles, with the response mean as the threshold between low and high probability. Correct responses keep their low-probability tokens, where the underrepresented steps lie (Corollary 3.6(i), Proposition 3.7), and incorrect responses keep their highprobability tokens, the confident errors of Corollary 3.6(ii). Both masks are applied in every run that uses GIST’s token selection. They depend only on the verifier outcome and the initial policy, not on the training objective, and are computed once with the frozen initial policy and stay fixed during training.

## 4.3 TRAINING WITH THE SELECTED DATA

GIST outputs the selected responses, their rewards, and the token masks, so it can be combined with any offline objective that trains on weighted responses; we use two. BOLT (Shu et al., 2026), the weighted-likelihood objective of Section 2, weights each response by $e ^ { r / \beta }$ , rescaled so that the weights of the prompt’s 32 candidates average to one. The weights are computed on the full candidate pool before selection and are not renormalized afterward. A correct response therefore receives a weight of about $3 2 / n _ { 1 }$ , where $n _ { 1 }$ is the number of correct candidates, so prompts that the policy solves less often count more, and an incorrect response receives $e ^ { - 1 / \beta }$ times that weight. SFT instead gives every retained response unit weight. Both objectives average, over the minibatch, each response’s weight times its mean negative log-likelihood on the masked positions; unmasked positions remain in the context but receive no loss.

## 5 EXPERIMENTS

## 5.1 MAIN RESULTS

Setup. We train Qwen3-0.6B, Qwen3.5-2B, and Gemma-4-E2B-it (Yang et al., 2025; Qwen Team, 2026; Gemma Team, 2026) and evaluate on GSM8K, MATH-500, and MBPP-full (Cobbe et al., 2021; Hendrycks et al., 2021; Lightman et al., 2024; Austin et al., 2021). For math, we report Strict accuracy, which requires the prescribed answer format, and Flexible accuracy (Flex.), which also credits correctly extracted answers; for code, we report pass@1. We pair GIST with two objectives: BOLT $( \beta = 0 . 1 )$ and SFT, which imitates every training response with unit weight. All trained methods share 32 candidates per prompt (16 per temperature) and validation-based checkpoint selection; each objective is trained either on all candidates or on GIST’s selected subset (+ GIST; details in Appendix B).

Table 2: Test results with BOLT and SFT (%). Avg.: mean of GSM8K Flex., MATH-500 Flex., and MBPP-full pass@1. Bold marks the best score within each model and column. Parentheses: change vs. Initial.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">GSM8K↑</td><td colspan="2">MATH-500 ↑</td><td>MBPP-full ↑</td><td></td></tr><tr><td>Strict</td><td>Flex.</td><td>Strict</td><td>Flex.</td><td>pass@1</td><td>Avg. ↑</td></tr><tr><td>Qwen3-0.6B</td><td>Initial</td><td>40.86</td><td>46.10</td><td>44.20</td><td>49.80</td><td>22.80</td><td>39.57</td></tr><tr><td></td><td>SFT</td><td> $4 2 . 0 8 ( + 1 . 2 2 )$ </td><td> $4 7 . 3 8 ( + 1 . 2 8 )$ </td><td> $4 4 . 4 0 \AA ( + 0 . 2 0 )$ </td><td> $5 0 . 0 0 ( + 0 . 2 0 ) $ </td><td> $2 4 . 2 0 ( + 1 . 4 0 )$ </td><td>40.53 (+0.96)</td></tr><tr><td></td><td></td><td>+ GIST 43.82 (+2.96)</td><td> $4 8 . 1 4 \ : ( + 2 . 0 4 )$ </td><td> $4 4 . 8 0 ( + 0 . 6 0 )$ </td><td> $5 0 . 4 0 ( + 0 . 6 0 )$ </td><td> $2 8 . 8 0 ( + 6 . 0 0 )$ </td><td>42.45 (+2.88)</td></tr><tr><td></td><td>BOLT</td><td> $4 3 . 2 9 \left( + 2 . 4 3 \right)$ </td><td> $4 7 . 4 6 \ : ( + 1 . 3 6 )$ </td><td> $4 4 . 6 0 ( + 0 . 4 0 )$ </td><td> $5 0 . 4 0 ( + 0 . 6 0 )$ </td><td> $2 6 . 6 0 ( + 3 . 8 0 )$ </td><td>41.49 (+1.92)</td></tr><tr><td></td><td>+ GIST</td><td>45.79 (+4.93)</td><td> ${ \pmb 4 9 . 7 3 } _ { ( + 3 . 6 3 ) }$ </td><td>45.40 (+1.20)</td><td>51.00 (+1.20)</td><td>31.40 (+8.60)</td><td>44.04(+4.47)</td></tr><tr><td>Qwen3.5-2B</td><td>Initial</td><td>54.51</td><td>55.19</td><td>46.00</td><td>55.00</td><td>37.40</td><td>49.20</td></tr><tr><td></td><td>SFT</td><td>55.04 (+0.53)</td><td>55.80 (+0.61)</td><td> $4 6 . 2 0 ( + 0 . 2 0 )$ </td><td>55.00 (+0.00)</td><td> $3 9 . 6 0 ( + 2 . 2 0 )$ </td><td>50.13 (+0.93)</td></tr><tr><td></td><td>+ GIST</td><td>56.18 (+1.67)</td><td>56.71 (+1.52)</td><td> $4 6 . 8 0 ( + 0 . 8 0 )$ </td><td>55.80 (+0.80)</td><td>41  $. 2 0 ( + 3 . 8 0 )$ </td><td>51.24 (+2.04)</td></tr><tr><td></td><td>BOLT</td><td>54.13 (−0.38)</td><td>54.51 (−0.68)</td><td> $4 6 . 4 0 ( + 0 . 4 0 )$ </td><td> $5 4 . 4 0 ( - 0 . 6 0 )$ </td><td> $\pm 2 . 4 \mathbf { 0 } \left( + 5 . 0 0 \right)$ </td><td>50.44 (+1.24)</td></tr><tr><td></td><td>+ GIST</td><td>56.63 (+2.12)</td><td>57.09 (+1.90)</td><td> ${ \pm 7 . 4 0 } \AA ( + 1 . 4 0 )$ </td><td> ${ \pmb 5 6 . 2 0 } _ { ( + 1 . 2 0 ) }$ </td><td> $4 1 . 8 0 ( + 4 . 4 0 )$ </td><td>51.70 (+2.50)</td></tr><tr><td>Gemma-4-</td><td>Initial</td><td>12.28</td><td>26.99</td><td>2.60</td><td>38.60</td><td>55.00</td><td>40.20</td></tr><tr><td>E2B-it</td><td>SFT</td><td>12.81 (+0.53)</td><td>27.82 (+0.83)</td><td>12.40 (+9.80)</td><td>42.00 (+3.40)</td><td>55.80(+0.80)</td><td>41.87(+1.67)</td></tr><tr><td></td><td>+ GIST</td><td>13.87 (+1.59)</td><td> $2 9 . 0 4 \ : ( + 2 . 0 5 )$ </td><td>41.80 (+39.20)</td><td> $5 3 . 0 0 ( + 1 4 . 4 0 ) $ </td><td> $5 6 . 8 0 ( + 1 . 8 0 )$ </td><td>46.28 (+6.08)</td></tr><tr><td></td><td>BOLT</td><td> $1 2 . 8 9 ( + 0 . 6 1 )$ </td><td>26.31 (−0.68)</td><td>40.80 (+38.20)</td><td> $5 2 . 8 0 ( + 1 4 . 2 0 ) $ </td><td> $5 6 . 4 0 ( + 1 . 4 0 )$ </td><td> $4 5 . 1 7 \ : ( + 4 . 9 7 )$ </td></tr><tr><td></td><td>+ GIST</td><td> $\mathbf { 1 4 . 1 0 } \left( + 1 . 8 2 \right)$ </td><td> ${ \pmb 2 9 . 5 7 } _ { ( + 2 . 5 8 ) }$ </td><td> $4 2 . 2 \mathbf { 0 } \left( + 3 9 . 6 0 \right)$ </td><td> ${ \pm 3 . 2 0 } \left( { + 1 4 . 6 0 } \right)$ </td><td> ${ \pmb 5 7 . \pmb 0 0 } \left( + 2 . 0 0 \right)$ </td><td> $\mathbf { 4 6 . 5 9 } _ { ( + 6 . 3 9 ) }$ </td></tr></table>

Table 3: Selection methods with BOLT on a shared candidate pool (Qwen3-0.6B/GSM8K, %).
<table><tr><td>Random PODS RFT-style RSI-S</td><td></td><td></td><td></td><td>ETS ProFit HES</td></tr><tr><td>Strict ↑ Flex. ↑ 48.60</td><td>43.14 43.52</td><td>44.05 48.22 48.14</td><td></td><td>GIST 43.2143.7542.4644.2045.79 47.8447.5446.4747.3149.73</td></tr></table>

Results. Averaged over the three benchmarks (Avg. in Table 2), GIST improves both objectives on every model, by 1.26–2.55 points over BOLT and 1.11–4.41 points over SFT, and BOLT + GIST has the highest average for all three models. GIST improves over BOLT on both mathematical benchmarks for all three models under both metrics (Table 2). On GSM8K, the gains reach 2.50 points on Strict and 3.26 on Flexible. On Flexible, BOLT falls below the initial checkpoint for two models, whereas BOLT + GIST improves on it. For Qwen3-0.6B, paired tests over the 1,319 GSM8K questions support both gains (Appendix B.5). BOLT also trains on prompts whose candidates are all incorrect (Appendix B), so this compares the two pipelines end to end; Sections 5.2 and 5.3 isolate the selection components. Gains on MATH-500 are smaller (0.40–1.80 points). On MBPP-full, GIST gains 4.80 points on Qwen3-0.6B and 0.60 on Gemma-4-E2B-it, and trails BOLT by 0.60 on Qwen3.5-2B.

GIST also improves over SFT for every model, benchmark, and metric (Table 2). The gains are 0.76–1.74 points on GSM8K, 0.40–0.80 on MATH-500 for the two Qwen models, and 1.00–4.60 on MBPP-full. On Gemma-4-E2B-it, MATH-500 Strict rises from 12.40% to 41.80% while Flexible rises by 11.00 points, so most of this gain comes from following the prescribed answer format. On GSM8K, the gains from GIST are similar under Flexible and Strict accuracy for both objectives, which suggests that they do not come only from answer formatting.

## 5.2 COMPARISON WITH SELECTION BASELINES

All selectors share the candidate pool, the 1,593 eligible prompts, and the BOLT objective (Table 3). Random samples four correct and four incorrect responses uniformly and supervises all tokens, RFT-style keeps all correct responses, and the others adapt HES, PODS, ProFit, ETS, and RSI-S (Li et al., 2026; Xu et al., 2025; Liu et al., 2026; Wang et al., 2025a; Lv et al., 2026b) (Appendix C).

GIST achieves the highest accuracy on both metrics, 1.59–3.33 points above all seven baselines on Strict and 1.13–3.26 points on Flexible. It exceeds Random by 2.65 points on Strict and 1.13 points on Flexible while supervising 31.5% fewer tokens (Section 5.4), and it exceeds training on all candidates by 1.97 and 3.03 points. Appendix B.5 reports paired significance tests for these comparisons.

Table 4: Component ablation, Qwen3-0.6B on GSM8K (%). Sub.: 4+4 responses instead of all 32; Pair.: correct from $\tau _ { h }$ , incorrect from τ ; Mask: $m ^ { + }$ of Eq. (12); Wt.: Boltzmann vs. unit. Parentheses: change vs. (f).
<table><tr><td>Sub. Pair. Mask Wt.</td><td></td><td></td><td></td><td></td><td>Strict ↑</td><td>Flex. ↑</td></tr><tr><td>(a)</td><td>×</td><td>×</td><td>×</td><td>√</td><td> $4 3 . 8 2 ( - 1 . 9 7 )$ </td><td> $4 6 . 7 0 ( - 3 . 0 3 )$ </td></tr><tr><td>(b)</td><td>√</td><td>X</td><td>×</td><td>√</td><td> $4 3 . 1 4 \ : ( - 2 . 6 5 )$ </td><td>48.60(−1.13)</td></tr><tr><td>(c)</td><td>√</td><td>X</td><td>√</td><td>√</td><td> $4 3 . 5 9 ( - 2 . 2 0 ) $ </td><td>48.75 (−0.98)</td></tr><tr><td>(d)</td><td>√</td><td>√</td><td>×</td><td>√</td><td> $4 3 . 7 5 \ : ( - 2 . 0 4 )$ </td><td> $4 7 . 6 1 ( - 2 . 1 2 )$ </td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>√</td><td>×</td><td> $4 3 . 8 2 ( - 1 . 9 7 )$ </td><td> $4 8 . 1 4 ( - 1 . 5 9 )$ </td></tr><tr><td>(f)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>45.79</td><td>49.73</td></tr></table>

![](images/d527ad88310a42f59ae5bc1b429c21cd565c26ecadf2d796db5b3bef5460c5ec.jpg)

<table><tr><td>Positions</td><td>Kept</td><td>Strict ↑ Flex. ↑</td></tr><tr><td>All tokens</td><td></td><td>10043.75 (−2.04) 47.61 (−2.12)</td></tr><tr><td>Random†</td><td></td><td>31 44.81 (−0.98) 45.64(−4.09)</td></tr><tr><td>Highest-prob.†</td><td></td><td>31 44.05 (−1.74)  $4 7 . 2 3 ( - 2 . 5 0 )$ </td></tr><tr><td>Below-mean (GIST)</td><td> 31</td><td>45.79 49.73</td></tr></table>

Table 5: Token rule for correct responses, Qwen3-0.6B on GSM8K (%). Kept: approximate share of positions supervised per correct response (%). <sup>†</sup>Same number of positions per response as GIST. Parentheses: change vs. GIST.

Figure 3: Pairing and masking on Qwen3- 0.6B/GSM8K. Unpaired responses are the Random selector; the mask helps mainly with paired responses.  
Table 6: Supervised tokens up to each method’s selected checkpoint (Qwen3-0.6B/GSM8K). Reduction is relative to all candidates.
<table><tr><td>All</td><td colspan="2">Random GIST</td></tr><tr><td>Tokens</td><td>13.60M</td><td>6.64M 4.55M</td></tr><tr><td>Reduction</td><td></td><td>51.2% 66.6%</td></tr><tr><td>Strict ↑</td><td>43.82 43.14</td><td>45.79</td></tr><tr><td>Flex. ↑</td><td>46.70 48.60</td><td>49.73</td></tr></table>

## 5.3 ABLATION STUDY

We ablate GIST on the shared pool, changing one design choice at a time (Tables 4 and 5).

Removing pairing (row c), the mask (row d), or the Boltzmann weights (row e, the SFT + GIST entry of Table 2) from GIST lowers Strict/Flexible accuracy by 2.20/0.98, 2.04/2.12, and 1.97/1.59 points, respectively, and removing all selection (row a) costs 1.97/3.03 (Table 4). Replacing GIST’s token rule also lowers both metrics (Table 5): supervising all tokens costs 2.04 and 2.12 points, a random subset of the same size costs 0.98 and 4.09, and keeping the highest-probability tokens of correct responses costs 1.74 and 2.50.

Rows (b), (c), (d), and (f) of Table 4 form a $2 \times 2$ design of pairing and masking (Figure 3). The mask adds 2.04 and 2.12 points with pairing but only 0.45 and 0.15 without it, an interaction of 1.59 and 1.97 points.

## 5.4 DISCUSSION

Token and response selection. The analysis in Section 3 holds at the policy that selects the data, so we compare its consequences with ablations that change one part of the data at a time. The token-rule ablation keeps GIST’s responses fixed and changes only which positions are supervised, which tests Corollary 3.6(i). By Proposition 3.7, a token can gain much soft value only if $q _ { t }$ is low $( \rho _ { t } \leq 1 / q _ { t } )$ , so the high-probability tokens of a correct response should carry little of its mismatch. Keeping them instead of the low-probability ones loses most of the mask’s benefit (Table 5), and the ProFit adaptation, which also keeps high-probability tokens, is the weakest selector (Table 3). Below-mean tokens make up about 31% of correct responses at both temperatures, but their mean q<sub>t</sub> is 0.41 at $\tau _ { h }$ and 0.58 at $\tau _ { l }$ (Appendix B.5); since the attainable contribution in Eq. (9) decreases in $q _ { t } .$ , the mask should find more informative tokens in on-policy responses. Without the mask, a whole correct response contributes in proportion to $q _ { y }$ (Proposition 3.4), which favors near-mode responses. Together, these results predict that pairing and masking help mainly in combination, which matches the interaction in Section 5.3.

Supervision and training cost. On the shared pool, GIST keeps 12,744 of the $^ { 5 0 , 9 7 6 }$ candidates and supervises 66.6% fewer tokens than training on all candidates and 31.5% fewer than Random, while scoring higher than both (Table 6, Figure 1, Appendix B.6). Masked positions still enter the forward and backward passes, so the compute saving over training on all candidates comes from keeping a quarter of the responses. GIST reaches its selected checkpoint in 2,394 optimizer steps, about half of the 4,779 needed when training on all candidates, although its selected checkpoint comes after twice as many epochs. Selection reuses BOLT’s candidate pool and adds one forward pass of the frozen initial policy to compute q<sub>t</sub>.

Generality. GIST selects data using only the sampling temperature, the verifier outcome, and the initial policy’s token probabilities, with the same settings for every model and benchmark (Appendix B). Because the selection does not depend on the training objective, it can be paired with other offline methods. SFT imitates every retained response with unit weight, including the incorrect ones, and SFT + GIST still matches or exceeds BOLT trained on all candidates in 14 of the 15 comparisons. The benefit of the selection therefore does not come only from BOLT’s reward weighting, although with GIST’s data the weights add a further 0.20–2.60 points in every comparison.

## 6 RELATED WORK

Offline RLVR and policy optimization. Reinforcement learning with verifiable rewards (RLVR) improves reasoning using automatically checkable outcomes (Lambert et al., 2024; Shao et al., 2024; DeepSeek-AI, 2025; Yu et al., 2025). Most RLVR methods optimize online with PPO-style updates (Schulman et al., 2017; Liu et al., 2025; Yue et al., 2025), whereas offline approaches (Levine et al., 2020) such as ILQL (Snell et al., 2023) and OREO (Wang et al., 2024b) learn from previously collected rollouts. BOLT (Shu et al., 2026) shows that Boltzmann-weighted likelihood, in the spirit of reward-weighted regression (Peters & Schaal, 2007; Peng et al., 2019), on reference-policy samples fits the Gibbs-optimal target of KL-regularized RLVR. These methods define the training objective; GIST instead decides which responses and tokens of a finite rollout pool to train on, and can be paired with BOLT or other offline objectives. Work on alignment and distillation chooses which KL direction or f-divergence the model should minimize (Go et al., 2023; Wang et al., 2024a; Gu et al., 2024), whereas we use both directions to select the training data.

Data selection for reasoning post-training. Selection for post-training (Zhou et al., 2023; Xia et al., 2024) has been studied at two granularities. At the response level, rejection-sampling fine-tuning and self-training (Zelikman et al., 2022; Yuan et al., 2023; Dong et al., 2023; Gulcehre et al., 2023; Singh et al., 2024) keep verified correct trajectories, HES (Li et al., 2026) ranks trajectories by the entropy of a few high-entropy tokens, PODS (Xu et al., 2025) subsamples rollouts for policy updates, and dynamic sampling in DAPO (Yu et al., 2025) discards prompts whose rollouts are all correct or all incorrect; other methods filter RL prompts by difficulty or learning impact (Bae et al., 2025; Li et al., 2025; Wang et al., 2025b). At the token level, Rho-1 (Lin et al., 2024a) and Token Cleaning (Pang et al., 2025) select tokens by excess loss or update influence, and critical-token estimation (Lin et al., 2024b) contrasts models trained on correct and incorrect trajectories. In RLVR, policy entropy governs exploration (Cui et al., 2025) and high-entropy forking tokens concentrate effective updates, which motivates entropy-based token selection (ETS) (Wang et al., 2025a); RSI-S (Lv et al., 2026b) and GMTS (Lv et al., 2026a) use surprisal- and gradient-based token scores, and ProFit (Liu et al., 2026) keeps high-probability tokens during SFT. Each of these methods selects at a single level with its own criterion. GIST derives both from the same bidirectional mismatch and applies them hierarchically, keeping, unlike ProFit, the low-probability tokens of correct responses.

## 7 CONCLUSION

We studied data selection for offline RLVR through the bidirectional mismatch between the policy and the Gibbs-optimal target. In the regimes we identify, the forward term governs correct responses and the reverse term governs incorrect ones. GIST turns this view into response- and token-level selection rules that use only the sampling temperature, the verifier outcome, and the initial policy’s token probabilities. Across three models and two offline objectives, it improves nearly all benchmark results, and it outperforms seven selectors on a shared candidate pool while supervising far fewer tokens than training on all candidates. More broadly, the analysis suggests that which responses and which tokens to train on can be read from the policy itself, and we expect this view to apply to other offline objectives and to settings in which the data are re-selected as the policy changes.

## ETHICS STATEMENT

This work uses publicly available models and benchmarks and involves no human subjects or personal data. We do not foresee ethical concerns beyond those of the underlying models.

## REPRODUCIBILITY STATEMENT

Proofs of all theoretical results are given in Appendix A. Appendix B specifies the data construction and eligibility rule (Appendix B.1), the token masks (Appendix B.2), the training objectives and hyperparameters (Appendix B.3), the evaluation protocol (Appendix B.4), and the paired tests (Appendix B.5); Appendix C describes every selection baseline and ablation control. All models and benchmarks are publicly available.

## AI USE STATEMENT

In this work, we used generative AI tools to edit the paper for readability, to polish the formatting of tables and figures, and to check that the numbers reported in the text match the tables. We have not used generative AI tools to develop the theoretical framework, formulate or prove the mathematica claims, design or implement the method and experiments, generate or process data, or interpret the results. We have reviewed all AI-assisted edits, checked every reported number against our experimental logs, and take full responsibility for the content of the paper.

## REFERENCES

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

Sanghwan Bae et al. Online difficulty filtering for reasoning oriented reinforcement learning, 2025. URL https://arxiv.org/abs/2504.03380.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/ abs/2110.14168.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley, second edition, 2006.

Imre Csiszar. Information-type measures of difference of probability distributions and indirect´ observations. Studia Scientiarum Mathematicarum Hungarica, 2:299–318, 1967.

Ganqu Cui et al. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https://arxiv.org/abs/2505.22617.

DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023. URL https: //arxiv.org/abs/2304.06767.

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. The hitchhiker’s guide to testing statistical significance in natural language processing. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018. URL https://aclanthology. org/P18-1128/.

Bradley Efron and Robert J. Tibshirani. An Introduction to the Bootstrap. Chapman & Hall, 1993.

Leo Gao et al. A framework for few-shot language model evaluation. https://github.com/ EleutherAI/lm-evaluation-harness, 2024.

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607. 02770.

Dongyoung Go, Tomasz Korbak, German Kruszewski, Jos Rozen, Nahyeon Ryu, and Marc Dymet-´ man. Aligning language models with preferences through f-divergence minimization. In International Conference on Machine Learning, 2023. URL https://arxiv.org/abs/2302. 08215.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024. URL https: //arxiv.org/abs/2306.08543.

Caglar Gulcehre et al. Reinforced self-training (ReST) for language modeling, 2023. URL https: //arxiv.org/abs/2308.08998.

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning, 2017. URL https://arxiv.org/abs/1702.08165.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2021. URL https://arxiv.org/abs/2103.03874.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https: //arxiv.org/abs/1904.09751.

Tomasz Korbak, Ethan Perez, and Christopher L. Buckley. RL with KL penalties is better viewed as Bayesian inference. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, 2022. URL https://arxiv.org/abs/2205.11275.

Nathan Lambert et al. Tulu 3: Pushing frontiers in open language model post-training, 2024. URL¨ https://arxiv.org/abs/2411.15124.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems, 2020. URL https://arxiv.org/abs/2005. 01643.

Xiaoyuan Li, Yubo Ma, Chengpeng Li, Fengbin Zhu, Yiyao Yu, Keqin Bao, Wenjie Wang, Fuli Feng, and Dayiheng Liu. Unified data selection for LLM reasoning, 2026. URL https: //arxiv.org/abs/2605.22389.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. LIMR: Less is more for RL scaling, 2025. URL https://arxiv.org/abs/2502.11886.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2305. 20050.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. Rho-1: Not all tokens are what you need, 2024a. URL https://arxiv.org/abs/2404.07965.

Zicheng Lin et al. Critical tokens matter: Token-level contrastive estimation enhances LLM’s reasoning capability, 2024b. URL https://arxiv.org/abs/2411.19943.

Tao Liu, Taiqiang Wu, Runming Yang, Shaoning Sun, Junjie Wang, and Yujiu Yang. ProFit: Leveraging high-value signals in SFT via probability-guided token selection. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 15383–15401, San Diego, California, United States, July 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026. findings-acl.755. URL https://aclanthology.org/2026.findings-acl.755/.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective, 2025. URL https: //arxiv.org/abs/2503.20783.

Outongyi Lv, Yuanwei Zhang, and Xiaoqun Zhang. GMTS: Gradient magnitude-based token selection improves RLVR training for LLM reasoning, 2026a. URL https://arxiv.org/abs/2608. 30632.

Outongyi Lv, Yanzhao Zheng, Yuanwei Zhang, Zhenghao Huang, Xingjun Wang, Baohua Dong, Hangcheng Zhu, and Yingda Chen. Which tokens matter? Adaptive token selection for RLVR with the relative surprisal index, 2026b. URL https://arxiv.org/abs/2606.31575.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

Tom Minka. Divergence measures and message passing. Technical Report MSR-TR-2005-173, Microsoft Research, 2005.

Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning, 2020. URL https: //arxiv.org/abs/1906.01827.

Long Ouyang et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/ 2203.02155.

Jinlong Pang, Na Di, Zhaowei Zhu, Jiaheng Wei, Hao Cheng, Chen Qian, and Yang Liu. Token cleaning: Fine-grained data selection for LLM supervised fine-tuning. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 47837–47858. PMLR, 2025. URL https://proceedings.mlr.press/ v267/pang25a.html.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning, 2019. URL https://arxiv.org/ abs/1910.00177.

Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In International Conference on Machine Learning, 2007.

Jan Peters, Katharina Mulling, and Yasemin Altun. Relative entropy policy search. In ¨ Proceedings of the AAAI Conference on Artificial Intelligence, 2010.

Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id= qwen3.5, 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741, 2023. URL https://arxiv.org/abs/2305.18290.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to q<sup>∗</sup>: Your language model is secretly a Q-function. In Conference on Language Modeling, 2024. URL https://arxiv. org/abs/2404.12358.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018. URL https:// arxiv.org/abs/1708.00489.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.

Yao Shu, Chenxing Wei, Hongbin Lin, Shuang Qiu, and Hui Xiong. Reference-sampled Boltzmann projection for KL-Regularized RLVR: Target-matched weighted SFT, finite one-shot gaps, and policy mirror descent, 2026. URL https://arxiv.org/abs/2605.02469.

Avi Singh et al. Beyond human data: Scaling self-training for problem-solving with language models. Transactions on Machine Learning Research, 2024. URL https://arxiv.org/abs/2312. 06585.

Charlie Victor Snell, Ilya Kostrikov, Yi Su, Sherry Yang, and Sergey Levine. Offline RL for natural language generation with implicit language Q learning. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2206.11871.

Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, and Yuxin Chen. Beyond reverse KL: Generalizing direct preference optimization with diverse divergence constraints. In International Conference on Learning Representations, 2024a. URL https://arxiv.org/abs/2309.16240.

Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, and Yi Wu. Offline reinforcement learning for LLM multi-step reasoning, 2024b. URL https://arxiv.org/ abs/2412.16145.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In Advances in Neural Information Processing Systems, 2025a. URL https://arxiv.org/abs/2506.01939.

Yiping Wang et al. Reinforcement learning for reasoning in large language models with one training example, 2025b. URL https://arxiv.org/abs/2504.20571.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for targeted instruction tuning. In International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/2402.04333.

Yixuan Even Xu, Yash Savani, Fei Fang, and J. Zico Kolter. Not all rollouts are useful: Downsampling rollouts in LLM reinforcement learning, 2025. URL https://arxiv.org/abs/ 2504.13818.

An Yang et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2503.14476.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023. URL https://arxiv.org/abs/2308.01825.

Yang Yue et al. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems, 2025. URL https: //arxiv.org/abs/2504.13837.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems, 2022. URL https: //arxiv.org/abs/2203.14465.

Chunting Zhou et al. LIMA: Less is more for alignment. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2305.11206.

Brian D. Ziebart. Modeling Purposeful Adaptive Behavior with the Principle ofMaximum Causal Entropy. PhD thesis, Carnegie Mellon University, 2010.

## A PROOFS

Throughout, $\varphi ( u ) = u \log u - u + 1$ , which satisfies $\varphi \geq 0 , \varphi ( 1 ) = \varphi ^ { \prime } ( 1 ) = 0 ,$ , and $\varphi ^ { \prime \prime } ( u ) = 1 / u$

## A.1 GIBBS TARGET AND OFFLINE FITTING

Proof of Proposition 2.1. Fix x and write $p = \pi ^ { \star } ( \cdot \mid x )$ . Since $r = \beta \log ( p / \pi _ { \mathrm { r e f } } ) + \beta \log Z ( x )$ , for any π with finite KL divergence to $\pi _ { \mathrm { r e f } }$

$$
\begin{array} { r } { J _ { x } ( \pi ) = \mathbb { E } _ { \pi } \left[ \beta \log { \frac { p } { \pi _ { \mathrm { r e f } } } } \right] + \beta \log Z ( x ) - \beta \mathbb { E } _ { \pi } \left[ \log { \frac { \pi } { \pi _ { \mathrm { r e f } } } } \right] = \beta \log Z ( x ) - \beta \operatorname { K L } ( \pi \| p ) . } \end{array}
$$

Nonnegativity of KL shows that p is the unique maximizer with $J _ { x } ( p ) = \beta \log Z ( x )$ , and subtracting gives Eq. (3). 口

Derivation of Eq. (4). $\begin{array} { r } { \mathrm { K L } ( p \| \pi _ { \theta } ) = \sum _ { y } p _ { y } \log p _ { y } - \sum _ { y } p _ { y } \log \pi _ { \theta } ( y \mid x ) } \end{array}$ . The first term does not depend on θ. If $\mu ( \boldsymbol { y } \mid \boldsymbol { x } ) > 0$ wherever $p _ { y } > 0 ;$ , the second term equals $\mathbb { E } _ { y \sim \mu } [ - ( p _ { y } / \mu _ { y } )$ log $\pi _ { \boldsymbol { \theta } } ( y ~ |$ x)], and $p _ { y } / \mu _ { y } = \rho ( x , y )$ by Eq. (2).

## A.2 KL-CURVATURE BOUNDS AND BUDGETED SELECTION

ProofofTheorem 3.1. Because $\begin{array} { r } { \sum _ { y } ( p _ { y } - q _ { y } ) = 0 } \end{array}$ , we have $\begin{array} { r } { \mathrm { K L } ( p \| q ) = \sum _ { y } q _ { y } \varphi ( p _ { y } / q _ { y } ) } \end{array}$ and $\begin{array} { r } { \mathrm { K L } ( q \| p ) = \sum _ { u } p _ { y } \varphi ( q _ { y } / p _ { y } ) } \end{array}$ . For u in an interval $[ m , M ]$ with $0 < m \le \mathrm { ~ i ~ } \le M$ , Taylor’s theorem around $u = 1$ gives $( u - 1 ) ^ { 2 } / ( 2 M ) \le \varphi ( u ) \le ( u - 1 ) ^ { 2 } / ( 2 m )$ . With $u = p _ { y } / q _ { y } \in [ \gamma , \Gamma ]$

$$
\frac { ( p _ { y } - q _ { y } ) ^ { 2 } } { 2 \Gamma q _ { y } } \leq q _ { y } \varphi \Bigl ( \frac { p _ { y } } { q _ { y } } \Bigr ) \leq \frac { ( p _ { y } - q _ { y } ) ^ { 2 } } { 2 \gamma q _ { y } } ,
$$

and with $u = q _ { y } / p _ { y } \in [ \Gamma ^ { - 1 } , \gamma ^ { - 1 } ]$

$$
\frac { \gamma ( p _ { y } - q _ { y } ) ^ { 2 } } { 2 p _ { y } } \leq p _ { y } \varphi \Big ( \frac { q _ { y } } { p _ { y } } \Big ) \leq \frac { \Gamma ( p _ { y } - q _ { y } ) ^ { 2 } } { 2 p _ { y } } .
$$

Multiplying the second pair by λ and adding gives $c _ { - } U _ { y } ^ { \lambda } \leq K _ { y } ^ { \lambda } \leq c _ { + } U _ { y } ^ { \lambda }$ for each $y ,$ and summing over y gives the theorem. For the local statement, $\begin{array} { r } { \varphi ( 1 + \bar { h } ) = \frac { \bar { 1 } } { 2 } h ^ { 2 } + O ( | h | ^ { 3 } ) } \end{array}$ applied to both ratios gives ${ \cal D } _ { \lambda } = { \textstyle { \frac { 1 } { 2 } } } U ^ { \lambda } + { \cal O } ( \delta U ^ { \lambda } )$ with $\begin{array} { r } { \delta = \operatorname* { s u p } _ { y } | p _ { y } / q _ { y } - 1 | \to 0 . } \end{array}$ □

ProofofCorollary 3.2. Using the per-response bounds and the optimality of $S _ { U }$ for $\begin{array} { r } { U ^ { \lambda } , \sum _ { S _ { U } } K _ { y } ^ { \lambda } \ge } \end{array}$ $\begin{array} { r } { c _ { - } \sum _ { S _ { U } } U _ { y } ^ { \lambda } \ge c _ { - } \sum _ { S ^ { \star } } U _ { y } ^ { \lambda } \ge ( c _ { - } / c _ { + } ) \sum _ { S ^ { \star } } K _ { y } ^ { \lambda } } \end{array}$ □

Proof of Corollary 3.3. Write $U _ { y } ^ { \lambda } = ( p _ { y } - q _ { y } ) ^ { 2 } / q _ { y } + \lambda ( p _ { y } - q _ { y } ) ^ { 2 } / p _ { y } . \mathrm { ( i ) }$ With $a = q _ { y } / p _ { y } \to 0$ , the first term is $p _ { y } ^ { 2 } ( 1 - a ) ^ { 2 } / q _ { y }$ and the second is $\lambda p _ { y } ( 1 - a ) ^ { 2 }$ ; their ratio is $1 / ( \lambda a ) \to \infty$ , so $U _ { y } ^ { \lambda } \sim p _ { y } ^ { 2 } / q _ { y }$ (ii) With $b = p _ { y } / q _ { y } \to 0$ , the second term is $\lambda q _ { y } ^ { 2 } ( 1 - b ) ^ { 2 } / p _ { y }$ and the first is $q _ { y } ( 1 - b ) ^ { 2 }$ ; their ratio is $\lambda / b  \infty ,$ so $U _ { y } ^ { \lambda } \sim \lambda q _ { y } ^ { 2 } / p _ { y }$ . At fixed target probability, the first expression decreases and the second increases in $q _ { y }$ □

## A.3 SELECTION AT THE REFERENCE POLICY

Proof of Proposition 3.4. By Eq. $( 2 ) , p _ { y } = q _ { y } e ^ { r / \beta } / Z ( x ) = \kappa _ { r } q _ { y }$ when $q = \pi _ { \mathrm { r e f } }$ . Substituting into $K _ { y } ^ { \lambda } = q _ { y } \varphi ( p _ { y } / q _ { y } ) + \lambda p _ { y } \varphi ( q _ { y } / p _ { y } )$ gives $q _ { y } \varphi ( \kappa _ { r } ) + \lambda \kappa _ { r } q _ { y } \varphi ( 1 / \kappa _ { r } )$ , and $\kappa \varphi ( 1 / \kappa ) = \kappa - 1 - \log \kappa$ yields $K _ { y } ^ { \lambda } ~ = ~ q _ { y } h _ { \lambda } ( \kappa _ { r } )$ . With $r \ \in \ \{ 0 , 1 \} , \ Z ( x ) \ = \ e ^ { 1 / \beta } P ( x ) + 1 - P ( x ) \ = \ e ^ { 1 / \beta } P ( x ) ( 1 +$ $O ( e ^ { - 1 / \beta } / P ( x ) ) )$ , so $\kappa _ { 1 } = e ^ { 1 / \beta } / Z ( x ) \approx 1 / P ( x )$ and $\kappa _ { 0 } = 1 / Z ( x ) \approx e ^ { - 1 / \beta } / P ( x )$ □

Between-class constants (Remark 3.5). At $q = \pi _ { \mathrm { r e f } }$ , the density ratios range from $\gamma = \kappa _ { 0 }$ to $\Gamma = \kappa _ { 1 } , \mathbf { s o ~ } c _ { - } / c _ { + } = \gamma ^ { 2 } \approx e ^ { - 2 / \beta } / P ( x ) ^ { 2 }$ , below $1 0 ^ { - 7 } \mathrm { a t } \beta = 0 . 1$ for $P ( x ) \ge 0 . 2 5$ . For $\lambda = 1$ and $P ( x ) \in [ 0 . 2 5 , 0 . 7 5 ]$ , Eq. (8) gives an incorrect-to-correct score ratio between about $5 \times 1 0 ^ { 2 }$ and $8 . 5 \times \mathrm { i } 0 ^ { 4 }$ for responses of equal probability, whereas the exact ratio $h _ { \lambda } ( \kappa _ { 0 } ) / h _ { \lambda } ( \kappa _ { 1 } )$ is between 2 and 101.

## A.4 TOKEN-LEVEL RESULTS

Chain rule and target conditional. With a finite horizon and terminated sequences padded by an absorbing state, $\begin{array} { r } { \tilde { \log } ( p ( \boldsymbol { y } \mid \boldsymbol { x } ) / q ( \boldsymbol { y } \mid \boldsymbol { x } ) ) = \sum _ { t } \log ( p _ { t } ( y _ { t } \mid \boldsymbol { s } _ { t } ) / q _ { t } ( y _ { t } \mid \boldsymbol { s } _ { t } ) ) } \end{array}$ . Taking expectations under $p$ and conditioning each term on its prefix gives the forward part of Eq. (6); exchanging p and q gives the reverse part. For the conditional, summing the Gibbs target over completions of a prefix gives $p ( s ) = \bar { \pi _ { \mathrm { r e f } } } ( s ) Z ( s ) / Z ( x )$ , and the ratio of $p ( s a ) ~ \mathsf { t o } ~ p ( s )$ is Eq. (5). Applying the proof of Theorem 3.1 to $( p _ { t } ( y _ { t } \mid s _ { t } ) , q _ { t } ( y _ { t } \mid s _ { t } ) )$ ) gives $c _ { - } \bar { U } _ { t } ^ { \lambda } \leq K _ { t } ^ { \lambda } \leq c _ { + } U _ { t } ^ { \bar { \lambda } }$ , and the proofs of Corollaries 3.2 and 3.3 apply verbatim to tokens, which proves Corollary 3.6.

ProofofProposition 3.7. At $\pi _ { \theta } = \pi _ { \mathrm { r e f } } .$ , Eq. (5) gives $p _ { t } / q _ { t } = \rho _ { t }$ , and the computation in the proof of Proposition 3.4 gives $K _ { t } ^ { \lambda } = q _ { t } h _ { \lambda } ( \rho _ { t } )$ . Since $Z ( s _ { 1 } ) = Z ( x )$ and $Z ( s _ { T + 1 } ) = e ^ { r ( x , y ) / \beta }$ , the product $\prod _ { t } \rho _ { t }$ telescopes to $\kappa _ { r ( x , y ) } ( x )$ . Finally, $\begin{array} { r } { Z ( s _ { t } ) = \sum _ { a } \pi _ { \mathrm { r e f } } ( a \mid s _ { t } ) Z ( s _ { t } a ) \ge q _ { t } Z ( s _ { t } y _ { t } ) , \mathrm { s o } \rho _ { t } \le 1 / q _ { t } } \end{array}$ For the prefix-weighted share, the forward part $p ( s _ { t } ) q _ { t } \varphi ( \rho _ { t } )$ uses $p ( s _ { t } ) = q ( s _ { t } ) Z ( s _ { t } ) / Z ( x )$ , and the reverse part $\lambda q ( s _ { t } ) p _ { t } \varphi ( q _ { t } / p _ { t } )$ uses $\rho \varphi ( 1 / \rho ) = \rho - 1 - \log \rho$ with $p _ { t } = \rho _ { t } q _ { t }$ □

Eq. (9). h<sup>′</sup> (κ) = log $\kappa + \lambda ( 1 - 1 / \kappa ) \geq 0$ for $\kappa \geq 1$ , so $K _ { t } ^ { \lambda } = q _ { t } h _ { \lambda } ( \rho _ { t } ) \leq q _ { t } h _ { \lambda } ( 1 / q _ { t } )$ whenever $1 \le \rho _ { t } \le 1 / \dot { q } _ { t }$ . Expanding gives the stated expression, whose derivative in $q _ { t } \mathrm { i s } ( 1 - 1 / q _ { t } ) + \lambda \log q _ { t } <$ 0 on (0, 1). Along a correct response $\begin{array} { r } { \sum _ { t } \bar { \log \rho _ { t } } = \log \kappa _ { 1 } \approx \log ( 1 / P ( x \bar { ) } ) } \end{array}$ ), and along an incorrect response $\begin{array} { r } { \sum _ { t } \log \rho _ { t } = \log \kappa _ { 0 } \approx - 1 / \beta \stackrel { \mathbf { \rho } } { - } \log P ( x ) } \end{array}$

Eq. (10). By Proposition 3.4, the total contribution of class r is $\begin{array} { r } { h _ { \lambda } \big ( \kappa _ { r } \big ) \sum _ { y : r ( x , y ) = r } q _ { y } , } \end{array}$ which equals $P ( x ) h _ { \lambda } ( \kappa _ { 1 } )$ for $r = 1$ and $( 1 - P ( x ) ) h _ { \lambda } ( \kappa _ { 0 } )$ $r = 0$ . The values quoted in Remark 3.8 follow by direct evaluation.

## B IMPLEMENTATION DETAILS

## B.1 DATA CONSTRUCTION

For each training prompt, the initial policy $\pi _ { 0 }$ generates $N _ { l } = 1 6$ responses at $\tau _ { l } = 0 . 3$ and $N _ { h } = 1 6$ at $\tau _ { h } = 1 . 0$ , without top-p or top-k truncation. Rollouts use the chat template with thinking disabled, seed 42, and a stop at the closing </answer> tag, with at most 1,024 new tokens for GSM8K and MATH and 2,048 for MBPP. Validation sets are 500 held-out training questions for GSM8K and for MATH, and MBPP task IDs 511–600. The verifier assigns $r = 1$ to correct and $r = 0$ to incorrect responses: final-answer matching for GSM8K and MATH-500, and the unit tests for MBPP-full. Malformed or truncated responses are kept and verified normally; a response from which no answer can be extracted or verified receives $r = 0 .$ . Duplicate responses are kept as separate instances. A prompt is eligible if it has at least $B _ { + } = 4$ correct $\tau _ { h }$ responses and at least $B _ { - } = 4$ incorrect τ responses, and GIST samples four of each uniformly without replacement. Table 7 reports the eligible prompts; on Qwen3-0.6B/GSM8K, GIST retains 12,744 of the 50,976 candidates of the 1,593 eligible prompts. In Table 2, BOLT and SFT train on all 32 candidates of every training prompt, and each objective with + GIST trains on GIST’s selected subset. The controlled comparisons of Sections 5.2–5.4 use only the eligible prompts, and “all candidates” there refers to the 32 candidates of each eligible prompt.

Table 7: Eligible prompts over training prompts.
<table><tr><td>Model</td><td>GSM8K</td><td>MATH</td><td>MBPP</td></tr><tr><td>Qwen3-0.6B</td><td>1,593 / 6,973</td><td>1,619 / 7,000</td><td>66/374</td></tr><tr><td>Qwen3.5-2B</td><td>2,574 / 6,973</td><td>1,100 / 7,000</td><td>65 / 374</td></tr><tr><td>Gemma-4-E2B-it</td><td>1,373 / 6,973</td><td>1,646 / 7,000</td><td>58 / 374</td></tr></table>

## B.2 TOKEN MASKS

Token probabilities $q _ { t } = \pi _ { 0 } ( y _ { t } \mid x , y _ { < t } )$ are computed once with the frozen initial policy and without temperature scaling, and the masks stay fixed during training. The mean q¯ is taken over the eligible response positions, which exclude prompt, padding, and EOS positions, so EOS never enters q¯ or the mask. Comparisons are strict, so positions equal to the mean are not selected. Whenever GIST’s token selection is applied, correct responses use $m _ { t } ^ { + } = \mathbf { 1 } [ q _ { t } < \bar { q } ]$ and incorrect responses use $m _ { t } ^ { - } = \mathbf { 1 } [ q _ { t } > \bar { q } ]$ ; the baselines keep their own token rules (Appendix C). Unselected positions remain in the context but receive no loss.

## B.3 TRAINING OBJECTIVE

BOLT weights are computed within each prompt’s full pool of 32 candidates before selection, as described in Section 4.3, with $r \in \{ 0 , 1 \}$ and $\beta \stackrel { - } { = } 0 . 1$ , and are not renormalized after selection. An incorrect response therefore has weight $e ^ { - 1 0 } \approx 4 . 5 \times 1 0 ^ { - 5 }$ times that of a correct response to the same prompt. No off-policy correction is applied to responses from the $\tau _ { l }$ pool, so $3 2 / n _ { 1 }$ estimates the inverse success rate of the mixed-temperature pool rather than $1 / P ( x )$ ; the BOLT baseline uses the same mixed pool. For a prompt whose 32 candidates are all incorrect, every weight equals one, so BOLT imitates these responses; GIST excludes such prompts through eligibility. The loss multiplies each response’s mean masked token negative log-likelihood by its weight and averages over the minibatch. SFT uses the same loss with $w = 1$ for every response. $\mathrm { S \bar { F } T + G I S T }$ , which is also row (e) of Table 4, applies GIST’s selection and masks, so the positions of incorrect responses that $m ^ { - }$ selects are imitated with the same weight as the positions of correct ones. All other settings match BOLT. We fine-tune LoRA adapters (rank $8 , \alpha = 6 4 .$ , dropout 0.05) with fused AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1), a constant schedule with 10% warmup, BF16, and gradient checkpointing. Learning rates are $\dot { 5 } \times 1 0 ^ { - 6 }$ for GSM8K and $1 0 ^ { - 5 }$ for MATH and MBPP; the global batch size is 32 for GSM8K and 128 for MATH and MBPP, and the maximum training length is 1,024 and 4,096 tokens, respectively. Training runs for up to 12 epochs with early stopping (patience 2) after at least 5 epochs, and the checkpoint with the best validation Strict accuracy (pass@1 for MBPP) is evaluated on the test set.

## B.4 EVALUATION

GSM8K is evaluated with lm-evaluation-harness (Gao et al., 2024) using 8-shot Q:/A: prompts without the chat template and at most 256 new tokens. GSM8K Strict requires “The answer is . . . ” followed by an exact match, and MATH Strict requires an exact match of the normalized last boxed answer; Flexible accuracy extracts the final answer from other formats. MBPP-full reports pass@1, executing the benchmark’s test cases in a bubblewrap sandbox with a 5-second timeout and CPU, memory, and file-size limits. Each entry is a single training run. The binomial standard error of a single test accuracy is about 1.4 points on GSM8K $( n = 1 , 3 1 9 )$ and 2.2 points on MATH-500 and MBPP-full $( n = 5 0 0 ) ;$ ; Appendix B.5 reports paired tests for the main GSM8K comparisons.

## B.5 PAIRED TESTS AND TOKEN STATISTICS

Table 8 reports paired tests on the 1,319 GSM8K test questions (Qwen3-0.6B), using per-question correctness of the two compared checkpoints: a paired bootstrap 95% interval for the accuracy difference (Efron & Tibshirani, 1993) and McNemar’s test (McNemar, 1947), following Dror et al. (2018).

Table 8: Paired tests on GSM8K (Qwen3-0.6B, 1,319 questions). Differences are computed from per-question counts and may differ by 0.01 from differences of the rounded accuracies in the main text.
<table><tr><td>Comparison</td><td>Metric</td><td>Difference (pp)</td><td>Bootstrap 95% CI</td><td>McNemar p</td></tr><tr><td>GIST – BOLT</td><td>Strict</td><td>+2.50</td><td>[+0.30, +4.62]</td><td>0.027</td></tr><tr><td rowspan="2">GIST – Random</td><td>Flex.</td><td>+2.27</td><td>[+0.15, +4.40]</td><td>0.044</td></tr><tr><td>Strict</td><td>+2.65</td><td>[+0.61, +4.70]</td><td>0.014</td></tr><tr><td rowspan="2">GIST — All candidates</td><td>Flex.</td><td>+1.14</td><td>[−0.83, +3.11]</td><td>0.293</td></tr><tr><td>Strict</td><td>+1.97</td><td>[−0.08, +4.02]</td><td>0.072</td></tr><tr><td></td><td>Flex.</td><td>+3.03</td><td>[+0.99, +5.08]</td><td>0.005</td></tr></table>

Below-mean tokens by sampling temperature. We measure the fraction of positions with $q _ { t } < \bar { q }$ in correct responses, excluding EOS. Pooled over tokens, it is 31.57% at $\tau _ { l }$ and 31.93% at $\tau _ { h }$ over all training prompts, and 31.38% and 31.75% over the 1,593 eligible prompts. With each response weighted equally, it is 31.39% and 30.69% over all prompts, and 31.22% and 30.56% over eligible prompts. Comparing correct responses within the same prompt, the high-temperature fraction is lower by 0.36 points over all prompts (95% CI [−0.50, −0.21]) and by $0 . { \bar { 5 } } 7$ points over eligible prompts $( [ - 0 . 7 4 , - 0 . 4 1 ] )$ . Both temperatures thus contain a substantial fraction of below-mean tokens, and the sign of the small difference depends on aggregation, reflecting differences in response length. The probability of the selected tokens differs more: on the eligible prompts, below-mean tokens of correct responses have mean (median) $q _ { t }$ of 0.583 (0.603) at $\tau _ { l }$ and 0.411 (0.415) at $\tau _ { h }$ . The above-mean tokens that $m ^ { - }$ selects in incorrect responses have mean (median) $q _ { t }$ of 0.974 (0.998) at $\tau _ { l }$ and 0.956 (0.993) at $\tau _ { h }$

## B.6 RETAINED DATA AND SUPERVISION

Table 9 extends Table 6 with the retained responses, selected epochs, and optimizer steps. Supervised tokens count active loss positions through the selected checkpoint, including repeated use across epochs and the positions that $m ^ { - }$ selects in incorrect responses. Relative to training on all candidates, Random supervises 51.2% fewer tokens and GIST 66.6% fewer; GIST supervises 31.5% fewer than Random. Random and GIST retain the same number of responses but not the same responses, so the per-epoch reduction from masking also reflects response-length differences.

Table 9: Retained data and supervision on the shared 1,593-prompt pool (Qwen3-0.6B/GSM8K, BOLT).
<table><tr><td>Quantity</td><td>All candidates</td><td>Random</td><td>GIST</td></tr><tr><td>Retained responses</td><td>50,976</td><td>12,744</td><td>12,744</td></tr><tr><td>Selected epoch</td><td>3</td><td>5</td><td>6</td></tr><tr><td>Supervised tokens (M)</td><td>13.596</td><td>6.641</td><td>4.546</td></tr><tr><td>Supervised tokens per epoch (M)</td><td>4.53</td><td>1.33</td><td>0.76</td></tr><tr><td>Optimizer steps</td><td>4,779</td><td>1,995</td><td>2,394</td></tr><tr><td>Test Strict / Flex. (%)</td><td>43.82 / 46.70</td><td>43.14 / 48.60</td><td>45.79 / 49.73</td></tr></table>

## C BASELINES AND ABLATION CONTROLS

All selectors and controls use the shared candidate pool, the 1,593 eligible prompts, and the BOLT objective of Appendix B, unless stated otherwise.

Selection baselines. Random samples four correct and four incorrect responses per prompt uniformly from the combined temperature pools and supervises all eligible tokens. RFT-style retains all verifier-correct responses. The HES, PODS, ProFit, ETS, and RSI-S adaptations apply each method’s selection rule within this pipeline instead of reproducing its full training algorithm, and each retains eight responses per prompt. HES scores a response by the summed predictive entropy of its top 0.5% highest-entropy positions and keeps the four highest-scoring correct responses and four uniformly chosen incorrect ones. PODS keeps the eight responses that maximize the variance of their binary rewards, which gives four of each class. Both supervise all eligible tokens. ProFit, ETS, and RSI-S keep GIST’s responses and select tokens in both classes with one shared rule. ProFit keeps tokens whose frozen-policy probability is at least 0.1 (96.9% of tokens), ETS keeps the 20% of tokens with the highest predictive entropy, and RSI-S keeps tokens whose score $1 + \log p _ { t } / H _ { t }$ lies in [−6, 0.95] (99.3% of tokens).

Ablation controls (Tables 4 and 5). The Mask column of Table 4 and the token rules of Table 5 vary the treatment of correct responses, on which the BOLT weight concentrates. In Table 4, row (a) trains on all 32 candidates of each eligible prompt with all eligible tokens, and row (b) is the Random selector. Row (c) applies GIST’s masks to the same responses as Random. Row (d) keeps GIST’s responses, supervises all eligible tokens of correct responses, and uses $m ^ { - }$ for incorrect ones; it is the All tokens entry of Table 5. Row (e) keeps GIST’s responses and masks and sets w = 1 for every response, which is SFT + GIST. In the token-rule ablation, Random keeps, in each correct response, as many uniformly chosen positions as GIST selects, and Highest-probability keeps the same number of highest-probability positions; incorrect responses use $m ^ { - }$ in all variants.