# Beyond Proxy Metrics: MLLMs as Human Surrogates for Explainable AI Evaluation

Eoin M. Kenny<sup>∗</sup>   
AI Research   
JP Morgan   
25 Bank Street, London, UK   
eoin.kenny@jpmorgan.com

Salim I. Amoukou AI Research JP Morgan 25 Bank Street, London, UK salim.ibrahimamoukou@jpmorgan.com

Tom Bewley Mistral AI London, UK tom.bewley@mistral.ai<sup>†</sup>

## Abstract

Explainable AI (XAI) has expanded into a broad and impactful research area, but the fundamental question of how best to evaluate explanation techniques has proven frustratingly elusive. Methodologies typically revolve around either proxy metrics which tell us little about the utility of an explanation, or human evaluation which—while generally considered the gold standard—is expensive, difficult to scale, and challenging to reproduce. In this paper, we investigate if Multimodal Large Language Models (MLLMs) can act as surrogate human participants to automate the evaluation of XAI. To do this, we first propose a “purpose-grounded” evaluation framework with metrics applicable to both MLLMs and human users alike. Then, we offer some conceptual insights as to why and when we can trust these models to automate this process. Lastly, we conduct extensive empirical testing across five purpose-grounded areas such as model improvement and fourteen popular MLLMs. Results indicate that all models trend positively, but OpenAI and Qwen models generally show higher alignment compared to other variants, which would be the recommended choices in practice. We expect our insights not to replace user testing, but rather to help XAI researchers complement their current evaluation methodologies and move the field away from heavy reliance on proxy metrics for measuring explanation utility. <sup>3</sup>

## 1 Introduction

Explainable AI (XAI) has existed since at least the 1980s [29], but its urgency has increased over the past decade as AI systems have become increasingly prevalent in high-stakes domains such as healthcare and criminal justice [2, 1]. In these contexts, understanding the rationale behind AI decisions is often as critical as the decisions themselves. However, the proliferation of XAI method has outpaced the development of meaningful evaluation methodologies.

Most studies and proposed evaluation metrics rely on proxy evaluations that rarely reflect an explanation’s usefulness to end-users [12]. For instance, counterfactual explanations frequently aim to minimize the $L _ { 0 }$ norm, indicating the fewest feature changes required to alter an outcome [38, 21, 22]. Yet, explanations optimized for such mathematical metrics often fail to offer actionable suggestions, or do not convince users to actually follow the advice [35], making their overall utility limited. This disconnect highlights a critical gap, proxy evaluations predominantly measure internal characteristics of explanations without sufficiently considering if this translates into achieving actual user goals or overall practical utility. User studies can be run to verify true explanation usefulness if focused on outcome-driven metrics such as model improvement that serve a clear purpose, but they are notoriously challenging to design and reproduce [4, 30]. As a result, researchers have begun exploring the usage of language models to automate this process, but the investigations so far have either not been XAI-specific [7] or narrowly focused [8].

![](images/e4ac0a2c5989dde8f165e7a02861e0ef19bb99e2e8a50ad0fa886abadd6f1293.jpg)  
Figure 1: Purpose-grounded evaluation framework. Our framework for how XAI evaluation can be complemented using MLLMs as surrogate human users. We structure this evaluation around purpose-grounded domains in the literature because they contain clear evaluation metrics and are the areas XAI has proven most useful, here we show five of those. With both the purpose and metric to hand, researchers can simulate appropriate user testing to help identify if their method is more promising than current baselines or not. Importantly, note this framework can (and should) also be used with real humans in place of the MLLM, as we illustrate during this paper’s experiments in Sections 4.1-4.5.

The idea of an explanation needing to serve a clear purpose has been discussed previously by several authors [17, 11, 12]. The core tenet of this approach is that the development of XAI techniques should be guided by their intended downstream applications and practical utility, rather than by methodological novelty alone. Several recent survey papers have begun converging on at least five clear areas for XAI in human-AI interaction, namely the fields of appropriate reliance [37], model improvement [15], algorithmic recourse [19], knowledge extraction [42], and system engagement [43], so we use these as a starting point for our framework in Figure 1, but acknowledge more may exist. Borrowing from the literature [11, 12], we refer to these as ‘purpose-grounded’ XAI areas. We are aligned with all these authors conceptually, but interested in how MLLMs can help automate such evaluations. Our contributions are:

Contributions (1) We propose a formal framework for purpose-grounded XAI evaluation in human-AI interaction, with precisely defined metrics across five areas (Section 2, Figure 1). (2) We provide a conceptual analysis of the conditions under which MLLMs can be trusted as surrogates for human responses to rank order explanation methods (Section 3). (3) We contribute an empirical benchmark across five purpose-grounded domains and 14 MLLMs, identifying where rank replication robustly holds and systematic failure conditions (Section 4).

## 2 Evaluation Metrics

In order to enable potential practitioners, we formally define all evaluation metrics for the usage of MLLMs—and indeed human participants—in our framework. Specifically, we examine five purpose-grounded areas for XAI defined within a human-computer interaction setting, where a human stakeholder H is receiving an explanation ξ from any source, while performing some activity. The explanation $\xi$ can be of any form (e.g., local, global, counterfactual, factual, and combinations thereof). Furthermore, we assume that each $( H , A I )$ pairing is sampled from a target population $\rho ,$ which could contain humans with different levels of expertise, and AI systems with different architectures, objectives and performance levels.<sup>4</sup> Given such a population, each purpose $p$ is associated with an explanation evaluation metric $M _ { p }$ , defined as the expected task-level performance induced by interaction with the explanation. Concretely, all evaluation metrics considered in this paper follow the generic pattern

$$
M _ { p } ( \xi ) = \mathbb { E } _ { ( H , A I ) \sim \rho } \big [ \phi ( \cdot ^ { + } ) \big ] , \quad \mathrm { w h e r e } \quad \cdot ^ { + } = \mathrm { I n t e r a c t i o n } ( \cdot \mid \cdot , \xi )\tag{1}
$$

where $\phi$ is a scalar criterion appropriate to purpose $p ,$ and · denotes the entity being evaluated $( \mathrm { e . g . }$ the AI system, the human, or the human–AI team), and Interaction $( \cdot \mid \cdot , \xi )$ captures the way in which interaction with explanation $\xi$ induces an updated state $\cdot ^ { + }$ . We now instantiate this into our five areas illustrated in Figure 1.

Model Improvement (MI) Here, the objective is to determine if an explanation $\xi$ aids an engineer or developer H in improving the system $A I$ according to some scalar performance criterion $\phi \left( \mathbf { e . g . } \right.$ accuracy, robustness, fairness), for which higher values are better. Let ${ \bar { A } } I ^ { + }$ denote the version of $A I$ improved by H using explanation $\xi .$ Aggregating over the population $\rho ,$ the evaluation metric $M _ { m i }$ is

$$
M _ { m i } ( \xi ) = \mathbb { E } _ { ( H , A I ) \sim \rho } \big [ \phi ( A I ^ { + } ) \big ] , \quad \mathrm { w h e r e } \quad A I ^ { + } = \mathrm { I m p r o v e } ( A I \mid H , \xi )\tag{2}
$$

where the conditioning “| $H , \xi ^ { , , }$ indicates that the way in which AI is improved depends on how H interacts with the provided explanation. For example, an engineer (H) could use feature attribution explanations (ξ) to debug and retrain a classifier model (AI), with the aim of obtaining an improved model $( A I ^ { + } )$ with better adversarial robustness (ϕ).

Knowledge Extraction (KE) This purpose assesses if an explanation $\xi$ from system AI effectively teaches a concept or skill to human $H ,$ which can be something known by domain experts, or new scientific knowledge. Success is measured by $H \mathbf { \bar { s } }$ ability to apply the learned knowledge independently on a downstream task, where performance is again measured by a scalar criterion ϕ (e.g., score on a post-teaching quiz). The evaluation metric $M _ { k e }$ is

$$
M _ { k e } ( \xi ) = \mathbb { E } _ { ( H , A I ) \sim \rho } \big [ \phi ( H ^ { + } ) \big ] , \quad \mathrm { w h e r e } \quad H ^ { + } = \mathrm { T e a c h } ( H \mid A I , \xi )\tag{3}
$$

where the conditioning $" | \ A I , \xi "$ indicates that the provided explanation is used in conjunction with any other outputs from AI to facilitate the teaching of $H$ . For example, this could measure the accuracy (ϕ) of a medical student (H) when diagnosing, after being taught by an AI using rule-based explanations (ξ).

Reliance Calibration (RC) In this setting, H and AI are jointly working to perform a task in a human-AI team, denoted by HAI. H is assumed to have the ultimate decision-making authority about which actions are taken, and thus must decide when to rely on (and conversely, when to disregard) the outputs of $A I$ . The purpose of an explanation $\xi$ in this setting is to assist $H$ in adapting their reliance strategy, such that the complementary performance of the human-AI team (once again defined by a scalar criterion ϕ) is improved. The evaluation metric $M _ { r c }$ is therefore

$$
M _ { r c } ( \xi ) = \mathbb { E } _ { ( H , A I ) \sim \rho } \big [ \phi ( H A I ^ { + } ) \big ] , \quad \mathrm { w h e r e } \quad H A I ^ { + } = \mathrm { A d a p t } ( H A I \mid \xi )\tag{4}
$$

where the conditioning $\mathbf { \partial } ^ { 6 6 } \vert \mathbf { \nabla } \xi ^ { 5 } \mathbf { \vec { \xi } }$ indicates that the adapted behavior of the human-AI team depends on the given explanation. For example, this metric could measure the accuracy of predicting phantom braking (ϕ) by a safety driver (H) working with a self-driving car $( A I )$ , where the car’s chosen trajectories are augmented by concept-based explanations (ξ) [23].

Actionable Recommendation (AR) This purpose evaluates if an explanation $\xi$ provides effective and adoptable actions for a user H that is subject to decisions made by AI. A canonical example of this case is automated credit scoring and loan approval, for which a rejected applicant may wish to take recourse actions to become accepted in future. Effectiveness depends on two factors: whether or not the user adopts the recommended action, denoted by the binary variable ‘Follows $( H \mid \xi ) ^ { \prime }$ , and the positive benefit derived from it if they do, ‘Benefit $( { \dot { H } } , A I \mid \xi ) ^ { \prime }$ . Note that the concept of benefit can encompass direct benefits to H (e.g., the money they receive as their loan) but also to other stakeholders $( \mathrm { e . g . }$ ., the interest that the bank stands to gain during repayment). The evaluation metric $M _ { a r }$ takes the product of these two factors for each $( H , A I )$ pair, and therefore gives the expected benefit weighted by the likelihood of adoption across the population:

$$
M _ { a r } ( \xi ) = \mathbb { E } _ { ( H , A I ) \sim \rho } \big [ \mathrm { F o l l o w s } ( H \mid \xi ) \times \mathrm { B e n e f i t } ( H , A I \mid \xi ) \big ] .\tag{5}
$$

For $M _ { a r }$ to be high, reflecting greater overall utility, both components must contribute positively. This signifies that the explanation leads to recommendations that are both followed and beneficial. For instance, robust explanations that are valid over time might have greater potential benefits [10], while clear, simple and persuasive explanations might have higher rates of being followed [35].

System Engagement (SE) This purpose evaluates whether an explanation ξ enhances the engagement of human H with system AI, where this system is typically one that recommends content for entertainment or education. Engagement is measured by a suitable metric (e.g., interaction time or depth). The evaluation metric $M _ { s e }$ is:

$$
M _ { s e } ( \xi ) = \mathbb { E } _ { ( H , A I ) \sim \rho } \big [ \mathrm { E n g a g e m e n t } ( H , A I \mid \xi ) \big ] .\tag{6}
$$

An example is measuring how frequently movie streaming platform users (H) follow the suggestions of a recommender system (AI) when textual explanations $( \xi )$ for those recommendations are provided.

## 2.1 Ranking explanations

For any given explanation $\xi ,$ each of these purpose-grounded metrics only gains meaning when compared to alternatives. In general, studies should consider a set of n candidate explanations (including a no-explanation control), and construct a ranking of this set by comparing their metric values over a population $\rho$ of experimental subjects and AI systems. For a given purpose $p ,$ the ranking is defined as follows:

$$
\xi _ { i } \succeq _ { p } \xi _ { j } \Leftrightarrow M _ { p } ( \xi _ { i } ) \geq M _ { p } ( \xi _ { j } ) , \qquad \xi _ { 1 } \succeq _ { p } \xi _ { 2 } \succeq _ { p } \dots \succeq _ { p } \xi _ { n } .\tag{7}
$$

When determining the best explanation for $p ,$ , it is this ranking that is of particular importance, not the raw metric scores themselves. This insight is inspired by recent research replicating human studies with LLMs and peer review [7, 8, 31], illustrating that main effects generally replicate, but not the subtleties regarding actual means and distributions.

## 3 Conceptual Analysis of the Ranking Phenomenon

The framework in Section 2 compares rankings rather than requiring agreement in raw metric values (Equation 7). This distinction is motivated by research comparing human and MLLM responses in experiments and peer review [7, 8, 31]. We sketch an argument for how rankings may be preserved despite differences in response distributions, and where this may break down. Our argument uses two working assumptions: (1) the MLLM has learned response patterns relevant to the task, allowing its responses to track aspects of human responses imperfectly; and (2) the comparison involves structurally distinct explanation types, such as counterfactual explanations and control.

## 3.1 Human heterogeneity and MLLM sampling variability

Consider a study population $\rho$ in which each individual h has latent attributes $\eta _ { h }$ (e.g., expertise, attention). Given an explanation $\xi ,$ individual h produces a response y drawn from $\bar { \cal P } ( y \mid \xi , \eta _ { h } )$ The full population response distribution is $\begin{array} { r } { P _ { H } (  { j } |  { \xi } ) = \int \bar { P ( y \mid \xi , \eta _ { h } ) } d \rho ( \eta _ { h } ) } \end{array}$ , with variance $\sigma _ { H } ^ { 2 } ( \boldsymbol { \xi } ) : = \operatorname { V a r } _ { P _ { H } ( \boldsymbol { y } | \boldsymbol { \xi } ) } ( \boldsymbol { y } )$ . By the law of total variance,

$$
\sigma _ { H } ^ { 2 } ( \boldsymbol { \xi } ) = \underbrace { \mathbb { E } _ { \eta _ { h } } [ \mathrm { { V a r } } ( y \mid \boldsymbol { \xi } , \eta _ { h } ) ] } _ { \mathrm { { i n d i v i d u a l ~ s t o c h a s t i c i t y } } } + \underbrace { \mathrm { V a r } _ { \eta _ { h } } [ \mathbb { E } ( y \mid \boldsymbol { \xi } , \eta _ { h } ) ] } _ { \mathrm { { p o p u l a t i o n ~ h e t e r o g e n e i t y } } } .\tag{8}
$$

The first term captures variation within individuals and the second systematic differences across people. For a fixed MLLM and prompt, let $p _ { \theta } ( y \mid \xi ; T )$ denote its response distribution at temperature $T ,$ , with variance $\sigma _ { M } ^ { 2 } ( \boldsymbol { \xi } ) : = \mathrm { V a i } \bar { { \bf { r } } } _ { p _ { \theta } ( y | \boldsymbol { \xi } ; T ) } ( \boldsymbol { \bar { y } } )$ . Repeated sampling captures model-output variability, but does not establish that human population heterogeneity has been reproduced.

## 3.2 Why ordering may survive compression

Let $\mu _ { H } ( \boldsymbol { \xi } )$ and $\mu _ { M } ( \boldsymbol { \xi } )$ denote the human and MLLM means for explanation $\xi .$ These need not match, and we define

$$
\mu _ { M } ( \boldsymbol { \xi } ) = \mu _ { H } ( \boldsymbol { \xi } ) + \epsilon ( \boldsymbol { \xi } ) ,\tag{9}
$$

where $\epsilon ( \xi )$ is a condition-specific calibration error. An analogy is how two miscalibrated thermometers can still order temperatures correctly when the difference between their errors is smaller than the temperature gap. Similarly, if MLLM calibration errors share a common additive component across conditions, that component cancels under pairwise comparison; the remaining difference determines whether the ordering is preserved.

Formally, fix two conditions with $\mu _ { H } ( \xi _ { i } ) > \mu _ { H } ( \xi _ { j } )$ . A sufficient condition for the MLLM to preserve this ordering is

$$
| \epsilon ( \xi _ { i } ) - \epsilon ( \xi _ { j } ) | < | \mu _ { H } ( \xi _ { i } ) - \mu _ { H } ( \xi _ { j } ) | ,\tag{10}
$$

i.e., the difference in calibration errors is smaller in magnitude than the human mean gap. This follows because the MLLM mean difference equals the human mean difference plus $\epsilon ( \xi _ { i } ) - \epsilon ( \xi _ { j } )$ The condition is sufficient but not necessary: when the difference in calibration errors aligns in sign with the human mean gap, it amplifies rather than reverses the ordering, even when Equation 10 fails. Shared response patterns may produce similar calibration errors across conditions, helping preserve their ordering. When the difference in calibration errors is large relative to the human mean gap, Equation 10 may fail, allowing but not requiring an ordering reversal. This illustrates a possible failure mechanism for MLLM surrogates.

## 4 Experiments

We evaluate MLLMs using adaptations of five human studies spanning the purpose-grounded areas in Figure 1. The benchmark includes 14 models: nine API models (GPT-4o, GPT-4o mini, o4-mini, GPT-5, GPT-5 mini, GPT-5.5, Claude 4.5 Sonnet, Claude 4.6 Sonnet, and Claude 4.8 Opus) and five open-weight models (Gemma 3 27B, Gemma 4 31B, Qwen3-VL 32B, Qwen3.5 122B, and Mistral Small 3.2). Exact model identifiers and inference settings are documented in Appendix A.1. The point is not only to demonstrate examples of how our framework fits in real setups, but also how well MLLMs can replicate the results of real human testing. These specific studies were chosen because they illustrate well the five purpose-grounded areas, but they are also methodologically sound in their design (e.g., reasonable samples and control conditions). We reached out to the authors when needed for clarifications and for permission to use their figures. Building upon the discoveries of similar research [7], all our experiments used a temperature of 1.0 when prompting the MLLM.

We report full-data Spearman’s $\rho$ between MLLM condition means and fixed human reference rankings, where $\rho = 1$ indicates identical rankings, $\rho = 0$ no monotonic association, and $\rho = - 1$ reversed rankings. We estimate 95% percentile intervals using 10,000 paired bootstrap resamples with replacement, each at the observed sample size. Strong positive correlations with narrow intervals entirely above zero indicate general trending agreement with the human reference that is stable under resampling of the collected MLLM responses.

To translate explanation information into prompt format for the MLLMs, we took a variety of approaches depending on the study. The reliance calibration and knowledge extraction studies used a pre-test phase in which participants were taught about the domain, so we included examples in the prompt to simulate this. The model improvement study used a GUI and repeated interactions, so we included an image of the data distribution shown to users in the real study to simulate this cognitive load, and accumulated all past interactions in the prompt summarised for each future interaction. Feature attribution information was written as text with two floating point numbers. Full details for all studies are in the appendix.

![](images/523ba967fe157b9a70e74e539c391b8b714f5dd16d6e2e7bf398381302d8656a.jpg)

![](images/513c2d4eea4f81dccc80e64ce1705493f67c918ea48ce29446c66f5fa3fd3d24.jpg)

![](images/69c16d0d766df24d431539f14b08c7109e01deb9b2dce8ea31116462b6e9ec7a.jpg)  
Figure 2: Human references and GPT-5 results. (A1) Model-improvement panel from Bhattacharya et al. [4]. (A2) GPT-5 relative accuracy improvement over each run’s baseline, using the best successful proposal among five attempts; boxes summarize 100 runs per condition. (B) GPT-5 synthetic teaching accuracy across 10,000 paired bootstrap resamples of 30 questions, with human reference means from Warren et al. [39] and Goyal et al. [14]. (C) Adult classification accuracy from Chen et al. [6] and GPT-5. Human bars show the published means and standard errors; GPT-5 bars show means and standard errors across 26 dataset-seed blocks.

## 4.1 Model Improvement

This area concerns improving model properties such as accuracy, fairness, safety, robustness, and alignment [15, 40], with researchers free to define the metric of improvement. Bhattacharya et al. [4] evaluated how explanation type supports model accuracy as the final metric. In a study with 70 healthcare professionals, participants used the EXMOS interactive system with one of three explanation dashboards: a data-centric version (DCE) visualising dataset characteristics and quality issues, a model-centric version (MCE) showing SHAP-based feature importance and surrogate decision rules, and a hybrid (HYB) combining both. Participants modified the training data based on explanations and retrained the model; HYB significantly outperformed DCE and MCE in postretraining accuracy (Figure 2(A1)). In our framework, H are physicians, AI is a random forest classifier, $\xi _ { 1 } , \xi _ { 2 } , \xi _ { 3 }$ are DCE, MCE, HYB, and $\phi ( \cdot )$ is the post-retraining accuracy delta.

Each MLLM was given five iterative attempts to improve accuracy, receiving a screenshot of the data distribution (Figure 10) together with feature-importance scores, decision rules, or dataset statistics according to its assigned condition. Each iteration contained information on all prior attempts, and we recorded the best configuration’s delta. We repeated this over 100 seeds with different data splits, yielding a per-condition mean delta-accuracy and an empirical condition ranking for each MLLM. To test whether this ranking matches the human ground truth (HYB > DCE > MCE) we computed Spearman’s $\rho .$ Precision was quantified by 95% confidence intervals from bootstrap resamples.

Table 1 shows five bootstrap intervals entirely above zero. Agreement nevertheless remains uncertain for several models, as discussed in Section 3, rankings can be fragile when differences between human condition means are small or explanation types overlap, which is the case for the original human study. In addition, our post-hoc analysis (Appendix C) identifies difficulties interpreting supplied information and using iterative feedback, alongside coarse feedback-set score resolution. These findings highlight challenges in this task, however 12/14 models did trend positively.

## 4.2 Knowledge Extraction

This area concerns the ability of explanations to teach humans new knowledge, either previously known [14] or entirely novel scientific insights [33]. We focus on Warren et al. [39], who compared counterfactual, causal, and control explanations for teaching humans about blood alcohol content. Participants underwent a training phase using one of the three explanation types before a testing phase requiring predictions on new data. The counterfactual group achieved the best accuracy (63.6%), followed by causal (61.4%) and control [59.0%; Figure 2(B)], converging with prior findings [14]. In our framework, H are non-experts, AI is the Widmark equation, and $\xi _ { 1 } , \xi _ { 2 } , \xi _ { 3 }$ are control, causal, and counterfactual; $\phi ( \cdot )$ is accuracy on test data.

![](images/d119c0438ec6693e3fa28e039097e28b0fed1231843c58b87f2a47f953dac758.jpg)

![](images/511d115fb50a54a06bcfad22f86bbecb4dc893becbc7dd750f4ac13f299155d2.jpg)

![](images/ba4c4f1681c60201831a891449fe8f5694d88d33af2abcc5f156f2d1a1cca914.jpg)  
Figure 3: Human references and GPT-5 results. (A) Original willingness-to-act panel from Tominaga et al. [35]. (B) GPT-5’s equivalent ratings versus $\mathsf { \bar { L } } _ { 1 }$ distance for 200 recourse plans sharing 39 applicants. The curve is a fitted population-mean GAMM trend, with an approximate pointwise 95% mean interval. (C) Mean viewing-intention ratings for six explanation conditions from Herlocker et al. [16]. Pearson’s $r = 0 . 9 0$ is computed across the six condition means. The dashed line is a descriptive linear fit.

Because the Widmark equation is widely documented and likely present in MLLM pre-training data [41], we constructed a synthetic dataset matching the original’s feature count, categorical structure, and value ranges, but with anonymised feature names $( \mathbf { e . g . , x 1 , x 2 , . . . } )$ and random causal relationships to increase task difficulty (Appendix A.3). All conditions saw the same training examples and labels; the counterfactual group received the mutations needed to flip each prediction, and the causal group additionally received feature-importance scores derived as the same feature used in the counterfactual perturbation. Each MLLM answered 300 test questions per condition.

Table 1 shows positive teaching correlations for 13 of the 14 models, with 11 bootstrap intervals entirely above zero. Eight models reproduce the human reference ordering exactly in their point estimates $( \rho = 1 )$ . Thus, most models show stable positive agreement in this synthetic adaptation, although uncertainty remains model-dependent.

## 4.3 Reliance Calibration

This pillar concerns whether users appropriately rely on an AI system to improve some aspect of performance. We focus on Chen et al. [6], who measured whether explanations could calibrate reliance to improve classification performance, testing three conditions (control, saliency, and example-based) across the same dataset in tabular [3] and NLP [24] domains. Example-based explanations were the only condition to improve overall performance in either study. In our framework, H are non-experts, AI is a random forest (tabular) or bag-of-words classifier (NLP), ξ<sub>1</sub>, ξ<sub>2</sub>, ξ<sub>3</sub> are no-AI, saliency, and example-based, and $\phi ( \cdot )$ is accuracy on novel test instances.

We adapted the Adult and BIOS tasks using reconstructed study materials. Each MLLM received five fixed labelled demonstrations and answered 16 cases in each of 26 dataset-seed blocks under three matched conditions: no AI advice, AI advice with saliency [32], and AI advice with two nearestneighbour examples. We compare full-data condition means with the human reference orderings: nearest neighbours > control > saliency for Adult, and nearest neighbours > saliency > control for BIOS. The latter showed high unaided MLLM accuracy, suggesting limited headroom and a mismatch with the human reliance setting. We therefore report Adult as the primary reliance analysis and retain BIOS in Appendix A for transparency.

All 14 models have positive Adult point estimates, with eight intervals entirely positive. Frontier APIs such as GPT-5 mini, Claude Opus 4.8, and Claude Sonnet 4.6 perform best in general, but other open source models such as Mistral Small 3.2 also do equally well.

## 4.4 Actionable Recommendation

Actionable recommendation is dominated by research in algorithmic recourse and finance [19, 21], with extensions to healthcare [25]. Two aspects matter: (1) whether users follow suggested recourses, and (2) whether the recourse improves their situation if followed, with consideration to e.g. causality [20], robustness [10], and diversity [28]. As our focus is human-AI interaction, we target the former. We replicate Tominaga et al. [35], who placed a Japanese cohort in a personalised loan-rejection scenario in which each participant rated two sparse and three random recourses. Their main finding was that $L _ { p } { \mathrm { - n o r m } }$ cost moderately correlates with willingness to act (Figure 3(A)). In our framework, H are non-experts, AI is a rule-based classifier, $\{ \xi _ { i } \} _ { i = 1 } ^ { N }$ are recourses categorized by $L _ { p }$ distance, and $\phi ( \cdot )$ is a $1 - 7$ Likert willingness rating; the “Benefit” term in Equation 5 defaults to a constant since only propensity to act is measured.

Table 1: Full-data Spearman’s $\rho$ estimates with 95% percentile intervals from 10,000 bootstrap resamples at each dataset’s observed sample size. Bold entries have a nonnegative lower interval bound (including zero). Overall weights five studies equally. Recourse uses negative $L _ { 1 }$ distance versus willingness and conditions on the observed applicant pool. Intervals condition on fixed human references.
<table><tr><td>Model</td><td>Teaching</td><td>Reliance</td><td>Engagement</td><td>Model improvement</td><td>Recourse</td><td>Overall</td></tr><tr><td>GPT-5 mini</td><td>1.00 [0.50, 1.00]</td><td>1.00 [0.50, 1.00]</td><td>0.90 [0.84, 0.90]</td><td>1.00 [0.50, 1.00]</td><td>0.50 [0.37, 0.61]</td><td>0.88 [0.66, 0.90]</td></tr><tr><td>Qwen3.5 122B</td><td>1.00 [0.50, 1.00]</td><td>0.87 [-0.50, 1.00]</td><td>0.81 [0.81, 0.81]</td><td>1.00 [-1.00, 1.00]</td><td>0.50 [0.38, 0.60]</td><td>0.83 [0.26, 0.87]</td></tr><tr><td>Člaude 4.5 Sonnet</td><td>1.00 [0.00, 1.00]</td><td>0.50 [-0.50, 1.00]</td><td>0.75 [0.75, 0.81]</td><td>1.00 [-1.00, 1.00]</td><td>0.52 [0.41, 0.62]</td><td>0.76 [0.28, 0.86]</td></tr><tr><td>GPT-5.5</td><td>1.00 [0.50, 1.00]</td><td>0.50 [-0.50, 1.00]</td><td>0.84 [0.78, 0.84]</td><td>0.50 [-1.00, 1.00]</td><td>0.73 [0.64, 0.79]</td><td>0.71 [0.41, 0.91]</td></tr><tr><td>GPT-5</td><td>1.00 [1.00, 1.00]</td><td>0.50 [0.50, 1.00]</td><td>0.81 [0.75, 0.90]</td><td>0.50 [0.50, 1.00]</td><td>0.74 [0.65, 0.80]</td><td>0.71 [0.69, 0.92]</td></tr><tr><td>GPT-4o mini</td><td>0.50 [-0.50, 0.50]</td><td>0.50 [0.50, 1.00]</td><td>0.87 [0.72, 0.93]</td><td>1.00 [0.50, 1.00]</td><td>0.29 [0.14, 0.43]</td><td>0.63 [0.40, 0.75]</td></tr><tr><td>GPT-40</td><td>1.00 [1.00, 1.00]</td><td>0.50 [0.50, 0.50]</td><td>0.64 [0.55, 0.64]</td><td>0.50 [-0.50, 1.00]</td><td>0.48 [0.37, 0.58]</td><td>0.62 [0.40, 0.73]</td></tr><tr><td>Qwen3-VL 32B</td><td>0.50 [0.50, 1.00]</td><td>0.50 [-0.50, 1.00]</td><td>0.81 [0.64, 0.81]</td><td>1.00 [0.50, 1.00]</td><td>0.30 [0.17, 0.41]</td><td>0.62 [0.32, 0.74]</td></tr><tr><td>o4-mini</td><td>0.50 [0.50, 1.00]</td><td>0.50 [0.50, 1.00]</td><td>0.90 [0.84, 0.90]</td><td>1.00 [0.50, 1.00]</td><td>0.15 [0.001, 0.30]</td><td>0.61 [0.50, 0.81]</td></tr><tr><td>Claude 4.8 Opus</td><td>1.00 [1.00, 1.00]</td><td>1.00 [0.50, 1.00]</td><td>0.79 [0.75, 0.81]</td><td>-0.50 [-1.00, 0.51]</td><td>0.74 [0.66, 0.81]</td><td>0.61 [0.40, 0.82]</td></tr><tr><td>Claude 4.6 Sonnet</td><td>1.00 [0.50, 1.00]</td><td>1.00 [0.50, 1.00]</td><td>0.75 [0.75, 0.81]</td><td>-0.50 [-1.00, 0.50]</td><td>0.76 [0.68, 0.82]</td><td>0.60 [0.41, 0.82]</td></tr><tr><td>Gemma 431B</td><td>0.50 [0.50, 0.50]</td><td>0.50 [-1.00, 1.00]</td><td>0.75 [0.75, 0.75]</td><td>0.50 [-0.50, 0.50]</td><td>0.75 [0.68, 0.81]</td><td>0.60 [0.30, 0.70]</td></tr><tr><td>Mistral Small 3.2 Gemma 3 27B</td><td>0.50 [0.50, 0.87]</td><td>1.00 [0.50, 1.00]</td><td>0.41 [-0.14, 0.49]</td><td>0.50 [-0.50, 0.50]</td><td>0.30 [0.16, 0.42]</td><td>0.54 [0.19, 0.57]</td></tr><tr><td></td><td>-0.50 [-1.00, 1.00]</td><td>0.50 [-1.00, 1.00]</td><td>0.67 [0.67, 0.67]</td><td>0.50 [-0.50, 1.00]</td><td>0.39 [0.27, 0.50]</td><td>0.31 [-0.09, 0.71]</td></tr><tr><td>Model average</td><td>0.71 [0.57, 0.82]</td><td>0.67 [0.44, 0.80]</td><td>0.76 [0.71, 0.77]</td><td>0.57 [0.21, 0.64]</td><td>0.51 [0.48, 0.54]</td><td>0.65 [0.54, 0.67]</td></tr></table>

We adapted the loan-rejection task using German Credit data because the original study materials were unavailable. Due to inconsistencies in datasets, we omit the $L _ { 0 }$ analysis from our main results, but include them in Figure 14 for transparency. We generated 200 distinct recourse plans for 39 rejected applicants, retaining only plans that changed rejection to acceptance under the same fitted random forest. Each MLLM was asked to rate its willingness to carry out each plan on a 1–7 scale. The main analysis correlates $L _ { 1 }$ distance with willingness, using bootstrap resampling conditional on the observed applicant pool.

Table 1 shows positive negative- $. L _ { 1 } .$ –willingness correlations for all 14 models, with point estimates ranging from $\rho = 0 . 1 5$ to 0.76 and all intervals entirely above zero. Thus, higher-cost plans generally receive lower willingness ratings. Figures 14 and 15 show fitted willingness curves, accounting for repeated plans from the same applicant.

## 4.5 System Engagement

Explanations have long been used to drive user engagement with AI, with the most prominent example being recommender systems [27, 13]. We focus on Herlocker et al. [16], who tested 21 explanation types for convincing users to act on a MovieLens recommendation, asking participants how likely they were to see the film on a 1–7 Likert scale. Nearest-neighbour explanations performed best, feature-based explanations (actor, awards) slightly worse, and statistical explanations worst overall. In our framework, H are non-experts, AI is a collaborative filtering recommender, $\{ \xi _ { i } \} _ { i = 1 } ^ { 2 1 }$ are the explanation types, and $\phi ( \cdot )$ is the Likert rating.

We adapted six explanation types documented by Herlocker et al. [16], including a no-explanation control, and substituted a fixed placeholder movie title. Each model generated $^ { 7 7 }$ response blocks, with one 1–7 likelihood rating per condition in each block, yielding 462 responses. We retained complete six-condition blocks for analysis and compared their condition means with the published human means, preserving the human tie between two conditions.

Engagement has the highest average correlation across models in our benchmark (Table 1). All 14 models have positive point estimates, ranging from $\rho = 0 . 4 1$ to 0.90, and 13 have intervals entirely above zero. GPT-5 mini and o4-mini share the highest point estimate $( \rho = 0 . 9 0$ , 95% CI [0.84, 0.90]). Figure 3(C) illustrates the relationship between human and MLLM mean ratings; the full model breakdown appears in Figure 16. These ratings measure stated viewing intention rather than observed engagement as these are shown to correlate [34].

## 5 Sampling stability

An important consideration for practitioners of our framework is what MLLM sample size is necessary to achieve reliable rankings for explanation methods. Figure 4 addresses this by examining how sample size affects the stability of the MLLM rankings observed in our main tests. For each model and study, we generated 10,000 bootstrap resamples using our existing data at sizes 1-300 and measured the fraction of pairwise condition orderings preserved relative to the fulldata MLLM ranking. We averaged these scores across all studies, with recourse instead measuring preservation of the sign of the correlation between negative $L _ { 1 }$ distance and willingness to act.

Average recovery across models increased from 93.4% at n = 100 to 96.7% at n = 300. These results suggest that 100 independent evaluation units provide a practical starting budget for recovering most comparison directions. However, some models such as Claude 4.5 Sonnet and Gemma 3 27B performed relatively poorly, and require many more samples for reliable rankings. In contrast, the fastest converging model was Gemma 4 31B.

![](images/f97df227a490bcc9927699b151345f1d81dfa18876adb6d8f99e2731d19c9641.jpg)  
Figure 4: Pairwise ranking stability by model, averaged across five studies. Curves show recovery of observed comparison directions under bootstrap resampling; the dotted line marks 95% recovery.

## 6 Related Work

There has been increasing investigation into whether LLMs can replicate human user study responses [9, 44, 7]. Cui et al. [7] showed that MLLMs can replicate the rankings of methods across large-scale experiments, but they were restricted to a single model (GPT-4) and psychological studies. De Bona et al. [8] provided the first proof-of-concept that LLMs could replicate conclusions from an XAI user study, operationalising alignment as ANOVA effect replication in a single 2×2 study. Three aspects limit the generalisability of their findings however; they evaluate a single study and domain, their alignment metric (effect replication) does not directly address the question of ranking multiple XAI candidates, which is what practitioners need, and they use models that—as they themselves note—are not representative of current frontier capabilities. We extend this line of work by evaluating multiple candidate explanations across five purpose-grounded studies and 14 MLLMs, including nine API and five open-weight models. We report rank correlations alongside bootstrap intervals and examine how agreement with human study findings varies by model and task. Our conceptual analysis provides sufficient conditions for rank preservation in terms of condition-specific calibration error relative to the human mean difference (Section 3), and we provide guidance on the sample size needed by varying model families in Figure 4.

## 7 Discussion

Here we discuss potential caveats and a practical workflow for our MLLM human surrogate evaluation framework before concluding. First, MLLMs are black-box evaluators, which limits interpretation of their responses. Our results and prior work [7, 8] indicate that they can reproduce some comparative human-study outcomes, with agreement depending on the model and task. Returned reasoning text supports behavioral inspection of how a model describes the supplied explanations and feedback, as illustrated by our post-hoc audit on the model improvement study (see Appendix C). However, it does not provide verified access to internal computation or establish that humans and models reach their conclusions through similar processes. Investigating those processes would require dedicated mental-model elicitation [18] and mechanistic analyses [26]. As an initial step, our recourse-rationale analysis (see Appendix F) provides preliminary evidence of mental model alignment between MLLM chain-of-thought reasoning and human written thought processes.

Another valid concern is that public datasets may appear in MLLM pre-training data, conflating results with memorization. However, eliciting such memorization is non-trivial [5], and in our framework this is only a meaningful concern for reliance calibration and knowledge extraction, where predictions on test data are required. The latter can be addressed through novel datasets, as we demonstrated. For the former, as our framework only modifies the explanation and all other confounders are controlled between conditions, it is less of an issue. To elaborate, even if the MLLM’s base accuracy is artificially inflated by memorization, this would (in theory) hold across all conditions, leaving the relative effect of the explanation intact. However, this would only be true if sufficient headroom remains; evidence of this is seen in Figure 13 with the NLP reliance study. Appendix B provides additional evidence that memorization was unlikely a significant confounder.

Lastly, our benchmark includes nine API models from two model families which may receive updates that cause our results to drift. However these drifts are largely seen to occur in waves before returning to baseline [36]. Nevertheless, to help address this, we ran our results with five open-weight models which will never drift, provide raw data and chain-of-thought responses from all our MLLMs, and a reproducible code base. Broader model coverage remains valuable however.

A practical workflow for our framework would likely benefit from the following approach:

1. Define comparison. Identify which purpose-grounded area and associated metric your application best fits in Figure 1. Then, design prompts that do a controlled comparison between candidate explanations and a no-explanation control (either within or between subjects design).

2. Check suitability. The current evidence suggests that structurally distinct explanation types which give the largest differences in human studies give the most reliable rankings in our framework (see Appendix G). As such, any measures to estimate the effect sizes in user testing would be warranted to help interpret the final results and plan evaluations. In addition, text classification tasks seem vulnerable to a type of ceiling effect due to the ability of MLLMs to perform well at the task with no assistance and should be approached with additional caution.

3. Model selection. The evidence from Table 1 provides a starting point for model selection. GPT models and larger open-weight Qwen variants are reasonable starting candidates, subject to computational budget. However, suitability varies by task, and parameter count alone does not establish performance. Where feasible, it is likely best to assess sensitivity across different model families.

4. Sample size and uncertainty. Continue sampling until bootstrap confidence intervals resolve the comparisons of interest, supporting either a meaningful directional difference or practical equivalence within a prespecified tolerance. Account for repeated checks and multiple comparisons when constructing these intervals. Our convergence analysis suggests reasonable starting budgets within our datasets (see Figure 4). This provides a simple starting heuristic rather than a validated sample-size requirement for new studies.

5. Use the evidence proportionately. Complement existing metrics with these comparisons and optionally pilot shortlisted methods with humans based on the results. Averaging results over several prompt variations would be beneficial for robustness.

To conclude, our results suggest that MLLM-based evaluation is a meaningful complement for XAI evaluation, as all models trended positively overall and most converge quickly (see Table 1 and Figure 4). The ranking phenomenon is robust enough to lend practical credibility, yet fragile enough in specific conditions to warrant caution, as discussed in Section 3 and demonstrated during the model improvement and NLP reliance studies. We expect our results not to replace user testing, but instead to complement existing evaluation methodologies and help move the field away from relying heavily on proxy metrics which struggle to evaluate an explanation’s true utility.

## DISCLAIMER

This paper was prepared for informational purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co., and its affiliates (“JP Morgan”) and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation, and warranty whatsoever, and disclaims all liability, for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation, offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful. © 2026 JP Morgan Chase & Co. All rights reserved.

## References

[1] Mirko Bagaric, Jennifer Svilar, Melissa Bull, Dan Hunter, and Nigel Stobbs. The solution to the pervasive bias and discrimination in the criminal justice system: transparent and fair artificial intelligence. Am. Crim. L. Rev., 59:95, 2022.

[2] Alina Jade Barnett, Zhicheng Guo, Jin Jing, Wendong Ge, Peter W Kaplan, Wan Yee Kong, Ioannis Karakis, Aline Herlopian, Lakshman Arcot Jayagopal, Olga Taraschenko, et al. Improving clinician performance in classifying eeg patterns on the ictal–interictal injury continuum using interpretable machine learning. NEJM AI, 1(6):AIoa2300331, 2024.

[3] Barry Becker and Ronny Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5XW20.

[4] Aditya Bhattacharya, Simone Stumpf, Lucija Gosak, Gregor Stiglic, and Katrien Verbert. Exmos: Explanatory model steering through multifaceted explanations and data configurations. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems, pages 1–27, 2024.

[5] Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations, 2022.

[6] Valerie Chen, Q Vera Liao, Jennifer Wortman Vaughan, and Gagan Bansal. Understanding the role of human intuition on reliance in human-ai decision-making with explanations. Proceedings ofthe ACM on Human-computer Interaction, 7(CSCW2):1–32, 2023.

[7] Ziyan Cui, Ning Li, and Huaikang Zhou. Can ai replace human subjects? a large-scale replication of psychological experiments with llms. A Large-Scale Replication ofPsychological Experiments with LLMs (August 25, 2024), 2024.

[8] Francesco Bombassei De Bona, Gabriele Dominici, Tim Miller, Marc Langheinrich, and Martin Gjoreski. Evaluating explanations through llms: Beyond traditional user studies. In GenAI for Health: Potential, Trust and Policy Compliance, 2025.

[9] Danica Dillion, Niket Tandon, Yuling Gu, and Kurt Gray. Can ai language models replace human participants? Trends in Cognitive Sciences, 27(7):597–600, 2023.

[10] Ricardo Dominguez-Olmedo, Amir H Karimi, and Bernhard Schölkopf. On the adversarial robustness of causal algorithmic recourse. In International Conference on Machine Learning, pages 5324–5342. PMLR, 2022.

[11] Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608, 2017.

[12] Timo Freiesleben and Gunnar König. Dear xai community, we need to talk! fundamental misconceptions in current xai research. In World conference on explainable artificial intelligence, pages 48–65. Springer, 2023.

[13] Fatih Gedikli, Dietmar Jannach, and Mouzhi Ge. How should i explain? a comparison of different explanation types for recommender systems. International Journal of Human-Computer Studies, 72(4):367–382, 2014.

[14] Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh, and Stefan Lee. Counterfactual visual explanations. In International Conference on Machine Learning, pages 2376–2384. PMLR, 2019.

[15] Avani Gupta and PJ Narayanan. A survey on concept-based approaches for model improvement. arXiv preprint arXiv:2403.14566, 2024.

[16] Jonathan L Herlocker, Joseph A Konstan, and John Riedl. Explaining collaborative filtering recommendations. In Proceedings of the 2000 ACM conference on Computer supported cooperative work, pages 241–250, 2000.

[17] Robert R Hoffman, Shane T Mueller, Gary Klein, and Jordan Litman. Metrics for explainable ai: Challenges and prospects. arXiv preprint arXiv:1812.04608, 2018.

[18] Robert R Hoffman, Shane T Mueller, Gary Klein, and Jordan Litman. Measures for explainable ai: Explanation goodness, user satisfaction, mental models, curiosity, trust, and human-ai performance. Frontiers in Computer Science, 5:1096257, 2023.

[19] Amir-Hossein Karimi, Gilles Barthe, Bernhard Schölkopf, and Isabel Valera. A survey of algorithmic recourse: contrastive explanations and consequential recommendations. ACM Computing Surveys, 55(5).

[20] Amir-Hossein Karimi, Bernhard Schölkopf, and Isabel Valera. Algorithmic recourse: from counterfactual explanations to interventions. In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pages 353–362, 2021.

[21] Mark T Keane, Eoin M Kenny, Eoin Delaney, and Barry Smyth. If only we had better counterfactual explanations: Five key deficits to rectify in the evaluation of counterfactual xai techniques. arXiv preprint arXiv:2103.01035, 2021.

[22] Eoin Kenny and Weipeng Huang. The utility of “even if” semifactual explanation to optimise positive outcomes. Advances in Neural Information Processing Systems, 36:52907–52935, 2023.

[23] Eoin M Kenny, Akshay Dharmavaram, Sang Uk Lee, Tung Phan-Minh, Shreyas Rajesh, Yunqing Hu, Laura Major, Momchil S Tomov, and Julie A Shah. Explainable deep learning improves human mental models of self-driving cars. Nature, 657(8130):114–120, 2026.

[24] Laboratoire Hubert Curien (LabHC). bias\_in\_bios. https://huggingface.co/datasets/ LabHC/bias\_in\_bios, 2021. Accessed: 2025-05-03.

[25] Anisio Lacerda, Claudio Almeida, Leonardo Ferreira, Adriano Pereira, Gisele L Pappa, Wagner Meira, Debora Miranda, Marco A Romano-Silva, and Leandro Malloy Diniz. Algorithmic recourse in mental healthcare. In 2023 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2023.

[26] Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, et al. Auditing language models for hidden objectives. arXiv preprint arXiv:2503.10965, 2025.

[27] David McSherry. Explanation in recommender systems. Artificial Intelligence Review, 24:179– 197, 2005.

[28] Ramaravind K Mothilal, Amit Sharma, and Chenhao Tan. Explaining machine learning classifiers through diverse counterfactual explanations. In Proceedings ofthe 2020 conference on fairness, accountability, and transparency, pages 607–617, 2020.

[29] J Munro. A review of:“rule-based expert systems: the mycin experiments of the stanford heuristic programming project” eds bruce b. buchanan and edward h. shortliffe addison-wesley, 1984, 748 pp,£ 37.00. Civil Engineering Systems, 1(6):342–343, 1984.

[30] David Piorkowski, Inge Vejsbjerg, Owen Cornec, Elizabeth M Daly, and Öznur Alkan. Aimee: An exploratory study of how rules support ai developers to explain and edit models. Proceedings ofthe ACM on Human-Computer Interaction, 7(CSCW2):1–25, 2023.

[31] Aaditya Ramdas and Arno Solin. Opening remarks. Presentation, 29th International Conference on Artificial Intelligence and Statistics (AISTATS), May 2026. Program Chairs’ address, AISTATS 2026, May 2–5, 2026.

[32] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135–1144, 2016.

[33] Lisa Schut, Nenad Tomasev, Tom McGrath, Demis Hassabis, Ulrich Paquet, and Been Kim. Bridging the human-ai knowledge gap: Concept discovery and transfer in alphazero. arXiv preprint arXiv:2310.16410, 2023.

[34] Paschal Sheeran. Intention—behavior relations: a conceptual and empirical review. European review ofsocial psychology, 12(1):1–36, 2002.

[35] Tomu Tominaga, Naomi Yamashita, and Takeshi Kurashima. Reassessing evaluation functions in algorithmic recourse: An empirical study from a human-centered perspective. arXiv preprint arXiv:2405.14264, 2024.

[36] Paul Tschisgale and Peter Wulff. Daily and weekly periodicity in large language model performance and its implications for research. arXiv preprint arXiv:2602.15889, 2026.

[37] Roel Visser, Tobias M Peters, Ingrid Scharlau, and Barbara Hammer. Trust, distrust, and appropriate reliance in (x) ai: a survey of empirical evaluation of user trust. CoRR, 2023.

[38] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: Automated decisions and the gdpr. Harv. JL & Tech., 31:841, 2017.

[39] Greta Warren, Ruth MJ Byrne, and Mark T Keane. Categorical and continuous features in counterfactual explanations of ai systems. ACM Transactions on Interactive Intelligent Systems, 14(4):1–37, 2024.

[40] Leander Weber, Sebastian Lapuschkin, Alexander Binder, and Wojciech Samek. Beyond explaining: Opportunities and challenges of xai-based model improvement. Information Fusion, 92:154–176, 2023.

[41] EMP Widmark. Principles and applications of medicolegal alcohol determination, 1932 (translated from original version) davis, 1981.

[42] Zhenxing Wu, Jihong Chen, Yitong Li, Yafeng Deng, Haitao Zhao, Chang-Yu Hsieh, and Tingjun Hou. From black boxes to actionable insights: a perspective on explainable artificial intelligence for scientific discovery. Journal ofChemical Information and Modeling, 2023.

[43] Yongfeng Zhang, Xu Chen, et al. Explainable recommendation: A survey and new perspectives. Foundations and Trends® in Information Retrieval, 14(1):1–101, 2020.

[44] Ruican Zhong, David W McDonald, and Gary Hsieh. Synthetic heuristic evaluation: A comparison between ai-and human-powered usability evaluation. arXiv preprint arXiv:2507.02306, 2025.

## A MLLM replication studies

## A.1 Models Used

<table><tr><td>Model</td><td>Snapshot, checkpoint or inference-profile identifier</td></tr><tr><td>API models</td><td></td></tr><tr><td>GPT-4o</td><td>gpt-4o-2024-11-20</td></tr><tr><td>GPT-4o mini</td><td>gpt-4o-mini-2024-07-18</td></tr><tr><td>o4-mini</td><td>o4-mini-2025-04-16</td></tr><tr><td>GPT-5</td><td>gpt-5-2025-08-07</td></tr><tr><td>GPT-5 mini</td><td>gpt-5-mini-2025-08-07</td></tr><tr><td>GPT-5.5</td><td>gpt-5.5-2026-04-24</td></tr><tr><td>Claude 4.5 Sonnet</td><td>anthropic.claude-sonnet-4-5-20250929-v1:0</td></tr><tr><td>Claude 4.6 Sonnet</td><td>NA</td></tr><tr><td>Claude 4.8 Opus</td><td>NA</td></tr><tr><td>Open-weight models</td><td></td></tr><tr><td>Gemma 3 27B</td><td>pytorch/gemma-3-27b-it-FP8</td></tr><tr><td>Gemma 4 31B</td><td>RedHatAI/gemma-4-31B-it-FP8-dynamic</td></tr><tr><td>Qwen3-VL 32B</td><td>Qwen/Qwen3-VL-32B-Instruct-FP8</td></tr><tr><td>Qwen3.5 122B</td><td>Qwen/Qwen3.5-122B-A10B-GPTQ-Int4</td></tr><tr><td>Mistral Small 3.2</td><td>stelterlab/Mistral-Smal1-3.2-24B-Instruct-2506-FP8</td></tr></table>

## A.2 Appropriate reliance

We adapted the income-prediction and biography-classification tasks of Chen et al. [6] using Adult Census and BIOS data. We reconstructed the selected cases (as the originals were not available), AI predictions and explanations from the available methodological information, including the target proportion of correctly classified cases. These are adaptations of the study materials rather than exact reproductions of every original participant’s task. The published mean orderings provide our human references.

## A.2.1 Adult Dataset Experiment

The Adult task uses nine displayed features: age, workclass, education, marital-status, occupation, native-country, hours-per-week, gender, and race. The target is whether income exceeds \$50,000; the source uses income as the target column. Categorical predictors are encoded for model fitting and displayed as readable category names in prompts.

The reconstructed classifier is a Random Forest with 100 estimators, maximum depth 10, and the dataset-block seed as its random state. Adult uses an 80/20 stratified train/test split. The case-selection procedure targets an experienced AI accuracy of 62.5%, rather than the classifier’s accuracy over its entire test bank. This was done to reflect the original study. Nearest-neighbour candidate search considers 50 training cases using Euclidean distance. Feature-based explanations use LIME and refer to the displayed AI prediction. The frozen materials correct the income-label mapping and present all five labelled demonstrations.

## A.2.2 BIOS Dataset Experiment

The BIOS task uses biography text from hard\_text, restricted to five professions: physician, professor, psychologist, surgeon, and teacher. We map the source profession codes to these labels and use the supplied training and test partitions. Text is represented by a CountVectorizer with English stop words and at most 5,000 features. The classifier is a Random Forest with 100 estimators, maximum depth 10, and the dataset-block seed as its random state. Nearest-neighbour candidate search uses Euclidean distance over the dense bag-of-words vectors and considers 50 training cases. LIME text explanations describe the displayed AI prediction.

The reconstruction targets 62.5% AI accuracy on the selected cases and approximately 63% agreement between the first neighbour’s true label and the test case’s true label. These are selection targets, not guaranteed rates in every exported cohort. The realised AI accuracy is 260/416 for GPT-4o, GPT-4o mini, GPT-5 mini and o4-mini, and 263/416 for the remaining models. The prompt labels and predictions use the same five-profession mapping.

## A.2.3 Experiment methodology

Each model is evaluated on 26 dataset-seed blocks per domain, with 16 cases in each block. Every case is presented under all three conditions, yielding 416 responses per condition and domain, with one response per case–condition pair. Each prompt includes five fixed labelled demonstrations; these are shared across conditions rather than resampled for every query.

1. Control: the case and demonstrations, with no AI prediction or explanation.

2. Saliency: the same case and demonstrations, the AI prediction, and a LIME explanation.

3. Nearest neighbours: the same case and demonstrations, the AI prediction, and two training cases with their true labels and AI predictions.

Consequently, comparisons with control measure the combined contribution of AI advice and explanation; they do not isolate explanation utility from advice alone. Conditions are matched within each model.

Accuracy is the proportion of responses with the correct class label; invalid answers count as incorrect. The primary table computes Spearman’s ρ from the full-data condition means and obtains percentile intervals from 10,000 matched block-bootstrap resamples at the observed 26-block sample size. Adult and BIOS are averaged within each reliance resample. The human reference orderings are nearest neighbours > control > saliency for Adult, and nearest neighbours > saliency > control for BIOS.

Table 2: Control prompt for appropriate reliance
<table><tr><td>Control prompt for appropriate reliance</td></tr><tr><td>Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a study to examine people&#x27;s reliance on AI systems with or</td></tr><tr><td>without explanations for the AI&#x27;s predictions.</td></tr><tr><td>The domain is Adult Census and the features in order are: {feature_names}</td></tr><tr><td>The possible labels are: {class_names} Here are 5 randomly sampled examples of training data to help</td></tr><tr><td>you understand the domain: {five_demonstration_instances},</td></tr><tr><td>and here are their labels: {five_demonstration_labels}</td></tr><tr><td>Your task is to look at a new test instance, and decide what you think the real label is.</td></tr><tr><td>Here is the new test instance: {row. test_instance}</td></tr></table>

The human reference orderings are based on reported condition means and do not imply that every pairwise difference was statistically significant. The descriptive accuracy plots do not establish replication of non-significant human comparisons.

## A.3 Human-machine teaching

Warren et al. [39] studied whether counterfactual and causal explanations help people learn a domain with a known outcome rule. We use a synthetic adaptation rather than the Widmark-based domain, whose widely documented formula may be familiar to an MLLM. Our domain has five anonymised features: two binary variables and three positive continuous variables.

Table 3: Saliency prompt for appropriate reliance  
Saliency prompt for appropriate reliance   
Pretend you are a participant in a user study with some   
technical machine learning knowledge being asked to assist in a   
study to examine people’s reliance on AI systems with or   
without explanations for the AI’s predictions.   
The domain is Adult Census and the features in order are: {feature\_names}   
The possible labels are: {class\_names}   
Here are 5 randomly sampled examples of training data to help   
you understand the domain: {five\_demonstration\_instances},   
and here are their labels: {five\_demonstration\_labels}   
Your task is to look at a new test instance, and decide what you think the real label is.   
Here is the new test instance: {row.test\_instance}   
Here is the AI prediction for the   
new test instance: {row.test\_predicted\_label}   
In addition, here is a feature saliency explanation for the AI prediction   
where you see each feature,   
and its contribution (positive or negative) towards the   
prediction: {round\_numbers\_in\_string(row.lime\_explanation)}   
Outline your reasoning process step by step, and give your   
final answer as one of the possible class labels in the tags   
<answer>...</answer>.

Let $X _ { 1 } , X _ { 2 } \sim$ Bernoulli(0.5) independently, and let $Z _ { 3 } , Z _ { 4 } , Z _ { 5 }$ be independent standard normal variables. The continuous features are generated sequentially as

$$
\begin{array} { r l } & { X _ { 3 } = \mathrm { r o u n d } _ { 2 } [ \mathrm { e x p } ( 0 . 5 + 0 . 8 X _ { 1 } + 0 . 3 Z _ { 3 } ) ] , } \\ & { X _ { 4 } = \mathrm { r o u n d } _ { 2 } [ \mathrm { e x p } ( 0 . 7 + X _ { 2 } + 0 . 3 Z _ { 4 } ) ] , } \\ & { X _ { 5 } = \mathrm { r o u n d } _ { 2 } [ \mathrm { e x p } \{ 0 . 1 \ln ( X _ { 3 } + \epsilon ) + 0 . 1 \ln ( X _ { 4 } + \epsilon ) + 0 . 3 Z _ { 5 } \} ] , } \end{array}
$$

where round rounds to two decimal places and $\epsilon = 1 0 ^ { - 9 }$ stabilises the logarithms. The outcome is $Y = 1$ when $S \geq 4 . 7 1$ , and $Y = 0$ otherwise, where

$$
\begin{array} { r l } & { S = X _ { 1 } + X _ { 2 } + 1 . 2 \ln ( X _ { 3 } + \epsilon ) + \ln ( X _ { 4 } + \epsilon ) + 1 . 5 X _ { 5 } } \\ & { \qquad + 0 . 1 X _ { 1 } \ln ( X _ { 3 } + \epsilon ) + 0 . 1 X _ { 2 } \ln ( X _ { 4 } + \epsilon ) . } \end{array}
$$

The graph contains $X _ { 1 }  X _ { 3 } , X _ { 2 }  X _ { 4 } , X _ { 3 }  X _ { 5 }$ , and $X _ { 4 } ~  ~ X _ { 5 }$ , with all five features contributing to Y (Figure 5). The graph and equations define the task but are not supplied directly in the teaching prompts.

The five demonstrations are fixed and shared across models and conditions; their counterfactuals are fixed and supplied only in the counterfactual condition. They include three negative and two positive examples, with one selected intervention for each feature. The negative example $( 1 , 0 , 2 . 6 3 , 2 . 6 9 , 0 . 8 0 )$ breaks the shortcut $Y = X _ { 1 }$ . The frozen original/counterfactual pairs, in feature order $( X _ { 1 } , \ldots , X _ { 5 } )$ , are

Table 4: Example-based prompt for appropriate reliance
<table><tr><td>Example-based prompt for appropriate reliance</td></tr><tr><td>Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a study to examine people&#x27;s reliance on AI systems with or without explanations for the AI&#x27;s predictions.</td></tr><tr><td>The domain is Adult Census and the features in order are: {feature_names}</td></tr><tr><td>The possible labels are: {class_names}</td></tr><tr><td>Here are 5 randomly sampled examples of training data to help you understand the domain: {five_demonstration_instances}, and here are their labels: {five_demonstration_labels}</td></tr><tr><td>Your task is to look at a new test instance, and decide what you think the real label is.</td></tr><tr><td>Here is the new test instance: {row. test_instance}</td></tr><tr><td>Here is the AI prediction for the new test instance: {row.test_predicted_label}</td></tr><tr><td>In addition, here is an explanation for the prediction in the form of the two nearest neighbors from the training data</td></tr><tr><td>that the AI learned from: The first nearest neighbor is {row.neighbor1_instance},</td></tr><tr><td>its label is &#x27;{row.neighbor1_true_label}&#x27;,</td></tr><tr><td>and the AI&#x27;s prediction of this nearest neighbor is &#x27;{row. neighbor1_predicted_label}&#x27; The second nearest neighbor is {row.neighbor2_instance},</td></tr><tr><td>its label is &#x27;{row.neighbor2_true_label}&#x27;,</td></tr><tr><td>and the AI&#x27;s prediction of this nearest neighbor is &#x27;{row. neighbor2_predicted_label}</td></tr></table>

<table><tr><td>Intervention</td><td>Original</td><td>Y</td><td>Counterfactual</td><td> $Y ^ { \prime }$ </td></tr><tr><td> $X _ { 1 }$ </td><td>(0, 1, 1.43, 5.79, 0.74)</td><td>0</td><td>(1, 1, 3.18, 5.79, 0.80)</td><td>1</td></tr><tr><td> $X _ { 2 }$ </td><td>(0, 0, 2.25, 1.71, 1.09)</td><td>0</td><td>(0, 1, 2.25, 4.65, 1.20)</td><td>1</td></tr><tr><td> $X _ { 3 }$ </td><td>(1, 0, 2.63, 2.69, 0.80)</td><td>0</td><td>(1, 0, 3.17, 2.69, 0.82)</td><td>1</td></tr><tr><td> $X _ { 4 }$ </td><td>(1, 1, 4.06, 7.86, 1.52)</td><td>1</td><td>(1, 1, 4.06, 0.46, 1.14)</td><td>0</td></tr><tr><td> $X _ { 5 }$ </td><td>(1, 0, 3.88, 2.00, 1.37)</td><td>1</td><td>(1, 0, 3.88, 2.00, 0.83)</td><td>0</td></tr></table>

Counterfactual construction intervenes on one selected feature, propagates the change to its descendants while holding the structural noise terms fixed, and recomputes Y . Binary interventions flip the selected value; continuous interventions search the 0.01 grid for the nearest selected-feature value that flips the label. Thus a single intervention can change several displayed features. The released runner uses the frozen pairs rather than regenerating them for each query.

The control receives the five original labelled instances. The counterfactual condition additionally receives the paired label-flipping instances. The causal condition instead adds the sentence: “In this specific example, part of the reason the label is [label] is because feature [feature] was [value].” The named feature is the corresponding intervention feature; no numerical feature-importance scores are provided.

The runner generates 10,000 source instances with seed 42 and an 80/20 stratified split. It generates a fresh pool of 20,000 instances with seed 20260912, removes feature-identical matches to the training set and demonstration counterfactuals, and samples 300 questions without replacement using seed 42. This is simple random test sampling with natural class balance, not a balanced test set. Each question is presented in all three conditions with one response per condition, for 900 responses per model. Every prompt ends with: “Explain your reasoning briefly, and give your final answer as 0 or 1 in the tags <answer>...</answer>.” The word ‘briefly’, was chosen to avoid models giving overly verbose responses, especially for the counterfactual condition which frequently reached token limits due to the complexity. Accuracy includes invalid answers as incorrect. The primary table uses full-data condition means and 10,000 bootstrap resamples of the 300 matched questions, with the fixed human ordering counterfactual > causal > control.

Table 5: Control prompt for human-machine teaching.
<table><tr><td>Pretend you are a participant in a user study designed to (1) teach you about a complex causal domain, and (2) get you to make a prediction on a novel test instance given what you learned. The data domain has 5 features and is structured as follows: - X1, X2: Binary variables (0 or 1) - X3, X4, X5: Continuous POSITIVE variables (representing quantities &gt; 0) - Y: Outcome (0 or 1) You are now going to learn about the underlying domain by studying the following information: Here</td></tr><tr><td>are [len(examples)] training examples, and their labels Y to help you understand the domain: [format- ted_examples]</td></tr><tr><td>Here is the new test instance: [test_instance] Now, given all this information, make an informed prediction on what Y is for this test instance. Explain</td></tr><tr><td>your reasoning briefly, and give your final answer as 0 or 1 in the tags &lt;answer&gt;...&lt;/answer&gt;.</td></tr></table>

Table 6: Causal prompt for human-machine teaching.

Pretend you are a participant in a user study designed to (1) teach you about a complex causal domain, and (2) get you to make a prediction on a novel test instance given what you learned.   
The data domain has 5 features and is structured as follows: - X1, X2: Binary variables (0 or 1) - X3, X4, X5: Continuous POSITIVE variables (representing quantities > 0) - Y: Outcome (0 or 1)   
You are now going to learn about the underlying causal graph by studying the following explanation: Here are [len(examples)] training examples, and a causal explanation for their classification, and their labels Y to help you learn the underlying causal graph, study this carefully: [formatted\_examples]   
Here is the new test instance: [test\_instance]   
Now, given all this information, and what you have learned from the explanation, make an informed prediction on what Y is for this test instance. Explain your reasoning briefly, and give your final answer as 0 or 1 in the tags <answer>...</answer>.

Table 7: Counterfactual prompt for human-machine teaching.
<table><tr><td>Pretend you are a participant in a user study designed to (1) teach you about a complex causal domain, and (2) get you to make a prediction on a novel test instance given what you learned.</td><td></td><td></td><td></td></tr><tr><td>The data domain has 5 features and is structured as follows: - X1, X2: Binary variables (0 or 1) - X3, X4, X5: Continuous POSITIVE variables (representing quantities &gt; 0) - Y: Outcome (0 or 1) You are now going to learn about the underlying causal graph by studying the following explanation:</td><td></td><td></td><td></td></tr><tr><td>Here are [len(examples)] training examples, counterfactual versions of them where Y changes to the</td><td></td><td></td><td></td></tr><tr><td>opposite class, and their labels Y to help you learn the underlying causal graph, study this carefully:</td><td></td><td></td><td></td></tr><tr><td>[formatted_examples]</td><td></td><td></td><td></td></tr><tr><td>Here is the new test instance: [test_instance]</td><td></td><td></td><td></td></tr><tr><td>Now, given all this information, and what you have learned from the explanation, make an informed</td><td></td><td></td><td></td></tr><tr><td>prediction on what Y is for this test instance. Explain your reasoning briefly, and give your final answer as</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>0 or 1 in the tags &lt;answer&gt;...&lt;/answer&gt;.</td><td></td><td></td><td></td></tr></table>

## A.4 System engagement

We adapted six recommendation explanations from Herlocker et al. [16]: explanation IDs 1, 2, 4, 5, 11 and 21. We chose these six because the original materials have been lost and the authors could not supply us with them, so we used all explanations which were detailed in the published paper only. The prompt presents a user with \$7 and a free evening and asks how likely they are to see a recommended movie on a 1–7 scale. The movie title is replaced by a fixed placeholder to reduce title-specific preferences; the recommendation explanations are supplied as text. Explanation 11 is the no-explanation control. Explanation 4 is described as a histogram in our prompts, whereas the original study presented a table of neighbour ratings; this is a presentation adaptation.

We collected 77 response blocks per model, with one response to each of the six conditions in each block: 462 responses per model. These are repeated model samples, not 77 independent human participants. We parse an explicit 1–7 answer and retain complete six-condition blocks for the comparative analysis. The published human reference means, in explanation-ID order, are 5.25, 5.19, 4.97, 4.97, 4.53 and 3.94, preserving the tie between explanations 4 and 5. The descriptive plots report Pearson’s r between the six human and model condition means; the primary table instead reports full-data Spearman’s ρ with 10,000 matched response-block bootstrap resamples at the observed complete-block count. The six condition means are the correlation observations, not the number of response blocks.

Table 8: Prompt for System Engagement
<table><tr><td>Prompt for System Engagement</td></tr><tr><td>Pretend you are a human user of a movie recommender system.</td></tr><tr><td>Imagine that you have $7 and a free evening coming up. You are considering going to the theater to see a movie, but only if there is a movie</td></tr><tr><td>worth seeing. To determine if there is a movie worth seeing, you consult MovieLens for a</td></tr><tr><td>personalized movie recommendation. MovieLens recommends one movie, and provides an explanation</td></tr><tr><td>to justify the recommendation.</td></tr><tr><td></td></tr><tr><td>The movie recommended was: {movie_title}</td></tr><tr><td>Here is its explanation for the recommendation: {explanation}</td></tr><tr><td>The movies are rated on a scale of 1-5.</td></tr><tr><td>Based on the provided explanation, please indicate on a scale of 1-7 how likely you are to see the movie.</td></tr><tr><td></td></tr><tr><td>First, outline your reasoning step by step, and put your answer in between the tags &lt;answer&gt;. . . &lt;/answer&gt;</td></tr></table>

## A.5 Actionable Recourse Experimental Setup

This experiment asks MLLMs to assess recourse plans generated from the German Credit dataset. We preserve both requested ratings but analyse only actionability\_score, the stated willingness to carry out a plan.

Feature constraints and action rules. The mutable numeric features are duration, credit\_amount and existing\_credits; permitted edits reduce their integer values within trainingderived bounds. The mutable categorical features are status and savings, with adjacent codebook transitions:

• status: A11 → A12, or A12 → A13.

• savings: A61 → A62, A62 → A63, or A63 → A64.

All other fields are held fixed, including credit history, employment tenure, housing, age and personal status. Categories are rendered using the historical Statlog codebook rather than treated as ordinal numbers. These constraints define the allowed plans; they do not establish that every plan is feasible for every applicant.

Table 9: Prompt for System Engagement – No Explanation
<table><tr><td>Prompt for System Engagement – No Explanation</td></tr><tr><td>Pretend you are a human user of a movie recommender system.</td></tr><tr><td>Imagine that you have  $\$ 7$  and a free evening coming up. You are considering going to the theater to see a movie, but only if there is a movie</td></tr><tr><td>worth seeing. To determine if there is a movie</td></tr><tr><td>worth seeing, you consult MovieLens for a personalized movie recommendation. MovieLens</td></tr><tr><td>recommends one movie.</td></tr><tr><td></td></tr><tr><td>The movie recommended was: {movie_title}</td></tr><tr><td>Please indicate on a scale of 1-7</td></tr><tr><td>First, outline your reasoning step by step, and put your answer</td></tr></table>

![](images/ef7e4826b2e4956e9fe4c96334769280602fae976fe4919a3b899cc7ebccec5c.jpg)  
Figure 5: The five-feature causal graph used in our synthetic adaptation of Warren et al. [39]. All five features contribute to the outcome $\bar { Y }$

For an original profile x and revised profile $x ^ { \prime } ,$ distances are

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { 0 } ( x , x ^ { \prime } ) = \sum _ { j } { \bf 1 } [ x _ { j } \ne x _ { j } ^ { \prime } ] } , \ ~ } \\ { { \displaystyle { \cal L } _ { 1 } ( x , x ^ { \prime } ) = \sum _ { j \in \mathcal { N } } \frac { | x _ { j } ^ { \prime } - x _ { j } | } { \sigma _ { j } } + \sum _ { j \in \mathcal { C } } { \bf 1 } [ x _ { j } \ne x _ { j } ^ { \prime } ] } , } \end{array}
$$

where $\mathcal { N }$ and $\mathcal { C }$ denote numeric and categorical fields, and $\sigma _ { j }$ is the numeric feature’s training-set standard deviation. $L _ { 0 }$ counts changes to original fields rather than one-hot columns.

Action generation and selection. The generator uses a 70/30 stratified train/test split with seed $^ { 4 2 , }$ one-hot categorical encoding, and a Random Forest with 100 estimators, maximum depth 12 and random state 42. It searches rejected test profiles over permitted categorical transitions and numeric reductions at predefined distance scales, retaining candidate plans with $L _ { 1 } \leq 1 3$ . Every retained plan must change rejection to acceptance under the same fitted classifier; no second classifier is used to enlarge the candidate pool. We select 200 distinct plans using round-robin allocation across $L _ { 0 }$ strata and occupied $L _ { 1 }$ bins, with finer bins near zero. The selected plans share 39 applicants and therefore are not 200 independent people. The rendered prompt states the actual loan purpose, duration and amount, using Deutsche Mark (DM) units from the source dataset.

Aggregate analysis. We plot willingness separately against $L _ { 0 }$ and $L _ { 1 }$ using Gaussian generalized additive mixed models fitted by REML in $\mathrm { \mathbf { R } } ^ { * } \mathrm { \mathbf { s } }$ mgcv package. The model includes a thin-plate smooth of distance, an applicant random intercept, and an applicant random slope:

$$
\mathrm {  ~ y ~ \tilde { ~ } { ~ s ~ } ~ } ( \mathrm {  ~ x ~ } , \mathrm { b } \mathrm { s } = ^ { \mathrm {  ~ u ~ } } \mathrm { t p } ^ { \mathrm {  ~ u ~ } } , \mathrm { k } = \mathrm { k } ) + \mathrm {  ~ s ~ } ( \mathrm { u } \mathrm { i } \mathrm { d } , \mathrm { b } \mathrm { s } = ^ { \mathrm {  ~ u ~ } } \mathrm { r } \mathrm { e } ^ { \mathrm {  ~ u ~ } } ) + \mathrm {  ~ s ~ } ( \mathrm {  ~ x ~ } , \mathrm { u } \mathrm { i } \mathrm { d } , \mathrm { b } \mathrm { s } = ^ { \mathrm {  ~ u ~ } } \mathrm { r } \mathrm { e } ^ { \mathrm {  ~ u ~ } } ) ,
$$

where $k = \mathrm { m i n } ( 1 0$ , number of distinct distance values). Only valid integer ratings from 1 to 7 enter a fit. Population curves exclude the applicant-specific effects; the bands are approximate pointwise 95% intervals for the population mean, calculated as fitted mean ±1.96 standard errors with smoothingparameter uncertainty included. They are not prediction intervals, and the fitted curves are not constrained to the rating range.

The main table reports Spearman’s correlation between negative $L _ { 1 }$ and willingness, with plan-level bootstrap resampling conditional on the observed applicant pool. This is a directional cost–willingness relationship, not direct agreement with matched human recourse ratings. Excluding $L _ { 0 }$ from the primary aggregate was a post-hoc decision: a small and a large numeric change count equally under $L _ { 0 } .$ , creating a mismatch with the original study’s feature representation. The $L _ { 0 }$ results are retained in Figure 14. The visual-comparison prompt below is retained as a historical inspection procedure and does not determine the primary correlation results.

Table 10: Prompt for rating recourse plans.  
Imagine you are a human applying for a loan. You are at a financial institution undergoing a loan assessment.   
An AI system determines approval or denial based on your profile data.   
The requested loan’s purpose, amount and duration are shown below. Monetary amounts are in Deutsche   
Mark (DM), as recorded in this historical dataset.   
YOUR CURRENT PROFILE: [profile with readable categories and units]   
An AI system has rejected your loan application. The following changes would make that same AI system   
accept your revised profile. All other profile details stay the same.   
AI Explanation: [numbered changes from original to proposed values]   
Your Task: Please answer the following two questions using an integer between 1–7 where 1=“Strongly   
no”, 7=“Strongly yes”, and 4=“Neutral”.   
1. Is the AI System’s plan a reasonable explanation for the rejection of your loan application?   
2. Would you carry out the plan presented by the AI system to obtain loan approval?   
Outline your reasoning and then give your scores in JSON like: {"reasoning": string,   
"acceptance\_score": integer, "actionability\_score": integer}   
Do not put anything except a single integer for the two scores.

## A.6 Model Improvement

We evaluate whether MLLMs can improve a diabetes classifier by proposing training-data configurations under data-centric (DCE), model-centric (MCE) and hybrid (HYB) conditions. Each run has five proposal attempts. The final release contains 100 seed blocks per model and condition, except Qwen3.5 122B, which has 40 due to computational constraints.

Dataset and baseline. The released Pima Indians Diabetes CSV has 768 rows, target Outcome, and eight numeric features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The released open-weight routine uses an unstratified 85/15 split with the run seed, producing 652 training rows and 116 feedback rows. The Random Forest uses 30 estimators, maximum depth 5, minimum split size 50, minimum leaf size 25, max\_features=sqrt, maximum 10 leaf nodes, and random\_state=42. The split changes across seeds; the classifier random state remains fixed. Conditions share the original partition within each seed.

## A.6.1 Optimization conditions: DCE, MCE, HYB

Each proposal specifies whether to include each feature and optional minimum/maximum values for filtering training rows. The configuration is applied afresh to the original training partition, not cumulatively to the previously filtered data. The classifier is retrained and evaluated on the same fixed feedback partition.

DCE supplies class balance, up to three highly correlated feature pairs $( | r | \ge 0 . 8 )$ , per-feature summaries, and data-quality scores for zeros, outliers, balance and correlation. The summaries include minimum, mean, maximum, fifth and 95th percentiles, zero frequency, IQR-based outliers, and a mean–median skewness classification. These summaries and observed feature ranges use the full source CSV and remain static across attempts. DCE and HYB receive the original static feature-distribution image; MCE does not receive it in the released open-weight routine.

MCE supplies the current best Random Forest’s impurity-based feature importance and rules from a surrogate Decision Tree trained on its predictions over the original training rows for the current features. The surrogate has maximum depth 4, minimum leaf size 20 and random state 42. Up to four leaf rules per class are shown, sorted by supporting sample count and confidence. HYB combines the DCE and MCE information.

## A.6.2 Prompt construction

Each prompt contains the baseline and current best accuracy, their difference, the available features, the best configuration when present, condition-specific signals, observed feature ranges, and the previous attempts’ configurations, scores and feedback. The source prompt’s displayed sample counts remain based on the full CSV rather than tracking filtered training-row counts. The five-attempt interaction is our adaptation of iterative user-driven model improvement. The prompt requests a reasoning section followed by a configuration JSON object (Table 11).

The primary endpoint takes the highest successfully evaluated accuracy among attempts 1–5, excluding baseline from the maximum. For baseline accuracy $a _ { 0 }$ and best proposal accuracy $a _ { \star } .$ , the reported relative improvement is

$$
1 0 0 ( a _ { \star } - a _ { 0 } ) / a _ { 0 } .
$$

This is a percentage change, not the percentage-point difference $1 0 0 ( a _ { \star } - a _ { 0 } ) ;$ ; negative gains remain possible. The source history’s final\_accuracy instead stores the best-so-far score including baseline and is not used directly as this endpoint. Repeated feedback and best-proposal selection use the same feedback partition, so the endpoint is not an untouched generalisation estimate. For the primary table, full-data condition means are compared with the figure-based human ordering $\mathrm { H Y B } >$ DCE > MCE, using 10,000 matched seed-bootstrap resamples at the observed sample size.

## B Memorization tests

We conducted targeted elicitation checks to examine whether GPT-4o could reproduce published study results from memory. These checks used the same gpt-4o-2024-11-20 snapshot as the benchmark. Failure to elicit a result does not establish that it was absent from training data or could not influence responses under another prompt.

## B.1 System engagement: Herlocker et al. [16]

We asked GPT-4o to report the published mean rating for each tested explanation method from Herlocker et al. [16], using Table 12.

We collected 30 responses per tested method at temperature 1, averaged the returned ratings, and compared them with the published means. The historical elicitation result was a weak Pearson correlation $( r = 0 . 2 0 8 ;$ Figure 6). This check did not recover the published pattern reliably, indicating memorization was unlikely an issue in our tests.

Table 11: Optimization prompt for diabetes prediction (DCE/MCE/HYB)  
```jinja
Optimization prompt structure
ROLE: You are a medical professional optimizing a diabetes prediction ML model by config
uring training data.
CURRENT STATUS
- Baseline accuracy: {baseline_accuracy}
- Current best accuracy: {current_best}
- Improvement so far: {delta}
- Original samples: {n_orig}, Current: {n_current}
- Available features: {feature_list}
CURRENT BEST CONFIGURATION
{formatted_config}
DATA-CENTRIC SIGNALS (for DCE/HYB)
- Class balance: pos %, neg %
- Highly correlated pairs (if any)
- Data quality scores: zeros, outliers, balance, correlation, overall
- Per-feature summary: min, mean, max, p5, p95, skew, zeros %, outliers %
MODEL-CENTRIC SIGNALS (for MCE/HYB)
- Feature importance ranking
- Top decision rules for Class 1 and 0 (confidence, samples)
OBSERVED FEATURE RANGES
- feat: [min, max], ...
HISTORY
{formatted_recent_iterations}
TASK
Propose a NEW configuration to beat {current_best}.
Return your REASONING followed by CONFIGURATION JSON:
{ "FeatureName": { "include": true/false, "min": number_or_null, "max": number_or_null } }
```

Table 12: The prompt used to try elicit memorization for our replication study of Herlocker et al. [16].
<table><tr><td>MLLM Prompt</td></tr><tr><td>In Herlocker et al.&#x27;s 2000 paper</td></tr><tr><td>“Explaining Collaborative Filtering Recommendations&quot;,</td></tr><tr><td>what did users rate {insert_method_name} on a 1-7 scale?</td></tr><tr><td>Give your answer in between the tags &lt;answer&gt;...&lt;/answer&gt;</td></tr></table>

## C Exploratory audit of model-improvement responses

Sample and procedure. We reused 420 randomly selected five-attempt runs: 30 per model, with 10 from each of DCE, MCE and HYB, selected without replacement using seed 20261010. Selection did not depend on outcomes. For this four-category audit, we directly reviewed the complete parser-facing response at iteration 4 of each run, together with its supplied explanations, current configuration and previous three attempts. Thus the analysis comprises 420 labeled iterations, not 2,100 independently labeled attempts. One LLM reviewer assigned the following labels.

Four categories. Visual interpretation records an explicit claim about the distribution image as consistent, mismatched, unassessable or absent. Tabular reasoning flags a concrete contradiction of supplied numbers, rule conditions or configurations, rather than an unsuccessful but plausible hypothesis. Use of iterative feedback distinguishes accurate, explicit use of earlier outcomes (G), a history/configuration contradiction or unacknowledged repeat of an ineffective proposal (Q), and no explicit outcome use (N). Labels may overlap.

![](images/b514227dd9f59fa538e0195ffe96ee02aff5a6f22e3c25bcd3a6288ebb0e0c86.jpg)  
Figure 6: Memorization elicitation with GPT-4o: returned mean ratings versus the published Herlocker et al. [16] means, using Table 12. The weak correlation (r = 0.208) indicates poor recovery under this prompt.

Improving and non-improving iterations. A good iteration strictly exceeds the best previously evaluated accuracy, including baseline. Of 420 fourth attempts, 40 (9.5%) establish a new best, 66 (15.7%) tie the previous best and 314 (74.8%) fall below it. All have evaluated scores. We retain ties separately in the exported data; the binary association analysis groups ties and declines as non-improving. A non-improving exploratory attempt is not itself a reasoning failure.

Table 13: Iteration-4 audit: 30 runs per model (10 per condition). Feedback and tabular columns are percentages of all 30 responses; visual mismatches are relative to the retained reference image, divided by explicit visual claims (including unassessable claims). One-case changes are absolute feedbackaccuracy differences of 1/116 (approximately 0.86 percentage points) relative to the previous best.
<table><tr><td>Model</td><td>New best</td><td>Accurate feedback (%)</td><td>Feedback problem (%)</td><td>Tabular error (%)</td><td>Visual mismatch/claims</td><td>One-case change</td></tr><tr><td>Claude 4.5 Sonnet</td><td>4/30</td><td>83.3</td><td>13.3</td><td>23.3</td><td>6/16</td><td>10/30</td></tr><tr><td>Claude 4.6 Sonnet</td><td>3/30</td><td>86.7</td><td>13.3</td><td>20.0</td><td>0/8</td><td>10/30</td></tr><tr><td>Claude 4.8 Opus</td><td>1/30</td><td>80.0</td><td>20.0</td><td>13.3</td><td>0/15</td><td>4/30</td></tr><tr><td>Gemma 3 27B</td><td>3/30</td><td>70.0</td><td>23.3</td><td>30.0</td><td>2/8</td><td>4/30</td></tr><tr><td>Gemma 431B</td><td>1/30</td><td>70.0</td><td>23.3</td><td>6.7</td><td>0/12</td><td>8/30</td></tr><tr><td>GPT-40</td><td>4/30</td><td>46.7</td><td>23.3</td><td>30.0</td><td>1/15</td><td>2/30</td></tr><tr><td>GPT-4o mini</td><td>4/30</td><td>40.0</td><td>13.3</td><td>16.7</td><td>0/7</td><td>4/30</td></tr><tr><td>GPT-5</td><td>3/30</td><td>93.3</td><td>0.0</td><td>6.7</td><td>0/10</td><td>7/30</td></tr><tr><td>GPT-5 mini</td><td>5/30</td><td>80.0</td><td>3.3</td><td>16.7</td><td>0/13</td><td>4/30</td></tr><tr><td>GPT-5.5</td><td>3/30</td><td>90.0</td><td>6.7</td><td>0.0</td><td>0/6</td><td>8/30</td></tr><tr><td>Mistral Small 3.2</td><td>0/30</td><td>46.7</td><td>23.3</td><td>16.7</td><td>0/1</td><td>4/30</td></tr><tr><td>04-mini Qwen3-VL 32B</td><td>2/30 4/30</td><td>43.3</td><td>13.3 36.7</td><td>40.0 43.3</td><td>0/1</td><td>8/30</td></tr><tr><td>Qwen3.5 122B INT4</td><td>3/30</td><td>56.7 80.0</td><td>16.7</td><td>30.0</td><td>1/20 0/14</td><td>6/30</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5/30</td></tr></table>

Findings. Accurate explicit feedback use occurs in 290/420 responses (69.0%); feedback problems occur in 69/420 (16.4%), and 61/420 (14.5%) contain no explicit use of earlier outcomes. Tabular errors occur in 88/420 (21.0%). There are 146 explicit visual claims: 124 consistent with the retained reference, 10 mismatched and 12 unassessable. Seven mismatches arise in API responses and three in open-weight responses; the API image-provenance limitation is particularly relevant to their interpretation.

New-best proposals occur in 32/290 responses with accurate feedback use (11.0%), versus 8/130 without this label (6.2%). Feedback problems and tabular errors have new-best rates of 5/69 (7.2%)

and 8/88 (9.1%), respectively. Their unlabelled counterparts have rates of 9.97% and 9.64%. Thus the inspected errors do not provide a strong separation between successful and unsuccessful exploration. Accurate feedback use has a small positive pooled correlation with fifth-attempt improvement over baseline, but its within-model–condition centered correlation is only 0.035 (versus 0.135 pooled). This attenuation and the uncertain association with fourth-attempt success argue against a causal interpretation or a general claim that feedback use explains performance differences.

Table 14: Exploratory point-biserial correlations (phi for binary new-best outcome), with 95% percentile intervals from 10,000 run-level resamples stratified within model and condition. Final accuracy is the evaluated fifth-attempt score (419 available; one failed attempt is missing); final gain is the percentage-point difference from the run’s baseline. No multiplicity adjustment. Visual comparisons use all responses, including those without an explicit visual claim.
<table><tr><td>Label</td><td>New best at attempt 4</td><td>Attempt-5 accuracy</td><td>Attempt-5 gain over baseline</td></tr><tr><td>Accurate feedback use</td><td>0.08 [-0.01, 0.15]</td><td>0.06 [-0.04, 0.16]</td><td>0.14 [0.04, 0.23]</td></tr><tr><td>Feedback problem</td><td>-0.03 [-0.11, 0.05]</td><td>0.02 [-0.07, 0.10]</td><td>-0.04 [-0.13, 0.03]</td></tr><tr><td>Tabular error</td><td>-0.01 [-0.10, 0.09]</td><td>-0.07 [-0.17, 0.03]</td><td>-0.09 [-0.19, 0.02]</td></tr><tr><td>Visual mismatch</td><td>0.00 [-0.06, 0.10]</td><td>0.02 [-0.05, 0.08]</td><td>0.01 [-0.04, 0.05]</td></tr></table>

Of the 40 new-best improvements, 19 (47.5%) increase feedback accuracy by exactly 1/116 (approximately 0.86 percentage points). Across all reviewed iterations, 84/420 (20.0%) have an absolute feedback-accuracy difference of 1/116 from the previous best. These counts demonstrate coarse feedback resolution, not that those improvements are spurious; repeated independent evaluation would be required to separate generalizable improvement from feedback-set optimization.

## D Compute resources

Collecting new responses has different requirements: API access for the nine proprietary models, or Linux/NVIDIA hardware with sufficient GPU memory and the pinned inference stack for openweight models. Four open-weight checkpoints use FP8 quantisation; Qwen3.5 122B uses GPTQ INT4. Running the entire collection took several days using a G7e GPU with 32GB RAM on AWS. The API calls by comparison are quicker, but entail possibly significant costs.

## E Prompt-Perturbation Robustness Experiment

## E.1 Motivation

We conducted a supplementary experiment to assess sensitivity to minor, meaning-preserving changes in prompt wording. This historical experiment is separate from the final 300-question teaching benchmark.

## E.2 Setup

We used GPT-5 mini on 50 class-balanced teaching questions under the control, causal and counterfactual conditions. Each question–condition pair received eight responses in each of two rounds, giving 1,200 calls per round and 2,400 calls in total.

## E.3 Perturbation Procedure

• Original: the unmodified prompt was sampled eight times.

• Minor edit: an auxiliary MLLM edited each base prompt once, replacing one to five words with synonyms or near-synonyms while preserving sentence meaning, numbers, feature names, instances and the required output format. That fixed edited prompt was then sampled eight times.

## E.4 Analysis and Results

We compared the proportion of positive answers and the agreement of modal answers across rounds. The question–condition pair, rather than each repeated answer, is the unit for modal agreement. The reported historical summary gives unchanged modal answers for 148 of 150 pairs, with the two changes occurring in opposite directions. Positive-answer rates differed by 0.4–3.3 percentage points (Table 15). The response-level two-proportion tests shown below are exploratory summaries and do not account for dependence among answers to the same question.

<table><tr><td>Condition</td><td colspan="2">Positive rate</td><td>Modal</td><td>Flips</td><td> $p$ </td></tr><tr><td></td><td>Original</td><td>Edited</td><td>agreement</td><td>(of 50)</td><td>(2-prop.)</td></tr><tr><td>Control</td><td>0.421</td><td>0.417</td><td>1.00</td><td>0</td><td>0.918</td></tr><tr><td>Causal</td><td>0.409</td><td>0.376</td><td>0.98</td><td>1</td><td>0.341</td></tr><tr><td>Counterfactual</td><td>0.495</td><td>0.472</td><td>0.98</td><td>1</td><td>0.520</td></tr><tr><td>All</td><td>0.441</td><td>0.422</td><td></td><td>2</td><td>0.335</td></tr></table>

Table 15: Historical prompt-perturbation summary: 50 questions, three conditions, eight responses per round, and GPT-5 mini. Positive rate is the proportion of answers equal to 1. Modal agreement compares aggregated answers between rounds. The reported response-level p-values are unadjusted exploratory summaries.

## F Mental-model alignment through recourse rationales

We investigated mental-model alignment through the rationales people and MLLMs express about classifier rejection and proposed changes. Specifically, do similar expressed beliefs about justification, feasibility and burden accompany similar ratings? Tominaga et al. [35] report 13 human quotations linked to acceptance or willingness ratings; all are listed in Table 16, grouped into eight themes. These illustrative cases motivate directional expectations, rather than establish population-level human correlations. Our comparison concerns expressed understanding, not direct access to internal mental models.

We sampled 30 existing responses per model using a seed-42 shuffled order (420 responses across 14 models). One automated labeller (GPT-6 Astra) applied a fixed codebook with model identities and explicit score fields masked. For each model and theme, we correlated label presence with its designated rating using Spearman’s $\rho ,$ excluding uncertain labels. We multiplied $\rho$ by the expected direction so positive values indicate alignment. We obtained 95% percentile intervals from 10,000 bootstrap resamples of applicant clusters.

Feasibility (T4) accompanied higher willingness, and disproportionate effort (T8) accompanied lower willingness: 10/14 and 11/14 models, respectively, had intervals entirely in the expected direction and passed the count/stability checks in Figure 7. Small-change criticism (T1) had the expected point-estimate direction in 12/14 models, but only four passed these interval and count checks. Other themes were mixed or rare.

## G Correlation between human mean differences and ranking success

We examined whether larger differences between human study outcomes were associated with more consistent reproduction of their rankings by MLLMs. For each of 26 condition comparisons across four studies, we calculated the relative difference between published human means, estimating Model Improvement means by digitising the participant scores in its published figure. We plotted this difference against the percentage of 14 MLLMs reproducing the same ordering. Larger human differences were associated with greater agreement (Spearman’s $\rho = 0 . 5 4 )$ . The absence of points in the lower-right region of Figure 8 indicates that large human differences were rarely accompanied by widespread ranking failures among the tested models. This descriptive pattern suggests that larger human differences may be easier to reproduce.

The comparisons used for Figure 8 are in Table 17.

Table 16: All 13 human quotations used in the audit, grouped into eight themes, from Tominaga et al. [35], pp. 6–7. A denotes acceptance of the original rejection; W denotes willingness to act. Ratings are out of seven. Arrows give the expected association with theme presence. Short quotations are interpreted using the authors’ surrounding text.
<table><tr><td></td><td>Code Theme</td><td>Human quotation (participant)</td><td>Rating; direction</td></tr><tr><td>T1</td><td rowspan="3">Small-change criticism</td><td>“Almost same&quot; (P006)</td><td>A: 1;↓</td></tr><tr><td>T1</td><td>&quot;That&#x27;s it?&quot; (P446)</td><td>A: 2; ↓</td></tr><tr><td>T1</td><td>“Not particularly different compared to the current situation&quot;A: 2; ↓ (P706)</td><td></td></tr><tr><td>T2</td><td>tion</td><td>Large gap justifies rejec- “The career is too different&quot; (P567)</td><td>A: 7;↑</td></tr><tr><td>T2</td><td rowspan="4"></td><td>&quot;I think it&#x27;s a solid reason, so I don&#x27;t blame it&quot; (P739)</td><td>A: 6; ↑</td></tr><tr><td>T3</td><td>“Because I realize how much I&#x27;m not enough&quot; (P732)</td><td>A: 5; ↑</td></tr><tr><td>T4 Readily feasible change</td><td>“Because I think I can tackle it right now&quot; (P674)</td><td>W: 5;↑</td></tr><tr><td>T4</td><td>“If it was just working hours, I&#x27;d do it in a heartbeat&quot; (P718)</td><td>)W: 7; ↑</td></tr><tr><td>T4</td><td rowspan="3"></td><td>“Because the contents are easy to change&quot; (P294)</td><td>W: 7;↑</td></tr><tr><td>T5</td><td>“All I have to do is wait a little&quot; (P655)</td><td>W: 6; ↑</td></tr><tr><td>T6 Time or money burden</td><td>“Because there&#x27;s too much to do, it takes too much time and W: 2; ↓</td><td></td></tr><tr><td>T7</td><td>Too many requirements</td><td>money&quot; (P401) “There are too many items to clear&quot; (P167)</td><td></td></tr><tr><td>T8</td><td>Disproportionate effort</td><td>“I don&#x27;t want to change various things just to apply for a W: 1; ↓</td><td>W: 1; ↓</td></tr><tr><td></td><td></td><td>loan&quot; (P468)</td><td></td></tr></table>

Table 17: The 26 comparisons included in the human-gap analysis. The symbol > denotes a higher human mean, not necessarily a statistically significant difference. NN denotes nearest neighbours; E1–E21 identify engagement explanation conditions. E4 versus E5 is excluded because their reported means are equal. Model Improvement means were estimated from participant points in the published figure. Recourse is excluded because comparable discrete condition means are unavailable.
<table><tr><td>Study/domain</td><td>Human ordering</td><td>Study/domain</td><td>Human ordering</td></tr><tr><td>Teaching</td><td>Counterfactual &gt; Control</td><td>Engagement</td><td>E1 &gt; E21</td></tr><tr><td>Teaching</td><td>Counterfactual &gt; Causal</td><td>Engagement</td><td>E2 &gt; E4</td></tr><tr><td>Teaching</td><td>Causal &gt; Control</td><td>Engagement</td><td>E2 &gt; E5</td></tr><tr><td>Reliance—Adult</td><td>NN &gt; Control</td><td>Engagement</td><td>E2 &gt; E11</td></tr><tr><td>Reliance—Adult</td><td>NN &gt; Saliency</td><td>Engagement</td><td>E2 &gt; E21</td></tr><tr><td>Reliance—Adult</td><td>Control &gt; Saliency</td><td>Engagement</td><td>E4 &gt; E11</td></tr><tr><td>Reliance—BIOS</td><td>NN &gt; Control</td><td>Engagement</td><td>E4 &gt; E21</td></tr><tr><td>Reliance—BIOS</td><td>NN &gt; Saliency</td><td>Engagement</td><td>E5 &gt; E11</td></tr><tr><td>Reliance—BIOS</td><td>Saliency &gt; Control</td><td>Engagement</td><td>E5 &gt; E21</td></tr><tr><td>Engagement</td><td>E1 &gt; E2</td><td>Engagement</td><td>E11 &gt; E21</td></tr><tr><td>Engagement</td><td>E1 &gt; E4</td><td>Model Improvement</td><td>HYB &gt; DCE</td></tr><tr><td>Engagement</td><td>E1 &gt; E5</td><td>Model Improvement</td><td>HYB &gt; MCE</td></tr><tr><td>Engagement</td><td>E1 &gt; E11</td><td>Model Improvement</td><td>DCE &gt; MCE</td></tr></table>

![](images/48610bfd4f74e2f77416b19d98427cbdf35c7c6a483f964b9ffd4fc82ec6ff47.jpg)  
\* CI excludes zero and passes count/stability checks. † Sparse or unstable. — Undefined.

Figure 7: Associations between expressed recourse rationales and ratings, separately for each model. Positive sign-aligned ρ follows the human-example direction in Table 16. Bold starred cells have 95% intervals excluding zero, at least five label-present and five label-absent cases, and at least 95% defined bootstrap replicates. Daggers flag sparse or unstable estimates; dashes denote undefined correlations. Applicant clustering accounts for multiple plans for the same applicant. Rare labels cannot be assessed reliably from this pilot.

![](images/a60e11e0b17b0c188231c87ab24dd35570fd8b0d917d22108f754f1592ec9e6d.jpg)  
Figure 8: Human performance differences and MLLM ranking agreement. Each point represents one of 26 condition comparisons across four human studies (no recourse study). The x-axis shows the relative difference between human means, $1 0 0 ( \mu _ { \mathrm { h i g h e r } } - \mu _ { \mathrm { l o w e r } } ) / \mu _ { \mathrm { l o w e r } } ;$ the y-axis shows the percentage of 14 MLLMs reproducing that ordering. Colours distinguish study domains; black crosses indicate Model Improvement comparisons using means estimated from published participant dots. Larger markers labelled ×2 represent overlapping comparisons. The dashed line shows a descriptive linear trend (Spearman’s $\rho = 0 . 5 4 )$ . The empty lower-right region suggests fewer ranking failures for large human differences within these studies. Comparisons share conditions and are not independent.

![](images/5469b9fb6173be8a677014d5763e0c8dbdf04317db6cbfa7dbf81ab1f82c983f.jpg)  
Figure 9: All-model model-improvement results: best successfully evaluated proposal among five attempts, excluding baseline from the maximum, expressed as relative percentage improvement over the run baseline. There are 100 runs per condition, except 40 for Qwen3.5 122B. Boxes show the interquartile range and median; whiskers span the central 95% of observations. Outliers omitted from display still contribute to the mean.

![](images/31cc95ca465e2de917ebed1fb061216f51de8004abf593f3d5dbd1406f02e2ad.jpg)  
Figure 10: Publication rendering of the feature distributions used in model improvement. The original static image supplied to the open-weight DCE and HYB conditions is preserved separately in the release; this grid-free rendering is not a replacement for the collection input.

![](images/fc426829a3fd59f4a07fbda9566044ce1010284af49a10e0c1d5c2fd8bdd3824.jpg)  
Figure 11: All-model teaching accuracy over the 300 matched synthetic questions per condition, compared descriptively with published human means. Warren et al. provide the benchmark's teaching reference; Goyal et al. provide a separate comparison whose middle condition is Grad-CAM rather than causal explanations. Invalid model answers count as incorrect.

![](images/b080d103b950914905034ab495f9b46783a88b24bb7df300f35c3dcd2b1c44af.jpg)  
Figure 12: Adult Census reliance results over 416 cases per condition. Overall denotes classification accuracy; AI Correct and AI Incorrect restrict cases by the classifier's actual correctness. AI Baseline is classifier accuracy on the selected cases (260/416). Control has no AI advice; the saliency and nearest-neighbour conditions include advice. These bars do not establish significance of condition differences.

Appropriate reliance · BIOS (NLP)  
![](images/db3355b2b9f4ddc44a4c9a02fb7fffb93bc1edf8fab315c475024c026fa18d5d.jpg)  
Figure 13: BIOS reliance results over 416 cases per condition. Overall denotes classification accuracy; AI Correct and AI Incorrect restrict cases by the classifier's actual correctness. AI Baseline is 260/416 for GPT-4o, GPT-4o mini, GPT-5 mini and o4-mini, and 263/416 for the other models. Some API case sets differ, so cross-model bars are descriptive rather than paired case comparisons.

![](images/a2b59ae5e2f78e06ac1d7792b32935e7cabd71690c118fed7bda9f84eb541ae8.jpg)  
Figure 14: Willingness to act versus $L _ { 0 }$ changed-feature count, retained as a supplementary diagnostic after its exclusion from the primary aggregate. Points show valid ratings; curves are fitted population mean GAMM trends, with approximate pointwise 95% mean intervals. The 200 plans share 39 applicants. Our adaptation permits numerical and categorical changes, whereas Tominaga et al. [35] represented many quantities using ordered categories. $L _ { 0 }$ counts changed features without measuring the magnitude of each change: reducing a loan amount by 1 DM or 1000 DM contributes equally. Differences in feature representation and permitted changes limit direct comparison with the original study and motivated reporting $L _ { 0 }$ separately. We retain these results for transparency.

![](images/80a407d4ec41e80b860c2c17a4c0e3df70c885946496081473cc9e2d923c8e7e.jpg)  
Figure 15: Willingness to act versus mixed $L _ { 1 }$ distance: training-standard-deviation-scaled numeric changes plus one unit per changed categorical field. Points show valid ratings; curves are populationmean GAMM trends, with approximate pointwise 95% mean intervals. The 200 plans share 39 applicants; the relationship is not a matched human-rating comparison.

![](images/35cd85f822f64d1112f149c2796b9844afe7878dc7ef9e2285b46889d74c3083.jpg)  
Figure 16: System-engagement comparisons between the six published human condition means and model means from complete response blocks. Each panel reports Pearson's r over six means; its response-block count is not the correlation sample size. The primary table uses Spearman's $\rho .$ Panel scales can differ.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The paper evaluates 14 models across five purpose-grounded studies and reports model- and task-dependent agreement. Its conceptual analysis concerns sufficient conditions for rank preservation; the empirical results support preliminary evaluation and prioritising human studies.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes].

Justification: We describe reconstructed study materials, fixed human references, conditional bootstrap uncertainty, feedback-set optimisation, model/version dependence, and the limited scope of contamination and prompt-perturbation checks. These results do not establish population-level human validity.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes].

Justification: Section 3 defines condition-specific calibration error. For a positive human mean gap $\Delta _ { H }$ , the corresponding model gap is $\Delta _ { H } + \epsilon _ { i } - \epsilon _ { j }$ , which is positive whenever $| \epsilon _ { i } - \epsilon _ { j } | < \Delta _ { H }$ . This is a sufficient algebraic condition, not an empirical guarantee that the assumptions hold.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes].

Justification: The release supplies frozen analysis inputs, available prompts/responses, manifests and scripts for reproducing the primary table and figures. Missing historical collection provenance is identified. Reproducing analysis from saved data is distinguished from re-running stochastic inference.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes].

Justification: The repository provides the analysis inputs, code, pinned dependencies and reproduction instructions. Third-party data and study materials retain their applicable terms.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes].

Justification: The study descriptions specify task inputs, conditions, sampling units, available model settings, classifier configurations and endpoints. The release records additional settings and provenance limits.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes].

Justification: The primary table reports full-data Spearman estimates and 95% percentile intervals from 10,000 matched bootstrap resamples at observed sample sizes. Human references are fixed for the condition-ranking studies; recourse instead correlates negative $L _ { 1 }$ cost with willingness, conditional on the observed applicant pool. Figure captions distinguish descriptive distributions, standard errors and GAMM population-mean bands. A nonnegative lower interval bound is a formatting convention rather than evidence of strictly positive correlation.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix D details it.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: We use existing study references and model-generated responses, preserve source attribution, and remove credentials and private deployment identifiers from the release.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes].

Justification: Surrogate evaluation can help prioritise human studies and reduce reliance on poorly targeted proxy metrics. Misusing model agreement as a substitute for human evidence could favour explanations that perform well for models but poorly for affected people. Our recommendations retain human validation and reporting of uncertainty.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [N/A].

Justification: We do not release model weights or a new high-risk generative model. The released benchmark preserves dataset restrictions and redacts credentials and private infrastructure identifiers; generated text is not presented as human participant data.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes].

Justification: We cite the source studies and datasets and provide checkpoint and license notices.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes].

Justification: The release documents synthetic teaching inputs, reconstructed task materials, saved model responses, analysis code, schemas, provenance, licenses and reproducibility limits. Project-owned data and annotations are offered under CC BY 4.0 with explicit third-party exclusions.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [N/A].

Justification: We did not recruit new human participants. Human findings are taken from the cited studies, and our new responses are generated by MLLMs.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A].

Justification: No new human-subject experiment was conducted. We report published human-study findings and generate new model responses.

## 16. Declaration of MLLM usage

Question: Does the paper describe the usage of MLLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the MLLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes].

Justification: MLLM surrogate evaluation is the core method. We describe the models, prompts, response processing, sampling and analysis throughout the paper and appendix.