# Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches

Marco Simnacher   
Georg Keilbar   
Benjamin K¨onig   
Chair of Statistics   
Humboldt-Universit¨at zu Berlin   
Spandauer Str. 1, 10178 Berlin, Germany

Christoph Lippert Chair Digital Health & Machine Learning Hasso-Plattner-Institut for Digital Engineering Prof.-Dr.-Helmert-Straße 2-3, 14482 Potsdam, Germany

Sonja Greven

Chair of Statistics

Humboldt-Universit¨at zu Berlin

Spandauer Str. 1, 10178 Berlin, Germany marco.simnacher@hu-berlin.de georg.keilbar@hu-berlin.de b.koenig@hu-berlin.de

Christoph.Lippert@hpi.de

sonja.greven@hu-berlin.de

## Abstract

Conditional independence tests (CITs) test for conditional dependence between two random objects X and Y given a third random object Z. Existing CITs have limited applicability to high-dimensional data, especially multimodal data like text. However, we show that such tests are of interest for large language model (LLM) outputs, where we test whether an output X generated from a source text Z carries information about an attribute Y beyond Z itself. For this purpose, we propose embedded CITs (eCITs), which embed X and Z and apply an existing CIT to the resulting representations and to Y . We show that, provided the embedding of Z is suficient, i.e. retains the information Z carries about either Y or the representation of X, the null hypothesis transfers from X and Z to their representations, so that a CIT valid for the embedded hypothesis is valid for the original one. We further give conditions for equivalence of the two hypotheses, and show that suficiency weakens to mean suficiency when the embedded test targets conditional mean independence. We propose a semi-synthetic simulation design to assess type I error (T1E) control and power of the eCITs for given embedding maps on a specific dataset and task, and use it to evaluate them on our application. Applying the eCITs to German Parliament speeches, we find for all combinations of embedding maps considered that the summaries of two LLMs contain information about the speaker’s faction and gender beyond the speech they were generated from.

Keywords: conditional independence testing, embedding maps, suficient representation, large language model

## 1 Introduction

A conditional independence test (CIT) is a statistical test for the null hypothesis of conditional independence (CI), $H _ { 0 } : X \perp \perp Y | Z$ , between two random variables, X and Y , given a third variable, Z, against the alternative hypothesis of conditional dependence, $H _ { 1 } : X \not \approx Y | Z .$ . X and Y are conditionally dependent, given Z, if X (or Y) provides additional information about Y (or X), given Z. A CIT should control the type I error (T1E), i.e. the probability of falsely rejecting the null hypothesis, while simultaneously achieving high power, i.e. a high probability of correctly rejecting the null hypothesis of CI.

One setting in which such conditional independencies are of interest is the evaluation of large language models (LLMs). Consider a source text Z, e.g. a parliament speech, observed together with a protected attribute Y , e.g. the speaker’s gender or faction, and an LLMgenerated summary X of the source text. Such summaries can contain biases regarding the protected attribute, framing e.g. the speeches of one faction more critically than those of another, although the content of the speeches does not difer. Due to the widespread use of LLMs across diferent tasks, such biases are widely studied (Gallegos et al., 2024). However, the complexity of LLMs and their task-specific behavior make general statements about the bias of a specific LLM challenging (Anthis et al., 2024), and existing bias metrics are of limited use in this setting. Intrinsic metrics based on embeddings or token probabilities often do not correlate with the bias observed in downstream tasks (Goldfarb-Tarrant et al., 2020; Delobelle et al., 2022). Extrinsic metrics are typically bound to benchmark datasets derived from sentence completion or occupation masking, so that their results need not translate to domain-specific tasks (Li et al., 2023). LLM-based evaluators are applicable to arbitrary tasks, but can themselves be biased (Sun et al., 2022; Gao et al., 2025), and their scores do not come with statistical guarantees.

We instead define bias as additional information in the LLM output X about the protected attribute Y beyond the information already contained in the source text Z. This directly corresponds to the alternative hypothesis of a CIT, so bias can be tested on a given task and dataset with a T1E guarantee. Since the summary is generated solely from the source text, any such additional information cannot come from the observed input itself, but must have entered through the model, for example via memorization of the source texts through the training data the model was fitted on. The definition is reference-free in the sense that no gold-standard output or counterfactually generated output has to be constructed, since the source text itself supplies the reference. While we focus on summarization as one instance, the definition applies to any task in which the output is generated from a source text. In our empirical application with a categorical variable Y , the definition is a conditional version of the partisanship measure of Gentzkow et al. (2019), who quantify how strongly congressional speech reveals a speaker’s faction. We ask the same of the summaries, treating the speech as the conditioning variable: whether the summary reveals the speaker’s faction beyond the speech it was generated from.

Testing this hypothesis, however, is not straightforward when X and Z are texts. Existing CITs focus on low-dimensional, vector-valued variables (Li and Fan, 2020), and both Z and X cause problems through their structure and high-dimensionality. The structure of Z afects the T1E of the CITs. Their null hypothesis comparisons require an object constructed from Z: the conditional distribution of X or of Y given Z for resamplingbased comparisons (Candes et al., 2018; Berrett et al., 2020), a kernel or a partition on the support of Z for kernel- and neighborhood-based comparisons (Zhang et al., 2011; Strobl et al., 2019), or a regression on Z for residual-based comparisons (Shah and Peters, 2020; Lundborg et al., 2024). None of these objects can be constructed on raw texts with the guarantees the tests require, whether because the construction itself is unavailable or because the rate conditions on which validity rests are not. The tests thus become applicable only on a feature representation of Z and condition on that representation rather than on Z, as in the application of algorithm-agnostic significance tests to multimodal data (Kook and Lundborg, 2024). The text structure of X afects both whether a CIT can be applied at all and, once it can, its power. Simnacher et al. (2026) address this for structured, highdimensional X by embedding it onto a feature representation and applying a CIT to the representation instead. Provided the embedding map is learned as they prescribe, e.g. on external data or through sample splitting, this leaves the T1E unafected in their setting, where Z is not embedded, and the choice of representation afects only the power, which depends on whether the representation retains the conditional association between X and Y given Z. The representation, however, remains high-dimensional, so that only a few CITs are applicable to it. Neither line of work addresses the representation of Z, on which the T1E of the resulting test depends.

We therefore require CITs that are applicable to texts X and Z, control the T1E for the hypothesis stated on the level of the texts, and have power against relevant alternatives. To meet these requirements, we propose embedded CITs (eCITs), which map both Z and X onto feature representations and apply an existing CIT for conditional dependence of the representation of X and Y, given the representation of Z, making the test applicable to structured, high-dimensional inputs. Since the CIT conditions on the representation of Z rather than on the source text, it is enough that that representation retain the information in Z about either Y or the representation of X. Any information it drops plays the role of an omitted conditioning variable and can inflate the T1E. We call embedding maps ω<sub>Z</sub> yielding such a representation suficient and characterize them in Subsection 3.1. The representation of X needs no suficiency condition of its own and mainly afects the power, since the test has power only if it retains the conditional association between X and Y given Z in a form the CIT can detect. The two choices nevertheless interact, since the suficiency of the embedding of Z is stated relative to the representation of X, unlike in Simnacher et al. (2026), where the conditioning variable is not embedded and the choice of representation afects the power alone. We discuss this further in Subsection 3.1 and Section 4. The CIT contributes its guarantees at the level of the representations, so whether they transfer to the texts depends on the embedding maps and the dataset at hand.

Our contributions to the existing literature are as follows. First, we justify CI testing as a bias criterion for LLM outputs and discuss when the null and the alternative can hold for a fixed LLM (Section 2). Second, we propose eCITs, which combine embedding maps for X and Z with existing CITs, making CI testing applicable when both X and Z are texts (Section 3). Third, we characterize suficient embedding maps for Z, under which the null hypothesis stated on the texts transfers to the null hypothesis stated on the representations, so that a valid CIT for the embedded hypothesis controls the T1E for the textual one. We also give conditions under which the two hypotheses are equivalent, and show that the requirement weakens to mean suficiency when the embedded test targets conditional mean independence (Subsection 3.1). Fourth, we adapt the projected covariance measure (PCM; Lundborg et al., 2024), the conditional mean independence test (CMIT) we apply, to the categorical Y of our application, with a precision-weighted projection direction that we show to be optimal, and record the conditions under which the validity argument of Lundborg et al. (2024) carries over (Appendix D). Fifth, we show empirically how insuficient embeddings of Z inflate the T1E, and provide a simulation design that can be used to assess the eCITs on a given dataset before applying them (Subsection 4.2). On speech-summary pairs, eCITs with a suficient embedding of $Z$ are calibrated at all sample sizes considered for the categorical outcomes, also for high-dimensional representations, while for insuficient ones the inflation ranges from negligible to growing steeply with the sample size, depending on how much of the dropped information the remaining embedding maps recover. Finally, we apply the eCITs to German Parliament speeches and find, for both summarizers and all combinations of embedding maps considered, that the summaries of two LLMs contain information about the speaker’s faction and gender beyond the speech (Subsection 4.3).

## 2 Conditional (In)dependence for LLM-Generated Summaries

Along with a protected scalar variable Y and a source text $Z ,$ we generate LLM outputs X from the source text Z. We define bias as additional information in the output X about a protected attribute Y given the source text $Z .$ . Specifically, we test for bias in an output by testing the hypothesis

$$
H _ { 0 } : Y \perp \perp X \mid Z \quad { \mathrm { v s . } } \quad H _ { 1 } : Y \nmid X \mid Z .\tag{1}
$$

Under the null, the output does not add any information about the protected attribute beyond the information already available in the source text. Under the alternative, the output provides additional information about the protected attribute that was not available in the source.

We distinguish between the text corpus $D _ { L L M }$ used to train the LLM and the test dataset $D _ { C I T } = ( X _ { i } , Y _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ used in the CIT. To generate $X _ { i }$ , we prompt the LLM with a task constant across all i and a source text $Z _ { i }$ , cf. Subsection 4.1 and Appendix B.1 for details. If the text corpus $D _ { L L M }$ and $( Z , Y ) \in D _ { C I T }$ are independent, we are trivially under the null hypothesis in (1), since the output X is a measurable function of $Z , D _ { L L M }$ and, under stochastic decoding, of sampling noise drawn independently of $( Y , Z )$ , and hence conditionally independent of $Y$ given Z. If $Z , Y$ or texts $\tilde { Z }$ with the same $Y$ are contained in the text corpus, then the null or alternative hypothesis can be true.

A plausible mechanism for such a conditional dependence is memorization. This is also the situation we expect in our application in Section 4, where the source texts are German Parliament speeches and the two LLMs used to summarize the speeches are pre-trained on web-scale German corpora that plausibly include Bundestag plenary protocols. If the LLM has seen a source text $Z _ { i } ,$ , or other texts by the same speaker, during training, its weights can carry information about the associated attribute $Y _ { i }$ that $Z _ { i }$ alone does not determine. When generating $X _ { i }$ from $Z _ { i }$ , the model may then reproduce phrasing or framing tied to that particular observation rather than to the content of $Z _ { i }$ , so that $X$ reveals $Y$ beyond $Z .$ The following example isolates this mechanism in a setting in which the null and the alternative can be constructed by design.

Example 1 We generate $D _ { t r } = ( Z _ { i } , Y _ { i } ) _ { i = 1 } ^ { N }$ , with independent copies $Z _ { i } \sim N ( 0 _ { 1 0 } , I _ { 1 0 } )$ and $Y _ { i } = Z _ { i , 1 } + \varepsilon _ { i } , \ \varepsilon _ { i } \stackrel { i i d } { \sim } N ( 0 , 1 )$ . We $\it { \Delta } f$ a k-nearest neighbor model $f _ { \theta }$ on $D _ { t r } = ( Z _ { i } , Y _ { i } )$ ), and afterwards fix the model $f _ { \hat { \theta } }$ . Then, we predict $Y _ { i }$ from $Z _ { i }$ using $( Z _ { i } , Y _ { i } ) _ { i = 1 } ^ { n } \subset D _ { t e s t }$ Under the null hypothesis, $( Z _ { i } , \dot { Y _ { i } } ) _ { i = 1 } ^ { n }$ from the test dataset $D _ { t e s t } = ( X _ { i } , Y _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ is sampled independently of $D _ { t r } , X _ { i } = f _ { \hat { \theta } } ( Z _ { i } )$ are out-of-sample predictions, and thus, $X \perp \perp Y \mid Z$ holds on $D _ { t e s t }$ . Under the alternative, $D _ { t e s t } \subset D _ { t r }$ , the predictions $X _ { i } = f _ { \hat { \theta } } ( Z _ { i } )$ are in-sample, and thus, $X \not \vdash Y \mid Z$ holds on $D _ { t e s t }$

![](images/6100ec2fc8f7f86747d9b3c0d5f6527e0482df15fa2e40c8e9c871b190286fec.jpg)  
Figure 1: Rejection rates of the PCM testing conditional mean independence of X and Y given Z with default parameters applied to data generated with $Z \sim N ( 0 _ { 1 0 } , I _ { 1 0 } )$ $Y _ { i } = Z _ { i , 1 } + \varepsilon _ { i } , \varepsilon _ { i } \overset { i i d } { \sim } N ( 0 , 1 )$ , and $X _ { i } = f _ { \hat { \theta } } ( Z _ { i } )$ , where $f _ { \hat { \theta } }$ is a knn model predicting Y from Z with $k = 2 , 5 , 1 0$ . Under the null, the training dataset $D _ { t r } = ( Z _ { i } , Y _ { i } ) _ { i = 1 } ^ { N }$ is independent of the test dataset $D _ { t e s t } = ( X _ { i } , Y _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ , under the alternative $D _ { t e s t } \subset D _ { t r }$

Figure 1 shows the rejection rates of the PCM with default parameters testing for conditional dependence for three k-nearest neighbor (knn) models $( k { = } 2 , 5 , 1 0 )$ under the null and alternative. Under the null, the T1E of the PCM is within one Monte Carlo standard error of the nominal level of 0.05. The power under the alternative depends on the number of nearest neighbors. For a specific $Y _ { i } ,$ , this $Y _ { i }$ contributes more to the model’s predictions for a model with smaller $k ,$ thus leading to larger conditional dependence and higher rejection rates. In the example, $D _ { t r }$ and $D _ { t e s t }$ mimic $D _ { L L M }$ and $D _ { C I T }$ . We observe that a model with a stronger memorization of individual observations leads to stronger conditional dependence, detectable by the PCM.

## 3 Embedded Conditional Independence Tests

Let $( \mathcal { X } , \mathcal { F } _ { \mathcal { X } } ) , ( \mathcal { V } , \mathcal { F } _ { \mathcal { Y } } ) , ( \mathcal { Z } , \mathcal { F } _ { \mathcal { Z } } )$ be measurable spaces and $X , Y , Z$ be $( \mathcal { X } , \mathcal { F } _ { \mathcal { X } } ) \ – , \ ( \mathcal { Y } , \mathcal { F } _ { \mathcal { Y } } ) \ –$ , $( \mathcal { Z } , \mathcal { F } _ { \mathcal { Z } } )$ -valued random variables with joint distribution $ { \mathbb { P } } ^ { X , Y , Z }$ . Moreover, let $\hat { \beta }$ and $\hat { \gamma }$ be $( B , { \mathcal { F } } _ { B } )$ - and $( \mathcal { C } , \mathcal { F } _ { \mathcal { C } } )$ -valued random variables representing estimated parameters of embedding maps ω<sub>X</sub> : $\mathcal { X } \times \mathcal { B }  \mathfrak { X } , ( X , \hat { \beta } ) \mapsto \omega _ { X } ( X , \hat { \beta } )$ and $\omega _ { Z } : \mathcal { Z } \times \mathcal { C } \to 3 , ( Z , \hat { \gamma } ) \mapsto \omega _ { Z } ( Z , \hat { \gamma } )$

We denote the resulting random representations by $X ^ { \omega _ { X } } = \omega _ { X } ( X , \hat { \beta } )$ and $Z ^ { \omega _ { Z } } = \omega _ { Z } ( Z , \hat { \gamma } )$ which are $( { \mathfrak { X } } , { \mathcal { F } } _ { \mathfrak { X } } )$ - and $( 3 , \mathcal { F } _ { 3 } )$ -valued random variables. The focus of this paper is on X and $\mathcal { Z }$ as spaces of text, $\mathcal { V } \subseteq \mathbb { R }$ or $\mathcal { V } = \{ 1 , \ldots , L \}$ as a univariate continuous or categorical variable, and $\mathcal { X } \subseteq \mathbb { R } ^ { q }$ and $3 \subseteq \mathbb { R } ^ { p }$ as the spaces of feature representations. Nevertheless, the theoretical results apply also to CITs applicable to vector-valued Y , as well as general X and $Z$ for which representations can be obtained. The notation and the assumptions on $\hat { \beta }$ follow Simnacher et al. (2026), where $Z$ is not embedded; we extend the setup by embedding Z and by conditioning on $\hat { \gamma }$

We assume that $( X _ { i } , Y _ { i } , Z _ { i } ) , i = 1 , . . . , n$ are n independently and identically distributed (i.i.d.) copies of $( X , Y , Z )$ . We define $X _ { i } ^ { \omega _ { X } } = \omega _ { X } ( X _ { i } , \hat { \beta } )$ , and write $X ^ { n } = ( X _ { 1 } , \ldots , X _ { n } ) $ ), $( X , Y ) ^ { n } = ( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n } ,$ and analogously for all other combinations of $X _ { i } ^ { \omega _ { X } } , Z _ { i } ^ { \omega _ { Z } } , X _ { i } , Y _ { i } , Z _ { i }$ $i = 1 , \ldots , n$ . For the whole sample with text and their feature representations, we write $S = ( X , Y , Z ) ^ { n }$ and $S ^ { \omega } = ( X ^ { \omega x } , Y , Z ^ { \omega z } ) ^ { n } , S ^ { \omega x } = ( X ^ { \omega x } , Y , Z ) ^ { n } , S ^ { \omega z } = ( X , Y , Z ^ { \omega z } ) ^ { n }$ respectively. Furthermore, let the underlying probability spaces for the samples be $( \mathcal { S } , \mathcal { F } _ { \mathcal { S } } , \mathbb { P } ^ { S } )$ and analogously $( \mathcal { S } ^ { \omega } , \mathcal { F } _ { S ^ { \omega } } , \mathbb { P } ^ { S ^ { \omega } } )$ . We call ${ \mathfrak { X } } \times { \mathfrak { y } } \times { \mathfrak { z } }$ the embedded observation space, in contrast to the embedded sample space $S ^ { \omega _ { X } }$ and $S ^ { \omega }$ , and denote the collections of null and alternative distributions of (1) by $\mathcal { P } _ { 0 } = \left\{ ( \mathbb { P } ^ { X , Y , Z } ) ^ { \otimes n } \ : \ X \ \perp \perp Y \ | \ Z \right.$ under $\mathbb { P } ^ { X , Y , Z } \}$ , and $\mathcal { P } _ { 1 } = \big \{ ( P ^ { X , Y , Z } ) ^ { \otimes n } ~ : ~ X \mathcal { M } ~ Y ~ | ~ Z$ under $\mathbf { \bar { \mathbb { P } } } ^ { X , Y , Z } \}$ , respectively. On the embedded sample spaces we define the corresponding null classes on $S ^ { \omega }$ and $S ^ { \omega _ { X } }$ as

$$
\mathcal { P } _ { 0 } ^ { \omega } : = \{ Q \mathrm { ~ p r o b a b i l i t y ~ m e a s u r e ~ o n ~ } ( S ^ { \omega } , \mathcal { F } _ { S ^ { \omega } } ) : \ x ^ { \omega x , n } \perp \perp Y ^ { n } \mid Z ^ { \omega z , n } \mathrm { ~ u n d e r ~ } Q \} ,\tag{2}
$$

$$
{ \mathcal { P } } _ { 0 } ^ { \omega _ { X } } : = \{ Q \ \mathrm { p r o b a b i l i t y ~ m e a s u r e ~ o n } \ ( { \mathcal { S } } ^ { \omega _ { X } } , { \mathcal { F } } _ { { \mathcal { S } } ^ { \omega _ { X } } } ) : \ X ^ { \omega _ { X } , n } \ \bot \ \bot \ Y ^ { n } \ | \ Z ^ { n } \ \mathrm { u n d e r } \ Q \} .\tag{3}
$$

Both classes refer only to the embedded sample space and carry no reference to $\hat { \beta } , \hat { \gamma }$ or $ { \mathbb { P } } ^ { X , Y , Z }$ . Their elements are not required to be product measures: this is needed because the conditional law of $S ^ { \omega }$ given the embedding parameters is in general not a product measure when the parameters are estimated in sample. Both classes depend on $n ,$ which we suppress in the notation.

Throughout, we assume that $( \mathcal { X } , \mathcal { F } _ { \mathcal { X } } ) , ( \mathcal { V } , \mathcal { F } _ { \mathcal { Y } } ) , ( \mathcal { Z } , \mathcal { F } _ { \mathcal { Z } } )$ and $( { \mathfrak { X } } , { \mathcal { F } } _ { { \mathfrak { X } } } ) , ( 3 , { \mathcal { F } } _ { 3 } )$ are standard Borel. Since finite products of standard Borel spaces are standard Borel, this carries over to the sample spaces ${ \mathcal { S } } , S ^ { \omega _ { X } }$ and $S ^ { \omega }$ , and guarantees the existence of the regular conditional distributions used below. The assumption is satisfied in our setting: X and Z are the countable sets of finite strings over a finite alphabet equipped with the discrete $\sigma { - } \mathrm { f i e l d }$ while $\mathcal { Y } \subseteq \mathbb { R } \ \mathrm { o r } \ \mathcal { Y } = \{ 1 , \dots , L \}$ and $\mathfrak { X } \subseteq \mathbb { R } ^ { q } , \ 3 \subseteq \mathbb { R } ^ { p }$ carry their Borel $\sigma { \mathrm { - f i e l d s } }$ . No such assumption is placed on the parameter spaces $( B , { \mathcal { F } } _ { B } )$ and $( \mathcal { C } , \mathcal { F } _ { \mathcal { C } } )$ , which may be arbitrary measurable spaces.

To test hypothesis (1), we embed the texts with the embedding maps $\omega _ { X } ( \cdot , \hat { \beta } )$ and $\omega _ { Z } ( \cdot , \hat { \gamma } )$ , and test instead

$$
H _ { 0 } ^ { \omega } : \mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \in \mathcal { P } _ { 0 } ^ { \omega } \mathbb { P } ^ { \hat { \theta } } \mathrm { - a . s . } \qquad \mathrm { v s . } \qquad H _ { 1 } ^ { \omega } : \mathbb { P } ^ { \hat { \theta } } \big ( \mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \in \mathcal { P } _ { 0 } ^ { \omega } \big ) < 1 ,\tag{4}
$$

where $\hat { \theta } \ : = \ : ( \hat { \beta } , \hat { \gamma } )$ and $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } }$ denotes a regular conditional distribution of the embedded sample given the embedding parameters.

We propose the eCIT as the measurable function

$$
\varphi _ { \omega } : \mathcal S \times \mathcal B \times \mathcal C \times [ 0 , 1 ] \to \{ 0 , 1 \} , \qquad ( s , b , c , u ) \mapsto \psi \big ( \bar { \omega } ( s , b , c ) , u \big ) ,\tag{5}
$$

where $\bar { \omega }$ takes the whole sample and the estimators $\hat { \beta } , \hat { \gamma }$ as input and maps it onto $S ^ { \omega }$ through the embedding maps, so that $\bar { \omega } ( S , \hat { \beta } , \hat { \gamma } ) = S ^ { \omega }$ , and where $\psi : S ^ { \omega } \times [ 0 , 1 ] \to \{ 0 , 1 \}$ is a CIT testing (4), with $\varphi _ { \omega } = 0$ and $\varphi _ { \omega } = 1$ defining decisions for $H _ { 0 }$ and $H _ { 1 }$ in (1). The last argument is reserved for a random variable $U \sim \mathrm { U n i f } [ 0 , 1 ]$ , independent of $( S , { \hat { \beta } } , { \hat { \gamma } } )$ , which carries any randomization employed by the CIT, such as a sample split (cf. e.g., Shah and Peters, 2020). Since a randomized test may always ignore this argument, deterministic CITs are included. We suppress U from the notation, writing $\varphi _ { \omega } ( S )$ and $\psi ( S ^ { \omega } )$ , and let all probabilities of rejection be taken under the joint law of $( S , { \hat { \beta } } , { \hat { \gamma } } , U )$ ; analogously, $Q ( \psi = 1 )$ abbreviates (Q ⊗ ${ \mathrm { ~ U n i f } } [ 0 , 1 ] ) ( \psi = 1 )$ for Q a probability measure on $( S ^ { \omega } , { \mathcal { F } } _ { S ^ { \omega } } )$ . The test statistic of the eCIT is defined analogously as $T _ { \varphi _ { \omega } } ( s , b , c , u ) = T _ { \psi } ( \bar { \omega } ( s , b , c ) , u )$ for a test statistic $T _ { \psi } : S ^ { \omega } \times [ 0 , 1 ] \to \mathbb { R }$ of the CIT $\psi$

Throughout, the estimation procedure for the embedding parameters is fixed: it is a Markov kernel from $( \mathcal { S } , \mathcal { F } _ { \mathcal { S } } )$ to $( B \times \mathcal { C } , \mathcal { F } _ { B } \otimes \mathcal { F } _ { C } )$ , specified for every $s \in { \mathcal { S } }$ and not depending on $\mathbb { P } ^ { S }$ . The parameters remain random, while their conditional distribution given $S = s$ is held fixed across the null class. If $\hat { \theta }$ is a measurable function of the sample, as for the in-sample construction of $\omega _ { Z }$ by regressing $X ^ { \omega _ { X } , n }$ on $Z ^ { n }$ , its conditional distribution given $S = s$ places probability one on the corresponding value. If $\hat { \theta }$ is instead learned on data independent of S, its conditional distribution does not depend on s and carries the randomness of that fit. In general the conditional distribution both varies with s and is non-degenerate given $S = s .$ , as when the sample is combined with external data or the fit is randomly initialized. Since U is independent of $( S , { \hat { \beta } } , { \hat { \gamma } } )$ , the joint law of $( S , { \hat { \beta } } , { \hat { \gamma } } , U )$ is obtained from $\mathbb { P } ^ { S }$ , this kernel and the law of $U .$ , and is therefore a function of $\mathbb { P } ^ { S }$ alone for a fixed estimation procedure. We write $\mathbb { P } ^ { S } ( \varphi _ { \omega } = 1 )$ for the rejection probability under it.

Both embedding maps $\omega _ { X }$ and $\omega _ { Z }$ have to be chosen in the eCIT. For the former, we follow the approaches discussed in Simnacher et al. (2026). This choice afects only the power of the test as long as the learning approaches characterized in Simnacher et al. (2026, Corollary 3), e.g. sample splitting or externally trained embedding maps, are used. The choice of $\omega _ { Z }$ afects the T1E and the power of the eCIT. We will characterize suficient embedding maps $\omega _ { Z }$ in the next subsection. Furthermore, we evaluate the validity and power of the corresponding eCITs on semi-synthetic data in Subsection 4.2.

## 3.1 Suficient Embedding Maps

In this subsection, we take the embedding map ω<sub>X</sub> as given, so that a valid test for (6) is already a valid test for (1), and characterize the embedding maps $\omega _ { Z }$ for which we may test (4) instead. We first restate both hypotheses as CI statements conditioning on the estimated parameters $\hat { \beta }$ and $\hat { \gamma } .$ . We then give conditions under which $H _ { 0 } ^ { \omega _ { X } }$ implies $H _ { 0 } ^ { \omega }$ so that a valid test of the embedded hypothesis controls the T1E for the hypothesis on the texts: $\hat { \gamma }$ must not introduce conditional dependence (Assumption 1), and $Z ^ { \omega _ { Z } }$ must retain all information in $Z$ about either $Y$ or $X ^ { \omega _ { X } }$ (Theorem 3). We call such $\omega _ { Z }$ suficient. Finally, we give conditions for the converse implication (Theorem 9) and for equivalence of the two hypotheses (Corollary 10).

We can either embed X or Z first. We embed X first, because this simplifies embedding $Z ,$ as we will see in the following suficiency characterizations for the embedding map of $Z .$ Thus, we assume throughout this subsection that X was embedded onto $X ^ { \omega _ { X } }$ . Then, to

test (1), we test instead

$$
H _ { 0 } ^ { \omega x } : \mathbb { P } ^ { S ^ { \omega } x | \hat { \beta } } \in \mathcal { P } _ { 0 } ^ { \omega _ { X } } \mathbb { P } ^ { \hat { \beta } } \mathrm { - a . s . } \quad v s . \quad H _ { 1 } ^ { \omega _ { X } } : \mathbb { P } ^ { \hat { \beta } } \big ( \mathbb { P } ^ { S ^ { \omega _ { X } } | \hat { \beta } } \in \mathcal { P } _ { 0 } ^ { \omega _ { X } } \big ) < 1 ,\tag{6}
$$

where $\mathbb { P } ^ { S ^ { \omega } X | \hat { \beta } }$ denotes the conditional distributions of the sample after embedding X given the parameter $\hat { \beta }$ of the embedding map $\omega _ { X }$ . We assume that $\omega _ { X }$ was obtained as specified in Simnacher et al. (2026, Theorem 1, Corollary 3), e.g. learned on external data, through sample splitting or in an unsupervised manner. This ensures that a valid test for (6) is a valid test for (1), although the embedding map can afect the power of the test.

Given such a $X ^ { \omega _ { X } }$ , we are interested in embedding maps $\omega _ { Z }$ to test (4) instead of (6). To do this, we will assume throughout that $\hat { \gamma }$ does not introduce conditional dependence.

$$
\mathbf { A s s u m p t i o n 1 \textit { E i t h e r a } } ) \widehat \gamma \perp \perp Y ^ { n } \mid ( ( X ^ { \omega _ { X } } , Z ) ^ { n } , \widehat \beta ) \ o r b ) \widehat \gamma \perp \perp X ^ { \omega _ { X } , n } \mid ( ( Y , Z ) ^ { n } , \widehat \beta ) .
$$

This is similar to the assumption placed on the embedding map parameters $\hat { \beta }$ here and in the theoretical results of Simnacher et al. (2026). Furthermore, it holds trivially for $\hat { \gamma }$ learned via sample splitting, when $S$ denotes the holdout split, or on data independent of the test sample $S ,$ but has to be ensured for $\hat { \gamma }$ learned from S. In particular, branch $\mathrm { a ) }$ holds by construction whenever $\hat { \gamma }$ is a measurable function of $( X ^ { \omega _ { X } } , Z ) ^ { n }$ , since $\hat { \gamma }$ is then measurable with respect to the conditioning σ-field. This covers the in-sample construction of $\omega _ { Z }$ by regressing $X ^ { \omega _ { X } , n }$ on $Z ^ { n }$ , which is the natural use of the representation $X ^ { \omega _ { X } }$ obtained in the first step. In contrast to $\hat { \beta } .$ , this assumption is not suficient for a valid test for $( 4 )$ to yield a valid test for (6) and $( 1 )$ . Specifically, we need also that $Z$ does not contain information about $Y$ or $X ^ { \omega _ { X } }$ beyond $Z ^ { \omega _ { Z } }$ , which we ensure below.

First, we show that the hypothesis $H _ { 0 } ^ { \omega }$ in (4), formulated as an almost-sure statement of the conditional distribution $\mathbb { P } ^ { S ^ { \omega } | \hat { \gamma } , \hat { \beta } }$ , is equivalent to a single conditional independence statement in which $\hat { \gamma }$ and $\hat { \beta }$ are added to the conditioning set.

Lemma 2 Let $\mathbb { P } ^ { S ^ { \omega } | \hat { \gamma } , \hat { \beta } }$ denote a regular conditional distribution of $S ^ { \omega }$ given $( \hat { \gamma } , \hat { \beta } )$ . Then

$$
X ^ { \omega x , n } \perp \perp Y ^ { n } | ( Z ^ { \omega z , n } , \hat { \gamma } , \hat { \beta } ) \Longleftrightarrow \mathbb { P } ^ { S ^ { \omega } | \hat { \gamma } , \hat { \beta } } \in \mathcal { P } _ { 0 } ^ { \omega } \mathbb { P } ^ { \hat { \gamma } , \hat { \beta } } - a . s .
$$

The analogous equivalence holds for $S ^ { \omega _ { X } } , \ \hat { \beta } , \ \mathcal { P } _ { 0 } ^ { \omega _ { X } }$ and $Z ^ { n }$ in place of $S ^ { \omega } , ( \hat { \gamma } , \hat { \beta } ) , \mathcal { P } _ { 0 } ^ { \omega }$ and $Z ^ { \omega _ { Z } , n }$

All proofs are provided in Appendix A.

Both instances follow from the general Lemma 17 in Appendix A. Note that no assumption is placed on the spaces $_ { B , \mathcal { C } }$ in which $\hat { \beta } , \hat { \gamma }$ take their values, and that $\mathcal { P } _ { 0 } ^ { \omega }$ is not required to consist of product measures, so that Lemma 2 applies unchanged when $\hat { \gamma }$ is estimated in sample and the rows of $S ^ { \omega }$ are conditionally dependent.

Next, we show that for an embedding map $\omega _ { Z }$ containing all information within $Z$ of either, $Y$ or $X ^ { \omega _ { X } }$ , the null hypothesis (6) including texts $Z$ implies the null hypothesis (4) including only the corresponding embeddings $Z ^ { \omega _ { Z } }$

Theorem 3 (Z-text null implies Z-embedded null) Let $\hat { \gamma }$ satisfy Assumption 1, and

$$
\begin{array} { r } { ( i ) ~ X ^ { \omega _ { X } , n } \bot \bot ~ Z ^ { n } \mid ( \widehat { \gamma } , \widehat { \beta } , Z ^ { \omega _ { Z } , n } ) \quad o r \quad ( i i ) ~ Y ^ { n } \bot \bot ~ Z ^ { n } \mid ( \widehat { \gamma } , \widehat { \beta } , Z ^ { \omega _ { Z } , n } ) . } \end{array}
$$

Then $H _ { 0 } ^ { \omega _ { X } } \implies H _ { 0 } ^ { \omega }$

The conditions on $\hat { \beta } , \hat { \gamma }$ and $\omega _ { Z }$ used above are statements about the joint law of $( S , { \hat { \beta } } , { \hat { \gamma } } )$ and hence properties of $\mathbb { P } ^ { S }$ , so we may collect the null distributions satisfying them:

$$
\mathcal { P } _ { 0 } ^ { \mathrm { s u f f } } : = \Big \{ \mathbb { P } ^ { S } \in \mathcal { P } _ { 0 } : \hat { \beta } \perp \perp Y ^ { n } \mid ( X , Z ) ^ { n } , \mathrm { ~ A s s u m p t i o n ~ 1 ~ h o l d s , ~ a n d } 
$$

$$
\mathrm { ( i ) ~ o r ~ ( i i ) ~ o f ~ T h e o r e m ~ 3 ~ h o l d s } \Big \} .
$$

The class depends on the estimation procedure, $\omega _ { X } , \omega _ { Z }$ and $n ,$ which we suppress in the notation.

Definition 4 (Validity) Let $\alpha \in ( 0 , 1 )$ and let $\mathcal { P } \subseteq \mathcal { P } _ { 0 }$ . A test $\varphi _ { \omega }$ is valid for $\mathcal { P }$ at level α $i f$

$$
\operatorname* { s u p } _ { \mathbb { P } ^ { S } \in \mathcal { P } } \mathbb { P } ^ { S } \big ( \varphi _ { \omega } = 1 \big ) \leq \alpha .
$$

Analogously, for a class $\mathcal { Q }$ of probability measures on $( \boldsymbol { S } ^ { \omega } , \mathcal { F } _ { S ^ { \omega } } )$ , the test $\psi$ is valid for $\mathcal { Q }$ at level α if su $\quad \cup _ { Q \in \mathcal { Q } } Q ( \psi = 1 ) \leq \alpha$

Theorem 5 (Validity of the eCIT) Let $\alpha \in ( 0 , 1 )$ . If the CIT $\psi$ is valid for $\mathcal { P } _ { 0 } ^ { \omega }$ at level $\alpha _ { ; }$ then the eCIT $\varphi _ { \omega } = \psi \circ \bar { \omega }$ is valid for $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f } }$ at level α.

Remark 6 Unlike in Simnacher et al. (2026, Theorem 2), the embedded null cannot be stated for the marginal law of $S ^ { \omega }$ . There, the assumption on $\hat { \beta }$ allows $\hat { \beta }$ to be removed from the conditioning set by contraction and decomposition, because $X ^ { \omega _ { X } , n }$ appears on the left of the conditioning bar and $\hat { \beta }$ is a free element of the conditioning set. Here $Z ^ { \omega _ { Z } , n }$ is $\sigma ( Z ^ { n } , \hat { \gamma } )$ measurable and appears on the right, so $\hat { \gamma }$ cannot be removed: deleting it changes the σ-field relative to which the conditioning variable is defined. Marginalizing instead averages over $\hat { \gamma } ,$ and a mixture of conditionally independent laws need not be conditionally independent. The almost-sure formulation (4) and the conditional argument in Step $\mathcal { B }$ of the proof of Theorem $5$ are therefore needed.

Theorem 5 requires $\psi$ to be valid over all of $\mathcal { P } _ { 0 } ^ { \omega }$ , whose elements are not required to be product measures. Two regimes have to be distinguished. If $\hat { \gamma }$ is estimated on data independent of S — externally, or on a training split disjoint from the test sample — then conditionally on $\hat { \theta }$ the rows of $S ^ { \omega }$ remain i.i.d., and validity over the i.i.d. null class sufices.

Let $\mathcal { P } _ { 0 } ^ { \omega , \tilde { \otimes } }$ denote the n-fold products $M ^ { \otimes n }$ of probability measures M on the embedded observation space under which $X ^ { \omega _ { X } } \perp \perp Y \mid Z ^ { \omega _ { Z } }$ , which by Lemma 19 implies $\mathcal { P } _ { 0 } ^ { \omega , \otimes } \subseteq \mathcal { P } _ { 0 } ^ { \omega }$

Corollary 7 (Independently estimated embedding parameters) Let $\alpha \in ( 0 , 1 )$ and let $\psi$ be valid for $\mathcal { P } _ { 0 } ^ { \omega , \otimes }$ at level $\alpha$ . Then $\varphi _ { \omega }$ is valid at level α for $\{ \mathbb { P } ^ { S } \in \mathcal { P } _ { 0 } ^ { \mathrm { s u f f } } : \hat { \theta } \overset { . } { \bot } S \}$

If instead $\hat { \gamma }$ is estimated in sample, Lemma 19 no longer applies at the embedded level: the conditional laws are in general not product measures, and $\psi$ must be valid over the strictly larger class $\mathcal { P } _ { 0 } ^ { \omega }$ . We therefore state Theorem 3 and Theorem 5 for general $\hat { \gamma } _ { ; }$ , since the hypothesis-transfer argument requires no product structure, but restrict $\hat { \gamma }$ to embedding maps trained on a held-out corpus in the application in Subsection 4.3, so that Corollary $7$ applies and the conditional null class is the i.i.d. one.

While Theorem 3 shows that a suficient embedding map $\omega _ { Z }$ transfers the null hypothesis for texts $Z$ to the null hypothesis for embedded texts $Z ^ { \omega _ { Z } }$ , it does not establish equivalence of the two hypotheses. Thus, the embedded null (4) could be true while the text null (6) is false. For example, $Z ^ { \omega _ { Z } }$ could contain $Y _ { i }$ so that conditioning on $Z ^ { \omega _ { Z } , n }$ makes $Y ^ { n }$ measurable with respect to the conditioning set and (4) holds trivially. For the converse implication we have to exclude such cases, and we additionally need that $\hat { \gamma }$ may be removed from the conditioning set.

$$
\mathbf { A s s u m p t i o n \otimes } \ E i t h e r \ \mathfrak { a } \big ) \widehat { \gamma } \ \underline { { \sqcup } } \ Y ^ { n } \ \big | \ \big ( Z ^ { n } , \widehat { \beta } \big ) \ o r \ b \big ) \widehat { \gamma } \ \underline { { \sqcup } } \ X ^ { \omega _ { X } , n } \ \big | \ \big ( Z ^ { n } , \widehat { \beta } \big ) .
$$

Assumption 8 holds trivially whenever $\hat { \gamma }$ is learned on data independent of $S ,$ and whenever $\hat { \gamma }$ is a measurable function of $Z ^ { n }$ alone, for instance for unsupervised embedding maps of the source texts.

Theorem 9 (Embedded null implies text null) Let

$$
( i ) ~ X ^ { \omega x , n } \perp \perp ~ Z ^ { n } \mid ( \hat { \beta } , \hat { \gamma } , Z ^ { \omega z , n } , Y ^ { n } ) \quad a n d ~ A s s u m p t i o n ~ \beta a \ j , \quad o r
$$

$$
( i i ) ~ Y ^ { n } ~ \bot \bot ~ Z ^ { n } ~ \vert ~ ( \hat { \beta } , \hat { \gamma } , Z ^ { \omega _ { Z } , n } , X ^ { \omega _ { X } , n } ) ~ a n d ~ A s s u m p t i o n ~ \& b ) .
$$

Then $H _ { 0 } ^ { \omega } \implies H _ { 0 } ^ { \omega _ { X } }$

Finally, a joint CI assumption gives the equivalence:

Corollary 10 Let Assumptions 1 and 8 hold and let

$$
\begin{array} { r l } { ( * ) : } & { { } ( X ^ { \omega _ { X } , n } , Y ^ { n } ) \perp \perp Z ^ { n } \mid ( \hat { \beta } , \hat { \gamma } , Z ^ { \omega _ { Z } , n } ) . } \end{array}
$$

Then $H _ { 0 } ^ { \omega } \iff H _ { 0 } ^ { \omega _ { X } }$

Corollary 10 is a statement about the hypotheses, not about the power of a specific test: under (∗), the embedding map cannot map an alternative in $\mathcal { P } _ { 1 } ^ { \omega _ { X } }$ into the embedded null, so no alternative becomes undetectable in principle. It does not imply that the eCIT has the same power as a test of (6), since the embedding may reduce the efect size and project onto alternatives not detectable by the CIT applied to test (4).

Throughout this subsection we treated $\hat { \beta }$ as already learned, e.g. on an external dataset, and considered only the estimation of $\hat { \gamma }$ . This weakens the requirement on $\omega _ { Z }$ . Since $X ^ { \omega _ { X } , n }$ is $\sigma ( X ^ { n } , { \hat { \beta } } )$ -measurable, we have $\sigma ( X ^ { \omega _ { X } , n } ) ~ \subseteq \sigma ( X ^ { n } , { \hat { \beta } } )$ , and condition (i) of Theorem 3 requires only

$$
X ^ { \omega x , n } \perp \perp Z ^ { n } | ( { \hat { \beta } } , { \hat { \gamma } } , Z ^ { \omega z , n } ) \qquad { \mathrm { i n s t e a d ~ o f } } \qquad X ^ { n } \perp \perp Z ^ { n } | ( { \hat { \beta } } , { \hat { \gamma } } , Z ^ { \omega z , n } ) ,
$$

so that $Z ^ { \omega _ { Z } }$ need only retain the information in $Z$ about the representation $X ^ { \omega _ { X } }$ , and not about the whole text X. The coarser the embedding $\omega _ { X }$ , the weaker this condition becomes. However, if $\omega _ { X }$ does not represent the conditional association between X and Y in a form the CIT can detect, the test loses power (Simnacher et al., 2026). As X is embedded in any case to apply Simnacher et al. (2026), the choice is not whether to embed X but in which order: embedding X first weakens the suficiency requirement on $\omega _ { Z }$ without any additional loss of power, and may thereby facilitate T1E control.

## 3.2 Conditional Mean Independence

Throughout the paper, we focus on the projected covariance measure (PCM; Lundborg et al., 2024) to test hypothesis (4), an algorithm-agnostic CMIT which performs well for lowdimensional, vector-valued $Z$ and embedded images X (Simnacher et al., 2026). However, other conditional (mean) independence tests (e.g., Strobl et al., 2019; Dai et al., 2022; Williamson et al., 2023; Cai et al., 2025) can be applied to test (4).

Throughout this subsection we take Y to be R<sup>K</sup>-valued, so that conditional means of Y are defined: $K = 1$ in the continuous case $\mathcal { V } \subseteq \mathbb { R }$ , and $K = L - 1$ in the categorical case $\mathcal { V } = \{ 1 , \ldots , L \}$ , where $Y$ is represented by its vector of indicators of the non-baseline classes (cf. Remark 16 and Appendix D). We assume throughout that $\mathbb { E } \| Y _ { 1 } \| < \infty$ , where $\| \cdot \|$ is any norm on $\mathbb { R } ^ { K }$ , all of which are equivalent, and which reduces to the absolute value for $K = 1$ . The assumption holds automatically in the categorical case and, since the rows of S are i.i.d., gives $\mathbb { E } \| Y _ { i } \| < \infty$ for all $i \leq n .$ , so that the conditional means of $Y _ { i }$ appearing below are well defined.

We test (4) by applying the PCM, a CMIT, which tests instead

$$
H _ { 0 } ^ { \omega , \mathrm { m e a n } } : \mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \in \mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } } \mathbb { P } ^ { \hat { \theta } } \mathrm { - a . s . \quad v s . \quad H _ { 1 } ^ { \omega , \mathrm { m e a n } } : \mathbb { P } ^ { \hat { \theta } } \big ( \mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \in \mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } } \big ) < 1 , }\tag{7}
$$

where

$$
\begin{array} { r l } & { \mathcal { P } _ { 0 } ^ { \omega , \operatorname* { m e a n } } : = \Big \{ Q \mathrm { ~ p r o b a b i l i t y ~ m e a s u r e ~ o n ~ } ( S ^ { \omega } , \mathcal { F } _ { S ^ { \omega } } ) : \mathbb { E } _ { Q } \| Y _ { i } \| < \infty \mathrm { ~ a n d ~ } } \\ & { \qquad \mathbb { E } _ { Q } \big [ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { \omega _ { Z } , n } \big ] = \mathbb { E } _ { Q } \big [ Y _ { i } \mid Z ^ { \omega _ { Z } , n } \big ] Q \mathrm { - a . s . ~ f o r ~ a l l ~ } i \leq n \Big \} . } \end{array}
$$

Since CI implies conditional mean independence and $Y _ { i }$ is a measurable function of $Y ^ { n }$ every integrable $Q \in \mathcal { P } _ { 0 } ^ { \omega }$ lies in $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } }$ . Integrability is not implied by $\mathcal { P } _ { 0 } ^ { \omega }$ and has to be required separately, but it is not a restriction here: the embedding maps act on X and Z only, so the standing assumption $\mathbb { E } \| Y _ { 1 } \| < \infty$ carries over to the embedded sample space and the conditional law $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } }$ is integrable almost surely. Whenever this conditional law lies in $\mathcal { P } _ { 0 } ^ { \omega }$ it therefore lies in $\mathcal { P } _ { 0 } ^ { \omega , m e a n }$ , and since Step 3 of the proof of Theorem 5 uses only that it lies almost surely in the class over which ψ is valid, Theorem 5 applies unchanged. However, power against alternatives in $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } } \setminus \mathcal { P } _ { 0 } ^ { \omega }$ can be lost (Lundborg et al., 2024), i.e. those in which the summary carries information about Y beyond the speech without shifting its conditional mean.

As in the previous subsection, we embed X first and characterize afterwards mean suficiency for the embedding of Z. Thus, we define

$$
H _ { 0 } ^ { \omega x , \mathrm { m e a n } } : \mathbb { P } ^ { S ^ { \omega x } | \hat { \beta } } \in \mathcal { P } _ { 0 } ^ { \omega x , \mathrm { m e a n } } \mathbb { P } ^ { \hat { \beta } } \mathrm { - a . s . \quad v s . \quad } H _ { 1 } ^ { \omega x , \mathrm { m e a n } } : \mathbb { P } ^ { \hat { \beta } } \big ( \mathbb { P } ^ { S ^ { \omega x } | \hat { \beta } } \in \mathcal { P } _ { 0 } ^ { \omega x , \mathrm { m e a n } } \big ) < 1 ,\tag{8}
$$

where

$$
\begin{array} { r l } & { \mathcal { P } _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } } : = \Big \{ Q \mathrm { ~ p r o b a b i l i t y ~ m e a s u r e ~ o n ~ } ( \mathcal { S } ^ { \omega _ { X } } , \mathcal { F } _ { S ^ { \omega _ { X } } } ) : \mathbb { E } _ { Q } \| Y _ { i } \| < \infty \mathrm { ~ a n d ~ } } \\ & { \qquad \mathbb { E } _ { Q } \big [ Y _ { i } \mid { X } ^ { \omega _ { X } , n } , Z ^ { n } \big ] = \mathbb { E } _ { Q } \big [ Y _ { i } \mid Z ^ { n } \big ] \ Q \mathrm { - a . s . ~ f o r ~ a l l ~ } i \leq n \Big \} . } \end{array}
$$

Because (8) is stated only for the conditional mean, the assumption ${ \hat { \beta } } \perp \perp Y ^ { n } \mid ( X , Z ) ^ { n }$ under which Simnacher et al. (2026) transfer $H _ { 0 }$ to $H _ { 0 } ^ { \omega _ { X } }$ can be weakened. Specifically, $\hat { \beta }$ is only required to introduce no conditional mean dependence:

Assumption 11 $\operatorname { \mathbb { E } } \big [ Y _ { i } \mid X ^ { n } , Z ^ { n } , \hat { \beta } \big ] = \operatorname { \mathbb { E } } \big [ Y _ { i } \mid X ^ { n } , Z ^ { n } \big ]$ a.s. for all $i \leq n$

Assumption 11 is implied by ${ \hat { \beta } } \perp \perp Y ^ { n } \mid ( X , Z ) ^ { n }$ and holds in particular whenever $\hat { \beta }$ is a measurable function of $( X , Z ) ^ { n }$ or is learned on data independent of $S$ (Simnacher et al., 2026, Corollary 3). Next, we show that this assumption is suficient to transfer the null hypothesis (1) including X and $Z$ to the null hypothesis $H _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } }$ including only a representation of $X$

Theorem 12 (Text null implies X-embedded mean null) Let Assumption 11 hold. Then $H _ { 0 } \implies H _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } }$

In addition, we can weaken the suficiency condition on the embedding of $Z$ if only the mean-level null has to be transferred. Write

$$
m _ { i } : = \mathbb { E } \big [ Y _ { i } \mid Z ^ { n } , \hat { \beta } \big ] , \qquad i = 1 , \ldots , n ,
$$

for the regression of $Y _ { i }$ on the source texts and the embedding parameter ${ \hat { \boldsymbol { \beta } } } .$ . When $\hat { \beta }$ is deterministic or learned on data independent of S, it is independent of $( Y _ { i } , Z ^ { n } )$ and $m _ { i } = \mathbb { E } [ Y _ { i } \mid Z ^ { n } ]$ is the regression of $Y _ { i }$ on the source texts alone; $\hat { \beta }$ is retained in general because an in-sample $\hat { \beta }$ is a function of $( X , Z ) ^ { n }$ and can then carry information about $Y _ { i }$ beyond $Z ^ { n }$ . We first state a mean-level version of Assumption 1a), which restricts $\hat { \gamma }$ to introduce no conditional mean dependence and holds in particular whenever $\hat { \gamma }$ is learned independently of $S$ or is a measurable function of $( X ^ { \omega _ { X } } , Z ) ^ { n }$ . We then characterize when $H _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } }$ transfers to $H _ { 0 } ^ { \omega , \mathrm { m e a n } }$ , and show that it does so whenever $m _ { i }$ is $\sigma ( Z ^ { \omega _ { Z } , n } , \hat { \theta } )$ measurable, that is, whenever $Z$ carries no information about the conditional mean of Y beyond $Z ^ { \omega _ { Z } }$ . This replaces the requirement that $Z ^ { \omega _ { Z } }$ preserve the entire conditional law of $Y$ or $X ^ { \omega _ { X } }$ given $Z$ by a requirement on a single regression function.

Assumption 13 $\operatorname { \mathbb { E } } [ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \theta } ] = \operatorname { \mathbb { E } } [ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \beta } ]$ a.s. for all $i \leq n$

Assumption 1a) implies Assumption 13, which restricts $\hat { \gamma }$ in the same way as Assumption 11 restricts $\hat { \beta } .$ . There is no analogue of Assumption 1b), because conditional mean independence is not symmetric in its two arguments.

Theorem 14 (Z-text mean null implies Z-embedded mean null) Let Assumption 13 hold, and let

$$
( i i ^ { \prime } ) \qquad m _ { i } = \mathbb { E } \big [ m _ { i } \mid Z ^ { \omega _ { Z } , n } , \widehat { \theta } \big ] \quad a . s . \ f o r \ a l l \ i \leq n ,
$$

that is, let each $m _ { i }$ be almost surely equal to $\textit { a } \sigma ( Z ^ { \omega z , n } , \hat { \theta } )$ -measurable random variable. Then $H _ { 0 } ^ { \omega x , \mathrm { m e a n } } \implies H _ { 0 } ^ { \omega , \mathrm { m e a n } }$

The proof of Theorem 14 shows that $\mathbb { E } [ Y _ { i } \mid Z ^ { \omega _ { Z } , n } , \hat { \theta } ] = \mathbb { E } [ m _ { i } \mid Z ^ { \omega _ { Z } , n } , \hat { \theta } ]$ , so $( \mathrm { i i } ^ { \prime } )$ holds if and only if $m _ { i } = \mathbb { E } [ Y _ { i } \mid Z ^ { \omega _ { Z } , n } , \hat { \theta } ]$ a.s. for all $i \leq n ,$ , that is, if and only if the regression of $Y _ { i }$ on the embedded source texts coincides with its regression on the source texts. Mean suficiency of $\omega _ { Z }$ is thus a condition on the regression of $Y$ on $Z$ alone and can be assessed without reference to $X ^ { \omega _ { X } }$ . Condition $( \mathrm { i i } ^ { \prime } )$ is an exact measurability requirement and is not plausible for embedding maps learned from data, which approximate e.g. the regression of

$Y$ on the source texts rather than reproducing it. What matters in practice is whether the residual $m _ { i } - \mathbb { E } [ m _ { i } \mid Z ^ { \omega _ { Z } , n } , \hat { \theta } ]$ is negligible relative to $n ^ { - 1 / 2 }$ , since a fixed nonzero residual enters the test statistic as a location shift growing with $\sqrt { n }$ . It is nevertheless approachable, because $m _ { i }$ need not depend on all of $Z _ { i } { \mathrm { : } }$ if $m _ { i } = g ( t ( Z _ { i } ) )$ for a map t and a function $^ { g , }$ then $( \mathrm { i i } ^ { \prime } )$ holds for every $\omega _ { Z }$ for which $t ( Z _ { i } )$ is $\sigma ( Z ^ { \omega _ { Z } , n } , \hat { \theta } )$ -measurable. In the simulation study of Subsection 4.2, t is a linear score in the embedding used to generate $Y .$ , so every eCIT whose $Z ^ { \omega _ { Z } }$ retains that embedding satisfies $( \mathrm { i i } ^ { \prime } )$ exactly; in the application in Subsection 4.3, t is unknown and we approximate it by a learned representation of the speech, the masked mean of the last hidden layer of a model predicting Y from $Z .$ Appending further embedding maps enlarges $\sigma ( Z ^ { \omega _ { Z } , n } )$ and can therefore only reduce the residual, which motivates the concatenated conditioning sets used in both. Whether the residual is negligible at a given sample size depends on the dataset and the embedding maps, and we propose to assess it with the simulation design of Subsection 4.2.

The null class $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } }$ constrains the conditional mean of $Y _ { i }$ given the whole embedded sample, whereas the PCM fits regressions of $Y _ { i }$ only on $( X _ { i } ^ { \omega _ { X } } , Z _ { i } ^ { \omega _ { Z } } )$ and on $Z _ { i } ^ { \omega _ { Z } }$ . The two agree whenever the rows of $S ^ { \omega }$ are i.i.d. conditionally on ${ \widehat { \theta } } ,$ which is $\mathrm { e . g . }$ the case when the embedding parameters are estimated independently of $S _ { i }$ , as in our application. Let $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \otimes }$ denote the n-fold products $M ^ { \otimes n }$ of probability measures M on the embedded observation space $( { \mathfrak { X } } \times { \mathcal { Y } } \times { \mathfrak { Z } } )$ under which $\mathbb { E } _ { M } \Vert Y \Vert < \infty$ and $\mathbb { E } _ { M } [ Y \mid X ^ { \omega _ { X } } , Z ^ { \omega _ { Z } } ] = \mathbb { E } _ { M } [ Y \mid$ $Z ^ { \omega _ { Z } } ]$ , which by Lemma 20 satisfies $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \otimes } \subseteq \mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } }$ . Then, we collect the null distri butions satisfying the conditions used above in

$\mathcal { P } _ { 0 } ^ { \mathrm { s u f f , m e a n } } : = \left\{ \mathbb { P } ^ { S } \in \mathcal { P } _ { 0 } \right.$ : Assumptions 11 and 13 hold, and (ii<sup>′</sup>) of Theorem 14 holdso,

which depends on the estimation procedure, ω<sub>X</sub>, ω<sub>Z</sub> and n, suppressed in the notation as for $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f } }$ . Step 3 of the proof of Theorem 5 uses only that $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } }$ lies almost surely in the class over which ψ is valid, so a $\psi$ valid for $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } }$ yields an eCIT valid for $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f , m e a n } }$ . Then, we can derive the analog of Corollary $7$ for the conditional mean independence hypothesis.

Corollary 15 (Independently estimated embedding parameters, mean version) Let $\alpha \in ( 0 , 1 )$ and let $\psi$ be valid for $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \otimes }$ at level α. Then $\varphi _ { \omega }$ is valid at level α for $\{ \mathbb { P } ^ { S } \in \mathcal { P } _ { 0 } ^ { \mathrm { s u f f , m e a n } } : \hat { \theta } \perp \perp S \}$

Remark 16 (Categorical Y) For Y finite and Y represented by its vector of category indicators, $\operatorname { \mathbb { E } } _ { Q } [ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { \omega _ { Z } , n } ] = \operatorname { \mathbb { E } } _ { Q } [ Y _ { i } \mid Z ^ { \omega _ { Z } , n } ]$ determines the conditional law of $Y _ { i }$ and is therefore equivalent to $X ^ { \omega _ { X } , n } ~ \bot \bot ~ Y _ { i } ~ \bot ^ { \omega _ { Z } , n }$ , for each $i \leq n$ . For product measures these row-wise statements lift to the joint one by Lemma $^ { 1 9 , }$ so $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \mathrm { \bar { \otimes } } } = \mathcal { P } _ { 0 } ^ { \omega , \otimes }$ , and $( i i ^ { \prime } )$ is the row-wise counterpart of condition $( i i )$ , the two coinciding for product measures for the same reason. For general Q the mean class is the larger one, since row-wise conditional independence need not imply $X ^ { \omega _ { X } , n } ~ \bot \bot Y ^ { n } \mid Z ^ { \omega _ { Z } , n }$ . In our application $\boldsymbol { \hat { \theta } } \perp \perp \boldsymbol { S }$ , so Corollary 15 applies with the product class and nothing is lost by using a CMIT.

Finally, we note that the PCM guarantees only asymptotic uniform validity, so the eCIT is asymptotically valid for mean suficient embedding maps. In our application the embedding parameters are estimated independently of $S ,$ , so Corollary 15 is the relevant statement and the null class is $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \otimes }$ , whose elements are the i.i.d. laws over which the PCM is asymptotically uniformly valid. The conditional argument in Step 3 of the proof of Theorem 5 transfers uniform asymptotic validity, since the bound holds simultaneously for all $Q \in \mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \otimes }$ and hence for the realized conditional law $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \mathbf { i }$ ; pointwise asymptotic validity would not sufice, as the conditioning value $\hat { \theta }$ varies with n. For $\hat { \gamma }$ estimated in sample the conditional laws need not be products, and validity over the larger class $\mathcal { P } _ { 0 } ^ { \omega }$ ,mean would be required, which the PCM is not known to provide.

## 4 Application

We apply the eCITs to German Parliament speeches and their summaries. First, we provide a description of the data in Subsection 4.1. Then, we evaluate the performance of the eCITs on semi-synthetic data based on the speech-summary pairs in Subsection 4.2. While our simulation study evaluates the T1E control and power of the eCITs on this specific dataset and for the LLM summaries, the simulation design can be used to evaluate if the eCITs can control T1E and have power against alternatives of interest for other datasets with Z, X pairs before applying the eCITs. On other datasets and their underlying data generating processes, the embedding maps used, their collinearity within and between Z and X, and the corresponding separation for categorical $Y$ can afect the performance of the eCITs and particularly the T1E control with respect to the suficiency of $Z ^ { \omega _ { Z } }$ and the convergence of the nuisance regressions required by the PCM or other CITs. Finally, we apply in Subsection 4.3 the eCITs to the full dataset, including speech-summary pairs together with the protected attributes of the speakers’ gender and faction.

## 4.1 Data

For the source texts, we use the German Parliament speeches provided in Richter et al. (2020). We clean and merge speeches as described in Appendix B.1. The resulting corpus comprises 274,598 merged speeches from electoral terms 1–20, dated 12 September 1949– 20 May 2022. Roles are dominated by Member of Parliament (MP; 185,009) and Presidium of Parliament (34,392), plus government speakers (Secretary of State 28,547; Minister 19,892; Chancellor 1,559), guests (5,184), and 15 speeches with unlabeled role. Merged speech length has median 382 words (mean 605; IQR 122–794), and there are 4,030 unique speaker identifiers.

For the analysis, we further restrict this corpus to the 274,579 speeches for which a Gemma 4 summary was generated, with 19 dropped speeches because they did not produce a summary. Among merged MP speeches, the main parliamentary groups are CDU/CSU (59,041), SPD (54,752), FDP (27,677), Gr¨une (21,369), DIE LINKE. (10,006), PDS (4,338), and AfD (2,742), plus smaller and historical groups (KPD, DP, GB/BHE, Zentrum, and others) and speakers without faction (Fraktionslos). We drop unlabeled factions and keep only factions at least as frequent as AfD, yielding seven factions (CDU/CSU, SPD, FDP, Gr¨une, DIE LINKE., PDS, $\mathrm { A f D } ; n = 1 7 9 , 9 2 5 )$ . For the analysis including the speakers faction, we randomly draw a held-out test set of size $n _ { \mathrm { t e s t } } = 5 3 , 9 7 8$ for the eCITs while the remainder $( n _ { \mathrm { t r a i n } } = 1 2 5 , 9 4 7 )$ is used to train embedding maps. For the second attribute we use the speaker’s gender as recorded in the corpus, which takes two values (L = 2). Here, we keep electoral terms 10–20 with non-missing speaker gender $( n = 1 6 9 , 4 6 3 ; n _ { \mathrm { t r a i n } } = 1 1 8 , 6 2 4 ,$ $n _ { \mathrm { t e s t } } = 5 0 { , } 8 3 9 )$ , of which around 71% are by male and 29% by female speakers. For the semi-synthetic simulation study, we use the intersection of the faction and gender train/test splits $( n _ { \mathrm { t r a i n } } = 6 1 , 8 0 7$ to train embeddings, and $n _ { \mathrm { t e s t } } = 1 1 { , } 3 5 7$ to apply the eCITs).

## 4.2 Simulation

We examine the empirical performance of the eCIT on simulated data. To evaluate whether the eCIT controls the T1E and has power against relevant alternatives on the dataset of interest, we resample Z, X pairs and generate Y from a pre-specified embedding of Z (under the null hypothesis) or from prespecified embeddings of $Z$ and $X$ (under the alternative hypothesis). Since we apply the PCM, a CMIT, and the embedding maps are trained on a holdout set disjoint from the test sample, so that ${ \hat { \theta } } \perp \perp S ,$ the relevant statements are Theorem 14 and Corollary 15, and the condition on $\omega _ { Z }$ is mean suficiency, i.e. (ii<sup>′</sup>). For the binary and multinomial outcomes this coincides with suficiency by Remark 16, and we write suficient for mean suficient below.

## 4.2.1 Design

We compare the proposed eCITs across several combinations of fixed and learned embedding maps for X and Z. Summaries are generated with Gemma 4 (Gemma Team, 2026) throughout, and additionally with Qwen 3.6 (Qwen Team, 2026) for the application in Subsection 4.3 (Appendix B.1).

Data generating mechanisms (DGMs): We consider 12 DGMs across three outcome types, two generating embedders, under the null and alternative hypothesis. For all, we vary the sample size over $n = 1 , 0 0 0 , 5 , 0 0 0 , 1 0 , 0 0 0$ . These DGMs extend the simulation studies of Simnacher et al. (2026), which evaluate CITs for low-dimensional Z and images X, to high-dimensional texts $Z$ and X.

For the source texts $Z ,$ we resample repeatedly n speeches from the test set (cf. Subsection 4.1). We obtain the corresponding summaries X using the Gemma 4 LLM. This provides speech-summary pairs $( Z _ { i } , X _ { i } )$ as used in the real-world application. We embed each $Z _ { i }$ and $X _ { i }$ with the Jina embedder for text classification (Akram et al., 2026) and the Qwen embedder for general tasks (Zhang et al., 2025), truncated with their Matryoshka representations (Kusupati et al., 2022) to $d = 3 2$ dimensions. Depending on the DGM, we generate Y from one of the two embedders, whose maps for the speech and the summary we denote by $\widetilde { \omega } _ { Z }$ and ${ \widetilde { \omega } } _ { X } .$ , i.e. $\widetilde { \omega } _ { Z } , \widetilde { \omega } _ { X } \in$ {Jina, Qwen}, so that $z _ { i } ^ { \tilde { \omega } _ { Z } } , x _ { i } ^ { \tilde { \omega } _ { X } } \in \mathbb { R } ^ { 3 2 }$

We draw two directions $B _ { Z } , B _ { X } \ \sim \ N ( 0 _ { 3 2 } , I _ { 3 2 } )$ once and compute the unnormalized scores

$$
\begin{array} { r } { \tilde { s } _ { Z , i } = \big ( z _ { i } ^ { \widetilde { \omega } z } \big ) ^ { \top } B _ { Z } , \qquad \tilde { s } _ { X , i } = \big ( x _ { i } ^ { \widetilde { \omega } x } - A ^ { \top } z _ { i } ^ { \widetilde { \omega } z } \big ) ^ { \top } B _ { X } , } \end{array}
$$

where A is the coeficient matrix of the regression of $x ^ { \widetilde { \omega } _ { X } }$ on $z ^ { \widetilde { \omega } z }$ , fitted on the training split. Then, the speech and summary scores used in the DGMs are $s _ { Z , i } = \tilde { s } _ { Z , i } / \widehat { \mathrm { s d } } ( \tilde { s } _ { Z } )$ and $s _ { X , i } = \tilde { s } _ { X , i } / \widehat { \mathrm { s d } } ( \tilde { s } _ { X } )$ , so that both have unit variance. The column standardization of the embeddings and the two standard deviations are likewise computed on the training split, so that all of these are held fixed across replications. Thus $s _ { Z }$ is a one-dimensional projection of the speech representation $z ^ { \widetilde { \omega } z }$ , and $s _ { X }$ a one-dimensional projection of the part of the summary representation $x ^ { \widetilde { \omega } _ { X } }$ that the speech representation does not explain. Given $s _ { Z }$ and $s _ { X }$ , we generate Y under $c = 0 ~ \mathrm { ( C I ) }$ and $c = 1$ (no CI) according to the outcome type:

$$
\begin{array} { r l } { \mathrm { c o n t i n u o u s : } } & { Y _ { i } = s _ { Z , i } + c s _ { X , i } + \varepsilon _ { i } , \quad \varepsilon _ { i } \sim N ( 0 , \sigma ^ { 2 } ) , } \\ { \mathrm { b i n a r y : } } & { Y _ { i } \sim \mathrm { B e r n o u l l i } ( \mathrm { e x p i t } ( \eta _ { i } ) ) , \quad \eta _ { i } = \alpha _ { Z } s _ { Z , i } + c \alpha _ { X } s _ { X , i } + b , } \\ { \mathrm { m u l t i n o m i a l : } } & { Y _ { i } \sim \mathrm { C a t e g o r i c a l } ( \mathrm { s o f t m a x } ( \eta _ { i } ) ) , \quad Y _ { i } \in \{ 0 , \ldots , L - 1 \} , } \end{array}\tag{9}
$$

where in the multinomial case the reference class has $\eta _ { i , 0 } = 0$ and the remaining $L - 1$ logits are fixed linear combinations of $s _ { Z , i }$ and $c s _ { X , i } .$ chosen so that the two scores load on the classes in diferent directions, with the softmax taken over all L logits. We set $L = 5$ The coeficients $\alpha _ { Z } , \alpha _ { X }$ , the class weights, the intercept b, and the noise variance $\sigma ^ { 2 }$ of the continuous outcome are given in Appendix B.3. Under the null, Y depends on the speech only through $s _ { Z , i }$ , and hence only through $z ^ { \widetilde { \omega } _ { Z } }$

Embeddings in eCITs: First, we apply the eCITs with only the embedding maps used to generate Y (oracle). Next, we apply the eCITs with additional embedding maps, including the BGE embedder (Chen et al., 2024), the Llama Nemotron embedder (Moreira et al., 2024; NVIDIA, 2025), the Jina or Qwen embedder, depending on which was not used in the corresponding Y generation, and learned embeddings. We refer to the eCITs combining the generating embedding with further embedding maps as diluted, and contrast them with those omitting the generating embedding. We selected the four pre-trained embedding maps because they cover embedding models based on diferent architectures, they allow for the long context of the speeches, and they performed best in a recent meta study of embedding maps for text across several tasks (Gjorgjevikj et al., 2026). For the trained embedding maps, we predict Y separately from the speech $Z$ and from the summary X using a text prediction model with 134 million parameters and classification or regression head (Wunderle et al., 2025) on a large holdout set of 61,807 speech-summary pairs together with the generated Y in the corresponding DGM.

Target and performance measures: Performance is measured in terms of the T1E and power. T1E and power are estimated from the rejection rates

$$
\widehat { R R } = \frac { 1 } { n _ { \mathrm { s i m } } } \sum _ { l = 1 } ^ { n _ { \mathrm { s i m } } } \mathbb { 1 } \{ p _ { l } \leq \alpha \}
$$

under the null $( c = 0 )$ and alternative $( c = 1 )$ hypothesis, respectively, for a significance level of $\alpha = 0 . 0 5$ , over $l = 1 , \ldots , n _ { \mathrm { s i m } }$ random data generations and using p-values $p _ { l }$ for each DGM. We choose $n _ { \mathrm { s i m } } = 5 0$ for computational reasons, resulting at level $\alpha = 0 . 0 5$ in a Monte Carlo (MC) standard error (Morris et al., 2019, sec. 5.3) of

$$
\operatorname { M C } \operatorname { S E } ( \widehat { R R } ) = \sqrt { \frac { \widehat { R R } ( 1 - \widehat { R R } ) } { n _ { \mathrm { s i m } } } } \approx \sqrt { \frac { 0 . 0 5 ^ { * } 0 . 9 5 } { 5 0 } } \approx 0 . 0 3 1 .
$$

## 4.2.2 Results

Figure 2 shows the rejection rates for binary and multinomial Y generated from the Jina embeddings. The oracle eCIT, which uses the embedding maps $\widetilde { \omega } _ { Z } , \widetilde { \omega } _ { X }$ that generate $Y ,$ satisfies condition (ii<sup>′</sup>) of Theorem 14 by construction, since under the null $m _ { i }$ depends on $Z _ { i }$ only through $s _ { Z , i }$ , which is a linear function of $z _ { i } ^ { \mathrm { J i n a } }$ . It therefore isolates the behavior of the PCM on this DGM, that is, whether the test controls the T1E and has power for the embedded hypothesis, and the results show that it holds the nominal level and rejects in every replication under the alternative already at $n = 1 , 0 0 0$ . It thus serves as the reference for the remaining eCITs: deviations from it are attributable to the embedding maps and the PCM applied to the larger dimension of the corresponding combined embeddings.

![](images/5df4f2418b886f648e272e145038bf6deb2da4a0a85b29649f690b0fc7d13c78.jpg)  
Figure 2: The rejection rates of the eCIT for binary and multinomial $Y$ generated under the null (first row) and alternative (second row) from the Jina embeddings $Z ^ { J i n a }$ and $X ^ { J i n a }$ . The eCIT is applied with the Jina embeddings (oracle), the specified diluted embeddings, including additionally learned, BGE, Qwen, and Llama embeddings, as well as the same combinations excluding the Jina embedding. For the PCM, the nuisance regressions on the training dataset are $L _ { \mathrm { { 2 } \mathrm { { - p e n a l i z e d } } } }$ and those on the evaluation set unpenalized; for categorical $Y$ the regressions of $Y$ are multinomial logistic and those of the fitted probabilities and of $\hat { f }$ are least squares (Appendix D.1). The dashed line marks the nominal level $\alpha = 0 . 0 5$ , the shaded area marking two MC standard errors around it. The continuous outcome and the corresponding QQ-plots are shown in Figures 3 and 4.

Under the null and the Jina DGM, all eCITs stay within or close to two MC standard errors of $\alpha = 0 . 0 5$ for binary and multinomial ${ \cal Y } ,$ with a maximum rejection rate of around 13% at $n = 1 0 \small { , } 0 0 0$ This holds also for the eCITs that omit the generating embedding $Z ^ { J i n a }$ entirely, i.e. for conditioning sets for which suficiency does not hold by construction, although for some of them, e.g. $Z ^ { \omega _ { Z } } = \mathrm { { B G E + L l a m a + Q w e n } }$ and $Z ^ { \omega _ { Z } } = \mathrm { { l e a r n e d } } .$ , the rejection rate increases mildly over the sample sizes considered. The QQ-plots in Figure 4 show the same picture: for the categorical outcomes the p-values track the uniform diagona closely at $n = 1 , 0 0 0$ , with departures for individual eCITs at $n = 1 0 { , } 0 0 0$ that remain small. In general, level control does not require $Z ^ { \omega _ { Z } }$ to contain the generating embedding, only that it retain enough of $s _ { Z }$ for the transfer of the null hypothesis on the level of the texts to the one on the level of the embedded texts (Theorem 14) to hold approximately. Here, the part of $s _ { Z }$ that is lost by omitting $Z ^ { \mathrm { { J i n a } } }$ is still negligible at $\sqrt { n }$ scale at the sample sizes considered.

Under the alternative and the Jina DGM, the combination of the embeddings of $X ^ { \omega _ { X } }$ and $Z ^ { \omega _ { Z } }$ afects the power of the corresponding eCITs. The ordering of the eCITs follows from how much of $s _ { X }$ is retained in $X ^ { \omega _ { X } }$ . At $n = 1 , 0 0 0$ no non-oracle arm exceeds around 10%. At $n = 5 { , } 0 0 0$ the arms that contain the generating embeddings $Z ^ { J i n a } , X ^ { J i n a }$ reach 55– $9 7 \%$ for binary and 56–90% for multinomial Y , whereas the eCITs omitting the generating embeddings have lower or similar power. The eCIT with $Z ^ { \omega _ { \cal Z } } = \mathrm { l e a r n e d } + \mathrm { B G E } + \mathrm { Q w e n } +$ Llama and $X ^ { \omega _ { X } } = \mathrm { { \ l e a r n e d } + Q w e n + L l a m a }$ achieves the lowest power for binary and multinomial Y. Here, the BGE embedding in $Z ^ { \omega _ { Z } }$ is not included in $X ^ { \omega _ { X } }$ , lowering the detectable signal. Adding embeddings to $X ^ { \omega _ { X } }$ can thus recover power, depending on how well the available maps span the conditional dependence of the DGM. Finally, eCITs for multinomial Y are less powerful than for binary Y at the largest sample size.

The Qwen DGM separates including and omitting the generating embedding $Z ^ { Q w e n }$ and the corresponding (in)suficiency of the embedding of $Z _ { i }$ more clearly (Figure 5). The eCITs with suficient embeddings containing $Z ^ { Q w e n } , \mathrm { i . e . }$ . the oracle and the three diluted arms, stay within or close to two MC standard errors of $\alpha = 0 . 0 5$ at every sample size, with rejection rates of at most 12% for binary and 11% for multinomial Y at $n = 1 0 { , } 0 0 0$ and their power reaches at least 90% at $n = 1 0 { , } 0 0 0 .$ . Diluting the generating embedding with further embedding maps therefore still leads to tests controlling T1E, and only raises the sample size at which full power is reached, as under the Jina DGM. This also indicates that the PCM performs well on this dataset and the DGMs, also in the higher-dimensional conditioning sets, as long as the embedding of $Z$ remains suficient. The T1E inflation of eCITs that omit $Z ^ { Q w e n }$ increases with the sample size. For example, for binary Y and $Z ^ { \omega _ { Z } } = \mathrm { { B G E } + \mathrm { { J i n a } + } }$ Llama the T1E rises from 6% at n = 1,000 to 48% at $n =$ 5,000 and 86% at $n = 1 0 { , } 0 0 0$ . Other DGMs show the same pattern at somewhat smaller magnitudes. Enlarging the conditioning set within this group reduces the inflation, with $Z ^ { \omega _ { Z } } =$ learned + BGE + Jina + Llama rejecting in 38% of the replications at $n = 1 0 { , } 0 0 0$ For the Qwen DGM, concatenating further embedding maps thus recovers part, but not all, of the information in $Z$ about $Y$ that $Z ^ { Q w e n }$ carries. For the corresponding eCITs, we cannot interpret their power, because they do not control the T1E.

For continuous $Y$ the results separate the sample size behavior of the arms by the suficiency of $Z ^ { \omega _ { Z } }$ (Figures 3 and 5). The oracle eCIT stays within two MC standard errors of $\alpha = 0 . 0 5$ at every sample size considered, as for the categorical outcomes. The diluted arms, whose $Z ^ { \omega _ { Z } }$ retains the generating embedding and is therefore suficient by construction, are inflated at $n = 1 , 0 0 0$ but calibrated at $n = 1 0 { , } 0 0 0$ , so that the nuisance regressions of the PCM converge more slowly here than in the categorical case. The arms omitting the generating embedding remain inflated at $n = 1 0 { , } 0 0 0$ . Under the Jina DGM the rejection rates of some of these arms are not monotone in $n ,$ consistent with the convergence of the nuisance regressions and the efect of the omitted information acting simultaneously.

Information in $Z$ about Y that $Z ^ { \omega _ { Z } }$ does not represent acts as an omitted conditioning variable, which condition $( \mathrm { i i } ^ { \prime } )$ of Theorem 14 excludes. It enters as conditional dependence between the embeddings, so the embedded null is false although the null on the texts holds. In addition, the nuisance regressions of the PCM, which are linear in the supplied embeddings, need no longer be correctly specified once the generating embedding is omitted. Both may contribute to the observed inflation, and the present simulation does not separate their contributions. Enlarging $Z ^ { \omega _ { Z } }$ then reduces the T1E inflation (cf. the T1E of the eCITs with $Z ^ { \omega _ { { Z } } } =$ learned + BGE + Jina + Llama against $Z ^ { \omega _ { { Z } } } =$ learned and $Z ^ { \omega _ { { Z } } } =$ BGE + Jina + Llama under the Qwen DGM in Figure 5), but does not guarantee that the remaining inflation is negligible at a given n. Both DGMs omit the generating embedding in some arms, but under the Jina DGM the induced conditional dependence stays negligible at $\sqrt { n }$ scale for the categorical outcomes over the sample sizes considered. Whether an insuficient $Z ^ { \omega _ { Z } }$ matters at a given n is thus a property of the DGM and the embeddings. The simulation design assesses this for the dataset at hand before the eCITs are applied. In our simulation, the arms retaining the generating embedding are calibrated throughout for the categorical outcomes, and for the continuous one the oracle is calibrated throughout while the diluted arms are calibrated from $n = 1 0 { , } 0 0 0$ , and the inflation of the remaining arms decreases as $Z ^ { \omega _ { Z } }$ is enlarged, motivating the full concatenation in Subsection 4.3.

## 4.3 Real-world case study

We apply the eCITs to the test sets described in Subsection 4.1, with Y the speaker’s faction $( L = 7 )$ or gender $\left( L = 2 \right)$ , Z the cleaned speeches and X its summary, for the Gemma 4 and Qwen 3.6 summarizers (cf. Appendix B.1 for additional details). Since $Y$ is categorical, testing conditional mean independence with the PCM is equivalent to testing CI (Remark 16), so nothing is lost by using a CMIT in our application. The specifically learned embedding maps are obtained from classifiers fitted on the training set, which is disjoint from the test set (see also Appendix B.2), while the pre-trained embedding maps were fitted on large corpora that may overlap with the German Parliament speeches. Treating all embedding maps as independent of the test set and the observations as i.i.d., we have $\hat { \theta } = ( \hat { \beta } , \hat { \gamma } )$ ⊥⊥ S, so the conditions on $\hat { \beta }$ (Assumption 11) and $\hat { \gamma }$ (Assumption 13) in $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f , m e a n } }$ hold and Corollary 15 applies with the i.i.d. null class, over which the PCM is asymptotically valid. Neither assumption is guaranteed here, and we return to both below, together with the suficiency of $Z ^ { \omega _ { Z } }$ , which cannot be verified and is instead examined through the simulation study in Subsection 4.2 and by varying $Z ^ { \omega _ { Z } }$ across the settings reported here.

We report four settings (Table 1), which difer in how much of the speech and the summary the embeddings are able to represent. First, we concatenate the four pre-trained embedders (BGE, Jina, Llama, Qwen) for both $Z$ and $X$ , giving $q = p = 1 1 2 0$ , leading to eCITs with diverse pre-trained embeddings but without specifically trained ones. Second, we additionally include the learned embeddings for X and $Z ,$ giving $q = p = 1 8 8 8$ . This setting includes all considered embeddings, and aims to minimize residual insuficiency of the embedding of $Z .$ Specifically, the learned representation of the speech is trained to extract the information in Z about $Y$ , and the pre-trained embeddings are general representations of $Z ,$ which together aim to satisfy condition (ii<sup>′</sup>), i.e. retaining the conditional mean information of Z about $Y$ . Third, we omit the pre-trained embeddings of the second setting from $X ^ { \omega _ { X } }$ , giving $p = 1 8 8 8$ and $q = 7 6 8$ , evaluating a potential power decrease under a less representative embedding of $X$ , and addressing potential violations of Assumption 11. Fourth, we use only the learned embeddings for X and $Z ,$ giving $q = p = 7 6 8$ . This setting drops the embedders whose training corpora may overlap with the German Parliament speeches, addressing potential violations of Assumption 11 and Assumption 13.

Table 1: Test statistics T and p-values of the eCITs on the test sets of Subsection 4.1, for the speaker’s gender and faction and the summaries of both summarizers. Gemma 4: gender $n = 5 0 { , } 8 3 9$ , faction $n = 5 3 , 9 7 8$ . Qwen 3.6: gender $n = 5 0 , 7 9 6$ , faction $n = 5 3 , 8 5 5 ;$ the sample sizes difer slightly because a few summary generations were empty. indicates $p < 0 . 0 5 . ~ Z ^ { \omega _ { Z } }$ and $X ^ { \omega _ { X } }$ give the embeddings of the speech and of the summary; pred-hidden denotes the masked mean of the last hidden layer of the classifier predicting $Y$ , and Qwen the Qwen3 embedder, as distinct from the Qwen 3.6 summarizer. For the PCM, the nuisance regressions on the training dataset are $L _ { \mathrm { 2 - p e n a l i z e d } }$ and those on the evaluation dataset unpenalized, the regressions of $Y$ being multinomial logistic and those of the fitted probabilities and of $\hat { f }$ least squares, with the ridge constant ˆc in the projection direction tuned on $D _ { 1 }$ (Appendix D.1).
<table><tr><td>Outcome</td><td>Zωz</td><td> $X ^ { \omega _ { X } }$ </td><td>Gemma 4 T</td><td>Gemma 4 p</td><td>Qwen 3.6 T</td><td>Qwen 3.6 p</td></tr><tr><td>gender</td><td>BGE+Jina+Qwen+Llama</td><td>BGE+Jina+Qwen+Llama</td><td>24.687</td><td> $< 1 0 ^ { - 1 5 * }$ </td><td>34.146</td><td> $< 1 0 ^ { - 1 5 * }$ </td></tr><tr><td>faction</td><td>BGE+Jina+Qwen+Llama</td><td>BGE+Jina+Qwen+Llama</td><td>34.705</td><td> $< 1 0 ^ { - 1 5 * }$ </td><td>51.317</td><td> $< 1 0 ^ { - 1 5 * }$ </td></tr><tr><td>gender</td><td>pred-hidden+BGE+Jina+Qwen+Llama</td><td>pred-hidden+BGE+Jina+Qwen+Llama</td><td>4.194</td><td> $1 . 3 7 \times 1 0 ^ { - 5 * }$ </td><td>5.505</td><td> $1 . 8 5 \times 1 0 ^ { - 8 * }$ </td></tr><tr><td>faction</td><td>pred-hidden+BGE+Jina+Qwen+Llama</td><td>pred-hidden+BGE+Jina+Qwen+Llama</td><td>11.859</td><td> $< 1 0 ^ { - 1 5 * }$ </td><td>18.334</td><td>&lt; 10−15*</td></tr><tr><td>gender</td><td>pred-hidden+BGE+Jina+Qwen+Llama</td><td>pred-hidden</td><td>3.513</td><td> $2 . 2 1 \times 1 0 ^ { - 4 * }$ </td><td>5.384</td><td> $3 . 6 4 \times 1 0 ^ { - 8 * }$ </td></tr><tr><td>faction</td><td>pred-hidden+BGE+Jina+Qwen+Llama</td><td>pred-hidden</td><td>10.664</td><td> $< 1 0 ^ { - 1 5 * }$ </td><td>17.495</td><td>&lt; 10−15*</td></tr><tr><td>gender</td><td>pred-hidden</td><td>pred-hidden</td><td>4.638</td><td> $1 . 7 6 \times 1 0 ^ { - 6 * }$ </td><td>5.980</td><td> $1 . 1 2 \times 1 0 ^ { - 9 * }$ </td></tr><tr><td>faction</td><td>pred-hidden</td><td>pred-hidden</td><td>12.683</td><td> $< 1 0 ^ { - 1 5 * }$ </td><td>20.010</td><td> $< 1 0 ^ { - 1 5 * }$ </td></tr></table>

All sixteen tests reject at $\alpha = 0 . 0 5$ . In the first setting the test statistics are large, at $T = 2 4 . 7$ (gender) and $T = 3 4 . 7$ (faction) for the Gemma summaries, and even larger for Qwen (Table 1). Adding the learned representations to $Z ^ { \omega _ { Z } }$ and $X ^ { \omega _ { X } }$ in the second setting reduces the test statistics strongly. The two settings difer only by the learned representations, so the reduction shows that the learned representation of the speech carries information about $Y$ that the pre-trained embeddings do not, and hence that the first conditioning set is insuficient and at least part of the conditional association detected in this setting is attributable to that insuficiency. Reducing $X ^ { \omega _ { X } }$ to the learned representation in the third setting lowers the statistics slightly further, in line with the smaller signal retained by a less representative embedding of the summary. Dropping the pre-trained embedders from $Z ^ { \omega _ { Z } }$ in the fourth setting increases the statistics only slightly, which suggests that these embedders carry little about $Y$ beyond what the learned representations already supply. The test decision does not change across the settings: the smallest statistic across all settings, outcomes and summarizers is $T = 3 . 5 .$ , corresponding to $p = 2 . 2 \times 1 0 ^ { - 4 }$ , and the faction statistics remain above 10 throughout. We therefore find that the summaries of both LLMs contain information about the speaker’s faction and gender beyond the speech they were generated from. The statistics are larger for Qwen than for Gemma in all eight comparisons, but $T$ depends on the suficiency of the embedding maps and the nuisance fits as well as on the strength of the conditional dependence, so we do not read this as a comparison of the two summarizers.

We discuss four explanations for the test decision other than conditional dependence. First, the embedding of $Z$ could be insuficient: exact suficiency is not plausible for embedding maps learned from data, and by Theorem 14 any information in $Z$ about the conditional mean of $Y$ that $Z ^ { \omega _ { Z } }$ drops can act as an omitted conditioning variable (cf. also the Qwen DGM results in Subsection 4.2). However, the eCITs with a large $Z ^ { \omega _ { Z } }$ consisting of concatenated pre-trained and learned embeddings, and thus covering both general and Y-specific information in $Z ,$ , still reject (second setting). Furthermore, adding the four pre-trained embedders to a $Z ^ { \omega _ { Z } }$ containing only learned embeddings, while holding $X ^ { \omega _ { X } } =$ pred-hidden fixed, changes the test statistics only slightly and leaves every statistic well above the critical value. The pre-trained embedders thus carry little information about $Y$ beyond what the learned representation of the speech already supplies, which suggests that the bias from residual insuficiency is not large here. The corresponding simulation arms are calibrated for categorical $Y$ up to $n = 1 0 { , } 0 0 0$ , but this is limited evidence at the sample sizes used here, since the inflation caused by an insuficient $Z ^ { \omega _ { Z } }$ grows with $n .$ . We can therefore not exclude residual insuficiency, although it would have to concern information that neither the pre-trained embedders nor a classifier trained for the task recovers from the speech.

Second, the nuisance regressions on which the validity of the PCM rests could be estimated inaccurately. The dimension of the conditioning set, between $p = 7 6 8$ and $p = 1 8 8 8$ is large relative to the $n _ { 2 } \approx 2 6 { , } 0 0 0$ observations in the evaluation half. However, this is probably not the main driver of the observed statistics, since they do not increase with $p ,$ whereas the corresponding bias term does. In particular, the largest statistics are obtained at $p = 1 1 2 0$ , while the settings with the highest and the lowest dimension, $p = 1 8 8 8$ and $p = 7 6 8$ , give smaller values of comparable size. Furthermore, the simulation study uses conditioning sets of comparable dimensions at the smaller sample size $n = 1 0 { , } 0 0 0$ , hence at a less favorable ratio of $p$ to $n _ { 2 }$ than here, and the arms with suficient $Z ^ { \omega _ { Z } }$ remain calibrated there.

Third, the detected conditional dependence could run through the embedding maps rather than the summarizers. The pre-trained embedders are fitted on external corpora, which may themselves contain German Parliament speeches, and the classifier supplying the learned representations is fine-tuned from a pre-trained encoder. The split is drawn at the level of speeches, so the training and test samples share speakers, topics and electoral terms, which can induce dependence between train and test sample. Thus, $\hat { \beta }$ and $\hat { \gamma }$ might not be independent of the test sample, and both Assumption 11 and Assumption 13 in the definition of $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f , m e a n } }$ can fail. The embeddings could then contain information about $Y$ from the embedder rather than the summarizer, so that a rejection need not reflect conditional dependence at the level of the texts. For this to produce a rejection, the efect has to difer between the speech and its summary, since information that the embedder attaches to both is conditioned on through $Z ^ { \omega _ { Z } }$ . Fine-tuning the learned maps on a holdout split removes the dependence on the test speeches themselves, but neither this nor the choice of split removes the dependence induced by the pre-training corpus of the encoder, so the fourth setting reduces but does not remove this dependence. The agreement across the two summarizers does not speak against this explanation, as the same embedders are used for both.

Fourth, the observations could be dependent rather than i.i.d. Speakers recur across speeches and speeches are dependent through shared topics and across time, so the rows of S are not i.i.d. and the attributes Y are constant within speaker. Lemmas 19 and 20 and the asymptotic validity of the PCM all rely on the i.i.d. assumption, and positive dependence between observations can make the PCM anti-conservative. We can therefore not exclude that this dependence contributes to the observed magnitude of the statistics; extending the eCITs to dependent samples is left for future work.

The result is consistent with the mechanism of Example 1. Both summarizers were pre-trained on web-scale corpora that plausibly include German Parliament plenary protocols, so speeches in the test set, speeches by the same speakers, or other texts associated with them may have been seen during training. Memorization of these texts then allows the summary to contain information about speaker attributes which is not in the speech. Under all four combinations of $Z ^ { \omega _ { Z } }$ and $X ^ { \omega _ { X } }$ the eCITs reject for both outcomes and both summarizers, and the changes in the test statistics across the settings follow the expected direction, which weakens some of the alternative explanations for the test decisions. The test thus certifies that the summaries carry additional information about the protected attribute on this task and dataset, with a T1E guarantee under the stated conditions. It does not guarantee suficiency of $Z ^ { \omega _ { Z } }$ , localize the information, quantify its magnitude on an interpretable scale, or attribute it to a specific mechanism, including the memorization discussed here.

## 5 Conclusion

We introduced embedded conditional independence tests (eCITs) to test for conditional dependence between a text and a scalar given a text, a setting in which existing CITs cannot be applied with T1E guarantees. We first justified CI testing as a bias criterion for LLM outputs and characterized when the null and alternative can hold for a fixed LLM, illustrating that memorization of individual observations can induce conditional dependence. Existing bias metrics do not provide such guarantees: intrinsic metrics need not correlate with downstream behavior, extrinsic metrics are bound to benchmark datasets with specific characteristics, and LLM-based evaluators can themselves be biased. In contrast, our criterion requires neither a gold-standard output nor an annotation of what an unbiased output would be: the source text supplies the baseline against which additional information is measured. We then characterized suficient embedding maps for the conditioned text, under which a valid CIT for the embedded hypothesis is a valid CIT for the textual hypothesis, and gave conditions under which the two hypotheses are equivalent. We further showed that the requirement on the embedding maps can be weakened to mean suficiency when the embedded test targets conditional mean independence, and that for categorical Y, as in our application, the two nulls coincide for the i.i.d. laws considered here. We further proposed a simulation design to assess T1E control and power of the eCITs on a given dataset before applying them. On German Parliament speeches, the design exhibits both regimes it is meant to detect: the eCITs are calibrated for some combinations of embedding maps, while others show T1E inflation that need not vanish, and can grow, with the sample size. Since validity depends on the suficiency of the embedding map under a given data generating process, the design serves to assess suficiency for the dataset and embedding maps at hand. Applying the eCITs to the speeches and the summaries of two LLMs, we reject the null for both the speaker’s faction and gender. Four alternative explanations cannot be excluded, although the calibration of comparable arms in the simulation and the changes in test statistics with consistent test decisions both speak against any of them accounting for the rejections on its own.

The main open challenge is exact (mean) suficiency, which is unlikely to hold for pretrained or learned embedding maps, and even small amounts of residual information can afect the T1E at larger sample sizes, as the simulation suggests. A diagnostic quantifying how much relevant information about Y or $X ^ { \omega _ { X } }$ is left in the source text, rather than certifying that none is, would be of considerable interest. Two further directions remain. We studied only embedding maps learned on a holdout set; (unsupervised) in-sample learning and the dependence it introduces in the test sample deserve theoretical and empirical study. And while we used the PCM, other CITs could be combined with suficient embedding maps in the same way.

eCITs make conditional independence testing available for multimodal data and thereby give statistical guarantees for bias claims about LLM outputs on a specified task and dataset. While our focus has been on summarization and text, our eCITs apply to any setting in which the conditioning variable and the tested variable admit representations satisfying the (mean) suficiency conditions.

## Acknowledgments and Disclosure of Funding

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 459422098.

## Use of AI Tools

The LLMs analysed in this work are described in Appendix B; their outputs are the object of study rather than part of the manuscript. In preparing the manuscript, the authors additionally used LLMs to assist with drafting and language editing. The authors take full responsibility for the contents of the paper, including all derivations, results and references.

<table><tr><td>Dawid (1979)</td><td>Statement</td><td>Name</td></tr><tr><td>Lemma 4.1</td><td> $A \perp B \mid C \iff A \perp ( B , C ) \mid C$ </td><td>redundancy</td></tr><tr><td>Lemma 4.2 (i)</td><td> $A \perp B \mid C \implies A \perp \mid f ( B ) \mid C$ </td><td>decomposition</td></tr><tr><td>Lemma 4.2 (ii)</td><td> $A \perp \perp B \mid C$  and  $D = f ( B ) \implies A \perp B \mid ( C , D )$ </td><td>weak union</td></tr><tr><td>Lemma 4.3</td><td>A  $B \mid C$  and D Ⅱ  $B \mid ( A , C ) \implies ( A , D ) \perp \mid B \mid C$ </td><td>contraction</td></tr></table>

Table 2: f denotes any measurable function of B.

## Appendix A. Proofs

The proofs are based on applying the lemmata from Dawid (1979). For readability, we refer to them as specified in Table 2.

The general form of Lemma 2 is given in the following:

Lemma 17 (Disintegration of conditional independence) Let $( E _ { A } , \mathcal { E } _ { A } ) , ( E _ { B } , \mathcal { E } _ { B } ) , ( E _ { C } , \mathcal { E } _ { C } )$ be standard Borel spaces and write $( \boldsymbol { E } , \boldsymbol { \mathcal { E } } ) = ( E _ { A } \times E _ { B } \times E _ { C } , \mathcal { E } _ { A } \otimes \mathcal { E } _ { B } \otimes \mathcal { E } _ { C } )$ , with coordinate projections A, B, C. Let

$$
\mathcal { P } _ { 0 } \mathrel { \mathop : } = \big \{ \mu \mathbin { \ p r o b a b i l i t y } \ m e a s u r e \ o n  \left( E , \mathcal { E } \right) \ \colon \ A \ \bot \ B \ | \ C \ u n d e r \ \mu \big \} .
$$

Let V be an E-valued and W a $( \mathcal W , \mathcal G )$ -valued random variable on $( \Omega , \mathcal { F } , \mathbb { P } )$ , where $( \mathcal W , \mathcal G )$ is an arbitrary measurable space, and let $\mathbb { P } ^ { V | W }$ be a regular conditional distribution $o f V$ given W. Then $\{ w \in \mathcal { W } : \mathbb { P } ^ { V | W = w } \in \mathcal { P } _ { 0 } \} \in \mathcal { G }$ and

$$
A \perp \perp B | ( C , W ) \quad \Longleftrightarrow \quad \mathbb { P } ^ { V | W } \in \mathcal { P } _ { 0 } \quad \mathbb { P } ^ { W } - a . s .
$$

Proof For a bounded measurable $h : E \to$ R choose, by the Doob–Dynkin lemma, a jointly measurable $\Phi _ { h } : \mathcal { W } \times E _ { C } \to$ R with $\mathbb { E } [ h ( V ) \mid C , W ] = \Phi _ { h } ( W , C )$ P-a.s.; this requires no assumption on $( \mathcal W , \mathcal G )$

Step 1 (two-stage conditioning). We show that for $\mathbb { P } ^ { W } \mathrm { - a . e . ~ } w$ , the map $\Phi _ { h } ( w , \cdot )$ is a version of $\mathbb { E } _ { \mathbb { P } V | W = w } \left[ h \mid C \right]$ . Fix $D \in { \mathcal { E } } _ { C }$ and set

$$
\psi _ { D } ( w ) : = \int \big ( \Phi _ { h } ( w , c ) - h ( v ) \big ) \mathbf { 1 } _ { D } ( c ) \mathbb { P } ^ { V | W = w } ( d v ) , \qquad v = ( a , b , c ) ,
$$

which is measurable in w by joint measurability of $\Phi _ { h }$ and of $w \mapsto \mathbb { P } ^ { V | W = w } ( F ) , F \in \mathcal { E }$ By the defining property of the regular conditional distribution, $\psi _ { D } ( W ) = \mathbb { E } \big [ ( \Phi _ { h } ( W , C ) -$ $h ( V ) ) \mathbf { 1 } _ { D } ( C ) \mid W \big ]$ a.s. For every $G \in { \mathcal { G } }$ the variable ${ \mathbf { 1 } } _ { D } ( C ) { \mathbf { 1 } } _ { G } ( W )$ is σ $\cdot ( C , W )$ -measurable, whence ${ \mathbb E } [ ( \Phi _ { h } ( \bar { W } , C ) - h ( V ) ) \mathbf { 1 } _ { D } ( C ) \mathbf { 1 } _ { G } ( W ) ] = 0$ and therefore $\mathbb { E } [ \psi _ { D } ( W ) \mathbf { 1 } _ { G } ( W ) ] = 0$ . As $\psi _ { D } ( W )$ is σ(W)-measurable, $\psi _ { D } ( W ) = 0$ a.s. Since $E _ { C }$ is standard Borel, $\mathcal { E } _ { C }$ is countably generated; applying the above to a countable generating algebra, discarding the union of the corresponding null sets and extending by a monotone class argument for each fixed w yields the claim.

Step 2 (reduction to a countable family). Let $\{ A _ { k } \} _ { k \in \mathbb { N } }$ and $\{ B _ { l } \} _ { l \in \mathbb { N } }$ be countable generating algebras of $\mathcal { E } _ { A }$ and $\mathcal { E } _ { B }$ , and abbreviate $f _ { k } : = \mathbf { 1 } _ { A _ { k } } \circ A , g _ { l } : = \mathbf { 1 } _ { B _ { l } } \circ B$ . By a monotone class argument, $A \perp \perp B \mid ( C , W )$ holds if and only if

$$
\Phi _ { f _ { k } g _ { l } } = \Phi _ { f _ { k } } \Phi _ { g _ { l } }\tag{10}
$$

holds at $( W , C ) \ \mathbb { P } \mathrm { { - a . s } }$ . for all $k , l ;$ and, by Step 1, for $\mathbb { P } ^ { W _ { \mathrm { - a . e } } }$ . w the measure $\mathbb { P } ^ { V | W = w }$ lies in $\mathcal { P } _ { 0 }$ if and only if (10) holds at $( w , C ) \ P ^ { V | W = w } \mathrm { - a . s }$ . for all $k , l .$

Step 3 (transfer). Let

$$
N _ { k , l } : = \left\{ ( w , v ) \in \mathcal { W } \times E : \Phi _ { f _ { k } g _ { l } } ( w , c ) \neq \Phi _ { f _ { k } } ( w , c ) \Phi _ { g _ { l } } ( w , c ) \right\} \in \mathcal { G } \otimes \mathcal { E } ,
$$

with sections $( N _ { k , l } ) _ { w }$ . The regular conditional distribution gives

$$
\mathbb { P } \big ( ( W , V ) \in N _ { k , l } \big ) = \int \mathbb { P } ^ { V | W = w } \big ( ( N _ { k , l } ) _ { w } \big ) \mathbb { P } ^ { W } ( d w ) ,
$$

so $\mathbb { P } ( ( W , V ) \in N _ { k , l } ) = 0$ if and only i $\uparrow \mathbb { P } ^ { V | W = w } ( ( N _ { k , l } ) _ { w } ) = 0 { \mathrm { ~ f o r ~ } } \mathbb { P } ^ { W _ { - \mathrm { { a . e . } ~ } w } }$ . Taking the union over the countably many pairs $( k , l )$ and combining with Step 2 proves the equivalence. Finally,

$$
\big \{ w : \mathbb { P } ^ { V | W = w } \in \mathcal { P } _ { 0 } \big \} = \bigcap _ { k , l } \big \{ w : \mathbb { P } ^ { V | W = w } \big ( ( N _ { k , l } ) _ { w } \big ) = 0 \big \} \in \mathcal { G } ,
$$

since $w \mapsto \mathbb { P } ^ { V | W = w } ( ( N _ { k , l } ) _ { w } )$ is measurable for each (k, l) by Fubini’s theorem for kernels.

Lemma 18 (Disintegration of conditional mean independence) Let $( E _ { A } , \mathcal { E } _ { A } ) , ( E _ { B } , \mathcal { E } _ { B } ) , ( E _ { C } , \mathcal { E } _ { C } )$ be standard Borel with $E _ { B } \subseteq \mathbb { R } ^ { d }$ Borel, and write $( \boldsymbol { E } , \boldsymbol { \mathcal { E } } ) = ( E _ { A } \times E _ { B } \times E _ { C } , \mathcal { E } _ { A } \otimes \mathcal { E } _ { B } \otimes \mathcal { E } _ { C } )$ with coordinate projections $A , B , C$ . Let

$$
\mathcal { P } _ { 0 } ^ { \mathrm { m e a n } } : = \Bigl \{ \mu \ p r o b a b i l i t y \ m e a s u r e \ o n \ ( E , \mathcal { E } ) : \int \| b \| \ \mu ( d v ) < \infty , \mathbb { E } _ { \mu } [ B \mid A , C ] = \mathbb { E } _ { \mu } [ B \mid C ] \ \mu - a . s . \Bigr \} .
$$

Let V be an E-valued and W a $( \mathcal { W } , \mathcal { G } )$ -valued random variable on $( \Omega , \mathcal { F } , \mathbb { P } )$ with $\mathbb { E } \Vert B ( V ) \Vert <$ ∞, where $( \mathcal W , \mathcal G )$ is an arbitrary measurable space, and let $\mathbb { P } ^ { V | W }$ be a regular conditional distribution of V given W. Then $\{ w \in \mathcal { W } : \mathbb { P } ^ { V | W = w } \in \mathcal { P } _ { 0 } ^ { \mathrm { m e a n } } \} \in \mathcal { G }$ and

$$
{ \mathbb E } [ B \mid A , C , W ] = { \mathbb E } [ B \mid C , W ] \ a . s . \ \Longleftrightarrow \ { \mathbb P } ^ { V | W } \in { \mathcal P } _ { 0 } ^ { \mathrm { m e a n } } \ \ \  { \mathbb P } ^ { W } - a . s . 
$$

Proof It sufices to treat $d = 1$ , applying the result to each coordinate of B and intersecting the countably many resulting null sets.

By the Doob–Dynkin lemma choose jointly measurable $\Phi _ { 1 } : \mathcal { W } \times E _ { A } \times E _ { C } \to \mathbb { R }$ and $\Phi _ { 2 } : \mathcal { W } \times E _ { C } \to \mathbb { R }$ with $\mathbb { E } [ B \mid A , C , W ] = \Phi _ { 1 } ( W , A , C )$ and $\operatorname { \mathbb { E } } [ B \mid C , W ] = \Phi _ { 2 } ( W , C ) \ \operatorname { \mathbb { P } } \mathrm { - a . s . ; }$ this requires no assumption on $( \mathcal W , \mathcal G )$

Step 1. Step 1 of Lemma 17 applies verbatim to integrable rather than bounded h: by Fubini’s theorem for kernels, $\mathbb { E } \| B ( V ) \| < \infty$ gives $\begin{array} { r } { \int \| b \| \mathbf { \bar { P } } ^ { V | W = w } ( d v ) < \infty } \end{array}$ for $\mathbb { P } ^ { W } \mathrm { - a . e . ~ } w .$ and dominated convergence replaces boundedness in the monotone class step. Applying it once with conditioning field $\sigma ( A , C )$ and once with $\sigma ( C )$ yields that, for $\mathbb { P } ^ { W } \mathrm { - a . e . ~ } w$ , the maps $\Phi _ { 1 } ( w , \cdot , \cdot )$ and $\Phi _ { 2 } ( w , \cdot )$ are versions of $\mathbb { E } _ { \mathbb { P } ^ { V | W = w } } [ B \mid A , C ]$ and $\mathbb { E } _ { \mathbb { P } ^ { V | W = w } } [ B \mid C ]$ , respectively. In particular $\mathbb { P } ^ { V | \dot { W } = \dot { w } } \in \mathcal { P } _ { 0 } ^ { \mathrm { m e a n } }$ if and only if $\Phi _ { 1 } ( w , A , C ) = \Phi _ { 2 } ( w , C )$ holds $\mathrm { | } \mathrm { P } ^ { \mathrm { \it V } | W = w } \mathrm { - a . s }$

Step 2. Let $N : = \{ ( w , v ) \in \mathcal { W } \times E : \Phi _ { 1 } ( w , a , c ) \neq \Phi _ { 2 } ( w , c ) \} \in \mathcal { G } \otimes \mathcal { E }$ , with sections $N _ { w }$ The regular conditional distribution gives

$$
\mathbb { P } \big ( ( W , V ) \in N \big ) = \int \mathbb { P } ^ { V | W = w } ( N _ { w } ) \mathbb { P } ^ { W } ( d w ) ,
$$

so $\mathbb { P } ( ( W , V ) \in N ) = 0$ if and only if $\mathsf { P } ^ { V | W = w } ( N _ { w } ) = 0$ for $\mathbb { P } ^ { W } \mathrm { - a . e . ~ } w$ . Combined with Step 1 this is the asserted equivalence. Finally $\{ w : \mathbb { P } ^ { V | W = w } \in \mathcal { P } _ { 0 } ^ { \operatorname* { m e a n } } \} = \{ w : \mathbb { P } ^ { V | W = w } ( N _ { w } ) =$ $\begin{array} { r } { 0 \} \cap \left\{ w : \int \left\| b \right\| \mathbb { P } ^ { V | W = w } ( d v ) < \infty \right\} \in \mathcal { G } } \end{array}$ , since $w \mapsto \mathbb { P } ^ { V | W = w } ( \tilde { N _ { w } } )$ and $\begin{array} { r } { w \mapsto \int \| b \| \mathbb { P } ^ { V | W = w } ( d v ) } \end{array}$ are measurable by Fubini’s theorem for kernels.

Lemma 19 (Lifting to the sample level) Let $( X _ { i } , Y _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ be i.i.d. copies of $( X , Y , Z )$ with values in the standard Borel spaces $( \mathcal { X } , \mathcal { F } _ { \mathcal { X } } ) , ( \mathcal { V } , \mathcal { F } _ { \mathcal { Y } } ) , ( \mathcal { Z } , \mathcal { F } _ { \mathcal { Z } } )$ . Then

$$
X \perp \bot Y \mid Z \iff X ^ { n } \perp \mid Y ^ { n } \mid Z ^ { n } .
$$

Proof [Proof of Lemma $1 9 ] \stackrel { 6 6 } { \Rightarrow } \stackrel { 5 9 } { }$ . Since the rows $( X _ { i } , Y _ { i } , Z _ { i } )$ are independent, for bounded measurable $h _ { i } : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ we have

$$
\begin{array} { r } { \mathbb { E } \Big [ \prod _ { i = 1 } ^ { n } h _ { i } ( X _ { i } , Y _ { i } ) \Big | Z ^ { n } \Big ] = \prod _ { i = 1 } ^ { n } \mathbb { E } \big [ h _ { i } ( X _ { i } , Y _ { i } ) \mid Z _ { i } \big ] \qquad \mathrm { a . s . } } \end{array}\tag{11}
$$

Indeed, the right-hand side is $\sigma ( Z ^ { n } )$ -measurable, and for $\begin{array} { r } { \phi ( Z ^ { n } ) = \prod _ { i } \phi _ { i } ( Z _ { i } ) } \end{array}$ with bounded measurable $\phi _ { i }$ both sides have the same expectation against $\phi ( Z ^ { n } )$ by independence across $i ;$ such $\phi$ generate a π-system generating $\sigma ( Z ^ { n } )$ , so (11) follows by a functional monotone class argument.

Now let $f _ { i } , g _ { i }$ be bounded and measurable and apply (11) with $h _ { i } ( x , y ) = f _ { i } ( x ) g _ { i } ( y )$ By $X \perp \perp Y \mid Z$ applied row-wise, $\mathbb { E } [ f _ { i } ( X _ { i } ) g _ { i } ( Y _ { i } ) \mid Z _ { i } ] = \mathbb { E } [ f _ { i } ( X _ { i } ) \mid Z _ { i } ] \mathbb { E } [ g _ { i } ( Y _ { i } ) \mid Z _ { i } ] .$ , so regrouping the product and applying (11) twice more, once with $g _ { i } \equiv 1$ and once with $f _ { i } \equiv 1$ , gives

$$
\begin{array} { r } { \mathbb { E } \Big [ \prod _ { i } f _ { i } ( X _ { i } ) \prod _ { i } g _ { i } ( Y _ { i } ) \Big | Z ^ { n } \Big ] = \mathbb { E } \Big [ \prod _ { i } f _ { i } ( X _ { i } ) \Big | Z ^ { n } \Big ] \mathbb { E } \Big [ \prod _ { i } g _ { i } ( Y _ { i } ) \Big | Z ^ { n } \Big ] . } \end{array}
$$

Functions of the form $\prod _ { i } f _ { i }$ with $f _ { i } = \mathbf { 1 } _ { A _ { i } } , A _ { i } \in \mathcal { F } _ { \mathcal { X } }$ , form a π-system generating $\mathcal { F } _ { \mathcal { X } } ^ { \otimes n }$ and analogously for $Y ^ { n } ; \mathrm { a }$ functional monotone class argument in each argument separately extends the display to all bounded ${ \mathcal { F } } _ { \mathcal { X } } ^ { \otimes n } .$ - and $\mathcal { F } _ { \mathcal { Y } } ^ { \otimes n }$ -measurable functions, which is $X ^ { n }$ ⊥⊥ $Y ^ { n } \mid Z ^ { n }$

$^ { 6 6 } \Leftarrow = ^ { 5 9 }$ . Decomposition applied twice gives $X _ { 1 } \perp \perp Y _ { 1 } \mid Z ^ { n }$ . Since $( X _ { 1 } , Y _ { 1 } , Z _ { 1 } ) \perp \perp Z _ { 2 : n } .$ redundancy and contraction give $X _ { 1 } \perp \perp Y _ { 1 } \mid Z _ { 1 }$ , and $( X _ { 1 } , Y _ { 1 } , Z _ { 1 } ) \stackrel { d } { = } ( X , Y , Z )$ ■

Lemma 20 (Lifting to the sample level, mean version) Let $( X _ { i } , Y _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ be i.i.d. copies $o f \ ( X , Y , Z )$ with values in the standard Borel spaces $( \boldsymbol { \mathcal { X } } , \mathcal { F } _ { \boldsymbol { \mathcal { X } } } ) , \ ( \boldsymbol { \mathcal { V } } , \mathcal { F } _ { \boldsymbol { \mathcal { Y } } } ) , \ ( \boldsymbol { \mathcal { Z } } , \mathcal { F } _ { \mathcal { Z } } )$ , with $\mathcal { V } \subseteq \mathbb { R } ^ { K }$ Borel and $\mathbb { E } \| Y \| < \infty$ . Then

$$
\mathbb { E } [ Y \mid X , Z ] = \mathbb { E } [ Y \mid Z ] \ a . s . \ \Longleftrightarrow \ \mathbb { E } [ Y _ { i } \mid X ^ { n } , Z ^ { n } ] = \mathbb { E } [ Y _ { i } \mid Z ^ { n } ] \ a . s . \ f o r \ a l l \ i \leq n .
$$

Proof Fix $i \leq n$ . Since $( X _ { i } , Y _ { i } , Z _ { i } )$ is independent of $( X , Z ) _ { - i } ^ { n } ,$ weak union applied with the measurable function $( X _ { i } , Z _ { i } )$ of $( X _ { i } , Y _ { i } , Z _ { i } )$ , followed by decomposition, gives $Y _ { i }$ ⊥⊥ $( X , Z ) _ { - i } ^ { n } \mid ( X _ { i } , Z _ { i } )$ and $Y _ { i } \perp \perp Z _ { - i } ^ { n } \mid Z _ { i }$ . Hence

$$
\mathbb { E } [ Y _ { i } \mid X ^ { n } , Z ^ { n } ] = \mathbb { E } [ Y _ { i } \mid X _ { i } , Z _ { i } ] , \qquad \mathbb { E } [ Y _ { i } \mid Z ^ { n } ] = \mathbb { E } [ Y _ { i } \mid Z _ { i } ] \quad { \mathrm { a . s . } }\tag{12}
$$

Both conditional expectations are well defined since $\mathbb { E } \| Y \| < \infty$ . The two right-hand sides agree a.s. for every i if and only if $\mathbb { E } [ Y \mid X , Z ] = \mathbb { E } [ Y \mid Z ]$ a.s., because $\left( X _ { i } , Y _ { i } , Z _ { i } \right) \stackrel { d } { = }$ $( X , Y , Z )$ ; by (12) this is the asserted equivalence. 7

Proof [Proof of Lemma 2]Apply Lemma 17 with $\boldsymbol { V } = S ^ { \omega } , W = ( \hat { \gamma } , \hat { \beta } )$ and $( A , B , C ) =$ $( X ^ { \omega _ { X } , n } , Y ^ { n } , Z ^ { \omega _ { Z } , n } )$ , so that $\mathcal { P } _ { 0 } ~ = ~ \mathcal { P } _ { 0 } ^ { \omega }$ ; and with $V ~ = ~ S ^ { \omega _ { X } } , ~ W ~ = ~ { \hat { \beta } }$ and $( A , B , C ) =$ $( X ^ { \omega _ { X } , n } , Y ^ { n } , Z ^ { n } )$ , so that $\mathcal { P } _ { 0 } = \mathcal { P } _ { 0 } ^ { \omega _ { X } }$ , for the second statement.

Proof [Proof of Theorem 3]By Lemma 2 applied to $\hat { \beta }$ and $S ^ { \omega _ { X } } , H _ { 0 } ^ { \omega _ { X } }$ is equivalent to

$$
X ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid ( \hat { \beta } , Z ^ { n } ) .\tag{13}
$$

Step 1: adjoining γˆ. We show that (13) together with Assumption 1 implies

$$
X ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid ( \hat { \beta } , \hat { \gamma } , Z ^ { n } ) .\tag{14}
$$

Under Assumption 1a), contraction applied to (13) and $\hat { \gamma } \perp \perp Y ^ { n } \mid ( ( X ^ { \omega _ { X } } , Z ) ^ { n } , \hat { \beta } )$ gives $\left( X ^ { \omega _ { X } , n } , \hat { \gamma } \right) \perp \perp Y ^ { n } \mid \left( \hat { \beta } , Z ^ { n } \right)$ . Under Assumption 1b), contraction applied to the symmetric form of (13) and $\hat { \gamma } \perp \perp X ^ { \omega _ { X } , n } \mid ( ( Y , Z ) ^ { n } , \hat { \beta } )$ gives $( Y ^ { n } , \hat { \gamma } ) \perp \perp X ^ { \omega _ { X } , n } \mid ( \hat { \beta } , Z ^ { n } )$ . In either case, weak union with the measurable function $\hat { \gamma }$ of the left-hand pair, followed by decomposition, yields (14).

Step 2: replacing $Z ^ { n } \ b y \ Z ^ { \omega z , n }$ . Write $C _ { 0 } = ( \hat { \beta } , \hat { \gamma } , Z ^ { \omega _ { Z } , n } )$ . Since $Z ^ { \omega _ { Z } , n }$ is $\sigma ( Z ^ { n } , \hat { \gamma } ) \cdot$ measurable, we have $\sigma ( C _ { 0 } , Z ^ { n } ) = \sigma ( \hat { \beta } , \hat { \gamma } , Z ^ { n } )$ , so (14) reads

$$
X ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid ( C _ { 0 } , Z ^ { n } ) ,\tag{15}
$$

which is symmetric in $X ^ { \omega _ { X } , n }$ and $Y ^ { n }$ . Under (i), contraction applied to $Z ^ { n } \perp \perp X ^ { \omega _ { X } , n } \mid C _ { 0 }$ and (15) gives $( Z ^ { n } , Y ^ { n } ) \perp \perp X ^ { \omega _ { X } , n } \mid C _ { 0 } ;$ under (ii), the same argument with the roles of $X ^ { \omega _ { X } , n }$ and $Y ^ { n }$ interchanged gives $( Z ^ { n } , X ^ { \omega _ { X } , n } ) \perp \perp Y ^ { n } \mid C _ { 0 }$ . In either case decomposition yields $X ^ { \omega _ { X } , n } ~ \bot \bot ~ Y ^ { n } ~ | ~ C _ { 0 }$ , which by Lemma 2 is $H _ { 0 } ^ { \omega }$ . All four combinations of Assumption 1a)–b) with (i)–(ii) are therefore covered.

Proof [Proof of Theorem 5]Fix $\mathbb { P } ^ { S } \in \mathcal { P } _ { 0 } ^ { \mathrm { s u f f } }$ and write $\boldsymbol { \hat { \theta } } = ( \hat { \boldsymbol { \beta } } , \hat { \boldsymbol { \gamma } } )$

Step 1: $H _ { 0 } \Rightarrow H _ { 0 } ^ { \omega _ { X } }$ . Since $\mathbb { P } ^ { S } \in \mathcal { P } _ { 0 }$ , the $( X _ { i } , Y _ { i } , Z _ { i } )$ are i.i.d. with X ⊥⊥ $Y \mid Z .$ , so $X ^ { n }$ ⊥⊥ $Y ^ { n } \mid Z ^ { n }$ by Lemma 19. By definition of $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f } }$ we have ${ \hat { \beta } } \perp \perp Y ^ { n } \mid ( X , Z ) ^ { n }$ , and contraction gives $( X ^ { n } , { \hat { \beta } } ) \bot \mid Y ^ { n } \mid Z ^ { n }$ . By symmetry of conditional independence, $Y ^ { n } \perp \perp ( X ^ { n } , { \hat { \boldsymbol { \beta } } } ) \mid Z ^ { n }$ ; weak union with the measurable function $\hat { \beta }$ of $( X ^ { n } , { \hat { \beta } } )$ yields $Y ^ { n } \perp \perp ( X ^ { n } , { \hat { \boldsymbol { \beta } } } ) \mid ( Z ^ { n } , { \hat { \boldsymbol { \beta } } } )$ , and decomposition, applied to the $\sigma ( X ^ { n } , { \hat { \beta } } )$ -measurable $X ^ { \omega _ { X } , n }$ , yields $K ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid ( Z ^ { n } , \hat { \beta } )$ ， which by Lemma 2 is $H _ { 0 } ^ { \omega _ { X } }$

Step 2: $H _ { 0 } ^ { \omega _ { X } } \Rightarrow H _ { 0 } ^ { \omega }$ . By definition of $\mathcal { P } _ { 0 } ^ { \mathrm { s u f f } } , \hat { \gamma }$ satisfies Assumption 1 and $\omega _ { Z }$ satisfies (i) or (ii) of Theorem 3, so Theorem 3 gives $X ^ { \omega _ { X } , n }$ ⊥⊥ $Y ^ { n } \mid ( Z ^ { \omega _ { Z } , n } , \hat { \theta } )$ . By Lemma 2 with $V = S ^ { \omega }$ and $W = { \hat { \theta } }$ ，

$$
\begin{array} { r l } { \mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \in \mathcal { P } _ { 0 } ^ { \omega } } & { { } \quad \mathbb { P } ^ { \hat { \theta } } \mathrm { - a . s . } } \end{array}\tag{16}
$$

Step 3: a valid CIT implies a valid $e C I T$ . Since $S ^ { \omega } = \bar { \omega } ( S , \hat { \theta } )$ is $\sigma ( S , \hat { \theta } )$ -measurable and $U \perp \perp ( S , \hat { \theta } )$ , we have $U \perp \perp ( S ^ { \omega } , \hat { \theta } )$ , so $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \otimes \mathrm { U n i f } [ 0 , 1 ]$ is a regular conditional distribution

of $( S ^ { \omega } , U )$ given ${ \hat { \theta } } .$ . The map $( s ^ { \omega } , u ) \mapsto \mathbf { 1 } \{ \psi ( s ^ { \omega } , u ) = 1 \}$ is bounded and measurable, so by the defining property of the regular conditional distribution,

$$
\mathbb { E } \big [ { \bf 1 } \{ \psi ( S ^ { \omega } , U ) = 1 \} \big | \hat { \theta } \big ] = \int { \bf 1 } \{ \psi = 1 \} { \bf d } \big ( \mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \otimes \mathrm { U n i f } [ 0 , 1 ] \big ) \qquad \mathbb { P } ^ { \hat { \theta } } \mathrm { - a . s . }
$$

By (16) the integrating measure is of the form $Q \otimes { \mathrm { U n i f } } [ 0 , 1 ]$ with $Q \in \mathcal { P } _ { 0 } ^ { \omega }$ almost surely, so validity of $\psi ,$ , being uniform over $\mathcal { P } _ { 0 } ^ { \omega }$ , bounds the right-hand side by α almost surely. Since $\varphi _ { \omega } ( S ) = \psi ( S ^ { \omega } , U )$ , the tower property gives

$$
\mathbb { P } ^ { S } \big ( \varphi _ { \omega } = 1 \big ) = \mathbb { E } \Big [ \mathbb { E } \big [ { \mathbf { 1 } } \{ \psi ( S ^ { \omega } , U ) = 1 \} \big | \big \theta \big ] \Big ] \leq \alpha .
$$

As $\mathbb { P } ^ { S } \in \mathcal { P } _ { 0 } ^ { \mathrm { s u f f } }$ was arbitrary, taking the supremum gives the claim.

Proof [Proof of Corollary 7]Fix such a $\mathbb { P } ^ { S }$ . Since $\hat { \theta }$ ⊥⊥ $S ,$ the conditional law of S given $\hat { \theta }$ is $\mathbb { P } ^ { S }$ itself, and conditionally on $\widehat { \theta } = ( b , c )$ the rows of $S ^ { \omega }$ are the images of the i.i.d. rows of S under the fixed measurable map $( x , y , z ) \mapsto ( \omega _ { X } ( x , b ) , y , \omega _ { Z } ( z , c ) )$ . Hence $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } }$ is an n-fold product of a common marginal $\mathbb { P } ^ { \tilde { \theta } _ { - \mathrm { a . s } } }$ . Combined with Lemmas 2 and 19 it lies in $\mathcal { P } _ { 0 } ^ { \omega , \otimes }$ a.s., and Step 3 of the proof of Theorem 5 applies with $\mathcal { P } _ { 0 } ^ { \omega , \otimes }$ in place of $\mathcal { P } _ { 0 } ^ { \omega }$

Proof [Proof of Theorem 9]Assume (i) and write $C _ { 0 } = ( Z ^ { \omega _ { Z } , n } , \hat { \beta } , \hat { \gamma } )$ . By Lemma 2, $H _ { 0 } ^ { \omega }$ is equivalent to $X ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid { \cal C } _ { 0 }$ . By (i), contraction gives

$$
X ^ { \omega _ { X } , n } \perp \perp ( Y , Z ) ^ { n } \mid { \cal C } _ { 0 } .
$$

Since $Z ^ { n }$ is a measurable function of $( Y , Z ) ^ { n }$ , weak union yields $X ^ { \omega _ { X } , n }$ ⊥⊥ $( Y , Z ) ^ { n } \mid ( C _ { 0 } , Z ^ { n } )$ and decomposition yields $K ^ { \omega _ { X } , n } \bot \bot Y ^ { n } \mid ( Z ^ { n } , Z ^ { \omega _ { Z } , n } , \hat { \beta } , \hat { \gamma } )$ . As $Z ^ { \omega _ { Z } , n }$ is $\sigma ( Z ^ { n } , \hat { \gamma } )$ -measurable, this is

$$
X ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid ( Z ^ { n } , \hat { \beta } , \hat { \gamma } ) .\tag{17}
$$

Applying contraction gives $\left( { \hat { \gamma } } , X ^ { \omega _ { X } , n } \right) \bot \bot Y ^ { n } \mid \left( Z ^ { n } , { \hat { \beta } } \right)$ by Assumption 8a), and decomposition gives

$$
X ^ { \omega _ { X } , n } \perp \perp Y ^ { n } \mid ( Z ^ { n } , \hat { \beta } ) ,
$$

which by Lemma 2 is $H _ { 0 } ^ { \omega _ { X } }$ . Under (ii) the same argument applies with the roles of $X ^ { \omega _ { X } , n }$ and $Y ^ { n }$ interchanged, using the symmetry of conditional independence.

Proof [Proof of Corollary 10]Write $C _ { 0 } ~ = ~ ( Z ^ { \omega _ { Z } , n } , \hat { \beta } , \hat { \gamma } )$ . By symmetry of $\mathrm { C I } , \ \left( * \right)$ reads $Z ^ { n } \perp \perp ( { \bar { X } } ^ { \omega _ { X } , n } , Y ^ { n } ) \mid C _ { 0 }$ , so decomposition gives both $X ^ { \omega _ { X } , n } \perp \perp Z ^ { n } \mid C _ { 0 }$ and $Y ^ { n }$ ⊥⊥ $Z ^ { n } \mid C _ { 0 }$ i.e. conditions (i) and (ii) of Theorem 3. Moreover, weak union applied to (∗) with the measurable functions $Y ^ { n }$ and $X ^ { \omega _ { X } , n }$ of $( X ^ { \omega _ { X } , n } , Y ^ { n } )$ , each followed by decomposition, gives $X ^ { \omega _ { X } , n } ~ \bot \bot ~ Z ^ { n } ~ | ~ ( C _ { 0 } , Y ^ { n } )$ and $Y ^ { n } \perp \perp Z ^ { n } \mid ( C _ { 0 } , X ^ { \omega _ { X } , n } )$ , i.e. conditions (i) and (ii) of Theorem 9. The claim now follows from Theorems 3 and 9.

Proof [Proof of Theorem 12]Write $\mu _ { i } : = \mathbb { E } [ Y _ { i } \mid Z ^ { n } ]$ , which is well defined since $\mathbb { E } \| Y _ { 1 } \| < \infty$

Step 1. Under $H _ { 0 }$ the rows of S are i.i.d. with $X \perp \perp Y \mid Z$ , so $X ^ { n }$ ⊥⊥ $Y ^ { n } \mid Z ^ { n }$ by Lemma 19. Decomposition applied to the measurable function $Y _ { i }$ of $Y ^ { n }$ gives $X ^ { n } \perp \perp Y _ { i } \mid Z ^ { n }$ hence

$$
\mathbb { E } \left[ Y _ { i } \mid X ^ { n } , Z ^ { n } \right] = \mu _ { i } \quad { \mathrm { a . s . ~ f o r ~ a l l ~ } } i \leq n .
$$

Step 2. By Assumption 11, $\mathbb { E } [ Y _ { i } \mid X ^ { n } , Z ^ { n } , { \hat { \beta } } ] = \mu _ { i } { \mathrm { ~ a . s . } }$

Step 3. Since ${ X ^ { \omega _ { X } , n } } = \omega _ { X } ( X ^ { n } , \hat { \beta } )$ is $\sigma ( X ^ { n } , \hat { \beta } )$ -measurable, we have $\sigma ( X ^ { \omega _ { X } , n } , Z ^ { n } , { \hat { \beta } } ) \subseteq$ $\sigma ( X ^ { n } , Z ^ { n } , { \hat { \beta } } )$ , so the tower property and Step 2 give

$$
\operatorname { \mathbb { E } } \left[ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , { \hat { \beta } } \right] = \operatorname { \mathbb { E } } \left[ \mu _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , { \hat { \beta } } \right] = \mu _ { i } ,
$$

the last equality because $\mu _ { i }$ is $\sigma ( Z ^ { n } )$ -measurable and $\sigma ( Z ^ { n } ) \subseteq \sigma ( X ^ { \omega _ { X } , n } , Z ^ { n } , { \hat { \beta } } )$

Step $\it 4 .$ Since $\sigma ( Z ^ { n } , \hat { \beta } ) \subseteq \sigma ( X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \beta } )$ , the tower property applied to Step 3 gives $\mathbb { E } [ Y _ { i } \mid Z ^ { n } , { \hat { \beta } } ] = \mathbb { E } [ \mu _ { i } \mid Z ^ { n } , { \hat { \beta } } ] = \mu _ { i }$

Steps 3 and 4 give $\mathbb { E } [ \dot { Y } _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \beta } ] = \mathbb { E } [ Y _ { i } \mid Z ^ { n } , \hat { \beta } ]$ a.s. for all $i \ \leq \ n$ , which by Lemma 18 applied with $V = S ^ { \omega _ { X } } , W = \hat { \beta }$ and $( A , B , C ) = ( X ^ { \omega _ { X } , n } , Y ^ { n } , Z ^ { n } )$ is $H _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } }$ .

Proof [Proof of Theorem 14]Step 1: restating the hypotheses. By Lemma 18 applied with $V = \bar { S } ^ { \bar { \omega _ { X } } } , W = \hat { \beta }$ and $( A , B , C ) = ( X ^ { \omega _ { X } , n } , Y ^ { n } , Z ^ { n } )$ , the hypothesis $H _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } }$ is equivalent to

$$
\mathbb { E } \left[ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \beta } \right] = \mathbb { E } \left[ Y _ { i } \mid Z ^ { n } , \hat { \beta } \right] = m _ { i } \quad \mathrm { a . s . ~ f o r ~ a l l ~ } i \leq n ,\tag{18}
$$

and by the same lemma with $V = S ^ { \omega } , W = \hat { \theta }$ and $( A , B , C ) = ( X ^ { \omega _ { X } , n } , Y ^ { n } , Z ^ { \omega _ { Z } , n } )$ , the hypothesis $H _ { 0 } ^ { \omega , \mathrm { m e a n } }$ is equivalent to

$$
\mathbb { E } \left[ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { \omega _ { Z } , n } , \hat { \theta } \right] = \mathbb { E } \left[ Y _ { i } \mid Z ^ { \omega _ { Z } , n } , \hat { \theta } \right] \quad \mathrm { a . s . ~ f o r ~ a l l ~ } i \leq n .\tag{19}
$$

All conditional expectations are well defined, since the rows of S are i.i.d. and $\mathbb { E } \| Y _ { 1 } \| < \infty$ Step 2: adjoining γˆ. Assumption 13 and (18) give

$$
\operatorname { \mathbb { E } } \left[ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \theta } \right] = \operatorname { \mathbb { E } } \left[ Y _ { i } \mid X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \beta } \right] = m _ { i } .
$$

Since $\sigma ( Z ^ { n } , \hat { \theta } ) \subseteq \sigma ( X ^ { \omega _ { X } , n } , Z ^ { n } , \hat { \theta } )$ and $m _ { i }$ is $\sigma ( Z ^ { n } , \hat { \beta } )$ -measurable, hence $\sigma ( Z ^ { n } , \hat { \theta } )$ -measurable, the tower property yields $\mathbb { E } [ Y _ { i } \mid Z ^ { n } , \hat { \theta } ] = \mathbb { E } [ m _ { i } \mid Z ^ { n } , \hat { \theta } ] = m _ { i }$ . Neither side of the text-level mean null is therefore changed by adjoining $\hat { \gamma }$ to the conditioning set.

Step 3: replacing $Z ^ { n }$ by $Z ^ { \omega _ { Z } , n }$ . As $Z ^ { \omega _ { Z } , n }$ is $\sigma ( Z ^ { n } , \hat { \gamma } )$ -measurable, we have $\sigma ( X ^ { \omega _ { X } , n } , Z ^ { \omega _ { Z } , n } , \hat { \theta } ) \subseteq$ $\sigma ( X ^ { \omega _ { X } , n } , Z ^ { n } , { \hat { \theta } } )$ and $\sigma ( Z ^ { \omega _ { Z } , n } , \hat { \theta } ) \subseteq \sigma ( Z ^ { n } , \hat { \theta } )$ . Applying the tower property to the two identities of Step 2 across these inclusions gives

$$
\begin{array} { r } { \mathbb { E } \big [ Y _ { i } \mid X ^ { \omega x , n } , Z ^ { \omega z , n } , \hat { \theta } \big ] = \mathbb { E } \big [ m _ { i } \mid X ^ { \omega x , n } , Z ^ { \omega z , n } , \hat { \theta } \big ] , \qquad \mathbb { E } \big [ Y _ { i } \mid Z ^ { \omega z , n } , \hat { \theta } \big ] = \mathbb { E } \big [ m _ { i } \mid Z ^ { \omega z , n } , \hat { \theta } \big ] . } \end{array}\tag{20}
$$

Under (ii<sup>′</sup>) both right-hand sides equal $m _ { i } \ \mathrm { a . s . }$ , so the two left-hand sides agree for all $i ,$ which is (19) and hence $H _ { 0 } ^ { \omega , \mathrm { m e a n } }$ ■

Proof [Proof of Corollary 15]Fix such a $\mathbb { P } ^ { S }$ . Since $\mathbb { P } ^ { S } \in { \mathcal { P } } _ { 0 }$ and Assumption 11 holds, Theorem 12 gives $H _ { 0 } ^ { \omega _ { X } , \mathrm { m e a n } }$ ; since Assumption 13 and (ii<sup>′</sup>) hold, Theorem 14 gives $H _ { 0 } ^ { \omega , \mathrm { m e a n } }$ ， that is, $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } } \in \mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } } \mathbb { P } ^ { \hat { \theta } _ { - \mathrm { a . S } } }$

Since $\hat { \theta } \perp \perp \boldsymbol { S }$ , the conditional law of S given $\hat { \theta }$ is $\mathbb { P } ^ { S }$ itself, and conditionally on $\widehat { \theta } = ( b , c )$ the rows of $S ^ { \omega }$ are the images of the i.i.d. rows of S under the fixed measurable map $( x , y , z ) \mapsto ( \omega _ { X } ( x , b ) , y , \omega _ { Z } ( z , c ) )$ . Hence $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } }$ is an n-fold product $M ^ { \otimes n }$ of a common marginal $\mathbb { P } ^ { \hat { \theta } _ { - \mathrm { a . s . } } }$ , and Lemma 20 applied to $M ^ { \otimes n }$ gives $M ^ { \otimes n } \in \mathcal { P } _ { 0 } ^ { \omega }$ ,mean,⊗ a.s.

Step 3 of the proof of Theorem 5 applies verbatim with $\mathcal { P } _ { 0 } ^ { \omega , \mathrm { m e a n } , \otimes }$ in place of $\mathcal { P } _ { 0 } ^ { \omega }$ , since it uses only that $S ^ { \omega }$ is $\sigma ( S , \hat { \theta } )$ -measurable, that $U \perp \perp ( S , \hat { \theta } )$ , and that $\psi$ is valid uniformly over a class containing $\mathbb { P } ^ { S ^ { \omega } | \hat { \theta } }$ almost surely.

## Appendix B. Application

## B.1 Source Texts, LLMs, and Summary Generation

We clean the German Parliament speeches from Richter et al. (2020) by removing parenthetical text such as applause or laughter, deleting common procedural phrases (e.g., “Das Wort hat”), stripping out text between dashes and extraneous whitespace, and normalizing quotation marks. Furthermore, we consider speeches as valid after filtering by minimum word count (50) and minimum sentences (2), and excluding speeches containing procedural indicators. For each speaker, we merge consecutive speeches if no valid speech from another speaker intervenes. We treat invalid interruptions from other speakers as [UN-TERBRECHUNG/REDE:...] appended to the main speech. We randomly sample cleaned speeches and manually inspect for remaining procedural or invalid content.

Next, we summarize the cleaned speeches Z with the LLM Gemma 4 12B (Gemma Team, 2026), used throughout the simulation study (Subsection 4.2) and the real-world application (Subsection 4.3). We allow for a maximum of 256 generated tokens per summary (3–4 sentences) and use greedy decoding so that the summaries are reproducible. For the real-world application (Subsection 4.3), we also summarize the source texts Z with the LLM Qwen 3.6 27B (Qwen Team, 2026). We again allow for a maximum of 256 generated tokens per summary (3–4 sentences) and sample with temperature 0.7, top-p 0.8, and top-k 20, the recommended decoding parameters for the model’s non-thinking (Instruct) mode.

The summaries X are generated by prompting the LLM with Du bist ein erfahrener politischer Analyst. Deine Aufgabe ist es, Reden aus dem deutschen Bundestag pr¨agnant und objektiv in einem Fließtext zusammenzufassen. Fokussiere auf die Hauptargumente und politischen Positionen. Die mit $\because ( \{ \backslash \} \backslash d + \} ) ^ { \colon }$ gekennzeichneten Bemerkungen im Text beziehen sich auf m¨ogliche Zwischenrufe oder kleinere Beitr¨age von anderen Abgeordneten. Die Rede kann außerdem [REDE] und [UNTERBRECHUNG] Marker enthalten. [REDE] markiert Teile der Hauptrede. [UNTERBRECHUNG] markiert Zwischenkommentare oder Fragen. WICHTIG: Gib NUR die Zusammenfassung zur¨uck, ohne einleitende Phrasen wie ’Hier ist eine Zusammenfassung’ oder <sup>¨</sup>Ahnliches.

## B.2 Embedding Maps

We embed speeches Z and summaries X with four multilingual models, using the same encoders in the simulation study (Subsection 4.2) and the real-world application (Subsection 4.3): BGE-M3 (Chen et al., 2024) at its native 1024 dimensions; jina-embeddings-v5- text-small (Akram et al., 2026) with the classification adapter, truncated to 32 dimensions; Llama Nemotron Embed 1B $\mathrm { v 2 }$ (NVIDIA, 2025), truncated to 32 dimensions; and Qwen3- Embedding-8B (Zhang et al., 2025), likewise truncated to 32 dimensions. The three 32- dimensional representations use Matryoshka truncation of the native embedding (Kusupati et al., 2022). Each embedding is $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ . We then column-standardize the coordinates (train-fitted mean and standard deviation) so that features enter the simulation DGP for $Y$ on a comparable scale, and because ℓ<sub>2</sub>-penalized PCM nuisances are sensitive to the scale of X and Z.

As additional representations we use learned embedding maps: we fit the ModernG-BERT 134M (Wunderle et al., 2025), a German encoder with 768-dimensional hidden states, to predict Y from the raw speech or summary, and take the masked mean of the last hidden layer. In the real-world application these models predict faction or gender on the training split, and the 768-dimensional hidden states on the test split are supplied to the CIT, either alone or concatenated with the pretrained embeddings. In the simulation study the same architecture is trained to predict the simulated outcome, and the resulting hidden states are used in the simulation.

In the simulation study we generate Y from a single 32-dimensional embedding of the Gemma summaries (Jina or Qwen3) and supply the CIT with the following concatenations of $Z ^ { \omega _ { Z } }$ and $X ^ { \omega _ { X } }$ . Oracle uses only that DGP pair: $Z ^ { \widetilde \omega _ { Z } } , X ^ { \widetilde \omega _ { X } } \in \mathbb { R } ^ { 3 2 }$ . The diluted eCITs prepend the generating embedder to three further concatenations: BGE with the two remaining 32-dimensional models $( Z ^ { \omega _ { Z } } , X ^ { \omega _ { X } } \in \mathbb { R } ^ { 1 1 2 0 } )$ ; the 768-dimensional learned speech hidden state, BGE, and those two models as $Z ^ { \omega _ { Z } } ~ \in ~ \mathbb { R } ^ { 1 8 8 8 }$ , and the learned summary hidden state with those two models as $X ^ { \omega _ { X } } \in \mathbb { R } ^ { 8 6 4 }$ ; and the learned hidden states alone $( Z ^ { \omega _ { Z } } , X ^ { \omega _ { X } } \in \mathbb { R } ^ { 8 0 0 } )$ . The eCITs omitting the generating embedder use the same three stacks without the DGP embedder: BGE with the two remaining 32-dimensional models $( Z ^ { \omega _ { Z } } , X ^ { \omega _ { X } } \in \mathbb { R } ^ { 1 0 8 8 } ) \colon$ ; the learned speech hidden state, BGE, and those two models as $Z ^ { \omega _ { Z } } \in \mathbb { R } ^ { 1 8 5 6 }$ , and the learned summary hidden state with those two models as $X ^ { \omega _ { X } } \in \mathbb { R } ^ { 8 3 2 }$ and the learned hidden states alone $( Z ^ { \omega _ { Z } } , X ^ { \omega _ { X } } \in \mathbb { R } ^ { 7 6 8 } )$ .

## B.3 Outcome Generation in the Simulation Study

We set $\alpha _ { Z } = \alpha _ { X } = 1$ throughout. For the continuous outcome, the noise variance is $\sigma ^ { 2 } = ( 1 - r ^ { 2 } ) / r ^ { 2 }$ with $r ^ { 2 } = 0 . 4$ , so that $s _ { Z }$ explains 40% of the variance of Y under the null; for the binary and multinomial outcomes the noise is determined by the link, so that the three outcome types are not on a common signal scale.

Write $\bar { s } _ { Z }$ for the training-split mean of the speech score. The binary intercept is $b = - \alpha _ { Z } \bar { s } _ { Z }$ . For the multinomial outcome, the reference class has logit $\eta _ { i , 0 } ~ = ~ 0$ and the remaining logits are

$$
\begin{array} { r } { \eta _ { i , k } = \cos \Bigl ( \frac { 2 \pi k } { L } \Bigr ) \alpha _ { Z } \bigl ( s _ { Z , i } - \bar { s } _ { Z } \bigr ) + c \sin \Bigl ( \frac { 2 \pi k } { L } \Bigr ) \alpha _ { X } s _ { X , i } , \qquad k = 1 , \ldots , L - 1 , } \end{array}
$$

so that the speech and the summary score load on the classes in orthogonal directions. The logits are clipped to [−10, 10] before the softmax, and the resulting probabilities are floored at $1 0 ^ { - 6 }$ and renormalized.

## Appendix C. Detailed Simulation Results C.1 Jina Embedding Data Generating Mechanism

![](images/df05b88d0817649361fd15c0a125cd2c17c0275c4b74017cd31d0466006c1dcd.jpg)  
Figure 3: The rejection rates of the eCIT for binary, multinomial and continuous $Y$ generated under the null (first row) and alternative (second row) from the Jina embeddings $Z ^ { J i n a }$ and $X ^ { J i n a }$ . The eCIT is applied with the Jina embeddings (oracle), the specified diluted embeddings, including additionally learned, BGE, Qwen, and Llama embeddings, as well as the same combinations excluding the Jina embedding. For the PCM, the nuisance regressions on the training dataset are $L _ { \mathrm { 2 - p e n a l i z e d } }$ and those on the evaluation set unpenalized; for categorical $Y$ the regressions of $Y$ are multinomial logistic and those of the fitted probabilities and of $\hat { f }$ are least squares (Appendix D.1). The dashed line marks the nominal level $\alpha = 0 . 0 5$ , the shaded area marking two MC standard errors around it.

![](images/6b12ec9fcee93ff9eb04bac0b0e0b329adcc3030a70131e6002db8b05a3354e2.jpg)  
Figure 4: QQ-plots of the observed p-values against the theoretical quantiles of a uniform distribution on the interval [0, 1] for eCITs with the specified embeddings. For the PCM, the nuisance regressions on the training dataset are $L _ { \mathrm { { 2 - p e n a l i z e d } } }$ and those on the evaluation set unpenalized; for categorical $Y$ the regressions of $Y$ are multinomial logistic and those of the fitted probabilities and of $\hat { f }$ are least squares (Appendix D.1). Y is generated under the null hypothesis from the Jina embedding $Z ^ { J i n a }$ at selected sample sizes (indicated in the panel titles). The diagonal black lines $y = x$ serve as references for the theoretical quantiles of the uniform distribution on [0, 1]. If the p-values are uniformly distributed, they should align along this line.

## C.2 Qwen Embedding Data Generating Mechanism

![](images/d90a4b66e5c084c16559918876d42d8b714a18f086f65de9b12b84eb23153664.jpg)  
Figure 5: The rejection rates of the eCIT for binary, multinomial and continuous $Y$ generated under the null (first row) and alternative (second row) from the Qwen embeddings $Z ^ { Q w e n }$ and $X ^ { Q w e n }$ . The eCIT is applied with the Qwen embeddings (oracle), the specified diluted embeddings, including additionally learned, BGE, Jina, and Llama embeddings, as well as the same combinations excluding the Qwen embedding. For the PCM, the nuisance regressions on the training dataset are $L _ { \mathrm { 2 - p e n a l i z e d } }$ and those on the evaluation set unpenalized; for categorical Y the regressions of $Y$ are multinomial logistic and those of the fitted probabilities and of $\hat { f }$ are least squares (Appendix D.1). The dashed line marks the nominal level $\alpha = 0 . 0 5$ , the shaded area marking two MC standard errors around it.

![](images/ed41e8ea65e5fadd0937167285055778f751e79ff9638f5f02cb093a98ea59ea.jpg)  
Figure 6: QQ-plots of the observed p-values against the theoretical quantiles of a uniform distribution on the interval [0, 1] for eCITs with the specified embeddings. For the PCM, the nuisance regressions on the training dataset are $L _ { \mathrm { { 2 - p e n a l i z e d } } }$ and those on the evaluation set unpenalized; for categorical $Y$ the regressions of $Y$ are multinomial logistic and those of the fitted probabilities and of $\hat { f }$ are least squares $( \mathrm { A p p e n d i x ~ D . 1 } )$ . Y is generated under the null hypothesis from the Qwen embedding $Z ^ { Q w e n }$ at selected sample sizes (indicated in the panel titles). The diagonal black lines $y = x$ serve as references for the theoretical quantiles of the uniform distribution on [0, 1]. If the p-values are uniformly distributed, they should align along this line.

## Appendix D. The PCM for Categorical Y

We suppress the embedding maps throughout this appendix and write $X \ \in \ \mathcal { X } \ \subseteq \ \mathbb { R } ^ { q }$ $Z \in \mathcal { Z } \subseteq \mathbb { R } ^ { p }$ for $X ^ { \omega _ { X } } , Z ^ { \omega _ { Z } }$ , so that $p = \dim ( { \mathcal { Z } } )$ . Let Y take L values, drop a baseline class and let $\tilde { Y } \in \{ 0 , 1 \} ^ { K } , K = L - 1$ , be the reduced vector of class indicators of Remark 16. Write

$$
g ( x , z ) = \mathbb { E } [ \tilde { Y } \mid X = x , Z = z ] , \qquad m _ { Y } ( z ) = \mathbb { E } [ \tilde { Y } \mid Z = z ] , \qquad h = g - m _ { Y } ,
$$

so that $g$ is the reduced conditional probability mass function of Y and $m _ { Y }$ is the population version of the regression $m _ { i }$ of Subsection 3.2, and let $\Sigma ( x , z ) = \mathrm { V a r } ( \tilde { Y } \mid X = x , Z = z ) =$ diag $\{ g ( x , z ) \} - g ( x , z ) g ( x , z ) ^ { \top }$

## D.1 The Categorical PCM

Following Lundborg et al. (2024), we split the sample into halves $D _ { 1 } , D _ { 2 }$ of sizes $n _ { 1 }$ , n<sub>2</sub>. On $D _ { 1 }$ we fit ${ \hat { g } } ,$ a multinomial $\left( L _ { \mathrm { 2 - p e n a l i z e d } } \right)$ logistic regression of $Y$ on $( X , Z )$ , and write $\hat { \boldsymbol g } ( \boldsymbol x , \boldsymbol z ) \in \mathbb R ^ { K }$ for its vector of fitted probabilities with the baseline class dropped, so that $\hat { g }$ estimates $g$ . Then we fit $\tilde { m }$ , a column-wise $\left( L _ { \mathrm { 2 - p e n a l i z e d } } \right)$ least-squares regression of the fitted probabilities $\hat { g }$ on $Z ,$ and set $\hat { h } = \hat { g } - \tilde { m }$ and

$$
\hat { f } ( x , z ) = \left( \hat { \Sigma } ( x , z ) + \hat { c } I _ { K } \right) ^ { - 1 } \hat { h } ( x , z ) \in \mathbb { R } ^ { K } ,\tag{21}
$$

where $\hat { \Sigma }$ is $\Sigma$ evaluated at $\hat { g } ( x , z )$ , the entries of the full fitted probability vector being floored at $\varepsilon = 1 0 ^ { - 6 }$ and renormalized before the baseline class is dropped.

The ridge penalty ˆc is selected on $D _ { 1 }$ . It is needed because class probabilities close to 0 or 1, as under quasi-complete separation, leave $\hat { \Sigma }$ nearly singular and the inverse in (21) unstable. We therefore split $D _ { 1 }$ into $D _ { 1 a }$ and $D _ { 1 b } ,$ , fit $\hat { g }$ and $\tilde { m }$ on $D _ { 1 a }$ , and for each candidate value of $c$ on a logarithmically spaced grid in $[ 1 0 ^ { - 6 } , 1 ]$ form the direction (21) and the corresponding scores $\ell _ { i }$ of (22) on $D _ { 1 b }$ , with the nuisance regressions $\hat { m } _ { Y }$ and $\hat { m } _ { f }$ fitted on $D _ { 1 b }$ as well. We select the candidate maximizing $\bar { \ell } / \hat { \sigma } _ { \ell }$ on $D _ { 1 b }$ and refit ${ \hat { g } } ,$ ˜m and the resulting direction on the whole of $D _ { 1 }$ at the selected ˆc. Since ˆc is a function of $D _ { 1 }$ alone, $\hat { f }$ remains $D _ { 1 }$ -measurable and the selection afects the power of the test but not the argument for its level.

On $D _ { 2 }$ we fit $\hat { m } _ { Y }$ , an unpenalized multinomial logistic regression of $Y$ on $Z ,$ , again reduced to its $K$ non-baseline fitted probabilities, and $\hat { m } _ { f } .$ , an unpenalized least-squares regression of $\hat { f }$ on $Z ,$ , and form $\hat { r } _ { i } ^ { Y } = \tilde { Y } _ { i } - \hat { m } _ { Y } ( Z _ { i } )$ and $\hat { r } _ { i } ^ { f } = \hat { f } ( X _ { i } , Z _ { i } ) - \hat { m } _ { f } ( Z _ { i } )$ . The test statistic is

$$
T = \sqrt { n _ { 2 } } \frac { \bar { \ell } } { \hat { \sigma } _ { \ell } } , \qquad \ell _ { i } = \langle \hat { r } _ { i } ^ { Y } , \hat { r } _ { i } ^ { f } \rangle , \quad \bar { \ell } = \frac { 1 } { n _ { 2 } } \sum _ { i \in D _ { 2 } } \ell _ { i } , \quad \hat { \sigma } _ { \ell } ^ { 2 } = \frac { 1 } { n _ { 2 } } \sum _ { i \in D _ { 2 } } ( \ell _ { i } - \bar { \ell } ) ^ { 2 } ,\tag{22}
$$

and we reject at level α when $T > z _ { 1 - \alpha }$ . This is Algorithm 1 of Lundborg et al. (2024), as implemented for continuous $Y$ in the R package comets (Kook and Lundborg, 2024), with their scalar direction replaced by (21) and their product $\hat { r } _ { i } ^ { Y } \hat { r } _ { i } ^ { f }$ by an inner product. The ridge constant ˆc has no counterpart there, and is required here to invert $\hat { \Sigma }$ stably.

## D.2 Validity

The summands $\ell _ { i }$ in (22) are scalar, since the K-vector residuals are projected onto one dimension by the inner product before averaging, so the numerator remains an average of real-valued summands that are conditionally independent and, under the null, conditionally mean-zero given $D _ { 1 }$ . The argument of Lundborg et al. (2024, Theorem 4) is therefore available with their product $\boldsymbol { \epsilon } _ { i } \xi _ { i }$ replaced by $\langle \epsilon _ { i } , \xi _ { i } \rangle$ , provided the conditions it places on the nuisance estimates are read in vector form. We record these conditions below; a detailed verification, including vector restatements of the supporting lemmas, is left to future work.

Assumption 21 $L e t \epsilon = \tilde { Y } - g ( X , Z ) , m _ { f } ( z ) = \mathbb { E } [ \hat { f } ( X , Z ) \mid Z = z , D _ { 1 } ] , \sigma ^ { 2 } = \mathrm { V a r } ( \langle \epsilon , \hat { f } ( X , Z ) - \epsilon , D _ { 1 } \rangle ) ,$ $m _ { f } ( Z ) \rangle \ | \ D _ { 1 } \rangle , \ E _ { 1 } = \mathbb { E } [ \Vert \hat { m } _ { Y } ( Z ) - m _ { Y } ( Z ) \Vert _ { 2 } ^ { 2 } \ \vert \ D _ { 1 } \rangle \ a n d \ E _ { 2 } = \sigma ^ { - 2 } \mathbb { E } [ \Vert \hat { m } _ { f } ( Z ) - m _ { f } ( Z ) \Vert _ { 2 } ^ { 2 } \ \vert \ D _ { 1 } \rangle \ a n d \ E _ { 2 } = \sigma ^ { - 2 } \mathbb { E } [ \Vert \hat { m } _ { f } ( Z ) - m _ { f } ( Z ) \Vert _ { 2 } ^ { 2 } \ \vert \ D _ { 1 } \rangle \ a n d \ E _ { 2 } = 0 ]$ Assume $\mathrm { \Sigma } ( a ) ~ E _ { 1 } E _ { 2 } = o _ { \mathbb { P } } ( n _ { 2 } ^ { - 1 } ) ; ~ ( b ) ~ \sigma ^ { - ( 2 + \delta ) } \mathbb { E } [ \| \hat { f } ( X , Z ) - m _ { f } ( Z ) \| _ { 2 } ^ { 2 + \delta } ~ | ~ { \cal D } _ { 1 } ] = O _ { \mathbb { P } } ( 1 ) ~ f o r ~ n _ { 2 } = \delta ,$ some $\delta > 0 ;$ and $( c ) \lambda _ { \operatorname* { m i n } } ( \Sigma ( X , Z ) ) \geq c _ { 0 }$ almost surely for some $c _ { 0 } > 0$

Two features of the categorical case are worth noting. First, by Remark 16 the conditional mean of $\tilde { Y }$ is the conditional distribution of $Y ,$ so conditional mean independence of $\tilde { Y }$ and $X \perp \perp Y \mid Z$ are the same statement and the mean-level null class coincides with the CI null class for the i.i.d. laws considered in Corollary 15. The PCM therefore tests the CI hypothesis itself here, and nothing is lost by using a CMIT. Second, $\tilde { Y }$ is bounded, which makes the moment condition (b) a statement about $\| { \hat { f } } - m _ { f } \| _ { 2 }$ alone. Condition (c), by contrast, requires every class probability to be bounded away from 0 and 1, which is a substantive restriction when L is large and some classes are rare.

We do not claim uniform asymptotic level over a nonparametric class for this variant. Null calibration is instead assessed by simulation in Subsection 4.2 and Appendix C.

## D.3 The Precision-Weighted Direction

The weighting in (21) is motivated by the following, which identifies the direction maximizing the population signal-to-noise ratio. At $K = 1$ it gives $f \propto h / v$ with $v = \mathrm { V a r } ( \tilde { Y } \mid X , Z )$

Proposition 22 Suppose $\Sigma ( X , Z )$ is almost surely positive definite with $0 < \mathbb { E } [ h ^ { \top } \Sigma ^ { - 1 } h ] <$ ∞, and let $\mathcal { F }$ be the set of measurable $f : \mathcal { X } \times \mathcal { Z } \to \mathbb { R } ^ { K }$ with $0 < \mathbb { E } [ f ^ { \top } \Sigma f ] < \infty$ , all functions being evaluated at $( X , Z )$ . Then

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \frac { \left( \mathbb { E } \langle h , f \rangle \right) ^ { 2 } } { \mathbb { E } \left[ f ^ { \top } \Sigma f \right] } = \mathbb { E } \big [ h ^ { \top } \Sigma ^ { - 1 } h \big ] ,\tag{23}
$$

and the supremum is attained at $f \in { \mathcal { F } }$ if and only if $f = \kappa \Sigma ^ { - 1 } h$ almost surely for some $\kappa > 0$

Proof Let $\Sigma ^ { 1 / 2 }$ be the symmetric positive definite square root of Σ and put $u = \Sigma ^ { - 1 / 2 } h$ ， $w = \Sigma ^ { 1 / 2 } f .$ , which are measurable since $A \mapsto A ^ { \pm 1 / 2 }$ is continuous on the open cone of positive definite matrices, and satisfy $\mathbb { E } \| u \| _ { 2 } ^ { 2 } = \mathbb { E } [ h ^ { \top } \Sigma ^ { - 1 } h ]$ and $\mathbb { E } \| w \| _ { 2 } ^ { 2 } = \mathbb { E } [ f ^ { \top } \Sigma f ]$ , both in $( 0 , \infty )$ . Pointwise, $\langle h , f \rangle = \langle u , w \rangle \leq \| u \| _ { 2 } \| w \| _ { 2 }$ almost surely, so taking expectations and applying Cauchy–Schwarz in $L ^ { 2 } ( \mathbb { P } )$ ，

$$
\begin{array} { r } { { \mathbb { E } } \langle h , f \rangle \leq { \mathbb { E } } \big [ \| u \| _ { 2 } \| w \| _ { 2 } \big ] \leq \big ( { \mathbb { E } } [ h ^ { \top } \Sigma ^ { - 1 } h ] \big ) ^ { 1 / 2 } \big ( { \mathbb { E } } [ f ^ { \top } \Sigma f ] \big ) ^ { 1 / 2 } . } \end{array}\tag{24}
$$

Since $f$ and $- f$ share the denominator in (23), we may assume $\mathbb { E } \langle h , f \rangle > 0 ;$ squaring and dividing bounds the left-hand side of (23) by $\mathbb { E } [ h ^ { \top } \Sigma ^ { - 1 } h ]$ , with equality at $f = \Sigma ^ { - 1 } h \in \mathcal { F }$ If $f \in \mathcal { F }$ attains the supremum, both inequalities in (24) are equalities. The first forces $\langle u , w \rangle = \| u \| _ { 2 } \| w \| _ { 2 }$ almost surely, the second $\| w \| _ { 2 } = \kappa \| u \| _ { 2 }$ almost surely for some $\kappa \in [ 0 , \infty )$ , with $\kappa > 0$ since $\mathbb { E } \Vert w \Vert _ { 2 } ^ { 2 } > 0 . \mathrm { ~ O n ~ } \left\{ u \neq 0 \right\}$ the equality case in $\mathbb { R } ^ { K }$ gives $w = \kappa u ,$ and on $\{ u = 0 \}$ we have $w = 0 , \bar { \mathrm { ~ s o ~ } } \Sigma ^ { 1 / 2 } f = \kappa \Sigma ^ { - 1 / 2 } h$ almost surely. The converse is the computation above.

Remark 23 The constant cˆ in (21) serves only to invert $\hat { \Sigma }$ stably when fitted probabilities are close to the boundary, and has no counterpart in Algorithm 1 of Lundborg et al. $( 2 0 \% )$ 2 whose Step $\mathcal { Q } ( i i )$ calibrates a constant of the same name against a variance identity. We apply no $\mathrm { s i g n } ( \rho )$ correction. Since $\hat { \Sigma } + \hat { c } I _ { K }$ is positive definite, $\langle \hat { h } _ { i } , \hat { f } _ { i } \rangle \geq 0$ by construction.

## D.4 Nuisance Estimation on the Evaluation Half

The evaluation-half nuisances are fitted unpenalized and in sample, which yields an exact finite-sample orthogonality. Let $W \in \mathbb { R } ^ { n _ { 2 } \times ( p + 1 ) }$ have rows $W _ { i } = ( 1 , Z _ { i } ^ { \top } )$ and collect the residuals in $\hat { R } ^ { Y } , \hat { R } ^ { f } \in \mathbb { R } ^ { n _ { 2 } \times K }$ . The score equations of the multinomial logistic fit give $W ^ { \top } \hat { R } ^ { Y } = 0$ and the normal equations of the least-squares fit give $W ^ { \top } \hat { R } ^ { f } = 0$ . Since $\hat { m } _ { f } ( Z _ { i } ) = W _ { i } \hat { B }$ for some $\hat { B } \in \mathbb { R } ^ { ( p + 1 ) \times K }$ , the first identity yields

$$
\sum _ { i \in D _ { 2 } } \langle \hat { r } _ { i } ^ { Y } , \hat { r } _ { i } ^ { f } \rangle = \sum _ { i \in D _ { 2 } } \langle \hat { r } _ { i } ^ { Y } , \hat { f } ( X _ { i } , Z _ { i } ) \rangle ,\tag{25}
$$

so the regression of $\hat { f }$ on $Z$ does not contribute to the numerator. Under the null, writing $\hat { r } _ { i } ^ { Y } = \epsilon _ { i } - M _ { i }$ with $M _ { i } = \hat { m } _ { Y } ( Z _ { i } ) - m _ { Y } ( Z _ { i } )$ , the numerator splits as $\Sigma _ { i \in D _ { 2 } } \langle \epsilon _ { i } , \hat { r } _ { i } ^ { f } \rangle -$ $\textstyle \sum _ { i \in D _ { 2 } } \langle M _ { i } , \hat { r } _ { i } ^ { f } \rangle$ , in which the first term is conditionally mean-zero given $( X , Z ) ^ { n _ { 2 } }$ and $D _ { 1 }$ because $\hat { r } ^ { f }$ does not depend on $Y$ on $D _ { 2 }$ , leaving the second as the only bias term.

A Cauchy–Schwarz bound on that term is of order $\sqrt { n _ { 2 } } ( E _ { 1 } E _ { 2 } ) ^ { 1 / 2 }$ , which is negligible under Assumption 21(a) and requires the two nuisance errors to be small relative to $n _ { 2 } ^ { - 1 / 2 }$ jointly. Each evaluation-half nuisance is an unpenalized fit with $K ( p + 1 )$ free parameters, and in Subsection 4.3 the conditioning sets consist of high-dimensional embeddings with $p$ between 768 and 1888 at n<sub>2</sub> of roughly 25,000 to 27,000, so $p$ is not small relative to $\sqrt { n _ { 2 } }$ and we cannot appeal to this bound there. It is suficient rather than necessary, and conservative in that it ignores the orthogonality (25). The simulations in Subsection 4.2 use conditioning sets of comparable dimension at smaller $n _ { 2 }$ , hence at a less favorable ratio, and the arms with suficient $Z ^ { \omega _ { Z } }$ remain calibrated there, which suggests that the bound overstates the requirement at these dimensions. Misspecification of $m _ { Y }$ on $Z$ is the more serious concern, since it is not addressed by the orthogonality. The simulation does not separate it from mean insuficiency: in the arms retaining the generating embedder the outcome is a link applied to a linear score in that embedding, so the linear nuisances are close to correctly specified by construction, while in the arms omitting it the conditional mean given the supplied embedding is in general no longer linear, so misspecification and mean insuficiency act together and their contributions to the observed inflation cannot be attributed separately.

## References

M. K. Akram, S. Sturua, N. Havriushenko, Q. Herreros, M. G¨unther, M. Werk, and H. Xiao. jina-embeddings-v5-text: Task-targeted embedding distillation, 2026. URL https:// arxiv.org/abs/2602.15547.

J. Anthis, K. Lum, M. Ekstrand, A. Feller, A. D’Amour, and C. Tan. The impossibility of fair llms. arXiv preprint arXiv:2406.03198, 2024.

T. Berrett, Y. Wang, R. Barber, and R. Samworth. The conditional permutation test for independence while controlling for confounders. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1):175–197, 2020.

L. Cai, X. Guo, and W. Zhong. Test and measure for partial mean dependence based on machine learning methods. Journal of the American Statistical Association, 120(550): 833–845, 2025.

E. Candes, Y. Fan, L. Janson, and J. Lv. Panning for gold: ‘Model-x’knockofs for high dimensional controlled variable selection. Journal of the Royal Statistical Society Series B: Statistical Methodology, 80(3):551–577, 2018.

J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the association for computational linguistics: ACL 2024, pages 2318–2335, 2024.

B. Dai, X. Shen, and W. Pan. Significance tests of feature relevance for a black-box learner. IEEE Transactions on Neural Networks and Learning Systems, 2022.

P. Dawid. Conditional independence in statistical theory. Journal of the Royal Statistical Society: Series B (Methodological), 41(1):1–15, 1979.

P. Delobelle, E. K. Tokpo, T. Calders, and B. Berendt. Measuring fairness with biased rulers: A comparative study on bias metrics for pre-trained language models. In Proceedings of the 2022 Conference of the North American chapter of the association for computational linguistics, pages 1693–1706. Association for Computational Linguistics, 2022.

I. O. Gallegos, R. A. Rossi, J. Barrow, M. M. Tanjim, S. Kim, F. Dernoncourt, T. Yu, R. Zhang, and N. K. Ahmed. Bias and fairness in large language models: A survey. Computational Linguistics, pages 1–79, 2024.

M. Gao, X. Hu, X. Yin, J. Ruan, X. Pu, and X. Wan. Llm-based nlg evaluation: Current status and challenges. Computational Linguistics, pages 1–28, 2025.

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607. 02770.

M. Gentzkow, J. M. Shapiro, and M. Taddy. Measuring group diferences in highdimensional choices: method and application to congressional speech. Econometrica, 87(4):1307–1340, 2019.

A. Gjorgjevikj, B. K. Seljak, and T. Eftimov. On the robustness of multilingual text embedding rankings across learning tasks, languages, and benchmark datasets. arXiv preprint arXiv:2605.31142, 2026.

S. Goldfarb-Tarrant, R. Marchant, R. M. S´anchez, M. Pandya, and A. Lopez. Intrinsic bias metrics do not correlate with application bias. arXiv preprint arXiv:2012.15859, 2020.

L. Kook and A. R. Lundborg. Algorithm-agnostic significance testing in supervised learning with multimodal data. Briefings in Bioinformatics, 25(6):bbae475, 2024.

A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S. Kakade, P. Jain, and A. Farhadi. Matryoshka representation learning. In Advances in Neural Information Processing Systems, 2022.

C. Li and X. Fan. On nonparametric conditional independence tests for continuous variables. Wiley Interdisciplinary Reviews: Computational Statistics, 12(3):e1489, 2020.

Y. Li, M. Du, R. Song, X. Wang, and Y. Wang. A survey on fairness in large language models. arXiv preprint arXiv:2308.10149, 2023.

A. R. Lundborg, I. Kim, R. D. Shah, and R. J. Samworth. The projected covariance measure for assumption-lean variable significance testing. The Annals of Statistics, 52 (6):2851–2878, 2024.

G. d. S. P. Moreira, R. Osmulski, M. Xu, R. Ak, B. Schiferer, and E. Oldridge. Nv-retriever: Improving text embedding models with efective hard-negative mining. arXiv preprint arXiv:2407.15831, 2024.

T. P. Morris, I. R. White, and M. J. Crowther. Using simulation studies to evaluate statistical methods. Statistics in medicine, 38(11):2074–2102, 2019.

NVIDIA. Llama nemotron embedding 1b v2. Hugging Face model repository, 2025. URL https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2. Revision 113abe4acafa848e77ead9c0623205e511932348, accessed 2026-08-11.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https://qwen.ai/blog?id=qwen3.6-27b.

F. Richter, P. Koch, O. Franke, J. Kraus, F. Kuruc, A. Thiem, J. H¨ogerl, S. Heine, and K. Sch¨ops. Open Discourse, 2020. URL https://doi.org/10.7910/DVN/FIKIBO.

R. Shah and J. Peters. The hardness of conditional independence testing and the generalised covariance measure. The Annals of Statistics, 48(3):1514–1538, 2020.

M. Simnacher, X. Xu, H. Park, C. Lippert, and S. Greven. Deep nonparametric conditional independence tests for images. Journal of Machine Learning Research, 27(96):1–73, 2026.

E. Strobl, K. Zhang, and S. Visweswaran. Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. Journal of Causal Inference, 7(1), 2019.

T. Sun, J. He, X. Qiu, and X.-J. Huang. Bertscore is unfair: On social bias in language model-based metrics for text generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3726–3739, 2022.

B. D. Williamson, P. B. Gilbert, N. R. Simon, and M. Carone. A general framework for inference on algorithm-agnostic variable importance. Journal of the American Statistical Association, 118(543):1645–1658, 2023.

J. Wunderle, A. Ehrmanntraut, J. Pfister, F. Jannidis, and A. Hotho. New encoders for german trained from scratch: Comparing moderngbert with converted llm2vec models. arXiv preprint arXiv:2505.13136, 2025.

K. Zhang, J. Peters, D. Janzing, and B. Sch¨olkopf. Kernel-based conditional independence test and application in causal discovery. In Uncertainty in Artificial Intelligence (UAI 2011), pages 804–813. AUAI Press, 2011.

Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.