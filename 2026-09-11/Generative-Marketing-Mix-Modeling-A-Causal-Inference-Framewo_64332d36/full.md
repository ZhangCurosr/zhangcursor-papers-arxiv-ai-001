# Generative Marketing Mix Modeling: A Causal Inference Framework Linking GEO and GEM to Business Impact

Masahiro Kato<sup>∗</sup> <sup>1</sup>, Daiki Honma<sup>2</sup>, and Taka Kato<sup>2</sup>

<sup>1</sup>The University of Tokyo and Mizuho-DL Financial Technology Co., Ltd. <sup>2</sup>NP-hard

September 11, 2026

## Abstract

Generative artificial intelligence changes how firms reach customers, but standard marketing data do not record how often users see and notice a firm’s name in generated answers. We develop Generative Marketing Mix Modeling (GMMM) to estimate the causal efects of Generative Engine Optimization (GEO) and Generative Engine Marketing (GEM). For GEO, GMMM combines repeated generated answers with question counts, shares of use across generative systems, and notice probabilities. For GEM, it combines records of sponsored placements with notice probabilities. GMMM compares expected business responses under alternative treatment sequences and establishes suficient conditions for identifying the resulting efects. We investigate the empirical performance of the proposed method using simulated answers to product recommendation in English and Japanese.

Keywords: marketing mix modeling; generative engine optimization; generative engine marketing; causal inference; measurement error; channel attribution; Bayesian inference

## 1 Introduction

Marketing mix modeling (MMM) relates an aggregate response, such as sales or conversions, to media inputs observed across markets and periods. Two features of advertising motivate the transformations used in MMM. An advertisement can afect the response after the period in which it is shown, so a carryover function combines current and past inputs. The marginal change in response can also become smaller when the accumulated input is already high, so a saturation function maps the carried-over input into a bounded or slowly increasing regressor. MMM applies these transformations before estimating the coeficients of the media channels (Nerlove & Arrow, 1962; Clarke, 1976; Hanssens et al., 2001).

Generative artificial intelligence (AI) changes what must be measured before this model can be used. Through Generative Engine Optimization (GEO), a firm modifies source material that may alter answers about the firm, for example by adding a frequently asked questions section or revising product documentation. Through Generative Engine Marketing (GEM), a firm pays for sponsored placements in generated answers (Aggarwal et al., 2024; Feizi et al., 2026; Hu et al., 2026). Conventional impression logs do not record nonsponsored occurrences of a firm’s name in generated answers, and a platform record of sponsored placements does not show whether users noticed them.

The frequency with which collected answers contain a firm’s name is not yet a media input for a market and period. To obtain an expected count for a market and period, that frequency must be combined with the number of relevant questions, the share handled by each generative system, and the probability that a user notices the name. Referral sessions measure a diferent event because users may read an answer without following a link. For sponsored placements, spending must likewise be related to the number of placements shown and the probability of notice. GEO and GEM require this additional measurement step before they can be analyzed alongside media channels whose impressions are recorded directly.

Estimating a treatment efect adds a second requirement. Carryover makes the response in period t depend on inputs from earlier periods, so disabling GEO or GEM changes the entire sequence of media inputs over the afected periods. The relevant comparison must replace the full sequence under one treatment with the full sequence under another treatment before the response is evaluated. A regression coeficient by itself does not perform this comparison. Throughout this study, GEO denotes the specified source modification, and its treatment efect is the total efect of applying that modification. The response may change through the generated answers represented by the GEO input and through other consequences of the modification, such as conventional search trafic or conversion on a landing page.

## 1.1 Contributions

We develop GMMM to connect measurements of generated answers and sponsored placements to MMM. For GEO, GMMM models the probability that an answer contains a specified feature and combines that probability with market query counts and notice probabilities to obtain the expected number of noticed occurrences under each source state. This count can be positive without the source modification because GEO changes the occurrence probability rather than creating the entire input from zero. For GEM, GMMM relates spending to the number of sponsored placements and then accounts for notice. The resulting inputs receive the same carryover and Hill transformations as established media before they enter the response model.

Estimation need not be Bayesian. A likelihood or a penalized criterion can support plug-in, joint, and other regularized procedures. We use a Bayesian formulation to average over uncertainty in the occurrence and notice probabilities. The cut posterior leaves the distribution of the measurement parameters determined by the measurement data, whereas the joint posterior also allows the response data to update it. Comparing the two procedures shows how estimation of the media transformations and uncertainty about the constructed inputs afect the treatment efect of GEO.

The identification analysis gives conditions under which the treatment efect remains identifiable even when the coeficients on the GEO source state and the GEO input cannot be determined separately after the media transformations are fixed. It also characterizes when replacing the expected market count by a sequence of occurrence probabilities changes only the scale and when changes in market composition prevent that reduction.

The empirical collection contains 2,240 complete answers from GPT-5.6 Luna and GPT-4o. Each model answers the same set of 56 questions, comprising 28 English questions and 28 Japanese questions. Every call uses the same system instruction and requires web search. The target name is not included in the system instruction or in the questions. Glasp occurs in 33.8% of the GPT-5.6 Luna answers and 27.8% of the GPT-4o answers, although GPT-4o has the higher rate for the English questions. These observations estimate occurrence probabilities under the source state present during collection. One simulation design uses those estimates for the baseline occurrence probabilities, while the simulation itself generates the treatment efect. Across the controlled comparisons, estimating carryover and saturation explains most of the improvement in coeficient recovery over a plug-in method that fixes them, while updating the measurement parameters with response data has no uniform advantage for estimating the treatment efect.

## 1.2 Related Work

Classical MMM combines distributed lags with nonlinear response functions, and recent methods use regularization or Bayesian pooling across markets (Jin et al., 2017; Ng et al., 2024; Runge et al., 2025; Gong et al., 2024; Sun et al., 2017). These methods estimate a response model once its media inputs have been defined. Causal interpretation requires further evidence because predictive fit alone does not identify an advertising efect. Work on incrementality and geographic experiments studies the assignment mechanisms and randomized comparisons that can support such an interpretation (Chan & Perry, 2017; Lewis & Rao, 2015; Gordon et al., 2019; Vaver & Koehler, 2011; Chen & Au, 2022).

Research on GEO studies how changes in source material afect generated answers and their ranking, while work on GEM considers sponsored retrieval and placement (Bagga et al., 2026; Kim et al., 2026; Martinez, 2026; Hajiaghayi et al., 2024; Dubey et al., 2024; D¨utting et al., 2024; Xu et al., 2026). Studies of GEO treat citation of a source and use of its content in an answer as diferent outcomes (Zhang et al., 2026). Neither quantity alone gives the number of users who notice the target feature. GMMM addresses the next step by placing the occurrence probability on a market and time scale suitable for MMM.

The use of records on reach and frequency in MMM provides a precedent for replacing spending with quantities closer to exposure (Zhang et al., 2023). Randomized estimates have likewise been incorporated through informative priors or likelihood terms (Zhang et al., 2024). Meridian and PyMC-Marketing are Bayesian implementations of MMM, and Meridian supports inputs based on reach and frequency.<sup>1</sup> Our contribution concerns the measurements required when the media input itself must be inferred from generated answers or sponsored placements.

Cut and joint posteriors arise in modular Bayesian inference (Plummer, 2015; Jacob et al., 2017; Carmona & Nicholls, 2020). A cut posterior passes uncertainty from one module to another without allowing the later module to revise the first distribution. Its interpretation in GMMM depends on the assignment of parameters to modules: fixing the prior for the media transformations also prevents their estimation from response data, whereas cutting the update of the measurement parameters still permits those transformations to be estimated.

## 2 Setup

We use response for the business variable analyzed by MMM and answer for text returned by a generative system. A treatment is a specified setting of GEO or GEM. When several periods are involved, a treatment sequence lists those settings from period 1 through period $T$ and includes any earlier values needed to initialize carryover.

## 2.1 Response and Media Inputs in MMM

For every observational unit $i \in \{ 1 , \ldots , n \}$ and period $t \in \{ 1 , \ldots , T \}$ , let $Y _ { i t }$ denote the response and let $W _ { i t }$ contain baseline variables and controls observed before the period-t treatment. The unit may be a geographic market or another aggregation used consistently in the measurement and response models. Let $\mathcal { M } _ { 0 }$ be the set of established media channels. For every $m \in \mathcal { M } _ { 0 }$ , let $E _ { m i t } \geq 0$ denote the media input in period t and define its sequence through period t by $\pmb { E _ { m i , 1 : t } } = ( E _ { m i 1 } , \ldots , E _ { m i t } )$

A standard MMM transforms each sequence of media inputs before including it in the response model. We write

$$
H _ { m i t } = h _ { m } \left( A _ { m } ( { \cal E } _ { m i , 1 : t } ; \alpha _ { m } ) ; \theta _ { m } \right) ,\tag{1}
$$

where $A _ { m }$ combines current and earlier inputs according to carryover parameters $\alpha _ { m }$ , and $h _ { m }$ represents saturation with parameters $\theta _ { m }$ . Let $b ( W ; \gamma )$ be the baseline response with coeficient vector $\gamma$ . If $\mathcal { T } _ { 0 }$ is a specified set of distinct pairs of media channels, the conditional mean satisfies

$$
g _ { Y } \left( \mu _ { i t } ^ { \mathrm { M M M } } \right) = b ( W _ { i t } ; \gamma ) + \sum _ { m \in \mathcal { M } _ { 0 } } \beta _ { m } H _ { m i t } + \sum _ { ( m , m ^ { \prime } ) \in \mathcal { T } _ { 0 } } \beta _ { m m ^ { \prime } } H _ { m i t } H _ { m ^ { \prime } i t } ,\tag{2}
$$

where $\mu _ { i t } ^ { \mathrm { M M M } } = \mathbb { E } [ Y _ { i t } \mid \mathcal { H } _ { i t } ]$ . The information set $\mathcal { H } _ { i t }$ contains the controls and the sequences of media inputs used to model $Y _ { i t }$ . We specify $Y _ { i t } \mid \mathcal { H } _ { i t } \sim \mathcal { F } ( \mu _ { i t } ^ { \mathrm { M M M } } , \psi )$ , where $\psi$ contains the remaining parameters of the response distribution. The identity link gives the usual model for a continuous response with additive noise of mean zero, while other links accommodate counts or rates.

## 2.2 Channels Through Generative AI

GMMM enlarges the media set to $\mathcal { M } = \mathcal { M } _ { 0 } \cup \{ G , P \}$ , where G indexes the input constructed from generated answers and P indexes the input constructed from sponsored placements. For each period, the first input is the expected number of generated answers in which the specified property occurs and is noticed under the current source state. The second is the expected number of sponsored placements that users notice. The first count need not be zero when the source modification is absent; GEO changes its value by changing generated answers. Inputs for established media are observed directly, while the two additional inputs are constructed from generated answers, platform records, market counts, and observations of user attention.

Let $Z _ { i t } \in \{ 0 , 1 \}$ denote the GEO source state, where $Z _ { i t } = 1$ means that the specified source modification is present and $Z _ { i t } = 0$ means that it is absent. The response specification studied here is

$$
g _ { Y } \left( \mu _ { i t } \right) = b ( W _ { i t } ; \gamma ) + \sum _ { m \in \mathcal { M } _ { 0 } } \beta _ { m } H _ { m i t } + \beta _ { D } Z _ { i t } + \beta _ { G } H _ { i t } ^ { G } + \beta _ { P } H _ { i t } ^ { P } + \beta _ { G P } H _ { i t } ^ { G } H _ { i t } ^ { P } .\tag{3}
$$

The term $\beta _ { D } Z _ { i t }$ permits the source modification to afect the response through variables that are not represented by the constructed GEO input, such as conventional search trafic or conversion on a landing page. The coeficient $\beta _ { G P }$ permits the efect associated with one generative channel to depend on the other. Either term may be omitted when the corresponding mechanism is excluded from the model.

## 2.3 Data Sources

Let $\mathcal { D } _ { M }$ contain the observations used to construct the GEO and GEM inputs. For GEO, these observations include repeated generated answers, counts of relevant questions, shares of use across generative systems, and observed notice indicators. For GEM, they include spending, the number of sponsored placements shown, predictors of that number, and observed notice indicators. Let $\mathcal { D } _ { Y }$ contain $Y _ { i t }$ together with the controls and media inputs in the response model. When a randomized experiment estimates the same treatment efect for a comparable population and evaluation period, its estimate and standard error form $\mathcal { D } _ { E }$

The collection described in Section 5.1 contains 20 answers from each of two GPT models for each of 56 questions, comprising 28 English questions and 28 Japanese questions. The public referral series analyzed in Appendix F comes from an earlier period and lacks the corresponding question counts, generated answers, and notice observations. These missing observations prevent the two sources from being combined in one GMMM analysis.

The simulations use units of the form $i = ( g , j )$ , where $g \in \{ 1 , \ldots , 5 \}$ indexes geographic markets and $j ~ \in ~ \{ 1 , \dots , 6 \}$ indexes product clusters. Each cluster contains pages and questions that respond to the same GEO treatment. A substantive application must use an observational unit that is common to the construction of the media inputs, the response model, and the treatment efect.

## 2.4 Treatment Efect

Because carryover links the period-t response to media inputs observed before t, we define the treatment efect from complete treatment sequences over a specified evaluation window.

Let $a _ { G } \in \{ 0 , 1 \}$ select one of two GEO sequences: $a _ { G } = 1$ selects the specified sequence with GEO, and $a _ { G } = 0$ selects the sequence with the source modification removed. Let $a _ { P } \in \{ 0 , 1 \}$ similarly select the specified GEM spending sequence or the sequence with GEM removed. The pair $( a _ { G } , a _ { P } )$ selects one complete GEO sequence and one complete GEM sequence across all periods. The comparison uses the same history before period 1 unless an intervention begins earlier; in that case, the treatment and input values before period 1 follow the selected sequence. The simulations set all media inputs before period 1 to zero. For an evaluation set $\mathcal { T } _ { E } \subseteq \{ 1 , \dots , T \}$ , define

$$
V ( a _ { G } , a _ { P } ) = \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } \mathbb { E } \left[ Y _ { i t } ( a _ { G } , a _ { P } ) \right] ,\tag{4}
$$

where $Y _ { i t } ( a _ { G } , a _ { P } )$ is the potential response under the selected treatment sequences and the specified evolution of all other variables. The primary estimand is

$$
\Delta _ { G } = V ( 1 , 1 ) - V ( 0 , 1 ) .\tag{5}
$$

We refer to $\Delta _ { G }$ as the treatment efect of GEO. It is the total efect of the specified source modification, including the change transmitted through $H ^ { G }$ and any additional change represented by $\beta _ { D } Z _ { i t }$ . The contrast retains the specified GEM sequence. The evaluation window may extend beyond the periods in which GEO is active so that the efect includes responses associated with carried-over inputs. The analogous treatment efect of GEM is $\Delta _ { P } = V ( 1 , 1 ) - V ( 1 , 0 )$ , which compares the specified GEM sequence with the sequence in which GEM is disabled while retaining the GEO sequence.

## 3 Generative Marketing Mix Modeling

The data record the source state, spending, and established media inputs, but they omit the expected counts associated with GEO and GEM. GMMM constructs these two counts before applying the standard media transformations and estimating the response model.

## 3.1 The GEO Input

Let $q \in \{ 1 , \ldots , Q \}$ index clusters of questions and let $p \in \mathcal P$ index generative systems. A system is defined by the model version and the settings used to obtain an answer. The set $\mathcal { Q } ( i )$ contains the question clusters associated with unit i. For every $i , q ,$ , and t, let $N _ { i q t } \geq 0$ be the number of times that a question in cluster q is submitted in unit i during period t. For every $i , p ,$ and t, let $\omega _ { i p t } \in [ 0 , 1 ]$ be the share of those questions handled by system $p ,$ with $\sum _ { p \in \mathcal { P } } \omega _ { i p t } = 1$

Before collecting answers, the analyst specifies the property to be recorded. For repetition $r \in \{ 1 , \ldots , R _ { q p t } \}$ and source state $z \in \{ 0 , 1 \}$ , let $X _ { q p t r } ( z ) = 1$ when the stored answer has that property and let it equal zero otherwise. In the empirical collection, the property is the occurrence of a spelling of the target name in a fixed dictionary. Let $\pi _ { q p t } ( z ) = \mathbb { P } \left[ ( \right] X _ { q p t r } ( z ) =$ 1) be its occurrence probability. Conditional on $X _ { q p t r } ( z ) = 1$ , let $\lambda _ { q p t } \in [ 0 , 1 ]$ be the probability that the user notices the recorded property. The input associated with GEO is

$$
E _ { i t } ^ { G } ( z ) = \sum _ { q \in \mathcal { Q } ( i ) } N _ { i q t } \sum _ { p \in \mathcal { P } } \omega _ { i p t } \pi _ { q p t } ( z ) \lambda _ { q p t } .\tag{6}
$$

The quantity $E _ { i t } ^ { G } ( z )$ is the expected number of generated answers in unit i and period t for which the specified property occurs and is noticed under source state z. It may be positive when $z = 0 \mathrm { : }$ ; the change in this input caused by the source modification is $E _ { i t } ^ { G } ( 1 ) - E _ { i t } ^ { G } ( 0 )$ . A user who encounters the property more than once contributes more than once to this count. The comparison between $z = 1$ and $z = 0$ holds $N _ { i q t }$ and $\omega _ { i p t }$ fixed. We assume that the source modification does not change notice conditional on occurrence, which is why $\lambda _ { q p t }$ has no argument z.

Equation (6) pools occurrence and notice probabilities across markets after conditioning on the question cluster, generative system, period, and source state. It also uses a system share that is common across question clusters within a market and period. When the data distinguish these cells, the same construction can use $\pi _ { i q p t } ( z ) , \lambda _ { i q p t }$ , and $\omega _ { i q p t }$ without changing the media transformations or the treatment contrasts.

We estimate the occurrence probabilities with the hierarchical model

$$
X _ { q p t r } \mid \pi _ { q p t } \sim \mathrm { B e r n o u l l i } ( \pi _ { q p t } ) ,\tag{7}
$$

$$
\mathrm { l o g i t } ( \pi _ { q p t } ) = A _ { q p t } ^ { \mathsf { T } } \alpha + u _ { q } , \qquad { \pmb u } = B _ { Q } \sigma _ { u } { \widetilde { \pmb u } } , \qquad { \widetilde { \pmb u } } \sim { \mathcal { N } } ( 0 , I _ { Q - 1 } ) .\tag{8}
$$

The columns of $B _ { Q } \in \mathbb { R } ^ { Q \times ( Q - 1 ) }$ form an orthonormal basis for vectors whose coordinates sum to zero, so $\begin{array} { r } { \sum _ { q = 1 } ^ { Q } u _ { q } = 0 } \end{array}$ . The predictor vector $A _ { q p t }$ may contain indicators for the generative system, the source state, and other observed determinants of occurrence. Its coeficient vector is $\alpha$ . The efect for question cluster $q$ is represented in noncentered form by standard normal coordinates $\widetilde { \mathbf { \ b { u } } }$ and scale $\sigma _ { u } > 0$ . Replacing the observed sequence of source states by another treatment sequence yields probabilities under that sequence only when the data contain the required variation in source state. Answers collected under one source state identify only the occurrence probabilities for that state; the change caused by the source modification requires observations under both states.

Suppose that a user study presents generated answers and records $C _ { q p t }$ cases in which participants notice the property among $n _ { q p t }$ answers in which it occurs. We use

$$
\begin{array} { r } { C _ { q p t } \mid \lambda _ { q p t } \sim \mathrm { B i n o m i a l } ( n _ { q p t } , \lambda _ { q p t } ) , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { \mathrm { l o g i t } ( \lambda _ { q p t } ) = ( D _ { q p t } ^ { \lambda } ) ^ { \top } \xi + v _ { q } , \qquad v = B _ { Q } \sigma _ { v } \widetilde { v } , \qquad \widetilde { v } \sim { \mathcal N } ( 0 , I _ { Q - 1 } ) . } \end{array}\tag{10}
$$

Here, $D _ { q p t } ^ { \lambda }$ is a vector of observed predictors with coeficient vector $\xi .$ The efect for question cluster $q ,$ denoted by $v _ { q } .$ , uses the same zero-sum basis and its own scale $\sigma _ { v } > 0$ . Together, the two models estimate the probabilities needed in (6).

## 3.2 The GEM Input

A sponsored placement contributes to the GEM input only if the platform shows it and the user notices it. Let $S _ { i t } ^ { P } \geq 0$ denote the spending level selected by the firm for the GEM intervention in unit i and period t, before the platform determines delivery. Let $L _ { i t } ^ { P }$ denote the number of sponsored placements shown. Conditional on a placement being shown, let $\rho _ { i t } ^ { P } \in [ 0 , 1 ]$ be the probability that the user notices it. For positive spending, we model the number shown by

$$
L _ { i t } ^ { P } \mid S _ { i t } ^ { P } > 0 \sim \mathrm { P o i s s o n } ( \mu _ { i t } ^ { P } ) ,\tag{11}
$$

$$
\log \mu _ { i t } ^ { P } = \phi _ { 0 } + \phi _ { S } \log S _ { i t } ^ { P } + \phi _ { D } D _ { i t } + \phi _ { R } R _ { i t } , \qquad \phi _ { S } > 0 ,\tag{12}
$$

where $D _ { i t }$ is a demand predictor observed before the period-t treatment and $R _ { i t }$ is a promotion indicator. We set $\mu _ { i t } ^ { P } = 0$ when $S _ { i t } ^ { P } = 0$ . The input associated with GEM is

$$
E _ { i t } ^ { P } = \mu _ { i t } ^ { P } \rho _ { i t } ^ { P } .\tag{13}
$$

The restriction $\phi _ { S } > 0$ makes the expected number of sponsored placements increase with positive spending. A user study in which participants report whether they noticed each sponsored placement can estimate $\rho _ { i t } ^ { P }$ with a binomial model. The analysis below uses one notice probability for all sponsored placements, while (13) permits variation across units and periods when the data support it. When $L _ { i t } ^ { P }$ is observed and the analysis conditions on realized delivery, $L _ { i t } ^ { P } \rho _ { i t } ^ { P }$ can be used directly. The Poisson model is needed for missing delivery counts and for spending sequences that were not observed.

## 3.3 Carryover and Saturation

The two expected counts are transformed in the same way as inputs for established media. We use normalized geometric carryover,

$$
A _ { m i t } = \frac { \sum _ { \ell = 0 } ^ { L _ { m } } \alpha _ { m } ^ { \ell } E _ { m i , t - \ell } / s _ { m } } { \sum _ { \ell = 0 } ^ { L _ { m } } \alpha _ { m } ^ { \ell } } , \qquad 0 \le \alpha _ { m } < 1 ,\tag{14}
$$

where $L _ { m } \in \{ 0 , 1 , . . . \}$ is the maximum lag and $s _ { m } > 0$ is a fixed scale used in the analysis. Values before period 1 come from the earlier history assigned to the treatment sequence; the simulations set these values to zero. The transformed input is

$$
H _ { m i t } = \frac { A _ { m i t } ^ { \kappa _ { m } } } { A _ { m i t } ^ { \kappa _ { m } } + \theta _ { m } ^ { \kappa _ { m } } } , \qquad \theta _ { m } > 0 , \qquad \kappa _ { m } > 0 .\tag{15}
$$

The Hill function equals $1 / 2$ when $A _ { m i t } = \theta _ { m }$ , and $\kappa _ { m }$ determines how sharply it changes near that point. Carryover is applied before the Hill transformation (Jin et al., 2017). Substituting $H _ { i t } ^ { G }$ and $H _ { i t } ^ { P }$ into (3) completes the response specification.

## 3.4 Estimation from Measurement and Response Data

Let $\eta _ { M }$ collect the parameters used in (6) and (13), let $\eta _ { T }$ collect the carryover and Hill parameters, and let $\vartheta$ collect the response parameters. We write $\mathcal { I } _ { M } ( \eta _ { M } ; \mathcal { D } _ { M } )$ for a loss based on the data used to construct the two inputs and $\mathcal { I } _ { Y } ( \vartheta , \eta _ { M } , \eta _ { T } ; \mathcal { D } _ { Y } )$ for a loss based on the response data. If $\mathcal { D } _ { E }$ contains a randomized estimate of the same treatment efect, its contribution is $\mathcal { T } _ { E } ( \vartheta , \eta _ { M } , \eta _ { T } ; { \mathcal { D } } _ { E } )$ ; otherwise, we set $\mathcal { I } _ { E } = 0$ . A general estimator minimizes

$$
\mathcal { I } ( \eta _ { M } , \eta _ { T } , \vartheta ) = \mathcal { I } _ { M } + \mathcal { I } _ { Y } + \mathcal { I } _ { E } + \mathcal { P } _ { M } ( \eta _ { M } ) + \mathcal { P } _ { T } ( \eta _ { T } ) + \mathcal { P } _ { Y } ( \vartheta ) ,\tag{16}
$$

where the penalty terms may be zero. With negative log likelihoods and no penalties, this criterion gives maximum likelihood estimation. Nonzero penalties permit regularization.

The criterion covers several ways of using the data. A plug-in method estimates $\eta _ { M }$ from $\mathcal { D } _ { M }$ , substitutes that estimate into the response model, and either fixes $\eta _ { T }$ or estimates it from $\mathcal { D } _ { Y }$ . A joint likelihood estimates $\eta _ { M } , \eta _ { T }$ , and $\vartheta$ together. Resampling can be used with either procedure to assess uncertainty. A Bayesian version makes both the treatment of uncertainty and the direction of updating explicit.

## 3.5 Bayesian Computation

Minimizing the criterion with negative log likelihoods gives point estimates. In the Bayesian analysis, the cut posterior leaves the distribution of the measurement parameters determined by $\mathcal { D } _ { M }$ , whereas the joint posterior allows $\mathcal { D } _ { Y }$ to update it. Both procedures average over uncertainty in the constructed inputs. The hierarchical occurrence and notice models are fitted in noncentered coordinates. Their posterior modes and analytic Hessians define a Laplace approximation $q _ { L } ( \eta _ { M } \mid \mathcal { D } _ { M } )$ to the posterior based only on $\mathcal { D } _ { M }$ . Sampling the transformation parameters from their prior gives

$$
q ( \eta \mid \mathcal { D } _ { M } ) = q _ { L } ( \eta _ { M } \mid \mathcal { D } _ { M } ) p ( \eta _ { T } ) , \qquad \eta = ( \eta _ { M } , \eta _ { T } ) .\tag{17}
$$

Let ${ \boldsymbol { \beta } } = ( \beta _ { D } , \beta _ { G } , \beta _ { P } , \beta _ { G P } ) ^ { \mathsf { T } }$ . After replacing the exact posterior for $\eta _ { M }$ by the Laplace approximation, the joint posterior is

$$
\widetilde { p } ( \eta , \beta , \sigma _ { Y } ^ { 2 } \mid \mathcal { D } _ { M } , \mathcal { D } _ { Y } ) \propto p ( \mathcal { D } _ { Y } \mid \eta , \beta , \sigma _ { Y } ^ { 2 } ) p ( \beta , \sigma _ { Y } ^ { 2 } ) q ( \eta \mid \mathcal { D } _ { M } ) .\tag{18}
$$

For each $k \in \{ 1 , \ldots , K \}$ , we sample $\eta _ { k } \sim q ( \eta \mid \mathcal { D } _ { M } )$ and construct the corresponding sequences of media inputs. Conditional on $\eta _ { k }$ , the response parameters are integrated under the normal inverse-gamma model with the sign restrictions in Appendix G.2. Let $m _ { k }$ be the numerical approximation to $p ( \mathcal { D } _ { Y } \mid \eta _ { k } )$ obtained with a Gaussian approximation to the posterior probability of the sign restrictions. The joint calculation assigns weight

$$
w _ { k } = \frac { m _ { k } } { \sum _ { h = 1 } ^ { K } m _ { h } } .\tag{19}
$$

After selecting $\eta _ { k }$ according to these weights, we sample the response parameters from their conditional posterior and evaluate $\Delta _ { G }$

The two-stage procedure uses equal weights for the samples from (17), leaving both the Laplace approximation for $\eta _ { M }$ and the prior for $\eta _ { T }$ unchanged by $\mathcal { D } _ { Y }$ . A cut posterior keeps only the first of these distributions fixed:

$$
p _ { \mathrm { c u t } } ( \eta _ { M } , \eta _ { T } , \beta , \sigma _ { Y } ^ { 2 } \mid \mathcal { D } _ { M } , \mathcal { D } _ { Y } ) = q _ { L } ( \eta _ { M } \mid \mathcal { D } _ { M } ) p ( \eta _ { T } , \beta , \sigma _ { Y } ^ { 2 } \mid \mathcal { D } _ { Y } , \eta _ { M } ) .\tag{20}
$$

Because the conditional posterior on the right is normalized separately for every $\eta _ { M }$ , integrating over $( \eta _ { T } , \beta , \sigma _ { Y } ^ { 2 } )$ leaves $q _ { L } ( \eta _ { M } \mid \mathcal { D } _ { M } )$ unchanged (Plummer, 2015; Jacob et al., 2017).

For computation, we use M samples of the measurement parameters and a common set of J samples from the transformation prior. Let $m _ { m j }$ approximate $p ( \mathcal { D } _ { Y } \mid \eta _ { M , m } , \eta _ { T , j } )$ . On this common set, the cut and joint weights are

$$
w _ { m j } ^ { \mathrm { c u t } } = \frac { 1 } { M } \frac { m _ { m j } } { \sum _ { h = 1 } ^ { J } m _ { m h } } , \qquad w _ { m j } ^ { \mathrm { j o i n t } } = \frac { m _ { m j } } { \sum _ { r = 1 } ^ { M } \sum _ { h = 1 } ^ { J } m _ { r h } } .\tag{21}
$$

The plug-in method that estimates the transformations uses the same J samples at the point estimate of $\eta _ { M }$ , while the two-stage procedure assigns weight $1 / ( M J )$ to every pair. For the independent samples in (19), we report

$$
\mathrm { E S S } = \left( \sum _ { k = 1 } ^ { K } w _ { k } ^ { 2 } \right) ^ { - 1 }\tag{22}
$$

as a numerical diagnostic. A small efective sample size means that a few samples carry most of the weight. For the common set of samples, Appendix C.3 reports the efective sample size within each measurement sample and the efective sample size of the joint weights across measurement samples.

Figure 1 summarizes the order of the calculations.

![](images/d2283e27efa797d67bf8b9692c0d8ef4bc55282c3971287d1e0c967616eab6d3.jpg)  
GMMM constructs generative exposure before applying the response and attribution stages of MMM.  
Figure 1: GMMM constructs the expected counts associated with GEO and GEM before applying carryover, the Hill transformation, and the response model. The plug-in, cut, and joint procedures difer in whether they average over uncertainty in the measurement parameters and whether the response data update those parameters.

## 3.6 Randomized Estimate of the GEO Treatment Efect

A randomized experiment can inform GMMM when it estimates the same treatment efect for a population, response definition, and evaluation window that can be related to the GMMM target. Suppose that the experiment reports $\widehat { \tau } _ { E }$ with standard error $s _ { E } ,$ and let $\tau _ { E } ( \vartheta , \eta )$ be the corresponding efect implied by the GMMM parameters. We use

$$
\begin{array} { r } { \widehat { \tau } _ { E } \mid \vartheta , \eta \sim { \mathcal N } \left( \tau _ { E } ( \vartheta , \eta ) , s _ { E } ^ { 2 } + \sigma _ { \mathrm { t r } } ^ { 2 } \right) , } \end{array}\tag{23}
$$

where $\sigma _ { \mathrm { t r } } \geq 0$ is fixed before estimation and represents residual diferences between the experimental population and the target population after the measured design variables have been aligned. This likelihood contributes $\mathcal { I } _ { E }$ to (16). In the Bayesian analysis, it changes the weights and the conditional posterior for the response coeficients. For every parameter sample, the experimental efect is computed from the complete treatment and control sequences before the media transformations are applied.

The aligned simulation uses the average treatment efect of GEO, including the term associated with the source state. A second simulation adds a nonzero diference between the experimental and target populations to the same efect and thereby examines misspecification of the transport model. If an additive attribution across interacting channels is also required, Appendix D gives a Shapley allocation as an additional summary.

## 4 Identification of the Treatment Efect of GEO

Recovering the treatment efect of GEO requires both statistical variation in the response model and observations that support the expected response under each treatment sequence. Statistical variation determines whether the response coeficients or the linear combination that defines the efect can be identified, while the observed market inputs and treatment assignments determine whether the comparison can be evaluated.

## 4.1 Identification Within the Response Model

If the two constructed inputs were observed directly, GMMM would difer from a standard MMM only through the additional term for the GEO source state. The following reduction states this relation before the rank condition for the response coeficients.

Proposition 4.1 (Reduction to standard MMM). Suppose that $\pmb { { E } } _ { i , 1 : T } ^ { G }$ and $\mathbf { \mathit { E } } _ { i , 1 : T } ^ { P }$ are observed for every $i \in \{ 1 , \ldots , n \}$ . If $\beta _ { D } ~ = ~ 0$ , then (3) is an instance of (2) with channel set ${ \mathcal { M } } _ { 0 } \cup \{ G , P \}$ and interaction set $\{ ( G , P ) \}$ $I f \beta _ { G P } = 0$ also holds, the response is additive on this enlarged channel set. Setting all four coeficients associated with GEO and GEM to zero gives the additive model for established media with $\mathcal { T } _ { 0 } = \emptyset$

Proof. The observed sequences determine $H ^ { G }$ and $H ^ { P }$ through (1). Substitution into (3), together with the stated restrictions on the coeficients, gives the corresponding cases of (2). □

GMMM adds the construction of $E ^ { G }$ and $E ^ { P }$ and the definition of their values under each treatment sequence. Once these quantities are fixed, identification of the response coeficients is a rank problem.

Proposition 4.2 (Identification of response coeficients and linear efects). Fix the sequences of media inputs and all parameters of the media transformations. Suppose that the response model has an identity link and $b ( W ; \gamma ) = W \gamma$ . Let $H _ { 0 }$ be the matrix whose columns are the transformed inputs for established media and let $W _ { \ast } = ( W , H _ { 0 } )$ . Let $X _ { \eta }$ contain the variable for source state, $H ^ { G } , H ^ { P }$ , and $H ^ { G } H ^ { P }$ $I f M _ { W _ { \bf 3 } }$ is the orthogonal projection onto the complement of the column space of $W _ { * }$ , define

$$
D _ { \eta } = M _ { W _ { \ast } } X _ { \eta } ,
$$

$$
G _ { \eta } = X _ { \eta } ^ { \mathsf { T } } M _ { W _ { \ast } } X _ { \eta } .\tag{24}
$$

Conditional on the fixed sequences and transformations, the four response coeficients are identified from the conditional mean if and only if $G _ { \eta }$ is nonsingular. Under a Gaussian conditional response with nonsingular variance, the same condition is equivalent to identification by the likelihood. For a fixed vector d $\in \mathbb { R } ^ { 4 }$ , the linear efect $d ^ { \mathsf { T } } \beta$ is identified if and only if $d ^ { \mathsf { T } } v = 0$ for every $\boldsymbol { v } \in \mathbb { R } ^ { 4 }$ such that $D _ { \eta } v = 0$

Proof. Two coeficient vectors $\beta$ and $\beta ^ { \prime }$ give the same conditional mean after the nuisance coeficients are adjusted if and only if $X _ { \eta } ( \beta - \beta ^ { \prime } )$ belongs to the column space of $W _ { * }$ Equivalently, $D _ { \eta } ( \beta - \beta ^ { \prime } ) = 0$ . The four coeficients are identified exactly when $D _ { \eta }$ has full column rank, which is equivalent to nonsingularity of $G _ { \eta }$ . The linear efect is constant over observationally equivalent coeficient vectors exactly when it vanishes on the null space of $D _ { \eta }$ . The necessity statement remains valid under the sign restrictions used in the simulations because their parameter space has nonempty interior. □

The simulations omit established media, so $H _ { 0 }$ has no columns and residualization with respect to W is suficient. In the general model, $H _ { 0 }$ must be included. For example, if a transformed input for an established channel equals $H ^ { G }$ , its coeficient cannot be separated from $\beta _ { G }$ even when the four columns of $X _ { \eta }$ are linearly independent after removing W alone.

A linear efect may remain identified when its component coeficients are not. This distinction is especially relevant when the variable for source state and $H ^ { G }$ are nearly proportional after the other regressors have been removed. Proposition 4.2 states the exact null-space condition; finite-sample stability still depends on the degree of collinearity.

The proposition treats the transformations as fixed. Joint identification of $( \beta _ { G } , \alpha _ { G } , \theta _ { G } , \kappa _ { G } )$ additionally requires the observed input sequences to provide independent information about response amplitude, carryover, and the Hill curve. With all other parameters fixed, full column rank of the Jacobian of the residualized mean with respect to these four parameters is suficient for local identification at an interior point of a continuously diferentiable model. If other parameters are unknown, full column rank of the Jacobian with respect to the complete free parameter vector after removal of the fixed linear controls is suficient. A step treatment whose untransformed GEO input has one level before treatment and another after treatment cannot separate response amplitude from the Hill parameters using steady-state observations alone. Identification then depends on transition dynamics and on variation within source states, including variation in question counts, system shares, or occurrence probabilities. A prior can stabilize estimation without identifying a likelihood that lacks this variation. Related ambiguities arise when nonlinear response and coeficients that change over time produce similar observed sequences but diferent allocation decisions (Dew et al., 2024).

## 4.2 Occurrence Probabilities and Market Counts

The market input in (6) weights an occurrence probability by the number and composition of questions, the shares of use across systems, and the probability of notice. In a special case, this diference amounts only to a change of scale. Let $A ( \pi )$ denote linear carryover applied to a sequence of exact occurrence probabilities, with the same initialization as (14).

Proposition 4.3 (Constant rescaling of the GEO input). Suppose that $E _ { i t } ^ { G } ( z ) = c _ { 0 } \pi _ { i t } ( z )$ for the same constant $c _ { 0 } > 0$ in every market, period, and evaluated source state, including values used to initialize carryover. For the Hill function $h ( a ; \theta , \kappa ) = a ^ { \kappa } / ( a ^ { \kappa } + \theta ^ { \kappa } )$ , it holds that

$$
h ( A ( { \cal E } ^ { G } ) ; \theta , \kappa ) = h ( c _ { 0 } A ( \pi ) ; \theta , \kappa ) = h ( A ( \pi ) ; \theta / c _ { 0 } , \kappa ) .\tag{25}
$$

Rescaling θ preserves the transformed sequence and the treatment efect computed from it. A single common value of θ does not generally absorb scale factors that vary across markets or periods.

Proof. Linearity gives $A ( c _ { 0 } \pi ) ~ = ~ c _ { 0 } A ( \pi )$ Dividing the numerator and denominator of $h ( c _ { 0 } a ; \theta , \kappa )$ by $c _ { 0 } ^ { \kappa }$ proves (25). For the final statement, consider two markets with the same positive occurrence-probability sequence and diferent constants $c _ { 1 }$ and $c _ { 2 }$ . The transformed market counts difer because h is strictly increasing, whereas a common transformation of the identical probability sequences is the same. One common value of $\theta$ cannot represent both. □

The same equivalence holds within each market if its scale factor $c _ { i }$ is constant over time and the model permits a market-specific parameter $\theta _ { i } .$ , rescaled to $\theta _ { i } / c _ { i }$ . A Bayesian analysis must transform the prior and its support consistently. Under the common-scale condition in Proposition 4.3, a common multiplicative level in question volume or notice probability is absorbed by the Hill midpoint. Question volume, system shares, and notice probabilities afect the treatment efect when their relative values vary across markets, periods, systems, question clusters, or source states. When that variation changes the proportionality between occurrence probabilities and expected counts, one occurrence-probability sequence cannot represent every market input. A finite collection of answers adds a separate source of uncertainty about the probabilities. Appendix C examines both issues in a static linear model and under a nonlinear transformation.

## 4.3 Identification across Treatment Sequences

The rank condition concerns the response model after the inputs have been fixed. Identification of $\Delta _ { G }$ also requires the observed data to determine the expected response under each complete treatment sequence. For every $i \in \{ 1 , \ldots , n \}$ and $t \in \{ 1 , \ldots , T \}$ , let $\mathcal { H } _ { i t } ^ { - }$ contain the variables observed immediately before the period-t GEO and GEM treatments. It includes earlier responses, prior media inputs, earlier treatments, and any aggregate variables used to represent interference. Let $\mathcal { H } _ { i t }$ add the period-t treatments and the values of $E _ { i t } ^ { G }$ and $E _ { i t } ^ { P }$ implied by them. For a pair $( a _ { G } , a _ { P } )$ of complete treatment sequences, let $F _ { i t } ^ { a _ { G } , a _ { P } }$ be the distribution of $\mathcal { H } _ { i t }$ obtained by recursively replacing the observed treatment assignment with those sequences. Finally, define $m _ { i t } ( h ) = \mathbb { E } [ Y _ { i t } \mid \mathcal { H } _ { i t } = h ]$ The following assumptions place the standard longitudinal $\mathrm { g } .$ -formula on these GMMM treatment sequences.

Assumption 4.1 (Consistency). If the observed GEO and GEM treatments through period t equal the values specified by $( a _ { G } , a _ { P } )$ , then the observed variables through period t equal their potential values under those treatment sequences.

Assumption 4.2 (Sequential exchangeability). Conditional on $\mathcal { H } _ { i t } ^ { - }$ , the period-t GEO and GEM treatments are independent of future potential information sets and potential responses under the treatment sequences evaluated in (4).

Assumption 4.3 (Positivity). For every value of $\mathcal { H } _ { i t } ^ { - }$ in the target support, each discrete treatment specified by the evaluated sequences has positive conditional probability. A positive continuous GEM spending level lies in the interior of its conditional support and has positive density in a neighborhood of that level. A sequence that sets spending to zero requires positive conditional mass at zero when the absence of a campaign is represented by a separate point mass.

Assumption 4.4 (Observed-data laws in the target population). The measurement and response data identify $m _ { i t }$ and the conditional laws used to construct $F _ { i t } ^ { a _ { G } , a _ { P } }$ on the support of the evaluated treatment sequences. When the response model uses the expected counts in (6) and (13), the measurement parameters and market inputs identify those counts for the target population. Identification of $m _ { i t }$ on the same support must also be established. If $\mathcal { H } _ { i t }$ instead contains unobserved realized impressions, their joint law with the observed variables and $Y _ { i t }$ must also be identified; a marginal distribution of impressions is insuficient.

Assumption 4.5 (Interference through specified aggregates). Spillovers within an observational unit are included in that unit’s treatments and response. Treatments assigned to other units may afect its potential response or generative inputs only through aggregate variables included in $\mathcal { H } _ { i t } ^ { - }$ . The distribution $F _ { i t } ^ { a _ { G } , a _ { P } }$ specifies how those aggregates evolve under the evaluated treatment sequences.

Under these conditions, the longitudinal g-formula applies to the complete GEO and GEM treatment sequences.

Theorem 4.4 (Longitudinal g-formula for GMMM). Under the preceding assumptions, the expected response in (4) is identified by

$$
V ( a _ { G } , a _ { P } ) = \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } \int m _ { i t } ( h ) d F _ { i t } ^ { a _ { G } , a _ { P } } ( h ) .\tag{26}
$$

The diference between this expression for $( a _ { G } , a _ { P } ) = ( 1 , 1 )$ and $( a _ { G } , a _ { P } ) = ( 0 , 1 )$ identifies $\Delta _ { G }$

Proof. Consistency links observed variables to their potential values under the realized treatments. Sequential exchangeability permits the observed assignment mechanism to be replaced, conditional on $\mathcal { H } _ { i t } ^ { - }$ , by the evaluated treatment sequences. Positivity ensures that the required conditional laws are defined on their support. The measurement assumption identifies the conditional laws and $m _ { i t }$ , including their dependence on the two constructed inputs. The interference assumption makes the potential response for each unit well defined at the chosen aggregation. Iterated expectation yields the longitudinal g-formula in (26) (Robins, 1987). □

Assumption 4.4 states the general observed-data requirement. The next corollary gives concrete suficient conditions for the parametric GMMM used in the simulations.

Corollary 4.5 (Parametric identification with fixed market inputs). Consider (4) conditional on measured market inputs, with every response input other than GEO fixed across the two GEO treatment sequences. Suppose that consistency, sequential exchangeability, positivity, and the interference assumption hold. Assume that $\eta _ { M }$ is identified on the required support and that the occurrence probabilities apply to the target population. Suppose that the media transformations are fixed or identified and that the response model with an identity link is correctly specified. If W<sub>∗</sub> has full column rank and $D _ { \eta }$ in Proposition 4.2 has rank four, then $\Delta _ { G }$ is identified. More generally, for a known vector $d _ { \eta . }$ , the condition $d _ { \eta } ^ { \mathsf { T } } v = 0$ whenever $D _ { \eta } v = 0$ is suficient to identify $d _ { \eta } ^ { \mathsf { T } } \beta$ even when the four coeficients are not separately identified.

Proof. The identified measurement parameters and fixed market inputs determine $E ^ { G }$ and $E ^ { P }$ under the two treatment sequences. Fixed or identified transformations then determine the corresponding regressors and their diferences. Proposition 4.2 identifies either the four coeficients or the specified linear efect. The causal assumptions permit these identified quantities to be evaluated under both treatment sequences. Because the response model is written in terms of expected counts, these identified quantities sufice without modeling unobserved realized impressions. □

The corollary relies on occurrence probabilities that apply to the target population and on a correctly specified conditional response mean. Under a nonlinear transformation, a model for an expected count difers from a model for an unobserved realized count: for a random input M, $\mathbb { E } [ h ( M ) \mid { \mathcal { D } } _ { M } ]$ generally difers from $h ( \mathbb { E } [ M \mid { \mathcal { D } } _ { M } ] )$ . Either specification also requires control of confounding between the media input and the response.

The general formula permits treatments to change variables observed later. The simulations condition on generated question counts, shares of system use, notice probabilities, and all response inputs other than GEO. For every parameter sample, the calculation replaces the complete GEO sequence before evaluating the response. The rollout dates are fixed by design, so the simulation obtains counterfactual values by evaluating the specified parametric response model under both sequences rather than by using nonparametric overlap for every cluster history.

A randomized rollout can supply treatment variation by design. An observational rollout instead requires pre-treatment controls and support for both treatment sequences. A variable such as referral trafic or brand search may transmit part of the efect of a current treatment to revenue or conversion. Fixing that variable does not recover the total efect and is not suficient by itself to identify a controlled direct efect (Acharya et al., 2016). If the variable is afected by an earlier treatment and also influences a later treatment, it may belong to $\mathcal { H } _ { i t } ^ { - }$ the longitudinal g-formula then averages over its distribution under the treatment sequence (Robins, 1987).

## 4.4 Decomposition under the Linear Response Model

Under the linear response used in the simulations, $\Delta _ { G }$ can be written as a component associated with the source state and a component due to the change in the transformed GEO input.

Proposition 4.6 (Decomposition under the linear response model). Suppose that $g _ { Y }$ is the identity link and that every input to the response model other than the GEO source state and the GEO input is the same under the two treatment sequences. Assume that the terms below are integrable. Let Z denote the sequence of GEO source states selected by $a _ { G } = 1$ , and let 0 denote the sequence selected by $a _ { G } = 0$ . Under (3), it holds that

$$
\begin{array} { l } { { \displaystyle \Delta _ { G } = \beta _ { D } \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } \mathbb { E } [ Z _ { i t } ] + \beta _ { G } \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } \mathbb { E } \left[ H _ { i t } ^ { G } ( \pmb { Z } ) - H _ { i t } ^ { G } ( \pmb { 0 } ) \right] } } \\ { { \displaystyle ~ + \beta _ { G P } \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } \mathbb { E } \left[ H _ { i t } ^ { P } \left( H _ { i t } ^ { G } ( \pmb { Z } ) - H _ { i t } ^ { G } ( \pmb { 0 } ) \right) \right] } . } \end{array}\tag{27}
$$

Proof. Subtract the two conditional means. Every term that contains neither the GEO source state nor the GEO input cancels. The remaining terms are the term associated with the source state, the change in $H ^ { G }$ , and the interaction between that change and $H ^ { P }$ . Taking expectations and summing over $\mathcal { T } _ { E }$ gives (27). □

The component due to the GEO input contains $( \beta _ { G } + \beta _ { G P } H _ { i t } ^ { P } ) ( H _ { i t } ^ { G } ( Z ) - H _ { i t } ^ { G } ( \bf { 0 } ) )$ , so the interaction makes this component depend on the GEM input. Any other regressor changed by the GEO treatment would add another term.

Equation (27) is an algebraic decomposition of the total GEO efect under the specified response model. We use it to study estimation error in the term for the source state and in the terms that contain the constructed GEO input. A causal mediation interpretation would require interventions that can vary the source state and the generated exposure separately, together with exchangeability and support for both interventions (Imai et al., 2010; Pearl, 1995). Our estimand remains the causal efect of the complete source modification.

## 5 Answer Collection and Simulation Evidence

The answer collection and the simulations serve diferent roles. The collection estimates occurrence probabilities under one observed source state. The simulations introduce changes in source state and generate the corresponding market responses, which permits evaluation of estimators of $\Delta _ { G }$ . Appendices A and B give the collection procedure and additional numerical results.

## 5.1 Target Name in Generated Answers

We submit a fixed list of 56 questions that ask for product recommendations in seven use cases to GPT-5.6 Luna and GPT-4o. The list contains 28 English questions and 28 Japanese questions. Each API call uses the same system instruction and requires web search. Its user message contains only the requested language and question, and Glasp appears in neither message. After storing the complete answer, we record one if it contains a spelling of Glasp in a fixed dictionary and zero otherwise; the indicator records occurrence regardless of sentiment.

Each model answers every question 20 times, giving 2,240 answers. The requests are randomized within four blocks collected during one window, and Table 1 summarizes the resulting occurrence rates. Because the source state is unchanged throughout collection, these observations estimate probabilities for that state. Identification of the GEO treatment efect also uses variation in source state, market counts of the questions, shares of use across systems, notice probabilities, and contemporaneous response data.

Glasp occurs in 33.8% of the GPT-5.6 Luna answers and 27.8% of the GPT-4o answers. Averaging the posterior distributions for the individual questions gives a diference of 5.70 percentage points with a 95% interval of [3.12, 8.27]. The ordering reverses for English questions, where GPT-4o has the higher rate. Conditional on occurrence, GPT-4o also names Glasp first among the candidate brands more often. These results show why the recorded property and the weights assigned to the questions must match the business application. Appendix A reports results by language and use case.

Table 1: Occurrence of Glasp in the Collected Answers
<table><tr><td>Model</td><td></td><td></td><td></td><td>English Japanese Panel Rate Posterior Mean (95% Interval) Mean Rank Rank 1 (%)</td><td></td><td></td></tr><tr><td>GPT-40</td><td>35.4</td><td>20.2</td><td>27.8</td><td>28.8 [27.0, 30.7]</td><td>1.62</td><td>60.5</td></tr><tr><td>GPT-5.6 Luna</td><td>28.6</td><td>38.9</td><td>33.8</td><td>34.5 [32.8, 36.3]</td><td>1.94</td><td>39.4</td></tr></table>

English, Japanese, and Panel Rate are percentages of answers that contain Glasp. Panel Rate averages the 56 observed rates with equal weights. Posterior means and intervals average the distributions for the 56 questions in (32). Mean Rank is the rank at the first occurrence of Glasp among 11 candidate brands, conditional on its occurrence.

## 5.2 Simulation Design

The simulations compare estimators on the same panels and with the same construction of $E ^ { G }$ and $E ^ { P }$ . Five geographic markets each contain six product clusters and are observed for 36 periods. Two clusters begin the GEO treatment in period 19 and two begin in period 23, while two never receive it. Every product cluster contains two question clusters observed on two generative systems. The resulting data contain 1,080 observations across units and periods but only six units of treatment assignment within each market. The response follows (3) with

$$
( \beta _ { D } , \beta _ { G } , \beta _ { P } , \beta _ { G P } ) = ( 0 . 5 5 , 4 . 0 0 , 2 . 2 0 , 0 . 5 0 ) .\tag{28}
$$

In the controlled design, baseline occurrence probabilities follow the hierarchical logit model. A second design uses probabilities estimated from the collected answers and preserves the pairing of the two models for each question, as specified in (33). The treatment efect of GEO is generated in both designs, and the evaluation window is $T _ { E } = \{ 1 , \dots , 3 6 \}$ . The simulations isolate uncertainty in the occurrence and notice models together with estimation of the media transformations: question counts and system shares are known, the demand variable is included among the controls, and every cell has 50 notice observations. A cell consists of one question cluster, one generative system, and one period. Established media channels are omitted. Appendix H gives the complete data-generating process.

The principal comparisons use 120 paired replications in every condition. We evaluate the four response coeficients and $\Delta _ { G }$ using root mean squared error (RMSE) and coverage of 95% intervals. Appendix B.1 defines the metrics and numerical settings. Paired intervals for diferences between methods resample the 120 common replication indices 5,000 times. Diferences between relative RMSE values are reported in percentage points. Appendix B gives additional results for one, five, and 20 answers per cell and for randomized estimates included in estimation.

## 5.3 Estimation of Carryover and Saturation

Before comparing how uncertainty passes between the two models, we examine whether carryover and saturation are estimated or fixed. On the same 120 controlled panels with five answers per cell, one plug-in method fixes the carryover and Hill parameters at their prior means, while a second estimates them from the response data using the same candidates as

Table 2: Efect of Estimating the Transformation Parameters With Five Answers per Cell
<table><tr><td>Estimator</td><td>Transformations</td><td>Effect RMSE (%)</td><td>Vector RMSE</td><td>Coverage</td></tr><tr><td>Plug-in</td><td>Prior mean</td><td>17.642</td><td>1.207</td><td>0.975</td></tr><tr><td>Plug-in</td><td>Estimated from response data</td><td>17.525</td><td>0.945</td><td>0.975</td></tr><tr><td>Two-stage</td><td>Prior distribution</td><td>17.705</td><td>1.032</td><td>0.958</td></tr><tr><td>Joint</td><td>Estimated from response data</td><td>17.541</td><td>0.950</td><td>0.975</td></tr><tr><td>Plug-in</td><td>Known</td><td>17.135</td><td>0.882</td><td>0.950</td></tr><tr><td>Two-stage</td><td>Known</td><td>17.066</td><td>0.876</td><td>0.950</td></tr><tr><td>Joint</td><td>Known</td><td>17.191</td><td>0.884</td><td>0.958</td></tr></table>

the joint posterior. Three diagnostic benchmarks use the true transformation parameters.   
Table 2 reports the results.

Estimating the transformation parameters reduces the RMSE of the coeficient vector from 1.207 to 0.945 for the plug-in method. The corresponding value for the joint posterior is 0.950. The 95% paired interval for the joint value minus the plug-in value is $[ - 0 . 0 0 6 , 0 . 0 1 6 ]$ and the interval for the diference in efect RMSE also contains zero. Estimation of carryover and saturation explains the principal improvement in coeficient recovery over the plug-in method with fixed transformations. The designs based on the collected answers and the designs with misspecification give the same qualitative result (Appendix B.4).

## 5.4 Use of Response Data in the Measurement Model

The cut posterior in (20) estimates the transformation parameters from the response data while keeping the distribution of $\eta _ { M }$ fixed by $\mathcal { D } _ { M }$ . Comparing it with the joint posterior isolates the efect of allowing $\mathcal { D } _ { Y }$ to revise $\eta _ { M }$ . We use M = 128 samples of the measurement parameters, a common set of $J = 7 0 0$ samples from the transformation prior, and 1,000 conditional samples of the response coeficients for each method. Appendix C.3 describes the computation.

Let $J _ { A }$ be the number of generated answers per cell. The controlled design combines $J _ { A } \in \{ 1 , 5 \}$ with standard deviations $\sigma _ { Y } \in \{ 0 . 4 , 1 . 2 \}$ for the response errors. A fifth condition uses occurrence probabilities estimated from the collected answers, with $J _ { A } = 5$ and $\sigma _ { Y } = 1 . 2$ The 120 replications in each condition give 600 panels. Methods use the same observations within a condition, and the controlled conditions use the same market inputs and standardized response innovations. Each row uses 120 paired replications. Efect RMSE is the relative RMSE of $\Delta _ { G }$ . Coverage refers to its 95% interval. The parenthetical labels indicate whether the transformation parameters are estimated from the response data or retained at their prior distribution. With 120 independent replications, coverage near 95% has a binomial Monte Carlo standard error of about 2 percentage points.

A smaller standard deviation for the response errors substantially reduces efect RMSE, but the joint posterior has no uniform advantage over the cut posterior or the plug-in method that estimates the transformations. When $J _ { A } = 1$ and $\sigma _ { Y } = 0 . 4 .$ , the joint posterior reduces coeficient RMSE relative to the cut posterior by 0.0300; the 95% paired interval for this reduction is [0.0221, 0.0383]. The corresponding interval for the diference in efect RMSE contains zero. In the condition based on estimated occurrence probabilities, efect RMSE is 17.326% for the cut posterior and 17.480% for the joint posterior. Their diference is 0.154 percentage points with an interval of [0.046, 0.259]. The interval comparing the cut posterior with the plug-in method contains zero. Table 13 reports all 15 paired comparisons.

Table 3: Comparison on a Common Set of Transformation Draws
<table><tr><td>Method Vector RMSE Effect RMSE (%)</td></tr><tr><td>Coverage Controlled:  $J _ { A } = 1 , \sigma _ { Y } = 0 . 4$ </td></tr><tr><td>Plug-in (estimated) 0.667 6.000 0.958</td></tr><tr><td>Cut (estimated) 0.687 6.010 0.950 Joint 0.657 6.046 0.950</td></tr><tr><td>Two-stage (prior) 0.918 7.075 0.975</td></tr><tr><td>Oracle 0.493 5.723 0.950</td></tr><tr><td>Controlled:</td></tr><tr><td> $J _ { A } = 1 , \sigma _ { Y } = 1 . 2$  Plug-in (estimated) 0.975 17.450 0.967</td></tr><tr><td>Cut (estimated) 0.943 17.519 0.975</td></tr><tr><td>Joint 0.974 17.545 0.975</td></tr><tr><td>Two-stage (prior) 1.020 17.882 0.967</td></tr><tr><td>Oracle 0.886 17.236 0.950</td></tr><tr><td>Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 0 . 4$ </td></tr><tr><td>Plug-in (estimated) 0.639 5.912 0.967</td></tr><tr><td>Cut (estimated) 0.642 5.930 0.967</td></tr><tr><td>Joint 0.636 5.923 0.975</td></tr><tr><td>Two-stage (prior) 0.897 7.070 0.958</td></tr><tr><td>Oracle 0.493 5.723 0.950</td></tr><tr><td>Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Plug-in (estimated) 0.956 17.513 0.975 Cut (estimated) 0.953 17.442 0.975</td></tr><tr><td>Joint 17.479 0.975</td></tr><tr><td>0.963</td></tr><tr><td>Two-stage (prior) 1.027 17.858 0.975</td></tr><tr><td>Oracle 0.886 17.236 0.950</td></tr><tr><td>Estimated occurrence probabilities:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Plug-in (estimated) 0.931 17.359 0.942</td></tr><tr><td>Cut (estimated) 0.912 17.326 0.942</td></tr><tr><td>Joint 0.939 17.480 0.950</td></tr><tr><td>Two-stage (prior)</td></tr><tr><td>0.974 18.099 0.950</td></tr><tr><td>Oracle 0.821 16.984 0.958</td></tr></table>

To assess numerical sensitivity, we doubled both numbers of candidates for eight replications in each condition. The cut and joint estimates of $\Delta _ { G }$ changed by at most 1.602% of the true efect, although the efective sample size for the transformation candidates can approach one when the standard deviation of the response errors is small. Appendix C.3 reports the full diagnostics and states which approximation errors remain when the candidate counts increase.

## 5.5 Accuracy of the Components of the Treatment Efect

The main simulation sets $\beta _ { D } = 0 . 5 5$ because a source modification can afect the response outside generated answers. The total efect therefore combines the term for the source state, the change in the transformed GEO input, and its interaction with GEM. Errors in the two components can partially cancel when the sum is estimated. Let $\begin{array} { r } { n _ { Z } = \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } Z _ { i t } } \end{array}$ be the number of treated observations across units and periods. Define the component for the source state by $C _ { D } = n _ { Z } \beta _ { D }$ and the component that contains the GEO input by $C _ { E } = \Delta _ { G } - C _ { D }$ Every method estimates them as $\widehat { C } _ { D } = n _ { Z } \widehat { \beta } _ { D }$ and $\widehat { C } _ { E } = \widehat { \Delta } _ { G } - \widehat { C } _ { D }$ , using its own estimate of $\beta _ { D }$ . The second component includes the interaction in Proposition 4.6 and is used here as a decomposition of the specified response model.

The treatment sequences give $n _ { Z } = 3 2 0$ and $C _ { D } = 1 7 6$ in every replication. With five answers per cell and $\sigma _ { Y } = 1 . 2$ , the mean total efect is 266.944 in the controlled design and 234.755 in the design based on estimated occurrence probabilities. The corresponding mean values of $C _ { D } / \Delta _ { G }$ are 66.34% and 75.38%. The term associated with the source state accounts for most of the simulated total efect even though its coeficient is unknown to the estimators. The first three numerical columns report relative RMSE for each component. For a component

Table 4: Accuracy of the Treatment Efect and Its Model Components
<table><tr><td colspan="4">Method Total (%) Source State (%) GEO Input (%) Error Correlation</td></tr><tr><td colspan="4">Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Plug-in (estimated)</td><td>17.513</td><td>32.638 48.351</td><td>-0.627</td></tr><tr><td>Cut (estimated)</td><td>17.442 32.526</td><td>47.756</td><td>-0.626</td></tr><tr><td>Joint</td><td>17.479</td><td>32.646 48.366</td><td>-0.631</td></tr><tr><td>Two-stage (prior)</td><td>17.858 36.426</td><td>47.860</td><td>-0.677</td></tr><tr><td>Oracle</td><td>17.236 29.351</td><td>34.825</td><td>-0.523</td></tr><tr><td colspan="4">Estimated occurrence probabilities:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Plug-in (estimated)</td><td>17.359 27.551</td><td>47.129</td><td>-0.555</td></tr><tr><td>Cut (estimated)</td><td>17.326</td><td>27.483 45.929</td><td>-0.556</td></tr><tr><td>Joint</td><td>17.480</td><td>27.872 47.295</td><td>-0.561</td></tr><tr><td>Two-stage (prior)</td><td>18.099</td><td>29.810 48.439</td><td>-0.614</td></tr><tr><td>Oracle</td><td>16.984</td><td>24.470 31.468</td><td>-0.409</td></tr></table>

C and R = 120 replications, the reported value is $\begin{array} { r l } { { 1 0 0 } ( R ^ { - 1 } \sum _ { r = 1 } ^ { R } ( ( { \widehat C _ { r } - C _ { r } } ) / C _ { r } ) ^ { 2 } ) ^ { 1 / 2 } } & { { } } \end{array}$ . The columns have diferent denominators. The final column is the correlation between estimation errors in the component associated with the source state and the component from the GEO input. Section 5.4 reports all five conditions.

In the condition based on estimated occurrence probabilities, the plug-in method has relative RMSE 17.359% for the total efect and 47.129% for the component from the GEO input. The latter value is 45.929% for the cut posterior, 47.295% for the joint posterior, and 31.468% for the oracle. The negative correlations in Table 4 explain why the total can be more accurate than either component. If $e _ { D }$ and $e _ { E }$ denote their errors, the squared error of the total contains $2 e _ { D } e _ { E }$ in addition to the two squared component errors. Errors with opposite signs can cancel through this term. We retain $\Delta _ { G }$ as the primary estimand and report the decomposition alongside the total efect.

## 6 Discussion

GMMM separates construction of the two media inputs from the causal comparison. Generated answers determine occurrence probabilities; question counts, system shares, and notice probabilities put those probabilities on the scale of an MMM; complete treatment sequences then define the efects of GEO and GEM. Plug-in, cut, and joint procedures difer only in how they estimate the transformations and use uncertainty in the constructed inputs.

## 6.1 Interpretation of the Results

The GEO input describes expected noticed occurrence under a source state, and it may be positive both with and without the source modification. GEO changes this input through the occurrence probabilities. The estimand $\Delta _ { G }$ compares the complete source-modification sequences, so it includes the path through the constructed input and the additional path represented by $\beta _ { D } Z _ { i t }$ . For GEM, the corresponding comparison changes the spending sequence that determines sponsored placements.

The collected answers also show that one overall occurrence rate is inadequate. The ordering of GPT-5.6 Luna and GPT-4o changes with language, and rates vary substantially across use cases. A feature used in an application should be chosen for the business question, and the probabilities for individual questions should receive the weights of the target population. Equal weights estimate the rate for the fixed panel used here. They also estimate a population rate when that population has the same distribution of questions.

Identification of $\beta _ { D }$ and $\beta _ { G }$ requires variation in the transformed GEO input beyond a proportional change in the variable for source state after the other regressors have been removed. When the two regressors move together, the condition on the null space in Proposition 4.2 determines whether $\Delta _ { G }$ remains identified. The component results show why this distinction matters in finite samples: errors in the component associated with the source state and the component from the GEO input often have opposite signs and partially cancel in the total efect.

The design that assigns the treatment supplies its causal interpretation. A randomized rollout provides treatment variation by design. An observational rollout requires pre-treatment controls, support for both treatment sequences, and a contemporaneous comparison group when treated and untreated units experience common changes in the generative systems. Variables afected by an earlier treatment may also influence later assignment, in which case the g-formula in (26) averages over their distribution under each treatment sequence.

For estimation, the plug-in method that estimates carryover and saturation is a useful reference when the measurement distribution is concentrated. The cut posterior averages over measurement uncertainty while keeping the distribution of the measurement parameters fixed by the data used to construct the inputs. The joint posterior also uses the response likelihood to revise those parameters. We find no universal ranking: the joint posterior improves coeficient RMSE in one condition with a small standard deviation for the response errors, but it does not improve efect RMSE there and performs worse than the cut posterior in the condition based on estimated occurrence probabilities.

Numerical approximation raises a diferent question. Concentrated importance weights require checks of the number and placement of candidates, while a Markov chain would require checks of mixing and convergence. Increasing the number of candidates leaves the errors from the Laplace approximation for the measurement parameters and the Gaussian approximation to the posterior probability of the sign restrictions unchanged.

The simulations hold the GEO and GEM inputs fixed across estimators and omit established media channels, which isolates estimation of a common GMMM response model. A separate comparison of input construction would keep the response model and transformation estimation fixed while comparing a source indicator, occurrence probabilities, and expected counts under changes in question volume, system shares, and attention. A randomized estimate included in the likelihood informs the same treatment efect; an independently evaluated treatment would address predictive transfer to a diferent intervention.

## 6.2 Alternative Response Models

The construction of $E ^ { G }$ and $E ^ { P }$ precedes the choice of response distribution, which allows an application to use a link and distribution suited to sales, conversions, counts, or rates. Hierarchical coeficients can share information across related units, and coeficients that vary over time can represent changes in the relation between a media input and the response. Other carryover or saturation functions can replace (14) and (15) without changing the definitions of the two expected counts or $\Delta _ { G }$

## 6.3 Alternative Estimation Procedures

The distinction between measurement and response parameters also applies outside the Bayesian formulation. A procedure based on likelihood can estimate all parameters together and use resampling for uncertainty, while a regularized procedure can penalize selected terms in (16). In the Bayesian analysis used here, the cut and joint posteriors difer in whether $\mathcal { D } _ { Y }$ updates $\eta _ { M }$

## 6.4 Uncertainty in Market Inputs

The simulations treat the question counts $N _ { i q t }$ and shares $\omega _ { i p t }$ as known. In an application, the counts may be estimated from a sample and the shares may change during the evaluation window. A probability model for either quantity can be added to $\eta _ { M }$ , after which its uncertainty passes through (6), the media transformations, and the calculation of $\Delta _ { G }$ . This extension changes the construction of the input but not the response model or the definition of the treatment efect.

## 7 Conclusion

GMMM extends MMM to settings in which the media inputs associated with generative AI are not recorded directly. Under each source state, the GEO input is the expected number of generated answers in which a specified property occurs and is noticed; GEO changes this count but need not create it from zero. The GEM input is the expected number of sponsored placements that users notice. Both inputs are defined for each market and period before carryover and the Hill transformation are applied. The treatment efect of GEO is the total efect of the specified source modification under two complete treatment sequences.

The identification results allow the treatment efect to be determined in some designs even when its component coeficients cannot be recovered separately. They also identify the conditions under which a sequence of occurrence probabilities difers from a market count only by scale. In the simulations, estimating carryover and saturation explains most of the improvement in coeficient recovery over a plug-in method that fixes them. Allowing response data to revise the measurement parameters has no consistent benefit for estimating $\Delta _ { G } .$ , and the total efect can conceal larger errors in its two model components.

The collected answers identify occurrence probabilities under the source state present during collection. Identification of the change caused by GEO uses observations under both source states together with contemporaneous market counts, notice observations, and business responses. GMMM combines these measurements and evaluates the causal efect from the complete sequence of media inputs.

## References

Adivit Acharya, Matthew Blackwell, and Maya Sen. Explaining causal findings without bias: Detecting and assessing direct efects. American Political Science Review, 110(3):512–529, 2016. 15

Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, Karthik Narasimhan, and Ameet Deshpande. Geo: Generative engine optimization. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2024. 2

Puneet S. Bagga, Vivek F. Farias, Tamar Korkotashvili, Tianyi Peng, and Yuhang Wu. E-geo: A testbed for generative engine optimization in e-commerce, 2026. arXiv: 2511.20867. 3

Christian Carmona and Geof Nicholls. Semi-modular inference: enhanced learning in multimodular models by tempering the influence of components. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2020. 3

David Chan and Mike Perry. Challenges and opportunities in media mix modeling. Technical report, Google Research, 2017. 3

Aiyou Chen and Timothy C. Au. Robust causal inference for incremental return on ad spend with randomized paired geo experiments. The Annals of Applied Statistics, 16(1):1 – 20, 2022. 3

Darral G. Clarke. Econometric measurement of the duration of advertising efect on sales. Journal of Marketing Research, 13(4):345–357, 1976. 1

Ryan Dew, Nicolas Padilla, and Anya Shchetkina. Your mmm is broken: Identification of nonlinear and time-varying efects in marketing mix models, 2024. arXiv: 2408.07678. 12

Avinava Dubey, Zhe Feng, Rahul Kidambi, Aranyak Mehta, and Di Wang. Auctions with llm summaries. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 713–722. Association for Computing Machinery, 2024. 3

Paul D¨utting, Vahab Mirrokni, Renato Paes Leme, Haifeng Xu, and Song Zuo. Mechanism design for large language models. In Web Conference, 2024. 3

Soheil Feizi, Mohammadtaghi Hajiaghayi, Keivan Rezaei, and Suho Shin. Online advertisements with llms: Opportunities and challenges. SIGecom Exchange, 22(2), 2026. 2

Chang Gong, Di Yao, Lei Zhang, Sheng Chen, Wenbin Li, Yueyang Su, and Jingping Bi. Causalmmm: Learning causal structure for marketing mix modeling. In International Conference on Web Search and Data Mining (WSDM), 2024. 3

Brett R. Gordon, Florian Zettelmeyer, Neha Bhargava, and Dan Chapsky. A comparison of approaches to advertising measurement: Evidence from big field experiments at facebook. Marketing Science, 38(2):193–225, 2019. 3

Mohammad Taghi Hajiaghayi, S´ebastien Lahaie, Keivan Rezaei, and Suho Shin. Ad auctions for llms via retrieval augmented generation. In International Conference on Neural Information Processing Systems (NeurIPS), 2024. 3

D.M. Hanssens, L.J. Parsons, and R.L. Schultz. Market Response Models: Econometric and Time Series Analysis. International Series in Quantitative Marketing. Springer US, 2001. 1

Silan Hu, Shiqi Zhang, Yimin Shi, and Xiaokui Xiao. Gem-bench: A benchmark for ad-injected response generation within generative engine marketing. In Conference on Knowledge Discovery and Data Mining (KDD). Association for Computing Machinery, 2026. 2

Kosuke Imai, Luke Keele, and Dustin Tingley. A general approach to causal mediation analysis. Psychological Methods, 15(4):309–334, Dec 2010. 16

Pierre E. Jacob, Lawrence M. Murray, Chris C. Holmes, and Christian P. Robert. Better together? statistical learning in models made of modules, 2017. arXiv: 1708.08719. 3, 9

Yuxue Jin, Yueqing Wang, Yunting Sun, David Chan, and Jim Koehler. Bayesian methods for media mix modeling with carryover and shape efects. Technical report, Google Inc., 2017. 3, 8

Sunghwan Kim, Wooseok Jeong, Serin Kim, Sangam Lee, and Dongha Lee. Sageo arena: A realistic environment for evaluating search-augmented generative engine optimization. In Conference on Knowledge Discovery and Data Mining (KDD), 2026. 3

Randall A. Lewis and Justin M. Rao. The unfavorable economics of measuring the returns to advertising. The Quarterly Journal of Economics, 130(4):1941–1973, 2015. 3

Olivier Martinez. Optimizing visibility in generative engines: A critical survey of generative engine optimization (2023-2026), 2026. arXiv: 2607.14035. 3

Marc Nerlove and Kenneth J. Arrow. Optimal advertising policy under dynamic conditions. Economica, 29(114):129–142, 1962. 1

Edwin Ng, Zhishi Wang, and Athena Dai. Bayesian time varying coeficient model with applications to marketing mix modeling, 2024. arXiv: 2106.03322. 3

Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669–688, 1995. 16

Martyn Plummer. Cuts in bayesian graphical models. Statistics and Computing, 25(1):37–43, 2015. 3, 9

James Robins. A graphical approach to the identification and estimation of causal parameters in mortality studies with sustained exposure periods. Journal of Chronic Diseases, 40: 139S–161S, 1987. 14, 15

Julian Runge, Igor Skokan, Gufeng Zhou, and Koen Pauwels. Packaging up media mix modeling: An introduction to robyn’s open-source approach, 2025. arXiv: 2403.14674. 3

Lloyd S. Shapley. A value for n-person games, pp. 31–40. Cambridge University Press, 1988. 43

Yunting Sun, Yueqing Wang, Yuxue Jin, David Chan, and Jim Koehler. Geo-level bayesian hierarchical media mix modeling. Technical report, Google Research, 2017. 3

Jon Vaver and Jim Koehler. Measuring ad efectiveness using geo experiments. Technical report, Google Research, 2011. 3

Keisuke Watanabe and Kazuki Nakayashiki. Disentangling answer engine optimization from platform growth: A log-based natural experiment on chatgpt referral trafic, 2026. arXiv: 2606.04362. 45, 46

Shengwei Xu, Zhaohua Chen, Xiaotie Deng, Zhiyi Huang, and Grant Schoenebeck. Ad insertion in llm-generated responses, 2026. arXiv: 2601.19435. 3

Kai Zhang, Xinyue He, and Jingang Yao. From citation selection to citation absorption: A measurement framework for generative engine optimization across ai search platforms, 2026. arXiv: 2604.25707. 3

Yingxiang Zhang, Mike Wurm, Alexander Wakim, Eddie Li, and Ying Liu. Bayesian hierarchical media mix model incorporating reach and frequency data. Technical report, Google Research, 2023. 3

Yingxiang Zhang, Mike Wurm, Eddie Li, Alexander Wakim, Joseph Kelly, Brenda Price, and Ying Liu. Media mix model calibration with bayesian priors. Technical report, Google Research, 2024. 3

## A Collection of Generated Answers and Simulation Inputs

Section 5.1 reports whether Glasp occurs in a fixed collection of generated answers. We specified the questions, model identifiers, language instructions, and search settings before collecting those answers.

Recorded indicator. An answer receives value one when it contains a spelling of Glasp in the target dictionary and zero otherwise. The dictionary is applied after the complete API record has been stored, so the target name is absent from the question and the instructions sent to the model. The indicator records occurrence regardless of whether the surrounding text endorses Glasp. The source state was constant during collection, so the answers identify occurrence probabilities only for that state.

Connection to the GEO input. The collected answers estimate occurrence probabilities under the information environment present during collection. Equation (6) combines these probabilities with the number of relevant questions, the share handled by each generative system, and the probability of notice. Repeated calls characterize variation for each question and model, whereas the choice of questions determines the population described by their weighted average. Applying the estimated probabilities to a market requires weights for that market and generation settings that correspond to those used in the collection.

## A.1 Collection Design and Target Quantities

The 56 questions ask for product recommendations in seven use cases, including web highlighting and summarization with AI. Each use case contains four questions in English and four in Japanese. The questions exclude the target name and its listed aliases. Each model receives the product question, the requested language, and a common system instruction that requires an independent recommendation and a web search before answering. The API call supplies a tool for web search and selects it through tool choice. Appendix J gives the complete instruction and API settings.

For each question, we obtained 20 answers from GPT-5.6 Luna and 20 from GPT-4o. Four randomized blocks contained five repetitions for each combination of question and model, giving

$$
5 6 \times 2 \times 4 \times 5 = 2 , 2 4 0\tag{29}
$$

API records. Collection ran from 05:58 UTC on September 3, 2026, to 04:40 UTC on September 4, 2026. The four blocks randomize request order within this single collection interval and do not represent distinct observation dates.

Let $R _ { q p b r } ^ { \mathrm { t e x t } }$ denote the complete answer for question $q \in \{ 1 , \ldots , 5 6 \}$ , model $p \in \{ 1 , 2 \}$ , block $b \in \{ 1 , \ldots , 4 \}$ , and repetition $r \in \{ 1 , \ldots , 5 \}$ . We define

$$
{ \cal I } _ { q p b r } = { \bf 1 } \left( \mathrm { a ~ l i s t e d ~ s p e l l i n g ~ o f ~ G l a s p ~ o c c u r s ~ i n ~ } R _ { q p b r } ^ { \mathrm { t e x t } } \right) .\tag{30}
$$

For the observed collection period and source state, $I _ { q p b r }$ realizes the Bernoulli variable in (7). The complete answer and the position of the first occurrence among the 11 candidate brands remain available for analyses of placement.

Let t<sup>∗</sup> denote the collection period and $z ^ { * }$ the prevailing source state. The occurrence probability for question $q$ and model p is $p _ { q p } = \pi _ { q p t ^ { * } } ( z ^ { * } )$ . Conditional on $p _ { q p }$ , the 20 calls are modeled as Bernoulli variables with a common probability during the collection interval. Let $J = 2 0$ and $\begin{array} { r } { S _ { q p } = \sum _ { b = 1 } ^ { 4 } \sum _ { r = 1 } ^ { 5 } I _ { q p b r } } \end{array}$ . The estimator for one question and model is $\widehat { p } _ { q p } = S _ { q p } / J$ Given weights $\omega _ { q } \geq 0$ such that $\textstyle \sum _ { q = 1 } ^ { 5 6 } \omega _ { q } = 1$ , define the index for model $p$ by

$$
P _ { p } = \sum _ { q = 1 } ^ { 5 6 } \omega _ { q } p _ { q p } .\tag{31}
$$

The primary analysis sets $\omega _ { q } = 1 / 5 6$ . The resulting index describes the fixed panel of questions; it equals a market occurrence rate only when the weights reproduce the market distribution of questions.

For each question and model, uncertainty from 20 answers is represented by

$$
p _ { q p } \mid S _ { q p } \sim \mathrm { B e t a } \left( S _ { q p } + \frac { 1 } { 2 } , J - S _ { q p } + \frac { 1 } { 2 } \right) .\tag{32}
$$

Independent samples from these distributions are averaged with the weights $\omega _ { q }$ to obtain intervals for $P _ { p }$ and diferences between models. These intervals condition on the Bernoull model and the fixed question weights. They exclude uncertainty from replacing the panel by a diferent population of questions.

All 2,240 requests returned complete answers, and no request or response identifier was duplicated. Each answer contains at least one completed call to the web search tool, giving 3,140 calls in total. GPT-5.6 Luna returned the requested model identifier, whereas GPT-4o returned the dated identifier gpt-4o-2024-08-06.

## A.2 Occurrence Results

Table 1 reports the occurrence rates. Glasp occurs in 378 of 1,120 GPT-5.6 Luna answers and 311 of 1,120 GPT-4o answers, which gives raw panel rates of 33.8% and 27.8%. Averaging the beta distributions for the 56 questions gives posterior means of 34.5% and 28.8%. The posterior mean of the diference between the models is 5.70 percentage points, with a 95% posterior interval of [3.12, 8.27] percentage points and posterior probability above 0.999 of being positive.

The aggregate rates conceal a reversal by language. GPT-4o exceeds GPT-5.6 Luna by 6.79 percentage points for the English questions, whereas GPT-5.6 Luna exceeds GPT-4o by 18.75 percentage points for the Japanese questions. The corresponding 95% posterior intervals are [2.90, 10.02] and [14.11, 21.57] percentage points. Conditional on occurrence, Glasp is the first brand from the candidate list in 60.5% of the GPT-4o answers and 39.4% of the GPT-5.6 Luna answers. GPT-5.6 Luna has the higher overall rate but the lower frequency of first placement.

Figure 2 shows variation across use cases. Both models have their highest occurrence rates for questions about web highlighting and their lowest rates for questions about knowledge management. The reversal by language is especially pronounced for summarization with AI and YouTube learning. A model with system-specific efects for the questions can represent this variation, whereas a model with one common occurrence probability cannot. The ordering

![](images/99d355e2ee4f8ab2908e2b6762d0cd3ca37fb5abef5fd68102c7a05af2a8554e.jpg)  
Figure 2: Occurrence rates by use case and answer language. Bars show posterior means, and error bars show 95% intervals obtained by averaging the distributions for the four questions in each cell.

of the two models is the same in every collection block. GPT-4o rates range from 24.3% to 30.0%, and GPT-5.6 Luna rates range from 31.4% to 36.8%. Because the blocks cover diferent parts of one collection interval, comparisons across them assess stability during collection and do not estimate a time trend.

## A.3 Occurrence Probabilities Used in the Simulations

The variation across questions supplies baseline probabilities for a second simulation design. Within each language, queries are sampled with replacement, and the same sampled query is used for both models. The treatment efect of GEO is generated by the simulation and does not equal the observed diference between the models.

Let $C _ { j p }$ be the occurrence count for question $j$ and model $p$ among J = 20 answers. In each replication, six English and six Japanese question indices are sampled with replacement within language. The selected index $s ( q )$ is common to the two models. Conditional on that selection, the baseline probabilities are sampled independently across models from

$$
\begin{array} { r } { \pi _ { q p } ^ { 0 } \mid s ( q ) , \mathcal { D } _ { M } \sim \mathrm { B e t a } \left( C _ { s ( q ) , p } + \frac { 1 } { 2 } , J - C _ { s ( q ) , p } + \frac { 1 } { 2 } \right) . } \end{array}\tag{33}
$$

The common question selection retains the observed pairing and language composition. Variation within each beta distribution represents uncertainty from the finite number of answers. The sampled probabilities determine the baseline log odds, after which the simulation

adds the same time component and treatment shift in log odds as the controlled design. The simulation specifies the response coeficients and treatment shift independently of the collected answers.

The fitted occurrence model includes a system-specific centered question efect, permitting the diference in log odds between systems to vary across questions. The remaining models used to construct the inputs and the equation for the business response are unchanged.

## B Additional Simulation Results

The simulations keep the panel structure fixed while varying the number of generated answers, the treatment of the transformation parameters, and the relation between the constructed inputs and the business response. Because the treatment assignments are common across methods, the comparisons can isolate the source of each diference before examining numerical accuracy and alternative data-generating processes.

## B.1 Panel Design and Evaluation Criteria

The controlled simulation observes five geographic markets for 36 periods, with $\mathcal { T } _ { E } ~ =$ $\{ 1 , \ldots , 3 6 \}$ . Each market contains six product clusters, every product cluster is linked to two question clusters, and every question cluster is observed on two generative systems. Two product clusters begin the GEO treatment in period 19 and two begin it in period 23; the remaining two never receive GEO. Occurrence of the target name follows the hierarchical logit model. The number of relevant questions varies with latent demand and seasonality, while system shares vary across markets. GEM spending depends on demand observed before treatment and on promotion, and a model for sponsored placements determines the number shown.

The design using estimated occurrence probabilities retains the same panel and treatment structure. Its baseline probabilities follow (33), and the fitted measurement model includes interactions between question and system. The demand variable that afects the response is included among the controls in both simulations. The panel has 1,080 observations, while treatment is assigned at the level of six product clusters and shared across markets.

The response coeficients are given in (28).

For each collection size of one, five, or 20 answers per cell, we run 120 paired replications. The market panel and the simulated randomized estimate are held fixed across collection sizes. Each fit uses 700 candidates from the proposal distribution and 1,000 samples from the conditional posterior for the response parameters defined in Section 3. The plug-in benchmark fixes the sequences of media inputs. The two-stage procedure gives equal weight to samples of the measurement parameters, whereas Joint GMMM reweights the same samples with the response data. The randomized estimate from the target population is the average treatment efect of GEO under the specified rollout relative to the complete sequence with GEO disabled. A second version adds 0.75 to represent a diference between the experimental and target populations. The oracle observes the true sequences of media inputs and transformations.

We assess the response coeficients and $\Delta _ { G }$ by root mean squared error (RMSE) and coverage of 95% intervals. For R replications, the RMSE of the coeficient vector is $( \sum _ { r = 1 } ^ { R } \| \widehat \beta _ { r } -$ $\beta _ { 0 } \| _ { 2 } ^ { 2 } / ( 4 R ) ) ^ { 1 / 2 }$ . Relative efect RMSE is $\begin{array} { r } { ( R ^ { - 1 } \sum _ { r = 1 } ^ { R } ( ( \widehat { \Delta } _ { G , r } - \Delta _ { G , r } ) / \Delta _ { G , r } ) ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ , reported as a percentage. Paired bootstrap intervals use 5,000 resamples of the common replication indices and recompute both RMSEs. Diferences between relative efect RMSEs are measured in percentage points. Efective sample size (ESS) describes concentration of the importance weights. Results with five answers per cell represent the central collection size, while the comparison that adds a randomized estimate uses 20 answers per cell to reduce uncertainty from the answer collection.

## B.2 Simulation Based on the Collected Answers

Using the paired question probabilities in (33), this design assigns GPT-5.6 Luna to the first system and GPT-4o to the second. Both models share the six selected query indices per language in each replication, while the fitted model with interactions between question and system allows their baseline log-odds diferences to vary across queries.

With five simulated answers per cell, the RMSE of the coeficient vector is 0.942 for Joint GMMM, 1.120 for Plug-in GMMM, and 0.981 for Two-stage GMMM (Table 5). The paired diference for Joint minus Plug-in is −0.178, with a 95% bootstrap interval of [−0.261, −0.096]. For Joint minus Two-stage, the diference is −0.038 and the interval is [−0.096, 0.021]. The comparison with Plug-in GMMM supports lower coeficient error when the transformation parameters are estimated. For $\Delta _ { G }$ , the RMSEs are 17.576%, 18.316%, and 18.258%, and both paired intervals for Joint GMMM include zero. Section 5.3 separates the role of estimating the transformation parameters from the role of averaging over measurement uncertainty.

Table 5: Simulation Based on Occurrence Probabilities Estimated From Collected Answers
<table><tr><td>Method</td><td>Vector RMSE</td><td>Effect Bias (%)</td><td>Effect RMSE (%)</td><td>Coverage</td><td>Median ESS</td></tr><tr><td>Plug-in GMMM</td><td>1.120</td><td>-0.966</td><td>18.316</td><td>0.950</td><td></td></tr><tr><td>Two-stage GMMM</td><td>0.981</td><td>-0.246</td><td>18.258</td><td>0.950</td><td></td></tr><tr><td>Joint GMMM</td><td>0.942</td><td>0.380</td><td>17.576</td><td>0.958</td><td>91.2</td></tr><tr><td>Joint + aligned estimate</td><td>0.938</td><td>0.458</td><td>17.057</td><td>0.967</td><td>91.3</td></tr><tr><td>Joint + shifted estimate</td><td>0.942</td><td>3.813</td><td>17.804</td><td>0.950</td><td>91.4</td></tr><tr><td>Oracle</td><td>0.823</td><td>0.369</td><td>17.049</td><td>0.958</td><td></td></tr></table>

Efect Bias and Efect RMSE report the relative bias and relative RMSE of the treatment efect of GEO as percentages. Coverage is the empirical coverage of its 95% interval. ESS is the efective sample size of the importance weights for the joint computations. “Aligned estimate” uses the randomized estimate from the target population, and “shifted estimate” adds 0.75 to that estimate. Dashes mark methods whose weights are fixed by construction. Each row is based on 120 paired replications. A cel is defined by a question, a generative system, and a period.

Adding the randomized estimate from the target population lowers the RMSE of $\Delta _ { G }$ from 17.576% to 17.057%. The paired diference is −0.519 percentage points with a 95% bootstrap interval of [−0.862, −0.182]. The RMSE of the coeficient vector changes only from 0.942 to 0.938, and the paired interval for that change includes zero. When 0.75 is added to the randomized estimate, signed bias in $\Delta _ { G }$ rises from 0.380% to 3.813%. Its RMSE is 17.804%, and the interval for its paired diference from Joint GMMM is [−0.387, 0.826] percentage points.

Figure 3 traces coeficient error as the number of answers increases. More answers reduce uncertainty in the occurrence probabilities, while variation in the response and uncertainty in the transformation parameters remain. The fitted model allows the diference between systems to vary across questions. The controlled comparison below holds the baseline probability model fixed and isolates the efect of averaging over measurement uncertainty.

![](images/911520f3d044e5b5c6468bd0e98b799a336f9ee0a745d0b790b72f90ced91339.jpg)  
Figure 3: Root mean squared error of the coeficient vector in the simulation using estimated occurrence probabilities. The baseline distributions of occurrence indicators are estimated from the collected generated answers. Lines connect ordered numbers of answers, and error bars are 95% bootstrap intervals over replications.

## B.3 Controlled Simulation Results

With five answers per cell, the RMSE of the coeficient vector is 0.950 for Joint GMMM, 1.207 for Plug-in GMMM, and 1.032 for Two-stage GMMM (Table 6). Joint GMMM also reduces the RMSE for the coeficient on the GEO input from 2.068 to 1.558 and for the coeficient on the GEM input from 0.754 to 0.618 relative to Plug-in GMMM. The oracle has vector RMSE 0.880, so variation in the response remains important even when the sequences of media inputs and transformations are known.

Because every method uses the same simulated panel, the paired comparisons attribute diferences to the estimators. Joint GMMM reduces the RMSE of the coeficient vector by 0.256 relative to Plug-in GMMM and by 0.082 relative to Two-stage GMMM. Both 95% bootstrap intervals exclude zero. The relative RMSE of the treatment efect of GEO is similar across the three methods because errors in the individual coeficients can ofset in the scalar estimand.

Figures 4 and 5 show the same pattern across numbers of answers: joint estimation has lower coeficient error than the plug-in estimator with fixed transformations, while their errors in the treatment efect of GEO remain close. This comparison changes both the treatment of measurement uncertainty and the estimation of the transformation parameters. For that reason, the comparison in Section 5.3 varies these features one at a time.

Table 6: Root Mean Squared Error With Five Answers per Cell
<table><tr><td>Method</td><td>Source State</td><td>GEO</td><td>Sponsored</td><td>Interaction</td><td>Vector</td><td>Effect RMSE (%)</td></tr><tr><td>Plug-in GMMM</td><td>0.216</td><td>2.068</td><td>0.754</td><td>0.965</td><td>1.207</td><td>17.642</td></tr><tr><td>Two-stage GMMM</td><td>0.201</td><td>1.704</td><td>0.722</td><td>0.893</td><td>1.032</td><td>17.705</td></tr><tr><td>Joint GMMM</td><td>0.183</td><td>1.558</td><td>0.618</td><td>0.876</td><td>0.950</td><td>17.541</td></tr><tr><td>Joint + aligned estimate</td><td>0.182</td><td>1.549</td><td>0.614</td><td>0.869</td><td>0.944</td><td>17.441</td></tr><tr><td>Joint + shifted estimate</td><td>0.186</td><td>1.559</td><td>0.622</td><td>0.884</td><td>0.953</td><td>17.932</td></tr><tr><td>Oracle</td><td>0.161</td><td>1.429</td><td>0.494</td><td>0.885</td><td>0.880</td><td>17.158</td></tr></table>

“Aligned estimate” uses the randomized estimate from the target population, and “shifted estimate” adds 0.75 to that estimate.

Table 7: Paired Root Mean Squared Error Diferences With Five Answers per Cell
<table><tr><td>Comparison</td><td>Metric</td><td>Difference</td><td>Lower</td><td>Upper</td></tr><tr><td>Joint minus Plug-in GMMM</td><td>Vector RMSE</td><td>-0.2564</td><td>-0.3322</td><td>-0.1745</td></tr><tr><td>Joint minus Plug-in GMMM</td><td>Effect RMSE (pp)</td><td>-0.1010</td><td>-0.7087</td><td>0.4884</td></tr><tr><td>Joint minus Two-stage GMMM</td><td>Vector RMSE</td><td>-0.0821</td><td>-0.1403</td><td>-0.0215</td></tr><tr><td>Joint minus Two-stage GMMM</td><td>Effect RMSE (pp)</td><td>-0.1645</td><td>-0.7435</td><td>0.3804</td></tr></table>

![](images/4e8e76c41d964dfc35733b99cc7a3221a651916b7676957f62eec226221b32dc.jpg)  
Figure 4: Root mean squared error of the coeficient vector by answers per cell. Error bars are 95% bootstrap intervals over replications.

![](images/4012e64b1b5b7811108c8e1bbf03a16ce8023473d8ada7333a1a643bfa324bf7.jpg)  
Figure 5: Relative root mean squared error for the treatment efect of GEO by answers per cell. Error bars are 95% bootstrap intervals over replications.

## B.4 Role of Estimating Carryover and Saturation

We use the same 120 controlled panels with five answers per cell to isolate the contribution of estimating the transformations. In the first comparison, the plug-in estimator uses the response data to estimate the parameters of the media transformations from the same candidates as Joint GMMM while its measurement parameters remain fixed. The second comparison supplies the true transformations to the plug-in, two-stage, and joint estimators, leaving only their treatment of measurement uncertainty to difer. This second comparison is diagnostic because the true transformations would be unavailable in an application. All comparisons retain 700 importance candidates and 1,000 posterior samples.

Estimating the transformation parameters while fixing the measurement inputs lowers vector RMSE from 1.207 to 0.945. The paired diference is −0.262 with a 95% interval of $[ - 0 . 3 3 9 , - 0 . 1 8 2 ]$ . Joint GMMM has vector RMSE 0.950, only 0.005 above the plug-in estimator with estimated transformations, and the interval for this diference is [−0.006, 0.016]. Their relative RMSEs for $\Delta _ { G }$ are 17.541% and 17.525%, with a paired interval that contains zero. When the true transformation parameters are supplied, vector RMSE ranges from 0.876 to 0.884 and relative efect RMSE ranges from 17.066% to 17.191%. The controlled design attributes most of the improvement over fixed transformations to their estimation. Averaging over measurement uncertainty provides no clear additional improvement in $\Delta _ { G }$ in this design.

We also compare estimated transformations on 120 matched panels in each of three designs: the design using estimated occurrence probabilities, shifted transformations, and combined misspecification. Each panel uses five answers per cell, 700 importance candidates, and 1,000 samples from the conditional posterior for the response parameters. Plug-in and joint estimators receive the same transformation samples and panels generated with common random numbers, but the plug-in estimator fixes the measurement parameters at their fitted values. Table 8 reports the three methods needed to isolate the role of estimating the transformation parameters.

Table 8: Comparison After Estimating the Transformation Parameters
<table><tr><td>Design</td><td>Method</td><td>Vector RMSE</td><td>Effect RMSE (%)</td><td>Coverage</td></tr><tr><td rowspan="3">estimated occurrence probabilities</td><td>Plug-in (fixed)</td><td>1.120</td><td>18.316</td><td>0.950</td></tr><tr><td>Plug-in (estimated)</td><td>0.929</td><td>17.417</td><td>0.958</td></tr><tr><td>Joint (estimated)</td><td>0.942</td><td>17.576</td><td>0.958</td></tr><tr><td rowspan="3">Shifted transformations</td><td>Plug-in (fixed)</td><td>1.815</td><td>23.131</td><td>0.808</td></tr><tr><td>Plug-in (estimated)</td><td>0.952</td><td>19.135</td><td>0.908</td></tr><tr><td>Joint (estimated)</td><td>0.961</td><td>19.133</td><td>0.900</td></tr><tr><td rowspan="3">Combined misspecification</td><td>Plug-in (fixed)</td><td>2.014</td><td>22.738</td><td>0.783</td></tr><tr><td>Plug-in (estimated)</td><td>1.071</td><td>18.048</td><td>0.917</td></tr><tr><td>Joint (estimated)</td><td>1.080</td><td>18.059</td><td>0.917</td></tr></table>

Vector RMSE measures recovery of the four response coeficients. Efect RMSE is relative to the true treatment efect of GEO, and coverage refers to its 95% interval. “Fixed” uses the transformations fixed at their prior means, whereas “estimated” allows the response data to change the weights assigned to transformation samples.

In the design using estimated occurrence probabilities, vector RMSE is 0.929 for the plug-in estimator with estimated transformations and 0.942 for Joint GMMM. The paired diference for Joint minus Plug-in is 0.0133, with a 95% bootstrap interval of [0.0008, 0.0262]. The corresponding relative RMSEs for $\Delta _ { G }$ are 17.417% and 17.576%; their diference is 0.159 percentage points with an interval of [−0.012, 0.330]. Estimating the transformations also brings the plug-in and joint results close under shifted transformations and combined misspecification. Table 21 shows that all three intervals for the diference in efect RMSE contain zero. Across these designs, the improvement over the plug-in method with fixed transformations comes chiefly from estimating the transformations.

## B.5 Cut and Joint Posteriors With Common Candidates

Because the two-stage benchmark retains the transformation prior, it cannot isolate the efect of allowing response data to revise the measurement parameters. We make that comparison by evaluating the cut posterior in (20) and the joint posterior on the same M × J combinations of measurement and transformation parameters. We also retain a plug-in estimator that estimates the transformations, the two-stage benchmark, and an oracle that observes the true sequences of media inputs and the true transformations. The computation crosses $M = 1 2 8$ samples of the measurement parameters with J = 700 common transformation samples and then takes 1,000 conditional coeficient samples for each method. This candidate set difers from the 700 independent pairs of measurement and transformation parameters used in the preceding tables, so the numerical levels can difer even when the market panels are the same.

Let $J _ { A }$ denote the number of answers per cell. The controlled design crosses $J _ { A } \in \{ 1 , 5 \}$ with standard deviations $\sigma _ { Y } \in \{ 0 . 4 , 1 . 2 \}$ for the response errors. A fifth condition uses estimated occurrence probabilities with $J _ { A } \ = \ 5$ and $\sigma _ { Y } ~ = ~ 1 . 2$ With 120 replications per condition, this gives 600 primary panels. Controlled conditions share market inputs and standardized response innovations, and all methods within a condition use the same observations. Query volumes and shares of system use remain known, with 50 notice observations per cell. Holding those sources of information and the construction of $E ^ { G }$ and $E ^ { P }$ fixed lets us compare the estimators as the occurrence probabilities become more precise and the standard deviation of the response errors changes. Comparing alternative media inputs would instead require changing how $E ^ { G }$ and $E ^ { P }$ are constructed.

Table 3 reports every condition. When $\sigma _ { Y }$ falls from 1.2 to 0.4, the oracle’s relative efect RMSE falls from 17.236% to 5.723%. Methods that estimate the transformation parameters have RMSE near $6 \% ,$ while the two-stage benchmark remains near $7 \%$ . The concentration near 17% in the original design reflects the standard deviation of the response errors. Even when that standard deviation is smaller, Joint GMMM does not uniformly improve estimation of $\Delta _ { G }$

Each row uses 120 paired replications. Efect RMSE is the relative RMSE of $\Delta _ { G }$ , expressed as a percentage, and Coverage is the empirical coverage of its 95% interval. Cut and Joint estimate the transformation parameters at each measurement sample, while Plug-in estimates them at the measurement point estimate. Two-stage retains the transformation prior. With 120 independent replications, coverage near 95% has a binomial Monte Carlo standard error of about 2 percentage points.

With one answer per cell and $\sigma _ { Y } = 0 . 4$ , Joint GMMM lowers vector RMSE relative to the cut posterior by 0.0300; the 95% paired bootstrap interval for this reduction is [0.0221, 0.0383]. The diference in relative efect RMSE is 0.035 percentage points with an interval of $[ - 0 . 0 5 3 , 0 . 1 3 1 ]$ . When $\sigma _ { Y } = 1 . 2$ , the cut posterior has lower vector RMSE at both collection sizes. Allowing the response data to revise the measurement parameters can afect coeficient recovery without improving estimation of $\Delta _ { G }$

In the condition using estimated occurrence probabilities, efect RMSE is 17.359% for Plug-in, 17.326% for Cut, and 17.480% for Joint. The diference for Joint minus Cut is 0.154 percentage points, with an interval of [0.046, 0.259], while the interval for Cut minus Plug-in includes zero. Across the five conditions, none of the paired comparisons of efect RMSE favors the joint posterior over the cut posterior or the plug-in estimator with estimated transformations. Table 13 reports all 15 comparisons, including the cases in which the joint posterior improves coeficient recovery.

These comparisons approximate the specified posterior distributions with a finite set of candidates. Doubling both candidate counts on eight replications selected in advance per condition changes estimates from the cut and joint posteriors by at most 1.602% of the true efect. The efective sample size within a measurement sample can approach one, especially when the standard deviation of the response errors is small. Appendix C.3 reports the checks in full. The paired intervals describe these computations; they do not determine the ranking under exact posterior integration.

## B.6 Numerical Accuracy

Increasing the number of importance candidates from 700 to 1,400 changes root mean squared error of the coeficient vector from 0.950 to 0.947 and root mean squared error of the treatment efect of GEO from 17.541% to 17.525%. Median efective sample size rises from 124.3 to 251.0. Across replications, the absolute change in the coeficient on the GEO input has median 0.087 and 90th percentile 0.253, while the median absolute change in $\Delta _ { G }$ is 0.386% of its true value.

Table 9: Sensitivity to the Number of Importance Candidates
<table><tr><td>Importance Candidates</td><td>GEO</td><td>GEM</td><td>Vector</td><td>Effect RMSE (%)</td><td>ESS</td><td>ESS q0.10</td></tr><tr><td>700</td><td>1.558</td><td>0.618</td><td>0.950</td><td>17.541</td><td>124.266</td><td>40.394</td></tr><tr><td>1400</td><td>1.556</td><td>0.613</td><td>0.947</td><td>17.525</td><td>250.993</td><td>84.884</td></tr></table>

The lower tail of ESS in Table 10 shows why we report weight concentration alongside the estimates. In the correctly specified design, the tenth percentile rises from 40.4 with 700 candidates to 84.9 with 1,400 candidates, whereas it is 8.6 under combined misspecification with 700 candidates. Panels with low ESS remain in the reported distributions. Increasing the candidate count assesses the stability of importance sampling, while the hierarchical Laplace approximation and the numerical probabilities used for coeficient constraints must be assessed by other calculations. We have not compared these computations with exact posterior integration.

Table 10: Diagnostics for the Importance Weights
<table><tr><td>Design</td><td>Median</td><td>q0.10</td><td>q0.05</td><td>Minimum</td></tr><tr><td>Correctly specified, K=700</td><td>124.266</td><td>40.394</td><td>15.465</td><td>3.042</td></tr><tr><td>Correctly specified, K=1400</td><td>250.993</td><td>84.884</td><td>36.550</td><td>8.176</td></tr><tr><td>Combined misspecification, K=700</td><td>25.947</td><td>8.616</td><td>5.466</td><td>2.779</td></tr></table>

## B.7 Alternative Data-Generating Processes

The remaining designs alter either the data used to construct the inputs or the response equation. Two add overdispersion to the occurrence indicators or the counts of sponsored placements; the others shift the transformation parameters, add a nonlinear term omitted from the fitted response model, or combine these changes. Each design uses 120 replications with five answers per cell, 700 importance candidates, and 1,000 samples from the conditional posterior for the response parameters. Table 11 reports relative RMSE for $\Delta _ { G } .$ , vector RMSE, interval coverage, and weight concentration.

Adding overdispersion to the occurrence indicators or the counts of sponsored placements produces similar relative efect RMSE across the three estimators. Under shifted transformations, relative efect RMSE is 19.133% for Joint GMMM, 23.131% for Plug-in GMMM, and 22.550% for Two-stage GMMM. When all departures are combined, the corresponding value for Joint GMMM is 18.059% with bias −4.683% and coverage 91.7%. Plug-in GMMM has RMSE 22.738% and bias −14.835%, while Two-stage GMMM has RMSE 22.110% and bias −14.082%. The matched comparisons in Table 8 attribute most of these diferences to estimation of the transformation parameters. Appendix H.2 reports the oracle, which isolates error from the response model after the media inputs are observed.

Table 11: Results Under Alternative Data-Generating Processes With Five Answers per Cell
<table><tr><td>Design</td><td>Plug-in</td><td>Two-stage</td><td>Joint</td><td>Joint Vector RMSE</td><td>Coverage</td><td>ESS</td></tr><tr><td>Occurrence overdispersion</td><td>19.005</td><td>18.689</td><td>18.869</td><td>0.887</td><td>0.908</td><td>3</td></tr><tr><td>Overdispersion in sponsored placements</td><td>18.934</td><td>18.677</td><td>18.972</td><td>0.881</td><td>0.900</td><td>38</td></tr><tr><td>Shifted transformations</td><td>23.131</td><td>22.550</td><td>19.133</td><td>0.961</td><td>0.900</td><td>1</td></tr><tr><td>Omitted nonlinear response</td><td>18.145</td><td>17.912</td><td>17.959</td><td>0.951</td><td>0.917</td><td>30</td></tr><tr><td>Combined misspecification</td><td>22.738</td><td>22.110</td><td>18.059</td><td>1.080</td><td>0.917</td><td></td></tr></table>

The first three method columns report the relative RMSE of the treatment efect of GEO in percent. Joint Vector RMSE, Coverage, and ESS $q _ { 0 . 1 0 }$ report the RMSE of the coeficient vector, interval coverage, and the tenth percentile of the Joint GMMM efective sample size.

## B.8 Randomized Estimate of the GEO Treatment Efect

With 20 answers per cell, adding the randomized estimate from the target population changes relative efect RMSE from 17.430% to 17.424% and vector RMSE from 0.950 to 0.947. When the randomized estimate is shifted, relative efect RMSE is 18.013% and signed bias is 4.169%, compared with bias 1.243% without that estimate. The randomized quantity is the average treatment efect of GEO, including the component associated with the source state. Appendix B.2 reports a larger reduction in RMSE in the design based on estimated occurrence probabilities.

Table 12: Use of a Randomized Estimate With 20 Answers per Cell
<table><tr><td>Method</td><td>Effect Bias (%)</td><td>Effect RMSE (%)</td><td>Coverage</td><td>Vector RMSE</td></tr><tr><td>Joint GMMM</td><td>1.243</td><td>17.430</td><td>0.967</td><td>0.950</td></tr><tr><td>Joint + aligned estimate</td><td>1.259</td><td>17.424</td><td>0.967</td><td>0.947</td></tr><tr><td>Joint + shifted estimate</td><td>4.169</td><td>18.013</td><td>0.967</td><td>0.951</td></tr></table>

“Aligned estimate” uses the randomized estimate from the target population, and “shifted estimate” adds 0.75 to that estimate.

## C Measurement Uncertainty and Numerical Integration

Once the occurrence probabilities have been estimated, two diferent sources of error remain. Replacing a market count by an estimated probability changes the regression input, whereas averaging over uncertain inputs and transformation parameters requires numerical integration.

Finite collections of repeated answers in a static linear model. Suppose that $Y = \beta _ { G } E ^ { G } + \varepsilon$ and $E ^ { G } = c \pi$ , where c converts the occurrence probability $\pi$ to an expected noticed count by combining the relevant market opportunity and notice probability. Let ${ \widehat { \pi } } = \pi + u$ be an estimate based on finitely many generated answers. Assume that all variables have finite second moments, $\operatorname { V a r } ( { \widehat { \pi } } ) > 0 , \mathbb { E } [ u \mid \pi , c ] = 0$ , and $\mathrm { C o v } ( \widehat { \pi } , \varepsilon ) = 0$ . The population least squares slope from regressing Y on $\widehat { \pi }$ with an intercept is

$$
b _ { \mathrm { r a t e } } = \beta _ { G } \frac { \mathrm { C o v } ( \pi , c \pi ) } { \mathrm { V a r } ( \pi ) + \mathrm { V a r } ( u ) } .\tag{34}
$$

If $c = c _ { 0 }$ is constant, then the slope becomes

$$
b _ { \mathrm { r a t e } } = \beta _ { G } c _ { 0 } \frac { \mathrm { V a r } ( \pi ) } { \mathrm { V a r } ( \pi ) + \mathrm { V a r } ( u ) } .\tag{35}
$$

Proof. The population slope is $\operatorname { C o v } ( \widehat { \pi } , Y ) / \operatorname { V a r } ( \widehat { \pi } )$ . The conditional mean restriction on u implies that $\mathrm { C o v } ( u , c \pi ) = 0$ and $\mathrm { C o v } ( u , \pi ) = 0$ . Together with the assumption on ε, these relations give $\operatorname { C o v } ( \widehat { \pi } , Y ) = \beta _ { G } \operatorname { C o v } ( \pi , c \pi )$ and $\operatorname { V a r } ( \widehat { \pi } ) = \operatorname { V a r } ( \pi ) + \operatorname { V a r } ( u )$ , which establish (34). If $c = c _ { 0 }$ holds, then $\mathrm { C o v } ( \pi , c \pi ) = c _ { 0 } \mathrm { V a r } ( \pi )$ , and (35) follows. □

When $u = 0$ and $c$ is constant, the change of scale can be absorbed by the response coeficient. Variation in c across observations changes the covariance in (34), while estimating $\pi$ from a finite collection of repeated answers adds the attenuation term in the denominator. These are distinct reasons why an occurrence probability can fail to represent the corresponding market count.

## C.1 Nonlinear Transformation of an Uncertain Input

Even after an input has been expressed as an expected market count, uncertainty about that count can matter because the media transformation is nonlinear. Conditional on $\mathcal { D } _ { M }$ , let $\widetilde { A }$ denote an uncertain adstock value and define $\overline { { A } } = \mathbb { E } [ \widetilde { A } \mid \mathcal { D } _ { M } ]$ . A plug-in analysis uses $h ( { \overline { { A } } } )$ Averaging over the measurement distribution instead gives $\mathbb { E } [ h ( \widetilde { A } ) \mid { \mathcal { D } } _ { M } ]$ before the response data alter any weights.

Proposition C.1 (Transformation of an uncertain media input). Suppose that $\widetilde { A }$ is supported on an interval ${ \mathcal { A } } \subset ( 0 , \infty )$ and that both $\widetilde { A }$ and $h ( \widetilde { A } )$ are integrable conditional on $\mathcal { D } _ { M }$ . If h is convex on $\mathcal { A } _ { : }$ , then it holds that

$$
\begin{array} { r } { \mathbb { E } \left[ h ( \widetilde { A } ) \mid { \mathcal D } _ { M } \right] \ge h \left( \mathbb { E } [ \widetilde { A } \mid { \mathcal D } _ { M } ] \right) . } \end{array}\tag{36}
$$

If h is concave on ${ \mathcal { A } } ,$ , then the reverse inequality holds. Equality holds when $\widetilde { A }$ is degenerate conditional on $\mathcal { D } _ { M }$ or when h is afine on the convex hull of its conditional support.

Proof. The convex case follows from Jensen’s inequality conditional on $\mathcal { D } _ { M }$ . Applying the same argument $\mathrm { t o } \mathrm { ~ - } h$ gives the concave case. The stated equality cases follow directly.

For the Hill function in (15), the second derivative at $a > 0$ is

$$
h ^ { \prime \prime } ( a ) = \frac { \kappa \theta ^ { \kappa } a ^ { \kappa - 2 } \left( ( \kappa - 1 ) \theta ^ { \kappa } - ( \kappa + 1 ) a ^ { \kappa } \right) } { \left( a ^ { \kappa } + \theta ^ { \kappa } \right) ^ { 3 } } .\tag{37}
$$

If $0 < \kappa \leq 1$ holds, then the Hill function is concave on $( 0 , \infty )$ . If $\kappa > 1$ holds, its curvature changes at $a = \theta ( ( \kappa - 1 ) / ( \kappa + 1 ) ) ^ { 1 / \kappa }$ . The direction of the plug-in discrepancy depends on the range of the input after carryover. Replacing an uncertain sequence by one fitted sequence need not reproduce the estimate obtained by averaging over that uncertainty.

## C.2 Joint Weights Under the Measurement Approximation

The joint calculation assigns weights based on the response likelihood to candidates sampled from the measurement approximation. To interpret the calculation, one must specify the distribution approached as the number of candidates increases while the measurement approximation remains fixed.

Let $\mathcal { D } _ { M }$ denote the measurement data and $\mathcal { D } _ { Y }$ the response data. The parameter vector $\eta$ determines the constructed media inputs and their transformations, and $q ( \eta \mid \mathcal { D } _ { M } )$ denotes the proposal distribution in (17). Let $\vartheta _ { Y }$ contain the response coeficients and the remaining parameters of the response distribution. For fixed $\eta ,$ define the response marginal likelihood by

$$
m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) = \int p ( \mathcal { D } _ { Y } \mid \eta , \vartheta _ { Y } ) p ( \vartheta _ { Y } \mid \eta ) d \vartheta _ { Y } .\tag{38}
$$

The weighted calculation targets

$$
{ \widetilde p } ( \eta \mid { \mathcal D } _ { M } , { \mathcal D } _ { Y } ) = \frac { m _ { Y } ( { \mathcal D } _ { Y } \mid \eta ) q ( \eta \mid { \mathcal D } _ { M } ) } { \int m _ { Y } ( { \mathcal D } _ { Y } \mid u ) q ( u \mid { \mathcal D } _ { M } ) d u } .\tag{39}
$$

Proposition C.2 (Limit of the joint weights). Suppose that $\eta _ { 1 } , \dots , \eta _ { K }$ are independent samples from $q ( \eta \mid D _ { M } )$ and that $\begin{array} { r } { 0 < \int m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) q ( \eta \mid \mathcal { D } _ { M } ) d \eta < \infty } \end{array}$ . Let $K  \infty$ while $\mathcal { D } _ { M } , \mathcal { D } _ { Y }$ , and $q$ remain fixed, and define

$$
w _ { k } = \frac { m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { k } ) } { \sum _ { j = 1 } ^ { K } m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { j } ) } .\tag{40}
$$

Let $\mathbb { E } _ { q }$ denote expectation under $q ( \eta \mid \mathcal { D } _ { M } )$ . For every function h satisfying $\mathbb { E } _ { q } [ | h ( \eta ) | m _ { Y } ( \mathcal { D } _ { Y } \mid$ $\eta ) ] < \infty$ , it holds that

$$
\sum _ { k = 1 } ^ { K } w _ { k } h ( \eta _ { k } ) \longrightarrow \int h ( \eta ) \widetilde { p } ( \eta \mid \mathcal { D } _ { M } , \mathcal { D } _ { Y } ) d \eta \quad a l m o s t s u r e l y .\tag{41}
$$

Let $\mu _ { h }$ denote the integral on the right. $I f \mathbb { E } _ { q } [ m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) ^ { 2 } ( h ( \eta ) - \mu _ { h } ) ^ { 2 } ] < \infty$ holds, then we also have

$$
\sqrt K ( \sum _ { k = 1 } ^ { K } w _ { k } h ( \eta _ { k } ) - \mu _ { h } ) \overset { d } {  } { \mathcal { N } } ( 0 , \sigma _ { h } ^ { 2 } ) ,\tag{42}
$$

where

$$
\sigma _ { h } ^ { 2 } = \frac { \mathbb { E } _ { q } \left[ m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) ^ { 2 } ( h ( \eta ) - \mu _ { h } ) ^ { 2 } \right] } { \mathbb { E } _ { q } \left[ m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) \right] ^ { 2 } } .\tag{43}
$$

If $q ( \eta \mid \mathcal { D } _ { M } ) = p ( \eta \mid \mathcal { D } _ { M } )$ holds, then $\widetilde { p }$ equals the joint posterior under the stated model.

Proof. The numerator and denominator of the self-normalized estimator are sample averages under $q .$ . The strong law of large numbers gives the almost-sure limits

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } h ( \eta _ { k } ) m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { k } ) \longrightarrow \int h ( \eta ) m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) q ( \eta \mid \mathcal { D } _ { M } ) d \eta ,\tag{44}
$$

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { k } ) \longrightarrow \int m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) q ( \eta \mid \mathcal { D } _ { M } ) d \eta .\tag{45}
$$

Their ratio is the expectation under ${ \widetilde { p } } .$ The central limit theorem for self-normalized importance sampling gives the second result under the stated second-moment condition. When q equals the measurement posterior, Bayes’ rule shows that multiplication by the response marginal likelihood gives the joint posterior up to normalization. □

Under the second-moment condition in Proposition C.2, the numerical integration error for the specified marginal likelihood is of order $K ^ { - 1 / 2 }$ . Replacing the measurement posterior by $q _ { L }$ , or replacing the Student probability of the sign restrictions by a Gaussian approximation, changes the target distribution. Increasing K leaves those approximation errors unchanged. When a numerical marginal likelihood is substituted for $m _ { Y }$ , the same argument gives convergence to the corresponding reweighted approximation. Efective sample size measures concentration of the weights. Approximation error must be assessed numerically, and the identification assumptions in Section 4 must be justified from the study design.

A second limit applies when the entire proposal distribution, including the transformation parameters, concentrates at one value.

Proposition C.3 (Agreement under concentration of the proposal distribution). Let $q _ { N } ( \eta )$ be a sequence of proposal distributions indexed by increasingly informative measurement designs, where $N \in  { \mathbb { N } }$ and $N \to \infty$ , and let $\widehat { \eta } _ { N }$ be the corresponding plug-in estimate. Suppose that $\widehat { \eta } _ { N } \overset { p } {  } \eta _ { 0 }$ and that $q _ { N }$ converges weakly in probability to a point mass at $\eta _ { 0 }$ . Conditional on fixed response data $\mathcal { D } _ { Y }$ , suppose that $h ( \eta )$ and $m _ { Y } ( \mathcal { D } _ { Y } \mid \eta )$ are bounded and continuous at $\eta _ { 0 }$ with $m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { 0 } ) > 0$ . Then, it holds that

$$
h ( \widehat { \eta } _ { N } ) \stackrel { p } {  } h ( \eta _ { 0 } ) ,\tag{46}
$$

$$
\int h ( \eta ) q _ { N } ( \eta ) d \eta \stackrel { p } { \to } h ( \eta _ { 0 } ) ,\tag{47}
$$

$$
\frac { \int h ( \eta ) m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) q _ { N } ( \eta ) d \eta } { \int m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) q _ { N } ( \eta ) d \eta } \stackrel { p } {  } h ( \eta _ { 0 } ) .\tag{48}
$$

Proof. Equation (46) follows from the continuous mapping theorem, and weak convergence of $q _ { N }$ gives (47). Applying the same argument to the bounded functions $h ( \eta ) m _ { Y } ( \mathcal { D } _ { Y } \mid \eta )$ and $m _ { Y } ( \mathcal { D } _ { Y } \mid \eta )$ gives limits $h ( \eta _ { 0 } ) m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { 0 } )$ and $m _ { Y } ( \mathcal { D } _ { Y } \mid \eta _ { 0 } )$ for the numerator and denominator in (48). The denominator has a positive limit, so their ratio converges to $h ( \eta _ { 0 } )$ . □

The proposition requires concentration of the entire proposal distribution. More generated answers provide information about the occurrence probabilities, but uncertainty about market opportunities, notice probabilities, sponsored placements, and media transformations must also vanish for its conclusion to apply. The comparisons in Section 5.3 isolate the contribution from estimating the transformation parameters.

Suppose that an external randomized experiment $\mathcal { D } _ { E }$ is conditionally independent of $\mathcal { D } _ { Y }$ given the common parameters. The integrated likelihood then factorizes as $p ( \mathcal { D } _ { Y } , \mathcal { D } _ { E }$ $\eta ) = m _ { Y } ( \mathcal { D } _ { Y } \mid \eta ) p ( \mathcal { D } _ { E } \mid \mathcal { D } _ { Y } , \eta )$ . Because the two data sources share response coeficients, the second factor averages the experimental likelihood under the response posterior given $\mathcal { D } _ { Y }$ . Integrating the experimental likelihood separately would omit this conditioning. The estimated efect must refer to the same treatment and population, except for any population diference represented by (23).

## C.3 Numerical Comparison on Common Samples

The paired diferences in Table 13 use the same 120 replications per condition as the comparison in which all methods estimate the transformations. Each entry gives the RMSE of the first method minus that of the second, followed by its 95% interval from 5,000 paired bootstrap resamples. Negative diferences favor the first method, and diferences in relative efect RMSE are expressed in percentage points.

Table 13: Paired RMSE Diferences When All Methods Estimate the Transformations
<table><tr><td>Comparison</td><td>Vector Difference</td><td>Effect Difference (pp)</td><td></td></tr><tr><td>Controlled:  $J _ { A } = 1 , \sigma _ { Y } = 0 . 4$ </td><td></td><td>0.010 [-0.026, 0.048]</td><td></td></tr><tr><td>Cut minus Plug-in Joint minus Cut</td><td>0.0201 [0.0142,0.0259] -0.0300 [-0.0383, -0.0221]</td><td></td><td>0.035 [-0.053, 0.131]</td></tr><tr><td>Joint minus Plug-in</td><td>-0.0098 [-0.0165, -0.0038]</td><td></td><td>0.046 [-0.053, 0.157]</td></tr><tr><td>Controlled:  $J _ { A } = 1 , \sigma _ { Y } = 1 . 2$ </td><td></td><td></td><td></td></tr><tr><td colspan="4"></td></tr><tr><td>Cut minus Plug-in</td><td>-0.0318 [−0.0447, -0.0192]</td><td></td><td>0.069 [-0.024, 0.166]</td></tr><tr><td>Joint minus Cut</td><td>0.0304 [0.0140, 0.0478]</td><td></td><td>0.025 [-0.081, 0.124]</td></tr><tr><td>Joint minus Plug-in</td><td>-0.0014 [-0.0131, 0.0106]</td><td></td><td>0.094 [-0.022, 0.217]</td></tr><tr><td colspan="4">Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 0 . 4$ </td></tr><tr><td>Cut minus Plug-in Joint minus Cut</td><td>0.0034 [0.0014, 0.0054]</td><td></td><td>0.017 [-0.007, 0.046]</td></tr><tr><td>Joint minus Plug-in</td><td>-0.0057 [-0.0081, -0.0033]</td><td></td><td>-0.006 [-0.034, 0.019]</td></tr><tr><td>Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td><td>-0.0024 [−0.0044, -0.0002]</td><td></td><td>0.011 [-0.022, 0.050]</td></tr><tr><td colspan="4"></td></tr><tr><td>Cut minus Plug-in</td><td>-0.0028 [-0.0091,0.0031]</td><td></td><td>-0.070 [-0.161,0.018]</td></tr><tr><td>Joint minus Cut</td><td>0.0101 [0.0025, 0.0180]</td><td></td><td>0.036 [-0.057, 0.125]</td></tr><tr><td>Joint minus Plug-in</td><td>0.0072 [0.0001,0.0147]</td><td></td><td>-0.034 [−0.123, 0.054]</td></tr><tr><td colspan="4">Estimated occurrence probabilities:</td></tr><tr><td>Cut minus Plug-in</td><td> $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$   $- 0 . 0 1 8 4 \ [ - 0 . 0 2 8 3 , - 0 . 0 0 9 2 ]$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>-0.033 [-0.152, 0.088]</td></tr><tr><td>Joint minus Cut</td><td>0.0265 [0.0136, 0.0406]</td><td></td><td>0.154 [0.046, 0.259]</td></tr><tr><td>Joint minus Plug-in</td><td>0.0081 [-0.0027, 0.0204]</td><td></td><td>0.121 [-0.016,0.261]</td></tr></table>

Table 14 reports concentration of the measurement weights and, within each measurement sample, the transformation weights. Measurement ESS is the median ESS of the joint measurement marginal across 120 replications. The cut marginal is uniform by definition and has no corresponding entry. Conditional ESS is the median across replications of the median transformation ESS within a replication, while Minimum ESS is the smallest conditional transformation ESS over all measurement samples and replications. The last two quantities agree for Cut and Joint because their conditional transformation weights are the same.

For the numerical check, we increase M from 128 to 256 and J from 700 to 1,400 for the first eight replications in every condition, leaving the 120-replication comparisons unchanged. The last two columns report the median and maximum absolute changes in the posterior mean of the treatment efect as percentages of the true efect. The largest change for Cut or Joint is 1.602%, and the largest change in an interval bound is 4.357%. This calculation examines sensitivity to the numbers of candidates and posterior samples. Assessing the Laplace approximation and the Gaussian approximation to the sign probability requires additional calculations.

Table 14: Integration Diagnostics With Twice as Many Candidates
<table><tr><td>Method Measurement ESS</td><td>Conditional ESS Minimum ESS Median Change Maximum Change</td><td></td><td></td><td></td></tr><tr><td colspan="5">Controlled:  $J _ { A } = 1 , \sigma _ { Y } = 0 . 4$ </td></tr><tr><td>Cut</td><td></td><td></td><td>0.356</td><td>0.876</td></tr><tr><td>Joint</td><td>10.9 10.9</td><td>1.0 1.0</td><td>0.454</td><td>0.901</td></tr><tr><td colspan="5">Controlled:  $J _ { A } = 1 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Cut</td><td></td><td></td><td>0.220</td><td>1.038</td></tr><tr><td>Joint</td><td>111.7</td><td>2.2 2.2</td><td>0.348</td><td>1.548</td></tr><tr><td colspan="5">Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 0 . 4$ </td></tr><tr><td>Cut</td><td>87.7</td><td></td><td>0.440</td><td>1.098</td></tr><tr><td>Joint</td><td></td><td>1.1 1.1</td><td>0.365</td><td>1.002</td></tr><tr><td colspan="5">Controlled:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Cut</td><td></td><td>138.3</td><td></td><td></td></tr><tr><td>Joint</td><td>121.9</td><td>138.3</td><td>0.240 0.293</td><td>0.566 1.190</td></tr><tr><td colspan="5">2.2 Estimated occurrence probabilities:  $J _ { A } = 5 , \sigma _ { Y } = 1 . 2$ </td></tr><tr><td>Cut</td><td>105.9</td><td>4.3</td><td>0.697</td><td>1.602</td></tr><tr><td>Joint</td><td>105.9</td><td>4.3</td><td>0.597</td><td>1.147</td></tr></table>

## D Attribution With Channel Interactions

The interaction term belongs jointly to GEO and GEM, so the two efects obtained by disabling one channel at a time need not sum to the efect of disabling both. When an application requires an additive allocation across the channels, the Shapley values are

$$
\Phi _ { G } = \frac { 1 } { 2 } \left( V ( 1 , 0 ) - V ( 0 , 0 ) \right) + \frac { 1 } { 2 } \left( V ( 1 , 1 ) - V ( 0 , 1 ) \right) ,\tag{49}
$$

$$
\Phi _ { P } = \frac { 1 } { 2 } \left( V ( 0 , 1 ) - V ( 0 , 0 ) \right) + \frac { 1 } { 2 } \left( V ( 1 , 1 ) - V ( 1 , 0 ) \right) .\tag{50}
$$

This allocation assigns the interaction once and satisfies $\Phi _ { G } + \Phi _ { P } = V ( 1 , 1 ) - V ( 0 , 0 )$ (Shapley, 1988). The same definition extends to more channels, although each additional interacting channel increases the number of treatment combinations that must be evaluated.

## E Consequences of Treatment Design in Simulation

Two auxiliary simulations isolate consequences of treatment design that the response equation alone cannot resolve: one introduces time variation shared by treated and untreated units, and the other conditions on a variable afected by treatment.

In the first simulation, treated and untreated clusters share platform growth, while treated clusters receive an additional GEO efect after treatment begins. Across 500 replications, the comparison of treated units before and after treatment has bias 0.081 and zero coverage for a 95% interval. Diference-in-diferences removes the shared growth component, reducing bias below 0.001 and RMSE to 0.008, with coverage of 0.954.

![](images/163fa7f407516a87051034f7ab248c1868f035f414cfec10cd48445d0bcc477a.jpg)  
Figure 6: Diference-in-diferences removes platform growth shared by treated and untreated clusters. Error bars equal 1.96 Monte Carlo standard errors.

Table 15: Simulation of Platform Growth
<table><tr><td>Method</td><td>Bias</td><td>RMSE</td><td>Coverage</td></tr><tr><td>Change within treated units</td><td>0.081</td><td>0.082</td><td>0.000</td></tr><tr><td>Difference after treatment</td><td>-0.000</td><td>0.032</td><td>0.952</td></tr><tr><td>Difference-in-differences</td><td>-0.000</td><td>0.008</td><td>0.954</td></tr></table>

The second simulation assigns a direct GEO efect of 0.70 and an efect through brand search of 0.88. A response model that omits brand search estimates their sum, 1.58, at 1.579. Because the disturbances for brand search and the response are independent in this additive design, conditioning on brand search estimates the direct efect at 0.700. This interpretation relies on the stated absence of confounding between brand search and the response; it does not justify conditioning on an arbitrary variable afected by treatment.

![](images/97f6ff93581d341c7b4b05f2783e11190691b8ceed696c68730fdca0fc9e10ed.jpg)  
Figure 7: Estimates of the total and direct efects in the additive mediation design with independent disturbances. Error bars equal 1.96 Monte Carlo standard errors.

Table 16: Simulation of Post-Treatment Adjustment
<table><tr><td>Method</td><td>Estimate</td><td>Standard Deviation</td><td>Bias for Total</td><td>Bias for Direct</td></tr><tr><td>Model for the total effect</td><td>1.579</td><td>0.028</td><td>-0.001</td><td>0.879</td></tr><tr><td>Model controlling for brand search</td><td>0.700</td><td>0.031</td><td>-0.880</td><td>0.000</td></tr></table>

## F Public Referral Trafic Analysis

The repository accompanying Watanabe & Nakayashiki (2026) contains weekly treated and control session indices from July 2025 through May 2026.<sup>2</sup> It does not contain the contemporaneous question counts, generated answers, or notice observations needed to construct the GEO input, and it has no information on sponsored placements. The analysis uses the referral indices alone. The source material places the treatment transition around December 2025 and January 2026. We use December 30, 2025 as the central first posttreatment week and examine the neighboring weekly dates.

Let $R _ { t }$ denote the ratio of treated to control sessions and let $r _ { t } = \log R _ { t }$ . For a specified first post-treatment week $t _ { 0 }$ , we estimate

$$
r _ { t } = \delta _ { 0 } + \delta _ { 1 } t + \delta _ { 2 } \mathbf { 1 } ( t \geq t _ { 0 } ) + \delta _ { 3 } ( t - t _ { 0 } ) \mathbf { 1 } ( t \geq t _ { 0 } ) + e _ { t } .\tag{51}
$$

Using Newey–West covariance with four lags, we report an interval and a t test for the immediate ratio $\exp ( \delta _ { 2 } )$ . A platform shock cancels from $R _ { t }$ when it changes treated and control trafic by the same proportion. The segmented terms describe how the groups diverge from the trend estimated before treatment, including changes that afect them diferently.

For December 30, the immediate ratio is 1.842, with a 95% interval from 1.306 to 2.599. The geometric mean ratio implied by the model over the final four weeks is 2.301, compared with an observed geometric mean of 2.055 after adjustment for the pre-treatment trend. The placebo rank is 0.150 among 19 admissible dates before treatment. Because the pre-treatment period contains changes at least as large as the estimate at rollout, this diagnostic does not by itself establish a treatment efect.

Table 17: Referral Trafic Analysis
<table><tr><td>Quantity</td><td>Estimate</td><td>Lower</td><td>Upper</td><td>P-value</td></tr><tr><td>Weekly trend factor before treatment</td><td>1.027</td><td>1.007</td><td>1.048</td><td>0.010</td></tr><tr><td>Immediate ratio: all referral sessions</td><td>1.842</td><td>1.306</td><td>2.599</td><td>&lt; 0.001</td></tr><tr><td>Immediate ratio: engaged referral sessions</td><td>2.388</td><td>1.487</td><td>3.833</td><td>&lt; 0.001</td></tr><tr><td>Final four weeks: fitted ratio</td><td>2.301</td><td>1.342</td><td>4.032</td><td>0.026</td></tr><tr><td>Final four weeks: adjusted observed ratio</td><td>2.055</td><td></td><td></td><td></td></tr><tr><td>Monthly ratio of growth factors</td><td>1.750</td><td></td><td></td><td></td></tr><tr><td>Placebo rank before treatment</td><td>0.150</td><td></td><td></td><td></td></tr></table>

![](images/cb26b6d1a2b8339be0b5c526545cd40d7291a320ec5cdec05b4ad4c4f0e2d0a8.jpg)  
Figure 8: Observed and fitted log ratio of treated to control referrals. The vertical line marks the first post-treatment week, and the dotted path extrapolates the relation estimated before treatment.

The date assigned to the treatment transition materially changes the estimate. Moving the first post-treatment week from December 16, 2025 to January 6, 2026 changes the immediate ratio from 1.183 to 2.404. The intervals for December 16 and December 23 include one, whereas those for December 30 and January 6 exclude it. We report this range because the source material describes a transition window and does not identify one breakpoint.

The source study reports that bot filtering changed in mid-March and that the composition of sessions subsequently changed, with a larger increase in engagement for the treated group (Watanabe & Nakayashiki, 2026). A diferential measurement change can remain in the ratio of treated to control sessions. Because the exact date is unavailable, we use March 10, March

Table 18: Sensitivity to the First Post-Treatment Week
<table><tr><td>First Post-Treatment Week</td><td>Ratio</td><td>Lower</td><td>Upper</td><td>P-value</td></tr><tr><td>2025-12-16</td><td>1.183</td><td>0.665</td><td>2.103</td><td>0.560</td></tr><tr><td>2025-12-23</td><td>1.412</td><td>0.852</td><td>2.341</td><td>0.176</td></tr><tr><td>2025-12-30</td><td>1.842</td><td>1.306</td><td>2.599</td><td>&lt; 0.001</td></tr><tr><td>2026-01-06</td><td>2.404</td><td>1.770</td><td>3.265</td><td> $< 0 . 0 0 1$ </td></tr></table>

17, and March 24 as alternative weekly starts. For each date, we first add a change in level to (51) and then allow both the level and slope to change. We also fit the original specification to the 36 weeks ending before March 10, using the terms described in Appendix I. Each

Table 19: Sensitivity to the Referral Measurement Change
<table><tr><td>Measurement Specification</td><td>Date</td><td>All Sessions</td><td>Engaged Sessions</td></tr><tr><td>No measurement change</td><td></td><td>1.842 [1.306, 2.599]</td><td>2.388 [1.487, 3.833]</td></tr><tr><td>Level change</td><td>2026-03-10</td><td>1.782 [1.233, 2.575]</td><td>2.328 [1.385, 3.914]</td></tr><tr><td>Level and slope change</td><td>2026-03-10</td><td>1.337 [0.937, 1.907]</td><td>1.543 [0.981, 2.427]</td></tr><tr><td>Level change</td><td>2026-03-17</td><td>1.713 [1.191, 2.464]</td><td>2.219 [1.322, 3.725]</td></tr><tr><td>Level and slope change</td><td>2026-03-17</td><td>1.387 [0.977, 1.967]</td><td>1.617 [1.032, 2.534]</td></tr><tr><td>Level change</td><td>2026-03-24</td><td>1.685 [1.174, 2.419]</td><td>2.125 [1.287, 3.508]</td></tr><tr><td>Level and slope change</td><td>2026-03-24</td><td>1.455 [1.025, 2.067]</td><td>1.696 [1.083, 2.655]</td></tr><tr><td>Sample ending before date</td><td>2026-03-10</td><td>1.337 [0.937, 1.906]</td><td>1.543 [0.982, 2.426]</td></tr></table>

entry reports $\exp ( \delta _ { 2 } )$ and its 95% interval. All specifications use Newey–West covariance with four lags, the finite-sample correction, and a t reference distribution. The final row uses 36 observations; all other rows use 47.

When a change in measurement level begins on March 17, the ratio for all sessions is 1.713, with an interval of [1.191, 2.464]. Allowing the slope to change reduces the ratio to 1.387, with an interval of [0.977, 1.967]. Across the three candidate dates, estimates from specifications with changes in both level and slope range from 1.337 to 1.455. The sample ending before March 10 gives 1.337, with an interval of [0.937, 1.906]. Later observations help estimate the post-treatment intercept and trend even though filtering changed after treatment began, so the two estimated changes are statistically dependent. The estimate at rollout is correspondingly sensitive to the specification of the later measurement change.

The week beginning December 30 has the largest Cook’s distance, 0.741, and the largest absolute studentized residual, 3.706. Leave-one-out estimates range from 1.700 to 2.272, while changes in the response specification produce ratios from 1.842 to 2.647. Newey–West lag choices from zero through 12 leave the baseline interval above one. The immediate ratio exceeds one in every reported specification, but whether its interval excludes one depends on the first post-treatment week and on the treatment of the measurement change.

## G Bayesian Computation

The Bayesian computation in Section 3 combines Laplace approximations for the models used to construct the inputs with conditional calculations for the response model. The priors, importance weights, and update from a randomized estimate are specified below.

## G.1 Noncentered Laplace Approximation for the Measurement Models

The hierarchical measurement models are approximated in coordinates that keep the standardized random efects independent of their unknown scale. For a hierarchical logit model, write ${ \pmb u } = B _ { Q } \sigma \widetilde { \pmb u }$ , where $\widetilde { \pmb { u } } \sim \mathcal { N } ( 0 , I _ { Q - 1 } )$ , and assign a normal prior to log σ. We optimize the posterior in $( \alpha , \tilde { { \pmb u } } , \log \sigma )$ coordinates. The analytic Hessian includes the cross derivatives between the fixed efects and the scale of the random efects. After inversion, the delta method maps the covariance to $( \alpha , \pmb { u } )$ coordinates.

The model for occurrence of the target name contains an intercept and a question characteristic. Its other fixed efects represent diferences between generative systems, calendar variation, and the change associated with the GEO source state. Calendar variation consists of a scaled trend and harmonic terms with period 18. The prior standard deviation is 3.0 for the intercept, 1.5 for the question characteristic and system efect, 1.0 for the calendar terms, and 1.5 for the GEO coeficient. The log standard deviation of the question efects has a normal prior with mean log(0.45) and standard deviation 0.55. The notice model omits the calendar and GEO terms, and its log standard deviation has prior mean log(0.35).

## G.2 Posterior for the Response Model

We place a normal inverse-gamma prior on the response coeficients and variance, with nonnegative main efects for GEO and GEM:

$$
\sigma _ { Y } ^ { 2 } \sim \mathrm { I n v e r s e G a m m a } ( 2 . 5 , 2 . 0 ) ,\tag{52}
$$

$$
\beta \mid \sigma _ { Y } ^ { 2 } \sim { \mathcal { N } } ( 0 , \sigma _ { Y } ^ { 2 } V _ { 0 } ) \quad { \mathrm { s u b j e c t ~ t o } } \quad \beta _ { G } \geq 0 , \quad \beta _ { P } \geq 0 ,\tag{53}
$$

where

$$
V _ { 0 } = \mathrm { d i a g } ( 2 . 5 ^ { 2 } , 5 . 0 ^ { 2 } , 4 . 0 ^ { 2 } , 1 . 5 ^ { 2 } ) .\tag{54}
$$

The simulations contain no established media channels. We residualize the response and the four regressors associated with GEO and GEM with respect to W, which is equivalent to assigning a flat prior to the control coeficients. This residualization requires a finite control matrix with full column rank and positive residual degrees of freedom.

For each channel m, the decay parameter is $\alpha _ { m } = 0 . 9 0 U _ { m }$ with $U _ { m } \sim \mathrm { B e t a } ( 2 , 2 )$ . The saturation midpoint is $\theta _ { m } = 0 . 2 5 + 1 . 2 5 V _ { m }$ with $V _ { m } \sim \mathrm { B e t a } ( 2 . 5 , 2 . 5 )$ , and the Hill exponent is $\kappa _ { m } = 0 . 2 5 + 0 . 7 5 R _ { m }$ with $R _ { m } \sim \mathrm { B e t a } ( 3 , 2 )$ . The lag length is $L _ { m } = 8$

For candidate $k ,$ let $\widetilde { Y }$ and $\bar { M _ { k } }$ denote the response and the four target regressors after residualization with respect to $W$ . Before the sign restrictions are imposed, the prior in (53)

yields a normal inverse-gamma posterior whose marginal distribution for the coeficient vector is multivariate Student t. We approximate the posterior probability that $( \beta _ { G } , \beta _ { P } )$ lies in the positive orthant with a Gaussian distribution matched to the posterior mean and covariance. The constrained integrated likelihood equals the unconstrained marginal likelihood multiplied by this posterior probability and divided by the prior probability $1 / 4$

For independent candidates, we first sample $\sigma _ { Y } ^ { 2 }$ from its inverse-gamma posterior and then sample the coeficient vector from the corresponding multivariate normal, retaining vectors that satisfy $\beta _ { G } \geq 0$ and $\beta _ { P } \geq 0$ . In the crossed comparison, the marginal Student distribution is first conditioned on the less probable of the two positivity events. That coeficient is sampled from its truncated univariate Student distribution, and the remaining coeficients are sampled from their conditional Student distribution. Rejection based on the other sign yields the same constrained marginal distribution. Both procedures continue until they obtain the required number of posterior samples; neither uses a fixed proposal limit. The Gaussian orthant approximation afects the candidate weights, whereas the conditional coeficient samples follow the Student distribution. The two-stage benchmark retains the prior for the transformations. The cut and joint procedures use the response data to estimate the transformation parameters, and the plug-in comparisons state whether those parameters are fixed or estimated.

## G.3 Update With a Randomized Estimate

An estimate from a randomized experiment can update the Gaussian approximation to the coeficient posterior when it refers to the same treatment and target population as the model estimand.

For candidate k, let $\mathcal { N } ( \mu _ { k } , \Sigma _ { k } )$ approximate the unconstrained coeficient posterior given the response data. Define the region satisfying the sign restrictions by $C = \{ \beta : \beta _ { G } \geq 0 , \beta _ { P } \geq$ 0} and let $P _ { k }$ be its probability under this approximation. Let $M _ { i t , k } ( a _ { G } , a _ { P } )$ denote the four regressors, including the indicator for the source state, constructed under the treatment sequences indexed by $( a _ { G } , a _ { P } )$ . With $N _ { E } = n | \mathcal { T } _ { E } |$ , the average diference between the design vectors is

$$
d _ { k } = \frac { 1 } { N _ { E } } \sum _ { i = 1 } ^ { n } \sum _ { t \in \mathcal { T } _ { E } } \left( M _ { i t , k } ( 1 , 1 ) - M _ { i t , k } ( 0 , 1 ) \right) ,\tag{55}
$$

$$
\tau _ { E , k } = d _ { k } ^ { \mathsf { T } } \beta .\tag{56}
$$

Each candidate supplies its own design vector, including the component associated with the source state. Write $v _ { E } = s _ { E } ^ { 2 } + \sigma _ { \mathrm { t r } } ^ { 2 }$ and $s _ { k } = v _ { E } + d _ { k } ^ { \mathsf { T } } \Sigma _ { k } d _ { k }$ . The unconstrained Gaussian update is

$$
\mu _ { k } ^ { + } = \mu _ { k } + \frac { \Sigma _ { k } d _ { k } } { s _ { k } } \left( \widehat { \tau } _ { E } - d _ { k } ^ { \mathsf { T } } \mu _ { k } \right) ,\tag{57}
$$

$$
\Sigma _ { k } ^ { + } = \Sigma _ { k } - \frac { \Sigma _ { k } d _ { k } d _ { k } ^ { \mathsf { T } } \Sigma _ { k } } { s _ { k } } .\tag{58}
$$

Let $P _ { k } ^ { + }$ be the probability of C under $\mathcal { N } ( \mu _ { k } ^ { + } , \Sigma _ { k } ^ { + } )$ . If $\varphi ( x ; m , v )$ denotes the normal density with mean m and variance v, the predictive factor under the sign restrictions and the updated

candidate weight are

$$
\ell _ { E , k } = \varphi \left( \widehat { \tau } _ { E } ; d _ { k } ^ { \mathsf { T } } \mu _ { k } , s _ { k } \right) \frac { P _ { k } ^ { + } } { P _ { k } } ,\tag{59}
$$

$$
w _ { k } ^ { + } = \frac { w _ { k } \ell _ { E , k } } { \sum _ { h } w _ { h } \ell _ { E , h } } .\tag{60}
$$

The ratio $P _ { k } ^ { + } / P _ { k }$ follows by integrating the product of the Gaussian likelihood and the truncated Gaussian distribution over C. After a candidate is selected with probability $w _ { k } ^ { + }$ the coeficients are sampled from $\mathcal { N } ( \mu _ { k } ^ { + } , \Sigma _ { k } ^ { + } )$ conditional on C. The weight calculation and the conditional coeficient distribution use the same Gaussian approximation and sign restrictions.

## G.4 Model for Sponsored Placements

The prior for the model of sponsored placements assigns standard deviation 3.0 to the intercept and 1.5 to the elasticity with respect to spending. The demand and promotion coeficients each have standard deviation 1.0. Optimization imposes $\phi _ { S } > 0$ , and Gaussian proposal samples that violate this restriction are discarded. Sponsored notice has a Beta(1, 1) prior, whose posterior mean supplies the plug-in value.

## H Simulation Models

The simulation designs vary the source of measurement uncertainty, the response equation, and the treatment sequence. The model and parameter values below define the principal comparisons and the additional data-generating processes.

## H.1 Main GMMM Simulation

In the controlled design, the true transformation parameters are sampled from the same distributions used in estimation. The resulting comparison is favorable to that prior specification. Another design fixes the transformation parameters away from the prior centers while keeping them within the prior support.

The panel contains five markets observed for 36 periods. Each market has six product clusters linked to 12 question clusters on two generative systems. Two product clusters first receive GEO in period 19, two first receive GEO in period 23, and two remain untreated. Periods are indexed by $t \in \{ 1 , \ldots , 3 6 \}$ . The log-odds coeficient of the GEO source state in the occurrence model is 0.72, and the standard deviation of the question efects is 0.45. The remaining variation reflects the generative system and calendar time: the coeficient for the second system is 0.24, and the calendar terms combine a linear trend with periodic variation. Question counts follow a log-normal distribution driven by latent demand and seasonality, while the shares of use across systems follow a symmetric Dirichlet distribution.

The design based on the collected answers uses (33). It samples one question index for each pair of models and gives equal representation to English and Japanese. With P generative systems, the fitted occurrence model contains $P ( Q - 1 )$ centered question efects. Each system has its own vector of efects, and the vectors share one scale parameter. This likelihood reproduces the estimated baseline probabilities while retaining the same coeficient for the GEO source state.

The probability of positive spending and the positive amount spent depend on promotion, market and product-cluster efects, and demand measured before treatment. The coeficient vector in the model for sponsored placements is (2.10, 0.82, 0.20, 0.14), and the notice probability for those placements is 0.57. The two generated channels use transformation parameters sampled from the distributions that generate the estimation candidates. The response errors have standard deviation 1.20. Controls consist of additive market and product-cluster efects, a linear time trend, two seasonal terms, and the two predictors observed before treatment.

Each conditional response posterior contains 1,000 samples. The independent randomized experiment has 400 equally allocated units and a response standard deviation of 3.5. The treated mean difers from the control mean by the true average treatment efect of GEO over the panel, $\Delta _ { G } / 1 0 8 0$ . Inference constructs $d _ { k }$ using (56). Its first coordinate is 320/1080, while the coordinate for the main GEM efect is zero because GEM spending is held fixed. A second version adds 0.75 to the same true efect. Both analyses use the specified transport standard deviation 0.20. The estimator continues to use the specified transport standard deviation without an additional mean shift, so the added 0.75 becomes an unmodeled diference between the experimental and target populations.

## H.2 Alternative Data-Generating Processes

Five designs alter either the measurement model, the response equation, or both while retaining the panel dimensions and treatment sequences. The numbers of importance candidates and conditional posterior samples are also unchanged.

Two designs change the distribution of the measurement data. Occurrence overdispersion first samples a cell probability from a Beta distribution with concentration 12 and then generates repeated Bernoulli indicators. Overdispersion in sponsored placements uses a gamma–Poisson mixture with dispersion 4. Two other designs change the response equation. The first uses $( \alpha , L , \theta , \kappa ) = ( 0 . 8 4 , 8 , 1 . 4 2 , 0 . 9 2 )$ for GEO and (0.78, 8, 1.32, 0.90) for GEM. The second adds $0 . 8 0 ( H ^ { G } ) ^ { 2 }$ to both the response and the true treatment efect of GEO. The fifth design combines all four departures. Each diference is Joint GMMM minus the plug-in

Table 20: Combined Departure in the Measurement and Response Models
<table><tr><td>Method</td><td>Effect Bias (%)</td><td>Effect RMSE (%)</td><td>Coverage</td><td>Vector RMSE</td><td>ESS</td><td>ESS q0.10</td></tr><tr><td>Plug-in GMMM</td><td>-14.835</td><td>22.738</td><td>0.783</td><td>2.014</td><td></td><td></td></tr><tr><td>Two-stage GMMM</td><td>-14.082</td><td>22.110</td><td>0.850</td><td>1.501</td><td></td><td></td></tr><tr><td>Joint GMMM</td><td>-4.683</td><td>18.059</td><td>0.917</td><td>1.080</td><td>25.947</td><td>8.616</td></tr><tr><td>Oracle inputs</td><td>-1.186</td><td>16.988</td><td>0.917</td><td>0.775</td><td></td><td></td></tr></table>

estimator that also estimates the transformations. The bounds are 95% paired bootstrap intervals from 5,000 resamples of the 120 common panels within each design. Diferences in efect RMSE are measured in percentage points. A positive value favors the plug-in estimator.

Table 21: Paired Diferences When Both Methods Estimate the Transformations
<table><tr><td>Design</td><td>Metric</td><td>Difference</td><td>Lower</td><td>Upper</td></tr><tr><td>Estimated occurrence probabilities</td><td>Vector RMSE</td><td>0.0133</td><td>0.0008</td><td>0.0262</td></tr><tr><td>Estimated occurrence probabilities</td><td>Effect RMSE (pp)</td><td>0.159</td><td>-0.012</td><td>0.330</td></tr><tr><td>Shifted transformations</td><td>Vector RMSE</td><td>0.0095</td><td>-0.0004</td><td>0.0194</td></tr><tr><td>Shifted transformations</td><td>Effect RMSE (pp)</td><td>-0.002</td><td>-0.165</td><td>0.150</td></tr><tr><td>Combined misspecification</td><td>Vector RMSE</td><td>0.0096</td><td>-0.0006</td><td>0.0196</td></tr><tr><td>Combined misspecification</td><td>Effect RMSE (pp)</td><td>0.012</td><td>-0.133</td><td>0.157</td></tr></table>

## H.3 Simulation With Platform Growth

This simulation contains 80 clusters observed for 48 periods on three platforms. Half of the clusters receive GEO beginning in period 24. Each platform has a common time path that combines deterministic growth, seasonal variation, and a random walk, while the log-odds coeficient of GEO is 0.65. Across 500 replications, we compare diference-in-diferences with the change before and after treatment among treated units and with the diference between treated and control units after treatment.

## H.4 Simulation With a Post-Treatment Variable

A continuous randomized encouragement changes the constructed GEO input, which afects the response directly and through brand search. The direct efect is 0.70, and the efect through brand search is 0.88, giving a total efect of 1.58. Across 800 replications, we compare a response regression on the treatment with a regression that also conditions on brand search.

## I Computation for the Referral Analysis

The referral analysis uses segmented regression and then varies the treatment date, trend specification, lag length, and influential observations. The details below define those calculations.

The analysis uses 47 weekly observations and the regression with four parameters in (51). Inference uses Newey–West covariance with four lags. The date analysis moves the first post-treatment week from two weeks before the central date to one week after it, while the lag analysis considers every integer from zero through 12. Other checks alter one part of the response specification at a time. They remove the first post-treatment week, shorten the period before treatment, change the baseline trend, or omit the common linear trend. Influence is assessed through 47 leave-one-out regressions, together with studentized residuals and Cook’s distance.

For a candidate week $t _ { c }$ at which measurement changes, the level specification adds $\zeta _ { 0 } \mathbf { 1 } ( t \geq t _ { c } )$ to (51). The specification allowing both level and slope changes also adds $\zeta _ { 1 } ( t - t _ { c } ) \mathbf { 1 } ( t \geq t _ { c } )$ . These terms represent diferential measurement changes that remain in the log ratio of treated to control sessions. Holding the first post-treatment week at December 30, we fit each candidate $t _ { c }$ to all sessions and to engaged sessions. The truncated specification retains the original four regressors and uses observations strictly before March 10. Every reported ratio exponentiates the treatment coeficient $\delta _ { 2 }$ , while the later measurement terms account for the subsequent change. With n observations and $p$ regressors, the Newey–West covariance uses the correction $n / ( n - p )$ , and the intervals use $n - p$ degrees of freedom.

The fitted ratio over the final four weeks is computed from the segmented regression. Its interval uses 5,000 residual moving-block bootstrap replications. Residuals are centered within the periods before and after treatment, sampled in circular blocks of four weeks within each segment, and added to the fitted values before the regression is refitted. The reported 95% interval contains the 2.5th and 97.5th percentiles of the resulting ratios. This interval is distinct from the Newey–West t test for the corresponding linear coeficient.

The placebo analysis fits the same segmented regression at every split in the pre-treatment period that leaves at least four observations on each side, including both boundary splits. Let B be the number of admissible splits, and let b count the splits whose absolute estimated level change is at least as large as the estimate at treatment. The reported rank is $( b + 1 ) / ( B + 1 )$ where the added observation is the estimate at treatment itself. Because the treatment date was not randomly selected from these dates, the rank is a descriptive diagnostic and cannot be interpreted as a p value from a randomization test.

Table 22: Sensitivity to the Response Specification
<table><tr><td>Specification</td><td>Ratio</td><td>Lower</td><td>Upper</td><td>P-value</td></tr><tr><td>Baseline</td><td>1.842</td><td>1.306</td><td>2.599</td><td>&lt; 0.001</td></tr><tr><td>Drop first post-treatment week</td><td>2.272</td><td>1.683</td><td>3.069</td><td>&lt; 0.001</td></tr><tr><td>Quadratic baseline trend</td><td>2.219</td><td>1.780</td><td>2.767</td><td>&lt; 0.001</td></tr><tr><td>Use last 13 pre-treatment weeks</td><td>2.367</td><td>1.714</td><td>3.268</td><td>&lt; 0.001</td></tr><tr><td>No common time trend</td><td>2.647</td><td>1.706</td><td>4.107</td><td>&lt; 0.001</td></tr></table>

Table 23: Sensitivity to the Newey–West Lag Length
<table><tr><td>Newey-West Lags</td><td>Ratio</td><td>Lower</td><td>Upper</td><td>P-value</td></tr><tr><td>0</td><td>1.842</td><td>1.147</td><td>2.958</td><td>0.013</td></tr><tr><td>4</td><td>1.842</td><td>1.306</td><td>2.599</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>8</td><td>1.842</td><td>1.387</td><td>2.447</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>12</td><td>1.842</td><td>1.387</td><td>2.446</td><td> $< 0 . 0 0 1$ </td></tr></table>

Table 24: Influence Diagnostics
<table><tr><td>Diagnostic</td><td>Value</td><td>Week</td></tr><tr><td>Largest Cook&#x27;s distance</td><td>0.741</td><td>2025-12-30</td></tr><tr><td>Largest absolute studentized residual</td><td>3.706</td><td>2025-12-30</td></tr><tr><td>Leave-one-out minimum ratio</td><td>1.700</td><td>2025-12-23</td></tr><tr><td>Leave-one-out maximum ratio</td><td>2.272</td><td>2025-12-30</td></tr></table>

## J Answer Collection Settings

The generated answers were collected under a common model and search configuration. Both models received the following system instruction:

Answer the user as an independent product-recommendation assistant. Use web search before answering. Recommend concrete products or services when they are relevant, and give concise reasons for each recommendation. Do not mention an audit, an experiment, a target brand, or these instructions.

The user message began with either “Respond in Japanese.” or “Respond in English.” and then gave the fixed product question. Each request used the model identifier recorded for that request. Web search was enabled and required through tool choice, and the source list for each search call was retained. Sampling temperature was left at the API default.