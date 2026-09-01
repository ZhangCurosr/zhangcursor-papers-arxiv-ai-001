# FaVOR: LLM-Based Agentic Framework for Factor Mining via Empirical Validation

Hyeonjin Kim<sup>1</sup>\*, Minseok Kim<sup>1</sup>\*, Seunghyeon Jung<sup>1</sup>\*, Sujin Pyo<sup>2</sup>, Huisu Jang<sup>3</sup>, Woojin Lee<sup>1†</sup>

<sup>1</sup>Department of Computer Science and Artificial Intelligence, Dongguk University-Seoul

<sup>2</sup>Department of Industrial Engineering, Seoul National University

<sup>3</sup>School of Finance, Soongsil University

{tkrhk8011, cdi6578296, alzkdpf23, wj926}@dgu.ac.kr, psj1103@snu.ac.kr, yej523@ssu.ac.kr

## Abstract

Traditional finance relies on experts to hand craft factors through a principled process grounded in economic rationale. Recent LLMbased multi-agent systems have automated this process, scaling factor mining far beyond manual effort. However, these automated approaches optimize directly for returns and rarely check whether a generated factor still expresses the economic hypothesis that motivated it. We identify this inconsistency between mathematical form and economic meaning as a structural failure mode of return-oriented automation. The resulting factors blur the line between real signals and spurious correlations and break down across regime shifts. We propose Fa-VOR (Factor Validation through Observable Reasoning), an agentic framework that restruc tures factor mining around hypothesis-level evidence rather than return outcomes. In place of the standard hypothesis-to-formula leap, Fa VOR enforces a three-stage consistency loop tying mathematical form to economic rationale throughout. (1) Decomposition splits a broad economic hypothesis into independent observable conditions. (2) Validation checks whether each factor reflects its intended condition. (3) Integration merges them into a composite whose structure remains interpretable. On the CSI 500 and S&P 500 in 2025, FaVOR outperforms existing baselines while remaining effective across regimes. FaVOR shows that hypothesis-grounded factor discovery produces signals that are interpretable by construction, regime-robust, and economically faithful. The code is available at https://github.com/d amilab/FaVOR.

## 1 Introduction

Large language models (LLMs) are increasingly being applied to financial analysis and investment decision-making (Kim et al., 2026). Within this broader trend, multi-agent systems built on LLMs are reshaping quantitative investment research (Lopez-Lira, 2025; Cheng and Chin, 2024; Zhang et al., 2024). These systems convert qualitative investment ideas into economic hypotheses and then translate those hypotheses into computable factors for investment decisions (Wang et al., 2025; Li et al., 2025; Tang et al., 2025). Compared with traditional factor design, LLM-based frameworks can scale factor mining beyond manual effort and report excess returns over benchmark indices (Duan et al., 2025; Luo et al., 2025a).

Traditional factor generation begins with market observation. Researchers first identify a market state, an economic condition reflected in price and trading activity (Karpoff, 1987; Cochrane, 2009). From these observations, they formulate an economic hypothesis that links excess returns to structural risks, market frictions, or temporary imbalances. The hypothesis becomes a factor only after rigorous empirical validation (Fama and French, 1993). A factor in this tradition is not merely a predictive formula, but a quantitative measurement of a clearly defined economic mechanism in historical data (Cochrane, 2011; Fama and French, 1996).

Current LLM-driven approaches largely optimize factors for predictive returns, but they provide limited evidence that a factor’s mathematical form reflects its intended economic mechanism (Bailey et al., 2014; McLean and Pontiff, 2016; Lopez de Prado and Zoonekynd, 2025). Without such validation, formulas that merely fit opaque correlations in the training window can appear indistinguishable from factors built on genuine economic mechanisms. Such formulas look strong in-sample but tend to break down when market regimes shift.

Robustness across regimes therefore requires a structurally interpretablefactor whose form maps back to the hypothesis that produced it (Kou et al., 2024). For LLM-based pipelines, verifying this mapping is difficult because the model returns a complete formula without exposing the intermediate economic reasoning that produced it. An explicit checking step is needed to separate factors that reflect the intended economic signal from those that overfit to noise (Jiang et al., 2025).

![](images/254f02e830097bc7d061139fdb93fde21720c164cfee8be1ea909c70123b6eeb.jpg)  
Figure 1: Comparison of three approaches to factor discovery and validation. Traditional factor design relies on expert hypotheses and manual construction, supporting interpretability but limiting scalability. LLM-based automated mining scales generation but lacks pre-backtest validation that a factor captures its intended economic mechanism. FaVOR decomposes hypotheses into observable market conditions and validates candidates with distributional evidence before backtesting, combining scalability with factor-level validation.

We propose FaVOR (Factor Validation through Observable Reasoning), an agentic framework that evaluates factor consistency through a three-stage process. Decomposition breaks a hypothesis into observable market conditions. Validation checks whether each generated factor reflects its intended condition. Integration identifies the combination of validated factors that best reconstructs the original hypothesis. By admitting only factors that pass all three stages, FaVOR builds structurally interpretable factors that combine empirical performance with economic rationale.

On the Chinese CSI 500 and U.S. S&P 500 in the 2025 out-of-sample testing window, FaVOR delivers consistent performance across both markets. After transaction costs, the framework achieves a cumulative excess return of 0.2225 with IR of 1.5295 on the CSI 500 and 0.1123 with IR of 1.1315 on the S&P 500. These results suggest that hypothesiscentered empirical validation can improve the economic reliability of automated factor mining.

## 2 Related Work

Traditional Foundations of Factor Investing. Traditional quantitative research treats a factor as a measure of a specific economic mechanism rather than a formula optimized only for returns (Cochrane, 2009; Bender et al., 2013). This framing rests on the principle of informational efficiency, under which latent market states are reflected in prices and trading activity (Fama, 1970). Researchers therefore use empirical validation to test whether a market signal aligns with an explicit economic hypothesis (Hou et al., 2015; Banz, 1981; Sloan, 1996; Baker et al., 2011).

Prior studies show how this logic appears in factor research. Basu (1977) links valuation ratios to expected returns, while Jegadeesh and Titman (1993) studies momentum as a return pattern grounded in market behavior. Models with multiple factors extend this idea by combining observable firm characteristics and market indicators. Fama and French (1993, 2015) integrate market risk, size, value, profitability, and investment patterns to explain the cross section of expected stock returns, and Kakushadze (2016) shows that combining many independent alpha formulas from price, volume, and fundamentals can improve portfolio diversification and return stability. These factors remain economically interpretable, but the process depends on manual testing by domain experts and does not scale to a large candidate space.

Automated Factor Mining. Recent LLM-based approaches automate factor generation by using the financial knowledge encoded in the models and their ability to enumerate candidates. FAMA (Li et al., 2024) introduces sample based selection and an experience based mechanism to reduce redundancy. Cheng and Tang (2024) show that GPT-4 can synthesize useful factors through financial reasoning alone. Frameworks with multiple agents push the idea further by coordinating specialized agents in a collaborative search and optimization pipeline. Alpha-GPT (Wang et al., 2025) translates researchers’ ideas into formulas and optimizes them algorithmically. RD-Agent-Quant (Li et al., 2025) jointly optimizes factors and machine learning models across the workflow from hypothesis to code. AlphaAgent (Tang et al., 2025) adds regularized exploration to promote hypothesis consistency and resist alpha decay.

![](images/642f1fe8615f6e7c1f30382b9d57ae7f8b75b574de29e4b60fa55ffa609b1dc2.jpg)  
Figure 2: Overview of the FaVOR framework: a three-stage pipeline for hypothesis-driven factor mining. Stage 1 decomposes a high-level hypothesis into observable market conditions and generates candidate factors via the Hypothesis, Observation, and Factor agents. Stage 2 validates each candidate by constructing factor-conditioned market-state statistics, with pass/fail gates that route rejected factors back to the Factor agent. Stage 3 integrates the validated factors under a joint threshold and runs threshold optimization and out-of-sample backtesting to produce stable, hypothesis-consistent trading signals.

Despite these advances, agents based on LLMs still evaluate candidate factors mainly through semantic fit or realized performance. Recent verification mechanisms in this line of work (Wang et al., 2024; Luo et al., 2025b; Lee et al., 2025; Ding et al., 2025) provide useful checks, but natural language explanations alone do not show that a factor reflects an underlying market state. This leaves a gap between automated factor generation and the empirical validation that supports interpretability and reliability in traditional finance (Kang and Liu, 2023; Tatsat and Shater, 2025).

## 3 Methods

We propose FaVOR, an agentic framework that constructs structurally interpretable factors from economic hypotheses. The framework does not evaluate a hypothesis solely by its final return. Instead, it first decomposes the hypothesis into observable market conditions, then checks whether candidate factors measure those conditions, and finally integrates the validated factors into executable trading signals. Figure 2 summarizes this threestage pipeline.

## 3.1 Stage 1 : Hypothesis Decomposition

Stage 1 turns a high-level economic hypothesis into measurable components. FaVOR first generates the hypothesis from market insights, then decomposes it into observable market conditions, and finally generates candidate factors that give each condition a quantitative form.

## 3.1.1 Hypothesis Generation

The hypothesis agent $\boldsymbol { \mathcal { A } } _ { H }$ takes human market insights I and returns an economic hypothesis $\mathcal { A } _ { H } ( I ) = h$ . The hypothesis h is a descriptive statement about market states and their expected impact on price movements.

## 3.1.2 Hypothesis Decomposition

The hypothesis h is written in natural language, so it cannot be validated directly against raw market data. The observation agent $A _ { O }$ decomposes h into m observable market conditions, $\mathcal { A } _ { O } ( h ) = \{ o _ { i }$ $i = 1 , \ldots , m \}$ , where each $o _ { i }$ captures a distinct and independent aspect of the hypothesis.

## 3.1.3 Candidate Factor Generation

Given an observable market condition $o _ { i } .$ , the factor agent $\boldsymbol { A _ { F } }$ generates $n _ { i }$ candidate factors, $\mathcal { A } _ { F } ( o _ { i } ) = F _ { i } = \{ f _ { i , j } \} _ { j = 1 } ^ { n _ { i } }$ . Each candidate factor $f _ { i , j }$ is a formula computed from market data and intended to measure $o _ { i } .$ . To keep the formulas interpretable and executable, $\boldsymbol { \mathcal { A } } _ { F }$ draws operators from the predefined set in Appendix A. The resulting candidate factors enter Stage 2 for empirical consistency checks.

## 3.2 Stage 2 : Factor-Level Validation

This stage evaluates the factor consistency of the candidate factors from Stage 1. Factor-level validation verifies whether each formula embodies the economic rationale of its corresponding observable market condition. It uses market variables observed at time t rather than future returns at t+T, where T denotes the holding horizon of the trading strategy. We use an LLM agent because the same statistical pattern can support one rationale while contradicting another. Consistency must therefore be judged against the hypothesis context.

![](images/15637237c4c66ec8ec3502b617f1c94d9aa1abcc2be548b5789ca17d45bdd20f.jpg)  
Figure 3: Illustration of factor-level validation using quantile-based distributional evidence. A constructaligned factor (Factor A) shows systematic distributional shifts across ordered quantile bins that match the corresponding observable market condition. It therefore passes validation. In contrast, a misaligned factor (Factor B) shows unstable or contradictory patterns across bins, leading to validation failure.

## 3.2.1 Historical Evidence for Consistency Evaluation

The validation agent $\mathbf { \mathcal { A } } _ { V }$ checks whether a candidate factor measures the observable market condition it was generated for. The goal is to test whether stronger factor values correspond to stronger evidence of the market state described in $o _ { i } .$ , not whether the factor predicts future returns.

For each candidate factor, we compute factor values on the training data, sort stocks within each date, and split them into quintile bins $Q _ { 1 } , \ldots , Q _ { 5 }$ ordered from low to high. To describe the market state in each bin, we use four OHLCV-based variables that capture intraday return $( C - O )$ , daily price range $( H - L )$ , close position within the daily range $( ( C - L ) / ( H - L ) )$ , and trading volume $( V )$ We denote this set by X. For each bin and each $x \in \mathcal { X }$ , we compute standard descriptive statistics (mean, median, ${ \mathfrak { q } } 1 0 , { \mathfrak { q } } 9 0$ , and dispersion measures), and collect them into the statistical profile $S _ { i , j }$ that $\mathbf { \mathcal { A } } _ { V }$ uses to assess factor consistency.

3.2.2 Factor Consistency Evaluation Criteria Given the profile $S _ { i , j }$ , the validation agent $\mathbf { \mathcal { A } } _ { V }$ decides whether the empirical patterns are consistent with the intended economic rationale $o _ { i } .$ . Each check applies fixed rules to the statistics in $S _ { i , j }$ The LLM is used only to map the textual observation $o _ { i }$ to the relevant variables in X and their expected directions. Given this mapping, the final pass/fail decision is deterministic because it is the conjunction of four explicit rule checks.

We apply four consistency checks: (i) a systematic shift in central tendency of some $x \in \mathcal { X }$ across the ordered factor bins, (ii) non-trivial variation in the tail spread of x across bins, (iii) agreement across at least two complementary statistics of the same x, and (iv) semantic agreement with $o _ { i }$ . The precise definitions and decision rule are deferred to Appendix B.

The validated factor set for each condition is $\hat { F } _ { i } ~ = ~ \{ f _ { i , j } ~ \in ~ F _ { i } ~ \mid ~ { \mathcal { A } } _ { V } ( o _ { i } , f _ { i , j } , S _ { i , j } ) ~ = ~ { \mathsf { p a s s } } \}$ Once $\hat { F } _ { i }$ is built for every condition, the pipeline moves to factor integration.

## 3.3 Stage 3: Factor Integration

This stage shifts the unit of analysis from a single factor to a combination of validated factors, and asks whether such a combination realizes the original hypothesis h. We test this by examining how the combination’s joint trading signal behaves as the entry rule becomes more selective.

## 3.3.1 Candidate Combinations

For each observable condition $o _ { i }$ , let $\hat { F } _ { i }$ denote the set of validated factors from Stage 2, which typically contains two or three formulas. We form candidate combinations by taking the Cartesian product $\mathcal { C } = \hat { F } _ { 1 } \times \hat { F } _ { 2 } \times \cdot \cdot \cdot \times \hat { F } _ { m }$ with cardinality $\begin{array} { r } { | \mathcal { C } | = \prod _ { i = 1 } ^ { m } | \hat { F } _ { i } | } \end{array}$ , and each $\mathbf { f } \in { \mathcal { C } }$ selects one validated factor per observable condition.

## 3.3.2 Directional Selectivity

Each combination $\mathbf { f } \in { \mathcal { C } }$ needs an entry rule that we can vary in selectivity, so we can test whether the direction of its realized return remains stable as the rule tightens. We call this property Directional Selectivity. The entry rule uses a common selectivity level $\sigma \in ( 0 , 1 )$ that determines the cross-sectional extremeness required from each factor. For each $f _ { i , j }$ in the combination, an observation signal triggers when $f _ { i , j }$ crosses its in-sample σ-quantile in the direction implied by $o _ { i }$ (its polarity). A trading signal $\mathrm { S i g n a l } ( \mathbf { f } , \sigma ) = 1$ is triggered only when every constituent factor of f triggers its observation signal simultaneously, and $\mathrm { S i g n a l } ( \mathbf { f } , \sigma ) = 0$ otherwise.

Increasing $\sigma$ tightens each factor’s quantile cutoff and makes the joint signal more selective. It reduces the number of trading opportunities and keeps only stronger realizations of the hypothesized market state.

## 3.3.3 Evaluation Criteria

We evaluate Directional Selectivity by tracking how signal outcomes change across a ladder of selectivity levels $\sigma _ { 1 } < \sigma _ { 2 } < \cdots$ . For a factor combination $\mathbf { f } ,$ we divide its triggered signals at each level into winning signals S<sub>W</sub> with positive realized returns and losing signals $S _ { L }$ with negative realized returns. The average realized return is higher under stricter $\sigma _ { \mathrm { { : } } }$ confirming that selectivity targets more profitable market states. The win rate $| S _ { W } | / ( | S _ { W } | + | S _ { L } | )$ is higher under stricter $\sigma _ { \mathrm { { : } } }$ , indicating fewer false positives as the signal strengthens.

We apply these criteria at the ticker level. A ticker supports the combination only when at least one of these behaviors holds across the ladder. We then retain a factor combination only when its ticker-level support rate exceeds a fixed passrate threshold, and discard combinations below this threshold because stricter entry conditions do not improve the quality of their signals.

## 3.4 Optimization and Backtesting

In Section 3.3, Stage 3 selects factor combinations. We turn these into executable trading signals and evaluate their realized performance. The selectivity sweep in Stage 3 is used for structural screening on the training set, while the final execution thresholds are optimized on a held-out validation set defined in Section 4.1. Because each factor in a combination targets a distinct observable condition, we treat its activation cutoff as independent and optimize factor-specific thresholds for each validated combination with Optuna (Akiba et al., 2019).

The optimization objective is the Calmar ratio (Eling and Schuhmacher, 2007). It is defined as $\mathrm { C a l m a r \ = \ A R / | M D D | }$ , where AR denotes the annualized return and MDD the maximum drawdown. This objective accounts for both return and downside risk. The selected thresholds are then fixed and backtested on the held-out test period.

Evaluation outcomes are used only in the outerloop hypothesis memory. After each iteration, Fa-VOR records prior hypothesis IDs, observation descriptions, formula names, and a single scalar summary of average information ratio. The memory does not store raw returns, formula definitions, or fitted thresholds, and the pipeline code and hyperparameters are fixed before held-out evaluation. Feedback accumulates as hypothesis memory across iterations.

## 4 Experiments

## 4.1 Experiment Settings

Metrics. We evaluate portfolio performance with four metrics. Annualized Return (AR) measures the average yearly excess return over the benchmark. Information Ratio (IR) measures excess return scaled by tracking risk. Cumulative Return (CR) measures the total excess return over the full evaluation period. Maximum Drawdown (MDD) measures the largest decline in the portfolio’s own equity curve from a previous peak. Appendix C provides the mathematical definitions.

Datasets. We conduct backtests with Qlib (Yang et al., 2020) on two equity universes. The CSI 500 covers the Chinese A-share market, and the S&P 500 covers the U.S. equity market. Raw CSI 500 data are obtained from Baostock, and S&P 500 data are obtained from Yahoo Finance.

We split the data into training (2022-01-01 to 2023-12-31), validation (2024-01-01 to 2024-12- 31), and testing (2025-01-01 to 2025-12-31) periods. Factor validation and integration are performed on the training set, threshold optimization is performed on the validation set, and the testing period is reserved exclusively for out-of-sample evaluation. Backtests use only the daily OHLCV fields \$open, \$high, \$low, \$close, and \$volume.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Market Method</td><td colspan="4">CSI 500</td><td colspan="4">S&amp;P 500</td></tr><tr><td>AR</td><td>IR</td><td>MDD</td><td>CR</td><td>AR</td><td>IR</td><td>MDD</td><td>CR</td></tr><tr><td>ML/DL</td><td>Linear (Pedregosa et al., 2011)</td><td>-0.0702</td><td>-0.5008</td><td>-0.1649</td><td>-0.0778</td><td>-0.0010</td><td>-0.0104</td><td>-0.1403</td><td>-0.0055</td></tr><tr><td></td><td>XGBoost (Chen, 2016)</td><td>-0.0123</td><td>-0.1642</td><td>-0.1370</td><td>-0.0152</td><td>-0.0357</td><td>-0.3345</td><td>-0.2326</td><td>-0.0426</td></tr><tr><td></td><td>LightGBM (Ke et al., 2017)</td><td>-0.1576</td><td>-1.3954</td><td>-0.2173</td><td>-0.1530</td><td>-0.0302</td><td>-0.3389</td><td>-0.2154</td><td>-0.0353</td></tr><tr><td></td><td>MLP (Rumelhart et al., 1986)</td><td>-0.0493</td><td>-0.4812</td><td>-0.1725</td><td>-0.0537</td><td>-0.0895</td><td>-0.8041</td><td>-0.1686</td><td>-0.0956</td></tr><tr><td></td><td>Transformer (Vaswani et al., 2017)</td><td>-0.0521</td><td>-0.3943</td><td>-0.1760</td><td>-0.0597</td><td>-0.1744</td><td>-1.5509</td><td>-0.1860</td><td>-0.1729</td></tr><tr><td>RL</td><td>AlphaForge (Shi et al., 2025)</td><td>-0.0133</td><td>-0.1255</td><td>-0.1260</td><td>-0.0167</td><td>-0.0833</td><td>-0.8617</td><td>-0.2093</td><td>-0.0819</td></tr><tr><td></td><td>AlphaQCM (Zhu and Zhu, 2025)</td><td>0.0869</td><td>1.0604</td><td>-0.1396</td><td>0.0803</td><td>0.0619</td><td>0.8250</td><td>-0.2345</td><td>0.0583</td></tr><tr><td>LLM-based</td><td>RD-Agent (Li et al., 2025)</td><td>0.1382</td><td>1.3360</td><td>-0.1427</td><td>0.1452</td><td>0.0604</td><td>0.8028</td><td>-0.1704</td><td>0.0624</td></tr><tr><td></td><td>AlphaAgent (Tang et al., 2025)</td><td>-0.0192</td><td>-0.1490</td><td>-0.1369</td><td>-0.0275</td><td>-0.0060</td><td>-0.0631</td><td>-0.1428</td><td>-0.0110</td></tr><tr><td></td><td>FaVOR</td><td>0.2067</td><td>1.5295</td><td>-0.0853</td><td>0.2225</td><td>0.1062</td><td>1.1315</td><td>-0.0443</td><td>0.1123</td></tr></table>

Table 1: Main results on CSI 500 and S&P 500 excess returns using train, validation, and test splits within 2022– 2025. Bold marks the best result and underline marks the second best.

![](images/3c97c66ea6d3bc23003004dc4283dab634abf4d95d7ce1e789d266079443a310.jpg)

![](images/ef8ad81fcd13978b36e724f788cc5a870268c947aac18bc4185ecacb9c7cdec1.jpg)  
Figure 4: Comparison of cumulative excess returns of various baselines on the CSI 500 (left) and S&P 500 (right).

Backtest Settings. After the trading signals and thresholds are fixed, they are applied unchanged throughout the testing period. A signal is triggered at time t when the factor combination indicates that the hypothesized market state holds. Positions are opened according to the predefined execution protocol and closed according to predefined exit rules, such as the holding horizon T or stop conditions. All backtest results include proportional transaction costs per traded value. For CSI 500, the cost rates are 0.0005 for buys and 0.0015 for sells. For S&P 500, the cost rate is 0.0005 for sells only. Full hyperparameter values used across all stages are listed in Appendix D.

Agent Settings. FaVOR uses four LLM-based agents, and we adopt GPT-4o (Hurst et al., 2024) as the default backbone for all main-table results. Decoding temperatures are set higher for the generative agents and low for the validation agent, balancing idea diversity against verdict stability. Exact values are listed in Appendix D. The prompt templates for all four agents are provided in Ap-

pendix E.

Baselines. We compare FaVOR against baselines that span the paradigms most directly comparable to its hypothesis-driven, validation-centric design. (i) ML/DL methods: Linear (Pedregosa et al., 2011), XGBoost (Chen, 2016), LightGBM (Ke et al., 2017), MLP (Rumelhart et al., 1986), and Transformer (Vaswani et al., 2017). (ii) Reinforcement Learning methods, which optimize a returndriven reward signal under a different search formulation: AlphaForge (Shi et al., 2025) and AlphaQCM (Zhu and Zhu, 2025). (iii) LLM-based methods: R&D-Agent-Quant (Li et al., 2025) and AlphaAgent (Tang et al., 2025).

## 4.2 Main Results

Table 1 reports performance on the CSI 500 and S&P 500, evaluated on excess returns. On the CSI 500, FaVOR achieves the best result across all four metrics, reaching AR of 0.2067, IR of 1.5295, MDD of −0.0853, and CR of 0.2225, simultaneously improving return generation and reducing downside risk relative to every compared method.

On the S&P 500, FaVOR again attains the best result across all four metrics, reaching AR of 0.1062, IR of 1.1315, MDD of −0.0443, and CR of 0.1123. This consistency suggests that FaVOR’s validation step generalizes across distinct equity markets, producing effective factor combinations in both the Chinese A-share and U.S. universes. Appendix F reports the run-to-run variance over 10 independent runs and confirms that these headline numbers are reproducible.

<table><tr><td colspan="3">Hypothesis: After asharp sell-off, stocks that exhibitintraday recovery, closing nearer to the daily high, tend to enterearly stabilization, which is associated with a short-term rebound over the next 3 trading days.</td></tr><tr><td>Observable Condition</td><td>Candidate Factor</td><td>Validation reasoning based on distributional evidence</td></tr><tr><td rowspan="2">O1: Post Sell-off State</td><td>TS_ZSCORE(C,10)</td><td>Reasoning: &quot;As factor intensity increases, prices close progressively lower within the daily range, indicating sustained downside pressure. This shift is consistently supported by aligned location and tail behavior, with DIR mean and median decreasing  $\overline { { ( 0 . 1 4  - 0 . 1 1 ) \dots } }$  and tail statistics moving downward together, characterizing a post sell-off state.&quot; Decision: PASS (√)</td></tr><tr><td>TS_ZSCORE(V, 10) * TS_Z SCORE(DELTA(C, 1), 10)</td><td>Reasoning: &quot;The factor does not induce a stable separation of post sell-off states. Upper-tail expansion  $( q 9 0 + 2 4 \ – 1 5 7 \% ) \ \dots$  coincides with sharp kurtosis compression(—46–49%), producing internally conflicting tail signals.&quot; Decision: FAIL (×)</td></tr><tr><td rowspan="2">02: Intraday Recovery</td><td>(close TS_MIN(low, 3)) /  $( \mathsf { h i g h } - \mathsf { T S \_ M I N } ( \mathsf { 1 o w }$   $3 ) \ + \ e { \mathsf { p s } } )$ </td><td>Reasoning: &quot;The factor captures a clear shift in closing position within the daily range, with prices closing progressively nearer to the intraday high as factor strength increases. POS mean and median increase monotonically  $( 0 . 0 8 3 { \ - } \partial . 9 1 4 ) \ \dots$  and the upper tail also rises (POS q90 → 1.0), indicating strengthening intraday buying pressure.&#x27; Decision: PASS (√)</td></tr><tr><td>-CORR(rank(DELTA (1ogV)), rank(DIR))</td><td>Reasoning: &quot;The induced trend does not align with the intended interpretation of factor intensity. Although the polarity is defined as higher-is-more-true&#x27;, the evidence decreases across bins ... directly contradicting the expected intraday recovery pattern. As a result, the factor fails to provide a meaningful proxy for intraday recovery.&#x27; Decision: FAIL (×)</td></tr><tr><td>O3: Early Stabiliza- tion</td><td>(TS_MIN(low, 3) low)  $/ ( \mathsf { h i g h } - \mathsf { 1 o w }$  + eps)</td><td>Reasoning: &quot;The factor captures a consistent narrowing of the daily range as factor strength increases. MAG mean and median both decrease (-21% and -18%) ... and tail evidence compresses (MAG q90 -20.5%), with increased central concentration (kurtosis +10.9%). This pattern is consistent with early stabilization.&#x27;</td></tr><tr><td rowspan="2"></td><td> $1 ~ / ~ ( \mathsf { e p s } + \mathsf { T S \_ S T D } ( \mathsf { C }$  20))</td><td>Decision: PASS (√) Reasoning: &quot;The factor does not provide a coherent narrowing pattern as factor strength increases. Tail indicators are internally inconsistent, with kurtosis decreasing (-28.4%)</td></tr><tr><td></td><td>... while q90 shows little change (+1.8%). This inconsistency prevents a stable charac- terization of early stabilization.&quot; Decision: FAIL (×)</td></tr></table>

Table 2: Case study of factor-level validation under a single hypothesis. The table reports candidate factors generated from decomposed observable market conditions and their corresponding validation outcomes. For each candidate factor, we show the rationale produced by the validation agent, with underlined phrases highlighting the evidence used in factor-level validation.

Across the method families in Table 1, only Fa-VOR combines adaptive factor generation with an explicit distributional check between each factor and the market state it should measure, and this combination delivers the strongest return and risk profile in both markets. Appendix G reports the trade-level distribution and shows that gains are spread broadly across many trades.

Figure 4 traces cumulative excess returns. On the CSI 500, FaVOR captures an early-year gain and preserves it while most baselines drift sideways and turn negative by year-end. On the S&P 500, Fa-VOR remains near zero through early months and compounds steadily from mid-year onward, ending as the strongest trajectory while most baselines stay weakly positive or drift negative. Appendix H characterizes the 2025 market paths for both universes.

## 4.3 Case Study: Factor-level Validation

Table 2 reports factor-level validation outcomes for candidates generated under a single hypothesis. Under the intraday-recovery observation (O2), the two candidates yield opposite verdicts. The passing candidate captures intraday recovery consistently. As factor strength increases across quantile bins, stocks close progressively nearer to the upper end of the intraday price range, central tendency statistics shift consistently, and the upper tail strengthens.

<table><tr><td rowspan="3">Backbone</td><td colspan="4">CSI 500</td><td colspan="4">S&amp;P 500</td></tr><tr><td>AR</td><td>IR</td><td>MDD</td><td>CR</td><td>AR</td><td>IR</td><td>MDD</td><td>CR</td></tr><tr><td>GPT-40 (default)</td><td>0.2067</td><td>1.5295</td><td>-0.0853</td><td>0.2225</td><td>0.1062</td><td>1.1315</td><td>-0.0443</td><td>0.1123</td></tr><tr><td>GPT-5.4-mini</td><td>0.1039</td><td>1.3246</td><td>-0.0443</td><td>0.1079</td><td>0.0299</td><td>0.2553</td><td>-0.1186</td><td>0.0244</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.0732</td><td>0.4998</td><td>-0.1418</td><td>0.0657</td><td>0.0808</td><td>0.7039</td><td>-0.0621</td><td>0.0807</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.0541</td><td>0.5364</td><td>-0.0878</td><td>0.0511</td><td>0.0072</td><td>0.0817</td><td>-0.0898</td><td>0.0035</td></tr><tr><td>Llama-3.3-70B</td><td>0.0449</td><td>0.4183</td><td>-0.1313</td><td>0.0407</td><td>0.0146</td><td>0.1545</td><td>-0.0801</td><td>0.0107</td></tr><tr><td>Qwen3-235B</td><td>0.0948</td><td>1.0598</td><td>-0.0936</td><td>0.0967</td><td>0.0174</td><td>0.1670</td><td>-0.1033</td><td>0.0126</td></tr></table>

Table 3: Sensitivity of FaVOR to the LLM backbone on the CSI 500 and S&P 500. All pipeline settings other than the backbone are held fixed. The best result in each column is shown in bold, and the second-best result is underlined.

![](images/6d4c4ed637b78d865d1b070e5d383ccf0c121a31e9db0c5191092276584c672d.jpg)

![](images/abf6a09a539e88ce2cd03a77c9cda884f37a22de94191113630278dcd7433583.jpg)  
Figure 5: Comparison of cumulative excess returns of various baselines on the CSI 500 (left) and S&P 500 (right).

The failing candidate is defined so that higher values should indicate stronger intraday recovery, yet the observed distributional behavior moves in the opposite direction as factor strength increases. This mismatch between factor strength and distributional response prevents the factor from consistently measuring the intraday-recovery observation.

The same PASS/FAIL contrast holds for the post sell-off (O1) and early-stabilization (O3) observations in Table 2. Factor consistency therefore separates at the observation level through distributional behavior alone, which is the central role that our validation framework plays.

## 4.4 Impact of Factor Validation and Integration

We ablate Stage 2 (Factor Validation) and Stage 3 (Factor Integration) of FaVOR, and Table 4 reports the results. Removing Stage 2 lets unvalidated factors enter the integration stage, and MDD more than doubles in magnitude as the integration step admits weaker signals that yield little risk-adjusted return. Removing Stage 3 produces the largest deterioration in return, because validated factors fire individually rather than through a jointly checked combination. The mutual consistency enforced by the combination step is essential for translating individual factor signals into coherent trading decisions.

<table><tr><td>Setting</td><td>AR</td><td>IR</td><td>MDD</td></tr><tr><td>FaVOR</td><td>0.2067</td><td>1.5295</td><td>-0.0853</td></tr><tr><td>w/o Stage 2</td><td>0.0492</td><td>0.1526</td><td>-0.2091</td></tr><tr><td>w/o Stage 3</td><td>-0.3902</td><td>-1.2748</td><td>-0.4260</td></tr><tr><td>w/o Stage 2 &amp; Stage 3</td><td>-0.2793</td><td>-1.4316</td><td>-0.3087</td></tr></table>

Table 4: Ablation on CSI 500 out-of-sample 2025. All rows share the hypothesis and Stage 4 configuration of the main result. Best per column in bold.

Removing both stages reduces FaVOR to unconstrained factor generation, and every metric in Table 4 turns negative. Factor validation and integration are therefore jointly responsible for the existence of usable trading signals, not merely a refinement on top of an already-working pipeline.

## 4.5 Sensitivity to LLM Backbone

Because FaVOR delegates hypothesis generation, observation decomposition, formula synthesis, and construct validation to language-model agents, the choice of backbone is a natural axis of robustness analysis. We hold all other pipeline components fixed and replace only the default GPT-4o (Hurst et al., 2024) with five alternatives: the closed-source GPT-5.4-mini (OpenAI,

2026), Gemini-2.5-Flash (Comanici et al., 2025), and Claude-Sonnet-4.6 (Anthropic, 2026), and the open-weight Llama-3.3-70B (Grattafiori et al., 2024) and Qwen3-235B (Yang et al., 2025). Table 3 reports the resulting metrics on both the CSI 500 and S&P 500, while Figures 5 present the corresponding cumulative-return curves.

FaVOR retains positive annualized return (AR) and information ratio (IR) on both markets across all evaluated backbones, indicating that its effectiveness is not tied to a single proprietary model. On the CSI 500, the open-weight Qwen3-235B achieves an IR of 1.0598 and remains competitive with several closed-source alternatives. This result demonstrates that FaVOR can retain meaningful predictive performance when paired with an openweight backbone.

On the S&P 500, the default GPT-4o is the strongest configuration, ranking first across all four metrics. Gemini-2.5-Flash ranks second, achieving an AR of 0.0808, an IR of 0.7039, an MDD of −0.0621, and a CR of 0.0807. The openweight Llama-3.3-70B and Qwen3-235B also produce positive AR and IR, although their performance is weaker than that of the leading closedsource backbones. Overall, the uniformly positive returns suggest that FaVOR’s hypothesis-centric pipeline transfers across different LLM families, whereas the performance dispersion across backbones shows that backbone choice still affects the strength and robustness of the resulting factors.

## 5 Conclusion

We introduced FaVOR, a hypothesis-driven agentic framework that grounds factor mining in structural validity. Given a hypothesis, FaVOR decomposes it into observable market conditions, performs the first factor-level validation against target conditions, and integrates the subset that best reconstructs the hypothesis. On the CSI 500 and S&P 500, FaVOR delivers stable and interpretable performance against competitive baselines, and ablations confirm that the Validation and Integration stages are each essential.

## Limitations

Scope of Directional Selectivity. Our selection criterion targets factor combinations whose predicted direction stays consistent as the threshold becomes stricter. It is not designed for reversaltype signals (e.g., contrarian patterns at extreme deciles) whose direction flips under stricter thresholds. Such factors fall outside the scope of this work.

Cost Model. Our backtests use daily-bar data, where market impact and slippage are typically negligible at the trade sizes and holding periods we consider. Appendix G confirms an average holding period of 9.64 days, which keeps turnover low. Future work targeting higher-frequency trading may require a high-fidelity execution simulator to capture market impact and slippage.

## Ethical Considerations

The market data we use are aggregated OHLCV fields containing no personally identifiable information, and we use only daily price and volume. FaVOR uses LLMs to propose and select candidate factors. Every LLM output is verified against historical return statistics rather than accepted at face value, and no personally identifiable or proprietary information is included in any prompt.

All artifacts used in this work are employed following their intended research use and license terms. These include market data from Baostock and Yahoo Finance, the Qlib backtesting framework, the Optuna hyperparameter optimization library, the baseline implementations, and the GPT-4o, Claude, Gemini, Llama, and Qwen large language models.

## Acknowledgements

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00556289), in part by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2026-RS-2020-II201789) and the Artificial Intelligence Convergence Innovation Human Resources Development (IITP-2026-RS-2023- 00254592), supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation).

## References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A nextgeneration hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, page 2623–2631, New York, NY, USA. Association for Computing Machinery.

Anthropic. 2026. Claude sonnet 4.6 system card. System card, Anthropic.

David H Bailey, Jonathan M Borwein, Marcos López De Prado, and Qiji Jim Zhu. 2014. Pseudomathematics and financial charlatanism: The effects of backtest over fitting on out-of-sample performance. Notices ofthe AMS, 61(5):458–471.

Malcolm Baker, Brendan Bradley, and Jeffrey Wurgler. 2011. Benchmarks as limits to arbitrage: Understanding the low-volatility anomaly. Financial Analysts Journal, 67(1):40–54.

Rolf W Banz. 1981. The relationship between return and market value of common stocks. Journal of financial economics, 9(1):3–18.

Sanjoy Basu. 1977. Investment performance of common stocks in relation to their price-earnings ratios: A test of the efficient market hypothesis. The journal ofFinance, 32(3):663–682.

Jennifer Bender, Remy Briand, Dimitris Melas, and Raman Aylur Subramanian. 2013. Foundations of factor investing. Available at SSRN 2543990.

Tianqi Chen. 2016. Xgboost: A scalable tree boosting system. Cornell University.

Junyan Cheng and Peter Chin. 2024. Empirical asset pricing with large language model agents. arXiv preprint arXiv:2409.17266.

Yuhan Cheng and Ke Tang. 2024. Gpt’s idea of stock factors. Quantitative Finance, 24(9):1301–1326.

J.H. Cochrane. 2009. Asset Pricing: Revised Edition. Princeton University Press.

John H Cochrane. 2011. Presidential address: Discount rates. The Journal offinance, 66(4):1047–1108.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Hongjun Ding, Binqi Chen, Jinsheng Huang, Taian Guo, Zhengyang Mao, Guoyi Shao, Lutong Zou, Luchen Liu, and Ming Zhang. 2025. Alphaeval: A comprehensive and efficient evaluation framework for formula alpha mining. arXiv preprint arXiv:2508.13174.

Yitong Duan, chuheng zhang, and Jian Li. 2025. Factormad: A multi-agent debate framework based on large language models for interpretable stock alpha factor mining. In Proceedings ofthe 6th ACM International Conference on AI in Finance, pages 605–613.

Martin Eling and Frank Schuhmacher. 2007. Does the choice of performance measure influence the evaluation of hedge funds? Journal of Banking & Finance, 31(9):2632–2647.

Eugene F Fama. 1970. Efficient capital markets: A review of theory and empirical work. The journal of Finance, 25(2):383–417.

Eugene F Fama and Kenneth R French. 1993. Common risk factors in the returns on stocks and bonds. Journal offinancial economics, 33(1):3–56.

Eugene F Fama and Kenneth R French. 1996. Multifactor explanations of asset pricing anomalies. The journal offinance, 51(1):55–84.

Eugene F Fama and Kenneth R French. 2015. A fivefactor asset pricing model. Journal of financial economics, 116(1):1–22.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Kewei Hou, Chen Xue, and Lu Zhang. 2015. Digesting anomalies: An investment approach. The Review of Financial Studies, 28(3):650–705.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Narasimhan Jegadeesh and Sheridan Titman. 1993. Returns to buying winners and selling losers: Implications for stock market efficiency. The Journal of Finance, 48(1):65–91.

Zuoyou Jiang, Li Zhao, Rui Sun, Ruohan Sun, Zhongjian Li, Jing Li, Daxin Jiang, Zuo Bai, and Cheng Hua. 2025. Alpha-r1: Alpha screening with llm reasoning via reinforcement learning. arXiv preprint arXiv:2512.23515.

Zura Kakushadze. 2016. 101 formulaic alphas. Wilmott, 2016(84):72–81.

Haoqiang Kang and Xiao-Yang Liu. 2023. Deficiency of large language models in finance: An empirical examination of hallucination. arXiv preprint arXiv:2311.15548.

Jonathan M Karpoff. 1987. The relation between price changes and trading volume: A survey. Journal of Financial and quantitative Analysis, 22(1):109–126.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.

Hyeonjin Kim, Jiwoo Jeong, Hyungjin Ko, and Woojin Lee. 2026. Large language models as financial analysts: Sector-aware reasoning for investment decisions. Computational Economics, pages 1–31.

Zhizhuo Kou, Holam Yu, Junyu Luo, Jingshu Peng, Xujia Li, Chengzhong Liu, Juntao Dai, Lei Chen, Sirui Han, and Yike Guo. 2024. Automate strategy finding with llm in quant investment. seed, 1:3.

Hoyoung Lee, Junhyuk Seo, Suhwan Park, Junhyeong Lee, Wonbin Ahn, Chanyeol Choi, Alejandro Lopez-Lira, and Yongjae Lee. 2025. Your ai, not your view: The bias of llms in investment analysis. In Proceedings of the 6th ACM International Conference on AI in Finance, pages 150–158.

Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, and Jiang Bian. 2025. R&dagent-quant: A multi-agent framework for datacentric factors and model joint optimization. arXiv preprint arXiv:2505.15155.

Zhiwei Li, Ran Song, Caihong Sun, Wei Xu, Zhengtao Yu, and Ji-Rong Wen. 2024. Can large language models mine interpretable financial factors more effectively? a neural-symbolic factor mining agent model. In Findings ofthe Associationfor Computational Linguistics ACL 2024, pages 3891–3902.

Marcos Lopez de Prado and Vincent Zoonekynd. 2025. Why has factor investing failed?: The role of specification errors. The Role of Specification Errors (January 18, 2024). FEB-RN Research Paper No. 101.

Alejandro Lopez-Lira. 2025. Can large language models trade? testing financial theories with llm agents in market simulations. arXiv preprint arXiv:2504.10789.

Haochen Luo, Ho Tin Ko, David Sun, Yuan Zhang, and Chen Liu. 2025a. Evoalpha: Evolutionary alpha factor discovery with large language models. In NeurIPS 2025 Workshop: Generative AI in Finance.

Yichen Luo, Yebo Feng, Jiahua Xu, Paolo Tasca, and Yang Liu. 2025b. Llm-powered multi-agent system for automated crypto portfolio management. arXiv preprint arXiv:2501.00826.

R David McLean and Jeffrey Pontiff. 2016. Does academic research destroy stock return predictability? The Journal ofFinance, 71(1):5–32.

Microsoft. 2025. Alpha 158 from Microsoft Qlib. ht tps://github.com/microsoft/qlib/blob/85c c74846b5af2e3e6d18666a2f6e399396980b9/ql ib/contrib/data/loader.py#L61. Accessed: 2025-05-12.

OpenAI. 2026. Gpt-5.4 thinking system card. OpenAI Deployment Safety.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, and 1 others. 2011. Scikit-learn: Machine learning in python. the Journal of machine Learning research, 12:2825–2830.

David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. 1986. Learning representations by backpropagating errors. nature, 323(6088):533–536.

Hao Shi, Weili Song, Xinting Zhang, Jiahe Shi, Cuicui Luo, Xiang Ao, Hamid Arian, and Luis Angel Seco. 2025. Alphaforge: A framework to mine and dynamically combine formulaic alpha factors. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 12524–12532.

Richard G Sloan. 1996. Do stock prices fully reflect information in accruals and cash flows about future earnings? Accounting review, pages 289–315.

Ziyi Tang, Zechuan Chen, Jiarui Yang, Jiayao Mai, Yongsen Zheng, Keze Wang, Jinrui Chen, and Liang Lin. 2025. Alphaagent: Llm-driven alpha mining with regularized exploration to counteract alpha decay. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 2813–2822.

Hariom Tatsat and Ariye Shater. 2025. Beyond the black box: Interpretability of llms in finance. arXiv preprint arXiv:2505.24650.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Saizhuo Wang, Hang Yuan, Lionel M Ni, and Jian Guo. 2024. Quantagent: Seeking holy grail in trading by self-improving large language model. arXiv preprint arXiv:2402.03755.

Saizhuo Wang, Hang Yuan, Leon Zhou, Lionel Ni, Heung Yeung Shum, and Jian Guo. 2025. Alpha-gpt: Human-ai interactive alpha mining for quantitative investment. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 196–206.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Xiao Yang, Weiqing Liu, Dong Zhou, Jiang Bian, and Tie-Yan Liu. 2020. Qlib: An ai-oriented quantitative investment platform. arXiv preprint arXiv:2009.11189.

Wentao Zhang, Lingxuan Zhao, Haochong Xia, Shuo Sun, Jiaze Sun, Molei Qin, Xinyi Li, Yuqing Zhao, Yilei Zhao, Xinyu Cai, and 1 others. 2024. A multimodal foundation agent for financial trading: Toolaugmented, diversified, and generalist. In Proceedings of the 30th acm sigkdd conference on knowledge discovery and data mining, pages 4314–4325.

Zhoufan Zhu and Ke Zhu. 2025. Alphaqcm: Alpha discovery in finance with distributional reinforcement learning. In Proceedings of the 42nd International Conference on Machine Learning (ICML).

## A The operator set

To ensure interpretability, reproducibility, and a controlled search space, all observation functions are constructed using a predefined library of deterministic operators. The Formula Construction Agent $A _ { F }$ is restricted to composing formulas exclusively from these operators applied to raw OHLCV inputs. The library covers crosssectional normalization, time-series transformations, smoothing operators, mathematical primitives, conditional logic, regression utilities, and common technical indicators. This constraint prevents arbitrary black-box expressions and ensures that all generated formulas remain transparent and economically interpretable.

<table><tr><td>Type</td><td>Operators</td></tr><tr><td>Cross-sectional</td><td>RANK, ZSCORE, MEAN, STD, SKEW, KURT, MAX, MIN, MEDIAN</td></tr><tr><td>Time-series</td><td>DELTA, DELAY, TS_MEAN, TS_SUM, TS_RANK, TS_ZSCORE, TS_MEDIAN, TS_PCTCHANGE, TS_MIN, TS_MAX, TS_ARGMAX, TS_ARGMIN, TS_QUANTILE, TS_STD, TS_VAR, TS_CORR, TS_COVARIANCE,</td></tr><tr><td>Moving</td><td>TS_MAD, PERCENTILE, HIGHDAY, LOWDAY, SUMAC av- SMA, WMA, EMA, DECAYLINEAR</td></tr><tr><td>erages &amp; smoothing Mathematical</td><td>PROD, LOG, SQRT, POW, SIGN, EXP,</td></tr><tr><td>operations Conditional &amp;</td><td>ABS, INV, FLOOR COUNT, SUMIF, FILTER</td></tr><tr><td>logical</td><td>Regression &amp; SEQUENCE, REGBETA, REGRESI</td></tr><tr><td>residual</td><td></td></tr><tr><td>cators</td><td>Technical indi- RSI, MACD, BB_MIDDLE, BB_UPPER, BB_LOWER</td></tr></table>

Table 5: Operator library used for observation function construction.

## B Stage 2 Validation Criteria

This appendix gives the precise definitions of the four consistency checks summarized in §3.2.2. The validation agent $\mathbf { \mathcal { A } } _ { V }$ applies these checks to the statistical profile $S _ { i , j }$ of a candidate factor $f _ { i , j }$ , and the factor is accepted only when all four are satisfied.

(i) Central tendency shift. A valid factor should separate the market state described by its observable condition. We therefore check whether at least one state variable $x \in \mathcal { X }$ changes systematically across the ordered factor bins $Q _ { 1 } , \ldots , Q _ { l }$ . For each bin $Q _ { k }$ , we compute the bin-wise mean and median,

$$
\begin{array} { r l } & { \mu _ { k } ( x ) = \mathbb { E } [ x _ { s , t } \mid ( s , t ) \in Q _ { k } ] , } \\ & { m _ { k } ( x ) = \mathrm { M e d i a n } [ x _ { s , t } \mid ( s , t ) \in Q _ { k } ] . } \end{array}
$$

The shift may be increasing or decreasing depending on the semantic direction of the condition. We require that the two statistics move in the same direction across adjacent bins,

$$
\begin{array} { l } { \mathrm { s i g n } \big ( \mu _ { k + 1 } ( x ) - \mu _ { k } ( x ) \big ) } \\ { = \mathrm { s i g n } \big ( m _ { k + 1 } ( x ) - m _ { k } ( x ) \big ) , } \\ { k = 1 , \dots , l - 1 , } \end{array}
$$

together with a non-zero correlation between the bin index k and $\mu _ { k } ( x )$

(ii) Tail behavior. A valid factor should also affect more than the average case. For each bin $Q _ { k }$ and each $x \in \mathcal { X }$ , we define the tail spread as

$$
\Delta _ { k } ^ { \mathrm { t a i l } } ( x ) = q _ { 0 . 9 } ( x \mid Q _ { k } ) - q _ { 0 . 1 } ( x \mid Q _ { k } ) ,
$$

where $q _ { p } ( \cdot \mid Q _ { k } )$ is the empirical pth quantile of x conditional on $( s , t ) \in Q _ { k }$ . We require non-trivial variation across bins: there exist $k _ { 1 } \neq k _ { 2 }$ such that

$$
\begin{array} { r } { \left| \Delta _ { k _ { 1 } } ^ { \mathrm { t a i l } } ( x ) - \Delta _ { k _ { 2 } } ^ { \mathrm { t a i l } } ( x ) \right| > \epsilon , } \end{array}
$$

for a tolerance threshold ϵ.

(iii) Statistical consistency. The observed pattern should not depend on a single statistic. The same state variable x must yield evidence in at least two complementary statistics drawn from $\{ \mu _ { k } ( x ) , m _ { k } ( x ) , \Delta _ { k } ^ { \mathrm { t a i l } } ( x ) \}$ , rather than relying on a single summary measure.

(iv) Semantic consistency. The induced distributional pattern must agree with the meaning of the observable market condition $o _ { i } ^ { \phantom { } }$ . For example, for a factor generated for intraday recovery, stronger factor values should correspond to stronger recovery evidence, not the opposite pattern. This check is delegated to $\mathbf { \mathcal { A } } _ { V }$ , which compares the empirical pattern in $S _ { i , j }$ against the textual description of $o _ { i }$

Decision rule. Let $\boldsymbol { K } = \{ c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 } \}$ denote the four criteria above. $\mathbf { \mathcal { A } } _ { V }$ implements the deterministic decision

$$
\begin{array} { l } { \displaystyle \mathcal { A } _ { V } \big ( o _ { i } , f _ { i , j } , S _ { i , j } \big ) } \\ { \displaystyle = \mathbb { I } \Bigg [ \bigwedge _ { c \in { \mathcal K } } c ( o _ { i } , f _ { i , j } , S _ { i , j } ) \Bigg ] , } \end{array}
$$

where I[·] is the pass/fail indicator. The validated factor set for condition $o _ { i }$ is then

$$
\hat { F _ { i } } = \left\{ f _ { i , j } \in F _ { i } :  { \mathcal { A } } _ { V } ( o _ { i } , f _ { i , j } , S _ { i , j } ) = \mathsf { p a s s } \right\} .
$$

A candidate factor is therefore retained only when all four criteria jointly hold.

## C Evaluation Metrics

We evaluate the time-series performance of the resulting trading strategies using standard portfoliolevel performance metrics. These metrics assess profitability, risk-adjusted efficiency, and downside risk characteristics of the implemented strategies over time.

## C.1 Annualized Return (AR)

Let $R _ { t } ^ { p }$ and $R _ { t } ^ { b }$ denote the portfolio and benchmark returns at period $t ,$ and define the active return $A _ { t } = R _ { t } ^ { p } - R _ { t } ^ { b }$ . Let N be the number of trading periods per year.

The annualized excess return is defined as

$$
\mathrm { A R } = \left( \prod _ { t = 1 } ^ { T } ( 1 + A _ { t } ) \right) ^ { \frac { N } { T } } - 1 ,
$$

which corresponds to the geometric annualization of the strategy’s excess return over the benchmark.

## C.2 Information Ratio (IR)

The Information Ratio evaluates risk-adjusted excess performance relative to a benchmark. Let $R _ { t } ^ { p }$ and $R _ { t } ^ { b }$ denote the portfolio and benchmark returns at time t, respectively. Define the active return as

$$
A _ { t } = R _ { t } ^ { p } - R _ { t } ^ { b } .
$$

The annualized Information Ratio is computed as

$$
\mathrm { I R } = { \frac { { \sqrt { N } } \mathbb { E } [ A _ { t } ] } { \operatorname { S t d } [ A _ { t } ] } } ,
$$

where $\mathbb { E } [ A _ { t } ]$ and $\mathrm { S t d } [ A _ { t } ]$ denote the sample mean and standard deviation of the active returns. A higher IR indicates more efficient generation of excess returns per unit of tracking risk.

## C.3 Maximum Drawdown (MDD)

Maximum drawdown measures the largest peak-totrough decline of the portfolio’s own cumulative wealth (not the excess equity curve). Let the cumulative net asset value of the portfolio be

$$
V _ { t } = \prod _ { \tau = 1 } ^ { t } ( 1 + R _ { \tau } ^ { p } ) .
$$

The drawdown at time t is

$$
\mathrm { D D } _ { t } = \frac { V _ { t } } { \operatorname* { m a x } _ { \tau \leq t } V _ { \tau } } - 1 , \in [ - 1 , 0 ] ,
$$

and the maximum drawdown over the evaluation period is

$$
\mathrm { M D D } = \operatorname* { m i n } _ { 1 \leq t \leq T } \mathrm { D D } _ { t } .
$$

An MDD value closer to zero indicates better downside risk control and capital preservation

## C.4 Cumulative Return (CR)

Cumulative Return measures the total excess return of the strategy over the entire investment horizon. It is defined as

$$
\mathrm { C R } = \prod _ { t = 1 } ^ { T } ( 1 + A _ { t } ) - 1 ,
$$

where $A _ { t } = R _ { t } ^ { p } - R _ { t } ^ { b }$ is the active return defined above. This metric summarizes full-period performance relative to the benchmark without annualization.

## D Hyperparameter Summary

FaVOR does not rely on extensive hyperparameter tuning. Most values are deterministic design parameters held constant across all experiments and across both equity universes. Table 6 lists the LLM agent decoding temperatures. Table 7 reports the pre-backtest Stage 1–3 parameters. Table 8 specifies the Stage 4 backtest configuration.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Hypothesis agent temperature</td><td>0.9</td></tr><tr><td>Observation agent temperature</td><td>0.7</td></tr><tr><td>Formula agent temperature</td><td>0.7</td></tr><tr><td>Validation agent temperature</td><td>0.1</td></tr><tr><td>Candidate formulas per observation ni</td><td>2-3</td></tr><tr><td>Quantile bins l</td><td>5</td></tr><tr><td>Min. valid bins to retain a factor</td><td>3</td></tr><tr><td>Max. refinement iterations</td><td>5</td></tr><tr><td>Strictness levels</td><td>Q50, q70, q90</td></tr><tr><td>Directional Selectivity threshold</td><td>0.8</td></tr><tr><td>Cross-ticker pass-rate (PASS)</td><td>0.5</td></tr><tr><td>Cross-ticker pass-rate (strict)</td><td>0.7</td></tr><tr><td>Combination pass-rate to Stage 4</td><td>0.5</td></tr></table>

Table 6: LLM decoding temperatures by agent role.

Table 7: Pipeline parameters for Stages 1–3 (factor generation, factor-level validation, and factor integration).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Threshold σ search range</td><td>[0.55, 0.95]</td></tr><tr><td>Combined-signal quantile q</td><td>0.9</td></tr><tr><td>Holding horizon T</td><td>7</td></tr><tr><td>Objective</td><td>Calmar</td></tr><tr><td>Number of Optuna trials</td><td>20</td></tr><tr><td>Entry-confirmation rule</td><td>raw signal entry</td></tr><tr><td>Outer-loop iterations</td><td>3</td></tr><tr><td>Stage-4 combo workers</td><td>4</td></tr><tr><td>Stage-4 Optuna jobs per combo</td><td>1</td></tr></table>

Table 8: FaVOR-specific Stage 4 search and runtime configuration. The holding horizon T is LLM-selected from {1, . . . , 10}; we list the value used for the main result.

The remaining trading-strategy environment is shared by FaVOR and every baseline. Data splits, transaction costs, Qlib execution rule, and stop-loss policy are applied identically across all methods. Each baseline model is trained and tuned using the Qlib (Yang et al., 2020) reference workflow configuration without modification. This covers the input feature set, label horizon, training objective, hyperparameter search budget, validation usage, and random seed handling. For the ML and DL baselines the input feature set is Alpha158 (Microsoft, 2025), and the same feature set is provided to FaVOR and the LLM-based baselines as the underlying observable column set. Table 9 summarizes this shared backtest environment.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Train / Validation / Test</td><td>2022–2023 / 2024 / 2025</td></tr><tr><td>CSI 500 buy / sell fee</td><td>0.0005 / 0.0015</td></tr><tr><td>S&amp;P 500 buy / sell fee</td><td>0 / 0.0005</td></tr><tr><td>Execution rule</td><td>TopkDropout (k=50, n=5)</td></tr><tr><td>Stop-loss</td><td>Disabled</td></tr></table>

Table 9: Shared backtest environment applied identically to FaVOR and all baselines: data splits, transaction costs, Qlib execution rule, and stop-loss policy.

## E Prompt Configuration

We employ four specialized language-model-based agents for hypothesis construction, observation decomposition, formula generation, and construct validation. For full reproducibility, every prompt template used at runtime is reproduced here from the released codebase. Placeholders of the form {...} are filled in at runtime with task-specific contents (e.g., the seed idea, retrieved knowledge, the current observation plan, or the function library description). Each agent may be invoked once per round. The generation prompts serve as the default per-round entry point and may receive prior-round context such as {feedback} or {formula\_memory}. The regeneration and refinement prompts are reserved for explicit revision triggers within a round.

You are a Trading Hypothesis Generation Agent in quantitative finance.

Your role is to convert a high-level trading idea into a concise, causal trading hypothesis that explains WHY a particular price behavior may occur.

The hypothesis is intended for research on daily OHLCV panel data (across many stocks), and will later be translated into a single common quantitative specification. Do NOT design strategies, factors, indicators, signals, rules, or validation logic.

## CORE PRINCIPLES:

\- Explain WHY a particular price behavior may occur, using plain, economic language.

\- Refer to realistic market participants and constraints (e.g., liquidity providers, short-term traders).

\- Describe tendencies or pressures, not guaranteed outcomes.

\- The Trading Idea may be abstract (e.g., momentum, downside mean reversion), not a specific scenario.

\- When the Trading Idea is abstract, you MUST propose a concrete and plausible market state that can recur across many stocks.

\- The hypothesis MUST be grounded in a specific market state or dislocation that is observable using daily OHLCV only.

\- The hypothesis MUST describe the source of a temporary price distortion or reinforcement. - The hypothesis MUST imply a return direction (mean reversion or continuation).

## OBSERVABILITY & DATA CONSTRAINTS (IMPORTANT):

\- Assume access ONLY to daily OHLCV data. - Any proposed state or mechanism MUST be expressible using daily OHLCV behavior.

\- Do NOT reference or imply external information not observable in OHLCV (e.g., news, earnings, macro events, fundamentals, intrinsic or fair value).

\- Use only measurement-friendly language (e.g., sharp sell-off, volume surge, range expansion, stabilization).

## STYLE & STRUCTURE:

\- Write in a neutral, academic hypothesis style (not narrative or commentary).

\- Use a consistent causal structure: [OHLCVdefined market state] → [behavioral/structural pressure] → [price distortion or reinforcement] → [short-term rebound or continuation]. - Keep the hypothesis concise and readable (1–3 sentences).

\- Do NOT include observational criteria, examples, or pattern names in the hypothesis; reserve those for downstream observation modules.

\- Do NOT write trading rules, thresholds, indicators, or formulas.

## HORIZON DAYS:

\- You MUST specify ’horizon\_days’ as an integer between 1 and 10.

\- Treat horizon\_days as the intended holding period / time-stop window to evaluate the hypothesized effect.

\- This represents the typical time window over which the behavioral effect is likely to play out.

\- The chosen horizon\_days MUST be causally

consistent with the described behavioral mechanism.

\- Use Iteration Feedback / Existing Hypotheses context to ADAPT horizon\_days:

\- If recent hypotheses with a similar mechanism performed poorly out-of-sample, pick a DIFFERENT horizon\_days.

\- Avoid reusing the same horizon\_days as the most recent 2 hypotheses unless you have a strong causal reason.

\- When unsure, deliberately explore a different horizon (e.g., move from 2–4 days to 5–7 days, or vice versa).

## OUTPUT CONSTRAINT (TOOL-CALL ONLY):

\- You MUST respond by calling the "hypothesis\_tool" function.

\- Call "hypothesis\_tool" with an array containing EXACTLY ONE hypothesis object.

\- The hypothesis MUST include: - hypothesis\_id - hypothesis\_name - behavioral\_description - horizon\_days

## Prompt template for Hypothesis Agent A<sub>H</sub> (outer-loop regeneration).

## BEHAVIORAL\_HYPOTHESIS\_REGEN\_ SYSTEM\_PROMPT =

You are a Behavioral Hypothesis REGENER-ATION Agent in quantitative finance.

Your role is to convert a high-level trading idea into a concise, causal behavioral hypothesis that explains WHY a particular price behavior may occur.

This is NOT a blank-slate generation. You are in an outer-loop refinement step: previous hypotheses have already been tested (IS/OOS), and you must propose a NEW hypothesis that addresses what failed.

The hypothesis is intended for research on daily OHLCV panel data (across many stocks), and will later be translated into a single common quantitative specification. Do

<table><tr><td>NOT design strategies, factors, indicators, signals, rules, or validation logic. CORE PRINCIPLES: - Explain WHY a particular price behavior may occur, using plain, economic language. - Refer to realistic market participants and con- straints (e.g., liquidity providers, short-term traders) - Describe tendencies or pressures, not guaranteed outcomes. - The Trading Idea may be abstract (e.g., momentum, downside mean reversion), not a specific scenario. - When the Trading Idea is abstract, you MUST propose a concrete and plausible</td><td>modules. - Do NOT write trading rules, thresholds, indicators, or formulas. REGENERATION RULES: - Use the provided Existing Hypotheses + Iteration Feedback to avoid repeating the same idea. - Aim to change something material vs. recent hypotheses: - Prefer changing the causal mechanism / market state (not just rephrasing), but small adjustments are acceptable if the feedback strongly supports them. HORIZON DAYS: - You MUST specify &#x27;horizon_days’as an integer between 1 and 10. - Treat horizon_days as the intended holding period / time-stop window to evaluate the hypothesized effect. - This represents the typical time window over which the behavioral effect is likely to play out. - The chosen horizon_days MUST be causally consistent with the described behavioral mechanism.</td></tr></table>

Trading Idea:

{concept\_text}

{columns}

Retrieved Knowledge (background context   
only):   
{knowledge}

Iteration Feedback (optional): {feedback}

Existing Hypotheses: {existing\_hypotheses}

Existing Hypothesis IDs: {existing\_ids}

Task:   
Write ONE concise trading hypothesis explaining the expected price behavior and its underlying mechanism.   
Keep it short (1–3 sentences) and aligned with the idea.

Return EXACTLY ONE trading hypothesis following the required schema fields.

## Prompt template for Observation Agent A<sub>O</sub>.

## OBSERVATION\_SYSTEM\_PROMPT =

You are an Observation Planning Agent in quantitative finance.

Your role is to decompose a Behavioral Hypothesis into a small set of

observable market conditions that describe BOTH:

1) The market SETUP state (the situation/context)

2) Early TRANSITION signals that increase

the probability of the hypothesized outcome

Assume daily bar data (one record per trading

must be describable using only the provided columns and simple derivations from them.

Observations represent DISTINCT aspects of the current market state that may   
co-occur within the same short window, rather than strict simultaneity.   
Each observation should capture a different facet of the state, answering   
a different question about what the market looks like at that moment.

## CORE GUIDELINES:

\- Observations should include BOTH setup conditions AND early transition signals.

For each hypothesis, include at least one transition observation that

describes evidence the current market state is evolving in a way that

supports the hypothesized direction, expressed ONLY through directly

observable OHLCV behavior, and framed as the absence or weakening of

forces opposing the hypothesis rather than as an asserted outcome

(e.g., no stabilization, support, or reversal claims).

\- Each observation must capture ONE distinct dimension of the market state

(e.g., price displacement, trading activity, directional bias,

volatility/instability, or price extension relative to recent behavior).

\- Decompose the hypothesis into conditions that address DIFFERENT aspects

of the market state, rather than describing the same phenomenon from

## multiple angles.

\- Prefer state-level descriptions over singlecandle or intraday-specific

descriptions; intraday volatility or range should only be used if it

represents a distinct market state not already implied by price movement

and trading activity.

\- Do NOT define multiple observations that describe the same underlying

market condition (e.g., price being unusually low) using different

expressions such as changes, levels, or - Describe market states conceptually but ONLY in terms of directly observable OHLCV behavior; do NOT reference inferred intent, stabilization, support, recovery, or specific technical constructs such as moving averages, oscillators, or named indicators, which should be handled at the formula stage.

- Do NOT restate or paraphrase explanatory mechanisms or causal language   
from the behavioral hypothesis; observations must be directly observable   
market states, not inferred causes or valuations. - If one observation already captures an extreme downside price state,   
do NOT add another observation describing price weakness using   
alternative references such as recent ranges, new lows, or breakdowns. - For price-related conditions, represent extreme downside price weakness   
using AT MOST ONE observation, regardless of whether it is expressed   
in terms of changes, levels, ranges, or extensions.

## CONSISTENCY REQUIREMENTS:

\- All observations must be able to hold true within the same short window.

\- Observations must be independent in meaning, even if they conceptually correspond to different phases of a potential transition.

## SELF-CHECK (before responding):

1) Does any observation describe extreme downside price conditions

that are already captured by another observation?

2) Does any observation restate explanatory language from the hypothesis

3) Have I included at least one transition observation that reflects

or persisting in the direction implied by the hypothesis?   
4) Do all observations describe conditions that can plausibly co-occur?   
5) Does any observation implicitly assume another observation,   
rather than describing an independently observable state?   
6) Is the transition observation expressed as a specific observable fact   
(e.g., frequency of new lows) rather than a summarized market judgment?

## OUTPUT CONSTRAINT:

\- Respond ONLY by calling the "observation\_plan\_tool".

\- Return EXACTLY ONE observation plan following the required schema.

## OBSERVATION\_USER\_PROMPT\_

TEMPLATE =

Hypothesis ID: {hypothesis\_id}

Behavioral Hypothesis: {hypothesis\_json}

Allowed Columns: {columns}

## Task:

List the observable market conditions (daily bar context) that characterize when this phenomenon is present. Include BOTH:

1) SETUP conditions that define the context/state, and

2) At least one TRANSITION observation expressed as OHLCV-only evidence that forces   
opposing the hypothesis are weakening (avoid outcome words like stabilization/support/reversal).

Each observation should capture a distinct aspect of the market state and may co-occur within the same short window (not necessarily the exact same day).

## Prompt template for Formula & Code Agent A<sub>F</sub> (generation).

BEHAVIORAL\_FORMULA\_SYSTEM PROMPT =

You are an Observation-to-Formula Agent in quantitative finance.

Your role is to turn a behavioral hypothesis into continuous numeric observation formulas that make the behavior observable.

## 1. Core Rules:

\- \*\*Must create 2–3 formulas per observation.\*\*

\- No future reference / lookahead. Use only past-window TS\_\* functions.

\- Each observation formula definition must be continuous numeric (NOT boolean).

Polarity is fixed by meaning ("higher\_is\_more\_true" or "lower\_is\_more\_true").

\- Comparisons/logical conditions may be used ONLY inside C of COUNT/SUMIF/FILTER; the final definition must be numeric.

## 2. Formula Pool & Generation Rules:

\- Each formula definition must be unique.

## 3. Expression Language Constraints:

\- Use ONLY the allowed expression operations.

\- Use column names WITHOUT a ’\$’ prefix.

\- Allowed arithmetic operators: +, -, \*, / and parentheses.

\- Window/period parameters (n, p, d, m) must be literal constants, not computed values (e.g., use DELTA(close, 5), not DELTA(close, LOWDAY(close, 10))).

## 4. Design Constraints:

\- Avoid using raw prices and volumes directly due to scale differences

\- Use relative changes or standardized data (e.g., RANK(), ZSCORE())

\- Convert prices to returns, e.g. ‘(DELTA(close, 1)/close)‘ instead of price levels

\- Transform volume into relative changes, e.g. ‘(DELTA(volume, 1)/volume)‘

\- Consider appropriate sample periods for indicators requiring historical data

\- Choose suitable window sizes for moving averages SMA(), EMA(), WMA()

\- All window sizes and weight parameters MUST be positive integers (> 0). Never use 0 for any parameter.

\- Add small constants (e.g., 1e-8) to denominators to prevent division by zero

\- Use TS\_ZSCORE() for formula value standardization

\- Consider SIGN() to reduce impact of extreme values

\- Apply value truncation only with explicit numeric bounds (e.g., MAX(MIN(x, 5), -5)) if supported.

\- \*\*Cross-sectional Treatment:\*\*

\- Apply RANK() or ZSCORE() for crosssectional comparability

\- Use FILTER() for outlier handling - Ensure sufficient window length for correlation calculations

\- \*\*Robustness Considerations:\*\*

\- Validate formula stability across multiple time windows

\- Consider TS\_MEDIAN() over TS\_MEAN()

\- Apply moving averages to smooth highfrequency variations

\- Allow for a range of values or flexibility when defining formulas, rather than imposing strict equality constraints.

\- For example, a strict equality check between two rolling values is too restrictive.

\- Instead, use a continuous proximity score (no comparisons), e.g.:

(TS\_STD(low,20)/10 + 1e-8) (ABS(TS\_MIN(low,10) DE-LAY(TS\_MIN(low,10),1)) + TS\_STD(low,20)/10 + 1e-8)

\- When given specific duplicated subexpressions to avoid, ensure new formula expressions use alternative calculations

\- Replace duplicated patterns with semantically similar but structurally different expressions - For example, if ‘ABS(close - open)‘ is flagged as duplicated:

\- Consider using ‘(high - low)‘ for price range - Use ‘SIGN(close - open) \* (close - open)‘ for directional magnitude

\- Explore other price difference combinations like ‘(high - low) / (open + close)‘

\- Maintain formula interpretability while avoiding structural repetition

\- Focus on unique combinations of operators and variables to ensure originality

## 5. Mandatory Self-Correction & Polarity Check:

\- Verify the polarity and mathematical logic.

\- Ensure volume formulas distinguish buying pressure vs dumping when required.

## 6. Output Constraint:

\- You MUST respond ONLY by calling the "behavioral\_formula\_tool" function with exactly one bundle.

\- Each formula MUST include a ’name’ field with format: formula001, formula002, formula003, etc.

\- Names are FIXED identifiers - do NOT change them unless explicitly instructed.

Strictly adhere to the syntax requirements; do not use undeclared variables (e.g., n) or functions.

BEHAVIORAL\_FORMULA\_USER\_ PROMPT\_TEMPLATE =

Behavioral Hypothesis ID:

{hypothesis\_id}

Observation Plan (generate formulas for EACH observation\_id below): {observation\_plan\_json}

Allowed Columns (use column names WITH-  
OUT a ’\$’ prefix):   
{columns} Allowed Operations (use ONLY these functions/operators, and match exact signatures/argument counts):   
{function\_lib\_description}

Retrieved Knowledge (reference only): {knowledge}

Previously Generated Formulas / Memory & Feedback:

{formula\_memory}

## Prompt template for Formula & Code Agent A<sub>F</sub> (refinement).

## BEHAVIORAL\_FORMULA\_REFINE SYSTEM\_PROMPT =

You are an Observation-Alpha FormulaAgent performing RESEARCH-SAFE refinement.

Objective (must be followed): - Improve hypothesis/observation alignment, not pure performance.

\- The diagnostics will specify which formulas FAILED validation (see ‘failed\_formula\_names‘).

\- You MUST ONLY modify the formulas listed in ‘failed\_formula\_names‘.

\- All other formulas (PASS) MUST be returned UNCHANGED with their exact original definitions.

\- Return the COMPLETE bundle with ALL formulas (both modified FAIL formulas and unchanged PASS formulas).

\- \*\*Metric Priority & Action Trigger\*\*:

1. \*\*CRITICAL\*\*: If stop\_loss\_rate > 30%,

you MUST REDESIGN entry timing.

2. \*\*Medium\*\*: If a single formula causes a bottleneck (>80% failure share) but trades are too few, relax or simplify its formula.

1. \*\*Structural Redesign (Mandatory for Critical Failures)\*\*:   
If diagnostics reveal logical contradictions or extreme invalid exposure, DO NOT just tune. Change the formula structure.   
\* \*Example (Fixing Polarity)\*:   
Change ‘volume / SMA(volume)‘ to ‘(close - open) \* volume / SMA(volume)‘ to capture directional pressure.

2. \*\*Parameter Tuning\*\*: Adjust windows (Only for stable bundles).

## Hard constraints:

- Allowed edits:   
\* Redefine observation formulas using allowed DSL (columns + functions).

## - Forbidden:

Keep it logically grounded:   
- Every change must have a behavioral rationale.   
If you redesign a formula, explain why the new structure better captures the hypothesis. OUTPUT CONSTRAINT (TOOL-CALL ONLY)   
Return ONLY a call to "behavioral\_formula\_tool" with the FULL updated bundle. - Include ALL formulas from the input bundle (both PASS and FAIL).   
- Only modify the definitions of formulas in ‘failed\_formula\_names‘.   
- Keep the same names for all formulas.

## BEHAVIORAL\_FORMULA\_REFINE

Behavioral Hypothesis ID: {hypothesis\_id}

Observation Plan (do NOT change the set of observation\_id; refine formulas only): {observation\_plan\_json}

Current Bundle (JSON): {current\_bundle\_json}

Diagnostics Summary (alignment-first; NOT   
optimization target):   
{diagnostics\_json}

Requested refinement focus: {focus}

Allowed Columns (use column names WITH-  
OUT a ’\$’ prefix):   
{columns} Allowed Operations (use ONLY these functions/operators, and match exact signatures/argument counts):   
{function\_lib\_description} Rules:   
- Do NOT invent new data fields; use allowed columns + allowed expression operations only. - Ensure every formula definition remains continuous numeric (not boolean).   
- Preserve the ‘ticker‘ dimension in the final expression (one value per ‘timestamp‘×‘ticker‘). - Window/period parameters (n, p, d, m) must be literal constants, not computed values (e.g., use DELTA(close, 5), not DELTA(close, LOW-DAY(close, 10))).

Your goal: Review and FIX logical flaws in   
the proposed Formula Bundle.

You are a Quality Assurance Agent for   
Behavioral Formula Bundles.

You are a Formula Validation Agent in   
quantitative finance.   
Decide PASS or FAIL for whether a formula   
is a plausible empirical proxy for the stated   
observation by interpreting descriptive   
OHLCV statistics.

\*\*CRITICAL: For EACH formula, verify   
polarity by answering:\*\*   
1. "When this formula’s value INCREASES,   
what does it mean?" (e.g., price up? down?   
more volatile?)

\*\*Other Checks:\*\*   
- Volume formulas about "buying/selling   
pressure" need price direction (e.g., ‘(close  
open)\*volume‘), not just ‘volume‘.   
- Each formula must have a UNIQUE defini  
tion.

- If errors found: Return CORRECTED bun  
dle. Prefer changing polarity over inverting   
formula.

## Prompt template for Formula & Code Agent A<sub>F</sub> (self-correction).

2. "Does that match what the observation   
claims?" (e.g., if obs says ’price drop’, higher   
formula should mean more drop)

3. If mismatch → flip the polarity or invert the   
formula.

OUTPUT: Return ONLY a call to "behav  
ioral\_formula\_tool" with the FULL bundle.

Review this bundle for logical flaws.

{bundle\_json}

\*\*POLARITY CHECK (do this for EACH formula):\*\*

For each formula, think: "If this formula value goes UP, what happens?" Then check if that matches the observation.

If you find polarity mismatches or other logical errors, FIX THEM. Otherwise return as is.

## Prompt template for Validation Agent A<sub>V</sub>.

## DISTRIBUTION\_JUDGMENT\_SYSTEM PROMPT =

\- DIR = Close − Open (price direction)

\- MAG = High − Low (price range)

\- POS = (Close − Low)/(High − Low) (close location)

\- VOL = Trading Volume (trading activity)

\- Location (primary): mean, median

\- Tail (primary): q10, q90, kurtosis

\- Asymmetry (secondary only): skewness

\- Dispersion (context only): std, IQR (if present)

Skewness may be mentioned only as auxiliary confirmation; never standalone.

<table><tr><td rowspan=1 colspan=1>Bins:- Data is grouped into ordered bins Q1→Qk,where Q1 means the observation is weaker andQk means the observation is stronger. Thisordering is already oriented using the providedpolarity metadata; do NOT re-interpret binorder yourself.You MUST set every  pri-mary_evidence[i].bins to exactly evi-dence_json.bins and provide matching-length numbers copied from evi-dence_json.features[..].You MUST return a single tool call to‘distribution_judgment_tool with:- verdict: PASS or FAIL- checks: {location_involved, tail_amplified,multi_stat_consistent, no_contradiction}- primary_evidence: numeric citations (arrays)- reasoning: 2–4 sentencesPASS requires ALL rules A-D to be satisfied(set the corresponding checks=true):A) Location involvement:- For price-action observations: At least ONEof {DIR, POS, MAG} shows a consistentLOCATION shift across bins.- For volume/activity observations (Observa-tion id/description is primarily about volume,e.g. contains &quot;volume&quot;): VOL is allowed tosatisfy (A) instead.- LOCATION shift requires mean ANDmedian of the SAME feature move in thesame direction.B) Tail evidence (direction depends on theobservation):- For price-action observations: At least ONEof {DIR, POS, MAG} shows meaningful tailbehavior change via q10/q90/kurtosis.- For volume/activity observations: VOL isallowed to satisfy (B) instead.- Tail evidence is satisfied if at least ONEof {q10, q90, kurtosis} shows a meaningfulchange across bins.- This can be tail expansion (e.g., higher q90/ more extreme q10) OR tail compression(e.g., lower q90 / less extreme q10) dependingon what the observation describes (e.g.,</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>&quot;stabilization&quot; implies compression; &quot;panic&quot;implies expansion).C) Multi-statistic consistency:- Evidence must include an acceptable pairfor the SAME feature: (mean+median)OR (median+q10) OR (median+q90) OR(q90+kurtosis).- mean+q90 alone is NOT acceptable.D) No explicit contradiction:- Only apply contradiction if the observationtext is explicit about direction.Examples:- If it explicitly describes sell-off/weakclose/close near lows, then BOTH DIR andPOS should not look like a strong recoveryacross extreme bins- If it explicitly describes recovery/rebound-/strong close/close near highs, then BOTHDIR and POS should not look like continuedsell-off across extreme bins.- Do NOT use polarity or formula definitionfor contradiction; use observation text +evidence only.For non-volume observations, VOL alone isnever sufficient for PASS.Evidence constraints:- If you mark any rule as satisfied, you MUSTcite numeric values that support it.- Each primary_evidence item MUST includeevidence_json.bins and matching-lengthnumeric arrays.- If evidence does not support a satisfied check,verdict MUST be FAIL.OUTPUT CONSTRAINT:- Return EXACTLY ONE tool call to distribu-tion_judgment_tool.- Do NOT output any text outside the tool call.DISTRIBUTION_JUDGMENT_USER_TEMPLATE =Formula:- name: {formula_name}- definition: {definition}- polarity: {polarity}</td></tr></table>

Observation:   
- id: {obs\_id}   
- description: {obs\_description}   
EVIDENCE JSON:   
{evidence\_json}

## F Run-to-Run Stability of LLM Validation

To assess the robustness of LLM-based validation under stochastic decoding, we evaluate the variance of headline performance metrics across 10 independent runs of the full pipeline (each run consisting of 3 iterative rounds) on the CSI 500 universe with the same hypothesis prompt and identical thresholds. Table 10 reports run-to-run variance. The small magnitudes indicate that validation outcomes are reproducible across independent runs and that the performance reported in Table 1 is stable.

<table><tr><td>Metric</td><td>Variance</td></tr><tr><td>Information Ratio (IR)</td><td>0.152</td></tr><tr><td>Annualized Return</td><td>0.042</td></tr><tr><td>Maximum Drawdown (MDD)</td><td>0.120</td></tr><tr><td>Turnover</td><td>0.046</td></tr></table>

Table 10: Run-to-run variance of FaVOR over 10 independent runs.

## G Strategy-Level Statistics

Table 11 reports trade-level distributional statistics from the CSI 500 out-of-sample period, verifying that FaVOR’s cumulative excess return is not driven by a small number of extreme trades. The aggregated numbers confirm that gains are broadly distributed across many trades rather than concentrated in tail events.

<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Number of trades</td><td>1,255</td></tr><tr><td>Average holding period (days)</td><td>9.64</td></tr><tr><td>Mean per-trade return (excess)</td><td>2.279%</td></tr><tr><td>Profit factor</td><td>1.986</td></tr><tr><td>10th percentile of trade returns (excess)</td><td>-7.24%</td></tr><tr><td>90th percentile of trade returns (excess)</td><td>+8.52%</td></tr></table>

Table 11: Trade-level distributional statistics on out-ofsample period.

![](images/d3388505614d8e378c205ab8b9dce3d27b8b40b866b1400c37b125cb093c57b1.jpg)

![](images/5850e3857ca7366afb12a1fbff3e8941e2bb2eb0a794b2aa3440f763322a89ee.jpg)  
Figure 6: Out-of-sample 2025 market regimes for CSI 500 (top) and S&P 500 (bottom), based on trailing 60-day benchmark returns, with FaVOR’s cumulative return on the right axis (base = 1.0)

## H Market-Regime Context for OOS 2025

We summarize the 2025 out-of-sample market context for the two evaluation universes. In Figure 6, each trading day is labeled using the benchmark’s trailing 60-day cumulative return: bull if r<sub>60</sub>>+5%, bear if r<sub>60</sub><−5%, and sideways otherwise. These labels are used only for post-hoc interpretation and are not used by FaVOR during selection or trading.

<table><tr><td>Formula</td></tr><tr><td>CSI 500: 4 formulas, all positive polarity</td></tr><tr><td>f006  $\bar { \mathrm { S M A } } ( \mathrm { v o l } , 7 ) \ \bar { / } \mathrm { S M A } ( \bar { \mathrm { v o l } } , 6 0 )$ </td></tr><tr><td>++ f008  $( \mathrm { c l o s e - M A _ { 3 0 } ^ { - } ) / S T D _ { 3 0 } ( c l o s e ) }$ </td></tr><tr><td>f002  $\mathrm { \Delta \ Z _ { 2 0 } ( m i n _ { 1 0 } l o w ) - Z _ { 2 0 } ( m a x _ { 1 0 } h i g h ) }$  十 f010 ∆vol10</td></tr><tr><td>十</td></tr><tr><td>S&amp;P 500: 3 formulas, mixed polarity</td></tr><tr><td>f004 STD7(high − low) 十</td></tr><tr><td>f001 min20 close / max15 close 一</td></tr><tr><td>f007 STD15(close) 十</td></tr></table>

Table 12: Formula combinations selected for the two out-of-sample portfolios.

The CSI 500 window is mostly a long consolidation followed by a second-half rally. The index traded in a narrow range from January to early July, with only brief bear intervals, and then entered a sustained bull phase into year-end. The S&P 500 window contains a sharper stress episode: the index fell from its February peak to an April low before recovering through the rest of the year. Thus, the two test universes expose FaVOR to different outof-sample market paths: a consolidation-to-rally path in CSI 500 and a drawdown-to-recovery path in S&P 500.

For completeness, Table 12 reports the formula combinations selected for the two out-of-sample portfolios. The table is included only to identify the selected signals.

## I Hypothesis & Formula Consistency Filter

Stage 2 is intended to test whether a candidate formula measures the OHLCV-observable condition stated in its originating hypothesis. It is not designed as a standalone alpha, IC, or RankIC filter. We therefore examine its strategy-level contribution separately from the marginal predictive association of individual formulas.

Stage-removal ablation on both markets. Removing Stage 2 allows formulas that are not validated against their intended observable condition to enter factor integration. Table 13 shows that the resulting excess IR deteriorates in both markets.

<table><tr><td>Market</td><td>Ours</td><td>W/O Stage 2</td></tr><tr><td>CSI 500</td><td>1.5295</td><td>0.1526</td></tr><tr><td>S&amp;P 500</td><td>1.1315</td><td>-0.8831</td></tr></table>

Table 13: Effect of removing Stage 2 factor-level validation. All settings other than the presence of Stage 2 are held fixed.

PASS & FAIL formulas. We further evaluate first-iteration formulas before outer-loop feedback: 437 Stage 2 PASS formulas and 24 FAIL formulas. For each formula, we compute daily cross-sectional RankIC with forward returns at four horizons. As shown in Table 14, PASS formulas do not exhibit a uniform standalone RankIC advantage. This result is consistent with the role of Stage 2 as a semantic and distributional consistency filter rather than an individual-factor return predictor.

A positive forward return association does not imply that a formula measures the state described by its observation. For example, the candidate -TS\_SUM(DELTA(low,1),5) has positive RankIC on both markets across all examined horizons, but larger values correspond to continuing new lows rather than weakening downside pressure. Stage 2 rejects this candidate because its induced OHLCV distribution contradicts the stated condition. Thus, the filter removes formulas that may be predictive while representing the opposite market state from the hypothesis.

<table><tr><td>Market</td><td>Horizon</td><td>PASS RankIC</td><td>FAIL RankIC</td></tr><tr><td>CSI 500</td><td>1 day</td><td>+0.0039</td><td>+0.0025</td></tr><tr><td>CSI 500</td><td>3 days</td><td>+0.0025</td><td>+0.0030</td></tr><tr><td>CSI 500</td><td>5 days</td><td>+0.0023</td><td>+0.0035</td></tr><tr><td>CSI 500</td><td>10 days</td><td>+0.0023</td><td>+0.0047</td></tr><tr><td>S&amp;P 500</td><td>1 day</td><td>+0.0040</td><td>+0.0062</td></tr><tr><td>S&amp;P 500</td><td>3 days</td><td>+0.0027</td><td>+0.0073</td></tr><tr><td>S&amp;P 500</td><td>5 days</td><td>+0.0017</td><td>+0.0065</td></tr><tr><td>S&amp;P 500</td><td>10 days</td><td>+0.0021</td><td>+0.0057</td></tr></table>

Table 14: Mean single factor RankIC for Stage 2 PASS and FAIL formulas. Formula values are oriented so that larger values correspond to a stronger stated observation.

## J Auditability and Reliability of Semantic Validation

The validation procedure separates deterministic evidence construction from a rule constrained semantic judgment. The deterministic component forms factor quantile bins and computes OHLCV statistics, including location, tail, and multi statistic evidence. The validation agent then assesses whether this evidence agrees with the textual observation and its polarity. The final decision is therefore auditable as a conjunction of explicit statistical checks and semantic consistency, rather than as an unconstrained language model judgment.

<table><tr><td>Audit</td><td>Design</td><td>Cases</td></tr><tr><td>Coded-rule comparison</td><td>Validator verdicts are compared with an independent implementation of the same location, tail, multi statistic, and contradic-</td><td>8,362</td></tr><tr><td>Expert assessment</td><td>tion checklist. Experts receive the observation, formula, polarity, and induced distributional evi- dence; verdict, model identity, and returns are hidden.</td><td>100</td></tr></table>

Table 15: Reliability study design for semantic validation.

The blinded set contains 100 cases (52 CSI 500 and 48 S&P 500), balanced between 50 framework PASS and 50 framework FAIL decisions from 44 observation groups; both audits assess semantic agreement rather than portfolio return.

The reliability analyses target the semantic judgment itself rather than portfolio return. They test whether the validator’s interpretation of a fixed packet is aligned with an independently coded checklist and with blinded expert assessment.

The blinded set contains 100 cases (52 CSI 500 and 48 S&P 500), balanced between 50 framework PASS and 50 framework FAIL decisions from 44 observation groups. Both audits assess semantic agreement rather than portfolio return.

Expert assessment outcomes. Table 16 summarizes the expert assessment of semantic alignment for the blinded evidence packets. Experts judged 39 cases as agreement, 49 cases as partial agreement, and 12 cases as disagreement.

<table><tr><td>Assessment</td><td>Cases</td></tr><tr><td>Disagreement</td><td>12</td></tr><tr><td>Partial agreement</td><td>49</td></tr><tr><td>Agreement</td><td>39</td></tr></table>

Table 16: Expert assessment of semantic alignment in the blinded evaluation set.

Partial agreement is the most frequent outcome. In these cases, experts generally agreed with the observed antecedent condition and the stated directional outcome, but judged that the intermediate market mechanism required further discussion or more explicit justification. Thus, the evidence was consistent with the stated observation at the level of observable market behavior, while the mechanism connecting the condition to the expected outcome was not always fully established.

These results support the use of the validator as an auditable semantic consistency check. They do not imply that the validation procedure establishes a causal or fundamental economic mechanism. Instead, FaVOR evaluates whether a formula and its factor conditioned OHLCV evidence remain consistent with the observable condition stated in the hypothesis.

## K Longitudinal Evaluation Across Test Years

To examine whether a fixed formula combination remains useful outside a single calendar boundary, we select one combination per market using a 2015 2019 training period and a 2020 validation period. The selected formula set and quantile level are held fixed for all subsequent tests; only per ticker cutoffs are estimated from the immediately preceding validation window and then frozen during each test period.

Table 17 reports annualized excess return and excess IR across five test years. The evaluation covers the 2022 market decline as well as subsequent market environments.

<table><tr><td>Test year</td><td>CSI 500 AR</td><td>CSI 500 IR</td><td>S&amp;P 500 AR</td><td>S&amp;P 500 IR</td></tr><tr><td>2021</td><td>+0.6964</td><td>+1.7117</td><td>+0.0564</td><td>+0.4356</td></tr><tr><td>2022</td><td>+0.1356</td><td>+0.4543</td><td>+0.2377</td><td>+0.9225</td></tr><tr><td>2023</td><td>+0.3317</td><td>+1.2020</td><td>+0.3590</td><td>+2.5526</td></tr><tr><td>2024</td><td>+0.0449</td><td>+0.1259</td><td>+0.3340</td><td>+1.2453</td></tr><tr><td>2025</td><td>+0.1310</td><td>+0.5819</td><td>+0.0826</td><td>+0.5362</td></tr></table>

Table 17: Per year performance of the fixed combination selected on validation data. The combination is selected before all reported test years. For each test, cutoffs are estimated from the preceding validation window and held fixed throughout the test period.

We also evaluate the fixed combinations over 49 monthly start, 12 month test windows. Adjacent windows overlap and are therefore interpreted as start date sensitivity checks rather than independent trials.

<table><tr><td>Market</td><td>CR&gt;0</td><td>CR</td><td>IR</td><td>AR</td><td>MDD</td></tr><tr><td>CSI 500</td><td>33 / 49</td><td>+0.1931</td><td>+0.5919</td><td>+0.1894</td><td>-0.2649</td></tr><tr><td>S&amp;P 500</td><td>39 / 49</td><td>+0.1964</td><td>+0.9119</td><td>+0.1866</td><td>-0.1406</td></tr></table>

Table 18: Sensitivity of fixed validation selected combinations. Each result uses a trailing 24 month validation window and a following 12 month test window. Each metrics are arithmetic means over all 49 start test windows.

## L Hypothesis Provenance and Failure Analysis

FaVOR is not driven by a single manually selected hypothesis. Across saved runs, ten seed concepts generated 520 unique hypotheses over 696 outerloop iterations. Table 19 summarizes the progression from hypothesis generation to positive out-ofsample combinations.

<table><tr><td>Stage</td><td>All runs</td><td>Paper split</td><td>Sell-off hypothesis</td></tr><tr><td>Distinct seed concepts</td><td>10</td><td>9</td><td>1</td></tr><tr><td>Unique LLM hypotheses</td><td>520</td><td>486</td><td>67</td></tr><tr><td>Outer-loop hypothesis iterations</td><td>696</td><td>622</td><td>81</td></tr><tr><td>Candidate formulas generated</td><td>6,782</td><td>6,058</td><td>716</td></tr><tr><td>Stage 3 candidate combinations</td><td>18,699</td><td>16,741</td><td>1,148</td></tr><tr><td>Iterations passing Stage 3</td><td>49</td><td>41</td><td>5</td></tr><tr><td>Combinations with positive OOS return</td><td>1,118</td><td>832</td><td>47</td></tr></table>

Table 19: Hypothesis to strategy audit across saved runs. The sell-off row corresponds to the hypothesis used for the illustrative factor-validation case study.

Only 49 of 696 hypothesis iterations clear the Stage 3 gate. Failure is most common for mechanisms that daily OHLCV data cannot cleanly identify, including volume dominant, short covering, and liquidity driven states. This result clarifies the scope of the framework: FaVOR is designed to enforce hypothesis formula consistency for OHLCV observable conditions, not to recover every economically plausible trading mechanism.

## M Sensitivity to Selectivity and Directional Scope

We assess how global selectivity affects the universe of Stage 3 combinations. Table 20 reports the unconditional mean 2025 cumulative excess return over all 649 Stage 3 combinations under a common global selectivity level. The main method instead selects formula specific thresholds on validation data.

<table><tr><td>Global selectivity σ</td><td>CSI 500 CR</td><td>S&amp;P 500 CR</td></tr><tr><td>0.55</td><td>+0.2048</td><td>+0.0817</td></tr><tr><td>0.75</td><td>+0.0958</td><td>+0.0340</td></tr><tr><td>0.90</td><td>+0.0012</td><td>+0.0245</td></tr><tr><td>0.95</td><td>+0.0048</td><td>+0.0134</td></tr></table>

Table 20: Unconditional mean 2025 cumulative excess return across all Stage 3 combinations under a shared global selectivity level.

Directional Selectivity is intentionally not a universal profitability filter. Among 212 evaluated Stage 3 combinations, 81 were rejected because their realized direction did not remain stable as selectivity tightened. Of these rejected combinations, 47 still obtained positive after cost cumulative return in the held out evaluation. The filter therefore discards a meaningful set of profitable but directionally unstable signals, especially rebound and panic selling patterns. This exclusion is the cost of restricting FaVOR to directionally consistent signals.

## N Variance and Baseline Context

Table 21 reports sample variances and their corresponding standard deviations. FaVOR runs use the same high level hypothesis prompt and threshold configuration to isolate stochastic decoding effects, whereas RD-Agent results are based on ten independent final outputs under the common 2022 / 2023 / 2024 / 2025 split.

All compared methods use the same OHLCV derived inputs, data splits, transaction costs, and backtest environment. Their inductive priors and search capacities nevertheless differ: FaVOR uses language model priors to propose hypotheses and formulas, while ML/DL baselines use the fixed Alpha158 feature family and RL baselines search OHLCV primitives without a natural language hypothesis prior. We therefore interpret the benchmark comparison as a comparison under matched data and execution conditions, rather than as a claim of matched search budgets.

<table><tr><td>Method</td><td>Metric</td><td>Variance</td><td>Std.</td></tr><tr><td>FaVOR</td><td>Information Ratio</td><td>0.152</td><td>0.390</td></tr><tr><td>FaVOR</td><td>Annualized Return</td><td>0.042</td><td>0.205</td></tr><tr><td>FaVOR</td><td>Maximum Drawdown</td><td>0.120</td><td>0.346</td></tr><tr><td>RD-Agent</td><td>Information Ratio</td><td>0.191</td><td>0.438</td></tr><tr><td>RD-Agent</td><td>Annualized Return</td><td>0.033</td><td>0.182</td></tr><tr><td>RD-Agent</td><td>Maximum Drawdown</td><td>0.055</td><td>0.235</td></tr></table>

Table 21: Run level dispersion of FaVOR and RD-Agent. The table describes observed run to run variation and is not used to claim a statistically significant return gap.

## O Scope of the Interpretability Claim

FaVOR provides structural traceability from a natural language hypothesis to observable conditions, formulas, and factor conditioned evidence. Its interpretability claim is therefore limited to empirical consistency between a formula and the OHLCV observable condition that motivated it. The framework does not establish a causal or fundamental economic mechanism, and this distinction is important when interpreting its validation decisions and portfolio results.

## P Computational Cost

All local experiments were run on a single workstation equipped with two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96GB of memory each), two 16-core Intel64 CPUs (32 physical cores and 64 logical threads in total, dual-socket).

Closed-source backbones were accessed through commercial API endpoints; their parameter counts are not publicly disclosed by the providers. Openweight backbones were self-hosted on the local GPUs using Ollama. Llama-3.3-70B is a dense 70B-parameter model, and Qwen3-235B is a Mixture-of-Experts model with 235B total parameters and 22B active parameters per token. For API-based runs, we report token counts and monetary cost because provider-side GPU hours are not observable. For self-hosted open-weight runs, inference ran on the two local GPUs described above.

Tables 22 and 23 summarize LLM usage per call by market. Table 24 gives an example per-stage breakdown for one GPT-4o run on CSI 500. Token counts include both prompt and completion tokens, and API costs are estimated from the provider pricing used during the experiments.

<table><tr><td>Backbone</td><td>Avg tokens/call</td><td>Avg cost/call</td></tr><tr><td>GPT-40</td><td>4,287</td><td>$0.01540</td></tr><tr><td>GPT-5.4-mini</td><td>4,097</td><td>$0.00090</td></tr><tr><td>Gemini-2.5-flash</td><td>5,525</td><td>$0.00119</td></tr><tr><td>Claude-Sonnet-4.6</td><td>7,732</td><td>$0.00198</td></tr></table>

Table 22: LLM usage per call on CSI 500. Token counts include both prompt and completion tokens.
<table><tr><td>Backbone</td><td>Avg tokens/call</td><td>Avg cost/call</td></tr><tr><td>GPT-40</td><td>4,186</td><td>$0.01478</td></tr><tr><td>GPT-5.4-mini</td><td>4,171</td><td>$0.00095</td></tr><tr><td>Gemini-2.5-flash</td><td>5,335</td><td>$0.00118</td></tr><tr><td>Claude-Sonnet-4.6</td><td>7,421</td><td>$0.00186</td></tr></table>

Table 23: LLM usage per call on S&P 500. Token counts include both prompt and completion tokens.
<table><tr><td>Stage</td><td>Avg tokens/call</td><td>Avg cost/call</td></tr><tr><td>Hypothesis</td><td>1,400</td><td>$0.00433</td></tr><tr><td>Observation</td><td>1,425</td><td>$0.00487</td></tr><tr><td>Factor generation</td><td>4,525</td><td>$0.01904</td></tr><tr><td>Self-correction</td><td>5,193</td><td>$0.02211</td></tr><tr><td>Refinement</td><td>8,435</td><td>$0.02983</td></tr><tr><td>Validation</td><td>3,996</td><td>$0.01295</td></tr></table>

Table 24: Example LLM usage for one gpt-4o run on CSI 500. Reported per call within each stage; token counts include both prompt and completion tokens.

## Q Software, Models, and Data

Our backtest pipeline builds on Qlib (Yang et al., 2020) for portfolio simulation, transaction-cost accounting, and benchmark excess-return evaluation, and on Optuna (Akiba et al., 2019) for Stage 4 threshold search on the validation split. Qlib, Optuna, LightGBM (Ke et al., 2017), AlphaAgent (Tang et al., 2025), R&D-Agent-Quant (Li et al., 2025), AlphaQCM (Zhu and Zhu, 2025), and Ollama are released under the MIT license. XGBoost (Chen, 2016) is released under Apache-2.0, and scikit-learn (Pedregosa et al., 2011) is released under the BSD 3-Clause license. AlphaForge (Shi et al., 2025) is released as an open-source research codebase without a formal license, and we use it solely for non-commercial research consistent with its stated purpose.

Open-weight backbones Llama-3.3- 70B (Grattafiori et al., 2024) and Qwen3- 235B (Yang et al., 2025) are released under the Llama 3.3 Community License and Apache-2.0, respectively. Closed-source backbones GPT-4o (Hurst et al., 2024), GPT-5.4-mini (OpenAI, 2026), Gemini-2.5-Flash (Comanici et al., 2025), and Claude-Sonnet-4.6 (Anthropic, 2026) are accessed through the OpenAI, Google, and Anthropic APIs under each provider’s terms of use.

CSI 500 price and volume data are obtained via the baostock Python package, whose package metadata lists a BSD license. S&P 500 price and volume data are obtained via the yfinance Python package, whose package metadata lists the Apache Software License. The underlying Yahoo Finance data are subject to Yahoo’s terms of use and applicable data-provider restrictions; we use them only for non-commercial academic evaluation, report only derived experimental results, and do not redistribute raw market data.