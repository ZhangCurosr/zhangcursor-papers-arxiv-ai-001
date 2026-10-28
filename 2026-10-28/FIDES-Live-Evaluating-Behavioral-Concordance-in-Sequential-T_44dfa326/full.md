# FIDES-Live: Evaluating Behavioral Concordance in Sequential Trading Agents

Arther Tian<sup>a</sup>, Alex Ding<sup>a,∗</sup>, Simon Wu<sup>a</sup>, Aaron Chan<sup>a</sup>

<sup>a</sup>DGrid AI

<sup>∗</sup>Corresponding author: alex.ding@dgrid.ai

## Abstract

Sequential trading agents produce decisions, orders, execution receipts, account states, and reports that need not agree, even when the final portfolio value is correctly computed. We present FIDES-Live, an operational extension of FIDES from generated strategy concordance to the audit of sequential agent traces. The protocol separates structured decision validity, optional numerical commitment adherence, execution and ledger reconciliation, and receiptgrounded reporting, with explicit coverage and unscorable outcomes. We retrospectively audit ten requested inference routes over 7,299 cryptocurrency account-rounds and a 310-round equity pilot, using real market observations and local paper execution. Cryptocurrency decision validity is 97.35% and structured reporting accuracy is 90.81% among scorable responses, but route-level decision and reporting performance can diverge sharply. Commitment adherence is 94.36% overall; it falls to 62.63% on constraints emitted with nonempty order proposals, compared with 97.56% for valid empty proposals. All retained cash and position transitions reconcile with recorded fills, while execution controls and unavailable quotes explain separate rejection populations. Short-window marked returns provide descriptive context rather than a new test of whether concordance predicts alpha. These observations motivate reporting concordance, evidence coverage, activity, and financial outcomes separately. They do not establish long-run trading skill, semantic faithfulness of explanations, or causal benefits from audit feedback.

## 1 Introduction

An investment agent can submit a valid order, fail to execute it, and subsequently describe it as completed. It can maintain an internally consistent account while violating its own allocation commitment. It can also report every transaction correctly and still lose money. These cases concern diferent properties, yet a portfolio return or a single aggregate score can make them indistinguishable. As language models move from proposing strategies to repeatedly acting on market observations, evaluating the artifacts surrounding each action becomes a separate measurement problem.

Our earlier FIDES protocol examined concordance among a natural-language strategy, its generated implementation, and its backtested outcomes [8]. The present study follows that work but changes the measurement object. Instead of a strategy generated once and subsequently executed, we consider an agent that observes a changing market and its own holdings, proposes orders, encounters execution controls, and reports the resulting state. A discrepancy can originate in the proposal, the environment, execution, or reporting. Assigning all unsuccessful interactions to the model obscures this distinction; assigning every correct final balance to model competence obscures it in the other direction.

Live and sequential financial evaluation is already an established research setting [1, 3, 12]. Our contribution is a narrower operational audit. FIDES-Live reconstructs a trace linking observations, decisions, explicit commitments, proposed orders, receipts, account states, and reports. Deterministic checks test the properties supported by those artifacts. Decision validity tests the action contract. Commitment checks test numerical constraints the model actually states. Ledger reconciliation checks arithmetic against recorded fills. Reporting checks compare structured factual claims with the receipt provided to the model. Missing evidence and explicitly unavailable calls remain visible rather than receiving a convenient score.

We apply the protocol retrospectively to two forward-running paper-trading environments: a cryptocurrency spot environment with ten requested model routes and an equity pilot using the same roster. The retained corpus comprises 7,609 account-rounds: 7,299 cryptocurrency observations across eight UTC dates and 310 equity observations across two trading days. All market interactions use real public data, but fills are produced by local paper executors. The environments difer in their risk controls, advisory signals, costs, and scheduling; we analyze them separately. Chronological account-rounds are dependent observations, not thousands of independent market experiments.

Three results motivate the audit. First, valid decisions and accurate reports separate sharply for some inference routes: in cryptocurrency traces, the DeepSeek route has 98.73% decision validity but 14.33% structured reporting accuracy among scorable responses. This is a finding about the recorded route and requested format, not a claim about immutable model weights or financial reasoning. Second, optional commitment adherence is dificult to interpret without coverage and activity. Cryptocurrency adherence is 94.36% overall, but 62.63% on constraints emitted with nonempty proposals and 97.56% with valid empty proposals. Third, execution failures and accounting errors do not coincide. All retained cash and position transitions reconcile with recorded fills, although the equity pilot contains 100 orders rejected for market unavailability and the cryptocurrency executor rejects 117 orders through its proposal gate.

## Contributions.

• An operational concordance protocol for sequential trading traces, with explicit eligibility, commitment coverage, and separate decision, execution, and reporting measurements.

• A provenance-tracked retrospective audit of two forward-running environments, with reproducible aggregation from normalized outcomes, file-integrity checks, timestamp-based sequence reconstruction, and conservative treatment of unavailable valuation evidence.

• An empirical decomposition showing how report format failures, optional commitments, activity, and execution controls change the interpretation of aggregate reliability measures.

The study establishes measurements and observed discrepancies. It does not establish profitable trading, internal reasoning faithfulness, deliberate deception, or a causal benefit from providing audit feedback.

## 2 Related Work

Trading agents and live evaluation. Financial agent research increasingly evaluates actions and portfolios rather than isolated financial answers. TradingAgents organizes analysis, deliberation, trading, and risk management into specialized roles [10]. StockBench studies sequential stock decisions with market, fundamental, and news inputs [1]. Agent Market Arena standardizes live inputs and execution feedback across agent architectures in equity and cryptocurrency environments [3]. LiveTradeBench extends live evaluation to portfolio allocation across stock and prediction markets [12]. Thus, live data, sequential decisions, and multiple markets are established evaluation settings. AlphaForgeBench further diagnoses unstable action sequences and instead evaluates generated factors and strategy implementations through deterministic execution [13]. FIDES-Live retains sequential agent actions and examines agreement among the artifacts surrounding them. Its purpose is to complement financial outcome evaluation with an inspectable operational audit rather than establish a new claim to live-market evaluation.

![](images/1148a205a852c2033f722395fcf54053f41adeb8ccdba27ce64430168bd7737b.jpg)  
Figure 1: The FIDES-Live measurement object. Intended orders, execution outcomes, and reported facts are distinct artifacts. The deterministic audit checks supported properties and preserves unavailable evidence; it does not infer hidden reasoning from a correct account balance.

State-grounded reliability and explanation faithfulness. AgentBench evaluates reasoning and action across interactive environments, including the failure modes arising during interaction [2]. The state-based evaluation in τ-bench compares final database states with annotated goals and measures reliability across repeated trials [11]. Our audit similarly uses observable state as evidence, but consecutive trading rounds are not independent repetitions of one task; they cannot support a direct interpretation as pass<sup>k</sup>. A separate literature shows that plausible explanations can omit factors influencing model outputs [9]. Agreement with a ledger is a narrower property: it establishes whether checkable claims match observable records, not whether a narrative reveals the model’s internal causal reasoning. We therefore use concordance for artifact agreement and avoid interpreting detected discrepancies as evidence of deliberate deception.

From FIDES to sequential operational concordance. FIDES reconciles natural-language strategy descriptions, generated code, and backtested outcomes through three concordance dimensions [8]. FIDES-Live extends the measurement object from one generated strategy to a sequence of decisions, submitted orders, execution receipts, account states, and reports. This changes the attribution problem: a valid decision may encounter unavailable quotes or rejected execution, while an inaccurate report may incorrectly describe what subsequently occurred. We keep these events distinguishable. Moreover, factual accuracy after observing a receipt does not validate an earlier investment forecast. The numerical checks reported here assess the structured commitments and factual fields supported by the retained traces; they do not constitute a complete semantic evaluation of every rationale.

Measurement before aggregation. Our earlier PoQ work examines evaluator robustness, dimension reliability, and evaluation cost [4–6]. SFGA applies a related principle: inexpensive measurements should be used within their evidential limits, with uncertainty made explicit [7]. FIDES-Live follows this measurement philosophy by prioritizing deterministic, inspectable checks and reporting eligible denominators and evidence coverage. These connections motivate design choices; they do not imply that the present corpus was produced with PoQ consensus, a trained PoQ-Judge, or SFGA adjudication. We report operational dimensions separately rather than treating an unvalidated weighted composite as an established measure of overall agent quality.

<table><tr><td>Property</td><td>Earlier FIDES</td><td>FIDES-Live</td></tr><tr><td>Observation unit</td><td>One generated strategy; 40 outputs</td><td>One retained account-round; 7,609 rounds</td></tr><tr><td>Evidence</td><td>Strategy description, generated code, historical backtest</td><td>Decision, commitments, order receipts, account state, report</td></tr><tr><td>Measurements</td><td>Language-code, implementation, and claim-outcome concordance</td><td>Action-contract validity, optional commitments, ledger reconciliation,</td></tr><tr><td>Financial setting</td><td>Historical out-of-sample strategy evaluation</td><td>structured report agreement Forward-running local paper execution on real market observations</td></tr></table>

Table 1: The studies share an audit perspective but use diferent observations and measurement definitions. Their scores are not directly comparable.

Relationship to the preceding FIDES study. Table 1 makes the change in measurement explicit. The preceding study evaluates 40 generated strategy outputs using historical backtests [8]; the present corpus contains newly recorded forward-running account traces and does not reuse those strategy outputs as observations. This is a continuation of the measurement perspective, rather than a replication of its three scores. Action-contract validity does not replace the earlier language–code comparison, and accurate reporting after seeing a receipt does not validate an earlier forecast. Consequently, scores from the two protocols are not directly comparable. The separation between concordance and financial performance is inherited motivation, not a new claim of statistica independence or profitable trading. The new empirical focus is the attribution of discrepancies across sequential proposals, execution controls, marked states, and structured reports, with explicit eligibility and coverage. Neither a semantic explanation benchmark nor an estimate of financial alpha is supplied by these measurements.

## 3 An Operational Concordance Protocol

## 3.1 The trace as the measurement object

An agent round produces several artifacts that should describe one interaction but need not agree. For model m and round t, write

$$
\tau _ { m , t } = ( o _ { m , t } , s _ { m , t } ^ { - } , d _ { m , t } , u _ { m , t } , e _ { m , t } , s _ { m , t } ^ { + } , r _ { m , t } ) ,\tag{1}
$$

where $o _ { m , t }$ is the supplied market observation, s<sup>−</sup> the pre-execution account, d the decision response, u its proposed orders, e the execution receipts, $s ^ { + }$ the post-execution account, and r the subsequent report. The observation can difer across accounts, even within a scheduled interval. The decision can contain optional numerical commitments. Orders and receipts are distinct: an intended buy can be rejected and leave the account unchanged. The report observes the receipt and cannot place further orders. Its accuracy is therefore a post-execution factual property, not a test of an earlier investment prediction.

The unit of analysis is an account-round, not an API call or an independent market trial. Accounts maintain separate cash and holdings, while observations and market shocks can be shared.

We evaluate the equity and cryptocurrency environments separately. Diferences in their inputs and execution controls prevent interpreting pooled counts as one homogeneous treatment.

## 3.2 Evidence reconstruction and eligibility

We recompute measurements ofline from retained round artifacts rather than reuse dashboard scores. The reconstruction records each source’s SHA-256 digest and preserves its decision response, receipts, account states, report, timestamps, and available call metadata. Indexed cryptocurrency round artifacts are checked against their recorded hashes and sizes. Summaries are not substitutes for full artifacts. Synthetic demonstrations and historical replay runs are excluded. Per-account sequences are ordered by recorded event timestamps; an ordinal that reverses the temporal sequence is a provenance anomaly to disclose, not a reason to silently rewrite its contents.

Each applicable check has three outcomes: pass, fail, or unscorable. An available response that violates the requested schema is a failure. Explicit api\_error, agent\_error, and interrupted statuses are treated as call unavailability and excluded from conditional content denominators; these labels do not identify which upstream component caused the failure. An unattempted report has no report-content outcome. We retain unavailable and unattempted cases in accounting totals. Execution rejection is separately classified from decision validity and report accuracy.

Let N be retained account-rounds, D the rounds with scorable decision responses, and R those with scorable attempted reports. With pass indicators $v _ { i }$ and $a _ { i }$ , respectively, we report

$$
V = \frac { \sum _ { i \in \mathcal { D } } v _ { i } } { | \mathcal { D } | } , \qquad A = \frac { \sum _ { i \in \mathcal { R } } a _ { i } } { | \mathcal { R } | } .\tag{2}
$$

We also give unavailable counts and successful outputs divided by N, termed retained-round yield. These quantities describe availability-adjusted output success within committed retained rounds, not completion probabilities across every started execution attempt. Attempts that never produce a retained round are outside this denominator. Primary rates pool eligible observations (microaveraging); equal-route macro-averages give each of the ten requested routes one weight. Empty denominators remain undefined. Neither unavailable evidence nor an absent optional commitment receives automatic credit.

## 3.3 Decision validity and explicit commitments

A valid decision is a JSON object containing a nonempty rationale, an orders array, and at most an optional commitments object. Each order has exactly an allowed symbol, a buy/sell side, and a finite positive quantity; its count must respect the recorded configuration. An empty array is valid. Commitments admit only finite fractions in [0, 1] for min\_cash\_ratio and max\_position\_ratio. Additional keys, malformed types, and nonfinite values fail validation. A syntactically valid order can still be unafordable or rejected. Consequently, V measures structured decision validity, not investment quality or semantic agreement between rationale and action. Order totals count orders in contract-valid decisions only; malformed or unavailable decisions are neither valid empty proposals nor admitted nonempty proposals. Recognizable order objects inside an invalid response do not enter the admitted-order population.

For a valid decision, let $\mathcal { C } _ { i } ^ { \mathrm { d e c l } }$ contain its declared nontrivial commitments and ${ \mathcal { C } } _ { i } \subseteq { \mathcal { C } } _ { i } ^ { \mathrm { d e c l } }$ the evaluable subset. Given post-execution cash $c _ { i } ,$ equity $E _ { i } > 0$ , quantities $q _ { i , s }$ , and execution-midpoint prices $p _ { i , s } ,$ , define $w _ { i , s } = q _ { i , s } p _ { i , s } / E _ { i }$ . Stored weights are cross-checked against this construction wherever complete price evidence is available. The two supported checks are

$$
c _ { i } / E _ { i } + \epsilon \ge \ell _ { i }\tag{3}
$$

$$
\operatorname* { m a x } \bigl ( \{ w _ { i , s } : s \in \mathrm { h o l d i n g s } _ { i } \} \cup \{ 0 \} \bigr ) \leq h _ { i } + \epsilon \qquad \mathrm { ( m a x i m u m ~ p o s i t i o n ) , }\tag{4}
$$

with $\epsilon = 1 0 ^ { - 9 }$ . They refer to the recorded post-execution marked state, including actual rejections, rather than a hypothetical full fill or future return. A zero cash floor and a unit position ceiling follow from the long-only, unlevered rules; we count them separately as trivial and exclude them from substantive compliance.

Commitments are evaluable only when the state has finite cash, positive finite equity, and finite position weights, with no recorded market/execution error or stale-state flag. Each held asset must have a finite positive uncrossed bid–ask quote, spread no greater than 100 basis points, and a timestamp between zero and 90 seconds before the execution snapshot in cryptocurrency, or 120 seconds in equities. These are project-defined eligibility thresholds, not universal financial benchmark standards. Otherwise the declared constraint is unscorable; a stale mark does not establish either compliance or violation.

Let V be valid retained decisions. With $z _ { i , c }$ indicating that constraint c passes, commitment coverage and compliance are

$$
\mathrm { C o v e r a g e } = \frac { \sum _ { i \in \mathcal { V } } \mathbf { 1 } [ | \mathcal { C } _ { i } ^ { \mathrm { d e c l } } | > 0 ] } { | \mathcal { V } | } , \qquad \mathrm { C o m p l i a n c e } = \frac { \sum _ { i \in \mathcal { V } } \sum _ { c \in \mathcal { C } _ { i } } z _ { i , c } } { \sum _ { i \in \mathcal { V } } | \mathcal { C } _ { i } | } .\tag{5}
$$

Coverage includes declarations whose marked state is unscorable; the diference $\begin{array} { r } { \sum _ { i } ( | \mathcal { C } _ { i } ^ { \mathrm { d e c l } } | - | \mathcal { C } _ { i } | ) } \end{array}$ is reported as unknown. The two measures answer diferent questions. High compliance at low coverage is weak evidence about general adherence; emitting no commitment is not a successful adherence test. Platform-imposed risk limits are not voluntary model commitments.

## 3.4 Receipt-grounded reporting

A report must contain exactly summary, cash, equity, positions, and filled\_orders. The summary must be a nonempty string. Cash and equity must agree with the supplied post-execution account within an absolute tolerance of 0.01 monetary units. The positions map must have exactly the recorded symbol set, with every quantity within $1 0 ^ { - 6 }$ asset units. The filled-order count must be an integer, excluding Boolean values, equal to the number of receipts marked filled in that round. A report passes only when its structure and all four factual fields pass. Component failures can overlap and are reported separately.

These checks evaluate structured claims against the receipt even when its market valuation is stale. Receipt agreement and current-market valuation quality are distinct properties. The checks do not semantically score the summary, compare the rationale’s prose with orders, infer hidden reasoning, or determine whether an error was intentional. No language-model judge or human semantic annotation is used for the reported measurements. A correct structured report can therefore coexist with an unchecked narrative inconsistency.

## 3.5 Execution and ledger reconciliation

We distinguish proposed, filled, and rejected orders and retain recorded rejection reasons and available pre-execution gate checks. A gate can block the whole proposal; an absence of fills is consequently not, by itself, evidence of inactivity or a model error. Gate-pass rates and filled/proposed counts are execution descriptors, not additional components of V or A.

To check the state anchor independently of legacy evaluation flags, we reconstruct cash and positions from the pre-execution account and recorded filled orders. For filled order $j ,$ , let $\sigma _ { j } = 1$ for a buy and −1 for a sell, $n _ { j }$ its recorded notional, $q _ { j }$ its quantity, and $f _ { j }$ its fee. The reconstructed state satisfies

$$
\widehat { c } _ { i } ^ { + } = c _ { i } ^ { - } - \sum _ { j } ( \sigma _ { j } n _ { j } + f _ { j } ) ,\tag{6}
$$

$$
\displaystyle \widetilde { q _ { i , s } } = q _ { i , s } ^ { - } + \sum _ { j : s _ { j } = s } \sigma _ { j } q _ { j } .\tag{7}
$$

We compare reconstructed cash with the recorded state within 10 $^ { - 5 }$ monetary units and quantities within $1 0 ^ { - 6 }$ asset units, discarding position dust of magnitude at most $1 0 ^ { - 9 }$ . Where prices are available, we also check $\begin{array} { r } { E _ { i } = c _ { i } + \sum _ { s } q _ { i , s } p _ { i , s } } \end{array}$ within $1 0 ^ { - 5 }$ and reconstructed position weights within 10<sup>−9</sup>. Recorded account prices take precedence; execution bid–ask midpoints provide a fallback. Reconciliation establishes internal accounting agreement with recorded fills and marks. It does not attest exchange execution, quote freshness, or realistic liquidity. Returns and token usage remain separate descriptive outcomes; we introduce no weighted overall quality score.

For the settled session, marked return is $R _ { m } = E _ { m } ^ { \mathrm { e n d } } / E _ { m } ^ { \mathrm { i n i t i a l } } - 1$ and final cash fraction is $c _ { m } ^ { \mathrm { e n d } } / E _ { m } ^ { \mathrm { e n d } }$ . Initial allocations and the common eligible saved endpoint determine the return; no completeness of the intervening valuation path is assumed.

## 4 Corpus and Experimental Setting

## 4.1 Two forward-running environments

Cryptocurrency spot. The main corpus records decisions from 2026-09-30 03:44:09 to 2026-10-07 22:14:24 UTC. The universe is BTC, ETH, SOL, BNB, DOGE, and XRP. Public Binance spot bid–ask snapshots supply observations and execution references. Quotes are denominated in USDT; account displays assume one USDT equals one USD. This assumption ignores stablecoin basis risk. Snapshot timestamps identify HTTP snapshots, not independently verified exchange event times. Each model begins each seven-day session with a separate \$10,000 paper account. One session is settled; the following session is only partially observed. We retain 6,608 account-rounds from the first and 691 from the second, with no attempt to impute the unfinished period.

The intended decision interval is 900 seconds, with account marks approximately every 30 seconds. Agents receive up to 90 daily historical bars per asset, recent account history, and a deterministic momentum/volatility analysis. The analysis is advisory rather than an LSTM or other trained forecasting model. The recorded method combines 20-period momentum and the observed spot move, scaled by 20-period return volatility. A fixed balanced policy requires at least 20% cash, at most 30% in one asset, and at most 80% total asset exposure. These limits belong to the platform policy, not to voluntary model commitments.

A pre-execution gate previews the entire proposal and checks schema, quotes, afordability, holdings, and risk limits. A failed check rejects the whole proposal without resizing individual orders. Even an empty proposal can fail when existing holdings drift beyond a risk limit. Long-only fractional quantities are allowed; shorting, borrowing, leverage, and external tools are unavailable. Filled buys use the ask and filled sells the bid, with 2 basis points of adverse slippage and a 10-basis-point notional fee. The maximum proposal contains 12 orders. Quote age is limited to 90 seconds and spread to 100 basis points. These are configured experimental assumptions, not universal benchmark standards or exchange fee tiers.

Equity pilot. The equity traces cover September 28–29, 2026. One empty initialization contains no account-rounds and is excluded from outcome denominators; two populated runs contain 5 and 26 joint decision rounds, respectively, or 310 account-rounds. The first uses SPY, QQQ, IWM, EFA, TLT, GLD, XLF, and XLE at a five-minute decision interval. The second adds AAPL, MSFT, NVDA, AMZN, GOOGL, META, and TSLA and uses 15-minute decisions during the US regular session, with 30-second marks. Alpaca IEX data provide bid–ask references. Each populated run starts independent \$10,000 model accounts with up to 30 historical daily bars. This operational evolution is retained rather than described as one frozen equity treatment.

The equity system supplies no mandatory balanced allocation gate or advisory momentum analysis. Its executor enforces cash and long-only holdings, charges 1 basis point in fees and 2 basis points of adverse slippage, admits at most 20 orders, and uses a 120-second quote-age limit; the main run records a 100-basis-point spread limit. In the observed unavailable-market rounds, the quote bundle is unavailable and the proposed orders receive explicit rejection receipts. A joint round shares an observation across models and waits for the decision batch before retrieving execution quotes; the cryptocurrency system instead records account-specific observation and execution snapshots. These diferences rule out interpreting the two environments as a controlled test of market transfer.

## 4.2 Inference routes and requested output

The ten requested routes are GPT-4.1 mini, GPT-5.4, DeepSeek V3.2, Gemini 2.5 Flash, Claude Haiku 4.5, Qwen3 235B A22B Instruct 2507, Kimi K2.5, GLM-5.3, MiniMax M2.5, and Grok 4.20 non-reasoning. Table 7 supplies exact request identifiers. They are accessed through a compatible chat-completion gateway. Returned identifiers are retained, but do not attest fixed model weights; observed aliases include multiple DeepSeek and Claude identifiers. Consequently, model names in tables abbreviate requested routes.

Each account-round normally involves one decision request followed by one report request. Prompts permit empty orders and require a nonempty rationale; optional numerical commitments are explicitly distinguished from future-return predictions. The report receives the execution receipt and account state. No hidden repair call is added during this audit. The recorded maximum output allowance is 8,192 tokens with a 120-second timeout for cryptocurrency and 4,096 tokens with a 90-second timeout for the main equity run. No fixed sampling seed or weight-level version attestation is established by the retained evidence.

## 4.3 Data reconstruction and validation

The cryptocurrency database indexes 7,299 round artifacts and 14,834 call artifacts. Every indexed round and call file is present and matches its recorded SHA-256 digest and byte length. The call table contains 14,833 entries: 14,831 completed lifecycle records and two interrupted records. One additional artifact is an orphan started report, retained as provenance evidence but excluded from completed-call totals. A completed lifecycle is not necessarily a successful model response. The call population also includes attempts not represented by a committed round; it is not forced into twice the round count.

The database lists 840 unavailable ancillary attempt journals. Their associated committed round evidence is present, so they do not represent 840 missing executions. They do limit complete reconstruction outside committed traces. Historical compact database rows are expanded from full artifacts, which contain the original observations, decisions, receipts, and reports. The analysis excludes demonstration and historical replay runs, copied equity archives, and personal-use fixtures.

<table><tr><td>Environment</td><td>N</td><td>Valid decisions</td><td>Accurate reports</td><td>Kept commitments</td><td>Unknown</td></tr><tr><td>Crypto</td><td>7,299</td><td>7,052/7,244</td><td>6,597/7,265</td><td>2,892/3,065</td><td>2</td></tr><tr><td>Equities</td><td>310</td><td>297/309</td><td>274/309</td><td>196/202</td><td>53</td></tr></table>

Table 2: Recomputed primary counts. Decision and report denominators exclude explicitly unavailable calls. Commitment denominators include only nontrivial constraints with eligible marked-state evidence. Unknown refers to declared constraints excluded for unavailable or stale valuation evidence.

One recovered cryptocurrency record has a round ordinal inconsistent with its timestamp and original call payload. Sorting by recorded time restores cash and position continuity, so we retain the observation and disclose the anomaly. There are no cash or position discontinuities across 7,279 consecutive cryptocurrency account transitions or 290 equity transitions after this normalization. Hashes establish consistency with the stored index, not external notarization. All reported checks are recomputed ofline; legacy decision-validity and report-accuracy flags agree with the recomputation, whereas conservative valuation eligibility removes 55 commitment checks from the compliance denominator.

We additionally reparse all retained no-error completed cryptocurrency call responses (14,034) and no-error equity participant responses (584) using the runtime’s JSON rules, including fence handling, duplicate-key rejection, and finite-number checks. Every result agrees with its retained parsed object. The primary audit uses these verified parsed objects while preserving recorded parser failures. The audit implementation is exercised by 36 deterministic controls covering valid receipts, altered balances, holdings and weights, wrong filled-order counts, malformed schemas, invalid nonempty responses, unavailable calls, missing, future, stale, crossed, or wide quotes, and noimputation/gap-breaking figure behavior. All controls pass. These controls validate implementation behavior; they are not additional model experiments or measurements of semantic detection accuracy.

Analysis policy. We report exact counts and descriptive rates. We do not use account-rounds as independent samples for significance testing, annualize a one-week return, or infer a stable model ranking. The experiment was not preregistered. Session-level and activity-conditioned tables make observation imbalance and composition visible. All displayed tables and numerical macros are generated from the normalized audit records.

## 5 Results

## 5.1 A valid decision does not ensure an accurate report

Table 2 gives the primary denominators. Cryptocurrency decision validity is 7,052/7,244 (97.35%), excluding 55 explicitly unavailable decision calls. Structured report accuracy is 6,597/7,265 (90.81%), excluding 33 API failures and one interrupted report. Using all retained account-rounds as the denominator yields 96.62% valid decisions and 90.38% accurate reports. These retained-round yields include unavailable calls in committed rounds but omit attempts that never produced a committed round; they are not all-attempt completion probabilities. Equal-route macro-averages are 97.34% for decision validity and 89.66% for report accuracy, compared with the primary pooled rates. Unequal observation counts therefore afect the aggregation, especially for reporting.

Diferences between requested routes are substantial (Table 3, Figure 2). The DeepSeek route produces valid decisions on 620/628 scorable calls, but accurate reports on only 90/628. The Claude route has 556/726 valid decisions and 605/732 accurate reports. All other routes reach 99.73–100% report accuracy. These are descriptive results under this gateway, prompt, and observation window. They do not establish an immutable ordering of the underlying model families.

<table><tr><td>Model</td><td>N</td><td>V (%)</td><td>A (%)</td><td>Cover. (%)</td><td>Adhere. (%)</td><td>Fills</td></tr><tr><td>Claude Haiku 4.5</td><td>739</td><td>76.58</td><td>82.65</td><td>99.28</td><td>98.09</td><td>38</td></tr><tr><td>DeepSeek V3.2</td><td>631</td><td>98.73</td><td>14.33</td><td>0.48</td><td>100.00</td><td>9</td></tr><tr><td>Gemini 2.5 Flash</td><td>740</td><td>99.86</td><td>99.86</td><td>10.20</td><td>100.00</td><td>5</td></tr><tr><td>MiniMax M2.5</td><td>741</td><td>100.00</td><td>100.00</td><td>1.08</td><td>100.00</td><td>49</td></tr><tr><td>Kimi K2.5</td><td>741</td><td>100.00</td><td>100.00</td><td>4.19</td><td>85.25</td><td>39</td></tr><tr><td>GPT-4.1 mini</td><td>742</td><td>100.00</td><td>99.73</td><td>31.30</td><td>90.04</td><td>6</td></tr><tr><td>GPT-5.4</td><td>741</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.80</td><td>35</td></tr><tr><td>Qwen3 235B</td><td>742</td><td>100.00</td><td>100.00</td><td>0.68</td><td>100.00</td><td>3</td></tr><tr><td>Grok 4.20</td><td>741</td><td>100.00</td><td>100.00</td><td>17.49</td><td>62.89</td><td>10</td></tr><tr><td>GLM-5.3</td><td>741</td><td>98.22</td><td>100.00</td><td>0.28</td><td>100.00</td><td>27</td></tr></table>

Table 3: Cryptocurrency audit by requested model route, combining the settled and partial sessions. V and A use their own scorable denominators. Coverage is the percentage of valid decisions declaring a nontrivial commitment; adherence is per scorable constraint, not per round. Fills are local simulated executions. Exact numerators and denominators appear in Appendix Table 11 and the accompanying data.

The 192 scorable cryptocurrency decision failures comprise 179 recorded JSON-parse errors, seven truncated outputs, and six action-contract violations after parsing. The 668 scorable report failures comprise 395 JSON-parse errors, one truncated output, one empty response, and 271 parsed reports failing the requested fields. Accordingly, poor reporting performance cannot be attributed exclusively to incorrect financial arithmetic: format delivery is itself a large part of the measured failure. Across the 6,868 parsed reports without a recorded response error, 6,597 pass all report checks (96.05%). This conditional view describes a diferent, selected population and is not a replacement for the primary accuracy rate.

## 5.2 Commitment coverage and activity change the interpretation

Cryptocurrency agents declare 3,067 nontrivial constraints in 1,772 valid decision rounds: coverage is 25.13%. Two constraints have stale or unavailable execution-state evidence and remain unscorable. Of the remaining 3,065, 2,892 pass (94.36%). No emitted constraints are excluded as trivial tautologies in this corpus.

The aggregate conceals two forms of selection. First, commitment coverage varies from 0.28% for GLM-5.3 to 100% for GPT-5.4. A 100% adherence figure based on four GLM constraints or six DeepSeek constraints does not establish the same breadth of evidence as GPT-5.4’s 1,007 constraints. Second, activity changes the observed adherence distribution. Only 219/7,052 valid cryptocurrency decisions contain an order (3.11%). On those rounds, 176/281 constraints pass (62.63%); on valid empty proposals, 2,716/2,784 pass (97.56%). Figure 3 and Table 4 expose this diference. It is observational: diferent models, constraints, and account states populate the strata, so it is not an estimated causal efect of trading.

Commitment violations need not imply a new unsafe trade. In the rejected-proposal stratum, only 37/131 commitments pass: the final portfolio can remain unchanged after a gate rejection while failing a promise about the intended post-execution allocation. Conversely, an empty proposal can preserve a compliant portfolio. This is why we check explicit promises against the actual recorded state, while separately retaining the proposed orders and gate outcome.

![](images/da6f6c7950b7b9c3762058b38539d371d119ece38d25dcc1d6dc65a2fa1c9374.jpg)

Figure 2: Decision validity and structured report accuracy are distinct operational properties. Rates are conditional on scorable responses; unequal sample sizes and infrastructure failures remain visible in the tables.
<table><tr><td>Env.</td><td>Round stratum</td><td>N</td><td>Reports</td><td>%</td><td>Commitments</td><td>%</td></tr><tr><td>Crypto</td><td>Nonempty proposal</td><td>219</td><td>205/219</td><td>93.61</td><td>176/281</td><td>62.63</td></tr><tr><td>Crypto</td><td>Valid empty proposal</td><td>6833</td><td>6258/6812</td><td>91.87</td><td>2716/2784</td><td>97.56</td></tr><tr><td>Crypto</td><td>Any rejection</td><td>87</td><td>82/87</td><td>94.25</td><td>37/131</td><td>28.24</td></tr><tr><td>Equities</td><td>Nonempty proposal</td><td>53</td><td>48/52</td><td>92.31</td><td>25/31</td><td>80.65</td></tr><tr><td>Equities</td><td>Valid empty proposal</td><td>244</td><td>221/244</td><td>90.57</td><td>171/171</td><td>100.00</td></tr><tr><td>Equities</td><td>Any rejection</td><td>25</td><td>24/25</td><td>96.00</td><td>0/0</td><td></td></tr></table>

Table 4: Activity-conditioned audit. Reports and commitments have diferent eligibility denominators. Any-rejection strata can overlap other strata; these rows are not a partition to sum. Undefined adherence is shown as a dash. An empty valid proposal is distinct from a malformed or unavailable decision.

## 5.3 Execution controls and ledger fidelity are diferent measurements

The cryptocurrency traces contain 338 orders in contract-valid proposals, of which 221 fill and 117 are rejected. All recorded rejections carry the gate’s quality\_rejected reason. The gate fails on 1,094 account-rounds: 760 valid empty proposals, 87 valid nonempty proposals, and 247 invalid or unavailable decision responses (Table 5). The last category must not be interpreted as an empty valid decision. Four parsed invalid cryptocurrency responses contain 13 recognizable order objects, excluded from admitted-order totals because the full decision violates the action contract. Unparseable responses may contain further uninterpretable intent. The failed-gate count is therefore neither a rejected-order count nor a count of 1,094 attempted trades.

Gate checks are not mutually exclusive. Recorded failures include 594 maximum-position checks,

![](images/236ca87b3e0b542a2e8aca2a74a7b4a8fbc318f7c907008e61387484d28cd201.jpg)

![](images/56650ec3b5aff54f056a479dccae678b116b75f1326a1fdfbc60f877ed9ab0f8.jpg)

Figure 3: Left: declared nontrivial commitment coverage among valid cryptocurrency decisions. Right: compliance among scorable constraints, conditioned on a nonempty or valid empty proposal. Counts above bars are passed/evaluated constraints. Empty proposals dominate the corpus and can conceal materially weaker adherence on active rounds.
<table><tr><td>Decision category</td><td>Account-rounds</td><td>Gate pass</td><td>Gate fail</td></tr><tr><td>Valid nonempty</td><td>219</td><td>132</td><td>87</td></tr><tr><td>Valid empty</td><td>6833</td><td>6073</td><td>760</td></tr><tr><td>Invalid or unavailable</td><td>247</td><td>0</td><td>247</td></tr></table>

Table 5: Cryptocurrency gate populations. Contract-invalid or unavailable responses form a third input category; zero admitted orders in that category does not establish a valid choice to do nothing. Counts partition retained account-rounds.

252 cash-floor checks, 252 total-exposure checks, five funds-and-positions checks, and one fresh-quote check, alongside 247 invalid or unavailable decision-schema cases. Within valid empty proposals alone, failed checks comprise 524 maximum-position, 235 cash-floor, 235 total-exposure, and one fresh-quote check, with overlap among risk checks. Risk-limit failures in this subgroup concern an existing marked state without a newly submitted order; they are distinct from quote unavailability and invalid input. Existing holdings can cross a limit as prices change. The gate also changes which actions become fills, preventing an attribution of the observed portfolio path solely to the language model.

Every retained account-round reconciles cash and positions against recorded filled orders: 7,299/7,299 in cryptocurrency and 310/310 in equities. Equity arithmetic has complete executionprice evidence on 279/310 rounds, all of which pass; the remaining 31 have no complete mark evidence and receive no equity-arithmetic score. Cryptocurrency equity arithmetic passes on all 7,299 recorded states, including a stale marked state whose numerical equality is valid but whose two commitments are conservatively unscorable. Internal arithmetic agreement and current market validity are separate checks.

## 5.4 The equity pilot illustrates environment-error attribution

The smaller equity corpus yields 297/309 valid decisions (96.12%) and 274/309 accurate reports (88.67%). One decision call and one report call are explicitly unavailable. Of 168 orders in contractvalid proposals, 63 fill, 100 receive market\_unavailable, and five are rejected for insuficient cash. One additional invalid parsed response contains three recognizable order objects and is excluded from the admitted-order population. Reporting remains independently evaluable from the supplied receipt: 24/25 scorable reports on rounds containing a rejection are accurate.

Agents declare 255 nontrivial constraints in 170 valid rounds, but 53 lack eligible execution-state evidence. Scoring those against carried-forward balances would turn unavailable evidence into apparent success. With those checks excluded, 196/202 constraints pass (97.03%). The activeproposal stratum has 25/31 passed constraints, compared with 171/171 on valid empty proposals. These counts echo the importance of activity conditioning, but the diferent windows and controls do not support a cross-market generalization claim.

## 5.5 A concrete factual discrepancy

One retained DeepSeek report on 2026-10-03 04:42:49 UTC correctly reproduces cash of \$8,201.03107724632, equity of \$9,985.95607724632, and a position of 15 SOL. Its summary states that no orders were submitted or executed. The recorded order list is empty, yet filled\_orders equals two. The report therefore fails exactly the round-specific filled-order count while passing cash, equity, positions, and schema. A second record returns 1,584,655,368 as that count on another empty-order round. These cases demonstrate the value of field-level reconciliation; they do not identify a mechanism for the error or intentional misrepresentation. Case identifiers and source digests are supplied in the reproducibility package.

## 5.6 Accurate reporting and short-window returns answer diferent questions

For a common marked outcome, we use the settled cryptocurrency session alone. Each account’s latest eligible saved valuation is at 2026-10-07 03:42:24 UTC, approximately five seconds before the recorded deadline. We independently reconstruct all ten endpoint equities from the last saved cash and holdings and one common cached quote bundle, whose recorded quote and receipt times precede the deadline; the maximum absolute discrepancy is below $2 \times 1 0 ^ { - 1 2 }$ . The source artifact carrying that cached bundle was written after the deadline, so this is consistency with recorded pre-deadline quote metadata rather than independent proof of a contemporaneous commitment. Table 6 reports full-window marked returns from the recorded initial allocation, after recorded fees and without forced liquidation. All ten returns are negative, ranging from −1.448% to −0.157%. Seven routes have 100% conditional reporting accuracy during this settled session, yet each loses value over the same window. A correct account report is therefore not suficient evidence of a positive return in these observations.

Figure 4 shows a narrower path: only saved valuations explicitly flagged nonstale. The first eligible point is 2026-10-01 00:31:24 UTC, about 20 hours 49 minutes after the session starts. Earlier marks have unknown staleness and are excluded, not imputed. A later saved-mark gap on October 5 spans approximately 97 minutes. The figure shades the initial exclusion and breaks curves across source gaps longer than five minutes. Endpoint returns remain defined by initial cash and final eligible state; an incomplete observed path must not be presented as a complete seven-day valuation history.

We do not infer that concordance is uncorrelated with returns. Ten dependent portfolio paths over one week cannot establish that conclusion. Nor do we benchmark investment skill against a passive market portfolio here. The observation supports the narrower distinction already built into the protocol: account-reporting correctness and investment outcomes should remain separate evaluation axes.

<table><tr><td>Model</td><td>N</td><td>Return (%)</td><td>A (%)</td><td>Cash (%)</td></tr><tr><td>Claude Haiku 4.5</td><td>664</td><td>-0.970</td><td>80.67</td><td>27.4</td></tr><tr><td>DeepSeek V3.2</td><td>615</td><td>-0.691</td><td>14.38</td><td>71.4</td></tr><tr><td>Gemini 2.5 Flash</td><td>665</td><td>-0.313</td><td>99.85</td><td>75.2</td></tr><tr><td>MiniMax M2.5</td><td>666</td><td>-1.448</td><td>100.00</td><td>20.8</td></tr><tr><td>Kimi K2.5</td><td>666</td><td>-0.833</td><td>100.00</td><td>20.3</td></tr><tr><td>GPT-4.1 mini</td><td>667</td><td>-0.157</td><td>100.00</td><td>70.4</td></tr><tr><td>GPT-5.4</td><td>666</td><td>-0.886</td><td>100.00</td><td>27.2</td></tr><tr><td>Qwen3 235B</td><td>667</td><td>-0.315</td><td>100.00</td><td>70.5</td></tr><tr><td>Grok 4.20</td><td>666</td><td>-0.314</td><td>100.00</td><td>70.5</td></tr><tr><td>GLM-5.3</td><td>666</td><td>-0.241</td><td>100.00</td><td>21.4</td></tr></table>

Table 6: Settled seven-day cryptocurrency session only. N is retained account-rounds; A is report accuracy conditional on a scorable response. Return uses the same saved endpoint for all accounts. Cash is the final fraction of marked equity. These are descriptive marked outcomes, not estimates of alpha.

## 5.7 Recorded resource use

The cryptocurrency call table contains 14,833 lifecycle entries, compared with 14,598 decision/report slots in committed rounds. The diference reflects a broader population of attempts and recovery records; it is not evidence of 235 successful additional decisions. Prompt and completion counts are both present on 14,737 entries. Their recorded sums are 202,982,942 prompt tokens and 8,341,624 completion tokens, or 211,324,566 tokens in total. Missing usage is not imputed as measured zero. Table 9 provides route-level accounting. These totals describe the recorded workload, including attempts outside committed rounds, rather than a cost-normalized model comparison. We report no reconstructed dollar bill or claimed savings from an untested routing policy.

## 6 Limitations

Observational and dependent evidence. This is a retrospective audit of retained forwardrunning traces, not a preregistered controlled experiment. The two environments cover diferent assets, dates, schedules, and execution policies. Consecutive account-rounds share market conditions, portfolio history, and often input data; their counts must not be read as independent replications. We have neither randomized ablations of audit feedback nor repeated independent runs suficient to isolate model efects from market path and accumulated state. Diferences between models are descriptive of these traces. The short windows do not establish stable rankings, long-run alpha, or generalization across market regimes.

The evidence boundary. Deterministic checks avoid requiring a semantic judge, at the cost of a narrower claim. Decision validity measures format and allowed values; reporting accuracy measures structured factual fields. Neither establishes that the rationale matches the orders or that narrative explanations reveal the model’s internal reasoning. Optional commitment compliance concerns the constraints actually emitted and evaluable; coverage is essential, and selection into commitment-making is uncontrolled. No human-labeled semantic benchmark supports a broader faithfulness or deception claim. Likewise, accurate reporting after seeing a receipt does not validate an ex-ante forecast.

![](images/3f8473c3b6c7008e5571fe24b4b5586863456f976639d0ea187c2f5a0dc7c60c.jpg)  
Figure 4: Available explicitly nonstale saved marks, with returns measured from initial cash. The shaded initial 20-hour-49-minute interval has no eligible marks; the later October 5 gap is approximately 97 minutes. Curves are subsampled and broken across source gaps longer than five minutes, without imputing missing observations. The full-session endpoint is retained. The cash reference assumes no interest or trading and is not an exposure-matched market benchmark.

Policy and execution confounding. The cryptocurrency agents receive algorithmic analysis, including momentum signals, and operate behind a deterministic proposal gate. These are components of the evaluated system. They can afect activity, exposure, returns, and apparent adherence; this study cannot attribute those outcomes solely to the language model. Whole-proposal rejection can suppress feasible orders alongside infeasible ones. Observation-to-execution latency and scheduling vary, and the retained traces do not provide a synchronized randomized comparison of execution times. Equity and cryptocurrency results therefore remain separate rather than being interpreted as a clean cross-market transfer test.

Local accounting and provenance. All fills are generated by a local paper executor using recorded public-market quotes. The audit verifies arithmetic against retained fills, not whether an exchange would have executed them. It does not model queue priority, partial fills, depth-dependent impact, or all venue-specific costs. Ledger agreement can also hold with stale prices and cannot certify external market truth. Saved valuation freshness is unknown for the initial 20 hours 49 minutes, and the later explicitly nonstale path has a roughly 97-minute gap; the displayed curve is therefore incomplete even though common endpoint values can be reconstructed. Recorded predeadline quote metadata is not independently notarized evidence of when it became available. One recovered cryptocurrency record has a nonmonotonic round ordinal; chronological state continuity is restored by timestamp ordering, and the record is retained with the anomaly disclosed. Hash agreement detects changes relative to the stored artifact index, not errors already present when artifacts were written. Missing ancillary execution-attempt artifacts limit complete reconstruction of attempts outside committed round records, so retained-round yield cannot estimate success across all started attempts.

Model and cost attribution. Model names identify requested inference routes. Returned identifiers and call metadata improve traceability but do not independently attest immutable model weights, provider routing, or serving configuration. Prompt and operational diferences also constrain attribution. Token totals describe observed workload, not a normalized computational budget or verified monetary bill. More complete identity attestation, prospective protocol freezing, semantic annotations, and controlled repeated trials are needed before broader causal or financial conclusions.

## 7 Discussion

The central implication is a separation of evidential claims. Action-contract validity establishes that an executor can interpret a decision. Commitment adherence establishes that particular stated constraints hold in a recorded state. Ledger reconciliation establishes arithmetic agreement with fills. Structured reporting accuracy establishes that selected factual claims match a supplied receipt. None alone establishes the other properties, and none establishes profitable investment or transparent internal reasoning.

The data also show why denominators belong in the result, rather than only in an appendix. An agent that rarely declares a constraint has few opportunities to violate one. A system that usually does nothing has many straightforward account reports. A gate that blocks a whole proposa changes the portfolio against which promises are checked. Combining these outcomes into one score without coverage, activity, and eligibility can reward the composition of the observations more than the intended behavior. We therefore provide a profile of measurements and retain undefined outcomes explicitly.

The audit remains useful even when a failure is elementary. A malformed JSON report may be less intellectually interesting than an elaborate false rationale, but it still prevents a downstream application from obtaining the contracted factual output. Conversely, detecting such a failure does not establish that the model cannot reason about finance. Reporting parse, schema, field, and infrastructure outcomes separately constrains this interpretation. The same discipline applies to execution: a missing market quote should not be counted as an invented trade, and a correct report of a rejected order should not be penalized merely because no trade occurred.

Three extensions follow from these limits. First, a prospectively frozen protocol with repeated independent accounts and explicit model-identity records would make comparisons less dependent on the observed path and recovery history. Second, human-annotated rationale–order and summary– receipt pairs would extend the structured checks to semantic concordance; independent adjudication would be needed to avoid treating one language-model judge as an oracle. Third, randomized feedback and gate ablations could test whether audits improve subsequent behavior or merely change activity and exposure. These are future experiments, not efects demonstrated by the present corpus.

## 8 Conclusion

FIDES-Live extends the FIDES measurement perspective from generated trading strategies to sequential agent traces. Its audit distinguishes valid decisions, optional commitments, local execution, account-state arithmetic, and post-execution factual reports while making unavailable evidence visible. Across 7,609 retained account-rounds, ledger reconciliation coexists with report failures, optional commitment coverage varies widely, and adherence difers substantially between active and empty proposals. The results support artifact-level auditing as a complement to financial outcome reporting. They establish neither persistent model rankings nor profitable trading; they establish a reproducible account of which observable claims agree with the recorded actions and states, and where that evidence stops.

Data and code availability. The accompanying source package includes the generated tables, figures, normalized audit outcomes, source digests, and ofline analysis code. Original local-market and inference artifacts are retained for provenance checking and are not assumed to be publicly hosted. No credentials or environment files are included. The normalized outcomes reproduce the reported audit aggregates; regeneration from raw evidence requires access to the retained source corpus. No real-money orders were placed for this study.

## References

[1] Yanxu Chen, Zijun Yao, Yantao Liu, Amy Xin, Jin Ye, Jianing Yu, Lei Hou, and Juanzi Li. StockBench: Can LLM agents trade stocks profitably in real-world markets?, 2025. URL https://arxiv.org/abs/2510.02209.

[2] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as agents, 2023. URL https://arxiv.org/abs/2308.03688. Published at ICLR 2024.

[3] Lingfei Qian, Xueqing Peng, Yan Wang, Vincent Jim Zhang, Huan He, Hanley Smith, Yi Han, Yueru He, Haohang Li, Yupeng Cao, Yangyang Yu, Alejandro Lopez-Lira, Peng Lu, Jian-Yun Nie, Guojun Xiong, Jimin Huang, and Sophia Ananiadou. When agents trade: Live multi-market trading benchmark for LLM agents, 2025. URL https://arxiv.org/abs/2510.11695.

[4] Arther Tian, Alex Ding, Frank Chen, Simon Wu, and Aaron Chan. A multi-dimensional quality scoring framework for decentralized LLM inference with proof of quality, 2026. URL https://arxiv.org/abs/2603.04028.

[5] Arther Tian, Alex Ding, Frank Chen, Simon Wu, and Aaron Chan. PoQ-Judge: A multiarchitecture evaluation framework for cost-aware proof-of-quality in decentralized LLM inference, 2026. URL https://arxiv.org/abs/2606.11196.

[6] Arther Tian, Alex Ding, Frank Chen, Simon Wu, and Aaron Chan. Adaptive and robust cost-aware proof of quality for decentralized LLM inference networks, 2026. URL https: //arxiv.org/abs/2601.21189.

[7] Arther Tian, Alex Ding, Frank Chen, Simon Wu, and Aaron Chan. SFGA: A statistics-first gating architecture with adjudicative escalation for trustworthy SFT data procurement, 2026. URL https://arxiv.org/abs/2607.18960.

[8] Arther Tian, Alex Ding, Simon Wu, and Aaron Chan. FIDES: A concordance protocol for LLM-generated trading strategies, 2026. URL https://arxiv.org/abs/2608.23308.

[9] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://arxiv.org/abs/ 2305.04388.

[10] Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. TradingAgents: Multi-agents LLM financial trading framework, 2024. URL https://arxiv.org/abs/2412.20138.

[11] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains, 2024. URL https://arxiv.org/abs/ 2406.12045.

[12] Haofei Yu, Fenghai Li, and Jiaxuan You. LiveTradeBench: Seeking real-world alpha with large language models, 2025. URL https://arxiv.org/abs/2511.03628.

[13] Wentao Zhang, Mingxuan Zhao, Jincheng Gao, Jieshun You, Huaiyu Jia, Yilei Zhao, Bo An, and Shuo Sun. AlphaForgeBench: Benchmarking end-to-end trading strategy design with large language models, 2026. URL https://arxiv.org/abs/2602.18481.

## A Recorded Routes and Protocol Diferences

<table><tr><td>Abbreviation</td><td>Exact requested route</td></tr><tr><td>GPT-4.1 mini</td><td>openai/gpt-4.1-mini</td></tr><tr><td>GPT-5.4</td><td>openai/gpt-5.4</td></tr><tr><td>DeepSeek V3.2 Gemini 2.5 Flash</td><td>deepseek/deepseek-v3.2</td></tr><tr><td>Claude Haiku 4.5</td><td>google/gemini-2.5-flash</td></tr><tr><td>Qwen3 235B</td><td>anthropic/claude-haiku-4.5</td></tr><tr><td></td><td>qwen/qwen3-235b-a22b-instruct-2507</td></tr><tr><td>Kimi K2.5</td><td>moonshotai/kimi-k2.5</td></tr><tr><td>GLM-5.3</td><td>z-ai/glm-5.3</td></tr><tr><td>MiniMax M2.5</td><td>minimax/minimax-m2.5</td></tr><tr><td></td><td></td></tr><tr><td>Grok 4.20</td><td>x-ai/grok-4.20-non-reasoning</td></tr></table>

Table 7: Model names denote request routes, not independently attested immutable model weights.

<table><tr><td>Setting</td><td>Cryptocurrency</td><td>Equity pilot</td></tr><tr><td>Assets</td><td>Six spot assets</td><td>Eight ETFs initially; fifteen assets in main run</td></tr><tr><td>Decision interval</td><td>15 minutes</td><td>5 minutes initially; 15 minutes in main run</td></tr><tr><td>Initial allocation</td><td>$10,000 per model/session</td><td>$10,000 per model/run</td></tr><tr><td>Historical context</td><td>Up to 90 daily bars</td><td>Up to 30 daily bars</td></tr><tr><td>Fee / adverse slippage</td><td>10 / 2 basis points</td><td>1 / 2 basis points</td></tr><tr><td>Cash / position / exposure</td><td>20% / 30% / 80% fixed gate</td><td>No matching balanced gate</td></tr><tr><td>Advisory analysis</td><td>Deterministic momentum/volatility</td><td>None supplied</td></tr><tr><td>Maximum orders Quote age / spread</td><td>12</td><td>20</td></tr><tr><td></td><td>90 seconds / 100 basis points</td><td>120 seconds / 100 basis points in main run</td></tr><tr><td>Execution synchronization</td><td>Per-account snapshots</td><td>Joint decision-batch barrier</td></tr></table>

Table 8: Material recorded diferences. These conditions are not pooled into a causal cross-market comparison.

The cryptocurrency decision system prompt is present in all 7,299 round artifacts with one SHA-256 value; 7,298 report prompts share one report hash, with one interrupted report lacking a prompt. Self-hashes match where present. The decision hash is a6eb0fdbf2b7569db5db457f5edd5 e907960f0e7509223d1404d2314d153cfd6; the report hash is 2ab3a395a589d6a35631683867ebb5 cef84b4f73fb4553d952789e8f558871c3. Long hashes in the source data provide exact byte-level identification. Two wordings of an advisory strategy-role field occur in 5,505 and 1,794 requests, respectively; recorded numerical risk and execution settings match. Prompt stability therefore does not imply that every complete request payload is identical.

## B Resource Accounting and Daily Observation Counts

<table><tr><td>Model</td><td>Calls</td><td>Tokens (M)</td><td>Usage pairs</td><td>Median seconds</td></tr><tr><td>Claude Haiku 4.5</td><td>1484</td><td>27.54</td><td>1463</td><td>9.43</td></tr><tr><td>DeepSeek V3.2</td><td>1484</td><td>16.52</td><td>1476</td><td>5.48</td></tr><tr><td>Gemini 2.5 Flash</td><td>1484</td><td>28.83</td><td>1476</td><td>2.47</td></tr><tr><td>MiniMax M2.5</td><td>1483</td><td>17.31</td><td>1479</td><td>10.77</td></tr><tr><td>Kimi K2.5</td><td>1483</td><td>22.50</td><td>1480</td><td>11.68</td></tr><tr><td>GPT-4.1 mini</td><td>1484</td><td>16.02</td><td>1479</td><td>2.40</td></tr><tr><td>GPT-5.4</td><td>1482</td><td>16.47</td><td>1476</td><td>4.55</td></tr><tr><td>Qwen3 235B</td><td>1484</td><td>25.29</td><td>1476</td><td>8.14</td></tr><tr><td>Grok 4.20</td><td>1482</td><td>16.17</td><td>1468</td><td>2.29</td></tr><tr><td>GLM-5.3</td><td>1483</td><td>24.67</td><td>1464</td><td>21.52</td></tr></table>

Table 9: Cryptocurrency call-table accounting, including attempts outside committed rounds. Tokens are recorded prompt plus completion counts; missing usage contributes no measured amount. Usage pairs counts calls with both components available. Median elapsed seconds uses recorded values and is descriptive of the gateway/system, not a hardware-normalized speed benchmark.

<table><tr><td>UTC date</td><td>Account-rounds</td><td>Accurate/scorable reports</td><td>Accuracy (%)</td></tr><tr><td>2026-09-30</td><td>820</td><td>686/809</td><td>84.80</td></tr><tr><td>2026-10-01</td><td>959</td><td>805/952</td><td>84.56</td></tr><tr><td>2026-10-02</td><td>960</td><td>860/960</td><td>89.58</td></tr><tr><td>2026-10-03</td><td>959</td><td>872/957</td><td>91.12</td></tr><tr><td>2026-10-04</td><td>957</td><td>867/949</td><td>91.36</td></tr><tr><td>2026-10-05</td><td>906</td><td>814/900</td><td>90.44</td></tr><tr><td>2026-10-06</td><td>913</td><td>890/913</td><td>97.48</td></tr><tr><td>2026-10-07</td><td>825</td><td>803/825</td><td>97.33</td></tr></table>

Table 10: Cryptocurrency observation counts by UTC date. Partial first/last days and operational gaps remain visible. Accurate/scorable reports use the same primary definition as the main text.

<table><tr><td>Route</td><td>Valid/scorable</td><td>Accurate/scorable</td><td>Declaring/valid</td><td>Passed/evaluable</td></tr><tr><td>Claude Haiku 4.5</td><td>556/726</td><td>605/732</td><td>552/556</td><td>1081/1102</td></tr><tr><td>DeepSeek V3.2</td><td>620/628</td><td>90/628</td><td>3/620</td><td>6/6</td></tr><tr><td>Gemini 2.5 Flash</td><td>735/736</td><td>738/739</td><td>75/735</td><td>150/150</td></tr><tr><td>MiniMax M2.5</td><td>740/740</td><td>738/738</td><td>8/740</td><td>8/8</td></tr><tr><td>Kimi K2.5</td><td>739/739</td><td>740/740</td><td>31/739</td><td>52/61</td></tr><tr><td>GPT-4.1 mini</td><td>738/738</td><td>739/741</td><td>231/738</td><td>416/462</td></tr><tr><td>GPT-5.4</td><td>737/737</td><td>739/739</td><td>737/737</td><td>1005/1007</td></tr><tr><td>Qwen3 235B</td><td>738/738</td><td>738/738</td><td>5/738</td><td>9/9</td></tr><tr><td>Grok 4.20</td><td>732/732</td><td>736/736</td><td>128/732</td><td>161/256</td></tr><tr><td>GLM-5.3</td><td>717/730</td><td>734/734</td><td>2/717</td><td>4/4</td></tr></table>

Table 11: Cryptocurrency exact numerators and denominators corresponding to Table 3. Declaring/valid is commitment coverage; passed/evaluable counts individual nontrivial constraints. A perfect rate based on a few constraints is not evidence of broad coverage.

## C Audit Reconstruction and Case Traceability

Inclusion. The corpus includes every available populated original equity paper run and every indexed cryptocurrency round belonging to API-driven arena paper accounts at the audit snapshot. Empty initializations, deterministic demonstrations, historical replay, copied equity archives, and personal fixtures are excluded. Combining datasets refers to a common audit schema, not to relabeling their actual market, policy, or source.

Missingness. The 840 unavailable ancillary attempt artifacts are counted as missing provenance, while the corresponding round and call populations are assessed from their independently checked files. One orphan started call artifact is not counted as a completed response. Database compaction is not treated as data loss when the full original is available. Report-accuracy component failures include unparseable or absent factual fields; they should not be read as counts of proven numerical fabrication.

Chronology. The recovered MiniMax observation has record identifier ce8e50adf8944db8a4e777 ce0b64c598, stored round ordinal 84, and timestamp 2026-09-30 06:57:53 UTC. Its original request payload identifies round 14. Timestamp ordering places it back in its account sequence and restores continuity. We retain both identifiers and the source digest; no original evidence is overwritten.

Reporting cases. The empty-order DeepSeek example with reported filled count two has record identifier 04a5af03a4d447f8abc429367025fa72 and source SHA-256 8c49dceef81e866515878633 0c4cff942163cbbd8ec4ebec0b6d41b6d454164a. The example with filled count 1,584,655,368 has identifier 01289cb3d9944bfea866112cdf8e5e57 and digest 3e3a0bca1b239b993f529c525eff737d 9454b6bf0d60b56e0ad72636e19fc279. These selected cases illustrate already-defined deterministic failures and are not a human-labeled semantic test set.

Reproduction. The ofline audit validates source identity, reconstructs each account-round into a common schema, computes content eligibility, rechecks action contracts and report fields, and reconciles account transitions. The summarization stage generates all tables, numerical macros, and figures. Normalized outcomes carry opaque record identifiers and source digests, allowing aggregate reproduction without exposing machine paths or credentials. Recomputing all factual checks from first principles additionally requires the original account and response artifacts. The source README distinguishes these two reproduction levels.

Report what ACTUALLY happened in this paper trading round, using only   
the supplied execution receipt. Distinguish rejected orders from filled orders.   
Return ONLY JSON with exactly these fields: summary (a nonempty factual recap),   
cash (number), equity (number), positions (ticker-to-share-count object),   
filled\_orders (integer count of filled orders in this round).   
positions contains share quantities, not dollar amounts. Include all nonzero   
holdings. Do not invent successful trades or profitability. This is a report,   
not another chance to place orders.   
Copy the supplied account values faithfully: money within 0.01 and share counts   
within 0.000001. Preserve the receipt’s precision rather than rounding holdings.

## D Historical Output Contracts

The following text is recovered from a retained historical cryptocurrency request, rather than reconstructed from the current application code. Decision excerpts omit other instructions; the report prompt is reproduced in full. The original record identifier, source digest, and full-systemprompt hashes accompany the source package. These contracts explain why a parsed but structurally invalid decision or a cumulative rather than round-specific fill count fails the requested interface.

Decision contract (excerpt).

Return ONLY a JSON object. Required fields:   
- rationale: a nonempty string explaining your own plan, evidence, sizing and risks.   
- orders: an array of order objects. Each order has exactly symbol (an allowed   
ticker string), side ("buy" or "sell"), and quantity (a positive asset-unit count).   
Optional field: commitments, an object containing any numerical promises you make.

## Numerical commitments (excerpt).

commitments is optional (or {}). Only include numerical promises you actually make about the portfolio IMMEDIATELY after execution, marked at execution mids. They are evaluated, not imposed on you. They do not describe future returns. Supported promises are min\_cash\_ratio and max\_position\_ratio, numbers in [0,1]. Do not add promises merely to fill the format. Cash >= 0 and position <= 100% already follow from the trading rules and are not informative promises.

Post-execution report (full system prompt).