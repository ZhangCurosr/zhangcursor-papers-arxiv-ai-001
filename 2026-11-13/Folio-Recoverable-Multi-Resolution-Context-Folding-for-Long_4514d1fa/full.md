# Folio: Recoverable Multi-Resolution Context Folding for Long-Horizon Language Agents

Yubo Ruan<sup>1</sup> Tao Peng<sup>2</sup> Jiao Zhang<sup>1</sup> Chuanguo Ma<sup>2</sup> Lei Xu<sup>2</sup> Han Zhou<sup>2</sup> Wenwu Wang<sup>2</sup> Xiafei Lv<sup>2</sup> Xingjie Zhou<sup>2</sup> Guiju Fan<sup>2</sup> Hao Wang<sup>1</sup> Fenghui Zhang<sup>2</sup> Mao Miao<sup>2</sup> Shuo Li<sup>2</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>Baidu, Inc.

{ruanyubo, jiaozhang, h.wang}@bupt.edu.cn

{pengtao04, machuanguo, xulei25, zhouhan06, wangwenwu01, lvxiafei, zhouxingjie01, fanguiju, zhangfenghui, miaomao01, lishuo02}@baidu.com

## Abstract

Long-horizon language agents accumulate tool results and replies until the history no longer fits the context window. Current agents shrink this history by having a model summarize it, so each compaction rereads and rewrites the history it replaces, and cacheaware triggers decide when to compact while taking the size of the result as given. We present FOLIO, a context manager that records, folds, and recovers history through one set of handles. While the agent runs, FOLIO records three linked levels: a short turn note the model writes as part of its ordinary reply (L1), an event manifest of every call written by the runtime (L2), and the captured source records (L3). Because these views already exist, the runtime folds the context programmatically, selecting coarser views for older turns. The fold prints the handles that index L2 and L3, so recovery tools that read bounded pages, compute over full text in files, and keep findings in a persistent dossier reach the evidence the agent needs with little added input. The fold trigger prices an explicit landing point that includes the retained budget. On 75 LOCAbench tasks with DeepSeek-V4-Pro behind every system, FOLIO solves 66 tasks against 54 for the same runtime with model-written compaction and 49–59 for Claude Code, Codex, and SoL-Pi, in about half their working time, at 2.4 times lower cost than that runtime; each fold takes a median of 0.6 seconds.

## 1 Introduction

A language agent that solves a long task alternates between generating an action, running a tool, and reading the result. Every result, reply, and intermediate decision is appended to the history that the next request carries. On hard tasks the history outgrows the context window well before the task ends, and even before that point it is reread on every request at the provider’s input price. The agent must therefore repeatedly decide what stays in the window, while keeping access to evidence that may matter many steps later.

Existing work addresses parts of this problem (§6). Compaction methods shorten the history by masking or truncating it by rule, or by having a model summarize it (Lindenbauer et al., 2025; Nguyen et al., 2026; Kang et al., 2026; Sun et al., 2025; Ye et al., 2025). Hierarchical and recoverable memory keeps the source reachable below the reduced view: HiAgent summarizes each completed subgoal and retrieves its trajectory on request (Hu et al., 2025), LCM builds a DAG of model-written summaries whose leaves point to the original messages (Ehrlich and Blackman, 2026), and SoL-Pi archives large tool outputs behind stable handles (Liu et al., 2026). Cacheeconomic models ask when a reduction pays for itself under prompt caching: Context Is Inventory treats a session as a rent-or-buy problem in which carrying context costs cached reads and a restart costs a rebuild (Cheng, 2026), and SoL-Pi’s Online Context Compact compares projected savings with the cache-rewrite cost before invoking the base agent’s summarizer (Liu et al., 2026).

Two gaps remain across these lines. First, the reduced representation is produced at compaction time, usually by a model. Each compaction reads the material it replaces and writes a new object, so the work grows with the number of compactions, and the link between the new object and its source is whatever the compaction step recorded. Second, the timing models take the post-reduction size as given. In the restart model of Context Is Inventory the context returns to its working floor after every rebuild, so the landing point is not a decision, and a cost gate that decides whether to compact leaves the size of the result to the summarizer. For folding, which keeps part of the history, how much is kept is a choice with its own cost, and the trigger should price it.

We present FOLIO, a context manager in which the history representation is formed during execution and folding becomes a selection among views that already exist. FOLIO records the history as three linked levels while the agent runs: a turn note of a few lines that the model writes as part of its ordinary reply (L1), instead of a summary produced by a separate request; an event manifest the runtime writes for every call, reply, and turn (L2); and the source records of the captured event log (L3). Every call has one handle, such as t3.12, that stays valid across levels and across folds. Because the views already exist, FOLIO’s runtimefolds the context programmatically: it selects coarser views for older turns and keeps the open turn’s earliest small results and newest results verbatim. The fold prints the same handles that index L2 and L3, so the agent can recover the evidence it actually needs quickly and with little added input: scan reads a bounded part of a source, ctx computes over its full text in files, and dossier keeps findings that persist across folds. The fold trigger prices the landing point $S = B + L + D$ , the stable prefix plus the retained layout budget plus the rendered dossier, so the retained size enters the cost decision. These parts are not independent features: one handle space and one renderer tie the representation, the fold, recovery, and cost control together.

This paper makes four contributions:

• An execution-time, multi-resolution history in which the identity of a historical object is separate from the resolution at which it is shown, with byte-offset and digest addressing into the event log (§3.2).

• A programmatic fold in which the runtime selects and demotes existing views and never rerenders an already sent line at a richer level, together with three recovery operations that share the fold’s handles and so reach folded evidence precisely and cheaply (§3.3, §3.4).

• A landing-priced fold trigger that accrues carried excess against each request’s own landing point, so it needs neither a fixed fold interval nor linear context growth (§3.5).

• An evaluation on 75 LOCA-bench tasks run as five 15-task sessions, against Claude Code, Codex, SoL-Pi, and the same runtime without FOLIO, all on DeepSeek-V4-Pro, with component ablations, a landing-budget sweep, a check of every fold, and a recovery case (§5).

With the same model behind every agent, FO-LIO solves 66 of 75 tasks, against 54 for the same runtime with model-written compaction, 59 for Claude Code, 57 for Codex, and 49 for SoL-Pi, and needs a mean of 10.2 minutes per task against 16.3–21.0. Against the same runtime it costs 2.4 times less in normalized tokens. Its folds are completed by the runtime alone in a median of 0.6 seconds, against 101 seconds for a model-written compaction. Removing the outline from the fold nearly doubles the number of folds and raises cost by 63%, and a sweep of the landing budget shows that the retained size trades folds and recoveries against carried context.

## 2 Problem Formulation

Sessions and contexts. A session consists of turns. A turn opens with a user message (or a runtime continuation) and may contain many model requests and tool calls before it closes. We index model requests by t, because each request is a unit of both context construction and billing. Let $\mathcal { H } _ { t }$ be the history captured before request t and $X _ { t }$ the context actually sent. A context manager constructs $X _ { t }$ from $\mathcal { H } _ { t }$ subject to $| X _ { t } | + r _ { t } \le W$ where W is the window and $r _ { t }$ the space reserved for the response. The history is a record of past observations, not the current environment: rerunning a query may return different data, and rerunning a write may repeat its side effects. Recovering history therefore means reading what was captured, not calling the tool again.

Cost. Prompt-cached APIs bill cached input, uncached input, cache creation, and output at different prices. For request t let $h _ { t } , \ u _ { t } , \ w _ { t }$ , and $o _ { t }$ be these mutually exclusive token counts and $p ^ { h } , p ^ { u } , p ^ { w } , p ^ { o }$ their prices. The cost of a run is

$$
C = \sum _ { t \in \mathcal { R } } \bigl ( p ^ { h } h _ { t } + p ^ { u } u _ { t } + p ^ { w } w _ { t } + p ^ { o } o _ { t } \bigr ) ,\tag{1}
$$

where $\mathcal { R }$ contains every billed request: the agent’s own, any auxiliary model calls for notes or memory maintenance, and retries. Tokens read back from folded history are already inside these requests and are not counted again. Dividing by $p ^ { h }$ expresses all terms in cached-read equivalents while keeping the price ratios $\rho _ { u } = p ^ { u } / p ^ { h } , \rho _ { w } =$ $p ^ { w } / p ^ { h }$ , and $\rho _ { o } = p ^ { o } / p ^ { h }$

![](images/c6c4412098a6662c3b6bf5ffcdc768b88b7ed0ea50215ac934dae7bc8250a5c0.jpg)  
Figure 1: Overview of FOLIO. Record: the runtime turns tool events into a three-level indexed history. Fold: the cost-aware trigger selects views of this history to build the active context. Recover: the handle in the outline (t3.12) is passed to scan for a bounded read or to ctx to compute over the full text, and the result is kept in the dossier.

Three decisions. A context manager makes three decisions. It chooses a representation of history that can be shown at less than full size. It chooses when to replace the resident history with a smaller view. It chooses how large that view is. Under Eq. 1 the last two interact: a fold breaks the cached prefix and the next request pays $\rho _ { u }$ or $\rho _ { w }$ on what is rebuilt, while every later request pays $p ^ { h }$ on what is kept. Keeping less also changes what follows, because the agent may search, reread, or reason again to recover what it no longer sees.

Restart and fold. The rent-or-buy model of Cheng (2026) analyzes restarts. After a restart the context returns to the working floor $L _ { \mathrm { { f l o o r } } }$ (system prompt, tool definitions, project instructions), and the rebuild cost is priced on that floor. The postrestart size is therefore fixed by construction. A fold keeps part of the history, so the context after it lands at

$$
S = B + L + D ,\tag{2}
$$

where B is the stable prefix, L the layout budget for the folded history and the open turn, and D the dossier of retained findings (§3.4). B plays the role of $L _ { \mathrm { f l o o r } }$ , but L is a parameter of the policy and D changes as the agent works. A trigger for folding has to price this landing point rather than assume it.

Recoverability contract. We call a fold recoverable if, for every object that was captured, is still stored, and has a valid handle, three properties hold after any number of folds: (i) retention, the fold does not delete the source record; (ii) resolution, the handle still resolves to the same record; and (iii) exact read, a read through the handle returns that record’s text, or reports that only a cut copy is available. The contract says nothing about whether the agent asks for the right record or uses it correctly; those are measured as task outcomes (§5). It also does not cover content the tool never returned or the runtime never captured.

## 3 Folio

## 3.1 Overview

FOLIO sits between the agent and its tools and runs three operations that repeat throughout a session (Figure 1). Record: as calls and results pass between the agent and its tools, the runtime appends each event to the source records (L3), writes an event-manifest row for each call and reply as it arrives (L2), and stores the model’s turn note when the turn closes (L1). Together these form an indexed history outside the context (§3.2). Fold: when the trigger fires (§3.5), the runtime selects views of these rows within the layout budget and builds the active context: the stable prefix, an outline of earlier turns, the open turn’s recent results, and the dossier (§3.3). Recover: the outline prints call handles; the agent passes a handle to scan or ctx to reach the folded source, and keeps what it learns in the dossier (§3.4). New calls extend the same levels. Figure 2 follows one call, t3.12, through all three operations.

![](images/6d9c9177ec0e1e2c039a13be3a0bfd39f78db24371af8c3e668d0b14b7ba7dac.jpg)  
Figure 2: Folding and recovery on one example. (a) Turn t3 exists as a turn note (L1), the manifest row of call t3.12 (L2), and the 900-row table it returned (L3). Within the layout budget L, the fold shows t1 briefly, t2 with its full note, and t3 with the row of t3.12; the open turn t4 keeps a raw head and tail. (b) Through the same handle, scan checks one value and ctx counts every row; the finding and its source go to the dossier, which the next fold carries.

## 3.2 Multi-Resolution History

The three levels have different authors and different jobs (Figure 2a, left). The model writes only L1; the runtime writes L2 and L3. None is a complete encoding of the level below: a coarse view can omit a value, and that value stays reachable through the handle.

L1: turn notes. The model ends a turn’s reply with a short note block containing an outcome line and any number of fact lines (user constraints, choices and their reasons). The runtime parses the block out of the user-visible reply and stores it with the turn’s status, goal, and raw size. A turn without a note falls back to the head of its final reply. Writing notes is part of ordinary generation,

not a separate request.

L2: event manifest. A tool row records the tool, an argument digest, the purpose the model gave for the call, status, output size, the first content line, and for results above 1K tokens a one-line shape and sample of the parsed structure (for example array[135] of {id:number,name:string,. . . }). A say row records text the model wrote between calls and where it fell in the call sequence. A turn row, written when the runtime declares the turn closed, records the goal, the reply head, tool counts, and the rendered size of the turn at every level. Tool and say rows of an open turn are written as their events land, so the manifest is current at any fold.

L3: source records. The append-only event log holds user messages, tool requests, routed results, and assistant messages. When the runtime cut a result before routing it to the model, the complete output is kept in a separate file referenced from the event.

Identity and addressing. A call is named tN.k: the k-th tool row of turn N in the order results settled, which is also the order scan resolves. Ranges are tN.a-b. Each tool row stores the byte range of its routed event in the log and a digest of those bytes. A read seeks to the range and checks the digest; a mismatch, a missing file, or a missing range falls back to parsing the log. The offset is thus an access hint and the handle is the identity. L1 rows point to the byte range of their turn’s L2 rows. A result that settles after its turn closed is appended later and listed in an index, so the handle still finds it. When the manifest file grows past a size limit or a day boundary, it is split at a turn boundary into archived segments with logical offsets, and an interrupted split is completed on the next load.

## 3.3 Folding as View Selection

Layout. A fold renders into a layout budget L (the landing budget of Eq. 2; Figure 2a, right). An outline share α holds the reading guide, the closed turns, and the rows of the open turn; the rest holds raw results of the open turn, split into a head share β and a tail:

$$
\begin{array} { r l } & { L _ { \mathrm { o u t } } = \alpha L , \qquad L _ { \mathrm { h e a d } } = \beta ( 1 - \alpha ) L , } \\ & { L _ { \mathrm { t a i l } } = ( 1 - \beta ) ( 1 - \alpha ) L . } \end{array}\tag{3}
$$

We use $\alpha ~ = ~ \beta ~ = ~ 0 . 2$ . The budgets are planning targets: rows are discrete, and the raw area is further capped so that the post-fold context leaves headroom below the window.

Closed turns. The planner draws each closed turn at one of three levels: l2 (head line, outcome and facts, one row per call), $l I \_ f u l l$ (the same without call rows), or l1\_brief (goal cut to 80 characters and the outcome). The planner starts with the newest turn at l2, the two before it at l1\_full, and older turns at l1\_brief; in Figure 2a, t1 is brief, t2 keeps its full note, and t3 keeps the manifest row of t3.12. If these floors exceed the budget, it demotes from the oldest turn; it then upgrades from the newest turn while the increase fits. A turn whose full call list does not fit is tried at l2 with its oldest calls folded into one paging row (tN.1-m Œm ok x failed y: tools) before it settles for l1\_full. Beyond the newest 30 turns, older turns merge into phase rows of 10 turns that keep each turn’s note lines.

Monotone demotion. The planner reads the levels recorded by the previous fold and caps each turn at its previous level and paging. A line that was sent once is never redrawn richer, even when a later fold has room. Repeated folds therefore only shorten the carried outline, and the part of it that does not change stays a byte-identical prefix for the provider’s cache.

The open turn. The open turn is laid out from its rows (t4 in Figure 2a). The head takes the turn’s earliest results of at most 1K tokens in call order until one does not fit; these are typically the task’s inputs. The tail takes results from the newest call backward until one does not fit. Every other call keeps its row, and a large one also keeps its structure line. If the rows still exceed the outline area, the oldest middle rows fold into a paging row. When the open turn continues across several folds, the next fold parses the block the previous fold sent, appends the new calls, and lays out the whole turn by the same rules with the same call numbers.

A programmatic fold. The runtime folds by selecting, cutting, and arranging rows that already exist. After rendering, it rebuilds the closed turns from the L1/L2 files alone, re-renders them, and compares the digest with the text it sent; a mismatch is recorded with the fold. Model work in FOLIO is limited to the L1 notes and dossier entries written during execution, plus one fallback: when the dossier’s entries exceed half the window, a low-effort auxiliary request merges duplicate entries in chunks, and a chunk that fails stays as it was. All of these are billed requests in Eq. 1.

## 3.4 Access Through Shared Handles

The outline tells the agent where evidence is; three operations bring it back (Figure 2b). All three take the handles the outline prints, so a read lands on exactly the object the fold removed.

scan: locate and read. scan is read-only. list expands a handle: a turn into its call rows, a range into rows one page at a time, a long call result into a structure map of line-numbered headings, signatures, or paragraph heads. The map is returned only when it is informative (at least three entries and one per 40 lines); otherwise the first text page comes back. read returns a bounded page of a call’s text by line range, by grep, or centered on a hit, at most 200 lines and 8K tokens per answer, and names the next page. find searches all folded results and replies and returns one line per hit with its handle, or snippets, rows, or whole calls, with a cursor to continue. Patterns are plain text with optional anchors, so a search cannot stall the runtime. When the routed copy of a result was cut, read pages the complete output file instead and says so if that file is missing or too large. scan does not read or search the results of earlier scan calls: a read of such a call returns an error that names the original handle, and find skips them, so a search does not count the same hit twice or match its own earlier pages.

ctx: compute in files. Some questions need the whole of a result: counting rows, joining two lists, or checking every entry against a condition. Paging such a result into the window costs input on every page and invites transcription errors. ctx binds the full text of a handle or range to a file, runs a shell or Python command the agent writes over it, and returns a bounded answer. Its output can be saved as derived/<name>, which later ctx calls take as input and which the dossier lists.

dossier: keep what the task needs. The dossier is the agent’s store of working results. A finding holds one fact under a key, with the handle it came from; a note holds a judgment and is append-only; a rule holds a constraint that must survive turns. Entries have stable ids (m12) and are rewritten by key or forgotten by id; notes cannot be forgotten. Each entry is written either for the current turn (the default) or as longterm. Current-turn entries are shown only while that turn runs; long-term entries, mostly rules, persist across turns. Separately, the runtime logs the user’s stated requirements verbatim and shows them in every turn. Every fold appends the dossier after the outline under its own budget (12% of the window). Within that budget, the visible notes are always shown and the newest 30 file pointers are listed; the remaining budget goes to long-term entries and then to the current turn’s findings, newest first in each group, so older entries are the first to be omitted. Entries new since the last fold are marked, and the agent’s latest plan in this turn is attached at the top together with the handles it has already read back in this turn. The agent can therefore resume from its unfinished steps without rereading what it already extracted.

Example. Figure 2b continues the example. Call t3.12 exported a 900-row table; after the fold, the outline keeps only its manifest row t3.12 export\_table  900 rows. To check one value, the agent reads t3.12 with scan and a grep for the id, and receives a bounded page with the row 102 | 0.58 instead of the table. To confirm that all 900 rows are present, it runs ctx over the same handle with a row count; the count runs over the full text in a file and returns only 900. The agent then writes the finding with its source t3.12 to the dossier. The next fold carries neither the table nor the reads, and the finding stays in the context with the handle to check it again.

## 3.5 Landing-Priced Fold Trigger

The trigger decides when to fold; the layout budget L decides where the fold lands. Following the rentor-buy view (Cheng, 2026), a fold is worth it once the cached reads carried above the landing point have cost as much as rebuilding the landing point.

Landing point. The stable prefix B is measured once: the provider-counted input of the session’s first request minus the estimated tokens of its message text. After every response, the runtime renders the dossier exactly as the next fold would and measures its size $D _ { t }$ . The landing point and the rebuild cost of a fold decided at request t are

$$
S _ { t } = B + L + D _ { t } , \qquad K _ { t } = \rho _ { w } ( L + D _ { t } ) ,\tag{4}
$$

since a fold rewrites everything above the stable prefix.

Carried excess. For each settled request with provider-counted cached input $h _ { t }$ , the excess is charged against that request’s own landing point:

$$
E _ { t } = E _ { t - 1 } + \operatorname* { m a x } ( 0 , h _ { t } - S _ { t } ) .\tag{5}
$$

A request that missed the cache has $h _ { t }$ below $S _ { t }$ and adds nothing. The runtime folds when $E _ { t } \ \geq$ $\lambda K _ { t } .$ , judged after a response and before its tool calls run; a committed fold resets E. A separate safety line folds whenever the context reaches a fixed share of the window, and the layout keeps a minimum headroom above the landing point.

What the rule does not assume. Eq. 5 sums observed requests. It does not need the number of requests between folds, a constant growth rate, or a fixed post-fold size: when the dossier grows, $S _ { t }$ and $K _ { t }$ move with it, and the interval between folds changes accordingly. The landing point includes the retained budget L, which a restart model sets to zero by construction. Two costs are left out of $K _ { t }$ on purpose: the extra reasoning after a fold and the tokens the agent reads back. Neither is known when the decision is made; both appear in later requests, where Eq. 5 and Eq. 1 count them. Our reference configuration uses $L = 5 0 \mathrm { K } , \lambda = 1$ , and DeepSeek-V4-Pro’s price ratios $( \rho _ { u } = \rho _ { w } = 3 0 , \rho _ { o } = 9 0 )$

## 4 Experimental Setup

Tasks. We use LOCA-bench (Zeng et al., 2026), an open benchmark of tool-using tasks whose context grows with the environment. It has 15 environments (course scheduling from Canvas announcements and email, inventory and sales analysis over WooCommerce, invoice checking, A/B-test analysis, and others), each with five seeds, for 75 tasks with deterministic graders. We build five chains: chain k takes the k-th seed of every environment, and its 15 tasks are given to the agent one after another in a single session, so later tasks run on top of the history of earlier ones. Each system runs each chain once, giving 75 graded runs per system on identical task sequences (the task prompts match by hash across systems). Under FOLIO, 60 of the 75 tasks trigger at least one fold, and the median context at a fold is 242K tokens.

Systems. All systems use DeepSeek-V4-Pro. FOLIO runs in the Piki runtime with a 1M-token window and the reference configuration of §3.5 $( L = 5 0 \mathrm { K } , \lambda = 1 )$ We also run FOLIO with a 128K window, where the window cap cuts the landing budget to about 16.5K. Piki-Classic is the same runtime without FOLIO, which summarizes the history with the model when the context reaches 90% of the window. Claude Code, Codex, and SoL-Pi (all four mechanisms enabled; Liu et al., 2026) run with a 1M-token window. No system has a step or time budget beyond the benchmark’s own timeout. Ablations (§5.2) use the same 1M configuration; the landing-budget sweep (§5.3) runs the fifth chain twice per budget, once in ascending and once in descending order of L.

Metrics. Success is the number of the 75 runs the grader accepts; we also report the mean and standard deviation of successes per chain (out of 15). Time is the agent’s working time per task in minutes, excluding environment setup and grading. For FOLIO and Piki-Classic, whose runtime records provider usage, we report tokens per task split into cached input, uncached input, and output, and a normalized cost in cached-read equivalents, $h + 3 0 u + 9 0 o .$ , using DeepSeek-V4-Pro’s price ratios (Eq. 1). The closed systems do not expose provider usage, so we compare them on success and time only. Paired comparisons over the 75 tasks use an exact McNemar test for success. The benchmark’s contamination scan flags runs that read grader data; flagged runs are kept and their counts are listed in Table 1.

<table><tr><td></td><td colspan="2">Success</td><td colspan="2">Time (min/task)</td><td></td></tr><tr><td>System</td><td>175</td><td>per chain</td><td>mean</td><td>median</td><td>Flag.</td></tr><tr><td>Claude Code</td><td>59</td><td> $1 1 . 8 \pm 0 . 8$ </td><td>18.5</td><td>13.7</td><td>9</td></tr><tr><td>Codex</td><td>57</td><td> $1 1 . 4 \pm { 1 . 7 }$ </td><td>20.2</td><td>15.4</td><td>27</td></tr><tr><td>SoL-Pi</td><td>49</td><td> $9 . 8 \pm 1 . 3$ </td><td>21.0</td><td>17.9</td><td>24</td></tr><tr><td>Piki-Classic</td><td>54</td><td> $1 0 . 8 \pm { 1 . 3 }$ </td><td>16.3</td><td>13.7</td><td>11</td></tr><tr><td>FOLIO (128K)</td><td>60</td><td> $1 2 . 0 \pm 1 . 2$ </td><td>16.4</td><td>7.8</td><td>9</td></tr><tr><td>FOLIO (1M)</td><td>66</td><td> ${ \bf 1 } 3 . 2 \pm { \bf 1 } . 5$ </td><td>10.2</td><td>6.9</td><td>8</td></tr></table>

Table 1: LOCA-bench, 5 chains 15 tasks, DeepSeek-V4-Pro for every system. Per chain: mean  s.d. of successes out of 15. Flag.: runs flagged by the benchmark’s contamination scan.

## 5 Results and Analysis

We report five experiments: a comparison with other agents (§5.1), component ablations (§5.2), a sweep of the landing budget (§5.3), the cost and correctness of the fold itself (§5.4), and a recovery case (§5.5).

## 5.1 Comparison with Other Agents

Against the same runtime. Piki-Classic differs from FOLIO only in context management. It solves 54 of 75 tasks against FOLIO’s 66; paired by task, FOLIO solves 14 tasks that Piki-Classic fails and fails 2 that it solves (exact McNemar $p \ : = \ : 0 . 0 0 4 )$ . FOLIO is faster on 57 of 75 tasks, with a median per-task time ratio of 1.45.

Against other agents. FOLIO also solves more tasks than Claude Code (59), Codex (57), and SoL-Pi (49) (Table 1, Figure 3a). Paired by task, it solves 14 tasks that Claude Code fails and fails 7 that Claude Code solves $( p \ = \ 0 . 1 9 ) ;$ against Codex the split is 14 to $5 ~ ( p = 0 . 0 6 )$ and against SoL-Pi 20 to 3 $( p \textless 0 . 0 0 1 )$ . With 75 tasks, the differences from Piki-Classic and SoL-Pi are significant at 0.05 and those from Claude Code and Codex are not. Figure 3a shows that FOLIO’s lead widens in the second half of the session, when the history of earlier tasks has been folded several times. Figure 4 breaks the result down by environment. The gap is concentrated in two environments in which every other system fails most seeds: NHL B2B analysis (FOLIO 4/5, every other system 0/5) and A/B-test analysis (4/5 against at most 2/5).

FOLIO is also faster (Figure 3b). Its mean working time is 10.2 minutes per task, against 16.3 for Piki-Classic, 18.5 for Claude Code, 20.2 for

(a) Successes along the session (mean of 5 chains)  
![](images/fc3193f9abda2c3c403a9f87c1e20b9f17c66c84d2a3aecd9e9c92773191d3f2.jpg)

![](images/7e6f565c71b8bfe15c34353bb29ec3a223fc31c142c5c530084e9bdeff076814.jpg)  
Figure 3: (a) Cumulative successes along the 15-task session, averaged over the five chains; the dotted line solves every task. (b) Distribution of working time per task over all 75 tasks.

![](images/fd877314c46b5eb951b96dcb0832740a0f08f080f12a1435e57c9d85217f904a.jpg)  
Figure 4: Successes per LOCA-bench environment (out of 5 seeds). Rows are sorted by the mean of Claude Code, Codex, and SoL-Pi, hardest first.

Codex, and 21.0 for SoL-Pi. It takes less time than Claude Code on 61 of 75 tasks, less than Codex on 65, and less than SoL-Pi on 64; the median pertask ratio of the baseline’s time to FOLIO’s is 1.56, 1.80, and 1.90. With the window cut to 128K, FO-LIO still solves 60 tasks, more than any baseline run with a 1M window, at 16.4 minutes per task.

Cost. Table 2 compares usage within the Piki runtime, where provider usage is recorded. Piki-Classic’s normalized cost is 2.38 times FOLIO’s (3,525M against 1,481M tokens over 75 tasks). FOLIO is cheaper on 68 of 75 tasks and in all 15 environments, with a median per-task ratio of 2.29. Both agents make about the same number of tool calls (10,701 against 10,173), so the difference comes from how much history each request carries: Piki-Classic compacts at 90% of the window, so its requests carry up to about 900K tokens before each compaction, and it reads 3.5 times as much input. Its 24 compactions also cost 420M normalized tokens for the summary requests alone, 28% of FOLIO’s total cost. Within FOLIO, 92% of input is served from cache, and the 128K configuration costs the same as the 1M one within 0.4%: it folds almost three times as often and reads 32% less input, but it issues 35% more requests and generates 46% more output.

<table><tr><td rowspan="2">Per task</td><td colspan="2">FOLIO</td><td>Piki-</td></tr><tr><td>1M</td><td>128K</td><td>Classic</td></tr><tr><td>Input tokens (M)</td><td>4.01</td><td>2.72</td><td>14.12</td></tr><tr><td>cached</td><td>3.69</td><td>2.46</td><td>13.29</td></tr><tr><td>uncached</td><td>0.32</td><td>0.27</td><td>0.83</td></tr><tr><td>Output tokens (M)</td><td>0.07</td><td>0.10</td><td>0.10</td></tr><tr><td>Cache hit rate</td><td>92.0%</td><td>90.2%</td><td>94.1%</td></tr><tr><td>Normalized cost (M)</td><td>19.75</td><td>19.83</td><td>47.00</td></tr><tr><td>Folds / compactions</td><td>1.13</td><td>3.15</td><td>0.32</td></tr><tr><td>Model requèsts</td><td>29.5</td><td>39.9</td><td>34.4</td></tr></table>

Table 2: Usage per task (means over 75 tasks). Normalized cost is $h + 3 0 u + 9 0 o$ in cached-read token equivalents; for Piki-Classic it includes its summary requests.

## 5.2 Ablations

We remove components cumulatively from the full system (A). B removes the dossier; C also removes scan and ctx; D also removes the multi-resolution outline, so the whole layout budget goes to raw text (outline share α = 0). A separate variant removes only the L1 notes. All variants keep L = 50K, the 1M window, and the landing-priced trigger.

<table><tr><td>Variant</td><td>n Succ.</td><td></td><td></td><td>Cost Folds Min</td></tr><tr><td>A FOLIO</td><td>75</td><td>66 19.75</td><td>85</td><td>10.2</td></tr><tr><td>B - dossier - L1 notes</td><td>75 75</td><td>65 63</td><td>19.73 85 19.43</td><td>10.7 80 9.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A FOLIO</td><td>15</td><td>13</td><td>18.07</td><td>16 7.8</td></tr><tr><td>B - dossier</td><td>15</td><td>13</td><td>20.89</td><td>18 12.5</td></tr><tr><td>C − scan, ctx 15</td><td></td><td>8†</td><td>18.65</td><td>18 21.7</td></tr><tr><td>D – outline</td><td>15</td><td>11</td><td>29.44</td><td>31 15.5</td></tr></table>

Table 3: Ablations (1M window, $L = 5 0 \mathrm { K } )$ . Top: all 75 tasks. Bottom: the first chain, on which C and D were run. Cost: normalized tokens per task (M); Min: minutes per task. Each of B–D removes one more component than the row above. †Two timeouts and one agent error.

Dossier and notes. On all 75 tasks, removing the dossier changes one success (66 to 65) and the cost by 0.1%; removing the L1 notes changes three successes (66 to 63) and lowers the cost by 1.6%. Running the same configuration on another chain moves its successes by 3 to 4 out of 15 (Table 1), so neither difference is distinguishable from that variation on this benchmark.

Recovery tools and outline. On the first chain, removing scan and ctx lowers successes from 13 to 8, although three of the seven failures are timeouts or agent errors and this run had 21 provider retries against none for A. Removing the outline as well has the clearest effect on cost. With $\alpha = 0 .$ the layout holds only raw results, so the context refills faster after each fold: D folds 31 times against 16 for A, issues 2.3 times as many requests, and costs 63% more per task (29.4M against 18.1M). Its success (11 of 15) is within the chain-to-chain variation of A. Keeping the outline thus does not cost budget; it lets each fold retain more of what the agent needs, so fewer folds and fewer requests are required.

## 5.3 Landing Budget

The trigger decides the folds. With L = 50K and a 1M window, all 85 folds of the 75 tasks were triggered by the landing-priced rule and none by the safety line. Figure 6 traces one session: the excess E accrues only while the cached context exceeds the landing point, so the interval between folds varies from task to task with the rate at which the context grows, and every fold returns the context close to S. On the 128K window the cap on the landing point leaves little room above it, and 201 of 236 folds were triggered by the safety line.

<table><tr><td rowspan="2"></td><td colspan="2">FOLIO fold</td><td rowspan="2">Classic (n=24)</td></tr><tr><td>1M (n=85)</td><td>128K (n=236)</td></tr><tr><td>Model calls</td><td>0</td><td>0</td><td>24</td></tr><tr><td>Latency</td><td>613 ms</td><td>346 ms</td><td>101 s</td></tr><tr><td>Context before</td><td>242K</td><td>121K</td><td>908K</td></tr><tr><td>Context after</td><td>64K</td><td>39K</td><td>58K</td></tr><tr><td>Compression</td><td>7.3×</td><td>7.0×</td><td></td></tr><tr><td>Dossier</td><td>1.2K</td><td>1.7K</td><td></td></tr><tr><td>Re-render mismatch</td><td>0</td><td>0</td><td></td></tr></table>

Table 4: Folds of the main FOLIO runs and compactions of Piki-Classic. Latency, context sizes, and ratios are medians. Compression is the folded source history over the rendered outline.

The trigger prices a fold at $K = \rho _ { w } ( L + D )$ ; on the 1M runs, $K / \rho _ { w }$ is a median 1.2 times the size the fold actually rendered, so the price of a fold is close to what it rewrites.

Sweep. We vary L from 10K to 150K on the fifth chain, running each budget once in ascending and once in descending order (Figure 5). Above 75K the raw area reaches its 60K cap, so configured budgets of 90K–150K take effect as 78K–90K while K is still priced on the configured value; we therefore read the trend from the four budgets that took effect as configured. From 10K to 70K, folds per chain fall from 49 to 13 and recoveries of folded content from 144 to 12, so a smaller landing point is paid for by more folds and more reads back. Successes rise from 11 to 13.5 of 15, though the two runs at one budget differ by up to three tasks (14 and 11 at 50K). Cost is lowest at 30K (17.5M per task) and rises on both sides: at 10K the extra folds and recoveries dominate, and above 30K each request carries more of the landing point. In the capped region cost keeps rising to 25.2M per task at a configured 150K with no further gain in success. The reference $L = 5 0 \mathsf { K }$ lies within 11% of the lowest cost and close to the highest success.

## 5.4 Cost and Correctness of the Fold

Table 4 summarizes the folds of the main runs. All were completed by the runtime without a model request, and the median fold took 613 ms on the 1M window. A fold reduced the median context from 242K to 64K tokens, compressing the folded history by a median factor of 7.3. Piki-Classic instead compacts once the context reaches about 908K tokens; each of its 24 compactions is a model request that rereads that history, and the median compaction took 101 seconds, 165 times as long as a

(a) Successes (of 15)  
![](images/3429b83ef4b7cb7b231ddef5f80bbaacf99975d019c7b561547652f124e25d09.jpg)

(b) Normalized cost per task (M)  
![](images/65b4a1b7c35b1212efdabc161e00eed63338232549ee17d91cf9acdb4c422b13.jpg)

(c) Folds and recoveries per chain  
![](images/275fbcd6b32e2206fa00e879d1d6706e3e8917f41a7f72cc155dd506a1f610a0.jpg)

Figure 5: Landing-budget sweep on the fifth chain (15 tasks; mean of an ascending and a descending run, one run at L = 10K). Recoveries are scan and ctx calls that reach content removed by the last fold. In the shaded region the raw area hits its 60K cap, so the effective budget is 78–90K (hollow markers).  
![](images/2b220c2c889a073fdd0ee73e505ed0ff9564224c1a2e4b8bbdaf3b820bfccd35.jpg)  
Figure 6: One FOLIO session (fifth chain, 1M window, 432 requests). Top: input tokens of each request and the landing point S; red dots mark the 17 folds. Bottom: carried excess relative to the fold price; a fold occurs when it reaches 1.

fold.

After every fold, the runtime rebuilt the closed turns from the L1/L2 files alone and compared the re-rendered text with the text it sent (§3.3); the check also verifies that L1 row ranges stay within their turn and that every tool output is covered by exactly one manifest row. Across the 809 folds of all LOCA-bench runs in this section, including the ablations and the sweep, none issued a model request and the check found one mismatch. It occurred in a zero-outline fold of a first run of variant D that was discarded for infrastructure errors and replaced by the run in Table 3; the other zerooutline folds passed.

The agent used the handles heavily after folds. In the main 1M runs it made 315 ctx calls, 55 scan calls, and 86 content-handle expansions; 21 of the ctx calls (6.7%) exited with a nonzero status.

## 5.5 Case Study

In a CanvasArrangeExam task of the fourth chain, the agent must build a final-exam schedule from course announcements and email. Early in the turn it fetched 160 courses (183 KB, call t8.13), 34 emails (t8.8), and announcements across calls t8.6–t8.177, and saved the course table with ctx as derived/courses. At 351K tokens the trigger folded the context to 83K in 928 ms; the raw announcements left the window and remained as rows in the outline. After the fold, the agent listed turn 8 with scan, then ran ctx over the range t8.6-177 (793 KB, 16,708 lines) to “extract final-exam status and details from every Canvas announcement call; one row per course”, receiving 6.4K tokens. It read the exam emails again through t8.8, joined the result with derived/courses, and searched both sources for exclusion signals before answering. The task passed. The 793 KB of announcements never re-entered the window; the agent reached them through the call range the outline printed.

## 6 Related Work

Organizing and compacting agent context. Observation masking drops old tool outputs (Lindenbauer et al., 2025); interaction-aware compression deletes old reasoning blocks while keeping actions and observations (Wang et al., 2026); CliffCompaction truncates tool calls and outputs by rule once a budget is spent (Nguyen et al., 2026). Generated and learned methods rewrite the history instead: ACON optimizes compression guidelines and distills them into a smaller compressor (Kang et al., 2026), and self-compaction, Context-Folding, and AgentFold train the agent to summarize sub-trajectories or fold its own workspace (Li et al., 2026; Sun et al., 2025; Ye et al., 2025). In all of these the reduced view is produced when the reduction happens, and neither selection nor generation dominates in general (Tirmazi et al., 2026). FOLIO produces its views during execution, so a fold selects among them.

Hierarchical and external memory. MemGPT pages between a bounded working context and external storage (Packer et al., 2023). HiAgent uses subgoals as memory chunks and replaces a finished subgoal’s trajectory with a summarized observation (Hu et al., 2025). AgeMem trains an agent to call store, retrieve, update, summarize, and discard operations over long- and short-term memory (Yu et al., 2026), and AgentRM manages context through a three-tier lifecycle inside an OSstyle runtime (She, 2026).Agent Workflow Memory instead reuses routines across tasks (Wang et al., 2025). FOLIO’s levels differ in that each has a fixed author and role: the model writes only a short turn note, and the runtime writes the event manifest and the source records.

Recoverable context. Several systems keep the source reachable after reduction. LCM stores every message immutably and compacts the active context into a DAG of model-written summaries that tools can grep and expand (Ehrlich and Blackman, 2026). SAM pairs raw trajectory pages with compact cues and has a trained memory model reread selected pages (Hu et al., 2026). SoL-Pi replaces large outputs with a handle and an excerpt and reduces logs to verified receipts (Liu et al., 2026). ReCAP restores raw message blocks selected from a persistent attention graph (Yang et al., 2026), and CMV versions sessions as a DAG with snapshots and structural trimming (Santoni, 2026). RLM, CLM, and VISTA let the model operate on long inputs or on its own context through code, files, or a dashboard (Zhang et al., 2025; Shao et al., 2026; Xu et al., 2026). FOLIO shares the goal of recovery; its handles are the call coordinates of the fold itself, and recovery offers bounded reads, file-side computation, and a dossier over the same coordinates.

Cost under prompt caching. Measurements of prompt caching for agents show that cachefriendly context layouts cut API cost substantially (Lumer et al., 2026), and that fewer tokens need not mean lower cost once rewrites and re-acquisition are billed (Weinberger and Hozez, 2026; Liu, 2026). Context Is Inventory derives restart intervals from an EOQ and ski-rental model (Cheng, 2026); CMV estimates how many turns a trim needs to repay its cache rebuild (Santoni, 2026); SoL-Pi’s Online Context Compact compacts at plan-step boundaries when projected savings exceed the rewrite cost (Liu et al., 2026). These models price when to reduce, but take the size after reduction as given by the reduction method.

Positioning. FOLIO is not a sum of a memory, a compactor, tools, and a trigger; the parts depend on each other through one handle space and one renderer. The levels exist before a fold, so the fold is a deterministic selection. Demotion is monotone, so the folded prefix stays cached. The fold prints the handles that L2 and L3 resolve, so recovery reads exactly the folded object and its cost lands in requests the trigger already accrues. The trigger takes the landing point from the renderer that builds the context, so the retained budget enters the cost decision. Table 5 summarizes the comparison.

## 7 Conclusion

FOLIO records a long agent history as three linked levels while the agent runs, folds it programmatically by selecting coarser views of existing rows, and lets the agent recover folded evidence through the handles the outline prints. Already sent lines are never redrawn richer, so the folded prefix stays cached, and the fold trigger prices an explicit landing point that includes the retained budget. On LOCA-bench it solves more tasks than Claude Code, Codex, SoL-Pi, and the same runtime with model-written compaction, in about half their time, and at 2.4 times lower cost than that runtime.

<table><tr><td>System</td><td>Reduced view</td><td>Model call to reduce</td><td>Source reachable</td><td>Access folded tent</td><td>to con-</td><td>Post-reduction size in trigger</td></tr><tr><td>HiAgent</td><td>subgoal sum- maries</td><td>yes</td><td>yes</td><td>retrieve subgoal trajectory</td><td>a</td><td></td></tr><tr><td>LCM</td><td>DAG of sum- maries</td><td>yes</td><td>yes</td><td>describe, grep, expand</td><td></td><td></td></tr><tr><td>CliffCompaction</td><td>rule- truncated history</td><td>no</td><td>re-run tools</td><td>tool signa- tures</td><td></td><td>fixed budget</td></tr><tr><td>SoL-Pi</td><td>handle + ex- cerpt; log re- ceipts; base</td><td>yes (compaction)</td><td>large outputs</td><td>paged han- dle read</td><td></td><td>no</td></tr><tr><td>Inventory</td><td>summary restart working floor</td><td></td><td>no</td><td></td><td></td><td>fixed at floor</td></tr><tr><td>FOLIO</td><td>L1/L2 views formed during</td><td>no</td><td>yes</td><td>call handles</td><td>scan, ctx, dossier on</td><td>yes (S = B + L + D)</td></tr></table>

Table 5: Comparison with the closest systems.

## Limitations

Evaluation scope. We evaluate one benchmark, LOCA-bench, with deterministic graders, and one model with one price structure. Each chain runs once per system, so the per-system comparison rests on 75 paired tasks; the closed systems expose no provider usage, so their cost is not compared. Variants C and D of the ablation and the landing sweep each cover one chain, and successes vary by 3–4 of 15 between chains of the same configuration. Tasks with open-ended outputs, or whose evidence can be cheaply re-queried from the environment, may benefit less from recovering captured history. Price ratios differ across providers, and the trigger’s behavior depends on them.

Recoverability boundary. The contract of §2 covers records that were captured, are still stored, and have valid handles. Content a tool never returned is outside it, and when a complete output file is missing or above 64 MB, scan returns the cut copy and says so. Recoverability does not imply that the agent finds or uses the right evidence.

Model-written state. L1 notes and dossier entries are written by the model and can be wrong or incomplete. An oversized dossier is consolidated by an auxiliary model request, which can drop or merge entries incorrectly.

Execution of ctx. ctx runs commands the model writes. Its isolation and permission model depend on the surrounding runtime, which this paper does not evaluate.

Fixed landing budget. We use one landing budget L and report its sensitivity on one chain; we do not choose L per task. Budgets above 75K were limited by the raw-area cap.

## References

Longju Cheng. 2026. Context is inventory: A rent-orbuy model for prompt-cached LLM sessions. Zenodo preprint.

Clint Ehrlich and Theodore Blackman. 2026. LCM: Lossless context management. arXiv preprint arXiv:2605.04050.

Mengkang Hu, Tianxing Chen, Qiguang Chen, Yao Mu, Wenqi Shao, and Ping Luo. 2025. HiAgent: Hierarchical working memory management for solving long-horizon agent tasks with large language model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32779–32798, Vienna, Austria. Association for Computational Linguistics.

Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Ziliang Zhao, Jiejun Tan, Zheng Liu, and

Zhicheng Dou. 2026. SAM: State-adaptive memory for long-horizon reasoning agent. arXiv preprint arXiv:2605.24468.

Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, and Saravan Rajmohan. 2026. ACON: Optimizing context compression for long-horizon LLM agents. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 55321–55375. PMLR.

Tianjian Li, Jingyu Zhang, William Jurayj, Xi Wang, Chuanyang Jin, Mehrdad Farajtabar, Eric Nalisnick, and Daniel Khashabi. 2026. Selfcompacting language model agents. arXiv preprint arXiv:2606.23525.

Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, and Yaroslav Zharov. 2025. The complexity trap: Simple observation masking is as efficient as LLM summarization for agent context management. arXiv preprint arXiv:2508.21433.

Haozhe Liu, Tian Ye, Sensen Gao, Qihang Cao, Yitong Li, Mingchen Zhuge, Duomin Wang, Ruihua Zhang, Ping Luo, Jiawang Bian, Lei Zhu, Ligeng Zhu, Enze Xie, and Song Han. 2026. SoL-Pi: Recursively scaling auto-research loops for efficient agent harness. arXiv preprint arXiv:2609.20519.

Shuyu Liu. 2026. What does context compression cost an agent? interaction costs unrevealed by task-completion metrics. arXiv preprint arXiv:2608.16370.

Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, and Vamse Kumar Subbiah. 2026. Don’t break the cache: An evaluation of prompt caching for long-horizon agentic tasks. arXiv preprint arXiv:2601.06007.

Trang Nguyen, Eulrang Cho, Bingqing Chen, and Tim Dettmers. 2026. CliffCompaction: Cost-efficient compaction for long-horizon coding agents. arXiv preprint arXiv:2609.26779.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Cosmo Santoni. 2026. Contextual memory virtualisation: DAG-based state management and structurally lossless trimming for LLM agents. arXiv preprint arXiv:2602.22402.

Rulin Shao, Shannon Zejiang Shen, Junjie Oscar Yin, Yuetai Li, Minheng Wang, Hamish Ivison, Radha Poovendran, Nathan Lambert, Teng Xiao, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, and Pang Wei Koh. 2026. Context language models. arXiv preprint arXiv:2609.37725.

Jianshu She. 2026. AgentRM: An OS-inspired resource manager for LLM agent systems. arXiv preprint arXiv:2603.13110.

Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. 2025. Scaling long-horizon LLM agent via context-folding. arXiv preprint arXiv:2510.11967.

Hayder Tirmazi, Sam Markelon, Allison Bishop, and Michael Mitzenmacher. 2026. Context compaction theory. arXiv preprint arXiv:2608.01326.

Mingxuan Wang, Fei Luo, Bo Wang, Guorun Yao, Yinglong Guo, Chao Ning, Hongyue Chen, Yanbiao Ma, and Jungong Han. 2026. When can agents forget their reasoning? ICLR for long-horizon agent context compression. arXiv preprint arXiv:2609.29875.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2025. Agent workflow memory. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 63897–63911. PMLR.

Sarel Weinberger and Amir Hozez. 2026. Token reduction is not cost reduction. arXiv preprint arXiv:2607.12161.

Binyan Xu, Haitao Li, and Kehuan Zhang. 2026. LLM agents are latent context managers: Eliciting selfmanaged context via a proprioceptive dashboard. arXiv preprint arXiv:2606.30005.

Jingbo Yang, Kwei-Herng Lai, Xiaowen Wang, Zhaoxuan Tan, Pei Zhou, Mengting Wan, Yaar Harari, Evgeniy Gabrilovich, and Shiyu Chang. 2026. Persistent context graphs for efficient memory compaction in LLM agents. arXiv preprint arXiv:2609.40118.

Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, Pengjun Xie, Fei Huang, Siheng Chen, Jingren Zhou, and Yong Jiang. 2025. AgentFold: Long-horizon web agents with proactive context management. arXiv preprint arXiv:2510.24699.

Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, and Libing Wu. 2026. Agentic memory: Learning unified long-term and shortterm memory management for large language model agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21457–21483.

Weihao Zeng, Yuzhen Huang, and Junxian He. 2026. LOCA-bench: Benchmarking language agents under controllable and extreme context growth. arXiv preprint arXiv:2602.07962.

Alex L. Zhang, Tim Kraska, and Omar Khattab. 2025. Recursive language models. arXiv preprint arXiv:2512.24601.