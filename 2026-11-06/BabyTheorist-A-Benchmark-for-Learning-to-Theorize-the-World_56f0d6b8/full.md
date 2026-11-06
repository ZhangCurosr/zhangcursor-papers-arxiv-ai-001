# BabyTheorist: A Benchmark for Learning to Theorize the World from Observation Alone

Doojin Baek<sup>∗†</sup> Junyeob Baek<sup>∗†</sup> Mingyu Jo

Hosung Lee Taegu Kang Gyubin Lee Sungjin Ahn

KAIST

§ Code: github.com/ahn-ml/babytheorist <sup>õ</sup> Dataset: huggingface.co/datasets/ahn-ml/babytheorist

## Abstract

How would we know whether a machine truly understands the world? We argue that one key capability is observational theory learning: acquiring reusable theories from pure observations and applying them to new situations. Yet current visual reasoning benchmarks, such as ARC-AGI, mainly evaluate the latter, presenting each task as a few-shot puzzle grouped by a common rule. Whether models acquire reusable theories from observations remains untested. We introduce BabyTheorist, a benchmark that brings this acquisition under controlled training and evaluation. BabyTheorist generates continual streams of before-after visual observations from hidden compositional programs; during training, learners see only the resulting transitions, with no program labels, task identities, or support sets. To support theory learning and evaluation, BabyTheorist combines three design controls—a hidden program language, frequent subprograms, and a curriculum over program complexity—with a continual train-test protocol featuring level skipping. Experiments with seven representative baselines show that BabyTheorist poses challenges that current program-learning approaches have yet to resolve.

## 1 Introduction

Long before they learn to solve problems, children learn how the world works. They watch ordinary phenomena unfold—objects falling, water freezing, people acting—and gradually accumulate reusable theories that explain them. Faced with a novel situation, they do not start from scratch but recombine theories they have already built. This suggests two distinct capabilities behind everyday understanding: theory acquisition from raw observation, and theory application, which recombines acquired theories to explain new situations (Figure 1).

Current visual reasoning benchmarks focus on the second. ARC-AGI [1–3], Bongard problems [4, 5], and Raven-style suites [6] each present a task as a pre-grouped support set sharing a hidden rule, then test whether a model can infer the rule and apply it to a query. This setting is valuable, but the process preceding such rule inference—accumulating reusable and composable primitives and rules from many ungrouped observations—is left outside the evaluation. Thus, a model can be scored on few-shot transfer without being asked how it acquired the theories behind that transfer.

We argue that progress toward observational theory learning— acquiring reusable theories from observations and applying them to new situations—requires closer study of theory acquisition from ungrouped observations as experience accumulates. Building a benchmark for this phase is the central goal of this paper. We further commit to a learning-focused stance: rather than giving learners a predefined domain-specific language (DSL) or primitive vocabulary for symbolic search, our setup asks whether neural learners can discover the relevant primitives and theories directly from raw observations.

![](images/97034de64aa47cddb2c3ef1a518259ad6bad4012edf0f9fb52663cf0b97be00e.jpg)

From paradigm to benchmark. Studying observational theory learning first requires a formal definition. The Learning-to-Theorize (L2T) paradigm [7] considers a learner that observes a scene before a change, x, and after it, y, without knowing which operations transformed one into the other. Each such observation pair (x, y) is called a phenomenon. To explain the observed change, the learner must discover reusable operations, called primitives, and infer how they were composed to transform x

Figure 1: Observational theory learning. Humans learn about the world through theory acquisition and theory application. For example, observations of falling objects can lead to a theory of gravity, which can then be applied to explain orbital motion.

into y. An ordered sequence of these primitives forms a program τ. Neither the primitives nor the programs are provided; the learner receives only raw (x, y) pairs, without program annotations, action labels, or task groupings. This commitment is intentionally minimal—any pair of observations connected by a hidden mechanism is a valid training example, opening the door to uncurated observational data at scale. However, L2T’s models were trained on static batch datasets with programs of depth 1–3, leaving theory acquisition over experience—building and updating a theory system as phenomena grow—out of scope.

BabyTheorist. We close this gap with BabyTheorist, a benchmark that extends L2T’s observational learning setting from static batch datasets to a continual curriculum of increasingly complex phenom ena. It provides three design controls: (i) a hidden DSL defines the primitives and their composition rules without revealing them to the learner; (ii)frequent subprograms introduce recurring sequences of primitives that learners can discover and reuse; and (iii) a level-based curriculum introduces increasingly complex compositions. Building on these controls, our continual train–test protocol is designed to reward the acquisition of reusable theories. Each level begins with an entry evaluation; learners that pass skip that level’s training observations (level skipping), testing whether their acquired theories transfer to increasingly complex phenomena.

Where existing methods stand. To characterize what BabyTheorist measures, we evaluate seven representative baselines: black-box predictors (LLMs, autoregressive Transformers), monolithic latent encoders (Cont-Mono, Disc-Mono) and their variants (Cont-Mono-Opt, Cont-Mono-OC), and the compositional discrete-program model NEO [7]. All baselines stall within the first two of the eight curriculum levels. Analyzing their errors, we identify three challenges (Sections 5.3–5.5): learners struggle to discover reusable primitives, to learn where to apply a transformation in multi-object scenes, and to explain higher-depth phenomena that combine several primitives.

Contributions. The main contributions of this paper are as follows:

• A learning-focused benchmark. BabyTheorist evaluates theory acquisition from ungrouped observations and application to new tasks throughout a continual curriculum.

• Three design controls. A hidden DSL, frequent subprograms, and a level-based curriculum enable controlled study of observational theory learning across four benchmark modes.

• A continual evaluation protocol. We design a continual train–test protocol with level skipping to reward theory reuse as program complexity increases, ranking learners by SCORE and using EFFICIENCY to break ties in favor of earlier progress.

• An empirical assessment. Seven representative baselines show limited progress through the curriculum, revealing challenges that existing approaches have yet to resolve.

## 2 Learning to Theorize the World Just by Seeing

To formalize observational theory learning (Figure 2), we build on Learning-to-Theorize [7], which models the hidden mechanism behind each observation pair as a latent compositional program.

![](images/fa012e18f78696915b151596747dbae3e0affd6b7e398c8708402d1d9ee0528a.jpg)  
Figure 2: Overview of continual observational theory learning. A useful theory system should continually evolve as a learner receives a stream of observations. During theory acquisition (blue arrows), the learner discovers new theories when existing ones cannot explain an observation (Mirror) and abstracts recurring compositions into reusable units (Up + Rotate). Theory application combines them to explain new observations.

Theorize Just by Seeing. Imagine watching a stream of (before, after) image pairs in which a hidden rule transforms each before-image into its after-image. A standard predictor scored on reproducing the after-image can succeed without ever isolating the rule. We ask a different question: Can a learner, just by watching, recover the hidden mechanisms behind what it sees and apply them to situations it has never encountered? L2T [7] answers by treating each (before, after) pair as a phenomenon and the rule behind it as a program: a short sequence of primitive operations whose execution reproduces the observed transformation. The learned primitives form a vocabulary of reusable operations, akin to the “mental programs” of compositional cognition [8, 9].

Phenomena and programs. Let X denote the space of observations. A phenomenon is an observation pair (x, y), where $x \sim p ( x )$ is a source observation and $y$ is a target observation generated by an unobserved program τ that transforms x into $y .$ Following L2T, we represent each program as a sequence of primitive operations, each transforming one observation into another. $\operatorname { L e t } { \mathcal { Z } } = \{ z _ { 1 } , \dots , z _ { M } \}$ denote the set of M primitives, with each primitive $z _ { i }$ associated with an execution function $f _ { z _ { i } } : \mathcal { X }  \mathcal { X }$ . A program of length $K$ is an ordered sequence of primitives, $\boldsymbol { \tau } = ( z _ { i _ { 1 } } , \dots , z _ { i _ { K } } ) \in \mathcal { Z } ^ { K }$ . Executing these primitives in order gives:

$$
f _ { \tau } = f _ { z _ { i _ { K } } } \circ f _ { z _ { i _ { K - 1 } } } \circ \cdot \cdot \cdot \circ f _ { z _ { i _ { 1 } } } , \qquad y = f _ { \tau } ( x ) .\tag{1}
$$

In this formulation, a learner’s theory of a phenomenon is represented as an executable program that explains the transformation from x to $y .$ The number of possible primitive sequences of length K is $\grave { M } ^ { K }$ , so exhaustive coverage becomes increasingly costly as program length grows. This motivates learning reusable primitives that can be recombined into new executable theories to explain phenomena beyond those observed during training.

Training data. L2T assumes a training dataset of i.i.d. observation pairs,

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { t r a i n } } = \{ ( x _ { n } , y _ { n } ) \} _ { n = 1 } ^ { N } , \qquad y _ { n } = f _ { \tau _ { n } } ( x _ { n } ) , \qquad \tau _ { n } \sim p _ { \mathrm { t r a i n } } ( \tau ) . } \end{array}\tag{2}
$$

Crucially, the programs $\left\{ \tau _ { n } \right\}$ and primitive execution functions $\{ f _ { z _ { i } } \}$ are latent and never observed. The learner receives only pairs $( x _ { n } , y _ { n } )$ , without program annotations, symbolic rules, action labels, or task groupings. Unlike few-shot abstract reasoning benchmarks such as ARC-AGI, where examples must be grouped by a shared hidden program, or imitation-learning and action-conditioned world-modeling settings, where transition-mediating actions are observed, L2T requires only raw observational pairs for training. For example, temporally separated trajectory observations, such as $\left( { { x } _ { t } } , { { x } _ { t + t ^ { \prime } } } \right)$ , already form valid training data. Thus, L2T opens the possibility of large-scale training on uncurated observational datasets.

Three capabilities. Theorizing the world, in this formulation, requires a model to (i) discover reusable abstract primitives across phenomena, (ii) learn how to compose them into structured explanations, and (iii) explain novel phenomena by forming new compositions of the same primitives. In L2T, the model is evaluated at test time on programs drawn from a set $\mathcal { T } _ { \mathrm { t e s t } }$ disjoint from the programs used to generate training observations, including programs longer than those seen during training—jointly probing compositional and length generalization.

![](images/0230e10f3e6116dfa405f60d8775a05c8d8de52b923bc7dda3b0dc061bab4e68.jpg)  
Figure 3: BabyTheorist program language and task generation. (a) Each program step pairs a selector, which identifies the objects to transform, with a primitive, which specifies the transformation. (b) The illustrated depth-6 program combines frequently recurring subprograms of depths 1, 3, and 2. Learners that discover such subprograms can reuse them to efficiently explain more complex phenomena.

Evaluating transferability. A central evaluation principle is transferability. Given two phenomena $( x ^ { ( 1 ) } , y ^ { ( 1 ) } )$ and $( x ^ { ( 2 ) } , y ^ { ( 2 ) } )$ generated by the same latent program τ, a model first induces τˆ from the first pair and then applies it to $x ^ { ( 2 ) }$ to predict $y ^ { ( 2 ) }$ . Transferability measures whether τˆ has captured a reusable mechanism rather than information specific to $( x ^ { ( 1 ) } , y ^ { ( \mathrm { i } ) } )$ . Self-explainability—re-applying τˆ to $x ^ { ( 1 ) }$ —can be passed by memorization; transferability cannot.

## 3 The BabyTheorist Environment

BabyTheorist turns L2T into a controlled benchmark that tests whether learners can discover, compose, reuse, and update theories from observations alone. We make this setting concrete through three controls: (i) a program language defining hidden compositional programs (Section 3.1), (ii) frequent subprograms that recur across observations (Section 3.2), and (iii) a level-based curriculum that gradually increases program complexity (Section 3.3). Building on these controls, BabyTheorist evaluates learners before and after training at each level (Section 3.4) and measures how far and how quickly they progress through the curriculum (Section 3.5).

## 3.1 The Program Language of BabyTheorist

Observations. The program language defines how BabyTheorist generates observations from hidden programs. Each observation pair consists of a before grid x and an after grid $y ,$ generated by a hidden program that maps x to y. Each grid is a $1 5 \times 1 5$ array of colors. Color 0 denotes the background. Objects are connected colored shapes rendered with non-background colors $1 , \ldots , 9 ;$ they may vary in color pattern, position, and shape, as detailed in Appendix A.1.

Programs. To apply a primitive to a scene, a program must specify which objects it acts on. Each program therefore consists of steps that pair a what component, given by a selector, with a how component, given by a primitive (Figure 3a):

$$
\mathbf { s t e p ( s e l e c t o r , \ p r i m i t i v e ) . }
$$

The number of steps in a program is its depth, corresponding to the program length K from Section 2.

Selectors define the what: the object or set of objects to which a step applies. BabyTheorist includes four kinds of selection rules: (i) global rules over all objects in the scene, (ii) attribute-based rules using object attributes such as color or hollowness, (iii) spatial rules based on positional criteria such as topmost or leftmost, and (iv) index-based rules using a fixed raster-scan ordering. The full selector vocabulary is given in Appendix A.2.

Primitives define the how: the type of change applied to the selected objects. BabyTheorist groups primitives into three categories: (i) transformation primitives, which modify an object’s color, shape, interior, or boundary, (ii) movement primitives, which change an object’s position, and (iii) relational primitives, which modify one object with respect to another. The full primitive vocabulary and execution semantics are given in Appendix A.3.

Table 1: BabyTheorist benchmark modes. The Single/Multi axis varies scene structure and relational reference, while the A/B axis varies whether the primitive vocabulary expands across levels. Program space counts nominal depth-d step sequences, including object/selector and primitive choices, after all primitives become available (Appendix A.4).
<table><tr><td>Mode</td><td># Objects</td><td>Selectors</td><td># Primitives</td><td>Primitive Expansion</td><td>Program space</td></tr><tr><td>Single-A</td><td>1</td><td>一</td><td>8</td><td></td><td>8d</td></tr><tr><td>Single-B</td><td>1</td><td></td><td>11</td><td>√(8→ 11)</td><td>11d</td></tr><tr><td>Multi-A</td><td>2</td><td>by_index</td><td>11</td><td></td><td>22d</td></tr><tr><td>Multi-B</td><td>3</td><td>all selectors</td><td>14</td><td>√(11→ 14)</td><td>216d</td></tr></table>

## 3.2 Frequent Subprograms for Reusable Abstractions

Natural environments are not arbitrary collections of unrelated events. Some patterns recur across many situations and can appear within more complex patterns. BabyTheorist mirrors this structure by making some primitive sequences recur across different programs. We call these recurring sequences frequent subprograms. Reusing these subprograms can reduce the effective search space for explaining more complex observations. This design therefore rewards learners capable of discovering and compressing recurring structure into reusable abstractions.

Subprogram pools. To implement this structure, BabyTheorist maintains a frequent subprogram pool for each program depth, hidden from the learner (Figure 3b). The depth-1 pool contains all available primitives. Deeper pools are built by combining shorter subprograms and keeping a limited number of the resulting sequences. Each entry preserves both its primitive sequence and which steps share a target. For example, “move Object up, then Object right” reuses one target, whereas “move Object up, then Object right” allows different targets. Targets are represented by placeholders, which receive concrete selectors during observation generation. The same subprogram can therefore recur across scenes while preserving its target relationships.

Program sampling. To generate a program of depth d, the generator either takes a complete program from the depth-d pool or chains shorter pool entries, including single primitives, until their depths sum to d; for example, the depth-6 program in Figure 3b chains entries of depths 1, 3, and 2. For depths 2–10, each option is chosen with probability 50%. As a result, each pool entry appears both as a program of its own depth and inside longer programs, so the same subprograms recur across observations of varying complexity. Appendix A.6 details pool construction, sampling, and primitive expansions.

## 3.3 A Curriculum for Continual Theory System Updates

A useful theory system should evolve by recognizing which phenomena its current theories can explain and which require a new theory. BabyTheorist studies this process through a level-based curriculum with increasing program depth and an expanding primitive vocabulary. These changes require learners to build new theories from familiar primitives and to acquire new ones when needed.

Per-level depth distributions. We implement this curriculum by defining each level as a distribution over program depths rather than a single depth. As the curriculum advances, this distribution shifts toward longer programs while still including shorter ones. Together with the frequent subprograms introduced in Section 3.2, this exposes learners to structures that recur across levels and become components of more complex transformations. Appendix A.7 specifies the depth distribution.

Keeping depth tied to complexity. To make program depth informative about transformation complexity, we apply rule-based filtering to screen out degenerate cases: programs whose observed transformations admit shorter explanations because of redundant or mutually cancelling steps (e.g., moving an object right and then left). Details of the rule-based filter are provided in Appendix E.

Vocabulary expansion. The curriculum can also introduce new primitives at later levels, adding one transformation primitive that alters an object’s interior or boundary at each of Levels 2, 4, and 6 (Appendix A.3). The learner must then go beyond recombining learned primitives and expand its theory vocabulary. Each new primitive first appears within inherited subprograms and then joins the existing pools (Appendix A.6).

![](images/d1e09fe5f96f75b5034ad2bbc25a60255697be606ed148839b1dde5a5fa19fe3.jpg)  
Figure 4: Observations across modes and levels. Each pair shows one before–after observation from Levels 1, 2, and 8 of each mode. Observations become more complex through longer hidden programs, more objects, relational structure, and primitive expansion.

Benchmark modes. We define four benchmark modes (Table 1; examples in Figure 4). The Single/Multi distinction controls scene structure: Single modes use one object, while Multi modes use multiple objects and relational primitives. The A/B distinction controls vocabulary expansion: A modes keep the primitive vocabulary fixed, while B modes introduce one new primitive at each of Levels 2, 4, and 6. Multi-B further increases the number of objects from two to three and expands the selector vocabulary to all selector types. As objects, selectors, and primitives increase, the program space grows rapidly, making theory-system expansion beneficial. Appendix A.4 gives the program-space counts.

## 3.4 Evaluating Theory Reuse and Length Generalization

To evaluate learners as the depth distribution shifts toward longer programs, BabyTheorist uses a continual train-test protocol over eight official levels (Figure 5). At each level, the learner receives 250,000 training observations (Appendix A.8) and then takes the level evaluation. It proceeds to the next level if it achieves at least 80% accuracy; otherwise, the curriculum ends. Notably, the protocol includes level skipping. Before training at each level, the learner takes an entry evaluation. If it meets the same passing threshold, it proceeds directly to the next level without receiving that level’s training observations. Successful skipping thus indicates length generalization to a more complex level, using the theories acquired earlier, without additional training on that level.

![](images/77a6f012710e54ad98c3b4ae5b0e00bd8b3f843e59243701768fc6e770bce989.jpg)

Each level evaluation consists of 10,000 tasks. Each task provides a support set of 2–5 before–after pairs generated by the same hidden program and a query input; the learner then predicts the query output. Importantly, the few-shot format is an evaluation device, not a training signal for theory acquisition. The supports reduce ambiguity about the task’s hidden program. We start with two pairs and check whether the selection rules that explain them give the same query output. If not, we add more supports, up to five.

Figure 5: Continual train–test protocol. At the start of level ℓ, the learner is evaluated before training. Passing (at least 80% accuracy) enables level skipping (red arrow): the learner advances without that level’s training observations, rewarding transfer to increasingly complex levels.

## 3.5 Measuring Theory Learning Through Curriculum Progress

Because hidden programs, observation complexity, and shared subprogram structure are never labeled, our metrics are designed to reward discovering reusable theories rather than fitting provided rules or memorizing task identities. A strong learner should discover primitive theories, compose and reuse them, and expand the system as new primitives appear.

We rank learners by SCORE and use EFFICIENCY to break ties. SCORE, our primary metric, measures how much of the curriculum the learner solves. Let $a \ell$ denote the exact-match accuracy of the evaluation that concludes level ℓ: the entry evaluation if skipped, otherwise the post-training evaluation. A task is solved only if the predicted output grid exactly matches the target. For example, solving 8,000 of the 10,000 tasks gives $a _ { \ell } = 0 . 8$ , earning the learner 80 points at that level. We sum these points as $\begin{array} { r } { \mathrm { S c o R E } = \sum _ { \ell = 1 } ^ { L _ { \mathrm { r e a c h e d } } } 1 0 0 a _ { \ell } } \end{array}$ , where $L _ { \mathrm { r e a c h e d } }$ denotes the final curriculum level attempted by the learner.

EFFICIENCY breaks ties in final SCORE by favoring learners that earn their points earlier in training. We count one training stage for each level trained on. Skipped levels add points without increasing this count. Let $S _ { t }$ denote the cumulative SCORE after t training stages, including points from levels skipped before the next stage, and let $S _ { \mathrm { f i n a l } }$ be the final SCORE. The eight-level curriculum allows at most $T _ { \mathrm { m a x } } = 8$ training stages. To compare learners over all $T _ { \mathrm { m a x } }$ stages, we set $S _ { t } = S _ { \mathrm { { f i n a l } } }$ after a learner stops. EFFICIENCY averages the fraction of final SCORE accumulated by each stage: EFFICIENC $\begin{array} { r } { \ell = \frac { 1 } { T _ { \mathrm { m a x } } S _ { \mathrm { f i n a l } } } \sum _ { t = 1 } ^ { T _ { \mathrm { m a x } } } S _ { t } } \end{array}$

Since BabyTheorist controls how observations are generated and how frequent subprograms are reused throughout the curriculum, high SCORE and EFFICIENCY are intended to serve as behavioral evidence for observational theory learning. These objectives do not prescribe a particular theory representation: theories may be represented discretely, continuously, or implicitly within the learner.

## 4 Related Works

BabyTheorist relates to four lines of work. (i) Program-induction and abstract-reasoning suites (ARC-AGI [1–3], COGITAO [10], Bongard [4, 5], RPM [6]) and (ii) compositional-generalization suites (SCAN [11], COGS [12], gSCAN [13], CLEVR/CLEVRER [14, 15]) provide structured supervision, such as pre-grouped tasks or labeled primitive actions; BabyTheorist instead provides an ungrouped, unlabeled observation stream in which hiddenfrequent subprograms recur across a level curriculum. (iii) Library-learning systems (DreamCoder [16], Bayesian program induction [17]) accumulate reusable abstractions on hand-curated suites, while BabyTheorist makes the abstraction set, its frequency, and its expansion controllable, and uses level skipping to test reuse before a level’s training observations are seen. (iv) World modeling (PlaNet [18], Dreamer [19–21]) and latent action learning (LAPO [22], AdaWorld [23], LAPA [24], Genie [25]) primarily target rollout prediction or control; BabyTheorist instead evaluates whether explanations inferred from observations transfer to new inputs, rather than only reconstructing the observed transitions, as the hidden programs grow deeper across the curriculum. See Appendix D for full treatment.

## 5 Experiments

We evaluate whether existing methods acquire reusable theories from ungrouped observations. After reporting curriculum performance, we examine three challenges revealed by their predictions: discovering reusable primitives, learning where to apply a transformation, and explaining higherdepth phenomena. Each analysis connects observed failures to a research direction for observational theory learning.

## 5.1 Baselines

We group baselines by how they represent the underlying program: implicit models predict outputs without an explicit program latent, whereas explicit latent models infer one, either as a single monolithic latent or as a compositional sequence of learned primitives.

(i) Implicit models. The autoregressive Transformer (AR) is trained on the same observation stream to generate the output grid token by token from the input, so any acquired theory remains implicit in its weights. We also include pretrained LLMs (GPT-OSS 120B and Qwen 3.5 27B), which are not trained on the stream and generate outputs in the same token-by-token manner.

(ii) Monolithic latent models. Cont-Mono and Disc-Mono infer a single continuous or discrete latent program from a before–after pair, respectively, following latent action models such as AdaWorld [23] and LAPO [22]. A learned transition model then predicts the output from the input and this latent program. We consider two extensions of Cont-Mono. Cont-Mono-Opt refines the inferred latent program to fit the support pairs at test time, similar to LPN [26]. Cont-Mono-OC uses object-centric representations and infers a separate latent action for each slot, following FLAM [27].

![](images/8d9d608206afae3744205ed243634b59a162c1c082e653f76d2d7af0074b6658.jpg)  
Figure 6: SCORE and highest curriculum level attempted in each benchmark mode. No baseline progresses beyond Lv2. SCORE sums 100× accuracy over attempted levels, so only models that reach Lv2 can exceed 100. Trained baselines show the mean ± one population standard deviation over three seeds; LLMs are evaluated once with greedy decoding on the first 100 tasks at each attempted level. Cont-Mono-OC is evaluated only on Multi-A and Multi-B. Labels above the bars give the highest level attempted.

(iii) Compositional latent models. NEO [7] infers a compositional program as an ordered sequence of discrete primitives from a shared vocabulary and executes their learned operations sequentially to predict outputs. This allows each learned primitive to be reused across many observed phenomena.

Evaluation setup. All models are evaluated on the same support–query tasks, but each uses the supports in a way suited to its representation, such as selecting among inferred programs, optimizing a latent, LoRA fine-tuning, or in-context prompting (Appendix C). Due to inference cost, we evaluate LLMs on the first 100 tasks at each attempted level. Architectures and hyperparameters appear in Appendices B and F.

## 5.2 Overall Performance: Limited Progress Through the Curriculum

Every baseline stalls within the first two of the eight curriculum levels (Figure 6; full results in Table 9, Appendix G.1). NEO alone reaches Level 2, only in the Single modes, and every learned baseline fails Level 1 in Multi. General-purpose reasoning, without learning the environment’s rules from the observation stream, does not close this gap. The LLMs fail Level 1 in every mode, with GPT-OSS 120B and Qwen 3.5 27B reaching at most 59% and 8%, and even the frontier model GPT-5.6 Sol reaches only 51–64% (Appendix G.3), well below the 80% threshold. A diagnostic learner, PPO with Known DSL, helps locate the difficulty. Given the ground-truth primitives, selectors, and executor, it clears all eight levels of Single-A (Appendix G.2). It is not a matched comparison, since it never has to discover the operations, but it points to acquiring the operations as a major source of difficulty in the Single modes; selecting and composing them remains hard in richer multi-object settings, particularly Multi-B, even when they are provided. We next examine where the learned baselines fail, identifying three challenges for future methods.

## 5.3 Challenge 1: Discovering Reusable Primitives

Before learners can build more complex theories, their basic transformations must work on new inputs. However, we found that models trained on the observation stream can explain all supports yet fail to transfer to the query, suggesting that their inferred programs fit the specific support pairs rather than capture a reusable transformation.

![](images/a93510c05cb00e21b466f5bb5da6b92819c2466905a755d719b7e6709eb11422.jpg)  
Figure 7: Transfer failures despite exact support fit. Five failure patterns on Lv1 tasks: (a) incorrect displacement (AR), (b) incorrect flip axis (Cont-Mono), (c) two color shifts instead of one (Cont-Mono), (d) a rotation-only output when rotation followed by a flip is required (Disc-Mono), and (e) the correct color change applied to the wrong object (NEO; dashed boxes mark the changed object). Rows show Input, Target, and Prediction. Both supports are predicted correctly (one shown), but the query is incorrect; green/red borders mark correct/incorrect predictions. Examples (a–d) use Single-A/B and (e) uses Multi-A.

Explaining the Supports, Failing on the Query. The gap appears even on single-object tasks requiring just one ground-truth primitive, without the additional demands of target selection or composition. Cont-Mono explains all supports yet fails to transfer to the query in 23.51% of tasks, compared with 8.48% for Disc-Mono and 2.36% for NEO (Table 14; examples in Figures 7a–c, 15, and 16; latent visualizations in Appendix I). Cont-Mono-Opt refines the inferred latent within the learned program space to fit the supports, yet yields no consistent improvement in SCORE over Cont-Mono (Figure 6); in the depth-one analysis, the query errors it corrects are largely offset by new ones (Table 16; Figure 17). The results motivate learning a program space that favors reusable transformations in the first place. NEO’s compositional representation, built from a shared primitive vocabulary, offers one design direction.

## 5.4 Challenge 2: Learning Where to Apply a Transformation

Reusing a primitive in a multi-object scene also requires knowing which object it should affect. We found that baselines with an explicit program latent, including NEO, can produce the correct primitive effect but apply it to the wrong query object, even after explaining all supports.

The Right Transformation on the Wrong Object. To examine target selection without multi-step composition, we analyze Multi-A Lv1, which uses index-based selectors throughout training and evaluation. Among depth-one tasks in which only one of multiple objects changes, NEO applies the correct transformation to a different object in 25.43% of cases, averaged across three seeds (Table 16; Appendix H.1). Figures 7e and 18c illustrate how such an error can arise. NEO changes the rightmost query object, although the task requires the first object in raster order. The two selection rules agree on both supports but select different objects in the query. The predictions therefore match a spatial rule that explains the supports but is absent from Multi-A’s selector vocabulary. The example show the need to learn the environment’s selection conventions and apply them to new object layouts.

Object-Centric Representations Still Need Selection Rules. Making individual objects explicit is one way to support target selection. Cont-Mono-OC uses object slots and learns their alignment from the supports to transfer object-specific actions. However, separating objects does not by itself specify which one to change; that choice still requires a learned selection rule. Object-centric structure therefore does not automatically guarantee better transfer: Cont-Mono-OC’s mean SCORE is 5.1 points higher than Cont-Mono’s in Multi-B but 7.3 points lower in Multi-A (Table 9). Future work should investigate learning reusable selectors from the observation stream and composing them with primitive transformations, separating which object to change from how to change it.

## 5.5 Challenge 3: Explaining Higher-Depth Phenomena

To explain higher-depth phenomena, learners must capture the combined effect of several primitive operations. Current methods struggle to meet this requirement in two ways. Models sometimes reproduce only part of the required change. Consistent execution of learned primitives remains a challenge even with compositional representations.

![](images/4b744d931bd00fc263e652c404b94679eb076e40ef3abfbac22a067534215064.jpg)  
Figure 8: Step-wise inference succeeds where joint inference fails. Joint predictions are wrong despite exact support fit. Given ground-truth intermediate supports, inferring a theory for each program step separately (Separate) yields correct intermediate and final outputs. Each step’s theory is inferred from its first support pair. This diagnostic adds supervision and changes inference, so it is not a baseline comparison.

Explaining the Whole Phenomenon. We found cases where a model explains only part of the required change. In Figure 7d, Disc-Mono’s output matches a rotation alone, although the phenomenon requires rotation followed by a flip. To probe this failure, we additionally give the model the ground-truth intermediate outputs of the supports and infer a theory for each step separately. Applied in sequence, these step-wise theories produce the correct query outputs for the cases in Figure 8, suggesting that the model can represent each step but struggles to decompose the combined change from before–after pairs alone. Moreover, on depth-two tasks, NEO achieves 88.2% query accuracy, compared with 61.6% for Disc-Mono (Table 15). This contrast motivates investigating compositional inductive biases that build complex explanations from reusable operations, as in NEO.

Consistent Execution of Learned Primitives. A learned primitive should have a consistent effect. However, we observed some cases where NEO’s learned primitives produce inconsistent effects during execution (Appendix H.4). NEO’s state-grounding loss is intended to reduce such inconsistencies (Appendix B), but these cases motivate further study of its formulation and tuning.

## 6 Limitations and Conclusions

We introduced BabyTheorist, a controlled benchmark for studying observational theory learning: acquiring reusable theories from streams of ungrouped before–after observations and applying them to new situations. The acquisition process is often left outside few-shot evaluations that present tasks already grouped by a hidden rule. Across seven representative baselines, all methods plateau well below ceiling. Compositional structure offers a promising direction, but even NEO—the strongest baseline—reaches only Level 2 in the Single modes and fails Level 1 in the Multi modes. Our analyses highlight difficulties in reusing primitives, selecting the intended objects, and explaining higher-depth phenomena. BabyTheorist is intentionally narrow—visually simple grids, a small primitive set, and deterministic dynamics—so the findings are limited to this controlled setting. With most of the curriculum still unsolved by the evaluated baselines, much remains to be done, and richer settings, real-world transfer, and the link to causal mechanism discovery are natural next steps.

## Acknowledgements

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (No. RS-2024-00509279, Global AI Frontier Lab; No. RS-2026-25518317, Development of AI memory mechanism that reflects human cognitive principles), the InnoCORE program of the Ministry of Science and ICT (AI Meta-Scientist, N10260110), and the Youlchon Foundation.

## Broader Impact

BabyTheorist is a benchmark for observational theory learning on synthetic grid-based observations; it contains no human, personally identifiable, or otherwise sensitive data, and its outputs are not directly deployable as user-facing systems. The intended positive impact is scientific: enabling controlled study of how learners acquire reusable, compositional theories from raw observations, which we believe is a prerequisite for sample-efficient, generalizable, and interpretable world models. Progress on this capability could in turn support downstream applications such as scientific discovery assistants, structured world models for embodied agents, and educational tools that probe and explain inferential reasoning. Because the benchmark is fully synthetic and procedurally generated, we see no immediate misuse vector beyond the general dual-use concerns that apply to any improvement in machine reasoning. We release the generator and evaluation code under the MIT License and the dataset under CC0 1.0 (links below the author list) to facilitate reproducibility and downstream research.

## References

[1] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.

[2] Francois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2024: Technical report, 2025. URL https://arxiv.org/abs/2412.04604.

[3] François Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2025: Technical report, 2026. URL https://arxiv.org/abs/2601.10904.

[4] Weili Nie, Zhiding Yu, Lei Mao, Ankit B. Patel, Yuke Zhu, and Animashree Anandkumar. Bongard-logo: A new benchmark for human-level concept learning and reasoning, 2021. URL https://arxiv.org/abs/2010.00763.

[5] Huaizu Jiang, Xiaojian Ma, Weili Nie, Zhiding Yu, Yuke Zhu, Song-Chun Zhu, and Anima Anandkumar. Bongard-hoi: Benchmarking few-shot visual reasoning for human-object interactions, 2023. URL https://arxiv.org/abs/2205.13803.

[6] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning, 2019. URL https://arxiv.org/abs/1903. 02741.

[7] Doojin Baek, Gyubin Lee, Junyeob Baek, Hosung Lee, and Sungjin Ahn. Learning to theorize the world from observation, 2026. URL https://arxiv.org/abs/2605.03413.

[8] Nick Chater and Mike Oaksford. Programs as causal models: Speculations on mental programs and mental representation. Cognitive science, 37(6):1171–1191, 2013.

[9] Stanislas Dehaene, Fosca Al Roumi, Yair Lakretz, Samuel Planton, and Mathias Sablé-Meyer. Symbols and mental programs: a hypothesis about human singularity. Trends in Cognitive Sciences, 26(9):751–766, 2022.

[10] Yassine Taoudi-Benchekroun, Klim Troyan, Pascal Sager, Stefan Gerber, Lukas Tuggener, and Benjamin Grewe. Cogitao: A visual reasoning framework to study compositionality & generalization, 2026. URL https://arxiv.org/abs/2509.05249.

[11] Brenden M. Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks, 2018. URL https://arxiv.org/ abs/1711.00350.

[12] Najoung Kim and Tal Linzen. Cogs: A compositional generalization challenge based on semantic interpretation. arXiv preprint arXiv:2010.05465, 2020.

[13] Laura Ruis, Jacob Andreas, Marco Baroni, Diane Bouchacourt, and Brenden M Lake. A benchmark for systematic generalization in grounded language understanding. Advances in neural information processing systems, 33:19861–19872, 2020.

[14] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2901–2910, 2017.

[15] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum. Clevrer: Collision events for video representation and reasoning, 2020. URL https://arxiv.org/abs/1910.01442.

[16] Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sable-Meyer, Luc Cary, Lucas Morales, Luke Hewitt, Armando Solar-Lezama, and Joshua B. Tenenbaum. Dreamcoder: Growing generalizable, interpretable knowledge with wake-sleep bayesian program learning, 2020. URL https://arxiv.org/abs/2006.08381.

[17] Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332–1338, 2015. doi: 10.1126/science.aab3050. URL https://www.science.org/doi/abs/10.1126/science. aab3050.

[18] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. arXiv preprint arXiv:1811.04551, 2018.

[19] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.

[20] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.

[21] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models, 2024. URL https://arxiv.org/abs/2301.04104.

[22] Dominik Schmidt and Minqi Jiang. Learning to act without actions. In International Conference on Learning Representations, 2024. Spotlight.

[23] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions, 2025. URL https://arxiv.org/abs/2503. 18938.

[24] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos, 2025. URL https://arxiv.org/abs/2410.11758.

[25] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments, 2024. URL https: //arxiv.org/abs/2402.15391.

[26] Matthew V Macfarlane and Clement Bonnet. Searching latent program spaces, 2025. URL https://arxiv.org/abs/2411.08706.

[27] Zizhao Wang, Chang Shi, Jiaheng Hu, Kevin Rohling, Roberto Martín-Martín, Amy Zhang, and Peter Stone. Factored latent action world models, 2026. URL https://arxiv.org/abs/ 2602.16229.

[28] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.

[29] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/ abs/2104.09864.

[30] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106.09685.

[31] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017.

[32] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

[33] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention, 2020.

[34] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[35] Junyeob Baek, Yi-Fu Wu, Gautam Singh, and Sungjin Ahn. Dreamweaver: Learning compositional world models from pixels. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=e5mTvjXG9u.

[36] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin T Chiu, and Volodymyr Kuleshov. The diffusion duality. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=9P9Y8FOSOk.

## A BabyTheorist Benchmark Details

## A.1 Objects and Scenes

Each BabyTheorist observation is generated on a 15 × 15 grid with color 0 as background and colors 1, . . . , 9 as object colors. Objects are connected colored shapes. Their silhouettes are sampled from compact random walks, and their color patterns include solid, bordered, hollow, striped, split, and checkerboard variants. During scene generation, one or more such objects are placed on the grid subject to non-overlap and validity constraints. An object’s position is the top-left corner of its bounding box, and its primary color is its most frequent color.

## A.2 Selectors

Selectors specify which object or object set a program step applies to. BabyTheorist uses eight selector forms: ALL, by\_color(c), is\_hollow, by\_index(k), topmost, bottommost, leftmost, and rightmost. Each selector picks its objects from the current scene at each step, not from the input scene. The multi-select selectors are ALL, by\_color(c), and is\_hollow. ALL selects every object in the scene; by\_color(c) selects objects whose primary color is c; and is\_hollow selects objects with enclosed interior background pixels.

The single-select selectors are by\_index(k), topmost, bottommost, leftmost, and rightmost. The by\_index(k) selector uses a 1-indexed raster ordering of objects in the input scene and remains tied to that object identity during execution. Positional selectors use deterministic tie-breaking: topmost chooses the smallest row, then smallest column; bottommost chooses the largest row, then largest column; leftmost chooses the smallest column, then smallest row; and rightmost chooses the largest column, then largest row.

## A.3 Primitives

Primitives specify the change applied to selected objects. BabyTheorist contains fourteen primitives, grouped as follows.

Transformation primitives. The footprint-preserving transformations are rot90, flip\_h, flip\_v, and next\_color. They rotate, reflect, or recolor a selected object while keeping its position fixed. rot90 rotates counter-clockwise, flip\_h flips upside down, flip\_v flips left to right, and next\_color cycles each color c to (c mod 9) + 1. The footprint-changing transformations are hollow, fill\_interior, and add\_border. They remove interiors, fill interiors, or add a one-pixel outer border, respectively.

Movement primitives. The movement primitives are translate\_up, translate\_down, translate\_left, and translate\_right. Each moves the selected object by its own height or width in the corresponding direction.

Relational primitives. The relational primitives are swap\_colors, align\_row, and align\_col. Each takes a selected source object and a target object. swap\_colors swaps primary colors between the two objects; align\_row moves the source object to the target object’s row; and align\_col moves the source object to the target object’s column.

The base single-object vocabulary contains eight non-relational primitives: rot90, flip\_h, flip\_v, next\_color, translate\_up, translate\_down, translate\_left, and translate\_right. Multi-object modes add the three relational primitives. B modes additionally introduce hollow, fill\_interior, and add\_border at Levels 2, 4, and 6, respectively.

A program is valid only if every step changes the scene and keeps all objects inside the grid without overlap; a step with a multi-object selector applies to its objects one at a time.

## A.4 Nominal Program-Space Counts

Table 1 reports a nominal depth-d program space for each benchmark mode after all primitives in that mode are available. These counts summarize the generator-side step-choice space before validity filtering, degeneracy filtering, and scene-specific selector failures.

In Single-A, there is one object and eight available primitives, so each step has 8 choices and a depth-d program has nominal space $8 ^ { d }$ . In Single-B, the three additional object-shape primitives raise the maximum primitive count to 11, giving $1 \widetilde { 1 ^ { d } }$

In Multi-A, there are two objects, the selector is by\_index, and the available vocabulary consists of eight non-relational primitives plus three relational primitives. A non-relational step has $2 \times 8 = 1 6$ source-object/primitive choices. A relational step has 2 ordered source-target choices for each of the three relational primitives, giving $3 \times 2 = 6$ choices. Thus each step has $1 6 + 6 = 2 2$ nominal choices, yielding 22<sup>d</sup>.

In Multi-B, the non-relational vocabulary has 11 primitives after all unlocks. The all-selector vocabulary gives 18 nominal selector instances over a 3-object scene: 3 by\_index choices, 9 by\_color(c) choices, four positional selectors, ALL, and is\_hollow. This gives $1 8 \times 1 1 = 1 9 8$ non-relational choices. For relational primitives, the table counts concrete ordered source-target object choices, giving 3 source objects times 2 distinct targets times 3 relational primitives, or 18 choices. Therefore the nominal step count is $1 9 8 + 1 8 = 2 1 { \bar { 6 } }$ , and the nominal depth-d space is $2 1 6 ^ { d }$ This is a compact benchmark-summary count; exact generation also samples selector parameters and applies symbolic and scene-level validity filters.

## A.5 Program Language Visual Examples

Figure 9 visualizes the selector types used to bind a program step to an object or object set. Figures 10– 13 show representative primitive executions, grouped by whether the primitive preserves the object footprint, changes the object shape, moves objects, or uses a relational target.

![](images/10acec9fa5bb5e306262749bcf4ea7da0cbf35dac99e0ae186698370a0633248.jpg)  
Figure 9: Selector types. Each row shows a source scene and the objects selected by different selector types. Global selection applies to all objects; attribute-based selectors choose objects by properties such as color or hollowness; spatial selectors choose objects by position; and index-based selectors use a fixed raster-scan ordering.

Transform-A (footprint-preserving)  
![](images/ab8845391fb387de3483a1f300f50ea2b6b3d9c45269d719e034152c719b8dcf.jpg)  
Figure 10: Footprint-preserving transformation primitives. Representative before-after examples for primitives that transform selected objects while preserving their footprint, including rotation, reflection, and color change.

Transform-B (footprint-changing)  
![](images/001f1baac55721aab93c25a0e8b66edc6ec820fc2e53e8ca7b97ef9c62345730.jpg)  
Figure 11: Footprint-changing transformation primitives. Representative before-after examples for objectshape primitives that change the interior or boundary of selected objects, including fill\_interior, hollow, and add\_border.

Movement  
![](images/d9f87913ec62b8c62cd5ef02bb6027ac58b2dda41f2f34592f9680f00330d07e.jpg)  
Figure 12: Movement primitives. Representative before-after examples for primitives that translate selected objects up, down, left, or right.

Relational  
![](images/9715a50868f9cccf5a1568adb01168303a888ae29e66186966bcdefdcdb6d86d.jpg)  
Figure 13: Relational primitives. Representative before-after examples for primitives that use a target object, including color swapping and row or column alignment.

## A.6 Frequent Subprogram Construction and Sampling

BabyTheorist maintains a frequent subprogram pool for each program depth. By sampling programs from these pools and combining their entries, the generator makes the same subprograms recur across programs of different depths. Below, we describe how primitive transition probabilities guide pool construction, how entries preserve their object relationships, and how the pools are used during program generation.

Primitive transition probabilities. To specify which primitives tend to occur together, the generator uses a transition matrix $T$ , where $T _ { i j }$ is the probability of choosing primitive $z _ { j }$ after primitive $z _ { i }$ in the primitive-sequence sampler. Each row sums to one. By default, the generator gives equal initial weight to all primitive pairs, sets known cancelling pairs to zero, and normalizes each row. For example, moving up followed immediately by moving down receives zero transition probability. Optional pair-specific weights can increase or decrease the relative probabilities of the remaining transitions.

These transition probabilities are used to score candidate primitive sequences. For a sequence $s = ( z _ { i _ { 1 } } , \ldots , z _ { i _ { d } } )$ , we compute

$$
w _ { \mathrm { s e q } } ( s ) = \left( \prod _ { k = 1 } ^ { d } w _ { i _ { k } } \right) \left( \prod _ { k = 1 } ^ { d - 1 } T _ { i _ { k } i _ { k + 1 } } \right) ,\tag{3}
$$

where $w _ { i }$ is a configurable weight for primitive $z _ { i } .$ , equal to one by default. Thus, the sequence weight accumulates the probabilities of its consecutive primitive pairs. For a concrete example under the default Single-A vocabulary, translate\_up has seven allowed successors, whereas rot90 has eight. The sequence translate $\mathsf { \Pi } _ { - } \mathtt { u p } _ { : }$ , rot90, next\_color therefore has weight $( 1 / 7 ) ( 1 / 8 ) = 1 / 5 6$ before object bindings and filtering are considered. The weights belong to the data generator and are hidden from the learner.

Representing the targets of operations. A primitive sequence alone does not specify whether its steps act on the same object. For example, move up, then move right can move one object twice or move two objects once each. Pool entries record this distinction using object placeholders: $A . \mathrm { u p }  A$ .right shares one target, whereas A.up → B.right uses separate targets. Placeholders can also denote object groups; relational primitives record a source and a target. Concrete selectors and scene objects are assigned during observation generation. When entries are combined, their internal target relationships are preserved, and placeholders in different entries are renamed so that they are not accidentally tied together.

Constructing the pools. The depth-1 pool contains one valid entry for each available primitive. For greater depths, the generator combines shorter primitive sequences and samples candidates according to Equation 3. It considers object-binding patterns that share one object, use two objects, share an object group, or use separate objects across steps. It also constructs candidates by combining entries from already validated, shallower pools. Candidates are checked using a temporary concrete assignment of their placeholders and the rule-based filter described in Appendix E.

To favor reusable target relationships, the generator applies a binding factor $b ( \tau ) = ( 1 + 0 . 2 5 r ) ( 1 +$ 0.10g), where r counts repeated references to placeholders beyond their first occurrence and g counts distinct group placeholders. A candidate obtained from a primitive sequence receives weight ${ w _ { \mathrm { s e q } } ( s ) \ r { b ( \tau ) } }$ . A candidate formed by combining two validated entries receives the product of their stored weights multiplied by $b ( \tau ) ;$ duplicate candidates retain the largest weight assigned to them. Valid candidates are sampled without replacement in proportion to these weights. If too few survive filtering, additional primitive sequences are sampled using $T$ and checked in the same way. The base pools retain up to twenty entries per depth above one, with direct pools built through depth ten.

Sampling programs from existing pools. The generator first samples a program depth d from the level-specific distribution in Appendix A.7. For $2 \leq d \leq 1 0$ , the basic pool sampler selects a complete entry from the depth-d pool with probability $0 . 5 ;$ otherwise, it combines shorter entries to reach depth $d .$ Direct selection is proportional to the entries’ stored weights. Depth-1 programs are sampled directly, while programs deeper than ten are always composed from pool entries.

For composition, the sampler repeatedly chooses a subprogram length uniformly from the allowable lengths and draws an entry from that pool. The first entry must be shorter than $d ,$ and subsequent entries cannot exceed the remaining depth or the maximum pool depth. When appending an entry, the sampler multiplies its stored weight by the transition probability between the last primitive already selected and the entry’s first primitive, then normalizes the resulting weights. For example, a depth-8 program can combine entries of depths three, two, and three. An entry that appears in a shorter program can consequently reappear within a longer program.

Adding new primitives in B modes. Single-B and Multi-B preserve the existing pools when a new primitive is introduced. The generator constructs additional candidates by placing the new primitive between a prefix and a suffix drawn from inherited pools; either may be empty. Each candidate contains the new primitive exactly once and must pass the same filter before being added. The new pools retain up to ten entries per depth above one, weighted by the inherited entries’ weights and the binding factor. Each new pool is sampled separately for two levels, then joins the inherited library.

The generator applies this two-level sampling window at Levels 2–3 for hollow, 4–5 for fill\_interior, and 6–7 for add\_border. During each window, it first decides whether to include the newly introduced primitive. For depth greater than one, it selects an entry from the newly added pools with probability 0.5 and, if necessary, places inherited subprograms before and after it to reach the requested depth. Otherwise, it uses only inherited pools, following the basic sampler above. The depth of the new entry is selected uniformly among available depths not exceeding d, and the remaining depth is split uniformly between the prefix and suffix. This 0.5 probability controls exposure to the new primitive; the basic sampler’s 0.5 probability controls direct selection versus composition within the inherited pools. Depth-1 sampling instead covers all currently available primitives.

Generating observations. After sampling an abstract program, the generator assigns concrete selectors and objects and applies the filter again, since concrete assignments can introduce additional interactions or redundancies. It then executes the program to generate before–after observations. Pool entries, transition probabilities, and generating programs remain hidden from the learner. All sampling probabilities above apply before filtering; the proportions among accepted observations can differ.

## A.7 Level Depth Distribution

BabyTheorist defines each level as a distribution over program depths rather than a fixed depth. The implementation uses a simple three-part discrete prior. For level $L ,$ let $c = L + 1$ be the center depth. The sampler allocates 80% probability mass to depths $1 , \ldots , c ,$ , 15% to the shoulder depths $c + 1 , c + 2$ , and 5% to a six-depth tail $c + 3 , \ldots , c + 8 \colon$

$$
\begin{array} { r l r } & { \begin{array} { r l } { p _ { L } ( d ) = 0 . 8 0 q _ { \mathrm { l e f t } } ( d ) + 0 . 1 5 q _ { \mathrm { s h o u l d e r } } ( d ) + 0 . 0 5 q _ { \mathrm { t a i l } } ( d ) , } \\ & { q _ { \mathrm { l e f t } } ( d ) \propto 0 . 6 0 ^ { c - d } } & { \mathrm { f o r } d \in \{ 1 , \dots , c \} , } \\ & { q _ { \mathrm { s h o u l d e r } } ( d ) \propto 0 . 6 0 ^ { d - ( c + 1 ) } } & { \mathrm { f o r } d \in \{ c + 1 , c + 2 \} , } \\ & { q _ { \mathrm { t a i l } } ( d ) \propto 0 . 3 5 ^ { d - ( c + 3 ) } } & { \mathrm { f o r } d \in \{ c + 3 , \dots , c + 8 \} . } \end{array} } \end{array}
$$

Each q is normalized over its own support and is zero outside that support. Thus the full support is $d \in \{ 1 , \ldots , L + 9 \}$ , and 95% of the mass lies at depths $d \leq c + 2$ . Figure 14 visualizes the resulting distributions across levels.

![](images/7815f85bfd5ef0a36e8b84b1ccec97977f0767d7ed6ec414f491c945ef82fd77.jpg)  
Figure 14: Level depth distributions. Each level defines a distribution over hidden program depths rather than a single depth. As the level increases, the distribution shifts toward longer programs while still mixing observations of different complexity within the same level.

## A.8 Training Observations per Level

For each level not cleared by the entry evaluation, the learner receives 250,000 training observations. This number was selected based on preliminary experiments in which smaller datasets yielded limited learning progress. We keep it fixed across levels and trainable methods to standardize data availability.

## B NEO: A Probabilistic Theory Induction Model

Generative model. NEO maximizes $p _ { \theta } ( y \mid x )$ by introducing two latent variables: a program $\tau = ( z _ { i _ { 1 } } , \dots , z _ { i _ { K } } )$ and its execution trace $s = ( s _ { 1 } , \dots , s _ { K + 1 } )$ . Under a Markov assumption,

$$
p _ { \theta } ( y \mid x ) = \int p _ { \theta } ( y \mid s _ { K + 1 } ) p _ { \theta } ( \tau , s \mid x ) d \tau d s ,\tag{4}
$$

with the joint prior factorized as

$$
p _ { \theta } ( \tau , s \mid s ) = p _ { \theta } ( s _ { 1 } \mid x ) \prod _ { k = 1 } ^ { K } p _ { \theta } ( z _ { i _ { k } } \mid s _ { k } ) p _ { \theta } ( s _ { k + 1 } \mid s _ { k } , z _ { i _ { k } } ) .\tag{5}
$$

Three components are shared across this generative process: an encoder $E _ { \theta }$ producing $s _ { 1 }$ from $x ,$ a theory programmer selecting primitives, and a shared executor $f _ { \theta }$ that realizes primitive semantics as latent state transitions.

Training. Since the marginalization in Eq. (4) is intractable, NEO introduces a variational posterior

$$
q _ { \phi } ( \tau , s \mid x , y ) = \underbrace { p _ { \theta } ( s _ { 1 } \mid x ) } _ { \mathrm { e n c o d e r } } \prod _ { k = 1 } ^ { K } \underbrace { q _ { \phi } ( z _ { i _ { k } } \mid s _ { k } , y ) } _ { \mathrm { t h e o r y r o g r a m m e r } } \underbrace { p _ { \theta } ( s _ { k + 1 } \mid s _ { k } , z _ { i _ { k } } ) } _ { \mathrm { e x e c u t i o n } } ,\tag{6}
$$

and optimizes the resulting ELBO. The theory programmer is a goal-conditioned policy: at step $k ,$ it observes the current latent state $s _ { k }$ and the encoded target $E _ { \theta } ( y )$ , and selects the next primitive that steers the trace toward a state explaining $y .$ . Discrete primitive selection is implemented through a VQ-VAE codebook $\mathcal { E } = \{ e _ { 1 } , \ldots , e _ { M ^ { \prime } } \}$ , where the codebook capacity $M ^ { \prime }$ is a hyperparameter and need not match the (unknown) true number of primitives $M$

A learned Language of Thought. Primitives in NEO have no predefined meaning. Their operational semantics emerge from the shared executor: each codebook entry corresponds to a learned state transition $s _ { k + 1 } = f _ { \theta } ( s _ { k } , z _ { i _ { k } } )$ . NEO therefore jointly learns the syntax of programs (how primitives compose) and their semantics (what each primitive does), forming a learned Language of Thought directly from observation, without symbolic supervision.

MDL and state grounding. Two design choices regulate learning. First, MDL-based length selection balances reconstruction error against program length,

$$
k ^ { \star } = \arg \operatorname* { m i n } _ { k \in \{ 1 , \ldots , K + 1 \} } \lambda _ { \mathrm { M D L } } ^ { k } \ell ( y , \hat { y } _ { k } ) , \quad \quad \lambda _ { \mathrm { M D L } } > 1 ,\tag{7}
$$

penalizing length exponentially and biasing the model toward reusable primitives rather than overly fine-grained ones. Second, a state-grounding loss anchors each intermediate state $s _ { k }$ to the manifold of valid encoder representations via a decode–encode consistency term, preventing shortcut transitions through arbitrary off-manifold states.

Test-time inference. Given $( x , y )$ , NEO iteratively selects primitives from $q _ { \phi } ( z \mid s _ { k } , y )$ and applies $f _ { \theta }$ up to a configured maximum number of steps. It then selects the shortest prefix that exactly reconstructs $y ,$ or the prefix minimizing length-weighted reconstruction loss if no exact match is found. The same learned primitives can be composed repeatedly at inference time, within this step budget.

## C Full Baseline Details

This appendix describes the architectures, training settings, and evaluation procedures of the baselines summarized in Section 5.1. All trainable baselines receive the same budget of 250,000 ungrouped observations per trained curriculum level. Evaluation uses support–query tasks and exact-match output-grid accuracy. Support usage is model-specific: LLMs place all support pairs in context, AR adapts LoRA parameters to the supports, and Cont-Mono-Opt optimizes a task-specific latent. Cont-Mono, Disc-Mono, and NEO use the Cross-Pair Validation procedure below with frozen model parameters. Cont-Mono-OC’s evaluation procedure is described separately in its baseline paragraph. Comparisons therefore reflect both the program representation and the associated inference procedure.

Cross-Pair Validation. For Cont-Mono, Disc-Mono, and NEO, we infer a candidate theory from each support pair using the trained inference model. Each candidate is applied to every support input, including the pair from which it was inferred, and its predictions are compared with the corresponding support outputs. We select the candidate most consistent with the full support set and apply it to the query input to predict the query output. All model parameters remain frozen throughout this procedure. The query output is used only to compute the final evaluation metric, not to infer or select a candidate. Unless otherwise specified, reported results for these three baselines use Cross-Pair Validation.

Large Language Model. We evaluate two open-weight LLM configurations: GPT OSS 120B with high reasoning effort and Qwen 3.5 27B with thinking disabled. This baseline tests whether generalpurpose LLMs can solve BabyTheorist tasks through direct few-shot prompting alone, and contrasts with models that must acquire reusable theories from ungrouped observations before evaluation. The full prompt is provided in Appendix C.1.

Autoregressive Transformer (AR). A standard decoder-only Transformer [28] with RoPE positional encodings [29], trained autoregressively under a causal mask on the ungrouped $( x , y )$ stream with no explicit program latent. Each $( x , y )$ pair is serialized into a single token sequence and the model is trained to predict $y$ tokens conditioned on x tokens, optimizing the standard next-token log-likelihood. At evaluation, the frozen base model is LoRA-adapted [30] on the supports for a fixed number of inner steps, with rank-8 adapters inserted into the attention query–key–value and output projections; we then generate $y _ { q }$ autoregressively from $x _ { q }$ under the adapted weights and discard the adapters before the next task. This baseline tests whether a generic sequence model with no inductive bias toward programs can, given a few supports, locate a parameter direction that recovers a transferable theory.

Continuous Monolithic (Cont-Mono). The theory inferred from a pair $( x , y )$ is represented by a single continuous vector $z \in \mathbb { R } ^ { d }$ , with no internal compositional structure. An encoder $q _ { \phi } ( z \mid x , y )$ maps the pair to a Gaussian posterior over z, and a decoder $p _ { \theta } ( y \mid x , z )$ predicts $y$ from x and a sampled z. The model is trained end-to-end as a conditional β-VAE [31], with the $\beta$ term controlling the trade-off between reconstruction fidelity and posterior compactness. At evaluation, the posterior mean $\hat { z } _ { s } = \mu _ { \phi } ( x _ { s } , y _ { s } )$ inferred from each support pair serves as a candidate continuous theory. We select among these candidates using Cross-Pair Validation (Appendix C) and condition the decoder on the selected latent to predict $y _ { q }$ from $x _ { q }$ . This baseline corresponds to continuous latent action models such as AdaWorld [23], and tests whether a single continuous theory vector suffices for transfer when no compositional bottleneck is imposed.

Discrete Monolithic (Disc-Mono). The theory inferred from a pair $( x , y )$ is represented by a single discrete vector $z \in \mathcal { E } = \{ e _ { 1 } , . . . , e _ { | \mathcal { E } | } \}$ drawn from a learned codebook. The encoder produces a continuous query that is quantized to its nearest codebook entry via straight-through estimation, and the decoder predicts $y$ from x and z as in Cont-Mono. The model is trained as a conditional VQ-VAE [32], with reconstruction, codebook, and commitment losses; the codebook size |E| is chosen per benchmark mode and held fixed. The theory is thus one symbol from a finite vocabulary, with no internal compositional structure. At evaluation, we infer a discrete theory candidate from each support pair and select the final candidate using Cross-Pair Validation (Appendix C) before applying it to the query input. This baseline corresponds to discrete latent action models such as LAPO [22] and Genie [25], and tests whether discretization to a finite, non-compositional vocabulary already buys transfer over a continuous monolithic latent.

Continuous Monolithic with Latent Optimization (Cont-Mono-Opt). Cont-Mono-Opt reuses the trained Cont-Mono model and optimizes a task-specific latent at evaluation time. We initialize the latent by averaging the vectors inferred from all support pairs, then optimize it to fit the full support set while keeping all model parameters frozen. The optimized latent is applied to the query input.

Object-Centric Continuous Monolithic (Cont-Mono-OC). Cont-Mono with the monolithic input encoder replaced by Slot Attention [33], which decomposes each observation into K slots, one of which is reserved for background, with a per-slot continuous latent $z _ { k } \in \mathbb { R } ^ { d }$ . A slot-level transition rolls each foreground slot forward independently conditioned on its own latent, and the predicted target is the aggregated reconstruction of the rolled-forward slots. The full model is trained end-to-end with the same conditional reconstruction objective as Cont-Mono, with no explicit slotcorrespondence supervision; slot identity emerges from the iterative competition inside Slot Attention. Because slot order varies across images, we learn to align slots before transferring object-level latents from supports to the query. We fit a lightweight slot-ranking network per task, keeping the base model frozen. To train this network, we use the object-level latents inferred from one support pair to predict the outputs of the other supports and minimize the prediction error. This baseline corresponds to object-centric continuous latent action models such as FLAM [27], and tests whether object-centric factorization alone, without compositional primitives over time, suffices in BabyTheorist’s Multi modes, where transformations apply to multiple objects simultaneously.

NEO. NEO [7], described in Appendix B, represents a program as an ordered sequence of discrete primitives drawn from a learned codebook, with MDL-based length selection that penalizes longer programs. A goal-conditioned theory programmer selects one primitive at a time from the current latent state and the encoded target; the shared executor applies it, and the final state is decoded to predict y. At evaluation, we infer a candidate primitive sequence from each support pair and select the final sequence using Cross-Pair Validation (Appendix C) before applying it to the query input. This baseline tests whether explicitly compositional discrete programs improve transfer over monolithic latents (continuous or discrete) in the absence of object-centric structure.

## C.1 LLM Prompt

The template below contains K support input–output pairs and one query input. Each grid is serialized as a nested list of 15 rows, ordered from top to bottom, with 15 integer color values per row, ordered from left to right: 0 denotes background, and 1–9 denote object colors. Each grid placeholder is replaced with the complete 15 × 15 array. The ellipsis abbreviates the support list for presentation only; all support pairs are included in the actual prompt.

LLM Prompt   
System message   
You are a puzzle solving wizard.   
User message   
Here are the example input and output pairs from which you should learn the underlying rule to later   
predict the output for the given test input:   
[{’input’: {support\_input\_1}, ’output’: {support\_output\_1}},   
{’input’: {support\_input\_K}, ’output’: {support\_output\_K}}]   
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the   
training data.:   
[{’input’: {input\_test\_data}, ’output’: [[]]}]   
What is the output grid? Only provide the output grid in the form as in the example input and output pairs.   
Do not provide reasoning or any additional information. Start with [[ and end with ]].:

## D Extended Related Works

We position BabyTheorist among existing benchmarks and ask, in turn, how each relates to observational theory learning.

Abstract reasoning and program-induction benchmarks. A long line of benchmarks frames reasoning as program induction from a few demonstration pairs. ARC-AGI [1–3] is the canonical example, where a learner must recover the program behind a small support set and apply it to a held-out query. COGITAO [10] extends this template into a generative grid-based visual reasoning framework that systematically samples compositions of primitive transformations, while Bongardstyle tasks [4, 5] and Raven’s Progressive Matrices [6] make the same demand visually. BabyTheorist shares the support–query evaluation format but differs in two design choices that target theory acquisition rather than puzzle-solving. First, training data are ungrouped: the learner sees a continual stream of input–output pairs with no task identities or support sets, and must discover that observations cluster around hidden programs. Second, beyond a controllable program language as in COGITAO, BabyTheorist also controls the distribution of recurring substructure and the complexity curriculum, enabling targeted study of how a theory system is built rather than how a fixed test set is solved.

Compositional generalization benchmarks. SCAN [11], COGS [12], gSCAN [13], and CLEVR/CLEVRER [14, 15] probe whether models trained on a finite set of primitive compositions can recombine them in novel ways. These benchmarks provide labeled task supervision and test recombination only at evaluation. BabyTheorist instead provides ungrouped observations and rewards models that recover the compositional structure itself, includingfrequent subprograms that recur across observations as natural pressure for reusable theories.

Library learning and concept accumulation. Library-learning systems such as DreamCoder [16] build up libraries of reusable abstractions over time by discovering recurring substructure in solved tasks; classical Bayesian program induction [17] similarly models concept acquisition as program learning over a generative grammar. These systems are typically evaluated on hand-curated domain suites where the abstraction set is implicit in the task design. BabyTheorist instead makes this abstraction set an explicit, controllable property of the data-generating process: frequent subprograms are embedded in the observation stream, the curriculum expands the program vocabulary at known levels, and level skipping directly measures whether previously acquired theories continue to explain new observations without further training.

World modeling and observation-only learning. Latent action models such as LAPO [22], AdaWorld [23], LAPA [24], and Genie [25] infer latent actions from visual observations without ground-truth action labels. World models such as PlaNet [18] and Dreamer [19–21] learn actionconditioned dynamics from interaction data. Their evaluations target downstream control or rollout fidelity. BabyTheorist evaluates theory transfer: whether explanations inferred from supports apply to held-out queries as the hidden programs grow deeper across the curriculum.

Table 2: Independent oracle validation of the depth filter on the Multi-B setting at level 4. We sampled 1,000 accepted tasks at each target depth and searched for shorter programs. Reliability is the fraction of accepted tasks for which the oracle found no shorter explanation.
<table><tr><td>Target depth</td><td>Samples</td><td>Validated</td><td>Shortcuts</td><td>Reliability</td></tr><tr><td>1</td><td>1,000</td><td>1,000</td><td>0</td><td>100.0%</td></tr><tr><td>2</td><td>1,000</td><td>998</td><td>2</td><td>99.8%</td></tr><tr><td>3</td><td>1,000</td><td>997</td><td>3</td><td>99.7%</td></tr><tr><td>4</td><td>1,000</td><td>987</td><td>13</td><td>98.7%</td></tr><tr><td>Total</td><td>4,000</td><td>3,982</td><td>18</td><td>99.55%</td></tr></table>

## E Filtering for Minimal Program Depth

Depth can be misleading. BabyTheorist uses program depth as a controlled proxy for task complexity. However, naively sampling a program with d steps is not sufficient by itself. The rendered pair of input and output grids should not be explainable by another valid program with fewer than d steps. Otherwise, an example sampled at depth d may in fact behave like an easier example at depth 1 or 2. The generator therefore applies a symbolic depth filter before accepting a task.

Program based screening. The first part of this symbolic filter runs while the program is being sampled. It keeps a symbolic description of each object rather than trying all possible scenes. This description records (i) where the object would be, (ii) how it has been rotated or flipped, (iii) how its color has changed, and (iv) whether its shape has been bordered, hollowed, or filled. After each new operation, the filter checks whether the same effect could have been obtained with fewer steps. For example, it rejects programs where different operations cancel each other, or where several objects are changed one by one even though the same objects could have been selected as a group. These checks catch shortcuts that are already visible from the program structure, but they do not cover every possible shortcut.

Scene dependent shortcuts. The remaining shortcuts arise only after the concrete objects and their positions are chosen. For example, a rotation may be unnecessary on a symmetric object. An align operation can also behave like an ordinary move on a particular scene, which can then merge with other moves and make the whole program shorter. The second part of the filter handles these cases by building constraints on the input scene while the program is sampled. These constraints rule out object symmetries that would hide rotations or flips. They also avoid ambiguous shape sizes when rotations, translations, and borders are mixed. For color and position based selectors, the constraints force the sampled scene to realize the object set that the symbolic sampler intended. A task is accepted only after it passes both the program screen and the scene constraints, and the resulting transition changes the grid.

Oracle validation. To measure how reliable the filter is, we validate it with an independent oracle that is not used by the generator. For each task accepted by the symbolic filter, the oracle runs breadth first search (BFS) from the sampled input scene and tries every shorter program. It merges duplicate rendered states during the search, so it does not waste work on different program strings that lead to the same intermediate scene. If any shorter program reaches the same output grid, the task is counted as a shortcut.

Table 2 shows that, on Multi-B tasks sampled at Level 4, the symbolic filter is highly reliable up to depth 4. Only 18 of 4,000 accepted samples had a shorter explanation found by the independent oracle. We therefore use the filter as a practical depth control mechanism, while treating depth as an empirically validated difficulty signal rather than a formal proof of minimal program length for every generated instance.

## F Model Hyperparameters

All trainable baselines were run on two NVIDIA RTX 4090 GPUs (24 GB each), and each run completed within 24 hours of wall-clock time. GPT-OSS 120B inference was run on eight NVIDIA RTX PRO 6000 GPUs, and GPT-5.6 Sol was queried through the OpenAI API.

This appendix lists the architecture and training hyperparameters for every model variant reported in the main paper. All shared backbone settings (encoder / decoder / policy / transition) are identical across models; we list them once per section and only flag the values that differ.

Common backbone (all variants):

• Encoder / Decoder: Transformer with RoPE, $d _ { \mathrm { m o d e l } } { = } 1 2 8 ,$ , 4 layers, 4 heads, $d _ { \mathrm { f f } } = 5 1 2 .$ dropout 0.

• Policy: MLP, $d _ { \mathrm { f f } } = 5 1 2$ , dropout 0.

• Transition: AdaLN Transformer, $d _ { \mathrm { m o d e l } } { = } 5 1 2$ , 3 layers, 4 heads, $d _ { \mathrm { f f } } { = } 2 0 4 8 ,$ dropout 0.

• State: 8 tokens of dim 512.

• For models with explicit latent representations, each latent vector has 16 dimensions. Cont-Mono, Cont-Mono-Opt, and Disc-Mono represent a program with a single vector, whereas NEO uses a sequence of vectors.

• Grid: 15×15, 10 colours.

• Optimiser: AdamW, weight decay 0.01, gradient clip 1.0 (except where noted), bf16-mixed precision.

Before any baseline-specific training, we pretrain the encoder / decoder on the Lv1 training data of the corresponding mode. All subsequent references to “pretrained encoder / decoder” refer to this checkpoint.

## F.1 AR (AutoRegressive)

Table 3: AR baseline hyperparameters.
<table><tr><td>Hyperparameter Backbone</td><td>Value</td></tr><tr><td>Hidden size Number of blocks Number of heads Sequence length Vocabulary size Prefix length Dropout Tie word embeddings Vocab lookup Loss on output tokens only LoRA rank / alpha LoRA target modules LoRA dropout Adaptation optimizer Adaptation steps</td><td>Causal Transformer with RoPE 384 12 8 451 (225 + 1 + 225) 12 226 (225 + 1 SEP) 0.1 false true true 8/16 attention QKV, attention output 0.0 AdamW, LR  $1 0 ^ { - 4 }$  , weight decay 0</td></tr></table>

## F.2 Cont-Mono

Continuous monolithic latent regularized as a $\beta \mathrm { - V A E , }$ with Gaussian sampling during training and the posterior mean at evaluation, and no vector quantization. The observation encoder is deterministic and uses its posterior mean during both training and evaluation. All Cont-Mono runs are initialized from the pretrained encoder / decoder and trained end-to-end. Auto-reconstruction loss is enabled.

Table 4: Cont-Mono training hyperparameters.
<table><tr><td></td><td>Single Lv1</td><td>Multi-A Lv1</td><td>Multi-B Lv1</td></tr><tr><td>Seeds</td><td>41, 42, 43</td><td>7,13,42</td><td>41, 42, 43</td></tr><tr><td>Max transition length</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Codebook size</td><td></td><td></td><td>0.9</td></tr><tr><td>Length-control coeff Latent VAE β</td><td>0.9</td><td>0.9</td><td></td></tr><tr><td>Recon loss weight</td><td>0.05</td><td>0.10</td><td>0.05</td></tr><tr><td>Grounding loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Auto-recon loss weight</td><td>5.0</td><td>1.0</td><td>5.0</td></tr><tr><td>Batch size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Devices (DDP)</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Learning rate</td><td>5×10-4</td><td>5×10-4</td><td>5×10-4</td></tr><tr><td>LR schedule</td><td>constant</td><td>constant</td><td>constant</td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Min LR ratio</td><td>0.005</td><td>0.005</td><td>0.005</td></tr><tr><td>Epochs</td><td>500</td><td>500</td><td>500</td></tr></table>

## F.3 Disc-Mono

Discrete monolithic latent with vector quantization (L2-normalised codebook, commitment cost 0.25, no EMA). All Disc-Mono runs are initialized from the pretrained encoder / decoder and trained end-to-end.

Table 5: Disc-Mono training hyperparameters.
<table><tr><td></td><td>Single Lv1</td><td>Multi-A Lv1</td><td>Multi-B Lv1</td></tr><tr><td>Seeds</td><td>41, 42, 43</td><td>42, 43, 44</td><td>42, 43, 44</td></tr><tr><td>Max transition length</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Codebook size</td><td>40</td><td>80</td><td>320</td></tr><tr><td>Recon loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Latent VQ loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Grounding loss weight</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Auto-recon loss weight</td><td>5.0</td><td>5.0</td><td>5.0</td></tr><tr><td>Batch size</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Devices</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Learning rate</td><td>5×10-4</td><td>1×10-3</td><td>1×10-3</td></tr><tr><td>LR schedule</td><td>constant</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>0.005</td><td>0.005</td></tr><tr><td>Min LR ratio</td><td>0.005</td><td>0.005</td><td>0.005</td></tr><tr><td>Epochs</td><td>500</td><td>500</td><td>500</td></tr></table>

## F.4 Cont-Mono-Opt (Test-time Latent Optimization)

Cont-Mono-Opt does not train any new model; it reuses the trained Cont-Mono checkpoints (Section F.2) and performs test-time gradient search on the inferred latent z with learning rate 0.01 for 10 epochs.

## F.5 Cont-Mono-OC (Slot-based)

Continuous object-centric action latents with a VAE regularizer and no vector quantization, with a separate action latent inferred for each slot. All Cont-Mono-OC runs are initialized from a pretrained object-centric autoencoder whose parameters are also updated during finetuning. Auto-reconstruction loss is enabled.

Table 6: Cont-Mono-OC finetuning hyperparameters. Batch size is per GPU.
<table><tr><td></td><td>Multi-A Lv1</td><td>Multi-B Lv1</td></tr><tr><td>Seeds</td><td>7, 13,2024</td><td>7, 13, 2024</td></tr><tr><td>Max transition length</td><td>1</td><td>1</td></tr><tr><td>Codebook size Length-control coeff</td><td>0.1</td><td>0.1</td></tr><tr><td>Latent VAE β</td><td>0.012</td><td>0.008</td></tr><tr><td>Recon loss weight</td><td>1.0</td><td>1.0</td></tr><tr><td>Grounding loss weight</td><td></td><td></td></tr><tr><td>Auto-recon loss weight</td><td>1.0</td><td>1.0</td></tr><tr><td>Number of slots</td><td>3</td><td>5</td></tr><tr><td>Slot dimension</td><td>256</td><td>256</td></tr><tr><td>Slot Attention iterations</td><td>5</td><td>5</td></tr><tr><td>Action dimension per slot</td><td>16</td><td>16</td></tr><tr><td>Batch size</td><td>128</td><td>128</td></tr><tr><td>Devices (DDP)</td><td>2</td><td>2</td></tr><tr><td>Learning rate</td><td> $7 \times 1 0 ^ { - 4 }$ </td><td> $7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>Min LR ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>Epochs</td><td>800</td><td>800</td></tr><tr><td></td><td></td><td></td></tr></table>

## F.6 NEO

NEO is trained on top of the pretrained encoder / decoder. All NEO runs use vector quantization for the learned codes, with an L2-normalised codebook, commitment cost 0.25, and auto-reconstruction loss.

Table 7: NEO single-object hyperparameters.
<table><tr><td rowspan="2"></td><td colspan="2">Lv1 (Single A &amp; B)</td><td colspan="2">Lv2</td></tr><tr><td>Phase 1 (mtl=2)</td><td>Phase 2 (mtl=4)</td><td>Single-A</td><td>Single-B</td></tr><tr><td>Seeds</td><td>40,41, 42</td><td>40, 41, 42</td><td>40,42</td><td>40,42</td></tr><tr><td>Max transition length</td><td>2</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Codebook size</td><td>30</td><td>30</td><td>30</td><td>30</td></tr><tr><td>Length-control coeff</td><td>0.9</td><td>1.0</td><td>1.2</td><td>1.2</td></tr><tr><td>Recon loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Latent VQ loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Grounding loss weight</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Auto-recon loss weight</td><td>5.0</td><td>5.0</td><td>5.0</td><td>5.0</td></tr><tr><td>Batch size</td><td>64</td><td>64</td><td>128</td><td>128</td></tr><tr><td>Devices (DDP)</td><td>2</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>LR schedule</td><td>constant</td><td>constant</td><td>warmup_ constant</td><td>constant</td></tr><tr><td>Warmup ratio</td><td>0.0</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Min LR ratio</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr><tr><td>Gradient clip</td><td>1.0</td><td>1.0</td><td>0.5</td><td>1.0</td></tr><tr><td>Epochs</td><td>500</td><td>200</td><td>50</td><td>50</td></tr><tr><td>Init</td><td>pretrained enc / dec</td><td>Phase 1 ckpt</td><td>Phase 1 ckpt</td><td>Phase 1 ckpt</td></tr></table>

Table 8: NEO multi-object hyperparameters.
<table><tr><td rowspan="2"></td><td colspan="2">Multi-A</td><td colspan="2">Multi-B</td></tr><tr><td>Phase 1 (mtl=2)</td><td>Phase 2 (mtl=3)</td><td>Phase 1 (mtl=2)</td><td>Phase 2 (mtl=3)</td></tr><tr><td>Seeds</td><td>42, 43, 44</td><td>42, 43, 44</td><td>42, 43, 44</td><td>42, 43, 44</td></tr><tr><td>Max transition length</td><td>2</td><td>3</td><td>2</td><td>3</td></tr><tr><td>Codebook size</td><td>40</td><td>40</td><td>160</td><td>160</td></tr><tr><td>Length-control coeff</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td></tr><tr><td>Recon loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Latent VQ loss weight</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Grounding loss weight</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Auto-recon loss weight</td><td>5.0</td><td>5.0</td><td>5.0</td><td>5.0</td></tr><tr><td>Batch size</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Devices</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="2">LR schedule</td><td>warmup_</td><td>warmup_</td><td>warmup_</td><td>warmup_</td></tr><tr><td>constant</td><td>constant</td><td>constant</td><td>constant</td></tr><tr><td>Warmup ratio</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr><tr><td>Min LR ratio</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr><tr><td>Gradient clip</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Epochs</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td rowspan="2">Init</td><td>pretrained enc /</td><td>Phase 1 ckpt</td><td>pretrained enc /</td><td>Phase 1 ckpt</td></tr><tr><td>dec</td><td></td><td>dec</td><td></td></tr></table>

## G Additional Results

## G.1 Full Score Results

Table 9 lists the SCORE values shown in Figure 6. Trained baselines report the mean and population standard deviation over three seeds. LLMs are evaluated once with greedy decoding on the first 100 tasks at each attempted level, so they have no standard deviation. Cont-Mono-OC is evaluated only in the Multi modes.

Table 9: SCORE on the four BabyTheorist benchmark modes. SCORE sums per-level exact-match accuracy across reached levels. Higher is better. Mean over $n = 3$ seeds (population std); LLM rows are single evaluations on the first 100 tasks per level.
<table><tr><td>Model</td><td>Single-A</td><td>Single-B</td><td>Multi-A</td><td>Multi-B</td></tr><tr><td>LLM (GPT-OSS 120B, high)</td><td>59</td><td>52</td><td>15</td><td>15</td></tr><tr><td>LLM (Qwen 3.5 27B, thinking-off)</td><td>8</td><td>8</td><td>4</td><td>6</td></tr><tr><td>AR</td><td> $6 . 7 \pm 0 . 6$ </td><td> $7 . 3 \pm 0 . 5$ </td><td> $7 . 7 \pm 0 . 6$ </td><td> $9 . 3 \pm 1 . 0$ </td></tr><tr><td>Cont-Mono</td><td> $1 5 . 9 \pm 1 . 4$ </td><td> $1 6 . 7 \pm 1 . 5$ </td><td> $1 0 . 5 \pm 5 . 7$ </td><td> $5 . 6 \pm 0 . 7$ </td></tr><tr><td>Disc-Mono</td><td> $5 2 . 5 \pm 7 . 0$ </td><td> $5 4 . 5 \pm 6 . 9$ </td><td> $1 6 . 1 \pm 2 . 9$ </td><td> $8 . 5 \pm 0 . 6$ </td></tr><tr><td>Cont-Mono-Opt</td><td> $1 6 . 8 \pm 1 . 4$ </td><td> $1 7 . 9 \pm 1 . 6$ </td><td> $8 . 7 \pm 4 . 9$ </td><td> $5 . 6 \pm 0 . 7$ </td></tr><tr><td>Cont-Mono-OC</td><td></td><td></td><td> $3 . 2 \pm 1 . 4$ </td><td> $1 0 . 7 \pm 5 . 6$ </td></tr><tr><td>NEO</td><td> $\mathbf { 1 3 6 . 8 \pm 4 9 . 6 }$ </td><td> ${ \bf 1 3 7 . 7 \pm 4 9 . 8 }$ </td><td> ${ \bf 3 6 . 7 \pm 4 . 3 }$ </td><td> ${ \bf 2 2 . 6 \pm 1 . 6 }$ </td></tr></table>

## G.2 PPO with Known DSL

To assess how difficult program inference and composition remain when the operations are already available, we evaluate PPO with Known DSL. This learner receives the ground-truth primitive and selector vocabulary available at each level, together with the exact executor. It must still infer a program that explains the observations, but it does not need to discover the operations or learn their execution semantics. We therefore report it separately from the learning-focused baselines in Section 5.1 as a diagnostic with privileged DSL information.

## G.2.1 Learning to Select Program Steps

Training uses independent before–after pairs $( x , y )$ , with no ground-truth programs, program depths, or task groupings. For each pair, an episode starts from x and builds a program one step at a time. The policy observes the current grid and the target $y ,$ selects a DSL action, and receives the resulting grid from the exact executor. A Single-mode action selects a primitive for the sole object. In Multi modes, the policy selects a source selector, a primitive, and, for relational operations, a target selector. These decisions specify one executable step. For execution, objects are recovered from eight-connected components of non-background pixels in the input grid; their identities are retained across subsequent steps.

The policy is Markov: each decision depends on the current grid, target grid, and available DSL vocabulary, without a recurrent state or previous-action input. A three-layer convolutional encoder processes the current and target grids together with grid-validity and coordinate channels. Its features condition the action distribution and value estimate. The Single-mode policy scores complete actions directly; the Multi-mode policy factorizes an action into the conditional source, primitive, and target choices described above. Proximal policy optimization (PPO) [34] trains the policy and value function from the resulting trajectories.

To provide feedback before an entire transformation is solved, the reward combines exact reconstruction with progress toward the target. Let $x _ { t }$ be the current grid, with $x _ { 0 } = x ,$ , and let $D ( u , v )$ count mismatched pixels. We define

$$
\begin{array} { r } { \Phi ( z ) = \mathrm { c l i p } \left( \frac { D ( x , y ) - D ( z , y ) } { D ( x , y ) } , - 1 , 1 \right) , } \\ { r _ { t } = \mathbf { 1 } \{ x _ { t + 1 } = y \} + \Phi ( x _ { t + 1 } ) - \Phi ( x _ { t } ) . } \end{array}\tag{8}
$$

The potential $\Phi$ measures progress relative to the initial mismatch, while the indicator gives a bonus for exact success. Training pairs have $x \neq y .$ An action that cannot be executed leaves the state unchanged, receives zero reward, and consumes one step. An episode ends on exact success or after 17 steps. The learner receives neither a state-dependent mask identifying executable actions nor auxiliary search-generated action labels.

Each level requiring training supplies 250,000 observations. We collect one trajectory per observation in a single pass through that level’s data and perform four PPO optimization epochs per rollout batch. The policy carries over to subsequent levels. Table 10 gives the configuration used for all four modes and seeds 0, 1, and 2.

Table 10: Training and inference settings for PPO with Known DSL. The policy head differs between Single and Multi modes as described in the text.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>CNN channels / hidden width Action-component embedding / action feature width</td><td>(64, 128, 128) / 512 64 / 256</td></tr><tr><td></td><td>Adam / 1.5 × 10−4</td></tr><tr><td>Optimizer / learning rate</td><td>10−⁵ / 0.5</td></tr><tr><td>Adam € / gradient-norm clipping Rollout batch / episode minibatch size</td><td></td></tr><tr><td>PPO epochs / clipping coefficient</td><td>512 /128</td></tr><tr><td></td><td>4/0.2</td></tr><tr><td>Entropy / value-loss coefficient</td><td>0.1 / 0.5</td></tr><tr><td>Discount γ / GAE λ</td><td>1.0 / 0.95</td></tr><tr><td>Advantage normalization</td><td>Within each minibatch</td></tr><tr><td>Pixel-progress coefficient</td><td>1.0</td></tr><tr><td>Beam width / branch width / maximum steps</td><td>8/16 /17</td></tr></table>

## G.2.2 Inferring a Program from the Support Set

At evaluation, the learned policy guides a beam search for one program that reproduces every support output. Each partial program is executed on all support inputs. To rank its possible next actions, we average the policy’s log probabilities over supports whose current grids do not yet match their targets; these scores accumulate along the program. We consider actions in score order, retain up to 16 continuations per partial program that execute successfully on every support, and keep the eight highest-scoring partial programs for the next step. Search is limited to 17 steps.

A completed candidate must reproduce all support outputs exactly. Among candidates that also execute successfully on the query input, the planner selects the highest-scoring program, preferring a shorter program when scores tie. Executing that program produces the query prediction. If search finds no such program, the prediction defaults to the unchanged query input. The query output is never used during program search or selection, and the policy parameters remain fixed throughout evaluation.

## G.2.3 Results and Interpretation

We evaluate on BabyTheorist v0.5.5, using all 10,000 evaluation tasks at each attempted level. Entry evaluations permit level skipping; a failed entry evaluation triggers training on that level’s 250,000 observations. A level is passed at 80% exact-match accuracy; the curriculum stops at the first post-training failure. Table 11 reports mean SCORE and population standard deviation across three seeds.<sup>1</sup>

Training settings were selected using public Multi-B Level-1 evaluations, and branch width 16 was selected after comparing widths 8 and 16 on the public evaluation set. These results therefore describe a diagnostic configuration selected with evaluation feedback. They should not be interpreted as performance on an untouched test set or as a matched comparison with learners that must acquire the DSL from observations.

With the operations and executor provided, this planner completes Single-A and progresses through five levels of Single-B and three of Multi-A. It still fails to pass Level 1 of Multi-B in every seed. The remaining difficulty involves proposing and combining actions under a finite search budget, even when their execution semantics are known. These results characterize the combined learned policy and search procedure: before PPO training, the same search procedure already achieves 84.8–85.5% accuracy at Level 1 of both Single modes. They do not establish an upper bound on performance with a known DSL.

Table 11: PPO with Known DSL on the four BabyTheorist modes. SCORE is reported as mean ± population standard deviation over three seeds. Level progression is identical across seeds; zero passed levels means failure at Level 1. Settings were selected using the public evaluation set, as described below.
<table><tr><td>Mode</td><td>SCORE</td><td>Highest level passed</td><td>First failed level</td></tr><tr><td>Single-A</td><td> $6 8 0 . 2 \pm 2 . 5$ </td><td>8</td><td>一</td></tr><tr><td>Single-B</td><td> $5 0 0 . 4 \pm 0 . 8$ </td><td>5</td><td>6</td></tr><tr><td>Multi-A</td><td> $3 3 1 . 8 \pm { 1 . 5 }$ </td><td>3</td><td>4</td></tr><tr><td>Multi-B</td><td> $5 2 . 5 \pm 5 . 9$ </td><td>0</td><td>1</td></tr></table>

## G.3 GPT-5.6 Sol: Performance and Error Analysis

We additionally evaluate GPT-5.6 Sol to examine how well a pretrained reasoning model can infer transformations from support examples. The model serves as a general-purpose reference and receives no training on BabyTheorist’s observation stream. Using high reasoning effort, we evaluate the first 100 tasks from each of the eight levels in all four modes of BabyTheorist v0.5.5, totaling 3,200 tasks. Each task is evaluated independently without tools, using the support–query prompt in Appendix C.1. Predictions are scored by exact match with the target grid.

GPT-5.6 Sol achieves 51–64% accuracy at Level 1, below the 80% passing threshold in every mode. This result cannot be explained by output-format errors alone: even counting every malformed Level-1 prediction as correct would leave accuracy at or below 69% in all four modes. To examine performance on more complex tasks, we also evaluate Levels 2–8 independently; these evaluations do not count as curriculum progression. As shown in Table 12, accuracy generally decreases at later levels, reaching 30–40% at Level 8.

Table 12: GPT-5.6 Sol exact-match accuracy (%) on 100 tasks per mode and level. Levels 2–8 are evaluated independently for diagnostic purposes.
<table><tr><td>Mode</td><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Level 4</td><td>Level 5</td><td>Level 6</td><td>Level 7</td><td>Level 8</td></tr><tr><td>Single-A</td><td>58</td><td>68</td><td>58</td><td>55</td><td>47</td><td>44</td><td>37</td><td>31</td></tr><tr><td>Single-B</td><td>64</td><td>69</td><td>60</td><td>57</td><td>51</td><td>39</td><td>45</td><td>40</td></tr><tr><td>Multi-A</td><td>51</td><td>48</td><td>42</td><td>50</td><td>42</td><td>42</td><td>31</td><td>30</td></tr><tr><td>Multi-B</td><td>52</td><td>48</td><td>51</td><td>53</td><td>40</td><td>29</td><td>35</td><td>33</td></tr></table>

To characterize the errors already present at Level 1, we compare its 400 predictions with the target grids and intermediate outputs of the generating programs. These programs are used only for post-hoc analysis and are not provided to the model. Of the 175 incorrect predictions, 21 have malformed grids, leaving 154 parseable but incorrect outputs.

More steps, more mistakes. Level 1 includes both single-step and multi-step programs. Accuracy decreases from 67.8% on single-step programs (103/152) to 49.2% on longer programs (122/248). This gap appears in all four modes and remains after excluding malformed outputs. Some predictions also reproduce only part of the required transformation. For example, one Multi-A task requires a left–right reflection of one object followed by two 90<sup>◦</sup> rotations of another. The prediction exactly matches the grid after the first step, leaving the second object’s required rotation absent from the output.

The right objects in the wrong places. In 89 of the 154 parseable errors (57.8%), the prediction can be reproduced exactly by changing only the positions of the target objects, preserving their shapes and internal colors without adding or omitting foreground pixels. Forty of these cases also reproduce required appearance changes, so they cannot be explained by simply relocating unchanged input objects. The remaining error lies in where the objects are placed. For example, a two-cell-high objec that should move upward by its own height is instead moved upward by three cells. Its appearance and movement direction are correct, but the displacement is not.

Relations pose a greater challenge. In Multi-B, accuracy is 31.8% on tasks containing relational operations such as alignment or color swapping (14/44), compared with 67.9% on the remaining tasks (38/56). Lower accuracy on relational tasks is also observed within the depth-1 and depth-2 subsets. Multi-A shows a different pattern, with corresponding accuracies of 61.5% and 44.3%. Thus, this analysis identifies relational tasks as a difficult subset of the sampled Multi-B setting, rather than establishing a general disadvantage for relational operations.

These comparisons reveal failures to reproduce complete transformations and exact spatial placement. They characterize the predicted outputs, but do not determine whether the model inferred an incorrect rule or failed to execute an intended rule accurately.

## H Failure Cases

This appendix gives the protocols, statistics, and examples supporting the three challenges in Sections 5.3–5.5. We first define the diagnostic scope, then present primitive transfer and latent refinement, target selection, and compositional execution. Each labeled panel group shows one task. S1/S2 denote supports and Q the query; green/red borders indicate correct/incorrect outputs. All grids retain their original coordinates.

## H.1 Evaluation scope and support–query statistics

We analyze saved Level-1 outputs from three seeds per method. Each set contains 10,000 tasks for shared Single, Multi-A, and Multi-B; shared Single-A/B tasks are counted once. The qualitative AR examples in Figure 15 are drawn from a fixed random subset of 256 tasks per mode and seed. OC and LLM outputs are excluded from these diagnostics.

All rates use full-grid exactness and report the mean ± population SD over three seeds (ddof=0). Conditional rates are computed per seed before aggregation. Every-support exactness (S) is a diagnostic, not an official benchmark metric. Cont-Mono, Disc-Mono, and NEO select among per-support candidates by exact support coverage, breaking ties by shorter length and then earlier support. Opt starts from the average of the latents inferred from each support pair and optimizes it to fit all supports; AR adapts on all supports. Thus conditional cohorts differ by model. These are checkpoint diagnostics, not curriculum SCORE reproductions or controlled architectural ablations.

Full-set Opt rates use archived final outputs. Depth-one and paired optimization rates use separately recorded step-10 trajectories; paired changes compare each trajectory to its own step 0, not to the cross-pair-selected Cont-Mono output.

Table 13: Support fit and query transfer at Lv1. Mean ± SD across three seeds (%). S: every support correct; Q: query correct; Q | S: query accuracy conditional on S. Each dataset contains 10,000 tasks per seed; Single-A/B share the same Lv1 data. See Appendix H.1 for protocols.
<table><tr><td>Dataset</td><td>Method</td><td>S (%)</td><td>Q (%)</td><td>Q | S (%)</td></tr><tr><td rowspan="5">Single-A/B</td><td>Cont-Mono</td><td> $2 1 . 9 3 \pm 1 . 8 1$ </td><td> $1 6 . 6 6 \pm 1 . 2 4$ </td><td> $2 5 . 9 8 \pm 2 . 3 2$ </td></tr><tr><td>Cont-Mono-Opt</td><td> $2 3 . 8 4 \pm 1 . 5 4$ </td><td> $1 6 . 9 7 \pm 1 . 2 3$ </td><td> $2 5 . 5 1 \pm 1 . 9 0$ </td></tr><tr><td>Disc-Mono</td><td> $5 0 . 1 9 \pm 5 . 3 3$ </td><td> $5 4 . 4 8 \pm 5 . 6 6$ </td><td> $8 4 . 8 6 \pm 5 . 5 0$ </td></tr><tr><td>NEO</td><td> $8 7 . 0 1 \pm 4 . 0 8$ </td><td> $8 5 . 8 1 \pm 4 . 1 7$ </td><td> $9 2 . 1 1 \pm 2 . 7 7$ </td></tr><tr><td>AR</td><td> $9 9 . 9 5 \pm 0 . 0 3$ </td><td> $7 . 3 2 \pm 0 . 4 5$ </td><td> $7 . 3 1 \pm 0 . 4 4$ </td></tr><tr><td rowspan="5">Multi-A</td><td>Cont-Mono</td><td> $1 0 . 6 4 \pm 3 . 9 8$ </td><td> $1 0 . 4 5 \pm 4 . 6 1$ </td><td> $2 2 . 7 4 \pm 9 . 3 1$ </td></tr><tr><td>Cont-Mono-Opt</td><td> $1 1 . 2 4 \pm 3 . 8 2$ </td><td> $1 0 . 0 6 \pm 4 . 3 7$ </td><td> $2 2 . 0 7 \pm 9 . 2 7$ </td></tr><tr><td>Disc-Mono</td><td> $1 0 . 8 7 \pm 2 . 5 5$ </td><td> $1 6 . 1 0 \pm 2 . 3 7$ </td><td> $5 1 . 8 4 \pm 0 . 9 7$ </td></tr><tr><td>NEO</td><td> $3 9 . 0 7 \pm 2 . 0 0$ </td><td> $3 6 . 6 6 \pm 3 . 5 4$ </td><td> $5 2 . 6 8 \pm 3 . 3 9$ </td></tr><tr><td>AR</td><td> $9 9 . 9 9 \pm 0 . 0 1$ </td><td> $7 . 7 8 \pm 0 . 4 5$ </td><td> $7 . 7 8 \pm 0 . 4 5$ </td></tr><tr><td rowspan="5">Multi-B</td><td>Cont-Mono</td><td> $4 . 3 5 \pm 0 . 4 7$ </td><td> $5 . 5 7 \pm 0 . 6 1$ </td><td> $2 7 . 4 3 \pm 1 . 2 2$ </td></tr><tr><td>Cont-Mono-Opt</td><td> $4 . 9 5 \pm 0 . 3 3$ </td><td> $5 . 4 8 \pm 0 . 6 0$ </td><td> $2 5 . 6 7 \pm 2 . 6 0$ </td></tr><tr><td>Disc-Mono</td><td> $4 . 9 8 \pm 0 . 0 6$ </td><td> $8 . 4 7 \pm 0 . 4 6$ </td><td> $4 6 . 4 0 \pm 2 . 4 8$ </td></tr><tr><td>NEO</td><td> $2 1 . 1 6 \pm 1 . 0 3$ </td><td> $2 2 . 5 5 \pm 1 . 3 4$ </td><td> $5 0 . 8 9 \pm 0 . 6 2$ </td></tr><tr><td>AR</td><td> $9 9 . 9 2 \pm 0 . 0 4$ </td><td> $9 . 2 9 \pm 0 . 8 7$ </td><td> $9 . 3 0 \pm 0 . 8 7$ </td></tr></table>

Table 14: Single-A/B transfer at Lv1, primitive depth one. Mean ± SD across three seeds (%). S: every support correct; Q: query correct. $S \cap \lnot Q$ uses all N tasks as its denominator. Inference follows Appendix H.1.
<table><tr><td>Method</td><td>N</td><td>S (%)</td></tr><tr><td>Cont-Mono</td><td>2944  $3 4 . 6 9 \pm 2 . 7 3$ </td></tr><tr><td> $2 6 . 1 8 \pm 2 . 0 1$   $2 3 . 5 1 \pm 1 . 0 4$   $2 4 . 6 3 \pm 0 . 7 9$ </td><td></td></tr><tr><td>Cont-Mono-Opt 2944</td><td> $3 6 . 1 6 \pm 2 . 2 0$   $2 6 . 5 1 \pm 1 . 9 7$ </td></tr><tr><td> $7 5 . 5 1 \pm 8 . 0 9$   $8 . 4 8 \pm 3 . 3 6$ </td><td>Disc-Mono 2944  $7 3 . 9 7 \pm 7 . 3 6$  NEO</td></tr><tr><td>2944  $9 7 . 5 4 \pm 2 . 2 9$  2944</td><td> $9 6 . 4 9 \pm 2 . 6 9$   $2 . 3 6 \pm 1 . 4 1$   $9 9 . 9 5 \pm 0 . 0 2$   $1 3 . 0 7 \pm 0 . 7 8$   $8 6 . 9 2 \pm 0 . 7 7$ </td></tr></table>

Table 15: Single-A/B transfer at Lv1, primitive depth two. Mean ± SD across three seeds (%). N: tasks per seed; S: every support correct; Q: query correct. S ∩ ¬Q uses all N tasks as its denominator. Inference follows Table 14, including step-10 trajectories for Cont-Mono-Opt. Depth is the number of primitives in the generating program.
<table><tr><td>Method</td><td>N</td><td>S (%)</td><td>Q (%)  $S \cap \lnot Q \left( \% \right)$ </td></tr><tr><td>Cont-Mono</td><td>5017</td><td> $2 0 . 0 1 \pm 1 . 8 3$ </td><td> $1 5 . 1 9 \pm 1 . 1 5$   $1 5 . 5 1 \pm 0 . 9 3$ </td></tr><tr><td>Cont-Mono-Opt</td><td>5017</td><td> $2 2 . 2 7 \pm 1 . 3 0$ </td><td> $1 5 . 5 9 \pm 1 . 1 1$   $1 7 . 3 4 \pm 0 . 5 6$ </td></tr><tr><td>Disc-Mono</td><td>5017</td><td> $5 6 . 1 9 \pm 6 . 7 0$ </td><td> $6 1 . 6 4 \pm 6 . 8 0$   $9 . 2 7 \pm 2 . 7 4$ </td></tr><tr><td>NEO</td><td>5017</td><td> $8 9 . 4 4 \pm 4 . 7 5$ </td><td> $8 8 . 2 2 \pm 4 . 7 8$   $6 . 5 4 \pm 2 . 2 4$ </td></tr><tr><td>AR</td><td>5017</td><td> $9 9 . 9 5 \pm 0 . 0 3$ </td><td> $5 . 9 7 \pm 0 . 3 3$   $9 3 . 9 8 \pm 0 . 3 6$ </td></tr></table>

Output-pattern counts. A one-constituent error exactly matches either constituent applied alone on a depth-two task; this does not identify an internal omitted step. Target errors require a one-step unary task with a unique target and an exact alternative-target match, excluding no-op candidates. Their frequency also depends on primitive execution accuracy.

Table 16: Output-pattern frequencies. Mean ± SD across three seeds (%). Eligible n: Single 5,017; Multi-A 2,177; Multi-B 2,117; full n: 10,000. Patterns are non-exclusive. Opt uses 2,944 depth-one tasks; full-set rescue/harm: $1 . 1 2 \pm 0 . 0 6 / 0 . 9 9 \pm 0 . 0 5 \%$ . Within-trajectory net gain: 0.43 ± 0.11 pp.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Denominator</td><td rowspan="2">Single One constituent Wrong target Wrong target</td><td rowspan="2">Multi-A</td><td rowspan="2">Multi-B</td></tr><tr><td></td></tr><tr><td>Cont-Mono</td><td>Eligible</td><td> $1 5 . 0 5 \pm 1 . 1 5$ </td><td> $8 . 8 2 \pm 4 . 8 9$ </td><td> $4 . 9 6 \pm 0 . 7 3$ </td></tr><tr><td rowspan="2">Cont-Mono-Opt</td><td>Full set</td><td> $7 . 5 5 \pm 0 . 5 8$ </td><td> $1 . 9 2 \pm 1 . 0 7$ </td><td> $1 . 0 5 \pm 0 . 1 6$ </td></tr><tr><td>Eligible</td><td> $1 5 . 1 6 \pm 0 . 9 4$ </td><td> $8 . 9 1 \pm 5 . 2 2$ </td><td> $5 . 2 3 \pm 0 . 8 5$ </td></tr><tr><td>Disc-Mono</td><td>Full set</td><td> $7 . 6 0 \pm 0 . 4 7$   $3 . 7 0 \pm 0 . 9 4$ </td><td> $1 . 9 4 \pm 1 . 1 4$   $1 4 . 9 1 \pm 4 . 2 9$ </td><td> $1 . 1 1 \pm 0 . 1 8$ </td></tr><tr><td rowspan="2"></td><td>Eligible</td><td></td><td></td><td> $5 . 7 8 \pm 0 . 2 2$ </td></tr><tr><td>Full set</td><td> $1 . 8 6 \pm 0 . 4 7$ </td><td> $3 . 2 5 \pm 0 . 9 3$ </td><td> $1 . 2 2 \pm 0 . 0 5$ </td></tr><tr><td rowspan="2">NEO</td><td>Eligible</td><td> $0 . 5 5 \pm 0 . 5 2$ </td><td> $2 5 . 4 3 \pm 2 . 0 4$ </td><td> $1 5 . 1 9 \pm 0 . 7 4$ </td></tr><tr><td>Full set</td><td> $0 . 2 8 \pm 0 . 2 6$ </td><td> $5 . 5 4 \pm 0 . 4 4$ </td><td> $3 . 2 2 \pm 0 . 1 6$ </td></tr><tr><td rowspan="2">Opt: rescue Opt: harm</td><td>Depth one</td><td> $3 . 8 0 \pm 0 . 2 1$ </td><td></td><td></td></tr><tr><td>Depth one</td><td> $3 . 3 7 \pm 0 . 1 8$ </td><td></td><td></td></tr></table>

Input

Pred.

## H.2 Primitive Transfer and Latent Refinement

The following examples examine whether fitting observed pairs yields transformations that remain correct on query inputs, and how latent refinement changes that behavior.

Target  
(a) AR  
![](images/20885923c2be5e57ac13f8eab1aaf4d00f6b9f773140e7122ea5df38fd369d62.jpg)

(b) AR  
![](images/0977e2ade448cabf47fc0b733d3decc3c5eff5a09fb471066ae635e3b370b8dc.jpg)

(c) AR  
![](images/43fba86eeb103908d75e8bd4732b8b4dc2bf7b37f126b9c66262237bf48b93a1.jpg)  
Figure 15: AR: support fit without query transfer. Each column shows a different depth-one Single-B task from this 256-task subset (seed 42). The adapted model fits both supports but fails on the query.

(a) Cont-Mono  
![](images/05eb55d1fdd13328651ebab8c8fb25d41c4dd79966f6d29966c8fdcacfcd7911.jpg)

(b) Cont-Mono  
![](images/3645f6c42242868f1c96183b7334ca0e728e2de2f35233a84fa6e827e2167c3a.jpg)

(c) Cont-Mono  
![](images/59da444bcfd98471852e69484bee164e6f17bc096cab6bca6c38164375c3815c.jpg)  
(f) Disc-Mono

![](images/25d975f4f63b9107ba0810bc275af7de0c48e999ba05451f8e87b59c69a0e2f4.jpg)

Input  
![](images/2a35b39443c4f0f3d7deb5a02d25d816a06c72c2e88968e64858c61fb44943b4.jpg)

Input  
![](images/b937a83ff91f8e813813d6b451b0118edfaef284cf21ce3f0e53eba400360410.jpg)  
Figure 16: Transfer errors on depth-1 programs. Cont-Mono (a–c) and Disc-Mono (d–f) fit both supports but make reflection-axis, color-change, and translation-distance errors, respectively (seed 42). Color predictions match two shifts instead of one. The Disc-Mono translation error is a rare illustrative case.

![](images/3936358b3cd213b334978bae0fc28ab7ee5107fc17021168b8f2371d1dee963c.jpg)  
Figure 17: Support fitting can damage a correct query. Three Cont-Mono-Opt cases change from correct at step 0 to incorrect at step 10. Both supports remain exact at recorded steps 0, 1, 5, and 10; support loss need not be constant.

## H.3 Target Selection

The predictions below match the intended primitive applied to a different object. These output matches identify target discrepancies but do not distinguish selection-rule inference from execution errors.

(a) Cont-Mono  
![](images/2d11309556f42df4c91cdfa863e9e150b5fa58d0ade8690970dc3f91ca359dbe.jpg)

(b) Disc-Mono  
![](images/2ae1190429c1eb4f66d7ccbc12380a42dffccd3fafab76e07d7c4e84ad039ac8.jpg)

(c) NEO  
![](images/576f449c94ab7eeae93329ef4ca844aa2e82f1153aa4a0230095649df05ee839.jpg)  
Figure 18: Correct transformation, wrong target. Cont-Mono, Disc-Mono, and NEO fit both supports but change the wrong query object in Multi-A (seed 42). Each prediction exactly matches the correct primitive applied to another target. Dashed outlines identify the affected objects.

![](images/6b2365a342822c0e48d4db165137cccaa75909b8e91b8f0bc40c3684403dfdb4.jpg)

![](images/72b015047e6632e3a1aedd567d90c1e7973087179ca34cab7167a4faeda18cbd.jpg)

## H.4 State-dependent execution

To test whether a learned code behaves consistently on intermediate execution states, we hold NEO weights and inferred codes fixed. Rotation and fresh reconstruction are exact on supports and query. “Continue” uses the existing hidden state; “Re-encode” re-encodes the decoded image. These selected cases do not estimate prevalence.

(a) NEO  
![](images/fd7857a1c6f5dedabd8165bfb30a722ffb28991665813fc96d2629c046d600ba.jpg)

(b) NEO  
![](images/3f7d97ce2c1e81a39db0be4a6166ecd54218d7c83ae2c3b59f6149962dbab5a0.jpg)  
(c) NEO  
Figure 19: Re-encoding corrects some NEO continuations. The first rotation is exact, but the following translation moves 3, 3, and 4 cells instead of 4, 5, and 5. Re-encoding the decoded image before applying the same second code corrects these cases.

(a) NEO  
![](images/de480c4feba241412554bd1f57ebd38776e7e99842c82729b65f5e6439767132.jpg)

(b) NEO  
![](images/9411db913300f86b215db79fc41e682d8328b9ae8b0aa18503530e9da5190da6.jpg)

(c) NEO  
![](images/a223ac3d917a9083835e58cd243fa9b6dd75be9652b7ac263c2329aa9ba68ce5.jpg)  
Figure 20: Re-encoding also breaks correct NEO continuations. Under the same intervention, three previously correct queries become incorrect. Together with Figure 19, these cases show state-dependent execution; reencoding is not a general remedy.

## I Latent-space visualization

We select tasks generated by depth-1 programs from the Single-A Level 1 dataset and provide the first support input–output pair of each task to each model’s programmer. We visualize the resulting latent representations using t-SNE. Each point is colored according to the primitive that generated its source task; colors do not denote the learned semantics of the latent vectors or codebook entries.

For Cont-Mono, we project the continuous latent vectors. For Disc-Mono and NEO, we jointly project the pre-quantization continuous vectors and the codebook vectors. We fit t-SNE independently for each checkpoint. All codebook entries are included in the projection, but only entries selected by the plotted vectors in the original latent space are displayed.

![](images/00cf0f8a9c855976e6f83407477190bc86f796dc332033e6e6d488a22738d04f.jpg)

Figure 21: Continuous latent representations of Cont-Mono. Each point represents the latent inferred from a task’s first support pair. Colors identify the primitives that generated the source tasks.  
![](images/007fca313b0ca6419b105d3e59f88401fa3dd368168d18ee83985b851a9c3c71.jpg)  
Figure 22: Pre-quantization representations and codebook vectors of Disc-Mono. Points represent continu ous vectors before quantization, colored by the primitives that generated the source tasks. Hollow diamond mark selected codebook entries, and numeric labels denote code IDs within each checkpoint.

![](images/c74c0bda09a742e305b332788d11f1bf1973da5cd0951d24a832111e500317e9.jpg)  
Figure 23: Pre-quantization representations and codebook vectors of NEO. Points represent continuous vectors before quantization, colored by the primitives that generated the source tasks. Hollow diamonds mark selected codebook entries, and numeric labels denote code IDs within each checkpoint.

## J Licenses

Table 17: Existing assets, licenses, and source links. We list the existing models and public code repositories used in our experiments. The BabyTheorist data and generator are created by the authors and are therefore not external assets.
<table><tr><td>Asset</td><td>Use in this paper</td><td>License / terms</td><td>Source link</td></tr><tr><td>GPT-OSS 120B</td><td>Open-weight LLM baseline (Section 5.1)</td><td>Apache License 2.0</td><td>https://huggingface.co/openai/ gpt-oss-120b</td></tr><tr><td>Qwen 3.5 27B</td><td>Open-weight LLM baseline (Section 5.1)</td><td>Apache License 2.0</td><td>https://huggingface.co/Qwen/ Qwen3.5-27B</td></tr><tr><td>NEO /L2T repository [7]</td><td>NEO baseline reference implementation</td><td>MIT License</td><td>https://github.com/ahn-ml/ learning-to-theorize</td></tr><tr><td>Dreamweaver repository [35]</td><td>(Appendix B) Slot Attention and object-centric components for</td><td>MIT License</td><td>https://github.com/ahn-ml/ dreamweaver-release</td></tr><tr><td>DUO repository [36]</td><td>Cont-Mono-OC Reference implementation used</td><td>Apache License 2.0</td><td>https://github.com/s-sahoo/duo</td></tr><tr><td>GPT-5.6 Sol</td><td>in our baseline code Proprietary reference model (Appendix G.3)</td><td>Accessed via the OpenAI API under OpenAI&#x27;s terms of use</td><td>https://openai.com/policies/ terms-of-use</td></tr></table>

BabyTheorist release. The generator and evaluation code are released under the MIT License at github.com/ahn-ml/babytheorist. The dataset is released under CC0 1.0 on Hugging Face at huggingface.co/datasets/ahn-ml/babytheorist and is archived on Harvard Dataverse with a persistent DOI, doi:10.7910/DVN/H9UGTL, which also provides its Croissant metadata.