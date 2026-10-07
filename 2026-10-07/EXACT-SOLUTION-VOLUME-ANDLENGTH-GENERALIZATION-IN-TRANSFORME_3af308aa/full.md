# EXACT-SOLUTION VOLUME ANDLENGTH GENERALIZATION IN TRANSFORMERS

Yijia Jessica Zhu and David Chiang University of Notre Dame {yzhu26,dchiang}@nd.edu

## ABSTRACT

Research on transformer expressivity shows whether a transformer is capable of solving a given task, but gives little indication of whether the solution, if learned, is generalizable to longer input lengths. We study this question through normalized exact-solution volume (NESV): the fraction of a bounded parameter region that achieves an exact solution on every input of length n. For fixed-width, single-layer transformers with log n-scaled attention, we establish asymptotic bounds on NESV for four tasks: FIRST (Θ(1)), MAJORITY (Θ(1/(n log n))), INDEX (Θ(1/n<sup>3</sup>)), and PARITY (0). These results are consistent with previous empirical results: the faster the exact-solution volume decays with input length, the harder it is to lengthgeneralize on that task. Looking deeper into INDEX, our volume analysis reveals two error sources that grow with n. Consequently, we study a transformer model that would structurally eliminate one of the terms, theoretically improving the NESV bound to Θ(n<sup>−1</sup>), and empirically achieving 85% accuracy when tested at 10× the training length, compared with the 60% accuracy of the original model. We conclude that volume analysis may be a useful approach to identify concrete sources of length sensitivity and thus provide insights into task-specific model refinements.

## 1 INTRODUCTION

Theoretical work on the expressivity of transformers (cf. the survey by Strobl et al., 2024) can be seen as a prelude to the study of learnability by transformers, since the existence of a transformer and parameters that solve a given problem is a prerequisite for a transformer to learn to solve the problem. But the existence of a single point in parameter space does not tell us much about whether that point is actually learnable. In this paper, following Kover et al. (2026), we ask a more difficult question: given that a problem is solvable by a transformer with some parameters, how large is the set of successful parameters? While Kover et al. (2026) arrived at somewhat pessimistic conclusions, we look at single-layer transformers with log n-scaling and find that, the volume of the set of successful parameters, as a function of the sequence length n, provides a way to quantify the difficulty of length-generalization in various problems.

The drive for looking beyond expressivity is also visible in experiments on simple sequence tasks. Chiang & Cholak (2022) construct exact transformers for FIRST (whether the first symbol is a 1) and PARITY (whether the number of 1’s is odd), but report that standard training fails on PARITY, and that FIRST does not necessarily extrapolate to longer inputs without log n-scaling. Hahn & Rofin (2024) report that fitted PARITY solutions become much sharper as input length grows than fitted solutions for FIRST and MAJORITY (whether there are more 1’s than 0’s). Thus, tasks that are all expressible can behave differently under training and length extrapolation.

Such findings motivate examining the geometry of the parameter sets that implement a task. In related work on loss landscapes, Chiang et al. (2023) argue that the geometry and volume of low-loss regions can explain part of neural-network generalization without relying entirely on the implicit bias of gradient descent. We study a related exact-solution quantity at each finite input length. For a fixed task and architecture, we restrict the trainable parameters to a bounded region and measure the fraction that correctly solves the task on every input of length n. We call this quantity the normalized exact-solution volume (NESV). Its dependence on n describes how the full set of exact solutions changes as the computation is required to handle longer inputs. With uniform sampling over the bounded parameter region, this is the probability of sampling an exact solution. Comparing this baseline with empirical training results can help distinguish the effects of initialization and optimization from the nature of the task and the selected transformer model in terms of learnability.

Table 1: NESV bounds and mean accuracy at 10× the maximum training length for the four tasks. The models $\mathcal { A } _ { \mathrm { g e n } } , \mathcal { A } _ { \mathrm { s i n } }$ , and ${ \mathcal A } _ { \mathrm { p o s } }$ are described in Section 2.2, while $\mathcal { A } _ { \mathrm { n o r m } }$ is described in Section 4.2.
<table><tr><td rowspan="3">Task</td><td colspan="8">Model</td></tr><tr><td colspan="2">Positional  $( \mathcal { A } _ { \mathrm { p o s } } )$ </td><td colspan="2">General  $\left( \mathcal { A } _ { \mathrm { g e n } } \right)$ </td><td colspan="2">Sinusoidal  $( \mathcal { A } _ { \mathrm { s i n } } )$ </td><td colspan="2">Normalized  $( \mathcal { A } _ { \mathrm { n o r m } } )$ </td></tr><tr><td>NESV Acc. (%)</td><td></td><td>NESV</td><td>Acc. (%)</td><td>NESV</td><td>Acc. (%)</td><td>NESV</td><td>Acc. (%)</td></tr><tr><td>FIRST</td><td>Θ(1)</td><td>100.0</td><td>0</td><td>51.12</td><td>0</td><td>55.06</td><td>Θ(1)</td><td>100.0</td></tr><tr><td>MAJORITY</td><td></td><td>92.7</td><td> $\Theta ( ( n \log n ) ^ { - 1 } )$ </td><td>82.6</td><td> $\Theta ( ( n ^ { 3 } \log n ) ^ { - 1 } )$ </td><td>87.5</td><td></td><td>92.7</td></tr><tr><td>INDEX</td><td> $\Theta ( n ^ { - 3 } )$ </td><td>59.6</td><td>0</td><td>51.9</td><td>0</td><td>50.2</td><td> $\Theta ( n ^ { - 1 } )$ </td><td>85.1</td></tr><tr><td>PARITY</td><td>0</td><td>一</td><td>0</td><td>一</td><td>0</td><td>一</td><td>0</td><td></td></tr></table>

We consider four binary classification tasks on binary strings. FIRST returns the first input symbol, whereas INDEX returns the symbol at a position specified by one of the input tokens. MAJORITY returns whether more than half of input symbols are 1’s, whereas PARITY returns whether the number of 1’s is odd. We refer to the first pair as positional tasks, and the second pair as counting tasks.

We conduct our analyses on the simplest models suited to solve the corresponding problems. In all cases, we only consider single-layer transformers. For the positional tasks, we use a model with positional encoding $( j , j ^ { 2 } )$ where the Query and Key maps only depend on token positions and the Value map only depends on the tokens themselves (following the form of the construction of Bhattamishra et al. (2024)). For the counting tasks, the positions of the tokens are irrelevant, so we primarily use a model without positional encoding. We also examined a model with sinusoidal positional encoding (sin(j), cos(j)).

See Table 1 for a summary of the asymptotic bounds obtained with respect to input length n. Under the positional model ${ \mathcal { A } } _ { \mathrm { p o s } } ,$ , the NESV for INDEX is $\Theta ( n ^ { - 3 } )$ , while FIRST has constant NESV. The sinusoidal model $\mathcal { A } _ { \mathrm { s i n } }$ has volume 0 for both. Under the position-free model ${ \mathcal { A } } _ { \mathrm { g e n } } ,$ the NESV for MAJORITY is $\Theta ( 1 / ( n \log n ) )$ . Under the sinusoidal model $\mathcal { A } _ { \mathrm { s i n } } .$ , the NESV for MAJORITY is instead $\Theta ( 1 / ( n ^ { 3 } \log n ) )$ . PARITY has a zero NESV whenever $n > h + 1$ for both models, h being the width of the transformer. This is a direct consequence of the standard one-dimensional capacity of shallow ReLU networks (Arora et al., 2018) in the finite-length setting, and is consistent with the broader result that no one-layer, one-head transformer computes PARITY across all lengths (Kozachinskiy et al., 2026b).

We train those tasks under the same protocol. The empirical results are in line with our NESV analysis: the task-model pair whose NESV shrinks faster as n grows shows worse length generalization. The only counterexample was that, under the general sinusoidal model, MAJORITY trains as well as under the position-free model $\mathcal { A } _ { \mathrm { g e n } }$ , although the former’s volume shrinks faster with n.

Among the tasks we examine, INDEX demonstrates particularly bad length generalization (60% average accuracy at ten times training length) despite being expressible (Kozachinskiy et al., 2026a). Our NESV analysis reveals the underlying mechanisms. Exact retrieval of a designated position k requires that position to receive enough attention, so its attention score must remain within a constant of the maximum attention score. Expressing the difference in attention scores between neighboring positions as a function of k reveals two key error components, one proportional to $k ^ { 2 }$ and another proportional to k, which are the sources of failing length generalization.

Guided by this analysis, we modify the model so that the error term proportional to $k ^ { 2 }$ no longer grows with input length. This is called the normalized positional model. The NESV improves from $\mathsf { \bar { \Theta } } ( n ^ { - 3 } )$ to $\Theta ( n ^ { - 1 } )$ . In our experiments, the modified model also generalizes substantially better on INDEX, achieving an average accuracy of 85% percent at ten times the training length.

Taken together, the results show that expressivity alone can conceal qualitatively different parameterspace behavior. Volume analysis, on the other hand, may offer another useful tool to further identify the computational constraints responsible for learning behaviors and offer insight for task-specific model refinements.

## 2 SETUP

In this section, we describe the tasks to be analyzed and the transformers to be used in the analysis.

## 2.1 TASKS AND EXACT-SOLUTION VOLUME

For $n \geq 1$ , write $[ n ] = \{ 1 , \dots , n \}$ and let $w = ( w _ { 1 } , \ldots , w _ { n } ) \in \{ 0 , 1 \} ^ { n }$ . We study

$$
\begin{array} { r } { \mathrm { F I R S T } _ { n } ( w ) = w _ { 1 } \qquad } \\ { \mathrm { I N D E X } _ { n } ( w , k ) = w _ { k } , \quad k \in [ n ] , } \end{array}
$$

$$
\mathbf { M A J O R I T Y } _ { n } ( w ) = \mathbf { 1 } \left\{ \sum _ { j = 1 } ^ { n } w _ { j } \geq \frac { n + 1 } { 2 } \right\} \quad \mathrm { f o r ~ o d d } \ n ,
$$

$$
\mathrm { P A R I T Y } _ { n } ( w ) = \left( \sum _ { j = 1 } ^ { n } w _ { j } \right) { \bmod { 2 } } .
$$

For INDEX, the n tokens $w _ { 1 } , \ldots , w _ { n } ,$ which we call data tokens, are followed by a single token, which we call the index token, encoding k. We refer to this as right-hand sequence INDEXing. For the other tasks, there is an index token for consistency’s sake, but it is fixed.

Definition 1 (Normalized exact-solution volume, NESV). For a task $T$ and transformer model A, let $S _ { n } ^ { T , A }$ be the parameters for which the output is strictly correct on every length-n input, and on every index k. For a fixed parameter bound $B \bar { > } 0$ , define

$$
V _ { B } ^ { T , A } ( n ) = \frac { \lambda _ { D _ { \cal A } } \left( S _ { n } ^ { T , A } \cap [ - B , B ] ^ { D _ { \cal A } } \right) } { ( 2 B ) ^ { D _ { \cal A } } } .\tag{1}
$$

Equivalently, (1) is the probability of obtaining a correct solution when each scalar parameter is sampled independently and uniformly from $[ { \bar { - B } } , B ]$ . Unless stated otherwise, B and the hidden width of the $\mathrm { F F N N } , h ,$ are fixed as $n$ grows. Throughout, log denotes the natural logarithm; $\lVert \cdot \rVert _ { 2 }$ and $\| \cdot \| _ { \infty }$ denote the Euclidean and maximum-coordinate norms. Throughout, $c , C > 0$ and the constants implicit in the subscripted asymptotic notation such as $\Theta _ { B , h }$ may depend on $B$ and $h ,$ but never on $n ; c , C$ may change from line to line.

## 2.2 TRANSFORMER MODELS

We consider only single-layer transformers. Let $x _ { j }$ denote the $j \cdot$ -th data token and let z denote the index token. The query, key, and value maps have the form $Q ( x ) = W _ { Q } x + b _ { Q } , K ( x ) =$ $W _ { K } x + b _ { K } , V ( x ) = \hat { W _ { V } } x \dot { + } b _ { V }$ . Writing $q = \bar { Q ( z ) } , k _ { j } = K ( x _ { j } ) , v _ { j } = V ( x _ { j } )$ , the attention output is

$$
\alpha _ { j } = \frac { \exp \bigl ( ( \log n ) q ^ { \top } k _ { j } \bigr ) } { \sum _ { i = 1 } ^ { n } \exp ( ( \log n ) q ^ { \top } k _ { i } ) } , \qquad A = \sum _ { j = 1 } ^ { n } \alpha _ { j } u _ { j } .\tag{2}
$$

The log n factor was proposed by Chiang & Cholak (2022) and Nakanishi (2025) and has been used in practice in Qwen (Bai et al., 2023). Chen et al. (2026) provide additional theoretical justification for this choice. In this paper, this choice is natural since we study positional tasks such as INDEX, where attention must continue to select a particular position as the number of competing positions grows with the input length. We omit the $1 / \sqrt { d }$ factor in conventional log n scaling because d is constant in n and does not affect the asymptotic bounds in $n .$

The self-attention is followed by a feed-forward network (FFNN) classifier with hidden width h:

$$
y = W _ { 2 } \operatorname { R e L U } ( W _ { 1 } A + b _ { 1 } ) + b _ { 2 } \in \mathbb { R } ^ { 2 } ,\tag{3}
$$

where $W _ { 1 } \in \mathbb { R } ^ { h \times 2 } , b _ { 1 } \in \mathbb { R } ^ { h } , W _ { 2 } \in \mathbb { R } ^ { 2 \times h } , b _ { 2 } \in \mathbb { R } ^ { 2 } .$

The two coordinates of y are indexed by the labels $\{ 0 , 1 \}$ , and the predicted label is $\widehat { y } \ =$ $\mathrm { a r g m a x } _ { c \in \{ 0 , 1 \} } ( y ) _ { c }$ . When the input needs to be explicit, we write $A ( w , k )$ and $y ( w , k )$

We do not use residual connections or layer normalization. Adding them could alter the size and geometry of the exact-solution set; we leave this extension for future work.

In this paper, we consider four models. We describe three of them here; the fourth, $A _ { \mathrm { n o r m } }$ , we leave for Section 4.2.

In the general model $A _ { \mathrm { g e n } } ( h )$ , the data tokens are $x _ { j } = e _ { w _ { j } } \in \mathbb { R } ^ { 2 }$ , and the index token is $z = 0 _ { 2 }$ The maps $Q , K , V : \mathbb { R } ^ { \bar { 2 } }  \mathbb { R } ^ { 2 }$ are unrestricted affine maps, so $q = Q ( 0 _ { 2 } ) , k _ { j } = K ( e _ { w _ { j } } )$ , and $v _ { j } = V ( e _ { w _ { j } } ) . \mathrm { S o } \mathcal { A } _ { \mathrm { g e n } } ( h )$ has $D _ { \mathrm { c o u n t } } = 5 h + 2 0$ trainable scalar parameters.

We also examined a model, $\mathcal { A } _ { \mathrm { s i n } } ( h )$ , with a sinusoidal positional encoding $( \sin ( j ) , \cos ( j ) )$ .

For the positional tasks, we use the positional model $A _ { \mathrm { p o s } } ( h )$ . The data token at position j and the index token encoding k are

$$
\begin{array} { r } { x _ { j } ^ { \mathrm { p o s } } = \left[ \begin{array} { c } { e _ { w _ { j } } } \\ { j } \\ { j ^ { 2 } } \end{array} \right] \in \mathbb { R } ^ { 4 } , \qquad z _ { k } ^ { \mathrm { p o s } } = \left[ \begin{array} { c } { 0 _ { 2 } } \\ { k } \\ { 0 } \end{array} \right] \in \mathbb { R } ^ { 4 } } \end{array}
$$

(where $0 _ { 2 }$ is the two-dimensional zero vector). For FIRST, the index is fixed at $k = 1$ . For INDEX, the index $k \in [ n ]$ is supplied as part of the input. Because j and $j ^ { 2 }$ grow without bound in the input length, allowing them to propagate through the network would like lead to instability; positional-encoding choices are known to strongly affect length extrapolation (Kazemnejad et al., 2023). Accordingly, we constrain the model to have the form of the index-lookup construction of Bhattamishra et al. (2024):

$$
\begin{array} { l l l } { { W _ { Q } ^ { \mathrm { p o s } } = \left[ 0 _ { 2 \times 2 } \quad a \quad 0 _ { 2 } \right] , } } & { { \qquad } } & { { b _ { Q } ^ { \mathrm { p o s } } = b , } } \\ { { W _ { K } ^ { \mathrm { p o s } } = \left[ 0 _ { 2 \times 2 } \quad r \quad s \right] , } } & { { \qquad } } & { { b _ { K } ^ { \mathrm { p o s } } = c _ { K } , } } \\ { { W _ { V } ^ { \mathrm { p o s } } = \left[ C _ { V } \quad 0 _ { 2 \times 2 } \right] , } } & { { \qquad } } & { { b _ { V } ^ { \mathrm { p o s } } = b _ { V } , } } \end{array}
$$

where $a , b , r , s , c _ { K } , b _ { V } \in \mathbb { R } ^ { 2 } , C _ { V } \in \mathbb { R } ^ { 2 \times 2 }$ (and $0 _ { 2 \times 2 }$ is the $2 \times 2$ zero matrix). This model has $D _ { \mathrm { p o s } } = 5 h + 1 8$ trainable scalar parameters.

## 3 NESV BOUNDS AND LENGTH GENERALIZATION

We first establish the volume bounds for the four tasks and then compare such results with those of length-generalization experiments.

## 3.1 MAIN RESULT

Theorem 1 (NESV bounds). Fix $B > 0$ and $h \geq 2 . A s n  \infty$

$$
V _ { B } ^ { \mathrm { F I R S T } , A _ { \mathrm { p o s } } ( h ) } ( n ) = \Theta _ { B , h } ( 1 )\tag{4}
$$

$$
V _ { B } ^ { \mathrm { I N D E X } , A _ { \mathrm { p o s } } ( h ) } ( n ) = \Theta _ { B , h } ( n ^ { - 3 } )\tag{5}
$$

$$
V _ { B } ^ { \mathrm { M A J O R I T Y } , \mathcal { A } _ { \mathrm { g e n } } ( h ) } ( n ) = \Theta _ { B , h } \bigl ( ( n \log n ) ^ { - 1 } \bigr )
$$

$$
n o d d\tag{6}
$$

$$
V _ { B } ^ { \mathrm { P A R I T Y } , A _ { \mathrm { g e n } } ( h ) } ( n ) = 0
$$

$$
n > h + 1 .\tag{7}
$$

Proofsketch. For FIRST, one can choose parameters such that the attention score decreases quadrati cally away from position one. A fixed neighborhood of these parameters preserves that decrease, and the resulting concentration of attention lets the classifier return the first symbol for all sufficiently large lengths. The neighborhood does not shrink with n, thus ensuring a constant NESV bound.

However, the same argument cannot be used for INDEX, because the attention score must peak at $j = k$ for every possible k, and small errors that are harmless at one position can be amplified at another. We examine the two relevant error sources in Section 4.

In the position-free model $\mathcal { A } _ { \mathrm { g e n } } ,$ , a word is represented only by its number of 1’s. For MAJORITY, the output changes only once, between counts $( n - 1 ) / 2$ and $( { \dot { n } } + 1 ) / 2$ . These counts differ by just one, so the interval that separates them narrows as n increases. Consequently, placing the decision boundary between them requires increasingly precise parameter choice.

For PARITY, by contrast, every increment in the count must flip the output. A fixed number of hidden ReLU units cannot provide arbitrarily many flips in output. The upper bound of zero, meaning that PARITY is not solvable by the class of transformers considered here, is consistent with previous theoretical results (Kozachinskiy et al., 2026b) and experiments (Chiang & Cholak, 2022; Hahn & Rofin, 2024).

The complete proofs of the four bounds, including the technical details omitted from the INDEX proof sketch in Section 4.1, are given in Appendix A. □

For detailed results related to the sinusoidal model $\mathcal { A } _ { \mathrm { s i n } }$ , see Appendix C.

## 3.2 EXPERIMENT

To compare these bounds with empirical results, we train the FIRST, INDEX, and MAJORITY models with hidden width $h = 8$ . We use 20 random seeds per task and draw a fresh batch at each update, sampling its length uniformly from $1 , \ldots , N ;$ ; for MAJORITY, we use only odd lengths. The main results use $N = 5 1$ . Checkpoints are selected using validation data at length $N _ { \ast }$ , without inspecting longer inputs, and tested on random examples at lengths up to 511. Additional experiments on $N = 1 1 , { \dot { 2 } } 1 , 3 1 , 4 1$ , which give similar results, are in Appendix D.

Table 1 shows that FIRST remains accurate at the longest test length (under $\mathcal { A } _ { \mathrm { p o s } } )$ , MAJORITY retains substantial accuracy (under $\mathcal { A } _ { \mathrm { g e n } } )$ , and INDEX falls much closer to chance (under $\mathcal { A } _ { \mathrm { p o s } } )$ . These observations agree with the ranking of the three volume bounds.

We found the results of INDEX particularly interesting. Despite the task being expressible and learnable on lengths up to 51, it showed poor length generalization, much worse than that of MAJORITY. We take a closer look at this task in the next section.

## 4 THE INDEX TASK

We first revisit the volume argument from the previous section, to identify the parameter combinations responsible for the n-dependent decay. We then modify the model to control one of these combinations structurally and compare the resulting theoretical and empirical behavior.

## 4.1 LENGTH-SENSITIVE COMPONENTS OF THE ATTENTION SCORE

Recall that, in the positional model $\begin{array} { r } { A _ { \mathrm { p o s } } ( h ) , q = a k + b , k _ { j } = r j + s j ^ { 2 } + c _ { K } , u _ { j } = C _ { V } e _ { w _ { j } } + b _ { V } . } \end{array}$ The attention score is

$$
g _ { j } = q ^ { \top } k _ { j } = ( a k + b ) ^ { \top } ( r j + s j ^ { 2 } + c _ { K } ) .\tag{8}
$$

We compare the scores at position k and position $k + \Delta$ . The expansion will produce the following quantities used repeatedly in later proofs, so we introduce some notations for future convenience.

$$
\chi = a ^ { \top } s \qquad \ell = 2 b ^ { \top } s + a ^ { \top } r \qquad \delta = b ^ { \top } s \qquad \phi = b ^ { \top } r .\tag{9}
$$

Now consider position $k + \Delta \in [ n ]$ . The score gap of this position is

$$
g _ { k + \Delta } - g _ { k } = \Delta \underbrace { \left( 2 \chi k ^ { 2 } + \ell k + \phi \right) } _ { C _ { \mathrm { l i n } } ( k ) } + \Delta ^ { 2 } \underbrace { \left( \chi k + \delta \right) } _ { C _ { \mathrm { q u a d } } ( k ) } .\tag{10}
$$

We want this gap to be negative. The coefficient $C _ { \mathrm { q u a d } } ( k )$ controls the curvature of the score around the indexed position, while $C _ { \mathrm { l i n } } ( k )$ measures its first-order drift and therefore whether the score remains centered at $j = k ;$ we want it to be zero. Because χ and ℓ are multiplied by k and $k ^ { 2 }$ , small errors in these parameter combinations can be amplified at large k (i.e., at large n).

## 4.1.1 SKETCH OF LOWER BOUND

See Section A.2.1 for the full proof of the lower bound.

Let $\eta = \operatorname* { m i n } \{ B / 1 6 , 1 / 1 6 \}$ and consider the solution (which we call the center):

$$
a ^ { 0 } = ( 2 \eta , 0 ) , \qquad b ^ { 0 } = ( 0 , - \eta ) , \qquad r ^ { 0 } = ( \eta , 0 ) , \qquad s ^ { 0 } = ( 0 , \eta ) .\tag{11}
$$

We want to find a region around the center that still solves the problem. The volume of this region will serve as our lower bound. At this center, $\chi = \ell = \phi = 0 , \delta = - \eta ^ { 2 }$ , and the score gap is

$$
g _ { k + \Delta } - g _ { k } = - \eta ^ { 2 } \Delta ^ { 2 } .\tag{12}
$$

In a sufficiently small neighborhood of the center, we may restrict

$$
| \chi | \leq c n ^ { - 2 } , \qquad | \ell | \leq c n ^ { - 1 } ,\tag{13}
$$

while allowing $a , b , r _ { 2 } $ , and $s _ { 2 }$ (the second coordinates of $r$ and s, resp.) to vary over fixed-width intervals independent of $n .$ Every parameter choice in this region then satisfies $\begin{array} { r } { g _ { k + \Delta } - g _ { k } \le - \frac { \eta ^ { 2 } } { 4 } \Delta ^ { 2 } } \end{array}$ for every $k \in [ n ]$ and every nonzero $\Delta$ such that $k + \Delta \in [ n ]$ . Under the log n attention scale, the total attention outside the indexed position then tends to zero as $n \to \infty$ uniformly in k. Hence, for all sufficiently large $n ,$ the indexed position receives at least a fixed fraction of the total attention, uniformly over all inputs and indexes. We may then restrict $( C _ { V } , b _ { V } )$ and $( W _ { 1 } , b _ { 1 } , W _ { 2 } , b _ { 2 } )$ to fixedsize neighborhoods, independent of $n ,$ , such that, for some constant $c _ { 0 } > 0$ independent of $n ,$ $y _ { w _ { k } } ( w , k ) - y _ { 1 - w _ { k } } ( w , k ) \geq c _ { 0 }$ for every input $( w , k )$ . To produce a exact solution, only the width of the interval $\chi$ and ℓ need to depend on $n .$ . Therefore the constructed set of exact solutions has normalized volume

$$
\Omega ( n ^ { - 2 } ) \cdot \Omega ( n ^ { - 1 } ) = \Omega ( n ^ { - 3 } ) .\tag{14}
$$

## 4.1.2 SKETCH OF UPPER BOUND

See Section A.2.2 for the full proof of the upper bound.

For each indexed position $k ,$ compare a constant word with a word that differs from it only at position k. These two inputs must be assigned different labels. We consider two cases. Let $m =$ $y _ { w _ { k } } ( w , k ) - y _ { 1 - w _ { k } } ( w , k )$ . In any correct solution, we will have $m > 0$ , but if $m \leq n ^ { - 5 }$ , we call the solution small-margin; otherwise, we call it large-margin.

Take any one of the inequalities $y _ { w _ { k } } ( w , k ) > y _ { 1 - w _ { k } } ( w , k )$ ; there are 4n such inequalities in total. Fix arbitrary values of $a , b , r , s , c _ { K } , C _ { V } , b _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 }$ , and $b _ { 2 , 1 - w _ { k } } . \mathrm { A s } \ b _ { 2 , w _ { k } }$ varies, the margin of the inequality varies with slope one. To satisfy $0 < y _ { w _ { k } } ( w , k ) - y _ { 1 - w _ { k } } ( w , k ) \leq n ^ { - 5 } , b _ { 2 , w _ { k } }$ needs to lie in an interval of length $n ^ { - 5 }$ . Thus the set of small-margin solutions has normalized volume at most $4 n \cdot n ^ { - 5 } / ( 2 B ) = \bar { O ( } n ^ { - 4 } )$

For every large-margin solution, changing only the symbol at position k from τ to σ $\neq \tau$ must change the predicted label from τ to $\sigma . { \mathrm { B y } }$ the definition of a large-margin solution, for each of the two words w in this comparison, $y _ { w _ { k } } ( w , k ) \dot { - y _ { 1 - w _ { k } } ( w , k ) } > n ^ { - 5 }$ . Since the $y _ { \sigma } ( w , k ) - y _ { \tau } ( w , k )$ , viewed as a function of the attention output $A ( w , k )$ is Lipschitz, this is impossible if the attention assigned to position k is too small. Hence the indexed position receives at least an inverse polynomial amount of attention. The softmax formula in $( 2 )$ then implies that $g _ { k }$ lies within a constant of the maximum score. Comparing it with the two neighboring scores in (10) gives $C _ { \mathrm { q u a d } } ( k ) + | C _ { \mathrm { l i n } } ( k ) | \le M _ { 0 }$ where $M _ { 0 }$ is independent of $n .$ Since every coordinate of $a , b , s$ lies in $[ - B , B ]$ , both $\chi = a ^ { \intercal } \ : s$ and $\delta = b ^ { \top } .$ s are bounded independently of n. Hence, for $k \le n , | C _ { \mathrm { q u a d } } ( \dot { k } ) | = | \dot { \chi } \dot { k } + \delta | = O ( n )$ , and thus $| C _ { \mathrm { l i n } } ( k ) | \le M _ { 0 } + | C _ { \mathrm { q u a d } } ( k ) | = O ( n )$ . Let $M = \lfloor n / 4 \rfloor$ . Since $C _ { \mathrm { l i n } } ( k ) = 2 \chi k ^ { 2 } + \ell k + \phi$ we have $C _ { \mathrm { l i n } } ( 3 M ) - 2 C _ { \mathrm { l i n } } ( 2 M ) + C _ { \mathrm { l i n } } ( M ) = 4 \chi M ^ { 2 }$ . Because each term on the left is $O ( n )$ and $M = \Theta ( n )$ , this gives $| \chi | = \dot { O } ( n ^ { - 1 } )$ . Moreover, $C _ { \mathrm { l i n } } ( 2 M ) - C _ { \mathrm { l i n } } ( M ) = 6 \chi M ^ { 2 } + \ell M$ . Using $| \chi | = O ( n ^ { \dot { - } 1 } )$ in this identity gives $| \ell | = O ( 1 )$ . Repeating the same argument gives

$$
| \chi | = O ( n ^ { - 2 } ) , \qquad | \ell | = O ( n ^ { - 1 } ) .\tag{15}
$$

We used the fact

$$
\operatorname* { P r } \left( | a ^ { \top } s | \leq \varepsilon _ { 1 } , | 2 b ^ { \top } s + a ^ { \top } r | \leq \varepsilon _ { 2 } \right) = O _ { B } ( \varepsilon _ { 1 } \varepsilon _ { 2 } )\tag{16}
$$

whenever $0 < \varepsilon _ { 1 } \le \varepsilon _ { 2 }$ are sufficiently small (for detailed proof see Section A.2.2, Lemma 3). Apply this fact to (15) and adding the $O ( n ^ { - 4 } )$ small-margin volume would gives the upper bound ${ \dot { O } } ( n ^ { - 3 } )$

![](images/cdb8c56f7c664026eb52066d6c95b0c1073508a5e5081fc433b847506906b6c0.jpg)

![](images/a121665db7f70d4371b382c507e70d42b422d167a4e9c00d090fd6ed180683b0.jpg)  
Figure 1: Scale-normalized drift coefficients and mean OOD accuracy for trained positional INDEX models. Markers indicate the maximum training length N.

## 4.1.3 EXPERIMENTAL ANALYSIS

We next check whether $\chi = a ^ { \top } .$ s and $\ell = 2 b ^ { \top } s + a ^ { \top } r$ identified by the volume proof track the behavior of trained models. In (10), χ controls the error term whose effect grows quadratically with k, while ℓ controls the error term whose effect grows linearly with k. For a model trained up to maximum length N, we therefore consider the quantities $N ^ { 2 } | \dot { \chi } |$ and $N \vert \ell \vert$ , allowing models trained at different maximum lengths to be comparable. Across the 100 positional models trained at five maximum lengths, smaller values of both quantities are associated with higher mean accuracy on unseen lengths; see Fig. 1. Ranking the models separately within each maximum training length gives correlations of −0.655 for $N ^ { 2 } | _ { \chi } |$ and −0.516 for $N | \dot { \ell } |$ . Both have $p < 1 0 ^ { - 3 }$ under permutation tests that preserve the five training-length groups. The training and evaluation protocol is given in Sections D.2 and D.3, and the coefficient analysis is detailed in Section D.4.

The above analysis suggests a direct architectural change: instead of relying on training to place two free coefficients inside shrinking intervals, control one of the length-growing error terms through the model itself.

## 4.2 THE NORMALIZED MODEL

We introduce a modified positional model, denoted $A _ { \mathrm { n o r m } } ( h )$ . It uses the same input vectors, value map, ReLU classifier, and trainable positional parameters $\mathrm { \bar { \Phi } } _ { a , b , r , s , c _ { K } \mathrm { \ a s \ } \bar { A } _ { \mathrm { p o s } } } \mathrm { \bar { ( } } h )$ . We split the computation of the attention score into two channels: a linear channel that does not depend on the $j ^ { 2 }$ coordinate of the keys’ positional encoding, and a quadratic channel that depends only on the $j ^ { 2 }$ coordinate. In the quadratic channel, we modify the queries as follows. Define the fixed smooth map

$$
F ( v ) = \frac { v } { \sqrt { 1 + \| v \| _ { 2 } ^ { 2 } } } \qquad ( v \in \mathbb { R } ^ { 2 } ) .\tag{17}
$$

The linear and quadratic positional channels are

$$
q ^ { \mathrm { l i n } } = a k + b , \qquad q ^ { \mathrm { q u a d } } = b + F ( a k ) , \qquad k _ { j } ^ { \mathrm { l i n } } = r j + c _ { K } , \qquad k _ { j } ^ { \mathrm { q u a d } } = s j ^ { 2 } .\tag{18}
$$

Thus the linear channel uses the full query $a k + b ,$ whereas the quadratic channel applies $F$ to the index-dependent contribution ak. The attention score is

$$
\begin{array} { r l } { g _ { j } ^ { \mathrm { n o r m } } = \left( q ^ { \mathrm { l i n } } \right) ^ { \top } k _ { j } ^ { \mathrm { l i n } } + \left( q ^ { \mathrm { q u a d } } \right) ^ { \top } k _ { j } ^ { \mathrm { q u a d } } } & { = ( a k + b ) ^ { \top } ( r j + c _ { K } ) + \left( b + F ( a k ) \right) ^ { \top } s j ^ { 2 } . } \end{array}\tag{19}
$$

The attention weights are obtained by applying the same log n-scaled softmax to (19); the values and the one-hidden-layer ReLU classifier are unchanged. The normalization introduces no new trainable parameters, so $\bar { A _ { \mathrm { n o r m } } } ( h )$ has the same parameter dimension and uses the same B as $A _ { \mathrm { p o s } } ( h )$

This modification could be applied to a general transformer by excluding $j ^ { 2 }$ from the position encoding at position j but adding an additional term $f ( q _ { i } , k _ { j } ) j ^ { 2 }$ directly into the attention score, with f being a scalar-valued function whose output remains bounded. We leave exploration of this more general setting for future work.

The map $F$ is approximately the identity near the origin, but its output has norm strictly smaller than one. Consequently, the component of the query that depends on k and is multiplied with $j ^ { 2 }$ no longer grows linearly with k. To make this precise, suppose first that $a \neq 0$ and write $\widehat { a } = a /  { \lVert a \rVert } _ { 2 }$ Retaining $\delta$ and $\phi$ from (9), define

$$
\widehat { \chi } : = \widehat { \boldsymbol { a } } ^ { \intercal } \boldsymbol { s } , \qquad \widehat { \ell } : = \ell + 2 \widehat { \chi } , \qquad \varepsilon _ { \chi } ( \boldsymbol { k } ) : = \big ( \boldsymbol { F } ( \boldsymbol { a } \boldsymbol { k } ) - \widehat { \boldsymbol { a } } \big ) ^ { \intercal } \boldsymbol { s } .\tag{20}
$$

Then the modified quadratic and linear coefficients are

$$
C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) = \widehat { \chi } + \delta + \varepsilon _ { \chi } ( k ) ,
$$

$$
C _ { \mathrm { l i n } } ^ { \mathrm { n o r m } } ( k ) = \widehat { \ell } k + \phi + 2 k \varepsilon _ { \chi } ( k ) .\tag{21}
$$

Thus,

$$
g _ { k + \Delta } ^ { \mathrm { n o r m } } - g _ { k } ^ { \mathrm { n o r m } } = \Delta C _ { \mathrm { l i n } } ^ { \mathrm { n o r m } } ( k ) + \Delta ^ { 2 } C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) .\tag{22}
$$

Moreover,

$$
| \varepsilon _ { \chi } ( k ) | \leq { \frac { \| s \| _ { 2 } } { 2 k ^ { 2 } \| a \| _ { 2 } ^ { 2 } } } , \qquad | 2 k \varepsilon _ { \chi } ( k ) | \leq { \frac { \| s \| _ { 2 } } { k \| a \| _ { 2 } ^ { 2 } } } .\tag{23}
$$

Thus the $k ^ { 2 }$ drift in (10) is replaced by a linear term and a bounded error $2 k \varepsilon _ { \chi } ( k ) = O ( k ^ { - 1 } )$ . Only the coefficient $\widehat { \ell }$ remains to be confined to an interval of width $O ( n ^ { - 1 } )$ .

Theorem 2 (NESV of INDEX under $\mathcal { A } _ { \mathrm { n o r m } } ( h ) )$ . For $B > 0$ and $h \geq 2$

$$
V _ { B } ^ { \mathrm { I N D E X } , A _ { \mathrm { n o r m } } ( h ) } ( n ) = \Theta _ { B , h } ( n ^ { - 1 } ) .\tag{24}
$$

Proof. The proof follows the same strategy as that of the INDEX bounds for the positional model in Section 4.1. The full proof is given in Appendix B. □

Compared with the original positional model, selective normalization removes one shrinking direction from the volume calculation. In $A _ { \mathrm { p o s } } ( h )$ , the two effective constraints have widths $n ^ { - \overset {  } { 2 } }$ and $n ^ { - 1 }$ in $A _ { \mathrm { n o r m } } ( h )$ , the saturation of the quadratic query channel leaves only the $n ^ { - 1 }$ constraint. The improvement from $\Theta ( n ^ { - 3 } )$ to $\Theta ( n ^ { \dot { - } 1 } )$ therefore follows from the same score-level mechanism identified by the original proof.

## 4.3 LENGTH-GENERALIZATION EXPERIMENTS

We train $A _ { \mathrm { p o s } } ( h )$ and $A _ { \mathrm { n o r m } } ( h )$ under the same protocol, using paired seeds, the same training batches, and identical initial values for all parameters shared by the two models. The main comparison uses classifier width $h = 8$ , maximum training length $N = 5 1$ , and 20 seeds.

As shown in Fig. 2, the original model degrades rapidly beyond the training range, whereas the selectively normalized model retains substantially higher accuracy. At length 511, approximately ten times the maximum training length, mean accuracy increases from 60% for $A _ { \mathrm { p o s } } ( h )$ to approximately 85% for $A _ { \mathrm { n o r m } } ( h )$ . Experiments with maximum training lengths $N \in \{ 1 1 , \dot { 2 } 1 , 3 1 , 4 1 \}$ show similar results and are reported in Section D.5.

Taken together, the coefficient analysis and the architectural intervention give two complementary forms of evidence. The original volume proof identifies the parameter combinations whose effects grow with the index k, and those combinations are associated with the behavior of trained models. Selectively normalizing the quadratic query channel removes the faster-growing contribution, enlarges the exact-solution volume from $\Theta ( n ^ { \dot { - } 3 } )$ to $\Theta ( n ^ { - 1 } )$ , and improves extrapolation in the model for which the volume calculation is carried out.

## 5 RELATED WORK

The INDEX task has been studied many times before. Bhattamishra et al. (2024) give an efficient one-layer construction in which the data appear before the indexed INDEX, as in our right-hand

![](images/02878e7ff19f52e0177f242fa87f39e10b2841028cd56fc8e72e5f46698aefe3.jpg)  
Figure 2: Length generalization on INDEX for the positional and $\mathscr { A } _ { \mathrm { n o r m } } ( h )$ models trained up to $N = 5 1$ . Shading shows 95% bootstrap confidence intervals.

INDEX task. This differs from left-hand INDEXing, where the INDEX appears before the data and the answer is produced later (Kozachinskiy et al., 2025; 2026a). One-layer constructions for related right-hand function-evaluation problems are also studied by Strobl et al. (2025).

A line of work related to our normalized model $\mathcal { A } _ { \mathrm { n o r m } }$ controls attention by normalizing query– key interactions. QKNorm $\ell _ { 2 } \cdot$ -normalizes complete query and key vectors and learns the attention scale (Henry et al., 2020), while scaled cosine attention likewise removes dependence on their magnitudes (Liu et al., 2022). Separately, approaches to length extrapolation often modify the positional contribution to attention, for example by adding distance-dependent score biases (Press et al., 2022) or by designing relative position encoding to improve attention resolution beyond the training lengths (Sun et al., 2023). Our $A _ { \mathrm { n o r m } } ( h )$ , on the other hand, does not normalize the full query or key vectors, nor the quadratic positional coordinate $j ^ { 2 }$ . Instead, it applies a fixed smooth bounded map to the index-dependent query component in the quadratic positional channel.

Another related line of work studies which functions occupy large or stable regions of transformer parameter space. Hahn & Rofin (2024) connect Boolean sensitivity to transformer norms and perturbation behavior, while Kover et al. (2026) study the parameter-space geometry of transformers computing Boolean functions. Empirical work on simplicity bias likewise suggests that the parameterto-function map favors some Boolean functions over others (Bhattamishra et al., 2023). Our volume measure tracks how volume changes with sequence length, rather than assigning a single complexity or sensitivity measure to the target task. For INDEX, such analysis not only shows that the volume decreases with longer input lengths, but also isolates the sources of such shrinkage.

## 6 DISCUSSION AND LIMITATIONS

Initialization and optimization Parameter-space volume depends on the distribution placed on the parameters. Equation (1) assumes a uniform distribution over a box of size 2B in the raw coordinates. But initialization and optimization can concentrate strongly on particular parts of parameter space, so differences in exact-solution volume might not translate exactly into differences in accuracy. Tightening the correspondence would therefore require combining the present analysis with the distributions over parameters induced by initialization and optimization.

Depth Our theoretical results concern fixed-width, one-layer, one-head transformers. This restriction can matter qualitatively. Bhattamishra et al. (2024) show examples where two-layer transformers of logarithmic size solve problems for which one-layer transformers require linear size. The parameter constraints governing length generalization may therefore change substantially with depth, and extending the volume analysis to deeper transformers is an important next step.

## 7 CONCLUSION

Parameter-space volume is useful not only for measuring how exact solutions change with input length, but also for revealing why they change. On the INDEX task, our volume analysis isolates the length-sensitive components, suggesting a modification to the model itself that improves length extrapolation. This suggests a broader use of volume analysis as a diagnostic for transformer model refinement, especially task-specific ones.

## REPRODUCIBILITY STATEMENT

The baseline architectures, parameter prior, and strict-correctness criterion are defined in Section 2. The $A _ { \mathrm { n o r m } } ( h )$ architecture is defined in Section 4.2, and the fixed-sinusoidal MAJORITY model is defined in Section C.1. The complete volume arguments, experimental protocol, checkpoint-selection rule, evaluation-bank construction, and supplementary results are presented in the appendix.

## AI USE STATEMENT

In this work, we used generative AI tools to aid literature exploration, assist in the writing of proofs (prose editing, LaTeX preparation, proofreading), create or modify scientific figures, and edit code (for experiments). We have not used generative AI tools to develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, propose or refine hypotheses, design or provide feedback on research methodology or experi ments, or assist with translation. We have reviewed all AI-assisted work and take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee. Understanding deep neural networks with rectified linear units. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2018. URL https://openreview.net/pdf?id=B1J\_rgWRW.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023. URL https://arxiv.org/abs/2309.16609.

Satwik Bhattamishra, Arkil Patel, Varun Kanade, and Phil Blunsom. Simplicity bias in transformers and their ability to learn sparse Boolean functions. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics, pp. 5767–5791, 2023. doi:10.18653/v1/2023.acllong.317.

Satwik Bhattamishra, Michael Hahn, Phil Blunsom, and Varun Kanade. Separations in the representational capabilities of transformers and recurrent architectures. In Advances in Neural Information Processing Systems, volume 37, 2024. doi:10.52202/079017-1135.

Shi Chen, Zhengjiang Lin, Yury Polyanskiy, and Philippe Rigollet. Critical attention scaling in longcontext transformers. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id=7SLtElfqCW.

David Chiang and Peter Cholak. Overcoming a theoretical limitation of self-attention. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, pp. 7654–7664, 2022. doi:10.18653/v1/2022.acl-long.527.

Ping-yeh Chiang, Renkun Ni, David Yu Miller, Arpit Bansal, Jonas Geiping, Micah Goldblum, and Tom Goldstein. Loss landscapes are all you need: Neural network generalization can be explained

without the implicit bias of gradient descent. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id= QC10RmRbZy9.

Michael Hahn and Mark Rofin. Why are sensitive functions hard for transformers? In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics, pp. 14973–15008, 2024. doi:10.18653/v1/2024.acl-long.800.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4246–4253, 2020. doi:10.18653/v1/2020.findings-emnlp.379.

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. In Advances in Neural Information Processing Systems, volume 36, pp. 24892–24928, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/4e85362c02172c0c6567ce593122d31c-Abstract-Conference.html.

Blanka Kover, Alexandra Butoi, Anej Svete, Michael Hahn, and Ryan Cotterell. Understanding the parameter space geometry of transformers encoding boolean functions. In Proceedings of the International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/ pdf/2606.08768.

Alexander Kozachinskiy, Felipe Urrutia, Hector Orellana, Tomasz Steifer, German Pizarro, Matias Fuentes, Francisco Meza Vasquez, Cristian Buc Calderon, and Cristobal Rojas. Strassen attention, split VC dimension and compositionality in transformers. In Advances in Neural Information Processing Systems, volume 38, pp. 10718–10749, 2025. doi:10.52202/085713-0327.

Alexander Kozachinskiy, Vicente Opazo, and Felipe Urrutia. Indexing: The beginning and the end, 2026a. URL https://arxiv.org/abs/2607.22361.

Alexander Kozachinskiy, Tomasz Steifer, and Przemyslaw Walega. Parity, sensitivity, and transformers, 2026b. URL https://arxiv.org/abs/2602.05896.

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin Transformer V2: Scaling up capacity and resolution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12009–12019, 2022. doi:10.1109/CVPR52688.2022.01170.

Ken M. Nakanishi. Scalable-softmax is superior for attention, 2025. URL https://arxiv.org/ abs/2501.19399.

Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id=R8sQPpGCv0.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? A survey. Transactions of the Association for Computational Linguistics, 12:543–561, 2024. doi:10.1162/tacl\_a\_00663.

Lena Strobl, Dana Angluin, and Robert Frank. Concise one-layer transformers can do function evaluation (sometimes), 2025. URL https://arxiv.org/abs/2503.22076.

Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable transformer. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14590– 14604, 2023. doi:10.18653/v1/2023.acl-long.816.

## A PROOFS OF THE BASELINE NESV BOUNDS

This appendix proves the four baseline bounds in Theorem 1.

## A.1 FIRST

The lower bound constructs fixed neighborhoods of parameters independent of input length that gives exact solutions to the task. The upper bound follows from symmetry between the two labels.

Proofofthe FIRST bound in Theorem 1. Choose $\eta = \operatorname* { m i n } \{ B / 1 6 , 1 / 1 6 \}$ and set

$$
\begin{array} { r } { a ^ { 0 } = ( 2 \eta , 0 ) , \qquad b ^ { 0 } = ( 0 , - \eta ) , \qquad r ^ { 0 } = ( \eta , 0 ) , \qquad s ^ { 0 } = ( 0 , \eta ) , } \end{array}\tag{25}
$$

with $c _ { K } ^ { 0 } = b _ { V } ^ { 0 } = 0$ and $C _ { V } ^ { 0 } = \eta I _ { 2 }$ . We call this set of $( a ^ { 0 } , b ^ { 0 } , r ^ { 0 } , s ^ { 0 } )$ the center. Put $q _ { F } : = a + b ;$ $\widehat { \ell } _ { 1 } : = q _ { F } ^ { \top } r , \widehat { \ell } _ { 2 } : = q _ { F } ^ { \top } s$ . At the center, $\widehat { \ell } _ { 1 } ^ { 0 } = 2 \eta ^ { 2 }$ and $\widehat { \ell } _ { 2 } ^ { 0 } = - \eta ^ { 2 }$ . By continuity, there is a fixed positive-volume neighborhood of $( a , b , r , s )$ on which

$$
| \widehat { \ell } _ { 1 } - 2 \eta ^ { 2 } | \leq \frac { \eta ^ { 2 } } { 8 } , \qquad | \widehat { \ell } _ { 2 } + \eta ^ { 2 } | \leq \frac { \eta ^ { 2 } } { 8 } .\tag{26}
$$

For $j = 1 + \Delta$ with $\Delta \geq 1$ , the key bias cancels and

$$
\begin{array} { l } { { \displaystyle g ( 1 ) - g ( 1 + \Delta ) = \Delta \big [ ( - \widehat { \ell } _ { 1 } - 2 \widehat { \ell } _ { 2 } ) + ( - \widehat { \ell } _ { 2 } ) \Delta \big ] } } \\ { { \displaystyle \geq \frac { \eta ^ { 2 } } { 2 } \Delta ^ { 2 } . } } \end{array}\tag{27}
$$

Consequently, throughout this fixed positional-parameter neighborhood,

$$
1 - \alpha _ { 1 } \leq \sum _ { \Delta = 1 } ^ { \infty } n ^ { - ( \eta ^ { 2 } / 2 ) \Delta ^ { 2 } } \longrightarrow 0 .\tag{28}
$$

Restrict $( C _ { V } , b _ { V } )$ to a fixed neighborhood on which

$$
\| C _ { V } e _ { \sigma } + b _ { V } - \eta e _ { \sigma } \| _ { \infty } \le \frac { \eta } { 1 6 } \qquad ( \sigma \in \{ 0 , 1 \} ) .\tag{29}
$$

For all sufficiently large n, (28) gives $\alpha _ { 1 } \geq 7 / 8 . \mathrm { I f } \ y ^ { \star } = w _ { 1 }$ and $\bar { y } ^ { \star } = 1 - y ^ { \star }$ , the total attention masses on the two symbols satisfy

$$
p _ { y ^ { \star } } - p _ { \bar { y } ^ { \star } } \geq \frac { 3 } { 4 } .\tag{30}
$$

Convexity of the value perturbation in (29) therefore gives

$$
A _ { y ^ { \star } } - A _ { \bar { y } ^ { \star } } \geq \eta \big ( p _ { y ^ { \star } } - p _ { \bar { y } ^ { \star } } \big ) - \frac { \eta } { 8 } \geq \frac { 5 \eta } { 8 } .\tag{31}
$$

Choose $W _ { 1 } ^ { 0 } , b _ { 1 } ^ { 0 } , W _ { 2 } ^ { 0 } , b _ { 2 } ^ { 0 }$ as follows. Take the first two rows of $W _ { 1 } ^ { 0 }$ to be $\eta e _ { 0 } ^ { \top }$ and $\eta e _ { 1 } ^ { \top }$ , with both corresponding biases equal to $\eta ^ { 2 }$ . Set every remaining row of $W _ { 1 } ^ { 0 }$ to zero and its bias $\mathbf { t o } - \eta$ . In $W _ { 2 } ^ { 0 }$ give output row zero coefficient η on hidden unit zero and output row one coefficient η on hidden unit one, with all other entries zero; set $b _ { 2 } ^ { 0 } = 0$ . The first two hidden units are active throughout a sufficiently small neighborhood of (29), whereas the remaining units are inactive, and

$$
y _ { 0 } - y _ { 1 } = \eta ^ { 2 } ( A _ { 0 } - A _ { 1 } ) .\tag{32}
$$

Thus (31) gives a correct-logit margin of at least that $y _ { 0 } - y _ { 1 } \ge 5 \eta ^ { 3 } / 8$ at the center. Allow $c _ { K }$ to vary over a sufficiently small fixed neighborhood of 0. This does not affect the attention weights, since $q _ { F } ^ { \top } c _ { K }$ is added to every attention score. Together with the fixed neighborhoods chosen above for $( \bar { a } , \bar { b } , r , s ) , ( \bar { C } _ { V } , b _ { V } )$ , and $( W _ { 1 } , b _ { 1 } , W _ { 2 } , b _ { 2 } )$ , this gives a set of normalized volume $c _ { F } ~ > ~ 0$ independent of n.

For the upper bound, swap the two rows of $W _ { 2 }$ and the two entries of $b _ { 2 } .$ , leaving every other coordinate unchanged. This is a measure-preserving involution of the raw parameter cube. It maps the exact solutions of FIRST to the exact solutions of the complementary task w $\mapsto 1 - w _ { 1 }$ . The two exact-solution sets are disjoint, so each occupies at most one half of the cube. In particular, the NESV is $O ( 1 )$ □

## A.2 INDEX

We use the notation from Section 4.1. Thus the attention score at position j is

$$
g _ { j } = { ( a k + b ) } ^ { \top } ( r j + s j ^ { 2 } + c _ { K } ) ,\tag{33}
$$

and

$$
\begin{array} { r } { \chi = a ^ { \top } s , \qquad \ell = 2 b ^ { \top } s + a ^ { \top } r , \qquad \delta = b ^ { \top } s , \qquad \phi = b ^ { \top } r . } \end{array}\tag{34}
$$

For $k + \Delta \in [ n ]$ , the score gap at position $k + \Delta$ is

$$
g _ { k + \Delta } - g _ { k } = \Delta C _ { \mathrm { l i n } } ( k ) + \Delta ^ { 2 } C _ { \mathrm { q u a d } } ( k ) ,\tag{35}
$$

where

$$
C _ { \mathrm { l i n } } ( k ) = 2 \chi k ^ { 2 } + \ell k + \phi , \qquad C _ { \mathrm { q u a d } } ( k ) = \chi k + \delta .\tag{36}
$$

## A.2.1 LOWER BOUND

Choose $\eta = \operatorname* { m i n } \{ B / 1 6 , 1 / 1 6 \}$ and consider the solution (which we call the center):

$$
a ^ { 0 } = ( 2 \eta , 0 ) , \qquad b ^ { 0 } = ( 0 , - \eta ) , \qquad r ^ { 0 } = ( \eta , 0 ) , \qquad s ^ { 0 } = ( 0 , \eta ) .\tag{37}
$$

At this point,

$$
\chi = \ell = \phi = 0 , \qquad \delta = - \eta ^ { 2 } ,\tag{38}
$$

and hence

$$
g _ { k } - g _ { j } = \eta ^ { 2 } ( j - k ) ^ { 2 } .\tag{39}
$$

To obtain a positive-volume family around this point, consider the local coordinate map

$$
\Psi ( a , b , r , s ) = ( \chi , \ell , a _ { 1 } , a _ { 2 } , b _ { 1 } , b _ { 2 } , r _ { 2 } , s _ { 2 } ) .\tag{40}
$$

We call $( a , b , r , s )$ the untransformed coordinates and $( \chi , \ell , a _ { 1 } , a _ { 2 } , b _ { 1 } , b _ { 2 } , r _ { 2 } , s _ { 2 } )$ the transformed coordinates. For fixed $( a , b , r _ { 2 } , s _ { 2 } )$ , the Jacobian of $( s _ { 1 } , r _ { 1 } ) \mapsto ( \chi , \ell )$ is

$$
{ \begin{array} { r l } { { \left[ \right.} a _ { 1 } \quad } & { 0 { \vphantom { { \mathrm { \tiny ~ \left( \frac { a _ { 1 } } { 1 } \right }) } } } , \quad } & { { \mathrm { \tiny ~ d e t } } = a _ { 1 } ^ { 2 } . } \end{array}  }\tag{41}
$$

The determinant is nonzero at (37). The inverse function theorem therefore gives a fixed neighborhood on which Ψ is one-to-one and the Jacobians of Ψ and $\Psi ^ { - 1 }$ are bounded. Hence there are constants $c , C > 0$ , independent of $n ,$ such that every measurable set $E$ in the transformed-coordinate neighborhood satisfies c vo $. ( E ) \leq \mathrm { v o l } { \big ( } \Psi ^ { - 1 } ( E ) { \big ) } \leq C \mathrm { v o l } ( E )$

Restrict

$$
| \chi | \leq \frac { \eta ^ { 2 } } { 6 4 n ^ { 2 } } , \qquad | \ell | \leq \frac { \eta ^ { 2 } } { 6 4 n } ,\tag{42}
$$

while restricting the remaining transformed coordinates to fixed-width intervals independent of n sufficiently close to the center that

$$
\delta \leq - \frac { 3 \eta ^ { 2 } } { 4 } , \qquad | \phi | \leq \frac { \eta ^ { 2 } } { 1 6 } .\tag{43}
$$

For every $k \leq n ,$ , these conditions imply

$$
C _ { \mathrm { q u a d } } ( k ) \leq - \frac { \eta ^ { 2 } } { 2 } , \qquad | C _ { \mathrm { l i n } } ( k ) | \leq \frac { \eta ^ { 2 } } { 4 } .\tag{44}
$$

Indeed, the contribution of $\chi k$ to $C _ { \mathrm { q u a d } } ( k )$ is at most $\eta ^ { 2 } / 6 4$ , and the three contributions to $C _ { \mathrm { l i n } } ( k )$ are bounded by $2 \eta ^ { 2 } / 6 4 , \eta ^ { 2 } / 6 4$ , and $\eta ^ { 2 } / 1 6$ , respectively. Therefore, for every nonzero integer $\Delta$ with $k + \Delta \in [ n ]$ ，

$$
g _ { k } - g _ { k + \Delta } = - \Delta C _ { \mathrm { l i n } } ( k ) - \Delta ^ { 2 } C _ { \mathrm { q u a d } } ( k )
$$

$$
\geq \frac { \eta ^ { 2 } } { 4 } \Delta ^ { 2 } .\tag{45}
$$

Under the log n attention scale, the total attention outside the indexed position is consequently at most

$$
2 \sum _ { d = 1 } ^ { \infty } n ^ { - ( \eta ^ { 2 } / 4 ) d ^ { 2 } } ,\tag{46}
$$

which tends to zero as $n \to \infty$ uniformly in the index k and over all positional parameters $( a , b , r , s )$ in the local neighborhood above satisfying (42) and (43).

Set $C _ { V } ^ { 0 } = \eta I _ { 2 } , b _ { V } ^ { 0 } = 0$ , and take the parameter values

$$
\begin{array} { c } { { W _ { 1 } ^ { 0 } = \left[ \begin{array} { c } { { \eta I _ { 2 } } } \\ { { 0 _ { ( h - 2 ) \times 2 } } } \end{array} \right] , \qquad b _ { 1 } ^ { 0 } = ( \eta ^ { 2 } , \eta ^ { 2 } , - \eta , . . . , - \eta ) ^ { \top } , } } \\ { { W _ { 2 } ^ { 0 } = \left[ \eta I _ { 2 } \quad 0 _ { 2 \times ( h - 2 ) } \right] , \qquad b _ { 2 } ^ { 0 } = 0 . } } \end{array}
$$

For all sufficiently large n, (46) makes the indexed position receive at least $3 / 4$ of the total attention. $\mathrm { I f } w _ { k } = \sigma$ , the σ coordinate of the attention output exceeds the other coordinate by at least $\eta / 2$ , even if every other token has symbol $1 - \sigma$ . With the chosen parameters, the first two hidden units are active and, for $c \in \{ 0 , 1 \}$ , the c-th unit outputs $\eta A _ { c } + \eta ^ { 2 }$ . The corresponding row of $W _ { 2 }$ multiplies this quantity by η, so $y _ { c } = \eta ^ { 2 } A _ { c } + \eta ^ { 3 }$ . The common term $\eta ^ { 3 }$ cancels between the two logits, and therefore $\begin{array} { r } { y _ { \sigma } - y _ { 1 - \sigma } = \eta ^ { 2 } \bigl ( A _ { \sigma } - A _ { 1 - \sigma } \bigr ) \geq \frac { \eta ^ { 3 } } { 2 } } \end{array}$ . This margin is uniform over all inputs and indexes, so there exists fixed neighborhoods independent of n of $\bar { C } _ { V } , b _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 } , b _ { 2 }$ that produce exact solutions. The key bias $c _ { K }$ may also vary over a fixed neighborhood because it cancels from the attention weights.

The two intervals in (42) contribute normalized widths of orders $n ^ { - 2 }$ and $n ^ { - 1 }$ , whereas all other coordinates vary over intervals of width independent of n. Therefore

$$
V _ { B } ^ { \mathrm { I N D E X } , A _ { \mathrm { p o s } } ( h ) } ( n ) = \Omega _ { B , h } ( n ^ { - 3 } ) .\tag{47}
$$

## A.2.2 UPPER BOUND

For each indexed position $k ,$ compare a constant word with a word that differs from it only at position k. These two inputs must be assigned different labels. We consider two cases. Let $m =$ $y _ { w _ { k } } ( w , k ) - y _ { 1 - w _ { k } } ( w , k )$ . In any correct solution, we will have $m > 0$ , but if $m \leq n ^ { - 5 }$ , we call the solution small-margin; otherwise, we call it large-margin.

Take any one of the inequalities $y _ { w _ { k } } ( w , k ) > y _ { 1 - w _ { k } } ( w , k ) ;$ ; there are 4n such inequalities in total. Fix arbitrary values of $a , b , r , s , c _ { K } , C _ { V } , b _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 }$ , and $b _ { 2 , 1 - w _ { k } } . \mathrm { A s } \ b _ { 2 , w _ { k } }$ varies, the margin of the inequality varies with slope one. To satisfy $0 < y _ { w _ { k } } ( w , k ) - y _ { 1 - w _ { k } } ( w , k ) \leq n ^ { - 5 } , b _ { 2 , w _ { k } }$ needs to lie in an interval of length $n ^ { - 5 }$ . Thus the set of small-margin solutions has normalized volume at most 4n $n ^ { - 5 } / ( 2 B ) = \check { O } ( n ^ { - 4 } )$

For every large-margin solution, changing only the symbol at position k from τ to σ $\neq \tau$ must change the predicted label from τ to σ. By the definition of a large-margin solution, for each of the two words w in this comparison, $y _ { w _ { k } } ( w , k ) \dot { - y _ { 1 - w _ { k } } ( w , k ) } > n ^ { - 5 }$ . Since the $y _ { \sigma } ( w , k ) - y _ { \tau } ( w , k )$ , viewed as a function of the attention output $A ( w , k )$ is Lipschitz, this is impossible if the attention assigned to position k is too small. To see this, let w be the constant-τ word and let $w ^ { \prime }$ be the word obtained by changing only position k to $\sigma .$ . Because the query and key maps read only positional coordinates, the two words have the same attention weights. Their attention outputs therefore satisfy

$$
A ( w ^ { \prime } , k ) - A ( w , k ) = \alpha _ { k } C _ { V } ( e _ { \sigma } - e _ { \tau } ) ,\tag{48}
$$

so

$$
\| A ( w ^ { \prime } , k ) - A ( w , k ) \| _ { \infty } \leq 2 B \alpha _ { k } .\tag{49}
$$

Moreover, ReLU is 1-Lipschitz, each row of $W _ { 1 }$ has $\ell _ { 1 }$ -norm at most 2B, and the difference between rows σ and τ of $W _ { 2 }$ has $\ell _ { 1 }$ -norm at most 2hB. Hence

$$
\begin{array} { r l } & { \lvert \left( y _ { \sigma } ( w ^ { \prime } , k ) - y _ { \tau } ( w ^ { \prime } , k ) \right) - \left( y _ { \sigma } ( w , k ) - y _ { \tau } ( w , k ) \right) \rvert } \\ & { \qquad \le 4 h B ^ { 2 } \lVert A ( w ^ { \prime } , k ) - A ( w , k ) \rVert _ { \infty } \le 8 h B ^ { 3 } \alpha _ { k } . } \end{array}\tag{50}
$$

The first difference is greater than $n ^ { - 5 }$ and the second is less than $- n ^ { - 5 }$ , so

$$
\alpha _ { k } \geq \frac { n ^ { - 5 } } { 4 h B ^ { 3 } } .\tag{51}
$$

Hence the indexed position receives at least an inverse polynomial amount of attention. The softmax formula in (2) then implies that $g _ { k }$ lies within a constant of the maximum score. Indeed,

$$
{ \frac { n ^ { - 5 } } { 4 h B ^ { 3 } } } \leq \alpha _ { k } = { \frac { n ^ { g _ { k } } } { \sum _ { j = 1 } ^ { n } n ^ { g _ { j } } } } \leq n ^ { g _ { k } - \operatorname* { m a x } _ { j } g _ { j } } ,\tag{52}
$$

and therefore

$$
\operatorname* { m a x } _ { j } g _ { j } - g _ { k } \leq 5 + { \frac { \log ( 4 h B ^ { 3 } ) } { \log n } } \leq M _ { 0 }\tag{53}
$$

for all sufficiently large n, where $M _ { 0 }$ is independent of n. For every interior index $2 \leq k \leq n - 1$ , comparing $g _ { k }$ with the two neighboring scores in (10) gives

$$
g _ { k + 1 } - g _ { k } = C _ { \mathrm { l i n } } ( k ) + C _ { \mathrm { q u a d } } ( k ) \leq M _ { 0 }\tag{54}
$$

and

$$
g _ { k - 1 } - g _ { k } = - C _ { \mathrm { l i n } } ( k ) + C _ { \mathrm { q u a d } } ( k ) \leq M _ { 0 } .\tag{55}
$$

Thus

$$
C _ { \mathrm { q u a d } } ( k ) + | C _ { \mathrm { l i n } } ( k ) | \leq M _ { 0 } ,\tag{56}
$$

where $M _ { 0 }$ is independent of n. Since every coordinate of $a , b , s$ lies in $\lceil - B , B \rceil$ , both $\chi = a ^ { \top } .$ s and $\delta = b ^ { \top } \varepsilon$ are bounded independently of n. Hence, for $k \le n , | C _ { \mathrm { q u a d } } ( \boldsymbol { k } ) | = | \chi \boldsymbol { k } + \delta | = O ( n )$ , and thus $| C _ { \mathrm { l i n } } ( k ) | \le M _ { 0 } + | C _ { \mathrm { q u a d } } ( k ) | = O ( n )$ . Let $M = \lfloor n / 4 \rfloor$ . Since $C _ { \mathrm { l i n } } ( k ) = 2 \chi k ^ { 2 } + \ell k + \phi .$ we have $C _ { \mathrm { l i n } } ( 3 M ) - 2 C _ { \mathrm { l i n } } ( 2 M ) + C _ { \mathrm { l i n } } ( M ) = 4 \chi M ^ { 2 }$ . Because each term on the left is $O ( n )$ and $M = \Theta ( n )$ , this gives $| \chi | = \dot { O } ( n ^ { - 1 } )$ ). Moreover, $C _ { \mathrm { l i n } } ( 2 M ) - C _ { \mathrm { l i n } } ( M ) = 6 \chi M ^ { 2 } + \ell M$ . Using $| \chi | = O ( n ^ { \ - 1 } )$ in this identity gives $| \ell | = O ( 1 )$ . These preliminary bounds make

$$
| C _ { \mathrm { q u a d } } ( k ) | = | \chi k + \delta | = O ( 1 )\tag{57}
$$

uniformly for $k \leq n$ . Equation (56) then gives $| C _ { \mathrm { l i n } } ( k ) | = O ( 1 )$ . Repeating the same argument, now using these improved bounds, gives

$$
| \chi | = O ( n ^ { - 2 } ) , \qquad | \ell | = O ( n ^ { - 1 } ) .\tag{58}
$$

We used the fact $\operatorname* { P r } \left( | a ^ { \top } s | \leq \varepsilon _ { 1 } , | 2 b ^ { \top } s + a ^ { \top } r | \leq \varepsilon _ { 2 } \right) = O _ { B } ( \varepsilon _ { 1 } \varepsilon _ { 2 } )$ whenever $0 < \varepsilon _ { 1 } \le \varepsilon _ { 2 }$ are sufficiently small.

Lemma 3 (Joint two-slab estimate). Let $a , b , r ,$ s be independent and uniform on $[ - B , B ] ^ { 2 }$ . There are constants $C _ { \mathrm { s l a b } } , \varepsilon _ { B } > 0 ,$ , depending only on $B ,$ , such that, whenever $0 < \varepsilon _ { 1 } \le \varepsilon _ { 2 } \le \varepsilon _ { B }$

$$
\operatorname* { P r } \big ( | a ^ { \top } s | \leq \varepsilon _ { 1 } , | 2 b ^ { \top } s + a ^ { \top } r | \leq \varepsilon _ { 2 } \big ) \leq C _ { \mathrm { s l a b } } \varepsilon _ { 1 } \varepsilon _ { 2 } .\tag{59}
$$

Proof. Scaling changes only the constant, so take $B = 1$ . Fix $( a , s )$ , and set

$$
\varrho = ( \| a \| _ { 2 } ^ { 2 } + \| s \| _ { 2 } ^ { 2 } ) ^ { 1 / 2 } .\tag{60}
$$

The second constraint defines a slab in $( b , r )$ with normal vector $( 2 s , a )$ . One coordinate of this vector has magnitude at least a constant multiple of $\varrho ,$ , so, after fixing the other three coordinates, the allowed fraction of the remaining coordinate is at most

$$
C \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon _ { 2 } } { \varrho } \right\} .\tag{61}
$$

To estimate the first constraint at fixed radius $\varrho .$ Write

$$
a = \varrho \cos \vartheta \widehat { a } , \qquad s = \varrho \sin \vartheta \widehat { s } , \qquad 0 \leq \vartheta \leq \frac { \pi } { 2 } ,\tag{62}
$$

where $\widehat { a } , \widehat { s } \in \mathbb { S } ^ { 1 }$ . If $\varphi$ is the angle between $\widehat { a }$ and ${ \widehat { s } } ,$ then

$$
\begin{array} { r } { | \boldsymbol { a } ^ { \top } \boldsymbol { s } | = \varrho ^ { 2 } \sin \vartheta \cos \vartheta \big | \cos \varphi \big | . } \end{array}\tag{63}
$$

For fixed $\vartheta ,$ the fraction of angle $\varphi$ satisfying the first constraint is at most

$$
C \operatorname* { m i n } \left\{ 1 , { \frac { \varepsilon _ { 1 } } { \varrho ^ { 2 } \sin \vartheta \cos \vartheta } } \right\} .\tag{64}
$$

The angular density contains the factor sin $\vartheta$ cos $\vartheta ,$ therefore integrating over ϑ bounds the angular fraction by

$$
C \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon _ { 1 } } { \varrho ^ { 2 } } \right\} .\tag{65}
$$

because when $t \leq 1 , \int _ { 0 } ^ { \pi / 2 }$ min $\left\{ \sin \vartheta \cos \vartheta , t \right\} d \vartheta = O ( t )$ . The radial volume element for $( a , s ) \in$ $\mathbb { R } ^ { 4 } \mathrm { i s }$ s proportional to $\varrho ^ { 3 } d \varrho$ . Enlarging the cube to a fixed ball and combining (61) and (65), the required probability is at most a constant times

$$
\int _ { 0 } ^ { R _ { 0 } } \varrho ^ { 3 } \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon _ { 1 } } { \varrho ^ { 2 } } \right\} \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon _ { 2 } } { \varrho } \right\} d \varrho .\tag{66}
$$

Assume $0 < \varepsilon _ { 1 } \leq \varepsilon _ { 2 } \leq 1$ . If $\varepsilon _ { 2 } \leq \sqrt { \varepsilon _ { 1 } }$ , split the integral at $\varepsilon _ { 2 }$ and $\sqrt { \varepsilon _ { 1 } }$ to obtain

$$
\begin{array} { l } { I \le \displaystyle \int _ { 0 } ^ { \varepsilon _ { 2 } } \varrho ^ { 3 } d \varrho + \varepsilon _ { 2 } \int _ { \varepsilon _ { 2 } } ^ { \sqrt { \varepsilon _ { 1 } } } \varrho ^ { 2 } d \varrho + \varepsilon _ { 1 } \varepsilon _ { 2 } \int _ { \sqrt { \varepsilon _ { 1 } } } ^ { R _ { 0 } } d \varrho } \\ { \le C \varepsilon _ { 1 } \varepsilon _ { 2 } . } \end{array}\tag{67}
$$

If $\sqrt { \varepsilon _ { 1 } } \leq \varepsilon _ { 2 }$ , split instead at $\sqrt { \varepsilon _ { 1 } }$ and $\varepsilon _ { 2 } .$ . Rescaling restores B and proves the lemma.

Applying the above lemma to $( 5 8 )$ , the large-margin solutions have normalized volume $O ( n ^ { - 3 } )$ : the entries of $a , b , r ,$ s are independent parameter coordinates, and all remaining coordinates integrate out. Adding the $\overset { \cdot } { O } ( n ^ { - 4 } )$ small-margin volume gives

$$
V _ { B } ^ { \mathrm { I N D E X } , A _ { \mathrm { p o s } } ( h ) } ( n ) = O _ { B , h } ( n ^ { - 3 } ) .\tag{68}
$$

Together with (47), this proves the INDEX claim in Theorem 1.

## A.3 MAJORITY

In the position-free model $A _ { \mathrm { g e n } }$ , every input with the same number of ones produces the same attention output. The proof therefore reduces MAJORITY to a one-dimensional sequence indexed by the count and measures how much of the parameter space places the decision boundary between the two central counts. Let

$$
t = \sum _ { j = 1 } ^ { n } w _ { j } , \qquad v _ { b } = V ( e _ { b } ) , \qquad q = Q ( 0 _ { 2 } ) ,\tag{69}
$$

and define the query–key score contrast

$$
X = q ^ { \top } W _ { K } ( e _ { 1 } - e _ { 0 } ) , \qquad \Gamma = \exp ( ( \log n ) X ) .\tag{70}
$$

All zero tokens share one score and all one tokens share another, so the total attention mass on the one tokens and the attention output are exactly

$$
\rho _ { t } = \frac { t \Gamma } { n - t + t \Gamma } , \qquad A _ { t } = v _ { 0 } + \rho _ { t } \bigl ( v _ { 1 } - v _ { 0 } \bigr ) .\tag{71}
$$

Thus the $2 ^ { n }$ inputs reduce to $n { \mathrel { + { 1 } } }$ possible count levels. Throughout this subsection, write $n = 2 m + 1$ For an attention scale $\beta > 0$ and a score contrast x, let

$$
\Delta \rho _ { n , \beta } ( x ) = \rho _ { m + 1 } - \rho _ { m } ,\tag{72}
$$

where $\Gamma = e ^ { \beta x }$ . The counts m and $m + 1$ are the two adjacent levels at which the majority label changes.

Lemma 4 (Central attention gap). For $\Gamma = e ^ { \beta x }$

$$
\Delta \rho _ { n , \beta } ( x ) = \frac { n e ^ { \beta x } } { ( m + 1 + m e ^ { \beta x } ) ( m + ( m + 1 ) e ^ { \beta x } ) }\tag{73}
$$

$$
= \overline { { n ^ { 2 } + ( n ^ { 2 } - 1 ) \sinh ^ { 2 } ( \beta x / 2 ) } } .\tag{74}
$$

Moreover,

$$
0 < \Delta \rho _ { n , \beta } ( x ) \leq \frac { 1 } { n } , \quad \quad \int _ { \mathbb { R } } \Delta \rho _ { n , \beta } ( x ) d x = \frac { 4 \operatorname { a r t a n h } ( 1 / n ) } { \beta } .\tag{75}
$$

Proof. Substitute $t = m , m + 1$ into (71) and subtract to obtain (73). Dividing its denominator by $e ^ { \beta x }$ and using $n = 2 m + 1$ gives (74). The maximum is attained at $x = 0$ . For the integral, set $z = \operatorname { t a n h } ( \beta x / 2 )$ . Then

$$
d x = \frac { 2 d z } { \beta ( 1 - z ^ { 2 } ) } , \qquad \Delta \rho _ { n , \beta } ( x ) = \frac { n ( 1 - z ^ { 2 } ) } { n ^ { 2 } - z ^ { 2 } } .\tag{76}
$$

Therefore

$$
\int _ { \mathbb { R } } \Delta \rho _ { n , \beta } ( x ) d x = \frac { 2 n } { \beta } \int _ { - 1 } ^ { 1 } \frac { d z } { n ^ { 2 } - z ^ { 2 } } = \frac { 4 \operatorname { a r t a n h } ( 1 / n ) } { \beta } .\tag{77}
$$

For fixed values of all parameters except $b _ { 2 , 1 }$ , exact correctness requires $b _ { 2 , 1 }$ to place the decision threshold between the two central count levels. Define the bias-free logit difference

$$
\begin{array} { r } { H ( z ) = ( e _ { 1 } - e _ { 0 } ) ^ { \top } W _ { 2 } \operatorname { R e L U } ( W _ { 1 } z + b _ { 1 } ) . } \end{array}\tag{78}
$$

Lemma 5 (Output-bias fiber). Condition on every raw parameter except $b _ { 2 , 1 }$ . The set of values of $b _ { 2 , \cdot }$ that can extend the conditioned coordinates to an exact MAJORITY solution has length at most

$$
8 h B ^ { 3 } \Delta \rho _ { n , \beta } ( X ) .\tag{79}
$$

Proof. Correctness on the count-m and count-(m + 1) levels requires

$$
H ( A _ { m } ) + b _ { 2 , 1 } - b _ { 2 , 0 } < 0 ,\tag{80}
$$

$$
H ( A _ { m + 1 } ) + b _ { 2 , 1 } - b _ { 2 , 0 } > 0 .\tag{81}
$$

Thus $b _ { 2 , 1 }$ must lie in

$$
\big ( b _ { 2 , 0 } - H ( A _ { m + 1 } ) , b _ { 2 , 0 } - H ( A _ { m } ) \big ) ,\tag{82}
$$

whose length is $H ( A _ { m + 1 } ) - H ( A _ { m } )$ when this difference is positive. ReLU is 1-Lipschitz, each row of $W _ { 1 }$ has $\ell _ { 1 }$ -norm at most $2 B$ , and the difference between the two rows of $W _ { 2 }$ has $\ell _ { 1 }$ -norm at most 2hB. Hence H is $4 h B ^ { 2 } { - } \mathrm { I }$ Lipschitz in $\ell _ { \infty }$ . Moreover, $A _ { m + 1 } - A _ { m } = \Delta \rho _ { n , \mathrm { l o g } n } ( X ) ( v _ { 1 } - v _ { 0 } )$ , so

$$
\| A _ { m + 1 } - A _ { m } \| _ { \infty } \leq 2 B \Delta \rho _ { n , \beta } ( X ) .\tag{83}
$$

Multiplying the two bounds proves the claim.

To average the fiber length over the query and key parameters, we use a uniform density bound for X. Lemma 6 (Density of the score contrast). Under the uniform raw-cube measure, the random variable $X = q ^ { \top } W _ { K } ( e _ { 1 } - e _ { 0 } )$ has a density $f _ { X }$ satisfying

$$
\| f _ { X } \| _ { \infty } \leq B ^ { - 2 } .\tag{84}
$$

Proof. Write $\Delta _ { K } = W _ { K } ( e _ { 1 } - e _ { 0 } )$ . Each coordinate of $\Delta _ { K }$ is the difference of two independent uniform variables and has triangular density bounded by $( 2 B ) ^ { - 1 }$ . Conditional on $q _ { \mathrm { a g g } } \neq 0$ , choose a coordinate with magnitude $\| q _ { \mathrm { a g g } } \| _ { \infty }$ and condition on the other coordinate of $\Delta _ { K }$ . Scaling gives

$$
\| f _ { X | q _ { \mathrm { a g g } } } \| _ { \infty } \le \frac { 1 } { 2 B \| q _ { \mathrm { a g g } } \| _ { \infty } } .\tag{85}
$$

Since $q _ { \mathrm { a g g } } = Q ( 0 ) = b _ { Q }$ is uniform on $[ - B , B ] ^ { 2 }$

$$
\mathbb { E } \big [ \| q _ { \mathrm { a g g } } \| _ { \infty } ^ { - 1 } \big ] = \frac { 2 } { B } .\tag{86}
$$

Averaging proves the result.

By Fubini’s theorem and Lemmas 4 to 6,

$$
\begin{array} { r l r } {  { V _ { B } ^ { \mathrm { M A J o R I T Y } , \mathcal { A } _ { \mathrm { g e n } } ( h ) } ( n ) \leq 4 h B ^ { 2 } \mathbb { E } \big [ \Delta \rho _ { n , \log n } ( X ) \big ] } } \\ & { } & { \leq 4 h B ^ { 2 } \| f _ { X } \| _ { \infty } \int _ { \mathbb { R } } \Delta \rho _ { n , \log n } ( x ) d x } \\ & { } & { \leq \frac { 1 6 h \mathrm { a r t a n h } ( 1 / n ) } { \log n } . } \end{array}\tag{87}
$$

Since artanh $( 1 / n ) \leq n / ( n ^ { 2 } - 1 )$ , this is $O _ { h } ( ( n \log n ) ^ { - 1 } )$

For the matching lower bound, we need a positive mass of score contrasts on which the central gap is of order $1 / n$ and a fixed-volume set of the remaining parameters that turns this gap into a correct bias interval.

Lemma 7 (Lower contrast mass). There is a constant $c _ { X } = c _ { X } ( B ) > 0$ such that,for all sufficiently large n,

$$
\operatorname* { P r } { \biggl ( } | X | \leq { \frac { 1 } { \log n } } { \biggr ) } \geq { \frac { c _ { X } } { \log n } } .\tag{88}
$$

Proof. Write $\Delta _ { K } = W _ { K } ( e _ { 1 } - e _ { 0 } )$ and $q _ { \mathrm { a g g } } = ( q _ { 1 } , q _ { 2 } )$ . Restrict to

$$
q _ { 1 } \in [ B / 2 , B ] , \qquad | q _ { 2 } | \leq B / 4 , \qquad | \Delta _ { K , 2 } | \leq B / 2 .\tag{89}
$$

These independent events have probabilities at least $1 / 4 , 1 / 4$ , and $3 / 8$ . Conditional on them, the inequality

$$
| q _ { 1 } \Delta _ { K , 1 } + q _ { 2 } \Delta _ { K , 2 } | \leq \varepsilon\tag{90}
$$

asks that $\Delta _ { K , 1 }$ lie in an interval of length at least $2 \varepsilon / B _ { ; }$ , centered in $[ - B / 4 , B / 4 ] . \mathrm { I f } \varepsilon \leq B ^ { 2 } / 8 .$ , the interval lies in $\left[ - B / 2 , B / 2 \right]$ , where the triangular density of $\Delta _ { K , 1 }$ is at least $3 / ( { \bar { 8 } } B )$ . Hence

$$
\operatorname* { P r } ( | X | \leq \varepsilon ) \geq { \frac { 9 \varepsilon } { 5 1 2 B ^ { 2 } } } .\tag{91}
$$

Taking $\varepsilon = 1 /$ log n proves the lemma.

Lemma 8 (Positive-volume monotone decoder family). There are constants $c _ { \Omega } , \kappa > 0$ and a set of values of $( W _ { V } , b _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 } , b _ { 2 , 0 } )$ ofnormalized volume at least $c _ { \Omega }$ such that the bias-free logit difference along $v _ { 0 } + p ( v _ { 1 } - v _ { 0 } )$ is affine with derivative at least κ for all $p \in [ 0 , 1 ]$ . For every parameter choice in this set, $b _ { 2 , 1 }$ may vary over an interval of length at least

$$
\kappa ( \rho _ { m + 1 } - \rho _ { m } )\tag{92}
$$

while preserving exact MAJORITY correctness.

Proof. Let $e _ { V } = ( 1 , 0 ) ^ { \top }$ and choose $W _ { V } , b _ { V }$ so that

$$
v _ { 0 } = 0 , \qquad v _ { 1 } = \eta e _ { V } .\tag{93}
$$

Give one hidden unit first-layer row $\eta e _ { V } ^ { \top }$ and bias η, make every other hidden unit strictly inactive, and set the difference between output rows on the active hidden unit to η. Set the other output-row differences and $b _ { 2 , 0 }$ to zero. Along the value segment, define

$$
\psi ( p ) = H { \big ( } v _ { 0 } + p ( v _ { 1 } - v _ { 0 } ) { \big ) } .\tag{94}
$$

For these parameter values,

$$
\psi ( p ) = \eta ^ { 2 } + \eta ^ { 3 } p .\tag{95}
$$

The active pre-activation lies in $[ \eta , \eta + \eta ^ { 2 } ]$ , while every other pre-activation equals −η. A fixed open raw-coordinate box around these parameter values therefore preserves the activation pattern, keeps ψ affine with derivative at least $\eta ^ { 3 } / \bar { 2 }$ , and keeps $b _ { 2 , 0 } - \psi$ inside $( - B , B )$ . Take $\kappa = \eta ^ { 3 } / 2$ and let $c _ { \Omega }$ be the normalized volume of this box. For parameters in the box, the interval

$$
\left( b _ { 2 , 0 } - \psi ( \rho _ { m + 1 } ) , b _ { 2 , 0 } - \psi ( \rho _ { m } ) \right)\tag{96}
$$

has length at least $\kappa ( \rho _ { m + 1 } - \rho _ { m } )$ and lies in the raw bias range. Since both $\rho _ { t }$ and $\psi$ are increasing, every $t \leq m$ gives a negative logit difference and every $t \geq m + 1$ gives a positive one. Thus all count levels are classified correctly. □

On the event in (88), | log n $X | \ \leq \ 1$ , so (74) gives $\begin{array} { r } { \rho _ { m + 1 } - \rho _ { m } \geq \frac { 1 } { n \left( 1 + \sinh ^ { 2 } ( 1 / 2 ) \right) } = \Omega ( n ^ { - 1 } ) } \end{array}$ Restrict the $( W _ { V } , b _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 } , b _ { 2 , 0 } )$ to the set in Lemma 8. The remaining bias interval then has normalized length $\Omega ( 1 / n )$ . Multiplying this length by the $\Omega ( 1 / \log n )$ contrast event and the fixed parameter box volume gives

$$
V _ { B } ^ { \mathrm { M A J O R I T Y } , A _ { \mathrm { g e n } } ( h ) } ( n ) \geq \frac { c _ { - } } { n \log n } .\tag{97}
$$

Together with (87), this proves the MAJORITY claim in Theorem 1.

## A.4 PARITY

For PARITY, the count reduction in (71) gives a sequence of points along the value segment on which the required label flips. The proof compares the resulting number of sign changes with the number of activation changes available to a one-hidden-layer ReLU classifier. $\mathrm { I f } \ v _ { 0 } \ne v _ { 1 }$ , then

$$
0 = \rho _ { 0 } < \rho _ { 1 } < \cdot \cdot \cdot < \rho _ { n } = 1 .\tag{98}
$$

Since every count-t attention output has the form $A _ { t } = v _ { 0 } + \rho _ { t } ( v _ { 1 } - v _ { 0 } )$ , all such outputs lie on the line segment joining $v _ { 0 }$ and $v _ { 1 }$ . Parameterize this segment by $A ( p ) = v _ { 0 } + p ( v _ { 1 } - v _ { 0 } ) , p \in [ 0 , 1 ]$ and define the difference between the two output logits along it by

$$
f ( p ) = ( e _ { 1 } - e _ { 0 } ) ^ { \top } \big [ W _ { 2 } \mathrm { R e L U } ( W _ { 1 } A ( p ) + b _ { 1 } ) + b _ { 2 } \big ] .\tag{99}
$$

This is a continuous piecewise-affine function of $p ,$ and each hidden preactivation can change sign at most once.

Proofofthe PARITY bound in Theorem 1. If $v _ { 0 } ~ = ~ v _ { 1 }$ , every count level has the same attention output, so exact parity is impossible. Otherwise, exact parity makes $f ( \rho _ { t } )$ alternate in sign. By continuity, for each $t = 0 , \ldots , n - 1$ , choose a zero $z _ { t } \in ( \rho _ { t } , \rho _ { t + 1 } )$ . For each $t = 1 , \ldots , n - 1$ , the interval $\left( z _ { t - 1 } , z _ { t } \right)$ contains $\rho _ { t }$ , where $f$ is nonzero. If no hidden pre-activation changed sign in this interval, the activation pattern would be fixed and $f$ would be affine there. An affine function that vanishes at both endpoints is identically zero, contradicting $f ( \rho _ { t } ) \neq 0$ . Hence each of the $n - 1$ disjoint intervals $\left( { z _ { t - 1 } , z _ { t } } \right)$ contains a point at which a hidden ReLU pre-activation changes sign. Since each hidden pre-activation changes sign at most once, this requires $h \geq n - 1$ . Therefore the exact-solution set is empty whenever $n > h + 1$ □

## B PROOF OF THE $A _ { \mathrm { n o r m } } ( h )$ INDEX BOUND

This appendix proves Theorem 2. Recall the fixed map

$$
F ( v ) = \frac { v } { \sqrt { 1 + \| v \| _ { 2 } ^ { 2 } } }\tag{100}
$$

and the $A _ { \mathrm { n o r m } } ( h )$ positional score

$$
g _ { j } ^ { \mathrm { n o r m } } = ( a k + b ) ^ { \top } ( r j + c _ { K } ) + \left( b + F ( a k ) \right) ^ { \top } s j ^ { 2 } .\tag{101}
$$

$$
\widehat { \boldsymbol { \chi } } = \widehat { \boldsymbol { a } } ^ { \intercal } \boldsymbol { s } , \qquad \widehat { \boldsymbol { \ell } } = \boldsymbol { \ell } + 2 \widehat { \boldsymbol { \chi } } , \qquad \varepsilon _ { \boldsymbol { \chi } } ( \boldsymbol { k } ) = \big ( \boldsymbol { F } ( \boldsymbol { a } \boldsymbol { k } ) - \widehat { \boldsymbol { a } } \big ) ^ { \intercal } \boldsymbol { s } ,\tag{102}
$$

with $\delta = b ^ { \top }$ s and $\phi = b ^ { \top } r$ as in (9). Then,

$$
C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) = \widehat { \chi } + \delta + \varepsilon _ { \chi } ( k ) , \qquad C _ { \mathrm { l i n } } ^ { \mathrm { n o r m } } ( k ) = \widehat { \ell } k + \phi + 2 k \varepsilon _ { \chi } ( k ) ,\tag{103}
$$

Whenever $k + \Delta \in [ n ]$

$$
g _ { k + \Delta } ^ { \mathrm { n o r m } } - g _ { k } ^ { \mathrm { n o r m } } = \Delta C _ { \mathrm { l i n } } ^ { \mathrm { n o r m } } ( k ) + \Delta ^ { 2 } C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) .\tag{104}
$$

Since $F ( a k )$ is collinear with a,

$$
F ( a k ) = \lambda _ { a } ( k ) \widehat { a } , \qquad \lambda _ { a } ( k ) = \frac { k \| a \| _ { 2 } } { \sqrt { 1 + k ^ { 2 } \| a \| _ { 2 } ^ { 2 } } } .\tag{105}
$$

Moreover,

$$
0 \leq 1 - \lambda _ { a } ( k ) \leq { \frac { 1 } { 2 k ^ { 2 } \| a \| _ { 2 } ^ { 2 } } } , \qquad | 2 k \varepsilon _ { \chi } ( k ) | \leq { \frac { \| s \| _ { 2 } } { k \| a \| _ { 2 } ^ { 2 } } } .\tag{106}
$$

## B.1 LOWER BOUND

Use the same positional center as in (37). At this point, $\widehat { a } ^ { 0 } = ( 1 , 0 )$ is orthogonal to $s ^ { 0 } = ( 0 , \eta )$ , and hence

$$
\widehat { \ell } = \phi = 0 , \qquad Q _ { a ^ { 0 } } ( k ) = - \eta ^ { 2 } \qquad ( k \geq 1 ) .\tag{107}
$$

Thus

$$
C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) = - \eta ^ { 2 } , \qquad C _ { \mathrm { l i n } } ^ { \mathrm { n o r m } } ( k ) = 0 ,\tag{108}
$$

and therefore,

$$
g _ { k } ^ { \mathrm { n o r m } } - g _ { j } ^ { \mathrm { n o r m } } = \eta ^ { 2 } ( j - k ) ^ { 2 } .\tag{109}
$$

Work in a fixed neighborhood of this center on which $a _ { 1 } > \eta .$ The map $a \mapsto { \widehat { a } }$ is smooth there. Replace the raw coordinate $r _ { 1 }$ by ${ \widehat { \ell } } ,$ keeping $( a , b , r _ { 2 } , s )$ as the other coordinates. Since

$$
\frac { \partial \widehat { \ell } } { \partial r _ { 1 } } = a _ { 1 } ,\tag{110}
$$

the corresponding coordinate change is locally invertible, with Jacobian and inverse Jacobian bounded on a sufficiently small fixed neighborhood. Thus boxes in these transformed coordinates and their preimages in the original coordinates have volumes within fixed multiplicative constants, independent of $n .$ . Shrink the fixed intervals for the other transformed coordinates so that throughout the neighborhood

$$
\| a \| _ { 2 } \geq \eta , \qquad b ^ { \top } s \leq - \frac { 7 \eta ^ { 2 } } { 8 } , \qquad | \widehat { a } ^ { \top } s | \leq \operatorname* { m i n } \left\{ \frac { \eta ^ { 2 } } { 8 } , \frac { \eta ^ { 4 } } { 1 6 } \right\} , \qquad | \phi | \leq \frac { \eta ^ { 2 } } { 1 6 } .\tag{111}
$$

By (105), for every $k \geq 1$

$$
C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) = \delta + \lambda _ { a } ( k ) \widehat \chi \leq - \frac { 3 \eta ^ { 2 } } { 4 } .\tag{112}
$$

Furthermore, (106) and (111) give

$$
| 2 k \varepsilon _ { \chi } ( k ) | \leq \frac { | \widehat { a } ^ { \top } s | } { \| a \| _ { 2 } ^ { 2 } } \leq \frac { \eta ^ { 2 } } { 1 6 } \qquad ( k \geq 1 ) .\tag{113}
$$

Here we used that $2 k \varepsilon _ { \chi } ( k ) = 2 k ( \lambda _ { a } ( k ) - 1 ) \widehat { a } ^ { \top } s$

Now restrict the only shrinking transformed coordinate by

$$
\vert \widehat { \ell } \vert \leq \frac { \eta ^ { 2 } } { 1 6 n } .\tag{114}
$$

For every $k \leq n ,$ equations (111), (113), and (114) imply

$$
| \widehat { \ell } k + \phi + 2 k \varepsilon _ { \chi } ( k ) | \leq \frac { 3 \eta ^ { 2 } } { 1 6 } .\tag{115}
$$

Therefore, for every nonzero integer $\Delta$ with $k + \Delta \in [ n ]$

$$
\begin{array} { r l r } {  { g _ { k } ^ { \mathrm { n o r m } } - g _ { k + \Delta } ^ { \mathrm { n o r m } } \geq \frac { 3 \eta ^ { 2 } } { 4 } \Delta ^ { 2 } - \frac { 3 \eta ^ { 2 } } { 1 6 } | \Delta | } } \\ & { } & { \geq \frac { \eta ^ { 2 } } { 2 } \Delta ^ { 2 } . } \end{array}\tag{116}
$$

Under the log n attention scale, the total attention outside position k is at most

$$
2 \sum _ { d = 1 } ^ { \infty } n ^ { - ( \eta ^ { 2 } / 2 ) d ^ { 2 } } ,\tag{117}
$$

which tends to zero uniformly in k and in the parameters in the neighborhood above. Thus the same fixed neighborhoods of $C _ { V } , \dot { b } _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 } , \dot { b } _ { 2 }$ used in the baseline INDEX lower bound preserve a positive logit difference for the correct label, and $c _ { K }$ may vary over a fixed neighborhood because it cancels from the attention weights. The transformed coordinate $\widehat { \ell }$ varies over an interval of width $\Theta ( n ^ { - 1 } )$ , whereas $a , b , r _ { 2 } , s , c _ { K } , C _ { V } , b _ { V } , W _ { 1 } , b _ { 1 } , W _ { 2 } , b _ { 2 }$ vary over fixed-width intervals. Boundedness of the local Jacobian then gives

$$
V _ { B } ^ { \mathrm { I N D E X } , A _ { \mathrm { n o r m } } ( h ) } ( n ) = \Omega _ { B , h } ( n ^ { - 1 } ) .\tag{118}
$$

## B.2 UPPER BOUND

Use the same 4n inequalities $y _ { w _ { k } } ( w , k ) > y _ { 1 - w _ { k } } ( w , k )$ as in the original INDEX upper bound. The same output-bias argument shows that exact solutions for which one of these margins lies in $( 0 , n ^ { - 5 } ]$ have normalized volume $O _ { B } ( n ^ { - 4 } )$ . For every remaining exact solution, the indexed position receives at least $c _ { 0 } n ^ { - 5 }$ attention in each constant-word/single-change comparison. Hence its score lies within a constant $M _ { 0 }$ of the maximum score, uniformly in n. Comparing the indexed position with its two neighbors in (104) gives

$$
C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) + | C _ { \mathrm { l i n } } ^ { \mathrm { n o r m } } ( k ) | \le M _ { 0 } \qquad ( 2 \le k \le n - 1 ) .\tag{119}
$$

Since $\| F ( a k ) \| _ { 2 } < 1$ and the parameters are bounded, there exists a constant $C _ { 0 } = C _ { 0 } ( B )$ such that

$$
\begin{array} { r } { | C _ { \mathrm { q u a d } } ^ { \mathrm { n o r m } } ( k ) | = | \big ( b + F ( a k ) \big ) ^ { \top } s | \le C _ { 0 } \qquad ( a , b , s \in [ - B , B ] ^ { 2 } , k \ge 1 ) . } \end{array}\tag{120}
$$

It follows from (119) that

$$
| \widehat { \ell } k + \phi + 2 k \varepsilon _ { \chi } ( k ) | \leq M _ { 1 } \qquad ( 2 \leq k \leq n - 1 )\tag{121}
$$

for another constant $M _ { 1 } = M _ { 1 } ( B , h )$

Split the raw parameter cube according to whether

$$
\| a \| _ { 2 } \leq n ^ { - 1 / 2 } .\tag{122}
$$

Because a consists of two independent raw coordinates, the normalized volume of this part is at most the area of $\mathrm { a ~ r a d i u s } { - } n ^ { - 1 / 2 }$ disk divided by $( 2 B ) ^ { 2 }$ , and hence is $O _ { B } ( n ^ { - 1 } )$ . On its complement, take $k _ { 0 } = \lfloor n / 2 \rfloor$ . For all sufficiently large $n , k _ { 0 }$ is interior and $k _ { 0 } \geq n / 3 . \mathrm { B y }$ (106),

$$
| 2 k \varepsilon _ { \chi } ( k _ { 0 } ) | \leq \frac { \| s \| _ { 2 } } { k _ { 0 } \| a \| _ { 2 } ^ { 2 } } = O _ { B } ( 1 )\tag{123}
$$

because $\| a \| _ { 2 } > n ^ { - 1 / 2 }$ . Boundedness also gives $| \phi | = O _ { B } ( 1 )$ . Equation (121) at $k _ { 0 }$ therefore forces

$$
{ \widehat { \ell ! } } \leq { \frac { C } { n } }\tag{124}
$$

for $C = C ( B , h )$

Conditional on $( a , b , s )$

$$
\widehat { \ell } = a ^ { \top } r + \big ( 2 b ^ { \top } s + 2 \widehat { a } ^ { \top } s \big ) ,\tag{125}
$$

so (124) defines a slab in the independent variable r with normal $a .$ Choose a coordinate i with $| a _ { i } | \geq \| a \| _ { 2 } / \sqrt { 2 }$ and fix the other coordinate of $r .$ The allowed values of $r _ { i }$ lie in an interval of length at most $C / ( n \| a \| _ { 2 } )$ , up to a constant depending only on B and $h .$ . Thus the normalized volume is at most

$$
C _ { B } \operatorname* { m i n } \left\{ 1 , { \frac { 1 } { n \| a \| _ { 2 } } } \right\} .\tag{126}
$$

Enlarging the square for a to a fixed-radius disk and integrating in polar coordinates on the complement of (122) gives

$$
\begin{array} { c } { { \displaystyle \int _ { a \in [ - B , B ] ^ { 2 } } \frac { C _ { B } } { n \| a \| _ { 2 } } d a \leq \frac { C _ { B } ^ { \prime } } { n } \int _ { n ^ { - 1 / 2 } } ^ { \sqrt { 2 } B } d \varrho } } \\ { { \displaystyle = O _ { B } ( n ^ { - 1 } ) . } } \end{array}\tag{127}
$$

All remaining raw coordinates integrate out. Adding the $O ( n ^ { - 1 } )$ ) small-a region and the $O ( n ^ { - 4 } )$ small-margin region gives

$$
V _ { B } ^ { \mathrm { I N D E X } , A _ { \mathrm { n o r m } } ( h ) } ( n ) = O _ { B , h } ( n ^ { - 1 } ) .
$$

Together with (118), this proves Theorem 2.

(128)

## C FIXED SINUSOIDAL ENCODING ON MAJORITY

This appendix analyzes the fixed-frequency sinusoidal MAJORITY model used in the discussion.

## C.1 MODEL AND MATCHED BOUND

Let $n = 2 m + 1$ be odd, let $p _ { j } = ( \sin j , \cos j ) ^ { \top }$ , and define

$$
\begin{array} { r } { x _ { j } ^ { \mathrm { s i n } } = \left[ { e _ { w _ { j } } } \right] \in \mathbb { R } ^ { 4 } , \qquad z ^ { \mathrm { s i n } } = 0 _ { 4 } . } \end{array}\tag{129}
$$

The frequency is fixed at one and is not a trainable coordinate. The query, key, and value maps are unrestricted affine maps from $\mathbb { R } ^ { 4 } : \mathbb { R } ^ { 2 }$ , the attention multiplier is $\beta _ { n } =$ log n, and the output is the same width-h one-hidden-layer ReLU classifier as in Section 2. We denote this architecture by $\mathcal { A } _ { \mathrm { s i n } } ( h )$ . All scalar parameters are bounded in $[ - B , B ]$

Theorem 9 (Fixed-sinusoidal MAJORITY volume). For everyfixed $B > 0$ and $h \geq 2 ,$

$$
V _ { B } ^ { \mathrm { M A J O R I T Y } , A _ { \mathrm { s i n } , \mathrm { g e n } } ( h ) } ( n ) = \Theta ( \frac { 1 } { n ^ { 3 } \log n } ) .\tag{130}
$$

Write the columns of $W _ { K }$ and $W _ { V }$ as

$$
W _ { K } = [ k _ { 0 } k _ { 1 } k _ { \mathrm { s i n } } k _ { \mathrm { c o s } } ] , \qquad W _ { V } = [ v _ { 0 } v _ { 1 } v _ { \mathrm { s i n } } v _ { \mathrm { c o s } } ] ,\tag{131}
$$

and define

$$
q = b _ { Q } ,
$$

$$
\lambda = \beta _ { n } q ^ { \top } ( k _ { 1 } - k _ { 0 } ) , \qquad \Gamma = e ^ { \lambda } ,
$$

$$
\begin{array} { c c c } { \displaystyle { a = \beta _ { n } \left[ q ^ { \top } k _ { \mathrm { s i n } } \right] , \qquad } } & { \qquad } & { \boldsymbol { P } _ { j } = e ^ { { a } ^ { \top } p _ { j } } , \qquad } & { \qquad T = \sum _ { j = 1 } ^ { n } P _ { j } , } \end{array}
$$

$$
v _ { b } = v _ { b } + b _ { V } \quad ( b \in \{ 0 , 1 \} ) , \qquad C _ { V } = [ v _ { \mathrm { s i n } } \ v _ { \mathrm { c o s } } ] , \qquad \mu = \frac { 1 } { T } \sum _ { j = 1 } ^ { n } P _ { j } p _ { j } .\tag{132}
$$

The common key-bias and symbol-zero score contributions cancel, so the attention weights are

$$
\alpha _ { j } ( w ) = \frac { P _ { j } \Gamma ^ { w _ { j } } } { \sum _ { i = 1 } ^ { n } P _ { i } \Gamma ^ { w _ { i } } } .\tag{133}
$$

For $t \in [ 0 , 1 ] ^ { n }$ , introduce the algebraic extensions

$$
{ \mathsf { D } } ( t ) = T + ( \Gamma - 1 ) \sum _ { j } P _ { j } t _ { j } ,\tag{134}
$$

$$
\mathsf { N } ( t ) = \sum _ { j } P _ { j } ( v _ { 0 } + C _ { V } p _ { j } ) + \sum _ { j } t _ { j } P _ { j } \left[ \Gamma v _ { 1 } - v _ { 0 } + ( \Gamma - 1 ) C _ { V } p _ { j } \right] .\tag{135}
$$

Then $\mathsf { D } ( t ) > 0$ on the cube and $A ( w ) = \mathsf { N } ( w ) / \mathsf { D } ( w )$ on binary inputs.

## C.2 THE NECESSARY POSITIONAL COMPENSATION

Let $d _ { i } = ( W _ { 2 } ) _ { 1 i } - ( W _ { 2 } ) _ { 0 i }$ and $b = b _ { 2 , 1 } - b _ { 2 , 0 }$ . After multiplying the logit difference by the positive denominator $\mathsf { D } ( t )$ , the resulting function is continuous and piecewise affine in t. For an activation pattern $\sigma \subseteq [ h ]$ ], define

$$
\begin{array} { l l } { { \ell _ { \sigma } = \displaystyle \sum _ { i \in \sigma } d _ { i } ( W _ { 1 } ) _ { i , : } ^ { \top } , ~ } } & { { ~ c _ { \sigma } = b + \displaystyle \sum _ { i \in \sigma } d _ { i } b _ { 1 , i } , } } \\ { { { } } } & { { { } } } \\ { { \delta _ { \sigma } = \ell _ { \sigma } ^ { \top } ( v _ { 1 } - v _ { 0 } ) , ~ } } & { { ~ \xi _ { \sigma } = C _ { V } ^ { \top } \ell _ { \sigma } , } } \end{array}\tag{136}
$$

and

$$
c _ { \sigma } ^ { * } = - \ell _ { \sigma } ^ { \top } \frac { v _ { 0 } + \Gamma v _ { 1 } } { 1 + \Gamma } - \xi _ { \sigma } ^ { \top } \mu .\tag{137}
$$

Lemma 10 (Positional compensation). Outside a set ofraw-cube measure zero, every exact solution at a sufficiently large odd length has an activation pattern $\sigma \subseteq [ h ]$ such that $\delta _ { \sigma } \neq 0$ and

$$
\left. a + \frac { \sinh \lambda } { \delta _ { \sigma } } \xi _ { \sigma } \right. _ { 2 } \leq \frac { C } { n } ,\tag{138}
$$

while

$$
| c _ { \sigma } - c _ { \sigma } ^ { * } | \leq \frac { C | \delta _ { \sigma } | } { n } \frac { \Gamma } { ( 1 + \Gamma ) ^ { 2 } } .\tag{139}
$$

The constants are uniform over the bounded raw parameter cube, including when q and $\delta _ { \sigma }$ are arbitrarily small but nonzero.

Proof. Exclude first the finite union of sets on which one of the affine pieces defined below vanishes at $t = 0 ;$ each is null by slicing along $b _ { 2 , 1 }$ . On activation pattern $\sigma ,$ the denominator-scaled logit difference is an affine function $\begin{array} { r } { L _ { \sigma } ( t ) = L _ { \sigma } ( 0 ) + \sum _ { j } \widehat { \ell } _ { \sigma , j } t _ { j } } \end{array}$ , where

$$
\widehat { \ell } _ { \sigma , j } = P _ { j } ( \kappa _ { \sigma } + \nu _ { \sigma } ^ { \top } p _ { j } ) , \qquad \kappa _ { \sigma } = \ell _ { \sigma } ^ { \top } ( \Gamma v _ { 1 } - v _ { 0 } ) + ( \Gamma - 1 ) c _ { \sigma } , \qquad \nu _ { \sigma } = ( \Gamma - 1 ) \xi _ { \sigma } .\tag{140}
$$

Fix $t \in [ 0 , 1 ] ^ { n }$ with $\textstyle \sum _ { j } t _ { j } = n / 2$ . Preserving $\begin{array} { r } { \sum _ { j } t _ { j } , \sum _ { j } P _ { j } t _ { j } , \sum _ { j } P _ { j } } \end{array}$ sin j t<sub>j</sub> , and $\sum _ { j } P _ { j }$ cos $j t _ { j }$ one can move to a point with at most four fractional coordinates. Rounding those coordinates to words with respectively m and $m + 1$ ones and using exact correctness forces a zero of one affine piece within four coordinate changes. Applying this to the centrally symmetric convex body generated by $P _ { j } ( 1 , p _ { j } ) \in \mathbb { R } ^ { 3 }$ and using the elementary convex-body slab bound yields, for some $\sigma ,$

$$
\sum _ { j = 1 } ^ { n } \left| { \widehat { \ell } } _ { \sigma , j } - \operatorname { m e d } ( { \widehat { \ell } } _ { \sigma , 1 } , \ldots , { \widehat { \ell } } _ { \sigma , n } ) \right| \leq C _ { h } M _ { \sigma } , \qquad \left| L _ { \sigma } { \left( { \frac { 1 } { 2 } } \mathbf { 1 } \right) } \right| \leq C _ { h } M _ { \sigma } ,\tag{141}
$$

where $M _ { \sigma } = \operatorname* { m a x } _ { j } | \widehat { \ell } _ { \sigma , j } |$ . The fixed phases $j$ mod $2 \pi$ are equidistributed because $\begin{array} { r } { | \sum _ { j = 1 } ^ { n } e ^ { i k j } | = } \end{array}$ $O _ { k } ( 1 )$ for each fixed nonzero integer k. Applied to the sequence $e ^ { a ^ { \top } p _ { j } } ( \kappa _ { \sigma } + \nu _ { \sigma } ^ { \top } p _ { j } )$ , the first inequality in (141) first bounds $\| a \| _ { 2 }$ , then a compactness argument forces a and $\nu _ { \sigma } / \kappa _ { \sigma }$ to approach zero, and finally the first two discrete Fourier modes of the second-order Taylor expansion give

$$
\kappa _ { \sigma } \neq 0 , \qquad M _ { \sigma } \asymp | \kappa _ { \sigma } | , \qquad \| a \| _ { 2 } ^ { 2 } \leq \frac { C } { n } , \qquad \left\| a + \frac { \nu _ { \sigma } } { \kappa _ { \sigma } } \right\| _ { 2 } \leq \frac { C } { n } .\tag{142}
$$

To return to the raw parameters, put $e _ { c } = c _ { \sigma } - c _ { \sigma } ^ { * }$ and $\tau = 2 \Gamma \delta _ { \sigma } / ( 1 + \Gamma )$ . Direct substitution gives

$$
L _ { \sigma } \left( { \textstyle \frac 1 2 } \mathbf { 1 } \right) = \frac { ( 1 + \Gamma ) T } { 2 } e _ { c } , \qquad \kappa _ { \sigma } = \tau - \nu _ { \sigma } ^ { \top } \mu + ( \Gamma - 1 ) e _ { c } .\tag{143}
$$

Since $T \asymp n , ( 1 4 1 )$ and $( 1 4 2 ) \mathrm { i m p l y } | e _ { c } | \le C | \kappa _ { \sigma } | / ( ( 1 + \Gamma ) n )$ and $| \kappa _ { \sigma } - \tau | \leq C n ^ { - 1 / 2 } | \kappa _ { \sigma } |$ . Hence $\delta _ { \sigma } \neq 0 , | \tau | \asymp | \kappa _ { \sigma } |$ , and $\| \nu _ { \sigma } / \tau - \nu _ { \sigma } / \kappa _ { \sigma } \| _ { 2 } \leq C / n$ . Using ν<sub>σ</sub>/τ = (sinh $\lambda / \dot { \delta } _ { \sigma } ) \xi _ { \sigma }$ proves (138), and the bound on $e _ { c }$ proves (139). □

## C.3 VOLUME INTEGRATION

ProofofTheorem 9. For the upper bound, fix one of the finitely many activation patterns σ and condition on $q ,$ the two symbol key columns, and all value and classifier coordinates except $b _ { 2 , 1 }$ , leaving the two positional key columns unconditioned. Each coordinate of (138) restricts one independent positional key column to a slab whose conditional normalized width is at most

$$
C _ { B } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { n \beta _ { n } \| \boldsymbol { q } \| _ { \infty } } \right\} .\tag{144}
$$

The two columns use disjoint raw coordinates. After they are fixed, (139) restricts $b _ { 2 , 1 }$ to an interval of normalized length $\bar { O _ { B , h } } ( 1 / n )$ . Summing over σ and integrating $t = \| q \| _ { \infty }$ , whose density under the uniform square measure is proportional to t, gives

$$
\begin{array} { r l r } {  { V _ { B } ^ { \mathrm { M a J o R I T Y } , A _ { \mathrm { s i n , g e n } } ( h ) } ( n ) \le \frac { C _ { B , h } } { n } \int _ { 0 } ^ { B } t \operatorname* { m i n } \bigg \{ 1 , \frac { 1 } { n \beta _ { n } t } \bigg \} ^ { 2 } d t } } \\ & { } & { \le \frac { C _ { B , h } ( 1 + \log ( n \beta _ { n } ) ) } { n ^ { 3 } \beta _ { n } ^ { 2 } } = O _ { B , h } \bigg ( \frac { 1 } { n ^ { 3 } \log n } \bigg ) . } \end{array}\tag{145}
$$

For the lower bound, restrict the value and classifier coordinates to a fixed positive-volume open box on which the bias-free logit difference has the form $\ell ^ { \top } A + c _ { H }$ for every attainable attention output,

with constants $0 < \delta _ { - } \leq \delta : = \ell ^ { \top } ( v _ { 1 } - v _ { 0 } ) \leq \delta _ { + }$ and $\xi = C _ { V } ^ { \top } \ell$ satisfying $\| \xi \| _ { 2 } / \delta \le 1 / 1 0 0$ . Such a box is obtained from an interior center with $\boldsymbol { v } _ { 0 } = 0 , \boldsymbol { v } _ { 1 } = \eta ( 1 , 0 ) ^ { \top } , C _ { V } = 0$ , and one uniformly active hidden unit. Set

$$
A ^ { * } = { \frac { v _ { 0 } + \Gamma v _ { 1 } } { 1 + \Gamma } } + C _ { V } \mu , \qquad b _ { 2 , 1 } ^ { * } = b _ { 2 , 0 } - c _ { H } - \ell ^ { \top } A ^ { * } .\tag{146}
$$

At this centered bias, the exact logit difference is

$$
( y ( w ) ) _ { 1 } - ( y ( w ) ) _ { 0 } = \frac { \sum _ { j } ( w _ { j } - 1 / 2 ) \widehat { \ell } _ { j } } { \mathsf { D } ( w ) } , \qquad \widehat { \ell } _ { j } = P _ { j } \left[ \frac { 2 \Gamma \delta } { 1 + \Gamma } + ( \Gamma - 1 ) \xi ^ { \top } ( p _ { j } - \mu ) \right] .\tag{147}
$$

$\operatorname { I f } | \lambda | \leq c _ { 1 } n ^ { - 1 / 2 }$ and $\| a + ( \sinh \lambda / \delta ) \xi \| _ { 2 } \le c _ { 2 } / n$ , a uniform first-order expansion of (147) gives $\widehat { \ell } _ { j } / \tau = F _ { 0 } + O ( 1 / n )$ with $\tau = 2 \Gamma \delta / ( 1 + \Gamma )$ and $F _ { 0 } \ge 3 / 4$ . Therefore the numerator has the MAJORITY sign and magnitude bounded below by a positive constant, while $\mathsf { D } ( w ) = O ( n ) ;$ ; an interval $| b _ { 2 , 1 } - b _ { 2 , 1 } ^ { * } | < c _ { 3 } \bar { / } n$ preserves strict correctness on every input.

It remains to measure a full-dimensional set satisfying these conditions. Restrict the query bias to

$$
q _ { 1 } > 0 , \qquad | q _ { 2 } | \leq q _ { 1 } / 4 , \qquad r : = \beta _ { n } q _ { 1 } \in \left[ \frac { L } { n } , \frac { c } { \sqrt { n } } \right] .\tag{148}
$$

For sufficiently small c, the symbol key columns may vary over their full raw squares while $| \lambda | \leq c _ { 1 } n ^ { - 1 / 2 }$ . Conditional on these choices and on the decoder box, the two coordinates of the compensation condition restrict one entry of each positional key column to an interval of length $\Theta ( 1 / ( n { \bar { r } } ) )$ ; fixed-width restrictions on the other entries keep these intervals inside the raw cube. Their joint normalized mass is therefore $\Omega _ { B , h } ( 1 / ( n ^ { 2 } r ^ { 2 } ) )$ , and the admissible output-bias interval contributes $\Omega _ { B , h } ( 1 / n )$ . Since the query-cone measure after integrating $q _ { 2 }$ is a constant multiple of $r d r / \beta _ { n } ^ { 2 }$

$$
\begin{array} { l } { { \displaystyle V _ { B } ^ { \mathrm { M A J O R I T Y } , \mathcal { A } _ { \mathrm { s i n , g e n } } ( h ) } ( n ) \geq \frac { c _ { B , h } } { n ^ { 3 } \beta _ { n } ^ { 2 } } \int _ { L / n } ^ { c / \sqrt { n } } \frac { d r } { r } } } \\ { { \displaystyle \geq \frac { c _ { B , h } ^ { \prime } } { n ^ { 3 } \log n } } . } \end{array}\tag{149}
$$

Every restricted raw coordinate varies over an interval of positive width and every unrestricted coordinate integrates out, so this is a full-dimensional lower bound. Together with (145), it proves the theorem. □

## D EXPERIMENTAL DETAILS AND SELECTED ADDITIONAL RESULTS

This appendix specifies the experimental protocol used for the main-text comparisons and reports the additional results referenced in the discussion of sinusoidal positional encoding.

## D.1 REPORTED MODELS AND BENCHMARK PROVENANCE

The baseline positional implementation is the model $A _ { \mathrm { p o s } } ( h )$ from Section 2, with quadratic positional coordinates $( j , j ^ { 2 } )$ , and the selectively normalized implementation is $\mathscr { A } _ { \mathrm { n o r m } } ( h )$ from Section 4.2. For the baseline MAJORITY experiment, we use the unrestricted affine NoPE implementation of $A _ { \mathrm { g e n } } ( h )$ The additional sinusoidal MAJORITY control is the fixed-frequency general model $\mathcal { A } _ { \mathrm { s i n } } ( h )$ defined in Section C.1.

We use the same 20 seeds, 2000–2019, at each maximum training length

$$
N \in \{ 1 1 , 2 1 , 3 1 , 4 1 , 5 1 \} .\tag{150}
$$

Training and qualification did not have access to the test banks, and every newly trained checkpoint was selected and frozen before its test evaluation.

## D.2 TRAINING, SELECTION, AND INITIALIZATION

Training uses Adam with learning rate $3 \times 1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , numerical stabilizer $1 0 ^ { - 8 }$ no weight decay, batch size 256, and gradient-norm clipping at one. Each update uses a fresh batch of independent uniformly random bits. For INDEX, the index k is sampled uniformly from the positions in the sampled sequence. For FIRST and INDEX, one data length is sampled uniformly from $\{ 1 , \ldots , N \}$ per batch. For MAJORITY, the sampled length is uniform over the odd integers in that range. Computation uses float64. At data length L, attention scores are multiplied by log L, without an additional $1 / \sqrt { d }$ factor.

Every run is trained for at most 30,000 updates and is validated every 500 updates on a fixed bank of 1,000 examples at length N. Training stops after three consecutive perfect validations. The selected checkpoint maximizes validation accuracy; ties are resolved first by lower cross-entropy and then by the earlier update. No input longer than N is used in checkpoint selection.

For the $A _ { \mathrm { p o s } } ( h )$ and $A _ { \mathrm { n o r m } } ( h )$ models, the vectors $a , r , s , b , c _ { K }$ are initialized from centered normal distributions with standard deviations $0 . 1 / N , 0 . 1 / N , 0 . 1 / N ^ { 2 } , 0 . 1 , 0 . 1$ , respectively. Other parameters use Xavier-uniform initialization and their biases are zero. $\mathcal { A } _ { \mathrm { p o s } } ( h ) / \mathcal { A } _ { \mathrm { n o r m } } ( h )$ pairs use identical initial tensors and the same training streams. For $\mathcal { A } _ { \mathrm { g e n } }$ and $\mathcal { A } _ { \mathrm { s i n } }$ , content query/key blocks and the bounded sinusoidal positional blocks use Xavier-uniform initialization, while shared value, bias, and classifier blocks are copied from the paired ${ \mathcal A } _ { \mathrm { p o s } }$ seed.

## D.3 EVALUATION BANKS, METRICS, AND UNCERTAINTY

For a maximum training length N, the common test grid is

$$
N , \quad 2 N + 1 , \quad 3 N , \quad 5 N , \quad 8 N + 1 , \quad 1 0 N + 1 .\tag{151}
$$

At each test length, all seeds and model variants for the same task use the same saved bank of 1,000 examples. For FIRST, the first bit is exactly balanced and the remaining bits are random. For MAJORITY, 500 random strings are paired with their complements. For INDEX, each of 500 random string–query pairs is paired with the complemented string and the same query.

Test accuracy is the fraction correct on one saved bank. The longest-test statistic is the mean across the 20 seeds at $L = 1 0 N + 1$ . For coefficient analyses and other OOD summaries, the seed-level mean OOD accuracy is the equal-weight mean across the five test lengths strictly greater than N. Confidence intervals are pointwise percentile 95% intervals.

## D.4 COEFFICIENT ANALYSIS

The coefficient analysis uses the 100 selected INDEX checkpoints on $\mathcal { A } _ { \mathrm { p o s } } \mathrm { : }$ 20 seeds at each of the five values of N in (150). For each checkpoint, we compute

$$
\chi = a ^ { \top } s , \qquad \ell = 2 b ^ { \top } s + a ^ { \top } r\tag{152}
$$

directly from the learned parameters.

To avoid treating differences among the five training maxima as evidence for a coefficient association, we rank each coefficient magnitude and the seed-level mean OOD accuracy separately within each value of N, and then pool the within-N ranks. The resulting correlations are −0.655 for $N ^ { 2 } | \chi |$ and −0.516 for $N | \ell |$ . The two-sided permutation test uses 10,000 permutations and shuffles ranks only within the same value of N; both tests give $p < 1 0 ^ { - 3 }$

## D.5 SELECTED RESULTS ACROSS MAXIMUM TRAINING LENGTHS

Table 2 reports the longest-test results. Each entry is the mean percentage accuracy across 20 seeds, followed by its pointwise 95% seed-bootstrap interval.

## D.6 SINUSOIDAL POSITIONAL ENCODING ON MAJORITY

Relevant results on the performance of $\mathcal { A } _ { \mathrm { s i n } }$ compared to $\mathcal { A } _ { \mathrm { g e n } }$ on MAJORITY is displayed in Table 3.

Table 2: Accuracy at the longest tested length $L = 1 0 N + 1$ . Entries are percentages, with pointwise 95% seed-bootstrap intervals in parentheses.
<table><tr><td>N</td><td> $L$ </td><td>FIRST  $( \mathcal { A } _ { \mathrm { p o s } } )$ </td><td>MAJORITY  $( \mathcal { A } _ { \mathrm { g e n } } )$ </td><td>INDEX  $( \mathcal { A } _ { \mathrm { p o s } } )$ </td><td>INDEX  $( \mathcal { A } _ { \mathrm { n o r m } } )$ </td></tr><tr><td>11</td><td>111</td><td>100.0 (100.0–100.0)</td><td>83.8 (78.0–89.4)</td><td> $5 7 . 2 \ : ( 5 5 . 9 - 5 8 . 6 )$ </td><td> $6 3 . 4 ( 6 2 . 7 - 6 4 . 1 )$ </td></tr><tr><td>21</td><td>211</td><td>100.0 (100.0–100.0)</td><td>91.3 (86.2–95.3)</td><td>55.2 (54.1–56.5)</td><td>74.1 (68.8–79.9)</td></tr><tr><td>31</td><td>311</td><td>100.0 (100.0–100.0)</td><td>87.1 (82.6–91.5)</td><td>59.2 (58.3–60.0)</td><td>90.6 (85.6–95.4)</td></tr><tr><td>41</td><td>411</td><td>100.0 (100.0–100.0)</td><td>88.5 (84.2–92.2)</td><td>55.1 (54.5–55.8)</td><td>84.4 (78.1–90.6)</td></tr><tr><td>51</td><td>511</td><td>100.0 (100.0–100.0)</td><td>82.6 (77.6–87.5)</td><td>59.6 (58.6–60.6)</td><td>85.1 (80.2–90.1)</td></tr></table>

Table 3: Mean OOD accuracy on MAJORITY for $\mathcal { A } _ { \mathrm { g e n } }$ and $\mathcal { A } _ { \mathrm { s i n } }$ general models. Entries are percentages, averaged equally over the five test lengths above N, with 95% confidence intervals in parentheses.
<table><tr><td> $N$ </td><td>NoPE General</td><td>Sinusoidal General</td></tr><tr><td>11</td><td>91.6 (88.1–95.0)</td><td rowspan="4">95.3 (93.6–96.8) 93.8 (91.5–95.9) 91.2 (88.2–94.0)</td></tr><tr><td>21</td><td>95.1 (91.8–97.6)</td></tr><tr><td>31</td><td>92.9 (90.0–95.6)</td></tr><tr><td>41 51</td><td>93.4 (90.6–95.8) 91.5 (88.7–94.1) 89.8 (86.2–92.9) 92.9 (90.0–95.6)</td></tr></table>

## D.7 SANITY CHECKS FOR $A _ { \mathrm { n o r m } } ( h )$ ON OTHER TASKS

We also compared $A _ { \mathrm { p o s } } ( h )$ to $\mathscr { A } _ { \mathrm { n o r m } } ( h )$ on FIRST and MAJORITY. For FIRST, both positional models achieve 100% mean accuracy at $\dot { L } = 1 0 N + 1$ for every N in (150).

For our setup of MAJORITY, ${ \mathcal A } _ { \mathrm { p o s } }$ and $\mathcal { A } _ { \mathrm { n o r m } }$ are equivalent, hence producing same results for every seed.