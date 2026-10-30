# EMERGENCE OF THE TRANSFORMER ARCHITECTURE AS THE SCORE OF ENTROPIC LAWS OF RANDOM MEASURES

GIACOMO GRECO

Abstract. We show that the transformer architecture emerges as the score of last-token laws in a natural statistical model of language, where sentences are empirical measures of tokens, a language is a law on ${ \mathcal { P } } ( \mathbb { S } ^ { d } ) ,$ and our prior belief is that sentences carry no structure at all. By the maximum-entropy principle, we select the law closest to this prior in relative entropy among those reproducing an observed sentence-level statistic. We discuss the relevance of the statistic in the posterior interaction between tokens and we build a statistic, able to capture that interaction, with Luce’s choice axiom truncated at the lowest relational order. This procedure gives the softmax choice rule, with the attention cross-moment as its mean parameter. The query–key matrix is then the Lagrange multiplier of a maximum-entropy problem inside each sentence, and the value matrix the multiplier of the constraint over the language, so that the two are untied. Under the optimal entropic law we then show that a transformer head appears in the score of the last-token law. For ordered sentences, causal masking follows from the masked reference choice rule, and with a next-token cross-moment the score equals one causally masked attention head. We also discuss multi-head attention, weight-tied unembedding, and a heuristic reading of residual blocks as projected Langevin steps.

## 1. Introduction

Transformers $[ \mathrm { V S P ^ { + } 1 7 } ]$ are nowadays state-of-the-art neural networks, behind modern large language models [DCLT19]. These networks process sentences by splitting them into tokens from a finite vocabulary , and then process them all at once in a three-stage procedure (see [PH22]). In the first step tokens are embedded into a possibly lower-dimensional space. Then, in the second step the corresponding sequence of vectors $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d + 1 }$ goes through a stack of identical layers, each consisting of a residual attention sub-layer and a residual multi-layer perceptron (MLP). Finally, an unembedding procedure maps the output representation of the last token to a vector of logits, one per element of the vocabulary. Their softmax is the law of the next token, from which a token is sampled and appended to the text. We describe this three-stage process in Figures 1 and 2.

The core mechanism of the transformer architecture is the second step: the only place where tokens interact is exactly in the attention sub-layer. In its simplest form, a single attention head with query, key and value matrices $Q , K , V \in \mathbb { M } _ { d + 1 }$ , maps the collection of tokens $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d + 1 }$ to

$$
x _ { i } \longmapsto \sum _ { j = 1 } ^ { n } { \frac { e ^ { \langle Q x _ { i } , K x _ { j } \rangle } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle Q x _ { i } , K x _ { \ell } \rangle } } } V x _ { j } .\tag{1.1}
$$

Two structural features of this map are by now well understood. Normalization layers place tokens (up to a learned scale) on a sphere, and, in the absence of positional information, the map is permutation-equivariant: it only sees the unordered collection of tokens. We therefore model sentences as unordered collections of points on the unit sphere $x _ { 1 } , \ldots , x _ { n } \in \mathbb { S } ^ { d } \subset \mathbb { R } ^ { d + 1 }$ , which, together with their length $n ,$ are in bijection with finitely supported probability measures on the sphere $\begin{array} { r } { \mu _ { n } = \frac { 1 } { n } \sum _ { i } \delta _ { x _ { i } } } \end{array}$

![](images/5008941fb6aec399b795d7d9ae9e71c2f092db9bb2e3fe4941349fe15d365f07.jpg)  
Figure 1. Transformer architecture as a three-step procedure. In the first step an embedding operator maps each token, represented by a one-hot vector in $\mathbf { \bar { \mathbb { R } } } ^ { | \nu | }$ , to a vector of $\mathbb { R } ^ { d + 1 }$ through a learned matrix $E \in \mathbb { R } ^ { ( d + 1 ) \times | \mathcal { V } | }$ , possibly combined with normalization and with a positional encoding which keeps track of token order. Then, in the second step the sequence of vectors $x _ { 1 } , \ldots , x _ { n }$ goes through a stack of L identical layers. Each layer consists of a residual attention sub-layer, in which every token updates its representation by attending to the other tokens of the sentence, and a residual MLP which acts on each token separately. The output is then projected back to $\mathbb { S } ^ { d }$ via LN. Finally, an unembedding matrix $W _ { \mathrm { o u t } } \in \mathbb { R } ^ { | \mathcal { \hat { V } } | \times ( d + \bar { 1 } ) }$ maps the output representation of the last token to a vector of logits, one per element of the vocabulary. Their softmax is the law of the next token, from which a token is sampled and appended to the text.

Transformers can therefore be seen as maps between probability measures on the sphere, and their composition in layers can be seen as a discretization of a continuity equation driven by the attention vector field [SABP22, GLPR23,GLPR25,GRRB26]. Most works attempt to explain why the transformer architecture works by studying these dynamics, describing how transformers work for a fixed given attention mechanism. Here we would like to explain why transformers work with a complementary statistical question:

Where does the transformer architecture come from? Which probabilistic model of language has softmax self-attention, with its query–key matrix $\Lambda = Q ^ { \top } K$ and its value matrix V, as its natural inference rule, and which variational principle trains these two matrices $\Rsh 2$

In light of token normalization and permutation-equivariance, our answer is formulated at the level of laws of measures, which leads us to an entropic minimization problem on $\mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) ,$ ). In particular, we posit that there exists an underlying, unobserved language law governing how sentences/measures $\mu$ arise. Formally, this is a probability measure $\mathbf { \tilde { P } ^ { \star } } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) )$ on the space of probability measures on $\mathbb { S } ^ { d }$ , a distribution over distributions. Therefore each sentence can be considered as a random (atomic) probability measure over the token space and sampling one sentence amounts to generating a sample from $\mathbf { P } ^ { \star } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) )$ .

In what follows we focus on the core step of the transformer architecture and directly assume that tokens are elements of $\mathbb { S } ^ { d }$ (i.e., the initial embedding is the identity). We postpone the discussion of embedding and unembedding steps to Section 1.6. We first address the unordered model, where tokens within a sentence do not carry positional information (as in Figure 1), while the causal model with positional encoding is treated in Section 1.4 (see also Figure 2). Lastly, the multi-head extension of the transformer architecture is discussed in Section 1.5 (see also Figure 3).

1.1. The choice of the prior. In the absence of data, our prior belief is that a sentence carries no structure at all. Therefore, in order to sample a sentence in an uninformative way, it is enough to consider a random length $N \sim \nu$ sampled according to a probability measure on $\{ 1 , 2 , \ldots \} ( \mathrm { e . g . } ,$ a Poisson distribution shifted by one), together with an independent collection of tokens $X _ { 1 } , X _ { 2 } , \ldots \stackrel { i . i . d . } { \sim }$ vol, and consider the corresponding sentence $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { X _ { i } } } \end{array}$ . Our prior will be the law of a random sentence/measure, built in this way

$$
\mathbf { N } : = \mathrm { L a w } \Big ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { X _ { i } } \Big ) \ \in \ \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) , \qquad N \sim \nu , \qquad X _ { 1 } , X _ { 2 } , \dots \overset { i i d } { \sim } \frac { \mathrm { s } a ! } { \sim } \operatorname { m l } , \ \mathrm { i n d e p e n d e n t ~ o f } \ N .\tag{1.2}
$$

Here vol is the normalized volume, the only rotation-invariant probability measure on $\mathbb { S } ^ { d }$ . Its predictive law is trivial (because of independence):

$$
X _ { n + 1 } \mid X _ { 1 } , \ldots , X _ { n } , ~ N > n \sim { \underline { { \mathrm { v o l } } } } , \qquad \mathbb { P } ^ { \mathbf { N } } { \bigl ( } N = n \mid X _ { 1 : n } , ~ N \geq n { \bigr ) } = { \frac { \nu ( n ) } { \nu ( [ n , \infty ) ) } } .\tag{1.3}
$$

The next token is a fresh uniform draw, independent of the context, and the sentence stops with the hazard rate of $\nu ,$ independently of its tokens.

In this paper we look for the minimal modification of this null model, in relative entropy (a.k.a. Kullback– Leibler divergence), that reproduces a vector of sentence-level statistics $\Phi = ( \phi _ { 1 } , \ldots , \phi _ { k } ) : \mathcal { P } ( \mathbb { S } ^ { d } ) \to \mathbb { R } ^ { k }$ observed in data:

$$
\mathrm { m i n i m i z e ~ K L } ( \mathrm { P } \mid \mathbf { N } ) \quad \mathrm { o v e r ~ P } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) \quad \mathrm { s u b j e c t ~ t o } \quad \mathbb { E } _ { \mathrm { P } } [ \Phi ( \mu ) ] = \mathbb { E } _ { \mathbf { P } ^ { \star } } [ \Phi ( \mu ) ] ,\tag{1.4}
$$

where KL is the Kullback–Leibler divergence, defined for any couple of probabilities P, Q as ${ \mathrm { K L } } ( { \mathrm { P } } | \mathrm { Q } ) : =$ $\mathbb { E } _ { \mathrm { P } } [ \log { \frac { \mathrm { d } \mathrm { P } } { \mathrm { d Q } } } ]$ whenever $\mathrm { P } \ll \mathrm { Q }$ , and set equal to + otherwise. The solution is the Gibbs tilt $\mathrm { P } ^ { \Phi } \propto e ^ { \langle V ^ { \Phi } , \Phi \rangle } \mathbf { N } \left[ \mathrm { C s i 7 5 } \right] .$ for some vector of Lagrange multipliers $V ^ { \Phi } \in \mathbb { R } ^ { k }$ , as soon as $\Phi$ is bounded, continuous and non-degenerate and the constraint lies in the interior of the convex support of Φ under N (Theorem 2.1). In particular, this means that under the Gibbs tilt $\mathrm { P } ^ { \Phi }$ , see Proposition 2.2, the last-token predictive law reads as

$$
\mathrm { P } ^ { \Phi } \big ( X _ { n } \in \mathrm { d } y \big | N = n , \ X _ { 1 : n - 1 } = x _ { 1 : n - 1 } \big ) \ \propto \ \exp \big ( \langle V ^ { \Phi } , \Phi ( \mu _ { y } ) \rangle \big ) \underline { { \mathrm { v o l } } } ( \mathrm { d } y ) , \quad \mathrm { w i t h } \ \mu _ { y } : = \frac { 1 } { n } \big ( \sum _ { i = 1 } ^ { n - 1 } \delta _ { x _ { i } } + \delta _ { y } \big ) .
$$

Therefore, in this model, we could sample the last token via a Langevin sampler associated to the score of its predictive law

$$
\nabla _ { y } \log p ( y \mid X _ { 1 : n - 1 } = x _ { 1 : n - 1 } , N = n ) = \nabla _ { y } \langle V ^ { \Phi } , \Phi ( \mu _ { y } ) \rangle = \frac { 1 } { n } \left. P _ { y } ^ { \perp } \nabla _ { z } \left. V ^ { \Phi } , \delta \Phi ( \mu _ { y } ) \right. ( z ) \right| _ { z = y } ,\tag{1.5}
$$

where $\delta \Phi = ( \delta \phi _ { 1 } , \ldots , \delta \phi _ { k } )$ denotes the first variation of the functional Φ and $\nabla _ { z } \langle V ^ { \Phi } , \delta \Phi ( \mu ) \rangle ( z )$ its gradient. This shows that the whole content of the model lies in the choice of the statistic Φ, and that its first variation is what should capture how tokens collectively behave and attend to each other within each sentence.

1.2. The choice of relevant statistics for the semantic structure of tokens. The semantic structure of tokens cannot be captured by the null model, where the tokens of a sentence are i.i.d. draws from the uniform measure vol. Statistics depending on $\mu$ only through its weights detect only the length of the sentence. Linear statistics $\textstyle \Phi ( \mu ) = \int h \mathrm { d } \mu$ do not create interactions either: given $N = n ,$ , the tilt $\left| { } _ { e } \langle V ^ { \Phi } , \Phi \rangle \right.$ factorizes as $\begin{array} { r } { \prod _ { i < n } e ^ { \langle V ^ { \Phi } , h ( x _ { i } ) \rangle / n } } \end{array}$ , so the tokens remain independent and the next token does not depend on the context. Interaction between tokens therefore requires statistics that involve at least pairs of tokens of the same sentence. Among these, the most natural choice would be the covariance among tokens $\Phi ( \mu ) = \operatorname { C o v } ( \mu )$ ; however, as we show in Proposition 3.1, any statistic that is additive over pairs (i.e. of the form $\begin{array} { r } { \int h \mathrm { d } \mu \mathrm { d } \mu = n ^ { - 2 } \sum _ { i , j } h ( X _ { i } , X _ { j } ) ) } \end{array}$ would produce reinforcements in which every key contributes with a fixed weight that does not depend on the other keys: there is no competition among keys. For moment statistics such as the covariance, the query even attends to the context only through the empirical mean of the other tokens. In particular, in the covariance case we would have

$$
\nabla _ { z } \left. V ^ { \Phi } , \delta \Phi ( \mu _ { y } ) \right. ( z ) \Big \rvert _ { z = y } = \frac { n - 1 } { n } \left( V ^ { \Phi } + ( V ^ { \Phi } ) ^ { \top } \right) \left( y - \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } x _ { i } \right) ,
$$

which shows that the previous tokens influence the next token y only via their empirical mean, and the next token candidate $y$ does not attend to each key individually.

On the contrary, linguistic structure is relational: the role of a token is determined by which other tokens of the sentence it relates to, that is, by a choice among competitors. We model this as a random choice of a key by a query, a choice rule for the sentence $\mu ,$ which we model as a coupling in

$$
\Pi _ { 1 } ( \mu ) = \big \{ \pi \in \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { S } ^ { d } ) : \pi ( \cdot \times \mathbb { S } ^ { d } ) = \mu \big \} ,
$$

the first coordinate being the query and the second the key; we write $\pi ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \pi _ { x } ( \mathrm { d } y )$ . Keys are tokens of the sentence: $\pi _ { x } \ll \mu$ for µ-almost every $x ,$ which is forced as soon as $\operatorname { K L } ( \pi \mid \mu \otimes \mu ) < \infty$ . The uniform choice µ $\mu ,$ in which the key is drawn from the sentence irrespective of the query, models the choice rule without relational content.

The weakest form of competition we can consider in our model is Luce’s choice axiom [Luc59]. In this model, the odds of choosing the key y rather than $y ^ { \prime }$ depend on the query and on the two keys, and not on the other tokens of the sentence. In our model, for a finite sentence with positive choice probabilities, Luce’s choice axiom holds if and only if there is a choice function $h : \mathbb { S } ^ { d } \times \mathbb { S } ^ { d }  \mathbb { R } \ \mathrm { \bar { [ L u c 5 9 } }$ , Chapter 1, Theorem 3], the same for all sentences, such that for any sentence $\mu$ its associated choice rule equals

$$
\pi _ { \mu } ^ { h } ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \frac { e ^ { h ( x , y ) } \mu ( \mathrm { d } y ) } { \int e ^ { h ( x , y ^ { \prime } ) } \mu ( \mathrm { d } y ^ { \prime } ) } .\tag{1.6}
$$

The choice function should encode relational preferences and syntactical relations between tokens, and in particular diferent choice functions might focus on diferent aspects of our language.

A priori the choice function h could be highly non-linear, with spherical harmonics decomposition in $\mathrm { L } ^ { 2 }$ (vol vol) equal to $\begin{array} { r } { h = \sum _ { \ell , \ell ^ { \prime } \geq 0 } h _ { \ell \ell ^ { \prime } } } \end{array}$ with $h _ { \ell \ell ^ { \prime } } \in \mathcal { V } _ { \ell } \otimes \mathcal { V } _ { \ell ^ { \prime } }$ , where $\mathcal { V } _ { \ell }$ denotes the set of spherical harmonics of degree ℓ on $\mathbb { S } ^ { d } .$ . In Proposition 3.4, we argue that the lowest-order term in this expansion that captures tokens’ relations, is $h _ { 1 1 } \in \mathcal { Y } _ { 1 } \otimes \mathcal { Y } _ { 1 } = \{ ( x , y ) \mapsto x ^ { \top } \Lambda y : ~ \Lambda \in \mathbb { M } _ { d + 1 } \}$ , which corresponds to a bilinear form induced by a matrix $\Lambda \in \mathbb { M } _ { d + 1 }$ . Therefore, in view of implementation, we approximate the choice function h with $h _ { 1 1 } = \langle { x , \Lambda y } \rangle$ and consider choice rules induced by the exponential bilinear kernel $k _ { \Lambda } ( x , y ) = e ^ { x ^ { \top } \Lambda y }$ (Proposition 3.4). We refer to the matrix Λ as the choice matrix or attention matrix. The corresponding choice rule is then given by the Gibbs coupling

$$
\pi _ { \mu } ^ { \Lambda } ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \frac { k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) } { Z _ { \mu } ^ { \Lambda } ( x ) } , \qquad Z _ { \mu } ^ { \Lambda } ( x ) = \int k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) .\tag{1.7}
$$

In light of this choice rule, we consider covariance statistics associated to $\pi _ { \mu } ^ { \Lambda }$ . More precisely, in this paper the statistic of interest is the attention cross-moment

$$
S _ { \Lambda } ( \mu ) : = \mathbb { E } _ { \pi _ { \mu } ^ { \Lambda } } [ X Y ^ { \top } ] = \mathbb { E } _ { \mu } [ X \mathrm { A t t n } _ { \Lambda } ( X ; \mu ) ^ { \top } ] \in \mathbb { M } _ { d + 1 } ,\tag{1.8}
$$

where the attention vector field is defined via conditional expectation

$$
\mathrm { A t t n } _ { \Lambda } ( x ; \mu ) : = \mathbb { E } _ { \pi _ { \mu } ^ { \Lambda } } [ Y | X = x ] = \frac { \int y k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) } { Z _ { \mu } ^ { \Lambda } ( x ) } .\tag{1.9}
$$

For each fixed sentence $\mu ,$ the attention cross-moment $S _ { \Lambda } ( \mu )$ is a matrix-valued statistic, whose entries are the (non-centered) cross-moments between the query and key directions of the sentence under the choice rule $\pi _ { \mu } ^ { \Lambda }$ It is the mean parameter (the expectation of the suficient statistic $x y ^ { \top } )$ of the conditional exponential family of choice rules $\{ \pi _ { \mu } ^ { \Lambda } \} _ { \Lambda ; }$ , and it is therefore a natural choice for capturing the relational structure of tokens in a sentence. Moreover, the choice rule $\pi _ { \mu } ^ { \Lambda }$ is the least informative choice rule with its prescribed cross-moment $S _ { \Lambda } ( \mu )$ fixed, that is a conditional entropic projection as in conditional maximum-entropy models of natural language [BDPDP96], with Λ the Lagrange multiplier (Proposition 3.6). Finally, we take the last term appearing in (1.9) as a definition for the attention vector field, whenever x $\not \in \operatorname { s u p p } ( \mu )$

In this paper the statistic of (1.4) is the attention cross-moment, $\Phi = S _ { \Lambda }$ . Its Lagrange multiplier is a matrix $V ^ { \star } = V ^ { \star } ( \Lambda ) \in \mathbb { M } _ { d + 1 } ,$ and the minimizer reads

$$
\mathrm { P } ^ { S _ { \Lambda } } = \mathrm { P } _ { \Lambda , V ^ { \star } } , \qquad \mathrm { P } _ { \Lambda , V ^ { \star } } \propto \exp \left( \left. V ^ { \star } , S _ { \Lambda } \right. \right) \mathbf { N } .\tag{1.10}
$$

With this choice attention appears twice: inside the statistic, and in the score of the law of new tokens (Theorem 3.17), whereas $V ^ { \star }$ corresponds to the value matrix (Theorem 3.17-(c)). In particular, the readout

$$
\langle V ^ { \star } , S _ { \Lambda } ( \mu ) \rangle = \mathbb { E } _ { \mu } [ X ^ { \top } V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( X ; \mu ) ] = \int x ^ { \top } V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) \mu ( \mathrm { d } x )
$$

is the average, over the tokens x of the sentence, of the pairing between x and the output, through $V ^ { \star }$ , of the attention head whose query is x. On a generic sentence in our model $\begin{array} { r } { \mu = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { x _ { i } } } \end{array}$ , for an attention matrix $\Lambda = Q ^ { \top } K$ , the readout explicitly becomes

$$
\langle V ^ { \star } , S _ { \Lambda } \big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { x _ { i } } \big ) \rangle = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle x _ { i } , V ^ { \star } \operatorname { A t t n } _ { \Lambda } ( x _ { i } ; \mu ) \rangle = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big \langle x _ { i } , \sum _ { j = 1 } ^ { n } \frac { e ^ { \langle Q x _ { i } , K x _ { j } \rangle } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle Q x _ { i } , K x _ { \ell } \rangle } } V ^ { \star } x _ { j } \Big \rangle ,
$$

and $V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( x _ { i } ; \mu )$ corresponds exactly to the transformer mechanism (1.1) with value matrix $V ^ { \star }$ and with query matrix $Q$ and key matrix $K$ (with $\Lambda = Q ^ { \top } K )$ . The same transformer architecture will appear as the leading (query) term of the score of the posterior distribution (1.5) which can then be used in a Langevin sampler for the last-token generation.

Let us conclude by noticing that a diferent statistic induced by the choice rule can be considered. Indeed, $\pi _ { \mu } ^ { \Lambda }$ can be variationally characterized as the optimizer of an entropy-regularized problem (cf. Proposition 3.5), whose optimal value equals the attention free energy $\begin{array} { r } { F _ { \Lambda } ( \mu ) = \int } \end{array}$ log $\bar { Z } _ { \mu } ^ { \Lambda } \mathrm { d } \mu$ . The attention cross-moment corresponds to the gradient in Λ of this scalar statistic (Proposition $3 . 5 \mathrm { - ( b ) ) }$ . In Remark 3.20 we also consider a diferent model in which the language is constrained through $F _ { \Lambda }$ instead of $S _ { \Lambda }$ . There, the Gibbs tilt (1.10) is replaced with $e ^ { \lambda ^ { \star } F _ { \Lambda } } \mathbf { N }$ , which as a side-efect necessarily ties the value matrix to the attention matrix $V = \lambda ^ { \star } \Lambda$ . Hence, in this case the values correspond to the keys, whereas in the cross-moment model the value matrix V is free. We refer the reader to Remark 3.20, where we discuss what changes in the free-energy setting.

We can summarize what we have discussed so far and will show below in the following informal statement.

Meta-Theorem (Unordered model). Let $\mathbf { N } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) )$ be the structureless null model (a random number $N \sim \nu$ of i.i.d. uniform tokens on $\mathbb { S } ^ { d } )$ and consider the KL minimization problem

$$
m i n i m i z e ~ \mathrm { K L } ( \mathbf { P } \mid \mathbf { N } ) \quad o v e r ~ \mathbf { P } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) \quad s u b j e c t ~ t o \quad \mathbb { E } _ { \mathbf { P } } [ \Phi ( \mu ) ] = C _ { \Phi } ,
$$

with $C _ { \Phi }$ being the empirical value of the statistic Φ observed from data samples. Assume that ν charges at least one length $n \geq 3$ . Then, the following statements hold true:

(i) Why attention. Statistics Φ that are polynomial in sentences $\mu \in \mathcal P ( \mathbb S ^ { d } )$ , or additive over pairs of tokens, give reinforcements without competition between keys. Statistics induced by choice rule couplings (1.7) preserve relational structure and the lowest-order relational choice rules correspond to couplings $\pi _ { \mu } ^ { \Lambda } ( \mathrm { d } x , \mathrm { d } y ) \ : = \ : \mu ( \mathrm { d } x ) e ^ { \langle x , \Lambda y \rangle } \mu ( \mathrm { d } y ) / Z _ { \mu } ^ { \Lambda } ( x ) \ : \in \ : \Pi _ { 1 } ( \mu )$ induced by attention choice matrices $\Lambda \in \mathbb { M } _ { d + 1 }$ The corresponding mean parameter is then the attention cross-moment $S _ { \Lambda }$ , which is non-degenerate for $\Lambda \neq 0$

(ii) Why this model. For any fixed attention choice matrix $\Lambda \in \mathbb { M } _ { d + 1 }$ , the minimal modification of N in relative entropy that reproduces $\mathbb { E } [ S _ { \Lambda } ]$ is the Gibbs tilt $\mathrm { P } _ { \Lambda , V ^ { \star } } \propto e ^ { \langle V ^ { \star } , S _ { \Lambda } ( \mu ) \rangle } \mathbf { N }$ , with value matrix $V ^ { \star }$ being the corresponding Lagrange multiplier (for $\Lambda \neq 0$ and an interior constraint, Theorem 3.11). In particular query–key attention and value matrices are untied.

(iii) Transformer as last-token prediction score. Conditionally on $N = n$ and on the first $n - 1$ tokens $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , under $\mathrm { P } _ { \Lambda , V ^ { \star } }$ , the law of the last token $X _ { n }$ has density, with respect to vol,

$$
p ( \boldsymbol { y } ) \propto \exp \big ( \langle \boldsymbol { V } ^ { \star } , \boldsymbol { S } _ { \Lambda } ( \mu _ { \boldsymbol { y } } ) \rangle \big ) , \boldsymbol { w } i t h \mu _ { \boldsymbol { y } } = \frac { 1 } { n } \big ( \sum _ { \boldsymbol { j } < n } \delta _ { x _ { j } } + \delta _ { \boldsymbol { y } } \big ) .
$$

Its score on $\mathbb { S } ^ { d }$ is

$$
\nabla _ { \mathbb { S } ^ { d } } \log p ( y ) = \frac { 1 } { n } \ L { P } _ { y } ^ { \perp } \nabla _ { z } \left. V ^ { \star } , \delta S _ { \Lambda } ( \mu _ { y } ) \right. ( z ) \Big | _ { z = y } , \qquad \ L _ { y } ^ { \perp } = \mathrm { I d } - y y ^ { \top } ,
$$

where (Theorem 3.17)

$$
\nabla _ { z } \left. V ^ { \star } , \delta S _ { \Lambda } ( \mu _ { y } ) \right. ( z ) = V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( z ; \mu _ { y } ) + \int \frac { e ^ { \langle x , \Lambda z \rangle } } { \int e ^ { \langle x , \Lambda z ^ { \prime } \rangle } \mu _ { y } ( \mathrm { d } z ^ { \prime } ) } V ^ { \star \top } x \mu _ { y } ( \mathrm { d } x ) + \rho _ { \mu _ { y } } ( z ) .
$$

The first term is the transformer architecture (1.1), the second is a key-side term, and $\rho _ { \mu _ { y } }$ is the correction due to the dependence of the attention weights on z. The score is therefore $n ^ { - 1 }$ times the tangential part of the full gradient of the first variation of $\langle V ^ { \star } , S _ { \Lambda } \rangle$ at the completed sentence $\mu _ { y }$

The first term, $V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( z ; \mu _ { y } )$ , is the query part: the last token z is pulled toward what it reads from the sentence, a softmax-weighted average of the other tokens mapped by $V ^ { \star } .$ . The second is the key part: z is also pulled toward the tokens that attend to it, weighted by how much attention each of them pays to z. The third, $\rho _ { \mu _ { y } } ( z )$ , is the correction due to the attention weights themselves depending on z: moving z changes where z looks, and where the other tokens look, and this changes the readout. The presence of these terms is a consequence of the candidate being, in the statistic, both a key of the other tokens and a query: permutation invariance symmetrizes the roles of query and key.

1.3. The choice of the optimal query–key attention matrix $\Lambda ^ { \star }$ and the value matrix $V ^ { \star }$ . The query–key attention matrix Λ and the value matrix $V ^ { \star }$ play diferent roles. Inside a sentence, Λ plays the role of the Lagrange multiplier of the choice problem (Proposition 3.6): it decides which keys a query selects. It is fixed language-wide, and it cannot be recovered from a single sentence. Over the language, $V ^ { \star }$ is the Lagrange multiplier of the same cross-moment averaged over sentences (Theorem 3.11): it decides which pairings between a token and the content its attention head reads are over-represented in the language, relative to the null model. Query–key attention and value matrices are therefore the inner and outer multipliers of the same cross-moment.

Both matrices are learned from the language by one variational principle: we aim at maximizing the expected log-likelihood ratio of the model against the null model,

$$
\begin{array} { r } { \mathcal { L } ( \Lambda , V ) = \mathbb { E } _ { \mathbf { P } ^ { \star } } \left[ \left. V , S _ { \Lambda } \right. \right] - \log \mathbb { E } _ { \mathbf { N } } \left[ e ^ { \left. V , S _ { \Lambda } \right. } \right] . } \end{array}\tag{1.11}
$$

Equivalently, we look for $( \Lambda , V )$ optimizing the minimax-entropy principle [ZWM97]

$$
\operatorname* { s u p } _ { \Lambda , V } \mathcal { L } ( \Lambda , V ) = \operatorname* { s u p } _ { \Lambda } \operatorname* { i n f } _ { { \mathbb { P } } : \mathbb { E } _ { \mathrm { P } } S _ { \Lambda } = \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } } \mathrm { K L } ( { \mathrm { P } } \mid \mathbf { N } ) .
$$

In this way, the query–key attention matrix selects the attention cross-moment that is least compatible with the null model, and the model is the least informative one that reproduces it. This principle was introduced in natural language processing long before large language models, in the maximum-entropy models of [BDPDP96, Ros96]. We combine this principle with the minimax-entropy models of [ZWM97]: the second term in the log-likelihood (1.11) penalizes readouts that are already large under the null model: a readout that is equally large on the data and on structureless sentences gains nothing. In this paper the matrix Λ is kept fixed: we study the model $\mathrm { P } _ { \Lambda , V ^ { \star } }$ , its predictive laws and its samplers, and we only discuss the training problem for the optimal attention matrix Λ in Remark 3.19.

1.4. Recovering the exact transformer architecture: positions, causal masking and next-token readout. By the Meta-Theorem for the unordered model, the score of the last token contains the transformer architecture (1.1) as its query part, but it also contains the key-side term and the correction $\rho _ { \mu _ { y } }$ . The key-side term, and the part of $\rho _ { \mu _ { y } }$ coming from the other queries, appear because the tokens of the sentence attend to y as one of their keys. The rest of $\rho _ { \mu _ { y } }$ appears because $y$ is itself a query, whose attention weights move with $y .$

Decoder transformers, which generate text, switch of both mechanisms, and we now show how to do the same in our setting. In practice transformers keep track of the position of each token, and the attention is causally masked $[ \mathrm { V S P ^ { + } 1 7 } ] \colon$ a token only attends to tokens that come before it, and to itself, and never sees future tokens. Order cannot be produced by our null model N, which is exchangeable, nor by a statistic of the sentence $\mu ,$ which is invariant under permutations of the tokens. Therefore our sentences must remember the positions of their tokens. One way of doing this is by lifting our sentence space to $\mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } )$ , the space of ordered sentences

$$
\vec { \mu } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { ( X _ { i } , i ) } \in \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } ) ,
$$

where we record each token together with its position. Clearly this measure is in bijection with the ordered sequence $( X _ { 1 } , \ldots , X _ { N } )$ , and its image under $( x , i ) \mapsto x$ is our unordered sentence $\begin{array} { r } { \mu = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \delta _ { X _ { i } } \in \mathcal P ( \mathbb S ^ { d } ) } \end{array}$

The null model in this ordered setting becomes the law $\vec { \bf N } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } ) )$ of $\vec { \mu }$ , under which the length and the ordered tokens have the law $\nu ( n ) \underline { { \mathrm { v o l } } } ^ { \otimes n }$ on $\{ N = n \}$

In practice, in order to prevent tokens from seeing the future, a causal masking mechanism is implemented in the attention softmax mechanism, which switches to zero all the attention weights appearing in (1.1) when the key token $( X _ { j } , j )$ comes after the query token $( X _ { i } , i )$ in our sentence, $\mathrm { i . e . }$ , for $i < j$ . In our model, the choice rule is the only place where relational structure is encoded; thus in order to enforce this property we should introduce the masking there. As a consequence, now Luce’s axiom should hold among the competitors that are not later than the query, and the uniform choice $\mu \otimes \mu$ should be replaced by the uniform choice among the keys $( y , j )$ with $j \leq i ,$ , for the query $( x , i )$ . Then, by arguing as in the unordered model, we are led to consider the lowest-order relational choice rule as the causal Gibbs coupling

$$
\pi _ { \vec { \mu } } ^ { \mathrm { A } , \mathrm { M } } ( \mathrm { d } x \mathrm { d } i , \mathrm { d } y \mathrm { d } j ) : = \vec { \mu } ( \mathrm { d } x \mathrm { d } i ) \frac { \mathbb { 1 } _ { j \leq i } e ^ { \langle x , \Lambda y \rangle } \vec { \mu } ( \mathrm { d } y \mathrm { d } j ) } { \int \mathbb { 1 } _ { j ^ { \prime } \leq i } e ^ { \langle x , \Lambda y ^ { \prime } \rangle } \vec { \mu } ( \mathrm { d } y ^ { \prime } \mathrm { d } j ^ { \prime } ) } = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \Pi _ { i j } ^ { \mathrm { M } } \delta _ { ( x _ { i } , i ) } \otimes \delta _ { ( x _ { j } , j ) } \ \in \mathcal { P } \big ( ( \mathbb { S } ^ { d } \times \mathbb { N } ) ^ { 2 } \big ) ,\tag{1.12}
$$

with $\begin{array} { r } { \Pi _ { i j } ^ { \mathrm { M } } : = n ^ { - 1 } \frac { \mathbb { 1 } _ { j \leq i } e ^ { \langle x _ { i } , \Lambda x _ { j } \rangle } } { \sum _ { \ell = 1 } ^ { n } \mathbb { 1 } _ { \ell \leq i } e ^ { \langle x _ { i } , \Lambda x _ { \ell } \rangle } } } \end{array}$ . Notice that this coupling can equivalently be rewritten as

$$
\pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } } = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { n } \frac { e ^ { \langle x _ { i } , \Lambda x _ { j } \rangle + M _ { i j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle x _ { i } , \Lambda x _ { \ell } \rangle + M _ { i \ell } } } \delta _ { ( x _ { i } , i ) } \otimes \delta _ { ( x _ { j } , j ) } ,\tag{1.13}
$$

where $M \in \{ 0 , - \infty \} ^ { n \times n }$ is the causal additive mask defined as $M _ { i j } = \left( - \infty \right) \mathbb { 1 } _ { i < j }$ , with the convention $e ^ { - \infty } = 0$ We refer to the above as the masked choice rule, which is again the least informative rule with its query–key cross-moment, the masked entries being forced by the finiteness of the relative entropy with respect to the masked uniform choice, in which each query chooses uniformly among the tokens not later than itself (Proposition 4.2).

Under this choice rule the attention that the token $( x _ { i } , i )$ gives to the already seen tokens, the attention vector field, is defined as the conditional expectation

$$
\begin{array} { r } { \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \stackrel {  } { \mu } ) : = \mathbb { E } _ { ( ( X , I ) , ( Y , J ) ) \sim \pi _ { \stackrel { \Lambda } { \mu } } ^ { \Lambda , \mathrm { M } } } [ Y \bigm | ( X , I ) = ( x _ { i } , i ) ] = \displaystyle \sum _ { j \leq i } \frac { e ^ { \langle x _ { i } , \Lambda x _ { j } \rangle } } { \sum _ { \ell \leq i } e ^ { \langle x _ { i } , \Lambda x _ { \ell } \rangle } } x _ { j } } \\ { = \displaystyle \sum _ { j = 1 } ^ { n } \frac { e ^ { \langle x _ { i } , \Lambda x _ { j } \rangle + M _ { i j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle x _ { i } , \Lambda x _ { \ell } \rangle + M _ { i \ell } } } x _ { j } , } \end{array}
$$

and we take the latter expression as a definition for any $( x , i ) \not \in \operatorname { s u p p } ( { \vec { \mu } } )$

We can then associate to this choice rule the cross-moment statistic. However, the attention cross-moment considered in (1.8) pairs each token with what it reads; for generation, instead, we are interested in how well the next token is predicted by what the current token reads, i.e., its attention vector field. That is, we are interested in the cross-moment between the next token $X _ { i + 1 }$ and the key chosen by $X _ { i }$ according to the choice rule, namely $\mathbb { E } _ { ( X , Y ) \sim \pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } } } [ Y | X = X _ { i } ]$ . This leads, for $n \geq 2$ , to the shifted masked cross-moment

$$
S _ { \Lambda } ^ {  } ( { \vec { \mu } } ) = { \frac { 1 } { n - 1 } } \sum _ { i = 1 } ^ { n - 1 } x _ { i + 1 } \operatorname { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; { \vec { \mu } } ) ^ { \top } = \mathbb { E } _ { \pi _ { \vec { \mu } } ^ { \Lambda ,  } } [ X Y ^ { \top } ] ,\tag{1.14}
$$

where we have introduced the next-token coupling $\begin{array} { r } { \pi _ { \vec { \mu } } ^ { \Lambda ,  } = \bigl ( \iota _ { \vec { \mu } } \times \iota _ { \vec { \mu } } \bigr ) _ { \# } \bigl ( \frac { n } { n - 1 } J \Pi ^ { \mathrm { M } } \bigr ) } \end{array}$ (cf. (4.3)), with $\iota _ { \vec { \mu } } ( i ) = ( x _ { i } , i )$ being the lift of positions to tokens in ${ \vec { \mu } } , J \in \mathbb { M } _ { n }$ the lower shift matrix, $J _ { i j } = \mathbb { 1 } _ { \{ i = j + 1 \} }$ , and $\Pi ^ { \mathrm { M } } = ( \Pi _ { i j } ^ { \mathrm { M } } ) \in \mathbb { M } _ { n }$ is the matrix associated to $\pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } }$ , so that $\pi _ { \vec { \pi } } ^ { \Lambda , \mathrm { M } } = ( \iota _ { \vec { \mu } } \times \iota _ { \vec { \mu } } ) _ { \# } \Pi ^ { \mathrm { M } }$

Two reasons support this choice. Firstly, it is a classical feature of the maximum-entropy language models of [Ros96,BDPDP96] to pair a summary of the history with the next word, and the shifted cross-moment is their relational version. The history up to position i is summarized by what $x _ { i }$ reads through the attention vector field $\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } )$ , and it is paired with the next token $x _ { i + 1 }$ . By contrast, $S _ { \Lambda }$ describes the sentence as a bag of tokens. Secondly, no token is paired with what it reads itself, whereas in $S _ { \Lambda }$ each token is paired with its own read, which contains the token itself (Remark 4.8).

The masked choice rule is unchanged: by Proposition 4.2 it is still the least informative masked rule given its own query–key cross-moment $\begin{array} { r } { \frac { 1 } { n } \sum _ { i } x _ { i } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top } } \end{array}$ , and Λ still plays the role of the corresponding Lagrange multiplier inside each sentence. What changes is the pairing of its output at the level of the language: the statistic $S _ { \Lambda } ^ {  }$ pairs what $x _ { i }$ reads with the next token $x _ { i + 1 }$ instead of with $x _ { i }$ itself. For every $V ,$ the model $\mathrm { P } _ { \Lambda , V } ^ {  } \propto e ^ { \langle V , S _ { \Lambda } ^ {  } \rangle } \vec { \mathbf { N } }$ is the entropic projection of the null model onto its own mean of $S _ { \Lambda } ^ {  }$ , with $V$ as the Lagrange multiplier (Section 4.2), and we denote by $V ^ { \star } = V ^ { \star } ( \Lambda )$ the multiplier associated to our constraint $\mathbb { E } _ { \mathrm { P } } [ S _ { \Lambda } ^ {  } ] = C \in \mathbb { M } _ { d + 1 }$ , with the latter matrix being what we observe from data samples.

With this statistic the transformer architecture appears exactly, without correction terms. Indeed, for $n \geq 2$ conditionally on $N = n$ and on the first $n - 1$ tokens $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , we will show in Theorem 4.6 that the last-token predictive law equals

$$
\mathrm { P } _ { \Lambda , V ^ { \star } } ^ {  } ( X _ { n } \in \mathrm { d } y \big | N = n , \ X _ { 1 : n - 1 } = x _ { 1 : n - 1 } ) \ \propto \ \exp \Big ( \frac { 1 } { n - 1 } y ^ { \top } V ^ { \star } \mathrm { A t t } \mathrm { n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \Big ) \underline { { \mathrm { v o l } } } ( \mathrm { d } y ) ,
$$

a von Mises–Fisher law with mean direction vector $V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \stackrel {  } { \mu } )$ (normalized) and concentration parameter $\left\| V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \right\| / ( n - 1 )$ . The score associated to this distribution then simply corresponds to the projection onto the tangent space $T _ { y } \mathbb { S } ^ { d }$ of the von Mises–Fisher mean direction vector, and if we

decompose the attention matrix as $\Lambda = Q ^ { \top } K$ we get

$$
\nabla _ { \mathsf { S } ^ { d } } \log p ( y ) = { \frac { 1 } { n - 1 } } P _ { y } ^ { \perp } V ^ { \star } \mathrm { A t t n } _ { \Lambda } { \big ( } ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } { \big ) } = { \frac { 1 } { n - 1 } } P _ { y } ^ { \perp } \sum _ { j = 1 } ^ { n } { \frac { e ^ { \langle Q x _ { n - 1 } , K x _ { j } \rangle + M _ { n - 1 , j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle Q x _ { n - 1 } , K x _ { \ell } \rangle + M _ { n - 1 , \ell } } } } V ^ { \star } x _ { j } .\tag{1.15}
$$

This is exactly one causally masked attention head (1.1) $( \mathrm { e . g . , [ V S P ^ { + } 1 7 }$ , Figure $2 ] ,$ [PH22, Algorithm 4]), with query matrix $Q ,$ , key matrix K and value matrix $V ^ { \star }$ , evaluated at the last position of the context, followed by the projection onto the tangent space of the sphere. There is no key-side term and no correction term, because the candidate $y$ is neither a key nor a query of the statistic: the mask removes it from the keys of every query, and the shifted pairing never uses it as a query.

We can summarize this part in the following informal statement.

Meta-Theorem (Causal model). Let $\vec { \bf N } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } ) )$ be the structureless null prior with positions (law of ordered sentences $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { ( X _ { i } , i ) } } \end{array}$ , with $N \sim \nu$ and independent $X _ { 1 } , X _ { 2 } , \dotsc \stackrel { i . i . d . } { \sim } \stackrel { \mathrm { v o l } } { \longrightarrow } \stackrel { } { \longrightarrow }$ , and consider the KL minimization problem

$$
\begin{array} { r l } { m i n i m i z e \mathrm { \normalfont ~ K L } ( \mathrm { P } | \overset {  } { \mathbf { N } } ) } & { o v e r \mathrm { \normalfont ~ P } \in  { \mathcal { P } } (  { \mathcal { P } } ( \mathbb { S } ^ { d } \times \mathbb { N } ) ) \quad s u b j e c t \mathrm { \normalfont ~ t o } \quad \mathbb { E } _ { \mathrm { P } } [ S _ { \Lambda } ^ {  } ( \xrightarrow [ \mu ] {  } ] ] = \widehat { S _ { \Lambda } ^ {  } } , } \end{array}\tag{1.16}
$$

with $\widehat { S _ { \Lambda } ^ {  } }$ being the empirical shifted masked cross-moment observed from samples (or its true mean value $\mathbb { E } [ S _ { \Lambda } ^ {  } ] )$ dAssume that ν charges at least one length $n \geq 2$ . Then, the following statements hold true:

(i) Causal choice rule. Assume Luce’s axiom among the competitors not later than the query, with a choice function that does not depend on positions. With the mask in the reference coupling, the lowest-order relational choice rule is causally masked softmax attention $\pi _ { \vec { \pi } } ^ { \Lambda , \mathrm { M } }$ defined in (1.12). It is the unique minimizer of the relative entropy with respect to the masked uniform choice, given its own query–key cross-moment, with Λ the Lagrange multiplier (Proposition 4.2).

(ii) Next-token model. For any fixed attention choice matrix $\Lambda \in \mathbb { M } _ { d + 1 }$ , the minimal modification $o f { \vec { \mathbf { N } } }$ in relative entropy that reproduces $\mathbb { E } [ S _ { \Lambda } ^ {  } ]$ is the Gibbs tilt $\mathrm { P } _ { \Lambda , V ^ { \star } } ^ {  } \propto e ^ { \langle V ^ { \star } , S _ { \Lambda } ^ {  } \rangle } \vec { \mathbf { N } } _ { }$ , with value matrix $V ^ { \star }$ being the corresponding Lagrange multiplier (for an interior constraint, Theorem $4 . 5 \mathrm { - } ( b ) )$ . In particular query–key attention and value matrices are untied.

(iii) Transformer as last-token prediction score. For n $\geq 2$ with $\nu ( n ) > 0$ , conditionally on $N = n$ and on $t h e$ first $n - 1$ tokens $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , under $\mathrm { P } _ { \Lambda , V ^ { \star } } ^ {  }$ , the last token $X _ { n }$ has, exactly, the von Mises–Fisher law with density proportional to $\begin{array} { r l } {  { \exp ( \frac { 1 } { n - 1 } \boldsymbol { y } ^ { \top } \boldsymbol { V } ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) ) } } \end{array}$ with respect to vol. For $\Lambda = Q ^ { \top } K$ its score on $\mathbb { S } ^ { d } \ i s$

$$
\nabla _ { \mathsf { S } ^ { d } } \log p ( y ) = { \frac { 1 } { n - 1 } } P _ { y } ^ { \perp } V ^ { \star } \mathrm { A t t n } _ { \Lambda } { \big ( } ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } { \big ) } = { \frac { 1 } { n - 1 } } P _ { y } ^ { \perp } \sum _ { i = 1 } ^ { n } { \frac { e ^ { \langle Q x _ { n - 1 } , K x _ { i } \rangle + M _ { n - 1 , j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle Q x _ { n - 1 } , K x _ { \ell } \rangle + M _ { n - 1 , \ell } } } } V ^ { \star } x _ { j } ,
$$

that is $\scriptstyle { \frac { 1 } { n - 1 } }$ times the tangential part of one causally masked attention head, with query matrix $Q$ , key matrix $K$ and value matrix $V ^ { \star } ,$ evaluated at the last position of the context.

This Meta-Theorem leads us to the exact transformer architecture. Indeed, the last token’s log-density is the smooth function $\begin{array} { r } { y \mapsto \frac { 1 } { n - 1 } y ^ { \top } V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) } \end{array}$ on $\mathbb { S } ^ { d }$ , with gradient $\begin{array} { r } { \frac { 1 } { n - 1 } P _ { y } ^ { \perp } ( V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \stackrel {  } { \mu } ) ) } \end{array}$ Therefore the last token’s law is the invariant law of the Langevin stochastic diferential equation

$$
\mathrm { d } Y _ { t } = \frac { 1 } { n - 1 } P _ { Y _ { t } } ^ { \perp } \big ( V ^ { \star } \mathrm { A t t n } _ { \Lambda } \big ( \big ( x _ { n - 1 } , n - 1 \big ) ; \vec { \mu } \big ) \big ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } W _ { t } ^ { \mathrm { S } ^ { d } } ,\tag{1.17}
$$

with $W ^ { \mathbb { S } ^ { d } }$ a Brownian motion on $\mathbb { S } ^ { d }$ with generator $\frac { 1 } { 2 } \Delta _ { \mathbb { S } ^ { d } }$ , which is a reversible and ergodic SDE [BGL14]. Therefore, in order to sample from the law of the last token, we are led to consider the projected Euler–Maruyama scheme with step $\tau$

$$
Y _ { k + 1 } = \mathrm { L N } \Big ( Y _ { k } + \frac { \tau } { n - 1 } P _ { Y _ { k } } ^ { \perp } \big ( V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \big ) + \sqrt { 2 \tau } P _ { Y _ { k } } ^ { \perp } \xi _ { k } \Big ) ,\tag{1.18}
$$

with LN the LayerNorm normalization or the plain normalization $\operatorname { L N } ( w ) = w / \left\| w \right\|$ onto the sphere, and $\xi _ { k } \overset { i i d } { \sim } \mathcal { N } ( 0 , 1 )$ . This recovers almost exactly one step of a single-head transformer block: a residual update by the output of a causally masked attention head, followed by a normalization onto the sphere, with injected noise. The identification concerns the form of one update only. The drift in (1.18) is computed once, from the context $x _ { 1 : n - 1 } .$ , and does not depend on $Y _ { k }$ . Depth is therefore not explained by this construction (see the conclusion).

![](images/107aefd8ffb2edf5a4737f7e7092b2f4f6fe398336e21b078ba1ea38abcd775f.jpg)  
Figure 2. Causal version of the pipeline of Figure 1. Two things change. First, each token is recorded together with its position, so that the sentence becomes the ordered measure $\begin{array} { r } { \vec { \mu } = \frac { 1 } { n } \sum _ { i } \delta _ { ( x _ { i } , i ) } \in \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } ) } \end{array}$ Second, the attention sub-layer is causally masked: the additive mask M sets to zero the weight of every key coming after the query $( j > i )$ , so that each token only reads itself and the tokens before it.

The only ingredient of a transformer block that is not present in this scheme is the position-wise multi-layer perceptron (MLP) that follows the attention step, while the only ingredient of the scheme that is not present in a deterministic transformer block is the noise $\sqrt { 2 \tau } P _ { Y _ { k } } ^ { \perp } \xi _ { k }$ . We argue, at a heuristic level, that the two play the same role. First, in the splitting interpretation of $\mathrm { [ L L H ^ { + } 1 9 ] }$ , a transformer layer is a Lie–Trotter splitting step of an interacting particle system: the attention sub-layer is the interaction term, while the position-wise MLP is the term acting on each token separately. In the scheme above the interaction is the attention drift, and the term acting on each token separately is the noise. Second, Agazzi et al. $\mathrm { \left[ A B M G ^ { + } 2 6 \right] }$ prove that in deep and wide transformers whose MLP blocks have random weights, as at initialization, the MLP blocks produce in the scaling limit a transport-type noise on the sphere, independent of the configuration of the tokens and common to all of them, so that the token dynamics converge to a stochastic interacting particle system on $\mathbb { S } ^ { d }$ whose density solves an SPDE with a Laplace–Beltrami difusion term. In our sampling problem only the candidate token moves, the context being fixed, so that the distinction between common and independent noise is immaterial: the MLP noise acts on the candidate as a Brownian perturbation, as in the Langevin scheme. Related stochastic versions of self-attention dynamics, in which the noise is added by hand [Rig26, SS26] or arises from random value matrices [FSE<sup>+</sup>26], display the same mechanism: noise prevents the collapse of the dynamics onto a single cluster, which the deterministic dynamics would reach. In the language of the scheme, without noise each step is a projected gradient ascent step of $y \mapsto y ^ { \top } V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } )$ , which converges to the mode of the von Mises–Fisher law instead of sampling from it. We stress that this identification is heuristic: in a trained transformer the MLP weights are not random, and the results of $[ \mathrm { A B M G ^ { + } 2 6 } ]$ concern the random-weight scaling regime.

1.5. Emergence of multi-head architectures: more choice rules in the statistics. A query need not choose its key under a single rule. Suppose that it chooses under H diferent rules, each satisfying Luce’s axiom with its own choice function and its own normalization, for instance because diferent rules capture diferent relations between tokens. Truncating each choice function at its lowest relational order, as above, gives H choice rules with pairwise distinct attention matrices $\Lambda _ { h } = { Q } _ { h } ^ { \top } K _ { h } , h = 1 , \tiny { \cdot \cdot \cdot } , H$ , and the natural statistic is the vector of the corresponding cross-moments. Below we distinguish how considering multiple choice functions in the statistical constraint afects our model.

![](images/a57cd212a38f8f23cd66b459cef4721a0d6c2528a6cb2f134555bf4d0fbf62b2.jpg)  
Figure 3. Multi-head version of the causal transformer block of Figure 2. The single masked head is replaced by H heads, each with its own query, key and value matrices $Q _ { h } , K _ { h } , V _ { h }$ , all sharing the same causal mask M. The head outputs are summed before the residual connection and the normalization. In our model each head corresponds to a diferent choice rule, with attention matrix $\Lambda _ { h } = Q _ { h } ^ { \top } K _ { h }$ , and $V _ { h }$ is the Lagrange multiplier of the corresponding cross-moment.

Unordered model. With $\Phi = ( S _ { \Lambda _ { 1 } } , \ldots , S _ { \Lambda _ { H } } )$ , the minimizer of (1.4) is

$$
{ \bf P } ^ { \star } \propto \exp \Big ( \sum _ { h = 1 } ^ { H } \langle V _ { h } ^ { \star } , S _ { \Lambda _ { h } } ( \mu ) \rangle \Big ) { \bf N } = \exp \Big ( \sum _ { h = 1 } ^ { H } \int x ^ { \top } V _ { h } ^ { \star } \mathrm { A t t n } _ { \Lambda _ { h } } ( x ; \mu ) \mu ( \mathrm { d } x ) \Big ) { \bf N } ,
$$

with one value matrix $V _ { h } ^ { \star }$ per head, each being the Lagrange multiplier of the corresponding cross-moment. Existence and uniqueness of the multipliers follow from Theorem 2.1, since the vector statistic is non-degenerate as soon as ν charges some length $n \geq 2 H + 1$ (Proposition A.4 with $A _ { 0 } = 0 )$ . Since first variations are linear in the statistic, Lemma 2.3 shows that the score of the last token is the sum over h of the single-head scores of Theorem 3.17, whose query part is

$$
\sum _ { h = 1 } ^ { H } V _ { h } ^ { \star } \mathrm { A t t n } _ { \Lambda _ { h } } ( y ; \mu _ { y } ) = \sum _ { h = 1 } ^ { H } \sum _ { j = 1 } ^ { n } \frac { e ^ { \langle Q _ { h } y , K _ { h } x _ { j } \rangle } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle Q _ { h } y , K _ { h } x _ { \ell } \rangle } } V _ { h } ^ { \star } x _ { j } ,
$$

that is, multi-head attention: concatenating the H head outputs amounts to summing over the heads, $V _ { h } ^ { \star }$ being the value projection of head h. As for one head, each head also contributes a key-side term and a correction.

Causal model. With the masked choice rules and the shifted cross-moments, the tilt is exp $\begin{array} { r l } { \phantom { \sum } } & { { } \Big ( \sum _ { h } \langle V _ { h } ^ { \star } , S _ { \Lambda _ { h } } ^ {  } \rangle \Big ) \vec { \bf N } } \end{array}$ Its readout is still linear in the last token, with a coeficient that depends on the context only, and the proof of Theorem 4.6 applies verbatim: given $N = n$ and $X _ { 1 : n - 1 } = x _ { 1 : n - 1 } ,$ , the last token has the von Mises–Fisher law with density proportional to

$$
\exp \Big ( \frac { 1 } { n - 1 } y ^ { \top } \sum _ { h = 1 } ^ { H } V _ { h } ^ { \star } \mathrm { A t t n } _ { \Lambda _ { h } } \big ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } \big ) \Big ) ,
$$

whose mean direction is the output, at the last position of the context, of a causally masked multi-head attention layer, without key-side or correction terms. Multi-head attention is thus the statistical counterpart of a query having several independent ways of choosing its keys and the corresponding sampling scheme reads as

$$
Y _ { k + 1 } = \mathrm { L N } \Bigl ( Y _ { k } + \frac { \tau } { n - 1 } P _ { Y _ { k } } ^ { \perp } \bigl ( \sum _ { h = 1 } ^ { H } V _ { h } ^ { \star } \mathrm { A t t n } _ { \Lambda _ { h } } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \bigr ) + \sqrt { 2 \tau } P _ { Y _ { k } } ^ { \perp } \xi _ { k } \Bigr ) ,
$$

which corresponds to a residual update by the output of a causally masked multi-head attention architecture, followed by a normalization onto the sphere, with noise injection.

1.6. Unembedding as a projection onto the finite vocabulary. In practice vocabularies are finite, and the transformer architecture includes an embedding of the vocabulary $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { | \mathcal { V } | } \}$ into a lower-dimensional space and a final unembedding operator, from which the logits of the law of the next token are computed. For ease of exposition, let us consider a finite vocabulary $\gamma \subset \mathbb { S } ^ { d }$ and the identity embedding. According to our model, in order to sample the last token we are led to simulate the Langevin SDE (1.17), for instance via the scheme (1.18), which mimics the evolution of the last token in a transformer. Since is a discrete subset of $\mathbb { S } ^ { d }$ the output $\widehat { Y } \in \mathbb S ^ { d }$ of the scheme is in general not an element of the vocabulary. To sample a token we therefore blook for the elements of $\nu$ that are closest to $\widehat { Y }$ , and we measure closeness through the inner products $\langle v , { \widehat { Y } } \rangle$ $v \in \mathcal V :$ if $\langle v , { \widehat { Y } } \rangle \approx 1$ b, the next token is most probably $v ,$ whereas if $\langle v , { \widehat { Y } } \rangle \leq 0$ it is unlikely to be v.

bIn light of these considerations, given $\widehat { Y }$ b, we define the law of the last token $Y ^ { \nu }$ on the vocabulary by

$$
p ^ { \mathcal { V } } \big ( Y ^ { \mathcal { V } } = v \big | \widehat { Y } \big ) = \frac { \exp \big ( \sigma ( \langle v , \widehat { Y } \rangle ) \big ) } { \sum _ { w \in \mathcal { V } } \exp \big ( \sigma ( \langle w , \widehat { Y } \rangle ) \big ) } , \qquad v \in \mathcal { V } , \qquad \sigma ( t ) = \tan \big ( \frac { \pi } { 4 } ( t - 1 ) \big ) ,\tag{1.19}
$$

that is, the logit of v is a non-decreasing, non-positive function of the inner product $\langle v , \widehat { Y } \rangle$ , which equals 0 when $\widehat { Y } = v$ (exact recovery of a token) and $- \infty$ in the antipodal case $\widehat { Y } = - v ,$ bso that antipodal tokens are never bsampled. The choice of σ is arbitrary: any non-decreasing function $\sigma \colon [ - 1 , 1 ] \to [ - \infty , 0 ]$ with $\sigma ( 1 ) = 0$ would do, possibly with a temperature parameter, $\sigma _ { T } ( t ) = \sigma ( t ) / T$

This construction recovers the unembedding step of transformers. In practice, the output $\widehat { Y }$ of the last layer is multiplied by an unembedding matrix $W _ { \mathrm { o u t } } \ \in \mathbb { R } ^ { | \bar { \mathcal { V } } | \times ( d + 1 ) }$ b, and the law of the next token is the softmax of the resulting logits,

$$
p ^ { \mathcal { V } } \big ( Y ^ { \mathcal { V } } = v \big | \widehat { Y } \big ) = \frac { \exp \big ( ( W _ { \mathrm { o u t } } \widehat { Y } ) _ { v } \big ) } { \sum _ { w \in \mathcal { V } } \exp \big ( ( W _ { \mathrm { o u t } } \widehat { Y } ) _ { w } \big ) } ,
$$

where $( W _ { \mathrm { o u t } } \widehat { Y } ) _ { \imath }$ is the entry of $W _ { \mathrm { o u t } } \widehat { Y } \in \mathbb { R } ^ { | \nu | }$ indexed by $v .$ In practice $W _ { \mathrm { o u t } }$ is learned from data. Our b binterpretation suggests instead building $W _ { \mathrm { o u t } }$ by stacking the tokens of the vocabulary as rows,

$$
W _ { \mathrm { o u t } } = \left( \begin{array} { c } { v _ { 1 } ^ { \top } } \\ { v _ { 2 } ^ { \top } } \\ { \vdots } \\ { v _ { | \mathcal { V } | } ^ { \top } } \end{array} \right) \in \mathbb { R } ^ { | \mathcal { V } | \times ( d + 1 ) } , \qquad \mathrm { s o ~ t h a t } \qquad W _ { \mathrm { o u t } } \widehat { Y } = \left( \begin{array} { c } { \langle v _ { 1 } , \widehat { Y } \rangle } \\ { \langle v _ { 2 } , \widehat { Y } \rangle } \\ { \vdots } \\ { \langle v _ { | \mathcal { V } | } , \widehat { Y } \rangle } \end{array} \right) \in \mathbb { R } ^ { | \mathcal { V } | } .
$$

With this notation, (1.19) reads

$$
p ^ { \mathcal { V } } \big ( Y ^ { \mathcal { V } } = \cdot \big | \widehat { Y } \big ) = \operatorname { s o f t m a x } \Big ( \sigma \big ( W _ { \mathrm { o u t } } \widehat { Y } \big ) \Big ) ,
$$

with σ applied componentwise. For the linear choice $\sigma ( t ) = ( t - 1 ) / T$ this is exactly the softmax of the logits $W _ { \mathrm { o u t } } \widehat { Y } / T$ , since the constant $- 1 / T$ cancels in the normalization: the standard unembedding with temperature $T .$ bThe unembedding matrix is then the matrix of the vocabulary, which is fixed and does not need to be trained, and it has a clear geometric meaning: $W _ { \mathrm { o u t } }$ projects the output of the scheme, a point of the sphere that is in general not a token, onto the finite vocabulary, by scoring each token by its alignment with $\widehat { Y }$ , and the next token is sampled with higher odds for the tokens closest to $\widehat { Y }$ . The non-linearity $\sigma$ bonly reshapes these scores, and it could be trained while keeping $W _ { \mathrm { o u t } }$ bfixed, since the latter encodes the proximity of the output to the admissible tokens. This reading of the unembedding as a projection onto the vocabulary is also the one underlying the “logit lens” and related interpretability methods, which decode intermediate representations of a transformer by multiplying them by the unembedding matrix [DGGB23]. An exact version of this construction holds when the reference law of tokens is itself discrete; we discuss it in Section B.

Finally, let us explain how the embedding can be included in this discussion. To lower the dimension of the tokens, the first step of a transformer is an embedding, which maps the vocabulary, represented by the one-hot vectors $e _ { 1 } , \ldots , e _ { | \mathcal { V } | } \in \mathbb { R } ^ { | \mathcal { V } | }$ , into a lower-dimensional space through a learned matrix $E \in \mathbb { R } ^ { ( d + 1 ) \times | \mathcal { V } | }$ ， followed by a normalization onto $\mathbb { S } ^ { d }$ . The embedded vocabulary is then $\{ v _ { i } = E e _ { i } / \left| E e _ { i } \right| \colon \ i \leq | \mathcal { V } | \} \subset \mathbb { S } ^ { d }$ , and our construction gives the unembedding matrix whose rows are the normalized columns of $E _ { : }$

$$
W _ { \mathrm { o u t } } = \mathrm { d i a g } \big ( | E e _ { 1 } | , \dots , | E e _ { | \mathcal { V } | } \big | \big ) ^ { - 1 } E ^ { \top } .
$$

Hence the unembedding is still learned, but it is not an independent parameter: up to the normalization of the rows, it is the transpose of the embedding matrix, which is learned once. This is the weight tying of input and output embeddings, used in the original transformer $[ \mathrm { V S P ^ { + } 1 7 } ]$ and introduced in [PW17, IKS17], where it is motivated empirically and, in [IKS17], through a loss framework; in our framework it is a consequence of interpreting the unembedding as the projection of the output of the sampling scheme onto the embedded vocabulary. Untying embedding and unembedding gives more flexibility to large models $[ \mathrm { C F T ^ { + } 2 1 } ]$ and is associated with more isotropic representations [MM24], while tying biases token embeddings toward the output space [LHAG26]. In our setting this can be circumvented without untying: $W _ { \mathrm { o u t } }$ only encodes the proximity of Y to the tokens, while the output-specific parameters are carried by the trainable non-linearity σ, applied to $W _ { \mathrm { o u t } } \widehat { Y }$

1.7. Comparison with existing contributions. A vast body of literature has addressed why transformers work well. Most contributions aim at explaining this by studying how sentences evolve when fed into a deep transformer architecture. This line of research treats tokens as interacting particles and layers of a deep architecture as splitting steps of a numerical ODE solver for a convection–difusion equation $\mathrm { [ L L H ^ { + } 1 9 ] }$ . The same approach has been lifted to the space of probability measures, where one could treat the mean-field limit of self-attention as a continuity equation on ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ [SABP22, GLPR25]. This last approach has led to many fruitful discoveries such as the emergence of clustering [GLPR23, CLPR25], of long metastable phases [GKPR24], also under causal masking [KPR24] (see also [GRRB26] for expressivity). These works take the attention field as given and study the flow it generates, and gradient-flow structures are obtained for tied and symmetric matrices or for a reweighted metric [GLPR25, Section 3]. In our work instead, we consider every possible untied pair (Λ, V) and argue that the self-attention field is the query part of a Wasserstein gradient of the attention readout (Remark 3.18), that the unnormalized field is the score of a kernel-sum statistic (Remark 3.2), and that in the causal model one masked head recovers exactly the score of the last token (Theorem 4.6). Similarly to [Rig26, SS26, FSE<sup>+</sup>26, ABMG<sup>+</sup>26], we also consider stochastic attention dynamics, whose noise appears here in the Langevin sampler of the last token.

Finally, the biggest diference between these works and ours concerns the architecture parameters: there the attention is a fixed given input (often symmetric or the identity) and the value matrix is often tied to the query–key matrix; here both matrices are obtained as optimal parameters of an entropic projection of a structureless null model on ${ \mathcal { P } } ( { \mathcal { P } } ( \mathbb { S } ^ { d } ) )$ .

Ours is not the first contribution establishing a connection between entropic transport problems and the softmax attention mechanism (Proposition 3.5). For a single query, this one-sided entropic transport reading of attention is developed in [Lit25], which also interprets the backward pass as an advantage-based policy gradient. There the score and the reference measure are given; here the bilinear score follows from Luce’s axiom and the harmonic truncation (Proposition 3.4), and Λ is the Lagrange multiplier of the query–key cross-moment (Proposition 3.6). The same entropic transport problem considered in [Lit25], with both marginals prescribed instead, corresponds to the Schrödinger problem [Sch31, Sch32, Léo14, Nut21]. This choice leads to doubly stochastic attention matrices, also known in this setting as Sinkformers [SABP22] (Remark 3.7). The same idea also appears in [Fra26], where a text is a distribution of tokens and attention is a probabilistic relation between them: softmax normalization produces conditional relations, Sinkhorn normalization produces joint relations with prescribed marginals. There the normalization is a design choice based on the Schrödinger problem and the Schrödinger/entropic plan adds to the softmax weights a key bias, the Schrödinger/entropic potential, which depends on the whole sentence. This is precisely what Luce’s axiom [Luc59, Axiom 1] excludes here in our derivation of the attention mechanism as the lowest harmonic order relational rule. In particular, the connection with one-sided entropic transport problems here is a consequence of our derivation rather than the starting point for it.

Diferent papers have addressed the derivation of the attention mechanism. In addition to the entropic transport approach, other regularizations can be considered which yield sparse variants (e.g., [MA16]). Our derivation is not based on a diferent regularization but rather on Luce’s axiom, the classical foundation of the softmax in discrete choice [Luc59, McF74, Yel77, Luc77]. A derivation close to ours is the one considered in modern Hopfield networks and energy transformers, where attention is identified as a descent step for an energy with values tied to keys [RSL<sup>+</sup>21]. This corresponds to our free-energy statistic (Remark 3.20), whereas our cross-moment model unties query–key and value matrices.

A complementary statistical line of work fixes the attention architecture and a data model, and studies what it can represent or what risk minimization recovers: the relevant token in single-location regression [MBBB25], the centroids of a Gaussian mixture [MSMB25], and the leading principal direction of the token covariance [MSB26]. Viewing prompts as measures, Boursier and Boyer [BB26] show that, for i.i.d. centered Gaussian tokens, softmax attention converges in the infinite-prompt limit to a linear operator on the token measure; as noted in [GBB26], this removes query-dependent selection, which Gaussian-mixture tokens restore. An analogous dichotomy appears in Proposition 3.1, although through a diferent mechanism: there, moment statistics only pool the context whatever the token law, and competition between keys requires relational structure, which in the present work is derived from Luce’s axiom rather than from the data distribution. Bohbot et al. [BLPV26] also model prompts as empirical measures of tokens and quantify how fast attention computed on n tokens converges to its infinite-token limit (see also the concentration bounds in [BB26]).

Finally, the link between language models, conditional maximum entropy and exponential families is well established, and dates back to before the word large appeared in front of language models. In [BDPDP96,Ros96] conditional maximum-entropy models pair features of the history with the next word, whereas in [RCZ01] whole-sentence exponential models are introduced as tilts of a baseline language model by sentence features. The model we consider here has this form as well, with a structureless baseline and a relational, matrix-valued feature, and the selection of Λ follows the minimax-entropy principle [ZWM97]. Gibbs measures and entropy also appear in statistical-mechanics analyses of transformers (e.g. [TMIS24, CBKZ25, TCD<sup>+</sup>25, BTEZ25]). There the Gibbs measure is placed on the weights, as a posterior or a Boltzmann measure of the training loss, for a fixed model of the data, whereas here the Gibbs measure lives on the space of sentences.

1.8. Conclusion and future perspectives. In this paper we have shown that the attention head of the transformer architecture is the score of the law of the last token in a simple statistical model of language. The setting is natural in light of two structural features of transformers, normalization and permutation equivariance: sentences are empirical measures of tokens on the sphere, a language is a law on ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ , and our prior belief is that sentences carry no structure at all. The only principle we use is the classical maximum-entropy, or minimal-information, principle: among all laws of sentences that reproduce an observed sentence-level statistic, we select the one closest to the structureless null model in relative entropy. Within this modeling setting, the softmax attention head, its untied query–key and value matrices, the causal mask and the multi-head sum are derived from our modeling choices rather than architectural inputs. Softmax attention is the lowest-order choice rule compatible with Luce’s axiom, whereas polynomial statistics, which do not let keys compete, only pool the context. Its query–key matrix is the Lagrange multiplier of a conditional maximum-entropy problem inside each sentence, while the value matrix is the Lagrange multiplier of the attention cross-moment over the language. In particular, query–key and value matrices are untied, as in practice. In the unordered model the score of the last token contains a transformer head, together with a key-side term and a correction. For ordered sentences, causal masking is placed in the reference (uniform) choice rule and propagates to the attention rule through the finiteness of the relative entropy, and the next-token cross-moment removes both extra terms. The last token then has exactly a von Mises–Fisher law, whose score is one causally masked attention head. A projected Euler discretization of its Langevin dynamics has the form of a residual attention update followed by normalization, with the injected noise playing, heuristically, the role of the MLP. Moreover, several constraining statistics give a multi-head attention model, and the projection onto a finite vocabulary gives a weight-tied unembedding. Finally, constraining the attention free energy instead of the cross-moment gives a model of the same kind, in which values are necessarily tied to keys (Remark 3.20): the untied architecture of transformers is singled out by the cross-moment statistic.

We believe our contribution opens the way to several possible research directions. In particular, we are currently investigating the use of diferent priors, the analysis of higher-order attention models, where we truncate the choice rule function at higher harmonic orders, and the inclusion of feature embeddings within the choice rule (cf. Proposition 3.6). From a statistical perspective, we have characterized an optimal choice rule Λ by the minimax-entropy principle (Remark 3.19), but its statistical analysis (consistency and rates when trained over J observed sentences) remains open. Lastly, in this work we have considered the law of the last token and we have seen how a one-layer transformer architecture emerges in its sampling scheme. However, in practice many diferent layers of this architecture are stacked on top of each other and all tokens undergo the same numerical steps. It is unclear how this fits the current setup, and we are currently investigating how this deep architecture could be explained from our entropic projection point of view. Understanding this in particular would lead us to consider evolution equations on $\bar { \mathcal { P } } ( \mathbb { S } ^ { d } )$ and ${ \mathcal { P } } ( { \mathcal { P } } ( \mathbb { S } ^ { d } ) )$ ).

1.9. Organization and notation. In Section 2 we solve the entropic problem (1.4) for a general statistic (Theorem 2.1), and express the law of the last token and its score through the first variation of the statistic (Lemma 2.3). In Section 3 we show that polynomial statistics only pool the context, derive the lowest-order choice rule from Luce’s axiom and study the model with the attention cross-moment at a fixed attention matrix. We show existence and uniqueness of the value matrix, and compute the first variation of the readout, and the exact law and score of the last token. We conclude with remarks on training and on the free-energy statistic. In Section 4 we treat ordered sentences, the causal choice rule and the shifted cross-moment, and prove that the last token has a von Mises–Fisher law whose score is one causal attention head. The appendices contain the proof of the non-degeneracy of attention cross-moments and the emergence of the unembedding matrix for discrete vocabularies.

Notation. We write $\mathbb { M } _ { d + 1 } = \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) }$ for the space of square matrices of size $d + 1$ , with the inner product $\langle A , B \rangle = \operatorname { t r } ( A ^ { \top } B )$ . Let $\left\| \cdot \right\| _ { \mathrm { o p } }$ and $\lVert \cdot \rVert _ { \mathrm { F } }$ be the operator and Frobenius norms, so that $x ^ { \top } \Lambda y = \langle \Lambda , x \bar { y } ^ { \top } \rangle$ and $\left| x ^ { \top } \Lambda y \right| \leq \left\| \Lambda \right\| _ { \mathrm { o p } } \leq \left\| \Lambda \right\| _ { \mathrm { F } }$ for $x , y \in \mathbb { S } ^ { d }$ . We assume d $\geq 1 .$ and $P _ { x } ^ { \perp } = \mathsf { I } - x x ^ { \top }$ is the orthogonal projection onto the tangent space $T _ { x } \mathbb { S } ^ { d } . \ \mathcal { P } ( \mathcal { X } )$ carries the weak topology, $\textstyle \mu ( f ) = \int f \mathrm { d } \mu$ , and $\begin{array} { r } { \mathrm { K L } ( \mathrm { P } \mid \mathrm { Q } ) = \int \log \frac { \mathrm { d P } } { \mathrm { d Q } } \mathrm { d P } \mathrm { i f } \mathrm { P } \ll \mathrm { Q } } \end{array}$ and + otherwise. We write $x _ { 1 : n } = ( x _ { 1 } , \ldots , x _ { n } )$ and $( x _ { 1 : n } , y ) \in ( \mathbb { S } ^ { d } ) ^ { n + 1 }$ for the vector obtained by appending $y . \ \mathrm { F i n a l l y } , \ \mathbb { E } _ { \mathbf { N } }$ and $\mathbb { E } _ { \mathrm { P } }$ denote expectations of functions of the sentence $\mu \in \mathcal P ( \mathbb { S } ^ { d } )$ .

Statement on use of AI. The author used Claude (Anthropic; models Sonnet 5.5 and Opus 5.5) for language editing, for rendering Figures 1 to 3 from the author’s sketches, for bibliographic exploration, and for discussing and checking mathematical arguments. The author takes full responsibility for the mathematical arguments, their correctness, and the final content of the manuscript.

## 2. Preliminaries

In this section the statistic Φ is general. We solve the entropic problem (1.4), compute the exact law of the last token of a sentence under the solution, and express its score through the first variation of $\Phi$

2.1. Sentences and the entropic projection. Let us start this preliminary section with a few basic remarks on our setup, on the null reference prior and on entropic minimization problems. Firstly, since $\mathbb { S } ^ { d }$ is compact, recall that ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ is compact for the weak topology, and so is ${ \mathcal { P } } ( { \mathcal { P } } ( \mathbb { S } ^ { d } ) )$ ). Notice also that, since vol is difuse, under N the atoms of $\mu$ are almost surely pairwise distinct, so $N$ is almost surely the number of atoms of $\mu$ and is a function of $\mu .$

For $m \geq 1$ and $\boldsymbol { x } _ { 1 : m } \in ( \mathbb { S } ^ { d } ) ^ { m }$ let us consider the function $\begin{array} { r } { L _ { m } ( x _ { 1 : m } ) = \frac { 1 } { m } \sum _ { i \leq m } \delta _ { x _ { i } } } \end{array}$ , and let

$$
{ \mathcal { E } } _ { m } = \bigl \{ L _ { m } ( x _ { 1 : m } ) : x _ { 1 : m } \in ( \mathbb { S } ^ { d } ) ^ { m } \bigr \} \subset { \mathcal { P } } ( \mathbb { S } ^ { d } ) , \qquad S _ { \mathrm { N } } = \bigcup _ { m \in \mathrm { s u p p } ( \nu ) } { \mathcal { E } } _ { m } .
$$

Since $L _ { m } \colon ( \mathbb { S } ^ { d } ) ^ { m } \to \mathcal { P } ( \mathbb { S } ^ { d } )$ is continuous and $( \mathbb { S } ^ { d } ) ^ { m }$ is a compact space, each ${ \mathcal { E } } _ { m }$ is compact too. Moreover, the set $\mathcal { S } _ { \mathbf { N } }$ corresponds to the support of the null prior N. Indeed for any sentence $\mu \sim \mathbf { N }$ , conditionally on having length $N = m$ , we have $\mu \in \mathcal { E } _ { m }$ (since $\underline { { \mathrm { v o l } } } ^ { \otimes m }$ has full support on $( \mathbb { S } ^ { d } ) ^ { m } )$ and the support of the mixture over $m \in \operatorname { s u p p } ( \nu )$ is the closure of the union of the supports. Finally, if supp(ν) is infinite, then N has full support, i.e. $\mathcal { S } _ { \bf N } = \mathcal { P } ( \mathbb { S } ^ { d } )$ . This directly follows from Varadarajan’s theorem [Var58]: for any $\rho \in \mathcal { P } ( \mathbb { S } ^ { d } )$ it is enough to consider a collection of i.i.d. random tokens $X _ { 1 } , X _ { 2 } , \ldots$ . with common law $\rho \colon$ then $L _ { m } ( X _ { 1 : m } )$ weakly converges almost surely toward $\rho .$ In particular it means that there exists a realization $( x _ { i } ) _ { i \geq 1 } \subset \mathbb { S } ^ { d }$ of these random variables such that $L _ { m } ( x _ { 1 : m } )$ weakly converges toward $\rho .$ . Since supp(ν) is infinite it contains a diverging sequence m along which we then have $L _ { m _ { k } } ( x _ { 1 : m _ { k } } ) \in \mathcal { E } _ { m _ { k } }$ weakly converging to $\rho ,$ which implies $\textstyle \rho \in \overline { { \bigcup _ { m \in \mathrm { s u p p } ( \nu ) } \mathcal { E } _ { m } } } = S _ { \mathbf { N } }$

Next, we recall a standard result for constrained KL-minimization problems, such as ours (1.4). In view of that, let Φ : $\mathcal { P } ( \mathbb { S } ^ { d } ) \to \mathbb { R } ^ { k }$ be a bounded and weakly continuous map and let us consider

$$
\Gamma _ { \Phi } ( \eta ) = \log \mathbb { E } _ { \mathbf { N } } \big [ e ^ { \langle \eta , \Phi \rangle } \big ] , \qquad \mathrm { P } _ { \eta } = e ^ { \langle \eta , \Phi \rangle - \Gamma _ { \Phi } ( \eta ) } \mathbf { N } , \qquad \mathcal { K } _ { \Phi } = \overline { { \mathrm { c o n v } } } \left( \Phi \big ( S _ { \mathbf { N } } \big ) \right) \subset \mathbb { R } ^ { k } ,
$$

where conv $\left( \Phi ( S _ { \bf N } ) \right)$ denotes the closure of the convex hull of $\Phi \left( { \cal S } _ { \bf N } \right)$

Theorem 2.1 (Entropic projection theorem). Let Φ : $\mathcal { P } ( \mathbb { S } ^ { d } ) \to \mathbb { R } ^ { k }$ be bounded, weakly continuous and assume that $\langle u , \Phi \rangle$ is not constant on <sub>N</sub> for every $u \in \mathbb { R } ^ { k } \setminus \{ 0 \}$ . Then, the log-partition function $\Gamma _ { \Phi }$ is real-analytic and strictly convex on $\mathbb { R } ^ { k }$ , and Γ is a difeomorphism $o f \mathbb { R } ^ { k }$ onto the interior of $\kappa _ { \Phi }$ . For $c \in \operatorname { i n t } ( \mathcal { K } _ { \Phi } )$ the problem

$$
\operatorname* { m i n } \big \{ \mathrm { K L } ( \mathrm { P } | \mathbf { N } ) : \mathrm { P } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) , \mathbb { E } _ { \mathrm { P } } [ \Phi ] = c \big \}\tag{2.1}
$$

has the unique solution $\mathrm { P } _ { \eta _ { c } } \propto \exp ( \langle \eta _ { c } , \Phi \rangle )$ N, where $\eta _ { c }$ is the Lagrange multiplier such that $\nabla \Gamma _ { \Phi } ( \eta _ { c } ) = c$ . Every admissible P satisfies the Pythagorean identity

$$
\mathrm { K L } ( \mathrm { P } \mid \mathbf { N } ) = \mathrm { K L } ( \mathrm { P } \mid \mathrm { P } _ { \eta _ { c } } ) + I _ { \Phi } ( c ) , \qquad I _ { \Phi } ( c ) : = \mathrm { K L } ( \mathrm { P } _ { \eta _ { c } } \mid \mathbf { N } ) = \operatorname* { s u p } _ { \eta \in \mathbb { R } ^ { k } } \left\{ \langle \eta , c \rangle - \Gamma _ { \Phi } ( \eta ) \right\} .\tag{2.2}
$$

For $c \notin { K _ { \Phi } }$ no P with $\mathrm { K L } ( \mathrm { P } | \mathrm { N } ) < \infty$ is admissible.

The characterization of minimizers for KL divergence as exponential tilting of the reference measure is a well-known fact in optimization analysis under linear constraints. We refer the reader to [FS11,Bro86,BN78,Csi75, CM05]. For the sake of completeness we provide below a proof of this result in our setting, based on [Bro86].

Proof. We divide the proof into steps.

Reduction to an exponential family on $\mathbb { R } ^ { k }$ . All the quantities in the statement depend on $\mu \in \mathcal P ( \mathbb { S } ^ { d } )$ only through $\Phi ( \mu )$ . Let $\nu _ { \Phi } : = \Phi _ { \# } \mathbf { N } \in \mathcal { P } ( \mathbb { R } ^ { k } )$ and $M : = \operatorname* { s u p } \left| \Phi \right| < \infty$ , so that $\nu _ { \Phi }$ is concentrated on the ball $\{ | t | \leq M \}$ and

$$
\Gamma _ { \Phi } ( \eta ) = \log \mathcal { Z } ( \eta ) , \qquad \mathcal { Z } ( \eta ) : = \int _ { \mathbb { R } ^ { k } } e ^ { \langle \eta , t \rangle } \nu _ { \Phi } ( \mathrm { d } t ) .
$$

Thus $\{ e ^ { \langle \eta , t \rangle - \Gamma _ { \Phi } \left( \eta \right) } \nu _ { \Phi } \} _ { \eta \in \mathbb { R } ^ { k } }$ is a standard exponential family on $\mathbb { R } ^ { k }$ in the sense of $\mathrm { \ B r o 8 6 }$ , Definition 1.1], with base measure $\nu _ { \Phi }$ and suficient statistic t, and $\mathrm { P } _ { \eta }$ is its lift to sentences: $\Phi _ { \# } \mathrm { P } _ { \eta } = e ^ { \langle \eta , t \rangle - \Gamma _ { \Phi } ( \eta ) } \nu _ { \Phi }$

Analyticity. For $\boldsymbol { \eta } \in \mathbb { C } ^ { k }$ we have $\left| e ^ { \left. \eta , t \right. } \right| \le e ^ { M | \mathrm { R e } \eta | }$ on the support of $\nu _ { \Phi }$ , so the integral defining $\mathcal { Z }$ converges for every $\eta \in \mathbb { C } ^ { k }$ and, by dominated convergence, may be diferentiated under the integral sign in each complex variable: is an entire function, the Laplace transform of a compactly supported measure (cf. [Rud91, Theorem $7 . 2 3 \mathrm { { J } ) }$ . On $\mathbb { R } ^ { k }$ it is real and $\mathcal { Z } ( \eta ) \geq e ^ { - M | \eta | } > 0$ , so $\Gamma _ { \Phi } = \log { \mathcal { Z } }$ is real-analytic on $\mathbb { R } ^ { k }$ too, with $| \Gamma _ { \Phi } ( \eta ) | \le M | \eta |$ Diferentiating under the integral sign gives

$$
\nabla \Gamma _ { \Phi } ( \eta ) = \mathbb { E } _ { \mathrm { P } _ { \eta } } [ \Phi ] , \qquad \nabla ^ { 2 } \Gamma _ { \Phi } ( \eta ) = \operatorname { C o v } _ { \mathrm { P } _ { \eta } } ( \Phi ) \succeq 0 .
$$

Minimality and strict convexity. We claim $\mathrm { C o v } _ { \mathrm { P } _ { \eta } } ( \Phi ) \succ 0$ . Since $\mathrm { P } _ { \eta }$ and N are equivalent (the density $e ^ { \langle \eta , \Phi \rangle - \Gamma _ { \Phi } ( \eta ) }$ is bounded away from 0 and ), $\mathrm { P } _ { \eta }$ has support $\mathcal { S } _ { \mathbf { N } }$ . If $u ^ { \top } \mathrm { C o v } _ { \mathrm { P } _ { n } } ( \Phi ) u = 0$ for some $u \ne 0$ , the continuous function $\langle u , \Phi \rangle$ is $\mathrm { P } _ { \eta } .$ -almost surely constant, hence constant on $\mathcal { S } _ { \mathbf { N } }$ , which is excluded by assumption. This is equivalent to saying that the exponential family $\{ e ^ { \langle \eta , t \rangle - \Gamma _ { \Phi } \left( \eta \right) } \nu _ { \Phi } \} _ { \eta \in \mathbb { R } ^ { k } }$ is minimal. Consequently, $\Gamma _ { \Phi }$ is strictly convex.

Difeomorphism. The parameter space $\{ \eta : \ \mathcal { Z } ( \eta ) < \infty \}$ is the whole of $\mathbb { R } ^ { k }$ , which is open: the family is regular, hence steep [Bro86, Definition 3.2]. For a minimal and steep standard exponential family, the mean value map $\eta \mapsto \nabla \Gamma _ { \Phi } ( \eta )$ is a bijection of the interior of the natural parameter space onto the interior of the convex support of the base measure [Bro86, Theorem 3.6]. Here the convex support is $\overline { { \mathrm { c o n v } } } ( \mathrm { s u p p } ( \nu _ { \Phi } ) ) = \mathcal { K } _ { \Phi } :$ indeed Φ is continuous and $S _ { \mathbf { N } } = \mathrm { s u p p } ( \mathbf { N } )$ is compact, so that $\Phi ( S _ { \bf N } )$ is compact as well and $\mathrm { s u p p } ( \nu _ { \Phi } ) = \Phi ( S _ { \bf N } )$ . Hence $\nabla \Gamma _ { \Phi } : \mathbb { R } ^ { k } \to \mathrm { i n t } ( \mathcal { K } _ { \Phi } )$ is a bijection. It is real-analytic and its Jacobian $\nabla ^ { 2 } \Gamma _ { \Phi }$ is invertible everywhere, so by the inverse function theorem its inverse is smooth: $\nabla \Gamma _ { \Phi }$ is a difeomorphism.

Existence and uniqueness. Since $\nabla \Gamma _ { \Phi } : \mathbb { R } ^ { k } \to \mathrm { i n t } ( \mathcal { K } _ { \Phi } )$ is a bijection, for any c int $\left( \kappa _ { \Phi } \right)$ there exists a unique $\eta _ { c } \in \mathbb { R } ^ { k }$ such that $\nabla \Gamma _ { \Phi } ( \eta _ { c } ) = c$ . As a consequence of this, since $\nabla \Gamma _ { \Phi } ( \eta ) = \mathbb { E } _ { \mathrm { P } _ { \eta } } [ \Phi ]$ , we have shown that $\mathrm { P } _ { \eta _ { c } }$ is a probability measure satisfying the constraint $\mathbb { E } _ { \mathrm { P } _ { \eta _ { c } } } [ \Phi ] = c .$ Moreover, by definition we have

$$
\mathrm { K L } ( \mathrm { P } _ { \eta _ { c } } | \mathbf { N } ) = \mathbb { E } _ { \mathrm { P } _ { \eta _ { c } } } [ \langle \eta _ { c } , \Phi \rangle - \Gamma _ { \Phi } ( \eta _ { c } ) ] = \langle \eta _ { c } , c \rangle - \Gamma _ { \Phi } ( \eta _ { c } ) < + \infty ,
$$

which is finite. Therefore, the set of admissible solutions to (2.1)

$$
\mathcal { A } _ { c } : = \left\{ \mathrm { P } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) : \mathbb { E } _ { \mathrm { P } } [ \Phi ] = c \right\}
$$

contains $\mathrm { P } _ { \eta _ { c } }$ , which has finite relative entropy. The set $A _ { c }$ is convex, by the linearity of the map $\mathrm { P } \mapsto \mathbb { E } _ { \mathrm { P } } [ \Phi ]$ , and weakly closed, since this map is weakly continuous, Φ being bounded and continuous. Since the sublevel sets of $\mathrm { K L } ( \cdot | \mathbf { N } )$ are weakly compact [DE97, Lemma 1.4.3], the set $\mathcal { A } _ { c } \cap \left\{ \mathrm { P } : \mathrm { \mathrm { ~ K L } } ( \mathrm { P } \mid \mathbf { N } ) \leq \mathrm { \mathrm { K L } } ( \mathrm { P } _ { \eta _ { c } } \mid \mathbf { N } ) \right\}$ is non-empty and weakly compact. This is enough for arguing that there exists a solution to (2.1): the relative entropy functional $\mathrm { P } \mapsto \mathrm { K L } ( \mathrm { P } \mid \mathbf { N } )$ is weakly lower semicontinuous, hence it attains its minimum over this compact set, which is its minimum over $\boldsymbol { \mathcal { A } } _ { c } ,$ and this minimum is finite. Moreover, this minimum is unique because $\mathrm { K L } ( \cdot | \mathbf { N } )$ is strictly convex on its domain $\{ \mathrm { P } : \mathrm { K L } ( \mathrm { P } \mid \mathrm { N } ) < \infty \}$ (cf. [DE97, Lemma 1.4.3]), and $\boldsymbol { A } _ { c }$ is convex (a strictly convex functional has at most one minimizer over a convex set, giving uniqueness).

Characterization of the minimizer. We aim at showing that $\mathrm { P } _ { \eta _ { c } }$ is the unique minimizer. Indeed, for any admissible $\mathrm { P } \in { \mathcal { A } } _ { c }$ we have

$$
\begin{array} { r l } & { \mathrm { K L } ( \mathrm { P } \left. \mathbf { N } \right) - \mathrm { K L } ( \mathrm { P } _ { \eta _ { c } } \left. \mathbf { N } \right) = \mathrm { K L } ( \mathrm { P } \left. \mathbf { N } \right) - \left. \eta _ { c } , c \right. + \log \mathbb { E } _ { \mathbf { N } } \left[ e ^ { \left. \eta _ { c } , \Phi \right. } \right] } \\ & { \qquad = \mathrm { K L } ( \mathrm { P } \left. \mathbf { N } \right) - \mathbb { E } _ { \mathrm { P } } [ \left. \eta _ { c } , \Phi \right. ] + \log \mathbb { E } _ { \mathbf { N } } \left[ e ^ { \left. \eta _ { c } , \Phi \right. } \right] } \\ & { \qquad = \mathrm { K L } ( \mathrm { P } \left. \mathrm { P } _ { \eta _ { c } } \right. \geq 0 , } \end{array}
$$

with equality if and only if $\mathrm { ~ P ~ } = \mathrm { ~ P ~ } _ { \eta _ { c } }$ . This shows that the minimizer over $\boldsymbol { A } _ { c }$ is necessarily $\mathrm { P } _ { \eta _ { c } } ,$ whereas the validity of (2.2) follows from the above computation combined with the Donsker–Varadhan variational formula [DE97, Lemma 1.4.3]

$$
\begin{array} { r } { \mathrm { K L } ( \mathbf { P } _ { \eta _ { c } } | \mathbf { N } ) = \underset { \varphi \in { \cal C } ( \mathcal { P } ( \mathbb { S } ^ { d } ) ) } { \operatorname* { s u p } } \left\{ \mathbb { E } _ { \mathbf { P } _ { \eta _ { c } } } [ \varphi ] - \log \mathbb { E } _ { \mathbf { N } } [ e ^ { \varphi } ] \right\} \geq \underset { \eta \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } \left\{ \mathbb { E } _ { \mathbf { P } _ { \eta _ { c } } } [ \langle \eta , \Phi \rangle ] - \Gamma _ { \Phi } ( \eta ) \right\} } \\ { = \underset { \eta \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } \left\{ \langle \eta , c \rangle - \Gamma _ { \Phi } ( \eta ) \right\} \geq \langle \eta _ { c } , c \rangle - \Gamma _ { \Phi } ( \eta _ { c } ) = \mathrm { K L } ( \mathbf { P } _ { \eta _ { c } } | \mathbf { N } ) . } \end{array}
$$

Constraints outside $\kappa _ { \Phi }$ . If $\mathrm { K L } ( \mathrm { P } | \mathrm { N } ) < \infty$ , then $\mathrm { P } \ll \mathbf { N }$ , so $\mu \in S _ { \mathbf { N } }$ and $\Phi ( { \boldsymbol { \mu } } ) \in \Phi ( S _ { \mathbf { N } } )$ for P-almost every $\mu .$ Hence $\mathbb { E } _ { \mathrm { P } } [ \Phi ]$ lies in the closed convex set $\kappa _ { \Phi }$ , and no such $\mathrm { P }$ is admissible when $c \notin { K _ { \Phi } }$

2.2. The law of the last token for a general statistic Φ. We conclude this preliminary section by computing the law of the last token under the unique minimizer $\mathrm { P } ^ { \Phi }$ of (2.1), when considering a general statistic Φ and observed value $c \in \operatorname { i n t } ( \mathcal { K } _ { \Phi } )$ . According to Theorem 2.1, we can write $\mathrm { P } ^ { \Phi } \propto \exp ( \langle V ^ { \Phi } , \bar { \Phi } \rangle )$ N (for consistency with the Introduction, let $V ^ { \Phi } = \eta _ { c }$ be the Lagrange multiplier appearing in Theorem 2.1). Clearly, $\mathrm { P } ^ { \Phi }$ is equivalent to N; hence, under $\mathrm { P } ^ { \Phi }$ , the sentence $\mu$ almost surely has $N$ pairwise distinct atoms. As in the sampling procedure of the Introduction, the tokens of a sentence are its atoms read in a uniformly random order: given $\mu , ( X _ { 1 } , \ldots , X _ { N } )$ is a uniformly random ordering of the atoms of $\mu .$

Before resuming our discussion, let us briefly comment on the conditional predictive law of the last token $X _ { n }$ given $X _ { 1 : n - 1 }$ , which exists by disintegration as a probability kernel in $x _ { 1 : n - 1 }$ [Kle14, Theorem 8.37]. Hereafter, given $N = n ,$ , the conditional law of $X _ { n }$ given $X _ { 1 : n - 1 }$ is understood as the regular conditional distribution version given by (2.3), which is defined for every context $x _ { 1 : n - 1 }$ . In particular, when $\Phi$ is weakly continuous, this is the unique continuous version.

Proposition 2.2 (The tokens and the last token). Let $\mathcal { Z } = \mathbb { E } _ { \mathbf { N } } [ e ^ { \langle V ^ { \Phi } , \Phi \rangle } ]$

(a) $\mathrm { P } ^ { \Phi } \big ( N = m , \ X _ { 1 : m } \in \mathrm { d } x _ { 1 : m } \big ) = \mathcal { Z } ^ { - 1 } \nu ( m ) e ^ { \langle V ^ { \Phi } , \Phi ( L _ { m } ( x _ { 1 : m } ) ) \rangle } \underline { { \mathrm { v o l } } } ^ { \otimes m } ( \mathrm { d } x _ { 1 : m } )$ , and $\mu = L _ { N } ( X _ { 1 : N } )$

(b) Let n  1 with $\nu ( n ) > 0$ . Given $N = n$ and $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , the last token $X _ { n }$ has density with respect to vol

$$
p ( y ) = \frac { e ^ { \langle V ^ { \Phi } , \Phi ( \mu _ { y } ) \rangle } } { \int e ^ { \langle V ^ { \Phi } , \Phi ( \mu _ { y ^ { \prime } } ) \rangle } \underline { { \mathrm { v o l } } } ( \mathrm { d } y ^ { \prime } ) } , \qquad \mu _ { y } = \frac { 1 } { n } \Big ( \sum _ { i < n } \delta _ { x _ { i } } + \delta _ { y } \Big ) .\tag{2.3}
$$

Proof. Claim (a) directly follows from the definition of N and $\mathrm { P } ^ { \Phi }$ . Claim (b) follows from an application of Bayes’ formula to (a) with $m = n \colon$ the conditional density of $X _ { n }$ given $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ and $N = n$ is proportional to $y \mapsto e ^ { \langle V ^ { \Phi } , \Phi ( L _ { n } ( x _ { 1 : n - 1 } , y ) ) \rangle }$ , and $L _ { n } ( x _ { 1 : n - 1 } , y ) = \mu _ { y }$ ■

Next, we are interested in computing the score associated to the predictive law (2.3). Indeed, having access to this score allows one to generate last-token samples via a Langevin sampler. More precisely, if log $p ( y ) \in \mathcal { C } ^ { 1 } ( \mathbb { S } ^ { d } )$ with Lipschitz gradient (as will be the case in our model, see Proposition 3.15) then we can use (a discretization of) the Langevin dynamics $\mathrm { d } Y _ { t } = \nabla _ { \mathbb { S } ^ { d } } \log p ( Y _ { t } ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } W _ { t } ^ { \mathbb { S } ^ { d } }$ on $\mathbb { S } ^ { d } .$ , with W a Brownian motion on $\mathbb { S } ^ { d }$ with generator $\frac { 1 } { 2 } \Delta _ { \mathbb { S } ^ { d } }$ , for sampling from p vol. Indeed it is a reversible and ergodic SDE, with invariant law p vol [BGL14], i.e. its law converges to the law of the last token.

Lemma 2.3 (Score of the last token). Let $\Phi = ( \phi _ { 1 } , \dots , \phi _ { k } ) : \mathcal { P } ( \mathbb { S } ^ { d } ) \to \mathbb { R } ^ { k }$ and $V \in \mathbb { R } ^ { k }$ , and assume that each component $\phi _ { \ell }$ is weakly continuous and satisfies the following two conditions:

(i) (Gâteaux diferentiability) There is a bounded measurable function $( \mu , z ) \mapsto \delta \phi _ { \ell } ( \mu ) ( z ) ~ o n ~ \mathcal { P } ( \mathbb { S } ^ { d } ) \times \mathbb { S } ^ { d }$ such that, for all $\mu , \varrho \in \mathcal { P } ( \mathbb { S } ^ { d } )$ , the map $t \mapsto \phi _ { \ell } ( \mu + t ( \varrho - \mu ) )$ is diferentiable on (0, 1) with

$$
\frac { \mathrm { d } } { \mathrm { d } t } \phi _ { \ell } \bigl ( \mu + t ( \varrho - \mu ) \bigr ) = \int \delta \phi _ { \ell } \bigl ( \mu + t ( \varrho - \mu ) \bigr ) \mathrm { d } ( \varrho - \mu ) .
$$

The function $\delta \phi _ { \ell } ( \mu )$ denotes the first variation of $\phi _ { \ell }$ at $\mu .$

(ii) (Continuous gradient of the first variation) For every $\mu ,$ the function $\delta \phi _ { \ell } ( \mu )$ is the restriction to $\mathbb { S } ^ { d }$ of a ${ \dot { C } } ^ { 1 }$ function defined on a neighborhood $o \dot { f } \mathbb { S } ^ { d }$ in $\mathbb { R } ^ { d + 1 }$ , whose gradient $( \mu , z ) \mapsto \nabla _ { z } \delta \phi _ { \ell } ( \mu ) ( z )$ is jointly continuous on $\mathcal { P } ( \mathbb { S } ^ { d } ) \times \mathbb { S } ^ { d }$ , where ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ carries the weak topology.

Write δ ${ \bf \dot { \Phi } } = ( \delta \phi _ { 1 } , \dots , \delta \phi _ { k } )$ . Let $n \ge 1 , x _ { 1 : n - 1 } \in ( \mathbb { S } ^ { d } ) ^ { n - 1 }$ and $\begin{array} { r } { \mu _ { y } = \frac { 1 } { n } \big ( \sum _ { i < n } \delta _ { x _ { i } } + \delta _ { y } \big ) } \end{array}$ for $\boldsymbol { y } \in \mathbb { S } ^ { d }$ . Then $y \mapsto \langle V , \Phi ( \mu _ { y } ) \rangle$ belongs to ${ \mathcal { C } } ^ { 1 } ( \mathbb { S } ^ { d } )$ , and its Riemannian gradient is

$$
\nabla _ { \mathbb { S } ^ { d } } \left. V , \Phi ( \mu _ { y } ) \right. = \frac { 1 } { n } \left. P _ { y } ^ { \perp } \nabla _ { z } \langle V , \delta \Phi ( \mu _ { y } ) ( z ) \rangle \right| _ { z = y } , \qquad P _ { y } ^ { \perp } = | - y y ^ { \top } .\tag{2.4}
$$

In particular, $i f \nu ( n ) > 0$ , the score of the last-token density p of Proposition $\it { 2 . 2 - ( b ) }$ is

$$
\nabla _ { \mathbb { S } ^ { d } } \log p ( y ) = \frac { 1 } { n } \left. P _ { y } ^ { \perp } \nabla _ { z } \langle V ^ { \Phi } , \delta \Phi ( \mu _ { y } ) ( z ) \rangle \right| _ { z = y } .
$$

Here $\nabla _ { z }$ is the gradient in the variable z with the measure $\mu _ { y }$ frozen; only its tangential part $P _ { y } ^ { \perp } \nabla _ { z }$ is intrinsic, as it does not depend on the $\mathcal { C } ^ { 1 }$ extension chosen in (ii). In the rest of the paper whenever we say that $\Phi \colon { \mathcal { P } } ( \mathbb { S } ^ { d } ) \to \mathbb { R } ^ { k }$ is a $\mathcal { C } ^ { 1 }$ function, we tacitly mean that it satisfies the assumptions of Lemma 2.3.

Proof. Firstly, let $\begin{array} { r } { d _ { \mathrm { B L } } ( \mu , \mu ^ { \prime } ) = \operatorname* { s u p } \big \{ \big | \int f \mathrm { d } ( \mu - \mu ^ { \prime } ) \big | \colon \| f \| _ { \infty } \leq 1 , \mathrm { L i p } ( f ) \leq 1 \big \} } \end{array}$ be the bounded-Lipschitz distance, which metrizes the weak topology on ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ . By linearity it sufices to prove the claim for a single component $\phi = \phi _ { \ell } , \ell \in \{ 1 , \ldots , k \}$ . Let $y \in \mathbb { S } ^ { d } , v \in T _ { y } \mathbb { S } ^ { d }$ , and let $\gamma : ( - 1 , 1 ) \to \mathbb { S } ^ { d }$ be a $\mathcal { C } ^ { 1 }$ curve with $\gamma ( 0 ) = y$ and $\gamma ^ { \prime } ( 0 ) = v$ For $s \neq 0$ , consider the signed measure $\begin{array} { r } { \mu _ { \gamma ( s ) } - \mu _ { y } = \frac { 1 } { n } ( \delta _ { \gamma ( s ) } - \delta _ { y } ) } \end{array}$ , and the segment $\mu _ { t } ^ { s } = \mu _ { y } + t ( \mu _ { \gamma ( s ) } - \mu _ { y } ) \in \mathcal { P } ( \mathbb { S } ^ { d } )$ for $t \in [ 0 , 1 ]$

Now, consider $g ( t ) = \phi ( \mu _ { t } ^ { s } )$ . Since $t \mapsto \mu _ { t } ^ { s }$ is weakly continuous on [0, 1] and $\phi$ is weakly continuous, $g$ is continuous on $[ 0 , 1 ]$ , and by (i) it is diferentiable on $( 0 , 1 )$ . By the mean value theorem there is $t _ { s } \in ( 0 , 1 )$ with $g ( 1 ) - g ( 0 ) = g ^ { \prime } ( t _ { s } )$ . Hence, by (i) and $\begin{array} { r } { \mu _ { \gamma ( s ) } - \mu _ { y } = \frac { 1 } { n } ( \delta _ { \gamma ( s ) } - \delta _ { y } ) } \end{array}$ ,

$$
\begin{array} { r l } & { \displaystyle \frac { \phi ( \mu _ { \gamma ( s ) } ) - \phi ( \mu _ { y } ) } { s } = \frac { 1 } { s } \int \delta \phi ( \mu _ { t _ { s } } ^ { s } ) \mathrm { d } ( \mu _ { \gamma ( s ) } - \mu _ { y } ) = \frac { 1 } { n } \frac { 1 } { s } \Big ( \delta \phi ( \mu _ { t _ { s } } ^ { s } ) \big ( \gamma ( s ) \big ) - \delta \phi ( \mu _ { t _ { s } } ^ { s } ) ( y ) \Big ) } \\ & { \qquad \quad = \frac { 1 } { n } \frac { 1 } { s } \int _ { 0 } ^ { s } \langle \nabla _ { z } \delta \phi ( \mu _ { t _ { s } } ^ { s } ) \big ( \gamma ( r ) \big ) , \gamma ^ { \prime } ( r ) \rangle \mathrm { d } r , } \end{array}
$$

where the last step follows from the chain rule along $\gamma ~ ( \delta \phi ( \mu _ { t _ { s } } ^ { s } )$ is ${ \mathcal { C } } ^ { 1 }$ near $\mathbb { S } ^ { d } )$ . We claim that

$$
\operatorname* { l i m } _ { s \to 0 } \frac { \phi ( \mu _ { \gamma ( s ) } ) - \phi ( \mu _ { y } ) } { s } = \frac { 1 } { n } \langle \nabla _ { z } \delta \phi ( \mu _ { y } ) ( y ) , v \rangle = \langle \frac { 1 } { n } P _ { y } ^ { \perp } \nabla _ { z } \delta \phi ( \mu _ { y } ) ( y ) , v \rangle ,\tag{2.5}
$$

where the last equality holds because $v \in T _ { y } \mathbb { S } ^ { d }$ , so that $\langle w , v \rangle = \langle P _ { u } ^ { \perp } w , v \rangle$ for every w $\in \mathbb { R } ^ { d + 1 }$ . In order to show the validity of the limit in (2.5), it is enough to argue that uniformly in $t \in [ 0 , 1 ]$

$$
\begin{array} { r } { \underset { t \in [ 0 , 1 ] } { \operatorname* { s u p } } d _ { \mathrm { B L } } ( \mu _ { t } ^ { s } , \mu _ { y } ) \leq \frac { 1 } { n } | \gamma ( s ) - y | \underset { s  0 } { \longrightarrow } 0 , } \end{array}
$$

and combine it with the fact that $( \mu , z ) \mapsto \nabla _ { z } \delta \phi ( \mu ) ( z )$ is continuous on the compact metric space $\mathcal { P } ( \mathbb { S } ^ { d } ) \times \mathbb { S } ^ { d }$ 2 hence bounded, and uniformly continuous, with a modulus of continuity $\varpi .$ . Indeed, for $t \in [ 0 , 1 ]$ and $| r | \leq | s |$ it then follows that

$$
\begin{array} { r } { \left. \langle \nabla _ { z } \delta \phi ( \mu _ { t _ { s } } ^ { s } ) \big ( \gamma ( r ) \big ) , \gamma ^ { \prime } ( r ) \rangle - \langle \nabla _ { z } \delta \phi ( \mu _ { y } ) ( y ) , v \rangle \right. \leq \varpi \Big ( \frac { 1 } { n } \left. \gamma ( s ) - y \right. + \underset { \left. r \right. \leq \left. s \right. } { \operatorname* { s u p } } \left. \gamma ( r ) - y \right. \Big ) \underset { \left. r \right. \leq \left. s \right. } { \operatorname* { s u p } } \left. \gamma ^ { \prime } ( r ) \right. } \\ { + \| \nabla _ { z } \delta \phi ( \mu ) ( z ) \| _ { \mathrm { L } ^ { \infty } ( \mathcal { P } ( \mathbb { S } ^ { d } ) \times \mathbb { S } ^ { d } ) } \underset { \left. r \right. \leq \left. s \right. } { \operatorname* { s u p } } \left. \gamma ^ { \prime } ( r ) - v \right. , } \end{array}
$$

and the right-hand side tends to 0 as $s \to 0$ , by the continuity of $\gamma$ and $\gamma ^ { \prime }$ , which implies (2.5). This shows the validity of the directional version of (2.4) (along a direction $v \in T _ { y } \mathbb { S } ^ { d } )$ for one fixed $\phi = \phi _ { \ell }$

Set $\begin{array} { r } { g _ { \ell } ( y ) = \frac { 1 } { n } P _ { y } ^ { \perp } \nabla _ { z } \delta \phi _ { \ell } ( \mu _ { y } ) ( y ) \mathrm { ~ a n d ~ } G ( y ) = \sum _ { \ell } V _ { \ell } g _ { \ell } ( y ) = \frac { 1 } { n } P _ { y } ^ { \perp } \nabla _ { z } \langle V , \delta \Phi ( \mu _ { y } ) ( z ) \rangle \big | _ { z = y } . } \end{array}$ By linearity, the derivative of $y \mapsto \langle V , \Phi ( \mu _ { y } ) \rangle$ along every $\mathcal { C } ^ { 1 }$ curve through y with velocity v equals $\langle \dot { G } ( y ) , v \rangle$ . Then, since $y \mapsto ( \mu _ { y } , y )$ is continuous and $\nabla _ { z } \delta \phi _ { \ell }$ jointly continuous, we conclude that $y \mapsto \langle V , \Phi ( \mu _ { y } ) \rangle \in \mathcal { C } ^ { 1 } ( \mathbb { S } ^ { d } )$ , and its Riemannian gradient is $G .$ Finally, since log $p ( y ) = \langle V ^ { \Phi } , \Phi ( \mu _ { y } ) \rangle + \mathrm { c o n s t }$ by (2.3), the formula for the score of p follows by taking $V = V ^ { \Phi }$ ■

## 3. The choice of the statistics: emergence of attention choice rules

The aim of this section is to prove the claims of Section 1.2, by relying on Lemma 2.3 where we have shown that the statistic Φ acts on the last token through the gradient of its first variation at the completed sentence. We first show that polynomial (in particular pair-additive) statistics only let a new token feel the sentence through fixed, unnormalized weights, and through its mean in the case of the covariance. This leads us to discard such statistics since linguistic structure should be relational, that is, the role of a token should be determined by which other tokens of the sentence it relates to the most. We then build in Section 3.1 a statistic that takes into account this linguistic structure by introducing the concept of choice rule for the sentence µ as a choice mechanism between keys and queries within the same sentence $\mu .$ Finally, we derive the attention choice rule from Luce’s choice axiom and a harmonic truncation, characterize it by entropy (Section 3.2), and define the statistic $\Phi = S _ { \Lambda }$ as its attention cross-moment. For presentation’s purposes we postpone the non-degeneracy of $S _ { \Lambda }$ , needed in the KL minimization (cf. Theorem 2.1), to Section A. Then, in Section 3.3 we solve the entropic problem for a fixed attention matrix Λ (1.4) for $\Phi = S _ { \Lambda }$ , compute the first variation of the attention readout and the exact law and score of the last token. We conclude with a brief comment on training of the attention and value matrices Λ and V, and on the possible use of the free energy as a diferent statistic (Section 3.4).

Proposition 3.1 (Polynomial and moment statistics in the measure). The following statements hold true.

(a) Fix a symmetric function $f \in \mathcal { C } ( ( \mathbb { S } ^ { d } ) ^ { r } )$ , with $r \geq 1$ . The corresponding polynomial statistic $\Phi ( \mu ) : = $ $\int f \mathrm { d } \mu ^ { \otimes r }$ has first variation $\begin{array} { r } { \delta \Phi ( \mu ) ( z ) = r \int f ( z , x _ { 2 } , \ldots , x _ { r } ) \mu ^ { \otimes ( r - 1 ) } ( \mathrm { d } x _ { 2 : r } ) } \end{array}$ . If f is $\mathcal { C } ^ { 1 }$ on a neighborhood of $( \mathbb { S } ^ { d } ) ^ { r }$ in $( \mathbb { R } ^ { d + 1 } ) ^ { r }$ , then Φ $i s \ { \mathcal { C } } ^ { 1 }$ (in the sense of Lemma 2.3). The score of the associated last-token law is an average, over the $( r - 1 )$ -tuples of tokens of the completed sentence, of $r \nabla _ { z } f ( z , x _ { 2 } , \ldots , x _ { r } ) | _ { z = y } .$ Each tuple enters with the same weight $n ^ { - ( r - 1 ) }$ , which depends only on the length of the sentence, and its contribution does not depend on the other tuples. There is no query-dependent normalization, as in the transformer architecture (1.1), and hence no competition between keys.

(b) For $r \ = \ 2$ and $h \in \mathcal { C } ( \mathbb { S } ^ { d } \times \mathbb { S } ^ { d } )$ , not necessarily symmetric, $\begin{array} { r } { \Phi _ { h } ( \mu ) = \int h \mathrm { d } \mu \mathrm { d } \mu } \end{array}$ has first variation δΦ $\begin{array} { r } { { \bf \Pi } _ { h } ( \mu ) ( z ) = \int \big ( h ( z , x ) + h ( x , z ) \big ) \mu ( \mathrm { d } x ) } \end{array}$

(c) $F o r \ h ( x , y ) = x ^ { \top }$ Ay (moment statistics), $\Phi _ { A } ( \mu ) = m _ { \mu } ^ { \top } A m _ { \mu }$ with $\begin{array} { r } { m _ { \mu } = \int \boldsymbol { x } \mu ( \mathrm { d } \boldsymbol { x } ) } \end{array}$ , and $\delta \Phi _ { A } ( \mu ) ( z ) =$ $z ^ { \top } ( A + A ^ { \top } ) m _ { \mu }$ . The force $\nabla _ { z } \delta \Phi _ { A } ( \mu ) ( z ) = ( A + A ^ { \top } ) m _ { \mu }$ is the same for every candidate, and the context enters only through its mean: the query attends to the empirical mean of the sentence, not to single keys.

(d) (Kernel sums) Let $\Lambda \in \mathbb { M } _ { d + 1 }$ and $h ( x , y ) = k _ { \Lambda } ( x , y ) = e ^ { x ^ { \prime } \Lambda y }$ , so that

$$
\Phi _ { k \wedge } ( \mu ) = \iint e ^ { x ^ { \top } \Lambda y } \mu ( \mathrm { d } x ) \mu ( \mathrm { d } y ) , \qquad \delta \Phi _ { k \wedge } ( \mu ) ( z ) = \int \left( e ^ { z ^ { \top } \Lambda x } + e ^ { x ^ { \top } \Lambda z } \right) \mu ( \mathrm { d } x ) ,
$$

and

$$
\nabla _ { z } \delta \Phi _ { k _ { \Lambda } } ( \mu ) ( z ) = \int { \left( { e ^ { z ^ { \top } \Lambda x } } \Lambda x + { e ^ { x ^ { \top } \Lambda z } } \Lambda ^ { \top } x \right) \mu ( \mathrm { d } x ) } .
$$

At the completed sentence $\begin{array} { r } { \mu _ { y } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \delta _ { x _ { j } } , } \end{array}$ with $x _ { n } = y$ , the score of the last token under the tilt $e ^ { \Phi _ { k _ { i } } }$ Λ N (Lagrange multiplier equal to 1) is therefore

$$
\nabla _ { \mathbb { S } ^ { d } } \log p ( y ) = \frac { 1 } { n ^ { 2 } } P _ { y } ^ { \perp } \sum _ { j = 1 } ^ { n } \Big ( e ^ { y ^ { \top } \Lambda x _ { j } } \Lambda x _ { j } + e ^ { x _ { j } ^ { \top } \Lambda y } \Lambda ^ { \top } x _ { j } \Big ) .
$$

Every key $x _ { j }$ acts with a weight that does not depend on the other keys, $e ^ { y ^ { \top } \Lambda x _ { j } }$ (as a key of the query y) and $e ^ { x _ { j } ^ { \top } \Lambda y }$ (as a query reading y). The weights are the unnormalized version of (1.1). Without normalization the weights are not a probability, so that adding a key never decreases the contribution of the others, and the force grows with the number of keys similar to y. Moreover the vector read from a key is $\Lambda x _ { j }$ , so that values are tied to the query–key matrix.

Remark 3.2 (Connection with unnormalized self-attention). Item $( d )$ is the statistical counterpart of the unnormalized self-attention dynamics of [GLPR25, Section 1.2.3]. For $\Lambda = \beta \mathsf { I }$ , the kernel sum $\Phi _ { k _ { \beta 1 } } ( \mu ) =$ $\begin{array} { r l } { \int e ^ { \beta \langle x , x ^ { \prime } \rangle } \mu ( \mathrm { d } x ) \mu ( \mathrm { d } x ^ { \prime } ) } \end{array}$ equals $2 \beta \operatorname { E } _ { \beta } [ \mu ]$ , where $\operatorname { E } _ { \beta }$ is the interaction energy of [GLPR25, Eq. (1.13)]. There the authors consider the unnormalized dynamics (USA), $\begin{array} { r } { \dot { x } _ { i } = P _ { x _ { i } } ^ { \perp } \big ( \frac { 1 } { n } \sum _ { j } e ^ { \beta \langle x _ { i } , x _ { j } \rangle } x _ { j } \big ) } \end{array}$ , which is the Wasserstein gradient flow of $\mathrm { E } _ { \beta } \ / G L P R { \ = } 5 ,$ Lemma 1.8]. Accordingly, for the tilt $\mathrm { P } \propto \stackrel { \cdot } { e } ^ { \lambda \Phi _ { k _ { \beta 1 } } } \mathbf { N } .$ , Lemma 2.3 and (d) show that the score of the last token is $\textstyle { \frac { 2 \lambda \beta } { n } }$ times the (USA) vector field evaluated at the candidate y, over the completed sentence $\mu _ { y }$ . The softmax normalization of attention divides this field by the scalar partition function (cf. (1.1)), and does not change its direction: in $I G L P R { \mathcal { Z } } 5 ,$ Section $\ 1 . \mathcal { Q } . \mathcal { 4 } J$ the normalized dynamics is recovered as a gradient flow of the same energy for a metric reweighted by the partition functions. In our setting, we will show how the normalization enters instead through the choice rule of Section ${ \it 3 . 1 , }$ at the level of the statistic.

Proof. (a) By symmetry and multilinearity, $\begin{array} { r } { t \mapsto \Phi ( \mu + t ( \varrho - \mu ) ) = \sum _ { k = 0 } ^ { r } \binom { r } { k } t ^ { k } \int f \mathrm { d } ( \varrho - \mu ) ^ { \otimes k } \otimes \mu ^ { \otimes ( r - k ) } } \end{array}$ is a polynomial. Since f is symmetric, we have $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \Phi ( \mu _ { t } ) = r \int f \mathrm { d } ( \varrho - \mu ) \otimes \mu _ { t } ^ { \otimes ( r - 1 ) } } \end{array}$ , with $\mu _ { t } = \mu + t ( \varrho - \mu )$ . Therefore, $\begin{array} { r } { \delta \Phi ( \mu ) ( z ) = r \int f ( z , \cdot ) \mathrm { d } \mu ^ { \otimes ( r - 1 ) } } \end{array}$ . The function $\delta \Phi$ is bounded by $r \left\| f \right\| _ { \infty } . \mathrm { ~ H ~ } f \in { \mathcal { C } } ^ { 1 }$ , then the gradient of the first variation $r \int \bar { \nabla } _ { z } f ( z , \cdot ) \mathrm { d } \mu ^ { \otimes ( r - 1 ) }$ is bounded and jointly continuous in $( \mu , z )$

Item (b) can be obtained from (a) applied to the symmetrization of h. Items (c) and (d) follow from direct computations. ■

Example 3.3 (Covariance). Let $\begin{array} { r } { \Phi ( \mu ) = \operatorname { C o v } ( \mu ) = \int x x ^ { \top } \mathrm { d } \mu - m _ { \mu } m _ { \mu } ^ { \top } } \end{array}$ and $V \in \mathbb { M } _ { d + 1 }$ . By Proposition ${ 3 . 1 } ,$ applied to $h ( x , y ) = x ^ { \top } V x$ , which depends on one variable, and to $h ( x , \dot { y } ) = x ^ { \top } V y$

$$
\delta \langle V , \mathrm { C o v } \rangle ( \mu ) ( z ) = z ^ { \top } V z - z ^ { \top } ( V + V ^ { \top } ) m _ { \mu } + \mathrm { c o n s t } , \qquad \nabla _ { z } \delta \langle V , \mathrm { C o v } \rangle ( \mu ) ( z ) = ( V + V ^ { \top } ) ( z - m _ { \mu } ) .
$$

At the completed sentence $\mu _ { y }$ of $\begin{array} { r } { ( 2 . 3 ) , y - m _ { \mu _ { y } } = \frac { n - 1 } { n } ( y - \bar { x } ) } \end{array}$ with $\begin{array} { r } { \bar { x } = \frac { 1 } { n - 1 } \sum _ { i < n } x _ { i } } \end{array}$ . Hence

$$
\nabla _ { z } \delta \langle V , \mathrm { C o v } \rangle ( \mu _ { y } ) ( z ) \big | _ { z = y } = \frac { n - 1 } { n } ( V + V ^ { \top } ) \big ( y - \bar { x } \big ) ,
$$

which is the formula of the introduction. Its tangential part, divided by $n ,$ is the score of the last token (Lemma 2.3).   
The context acts only through its mean x¯.

3.1. Choosing keys: the lowest-order relational choice rule. As we have already mentioned in the introduction, since linguistic structure is relational we introduce the concept of choice rule for the sentence $\mu ,$ which is a coupling in

$$
\Pi _ { 1 } ( \mu ) = \big \{ \pi \in \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { S } ^ { d } ) : \pi ( \cdot \times \mathbb { S } ^ { d } ) = \mu \big \} ,
$$

the first coordinate being the query and the second the key; we write $\pi ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \pi _ { x } ( \mathrm { d } y )$ . Among all possible choice rules, the weakest form of competition we can consider in our model is Luce’s choice axiom $\mathrm { [ L u c 5 9 ] }$ where the odds of choosing the key $y$ rather than $y ^ { \prime }$ depend on the query and on the two keys, and not on the other tokens of the sentence. Under this assumption we are guaranteed the existence of a choice function $h : \mathbb { S } ^ { d } \times \mathbb { S } ^ { d }  \mathbb { R }$ , the same for all sentences, such that for any sentence $\mu$ the associated choice rule is

$$
\pi _ { \mu } ^ { h } ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \frac { e ^ { h ( x , y ) } \mu ( \mathrm { d } y ) } { \int e ^ { h ( x , y ^ { \prime } ) } \mu ( \mathrm { d } y ^ { \prime } ) } .\tag{3.1}
$$

The choice function encodes relational preferences and syntactical relations between tokens, and in particular diferent choice functions might focus on diferent aspects of our language. Notice that the rule depends on h only through the log-odds $h ( x , y ) - h ( x , y ^ { \prime } )$ , so the choice function is defined up to the gauge shifts $h \mapsto h + a ( x )$

In what follows we let Y<sub>ℓ</sub> denote the set of spherical harmonics of degree ℓ on $\mathbb { S } ^ { d }$ , so that $\mathcal { I } _ { 0 }$ are the constants and $\mathcal { Y } _ { 1 } = \{ x \mapsto v ^ { \top } x \}$ . Finally, for any choice function $h \in \mathrm { L } ^ { 2 } ( \underline { { \mathrm { v o l } } } \otimes \underline { { \mathrm { \bar { v o l } } } } )$ we consider its harmonic expansion $\begin{array} { r } { h = \sum _ { \ell , \ell ^ { \prime } \geq 0 } h _ { \ell \ell ^ { \prime } } } \end{array}$ with $h _ { \ell \ell ^ { \prime } } \in \mathcal { V } _ { \ell } \otimes \mathcal { V } _ { \ell ^ { \prime } }$

Proposition 3.4 (The lowest-order relational interaction). Let $h , h ^ { \prime } \in \mathcal { C } ( \mathbb { S } ^ { d } \times \mathbb { S } ^ { d } )$

(a) $\pi _ { \mu } ^ { h } = \pi _ { \mu } ^ { h ^ { \prime } }$ for every $\mu \in \mathcal P ( \mathbb { S } ^ { d } )$ if and only $i f h { - } h ^ { \prime }$ depends on the query only, that is, $h - h ^ { \prime } \in \bigoplus _ { \ell > 0 } \mathcal { W } _ { \ell } \otimes \mathcal { W } _ { 0 } .$ the gauge removes exactly the components $h _ { \ell 0 }$

(b) The components $h _ { 0 \ell ^ { \prime } } , \ell ^ { \prime } \geq 1$ , depend on the key only: they shift the log-odds by the same amount for every query, a key bias without relational content.

(c) The relational part $\begin{array} { r } { h _ { \mathrm { r e l } } = \sum _ { \ell , \ell ^ { \prime } > 1 } h _ { \ell \ell ^ { \prime } } } \end{array}$ is gauge invariant, and its component of lowest total degree is $h _ { 1 1 } \in \mathcal { V } _ { 1 } \otimes \mathcal { V } _ { 1 } = \{ ( x , y ) \mapsto \langle x , \overset { \cdot } { \Lambda } \overset { \cdot } { y } \rangle : \ \Lambda \in \mathbb { M } _ { d + 1 } \}$

Proof. Let us first show item (a). Clearly, if $h - h ^ { \prime } = a ( x )$ , the factor $e ^ { a ( x ) }$ cancels in (3.1). Conversely, let $x \neq y$ and $\begin{array} { r } { \mu = \frac { 1 } { 2 } ( \delta _ { x } + \delta _ { y } ) } \end{array}$ . The query x chooses the key $z \in \{ x , y \}$ with probability $e ^ { h ( \stackrel { . } { x } , z ) } / ( e ^ { h ( x , x ) } + e ^ { h ( \stackrel { . } { ( x , y ) } ) } )$ and equality of the two rules gives $\begin{array} { r } { h ( x , z ) - h ^ { \prime } ( x , z ) = \log \frac { e ^ { h ( x , x ) } + e ^ { h ( x , y ) } } { e ^ { h ^ { \prime } ( x , x ) } + e ^ { h ^ { \prime } ( x , y ) } } } \end{array}$ for both $z = x$ and $z = y$ , hence $\boldsymbol { g } ( x , y ) = \boldsymbol { g } ( x , x )$ for $g = h - h ^ { \prime }$ . Since $y \neq x$ is arbitrary, $g ( x , \cdot )$ is constant for every x, that is, $g ( x , y ) = a ( x )$ with $a ( x ) = g ( x , x )$ . Finally, a function of the query alone has only components with $\ell ^ { \prime } = 0$ , and conversely. I+

Item (b) is immediate.

Item (c) can be proven as follows. Since the gauge changes only components with $\ell ^ { \prime } = 0$ , the relational part of the choice function $\begin{array} { r } { h _ { \mathrm { r e l } } = \sum _ { \ell , \ell ^ { \prime } > 1 } h _ { \ell \ell ^ { \prime } } } \end{array}$ is gauge invariant. Since the space Y consists of the restrictions of linear forms, we conclude that $\mathcal { Y } _ { 1 } \stackrel { - } { \otimes } \mathcal { H } _ { 1 }$ is the space of bilinear forms $\{ ( x , y ) \mapsto \langle x , \Lambda y \rangle : \Lambda \in \mathbb { M } _ { d + 1 } \}$

Therefore, there exists a matrix $\Lambda \in \mathbb { M } _ { d + 1 }$ such that the lowest-order term in the harmonic expansion equals $h _ { 1 1 } ( x , y ) = \langle x , \Lambda y \rangle \in \mathcal { Y } _ { 1 } \otimes \mathcal { Y } _ { 1 }$

Truncating the relational part of the choice function at its lowest order, and setting the key bias to zero, gives our lowest-order relational choice rule $\pi _ { \mu } ^ { \Lambda }$ of (1.7), with $k _ { \Lambda } ( x , y ) = e ^ { \langle x , \Lambda y \rangle } , \Lambda \in \mathbb { M } _ { d + 1 } \colon$

$$
\pi _ { \mu } ^ { \Lambda } ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \frac { k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) } { Z _ { \mu } ^ { \Lambda } ( x ) } , \qquad Z _ { \mu } ^ { \Lambda } ( x ) = \int k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) .\tag{3.2}
$$

The query x chooses its key by softmax attention with query–key matrix Λ, and $\Lambda = 0$ is the uniform choice. We stress that this truncation is a modeling choice. The component $h _ { 1 1 }$ is the relational component of lowest degree, not in general the dominant one, and we plan to address this point of view in future work.

3.2. Entropy: the lowest-order rule as an exponential family. For a fixed sentence, the rules (3.2) form an exponential family through the uniform choice,

$$
\frac { \mathrm { d } \pi _ { \mu } ^ { \Lambda } } { \mathrm { d } ( \mu \otimes \mu ) } ( x , y ) = \exp \big ( \langle \Lambda , x y ^ { \top } \rangle - \log Z _ { \mu } ^ { \Lambda } ( x ) \big ) , \qquad \Lambda \in \mathbb { M } _ { d + 1 } ,\tag{3.3}
$$

with natural parameter Λ and suficient statistic $x y ^ { \top }$ . Its log-partition function, integrated over the queries, is the attention free energy

$$
F _ { \Lambda } ( \mu ) = \int \log Z _ { \mu } ^ { \Lambda } ( x ) \mu ( \mathrm { d } x ) .\tag{3.4}
$$

For notational convenience we write $Z _ { \mu } = Z _ { \mu } ^ { \Lambda }$ . Finally, recall from (1.9) our definition of attention vector field and attention cross-moment associated to Λ:

$$
\mathrm { A t t n } _ { \Lambda } ( x ; \mu ) : = \mathbb { E } _ { \pi _ { \mu } ^ { \Lambda } } [ Y | X = x ] = \frac { \int y k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) } { Z _ { \mu } ^ { \Lambda } ( x ) } , \quad S _ { \Lambda } ( \mu ) : = \mathbb { E } _ { \pi _ { \mu } ^ { \Lambda } } [ X Y ^ { \top } ] = \int x y ^ { \top } \mathrm { d } \pi _ { \mu } ^ { \Lambda } .
$$

Note that $Z _ { \mu } ( x )$ and $\operatorname { A t t n } _ { \Lambda } ( x ; \mu )$ are defined for every $x \in \mathbb { R } ^ { d + 1 }$ , not only on the support of $\mu ,$ and that $e ^ { - \| \Lambda \| _ { \mathrm { o p } } } \leq Z _ { \mu } \leq e ^ { \| \Lambda \| _ { \mathrm { o p } } }$ on $\mathbb { S } ^ { d }$

Proposition 3.5 (Attention as semi-relaxed entropic self-transport). For every $\mu \in \mathcal P ( \mathbb S ^ { d } )$ and $\Lambda \in \mathbb { M } _ { d + 1 }$ consider the entropic optimization problem

$$
F _ { \Lambda } ( \mu ) = \operatorname* { m a x } _ { \pi \in \Pi _ { 1 } ( \mu ) } \left\{ \int x ^ { \top } \Lambda y \pi ( \mathrm { d } x , \mathrm { d } y ) - \mathrm { K L } ( \pi | \mu \otimes \mu ) \right\} .\tag{3.5}
$$

Then, $\pi _ { \mu } ^ { \Lambda }$ corresponds to the unique maximizer of this problem. Moreover:

(a) $\mathcal { H } _ { \Lambda } ( \mu ) : = \mathrm { K L } ( \pi _ { \mu } ^ { \Lambda } | \mu \otimes \mu ) = \langle \Lambda , S _ { \Lambda } ( \mu ) \rangle - F _ { \Lambda } ( \mu )$ , with $\begin{array} { r } { S _ { \Lambda } ( \mu ) = \int \ d x y ^ { \top } \mathrm d \pi _ { \mu } ^ { \Lambda } } \end{array}$ the attention cross-moment; (b) $\Lambda \mapsto F _ { \Lambda } ( \mu )$ is convex and smooth, with $\nabla _ { \Lambda } F _ { \Lambda } ( { \boldsymbol \mu } ) = S _ { \Lambda } ( { \boldsymbol \mu } ) ,$

(c) $\nabla _ { x } \log Z _ { \mu } ^ { \Lambda } ( x ) = \Lambda \mathrm { A t t n } _ { \Lambda } ( x ; \mu )$ , whose tangential part $P _ { x } ^ { \perp } \Lambda \mathrm { A t t n } _ { \Lambda } ( x ; \mu )$ is the self-attention vector field on $\mathbb { S } ^ { d }$ with value matrix Λ;

(d) the Jacobian of $x \mapsto \operatorname { A t t n } _ { \Lambda } ( x ; \mu )$ is $\mathrm { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | x ) } ( Y ) \Lambda ^ { \top }$ , where, with a slight abuse of notation, for any $x \in \mathbb { S } ^ { d }$ we write $\pi _ { \mu } ^ { \Lambda } ( \cdot | x ) ( \mathrm { d } y ) = k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) / Z _ { \mu } ( x )$ for the law of the key chosen by the query x, and $\operatorname { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | x ) } ( Y )$ is its covariance.

Proof. Let us first notice that for any $\pi \in \Pi _ { 1 } ( \mu )$ we have

$$
\int x ^ { \top } \Lambda y \pi ( \mathrm { d } x , \mathrm { d } y ) - \mathrm { K L } ( \pi \mid \mu \otimes \mu ) = \int \log Z _ { \mu } ( x ) \pi ( \mathrm { d } x , \mathrm { d } y ) - \mathrm { K L } ( \pi \mid \pi _ { \mu } ^ { \Lambda } ) = F _ { \Lambda } ( \mu ) - \mathrm { K L } ( \pi \mid \pi _ { \mu } ^ { \Lambda } ) .
$$

Since $\mathrm { K L } ( \pi | \pi _ { \mu } ^ { \Lambda } ) \ge 0$ , with equality if and only $\mathrm { i f } \ \pi = \pi _ { \mu } ^ { \Lambda }$ , this shows that the supremum in (3.5) equals $F _ { \Lambda } ( \mu )$ and is attained only at $\pi _ { \mu } ^ { \Lambda }$ . Taking $\pi = \pi _ { \mu } ^ { \Lambda }$ gives $\langle \Lambda , \dot { S } _ { \Lambda } ( \mu ) \rangle - \mathcal { H } _ { \Lambda } ( \mu ) = F _ { \Lambda } ( \mu )$ and hence Item (a), because $\begin{array} { r } { \int x ^ { \top } \Lambda y \mathrm { d } \pi _ { \mu } ^ { \Lambda } = \langle \Lambda , \int x y ^ { \top } \dot { \mathrm { d } \pi } _ { \mu } ^ { \Lambda } \rangle = \langle \Lambda , S _ { \Lambda } ( \mu ) \rangle } \end{array}$

Item (b) can be proven as follows. By $( 3 . 5 ) , \Lambda \mapsto F _ { \Lambda } ( \mu )$ is a supremum of afine functions over a set independent of Λ, hence convex. For smoothness, note that $\Lambda \mapsto k _ { \Lambda } ( x , y )$ is smooth with $\nabla _ { \Lambda } k _ { \Lambda } ( x , y ) = x y ^ { \top } k _ { \Lambda } ( x , y )$ , and all derivatives are bounded uniformly in $( x , y ) \in \mathbb { S } ^ { d } \times \mathbb { S } ^ { d }$ for Λ in a bounded set. Diferentiating under the integral sign, and using $Z _ { \mu } \geq e ^ { - \| \Lambda \| _ { \mathrm { o p } } }$ ,

$$
\nabla _ { \boldsymbol { \Lambda } } \log Z _ { \mu } ( x ) = \frac { \int x y ^ { \top } k _ { \boldsymbol { \Lambda } } ( x , y ) \mu (  { \mathrm { d } } y ) } { Z _ { \mu } ( x ) } = x \mathrm { A t t n } _ { \boldsymbol { \Lambda } } ( x ; \mu ) ^ { \top } , \qquad \nabla _ { \boldsymbol { \Lambda } } F _ { \boldsymbol { \Lambda } } ( \mu ) = \int x \mathrm { A t t n } _ { \boldsymbol { \Lambda } } ( x ; \mu ) ^ { \top } \mu (  { \mathrm { d } } x ) = S _ { \boldsymbol { \Lambda } } ( \mu ) .
$$

Item (c) follows in a similar way: $\nabla _ { x } k _ { \Lambda } ( x , y ) = k _ { \Lambda } ( x , y )$ Λy and hence

$$
\nabla _ { x } \log { Z _ { \mu } ^ { \Lambda } ( x ) } = \frac { \int \Lambda y k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) } { Z _ { \mu } ( x ) } = \Lambda \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) ,
$$

whose tangential part $P _ { x } ^ { \perp } \Lambda \mathrm { A t t n } _ { \Lambda } ( x ; \mu )$ is the self-attention vector field on $\mathbb { S } ^ { d }$ with value matrix Λ.

Item (d) is a direct consequence of the properties of exponential families. Indeed, the derivative of the mean of the exponential family $\mu _ { \theta } ( \mathrm { d } y ) \propto e ^ { \langle \theta , y \rangle } \mu ( \mathrm { d } y )$ with respect to its natural parameter θ is its covariance. Therefore, if we consider the exponential family $\mu _ { \Lambda } \tau _ { x } .$ we have $\begin{array} { r } { \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) = \int y \mu _ { \Lambda ^ { \top } x } ( \mathrm { d } y ) } \end{array}$ $\operatorname { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | x ) } ( Y ) =$ $\operatorname { C o v } _ { \mu _ { \Lambda } \tau _ { x } } ( Y )$ and, by the chain rule, the Jacobian of $x \mapsto \operatorname { A t t n } _ { \Lambda } ( x ; \mu )$ is then Cov $\mathop { \pi _ { \mu } ^ { \Lambda } } ( \cdot | x ) ^ { \left( Y \right) \Lambda ^ { \intercal } }$

Items (a) and (b) identify the attention cross-moment $S _ { \Lambda } ( \mu )$ as the mean parameter of the family, and $\mathcal { H } _ { \Lambda } = \left. \Lambda , S _ { \Lambda } \right. - F _ { \Lambda }$ as the Legendre transform of the free energy evaluated at $S _ { \Lambda } ( \mu )$ . The constrained counterpart of (3.5) is a conditional maximum-entropy problem. We state it for general bounded query and key features $\psi ( x ) \in \mathbb { R } ^ { p }$ and $\ b \chi ( y ) \in \mathbb { R } ^ { q }$ , which shows that our model can be further generalized to include feature embeddings. For the sake of clarity, in this paper we work with $\psi = \chi = \mathrm { i d }$ , which gives the rule (1.7).

Proposition 3.6 (Conditional maximum entropy). Let $\psi , \chi$ be bounded and measurable, $\Lambda \in \mathbb { R } ^ { p \times q }$ , and

$$
\pi _ { \mu } ^ { \Lambda } ( \mathrm { d } x , \mathrm { d } y ) = \mu ( \mathrm { d } x ) \frac { e ^ { \psi ( x ) ^ { \top } \Lambda \chi ( y ) } \mu ( \mathrm { d } y ) } { \int e ^ { \psi ( x ) ^ { \top } \Lambda \chi ( y ^ { \prime } ) } \mu ( \mathrm { d } y ^ { \prime } ) } , \qquad C _ { \Lambda } = \int \psi ( x ) \chi ( y ) ^ { \top } \pi _ { \mu } ^ { \Lambda } ( \mathrm { d } x , \mathrm { d } y ) .
$$

Then $\pi _ { \mu } ^ { \Lambda }$ is the unique minimizer of $\mathrm { K L } ( \pi | \mu \otimes \mu )$ over $\{ \pi \in \Pi _ { 1 } ( \mu ) : \int \psi \chi ^ { \top } \mathrm { d } \pi = C _ { \Lambda } \}$

Proof. By reasoning as in the previous proof, for $\pi \in \Pi _ { 1 } ( \mu )$ with finite entropy we can show that

$$
\begin{array} { r } { \mathrm { K L } ( \pi | \mu \otimes \mu ) = \mathrm { K L } ( \pi | \pi _ { \mu } ^ { \Lambda } ) + \langle \Lambda , \int \psi \chi ^ { \top } \mathrm { d } \pi \rangle - \int \log \Big ( \int e ^ { \psi ( x ) ^ { \top } \Lambda \chi ( y ) } \mu ( \mathrm { d } y ) \Big ) \mu ( \mathrm { d } x ) . } \end{array}
$$

On the constraint set the last two terms do not depend on $\pi .$ , which means that minimizing $\mathrm { K L } ( \pi | \mu \otimes \mu )$ on the constrained set is equivalent to minimizing $\mathrm { K L } ( \pi | \pi _ { \mu } ^ { \Lambda } )$ there, which admits the choice rule $\pi _ { \mu } ^ { \Lambda }$ as unique minimizer. ■

This result describes the choice rule $\pi _ { \mu } ^ { \Lambda }$ as a conditional entropic projection [Csi75]: the least informative rule with its own query–key cross-moment. If $\begin{array} { r } { \mu = \frac { 1 } { n } \sum _ { i \leq n } \delta _ { x _ { i } } } \end{array}$ with distinct tokens and $\pi \in \Pi _ { 1 } ( \mu )$ chooses the key $x _ { j }$ for the query $x _ { i }$ with probability $p _ { i j }$ , then

$$
\mathrm { K L } ( \pi | \mu \otimes \mu ) = \log n + \frac { 1 } { n } \sum _ { i , j } p _ { i j } \log p _ { i j } = \log n - H ( Y | X ) ,
$$

where $H ( Y \mid X )$ is the Shannon entropy of the key given the (uniformly distributed) query. Minimizing the relative entropy under a constraint $\begin{array} { r } { \int { \psi \chi ^ { \top } } \mathrm { d } \pi = C } \end{array}$ is therefore the conditional maximum-entropy problem of [BDPDP96, §3], with context law $\mu$ and features $\psi \chi ^ { \top }$ . By Lagrangian duality, if C lies in the relative interior of the set of attainable cross-moments $\begin{array} { r } { \{ \int \psi \chi ^ { \top } \mathrm { d } \pi : \ \bar { \pi } \in \bar { \Pi _ { 1 } } ( \mu ) \} } \end{array}$ , the constrained minimizer is $\pi _ { \mu } ^ { \Lambda }$ for any maximizer Λ of the concave function $\Lambda \mapsto \langle \Lambda , C \rangle - F _ { \Lambda } ( \mu )$ , whose maximizers are characterized by $\begin{array} { r } { \int { \psi \chi ^ { \dagger } } \mathrm { d } \pi _ { \mu } ^ { \Lambda } = C } \end{array}$ [Csi75,BDPDP96]. The maximizer need not be unique, but the minimizer $\pi _ { \mu } ^ { \Lambda }$ is, by Proposition 3.6. In words, the lowest-order relational choice rule is the least informative rule that is informed of the query–key cross-moment.

Remark 3.7 (One marginal versus two: the diference with Sinkformers). Restricting (3.5) to couplings with both marginals equal to µ gives the entropic optimal transport (Schrödinger) problem with reward $x ^ { \top } \Lambda y$ [Sch31, Sch32, Léo14, Nut21], whose optimizer is the doubly stochastic attention of Sinkformers $I S A B P \mathcal { Z } \mathcal { Z } J$ Softmax attention is the semi-relaxed problem: the query marginal is fixed, while the key marginal is $K _ { \mu } ^ { * } ( y ) \mu ( \mathrm { d } y )$ with

$$
K _ { \mu } ^ { * } ( y ) = \int \frac { k _ { \Lambda } ( x , y ) } { Z _ { \mu } ( x ) } \mu ( \mathrm { d } x ) ,\tag{3.6}
$$

the attention that the sentence pays to $y .$

3.3. Attention cross-moment and readouts in the minimization problem. The choice rule enters the language model through the mean parameter of the family (3.3), the attention cross-moment $S _ { \Lambda } ( \mu ) = \nabla _ { \Lambda } F _ { \Lambda } ( \mu )$ of (1.8). It is suficient for the rule: by Proposition $3 . 6 , \pi _ { \mu } ^ { \Lambda }$ is the only least informative rule with cross-moment $S _ { \Lambda } ( \mu )$ , so µ and $S _ { \Lambda } ( \mu )$ determine the plan. It also collects every lowest-order relational readout of the choice: for $g ( x , y ) = x ^ { \top } V y \in \mathcal { Y } _ { 1 } \otimes \mathcal { Y } _ { 1 }$ one has $\mathbb { E } _ { \pi _ { u } ^ { \Lambda } } [ g ] = \langle V , S _ { \Lambda } ( \mu ) \rangle$

Definition 3.8 (Attention readout). For $\Lambda , V \in \mathbb { M } _ { d + 1 }$ , the attention readout of a sentence $\mu \in \mathcal P ( \mathbb { S } ^ { d } )$ is

$$
\Psi _ { \Lambda , V } ( \mu ) = \langle V , S _ { \Lambda } ( \mu ) \rangle = \mathbb { E } _ { \pi _ { \mu } ^ { \Lambda } } \big [ X ^ { \top } V Y \big ] = \int x ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) \mu ( \mathrm { d } x ) .\tag{3.7}
$$

The attention readout naturally appears in the Gibbs tilts of Theorem 2.1, with $\Phi = S _ { \Lambda }$ . In particular, their laws are $\mathrm { P } _ { \Lambda , V } \propto e ^ { \Psi _ { \Lambda , V } } \mathrm { N }$ , the value matrix V being the multiplier (cf. Theorem 3.11). In $\Psi _ { \Lambda , V }$ the attention plan is chosen with Λ and its payof is evaluated with $V$

In the next lemma we collect some useful properties of the attention cross-moment and readout.

Lemma 3.9. Let $\Lambda , V \in \mathbb { M } _ { d + 1 }$ and $\mu \in \mathcal P ( \mathbb { S } ^ { d } )$

(a) $\| S _ { \Lambda } ( \mu ) \| _ { \mathrm { o p } } \leq 1$ and $\left| \Psi _ { \Lambda , V } ( \mu ) \right| \leq \left\| V \right\| _ { \mathrm { o p } }$ . Moreover $S _ { \Lambda } ( \delta _ { x } ) = x x ^ { \top }$ and $\Psi _ { \Lambda , V } ( \delta _ { x } ) = x ^ { \top } V x .$

(b) $\mu \mapsto S _ { \Lambda } ( \mu )$ is weakly continuous. More precisely, if $\mu _ { k } \ \to \ \mu$ weakly, then $Z _ { \mu _ { k } } , { \mathrm { ~ A t t n } } _ { \Lambda } ( \cdot ; \mu _ { k } )$ and $\mathrm { C o v } _ { \pi _ { \mu _ { k } } ^ { \Lambda } \left( \cdot | x \right) } ( Y )$ converge uniformly on $\mathbb { S } ^ { d }$ to $Z _ { \mu } , \mathrm { A t t n } _ { \Lambda } ( \cdot ; \mu )$ and $\operatorname { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | x ) } ( Y )$

Proof. Statement (a) is immediate. Statement (b) follows from $\{ k _ { \Lambda } ( x , \cdot ) \} _ { x \in \mathbb { S } ^ { d } }$ being uniformly bounded and equicontinuous, and from $Z _ { \mu } \geq e ^ { - \| \Lambda \| _ { \mathrm { o p } } }$ ■

To apply Theorem 2.1 to the statistic $S _ { \Lambda }$ we need it to be non-degenerate on the support of the null model, $\mathrm { i . e . } \ \langle A , S _ { \Lambda } \rangle$ must not be constant on $\mathcal { S } _ { \mathbf { N } }$ unless $A = 0$ . This holds for every non-zero choice matrix.

Proposition 3.10 (Non-degeneracy). Let $\Lambda \neq 0$ and assume that $\nu ( m ) > 0$ for some $m \ge 3 . ~ I f \left. A , S _ { \Lambda } \right.$ is constant on $\mathcal { S } _ { \mathbf { N } }$ , then $A = 0$

The proof, in Section A, restricts $\langle A , S _ { \Lambda } \rangle$ to two-point sentences $( 1 - t ) \delta _ { x } + t \delta _ { y }$ . The softmax normalization produces factors $e ^ { x ^ { \top } \Lambda ( y - x ) }$ which, along great circles, are exponentials of trigonometric polynomials and cannot be compensated by polynomial terms. No symmetry of Λ is needed, whereas for $\Lambda = 0 , \mathrm { i . e . }$ . for the uniform choice µ ${ \bf \nabla } \mu , \left. A , S _ { 0 } \right.$ is constant for every antisymmetric A. The bound $m \geq 3$ comes from the rational dependence of $S _ { \Lambda }$ on the weights of a sentence.

Proposition 3.10 allows us to consider for any non-zero choice matrix $\Lambda \in \mathbb { M } _ { d + 1 }$ the minimization problem (1.4) with constraining statistic $\Phi = S _ { \Lambda }$ , that is, the problem of finding

$$
R _ { S } ( \boldsymbol { \Lambda } ) = \operatorname* { i n f } \left\{ \mathrm { K L } ( \mathrm { P } \mid \mathbf { N } ) : \ \mathbb { E } _ { \mathrm { P } } [ S _ { \boldsymbol { \Lambda } } ] = \mathbb { E } _ { \mathbf { P } ^ { \star } } [ S _ { \boldsymbol { \Lambda } } ] \right\}
$$

which corresponds to the amount of information that must be added to the null model to reproduce the attention cross-moment of the language. $\mathbf { P } ^ { \star } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) )$ is the language law which is generally unknown, and in practice $\mathbb { E } _ { \mathbf { P } ^ { \star } } [ S _ { \Lambda } ]$ is replaced by its empirical counterpart over a corpus $\widehat { S _ { \Lambda } }$

cWe are now ready to specialize the abstract results from Section 2 to our language model setting. In this setting the exponential family associated to the minimization problem is the exponential family with suficient statistic $S _ { \Lambda }$ and parametrized by value matrices $V \in \mathbb { M } _ { d + 1 }$

$$
\mathrm { P } _ { \Lambda , V } = e ^ { \Psi _ { \Lambda , V } - \Gamma ( \Lambda , V ) } { \bf N } , \quad \mathrm { w i t h ~ } \Gamma ( \Lambda , V ) = \log \mathbb { E } _ { \mathbf { N } } \big [ e ^ { \Psi _ { \Lambda , V } } \big ]
$$

Further recall the definition of the closed convex hull $\mathcal { K } _ { \Lambda } = \overline { { \mathrm { c o n v } } } S _ { \Lambda } \big ( S _ { \mathbf { N } } \big ) \subset \mathbb { M } _ { d + 1 }$ , associated to the cross-moment statistic, where $\mathcal { S } _ { \mathbf { N } }$ is the support of the prior model N.

Theorem 3.11 (Entropic minimization of language laws with cross-moment statistics). Assume that $\nu ( m ) > 0$ for some $m \geq 3$ and let $\Lambda \neq 0$

(a) $V \mapsto { \mathcal { M } } _ { \Lambda } ( V ) : = \mathbb { E } _ { \mathrm { P } _ { \Lambda , V } } S _ { \Lambda }$ is a real-analytic difeomorphism of $\mathbb { M } _ { d + 1 }$ onto the interior of $\kappa _ { \Lambda }$ . Its diferential at V is the covariance operator Cov $\mathrm { \Sigma } _ { \mathrm { P } _ { \Lambda . V } } ( S _ { \Lambda } )$ , which is positive definite on $\mathbb { M } _ { d + 1 }$

(b) If $\mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } \ \in$ int $\kappa _ { \Lambda }$ , the problem defining $R _ { S } ( \Lambda )$ has the unique solution $\mathrm { P } _ { \Lambda , V ^ { \star } ( \Lambda ) }$ , where $V ^ { \star } ( \Lambda ) =$

$\mathcal { M } _ { \Lambda } ^ { - 1 } ( \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } )$ is the unique value matrix with $\mathbb { E } _ { \mathrm { P } _ { \Lambda , V ^ { \star } } } S _ { \Lambda } = \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda }$ , and

$$
R _ { S } ( \boldsymbol { \Lambda } ) = \operatorname* { s u p } _ { V \in \mathbb { M } _ { d + 1 } } \Big \{ \langle V , \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \boldsymbol { \Lambda } } \rangle - \Gamma ( \boldsymbol { \Lambda } , V ) \Big \} .
$$

Every admissible P satisfies KL $. ( \mathrm { P } | \mathbf { N } ) = \mathrm { K L } ( \mathrm { P } | \mathrm { P } _ { \Lambda , V ^ { \star } } ) + R _ { S } ( \Lambda )$

(c) (Every value matrix is realized) For every $V \in \mathbb { M } _ { d + 1 }$ , the language $\mathbf { P } ^ { \star } = \mathrm { P } _ { \Lambda , V }$ has value matrix $\dot { V } ^ { \star } ( \Lambda ) = V$ . In particular, the value matrix is not determined by the query–key matrix: query–key and value matrices are untied.

Proof. The first two claims follow by applying Theorem 2.1 to $\Phi = S _ { \Lambda } \in \mathbb { M } _ { d + 1 } \cong \mathbb { R } ^ { ( d + 1 ) ^ { 2 } }$ , which is bounded and weakly continuous by Lemma 3.9, and ν-non-degenerate by Proposition 3.10. As regards Item (c), for $\mathbf { P } ^ { \star } = \mathrm { P } _ { \Lambda , V }$ we have $\mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } = \mathcal { M } _ { \Lambda } ( V ) \in \operatorname { i n t } \mathcal { K } _ { \Lambda }$ , so (b) applies and $V ^ { \star } ( \Lambda ) = { \mathcal { M } } _ { \Lambda } ^ { - 1 } ( { \mathcal { M } } _ { \Lambda } ( V ) ) = V$ ■

Remark 3.12 (Interior constraints for a corpus). In practice the constraint of Theorem $\it { 3 . 1 1 - ( b ) }$ is the empirical cross-moment $\begin{array} { r } { \widehat { S _ { \Lambda } } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } S _ { \Lambda } ( \mu _ { j } ) } \end{array}$ of a corpus of sentences $\mu _ { 1 } , \dotsc , \mu _ { J } \in { \mathcal { S } } _ { \mathbf { N } }$ , and it is natural to ask when it clies in int . A simple suficient condition is that the centered matrices $C _ { j } = S _ { \Lambda } ( \mu _ { j } ) - \widehat { S _ { \Lambda } } , j \le J ,$ span $\mathbb { M } _ { d + 1 }$ Since $\textstyle \sum _ { j } C _ { j } = 0$ , these J matrices span a space of dimension at most $J - 1$ c, so this requires $J \geq ( d + 1 ) ^ { 2 } + 1$ sentences, in general position.

To see why the condition sufices, assume by contradiction that $\widehat { S _ { \Lambda } }$ is not an interior point of the closed convex set $\ K _ { \Lambda } . \ B y$ c the supporting hyperplane theorem, there is a direction $A \in \mathbb { M } _ { d + 1 } \setminus \{ 0 \}$ along which $\widehat { S _ { \Lambda } }$ is extremal in $\kappa _ { \Lambda }$ , that is, $\langle A , S \rangle \leq \langle A , { \widehat { S _ { \Lambda } } } \rangle$ for every $S \in { \mathcal { K } } _ { \Lambda }$ . Each $S _ { \Lambda } ( \mu _ { j } )$ belongs to $\kappa _ { \Lambda }$ c, so the numbers $\langle A , \widehat { S _ { \Lambda } } \rangle - \langle A , S _ { \Lambda } ( \mu _ { j } ) \rangle = - \langle A , C _ { j } \rangle$ care all non-negative, and their average is zero. Hence they all vanish: $A$ is corthogonal to every $C _ { j }$ , which contradicts the spanning assumption.

Remark 3.13 (Two multipliers of one moment). In an attention head the query, key, value and output projections enter only through the two products $Q ^ { \top } K$ and $W _ { O } W _ { V }$ , which play independent roles: the first determines the attention pattern, the second what is transmitted along it. In our model these are exactly the two parameters, $\Lambda = Q ^ { \top } K$ and $V ,$ and they arise as the multipliers of two diferent variational problems. Indeed, for a fixed sentence, Proposition 3.6 identifies Λ as the Lagrange multiplier of the constraint $\bar { \mathbb { E } } _ { \pi } [ X Y ^ { \top } ]$ = C on choice rules, the constraint being met with $C = S _ { \Lambda } ( \mu )$ . Over the language, $V ^ { \star }$ is the Lagrange multiplier of the same moment averaged over sentences. The two multipliers are determined by diferent problems, one inside each sentence and one across sentences, and in the cross-moment model nothing relates them.

Remark 3.14 (Tied values are the exception). Under the assumptions of Theorem 3.11, the data cross-moments $c \in$ int $\kappa _ { \Lambda }$ for which the value matrix is tied to the attention matrix, $V ^ { \star } ( \Lambda ) \in \mathbb { R } \Lambda$ , form the curve $\mathcal { M } _ { \Lambda } ( \mathbb { R } \Lambda )$ Since $\mathcal { M } _ { \Lambda }$ is a real-analytic difeomorphism onto the open set int $\kappa _ { \Lambda }$ and RΛ is a closed embedded line, this curve is a closed one-dimensional embedded real-analytic submanifold of int $\kappa _ { \Lambda } ;$ as $( d + 1 ) ^ { 2 } > 1$ , it has empty interior and zero Lebesgue measure in $\mathbb { M } _ { d + 1 } \cong \mathbb { R } ^ { ( d + 1 ) ^ { 2 } }$ . At a fixed attention matrix the value matrix is thus a free parameter, whatever matrix makes the model reproduce the cross-moment of the language, and tied values are the exception. This is in contrast with the free-energy statistic of Remark 3.20, for which values are always tied.

We now compute the first variation of the attention readout and its gradient, which by Lemma 2.3 is the force that a new token undergoes, conditioned on the presence of context in our model.

Proposition 3.15 (First variation of the attention readout). Let $\Lambda , V \in \mathbb { M } _ { d + 1 }$ . Then $\Psi _ { \Lambda , V }$ satisfies the hypotheses of Lemma 2.3: it is weakly continuous, and the diferentiability conditions $( i ) - ( i i )$ hold with

$$
( 3 . 8 ) ~ \delta \Psi _ { \Lambda , V } ( \mu ) ( z ) = z ^ { \top } V \mathrm { A t t m } _ { \Lambda } ( z ; \mu ) + \int \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } x ^ { \top } V \big ( z - \mathrm { A t t m } _ { \Lambda } ( x ; \mu ) \big ) ~ \mu ( \mathrm { d } x ) , \qquad \ \mu \big ( \delta \Psi _ { \Lambda , V } ( \mu ) \big ) = \Psi _ { \Lambda , V } ( \mu ) ,
$$

$| \delta \Psi _ { \Lambda , V } ( \mu ) ( z ) | \leq \| V \| _ { \mathrm { o p } } \left( 1 + 2 e ^ { 2 \| \Lambda \| _ { \mathrm { o p } } } \right) f o r z \in \mathbb { S } ^ { d }$ , and, for $z \in \mathbb { R } ^ { d + 1 }$ ,

$$
\nabla _ { z } \delta \Psi _ { \Lambda , V } ( \mu ) ( z ) = V \mathrm { A t t n } _ { \Lambda } ( z ; \mu ) + V ^ { \top } \int x \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } \mu ( \mathrm { d } x ) + \rho _ { \mu } ( z ) ,\tag{3.9}
$$

$$
w i t h ~ \rho _ { \mu } ( z ) = \Lambda \operatorname { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | z ) } ( Y ) V ^ { \top } z + \Lambda ^ { \top } \int \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } \left[ x ^ { \top } V \big ( z - \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) \big ) \right] x \mu ( \mathrm { d } x ) ,
$$

where we recall our abuse of notation for $\pi _ { \mu } ^ { \Lambda } ( \cdot | z ) ( \mathrm { d } y ) = k _ { \Lambda } ( z , y ) \mu ( \mathrm { d } y ) / Z _ { \mu } ( z )$ , which indicates the law of the key chosen by any query $z \in \mathbb { S } ^ { d }$

Proof. The weak continuity of the readout $\Psi _ { \Lambda , V }$ directly follows from Lemma 3.9-(b). For notational convenience let $\begin{array} { r } { N _ { \mu } ( x ) : = \int y k _ { \Lambda } ( x , y ) \mu ( \mathrm { d } y ) } \end{array}$ so that Attn $\ L _ { \Lambda } ( \cdot ; \mu ) = N _ { \mu } / Z _ { \mu }$ . Fix $\mu , \varrho \in \mathcal { P } ( \mathbb { S } ^ { d } )$ , consider the line $\mu _ { t } = \mu + t ( \varrho - \mu )$ for $t \in [ 0 , 1 ]$ and let $N _ { t } = N _ { \mu _ { t } }$ and $Z _ { t } = Z _ { \mu _ { t } }$ . These two are afine functions in $t ,$ with $\begin{array} { r } { \dot { N } _ { t } ( x ) = \int y k _ { \Lambda } ( x , y ) \ d y ( \varrho - } \end{array}$ $\begin{array} { r } { \mu ) ( \mathrm { d } y ) , \dot { Z } _ { t } ( x ) = \int k _ { \Lambda } ( x , y ) ( \varrho - \mu ) ( \mathrm { d } y ) } \end{array}$ and $Z _ { t } \geq e ^ { - \| \Lambda \| _ { \mathrm { o p } } }$ on $\mathbb { S } ^ { d }$ . Hence $\mathrm { A t t n } _ { \Lambda } ( \cdot ; \mu _ { t } ) = N _ { t } / Z _ { t }$ is smooth in $t ,$ with

$$
\partial _ { t } \mathrm { A t t n } _ { \Lambda } ( x ; \mu _ { t } ) = \frac { \dot { N } _ { t } ( x ) - \mathrm { A t t n } _ { \Lambda } ( x ; \mu _ { t } ) \dot { Z } _ { t } ( x ) } { Z _ { t } ( x ) } = \int \frac { k _ { \Lambda } ( x , y ) } { Z _ { t } ( x ) } \left( y - \mathrm { A t t n } _ { \Lambda } ( x ; \mu _ { t } ) \right) ( \varrho - \mu ) ( \mathrm { d } y ) .
$$

Since $\mu _ { t }$ is afine in t and $\mathrm { A t t n } _ { \Lambda } ( \cdot ; \mu _ { t } )$ $\partial _ { t } \mathrm { A t t n } _ { \Lambda } ( \cdot ; \mu _ { t } )$ are bounded on $\mathbb { S } ^ { d }$ uniformly in t, we may diferentiate $\begin{array} { r } { \Psi _ { \Lambda , V } ( \mu _ { t } ) = \int x ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( x ; \mu _ { t } ) \mu _ { t } ( \mathrm { d } x ) } \end{array}$ under the integral sign, which gives the first variation expression in $\left( 3 . 8 \right)$ . The identity $\mu ( \delta \Psi _ { \Lambda , V } ( \mu ) ) = \Psi _ { \Lambda , V } ( \mu )$ follows from $\begin{array} { r } { \int k _ { \Lambda } ( x , z ) \mu ( \mathrm { d } z ) = Z _ { \mu } ( x ) } \end{array}$ and $\begin{array} { r l } { \int k _ { \Lambda } ( x , z ) z \mu ( \mathrm { d } z ) = } \end{array}$ $Z _ { \mu } ( x ) \mathrm { A t t n } _ { \Lambda } ( x ; \mu )$

On $\begin{array} { r } { \mathbb { S } ^ { d } , \ \big | z ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( z ; \mu ) \big | \leq \| V \| _ { \mathrm { o p } } , \ k _ { \Lambda } ( x , z ) / Z _ { \mu } ( x ) \leq \epsilon ^ { \| \Lambda \| _ { \infty } } / \epsilon ^ { - \| \Lambda \| _ { \infty } } = \epsilon ^ { 2 \| \Lambda \| _ { \infty } } \mathrm { ~ a n d ~ } | z - \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) | \leq 2 \delta } \end{array}$ which gives the bound on $\delta \Psi _ { \Lambda , V }$

The formula (3.8) defines a smooth function of $z \in \mathbb { R } ^ { d + 1 }$ , since $Z _ { \mu } ( z ) > 0$ . By Proposition $\mathrm { 3 . 5 \mathrm { - } ( d ) }$ , the Jacobian of $z \mapsto \operatorname { A t t n } _ { \Lambda } ( z ; \mu )$ is $\mathrm { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | z ) } ( Y ) \Lambda ^ { \top }$ , and the covariance is symmetric, so that

$$
\nabla _ { z } \left( z ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( z ; \mu ) \right) = V \mathrm { A t t n } _ { \Lambda } ( z ; \mu ) + \Lambda \operatorname { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | z ) } ( Y ) V ^ { \top } z .
$$

Moreover $\nabla _ { z } \big ( k _ { \Lambda } ( x , z ) / Z _ { \mu } ( x ) \big ) = \big ( k _ { \Lambda } ( x , z ) / Z _ { \mu } ( x ) \big ) \Lambda ^ { \top } x ,$ with $\mu$ and x fixed. This gives (3.9), the gradient of the extension of $\delta \Psi _ { \Lambda , V } ( \mu )$ to $\mathbb { R } ^ { d + 1 }$ given by (3.8).

It remains to check that $( \mu , z ) \mapsto \nabla _ { z } \delta \Psi _ { \Lambda , V } ( \mu ) ( z )$ is jointly continuous on $\mathcal { P } ( \mathbb { S } ^ { d } ) \times \mathbb { S } ^ { d }$ . By Lemma $3 . 9 \mathrm { - ( b ) }$ , the functions $Z _ { \mu } , \mathrm { A t t n } _ { \Lambda } ( \cdot ; \mu )$ and $\operatorname { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot \vert \cdot ) } ( Y )$ depend continuously on $( \mu , z )$ , and the integrals in (3.9) are of the form $\begin{array} { r } { \int g _ { \mu } ( x , z ) \mu ( \mathrm { d } x ) } \end{array}$ with $g _ { \mu }$ continuous and depending continuously on $\mu ,$ uniformly in $( x , z )$ ; hence they are jointly continuous as well. In particular, using $k _ { \Lambda } ( x , z ) / Z _ { \mu } ( x ) \leq e ^ { 2 \| \Lambda \| _ { \mathrm { o p } } }$ and $\left. \mathrm { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | z ) } ( Y ) \right. _ { \mathrm { o p } } \leq 1$ , one gets $| \nabla _ { z } \delta \Psi _ { \Lambda , V } ( \mu ) ( z ) | \leq \| V \| _ { \mathrm { o p } } \left( 1 + \| \Lambda \| _ { \mathrm { o p } } \right) \left( 1 + 2 e ^ { 2 \| \Lambda \| _ { \mathrm { o p } } } \right) \mathrm { o n } \ \mathbb { S } ^ { d }$

Let us interpret these formulas. The first variation (3.8) has two terms. The first, $z ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( z ; \mu )$ , is the alignment, through V , between z and what it reads as a query. The second,

$$
\int \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } x ^ { \top } V \big ( z - \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) \big ) \mu ( \mathrm { d } x ) ,
$$

measures the usefulness of z as a key: each token $x$ that attends to $z ,$ with weight $k _ { \Lambda } ( x , z ) / Z _ { \mu } ( x )$ , compares its alignment with z to its alignment with what it currently reads.

Correspondingly, the force (3.9) on z has three terms.

(i) The query term $V \mathrm { A t t n } _ { \Lambda } ( z ; \mu )$ : an average of the keys with softmax weights, which compete through the normalization $Z _ { \mu } ( z )$ , transported by the value matrix $V .$ This is the transformer head (1.1).

(ii) The key-side term $\begin{array} { r } { V ^ { \top } \int x \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } \mu ( \mathrm { d } x ) } \end{array}$ : the tokens that attend to z, weighted by the attention they pay to it and read through $V ^ { \top }$

(iii) The correction $\rho _ { \mu } ( z ) \stackrel { \cdot } { \in } \mathrm { r a n g e } \Lambda + \mathrm { r a n g e } \Lambda ^ { \top }$ , which records that moving z changes not only what is read but also where tokens look. Its query part $\Lambda \mathrm { C o v } _ { \pi _ { u } ^ { \Lambda } ( \cdot | z ) } ( Y ) V ^ { \top } z$ is the gradient of the alignment through the attention weights of $z ;$ it vanishes when z attends to a single key, since the covariance is then zero. Its key part is the gradient of the usefulness through the weights $k _ { \Lambda } ( x , z ) / Z _ { \mu } ( x )$ with which the other tokens read z.

Remark 3.16 (The correction does not vanish). For $\mu = \delta _ { x }$ one has $\mathrm { C o v } _ { \pi _ { \mu } ^ { \Lambda } ( \cdot | x ) } = 0$ and $\mathrm { A t t n } _ { \Lambda } ( x ; \mu ) = x _ { \Sigma }$ , so $\rho _ { \delta _ { x } } ( z ) = e ^ { x ^ { \top } \Lambda ( z - x ) } x ^ { \top } V ( z - x ) \Lambda ^ { \top } x . \ I f \rho \equiv 0$ , then $x ^ { \top } V ( z - x ) = 0$ for every z whenever $\Lambda ^ { \top } x \neq 0 ,$ that $i s ,$ $V ^ { \top } x = 0$ on a dense set of x when $\Lambda \neq 0$ . So $\rho _ { ; }$ , as a function of $( \mu , z )$ , vanishes identically only if $\colon \Lambda = 0$ or $V = 0$ . In the free-energy model (Remark 3.20), by contrast, the plan is optimal for the payof, and there is no such correction.

We now combine Proposition 2.2, Lemma 2.3, and Proposition 3.15 in order to get the last token law and its score under any Gibbs tilt $\mathrm { P } _ { \Lambda , V }$ . For any fixed $\Lambda ,$ this in particular applies to $V ^ { \star } = V ^ { \star } ( \Lambda )$ as in Theorem 3.11, and hence to our optimizer $\mathrm { P } _ { \Lambda , V ^ { \star } }$ for (1.4), which corresponds to Item (iii) of the Meta-Theorem for the unordered model.

Theorem 3.17 (Last token of the attention model). Let $\Lambda , V \in \mathbb { M } _ { d + 1 } , n \geq 1$ with $\nu ( n ) > 0 , x _ { 1 : n - 1 } \in ( \mathbb { S } ^ { d } ) ^ { n - 1 }$ 2 and $\begin{array} { r } { \mu _ { y } = \frac { 1 } { n } ( \sum _ { i < n } \delta _ { x _ { i } } + \delta _ { y } ) } \end{array}$

(a) (Exact law) Under $\mathrm { P } _ { \Lambda , V ; }$ , given $N = n$ and $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , the last token has density with respect to vol

$$
p ( y ) = \frac { \exp \left( \Psi _ { \Lambda , V } ( \mu _ { y } ) \right) } { \int \exp \left( \Psi _ { \Lambda , V } ( \mu _ { y ^ { \prime } } ) \right) \underline { { \operatorname { v o l } ( \mathrm { d } y ^ { \prime } ) } } } , \qquad y \in \mathbb { S } ^ { d } .
$$

(b) (Score) log p is smooth on $\mathbb { S } ^ { d }$ , and its Riemannian gradient is

$$
\nabla _ { \mathbb { S } ^ { d } } \log p ( y ) = \frac { 1 } { n } P _ { y } ^ { \perp } \Big ( V \mathrm { A t t n } _ { \Lambda } ( y ; \mu _ { y } ) + V ^ { \top } \int x \frac { e ^ { \langle x , \Lambda y \rangle } } { \int e ^ { \langle x , \Lambda z ^ { \prime } \rangle } \mu _ { y } ( \mathrm { d } z ^ { \prime } ) } \mu _ { y } ( \mathrm { d } x ) + \rho _ { \mu _ { y } } ( y ) \Big ) ,\tag{3.10}
$$

with ρ as in (3.9).

(c) (Transformer reading) For $\Lambda = Q ^ { \top } K$ , the first term in (3.10) is the transformer map (1.1) with query matrix Q, key matrix K and value matrix V, applied to the candidate y as a query over the completed sentence:

$$
V \mathrm { A t t n } _ { \Lambda } ( y ; \mu _ { y } ) = \sum _ { j = 1 } ^ { n } \frac { e ^ { \langle Q y , K x _ { j } \rangle } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle Q y , K x _ { \ell } \rangle } } V x _ { j } , \qquad x _ { n } = y .
$$

Proof. Item (a) is Proposition $2 . 2 \mathrm { - ( b ) }$ with $\Phi = S _ { \Lambda }$ , which is bounded and weakly continuous by Lemma 3.9. Claim (b) can be proven as follows. The smoothness of log p directly follows from the one of $y \mapsto \Psi _ { \Lambda , V } ( \mu _ { y } ) =$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i \leq n } x _ { i } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( x _ { i } ; \mu _ { y } ) } \end{array}$ , with $x _ { n } = y$ , which is a finite combination of smooth functions of y (because all the denominators are at least $e ^ { - \| \Lambda \| _ { \mathrm { o p } } } )$ Then, by Proposition 3.15, $\Psi _ { \Lambda , V }$ is $\mathcal { C } ^ { 1 }$ in the sense of Lemma 2.3 (diferentiability conditions $( \mathrm { i } ) { - } ( \mathrm { i i } ) )$ , which gives $\begin{array} { r } { \nabla _ { \mathbb { S } ^ { d } } \log p ( y ) = \frac { 1 } { n } { P } _ { y } ^ { \perp } \nabla _ { z } \delta \Psi _ { \Lambda , V } ( \mu _ { y } ) ( z ) | _ { z = y } } \end{array}$ . Combining this last identity with (3.9) at $\mu = \mu _ { y }$ and $z = y$ gives (3.10).

Item (c) is a straightforward computation.

Remark 3.18 (All tokens at once). Lemma 2.3 applies to any token of a sentence, not only to the last one. Let $\begin{array} { r } { \mu = \frac { 1 } { n } \sum _ { j \leq n } \delta _ { z _ { j } } } \end{array}$ and $\mathit { f i x } \mathit { i } \leq n$ . Taking the other tokens $( z _ { j } ) _ { j \neq i }$ as context and $y = z _ { i }$ , the completed sentence $\mu _ { y }$ of Lemma 2.3 is $\mu$ itself, so that the Riemannian gradient of $z _ { i } \mapsto \Psi _ { \Lambda , V } ( \mu )$ is

$$
\nabla _ { z _ { i } } \Psi _ { \Lambda , V } ( \mu ) = \frac { 1 } { n } \left. P _ { z _ { i } } ^ { \perp } \nabla _ { z } \delta \Psi _ { \Lambda , V } ( \mu ) ( z ) \right| _ { z = z _ { i } } .
$$

Hence gradient ascent of $\Psi _ { \Lambda , V }$ on $( \mathbb { S } ^ { d } ) ^ { n }$ , accelerated by the factor n, moves every token $z _ { i }$ along the vector field $z \mapsto P _ { z } ^ { \perp } \nabla _ { z } \delta \Psi _ { \Lambda , V } ( \mu ) ( z ) ~ o f ~ ( 3 . 9 )$ , which is the Wasserstein gradient of $\Psi _ { \Lambda , V }$ on ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ . In the measure-to-measure picture of transformers [SABP22, GLPR23, GLPR25, GRRB26], a layer instead moves each token along the self-attention field $P _ { z } ^ { \perp } { \bar { V _ { \mathrm { A t t n } } } } _ { \Lambda } ( z ; \mu )$ . By (3.9), this is only the query term of the Wasserstein gradient of the attention readout: the key-side term and the correction $\rho _ { \mu }$ are what make the field an exact gradient.

## 3.4. Concluding remarks on training and free energy.

Remark 3.19 (Training). In this paper the attention matrix Λ is fixed, while the value matrix is not tuned independently of the data: it is the Lagrange multiplier $V ^ { \star } ( \Lambda )$ of Theorem 3.11, determined by the constraint $\mathbb { E } _ { \mathrm { P } } S _ { \Lambda } = \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda }$ . It is also the maximum-likelihood value matrix: for the expected log-likelihood ratio of the model against the null model,

$$
\mathcal { L } ( \Lambda , V ) = \mathbb { E } _ { \mathbf { P } ^ { \star } } \left[ \log \frac { \mathrm { d } \mathbb { P } _ { \Lambda , V } } { \mathrm { d } \mathbf { N } } \right] = \mathbb { E } _ { \mathbf { P } ^ { \star } } \left[ \Psi _ { \Lambda , V } \right] - \log \mathbb { E } _ { \mathbf { N } } \left[ e ^ { \Psi _ { \Lambda , V } } \right] ,
$$

the map $V \mapsto { \mathcal { L } } ( \Lambda , V )$ is concave, its maximizer is $V ^ { \star } ( \Lambda )$ , and its maximum is $R _ { S } ( \Lambda )$ , by Theorem 3.11 and convex duality. Only Λ remains to be learned, and the natural criterion is

$$
\Lambda ^ { \star } \in \underset { \Lambda } { \operatorname { a r g m a x } } \ \mathcal { L } \big ( \Lambda , V ^ { \star } ( \Lambda ) \big ) = \underset { \Lambda } { \operatorname { a r g m a x } } \ \operatorname* { i n f } \big \{ \operatorname { K L } ( \mathrm { P } \mid \mathbf { N } ) : \ \mathbb { E } _ { \mathrm { P } } S _ { \Lambda } = \mathbb { E } _ { \mathrm { P } ^ { \star } } S _ { \Lambda } \big \} ,
$$

over a compact set of admissible attention matrices: $\Lambda ^ { \star }$ selects the attention cross-moment least compatible with the null model, and $\mathrm { P } _ { \Lambda ^ { \star } , V ^ { \star } ( \Lambda ^ { \star } ) }$ is the least informative model reproducing it. This is the minimax-entropy principle of [ZWM97]. The maximization is over a compact capacity set ${ \mathfrak { C } } \subset \mathbb { M } _ { d + 1 }$ , as in practice (norm or rank constraints on $\overset { \cdot } { \boldsymbol { \Lambda } } = \boldsymbol { Q } ^ { \intercal } \boldsymbol { K } )$ . A maximizer then exists as soon as $\Lambda \mapsto { \mathcal { L } } ( \Lambda , V ^ { \star } ( \Lambda ) )$ is continuous on C, which is not automatic since $V ^ { \star } ( \Lambda )$ is only implicitly defined. This holds, e.g., if 0 / C and the constraint is interior for every $\Lambda \in { \mathfrak { C } }$ , by the implicit function theorem applied to $\nabla _ { V } \Gamma ( \Lambda , V ) = \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda }$ . A complete analysis of this training problem is beyond the scope of the present work, and we plan to address it in future work.

Remark 3.20 (The free-energy statistic: tied values). By Proposition 3.5, the attention cross-moment is the gradient of the attention free energy $\begin{array} { r } { F _ { \Lambda } ( \mu ) = \int \log Z _ { \mu } \mathrm { d } \mu , \ i . e . \ S _ { \Lambda } = \nabla _ { \Lambda } F _ { \Lambda } } \end{array}$ . One may constrain the scalar $F _ { \Lambda }$ instead of $S _ { \Lambda } . ~ H F _ { \Lambda }$ is not constant on and the constraint is interior, Theorem 2.1 gives the minimizer $\mathrm { P } _ { \lambda ^ { \star } , \Lambda } ^ { F } \propto \dot { e } ^ { \lambda ^ { \star } F _ { \Lambda } } \mathrm { N }$ , with a scalar Lagrange multiplier $\lambda ^ { \star }$ . Three things change with respect to the cross-moment model.

(i) Values are tied. By Proposition ${ \mathcal { S } } . { \mathcal { 5 } } \mathbf { - } ( a ) , \lambda F _ { \Lambda } = \Psi _ { \Lambda , \lambda \Lambda } - \lambda \mathcal { H } _ { \Lambda }$ : the plan is read with the value matrix λΛ, whereas in the cross-moment model $V ^ { \star }$ is free (Theorem $\it { 3 . 1 1 - ( c ) }$

(ii) There is no correction term. A direct computation gives $\begin{array} { r } { \delta F _ { \Lambda } ( \mu ) ( z ) = \log Z _ { \mu } ( z ) + \int \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } \mu ( \mathrm { d } x ) } \end{array}$ , hence

$$
\nabla _ { z } \delta F _ { \Lambda } ( \mu ) ( z ) = { \Lambda } \mathrm { A t t n } _ { \Lambda } ( z ; \mu ) + { \Lambda } ^ { \top } \int x { \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } } \mu ( \mathrm { d } x ) ,
$$

which is (3.9) with $V = \Lambda$ and without $\rho _ { \mu }$

(iii) The value read from a key is the key. For $\Lambda = Q ^ { \top } K$ , the query term is $\begin{array} { r } { Q ^ { \top } \sum _ { j } \operatorname { s o f t m a x } _ { j } ( \langle Q y , K x _ { j } \rangle ) K x _ { j } } \end{array}$

## 4. The causal model: masking and next-token readout

$\mathrm { B y }$ Theorem 3.17, the score of the last token of the model $\mathrm { P } _ { \Lambda , V ^ { \star } }$ is ${ \scriptstyle { \frac { 1 } { n } } } P _ { y } ^ { \perp }$ applied to the sum of three terms: the attention head $V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( y ; \mu _ { y } )$ of the candidate y, a key-side term and a correction $\rho _ { \mu _ { y } } .$ These terms appear in the bidirectional model when diferentiating the first variation (3.8) of the readout, which consists of two parts, the payof of $z$ as a query and its efect on the readouts of the other tokens:

$$
\delta \Psi _ { \Lambda , V } ( \mu ) ( z ) = z ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( z ; \mu ) + \int \frac { k _ { \Lambda } ( x , z ) } { Z _ { \mu } ( x ) } x ^ { \top } V \big ( z - \mathrm { A t t n } _ { \Lambda } ( x ; \mu ) \big ) \mu ( \mathrm { d } x ) .
$$

The gradient of the first part is $V \mathrm { A t t n } _ { \Lambda } ( z ; \mu )$ plus the query part of $\rho _ { \mu } .$ , which comes from the dependence on z of the weights with which z reads. The gradient of the second part is the key-side term plus the key part of $\rho _ { \mu } ,$ which comes from the dependence on z of the weights with which the other tokens read $z .$ Hence:

(i) the key-side term and the key part of $\rho _ { \mu }$ vanish if no query reads the candidate;

(ii) the query part of $\rho _ { \mu }$ vanishes if the candidate is not a query.

A decoder transformer uses only an attention head, and in this section we explain why the other two terms are absent there. More precisely, we are going to change only the statistic: the choice rule is the causally masked rule (1.12), and the readout pairs what a token reads with the token that follows it. Indeed, causal masking gives property (i) for the last token, except for the self-term in which the last token reads itself: no query comes after it. A decoder also satisfies property (ii): the logit of a candidate next token is the inner product of the candidate with the output of the head whose query is the last token read. The candidate is compared with this output and never acts as a query. In particular the head term itself is evaluated at the query $x _ { n - 1 }$ , not at the candidate. The statistic below encodes this.

The rest of the section is organized as follows. In Section 4.1 we set up ordered sentences and derive the causal choice rule, in Section 4.2 we define the shifted cross-moment, solve the corresponding entropic projection, and show that the law of the last token is exactly a softmax over the sphere whose logit is one causal attention head, so that its score has no key-side term and no correction. This proves the Meta-Theorem for the masked model.

4.1. Ordered sentences, the null model and the causal choice rule. As explained in Section 1.4 we are interested in keeping track of the positions of tokens in sentences. Since our previous model was set up in the space of laws of random measures $\mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) )$ ), one way of keeping track of positions there is by lifting positions as well and considering ordered sentences as

$$
\vec { \mu } = \iota ( n , x _ { 1 : n } ) : = \frac 1 n \sum _ { i = 1 } ^ { n } \delta _ { ( x _ { i } , i ) } \in \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } ) \mathrm { ~ f o r ~ } n \geq 1 .
$$

The map ι is injective on $\begin{array} { r } { \mathcal { O } : = \bigsqcup _ { n \geq 1 } ( \mathbb { S } ^ { d } ) ^ { n } } \end{array}$ , even when tokens are repeated, because the positions are recorded: n is the number of atoms of $\vec { \mu }$ and $x _ { i }$ is the token at position i. It is continuous on each $( \mathbb { S } ^ { d } ) ^ { n }$ . The null mode $\begin{array} { r } { \vec { \bf N } = \iota _ { \# } \big ( \sum _ { n } \nu ( n ) \delta _ { n } \otimes { \underline { { \bf v o l } } } ^ { \otimes n } \big ) } \end{array}$ is the law of $\vec { \mu }$ when $N \sim \nu$ and the tokens are read in their order of generation; it is concentrated on $\iota ( { \mathcal { O } } )$ , and so is every P with $\mathrm { K L } ( \mathbf { P } | \overset {  } { \mathbf { N } } ) < \infty$ . The image of $\vec { \mu }$ under $( x , i ) \mapsto x$ is the unordered sentence $\mu _ { ; }$ , and the image of $\vec { \bf N }$ is then N.

In this ordered setting, a choice rule for $\vec { \mu }$ is a coupling in the set $\Pi _ { 1 } ( \vec { \mu } )$ of probability measures on $( \mathbb { S } ^ { d } \times \mathbb { N } ) ^ { 2 }$ with first marginal $\vec { \mu }$ , the query being written first. In a causally masked rule the query $( x _ { i } , i )$ only chooses among the keys $( x _ { j } , j )$ with $j \leq i .$ . Then, the least informative choice rule we can consider here is the masked uniform choice

$$
\mathbb { R } _ { \vec { \mu } } ^ { \mathtt { M } } ( \mathrm { d } x \mathrm { d } i , \mathrm { d } y \mathrm { d } j ) = \vec { \mu } ( \mathrm { d } x \mathrm { d } i ) \frac { \mathbb { 1 } _ { j \leq i } \vec { \mu } ( \mathrm { d } y \mathrm { d } j ) } { \vec { \mu } ( \mathbb { S } ^ { d } \times \{ 1 , \dots , i \} ) } = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { n } \frac { e ^ { M _ { i j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { M _ { i \ell } } } \delta _ { ( x _ { i } , i ) } \otimes \delta _ { ( x _ { j } , j ) } ,
$$

where $M \in \{ 0 , - \infty \} ^ { n \times n }$ is the causal additive mask defined as $M _ { i j } = \left( - \infty \right) \mathbb { 1 } _ { i < j }$ , with the convention $e ^ { - \infty } = 0$ This is the masked rule without relational content: the query $( x _ { i } , i )$ chooses its key uniformly among the i tokens that are not later than itself. In the unordered model this corresponds to the independent coupling $\mu \otimes \mu$

Remark 4.1 (Masked Luce axiom and the lowest-order masked rule). As in the Introduction, we postulate Luce’s axiom among the competitors that are not later than the query, with a choice function $h \in \mathcal { C } ( \mathbb { S } ^ { d } \times \mathbb { S } ^ { d } )$ that is the same for all sentences and does not depend on positions: the query $( x _ { i } , i )$ chooses the key $( x _ { j } , j )$ with probability $\mathbb { 1 } _ { j \leq i } \mathop { e ^ { h ( x _ { i } , x _ { j } ) } } / \sum _ { \ell < i } e ^ { h ( x _ { i } , x _ { \ell } ) }$ . This is the masked version of the ratio form (1.6). Proposition $\it 3 . 4$ holds verbatim for masked rules. The only point to check is the converse in (a): if h and $h ^ { \prime }$ give the same masked rule on every ordered sentence, consider the ordered sentence $( y , x )$ of length 2 with $y \neq x ;$ the query $( x , 2 )$ chooses between y and $x ,$ and equality of the two rules gives $\boldsymbol { g } ( x , y ) = \boldsymbol { g } ( x , x )$ for $g = h - h ^ { \prime }$ and every $y \neq x ,$ hence $g ( x , \cdot )$ is constant for every x.

Truncating the relational part of h at its lowest order and setting the key bias to zero gives $h ( x , y ) = x ^ { \top } \Lambda y$ that is, the causally masked rule $\pi _ { \vec { \mu } } ^ { \mathrm { { \check { A } , M } } }$ of (1.12)

$$
\pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } } = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { n } \frac { e ^ { \langle x _ { i } , \Lambda x _ { j } \rangle + M _ { i j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle x _ { i } , \Lambda x _ { \ell } \rangle + M _ { i \ell } } } \delta _ { ( x _ { i } , i ) } \otimes \delta _ { ( x _ { j } , j ) } .
$$

In coordinates, the query $( x _ { i } , i )$ of $\pi _ { \vec { \vec { \mu } } } ^ { \Lambda , \mathrm { M } }$ chooses the key $( x _ { j } , j )$ with probability $n \Pi _ { i j } ^ { \mathrm { M } }$ , and reads

$$
n \Pi _ { i j } ^ { \mathrm { M } } = \frac { \mathbb { 1 } _ { j \leq i } e ^ { x _ { i } ^ { \top } \Lambda x _ { j } } } { \sum _ { \ell \leq i } e ^ { x _ { i } ^ { \top } \Lambda x _ { \ell } } } = \frac { e ^ { \langle x _ { i } , \Lambda x _ { j } \rangle + M _ { i j } } } { \sum _ { \ell = 1 } ^ { n } e ^ { \langle x _ { i } , \Lambda x _ { \ell } \rangle + M _ { i \ell } } } , \qquad \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) = \sum _ { j \leq i } n \Pi _ { i j } ^ { \mathrm { M } } x _ { j } .\tag{4.1}
$$

Note that $n \Pi _ { i j } ^ { \mathrm { M } }$ and $\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; { \vec { \mu } } )$ depend on the sentence only through $x _ { 1 : i } ,$ and not on its length $n .$

Proposition 4.2 (The causal choice rule). Let $\vec { \mu } = \iota ( n , x _ { 1 : n } )$ and $\begin{array} { r } { C ^ { \mathrm { M } } = \int \ d x y ^ { \top } \mathrm { d } \pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } } } \end{array}$

(a) Every $\pi \in \Pi _ { 1 } ( \vec { \mu } )$ with KL $_ { ^ { \prime } } ( \pi | \mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } } ) < \infty$ gives zero mass to $\{ j > i \}$

(b) $\pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } }$ is the unique minimizer of $\mathrm { K L } ( \pi | \mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } } )$ over $\{ \pi \in \Pi _ { 1 } ( \vec { \mu } ) : \int x y ^ { \top } \mathrm { d } \pi = C ^ { \mathrm { M } } \}$

(c) The query $( x _ { i } , i )$ chooses the key $( x _ { j } , j )$ with probability nΠ<sup>M</sup><sub>i</sub> of (4.1), and $\begin{array} { r } { C ^ { \mathrm { M } } = \frac { 1 } { n } \sum _ { i } x _ { i } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top } } \end{array}$

Proof. Write $q = ( x , i )$ for a query, $\pi ( \mathrm { d } q , \mathrm { d } k ) = { \vec { \mu } } ( \mathrm { d } q ) \pi _ { q } ( \mathrm { d } k )$ , and $\mathrm { R } _ { q }$ for the conditional law of the key under $\mathrm { R } _ { \vec { \mathbf { \Gamma } } _ { \vec { \mu } } } ^ { \mathrm { M } }$ . Since π and $\mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } }$ have the same first marginal, the chain rule [DE97, Lemma 1.4.3] gives $\mathrm { K L } ( \pi | \mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } } ) =$ $\begin{array} { r } { \int { \vec { \mu } } ( \mathrm { d } q ) \operatorname { K L } ( \pi _ { q } \mid \operatorname { R } _ { q } ) } \end{array}$ . If this is finite, then $\pi _ { q } \ll \mathrm { R } _ { q }$ for ${ \vec { \mu } } .$ -almost every $q .$ In particular, since $\mathrm { R } _ { q }$ is carried by $\{ j \leq i \}$ , the same must hold for $\pi _ { q }$ µ-almost every q, and therefore π must give zero mass to $\{ j > i \}$ . This shows (a).

Item (b) can be proven as we did for the unordered setting in Proposition 3.6. Indeed, for any $\pi \in \Pi _ { 1 } ( \vec { \mu } )$ with $\mathrm { K L } ( \pi | \mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } } ) < \dot { \infty }$ we have

$$
\mathrm { K L } ( \pi | \operatorname { R } _ { \vec { \mu } } ^ { \mathbf { M } } ) = \mathrm { K L } ( \pi | \pi _ { \vec { \mu } } ^ { \Lambda , \mathbf { M } } ) + \left. \Lambda , \int x y ^ { \top } \mathrm { d } \pi \right. - \int \log \mathbb { E } _ { \mathrm { R } _ { q } } \left[ e ^ { x ^ { \top } \Lambda y } \right] \ \vec { \mu } ( \mathrm { d } q ) .
$$

On the constraint set the last two terms do not depend on $\pi ,$ and $\operatorname { K L } ( \pi | \pi _ { \underset { \mu } { \vec { \mu } } } ^ { \Lambda , \mathrm { M } } ) = 0$ if and only if $\pi = \pi _ { \vec { \mu } } ^ { \Lambda , \mathrm { M } }$ Claim (c) is a straightforward computation. 1

The masked entries of the coupling are therefore not set to zero by hand in the rule: the mask is placed in the reference coupling $\mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } }$ , and the finiteness of the relative entropy imposes their presence in the choice rule as well. Therefore the masking in the choice rule (1.12) is not imposed on the rule by hand, but is rather a consequence of the masked reference rule $\mathrm { R } _ { \vec { \mu } } ^ { \mathrm { M } }$ . As in the unordered case, Λ plays the role of the multiplier of the constraint, the constraint level $C ^ { \mathrm { M } }$ being computed from Λ itself. In the rest of the section we only use two facts:

$$
\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \stackrel {  } { \mu } ) ~ \mathrm { d e p e n d s ~ o n l y ~ o n } ~ x _ { 1 : i } , \qquad \Big | \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \stackrel {  } { \mu } ) \Big | \leq 1 .\tag{4.2}
$$

In particular $\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } )$ does not depend on the length n of the sentence, nor on the tokens after position i.

4.2. The shifted cross-moment and the next-token model. In the previous section we have shown how the masking takes care of property (i). In this section we show how a small modification of the cross-moment statistic yields property (ii).

Lemma 4.3 (Shifted cross-moment). Let $n \geq 2 , \ { \vec { \mu } } = \iota ( n , x _ { 1 : n } ) , X = ( x _ { 1 } , \ldots , x _ { n } ) \in \mathbb { R } ^ { ( d + 1 ) \times n } , \Pi ^ { \mathrm { M } } = ( \Pi _ { i j } ^ { \mathrm { M } } ) _ { i , j \leq n }$ as in (1.12), and $J \in \mathbb { M } _ { r }$ <sub>n</sub> the lower shift matrix, $J _ { i j } = \mathbb { 1 } _ { \{ i = j + 1 \} }$ . Let $\begin{array} { r } { \vec { \mu } _ { 1 : n - 1 } = \frac { 1 } { n - 1 } \sum _ { i < n } \delta _ { ( x _ { i } , i ) } } \end{array}$ be the context and $s ( x , i ) = ( x _ { i + 1 } , i + 1 ) , i < n$ , the successor map on its atoms. Then

$$
S _ { \Lambda } ^ {  } ( \vec { \mu } ) : = \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } x _ { i + 1 } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top } = \frac { n } { n - 1 } X J \Pi ^ { \mathrm { M } } X ^ { \top } = \int x y ^ { \top } \mathrm { d } \pi _ { \vec { \mu } } ^ { \Lambda ,  } ,\tag{4.3}
$$

where $\begin{array} { r } { \pi _ { \vec { \mu } } ^ { \Lambda , \to } = ( \iota _ { \vec { \mu } } \times \iota _ { \vec { \mu } } ) _ { \# } \big ( \frac { n } { n - 1 } J \Pi ^ { \mathrm { M } } \big ) = ( s \times \mathrm { i d } ) _ { \# } \pi _ { \vec { \mu } _ { 1 : n - 1 } } ^ { \Lambda , \mathrm { M } } } \end{array}$ , with $\iota _ { \vec { \mu } } ( i ) = ( x _ { i } , i )$ , is a probability measure on $( \mathbb { S } ^ { d } \times \mathbb { N } ) ^ { 2 }$ . Its first marginal is the uniform law on $\{ ( x _ { i } , i ) : ~ 2 \leq i \leq n \}$ , and, given that the first coordinate is $( x _ { i + 1 } , i + 1 )$ , the second one is $( x _ { j } , j )$ with probability n $\begin{array} { r }  \mathrm { I } _ { i j } ^ { \mathrm { M } } , j \leq i \ \end{array}$ the key is chosen by the masked choice rule of the preceding token $( x _ { i } , i )$ , and paired with its successor $x _ { i + 1 }$

Proof. $( J \Pi ^ { \mathrm { M } } ) _ { i j } = \Pi _ { i - 1 , j } ^ { \mathrm { M } }$ for $i \geq 2$ , and the first row of $J \Pi ^ { \mathrm { M } }$ vanishes. Hence $\begin{array} { r } { X J \Pi ^ { \mathrm { M } } X ^ { \top } = \sum _ { i = 2 } ^ { n } \sum _ { j } \Pi _ { i - 1 , j } ^ { \mathrm { M } } x _ { i } x _ { j } ^ { \top } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n - 1 } x _ { i + 1 } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top } } \end{array}$ , which gives the second equality. Each row of $\Pi ^ { \mathrm { M } }$ sums to ${ \scriptstyle \frac { 1 } { n } } , \mathrm { ~ s o } \ { \scriptstyle \frac { n } { n - 1 } } J \Pi ^ { \mathrm { M } }$ is a non-negative matrix with total mass 1, whose row $i + 1$ is the masked choice $( n \Pi _ { i j } ^ { \mathrm { M } } ) _ { j = 1 , \dots , n }$ of the query $i ,$ multiplied by $\scriptstyle { \frac { 1 } { n - 1 } }$ . Its pushforward under $\iota _ { \vec { \mu } } \times \iota _ { \vec { \mu } }$ is therefore a probability measure with the stated marginal and conditional law, and $\begin{array} { r } { \int { x y } ^ { \top } \mathrm { d } \pi _ { \stackrel {  } { u } } ^ { \Lambda ,  } = \frac { \stackrel { \cdot } { n } ^ { \cdot } } { n - 1 } \sum _ { i , j } ( J \Pi ^ { \mathrm { M } } ) _ { i j } x _ { i } x _ { j } ^ { \top } } \end{array}$ , which is the third equality.

Finally, consider the context $\begin{array} { r } { \vec { \mu } _ { 1 : n - 1 } = \frac { 1 } { n - 1 } \sum _ { i < n } \delta _ { ( x _ { i } , i ) } } \end{array}$ . A query $( x _ { i } , i )$ with $i \le n - 1$ only chooses among the keys at positions $j \leq i ,$ , which all belong to the context, so its masked choice probabilities are the same in $\vec { \mu }$ and in $\vec { \mu } _ { 1 : n - 1 }$ , namely $n \Pi _ { i j } ^ { \mathrm { M } }$ . Hence $\pi _ { \vec { \frac { \mu } { \Lambda } } _ { 1 : n - 1 } } ^ { \Lambda , \mathrm { M } }$ gives mass $\frac { n \Pi _ { i j } ^ { \mathrm { M } } } { n - 1 }$ to $\{ ( ( x _ { i } , i ) , ( x _ { j } , j ) ) : j \leq i < n \}$ . The map $( s \times \operatorname { i d } )$ replaces each query by its successor and leaves the key unchanged, so the pushforward under $( s \times \mathrm { i d } ) _ { \# } \pi _ { \vec { \mu } _ { 1 : n - 1 } } ^ { \Lambda , \mathrm { M } }$ gives mass $\begin{array} { r } { \frac { n \Pi _ { i j } ^ { \mathrm { M } } } { n - 1 } = \frac { n } { n - 1 } ( J \Pi ^ { \mathrm { M } } ) _ { i + 1 , j } \mathrm { ~ t o ~ } ( ( x _ { i + 1 } , i + 1 ) , ( x _ { j } , j ) ) } \end{array}$ , which is the mass given by $\begin{array} { r } { \big ( \iota _ { \vec { \mu } } \times \iota _ { \vec { \mu } } \big ) _ { \# } \Big ( \frac { n } { n - 1 } J \Pi ^ { \mathrm { M } } \Big ) } \end{array}$

In our model we will therefore consider the shifted cross-moment statistic. This leads to the next-token attention readout.

Definition 4.4 (Next-token attention readout). For $\Lambda , V \in \mathbb { M } _ { d + 1 }$ and an ordered sentence $\vec { \mu } = \iota ( n , x _ { 1 : n } )$ let $\begin{array} { r } { \Psi _ { \Lambda , V } ^ {  } ( \vec { \mu } ) = \langle V , S _ { \Lambda } ^ {  } ( \vec { \mu } ) \rangle = \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } x _ { i + 1 } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) \ i f n \geq 2 , } \end{array}$ , and $S _ { \Lambda } ^ {  } ( { \vec { \mu } } ) = 0 , \Psi _ { \Lambda , V } ^ {  } ( { \vec { \mu } } ) = 0 i f n = 1$ The next-token model is

$$
\mathrm { P } _ { \Lambda , V } ^ {  } = \frac { e ^ { \Psi _ { \Lambda , V } ^ {  } } \vec { \bf N } } { \mathbb { E } _ { \searrow } e ^ { \Psi _ { \Lambda , V } ^ {  } } } , \qquad \mathit { t h a t ~ i s , } \qquad \mathrm { P } _ { \Lambda , V } ^ {  } ( N = n , \ X _ { 1 : n } \in \mathrm { d } x _ { 1 : n } ) = \frac { \nu ( n ) e ^ { \Psi _ { \Lambda , V } ^ {  } ( x _ { 1 : n } ) } \underline { { \mathbf { v } } } \mathbf { q } ^ { \intercal \otimes n } ( \mathrm { d } x _ { 1 : n } ) } { \sum _ { n ^ { \prime } } \nu ( n ^ { \prime } ) \int e ^ { \Psi _ { \Lambda , V } ^ {  } } \mathrm { d } \underline { { \mathbf { v } } } \mathbf { q } ^ { \intercal \otimes n ^ { \prime } } } .
$$

We sometimes write $\Psi ^ {  }$ for $\Psi _ { \Lambda , V } ^ {  } \cdot \mathrm { B y } ( 4 . 2 ) , | x _ { i + 1 } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) | \leq \| V \| _ { \mathrm { o p } } , | \Psi _ { \Lambda , V } ^ {  } | \leq \| V \| _ { \mathrm { o p } }$ and $\| S _ { \Lambda } ^ {  } \| _ { \mathrm { o p } } \leq 1$

The masked choice rule is unchanged and the decoder property (ii) changes only the readout, which difers from the causal query–key cross-moment $\begin{array} { r } { C ^ { \mathrm { M } } = \frac { 1 } { n } \sum _ { i } x _ { i } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top } } \end{array}$ of Proposition $4 . 2 \mathrm { - } ( \mathrm { c } ) \mathrm { : }$ what the query $x _ { i }$ reads is no longer paired with $x _ { i }$ itself but with the next token $x _ { i + 1 } .$ . The normalization $\scriptstyle { \frac { 1 } { n - 1 } }$ makes $S _ { \Lambda } ^ {  }$ the cross-moment of the probability coupling $\pi _ { \vec { \mu } } ^ { \Lambda , \to }$ (Lemma 4.3): the choice rule is run on the context, and each query is replaced by its successor.

We are now ready to prove Item (ii) from the Meta-Theorem in the ordered setting.

Theorem 4.5 (Entropic minimization of language laws with shifted cross-moment statistics). Let $\Lambda \in \mathbb { M } _ { d + 1 }$ be arbitrary, including $\Lambda = 0$

(a) (Non-degeneracy) Let $n \ge 2 . ~ I f ~ \langle V , S _ { \Lambda } ^ {  } \rangle$ is constant on $( \mathbb { S } ^ { d } ) ^ { n }$ for some $V \in \mathbb { M } _ { d + 1 }$ , then $V = 0$

(b) Assume that $\nu ( n ) > 0$ for some $n \geq 2 .$ , and let $\mathcal { K } _ { \Lambda } ^ {  }$ be the closed convex hull of $S _ { \Lambda } ^ {  } ( \mathrm { s u p p N } )$ . Then $V \mapsto { \mathcal { M } } _ { \Lambda } ^ {  } ( V ) : = \mathbb { E } _ { \mathrm { P } _ { \Lambda , V } ^ {  } } S _ { \Lambda } ^ {  }$ is a real-analytic difeomorphism of $\mathbb { M } _ { d + 1 }$ onto int $\mathcal { K } _ { \Lambda } ^ {  } . \mathrm { ~ } I f \bar { \mathbb { E } } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } ^ {  } \in \mathrm { i n t } \mathcal { K } _ { \Lambda } ^ {  }$ the problem

$$
\operatorname* { m i n } \big \{ \operatorname { K L } ( \mathrm { P } | \vec { \mathbf { N } } ) : \mathrm { ~ P } \in \mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } ) ) , \ \mathbb { E } _ { \mathrm { P } } S _ { \Lambda } ^ { \to } = \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } ^ { \to } \big \}
$$

has the unique solution $\mathrm { P } _ { \Lambda , V ^ { \star } } ^ {  }$ , with $V ^ { \star } = V ^ { \star } ( \Lambda ) = ( \mathcal { M } _ { \Lambda } ^ {  } ) ^ { - 1 } ( \mathbb { E } _ { \mathbf { P } ^ { \star } } S _ { \Lambda } ^ {  } )$ , and the Pythagorean identity (2.2) holds with $\vec { \bf N } , S _ { \Lambda } ^ {  }$ and $\mathrm { P } _ { \Lambda , V } ^ {  } ,$ <sub>⋆</sub> in place of N, Φ and $\mathrm { P } _ { \eta _ { c } }$

(c) (Every value matrix is realized) Under the assumption of (b), for every $V \in \mathbb { M } _ { d + 1 }$ the language $\mathbf { P } ^ { \star } = \mathbf { P } _ { \Lambda , V } ^ {  }$ has value matrix $V ^ { \star } ( \Lambda ) = V$

Proof. In $\begin{array} { r } { \langle V , S _ { \Lambda } ^ {  } \rangle = \frac { 1 } { n - 1 } \sum _ { i < n } x _ { i + 1 } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) } \end{array}$ , the token $x _ { n }$ appears only in the term $i = n - 1$ because $\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; { \vec { \mu } } )$ depends on $x _ { 1 : i }$ only. If we assume it to be constant and evaluate it on $( x _ { 1 } , \ldots , x _ { n } )$ and on $( x _ { 1 } , \ldots , - x _ { n } )$ we immediately deduce $x _ { n } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) = 0$ for any $x _ { n } \in \mathbb { S } ^ { d }$ . Therefore V Attn $\scriptstyle \Lambda \left( ( x _ { n - 1 } , n - 1 ) ; { \vec { \mu } } \right) = 0$ for any $\left( x _ { 1 } , \ldots , x _ { n - 1 } \right)$ . In the constant configuration $x _ { 1 } = \cdot \cdot \cdot = x _ { n - 1 } = x$ every key equals $x ,$ so $\mathrm { A t t n } _ { \Lambda } { \big ( } ( x _ { n - 1 } , n - 1 ) ; { \vec { \mu } } { \big ) } = x _ { ; }$ , and we get $V x = 0$ for every $x \in \mathbb { S } ^ { d }$ , so $V = 0$ . This proves (a).

Claim (b) can be proven as Theorem 2.1, whose proof only uses boundedness and measurability of the statistic, the fact that it is not a.s. constant in any direction under the null law, and the identification of the convex support of its image law. In particular it does not rely on the compactness of ${ \mathcal { P } } ( { \mathcal { P } } ( \mathbb { S } ^ { d } ) )$ . Since ι is an injective Borel map on the Polish space , it is a Borel isomorphism onto its (Borel) image by the Lusin–Souslin theorem, and it preserves relative entropies. We can therefore work indiferently with laws of $( N , X _ { 1 : N } )$ on $\begin{array} { r } { \mathcal { O } = \bigcup _ { n > 1 } ( \mathbb { S } ^ { d } ) ^ { n } } \end{array}$ here, instead of with laws on $\mathcal { P } ( \mathbb { S } ^ { d } \times \mathbb { N } )$ concentrated on $\iota ( { \mathcal { O } } )$ . Notice that under $\vec { \bf N } .$ , the law of $( N , X _ { 1 : N } )$ is $\begin{array} { r } { \sum _ { n } \nu ( n ) \delta _ { n } \otimes \underline { { \mathbf { v o l } } } ^ { \otimes n } } \end{array}$ . We show the applicability of Theorem 2.1 in this setting.

(1) Boundedness of the shifted cross-moment $S _ { \Lambda } ^ {  }$ is trivial since it takes values in the compact set $\{ S \in$ $\mathbb { M } _ { d + 1 } : ~ \| S \| _ { \mathrm { F } } \leq 1 \}$

(2) Non-degeneracy follows from (a). Indeed, if we suppose that $\langle V , S _ { \Lambda } ^ {  } \rangle$ is $\vec { \bf N } .$ -almost surely constant then for any fixed $n \geq 2$ with $\nu ( n ) > 0 , \langle V , S _ { \Lambda } ^ {  } \rangle$ is $\underline { { \mathrm { v o l } } } ^ { \otimes n }$ -almost surely constant on $( \mathbb { S } ^ { d } ) ^ { n }$ and by continuity constant everywhere in $( \mathbb { S } ^ { d } ) ^ { n }$ . By (a) we would then deduce $V = 0$

(3) We need to show that conv $\begin{array} { r } { \cdot ( \mathrm { s u p p } ( ( S _ { \Lambda } ^ {  } ) _ { \# } \vec { \bf N } ) ) = \overline { { \mathrm { c o n v } } } ( \bigcup _ { n \in \mathrm { s u p p } ( \nu ) } S _ { \Lambda } ^ {  } \big ( ( \mathbb { S } ^ { d } ) ^ { n } \big ) ) = K _ { \Lambda } ^ {  } } \end{array}$ . The image law of $S _ { \Lambda } ^ {  }$ under the prior $\vec { \bf N }$ is the mixture $\begin{array} { r } { \sum _ { n } \nu ( n ) ( S _ { \Lambda } ^ {  } ) _ { \# \underline { { \bf V } } 0 } ] ^ { \otimes n } } \end{array}$ . Since $( S _ { \Lambda } ^ {  } ) _ { \# \underline { { \bf V o l } } } ^ { \otimes n }$ has support $S _ { \Lambda } ^ {  } ( ( \mathbb { S } ^ { d } ) ^ { n } )$ the support of the mixture is the closure $\overline { { \cup _ { n \in \mathrm { s u p p } ( \nu ) } S _ { \Lambda } ^ {  } ( ( \mathbb { S } ^ { d } ) ^ { n } ) } }$ of the union of these supports, and its closed convex hull is exactly $\mathcal { K } _ { \Lambda } ^ {  }$

Hence the proof of Theorem 2.1 applies verbatim, which proves (b).

Finally, Item (c) can be proven as in Theorem 3.11-(c): for $\mathbf { P } ^ { \star } = \mathbf { P } _ { \Lambda , V } ^ { \to }$ the constraint $\mathcal { M } _ { \Lambda } ^ {  } ( V )$ lies in int $\mathcal { K } _ { \Lambda } ^ {  }$ and $V ^ { \star } ( \Lambda ) = ( \mathcal { M } _ { \Lambda } ^ {  } ) ^ { - 1 } ( \mathcal { M } _ { \Lambda } ^ {  } ( V ) ) = V$

As in Remark 3.12, the interior assumption in (b) holds for the empirical shifted cross-moment ${ \widehat { S _ { \Lambda } ^ {  } } } =$ $\textstyle { \frac { 1 } { J } } \sum _ { j = 1 } ^ { J } S _ { \Lambda } ^ { \to } ( { \vec { \mu } } _ { j } )$ of a corpus as soon as the centered matrices $S _ { \Lambda } ^ {  } ( \vec { \mu } _ { j } ) - \widehat { S _ { \Lambda } ^ {  } } , j \leq J ,$ span $\mathbb { M } _ { d + 1 }$

We can now conclude this section with the proof of Item (iii) from the Meta-Theorem in the ordered setting. The following result applies to any Gibbs tilt $\mathrm { P } _ { \Lambda , V } ^ {  }$ . For any fixed Λ, this in particular applies to $V ^ { \star } = V ^ { \star } ( \Lambda )$ as in Theorem 4.5, and hence to our optimizer $\mathrm { P } _ { \Lambda , V } ^ {  } ,$ <sub>⋆</sub> for (1.16), which corresponds to Item (iii) of the Meta-Theorem for the ordered model.

Theorem 4.6 (Last token: one decoder head, exactly). Let $n \geq 2$ with $\nu ( n ) > 0 , \Lambda , V \in \mathbb { M } _ { d + 1 } , x _ { 1 : n - 1 } \in ( \mathbb { S } ^ { d } ) ^ { n - 1 }$ and let Attn $\mathbf { \Omega } _ { \cdot \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } )$ be the causal attention output (4.1) of the last context token, which is a function of $x _ { 1 : n - 1 } ~ o n l y$

(a) (Exact law) Under $\mathrm { P } _ { \Lambda , V } ^ {  } ,$ given $N = n$ and $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , the last token has density with respect to vol

$$
p ( \boldsymbol { y } ) = \frac { \exp \big ( \frac { 1 } { n - 1 } \boldsymbol { y } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \big ) } { \int \exp \big ( \frac { 1 } { n - 1 } \boldsymbol { y } ^ { \prime \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \big ) \underline { { \operatorname { v o l } ( \mathrm { d } \boldsymbol { y } ^ { \prime } ) } } } , \qquad \boldsymbol { y } \in \mathbb { S } ^ { d } .\tag{4.4}
$$

$I f V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \not = 0$ , this is the von Mises–Fisher law with mean direction VAttn $\operatorname { \mathrm { 1 } } _ { \Lambda } ( ( x _ { n - 1 } , n - $ $1 ) ; { \vec { \mu } } )$ (normalized) and concentration $| V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \stackrel {  } { \mu } ) | / ( n - 1 )$

(b) (Score) The Riemannian gradient of log p on $\mathbb { S } ^ { d }$ is

$$
\nabla _ { \mathsf { S } ^ { d } } \log p ( y ) = { \frac { 1 } { n - 1 } } P _ { y } ^ { \perp } V \mathrm { A t t n } _ { \Lambda } { \big ( } ( x _ { n - 1 } , n - 1 ) ; { \vec { \mu } } { \big ) } = { \frac { 1 } { n - 1 } } P _ { y } ^ { \perp } \sum _ { j = 1 } ^ { n - 1 } { \frac { e ^ { x _ { n - 1 } ^ { \top } \Lambda x _ { j } + M _ { n - 1 , j } } } { \sum _ { \ell = 1 } ^ { n - 1 } e ^ { x _ { n - 1 } ^ { \top } \Lambda x _ { \ell } + M _ { n - 1 , \ell } } } } V x _ { j } ,\tag{4.5}
$$

with the mask $M \in \{ 0 , - \infty \} ^ { n \times n }$ defined as $M _ { i j } = \left( - \infty \right) \mathbb { 1 } _ { i < j }$ , with the convention $e ^ { - \infty } = 0$

(c) (Transformer reading) For $\Lambda = Q ^ { \top } K$ , the logit $y ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \stackrel {  } { \mu } )$ in (4.4) is the inner product of the candidate y with the output, at position $n - 1$ (the last position of the context), of the causally masked version of the head (1.1), with query matrix Q, key matrix K and value matrix $V$

Proof. By Definition $4 . 4 ,$ given $N = n$ the ordered tokens have density proportional to $e ^ { \Psi _ { \Lambda , V } ^ {  } }$ with respect to $\underline { { \mathrm { v o l } } } ^ { \otimes { \bar { n } } }$ , and $\nu ( n ) > 0$ . Conditioning on $X _ { 1 : n - 1 } = x _ { 1 : n - 1 } \ \mathrm { ( B a y e s ) } ^ { }$ formula, as in Proposition $2 . 2 \mathrm { - ( b ) \ ) }$ gives the density proportional to $y \mapsto e ^ { \Psi _ { \Lambda , V } ^ {  } ( x _ { 1 : n - 1 } , y ) }$ . In Definition 4.4 with $x _ { n } = y$ , the terms $i \le n - 2$ involve $x _ { i + 1 }$ with $i + 1 \le n - 1$ and $\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } )$ with $i \le n - 2$ , which depend only on $x _ { 1 : n - 1 }$ by (4.2); they cancel in the normalization. The remaining term $i = n - 1$ is $\begin{array} { r l } {  { \frac { 1 } { n - 1 } y ^ { \top } V \mathrm { A t t n } _ { \Lambda } \big ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } \big ) } } \end{array}$ , and $\mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \stackrel {  } { \mu } )$ depends only on $x _ { 1 : n - 1 }$ . The last token appears neither among the keys of a query, since every key of $\mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } )$ has index at most $i \le n - 1$ , nor as a query, since the queries are $x _ { 1 } , \ldots , x _ { n - 1 }$ . This proves $\mathrm { ( a ) }$ Claims (b) and (c) are straightforward computations.

Notice that the readout depends on the candidate y only through the linear term $\begin{array} { r } { \frac { 1 } { n - 1 } y ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - } \end{array}$ $1 ) ; { \vec { \mu } } )$ , so the score is the tangential part of a vector that does not depend on $y .$ The candidate is neither a key nor a query of the statistic, and both mechanisms that produce the key-side term and the correction term are absent. Together with Theorem 4.5, this proves Items (ii) and (iii) of the Meta-Theorem for the masked model. Item (i) is Proposition 4.2, given the masked Luce postulate and the lowest-order truncation of Remark 4.1.

Remark 4.7 (Last token versus autoregressive next token). Theorem $\it 4 . 6$ gives the law of $X _ { n }$ given $X _ { 1 : n - 1 }$ and $N = n ,$ , i.e. the law of the last token of a sentence whose length is known. It does not give the law of an intermediate token given its past. For $1 \leq i \leq n - 2$ with $\nu ( n ) > 0$ , integrating out $X _ { i + 2 : n }$ in Definition 4.4 gives

$$
\begin{array} { r } { \mathrm { P } _ { \Lambda , V } ^ {  } ( X _ { i + 1 } \in \mathrm { d } y \mid N = n , \ X _ { 1 : i } = x _ { 1 : i } ) \ \propto \ \exp ( \frac { 1 } { n - 1 } \ y ^ { \top } V \mathrm { A t t } \mathrm { n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ) G _ { i + 1 } ( x _ { 1 : i } , y ) \underline { { \mathrm { v o l } } } ( \mathrm { d } y ) , } \end{array}
$$

$$
G _ { i + 1 } ( x _ { 1 : i } , y ) = \int \exp \Big ( \frac { 1 } { n - 1 } \sum _ { k = i + 1 } ^ { n - 1 } x _ { k + 1 } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { k } , k ) ; \vec { \mu } ) \Big ) \underline { { \mathrm { v o l } } } ^ { \otimes ( n - i - 1 ) } ( \mathrm { d } x _ { i + 2 : n } ) \Big | _ { x _ { i + 1 } = y } ,
$$

and $G _ { i + 1 }$ depends on $y ,$ which is the query at position $i + 1$ and a key of all later queries. In particular the conditional law of intermediate tokens is not a von Mises–Fisher law with attention parameter: a single attention head governs the law of the final token only; we plan to address the extension to intermediate tokens in future work.

Remark 4.8 (No self-pairing). Let $\Lambda = \kappa !$ with $\kappa  \infty$ , and consider an ordered sentence with pairwise distinct tokens, which is the case $\vec { \bf N }$ -almost surely. The masked weights nΠ $\begin{array} { r } { \mathsf { I } _ { i j } ^ { \mathrm { M } } \propto e ^ { \kappa x _ { i } ^ { \top } x _ { j } } , j \le i ; } \end{array}$ concentrate on the largest exponent, which is attained only at $j = i$ since $x _ { i } ^ { \top } x _ { i } = 1 > x _ { i } ^ { \top } x _ { j } ~ f o r ~ j \neq i$ . Hence each token reads itself in the limit, $\dim _ { \kappa \mathsf { I } } ( ( x _ { i } , i ) ; { \vec { \mu } } ) \to x _ { i }$ , and

$$
S _ { \kappa \mathrm { l } } ^ {  } ( \vec { \mu } )  \frac { 1 } { n - 1 } \sum _ { i < n } x _ { i + 1 } x _ { i } ^ { \top } = \frac { 1 } { n - 1 } X J X ^ { \top } , \qquad \Psi _ { \kappa \mathrm { l } , V } ^ {  } ( \vec { \mu } )  \frac { 1 } { n - 1 } \sum _ { i < n } x _ { i + 1 } ^ { \top } V x _ { i } , \qquad \mathrm { i } = \frac { n - 1 } { n - 1 } X J X ^ { \top } .
$$

the bigram moment and the bigram readout. The limit readout is not constant, and it has mean 0 under $\vec { \textbf { N } } f o r$ every $V _ { i }$ , because $x _ { i + 1 }$ is centered and independent of $x _ { i } ,$ so the limit model $\propto e ^ { \frac { 1 } { n - 1 } \sum _ { i } x _ { i + 1 } ^ { \top } V x _ { i } } \vec { \bf N }$ is a non-trivial bigram model. By contrast, in the bidirectional model each read is paired with the reading token itself, so the limit model is $\propto e ^ { \frac { 1 } { n } \sum _ { i } x _ { i } ^ { \top } V x _ { i } } \mathbf { N }$ which is a tilt of the null model without interaction between tokens. This is the precise sense in which the shifted readout “removes self-reading”: every token is still a key of itself in the masked choice rule, but no token is paired with its own read.

In Theorem 4.6 the concentration of the last-token law is $\left| V ^ { \star } \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \right| / ( n - 1 )$ , which vanishes for long sentences since a single matrix $V ^ { \star }$ serves all lengths. This is due to the normalization $\scriptstyle { \frac { 1 } { n - 1 } }$ of the statistic.

One may instead constrain the extensive statistic

$$
T _ { \Lambda } ^ {  } ( \vec { \mu } ) : = ( n - 1 ) S _ { \Lambda } ^ {  } ( \vec { \mu } ) = \sum _ { i = 1 } ^ { n - 1 } x _ { i + 1 } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top } .
$$

For this modified statistic we are able to show the following result.

Proposition 4.9 (Extensive next-token statistic). Assume that $\nu ( n ) > 0$ for some n $, \geq 2$ and that $\begin{array} { r } { \sum _ { n } \nu ( n ) e ^ { t n } < } \end{array}$ for every $t > 0 \ ( e . g .$ . a Poisson law shifted by one). For $\vec { \mu } = \iota ( n , x _ { 1 : n } )$ let $\mathrm { P } _ { \Lambda , V } ^ { \Rightarrow } \propto e ^ { \langle V , T _ { \Lambda } ^ { \to } \rangle } \vec { \mathbf { N } }$

(a) $\Gamma ^ { \Rightarrow } ( V ) = \log \mathbb { E } _ { \vec { \mathbf { v } } } e ^ { \langle V , T _ { \Lambda } ^ { \to } \rangle }$ is finite and real-analytic on $\mathbb { M } _ { d + 1 }$ , and $V \mapsto \mathbb { E } _ { \mathrm { P } _ { \Lambda , V } ^ { \Rightarrow } } T _ { \Lambda } ^ { \to }$ is a difeomorphism of

$\mathbb { M } _ { d + 1 }$ onto the interior of the closed convex hull $\kappa _ { \Lambda } ^ { \Rightarrow }$ of the support of $( T _ { \Lambda } ^ {  } ) _ { \# } \vec { \bf N }$

(b) For $c \in \mathrm { i n t } { \mathcal { K } } _ { \Lambda } ^ { \Rightarrow } , \mathrm { P } _ { \Lambda , V } ^ { \Rightarrow } .$ <sub>⋆</sub> is the unique minimizer of $\mathrm { K L } ( \mathrm { P } | \vec { \bf N } )$ under $\mathbb { E } _ { \mathrm { P } } T _ { \Lambda } ^ {  } = c ,$ where $V ^ { \star }$ is the unique matrix such that $\mathbb { E } _ { \mathrm { P } _ { \Lambda , V ^ { \star } } ^ { \Rightarrow } } T _ { \Lambda } ^ {  } = c$

(c) Under $\mathrm { P } _ { \Lambda , V } ^ { \Rightarrow }$ , given $N = n \geq 2$ with $\nu ( n ) > 0$ and $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , the last token has density

$$
p ( y ) \propto \exp \left( y ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) \right)
$$

with respect to vol, and score $P _ { y } ^ { \perp } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } )$

Proof. We write $T = T _ { \Lambda } ^ {  }$ and $\Gamma = \Gamma ^ { \Rightarrow }$ for brevity, and proceed in four steps.

Step 1: the log-partition function is finite and real-analytic. On $\{ N = n \}$ the matrix T is a sum of $n - 1$ matrices $x _ { i + 1 } \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) ^ { \top }$ of Frobenius norm at most 1, hence $\Vert T \Vert _ { \mathrm { F } } \leq n - 1$ . Therefore, for every complex matrix $\eta ,$

$$
\mathbb { E } _ { \vec { \mathbf { N } } } \left| e ^ { \langle \eta , T \rangle } \right| \le \sum _ { n \ge 1 } \nu ( n ) e ^ { ( n - 1 ) \| \eta \| _ { \mathrm { F } } } < \infty
$$

by the assumption on ν. As in the proof of Theorem 2.1, this domination allows us to diferentiate under the integral sign: $\bf \dot { V } \mapsto \mathbb { E } _ { \frac { \pi } { N } } e ^ { \langle V , T \rangle }$ extends to an entire function, Γ is real-analytic on $\mathbb { M } _ { d + 1 }$ , and

$$
\begin{array} { r } { \nabla \Gamma ( V ) = \mathbb { E } _ { \mathrm { P } _ { \Lambda , V } ^ { \Rightarrow } } T , \qquad \nabla ^ { 2 } \Gamma ( V ) = \mathrm { C o v } _ { \mathrm { P } _ { \Lambda , V } ^ { \Rightarrow } } ( T ) . } \end{array}
$$

Step 2: non-degeneracy. We show that $\nabla ^ { 2 } \Gamma ( V )$ is positive definite. Assume that $\langle A , T \rangle$ is $\vec { \bf N } { \bf - a . s . }$ . constant for some $A \in \mathbb { M } _ { d + 1 }$ , and fix $n \geq 2$ with $\nu ( n ) > 0$ . On $\{ N = n \}$ we have $\langle A , T \rangle = ( n - 1 ) \langle A , S _ { \Lambda } ^ {  } \rangle$ , so $\langle A , S _ { \Lambda } ^ {  } \rangle$ is $\underline { { \mathbf { v o l } } } ^ { \otimes n }$ -a.s. constant on $( \mathbb { S } ^ { d } ) ^ { n }$ . Being continuous, it is constant everywhere on $( \mathbb { S } ^ { d } ) ^ { n }$ , and Theorem 4.5-(a) gives $A = 0$ . Hence $\operatorname { C o v } _ { \operatorname { P } _ { \Lambda , V } } ( T )$ is positive definite and Γ is strictly convex.

Step 3: proof $o f \left( a \right)$ . By Step 1 the natural parameter space of the exponential family $\{ \mathrm { P } _ { \Lambda , V } ^ { \Rightarrow } \} _ { V \in \mathbb { M } _ { d + 1 } }$ is the whole of $\mathbb { M } _ { d + 1 } .$ , which is open: the family is regular, hence steep, and it is minimal by Step 2. By [Bro86, Theorem 3.6], the mean map Γ is a bijection of $\mathbb { M } _ { d + 1 }$ onto int $\mathcal { K } _ { \Lambda } ^ { \Rightarrow }$ ; it is a difeomorphism by the inverse function theorem, since its Jacobian $\nabla ^ { 2 } \Gamma$ is invertible.

Step 4: proof of (b). Let $c \in \mathrm { i n t } { K _ { \Lambda } ^ { \Rightarrow } }$ and let $V ^ { \star }$ be the unique matrix with $\nabla \Gamma ( V ^ { \star } ) = c ,$ , given by Step 3. Then $\mathrm { P } ^ { \star } : = \mathrm { P } _ { \Lambda , V ^ { \star } } ^ { \Rightarrow }$ satisfies the constraint and $\mathrm { K L } ( \mathrm { P } ^ { \star } | \vec { \mathrm { N } } ) = \langle V ^ { \star } , c \rangle - \Gamma ( V ^ { \star } ) < \infty$ . Now let P be admissible with $\mathrm { K L } ( \mathrm { P } | \vec { \mathbf { N } } ) < \infty$ (otherwise there is nothing to prove). $\mathrm { B y }$ the entropy inequality $\mathbb { E } _ { \mathrm { P } } f \le \mathrm { K L } ( \mathrm { P } | \vec { \mathbf { N } } ) + \log \mathbb { E } _ { \vec { \mathbf { N } } } e ^ { f }$ applied to $f = \Vert T \Vert _ { \mathrm { F } }$ , and by Step 1,

$$
\mathbb { E } _ { \mathrm { P } } \| T \| _ { \mathrm { F } } \le \mathrm { K L } ( \mathrm { P } | \overset {  } { \mathbf { N } } ) + \log \sum _ { n \ge 1 } \nu ( n ) e ^ { n - 1 } < \infty .
$$

Hence log $\begin{array} { r } { { \frac { \mathrm { d } \mathrm { P } ^ { \star } } { \mathrm { d } \vec { \mathsf { N } } } } = \langle V ^ { \star } , T \rangle - \Gamma ( V ^ { \star } ) } \end{array}$ is P-integrable, with integral $\langle V ^ { \star } , c \rangle - \Gamma ( V ^ { \star } )$ . Since $\mathbf { P } \ll \vec { \mathbf { N } }$ and $\vec { \bf N }$ is equivalent to $\mathrm { P } ^ { \star }$ , we have log $\begin{array} { r } { { \frac { \mathrm { d } \mathrm { P } } { \mathrm { d } \vec { \bf N } } } = \log { \frac { \mathrm { d } \mathrm { P } } { \mathrm { d } \mathrm { P } ^ { \star } } } + \log { \frac { \mathrm { d } \mathrm { P } ^ { \star } } { \mathrm { d } \vec { \bf N } } } } \end{array}$ , and integrating against P gives the Pythagorean identity

$$
\mathrm { K L } ( \mathrm { P } \mid \vec { \bf N } ) = \mathrm { K L } ( \mathrm { P } \mid \mathrm { P } ^ { \star } ) + \langle V ^ { \star } , c \rangle - \Gamma ( V ^ { \star } ) = \mathrm { K L } ( \mathrm { P } \mid \mathrm { P } ^ { \star } ) + \mathrm { K L } ( \mathrm { P } ^ { \star } \mid \vec { \bf N } ) .
$$

Thus $\mathrm { K L } ( \mathrm { P } | \vec { \mathbf { N } } ) \geq \mathrm { K L } ( \mathrm { P } ^ { \star } | \vec { \mathbf { N } } )$ , with equality if and only if $\mathrm { P } = \mathrm { P } ^ { \star }$

Proof of (c). As in Theorem 4.6, given $N = n$ and $X _ { 1 : n - 1 } = x _ { 1 : n - 1 }$ , the density of the last token is proportional to $y \mapsto e ^ { \langle } V , T ( \iota ( n , ( x _ { 1 : n - 1 } , y ) ) ) \rangle$ . By (4.2), the terms $i \le n - 2$ of $T$ do not depend on $x _ { n } = y$ , and they cancel in the normalization. The remaining term $i = n - 1$ equals $y ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } )$ ■

## Appendix A. Non-degeneracy of attention readouts

In this appendix we show the non-degeneracy of attention readouts, a property needed for applying Theorem 2.1 to our KL minimization problem in $\mathcal { P } ( \mathcal { P } ( \mathbb { S } ^ { d } ) )$ . We start with a technical lemma that computes how the attention cross-moment and readout behave on two-point sentences.

Lemma A.1. Let Λ, $V \in \mathbb { M } _ { d + 1 } , f x \ x , y \in \mathbb { S } ^ { d }$ and consider $\mu _ { t } = ( 1 - t ) \delta _ { x } + t \delta _ { y } , t \in [ 0 , 1 ]$ . Then

$$
\Psi _ { \Lambda , V } ( \mu _ { t } ) = \left( 1 - t \right) x ^ { \top } V x + t y ^ { \top } V y + t ( 1 - t ) \Big [ \frac { e ^ { x ^ { \top } \Lambda y } } { Z _ { x } ( t ) } x ^ { \top } V ( y - x ) + \frac { e ^ { y ^ { \top } \Lambda x } } { Z _ { y } ( t ) } y ^ { \top } V ( x - y ) \Big ] ,\tag{A.1}
$$

with $Z _ { x } ( t ) = ( 1 - t ) e ^ { x ^ { \top } \Lambda x } + t e ^ { x ^ { \top } \Lambda y }$ and $Z _ { y } ( t ) = ( 1 - t ) e ^ { y ^ { \top } \Lambda x } + t e ^ { y ^ { \top } \Lambda y }$ being the partition functions of the two queries associated to $\pi _ { \mu _ { t } } ^ { \Lambda }$ . As a consequence we have

$$
\frac { \mathrm { d } } { \mathrm { d } t } \Big \vert _ { t = 0 } \Psi _ { \Lambda , V } ( \mu _ { t } ) = e ^ { x ^ { \top } \Lambda ( y - x ) } x ^ { \top } V ( y - x ) + y ^ { \top } V x - x ^ { \top } V x , \qquad \frac { \mathrm { d } } { \mathrm { d } t } \Big \vert _ { t = 0 } S _ { \Lambda } ( \mu _ { t } ) = e ^ { x ^ { \top } \Lambda ( y - x ) } x ( y - x ) ^ { \top } + y x ^ { \top } - x x ^ { \top } .\tag{A.2}
$$

Proof. Given $x , y \in \mathbb { S } ^ { d }$ and $t \in [ 0 , 1 ]$ , the corresponding choice rule $\begin{array} { r } { \pi _ { \mu _ { t } } ^ { \Lambda } = \sum _ { a , b \in \{ x , y \} } P _ { a b } \delta _ { a } \otimes \delta _ { b } } \end{array}$ can be represented by the matrix

$$
P = \left( { P _ { x x } } \quad P _ { x y } \right) = \left( \begin{array} { c c } { \left( 1 - t \right) \frac { \left( 1 - t \right) { e ^ { x ^ { \top } \Lambda x } } } { Z _ { x } \left( t \right) } } & { \left( 1 - t \right) \frac { t { e ^ { x ^ { \top } \Lambda y } } } { Z _ { x } \left( t \right) } } \\ { { t \frac { \left( 1 - t \right) { e ^ { y ^ { \top } \Lambda x } } } { Z _ { y } \left( t \right) } } } & { { t \frac { t { e ^ { y ^ { \top } \Lambda y } } } { Z _ { y } \left( t \right) } } } \end{array} \right) ,\tag{A.3}
$$

with rows indexed by the query and columns by the key. Consequently, with $X = ( x , y ) \in \mathbb { R } ^ { ( d + 1 ) \times 2 }$

$$
S _ { \Lambda } ( \mu _ { t } ) = X P X ^ { \top } = ( 1 - t ) x x ^ { \top } + t y y ^ { \top } + t ( 1 - t ) \Big [ \frac { e ^ { x ^ { \top } \Lambda y } } { Z _ { x } ( t ) } x ( y - x ) ^ { \top } + \frac { e ^ { y ^ { \top } \Lambda x } } { Z _ { y } ( t ) } y ( x - y ) ^ { \top } \Big ] ,\tag{A.4}
$$

which combined with $\Psi _ { \Lambda , V } ( \mu _ { t } ) = \left. V , S _ { \Lambda } ( \mu _ { t } ) \right.$ shows (A.1). Since $Z _ { x } ( 0 ) = e ^ { x ^ { \top } \Lambda x }$ and $Z _ { y } ( 0 ) = e ^ { y ^ { \top } \Lambda x }$ , diferentiating at t = 0 gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } \Big | _ { t = 0 } S _ { \Lambda } ( \mu _ { t } ) = \frac { e ^ { x ^ { \top } \Lambda y } } { e ^ { x ^ { \top } \Lambda x } } x ( y - x ) ^ { \top } + y x ^ { \top } - x x ^ { \top } ,
$$

from which (A.2) immediately follows.

We will see below that the expression of the derivatives in (A.2) can be rearranged in order to take the form of combinations of trigonometric polynomials combined with exponentials. The main goal of this section is to show that if linear combinations of those derivatives are null, then the coeficients are equal to zero. The next lemma is the general linear independence result we are going to rely on, which states that exponentials of distinct first-degree trigonometric polynomials are linearly independent over trigonometric polynomials.

Lemma A.2 (Exponential polynomials on a circle). Let $( a _ { 1 } , b _ { 1 } ) , \ldots , ( a _ { m } , b _ { m } ) \in \mathbb { R } ^ { 2 }$ be pairwise distinct and let $p _ { 1 } , \ldots , p _ { m }$ be trigonometric polynomials. $\begin{array} { r } { I f \sum _ { k = 1 } ^ { m } p _ { k } ( \theta ) \dot { e } ^ { a _ { k } \cos \theta + b _ { k } \sin \theta } = 0 } \end{array}$ for every $\theta \in \mathbb { R }$ , then $p _ { 1 } = \cdots =$ $p _ { m } = 0$

Proof. Put $z = e ^ { i \theta }$ and $\alpha _ { k } = ( a _ { k } - i b _ { k } ) / 2 \in \mathbb { C }$ The map $( a , b ) \mapsto ( a - i b ) / 2$ is injective, so the $\alpha _ { k }$ are pairwise distinct. Each trigonometric polynomial $p _ { k }$ can be written $p _ { k } ( \theta ) = L _ { k } ( e ^ { i \theta } )$ for a Laurent polynomial $\begin{array} { r } { L _ { k } ( z ) = \sum _ { j = - J } ^ { J } c _ { k , j } z ^ { j } } \end{array}$ , with $J \in \mathbb { N }$ large enough for all $k ,$ and $L _ { k } \equiv 0$ if and only if $p _ { k } \equiv 0$ . Consider

$$
H ( z ) = \sum _ { k = 1 } ^ { m } L _ { k } ( z ) \exp { \big ( } \alpha _ { k } z + \bar { \alpha } _ { k } / z { \big ) } , \qquad z \in \mathbb { C } \setminus \{ 0 \} .
$$

Each $L _ { k }$ and each $z \mapsto \alpha _ { k } z + \bar { \alpha } _ { k } / z$ is holomorphic on $\mathbb { C } \backslash \{ 0 \}$ , hence so is H. For $| z | = 1$ we have $1 / z = { \bar { z } }$ so that $\alpha _ { k } z + \bar { \alpha } _ { k } / z = 2 \operatorname { R e } ( \alpha _ { k } z ) = a _ { k }$ cos $\theta + b _ { k }$ sin θ for $z = e ^ { i \theta }$ , and therefore $\begin{array} { r } { H ( e ^ { i \theta } ) = \dot { \sum } _ { k } p _ { k } ( \theta ) e ^ { a _ { k } \cos \theta + b _ { k } \sin \theta } = 0 } \end{array}$ for every $\theta \in \mathbb { R }$ . Since $\mathbb { C } \setminus \{ 0 \}$ is open and connected and H vanishes on the unit circle, which has accumulation points in $\mathbb { C } \setminus \{ 0 \}$ , the identity theorem gives $H \equiv 0 \ \mathrm { o n } \ \mathbb { C } \setminus \{ 0 \}$ .

Assume, by contradiction, that not all $L _ { k }$ vanish, and discard the indices with $L _ { k } \equiv 0$ . We keep the notation $k = 1 , \ldots , m$ for the remaining ones, so that every $L _ { k }$ is a non-zero Laurent polynomial, the $\alpha _ { k }$ are still pairwise distinct, and still $H \equiv 0$ . For every pair $j \neq k$ , writing $\alpha _ { j } - \alpha _ { k } = | \alpha _ { j } - \alpha _ { k } | e ^ { i \varphi _ { j k } }$ , we have $\mathrm { R e } \big ( ( \alpha _ { j } - \alpha _ { k } ) e ^ { i \psi } \big ) = | \alpha _ { j } - \alpha _ { k } | \cos ( \varphi _ { j k } + \psi )$ , which vanishes for exactly two values of ψ modulo 2π. Choose $\psi \in$ R outside these finitely many values, so that the numbers $\operatorname { R e } ( \alpha _ { k } e ^ { i \psi } ) , k \leq m$ , are pairwise distinct. Let $k ^ { \star }$ be the index at which the largest of them, $M ^ { \star } = \operatorname { R e } ( \alpha _ { k ^ { \star } } e ^ { i \psi } )$ , is attained, and let

$$
\eta = \operatorname* { m i n } _ { k \neq k ^ { \star } } \left( M ^ { \star } - \mathrm { R e } ( \alpha _ { k } e ^ { i \psi } ) \right) > 0 ,
$$

with $\eta = + \infty \mathrm { i f } \ m = 1$

For every $r > 0$ we have $H ( r e ^ { i \psi } ) = 0$ . Dividing by $e ^ { \alpha _ { k } \star r e ^ { i \psi } }$ and isolating the term $k ^ { \star }$ , we obtain

$$
\begin{array} { r l } & { L _ { k ^ { \star } } ( r e ^ { i \psi } ) e ^ { \bar { \alpha } _ { k ^ { \star } } / ( r e ^ { i \psi } ) } = - \displaystyle \sum _ { k \neq k ^ { \star } } L _ { k } ( r e ^ { i \psi } ) e ^ { ( \alpha _ { k } - \alpha _ { k ^ { \star } } ) r e ^ { i \psi } + \bar { \alpha } _ { k ^ { \prime } } / ( r e ^ { i \psi } ) } } \\ & { \quad \quad \quad = - \displaystyle \sum _ { k \neq k ^ { \star } } L _ { k } ( r e ^ { i \psi } ) e ^ { - r \left( M ^ { \star } - \mathrm { R e } ( \alpha _ { k } e ^ { i \psi } ) \right) } e ^ { i r \mathrm { I m } \left( ( \alpha _ { k } - \alpha _ { k ^ { \star } } ) e ^ { i \psi } \right) } e ^ { \bar { \alpha } _ { k ^ { \prime } } / ( r e ^ { i \psi } ) } , } \end{array}
$$

where we used $e ^ { w } = e ^ { \mathrm { R e } w } e ^ { i }$ <sup>Im</sup> <sup>w</sup> and Re $\left( ( \alpha _ { k } - \alpha _ { k ^ { \star } } ) e ^ { i \psi } \right) = - \left( M ^ { \star } - \mathrm { R e } ( \alpha _ { k } e ^ { i \psi } ) \right)$ . The middle exponential has modulus one, and $\left| e ^ { \bar { \alpha } _ { k } / ( r e ^ { i \psi } ) } \right| = e ^ { \mathrm { R e } ( \bar { \alpha } _ { k } e ^ { - i \psi } ) / r } \in \left[ e ^ { - | \alpha _ { k } | / r } , e ^ { | \alpha _ { k } | / r } \right]$ . Taking moduli, for $r \geq 1$

$$
e ^ { - | \alpha _ { k ^ { \star } } | } \left| { \cal L } _ { k ^ { \star } } ( r e ^ { i \psi } ) \right| \leq e ^ { - r \eta } \sum _ { k \neq k ^ { \star } } e ^ { | \alpha _ { k } | } \left| { \cal L } _ { k } ( r e ^ { i \psi } ) \right| .\tag{A.5}
$$

For each k let $q _ { k } = \operatorname* { m a x } \{ j : ~ c _ { k , j } \neq 0 \}$ , which exists because $\begin{array} { r } { L _ { k } \not \equiv 0 , } \end{array}$ and let $q = \operatorname* { m a x } _ { k } q _ { k }$ . Then $\begin{array} { r l } { \left| L _ { k } ( r e ^ { i \psi } ) \right| = } \end{array}$ $\left| c _ { k , q _ { k } } \right| r ^ { q _ { k } } \left( 1 + O ( r ^ { - 1 } ) \right)$ as $r  \infty$ . Hence the left-hand side of $\left( \mathrm { { A . 5 } } \right)$ is at least $\frac { 1 } { 2 } e ^ { - \lvert \alpha _ { k } \star \rvert } \left. c _ { k ^ { \star } , q _ { k } \star } \right. r ^ { q _ { k } \star }$ for r large, whereas the right-hand side is $O \left( r ^ { q } e ^ { - r \eta } \right)$ . Since $r ^ { q _ { k } \star - q } e ^ { r \eta }  \infty { \mathrm { ~ a s ~ } } r  \infty , ( \mathrm { A . 5 } )$ fails for r large, which is a contradiction. Therefore $L _ { k } \equiv 0$ for every $k ,$ that is, $p _ { k } \equiv 0$ for every k.

Under N only the sentences of lengths $m \in \operatorname { s u p p } ( \nu )$ occur, so that, when $\operatorname { s u p p } ( \nu )$ is finite, the support $\mathcal { S } _ { \mathbf { N } }$ is strictly smaller than ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$ , and a statistic must be shown to be non-constant on the sets ${ \mathcal { E } } _ { m }$ of m-point empirical measures. Our argument only uses two-point measures $\mu _ { t } = ( 1 - t ) \delta _ { x } + t \delta _ { y }$ , through the derivative at $t = 0$ (see the proof of Proposition A.4 below), whereas on ${ \mathcal { E } } _ { m }$ only the weights $t \in \{ 0 , \frac { 1 } { m } , \ldots , 1 \}$ occur. The following lemma bridges the gap.

Lemma A.3 (Rational dependence on the weight). Fix $x , y \in \mathbb { S } ^ { d }$ and let $\mu _ { t } = ( 1 - t ) \delta _ { x } + t \delta _ { y }$ . For $\Lambda \in \mathbb { M } _ { d + 1 }$ let $q _ { \Lambda } ( t ) = \big ( 1 + t ( e ^ { x ^ { \top } \Lambda ( y - x ) } - 1 ) \big ) \big ( 1 + t ( e ^ { y ^ { \top } \Lambda ( y - x ) } - 1 ) \big )$ , a polynomial of degree at most 2 which is positive on $[ 0 , 1 ]$ . For $\dot { \Lambda _ { 1 } } , \dotsc , \Lambda _ { H } \in \mathbb { M } _ { d + 1 }$ and $A _ { 0 } , A _ { 1 } , \dotsc , A _ { H } \in \mathbb { M } _ { d + 1 }$ , the function

$$
f ( t ) = \langle A _ { 0 } , m _ { \mu _ { t } } m _ { \mu _ { t } } ^ { \top } \rangle + \sum _ { k = 1 } ^ { H } \langle A _ { k } , S _ { \Lambda _ { k } } ( \mu _ { t } ) \rangle , \qquad t \in [ 0 , 1 ] ,
$$

is of the form $\begin{array} { r } { f = P / \prod _ { k } q _ { \Lambda _ { k } } } \end{array}$ with P a polynomial of degree at most $2 H + 2 ,$ and of degree at most $2 H + 1 \ i f$ $A _ { 0 } = 0$ . Consequently, if m $\geq 2 H + 2$ (respectively $m \ge 2 H + 1 ~ i f ~ A _ { 0 } = 0 )$ and f is constant on $\{ 0 , \frac { 1 } { m } , \ldots , 1 \}$ ， then f is constant on $[ 0 , 1 ]$

Proof. As in the proof of Lemma A.1 we have

$$
S _ { \Lambda } ( \mu _ { t } ) = ( 1 - t ) x x ^ { \top } + t y y ^ { \top } + t ( 1 - t ) \Bigl [ \frac { e ^ { x ^ { \top } \Lambda ( y - x ) } } { 1 + t ( e ^ { x ^ { \top } \Lambda ( y - x ) } - 1 ) } x ( y - x ) ^ { \top } + \frac { 1 } { 1 + t ( e ^ { y ^ { \top } \Lambda ( y - x ) } - 1 ) } y ( x - y ) ^ { \top } \Bigr ] .
$$

Multiplying by both denominators, each term $q _ { \Lambda _ { k } } \langle A _ { k } , S _ { \Lambda _ { k } } ( \mu _ { t } ) \rangle$ becomes a polynomial of degree at most $^ { 3 , }$ and $\langle A _ { 0 } , \bar { m } _ { \mu _ { t } } m _ { \mu _ { t } } ^ { \bar { \top } } \rangle$ is a polynomial of degree at most 2, since $m _ { \mu _ { t } } = x + t ( y - x )$ . Multiplying by the remaining factors q gives the degrees $q _ { \Lambda _ { j } }$ $3 + 2 ( H - 1 ) = 2 H + 1$ and $2 + 2 H$ . I $\colon f = P / R$ , with $\begin{array} { r } { R = \prod _ { k } q _ { \Lambda _ { k } } > 0 \mathrm { o n } \ [ 0 , 1 ] } \end{array}$ equals a constant c at the m + 1 distinct points $\begin{array} { r } { 0 , \frac { 1 } { m } , \dots , 1 , } \end{array}$ , then the polynomial $\displaystyle P - c R ,$ of degree at most max(deg P, deg R), has at least $m + 1$ zeros. It therefore vanishes identically when $m + 1$ exceeds this degree, which is at most $2 H + 2$ (respectively $2 H + 1 )$ , since deg $R \leq 2 H$ ; then $f \equiv c \mathrm { o n } \ [ 0 , 1 ]$

We are now ready to prove the linear independence of attention cross-moments, from which the non-degeneracy of attention readouts follows.

Proposition A.4 (Linear independence of attention cross-moments). Let $\Lambda _ { 1 } , \ldots , \Lambda _ { H } \in \mathbb { M } _ { d + 1 } \backslash \{ 0 \}$ be pairwise distinct, $A _ { 0 } , A _ { 1 } , \dotsc , A _ { H } \in \mathbb { M } _ { d + 1 }$ , and m $\geq 2 H + 2 \ ( m \geq 2 H + 1 \ i f \ A _ { 0 } = 0 )$ . For $\mu \in \mathcal P ( \mathbb { S } ^ { d } )$ let $\begin{array} { r } { m _ { \mu } = \int \boldsymbol { x } \mu ( \mathrm { d } \boldsymbol { x } ) } \end{array}$ be its mean. If the function

$$
\mu \longmapsto \langle A _ { 0 } , m _ { \mu } m _ { \mu } ^ { \top } \rangle + \sum _ { k = 1 } ^ { H } \langle A _ { k } , S _ { \Lambda _ { k } } ( \mu ) \rangle
$$

is constant on ${ \mathcal { E } } _ { m }$ , then $A _ { 1 } = \cdots = A _ { H } = 0$ and $A _ { 0 }$ is antisymmetric. In particular, the same conclusion holds if the function is constant on ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$

Proof. Fix $x , y \in \mathbb { S } ^ { d } .$ , let $\mu _ { t } = ( 1 - t ) \delta _ { x } + t \delta _ { y }$ and let $\begin{array} { r } { f ( t ) = \langle A _ { 0 } , m _ { \mu _ { t } } m _ { \mu _ { t } } ^ { \top } \rangle + \sum _ { k = 1 } ^ { H } \langle A _ { k } , S _ { \Lambda _ { k } } ( \mu _ { t } ) \rangle , t \in [ 0 , 1 ] } \end{array}$ . Since $\begin{array} { r } { \mu _ { j / m } = ( 1 - \frac { j } { m } ) \delta _ { x } + \frac { j } { m } \delta _ { y } \in \mathcal { E } _ { m } } \end{array}$ for $0 \leq j \leq m , f$ is constant on $\{ 0 , \frac { 1 } { m } , \ldots , 1 \}$ , hence on $[ 0 , 1 ]$ by Lemma $\mathrm { { A . 3 } } .$ . By Lemma $_ { \textrm { A . 1 , } f }$ is diferentiable, so that $f ^ { \prime } ( 0 ) = 0$ . Since $m _ { \mu _ { t } } = x + t ( y - x )$ and $m ^ { \top } A _ { 0 } m = m ^ { \top } ( A _ { 0 } ) _ { s } m$ , with $( A _ { 0 } ) _ { s } = \textstyle { \frac { 1 } { 2 } } ( A _ { 0 } + A _ { 0 } ^ { \top } )$ , the moment term contributes $2 x ^ { \top } ( A _ { 0 } ) _ { s } ( y - x ) \mathrm { t o } f ^ { \prime } ( 0 )$ , while by (A.2) each attention term contributes $e ^ { x ^ { \top } \Lambda _ { k } ( y - x ) } x ^ { \top } A _ { k } ( y - x ) + y ^ { \top } A _ { k } x - x ^ { \top } A _ { k } x$ . Hence, for all $x , y \in \mathbb { S } ^ { d }$

$$
0 = 2 x ^ { \top } ( A _ { 0 } ) _ { s } ( y - x ) + \sum _ { k = 1 } ^ { H } \left[ e ^ { x ^ { \top } \Lambda _ { k } ( y - x ) } x ^ { \top } A _ { k } ( y - x ) + y ^ { \top } A _ { k } x - x ^ { \top } A _ { k } x \right] .\tag{A.6}
$$

Consider the set

$$
\mathcal { X } : = \big \{ x \in \mathbb { S } ^ { d } : \Lambda _ { k } ^ { \top } x \ne 0 \forall k \le H , \mathrm { a n d } ( \Lambda _ { k } - \Lambda _ { j } ) ^ { \top } x \ne 0 \forall j \ne k \le H \big \} .
$$

It is open and dense in $\mathbb { S } ^ { d }$ , since it is the complement of finitely many proper linear subspaces. Fix $x \in \mathcal X$ a unit vector $v \perp x ,$ , and let $y =$ cos $\theta x +$ sin $\theta v \in \mathbb { S } ^ { d } , \theta \in \mathbb { R }$ . With $( a _ { k } , b _ { k } ) : = ( x ^ { \top } \Lambda _ { k } x , x ^ { \top } \Lambda _ { k } v )$ we have $x ^ { \top } \Lambda _ { k } ( y - x ) = ( \cos \theta - 1 ) a _ { k } + \sin \theta b _ { k }$ , and (A.6) becomes

$$
\sum _ { k = 1 } ^ { H } e ^ { - a _ { k } } e ^ { a _ { k } \cos \theta + b _ { k } \sin \theta } \Big [ \big ( \cos \theta - 1 \big ) x ^ { \top } A _ { k } x + \sin \theta x ^ { \top } A _ { k } v \Big ] + p _ { 0 } ( \theta ) = 0 , \qquad \theta \in \mathbb { R } ,
$$

where $p _ { 0 }$ is a trigonometric polynomial, which collects the terms without exponential factors.

We now check that Lemma A.2 applies. Its exponent pairs are $( a _ { k } , b _ { k } ) , k \leq H$ , and (0, 0) for $p _ { 0 }$ , and they must be pairwise distinct: we need $( a _ { k } , b _ { k } ) \neq ( 0 , 0 )$ and $( a _ { k } , b _ { k } ) \neq ( a _ { j } , b _ { j } )$ for $j \neq k$ . Both conditions have the form $( w ^ { \top } x , w ^ { \top } v ) \neq ( 0 , 0 )$ , with $w = \Lambda _ { k } ^ { \top }$ x and $w = \left( \Lambda _ { k } - \Lambda _ { j } \right) ^ { - }$ <sup>⊤</sup>x respectively, and w = 0 because $x \in \mathcal { X }$ . For such w, the pair $( w ^ { \top } x , w ^ { \top } v )$ vanishes only if w $\perp x$ and w v. In that case $w \in x ^ { \bot } \setminus \{ 0 \}$ , and the unit vectors $v \in x ^ { \perp }$ with $v \perp$ w form a great subsphere of the unit sphere of $x ^ { \perp }$ , which is closed with empty interior (and is empty when $d = 1 )$ . Since there are finitely many such $w ,$ the exponent pairs are pairwise distinct for every v in an open dense subset of the unit sphere of $x ^ { \perp }$

For such $v ,$ Lemma $\mathrm { A . 2 }$ shows that all the coeficients vanish identically, in particular (cos $\theta - 1 ) x ^ { \top } A _ { k } x +$ sin $\theta x ^ { \top } A _ { k } v = 0$ for all θ. Since cos θ 1 and sin θ are linearly independent, $x ^ { \top } A _ { k } x = x ^ { \top } A _ { k } v = 0$ . By continuity in $v , x ^ { \top } A _ { k } v = 0$ for every unit $v \perp x ,$ , so that the vector $A _ { k } ^ { \top } x$ is orthogonal both to x and to $x ^ { \perp }$ , that is, $A _ { k } ^ { \top } x = 0$ . This holds for every $x \in \mathcal { X }$ , which is dense in $\mathbb { S } ^ { d }$ , hence for every $x \in \mathbb { S } ^ { d }$ by continuity, and $A _ { k } = 0$ for every k.

Then (A.6) reduces to $x ^ { \top } ( A _ { 0 } ) _ { s } ( y - x ) = 0$ for all $x , y \in \mathbb { S } ^ { d }$ . With $y = - x$ it gives $x ^ { \top } ( A _ { 0 } ) _ { s } x = 0$ , hence $x ^ { \top } ( A _ { 0 } ) _ { s } y = 0$ for all $x , y \in \mathbb { S } ^ { d }$ and, by homogeneity, for all $x , y \in \mathbb { R } ^ { d + 1 }$ , so that $( A _ { 0 } ) _ { s } = 0$ , which is equivalent to $A _ { 0 }$ being antisymmetric. The last claim follows since $\mathcal { E } _ { m } \subset \mathcal { P } ( \mathbb { S } ^ { d } )$ ■

Non-degeneracy holds for every non-zero choice matrix Λ, with no symmetry assumption: the proof only uses that $\{ x : \ \Lambda _ { k } ^ { \top } x \neq 0 \}$ is dense, which holds for every $\Lambda _ { k } \neq 0$ , including antisymmetric ones. In particular $S _ { \Lambda }$ and $S _ { \Lambda ^ { \top } }$ are linearly independent when $\Lambda \neq \Lambda ^ { \top }$ , in contrast with the kernel sums of Proposition $\mathrm { 3 . 1 \mathrm { - } ( d ) }$ . This explains why our model admits a query–key matrix $\Lambda = Q ^ { \top } K$ that is not necessarily symmetric. The mechanism is the softmax normalization: it produces the exponential factors in (A.6), which cannot be compensated by trigonometric polynomials (Lemma A.2).

Corollary A.5 (Non-degeneracy).

(a) Let $\Lambda \neq 0$ $I f m \ge 3$ and $\langle A , S _ { \Lambda } \rangle$ is constant on $\mathcal { E } _ { m }$ , then $A = 0$ . Consequently, $i f \nu ( m ) > 0$ for some m $\geq 3$ , then $S _ { \Lambda }$ is ν-non-degenerate: $\langle A , S _ { \Lambda } \rangle$ is constant on $\mathcal { S } _ { \mathbf { N } }$ only for $A = 0$ . The same holds on ${ \mathcal { P } } ( \mathbb { S } ^ { d } )$

(b) Let $\Lambda = 0 . ~ I f m \geq 2$ , then $\langle A , S _ { 0 } \rangle$ is constant on ${ \mathcal { E } } _ { m }$ if and only if A is antisymmetric. Consequently, if $\nu ( m ) > 0 ~ f o r$ some m $\geq 2$ , then $\langle A , S _ { 0 } \rangle$ is constant on if and only if A is antisymmetric.

Proof. The first claim in (a) is Proposition A.4 with $H = 1$ and $A _ { 0 } = 0$ , since m $\geq 3 = 2 H + 1$ . If $\langle A , S _ { \Lambda } \rangle$ is constant on $\mathcal { S } _ { \mathbf { N } }$ , it is constant on ${ \mathcal { E } } _ { m }$ for every m supp(ν), since $\mathcal { E } _ { m } \subset S _ { \bf N }$ , and the first claim applies; the last claim follows from $\mathcal { E } _ { m } \subset \mathcal { P } ( \mathbb { S } ^ { d } )$ .

Claim (b) can be proven as follows. For $\Lambda = 0$ the choice rule is $\mu \otimes \mu ,$ , so that $\mathrm { A t t n } _ { 0 } ( x ; \mu ) = m _ { \mu }$ and $S _ { 0 } ( \mu ) = m _ { \mu } m _ { \mu } ^ { \top }$ . Hence $\langle A , S _ { 0 } ( \mu ) \rangle = m _ { \mu } ^ { \top } A m _ { \mu } = m _ { \mu } ^ { \top } A _ { s } m _ { \mu }$ , with $\begin{array} { r } { A _ { s } = \frac { 1 } { 2 } ( A + A ^ { \top } ) } \end{array}$ , which vanishes identically if A is antisymmetric. Conversely, assume that $\langle A , S _ { 0 } \rangle = c$ on <sub>m</sub>. At $\mu = \delta _ { x } \in \mathcal { E } _ { m }$ this gives $x ^ { \top } A _ { s } x = c$ for every $x \in \mathbb { S } ^ { d } ;$ , so that $A _ { s } = $ cI. At $\begin{array} { r } { \mu = ( 1 - \frac { 1 } { m } ) \delta _ { x } + \frac { 1 } { m } \delta _ { - x } \in \mathcal { E } _ { m } . } \end{array}$ whose mean $\textstyle { \left( { 1 - { \frac { 2 } { m } } } \right) }$ x has norm $\textstyle \left| 1 - { \frac { 2 } { m } } \right| < 1$ , it gives $\begin{array} { r } { c ( 1 - \frac { 2 } { m } ) ^ { 2 } = c \mathrm { . } } \end{array}$ so that $c = 0$ and $A _ { s } = 0$ . The last claim follows as in (a). ■

## Appendix B. Finite vocabularies and the unembedding

In the body of the paper tokens are points of the sphere drawn from the uniform law vol, and the unembedding of Section 1.6 is obtained by projecting the output of the Langevin scheme onto the vocabulary through a non-linearity σ. In this appendix we show that, if the null model draws its tokens directly from a finite vocabulary, the causal model of Section 4 gives the standard softmax unembedding exactly, with weight tying and without any sampler or non-linearity.

Let $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { | \mathcal { V } | } \} \subset \mathbb { S } ^ { d }$ be a finite vocabulary of embedded tokens and let $\omega$ be a probability on with $\omega ( v ) > 0$ for every $v \in \mathcal { V } \ ( \mathrm { e . g . }$ , the uniform law). Then, in this setting, the discrete null model $\vec { \bf N } _ { \nu }$ is the law of the ordered sentence $\iota ( N , X _ { 1 : N } )$ when $N \sim \nu$ and, independently, $X _ { 1 } , X _ { 2 } , \ldots$ are i.i.d. with law $\omega .$ It is concentrated on the countable set $\begin{array} { r } { \iota ( \lfloor \lfloor \geq 1 \rceil ) ^ { n } ) } \end{array}$ , and

$$
\vec { \bf N } _ { \mathcal { V } } \big ( N = n , \ X _ { 1 : n } = x _ { 1 : n } \big ) = \nu ( n ) \prod _ { i = 1 } ^ { n } \omega ( x _ { i } ) , \qquad x _ { 1 : n } \in \mathcal { V } ^ { n } .
$$

The masked choice rule, the shifted cross-moment $S _ { \Lambda } ^ {  }$ and the readout $\Psi _ { \Lambda , V } ^ {  } = \langle V , S _ { \Lambda } ^ {  } \rangle$ are defined as in Section 4, and for Λ, $V \in \mathbb { M } _ { d + 1 }$ we set $\mathrm { P } _ { \Lambda , V ; \nu } ^ {  } \propto e ^ { \Psi _ { \Lambda , V } ^ {  } } \vec { \bf N } _ { \nu }$ . Finally, recall from Section 1.6 the definition of the unembedding matrix $W _ { \mathrm { o u t } } \in \mathbb { R } ^ { | \mathcal { V } | \times ( d + 1 ) }$ , whose rows are $v _ { 1 } ^ { \top } , \ldots , v _ { | \mathcal { V } | } ^ { \top } ,$ so that $( W _ { \mathrm { o u t } } \widehat { Y } ) _ { v } = \langle v , \widehat { Y } \rangle$ for $\widehat { Y } \in \mathbb { R } ^ { d + 1 }$

Proposition B.1 (Finite vocabulary: exact softmax unembedding). Let Λ, $V \in \mathbb { M } _ { d + 1 }$

(a) For every $n \geq 2$ with $\nu ( n ) > 0$ and every context $x _ { 1 : n - 1 } \in \mathcal { V } ^ { n - 1 }$ , for any $v \in \mathcal V$

$$
\begin{array} { r } { \mathrm { P } _ { \Lambda , V ; \mathcal { V } } ^ {  } ( X _ { n } = v \middle \vert N = n , \ X _ { 1 : n - 1 } = x _ { 1 : n - 1 } ) = \mathrm { s o f t m a x } \Big ( \frac { 1 } { n - 1 } W _ { \mathrm { o u t } } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \stackrel {  } { \mu } ) + \log \omega \Big ) _ { * } \ ! } \end{array}
$$

where log $\boldsymbol { \omega } = ( \log \omega ( v _ { 1 } ) , \dots , \log \omega ( v _ { | \mathcal { V } | } ) )$ is applied componentwise.

(b) Assume that $\nu ( n ) > 0$ for some $n \geq 2$ and that is not contained in an afine hyperplane $o f \mathbb { R } ^ { d + 1 }$ . If $\langle A , S _ { \Lambda } ^ {  } \rangle$ is $\vec { \bf N } _ { \nu ^ { - } a . s }$ . constant for some $A \in \mathbb { M } _ { d + 1 }$ , then $A = 0$

(c) Under the assumptions of (b), let ${ \mathcal { K } } _ { \Lambda ; \nu } ^ {  }$ be the closed convex hull of the support of $( S _ { \Lambda } ^ {  } ) _ { \# } \vec { \bf N } _ { \nu }$ . For every $c \in \mathrm { i n t } { \mathcal { K } _ { \Lambda ; \nu } ^ {  } }$ there is a unique $V ^ { \star } \in \mathbb { M } _ { d + } ^ { }$ with $\mathbb { E } _ { \mathrm { P } _ { \Lambda , V ^ { \star } ; \mathcal { V } } } S _ { \Lambda } ^ {  } = c ,$ and $\mathrm { P } _ { \Lambda , V ^ { \star } ; \mathcal { V } } ^ {  }$ is the unique minimizer of $\operatorname { K L } ( \mathbf { P } | \vec { \mathbf { N } } _ { \nu } )$ over the laws P with $\mathbb { E } _ { \mathrm { P } } S _ { \Lambda } ^ {  } = c$

Proof. Throughout, $n \geq 2$ and we use the explicit form of the readout from Definition 4.4,

$$
\Psi _ { \Lambda , V } ^ {  } ( x _ { 1 : n } ) = \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 2 } x _ { i + 1 } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { i } , i ) ; \vec { \mu } ) \ + \ \frac { 1 } { n - 1 } x _ { n } ^ { \top } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) .\tag{B.1}
$$

By (4.2), Att $\mathrm { n } _ { \Lambda } \big ( ( x _ { i } , i ) ; \vec { \mu } \big )$ depends only on $x _ { 1 : i }$ . Hence the first sum depends only on the context $x _ { 1 : n - 1 }$ , and the last token $x _ { n }$ enters only through the last term, linearly.

Since, by definition of $\mathrm { P } _ { \Lambda , V ; \mathcal { V } } ^ {  }$ and of $\vec { \bf N } _ { \nu }$ , for $x _ { 1 : n } \in \mathcal { V } ^ { n }$ we have

$$
\mathrm { P } _ { \Lambda , V ; \mathcal { V } } ^ {  } \big ( N = n , ~ X _ { 1 : n } = x _ { 1 : n } \big ) \ \propto \ \nu ( n ) \prod _ { i = 1 } ^ { n } \omega ( x _ { i } ) \ e ^ { \Psi _ { \Lambda , V } ^ {  } ( x _ { 1 : n } ) } ,
$$

for any fixed context $x _ { 1 : n - 1 }$ and any $v \in \mathcal V$ it holds

$$
\begin{array} { r l } { \mathrm { P } _ { \Lambda , V ; V } ^ {  } \big ( X _ { n } = v \big | N = n , ~ X _ { 1 : n - 1 } = x _ { 1 : n - 1 } \big ) } & { \propto \omega ( v ) \exp \Big ( \frac { 1 } { n - 1 } v ^ { \top } V \mathrm { A t t } \mathrm { n } _ { \Lambda } \big ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } \big ) \Big ) } \\ & { \propto \exp \Big ( \frac { 1 } { n - 1 } \big ( W _ { \mathrm { o u t } } V \mathrm { A t t } \mathrm { n } _ { \Lambda } \big ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } \big ) \big ) _ { v } + \log \omega ( v ) \Big ) . } \end{array}
$$

Normalizing over $v \in \mathcal V$ gives (a).

Item (b) can be proven as follows. Fix $n \geq 2$ with $\nu ( n ) > 0$ , and assume that $\langle A , S _ { \Lambda } ^ {  } \rangle$ is constant $\vec { \bf N } _ { \nu ^ { - } \mathrm { a . s . } }$ and therefore it is constant for every $x _ { 1 : n } \in \mathcal { V } ^ { n } \ ( \overline { { \mathbf { N } } } _ { \mathcal { V } }$ gives positive mass $\nu ( n ) \prod _ { i } \omega ( x _ { i } )$ to every $x _ { 1 : n } \in \mathcal { V } ^ { n } )$

Now, if we apply (B.1) with $V = A$ and $x _ { 1 : n - 1 } \in \mathcal { V } ^ { n - 1 }$ fixed, since $\langle A , S _ { \Lambda } ^ {  } \rangle$ is constant, for any $v , v ^ { \prime } \in \mathcal { V }$ we immediately get $( v - v ^ { \prime } ) ^ { \top } A \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) = 0$ . Since is not contained in an afine hyperplane, the diferences $\{ v - v _ { 1 } : v \in \mathcal { V } \}$ span $\mathbb { R } ^ { d + 1 }$ , which means that A Attn $\backslash ( ( x _ { n - 1 } , n - 1 ) ; \vec { \mu } ) = 0 .$

When considering the constant context $x _ { 1 } = \cdot \cdot \cdot = x _ { n - 1 } = v \in \mathcal { V } .$ , where every key of the query $( v , n - 1 )$ equals v so that Attn $\operatorname { \ a } ( ( x _ { n - 1 } , n - 1 ) ; { \vec { \mu } } ) = v$ , we conclude that $A v = 0$ . This is enough to conclude that $A = 0 { : }$ indeed is not contained in any afine hyperplane and therefore it spans the whole $\bar { \mathbb R } ^ { d + 1 }$ , which means that A vanishes on $\mathbb { R } ^ { d + 1 }$ . This concludes the proof of (b).

The proof of Claim (c) is similar to that of Theorem 2.1 and relies on [Bro86, Theorem 3.6]. Since $\| S _ { \Lambda } ^ {  } \| _ { \mathrm { F } } \leq 1$ the log-partition function $\Gamma _ { \mathcal { V } } ( V ) = \log \mathbb { E } _ { \vec { \mathbf { N } } _ { \mathcal { V } } } e ^ { \langle V , S _ { \Lambda } ^ {  } \rangle }$ is finite on the whole of $\mathbb { M } _ { d + 1 }$ , which is open, so the exponential family $\{ \mathrm { P } _ { \Lambda , V ; \mathcal { V } } ^ {  } \} _ { V \in \mathbb { M } _ { d + 1 } }$ is regular, hence steep. By (b) it is minimal. Then [Bro86, Theorem 3.6] shows that $V \mapsto \nabla \Gamma _ { \mathcal { V } } ( V ) = \mathbb { E } _ { \mathrm { P } _ { \Lambda , V ; \mathcal { V } } ^ {  } } S _ { \Lambda } ^ {  }$ is a bijection of $\mathbb { M } _ { d + 1 }$ onto $\mathrm { i n t } { \cal K } _ { \Lambda ; \nu } ^ {  } ,$ which gives existence and uniqueness of $V ^ { \star }$ . Set $\mathrm { P } ^ { \star } = \mathrm { P } _ { \Lambda , V ^ { \star } ; \mathcal { V } } ^ {  }$ . Its density $e ^ { \langle V ^ { \star } , S _ { \Lambda } ^ {  } \rangle - \Gamma \nu ( V ^ { \star } ) }$ with respect to $\vec { \bf N } _ { \nu }$ is bounded away from 0 and $\infty ,$ so $\mathrm { P } ^ { \star }$ and $\vec { \bf N } _ { \nu }$ are equivalent. Let P satisfy the constraint with $\mathrm { K L } ( \mathbf { P } | \vec { \mathbf { N } } _ { \nu } ) < \infty$ . Then $\mathrm { P } \ll \mathrm { P } ^ { \star }$ , and integrating log $\begin{array} { r l } & { \frac { \mathrm { d } \mathrm { P } } { \mathrm { d } \vec { \mathbf { N } } \nu } = \log \frac { \mathrm { d } \mathrm { P } } { \mathrm { d } \mathrm { P } ^ { \star } } + \langle V ^ { \star } , S _ { \Lambda } ^ {  } \rangle - \Gamma _ { \mathcal { V } } ( V ^ { \star } ) } \end{array}$ against P gives the Pythagorean identity

$$
\operatorname { K L } ( \mathrm { P } \mid \vec { \mathbf { N } } \nu ) = \operatorname { K L } ( \mathrm { P } \mid \mathrm { P } ^ { \star } ) + \langle V ^ { \star } , c \rangle - \Gamma \nu ( V ^ { \star } ) = \operatorname { K L } ( \mathrm { P } \mid \mathrm { P } ^ { \star } ) + \operatorname { K L } ( \mathrm { P } ^ { \star } \mid \vec { \mathbf { N } } \nu ) .
$$

From this last identity we conclude that $\mathrm { K L } ( \mathrm { P } | \vec { \mathbf { N } } _ { \mathcal { V } } ) \geq \mathrm { K L } ( \mathrm { P } ^ { \star } | \vec { \mathbf { N } } _ { \mathcal { V } } )$ , with equality if and only if $\mathrm { P } = \mathrm { P } ^ { \star }$ Remark B.2 (On the assumptions and on weight tying).

(i) The geometric assumption of (b) requires $| \nu | \geq d + 2 ;$ , and it holds as soon as the embedded vocabulary is not contained in an afine hyperplane, which is the generic situation since in practice $| \nu | \gg d + 1$

(ii) In (a) the unembedding matrix is the matrix of the vocabulary itself. If the vocabulary is obtained from one-hot vectors through an embedding matrix $\dot { E } \in \mathbb { R } ^ { ( d + 1 ) \times | \mathcal { V } | }$ followed by normalization, $v _ { i } = E e _ { i } / \left| E e _ { i } \right|$ then $W _ { \mathrm { o u t } } = \mathrm { d i a g } \big ( | E e _ { 1 } | , \dots , | E e _ { | \mathcal { V } | } | \big ) ^ { - 1 } E ^ { \top }$ , i.e. the unembedding is tied to the embedding. The reference law ω only adds the output bias log ω, which vanishes, up to an additive constant, for the uniform law.

(iii) The same proof applies verbatim to the extensive statistic $T _ { \Lambda } ^ {  } = ( n - 1 ) S _ { \Lambda } ^ {  }$ of Proposition $4 . 9 ,$ under the exponential-moment assumption on ν made there. The logits then become $W _ { \mathrm { o u t } } V \mathrm { A t t n } _ { \Lambda } ( ( x _ { n - 1 } , n -$ 1); ${ \vec { \mu } } ) + \log \omega$ , without the $f a c t o r \ \frac { 1 } { n - 1 }$

## References

[ABMG<sup>+</sup>26] Andrea Agazzi, Giuseppe Bruno, Eloy Mosig García, Samuele Saviozzi, and Marco Romito. Stochastic Scaling Limits and Synchronization by Noise in Deep Transformer Models. arXiv preprint arXiv:2604.26898, 2026.

[BB26] Etienne Boursier and Claire Boyer. Softmax as Linear Attention in the Large-Prompt Regime: a Measure-based Perspective. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 9392–9429, 2026.

[BDPDP96] Adam L. Berger, Stephen A. Della Pietra, and Vincent J. Della Pietra. A Maximum Entropy Approach to Natura Language Processing. Computational Linguistics, 22(1):39–71, 1996.

[BGL14] Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and Geometry of Markov Difusion Operators, volume 348 of Grundlehren der mathematischen Wissenschaften. Springer, Cham, 2014.

[BLPV26] Léa Bohbot, Cyril Letrouit, Gabriel Peyré, and François-Xavier Vialard. Token Sample Complexity of Attention. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 8899–8947, 2026.

[BN78] Ole E. Barndorf-Nielsen. Information and Exponential Families in Statistical Theory. John Wiley & Sons, New York, 1978.

[Bro86] Lawrence D. Brown. Fundamentals of Statistical Exponential Families, with Applications in Statistical Decision Theory, volume 9 of Institute of Mathematical Statistics Lecture Notes—Monograph Series. Institute of Mathematica Statistics, Hayward, CA, 1986.

[BTEZ25] Fabrizio Boncoraglio, Emanuele Troiani, Vittorio Erba, and Lenka Zdeborová. Bayes Optimal Learning of Attention-Indexed Models. In Advances in Neural Information Processing Systems, volume 38, pages 105029–105074, 2025.

[CBKZ25] Hugo Cui, Freya Behrens, Florent Krzakala, and Lenka Zdeborová. A Phase Transition between Positional and Semantic Learning in a Solvable Model of Dot-Product Attention. Journal of Statistical Mechanics: Theory and Experiment, 2025:074001, 2025.

[CFT<sup>+</sup>21] Hyung Won Chung, Thibault Févry, Henry Tsai, Melvin Johnson, and Sebastian Ruder. Rethinking Embedding Coupling in Pre-trained Language Models. In International Conference on Learning Representations, 2021.

[CLPR25] Shi Chen, Zhengjiang Lin, Yury Polyanskiy, and Philippe Rigollet. Quantitative Clustering in Mean-Field Transformer Models, 2025. arXiv:2504.14697.

[CM05] Imre Csiszár and František Matúš. Closures of exponential families. The Annals of Probability, 33(2):582–600, 2005.

[Csi75] Imre Csiszár. I-Divergence Geometry of Probability Distributions and Minimization Problems. The Annals of Probability, 3(1):146–158, 1975.

[DCLT19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4171–4186, 2019.

[DE97] Paul Dupuis and Richard S. Ellis. A Weak Convergence Approach to the Theory of Large Deviations. Wiley series in Probability and Statistics. John Wiley & sons, 1997.

[DGGB23] Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing Transformers in Embedding Space. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023.

[Fra26] Demián Fraiman. On the Expressive Power of Transformers for Contextual Relations, 2026. arXiv:2603.25860.

[FS11] Hans Föllmer and Alexander Schied. Stochastic Finance: An Introduction in Discrete Time. W. de Gruyter, Berlin, 3rd edition, 2011.

[FSE<sup>+</sup>26] Lev Fedorov, Michaël E. Sander, Romuald Elie, Pierre Marion, and Mathieu Laurière. Clustering in Deep Stochastic Transformers. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 29779–29803. PMLR, 2026.

[GBB26] Simon Gabet, Etienne Boursier, and Claire Boyer. Softmax Attention on Gaussian Mixtures: Linear When It Can, Selective When It Must. arXiv preprint arXiv:2610.11798, 2026.

[GKPR24] Borjan Geshkovski, Hugo Koubbi, Yury Polyanskiy, and Philippe Rigollet. Dynamic Metastability in the Self-Attention Model, 2024. arXiv:2410.06833.

[GLPR23] Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. The emergence of clusters in self-attention dynamics. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 57026–57037. Curran Associates, Inc., 2023.

[GLPR25] Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. A mathematical perspective on Transformers. Bulletin of the American Mathematical Society, 62(3):427–479, 2025.

[GRRB26] Borjan Geshkovski, Philippe Rigollet, and Domènec Ruiz-Balet. Measure-to-Measure Interpolation Using Transformers. Foundations of Computational Mathematics, 2026.

[IKS17] Hakan Inan, Khashayar Khosravi, and Richard Socher. Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling. In International Conference on Learning Representations, 2017.

[Kle14] Achim Klenke. Probability Theory: A Comprehensive Course. Springer, 2014.

[KPR24] Nikita Karagodin, Yury Polyanskiy, and Philippe Rigollet. Clustering in Causal Attention Masking. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 115652–115681. Curran Associates, Inc., 2024.

[Léo14] Christian Léonard. A survey of the Schrödinger problem and some of its connections with optimal transport. Discrete and Continuous Dynamical Systems, 34(4):1533–1574, 2014.

[LHAG26] Antonio Lopardo, Avyukth Harish, Catherine Arnett, and Akshat Gupta. Weight Tying Biases Token Embeddings Towards the Output Space. In Findings of the Association for Computational Linguistics: ACL 2026, pages 40795– 40810. Association for Computational Linguistics, 2026.

[Lit25] Elon Litman. Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport, 2025. arXiv:2508.08369.

[LLH<sup>+</sup>19] Yiping Lu\*, Zhuohan Li\*, Di He, Zhiqing Sun, Bin Dong, Tao Qin, Liwei Wang, and Tie yan Liu. Understanding and improving transformer from a multi-particle dynamic system point of view. In ICLR 2020 Workshop on Integration of Deep Neural Models and Diferential Equations, 2019.

[Luc59] R. Duncan Luce. Individual Choice Behavior: A Theoretical Analysis. Wiley, New York, 1959.

[Luc77] R.Duncan Luce. The choice axiom after twenty years. Journal of Mathematical Psychology, 15(3):215–233, 1977.

[MA16] Andre Martins and Ramon Astudillo. From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1614–1623, New York, New York, USA, 20–22 Jun 2016. PMLR.

[MBBB25] Pierre Marion, Raphaël Berthier, Gérard Biau, and Claire Boyer. Attention layers provably solve single-location regression. In The Thirteenth International Conference on Learning Representations, 2025.

[McF74] Daniel McFadden. Conditional Logit Analysis of Qualitative Choice Behavior. In Paul Zarembka, editor, Frontiers in Econometrics, pages 105–142. Academic Press, New York, 1974.

[MM24] Anemily Machina and Robert Mercer. Anisotropy is Not Inherent to Transformers. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4892–4907. Association for Computational Linguistics, 2024.

[MSB26] Rodrigo Maulen-Soto and Claire Boyer. Attention-based PCA. In Advances in Neural Information Processing Systems (NeurIPS 2026, spotlight), 2026+. arXiv:2605.18315.

[MSMB25] Rodrigo Maulen-Soto, Pierre Marion, and Claire Boyer. Attention-based clustering. In Advances in Neural Information Processing Systems, volume 38, pages 66455–66506, 2025.

[Nut21] Marcel Nutz. Introduction to Entropic Optimal Transport. Lecture notes, Columbia University, 2021.

[PH22] Mary Phuong and Marcus Hutter. Formal Algorithms for Transformers. arXiv preprint arXiv:2207.09238, 2022.

[PW17] Ofir Press and Lior Wolf. Using the output embedding to improve language models. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, Volume 2: Short Papers, pages 157–163, 2017.

[RCZ01] Ronald Rosenfeld, Stanley F. Chen, and Xiaojin Zhu. Whole-Sentence Exponential Language Models: a Vehicle for Linguistic-Statistical Integration. Computer Speech & Language, 15(1):55–73, 2001.

[Rig26] Philippe Rigollet. The Mean-Field Dynamics of Transformers, pages 389–404. 2026.

Università degli studi di Roma Tor Vergata

Current address: RoMaDS - Department of Mathematics, 00133RM Rome, Italy.

[Ros96] Ronald Rosenfeld. A maximum entropy approach to adaptive statistical language modelling. Computer Speech & Language, 10(3):187–228, 1996.

[RSL<sup>+</sup>21] Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, et al. Hopfield Networks is All You Need. In International Conference on Learning Representations (ICLR), 2021.

[Rud91] Walter Rudin. Functional Analysis. International Series in Pure and Applied Mathematics. McGraw-Hill, New York, 2 edition, 1991.

[SABP22] Michael E. Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyré. Sinkformers: Transformers with Doubly Stochastic Attention . In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 3515–3530. PMLR, 28–30 Mar 2022.

[Sch31] Erwin Schrödinger. Über die Umkehrung der Naturgesetze. Sitzungsberichte der Preussischen Akademie der Wissenschaften, Physikalisch-Mathematische Klasse, pages 144–153, 1931.

[Sch32] Erwin Schrödinger. Sur la théorie relativiste de l’électron et l’interprétation de la mécanique quantique. Annales de l’Institut Henri Poincaré, 2(4):269–310, 1932.

[SS26] Anna Shalova and André Schlichting. Solutions of stationary McKean–Vlasov equation on a high–dimensional sphere and other Riemannian manifolds. Advances in Nonlinear Analysis, 15(1):20250141, 2026.

[TCD<sup>+</sup>25] Emanuele Troiani, Hugo Cui, Yatin Dandi, Florent Krzakala, and Lenka Zdeborová. Fundamental Limits of Learning in Sequence Multi-Index Models and Deep Attention Networks: High-Dimensional Asymptotics and Sharp Thresholds. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 60147–60182. PMLR, 2025.

[TMIS24] Lorenzo Tiberi, Francesca Mignacco, Kazuki Irie, and Haim Sompolinsky. Dissecting the Interplay of Attention Paths in a Statistical Mechanics Theory of Transformers. In Advances in Neural Information Processing Systems, volume 37, pages 72710–72753, 2024.

[Var58] V. S. Varadarajan. On the convergence of sample probability distributions. Sankhy¯a: The Indian Journal of Statistics (1933-1960), 19(1/2):23–26, 1958.

[VSP<sup>+</sup>17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[Yel77] John I. Yellott, Jr. The Relationship between Luce’s Choice Axiom, Thurstone’s Theory of Comparative Judgment, and the Double Exponential Distribution. Journal of Mathematical Psychology, 15(2):109–144, 1977.

[ZWM97] Song Chun Zhu, Ying Nian Wu, and David Mumford. Minimax Entropy Principle and Its Application to Texture Modeling. Neural Computation, 9(8):1627–1660, 1997.

Email address: greco@mat.uniroma2.it