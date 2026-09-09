# A Generalization of Amari’s Bayesian Duality

Mohammad Emtiyaz Khan<sup>1,2,3</sup> and Thomas M¨ollenhof<sup>1</sup>

<sup>1</sup>RIKEN Center for Advanced Intelligence Project, 1-4-1 Nihonbashi, Chuo-ku, Tokyo, 103-0027, Tokyo, Japan.

<sup>2</sup>Department of Computer Science, Technische Universit¨at Darmstadt, Hochschulstraße 10, Darmstadt, 64289, Hessen, Germany.   
<sup>3</sup>The Hessian Center for Artificial Intelligence, Landwehrstraße 50a, Darmstadt, 64293, Hessen, Germany.

Contributing authors: emtiyaz.khan@riken.jp; thomas.moellenhof@riken.jp;

## Abstract

Amari’s contributions to information geometry and machine learning are well known. Here, we revisit Amari’s work on Bayesian duality which has not received as much attention. We connect Amari’s Bayesian duality to a convex duality of Bayes’ rule. Using this connection, we present a generalization of Amari’s Bayesian duality and discuss its relevance for modern artificial intelligence.

Keywords: Bayes’ Rule, Information Geometry, Convex Duality

## 1 Introduction

Amari has made many significant contributions to information geometry and machine learning. He was one of the first to use stochastic gradient descent to train neural networks (Amari, 1967). He proposed recurrent neural networks that were earlier versions of what is now called Hopfield networks (Amari, 1979). He also established a connection between em and EM algorithms (Amari, 1995a,b) and put forward a proposal to use natural-gradient methods to train neural networks (Amari, 1998). All these works (and many others) are now foundational concepts in machine learning and have led to new advances in artificial intelligence.

Here, we revisit a relatively less known work by Amari (1996) on the development of a Bayesian duality theory. This work aimed to explain the ‘basic but primitive’ mechanisms of information processing in the brain and was motivated by the recent works of that time on models such as mixture-of-expert nets, the Helmholtz machine, and the Ying-Yang machine. The work did not receive much attention back then and the focus of the machine-learning community also partly drifted away from such ideas. Our goal here is to connect Amari’s Bayesian duality to other works in the machine-learning literature, so as to generalize its scope and discuss its relevance.

![](images/13cedbb405c64300f35de2a8522840bd5b24c87d171b9e7eea3feb60dee076fa.jpg)  
Fig. 1: Amari (1996) proposed a dual structure connecting the manifolds of posterior and likelihood (Panel a). In his case, a likelihood function $p ( \mathbf { y } \vert \pmb { \theta } )$ over data vector y given parameter vector θ is connected to a unique posterior $p ( \pmb \theta | \mathbf y )$ and vice versa. His framework relied on a simple posterior form. We present a generalization by using convex duality which not only recovers Amari’s case but applies much more generally.

We will start with a description of the basic setup of Amari (1996) where he introduced a new duality structure for the Bayesian framework from the point of view of information geometry. He focused on a specific case where the likelihood and posterior have the same exponential-family (EF) forms and showed that there is a bijection between the two quantities where the roles of their dual coordinates are interchanged (Fig. 1a). The theory however does not apply to the majority of Bayesian cases. For example, in conjugate Bayesian models, the form of the posterior matches the prior, not the likelihood. We will extend the scope of Amari’s theory by proposing a more general Bayesian duality through a diferent mathematical framework that relies on the convex duality of a variational formulation of Bayes’ rule (Fig. 1b). This will also enable us to connect to a much broader literature in Bayesian inference and convex optimization. We will conclude the paper by discussing the relevance of Bayesian duality for modern AI.

## 2 Amari’s Bayesian Duality Theory

We start with a brief description of Amari’s Bayesian duality theory. With a focus on the mechanisms of information processing in the brain, Amari attempted to explain the dynamic interactions between lower and higher systems. We will first discuss his framework and return to the original motivation again later in the last section.

Amari introduced a new duality structure for a simple case of Bayesian inference by using ideas from information geometry. In his example, we have an observation vector y consisting of scalar entries y which are connected to the parameter vector θ of scalar entries $\theta _ { i }$ . The two vectors are assumed to be of the same size (although an extension to diferent lengths was also briefly mentioned). Amari described a duality theory associated with the likelihood function and posterior density, denoted as,

$$
p ( \mathbf { y } \mid { \pmb \theta } ) \qquad \mathrm { a n d } \qquad p ( { \pmb \theta } \mid \mathbf { y } ) ,\tag{1}
$$

respectively. The notation $p$ is used to denote both densities over y and $\theta ,$ which is an abuse of notation for simplicity but is standard practice in many Bayesian texts.

## 2.1 The Likelihood Function

Amari considered a simple case where both the likelihood and posterior take a canonical exponential-family (EF) form. Let us define the likelihood to be

$$
p ( \mathbf { y } \mid \pmb { \theta } ) = h _ { \mathrm { l i k } } ( \mathbf { y } ) \exp \left[ \pmb { \theta } ^ { \top } \mathbf { y } - A _ { \mathrm { l i k } } ( \pmb { \theta } ) \right] .\tag{2}
$$

This is a familiar form: $A _ { \mathrm { l i k } } ( \pmb \theta )$ is the log-partition function and $h _ { \mathrm { l i k } } ( \mathbf { y } )$ is a function of $\mathbf { y } _ { \mathrm { : } }$ sometimes referred to as the base measure (Wainwright and Jordan, 2008).

The set of such densities is a manifold (denote it by $S _ { \mathrm { l i k } } )$ with two coordinate systems consisting of natural parameters (denoted by $\lambda _ { \mathrm { l i k } } )$ and expectation parameters (denoted by $\mu _ { \mathrm { l i k } } )$ , respectively. For the EF in Eq. 2, we have the following:

$$
\lambda _ { \mathrm { l i k } } = \theta , \qquad \mu _ { \mathrm { l i k } } = \mathbb { E } _ { p ( \mathbf { y } \mid \mathbf { \theta } ) } [ \mathbf { y } ] .\tag{3}
$$

It is a well-known fact that these two quantities provide two diferent but equivalent ways to parameterize EFs (for example, see Barndorf-Nielsen (1978)). The coordinates are dual of each other due to their relationship through the Legendre transform. Specifically, we have $\pmb { \mu } _ { \mathrm { l i k } } = \nabla A _ { \mathrm { l i k } } ( \lambda _ { \mathrm { l i k } } )$ and $\lambda _ { \mathrm { l i k } } = \nabla A _ { \mathrm { l i k } } ^ { * } ( \mu _ { \mathrm { l i k } } )$ where $A _ { \mathrm { l i k } } ^ { * }$ is the convex dual of $A _ { \mathrm { l i k } }$ . In information geometry, this is referred to as the dually-flat structure.

## 2.2 The Posterior Distribution

Now, let us introduce the Bayesian framework where a posterior density is obtained by using the above likelihood along with a prior denoted by $p ( \pmb \theta )$ . Using Bayes’ rule, we can write the posterior density as

$$
p ( \pmb \theta | \mathbf y ) = \frac { p ( \mathbf y | \pmb \theta ) p ( \pmb \theta ) } { p ( \mathbf y ) } ,\tag{4}
$$

where the marginal density is denoted by $\begin{array} { r } { p ( \mathbf { y } ) = \int p ( \mathbf { y } \mid \pmb { \theta } ) p ( \pmb { \theta } ) d \pmb { \theta } } \end{array}$

Amari considered a case where the form of the posterior distribution $p ( \pmb \theta | \mathbf { y } )$ with respect to θ is the same as the form of $p ( \mathbf { y } \mid \pmb { \theta } )$ with respect to $\mathbf { y }$ . This difers from the more popular case of conjugate-Bayes where this is not necessarily the case. There, the form of the posterior matches the form of the prior. The likelihood can be expressed in the same form with respect to $\theta ,$ but it is rarely the case that is has the same form also with respect to $\mathbf { y }$ . Amari considered a restrictive case where this is possible by absorbing the prior into the base measure of the likelihood. This is shown below where we first substitute the likelihood from Eq. 2, then rearrange the numerator to write it in a similar form as the likelihood:

$$
\begin{array} { r l } & { p ( \pmb \theta | \mathbf y ) = \underbrace { h _ { \mathrm { l i k } } \big ( \mathbf y \big ) } _ { p ( \mathbf y ) } \exp \left[ \pmb \theta ^ { \top } \mathbf y - A _ { \mathrm { l i k } } ( \pmb \theta ) \right] p ( \pmb \theta ) } \\ & { \qquad = \underbrace { p ( \pmb \theta ) \exp [ - A _ { \mathrm { l i k } } ( \pmb \theta ) ] } _ { = h _ { \mathrm { p o s t } } ( \pmb \theta ) } \exp \left[ \pmb \theta ^ { \top } \underbrace { \textbf { y } } _ { = \lambda _ { \mathrm { p o s t } } } - \underbrace { \big ( \mathrm { l o g } p ( \mathbf y ) - \log h _ { \mathrm { l i k } } ( \mathbf y ) \big ) } _ { = A _ { \mathrm { p o s t } } ( \mathbf y ) } \right] } \\ & { \qquad = h _ { \mathrm { p o s t } } ( \pmb \theta ) \exp \left[ \mathbf y ^ { \top } \pmb \theta - A _ { \mathrm { p o s t } } ( \mathbf y ) \right] . } \end{array}\tag{5}
$$

In the last line, we redefined the base measure and log-partition function to write the posterior in the exact same form as the likelihood in Eq. 2. Note that this rearrangement is only possible if absorbing $p ( \pmb \theta )$ still defines a valid EF distribution where y is the natural parameter (which is rarely the case as mentioned earlier). However, when it is indeed possible to do so, then, similarly to the likelihood, the posterior also forms a manifold (denote it by $S _ { \mathrm { p o s t } } )$ with dual-coordinate systems consisting of the natural and expectation parameters defined below:

$$
\lambda _ { \mathrm { p o s t } } = \mathbf { y } , \qquad \pmb { \mu } _ { \mathrm { p o s t } } = \mathbb { E } _ { p ( \pmb { \theta } | \mathbf { y } ) } [ \pmb { \theta } ] .\tag{6}
$$

Together, the likelihood and posterior give rise to two distinct manifolds.

## 2.3 Bayesian Duality

Amari (1996) showed that the two manifold structures associated with the likelihood and posterior, respectively, are connected to each other through a bijection (Fig. 1a) and the roles of the dual coordinates are also interchanged. This can be more clearly seen in the table below which summarizes the density, natural parameters, suficient statistics, and expectation parameters, respectively. If all instances of y and θ are swapped in the top row, we get the bottom row.

<table><tr><td>Name</td><td>Density</td><td>Nat. param.</td><td>Suff. Stat.</td><td>Exp. param.</td></tr><tr><td>Likelihood</td><td> $p ( \mathbf { y } | \pmb { \theta } ) = h _ { \mathrm { l i k } } ( \mathbf { y } )$  exp  $\left[ \pmb { \theta } ^ { \top } \mathbf { y } - A _ { \mathrm { l i k } } ( \pmb { \theta } ) \right]$ </td><td>θ</td><td>y</td><td> $\mathbb { E } _ { p ( \mathbf { y } \mid \pmb { \theta } ) } [ \mathbf { y } ]$ </td></tr><tr><td>Posterior</td><td> $p ( \pmb \theta | \mathbf y ) = h _ { \mathrm { p o s t } } ( \pmb \theta ) \exp \big [ \mathbf y ^ { \top } \pmb \theta - A _ { \mathrm { p o s t } } ( \mathbf y ) \big ]$ </td><td>y</td><td>θ</td><td> $\mathbb { E } _ { p ( \pmb { \theta } | \mathbf { y } ) } [ \pmb { \theta } ]$ </td></tr></table>

This connection between the likelihood and posterior is formalized by Amari via the new notion of Bayesian duality. Due to bijection, there is a unique mapping between the two manifolds. Therefore, for a likelihood $p ( \mathbf { y } \mid \pmb { \theta } )$ in $S _ { \mathrm { l i k } }$ , there exists a unique posterior $p ( \pmb \theta | \mathbf { y } )$ in $S _ { \mathrm { p o s t } }$ , and vice versa. The roles of natural parameters and suficient statistics are swapped. Amari referred to this as a new dual structure associated with Bayesian inference, giving rise to a new Bayesian duality theory.

To further illustrate the point, we give a simple example of the dual structure for a case where both likelihood and posterior are isotropic Gaussians.

Ex 1. Consider a Gaussian likelihood function for y written in an EF form:

$$
\begin{array} { r } { p ( { \mathbf y } | { \pmb \theta } ) = \mathcal { N } ( { \mathbf y } | { \pmb \theta } , { \mathbf I } ) = \underbrace { ( 2 \pi ) ^ { - D / 2 } \exp ( - \frac { 1 } { 2 } { \mathbf y } ^ { \top } { \mathbf y } ) } _ { h _ { l i k } ( { \mathbf y } ) } \exp \Big [ { \mathbf y } ^ { \top } { \pmb \theta } - \underbrace { \frac { 1 } { 2 } { \pmb \theta } ^ { \top } { \pmb \theta } } _ { A _ { l i k } ( { \pmb \theta } ) } \Big ] , } \end{array}\tag{7}
$$

For this likelihood, the natural parameter is $\lambda _ { l i k } = \pmb \theta$ and the expectation parameter is also the same because $\pmb { \mu } _ { l i k } = \nabla A _ { l i k } ( \pmb { \theta } ) = \pmb { \theta }$ The dual-coordinate pair is simply $( \lambda _ { l i k } , \pmb { \mu } _ { l i k } ) = ( \pmb { \theta } , \pmb { \theta } )$

Let us turn to the posterior density next. We want the posterior to have the same form as the likelihood. A Gaussian prior ensures that the posterior is Gaussian as well, but it does not ensure that the posterior covariance is also identity. However, this happens if we choose a uniform prior. In that case, the marginal density is also uniform, as shown below,

$$
p ( \mathbf { y } ) = \int { \mathcal { N } } ( \mathbf { y } \mid \theta , \mathbf { I } ) d \theta = \int { \mathcal { N } } ( \theta \mid \mathbf { y } , \mathbf { I } ) d \theta = 1 .\tag{8}
$$

As a result, we can write the posterior as an isotropic Gaussian density:

$$
p ( \pmb \theta | \mathbf y ) = \frac { \mathcal { N } ( \mathbf { y } | \pmb \theta , \mathbf I ) p ( \pmb \theta ) } { p ( \mathbf { y } ) } = \mathcal { N } ( \pmb \theta | \mathbf y , \mathbf I ) .\tag{9}
$$

This also shows that the posterior mean is simply equal to $\mathbf { y } . \ B y$ writing the density in the EF form as in Eq. 7 we can show that the natural parameter is also equal to y and that the posterior takes the same form as the likelihood:

$$
\begin{array} { r } { p ( \pmb { \theta } | \mathbf { y } ) = \underbrace { ( 2 \pi ) ^ { - D / 2 } \exp ( - \frac { 1 } { 2 } \pmb { \theta } ^ { \top } \pmb { \theta } ) } _ { h _ { p o s t } ( \pmb { \theta } ) } \exp \Big [ \pmb { \theta } ^ { \top } \mathbf { y } - \underbrace { \frac { 1 } { 2 } \mathbf { y } ^ { \top } \mathbf { y } } _ { A _ { p o s t } ( \mathbf { y } ) } \Big ] , } \end{array}\tag{10}
$$

Therefore, the dual-coordinate pair is $( \lambda _ { p o s t } , \pmb { \mu } _ { p o s t } ) = ( \mathbf { y } , \mathbf { y } )$ . As Amari pointed out, the roles of θ and y are interchanged for manifolds S<sub>lik</sub> and $S _ { p o s t }$ and there is a trivial bijection between them.

## 2.4 How to Generalize Amari’s Framework?

The main limitation of Amari’s Bayesian duality is the restriction that the likelihood and posterior need have the same form. Amari did show an example on the Boltzmann machine but applying it to even simple machine-learning models runs into dificulties. We will now discuss this dificulty for a simple ridge regression problem. We will show that a dual structure can still be found but handling the general cases requires a diferent mathematical framework. In the next section, we will present one such framework based on convex duality.

Consider a one-dimensional ridge regression problem with a scalar parameter $\theta \in \mathbb { R }$ to model N scalar outputs $y _ { i } \in \mathbb { R }$ given scalar inputs $x _ { i } \in \mathbb { R }$ . We denote the vectors of outputs and inputs by y and x respectively (both of length N). We assume a Gaussian likelihood with variance 1 and a standard Gaussian prior:

$$
p ( \mathbf { y } \mid { \boldsymbol { \theta } } ) = { \mathcal { N } } ( \mathbf { y } \mid \mathbf { x } { \boldsymbol { \theta } } , \mathbf { I } ) \qquad p ( { \boldsymbol { \theta } } ) = { \mathcal { N } } ( { \boldsymbol { \theta } } \mid 0 , 1 ) .\tag{11}
$$

With this choice, the posterior is a Gaussian density as well:

$$
p ( { \boldsymbol { \theta } } \mid \mathbf y ) = { \mathcal { N } } ( { \boldsymbol { \theta } } \mid m _ { * } , \sigma _ { * } ^ { 2 } ) { \mathrm { ~ w h e r e ~ } } m _ { * } = \sigma _ { * } ^ { 2 } \mathbf x ^ { \top } \mathbf y { \mathrm { ~ a n d ~ } } \sigma _ { * } ^ { 2 } = 1 / ( \mathbf x ^ { \top } \mathbf x + 1 ) .\tag{12}
$$

Unlike the likelihood, the posterior has a variance that is not equal to 1. The likelihood and posterior therefore take diferent EF forms, and interchanging the roles of the dual coordinates does not make much sense. However, it turns out that we can still find a mapping between the two manifolds. We will first describe the interchanging process and give a formal mathematical framework in the next section.

The key idea is to express the likelihood in terms of the posterior’s suficient statistics. For example, for the posterior in Eq. 12, the suficient statistic is a 2D vector

$$
\mathbf { T } ( \theta ) = ( \theta , \theta ^ { 2 } ) ^ { \top } ,\tag{13}
$$

and it is therefore possible to write the posterior in an EF form as

$$
p ( \boldsymbol { \theta } \mid \mathbf { y } ) \propto \exp ( \langle \lambda _ { \mathrm { p o s t } } , \mathbf { T } ( \boldsymbol { \theta } ) \rangle ) ,\tag{14}
$$

where $\langle \cdot , \cdot \rangle$ is an inner product and $\lambda _ { \mathrm { p o s t } }$ is the natural parameter of the posterior; see Khan and Rue (2023, Sec. 5). Our idea is to first write the likelihood in the same form, for example, we can find a vector $\widetilde { \lambda } _ { \mathrm { l i k } }$ such that

$$
p ( \mathbf { y } \mid \boldsymbol { \theta } ) \propto t ( \boldsymbol { \theta } ) = \exp ( \langle \widetilde { \lambda } _ { \mathrm { l i k } } , \mathbf { T } ( \boldsymbol { \theta } ) \rangle ) ,\tag{15}
$$

and then write a dual problem as an inverse mapping to find $t ( \theta )$ , giving rise to a dual structure. This is the basic idea of our setup.

We will now show how to write the likelihood in this form and that doing so yields an interchanging of the coordinates, which is similar in spirit to Amari’s case.

Ex 2. To show this, we first write the likelihood in the canonical EF form as in the previous example and then redefine the suficient statistics and natural parameters to write them as an inner product,

$$
\begin{array}{c} \begin{array} { r l } & { p ( \mathbf { y } \mid \boldsymbol { \theta } ) = \underbrace { ( 2 \pi ) ^ { - \frac { N } { 2 } } \exp ( - \frac { 1 } { 2 } \mathbf { y } ^ { \top } \mathbf { y } ) } _ { h _ { l i k } ( \mathbf { y } ) } \exp \Big ( \underbrace { \mathbf { y } } _ { \begin{array} { c } { \mathbf { T } _ { l i k } ( \mathbf { y } ) } \end{array} } \underbrace { \boldsymbol { \tau } ( \underline { { \mathbf { x } } } \boldsymbol { \theta } ) } _ { \begin{array} { c } { \lambda _ { l i k } } \end{array} } - \underbrace { \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { x } \boldsymbol { \theta } ^ { 2 } } _ { A _ { l i k } ( \lambda _ { l i k } ) } \Big ) } \\ & { \quad \quad \quad = h _ { l i k } ( \mathbf { y } ) \exp \Bigg [ \Bigg \langle \underbrace { \Bigg [ \mathbf { y } _ { \bigg ] } ^ { \prime } } _ { \begin{array} { c } { \Bigg [ \underline { { \boldsymbol { \mathrm {  ~ \delta ~ } } } } _ { \begin{ c } { 1 } \\ { 1 } \end{array} } \Bigg ] } , \underbrace { \Bigg [ \mathbf { x } \boldsymbol { \theta } } _ { \begin{array} { c } { 1 } \\ { - \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { x } \boldsymbol { \theta } ^ { 2 } } \end{array} } \Bigg ] } \Bigg \rangle \Bigg ] } \end{array}  \end{array}\tag{16}
$$

In the second line, we have defined an inner product between $( N + 1 )$ -dimensional vectors where we denote the new suficient statistics and natural parameter by $\widehat { \mathbf { T } } _ { l i k }$ and $\widehat { \lambda } _ { l i k }$ , respectively. Next, we write the term inside the exponential as an inner product between 2D vectors which yields a form in terms of $\mathbf { T } ( \theta )$ on the right,

$$
\underbrace {  [ \mathbf { y } ]  } _ { \mathbf { \hat { T } } _ { l i k } ( \mathbf { y } ) } , \underbrace { [ \mathbf { x } \theta  } _ {  \mathbf { \hat { \lambda } } _ { l i k }  } ]  _ { \mathbf { \hat { \lambda } } _ { l i k } }  = \underbrace {  [ \begin{array} { l } { \mathbf { x } ^ { \top } \mathbf { y } } \\ { \mathbf { \bar { \lambda } } _ { - \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { x } } ^ { } } \end{array} ] } _ { \mathbf { \hat { \lambda } } _ { l i k } } , \underbrace { [ \theta ] } _ { \mathbf { T } ( \theta ) }   _ { \mathbf { \hat { \lambda } } _ { l i k } } .\tag{17}
$$

The whole rearrangement can be more compactly written as

$$
\begin{array} { r } { \langle \widehat { \mathbf { T } } _ { l i k } ( \mathbf { y } ) , \widehat { \lambda } _ { l i k } \rangle = \langle \widetilde { \lambda } _ { l i k } , \mathbf { T } ( \theta ) \rangle , } \end{array}\tag{18}
$$

which shows an interchanging of the suficient statistic and natural parameter.

The example suggests that, do find a dual mapping, we rewrite Bayes’ rule in a form where we replace the likelihood by its mapping t(θ) in the space of $\mathbf { T } ( \pmb \theta )$

$$
p ( \pmb \theta | \mathbf y ) \propto t ( \pmb \theta ) p ( \pmb \theta )\tag{19}
$$

This way the forward map is simply Bayes’ rule, while the reverse map is to find $t ( \pmb \theta )$ given the posterior $p ( \pmb \theta | \mathbf { y } )$ . We will show next that these operations are written as duals of each other through convex duality. We will use it to generalize Amari’s Bayesian duality to general Bayesian cases.

## 3 Bayesian Duality via Convex Duality

We will now derive a dual structure of Bayes’ rule by using convex duality. This framework removes the restrictions imposed in Amari’s work. We do not claim that this solution is better than using information geometry. Rather, we think that this viewpoint is helpful to connect Amari’s work to other works that also use convex duality for Bayesian inference. The key idea is to write Bayes’ rule as an optimization problem and then use its dual to find an inverse mapping to search for the likelihood mapping $t ( \pmb \theta ) = \mathrm { e x p } ( \langle \widetilde \lambda , \mathbf T ( \pmb \theta ) \rangle )$ ) in a dual space.

## 3.1 Variational Formulation of Bayes’ Rule

We start by writing a variational form of Bayes’ rule, also known as Gibbs’ variational principle in information theory. It is well known that Bayes’ rule can be written as an optimization problem in the space of all densities (denoted by $\mathcal { P } )$ over θ. Consider the following optimization problem over candidate densities $q \in \mathcal { P }$

$$
\boldsymbol { q } _ { \ast } ( \pmb { \theta } ) = \underset { \boldsymbol { q } \in \mathcal { P } } { \arg \operatorname* { s u p } } ~ \mathbb { E } _ { \boldsymbol { q } ( \pmb { \theta } ) } [ \log p ( \mathbf { y } | \pmb { \theta } ) ] - \mathbb { D } _ { \mathrm { K L } } ( \boldsymbol { q } ( \pmb { \theta } ) \| \boldsymbol { p } ( \pmb { \theta } ) ) ,\tag{20}
$$

where the second term on the right is the Kullback-Leibler divergence (KLD) between the candidate $q ( \pmb \theta )$ and the prior $p ( \pmb \theta )$ . It is well-known that the optimal $q _ { * }$ is the posterior density, that is, $q _ { * } ( \pmb { \theta } ) = p ( \pmb { \theta } | \mathbf { y } )$ . One can also show that the optimal value is simply the log-partition function. More precisely, we have

$$
\log \int p ( \mathbf { y } | \pmb { \theta } ) p ( \pmb { \theta } ) d \pmb { \theta } = \operatorname* { s u p } _ { \pmb { q } \in \mathcal { P } } \ \mathbb { E } _ { \pmb { q } ( \pmb { \theta } ) } [ \log p ( \mathbf { y } | \pmb { \theta } ) ] - \mathbb { D } _ { \mathrm { K L } } ( \pmb { q } ( \pmb { \theta } ) | | p ( \pmb { \theta } ) ) .\tag{21}
$$

The left hand side is the log-partition function $\log p ( \mathbf { y } )$ . The result is easy to verify by simply setting $q = q ,$ <sub>∗</sub> on the right-hand side:

$$
\mathbb { E } _ { q _ { * } ( \pmb { \theta } ) } [ \log p ( \mathbf { y } | \pmb { \theta } ) ] - \mathbb { D } _ { \mathrm { K L } } ( q _ { * } ( \pmb { \theta } ) | | p ( \pmb { \theta } ) ) = \mathbb { E } _ { q _ { * } ( \pmb { \theta } ) } \left[ \log \frac { p ( \mathbf { y } | \pmb { \theta } ) p ( \pmb { \theta } ) } { q _ { * } ( \pmb { \theta } ) } \right] = \log p ( \mathbf { y } ) .\tag{22}
$$

This result is attributed to Gibbs (1902) and is sometimes referred to as the Gibbs variational principle. The connection to $\mathrm { B a y e s } ^ { \mathrm { , } }$ rule was formalized by Jaynes, starting from his work on the maximum-entropy principle (Jaynes, 1957). A general result that goes beyond likelihood functions and applies to generic losses is given in a series of papers by Donsker and Varadhan (1975a,b, 1976, 1983).

## 3.2 A Dual of the Gibbs Variational Formulation

Amari’s bijection between posterior and likelihood can be derived as a special case of a dual version of the Gibbs variational principle. The goal of the dual problem is to find a likelihood function given a posterior distribution $q ( \pmb { \theta } ) \in \mathcal { P }$ . For this purpose, let us define another space ${ \mathcal { P } } ^ { * }$ consisting of valid likelihood functions. For instance, we can choose potential likelihoods $t ( \pmb \theta )$ from a function space $\mathcal { F }$ for which the log-partition is finite. Formally, we define

$$
\mathcal { P } ^ { * } = \left\{ t \in \mathcal { F } : \log \int t ( \pmb \theta ) p ( \pmb \theta ) d \pmb \theta < \infty \right\} .\tag{23}
$$

Given such a space, we can write the dual problem as the following:

$$
\mathbb { D } _ { \mathrm { K L } } \big ( q ( \pmb \theta ) \| p ( \pmb \theta ) \big ) = \operatorname* { s u p } _ { t \in \mathcal { P } ^ { * } } \mathbb { E } _ { q ( \pmb \theta ) } [ \log t ( \pmb \theta ) ] - \log \int t ( \pmb \theta ) p ( \pmb \theta ) d \pmb \theta .\tag{24}
$$

This dual problem recovers the KLD between the posterior $q ( \pmb \theta )$ and the prior $p ( \pmb \theta )$ by solving an optimization problem over a function space ${ \mathcal { P } } ^ { * }$ . We propose it as a generalization of Amari’s Bayesian duality (Fig. 1b). We refer to $\operatorname { E q . }$ 24 as the dual of the Gibbs’ variational principle, even though diferent names are used in the literature. For instance, in information theory, this is sometimes referred to as the Donsker-Varadhan formula (Polyanskiy and Wu, 2025, Thm. 4.6).

For a given y, if we set $q ( \pmb \theta ) = q _ { * } ( \pmb \theta )$ (the posterior $p ( \pmb \theta | \mathbf { y } )$ from Eq. 20), then the assertion is that an optimal solution $t _ { * } ( \pmb \theta )$ recovers the likelihood function $p ( \mathbf { y } \mid \pmb { \theta } )$ up to a constant factor that is absorbed in the normalization. More precisely, when $t _ { * } ( \pmb \theta )$ is plugged in Bayes’ rule, it will yield the posterior $p ( \pmb \theta | \mathbf { y } )$ . The equality can be verified by substituting the optimal $t ( { \pmb \theta } ) = p ( \mathbf { y } | { \pmb \theta } )$ and $q = q _ { * }$ ,

$$
\mathbb { E } _ { q _ { * } ( \theta ) } [ \log p ( \mathbf { y } | \theta ) ] - \log \int p ( \mathbf { y } | \theta ) p ( \theta ) d \theta = \mathbb { E } _ { q _ { * } ( \theta ) } \left[ \log \frac { p ( \mathbf { y } | \theta ) } { p ( \mathbf { y } ) } \right] = \mathbb { E } _ { q _ { * } ( \theta ) } \left[ \log \frac { q _ { * } ( \theta ) } { p ( \theta ) } \right] .
$$

## 3.3 Amari’s Bayesian Duality via Convex Duality

Before giving more details, we connect the above duality to Amari’s bijection by applying it to the Gaussian case discussed in Ex 1.

Ex 3. We start by writing the variational form of Eq. 20. Because we know that the posterior is an isotropic Gaussian, we can simplify the optimization by restricting it to that space, that is, we set $q ( \pmb \theta ) = \mathcal { N } ( \pmb \theta | \mathbf { m } , \mathbf { I } )$ . This enables us to write the optimization problem over the mean m. This is shown below:

$$
\begin{array} { c } { { \bf m _ { * } = \arg \operatorname* { s u p } _ { \bf m } ~ \mathbb { E } _ { \mathcal { N } ( \pmb { \theta } \mid { \bf m } , { \bf I } ) } [ \log \mathcal { N } ( \bf y \mid \pmb { \theta } , { \bf I } ) ] - \mathbb { D } _ { K L } ( \mathcal { N } ( \pmb { \theta } \mid { \bf m } , { \bf I } ) \parallel p ( \pmb { \theta } ) ) } } \\ { { \bf \alpha = \arg \operatorname* { s u p } _ { \bf m } ~ - \frac { 1 } { 2 } ( \bf y - m ) ^ { \top } ( \bf y - m ) . } } \end{array}\tag{25}
$$

We see that Bayes’ rule in this case reduces to optimization of a quadratic function with maximum $\mathbf { m } _ { * } = \mathbf { y }$

A similar derivation can be used to write the dual form as a quadratic problem. Mimicking the form of the likelihood, we define log $t ( \pmb \theta )$ to be a quadratic,

$$
\begin{array} { r } { t ( \pmb \theta ) = \exp \left( \widetilde { \lambda } ^ { \top } \pmb \theta - \frac { 1 } { 2 } \pmb \theta ^ { \top } \pmb \theta \right) , } \end{array}\tag{26}
$$

with $\widetilde { \lambda }$ as the free parameter. Given a posterior $q _ { * } = \mathcal { N } ( \pmb { \theta } | \mathbf { m } _ { * } , \mathbf { I } )$ our goal then is to find the optimal $\widetilde { \lambda } _ { * }$ . The dual problem in Eq. 24 can then be written as

$$
\begin{array} { l } { { \displaystyle \widetilde \lambda _ { * } = \arg \operatorname* { s u p } _ { { \widetilde \lambda } } ~ \mathbb { E } _ { \mathcal { N } ( \theta | { \bf m } _ { * } , { \bf I } ) } [ \log t ( \theta ) ] - \log \int t ( \theta ) p ( \theta ) d \theta } } \\ { { \displaystyle ~ = \arg \operatorname* { s u p } _ { { \widetilde \lambda } } ~ \mathbb { E } _ { \mathcal { N } ( \theta | { \bf m } _ { * } , { \bf I } ) } \left[ { \widetilde \lambda } ^ { \top } \theta - \frac { 1 } { 2 } \theta ^ { \top } \theta \right] - \log \int \exp \left( { \widetilde \lambda } ^ { \top } \theta - \frac { 1 } { 2 } \theta ^ { \top } \theta \right) p ( \theta ) d \theta } } \\ { { \displaystyle ~ = \arg \operatorname* { s u p } _ { { \widetilde \lambda } } ~ \widetilde \lambda ^ { \top } { \bf m } _ { * } - \frac { 1 } { 2 } { \bf m } _ { * } ^ { \top } { \bf m } _ { * } - \frac { D } { 2 } \log ( 2 \pi ) - \frac { 1 } { 2 } { \widetilde \lambda } ^ { \top } { \widetilde \lambda } } } \\ { { \displaystyle ~ = \arg \operatorname* { s u p } _ { { \widetilde \lambda } } ~ \widetilde \lambda ^ { \top } { \bf m } _ { * } - \frac { 1 } { 2 } { \widetilde \lambda } ^ { \top } { \widetilde \lambda } . } } \end{array}
$$

This is a quadratic function whose solution is $\widetilde { \lambda } _ { * } = \mathbf { m } _ { * } = \mathbf { y }$ Therefore, we have $t _ { * } ( \pmb \theta ) \propto \mathcal { N } ( \mathbf { y } | \pmb \theta , \mathbf { I } )$ We note that the function $t ( \theta )$ need not be a probability density, but its role is identical. If we replace $p ( \mathbf { y } \mid \pmb { \theta } )$ in $E q .$ 20 by $t _ { * } ( \pmb \theta )$ then we recover the posterior $q _ { * } ( \theta )$ which is also equal to the posterior $p ( \pmb \theta | \mathbf { y } )$

This example illustrates that the primal-dual formulation of the Gibbs’ variational principle can realize the bijection proposed by Amari when we use the specific form of the likelihoods and posterior. This dual version however applies much more generally. In fact, it applies even when likelihoods are replaced by generic loss functions. We will now show an application to the ridge regression case.

## 3.4 Bayesian Duality of Ridge Regression

We return to the example of ridge regression shown in Eq. 11. Amari’s framework is dificult to apply to this case and we will show that by using the dual problem over $t ( \theta )$ , we can solve this issue. We showed in Eq. 17 of Ex 2 that the optimal likelihood mapping is a 2D vector: $\begin{array} { r } { \widetilde { \lambda } _ { \mathrm { l i k } } = ( \mathbf { x } ^ { \top } \mathbf { y } , - \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { x } ) ^ { \top } } \end{array}$ . We will now show that this solution can be recovered by solving the dual problem.

Ex 4. We restrict log $t ( \theta )$ to be a quadratic function with two scalar parameters $\widetilde { \lambda } _ { 1 }$ and $\widetilde { \lambda } _ { 2 }$ , as shown below:

$$
\begin{array} { r } { t ( \theta ) = \exp \left( \widetilde { \lambda } _ { 1 } \theta - \frac { 1 } { 2 } \widetilde { \lambda } _ { 2 } \theta ^ { 2 } \right) . } \end{array}\tag{27}
$$

Given a posterior $q _ { * } = \mathcal { N } ( \theta | m _ { * } , \sigma _ { * } ^ { 2 } )$ our goal is to find the optimal ${ \widetilde { \lambda } } _ { 1 , }$ <sub>∗</sub> and $\widetilde { \lambda } _ { 2 , * }$ The dual problem in Eq. 24 can be written as

$$
\begin{array} { r l } & { \widetilde { \lambda } _ { * } = \underset { \widetilde { \lambda } } { \arg \operatorname* { s u p } } ~ \mathbb { E } _ { \mathcal { N } ( \theta \mid m _ { * } , \sigma _ { * } ^ { 2 } ) } [ \log t ( \theta ) ] - \log \int t ( \theta ) p ( \theta ) d \theta } \\ & { \quad \quad = \underset { \widetilde { \lambda } } { \arg \operatorname* { s u p } } ~ \widetilde { \lambda } _ { 1 } m _ { * } - \frac { 1 } { 2 } \widetilde { \lambda } _ { 2 } ( m _ { * } ^ { 2 } + \sigma _ { * } ^ { 2 } ) - \log \int \exp \left( \widetilde { \lambda } _ { 1 } \theta - \frac { 1 } { 2 } \widetilde { \lambda } _ { 2 } \theta ^ { 2 } \right) p ( \theta ) d \theta . } \end{array}\tag{28}
$$

The integral is available in closed form:

$$
\log \int \exp \left( \widetilde { \lambda } _ { 1 } \theta - \textstyle { \frac { 1 } { 2 } } \widetilde { \lambda } _ { 2 } \theta ^ { 2 } \right) p ( \theta ) d \theta = - \textstyle { \frac { 1 } { 2 } } \log ( \widetilde { \lambda } _ { 2 } + 1 ) + \textstyle { \frac { 1 } { 2 } } \frac { \widetilde { \lambda } _ { 1 } ^ { 2 } } { \widetilde { \lambda } _ { 2 } + 1 } .\tag{29}
$$

Substituting this in the optimization problem and taking derivatives, we get

$$
\begin{array} { r } { \widetilde { \lambda } _ { 1 , * } = m _ { * } / \sigma _ { * } ^ { 2 } = \mathbf { x } ^ { \top } \mathbf { y } , \qquad \widetilde { \lambda } _ { 2 , * } = 1 / \sigma _ { * } ^ { 2 } - 1 = \mathbf { x } ^ { \top } \mathbf { x } , } \end{array}\tag{30}
$$

which is the desired solution. A detailed derivation is in Sec. A.1.

The dual problem yields the optimal mapping for the likelihood in the $\mathbf { T } ( \theta )$ space. Using the interchanging operation shown in Eq. 18 it may be possible to then write an appropriate likelihood. Unlike Amari’s example, this operation will not be a bijection in general because there could be several ways to aggregate the information.

It is also possible to write the dual problem directly in the data space. For instance, we can rewrite the dual problem in the space of the data vector y which is of length

N. This is the space of linear predictions denoted by

$$
\mathbf { f } = \mathbf { x } \theta .\tag{31}
$$

This formulation can recover the likelihood exactly, as shown next.

Ex 5. The dual problem can be written in the f-space by using a pushforward of the distribution. Specifically, we can write the prior and posterior in the f-space using a change of variable.

$$
\begin{array} { r } { p ( \mathbf { f } ) = \mathcal { N } ( \mathbf { f } \mid 0 , \mathbf { x x } ^ { \top } ) , \qquad q _ { * } ( \mathbf { f } ) = \mathcal { N } ( \mathbf { f } \mid \mathbf { x } m _ { * } , \sigma _ { * } ^ { 2 } \mathbf { x x } ^ { \top } ) . } \end{array}\tag{32}
$$

The prior is degenerate but that does not pose a problem as long as the integrals with respect to it are finite. The likelihood candidate can also be mapped in that space by redefining the free parameters to be of appropriate sizes:

$$
\begin{array} { r } { t ( \mathbf { f } ) = \exp \left( \mathbf { a } ^ { \top } \mathbf { f } - \frac { 1 } { 2 } \mathbf { f } ^ { \top } \mathbf { B } \mathbf { f } \right) , } \end{array}\tag{33}
$$

where a is an N-length real vector while B is an $N \times N$ real matrix. With these, the dual problem can be written in the f-space, as an optimization over a and B,

$$
\begin{array} { r l r } {  { ( \mathbf { a } _ { * } , \mathbf { B } _ { * } ) = \arg \operatorname* { s u p } _ { \mathbf { a } , \mathbf { B } } \ \mathbb { E } _ { q _ { * } ( \mathbf { f } ) } [ \log { t ( \mathbf { f } ) } ] - \log \int t ( \mathbf { f } ) p ( \mathbf { f } ) d \mathbf { f } } }  & { } & { ( 3 4 ) } \\ & { } & { = \arg \operatorname* { s u p } _ { \mathbf { a } , \mathbf { B } } \ \mathbf { a } ^ { \top } \mathbf { x } m _ { * } - \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x } ( m _ { * } ^ { 2 } + \sigma _ { * } ^ { 2 } ) - \log \int \exp ( \mathbf { a } ^ { \top } \mathbf { f } - \frac { 1 } { 2 } \mathbf { f } ^ { \top } \mathbf { B } \mathbf { f } ) p ( \mathbf { f } ) d \mathbf { f } . } \end{array}
$$

The integral in the last term can be obtained by changing back the variable to θ and then substituting in $E q .$ 29 the following: $\lambda _ { 1 } = \mathbf { a } ^ { \top } \mathbf { x }$ and $\tilde { \lambda } _ { 2 } = \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x }$ Using this, we can write the optimization problem as follows:

$$
\begin{array} { r } { \mathbf { a } ^ { \top } \mathbf { x } m _ { * } - \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x } ( m _ { * } ^ { 2 } + \sigma _ { * } ^ { 2 } ) + \frac { 1 } { 2 } \log ( \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x } + 1 ) - \frac { 1 } { 2 } \frac { ( \mathbf { a } ^ { \top } \mathbf { x } ) ^ { 2 } } { \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x } + 1 } . } \end{array}\tag{35}
$$

Taking the derivatives, we get the optimality condition that is identical to Eq. 30,

$$
\begin{array} { r } { \mathbf { x } ^ { \top } \mathbf { a } _ { * } = m _ { * } / \sigma _ { * } ^ { 2 } = \mathbf { x } ^ { \top } \mathbf { y } , \qquad \mathbf { x } ^ { \top } \mathbf { B } _ { * } \mathbf { x } = 1 / \sigma _ { * } ^ { 2 } - 1 = \mathbf { x } ^ { \top } \mathbf { x } . } \end{array}\tag{36}
$$

These are satisfied by $\mathbf { a } _ { * } = \mathbf { y }$ and $\mathbf { B } _ { * } = \mathbf { I } ,$ so a solution recovers the likelihood,

$$
\begin{array} { r } { t _ { * } ( \mathbf { f } ) = \exp \big ( \mathbf { y } ^ { \top } \mathbf { x } \theta - \frac { 1 } { 2 } \mathbf { x } ^ { \top } \mathbf { x } \theta ^ { 2 } \big ) \propto \mathcal { N } ( \mathbf { y } | \mathbf { x } \theta , \mathbf { I } ) = p ( \mathbf { y } | \theta ) . } \end{array}\tag{37}
$$

This dual formulation yields the likelihood in $p ( \mathbf { y } \mid \boldsymbol { \theta } )$ space, but since the dual solution is not unique, there is no bijection between posterior and likelihood. The formulation automatically performs the interchanging operation shown in Eq. 18 similarly to Amari’s case.

![](images/d2d8c5ac20e303580ba5be489d5930946143d4d1df31b75bcd2187fdd1386cff.jpg)  
Fig. 2: Amari’s motivation was to explain the dynamic interaction between a higherorder system (concept machine) and a lower order system, through an iterative feedforward and feedback mechanism. In modern times, Bayesian duality could be useful to study the relationship between the model and data. For example, the dual problem can be used to trace the knowledge of an LLM.

## 4 Relevance for AI and Connections to Other Fields

So far, we have shown that Amari’s Bayesian duality can be generalized through convex duality where the dual problem can be seen as an inverse (and sometimes bijective) mapping from the posterior manifold to the likelihood manifold. But, why is this relevant for machine learning and AI?

Amari’s original proposal was focused on information processing in the brain. At the time, he aimed to develop a new theory of “dynamic interaction between a lower and a higher neural systems” (Fig. 2a). The lower order system, modeled by y, is connected to a sensory input while the higher order system is modeled by θ which is connected to a ‘concept’ machine. He imagined Bayesian duality as a stochastic version of the autoassociative-memory model where the concepts are encoded as binary vectors in the higher system. When a new sensory input is shown, the concept machine uses a ‘feedforward’ process to estimate a concept that best explains it. This is the inference part to get $p ( \pmb \theta | \mathbf y )$ . Then, the higher neural system stimulates the lower one by proposing a new y through the feedback process. This is realized by $p ( \mathbf { y } \vert \pmb { \theta } )$ . This dynamic interaction was the motivation for Amari to develop Bayesian duality, which he proposed to implement through an e-m procedure.

For us too, the interaction between y and θ is the most attractive part of Bayesian duality. In our research, we are interested in Bayesian duality as a tool to study the relationship between the model and data. For example, given a Large Language Model (LLM) trained on a large data, we would like to understand the model’s knowledge. This question can be framed as the inverse mapping: we want to find a compact set of data examples (or prompts) that can summarize the main crux of the LLM’s knowledge. The dual problem can be used to address such problems: given $p ( \pmb \theta | \mathbf y )$ , the mapping $t ( \pmb \theta )$ in the likelihood space can be used to ‘trace’ a compact summary of the model’s knowledge (Fig. 2b). Amari himself was interested in understanding the underlying concepts encoded in θ, which he modeled through a subspace in the manifold spanned by a smaller set of coordinates.

Our work here connects Amari’s ideas to other works that rely on convex duality to find compact representations. One of the earliest works on this is the representer theorem proposed by Kimeldorf and Wahba (1970) in a Bayesian context. Later, this led to the representer theorem used in kernel methods (Sch¨olkopf et al., 2001) and the support vector machine (SVM) where compact representations in the data space are found by solving an eficient convex-optimization problem. A representer theorem for Gaussian process models was proposed in Csat´o and Opper (2002, Lemma 1). General versions of such dualities have also been studied extensively, for example, by Altun and Smola (2006) for function spaces and by Dud´ık et al. (2007) to generalize the maximum entropy principle. An $\mathrm { S V M - s t y }$ le generalization of Bayesian inference is proposed by Zhu et al. (2014). We also note that an informal mention of Bayesian duality is made in Robert (2007) for Bayes’ rule as an inversion but Amari’s usage is more precise. More recently, such dualities are used to obtain primal-dual adversarial interpretations of Bayes (Husain and Knoblauch, 2022; M¨ollenhof and Khan, 2023).

In fact, Bayesian duality can recover some non-Bayesian applications of convex duality in machine learning. For instance, the dual problem for ridge regression of Eq. 11 is a classical reformulation of minimization over $\theta$ as a maximization over an $N .$ -length real-valued dual vector α (Rockafellar, 1967). This is shown below,

$$
\begin{array} { r } { \underset { \theta } { \mathop { \operatorname* { m i n } } } \ - \log p ( \mathbf { y } | \theta ) - \log p ( \theta ) \quad \Longleftrightarrow \quad \underset { \alpha } { \mathop { \operatorname* { m a x } } } \ \mathbf { y } ^ { \top } \alpha - \frac { 1 } { 2 } \alpha ^ { \top } \big ( \mathbf { I } + \mathbf { x x } ^ { \top } \big ) \alpha . } \end{array}\tag{38}
$$

We can show that, for this case, the optimal dual vector $\pmb { \alpha } _ { \mathrm { : } }$ <sub>∗</sub> is related to $t _ { * } ( \mathbf { f } _ { * } )$ in Eq. 33 as follows,

$$
\begin{array} { r } { \pmb { \alpha } _ { * } = \mathbf { y } - \mathbf { f } _ { * } = \nabla \log t _ { * } ( \mathbf { f } _ { * } ) , } \end{array}\tag{39}
$$

where $\mathbf { f } _ { * } = \mathbf { x } \theta ,$ <sub>∗</sub> is the optimal prediction. A proof is given in Sec. A.2. In general, such equivalences between the non-Bayesian and Bayesian dualities are to be expected due to their common origins in convex duality. We also expect Bayesian duality with more expressive posterior forms to contain as special cases other dualities based on less flexible posteriors and also those that arise in non-Bayesian scenarios.

Recent works have tried to build such versions of Bayesian dualities by using variational approximations. Such methods use dual structures associated with simple posterior approximations (such as Gaussian approximations to the posterior). This makes it possible to derive practical algorithms that can exploit the benefits of Bayesian duality. One such early work is by Khan et al. (2013) which proposed a dual formulation for Gaussian variational inference. More recently, Adam et al. (2021) proposed the use of such dual variables for approximate inference in Gaussian processes. Khan (2025) uses dual representations to unify knowledge-adaptation methods, while M¨ollenhof et al. (2026) used a dual structure to generalize ADMM for federated learning. The work started by Amari on Bayesian duality is now finally having an impact on modern AI and we hope that it will continue to benefit the community.

Acknowledgments. This work is supported by JST CREST Grant Number JPMJCR2112. We thank many current and past members of the ABI team and the Bayes-duality project for discussions over the years which played an important role in shaping the ideas presented in this paper.

## A Appendix

## A.1 Derivation of the Dual Problem for Ridge Regression

To evaluate the integral in Eq. 29, we use a change of variables and the fact that $\textstyle \int \exp ( - z ^ { 2 } / 2 ) \mathrm { d } z = { \sqrt { 2 \pi } }$ . First, we bring the integral into a form amenable to the change of variables.

$$
\int \exp \left( \widetilde { \lambda } _ { 1 } \theta - \textstyle { \frac { 1 } { 2 } } \widetilde { \lambda } _ { 2 } \theta ^ { 2 } \right) p ( \theta ) d \theta = \frac { 1 } { \sqrt { 2 \pi } } \int \exp \left( \widetilde { \lambda } _ { 1 } \theta - \frac { 1 } { 2 } ( \widetilde { \lambda } _ { 2 } + 1 ) \theta ^ { 2 } \right) d \theta\tag{40}
$$

$$
= \frac { 1 } { \sqrt { 2 \pi } } \exp \left( \frac { \widetilde { \lambda } _ { 1 } ^ { 2 } } { 2 ( \widetilde { \lambda } _ { 2 } + 1 ) } \right) \int \exp \left( - \frac { 1 } { 2 } ( \widetilde { \lambda } _ { 2 } + 1 ) \left( \theta - \frac { \widetilde { \lambda } _ { 1 } } { \widetilde { \lambda } _ { 2 } + 1 } \right) ^ { 2 } \right) \mathrm { d } \theta .\tag{41}
$$

Then, setting $\begin{array} { r } { z = \sqrt { { { \tilde { \lambda } } _ { 2 } } + 1 } \left( { \theta - \frac { { { \tilde { \lambda } } _ { 1 } } } { { { \tilde { \lambda } } _ { 2 } } + 1 } } \right) , \mathrm { d } \theta = \frac { \mathrm { d } z } { \sqrt { { { \tilde { \lambda } } _ { 2 } } + 1 } } \mathrm { \ g i v e s } , } \end{array}$

$$
\int \exp \left( - \frac { 1 } { 2 } ( \widetilde { \lambda } _ { 2 } + 1 ) \left( \theta - \frac { \widetilde { \lambda } _ { 1 } } { \widetilde { \lambda } _ { 2 } + 1 } \right) ^ { 2 } \right) \mathrm { d } \theta = \frac { 1 } { \sqrt { \widetilde { \lambda } _ { 2 } + 1 } } \underset { = \sqrt { 2 \pi } } { \underbrace { \int \exp \left( - \frac { z ^ { 2 } } { 2 } \right) \mathrm { d } z } } .\tag{42}
$$

Substituting that back into Eq. 41 and taking the log gives,

$$
\log \frac { 1 } { \sqrt { \widetilde { \lambda } _ { 2 } + 1 } } \exp \left( \frac { \widetilde { \lambda } _ { 1 } ^ { 2 } } { 2 ( \widetilde { \lambda } _ { 2 } + 1 ) } \right) = - \textstyle \frac { 1 } { 2 } \log ( \widetilde { \lambda } _ { 2 } + 1 ) + \textstyle \frac { 1 } { 2 } \frac { \widetilde { \lambda } _ { 1 } ^ { 2 } } { \widetilde { \lambda } _ { 2 } + 1 } .\tag{43}
$$

This is the result shown in the main paper. We now diferentiate the full objective in Eq. 28 with respect to $\widetilde { \lambda } _ { 1 }$ and $\widetilde { \lambda } _ { 2 }$ and set the resulting derivatives to zero.

$$
m _ { * } - \frac { \widetilde { \lambda } _ { 1 } } { \widetilde { \lambda } _ { 2 } + 1 } = 0 , \qquad - m _ { * } ^ { 2 } - \sigma _ { * } ^ { 2 } + \frac { 1 } { \widetilde { \lambda } _ { 2 } + 1 } + \frac { \widetilde { \lambda } _ { 1 } ^ { 2 } } { ( 1 + \widetilde { \lambda } _ { 2 } ) ^ { 2 } } = 0 .\tag{44}
$$

Solving the first equation yields $\widetilde { \lambda } _ { 1 } = m _ { * } ( \widetilde { \lambda } _ { 2 } + 1 )$ . Inserting that into the equation on the right, we arrive at

$$
\sigma _ { * } ^ { 2 } = \frac { 1 } { \widetilde { \lambda } _ { 2 } + 1 } \quad \Leftrightarrow \quad \widetilde { \lambda } _ { 2 } = \frac { 1 } { \sigma _ { * } ^ { 2 } } - 1 .\tag{45}
$$

Plugging that back into the expression for $\widetilde { \lambda } _ { 1 }$ , we get $\widetilde { \lambda } _ { 1 } = m _ { * } / \sigma _ { * } ^ { 2 }$ . Since $m _ { * } = \sigma _ { * } ^ { 2 } \mathbf { x } ^ { \top } \mathbf { y }$ as per Eq. 12, we have $\widetilde { \lambda } _ { 1 } = \mathbf { x } ^ { \top } \mathbf { y }$ . Since $\sigma _ { * } ^ { 2 } = 1 / ( \mathbf { x } ^ { \top } \mathbf { x } + 1 )$ , we get $\widetilde { \lambda } _ { 2 } = \mathbf { x } ^ { \top } \mathbf { x }$ as claimed.

## A.2 Connection to Convex Dual of Ridge Regression

The primal solution is just the ridge solution which has the following form:

$$
\theta _ { * } = ( \mathbf { x } ^ { \top } \mathbf { x } + 1 ) ^ { - 1 } \mathbf { x } ^ { \top } \mathbf { y }\tag{46}
$$

The dual solution is obtained from the stationarity condition of the dual problem:

$$
\mathbf { y } - ( \mathbf { I } + \mathbf { x } \mathbf { x } ^ { \top } ) \pmb { \alpha } _ { * } = 0 \qquad \implies \qquad \pmb { \alpha } _ { * } = ( \mathbf { x } \mathbf { x } ^ { \top } + \mathbf { I } ) ^ { - 1 } \mathbf { y } .\tag{47}
$$

From here, we can represent θ<sub>∗</sub> in terms of α<sub>∗</sub> by using the matrix inversion lemma,

$$
\theta _ { * } = ( \mathbf { x } ^ { \top } \mathbf { x } + 1 ) ^ { - 1 } \mathbf { x } ^ { \top } \mathbf { y } = \mathbf { x } ^ { \top } ( \mathbf { x } \mathbf { x } ^ { \top } + \mathbf { I } ) ^ { - 1 } \mathbf { y } = \mathbf { x } ^ { \top } \pmb { \alpha } _ { * }\tag{48}
$$

Using this, we can prove Eq. 39 from the stationarity condition of the dual problem:

$$
{ \bf y } - ( { \bf I } + { \bf x } { \bf x } ^ { \top } ) \alpha _ { * } = 0 \quad \implies \quad { \bf y } - \alpha _ { * } - { \bf x } \theta _ { * } = 0 \quad \implies \quad \alpha _ { * } = { \bf y } - { \bf f } _ { * } .\tag{49}
$$

This is equivalent to

$$
\begin{array} { r } { \nabla \log t _ { * } ( \mathbf { f } _ { * } ) = \nabla ( \mathbf { a } _ { * } ^ { \top } \mathbf { f } _ { * } - \frac { 1 } { 2 } \mathbf { f } _ { * } ^ { \top } \mathbf { B } _ { * } \mathbf { f } _ { * } ) = \mathbf { a } _ { * } - \mathbf { B } _ { * } \mathbf { f } _ { * } = \mathbf { y } - \mathbf { f } _ { * } . } \end{array}\tag{50}
$$

Conflict of Interest: One of the authors has co-authored several papers with Dr.   
Frank Nielsen. There are no other obvious conflicts of interest to report at this moment.

## References

Amari, S.: A theory of adaptive pattern classifiers. IEEE Transactions on Electronic Computers (3), 299–307 (1967)

Amari, S.: Theory of self-organizing nerve nets with special reference to association and concept formation. In: Proceedings of the 6th International Joint Conference on Artificial Intelligence (1979)

Amari, S.: Information geometry of the EM and em algorithms for neural networks. Neural Networks 8(9), 1379–1408 (1995)

Amari, S.: The EM algorithm and information geometry in neural network learning. Neural Comput. 7(1), 13–18 (1995)

Amari, S.: Natural gradient works eficiently in learning. Neural computation 10(2), 251–276 (1998)

Amari, S.: Information geometry of neural networks –new Bayesian duality theory–. In: International Conference on Neural Information Processing (ICONIP) (1996)

Wainwright, M.J., Jordan, M.I.: Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning 1–2, 1–305 (2008)

Barndorf-Nielsen, O.: Information and Exponential Families: In Statistical Theory. Wiley, Chichester, England (1978)

Khan, M.E., Rue, H.: The Bayesian learning rule. Journal of Machine Learning Research 24(281), 1–46 (2023)

Gibbs, J.W.: Elementary Principles in Statistical Mechanics. C. Scribner’s Sons, New York (1902)

Jaynes, E.T.: Information Theory and Statistical Mechanics. Physical Review 106(4), 620–630 (1957)

Donsker, M.D., Varadhan, S.S.: Asymptotic evaluation of certain Markov process expectations for large time, I. Communications on Pure and Applied Mathematics 28(1), 1–47 (1975)

Donsker, M.D., Varadhan, S.R.S.: Asymptotic evaluation of certain Markov process expectations for large time, II. Communications on Pure and Applied Mathematics 28(2), 279–301 (1975)

Donsker, M.D., Varadhan, S.R.S.: Asymptotic evaluation of certain Markov process expectations for large time, III. Communications on Pure and Applied Mathematics 29(4), 389–461 (1976)

Donsker, M.D., Varadhan, S.R.S.: Asymptotic evaluation of certain Markov process expectations for large time. IV. Communications on Pure and Applied Mathematics 36(2), 183–212 (1983)

Polyanskiy, Y., Wu, Y.: Information Theory: From Coding to Learning. Cambridge University Press, Cambridge, United Kingdom (2025)

Kimeldorf, G.S., Wahba, G.: A correspondence between Bayesian estimation on stochastic processes and smoothing by splines. Annals of Mathematical Statistics 41(2), 495–502 (1970)

Sch¨olkopf, B., Herbrich, R., Smola, A.J.: A generalized representer theorem. In: Proceedings of the Workshop on Computational Learning Theory, vol. 2111, pp. 416–426 (2001)

Csat´o, L., Opper, M.: Sparse on-line Gaussian processes. Neural computation 14(3), 641–668 (2002)

Altun, Y., Smola, A.: Unifying divergence minimization and statistical inference via convex duality. In: International Conference on Computational Learning Theory (2006)

Dud´ık, M., Phillips, S.J., Schapire, R.E.: Maximum entropy density estimation with generalized regularization and an application to species distribution modeling. Journal of Machine Learning Research 8(44), 1217–1260 (2007)

Zhu, J., Chen, N., Xing, E.P.: Bayesian inference with posterior regularization and applications to infinite latent SVMs. Journal of Machine Learning Research 15(1), 1799–1847 (2014)

Robert, C.P.: The Bayesian Choice: from Decision-theoretic Foundations to Computational Implementation. Springer, New York (2007)

Husain, H., Knoblauch, J.: Adversarial interpretation of Bayesian inference. In: International Conference on Artificial Intelligence and Statistics (2022)

M¨ollenhof, T., Khan, M.E.: SAM as an optimal relaxation of Bayes. In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 (2023). https://openreview.net/forum?id=k4fevFqSQcX

Rockafellar, R.: Duality and stability in extremum problems involving convex functions. Pacific Journal of Mathematics 21(1), 167–187 (1967)

Khan, M.E., Aravkin, A.Y., Friedlander, M.P., Seeger, M.: Fast dual variational inference for non-conjugate latent Gaussian models. In: International Conference on Machine Learning (2013)

Adam, V., Chang, P., Khan, M.E., Solin, A.: Dual parameterization of sparse variational Gaussian processes. In: Advances in Neural Information Processing Systems (2021)

Khan, M.E.: Knowledge adaptation as posterior correction. arXiv:2506.14262 (2025)

M¨ollenhof, T., Swaroop, S., Doshi-Velez, F., Khan, M.E.: Federated ADMM from Bayesian duality. In: International Conference on Learning Representations (2026)