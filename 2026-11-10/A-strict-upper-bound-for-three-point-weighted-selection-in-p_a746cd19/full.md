# A strict upper bound for three-point weighted selection in planar vector-valued regression

Volkov Sergey

November 10, 2026

## Abstract

We prove $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) < 1 1 / 6$ for minimum-Frobenius-norm least-squares regression with two input and two output coordinates, improving the upper bound $1 5 / 8$ . More precisely, a universal $\varepsilon _ { 0 } > 0$ satisfies $\mathcal { T } _ { 3 } \le ( 5 / 6 - \varepsilon _ { 0 } ) E$ for every finite residual frame. The proof combines an exact scalar certificate, rigidity of equality in planar selection, and a compactness argument that controls disappearing weights. Together with the known lower bound, this gives $1 3 / 8 \leq F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) <$ $1 1 / 6$ . No numerical value of $\varepsilon _ { 0 }$ is claimed.

## 1 Problem and result

Weighted data selection for regression was proposed by Hanneke, Moran, Shlimovich and Yehudayof [1, 2]. Zhang [3] established $1 3 / 8 \le F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) \le 1 5 / 8$ and reduced the conjectured value $1 3 / 8$ to a finite planar moment problem. We use Zhang’s residual-frame formulation, interpolation hulls, star localization and seven-atom reduction. An exact scalar certificate, rigidity of equality and compactness yield a strict uniform improvement over all finite datasets. The conjectured value $1 3 / 8$ remains open. For a finite sequence $D = ( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { N }$ in $\mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 2 } , N \geq 1$ , let $\begin{array} { r } { L _ { D } ( W ) = N ^ { - 1 } \sum _ { i } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ i ~ } W x _ { i } - y _ { i } \| ^ { 2 } } \end{array}$ and $L _ { D } ^ { * } = \mathrm { m i n } _ { W } L _ { D } ( W )$ . For nonnegative weights c of sum one, let $\widehat { W } _ { c }$ be the minimum-Frobenius-norm minimizer of $\textstyle \sum _ { i } c _ { i } \| W x _ { i } - y _ { i } \| ^ { 2 }$ . Then $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 )$ is the supremum over datasets of

$$
\frac { \operatorname* { i n f } _ { c _ { i } \geq 0 , \sum _ { i } c _ { i } = 1 , | \mathrm { s u p p } c | \leq 3 } L _ { D } ( \widehat { W } _ { c } ) } { L _ { D } ^ { * } } ,
$$

with $0 / 0 = 1$ and positive-over-zero equal to +∞. We take the infimum because the selected fit can jump when the weighted design loses rank. A limit of full-rank fits bounds the infimum via the continuity of $L _ { D }$ as a function of W.

A residual frame is a finite collection $p _ { i } > 0 , u _ { i } , w _ { i } \in \mathbb { R } ^ { 2 }$ with

$$
\sum _ { i } p _ { i } = 1 , \quad \| u _ { i } \| ^ { 2 } = 2 , \quad \sum _ { i } p _ { i } u _ { i } u _ { i } ^ { \top } = I _ { 2 } , \quad \sum _ { i } p _ { i } w _ { i } u _ { i } ^ { \top } = 0 .\tag{1}
$$

Put $\begin{array} { r } { E = \sum _ { i } p _ { i } \| w _ { i } \| ^ { 2 } } \end{array}$ and $\begin{array} { r } { R = \sum _ { i } p _ { i } w _ { i } w _ { i } ^ { \top } } \end{array}$ . For independent $u _ { i } , u _ { j }$ , let $G _ { i j } u _ { i } = w _ { i } , G _ { i j } u _ { j } = w _ { j }$ . Using Frobenius distance, define

$$
\mathcal { T } _ { 3 } = \operatorname* { m i n } _ { \substack { 2 \leq | S | \leq 3 } } \operatorname { d i s t } \left( 0 , \mathrm { c o n v } \{ G _ { i j } : i , j \in S , \mathrm { ~ d e t } ( u _ { i } , u _ { j } ) \neq 0 \} \right) ^ { 2 } .
$$

Tightness guarantees an independent pair.

Theorem 1. There exists a universal $\varepsilon _ { 0 } > 0$ such that every residual frame satisfies $\mathcal { T } _ { 3 } \leq ( 5 / 6 - \varepsilon _ { 0 } ) E$ Consequently $1 3 / 8 \le F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) \le 1 1 / 6 - \varepsilon _ { 0 } < 1 1 / 6$

## 2 Planar selection and the scalar certificate

Lemma 2. Let $v _ { j } \in \mathbb { R } ^ { 2 }$ have positive probabilities $p _ { j }$ and mean zero. For any real $2 \times 2$ matrix A with $\operatorname { t r } A = 1$ , some convex combination v of at most two of these points satisfies

$$
\| v \| ^ { 2 } \leq \sum _ { j } p _ { j } \| A v _ { j } \| ^ { 2 } .
$$

Proof. Minimize the right-hand linear objective over centered probability vectors on the fixed points, and choose a minimizer q of least support. If the columns $( 1 , v _ { j } )$ on its positive support are dependent, a supported kernel vector has both signs. Small perturbations in either direction remain feasible, so optimality forces zero objective slope. Continuing until one weight vanishes contradicts least support. Hence at most three afinely independent points remain and their objective is no larger than the original. If at most two remain, their centered mean itself is zero.

Otherwise let $\begin{array} { r } { S _ { q } = \sum _ { j } q _ { j } v _ { j } v _ { j } ^ { \top } } \end{array}$ . It is positive definite: a null direction would place all three points on a line. Put $n _ { j } = S _ { q } ^ { - 1 } v _ { j }$ , choose i maximizing $r = \| n _ { i } \| ^ { 2 } > 0$ , and set

$$
\lambda _ { j } = q _ { j } \left( 1 - \frac { n _ { j } ^ { \top } n _ { i } } { r } \right) .
$$

Cauchy–Schwarz and maximality give $\lambda _ { j } ~ \geq ~ 0$ . Centering gives $\textstyle \sum _ { j } \lambda _ { j } = 1$ , and $\lambda _ { i } ~ = ~ 0$ . Since $\begin{array} { r } { \sum _ { j } q _ { j } v _ { j } n _ { j } ^ { \top } = I _ { 2 } } \end{array}$ ,

$$
v = \sum _ { j } \lambda _ { j } v _ { j } = - { \frac { n _ { i } } { r } } , \qquad \| v \| ^ { 2 } = { \frac { 1 } { r } } .
$$

Finally, writing $\begin{array} { r } { C _ { q } = \sum _ { j } q _ { j } \| A v _ { j } \| ^ { 2 } } \end{array}$ ，

$$
1 = \sum _ { j } q _ { j } n _ { j } ^ { \top } A v _ { j } , \qquad 1 \leq \left( \sum _ { j } q _ { j } \| n _ { j } \| ^ { 2 } \right) C _ { q } \leq r C _ { q } .
$$

Thus $\| v \| ^ { 2 } \leq C _ { q } .$ , proving the assertion.

For positive semidefinite S with tr $S > 0$ , the matrix $A = \operatorname { a d j } S /$ tr S has trace one, and in dimension two

$$
\operatorname { t r } ( A S A ^ { \top } ) = { \frac { \operatorname* { d e t } S } { \operatorname { t r } S } } .\tag{2}
$$

This identity holds also when $S$ is singular. If a covariance has trace zero, all its positive-weight points vanish.

Fix an anchor i and a unit vector $n _ { i } \perp u _ { i }$ . Put

$$
\begin{array} { r } { M _ { i } = \frac { 1 } { 2 } w _ { i } u _ { i } ^ { \top } , \quad b _ { i j } = u _ { j } ^ { \top } n _ { i } , \quad c _ { i j } = \frac { 1 } { 2 } u _ { i } ^ { \top } u _ { j } , \quad \xi _ { i j } = \frac { w _ { j } - c _ { i j } w _ { i } } { b _ { i j } } \quad ( b _ { i j } \ne 0 ) . } \end{array}
$$

Because $u _ { j } = c _ { i j } u _ { i } + b _ { i j } n _ { i }$

$$
\begin{array} { r } { G _ { i j } = M _ { i } + \xi _ { i j } n _ { i } ^ { \top } , \qquad \| M _ { i } + \xi n _ { i } ^ { \top } \| _ { F } ^ { 2 } = \frac { 1 } { 2 } \| w _ { i } \| ^ { 2 } + \| \xi \| ^ { 2 } . } \end{array}
$$

The cross term vanishes since $u _ { i } ^ { \top } n _ { i } = 0$ . The positive star weights $P _ { i j } = p _ { j } b _ { i j } ^ { 2 }$ satisfy

$$
\sum _ { b _ { i j } \ne 0 } { P } _ { i j } = 1 , \qquad \sum _ { b _ { i j } \ne 0 } { P } _ { i j } \xi _ { i j } = 0 .
$$

Indeed these are respectively $n _ { i } ^ { \top } I _ { 2 } n _ { i } = 1$ and the diference between the cross-moment $\begin{array} { r } { \sum _ { j } p _ { j } w _ { j } u _ { j } ^ { \top } n _ { i } = 0 } \end{array}$ and $w _ { i } u _ { i } ^ { \top } I _ { 2 } n _ { i } / 2 = 0$

Define

$$
L _ { i } = \sum _ { b _ { i j } = 0 } p _ { j } ( w _ { j } - c _ { i j } w _ { i } ) ( w _ { j } - c _ { i j } w _ { i } ) ^ { \top } \succeq 0 , \qquad H _ { i } = R + \textstyle { \frac { 1 } { 2 } } w _ { i } w _ { i } ^ { \top } .
$$

The star covariance is

$$
S _ { i } = \sum _ { b _ { i j } \neq 0 } P _ { i j } \xi _ { i j } \xi _ { i j } ^ { \top } = H _ { i } - L _ { i } .\tag{3}
$$

To check ${ \mathrm { i t } } ,$ expand the same sum over all $j .$ The two cross terms vanish because $\begin{array} { r } { \sum _ { j } p _ { j } c _ { i j } w _ { j } = 0 } \end{array}$ , and $\textstyle \sum _ { j } p _ { j } c _ { i j } ^ { 2 } = 1 / 2$ . Subtracting the parallel terms yields (3).

Apply Lemma 2 with $A _ { i } = \operatorname { a d j } H _ { i } / \operatorname { t r } H _ { i }$ ; its denominator $E + \| w _ { i } \| ^ { 2 } / 2$ is positive for $E > 0$ . PSD monotonicity and (2) give a two-neighbor convex mean with

$$
\| \xi \| ^ { 2 } \leq \mathrm { t r } ( A _ { i } S _ { i } A _ { i } ^ { \top } ) = { \frac { \operatorname* { d e t } H _ { i } } { \mathrm { t r } H _ { i } } } - \mathrm { t r } ( A _ { i } L _ { i } A _ { i } ^ { \top } ) .
$$

Consequently, for every anchor,

$$
\mathcal { T } _ { 3 } \leq \frac { \Vert w _ { i } \Vert ^ { 2 } } { 2 } + \frac { \operatorname* { d e t } ( R + w _ { i } w _ { i } ^ { \top } / 2 ) } { E + \Vert w _ { i } \Vert ^ { 2 } / 2 } - \mathrm { t r } ( A _ { i } L _ { i } A _ { i } ^ { \top } ) .\tag{4}
$$

This involves at most three atoms. Dropping the last, nonnegative term is valid but potentially loses information.

Normalize $E = 1$ and rotate output coordinates so $R = \mathrm { d i a g } ( r , 1 - r )$ . Set $a _ { i } = w _ { i 1 } ^ { 2 } , b _ { i } = w _ { i 2 } ^ { 2 }$ , and $s = a + b$ . The first two terms of (4) are

$$
g _ { r } ( a , b ) = \frac { s } { 2 } + \frac { r ( 1 - r ) + ( ( 1 - r ) a + r b ) / 2 } { 1 + s / 2 } , \qquad \sum _ { i } p _ { i } a _ { i } = r , \quad \sum _ { i } p _ { i } b _ { i } = 1 - r .
$$

Put $\delta = 2 r - 1 , x = 1 + 3 \delta / 1 0 , y = 1 - 3 \delta / 1 0 , P = x y , Q = 7 / 3 - 2 1 \delta ^ { 2 } / 1 0 0 ,$ , and $h = P - a y - b x$ . Since

$$
\sum _ { i } p _ { i } h ( a _ { i } , b _ { i } ) = \frac { 2 1 } { 1 0 0 } \delta ^ { 2 } \geq 0 ,
$$

some positive-weight atom has $h \geq 0$ . Define

$$
F ( a , b ) = s ^ { 2 } + \frac { 4 } { 3 } s - \delta ( a - b ) - \frac { 7 } { 3 } - \delta ^ { 2 } .
$$

Exact expansion gives

$$
P F = - \frac { 1 2 1 } { 1 0 0 } \delta ^ { 2 } P - h ( Q + a x + b y ) - a b ( x - y ) ^ { 2 } ,\tag{5}
$$

$$
4 ( 1 + s / 2 ) ( g _ { r } ( a , b ) - 5 / 6 ) = F ( a , b ) .\tag{6}
$$

Here $x , y , P , Q > 0$ for $| \delta | \leq 1$ . At the selected atom all terms on the right of (5) are nonpositive. This proves $\tau _ { 3 } \leq 5 / 6$ . Homogeneity gives $\mathcal { T } _ { 3 } \le 5 E / 6$ for general energy.

We also need its quantitative anisotropy consequence. At $h \geq 0 , s \leq P / \operatorname* { m i n } ( x , y ) = \operatorname* { m a x } ( x , y ) \leq$ $1 3 / 1 0 .$ . Thus (5)–(6) give

$$
\mathcal { T } _ { 3 } \leq \frac { 5 } { 6 } - \frac { 1 1 } { 6 0 } \delta ^ { 2 } .\tag{7}
$$

The coeficient is $( 1 2 1 / 1 0 0 ) / ( 4 ( 1 + 1 3 / 2 0 ) ) = 1 1 / 6 0 .$

The scalar relaxation alone is sharp: the one-atom scalar distribution $r = a = b = 1 / 2$ has $g = 5 / 6$ It is not a valid tight frame. The strict improvement must therefore use geometry.

## 3 Equality cannot occur

Lemma 3 (Equality rigidity). Suppose the centered positive distribution has positive definite covariance S, put $C = \operatorname* { d e t } S / \operatorname { t r } S > 0$ and $A = \operatorname { a d j } S / \operatorname { t r } S = C S ^ { - 1 }$ , and let $d ^ { 2 }$ be the minimum squared norm of a convex combination of at most two original points. $I f d ^ { 2 } = C$ , then every original point satisfies

$$
\| A v _ { j } \| ^ { 2 } = C , \qquad \| S ^ { - 1 } v _ { j } \| ^ { 2 } = \operatorname { t r } ( S ^ { - 1 } ) = 1 / C .
$$

Proof. The original A-objective is C by (2). Minimize it as in Lemma 2. Its minimum $C _ { q }$ obeys $d ^ { 2 } \leq C _ { q } \leq C _ { \mathrm { : } }$ , so all equal C. A minimizing least-support distribution must be a centered positive triangle, since otherwise $d = 0$

In its planar proof all inequalities are equalities:

$$
C = d ^ { 2 } \leq \frac { 1 } { r } \leq \frac { 1 } { \sum _ { j } q _ { j } \| S _ { q } ^ { - 1 } v _ { j } \| ^ { 2 } } \leq C _ { q } = C .
$$

Equality in the maximum step forces all three dual norms to equal $r = 1 / C$ . Equality in weighted Cauchy–Schwarz forces $A v _ { j } = C S _ { q } ^ { - 1 } v _ { j }$ on the triangle. Its vertices span $\mathrm { { \mathbb { R } } ^ { 2 } }$ , hence $\begin{array} { r } { A = C S _ { q } ^ { - 1 } , } \end{array}$ , and comparison with $A = C S ^ { - 1 }$ gives $\dot { S } _ { q } = S$ . Each of its three costs $\| A v _ { j } \| ^ { 2 }$ therefore equals C.

For an original point $v _ { k }$ not in the selected support, write its afine coordinates in the triangle (which need not be nonnegative): $\begin{array} { r } { v _ { k } = \sum _ { j = 1 } ^ { 3 } \beta _ { j } v _ { j } , \sum _ { j } \beta _ { j } = 1 } \end{array}$ . For suficiently small $t > 0 ,$ put weight t on $v _ { k }$ and change triangle weights to $q _ { j } - t \beta _ { j } \geq 0 .$ . This is still a centered probability vector. Its cost is $C + t ( \| A \bar { v } _ { k } \| ^ { 2 } - \bar { C } )$ . Minimality forces $\| A v _ { k } \| ^ { 2 } \geq C$ . Every original weight is positive and their costs average $C _ { i }$ , so every cost is exactly C. The second identity follows from $A = C S ^ { - 1 }$ and tr $S ^ { - 1 } = \operatorname { t r } S /$ det S. □

Lemma 4. A tight planar frame contains two indices with $c ^ { 2 } = ( u _ { i } ^ { \top } u _ { j } / 2 ) ^ { 2 } \leq 1 / 4$

Proof. Write $u _ { i } / \sqrt { 2 } = \left( \cos { \theta _ { i } } , \sin { \theta _ { i } } \right)$ . Tightness implies that the unit-circle points $t _ { i } = ( \cos 2 \theta _ { i } , \sin 2 \theta _ { i } )$ have weighted mean zero. Reduce a convex representation of zero to at most three points. If two sufice, they are antipodal and the corresponding $c ^ { 2 } = 0$ . Otherwise their triangle contains zero in its interior; its three cyclic angular gaps are at most π and sum $2 \pi$ . Some gap lies in $[ 2 \pi / 3 , \pi ]$ , so that pair has $c ^ { 2 } = ( 1 + \cos ( 2 \theta _ { i } - 2 \theta _ { j } ) ) / 2 \leq 1 / 4$ . A boundary triangle likewise has an antipodal pair. □

Proposition 5. Every finite positive-energy residual frame satisfies $\mathcal { T } _ { 3 } < 5 E / 6$

Proof. Normalize $E = 1$ and suppose $\mathcal { T } _ { 3 } = 5 / 6$ . Equation (7) forces $R = I _ { 2 } / 2$ . In this case the dropped-correction anchor expression is

$$
f ( s ) = s / 2 + ( 1 + s ) / ( 4 + 2 s ) , \qquad f ( 1 ) = 5 / 6 , \quad f ^ { \prime } ( s ) = 1 / 2 + 1 / ( 2 ( s + 2 ) ^ { 2 } ) > 0 .
$$

Since $\mathcal { T } _ { 3 } \leq f ( \| w _ { i } \| ^ { 2 } )$ for every i, all $\| w _ { i } \| ^ { 2 } \geq 1$ . Their weighted mean is one and weights are positive, hence all equal one.

For every anchor, $H _ { i } = I _ { 2 } / 2 + w _ { i } w _ { i } ^ { \top } / 2$ has eigenvalues $1 , 1 / 2 ,$ so $A _ { i }$ is invertible and det $H _ { i } /$ tr $H _ { i } =$ $1 / 3$ . Equation (4) forces $\mathrm { t r } ( A _ { i } L _ { i } A _ { i } ^ { \top } ) = 0$ , hence $L _ { i } = 0$ and $S _ { i } = H _ { i }$ . The minimum squared norm of a two-point convex mean in this star must equal $1 / 3 { : }$ Lemma 2 bounds it above by $1 / 3$ , while any smaller value would produce a certificate below $5 / 6$ . Lemma 3 therefore forces, for every star neighbor,

$$
\begin{array} { r } { \| H _ { i } ^ { - 1 } \xi _ { i j } \| ^ { 2 } = \mathrm { t r } ( H _ { i } ^ { - 1 } ) = 3 . } \end{array}\tag{8}
$$

Choose $i , j$ from Lemma $^ { 4 , }$ and set $c = u _ { i } ^ { \top } u _ { j } / 2 , d = w _ { i } ^ { \top } w _ { j }$ . This is an independent pair. In output coordinates with first axis $w _ { i }$ , the two components of $w _ { j } - c w _ { i }$ are $d - c$ and a number of squared magnitude $1 - d ^ { 2 }$ . Also $b _ { i j } ^ { 2 } = 2 ( 1 - c ^ { 2 } )$ , and $H _ { i } ^ { - 1 }$ has eigenvalues 1, 2. Thus (8) is equivalent to

$$
{ \frac { ( d - c ) ^ { 2 } + 4 ( 1 - d ^ { 2 } ) } { 2 ( 1 - c ^ { 2 } ) } } = 3 , \qquad 7 c ^ { 2 } - 2 c d - 3 d ^ { 2 } = 2 .
$$

But exact square completion gives

$$
7 c ^ { 2 } - 2 c d - 3 d ^ { 2 } = { \frac { 2 2 } { 3 } } c ^ { 2 } - 3 ( d + c / 3 ) ^ { 2 } \leq { \frac { 1 1 } { 6 } } < 2 ,
$$

a contradiction. This proves pointwise strictness.

## 4 A uniform gap

Lemma 6. For each positive-energy frame there is a residual frame on a subset of at most seven of its fixed atoms, with energy $E ^ { \prime } \leq E$ , and with $\mathcal { T } _ { 3 } ^ { \prime } \geq \mathcal { T } _ { 3 } . ~ I f E ^ { \prime } = 0 _ { ! }$ then $\mathcal { T } _ { 3 } = 0$

Proof. We recall Zhang’s support reduction [3]. Minimize $\textstyle \sum _ { i } q _ { i } \| w _ { i } \| ^ { 2 }$ over nonnegative weights on the fixed atoms satisfying the frame constraints. The symmetric input covariance gives three linear equations and the cross-covariance gives four. Normalization follows by taking the trace of the input equation, since $\| u _ { i } \| ^ { 2 } = 2 .$ . Thus there are at most seven independent linear constraints. The feasible probability polytope is compact and nonempty. At a minimizing weight vector of least support, a dependence of its active constraint columns would permit both signs of small feasible perturbation. The energy slope must be zero by optimality. Continuing to the first zero weight would reduce support while preserving energy, a contradiction. The positive support consequently has at most seven atoms. The pair interpolators do not depend on the frame weights. Deleting atoms only removes candidates from the selection minimum, giving $\mathcal { T } _ { 3 } ^ { \prime } \geq \mathcal { T } _ { 3 }$ . If $E ^ { \prime } = 0$ , all remaining w vanish; tightness still supplies an independent pair and its interpolator is zero. Then $\mathcal { T } _ { 3 } ^ { \prime } = 0$ , so $\mathcal { T } _ { 3 } = 0$ □

Proof of the uniform frame assertion in Theorem 1. Let $\Gamma = \operatorname* { s u p } \mathcal { T } _ { 3 } / E$ over all finite positive-energy frames. We know $\Gamma \leq 5 / 6$ . Suppose $\Gamma = 5 / 6$ . For each k choose a frame with positive ratio at least $5 / 6 - \epsilon _ { k }$ , where $\epsilon _ { k } \downarrow 0$ . Lemma 6 gives a reduced frame with $E ^ { \prime } > 0$ and ${ \mathcal { T } } _ { 3 } ^ { \prime } / E ^ { \prime } \geq { \mathcal { T } } _ { 3 } / E$ . Scale the reduced $w _ { i }$ by $1 / \sqrt { E ^ { \prime } }$ . The resulting frames have energy one and $\mathcal { T } _ { 3 } ^ { ( k ) } \geq 5 / 6 - \epsilon _ { k }$ . There are at most seven atoms. Relabel and pass to a subsequence of constant support size, and diagonalize $R _ { k } = \mathrm { d i a g } ( r _ { k } , 1 - r _ { k } )$ by output rotations, which preserve all frame identities and costs.

$$
| \delta _ { k } | = | 2 r _ { k } - 1 | \le \sqrt { 6 0 \epsilon _ { k } / 1 1 } \longrightarrow 0 .
$$

For an arbitrary atom put $s _ { i } ^ { ( k ) } = \| w _ { i } ^ { ( k ) } \| ^ { 2 }$ . The normalized anchor expression satisfies

$$
g _ { r _ { k } } ( a _ { i } , b _ { i } ) \leq f ( s _ { i } ^ { ( k ) } ) + | \delta _ { k } | / 2 .
$$

This follows from $r _ { k } ( 1 - r _ { k } ) \leq 1 / 4$ and $( ( 1 - r _ { k } ) a _ { i } + r _ { k } b _ { i } ) / 2 \leq ( 1 + | \delta _ { k } | ) s _ { i } ^ { ( k ) } / 4 . \mathrm { ~ A s ~ } \mathcal { T } _ { 3 } ^ { ( k ) } \leq g _ { r _ { k } } ( a _ { i } , b _ { i } )$ for every atom and $f ^ { \prime } \geq 1 / 2$

$$
s _ { i } ^ { ( k ) } \geq 1 - \eta _ { k } , \qquad \eta _ { k } = 2 \epsilon _ { k } + | \delta _ { k } | \longrightarrow 0 .
$$

Every summand below is nonnegative, and

$$
\sum _ { i } p _ { i } ^ { ( k ) } ( s _ { i } ^ { ( k ) } - 1 + \eta _ { k } ) = \eta _ { k } .\tag{9}
$$

For every index set $B ,$ summing its nonnegative terms gives

$$
\sum _ { i \in B } p _ { i } ^ { ( k ) } s _ { i } ^ { ( k ) } \leq ( 1 - \eta _ { k } ) \sum _ { i \in B } p _ { i } ^ { ( k ) } + \eta _ { k } \leq \sum _ { i \in B } p _ { i } ^ { ( k ) } + \eta _ { k } .
$$

Thus atoms with vanishing probability cannot retain positive energy.

Pass to subsequences with $p _ { i } ^ { ( k ) } \to p _ { i }$ and $u _ { i } ^ { ( k ) } \to u _ { i } ;$ probabilities and input directions lie in compact sets. For $J = \{ i : p _ { i } > 0 \}$ , the bounds $0 \leq s _ { i } ^ { ( k ) } - 1 + \eta _ { k } \leq \eta _ { k } / p _ { i } ^ { ( k ) }$ imply $s _ { i } ^ { ( k ) } \to 1$ . Pass to a further subsequence with $w _ { i } ^ { ( k ) }  w _ { i }$ for $i \in J$ . For $i \not \in J ,$ the input covariance contribution has norm $2 p _ { i } ^ { ( k ) }$ , the output covariance contribution has norm $p _ { i } ^ { ( k ) } s _ { i } ^ { ( k ) } \leq p _ { i } ^ { ( k ) } + \eta _ { k }$ , and the cross-covariance contribution has norm $\sqrt { 2 p _ { i } ^ { ( k ) } } \sqrt { p _ { i } ^ { ( k ) } s _ { i } ^ { ( k ) } }$ . All three vanish. The surviving positive-weight atoms thus satisfy all frame identities, have unit output norm, total energy one, and covariance $R = I _ { 2 } / 2$

By Proposition 5, this limiting frame has $T _ { 3 } ^ { ( \infty ) } < 5 / 6$ . The minimum is attained because there are finitely many nonempty compact hulls. Fix a minimizing convex combination of pair interpolators on at most three surviving indices. Every pair used has a nonzero limiting determinant; the same pairs are independent for all suficiently large k, and their interpolators converge. Keeping its convex coeficients gives

$$
\operatorname* { l i m } _ { k } \operatorname* { s u p } _ { \bf \mathcal { T } _ { 3 } ^ { ( k ) } } \le \mathcal { T } _ { 3 } ^ { ( \infty ) } < 5 / 6 ,
$$

contradicting $T _ { 3 } ^ { ( k ) } \to 5 / 6$ . Consequently $\Gamma < 5 / 6$ . Set $\varepsilon _ { 0 } = 5 / 6 - \Gamma > 0$ . This proves the uniform bound for positive energy. If $E = 0$ , all $w _ { i }$ vanish and $\mathcal { T } _ { 3 } = 0$ , so the same bound holds. No numerical lower bound on $\varepsilon _ { \mathrm { 0 } }$ is used. □

## 5 Return to regression

Let $W ^ { \circ }$ be the full-data minimum-norm minimizer and $\rho _ { i } = W ^ { \circ } x _ { i } - y _ { i }$ . The normal equations imply $\begin{array} { r } { \sum _ { i } \rho _ { i } x _ { i } ^ { \top } = 0 } \end{array}$ , and for every B,

$$
L _ { D } ( W ^ { \circ } + B ) = L _ { D } ^ { * } + N ^ { - 1 } \sum _ { i } \| B x _ { i } \| ^ { 2 } .
$$

Suppose $\begin{array} { r } { K = \sum _ { i } x _ { i } x _ { i } ^ { \top } \succ 0 } \end{array}$ . Write $v _ { i } = K ^ { - 1 / 2 } x _ { i }$ and $\ell _ { i } = \| v _ { i } \| ^ { 2 }$ . For $\ell _ { i } > 0$ , set

$$
p _ { i } = \ell _ { i } / 2 , \qquad u _ { i } = \sqrt { 2 / \ell _ { i } } v _ { i } , \qquad w _ { i } = \sqrt { 2 / \ell _ { i } } \rho _ { i } .
$$

The omitted indices have $x _ { i } = 0$ and contribute only unavoidable loss. The remaining atoms satisfy (1), with $\begin{array} { r } { E = \sum _ { \ell _ { i } > 0 } \| \rho _ { i } \| ^ { 2 } \leq N L _ { D } ^ { * } } \end{array}$ . For $\Theta = ( W - W ^ { \circ } ) K ^ { 1 / 2 }$ , the excess loss is $\lVert \Theta \rVert _ { F } ^ { 2 } / \bar { N }$

Given frame weights $q ,$ choose original weights proportional to $q _ { i } / \ell _ { i }$ . The selected objective, in Θ coordinates, is a positive multiple of $\begin{array} { r } { \sum _ { i } q _ { i } \| \Theta u _ { i } + w _ { i } \| ^ { 2 } . \mathrm { ~ I f ~ } U ( q ) = \sum _ { i } q _ { i } u _ { i } u _ { i } ^ { \top } \succ 0 , } \end{array}$ , its unique minimizer is $\Theta = - G ( q )$ , where

$$
G ( q ) = \left( \sum _ { i } q _ { i } w _ { i } u _ { i } ^ { \top } \right) U ( q ) ^ { - 1 } = \sum _ { i < j : D _ { i j } > 0 } \frac { q _ { i } q _ { j } D _ { i j } } { \sum _ { a < b } q _ { a } q _ { b } D _ { a b } } G _ { i j } , \qquad D _ { i j } = \operatorname * { d e t } ( u _ { i } , u _ { j } ) ^ { 2 } .\tag{10}
$$

The determinant identity follows from Cauchy–Binet. By linearity of the adjugate in dimension two, the coeficient of $q _ { i } q _ { j }$ in the numerator is $M _ { i j } \operatorname { a d j } U _ { i j }$ , where $U _ { i j } = u _ { i } u _ { i } ^ { \top } + u _ { j } u _ { j } ^ { \top }$ and $\begin{array} { r } { M _ { i j } = w _ { i } u _ { i } ^ { \top } + w _ { j } u _ { j } ^ { \top } } \end{array}$

For an independent pair, $M _ { i j } = G _ { i j } U _ { i j }$ , giving $D _ { i j } G _ { i j }$ . For a dependent pair and for diagonal terms, the adjugate annihilates the common input span.

For each eligible $S$ in the definition of $\tau _ { 3 }$ , its hull is the closure of the values $G ( q )$ with supp $q \subseteq S$ and $U ( q ) \succ 0$ (cf. Zhang [3]). Equation (10) gives one inclusion. An independent pair realizes its single interpolator. For three pairwise independent directions and positive coeficients $\beta _ { i j } , \beta _ { i k } , \beta _ { j k }$ of sum one, choose

$$
q _ { i } : q _ { j } : q _ { k } = D _ { j k } / \beta _ { j k } : D _ { i k } / \beta _ { i k } : D _ { i j } / \beta _ { i j } .
$$

Substitution in (10) gives the prescribed coeficients. Approximate any boundary coeficient vector by positive ones. If two directions are parallel, both valid pairs share an anchor i. For a convex combination of its valid neighbors with coeficients $\alpha _ { j }$ , take

$$
q _ { i } = 1 - t , \qquad q _ { j } = t \frac { \alpha _ { j } / D _ { i j } } { \sum _ { k } \alpha _ { k } / D _ { i k } } , \qquad 0 < t < 1 .
$$

The leaf pair has zero determinant, so (10) gives $\begin{array} { r } { G ( q ) = \sum _ { j } \alpha _ { j } G _ { i j } } \end{array}$ for every $0 < t < 1$ . All realizing and approximating fits have full rank and hence a unique minimizer in the original coordinates. Consequently

$$
\operatorname* { i n f } _ { | \operatorname { s u p p } c | \leq 3 } L _ { D } ( \widehat { W } _ { c } ) \leq L _ { D } ^ { * } + \mathcal { T } _ { 3 } / N \leq ( 1 1 / 6 - \varepsilon _ { 0 } ) L _ { D } ^ { * } .
$$

If the feature span has dimension one, let e be a unit spanning vector and write $x _ { i } = a _ { i } e$ and $W ^ { \circ } = \beta e ^ { \top }$ . The normal equation $\textstyle \sum _ { i } a _ { i } \rho _ { i } = 0$ centers the vectors $\rho _ { i } / a _ { i }$ on nonzero $a _ { i }$ with probabilities $a _ { i } ^ { 2 } / \textstyle \sum _ { j } a _ { j } ^ { 2 }$ . By planar Carath´eodory, choose a convex representation of zero using at most three of them. If its coeficients are $\lambda _ { i }$ , take $c _ { i } \propto \lambda _ { i } / a _ { i } ^ { 2 }$ . Then $\textstyle \sum _ { i } c _ { i } a _ { i } \rho _ { i } = 0$ and $\textstyle \sum _ { i } c _ { i } a _ { i } ^ { 2 } > 0$ , so the selected fit recovers $\beta$ on $e .$ . Minimum norm sets its action on $e ^ { \perp }$ to zero. If the feature span has dimension zero, choose any one observation. All losses are independent of W in that case. In the full-rank branch, $E = 0$ means that every nonzero-feature residual vanishes, and a basis of two feature vectors recovers $W ^ { \circ }$ . This covers $L _ { D } ^ { * } = 0$ in that branch, since $E \le N L _ { D } ^ { * }$

For completeness, Zhang’s axial example [3] gives the lower bound. Take $( e _ { i } , a + r _ { j } )$ for the two coordinate inputs $e _ { i }$ and the three vertices $r _ { j }$ of a centered equilateral triangle with $\| r _ { j } \| ^ { 2 } = 2 / 3$ Choose $\| a \| ^ { 2 } = 5 / 6$ . The full-data fit has both columns equal to a and $L _ { D } ^ { * } = 2 / 3$ . With $k = 0 , 1$ 1, 2, 3 selected points on one axis, the best squared column deviation from a is respectively $5 / 6 , 2 / 3 , 1 / 6 , 0$ For $k = 0$ this uses minimum-norm tie breaking, and for $k > 0$ it follows by projecting onto the convex hull of the chosen vertices after subtracting a. The excess loss is half the sum of the two squared column deviations. The best allocations at budget three are (0, 3) and (1, 2) up to order, both giving excess $5 / 1 2$ and ratio $1 + ( 5 / 1 2 ) / ( 2 / 3 ) = 1 3 / 8$

## 6 Formal verification and code

The companion source at https://github.com/nZiben/weighted-selection-gap formalizes scalar moment selection under the two moment constraints, its anisotropy bound (7), scalar sharpness, and the algebraic obstruction in Proposition 5. The planar lemma, equality rigidity, support reduction, compactness and regression bridge are proved in the text. The repository includes reproduction commands and the compiler record for Lean 4.34.0 and pinned Mathlib. The audited declarations use only propext, Classical.choice and Quot.sound.

## Use of AI

The proof was produced by GPT-6 Pro (GPT-6 Astra) in the ChatGPT user interface.

## References

[1] S. Hanneke, S. Moran, A. Shlimovich and A. Yehudayof. Open Problem: Data Selection for Regression Tasks. COLT 2025, PMLR 291, pp. 6225–6229. https://proceedings.mlr.press/ v291/hanneke25e.html.

[2] S. Hanneke, S. Moran, A. Shlimovich and A. Yehudayof. Data Selection for ERMs. 2025. https: //arxiv.org/abs/2504.14572.

[3] G. Zhang. Exact Recovery Thresholds for Weighted Data Selection in Vector-Valued Linear Regression. arXiv:2608.30254, version 2, 2026. https://arxiv.org/abs/2608.30254v2.