# REMEMBERING SOLOMON MARCUS

FLORIN FELIX NICHITA

Abstract. From the Andre Breton’s manifest of surrealism, through the transdisciplinary understanding, we arrive at “A post-modern manifest” (Sci 2022). A talk by Laura De Marco (Harvard) will provide scientific background to approach an AMS poetry. The next section will be a qualitative analysis of some new operations on the real numbers. The conclusions will be given in the last section, and an appendix will recall some previous work with some new comments.

## 1. Introduction

Solomon Marcus extended his activities to many domains. Meeting my university colleagues and me, he remarked the Ph.D. Thesis of Nicusor Dan, the achievements of Marian Aprodu, the contributions of Gabriel Istrate in Computer Science (although he graduated as a pure mathematician), the carrier of Ovidiu Calin (a Fulbright Fellow in Romania), the Euler type formulas found by myself, etc.

Our papers on meetings with Solomon Marcus, which might become a book, received excellent feedback from many readers. We present an aftermath of that series. Trying to recreate the spirit of those meetings, we involuntarily gathered an entire “choir” on many “voices”: Horatiu Malaele, Mircea Cartarescu, Sanda Golopentia, Rodica Zafiu, Gabriel Prajitura (the literature voices), Gheorghe Gussi, Ovidiu Pasarescu, Marius Buliga, Remus Radu, Ovidiu Calin, Radu Iordanescu (the math choir), Violeta Dinescu, Nick Mihalache, Dan Vuza, Aurelian Gheondea, Ion Mihail Nichita, Iosif Imbrisca (the “music department”) and the Math Cafe Team (Naomi, David, Razvan and Rarea, among others). From Marius Buliga, I have learned that the mathematics archive $\left( \mathrm { ^ { 6 6 } a r X i v . o r g ^ { 3 7 } } \right)$ posts many papers of Leonard Euler in English (in relation with the paper [1]). Also, I found at “Project Euler” the following sum: $\dot { \sum } _ { 1 } ^ { 1 0 } k ^ { k }$ (this would be related to [2]).

Just as the Andre Breton’s manifest of surrealism led to masterpieces of artwork combining the real life with the dreams, in mathematics, the real numbers and the imaginary numbers live in the field of complex numbers. Combining the arithmetic with the world of imaginary numbers, the Euler identity, $e ^ { i \pi } + 1 = 0$ , is an example of “surrealistic” masterpiece in mathematics. The Euler formula and the Euler identity were our favourite topics in our meetings with Solomon Marcus. These led to a publication in Axioms ([1]).

The “Manifesto of Transdisciplinarity” of Basarab Nicolescu, encouraged me for the involvement in many disciplines. Solomon Marcus could be considered an example in this regard. The poem “A post-modern manifest” (Sci, 2022) invites the poets to explore the mathematical universe. Now, I propose a short intermezzo in our discourse: Laura De Marco, from Harvard University, gave a talk on the Mandelbrot set at the Romanian Academy in June 2026. This vibrant presentation attracted the attention of my colleagues. Four collaborators of Laura De Marco are Romanians. And a final comment for this intermezzo: Mandelbrot was a polymath. Back to our main discourse. The AMS poetry “Pantoum for the Mandelbrot set” won the second prize in 2026. It is both full of lyricism and of scientific information.

The next part of the paper is an analysis of operations defined on the set of real numbers. The conclusions will be given in the last section, and an appendix will recall some previous work and some possible continuations.

## 2. Mathematical background

2.1. New logarithms and radicals. We will define an “universal logarithm”, a kind of logarithm for which there is no need to give a fixed basis. Thus, we denote by $L _ { 2 }$ the universal logaritm of order two: given a number n, the universal logarithm associates the bigest number m, such that $n = m ^ { m }$ . For example: $L _ { 2 } ( 4 ) = 2$ and $L _ { 2 } ( 2 7 ) = 3$

The universal logarith of order three has the following values:

$$
L _ { 3 } ( 1 6 ) = L _ { 3 } ( 2 ^ { 2 ^ { 2 } } ) = 2 , L _ { 3 } ( 1 ) = L _ { 3 } ( 1 ^ { 1 ^ { 1 } } ) = 2 , L _ { 3 } ( 1 0 ^ { 1 0 ^ { 1 0 } } ) = 1 0 ,
$$

We also define a generalized radical: it takes the radical of the exponent.

$R _ { 2 } \ ( x ) = { \sqrt { 2 } } { \sqrt { \log _ { { \sqrt { 2 } } } x } }$ . For example, $R _ { 2 } \left( 4 \right) = \sqrt { 2 } ^ { \sqrt { \log _ { \sqrt { 2 } } 4 } } = \sqrt { 2 } ^ { \sqrt { 4 } } = \sqrt { 2 } ^ { 2 } = 2 .$

In general, $L _ { 2 } ( x ) ~ \le ~ R _ { 2 } ~ ( x )$

2.2. New operations. The following operation could be considered an operation which (approximately) generates the addition of positive real numbers:

$a \circ b = \log _ { \sqrt { 2 } } \ \sqrt { 2 } \ ^ { a } \ + \ \sqrt { 2 } \ ^ { b }$ . For example, a ◦ $a \circ a \circ a = a + 4$

The following operation also generates the addition of positive real numbers:

$a \oplus a = a + 2 ~ ; ~ a \oplus b = \operatorname* { m a x } ( a , b ) + 1 ~ , ~ a \neq b$ . For example, $a \circ a \circ a = a + 3$

We now define a symmetric exponentiation:

$$
{ \widehat { a ^ { b } } } = { \widehat { b ^ { a } } } = { \sqrt { 2 } } ^ { ( \log _ { { \sqrt { 2 } } } a ) \times ( \log _ { { \sqrt { 2 } } } b ) } = a ^ { \log _ { { \sqrt { 2 } } } b } = b ^ { \log _ { { \sqrt { 2 } } } a } .
$$

For example, $\widehat { 2 ^ { 2 } } = 4 \ , \ \widehat { 2 } \sqrt { ^ { 2 } } = 2 , \widehat { 2 ^ { 1 } } = 1 \mathrm { a n d } \widehat { 2 ^ { 4 } } = 1 6 .$

2.3. Mean type inequalities. For $a , b \geq 4 .$ , we have the following inequalities:

$$
L _ { 2 } ( a ^ { b } ) \leq \sqrt { a b } \leq \frac { a + b } { 2 } \leq a \circ b - 2
$$

For $a , b \geq 1$ , natural numbers, we have the following inequalities:

$$
R _ { 2 } ( \widehat { a ^ { b } } ) \leq \sqrt { a b } \leq \frac { a + b } { 2 } \leq a \oplus b - \frac { 3 } { 2 }
$$

2.4. Euler formula. There exists an Euler formula for the set of real numbers endowed with the operations ⊕ and + . A new element −∞ will be added to the real numbers, and it is a neutral elemnent for ⊕.

The “exponential function” $f ( t ) = \sqrt { 2 } ^ { ~ t } , ~ \sqrt { 2 } ^ { ~ s \oplus t } = \sqrt { 2 } ^ { ~ s } + \sqrt { 2 } ^ { ~ t }$ , can be extended to $\mathbb { R } \times \mathbb { R }$ as follows: $\sqrt { 2 } \ ^ { ( - \infty , t ) } \ = \ \left[ \ \log _ { 2 } \ ( \cos ^ { 2 } \sqrt { 2 } ^ { t } ) \ , \ \log _ { 2 } \ ( \sin ^ { 2 } \sqrt { 2 } ^ { t } ) \ \right]$

## 3. Commentaries and conclusions

Some traces of surrealism could be found in (the pictures of) the famous books of Lewis Carroll. Forms of contemporary surrealism are present in the magical realism (of Gabriel Garcia Marquez, Jorge Luis Borges, etc) and in the works of the street artist Banksy (probably an acronym for the “Bristol underground scene”). A dual “art” has emerged: finding Banksy’s works across the globe, admiring them, capturing them, transporting them and exposing them in oficial exhibitions. (The last three steps are not fully accepted by Banksy and his admires.) Other contemporary artists created Dog Man, present the aboriginal culture or use special organic pigments in their “surrealistic art” (Bojaxhiu, Yuguo, etc).

## References

[1] Solomon Marcus and Florin Nichita, On transcendental numbers: new results and a little history, Axioms, 2018, 7, 15.

[2] Nichita, F., Memories with Solomon Marcus (III), arXiv: 2604.15970.

[3] Nichita, F., Memories with Solomon Marcus, Libertas Matematica (new series), Volume 45 (2025), No. 1, 103-108.

[4] Nichita, F., Memories with Solomon Marcus (II), in press.

[5] Golopentia, S. The conferences given at Brown University by Solomon Marcus in the years 2008 and 2011 (in Romanian), Sept. 2025.

[6] Golopentia, S. When I think at Solomon Marcus (in Romanian), Dec. 2025.

[7] Golopentia, S. Reactions to some texts about the Romanian language of the Acad. Solomon Marcus (in Romanian), August 2013.

[8] Golopentia, S. Solomon Marcus or about the serious joys (in Romanian), Jan. 2017.

[9] B.K. Winter, A.T. Lipnicki, A braid box, arXiv: 2604.20884.

[10] Z.P. Bradshaw and C. Vignat, Learning from Ramanujan: Elementary Approach to Profound Ideas. arXiv: 2605.08484.

[11] Marcus, S. Transcendence, as a universal paradigm (in Romanian), Convorbiri Literare 2014, February, 17-28.

[12] Marcus, S. Transcendence, as a universal paradigm, BALANCE, A Club of Rome Magazine 2015, no.1, March, 50–70.

[13] Petrie, B.J. Leonhard Euler’s use and understanding of mathematical transcendence, Historia Mathematica 2012, 39, 280-291.

[14] Nicolescu, B. Transdisciplinarity - past, present and future, in Moving Worldviews - Reshaping sciences, policies and practices for endogenous sustainable development 2006, COMPAS Editions, Holland, edited by Bertus Haverkort and Coen Reijntjes, 142-166.

[15] Nicolescu, B. The unexplected way to holiness: Simone Weil (II) (in Romanian), Convorbiri Literare 2017, October, No. 10(262),26–28.

[16] Nichita, F.F. On Models for Transdisciplinarity, Transdisciplinary Journal of Engineering and Science 2011, Vol. 2011, 42-46.

[17] B.E. Tonn and C.J. Hill Capturing the expanding research areas of the future of humanity within the field of future studies: The case for a transcendental future subdiscipline, Futures 174 (2025) 103692.

[18] Racks and Quandles, Wikipedia, 2026.

[19] T. Brzezi´nski and F. F. Nichita, Yang–Baxter Systems and Entwining Structures, Comm. Algebra 33 (2005) 1083-1093.

[20] Terence Tao, Mathematics in the age of AI, 2608.16753.

[21] Marta Petreu, Filozofia lui Blaga, Polirom, 2024.

[22] Calin Vlasie, Viitorul literaturii. Spre o noua poetica a creatiei in era post-algoritmica (I), Ramuri, 8 / 2026.

## 4. APPENDIX – Previous work revisited

I received an invitation from the journal “Libertas Matematica (new series)” to write an article about Solomon Marcus on the occasion of one hundred years from his birth. I submitted my manuscript ([3]), and it was accepted soon afterwards. This determined me to write a second paper dedicated to Solomon Marcus ([4]) . Also, I posted a survey of [3] and [4] on the Mathematical Archive ([2]), adding some new content. (For example, I gave applications of the B-ring Euler formula in finding solutions for the braid condition.) From Brown University (Rhode Island), I received four articles about Solomon Marcus (see [5, 6, 7, 8]). Also, I would like to mention another Mathematical Archive communication ([9]), which was posted soon after our preprint ([2]), and it gives some weight to our results, because the solutions to the braid equation lead to representations of the braid group. More recently, Bradshaw and Vignat wrote a paper, in a similar manner with ours, about another “beautiful mind” (see [10]). Also, Terence Tao has posted an essay on Mathematics and AI, reminding me of a book by Marta Petru and a series of articles by Calin Vlasie (see [20, 21, 22]). The International Poetry Festival in Bucharest (FIPB 2026) is a sourse of wonderful discussions on the topics of our paper.

## 4.1. Trigonometry in racks.

Definition 4.1. A rack is triple $( S , \cdot , \circ )$ , where S is a set with two binary operations, satisfying the following four axioms:

a · (b · c) = (a · b) · (a · c) , (a · b) ⋄ a = b ,   
a · (b ⋄ a) = b , (c ⋄ b) ⋄ a = (c ⋄ a) ⋄ (b ⋄ a) .

The operation · is called the main operation, we will write $a \cdot b = a b$ , and it has priority over ⋄ in formulas.

Remark 4.2. The following is an example of rack associated to a group:

ab = aba<sup>−1</sup>, a ⋄ b = b<sup>−1</sup>ab.

We now choose e, O ∈ S, and let Π = eO and $U = e ( e O )$   
Let cos x = ex , sin x = x ⋄ e be “trigonometric” functions in our rack.   
The following properties hold for our “trigonometric” functions:   
cos Π = U ; sin Π = O ;   
cos xy = cos x cos y ; cos(x ⋄ y) = cos x ⋄ cos y ;   
sin xy = sin x sin y ; sin(x ⋄ y) = sin x ⋄ sin y .   
The fundamental formula for this “trigonometry” is the following:   
sin (cos x) = cos(sin x) = x.

Remark 4.3. The following rack can be defined now: $a b = \cos b , \quad a \diamond b = \sin a .$

## 4.2. Weak racks.

Definition 4.4. A weak rack is triple $( S , \ \cdot , \ \diamond )$ , where S is a set with two binary operations, satisfying the following three axioms:

a · (b · c) = (a · b) · (a · c) ,   
(a · b) ⋄ a = a · (b ⋄ a) ,   
(c ⋄ b) ⋄ a = (c ⋄ a) ⋄ (b ⋄ a) .

Again, the for the main operation we will write $a \cdot b = a b$ , and it has priority over ⋄ in formulas.

Remark 4.5. The following weak rack can be associated to a Boolean algebra:

$$
a b = a \to b , a \diamond b = a \setminus b ~ .
$$

Also, the following weak rack can be associated: $a b = a \vee b , ~ a \diamond b = a \wedge b .$

Let $e , \ O \in S , \ \Pi = e O , \ U = e ( e O )$ , cos $x = e x$ , and sin $x = x \ \diamond \ e$ . We have the following properties: cos Π = U ; sin $\Pi = O$ ; cos xy = cos x cos y ;

cos(x ⋄ y) = cos x ⋄ cos y ; sin xy = sin x sin y ; $\sin ( x \diamond y ) =$ sin x ⋄ sin $y$ .

The fundamental formula is the following: sin $( \cos x ) \ = \cos ( \sin x )$

## 4.3. The Euler formula and the Euler identity in (weak) racks.

Definition 4.6. We can define a dual rack with the opposite operations, $( S , \ * \ , \bullet \ )$ where the new operations are the following: $a * b = b \diamond a \mathrm { a n d } a \bullet b = b \cdot a$

Remark 4.7. The rack having the following operations is self-dual: $a b = b , ~ a \diamond b = a$

On the set $S \times S$ , we can put a rack structure obtained as the product of the initial rack with its dual. Let us denote its main operation as follows:

$$
( x , \ y ) \ \boxdot ( u , \ v ) = ( x u , \ v \diamond y ) = ( x u , \ y \ast v ) .
$$

For $a \in S$ , we define an “exponential” function on $S \times S \colon$

$$
\exp _ { a } ( x , \ y ) = a ^ { ( x , \ y ) } = ( a x , \ a * y ) .
$$

The “exponential” map has the property: $a ^ { ( x , \ y ) \sqcup ( u , \ v ) } = \ a ^ { ( x , \ y ) } \sqcup a ^ { ( u , \ v ) } ,$

The diagonal map, $\Delta : S  S \times S , \ x \mapsto ( x , \ x )$ , is a rack morphism for a certain rack structure on $S \times S$

## Theorem 4.8. (Euler formula in racks.) The following formula holds:

$$
\exp _ { e } \circ \Delta = [ \cos \ x \ \wedge \mathrm { i n } ] \circ \Delta \ .\tag{4.1}
$$

Equivalently,

$$
e ^ { ( x , \ x ) } = ( \cos x , \ \sin x ) .\tag{4.2}
$$

Moreover, the following identity is true: $e ^ { \Delta ( \Pi ) } = ( U , ~ O )$

Proof. The left hand side reads: $\exp _ { e } \circ \Delta ( x ) = e ^ { ( x , x ) } = ( e x , e \ast x )$ . The right hand side reads: $[ \cos \times \sin ] \circ \Delta ( x ) = ( \cos x , \sin x )$ . So, the left hand side equals the right hand side, because cos $x = e x$ from the definition, and sin $x = x \circ e = e * x$

For the last identity we have: $e ^ { \Delta ( \Pi ) } = e ^ { ( \Pi , \Pi ) } = ( e \Pi , \ e * \Pi ) = ( U , \ O )$

Remark 4.9. The classical Euler’s formula states that:

$$
e ^ { i x } = \cos x + i \sin x \qquad \forall x \in \mathbb { R } .\tag{4.3}
$$

This formula could be also written as:

$$
\exp _ { e } \circ j = [ \cos \ x \ \mathrm { s i n } ] \circ \Delta \ ,\tag{4.4}
$$

where $j : \mathbb { R }  \mathbb { R } \times \mathbb { R } \ , x \mapsto ( 0 , \ x )$ . For $x = \pi$ the Euler’s formula becomes the Euler’s identity.

Remark 4.10. The conics are generated by:

(4.5)

$$
e ^ { i x } = \cos x + i \sin x \qquad \forall x \in \mathbb { R }\tag{4.6}
$$

$$
e ^ { j x } = \cosh x + j \sinh x \ , \ j ^ { 2 } = 1 , \forall x \in \mathbb { R }\tag{4.7}
$$

$$
x e ^ { h x } = x ( 1 + h x ) \ , \ h ^ { 2 } = 0 , \forall x \in \mathbb { R } .
$$

Remark 4.11. The quadrics are generated by:

(4.8)

$$
e ^ { x ( i \cos y + I \sin y ) } = \cos x + i \sin x \cos y + I \sin x \sin y , I ^ { 2 } = - 1 , i I = 0 , \forall x \in \mathbb { R }\tag{4.9}
$$

$$
e ^ { x ( i \cosh y + j \sinh y ) } = \cos x + i \sin x \cosh y + I \sin x \sinh y\tag{4.10}
$$

$$
e ^ { x ( j \cos y + J \sin y ) } , \ J ^ { 2 } = 1 \ , \ j J = 0\tag{4.11}
$$

$$
x e ^ { h \cos y + H \sin y ) } , ~ H ^ { 2 } = 0 ~ , h H = 0\tag{4.12}
$$

$$
x ^ { 2 } e ^ { h \frac { \cosh y } { x } + H \frac { \sin y } { x } } , x ^ { 2 } e ^ { h \frac { \cosh y } { x } + H \frac { \sinh y } { x } } .
$$

We consider the following “hyperbolic” functions:

cosh $\boldsymbol { \mathbf { \mathit { x } } } , \boldsymbol { \mathbf { \mathit { y } } } ) = ( e \boldsymbol { \mathit { x } } , \boldsymbol { \mathbf { \mathit { y } } } )$ and sinh $( x , \ y ) = ( x , e * y )$

It is easy to check the hyperbolic functions Euler formula:

$$
\exp _ { e } \ = \ \cosh \circ \ \sinh \ = \ \sinh \circ \ \cosh .
$$

Theorem 4.12. The functions $\mathrm { e x p } _ { e }$ , cosh and sinh are solutions for the Quantum Yang-Baxter equation $\mathring ( R ^ { 1 2 } \circ R ^ { 1 3 } \circ \mathring { R } ^ { 2 3 } \ = \ R ^ { 2 3 } \circ R ^ { 1 3 } \circ R ^ { 1 2 } \ )$