# Improved Training of Strong Stochastic Flow Maps

Zander W. Blasingame<sup>1</sup> Antonio Franca<sup>1,2</sup> Raúl Miñán<sup>1,2</sup> Christian Koke<sup>1,3,4</sup> Niklas Rindtorff<sup>1,2</sup> Alexander Tong<sup>1</sup>

<sup>1</sup>AITHYRA <sup>2</sup>Technische Universität Wien <sup>3</sup>Technische Universität München <sup>4</sup>Munich Center for Machine Learning

## Abstract

Stochastic flow maps promise few-step sampling of stochastic differential equations (SDEs) by learning to directly jump between times along the solution path, yet training these models to learn this solution map remains difficult. The primary obstacle is the Brownian path itself: it is an infinite-dimensional object which needs to be passed to the neural network and is nowhere differentiable, which makes the implementation of derivative-based losses that power the state-of-the-art deterministic flow maps just out of reach. We identify two core design principles which address these issues: a) represent the Brownian path with a basis which carries higher-order local information, and b) train with an objective that forces the network to use the full representation of the path, not just the first order. To address this, we leverage the shifted Legendre polynomial expansion with a Wong–Zakai styled objective, unlocking derivative-based losses. Finally, we propose a recipe for fine-tuning large pre-trained flow models into strong stochastic flow maps. We ablate through this design space on several toy problems and use our findings to transform a large flow model into a strong stochastic flow map on ImageNet, achieving roughly a three-fold improvement in FD<sub>DINOv2</sub> over SSFM at four NFE.

Code: https://github.com/zblasingame/improved-strong-stochastic-flow-maps   
Models: https://huggingface.co/zblasingame/improved-strong-stochastic-flow-maps   
Correspondence: zblasingame@aithyra.at   
Date: 21 October, 2026

## 1 Introduction

Neura<sup>l</sup> stochastic diferential equations (SDEs) <sup>h</sup>ave <sup>b</sup>ecome a power<sup>f</sup>u<sup>l f</sup>ramewor<sup>k f</sup>or <sup>d</sup>eep enerative mo<sup>d</sup>e<sup>l</sup>s like difusion models (Son<sub>g</sub> et al. 2021) with man<sub>y</sub> a<sub>pp</sub>lications in ima<sub>g</sub>e <sub>g</sub>eneration (Rombach et al. 2022), video <sub>g</sub>eneration (Blattmann et al. 2023), <sub>p</sub>rotein desi<sub>g</sub>n (Watson et al. 2023; Skreta, Atanackovic, et al. 2025), and Boltzmann sam<sub>p</sub>lin<sub>g</sub> (Akhound-Sade<sub>g</sub>h et al. 2025). Solvin<sub>g</sub> these SDEs <sub>q</sub>uickl<sub>y</sub> and accuratel<sub>y</sub> has become <sub>a</sub>n im<sub>po</sub>rt<sub>a</sub>nt <sub>pa</sub>rt <sub>o</sub>f th<sub>e</sub> fi<sub>e</sub>ld with <sub>se</sub>v<sub>e</sub>r<sub>a</sub>l r<sub>esea</sub>r<sub>c</sub>h<sub>e</sub>r<sub>s p</sub>r<sub>opos</sub>in<sub>g</sub> n<sub>e</sub>w n<sub>u</sub>m<sub>e</sub>ri<sub>ca</sub>l SDE <sub>so</sub>lv<sub>e</sub>r<sub>s</sub> whi<sub>c</sub>h <sub>e</sub>n<sub>a</sub>bl<sub>e</sub>d <sub>a</sub> sma<sup>ll</sup>er number offunction evaluations (NFE), i.e., t<sup>h</sup>e num<sup>b</sup>er o<sup>f</sup> times t<sup>h</sup>e expensive neura<sup>l</sup> networ<sup>k</sup> is ca<sup>ll</sup>e<sup>d</sup> (Lu et al. 2022; Gonzalez et al. 2023; Blasin<sub>g</sub>ame and Liu 2026). A new thread of research has looked at amortizin<sub>g</sub> the cost of inte<sub>g</sub>ration durin<sub>g</sub> inference over trainin<sub>g</sub> b<sub>y</sub> learnin<sub>g</sub> how to inte<sub>g</sub>rate ODEs (Son<sub>g</sub> et al. 2023; Gen<sub>g</sub>, Den<sub>g</sub>, et al. 2025). Remarkabl<sub>y</sub>, such a<sub>pp</sub>roaches have been able to <sub>p</sub>roduce <sub>p</sub>henomenall<sub>y p</sub>owerful few-ste<sub>p</sub> m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> th<sub>e pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce o</sub>f tr<sub>a</sub>diti<sub>o</sub>n<sub>a</sub>l fl<sub>o</sub>w/dif<sub>us</sub>i<sub>o</sub>n m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> with <sub>a s</sub>i<sub>g</sub>nifi<sub>ca</sub>nt r<sub>e</sub>d<sub>uc</sub>ti<sub>o</sub>n in NFE<sub>.</sub> S<sub>uc</sub>h

![](images/16726800aa617bc329014b530a33edc27ee7a01f509cf7aa525721f7111d27df.jpg)

![](images/bf0e37df206ee5bbb6ca41c66cc4c36d5fe9bc9c94a40566a330aee9cee72744.jpg)  
Figure 1 Learning strong stochastic flow maps. (Left): A strong stochastic flow map $\Psi _ { s , t }$ l<sub>earns</sub> th<sub>e so</sub>l<sub>u</sub>ti<sub>on map</sub> t<sub>o an</sub> SDE driven b<sub>y</sub> a rou<sub>g</sub>h Brownian <sub>p</sub>ath<sub>,</sub> which is nowhere diferentiable. We re<sub>p</sub>lace the SDE b<sub>y</sub> a random ODE driven b<sub>y</sub> a smooth <sub>p</sub>ol<sub>y</sub>nomial a<sub>pp</sub>roximation ${ \cal W } ^ { ( N ) }$ . (Right): Comparing our iSSFM and SSFM to the SiT-XL/2+REPA pre-trained m<sub>o</sub>d<sub>e</sub>l in $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ -50k on Ima<sub>g</sub>eNet-256, with u<sub>p</sub> to a rou<sub>g</sub>hl<sub>y</sub> 3× im<sub>p</sub>rovement over SSFM.

models are broadl<sub>y</sub> called deterministic flow ma<sub>p</sub>s as the<sub>y</sub> learn the solution of an ODE <sub>p</sub>athwise (Bofi et al.   
2025).

Learnin an analo ous object, i.e., a stochastic flow ma , is uite dificult due to the com lexit of handlin the Brownian path—an infinite-dimensional object which needs to be encoded for the neural network. Recently, McCa<sup>ll</sup>um et a<sup>l</sup>. (2026) an<sup>d</sup> Pan et a<sup>l</sup>. (2026) <sup>fi</sup>gure<sup>d</sup> out <sup>h</sup>ow to <sup>l</sup>earn t<sup>h</sup>e strong solution o<sup>f</sup> t<sup>h</sup>e un<sup>d</sup>er<sup>l</sup>ying SDE (also known as an Itô ma<sub>p</sub>). While <sub>q</sub>uite <sub>p</sub>romisin<sub>g</sub>, the trainin<sub>g</sub> methods for such models are unstable and r<sub>e</sub>l<sub>a</sub>tiv<sub>e</sub>l<sub>y</sub> limit<sub>e</sub>d <sub>co</sub>m<sub>pa</sub>r<sub>e</sub>d t<sub>o</sub> th<sub>e</sub>ir d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c pee</sub>r<sub>s,</sub> with h<sub>ea</sub>v<sub>y</sub> r<sub>es</sub>tri<sub>c</sub>ti<sub>o</sub>n<sub>s o</sub>n th<sub>e</sub> v<sub>a</sub>lid l<sub>osses a</sub>bl<sub>e</sub> t<sub>o</sub> b<sub>e</sub> use<sup>d f</sup>or training. E.g., prior wor<sup>k</sup> in <sup>d</sup>eterministic <sup>fl</sup>ow maps <sup>h</sup>as <sup>f</sup>oun<sup>d</sup> t<sup>h</sup>e Eu<sup>l</sup>erian-sty<sup>l</sup>e <sup>l</sup>oss emp<sup>l</sup>oye<sup>d b</sup>y Gen<sub>g</sub>, Den<sub>g</sub>, et al. (2025) and Gen<sub>g</sub>, Lu, et al. (2025) to be the most <sub>p</sub>erformant in trainin<sub>g</sub> few-ste<sub>p</sub> models at <sub>sca</sub>l<sub>e;</sub> h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r<sub>,</sub> <sub>suc</sub>h <sub>a</sub> d<sub>e</sub>riv<sub>a</sub>tiv<sub>e</sub>-b<sub>ase</sub>d l<sub>oss</sub> <sub>poses</sub> <sub>se</sub>v<sub>e</sub>r<sub>e</sub> im<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>ti<sub>o</sub>n i<sub>ssues</sub> f<sub>o</sub>r <sub>s</sub>tr<sub>o</sub>n<sub>g</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>aps</sub> b<sub>y</sub> re<sub>q</sub>uirin<sub>g</sub> the time-derivative of the Brownian <sub>p</sub>ath (McCallum et al. 2026), and the objective <sub>p</sub>rovides no <sub>p</sub>athwise constraint and is thus ill-<sub>p</sub>osed (Pan et al. 2026).

Fundamentally, our key observation is that to train a successful strong stochastic flow map, one needs both a) a representation of the Brownian path which eficiently encodes higher-order local information about the path, and b) a training loss which forces the network to use the full Brownian path approximation. We find that adherin<sub>g</sub> to these two basic desi<sub>g</sub>n <sub>p</sub>rinci<sub>p</sub>les results in im<sub>p</sub>roved trainin<sub>g</sub> of stron<sub>g</sub> stochastic flow ma<sub>p</sub>s. As such<sub>,</sub> we <sub>p</sub>ro<sub>p</sub>ose a new framework for re<sub>p</sub>resentin<sub>g</sub> the underl<sub>y</sub>in<sub>g</sub> SDE based on the shifted Le<sub>g</sub>endre <sub>p</sub>ol<sub>y</sub>nomial a<sub>pp</sub>roximation used b<sub>y</sub> McCallum et al. (2026) with a Won<sub>g</sub>–Zakai st<sub>y</sub>le loss, which enables us to use the powerful Eulerian training objective. Additionally, we construct a framework which allows us to fine-tune a large pre-trained deterministic flow matching model into a strong stochastic flow map à la Lee et al. (2026) but with several modifications to make the idea work in the stochastic settin<sub>g</sub>. With our im<sub>p</sub>roved trainin<sub>g</sub> techni<sub>q</sub>ues<sub>,</sub> we are able to achieve hi<sub>g</sub>h-<sub>q</sub>ualit<sub>y</sub> flow ma<sub>p</sub>s which <sub>p</sub>reserve the stron<sub>g</sub> solution <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub>. We <sub>e</sub>n<sub>u</sub>m<sub>e</sub>r<sub>a</sub>t<sub>e ou</sub>r m<sub>a</sub>in <sub>co</sub>ntrib<sub>u</sub>ti<sub>o</sub>n<sub>s</sub> b<sub>e</sub>l<sub>o</sub>w<sub>:</sub>

• We com<sub>p</sub>are several re<sub>p</sub>resentations of the Brownian <sub>p</sub>ath to find the o<sub>p</sub>timal re<sub>p</sub>resentation.

• We propose a Wong–Zakai style objective for training the strong stochastic flow map.

• W<sub>e</sub> <sub>c</sub>r<sub>ea</sub>t<sub>e</sub> <sub>a</sub> n<sub>o</sub>v<sub>e</sub>l m<sub>e</sub>th<sub>o</sub>d t<sub>o</sub> fin<sub>e</sub>-t<sub>u</sub>n<sub>e</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> int<sub>o</sub> <sub>s</sub>tr<sub>o</sub>n<sub>g</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>aps.</sub>

## 2 Background

Flow matching and diffusion models. We provide a brief overview of flow and difusion models using the flow m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> n<sub>o</sub>m<sub>e</sub>n<sub>c</sub>l<sub>a</sub>t<sub>u</sub>r<sub>e.</sub> W<sub>e</sub> d<sub>e</sub>n<sub>o</sub>t<sub>e</sub> d<sub>a</sub>t<sub>a sa</sub>m<sub>p</sub>l<sub>es as</sub> th<sub>e</sub> $\mathbb { R } ^ { d }$ -v<sub>a</sub>l<sub>ue</sub>d r<sub>a</sub>nd<sub>o</sub>m v<sub>a</sub>ri<sub>a</sub>bl<sub>e</sub> $X _ { 1 }$ d<sub>rawn</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $p _ { 1 }$ <sub>a</sub>nd d<sub>e</sub>n<sub>o</sub>t<sub>e</sub> th<sub>e sou</sub>r<sub>ce</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>as</sub> $X _ { 0 } ~ \sim ~ p _ { 0 }$ (often the unit Gaussian) also on $\mathbb { R } ^ { d }$ <sub>.</sub> Th<sub>e</sub>n<sub>,</sub> <sup>f</sup>o<sup>ll</sup>owing Lipman et a<sup>l</sup>. (2024), we consi<sup>d</sup>er t<sup>h</sup>e scenario o<sup>f</sup> afine Gaussian probability paths w<sup>h</sup>ere we <sup>d</sup>e<sup>fi</sup>ne t<sup>h</sup>e interpolant (A<sup>lb</sup>ergo et a<sup>l</sup>. 2025) as $X _ { t } = \alpha _ { t } X _ { 1 } + \sigma _ { t } X _ { 0 }$ f<sub>or some su</sub>it<sub>a</sub>bl<sub>e no</sub>i<sub>se sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub> $( \alpha _ { t } , \sigma _ { t } ) .$ <sup>1</sup> W<sub>e ca</sub>n th<sub>e</sub>n <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a vec</sub>t<sub>or</sub> fi<sub>e</sub>ld ${ \pmb u } _ { t } ( { \pmb x } ) = \mathbb { E } [ \dot { \alpha } _ { t } { \pmb X } _ { 1 } + \dot { \sigma } _ { t } { \pmb X } _ { 0 } | { \pmb X } _ { t } = { \pmb x } ]$ <sub>suc</sub>h th<sub>a</sub>t it<sub>s</sub> i<sub>n</sub>d<sub>uce</sub>d fl<sub>ow map</sub> � <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\begin{array} { r } { \Phi _ { s , t } ( { \bf x } _ { s } ) = { \bf x } _ { s } + \int _ { s } ^ { t } \mathrm { d } r \ { \bf u } _ { r } ( { \bf x } _ { r } ) } \end{array}$ <sub>a</sub>nd $\Phi _ { s , t \sharp } p _ { s } = p _ { t }$ f<sub>or a</sub>ll $s , t \in [ 0 , 1 ]$ <sub>.</sub> T<sub>o</sub> l<sub>earn</sub> thi<sub>s vec</sub>t<sub>or</sub> fi<sub>e</sub>ld <sub>we use con</sub>diti<sub>ona</sub>l flow matchin<sub>g</sub> (Peluchetti 2021; Li<sub>p</sub>man et al. 2023; Liu et al. 2023) and find ${ \pmb u } _ { t } ^ { \theta } \approx { \pmb u } _ { t }$ <sub>.</sub> Th<sub>e</sub>n t<sub>o</sub> dr<sub>a</sub>w <sub>sa</sub>m<sub>p</sub>l<sub>es</sub> f<sub>rom</sub> $p _ { 1 }$ w<sub>e</sub> <sub>s</sub>im<sub>p</sub>l<sub>y</sub> dr<sub>a</sub>w tr<sub>ac</sub>t<sub>a</sub>bl<sub>e</sub> <sub>sa</sub>m<sub>p</sub>l<sub>es</sub> fr<sub>o</sub>m $X _ { 0 } \sim p _ { 0 }$ and inte<sub>g</sub>rate from 0 to 1 usin<sub>g</sub> a numerical scheme. Lik<sub>e</sub>wi<sub>se,</sub> w<sub>e ca</sub>n r<sub>e</sub>writ<sub>e</sub> th<sub>e</sub> fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> ODE <sub>as a</sub>n SDE with <sub>equ</sub>iv<sub>a</sub>l<sub>e</sub>nt m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s, g</sub>iv<sub>e</sub>n b<sub>y</sub>

$$
\mathrm { d } X _ { t } = \left[ 2 \pmb { u } _ { t } ( X _ { t } ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { X } _ { t } \right] \ \mathrm { d } t + \nu _ { t } \ \mathrm { d } W _ { t } , \quad \nu _ { t } ^ { 2 } = 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \sigma _ { t } ^ { 2 } - 2 \sigma _ { t } \dot { \sigma } _ { t } ,\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> ${ \mathbf { } } W _ { t }$ i<sub>s</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd Br<sub>o</sub>wni<sub>a</sub>n m<sub>o</sub>ti<sub>o</sub>n <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> filt<sub>e</sub>r<sub>e</sub>d <sub>p</sub>r<sub>o</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>space</sub> $( \Omega , { \mathcal { F } } , \mathbb { F } , P )$ (Holderrieth, Sin<sub>g</sub>er, et al. 2026).

Flow maps. A natura<sup>l</sup> question to as<sup>k</sup> is: why bother with a very expensive integration procedure when we are already learning a neural network? T<sup>h</sup>is question <sup>h</sup>as prompte<sup>d</sup> wor<sup>k</sup> on <sup>l</sup>earning neura<sup>l fl</sup>ow maps, i.e., learnin<sub>g</sub> $\Phi _ { s , t } ^ { \theta }$ (Bofi et al. 2025; Gen<sub>g</sub>, Den<sub>g</sub>, et al. 2025). Generall<sub>y</sub>, the loss amounts to <sub>p</sub>erformin<sub>g</sub> a standard flow matchin<sub>g</sub> (dia<sub>g</sub>onal) loss to learn the vector field ${ \pmb u } _ { t } ^ { \theta }$ <sub>a</sub>nd th<sub>e</sub>n <sub>us</sub>in<sub>g a seco</sub>nd<sub>a</sub>r<sub>y</sub> l<sub>oss</sub> t<sub>o</sub> di<sub>s</sub>till <sub>ou</sub>t th<sub>e</sub> v<sub>ec</sub>t<sub>o</sub>r field into a flow ma<sub>p</sub>. Note that these two losses are sometimes combined into a sin<sub>g</sub>le loss (Son<sub>g</sub> et al. 2023; Gen<sub>g</sub>, Lu, et al. 2025; Sabour, Fidler, et al. 2025). These models are <sub>g</sub>enerall<sub>y p</sub>arameterized as an Euler ste<sub>p</sub> of th<sub>e</sub> f<sub>orm</sub> $\Phi _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { \mathit { x } } } } ) = { \boldsymbol { \mathbf { \mathit { x } } } } + ( t - s ) \mathbf { \mathit { { u } } } _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { \mathit { x } } } } )$ <sub>, w</sub>h<sub>ere</sub> $\pmb { u } _ { s , t } ^ { \theta }$ <sup>l</sup>earns t<sup>h</sup>e mean ve<sup>l</sup>ocity. T<sup>h</sup>e most per<sup>f</sup>ormant <sup>l</sup>osses o<sup>f</sup>ten involve evaluatin<sub>g</sub> derivatives of the flow ma<sub>p</sub> (Sabour, Fidler, et al. 2025).

Strong stochastic flow maps. Analogously for SDEs we are interested in learning a stochastic flow map which describes how to jump from $X _ { s }$ to $X _ { t }$ $I . e .$ , for the SDE in E<sub>q</sub>uation (1) we wish to learn the solution ma<sub>p</sub> (or Itô ma ), a random variable $\Psi _ { s , t } ( \cdot ; \omega )$ f<sub>or</sub> $\omega \in \Omega$ on a <sub>p</sub>robabilit<sub>y</sub> s<sub>p</sub>ace (Ω, F, �) which describes the dif<sub>eo</sub>m<sub>o</sub>r<sub>p</sub>hi<sub>s</sub>m

$$
\Psi _ { s , t } ( X _ { s } ; \omega ) = X _ { s } ( \omega ) + \int _ { s } ^ { t } \mathrm { d } r ~ \left[ 2 \boldsymbol { u } _ { r } ( X _ { r } ( \omega ) ) - \frac { \dot { \alpha } _ { r } } { \alpha _ { r } } X _ { r } ( \omega ) \right] + \int _ { s } ^ { t } \mathrm { d } W _ { r } ( \omega ) ~ \nu _ { r } .\tag{2}
$$

Broa<sup>dl</sup>y, we can <sup>l</sup>earn eit<sup>h</sup>er a weak stoc<sup>h</sup>astic <sup>fl</sup>ow map or a strong stoc<sup>h</sup>astic <sup>fl</sup>ow map (McCa<sup>ll</sup>um et a<sup>l</sup>. 2026) wh<sub>e</sub>r<sub>e</sub> th<sub>e</sub> f<sub>o</sub>rm<sub>e</sub>r im<sub>p</sub>li<sub>es</sub> w<sub>e o</sub>nl<sub>y</sub> l<sub>ea</sub>rn <sub>equa</sub>lit<sub>y</sub> in di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>,</sub> wh<sub>e</sub>r<sub>eas</sub> f<sub>o</sub>r th<sub>e</sub> l<sub>a</sub>tt<sub>e</sub>r w<sub>e</sub> l<sub>ea</sub>rn <sub>equa</sub>lit<sub>y</sub> in <sub>p</sub>aths.<sup>2</sup> In <sub>p</sub>ractice<sub>,</sub> stron<sub>g</sub> stochastic flow ma<sub>p</sub>s are <sub>p</sub>arameterized as an Euler–Maru<sub>y</sub>ama-st<sub>y</sub>le ste<sub>p</sub> of the f<sub>orm</sub> $\begin{array} { r } { \Psi _ { s , t } ^ { \theta } ( { \pmb x } ; { \widetilde { J } } ) = { \pmb x } + ( t - s ) { \pmb u } _ { s , t } ^ { \theta } ( { \pmb x } ; { \widetilde { J } } ) + ( { \pmb W } _ { t } - { \pmb W } _ { s } ) { \pmb g } _ { s , t } ( { \widetilde { J } } ) } \end{array}$ , where I denotes some kind of finite-dimensional re<sub>p</sub>resentation of a realization of the Brownian <sub>p</sub>ath $W _ { [ s , t ] }$ <sub>, a</sub>nd $\pmb { g } _ { s , t }$ <sub>can</sub> b<sub>e e</sub>ith<sub>er a</sub> l<sub>earne</sub>d <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k (McCallum et al. 2026) or numerical <sub>q</sub>uadrature of the mean stochastic inte<sub>g</sub>ral (Pan et al. 2026). This construction raises two natural <sub>q</sub>uestions: a) how does one <sub>g</sub>o from the Brownian <sub>p</sub>ath ${ \mathbf { } } W _ { t }$ to some finite re<sub>p</sub>resentation I which can be fed to a neural network and b) what losses can be used to train the ma<sub>p</sub> in the <sub>p</sub>resence of the n<sub>o</sub>wh<sub>e</sub>r<sub>e</sub> dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>bl<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th?

## 3 Improved training of strong stochastic flow maps

Str<sub>o</sub>n <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>a s</sub> r<sub>o</sub>mi<sub>se</sub> t<sub>o u</sub>nl<sub>oc</sub>k n<sub>e</sub>w m<sub>o</sub>d<sub>e</sub>lin <sub>a</sub>r<sub>a</sub>di m<sub>s, e</sub>t th<sub>e cu</sub>rr<sub>e</sub>nt tr<sub>a</sub>inin r<sub>ec</sub>i <sub>es</sub> f<sub>o</sub>r th<sub>e</sub>m <sub>a</sub>r<sub>e qu</sub>it<sub>e pa</sub>rti<sub>cu</sub>l<sub>a</sub>r <sub>a</sub>nd <sub>u</sub>nd<sub>e</sub>r<sub>e</sub>x<sub>p</sub>l<sub>o</sub>r<sub>e</sub>d<sub>.</sub> A<sub>s</sub> m<sub>e</sub>nti<sub>o</sub>n<sub>e</sub>d b<sub>e</sub>f<sub>o</sub>r<sub>e,</sub> th<sub>e</sub> infinit<sub>e</sub>-dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>a</sub>l n<sub>a</sub>t<sub>u</sub>r<sub>e o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th <sub>a</sub>nd it<sub>s</sub> n<sub>o</sub>wh<sub>e</sub>r<sub>e</sub> dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>bilit<sub>y</sub> m<sub>a</sub>k<sub>e</sub> it difi<sub>cu</sub>lt t<sub>o</sub> w<sub>o</sub>rk with<sub>.</sub> Th<sub>e</sub> m<sub>os</sub>t <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>nt l<sub>osses</sub> f<sub>o</sub>r tr<sub>a</sub>inin<sub>g</sub> deterministic flow ma s re uire takin time derivatives of the flow ma to satisf the Eulerian (sometimes La<sub>g</sub>ran<sub>g</sub>ian) condition (Sabour, Fidler, et al. 2025), but as the Brownian <sub>p</sub>ath is nowhere diferentiable, this <sub>p</sub>oses serious theoretical and im<sub>p</sub>lementation issues. McCallum et al. (2026) avoids this issue b<sub>y</sub> usin<sub>g</sub> the semi<sub>g</sub>rou<sub>p</sub> l<sub>oss;</sub> h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r<sub>,</sub> in th<sub>e</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c se</sub>ttin<sub>g</sub> thi<sub>s</sub> w<sub>as s</sub>h<sub>o</sub>wn t<sub>o</sub> b<sub>e</sub> l<sub>ess pe</sub>rf<sub>o</sub>rm<sub>a</sub>nt th<sub>a</sub>n th<sub>e</sub> d<sub>e</sub>riv<sub>a</sub>tiv<sub>e</sub>-b<sub>ase</sub>d l<sub>osses</sub> in <sub>p</sub>ractice. Likewise<sub>,</sub> there is also the <sub>q</sub>uestion of how to re<sub>p</sub>resent the Brownian <sub>p</sub>ath as a tractable in<sub>p</sub>ut to the neural network. Lastl<sub>y</sub>, we would like to be able to distill lar<sub>g</sub>e flow matchin<sub>g</sub> models (or deterministic flow ma<sub>p</sub>s) into stron<sub>g</sub> stochastic flow ma<sub>p</sub>s while levera<sub>g</sub>in<sub>g</sub> their <sub>p</sub>re-trainin<sub>g</sub> efectivel<sub>y</sub>. These three desi<sub>g</sub>n <sub>g</sub>oals and <sub>q</sub>uestions lead us to discover three desi n <sub>p</sub>rinci<sub>p</sub>les for <sub>p</sub>arameterizin and trainin lar e stron stochastic flow ma<sub>p</sub>s which we summarize as: 1) choose a smooth hi<sub>g</sub>her-order a<sub>pp</sub>roximation of the Brownian <sub>p</sub>ath which has closed-form composition laws, 2) choose an objective which ensures that the neural network learns to use the <sub>p</sub>ath, and 3) choose a <sub>p</sub>arameterization such that the drift matches the <sub>p</sub>re-trained flow model.

## 3.1 How to represent the Brownian path

One of the main <sub>q</sub>uestions is how to re<sub>p</sub>resent the Brownian <sub>p</sub>ath $W _ { [ s , t ] }$ . Let ${ \pmb W } _ { [ s , t ] } ^ { ( N ) }$ d<sub>e</sub>n<sub>o</sub>t<sub>e ou</sub>r �-th <sub>o</sub>rd<sub>e</sub>r a<sub>pp</sub>roximation of the Brownian <sub>p</sub>ath where � denotes how man<sub>y</sub> $\mathbb { R } ^ { d }$ <sub>va</sub>l<sub>ues we pass</sub> t<sub>o</sub> th<sub>e ne</sub>t<sub>wor</sub>k<sub>.</sub> W<sub>e</sub> h<sub>ave</sub> two main <sub>p</sub>ro<sub>p</sub>erties we would like: 1) we want to have a com<sub>p</sub>osition law such that <sub>g</sub>iven ${ \pmb W } _ { [ s , u ] } ^ { ( N ) }$ <sub>a</sub>nd ${ \pmb W } _ { [ u , t ] } ^ { ( N ) }$ we <sub>ca</sub>n b<sub>u</sub>ild ${ \pmb W } _ { [ s , t ] } ^ { ( N ) }$ <sub>w</sub>ith $u \in \left( s , t \right)$ , and 2) we would ideall<sub>y</sub> like to have a smooth a<sub>pp</sub>roximation which admits <sub>c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm d<sub>e</sub>riv<sub>a</sub>tiv<sub>es o</sub>f th<sub>e pa</sub>th<sub>.</sub> Thi<sub>s</sub> l<sub>ea</sub>d<sub>s</sub> n<sub>a</sub>t<sub>u</sub>r<sub>a</sub>ll<sub>y</sub> t<sub>o ou</sub>r fir<sub>s</sub>t d<sub>es</sub>i<sub>g</sub>n <sub>p</sub>rin<sub>c</sub>i<sub>p</sub>l<sub>e:</sub>

Principle 1. Choose a smooth higher-order approximation of the Brownian path which has closed-form com<sub>p</sub>osition laws.

Brownian increments. The naïve approach would be to sam-<sub>p</sub>le the <sub>p</sub>ath at � discrete <sub>p</sub>oints $\{ t _ { n } \} _ { n = 1 } ^ { N } \subseteq [ s , t ]$ <sub>an</sub>d th<sub>en</sub> pass this trajectory to the network, i.e., give the neural net-<sub>wor</sub>k <sub>a ro</sub>ll<sub>ou</sub>t <sub>o</sub>f th<sub>e</sub> B<sub>rown</sub>i<sub>an pa</sub>th <sub>over</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l $[ s , t ]$ Thi<sub>s</sub> i<sub>s so</sub>m<sub>e</sub>wh<sub>a</sub>t <sub>a</sub>n<sub>a</sub>l<sub>ogous</sub> t<sub>o</sub> in<sub>c</sub>r<sub>eas</sub>in<sub>g</sub> th<sub>e</sub> fid<sub>e</sub>lit<sub>y o</sub>f <sub>a</sub>n E<sub>u</sub>l<sub>e</sub>r–M<sub>a</sub>r<sub>uya</sub>m<sub>a so</sub>lv<sub>e</sub> b<sub>y</sub> t<sub>a</sub>kin<sub>g s</sub>m<sub>a</sub>ll<sub>e</sub>r <sub>s</sub>t<sub>eps;</sub> h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r<sub>,</sub> thi<sub>s</sub> l<sub>ac</sub>k<sub>s</sub> hi<sub>g</sub>h<sub>er-or</sub>d<sub>er</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t th<sub>e pa</sub>th<sub>, w</sub>hi<sub>c</sub>h <sub>we w</sub>ill l<sub>a</sub>t<sub>er s</sub>h<sub>ow</sub> i<sub>s essen</sub>ti<sub>a</sub>l<sub>.</sub>

Karhunen–Loève expansion. A classic tool in stochastic anal<sub>y</sub>sis is the Karhunen–Loève ex<sub>p</sub>ansion of the Brownian <sub>p</sub>ath which is a Fourier-es<sub>q</sub>ue decom<sub>p</sub>osition of the Brownian <sub>p</sub>ath (Loève 1945; Karhunen 1950). The main limitation of the Karhunen–Loève ex<sub>p</sub>ansion in this context is that it d<sub>oes</sub> n<sub>o</sub>t <sub>a</sub>dmit <sub>c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm Ch<sub>e</sub>n r<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s o</sub>n it<sub>s coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s.</sub> This is sub-o<sub>p</sub>timal for this a<sub>pp</sub>lication as the learned Itô map jumping <sup>f</sup>rom � to � <sup>h</sup>as extra unneeded in<sup>f</sup>ormation on $[ 0 , s ) \cup ( t , 1 ]$ <sub>.</sub> Additi<sub>o</sub>n<sub>a</sub>ll<sub>y,</sub> f<sub>o</sub>r finit<sub>e</sub> �<sub>,</sub> thi<sub>s e</sub>x<sub>pa</sub>n<sub>s</sub>i<sub>o</sub>n h<sub>as</sub> an end<sub>p</sub>oint that is not e<sub>q</sub>ual to the true end<sub>p</sub>oint ${ \mathbf { } } W _ { t }$ . As such, Pan et al. (2026) em<sub>p</sub>iricall<sub>y</sub> found it necessar<sub>y</sub> to add the local increment $W _ { t } - W _ { s }$ t<sub>o</sub> <sub>ge</sub>t th<sub>ese</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> t<sub>o per</sub>f<sub>orm we</sub>ll<sub>.</sub>

![](images/2f48752a234f1cb11179d398dc6c635926b48ffc9bc9c139d82d718fa7b4f527.jpg)  
Figure 2 Algebra of the shifted Legendre polynomials. Com<sup>b</sup>ining t<sup>h</sup>ir<sup>d</sup>-or<sup>d</sup>er approximations at arbitrary s<sub>p</sub>lit <sub>p</sub>oints to create summar<sub>y</sub> third-order Brownian <sub>p</sub>aths over lar<sub>g</sub>er intervals via Theorem B.<sub>3</sub>.

Shifted Legendre polynomial expansion. We could also decompose the Brownian path with a diferent basis <sub>on</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l $[ s , t ]$ <sub>,</sub> n<sub>a</sub>m<sub>e</sub>l<sub>y, s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l<sub>s</sub>—thi<sub>s co</sub>m<sub>es</sub> with th<sub>e a</sub>dd<sub>e</sub>d b<sub>e</sub>n<sub>e</sub>fit <sub>o</sub>f <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s</sub> corres<sub>p</sub>ondin<sub>g</sub> to terms in the stochastic Ta<sub>y</sub>lor ex<sub>p</sub>ansion of an additive noise SDE<sub>,</sub> $i . e .$ , a corres<sub>p</sub>on<sup>d</sup>ence to t<sup>h</sup>e space-time Lévy areas (Foster et a<sup>l</sup>. 2020).<sup>3</sup> N.B., it is <sup>f</sup>or t<sup>h</sup>is reason t<sup>h</sup>at t<sup>h</sup>e s<sup>h</sup>i<sup>f</sup>ted Legendre basis is o<sup>f</sup>ten <sub>p</sub>referred over the Karhunen–Loève basis for desi<sub>g</sub>nin<sub>g</sub> state-of-the-art SDE solvers (Foster et al. 2024). For the Brownian <sub>p</sub>ath from � to � on an interval $\left[ s , t \right] \supseteq \left[ u , \nu \right]$ , the basis used b McCallum et al. (2026) takes the form <sub>o</sub>f

$$
\pmb { W } _ { u , v } ^ { ( N , s , t ) } : = \sum _ { n = 0 } ^ { N - 1 } \frac { 2 n + 1 } { t - s } \pmb { I } _ { s , t } ^ { ( n ) } \int _ { u } ^ { \nu } \mathrm { d } r ~ \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) , \quad \pmb { I } _ { s , t } ^ { ( n ) } \sim \mathcal { N } \left( \pmb { 0 } , \frac { t - s } { 2 n + 1 } \pmb { I } \right) ,\tag{3}
$$

<sub>w</sub>h<sub>ere</sub> ${ \widetilde { P } } _ { n }$ i<sub>s</sub> th<sub>e �</sub>-th <sub>s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l <sub>o</sub>n $[ 0 , 1 ]$ <sub>.</sub> A<sub>no</sub>th<sub>er</sub> hi<sub>g</sub>hl<sub>y use</sub>f<sub>u</sub>l <sub>componen</sub>t <sub>o</sub>f thi<sub>s s</sub>t<sub>ruc</sub>t<sub>ure</sub> i<sub>s</sub> that there are closed-form Chen relations (composition laws) for all the coeficients, i.e., given $\{ I _ { s , u } ^ { ( n ) } \}$ <sub>a</sub>nd $\{ I _ { u , t } ^ { ( n ) } \}$ <sub>we</sub> h<sub>ave a</sub> l<sub>aw</sub> f<sub>or crea</sub>ti<sub>ng</sub> $\{ { \cal I } _ { s , t } ^ { ( n ) } \}$ f<sub>or</sub> $\begin{array} { r } { u = \frac { 1 } { 2 } \big ( s + t \big ) } \end{array}$ . N.B., t<sup>h</sup>is a<sup>ll</sup>ows <sup>f</sup>or t<sup>h</sup>e creation o<sup>f l</sup>arger trees su<sup>b</sup>suming t<sup>h</sup>e d <sub>a</sub>di<sub>c</sub> in<sub>c</sub>r<sub>e</sub>m<sub>e</sub>nt tr<sub>ee</sub> d<sub>esc</sub>rib<sub>e</sub>d <sub>ea</sub>rli<sub>e</sub>r<sub>,</sub> n<sub>o</sub>w <sub>e</sub>nh<sub>a</sub>n<sub>ce</sub>d with hi h<sub>e</sub>r-<sub>o</sub>rd<sub>e</sub>r <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> int<sub>e</sub> r<sub>a</sub>l<sub>s.</sub>

The algebra of the shifted Legendre polynomial. As mentioned earlier, there exist Chen relations (Chen 1954, 1957), or interva<sup>l</sup> a<sup>l</sup>ge<sup>b</sup>ra, <sup>f</sup>or mi<sup>d</sup>point composition o<sup>f</sup> t<sup>h</sup>e coe<sup>fi</sup>cients, viz., a met<sup>h</sup>o<sup>d</sup> to com<sup>b</sup>ine two po<sup>l</sup>ynomia<sup>l</sup>s <sup>d</sup>erive<sup>d f</sup>rom t<sup>h</sup>e true Brownian pat<sup>h</sup> ${ \mathbf { } } W _ { t }$ into anot<sup>h</sup>er po<sup>l</sup>ynomia<sup>l</sup> o<sup>f</sup> t<sup>h</sup>e same order <sup>b</sup>ut over a <sup>l</sup>arger interva<sup>l</sup>. We derive a <sup>f</sup>ormu<sup>l</sup>a <sup>f</sup>or composition a<sup>l</sup>ong an ar<sup>b</sup>itrary sp<sup>l</sup>it $u \in ( s , t )$ <sub>,</sub> whi<sub>c</sub>h w<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il in Th<sub>eo</sub>r<sub>e</sub>m $\mathrm { B } . 3$ i<sub>n</sub> th<sub>e</sub> appen<sup>d</sup>ix, $q . \nu .$ This arbitrar<sub>y</sub> s<sub>p</sub>lit allows us to later im<sub>p</sub>lement the discretized Eulerian and La<sub>g</sub>ran<sub>g</sub>ian losses (q.v. Section $3 { \cdot } 2 )$ <sub>.</sub> Additi<sub>o</sub>n<sub>a</sub>ll<sub>y,</sub> thi<sub>s</sub> <sub>a</sub>ll<sub>o</sub>w<sub>s</sub> <sub>us</sub> t<sub>o</sub> <sub>sa</sub>m<sub>p</sub>l<sub>e</sub> <sub>o</sub>n n<sub>o</sub>n-d<sub>ya</sub>di<sub>c</sub> int<sub>e</sub>rv<sub>a</sub>l<sub>s</sub> <sub>a</sub>nd <sub>sa</sub>m<sub>p</sub>l<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s</sub> <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e u</sub>nd<sub>e</sub>rl in Br<sub>o</sub>wni<sub>a</sub>n <sub>a</sub>th<sub>,</sub> whi<sub>c</sub>h w<sub>e</sub> ill<sub>us</sub>tr<sub>a</sub>t<sub>e</sub> in Fi <sub>u</sub>r<sub>e</sub> 2<sub>.</sub> T<sub>o e</sub>n<sub>a</sub>bl<sub>e</sub> E<sub>u</sub>l<sub>e</sub>ri<sub>a</sub>n <sub>a</sub>nd L<sub>a</sub> r<sub>a</sub>n i<sub>a</sub>n l<sub>osses</sub> in <sub>co</sub>ntin<sub>uous</sub> tim<sub>e,</sub> w<sub>e use</sub> th<sub>a</sub>t E<sub>qua</sub>ti<sub>o</sub>n $\left( 3 \right)$ <sub>a</sub>dmit<sub>s c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm d<sub>e</sub>riv<sub>a</sub>tiv<sub>es,</sub> whi<sub>c</sub>h w<sub>e s</sub>h<sub>o</sub>w in L<sub>e</sub>mm<sub>a</sub> B<sub>.</sub>1<sub>.</sub>

Interesting<sup>l</sup>y, t<sup>h</sup>e coe<sup>fi</sup>cients <sup>b</sup>etween t<sup>h</sup>e two representations are roug<sup>hl</sup>y t<sup>h</sup>e same, viz. norma<sup>l d</sup>istri<sup>b</sup>utions with decreasin<sub>g</sub> variance in �—which ma<sub>y</sub> be normalized to unit Gaussians. So wh<sub>y</sub> do we observe em<sub>p</sub>irical (see Section $4 { \cdot } 1 )$ an<sup>d</sup> t<sup>h</sup>eoretica<sup>l d</sup>i<sup>f</sup>erences <sup>b</sup>etween t<sup>h</sup>em i<sup>f</sup> our neura<sup>l</sup> networ<sup>k</sup> e<sup>f</sup>ective<sup>l</sup>y sees t<sup>h</sup>e same Gaussians? $V i z .$ <sub>.,</sub> h<sub>ow we use</sub> Th<sub>eorem</sub> $\mathrm { B } . 3$ <sub>an</sub>d L<sub>emma</sub> B<sub>.1</sub> i<sub>s paramoun</sub>t t<sub>o ex</sub>t<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> f<sub>u</sub>ll th<sub>eore</sub>ti<sub>ca</sub>l b<sub>ene</sub>fit<sub>s o</sub>f th<sub>e</sub> shifted Legendre polynomial. As such we need a training objective which ensures the algebra of the shifted Legen<sup>d</sup>re po<sup>l</sup>ynomia<sup>l</sup> is taug<sup>h</sup>t to t<sup>h</sup>e neura<sup>l</sup> networ<sup>k</sup>. N.B., we re<sup>f</sup>er t<sup>h</sup>e rea<sup>d</sup>er to an in-<sup>d</sup>ept<sup>h d</sup>iscussion on t<sup>h</sup>e diferent a<sub>pp</sub>roximations of the Brownian <sub>p</sub>ath in A<sub>pp</sub>endix B.

## 3.2 Training objectives for stochastic flow maps

Th<sub>ere</sub>f<sub>ore, we requ</sub>i<sub>re a</sub> l<sub>oss</sub> th<sub>a</sub>t t<sub>eac</sub>h<sub>es</sub> th<sub>e neura</sub>l <sub>ne</sub>t<sub>wor</sub>k <sub>a</sub>b<sub>ou</sub>t th<sub>e</sub> B<sub>rown</sub>i<sub>an pa</sub>th<sub>.</sub> Whil<sub>e ear</sub>li<sub>er wor</sub>k<sub>s use</sub>d the semigroup objective, the most performant objectives for deterministic flow maps are the derivative-based Eulerian losses (Gen<sub>g</sub>, Lu, et al. 2025; Sabour, Fidler, et al. 2025); however, as noted in Pan et al. (2026, Pro<sub>p</sub>osition 2) the Eulerian loss fails to teach the neural network about the Brownian <sub>p</sub>ath when naïvel<sub>y</sub> a<sub>pp</sub>lied. Thus we reach our second main desi<sub>g</sub>n <sub>p</sub>rinci<sub>p</sub>le:

Principle 2. Choose an objective which ensures that the neural network learns to use the path.

We will discuss three possible objectives beyond the semigroup loss: 1) discretized Eulerian and Lagrangian conditions, 2) transformin<sub>g</sub> the SDE into a <sub>p</sub>athwise e<sub>q</sub>uivalent random ODE (Doss–Sussmann transform), and

3) usin<sub>g</sub> a famil<sub>y</sub> of random ODEs which conver<sub>g</sub>e <sub>p</sub>athwise to the SDE (Won<sub>g</sub>–Zakai ar<sub>g</sub>ument). For a more <sup>d</sup>etai<sup>l</sup>e<sup>d d</sup>iscussion consu<sup>l</sup>t Appen<sup>d</sup>ix C, qq.v.

Discretized Eulerian and Lagrangian conditions. Early training with consistency models used a discretized E<sub>u</sub>l<sub>e</sub>ri<sub>a</sub>n <sub>co</sub>nditi<sub>o</sub>n with th<sub>e</sub> t<sub>e</sub>rmin<sub>a</sub>l tim<sub>e</sub> fix<sub>e</sub>d t<sub>o</sub> 1 <sub>a</sub>nd <sub>s</sub>h<sub>o</sub>w<sub>e</sub>d th<sub>a</sub>t <sub>as</sub> th<sub>e s</sub>t<sub>ep s</sub>iz<sub>e</sub> in th<sub>e s</sub>m<sub>a</sub>ll ODE <sub>s</sub>t<sub>ep s</sub>hrink<sub>s</sub> to 0, one obtains the continuous-time objective (Song et al. 2023, Theorem $5 )$ <sub>.</sub> Th<sub>ese</sub> l<sub>osses are c</sub>l<sub>ose</sub>l<sub>y re</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>o</sub> the semi<sub>g</sub>rou<sub>p</sub> loss but use a small ODE ste<sub>p</sub> (often Euler) at the start or terminal time which corres<sub>p</sub>onds to the Eulerian and La<sub>g</sub>ran<sub>g</sub>ian conditions res<sub>p</sub>ectivel<sub>y</sub>. In the SDE case this small Euler ste<sub>p</sub> is re<sub>p</sub>laced with a small Euler–Maru<sub>y</sub>ama ste<sub>p</sub> at the start or terminal time <sub>p</sub>oint<sub>,</sub> which is enabled b<sub>y</sub> Theorem B.<sub>3</sub>. In our to<sub>y</sub> <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> w<sub>e</sub> f<sub>ou</sub>nd thi<sub>s</sub> l<sub>oss</sub> f<sub>o</sub>rm<sub>u</sub>l<sub>a</sub>ti<sub>o</sub>n t<sub>o ou</sub>t<sub>pe</sub>rf<sub>o</sub>rm th<sub>e s</sub>t<sub>a</sub>nd<sub>a</sub>rd <sub>se</sub>mi<sub>g</sub>r<sub>oup</sub> l<sub>oss,</sub> b<sub>u</sub>t in l<sub>a</sub>r<sub>ge</sub>r <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> f<sub>ou</sub>nd it t<sub>o s</sub>li<sub>g</sub>htl<sub>y u</sub>nd<sub>e</sub>r<sub>pe</sub>rf<sub>o</sub>rm it<sub>.</sub>

The Doss–Sussmann transform. One other strategy one could use to avoid taking time-derivatives of the Brownian <sub>p</sub>ath is to use a clever techni<sub>q</sub>ue <sub>p</sub>ro<sub>p</sub>osed inde<sub>p</sub>endentl<sub>y</sub> b<sub>y</sub> Doss $\left( 1 9 7 7 \right)$ and Sussmann $\left( 1 9 7 8 \right)$ k<sub>nown as</sub> th<sub>e</sub> D<sub>oss</sub>–S<sub>uss</sub>m<sub>a</sub>nn tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm whi<sub>c</sub>h tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>s ou</sub>r SDE int<sub>o a</sub>n <sub>equ</sub>iv<sub>a</sub>l<sub>e</sub>nt <sub>pa</sub>thwi<sub>se</sub> r<sub>a</sub>nd<sub>o</sub>m ODE<sub>.</sub><sup>4</sup> Sin<sub>ce</sub> our SDE in E<sub>q</sub>uation (1) has additive noise and the Itô inte<sub>g</sub>ral itself is a martin<sub>g</sub>ale <sub>p</sub>rocess, we can define th<sub>e</sub> m<sub>a</sub>rtin<sub>ga</sub>l<sub>e as</sub> $\begin{array} { r } { \pmb { M } _ { t } : = \int _ { t _ { n } } ^ { t } \mathrm { d } \pmb { W } _ { r } \ \nu _ { r } } \end{array}$ <sub>on</sub> th<sub>e</sub> ti<sub>me</sub> h<sub>or</sub>i<sub>zon</sub> $[ t _ { 0 } , 1 ]$ f<sub>or some sma</sub>ll $t _ { 0 } > 0 . { } ^ { 5 }$ N<sub>o</sub>w w<sub>e</sub> intr<sub>o</sub>d<sub>uce a</sub> new stochastic <sub>p</sub>rocess $\{ Z _ { t } \} _ { t \in [ t _ { 0 } , 1 ] }$ d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> $Z _ { t } : = X _ { t } - M _ { t }$ . With sim<sub>p</sub>le substitution into E<sub>q</sub>uation (1) we fi<sub>n</sub>d th<sub>a</sub>t th<sub>e</sub> d<sub>ynam</sub>i<sub>cs o</sub>f $\scriptstyle { Z _ { t } }$ are governe<sup>d b</sup>y an ODE, i.e., $\begin{array} { r } { \dot { \pmb { Z } } _ { t } = 2 \pmb { u } _ { t } ( \pmb { Z } _ { t } + \pmb { M } _ { t } ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { \star } } ( \pmb { Z } _ { t } + \pmb { M } _ { t } ) } \end{array}$ <sub>.</sub> R<sub>e</sub>m<sub>a</sub>rk<sub>a</sub>bl<sub>y,</sub> $\scriptstyle { Z _ { t } }$ is diferentiable in time, un<sup>l</sup>i<sup>k</sup>e t<sup>h</sup>e roug<sup>h</sup> processes $X _ { t }$ <sub>a</sub>nd ${ \mathbf { } } W _ { t }$ (its derivative inherits the rou<sub>g</sub>hness of $M _ { t } )$ Thus one could write a flow map in �-space as $\Upsilon _ { s , t } ^ { \theta } ( z ; \mathcal { T } _ { s , t } ) = z + ( t - s ) u _ { s , t } ^ { \theta } ( z + M _ { s } ; \mathcal { T } _ { s , t } )$ . This Doss–Sussmann techni<sub>q</sub>ue can be seen as a <sub>g</sub>eneralization of the <sub>p</sub>arameterization considered in Pan et al. (2026). We <sub>p</sub>rovide an in-de<sub>p</sub>th anal<sub>y</sub>sis of this loss in A<sub>pp</sub>endix C.2.

A Wong–Zakai objective. Then is it possible to train with a derivative-style loss that uses all � orders of information about the Brownian <sub>p</sub>ath we <sub>p</sub>ass to the network? We <sub>p</sub>ro<sub>p</sub>ose a solution usin<sub>g</sub> a Won<sub>g</sub>–Zakai st<sub>y</sub>led ar<sub>g</sub>ument (Won<sub>g</sub> and Zakai 1965a,b; Stroock and Varadhan $1 9 7 2 )$ . Consider our SDE in E<sub>q</sub>uation (1); we can instead write it as another random ODE w.r.t. our <sub>p</sub>ol<sub>y</sub>nomial a<sub>pp</sub>roximation ${ W _ { r } } ^ { ( N , s , t ) }$ <sub>on</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l $[ s , t ] \subseteq [ 0 , 1 ]$ , i.e., a <sup>f</sup>ami<sup>l</sup>y o<sup>f</sup> ran<sup>d</sup>om ODEs w<sup>h</sup>ic<sup>h</sup> converge to our un<sup>d</sup>er<sup>l</sup>ying SDE, see Proposition $\mathrm { C } . 3 \cdot$ <sub>.</sub> It i<sub>s</sub> th<sub>us</sub> <sub>q</sub>uite strai<sub>g</sub>htforward to a<sub>pp</sub>l<sub>y</sub> the usual Eulerian or La<sub>g</sub>ran<sub>g</sub>ian losses<sub>;</sub> we <sub>p</sub>rovide more details in A<sub>pp</sub>endix $\mathrm { C } . 3 \AA$

## 3.3 Turning flow models into strong stochastic flow maps

Th<sub>e a</sub>bilit<sub>y</sub> t<sub>o</sub> l<sub>e</sub>v<sub>e</sub>r<sub>age po</sub>w<sub>e</sub>rf<sub>u</sub>l <sub>p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> f<sub>o</sub>r v<sub>a</sub>ri<sub>ous</sub> d<sub>o</sub>wn<sub>s</sub>tr<sub>ea</sub>m t<sub>as</sub>k<sub>s</sub> i<sub>s a</sub> k<sub>ey app</sub>li<sub>ca</sub>ti<sub>o</sub>n f<sub>o</sub>r <sub>u</sub>nl<sub>oc</sub>kin<sub>g</sub> ML fr<sub>a</sub>m<sub>e</sub>w<sub>o</sub>rk<sub>s</sub> f<sub>o</sub>r <sub>so</sub>lvin<sub>g p</sub>r<sub>o</sub>bl<sub>e</sub>m<sub>s a</sub>t <sub>sca</sub>l<sub>e.</sub> H<sub>e</sub>r<sub>e</sub> w<sub>e p</sub>r<sub>opose a</sub> fin<sub>e</sub>-t<sub>u</sub>nin<sub>g</sub> m<sub>e</sub>th<sub>o</sub>d t<sub>o</sub> t<sub>a</sub>k<sub>e a p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> m<sub>o</sub>d<sub>e</sub>l ${ \pmb u } _ { t } ^ { \theta }$ (or deterministic flow ma<sub>p</sub>) and convert it into its own Itô ma<sub>p</sub> $\Psi _ { s , t } ^ { \theta }$ of the solution in E<sub>q</sub>uation (1). This ofers a clear benefit over trainin<sub>g</sub> from scratch allowin<sub>g</sub> us to train lar<sub>g</sub>er models at a much smaller compute <sup>b</sup>u<sup>d</sup>get. Fo<sup>ll</sup>owing Lee et a<sup>l</sup>. (2026) we wou<sup>ld l</sup>i<sup>k</sup>e to en<sup>f</sup>orce a boundary condition t<sup>h</sup>at t<sup>h</sup>e <sup>d</sup>iagona<sup>l</sup> matches the <sub>p</sub>re-trained teacher via this architectural <sub>p</sub>arameterization<sub>,</sub> but in the stochastic settin<sub>g,</sub> leadin<sub>g</sub> to:

Principle 3. Choose a parameterization such that the drift matches the pre-trained flow model.

Improved parameterization of the strong stochastic flow map. For the flow SDE in Equation (1) this boundary condition is not <sub>q</sub>uite as sim<sub>p</sub>le as the drift is $\begin{array} { r } { 2 { \pmb u } _ { t } ^ { \theta } ( { \pmb x } ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \mathfrak { z } } \end{array}$ � which has the additional linear term $\begin{array} { r } { \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { x } . } \end{array}$ <sub>.</sub> W<sub>e cou</sub>ld <sub>a</sub>dd <sub>a</sub> <sub>s</sub>im<sub>p</sub>l<sub>e</sub> lin<sub>ea</sub>r h<sub>ea</sub>d t<sub>o</sub> th<sub>e</sub> n<sub>eu</sub>r<sub>a</sub>l n<sub>e</sub>tw<sub>o</sub>rk <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> b<sub>u</sub>t th<sub>e</sub>n w<sub>e</sub> n<sub>ee</sub>d t<sub>o</sub> l<sub>ea</sub>rn th<sub>e</sub> int<sub>eg</sub>r<sub>a</sub>l <sub>o</sub>f $\begin{array} { r } { \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { x } } \end{array}$ <sub>coup</sub>l<sub>e</sub>d with <sub>ou</sub>r <sub>p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d drift<sub>,</sub> whi<sub>c</sub>h i<sub>s</sub> n<sub>o</sub>t <sub>o</sub>nl<sub>y co</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>ll<sub>y</sub> h<sub>a</sub>rd<sub>e</sub>r t<sub>o</sub> l<sub>ea</sub>rn<sub>,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> m<sub>a</sub>k<sub>es</sub> th<sub>e</sub> SDE stif (Lu et al. 2022), $i . e .$ <sub>,</sub> h<sub>a</sub>rd t<sub>o</sub> di<sub>s</sub>till <sub>ou</sub>t<sub>.</sub> Additi<sub>o</sub>n<sub>a</sub>ll<sub>y,</sub> f<sub>o</sub>r <sub>co</sub>mm<sub>o</sub>n <sub>c</sub>h<sub>o</sub>i<sub>ces o</sub>f n<sub>o</sub>i<sub>se sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>es</sub> thi<sub>s</sub> lin<sub>ea</sub>r term diver<sub>g</sub>es to infinit<sub>y</sub> as $t  0$ which makes matchin<sub>g</sub> the drift <sub>q</sub>uite com<sub>p</sub>licated. However<sub>,</sub> b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> ex<sub>p</sub>onential inte<sub>g</sub>rators (Hochbruck and Ostermann 2010) to the SDE we can sim<sub>p</sub>lif<sub>y</sub> man<sub>y</sub> of these desi<sub>g</sub>n <sub>p</sub>r<sub>o</sub>bl<sub>e</sub>m<sub>s.</sub> $I . e . _ { \cdot }$ we rewrite E<sub>q</sub>uation (1) to be

$$
\mathrm { d } \big [ \alpha _ { t } \boldsymbol { X } _ { t } \big ] = 2 \alpha _ { t } \boldsymbol { u } _ { t } \big ( \boldsymbol { X } _ { t } \big ) \ \mathrm { d } t + \alpha _ { t } \boldsymbol { \nu } _ { t } \ \mathrm { d } { \boldsymbol { W } _ { t } } ,\tag{4}
$$

<sub>w</sub>hi<sub>c</sub>h h<sub>as now remove</sub>d th<sub>e</sub> li<sub>near</sub> t<sub>erm</sub> f<sub>rom</sub> th<sub>e</sub> $\mathrm { S D E } ^ { 6 }$ Withi<sub>n</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>o</sub>f thi<sub>s exponen</sub>ti<sub>a</sub>l lift <sub>we now</sub> h<sub>ave a</sub> new <sub>p</sub>arameterization of the flow ma<sub>p</sub> which we can write as the ex<sub>p</sub>onentiall<sub>y</sub> wei<sub>g</sub>hted Euler–Maru<sub>y</sub>ama ste<sub>p</sub> <sub>o</sub>f th<sub>e</sub> f<sub>orm</sub>

$$
\Psi _ { s , t } ^ { \theta } ( \pmb { x } ; T _ { s , t } ) = \frac { \alpha _ { s } } { \alpha _ { t } } \pmb { x } + 2 \frac { \mathcal { A } _ { s , t } } { \alpha _ { t } } \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ; T _ { s , t } ) + \frac { 1 } { \alpha _ { t } } ( \pmb { W } _ { \langle \pmb { M } ^ { \alpha } \rangle _ { t } } - \pmb { W } _ { \langle \pmb { M } ^ { \alpha } \rangle _ { s } } ) ,\tag{5}
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { \mathcal { A } _ { s , t } : = \int _ { s } ^ { t } \mathrm { d } r ~ \alpha _ { r } } \end{array}$ <sub>a</sub>nd $\begin{array} { r } { M _ { t } ^ { \alpha } : = \int _ { t _ { 0 } } ^ { t } \mathrm { d } W _ { r } \ \alpha _ { r } \nu _ { r } } \end{array}$ <sub>a</sub>nd $\langle M ^ { \alpha } \rangle _ { i }$ <sub>�</sub> i<sub>s</sub> it<sub>s qua</sub>dr<sub>a</sub>ti<sub>c</sub> v<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> N<sub>o</sub>t<sub>a</sub>bl<sub>y,</sub> th<sub>ese qua</sub>ntiti<sub>es</sub> <sub>ca</sub>n b<sub>e</sub> writt<sub>e</sub>n in <sub>c</sub>l<sub>ose</sub>d f<sub>o</sub>rm f<sub>o</sub>r <sub>co</sub>mm<sub>o</sub>n <sub>c</sub>h<sub>o</sub>i<sub>ces o</sub>f n<sub>o</sub>i<sub>se sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>es,</sub> A<sub>ppe</sub>ndix $\mathrm { B . } 4 ~ q q . \nu .$ On th<sub>e</sub> di<sub>ago</sub>n<sub>a</sub>l th<sub>e</sub> drift <sub>o</sub>f thi<sub>s pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub> $\begin{array} { r } { 2 { \pmb u } _ { t , t } ^ { \theta } ( { \pmb x } ; { \mathbf 0 } ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } { \pmb x } , } \end{array}$ whi<sub>c</sub>h <sub>equa</sub>l<sub>s</sub> th<sub>e</sub> t<sub>eac</sub>h<sub>e</sub>r SDE drift if <sub>a</sub>nd <sub>o</sub>nl<sub>y</sub> if ${ \pmb u } _ { t , t } ^ { \theta } ( { \pmb x } ; { \mathbf 0 } ) = { \pmb u } _ { t } ^ { \theta } ( { \pmb x } )$ the decoupled architecture of Lee et al. (2026) to<sub>g</sub>ether with our (� − �)-<sub>g</sub>ated coeficient injection makes this <sub>e</sub>x<sub>ac</sub>t <sub>a</sub>t initi<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> M<sub>o</sub>r<sub>eo</sub>v<sub>e</sub>r<sub>,</sub> th<sub>e</sub> Itô int<sub>eg</sub>r<sub>a</sub>l it<sub>se</sub>lf i<sub>s a</sub> m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> whi<sub>c</sub>h i<sub>s a</sub> tim<sub>e</sub>-<sub>c</sub>h<sub>a</sub>n<sub>ge</sub>d Br<sub>o</sub>wni<sub>a</sub>n m<sub>o</sub>ti<sub>o</sub>n and thus we construct our Brownian a<sub>pp</sub>roximation around this <sub>p</sub>ath<sub>,</sub> $i . e . , \mathcal { I } _ { s , t }$ i<sub>s</sub> b<sub>u</sub>ilt fr<sub>o</sub>m $W _ { \langle M ^ { \alpha } \rangle _ { t } }$ <sub>.</sub> Thi<sub>s</sub> l<sub>as</sub>t detail is im<sub>p</sub>ortant: if the coeficients of the a<sub>pp</sub>roximation are constructed from the ori<sub>g</sub>inal Brownian <sub>p</sub>ath<sub>,</sub> t<sup>h</sup>en one must use numerica<sup>l</sup> qua<sup>d</sup>rature to approximate t<sup>h</sup>e Itô integra<sup>l</sup> rat<sup>h</sup>er t<sup>h</sup>an computing it exactly as we <sub>can v</sub>i<sub>a our c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> f<sub>ormu</sub>l<sub>æ.</sub> Thi<sub>s c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm mar</sub>ti<sub>nga</sub>l<sub>e a</sub>l<sub>so ena</sub>bl<sub>es us</sub> t<sub>o use an</sub> i<sub>ns</sub>t<sub>an</sub>t<sub>aneous</sub> d<sub>r</sub>ift matchin<sub>g</sub> loss unlike re<sub>g</sub>ressin<sub>g</sub> to a small Euler–Maru<sub>y</sub>ama ste<sub>p</sub> as used in McCallum et al. (2026) due to the <sup>l</sup>earne<sup>d d</sup>i<sup>f</sup>usion <sup>h</sup>ea<sup>d</sup>.<sup>7</sup> T<sup>h</sup>us we can <sup>b</sup>rea<sup>k d</sup>own t<sup>h</sup>is improve<sup>d</sup> parameterization into three c<sup>h</sup>anges <sup>f</sup>rom SSFMs: 1) exponential lift of the stochastic flow map, 2) closed-form martingale of the Brownian path, and 3) drift matching loss versus the small Euler–Maruyama step regression.

## 4 Experiments

In this section we em<sub>p</sub>iricall<sub>y</sub> evaluate our im<sub>p</sub>roved trainin<sub>g</sub> reci<sub>p</sub>e for stron<sub>g</sub> stochastic flow ma<sub>p</sub>s first on to<sub>y</sub> pro<sup>bl</sup>ems, t<sup>h</sup>en <sup>d</sup>isti<sup>ll</sup>ing a <sup>l</sup>arge pre-traine<sup>d</sup> Decoupled MeanFlow (DMF) (Lee et a<sup>l</sup>. 2026), viz., $\mathrm { D M F } { \cdot } \mathrm { X L } / 2 { + }$ , into a stron<sub>g</sub> stochastic flow ma<sub>p</sub> for <sub>p</sub>erformin<sub>g</sub> class conditional sam<sub>p</sub>lin<sub>g</sub> of E<sub>q</sub>uation (1), and finall<sub>y</sub> demonstratin<sub>g</sub> some new a<sub>pp</sub>lications of stron<sub>g</sub> stochastic flow ma<sub>p</sub>s with Fe<sub>y</sub>nman–Kac correctors (Skreta, Akhound-Sade<sub>g</sub>h, et al. 2025). We observe that <sub>p</sub>airin<sub>g</sub> the shifted Le<sub>g</sub>endre <sub>p</sub>ol<sub>y</sub>nomial with the Won<sub>g</sub>–Zakai formulation trained with an im<sub>p</sub>roved MeanFlow loss (Gen<sub>g</sub>, Lu, et al. 2025) and usin<sub>g</sub> our im<sub>p</sub>roved <sub>p</sub>arameterization results in the stron<sub>g</sub>est model overall; we refer to such a model as the IMPROVED STRONG STOCHASTIC FLOW MAP (iSSFM).

## 4.1 Toy experiments

Comparing noise representations. First we set up a simple 2D flow matching problem of mapping the unit Gaussian to a checkerboard distribution. We com<sub>p</sub>are the shifted Le<sub>g</sub>endre coeficients (McCallum et al. 2026), passing multiple first-order increments over the interval [�, �], using the first-order increment over the whole interval [�, �] <sub>p</sub>lus the <sub>g</sub>lobal Karhunen–Loève (KL) coeficients (Pan et al. 2026), and just usin<sub>g</sub> the KL coeficients to iso<sup>l</sup>ate t<sup>h</sup>e e<sup>f</sup>ect o<sup>f</sup> t<sup>h</sup>e a<sup>dd</sup>e<sup>d</sup> increment. We measure t<sup>h</sup>e strong so<sup>l</sup>ution root mean squared error (RMSE) w.r.t. a <sub>5</sub>12-ste<sub>p</sub> Euler–Maru<sub>y</sub>ama inte<sub>g</sub>ration of the model’s own dia<sub>g</sub>onal $\pmb { u } _ { t , t } ^ { \theta }$ and re<sub>p</sub>ort the 2-Wasserstein di<sub>s</sub>t<sub>a</sub>n<sub>ce</sub> b<sub>e</sub>tw<sub>ee</sub>n <sub>409</sub>6 <sub>sa</sub>m<sub>p</sub>l<sub>es</sub> fr<sub>o</sub>m th<sub>e ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d <sub>a</sub>nd tr<sub>ue c</sub>h<sub>ec</sub>k<sub>e</sub>rb<sub>oa</sub>rd di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> All fl<sub>o</sub>w m<sub>aps a</sub>r<sub>e</sub> tr<sub>a</sub>in<sub>e</sub>d with the semi<sub>g</sub>rou<sub>p</sub> loss used b<sub>y</sub> McCallum et al. (2026); we re<sub>p</sub>ort these results in Fi<sub>g</sub>ure 3a. Outside of the 1-NFE <sub>case</sub> wh<sub>e</sub>r<sub>e</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l KL inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub> w<sub>e</sub>ll-<sub>pose</sub>d<sub>,</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l KL <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s</sub> with <sub>a</sub>nd with<sub>ou</sub>t th<sub>e</sub> l<sub>oca</sub>l in<sub>c</sub>r<sub>e</sub>m<sub>e</sub>nt <sub>su</sub>f<sub>e</sub>r <sub>a</sub>nd f<sub>a</sub>il t<sub>o p</sub>r<sub>ope</sub>rl<sub>y</sub> im<sub>pa</sub>rt <sub>use</sub>f<sub>u</sub>l <sub>s</sub>i<sub>g</sub>n<sub>a</sub>l t<sub>o</sub> th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l b<sub>eyo</sub>nd th<sub>e</sub> fir<sub>s</sub>t-<sub>o</sub>rd<sub>e</sub>r in<sub>c</sub>r<sub>e</sub>m<sub>e</sub>nt<sub>.</sub> Th<sub>e</sub> <sub>s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f<sub>e</sub>r<sub>s</sub> m<sub>o</sub>r<sub>e</sub> inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n t<sub>o</sub> th<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>aps.</sub>

![](images/23957634914d9be65c21bee690336857c99d98b95a146ba9785c554c879e6fba.jpg)  
(a) Impact of Brownian representations (2D checkerboard).

![](images/f952767ffb143333c5c78826dc91586c27283bada649d41577d800439fdc170b.jpg)  
(b) Impact of training losses (1D SDE).  
Figure 3 Toy experiments. (a) Strong RMSE w.r.t. the number of visible coeficients � for each Brownian representation on the 2D checkerboard problem. (b) One-step strong error w.r.t. � for each training loss on the 1D non-linear SDE; labels give the ratio of the error at �=1 to �=4.

Comparing losses. Now we compare methods for learning to distill the drift of the SDE into a strong stochastic flow ma<sub>p</sub>. We com<sub>p</sub>are the semi<sub>g</sub>rou<sub>p</sub> loss used b<sub>y</sub> McCallum et al. (2026), the discrete Eulerian (dESD), discrete La<sub>g</sub>ran<sub>g</sub>ian (dLSD), the Doss–Sussmann (DS) transformation with the Eulerian loss from Sabour, Fidler, et al. (2025) (AYF-EMD) and im<sub>p</sub>roved MeanFlow loss (Gen<sub>g</sub>, Lu, et al. 2025) (iMF), and the Won<sub>g</sub>–Zakai (WZ) framework with the AYF-EMD and iMF losses. Followin<sub>g</sub> McCallum et al. (2026), we simulate the unstable non-linear SDE described b<sub>y</sub> $\mathrm { d } X _ { t } = [ X _ { t } - X _ { t } ^ { 3 } ]$ d� + � d�<sub>�</sub> and measure the strong pathwise error to a 512-step E<sub>u</sub>l<sub>e</sub>r–M<sub>a</sub>r<sub>uya</sub>m<sub>a</sub> r<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce.</sub> W<sub>e</sub> tr<sub>a</sub>in f<sub>o</sub>r <sub>5</sub>k tr<sub>a</sub>inin<sub>g s</sub>t<sub>eps</sub> with <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>iz<sub>e o</sub>f <sub>4</sub>0<sub>9</sub>6 <sub>a</sub>nd <sub>use</sub> th<sub>e s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e</sub> r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r <sub>a</sub>ll r<sub>u</sub>n<sub>s.</sub> In Fi<sub>gu</sub>r<sub>e 3</sub>b w<sub>e o</sub>b<sub>se</sub>rv<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> D<sub>oss</sub>–S<sub>uss</sub>m<sub>a</sub>nn m<sub>o</sub>d<sub>e</sub>l<sub>s a</sub>t 1 NFE f<sub>a</sub>il t<sub>o</sub> m<sub>a</sub>k<sub>e use o</sub>f th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s a</sub>t <sub>any</sub> d<sub>egree, mean</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>oss</sub> d<sub>oes no</sub>t <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> t<sub>eac</sub>h th<sub>e mo</sub>d<sub>e</sub>l h<sub>ow</sub> t<sub>o use</sub> th<sub>em.</sub> C<sub>onverse</sub>l<sub>y,</sub> th<sub>e</sub> di<sub>sc</sub>r<sub>e</sub>t<sub>e a</sub>nd W<sub>o</sub>n<sub>g</sub>–Z<sub>a</sub>k<sub>a</sub>i l<sub>osses</sub> d<sub>o</sub> t<sub>eac</sub>h th<sub>e</sub> n<sub>e</sub>tw<sub>o</sub>rk h<sub>o</sub>w t<sub>o use</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s,</sub> with th<sub>e</sub> iMF v<sub>a</sub>ri<sub>a</sub>nt <sub>o</sub>f th<sub>e</sub> W<sub>on –</sub>Z<sub>a</sub>k<sub>a</sub>i l<sub>oss er</sub>f<sub>orm</sub>i<sub>n</sub> th<sub>e</sub> b<sub>es</sub>t <sub>overa</sub>ll<sub>,</sub> b<sub>e</sub>i<sub>n</sub> th<sub>e mos</sub>t <sub>sens</sub>iti<sub>ve</sub> t<sub>o</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>no</sub>i<sub>se coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> W<sub>e a</sub>l<sub>so</sub> tr<sub>a</sub>in<sub>e</sub>d L<sub>ag</sub>r<sub>a</sub>n<sub>g</sub>i<sub>a</sub>n v<sub>a</sub>ri<sub>a</sub>nt<sub>s,</sub> b<sub>u</sub>t f<sub>ou</sub>nd th<sub>e</sub>m t<sub>o</sub> b<sub>e</sub> hi<sub>g</sub>hl<sub>y u</sub>n<sub>s</sub>t<sub>a</sub>bl<sub>e</sub> in b<sub>o</sub>th th<sub>e</sub> D<sub>oss</sub>–S<sub>uss</sub>m<sub>a</sub>nn <sub>a</sub>nd Won<sub>g</sub>–Zakai settin<sub>g</sub>s<sub>;</sub> we re<sub>p</sub>ort the full results in the a<sub>pp</sub>endix in Table <sub>4</sub>.

## 4.2 Fine-tuning deterministic flow models into strong stochastic flow maps

In these ex<sub>p</sub>eriments we take a <sub>p</sub>re-trained DMF-XL/2+ (Lee et al. 2026)—itself distilled from a <sub>p</sub>re-trained SiT-XL/2+REPA (Yu et al. 2025)—and turn it into a stron<sub>g</sub> stochastic flow ma<sub>p</sub>. We fine-tune each model accordin<sub>g</sub> to Section <sub>3</sub>.<sub>3</sub> for 100k trainin<sub>g</sub> iterations. For all variants and re<sub>p</sub>lications<sub>,</sub> we add the Brownian coeficients of the s<sup>h</sup>i<sup>f</sup>te<sup>d</sup> Legen<sup>d</sup>re po<sup>l</sup>ynomia<sup>l</sup>, mutatis mutandis <sup>f</sup>or ot<sup>h</sup>er representations, <sup>f</sup>o<sup>ll</sup>owing Appen<sup>d</sup>ix D q.v. Figure 12. Unless stated otherwise, all runs used � = 3 for the representation of the Brownian path.

Table 1 Improved parameterization. Improvements from changing the parameterization and diagonal loss at various NFE.
<table><tr><td rowspan="2">Method</td><td colspan="5">FID (↓) at NFE</td><td colspan="5"> $\mathbf { F D _ { D I N O v 2 } ( \downarrow ) }$  at NFE</td></tr><tr><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>SSFM (McCallum et al. 2026)</td><td>197.11</td><td>83.56</td><td>31.00</td><td>11.09</td><td>4.82</td><td>2116.9</td><td>1061.3</td><td>475.1</td><td>229.1</td><td>135.6</td></tr><tr><td>+ exponential lift</td><td>208.25</td><td>90.16</td><td>28.15</td><td>8.07</td><td>3.16</td><td>2112.2</td><td>1113.1</td><td>447.4</td><td>190.8</td><td>104.0</td></tr><tr><td>+ closed-form martingale</td><td>163.56</td><td>60.21</td><td>18.33</td><td>4.95</td><td>2.98</td><td>1561.2</td><td>723.0</td><td>301.6</td><td>125.7</td><td>72.6</td></tr><tr><td>+ drift matching</td><td>184.74</td><td>57.99</td><td>14.67</td><td>4.15</td><td>2.03</td><td>1817.6</td><td>772.3</td><td>288.1</td><td>128.2</td><td>70.8</td></tr></table>

Improved parameterization. We ablate through a series of improvements to the parameterization of strong stochastic flow ma<sub>p</sub>s in this stud<sub>y,</sub> culminatin<sub>g</sub> in our im<sub>p</sub>roved <sub>p</sub>arameterization from Section <sub>3</sub>.<sub>3</sub>. We be-<sub>g</sub>in with th<sub>e p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d DMF-XL/2+ <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>int <sub>a</sub>nd <sub>a</sub>dd <sub>a s</sub>m<sub>a</sub>ll dif<sub>us</sub>i<sub>o</sub>n h<sub>ea</sub>d t<sub>o</sub> th<sub>e o</sub>ri<sub>g</sub>in<sub>a</sub>l b<sub>ac</sub>kb<sub>o</sub>n<sub>e,</sub> followin<sub>g</sub> McCallum et al. (2026). We do the whole <sub>p</sub>arameterization w.r.t. the non-lifted SDE described in Equation (1); this entry forms our first row in Table 1, i.e., the most faithful replication of SSFM in our setting of fin<sub>e</sub>-t<sub>u</sub>nin<sub>g</sub> fr<sub>o</sub>m <sub>a p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d l<sub>a</sub>r<sub>ge</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> m<sub>o</sub>d<sub>e</sub>l<sub>.</sub> W<sub>e</sub> th<sub>e</sub>n <sub>a</sub>dd in <sub>ou</sub>r im<sub>p</sub>r<sub>o</sub>v<sub>e</sub>m<sub>e</sub>nt<sub>s</sub> in th<sub>e</sub> f<sub>o</sub>ll<sub>o</sub>win<sub>g</sub> <sub>o</sub>rd<sub>e</sub>r<sub>:</sub> th<sub>e e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>l lift<sub>,</sub> th<sub>e c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm m<sub>a</sub>rtin<sub>ga</sub>l<sub>e o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th<sub>, a</sub>nd tr<sub>a</sub>inin<sub>g</sub> vi<sub>a</sub> drift m<sub>a</sub>t<sub>c</sub>hin<sub>g.</sub>

We eva<sup>l</sup>uate in terms o<sup>f</sup> t<sup>h</sup>e Fréchet Inception Distance (FID) (Heuse<sup>l</sup> et al. 2017) and $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ (Stein et al. 2023) in Table 1. We observe a consistent im<sub>p</sub>rovement across our im<sub>p</sub>rovements with the ex<sub>p</sub>onential lift and closed-form martin<sub>g</sub>ale <sub>p</sub>rovidin<sub>g</sub> a substantial im<sub>p</sub>rovement. Notabl<sub>y,</sub> the order in which we chan<sub>g</sub>e the <sub>p</sub>arameterization is not arbitrar<sub>y;</sub> th<sub>e</sub> <sub>c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> i<sub>s</sub> <sub>qu</sub>it<sub>e</sub> <sub>u</sub>nwi<sub>e</sub>ld<sub>y</sub> with<sub>ou</sub>t th<sub>e</sub> <sub>e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>l <sup>l</sup>i<sup>f</sup>t (q.v. Appen<sup>d</sup>ix B.4) as it ten<sup>d</sup>s to in<sup>fi</sup>nite variance, causing numeri<sub>ca</sub>l <sub>p</sub>r<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n i<sub>ssues,</sub> <sub>e</sub>v<sub>e</sub>n th<sub>oug</sub>h t<sub>e</sub>rm<sub>s</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>ca</sub>n<sub>ce</sub>l <sub>ou</sub>t in th<sub>eo</sub>r<sub>y.</sub> In <sub>ou</sub>r r<sub>u</sub>n<sub>s,</sub> tr<sub>a</sub>inin<sub>g</sub> with th<sub>e c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> b<sub>u</sub>t with<sub>ou</sub>t th<sub>e</sub> ex<sub>p</sub>onential lift diver<sub>g</sub>ed com<sub>p</sub>letel<sub>y</sub> into <sub>p</sub>ure noise<sub>,</sub> unable to recover <sup>d</sup>ue to numerics. T<sup>h</sup>us t<sup>h</sup>e exponentia<sup>l l</sup>i<sup>f</sup>t allows us to now use t<sup>h</sup>e <sub>c</sub>l<sub>ose</sub>d-f<sub>o</sub>rm m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> <sub>a</sub>nd drift m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> f<sub>o</sub>rm<sub>s</sub> in thi<sub>s</sub> <sub>se</sub>ttin<sub>g.</sub> Thi<sub>s</sub> im<sub>p</sub>r<sub>o</sub>v<sub>e</sub>d <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>a</sub>ti<sub>o</sub>n will <sub>se</sub>rv<sub>e</sub> <sub>as</sub> <sub>ou</sub>r l<sub>au</sub>n<sub>c</sub>h <sub>pa</sub>d f<sub>o</sub>r <sub>a</sub>bl<sub>a</sub>tin<sub>g</sub> desi<sub>g</sub>n decisions on the distillation loss and re<sub>p</sub>resentation of the Brownian <sub>p</sub>ath.

Table 2 Preliminary sweep: FID-50k (↓) on Ima<sub>g</sub>eNet-2<sub>5</sub>6 at <sub>4</sub> NFE.
<table><tr><td>Loss</td><td>Path</td><td>FID (↓)</td></tr><tr><td>Semigroup</td><td>Legendre</td><td>14.68</td></tr><tr><td>Semigroup</td><td>KL + inc.</td><td>24.92</td></tr><tr><td>dESD</td><td>Legendre</td><td>15.95</td></tr><tr><td>dLSD</td><td>Legendre</td><td>23.81</td></tr><tr><td>WZ-LMD</td><td>Legendre</td><td>214.91</td></tr><tr><td>WZ-iMF</td><td>Legendre</td><td>6.75</td></tr></table>

Preliminary objective sweep. Before scaling up, we r<sub>a</sub>n <sub>a s</sub>m<sub>a</sub>ll<sub>e</sub>r <sub>sc</sub>r<sub>ee</sub>nin<sub>g s</sub>w<sub>eep o</sub>v<sub>e</sub>r di<sub>s</sub>till<sub>a</sub>ti<sub>o</sub>n l<sub>osses</sub> and Brownian re<sub>p</sub>resentations to decide which confi<sub>g</sub>uration to <sub>p</sub>ursue (Table 2). These runs use eval th<sub>e</sub> b<sub>es</sub>t li<sub>ve</sub> <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t <sub>o</sub>f th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>so</sub> th<sub>ey</sub> <sub>s</sub>h<sub>ou</sub>ld b<sub>e co</sub>m<sub>pa</sub>r<sub>e</sub>d <sub>o</sub>nl<sub>y</sub> with <sub>eac</sub>h <sub>o</sub>th<sub>e</sub>r <sub>a</sub>nd n<sub>o</sub>t dir<sub>ec</sub>tl<sub>y</sub> with Table <sub>3</sub>. The trends a<sub>g</sub>ree with our to<sub>y</sub> ex<sub>p</sub>erim<sub>e</sub>nt<sub>s:</sub> th<sub>e</sub> W<sub>o</sub>n<sub>g</sub>–Z<sub>a</sub>k<sub>a</sub>i iMF l<sub>oss</sub> i<sub>s</sub> th<sub>e c</sub>l<sub>ea</sub>r winn<sub>e</sub>r<sub>,</sub> the Karhunen–Loève re<sub>p</sub>resentation under<sub>p</sub>erforms th<sub>e s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e o</sub>n<sub>e,</sub> th<sub>e</sub> di<sub>sc</sub>r<sub>e</sub>tiz<sub>e</sub>d l<sub>osses</sub> d<sub>o</sub> n<sub>o</sub>t im<sub>p</sub>rove on the semi<sub>g</sub>rou<sub>p</sub> loss<sub>,</sub> and the La<sub>g</sub>ran<sub>g</sub>ian WZ-LMD <sup>l</sup>oss is unsta<sup>bl</sup>e, co<sup>ll</sup>apsing to <sup>bl</sup>urre<sup>d</sup> averaged images as noted in Sabour, Fidler, et al. (2025). We therefore com<sub>p</sub>are the semi<sub>g</sub>rou<sub>p</sub> and iMF losses in d<sub>e</sub>t<sub>a</sub>il b<sub>e</sub>l<sub>o</sub>w<sub>.</sub>

![](images/9b35e10f54b74ae7abf5e4d33538806c6c3b50f2b503897df96d2cc74a85cd74.jpg)  
Figure $4 ~ \mathrm { F D } _ { \mathrm { D I N O v } 2 } { - } 5 0 \mathrm { k }$ on Ima<sub>g</sub>eNet-2<sub>5</sub>6 for SSFM<sub>,</sub> SSFM with th<sub>e</sub> im<sub>p</sub>r<sub>o</sub>v<sub>e</sub>d <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>a</sub>ti<sub>o</sub>n<sub>, a</sub>nd iSSFM<sub>.</sub>

Table 3 Impact of improved training. Comparison of SSFM, SSFM with improved parameterization, and iSSFM on distributi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>p</sub>r<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> m<sub>e</sub>tri<sub>cs</sub>
<table><tr><td></td><td colspan="3">FID (↓)</td><td colspan="3"> $\mathbf { F D _ { D I N O v 2 } } \left( \downarrow \right)$ </td><td colspan="3">ImageReward (↑)</td><td colspan="3">PickScore (↑)</td><td colspan="3">HPSv2.1 (↑)</td></tr><tr><td></td><td></td><td>NFE SSFM + param.</td><td>iSSFM</td><td></td><td>SSFM + param.</td><td>iSSFM</td><td></td><td>SSFM + param.</td><td>iSSFM</td><td>SSFM + param.</td><td></td><td>iSSFM</td><td>SSFM + param.</td><td></td><td>iSSFM</td></tr><tr><td>1</td><td>197.11</td><td>184.74</td><td>111.61</td><td>2116.9</td><td>1817.6</td><td>994.2</td><td>-2.214</td><td>-2.204</td><td>-1.935</td><td>18.13</td><td>18.39</td><td>18.50</td><td>0.1506</td><td>0.1502</td><td>0.1483</td></tr><tr><td>2</td><td>83.56</td><td>57.99</td><td>53.11</td><td>1061.3</td><td>772.3</td><td>583.5</td><td>-1.532</td><td>-1.333</td><td>-1.328</td><td>18.69</td><td>19.09</td><td>18.91</td><td>0.1806</td><td>0.1819</td><td>0.1798</td></tr><tr><td>4</td><td>31.00</td><td>14.67</td><td>5.96</td><td>475.1</td><td>288.1</td><td>143.0</td><td>-0.872</td><td>-0.683</td><td>-0.453</td><td>19.33</td><td>19.75</td><td>19.99</td><td>0.2031</td><td>0.2060</td><td>0.2143</td></tr><tr><td>8</td><td>11.09</td><td>4.15</td><td>2.37</td><td>229.1</td><td>128.2</td><td>75.2</td><td>-0.493</td><td>-0.366</td><td>-0.219</td><td>19.78</td><td>20.15</td><td>20.35</td><td>0.2177</td><td>0.2218</td><td>0.2282</td></tr><tr><td>16</td><td>4.82</td><td>2.03</td><td>2.07</td><td>135.6</td><td>70.8</td><td>55.1</td><td>-0.312</td><td>-0.189</td><td>-0.127</td><td>20.04</td><td>20.40</td><td>20.49</td><td>0.2253</td><td>0.2317</td><td>0.2346</td></tr></table>

Comparing SSFM and iSSFM. Now we extend our

anal<sub>y</sub>sis from Table 1 and com<sub>p</sub>are SSFM (McCallum et al. 2026) to our SSFM with im<sub>p</sub>roved <sub>p</sub>arameterization and to our iSSFM (im<sub>p</sub>roved <sub>p</sub>arameterization + Won<sub>g</sub>–Zakai iMF loss). We evaluate on FID and $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ as before along with three human preference metrics, viz. ImageReward (Xu et al. 2023), PickScore (Kirstain et al. 2023), and HPSv2.1 (Wu, Hao, et al. 2023). We re<sub>p</sub>ort the results of these ex<sub>p</sub>eriments in Table 3. We observe that iSSFM o<sup>f</sup>ers a consistent an<sup>d</sup> genera<sup>l</sup> improvement across NFEs wit<sup>h</sup> a <sup>f</sup>ew caveats, viz. HPSv2.1 at 1 an<sup>d</sup> 2 NFE. In accordance with our earlier ablations (Table 1), the im<sub>p</sub>roved <sub>p</sub>arameterization trained with the semi<sub>g</sub>rou<sub>p</sub> loss still si<sub>g</sub>nificantl<sub>y</sub> im<sub>p</sub>roves the baseline SSFM with a rou<sub>g</sub>hl<sub>y</sub> 1.5× im<sub>p</sub>rovement in $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ f<sub>or</sub> ${ \mathrm { N F E } } \geq 4 ,$ and iSSFM im<sub>p</sub>roves this with a rou<sub>g</sub>hl<sub>y</sub> 3× im<sub>p</sub>rovement in $\mathrm { F D } _ { \mathrm { D I N O v } 2 } ;$ see Fi<sub>g</sub>ure <sub>4</sub> for a visual illustration. Thus we can see here that the switch from the semi<sub>g</sub>rou<sub>p</sub> loss to the iMF-st<sub>y</sub>led loss <sub>y</sub>ields a rou<sub>g</sub>hl<sub>y</sub> 2× increase over just our improved parameterization, yielding the headline roughly 3× improvement in distributional metrics. F<sub>u</sub>ll <sub>resu</sub>lt<sub>s are</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e 5.</sub>

![](images/f4a970e2ca89ee467d4da82cc0144a68cedca1dc60e09b419d8c34fa968ead81.jpg)  
(a) Rollouts at 4–16 NFE on one Brownian path per row.

![](images/57f6a97d48a0f3c33fa45131acd9061bae2e9864cd1148ac4569af9028604cdb.jpg)  
(b) Strong self-consistency.  
Figure 5 Strong solutions on ImageNet-256. (a) iSSFM samples at 4, 6, 8, 10, 12, 14, and 16 NFE (left to right); each row shares one realization of the Brownian path, so the images agree across NFE. (b) Latent MSE between steps of the flow map vs a 32-ste<sub>p</sub> solution.

Strong solution maps. Here we verify that our fine-tuned map is not only stochastic—this is trivially satisfied <sup>b</sup>y t<sup>h</sup>e Itô integra<sup>l</sup> <sup>b</sup>eing compute<sup>d</sup> in c<sup>l</sup>ose<sup>d</sup> <sup>f</sup>orm—<sup>b</sup>ut a<sup>l</sup>so t<sup>h</sup>at it is a strong map. In Figure 5<sup>b</sup> we report th<sub>e s</sub>tr<sub>o</sub>n<sub>g se</sub>lf-<sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>n<sub>cy e</sub>rr<sub>o</sub>r<sub>,</sub> $i . e .$ <sub>,</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t MSE <sub>o</sub>f <sub>a</sub> �<sub>-s</sub>t<sub>ep ro</sub>ll<sub>ou</sub>t <sub>aga</sub>i<sub>ns</sub>t th<sub>e same mo</sub>d<sub>e</sub>l’<sub>s own 32-s</sub>t<sub>ep</sub> <sub>so</sub>l<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>n th<sub>e sa</sub>m<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th<sub>, o</sub>v<sub>e</sub>r 1k im<sub>ages.</sub> W<sub>e co</sub>m<sub>pa</sub>r<sub>e</sub> t<sub>o</sub> th<sub>e</sub> SSFM with im<sub>p</sub>r<sub>o</sub>v<sub>e</sub>d <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>a</sub>ti<sub>o</sub>n trained with the semi<sub>g</sub>rou<sub>p</sub> loss to isolate the im<sub>p</sub>act the diferent losses have on the stron<sub>g p</sub>ro<sub>p</sub>ert<sub>y</sub>. We notice t<sup>h</sup>at <sup>b</sup>ot<sup>h</sup> maps possess t<sup>h</sup>e stron<sub>g</sub> propert<sub>y</sub> empirica<sup>ll</sup><sub>y</sub>, $i . e .$ <sub>,</sub> th<sub>e</sub> <sub>s</sub>tr<sub>o</sub>n<sub>g</sub> <sub>e</sub>rr<sub>o</sub>r t<sub>e</sub>nd<sub>s</sub> t<sub>o</sub> <sub>0</sub> <sub>as</sub> th<sub>e</sub> n<sub>u</sub>mb<sub>e</sub>r <sub>o</sub>f <sub>s</sub>t<sub>eps</sub> increases<sub>;</sub> however<sub>,</sub> the ma<sub>p</sub> trained with the semi<sub>g</sub>rou<sub>p</sub> loss exhibits stron<sub>g</sub>er <sub>p</sub>athwise consistenc<sub>y,</sub> at the <sub>cos</sub>t <sub>o</sub>f <sub>worse</sub> <sub>me</sub>t<sub>r</sub>i<sub>cs</sub> $c f .$ T<sub>a</sub>bl<sub>e</sub> <sub>3.</sub> W<sub>e</sub> <sub>a</sub>l<sub>so</sub> in<sub>c</sub>l<sub>u</sub>d<sub>e</sub> <sub>a</sub> w<sub>ea</sub>k b<sub>ase</sub>lin<sub>e</sub> wh<sub>e</sub>r<sub>e</sub>in w<sub>e</sub> r<sub>a</sub>nd<sub>o</sub>ml<sub>y</sub> <sub>sa</sub>m<sub>p</sub>l<sub>e</sub> th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n pat<sup>h</sup> eac<sup>h</sup> time <sup>f</sup>or every NFE; unsurprising<sup>l</sup>y, t<sup>h</sup>e strong error <sup>d</sup>oes not <sup>d</sup>ecrease, as suc<sup>h</sup> weak maps cannot <sub>p</sub>ossess the stron<sub>g</sub> <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub>. We demonstrate the stron<sub>g</sub> <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> of iSSFM <sub>q</sub>ualitativel<sub>y</sub> in Fi<sub>g</sub>ure <sub>5</sub>a across <sub>se</sub>v<sub>e</sub>r<sub>a</sub>l dif<sub>e</sub>r<sub>e</sub>nt <sub>c</sub>l<sub>ass co</sub>nditi<sub>o</sub>n<sub>s.</sub>

![](images/926001a406d98be0c022da9eca3a45cb099a796770ccab712046f8c889cddc7d.jpg)  
(a) Samples at 4 NFE.

![](images/c9cf5781feb39086606d640f35f4da1ca991fafc559cf7808ce2f16f1fdf664c.jpg)  
(b) Total variation vs. NFE.  
Figure 6 Feynman–Kac correctors on 40 Gaussians. (a) Starting distribution, iSSFM at 4 NFE, the ground truth, and iSSFM with FKC at 4 NFE. (b) Total variation to the target with and without FKC for iSSFM and a difusion SDE.

## 4.3 Feynman–Kac correctors

Fe<sub>y</sub>nman–Kac correctors (FKC) (Skreta, Akhound-Sade<sub>g</sub>h, et al. 2025) and the closel<sub>y</sub> related Fe<sub>y</sub>nman–Kac steering met<sup>h</sup>o<sup>d</sup> (Sing<sup>h</sup>a<sup>l</sup> et a<sup>l</sup>. 2025) samp<sup>l</sup>e <sup>f</sup>rom <sup>d</sup>istri<sup>b</sup>utions t<sup>h</sup>at are not t<sup>h</sup>e margina<sup>l</sup>s o<sup>f</sup> any sing<sup>l</sup>e traine<sup>d</sup> m<sub>o</sub>d<sub>e</sub>l<sub>, suc</sub>h <sub>as a</sub>nn<sub>ea</sub>l<sub>e</sub>d d<sub>e</sub>n<sub>s</sub>iti<sub>es</sub> $\pi _ { t } \propto p _ { t } ^ { \beta }$ or <sub>p</sub>ro<sup>d</sup>ucts o<sup>f</sup> ex<sub>p</sub>erts $\pi _ { t } \propto \prod _ { i } ( q _ { t } ^ { i } ) ^ { \beta _ { i } }$ . The<sub>y</sub> simulate an SDE driven b<sub>y</sub> th<sub>e co</sub>m<sub>pose</sub>d <sub>sco</sub>r<sub>e</sub> $\textstyle \sum _ { i } \beta _ { i } { \pmb s } _ { t } ^ { i }$ <sub>an</sub>d <sub>correc</sub>t th<sub>e m</sub>i<sub>sma</sub>t<sub>c</sub>h b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> l<sub>aw o</sub>f thi<sub>s proposa</sub>l <sub>an</sub>d th<sub>e</sub> t<sub>arge</sub>t <sub>w</sub>ith im<sub>p</sub>ortance wei<sub>g</sub>hts<sub>,</sub> which evolve as a <sub>p</sub>athwise inte<sub>g</sub>ral<sub>,</sub> which is followed b<sub>y</sub> resam<sub>p</sub>lin<sub>g</sub> on the lo<sub>g</sub> wei<sub>g</sub>hts $w _ { t }$ (q.v. Appen<sup>d</sup>ix E). Because $w _ { t }$ is an integra<sup>l</sup> along the particle’s own trajectory, t<sup>h</sup>e weig<sup>h</sup>t is t<sup>h</sup>us on<sup>l</sup>y meaning<sup>f</sup>u<sup>l</sup> i<sup>f</sup> t<sup>h</sup>e partic<sup>l</sup>e actua<sup>ll</sup>y <sup>f</sup>o<sup>ll</sup>owe<sup>d</sup> t<sup>h</sup>at trajectory. Hence, t<sup>h</sup>e corrector requires a strong map. We propose to l<sub>earn</sub> th<sub>e coup</sub>l<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> $i . e .$ , t<sup>h</sup>e <sub>p</sub>rocess $( X _ { t } , w _ { t } )$ jointly with one augmented map on $\mathbb { R } ^ { d }$ ⊕ ℝ, where the first com<sub>p</sub>onent is our ex<sub>p</sub>onentiall<sub>y</sub> lifted <sub>p</sub>arameterization from Section <sub>3</sub>.<sub>3</sub> a<sub>pp</sub>lied to the com<sub>p</sub>osed drift and the second learns the mean weight increment over [�,�] on the given Brownian path. Each step of the map now a<sup>d</sup>vances t<sup>h</sup>e partic<sup>l</sup>e and its <sup>l</sup>og-weig<sup>h</sup>t, so FKC resamp<sup>l</sup>ing can <sup>b</sup>e app<sup>l</sup>ie<sup>d</sup> at step <sup>b</sup>oun<sup>d</sup>aries wit<sup>h</sup> a <sup>f</sup>ew-step bud<sub>g</sub>et. This is similar in s<sub>p</sub>irit to the au<sub>g</sub>mented flow ma<sub>p</sub>s of Ai et al. (2026), who track the lo<sub>g</sub>-densit<sub>y</sub> of a deterministic flow<sub>;</sub> our ma<sub>p</sub> instead tracks a wei<sub>g</sub>ht alon<sub>g</sub> a stochastic <sub>p</sub>ath.

We follow Skreta, Akhound-Sade<sub>g</sub>h, et al. (2025) and anneal a 40-com<sub>p</sub>onent Gaussian mixture in 2D with $\beta { = } 3 ,$ <sub>co</sub>m<sub>pa</sub>rin<sub>g</sub> iSSFM <sub>aga</sub>in<sub>s</sub>t <sub>a</sub> dif<sub>us</sub>i<sub>o</sub>n SDE <sub>sa</sub>m<sub>p</sub>l<sub>e</sub>r int<sub>eg</sub>r<sub>a</sub>t<sub>e</sub>d with E<sub>u</sub>l<sub>e</sub>r–M<sub>a</sub>r<sub>uya</sub>m<sub>a, eac</sub>h with <sub>a</sub>nd with<sub>ou</sub>t FKC. In Fi<sub>g</sub>ure 6a iSSFM at <sub>4</sub> NFE alread<sub>y p</sub>laces mass on the ri<sub>g</sub>ht modes but with the wron<sub>g</sub> wei<sub>g</sub>hts<sub>;</sub> FKC redistributes mass towards the tar<sub>g</sub>et. Full metrics (W<sub>1</sub>, W<sub>2</sub>, MMD, ener<sub>gy</sub> distance) across cli<sub>pp</sub>in<sub>g</sub> levels are in Appen<sup>d</sup>ix H.3, q.v.; wit<sup>h</sup> 25% <sup>l</sup>og-weig<sup>h</sup>t c<sup>l</sup>ipping, FKC improves every metric except t<sup>h</sup>e energy <sup>d</sup>istance, w<sup>h</sup>ic<sup>h</sup> increases at most NFEs<sub>,</sub> likel<sub>y</sub> because the heav<sub>y</sub>-tailed wei<sub>g</sub>hts inflate its variance.

## 5 Conclusion

In this work we im<sub>p</sub>roved the trainin<sub>g</sub> of stron<sub>g</sub> stochastic flow ma<sub>p</sub>s b<sub>y</sub> cou<sub>p</sub>lin<sub>g</sub> the shifted-Le<sub>g</sub>endre re<sub>p</sub>resentation of the Brownian motion with a Wong–Zakai style improved MeanFlow objective, along with improving the <sub>p</sub>arameterization of the stochastic flow ma<sub>p</sub>. Our new fine-tunin<sub>g</sub> reci<sub>p</sub>e reduces Ima<sub>g</sub>eNet-2<sub>5</sub>6 FID and $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ relative to SSFMs at matched NFE<sub>,</sub> and matches the <sub>p</sub>erformance of 16-ste<sub>p</sub> SSFM with onl<sub>y</sub> four ste<sub>p</sub>s in FD<sub>DINOv2</sub>. We also demonstrated a novel a<sub>pp</sub>lication of stron<sub>g</sub> stochastic flow ma<sub>p</sub>s in creatin<sub>g</sub> few-ste<sub>p</sub> correctable sam<sub>p</sub>lers usin<sub>g</sub> amortized densities and Fe<sub>y</sub>nman–Kac correctors. B<sub>y</sub> unlockin<sub>g</sub> the <sub>p</sub>owerful losses <sub>use</sub>d b<sub>y</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>aps,</sub> w<sub>e</sub> n<sub>o</sub>w <sub>e</sub>nd<sub>o</sub>w <sub>s</sub>tr<sub>o</sub>n<sub>g s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>aps</sub> with th<sub>e a</sub>bilit<sub>y</sub> t<sub>o</sub> b<sub>e</sub> tr<sub>a</sub>in<sub>e</sub>d <sub>o</sub>n <sup>l</sup>ar<sub>g</sub>er an<sup>d</sup> more comp<sup>l</sup>ex s<sub>y</sub>stems, openin<sub>g</sub> up excitin<sub>g</sub> new app<sup>l</sup>ications.

Limitations. While we improved over existing strong stochastic flow maps, iSSFMs still struggle to match the <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce o</sub>f d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> f<sub>e</sub>w-<sub>s</sub>t<sub>ep</sub> m<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> Additi<sub>o</sub>n<sub>a</sub>ll<sub>y,</sub> whil<sub>e</sub> w<sub>e</sub> did <sub>so</sub>m<sub>e s</sub>m<sub>a</sub>ll <sub>e</sub>x<sub>p</sub>l<sub>o</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> w<sub>e</sub> h<sub>a</sub>v<sub>e</sub> n<sub>o</sub>t f<sub>oun</sub>d <sub>a</sub> b<sub>e</sub>tt<sub>er s</sub>t<sub>ra</sub>t<sub>egy</sub> f<sub>or pass</sub>i<sub>ng</sub> th<sub>e</sub> B<sub>rown</sub>i<sub>an coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e ne</sub>t<sub>wor</sub>k <sub>o</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e conca</sub>t<sub>ena</sub>ti<sub>on s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 12<sub>; we</sub> l<sub>eave</sub> th<sub>e s</sub>t<sub>u</sub>d<sub>y o</sub>f h<sub>ow</sub> t<sub>o more e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y process</sub> th<sub>e</sub> B<sub>rown</sub>i<sub>an coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> t<sub>o</sub> f<sub>u</sub>t<sub>ure wor</sub>k<sub>.</sub>

## Author contributions

ZB <sup>d</sup>eve<sup>l</sup>ope<sup>d</sup> t<sup>h</sup>e initia<sup>l</sup> i<sup>d</sup>ea, viz. t<sup>h</sup>e Doss–Sussmann an<sup>d</sup> Wong–Za<sup>k</sup>ai arguments <sup>f</sup>or training wit<sup>h</sup> Eu<sup>l</sup>erian <sub>a</sub>nd L<sub>ag</sub>r<sub>a</sub>n<sub>g</sub>i<sub>a</sub>n l<sub>osses a</sub>nd th<sub>e</sub> id<sub>ea o</sub>f fin<sub>e</sub>-t<sub>u</sub>nin<sub>g e</sub>xi<sub>s</sub>tin<sub>g</sub> fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> int<sub>o s</sub>tr<sub>o</sub>n<sub>g s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>a s.</sub> AF l<sub>e</sub>d th<sub>e e</sub>x <sub>e</sub>rim<sub>e</sub>nt<sub>s a</sub>nd inv<sub>es</sub>ti <sub>a</sub>ti<sub>o</sub>n int<sub>o</sub> th<sub>e</sub> dif<sub>e</sub>r<sub>e</sub>nt r<sub>e</sub> r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>a</sub>th with assistance from RM. ZB led the Ima<sub>g</sub>eNet ex<sub>p</sub>eriments and the com<sub>p</sub>arisons amon<sub>g</sub>st the diferent trainin<sub>g</sub> objectives, with RM contributing the strong consistency experiments. CK worked on the theory surrounding th<sub>e</sub> dif<sub>e</sub>r<sub>e</sub>nt l<sub>osses.</sub> ZB <sub>p</sub>r<sub>opose</sub>d <sub>a</sub>nd r<sub>e</sub>fin<sub>e</sub>d m<sub>a</sub>n<sub>y o</sub>f th<sub>e co</sub>r<sub>e</sub> id<sub>eas</sub> in <sub>co</sub>rr<sub>espo</sub>nd<sub>e</sub>n<sub>ce</sub> with NR<sub>.</sub> ZB wr<sub>o</sub>t<sub>e</sub> th<sub>e</sub> paper, with contributions from all other authors. AT guided the project.

## Acknowledgments

ZB wishes to acknowled<sub>g</sub>e Sam McCallum and Shuchen<sub>g</sub> Li for feedback on the manuscri<sub>p</sub>t in various sta<sub>g</sub>es. This research was enabled in <sub>p</sub>art b<sub>y</sub> com<sub>p</sub>utational resources <sub>p</sub>rovided b<sub>y</sub> AITHYRA. AITHYRA is su<sub>pp</sub>orted b<sub>y</sub> the Austrian Academ<sub>y</sub> of Sciences and the not-for-<sub>p</sub>rofit Boehrin<sub>g</sub>er In<sub>g</sub>elheim Stiftun<sub>g</sub>. We also acknowled<sub>g</sub>e <sub>a</sub>nd th<sub>a</sub>nk E<sub>u</sub>r<sub>o</sub>HPC J<sub>o</sub>int Und<sub>e</sub>rt<sub>a</sub>kin<sub>g</sub> f<sub>o</sub>r <sub>a</sub>w<sub>a</sub>rdin<sub>g us access</sub> t<sub>o</sub> th<sub>e</sub> Arrh<sub>e</sub>ni<sub>us</sub> GPU <sub>pa</sub>rtiti<sub>o</sub>n <sub>a</sub>t NAISS<sub>,</sub> Sw<sub>e</sub>d<sub>e</sub>n<sub>,</sub> through its Benchmark Access call under project EHPC-BEN-2026B09-140.

## References

Ai, X., He, Y., Gu, A., Salakhutdinov, R., Kolter, J. Z., Bofi, N. M., and Simchowitz, M. (2026). “Joint Distillation for Fast Li<sup>k</sup>e<sup>l</sup>i<sup>h</sup>oo<sup>d</sup> Eva<sup>l</sup>uation an<sup>d</sup> Samp<sup>l</sup>ing in F<sup>l</sup>ow-<sup>b</sup>ase<sup>d</sup> Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: The Fourteenth International Conference on Learning Representations (cit. on pp. 11, 45).

Akh<sub>oun</sub>d<sub>-</sub>S<sub>a</sub>d<sub>eg</sub>h<sub>,</sub> T<sub>.,</sub> L<sub>ee,</sub> J<sub>.,</sub> B<sub>ose,</sub> J<sub>.,</sub> B<sub>or</sub>t<sub>o</sub>li<sub>,</sub> V<sub>.</sub> D<sub>.,</sub> D<sub>ouce</sub>t<sub>,</sub> A<sub>.,</sub> B<sub>rons</sub>t<sub>e</sub>i<sub>n,</sub> M<sub>.</sub> M<sub>.,</sub> B<sub>ea</sub>i<sub>n</sub>i<sub>,</sub> D<sub>.,</sub> R<sub>avan</sub>b<sub>a</sub>kh<sub>s</sub>h<sub>,</sub> S<sub>.,</sub> Nekl<sub>y</sub>udov, K., and Ton<sub>g</sub>, A. (2025). “Pro<sub>g</sub>ressive Inference-Time Annealin<sub>g</sub> of Difusion Models for Sam<sub>p</sub>lin<sub>g</sub> <sup>f</sup>rom Bo<sup>l</sup>tzmann Densities<sup>”</sup>. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (cit. on <sub>p</sub>. 1).

Alber<sub>g</sub>o, M., Bofi, N. M., and Vanden-Eijnden, E. (2025). “Stochastic inter<sub>p</sub>olants: A unif<sub>y</sub>in<sub>g</sub> framework for flows an<sup>d</sup> <sup>d</sup>i<sup>f</sup>usions<sup>”</sup>. In: Journal of Machine Learning Research 26.209, pp. 1–80 (cit. on p. 3).

An<sup>d</sup>erson, B. D. (1982). <sup>“</sup>Reverse-time <sup>d</sup>i<sup>f</sup>usion equation mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Stochastic Processes and their Applications 12.3, <sub>pp</sub>. 313–326 (cit. on <sub>p</sub>. 38).

Bansal, A., Chu, H.-M., Schwarzschild, A., Sen u ta, S., Goldblum, M., Gei in , J., and Goldstein, T. (2023). <sup>“</sup>Universa<sup>l</sup> gui<sup>d</sup>ance <sup>f</sup>or <sup>d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition, pp. 843–852 (cit. on p. 45).

Baranek, M. et al. (2025). “SPINNs–Dee<sub>p</sub> learnin<sub>g</sub> framework for a<sub>pp</sub>roximation of stochastic diferential e<sub>q</sub>uations<sup>”</sup>. In: arXiv preprint arXiv:2512.14258 (cit. on p. 32).

Ben-Hamu, H., Pun<sub>y</sub>, O., Gat, I., Karrer, B., Sin<sub>g</sub>er, U., and Li<sub>p</sub>man, Y. (2024). “D-Flow: Diferentiatin<sub>g</sub> throu<sub>g</sub>h F<sup>l</sup>ows <sup>f</sup>or Contro<sup>ll</sup>e<sup>d</sup> Generation<sup>”</sup>. In: Forty-first International Conference on Machine Learning (cit. on p. 45).

Bickel, P., Li, B., and Ben<sub>g</sub>tsson, T. (2008). “Shar<sub>p</sub> failure rates for the bootstra<sub>p p</sub>article filter in hi<sub>g</sub>h dimensions”. In: Pushing the limits ofcontemporary statistics: Contributions in honor ofJayanta K. Ghosh. Vo<sup>l</sup>. 3. Institute o<sup>f</sup> Mathematical Statistics, <sub>pp</sub>. 318–330 (cit. on <sub>p</sub>. 45).

Biloš, M., Sommer, J., Ran<sub>g</sub>a<sub>p</sub>uram, S. S., Januschowski, T., and Günnemann, S. (2021). “Neural Flows: Eficient A<sup>l</sup>ternative to Neura<sup>l</sup> ODEs<sup>”</sup>. In: Advances in Neural Information Processing Systems. E<sup>d</sup>. <sup>b</sup>y A. Beyge<sup>l</sup>zimer, Y. Dau<sub>p</sub>hin, P. Lian<sub>g</sub>, and J. W. Vau<sub>g</sub>han (cit. on <sub>p</sub>. 20).

B<sup>l</sup>asingame, Z. W. an<sup>d</sup> Liu, C. (2024). <sup>“</sup>A<sup>d</sup>jointDEIS: E<sup>fi</sup>cient Gra<sup>d</sup>ients <sup>f</sup>or Di<sup>f</sup>usion Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (cit. on p. 45).

Blasingame, Z. W. and Liu, C. (2025). “Greed is Good: A Unifying Perspective on Guided Generation”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (cit. on p. 45).

Blasin<sub>g</sub>ame, Z. W. and Liu, C. (2026). “Rex: A Famil<sub>y</sub> of Reversible Ex<sub>p</sub>onential (Stochastic) Run<sub>g</sub>e-Kutta Solvers”. In: Forty-third International Conference on Machine Learning (cit. on pp. 1, 30).

Blattmann, A., Rombach, R., Lin , H., Dockhorn, T., Kim, S. W., Fidler, S., and Kreis, K. (2023). “Ali n our latents: Hig<sup>h</sup>-reso<sup>l</sup>ution vi<sup>d</sup>eo synt<sup>h</sup>esis wit<sup>h l</sup>atent <sup>d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 22563–22575 (cit. on p. 1).

Bo<sup>fi</sup>, N. M., A<sup>lb</sup>ergo, M. S., an<sup>d</sup> Van<sup>d</sup>en-Eijn<sup>d</sup>en, E. (2024). <sup>“</sup>F<sup>l</sup>ow map matc<sup>h</sup>ing<sup>”</sup>. In: arXiv preprint arXiv:2406.07507 (cit. on <sub>p</sub>. 20).

Bofi, N. M., Alber<sub>g</sub>o, M. S., and Vanden-Eijnden, E. (2025). “How to build a consistenc<sub>y</sub> model: Learnin<sub>g</sub> flow maps via se<sup>lf</sup>-<sup>d</sup>isti<sup>ll</sup>ation<sup>”</sup>. In: arXiv preprint arXiv:2505.18825 (cit. on pp. 2, 3, 20).

Bortoli, V. D., Galashov, A., Guntu<sub>p</sub>alli, J. S., Zhou, G., Mur<sub>p</sub>h<sub>y</sub>, K. P., Gretton, A., and Doucet, A. (2025). “Distributiona<sup>l</sup> Di<sup>f</sup>usion Mo<sup>d</sup>e<sup>l</sup>s wit<sup>h</sup> Scoring Ru<sup>l</sup>es<sup>”</sup>. In: Forty-second International Conference on Machine Learning (cit. on <sub>p</sub>. 43).

Chen, H., Zhan<sub>g</sub>, K., Tan, H., Xu, Z., Luan, F., Guibas, L., Wetzstein, G., and Bi, S. (2025). “Gaussian Mixture Flow Matc<sup>h</sup>ing Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Forty-second International Conference on Machine Learning (cit. on p. 43).

C<sup>h</sup>en, K.-T. (1954). <sup>“</sup>Iterate<sup>d</sup> integra<sup>l</sup>s an<sup>d</sup> exponentia<sup>l h</sup>omomorp<sup>h</sup>isms<sup>”</sup>. In: Proceedings ofthe London Mathematical Society 3.1, pp. 502–512 (cit. on p. 5).

Chen, K.-T. (1957). “Inte<sub>g</sub>ration of <sub>p</sub>aths, <sub>g</sub>eometric invariants and a <sub>g</sub>eneralized Baker-Hausdorf formula”. In: Annals of Mathematics 65.1, pp. 163–178 (cit. on p. 5).

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. (2018). “Neural Ordinar<sub>y</sub> Diferential E<sub>q</sub>uations”. In: Advances in Neural Information ProcessingSystems. E<sup>d</sup>. <sup>b</sup>y S. Bengio, H. Wa<sup>ll</sup>ac<sup>h</sup>, H. Laroc<sup>h</sup>e<sup>ll</sup>e, K. Grauman, N. Cesa-Bianchi, and R. Garnett. Vol. 31. Curran Associates, Inc. (cit. on <sub>p</sub>. 20).

Chun<sub>g</sub>, H., Kim, J., Mccann, M. T., Klask<sub>y</sub>, M. L., and Ye, J. C. (2023). “Difusion Posterior Sam<sub>p</sub>lin<sub>g</sub> for General Noisy Inverse Pro<sup>bl</sup>ems<sup>”</sup>. In: The Eleventh International Conference on Learning Representations, ICLR 2023. The International Conference on Learnin<sub>g</sub> Re<sub>p</sub>resentations (cit. on <sub>p</sub>. 45).

Claessen, K. and Pałka, M. H. (2013). “S<sub>p</sub>littable <sub>p</sub>seudorandom number <sub>g</sub>enerators usin<sub>g</sub> cr<sub>yp</sub>to<sub>g</sub>ra<sub>p</sub>hic hashin<sub>g</sub>”. In: ACM SIGPLAN Notices 48.12, pp. 47–58 (cit. on p. 24).

Doss, H. (1977). <sup>“</sup>Liens entre équations <sup>d</sup>i<sup>f</sup>érentie<sup>ll</sup>es stoc<sup>h</sup>astiques et or<sup>d</sup>inaires<sup>”</sup>. In: Annales de l’institut Henri Poincaré. Section B. Calcul des probabilités et statistiques. Vo<sup>l</sup>. 13. 2, pp. 99–125 (cit. on p. 6).

Du<sup>b</sup>ins, L. E. an<sup>d</sup> Sc<sup>h</sup>warz, G. (1965). <sup>“</sup>On continuous martinga<sup>l</sup>es<sup>”</sup>. In: Proceedings of the National Academy of Sciences 53.5, pp. 913–916 (cit. on pp. 30, 31).

Dyson, F. J. (1949). <sup>“</sup>T<sup>h</sup>e ra<sup>d</sup>iation t<sup>h</sup>eories o<sup>f</sup> Tomonaga, Sc<sup>h</sup>winger, an<sup>d</sup> Feynman<sup>”</sup>. In: Physical Review 75.3, <sub>p</sub>. 486 (cit. on <sub>p</sub>. 22).

Feynman, R. P. (1948). <sup>“</sup>Space-time approac<sup>h</sup> to non-re<sup>l</sup>ativistic quantum mec<sup>h</sup>anics<sup>”</sup>. In: Reviews ofmodern physics 20.2, p. 367 (cit. on p. 41).

Foster, J., L<sub>y</sub>ons, T., and Oberhauser, H. (2020). “An o<sub>p</sub>timal <sub>p</sub>ol<sub>y</sub>nomial a<sub>pp</sub>roximation of Brownian motion”. In: SIAMJournal on Numerical Analysis 58.3, pp. 1393–1421 (cit. on pp. 5, 25).

Foster, J. M., Dos Reis, G., and Stran<sub>g</sub>e, C. (2024). “Hi<sub>g</sub>h order s<sub>p</sub>littin<sub>g</sub> methods for SDEs satisf<sub>y</sub>in<sub>g</sub> a commutativity con<sup>d</sup>ition<sup>”</sup>. In: SIAMJournal on Numerical Analysis 62.1, pp. 500–532 (cit. on pp. 5, 32).

Frans, K., Ha<sup>f</sup>ner, D., Levine, S., an<sup>d</sup> A<sup>bb</sup>ee<sup>l</sup>, P. (2025). <sup>“</sup>One Step Di<sup>f</sup>usion via S<sup>h</sup>ortcut Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: The Thirteenth International Conference on Learning Representations (cit. on p. 20).

Frazer, R. A., Duncan, W. J., an<sup>d</sup> Co<sup>ll</sup>ar, A. R. (1938). Elementary matrices and some applications to dynamics and diferential equations. Cam<sup>b</sup>ri<sup>d</sup>ge University Press (cit. on p. 22).

Friz, P. K. an<sup>d</sup> Hairer, M. (May 2020). A course on rough paths: With an introduction to regularity structures. en. Cham, Switzerland: S<sub>p</sub>rin<sub>g</sub>er Nature (cit. on <sub>p</sub>. 23).

Gaines, J. G. and L<sub>y</sub>ons, T. J. (1997). “Variable ste<sub>p</sub> size control in the numerical solution of stochastic diferential equations<sup>”</sup>. In: SIAMJournal on Applied Mathematics 57.5, pp. 1455–1484 (cit. on p. 24).

Gen<sub>g</sub>, Z., Den<sub>g</sub>, M., Bai, X., Kolter, J. Z., and He, K. (2025). “Mean Flows for One-ste<sub>p</sub> Generative Modelin<sub>g</sub>”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (cit. on pp. 1–3, 20).

Gen<sub>g</sub>, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J. Z., and He, K. (2025). “Im<sub>p</sub>roved mean flows: On the challen<sub>g</sub>es o<sup>f</sup> <sup>f</sup>ast<sup>f</sup>orwar<sup>d</sup> generative mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: arXiv preprint arXiv:2512.02012 (cit. on pp. 2, 3, 5, 7, 8, 20, 21).

Gonzalez, M., Fernandez, N., Tran, T. V. D., Gherbi, E., Hajri, H., and Masmoudi, N. (2023). “SEEDS: Ex<sub>p</sub>onential SDE So<sup>l</sup>vers <sup>f</sup>or Fast Hig<sup>h</sup>-Qua<sup>l</sup>ity Samp<sup>l</sup>ing <sup>f</sup>rom Di<sup>f</sup>usion Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Thirty-seventh Conference on Neural Information Processing Systems (cit. on p. 1).

Grossman, M. an<sup>d</sup> Katz, R. (1972). Non-Newtonian Calculus: A Self-contained, Elementary Exposition ofthe Authors’ Investigations... Non-Newtonian Ca<sup>l</sup>cu<sup>l</sup>us (cit. on p. 22).

Hee<sup>k</sup>, J., Hooge<sup>b</sup>oom, E., an<sup>d</sup> Sa<sup>l</sup>imans, T. (2024). Multistep Consistency Models (cit. on p. 20).

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). “GANs Trained b<sub>y</sub> a Two Time-Sca<sup>l</sup>e Up<sup>d</sup>ate Ru<sup>l</sup>e Converge to a Loca<sup>l</sup> Nas<sup>h</sup> Equi<sup>l</sup>i<sup>b</sup>rium<sup>”</sup>. In: Advances in Neural Information Processing Systems. E<sup>d</sup>. <sup>b</sup>y I. Guyon, U. V. Lux<sup>b</sup>urg, S. Bengio, H. Wa<sup>ll</sup>ac<sup>h</sup>, R. Fergus, S. Vis<sup>h</sup>wanat<sup>h</sup>an, an<sup>d</sup> R. Garnett. Vol. 30. Curran Associates, Inc. (cit. on <sub>pp</sub>. 9, 46).

Hoc<sup>hb</sup>ruc<sup>k</sup>, M. an<sup>d</sup> Ostermann, A. (2010). <sup>“</sup>Exponentia<sup>l</sup> integrators<sup>”</sup>. In: Acta Numerica 19, pp. 209–286 (cit. on <sub>pp</sub>. 7, 36).

H<sub>o</sub>ld<sub>err</sub>i<sub>e</sub>th<sub>,</sub> P<sub>.,</sub> Ch<sub>en,</sub> D<sub>.,</sub> E<sub>yr</sub>i<sub>ng,</sub> L<sub>.,</sub> Sh<sub>a</sub>h<sub>,</sub> I<sub>.,</sub> A<sub>nan</sub>th<sub>araman,</sub> G<sub>.,</sub> H<sub>e,</sub> Y<sub>.,</sub> Ak<sub>a</sub>t<sub>a,</sub> Z<sub>.,</sub> J<sub>aa</sub>kk<sub>o</sub>l<sub>a,</sub> T<sub>.,</sub> B<sub>o</sub>fi<sub>,</sub> N<sub>.</sub> M<sub>., an</sub>d Simc<sup>h</sup>owitz, M. (2026). <sup>“</sup>Diamon<sup>d</sup> Maps: E<sup>fi</sup>cient Rewar<sup>d</sup> A<sup>l</sup>ignment via Stoc<sup>h</sup>astic F<sup>l</sup>ow Maps<sup>”</sup>. In: arXiv preprint arXiv:2602.05993 (cit. on pp. 35, 43–45).

Holderrieth, P., Sin<sub>g</sub>er, U., Jaakkola, T., Chen, R. T. Q., Li<sub>p</sub>man, Y., and Karrer, B. (2026). “GLASS Flows: Eficient In<sup>f</sup>erence <sup>f</sup>or Rewar<sup>d</sup> A<sup>l</sup>ignment o<sup>f</sup> F<sup>l</sup>ow an<sup>d</sup> Di<sup>f</sup>usion Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: The Fourteenth International Conference on Learning Representations (cit. on pp. 3, 42, 43).

Huan<sub>g</sub>, J. Y., Lin, J., Shah, S., Nair, K., and Bofi, N. M. (2026). “How to <sub>g</sub>uide <sub>y</sub>our flow: Few-ste<sub>p</sub> ali<sub>g</sub>nment via <sup>fl</sup>ow map rewar<sup>d</sup> gui<sup>d</sup>ance<sup>”</sup>. In: arXiv preprint arXiv:2604.27147 (cit. on p. 45).

Jelinčič, A., Foster, J., and Kid<sub>g</sub>er, P. (2024). “Sin<sub>g</sub>le-seed <sub>g</sub>eneration of Brownian <sub>p</sub>aths and inte<sub>g</sub>rals for ada<sub>p</sub>tive an<sup>d</sup> <sup>h</sup>ig<sup>h</sup> or<sup>d</sup>er SDE so<sup>l</sup>vers<sup>”</sup>. In: arXiv preprint arXiv:2405.06464 (cit. on p. 24).

Kac, M. (1949). <sup>“</sup>On <sup>d</sup>istri<sup>b</sup>utions o<sup>f</sup> certain Wiener <sup>f</sup>unctiona<sup>l</sup>s<sup>”</sup>. In: Transactions ofthe American Mathematical Society 65.1, pp. 1–13 (cit. on p. 41).

Kar<sup>h</sup>unen, K. (1950). <sup>“Üb</sup>er <sup>d</sup>ie Stru<sup>k</sup>tur stationärer zu<sup>f</sup>ä<sup>ll</sup>iger Fun<sup>k</sup>tionen<sup>”</sup>. In: Arkiv för Matematik 1.2, pp. 141–160 (cit. on <sub>pp</sub>. 4, 24).

Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. (2024). “Anal<sub>y</sub>zin<sub>g</sub> and im<sub>p</sub>rovin<sub>g</sub> the trainin<sub>g</sub> <sup>d</sup>ynamics o<sup>f d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 24174–24184 (cit. on p. 46).

Kendall, A., Gal, Y., and Ci<sub>p</sub>olla, R. (2018). “Multi-task learnin<sub>g</sub> usin<sub>g</sub> uncertaint<sub>y</sub> to wei<sub>g</sub>h losses for scene geometry an<sup>d</sup> semantics<sup>”</sup>. In: Proceedings of the IEEE conference on computer vision and pattern recognition, <sub>pp</sub>. 7482–7491 (cit. on <sub>p</sub>. 46).

Kidger, P. (2022). “On Neural Diferential Equations”. Available at https://arxiv.org/abs/2202.02435. Ph.D. thesis. Oxford Universit<sub>y</sub> (cit. on <sub>p</sub>. 23).

Ki<sup>d</sup> er, P., Foster, J., Li, X., an<sup>d</sup> L ons, T. (2021). <sup>“</sup>E<sup>fi</sup>cient an<sup>d</sup> Accurate Gra<sup>d</sup>ients <sup>f</sup>or Neura<sup>l</sup> SDEs<sup>”</sup>. In: Advances in Neural Information Processing Systems. E<sup>d</sup>. <sup>b</sup>y A. Beyge<sup>l</sup>zimer, Y. Daup<sup>h</sup>in, P. Liang, an<sup>d</sup> J. W. Vaug<sup>h</sup>an (cit. on <sub>p</sub>. 24).

Kim, D., Lai, C.-H., Liao, W.-H., Murata, N., Takida, Y., Uesaka, T., He, Y., Mitsufuji, Y., and Ermon, S. (2024). <sup>“</sup>Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Difusion”. In: The Twelfth International Conference on Learning Representations (cit. on p. 20).

Kirstain, Y., Pol<sub>y</sub>ak, A., Sin<sub>g</sub>er, U., Matiana, S., Penna, J., and Lev<sub>y</sub>, O. (2023). “Pick-a-Pic: An O<sub>p</sub>en Dataset of User Preferences for Text-to-Ima<sub>g</sub>e Generation”. In: (cit. on <sub>pp</sub>. 10, 46).

Ki<sub>y</sub>ohara, N., Johns, E., and Li, Y. (2025). “Neural Stochastic Flows: Solver-Free Modellin<sub>g</sub> and Inference for SDE So<sup>l</sup>utions<sup>”</sup>. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (cit. on p. 43).

K<sub>y</sub>nkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., and Lehtinen, J. (2024). “A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Guidance in a Limite<sup>d</sup> Interva<sup>l</sup> Improves Samp<sup>l</sup>e an<sup>d</sup> Distri<sup>b</sup>ution Qua<sup>l</sup>ity in Di<sup>f</sup>usion Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Advances in Neural Information Processing Systems (cit. on p. 46).

Lee, K., Yu, S., and Shin, J. (2026). “Decou<sub>p</sub>led MeanFlow: Turnin<sub>g</sub> Flow Models into Flow Ma<sub>p</sub>s for Accelerated Samp<sup>l</sup>ing<sup>”</sup>. In: The Fourteenth International Conference on Learning Representations (cit. on pp. 2, 6–8, 20, 35, 36, 45, 46).

Lévy, M. P. (1940). <sup>“</sup>Le mouvement <sup>b</sup>rownien p<sup>l</sup>an<sup>”</sup>. In: American Journal of Mathematics 62.1, pp. 487–550 (cit. on <sub>p</sub>. 24).

Li, X., Won<sub>g</sub>, T.-K. L., Chen, R. T. Q., and Duvenaud, D. (26–28 Au<sub>g</sub> 2020). “Scalable Gradients for Stochastic Di<sup>f</sup>erentia<sup>l</sup> Equations<sup>”</sup>. In: Proceedings ofthe Twenty Third International Conference on Artificial Intelligence and Statistics. E<sup>d</sup>. <sup>b</sup>y S. C<sup>h</sup>iappa an<sup>d</sup> R. Ca<sup>l</sup>an<sup>d</sup>ra. Vo<sup>l</sup>. 108. Procee<sup>d</sup>ings o<sup>f</sup> Mac<sup>h</sup>ine Learning Researc<sup>h</sup>. PMLR, <sub>pp</sub>. 3870–3882 (cit. on <sub>p</sub>. 24).

Li<sub>p</sub>man, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). “Flow Matchin<sub>g</sub> for Generative Modelin<sub>g</sub>”. In: The Eleventh International Conference on Learning Representations (cit. on pp. 3, 38).

Li<sub>p</sub>man<sub>,</sub> Y.<sub>,</sub> Havasi<sub>,</sub> M.<sub>,</sub> Holderrieth<sub>,</sub> P.<sub>,</sub> Shaul<sub>,</sub> N.<sub>,</sub> Le<sub>,</sub> M.<sub>,</sub> Karrer<sub>,</sub> B.<sub>,</sub> Chen<sub>,</sub> R. T.<sub>,</sub> Lo<sub>p</sub>ez-Paz<sub>,</sub> D.<sub>,</sub> Ben-Hamu<sub>,</sub> H.<sub>,</sub> an<sup>d</sup> Gat, I. (2024). Flow Matching Guide and Code (cit. on p. 3).

Liu, X., Gon<sub>g</sub>, C., and Liu, Q. (2023). “Flow Strai<sub>g</sub>ht and Fast: Learnin<sub>g</sub> to Generate and Transfer Data with Recti<sup>fi</sup>e<sup>d</sup> F<sup>l</sup>ow<sup>”</sup>. In: The Eleventh International Conference on Learning Representations (cit. on p. 3).

Loève, M. (1945). <sup>“</sup>Sur <sup>l</sup>es <sup>f</sup>onctions a<sup>l</sup>éatoires stationnaires <sup>d</sup>u secon<sup>d</sup> or<sup>d</sup>re<sup>”</sup>. In: Revue Scientifique 83, pp. 297– 303 (cit. on <sub>pp</sub>. 4, 24).

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. (2022). “DPM-Solver: A Fast ODE Solver for Difusion Pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic Mo<sup>d</sup>e<sup>l</sup> Samp<sup>l</sup>ing in Aroun<sup>d</sup> 10 Steps<sup>”</sup>. In: Advances in Neural Information Processing Systems. Ed. b<sub>y</sub> A. H. Oh, A. A<sub>g</sub>arwal, D. Bel<sub>g</sub>rave, and K. Cho (cit. on <sub>pp</sub>. 1, 7).

Magnus, W. (1954). <sup>“</sup>On t<sup>h</sup>e exponentia<sup>l</sup> so<sup>l</sup>ution o<sup>f</sup> <sup>d</sup>i<sup>f</sup>erentia<sup>l</sup> equations <sup>f</sup>or a <sup>l</sup>inear operator<sup>”</sup>. In: Communications on pure and applied mathematics 7.4, pp. 649–673 (cit. on p. 22).

Maoutsa, D., Reich, S., and O<sub>pp</sub>er, M. (2020). “Interactin<sub>g</sub> <sub>p</sub>article solutions of fokker–<sub>p</sub>lanck e<sub>q</sub>uations throu<sub>g</sub>h gra<sup>d</sup>ient–<sup>l</sup>og–<sup>d</sup>ensity estimation<sup>”</sup>. In: Entropy 22.8, p. 802 (cit. on p. 38).

Marion<sub>,</sub> P.<sub>,</sub> Korba<sub>,</sub> A.<sub>,</sub> Bartlett<sub>,</sub> P.<sub>,</sub> Blondel<sub>,</sub> M.<sub>,</sub> Bortoli<sub>,</sub> V. D.<sub>,</sub> Doucet<sub>,</sub> A.<sub>,</sub> Llinares-Ló<sub>p</sub>ez<sub>,</sub> F.<sub>,</sub> Pa<sub>q</sub>uette<sub>,</sub> C.<sub>,</sub> and Berthet, Q. (2025). “Implicit Difusion: Eficient optimization through stochastic sampling”. In: The 28th International Conference on Artificial Intelligence and Statistics (cit. on p. 45).

McCallum, S., Blasin<sub>g</sub>ame, Z. W., Herschell, T., Rindtorf, N., Ton<sub>g</sub>, A., and Foster, J. (2026). “Stron<sub>g</sub> Stochastic F<sup>l</sup>ow Maps<sup>”</sup>. In: The Fortieth Annual Conference on Neural Information Processing Systems (cit. on pp. 2–5, 7–10, 26, 30, 35, 43, 44, 46, 47, 50).

Moufad, B., Janati, Y., Bedin, L., Durmus, A. O., Douc, R., Moulines, E., and Olsson, J. (2025). “Variational Difusion Posterior Samp<sup>l</sup>ing wit<sup>h</sup> Mi<sup>d</sup>point Gui<sup>d</sup>ance<sup>”</sup>. In: The Thirteenth International Conference on Learning Representations (cit. on p. 45).

Nie, W., Guo, B., Huan<sub>g</sub>, Y., Xiao, C., Vahdat, A., and Anandkumar, A. (17–23 Jul 2022). “Difusion Models for A<sup>d</sup>versaria<sup>l</sup> Puri<sup>fi</sup>cation<sup>”</sup>. In: Proceedings ofthe 39th International Conference on Machine Learning. E<sup>d</sup>. <sup>b</sup>y K. Ch<sub>au</sub>dh<sub>u</sub>ri<sub>,</sub> S<sub>.</sub> J<sub>ege</sub>lk<sub>a,</sub> L<sub>.</sub> S<sub>o</sub>n<sub>g,</sub> C<sub>.</sub> Sz<sub>epes</sub>v<sub>a</sub>ri<sub>,</sub> G<sub>.</sub> Ni<sub>u,</sub> <sub>a</sub>nd S<sub>.</sub> S<sub>a</sub>b<sub>a</sub>t<sub>o.</sub> V<sub>o</sub>l<sub>.</sub> 162<sub>.</sub> Pr<sub>ocee</sub>din<sub>gs</sub> <sub>o</sub>f M<sub>ac</sub>hin<sub>e</sub> L<sub>ea</sub>rnin<sub>g</sub> Research. PMLR, <sub>pp</sub>. 16805–16827 (cit. on <sub>p</sub>. 45).

Ø<sup>k</sup>sen<sup>d</sup>a<sup>l</sup>, B. (Ju<sup>l</sup>y 2003). Stochastic Diferential Equations: An Introduction with Applications. Universitext. Ber<sup>l</sup>in, Germany: Springer Berlin Heidelberg. ISBN: 9783662036204. DOI: 10.1007/978-3-642-14394-6 (cit. on p. 44).

O<sub>u</sub>Y<sub>ang,</sub> R<sub>.,</sub> Y<sub>u,</sub> H<sub>.,</sub> Ai<sub>,</sub> X<sub>.,</sub> H<sub>e,</sub> Y<sub>.,</sub> B<sub>o</sub>fi<sub>,</sub> N<sub>.</sub> M<sub>.,</sub> R<sub>av</sub>ik<sub>umar,</sub> P<sub>.,</sub> H<sub>ern</sub>á<sub>n</sub>d<sub>ez-</sub>L<sub>o</sub>b<sub>a</sub>t<sub>o,</sub> J<sub>.</sub> M<sub>.,</sub> Si<sub>mc</sub>h<sub>ow</sub>it<sub>z,</sub> M<sub>.,</sub> Mill<sub>er,</sub> B. K., and C<sup>h</sup>e<sup>h</sup>ab, O. (2026). <sup>“</sup>Few-Step Bo<sup>l</sup>tzmann Generators via Sca<sup>l</sup>ab<sup>l</sup>e Like<sup>l</sup>i<sup>h</sup>ood F<sup>l</sup>ow Maps<sup>”</sup>. In: ICML 2026 Workshop on Structured Probabilistic Inference & Generative Modeling (cit. on p. 45).

Pan, J., Liew, J. H., Tan, V., Fen , J., and Yan, H. (2024). “AdjointDPM: Adjoint Sensitivit Method for Gradient Bac<sup>k</sup>propagation o<sup>f</sup> Di<sup>f</sup>usion Pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: The Twelfth International Conference on Learning Representations (cit. on p. 45).

Pan, Z., Potaptc<sup>h</sup>i<sup>k</sup>, P., Yao, W., A<sup>lb</sup>ergo, M. S., an<sup>d</sup> Pi<sup>d</sup>strigac<sup>h</sup>, J. (2026). <sup>“</sup>Itô maps <sup>f</sup>or any-step SDEs<sup>”</sup>. In: arXiv preprint arXiv:2606.11156 (cit. on pp. 2, 3, 5–8, 24, 25, 30, 33–35, 43, 44).

Passaro, R., B<sup>l</sup>asingame, Z. W., Bronstein, M. M., and Tong, A. (2026). <sup>“</sup>Stoc<sup>h</sup>astic Few-step Mode<sup>l</sup>s<sup>”</sup>. In: ICLR 2026 2nd Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Eficacy (cit. on <sub>pp</sub>. 35, 43–45).

Pe<sup>l</sup>uc<sup>h</sup>etti, S. (2021). Non-Denoising Forward-Time Difusions (cit. on p. 3).

Pota<sub>p</sub>tchik, P., Saravanan, A., Mammadov, A., Prat, A., Alber<sub>g</sub>o, M. S., and Teh, Y. W. (2026). “Meta Flow Ma<sub>p</sub>s ena<sup>bl</sup>e sca<sup>l</sup>a<sup>bl</sup>e rewar<sup>d</sup> a<sup>l</sup>ignment<sup>”</sup>. In: arXiv preprint arXiv:2601.14430 (cit. on pp. 35, 43–45).

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). “Hi<sub>g</sub>h-resolution ima<sub>g</sub>e s<sub>y</sub>nthesis with <sup>l</sup>atent <sup>d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Proceedings ofthe IEEE/CVFconference on computer vision and pattern recognition, <sub>pp</sub>. 10684–10695 (cit. on <sub>p</sub>. 1).

Sabour, A., Alber<sub>g</sub>o, M. S., Domin<sub>g</sub>o-Enrich, C., Bofi, N. M., Fidler, S., Kreis, K., and Vanden-Eijnden, E. (2025). <sup>“</sup>Test-time sca<sup>l</sup>ing o<sup>f</sup> <sup>d</sup>i<sup>f</sup>usions wit<sup>h</sup> <sup>fl</sup>ow maps<sup>”</sup>. In: arXiv preprint arXiv:2511.22688 (cit. on p. 45).

Sabour, A., Fidler, S., and Kreis, K. (2025). “Ali<sub>g</sub>n Your Flow: Scalin<sub>g</sub> Continuous-Time Flow Ma<sub>p</sub> Distillation”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (cit. on pp. 3–5, 8, 9, 20, 21, 32).

Sa<sup>k</sup>urai, J. J. an<sup>d</sup> Napo<sup>l</sup>itano, J. (2020). Modern quantum mechanics. Cam<sup>b</sup>ri<sup>d</sup>ge University Press (cit. on p. 22).

Salmon, J. K., Moraes, M. A., Dror, R. O., and Shaw, D. E. (2011). “Parallel random numbers: as eas<sub>y</sub> as 1, 2, 3”. In: Proceedings of 2011 international conference for high performance computing, networking, storage and analysis, <sub>pp</sub>. 1–12 (cit. on <sub>p</sub>. 24).

Sin <sup>h</sup>a<sup>l</sup>, R., Horvitz, Z., Tee<sup>h</sup>an, R., Ren, M., Yu, Z., McKeown, K., an<sup>d</sup> Ran anat<sup>h</sup>, R. (2025). A General Framework for Inference-time Scaling and Steering ofDifusion Models. arXiv: 2501.06848 [cs.LG] (cit. on p. 11).

Skreta<sub>,</sub> M.<sub>,</sub> Akhound-Sade<sub>g</sub>h<sub>,</sub> T.<sub>,</sub> Ohanesian<sub>,</sub> V.<sub>,</sub> Bondesan<sub>,</sub> R.<sub>,</sub> As<sub>p</sub>uru-Guzik<sub>,</sub> A.<sub>,</sub> Doucet<sub>,</sub> A.<sub>,</sub> Brekelmans<sub>,</sub> R.<sub>,</sub> Ton<sub>g</sub>, A., and Nekl<sub>y</sub>udov, K. (2025). “Fe<sub>y</sub>nman-Kac Correctors in Difusion: Annealin<sub>g</sub>, Guidance, and Produc o<sup>f</sup> Experts<sup>”</sup>. In: Forty-second International Conference on Machine Learning (cit. on pp. 7, 11, 39, 45, 47).

Skreta, M., Atanackovic, L., Bose, J., Ton<sub>g</sub>, A., and Nekl udov, K. (2025). “The Su<sub>p</sub>er<sub>p</sub>osition of Difusion Models Using t<sup>h</sup>e Itô Density Estimator<sup>”</sup>. In: The Thirteenth International Conference on Learning Representations (cit. on <sub>pp</sub>. 1, 40).

Sn<sub>y</sub>der, C., Ben<sub>g</sub>tsson, T., Bickel, P., and Anderson, J. (2008). “Obstacles to hi<sub>g</sub>h-dimensional <sub>p</sub>article filterin<sub>g</sub>”. In: Monthly Weather Review 136.12, pp. 4629–4640 (cit. on p. 45).

Song, Y., D<sup>h</sup>ariwa<sup>l</sup>, P., C<sup>h</sup>en, M., an<sup>d</sup> Suts<sup>k</sup>ever, I. (23–29 Ju<sup>l</sup> 2023). <sup>“</sup>Consistency Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Proceedings ofthe 40th International Conference on Machine Learning. E<sup>d</sup>. <sup>b</sup>y A. Krause, E. Bruns<sup>k</sup>i<sup>ll</sup>, K. C<sup>h</sup>o, B. Enge<sup>lh</sup>ar<sup>d</sup>t, S. Sabato, and J. Scarlett. Vol. 202. Proceedin<sub>g</sub>s of Machine Learnin<sub>g</sub> Research. PMLR, <sub>pp</sub>. 32211–32252 (cit. on <sub>pp</sub>. 1, 3, 6, 20, 21).

Son<sub>g</sub>, Y., Sohl-Dickstein, J., Kin<sub>g</sub>ma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). “Score-Based Generative Mo<sup>d</sup>e<sup>l</sup>ing t<sup>h</sup>roug<sup>h</sup> Stoc<sup>h</sup>astic Di<sup>f</sup>erentia<sup>l</sup> Equations<sup>”</sup>. In: International Conference on Learning Representations (cit. on <sub>pp</sub>. 1, 31, 38).

St<sub>e</sub>i<sub>n,</sub> G<sub>.,</sub> C<sub>resswe</sub>ll<sub>,</sub> J<sub>.</sub> C<sub>.,</sub> H<sub>osse</sub>i<sub>nza</sub>d<sub>e</sub>h<sub>,</sub> R<sub>.,</sub> S<sub>u</sub>i<sub>,</sub> Y<sub>.,</sub> R<sub>oss,</sub> B<sub>.</sub> L<sub>.,</sub> Vill<sub>ecroze,</sub> V<sub>.,</sub> Li<sub>u,</sub> Z<sub>.,</sub> C<sub>a</sub>t<sub>er</sub>i<sub>n</sub>i<sub>,</sub> A<sub>.</sub> L<sub>.,</sub> T<sub>ay</sub>l<sub>or,</sub> E<sub>., an</sub>d Loaiza-Ganem, G. (2023). “Ex osin flaws of enerative model evaluation metrics and their unfair treatment o<sup>f</sup> <sup>d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Thirty-seventh Conference on Neural Information Processing Systems (cit. on pp. 9, 46).

Stroock, D. and Varadhan, S. (1972). “On the su<sub>pp</sub>ort of difusion <sub>p</sub>rocesses with a<sub>pp</sub>lications to the st<sub>g</sub>ron<sub>g</sub> maximum princip<sup>l</sup>e<sup>”</sup>. In: Proceedings of the Berkeley symposium on mathematical statistics and probability. Vol. 1, <sub>p</sub>. 333 (cit. on <sub>pp</sub>. 6, 22).

Sussmann, H. J. (1978). “On the <sub>g</sub>a<sub>p</sub> between deterministic and stochastic ordinar<sub>y</sub> diferential e<sub>q</sub>uations”. In: The Annals of Probability, pp. 19–41 (cit. on p. 6).

Tang, Z., Bao, J., C<sup>h</sup>en, D., an<sup>d</sup> Guo, B. (2025). <sup>“</sup>Di<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s wit<sup>h</sup>out c<sup>l</sup>assi<sup>fi</sup>er-<sup>f</sup>ree gui<sup>d</sup>ance<sup>”</sup>. In: arXiv preprint arXiv:2502.12154 (cit. on p. 46).

Uehara, M., Zhao, Y., Wan<sub>g</sub>, C., Li, X., Re<sub>g</sub>ev, A., Levine, S., and Biancalani, T. (2025). “Inference-time ali<sub>g</sub>nment in <sup>d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l</sup>s wit<sup>h</sup> rewar<sup>d</sup>-gui<sup>d</sup>e<sup>d</sup> generation: Tutoria<sup>l</sup> an<sup>d</sup> review<sup>”</sup>. In: arXiv preprint arXiv:2501.09685 (cit. on <sub>p</sub>. 45).

Watson, J. L., Juer<sub>g</sub>ens, D., Bennett, N. R., Tri<sub>pp</sub>e, B. L., Yim, J., Eisenach, H. E., Ahern, W., Borst, A. J., Ra<sub>g</sub>otte, R. J., Mi<sup>ll</sup>es, L. F., et a<sup>l</sup>. (2023). <sup>“</sup>De novo <sup>d</sup>esign o<sup>f</sup> protein structure an<sup>d f</sup>unction wit<sup>h</sup> RF<sup>d</sup>i<sup>f</sup>usion<sup>”</sup>. In: Nature 620.7976, <sub>pp</sub>. 1089–1100 (cit. on <sub>p</sub>. 1).

Wein<sup>b</sup>erg, S. (1995). The quantum theory of fields. Vo<sup>l</sup>. 2. Cam<sup>b</sup>ri<sup>d</sup>ge university press (cit. on p. 22).

Wong, E. an<sup>d</sup> Za<sup>k</sup>ai, M. (1965a). <sup>“</sup>On t<sup>h</sup>e convergence o<sup>f</sup> or<sup>d</sup>inary integra<sup>l</sup>s to stoc<sup>h</sup>astic integra<sup>l</sup>s<sup>”</sup>. In: The Annals ofMathematical Statistics 36.5, pp. 1560–1564 (cit. on pp. 6, 22).

Won<sub>g</sub>, E. and Zakai, M. (1965b). “On the relation between ordinar<sub>y</sub> and stochastic diferential e<sub>q</sub>uations”. In: International Journal of Engineering Science 3.2, pp. 213–229 (cit. on pp. 6, 22).

Wu, L., Tri<sub>pp</sub>e, B. L., Naesseth, C. A., Cunnin<sub>g</sub>ham, J. P., and Blei, D. (2023). “Practical and As<sub>y</sub>m<sub>p</sub>toticall<sub>y</sub> Exact Con<sup>d</sup>itiona<sup>l</sup> Samp<sup>l</sup>ing in Di<sup>f</sup>usion Mo<sup>d</sup>e<sup>l</sup>s<sup>”</sup>. In: Thirty-seventh Conference on Neural Information Processing Systems (cit. on p. 45).

Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., and Li, H. (2023). “Human Preference Score v2: A Solid Benc<sup>h</sup>mar<sup>k</sup> <sup>f</sup>or Eva<sup>l</sup>uating Human Pre<sup>f</sup>erences o<sup>f</sup> Text-to-Image Synt<sup>h</sup>esis<sup>”</sup>. In: arXiv preprint arXiv:2306.09341 (cit. on <sub>pp</sub>. 10, 46).

Xu, J., Liu, X., Wu, Y., Ton<sub>g</sub>, Y., Li, Q., Din<sub>g</sub>, M., Tan<sub>g</sub>, J., and Don<sub>g</sub>, Y. (2023). “Ima<sub>g</sub>eReward: learnin<sub>g</sub> and evaluatin<sub>g</sub> <sup>h</sup>uman pre<sup>f</sup>erences <sup>f</sup>or text-to-image generation<sup>”</sup>. In: Proceedings ofthe 37th International Conference on Neural Information Processing Systems, pp. 15903–15935 (cit. on pp. 10, 46).

Yu, J., Wan<sub>g</sub>, Y., Zhao, C., Ghanem, B., and Zhan<sub>g</sub>, J. (2023). “Freedom: Trainin<sub>g</sub>-free ener<sub>gy</sub>-<sub>g</sub>uided conditional <sup>d</sup>i<sup>f</sup>usion mo<sup>d</sup>e<sup>l”</sup>. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23174–23184 (cit. on <sub>p</sub>. 45).

Yu, S., Kwak, S., Jan<sub>g</sub>, H., Jeon<sub>g</sub>, J., Huan<sub>g</sub>, J., Shin, J., and Xie, S. (2025). “Re<sub>p</sub>resentation Ali<sub>g</sub>nment for Generation: Training Di<sup>f</sup>usion Trans<sup>f</sup>ormers Is Easier T<sup>h</sup>an You T<sup>h</sup>in<sup>k”</sup>. In: International Conference on Learning Representations (cit. on p. 8).

Zhan<sub>g</sub>, B., Chu, W., Berner, J., Men<sub>g</sub>, C., Anandkumar, A., and Son<sub>g</sub>, Y. (2025). “Im<sub>p</sub>rovin<sub>g</sub> difusion inverse <sub>p</sub>rob-<sup>l</sup>em so<sup>l</sup>ving wit<sup>h d</sup>ecoup<sup>l</sup>e<sup>d</sup> noise annea<sup>l</sup>ing<sup>”</sup>. In: Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 20895–20905 (cit. on p. 45).

## Appendices

A Preliminaries 20   
A.1 Flow maps 20   
A.2 Training objectives for flow maps 20   
A.3 Doss–Sussmann transform 21   
A.4 The Wong–Zakai theorem for multi-dimensional SDEs 22   
B On the Brownian path 23   
B.1 Representations ofthe Brownian path . 23   
Dyadic increment tree • Karhunen–Loève expansion • Shifted Legendre polynomial expansion   
B.2 Derivatives ofthe shifted Legendre polynomial approximation 26   
B.3 Extended Chen relations for the shifted Legendre polynomial approximation . 26   
B.4 Closed-form martingales for common schedules. 30   
Linear • Variance preserving   
C On training objectives 32   
C.1 Discretized objectives . 32   
C.2 Doss–Sussmann objectives 32   
C.3 Wong–Zakai objectives 34   
D On distillation . 35   
D.1 Annealing the SDE. 37   
E Feynman–Kac correctors . 39   
E.1 Background. 39   
E.2 Augmented space stochastic flow maps for FKC . 42   
F Related work 43   
F.1 Stochastic flow maps . 43   
Weak vs strong stochastic flow maps   
F.2 Distilling path-based measurements in augmented space . 45   
F.3 Inference-time alignment and steering 45   
G Implementation details 45   
G.1 ImageNet-256 architecture 45   
G.2 ImageNet-256 training objectives 46   
G.3 ImageNet-256 evaluation. 46   
G.4 Toy experiments 47   
G.5 Hardware 47   
H Additional results . 47   
H.1 Toy experiments 48   
H.2 ImageNet 50   
51

## A Preliminaries

W<sub>e</sub> <sub>p</sub>r<sub>o</sub>vid<sub>e</sub> <sub>a</sub>dditi<sub>o</sub>n<sub>a</sub>l b<sub>ac</sub>k<sub>g</sub>r<sub>ou</sub>nd inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n whi<sub>c</sub>h w<sub>e</sub> w<sub>e</sub>r<sub>e</sub> n<sub>o</sub>t <sub>a</sub>bl<sub>e</sub> t<sub>o</sub> di<sub>scuss</sub> in th<sub>e</sub> m<sub>a</sub>in <sub>pape</sub>r<sub>.</sub>

## A.1 Flow maps

Earl<sub>y</sub> work on learnin<sub>g</sub> flow ma<sub>p</sub>s of ODEs was done b<sub>y</sub> Biloš et al. (2021) in the context of convertin<sub>g</sub> continuous normalizin<sub>g</sub> flows (Chen et al. 2018) into a flow ma<sub>p</sub>; however, such trainin<sub>g</sub> methods were not simulation-free. These ideas were revisited a<sub>g</sub>ain later b<sub>y</sub> consistenc<sub>y</sub> models (Heek et al. 2024; Kim et al. 2024; Son<sub>g</sub> et al. 2023) <sub>w</sub>hi<sub>c</sub>h l<sub>earn</sub> th<sub>e</sub> fl<sub>ow map</sub> $\Phi _ { t , 1 }$ f<sub>or a</sub>ll $t \in [ 0 , 1 ] , i . e .$ .<sub>,</sub> the<sub>y p</sub>redict the end<sub>p</sub>oint. Multi-ste<sub>p</sub> sam<sub>p</sub>lin<sub>g</sub> is done b<sub>y</sub> r<sub>e</sub>n<sub>o</sub>i<sub>s</sub>in<sub>g</sub> $\Phi _ { s . 1 } ^ { \theta } ( { \pmb x } _ { s } )$ vi<sub>a</sub> $\alpha _ { t } \Phi _ { s . 1 } ^ { \theta } ( { \pmb x } _ { s } ) + \sigma _ { t } { \pmb \varepsilon }$ <sup>f</sup>or unit Gaussian �—in t<sup>h</sup>e <sup>l</sup>iterature t<sup>h</sup>is is sometimes ca<sup>ll</sup>e<sup>d</sup> restart sampling (Lee et al. 2026). Later work by Heek et al. (2024) proposed partitioning the time interval [0, 1] into <sub>sma</sub>ll<sub>er</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s</sub> $\Pi = \{ 0 = t _ { 0 } < t _ { 1 } < \ldots < t _ { M } = 1 \}$ } ahead of training and learning a consistency model on <sub>eac</sub>h i<sub>n</sub>t<sub>erva</sub>l $\left[ t _ { m - 1 } , t _ { m } \right]$ . Shortcut models (Frans et al. 2025) are similar to the <sub>g</sub>eneral deterministic flow ma<sub>p</sub>s <sup>d</sup>iscusse<sup>d</sup> in t<sup>h</sup>is paper, i.e., <sup>l</sup>earning t<sup>h</sup>e map $\Phi _ { s , t } ,$ the main diferentiator is that the<sub>y</sub> train with a semi<sub>g</sub>rou<sub>p</sub> loss (see E<sub>q</sub>uation (8)). Mean flow models train flow ma<sub>p</sub>s with the Eulerian condition (Gen<sub>g</sub>, Den<sub>g</sub>, et al. 2025; Gen , Lu, et al. 2025) which was enerall found to be the most erformant (Sabour, Fidler, et al. 2025). Flow ma s were trained b Bofi et al. (2024) with a flow ma matchin loss that a lies no sto rad o eration, which in <sub>p</sub>ractice makes the em<sub>p</sub>irical <sub>p</sub>erformance much worse; later work b<sub>y</sub> Bofi et al. (2025) introduced the La<sub>g</sub>ran<sub>g</sub>ian condition—which was later found to onl<sub>y</sub> work well in certain settin<sub>g</sub>s (Sabour, Fidler, et al. 2025)—and revisited the semi<sub>g</sub>rou<sub>p</sub> and Eulerian conditions.

## A.2 Training objectives for flow maps

The primary training objectives for flow maps generally take the form:

(Eulerian)

$$
\mathbb { E } _ { s , t , \boldsymbol { x } } \left[ \left. \partial _ { s } \boldsymbol { \Phi } _ { s , t } ^ { \theta } ( \boldsymbol { x } ) + \pmb { \nu } _ { s } ( \boldsymbol { x } ) ^ { \top } \nabla _ { \pmb { x } } \boldsymbol { \Phi } _ { s , t } ^ { \theta } ( \pmb { x } ) \right. ^ { 2 } \right] ,\tag{6}
$$

(Lagrangian)

$$
\mathbb { E } _ { s , t , \mathbf { x } } \left[ \left\| \partial _ { t } \Phi _ { s , t } ^ { \theta } ( \mathbf { x } ) - \nu _ { t } \left( \Phi _ { s , t } ^ { \theta } ( \mathbf { x } ) \right) \right\| ^ { 2 } \right] ,\tag{7}
$$

(Semigroup)

$$
\begin{array} { r } { \mathbb { E } _ { s , t , x } \left[ \left. \Phi _ { s , t } ^ { \theta } ( { \pmb x } ) - \left( \Phi _ { u , t } ^ { \theta } \circ \Phi _ { s , u } ^ { \theta } \right) ( { \pmb x } ) \right. ^ { 2 } \right] , } \end{array}\tag{8}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { } _ { t }$ i<sub>s so</sub>m<sub>e</sub> v<sub>ec</sub>t<sub>o</sub>r fi<sub>e</sub>ld <sub>co</sub>rr<sub>espo</sub>ndin<sub>g</sub> t<sub>o e</sub>ith<sub>e</sub>r th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l’<sub>s</sub> l<sub>ea</sub>rn<sub>e</sub>d di<sub>ago</sub>n<sub>a</sub>l $\pmb { u } _ { t , t } ^ { \theta }$ <sub>, so</sub>m<sub>e g</sub>r<sub>ou</sub>nd tr<sub>u</sub>th v<sub>ec</sub>t<sub>o</sub>r fi<sub>e</sub>ld<sub>,</sub> <sub>or</sub> t<sub>eac</sub>h<sub>er</sub> fi<sub>e</sub>ld d<sub>epen</sub>di<sub>ng</sub> <sub>on</sub> th<sub>e</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> fl<sub>avour.</sub> $N . B .$ <sub>,</sub> th<sub>e</sub>r<sub>e</sub> <sub>a</sub>r<sub>e</sub> m<sub>a</sub>n<sub>y</sub> <sub>pa</sub>rti<sub>cu</sub>l<sub>a</sub>r d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> about how to im<sub>p</sub>lement these losses which massivel<sub>y</sub> im<sub>p</sub>act their utilit<sub>y,</sub> $e . g .$ <sub>,</sub> wh<sub>e</sub>r<sub>e</sub> t<sub>o p</sub>l<sub>ace</sub> th<sub>e s</sub>t<sub>opg</sub>r<sub>a</sub>d o<sub>p</sub>eration<sub>,</sub> what $\mathbf { } _ { t }$ represents, using EMA, &c.

Remark A.1. The semigroup is very similar to the discretized Eulerian and Lagrangian conditions which leverage the semi<sub>g</sub>rou<sub>p</sub> <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> alon<sub>g</sub> with a small ODE ste<sub>p</sub> from the start <sub>p</sub>oint or to the end <sub>p</sub>oint<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>.

Improved MeanFlow. One notable variant was introduced by Geng, Lu, et al. (2025) which rearranges the Eulerian residual. Let us rewrite the residual in E<sub>q</sub>uation (6) w.r.t. the mean velocit<sub>y</sub> network $\pmb { u } _ { s , t } ^ { \theta }$ . Then lettin<sub>g</sub> $h = t - s$ <sub>a</sub>nd $D _ { s } : = \partial _ { s } + \pmb { \nu } _ { s } ( \pmb { x } ) ^ { \top } \nabla _ { \pmb { x } }$ b<sub>e</sub> th<sub>e</sub> m<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l d<sub>e</sub>riv<sub>a</sub>tiv<sub>e ope</sub>r<sub>a</sub>t<sub>o</sub>r<sub>,</sub> w<sub>e</sub> h<sub>a</sub>v<sub>e</sub>

$$
{ D _ { s } } \Phi _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { x } } } ) = \partial _ { s } \left( { \boldsymbol { \mathbf { x } } } + h { \boldsymbol { \mathbf { u } } } _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { x } } } ) \right) + \nu _ { s } ( { \boldsymbol { \mathbf { x } } } ) ^ { \top } { \nabla _ { \boldsymbol { x } } } \left( { \boldsymbol { \mathbf { x } } } + h { \boldsymbol { \mathbf { u } } } _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { x } } } ) \right) ,\tag{9}
$$

$$
= - \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) + h \partial _ { s } \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) + \pmb { \nu } _ { s } ( \pmb { x } ) + h \pmb { \nu } _ { s } ( \pmb { x } ) ^ { \top } \nabla _ { \pmb { x } } \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) ,\tag{10}
$$

$$
= { \pmb v } _ { s } ( { \pmb x } ) - { \pmb u } _ { s , t } ^ { \theta } ( { \pmb x } ) + h \left( \partial _ { s } { \pmb u } _ { s , t } ^ { \theta } ( { \pmb x } ) + { \pmb v } _ { s } ( { \pmb x } ) ^ { \top } \nabla _ { \pmb x } { \pmb u } _ { s , t } ^ { \theta } ( { \pmb x } ) \right) ,\tag{11}
$$

$$
\begin{array} { r } { = \pmb { \nu } _ { s } ( \pmb { x } ) - \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) + h D _ { s } \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) . } \end{array}\tag{12}
$$

Then the training objective can be formulated as

(iMF)

$$
\begin{array} { r } { \mathbb { E } _ { s , t , \mathbf { x } , \dot { \mathbf { x } } } \left[ \left\| \pmb { \nu } _ { s } ( \pmb { x } ) - \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) + h \cdot \mathbf { s } \mathbf { g } ( D _ { s } \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ) ) \right\| ^ { 2 } \right] , } \end{array}\tag{13}
$$

where s<sub>g</sub> is the sto<sub>pg</sub>rad o<sub>p</sub>erator. In essence, the insi<sub>g</sub>ht of Gen<sub>g</sub>, Lu, et al. (2025) is to re<sub>g</sub>ress to the tar<sub>g</sub>et instantaneous velocit<sub>y</sub> ${ \pmb { \nu } } _ { s } ( { \pmb x } )$

Align your flow. Sabour, Fidler, et al. (2025) derive another form of the Eulerian and Lagrangian losses. Consider fi<sub>rs</sub>t th<sub>e</sub> di<sub>scre</sub>ti<sub>ze</sub>d l<sub>osses</sub> b<sub>e</sub>l<sub>ow</sub>

(Discrete Eulerian)

$$
\begin{array} { r } { \mathbb { E } _ { s , t , x } \left[ \left. \Phi _ { s , t } ^ { \theta } ( \pmb { x } ) - \mathrm { s g } \left[ \Phi _ { s + \delta , t } ^ { \theta } \left( \mathrm { O D E S o l v e r } _ { s , s + \delta } ( \pmb { x } ) \right) \right] \right. ^ { 2 } \right] , } \end{array}\tag{14}
$$

(Discrete La<sub>g</sub>ran<sub>g</sub>ian)

$$
\begin{array} { r } { \mathbb { E } _ { s , t , x } \left[ \left. \Phi _ { s , t } ^ { \theta } ( \pmb { x } ) - \mathrm { s g } \left[ \mathrm { O D E S o l v e r } _ { t - \delta , t } \left( \Phi _ { s , t - \delta } ^ { \theta } ( \pmb { x } ) \right) \right] \right. ^ { 2 } \right] , } \end{array}\tag{15}
$$

<sub>w</sub>h<sub>ere</sub> $\delta \in ( 0 , t - s )$ . N.B., Equation $\left( 1 4 \right)$ is <sub>q</sub>uite similar to the loss em<sub>p</sub>lo<sub>y</sub>ed in Son<sub>g</sub> et al. (2023). The<sub>y</sub> then s<sup>h</sup>ow (qq.v. Sa<sup>b</sup>our, Fi<sup>dl</sup>er, et a<sup>l</sup>. 2025, T<sup>h</sup>eorem C.2 an<sup>d</sup> $\operatorname { C . 4 } )$ th<sub>a</sub>t th<sub>e g</sub>r<sub>a</sub>di<sub>e</sub>nt <sub>o</sub>f th<sub>e co</sub>ntin<sub>uous</sub>-tim<sub>e</sub> limit <sub>as</sub> $\delta  0$ of the discrete Eulerian and La<sub>g</sub>ran<sub>g</sub>ian conditions conver<sub>g</sub>es to

(AYF-EMD)

$$
\mathbb { E } _ { s , t , \mathbf { x } } \left[ - ( t - s ) \left. \Phi _ { s , t } ^ { \theta } ( \mathbf { x } ) \middle \vert \ \mathrm { s g } \Big ( D _ { s } \Phi _ { s , t } ^ { \theta } ( \mathbf { x } ) \Big ) \right. \right] ,\tag{16}
$$

(AYF-LMD)

$$
\mathbb { E } _ { s , t , x } \left[ \left( t - s \right) \left. \Phi _ { s , t } ^ { \theta } ( { \pmb x } ) \Big \vert \ \mathrm { s g } \Big ( \partial _ { t } \Phi _ { s , t } ^ { \theta } ( { \pmb x } ) - \nu _ { t } \Big ( \Phi _ { s , t } ^ { \theta } ( { \pmb x } ) \Big ) \Big ) \right. \right] .\tag{17}
$$

Remark A.2. Surprisingly, these forms of the Eulerian and Lagrangian losses look quite diferent from Equations (6) and (7). This is due to <sub>p</sub>lacin<sub>g</sub> a sto<sub>pg</sub>rad o<sub>p</sub>eration on the terms with the small ODE ste<sub>p</sub> in the discretized conditions before takin<sub>g</sub> the continuous time limit, see Sabour, Fidler, et al. (2025, A<sub>pp</sub>endix D.4) for f<sub>ur</sub>th<sub>er</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s.</sub>

## A.3 Doss–Sussmann transform

C<sub>o</sub>n<sub>s</sub>id<sub>e</sub>r <sub>a</sub> $\mathbb { R } ^ { d }$ -v<sub>a</sub>l<sub>ue</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c p</sub>r<sub>ocess</sub> $\{ X _ { t } \} _ { t \ge 0 }$ <sub>w</sub>hi<sub>c</sub>h f<sub>o</sub>ll<sub>ows a</sub> St<sub>ra</sub>t<sub>onov</sub>i<sub>c</sub>h<sup>8</sup> SDE

$$
\mathrm { d } X _ { t } = \pmb { \mu } ( t , X _ { t } ) \mathrm { ~ d } t + \pmb { \sigma } ( t , X _ { t } ) \circ \mathrm { d } \pmb { W } _ { t } ,\tag{18}
$$

driven b<sub>y</sub> an $\mathbb { R } ^ { d _ { w } }$ -dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>a</sub>l Br<sub>o</sub>wni<sub>a</sub>n m<sub>o</sub>ti<sub>o</sub>n $\{ W _ { t } \} _ { t \ge 0 }$ <sub>w</sub>h<sub>ere</sub> $\pmb { \mu } \in C ( \mathbb { R } \times \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ <sub>a</sub>nd $\pmb { \sigma } \in C ( \mathbb { R } \times \mathbb { R } ^ { d } ; \mathbb { R } ^ { d \times d _ { w } } )$ <sub>a</sub>r<sub>e</sub> th<sub>e</sub> drift <sub>a</sub>nd dif<sub>us</sub>i<sub>o</sub>n <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s,</sub> r<sub>espec</sub>tiv<sub>e</sub>l<sub>y,</sub> whi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e usua</sub>l r<sub>egu</sub>l<sub>a</sub>rit<sub>y co</sub>nditi<sub>o</sub>n<sub>s,</sub> b<sub>u</sub>t d<sub>o</sub> n<sub>o</sub>t <sub>necessar</sub>il<sub>y commu</sub>t<sub>e.</sub> N<sub>ex</sub>t <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> fl<sub>ow o</sub>f th<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> t<sub>erms</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $\mathbf { \boldsymbol { r } } _ { t }$ as

$$
\mathrm { d } \Upsilon _ { t } ( \pmb { y } ) = \pmb { \sigma } ( t , \Upsilon _ { t } ( \pmb { y } ) ) \circ \mathrm { d } { \pmb { W } } _ { t } , \quad \Upsilon _ { 0 } ( \pmb { y } ) = \pmb { y } .\tag{19}
$$

![](images/0191e40ac0c6ebe916bf00b976fb3a705c623593c03c497f62680bd8ab26b3fa.jpg)  
Figure 7 Illustration of the Doss–Sussmann transform.

In <sub>g</sub>eneral, for non-commutin<sub>g</sub> fields, the solution is <sub>g</sub>iven b<sub>y</sub> the stochastic time-ordered ex<sub>p</sub>onential (Grossman and Katz 1972)<sup>9</sup>

$$
\begin{array} { l } { { \displaystyle \Upsilon _ { t } = \mathcal { T } \exp \left( \int _ { 0 } ^ { t } \mathrm { \mathrm { o d } W } _ { s } ~ { \boldsymbol \sigma } ( s , \cdot ) \right) , } \ ~ } \\ { { \displaystyle = \sum _ { n = 0 } ^ { \infty } \frac { 1 } { n ! } \int _ { 0 } ^ { t } \mathrm { \mathrm { \ o d } W } _ { t _ { 1 } } \int _ { 0 } ^ { t } \mathrm { \ o d } W _ { t _ { 2 } } \cdot \cdot \cdot \int _ { 0 } ^ { t } \mathrm { \ o d } W _ { t _ { n } } ~ { \mathcal { T } } \{ { \boldsymbol \sigma } ( t _ { 1 } , \cdot ) { \boldsymbol \sigma } ( t _ { 2 } , \cdot ) \cdot \cdot \cdot { \boldsymbol \sigma } ( t _ { n } , \cdot ) \} } , } \end{array}\tag{20}
$$

(21)

<sub>w</sub>h<sub>ere</sub> $\mathcal { T }$ is the time-orderin<sub>g</sub> o<sub>p</sub>erator (D<sub>y</sub>son 1949). $E . g .$ , for operators �, � the time ordering is given as

$$
\mathcal T \{ A ( s ) B ( t ) \} : = \left\{ \begin{array} { r l r } { A ( s ) B ( t ) } & { { } \quad } & { \mathrm { i f ~ } s > t , } \\ { B ( t ) A ( s ) } & { { } \quad } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{22}
$$

Th<sub>en</sub> if $\Upsilon _ { t }$ i<sub>s a</sub> $C ^ { 1 }$ <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> dif<sub>eo</sub>m<sub>o</sub>r<sub>p</sub>hi<sub>s</sub>m<sub>,</sub> d<sub>e</sub>fin<sub>e</sub> $Y _ { t } : = \Upsilon _ { t } ^ { - 1 } ( X _ { t } )$ ). Then the Doss–Sussmann transform yields the random ODE

$$
\dot { \pmb { Y } } _ { t } = [ D _ { y } \pmb { \Upsilon } _ { t } ( \pmb { Y } _ { t } ) ] ^ { - 1 } \pmb { \mu } \left( t , \pmb { \Upsilon } _ { t } ( \pmb { Y } _ { t } ) \right) , \qquad \pmb { Y } _ { 0 } = \pmb { x } .\tag{23}
$$

F<sub>o</sub>r <sub>a</sub>dditiv<sub>e</sub> n<sub>o</sub>i<sub>se</sub> $\pmb { \sigma } \in C ( \mathbb { R } ; \mathbb { R } ^ { d \times d _ { w } } )$ <sub>,</sub> this sim<sub>p</sub>lifies to the form <sub>p</sub>resented in the main <sub>p</sub>a<sub>p</sub>er

$$
\dot { Y } _ { t } = \pmb \mu \left( t , Y _ { t } + \int _ { 0 } ^ { t } \mathsf { o d } W _ { s } ~ \pmb \sigma ( s ) \right) , \qquad Y _ { 0 } = \pmb x ,\tag{24}
$$

and the Itô and Stratonovich inter<sub>p</sub>retations coincide.

## A.4 The Wong–Zakai theorem for multi-dimensional SDEs

Consider the same SDE from the <sub>p</sub>revious subsection in E<sub>q</sub>uation (18). A classic result shown b<sub>y</sub> Won<sub>g</sub> and Zakai (1965a,b) for one-dimensional SDEs and Stroock and Varadhan (1972) for multi-dimensional SDEs is that if $W ^ { \varepsilon }$ <sup>d</sup>enotes some natural smoot<sup>h</sup> �-approximation to a <sup>�</sup>-<sup>d</sup>imensiona<sup>l</sup> Brownian motion, an<sup>d</sup> $\boldsymbol { x } ^ { \varepsilon }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> solution to a random ODE

$$
\begin{array} { r } { \dot { { \mathbf { x } } } _ { t } ^ { \varepsilon } = \pmb { \mu } ( t , { \mathbf { x } } _ { t } ^ { \varepsilon } ) + \pmb { \sigma } ( t , { \mathbf { x } } _ { t } ^ { \varepsilon } ) \dot { W } _ { t } ^ { \varepsilon } , } \end{array}\tag{25}
$$

![](images/220e968a8b9d72757de926c7b10ede9278645816622f448ebf5852b84921a9c6.jpg)  
Time �  
Figure 8 Illustration of the Wong–Zakai principle: as piecewise linear Brownian motion converges to the true Brownian path, th<sub>e so</sub>l<sub>u</sub>ti<sub>on s</sub>t<sub>a</sub>t<sub>e</sub> $\pmb { X } _ { t } ^ { ( n ) }$ conver<sub>g</sub>es likewise.

th<sub>en</sub> $\boldsymbol { x } ^ { \varepsilon }$ conver<sub>g</sub>es in <sub>p</sub>robabilit<sub>y</sub> to the SDE in E<sub>q</sub>uation (18). The standard construction for the Won<sub>g</sub>–Zakai ar<sub>g</sub>ument is that of <sub>p</sub>iecewise linear inter<sub>p</sub>olation. For a mesh size � define $t _ { k } = k \varepsilon$ <sub>an</sub>d f<sub>or</sub> $t \in \left[ t _ { k } , t _ { k + 1 } \right]$ d<sub>e</sub>fi<sub>ne</sub>

$$
W _ { t } ^ { \varepsilon } = W _ { t _ { k } } + { \frac { t - t _ { k } } { \varepsilon } } ( W _ { t _ { k + 1 } } - W _ { t _ { k } } ) .\tag{26}
$$

Th<sub>en eac</sub>h <sub>pa</sub>th $t \mapsto W _ { t } ^ { \varepsilon }$ i<sub>s s</sub>m<sub>oo</sub>th <sub>a</sub>nd $W _ { t } ^ { \varepsilon } \to W$ uniforml<sub>y</sub> on com<sub>p</sub>act time intervals in <sub>p</sub>robabilit<sub>y</sub>. Other <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n<sub>s a</sub>r<sub>e poss</sub>ibl<sub>e,</sub> b<sub>u</sub>t w<sub>e e</sub>lid<sub>e suc</sub>h di<sub>scuss</sub>i<sub>o</sub>n<sub>s.</sub> S<sub>uc</sub>h <sub>a</sub>r<sub>gu</sub>m<sub>e</sub>nt<sub>s ca</sub>n <sub>a</sub>l<sub>so</sub> b<sub>e p</sub>r<sub>ese</sub>nt<sub>e</sub>d m<sub>o</sub>r<sub>e ge</sub>n<sub>e</sub>r<sub>a</sub>ll<sub>y</sub> in the lan<sub>g</sub>ua<sub>g</sub>e of rou<sub>g</sub>h <sub>p</sub>aths (Friz and Hairer 2020; Kid<sub>g</sub>er 2022).

## B On the Brownian path

In this section we discuss various details concernin<sub>g</sub> a<sub>pp</sub>roximations of the Brownian <sub>p</sub>ath<sub>,</sub> <sub>p</sub>roofs of various <sub>p</sub>ro<sub>p</sub>erties of the shifted Le<sub>g</sub>endre re<sub>p</sub>resentation<sub>,</sub> and closed forms for the Itô inte<sub>g</sub>ral of the FM SDE under common sett<sup>i</sup>n<sub>g</sub>s.

## B.1 Representations of the Brownian path

In the main <sub>p</sub>a<sub>p</sub>er we briefl<sub>y</sub> discussed several wa<sub>y</sub>s to re<sub>p</sub>resent the Brownian <sub>p</sub>ath $\{ W _ { t } \} _ { t \in [ 0 , 1 ] }$ . Here we <sub>p</sub>rovide a more thorou<sub>g</sub>h discussion on these re<sub>p</sub>resentations.

## B.1.1 Dyadic increment tree

One wa<sub>y</sub> to a<sub>pp</sub>roximate the true Brownian <sub>p</sub>ath is via a d<sub>y</sub>adic tree of Brownian increments (see Fi<sub>g</sub>ure 9).

F<sub>o</sub>r <sub>so</sub>m<sub>e</sub> int<sub>e</sub>rv<sub>a</sub>l $[ s , t ]$ <sub>,</sub> <sub>pa</sub>rtiti<sub>o</sub>n it int<sub>o</sub> $2 ^ { L }$ <sub>equa</sub>l <sub>su</sub>b-int<sub>e</sub>rv<sub>a</sub>l<sub>s</sub> $\left[ t _ { l } , t _ { l + 1 } \right]$ f<sub>or</sub> $0 \leq l < 2 ^ { L }$ <sub>w</sub>h<sub>ere</sub> $t _ { l } = s + l ( t - s ) 2 ^ { - L }$ <sub>, so</sub> th<sub>a</sub>t $t _ { 0 } ~ = ~ s$ <sub>a</sub>nd $t _ { 2 ^ { L } } ~ = ~ t$ . Then sam<sub>p</sub>le $2 ^ { L }$ inde<sub>p</sub>endent Brownian increments, i.e.,

$$
{ \cal W } _ { t _ { l } , t _ { l + 1 } } \sim { \cal N } \left( \mathbf { 0 } , \frac { t - s } { 2 ^ { L } } { \cal I } \right) .\tag{27}
$$

![](images/ffd78d6f450ef8b2433cae289d79a98c8aba35c770e9fdd1044e563032ce7271.jpg)  
Figure 9 The dyadic increment tree illustrated.

![](images/17bfa4966c27ca86aa9414cc4ad97c0af705df68fa17b30045fb2530a3d84d34.jpg)  
Figure 10 Karhunen–Loève versus shifted Legendre expansion. We visualize the two approximations at various � for a sin<sub>g</sub>le one-dimensional Brownian <sub>p</sub>ath on $r \in [ 0 ,$ 1] simulated for 2048 steps.

N<sub>o</sub>d<sub>es</sub> f<sub>u</sub>rth<sub>e</sub>r <sub>u</sub> th<sub>e</sub> tr<sub>ee ca</sub>n b<sub>e co</sub>n<sub>s</sub>tr<sub>uc</sub>t<sub>e</sub>d vi<sub>a</sub> $\boldsymbol { W _ { s , t } } = \boldsymbol { W _ { s , u } } + \boldsymbol { W _ { u , t } }$ f<sub>or</sub> th<sub>e m</sub>id<sub>po</sub>i<sub>n</sub>t <sub>�.</sub> Thi<sub>s</sub> f<sub>o</sub>ll<sub>ows</sub> th<sub>e c</sub>l<sub>ass</sub>i<sub>c</sub> construction of Lév<sub>y</sub> (1940) viewed as an increment tree. Instead of buildin<sub>g</sub> the tree from the bottom u<sub>p</sub>, one <sub>ca</sub>n b<sub>u</sub>ild fr<sub>o</sub>m th<sub>e</sub> t<sub>op</sub> d<sub>o</sub>wn b<sub>y</sub> <sub>sa</sub>m<sub>p</sub>lin<sub>g</sub> <sub>eac</sub>h mid<sub>po</sub>int <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>ll<sub>y</sub> <sub>o</sub>n it<sub>s</sub> <sub>a</sub>lr<sub>ea</sub>d<sub>y</sub> <sub>sa</sub>m<sub>p</sub>l<sub>e</sub>d <sub>e</sub>nd<sub>po</sub>int<sub>s,</sub> vi<sub>a</sub> th<sub>e</sub> Brownian <sup>b</sup>ri<sup>d</sup>ge, viz.

$$
W _ { u } \mid W _ { s } , W _ { t } \sim { \cal N } \left( W _ { s } + \frac { u - s } { t - s } ( W _ { t } - W _ { s } ) , \frac { ( u - s ) ( t - u ) } { t - s } I \right) ,\tag{28}
$$

f<sub>or</sub> $0 \leq s < u < t$ . Wor<sup>k</sup> on SDE so<sup>l</sup>vers exp<sup>l</sup>ores t<sup>h</sup>ese Brownian trees <sup>f</sup>or a<sup>d</sup>aptive step-size so<sup>l</sup>vers an<sup>d</sup> recursivel<sub>y</sub> a<sub>pp</sub>lies E<sub>q</sub>uation (28) to sam<sub>p</sub>le the Brownian motion at an<sub>y</sub> mid<sub>p</sub>oint, constructin<sub>g</sub> the tree (Gaines and L<sub>y</sub>ons 1997). Storin<sub>g</sub> the whole tree, however, is memor<sub>y</sub> intensive. Work on neural SDE solvers (Li et al. 2020) has made use of s<sub>p</sub>littable <sub>p</sub>seudo-random number <sub>g</sub>enerators (Claessen and Pałka 2013; Salmon et al. 2011) which can deterministicall<sub>y</sub> create two seeds <sub>g</sub>iven an existin<sub>g</sub> seed. Then one can evaluate the Brownian motion at an<sub>y p</sub>oint in time b<sub>y</sub> recursivel<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the Brownian tree construction to rebuild the tree until the <sub>q</sub>uer<sub>y</sub> mid<sub>p</sub>oint time is suitabl<sub>y</sub> close to the desired timeste<sub>p</sub>. Later work b<sub>y</sub> Jelinčič et al. (2024) and Kid<sub>g</sub>er et al. (2021) has im<sub>p</sub>roved u<sub>p</sub>on the <sub>p</sub>rocess of traversin<sub>g</sub> and storin<sub>g</sub> this tree.

A <sub>p</sub>iecewise <sub>p</sub>ol<sub>y</sub>nomial can be built from this <sub>p</sub>ath usin<sub>g</sub> each coeficient as the slo<sub>p</sub>e for a linear inter<sub>p</sub>olation b<sub>e</sub>tw<sub>ee</sub>n <sub>eac</sub>h b<sub>ou</sub>nd<sub>a</sub>r<sub>y po</sub>int <sub>a</sub>t tim<sub>es</sub> $t _ { l } , N . B .$ , in practice one would just pass the $2 ^ { L }$ <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e ne</sub>t<sub>wor</sub>k as is done in Pan et al. (2026)—althou<sub>g</sub>h we should note that the<sub>y</sub> onl<sub>y</sub> use de<sub>p</sub>th � and i<sub>g</sub>nore an<sub>y</sub> <sub>p</sub>arent nodes i<sub>n</sub> th<sub>e</sub> t<sub>ree.</sub>

## B.1.2 Karhunen–Loève expansion

A classic tool in stochastic anal<sub>y</sub>sis is the Karhunen–Loève ex<sub>p</sub>ansion of the Brownian <sub>p</sub>ath which is a Fourieres<sub>q</sub>ue decom<sub>p</sub>osition of the Brownian <sub>p</sub>ath (Karhunen 1950; Loève 1945). The resultin<sub>g</sub> ex<sub>p</sub>ansion over the interval [0, 1] can be written as

$$
W _ { r } ^ { \mathrm { K L } , N } : = \sum _ { n = 1 } ^ { N } \frac { 1 } { ( n - \frac { 1 } { 2 } ) \pi } \xi _ { n } \sqrt { 2 } \sin \left( ( n - \frac { 1 } { 2 } ) \pi r \right) , \qquad \xi _ { n } \sim N ( 0 , I ) .\tag{29}
$$

T<sup>h</sup>e expansion is g<sup>l</sup>o<sup>b</sup>a<sup>l</sup>, i.e., a <sup>d</sup>egree-� truncation is a <sup>f</sup>unctiona<sup>l</sup> o<sup>f</sup> t<sup>h</sup>e rea<sup>l</sup>ize<sup>d</sup> pat<sup>h</sup> on t<sup>h</sup>e w<sup>h</sup>o<sup>l</sup>e <sup>h</sup>orizon [0, 1]. Unfortunately, this expansion comes with some drawbacks for our setting. One is that there is no interval al<sub>g</sub>ebra (Chen relations) for the coeficients of this ex<sub>p</sub>ansion. To elucidate, for the d<sub>y</sub>adic tree there was the sim<sub>p</sub>le com<sub>p</sub>osition law $\boldsymbol { W _ { s , t } } \ : = \ : \boldsymbol { W _ { s , u } } \ : + \ : \boldsymbol { W _ { u , t } }$ <sub>, a</sub>nd w<sub>e s</sub>h<sub>o</sub>w in Th<sub>eo</sub>r<sub>e</sub>m B<sub>.3 a co</sub>m<sub>pos</sub>iti<sub>o</sub>n l<sub>a</sub>w f<sub>o</sub>r th<sub>e s</sub>hift<sub>e</sub>d Le endre re<sub>p</sub>resentation.

![](images/007f0135297d257cef05ab077f51db88a578bf7b88556596f6f6cf8e421d156f.jpg)  
Figure 11 Karhunen–Loève vs. shifted Legendre approximation error.

Additionall<sub>y,</sub> the ex<sub>p</sub>ansion is not ada<sub>p</sub>ted to the Brownian filtration as it is a <sub>g</sub>lobal re<sub>p</sub>resentation. The stochastic p<sup>r</sup>ocess $\{ X _ { t } \} _ { t \in [ 0 , 1 ] }$ which is the solution to our SDE in E<sub>q</sub>uation (1) is ada<sub>p</sub>ted to �, $i . e . , X _ { t }$ i<sub>s a</sub>n $\mathcal { F } _ { t }$ <sub>-measura</sub>bl<sub>e</sub> <sub>process</sub> f<sub>or eac</sub>h $\mathcal { F } _ { t }$ in the filtration �. Thus the process $X _ { t }$ ind<sub>uce</sub>d b<sub>y</sub> th<sub>e</sub> l<sub>ea</sub>rn<sub>e</sub>d Itô m<sub>ap co</sub>nditi<sub>o</sub>n<sub>e</sub>d <sub>o</sub>n thi<sub>s</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l K<sub>a</sub>rh<sub>u</sub>n<sub>e</sub>n–L<sub>o</sub>èv<sub>e e</sub>x<sub>pa</sub>n<sub>s</sub>i<sub>o</sub>n w<sub>ou</sub>ld n<sub>o</sub> l<sub>o</sub>n<sub>ge</sub>r <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>ll<sub>y</sub> b<sub>e</sub> $\mathcal { F } _ { t }$ -m<sub>easu</sub>r<sub>a</sub>bl<sub>e.</sub> F<sub>o</sub>r n<sub>o</sub>n-<sub>a</sub>dditiv<sub>e</sub> n<sub>o</sub>i<sub>se</sub> SDE<sub>s</sub> thi<sub>s</sub> m<sub>a</sub>k<sub>es suc</sub>h <sub>a</sub> m<sub>ap</sub> ill-<sub>pose</sub>d <sub>as</sub> th<sub>e</sub> Itô int<sub>eg</sub>r<sub>a</sub>l i<sub>s o</sub>nl<sub>y</sub> d<sub>e</sub>fin<sub>e</sub>d f<sub>o</sub>r <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d <sub>p</sub>r<sub>ocesses.</sub>

Remark B.1. For the case of an additive noise Itô SDE the learned Itô map conditioned on the global Karhunen– L<sub>o</sub>èv<sub>e e</sub>x<sub>pa</sub>n<sub>s</sub>i<sub>o</sub>n i<sub>s</sub> w<sub>e</sub>ll-d<sub>e</sub>fin<sub>e</sub>d<sub>,</sub> h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r<sub>,</sub> it i<sub>s</sub> n<sub>o</sub>t <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub>d t<sub>o</sub> b<sub>e a</sub>n <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d <sub>p</sub>r<sub>ocess.</sub>

Th<sub>ere</sub> i<sub>s</sub> <sub>a</sub>l<sub>so</sub> th<sub>e</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>pro</sub>bl<sub>em</sub> <sub>o</sub>f <sub>o</sub>b<sub>scur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>use</sub>f<sub>u</sub>l <sub>pa</sub>th i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>seen</sub> b<sub>y</sub> th<sub>e</sub> <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>,</sub> $i . e . _ { \cdot }$ <sub>,</sub> f<sub>or</sub> learning a jump from � to � the network has additional information on $[ 0 , s ) \cup ( t , 1 ]$ <sub>w</sub>hi<sub>c</sub>h i<sub>s no</sub>t h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l <sub>an</sub>d <sub>a</sub>dd<sub>s more</sub> “<sub>no</sub>i<sub>se</sub>” t<sub>o</sub> th<sub>e re</sub>l<sub>evan</sub>t <sub>pa</sub>th <sub>s</sub>i<sub>gna</sub>l <sub>on</sub> $[ s , t ]$ . Em<sub>p</sub>iricall<sub>y,</sub> this is verified b<sub>y</sub> our ex<sub>p</sub>eriments and is likel<sub>y</sub> wh<sub>y</sub> Pan et al. (2026) needed to add the first-order Brownian increment to their model.

## B.1.3 Shifted Legendre polynomial expansion

We restate the shifted Le<sub>g</sub>endre ex<sub>p</sub>ansion discussed in the main <sub>p</sub>a<sub>p</sub>er below for convenience: writin<sub>g</sub> ${ \widetilde { P } } _ { n }$ f<sub>or</sub> the �-th shifted Legendre polynomial on [0, 1], the coeficients and the degree-� reconstruction on $[ s , t ]$ are

$$
I _ { s , t } ^ { ( n ) } : = \int _ { s } ^ { t } \mathrm { d } W _ { r } \widetilde { P } _ { n } \bigg ( \frac { r - s } { t - s } \bigg ) \sim N \bigg ( 0 , \frac { t - s } { 2 n + 1 } I \bigg ) ,\tag{30}
$$

$$
\pmb { W } _ { u , \nu } ^ { ( N , s , t ) } : = \sum _ { n = 0 } ^ { N - 1 } \frac { 2 n + 1 } { t - s } \pmb { I } _ { s , t } ^ { ( n ) } \int _ { u } ^ { \nu } \mathrm { d } r \widetilde { P } _ { n } \Big ( \frac { r - s } { t - s } \Big ) .\tag{31}
$$

N.B., as $\int _ { 0 } ^ { 1 } \widetilde { P } _ { n } = 0$ <sup>f</sup>or ever<sub>y</sub> $n \geq 1$ <sub>,</sub> th<sub>e</sub> $n = 0$ t<sub>e</sub>rm i<sub>s</sub> th<sub>e pu</sub>r<sub>e c</sub>h<sub>o</sub>rd <sub>o</sub>v<sub>e</sub>r th<sub>e</sub> int<sub>e</sub>rv<sub>a</sub>l <sub>a</sub>nd hi<sub>g</sub>h<sub>e</sub>r m<sub>o</sub>d<sub>es a</sub>r<sub>e</sub> z<sub>e</sub>r<sub>o</sub>- <sub>p</sub>inned brid<sub>g</sub>e harmonics. Foster et al. (2020) introduces this <sub>p</sub>ol<sub>y</sub>nomial a<sub>pp</sub>roximation as a Karhunen–Loève expansion in t<sup>h</sup>e weighted inner product wit<sup>h</sup> weig<sup>h</sup>ting

$$
w _ { r } = \frac { 1 } { r ( 1 - r ) } .\tag{32}
$$

Fo<sup>ll</sup>owing t<sup>h</sup>eir conventions, we reserve t<sup>h</sup>e term Karhunen–Loève expansion <sup>f</sup>or t<sup>h</sup>e sine expansion a<sup>b</sup>ove an<sup>d</sup> use the shifted Le endre ol nomial a roximation to refer to E uations (30) and (31). The standard Karhunen– Loève ex<sub>p</sub>ansion <sub>p</sub>rovides $L ^ { 2 }$ o<sub>p</sub>timalit<sub>y</sub> via a sinusoidal basis and o<sub>p</sub>timizes the unwei<sub>g</sub>hted error<sub>,</sub> conversel<sub>y,</sub> the shifted Le<sub>g</sub>endre ex<sub>p</sub>ansion <sub>p</sub>rovides wei<sub>g</sub>hted $L ^ { 2 }$ o<sub>p</sub>timalit<sub>y</sub> with the wei<sub>g</sub>htin<sub>g</sub> $w _ { r }$ <sub>p</sub>enalizin<sub>g</sub> terms near the end <sub>p</sub>oints (normalized to the interval [0, 1]). We also have the benefits of interval al<sub>g</sub>ebra via the Chen relations and the nice analo<sub>g</sub>ue to the s<sub>p</sub>ace-time Lév<sub>y</sub> areas which a<sub>pp</sub>ear in the stochastic Ta<sub>y</sub>lor ex<sub>p</sub>ansion.

Remark B.2. The dyadic tree can be seen as a special case of the shifted Legendre expansion with $N = 1$ , s<sup>i</sup>nce, via the Chen relations for the increments, we can construct a dyadic tree out of intervals partitioning [0, 1].

## B.2 Derivatives of the shifted Legendre polynomial approximation

A <sub>s</sub>tr<sub>a</sub>i<sub>g</sub>htf<sub>o</sub>rw<sub>a</sub>rd r<sub>esu</sub>lt <sub>o</sub>f th<sub>e</sub> <sub>s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e</sub> <sub>po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l <sub>app</sub>r<sub>o</sub>xim<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th i<sub>s</sub> th<sub>a</sub>t w<sub>e</sub> h<sub>a</sub>v<sub>e</sub> <sub>c</sub>l<sub>ose</sub>d f<sub>o</sub>rm d<sub>e</sub>riv<sub>a</sub>tiv<sub>es.</sub>

Lemma B.1 (Derivatives of $W _ { [ u , \nu ] } ^ { ( N , s , t ) } )$ . For Equation (3) we have:

$$
\frac { \mathrm { d } } { \mathrm { d } u } W _ { u , \nu } ^ { ( N , s , t ) } = - \sum _ { n = 0 } ^ { N - 1 } \frac { 2 n + 1 } { t - s } I _ { s , t } ^ { ( n ) } \widetilde { P } _ { n } \left( \frac { u - s } { t - s } \right) ,\tag{33}
$$

$$
\frac { \mathrm { d } } { \mathrm { d } \nu } W _ { u , \nu } ^ { ( N , s , t ) } = \sum _ { n = 0 } ^ { N - 1 } \frac { 2 n + 1 } { t - s } I _ { s , t } ^ { ( n ) } \widetilde { P } _ { n } \left( \frac { \nu - s } { t - s } \right) .\tag{34}
$$

Proof. T<sup>h</sup>is <sup>f</sup>o<sup>ll</sup>ows <sup>b</sup>y t<sup>h</sup>e <sup>f</sup>un<sup>d</sup>amenta<sup>l</sup> t<sup>h</sup>eorem o<sup>f</sup> ca<sup>l</sup>cu<sup>l</sup>us app<sup>l</sup>ie<sup>d</sup> to $\int _ { u } ^ { \nu }$ d<sub>�.</sub>

For $r \in [ s , t ]$ l<sub>e</sub>t <sub>us</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>s</sub>h<sub>or</sub>th<sub>an</sub>d

$$
\dot { W } _ { r } ^ { ( N , s , t ) } : = \left. \frac { \mathrm { d } } { \mathrm { d } \tau } W _ { s , \tau } ^ { ( N , s , t ) } \right| _ { \tau = r } = \sum _ { n = 0 } ^ { N - 1 } \frac { 2 n + 1 } { t - s } I _ { s , t } ^ { ( n ) } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) ,\tag{35}
$$

lik<sub>e</sub>wi<sub>se</sub> ${ \pmb W } _ { r } ^ { ( N , s , t ) } = { \pmb W } _ { s , r } ^ { ( N , s , t ) }$

## B.3 Extended Chen relations for the shifted Legendre polynomial approximation

We discuss the Chen relations for com<sub>p</sub>osin<sub>g</sub> two de<sub>g</sub>ree-� <sub>p</sub>ol<sub>y</sub>nomial a<sub>pp</sub>roximations of the <sub>p</sub>ath on $[ s , u ]$ <sub>a</sub>nd $[ u , t ]$ into a sin<sub>g</sub>le � -th de<sub>g</sub>ree <sub>p</sub>ol<sub>y</sub>nomial over $[ s , t ]$ of the same underl<sub>y</sub>in<sub>g</sub> Brownian <sub>p</sub>ath.

Composition along the midpoint. For some $s \mathrm { ~ < ~ } t$ l<sub>e</sub>t $u = ( s + t ) / 2$ b<sub>e</sub> th<sub>e</sub> mid<sub>po</sub>int<sub>, a</sub>nd <sub>suppose</sub> tw<sub>o su</sub>b-<sub>po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l<sub>s a</sub>r<sub>e</sub> d<sub>e</sub>fin<sub>e</sub>d <sub>as</sub> ${ W ^ { ( N , s , u ) } }$ <sub>a</sub>nd ${ W ^ { ( N , u , t ) } }$ <sub>.</sub> N<sub>o</sub>t<sub>e, we om</sub>it th<sub>e</sub> f<sub>ree var</sub>i<sub>a</sub>bl<sub>es</sub> h<sub>ere</sub> b<sub>u</sub>t th<sub>ey are</sub> th<sub>ere</sub> im<sub>p</sub>licitl<sub>y</sub>. Our <sub>g</sub>oal is to combine these two sub-<sub>p</sub>ol<sub>y</sub>nomials into a lar<sub>g</sub>er one $W ^ { ( N , s , t ) }$ <sub>.</sub> N<sub>o</sub>w M<sub>c</sub>C<sub>a</sub>ll<sub>u</sub>m <sub>e</sub>t <sub>a</sub>l<sub>.</sub> (2026, Pro<sub>p</sub>osition $\mathbf { A . } 5 )$ <sub>s</sub>h<sub>owe</sub>d th<sub>a</sub>t <sub>w</sub>h<sub>en</sub> <sub>�</sub> i<sub>s</sub> th<sub>e</sub> <sub>m</sub>id<sub>po</sub>i<sub>n</sub>t <sub>we</sub> h<sub>ave</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>compos</sub>iti<sub>on</sub> <sub>ru</sub>l<sub>es—a</sub>l<sub>so</sub> k<sub>nown as</sub> Ch<sub>en re</sub>l<sub>a</sub>ti<sub>ons—</sub>f<sub>or</sub> $0 \leq n < N$ ，

$$
I _ { s , t } ^ { ( n ) } = \sum _ { m = 0 } ^ { n } c _ { n , m } \left( I _ { s , u } ^ { ( m ) } + ( - 1 ) ^ { n + m } I _ { u , t } ^ { ( m ) } \right) ,\tag{36}
$$

<sub>w</sub>h<sub>ere</sub> $c _ { n , m }$ i<sub>s</sub> d<sub>e</sub>fin<sub>e</sub>d <sub>as</sub>

$$
c _ { n , m } : = ( - 1 ) ^ { n } ( 2 m + 1 ) \sum _ { k = m } ^ { n } \left( - { \frac { 1 } { 2 } } \right) ^ { k } { \frac { ( n + k ) ! } { ( n - k ) ! ( k - m ) ! ( k + m + 1 ) ! } } .\tag{37}
$$

However<sub>,</sub> this Chen relation onl<sub>y</sub> works at the mid<sub>p</sub>oint. Can we extend this to an<sub>y</sub> $u \in ( s , t ) \ d \colon$

N<sub>o</sub>w <sub>co</sub>n<sub>s</sub>id<sub>e</sub>r <sub>so</sub>m<sub>e</sub> $u \in ( s , t )$ <sub>a</sub>nd tw<sub>o su</sub>b-Br<sub>o</sub>wni<sub>a</sub>n <sub>po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l<sub>s</sub> d<sub>e</sub>n<sub>o</sub>t<sub>e</sub>d ${ W _ { r } ^ { ( N , s , u ) } }$ <sub>a</sub>nd ${ W _ { r } ^ { ( N , u , t ) } }$ . <sup>W</sup>e p<sup>r</sup>opose th<sub>e</sub> f<sub>o</sub>ll<sub>o</sub>win<sub>g co</sub>m<sub>pos</sub>iti<sub>o</sub>n l<sub>a</sub>w<sub>:</sub>

Lemma B.2. For $0 \leq n < N$ , the �-th degree coeficient $\pmb { I } _ { s , t } ^ { ( n ) }$ can be constructed from two sub-Brownian polynomials denoted $W _ { r } ^ { ( N , s , u ) }$ and ${ W _ { r } ^ { ( N , u , t ) } }$ as

$$
I _ { s , t } ^ { ( n ) } = \int _ { s } ^ { u } \mathrm { d } r \dot { W } _ { r } ^ { ( N , s , u ) } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) + \int _ { u } ^ { t } \mathrm { d } r \dot { W } _ { r } ^ { ( N , u , t ) } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) .\tag{38}
$$

Proof. For convenience we <sup>d</sup>e<sup>fi</sup>ne t<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing i<sup>d</sup>entities

$$
\mathbf { \boldsymbol { W } } _ { r } ^ { L } : = \mathbf { \boldsymbol { W } } _ { r } ^ { ( N , s , u ) } , \qquad \mathbf { \boldsymbol { W } } _ { r } ^ { R } : = \mathbf { \boldsymbol { W } } _ { r } ^ { ( N , u , t ) } ,\tag{39}
$$

<sub>a</sub>nd

$$
p _ { n } ( r ) : = \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) , \qquad p _ { n } ^ { L } ( r ) : = \widetilde { P } _ { n } \left( \frac { r - s } { u - s } \right) , \qquad p _ { n } ^ { R } ( r ) : = \widetilde { P } _ { n } \left( \frac { r - u } { t - u } \right) .\tag{40}
$$

Now clearl<sub>y</sub> we can write

$$
I _ { s , t } ^ { ( n ) } = \int _ { s } ^ { t } \mathrm { d } W _ { r } p _ { n } ( r ) ,\tag{41}
$$

$$
= \int _ { s } ^ { u } \mathrm { d } W _ { r } \ : p _ { n } ( r ) + \int _ { u } ^ { t } \mathrm { d } W _ { r } \ : p _ { n } ( r ) .\tag{42}
$$

Since $p _ { n }$ h<sub>as</sub> d<sub>eg</sub>r<sub>ee</sub> $n < N$ th<sub>e</sub>n th<sub>e</sub> L<sub>ege</sub>ndr<sub>e</sub> <sub>po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l <sub>ca</sub>n b<sub>e</sub> writt<sub>e</sub>n in t<sub>e</sub>rm<sub>s</sub> <sub>o</sub>f th<sub>e</sub> <sub>su</sub>b-L<sub>ege</sub>ndr<sub>e</sub> <sub>po</sub>l<sub>y</sub>n<sub>o</sub>- <sub>m</sub>i<sub>a</sub>l<sub>s</sub> $p _ { n } ^ { L }$ <sub>a</sub>nd $p _ { n } ^ { R }$ , i.e.,

$$
p _ { n } ( r ) = \sum _ { m = 0 } ^ { n } a _ { m } ^ { L } p _ { m } ^ { L } ( r ) , \qquad a _ { m } ^ { L } = \frac { 2 m + 1 } { u - s } \int _ { s } ^ { u } \mathrm { d } r ~ p _ { n } ( r ) p _ { m } ^ { L } ( r ) , \qquad r \in [ s , u ] ,\tag{43}
$$

$$
p _ { n } ( r ) = \sum _ { m = 0 } ^ { n } a _ { m } ^ { R } p _ { m } ^ { R } ( r ) ,
$$

$$
a _ { m } ^ { R } = \frac { 2 m + 1 } { t - u } \int _ { u } ^ { t } { \mathrm { d } } r ~ p _ { n } ( r ) p _ { m } ^ { R } ( r ) ,
$$

$$
r \in [ u , t ] ,\tag{44}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e secon</sub>d <sub>co</sub>l<sub>umn o</sub>f <sub>equa</sub>liti<sub>es</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d F<sub>our</sub>i<sub>er–</sub>L<sub>egen</sub>d<sub>re coe</sub>fi<sub>c</sub>i<sub>en</sub>t f<sub>ormu</sub>l<sub>a.</sub> B<sub>y</sub> th<sub>e</sub> lin<sub>ea</sub>rit<sub>y o</sub>f th<sub>e</sub> Itô int<sub>eg</sub>r<sub>a</sub>l <sub>a</sub>nd d<sub>e</sub>finiti<sub>o</sub>n<sub>s o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s</sub> w<sub>e</sub> h<sub>a</sub>v<sub>e</sub> th<sub>a</sub>t

$$
\int _ { s } ^ { u } \mathrm { d } W _ { r } \ p _ { n } ( r ) = \sum _ { m = 0 } ^ { n } a _ { m } ^ { L } I _ { s , u } ^ { ( m ) } ,\tag{45}
$$

$$
\int _ { u } ^ { t } \mathrm { d } W _ { r } \ : p _ { n } ( r ) = \sum _ { m = 0 } ^ { n } a _ { m } ^ { R } I _ { u , t } ^ { ( m ) } .\tag{46}
$$

B<sub>y</sub> definition the derivatives of the <sub>p</sub>ol<sub>y</sub>nomial sub-Brownian <sub>p</sub>aths are

$$
\dot { \pmb { W } } _ { r } ^ { L } = \sum _ { m = 0 } ^ { N - 1 } \frac { 2 m + 1 } { u - s } \pmb { I } _ { s , u } ^ { ( m ) } p _ { m } ^ { L } ( r ) ,\tag{47}
$$

$$
\dot { \pmb { W } } _ { r } ^ { R } = \sum _ { m = 0 } ^ { N - 1 } \frac { 2 m + 1 } { t - u } \pmb { I } _ { u , t } ^ { ( m ) } \pmb { p } _ { m } ^ { R } ( r ) .\tag{48}
$$

N<sub>o</sub>w <sub>us</sub>in<sub>g</sub> th<sub>e</sub> id<sub>e</sub>ntiti<sub>es</sub> f<sub>o</sub>r $a _ { m } ^ { L }$ <sub>a</sub>nd $a _ { m } ^ { R }$ <sub>we</sub> k<sub>now</sub> th<sub>a</sub>t t<sub>erms w</sub>ith $m > n$ make no contribution b<sub>y</sub> ortho<sub>g</sub>onalit<sub>y</sub> <sub>an</sub>d th<sub>us</sub>

$$
\sum _ { m = 0 } ^ { n } a _ { m } ^ { L } { \cal I } _ { s , u } ^ { ( m ) } = \int _ { s } ^ { u } \mathrm { d } r \ \dot { \cal W } _ { r } ^ { L } p _ { n } ( r ) ,\tag{49}
$$

$$
\sum _ { m = 0 } ^ { n } a _ { m } ^ { R } { \cal I } _ { u , t } ^ { ( m ) } = \int _ { u } ^ { t } \mathrm { d } r \dot { W } _ { r } ^ { R } p _ { n } ( r ) .\tag{50}
$$

Th<sub>us we s</sub>h<sub>ow</sub>

$$
I _ { s , t } ^ { ( n ) } = \int _ { s } ^ { u } \mathrm { d } r \dot { W } _ { r } ^ { ( N , s , u ) } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) + \int _ { u } ^ { t } \mathrm { d } r \dot { W } _ { r } ^ { ( N , u , t ) } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) ,\tag{51}
$$

th<sub>ere</sub>b<sub>y</sub> fi<sub>n</sub>i<sub>s</sub>hi<sub>ng</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub>

We have now shown a formula for com<sub>p</sub>osin<sub>g</sub> two Brownian <sub>p</sub>ol<sub>y</sub>nomials over $[ s , u ]$ <sub>a</sub>nd $[ u , t ]$ to <sub>g</sub>et t<sup>h</sup>e coe<sup>fi</sup>cients <sup>f</sup>or t<sup>h</sup>e po<sup>l</sup>ynomia<sup>l</sup> on [�,�]. However, can we tractably compute t<sup>h</sup>is composition <sup>l</sup>aw? We now writ<sub>e a</sub> tr<sub>ac</sub>t<sub>a</sub>bl<sub>e co</sub>m<sub>pos</sub>iti<sub>o</sub>n l<sub>a</sub>w b<sub>e</sub>l<sub>o</sub>w<sub>.</sub>

Theorem B.3 (Composition law for the shifted Legendre coefficients). For $0 \leq n < N$ , the �-th degree coeficient $I _ { s , t } ^ { ( n ) }$ can be constructed from the coeficients oftwo sub-Brownian polynomials over intervals [�, �] and $[ u , t ]$ respectively for some arbitrary � $\in \left( s , t \right)$ . This composition law is given as

$$
I _ { s , t } ^ { ( n ) } = \sum _ { m = 0 } ^ { n } \left[ A _ { n , m } ( \lambda ) I _ { s , u } ^ { ( m ) } + B _ { n , m } ( \lambda ) I _ { u , t } ^ { ( m ) } \right] ,\tag{52}
$$

where $\lambda = ( u - s ) / ( t - s ) \in ( 0 , 1 )$ and the coeficients $A _ { n , m } ( \lambda )$ and $B _ { n , m } ( \lambda )$ are defined as

$$
A _ { n , m } ( \lambda ) = ( - 1 ) ^ { n } ( 2 m + 1 ) \sum _ { k = m } ^ { n } ( - \lambda ) ^ { k } \frac { ( n + k ) ! } { ( n - k ) ! ( k - m ) ! ( k + m + 1 ) ! } ,\tag{53}
$$

$$
B _ { n , m } ( \lambda ) = ( - 1 ) ^ { m } ( 2 m + 1 ) \sum _ { k = m } ^ { n } ( - ( 1 - \lambda ) ) ^ { k } \frac { ( n + k ) ! } { ( n - k ) ! ( k - m ) ! ( k + m + 1 ) ! } .\tag{54}
$$

Proof. Reca<sup>ll</sup> t<sup>h</sup>at t<sup>h</sup>e two su<sup>b</sup>-Brownian po<sup>l</sup>ynomia<sup>l</sup>s <sup>h</sup>ave <sup>d</sup>erivatives

$$
\dot { \pmb { W } } _ { r } ^ { L } = \sum _ { m = 0 } ^ { N - 1 } \frac { 2 m + 1 } { u - s } \pmb { I } _ { s , u } ^ { ( m ) } p _ { m } ^ { L } ( r ) ,\tag{55}
$$

$$
\dot { \pmb { W } } _ { r } ^ { R } = \sum _ { m = 0 } ^ { N - 1 } \frac { 2 m + 1 } { t - u } \pmb { I } _ { u , t } ^ { ( m ) } p _ { m } ^ { R } ( r ) .\tag{56}
$$

Left interval. Set $x = ( r - s ) / ( u - s )$ <sub>,</sub> th<sub>en</sub> $r = s + ( u - s ) x$ <sub>a</sub>nd $\mathrm { d } r = ( u - s ) \mathrm { d } x$ <sub>,</sub> which substitutin<sub>g</sub> into E<sub>q</sub>uation (55) <sub>y</sub>ields

$$
\int _ { s } ^ { u } \mathrm { d } r \dot { W } _ { r } ^ { L } \widetilde { P } _ { n } \Big ( \frac { r - s } { t - s } \Big ) = \sum _ { m = 0 } ^ { N - 1 } ( 2 m + 1 ) I _ { s , u } ^ { ( m ) } \int _ { 0 } ^ { 1 } \mathrm { d } x \widetilde { P } _ { m } ( x ) \widetilde { P } _ { n } ( \lambda x ) ,\tag{57}
$$

wh<sub>e</sub>r<sub>e</sub> th<sub>e</sub> <sub>equa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>y</sub> $\lambda x = ( r - s ) / ( t - s )$ <sub>.</sub> N<sub>ow</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> <sub>our</sub> l<sub>e</sub>ft <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t $A _ { n , m } ( \lambda )$ t<sub>o</sub> b<sub>e</sub>

$$
A _ { n , m } ( \lambda ) : = ( 2 m + 1 ) \int _ { 0 } ^ { 1 } \mathrm { d } x \ \widetilde P _ { m } ( x ) \widetilde P _ { n } ( \lambda x ) .\tag{58}
$$

Since $\widetilde { P } _ { n } ( \lambda x )$ h<sub>as</sub> d<sub>eg</sub>r<sub>ee</sub> $n ,$ ortho<sub>g</sub>onalit<sub>y</sub> im<sub>p</sub>lies that $A _ { n , m } ( \lambda ) = 0$ <sub>w</sub>h<sub>en</sub> $m > n ,$ <sub>, an</sub>d th<sub>us</sub>

$$
\int _ { s } ^ { u } \mathrm { d } r \ \dot { W } _ { r } ^ { L } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) = \sum _ { m = 0 } ^ { n } A _ { n , m } ( \lambda ) I _ { s , u } ^ { ( m ) } .\tag{59}
$$

Right interval. For the right interval we have a similar calculation. Let $y = ( r - u ) / ( t - u )$ <sub>w</sub>ith $r = u + ( t - u ) y$ <sub>a</sub>nd $\mathrm { d } r = ( t - u ) \mathrm { d } y$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub> $( r - s ) / ( t - s ) = \lambda + ( 1 - \lambda ) y$ . Now via E<sub>q</sub>uation (56) we can show

$$
\int _ { u } ^ { t } \mathrm { d } r \dot { W } _ { r } ^ { R } \widetilde { P } _ { n } \Big ( \frac { r - s } { t - s } \Big ) = \sum _ { m = 0 } ^ { N - 1 } ( 2 m + 1 ) I _ { u , t } ^ { ( m ) } \int _ { 0 } ^ { 1 } \mathrm { d } y \widetilde { P } _ { m } ( y ) \widetilde { P } _ { n } ( \lambda + ( 1 - \lambda ) y ) .\tag{60}
$$

N<sub>o</sub>w w<sub>e</sub> d<sub>e</sub>fin<sub>e ou</sub>r ri<sub>g</sub>ht <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt t<sub>o</sub> b<sub>e</sub>

$$
B _ { n , m } ( \lambda ) : = ( 2 m + 1 ) \int _ { 0 } ^ { 1 } \mathrm { d } y \ \widetilde P _ { m } ( y ) \widetilde P _ { n } ( \lambda + ( 1 - \lambda ) y ) .\tag{61}
$$

Th<sub>us</sub> b<sub>y o</sub>rth<sub>ogo</sub>n<sub>a</sub>lit<sub>y</sub> w<sub>e</sub> h<sub>a</sub>v<sub>e</sub>

$$
\int _ { u } ^ { t } \mathrm { d } r \ \dot { W } _ { r } ^ { R } \widetilde { P } _ { n } \left( \frac { r - s } { t - s } \right) = \sum _ { m = 0 } ^ { n } B _ { n , m } ( \lambda ) I _ { u , t } ^ { ( m ) } .\tag{62}
$$

Closed-form coefficients. Consider the monomial expansion of the shifted Legendre polynomial, $i . e .$

$$
{ \widetilde { P } } _ { n } ( z ) = \sum _ { k = 0 } ^ { n } ( - 1 ) ^ { n - k } { \frac { ( n + k ) ! } { ( n - k ) ! ( k ! ) ^ { 2 } } } z ^ { k } .\tag{63}
$$

Also the shifted Le<sub>g</sub>endre moment identit<sub>y</sub> is

$$
\int _ { 0 } ^ { 1 } \mathrm { d } x \ x ^ { k } \widetilde { P } _ { m } ( x ) = \left\{ \begin{array} { l l } { 0 , } & { k < m , } \\ { \frac { ( k ! ) ^ { 2 } } { ( k - m ) ! ( k + m + 1 ) ! } , } & { k \ge m . } \end{array} \right.\tag{64}
$$

Combinin<sub>g</sub> these two e<sub>q</sub>uations into our definition for $A _ { n , m }$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
A _ { n , m } ( \lambda ) \overset { ( i ) } { = } ( 2 m + 1 ) \sum _ { k = 0 } ^ { n } ( - 1 ) ^ { n - k } \frac { ( n + k ) ! } { ( n - k ) ! ( k ! ) ^ { 2 } } \lambda ^ { k } \int _ { 0 } ^ { 1 } \mathrm { d } x x ^ { k } \widetilde { P } _ { m } ( x ) ,\tag{65}
$$

$$
\stackrel { ( i i ) } { = } ( 2 m + 1 ) \sum _ { k = m } ^ { n } ( - 1 ) ^ { n - k } \lambda ^ { k } \frac { ( n + k ) ! } { ( n - k ) ! ( k - m ) ! ( k + m + 1 ) ! } ,\tag{66}
$$

$$
= ( - 1 ) ^ { n } ( 2 m + 1 ) \sum _ { k = m } ^ { n } ( - \lambda ) ^ { k } { \frac { ( n + k ) ! } { ( n - k ) ! ( k - m ) ! ( k + m + 1 ) ! } } ,\tag{67}
$$

where (i) holds b<sub>y</sub> settin<sub>g</sub> $z = \lambda x$ in E<sub>q</sub>uation (63) and (ii) holds b<sub>y</sub> definition of E<sub>q</sub>uation $( 6 4 )$

Likewise<sub>,</sub> for the ri<sub>g</sub>ht interval the reflection identit<sub>y</sub> $\widetilde { P } _ { j } ( 1 - x ) = ( - 1 ) ^ { j } \widetilde { P } _ { j } ( x )$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { l } { { \displaystyle B _ { n , m } ( \lambda ) = ( - 1 ) ^ { n + m } A _ { n , m } ( 1 - \lambda ) , } } \\ { { \displaystyle \qquad = ( - 1 ) ^ { m } ( 2 m + 1 ) \sum _ { k = m } ^ { n } ( - ( 1 - \lambda ) ) ^ { k } \frac { ( n + k ) ! } { ( n - k ) ! ( k - m ) ! ( k + m + 1 ) ! } , } } \end{array}\tag{68}
$$

(69)

th<sub>ere</sub>b<sub>y</sub> fi<sub>n</sub>i<sub>s</sub>hi<sub>ng</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub>

Remark B.3. When $\begin{array} { r } { \lambda = \frac { 1 } { 2 } } \end{array}$ we recover the ori<sub>g</sub>inal result of McCallum et al. (2026, Lemma $\mathsf { A } . 4 )$

## B.4 Closed-form martingales for common schedules

We can define the Itô inte<sub>g</sub>ral of our additive noise as a martin<sub>g</sub>ale<sub>,</sub> $\begin{array} { r } { M _ { t } : = \int _ { t _ { 0 } } ^ { t } \mathrm { d } W _ { r } \ \nu _ { r } - } \end{array}$ mutatis mutandis t<sup>h</sup>e <sub>e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>ll<sub>y</sub> lift<sub>e</sub>d v<sub>e</sub>r<sub>s</sub>i<sub>o</sub>n—wh<sub>e</sub>r<sub>e</sub> $t _ { 0 } > 0$ i<sub>s so</sub>m<sub>e s</sub>m<sub>a</sub>ll <sub>co</sub>n<sub>s</sub>t<sub>a</sub>nt f<sub>o</sub>r n<sub>u</sub>m<sub>e</sub>ri<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>bilit <sub>.</sub> In ri<sub>o</sub>r w<sub>o</sub>rk thi<sub>s</sub> w<sub>as</sub> modelled with an extra difusion head (McCallum et al. 2026) or with numerical <sub>q</sub>uadrature (Pan et al. 2026), which incurs additional com<sub>p</sub>ute alon<sub>g</sub> with a<sub>pp</sub>roximation errors. We <sub>p</sub>refer to instead com<sub>p</sub>ute this martin<sub>g</sub>ale in <sub>c</sub>l<sub>ose</sub>d f<sub>o</sub>rm<sub>.</sub> F<sub>o</sub>r m<sub>os</sub>t d<sub>e</sub>finiti<sub>o</sub>n<sub>s o</sub>f $\nu _ { t }$ in th<sub>e</sub> lit<sub>e</sub>r<sub>a</sub>t<sub>u</sub>r<sub>e</sub> w<sub>e ca</sub>n r<sub>e</sub>writ<sub>e</sub> th<sub>e</sub> m<sub>a</sub>rtin <sub>a</sub>l<sub>e as a</sub> tim<sub>e</sub>-<sub>c</sub>h<sub>a</sub>n <sub>e</sub>d Brownian motion<sub>,</sub> $i . e . , M _ { t } = \widetilde { W } _ { \langle M \rangle }$ for a Brownian motion �e (distinct from the drivin �), where $\langle M \rangle _ { t }$ i<sub>s</sub> th<sub>e</sub> <sub>q</sub>uadratic variation of $M _ { t } ,$ , b<sub>y</sub> the Dambis–Dubins–Schwarz theorem (Dubins and Schwarz 1965); we avoid the technical com<sub>p</sub>lications of random sto<sub>pp</sub>in<sub>g</sub> times b<sub>y</sub> ensurin<sub>g</sub> that the <sub>q</sub>uadratic variation is a monotonic f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e p</sub>h<sub>ys</sub>i<sub>ca</sub>l tim<sub>e</sub> �<sub>.</sub> N<sub>o</sub>t<sub>a</sub>bl<sub>y,</sub> th<sub>e qua</sub>dr<sub>a</sub>ti<sub>c</sub> v<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n <sub>ca</sub>n b<sub>e co</sub>m<sub>pu</sub>t<sub>e</sub>d in <sub>c</sub>l<sub>ose</sub>d f<sub>o</sub>rm f<sub>o</sub>r <sub>co</sub>mm<sub>o</sub>n <sub>c</sub>h<sub>o</sub>i<sub>ces o</sub>f $\nu _ { t }$ (Blasingame and Liu 2026). We perform the shifted Legendre polynomial expansion of�e in the lift<sub>e</sub>d tim<sub>e</sub> ${ \boldsymbol { \tau } } = \langle \pmb { M } \rangle _ { t } ,$ over $[ \langle M \rangle _ { s } , \langle M \rangle _ { t } ]$ <sub>,</sub> to ensure <sub>p</sub>athwise a<sub>g</sub>reement between $M _ { t }$ <sub>an</sub>d <sub>our coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub>

## B.4.1 Linear

F<sub>o</sub>r th<sub>e</sub> lin<sub>ea</sub>r int<sub>e</sub>r<sub>po</sub>l<sub>a</sub>nt w<sub>e</sub> h<sub>a</sub>v<sub>e</sub> $\alpha _ { t } = t$ <sub>a</sub>nd $\sigma _ { t } = 1 - t$ with d<sub>e</sub>riv<sub>a</sub>tiv<sub>es</sub> $\dot { \alpha } _ { t } = 1$ <sub>a</sub>nd $\dot { \sigma } _ { t } = - 1$ <sub>.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>tin<sub>g</sub> thi<sub>s</sub> into E<sub>q</sub>uation (1) <sub>y</sub>ields

$$
\nu _ { t } ^ { 2 } = 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \sigma _ { t } ^ { 2 } - 2 \sigma _ { t } \dot { \sigma } _ { t } = \frac { 2 ( 1 - t ) ^ { 2 } } { t } + 2 ( 1 - t ) = 2 ( 1 - t ) \left[ \frac { 1 - t } { t } + 1 \right] = \frac { 2 ( 1 - t ) } { t } ,\tag{70}
$$

<sub>so</sub> th<sub>e</sub> dif<sub>us</sub>i<sub>on</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t bl<sub>ows</sub> <sub>up</sub> lik<sub>e</sub> $t ^ { - 1 / 2 }$ as $t \downarrow 0$ <sub>.</sub> Th<sub>e</sub> m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> tim<sub>e</sub> th<sub>e</sub>n f<sub>o</sub>ll<sub>o</sub>w<sub>s</sub> b<sub>y</sub> dir<sub>ec</sub>t <sub>co</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n of the <sub>q</sub>uadratic variation<sub>,</sub>

$$
\langle \boldsymbol { M } \rangle _ { t } = \int _ { t _ { 0 } } ^ { t } \mathrm { d } s ~ \boldsymbol { v } _ { s } ^ { 2 } = \int _ { t _ { 0 } } ^ { t } \mathrm { d } s ~ \left[ \frac { 2 } { s } - 2 \right] = 2 \log \frac { t } { t _ { 0 } } - 2 ( t - t _ { 0 } ) .\tag{71}
$$

$N . B .$ <sub>,</sub> th<sub>e</sub> lift<sub>e</sub>d tim<sub>e</sub> div<sub>e</sub>r<sub>ges as</sub> $t _ { 0 } \downarrow 0$ thr<sub>oug</sub>h th<sub>e</sub> l<sub>oga</sub>rithmi<sub>c</sub> t<sub>e</sub>rm<sub>,</sub> whi<sub>c</sub>h i<sub>s</sub> wh<sub>y</sub> th<sub>e</sub> lift<sub>e</sub>d <sub>o</sub>ri<sub>g</sub>in m<sub>us</sub>t b<sub>e</sub> k<sub>ep</sub>t at some $t _ { 0 } > 0$

F<sub>o</sub>r th<sub>e</sub> <sub>e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>ll<sub>y</sub> lift<sub>e</sub>d m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> $\begin{array} { r } { M _ { t } ^ { \alpha } : = \int _ { t _ { 0 } } ^ { t } \mathrm { d } W _ { r } \ \alpha _ { r } \nu _ { r } } \end{array}$ <sub>,</sub> whi<sub>c</sub>h <sub>appea</sub>r<sub>s</sub> in th<sub>e</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> t<sub>e</sub>rm <sub>o</sub>f th<sub>e</sub> <sub>e</sub>x<sub>po</sub>-

n<sub>e</sub>nti<sub>a</sub>ll<sub>y</sub> int<sub>eg</sub>r<sub>a</sub>t<sub>e</sub>d fl<sub>o</sub>w m<sub>ap,</sub> th<sub>e</sub> lift b<sub>y</sub> $\alpha _ { r } = r$ tames the sin<sub>g</sub>ularit<sub>y</sub> and the lifted time is a sim<sub>p</sub>le <sub>p</sub>ol<sub>y</sub>nomial<sub>,</sub>

$$
\langle { \pmb M } ^ { \alpha } \rangle _ { t } = \int _ { t _ { 0 } } ^ { t } \mathrm { d } r ~ \alpha _ { r } ^ { 2 } \nu _ { r } ^ { 2 }\tag{72}
$$

$$
= \int _ { t _ { 0 } } ^ { t } \mathrm { d } r \ r ^ { 2 } \frac { 2 ( 1 - r ) } { r }\tag{73}
$$

$$
= \int _ { t _ { 0 } } ^ { t } \mathrm { d } r ~ 2 r ( 1 - r )\tag{74}
$$

$$
= \left[ r ^ { 2 } - { \frac { 2 } { 3 } } r ^ { 3 } \right] _ { t _ { 0 } } ^ { t }\tag{75}
$$

$$
= t ^ { 2 } - \frac { 2 } { 3 } t ^ { 3 } - t _ { 0 } ^ { 2 } + \frac { 2 } { 3 } t _ { 0 } ^ { 3 } ,\tag{76}
$$

whi<sub>c</sub>h<sub>, u</sub>nlik<sub>e</sub> th<sub>e o</sub>ri<sub>g</sub>in<sub>a</sub>l tim<sub>e,</sub> r<sub>e</sub>m<sub>a</sub>in<sub>s</sub> b<sub>ou</sub>nd<sub>e</sub>d <sub>as</sub> $t _ { 0 } \downarrow 0 , i . e . , \langle M ^ { \alpha } \rangle _ { t }  t ^ { 2 } - { \textstyle \frac { 2 } { 3 } } t ^ { 3 }$ <sub>.</sub> Eith<sub>e</sub>r tim<sub>e</sub> d<sub>e</sub>t<sub>e</sub>rmin<sub>es</sub> th<sub>e</sub> law of the corres<sub>p</sub>ondin<sub>g</sub> martin<sub>g</sub>ale throu<sub>g</sub>h the Dambis–Dubins–Schwarz theorem (Dubins and Schwarz 1965).

## B.4.2 Variance preserving

T<sup>h</sup>e variance preserving (VP) sc<sup>h</sup>e<sup>d</sup>u<sup>l</sup>e matc<sup>h</sup>es t<sup>h</sup>e margina<sup>l</sup>s o<sup>f</sup> t<sup>h</sup>e VP SDE wit<sup>h</sup> t<sup>h</sup>e <sup>l</sup>inear �-sc<sup>h</sup>e<sup>d</sup>u<sup>l</sup>e o<sup>f</sup> Song et al. (2021) ex<sub>p</sub>ressed in the flow matchin<sub>g</sub> convention $( t = 0$ noise, $t = 1$ data). Let $\beta ( s ) : = \beta _ { \mathrm { m i n } } + ( \beta _ { \mathrm { m a x } } - \beta _ { \mathrm { m i n } } ) s$ denote the linear �-schedule in VP tim $s = 1 - t$ <sub>,</sub> th<sub>en</sub> th<sub>e s</sub>i<sub>gna</sub>l <sub>an</sub>d <sub>no</sub>i<sub>se coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s are</sub>

$$
\alpha _ { t } = \exp \left( - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 - t } \mathrm { d } u \ \beta ( u ) \right) = \exp \left( - \frac { \beta _ { \operatorname* { m i n } } } { 2 } ( 1 - t ) - \frac { \beta _ { \operatorname* { m a x } } - \beta _ { \operatorname* { m i n } } } { 4 } ( 1 - t ) ^ { 2 } \right) ,\tag{77}
$$

$$
\sigma _ { t } = \sqrt { 1 - \alpha _ { t } ^ { 2 } } ,\tag{78}
$$

w<sup>h</sup>ic<sup>h</sup> is variance preserving, i.e $\therefore , \alpha _ { t } ^ { 2 } + \sigma _ { t } ^ { 2 } = 1$ . B<sub>y</sub> the chain rule throu<sub>g</sub>h the u<sub>pp</sub>er limit 1−� we have $\begin{array} { r } { \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } = \frac { 1 } { 2 } \beta ( 1 - t ) } \end{array}$ For variance-<sub>p</sub>reservin<sub>g</sub> <sub>p</sub>aths the difusion coeficient sim<sub>p</sub>lifies considerabl<sub>y;</sub> usin<sub>g</sub> $\begin{array} { r } { \sigma _ { t } \dot { \sigma } _ { t } = \frac { 1 } { 2 } \frac { \mathrm { d } } { \mathrm { d } t } \sigma _ { t } ^ { 2 } = - \alpha _ { t } \dot { \alpha } _ { t } } \end{array}$ we fi<sub>n</sub>d

$$
\nu _ { t } ^ { 2 } = 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \sigma _ { t } ^ { 2 } - 2 \sigma _ { t } \dot { \sigma } _ { t } = 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \left( 1 - \alpha _ { t } ^ { 2 } \right) + 2 \alpha _ { t } \dot { \alpha } _ { t } = 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } = \beta ( 1 - t ) ,\tag{79}
$$

w<sup>h</sup>ic<sup>h</sup> is <sup>b</sup>oun<sup>d</sup>e<sup>d</sup>, i.e., $\nu _ { t } ~ \in ~ \left\lceil \sqrt { \beta _ { \operatorname* { m i n } } } , \sqrt { \beta _ { \operatorname* { m a x } } } \right\rceil$ <sub>,</sub> i<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t t<sub>o</sub> th<sub>e</sub> $t ^ { - 1 / 2 }$ bl<sub>ow-up o</sub>f th<sub>e</sub> li<sub>near sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e.</sub> Th<sub>e</sub> martin<sub>g</sub>ale time is then an elementar<sub>y</sub> inte<sub>g</sub>ral<sub>,</sub>

$$
\langle M \rangle _ { t } = \int _ { t _ { 0 } } ^ { t } \mathrm { d } s ~ \beta ( 1 - s ) = \beta _ { \operatorname* { m i n } } ( t - t _ { 0 } ) + ( \beta _ { \operatorname* { m a x } } - \beta _ { \operatorname* { m i n } } ) \left[ \left( t - \frac { t ^ { 2 } } { 2 } \right) - \left( t _ { 0 } - \frac { t _ { 0 } ^ { 2 } } { 2 } \right) \right] ,\tag{80}
$$

whi<sub>c</sub>h<sub>, u</sub>nlik<sub>e</sub> th<sub>e</sub> lin<sub>ea</sub>r tim<sub>e, s</sub>t<sub>ays</sub> b<sub>ou</sub>nd<sub>e</sub>d <sub>as</sub> $t _ { 0 } \downarrow 0$

F<sub>o</sub>r th<sub>e</sub> lift<sub>e</sub>d m<sub>a</sub>rtin<sub>ga</sub>l<sub>e</sub> $\begin{array} { r } { M _ { t } ^ { \alpha } : = \int _ { t _ { 0 } } ^ { t } \mathrm { d } W _ { r } \ \alpha _ { r } \nu _ { r } } \end{array}$ th<sub>e</sub>r<sub>e</sub> i<sub>s</sub> <sub>a</sub>n <sub>e</sub>l<sub>ega</sub>nt <sub>c</sub>l<sub>ose</sub>d f<sub>o</sub>rm<sub>.</sub> Ob<sub>se</sub>rv<sub>e</sub> th<sub>a</sub>t f<sub>o</sub>r <sub>a</sub>n<sub>y</sub> v<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub>- <sub>p</sub>reservin<sub>g</sub> afine <sub>p</sub>ath<sub>,</sub>

$$
\frac { \mathrm { d } } { \mathrm { d } r } \alpha _ { r } ^ { 2 } = 2 \alpha _ { r } \dot { \alpha } _ { r } = 2 \alpha _ { r } ^ { 2 } \frac { \dot { \alpha } _ { r } } { \alpha _ { r } } = \alpha _ { r } ^ { 2 } \nu _ { r } ^ { 2 } ,\tag{81}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> <sub>s</sub>i<sub>nce</sub> $\textstyle \nu _ { r } ^ { 2 } = 2 { \frac { { \dot { \alpha } } _ { r } } { \alpha _ { r } } }$ for variance-<sub>p</sub>reservin<sub>g</sub> <sub>p</sub>aths b<sub>y</sub> the same sim<sub>p</sub>lification as above. Hence the time of the lifted martingale is just the increment of the squared signal coeficient,

$$
\langle { \pmb M } ^ { \alpha } \rangle _ { t } = \int _ { t _ { 0 } } ^ { t } \mathrm { d } r ~ \alpha _ { r } ^ { 2 } \nu _ { r } ^ { 2 } = \int _ { t _ { 0 } } ^ { t } \mathrm { d } r ~ \frac { \mathrm { d } } { \mathrm { d } r } \alpha _ { r } ^ { 2 } = \alpha _ { t } ^ { 2 } - \alpha _ { t _ { 0 } } ^ { 2 } .\tag{82}
$$

N.B., t<sup>h</sup>is i<sup>d</sup>entity is particu<sup>l</sup>ar to variance-preserving pat<sup>h</sup>s: <sup>f</sup>or t<sup>h</sup>e <sup>l</sup>inear sc<sup>h</sup>e<sup>d</sup>u<sup>l</sup>e $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } r } \alpha _ { r } ^ { 2 } = 2 r } \end{array}$ <sub>w</sub>hil<sub>e</sub> $\alpha _ { r } ^ { 2 } \nu _ { r } ^ { 2 } =$ $2 r ( 1 - r )$ <sub>,</sub> h<sub>e</sub>n<sub>ce</sub> th<sub>e</sub> <sub>po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l tim<sub>e</sub> d<sub>e</sub>riv<sub>e</sub>d <sub>a</sub>b<sub>o</sub>v<sub>e.</sub>

## C On training objectives

## C.1 Discretized objectives

Recall the discrete Eulerian and La<sub>g</sub>ran<sub>g</sub>ian conditions from E<sub>q</sub>uations (14) and (15). We now rewrite these conditions followin<sub>g</sub> Sabour, Fidler, et al. (2025) but for the SDE case below:

$$
\mathbb { E } _ { s , t , x } \left[ \left. \Psi _ { s , t } ^ { \theta } ( x ; \mathcal { T } _ { s , t } ) - \mathrm { s g } \left[ \Psi _ { s + \delta , t } ^ { \theta } \left( \mathrm { S D E S o l v e r } _ { s , s + \delta } ( x ; \mathcal { T } _ { s , s + \delta } ) ; \mathcal { T } _ { s + \delta , t } \right) \right] \right. ^ { 2 } \right] ,\tag{83}
$$

$$
\begin{array} { r } { \mathbb { E } _ { s , t , x } \left[ \left. \Psi _ { s , t } ^ { \theta } ( x ; \mathcal { T } _ { s , t } ) - \mathrm { s g } \left[ \mathrm { S D E S o l v e r } _ { t - \delta , t } \left( \Psi _ { s , t - \delta } ^ { \theta } ( x ; \mathcal { T } _ { s , t - \delta } ) ; \mathcal { T } _ { t - \delta , t } \right) \right] \right. ^ { 2 } \right] , } \end{array}\tag{84}
$$

for the discrete Eulerian and discrete La<sub>g</sub>ran<sub>g</sub>ian losses<sub>,</sub> res<sub>p</sub>ectivel<sub>y,</sub> where $\delta \in ( 0 , t - s )$ i<sub>s so</sub>m<sub>e s</sub>m<sub>a</sub>ll <sub>s</sub>t<sub>ep</sub> <sub>s</sub>iz<sub>e, s</sub> d<sub>e</sub>n<sub>o</sub>t<sub>es</sub> th<sub>e s</sub>t<sub>o</sub> r<sub>a</sub>d <sub>o e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd $\mathrm { S D E S o l v e r } _ { s , t } ( \cdot , \varbar { L } _ { s , t } )$ d<sub>e</sub>n<sub>o</sub>t<sub>es a</sub>n <sub>e</sub>x li<sub>c</sub>it <sub>s</sub>in l<sub>e</sub>-<sub>s</sub>t<sub>e</sub> SDE <sub>so</sub>lv<sub>e</sub>r whi<sub>c</sub>h uses the coeficients I ste<sub>pp</sub>in<sub>g</sub> from � to $t , e . g .$ ., an Euler–Maru<sub>y</sub>ama or ShARK ste<sub>p</sub> (Foster et al. 2024). Our <sub>resu</sub>lt f<sub>rom</sub> Th<sub>eorem</sub> $\mathrm { B } . 3$ <sub>e</sub>n<sub>a</sub>bl<sub>es</sub> <sub>us</sub> t<sub>o</sub> <sub>co</sub>mbin<sub>e</sub> <sub>ou</sub>r <sub>sp</sub>lit<sub>s</sub> vi<sub>a</sub> Ch<sub>e</sub>n’<sub>s</sub> r<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> with <sub>so</sub>m<sub>e</sub> <sub>a</sub>rbitr<sub>a</sub>r<sub>y</sub> $\delta$ <sub>e</sub>n<sub>a</sub>blin<sub>g</sub> thi<sub>s</sub> l<sub>oss</sub> f<sub>or s</sub>t<sub>rong s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>ow maps.</sub> I<sub>n our</sub> t<sub>oy exper</sub>i<sub>men</sub>t<sub>s we</sub> f<sub>oun</sub>d thi<sub>s</sub> l<sub>oss</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> t<sub>o ou</sub>t<sub>per</sub>f<sub>orm</sub> th<sub>e s</sub>t<sub>a</sub>nd<sub>a</sub>rd <sub>se</sub>mi<sub>g</sub>r<sub>oup</sub> l<sub>oss,</sub> b<sub>u</sub>t in l<sub>a</sub>r<sub>ge</sub>r <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> f<sub>ou</sub>nd it t<sub>o s</sub>li<sub>g</sub>htl<sub>y u</sub>nd<sub>e</sub>r<sub>pe</sub>rf<sub>o</sub>rm it<sub>.</sub>

## C.2 Doss–Sussmann objectives

We can use the Doss–Sussmann transform to rewrite our difusion SDE as a random ODE. Consider our <sub>g</sub>eneral Itô SDE<sub>,</sub> namel<sub>y,</sub>

$$
\mathrm { d } X _ { t } = b ( t , X _ { t } ) \mathrm { ~ d } t + \nu _ { t } \mathrm { ~ d } W _ { t } .\tag{85}
$$

R<sub>eca</sub>ll th<sub>a</sub>t $M _ { t }$ is t<sup>h</sup>e martinga<sup>l</sup>e so<sup>l</sup>ution to t<sup>h</sup>e Itô integra<sup>l</sup> as <sup>d</sup>e<sup>fi</sup>ne<sup>d</sup> ear<sup>l</sup>ier (mutatis mutandis <sup>f</sup>or t<sup>h</sup>e exponentiall<sub>y</sub> lifted SDE).

Doss–Sussmann trick. Now on each interval [�,�] we define the locally anchored transformed state as $\begin{array} { r l r } {  { Z _ { r } ^ { s } : = } } & { { } } & { } \end{array}$ $\pmb { X } _ { r } - \left( \pmb { M } _ { r } - \pmb { M } _ { s } \right)$ <sub>w</sub>ith $Z _ { s } ^ { s } = X _ { s }$ . Thus we can rewrite our SDE in terms of $ { \boldsymbol { Z } } _ { r } ^ { s }$ as

$$
\mathrm { d } Z _ { r } ^ { s } = \pmb { b } \big ( r , Z _ { r } ^ { s } + \pmb { M } _ { r } - \pmb { M } _ { s } \big ) \ \mathrm { d } r ,\tag{86}
$$

which is a smooth random ODE (rODE) (Baranek et al. 2025, Section 3) as the rou<sub>g</sub>hness is com<sub>p</sub>letel<sub>y</sub> absorbed into the ex<sub>p</sub>licit additive shift $M _ { r } - M _ { s }$

Let $\Xi _ { s , t } ( z ; M )$ d<sub>e</sub>n<sub>o</sub>t<sub>e</sub> th<sub>e</sub> fl<sub>o</sub>w <sub>o</sub>f thi<sub>s</sub> r<sub>a</sub>nd<sub>o</sub>m ODE<sub>,</sub> $i . e .$

$$
\Xi _ { s , t } \left( z ; M \right) = z + \int _ { s } ^ { t } \mathrm { d } \tau \pmb { b } \left( \tau , \Xi _ { s , \tau } \left( z ; M \right) + M _ { \tau } - M _ { s } \right) .\tag{87}
$$

W<sub>e pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>e</sub> th<sub>e s</sub>t<sub>u</sub>d<sub>e</sub>nt fl<sub>o</sub>w m<sub>ap</sub> with <sub>a</sub>n E<sub>u</sub>l<sub>e</sub>r h<sub>ea</sub>d vi<sub>a</sub>

$$
\Xi _ { s , t } ^ { \theta } \left( z ; \mathcal { T } _ { s , t } ^ { ( N ) } \right) = z + ( t - s ) \pmb { \nu } _ { s , t } ^ { \theta } \left( z ; \mathcal { T } _ { s , t } ^ { ( N ) } \right) ,\tag{88}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { I } _ { s , t } ^ { ( N ) }$ denotes the shifted Le<sub>g</sub>endre coeficients of the martin<sub>g</sub>ale (a time-chan<sub>g</sub>ed Brownian motion) over $[ \langle M \rangle _ { s } , \langle M \rangle _ { t } ]$ (q.v. Appen<sup>d</sup>ix B.4).

F<sub>or</sub> th<sub>e</sub> di<sub>agona</sub>l l<sub>oss, we</sub> h<sub>ave</sub> $\pmb { Z } _ { t } ^ { t } = \pmb { X } _ { t }$ <sub>a</sub>nd th<sub>e</sub> di<sub>ago</sub>n<sub>a</sub>l drift <sub>co</sub>nditi<sub>o</sub>n <sub>s</sub>im<sub>p</sub>lifi<sub>es</sub> t<sub>o:</sub>

$$
\pmb { \nu } _ { t , t } ^ { \theta } \left( z ; \mathcal { T } _ { t , t } ^ { \left( N \right) } \right) = \pmb { b } \left( t , z \right) = \pmb { b } \left( t , \pmb { X } _ { t } \right) .\tag{89}
$$

Distillation losses. We now explore the Eulerian and Lagrangian distillation losses. Recall that the main goal of thi<sub>s</sub> D<sub>oss</sub>–S<sub>uss</sub>m<sub>a</sub>nn tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm w<sub>as</sub> t<sub>o</sub> r<sub>e</sub>m<sub>o</sub>v<sub>e</sub> th<sub>e</sub> n<sub>ee</sub>d t<sub>o</sub> dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>t<sub>e</sub> th<sub>e</sub> n<sub>o</sub>n-dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>bl<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th b<sub>y</sub> transforming our SDE into a smooth random ODE and now learning a flow map for �. As such, we avoid any <sub>suc</sub>h W<sub>o</sub>n<sub>g</sub>–Z<sub>a</sub>k<sub>a</sub>i t<sub>ype a</sub>r<sub>gu</sub>m<sub>e</sub>nt<sub>s</sub> whi<sub>c</sub>h w<sub>ou</sub>ld n<sub>ega</sub>t<sub>e</sub> th<sub>e</sub> b<sub>e</sub>n<sub>e</sub>fit<sub>s o</sub>f <sub>us</sub>in<sub>g</sub> thi<sub>s</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>.</sub> F<sub>o</sub>r <sub>o</sub>f-di<sub>ago</sub>n<sub>a</sub>l trainin<sub>g,</sub> the La<sub>g</sub>ran<sub>g</sub>ian condition matches the terminal derivative:

$$
\partial _ { t } \Xi _ { s , t } ^ { \theta } \left( z ; \mathcal { I } _ { s , t } ^ { ( N ) } \right) - b \left( t , \Xi _ { s , t } ^ { \theta } \left( z ; \mathcal { I } _ { s , t } ^ { ( N ) } \right) + M _ { t } - M _ { s } \right) = 0 .\tag{90}
$$

Lik<sub>e</sub>wi<sub>se,</sub> <sub>o</sub>n<sub>e</sub> <sub>cou</sub>ld writ<sub>e</sub> th<sub>e</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> E<sub>u</sub>l<sub>e</sub>ri<sub>a</sub>n <sub>co</sub>nditi<sub>o</sub>n t<sub>o</sub> $\Xi$ as

$$
\begin{array} { r } { \partial _ { s } \Xi _ { s , t } ^ { \theta } \left( z ; T _ { s , t } ^ { \left( N \right) } \right) + \left( \nabla _ { z } \Xi _ { s , t } ^ { \theta } \left( z ; T _ { s , t } ^ { \left( N \right) } \right) \right) \boldsymbol { b } \left( s , z \right) = 0 , } \end{array}\tag{91}
$$

which is the condition the DS Eulerian losses enforce. As noted in Pan et al. (2026, Pro<sub>p</sub>osition 2) the Eulerian con<sup>d</sup>ition <sup>f</sup>ai<sup>l</sup>s to <sup>l</sup>earn t<sup>h</sup>e Itô or strong so<sup>l</sup>ution map o<sup>f</sup> t<sup>h</sup>e SDE, n.b. t<sup>h</sup>is Doss–Sussmann <sup>f</sup>ormu<sup>l</sup>ation can <sup>b</sup>e seen as a <sub>g</sub>eneralization of what was considered in Pan et al. (2026).

W<sub>e</sub> n<sub>o</sub>w m<sub>a</sub>k<sub>e</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>o</sub>win<sub>g</sub> <sub>assu</sub>m<sub>p</sub>ti<sub>o</sub>n<sub>s:</sub>

Assumption C.1. Assume that � and $\nabla _ { z } \boldsymbol { b }$ are jointly continuous and that � is globally Lipschitz in state uniformly on a compact time horizon.

Assumption C.2. Assume that� is deterministic and continuous.

Assumption C.3. Assume further that the learned head is continuously diferentiable and satisfies the identity boundary condition locally uniformly in �, for everyfixed ${ \widehat { \cal I } } ^ { ( N ) }$ and every diagonal time �.

Proposition C.1 (Coefficient independence under the Eulerian objective). Under the assumptions above, suppose that the Eulerian condition Equation (91) holds identically, i.e., in terms of the flow map head,

$$
b ( s , z ) - \pmb { \nu } _ { s , t } ^ { \theta } ( z ) + h \partial _ { s } \pmb { \nu } _ { s , t } ^ { \theta } ( z ) + h \left( \nabla _ { z } \pmb { \nu } _ { s , t } ^ { \theta } ( z ) \right) \pmb { b } ( s , z ) = \mathbf { 0 } .\tag{92}
$$

Then

$$
{ \pmb { \nu } } _ { s , t } ^ { \theta } ( z ) = \frac { { \pmb { \varphi } } _ { s , t } ( z ) - z } { t - s } ,\tag{93}
$$

where $\pmb { \varphi }$ is the deterministic flow of $\mathbf { \dot { y } } _ { r } = \pmb { b } ( r , \mathbf { y } _ { r } )$ . Consequently, the learned head is independent of every supplied coeficient.

Proof. Fix $\smash { \mathcal { T } ^ { ( N ) } }$ <sub>an</sub>d f<sub>o</sub>ll<sub>ow</sub> ${ \pmb y } _ { r } = \pmb { \varphi } _ { s , r } ( { \pmb z } )$ . The chain rule and E<sub>q</sub>uation (92) <sub>g</sub>ive

$$
\frac { \mathrm { d } } { \mathrm { d } r } \left[ \pmb { y } _ { r } + \left( t - r \right) \pmb { \nu } _ { r , t } ^ { \theta } ( \pmb { y } _ { r } ) \right] = \mathbf { 0 } ,\tag{94}
$$

so this ex ression is constant in �. The trajector remains in a com act set and tends to � ; b the locall uniform identit<sub>y</sub> boundar<sub>y</sub> condition Assum<sub>p</sub>tion $\mathrm { C } . 3 ,$ th<sub>e</sub> b<sub>rac</sub>k<sub>e</sub>t th<sub>ere</sub>f<sub>ore</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> $y _ { t }$ as $r \uparrow t .$ It<sub>s</sub> v<sub>a</sub>l<sub>ue a</sub>t $r = s$ i<sub>s</sub> co<sup>n</sup>seque<sup>ntl</sup>y $y _ { t }$ , which <sub>p</sub>roves E<sub>q</sub>uation (93) for ever<sub>y</sub> coeficient vector. □

Remark C.1. This is an analogous result to Pan et al. (2026) rewritten within our notation and nomenclature surroundin<sub>g</sub> the more <sub>g</sub>eneral Doss–Sussmann transform.

Proposition C.2 (Limited coefficient dependence under the Lagrangian objective). Under the same assumptions, suppose that the Lagrangian condition above holds identically, i.e., in terms ofthe flow map head,

$$
\pmb { \nu } _ { s , t } ^ { \theta } ( z ) + h \partial _ { t } \pmb { \nu } _ { s , t } ^ { \theta } ( z ) = \pmb { b } \Big ( t , z + h \pmb { \nu } _ { s , t } ^ { \theta } ( z ) + \pmb { I } ^ { ( 0 ) } \Big ) .\tag{95}
$$

Then for each fixed $( s , t , z )$ , the learned head depends on the supplied coeficients at most through the zeroth coeficient. In particular, it is independent ofevery higher coeficient.

Proof. Fix $s , z ,$ <sub>a</sub>nd ${ \cal { T } } ^ { ( N ) }$ <sub>, a</sub>nd <sub>se</sub>t $\xi _ { r } : = z + ( r - s ) \ : \pmb { \nu } _ { s , r } ^ { \theta } ( z )$ . Then E<sub>q</sub>uation (95) and the identit<sub>y</sub> boundar<sub>y</sub> <sub>co</sub>nditi<sub>o</sub>n A<sub>ssu</sub>m<sub>p</sub>ti<sub>o</sub>n C<sub>.3</sub> <sub>g</sub>iv<sub>e</sub>

$$
\dot { \xi } _ { r } = { \pmb b } \Big ( r , \xi _ { r } + { \pmb I } ^ { ( 0 ) } \Big ) , \qquad \operatorname* { l i m } _ { r \downarrow s } \xi _ { r } = z .\tag{96}
$$

Since ${ \pmb { \xi } } _ { r }  z$ <sub>an</sub>d th<sub>e var</sub>i<sub>ance o</sub>f th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>goes</sub> t<sub>o zero as</sub> $r \downarrow s ,$ <sub>,</sub> th<sub>e</sub> ri<sub>g</sub>ht-h<sub>a</sub>nd <sub>s</sub>id<sub>e e</sub>xt<sub>e</sub>nd<sub>s co</sub>ntin<sub>uous</sub>l<sub>y</sub> to $r = s$ <sub>an</sub>d <sub>we can</sub> th<sub>ere</sub>f<sub>ore wr</sub>it<sub>e</sub>

$$
\pmb { \xi } _ { t } = \pmb { z } + \int _ { s } ^ { t } \mathrm { d } \pmb { r } \ \pmb { b } \Big ( r , \pmb { \xi } _ { r } + \widehat { I } _ { s , r } ^ { ( 0 ) } \Big ) .\tag{97}
$$

For fixed � and � , this equation has a unique solution as � is globally Lipschitz in its state argument. The only coeficient a<sub>pp</sub>earin<sub>g</sub> in the e<sub>q</sub>uation is $\widehat { I } ^ { ( 0 ) }$ <sub>, so c</sub>h<sub>a</sub>n<sub>g</sub>in<sub>g a</sub>n<sub>y</sub> hi<sub>g</sub>h<sub>e</sub>r <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt l<sub>ea</sub>v<sub>es</sub> b<sub>o</sub>th th<sub>e equa</sub>ti<sub>o</sub>n <sub>a</sub>nd it<sub>s</sub> initial condition unchan<sub>g</sub>ed. B<sub>y</sub> uni<sub>q</sub>ueness<sub>,</sub> it also leaves ${ \pmb { \xi } } _ { t }$ <sub>unc</sub>h<sub>ange</sub>d<sub>, an</sub>d h<sub>ence</sub> l<sub>eaves</sub> th<sub>e</sub> l<sub>earne</sub>d h<sub>ea</sub>d ${ \pmb { \nu } } _ { s , t } ^ { \theta } ( z ) = ( { \pmb { \xi } } _ { t } - z ) / ( t - s )$ <sub>u</sub>n<sub>c</sub>h<sub>a</sub>n<sub>ge</sub>d<sub>.</sub> □

The ill-suitedness of the Doss–Sussmann losses Thus the DS Eulerian loss is minimized by a map that ignores the Brownian <sub>p</sub>ath entirel<sub>y</sub>, recoverin<sub>g</sub> the de<sub>g</sub>enerac<sub>y</sub> of Pan et al. (2026, Pro<sub>p</sub>osition 2), while the DS La<sub>g</sub>ran<sub>g</sub>ian l<sub>oss</sub> <sub>ca</sub>n <sub>e</sub>x<sub>p</sub>l<sub>o</sub>it <sub>a</sub>t m<sub>os</sub>t th<sub>e</sub> in<sub>c</sub>r<sub>e</sub>m<sub>e</sub>nt $M _ { t } - M _ { s }$ <sub>.</sub> Thi<sub>s</sub> i<sub>s co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>nt with Fi<sub>gu</sub>r<sub>e 3</sub>b<sub>,</sub> wh<sub>e</sub>r<sub>e</sub> th<sub>e</sub> DS m<sub>o</sub>d<sub>e</sub>l<sub>s a</sub>r<sub>e</sub> fl<sub>a</sub>t in �<sub>.</sub> It i<sub>s</sub> f<sub>o</sub>r thi<sub>s</sub> r<sub>easo</sub>n w<sub>e</sub> t<sub>u</sub>rn<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> W<sub>o</sub>n<sub>g</sub>–Z<sub>a</sub>k<sub>a</sub>i <sub>a</sub>r<sub>gu</sub>m<sub>e</sub>nt <sub>as</sub> w<sub>e</sub> f<sub>u</sub>nd<sub>a</sub>m<sub>e</sub>nt<sub>a</sub>ll<sub>y</sub> m<sub>us</sub>t w<sub>o</sub>rk with th<sub>e</sub> Brownian <sub>p</sub>ath and can not avoid it even with the ver<sub>y</sub> ele<sub>g</sub>ant Doss–Sussmann transform.

## C.3 Wong–Zakai objectives

In this section we study the Wong–Zakai style objectives. Fix an interval $[ s , t ] \subseteq [ t _ { 0 } , 1 ]$ <sub>an</sub>d l<sub>e</sub>t ${ \pmb b } _ { r } ( { \pmb x } ) : =$ $\begin{array} { r } { 2 { \pmb u } _ { r } ( { \pmb x } ) - \frac { \dot { \alpha } _ { r } } { \alpha _ { r } } { \pmb x } } \end{array}$ denote the drift of E<sub>q</sub>uation (1). Now the Won<sub>g</sub>–Zakai random ODE re<sub>p</sub>laces the Brownian <sub>p</sub>ath with its shifted Le<sub>g</sub>endre a<sub>pp</sub>roximation on $[ s , t ] , i . e .$ ,

$$
\dot { X } _ { r } ^ { ( N ) } = \pmb { \nu } _ { r } ^ { ( N ) } ( { X } _ { r } ^ { ( N ) } ) , \qquad \pmb { \nu } _ { r } ^ { ( N ) } ( \pmb { x } ) : = \pmb { b } _ { r } ( \pmb { x } ) + \nu _ { r } \dot { W } _ { r } ^ { ( N , s , t ) } , \qquad { X } _ { s } ^ { ( N ) } = \pmb { x } ,\tag{98}
$$

<sub>a</sub>nd w<sub>e</sub> d<sub>e</sub>n<sub>o</sub>t<sub>e</sub> it<sub>s</sub> fl<sub>o</sub>w b<sub>y</sub> $\Psi _ { s , t } ^ { ( N ) } ( { \pmb x } ; { \pmb T } _ { s , t } )$ . As $W ^ { ( N , s , t ) }$ is a <sub>p</sub>ol<sub>y</sub>nomial<sub>,</sub> E<sub>q</sub>uation $( 9 8 )$ is a random ODE whose <sup>d</sup>erivative is avai<sup>l</sup>a<sup>bl</sup>e in c<sup>l</sup>ose<sup>d f</sup>orm <sup>b</sup>y Lemma B.1. N.B., since t<sup>h</sup>e noise is a<sup>dd</sup>itive, t<sup>h</sup>e Itô an<sup>d</sup> Stratonovic<sup>h</sup> inter<sub>p</sub>retations of E<sub>q</sub>uation (1) coincide, sim<sub>p</sub>lif<sub>y</sub>in<sub>g</sub> the anal<sub>y</sub>sis. We now <sub>p</sub>resent the followin<sub>g p</sub>ro<sub>p</sub>osition.

Proposition C.3 (Wong–Zakai limit for the shifted Legendre polynomial). Assume ${ \pmb b } _ { r }$ is globally Lipschitz in state with constant � uniformly $f o r r \in [ s , t ]$ , and that $\nu _ { r }$ is absolutely continuous on $[ s , t ]$ . Let � solve Equation (1) and $\pmb { X } ^ { ( N ) }$ solve Equation (98), with initial condition $\pmb { X _ { s } } = \pmb { X _ { s } ^ { ( N ) } } = \pmb { x }$ . Then, almost surely,

$$
\operatorname* { s u p } _ { r \in [ s , t ] } \left\| X _ { r } ^ { ( N ) } - X _ { r } \right\| \leq e ^ { L ( t - s ) } \left( \operatorname* { s u p } _ { r \in [ s , t ] } \vert \nu _ { r } \vert + \int _ { s } ^ { t } \mathrm { d } r \vert \dot { \nu } _ { r } \vert \right) \operatorname* { s u p } _ { r \in [ s , t ] } \left\| W _ { s , r } ^ { ( N , s , t ) } - W _ { s , r } \right\| ,\tag{99}
$$

and the right-hand side tends to zero as $N \to \infty$ . Thus $\Psi _ { s , t } ^ { ( N ) } ( \pmb { x } ; \mathcal { T } _ { s , t } )  \Psi _ { s , t } ( \pmb { x } ; \pmb { W } )$ almost surely.

Proof. Let $\pmb { \Delta } _ { r } : = \pmb { W } _ { s , r } ^ { ( N , s , t ) } - \pmb { W } _ { s , l }$ <sub>�</sub> denote the <sub>g</sub>a<sub>p</sub> between the <sub>p</sub>ol<sub>y</sub>nomial and true Brownian <sub>p</sub>ath. As $\nu _ { r }$ i<sub>s</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> w<sub>e ca</sub>n find

$$
\int _ { s } ^ { r } \mathrm { d } \tau \ \nu _ { \tau } \dot { W } _ { \tau } ^ { ( N , s , t ) } - \int _ { s } ^ { r } \mathrm { d } W _ { \tau } \ \nu _ { \tau } = \nu _ { r } \Delta _ { r } - \int _ { s } ^ { r } \mathrm { d } \tau \ \dot { \nu } _ { \tau } \Delta _ { \tau } ,\tag{100}
$$

wh<sub>ose</sub> n<sub>o</sub>rm i<sub>s</sub> b<sub>ou</sub>nd<sub>e</sub>d b<sub>y</sub> $C _ { \nu } \operatorname { s u p } _ { r } \| \pmb { \Delta } _ { r } \|$ <sub>w</sub>ith $C _ { \nu }$ the bracket in E<sub>q</sub>uation (99). Next, subtractin<sub>g</sub> the inte<sub>g</sub>ral e<sub>q</sub>uations for $X ^ { ( N ) }$ and � and using the Lipschitz bound on � yields

$$
\left\| { \boldsymbol X } _ { r } ^ { ( N ) } - { \boldsymbol X } _ { r } \right\| \leq L \int _ { s } ^ { r } \mathrm { d } \tau \left\| { \boldsymbol X } _ { \tau } ^ { ( N ) } - { \boldsymbol X } _ { \tau } \right\| + C _ { \nu } \operatorname* { s u p } _ { \tau \in [ s , t ] } \| \Delta _ { \tau } \| ,\tag{101}
$$

and Grönwall’s ine<sub>q</sub>ualit<sub>y y</sub>ields E<sub>q</sub>uation (99). Finall<sub>y</sub>, the shifted Le<sub>g</sub>endre a<sub>pp</sub>roximation conver<sub>g</sub>es to � in the �-Hölder rou<sub>g</sub>h-<sub>p</sub>ath metric almost surel<sub>y</sub> (McCallum et al. 2026, Pro<sub>p</sub>osition A.3), which in <sub>p</sub>articular im<sub>p</sub>lies su<sub>p�</sub> $\| \Delta _ { r } \|  0$ □

Fixed-path derivatives. We now discuss some practical concerns on implementation. As the shifted Legendre basis in E<sub>q</sub>uation (3) de<sub>p</sub>ends on the interval [�,�], movin<sub>g</sub> either end<sub>p</sub>oint re-ex<sub>p</sub>resses the same curve in <sub>a new</sub> b<sub>as</sub>i<sub>s, so</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s c</sub>h<sub>ange even</sub> th<sub>oug</sub>h th<sub>e pa</sub>th d<sub>oes no</sub>t<sub>.</sub> C<sub>oncre</sub>t<sub>e</sub>l<sub>y, we</sub> h<sub>o</sub>ld th<sub>e po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>p</sub>at<sup>h</sup> $W ^ { ( N , s , t ) }$ fi<sub>xe</sub>d <sub>an</sub>d<sub>,</sub> f<sub>or</sub> $[ u , \nu ] \subseteq [ s , t ]$ , write $\mathcal { I } _ { u , \nu } = \{ I _ { u , \nu } ^ { ( n ) } \} _ { n = 0 } ^ { N - 1 }$ f<sub>or</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s o</sub>f it<sub>s res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> t<sub>o</sub> $[ u , \nu ]$ as $\dot { W } ^ { ( N , s , t ) }$ is a <sub>p</sub>ol<sub>y</sub>nomial of de<sub>g</sub>ree $N - 1$ <sub>,</sub> this restriction is re<sub>p</sub>resented exactl<sub>y</sub> b<sub>y</sub> � coeficients<sub>,</sub> which are <sub>smoo</sub>th i<sub>n</sub> $( u , \nu )$ <sub>.</sub> Th<sub>e c</sub>h<sub>a</sub>in r<sub>u</sub>l<sub>e</sub> th<sub>e</sub>n <sub>g</sub>iv<sub>es</sub>

$$
\mathfrak { D } _ { \tau } \Psi _ { s , t } ^ { \theta } = \partial _ { \tau } \Psi _ { s , t } ^ { \theta } \big | _ { \tau } + \sum _ { n = 0 } ^ { N - 1 } D _ { I ^ { ( n ) } } \Psi _ { s , t } ^ { \theta } \partial _ { \tau } I _ { u , \nu } ^ { ( n ) } \big | _ { ( u , \nu ) = ( s , t ) } , \qquad \tau \in \{ s , t \} ,\tag{102}
$$

<sub>w</sub>h<sub>ere</sub> $\partial _ { s }$ <sub>a</sub>nd $\partial _ { t }$ <sub>ac</sub>t <sub>o</sub>n <sub>� a</sub>nd <sub>�</sub> r<sub>espec</sub>tiv<sub>e</sub>l<sub>y,</sub> t<sub>a</sub>k<sub>e</sub>n <sub>o</sub>n<sub>e</sub>-<sub>s</sub>id<sub>e</sub>dl<sub>y</sub> fr<sub>o</sub>m in<sub>s</sub>id<sub>e</sub> $[ s , t ]$ , a<sup>nd w</sup>e supp<sup>r</sup>ess <sup>th</sup>e a<sup>r</sup>gu<sup>m</sup>e<sup>nt</sup>s <sub>o</sub>f $\Psi _ { s , t } ^ { \theta } ( \pmb { x } ; \pmb { \mathscr { I } } _ { s , t } )$ . These derivatives follow in closed form from Lemma B.1, but in <sub>p</sub>ractice we obtain E<sub>q</sub>uation (102) <sub>w</sub>ith <sub>a s</sub>i<sub>n</sub> l<sub>e</sub> f<sub>orwar</sub>d<sub>-mo</sub>d<sub>e</sub> JVP<sub>.</sub>

Wong–Zakai residuals. With this convention the Wong–Zakai Eulerian and Lagrangian residuals are written as

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { E } } ^ { N , \theta } : = \mathfrak { D } _ { s } \Psi _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { \mathit { x } } } } ; \mathcal { T } _ { s , t } ) + \left( \nabla _ { \boldsymbol { \mathbf { \mathit { x } } } } \Psi _ { s , t } ^ { \theta } ( { \boldsymbol { \mathbf { \mathit { x } } } } ; \mathcal { T } _ { s , t } ) \right) \pmb { \nu } _ { s } ^ { ( N ) } ( \boldsymbol { \mathbf { \mathit { x } } } ) , } \end{array}\tag{103}
$$

$$
\mathcal { R } _ { \mathrm { L } } ^ { N , \theta } : = \mathfrak { D } _ { t } \Psi _ { s , t } ^ { \theta } ( { \pmb x } ; \mathcal { T } _ { s , t } ) - { \pmb \nu } _ { t } ^ { ( N ) } \big ( \Psi _ { s , t } ^ { \theta } ( { \pmb x } ; \mathcal { T } _ { s , t } ) \big ) .\tag{104}
$$

E<sub>q</sub>uivalentl<sub>y,</sub> $\mathcal { R } _ { \mathrm { E } } ^ { N , \theta } = D _ { s } ^ { ( N ) } \Psi _ { s , t } ^ { \theta }$ <sub>, w</sub>h<sub>ere</sub> $D _ { s } ^ { ( N ) } : = \mathfrak { D } _ { s } + \pmb { \nu } _ { s } ^ { ( N ) } ( \pmb { x } ) ^ { \top } \nabla _ { \pmb { x } }$ is the material derivative ofA<sub>pp</sub>endix A.2 with $\partial _ { s }$ r<sub>ep</sub>l<sub>ace</sub>d b<sub>y</sub> th<sub>e</sub> fix<sub>e</sub>d-<sub>pa</sub>th d<sub>e</sub>riv<sub>a</sub>tiv<sub>e a</sub>nd $\nu _ { s }$ b<sub>y</sub> th<sub>e</sub> W<sub>o</sub>n<sub>g</sub>–Z<sub>a</sub>k<sub>a</sub>i fi<sub>e</sub>ld<sub>;</sub> thi<sub>s</sub> i<sub>s</sub> th<sub>e ope</sub>r<sub>a</sub>t<sub>o</sub>r $D _ { s }$ used in A<sub>pp</sub>endix G. In contrast to the Doss–Sussmann settin<sub>g</sub> (Pro<sub>p</sub>ositions $\mathrm { { C . 1 } }$ and C.2), the vector field $\pmb { \nu } ^ { ( N ) }$ <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> $\bar { \nu _ { r } } \dot { W _ { r } } ^ { ( N , s , t ) }$ which depends on all � coeficients by Lemma B.1; thus both residuals carry the full path information to the network. The concrete iMF instantiation of E<sub>q</sub>uation (103) in the ex<sub>p</sub>onential-inte<sub>g</sub>rator lifted state is <sub>g</sub>iven in A<sub>pp</sub>endix G.

## D On distillation

Prior work b<sub>y</sub> McCallum et al. (2026) and Pan et al. (2026) trained stron<sub>g</sub> stochastic flow ma<sub>p</sub>s onl<sub>y</sub> from scratch or use<sup>d d</sup>isti<sup>ll</sup>ation met<sup>h</sup>o<sup>d</sup>s (Pan et a<sup>l</sup>. 2026). Rat<sup>h</sup>er, in t<sup>h</sup>is wor<sup>k</sup>, we fine-tune pre-existing <sup>l</sup>arge <sup>fl</sup>ow matc<sup>h</sup>ing (e<sub>q</sub>uivalentl<sub>y</sub> difusion) models and deterministic flow ma<sub>p</sub>s into our stron<sub>g</sub> stochastic flow ma<sub>p</sub>s. Related work on weak stochastic flow ma<sub>p</sub>s b<sub>y</sub> Holderrieth, Chen, et al. (2026), Passaro et al. (2026), and Pota<sub>p</sub>tchik et al. (2026) also fine-tune their models, but as the weak formulation is sim<sub>p</sub>l<sub>y</sub> a re<sub>p</sub>arameterized conditional ODE (Passaro et al. 2026, Pro<sub>p</sub>ositions 1 and 2) the im<sub>p</sub>lementation of such fine-tunin<sub>g</sub> is a strai<sub>g</sub>htforward a<sub>pp</sub>lication of the Decoupled MeanFlow (DMF) tec<sup>h</sup>nique propose<sup>d</sup> in Lee et a<sup>l</sup>. (2026). However, in our case it is non-trivia<sup>l</sup> to ada<sub>p</sub>t the DMF idea and so we <sub>p</sub>erform several modifications which we hi<sub>g</sub>hli<sub>g</sub>ht in Fi<sub>g</sub>ure 12.

(a) Flow matching parameterizati<sub>o</sub>n<sub>.</sub>  
![](images/0ed93334a827ca7ee5b2931c0276aa157ac6be06bc20821acc99de00bde3b947.jpg)  
Figure 12 From FM to iSSFM parameterization. (a) The pre-trained one-time flow matching model. (b) Fine-tuning into a two-time DMF model. (c) iSSFM adds the Brownian coeficients to the network.

A brief overview of DMF. Suppose we have a pre-trained flow matching model ${ \pmb u } _ { t } ^ { \theta }$ <sub>w</sub>hi<sub>c</sub>h <sub>we</sub> <sub>w</sub>i<sub>s</sub>h t<sub>o</sub> <sub>conver</sub>t i<sub>n</sub>t<sub>o</sub> a flow ma<sub>p</sub>. The insi<sub>g</sub>ht of Lee et al. (2026) is that we can chan<sub>g</sub>e our sin<sub>g</sub>le time neural network into a two-time neural network b<sub>y</sub> sim<sub>p</sub>l<sub>y p</sub>assin<sub>g</sub> the first time � to the first � la<sub>y</sub>ers and <sub>p</sub>assin<sub>g</sub> � to the remainin<sub>g</sub> la<sub>y</sub>ers as a conditionin<sub>g</sub> si<sub>g</sub>nal. This has the nice <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> that when $s = \dot { t } , \pmb { u } _ { s , t } ^ { \theta } = \pmb { u } _ { t } ^ { \bar { \theta } }$ ensurin<sub>g</sub> the boundar<sub>y</sub> condition is the <sub>p</sub>re-trained flow matchin<sub>g</sub> model (before fine-tunin<sub>g</sub> at least). Thus the neural network onl<sub>y</sub> needs to l<sub>ea</sub>rn h<sub>o</sub>w t<sub>o</sub> di<sub>s</sub>till <sub>ou</sub>t th<sub>e</sub> int<sub>eg</sub>r<sub>a</sub>l <sub>a</sub>nd n<sub>o</sub>t b<sub>o</sub>th<sub>e</sub>r b<sub>u</sub>rnin<sub>g co</sub>m<sub>pu</sub>t<sub>e o</sub>n l<sub>ea</sub>rnin<sub>g</sub> th<sub>e</sub> di<sub>ago</sub>n<sub>a</sub>l r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>;</sub> in <sub>p</sub>r<sub>ac</sub>ti<sub>ce</sub> w<sub>e</sub> <sub>use</sub> <sub>a</sub> <sub>s</sub>m<sub>a</sub>ll l<sub>oss</sub> t<sub>o</sub> <sub>e</sub>n<sub>su</sub>r<sub>e</sub> th<sub>e</sub> di<sub>ago</sub>n<sub>a</sub>l d<sub>oes</sub> n<sub>o</sub>t div<sub>e</sub>r<sub>ge</sub> b<sub>u</sub>t it i<sub>s</sub> <sub>a</sub> <sub>s</sub>m<sub>a</sub>ll<sub>e</sub>r <sub>pe</sub>r<sub>ce</sub>nt<sub>age</sub> <sub>o</sub>f th<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub>

Exponential lift. Recall that for a pre-trained flow matching model the drift of the Itô SDE is $\begin{array} { r } { 2 { \pmb u } _ { t } ^ { \theta } ( { \pmb X } _ { t } ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } { \pmb X } _ { t } } \end{array}$ . So t<sup>h</sup>e question is <sup>h</sup>ow <sup>d</sup>o we get our <sup>b</sup>oun<sup>d</sup>ary con<sup>d</sup>itions to a<sup>l</sup>ign, i.e., <sup>h</sup>ow to ensure t<sup>h</sup>at our <sup>fl</sup>ow map $\pmb { u } _ { t , t } ^ { \theta }$ e<sub>q</sub>ua<sup>l</sup>s thi<sub>s</sub> drift<sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> tri<sub>c</sub>k<sub>y</sub> n<sub>o</sub>w b<sub>ecause o</sub>f th<sub>e</sub> lin<sub>ea</sub>r t<sub>e</sub>rm whi<sub>c</sub>h i<sub>s u</sub>nr<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>o ou</sub>r <sub>p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> model. Instead, we can use ex<sub>p</sub>onential inte<sub>g</sub>rators (Hochbruck and Ostermann 2010) to lift the SDE into a form that removes the linear term. Thus for the flow matchin<sub>g</sub> SDE in E<sub>q</sub>uation (1) define the inte<sub>g</sub>ratin<sub>g</sub> factor as

$$
\exp \left( \int \mathrm { d } t \ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \right) = \exp \log \alpha _ { t } = \alpha _ { t } .\tag{105}
$$

Then via Itô’s <sub>p</sub>roduct rule a<sub>pp</sub>lied to E<sub>q</sub>uation (1) we find

$$
{ \mathrm { d } } [ \alpha _ { t } X _ { t } ] = \alpha _ { t } { \mathrm { ~ d } } X _ { t } + X _ { t } { \mathrm { ~ d } } \alpha _ { t } + \underbrace { { \mathrm { ~ d } } \left. \alpha \mid X \right. _ { t } } _ { = 0 } ,\tag{106}
$$

$$
= \alpha _ { t } \left( \left[ 2 \pmb { u } _ { t } ^ { \theta } ( \pmb { X } _ { t } ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { X } _ { t } \right] \ \mathrm { d } t + \nu _ { t } \ \mathrm { d } \pmb { W } _ { t } \right) + \pmb { X } _ { t } [ \dot { \alpha } _ { t } ] \ \mathrm { d } t ,\tag{107}
$$

$$
\mathbf { \sigma } = 2 \alpha _ { t } \pmb { u } _ { t } ^ { \theta } ( \pmb { X } _ { t } ) \ \mathrm { d } t + \alpha _ { t } \nu _ { t } \ \mathrm { d } \pmb { W } _ { t } ,\tag{108}
$$

<sub>w</sub>h<sub>ere</sub> $\langle \cdot | \cdot \rangle _ { t }$ is the covariation of two <sub>p</sub>rocesses. Now<sub>,</sub> with the linear term absorbed<sub>,</sub> this <sub>p</sub>arameterization enables us to more easil<sub>y</sub> satisf<sub>y</sub> the boundar<sub>y</sub> condition. This SDE can be written as an inte<sub>g</sub>ral e<sub>q</sub>uation of the f<sub>orm</sub>

$$
X _ { t } = \frac { \alpha _ { s } } { \alpha _ { t } } X _ { s } + \frac { 2 } { \alpha _ { t } } \int _ { s } ^ { t } \mathrm { d } r \ \alpha _ { r } { u } _ { r } ^ { \theta } ( X _ { r } ) + \frac { 1 } { \alpha _ { t } } \int _ { s } ^ { t } \mathrm { d } W _ { r } \ \alpha _ { r } \nu _ { r } .\tag{109}
$$

Th<sub>us</sub> w<sub>e</sub> <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>e</sub> <sub>ou</sub>r <sub>e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>ll<sub>y</sub> lift<sub>e</sub>d fl<sub>o</sub>w m<sub>ap</sub> <sub>as</sub>

$$
\Psi _ { s , t } ^ { \theta } ( \pmb { x } ; \pmb { \mathscr { T } } _ { s , t } ^ { ( N ) } ) = \frac { \alpha _ { s } } { \alpha _ { t } } \pmb { x } + \frac { 2 \int _ { s } ^ { t } \mathrm { d } r ~ \alpha _ { r } } { \alpha _ { t } } \pmb { u } _ { s , t } ^ { \theta } ( \pmb { x } ; \pmb { \mathscr { T } } _ { s , t } ^ { ( N ) } ) + \frac { 1 } { \alpha _ { t } } ( \pmb { M } _ { t } ^ { \alpha } - \pmb { M } _ { s } ^ { \alpha } ) ,\tag{110}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { I } _ { s , t } ^ { ( N ) }$ <sub>a</sub>r<sub>e</sub> th<sub>e</sub> � <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s</sub> <sub>o</sub>f th<sub>e</sub> tim<sub>e</sub>-<sub>c</sub>h<sub>a</sub>n<sub>ge</sub>d Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th $M _ { t } ^ { \alpha }$ <sub>a</sub>nd $\int _ { s } ^ { t }$ d<sub>�</sub> $\alpha _ { r }$ i<sub>s</sub> <sub>co</sub>m<sub>pu</sub>t<sub>e</sub>d in <sub>c</sub>l<sub>ose</sub>df<sub>orm.</sub>

Adding the Brownian path coefficients to the network. We add a new neural network $\pmb { f } ^ { \theta } : \mathbb { R } ^ { d } \oplus ( \mathbb { R } ^ { d } ) ^ { \oplus N }  \mathbb { R } ^ { d }$ whi<sub>c</sub>h w<sub>e a</sub>dd t<sub>o</sub> th<sub>e</sub> b<sub>eg</sub>innin<sub>g o</sub>f th<sub>e</sub> dif<sub>us</sub>i<sub>o</sub>n tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>e</sub>r <sub>p</sub>i<sub>pe</sub>lin<sub>e a</sub>ft<sub>e</sub>r th<sub>e pa</sub>t<sub>c</sub>h <sub>e</sub>mb<sub>e</sub>ddin<sub>g</sub> with <sub>a</sub> r<sub>es</sub>id<sub>ua</sub> connection, i.e.,

$$
{ \pmb y } = { \pmb x } + ( t - s ) { \pmb f } ^ { \theta } \left( { \pmb x } \oplus \bigoplus _ { n = 0 } ^ { N - 1 } { \pmb I } _ { s , t } ^ { ( n ) } \right) ,\tag{111}
$$

where � here denotes the patch-embedded tokens and the (� −�) gate makes the injection vanish on the diagonal $s = t$ (see A<sub>pp</sub>endix G for details).

W<sub>e</sub> tri<sub>e</sub>d <sub>o</sub>th<sub>e</sub>r v<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> b<sub>u</sub>t f<sub>ou</sub>nd thi<sub>s</sub> t<sub>o</sub> b<sub>e</sub> th<sub>e</sub> m<sub>os</sub>t <sub>e</sub>f<sub>ec</sub>tiv<sub>e.</sub>

## D.1 Annealing the SDE

W<sub>e a</sub>l<sub>so</sub> tri<sub>e</sub>d <sub>a</sub>nn<sub>ea</sub>lin<sub>g</sub> th<sub>e</sub> SDE<sub>,</sub> whi<sub>c</sub>h w<sub>e</sub> d<sub>esc</sub>rib<sub>e</sub> b<sub>e</sub>l<sub>o</sub>w<sub>.</sub> Th<sub>us, o</sub>n<sub>e cou</sub>ld <sub>s</sub>t<sub>a</sub>rt b<sub>y</sub> tr<sub>a</sub>inin<sub>g a</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>ap a</sub>nd <sub>s</sub>l<sub>o</sub>wl<sub>y</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n t<sub>o</sub> l<sub>ea</sub>rnin<sub>g a s</sub>tr<sub>o</sub>n<sub>g s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>ap.</sub> Em<sub>p</sub>iri<sub>ca</sub>ll<sub>y</sub> w<sub>e</sub> f<sub>ou</sub>nd thi<sub>s</sub> did n<sub>o</sub>t h<sub>e</sub>l<sub>p</sub> m<sub>uc</sub>h with trainin<sub>g</sub>, resultin<sub>g</sub> in the same <sub>p</sub>erformance re<sub>g</sub>ardless (over our 50k iteration schedule), but we include it f<sub>or</sub> th<sub>e rea</sub>d<sub>er.</sub>

Recall our flow matchin<sub>g</sub> ODE and e<sub>q</sub>uivalent (in law) SDE

$$
\dot { \mathbf { x } } _ { t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \mathbf { x } _ { t } + \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla _ { x } \log p _ { t } ( \mathbf { x } _ { t } ) ,\tag{112}
$$

$$
\mathrm { d } X _ { t } = \left[ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \nu _ { t } ^ { 2 } \nabla _ { x } \log p _ { t } ( X _ { t } ) \right] \mathrm { ~ d } t + \nu _ { t } \mathrm { ~ d } W _ { t } ,\tag{113}
$$

which we write w.r.t. the score function to make subse<sub>q</sub>uent derivations easier recallin<sub>g</sub> the identit<sub>y</sub>

$$
\pmb { u } _ { t } ( \pmb { x } _ { t } ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { x } _ { t } + \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla _ { \pmb { x } } \log p _ { t } ( \pmb { x } _ { t } ) .\tag{114}
$$

W<sub>e</sub> n<sub>o</sub>w r<sub>e</sub>d<sub>e</sub>riv<sub>e</sub> <sub>a</sub> m<sub>o</sub>r<sub>e</sub> <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>l f<sub>o</sub>rm <sub>us</sub>in<sub>g</sub> th<sub>e</sub> F<sub>o</sub>kk<sub>e</sub>r–Pl<sub>a</sub>n<sub>c</sub>k–K<sub>o</sub>lm<sub>ogo</sub>r<sub>o</sub>v <sub>equa</sub>ti<sub>o</sub>n<sub>.</sub> F<sub>o</sub>r n<sub>o</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>s</sub>h<sub>o</sub>rth<sub>a</sub>nd l<sub>e</sub>t $\begin{array} { r } { \pmb { a } _ { t } ( \pmb { x } ) : = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { x } } \end{array}$ . T<sup>h</sup>e <sup>d</sup>ensity o<sup>f</sup> t<sup>h</sup>e <sup>fl</sup>ow matc<sup>h</sup>ing ODE <sup>f</sup>o<sup>ll</sup>ows t<sup>h</sup>e continuity equation, i.e.,

$$
\partial _ { t } p _ { t } = - \left. \nabla \mid p _ { t } \pmb { u } _ { t } \right. ,\tag{115}
$$

where we dro<sub>p</sub> the ar<sub>g</sub>uments for notational sim<sub>p</sub>licit<sub>y</sub>. Then we can mani<sub>p</sub>ulate this e<sub>q</sub>uation to show

$$
\partial _ { t } p _ { t } = - \left. \nabla \mid p _ { t } \pmb { u } _ { t } \right. ,
$$

$$
\mathbf { \tau } = -  \nabla | p _ { t } [ \pmb { a } _ { t } + \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ]  ,\tag{116}
$$

(117)

$$
\boldsymbol { \stackrel { ( i ) } { = } } -  \nabla | p _ { t } [ \pmb { a } _ { t } + { ( 1 + \zeta ^ { 2 } ) } \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } - \zeta ^ { 2 } \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ]  ,\tag{118}
$$

$$
= - \left. \nabla \bigg | p _ { t } \left[ a _ { t } + \big ( 1 + \zeta ^ { 2 } \big ) \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } \right] \right. + \zeta ^ { 2 } \frac { \nu _ { t } ^ { 2 } } { 2 } \left. \nabla \mid p _ { t } \nabla \log p _ { t } \right. ,\tag{119}
$$

$$
= -  \nabla | p _ { t } [ \pmb { a } _ { t } + ( 1 + \zeta ^ { 2 } ) \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ]  + \zeta ^ { 2 } \frac { \nu _ { t } ^ { 2 } } { 2 }  \nabla \mid \nabla p _ { t }  ,\tag{120}
$$

$$
= -  \nabla |  p _ { t } [ a _ { t } + ( 1 + \zeta ^ { 2 } ) \frac { \nu _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ]  + \zeta ^ { 2 } \frac { \nu _ { t } ^ { 2 } } { 2 } \Delta p _ { t } ,\tag{121}
$$

where (i) holds b<sub>y</sub> the identit<sub>y</sub> $x = ( 1 + \zeta ^ { 2 } ) x - \zeta ^ { 2 } { \overset { } { . } { } }$ <sub>�</sub> <sub>a</sub>nd $\Delta$ <sup>d</sup>enotes t<sup>h</sup>e Lap<sup>l</sup>ace operator, i.e., $\Delta p _ { t } = \langle \nabla \mid \nabla p _ { t } \rangle$ This PDE for $p _ { t }$ has an e<sub>q</sub>uivalent Itô SDE <sub>g</sub>iven b<sub>y</sub>

$$
\mathrm { d } X _ { t } = \left[ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \frac { 1 + \zeta ^ { 2 } } { 2 } \nu _ { t } ^ { 2 } \nabla _ { x } \log p _ { t } ( X _ { t } ) \right] \mathrm { ~ d } t + \zeta \nu _ { t } \mathrm { ~ d } W _ { t } .\tag{122}
$$

Remark D.1. When $\zeta = 1$ we have the standard difusion SDE written in flow matchin<sub>g</sub> conventions (Anderson 1982; Son<sub>g</sub> et al. 2021) and when $\zeta = 0$ we have the standard flow matchin<sub>g</sub> ODE or <sub>p</sub>robabilit<sub>y</sub> flow ODE (Li<sub>p</sub>man et al. 2023; Maoutsa et al. 2020; Son<sub>g</sub> et al. 2021).

W<sub>e</sub> <sub>ca</sub>n r<sub>e</sub>writ<sub>e</sub> thi<sub>s</sub> w<sub>.</sub>r<sub>.</sub>t<sub>.</sub> ${ \pmb u } _ { t }$ as

$$
\mathrm { d } X _ { t } = \left[ \bigl ( 1 + \zeta ^ { 2 } \bigr ) { \pmb u } _ { t } \bigl ( X _ { t } \bigr ) - \zeta ^ { 2 } \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } \right] \ \mathrm { d } t + \zeta \nu _ { t } \ \mathrm { d } W _ { t } .\tag{123}
$$

Let $\pmb { X } _ { t } ^ { \zeta }$ denote the solution trajectories for diferent values of $\zeta \in [ 0 , 1 ]$ . One idea for trainin<sub>g</sub> is that we could anneal � from 0 to 1 slowl<sub>y</sub> transitionin<sub>g</sub> from the full ODE to the SDE.

Exponential integrators. As we use an exponentially lifted version to fine-tune pre-trained flow matching models we now <sub>p</sub>resent the derivation for an ex<sub>p</sub>onentiall<sub>y</sub> lifted version of our SDE. The inte<sub>g</sub>ratin<sub>g</sub> factor is now

$$
\exp \left( \int \mathrm { d } t \ \zeta ^ { 2 } \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \right) = \exp \log \alpha _ { t } ^ { \zeta ^ { 2 } } = \alpha _ { t } ^ { \zeta ^ { 2 } } ,\tag{124}
$$

whi<sub>c</sub>h <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\mathbf { d } \left[ \alpha _ { t } ^ { \zeta ^ { 2 } } X _ { t } \right] = ( 1 + \zeta ^ { 2 } ) \alpha _ { t } ^ { \zeta ^ { 2 } } { \pmb u } _ { t } ( X _ { t } ) \mathbf { d } t + \zeta \alpha _ { t } ^ { \zeta ^ { 2 } } { \pmb v } _ { t } \mathbf { d } W _ { t } .\tag{125}
$$

Conse<sub>q</sub>uentl<sub>y,</sub> for $0 < s \leq t \leq 1$ ,

$$
X _ { t } = \left( \frac { \alpha _ { s } } { \alpha _ { t } } \right) ^ { \zeta ^ { 2 } } X _ { s } + \frac { 1 + \zeta ^ { 2 } } { \alpha _ { t } ^ { \zeta ^ { 2 } } } \int _ { s } ^ { t } \mathrm { d } r ~ \alpha _ { r } ^ { \zeta ^ { 2 } } u _ { r } ( X _ { r } ) + \frac { \zeta } { \alpha _ { t } ^ { \zeta ^ { 2 } } } \int _ { s } ^ { t } \mathrm { d } W _ { r } ~ \alpha _ { r } ^ { \zeta ^ { 2 } } \nu _ { r } .\tag{126}
$$

Let

$$
\mathcal { A } _ { s , t } ^ { ( \zeta ) } : = \int _ { s } ^ { t } \mathrm { d } r ~ \alpha _ { r } ^ { \zeta ^ { 2 } } , \qquad \boldsymbol { u } _ { s , t } ^ { ( \zeta ) } : = \frac { 1 } { \mathcal { A } _ { s , t } ^ { ( \zeta ) } } \int _ { s } ^ { t } \mathrm { d } r ~ \alpha _ { r } ^ { \zeta ^ { 2 } } \boldsymbol { u } _ { r } ( \boldsymbol { X } _ { r } ) .\tag{127}
$$

Then the ex<sub>p</sub>onentiall<sub>y</sub> lifted �-SDE stochastic flow ma<sub>p</sub> is

$$
\Phi _ { s , t } ^ { \zeta } ( \pmb { x } ) = \left( \frac { \alpha _ { s } } { \alpha _ { t } } \right) ^ { \zeta ^ { 2 } } \pmb { x } + \frac { ( 1 + \zeta ^ { 2 } ) \mathcal { \mathcal { R } } _ { s , t } ^ { ( \zeta ) } } { \alpha _ { t } ^ { \zeta ^ { 2 } } } \pmb { u } _ { s , t } ^ { \theta , ( \zeta ) } ( \pmb { x } ; T ) + \frac { \zeta } { \alpha _ { t } ^ { \zeta ^ { 2 } } } \int _ { s } ^ { t } \mathrm { d } W _ { r } \ \alpha _ { r } ^ { \zeta ^ { 2 } } \nu _ { r } .\tag{128}
$$

Th<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> t<sub>e</sub>rm i<sub>s</sub> G<sub>auss</sub>i<sub>a</sub>n with <sub>co</sub>v<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub>

$$
\frac { \zeta ^ { 2 } } { \alpha _ { t } ^ { 2 \zeta ^ { 2 } } } \left( \int _ { s } ^ { t } \mathrm { d } r ~ \alpha _ { r } ^ { 2 \zeta ^ { 2 } } \nu _ { r } ^ { 2 } \right) I .\tag{129}
$$

At $\zeta = 0$ , E<sub>q</sub>uation (128) reduces to the usual Euler flow ma<sub>p</sub> $\mathbf { \Delta } \mathbf { x } + ( t - s ) \mathbf { \Delta } u _ { s , t } ^ { \theta , ( 0 ) }$ <sub>;</sub> <sub>converse</sub>l<sub>y,</sub> <sub>w</sub>h<sub>en</sub> $\zeta = 1$ it r<sub>eco</sub>v<sub>e</sub>r<sub>s</sub> <sub>ou</sub>r <sub>o</sub>ri<sub>g</sub>in<sub>a</sub>l <sub>e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>ll<sub>y</sub> lift<sub>e</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>ap.</sub>

Findings. While the annealing works we did not find it to meaningfully accelerate or make the fine-tuning <sub>p</sub>rocess <sup>b</sup>etter, $i . e .$ <sub>,</sub> after 100k trainin<sub>g</sub> iterations it rou<sub>g</sub>hl<sub>y</sub> conver<sub>g</sub>ed to our non-annealed iSSFM runs. Whilst <sub>a</sub>n int<sub>e</sub>r<sub>es</sub>tin<sub>g</sub> id<sub>ea,</sub> w<sub>e</sub> didn’t find thi<sub>s</sub> t<sub>o</sub> m<sub>ea</sub>nin<sub>g</sub>f<sub>u</sub>ll<sub>y</sub> im<sub>pac</sub>t th<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> f<sub>o</sub>r b<sub>e</sub>tt<sub>e</sub>r <sub>o</sub>r f<sub>o</sub>r w<sub>o</sub>r<sub>se.</sub>

## E Feynman–Kac correctors

## E.1 Background

We introduce Fe<sub>y</sub>nman–Kac correctors (FKC) from Skreta, Akhound-Sade<sub>g</sub>h, et al. (2025) but in the conventions <sub>o</sub>f fl<sub>ow ma</sub>t<sub>c</sub>hi<sub>ng over</sub> dif<sub>us</sub>i<sub>on.</sub>

Recall our flow matchin<sub>g</sub> SDE from E<sub>q</sub>uation (1), which we can rewrite w.r.t. the score function $\nabla _ { x } \log { p _ { t } ( x ) }$ as

$$
\mathrm { d } X _ { t } = \left[ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \nu _ { t } ^ { 2 } \nabla _ { x } \log p _ { t } ( X _ { t } ) \right] \mathrm { ~ d } t + \nu _ { t } \mathrm { ~ d } W _ { t } .\tag{130}
$$

Followin<sub>g</sub> Skreta, Akhound-Sade<sub>g</sub>h, et al. (2025) consider � diferent flow matchin<sub>g</sub> models which admit d<sub>e</sub>n<sub>s</sub>iti<sub>es</sub> $\{ q _ { t } ^ { i } \} _ { i = 1 } ^ { M }$ f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub>

$$
\partial _ { t } q _ { t } ^ { i } ( \pmb { x } ) = - \left. \nabla _ { \pmb { x } } \middle | q _ { t } ^ { i } ( \pmb { x } ) \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { x } + \nu _ { t } ^ { 2 } \nabla _ { \pmb { x } } \log q _ { t } ^ { i } ( \pmb { x } ) \right) \right. + \frac { \nu _ { t } ^ { 2 } } { 2 } \Delta _ { \pmb { x } } q _ { t } ^ { i } ( \pmb { x } ) ,\tag{131}
$$

$$
\mathrm { d } X _ { t } = \left[ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \nu _ { t } ^ { 2 } \nabla _ { x } \log q _ { t } ^ { i } ( X _ { t } ) \right] \mathrm { ~ d } t + \nu _ { t } \mathrm { ~ d } W _ { t } ,\tag{132}
$$

<sub>w</sub>h<sub>ere</sub> $\Delta _ { x }$ denotes the Laplace operator w.r.t. �. Let $\begin{array} { r } { { \pmb s } _ { t } ^ { i } ( { \pmb x } ) = \nabla _ { { \pmb x } } \log q _ { t } ^ { i } ( { \pmb x } ) , { \pmb a } _ { t } ( { \pmb x } ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } { \pmb x } , } \end{array}$ <sub>a</sub>nd ${ \pmb f } _ { t } ^ { i } ( { \pmb x } ) = { \pmb a } _ { t } ( { \pmb x } ) +$ $\nu _ { t } ^ { 2 } \pmb { s } _ { t } ^ { i } ( \pmb { x } )$ for sim<sub>p</sub>licit<sub>y</sub>; we can find the evolution of the lo<sub>g</sub> densit<sub>y</sub> (dro<sub>pp</sub>in<sub>g</sub> ar<sub>g</sub>uments for notational brevit<sub>y</sub>)

followin<sub>g</sub> a similar derivation to Skreta, Atanackovic, et al. (2025, Theorem 1) via

$$
\partial _ { t } q _ { t } ^ { i } = -  \nabla | q _ { t } ^ { i } { f _ { t } ^ { i } }  + \frac { \nu _ { t } ^ { 2 } } { 2 } \Delta q _ { t } ^ { i } ,\tag{133}
$$

$$
q _ { t } ^ { i } \partial _ { t } \log q _ { t } ^ { i } = -  \nabla | q _ { t } ^ { i } \pmb { f } _ { t } ^ { i }  + \frac { \nu _ { t } ^ { 2 } } { 2 } \Delta q _ { t } ^ { i } ,\tag{134}
$$

$$
\partial _ { t } \log { q _ { t } ^ { i } } = - \frac { 1 } { q _ { t } ^ { i } }  \nabla | q _ { t } ^ { i } \mathbf { f } _ { t } ^ { i }  + \frac { \nu _ { t } ^ { 2 } } { 2 } \frac { 1 } { q _ { t } ^ { i } } \Delta q _ { t } ^ { i } ,\tag{135}
$$

$$
= - \frac { 1 } { { q } _ { t } ^ { i } } ( { q } _ { t } ^ { i }  \nabla | { f } _ { t } ^ { i }  +  \nabla { q } _ { t } ^ { i } | { f } _ { t } ^ { i }  ) + \frac { \nu _ { t } ^ { 2 } } { 2 } \frac { 1 } { { q } _ { t } ^ { i } }  \nabla | \nabla { q } _ { t } ^ { i }  ,\tag{136}
$$

$$
= -  \nabla | f _ { t } ^ { i }  - \frac { 1 } { q _ { t } ^ { i } }  \nabla q _ { t } ^ { i } | f _ { t } ^ { i }  + \frac { \nu _ { t } ^ { 2 } } { 2 } (  \nabla | \frac { 1 } { q _ { t } ^ { i } } \nabla q _ { t } ^ { i }  -  \nabla \frac { 1 } { q _ { t } ^ { i } } | \nabla q _ { t } ^ { i }  ) ,\tag{137}
$$

$$
= -  \nabla | \mathbf { \nabla } f _ { t } ^ { i }  -  \nabla \log q _ { t } ^ { i } | f _ { t } ^ { i }  + \frac { \nu _ { t } ^ { 2 } } { 2 } (  \nabla | \nabla \log q _ { t } ^ { i }  + \| \nabla \log q _ { t } ^ { i } \| ^ { 2 } ) ,\tag{138}
$$

$$
= - \left. \nabla \left| \mathbf { \nabla } f _ { t } ^ { i } \right. - \left. \nabla \log q _ { t } ^ { i } \right| f _ { t } ^ { i } \right. + \frac { \nu _ { t } ^ { 2 } } { 2 } \left( \Delta \log q _ { t } ^ { i } + \Vert \nabla \log q _ { t } ^ { i } \Vert ^ { 2 } \right) ,\tag{139}
$$

$$
= - \left. \nabla \mid \mathbf { a } _ { t } \right. - \left. \nabla \log { q } _ { t } ^ { i } \mid \mathbf { a } _ { t } \right. - \frac { \nu _ { t } ^ { 2 } } { 2 } \left( \Delta \log { q } _ { t } ^ { i } + \Vert \nabla \log { q } _ { t } ^ { i } \Vert ^ { 2 } \right) ,\tag{140}
$$

$$
= - \left. \nabla \mid \pmb { a } _ { t } \right. - \left. \pmb { s } _ { t } ^ { i } \mid \pmb { a } _ { t } \right. - \frac { \nu _ { t } ^ { 2 } } { 2 } \left( \left. \nabla \mid \pmb { s } _ { t } ^ { i } \right. + \| \pmb { s } _ { t } ^ { i } \| ^ { 2 } \right) .\tag{141}
$$

N<sub>e</sub>xt w<sub>e</sub> l<sub>e</sub>t $\beta = ( \beta _ { 1 } , \dots , \beta _ { M } )$ d<sub>e</sub>n<sub>o</sub>t<sub>e a co</sub>m<sub>pos</sub>iti<sub>o</sub>n r<sub>u</sub>l<sub>e</sub> f<sub>o</sub>r th<sub>e</sub> dif<sub>e</sub>r<sub>e</sub>nt m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> wh<sub>e</sub>r<sub>e</sub> $\beta _ { i } \in \mathbb { R }$ <sub>.</sub> Th<sub>en</sub> th<sub>e</sub> <sub>u</sub>nn<sub>o</sub>rm<sub>a</sub>liz<sub>e</sub>d <sub>p</sub>r<sub>o</sub>d<sub>uc</sub>t t<sub>a</sub>r<sub>ge</sub>t <sub>ca</sub>n b<sub>e</sub> writt<sub>e</sub>n <sub>as</sub>

$$
\widetilde { \pi } _ { t } ^ { \beta } ( \pmb { x } ) = \prod _ { i = 1 } ^ { M } q _ { t } ^ { i } ( \pmb { x } ) ^ { \beta _ { i } } ,\tag{142}
$$

with <sub>pa</sub>rtiti<sub>o</sub>n f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

$$
Z _ { t } ^ { \beta } = \int _ { \mathbb { R } ^ { d } } \mathrm { d } \pmb { x } \ \widetilde { \pi } _ { t } ^ { \beta } ( \pmb { x } ) .\tag{143}
$$

Th<sub>en</sub> l<sub>e</sub>t

$$
\begin{array} { r } { \pmb { s } _ { t } ^ { \beta } ( \pmb { x } ) = \nabla _ { \pmb { x } } \log \frac { \widetilde { \pi } _ { t } ^ { \beta } ( \pmb { x } ) } { Z _ { t } ^ { \beta } } = \displaystyle \sum _ { i = 1 } ^ { M } \beta _ { i } \pmb { s } _ { t } ^ { i } ( \pmb { x } ) . } \end{array}\tag{144}
$$

This com<sub>p</sub>osed score SDE is thus

$$
\mathrm { d } X _ { t } = \underbrace { \Bigg [ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \nu _ { t } ^ { 2 } s _ { t } ^ { \beta } ( \pmb { x } ) \Bigg ] } _ { : = f _ { t } ^ { \beta } ( \pmb { x } ) } \mathrm { d } t + \nu _ { t } \mathrm { d } { \pmb { W } } _ { t } .\tag{145}
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the Fokker–Planck o<sub>p</sub>erator of this SDE to the tar<sub>g</sub>et densit<sub>y</sub> and dividin<sub>g</sub> b<sub>y</sub> it (usin<sub>g</sub> ∇ lo<sub>g</sub> $\pi _ { t } = \mathbf { \vec { s } } _ { t } ^ { \beta } )$ <sub>g</sub><sup>i</sup>ves

$$
\frac { \mathcal { L } _ { t } ^ { \beta * } \pi _ { t } } { \pi _ { t } } = - \left. \nabla \mid \pmb { a } _ { t } \right. - \left. \pmb { s } _ { t } ^ { \beta } \middle | \pmb { a } _ { t } \right. - \frac { \nu _ { t } ^ { 2 } } { 2 } \left( \left. \nabla \middle | \pmb { s } _ { t } ^ { \beta } \right. + \Vert \pmb { s } _ { t } ^ { \beta } \Vert ^ { 2 } \right) .\tag{146}
$$

Now<sub>,</sub> we can write the tar<sub>g</sub>et lo<sub>g</sub>-densit<sub>y</sub> as

$$
\log \pi _ { t } ( x ) = \sum _ { i = 1 } ^ { M } \beta _ { i } \log q _ { t } ^ { i } ( x ) - \log Z _ { t } ,\tag{147}
$$

<sub>a</sub>nd th<sub>us</sub> th<sub>e</sub> tim<sub>e e</sub>v<sub>o</sub>l<sub>u</sub>ti<sub>o</sub>n i<sub>s</sub> f<sub>ou</sub>nd vi<sub>a</sub>

(148)

(149)

$$
\begin{array} { r l } & { \frac { \lambda } { \mu } \int _ { \mathbb { R } } \log ( x ( \theta ) ) = \displaystyle \sum _ { j = 1 } ^ { \infty } \beta _ { j } \log ( \frac { 1 } { x } ) - \delta \log \int _ { \mathbb { R } } \log ( x ^ { 0 } ) ^ { \theta } \log ^ { \theta } \Bigg [ \frac { 1 } { x } \Bigg [ \int _ { 0 } ^ { x } \log ( x ^ { 0 } ) ^ { \theta } \cdot } \\ & { \quad - \displaystyle \sum _ { j = 1 } ^ { \infty } \beta _ { j } \log ( x ^ { 0 } ) - \frac { \delta \log ( x ^ { 0 } ) } { \int _ { 0 } ^ { x } \log ( 1 + x ^ { 0 } ) ^ { \theta } ( \log ^ { \theta } ( x ^ { 0 } ) ) } \cdot } \\ & { \quad - \displaystyle \sum _ { j = 1 } ^ { \infty } \beta _ { j } \log ( x ^ { 0 } ) - \frac { \delta \log ( x ^ { 0 } ) } { \int _ { 0 } ^ { x } \log ( 1 + x ^ { 0 } ) ^ { \theta } ( \log ^ { \theta } ( x ^ { 0 } ) ) } \cdot } \\ & { \quad - \displaystyle \sum _ { j = 1 } ^ { \infty } \beta _ { j } \log ( x ^ { 0 } ) - \frac { \delta \log ( x ^ { 0 } ) } { \int _ { 0 } ^ { x } \log ( 1 + x ^ { 0 } ) ^ { \theta } ( \log ^ { \theta } ( x ^ { 0 } ) ) } \cdot } \\ & { \quad - \displaystyle \sum _ { j = 1 } ^ { \infty } \beta _ { j } \log ( x ^ { 0 } ) - \frac { \delta \log ( x ^ { 0 } ) } { \int _ { 0 } ^ { x } \log ( 1 + x ^ { 0 } ) ^ { \theta } ( \log ^ { \theta } ( x ^ { 0 } ) ) } \cdot } \\ &  \quad - \displaystyle \sum _ { j = 1 } ^ { \infty } \beta _ { j } \log ( x ^ { 0 } ) - \frac  \delta \log ( x ^ { 0 }  \end{array}\tag{150}
$$

(151)

(152)

(153)

(154)

Then b<sub>y</sub> E<sub>q</sub>uation (141) for the first term we have

$$
h _ { t } = \sum _ { i = 1 } ^ { M } \beta _ { i } \partial _ { t } \log { q _ { t } ^ { i } } = - B \left. \nabla \mid \boldsymbol { a } _ { t } \right. - \left. \boldsymbol { s } _ { t } ^ { \beta } \left. \boldsymbol { a } _ { t } \right. - \frac { \nu _ { t } ^ { 2 } } { 2 } \left( \left. \nabla \left| \boldsymbol { s } _ { t } ^ { \beta } \right. + \sum _ { i = 1 } ^ { M } \beta _ { i } \right. \boldsymbol { s } _ { t } ^ { i } \right. ^ { 2 } \right) ,\tag{155}
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { B : = \sum _ { i = 1 } ^ { M } \beta _ { i } } \end{array}$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t $\partial _ { t }$ <sup>l</sup>o<sub>g</sub> $\pi _ { t } = h _ { t } - \mathbb { E } _ { \pi _ { t } } [ h _ { t } ]$ <sub>.</sub> Th<sub>us</sub> i<sub>n</sub> <sub>genera</sub>l <sub>we</sub> d<sub>o</sub> <sub>no</sub>t h<sub>ave</sub> th<sub>a</sub>t th<sub>e</sub> l<sub>aw</sub> <sub>o</sub>f th<sub>e</sub> <sub>proposa</sub>l SDE in E<sub>q</sub>uation (145) and the tar<sub>g</sub>et $\pi _ { t }$ <sub>co</sub>in<sub>c</sub>id<sub>e.</sub>

To correct this mismatch between the <sub>p</sub>ro<sub>p</sub>osal d<sub>y</sub>namics and our tar<sub>g</sub>et distribution we introduce a scalar <sub>p</sub>otential $g _ { t } : \mathbb { R } ^ { d }  \mathbb { I }$ ℝ. For some terminal test function � define

$$
u ( s , \pmb { x } ) = \mathbb { E } \left[ \exp \left( \int _ { s } ^ { t } \mathrm { d } r \ g _ { r } ( X _ { r } ) \right) \phi ( X _ { t } ) \ \Bigg | \ d X _ { s } = \pmb { x } \right] .\tag{156}
$$

Then under suitable re<sub>g</sub>ularit<sub>y</sub> conditions the Fe<sub>y</sub>nman–Kac formula (Fe<sub>y</sub>nman 1948; Kac 1949) describes a PDE corres<sub>p</sub>ondin<sub>g</sub> to the conditional ex<sub>p</sub>ectation<sub>,</sub>

$$
\partial _ { s } u + \mathcal { L } _ { s } ^ { \beta } u + g _ { s } u = 0 , \qquad u ( t , \cdot ) = \phi ,\tag{157}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { L } _ { s } ^ { \beta }$ is the <sub>g</sub>enerator of the SDE in E<sub>q</sub>uation (145). Likewise, the forward adjoint e<sub>q</sub>uation of this Fe<sub>y</sub>nman– K<sub>ac equa</sub>ti<sub>o</sub>n d<sub>esc</sub>rib<sub>es</sub> th<sub>e e</sub>v<sub>o</sub>l<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e u</sub>nn<sub>o</sub>rm<sub>a</sub>liz<sub>e</sub>d w<sub>e</sub>i<sub>g</sub>ht<sub>e</sub>d m<sub>easu</sub>r<sub>e</sub> $\mu _ { t } ( \phi ) = \mathbb { E } [ e ^ { w _ { t } } \phi ( X _ { t } ) ]$ <sub>w</sub>ith $\dot { w } _ { t } =$ $g _ { t } ( \pmb { X } _ { t } )$ , viz.

$$
\partial _ { t } \mu _ { t } = \mathcal { L } _ { t } ^ { \beta * } \mu _ { t } + g _ { t } \mu _ { t } .\tag{158}
$$

N<sub>o</sub>w l<sub>e</sub>t $\mu _ { t } \propto \pi _ { t }$ <sub>;</sub> normalizin<sub>g</sub> <sub>g</sub>ives $\partial _ { t } \pi _ { t } = \mathcal { L } _ { t } ^ { \beta * } \pi _ { t } + \bar { g } _ { t } \pi _ { t }$ <sub>w</sub>ith $\bar { g } _ { t } : = g _ { t } - \mathbb { E } _ { \pi _ { t } } [ g _ { t } ]$ <sub>,</sub> <sub>a</sub>nd <sub>c</sub>h<sub>oos</sub>in<sub>g</sub> $g _ { t } = \partial _ { t }$ <sup>l</sup>o<sub>g</sub> $\widetilde { \pi } _ { t } -$

$\mathcal { L } _ { t } ^ { \beta * } \pi _ { t } / \pi _ { t }$ (uni<sub>q</sub>ue u<sub>p</sub> to an �-inde<sub>p</sub>endent constant) <sub>y</sub>ields

$$
\dot { w } _ { t } = \partial _ { t } \log \pi _ { t } - \frac { \mathcal { L } _ { t } ^ { \beta * } \pi _ { t } } { \pi _ { t } } ,\tag{159}
$$

$$
= h _ { t } - \mathbb { E } _ { { \boldsymbol { x } } \sim \pi _ { t } } \left[ h _ { t } \right] + \left. \nabla \mid { \boldsymbol { a } } _ { t } \right. + \left. { \boldsymbol { s } } _ { t } ^ { \beta } \middle | { \boldsymbol { a } } _ { t } \right. + { \frac { v _ { t } ^ { 2 } } { 2 } } \left( \left. \nabla \middle | { \boldsymbol { s } } _ { t } ^ { \beta } \right. + \Vert { \boldsymbol { s } } _ { t } ^ { \beta } \Vert ^ { 2 } \right) ,\tag{160}
$$

$$
= \left( 1 - B \right) \left. \nabla \mid a _ { t } \right. - \operatorname { \mathbb { E } } _ { x \sim \pi _ { t } } [ h _ { t } ] + \frac { \nu _ { t } ^ { 2 } } { 2 } \Vert s _ { t } ^ { \beta } \Vert ^ { 2 } - \frac { \nu _ { t } ^ { 2 } } { 2 } \sum _ { i = 1 } ^ { M } \beta _ { i } \Vert s _ { t } ^ { i } \Vert ^ { 2 } .\tag{161}
$$

Since $\mathbb { E } _ { \pi _ { t } } [ h _ { t } ] \ : = \ : \partial _ { t }$ <sup>l</sup>o<sub>g</sub> $Z _ { t }$ i<sub>s</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>o</sub>f $\mathbf { x } ,$ th<sub>e norma</sub>li<sub>za</sub>ti<sub>on o</sub>f th<sub>e we</sub>i<sub>g</sub>ht<sub>s</sub> $w _ { t }$ ensures t<sup>h</sup>at an<sub>y</sub> terms independent of � cancel out—this includes ${ \pmb a } _ { t } ( { \pmb x } )$ <sub>w</sub>h<sub>ose</sub> di<sub>vergence cance</sub>l<sub>s ou</sub>t <sub>as we</sub>ll<sub>.</sub> Th<sub>us we can wr</sub>it<sub>e</sub> th<sub>e</sub> Fe<sub>y</sub>nman–Kac corrector (FKC) sam<sub>p</sub>ler as

$$
\mathrm { d } X _ { t } = \left[ \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \nu _ { t } ^ { 2 } \sum _ { i = 1 } ^ { M } \beta _ { i } s _ { t } ^ { i } ( X _ { t } ) \right] \mathrm { ~ d } t + \nu _ { t } \mathrm { ~ d } W _ { t } ,\tag{162}
$$

$$
\mathrm { d } w _ { t } = \left[ \frac { \nu _ { t } ^ { 2 } } { 2 } \left\| \sum _ { i = 1 } ^ { M } \beta _ { i } \pmb { s } _ { t } ^ { i } ( \pmb { X } _ { t } ) \right\| ^ { 2 } - \frac { \nu _ { t } ^ { 2 } } { 2 } \sum _ { i = 1 } ^ { M } \beta _ { i } \| \pmb { s } _ { t } ^ { i } ( \pmb { X } _ { t } ) \| ^ { 2 } \right] \ \mathrm { d } t .\tag{163}
$$

## E.2 Augmented space stochastic flow maps for FKC

Before a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> our stochastic flow ma<sub>p</sub>s to FKC we first need to rewrite the cou<sub>p</sub>led d<sub>y</sub>namics in terms of the <sub>vec</sub>t<sub>or</sub> fi<sub>e</sub>ld ${ \pmb u } _ { t }$ <sub>ra</sub>th<sub>er</sub> th<sub>an score as our mo</sub>d<sub>e</sub>l<sub>s use</sub> thi<sub>s</sub> l<sub>anguage.</sub> R<sub>eca</sub>ll th<sub>a</sub>t

$$
\begin{array} { r } { \pmb { u } _ { t } ( \pmb { x } ) = \underbrace { \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \pmb { x } } _ { : = \pmb { a } _ { t } ( \pmb { x } ) } + \frac { \nu _ { t } ^ { 2 } } { 2 } \underbrace { \nabla \log p _ { t } ( \pmb { x } ) } _ { : = \pmb { s } _ { t } ( \pmb { x } ) } , } \end{array}\tag{164}
$$

see Holderrieth, Sin<sub>g</sub>er, et al. (2026, E<sub>q</sub>uation (30)). Thus we can find

$$
\boldsymbol { s } _ { t } ^ { \beta } ( \boldsymbol { \mathbf { x } } ) = \sum _ { i = 1 } ^ { M } \beta _ { i } \boldsymbol { s } _ { t } ^ { i } ( \boldsymbol { \mathbf { x } } ) ,\tag{165}
$$

$$
{ \pmb u } _ { t } ^ { \beta } ( { \pmb x } ) - { \pmb a } _ { t } ( { \pmb x } ) = \sum _ { i = 1 } ^ { M } \beta _ { i } \left( { \pmb u } _ { t } ^ { i } ( { \pmb x } ) - { \pmb a } _ { t } ( { \pmb x } ) \right) ,\tag{166}
$$

$$
\pmb { u } _ { t } ^ { \beta } ( \pmb { x } ) = \sum _ { i = 1 } ^ { M } \beta _ { i } \pmb { u } _ { t } ^ { i } ( \pmb { x } ) + ( 1 - B ) \pmb { a } _ { t } ( \pmb { x } ) .\tag{167}
$$

Likewise we can sim<sub>p</sub>lif<sub>y</sub> both <sub>p</sub>arts of the d<sub>y</sub>namics of the wei<sub>g</sub>hts with

$$
\frac { \nu _ { t } ^ { 2 } } { 2 } \left\| \sum _ { i = 1 } ^ { M } \beta _ { i } s _ { t } ^ { i } ( \pmb { x } ) \right\| ^ { 2 } = \frac { \nu _ { t } ^ { 2 } } { 2 } \left\| \frac { 2 } { \nu _ { t } ^ { 2 } } \sum _ { i = 1 } ^ { M } \beta _ { i } \left( \pmb { u } _ { t } ^ { i } ( \pmb { x } ) - \pmb { a } _ { t } ( \pmb { x } ) \right) \right\| ^ { 2 } ,\tag{168}
$$

$$
= \frac { 2 } { \nu _ { t } ^ { 2 } } \left\| \sum _ { i = 1 } ^ { M } \beta _ { i } \left( { \pmb u } _ { t } ^ { i } ( { \pmb x } ) - { \pmb a } _ { t } ( { \pmb x } ) \right) \right\| ^ { 2 } ,\tag{169}
$$

<sub>a</sub>nd

$$
\frac { \nu _ { t } ^ { 2 } } { 2 } \sum _ { i = 1 } ^ { M } \beta _ { i } \| \pmb { s } _ { t } ^ { i } ( \pmb { x } ) \| ^ { 2 } = \frac { \nu _ { t } ^ { 2 } } { 2 } \sum _ { i = 1 } ^ { M } \beta _ { i } \left\| \frac { 2 } { \nu _ { t } ^ { 2 } } ( \pmb { u } _ { t } ^ { i } ( \pmb { x } ) - \pmb { a } _ { t } ( \pmb { x } ) ) \right\| ^ { 2 } ,\tag{170}
$$

$$
= \frac { 2 } { { \nu } _ { t } ^ { 2 } } \sum _ { i = 1 } ^ { M } { \beta } _ { i } \left\| \pmb { u } _ { t } ^ { i } ( \pmb { x } ) - \pmb { a } _ { t } ( \pmb { x } ) \right\| ^ { 2 }\tag{171}
$$

Then we can rewrite E<sub>q</sub>uation (162) in terms of vector fields to find

$$
\mathrm { d } X _ { t } = \left[ 2 \pmb { u } _ { t } ^ { \beta } ( X _ { t } ) - \pmb { a } _ { t } ( X _ { t } ) \right] \ \mathrm { d } t + \nu _ { t } \ \mathrm { d } W _ { t } ,\tag{172}
$$

$$
\mathrm { d } w _ { t } = \frac { 2 } { \nu _ { t } ^ { 2 } } \left[ \left\| u _ { t } ^ { \beta } ( \boldsymbol { X } _ { t } ) \right\| ^ { 2 } - \sum _ { i = 1 } ^ { M } \beta _ { i } \left\| u _ { t } ^ { i } ( \boldsymbol { X } _ { t } ) \right\| ^ { 2 } - \left( 1 - B \right) \| a _ { t } ( \boldsymbol { X } _ { t } ) \| ^ { 2 } \right] \mathrm { d } t .\tag{173}
$$

This cou<sub>p</sub>led s<sub>y</sub>stem is what we want to learn our flow ma<sub>p</sub> on. Usin<sub>g</sub> the ex<sub>p</sub>onential lift <sub>p</sub>arameterization we n<sub>o</sub>w writ<sub>e</sub> th<sub>e aug</sub>m<sub>e</sub>nt<sub>e</sub>d lift <sub>ope</sub>r<sub>a</sub>t<sub>o</sub>r <sub>o</sub>n $\mathbb { R } ^ { d }$ ⊕ ℝ as

$$
\begin{array} { r } { \Phi _ { s , t } ^ { \beta , \theta } ( \pmb { x } , \pmb { w } ; T _ { s , t } ) = \left[ \frac { \alpha _ { s } } { \alpha _ { t } } \pmb { x } + \frac { 2 \mathcal { A } _ { s , t } } { \alpha _ { t } } \pmb { u } _ { s , t } ^ { \beta , \theta } ( \pmb { x } ; T _ { s , t } ) + \frac { 1 } { \alpha _ { t } } ( \pmb { M } _ { t } ^ { \alpha } - \pmb { M } _ { s } ^ { \alpha } ) \right] . } \\ { \pmb { w } + ( t - s ) g _ { s , t } ^ { \beta , \theta } ( \pmb { x } ; T _ { s , t } ) ~ } \end{array}\tag{174}
$$

## F Related work

In this section we discuss related work in <sub>g</sub>reater detail than our initial surve<sub>y</sub> in Section 2 <sub>p</sub>rovided.

## F.1 Stochastic flow maps

Recent work b<sub>y</sub> Ki<sub>y</sub>ohara et al. (2025) <sub>p</sub>ro<sub>p</sub>osed to learn the weak stochastic flow ma<sub>p</sub> for latent neural SDEs in a simulation-based manner, and later work b Holderrieth, Chen, et al. (2026), Passaro et al. (2026), and Pota tchik et al. (2026) all concurrentl<sub>y</sub> ex<sub>p</sub>lored learnin<sub>g</sub> these weak stochastic flow ma<sub>p</sub>s for the SDE in E<sub>q</sub>uation (1) in a simulation-free manner. This is accom<sub>p</sub>lished b<sub>y</sub> usin<sub>g</sub> a GLASS flow transformation (Holderrieth, Sin<sub>g</sub>er, et al. 2026) into an ODE and then learnin a standard deterministic flow ma on that ODE. $I . e .$ <sub>,</sub> th<sub>e</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> fl<sub>ow map</sub> th<sub>en</sub> l<sub>earns</sub> t<sub>o ma</sub>t<sub>c</sub>h th<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> k<sub>erne</sub>l $p _ { t \mid s }$ o<sup>f</sup> t<sup>h</sup>e SDE, t<sup>h</sup>ere<sup>b</sup>y <sup>l</sup>earning t<sup>h</sup>e weak so<sup>l</sup>ution. Ot<sup>h</sup>er a<sub>pp</sub>roaches to learn this transition kernel include Chen et al. (2025)—but this method still re<sub>q</sub>uires ODE/SDE inte<sub>g</sub>ration—and Bortoli et al. (2025) who train a weak stochastic flow ma<sub>p</sub> via scorin<sub>g</sub> rules and not the standard fl<sub>o</sub>w m<sub>ap</sub> l<sub>osses e</sub>n<sub>u</sub>m<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d in A<sub>ppe</sub>ndix A<sub>.</sub>2<sub>.</sub> N<sub>e</sub>xt w<sub>e</sub> will <sub>e</sub>n<sub>u</sub>m<sub>e</sub>r<sub>a</sub>t<sub>e</sub> th<sub>e</sub> k<sub>ey</sub> dif<sub>e</sub>r<sub>e</sub>n<sub>ces</sub> b<sub>e</sub>tw<sub>ee</sub>n <sub>ou</sub>r w<sub>o</sub>rk <sub>a</sub>nd <sub>p</sub>rior work on stron<sub>g</sub> stochastic flow ma<sub>p</sub>s b<sub>y</sub> McCallum et al. (2026) and Pan et al. (2026).

Comparison to McCallum et al. (2026). This work proposed to learn strong stochastic flow maps via the Euler– M<sub>a</sub>r<sub>uya</sub>m<sub>a pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>riz<sub>a</sub>ti<sub>o</sub>n with <sub>a</sub> l<sub>ea</sub>rn<sub>e</sub>d dif<sub>us</sub>i<sub>o</sub>n h<sub>ea</sub>d<sub>.</sub> T<sub>o</sub> tr<sub>a</sub>in th<sub>e</sub> n<sub>e</sub>tw<sub>o</sub>rk th<sub>ey use</sub>d <sub>a s</sub>m<sub>a</sub>ll E<sub>u</sub>l<sub>e</sub>r– M<sub>a</sub>r<sub>uya</sub>m<sub>a s</sub>t<sub>ep</sub> r<sub>a</sub>th<sub>e</sub>r th<sub>a</sub>n m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> th<sub>e</sub> drift <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt dir<sub>ec</sub>tl<sub>y.</sub> Th<sub>ey use</sub>d th<sub>e se</sub>mi<sub>g</sub>r<sub>oup</sub> l<sub>oss</sub> t<sub>o</sub> di<sub>s</sub>till th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l f<sub>or</sub> <sub>s</sub>t<sub>eps</sub> $s < t .$ <sub>.</sub> Th<sub>ey</sub> <sub>use</sub>d th<sub>e</sub> <sub>s</sub>hift<sub>e</sub>d L<sub>ege</sub>ndr<sub>e</sub> <sub>po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l <sub>e</sub>x<sub>pa</sub>n<sub>s</sub>i<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th<sub>,</sub> whi<sub>c</sub>h w<sub>e</sub> found was better than the Karhunen–Loève re<sub>p</sub>resentation in our ex<sub>p</sub>eriments.

Comparison to Pan et al. (2026). This work proposed to learn the strong stochastic flow map via the Euler– Maru<sub>y</sub>ama <sub>p</sub>arameterization with ex<sub>p</sub>licit numeric <sub>q</sub>uadrature of the Itô inte<sub>g</sub>ral. To train the network the<sub>y</sub> matched the drift coeficient directl<sub>y</sub>. The<sub>y</sub> used the La<sub>g</sub>ran<sub>g</sub>ian condition <sub>p</sub>rimaril<sub>y</sub> to distill out the SDE usin<sub>g</sub> a form similar to our Doss–Sussmann transform discussed in the main <sub>p</sub>a<sub>p</sub>er. The<sub>y</sub> used the Karhunen–Loève expansion plus a single Brownian increment over [�,�] to encode the Brownian path.

## F.1.1 Weak vs strong stochastic flow maps

T<sub>o</sub> f<sub>u</sub>rth<sub>e</sub>r <sub>e</sub>l<sub>uc</sub>id<sub>a</sub>t<sub>e</sub> th<sub>e</sub> di<sub>s</sub>tin<sub>c</sub>ti<sub>o</sub>n b<sub>e</sub>tw<sub>ee</sub>n w<sub>ea</sub>k <sub>a</sub>nd <sub>s</sub>tr<sub>o</sub>n<sub>g s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>aps a</sub>t <sub>a</sub> t<sub>ec</sub>hni<sub>ca</sub>l l<sub>e</sub>v<sub>e</sub>l w<sub>e</sub> introduce the classic notion of weak and stron<sub>g</sub> solutions to SDEs followin<sub>g</sub> Øksendal (2003, Cha<sub>p</sub>ter 5.3). Let $\pmb { b } : \mathbb { R } \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ <sub>a</sub>nd $\pmb { g } : \mathbb { R } \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d _ { w } }$ <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e s</sub>t<sub>a</sub>nd<sub>a</sub>rd <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l Li<sub>psc</sub>hitz <sub>a</sub>nd lin<sub>ea</sub>r <sub>g</sub>r<sub>o</sub>wth <sub>assu</sub>m<sub>p</sub>ti<sub>o</sub>n<sub>s,</sub><sup>10</sup> th<sub>e</sub>n w<sub>e</sub> d<sub>e</sub>fin<sub>e</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>o</sub>win<sub>g</sub> Itô SDE<sub>:</sub>

$$
\mathrm { d } X _ { t } = b ( t , X _ { t } ) \mathrm { ~ d } t + \pmb { g } ( t , X _ { t } ) \mathrm { ~ d } W _ { t } ,\tag{175}
$$

<sub>w</sub>h<sub>ere</sub> ${ \mathbf { } } W _ { t }$ i<sub>s so</sub>m<sub>e</sub> $d _ { w }$ -dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>a</sub>l Br<sub>o</sub>wni<sub>a</sub>n m<sub>o</sub>ti<sub>o</sub>n<sub>.</sub> L<sub>e</sub>t $( \Omega , { \mathcal { F } } , P )$ be some <sub>p</sub>robabilit<sub>y</sub> s<sub>p</sub>ace. Now <sub>g</sub>iven ${ \mathbf { } } W _ { t }$ <sub>a</sub>nd th<sub>e</sub> filt<sub>ra</sub>ti<sub>on</sub> $\{ \mathcal { F } _ { t } ^ { W , X _ { 0 } } \} _ { t \geq 0 }$ generated by � and $X _ { 0 }$ (suitabl<sub>y</sub> au<sub>g</sub>mented), we sa<sub>y</sub> that an $\dot { \mathcal { F } } _ { t } ^ { W , X _ { 0 } }$ -a<sup>d</sup>a<sub>p</sub>te<sup>d</sup> <sub>p</sub>rocess $X _ { t }$ is a strong solution to Equation $\left( 1 7 5 \right)$ with initi<sub>a</sub>l <sub>co</sub>nditi<sub>o</sub>n $\mathbf { \delta X } _ { 0 } = \mathbf { \delta x } _ { 0 }$ if

$$
X _ { t } = X _ { 0 } + \int _ { 0 } ^ { t } \mathrm { d } r ~ \pmb { b } ( r , \pmb { X } _ { r } ) + \int _ { 0 } ^ { t } \mathrm { d } { \pmb { W } } _ { r } ~ \pmb { g } ( r , \pmb { X } _ { r } ) ,\tag{176}
$$

h<sub>o</sub>ld<sub>s a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y</sub> f<sub>or a</sub>ll $t \geq 0$ . We sa<sub>y</sub> that a stochastic <sub>p</sub>rocess $( X _ { t } , \mathcal { F } _ { t } )$ on a <sub>p</sub>robabilit<sub>y</sub> s<sub>p</sub>ace $( \Omega , { \mathcal { F } } , P )$ is a weak solution i<sup>f</sup> t<sup>h</sup>ere exists a Brownian motion $\{ W _ { t } \} _ { t \ge 0 }$ on $( \Omega , { \mathcal { F } } , P )$ <sub>suc</sub>h th<sub>a</sub>t $\{ \mathcal { F } _ { t } \} _ { t \ge 0 }$ i<sub>s a</sub>n <sub>a</sub>dmi<sub>ss</sub>ibl<sub>e</sub> filt<sub>ra</sub>ti<sub>on,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s some measure</sub> $\mu$ <sub>suc</sub>h th<sub>a</sub>t $P ( X _ { 0 } \in A ) = \mu ( A )$ f<sub>or a</sub>ll B<sub>ore</sub>l <sub>se</sub>t<sub>s</sub> $A \subseteq \mathbb { R } ^ { d }$ <sub>,</sub> <sub>a</sub>nd $X _ { t }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> E<sub>q</sub>uation $\left( 1 7 6 \right)$

Remark F.1. The diference between the weak and strong solution is that in the strong solution we are prescribed <sub>a</sub> <sub>pa</sub>rti<sub>cu</sub>l<sub>a</sub>r Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th<sub>,</sub> wh<sub>e</sub>r<sub>eas</sub> in th<sub>e</sub> w<sub>ea</sub>k <sub>case</sub> w<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t <sub>a</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th t<sub>o</sub> whi<sub>c</sub>h th<sub>e</sub> <sub>so</sub>l<sub>u</sub>ti<sub>o</sub>n <sub>ca</sub>n b<sub>e</sub> <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d<sub>.</sub>

Within the realm of <sub>g</sub>enerative modelin<sub>g</sub> our SDEs of interest admit stron<sub>g</sub> solutions<sub>,</sub> hence the interest in creating strong <sup>fl</sup>ow maps.

Weak stochastic flow maps. We now formally define weak stochastic flow maps. Let $\{ X _ { t } \} _ { t \in [ 0 , 1 ] }$ b<sub>e a s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> p<sup>r</sup>ocess o<sup>n</sup> $\mathbb { R } ^ { d }$ <sub>g</sub>overned b<sub>y</sub> a standard additive-noise Itô SDE and let $( \Omega , { \mathcal { F } } , \mathbb { F } , P )$ b<sub>e a</sub> filt<sub>e</sub>r<sub>e</sub>d <sub>p</sub>r<sub>o</sub>b<sub>a</sub>bilit<sub>y space</sub> where � is the natural filtration of F w.r.t. �<sub>�</sub>. We define a weak flow map (Holderrieth, Chen, et al. 2026; Passaro et al. 2026; Pota<sub>p</sub>tchik et al. 2026) as a <sub>p</sub>arameterized stochastic flow ma<sub>p</sub> $\Psi _ { s , t } ^ { \theta } : \mathbb { R } ^ { 2 } \times \Omega  C ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ , viz., a tw<sub>o</sub>-tim<sub>e</sub> ind<sub>e</sub>x<sub>e</sub>d r<sub>a</sub>nd<sub>o</sub>m v<sub>a</sub>ri<sub>a</sub>bl<sub>e.</sub> W<sub>e say</sub> thi<sub>s</sub> m<sub>ap</sub> i<sub>s a</sub> w<sub>ea</sub>k <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w m<sub>ap</sub> if it <sub>co</sub>nv<sub>e</sub>r<sub>ges</sub> w<sub>ea</sub>kl<sub>y</sub> t<sub>o</sub> th<sub>e</sub> <sup>tr</sup>ue <sup>m</sup>ap as $\theta  \theta ^ { * } , i . e .$ ,

$$
\int _ { \mathbb { R } ^ { d } } \mathrm { d } ( \Psi _ { s , t } ^ { \theta } ( X _ { s } ) ) _ { \sharp } P ~ \phi \to \int _ { \mathbb { R } ^ { d } } \mathrm { d } ( X _ { t \sharp } P ) ~ \phi ,\tag{177}
$$

f<sub>o</sub>r <sub>a</sub>ll b<sub>ou</sub>nd<sub>e</sub>d Li<sub>psc</sub>hitz t<sub>es</sub>t f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n<sub>s</sub> $\phi \in \mathrm { B L } ( \mathbb { R } ^ { d } ; \mathbb { R } )$ <sub>an</sub>d <sub>a</sub>ll $s , t \in [ 0 , 1 ] , s \leq t ,$ <sub>, a</sub>nd m<sub>o</sub>r<sub>eo</sub>v<sub>e</sub>r <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>ll<sub>y</sub> on $X _ { s } = x$ f<sub>or</sub> $p _ { s } \mathbf { - a . e . } \mathbf { \lambda } \mathbf { x } , i . e .$ <sub>,</sub> th<sub>e</sub> l<sub>aw o</sub>f $\Psi _ { s , t } ^ { \theta } ( { \pmb x } )$ <sub>converges wea</sub>kl<sub>y</sub> t<sub>o</sub> th<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> k<sub>erne</sub>l $p _ { t | s } ( \cdot \mid x ) .$ 11

Strong stochastic flow maps. Conversely, the strong stochastic flow maps defined in McCallum et al. $\left( 2 0 2 6 \right) \left( c f \right.$ Itô ma<sub>p</sub>s Pan et al. 2026) are similarl<sub>y</sub> denoted b<sub>y</sub> the random variable $\Psi _ { s , t } ^ { \theta }$ <sub>;</sub> h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r<sub>,</sub> th<sub>ey</sub> in<sub>s</sub>t<sub>ea</sub>d <sub>co</sub>nv<sub>e</sub>r<sub>ge</sub> t<sub>o</sub> th<sub>e s</sub>tr<sub>o</sub>n<sub>g so</sub>l<sub>u</sub>ti<sub>o</sub>n <sub>as</sub> $\theta \to \theta ^ { * }$ ，

$$
P ( \omega \in \Omega : \operatorname* { l i m } _ { \theta  \theta ^ { * } } \Psi _ { s , t } ^ { \theta } ( \omega ) ( X _ { s } ( \omega ) ) = X _ { t } ( \omega ) ) = 1 ,\tag{178}
$$

$i . e . ,$ t<sup>h</sup>e<sub>y</sub> conver<sub>g</sub>e a<sup>l</sup>most sure<sup>l</sup><sub>y</sub>.

## F.2 Distilling path-based measurements in augmented space

In our FKC ex<sub>p</sub>eriments we learn to distill the evolution of the lo<sub>g</sub> wei<sub>g</sub>hts $\dot { w } _ { t }$ vi<sub>a</sub> <sub>a</sub>n <sub>aug</sub>m<sub>e</sub>nt<sub>e</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fl<sub>o</sub>w ma<sub>p</sub>. Recent work b<sub>y</sub> Ai et al. (2026) and OuYan<sub>g</sub> et al. (2026) has looked at a similar strate<sub>gy</sub> for deterministic flow ma<sub>p</sub>s and the continuit<sub>y</sub> e<sub>q</sub>uation. Recall that the continuit<sub>y</sub> e<sub>q</sub>uation $\partial _ { t } p _ { t } = - \left. \nabla \mid \pmb { u } _ { t } p _ { t } \right.$ im<sub>p</sub>lies that<sub>,</sub> along a trajectory $\dot { \pmb x } _ { t } = { \pmb u } _ { t } ( { \pmb x } _ { t } )$ <sub>,</sub> th<sub>e</sub> l<sub>og</sub>-d<sub>e</sub>n<sub>s</sub>it<sub>y</sub> <sub>e</sub>v<sub>o</sub>lv<sub>es</sub> <sub>as</sub> $\frac { \mathrm { d } } { \mathrm { d } t }$ <sup>l</sup>o<sub>g</sub> $p _ { t } ( { \pmb x } _ { t } ) = - \left. \nabla \mid { \pmb u } _ { t } \right. ( { \pmb x } _ { t } )$ . Ai et al. (2026) then <sub>p</sub>r<sub>opose</sub> t<sub>o</sub> l<sub>ea</sub>rn th<sub>e aug</sub>m<sub>e</sub>nt<sub>e</sub>d fl<sub>o</sub>w m<sub>ap o</sub>n $\mathbb { R } ^ { d }$ ⊕ ℝ defined as

$$
\Phi _ { s , t } ( { \pmb x } _ { s } ) = \left[ \pmb { x } _ { s } \right] + \int _ { s } ^ { t } \mathrm { d } { \boldsymbol r } \ \left[ \begin{array} { c } { { \pmb u } _ { r } ( { \pmb x } _ { r } ) } \\ { - \langle \nabla \mid { \pmb u } _ { r } ( { \pmb x } _ { r } ) \rangle } \end{array} \right] .\tag{179}
$$

This is similar to our a<sub>pp</sub>roach<sub>,</sub> exce<sub>p</sub>t that their au<sub>g</sub>mented <sub>p</sub>ath is an inte<sub>g</sub>ral and<sub>,</sub> more im<sub>p</sub>ortantl<sub>y,</sub> ever<sub>y</sub>thin<sub>g</sub> is a deterministic <sub>p</sub>rocess of bounded variation<sub>,</sub> unlike our rou<sub>g</sub>h stochastic <sub>p</sub>rocesses.

## F.3 Inference-time alignment and steering

M<sub>a</sub>n<sub>y</sub> inf<sub>e</sub>r<sub>e</sub>n<sub>ce</sub>-tim<sub>e</sub> m<sub>e</sub>th<sub>o</sub>d<sub>s e</sub>xi<sub>s</sub>t f<sub>o</sub>r fl<sub>o</sub>w/dif<sub>us</sub>i<sub>o</sub>n whi<sub>c</sub>h <sub>a</sub>im t<sub>o a</sub>li<sub>g</sub>n th<sub>e</sub> v<sub>ec</sub>t<sub>o</sub>r fi<sub>e</sub>ld d<sub>u</sub>rin<sub>g</sub> int<sub>eg</sub>r<sub>a</sub>ti<sub>o</sub>n t<sub>o s</sub>t<sub>ee</sub>r th<sub>e sa</sub>m l<sub>e</sub> t<sub>o</sub>w<sub>a</sub>rd<sub>s a a</sub>rti<sub>cu</sub>l<sub>a</sub>r r<sub>e</sub>w<sub>a</sub>rd<sub>.</sub> A <sub>o u</sub>l<sub>a</sub>r <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s</sub> t<sub>o</sub> d<sub>o</sub> r<sub>a</sub>di<sub>e</sub>nt-b<sub>ase</sub>d <sub>u</sub>id<sub>a</sub>n<sub>ce a</sub>t th<sub>e</sub> trajectory or sample level. $I . e .$ <sub>, assu</sub>m<sub>e a</sub> dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>bl<sub>e</sub> r<sub>e</sub>w<sub>a</sub>rd $r : \mathbb { R } ^ { d } \to \mathbb { R }$ d<sub>e</sub>fi<sub>ne</sub>d <sub>on</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a space,</sub> th<sub>en</sub> id<sub>ea</sub>ll<sub>y</sub> w<sub>e cou</sub>ld <sub>use</sub> th<sub>e g</sub>r<sub>a</sub>di<sub>e</sub>n $\nabla _ { x _ { t } } r ( \Phi _ { t , 1 } ( { \boldsymbol x } _ { t } ) )$ t<sub>o gu</sub>id<sub>e</sub> th<sub>e sa</sub>m<sub>p</sub>l<sub>e</sub> $\mathbf { \boldsymbol { x } } _ { t }$ at time �. However<sub>,</sub> the true flow ma<sub>p</sub> is ex<sub>p</sub>ensive to simulate. Thus several works have ex<sub>p</sub>lored usin<sub>g</sub> a sin<sub>g</sub>le Euler ste<sub>p</sub> (often from the <sub>p</sub>ers<sub>p</sub>ective of Tweedie’s formula), $i . e . , \nabla _ { { x } _ { t } } r ( { x } _ { t } + ( 1 - t ) { \pmb u } _ { t } ^ { \theta } ( { \pmb x } _ { t } ) )$ to steer the sam<sub>p</sub>le (Bansal et al. 2023; Chun<sub>g</sub> et al. 2023; Yu et al. 2023; Zhan<sub>g</sub> et al. 2025). This can be im<sub>p</sub>roved b<sub>y</sub> takin<sub>g</sub> more ste<sub>p</sub>s in the rollout (Blasin<sub>g</sub>ame and Liu 2025; Moufad et al. 2025) which eventuall<sub>y</sub> conver<sub>g</sub>es to doin<sub>g</sub> full back<sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ation throu<sub>g</sub>h the true solution ma<sub>p</sub>, $i . e . ,$ performing backpropagation through the full neural ODE/SDE rollout via the continuous adjoint equations or the full com<sub>p</sub>utational <sub>g</sub>ra<sub>p</sub>h (Ben-Hamu et al. 2024; Blasin<sub>g</sub>ame and Liu 2024; Marion et al. 2025; Nie et al. 2022; Pan et al. 2024). More recentl<sub>y</sub>, researchers have looked at doin<sub>g</sub> such <sub>g</sub>uidance with a learned flow ma<sub>p</sub> (Huan<sub>g</sub> et al. 2026; Sabour, Alber<sub>g</sub>o, et al. 2025) instead of a few-ste<sub>p</sub> rollout which has error on the order of the inte<sub>g</sub>ration scheme<sub>,</sub> $i . e . , O ( h ^ { k } )$ for a �-th order method (Blasin<sub>g</sub>ame and Liu 2025, Theorem 6.1). The main problem with this category of trajectory-based guidance is that such methods are biased and are much harder to em<sub>p</sub>lo<sub>y</sub> for non-diferentiable rewards (Pota<sub>p</sub>tchik et al. 2026).

A<sup>l</sup>ternative<sup>l</sup>y, one cou<sup>ld</sup> use exact samp<sup>l</sup>ing met<sup>h</sup>o<sup>d</sup>s suc<sup>h</sup> as sequential Monte Carlo (SMC) to target a ti<sup>l</sup>te<sup>d</sup> distribution (Skreta, Akhound-Sade<sub>g</sub>h, et al. 2025; Wu, Tri<sub>pp</sub>e, et al. 2023). However, such methods re<sub>q</sub>uire ex<sub>p</sub>ensive rollouts and a lar<sub>g</sub>e number of sam<sub>p</sub>les to avoid wei<sub>g</sub>ht de<sub>g</sub>enerac<sub>y</sub> and colla<sub>p</sub>se (Bickel et al. 2008; Sn<sub>y</sub>der et al. 2008). We refer to Uehara et al. (2025) for further overview on such techni<sub>q</sub>ues and their recent <sub>a</sub>dv<sub>a</sub>n<sub>ces.</sub>

As mentione<sup>d</sup> in an ear<sup>l</sup>ier section, weak stoc<sup>h</sup>astic <sup>fl</sup>ow maps present a promising a<sup>l</sup>ternative, ena<sup>bl</sup>ing certain Monte Carlo tec<sup>h</sup>niques (Ho<sup>ld</sup>erriet<sup>h</sup>, C<sup>h</sup>en, et a<sup>l</sup>. 2026; Passaro et a<sup>l</sup>. 2026; Potaptc<sup>h</sup>i<sup>k</sup> et a<sup>l</sup>. 2026).

## G Implementation details

Here we describe the im<sub>p</sub>lementation details for the various ex<sub>p</sub>eriments in the main <sub>p</sub>a<sub>p</sub>er.

## G.1 ImageNet-256 architecture

All ImageNet models are fine-tuned from a pre-trained DMF-XL/2+ (Lee et al. 2026) with SD-VAE (sd-vae-ft-mse) l<sub>a</sub>t<sub>en</sub>t<sub>s o</sub>f <sub>s</sub>h<sub>ape</sub> $4 \times 3 2 \times 3 2$ <sub>un</sub>d<sub>er</sub> th<sub>e</sub> li<sub>near sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub> $\alpha _ { t } = t , \sigma _ { t } = 1 - t$ <sub>.</sub> F<sub>o</sub>ll<sub>o</sub>win<sub>g</sub> th<sub>e</sub> d<sub>ecoup</sub>l<sub>e</sub>d tim<sub>e co</sub>nditionin<sub>g</sub> of Lee et al. (2026), the first 20 of the 28 transformer blocks are conditioned on � with the remainin<sub>g</sub> 8 bl<sub>oc</sub>k<sub>s</sub> <sub>o</sub>n �<sub>.</sub> Th<sub>e</sub> <sub>p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d b<sub>ac</sub>kb<sub>o</sub>n<sub>e</sub> i<sub>s</sub> fr<sub>o</sub>z<sub>e</sub>n with th<sub>e</sub> l<sub>aye</sub>r<sub>s</sub> wr<sub>appe</sub>d in L<sub>o</sub>RA <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>r<sub>s</sub> <sub>o</sub>f r<sub>a</sub>nk 16<sub>.</sub>

## G.2 ImageNet-256 training objectives

Every objective uses a fixed batch split with a fraction $p = 0 . 7 5$ tr<sub>a</sub>inin<sub>g</sub> th<sub>e o</sub>f-di<sub>ago</sub>n<sub>a</sub>l di<sub>s</sub>till<sub>a</sub>ti<sub>o</sub>n t<sub>e</sub>rm <sub>a</sub>nd the remainin<sub>g</sub> 0.25 trainin<sub>g</sub> the dia<sub>g</sub>onal term. We dro<sub>p</sub> labels to the null class with <sub>p</sub>robabilit<sub>y</sub> 0.1. Intervals are <sub>sam</sub> l<sub>e</sub>d <sub>as</sub> $t \sim \mathcal { U } [ \epsilon + \delta , 1 ]$ <sub>a</sub>nd $h = t - s \sim \mathcal { U } [ \delta$ , min $\left( h _ { \operatorname* { m a x } } , t - \epsilon \right) ]$ <sub>w</sub>ith $\epsilon = 0 . 0 1$ <sub>,</sub> minim<sub>u</sub>m int<sub>e</sub>rv<sub>a</sub>l $\delta = 0 . 0 5$ <sub>a</sub>nd $h _ { \mathrm { m a x } } = 1$ <sub>.</sub> Th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l l<sub>oss</sub> i<sub>s</sub> $p \mathcal { L } _ { \mathrm { d i s t i l l } } + ( 1 - p )$ L<sub>anchor</sub>.

Diagonal. The diagonal loss ensures the student $\pmb { u } _ { t , t } ^ { \theta }$ d<sub>oes</sub> n<sub>o</sub>t drift t<sub>oo</sub> f<sub>a</sub>r fr<sub>o</sub>m th<sub>e p</sub>r<sub>e</sub>-tr<sub>a</sub>in<sub>e</sub>d t<sub>eac</sub>h<sub>e</sub>r $\pmb { u } _ { t } ^ { \theta }$ <sub>.</sub> Th<sub>e</sub> t<sub>eac</sub>h<sub>er</sub> i<sub>s a s</sub>i<sub>ng</sub>l<sub>e</sub> f<sub>orwar</sub>d <sub>pass o</sub>f th<sub>e</sub> f<sub>rozen</sub> ${ \mathrm { D M F } } { \mathrm { - } } { \mathrm { X L } } / 2 +$ <sub>as</sub> th<sub>a</sub>t <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>int w<sub>as</sub> tr<sub>a</sub>in<sub>e</sub>d with m<sub>o</sub>d<sub>e</sub>l <sub>gu</sub>id<sub>a</sub>n<sub>ce</sub> (Tan<sub>g</sub> et al. 2025); its conditional out<sub>p</sub>ut is alread<sub>y</sub> the <sub>g</sub>uided field. For the first three rows of Table 1 (SSFM, + ex<sub>p</sub>onential lift, + closed-form martin<sub>g</sub>ale) we instead use a sin<sub>g</sub>le Euler–Maru<sub>y</sub>ama ste<sub>p</sub> of size 0.01 followin<sub>g</sub> McCallum et al. (2026), the SSFM and ex<sub>p</sub>onential-lift variants additionall<sub>y</sub> learn a randoml<sub>y</sub> initialized difusion h<sub>ea</sub>d<sub>.</sub>

Wong–Zakai and iMF. In the lifted state $y = \alpha x$ , the improved MeanFlow objective regresses the compound 2�<sub>�</sub> $\pmb { u } _ { s , t } ^ { \theta } - 2 \mathcal { R } _ { s , t } \mathbf { s } \mathbf { g } \big [ D _ { s } \pmb { u } _ { s , t } ^ { \theta } \big ]$ onto $2 \alpha _ { s } \mathrm { s g } [ { \pmb u } _ { s } ^ { \mathrm { t e a c h e r } } ]$ <sub>, w</sub>ith $\textstyle { \mathcal { A } } _ { s , t } = \int _ { s } ^ { t } \alpha _ { r }$ d<sub>�.</sub> Th<sub>e</sub> m<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l d<sub>e</sub>riv<sub>a</sub>tiv<sub>e</sub> $D _ { s } \pmb { u } _ { s , t } ^ { \theta }$ is a sin<sub>g</sub>le f<sub>o</sub>rw<sub>a</sub>rd-m<sub>o</sub>d<sub>e</sub> $\mathrm { { \cal J } V P }$ <sub>a</sub>l<sub>ong</sub> th<sub>e</sub> lift<sub>e</sub>d <sub>smoo</sub>th fi<sub>e</sub>ld $2 \alpha _ { s } u _ { s } ^ { \mathrm { t e a c h e r } } + \dot { M } _ { s } ^ { \alpha , ( N ) }$ with the sto<sub>pg</sub>rad o<sub>p</sub>eration a<sub>pp</sub>lied. Both th<sub>e</sub> di<sub>ago</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>o</sub>f-di<sub>ago</sub>n<sub>a</sub>l l<sub>osses</sub> <sub>a</sub>r<sub>e</sub> r<sub>e</sub>w<sub>e</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>pe</sub>r <sub>sa</sub>m<sub>p</sub>l<sub>e</sub> with <sub>a</sub> l<sub>ea</sub>rn<sub>e</sub>d l<sub>og</sub>-v<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub> $l _ { s , t }$ (Kendall et al. 2018), predicted from random Fourier features of (�, �) by a linear layer and applied as log $\left( e ^ { - l _ { s , t } } \ell + 1 0 ^ { - 8 } \right) + l _ { s , t }$ followin<sub>g</sub> Lee et al. (2026).

Semigroup. Each interval is split at its midpoint $m = { \textstyle \frac { 1 } { 2 } } ( s + t )$ <sub>;</sub> tw<sub>o</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>c</sub>hild <sub>pa</sub>th<sub>s a</sub>r<sub>e sa</sub>m<sub>p</sub>l<sub>e</sub>d on [�, �] and $[ m , t ]$ <sub>a</sub>nd th<sub>e pa</sub>r<sub>e</sub>nt <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s a</sub>r<sub>e co</sub>m<sub>pose</sub>d fr<sub>o</sub>m th<sub>e</sub>m with th<sub>e</sub> Ch<sub>e</sub>n r<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> Th<sub>e</sub> EMA t<sub>eac</sub>h<sub>er</sub> t<sub>a</sub>k<sub>es</sub> th<sub>e</sub> t<sub>wo</sub> h<sub>a</sub>lf<sub>-s</sub>t<sub>eps, an</sub>d th<sub>e s</sub>t<sub>u</sub>d<sub>en</sub>t’<sub>s</sub> i<sub>n</sub>t<sub>erva</sub>l d<sub>r</sub>ift i<sub>s regresse</sub>d <sub>on</sub>t<sub>o</sub> $\begin{array} { r } { \frac { \mathcal { A } _ { s , m } } { \mathcal { A } _ { s , t } } \pmb { u } _ { s , m } ^ { \hat { \theta } } + \frac { \mathcal { A } _ { m , t } } { \mathcal { A } _ { s , t } } \pmb { u } _ { m , t } ^ { \hat { \theta } } } \end{array}$ <sub>,</sub> th<sub>e exac</sub>t semigroup <sup>l</sup>aw o<sup>f</sup> t<sup>h</sup>e <sup>l</sup>i<sup>f</sup>te<sup>d</sup> map. N.B., we trie<sup>d</sup> an ar<sup>b</sup>itrary sp<sup>l</sup>it point, <sup>b</sup>ut <sup>f</sup>oun<sup>d</sup> no <sup>d</sup>i<sup>f</sup>erence empirica<sup>ll</sup>y.

Optimization. We use the AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 8 }$ , no wei<sub>g</sub>ht deca<sub>y</sub>) o<sub>p</sub>timizer with a learnin<sub>g</sub> rate $1 0 ^ { - 4 }$ , gradient clipping at 1.0, bf16-mixed precision, and an efective batch size of 128, viz. 64 × 2 gradient accumulation. We train for 100k o<sub>p</sub>timizer ste<sub>p</sub>s with an EMA of deca<sub>y</sub> 0.9999. We use the Triton attention kernel with forward-mode (dual) su<sub>pp</sub>ort for JVPs.

## G.3 ImageNet-256 evaluation

Sampling. Every model is evaluated on the same fixed set of 50k seeds with labels that are class-balanced, $i . e . ,$ 50 <sub>pe</sub>r <sub>c</sub>l<sub>ass, a</sub>nd th<sub>e</sub>n <sub>s</sub>h<sub>u</sub>fl<sub>e</sub>d with <sub>a</sub> fix<sub>e</sub>d <sub>see</sub>d<sub>.</sub> S<sub>a</sub>m<sub>p</sub>l<sub>es a</sub>r<sub>e ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d in bl<sub>oc</sub>k<sub>s o</sub>f 250 wh<sub>ose</sub> initi<sub>a</sub>l n<sub>o</sub>i<sub>se</sub> i<sub>s</sub> <sub>see</sub>d<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> bl<sub>oc</sub>k ind<sub>e</sub>x <sub>a</sub>nd wh<sub>ose</sub> r<sub>ea</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th i<sub>s</sub> <sub>see</sub>d<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> bl<sub>oc</sub>k ind<sub>e</sub>x <sub>a</sub>nd th<sub>e</sub> <sub>s</sub>t<sub>ep</sub> count.

Samplers. We evaluate on a uniform time grid over [�, 1] with $\epsilon = 0 . 0 1$ <sub>.</sub> Th<sub>e</sub> $\mathrm { S i T - X L } / 2 \mathrm { + R E P A }$ baseline inte<sub>g</sub>rates the flow as an SDE followin<sub>g</sub> E<sub>q</sub>uation (1) with an Euler–Maru<sub>y</sub>ama scheme usin<sub>g</sub> classifier-free <sub>g</sub>uidance of scale 2.0 onl<sub>y</sub> for times $t \in [ 0 , 0 . 7 ]$ (K<sub>y</sub>nkäänniemi et al. 2024).

Metrics. We measure FID (Heusel et al. 2017) via torchmetrics/torch-fidelity Inception features against the standard Ima<sub>g</sub>eNet-2<sub>5</sub>6 reference statistics. $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ (Stein et al. 2023) follows the <sub>p</sub>rotocol of Karras et al. (2024): DINOv2 ViT-L/14 CLS features (1024-d) of ima<sub>g</sub>es resized to $2 2 4 \times 2 2 4$ with antialiased bicubic inter<sub>p</sub>olation<sub>,</sub> <sub>aga</sub>in<sub>s</sub>t r<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> m<sub>o</sub>m<sub>e</sub>nt<sub>s co</sub>m<sub>pu</sub>t<sub>e</sub>d <sub>o</sub>n th<sub>e</sub> f<sub>u</sub>ll Im<sub>age</sub>N<sub>e</sub>t tr<sub>a</sub>inin<sub>g sp</sub>lit <sub>a</sub>ft<sub>e</sub>r th<sub>e s</sub>t<sub>a</sub>nd<sub>a</sub>rd $2 5 6 \times 2 5 6$ center cro<sub>p</sub>. We measure Ima<sub>g</sub>eReward (Xu et al. 2023), PickScore (Kirstain et al. 2023) and HPSv2.1 (Wu, Hao, et al.

2023) scores for each ima<sub>g</sub>e a<sub>g</sub>ainst the ca<sub>p</sub>tion $\mathrm { { ^ { 4 } A } }$ <sub>p</sub>hoto of a/an {class name}” and we re<sub>p</sub>ort the mean over the   
50k sam<sub>p</sub>les.

## G.4 Toy experiments

2D checkerboard (Figure 3a). We use an MLP flow map trained on the checkerboard with the semigroup loss under the linear schedule with a batch size of 4096. A<sub>g</sub>ain we use the AdamW o<sub>p</sub>timizer with a learnin<sub>g</sub> rate $1 0 ^ { - 3 }$ for 200 e<sub>p</sub>ochs<sub>,</sub> other h<sub>yp</sub>er<sub>p</sub>arameters match the Ima<sub>g</sub>eNet settin<sub>g</sub>. Each re<sub>p</sub>resentation is trained with its f<sub>u</sub>ll <sub>suppo</sub>rt <sub>o</sub>f 8 <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s a</sub>nd <sub>e</sub>v<sub>a</sub>l<sub>ua</sub>t<sub>e</sub>d with <sub>o</sub>nl<sub>y</sub> th<sub>e</sub> fir<sub>s</sub>t � vi<sub>s</sub>ibl<sub>e.</sub> W<sub>e</sub> m<sub>easu</sub>r<sub>e</sub> th<sub>e s</sub>tr<sub>o</sub>n<sub>g</sub> RMSE <sub>aga</sub>in<sub>s</sub>t <sub>a</sub>n ind<sub>epe</sub>nd<sub>e</sub>ntl<sub>y</sub> <sub>so</sub>lv<sub>e</sub>d 512-<sub>s</sub>t<sub>ep</sub> <sub>e</sub>x<sub>po</sub>n<sub>e</sub>nti<sub>a</sub>l-int<sub>eg</sub>r<sub>a</sub>t<sub>o</sub>r E<sub>u</sub>l<sub>e</sub>r <sub>so</sub>lv<sub>e</sub> <sub>o</sub>f th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l’<sub>s</sub> <sub>o</sub>wn l<sub>ea</sub>rn<sub>e</sub>d di<sub>ago</sub>n<sub>a</sub>l SDE <sub>o</sub>n th<sub>e sa</sub>m<sub>e</sub> Br<sub>o</sub>wni<sub>a</sub>n r<sub>ea</sub>liz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> Lik<sub>e</sub>wi<sub>se,</sub> th<sub>e</sub> 2-W<sub>asse</sub>r<sub>s</sub>t<sub>e</sub>in di<sub>s</sub>t<sub>a</sub>n<sub>ce uses</sub> 4096 <sub>sa</sub>m<sub>p</sub>l<sub>es.</sub>

1D non-linear SDE (Figure 3b). Following McCallum et al. (2026), the SDE is ${ \mathrm { d } } X _ { t } = ( X _ { t } - X _ { t } ^ { 3 } ) { \mathrm { d } } t + { \sqrt { \beta _ { t } } }$ d�<sub>�</sub> <sub>w</sub>ith $\beta _ { t }$ lin<sub>ea</sub>r fr<sub>o</sub>m 20 <sub>a</sub>t $t = 0$ to 0.1 at $t = 1$ <sub>a</sub>nd $X _ { 0 } \sim { \cal N } ( 0 , 1 )$ <sub>.</sub> E<sub>ac</sub>h <sub>s</sub>t<sub>a</sub>rt <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $X _ { s }$ i<sub>s s</sub>im<sub>u</sub>l<sub>a</sub>t<sub>e</sub>d with 128 fin<sub>e</sub> Euler–Maru<sub>y</sub>ama sub-ste<sub>p</sub>s over $[ 0 , s ]$ <sub>.</sub> All l<sub>osses</sub> <sub>s</sub>h<sub>are</sub> th<sub>e</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm-</sub>i<sub>ncremen</sub>t <sub>map</sub> $\pmb { x } + \left( t - s \right) \pmb { u } _ { s , t } ^ { \theta } + \left( \pmb { M } _ { t } - \pmb { M } _ { s } \right)$ with an MLP and $N \leq 4$ <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>s,</sub> tr<sub>a</sub>in<sub>e</sub>d f<sub>o</sub>r 5k <sub>s</sub>t<sub>eps a</sub>t b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>iz<sub>e</sub> 4096 with <sub>a</sub>n EMA t<sub>eac</sub>h<sub>e</sub>r <sub>o</sub>f d<sub>ecay</sub> 0<sub>.</sub>999<sub>.</sub> Stron<sub>g</sub> error is the mean s<sub>q</sub>uared end<sub>p</sub>oint error a<sub>g</sub>ainst a 512-ste<sub>p</sub> Euler–Maru<sub>y</sub>ama reference on 1024 shared initi<sub>a</sub>l <sub>co</sub>nditi<sub>o</sub>n<sub>s a</sub>nd Br<sub>o</sub>wni<sub>a</sub>n <sub>pa</sub>th<sub>s.</sub>

Feynman–Kac correctors (Figure 6). The target is the annealed $4 0 \cdot$ -G<sub>auss</sub>i<sub>a</sub>n mixt<sub>u</sub>r<sub>e o</sub>f Skr<sub>e</sub>t<sub>a,</sub> Akh<sub>ou</sub>nd-S<sub>a</sub>d<sub>eg</sub>h<sub>,</sub> et al. (2025). The au<sub>g</sub>mented ma<sub>p</sub> (Section 4.3) is an MLP that <sub>p</sub>redicts the scaled score and a bounded wei<sub>g</sub>ht inte<sub>g</sub>rand $\psi = g / \nu ^ { 2 }$ <sub>,</sub> which is inte<sub>g</sub>rated in the wei<sub>g</sub>hted time variable $\int \mathrm { d } \boldsymbol { r } ~ \nu _ { r } ^ { 2 }$ <sub>so</sub> th<sub>a</sub>t th<sub>e</sub> E<sub>u</sub>l<sub>er</sub>i<sub>an correc</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ays</sub> finit<sub>e</sub> <sub>as</sub> $s \to 0$ . It is trained with the Wong–Zakai iMF objective against the analytic composed field and <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd <sub>se</sub>ttin<sub>gs.</sub> Th<sub>e</sub> dif<sub>us</sub>i<sub>o</sub>n b<sub>ase</sub>lin<sub>e</sub> i<sub>s a sco</sub>r<sub>e</sub> n<sub>e</sub>tw<sub>o</sub>rk <sub>o</sub>f th<sub>e sa</sub>m<sub>e s</sub>iz<sub>e</sub> tr<sub>a</sub>in<sub>e</sub>d with th<sub>e sa</sub>m<sub>e sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub> an<sup>d</sup> <sup>b</sup>u<sup>d</sup>get on t<sup>h</sup>e untempere<sup>d</sup> target, viz. $\beta = 1$ . At inference<sub>,</sub> we s<sub>p</sub>lit 10k <sub>p</sub>articles into inde<sub>p</sub>endent <sub>g</sub>rou<sub>p</sub>s of � that are s<sub>y</sub>stematicall<sub>y</sub> resam<sub>p</sub>led amon<sub>g</sub> themselves after ever<sub>y</sub> ste<sub>p</sub>. Before this resam<sub>p</sub>lin<sub>g</sub> we cli<sub>p</sub> the l<sub>og</sub>-w<sub>e</sub>i<sub>g</sub>ht<sub>s a</sub>t <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>qua</sub>ntil<sub>es</sub> r<sub>e</sub>m<sub>o</sub>vin<sub>g a</sub> t<sub>o</sub>t<sub>a</sub>l fr<sub>ac</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> t<sub>a</sub>il<sub>s</sub>—2<sub>5</sub>% in th<sub>e</sub> m<sub>a</sub>in t<sub>e</sub>xt<sub>; see</sub> T<sub>a</sub>bl<sub>es</sub> 6 t<sub>o</sub> 11 f<sub>o</sub>r other rates. A run is marked as diver<sub>g</sub>ed (∞) if an<sub>y p</sub>article is non-finite or leaves the box $\| \pmb { x } \| _ { \infty } \leq 1 0 ^ { 3 }$

FKC metrics. We follow the sampling metrics of Skreta, Akhound-Sadegh, et al. (2025, Appendix F.1). Each <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d <sub>c</sub>l<sub>ou</sub>d $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ <sub>o</sub>f $n = 1 0 \mathrm { k \Omega }$ <sub>p</sub>articles is com<sub>p</sub>ared a<sub>g</sub>ainst � exact sam<sub>p</sub>les $\{ y _ { j } \}$ f<sub>rom</sub> th<sub>e</sub> t<sub>arge</sub>t $\pi _ { 1 }$ <sub>,</sub> dr<sub>a</sub>wn with <sub>a see</sub>d ind<sub>epe</sub>nd<sub>e</sub>nt <sub>o</sub>f th<sub>e sa</sub>m<sub>p</sub>l<sub>e</sub>r<sub>;</sub> b<sub>o</sub>th <sub>c</sub>l<sub>ou</sub>d<sub>s a</sub>r<sub>e e</sub>x<sub>p</sub>r<sub>esse</sub>d in th<sub>e u</sub>nn<sub>o</sub>rm<sub>a</sub>liz<sub>e</sub>d <sub>coo</sub>rdin<sub>a</sub>t<sub>es.</sub>

$\mathbf { E n e r g y } – \mathcal { W } _ { 2 }$ i<sub>s</sub> th<sub>e e</sub>x<sub>ac</sub>t 1D 2-W<sub>asse</sub>r<sub>s</sub>t<sub>e</sub>in di<sub>s</sub>t<sub>a</sub>n<sub>ce</sub> b<sub>e</sub>tw<sub>ee</sub>n $\{ E ( { \pmb x } _ { i } ) \}$ <sub>a</sub>nd $\{ E ( \boldsymbol { y } _ { j } ) \}$ <sub>w</sub>h<sub>ere</sub> $E ( \pmb { x } ) = - \log \pi _ { 1 } ( \pmb { x } )$

• MMD is the biased estimate of the squared maximum mean discrepancy.

• TV is the total variation distance.

$\mathcal { W } _ { 1 }$ and $\mathcal { W } _ { 2 }$ <sub>a</sub>r<sub>e</sub> th<sub>e</sub> E<sub>uc</sub>lid<sub>ea</sub>n W<sub>asse</sub>r<sub>s</sub>t<sub>e</sub>in di<sub>s</sub>t<sub>a</sub>n<sub>ces</sub> b<sub>e</sub>tw<sub>ee</sub>n th<sub>e c</sub>l<sub>ou</sub>d<sub>s.</sub>

W<sub>e</sub> r<sub>epo</sub>rt m<sub>ea</sub>n<sub>s a</sub>nd <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd d<sub>e</sub>vi<sub>a</sub>ti<sub>o</sub>n<sub>s o</sub>v<sub>e</sub>r 4 <sub>see</sub>d<sub>s.</sub>

## G.5 Hardware

Each Ima<sub>g</sub>eNet trainin<sub>g</sub> run used a sin<sub>g</sub>le NVIDIA H100 NVL (94 GB) GPU.

## H Additional results

H<sub>e</sub>r<sub>e</sub> w<sub>e</sub> <sub>p</sub>r<sub>o</sub>vid<sub>e</sub> <sub>a</sub>dditi<sub>o</sub>n<sub>a</sub>l r<sub>esu</sub>lt<sub>s</sub> whi<sub>c</sub>h did n<sub>o</sub>t fit in th<sub>e</sub> m<sub>a</sub>in <sub>pape</sub>r<sub>.</sub>

![](images/2f7fd07d55fcf78f47e01943ee18bcce12010ed330f00015aac50fed9ad5e72b.jpg)  
Figure 13 Comparing Brownian representations on the 2D checkerboard problem.

## H.1 Toy experiments

We <sub>p</sub>rovide additional results on our checkerboard swee<sub>p</sub>s in Fi<sub>g</sub>ure 1<sub>3</sub>.

I<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> <sub>4</sub> <sub>we</sub> <sub>presen</sub>t th<sub>e</sub> f<sub>u</sub>ll <sub>resu</sub>lt<sub>s</sub> f<sub>or</sub> th<sub>e</sub> t<sub>oy</sub> <sub>1</sub>D <sub>pro</sub>bl<sub>em</sub> f<sub>rom</sub> th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>paper.</sub>

Table 4 Full strong error grid behind Figure 3b: $\mathbb { E } \| X _ { 1 } - X _ { 1 } ^ { \theta } \| ^ { 2 }$ a<sub>g</sub>ainst a <sub>5</sub>12-ste<sub>p</sub> Euler–Maru<sub>y</sub>ama reference on 102<sub>4</sub> shared initial conditions and Brownian <sub>p</sub>aths (eval seed 12345). All losses share the closed-form-increment ma<sub>p</sub>. Best <sub>p</sub>er column in bold.
<table><tr><td colspan="2"></td><td colspan="6">NFE</td></tr><tr><td>Loss</td><td>N</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td rowspan="3">WZ AYF-EMD</td><td>1</td><td>0.529</td><td>0.239</td><td>0.146</td><td>0.0762</td><td>0.0349</td><td>0.0289</td></tr><tr><td>2</td><td>0.174</td><td>0.198</td><td>0.219</td><td>42.1</td><td>0.44</td><td>0.46</td></tr><tr><td>3</td><td>0.175</td><td>0.197</td><td>0.311</td><td>0.443</td><td>0.38</td><td>0.34</td></tr><tr><td rowspan="5">WZ iMF</td><td>4</td><td>0.141</td><td>0.13</td><td>0.153</td><td>0.165</td><td>0.18</td><td>0.177</td></tr><tr><td>1</td><td>0.65</td><td>0.229</td><td>0.0993</td><td>0.0408</td><td>0.0139</td><td>0.00659</td></tr><tr><td>2</td><td>0.207</td><td>0.0652</td><td>0.0331</td><td>0.025</td><td>0.0447</td><td>0.0455</td></tr><tr><td>3</td><td>0.09</td><td>0.0466</td><td>0.0362</td><td>0.0256</td><td>0.0201</td><td>0.0191</td></tr><tr><td>4</td><td>0.0763</td><td>0.0463</td><td>0.0365</td><td>0.0279</td><td>0.0407</td><td>0.0285</td></tr><tr><td rowspan="4">WZ AYF-LMD</td><td>1</td><td>12.6</td><td>2.35</td><td>0.517</td><td>0.232</td><td>0.122</td><td>0.109</td></tr><tr><td>2</td><td>30.5</td><td>1.14</td><td>0.638</td><td>0.575</td><td>0.56</td><td>0.581</td></tr><tr><td>3</td><td>2.66</td><td>28.2</td><td> $1 . 6 6 \times 1 0 ^ { 3 }$ </td><td>4.96</td><td>0.211</td><td>0.178</td></tr><tr><td>4</td><td>51</td><td>1.78</td><td>0.637</td><td>0.693</td><td>0.709</td><td>0.717</td></tr><tr><td rowspan="4">DS AYF-EMD</td><td>1</td><td>6.26</td><td>1.25</td><td>0.249</td><td>0.122</td><td>0.0401</td><td>0.00962</td></tr><tr><td>2</td><td>6.32</td><td>0.936</td><td>0.209</td><td>0.0964</td><td>0.0298</td><td>0.00745</td></tr><tr><td>3</td><td>6.29</td><td>0.999</td><td>0.258</td><td>0.124</td><td>0.034</td><td>0.0076</td></tr><tr><td>4</td><td>6.28</td><td>0.94</td><td>0.267</td><td>0.114</td><td>0.0312</td><td>0.00703</td></tr><tr><td rowspan="4">DS iMF</td><td>1</td><td>6.56</td><td>1.07</td><td>0.201</td><td>0.0466</td><td>0.00982</td><td>0.00224</td></tr><tr><td>2</td><td>6.51</td><td>1.08</td><td>0.202</td><td>0.0476</td><td>0.00984</td><td>0.00293</td></tr><tr><td>3</td><td>6.53</td><td>1.05</td><td>0.197</td><td>0.0447</td><td>0.00976</td><td>0.00229</td></tr><tr><td>4</td><td>6.53</td><td>1.11</td><td>0.201</td><td>0.0523</td><td>0.0111</td><td>0.0027</td></tr><tr><td rowspan="4">DS AYF-LMD</td><td>1</td><td>38</td><td>0.957</td><td>0.298</td><td>0.321</td><td>0.136</td><td>0.0998</td></tr><tr><td>2</td><td>37.4</td><td>1.25</td><td>0.746</td><td>1.56</td><td>0.597</td><td>0.595</td></tr><tr><td>3</td><td>19.8</td><td>1.46</td><td>0.573</td><td>16.1</td><td>0.37</td><td>0.376</td></tr><tr><td>4</td><td>2.8</td><td>3.76</td><td>517</td><td> $1 . 4 1 \times 1 0 ^ { 5 }$ </td><td>0.657</td><td>0.546</td></tr><tr><td rowspan="4">Semigroup</td><td>1</td><td>0.513</td><td>0.561</td><td>0.286</td><td>0.174</td><td>0.146</td><td>0.161</td></tr><tr><td>2</td><td>1.62</td><td>0.618</td><td>0.338</td><td>0.172</td><td>0.0556</td><td>0.0131</td></tr><tr><td>3</td><td>1.28</td><td>0.585</td><td>0.311</td><td>0.131</td><td>0.0413</td><td>0.0117</td></tr><tr><td>4</td><td>2.44</td><td>3.42</td><td>289</td><td> $6 . 7 \times 1 0 ^ { 4 }$ </td><td>0.139</td><td>0.0867</td></tr><tr><td rowspan="4">dESD</td><td>1</td><td>0.625</td><td>0.216</td><td>0.0656</td><td>0.0285</td><td>0.0126</td><td>0.0055</td></tr><tr><td>2</td><td>0.176</td><td>0.06</td><td>0.0345</td><td>0.0195</td><td>0.0127</td><td>0.0079</td></tr><tr><td>3</td><td>0.159</td><td>0.0505</td><td>0.022</td><td>0.0137</td><td>0.0102</td><td>0.00677</td></tr><tr><td>4</td><td>0.161</td><td>0.0484</td><td>0.0267</td><td>0.0171</td><td>0.0118</td><td>0.00857</td></tr><tr><td rowspan="5">dLSD</td><td>1</td><td>0.54</td><td>0.169</td><td>0.0662</td><td>0.0314</td><td>0.0135</td><td>0.00586</td></tr><tr><td>2</td><td>0.261</td><td>0.0586</td><td>0.0298</td><td>0.015</td><td>0.00979</td><td>0.00765</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td>0.152</td><td>0.0497</td><td>0.0234</td><td>0.0151</td><td>0.0102</td><td>0.00798</td></tr><tr><td>4</td><td>0.248</td><td>0.0578</td><td>0.0217</td><td>0.0169</td><td>0.0165</td><td>0.0148</td></tr></table>

## H.2 ImageNet

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on we presen</sub>t th<sub>e</sub> f<sub>u</sub>ll <sub>resu</sub>lt<sub>s</sub> f<sub>rom our a</sub>bl<sub>a</sub>ti<sub>on s</sub>t<sub>u</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e 5.</sub> All <sub>me</sub>t<sub>r</sub>i<sub>cs use 50</sub>k <sub>c</sub>l<sub>ass-</sub>b<sub>a</sub>l<sub>ance</sub>d <sub>samp</sub>l<sub>es w</sub>ith <sub>s</sub>h<sub>are</sub>d l<sub>a</sub>b<sub>e</sub>l<sub>s an</sub>d <sub>see</sub>d<sub>s an</sub>d <sub>a</sub>ll <sub>runs are</sub> fi<sub>ne-</sub>t<sub>une</sub>d f<sub>or 100</sub>k <sub>s</sub>t<sub>eps</sub> f<sub>rom</sub> th<sub>e same pre-</sub>t<sub>ra</sub>i<sub>ne</sub>d DMF-XL/2+.

Table 5 ImageNet-256 results and reference SiT-XL/2+REPA teacher.
<table><tr><td>Model</td><td></td><td>N NFE</td><td>FID (↓)</td><td>FDDINOv2 (↓)</td><td>ImageReward (↑)</td><td>PickScore (↑)</td><td>HPSv2.1 (↑)</td></tr><tr><td rowspan="5">SSFM (McCallum et al. 2026)</td><td>3</td><td>1</td><td>197.11</td><td>2116.9</td><td>-2.214</td><td>18.13</td><td>0.1506</td></tr><tr><td>3</td><td>2</td><td>83.56</td><td>1061.3</td><td>-1.532</td><td>18.69</td><td>0.1806</td></tr><tr><td>3</td><td>4</td><td>31.00</td><td>475.1</td><td>-0.872</td><td>19.33</td><td>0.2031</td></tr><tr><td>3</td><td>8</td><td>11.09</td><td>229.1</td><td>-0.493</td><td>19.78</td><td>0.2177</td></tr><tr><td>3</td><td>16</td><td>4.82</td><td>135.6</td><td>-0.312</td><td>20.04</td><td>0.2253</td></tr><tr><td></td><td>3</td><td>1</td><td>208.25</td><td>2112.2</td><td>-2.246</td><td>18.21</td><td>0.1470</td></tr><tr><td rowspan="5">SSFM + exponential lift</td><td>3</td><td>2</td><td>90.16</td><td>1113.1</td><td>-1.635</td><td>18.72</td><td>0.1746</td></tr><tr><td>3</td><td>4</td><td>28.15</td><td>447.4</td><td>-0.867</td><td>19.44</td><td>0.2015</td></tr><tr><td>3</td><td>8</td><td>8.07</td><td>190.8</td><td>-0.454</td><td>19.92</td><td>0.2187</td></tr><tr><td>3</td><td>16</td><td>3.16</td><td>104.0</td><td>-0.249</td><td>20.22</td><td>0.2292</td></tr><tr><td>3</td><td>1</td><td>163.56</td><td>1561.2</td><td>-2.144</td><td>18.62</td><td>0.1492</td></tr><tr><td rowspan="5">SSFM + exponential lift + closed-form martingale</td><td>3</td><td>2</td><td>60.21</td><td>723.0</td><td>-1.372</td><td>19.20</td><td>0.1759</td></tr><tr><td>3</td><td>4</td><td>18.33</td><td>301.6</td><td>-0.790</td><td>19.78</td><td>0.1961</td></tr><tr><td>3</td><td>8</td><td>4.95</td><td>125.7</td><td>-0.427</td><td>20.22</td><td>0.2138</td></tr><tr><td>3</td><td>16</td><td>2.98</td><td>72.6</td><td>-0.258</td><td>20.45</td><td>0.2237</td></tr><tr><td>3</td><td>1</td><td>184.74</td><td>1817.6</td><td>-2.204</td><td>18.39</td><td>0.1502</td></tr><tr><td>SSFM + exponential lift</td><td>3</td><td>2</td><td>57.99</td><td>772.3</td><td>-1.333</td><td>19.09</td><td>0.1819</td></tr><tr><td>+ closed-form martingale + drift matching</td><td>3</td><td>4</td><td>14.67</td><td>288.1</td><td>-0.683</td><td>19.75</td><td>0.2060</td></tr><tr><td rowspan="5"></td><td>3</td><td>8</td><td>4.15</td><td>128.2</td><td>-0.366</td><td>20.15</td><td>0.2218</td></tr><tr><td>3</td><td>16</td><td>2.03</td><td>70.8</td><td>-0.189</td><td>20.40</td><td>0.2317</td></tr><tr><td>3</td><td>1</td><td>111.61</td><td>994.2</td><td>-1.935</td><td>18.50</td><td>0.1483</td></tr><tr><td>3</td><td>2</td><td>53.11</td><td>583.5</td><td>-1.328</td><td>18.91</td><td>0.1798</td></tr><tr><td>3</td><td>4</td><td>5.96</td><td>143.0</td><td>-0.453</td><td>19.99</td><td>0.2143</td></tr><tr><td></td><td>3</td><td>8</td><td>2.37</td><td>75.2</td><td>-0.219</td><td>20.35</td><td>0.2282</td></tr><tr><td></td><td>3</td><td>16</td><td>2.07</td><td>55.1</td><td>-0.127</td><td>20.49</td><td>0.2346</td></tr><tr><td>Pre-trained teacher</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">SiT-XL/2+REPA (SDE)</td><td></td><td>1</td><td>309.28</td><td>3612.8</td><td>-2.278</td><td>16.624</td><td>0.1297</td></tr><tr><td>一</td><td>3</td><td>458.06</td><td>3269.6</td><td>-2.277</td><td>17.037</td><td>0.1336</td></tr><tr><td>一</td><td>8</td><td>335.84</td><td>2843.7</td><td>-2.242</td><td>17.235</td><td>0.1486</td></tr><tr><td>一</td><td>17</td><td>183.85</td><td>1288.1</td><td>-1.854</td><td>17.957</td><td>0.1744</td></tr><tr><td>一</td><td>85</td><td>3.65</td><td>88.5</td><td>-0.193</td><td>20.205</td><td>0.2316</td></tr><tr><td>一</td><td>170</td><td>1.77</td><td>60.1</td><td>-0.126</td><td>20.388</td><td>0.2342</td></tr><tr><td></td><td>426</td><td>1.53</td><td>47.5</td><td>-0.089</td><td>20.493</td><td>0.2353</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 6 40 Gaussians, $\beta = 3 .$ FKC with $K = 8 ,$ no cli<sub>pp</sub>in<sub>g</sub>. Mean over 4 seeds (10k sam<sub>p</sub>les) with standard deviation as a subscri t; lower is better; best er NFE in bold. ∞: diver ed. Rows without FKC are uncli ed and identical across tables.
<table><tr><td>Model</td><td>NFE</td><td>FKC</td><td> $\mathrm { E n e r g y } – \mathcal { W } _ { 2 }$ </td><td>MMD</td><td>TV</td><td> $\mathcal { W } _ { 1 }$ </td><td> $\mathcal { W } _ { 2 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>x</td><td> $\mathbf { 1 . 6 3 8 _ { \pm 0 . 1 0 5 } }$ </td><td> $0 . 0 2 0 7 _ { \pm 0 . 0 0 0 7 }$ </td><td> $\mathbf { 0 . 4 9 3 _ { \pm 0 . 0 0 7 } }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 2 5 }$ </td><td> $1 6 . 2 2 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>√</td><td> $2 3 . 2 8 4 _ { \pm 3 0 . 8 1 8 }$ </td><td> $\mathbf { 0 . 0 1 4 6 _ { \pm 0 . 0 0 0 9 } }$ </td><td> $0 . 5 0 2 _ { \pm 0 . 0 0 4 }$ </td><td>8.55±0.18</td><td> $\mathbf { 1 2 . 9 6 _ { \pm 0 . 4 4 } }$ </td></tr><tr><td>Diffusion SDE</td><td>4</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>4</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>8</td><td>x</td><td> $\mathbf { 0 . 8 2 1 _ { \pm 0 . 0 9 9 } }$ </td><td> $0 . 0 2 3 0 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 8 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 0 . 6 5 _ { \pm 0 . 1 4 }$ </td><td> $1 5 . 4 4 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>8</td><td>√</td><td> $2 . 7 8 7 _ { \pm 0 . 2 6 3 }$ </td><td> $\mathbf { 0 . 0 1 0 6 _ { \pm 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 0 . 4 2 0 _ { \pm 0 . 0 0 4 } }$ </td><td> ${ \bf 6 . 4 0 _ { \pm 0 . 1 6 } }$ </td><td> $\mathbf { 1 0 . 6 0 _ { \pm 0 . 2 6 } }$ </td></tr><tr><td>Diffusion SDE</td><td>8</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>8</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>12</td><td>x</td><td> $\mathbf { 0 . 4 9 6 _ { \pm 0 . 0 3 6 } }$ </td><td> $0 . 0 2 5 7 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 9 5 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 2 5 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 6 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>12</td><td>√</td><td> $2 . 4 1 6 _ { \pm 0 . 4 0 9 }$ </td><td> $\mathbf { 0 . 0 1 1 1 _ { \pm 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 0 . 4 1 0 _ { \pm 0 . 0 0 2 } }$ </td><td> ${ \bf 6 . 4 1 _ { \pm 0 . 1 2 } }$ </td><td> $\mathbf { 1 0 . 6 7 _ { \pm 0 . 5 9 } }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>x</td><td> $8 . 4 6 \times 1 0 _ { ~ + 7 . 8 3 \times 1 0 ^ { 4 } } ^ { 5 }$ </td><td> $0 . 0 4 0 7 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 6 1 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 2 0 . 1 1 _ { \pm 4 . 1 9 }$ </td><td> $5 8 7 . 2 2 _ { \pm 3 3 . 3 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>√</td><td> $2 . 5 9 \times 1 0 _ { + 2 . 2 4 \times 1 0 ^ { 5 } } ^ { 6 }$ </td><td> $0 . 0 4 7 6 _ { \pm 0 . 0 0 1 9 }$ </td><td> $0 . 7 0 1 _ { \pm 0 . 0 1 5 }$ </td><td> $8 0 7 . 7 8 _ { \pm 2 4 . 2 8 }$ </td><td> $1 4 4 7 . 7 1 _ { \pm 5 9 . 2 1 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>x</td><td> $\mathbf { 0 . 4 9 4 _ { \pm 0 . 0 8 2 } }$ </td><td> $0 . 0 2 6 7 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $1 5 . 8 4 _ { \pm 0 . 5 2 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>✓</td><td> $2 . 0 8 6 _ { \pm 0 . 1 4 8 }$ </td><td> $\mathbf { 0 . 0 1 1 3 _ { \pm 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 0 . 4 0 5 _ { \pm 0 . 0 0 2 } }$ </td><td> ${ \bf 6 . 2 1 _ { \pm 0 . 2 4 } }$ </td><td> $\mathbf { 1 0 . 2 3 _ { \pm 0 . 4 3 } }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>x</td><td> $1 . 6 1 \times 1 0 _ { \substack { \scriptscriptstyle + A \mathrm { ~ g ~ } \mathrm { ~ v ~ } 1 0 } 3 } ^ { 4 }$ </td><td> $0 . 0 3 9 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 5 7 8 _ { \pm 0 . 0 0 7 }$ </td><td> $1 5 . 4 6 _ { \pm 0 . 5 0 }$ </td><td> $3 6 . 5 6 _ { \pm 8 . 8 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td> $\checkmark$ </td><td> $6 . 2 5 \times 1 0 _ { \pm 1 . 6 7 \times 1 0 ^ { 4 } } ^ { \mp }$ </td><td> $0 . 0 3 7 5 _ { \pm 0 . 0 0 0 8 }$ </td><td> $0 . 5 6 7 _ { \pm 0 . 0 0 6 }$ </td><td> $2 7 . 0 3 _ { \pm 2 . 8 2 }$ </td><td> $1 1 1 . 5 5 _ { \pm 1 8 . 1 6 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>x</td><td> $0 . 4 6 9 _ { \pm 0 . 0 3 2 }$ </td><td> $0 . 0 2 7 1 { _ { \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 5 0 7 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 6 6 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>✓</td><td> $1 . 6 5 1 _ { \pm 0 . 1 8 9 }$ </td><td> $\mathbf { 0 . 0 1 1 1 _ { \pm 0 . 0 0 0 6 } }$ </td><td> $\mathbf { 0 . 4 0 4 _ { \pm 0 . 0 0 5 } }$ </td><td> ${ \bf 6 . 3 2 _ { \pm 0 . 2 6 } }$ </td><td> $\mathbf { 1 0 . 1 9 _ { \pm 0 . 4 9 } }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>x</td><td> $0 . 5 6 4 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 3 3 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 1 3 }$ </td><td> $1 4 . 5 3 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>√</td><td> $\mathbf { 0 . 4 4 4 _ { \pm 0 . 0 1 7 } }$ </td><td> $0 . 0 2 9 3 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 0 4 _ { \pm 0 . 0 0 3 }$ </td><td> $1 3 . 6 5 _ { \pm 0 . 5 5 }$ </td><td> $1 9 . 2 2 _ { \pm 0 . 6 1 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>x</td><td> $0 . 4 8 6 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 0 2 8 8 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 1 0 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 6 7 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>√</td><td> $1 . 7 3 9 _ { \pm 0 . 2 5 2 }$ </td><td> $\mathbf { 0 . 0 1 1 8 _ { \pm 0 . 0 0 0 2 } }$ </td><td> $0 . 4 0 4 _ { \pm 0 . 0 0 3 }$ </td><td> ${ \bf 6 . 3 8 _ { \pm 0 . 1 1 } }$ </td><td> ${ \bf 1 0 . 2 1 _ { \pm 0 . 4 7 } }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>x</td><td> $0 . 5 2 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 3 5 5 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 4 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 0 9 }$ </td><td> $1 6 . 1 5 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>√</td><td> $\mathbf { 0 . 1 8 3 _ { \pm 0 . 0 1 8 } }$ </td><td> $0 . 0 1 2 9 _ { \pm 0 . 0 0 0 7 }$ </td><td> $\mathbf { 0 . 4 0 1 _ { \pm 0 . 0 0 4 } }$ </td><td> $6 . 6 6 _ { \pm 0 . 2 6 }$ </td><td> $1 1 . 0 0 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>x</td><td> $0 . 4 7 6 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 0 2 9 7 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $1 6 . 0 8 _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>√</td><td> $1 . 6 4 6 _ { \pm 0 . 3 0 1 }$ </td><td> $\mathbf { 0 . 0 1 1 4 { \scriptstyle \pm 0 . 0 0 0 6 } }$ </td><td> $0 . 4 0 2 _ { \pm 0 . 0 0 6 }$ </td><td> $6 . 0 9 _ { \pm 0 . 4 3 }$ </td><td> $9 . 7 3 _ { \pm 0 . 6 8 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>x</td><td> $0 . 4 7 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 3 2 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 2 2 }$ </td><td> $1 6 . 6 1 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>√</td><td> $\mathbf { 0 . 1 8 0 _ { \pm 0 . 0 2 5 } }$ </td><td> $0 . 0 1 1 6 _ { \pm 0 . 0 0 0 5 }$ </td><td> $\mathbf { 0 . 3 9 0 _ { \pm 0 . 0 0 6 } }$ </td><td> ${ \bf 5 . 3 0 _ { \pm 0 . 2 3 } }$ </td><td> $\mathbf { 8 . 5 8 _ { \pm 0 . 2 7 } }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>x</td><td> $0 . 4 4 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 3 0 7 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 1 }$ </td><td> $1 2 . 2 8 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 4 6 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>√</td><td> $1 . 6 3 5 _ { \pm 0 . 1 2 6 }$ </td><td> $0 . 0 1 2 0 _ { \pm 0 . 0 0 0 2 }$ </td><td> $0 . 4 1 0 _ { \pm 0 . 0 0 4 }$ </td><td> $6 . 7 8 _ { \pm 0 . 2 5 }$ </td><td> $1 0 . 4 5 _ { \pm 0 . 3 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>x</td><td> $0 . 4 3 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 9 6 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 2 . 0 7 _ { \pm 0 . 2 4 }$ </td><td> $1 6 . 3 8 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>V</td><td> $\mathbf { 0 . 1 5 7 _ { \pm 0 . 0 0 9 } }$ </td><td> $\mathbf { 0 . 0 1 1 8 _ { \pm 0 . 0 0 0 5 } }$ </td><td> $\mathbf { 0 . 3 9 3 _ { \pm 0 . 0 0 9 } }$ </td><td> ${ \bf 6 . 0 0 _ { \pm 0 . 2 1 } }$ </td><td> $\mathbf { 9 . 8 7 _ { \pm 0 . 4 6 } }$ </td></tr></table>

## H.3 Feynman–Kac correctors

W<sub>e</sub> <sub>p</sub>r<sub>ese</sub>nt <sub>a</sub>dditi<sub>o</sub>n<sub>a</sub>l r<sub>esu</sub>lt<sub>s</sub> f<sub>o</sub>r th<sub>e</sub> FKC <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> in v<sub>a</sub>ri<sub>ous</sub> <sub>se</sub>ttin<sub>gs.</sub>

Table 7 40 Gaussians, $\beta = 3 . \operatorname { F K C }$ <sub>w</sub>ith $K = 8 ,$ 10% total lo<sub>g</sub>-wei<sub>g</sub>ht cli<sub>pp</sub>in<sub>g</sub>. Mean over 4 seeds (10k sam<sub>p</sub>les) with standard deviation as a subscri<sub>p</sub>t; lower is better; best <sub>p</sub>er NFE in bold. ∞: diver<sub>g</sub>ed. Rows without FKC are uncli<sub>pp</sub>ed and identical <sub>across</sub> t<sub>a</sub>bl<sub>es.</sub>
<table><tr><td>Model</td><td>NFE</td><td>FKC</td><td> $\mathrm { E n e r g y } – \mathcal { W } _ { 2 }$ </td><td>MMD</td><td>TV</td><td> $\mathcal { W } _ { 1 }$ </td><td>W2</td></tr><tr><td>iSSFM</td><td>4</td><td>x</td><td> $\mathbf { 1 . 6 3 8 _ { \pm 0 . 1 0 5 } }$ </td><td> $0 . 0 2 0 7 _ { \pm 0 . 0 0 0 7 }$ </td><td> $0 . 4 9 3 _ { \pm 0 . 0 0 7 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 2 5 }$ </td><td> $1 6 . 2 2 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>✓</td><td> $1 7 . 9 1 6 _ { \pm 2 3 . 6 2 8 }$ </td><td> $\mathbf { 0 . 0 1 4 2 _ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 0 . 4 8 9 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 8 . 5 4 _ { \pm 0 . 1 6 } }$ </td><td> $\mathbf { 1 3 . 0 6 _ { \pm 0 . 4 0 } }$ </td></tr><tr><td>Diffusion SDE</td><td>4</td><td>x</td><td> $\infty$ </td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>4</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>8</td><td>x</td><td> $\mathbf { 0 . 8 2 1 _ { \pm 0 . 0 9 9 } }$ </td><td> $0 . 0 2 3 0 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 8 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 0 . 6 5 _ { \pm 0 . 1 4 }$ </td><td> $1 5 . 4 4 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>8</td><td>√</td><td> $1 . 8 9 1 _ { \pm 0 . 1 8 8 }$ </td><td> $\mathbf { 0 . 0 1 1 3 _ { \pm 0 . 0 0 0 2 } }$ </td><td> $\mathbf { 0 . 4 2 5 _ { \pm 0 . 0 0 6 } }$ </td><td> ${ \bf 6 . 8 7 _ { \pm 0 . 1 3 } }$ </td><td> ${ \bf 1 1 . 2 2 _ { \pm 0 . 3 2 } }$ </td></tr><tr><td>Diffusion SDE</td><td>8</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>8</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>12</td><td>x</td><td> $\mathbf { 0 . 4 9 6 _ { \pm 0 . 0 3 6 } }$ </td><td> $0 . 0 2 5 7 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 9 5 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 2 5 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 6 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>12</td><td>√</td><td> $1 . 5 9 1 _ { \pm 0 . 2 2 8 }$ </td><td> $\mathbf { 0 . 0 1 1 9 { \scriptstyle \pm 0 . 0 0 0 1 } }$ </td><td> $\mathbf { 0 . 4 1 0 _ { \pm 0 . 0 0 4 } }$ </td><td> ${ \bf 6 . 7 8 _ { \pm 0 . 1 7 } }$ </td><td> $\mathbf { 1 1 . 0 5 _ { \pm 0 . 5 1 } }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>x</td><td> $8 . 4 6 \times 1 0 _ { \pm 7 . 8 3 \times 1 0 ^ { 4 } } ^ { 5 }$ </td><td> $0 . 0 4 0 7 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 6 1 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 2 0 . 1 1 _ { \pm 4 . 1 9 }$ </td><td> $5 8 7 . 2 2 _ { \pm 3 3 . 3 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td> $\checkmark$ </td><td> $2 . 4 6 \times 1 0 _ { \pm 2 . 2 9 \times 1 0 ^ { 5 } } ^ { 6 }$ </td><td> $0 . 0 4 7 9 _ { \pm 0 . 0 0 2 0 }$ </td><td> $0 . 7 0 6 _ { \pm 0 . 0 1 0 }$ </td><td> $8 1 5 . 7 0 _ { \pm 2 5 . 4 8 }$ </td><td> $1 4 4 0 . 5 7 _ { \pm 6 1 . 8 9 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>x</td><td> $\mathbf { 0 . 4 9 4 _ { \pm 0 . 0 8 2 } }$ </td><td> $0 . 0 2 6 7 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $1 5 . 8 4 _ { \pm 0 . 5 2 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>S</td><td> $1 . 2 2 4 _ { \pm 0 . 1 9 3 }$ </td><td> $\mathbf { 0 . 0 1 2 7 _ { \pm 0 . 0 0 0 7 } }$ </td><td> $\mathbf { 0 . 4 1 1 _ { \pm 0 . 0 0 4 } }$ </td><td> ${ \bf 6 . 9 9 _ { \pm 0 . 3 2 } }$ </td><td> $\mathbf { 1 1 . 2 2 _ { \pm 0 . 6 0 } }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>x</td><td> $1 . 6 1 \times 1 0 ^ { 4 }$  ≠4.82×103</td><td> $0 . 0 3 9 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 5 7 8 _ { \pm 0 . 0 0 7 }$ </td><td> $1 5 . 4 6 _ { \pm 0 . 5 0 }$ </td><td> $3 6 . 5 6 _ { \pm 8 . 8 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td> $\checkmark$ </td><td> $4 . 8 7 \times 1 0 _ { \pm 1 . 4 6 \times 1 0 ^ { 4 } } ^ { \mp }$ </td><td> $0 . 0 3 8 2 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 7 0 _ { \pm 0 . 0 0 5 }$ </td><td> $2 5 . 9 5 _ { \pm 2 . 2 8 }$ </td><td> $9 9 . 2 2 _ { \pm 1 6 . 2 1 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td></td><td> $0 . 4 6 9 _ { \pm 0 . 0 3 2 }$ </td><td> $0 . 0 2 7 1 { _ { \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 5 0 7 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 6 6 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>× √</td><td> $1 . 2 8 6 _ { \pm 0 . 2 5 2 }$ </td><td> $\mathbf { 0 . 0 1 2 7 _ { \pm 0 . 0 0 0 5 } }$ </td><td> $\mathbf { 0 . 4 1 5 _ { \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 7 . 0 0 _ { \pm 0 . 3 0 } }$ </td><td> $\mathbf { 1 0 . 9 4 _ { \pm 0 . 4 8 } }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>× √</td><td> $0 . 5 6 4 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 3 3 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 1 3 }$ </td><td> $1 4 . 5 3 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td></td><td> $\mathbf { 0 . 4 5 5 _ { \pm 0 . 0 1 5 } }$ </td><td> $0 . 0 3 1 0 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 2 _ { \pm 0 . 0 1 2 }$ </td><td> $1 4 . 3 3 _ { \pm 0 . 4 4 }$ </td><td> $1 9 . 8 6 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>x</td><td> $0 . 4 8 6 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 0 2 8 8 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 1 0 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 6 7 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>√</td><td> $1 . 0 1 8 _ { \pm 0 . 3 1 9 }$ </td><td> $\mathbf { 0 . 0 1 3 3 _ { \pm 0 . 0 0 0 9 } }$ </td><td> $0 . 4 1 4 _ { \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 7 . 0 9 _ { \pm 0 . 0 4 } }$ </td><td> $\mathbf { 1 0 . 9 7 _ { \pm 0 . 2 3 } }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>x</td><td> $0 . 5 2 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 3 5 5 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 4 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 0 9 }$ </td><td> $1 6 . 1 5 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>S</td><td> $\mathbf { 0 . 2 0 4 } _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 0 1 3 5 _ { \pm 0 . 0 0 0 5 }$ </td><td> $\mathbf { 0 . 4 1 2 _ { \pm 0 . 0 0 3 } }$ </td><td> $7 . 4 2 _ { \pm 0 . 1 9 }$ </td><td> $1 1 . 8 6 _ { \pm 0 . 3 2 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>x</td><td> $0 . 4 7 6 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 0 2 9 7 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $1 6 . 0 8 _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>√</td><td> $0 . 9 5 0 _ { \pm 0 . 2 5 4 }$ </td><td> $\mathbf { 0 . 0 1 3 7 _ { \pm 0 . 0 0 0 4 } }$ </td><td> $0 . 4 1 9 _ { \pm 0 . 0 0 3 }$ </td><td> $7 . 3 9 _ { \pm 0 . 1 8 }$ </td><td> $1 1 . 3 2 _ { \pm 0 . 4 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>× √</td><td> $0 . 4 7 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 3 2 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 2 2 }$ </td><td> $1 6 . 6 1 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td></td><td> $\mathbf { 0 . 1 9 1 } _ { \pm 0 . 0 0 6 }$ </td><td> $0 . 0 1 3 7 _ { \pm 0 . 0 0 0 2 }$ </td><td> $\mathbf { 0 . 4 1 2 _ { \pm 0 . 0 0 6 } }$ </td><td> ${ \bf 6 . 5 9 _ { \pm 0 . 1 4 } }$ </td><td> ${ \bf 9 . 9 8 _ { \pm 0 . 2 9 } }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>x</td><td> $0 . 4 4 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 3 0 7 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 1 }$ </td><td> $1 2 . 2 8 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 4 6 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>√</td><td> $0 . 6 4 7 _ { \pm 0 . 1 3 1 }$ </td><td> $0 . 0 1 4 2 _ { \pm 0 . 0 0 0 6 }$ </td><td> $0 . 4 2 6 _ { \pm 0 . 0 0 2 }$ </td><td> $7 . 7 0 _ { \pm 0 . 1 3 }$ </td><td> $1 1 . 5 7 _ { \pm 0 . 2 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>× &gt;</td><td> $0 . 4 3 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 9 6 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 2 . 0 7 _ { \pm 0 . 2 4 }$ </td><td> $1 6 . 3 8 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td></td><td> $\mathbf { 0 . 2 0 5 _ { \pm 0 . 0 1 0 } }$ </td><td> $\mathbf { 0 . 0 1 4 1 _ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 0 . 4 1 1 _ { \pm 0 . 0 0 7 } }$ </td><td> $7 . 1 7 _ { \pm 0 . 2 2 }$ </td><td> $\mathbf { 1 1 . 1 2 _ { \pm 0 . 4 8 } }$ </td></tr></table>

Table 8 40 Gaussians, $\beta = 3 .$ . FKC with $K = 8 ,$ 20% total lo<sub>g</sub>-wei<sub>g</sub>ht cli<sub>pp</sub>in<sub>g</sub>. Mean over 4 seeds (10k sam<sub>p</sub>les) with standard deviation as a subscri<sub>p</sub>t; lower is better; best <sub>p</sub>er NFE in bold. ∞: diver<sub>g</sub>ed. Rows without FKC are uncli<sub>pp</sub>ed and identical <sub>across</sub> t<sub>a</sub>bl<sub>es.</sub>
<table><tr><td>Model</td><td>NFE</td><td>FKC</td><td> $\mathrm { E n e r g y } – \mathcal { W } _ { 2 }$ </td><td>MMD</td><td>TV</td><td>W1</td><td>W2</td></tr><tr><td>iSSFM</td><td>4</td><td>x</td><td> $\mathbf { 1 . 6 3 8 _ { \pm 0 . 1 0 5 } }$ </td><td> $0 . 0 2 0 7 _ { \pm 0 . 0 0 0 7 }$ </td><td> $0 . 4 9 3 _ { \pm 0 . 0 0 7 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 2 5 }$ </td><td> $1 6 . 2 2 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>✓</td><td> $2 0 . 1 7 2 _ { \pm 2 8 . 8 6 6 }$ </td><td> $\mathbf { 0 . 0 1 4 2 _ { \pm 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 0 . 4 7 5 _ { \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 8 . 7 0 _ { \pm 0 . 0 9 } }$ </td><td> ${ \bf 1 3 . 4 4 } _ { \pm 0 . 2 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>4</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>4</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>8</td><td>x</td><td> $\mathbf { 0 . 8 2 1 _ { \pm 0 . 0 9 9 } }$ </td><td> $0 . 0 2 3 0 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 8 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 0 . 6 5 _ { \pm 0 . 1 4 }$ </td><td> $1 5 . 4 4 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>8</td><td>√</td><td> $1 . 4 4 6 _ { \pm 0 . 1 3 3 }$ </td><td> $\mathbf { 0 . 0 1 3 0 } _ { \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 4 2 7 _ { \pm 0 . 0 0 6 } }$ </td><td> ${ \bf 7 . 3 9 _ { \pm 0 . 1 2 } }$ </td><td> $\mathbf { 1 1 . 9 1 _ { \pm 0 . 3 1 } }$ </td></tr><tr><td>Diffusion SDE</td><td>8</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>8</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>12</td><td>x</td><td> $\mathbf { 0 . 4 9 6 _ { \pm 0 . 0 3 6 } }$ </td><td> $0 . 0 2 5 7 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 9 5 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 2 5 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 6 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>12</td><td>√</td><td> $0 . 9 6 5 _ { \pm 0 . 1 5 5 }$ </td><td> $\mathbf { 0 . 0 1 3 6 _ { \pm 0 . 0 0 0 6 } }$ </td><td> $\mathbf { 0 . 4 2 0 _ { \pm 0 . 0 0 4 } }$ </td><td> $7 . 4 5 _ { \pm 0 . 1 4 }$ </td><td> $\mathbf { 1 1 . 9 4 _ { \pm 0 . 4 5 } }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>x</td><td> $8 . 4 6 \times 1 0 _ { \pm 7 . 8 3 \times 1 0 ^ { 4 } } ^ { 5 }$ </td><td> $0 . 0 4 0 7 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 6 1 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 2 0 . 1 1 _ { \pm 4 . 1 9 }$ </td><td> $5 8 7 . 2 2 _ { \pm 3 3 . 3 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td> $\checkmark$ </td><td> $2 . 4 2 \times 1 0 _ { \pm 2 . 1 3 \times 1 0 ^ { 5 } } ^ { 6 }$ </td><td> $0 . 0 4 8 7 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 7 1 5 _ { \pm 0 . 0 1 8 }$ </td><td> $8 2 4 . 4 9 _ { \pm 2 4 . 9 0 }$ </td><td> $1 4 4 0 . 3 3 _ { \pm 6 0 . 1 6 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>x</td><td> $\mathbf { 0 . 4 9 4 _ { \pm 0 . 0 8 2 } }$ </td><td> $0 . 0 2 6 7 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $1 5 . 8 4 _ { \pm 0 . 5 2 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>S</td><td> $0 . 9 3 2 _ { \pm 0 . 1 7 4 }$ </td><td> $\mathbf { 0 . 0 1 4 1 _ { \pm 0 . 0 0 0 5 } }$ </td><td> $\mathbf { 0 . 4 2 2 _ { \pm 0 . 0 0 1 } }$ </td><td> $7 . 4 7 _ { \pm 0 . 2 2 }$ </td><td> $\mathbf { 1 1 . 8 9 _ { \pm 0 . 4 8 } }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>x</td><td> $1 . 6 1 \times 1 0 ^ { 4 }$  ≠4.82×103</td><td> $0 . 0 3 9 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 5 7 8 _ { \pm 0 . 0 0 7 }$ </td><td> $1 5 . 4 6 _ { \pm 0 . 5 0 }$ </td><td> $3 6 . 5 6 _ { \pm 8 . 8 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td> $\checkmark$ </td><td> $4 . 8 7 \times 1 0 _ { \pm 1 . 5 6 \times 1 0 ^ { 4 } } ^ { \mp }$ </td><td> $0 . 0 3 9 3 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 6 6 _ { \pm 0 . 0 0 9 }$ </td><td> $2 5 . 6 4 _ { \pm 2 . 5 0 }$ </td><td> $9 8 . 4 1 _ { \pm 1 8 . 9 8 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td></td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 0 3 2 } }$ </td><td> $0 . 0 2 7 1 { _ { \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 5 0 7 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 6 6 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>× √</td><td> $0 . 8 4 3 _ { \pm 0 . 1 8 2 }$ </td><td> $\mathbf { 0 . 0 1 4 9 _ { \pm 0 . 0 0 0 8 } }$ </td><td> $\mathbf { 0 . 4 2 6 _ { \pm 0 . 0 0 6 } }$ </td><td> $7 . 6 4 _ { \pm 0 . 2 4 }$ </td><td> $\mathbf { 1 1 . 8 5 _ { \pm 0 . 5 5 } }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>× √</td><td> $0 . 5 6 4 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 3 3 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 1 3 }$ </td><td> $1 4 . 5 3 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td></td><td> $0 . 4 8 5 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 0 3 5 1 _ { \pm 0 . 0 0 2 1 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 7 }$ </td><td> $1 4 . 9 6 _ { \pm 0 . 5 7 }$ </td><td> $2 0 . 4 3 _ { \pm 0 . 6 0 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>x</td><td> $0 . 4 8 6 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 0 2 8 8 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 1 0 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 6 7 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>√</td><td> $0 . 7 7 7 _ { \pm 0 . 2 3 0 }$ </td><td> $0 . 0 1 5 4 _ { \pm 0 . 0 0 0 8 }$ </td><td> $0 . 4 2 8 _ { \pm 0 . 0 0 2 }$ </td><td> $7 . 8 5 _ { \pm 0 . 1 7 }$ </td><td> $1 1 . 9 4 _ { \pm 0 . 3 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>x</td><td> $0 . 5 2 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 3 5 5 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 4 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 0 9 }$ </td><td> $1 6 . 1 5 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>V</td><td> $\mathbf { 0 . 2 1 9 _ { \pm 0 . 0 0 7 } }$ </td><td> $\mathbf { 0 . 0 1 5 0 } _ { \pm 0 . 0 0 0 6 }$ </td><td> $\mathbf { 0 . 4 2 1 _ { \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 7 . 0 7 _ { \pm 0 . 2 2 } }$ </td><td> ${ \bf 1 1 . 1 1 _ { \pm 0 . 2 9 } }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>x</td><td> $0 . 4 7 6 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 0 2 9 7 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $1 6 . 0 8 _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>√</td><td> $0 . 7 5 6 _ { \pm 0 . 1 1 1 }$ </td><td> $\mathbf { 0 . 0 1 5 4 } _ { \pm 0 . 0 0 0 8 }$ </td><td> $0 . 4 3 3 _ { \pm 0 . 0 0 2 }$ </td><td> $8 . 0 3 _ { \pm 0 . 3 0 }$ </td><td> $1 2 . 1 0 _ { \pm 0 . 6 1 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>× √</td><td> $0 . 4 7 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 3 2 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 2 2 }$ </td><td> $1 6 . 6 1 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td></td><td> $\mathbf { 0 . 2 2 1 _ { \pm 0 . 0 0 9 } }$ </td><td> $0 . 0 1 6 1 _ { \pm 0 . 0 0 0 6 }$ </td><td> $\mathbf { 0 . 4 2 9 _ { \pm 0 . 0 0 3 } }$ </td><td> ${ \bf 7 . 5 1 { _ { \pm 0 . 1 6 } } }$ </td><td> $\mathbf { 1 1 . 0 9 } _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>x</td><td> $0 . 4 4 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 3 0 7 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 1 }$ </td><td> $1 2 . 2 8 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 4 6 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>√</td><td> $0 . 6 3 7 _ { \pm 0 . 1 8 6 }$ </td><td> $0 . 0 1 6 7 _ { \pm 0 . 0 0 1 0 }$ </td><td> $0 . 4 4 2 _ { \pm 0 . 0 0 4 }$ </td><td> $8 . 4 3 _ { \pm 0 . 2 0 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>x</td><td> $0 . 4 3 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 9 6 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 2 . 0 7 _ { \pm 0 . 2 4 }$ </td><td> $1 6 . 3 8 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>√</td><td> $\mathbf { 0 . 2 4 0 _ { \pm 0 . 0 1 3 } }$ </td><td> $\mathbf { 0 . 0 1 6 3 _ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 0 . 4 2 8 _ { \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 8 . 1 5 _ { \pm 0 . 1 4 } }$ </td><td> $\mathbf { 1 2 . 3 0 _ { \pm 0 . 4 2 } }$ </td></tr></table>

Table 9 40 Gaussians, $\beta = 3 .$ . FKC with $K = 8 ,$ 25% total lo<sub>g</sub>-wei<sub>g</sub>ht cli<sub>pp</sub>in<sub>g</sub>. Mean over 4 seeds (10k sam<sub>p</sub>les) with standard deviation as a subscri<sub>p</sub>t; lower is better; best <sub>p</sub>er NFE in bold. ∞: diver<sub>g</sub>ed. Rows without FKC are uncli<sub>pp</sub>ed and identical <sub>across</sub> t<sub>a</sub>bl<sub>es.</sub>
<table><tr><td>Model</td><td>NFE</td><td>FKC</td><td> $\mathrm { E n e r g y } – \mathcal { W } _ { 2 }$ </td><td>MMD</td><td>TV</td><td> $\mathcal { W } _ { 1 }$ </td><td>W2</td></tr><tr><td>iSSFM</td><td>4</td><td>x</td><td> $\mathbf { 1 . 6 3 8 _ { \pm 0 . 1 0 5 } }$ </td><td> $0 . 0 2 0 7 _ { \pm 0 . 0 0 0 7 }$ </td><td> $0 . 4 9 3 _ { \pm 0 . 0 0 7 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 2 5 }$ </td><td> $1 6 . 2 2 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>✓</td><td> $1 8 . 9 0 0 _ { \pm 2 7 . 0 1 1 }$ </td><td> $\mathbf { 0 . 0 1 5 2 _ { \pm 0 . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 4 7 3 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 9 . 0 0 _ { \pm 0 . 1 8 } }$ </td><td> $\mathbf { 1 3 . 7 0 _ { \pm 0 . 2 3 } }$ </td></tr><tr><td>Diffusion SDE</td><td>4</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>4</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>8</td><td>x</td><td> $\mathbf { 0 . 8 2 1 _ { \pm 0 . 0 9 9 } }$ </td><td> $0 . 0 2 3 0 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 8 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 0 . 6 5 _ { \pm 0 . 1 4 }$ </td><td> $1 5 . 4 4 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>8</td><td>√</td><td> $1 . 3 2 1 _ { \pm 0 . 0 9 7 }$ </td><td> $\mathbf { 0 . 0 1 3 5 _ { \pm 0 . 0 0 0 2 } }$ </td><td> $\mathbf { 0 . 4 2 8 _ { \pm 0 . 0 0 5 } }$ </td><td> ${ \bf 7 . 5 8 _ { \pm 0 . 0 9 } }$ </td><td> $\mathbf { 1 2 . 1 0 _ { \pm 0 . 2 8 } }$ </td></tr><tr><td>Diffusion SDE</td><td>8</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>8</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>12</td><td>x</td><td> $\mathbf { 0 . 4 9 6 _ { \pm 0 . 0 3 6 } }$ </td><td> $0 . 0 2 5 7 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 9 5 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 2 5 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 6 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>12</td><td>√</td><td> $0 . 8 1 3 _ { \pm 0 . 1 4 1 }$ </td><td> $\mathbf { 0 . 0 1 4 6 _ { \pm 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 0 . 4 2 8 _ { \pm 0 . 0 0 2 } }$ </td><td> ${ \bf 7 . 8 1 _ { \pm 0 . 1 9 } }$ </td><td> $\mathbf { 1 2 . 4 5 _ { \pm 0 . 4 0 } }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>x</td><td> $8 . 4 6 \times 1 0 _ { \pm 7 . 8 3 \times 1 0 ^ { 4 } } ^ { 5 }$ </td><td> $0 . 0 4 0 7 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 6 1 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 2 0 . 1 1 _ { \pm 4 . 1 9 }$ </td><td> $5 8 7 . 2 2 _ { \pm 3 3 . 3 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td> $\checkmark$ </td><td> $2 . 4 0 \times 1 0 _ { \pm 2 . 0 0 \times 1 0 ^ { 5 } } ^ { 6 }$ </td><td> $0 . 0 4 9 5 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 7 2 6 _ { \pm 0 . 0 1 9 }$ </td><td> $8 2 9 . 8 6 _ { \pm 2 7 . 5 7 }$ </td><td> $1 4 3 8 . 3 1 _ { \pm 6 1 . 9 6 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>x</td><td> $\mathbf { 0 . 4 9 4 _ { \pm 0 . 0 8 2 } }$ </td><td> $0 . 0 2 6 7 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $1 5 . 8 4 _ { \pm 0 . 5 2 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>S</td><td> $0 . 8 1 4 _ { \pm 0 . 1 7 4 }$ </td><td> $\mathbf { 0 . 0 1 5 1 } _ { \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 4 2 7 _ { \pm 0 . 0 0 4 } }$ </td><td> $7 . 6 7 _ { \pm 0 . 2 1 }$ </td><td> $\mathbf { 1 2 . 0 6 _ { \pm 0 . 4 7 } }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>x</td><td> $1 . 6 1 \times 1 0 ^ { 4 }$  ≠4.82×103</td><td> $0 . 0 3 9 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 5 7 8 _ { \pm 0 . 0 0 7 }$ </td><td> $1 5 . 4 6 _ { \pm 0 . 5 0 }$ </td><td> $3 6 . 5 6 _ { \pm 8 . 8 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td> $\checkmark$ </td><td> $4 . 7 9 \times 1 0 _ { \pm 1 . 5 2 \times 1 0 ^ { 4 } } ^ { \mp }$ </td><td> $0 . 0 3 8 4 _ { \pm 0 . 0 0 1 3 }$ </td><td> $0 . 5 6 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 5 . 3 5 _ { \pm 2 . 2 7 }$ </td><td> $9 6 . 5 8 _ { \pm 1 7 . 4 4 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td></td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 0 3 2 } }$ </td><td> $0 . 0 2 7 1 { _ { \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 5 0 7 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 6 6 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>× √</td><td> $0 . 6 7 8 _ { \pm 0 . 1 4 8 }$ </td><td> $\mathbf { 0 . 0 1 5 4 } _ { \pm 0 . 0 0 0 6 }$ </td><td> $\mathbf { 0 . 4 2 9 _ { \pm 0 . 0 0 5 } }$ </td><td> $7 . 8 6 _ { \pm 0 . 1 2 }$ </td><td> $\mathbf { 1 2 . 1 5 _ { \pm 0 . 3 9 } }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>× √</td><td> $0 . 5 6 4 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 3 3 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 1 3 }$ </td><td> $1 4 . 5 3 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td></td><td> $0 . 4 9 8 _ { \pm 0 . 0 2 5 }$ </td><td> $0 . 0 3 6 7 _ { \pm 0 . 0 0 1 4 }$ </td><td> $0 . 5 3 0 _ { \pm 0 . 0 0 3 }$ </td><td> $1 4 . 9 5 _ { \pm 0 . 3 8 }$ </td><td> $2 0 . 3 5 _ { \pm 0 . 4 6 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>x</td><td> $0 . 4 8 6 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 0 2 8 8 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 1 0 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 6 7 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>√</td><td> $0 . 5 4 1 _ { \pm 0 . 0 7 7 }$ </td><td> $\mathbf { 0 . 0 1 5 7 { \scriptstyle \pm 0 . 0 0 1 2 } }$ </td><td> $0 . 4 3 1 _ { \pm 0 . 0 0 6 }$ </td><td> $7 . 9 5 _ { \pm 0 . 1 9 }$ </td><td> $1 1 . 9 9 _ { \pm 0 . 3 6 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>x</td><td> $0 . 5 2 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 3 5 5 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 4 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 0 9 }$ </td><td> $1 6 . 1 5 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>V</td><td> $\mathbf { 0 . 2 2 6 _ { \pm 0 . 0 1 2 } }$ </td><td> $0 . 0 1 5 7 _ { \pm 0 . 0 0 0 3 }$ </td><td> $\mathbf { 0 . 4 2 6 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 7 . 0 5 _ { \pm 0 . 2 0 } }$ </td><td> $\mathbf { 1 0 . 9 4 _ { \pm 0 . 3 0 } }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>x</td><td> $0 . 4 7 6 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 0 2 9 7 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $1 6 . 0 8 _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>√</td><td> $0 . 4 4 2 _ { \pm 0 . 0 7 7 }$ </td><td> $\mathbf { 0 . 0 1 6 4 } _ { \pm 0 . 0 0 0 8 }$ </td><td> $0 . 4 4 0 _ { \pm 0 . 0 0 2 }$ </td><td> $8 . 4 0 _ { \pm 0 . 3 0 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 5 7 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>× √</td><td> $0 . 4 7 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 3 2 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 2 2 }$ </td><td> $1 6 . 6 1 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td></td><td> $\mathbf { 0 . 2 2 2 _ { \pm 0 . 0 0 8 } }$ </td><td> $0 . 0 1 7 0 _ { \pm 0 . 0 0 1 2 }$ </td><td> $\mathbf { 0 . 4 3 7 _ { \pm 0 . 0 0 2 } }$ </td><td> ${ \bf 7 . 9 4 _ { \pm 0 . 0 4 } }$ </td><td> $\mathbf { 1 1 . 5 7 _ { \pm 0 . 2 1 } }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>x</td><td> $0 . 4 4 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 3 0 7 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 1 }$ </td><td> $1 2 . 2 8 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 4 6 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>√</td><td> $0 . 5 8 9 _ { \pm 0 . 3 3 6 }$ </td><td> $\mathbf { 0 . 0 1 7 3 _ { \pm 0 . 0 0 1 1 } }$ </td><td> $0 . 4 4 5 _ { \pm 0 . 0 0 7 }$ </td><td> $8 . 6 3 _ { \pm 0 . 0 9 }$ </td><td> $1 2 . 6 2 _ { \pm 0 . 2 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>× &gt;</td><td> $0 . 4 3 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 9 6 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 2 . 0 7 _ { \pm 0 . 2 4 }$ </td><td> $1 6 . 3 8 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td></td><td> $\mathbf { 0 . 2 4 5 _ { \pm 0 . 0 1 5 } }$ </td><td> $0 . 0 1 7 3 _ { \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 4 3 5 _ { \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 8 . 2 9 _ { \pm 0 . 1 2 } }$ </td><td> $\mathbf { 1 2 . 4 9 _ { \pm 0 . 4 1 } }$ </td></tr></table>

Table 10 40 Gaussians, $\beta = 3 .$ FKC with $K = 8 ,$ , 50% total lo<sub>g</sub>-wei<sub>g</sub>ht cli<sub>pp</sub>in<sub>g</sub>. Mean over 4 seeds (10k sam<sub>p</sub>les) with standard deviation as a subscri<sub>p</sub>t; lower is better; best <sub>p</sub>er NFE in bold. ∞: diver<sub>g</sub>ed. Rows without FKC are uncli<sub>pp</sub>ed and identical <sub>across</sub> t<sub>a</sub>bl<sub>es.</sub>
<table><tr><td>Model</td><td>NFE</td><td>FKC</td><td> $\mathrm { E n e r g y } – \mathcal { W } _ { 2 }$ </td><td>MMD</td><td>TV</td><td>W1</td><td>W2</td></tr><tr><td>iSSFM</td><td>4</td><td>x</td><td> $\mathbf { 1 . 6 3 8 _ { \pm 0 . 1 0 5 } }$ </td><td> $0 . 0 2 0 7 _ { \pm 0 . 0 0 0 7 }$ </td><td> $0 . 4 9 3 _ { \pm 0 . 0 0 7 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 2 5 }$ </td><td> $1 6 . 2 2 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>✓</td><td> $1 1 . 8 0 1 _ { \pm 1 6 . 0 9 3 }$ </td><td> $\mathbf { 0 . 0 1 6 3 _ { \pm 0 . 0 0 0 7 } }$ </td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 0 0 9 } }$ </td><td> $\mathbf { 9 . 4 6 _ { \pm 0 . 1 7 } }$ </td><td> $\mathbf { 1 4 . 4 3 _ { \pm 0 . 3 4 } }$ </td></tr><tr><td>Diffusion SDE</td><td>4</td><td>x</td><td> $\infty$ </td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>4</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>8</td><td>x</td><td> $\mathbf { 0 . 8 2 1 _ { \pm 0 . 0 9 9 } }$ </td><td> $0 . 0 2 3 0 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 8 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 0 . 6 5 _ { \pm 0 . 1 4 }$ </td><td> $1 5 . 4 4 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>8</td><td>√</td><td> $0 . 9 5 0 _ { \pm 0 . 0 4 9 }$ </td><td> $\mathbf { 0 . 0 1 5 5 _ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 0 . 4 4 1 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 8 . 4 3 _ { \pm 0 . 1 5 } }$ </td><td> ${ \bf 1 3 . 1 1 _ { \pm 0 . 2 9 } }$ </td></tr><tr><td>Diffusion SDE</td><td>8</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>8</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>12</td><td>x</td><td> $\mathbf { 0 . 4 9 6 _ { \pm 0 . 0 3 6 } }$ </td><td> $0 . 0 2 5 7 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 9 5 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 2 5 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 6 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>12</td><td>√</td><td> $0 . 5 5 3 _ { \pm 0 . 0 9 7 }$ </td><td> $\mathbf { 0 . 0 1 7 7 _ { \pm 0 . 0 0 1 2 } }$ </td><td> $\mathbf { 0 . 4 4 7 _ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 8 . 6 8 _ { \pm 0 . 1 2 } }$ </td><td> $\mathbf { 1 3 . 3 0 _ { \pm 0 . 4 7 } }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>x</td><td> $8 . 4 6 \times 1 0 _ { \pm 7 . 8 3 \times 1 0 ^ { 4 } } ^ { 5 }$ </td><td> $0 . 0 4 0 7 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 6 1 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 2 0 . 1 1 _ { \pm 4 . 1 9 }$ </td><td> $5 8 7 . 2 2 _ { \pm 3 3 . 3 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td> $\checkmark$ </td><td> $2 . 3 5 \times 1 0 _ { \pm 2 . 2 6 \times 1 0 ^ { 5 } } ^ { 6 }$ </td><td> $0 . 0 4 8 9 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 7 0 7 _ { \pm 0 . 0 1 2 }$ </td><td> $8 4 0 . 6 8 _ { \pm 1 9 . 7 2 }$ </td><td> $1 4 2 8 . 3 7 _ { \pm 4 8 . 9 0 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>x</td><td> $\mathbf { 0 . 4 9 4 _ { \pm 0 . 0 8 2 } }$ </td><td> $0 . 0 2 6 7 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $1 5 . 8 4 _ { \pm 0 . 5 2 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>√</td><td> $0 . 6 2 3 _ { \pm 0 . 1 6 1 }$ </td><td> $\mathbf { 0 . 0 1 8 0 _ { \pm 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 0 . 4 4 7 _ { \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 8 . 6 5 _ { \pm 0 . 0 8 } }$ </td><td> $\mathbf { 1 3 . 1 6 _ { \pm 0 . 4 8 } }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>x</td><td> $1 . 6 1 \times 1 0 ^ { 4 }$  ≠4.82×103</td><td> $0 . 0 3 9 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 5 7 8 _ { \pm 0 . 0 0 7 }$ </td><td> $1 5 . 4 6 _ { \pm 0 . 5 0 }$ </td><td> $3 6 . 5 6 _ { \pm 8 . 8 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>√</td><td> $4 . 8 3 \times 1 0 _ { \pm 1 . 4 5 \times 1 0 ^ { 4 } } ^ { \mp }$ </td><td> $0 . 0 3 9 0 _ { \pm 0 . 0 0 0 9 }$ </td><td> $0 . 5 7 0 _ { \pm 0 . 0 0 3 }$ </td><td> $2 5 . 5 5 _ { \pm 2 . 7 9 }$ </td><td> $9 5 . 8 6 _ { \pm 1 7 . 4 1 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>x</td><td> $0 . 4 6 9 _ { \pm 0 . 0 3 2 }$ </td><td> $0 . 0 2 7 1 { _ { \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 5 0 7 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 6 6 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>√</td><td> $\mathbf { 0 . 4 6 8 _ { \pm 0 . 0 5 7 } }$ </td><td> $\mathbf { 0 . 0 1 9 1 _ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 0 . 4 5 2 _ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 9 . 1 4 _ { \pm 0 . 2 3 } }$ </td><td> ${ \bf 1 3 . 4 1 _ { \pm 0 . 4 3 } }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>x</td><td> $0 . 5 6 4 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 3 3 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 1 3 }$ </td><td> $1 4 . 5 3 _ { \pm 0 . 1 5 }$ </td></tr><tr><td>Diffusion SDE</td><td>20</td><td>√</td><td> $0 . 5 2 8 _ { \pm 0 . 0 1 8 }$ </td><td> $0 . 0 3 4 6 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 7 }$ </td><td> $1 4 . 2 7 _ { \pm 0 . 4 1 }$ </td><td> $1 9 . 5 9 _ { \pm 0 . 4 2 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>x</td><td> $0 . 4 8 6 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 0 2 8 8 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 1 0 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 6 7 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>√</td><td> $0 . 4 6 3 _ { \pm 0 . 0 7 5 }$ </td><td> $\mathbf { 0 . 0 1 9 3 _ { \pm 0 . 0 0 1 3 } }$ </td><td> $\mathbf { 0 . 4 5 4 _ { \pm 0 . 0 0 7 } }$ </td><td> $9 . 2 8 _ { \pm 0 . 2 9 }$ </td><td> $1 3 . 4 9 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>x</td><td> $0 . 5 2 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 3 5 5 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 4 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 0 9 }$ </td><td> $1 6 . 1 5 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>√</td><td> $\mathbf { 0 . 3 1 7 _ { \pm 0 . 0 1 7 } }$ </td><td> $0 . 0 2 0 8 _ { \pm 0 . 0 0 1 3 }$ </td><td> $0 . 4 5 9 _ { \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 8 . 4 6 _ { \pm 0 . 1 1 } }$ </td><td> $\mathbf { 1 1 . 6 0 _ { \pm 0 . 1 3 } }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>x</td><td> $0 . 4 7 6 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 0 2 9 7 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $1 6 . 0 8 _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>√</td><td> $0 . 4 4 2 _ { \pm 0 . 0 4 0 }$ </td><td> $\mathbf { 0 . 0 1 9 5 _ { \pm 0 . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 4 6 4 _ { \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 9 . 5 5 _ { \pm 0 . 2 8 } }$ </td><td> $1 3 . 6 6 _ { \pm 0 . 5 8 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>x</td><td> $0 . 4 7 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 3 2 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 2 2 }$ </td><td> $1 6 . 6 1 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>V</td><td> $\mathbf { 0 . 2 9 9 _ { \pm 0 . 0 2 3 } }$ </td><td> $0 . 0 2 2 9 _ { \pm 0 . 0 0 1 4 }$ </td><td> $0 . 4 7 0 _ { \pm 0 . 0 0 3 }$ </td><td> $9 . 7 6 _ { \pm 0 . 3 0 }$ </td><td> $\mathbf { 1 3 . 6 5 _ { \pm 0 . 5 7 } }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>x</td><td> $0 . 4 4 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 3 0 7 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 1 }$ </td><td> $1 2 . 2 8 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 4 6 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>√</td><td> $0 . 4 4 2 _ { \pm 0 . 0 9 3 }$ </td><td> $0 . 0 2 2 5 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 7 3 _ { \pm 0 . 0 0 5 }$ </td><td> $1 0 . 0 8 _ { \pm 0 . 1 2 }$ </td><td> $1 4 . 2 7 _ { \pm 0 . 1 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>× √</td><td> $0 . 4 3 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 9 6 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 2 . 0 7 _ { \pm 0 . 2 4 }$ </td><td> $1 6 . 3 8 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td></td><td> $\mathbf { 0 . 3 0 9 _ { \pm 0 . 0 1 1 } }$ </td><td> $\mathbf { 0 . 0 2 2 4 { \scriptstyle \pm 0 . 0 0 1 1 } }$ </td><td> $\mathbf { 0 . 4 6 2 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 9 . 6 7 _ { \pm 0 . 1 4 } }$ </td><td> $\mathbf { 1 3 . 9 7 _ { \pm 0 . 3 1 } }$ </td></tr></table>

Table 11 40 Gaussians, $\beta = 3 .$ FKC with $K = 8 ,$ 75% total lo<sub>g</sub>-wei<sub>g</sub>ht cli<sub>pp</sub>in<sub>g</sub>. Mean over 4 seeds (10k sam<sub>p</sub>les) with standard deviation as a subscri<sub>p</sub>t; lower is better; best <sub>p</sub>er NFE in bold. ∞: diver<sub>g</sub>ed. Rows without FKC are uncli<sub>pp</sub>ed and identical <sub>across</sub> t<sub>a</sub>bl<sub>es.</sub>
<table><tr><td>Model</td><td>NFE</td><td>FKC</td><td> $\mathrm { E n e r g y } – \mathcal { W } _ { 2 }$ </td><td>MMD</td><td>TV</td><td>W1</td><td>W2</td></tr><tr><td>iSSFM</td><td>4</td><td>x</td><td> $\mathbf { 1 . 6 3 8 _ { \pm 0 . 1 0 5 } }$ </td><td> $0 . 0 2 0 7 _ { \pm 0 . 0 0 0 7 }$ </td><td> $0 . 4 9 3 _ { \pm 0 . 0 0 7 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 2 5 }$ </td><td> $1 6 . 2 2 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>iSSFM</td><td>4</td><td>✓</td><td> $6 . 5 2 8 _ { \pm 7 . 8 1 3 }$ </td><td> $\mathbf { 0 . 0 1 8 8 _ { \pm 0 . 0 0 0 9 } }$ </td><td> $\mathbf { 0 . 4 7 7 _ { \pm 0 . 0 1 0 } }$ </td><td> $\mathbf { 1 0 . 1 9 _ { \pm 0 . 1 6 } }$ </td><td> $\mathbf { 1 5 . 3 5 _ { \pm 0 . 2 8 } }$ </td></tr><tr><td>Diffusion SDE</td><td>4</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>4</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>8</td><td>x</td><td> $0 . 8 2 1 _ { \pm 0 . 0 9 9 }$ </td><td> $0 . 0 2 3 0 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 8 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 0 . 6 5 _ { \pm 0 . 1 4 }$ </td><td> $1 5 . 4 4 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>8</td><td>√</td><td> $\mathbf { 0 . 7 5 9 _ { \pm 0 . 1 3 6 } }$ </td><td> $\mathbf { 0 . 0 1 8 9 } _ { \pm 0 . 0 0 1 0 }$ </td><td> $\mathbf { 0 . 4 6 0 _ { \pm 0 . 0 0 6 } }$ </td><td> $\mathbf { 9 . 6 4 _ { \pm 0 . 1 9 } }$ </td><td> $\mathbf { 1 4 . 3 9 _ { \pm 0 . 1 6 } }$ </td></tr><tr><td>Diffusion SDE</td><td>8</td><td>x</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Diffusion SDE</td><td>8</td><td>√</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>iSSFM</td><td>12</td><td>x</td><td> $\mathbf { 0 . 4 9 6 _ { \pm 0 . 0 3 6 } }$ </td><td> $0 . 0 2 5 7 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 4 9 5 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 2 5 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 8 6 _ { \pm 0 . 3 3 }$ </td></tr><tr><td>iSSFM</td><td>12</td><td>√</td><td> $0 . 5 8 0 _ { \pm 0 . 0 5 2 }$ </td><td> $\mathbf { 0 . 0 2 0 8 _ { \pm 0 . 0 0 1 2 } }$ </td><td> $\mathbf { 0 . 4 6 7 _ { \pm 0 . 0 0 6 } }$ </td><td> $\mathbf { 9 . 7 2 _ { \pm 0 . 1 1 } }$ </td><td> ${ \bf 1 4 . 3 6 _ { \pm 0 . 2 3 } }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td>x</td><td> $8 . 4 6 \times 1 0 _ { \pm 7 . 8 3 \times 1 0 ^ { 4 } } ^ { 5 }$ </td><td> $0 . 0 4 0 7 _ { \pm 0 . 0 0 1 1 }$ </td><td> $0 . 6 1 7 _ { \pm 0 . 0 0 4 }$ </td><td> $2 2 0 . 1 1 _ { \pm 4 . 1 9 }$ </td><td> $5 8 7 . 2 2 _ { \pm 3 3 . 3 9 }$ </td></tr><tr><td>Diffusion SDE</td><td>12</td><td> $\checkmark$ </td><td> $2 . 3 1 \times 1 0 _ { \pm 2 . 3 2 \times 1 0 ^ { 5 } } ^ { 6 }$ </td><td> $0 . 0 4 8 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 7 0 9 _ { \pm 0 . 0 1 0 }$ </td><td> $9 1 3 . 8 5 _ { \pm 2 4 . 3 9 }$ </td><td> $1 4 4 7 . 2 9 _ { \pm 5 4 . 2 3 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>x</td><td> $\mathbf { 0 . 4 9 4 _ { \pm 0 . 0 8 2 } }$ </td><td> $0 . 0 2 6 7 _ { \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $1 5 . 8 4 _ { \pm 0 . 5 2 }$ </td></tr><tr><td>iSSFM</td><td>16</td><td>√</td><td> $0 . 5 7 0 _ { \pm 0 . 1 1 1 }$ </td><td> $\mathbf { 0 . 0 2 1 7 _ { \pm 0 . 0 0 0 5 } }$ </td><td> $\mathbf { 0 . 4 7 3 _ { \pm 0 . 0 0 3 } }$ </td><td> ${ \bf 9 . 9 1 _ { \pm 0 . 0 3 } }$ </td><td> $\mathbf { 1 4 . 3 9 _ { \pm 0 . 2 9 } }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>x</td><td> $1 . 6 1 \times 1 0 ^ { 4 }$  ≠4.82×103</td><td> $0 . 0 3 9 8 _ { \pm 0 . 0 0 1 7 }$ </td><td> $0 . 5 7 8 _ { \pm 0 . 0 0 7 }$ </td><td> $1 5 . 4 6 _ { \pm 0 . 5 0 }$ </td><td> $3 6 . 5 6 _ { \pm 8 . 8 2 }$ </td></tr><tr><td>Diffusion SDE</td><td>16</td><td>√</td><td> $4 . 8 7 \times 1 0 _ { \pm 1 . 5 8 \times 1 0 ^ { 4 } } ^ { \mp }$ </td><td> $0 . 0 4 4 4 _ { \pm 0 . 0 0 1 0 }$ </td><td> $0 . 5 8 9 _ { \pm 0 . 0 0 3 }$ </td><td> $2 6 . 2 4 _ { \pm 2 . 8 5 }$ </td><td> $9 5 . 3 8 _ { \pm 1 8 . 8 4 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td></td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 0 3 2 } }$ </td><td> $0 . 0 2 7 1 { _ { \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 5 0 7 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 6 6 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>iSSFM</td><td>20</td><td>× √</td><td> $0 . 4 9 5 _ { \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 0 2 2 7 _ { \pm 0 . 0 0 0 9 } }$ </td><td> $\mathbf { 0 . 4 8 0 _ { \pm 0 . 0 0 6 } }$ </td><td> $\mathbf { 1 0 . 2 3 _ { \pm 0 . 3 5 } }$ </td><td> $1 4 . 6 0 _ { \pm 0 . 4 5 }$ </td></tr><tr><td>Diffusion SDE Diffusion SDE</td><td>20</td><td>× √</td><td> $0 . 5 6 4 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 3 3 _ { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 1 . 0 4 _ { \pm 0 . 1 3 }$ </td><td> $\mathbf { 1 4 . 5 3 _ { \pm 0 . 1 5 } }$ </td></tr><tr><td></td><td>20</td><td></td><td> $0 . 5 4 5 _ { \pm 0 . 0 1 8 }$ </td><td> $0 . 0 3 4 1 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 3 5 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 6 8 _ { \pm 0 . 4 3 }$ </td><td> $1 7 . 4 2 _ { \pm 0 . 5 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>x</td><td> $0 . 4 8 6 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 0 2 8 8 _ { \pm 0 . 0 0 1 6 }$ </td><td> $0 . 5 1 0 _ { \pm 0 . 0 0 5 }$ </td><td> $1 1 . 6 7 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 0 3 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>iSSFM</td><td>24</td><td>√</td><td> $\mathbf { 0 . 3 7 9 _ { \pm 0 . 0 5 8 } }$ </td><td> $\mathbf { 0 . 0 2 3 5 _ { \pm 0 . 0 0 1 5 } }$ </td><td> $\mathbf { 0 . 4 8 1 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 1 0 . 4 3 _ { \pm 0 . 2 8 } }$ </td><td> $1 4 . 7 2 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>x</td><td> $0 . 5 2 3 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 3 5 5 _ { \pm 0 . 0 0 2 5 }$ </td><td> $0 . 5 4 0 _ { \pm 0 . 0 0 4 }$ </td><td> $1 2 . 3 9 _ { \pm 0 . 0 9 }$ </td><td> $1 6 . 1 5 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Diffusion SDE</td><td>24</td><td>V</td><td> $0 . 4 0 7 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 6 9 _ { \pm 0 . 0 0 0 7 }$ </td><td> $0 . 4 9 7 _ { \pm 0 . 0 0 1 }$ </td><td> $1 0 . 5 2 _ { \pm 0 . 2 9 }$ </td><td> $\mathbf { 1 3 . 8 8 _ { \pm 0 . 3 8 } }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>x</td><td> $0 . 4 7 6 _ { \pm 0 . 0 4 9 }$ </td><td> $0 . 0 2 9 7 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 8 1 _ { \pm 0 . 1 9 }$ </td><td> $1 6 . 0 8 _ { \pm 0 . 3 7 }$ </td></tr><tr><td>iSSFM</td><td>32</td><td>√</td><td> $\mathbf { 0 . 3 8 0 _ { \pm 0 . 0 2 2 } }$ </td><td> $\mathbf { 0 . 0 2 5 0 } _ { \pm 0 . 0 0 0 7 }$ </td><td> $\mathbf { 0 . 4 9 0 _ { \pm 0 . 0 0 5 } }$ </td><td> ${ \bf 1 0 . 8 1 _ { \pm 0 . 3 8 } }$ </td><td> $\mathbf { 1 5 . 0 5 _ { \pm 0 . 6 2 } }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td>× √</td><td> $0 . 4 7 9 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 0 3 3 2 _ { \pm 0 . 0 0 1 2 }$ </td><td> $0 . 5 3 2 _ { \pm 0 . 0 0 3 }$ </td><td> $1 2 . 4 9 _ { \pm 0 . 2 2 }$ </td><td> $1 6 . 6 1 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>Diffusion SDE</td><td>32</td><td></td><td> $0 . 3 8 4 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 0 2 6 9 _ { \pm 0 . 0 0 0 5 }$ </td><td> $0 . 4 9 9 _ { \pm 0 . 0 0 2 }$ </td><td> $1 1 . 0 9 _ { \pm 0 . 2 5 }$ </td><td> $1 5 . 0 8 _ { \pm 0 . 4 5 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>x</td><td> $0 . 4 4 1 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 3 0 7 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 2 5 _ { \pm 0 . 0 0 1 }$ </td><td> $1 2 . 2 8 _ { \pm 0 . 1 8 }$ </td><td> $1 6 . 4 6 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>iSSFM</td><td>64</td><td>√</td><td> $0 . 4 1 9 _ { \pm 0 . 0 2 8 }$ </td><td> $0 . 0 2 7 2 _ { \pm 0 . 0 0 1 0 }$ </td><td> $0 . 4 9 7 _ { \pm 0 . 0 0 3 }$ </td><td> $1 1 . 2 6 _ { \pm 0 . 0 2 }$ </td><td> $1 5 . 4 3 _ { \pm 0 . 1 8 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>x</td><td> $0 . 4 3 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 9 6 _ { \pm 0 . 0 0 1 5 }$ </td><td> $0 . 5 1 5 _ { \pm 0 . 0 0 2 }$ </td><td> $1 2 . 0 7 _ { \pm 0 . 2 4 }$ </td><td> $1 6 . 3 8 _ { \pm 0 . 4 3 }$ </td></tr><tr><td>Diffusion SDE</td><td>64</td><td>√</td><td> $\mathbf { 0 . 3 6 2 _ { \pm 0 . 0 1 9 } }$ </td><td> $\mathbf { 0 . 0 2 6 1 { \scriptstyle \pm 0 . 0 0 1 1 } }$ </td><td> $\mathbf { 0 . 4 8 7 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 1 0 . 9 1 _ { \pm 0 . 1 6 } }$ </td><td> ${ \bf 1 5 . 3 0 _ { \pm 0 . 2 3 } }$ </td></tr></table>