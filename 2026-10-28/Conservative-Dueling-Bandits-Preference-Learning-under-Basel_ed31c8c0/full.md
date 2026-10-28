# Conservative Dueling Bandits: Preference Learning under Baseline Constraints

Rohan Deb University of Illinois, Urbana Champaign

Mohammad Ghavamzadeh Qualcomm AI Research

Arindam Banerjee

University of Illinois, Urbana Champaign

<sub>r</sub>d22@illi<sub>no</sub>i<sub>s.e</sub>d<sub>u</sub>

<sub>mo</sub>h<sub>amma</sub>d<sub>.g</sub>h<sub>avamza</sub>d<sub>e</sub>h51@<sub>gma</sub>il<sub>.com</sub>

<sub>ar</sub>i<sub>n</sub>d<sub>am</sub>b@illi<sub>no</sub>i<sub>s.e</sub>d<sub>u</sub>

## Abstract

C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s</sub> l<sub>earn</sub> f<sub>rom pa</sub>i<sub>rw</sub>i<sub>se compar</sub>i<sub>sons,</sub> th<sub>e same</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>use</sub>d t<sub>o</sub> fi<sub>ne-</sub>t<sub>une</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>rom</sub> h<sub>uman pre</sub>f<sub>erences.</sub> St<sub>an</sub>d<sub>ar</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>ms exp</sub>l<sub>ore aggress</sub>i<sub>ve</sub>l<sub>y an</sub>d <sub>can s</sub>h<sub>ow</sub> <sub>users</sub> t<sub>wo poor op</sub>ti<sub>ons.</sub> W<sub>e s</sub>t<sub>u</sub>d<sub>y conserva</sub>ti<sub>ve con</sub>t<sub>ex</sub>t<sub>ua</sub>l d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s, w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>re</sub> th<sub>a</sub>t<sub>, a</sub>t every round, the cumulative score of the better displayed item stay above a (1 − α) fraction of <sub>a</sub> t<sub>rus</sub>t<sub>e</sub>d b<sub>ase</sub>li<sub>ne</sub>’<sub>s.</sub> A<sub>n</sub> it<sub>em</sub>’<sub>s score</sub> i<sub>s</sub> it<sub>s pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f b<sub>ea</sub>ti<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> it<sub>em, so</sub> th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t d<sub>epen</sub>d<sub>s on</sub>l<sub>y on u</sub>tilit<sub>y</sub> dif<sub>erences an</sub>d <sub>nee</sub>d<sub>s no a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e rewar</sub>d<sub>s.</sub> Thi<sub>s a</sub>l<sub>so c</sub>h<sub>anges</sub> h<sub>ow</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y can</sub> b<sub>e cer</sub>tifi<sub>e</sub>d<sub>: we s</sub>h<sub>ow</sub> th<sub>a</sub>t t<sub>rans</sub>f<sub>err</sub>i<sub>ng</sub> th<sub>e sa</sub>f<sub>e</sub>t<sub>y</sub> t<sub>es</sub>t <sub>o</sub>f th<sub>e conserva</sub>ti<sub>ve</sub> b<sub>an</sub>dit <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> CLUCB <sub>unc</sub>h<sub>ange</sub>d <sub>can</sub> i<sub>ncur</sub> li<sub>near regre</sub>t<sub>, even w</sub>h<sub>en</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub>’<sub>s u</sub>tiliti<sub>es are</sub> k<sub>nown exac</sub>tl<sub>y.</sub> <sup>F</sup>or <sup>li</sup>near ut<sup>ili</sup>t<sup>i</sup>es, we <sup>d</sup>eve<sup>l</sup>op MaxInP<sup>9</sup>BC, w<sup>hi</sup>c<sup>h</sup> cert<sup>ifi</sup>es scores t<sup>h</sup>roug<sup>h</sup> ut<sup>ili</sup>ty gaps an<sup>d</sup>, w<sup>h</sup>en it <sub>mus</sub>t f<sub>a</sub>ll b<sub>ac</sub>k<sub>, pa</sub>i<sub>rs</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne w</sub>ith <sub>an</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve par</sub>t<sub>ner so</sub> th<sub>a</sub>t f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s s</sub>till d<sub>r</sub>i<sub>ve</sub> l<sub>earn</sub>i<sub>ng.</sub> It i<sub>s</sub> <sub>sa</sub>f<sub>e</sub> <sub>w</sub>ith hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t <sub>a</sub>ll ti<sub>mes,</sub> <sub>w</sub>ith <sub>regre</sub>t $\tilde { \mathcal { O } } ( d \sqrt { T } / \kappa )$ <sub>p</sub>l<sub>us</sub> <sub>an</sub> <sub>a</sub>dditi<sub>ve</sub> <sub>cos</sub>t <sub>o</sub>f conservat<sup>i</sup>sm t<sup>h</sup>at <sup>d</sup>oes not grow w<sup>i</sup>t<sup>h</sup> t<sup>h</sup>e <sup>h</sup>or<sup>i</sup>zon. <sup>F</sup>or genera<sup>l</sup> ut<sup>ili</sup>ty c<sup>l</sup>asses, we <sup>d</sup>eve<sup>l</sup>op BTIGW<sup>9</sup>BC, <sub>w</sub>hi<sub>c</sub>h t<sub>urns</sub> th<sub>e guaran</sub>t<sub>ee o</sub>f <sub>an on</sub>li<sub>ne</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c-regress</sub>i<sub>on orac</sub>l<sub>e</sub> i<sub>n</sub>t<sub>o cumu</sub>l<sub>a</sub>ti<sub>ve sa</sub>f<sub>e</sub>t<sub>y cer</sub>tifi<sub>ca</sub>t<sub>es.</sub> It i<sub>s sa</sub>f<sub>e a</sub>t <sub>a</sub>ll ti<sub>mes, ma</sub>k<sub>es</sub> fi<sub>n</sub>it<sub>e</sub>l<sub>y many</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>s, an</sub>d h<sub>as su</sub>bli<sub>near regre</sub>t <sub>w</sub>h<sub>enever</sub> th<sub>e orac</sub>l<sub>e</sub>’<sub>s</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> <sub>regre</sub>t i<sub>s</sub> <sub>su</sub>bli<sub>near.</sub> I<sub>n</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s,</sub> b<sub>o</sub>th <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub>i<sub>r</sub> <sub>uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>coun</sub>t<sub>erpar</sub>t<sub>s o</sub>ft<sub>en v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>e, a</sub>t <sub>a mo</sub>d<sub>era</sub>t<sub>e cos</sub>t i<sub>n regre</sub>t<sub>.</sub>

## 1. Introduction

M<sub>any</sub> l<sub>earn</sub>i<sub>ng sys</sub>t<sub>ems</sub> th<sub>a</sub>t i<sub>n</sub>t<sub>erac</sub>t <sub>w</sub>ith <sub>peop</sub>l<sub>e canno</sub>t <sub>re</sub>l<sub>y on numer</sub>i<sub>ca</sub>l <sub>rewar</sub>d<sub>s.</sub> Wh<sub>en users are</sub> <sub>as</sub>k<sub>e</sub>d t<sub>o ra</sub>t<sub>e a searc</sub>h <sub>resu</sub>lt<sub>, a recommen</sub>d<sub>e</sub>d it<sub>em, or a p</sub>i<sub>ece o</sub>f <sub>genera</sub>t<sub>e</sub>d t<sub>ex</sub>t <sub>on an a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e sca</sub>l<sub>e,</sub> th<sub>e scores</sub> th<sub>ey g</sub>i<sub>ve are no</sub>i<sub>sy, poor</sub>l<sub>y ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>across</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>s, an</sub>d <sub>expens</sub>i<sub>ve</sub> t<sub>o co</sub>ll<sub>ec</sub>t <sub>a</sub>t <sub>sca</sub>l<sub>e.</sub> A<sub>s</sub>ki<sub>ng</sub> th<sub>e same users w</sub>hi<sub>c</sub>h <sub>o</sub>f t<sub>wo op</sub>ti<sub>ons</sub> th<sub>ey pre</sub>f<sub>er</sub> i<sub>s eas</sub>i<sub>er</sub> f<sub>or</sub> th<sub>em an</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s a more cons</sub>i<sub>s</sub>t<sub>en</sub>t si<sub>g</sub>nal (Jones et al., 2011). This has motivated a lar<sub>g</sub>e bod<sub>y</sub> of work on learnin<sub>g</sub> from <sub>p</sub>airwise <sub>p</sub>references (Bradle<sub>y</sub> and Terr<sub>y</sub>, 1952; Fürnkranz and Hüllermeier, 2010; Yue et al., 2012; Wirth et al., 2017; Ben<sub>g</sub>s et al., 2021). Its a<sub>pp</sub>lications include online evaluation and o<sub>p</sub>timization of search rankers (Joachims, 2002; Radlinski et al., 2008; Yue and Joachims, 2009; Cha<sub>p</sub>elle et al., 2012; Hofmann et al., 2016), recommender s<sub>y</sub>stems (Jones et al., 2011; Kalloori et al., 2016), human-in-the-loo<sub>p</sub> desi<sub>g</sub>n (Brochu et al., 2007; González et al., 2017; Tucker et al., 2020), and learnin<sub>g</sub> from human com<sub>p</sub>arisons (Christiano et al., 2017), which has more recentl<sub>y</sub> become a standard wa<sub>y</sub> to fine-tune lan<sub>g</sub>ua<sub>g</sub>e models, either b<sub>y</sub> reinforcement learnin<sub>g</sub> a<sub>g</sub>ainst a learned reward model (Zie<sub>g</sub>ler et al., 2020; Stiennon et al., 2020; Ou<sub>y</sub>an<sub>g</sub> et al., 2022; Bai et al., 2022) or b<sub>y</sub> o<sub>p</sub>timizin<sub>g</sub> the <sub>p</sub>olic<sub>y</sub> directl<sub>y</sub> on the <sub>p</sub>reference data (Rafailov et al., 2023).

Th<sub>e</sub> d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit <sub>pro</sub>bl<sub>em g</sub>i<sub>ves a c</sub>l<sub>ean a</sub>b<sub>s</sub>t<sub>rac</sub>ti<sub>on o</sub>f thi<sub>s</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on an</sub>d h<sub>as</sub> b<sub>een s</sub>t<sub>u</sub>di<sub>e</sub>d f<sub>or</sub> finitel<sub>y</sub> man<sub>y</sub> actions (Yue et al., 2012). In each round, the learner selects a <sub>p</sub>air of actions, <sub>p</sub>resents th<sub>em</sub> t<sub>o a user, an</sub>d <sub>o</sub>b<sub>serves on</sub>l<sub>y a no</sub>i<sub>sy</sub> bi<sub>nary ou</sub>t<sub>come</sub> i<sub>n</sub>di<sub>ca</sub>ti<sub>ng w</sub>hi<sub>c</sub>h <sub>one was pre</sub>f<sub>erre</sub>d<sub>.</sub> B<sub>ecause no s</sub>i<sub>ng</sub>l<sub>e o</sub>b<sub>serva</sub>ti<sub>on revea</sub>l<sub>s</sub> h<sub>ow goo</sub>d <sub>an ac</sub>ti<sub>on</sub> i<sub>s,</sub> th<sub>e</sub> l<sub>earner mus</sub>t i<sub>n</sub>f<sub>er qua</sub>lit<sub>y</sub> f<sub>rom</sub> <sub>pa</sub>tt<sub>erns o</sub>f <sub>w</sub>i<sub>ns an</sub>d l<sub>osses.</sub> A <sub>common way</sub> t<sub>o a</sub>dd <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> i<sub>s</sub> t<sub>o pos</sub>it <sub>a pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c c</sub>h<sub>o</sub>i<sub>ce mo</sub>d<sub>e</sub>l<sub>.</sub> Under the Bradle<sub>y</sub>–Terr<sub>y</sub>–Luce (BTL) model (Bradle<sub>y</sub> and Terr<sub>y</sub>, 1952; Hunter, 2004; Luce, 2005), <sub>eac</sub>h <sub>ac</sub>ti<sub>on</sub> h<sub>as a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>u</sub>tilit<sub>y, an</sub>d th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t <sub>one ac</sub>ti<sub>on</sub> b<sub>ea</sub>t<sub>s ano</sub>th<sub>er</sub> i<sub>s a</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> f<sub>unc</sub>ti<sub>on</sub> of their utilit<sub>y</sub> diference (cf. Assum<sub>p</sub>tion 1). The Plackett–Luce model extends this to com<sub>p</sub>arisons amon<sub>g</sub> more than two items (Zhao et al., 2020). Contextual duelin<sub>g</sub> bandits (Saha, 2021) let the <sub>u</sub>tilit<sub>y a</sub>l<sub>so</sub> d<sub>epen</sub>d <sub>on o</sub>b<sub>serve</sub>d <sub>s</sub>id<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on, suc</sub>h <sub>as a query, a user pro</sub>fil<sub>e, or a promp</sub>t<sub>.</sub> Th<sub>ey</sub> h<sub>ave</sub> l<sub>e</sub>d t<sub>o</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>w</sub>ith <sub>su</sub>bli<sub>near</sub> <sub>regre</sub>t <sub>guaran</sub>t<sub>ees</sub> f<sub>or</sub> li<sub>near</sub> <sub>an</sub>d <sub>more</sub> <sub>genera</sub>l <sub>u</sub>tilit<sub>y</sub> <sub>c</sub>l<sub>asses,</sub> th<sub>e</sub> latter t<sub>yp</sub>icall<sub>y</sub> throu<sub>g</sub>h reductions to online re<sub>g</sub>ression oracles (Sekhari et al., 2023), an a<sub>pp</sub>roach that extends to neural function classes in contextual bandits with cardinal rewards (Deb et al., 2024a). C<sub>ons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s are</sub> f<sub>ar</sub> l<sub>ess un</sub>d<sub>ers</sub>t<sub>oo</sub>d <sub>un</sub>d<sub>er pre</sub>f<sub>erence</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> F<sub>or</sub> d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s w</sub>ith k<sub>napsac</sub>k constraints, Deb et al. (2024b) show that relative feedback makes the <sub>p</sub>roblem harder than its bandit <sub>coun</sub>t<sub>erpar</sub>t<sub>: w</sub>ith<sub>ou</sub>t f<sub>ur</sub>th<sub>er assump</sub>ti<sub>ons on</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> it i<sub>s no</sub>t l<sub>earna</sub>bl<sub>e</sub> f<sub>rom a regre</sub>t<sub>-m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> p<sup>ers</sup>p<sup>ecti</sup>v<sup>e</sup>.

H<sub>owever,</sub> <sub>a</sub>ll <sub>o</sub>f thi<sub>s</sub> <sub>wor</sub>k <sub>measures</sub> <sub>per</sub>f<sub>ormance</sub> b<sub>y</sub> <sub>regre</sub>t <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> b<sub>es</sub>t <sub>compar</sub>i<sub>son</sub> i<sub>n</sub> hi<sub>n</sub>d<sub>s</sub>i<sub>g</sub>ht<sub>.</sub> Th<sub>e resu</sub>lti<sub>ng a</sub>l<sub>gor</sub>ith<sub>ms</sub> th<sub>ere</sub>f<sub>ore exp</sub>l<sub>ore as aggress</sub>i<sub>ve</sub>l<sub>y as</sub> th<sub>ey can.</sub> Thi<sub>s s</sub>it<sub>s uneas</sub>il<sub>y w</sub>ith h<sub>ow</sub> <sub>pre</sub>f<sub>erence-</sub>b<sub>ase</sub>d l<sub>earners are</sub> d<sub>ep</sub>l<sub>oye</sub>d i<sub>n prac</sub>ti<sub>ce,</sub> b<sub>ecause every compar</sub>i<sub>son</sub> i<sub>s s</sub>h<sub>own</sub> t<sub>o a rea</sub>l <sub>person.</sub> A <sub>user</sub> <sub>w</sub>h<sub>o</sub> i<sub>s</sub> <sub>repea</sub>t<sub>e</sub>dl<sub>y</sub> <sub>o</sub>f<sub>ere</sub>d t<sub>wo</sub> <sub>unappea</sub>li<sub>ng</sub> <sub>recommen</sub>d<sub>a</sub>ti<sub>ons,</sub> <sub>or</sub> t<sub>wo</sub> <sub>un</sub>h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l <sub>answers,</sub> h<sub>as</sub> littl<sub>e reason</sub> t<sub>o</sub> k<sub>eep</sub> t<sub>rus</sub>ti<sub>ng</sub> th<sub>e sys</sub>t<sub>em,</sub> h<sub>owever</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> th<sub>ose compar</sub>i<sub>sons are</sub> f<sub>or</sub> th<sub>e</sub> l<sub>earner.</sub> M<sub>oreover,</sub> <sub>a</sub> <sub>new</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>sys</sub>t<sub>em</sub> <sub>rare</sub>l<sub>y</sub> <sub>s</sub>t<sub>ar</sub>t<sub>s</sub> f<sub>rom</sub> <sub>scra</sub>t<sub>c</sub>h<sub>.</sub> It <sub>usua</sub>ll<sub>y</sub> <sub>rep</sub>l<sub>aces</sub> <sub>or</sub> <sub>augmen</sub>t<sub>s</sub> <sub>an</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>po</sub>li<sub>cy</sub> <sub>w</sub>h<sub>ose</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> i<sub>s</sub> <sub>cons</sub>id<sub>ere</sub>d <sub>accep</sub>t<sub>a</sub>bl<sub>e.</sub> I<sub>n</sub> <sub>suc</sub>h <sub>se</sub>tti<sub>ngs,</sub> it i<sub>s</sub> <sub>na</sub>t<sub>ura</sub>l t<sub>o</sub> <sub>requ</sub>i<sub>re</sub> th<sub>a</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms avo</sub>id <sub>aggress</sub>i<sub>ve exp</sub>l<sub>ora</sub>ti<sub>on, an</sub>d <sub>ensure</sub> th<sub>a</sub>t th<sub>e</sub> d<sub>ep</sub>l<sub>oye</sub>d <sub>po</sub>li<sub>cy per</sub>f<sub>orms a</sub>t l<sub>eas</sub>t <sub>compara</sub>bl<sub>y</sub> t<sub>o</sub> <sub>an</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng,</sub> t<sub>rus</sub>t<sub>e</sub>d b<sub>ase</sub>li<sub>ne</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy.</sub> C<sub>oncre</sub>t<sub>e</sub>l<sub>y,</sub> <sub>we</sub> <sub>as</sub>k th<sub>a</sub>t <sub>a</sub>t <sub>every</sub> <sub>s</sub>t<sub>ep,</sub> th<sub>e</sub> better of the two displayed items be, in aggregate, at least a (1 − α) fraction as good as what the b<sub>ase</sub>li<sub>ne wou</sub>ld h<sub>ave o</sub>f<sub>ere</sub>d<sub>.</sub>

F<sub>or</sub> b<sub>an</sub>dit<sub>s w</sub>ith <sub>numer</sub>i<sub>ca</sub>l <sub>rewar</sub>d<sub>s,</sub> thi<sub>s</sub> ki<sub>n</sub>d <sub>o</sub>f <sub>sa</sub>f<sub>e exp</sub>l<sub>ora</sub>ti<sub>on</sub> i<sub>s cap</sub>t<sub>ure</sub>d b<sub>y conserva</sub>ti<sub>ve</sub> b<sub>an</sub>dit<sub>s</sub> (Wu et al., 2016; Kazerouni et al., 2017; Garcelon et al., 2020a). Conservative al<sub>g</sub>orithms are now available for linear rewards under both o<sub>p</sub>timism and Thom<sub>p</sub>son sam<sub>p</sub>lin<sub>g</sub> (Kazerouni et al., 2017; Deb et al., 2025b), and for <sub>g</sub>eneral reward classes accessed throu<sub>g</sub>h online re<sub>g</sub>ression oracles (Deb et al., 2025a). These methods kee<sub>p</sub> a bud<sub>g</sub>et of <sub>p</sub>ermissible loss relative to the baseline: the<sub>y</sub> l<sub>ower-</sub>b<sub>oun</sub>d th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve rewar</sub>d <sub>o</sub>f th<sub>e ac</sub>ti<sub>ons p</sub>l<sub>aye</sub>d <sub>so</sub> f<sub>ar, cre</sub>dit th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub>’<sub>s rewar</sub>d <sub>on roun</sub>d<sub>s</sub> <sub>w</sub>h<sub>ere</sub> it i<sub>s</sub> <sub>p</sub>l<sub>aye</sub>d<sub>,</sub> <sub>an</sub>d <sub>exp</sub>l<sub>ore</sub> <sub>on</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> thi<sub>s</sub> b<sub>oun</sub>d <sub>s</sub>t<sub>ays</sub> <sub>a</sub>b<sub>ove</sub> th<sub>e</sub> t<sub>arge</sub>t<sub>.</sub> Th<sub>a</sub>t b<sub>oo</sub>kk<sub>eep</sub>i<sub>ng</sub> <sub>requ</sub>i<sub>res</sub> k<sub>now</sub>i<sub>ng</sub> h<sub>ow goo</sub>d th<sub>e</sub> l<sub>earner</sub>’<sub>s ac</sub>ti<sub>ons are</sub> i<sub>n a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> t<sub>erms, an</sub>d <sub>pre</sub>f<sub>erence</sub> f<sub>ee</sub>db<sub>ac</sub>k d<sub>oes no</sub>t <sub>prov</sub>id<sub>e</sub> it<sub>.</sub> A <sub>compar</sub>i<sub>son revea</sub>l<sub>s on</sub>l<sub>y</sub> th<sub>e</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo u</sub>tiliti<sub>es, so a con</sub>fid<sub>ence se</sub>t b<sub>u</sub>ilt f<sub>rom compar</sub>i<sub>sons can</sub> l<sub>eave</sub> th<sub>e overa</sub>ll l<sub>eve</sub>l <sub>o</sub>f <sub>u</sub>tilit<sub>y uncer</sub>t<sub>a</sub>i<sub>n no ma</sub>tt<sub>er</sub> h<sub>ow muc</sub>h d<sub>a</sub>t<sub>a arr</sub>i<sub>ves.</sub> A l<sub>ower</sub> b<sub>oun</sub>d <sub>on cumu</sub>l<sub>a</sub>ti<sub>ve u</sub>tilit<sub>y compu</sub>t<sub>e</sub>d f<sub>rom suc</sub>h <sub>a se</sub>t <sub>may</sub> th<sub>en s</sub>t<sub>ay</sub> l<sub>oose</sub> i<sub>n</sub>d<sub>e</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> <sub>we s</sub>h<sub>ow</sub> th<sub>a</sub>t t<sub>rans</sub>f<sub>err</sub>i<sub>ng</sub> CLUCB’<sub>s sa</sub>f<sub>e</sub>t<sub>y</sub> t<sub>es</sub>t <sub>unc</sub>h<sub>ange</sub>d <sub>can</sub> i<sub>ncur</sub> li<sub>near regre</sub>t<sub>, even w</sub>h<sub>en</sub> th<sub>e</sub> baseline’s utilities are known exactl<sub>y</sub>. As with kna<sub>p</sub>sack constraints (Deb et al., 2024b), relative f<sub>ee</sub>db<sub>ac</sub>k <sub>c</sub>h<sub>anges w</sub>h<sub>a</sub>t <sub>a cons</sub>t<sub>ra</sub>i<sub>ne</sub>d l<sub>earner can cer</sub>tif<sub>y.</sub> Thi<sub>s s</sub>h<sub>apes</sub> b<sub>o</sub>th th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>an</sub>d h<sub>ow</sub> it i<sub>s en</sub>f<sub>orce</sub>d<sub>.</sub> W<sub>e measure eac</sub>h it<sub>em</sub> b<sub>y</sub> it<sub>s pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f b<sub>ea</sub>ti<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> it<sub>em, w</sub>hi<sub>c</sub>h d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> <sub>u</sub>tilit<sub>y</sub> dif<sub>erences,</sub> <sub>an</sub>d <sub>we</sub> <sub>cer</sub>tif<sub>y</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> th<sub>roug</sub>h <sub>u</sub>tilit<sub>y</sub> <sub>gaps</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>payo</sub>f<sub>s.</sub> A <sub>secon</sub>d difi<sub>cu</sub>lt<sub>y ar</sub>i<sub>ses w</sub>h<sub>en</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>runs ou</sub>t<sub>.</sub> A <sub>conserva</sub>ti<sub>ve</sub> b<sub>an</sub>dit th<sub>en p</sub>l<sub>ays</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne,</sub> b<sub>u</sub>t th<sub>e</sub> d<sub>ue</sub>li<sub>ng</sub> <sub>ana</sub>l<sub>ogue,</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>aga</sub>i<sub>ns</sub>t it<sub>se</sub>lf<sub>,</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>a</sub> f<sub>a</sub>i<sub>r</sub> <sub>co</sub>i<sub>n</sub> fli<sub>p</sub> <sub>an</sub>d <sub>no</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on.</sub> A l<sub>earner</sub> th<sub>a</sub>t f<sub>a</sub>ll<sub>s</sub> b<sub>ac</sub>k thi<sub>s way s</sub>t<sub>ops</sub> l<sub>earn</sub>i<sub>ng exac</sub>tl<sub>y w</sub>h<sub>en</sub> it i<sub>s uncer</sub>t<sub>a</sub>i<sub>n enoug</sub>h t<sub>o</sub> <sub>nee</sub>d th<sub>e</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>.</sub> B<sub>ecause</sub> <sub>our</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>scores</sub> <sub>a</sub> <sub>pa</sub>i<sub>r</sub> b<sub>y</sub> it<sub>s</sub> b<sub>e</sub>tt<sub>er</sub> it<sub>em,</sub> <sub>any</sub> <sub>pa</sub>i<sub>r</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>s</sub> <sub>sa</sub>f<sub>e</sub> <sub>w</sub>h<sub>a</sub>t<sub>ever</sub> it<sub>s</sub> <sub>par</sub>t<sub>ner.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>pa</sub>i<sub>r</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>w</sub>ith <sub>an</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> <sub>par</sub>t<sub>ner,</sub> <sub>so</sub> th<sub>a</sub>t f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s s</sub>till <sub>s</sub>h<sub>arpen</sub> th<sub>e cer</sub>tifi<sub>ca</sub>t<sub>es</sub> th<sub>a</sub>t <sub>govern</sub> f<sub>u</sub>t<sub>ure exp</sub>l<sub>ora</sub>ti<sub>on.</sub>

In this work, we study Preference Learning under Baseline Constraints. Our technical contributions <sub>are as summar</sub>i<sub>ze</sub>d b<sub>e</sub>l<sub>ow.</sub>

1. Framework: We formulate conservative contextual dueling bandits under Bradley–Terry f<sub>ee</sub>db<sub>ac</sub>k<sub>,</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> th<sub>e</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>pre</sub>f<sub>erence</sub> <sub>score</sub> <sub>o</sub>f th<sub>e</sub> b<sub>e</sub>tt<sub>er</sub> di<sub>sp</sub>l<sub>aye</sub>d it<sub>em</sub> t<sub>o</sub> <sub>rema</sub>i<sub>n</sub> <sub>a</sub>b<sub>ove</sub> a $( 1 - \alpha )$ f<sub>rac</sub>ti<sub>on o</sub>f th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub>’<sub>s a</sub>t <sub>every pre</sub>fi<sub>x.</sub> Thi<sub>s score-</sub>b<sub>ase</sub>d <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>s</sub> i<sub>nvar</sub>i<sub>an</sub>t t<sub>o</sub> <sub>u</sub>tilit<sub>y s</sub>hift<sub>s an</sub>d <sub>requ</sub>i<sub>res no a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e rewar</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons.</sub>

2. Linear utilities: We show that a direct CLUCB transfer can incur linear regret and develop MaxInP<sup>9</sup>BC, w<sup>hi</sup>c<sup>h</sup> cert<sup>ifi</sup>es scores t<sup>h</sup>roug<sup>h</sup> ut<sup>ili</sup>ty gaps an<sup>d</sup> uses <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>ve <sup>b</sup>ase<sup>li</sup>ne <sup>f</sup>a<sup>llb</sup>ac<sup>k</sup>s. W<sub>e</sub> <sub>prove</sub> <sub>any</sub>ti<sub>me</sub> hi<sub>g</sub>h<sub>-pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> <sub>an</sub>d <sub>regre</sub>t $\tilde { \mathcal { O } } \big ( d \sqrt { T } / \kappa + \Delta _ { h } d ^ { 2 } / ( \kappa ^ { 2 } \alpha ^ { 2 } \underline { { s } } ^ { 2 } ) \big )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\underline { { s } } = \sigma \big ( - \Delta _ { h } \big )$ <sub>.</sub> Th<sub>e a</sub>dditi<sub>ve conserva</sub>ti<sub>sm</sub> t<sub>erm</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> h<sub>or</sub>i<sub>zon, an</sub>d k<sub>nown score</sub> b<sub>oun</sub>d<sub>s su</sub>fi<sub>ce</sub> i<sub>n p</sub>l<sub>ace o</sub>f <sub>exac</sub>t b<sub>ase</sub>li<sub>ne scores.</sub>

3. General utility functions: We develop BTIGW9BC, which converts online logistic-regression <sub>guaran</sub>t<sub>ees</sub> i<sub>n</sub>t<sub>o cumu</sub>l<sub>a</sub>ti<sub>ve score cer</sub>tifi<sub>ca</sub>t<sub>es an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>e orac</sub>l<sub>e on</sub> b<sub>o</sub>th <sub>can</sub>did<sub>a</sub>t<sub>e an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>compar</sub>i<sub>sons.</sub> F<sub>or</sub> <sub>rea</sub>li<sub>za</sub>bl<sub>e</sub> <sub>c</sub>l<sub>asses</sub> <sub>w</sub>ith <sub>orac</sub>l<sub>e</sub> <sub>regre</sub>t $\mathcal { R } _ { \mathrm { l o g } } ( T ) = o ( T )$ , w<sup>e</sup> p<sup>ro</sup>v<sup>e</sup> <sub>any</sub>ti<sub>me</sub> hi<sub>g</sub>h<sub>-pro</sub>b<sub>a</sub>bilit<sub>y sa</sub>f<sub>e</sub>t<sub>y an</sub>d <sub>regre</sub>t $\tilde { \mathcal { O } } ( \sqrt { K T [ \mathcal { R } _ { \log } ( T ) / \kappa + 1 / \kappa ^ { 2 } ] } + \Delta _ { h } K [ \mathcal { R } _ { \log } ( T ) / \kappa +$ $1 / \kappa ^ { 2 } ] / ( \alpha ^ { 2 } \underline { { s } } ^ { 2 } ) \big )$ .

<sup>4</sup>. Experiments: <sup>O</sup>n synt<sup>h</sup>et<sup>i</sup>c <sup>li</sup>near an<sup>d</sup> genera<sup>l</sup>-ut<sup>ili</sup>ty <sup>i</sup>nstances, MaxInP<sup>9</sup>BC an<sup>d</sup> BTIGW<sup>9</sup>BC <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>a</sub>t <sub>every</sub> <sub>roun</sub>d<sub>.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>coun</sub>t<sub>erpar</sub>t<sub>s</sub> <sub>v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>e</sub> it i<sub>n</sub> <sub>ear</sub>l<sub>y</sub> <sub>roun</sub>d<sub>s</sub> i<sub>n</sub> <sub>near</sub>l<sub>y</sub> <sub>a</sub>ll <sub>se</sub>tti<sub>ngs.</sub> C<sub>onserva</sub>ti<sub>sm</sub> <sub>cos</sub>t<sub>s</sub> <sub>some</sub> <sub>ex</sub>t<sub>ra</sub> <sub>regre</sub>t <sub>over</sub> th<sub>e</sub> <sub>uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d l<sub>earners,</sub> <sub>concen</sub>t<sub>ra</sub>t<sub>e</sub>d <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>s</sub> ti<sub>g</sub>ht<sub>,</sub> b<sub>u</sub>t f<sub>ar</sub> l<sub>ess</sub> th<sub>an</sub> th<sub>e</sub> li<sub>near</sub> <sub>regre</sub>t <sub>o</sub>f <sub>a</sub>l<sub>ways</sub> <sub>p</sub>l<sub>ay</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub>

## 2. Problem Formulation

We consider a stochastic Contextual Duelin<sub>g</sub> Bandit (CDB) <sub>p</sub>roblem. At each round $t \in [ T ]$ <sub>,</sub> th<sub>e</sub> l<sub>earner</sub> i<sub>s presen</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>a con</sub>t<sub>ex</sub>t <sub>se</sub>t $S _ { t } = \{ \mathbf { x } _ { t , 1 } , \dotsc , \mathbf { x } _ { t , K } \} \subset \mathcal { D }$ of K items (or arms), where <sub>eac</sub>h $\mathbf { x } _ { t , i }$ i<sub>s</sub> d<sub>rawn</sub> i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> f<sub>rom an un</sub>k<sub>nown</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\mathcal { P } _ { \mathcal { D } }$ <sub>over a</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on space</sub> $\mathcal { D } \subset \mathbb { R } ^ { d }$ <sub>.</sub> Th<sub>e</sub> <sub>agen</sub>t <sub>se</sub>l<sub>ec</sub>t<sub>s an or</sub>d<sub>ere</sub>d <sub>pa</sub>i<sub>r</sub> $( \mathbf { x } _ { t } , \mathbf { y } _ { t } ) \in S _ { t } \times S _ { t }$ (re<sub>p</sub>etitions are allowed), and receives nois<sub>y</sub> binar<sub>y</sub> f<sub>ee</sub>db<sub>ac</sub>k $o _ { t } \in \{ 0 , 1 \}$ i<sub>n</sub>di<sub>ca</sub>ti<sub>ng w</sub>hi<sub>c</sub>h it<sub>em</sub> i<sub>s pre</sub>f<sub>erre</sub>d<sub>.</sub> I<sub>n</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>o</sub>f LLM<sub>s,</sub> $\left( \mathbf { x } _ { t } , \mathbf { y } _ { t } \right)$ corres<sub>p</sub>on<sup>d</sup> t<sub>o</sub> th<sub>e</sub> t<sub>wo responses s</sub>h<sub>own</sub> t<sub>o</sub> th<sub>e user, an</sub>d $o _ { t }$ t<sub>o</sub> th<sub>e user</sub>’<sub>s recor</sub>d<sub>e</sub>d <sub>pre</sub>f<sub>erence.</sub> L<sub>e</sub>t $\mathcal { H } _ { t } ^ { - }$ d<sub>eno</sub>t<sub>e a</sub>ll i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> b<sub>e</sub>f<sub>ore</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> $o _ { t } .$ <sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>a</sub>ll <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d <sub>roun</sub>d<sub>s,</sub> th<sub>e</sub> <sub>curren</sub>t contexts and baseline information, and the selected <sub>p</sub>air. As in Saha (2021); Ben<sub>g</sub>s et al. (2022); Li et al. (2024), we assume <sub>p</sub>references are <sub>g</sub>enerated b<sub>y</sub> a latent utilit<sub>y</sub> function $f : \mathcal { D }  \mathbb { R }$ <sub>,</sub> th<sub>roug</sub>h the Bradle<sub>y</sub>–Terr<sub>y</sub>–Luce (BTL) model.

Assumption 1 (BTL Model) There exists a utilityfunction $f : \mathcal { D }  \mathbb { R }$ <sub>suc</sub>h th<sub>a</sub>t th<sub>e con</sub>diti<sub>ona</sub>l preference probability satisfies

$$
\mathbb { P } ( o _ { t } = 1 | \mathcal { H } _ { t } ^ { - } ) = \frac { \exp ( f ( \mathbf { x } _ { t } ) ) } { \exp ( f ( \mathbf { x } _ { t } ) ) + \exp ( f ( \mathbf { y } _ { t } ) ) } = \sigma \big ( f ( \mathbf { x } _ { t } ) - f ( \mathbf { y } _ { t } ) \big ) ,\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> $\sigma ( z ) = 1 / ( 1 + e ^ { - z } )$ is the sigmoid function.

Two structural <sub>p</sub>ro<sub>p</sub>erties of (1) will be useful. The com<sub>p</sub>arison law de<sub>p</sub>ends onl<sub>y</sub> on utilit<sub>y</sub> diferences <sub>an</sub>d i<sub>s</sub> i<sub>nvar</sub>i<sub>an</sub>t t<sub>o a</sub>ddi<sub>ng a cons</sub>t<sub>an</sub>t t<sub>o every u</sub>tilit<sub>y.</sub> A <sub>se</sub>lf<sub>-</sub>d<sub>ue</sub>l h<sub>as success pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 / 2$ <sub>, so</sub> it<sub>s</sub> <sub>ou</sub>t<sub>come prov</sub>id<sub>es no</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t $f .$

Assumption 2 (Bounded utility range) There is a known constant $B > 0$ <sub>suc</sub>h th<sub>a</sub>t th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> diference $| f ( \mathbf { x } ) - f ( \mathbf { y } ) | \leq B$ for all $\mathbf { x } , \mathbf { y } \in { \mathcal { D } }$ <sub>.</sub> I<sub>n</sub> th<sub>e</sub> li<sub>near case</sub> $f ( \mathbf { x } ) = \langle \theta ^ { \star } , \mathbf { x } \rangle$ <sub>w</sub>ith $\| \theta ^ { \star } \| _ { 2 } \leq S$ <sub>an</sub>d $\| \mathbf { x } \| _ { 2 } \leq L$ , one ma<sub>y</sub> ta<sup>k</sup>e $B = 2 L S$ . We define the minimum slope of the logistic link on $[ - B , B ]$

$$
\kappa : = \operatorname* { i n f } _ { | z | \leq B } \sigma ^ { \prime } ( z ) = \sigma ( B ) \big ( 1 - \sigma ( B ) \big ) > 0 .
$$

L<sub>e</sub>t $\mathbf { x } _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { S } _ { t } } f ( \mathbf { x } )$ d<sub>eno</sub>t<sub>e a</sub> b<sub>es</sub>t it<sub>em</sub> i<sub>n roun</sub>d $t ,$ ti<sub>es</sub> b<sub>ro</sub>k<sub>en</sub> b<sub>y a</sub> fi<sub>xe</sub>d <sub>ru</sub>l<sub>e.</sub> N<sub>ex</sub>t <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> the re<sub>g</sub>ret of an al<sub>g</sub>orithm followin<sub>g</sub> Saha (2021); Verma et al. (2025). It ca<sub>p</sub>tures the cumulative <sub>g</sub>a<sub>p</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> b<sub>es</sub>t <sub>poss</sub>ibl<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> <sub>an</sub>d th<sub>e</sub> <sub>average</sub> <sub>u</sub>tilit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> l<sub>earner</sub>’<sub>s</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> <sub>over</sub> $T$ <sub>roun</sub>d<sub>s.</sub>

Regret. Suppose an algorithm chooses the pair $\left( \mathbf { x } _ { t } , \mathbf { y } _ { t } \right)$ <sub>a</sub>t ti<sub>me</sub> $t \in [ T ]$ <sub>.</sub> Th<sub>e</sub> i<sub>ns</sub>t<sub>an</sub>t<sub>aneous regre</sub>t incurred at round t is the gap between the utility of the best item and the average utility of the <sub>compare</sub>d <sub>pa</sub>i<sub>r,</sub> d<sub>eno</sub>t<sub>e</sub>d $\begin{array} { r } { R _ { t } : = f ( \mathbf { x } _ { t } ^ { \star } ) - \frac { 1 } { 2 } \big ( f ( \mathbf { x } _ { t } ) + f ( \mathbf { y } _ { t } ) \big ) } \end{array}$ , and the cumulative T-round regret is

$$
\mathtt { R e g } ( T ) : = \sum _ { t = 1 } ^ { T } R _ { t } = \sum _ { t = 1 } ^ { T } \left( f ( \mathbf { x } _ { t } ^ { \star } ) - { \textstyle \frac { 1 } { 2 } } \big ( f ( \mathbf { x } _ { t } ) + f ( \mathbf { y } _ { t } ) \big ) \right) .\tag{2}
$$

N<sub>ex</sub>t <sub>we</sub> d<sub>e</sub>fi<sub>ne a per</sub>f<sub>ormance cons</sub>t<sub>ra</sub>i<sub>n</sub>t th<sub>a</sub>t th<sub>e a</sub>l<sub>gor</sub>ith<sub>m mus</sub>t <sub>sa</sub>ti<sub>s</sub>f<sub>y so</sub> th<sub>a</sub>t th<sub>e overa</sub>ll <sub>user</sub> <sub>exper</sub>i<sub>ence s</sub>t<sub>ays a</sub>b<sub>ove a</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>se</sub>t b<sub>y an ex</sub>i<sub>s</sub>ti<sub>ng</sub> t<sub>rus</sub>t<sub>e</sub>d b<sub>ase</sub>li<sub>ne po</sub>li<sub>cy.</sub>

Definition 2.1 (Baseline Performance Constraint) The preference score ofan item $\mathbf { x } \in S _ { t }$ i<sub>s</sub> it<sub>s</sub> probability of beating the best item of the round: $s _ { t } ( \mathbf { x } ) : = \mathbb { P } \{ \mathbf { x } \succ \mathbf { x } _ { t } ^ { \star } \} = \sigma \big ( f ( \mathbf { x } ) - f ( \mathbf { x } _ { t } ^ { \star } ) \big ) \in ( 0 , \frac { 1 } { 2 } ]$ S<sub>uppose</sub> th<sub>e</sub> <sub>agen</sub>t h<sub>as</sub> <sub>access</sub> t<sub>o</sub> b<sub>ase</sub>li<sub>ne</sub> it<sub>ems</sub> $\mathbf { x } _ { b _ { t } } \in S _ { t } , t \in [ T ]$ <sub>, w</sub>ith k<sub>nown scores</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } )$ <sub>.</sub> Gi<sub>ven a</sub> <sub>r</sub>i<sub>s</sub>k <sub>parame</sub>t<sub>er</sub> $\alpha \in ( 0 , 1 )$ , a sequence of displayed pairs $( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) , i \in [ t ]$ is α-conservative $i f$

$$
\sum _ { i = 1 } ^ { t } \operatorname* { m a x } \left\{ s _ { i } ( \mathbf { x } _ { i } ) , s _ { i } ( \mathbf { y } _ { i } ) \right\} \geq ( 1 - \alpha ) \sum _ { i = 1 } ^ { t } s _ { i } ( \mathbf { x } _ { b _ { i } } ) , \quad f o r a l l t \in [ T ] .\tag{3}
$$

Assumption 3 (Baseline Gap and Bounds) Let $\Delta _ { t , b _ { t } } : = f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } _ { b _ { t } } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>gap.</sub> Th<sub>ere are</sub> k<sub>nown cons</sub>t<sub>an</sub>t<sub>s</sub> $0 \le \Delta _ { l } \le \Delta _ { h } \le B$ <sub>w</sub>ith $\Delta _ { l } \leq \Delta _ { t , b _ { t } } \leq \Delta _ { h }$ for all $t \in [ T ] ,$ <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y</sub>

$$
\begin{array} { r } { \underline { { s } } : = \sigma ( - \Delta _ { h } ) \ \le \ s _ { t } ( \mathbf x _ { b _ { t } } ) \ \le \ \sigma ( - \Delta _ { l } ) = : \bar { s } \ \le \ \frac 1 2 . } \end{array}
$$

Remark 1 (Why the maximum and why scores) The constraint in (3) compares cumulative preference probabilities along the learner’s trajectory, rather than realized comparison labels. Using the maximum ensures that any pair containing the baseline has score at least that of the baseline, allowing its partner to be selectedfor information acquisition.

## 3. Linear Utilities: Conservative Maximum-Informative-Pair

Assumption 4 (Linear Utility) There is an unknown $\theta ^ { \star } \in \mathbb { R } ^ { d }$ <sub>w</sub>ith $\| \theta ^ { \star } \| _ { 2 } \leq S$ <sub>suc</sub>h th<sub>a</sub>t $f ( \mathbf { x } ) =$ $\langle { \theta ^ { \star } , \bf { x } } \rangle$ <sub>,</sub> <sub>an</sub>d $\| \mathbf { x } \| _ { 2 } \leq L , f o r$ <sup>e</sup>v<sup>er</sup>y $\mathbf { x } \in \mathcal { D }$ <sub>.</sub> Th<sub>e</sub> l<sub>earner op</sub>ti<sub>m</sub>i<sub>ses over</sub> $\Theta = \{ \theta : \| \theta \| _ { 2 } \leq S \}$ , so A<sub>ssump</sub>ti<sub>on</sub> 2 h<sub>o</sub>ld<sub>s w</sub>ith $B = 2 L S$ <sub>an</sub>d $\kappa = \sigma ( B ) ( 1 - \sigma ( B ) )$ .

In the unconstrained settin<sub>g</sub>, MaxInP (Saha, 2021) maintains a set of items that ma<sub>y</sub> still be o<sub>p</sub>timal <sub>an</sub>d di<sub>sp</sub>l<sub>ays</sub> th<sub>e mos</sub>t <sub>uncer</sub>t<sub>a</sub>i<sub>n pa</sub>i<sub>r among</sub> th<sub>em.</sub> S<sub>uc</sub>h <sub>a pa</sub>i<sub>r</sub> i<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve,</sub> b<sub>u</sub>t <sub>no</sub>thi<sub>ng preven</sub>t<sub>s</sub> <sup>b</sup>ot<sup>h</sup> o<sup>f i</sup>ts <sup>i</sup>tems <sup>f</sup>rom <sup>b</sup>e<sup>i</sup>ng poor. <sup>W</sup>e t<sup>h</sup>ere<sup>f</sup>ore <sup>d</sup>es<sup>i</sup>gn MaxInP<sup>9</sup>BC aroun<sup>d</sup> a sa<sup>f</sup>ety <sup>b</sup>u<sup>d</sup>get. <sup>Th</sup>e <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> fit<sub>s</sub> $\theta ^ { \star }$ b<sub>y</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub>d <sub>max</sub>i<sub>mum</sub> lik<sub>e</sub>lih<sub>oo</sub>d <sub>on</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>compar</sub>i<sub>sons</sub> <sub>an</sub>d <sub>uses</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng con</sub>fid<sub>ence e</sub>lli<sub>pso</sub>id t<sub>o</sub> id<sub>en</sub>tif<sub>y prom</sub>i<sub>s</sub>i<sub>ng</sub> it<sub>ems.</sub> It th<sub>en proposes</sub> th<sub>e mos</sub>t i<sub>n</sub>f<sub>orma</sub>ti<sub>ve pa</sub>i<sub>r</sub> <sub>an</sub>d di<sub>sp</sub>l<sub>ays</sub> it <sub>on</sub>l<sub>y</sub> if th<sub>e</sub> b<sub>ase</sub>li<sub>ne cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>rema</sub>i<sub>ns cer</sub>tifi<sub>a</sub>bl<sub>y sa</sub>ti<sub>s</sub>fi<sub>e</sub>d<sub>.</sub> Oth<sub>erw</sub>i<sub>se,</sub> it f<sub>a</sub>ll<sub>s</sub> b<sub>ac</sub>k t<sub>o</sub> <sub>a</sub> <sub>pa</sub>i<sub>r</sub> th<sub>a</sub>t <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> it<sub>em</sub> <sub>an</sub>d i<sub>s</sub> <sub>s</sub>till i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> f<sub>or</sub> l<sub>earn</sub>i<sub>ng.</sub>

Estimator. Each round the learner is given $\boldsymbol { S } _ { t } ,$ th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> it<sub>em</sub> $\mathbf { x } _ { b _ { t } } \in S _ { t }$ <sub>an</sub>d it<sub>s score</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } )$ Th<sub>e</sub> l<sub>earner</sub> di<sub>sp</sub>l<sub>ays</sub> $( { \bf x } _ { t } , { \bf y } _ { t } ) \in S _ { t } \times S _ { t }$ <sub>, an</sub>d <sub>o</sub>b<sub>serves</sub> $O t$ <sub>.</sub> W<sub>r</sub>it<sub>e</sub> $\mathbf z _ { t } : = \mathbf x _ { t } - \mathbf y _ { t }$ f<sub>or</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d com<sub>p</sub>ar<sup>i</sup>son vector, so $\| \mathbf { z } _ { t } \| _ { 2 } \leq 2 L$ <sub>.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
V _ { t } = \lambda I _ { d } + \sum _ { i < t } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } ,\tag{4}
$$

$$
\displaystyle \hat { \theta } _ { t } \in \mathop { \arg \operatorname* { m i n } } _ { \theta \in \Theta } \Big \{ \sum _ { i < t } \big [ \log ( 1 + e ^ { \mathbf { z } _ { i } ^ { \top } \theta } ) - o _ { i } \mathbf { z } _ { i } ^ { \top } \theta \big ] + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } \Big \} ,\tag{5}
$$

<sub>w</sub>ith $\lambda > 0 ;$ the objective is strongly convex, so $\widehat { \theta } _ { t }$ i<sub>s un</sub>i<sub>que.</sub> A <sub>se</sub>lf<sub>-</sub>d<sub>ue</sub>l h<sub>as</sub> $\mathbf { z } _ { t } = 0$ <sub>an</sub>d <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> the constant log 2 to the objective, leaving the design matrix and the minimizer unchanged. For $\delta \in ( 0 , 1 )$ , set

$$
\beta _ { t } = \frac { 1 } { \kappa } \Bigl [ \textstyle { \frac { 1 } { 2 } } \sqrt { d \log \left( 1 + \frac { 4 L ^ { 2 } ( t - 1 ) } { \lambda d } \right) + 2 \log \frac { 1 } { \delta } } + \sqrt { \lambda } S \Bigr ] .\tag{6}
$$

B<sub>o</sub>th th<sub>e</sub> it<sub>em se</sub>l<sub>ec</sub>ti<sub>on an</sub>d th<sub>e sa</sub>f<sub>e</sub>t<sub>y cer</sub>tifi<sub>ca</sub>t<sub>e re</sub>l<sub>y on</sub> $\theta ^ { \star }$ l<sub>y</sub>i<sub>ng</sub> i<sub>n</sub> th<sub>e e</sub>lli<sub>pso</sub>id <sub>aroun</sub>d $\widehat { \theta } _ { t }$ <sup>at ever</sup>y round. The following lemma guarantees this with high probability, uniformly over t.

Lemma 3.1 (Anytime confidence ellipsoid) Under Assumptions 1 and 4, $\mathbb { P } ( \mathcal { E } ) \geq 1 - \delta$ for the event $\mathcal { E } : = \{ \| \hat { \theta } _ { t } - \theta ^ { \star } \| _ { V _ { t } } \leq \beta _ { t } , \forall t \geq 1 \}$ .

Proof sketch. The residuals $o _ { t } - \sigma ( \mathbf { z } _ { t } ^ { \top } \theta ^ { \star } )$ <sub>are con</sub>diti<sub>ona</sub>ll<sub>y</sub> $\scriptstyle { \frac { 1 } { 2 } } - \operatorname { s u b - G a u s s i a n }$ <sub>, so a m</sub>i<sub>x</sub>t<sub>ure supermar</sub>ti<sub>n-</sub> <sub>ga</sub>l<sub>e</sub> <sub>w</sub>ith Vill<sub>e</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves</sub> <sub>an</sub> <sub>any</sub>ti<sub>me</sub> <sub>se</sub>lf<sub>-norma</sub>li<sub>se</sub>d b<sub>oun</sub>d<sub>.</sub> Fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> <sub>o</sub>f $\widehat { \theta } _ { t }$ <sub>an</sub>d th<sub>e curva</sub>t<sub>ure</sub> b<sub>oun</sub>d $\nabla ^ { 2 } \mathcal { L } _ { t } \succeq \kappa V _ { t }$ turn this into the stated <sub>p</sub>arameter concentration (A<sub>pp</sub>endix A.3).

Promising items and the regret certificate. On E, we can discard any item that is certifiably worse th<sub>an ano</sub>th<sub>er, an</sub>d b<sub>oun</sub>d th<sub>e regre</sub>t <sub>o</sub>f th<sub>e</sub> it<sub>ems</sub> th<sub>a</sub>t <sub>rema</sub>i<sub>n.</sub> L<sub>e</sub>t

$$
\mathcal { C } _ { t } = \big \{ \mathbf { x } \in \mathcal { S } _ { t } : \langle \hat { \theta } _ { t } , \mathbf { x } - \mathbf { y } \rangle + \beta _ { t } \| \mathbf { x } - \mathbf { y } \| _ { V _ { t } ^ { - 1 } } \geq 0 , \ \forall \mathbf { y } \in \mathcal { S } _ { t } \big \} ,\tag{7}
$$

<sub>an</sub>d l<sub>e</sub>t th<sub>e can</sub>did<sub>a</sub>t<sub>e pa</sub>i<sub>r</sub> b<sub>e</sub> th<sub>e max</sub>i<sub>mum-w</sub>idth <sub>pa</sub>i<sub>r o</sub>f <sub>prom</sub>i<sub>s</sub>i<sub>ng</sub> it<sub>ems,</sub>

$$
\left( \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } \right) \in \underset { \mathbf { x } , \mathbf { y } \in \mathcal { C } _ { t } } { \arg \operatorname* { m a x } } \left\| \mathbf { x } - \mathbf { y } \right\| _ { V _ { t } ^ { - 1 } } ,\tag{8}
$$

$$
w _ { t } = \left. \tilde { \mathbf { x } } _ { t } - \tilde { \mathbf { y } } _ { t } \right. _ { V _ { t } ^ { - 1 } } ,\tag{9}
$$

$$
c _ { t } = \operatorname* { m i n } \{ B , 2 \beta _ { t } w _ { t } \} .\tag{10}
$$

Si<sub>nce</sub> $\mathbf { x } _ { t } ^ { \star }$ i<sub>s</sub> <sub>never</sub> <sub>e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>e</sub>d <sub>on</sub> $\mathcal { E } ,$ <sub>every</sub> <sub>prom</sub>i<sub>s</sub>i<sub>ng</sub> it<sub>em</sub> i<sub>s</sub> <sub>c</sub>l<sub>ose</sub> t<sub>o</sub> it i<sub>n</sub> th<sub>e</sub> $V _ { t } ^ { - 1 }$ <sub>g</sub>eometr<sub>y</sub>. <sup>Th</sup>e f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> l<sub>emma ma</sub>k<sub>es</sub> thi<sub>s prec</sub>i<sub>se:</sub> th<sub>e w</sub>idth $w _ { t }$ b<sub>oun</sub>d<sub>s</sub> th<sub>e regre</sub>t <sub>o</sub>f <sub>any prom</sub>i<sub>s</sub>i<sub>ng</sub> it<sub>em, an</sub>d i<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> th<sub>a</sub>t <sub>o</sub>f th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>pa</sub>i<sub>r.</sub>

Lemma 3.2 (Regret certificate) $\mathcal { C } _ { t } \neq \emptyset$ for all t and on $\mathcal { E } , \mathbf { x } _ { t } ^ { \star } \in \mathcal { C } _ { t }$ <sub>an</sub>d $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \leq c _ { t }$ for <sup>e</sup>v<sup>er</sup>y $\mathbf { y } \in \mathcal { C } _ { t } .$ <sub>.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>pa</sub>i<sub>r</sub> h<sub>as</sub> <sub>regre</sub>t

$$
\begin{array} { r } { R _ { t } ^ { \mathrm { c a n d } } : = f ( \mathbf { x } _ { t } ^ { \star } ) - \frac { 1 } { 2 } \big ( f ( \tilde { \mathbf { x } } _ { t } ) + f ( \tilde { \mathbf { y } } _ { t } ) \big ) \leq c _ { t } . } \end{array}
$$

We next convert this utilit<sub>y</sub>-<sub>g</sub>a<sub>p</sub> certificate into a lower bound on the score <sub>p</sub>erformance in (3).

Lemma 3.3 (Score certificate) If a pair $\displaystyle ( \mathbf { x } , \mathbf { y } )$ satisfies $\begin{array} { r } { f ( \mathbf { x } _ { t } ^ { \star } ) - \frac { 1 } { 2 } ( f ( \mathbf { x } ) + f ( \mathbf { y } ) ) \leq c } \end{array}$ for some $c \geq 0 ,$ , then max $\{ s _ { t } ( \mathbf { x } ) , s _ { t } ( \mathbf { y } ) \} \geq \sigma ( - c ) \geq { \textstyle { \frac { 1 } { 2 } } } - { \frac { c } { 4 } }$

I<sub>n</sub>d<sub>ee</sub>d<sub>, w</sub>ith $g _ { \mathbf x } : = f ( \mathbf x _ { t } ^ { \star } ) - f ( \mathbf x ) \geq 0$ <sub>an</sub>d $g _ { \mathbf { y } }$ likewise, max $\{ s _ { t } ( \mathbf { x } ) , s _ { t } ( \mathbf { y } ) \} = \sigma ( - \operatorname* { m i n } \{ g _ { \mathbf { x } } , g _ { \mathbf { y } } \} )$ and min $\begin{array} { r } { \{ g _ { \mathbf { x } } , g _ { \mathbf { y } } \} \le \frac 1 2 ( g _ { \mathbf { x } } + g _ { \mathbf { y } } ) \le c ; } \end{array}$ th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> i<sub>s</sub> $\sigma ^ { \prime } \leq \frac { 1 } { 4 }$ <sub>.</sub> Th<sub>e anc</sub>h<sub>or</sub> i<sub>s</sub> $\begin{array} { r } { s _ { t } ( \mathbf { x } _ { t } ^ { \star } ) = \frac { 1 } { 2 } . } \end{array}$ <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>es</sub>ti<sub>ma</sub>ti<sub>ng</sub> <sub>any</sub> <sub>u</sub>tilit<sub>y.</sub>

Informative fallback. When the safety test rejects the candidate, we retain the baseline item and <sub>se</sub>l<sub>ec</sub>t th<sub>e</sub> <sub>prom</sub>i<sub>s</sub>i<sub>ng</sub> <sub>par</sub>t<sub>ner</sub> <sub>w</sub>ith th<sub>e</sub> l<sub>arges</sub>t <sub>compar</sub>i<sub>son</sub> <sub>w</sub>idth<sub>:</sub>

$$
\mathbf { y } _ { t } ^ { \mathrm { f b } } \in \underset { \mathbf { y } \in \mathcal { C } _ { t } } { \arg \operatorname* { m a x } } \left\| \mathbf { x } _ { b _ { t } } - \mathbf { y } \right\| _ { V _ { t } ^ { - 1 } } ,\tag{11}
$$

$$
\mathrm { d i s p l a y i n g \left( x \right)} _ { b _ { t } } , \mathbf { y } _ { t } ^ { \mathrm { f b } }  .\tag{12}
$$

Including the baseline guarantees that max $\{ s _ { t } ( \mathbf x _ { b _ { t } } ) , s _ { t } ( \mathbf y _ { t } ^ { \mathrm { f b } } ) \} \geq s _ { t } ( \mathbf x _ { b _ { t } } )$ <sub>.</sub> Th<sub>e max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng c</sub>h<sub>o</sub>i<sub>ce o</sub>f <sub>par</sub>t<sub>ner</sub> <sub>a</sub>l<sub>so</sub> <sub>connec</sub>t<sub>s</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>compar</sub>i<sub>son</sub> t<sub>o</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>w</sub>idth<sub>.</sub> Si<sub>nce</sub> $\tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } \in \mathcal { C } _ { t }$ <sub>,</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves</sub>

$$
w _ { t } \leq \left\| \tilde { \mathbf { x } } _ { t } - \mathbf { x } _ { b _ { t } } \right\| _ { V _ { t } ^ { - 1 } } + \left\| \mathbf { x } _ { b _ { t } } - \tilde { \mathbf { y } } _ { t } \right\| _ { V _ { t } ^ { - 1 } } \leq 2 \| \mathbf { x } _ { b _ { t } } - \mathbf { y } _ { t } ^ { \mathrm { { f b } } } \| _ { V _ { t } ^ { - 1 } } .\tag{13}
$$

Algorithm 1 MaxInP9BC (Conservative Maximum-Informative-Pair)   
Require: risk level $\alpha ,$ confidence δ, regularisation λ, bounds S, L   
1<sub>:</sub> I<sub>n</sub>iti<sub>a</sub>li<sub>se</sub> $\Lambda _ { 0 }  0 , V _ { 1 }  \lambda I _ { d }$   
2: for $t = 1 , 2 , \ldots$ . do   
3<sub>:</sub> Ob<sub>serve</sub> $\boldsymbol { S } _ { t } ,$ b<sub>ase</sub>li<sub>ne</sub> $\mathbf { x } _ { b _ { t } }$ <sub>an</sub>d it<sub>s</sub> k<sub>nown score</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } )$ ; <sup>com</sup>pu<sup>te</sup> $\hat { \theta } _ { t } , \beta _ { t }$ b<sub>y</sub> (5)–(6)   
4<sub>:</sub> F<sub>orm</sub> $\mathcal { C } _ { t }$ b<sub>y</sub> (7); com<sub>p</sub>ute $( \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } ) , w _ { t } , c _ { t }$ b<sub>y</sub> (10)   
5<sub>:</sub> if $\begin{array} { r } { \Lambda _ { t - 1 } + \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) s _ { t } ( { \bf x } _ { b _ { t } } ) \geq 0 } \end{array}$ then   
6<sub>:</sub> Di<sub>sp</sub>l<sub>ay</sub> $( { \bf x } _ { t } , { \bf y } _ { t } ) \gets ( \tilde { { \bf x } } _ { t } , \tilde { { \bf y } } _ { t } )$ <sub>an</sub>d <sub>se</sub>t $\begin{array} { r } { \Lambda _ { t }  \Lambda _ { t - 1 } + \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) s _ { t } ( \mathbf x _ { b _ { t } } ) } \end{array}$ {candidate}   
7<sub>:</sub> else   
8<sub>:</sub> Di<sub>sp</sub>l<sub>ay</sub> $( { \bf x } _ { t } , { \bf y } _ { t } ) \gets ( { \bf x } _ { b _ { t } } , { \bf y } _ { t } ^ { \mathrm { f b } } )$ from (12) and set $\Lambda _ { t }  \Lambda _ { t - 1 } + \alpha s _ { t } ( \mathbf { x } _ { b _ { t } } )$ {fallback}   
9: end if   
10<sub>:</sub> Ob<sub>serve</sub> $o _ { t } ;$ set ${ \bf z } _ { t }  { \bf x } _ { t } - { \bf y } _ { t }$ <sub>an</sub>d $V _ { t + 1 } \gets V _ { t } + \mathbf { z } _ { t } \mathbf { z } _ { t } ^ { \top }$   
11: end for

Th<sub>us,</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>compar</sub>i<sub>son re</sub>t<sub>a</sub>i<sub>ns a</sub>t l<sub>eas</sub>t h<sub>a</sub>lf th<sub>e can</sub>did<sub>a</sub>t<sub>e w</sub>idth<sub>, w</sub>h<sub>e</sub>th<sub>er or no</sub>t $\mathbf { x } _ { b _ { t } } \in \mathcal { C } _ { t }$ Note that because the fallback <sub>p</sub>air contains the baseline, its one-round sur<sub>p</sub>lus under (3) is at least $\alpha s _ { t } ( { \mathbf { x } } _ { b _ { t } } )$ , re<sub>g</sub>ardless of its <sub>p</sub>artner. The <sub>p</sub>artner can therefore be chosen for information, and b<sub>y</sub> (13) it <sub>re</sub>t<sub>a</sub>i<sub>ns a</sub>t l<sub>eas</sub>t h<sub>a</sub>lf th<sub>e can</sub>did<sub>a</sub>t<sub>e w</sub>idth<sub>.</sub> It<sub>s regre</sub>t i<sub>s a</sub>l<sub>so con</sub>t<sub>ro</sub>ll<sub>e</sub>d<sub>: s</sub>i<sub>nce</sub> ${ \bf y } _ { t } ^ { \mathrm { f b } } \in \mathcal { C } _ { t }$ , on $\mathcal { E }$

$$
\begin{array} { r } { R _ { t } = \frac { 1 } { 2 } \Big ( \Delta _ { t , b _ { t } } + f ( { \bf x } _ { t } ^ { \star } ) - f ( { \bf y } _ { t } ^ { \mathrm { f b } } ) \Big ) \leq \frac { 1 } { 2 } ( \Delta _ { h } + c _ { t } ) . } \end{array}
$$

Th<sub>e</sub> b<sub>ase</sub>li<sub>ne con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es a</sub>t <sub>mos</sub>t $\Delta _ { h } / 2 ,$ <sub>, an</sub>d th<sub>e par</sub>t<sub>ner</sub>’<sub>s con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s con</sub>t<sub>ro</sub>ll<sub>e</sub>d b<sub>y</sub> th<sub>e same</sub> <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> $c _ { t }$ <sub>use</sub>d f<sub>or can</sub>did<sub>a</sub>t<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub>

Safety budget. The resulting algorithm, MaxInP9BC, is given in Algorithm 1. It maintains a com<sub>p</sub>uta<sup>bl</sup>e <sub>q</sub>uant<sup>i</sup>t<sub>y</sub> $\Lambda _ { t } .$ <sub>,</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>ze</sub>d <sub>a</sub>t $\Lambda _ { 0 } = 0 .$ <sub>,</sub> th<sub>a</sub>t l<sub>ower-</sub>b<sub>oun</sub>d<sub>s</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve sa</sub>f<sub>e</sub>t<sub>y surp</sub>l<sub>us on</sub> E. B<sub>y</sub> Lemma 3.3, the candidate increment is $\begin{array} { r } { \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) s _ { t } ( \mathbf x _ { b _ { t } } ) } \end{array}$ <sub>.</sub> A f<sub>a</sub>llb<sub>ac</sub>k i<sub>s cre</sub>dit<sub>e</sub>d <sub>w</sub>ith $\alpha s _ { t } ( { \mathbf { x } } _ { b _ { t } } )$ <sub>, w</sub>hi<sub>c</sub>h l<sub>ower-</sub>b<sub>oun</sub>d<sub>s</sub> it<sub>s ac</sub>t<sub>ua</sub>l <sub>surp</sub>l<sub>us</sub> b<sub>ecause</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub> E<sub>ac</sub>h <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory, curren</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>s,</sub> b<sub>ase</sub>li<sub>ne score, an</sub>d <sub>c</sub>h<sub>osen pa</sub>i<sub>r</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>ou</sub>t<sub>come</sub> i<sub>s</sub> <sub>o</sub>b<sub>serve</sub>d<sub>.</sub> Th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>s</sub> <sub>accep</sub>t<sub>e</sub>d <sub>exac</sub>tl<sub>y</sub> <sub>w</sub>h<sub>en</sub> it<sub>s</sub> <sub>propose</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> l<sub>eaves</sub> $\Lambda _ { t }$ nonne<sub>g</sub>ative; otherwise, the fallback <sub>p</sub>reserves nonne<sub>g</sub>ativit<sub>y</sub>. On E, these <sub>p</sub>er-round lower bounds th<sub>ere</sub>f<sub>ore</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>e</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub>

L<sub>e</sub>t $A _ { T } , B _ { T }$ b<sub>e</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s up</sub> t<sub>o</sub> $T _ { \cdot }$ <sub>, w</sub>ith $M _ { T } = | A _ { T } |$ <sub>an</sub>d $N _ { T } = | \boldsymbol { B _ { T } } |$ <sub>.</sub> Si<sub>nce</sub> $\Lambda _ { t - 1 } \geq 0$ b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d $\begin{array} { r } { s _ { t } ( \mathbf { x } _ { b _ { t } } ) \leq \frac { 1 } { 2 } } \end{array}$ <sub>,</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>ncremen</sub>t i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $\begin{array} { r } { \frac { \alpha } { 2 } \mathrm { ~ - ~ } \frac { c _ { t } } { 4 } } \end{array}$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, <sub>every roun</sub>d <sub>w</sub>ith $c _ { t } \leq 2 \alpha$ is accepted, and any rejection requires $c _ { t } > 2 \alpha$ <sub>.</sub> Thi<sub>s cr</sub>it<sub>er</sub>i<sub>on app</sub>li<sub>es</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y a</sub>t <sub>eac</sub>h <sub>roun</sub>d<sub>:</sub> th<sub>e cer</sub>tifi<sub>ca</sub>t<sub>e can c</sub>h<sub>ange w</sub>ith th<sub>e con</sub>t<sub>ex</sub>t<sub>s, es</sub>ti<sub>ma</sub>t<sub>es, an</sub>d <sub>con</sub>fid<sub>ence</sub> <sub>ra</sub>di<sub>us, so accep</sub>t<sub>ance a</sub>t <sub>one roun</sub>d d<sub>oes no</sub>t i<sub>mp</sub>l<sub>y accep</sub>t<sub>ance a</sub>t <sub>a</sub>ll <sub>su</sub>b<sub>sequen</sub>t <sub>roun</sub>d<sub>s.</sub>

Main result. With $\underline { { s } } : = \sigma ( - \Delta _ { h } )$ <sub>an</sub>d $\bar { \beta } _ { t } : = \operatorname* { m a x } \{ B / 2 , \beta _ { t } \}$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> <sub>enve</sub>l<sub>ope</sub> <sub>an</sub>d th<sub>e</sub> f<sub>a</sub>llb<sub>ac</sub>k b<sub>oun</sub>d

$$
G ( t ) : = 4 \bar { \beta } _ { t } \sqrt { 2 d t \log { \left( 1 + \frac { 4 L ^ { 2 } t } { \lambda d } \right) } } ,\tag{14}
$$

$$
\begin{array} { r } { N _ { \mathrm { c o n s } } : = \operatorname* { s u p } \big \{ t \in \mathbb { N } _ { \geq 1 } : 4 \alpha \underline { { s } } t < G ( t ) \big \} , } \end{array}\tag{15}
$$

<sub>w</sub>ith $G ( 0 ) = 0$ and sup $\varnothing : = 0$ . The function G is nondecreasing. For fixed problem parameters and <sub>con</sub>fid<sub>ence</sub> l<sub>eve</sub>l<sub>,</sub> $G ( t ) = O ( \sqrt t \log ( e t ) ) = o ( t ) , \operatorname { s o } \ N _ { \mathrm { c o n s } } < \infty .$

Theorem 3.1 (Anytime safety and regret) LetAssumptions 1, 2, 3, and 4 hold, and run Algorithm 1 <sub>w</sub>ith $\alpha , \delta \in ( 0 , 1 )$ <sub>an</sub>d $\lambda > 0$ <sub>.</sub> With <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta ,$ , simultaneously for every $T \geq 1$ the displayed pairs are α-conservative in the sense of (3), the fallback count satisfies $N _ { T } \ \leq$ min $\{ T , N _ { \mathrm { c o n s } } \}$ <sub>, an</sub>d $\begin{array} { r } { \mathsf { R e g } ( T ) \leq G ( T ) + \frac { 1 } { 2 } \Delta _ { h } N _ { \mathrm { c o n s } } } \end{array}$ <sub>.</sub> Wh<sub>en</sub> $L , S , \lambda$ are treated as fixed constants, suppressing logarithmicfactors in the remaining parameters gives

$$
\mathsf { R e g } ( T ) = \tilde { \mathcal { O } } \left( \frac { d \sqrt { T } } { \kappa } + \frac { \Delta _ { h } d ^ { 2 } } { \kappa ^ { 2 } \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \right) .\tag{16}
$$

Proofsketch. Work on $\mathcal { E } _ { : }$ <sub>, w</sub>hi<sub>c</sub>h h<sub>as pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta$ b<sub>y</sub> L<sub>emma</sub> 3<sub>.</sub>1<sub>; a</sub>ll <sub>guaran</sub>t<sub>ees</sub> h<sub>o</sub>ld <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y on</sub> thi<sub>s even</sub>t<sub>.</sub>

(a) Envelope. On a candidate round, $c _ { t } = \operatorname* { m i n } \{ B , 2 \beta _ { t } \| \mathbf { z } _ { t } \| _ { V _ { t } ^ { - 1 } } \}$ . On a fallback round, (13) <sub>g</sub><sup>i</sup>ves $c _ { t } \leq \operatorname* { m i n } \{ B , 4 \beta _ { t } \| \mathbf { z } _ { t } \| _ { V _ { t } ^ { - 1 } } \}$ <sub>.</sub> Th<sub>us,</sub> i<sub>n</sub> b<sub>o</sub>th <sub>cases,</sub> $c _ { t } \leq 4 \beta _ { t }$ min $\left\{ 1 , \left\| \mathbf { z } _ { t } \right\| _ { V _ { t } ^ { - 1 } } \right\}$ . Cauchy– S<sub>c</sub>h<sub>warz an</sub>d th<sub>e e</sub>lli<sub>p</sub>ti<sub>ca</sub>l<sub>-po</sub>t<sub>en</sub>ti<sub>a</sub>l b<sub>oun</sub>d $\textstyle \sum _ { i = 1 } ^ { t }$ min $\begin{array} { r } { \{ 1 , \Vert \mathbf { z } _ { i } \Vert _ { V _ { i } ^ { - 1 } } ^ { 2 } \} \leq 2 d \log \left( 1 + \frac { 4 L ^ { 2 } t } { \lambda d } \right) } \end{array}$ <sub>y</sub>i<sub>e</sub>ld $\textstyle \sum _ { i = 1 } ^ { t } c _ { i } \leq G ( t )$ . This includes the certificate computed on a rejected round, since the displayed fallback com<sub>p</sub>arison controls that certificate throu<sub>g</sub>h (13).

(b) Safety. On candidate rounds, Lemma 3.3 shows that the true one-round surplus is at least the b<sub>u</sub>d<sub>ge</sub>t i<sub>ncremen</sub>t<sub>.</sub> O<sub>n</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>guaran</sub>t<sub>ees</sub> <sub>a</sub> t<sub>rue</sub> <sub>surp</sub>l<sub>us</sub> <sub>o</sub>f <sub>a</sub>t l<sub>eas</sub>t $\alpha s _ { t } ( \mathbf { x } _ { b _ { t } } )$ <sub>, aga</sub>i<sub>n</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>ncremen</sub>t<sub>.</sub> I<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> th<sub>ere</sub>f<sub>ore s</sub>h<sub>ows</sub> th<sub>a</sub>t $\Lambda _ { t }$ l<sub>ower-</sub>b<sub>oun</sub>d<sub>s</sub> th<sub>e</sub> t<sub>rue</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>surp</sub>l<sub>us.</sub> C<sub>an</sub>did<sub>a</sub>t<sub>e</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>preserve</sub> $\Lambda _ { t } \geq 0$ b<sub>y</sub> th<sub>e</sub> <sub>accep</sub>t<sub>ance</sub> t<sub>es</sub>t<sub>,</sub> <sub>an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>up</sub>d<sub>a</sub>t<sub>es a</sub>dd <sub>a pos</sub>iti<sub>ve quan</sub>tit<sub>y, prov</sub>i<sub>ng sa</sub>f<sub>e</sub>t<sub>y.</sub>

(c) Fallbacks. Expanding the failed acceptance test at a fallback round $\tau _ { \ast }$ <sub>, an</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $\frac { 1 } { 2 } - ( 1 - \alpha ) s _ { i } ( \mathbf { x } _ { b _ { i } } ) \geq \alpha / 2$ <sub>an</sub>d $\alpha s _ { i } ( { \bf x } _ { b _ { i } } ) \geq \alpha \underline { { s } }$ , <sub>g</sub><sup>i</sup>ves

$$
\frac { \alpha } { 2 } ( M _ { \tau - 1 } + 1 ) + \alpha \underline { { s } } N _ { \tau - 1 } < \frac { 1 } { 4 } \left( \sum _ { i \in \mathcal { A } _ { \tau - 1 } } c _ { i } + c _ { \tau } \right) \leq \frac { 1 } { 4 } G ( \tau ) .
$$

B<sub>ecause</sub> $\underline { { s } } \le 1 / 2$ <sub>an</sub>d $M _ { \tau - 1 } + { \cal N } _ { \tau - 1 } + 1 = \tau$ , the left side is at least αsτ. Hence every fallback ti<sub>me sa</sub>ti<sub>s</sub>fi<sub>es</sub> $4 \alpha \underline { s } \tau < G ( \tau )$ <sub>an</sub>d i<sub>s a</sub>t <sub>mos</sub>t $N _ { \mathrm { c o n s } }$ , p<sup>ro</sup>v<sup>in</sup>g $N _ { T } \leq \operatorname* { m i n } \{ T , N _ { \mathrm { c o n s } } \}$

(d) Regret. Lemma 3.2 gives $R _ { t } \leq c _ { t }$ <sub>on</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>roun</sub>d<sub>s.</sub> O<sub>n</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s,</sub> ${ \bf y } _ { t } ^ { \mathrm { f b } } \in \mathcal { C } _ { t }$ <sub>g</sub><sup>i</sup>ves $R _ { t } \leq ( \Delta _ { h } + c _ { t } ) / 2$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
\mathtt { R e g } ( T ) \le \sum _ { t = 1 } ^ { T } c _ { t } + \frac { \Delta _ { h } } { 2 } N _ { T } \le G ( T ) + \frac { \Delta _ { h } } { 2 } N _ { \mathrm { c o n s } } .
$$

F<sub>or</sub> fi<sub>xe</sub>d $L , S , \lambda .$ <sub>,</sub> th<sub>e</sub> <sub>s</sub>i<sub>mp</sub>lifi<sub>e</sub>d <sub>ra</sub>t<sub>e</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $\bar { \beta } _ { t } = \tilde { \mathcal { O } } ( \sqrt { d } / \kappa )$ <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>or</sub> $N _ { \mathrm { c o n s } }$ <sub>.</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>are</sub> <sub>g</sub>i<sub>ven</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub> □

Remark 2 (An additive cost of conservatism) The leadin term in the re ret bound in (16) has the same dependence on d and T as unconstrained MaxInP (Saha, 2021), while the additional term $\Delta _ { h } N _ { \mathrm { c o n s } } / 2$ is independent ofTforfixed parameters.

Remark 3 (Why not transfer CLUCB directly) For bandits with numerical rewards, CLUCB (Kazerouni et al., 2017) certifies safety by lower-bounding cumulative candidate rewards with a confidence set, and it credits known baseline rewards separately. Under preference feedback, the desi<sub>g</sub>n matrix in (5) accumulates com<sub>p</sub>arison vectors ${ \bf z } _ { t } = { \bf x } _ { t } - { \bf y } _ { t } ,$ <sub>, w</sub>h<sub>ereas average u</sub>tiliti<sub>es</sub> d<sub>epen</sub>d on $( { \bf x } _ { t } + { \bf y } _ { t } ) / 2$ . A comparison-based confidence set can therefore retain uncertainty in a common utility component that comparisons never reveal. Proposition A.2 (Appendix A.9) gives an i.i.d. two-dimensional instance on which this transfer incurs $\mathsf { R e g } ( T ) = \Omega ( T )$ <sub>w</sub>ith hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> <sub>even w</sub>ith <sub>exac</sub>t b<sub>ase</sub>li<sub>ne u</sub>tiliti<sub>es an</sub>d <sub>ar</sub>bit<sub>rary can</sub>did<sub>a</sub>t<sub>e se</sub>l<sub>ec</sub>ti<sub>on.</sub> Th<sub>e o</sub>b<sub>s</sub>t<sub>ruc</sub>ti<sub>on concerns</sub> th<sub>e</sub> unchanged certification rule: known baseline utilities would also permit a baseline-centered cardinal certificate. MaxInP<sup>9</sup>BC instea<sup>d</sup> certifies scores t<sup>h</sup>roug<sup>h</sup> uti<sup>l</sup>ity gaps (<sup>L</sup>emmas <sup>3</sup>.<sup>2</sup> an<sup>d 3</sup>.<sup>3</sup>), w<sup>h</sup>ic<sup>h</sup> depend only on pairwise diferences. Through (13), it also ties the uncertainty in this certificate to the comparisons displayed on both candidate andfallback rounds.

Remark 4 (Relaxing baseline-score knowledge) For ease ofpresentation, we assume that baseline scores are known exactly. Appendix A.10 relaxes this assumption by modifying the safety budget to use lower and upper score bounds together with the known range [s, 1/2].

## 4. General Utility Functions

Li<sub>near u</sub>tiliti<sub>es can</sub> b<sub>e res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ve w</sub>h<sub>en</sub> it<sub>ems are</sub> d<sub>escr</sub>ib<sub>e</sub>d b<sub>y r</sub>i<sub>c</sub>h f<sub>ea</sub>t<sub>ures.</sub> W<sub>e now</sub> l<sub>e</sub>t $f$ li<sub>e</sub> i<sub>n a</sub> <sub>genera</sub>l <sub>c</sub>l<sub>ass</sub> $\mathcal { F }$ <sub>,</sub> <sub>accesse</sub>d <sub>on</sub>l<sub>y</sub> th<sub>roug</sub>h <sub>an</sub> <sub>on</sub>li<sub>ne</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c-regress</sub>i<sub>on</sub> <sub>orac</sub>l<sub>e.</sub> I<sub>n</sub> th<sub>e</sub> <sub>uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d settin<sub>g</sub>, Sekhari et al. (2023) sam<sub>p</sub>le com<sub>p</sub>arisons from an inverse-<sub>g</sub>a<sub>p</sub>-wei<sub>g</sub>hted (IGW) distribution b<sub>u</sub>ilt f<sub>rom</sub> th<sub>e orac</sub>l<sub>e</sub>’<sub>s pre</sub>di<sub>c</sub>ti<sub>ons.</sub> With<sub>ou</sub>t <sub>a con</sub>fid<sub>ence e</sub>lli<sub>pso</sub>id<sub>, we canno</sub>t <sub>cer</sub>tif<sub>y eac</sub>h <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sup>i</sup>n<sup>di</sup>v<sup>id</sup>ua<sup>ll</sup>y as <sup>i</sup>n <sup>L</sup>emma <sup>3</sup>.<sup>2</sup>. <sup>W</sup>e t<sup>h</sup>ere<sup>f</sup>ore <sup>d</sup>eve<sup>l</sup>op BTIGW<sup>9</sup>BC (<sup>Al</sup>gor<sup>i</sup>t<sup>h</sup>m <sup>2</sup>) an<sup>d</sup> cert<sup>if</sup>y sa<sup>f</sup>ety <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y:</sub> th<sub>e orac</sub>l<sub>e guaran</sub>t<sub>ee y</sub>i<sub>e</sub>ld<sub>s a compu</sub>t<sub>a</sub>bl<sub>e enve</sub>l<sub>ope on</sub> t<sub>o</sub>t<sub>a</sub>l <sub>regre</sub>t<sub>, an</sub>d <sub>a can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>s</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>on</sub>l<sub>y</sub> if th<sub>e accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>score cre</sub>dit<sub>s cover</sub> it<sub>.</sub> Oth<sub>erw</sub>i<sub>se,</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>a</sub>ll<sub>s</sub> b<sub>ac</sub>k t<sub>o a</sub> b<sub>ase</sub>li<sub>ne-con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng pa</sub>i<sub>r w</sub>h<sub>ose par</sub>t<sub>ner</sub> i<sub>s s</sub>till d<sub>rawn</sub> b<sub>y</sub> IGW<sub>.</sub>

Assumption 5 (Realizability and logistic oracle) The utility f belongs to a class F whosefunctions satisfy $| g ( \mathbf { x } ) - g ( \mathbf { y } ) | \leq B$ on D. Before its m-th comparison $\left( \mathbf { x } _ { m } , \mathbf { y } _ { m } \right)$ i<sub>s</sub> d<sub>rawn,</sub> th<sub>e orac</sub>l<sub>e ou</sub>t<sub>pu</sub>t<sub>s</sub> $\hat { f } _ { m }$ <sub>w</sub>ith th<sub>e same range</sub> b<sub>oun</sub>d<sub>;</sub> $\hat { f } _ { m }$ <sub>may use</sub> th<sub>e curren</sub>t it<sub>em se</sub>t <sub>an</sub>d <sub>nee</sub>d <sub>no</sub>t li<sub>e</sub> i<sub>n</sub> ${ \mathcal F } .$ <sub>.</sub> With $\ell _ { m } ( g ) = \log ( 1 + e ^ { g ( \mathbf { x } _ { m } ) - g ( \mathbf { y } _ { m } ) } ) - o _ { m } [ g ( \mathbf { x } _ { m } ) - g ( \mathbf { y } _ { m } ) ]$ , every fresh instance satisfies, on every adaptive sequence andfor all $n \geq 1$

$$
\sum _ { m = 1 } ^ { n } \ell _ { m } ( \hat { f } _ { m } ) - \operatorname* { i n f } _ { g \in \mathcal { F } } \sum _ { m = 1 } ^ { n } \ell _ { m } ( g ) \leq \mathcal { R } _ { \log } ( n ) ,\tag{17}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { R } _ { \mathrm { l o g } }$ i<sub>s</sub> k<sub>nown,</sub> <sub>nonnega</sub>ti<sub>ve,</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng,</sub> <sub>an</sub>d $o ( n )$ .

Algorithm 2 BTIGW9BC (Bradley–Terry Inverse-Gap-Weighting Baseline-Conservative)   
Require: $\alpha , \delta \in ( 0 , 1 ) , B , \mathcal { R } _ { \mathrm { l o g } }$ <sub>,</sub> <sub>an</sub>d <sub>an</sub> <sub>orac</sub>l<sub>e</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> A<sub>ssump</sub>ti<sub>on</sub> 5   
1<sub>:</sub> I<sub>n</sub>iti<sub>a</sub>li<sub>se</sub> $Q _ { 0 } \gets 0$   
2: for $t = 1 , 2 , \ldots$ . do   
3<sub>:</sub> If $t = 2 ^ { e }$ <sub>, res</sub>t<sub>ar</sub>t th<sub>e orac</sub>l<sub>e an</sub>d <sub>se</sub>t $\iota _ { e } , E _ { e } , C _ { e }$ b<sub>y</sub> (21)   
4<sub>:</sub> Ob<sub>serve</sub> $S _ { t } , \mathbf { x } _ { b _ { t } } , s _ { t } ( \mathbf { x } _ { b _ { t } } )$ <sub>; o</sub>bt<sub>a</sub>i<sub>n</sub> $\hat { f } _ { t } \colon$ f<sub>orm</sub> $p _ { t }$ <sub>w</sub>ith $\gamma _ { t } ;$ set $\begin{array} { r } { h _ { t }  \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { t } ( \mathbf x _ { b _ { t } } ) } \end{array}$   
5: if $Q _ { t - 1 } + h _ { t } \geq G _ { \mathrm { I G W } } ( t ) / 4$ then   
6<sub>:</sub> D<sub>raw</sub> $I _ { t } , J _ { t } \sim p _ { t }$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y;</sub> di<sub>sp</sub>l<sub>ay</sub> $( \mathbf { x } _ { t , I _ { t } } , \mathbf { x } _ { t , J _ { t } } ) ; Q _ { t }  Q _ { t - 1 } + h _ { t }$ {candidate}   
7: else   
8<sub>:</sub> D<sub>raw</sub> $J _ { t } \sim p _ { t } ;$ di<sub>sp</sub>l<sub>ay</sub> $( { \bf x } _ { b _ { t } } , { \bf x } _ { t , J _ { t } } ) ; Q _ { t }  Q _ { t - 1 } + \alpha s _ { t } ( { \bf x } _ { b _ { t } } )$ {fallback}   
9: end if   
10<sub>:</sub> Ob<sub>serve</sub> $O t$ <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>orac</sub>l<sub>e</sub> <sub>w</sub>ith thi<sub>s</sub> <sub>compar</sub>i<sub>son</sub>   
11: end for

IGW with a fixed partner. Given $\hat { f } _ { t } ^ { \phantom { } } .$ <sub>,</sub> l<sub>e</sub>t $\hat { a } _ { t } \in \arg \operatorname* { m a x } _ { a } \hat { f } _ { t } \big ( \mathbf { x } _ { t , a } \big )$ <sub>an</sub>d

$$
p _ { t } ( a ) = \frac { 1 } { K + \gamma _ { t } [ \hat { f } _ { t } ( \mathbf { x } _ { t , \hat { a } _ { t } } ) - \hat { f } _ { t } ( \mathbf { x } _ { t , a } ) ] } \quad ( a \neq \hat { a } _ { t } ) ,\tag{18}
$$

$$
p _ { t } ( \hat { a } _ { t } ) = 1 - \sum _ { a \neq \hat { a } _ { t } } p _ { t } ( a ) .\tag{19}
$$

W<sub>r</sub>it<sub>e</sub> $\Delta _ { t } ( \mathbf { x } ) = f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } ) \mathrm { a n d } \varepsilon _ { t } ( \mathbf { x } , \mathbf { y } ) = ( \hat { f } _ { t } - f ) ( \mathbf { x } ) - ( \hat { f } _ { t } - f ) ( \mathbf { y } )$ f<sub>or</sub> th<sub>e</sub> <sub>marg</sub>i<sub>n</sub> <sub>error.</sub>

Lemma 4.1 (IGW with a fixed partner) For every $\gamma _ { t } > 0$ <sub>an</sub>d $\mathbf { u } \in { S } _ { t } ,$ the following holds

$$
\mathbb { E } _ { a \sim p { t } } \Delta _ { t } ( \mathbf { x } _ { t , a } ) \leq \frac { 2 K } { \gamma _ { t } } + \frac { \gamma _ { t } } { 4 } \mathbb { E } _ { a \sim p { t } } \varepsilon _ { t } ( \mathbf { x } _ { t , a } , \mathbf { u } ) ^ { 2 } .
$$

This is the standard IGW bound applied after subtracting the values at u. The free partner lets the <sub>error</sub> b<sub>e</sub> <sub>measure</sub>d <sub>on</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r:</sub> f<sub>or</sub> <sub>a</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> $\left( \mathbf { x } _ { t , I _ { t } } , \mathbf { x } _ { t , J _ { t } } \right)$ <sub>w</sub>ith $I _ { t } , J _ { t } \sim p _ { t }$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>,</sub> take u to be the other item; for a fallback $\left( \mathbf { x } _ { b _ { t } } , \mathbf { x } _ { t , J _ { t } } \right)$ <sub>,</sub> t<sub>a</sub>k<sub>e</sub> $\mathbf { u } = \mathbf { x } _ { b _ { t } }$ . This <sub>p</sub>la<sub>y</sub>s the role of (13): the f<sub>a</sub>llb<sub>ac</sub>k <sub>par</sub>t<sub>ner</sub>’<sub>s gap</sub> i<sub>s con</sub>t<sub>ro</sub>ll<sub>e</sub>d <sub>exac</sub>tl<sub>y as a can</sub>did<sub>a</sub>t<sub>e</sub>’<sub>s regre</sub>t<sub>.</sub>

An anytime envelope. We run the oracle in epochs $e = 0 , 1 , \ldots$ . of length $n _ { e } = 2 ^ { e }$ <sub>, res</sub>t<sub>ar</sub>ti<sub>ng</sub> it <sub>a</sub>t <sub>eac</sub>h<sub>, w</sub>ith <sub>con</sub>fid<sub>ence</sub> $\delta _ { e } = \delta / [ ( e + 1 ) ( e + 2 ) ]$ <sub>.</sub> F<sub>or</sub> l<sub>oca</sub>l <sub>roun</sub>d $m ,$ set

$$
\iota _ { e } = \log \frac { 3 n _ { e } } { \delta _ { e } } , \quad E _ { e } = \frac { 8 \mathcal { R } _ { \log } ( n _ { e } ) } { \kappa } + \frac { 5 \iota _ { e } } { \kappa ^ { 2 } } ,\tag{20}
$$

$$
\gamma _ { e , m } = \sqrt { \frac { 8 K m } { E _ { e } } } , C _ { e } = \frac { 3 } { \sqrt { 2 } } \sqrt { K E _ { e } } + B \sqrt { 2 \iota _ { e } } .\tag{21}
$$

At time t, let $e ( t ) = \lfloor \log _ { 2 } { t } \rfloor , m ( t ) = t - 2 ^ { e ( t ) } + 1 , \gamma _ { t } = \gamma _ { e ( t ) , m ( t ) }$ <sub>,</sub> <sub>an</sub>d $\begin{array} { r } { G _ { \mathrm { I G W } } ( t ) = \sum _ { j < e ( t ) } C _ { j } \sqrt { 2 ^ { j } } + } \end{array}$ $C _ { e ( t ) } \sqrt { m ( t ) } , G _ { \mathrm { I G W } } ( 0 ) = 0 . \mathrm { L e t } r _ { t } ^ { \circ } = R _ { t }$ <sub>on</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>roun</sub>d<sub>s</sub> <sub>an</sub>d $r _ { t } ^ { \circ } = \Delta _ { t } ( \mathbf { y } _ { t } )$ <sub>on</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s.</sub>

Lemma 4.2 (Anytime regret envelope) Under Assumptions 1, 2, and 5, with probability at least 1 − δ, Algorithm 2 satisfies $\begin{array} { r } { \sum _ { i < t } r _ { i } ^ { \circ } \le G _ { \mathrm { I G W } } ( t ) } \end{array}$ for all $t \geq 1 .$ <sub>.</sub> M<sub>oreover, w</sub>ith

$$
A ( t ) : = \mathcal { R } _ { \log } ( t ) / \kappa + 1 / \kappa ^ { 2 } , G _ { \mathrm { I G W } } ( t ) = \tilde { \mathcal { O } } ( \sqrt { K t A ( t ) } ) = o ( t ) .
$$

Th<sub>e</sub> <sub>proo</sub>f <sub>app</sub>li<sub>es</sub> L<sub>emma</sub> 4<sub>.</sub>1<sub>,</sub> th<sub>e</sub> <sub>orac</sub>l<sub>e</sub> b<sub>oun</sub>d <sub>an</sub>d <sub>mar</sub>ti<sub>nga</sub>l<sub>e</sub> <sub>concen</sub>t<sub>ra</sub>ti<sub>on</sub> <sub>w</sub>ithi<sub>n</sub> <sub>eac</sub>h <sub>epoc</sub>h (A<sub>pp</sub>endix B).

Safety credits. Instead of charging each candidate its own certificate, BTIGW9BC charges the envelope <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y.</sub> It k<sub>eeps</sub> <sub>cre</sub>dit<sub>s</sub> $Q _ { t }$ <sub>,</sub> <sub>w</sub>ith $Q _ { 0 } = 0$ <sub>an</sub>d <sub>no</sub> <sub>rese</sub>t b<sub>e</sub>t<sub>ween</sub> <sub>epoc</sub>h<sub>s:</sub> <sub>a</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>earns</sub> $\begin{array} { r } { h _ { t } : = \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { t } ( \mathbf x _ { b _ { t } } ) } \end{array}$ <sub>an</sub>d <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>earns</sub> $\alpha s _ { t } ( { \mathbf { x } } _ { b _ { t } } )$ <sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>3 <sub>w</sub>ith $c = R _ { t }$ <sub>, a can</sub>did<sub>a</sub>t<sub>e</sub>’<sub>s</sub> t<sub>rue surp</sub>l<sub>us</sub> i<sub>s a</sub>t l<sub>eas</sub>t $h _ { t } - R _ { t } / 4 ;$ <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k’<sub>s</sub> i<sub>s a</sub>t l<sub>eas</sub>t $\alpha s _ { t } ( { \mathbf { x } } _ { b _ { t } } )$ b<sub>ecause</sub> it <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub> Th<sub>e</sub> t<sub>rue cumu</sub>l<sub>a</sub>ti<sub>ve surp</sub>l<sub>us</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore a</sub>t l<sub>eas</sub>t $\begin{array} { r } { Q _ { t } - \frac { 1 } { 4 } \sum _ { i \in \mathcal { A } _ { t } } R _ { i } } \end{array}$ <sub>, w</sub>h<sub>ere</sub> $\boldsymbol { A } _ { t }$ i<sub>s</sub> th<sub>e se</sub>t <sub>o</sub>f <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>roun</sub>d<sub>s, an</sub>d b<sub>y</sub> L<sub>emma</sub> 4<sub>.</sub>2 th<sub>e c</sub>h<sub>arge</sub> i<sub>s a</sub>t <sub>mos</sub>t $G _ { \mathrm { I G W } } ( t ) / 4$ <sub>, w</sub>hi<sub>c</sub>h th<sub>e accep</sub>t<sub>ance</sub> t<sub>es</sub>t i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 2 <sub>requ</sub>i<sub>res</sub> th<sub>e</sub> <sub>cre</sub>dit<sub>s</sub> t<sub>o</sub> <sub>cover.</sub>

Main result. Since $s _ { t } ( \mathbf x _ { b _ { t } } ) \in [ \underline { { s } } , \frac { 1 } { 2 } ]$ <sub>w</sub>ith $\underline { { s } } = \sigma \big ( - \Delta _ { h } \big )$ , every credit is at least αs, so a rejection at time t requires 4αs $t < G _ { \mathrm { I G W } } ( t )$ <sub>.</sub> L<sub>e</sub>t $N _ { T }$ <sub>coun</sub>t f<sub>a</sub>llb<sub>ac</sub>k<sub>s</sub> <sub>up</sub> t<sub>o</sub> $T ,$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t $N _ { \mathrm { I G W } } : = \operatorname* { s u p } \{ t \geq 1$ 4αs $t < G _ { \mathrm { I G W } } ( t ) \}$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> fi<sub>n</sub>it<sub>e s</sub>i<sub>nce</sub> $G _ { \mathrm { I G W } } ( t ) = o ( t )$

Theorem 4.1 (Anytime safety and regret) Under Assumptions $I , 2 , 3 ,$ <sub>, an</sub>d 5<sub>, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta ,$ , simultaneously for every $T \geq 1$ , Al<sub>g</sub>orithm 2 is α-conservative in the sense of (3), $N _ { T } \leq \mathrm { m i n } \{ T , N _ { \mathrm { I G W } } \}$ <sub>,</sub> <sub>an</sub>d $\begin{array} { r } { \mathsf { R e g } ( T ) \leq G _ { \mathrm { I G W } } ( T ) + \frac { 1 } { 2 } \Delta _ { h } N _ { T } } \end{array}$ . Suppressing log factors in T and $1 / \delta ,$

$$
\mathsf { R e g } ( T ) = \tilde { \mathcal { O } } \left( \sqrt { K T A ( T ) } + \frac { \Delta _ { h } K A ( T ) } { \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \right) .\tag{22}
$$

Proof sketch. Work on the event of Lemma 4.2. Safety: if $\tau \leq t$ i<sub>s</sub> th<sub>e</sub> l<sub>as</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>,</sub> th<sub>en</sub> <sub>s</sub>i<sub>nce</sub> $r _ { i } ^ { \circ } \geq 0$ <sub>an</sub>d <sub>cre</sub>dit<sub>s are pos</sub>iti<sub>ve,</sub>

$$
\sum _ { i \in \mathcal { A } _ { t } } R _ { i } \le \sum _ { i \le \tau } r _ { i } ^ { \circ } \le G _ { \mathrm { I G W } } ( \tau ) \le 4 Q _ { \tau } \le 4 Q _ { t } ,
$$

<sub>so</sub> th<sub>e</sub> t<sub>rue</sub> <sub>surp</sub>l<sub>us</sub> i<sub>s</sub> <sub>nonnega</sub>ti<sub>ve.</sub>

Fallbacks: every fallback time satisfies 4αs $\tau < G _ { \mathrm { I G W } } ( \tau )$ , so $\tau \leq N _ { \mathrm { I G W } } ;$ <sub>a</sub>t th<sub>e</sub> l<sub>as</sub>t f<sub>a</sub>llb<sub>ac</sub>k $\tau \leq T$ $G _ { \mathrm { I G W } } ( \tau ) = \tilde { \mathcal { O } } ( \sqrt { K \tau A ( T ) } )$ <sub>g</sub><sup>i</sup>ves $N _ { T } = \tilde { \mathcal { O } } ( K A ( T ) / ( \alpha ^ { 2 } \underline { { s } } ^ { 2 } ) )$ .

<sup>R</sup>e<sub>g</sub>ret: $R _ { t } = r _ { t } ^ { \circ }$ <sub>on can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>s an</sub>d $\begin{array} { r } { R _ { t } \leq \frac { 1 } { 2 } ( \Delta _ { h } + r _ { t } ^ { \circ } ) } \end{array}$ <sub>on</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>s, so</sub>

$$
\begin{array} { r } { \mathsf { R e g } ( T ) \leq G _ { \mathrm { I G W } } ( T ) + \frac { 1 } { 2 } \Delta _ { h } N _ { T } . } \end{array}
$$

D<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>are</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>6<sub>.</sub>

Remark 5 (Finite classes) The oracle in Appendix B.7 has $\mathcal { R } _ { \mathrm { l o g } } ( n ) = e ^ { B } \log | \mathcal { F } |$ , so $A ( T ) \equiv$ $A _ { \mathcal { F } } : = e ^ { B } \log | \mathcal { F } | / \kappa + 1 / \kappa ^ { 2 }$ and, as in the linear case, the cost of conservatism is independent $o f T$

Remark 6 (From C-SquareCB to preference feedback) C-SquareCB (Deb et al., 2025a) certifies safety by calibrating cumulative predictions of cardinal costs. Our oracle controls only comparison margins, so we certify scores through cumulative regret (Lemmas 3.3 and 4.2), and Lemma 4.1 lets fallbacks keep learning. Appendix B.1 explains why the calibration argument does not transfer.

![](images/7d4ec21e0e3d1657f8cfe2b37e4605edfc8dad765a37ab6247ac57bf703935da.jpg)  
(a) d = 10, α = 0.05

![](images/928d9730dd33aa0920c08ecab7f35fd56f0cdcfa6954488a2f5537384a00b52b.jpg)  
(b) Varying α (d = 10)

![](images/3ef2a7a17c3d3549ba54e2bf46124b8e240f875d3e368f40f29fdbb6777d1fbf.jpg)  
(c) Varying d (α = 0.05)  
Fi<sub>gure</sub> 1<sub>:</sub> Li<sub>near u</sub>tiliti<sub>es.</sub> T<sub>op: cumu</sub>l<sub>a</sub>ti<sub>ve regre</sub>t<sub>.</sub> B<sub>o</sub>tt<sub>om: mean sa</sub>f<sub>e</sub>t<sub>y ra</sub>ti<sub>o</sub> $\bar { \rho } _ { t }$ in (a) and worst violation in (b, c). Dashed lines mark the bar 1 − α in (a) and zero violation in (b, c); the vertical band in (a) marks the rounds where MaxInP is below the bar. Baseline-onl<sub>y</sub> is shown in (a) onl<sub>y</sub>. Intervals are 95% confidence intervals over 50 seeds. MaxInP9BC has zero violation in every setting.

## 5. Experiments

We com<sub>p</sub>are each conservative al<sub>g</sub>orithm with its unconstrained counter<sub>p</sub>art, MaxInP (Saha, 2021) or BT-IGW (Sekhari et al., 2023), and with baseline-onl<sub>y</sub>, which selects the baseline item.

Protocol. Each configuration runs for $T = 1 0 ^ { 4 }$ rounds on 50 independent instances. We report cumulative regret and the safety ratio $\rho _ { t } = \sum _ { i < t }$ max $\begin{array} { r } { \{ s _ { i } ( \mathbf { x } _ { i } ) , s _ { i } ( \mathbf { y } _ { i } ) \} / \sum _ { i < t } s _ { i } ( \mathbf { x } _ { b _ { i } } ) ; } \end{array}$ <sub>;</sub> th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t (3) holds at <sub>p</sub>refix t if and onl<sub>y</sub> if $\rho _ { t } \ge 1 - \alpha$ . Sweeps report the worst violation max<sub>t</sub> $[ ( 1 - \alpha ) - \bar { \rho } _ { t } ] _ { - }$ + <sub>o</sub>f th<sub>e cross-see</sub>d <sub>mean</sub> $\bar { \rho } _ { t }$ . Intervals are 95% confidence intervals over seeds. The baseline is the it<sub>em a</sub>t f<sub>rac</sub>ti<sub>ona</sub>l <sub>u</sub>tilit<sub>y ran</sub>k $q ,$ th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f it<sub>ems w</sub>ith hi<sub>g</sub>h<sub>er u</sub>tilit<sub>y.</sub> W<sub>e</sub> t<sub>une on</sub>l<sub>y</sub> th<sub>e mu</sub>lti<sub>p</sub>li<sub>ers</sub> on $\beta _ { t } , \gamma _ { e , m }$ <sub>an</sub>d $G _ { \mathrm { I G W } }$ , by grid search on seeds disjoint from those reported: unconstrained learners minimise regret, and conservative learners minimise regret subject to the constraint. Tuned multipliers f<sub>a</sub>ll <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> th<sub>eory, so</sub> th<sub>e sa</sub>f<sub>e</sub>t<sub>y repor</sub>t<sub>e</sub>d h<sub>ere</sub> i<sub>s emp</sub>i<sub>r</sub>i<sub>ca</sub>l<sub>.</sub>

Linear utilities: We take d = 10, K = 50, α = 0.05 and $q = 0 . 1 4 .$ , with β-multipliers 0.02 for <sup>M</sup>ax<sup>I</sup>n<sup>P</sup> an<sup>d</sup> 0.05 <sup>f</sup>or MaxInP<sup>9</sup>BC. <sup>M</sup>ax<sup>I</sup>n<sup>P</sup> atta<sup>i</sup>ns t<sup>h</sup>e <sup>l</sup>owest regret, <sup>b</sup>ut <sup>i</sup>ts mean sa<sup>f</sup>ety rat<sup>i</sup>o <sup>d</sup>rops to 0.839 and stays below the bar for roughly the first 100 rounds, because its most-uncertain pair can contain two poor items. MaxInP9BC never drops below 0.991, at 1.39× the regret of MaxInP, while baseline-onl<sub>y</sub> is safe but incurs linear regret, 3.5× that of MaxInP9BC (Figure 1(a)). The sweeps con<sup>fi</sup>rm t<sup>hi</sup>s across sett<sup>i</sup>ngs: MaxInP<sup>9</sup>BC <sup>h</sup>as zero v<sup>i</sup>o<sup>l</sup>at<sup>i</sup>on <sup>f</sup>or every α an<sup>d</sup> $d ,$ <sub>w</sub>h<sub>ereas</sub> M<sub>ax</sub>I<sub>n</sub>P <sub>v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>n a</sub>ll b<sub>u</sub>t th<sub>e</sub> l<sub>ooses</sub>t <sub>se</sub>tti<sub>ng.</sub>

General utilities: Here d = 5, K = 40, α = 0.02 and $q = 0 . 1 0$ <sub>, w</sub>ith <sub>a rea</sub>li<sub>za</sub>bl<sub>e c</sub>l<sub>ass o</sub>f $| \mathcal { F } | = 6 4$ <sub>ran</sub>d<sub>om qua</sub>d<sub>ra</sub>ti<sub>cs an</sub>d <sub>an exponen</sub>ti<sub>a</sub>l<sub>-we</sub>i<sub>g</sub>ht<sub>s orac</sub>l<sub>e,</sub> f<sub>or w</sub>hi<sub>c</sub>h $\mathcal { R } _ { \mathrm { l o g } } ( n ) = \log | \mathcal { F } |$ <sub>.</sub> BT<sub>-</sub>IGW h<sub>as</sub> the lowest regret (464.5), but its mean safety ratio starts at 0.862 and stays below the bar for the first ≈190 rounds, whereas BTIGW9BC stays at or above 1.014 at 2.69× the regret, and baseline-only is a further 2.47× worse (Figure 2(a)). Across the sweeps, BTIGW9BC has zero violation while BT-IGW <sub>v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>n</sub> <sub>every</sub> <sub>se</sub>tti<sub>ng.</sub> Th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f <sub>conserva</sub>ti<sub>sm</sub> i<sub>s</sub> l<sub>arges</sub>t <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>s</sub> ti<sub>g</sub>ht (small α, large K), and when it is loose (α ≥ 0.05 or K = 10), BTIGW9BC matches or beats BT-IGW.

![](images/e992c4e43e9ed253fa6049ad0105a5c077eab53038b1ce87f2abd9e2f989176f.jpg)  
(a) K = 40, α = 0.02

![](images/ed9bbe885d98f5ad1f3c4cf724c60e838730d8fca2e7f95688477dd0f7d60fb2.jpg)  
(b) Varying α (K = 40)

![](images/1e1f65680e6fc2c109193a53dcf47f43a3b4859b8161ffaa27c79046b15f0c53.jpg)  
(c) Varying K (α = 0.02)  
Fi<sub>gure</sub> 2<sub>:</sub> G<sub>enera</sub>l <sub>u</sub>tiliti<sub>es</sub> $( d = 5 , | \mathcal { F } | = 6 4 , q = 0 . 1 0 )$ <sub>,</sub> l<sub>a</sub>id <sub>ou</sub>t <sub>as</sub> i<sub>n</sub> Fi<sub>gure</sub> 1 <sub>w</sub>ith BT<sub>-</sub>IGW i<sub>n</sub> <sub>p</sub>l<sub>ace</sub> o<sup>f</sup> <sup>M</sup>ax<sup>I</sup>n<sup>P</sup>. BTIGW<sup>9</sup>BC stays a<sup>b</sup>ove t<sup>h</sup>e <sup>b</sup>ar <sup>i</sup>n (a), an<sup>d</sup> <sup>i</sup>ts extra regret over <sup>BT</sup>-<sup>IGW</sup> ar<sup>i</sup>ses ma<sup>i</sup>n<sup>l</sup>y <sub>w</sub>h<sub>ere</sub> BT<sub>-</sub>IGW <sub>v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub>

## 6. Conclusion

W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d <sub>conserva</sub>ti<sub>ve con</sub>t<sub>ex</sub>t<sub>ua</sub>l d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s, w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>re</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve score o</sub>f th<sub>e</sub> better displayed item to stay above a (1 − α) fraction of a trusted baseline’s at all times. Because the <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>s s</sub>t<sub>a</sub>t<sub>e</sub>d th<sub>roug</sub>h <sub>pre</sub>f<sub>erence scores ra</sub>th<sub>er</sub> th<sub>an rewar</sub>d<sub>s,</sub> it i<sub>s</sub> i<sub>nvar</sub>i<sub>an</sub>t t<sub>o s</sub>hift<sub>s o</sub>f th<sub>e u</sub>tilit<sub>y</sub> f<sub>unc</sub>ti<sub>on an</sub>d <sub>can</sub> b<sub>e en</sub>f<sub>orce</sub>d f<sub>rom</sub> bi<sub>nary compar</sub>i<sub>sons a</sub>l<sub>one.</sub> W<sub>e s</sub>h<sub>owe</sub>d th<sub>a</sub>t thi<sub>s se</sub>tti<sub>ng</sub> i<sub>s no</sub>t <sub>a</sub> <sub>rou</sub>ti<sub>ne</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>on</sub> <sub>o</sub>f <sub>conserva</sub>ti<sub>ve</sub> b<sub>an</sub>dit<sub>s</sub> <sub>w</sub>ith <sub>numer</sub>i<sub>ca</sub>l <sub>rewar</sub>d<sub>s,</sub> <sub>s</sub>i<sub>nce</sub> t<sub>rans</sub>f<sub>err</sub>i<sub>ng</sub> CLUCB di<sub>rec</sub>tl<sub>y</sub> can <sup>i</sup>ncur <sup>li</sup>near regret. MaxInP<sup>9</sup>BC an<sup>d</sup> BTIGW<sup>9</sup>BC avo<sup>id</sup> t<sup>hi</sup>s <sup>f</sup>a<sup>il</sup>ure <sup>b</sup>y cert<sup>if</sup>y<sup>i</sup>ng t<sup>h</sup>e constra<sup>i</sup>nt th<sub>roug</sub>h <sub>u</sub>tilit<sub>y gaps</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>o</sub>b<sub>serve</sub>d <sub>payo</sub>f<sub>s an</sub>d b<sub>y con</sub>ti<sub>nu</sub>i<sub>ng</sub> t<sub>o</sub> l<sub>earn</sub> f<sub>rom</sub> th<sub>e compar</sub>i<sub>sons</sub> <sub>ma</sub>d<sub>e w</sub>h<sub>en</sub> th<sub>ey</sub> f<sub>a</sub>ll b<sub>ac</sub>k t<sub>o</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub> B<sub>o</sub>th <sub>are sa</sub>f<sub>e a</sub>t <sub>a</sub>ll ti<sub>mes w</sub>ith hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> f<sub>or</sub> li<sub>near</sub> <sub>an</sub>d <sub>genera</sub>l <sub>u</sub>tilit<sub>y</sub> <sub>c</sub>l<sub>asses</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> I<sub>n</sub> b<sub>o</sub>th <sub>cases,</sub> <sub>conserva</sub>ti<sub>sm</sub> <sub>a</sub>dd<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>an</sub> <sub>a</sub>dditi<sub>ve</sub> t<sub>erm</sub> t<sub>o</sub> th<sub>e</sub> <sub>regre</sub>t<sub>.</sub> O<sub>ver</sub> l<sub>ong</sub> h<sub>or</sub>i<sub>zons,</sub> th<sub>e</sub> <sub>pr</sub>i<sub>ce</sub> <sub>o</sub>f <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>neg</sub>li<sub>g</sub>ibl<sub>e</sub> <sub>compare</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f l<sub>earn</sub>i<sub>ng</sub> it<sub>se</sub>lf<sub>.</sub>

S<sub>evera</sub>l di<sub>rec</sub>ti<sub>ons rema</sub>i<sub>n open.</sub> O<sub>ne</sub> i<sub>s conserva</sub>ti<sub>ve pre</sub>f<sub>erence-</sub>b<sub>ase</sub>d <sub>re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> (Garcelon et al., 2020b), where a <sub>p</sub>olic<sub>y</sub> learns from trajector<sub>y</sub> com<sub>p</sub>arisons while its cumulative <sub>score s</sub>t<sub>ays c</sub>l<sub>ose</sub> t<sub>o</sub> th<sub>a</sub>t <sub>o</sub>f <sub>a</sub> d<sub>ep</sub>l<sub>oye</sub>d b<sub>ase</sub>li<sub>ne po</sub>li<sub>cy.</sub> C<sub>er</sub>tifi<sub>ca</sub>t<sub>es</sub> th<sub>ere mus</sub>t <sub>accoun</sub>t f<sub>or</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>e</sub> l<sub>earner</sub>’<sub>s own po</sub>li<sub>cy, no</sub>t <sub>an exogenous con</sub>t<sub>ex</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> A<sub>no</sub>th<sub>er</sub> is re<sub>p</sub>licable constrained duelin<sub>g</sub> bandits (Im<sub>p</sub>a<sub>g</sub>liazzo et al., 2022; Deb et al., 2026a,b), where the d<sub>a</sub>t<sub>a-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>accep</sub>t<sub>ance</sub> t<sub>es</sub>t <sub>mus</sub>t <sub>g</sub>i<sub>ve</sub> th<sub>e</sub> <sub>same</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>across</sub> <sub>runs</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>comprom</sub>i<sub>s</sub>i<sub>ng</sub> <sub>any</sub>ti<sub>me sa</sub>f<sub>e</sub>t<sub>y.</sub> Cl<sub>oser</sub> t<sub>o</sub> th<sub>e presen</sub>t <sub>se</sub>tti<sub>ng,</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>compar</sub>i<sub>sons</sub> i<sub>nvo</sub>l<sub>ve</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne an</sub>d <sub>are</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve a</sub>b<sub>ou</sub>t it<sub>s score, so</sub> b<sub>ase</sub>li<sub>ne scores cou</sub>ld b<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>on</sub>li<sub>ne ra</sub>th<sub>er</sub> th<sub>an</sub> b<sub>oun</sub>d<sub>e</sub>d in advance as conservative bandits do for unknown baseline rewards (Wu et al. 2016 Kazerouni et al. 2017). A stricter, sta ewise variant would re uire ever dis la ed air, rather than the cumulative <sub>sum,</sub> t<sub>o mee</sub>t th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> b<sub>ar, ana</sub>l<sub>ogous</sub> t<sub>o s</sub>t<sub>agew</sub>i<sub>se sa</sub>f<sub>e</sub>t<sub>y cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> f<sub>or</sub> b<sub>an</sub>dit<sub>s w</sub>ith <sub>numer</sub>i<sub>ca</sub>l rewards (Khezeli and Bitar, 2020; Moradi<sub>p</sub>ari et al., 2020). Under non-transitive <sub>p</sub>references be<sub>y</sub>ond th<sub>e</sub> B<sub>ra</sub>dl<sub>ey–</sub>T<sub>erry–</sub>L<sub>uce mo</sub>d<sub>e</sub>l<sub>, scores wou</sub>ld <sub>nee</sub>d t<sub>o</sub> b<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>aga</sub>i<sub>ns</sub>t <sub>a von</sub> N<sub>eumann w</sub>i<sub>nner</sub> (Dudík et al., 2015) rather than a best item. Finall<sub>y</sub>, lower bounds on the cost of conservatism and f<sub>ee</sub>db<sub>ac</sub>k <sub>over</sub> <sub>more</sub> th<sub>an</sub> t<sub>wo</sub> it<sub>ems</sub> <sub>rema</sub>i<sub>n</sub> <sub>open.</sub>

## References

Y<sub>un</sub>t<sub>ao</sub> B<sub>a</sub>i<sub>,</sub> A<sub>n</sub>d<sub>y</sub> J<sub>ones,</sub> K<sub>ama</sub>l Nd<sub>ousse,</sub> A<sub>man</sub>d<sub>a</sub> A<sub>s</sub>k<sub>e</sub>ll<sub>,</sub> A<sub>nna</sub> Ch<sub>en,</sub> N<sub>ova</sub> D<sub>as</sub>S<sub>arma,</sub> D<sub>awn</sub> D<sub>ra</sub>i<sub>n,</sub> St<sub>an</sub>i<sub>s</sub>l<sub>av</sub> F<sub>or</sub>t<sub>,</sub> D<sub>eep</sub> G<sub>angu</sub>li<sub>,</sub> T<sub>om</sub> H<sub>en</sub>i<sub>g</sub>h<sub>an, e</sub>t <sub>a</sub>l<sub>.</sub> T<sub>ra</sub>i<sub>n</sub>i<sub>ng a</sub> h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l <sub>an</sub>d h<sub>arm</sub>l<sub>ess ass</sub>i<sub>s</sub>t<sub>an</sub>t <sub>w</sub>ith <sub>re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> f<sub>rom</sub> h<sub>uman</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>,</sub> 2022<sub>.</sub>

Vikt<sub>or</sub> B<sub>engs,</sub> Rób<sub>er</sub>t B<sub>usa-</sub>F<sub>e</sub>k<sub>e</sub>t<sub>e,</sub> Adil El M<sub>esaou</sub>di<sub>-</sub>P<sub>au</sub>l<sub>, an</sub>d E<sub>y</sub>k<sub>e</sub> Hüll<sub>erme</sub>i<sub>er.</sub> P<sub>re</sub>f<sub>erence-</sub>b<sub>ase</sub>d online learnin<sub>g</sub> with duelin<sub>g</sub> bandits: A surve<sub>y</sub>. Journal of Machine Learnin<sub>g</sub> Research, 22(7): 1<sub>–</sub>108<sub>,</sub> 2021<sub>.</sub>

Vikt<sub>or</sub> B<sub>engs,</sub> A<sub>a</sub>di<sub>rupa</sub> S<sub>a</sub>h<sub>a,</sub> <sub>an</sub>d E<sub>y</sub>k<sub>e</sub> Hüll<sub>erme</sub>i<sub>er.</sub> St<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s</sub> <sub>un</sub>d<sub>er</sub> linear stochastic transitivity models. In International Conference on Machine Learning, pages 1764<sub>–</sub>1786<sub>.</sub> PMLR<sub>,</sub> 2022<sub>.</sub>

R<sub>a</sub>l<sub>p</sub>h All<sub>an</sub> B<sub>ra</sub>dl<sub>ey</sub> <sub>an</sub>d Milt<sub>on</sub> E<sub>.</sub> T<sub>erry.</sub> R<sub>an</sub>k <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f i<sub>ncomp</sub>l<sub>e</sub>t<sub>e</sub> bl<sub>oc</sub>k d<sub>es</sub>i<sub>gns:</sub> I<sub>.</sub> th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d of <sub>p</sub>aired com<sub>p</sub>arisons. Biometrika, 39(3/4):324–345, 1952. ISSN 00063444, 14643510. URL http://www.jstor.org/stable/2334029.

Eric Brochu, Nando de Freitas, and Abhijeet Ghosh. Active preference learning with discrete choice data. In Advances in Neural Information Processing Systems 20 (NIPS), 2007.

Oli<sub>v</sub>i<sub>er</sub> Ch<sub>ape</sub>ll<sub>e,</sub> Th<sub>ors</sub>t<sub>en</sub> J<sub>oac</sub>hi<sub>ms,</sub> Fili<sub>p</sub> R<sub>a</sub>dli<sub>ns</sub>ki<sub>, an</sub>d Yi<sub>song</sub> Y<sub>ue.</sub> L<sub>arge-sca</sub>l<sub>e va</sub>lid<sub>a</sub>ti<sub>on</sub> and anal<sub>y</sub>sis of interleaved search evaluation. ACM Transactions on Information S<sub>y</sub>stems, 30(1): 6<sub>:</sub>1<sub>–</sub>6<sub>:</sub>41<sub>,</sub> 2012<sub>.</sub>

Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep <sub>re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> f<sub>rom</sub> h<sub>uman pre</sub>f<sub>erences.</sub> I<sub>n</sub> N<sub>eur</sub>IPS<sub>,</sub> 2017<sub>.</sub>

Rohan Deb, Yikun Ban, Shiliang Zuo, Jingrui He, and Arindam Banerjee. Contextual bandits with online neural regression. In The Twelfth International Conference on Learning Representations, 2024<sub>a.</sub> URL htt<sub>ps:</sub>//<sub>ope</sub>nr<sub>ev</sub>i<sub>ew.</sub>n<sub>e</sub>t/f<sub>o</sub>r<sub>u</sub>m?id<sub>=</sub>5<sub>ep</sub>85<sub>sa</sub>kT3<sub>.</sub>

Rohan Deb, Aadirupa Saha, and Arindam Banerjee. Think before you duel: Understanding complexities of preference learning under constrained resources. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li, editors, Proceedings ofThe 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 4546<sub>–</sub>4554<sub>.</sub> PMLR<sub>,</sub> 02<sub>–</sub>04 M<sub>ay</sub> 2024b<sub>.</sub> URL htt<sub>ps:</sub>//<sub>procee</sub>di<sub>ngs.m</sub>l<sub>r.press</sub>/<sub>v</sub>238/d<sub>e</sub>b24<sub>a.</sub>ht<sub>m</sub>l<sub>.</sub>

Rohan Deb, Mohammad Ghavamzadeh, and Arindam Banerjee. Conservative contextual bandits: Beyond linear representations. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum?id=SThJXvucjQ.

Rohan Deb, Mohammad Ghavamzadeh, and Arindam Banerjee. Thompson sampling for constrained bandits. In Reinforcement Learning Conference, 2025b. URL https://openreview.net/forum?id= 5HL4QaNQcf.

Rohan Deb, Udaya Ghai, Karan Singh, and Arindam Banerjee. Replicable bandits with ucb based <sub>exp</sub>l<sub>ora</sub>ti<sub>on,</sub> 2026<sub>a.</sub> URL htt<sub>ps:</sub>//<sub>arx</sub>i<sub>v.org</sub>/<sub>a</sub>b<sub>s</sub>/2604<sub>.</sub>20024<sub>.</sub>

Rohan Deb, Aadirupa Saha, Udaya Ghai, Karan Singh, and Arindam Banerjee. Replicable dueling b<sub>a</sub>ndit<sub>s,</sub> 2026b<sub>.</sub> URL htt<sub>ps:</sub>//<sub>a</sub>r<sub>x</sub>i<sub>v.o</sub>r<sub>g</sub>/<sub>a</sub>b<sub>s</sub>/2610<sub>.</sub>20908<sub>.</sub>

Miroslav Dudík, Katja Hofmann, Robert E. Schapire, Aleksandrs Slivkins, and Masrour Zoghi. C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s.</sub> I<sub>n</sub> P<sub>e</sub>t<sub>er</sub> G<sub>r</sub>ü<sub>nwa</sub>ld<sub>,</sub> El<sub>a</sub>d H<sub>azan,</sub> <sub>an</sub>d S<sub>a</sub>t<sub>yen</sub> K<sub>a</sub>l<sub>e,</sub> <sub>e</sub>dit<sub>ors,</sub> P<sub>rocee</sub>di<sub>ngs</sub> of The 28th Conference on Learning Theory, volume 40 of Proceedings of Machine Learning R<sub>esearc</sub>h<sub>, pages</sub> 563<sub>–</sub>587<sub>.</sub> PMLR<sub>,</sub> 2015<sub>.</sub> URL htt<sub>ps:</sub>//<sub>p</sub>r<sub>ocee</sub>din<sub>gs.</sub>mlr<sub>.p</sub>r<sub>ess</sub>/<sub>v</sub>40/D<sub>u</sub>dik15<sub>.</sub>html<sub>.</sub>

Johannes Fürnkranz and Eyke Hüllermeier, editors. Preference Learning. Springer, 2010.

E<sub>vrar</sub>d G<sub>arce</sub>l<sub>on,</sub> M<sub>o</sub>h<sub>amma</sub>d Gh<sub>avamza</sub>d<sub>e</sub>h<sub>,</sub> Al<sub>essan</sub>d<sub>ro</sub> L<sub>azar</sub>i<sub>c,</sub> <sub>an</sub>d M<sub>a</sub>tt<sub>eo</sub> Pi<sub>ro</sub>tt<sub>a.</sub> I<sub>mprove</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or conserva</sub>ti<sub>ve exp</sub>l<sub>ora</sub>ti<sub>on</sub> i<sub>n</sub> b<sub>an</sub>dit<sub>s.</sub> I<sub>n</sub> AAAI<sub>,</sub> 2020<sub>a.</sub>

E<sub>vrar</sub>d G<sub>arce</sub>l<sub>on,</sub> M<sub>o</sub>h<sub>amma</sub>d Gh<sub>avamza</sub>d<sub>e</sub>h<sub>,</sub> Al<sub>essan</sub>d<sub>ro</sub> L<sub>azar</sub>i<sub>c, an</sub>d M<sub>a</sub>tt<sub>eo</sub> Pi<sub>ro</sub>tt<sub>a.</sub> C<sub>onserva</sub>ti<sub>ve</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub> i<sub>n re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng.</sub> I<sub>n</sub> Sil<sub>v</sub>i<sub>a</sub> Chi<sub>appa an</sub>d R<sub>o</sub>b<sub>er</sub>t<sub>o</sub> C<sub>a</sub>l<sub>an</sub>d<sub>ra, e</sub>dit<sub>ors,</sub> P<sub>rocee</sub>di<sub>ngs</sub> of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 1431–1441. PMLR, 2020b. URL htt<sub>ps:</sub>//<sub>p</sub>r<sub>ocee</sub>din<sub>gs.</sub>mlr<sub>.p</sub>r<sub>ess</sub>/<sub>v</sub>108/<sub>ga</sub>r<sub>ce</sub>l<sub>o</sub>n20<sub>a.</sub>html<sub>.</sub>

J<sub>av</sub>i<sub>er</sub> G<sub>onz</sub>ál<sub>ez,</sub> Zh<sub>enwen</sub> D<sub>a</sub>i<sub>,</sub> A<sub>n</sub>d<sub>reas</sub> D<sub>am</sub>i<sub>anou, an</sub>d N<sub>e</sub>il D<sub>.</sub> L<sub>awrence.</sub> P<sub>re</sub>f<sub>eren</sub>ti<sub>a</sub>l B<sub>ayes</sub>i<sub>an</sub> optimization. In Proceedings of the 34th International Conference on Machine Learning (ICML), <sub>vo</sub>l<sub>ume</sub> 70 <sub>o</sub>f PMLR<sub>,</sub> <sub>pages</sub> 1282<sub>–</sub>1291<sub>,</sub> 2017<sub>.</sub>

Katja Hofmann, Lihong Li, and Filip Radlinski. Online evaluation for information retrieval. Foundations and Trends in Information Retrieval, 10(1):1–117, 2016.

David R Hunter. Mm al<sub>g</sub>orithms for <sub>g</sub>eneralized bradle<sub>y</sub>–terr<sub>y</sub> models. Annals of Statistics, 32(1): 384<sub>–</sub>406<sub>,</sub> 2004<sub>.</sub>

R<sub>usse</sub>ll I<sub>mpag</sub>li<sub>azzo,</sub> R<sub>ex</sub> L<sub>e</sub>i<sub>,</sub> T<sub>on</sub>i<sub>ann</sub> Pit<sub>ass</sub>i<sub>,</sub> <sub>an</sub>d J<sub>ess</sub>i<sub>ca</sub> S<sub>orre</sub>ll<sub>.</sub> R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y</sub> i<sub>n</sub> l<sub>earn</sub>i<sub>ng.</sub> I<sub>n</sub> Proceedings ofthe 54th Annual ACM SIGACT Symposium on Theory ofComputing (STOC), pages 818<sub>–</sub>831<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>ti<sub>ng</sub> M<sub>ac</sub>hi<sub>nery,</sub> 2022<sub>.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>1145/3519935<sub>.</sub>3519973<sub>.</sub>

Thorsten Joachims. Optimizing search engines using clickthrough data. In Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 133<sub>–</sub>142<sub>,</sub> 2002<sub>.</sub>

Ni<sub>co</sub>l<sub>as</sub> J<sub>ones,</sub> A<sub>rme</sub>ll<sub>e</sub> B<sub>run,</sub> <sub>an</sub>d A<sub>nne</sub> B<sub>oyer.</sub> C<sub>ompar</sub>i<sub>sons</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>ra</sub>ti<sub>ngs:</sub> T<sub>owar</sub>d<sub>s</sub> <sub>more</sub> stable preferences. In 2011 IEEE/WIC/ACM International Conferences on Web Intelligence and I<sub>n</sub>t<sub>e</sub>lli<sub>gen</sub>t A<sub>gen</sub>t T<sub>ec</sub>h<sub>no</sub>l<sub>ogy, vo</sub>l<sub>ume</sub> 1<sub>, pages</sub> 451<sub>–</sub>456<sub>,</sub> 2011<sub>.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>1109/WI<sub>-</sub>IAT<sub>.</sub>2011<sub>.</sub>13<sub>.</sub>

S<sub>a</sub>iki<sub>s</sub>h<sub>ore</sub> K<sub>a</sub>ll<sub>oor</sub>i<sub>,</sub> F<sub>rancesco</sub> Ri<sub>cc</sub>i<sub>, an</sub>d M<sub>ar</sub>k<sub>o</sub> Tk<sub>a</sub>l<sub>c</sub>i<sub>c.</sub> P<sub>a</sub>i<sub>rw</sub>i<sub>se pre</sub>f<sub>erences</sub> b<sub>ase</sub>d <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> factorization and nearest neighbor recommendation techniques. In Proceedings ofthe 10th ACM Conference on Recommender Systems (RecSys), 2016.

Abbas Kazerouni, Mohammad Ghavamzadeh, Yasin Abbasi Yadkori, and Benjamin Van Roy. C<sub>o</sub>n<sub>se</sub>r<sub>va</sub>ti<sub>ve co</sub>nt<sub>ex</sub>t<sub>ua</sub>l lin<sub>ea</sub>r b<sub>a</sub>ndit<sub>s.</sub> In I<sub>.</sub> G<sub>uyo</sub>n<sub>,</sub> U<sub>.</sub> V<sub>o</sub>n L<sub>ux</sub>b<sub>u</sub>r<sub>g,</sub> S<sub>.</sub> B<sub>e</sub>n<sub>g</sub>i<sub>o,</sub> H<sub>.</sub> W<sub>a</sub>ll<sub>ac</sub>h<sub>,</sub> R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing S<sub>ys</sub>t<sub>ems,</sub> <sub>vo</sub>l<sub>u</sub>m<sub>e</sub> 30<sub>.</sub> C<sub>u</sub>rr<sub>a</sub>n A<sub>ssoc</sub>i<sub>a</sub>t<sub>es,</sub> In<sub>c.,</sub> 2017<sub>.</sub> URL htt<sub>ps:</sub>//<sub>p</sub>r<sub>ocee</sub>din<sub>gs.</sub>n<sub>eu</sub>ri<sub>ps.cc</sub>/<sub>pape</sub>r<sub>\_</sub> fil<sub>es</sub>/<sub>paper</sub>/2017/fil<sub>e</sub>/bd<sub>c</sub>4626<sub>aa</sub>1d1df8<sub>e</sub>14d80d345b2<sub>a</sub>442d<sub>-</sub>P<sub>aper.p</sub>df<sub>.</sub>

Kia Khezeli and Eilyan Bitar. Safe linear stochastic bandits. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 10202–10209, 2020. doi: 10.1609/aaai.v34i06.6581.

Xuhen<sub>g</sub> Li, He<sub>y</sub>an<sub>g</sub> Zhao, and Quanquan Gu. Feel-<sub>g</sub>ood thompson samplin<sub>g</sub> for contextual duelin<sub>g</sub> b<sub>an</sub>dit<sub>s.</sub> I<sub>n</sub> R<sub>us</sub>l<sub>an</sub> S<sub>a</sub>l<sub>a</sub>kh<sub>u</sub>tdi<sub>nov,</sub> Zi<sub>co</sub> K<sub>o</sub>lt<sub>er,</sub> K<sub>a</sub>th<sub>er</sub>i<sub>ne</sub> H<sub>e</sub>ll<sub>er,</sub> Ad<sub>r</sub>i<sub>an</sub> W<sub>e</sub>ll<sub>er,</sub> N<sub>ur</sub>i<sub>a</sub> Oli<sub>ver,</sub> Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 29406– 29426<sub>.</sub> PMLR<sub>,</sub> 21<sub>–</sub>27 J<sub>u</sub>l 2024<sub>.</sub> URL htt<sub>ps:</sub>//<sub>procee</sub>di<sub>ngs.m</sub>l<sub>r.press</sub>/<sub>v</sub>235/li24<sub>co.</sub>ht<sub>m</sub>l<sub>.</sub>

R<sub>.</sub> D<sub>uncan</sub> L<sub>uce.</sub> I<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l Ch<sub>o</sub>i<sub>ce</sub> B<sub>e</sub>h<sub>av</sub>i<sub>or:</sub> A Th<sub>eore</sub>ti<sub>ca</sub>l A<sub>na</sub>l<sub>ys</sub>i<sub>s.</sub> C<sub>our</sub>i<sub>er</sub> C<sub>orpora</sub>ti<sub>on,</sub> 2005<sub>.</sub>

Ah<sub>ma</sub>d<sub>reza</sub> M<sub>ora</sub>di<sub>par</sub>i<sub>,</sub> Ch<sub>r</sub>i<sub>s</sub>t<sub>os</sub> Th<sub>rampou</sub>lidi<sub>s, an</sub>d M<sub>a</sub>h<sub>noos</sub>h Ali<sub>za</sub>d<sub>e</sub>h<sub>.</sub> St<sub>age-w</sub>i<sub>se conserva</sub>ti<sub>ve</sub> linear bandits. In Advances in Neural Information Processing Systems, volume 33, pages 11191– 11201<sub>,</sub> 2020<sub>.</sub>

L<sub>o</sub>n<sub>g</sub> O<sub>uya</sub>n<sub>g,</sub> J<sub>e</sub>fr<sub>ey</sub> W<sub>u,</sub> X<sub>u</sub> Ji<sub>a</sub>n<sub>g,</sub> Di<sub>ogo</sub> Alm<sub>e</sub>id<sub>a,</sub> C<sub>a</sub>rr<sub>o</sub>ll W<sub>a</sub>in<sub>w</sub>ri<sub>g</sub>ht<sub>,</sub> P<sub>a</sub>m<sub>e</sub>l<sub>a</sub> Mi<sub>s</sub>hkin<sub>,</sub> Ch<sub>o</sub>n<sub>g</sub> Zh<sub>ang,</sub> S<sub>an</sub>dhi<sub>n</sub>i A<sub>garwa</sub>l<sub>,</sub> K<sub>a</sub>t<sub>ar</sub>i<sub>na</sub> Sl<sub>ama,</sub> Al<sub>ex</sub> R<sub>ay,</sub> J<sub>o</sub>h<sub>n</sub> S<sub>c</sub>h<sub>u</sub>l<sub>man,</sub> J<sub>aco</sub>b Hilt<sub>on,</sub> F<sub>raser</sub> K<sub>e</sub>lt<sub>on,</sub> L<sub>u</sub>k<sub>e</sub> Mill<sub>er,</sub> M<sub>a</sub>ddi<sub>e</sub> Si<sub>mens,</sub> A<sub>man</sub>d<sub>a</sub> A<sub>s</sub>k<sub>e</sub>ll<sub>,</sub> P<sub>e</sub>t<sub>er</sub> W<sub>e</sub>li<sub>n</sub>d<sub>er,</sub> P<sub>au</sub>l F Ch<sub>r</sub>i<sub>s</sub>ti<sub>ano,</sub> J<sub>an</sub> L<sub>e</sub>ik<sub>e, an</sub>d R<sub>yan</sub> L<sub>owe.</sub> T<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>o</sub> f<sub>o</sub>ll<sub>ow</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>ons w</sub>ith h<sub>uman</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022<sub>.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>52202/068431<sub>-</sub>2011<sub>.</sub> URL htt<sub>ps:</sub>//<sub>procee</sub>di<sub>ngs.neur</sub>i<sub>ps.cc</sub>/<sub>paper\_</sub>fil<sub>es</sub>/<sub>paper</sub>/2022/ fil<sub>e</sub>/b1<sub>e</sub>fd<sub>e</sub>53b<sub>e</sub>364<sub>a</sub>73914f58805<sub>a</sub>001731<sub>-</sub>P<sub>aper-</sub>C<sub>on</sub>f<sub>erence.p</sub>df<sub>.</sub>

Fili<sub>p</sub> R<sub>a</sub>dli<sub>ns</sub>ki<sub>,</sub> M<sub>a</sub>dh<sub>u</sub> K<sub>urup,</sub> <sub>an</sub>d Th<sub>ors</sub>t<sub>en</sub> J<sub>oac</sub>hi<sub>ms.</sub> H<sub>ow</sub> d<sub>oes</sub> <sub>c</sub>li<sub>c</sub>kth<sub>roug</sub>h d<sub>a</sub>t<sub>a</sub> <sub>re</sub>fl<sub>ec</sub>t <sub>re</sub>t<sub>r</sub>i<sub>eva</sub>l quality? In Proceedings of the 17th ACM Conference on Information and Knowledge Management (CIKM), pages 43–52, 2008.

R<sub>a</sub>f<sub>ae</sub>l R<sub>a</sub>f<sub>a</sub>il<sub>ov,</sub> A<sub>rc</sub>hit Sh<sub>arma,</sub> E<sub>r</sub>i<sub>c</sub> Mit<sub>c</sub>h<sub>e</sub>ll<sub>,</sub> Ch<sub>r</sub>i<sub>s</sub>t<sub>op</sub>h<sub>er</sub> D M<sub>ann</sub>i<sub>ng,</sub> St<sub>e</sub>f<sub>ano</sub> E<sub>rmon, an</sub>d Ch<sub>e</sub>l<sub>sea</sub> Fi<sub>nn.</sub> Di<sub>rec</sub>t <sub>pre</sub>f<sub>erence op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on:</sub> Y<sub>our</sub> l<sub>anguage mo</sub>d<sub>e</sub>l i<sub>s secre</sub>tl<sub>y a rewar</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> I<sub>n</sub> A<sub>.</sub> Oh<sub>,</sub> T<sub>.</sub> N<sub>aumann,</sub> A<sub>.</sub> Gl<sub>o</sub>b<sub>erson,</sub> K<sub>.</sub> S<sub>aen</sub>k<sub>o,</sub> M<sub>.</sub> H<sub>ar</sub>dt<sub>,</sub> <sub>an</sub>d S<sub>.</sub> L<sub>ev</sub>i<sub>ne,</sub> <sub>e</sub>dit<sub>ors,</sub> Ad<sub>vances</sub> i<sub>n</sub> N<sub>eura</sub>l Information Processing Systems, volume 36, pages 53728–53741. Curran Associates, Inc., 2023. d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>52202/075280<sub>-</sub>2338<sub>.</sub> URL htt<sub>ps:</sub>//<sub>p</sub>r<sub>ocee</sub>din<sub>gs.</sub>n<sub>eu</sub>ri<sub>ps.cc</sub>/<sub>pape</sub>r<sub>\_</sub>fil<sub>es</sub>/<sub>pape</sub>r/2023/fil<sub>e</sub>/ <sub>a</sub>85b405<sub>e</sub>d65<sub>c</sub>6477<sub>a</sub>4f<sub>e</sub>8302b5<sub>e</sub>06<sub>ce</sub>7<sub>-</sub>P<sub>aper-</sub>C<sub>on</sub>f<sub>erence.p</sub>df<sub>.</sub>

A<sub>a</sub>di<sub>rupa</sub> S<sub>a</sub>h<sub>a.</sub> O<sub>p</sub>ti<sub>ma</sub>l <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>pre</sub>f<sub>erence</sub> b<sub>an</sub>dit<sub>s.</sub> I<sub>n</sub> M<sub>.</sub> R<sub>an-</sub> <sub>za</sub>t<sub>o,</sub> A<sub>.</sub> B<sub>eyge</sub>l<sub>z</sub>im<sub>e</sub>r<sub>,</sub> Y<sub>.</sub> D<sub>aup</sub>hin<sub>,</sub> P<sub>.</sub>S<sub>.</sub> Li<sub>a</sub>n<sub>g, a</sub>nd J<sub>.</sub> W<sub>o</sub>rtm<sub>a</sub>n V<sub>aug</sub>h<sub>a</sub>n<sub>, e</sub>dit<sub>o</sub>r<sub>s,</sub> Ad<sub>-</sub> vances in Neural Information Processing Systems, volume 34, pages 30050–30062. Cur-

r<sub>a</sub>n A<sub>ssoc</sub>i<sub>a</sub>t<sub>es,</sub> In<sub>c.,</sub> 2021<sub>.</sub> URL htt<sub>ps:</sub>//<sub>p</sub>r<sub>ocee</sub>din<sub>gs.</sub>n<sub>eu</sub>ri<sub>ps.cc</sub>/<sub>pape</sub>r<sub>\_</sub>fil<sub>es</sub>/<sub>pape</sub>r/2021/fil<sub>e</sub>/ f<sub>c</sub>3<sub>c</sub>f452d3d<sub>a</sub>8402b<sub>e</sub>bb765225<sub>ce</sub>8<sub>c</sub>0<sub>e-</sub>P<sub>aper.p</sub>df<sub>.</sub>

A<sub>yus</sub>h S<sub>e</sub>kh<sub>ar</sub>i<sub>,</sub> K<sub>ar</sub>thik S<sub>r</sub>idh<sub>aran,</sub> W<sub>en</sub> S<sub>un, an</sub>d R<sub>unz</sub>h<sub>e</sub> W<sub>u.</sub> C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l b<sub>an</sub>dit<sub>s an</sub>d i<sub>m</sub>it<sub>a</sub>ti<sub>on</sub> learning with preference-based active queries. In Advances in Neural Information Processi<sub>ng</sub> S<sub>ys</sub>t<sub>ems,</sub> <sub>vo</sub>l<sub>u</sub>m<sub>e</sub> 36<sub>,</sub> 2023<sub>.</sub> URL htt<sub>ps:</sub>//<sub>p</sub>r<sub>ocee</sub>din<sub>gs.</sub>n<sub>eu</sub>ri<sub>ps.cc</sub>/<sub>pape</sub>r<sub>\_</sub>fil<sub>es</sub>/<sub>pape</sub>r/2023/fil<sub>e</sub>/ 2567<sub>c</sub>95fd41459<sub>a</sub>98<sub>a</sub>73b<sub>a</sub>893775d22<sub>a-</sub>P<sub>aper-</sub>C<sub>on</sub>f<sub>erence.p</sub>df<sub>.</sub>

Ni<sub>sa</sub>n Sti<sub>e</sub>nn<sub>o</sub>n<sub>,</sub> L<sub>o</sub>n<sub>g</sub> O<sub>uya</sub>n<sub>g,</sub> J<sub>e</sub>fr<sub>ey</sub> W<sub>u,</sub> D<sub>a</sub>ni<sub>e</sub>l Zi<sub>eg</sub>l<sub>e</sub>r<sub>,</sub> R<sub>ya</sub>n L<sub>owe,</sub> Ch<sub>e</sub>l<sub>sea</sub> V<sub>oss,</sub> Al<sub>ec</sub> R<sub>a</sub>df<sub>o</sub>rd<sub>,</sub> D<sub>ar</sub>i<sub>o</sub> A<sub>mo</sub>d<sub>e</sub>i<sub>, an</sub>d P<sub>au</sub>l F<sub>.</sub> Ch<sub>r</sub>i<sub>s</sub>ti<sub>ano.</sub> L<sub>earn</sub>i<sub>ng</sub> t<sub>o summar</sub>i<sub>ze w</sub>ith h<sub>uman</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> I<sub>n</sub> Ad<sub>vances</sub> in Neural Information Processing Systems 33 (NeurIPS), pages 3008–3021, 2020.

M<sub>aegan</sub> T<sub>uc</sub>k<sub>er,</sub> Ell<sub>en</sub> N<sub>ovose</sub>ll<sub>er,</sub> Cl<sub>au</sub>di<sub>a</sub> K<sub>ann,</sub> Y<sub>anan</sub> S<sub>u</sub>i<sub>,</sub> Yi<sub>song</sub> Y<sub>ue,</sub> J<sub>oe</sub>l W<sub>.</sub> B<sub>ur</sub>di<sub>c</sub>k<sub>,</sub> <sub>an</sub>d A<sub>aron</sub> D<sub>.</sub> A<sub>mes.</sub> P<sub>re</sub>f<sub>erence-</sub>b<sub>ase</sub>d l<sub>earn</sub>i<sub>ng</sub> f<sub>or</sub> <sub>exos</sub>k<sub>e</sub>l<sub>e</sub>t<sub>on</sub> <sub>ga</sub>it <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on.</sub> I<sub>n</sub> IEEE I<sub>n</sub>t<sub>erna</sub>ti<sub>ona</sub>l Conference on Robotics and Automation (ICRA), pages 2351–2357, 2020.

Ar<sub>u</sub>n V<sub>e</sub>rm<sub>a,</sub> Zh<sub>o</sub>n<sub>gx</sub>i<sub>a</sub>n<sub>g</sub> D<sub>a</sub>i<sub>,</sub> Xi<sub>aoq</sub>i<sub>a</sub>n<sub>g</sub> Lin<sub>,</sub> P<sub>a</sub>tri<sub>c</sub>k J<sub>a</sub>ill<sub>e</sub>t<sub>, a</sub>nd Br<sub>ya</sub>n Ki<sub>a</sub>n H<sub>s</sub>i<sub>a</sub>n<sub>g</sub> L<sub>ow.</sub> N<sub>eu</sub>r<sub>a</sub>l d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s:</sub> P<sub>re</sub>f<sub>erence-</sub>b<sub>ase</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>w</sub>ith h<sub>uman</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> I<sub>n</sub> Th<sub>e</sub> Thi<sub>r</sub>t<sub>een</sub>th I<sub>n</sub>t<sub>erna-</sub> tional Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= VELh<sub>v</sub>9BBf<sub>n.</sub>

Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> Wi<sub>r</sub>th<sub>,</sub> Ri<sub>a</sub>d Ak<sub>rour,</sub> G<sub>er</sub>h<sub>ar</sub>d N<sub>eumann,</sub> <sub>an</sub>d J<sub>o</sub>h<sub>annes</sub> Fü<sub>rn</sub>k<sub>ranz.</sub> A <sub>survey</sub> <sub>o</sub>f <sub>pre</sub>f<sub>erence-</sub> based reinforcement learnin<sub>g</sub> methods. Journal ofMachine Learnin<sub>g</sub> Research, 18(136):1–46, 2017<sub>.</sub>

Yif<sub>an</sub> W<sub>u,</sub> R<sub>os</sub>h<sub>an</sub> Sh<sub>ar</sub>if<sub>,</sub> T<sub>or</sub> L<sub>a</sub>tti<sub>more, an</sub>d C<sub>sa</sub>b<sub>a</sub> S<sub>zepesvar</sub>i<sub>.</sub> C<sub>onserva</sub>ti<sub>ve</sub> b<sub>an</sub>dit<sub>s.</sub> I<sub>n</sub> Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings ofThe 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, <sub>pages</sub> 1254<sub>–</sub>1262<sub>,</sub> N<sub>ew</sub> Y<sub>or</sub>k<sub>,</sub> N<sub>ew</sub> Y<sub>or</sub>k<sub>,</sub> USA<sub>,</sub> 20<sub>–</sub>22 J<sub>un</sub> 2016<sub>.</sub> PMLR<sub>.</sub> URL htt<sub>ps:</sub>//<sub>procee</sub>di<sub>ngs.</sub> <sub>m</sub>l<sub>r.press</sub>/<sub>v</sub>48/<sub>wu</sub>16<sub>.</sub>ht<sub>m</sub>l<sub>.</sub>

Yi<sub>song</sub> Y<sub>ue</sub> <sub>an</sub>d Th<sub>ors</sub>t<sub>en</sub> J<sub>oac</sub>hi<sub>ms.</sub> I<sub>n</sub>t<sub>erac</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>op</sub>ti<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>re</sub>t<sub>r</sub>i<sub>eva</sub>l <sub>sys</sub>t<sub>ems</sub> <sub>as</sub> <sub>a</sub> dueling bandits problem. In Proceedings ofthe 26th Annual International Conference on Machine L<sub>earn</sub>i<sub>ng,</sub> <sub>pages</sub> 1201<sub>–</sub>1208<sub>,</sub> 2009<sub>.</sub>

Yi<sub>song</sub> Y<sub>ue,</sub> J<sub>ose</sub>f B<sub>ro</sub>d<sub>er,</sub> R<sub>o</sub>b<sub>er</sub>t Kl<sub>e</sub>i<sub>n</sub>b<sub>erg, an</sub>d Th<sub>ors</sub>t<sub>en</sub> J<sub>oac</sub>hi<sub>ms.</sub> Th<sub>e</sub> k<sub>-arme</sub>d d<sub>ue</sub>li<sub>ng</sub> b<sub>an</sub>dit<sub>s</sub> <sub>p</sub>roblem. J. Com<sub>p</sub>ut. S<sub>y</sub>st. Sci., 78(5):1538–1556, Se<sub>p</sub>tember 2012. ISSN 0022-0000. doi: 10.1016/j.jcss.2011.12.028. URL https://doi.org/10.1016/j.jcss.2011.12.028.

Zhibi<sub>ng</sub> Zh<sub>ao,</sub> P<sub>e</sub>t<sub>er</sub> Pi<sub>ec</sub>h<sub>, an</sub>d Li<sub>rong</sub> Xi<sub>a.</sub> L<sub>earn</sub>i<sub>ng m</sub>i<sub>x</sub>t<sub>ures o</sub>f <sub>p</sub>l<sub>ac</sub>k<sub>e</sub>tt<sub>-</sub>l<sub>uce mo</sub>d<sub>e</sub>l<sub>s,</sub> 2020<sub>.</sub> URL htt<sub>ps:</sub>//<sub>a</sub>r<sub>x</sub>i<sub>v.o</sub>r<sub>g</sub>/<sub>a</sub>b<sub>s</sub>/1603<sub>.</sub>07323<sub>.</sub>

D<sub>an</sub>i<sub>e</sub>l M<sub>.</sub> Zi<sub>eg</sub>l<sub>er,</sub> Ni<sub>san</sub> Sti<sub>ennon,</sub> J<sub>e</sub>f<sub>rey</sub> W<sub>u,</sub> T<sub>om</sub> B<sub>.</sub> B<sub>rown,</sub> Al<sub>ec</sub> R<sub>a</sub>df<sub>or</sub>d<sub>,</sub> D<sub>ar</sub>i<sub>o</sub> A<sub>mo</sub>d<sub>e</sub>i<sub>,</sub> P<sub>au</sub>l Ch<sub>r</sub>i<sub>s</sub>ti<sub>ano,</sub> <sub>an</sub>d G<sub>eo</sub>f<sub>rey</sub> I<sub>rv</sub>i<sub>ng.</sub> Fi<sub>ne-</sub>t<sub>un</sub>i<sub>ng</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>rom</sub> h<sub>uman</sub> <sub>pre</sub>f<sub>erences,</sub> 2020<sub>.</sub> URL htt<sub>ps:</sub>//<sub>a</sub>r<sub>x</sub>i<sub>v.o</sub>r<sub>g</sub>/<sub>a</sub>b<sub>s</sub>/1909<sub>.</sub>08593<sub>.</sub>

## Appendix A. Proofs for Section 3

Th<sub>roug</sub>h<sub>ou</sub>t thi<sub>s appen</sub>di<sub>x</sub> A<sub>ssump</sub>ti<sub>ons</sub> 1<sub>,</sub> 2<sub>,</sub> 3 <sub>an</sub>d 4 <sub>are</sub> i<sub>n</sub> f<sub>orce, an</sub>d $\lambda > 0 , \alpha , \delta \in ( 0 , 1 )$ <sub>are</sub> th<sub>e</sub> i<sub>npu</sub>t<sub>s o</sub>f Al<sub>gor</sub>ith<sub>m</sub> 1<sub>.</sub>

## A.1. Notation and filtration

R<sub>eca</sub>ll ${ \bf z } _ { t } = { \bf x } _ { t } - { \bf y } _ { t }$ is the comparison vector of the pair actually displayed at round t, that $V _ { 1 } = \lambda I _ { d }$ <sub>an</sub>d $V _ { t + 1 } = V _ { t } + \mathbf { z } _ { t } \mathbf { z } _ { t } ^ { \top }$ <sub>, an</sub>d <sub>wr</sub>it<sub>e</sub>

$$
\varepsilon _ { t } : = o _ { t } - \sigma ( \mathbf { z } _ { t } ^ { \mathsf { T } } \theta ^ { \star } ) , \qquad \mathbf { S } _ { t } : = \sum _ { i = 1 } ^ { t } \varepsilon _ { i } \mathbf { z } _ { i } , \qquad v _ { t } : = \left. \mathbf { z } _ { t } \right. _ { V _ { t } ^ { - 1 } } .
$$

L<sub>e</sub>t $\mathcal { F } _ { t - 1 }$ be the σ-algebra generated by all contexts, displayed pairs and outcomes of rounds $1 , \ldots , t - 1$ , together with the round-t context set $S _ { t }$ <sub>,</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> it<sub>em</sub> $\mathbf { x } _ { b _ { t } }$ <sub>an</sub>d it<sub>s score, an</sub>d th<sub>e</sub> round-t displayed pair. Every quantity the algorithm computes at round $t - V _ { t } , \hat { \theta } _ { t } , \beta _ { t } , \mathcal { C } _ { t } , ( \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } )$ $w _ { t } , c _ { t } , \Lambda _ { t - 1 }$ <sub>,</sub> th<sub>e accep</sub>t<sub>ance</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on, an</sub>d h<sub>ence</sub> $\mathbf { z } _ { t } - \mathrm { i s } \mathcal { F } _ { t }$ <sub>−1-measura</sub>bl<sub>e,</sub> <sub>an</sub>d $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ i<sub>s a</sub> filt<sub>ra</sub>ti<sub>on</sub> <sub>w</sub>ith $\mathbb { E } [ o _ { t } \mid { \mathcal { F } } _ { t - 1 } ] = \sigma ( \mathbf { z } _ { t } ^ { \top } \theta ^ { \star } )$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 1<sub>.</sub> U<sub>n</sub>lik<sub>e ana</sub>l<sub>yses</sub> th<sub>a</sub>t di<sub>scar</sub>d <sub>un</sub>i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>roun</sub>d<sub>s, no ran</sub>d<sub>om</sub> ti<sub>me c</sub>h<sub>ange</sub> i<sub>s nee</sub>d<sub>e</sub>d h<sub>ere.</sub>

Remark 7 (Degenerate rounds) A round with $\mathbf { z } _ { t } = 0$ <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> $\varepsilon _ { t } \mathbf { z } _ { t } = 0 \mathrm { ~ } t o \mathbf { S } _ { t } ,$ <sub>,</sub> l<sub>eaves</sub> $V _ { t + 1 } = V _ { t }$ and adds the constant log 2 to the objective in (5); it therefore afects neither the estimator nor any ar<sub>g</sub>ument below. Such a round can onl<sub>y</sub> be a candidate round: b<sub>y</sub> (13), $i f \mathrm { m a x } _ { \mathbf { y } \in \mathcal { C } _ { t } } \| \mathbf { x } _ { b _ { t } } - \mathbf { y } \| _ { V _ { \star } ^ { - 1 } } = 0$ th<sub>en</sub> $w _ { t } = 0$ <sub>,</sub> h<sub>ence</sub> $c _ { t } = 0 _ { : }$ <sub>, an</sub>d th<sub>e accep</sub>t<sub>ance</sub> t<sub>es</sub>t<sub>passes</sub> b<sub>ecause</sub> $\begin{array} { r } { \Lambda _ { t - 1 } \geq 0 a n d \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { t } ( \mathbf { x } _ { b _ { t } } ) \geq } \end{array}$ $\frac { \alpha } { 2 } > 0$ (Lemma A.1). Consequently everyfallback round has $v _ { t } \ge w _ { t } / 2 > 0$

## A.2. Elementary bounds on scores

Lemma A.1 For every t and every $\begin{array} { r } { \mathbf { x } \in \mathcal { S } _ { t } , \sigma ( - B ) \le s _ { t } ( \mathbf { x } ) \le \frac { 1 } { 2 } } \end{array}$ <sub>,</sub> <sub>an</sub>d $\begin{array} { r } { s _ { t } ( \mathbf { x } _ { t } ^ { \star } ) = \frac { 1 } { 2 } } \end{array}$ <sub>.</sub> M<sub>oreover</sub> $\underline { { s } } : = \sigma ( - \Delta _ { h } )$ satisfies $\begin{array} { r } { \underline { { s } } \le s _ { t } ( { \mathbf x } _ { b _ { t } } ) \le \frac { 1 } { 2 } } \end{array}$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t

$$
\begin{array} { r } { \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { t } ( \mathbf x _ { b _ { t } } ) \geq \frac { \alpha } { 2 } \qquad a n d \qquad \alpha s _ { t } ( \mathbf x _ { b _ { t } } ) \geq \alpha \underline { { s } } . } \end{array}
$$

Proof By definition $s _ { t } ( \mathbf { x } ) = \sigma \big ( f ( \mathbf { x } ) - f ( \mathbf { x } _ { t } ^ { \star } ) \big )$ <sub>w</sub>ith $f ( \mathbf { x } ) - f ( \mathbf { x } _ { t } ^ { \star } ) \in [ - B , 0 ]$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 2 <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $\mathbf { x } _ { t } ^ { \star } \mathbf { ; }$ ; monotonicity of σ and $\begin{array} { r } { \sigma ( 0 ) = \frac { 1 } { 2 } } \end{array}$ <sub>g</sub>i<sub>ve</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> F<sub>or</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne,</sub> $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } _ { b _ { t } } ) = \Delta _ { t , b _ { t } } \in [ \Delta _ { l } , \Delta _ { h } ]$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 3<sub>,</sub> <sub>so</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } ) \in [ \sigma ( - \Delta _ { h } ) , \sigma ( - \Delta _ { l } ) ] \subseteq [ \underline { { s } } , \frac { 1 } { 2 } ]$ Fi<sub>na</sub>ll<sub>y</sub> $\begin{array} { r } { \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { t } ( \mathbf { x } _ { b _ { t } } ) \geq \frac { 1 } { 2 } - ( 1 - \alpha ) \frac { 1 } { 2 } = \frac { \alpha } { 2 } } \end{array}$

N<sub>o</sub>t<sub>e</sub> f<sub>or</sub> l<sub>a</sub>t<sub>er use</sub> th<sub>a</sub>t $\underline { { s } } \le \frac { 1 } { 2 }$ <sub>, s</sub>i<sub>nce</sub> $\Delta _ { h } \geq 0$

## A.3. Anytime logistic confidence ellipsoid

Thi<sub>s</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>on</sub> <sub>proves</sub> L<sub>emma</sub> 3<sub>.</sub>1<sub>.</sub> Th<sub>e</sub> <sub>argumen</sub>t i<sub>s</sub> <sub>se</sub>lf<sub>-con</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub>

Lemma A.2 (Hoefding’s lemma) Let X be a random variable supported on an interval oflength $b - a$ . Then E exp $\left( \eta ( X - \mathbb { E } X ) \right) \leq \exp \left( \eta ^ { 2 } ( b - a ) ^ { 2 } / 8 \right) f o r e \nu e r y \eta \in \mathbb { R }$

Proof After the afine rescaling $a = 0 , b = 1$ it <sub>su</sub>fi<sub>ces</sub> t<sub>o</sub> t<sub>rea</sub>t $X \in [ 0 , 1 ]$ <sub>; wr</sub>it<sub>e</sub> $\mu = \mathbb { E } X$ <sub>.</sub> C<sub>o</sub>n<sub>vex</sub>it<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>g</sub>i<sub>ves</sub> $e ^ { \eta x } \leq ( 1 - x ) e ^ { 0 } + x e ^ { \eta }$ f<sub>or</sub> $x \in [ 0 , 1 ]$ <sub>,</sub> <sub>so</sub> t<sub>a</sub>ki<sub>ng</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d <sub>cen</sub>t<sub>r</sub>i<sub>ng,</sub>

$$
\begin{array} { r } { \mathbb { E } e ^ { \eta ( X - \mu ) } \le ( 1 - \mu ) e ^ { - \mu \eta } + \mu e ^ { ( 1 - \mu ) \eta } . } \end{array}\tag{23}
$$

L<sub>e</sub>t $h ( \eta ) = \log \left( ( 1 - \mu ) e ^ { - \mu \eta } + \mu e ^ { ( 1 - \mu ) \eta } \right)$ <sub>an</sub>d $q ( \eta ) ~ = ~ \mu e ^ { ( 1 - \mu ) \eta } / \bigl ( ( 1 - \mu ) e ^ { - \mu \eta } + \mu e ^ { ( 1 - \mu ) \eta } \bigr )$ Dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng,</sub> $h ^ { \prime } ( \eta ) ~ = ~ q ( \eta ) ~ - ~ \mu$ <sub>an</sub>d $h ^ { \prime \prime } ( \eta ) ~ = ~ q ( \eta ) ( 1 - q ( \eta ) ) ~ \leq ~ \frac { 1 } { 4 }$ <sub>, w</sub>hil<sub>e</sub> $h ( 0 ) ~ = ~ 0$ <sub>an</sub>d $h ^ { \prime } ( 0 ) = q ( 0 ) - \mu = 0$ <sub>.</sub> T<sub>ay</sub>l<sub>or</sub>’<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub> <sub>w</sub>ith i<sub>n</sub>t<sub>egra</sub>l <sub>rema</sub>i<sub>n</sub>d<sub>er,</sub> <sub>va</sub>lid f<sub>or</sub> <sub>e</sub>ith<sub>er</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f $\eta ,$ , <sub>g</sub><sup>i</sup>ves

$$
h ( \eta ) = \eta ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - s ) h ^ { \prime \prime } ( s \eta ) \mathrm { d } s \le \frac { \eta ^ { 2 } } { 8 } .
$$

Ex<sub>p</sub>onentiatin<sub>g</sub> and combinin<sub>g</sub> with (23) <sub>p</sub>roves the normalised case; for a <sub>g</sub>eneral interval a<sub>pp</sub>l<sub>y</sub> it to $( X - a ) / ( b - a )$ <sub>w</sub>ith <sub>parame</sub>t<sub>er</sub> $\eta ( b - a )$ ■

Lemma A.3 (Conditional sub-Gaussianity of preference residuals) For every $t \geq 1 , \mathbb { E } [ \varepsilon _ { t } \ ]$ $\mathcal { F } _ { t - 1 } ] = 0$ <sub>an</sub>d $\mathbb { E } [ e ^ { \eta \varepsilon _ { t } } \ | \ \mathcal { F } _ { t - 1 } ] \le e ^ { \eta ^ { 2 } / 8 }$ for all $\eta \in \mathbb { R }$ <sub>.</sub> Th<sub>a</sub>t i<sub>s,</sub> $\varepsilon _ { t }$ is conditionally R-sub-Gaussian <sub>w</sub>ith $\begin{array} { r } { R = \frac { 1 } { 2 } } \end{array}$ i<sub>n</sub> th<sub>e conven</sub>ti<sub>on</sub> $\mathbb E [ e ^ { \eta \varepsilon } \mid \dot { \mathcal F } _ { t - 1 } ] \le \dot { e } ^ { R ^ { 2 } \eta ^ { 2 } / 2 }$

Proof Conditionally on $\mathcal { F } _ { t - 1 }$ th<sub>e ou</sub>t<sub>come</sub> $o _ { t }$ i<sub>s</sub> B<sub>ernou</sub>lli <sub>w</sub>ith <sub>mean</sub> $\sigma ( \mathbf { z } _ { t } ^ { \top } \theta ^ { \star } )$ <sub>, s</sub>i<sub>nce</sub> $\mathbf { z } _ { t }$ i<sub>s</sub> $\mathcal { F } _ { t - 1 } .$ <sub>measura</sub>bl<sub>e;</sub> it i<sub>s suppor</sub>t<sub>e</sub>d <sub>on an</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>o</sub>f l<sub>eng</sub>th <sub>one.</sub> A<sub>pp</sub>l<sub>y</sub> L<sub>emma</sub> $_ { \mathrm { A } . 2 }$ <sub>con</sub>diti<sub>ona</sub>ll<sub>y.</sub>

Lemma A.4 (Ville’s inequality) Let $( M _ { n } ) _ { n \ge 0 }$ b<sub>e a nonnega</sub>ti<sub>ve supermar</sub>ti<sub>nga</sub>l<sub>e w</sub>ith $\mathbb { E } M _ { 0 } \le 1$ Th<sub>en</sub> $\begin{array} { r } { \mathbb { P } ( \operatorname* { s u p } _ { n \geq 0 } M _ { n } \geq c ) \leq 1 / c f o r } \end{array}$ <sup>e</sup>v<sup>er</sup>y $c > 0$

Proof Fix a deterministic $N$ <sub>an</sub>d l<sub>e</sub>t $\tau _ { N } = \operatorname* { m i n } \{ n \leq N : M _ { n } \geq c \}$ <sub>,</sub> <sub>w</sub>ith $\tau _ { N } = N$ if th<sub>e</sub> <sub>se</sub>t i<sub>s</sub> <sub>emp</sub>t<sub>y.</sub> Si<sub>nce</sub> $\{ \tau _ { N } \ge n \} \in \mathcal { F } _ { n - 1 }$

$$
\mathbb { E } M _ { \tau _ { N } } = \mathbb { E } M _ { 0 } + \sum _ { n = 1 } ^ { N } \mathbb { E } \big [ \mathbb { I } \{ \tau _ { N } \ge n \} \mathbb { E } \big [ M _ { n } - M _ { n - 1 } \mid \mathcal { F } _ { n - 1 } \big ] \big ] \ \le \ \mathbb { E } M _ { 0 } \ \le \ 1 .
$$

On the event max $) \leq n \leq N M _ { n } \geq c$ th<sub>e</sub> <sub>s</sub>t<sub>oppe</sub>d <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $^ { c , }$ so $\begin{array} { r } { c \mathbb { P } ( \operatorname* { m a x } _ { n \leq N } M _ { n } \geq c ) \leq } \end{array}$ $\mathbb { E } M _ { \tau _ { N } } \le 1$ <sub>.</sub> L<sub>e</sub>t $N $ ∞ and use continuit<sub>y</sub> from below.

Lemma A.5 (Method-of-mixtures self-normalised bound) Let $( \varepsilon _ { i } , z _ { i } ) _ { i \geq 1 }$ b<sub>e a</sub>d<sub>ap</sub>t<sub>e</sub>d <sub>w</sub>ith $z _ { i } \in$ $\mathbb { R } ^ { d }$ b<sub>e</sub>i<sub>ng</sub> $\mathcal { F } _ { i - 1 }$ <sub>-measura</sub>bl<sub>e</sub> <sub>an</sub>d $\begin{array} { r } { \mathbb { E } \big [ e ^ { \eta \varepsilon _ { i } } \mid \mathcal { F } _ { i - 1 } \big ] \le e ^ { R ^ { 2 } \eta ^ { 2 } / 2 } f o r a l l \eta \in \mathbb { R } . } \end{array}$ <sub>.</sub> L<sub>e</sub>t $\begin{array} { r } { V _ { 0 } \succ 0 , S _ { n } = \sum _ { i < n } } \end{array}$ ε<sub>i</sub>z<sub>i</sub> <sub>an</sub>d $\begin{array} { r } { V _ { n } ^ { \prime } = V _ { 0 } + \sum _ { i \leq n } z _ { i } z _ { i } ^ { \top } } \end{array}$ <sub>.</sub> Th<sub>en</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ,$ simultaneouslyfor every $n \geq 0 _ { : }$

$$
\| S _ { n } \| _ { ( V _ { n } ^ { \prime } ) ^ { - 1 } } ^ { 2 } ~ \le ~ 2 R ^ { 2 } \log \left( \frac { \operatorname * { d e t } ( V _ { n } ^ { \prime } ) ^ { 1 / 2 } } { \operatorname * { d e t } ( V _ { 0 } ) ^ { 1 / 2 } \delta } \right) .\tag{24}
$$

Proof Fix $q \in \mathbb { R } ^ { d }$ <sub>.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>su</sub>b<sub>-</sub>G<sub>auss</sub>i<sub>an</sub>it<sub>y</sub> <sub>w</sub>ith $\eta = q ^ { \top }$ z<sub>i</sub> gives $\begin{array} { r } { \mathbb { E } \big [ \exp \big ( { q ^ { \top } } z _ { i } \varepsilon _ { i } - \frac { R ^ { 2 } } { 2 } ( { q ^ { \top } } z _ { i } ) ^ { 2 } \big ) } \end{array}$ | $\mathcal { F } _ { i - 1 } \big ] \leq 1$ , so

$$
\begin{array} { r } { M _ { n } ( q ) = \exp { \left( q ^ { \top } S _ { n } - \frac { R ^ { 2 } } { 2 } q ^ { \top } ( V _ { n } ^ { \prime } - V _ { 0 } ) q \right) } } \end{array}
$$

i<sub>s a nonnega</sub>ti<sub>ve supermar</sub>ti<sub>nga</sub>l<sub>e w</sub>ith $M _ { 0 } ( q ) = 1$ . Let Q be an independent Gaussian vector <sub>w</sub>ith d<sub>ens</sub>it<sub>y</sub> $f ( q ) = R ^ { d } \operatorname * { d e t } ( V _ { 0 } ) ^ { 1 / 2 } ( 2 \pi ) ^ { - d / 2 } \exp ( - { \textstyle { \frac { R ^ { 2 } } { 2 } } } q ^ { \top } V _ { 0 } q )$ <sub>, w</sub>hi<sub>c</sub>h i<sub>n</sub>t<sub>egra</sub>t<sub>es</sub> t<sub>o one</sub> b<sub>ecause</sub> $\begin{array} { r } { \int e ^ { - \frac { 1 } { 2 } q ^ { \top } A q } \mathrm { d } q = ( 2 \pi ) ^ { d / 2 } \operatorname* { d e t } ( A ) ^ { - 1 / 2 } } \end{array}$ <sub>w</sub>ith $A = R ^ { 2 } V _ { 0 }$ <sub>.</sub> B<sub>y</sub> T<sub>one</sub>lli’<sub>s</sub> th<sub>eorem an</sub>d th<sub>e con</sub>diti<sub>ona</sub>l supermartingale property for each fixed q, the mixture $\begin{array} { r } { M _ { n } = \int M _ { n } ( q ) f ( q ) \mathrm { d } q } \end{array}$ i<sub>s a nonnega</sub>ti<sub>ve</sub> <sub>supermar</sub>ti<sub>nga</sub>l<sub>e w</sub>ith $M _ { 0 } = 1$ <sub>,</sub> <sub>an</sub>d

$$
M _ { n } = \frac { R ^ { d } \operatorname* { d e t } ( V _ { 0 } ) ^ { 1 / 2 } } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } \exp \Big ( q ^ { \top } S _ { n } - \frac { R ^ { 2 } } { 2 } q ^ { \top } V _ { n } ^ { \prime } q \Big ) \mathrm { d } q .
$$

C<sub>omp</sub>l<sub>e</sub>ti<sub>ng</sub> th<sub>e square w</sub>ith $a = R ^ { - 2 } ( V _ { n } ^ { \prime } ) ^ { - 1 } S _ { n }$ <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r } { q ^ { \top } S _ { n } - \frac { R ^ { 2 } } { 2 } q ^ { \top } V _ { n } ^ { \prime } q = - \frac { R ^ { 2 } } { 2 } ( q - a ) ^ { \top } V _ { n } ^ { \prime } ( q - a ) + \frac { 1 } { 2 R ^ { 2 } } \| S _ { n } \| _ { ( V _ { n } ^ { \prime } ) ^ { - 1 } } ^ { 2 } , } \end{array}
$$

<sub>an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> i<sub>n</sub>t<sub>egra</sub>l <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
M _ { n } = \left( \frac { \operatorname * { d e t } V _ { 0 } } { \operatorname * { d e t } V _ { n } ^ { \prime } } \right) ^ { 1 / 2 } \exp \left( \frac { 1 } { 2 R ^ { 2 } } \| S _ { n } \| _ { ( V _ { n } ^ { \prime } ) ^ { - 1 } } ^ { 2 } \right) .\tag{25}
$$

L<sub>emma</sub> A<sub>.</sub>4 <sub>w</sub>ith $c = 1 / \delta$ b<sub>oun</sub>d<sub>s</sub> $\operatorname* { s u p } _ { n } M _ { n } < 1 / \delta$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta ;$ t<sub>a</sub>ki<sub>ng</sub> l<sub>ogar</sub>ith<sub>ms</sub> in (25) and rearran<sub>g</sub>in<sub>g</sub> <sub>g</sub>ives (24). ■

Lemma A.6 (Trace–determinant bound) $I f \| z _ { i } \| _ { 2 } \leq 2 L$ for all i and $\begin{array} { r } { W _ { m } = \lambda I _ { d } + \sum _ { i = 1 } ^ { m - 1 } z _ { i } z _ { i } ^ { \top } } \end{array}$ th<sub>en</sub>

$$
\log \frac { \operatorname * { d e t } W _ { m } } { \operatorname * { d e t } ( \lambda I _ { d } ) } ~ \leq ~ d \log \Bigl ( 1 + \frac { 4 L ^ { 2 } ( m - 1 ) } { \lambda d } \Bigr ) .
$$

Proof Let $\begin{array} { r } { A = \lambda ^ { - 1 } \sum _ { i < m } z _ { i } z _ { i } ^ { \top } \succeq 0 } \end{array}$ <sub>w</sub>ith <sub>e</sub>i<sub>genva</sub>l<sub>ues</sub> $\rho _ { 1 } , \ldots , \rho _ { d } \ge 0$ , so that det $W _ { m } / \operatorname* { d e t } ( \lambda I _ { d } ) =$ $\Pi _ { j } ( 1 + \rho _ { j } )$ <sub>.</sub> B<sub>y</sub> th<sub>e</sub> <sub>ar</sub>ith<sub>me</sub>ti<sub>c–geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>mean</sub> i<sub>nequa</sub>lit<sub>y,</sub> $\prod _ { j } ( 1 + \rho _ { j } ) \leq \left( { 1 + { { \mathrm { T r } } ( A ) } / { d } } \right) ^ { d }$ <sub>,</sub> <sub>an</sub>d $\begin{array} { r } { \mathrm { T r } ( A ) = \lambda ^ { - 1 } \sum _ { i < m } \| z _ { i } \| _ { 2 } ^ { 2 } \leq 4 L ^ { 2 } ( m - 1 ) / \lambda } \end{array}$ <sub>.</sub> T<sub>a</sub>k<sub>e</sub> l<sub>ogar</sub>ith<sub>ms.</sub> ■

Lemma A.7 (Global curvature of the regularised likelihood) Let $\begin{array} { r } { F _ { t } ( \theta ) = \sum _ { i < t } \big [ \log ( 1 + e ^ { \mathbf { z } _ { i } ^ { \top } \theta } ) - } \end{array}$ $\begin{array} { r } { o _ { i } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \theta } ] + \frac { \lambda } { 2 } \| \boldsymbol { \theta } \| _ { 2 } ^ { 2 } } \end{array}$ <sub>.</sub> F<sub>or</sub> <sub>a</sub>ll $\theta , \theta ^ { \prime } \in \Theta$

$$
\left. \nabla F _ { t } ( { \boldsymbol { \theta } } ) - \nabla F _ { t } ( { \boldsymbol { \theta } } ^ { \prime } ) , { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { \prime } \right. \geq \kappa \| { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { \prime } \| _ { V _ { t } } ^ { 2 } .
$$

Proof For $\vartheta \in \Theta$ <sub>we</sub> h<sub>ave</sub> $\begin{array} { r } { \nabla ^ { 2 } F _ { t } ( \vartheta ) = \sum _ { i < t } \sigma ^ { \prime } ( \mathbf { z } _ { i } ^ { \top } \vartheta ) \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } + \lambda I _ { d } } \end{array}$ <sub>.</sub> Si<sub>nce</sub> $\| \mathbf { z } _ { i } \| _ { 2 } \leq 2 L$ <sub>an</sub>d $\| \vartheta \| _ { 2 } \leq S$ $| \mathbf { z } _ { i } ^ { \top } \vartheta | \le 2 L S = B$ , so $\sigma ^ { \prime } ( \mathbf { z } _ { i } ^ { \top } \vartheta ) \geq \kappa$ <sub>an</sub>d th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $\sigma ^ { \prime }$ i<sub>s</sub> <sub>even</sub> <sub>an</sub>d d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n</sub> $| s |$ <sub>.</sub> A<sub>s</sub> $\begin{array} { r } { 0 < \kappa \leq \frac { 1 } { 4 } < 1 } \end{array}$

$$
\nabla ^ { 2 } F _ { t } ( \vartheta ) \ \succeq \ \kappa \sum _ { i < t } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } + \lambda I _ { d } \ \succeq \ \kappa \Big ( \sum _ { i < t } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } + \lambda I _ { d } \Big ) = \kappa V _ { t } .
$$

W<sub>r</sub>iti<sub>ng</sub> $\mathbf { u } = \boldsymbol \theta - \boldsymbol \theta ^ { \prime }$ and noting Θ is convex, the fundamental theorem of calculus gives $\langle \nabla F _ { t } ( { \boldsymbol { \theta } } ) -$ $\begin{array} { r } { \nabla F _ { t } ( \bar { \theta ^ { \prime } } ) , \mathbf { u } \rangle = \int _ { 0 } ^ { 1 } \mathbf { u } ^ { \top } \nabla ^ { 2 } F _ { t } ( \bar { \theta ^ { \prime } } + s \mathbf { u } ) \mathbf { u } \mathrm { d } s \geq \kappa \mathbf { u } ^ { \top } V _ { t } \mathbf { u } } \end{array}$

Lemma 3.1 (restated). With probability at least $1 - \delta ,$ <sub>,</sub> th<sub>e even</sub>t $\mathcal { E } = \{ \| \hat { \theta } _ { t } - \theta ^ { \star } \| _ { V _ { t } } \leq \beta _ { t } \forall t \geq 1 \}$ h<sub>o</sub>ld<sub>s, w</sub>ith $\beta _ { t }$ as in (6).

Proof Set $\mathbf { u } _ { t } = \hat { \theta } _ { t } - \theta ^ { \star }$ <sub>.</sub> Si<sub>nce</sub> $\widehat { \theta } _ { t }$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ses</sub> th<sub>e</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e convex</sub> $F _ { t }$ over the closed convex Θ <sub>an</sub>d $\theta ^ { \star } \in \Theta$ <sub>,</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves</sub> $\langle \nabla F _ { t } ( \hat { \theta } _ { t } ) , \theta ^ { \star } - \hat { \theta } _ { t } \rangle \geq 0 , { \mathrm { i . e . } } \langle \nabla F _ { t } ( \hat { \theta } _ { t } ) , \mathbf { u } _ { t } \rangle \leq 0$ <sub>.</sub> L<sub>emma</sub> $\mathrm { A } . 7$ <sub>w</sub>ith $\theta = \widehat { \theta } _ { t } , \theta ^ { \prime } = \theta ^ { \star }$ th<sub>en y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { r } { \kappa \| \mathbf { u } _ { t } \| _ { V _ { t } } ^ { 2 } \leq \big \langle \nabla F _ { t } ( \hat { \theta } _ { t } ) - \nabla F _ { t } ( \theta ^ { \star } ) , \mathbf { u } _ { t } \big \rangle \leq - \big \langle \nabla F _ { t } ( \theta ^ { \star } ) , \mathbf { u } _ { t } \big \rangle . } \end{array}\tag{26}
$$

Th<sub>e gra</sub>di<sub>en</sub>t <sub>a</sub>t th<sub>e</sub> t<sub>ru</sub>th i<sub>s</sub> $\begin{array} { r } { \nabla F _ { t } ( \theta ^ { \star } ) = \sum _ { i < t } \big ( \sigma ( \mathbf { z } _ { i } ^ { \top } \theta ^ { \star } ) - o _ { i } \big ) \mathbf { z } _ { i } + \lambda \theta ^ { \star } = - \mathbf { S } _ { t - 1 } + \lambda \theta ^ { \star } } \end{array}$ , so <sup>b</sup><sub>y</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz</sub> i<sub>n</sub> th<sub>e</sub> $V _ { t }$ g<sup>eometr</sup>y,

$$
- \big \langle \nabla F _ { t } ( { \boldsymbol { \theta } } ^ { \star } ) , \mathbf { u } _ { t } \big \rangle = \langle \mathbf { S } _ { t - 1 } , \mathbf { u } _ { t } \rangle - \lambda \langle { \boldsymbol { \theta } } ^ { \star } , \mathbf { u } _ { t } \rangle \leq \Big ( \big \| \mathbf { S } _ { t - 1 } \big \| _ { V _ { t } ^ { - 1 } } + \lambda \big \| { \boldsymbol { \theta } } ^ { \star } \big \| _ { V _ { t } ^ { - 1 } } \Big ) \| \mathbf { u } _ { t } \big \| _ { V _ { t } } .
$$

Si<sub>nce</sub> $V _ { t } \succeq \lambda I _ { d }$ <sub>we</sub> h<sub>ave</sub> $V _ { t } ^ { - 1 } \preceq \lambda ^ { - 1 } I _ { d }$ <sub>,</sub> h<sub>ence</sub> $\lambda \| \theta ^ { \star } \| _ { V _ { \star } ^ { - 1 } } \leq \sqrt { \lambda } \| \theta ^ { \star } \| _ { 2 } \leq \sqrt { \lambda } S$ <sub>.</sub> If $\mathbf { u } _ { t } = 0$ th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m</sub> is trivial; otherwise divide (26) b<sub>y</sub> $\kappa \| \mathbf { u } _ { t } \| _ { V _ { t } }$ t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\| \mathbf { u } _ { t } \| _ { V _ { t } } \leq \frac { 1 } { \kappa } \Bigl ( \| \mathbf { S } _ { t - 1 } \| _ { V _ { t } ^ { - 1 } } + \sqrt { \lambda } S \Bigr ) .\tag{27}
$$

<sup>A</sup><sub>pp</sub><sup>l</sup><sub>y</sub> <sup>L</sup>emma $\mathrm { A . 5 }$ <sub>w</sub>ith $\begin{array} { r } { R = \frac { 1 } { 2 } , V _ { 0 } = \lambda I _ { d } , z _ { i } = \mathbf { z } _ { i } } \end{array}$ <sub>an</sub>d $n = t - 1$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t $V _ { t - 1 } ^ { \prime } = V _ { t } ;$ <sub>:</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta$ <sub>,</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>or</sub> <sub>every</sub> $t \geq 1$ ,

$$
\| \mathbf { S } _ { t - 1 } \| _ { V _ { t } ^ { - 1 } } \leq \frac { 1 } { 2 } \sqrt { 2 \log \left( \frac { \operatorname* { d e t } ( V _ { t } ) ^ { 1 / 2 } } { \operatorname* { d e t } ( \lambda I _ { d } ) ^ { 1 / 2 } \delta } \right) } = \frac { 1 } { 2 } \sqrt { \log \frac { \operatorname* { d e t } V _ { t } } { \operatorname* { d e t } ( \lambda I _ { d } ) } + 2 \log \frac { 1 } { \delta } } .
$$

B<sub>oun</sub>di<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t <sub>ra</sub>ti<sub>o</sub> b<sub>y</sub> L<sub>emma</sub> $_ { \mathrm { A } . 6 }$ <sub>w</sub>ith $m = t$ and substitutin<sub>g</sub> into (27) <sub>g</sub>ives exactl<sub>y</sub> $\beta _ { t }$ of (6).

F<sub>or</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>d<sub>er</sub> <sub>o</sub>f th<sub>e</sub> <sub>appen</sub>di<sub>x</sub> <sub>we</sub> <sub>wor</sub>k <sub>on</sub> $\mathcal { E } ,$ <sub>w</sub>hi<sub>c</sub>h b<sub>y</sub> L<sub>emma</sub> 3<sub>.</sub>1 h<sub>as</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta$ <sub>.</sub> All <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> <sub>are</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> <sub>consequences</sub> <sub>o</sub>f $\mathcal { E } ;$ i<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>no</sub> f<sub>ur</sub>th<sub>er</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> <sub>argumen</sub>t i<sub>s</sub> <sub>a</sub>tt<sub>ac</sub>h<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub>

## A.4. The promising set and the regret certificate

W<sub>r</sub>it<sub>e</sub> $U _ { t } ( \mathbf { x } , \mathbf { y } ) = \langle \hat { \theta } _ { t } , \mathbf { x } - \mathbf { y } \rangle + \beta _ { t } \| \mathbf { x } - \mathbf { y } \| _ { V _ { t } ^ { - 1 } }$ <sub>, so</sub> th<sub>a</sub>t $\mathcal { C } _ { t } = \left\{ \mathbf { x } \in \mathcal { S } _ { t } : U _ { t } ( \mathbf { x } , \mathbf { y } ) \geq 0 \forall \mathbf { y } \in \mathcal { S } _ { t } \right\}$ <sub>as</sub> i<sub>n</sub> (7).

## Lemma A.8 (Nonemptiness) For every round t and every value of $\begin{array} { r } { \cdot \hat { \theta } _ { t } , \mathcal { C } _ { t } \ne \varnothing } \end{array}$

Proof Let $\hat { \mathbf { x } } _ { t } \in \arg \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { S } _ { t } } \langle \hat { \boldsymbol { \theta } } _ { t } , \mathbf { x } \rangle$ . <sup>F</sup>or ever<sub>y</sub> $\mathbf { y } \in S _ { t }$ <sub>we</sub> h<sub>ave</sub> $\langle \hat { \theta } _ { t } , \hat { \mathbf { x } } _ { t } - \mathbf { y } \rangle \geq 0$ <sub>, an</sub>d th<sub>e con</sub>fid<sub>ence</sub> b<sub>onus</sub> i<sub>s nonnega</sub>ti<sub>ve, so</sub> $U _ { t } ( \hat { \mathbf x } _ { t } , \mathbf y ) \geq 0$ <sub>.</sub> H<sub>ence</sub> $\hat { \mathbf { x } } _ { t } \in \mathcal { C } _ { t }$ ■

Lemma A.9 (The optimal item is promising) On $\mathcal { E } , \mathbf { x } _ { t } ^ { \star } \in \mathcal { C } _ { t }$ for every t.

Proof Fix $\mathbf { y } \in S _ { t }$ an<sup>d</sup> <sub>p</sub>ut $\mathbf z = \mathbf x _ { t } ^ { \star } - \mathbf y$ <sub>.</sub> On $\mathcal { E } ,$ C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>wa</sub>r<sub>z</sub> <sub>g</sub>i<sub>ves</sub> $\langle \hat { \theta } _ { t } - \theta ^ { \star } , \mathbf { z } \rangle \geq - \beta _ { t } \Vert \mathbf { z } \Vert _ { V _ { t } ^ { - 1 } }$ <sub>w</sub>h<sub>ence</sub>

$$
\begin{array} { r } { U _ { t } ( \mathbf { x } _ { t } ^ { \star } , \mathbf { y } ) = \langle \hat { { \boldsymbol \theta } } _ { t } , \mathbf { z } \rangle + \beta _ { t } \| \mathbf { z } \| _ { V _ { t } ^ { - 1 } } \geq \langle { \boldsymbol \theta } ^ { \star } , \mathbf { z } \rangle = f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \geq 0 . } \end{array}
$$

Lemma A.10 (Gap of a promising item) On $\mathcal { E } ,$ , <sup>e</sup>v<sup>er</sup>y $\mathbf { x } \in \mathcal { C } _ { t }$ satisfies $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } ) \leq 2 \beta _ { t } \| \mathbf { x } _ { t } ^ { \star } -$ $\mathbf { x } \| _ { V _ { t } ^ { - 1 } }$

Proof Since $\mathbf { x } \in \mathcal { C } _ { t }$ <sub>an</sub>d $\mathbf { x } _ { t } ^ { \star } \in S _ { t }$ <sub>,</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $\mathcal { C } _ { t }$ <sub>w</sub>ith $\mathbf y = \mathbf x _ { t } ^ { \star }$ <sub>g</sub><sup>i</sup>ves $U _ { t } ( \mathbf { x } , \mathbf { x } _ { t } ^ { \star } ) \geq 0 .$ <sub>,</sub> i<sub>.e.</sub>

$$
\begin{array} { r l r } { \langle \hat { \theta } _ { t } , { \bf x } _ { t } ^ { \star } - { \bf x } \rangle } & { { } \le } & { \beta _ { t } \| { \bf x } _ { t } ^ { \star } - { \bf x } \| _ { V _ { t } ^ { - 1 } } . } \end{array}\tag{28}
$$

On E, Cauch<sub>y</sub>–Schwarz also <sub>g</sub>ives

$$
\begin{array} { r } { \langle \theta ^ { \star } - \hat { \theta } _ { t } , \mathbf { x } _ { t } ^ { \star } - \mathbf { x } \rangle \leq \beta _ { t } \| \mathbf { x } _ { t } ^ { \star } - \mathbf { x } \| _ { V _ { t } ^ { - 1 } } . } \end{array}\tag{29}
$$

Addin<sub>g</sub> (28) and (29) <sub>y</sub>ields $\begin{array} { r } { \langle \theta ^ { \star } , \mathbf { x } _ { t } ^ { \star } - \mathbf { x } \rangle \leq 2 \beta _ { t } \| \mathbf { x } _ { t } ^ { \star } - \mathbf { x } \| _ { V _ { t } ^ { - 1 } } } \end{array}$

Lemma 3.2 (restated). $\mathcal { C } _ { t } \neq \emptyset$ <sub>a</sub>l<sub>ways.</sub> O<sub>n</sub> $\mathcal { E } , \mathbf { x } _ { t } ^ { \star } \in \mathcal { C } _ { t }$ <sub>an</sub>d $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \leq c _ { t }$ for every $\mathbf { y } \in \mathcal { C } _ { t } ;$ i<sub>n</sub> particular the candidate pair satisfies $\begin{array} { r } { R _ { t } = f ( \mathbf { x } _ { t } ^ { \star } ) - \frac { 1 } { 2 } \big ( f ( \tilde { \mathbf { x } } _ { t } ) + f ( \tilde { \mathbf { y } } _ { t } ) \big ) \leq c _ { t } } \end{array}$

Proof Nonemptiness is Lemma A.8 and $\mathbf { x } _ { t } ^ { \star } \in \mathcal { C } _ { t }$ i<sub>s</sub> L<sub>emma</sub> A<sub>.</sub>9<sub>.</sub> Fi<sub>x</sub> $\mathbf { y } \in \mathcal { C } _ { t }$ <sub>.</sub> B<sub>o</sub>th $\mathbf { x } _ { t } ^ { \star }$ and y lie in $\mathcal { C } _ { t }$ <sub>,</sub> <sub>an</sub>d $\left( \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } \right)$ <sub>max</sub>i<sub>m</sub>i<sub>ses</sub> $\| \cdot - \cdot \| _ { V _ { t } ^ { - 1 } }$ over $\mathcal { C } _ { t } \times \mathcal { C } _ { t }$ b<sub>y</sub> (10), so $\| \mathbf { x } _ { t } ^ { \star } - \mathbf { y } \| _ { V _ { \star } ^ { - 1 } } \leq w _ { t }$ <sub>.</sub> L<sub>emma</sub> A<sub>.</sub>10 th<sub>en</sub> <sub>g</sub>i<sub>ves</sub> $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \leq 2 \beta _ { t } w _ { t }$ <sub>.</sub> M<sub>oreover</sub> $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \leq B$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 2<sub>.</sub> H<sub>ence</sub> $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \leq$ min $\left\{ B , 2 \beta _ { t } w _ { t } \right\} = c _ { t }$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> thi<sub>s</sub> t<sub>o</sub> $\mathbf { y } = \tilde { \mathbf { x } } _ { t }$ <sub>an</sub>d $\mathbf { y } = \tilde { \mathbf { y } } _ { t }$ <sub>,</sub> b<sub>o</sub>th <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h li<sub>e</sub> i<sub>n</sub> $\mathcal { C } _ { t } .$ <sub>an</sub>d <sub>averag</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> i<sub>nequa</sub>liti<sub>es</sub> <sub>g</sub>i<sub>ves</sub> $R _ { t } \leq c _ { t }$

Lemma 3.3 (restated). If a pair $\displaystyle ( \mathbf { x } , \mathbf { y } )$ satisfies $\begin{array} { r } { f ( \mathbf { x } _ { t } ^ { \star } ) - \frac { 1 } { 2 } ( f ( \mathbf { x } ) + f ( \mathbf { y } ) ) \leq c f o r } \end{array}$ some $c \geq 0 ,$ <sub>,</sub> th<sub>en</sub> max $\{ s _ { t } ( \mathbf { x } ) , s _ { t } ( \mathbf { y } ) \} \geq \sigma ( - c ) \geq { \textstyle { \frac { 1 } { 2 } } } - { \frac { c } { 4 } }$

Proof Write $g _ { \mathbf x } = f ( \mathbf x _ { t } ^ { \star } ) - f ( \mathbf x ) \geq 0$ <sub>an</sub>d $g _ { \mathbf y } = f ( \mathbf x _ { t } ^ { \star } ) - f ( \mathbf y ) \geq 0$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t $s _ { t } ( \mathbf { x } ) = \sigma ( - g _ { \mathbf { x } } )$ <sub>an</sub>d $s _ { t } ( \mathbf { y } ) = \sigma ( - g _ { \mathbf { y } } )$ . Since σ is increasing,

$$
\begin{array} { r } { \operatorname* { m a x } \{ s _ { t } ( \mathbf { x } ) , s _ { t } ( \mathbf { y } ) \} = \sigma \big ( - \operatorname* { m i n } \{ g _ { \mathbf { x } } , g _ { \mathbf { y } } \} \big ) , \qquad \operatorname* { m i n } \{ g _ { \mathbf { x } } , g _ { \mathbf { y } } \} \leq \frac { 1 } { 2 } ( g _ { \mathbf { x } } + g _ { \mathbf { y } } ) \leq c , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y.</sub> F<sub>or</sub> th<sub>e secon</sub>d<sub>,</sub> $\begin{array} { r } { \frac { 1 } { 2 } - \sigma ( - c ) = \sigma ( 0 ) - \sigma ( - c ) = \int _ { - c } ^ { 0 } \sigma ^ { \prime } ( u ) \mathrm { d } u \leq c / 4 } \end{array}$ b<sub>ecause</sub> $\sigma ^ { \prime } \leq { \frac { 1 } { 4 } }$ ever<sub>y</sub>w<sup>h</sup>ere.

Width transfer (13). For every round t, $w _ { t } \leq 2 \| \mathbf x _ { b _ { t } } - \mathbf y _ { t } ^ { \mathrm { f b } } \| _ { V _ { t } ^ { - 1 } }$

Proof $V _ { t } \succ 0 , \operatorname { s o } \parallel \cdot \parallel _ { V _ { t } ^ { - 1 } }$ i<sub>s a norm.</sub> Si<sub>nce</sub> $\tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } \in \mathcal { C } _ { t }$

$$
w _ { t } = \left\| \tilde { { \mathbf { x } } } _ { t } - \tilde { { \mathbf { y } } } _ { t } \right\| _ { V _ { t } ^ { - 1 } } \leq \left\| \tilde { { \mathbf { x } } } _ { t } - { \mathbf { x } } _ { b _ { t } } \right\| _ { V _ { t } ^ { - 1 } } + \left\| { \mathbf { x } } _ { b _ { t } } - \tilde { { \mathbf { y } } } _ { t } \right\| _ { V _ { t } ^ { - 1 } } \leq 2 \operatorname* { m a x } _ { { \mathbf { y } } \in { \cal C } _ { t } } \left\| { \mathbf { x } } _ { b _ { t } } - { \mathbf { y } } \right\| _ { V _ { t } ^ { - 1 } } = 2 \| { \mathbf { x } } _ { b _ { t } } - { \mathbf { y } } _ { t } ^ { \mathrm { { f b } } } \| _ { V _ { t } ^ { - 1 } } .
$$

the last ste<sub>p</sub> b<sub>y</sub> the definition (12) of ${ \bf y } _ { t } ^ { \mathrm { f b } }$ <sub>.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t $\mathbf { x } _ { b _ { t } } \in \mathcal { C } _ { t }$ i<sub>s</sub> <sub>no</sub>t <sub>requ</sub>i<sub>re</sub>d<sub>.</sub>

## A.5. Elliptical potential and the certificate envelope

Lemma A.11 (Matrix determinant recursion) $\boldsymbol { I } f W ^ { \prime } = W + z z ^ { \top }$ <sub>w</sub>ith $W \succ 0$ , then det $W ^ { \prime } =$ det $W \left( 1 + \| z \| _ { W ^ { - 1 } } ^ { 2 } \right)$

Proof Factoring $W ^ { 1 / 2 }$ from both sides, det $( W { + } z z ^ { \top } ) = \operatorname* { d e t } ( W )$ det $( I _ { d } \mathrm { + } w w ^ { \top } )$ <sub>w</sub>ith $w = W ^ { - 1 / 2 } z$ Th<sub>e ran</sub>k<sub>-one ma</sub>t<sub>r</sub>i<sub>x</sub> $w w ^ { \top }$ h<sub>as e</sub>i<sub>genva</sub>l<sub>ues</sub> $\| w \| _ { 2 } ^ { 2 } , 0 , \ldots , 0 .$ , so det $( I _ { d } + w w ^ { \top } ) = 1 + \| w \| _ { 2 } ^ { 2 } =$ $1 + z ^ { \top } W ^ { - 1 } z$

Lemma A.12 (Elliptical potential) Let $W _ { 1 } = \lambda I _ { d }$ <sub>an</sub>d $W _ { i + 1 } = W _ { i } + z _ { i } z _ { i } ^ { \top }$ <sub>w</sub>ith $\| z _ { i } \| _ { 2 } \leq 2 L$ <sub>.</sub> Th<sub>en</sub> for every $m \geq 1$

$$
\sum _ { i = 1 } ^ { m } \operatorname* { m i n } \left\{ 1 , \| z _ { i } \| _ { W _ { i } ^ { - 1 } } ^ { 2 } \right\} \leq 2 d \log \Bigl ( 1 + \frac { 4 L ^ { 2 } m } { \lambda d } \Bigr ) .
$$

Proof First, min $\{ 1 , x \} \leq 2 \log ( 1 + x )$ <sup>f</sup>or ever<sub>y</sub> $x \ge 0 \colon$ if $0 \leq x \leq 1$ then log $\scriptstyle ( 1 + x ) \geq x - x ^ { 2 } / 2 \geq$ $x / 2 ; \operatorname { i f } x \geq 1$ th<sub>en</sub> $2 \log ( 1 + x ) \geq 2 \log 2 > 1$ <sub>.</sub> W<sub>r</sub>iti<sub>ng</sub> $\omega _ { i } ^ { 2 } = \| z _ { i } \| _ { W _ { i } ^ { - 1 } } ^ { 2 }$ <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> L<sub>emma</sub> A<sub>.</sub>11<sub>,</sub>

$$
\sum _ { i = 1 } ^ { m } \operatorname* { m i n } \{ 1 , \omega _ { i } ^ { 2 } \} \leq 2 \sum _ { i = 1 } ^ { m } \log ( 1 + \omega _ { i } ^ { 2 } ) = 2 \log \frac { \operatorname* { d e t } W _ { m + 1 } } { \operatorname* { d e t } W _ { 1 } } .
$$

L<sub>emma</sub> A<sub>.</sub>6 <sub>app</sub>li<sub>e</sub>d <sub>w</sub>ith $m + 1$ in place of m bounds the right side by $2 d \log ( 1 + 4 L ^ { 2 } m / ( \lambda d ) )$ .

Lemma A.13 (Pointwise certificate bound) Let $\bar { \beta } _ { t } ~ = ~ \operatorname* { m a x } \{ B / 2 , \beta _ { t } \}$ . For every round t of Al<sub>gor</sub>ith<sub>m</sub> 1<sub>,</sub>

$$
c _ { t } \leq 4 \bar { \beta } _ { t } \operatorname* { m i n } \{ 1 , v _ { t } \} , \qquad v _ { t } = \| \mathbf { z } _ { t } \| _ { V _ { t } ^ { - 1 } } .
$$

The same bound holds for the certificate $c _ { \tau }$ of a rejected proposal with $v _ { \tau }$ <sub>rep</sub>l<sub>ace</sub>d b<sub>y</sub> $w _ { \tau } =$ $\| \tilde { \mathbf { x } } _ { \tau } - \tilde { \mathbf { y } } _ { \tau } \| _ { V _ { \tau } ^ { - 1 } }$

Proof Recall ${ c _ { t } } = { \operatorname* { m i n } } \{ B , 2 \beta _ { t } w _ { t } \}$ <sub>an</sub>d $2 \bar { \beta } _ { t } \geq B , \bar { \beta } _ { t } \geq \beta _ { t }$ <sub>.</sub> W<sub>e</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h th<sub>e</sub> t<sub>wo</sub> <sub>roun</sub>d t<sub>ypes.</sub>

C<sub>an</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>.</sub> H<sub>e</sub>r<sub>e</sub> ${ \bf z } _ { t } = \tilde { { \bf x } } _ { t } - \tilde { { \bf y } } _ { t } .$ , so $v _ { t } ~ = ~ w _ { t }$ <sub>.</sub> If $v _ { t } \ \leq \ 1$ th<sub>en</sub> $c _ { t } \leq 2 \beta _ { t } v _ { t } \leq 2 \bar { \beta } _ { t } v _ { t } \leq$ $4 \bar { \beta } _ { t } \operatorname* { m i n } \{ 1 , v _ { t } \} . \ \mathrm { I f } \ v _ { t } > 1$ th<sub>en</sub> $c _ { t } \leq B \leq 2 \bar { \beta } _ { t } \leq 4 \bar { \beta } _ { t } = 4 \bar { \beta } _ { t }$ min $\left\{ 1 , v _ { t } \right\}$

F<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>.</sub> H<sub>ere</sub> ${ \bf z } _ { t } = { \bf x } _ { b _ { t } } - { \bf y } _ { t } ^ { \mathrm { f b } }$ , and (13) <sub>g</sub>ives $w _ { t } \leq 2 v _ { t } , \mathrm { s o } c _ { t } \leq$ min $\{ B , 4 \beta _ { t } v _ { t } \}$ <sub>.</sub> If $v _ { t } \leq 1$ th<sub>en</sub> $c _ { t } \leq 4 \beta _ { t } v _ { t } \leq 4 \bar { \beta } _ { t }$ min $\{ 1 , v _ { t } \} ; \mathrm { i f } v _ { t } > 1$ th<sub>en</sub> $c _ { t } \leq B \leq 2 \bar { \beta } _ { t } \leq 4 \bar { \beta } _ { t }$

Th<sub>e</sub> fi<sub>na</sub>l <sub>c</sub>l<sub>a</sub>i<sub>m</sub> i<sub>s</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e-roun</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>on</sub> <sub>w</sub>ith $w _ { \tau }$ i<sub>n</sub> <sub>p</sub>l<sub>ace</sub> <sub>o</sub>f $v _ { \tau } .$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>oes</sub> <sub>no</sub>t <sub>use</sub> th<sub>a</sub>t th<sub>e pa</sub>i<sub>r</sub> i<sub>s</sub> di<sub>sp</sub>l<sub>aye</sub>d<sub>.</sub> ■

Lemma A.14 (Cumulative certificate envelope) On any trajectory of Algorithm 1, for every t,

$$
\sum _ { i = 1 } ^ { t } c _ { i } \ \leq \ G ( t ) .\tag{30}
$$

Moreover, ifthe proposal at round τ is rejected, then

$$
\sum _ { i = 1 } ^ { \tau - 1 } c _ { i } + c _ { \tau } \ \leq \ G ( \tau ) .\tag{31}
$$

Proof Since $t \mapsto \beta _ { t }$ i<sub>s non</sub>d<sub>ecreas</sub>i<sub>ng, so</sub> i<sub>s</sub> $t \mapsto \beta _ { t }$ <sub>,</sub> <sub>an</sub>d h<sub>ence</sub> $\bar { \beta } _ { i } \le \bar { \beta } _ { t }$ f<sub>or</sub> $i \leq t .$ B<sub>y</sub> L<sub>emma</sub> A<sub>.</sub>13 <sub>an</sub>d C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz,</sub>

$$
\sum _ { i = 1 } ^ { t } c _ { i } \ \leq \ 4 \bar { \beta } _ { t } \sum _ { i = 1 } ^ { t } \operatorname* { m i n } \{ 1 , v _ { i } \} \ \leq \ 4 \bar { \beta } _ { t } \sqrt { t \sum _ { i = 1 } ^ { t } \operatorname* { m i n } \{ 1 , v _ { i } \} ^ { 2 } } \ = \ 4 \bar { \beta } _ { t } \sqrt { t \sum _ { i = 1 } ^ { t } \operatorname* { m i n } \{ 1 , v _ { i } ^ { 2 } \} } ,
$$

using min $\{ 1 , v \} ^ { 2 } = \operatorname* { m i n } \{ 1 , v ^ { 2 } \}$ f<sub>or</sub> $v \geq 0$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> L<sub>emma</sub> A<sub>.</sub>12 <sub>w</sub>ith $z _ { i } = \mathbf { z } _ { i }$ <sub>, w</sub>hi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\| \mathbf { z } _ { i } \| _ { 2 } \leq 2 L$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 4<sub>,</sub> b<sub>oun</sub>d<sub>s</sub> th<sub>e</sub> i<sub>nner</sub> <sub>sum</sub> b<sub>y</sub> $2 d \log ( 1 + 4 L ^ { 2 } t / ( \lambda d ) )$ , <sub>g</sub>ivin<sub>g</sub> (30) b<sub>y</sub> the definition (15) of $G$

For (31), a<sub>pp</sub>l<sub>y</sub> the same com<sub>p</sub>utation to the len<sub>g</sub>th-τ vector sequence $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { \tau - 1 } , \tilde { \mathbf { z } } _ { \tau }$ <sub>w</sub>ith $\tilde { \mathbf { z } } _ { \tau } : =$ $\tilde { \mathbf { x } } _ { \tau } - \tilde { \mathbf { y } } _ { \tau }$ <sub>.</sub> It<sub>s assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>ma</sub>t<sub>r</sub>i<sub>ces are</sub> $V _ { 1 } , \ldots , V _ { \tau }$ <sub>, w</sub>hi<sub>c</sub>h <sub>are exac</sub>tl<sub>y</sub> th<sub>ose</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>orme</sub>d<sub>, s</sub>i<sub>nce</sub> th<sub>e</sub> fi<sub>rs</sub>t $\tau - 1$ vectors are the displayed ones; the τ-th width in this sequence is $\lVert \tilde { \mathbf { z } } _ { \tau } \rVert _ { V _ { \tau } ^ { - 1 } } = w _ { \tau } .$ <sub>an</sub>d $\| \tilde { \mathbf { z } } _ { \tau } \| _ { 2 } \leq 2 L$ <sub>.</sub> L<sub>emma</sub> A<sub>.</sub>13 b<sub>oun</sub>d<sub>s</sub> th<sub>e</sub> fi<sub>rs</sub>t $\tau - 1$ <sub>cer</sub>tifi<sub>ca</sub>t<sub>es an</sub>d<sub>,</sub> b<sub>y</sub> it<sub>s</sub> fi<sub>na</sub>l <sub>c</sub>l<sub>a</sub>i<sub>m, a</sub>l<sub>so</sub> $c _ { \tau } ;$ L<sub>emma</sub> $_ { \mathrm { A . 1 2 } }$ <sub>w</sub>ith $m = \tau$ <sub>nee</sub>d<sub>s on</sub>l<sub>y</sub> $V _ { 1 } , \ldots , V _ { \tau }$ t<sub>oge</sub>th<sub>er w</sub>ith $\| \tilde { \mathbf { z } } _ { \tau } \| _ { 2 } \leq 2 L$ <sub>.</sub> H<sub>e</sub>n<sub>ce</sub> th<sub>e</sub> <sub>a</sub>r<sub>gu</sub>m<sub>e</sub>nt <sub>app</sub>li<sub>es w</sub>ith<sub>ou</sub>t $\tilde { \mathbf { z } } _ { \tau }$ <sub>ever</sub> b<sub>e</sub>i<sub>ng</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>or a</sub>dd<sub>e</sub>d t<sub>o</sub> $V$

## A.6. Safety, fallback count, and regret

L<sub>e</sub>t $\boldsymbol { A } _ { t } , \boldsymbol { B } _ { t }$ be the candidate and fallback rounds up to t, with $M _ { t } = | { \cal A } _ { t } |$ <sub>an</sub>d $N _ { t } = | B _ { t } |$ , so $M _ { t } + N _ { t } = t$ <sub>.</sub> It<sub>era</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 1<sub>,</sub>

$$
\Lambda _ { t } = \sum _ { i \in \mathcal { A } _ { t } } \left[ \frac { 1 } { 2 } - \frac { c _ { i } } { 4 } - ( 1 - \alpha ) s _ { i } ( \mathbf { x } _ { b _ { i } } ) \right] + \sum _ { i \in \mathcal { B } _ { t } } \alpha \ : s _ { i } ( \mathbf { x } _ { b _ { i } } ) ,\tag{32}
$$

<sub>an</sub>d <sub>we wr</sub>it<sub>e</sub> th<sub>e</sub> t<sub>rue cumu</sub>l<sub>a</sub>ti<sub>ve surp</sub>l<sub>us as</sub>

$$
\Sigma _ { t } = \sum _ { i = 1 } ^ { t } \left[ \operatorname* { m a x } \{ s _ { i } ( \mathbf { x } _ { i } ) , s _ { i } ( \mathbf { y } _ { i } ) \} - ( 1 - \alpha ) s _ { i } ( \mathbf { x } _ { b _ { i } } ) \right] .\tag{33}
$$

Lemma A.15 (Budget invariant) On E, $\Sigma _ { t } \geq \Lambda _ { t } \geq 0$ for every $t \geq 0$

Proof Induction on t; both quantities vanish at $t = 0$ . <sup>S</sup>upp<sup>ose</sup> $\Sigma _ { t - 1 } \geq \Lambda _ { t - 1 } \geq 0$

C<sub>an</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>.</sub> On $\mathcal { E } ,$ <sub>,</sub> L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>2 <sub>g</sub>i<sub>ves</sub> $R _ { t } \ \leq \ c _ { t }$ f<sub>or</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r, so</sub> L<sub>emma</sub> 3<sub>.</sub>3 <sub>w</sub>ith $c = c _ { t }$ yields max $\begin{array} { r } { \left\{ s _ { t } ( \mathbf { x } _ { t } ) , s _ { t } ( \mathbf { y } _ { t } ) \right\} \geq \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } } \end{array}$ . Hence the true one-round increment of (33) is at least $\begin{array} { r } { \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) s _ { t } ( \mathbf x _ { b _ { t } } ) } \end{array}$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>ncremen</sub>t<sub>,</sub> <sub>an</sub>d $\Sigma _ { t } \geq \Lambda _ { t }$ <sub>.</sub> N<sub>onnega</sub>ti<sub>v</sub>it<sub>y</sub> <sub>o</sub>f $\Lambda _ { t }$ i<sub>s prec</sub>i<sub>se</sub>l<sub>y</sub> th<sub>e accep</sub>t<sub>ance</sub> t<sub>es</sub>t<sub>.</sub>

F<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>.</sub> Th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r</sub> i<sub>s</sub> $( \mathbf { x } _ { b _ { t } } , \mathbf { y } _ { t } ^ { \mathrm { f b } } )$ , so max $\{ s _ { t } ( \mathbf { x } _ { t } ) , s _ { t } ( \mathbf { y } _ { t } ) \} \geq s _ { t } ( \mathbf { x } _ { b _ { t } } )$ <sub>an</sub>d th<sub>e</sub> t<sub>rue</sub> i<sub>ncremen</sub>t i<sub>s a</sub>t l<sub>eas</sub>t $s _ { t } ( \mathbf { x } _ { b _ { t } } ) - ( 1 - \alpha ) s _ { t } ( \mathbf { x } _ { b _ { t } } ) = \alpha s _ { t } ( \mathbf { x } _ { b _ { t } } )$ <sub>,</sub> <sub>aga</sub>i<sub>n</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>ncremen</sub>t<sub>;</sub> <sub>so</sub> $\Sigma _ { t } \geq \Lambda _ { t }$ Si<sub>nce</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } ) > 0$ <sub>an</sub>d $\Lambda _ { t - 1 } \geq 0$ , w<sup>e</sup> g<sup>et</sup> $\Lambda _ { t } \geq 0$

Lemma A.16 (Fallback rounds are early) On E, every fallback round τ satisfies 4αs $\tau < G ( \tau )$ <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub> $N _ { T } \leq \operatorname* { m i n } \{ T , N _ { \mathrm { c o n s } } \}$ for every $T \geq 1$ <sub>, w</sub>ith $N _ { \mathrm { c o n s } }$ as in (15).

Proof Let τ be a fallback round, so the acceptance test failed:

$$
\begin{array} { r } { \Lambda _ { \tau - 1 } + \frac { 1 } { 2 } - \frac { c _ { \tau } } { 4 } - ( 1 - \alpha ) s _ { \tau } ( \mathbf x _ { b _ { \tau } } ) < 0 . } \end{array}
$$

E<sub>xpa</sub>ndin<sub>g</sub> $\Lambda _ { \tau - 1 }$ b<sub>y</sub> (32) and rearran<sub>g</sub>in<sub>g</sub>,

$$
\sum _ { i \in A _ { \tau - 1 } } \left[ \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { i } ( \mathbf { x } _ { b _ { i } } ) \right] + \sum _ { i \in B _ { \tau - 1 } } \alpha s _ { i } ( \mathbf { x } _ { b _ { i } } ) + \Bigl [ \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { \tau } ( \mathbf { x } _ { b _ { \tau } } ) \Bigr ] \ < \ \frac { 1 } { 4 } \Bigl ( \sum _ { i \in A _ { \tau - 1 } } c _ { i } + c _ { \tau } \Bigr ) .
$$

B<sub>y</sub> L<sub>emma</sub> A<sub>.</sub>1 th<sub>e</sub> fi<sub>rs</sub>t b<sub>rac</sub>k<sub>e</sub>t i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $\frac { \alpha } { 2 }$ f<sub>or</sub> <sub>eac</sub>h $i ,$ the second summand is at least αs for each i, and the third bracket is at least $\frac { \alpha } { 2 }$ <sub>.</sub> O<sub>n</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht<sub>,</sub> <sub>cer</sub>tifi<sub>ca</sub>t<sub>es</sub> <sub>are</sub> <sub>nonnega</sub>ti<sub>ve,</sub> <sub>so</sub> $\begin{array} { r } { \sum _ { i \in { \mathcal A } _ { \tau - 1 } } c _ { i } + c _ { \tau } \le \sum _ { i = 1 } ^ { \tau - 1 } c _ { i } + c _ { \tau } \le G ( \tau ) } \end{array}$ b<sub>y</sub> (31). Therefore

$$
\begin{array} { r } { \frac { \alpha } { 2 } M _ { \tau - 1 } + \alpha \underline { { s } } \ : N _ { \tau - 1 } + \frac { \alpha } { 2 } \ < \ \frac { 1 } { 4 } G ( \tau ) . } \end{array}
$$

Si<sub>nce</sub> $\begin{array} { r } { \underline { { s } } \le \frac { 1 } { 2 } } \end{array}$ <sub>we</sub> h<sub>ave</sub> ${ \frac { \alpha } { 2 } } \geq \alpha \underline { { s } }$ <sub>, so</sub> th<sub>e</sub> l<sub>e</sub>ft <sub>s</sub>id<sub>e</sub> i<sub>s a</sub>t l<sub>eas</sub>t $\alpha \underline { { s } } \left( M _ { \tau - 1 } + N _ { \tau - 1 } + 1 \right) = \alpha \underline { { s } } \tau$ <sub>.</sub> H<sub>ence</sub> 4αs $\tau < G ( \tau )$

E<sub>very</sub> f<sub>a</sub>llb<sub>ac</sub>k ti<sub>me</sub> th<sub>ere</sub>f<sub>ore</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $\{ t \geq 1 : 4 \alpha \underline { { s } } t < G ( t ) \}$ <sub>an</sub>d <sub>so</sub> i<sub>s</sub> <sub>a</sub>t <sub>mos</sub>t th<sub>e</sub> <sub>supremum</sub> $N _ { \mathrm { c o n s } }$ <sub>o</sub>f th<sub>a</sub>t <sub>se</sub>t<sub>.</sub> F<sub>a</sub>llb<sub>ac</sub>k ti<sub>mes</sub> <sub>are</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>pos</sub>iti<sub>ve</sub> i<sub>n</sub>t<sub>egers</sub> i<sub>n</sub> $\{ 1 , \ldots , N _ { \mathrm { c o n s } } \}$ , <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> $N _ { T } \leq N _ { \mathrm { c o n s } } ;$ <sub>an</sub>d $N _ { T } \leq T$ t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>ll<sub>y.</sub>

Lemma A.17 (Finiteness of $N _ { \mathrm { c o n s } } ) G ( t ) = o ( t )$ , an<sup>d</sup> conse<sub>q</sub>uent<sup>l</sup><sub>y</sub> $N _ { \mathrm { c o n s } } < \infty$

Proof From $( 6 ) , \beta _ { t } = O \big ( \sqrt { d \log t + \log ( 1 / \delta ) } / \kappa + \sqrt { \lambda } S / \kappa \big ) , \mathrm { s o } \bar { \beta } _ { t } = O ( \sqrt { d \log ( e t ) } )$ f<sub>or</sub> fi<sub>xe</sub>d $d , L , S , \lambda , \delta , \kappa .$ <sub>,</sub> <sub>an</sub>d h<sub>ence</sub>

$$
G ( t ) = 4 \bar { \beta } _ { t } \sqrt { 2 d t \log \Bigl ( 1 + \frac { 4 L ^ { 2 } t } { \lambda d } \Bigr ) } = O \bigl ( \sqrt { t } \log ( e t ) \bigr ) = o ( t ) .
$$

Th<sub>us</sub> th<sub>ere</sub> i<sub>s</sub> $t _ { 0 }$ <sub>w</sub>ith $G ( t ) <$ 4αs t for all $t \geq t _ { 0 }$ , so the set in (15) is contained in $\{ 1 , \ldots , t _ { 0 } - 1 \}$ <sub>an</sub>d it<sub>s supremum</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e.</sub> N<sub>o</sub>t<sub>e</sub> $\underline { { s } } \geq \sigma ( - B ) > 0$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 2 <sub>an</sub>d $\Delta _ { h } \leq B$ <sub>, so</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y pos</sub>iti<sub>ve on every</sub> i<sub>ns</sub>t<sub>ance.</sub>

Theorem 3.1 (restated). With probability at least $1 - \delta ,$ , simultaneously for every $T \geq 1$ <sub>:</sub> th<sub>e</sub> dis<sub>p</sub>la<sub>y</sub>ed <sub>p</sub>airs are α-conservative in the sense of (3); $N _ { T } \leq \operatorname* { m i n } \{ T , N _ { \mathrm { c o n s } } \}$ <sub>;</sub> <sub>an</sub>d $\mathsf { R e g } ( T ) \leq$ $\begin{array} { r } { G ( T ) + \frac { 1 } { 2 } \Delta _ { h } N _ { \mathrm { c o n s } } } \end{array}$

Proof By Lemma $3 . 1 , \mathbb { P } ( \mathcal { E } ) \geq 1 - \delta ;$ <sub>a</sub>ll th<sub>ree c</sub>l<sub>a</sub>i<sub>ms are es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>on</sub> $\mathcal { E } ,$ <sub>,</sub> h<sub>ence</sub> h<sub>o</sub>ld <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> <sub>on</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>even</sub>t<sub>.</sub>

Safety. By Lemma A. $1 5 , \Sigma _ { T } \Sigma 0$ <sup>f</sup>or ever<sub>y</sub> $T _ { \mathbf { \delta } }$ , and ex<sub>p</sub>andin<sub>g</sub> (33) this is exactl<sub>y</sub> $\begin{array} { r } { \sum _ { t \leq T } \operatorname* { m a x } \{ s _ { t } ( \mathbf x _ { t } ) , s _ { t } ( \mathbf y _ { t } ) \} \geq } \end{array}$ $\begin{array} { r } { \big ( 1 - \alpha \big ) \sum _ { t < T } s _ { t } \big ( \mathbf { x } _ { b _ { t } } \big ) } \end{array}$ , i.e. (3). Since the invariant holds at ever<sub>y</sub> t, the <sub>g</sub>uarantee is an<sub>y</sub>time.

F<sub>a</sub>llb<sub>ac</sub>k <sub>coun</sub>t<sub>.</sub> L<sub>emma</sub> A<sub>.</sub>16<sub>, w</sub>ith $N _ { \mathrm { c o n s } } < \infty$ b<sub>y</sub> L<sub>emma</sub> A<sub>.</sub>17<sub>.</sub>

R<sub>egre</sub>t<sub>.</sub> S<sub>p</sub>lit b<sub>y roun</sub>d t<sub>ype.</sub> O<sub>n a can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>,</sub> $R _ { t } \leq c _ { t }$ b<sub>y</sub> L<sub>emma</sub> 3<sub>.</sub>2<sub>.</sub> O<sub>n a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r</sub> i<sub>s</sub> $( \mathbf { x } _ { b _ { t } } , \mathbf { y } _ { t } ^ { \mathrm { f b } } )$ <sub>w</sub>ith ${ \bf y } _ { t } ^ { \mathrm { f b } } \in \mathcal { C } _ { t } .$ <sub>,</sub> <sub>so</sub> L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>2 <sub>g</sub>i<sub>ves</sub> $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } _ { t } ^ { \mathrm { f b } } ) \leq c _ { t }$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> $f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } _ { b _ { t } } ) = \Delta _ { t , b _ { t } } \leq \Delta _ { h }$ b<sub>y</sub> A<sub>ssu</sub>m<sub>p</sub>ti<sub>o</sub>n 3<sub>;</sub> <sub>ave</sub>r<sub>ag</sub>in<sub>g,</sub>

$$
\begin{array} { r } { { R } _ { t } = \frac 1 2 \Big [ \big ( f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } _ { b _ { t } } ) \big ) + \big ( f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } _ { t } ^ { \mathrm { f b } } ) \big ) \Big ] \ \leq \ \frac 1 2 \big ( \Delta _ { h } + c _ { t } \big ) . } \end{array}
$$

S<sub>umm</sub>i<sub>ng an</sub>d <sub>us</sub>i<sub>ng</sub> $c _ { t } \geq 0$

$$
\mathtt { R e g } ( T ) \ \le \ \sum _ { t \in \mathcal { A } _ { T } } c _ { t } + \sum _ { t \in \mathcal { B } _ { T } } \frac { c _ { t } } { 2 } + \frac { \Delta _ { h } } { 2 } N _ { T } \ \le \ \sum _ { t = 1 } ^ { T } c _ { t } + \frac { \Delta _ { h } } { 2 } N _ { T } \ \le \ G ( T ) + \frac { \Delta _ { h } } { 2 } N _ { \mathrm { c o n s } } ,
$$

b<sub>y</sub> (30) and Lemma A.16.

## A.7. The simplified rate

Lemma A.18 Let $H ( t ) : = G ( t ) / \sqrt { t } = 4 \bar { \beta } _ { t } \sqrt { 2 d \log \left( 1 + 4 L ^ { 2 } t / ( \lambda d ) \right) }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng.</sub> Th<sub>en</sub>

$$
N _ { \mathrm { c o n s } } \ \leq \ { \frac { H ( N _ { \mathrm { c o n s } } ) ^ { 2 } } { 1 6 { \alpha ^ { 2 } } { \underline { { s } } } ^ { 2 } } } \ = \ { \frac { 2 d { \bar { \beta } } _ { N _ { \mathrm { c o n s } } } ^ { 2 } \log \left( 1 + 4 L ^ { 2 } N _ { \mathrm { c o n s } } / ( \lambda d ) \right) } { { \alpha ^ { 2 } } { \underline { { s } } } ^ { 2 } } } ,\tag{34}
$$

and, suppressing logarithmic factors in $T , d , L , S , \lambda , 1 / \delta$

$$
N _ { \mathrm { c o n s } } = \tilde { \mathcal { O } } \Big ( \frac { d ^ { 2 } } { \kappa ^ { 2 } \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \Big ) , \qquad G ( T ) = \tilde { \mathcal { O } } \Big ( \frac { d \sqrt { T } } { \kappa } \Big ) , \qquad \mathtt { R e g } ( T ) = \tilde { \mathcal { O } } \Big ( \frac { d \sqrt { T } } { \kappa } + \frac { \Delta _ { h } d ^ { 2 } } { \kappa ^ { 2 } \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \Big ) .
$$

Proof If $N _ { \mathrm { c o n s } } = 0$ th<sub>ere</sub> i<sub>s no</sub>thi<sub>ng</sub> t<sub>o prove.</sub> Oth<sub>erw</sub>i<sub>se</sub> $N _ { \mathrm { c o n s } }$ belon<sub>g</sub>s to the set in (15), so 4αs $N _ { \mathrm { c o n s } } < G ( N _ { \mathrm { c o n s } } ) = H ( N _ { \mathrm { c o n s } } ) \sqrt { N _ { \mathrm { c o n s } } } .$ <sub>,</sub> <sub>w</sub>h<sub>ence</sub> $\sqrt { N _ { \mathrm { c o n s } } } < H ( N _ { \mathrm { c o n s } } ) / ( 4 \alpha \underline { { s } } )$ and (34) follows by squaring and substituting the definition of H.

For the orders, (6) <sub>g</sub>ives $\bar { \beta } _ { t } = \tilde { \mathcal { O } } ( \sqrt { d } / \kappa )$ , so $H ( t ) = \tilde { \mathcal { O } } ( d / \kappa )$ <sub>an</sub>d $G ( T ) = H ( T ) \sqrt { T } = \tilde { \mathcal { O } } ( d \sqrt { T } / \kappa )$ S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $H ( N _ { \mathrm { c o n s } } ) = \tilde { \mathcal { O } } ( d / \kappa )$ into (34) <sub>g</sub>ives $N _ { \mathrm { c o n s } } = \tilde { \mathcal { O } } \big ( d ^ { 2 } / ( \kappa ^ { 2 } \alpha ^ { 2 } \underline { { s } } ^ { 2 } ) \big )$ <sub>, w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> f<sub>ac</sub>t<sub>ors</sub> i<sub>nvo</sub>l<sub>v</sub>i<sub>ng</sub> $N _ { \mathrm { c o n s } }$ <sub>are</sub> th<sub>emse</sub>l<sub>ves</sub> $\mathcal { \tilde { O } } \big ( \log ( d / ( \kappa \alpha \underline { { s } } ) ) \big )$ b<sub>y</sub> b<sub>ac</sub>k<sub>-su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on.</sub> C<sub>om</sub>bi<sub>n</sub>i<sub>ng w</sub>ith Th<sub>eorem</sub> 3<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> th<sub>e regre</sub>t b<sub>oun</sub>d<sub>.</sub> Th<sub>e</sub> b<sub>oun</sub>d i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>1 i<sub>s</sub> th<sub>e exac</sub>t <sub>nonasymp</sub>t<sub>o</sub>ti<sub>c s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>;</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>ay</sub> <sub>a</sub>b<sub>ove</sub> i<sub>s</sub> <sub>an</sub> <sub>or</sub>d<sub>er</sub> <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>on</sub>l<sub>y.</sub>

## A.8. Equivalence of latent and preference regret

Proposition A.1 Define the preference regret of a displayed pair as $\begin{array} { r } { R _ { t } ^ { \mathrm { B T } } : = s _ { t } ( \mathbf { x } _ { t } ^ { \star } ) - \frac { 1 } { 2 } \big ( s _ { t } ( \mathbf { x } _ { t } ) + } \end{array}$ $s _ { t } ( \mathbf { y } _ { t } ) )$ . Thenfor every pair,

$$
\begin{array} { r } { \kappa R t \leq R _ { t } ^ { \mathrm { B T } } \leq \frac { 1 } { 4 } R _ { t } . } \end{array}
$$

Proof With $g _ { \mathbf x } = f ( \mathbf x _ { t } ^ { \star } ) - f ( \mathbf x _ { t } )$ <sub>an</sub>d $g _ { \mathbf y } = f ( \mathbf x _ { t } ^ { \star } ) - f ( \mathbf y _ { t } )$ <sub>,</sub> b<sub>o</sub>th i<sub>n</sub> $[ 0 , B ]$ b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 2<sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\begin{array} { r } { R _ { t } = \frac { 1 } { 2 } ( g _ { \mathbf { x } } + g _ { \mathbf { y } } ) } \end{array}$ <sub>an</sub>d<sub>,</sub> <sub>us</sub>i<sub>ng</sub> $\begin{array} { r } { s _ { t } ( \mathbf { x } _ { t } ^ { \star } ) = \sigma ( 0 ) = \frac { 1 } { 2 } } \end{array}$

$$
\begin{array} { r } { R _ { t } ^ { \mathrm { B T } } = \frac { 1 } { 2 } \Big [ \big ( \frac { 1 } { 2 } - \sigma ( - g _ { \mathbf { x } } ) \big ) + \big ( \frac { 1 } { 2 } - \sigma ( - g _ { \mathbf { y } } ) \big ) \Big ] . } \end{array}
$$

F<sub>or</sub> $\begin{array} { r } { g \in [ 0 , B ] , \frac { 1 } { 2 } - \sigma ( - g ) = \int _ { - q } ^ { 0 } \sigma ^ { \prime } ( u ) \mathrm { d } u } \end{array}$ <sub>,</sub> <sub>an</sub>d $\sigma ^ { \prime } \mathrm { ~ o n ~ } \left[ - B , 0 \right]$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\kappa \leq \sigma ^ { \prime } \leq \frac { 1 } { 4 }$ <sub>;</sub> h<sub>ence</sub> $\begin{array} { r } { \kappa g \le \frac 1 2 - \sigma ( - g ) \le \frac { g } { 4 } } \end{array}$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub> thi<sub>s</sub> t<sub>o</sub> $g _ { \mathbf { x } }$ <sub>an</sub>d $g _ { \mathbf { y } }$ <sub>,</sub> <sub>a</sub>dd<sub>,</sub> <sub>an</sub>d di<sub>v</sub>id<sub>e</sub> b<sub>y</sub> t<sub>wo.</sub> ■

## A.9. Failure of Direct CLUCB Certification under Preference Feedback

We examine a direct transfer of CLUCB’s cumulative safet<sub>y</sub> test (Kazerouni et al., 2017) to <sub>p</sub>reference f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> Th<sub>e</sub> <sub>resu</sub>lt <sub>concerns</sub> th<sub>e</sub> <sub>use</sub> <sub>o</sub>f thi<sub>s</sub> t<sub>es</sub>t <sub>w</sub>ith <sub>con</sub>fid<sub>ence</sub> <sub>se</sub>t<sub>s</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d f<sub>rom</sub> <sub>compar</sub>i<sub>sons.</sub> K<sub>nown</sub> b<sub>ase</sub>li<sub>ne u</sub>tiliti<sub>es perm</sub>it <sub>a</sub>lt<sub>erna</sub>ti<sub>ve,</sub> b<sub>ase</sub>li<sub>ne-cen</sub>t<sub>ere</sub>d <sub>cer</sub>tifi<sub>ca</sub>t<sub>es, w</sub>hi<sub>c</sub>h <sub>we</sub> di<sub>scuss</sub> b<sub>e</sub>l<sub>ow.</sub>

Th<sub>roug</sub>h<sub>ou</sub>t thi<sub>s</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>on,</sub> th<sub>e</sub> l<sub>earner</sub> i<sub>s</sub> <sub>a</sub>dditi<sub>ona</sub>ll<sub>y</sub> <sub>supp</sub>li<sub>e</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>exac</sub>t b<sub>ase</sub>li<sub>ne</sub> <sub>u</sub>tiliti<sub>es</sub>

$$
\mu _ { t } ^ { 0 } : = f ( \mathbf { x } _ { b _ { t } } ) .
$$

Th<sub>ese are</sub> di<sub>s</sub>ti<sub>nc</sub>t f<sub>rom</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne pre</sub>f<sub>erence scores use</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on.</sub> P<sub>rov</sub>idi<sub>ng</sub> thi<sub>s</sub> <sub>a</sub>dditi<sub>ona</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>w</sub>ith CLUCB <sub>exp</sub>li<sub>c</sub>it<sub>.</sub>

Definition A.1 (Direct cumulative CLUCB certification) Let $\boldsymbol { A } _ { t }$ <sub>an</sub>d $B _ { t }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e a</sub>d<sub>m</sub>itt<sub>e</sub>d <sub>can</sub> didate rounds and fallback rounds through time $t ,$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>an</sub>d <sub>wr</sub>it<sub>e</sub>

$$
\begin{array} { r } { M _ { t } : = | { \mathcal { A } } _ { t } | , \qquad N _ { t } : = | { \mathcal { B } } _ { t } | , \qquad { \bf m } _ { i } : = \frac { 1 } { 2 } ( { \bf x } _ { i } + { \bf y } _ { i } ) . } \end{array}
$$

At <sub>roun</sub>d $t ,$ l<sub>e</sub>t $( \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { y } } _ { t } ) \in S _ { t } \times S _ { t }$ b<sub>e</sub> <sub>an</sub> <sub>ar</sub>bit<sub>rary</sub> <sub>propose</sub>d <sub>pa</sub>i<sub>r,</sub> <sub>w</sub>ith

$$
\begin{array} { r } { \widetilde { \bf m } _ { t } : = \frac { 1 } { 2 } ( \widetilde { \bf x } _ { t } + \widetilde { \bf y } _ { t } ) . } \end{array}
$$

Construct the confidence set

$$
\mathcal { K } _ { t } : = \Big \{ \theta \in \Theta : \| \theta - \hat { \theta } _ { t } \| _ { V _ { t } } \leq \beta _ { t } \Big \} ,\tag{35}
$$

<sub>w</sub>h<sub>ere</sub> ${ \widehat { \theta } } _ { t } , V _ { t } ,$ <sub>an</sub>d $\beta _ { t }$ are defined b<sub>y</sub> (5)–(6), usin<sub>g</sub> the observed com<sub>p</sub>arisons. The al<sub>g</sub>orithm admits th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>on</sub>l<sub>y</sub> $i f$

$$
L _ { t } : = \operatorname* { i n f } _ { \theta \in \mathcal { K } _ { t } } \left. \theta , \sum _ { i \in \mathcal { A } _ { t - 1 } } \mathbf { m } _ { i } + \widetilde { \mathbf { m } } _ { t } \right. ,\tag{36}
$$

$$
L _ { t } + \sum _ { i \in \mathcal { B } _ { t - 1 } } \mu _ { i } ^ { 0 } \geq ( 1 - \alpha ) \sum _ { i = 1 } ^ { t } \mu _ { i } ^ { 0 } .\tag{37}
$$

Oth<sub>erw</sub>i<sub>se,</sub> it di<sub>sp</sub>l<sub>ays</sub> <sub>a</sub> <sub>pa</sub>i<sub>r</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> $\mathbf { x } _ { b _ { t } }$ <sub>an</sub>d <sub>cre</sub>dit<sub>s</sub> $\mu _ { t } ^ { 0 }$ for that round in subsequent applications $o f \left( 3 7 \right)$ ). Candidate pairs andfallback partners may be selected by arbitrary nonanticipating rules.

The baseline utilities are credited se<sub>p</sub>aratel<sub>y</sub> in (37); the confidence set (35) is not intersected with th<sub>e</sub> <sub>equa</sub>liti<sub>es</sub> $\langle \boldsymbol { \theta } , \mathbf { x } _ { b _ { i } } \rangle = \mu _ { i } ^ { 0 }$

Thi<sub>s</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on re</sub>t<sub>a</sub>i<sub>ns</sub> CLUCB’<sub>s use o</sub>f th<sub>e curren</sub>t <sub>con</sub>fid<sub>ence se</sub>t t<sub>o cer</sub>tif<sub>y cumu</sub>l<sub>a</sub>ti<sub>ve can</sub>did<sub>a</sub>t<sub>e</sub> <sub>per</sub>f<sub>ormance, w</sub>hil<sub>e rep</sub>l<sub>ac</sub>i<sub>ng sca</sub>l<sub>ar-rewar</sub>d <sub>regress</sub>i<sub>on</sub> b<sub>y</sub> th<sub>e compar</sub>i<sub>son es</sub>ti<sub>ma</sub>t<sub>or.</sub> Th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>se</sub>lf<sub>-</sub>d<sub>ue</sub>l $\left( \mathbf { x } _ { b _ { t } } , \mathbf { x } _ { b _ { t } } \right)$ reproduces the usual baseline fallback and justifies crediting $\mu _ { t } ^ { 0 }$ <sub>.</sub> W<sub>e a</sub>l<sub>so a</sub>ll<sub>ow</sub> <sub>o</sub>th<sub>er</sub> b<sub>ase</sub>li<sub>ne-con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>s: on</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>ance cons</sub>t<sub>ruc</sub>t<sub>e</sub>d b<sub>e</sub>l<sub>ow,</sub> th<sub>e</sub>i<sub>r average u</sub>tilit<sub>y</sub> i<sub>s a</sub>l<sub>ways</sub> <sub>a</sub>t l<sub>eas</sub>t $\mu _ { t } ^ { 0 }$ <sub>, so</sub> th<sub>e same cre</sub>dit i<sub>s va</sub>lid<sub>.</sub> O<sub>n a genera</sub>l i<sub>ns</sub>t<sub>ance, suc</sub>h <sub>a cre</sub>dit f<sub>or a non-se</sub>lf f<sub>a</sub>llb<sub>ac</sub>k would require a separate justification.

Proposition A.2 (Failure of the direct cumulative transfer) Fix $\lambda > 0$ <sub>an</sub>d $\alpha , \delta \in ( 0 , 1 )$ <sub>, an</sub>d <sub>wr</sub>it<sub>e</sub> $p : = 1 - \alpha$ <sub>.</sub> Th<sub>ere</sub> i<sub>s a</sub> t<sub>wo-</sub>di<sub>mens</sub>i<sub>ona</sub>l i<sub>ns</sub>t<sub>ance w</sub>ith $K = 2 ,$ , satisfying the itemwise $i . i . d .$ <sub>con</sub>t<sub>ex</sub>t <sub>assump</sub>ti<sub>on,</sub> A<sub>ssump</sub>ti<sub>ons</sub> 1<sub>,</sub> 2<sub>,</sub> 3<sub>, an</sub>d 4<sub>, an</sub>d th<sub>e norma</sub>li<sub>za</sub>ti<sub>on</sub> $f ( \cdot ) \in [ 0 , 1 ]$ , with thefollowing p<sup>ro</sup>p<sup>ertie</sup>s.

Every possible displayed pair satisfies both the cardinal average-utility constraint and the score constraint (3). Nevertheless, every algorithm satisfying Definition A.1 obeys, on its confidence event $\mathcal { E } ,$

$$
N _ { T } \geq { \frac { p } { 2 } } T \qquad f o r e \nu e r y T \geq 1 .\tag{38}
$$

Moreover, for every $\eta \in ( 0 , 1 )$ and every fixed integer

$$
T \geq { \frac { 3 2 } { p ^ { 2 } } } \log { \frac { 1 } { \eta } } ,
$$

$$
\mathbb { P } \left\{ { \tt R e g } ( T ) \geq \frac { p ^ { 3 } } { 5 1 2 } T \right\} \geq 1 - \delta - \eta .\tag{39}
$$

If all fallbacks are baseline self-duels, the denominator in (39) improves from 512 to 256.

<sup>O</sup>n t<sup>h</sup>e same <sup>i</sup>nstance, MaxInP<sup>9</sup>BC sat<sup>i</sup>sfies, w<sup>i</sup>t<sup>h</sup> pro<sup>b</sup>a<sup>bili</sup>ty at <sup>l</sup>east $1 - \delta ,$ , simultaneouslyfor every $T \geq 1$

$$
N _ { T } \leq N _ { \mathrm { c o n s } } , \qquad \mathtt { R e g } ( T ) \leq G ( T ) + \frac { p } { 8 } N _ { \mathrm { c o n s } } = \tilde { \mathcal { O } } ( \sqrt { T } ) ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e ra</sub>t<sub>e</sub> t<sub>rea</sub>t<sub>s</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>ance parame</sub>t<sub>ers an</sub>d $\alpha , \delta , \lambda$ as fixed.

Proof The instance. Set

$$
g : = \frac { p } { 4 } , \qquad \theta ^ { \star } : = \binom { 1 / 2 } { g } , \qquad \mathcal { D } : = \Big \{ ( 1 , u ) ^ { \top } : u \in [ 0 , 1 ] \Big \} .
$$

I<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y over roun</sub>d<sub>s an</sub>d it<sub>ems,</sub> d<sub>raw</sub>

$$
U _ { t , 1 } , U _ { t , 2 } \sim \operatorname { U n i f } [ 0 , 1 ] , \qquad \mathbf { x } _ { t , i } : = ( 1 , U _ { t , i } ) ^ { \top } .
$$

Th<sub>us</sub> <sub>eac</sub>h it<sub>em</sub> i<sub>s</sub> <sub>samp</sub>l<sub>e</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> <sub>same</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>on</sub> ${ \mathcal { D } } ,$ <sub>an</sub>d th<sub>e</sub> t<sub>wo</sub> it<sub>ems</sub> <sub>are</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y.</sub>

L<sub>e</sub>t th<sub>e</sub> b<sub>ase</sub>li<sub>ne se</sub>l<sub>ec</sub>t th<sub>e</sub> it<sub>em w</sub>ith th<sub>e sma</sub>ll<sub>er secon</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e,</sub> b<sub>rea</sub>ki<sub>ng</sub> ti<sub>es ar</sub>bit<sub>rar</sub>il<sub>y.</sub> Si<sub>nce</sub> $g > 0$ <sub>,</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l it<sub>em</sub> h<sub>as</sub> th<sub>e</sub> l<sub>arger</sub> <sub>secon</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e.</sub> C<sub>onsequen</sub>tl<sub>y,</sub>

$$
\begin{array} { r l } & { f ( \mathbf { x } _ { t , i } ) = \displaystyle \frac { 1 } { 2 } + g U _ { t , i } , } \\ & { ~ \displaystyle \mu _ { t } ^ { 0 } = \frac { 1 } { 2 } + g \operatorname* { m i n } \{ U _ { t , 1 } , U _ { t , 2 } \} , } \\ & { ~ \Delta _ { t , b _ { t } } = g Q _ { t } , ~ Q _ { t } : = | U _ { t , 1 } - U _ { t , 2 } | . } \end{array}
$$

I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub>

$$
\frac { 1 } { 2 } \leq \mu _ { t } ^ { 0 } \leq \frac { 1 } { 2 } + g , \qquad 0 \leq \Delta _ { t , b _ { t } } \leq g , \qquad \frac { 1 } { 2 } \leq f ( { \bf x } ) \leq \frac { 3 } { 4 } \quad \mathrm { f o r e v e r y ~ } { \bf x } \in \mathcal { D } .
$$

Th<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>ness an</sub>d li<sub>near</sub>it<sub>y assump</sub>ti<sub>ons</sub> h<sub>o</sub>ld <sub>w</sub>ith $S = 1 , L = { \sqrt { 2 } }$ <sub>,</sub> <sub>an</sub>d $B = 2 L S = 2 \sqrt { 2 }$ <sub>.</sub> Th<sub>e</sub> b<sub>ase</sub>li<sub>ne-gap</sub> <sub>assump</sub>ti<sub>on</sub> h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>ith $\Delta _ { l } = 0$ <sub>an</sub>d $\Delta _ { h } = g$ <sub>.</sub> G<sub>enera</sub>t<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>ou</sub>t<sub>comes</sub> <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> BTL <sub>mo</sub>d<sub>e</sub>l <sub>w</sub>ith <sub>parame</sub>t<sub>er</sub> $\theta ^ { \star }$ <sub>.</sub> Th<sub>e</sub> <sub>exac</sub>t b<sub>ase</sub>li<sub>ne</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>s</sub> <sub>supp</sub>li<sub>e</sub>d <sub>as</sub> <sub>s</sub>ti<sub>pu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>a</sub>b<sub>ove;</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne score</sub> i<sub>s</sub>

$$
s _ { t } ( \mathbf x _ { b _ { t } } ) = \sigma ( - g Q _ { t } ) .
$$

B<sub>ecause</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>s</sub> <sub>a</sub> l<sub>eas</sub>t<sub>-u</sub>tilit<sub>y</sub> it<sub>em</sub> i<sub>n</sub> $S _ { t } ,$ , <sup>e</sup>v<sup>er</sup>y p<sup>air</sup> $( \mathbf { x } , \mathbf { y } ) \in S _ { t } \times S _ { t }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\frac { f ( \mathbf { x } ) + f ( \mathbf { y } ) } { 2 } \geq \mu _ { t } ^ { 0 } ,
$$

$$
\operatorname* { m a x } \{ s _ { t } ( \mathbf { x } ) , s _ { t } ( \mathbf { y } ) \} \geq s _ { t } ( \mathbf { x } _ { b _ { t } } ) .\tag{40}
$$

Summing these inequalities proves the asserted safety properties at every prefix. It also justifies the f<sub>a</sub>llb<sub>ac</sub>k <sub>cre</sub>dit f<sub>or</sub> <sub>every</sub> <sub>poss</sub>ibl<sub>e</sub> b<sub>ase</sub>li<sub>ne-con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>pa</sub>i<sub>r</sub> <sub>on</sub> thi<sub>s</sub> i<sub>ns</sub>t<sub>ance.</sub>

The confidence set retains a lower utility ofset. Every comparison vector has zero first coordinate. H<sub>ence,</sub> <sub>regar</sub>dl<sub>ess</sub> <sub>o</sub>f th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub>’<sub>s</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces,</sub>

$$
V _ { t } = \left( \begin{array} { c c } { { \lambda } } & { { 0 } } \\ { { 0 } } & { { \lambda + W _ { t } } } \end{array} \right) , \qquad W _ { t } : = \sum _ { i < t } ( { \bf z } _ { i } ) _ { 2 } ^ { 2 } \ge 0 .
$$

E<sub>very poss</sub>ibl<sub>e pa</sub>i<sub>r average</sub> i<sub>s</sub> $\mathbf { m } = ( 1 , v ) ^ { \top }$ <sub>w</sub>ith $v \in [ 0 , 1 ]$ , so

$$
\| \mathbf { m } \| _ { V _ { t } ^ { - 1 } } ^ { 2 } = \frac { 1 } { \lambda } + \frac { v ^ { 2 } } { \lambda + W _ { t } } \geq \frac { 1 } { \lambda } .\tag{41}
$$

The likelihood term in (5) does not de<sub>p</sub>end on the first coordinate of θ. Settin<sub>g</sub> that coordinate to zero preserves feasibilit<sub>y</sub> in the Euclidean ball Θ and reduces the regularization penalt<sub>y</sub> unless the <sub>coor</sub>di<sub>na</sub>t<sub>e was a</sub>l<sub>rea</sub>d<sub>y zero.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
{ \hat { \theta } } _ { t } = ( 0 , { \hat { g } } _ { t } ) ^ { \top } \qquad { \mathrm { f o r } } { \mathrm { e v e r y } } t .
$$

C<sub>ons</sub>id<sub>er</sub> th<sub>e a</sub>lt<sub>erna</sub>ti<sub>ve parame</sub>t<sub>er</sub>

$$
\theta ^ { \circ } : = ( 0 , g ) ^ { \top } .
$$

O<sub>n</sub> th<sub>e even</sub>t $\mathcal { E }$ <sub>o</sub>f L<sub>emma</sub> 3<sub>.</sub>1<sub>,</sub>

$$
\begin{array} { r l } & { \| \theta ^ { \circ } - \hat { \theta } _ { t } \| _ { V _ { t } } ^ { 2 } = ( \lambda + W _ { t } ) ( g - \hat { g } _ { t } ) ^ { 2 } } \\ & { \qquad \leq \displaystyle \frac { \lambda } { 4 } + ( \lambda + W _ { t } ) ( g - \hat { g } _ { t } ) ^ { 2 } } \\ & { \qquad = \| \theta ^ { \star } - \hat { \theta } _ { t } \| _ { V _ { t } } ^ { 2 } \leq \beta _ { t } ^ { 2 } . } \end{array}
$$

Si<sub>nce</sub> $\| \theta ^ { \circ } \| _ { 2 } = g < 1$ <sub>,</sub> thi<sub>s</sub> <sub>proves</sub> $\theta ^ { \circ } \in \kappa _ { t }$ for every t on $\mathcal { E } .$ M<sub>oreover,</sub>

$$
0 \leq \left. { \theta } ^ { \circ } , { \bf x } \right. \leq g \qquad \mathrm { f o r e v e r y } { \bf x } \in \mathcal { D } .
$$

Th<sub>us</sub> th<sub>e</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on</sub> <sub>a</sub>l<sub>so</sub> <sub>surv</sub>i<sub>ves</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>ng</sub> $\kappa _ { t }$ <sub>w</sub>ith th<sub>e</sub> <sub>u</sub>tilit<sub>y-norma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> $0 \leq$ $\langle \theta , \mathbf { x } \rangle \leq 1$ f<sub>or a</sub>ll $\mathbf { x } \in \mathcal { D }$

F<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>pa</sub>ir <sub>ave</sub>r<sub>age</sub> $\mathbf { m } = ( 1 , v ) ^ { \top }$

$$
\left. \theta ^ { \circ } , \mathbf { m } \right. = g v \leq g .
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, on $\mathcal { E } ,$

$$
L _ { t } \le \left. \theta ^ { \circ } , \sum _ { i \in \mathcal { A } _ { t - 1 } } \mathbf { m } _ { i } + \widetilde \mathbf { m } _ { t } \right. \le g ( M _ { t - 1 } + 1 ) .\tag{42}
$$

Thi<sub>s</sub> b<sub>oun</sub>d h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>ar</sub>bit<sub>rary</sub> <sub>propose</sub>d <sub>pa</sub>i<sub>rs,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>se</sub>lf<sub>-</sub>d<sub>ue</sub>l<sub>s.</sub>

A constantfraction ofrounds must befallbacks. At an admitted round $t ,$ <sub>we</sub> h<sub>ave</sub> $M _ { t } = M _ { t - 1 } + 1$ <sub>an</sub>d $N _ { t } = N _ { t - 1 }$ . Combinin<sub>g</sub> (37), (42), and the bounds on $\mu _ { i } ^ { 0 }$ <sub>g</sub><sup>i</sup>ves

$$
g M _ { t } + \left( { \frac { 1 } { 2 } } + g \right) N _ { t } \geq { \frac { p } { 2 } } t .\tag{43}
$$

Thi<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>a</sub>l<sub>so</sub> <sub>pers</sub>i<sub>s</sub>t<sub>s</sub> th<sub>roug</sub>h <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>:</sub> it<sub>s</sub> l<sub>e</sub>ft <sub>s</sub>id<sub>e</sub> i<sub>ncreases</sub> b<sub>y</sub> ${ \frac { 1 } { 2 } } + g$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> it<sub>s</sub> <sub>r</sub>i<sub>g</sub>ht <sub>s</sub>id<sub>e</sub> i<sub>ncreases</sub> b<sub>y</sub> $p / 2 .$ <sub>, an</sub>d

$$
{ \frac { 1 } { 2 } } + g - { \frac { p } { 2 } } = { \frac { \alpha } { 2 } } + g > 0 .
$$

Since both sides are zero at time zero, (43) holds for ever<sub>y</sub> t on $\mathcal { E }$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $M _ { t } = t - N _ { t }$ <sub>an</sub>d $g = p / 4$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\frac { p } { 4 } t + \frac { 1 } { 2 } N _ { t } \geq \frac { p } { 2 } t ,
$$

which <sub>p</sub>roves (38). Lemma 3.1 <sub>g</sub>ives $\mathbb { P } ( \mathcal { E } ) \geq 1 - \delta .$

A linear number of fallbacks implies linear regret here. The baseline gaps are not uniformly bounded <sub>away</sub> f<sub>rom</sub> <sub>zero,</sub> <sub>so</sub> <sub>we</sub> <sub>mus</sub>t <sub>accoun</sub>t f<sub>or</sub> th<sub>e</sub> <sub>poss</sub>ibilit<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>sma</sub>ll<sub>-gap</sub> <sub>roun</sub>d<sub>s</sub> f<sub>or</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>.</sub>

S<sub>e</sub>t $\varepsilon : = p / 8$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
H _ { T } : = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ Q _ { t } < \varepsilon \} .
$$

Th<sub>e ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>es</sub> $Q _ { t }$ <sub>are</sub> i<sub>.</sub>i<sub>.</sub>d<sub>., w</sub>ith

$$
\mathbb { P } \{ Q _ { t } < \varepsilon \} = 2 \varepsilon - \varepsilon ^ { 2 } \leq 2 \varepsilon = \frac { p } { 4 } .
$$

H<sub>oe</sub>fdi<sub>ng</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>g</sub>i<sub>ves,</sub> f<sub>or</sub> <sub>eac</sub>h fi<sub>xe</sub>d $T _ { \cdot }$

$$
\mathbb { P } \left\{ H _ { T } \leq \frac { p } { 4 } T + \sqrt { \frac { T } { 2 } \log \frac { 1 } { \eta } } \right\} \geq 1 - \eta .
$$

$\mathrm { I f } \ T \geq 3 2 p ^ { - 2 } \log ( 1 / \eta )$ <sub>,</sub> th<sub>e square-roo</sub>t t<sub>erm</sub> i<sub>s a</sub>t <sub>mos</sub>t $p T / 8 .$ <sub>.</sub> O<sub>n</sub> thi<sub>s even</sub>t <sub>an</sub>d $\mathcal { E } ,$ <sub>,</sub> th<sub>e num</sub>b<sub>er o</sub>f f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s w</sub>ith $Q _ { t } \geq \varepsilon$ i<sub>s a</sub>t l<sub>eas</sub>t

$$
N _ { T } - H _ { T } \geq { \frac { p } { 2 } } T - { \frac { 3 p } { 8 } } T = { \frac { p } { 8 } } T .
$$

A<sub>ny</sub> <sub>pa</sub>i<sub>r</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>ncurs</sub> <sub>regre</sub>t <sub>a</sub>t l<sub>eas</sub>t $\Delta _ { t , b _ { t } } / 2 = g Q _ { t } / 2$ <sub>.</sub> Si<sub>nce a</sub>ll <sub>o</sub>th<sub>er roun</sub>d<sub>s</sub> h<sub>ave</sub> <sub>nonnega</sub>ti<sub>ve</sub> <sub>regre</sub>t<sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathsf { R e g } ( T ) \geq \frac { p } { 8 } T \cdot \frac { g \varepsilon } { 2 } = \frac { p ^ { 3 } } { 5 1 2 } T .
$$

A union bound <sub>p</sub>roves (39); inde<sub>p</sub>endence between $\mathcal { E }$ <sub>an</sub>d th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>-coun</sub>ti<sub>ng</sub> <sub>even</sub>t i<sub>s</sub> <sub>no</sub>t <sub>requ</sub>i<sub>re</sub>d<sub>.</sub> A b<sub>ase</sub>li<sub>ne</sub> <sub>se</sub>lf<sub>-</sub>d<sub>ue</sub>l i<sub>ncurs</sub> <sub>regre</sub>t $g Q _ { t }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>ou</sub>bl<sub>es</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>.</sub>

<sup>P</sup>erformance of MaxInP<sup>9</sup>BC. <sup>Th</sup>e same <sup>i</sup>nstance sat<sup>i</sup>s<sup>fi</sup>es every assumpt<sup>i</sup>on o<sup>f</sup> <sup>Th</sup>eorem <sup>3</sup>.<sup>1</sup>, w<sup>i</sup>t<sup>h</sup>

$$
\Delta _ { h } = g = \frac { p } { 4 } , \qquad \underline { { s } } = \sigma ( - g ) > 0 .
$$

<sup>A</sup>pp<sup>l</sup>y<sup>i</sup>ng t<sup>h</sup>at t<sup>h</sup>eorem to t<sup>h</sup>e trajectory o<sup>f</sup> MaxInP<sup>9</sup>BC g<sup>i</sup>ves, w<sup>i</sup>t<sup>h</sup> pro<sup>b</sup>a<sup>bili</sup>ty at <sup>l</sup>east $1 - \delta .$ <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>or every</sub> $T \geq 1$

$$
N _ { T } \leq N _ { \mathrm { c o n s } } , \qquad \mathtt { R e g } ( T ) \leq G ( T ) + \frac { g } { 2 } N _ { \mathrm { c o n s } } = G ( T ) + \frac { p } 8 N _ { \mathrm { c o n s } } .
$$

F<sub>or</sub> fi<sub>xe</sub>d <sub>parame</sub>t<sub>ers,</sub> $G ( T ) = O ( \sqrt { T } \log ( e T ) ) = o ( T )$ <sub>an</sub>d $N _ { \mathrm { c o n s } } < \infty$ b<sub>y</sub> (15). This <sub>p</sub>roves the <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>su</sub>bli<sub>near regre</sub>t b<sub>oun</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> li<sub>near grow</sub>th <sub>o</sub>f $W _ { t }$ <sub>or po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se convergence o</sub>f $c _ { t } . \mathsf { l }$

Remark 8 (Scope of the separation) Proposition A.2 concerns the specific certification rule in Definition A.1. The parameters $\theta ^ { \star }$ <sub>an</sub>d $\theta ^ { \circ }$ i<sub>n</sub>d<sub>uce</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>compar</sub>i<sub>son</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es,</sub> b<sub>u</sub>t th<sub>ey</sub> d<sub>o</sub> <sub>no</sub>t agree with the same supplied absolute baseline utilities. The latter information is not imposed on the confidence set (35).

K<sub>nown</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>u</sub>tiliti<sub>es</sub> <sub>perm</sub>it th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub>

$$
f ( \mathbf { m } _ { i } ) = \mu _ { i } ^ { 0 } + \langle \theta ^ { \star } , \mathbf { m } _ { i } - \mathbf { x } _ { b _ { i } } \rangle .
$$

Consequently,for baseline self-duelfallbacks, a valid alternative cumulative test is

$$
\alpha \sum _ { i = 1 } ^ { t } \mu _ { i } ^ { 0 } + \operatorname* { i n f } _ { \theta \in { \cal K } _ { t } } \left. \theta , \sum _ { i \in \mathcal { A } _ { t - 1 } } ( { \bf m } _ { i } - { \bf x } _ { b _ { i } } ) + \widetilde { { \bf m } } _ { t } - { \bf x } _ { b _ { t } } \right. \ge 0 .\tag{44}
$$

On E, the left side lower-bounds the cumulative cardinal surplus after admitting the candidate. The same conclusion holds for otherfallbacks whose average utilities are certified to be at least their b<sub>ase</sub>li<sub>ne u</sub>tiliti<sub>es.</sub>

On the constructed instance, every centered vector in (44) has zero first coordinate, so the common utility component disappears. Incorporating the known-baseline equalities into the confidence set is another possible modification. The proposition therefore establishes a limitation of the direct cumulative transfer, rather than an impossibility of conservative cardinal certification under preferencefeedback.

Remark 9 (Score certification through comparisons) Scores can also be certified directlyfrom pairwise diferences. On E,for every $\mathbf { y } \in { S } _ { t }$

$$
s _ { t } ( \mathbf { y } ) \geq \sigma \biggl ( - \operatorname* { m a x } _ { \mathbf { x } \in S _ { t } } \Big \{ \langle \hat { \theta } _ { t } , \mathbf { x } - \mathbf { y } \rangle + \beta _ { t } \| \mathbf { x } - \mathbf { y } \| _ { V _ { t } ^ { - 1 } } \Big \} \biggr ) .\tag{45}
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>, eac</sub>h <sub>express</sub>i<sub>on</sub> i<sub>ns</sub>id<sub>e</sub> th<sub>e max</sub>i<sub>mum upper-</sub>b<sub>oun</sub>d<sub>s</sub> $f ( \mathbf { x } ) - f ( \mathbf { y } )$ <sub>,</sub> <sub>an</sub>d

$$
s _ { t } ( \mathbf { y } ) = \sigma \left( - \operatorname* { m a x } _ { \mathbf { x } \in S _ { t } } \{ f ( \mathbf { x } ) - f ( \mathbf { y } ) \} \right) .
$$

No estimate ofan absolute utility is needed.

Our construction uses the uniform optimality-gap certificate c from Lemma 3.2 and converts it into a score guarantee through Lemma 3.3. The candidate selection and informativefallback ensure, by (13), that

$$
c _ { t } \leq 4 \bar { \beta } _ { t } \operatorname* { m i n } \{ 1 , \| \mathbf { z } _ { t } \| _ { V _ { \star } ^ { - 1 } } \} , \qquad \bar { \beta } _ { t } : = \operatorname* { m a x } \{ B / 2 , \beta _ { t } \} .
$$

Thus its cumulative certification error is bounded by $G ( T )$ <sub>a</sub>l<sub>ong</sub> th<sub>e compar</sub>i<sub>sons ac</sub>t<sub>ua</sub>ll<sub>y o</sub>b<sub>serve</sub>d<sub>.</sub> This connection between certification and information acquisition yields the finite fallback bound in Th<sub>eorem</sub> 3<sub>.</sub>1<sub>.</sub>

## A.10. Relaxing knowledge of baseline scores

The baseline constraint (3) de<sub>p</sub>ends onl<sub>y</sub> on utilit<sub>y</sub> diferences and is therefore invariant to additive <sub>s</sub>hift<sub>s o</sub>f $f .$ <sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> f<sub>or every</sub> $a \in \mathbb { R }$

$$
\sigma \big ( \big ( f ( \mathbf { x } ) + a \big ) - \big ( f ( \mathbf { x } _ { t } ^ { \star } ) + a \big ) \big ) = \sigma \big ( f ( \mathbf { x } ) - f ( \mathbf { x } _ { t } ^ { \star } ) \big ) .
$$

C<sub>onsequen</sub>tl<sub>y, exac</sub>t k<sub>now</sub>l<sub>e</sub>d<sub>ge o</sub>f $s _ { t } ( \mathbf { x } _ { b _ { t } } )$ i<sub>s an op</sub>ti<sub>ona</sub>l <sub>source o</sub>f i<sub>n</sub>f<sub>orma</sub>ti<sub>on, ra</sub>th<sub>er</sub> th<sub>an a</sub> <sub>requ</sub>i<sub>remen</sub>t f<sub>or</sub> <sub>reso</sub>l<sub>v</sub>i<sub>ng</sub> <sub>an</sub> <sub>a</sub>dditi<sub>ve</sub> <sub>u</sub>tilit<sub>y</sub> <sub>o</sub>f<sub>se</sub>t<sub>.</sub> Thi<sub>s</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>on</sub> <sub>proves</sub> th<sub>a</sub>t <sub>va</sub>lid l<sub>ower</sub> <sub>an</sub>d <sub>upper</sub> b<sub>oun</sub>d<sub>s on</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne score su</sub>fi<sub>ce.</sub>

Available baseline-score bounds. Retain Assumptions 1, 2, 3, and 4, but replace the exact-score i<sub>npu</sub>t t<sub>o</sub> Al<sub>gor</sub>ith<sub>m</sub> 1 b<sub>y</sub> t<sub>wo</sub> b<sub>oun</sub>d<sub>s</sub> $L _ { t } , U _ { t }$ . These bounds must be available before the round-t pair i<sub>s se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>an</sub>d <sub>sa</sub>ti<sub>s</sub>f<sub>y, on every rea</sub>li<sub>za</sub>ti<sub>on,</sub>

$$
\begin{array} { r } { \underline { { s } } \le L _ { t } \le U _ { t } \le \frac { 1 } { 2 } , \qquad \underline { { s } } = \sigma ( - \Delta _ { h } ) > 0 . } \end{array}\tag{46}
$$

Th<sub>e</sub> BTL <sub>assump</sub>ti<sub>on</sub> i<sub>s</sub> <sub>un</sub>d<sub>ers</sub>t<sub>oo</sub>d <sub>con</sub>diti<sub>ona</sub>ll<sub>y</sub> <sub>on</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>pre-</sub>f<sub>ee</sub>db<sub>ac</sub>k i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>any</sub> <sub>s</sub>id<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on use</sub>d t<sub>o cons</sub>t<sub>ruc</sub>t th<sub>ese</sub> b<sub>oun</sub>d<sub>s.</sub>

F<sub>or some</sub> $\delta _ { b } \in [ 0 , 1 )$ <sub>,</sub> <sub>assume</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> <sub>coverage:</sub>

$$
\mathbb { P } ( \mathcal { E } _ { b } ) \geq 1 - \delta _ { b } , \qquad \mathcal { E } _ { b } : = \bigcap _ { t \geq 1 } \left\{ L _ { t } \leq s _ { t } ( \mathbf { x } _ { b _ { t } } ) \leq U _ { t } \right\} .\tag{47}
$$

D<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y va</sub>lid b<sub>oun</sub>d<sub>s correspon</sub>d t<sub>o</sub> $\delta _ { b } \ : = \ : 0$ <sub>.</sub> N<sub>o</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> b<sub>e</sub>t<sub>ween</sub> $\mathcal { E } _ { b }$ <sub>an</sub>d th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-con</sub>fid<sub>ence</sub> <sub>even</sub>t i<sub>s</sub> <sub>requ</sub>i<sub>re</sub>d<sub>.</sub>

Modified safety budget. Compute $\widehat { \theta } _ { t } , \beta _ { t } , { \mathcal { C } } _ { t } , ( \widetilde { { \mathbf x } } _ { t } , \widetilde { { \mathbf y } } _ { t } ) , w _ { t }$ <sub>,</sub> <sub>an</sub>d $c _ { t }$ usin<sub>g</sub> (5)–(10), and define the fallback <sub>p</sub>artner b<sub>y</sub> (12). Initialize $\Lambda _ { 0 } = 0$ <sub>an</sub>d <sub>se</sub>t

$$
q _ { t } : = \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) U _ { t } .\tag{48}
$$

Th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r</sub> <sub>an</sub>d b<sub>u</sub>d<sub>ge</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>are</sub>

$$
\begin{array} { r } { \big ( \mathbf x _ { t } , \mathbf y _ { t } \big ) = \left\{ \begin{array} { l l } { \big ( \tilde { \mathbf x } _ { t } , \tilde { \mathbf y } _ { t } \big ) , } & { \Lambda _ { t - 1 } + q _ { t } \geq 0 , } \\ { \big ( \mathbf x _ { b _ { t } } , \mathbf y _ { t } ^ { \mathrm { f b } } \big ) , } & { \Lambda _ { t - 1 } + q _ { t } < 0 , } \end{array} \right. } \end{array}\tag{49}
$$

$$
\Lambda _ { t } = \left\{ \begin{array} { l l } { \Lambda _ { t - 1 } + q _ { t } , } & { \Lambda _ { t - 1 } + q _ { t } \geq 0 , } \\ { \Lambda _ { t - 1 } + \alpha L _ { t } , } & { \Lambda _ { t - 1 } + q _ { t } < 0 . } \end{array} \right.\tag{50}
$$

Aft<sub>er</sub> di<sub>sp</sub>l<sub>ay</sub>i<sub>ng</sub> th<sub>e pa</sub>i<sub>r, o</sub>b<sub>serve</sub> $o _ { t } .$ , set ${ \bf z } _ { t } = { \bf x } _ { t } - { \bf y } _ { t }$ <sub>, an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gn ma</sub>t<sub>r</sub>i<sub>x</sub> b<sub>y</sub> $V _ { t + 1 } =$ $V _ { t } + \mathbf { z } _ { t } \mathbf { z } _ { t } ^ { \top }$ <sub>.</sub> Th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> <sub>con</sub>ti<sub>nues</sub> t<sub>o</sub> <sub>use</sub> <sub>a</sub>ll di<sub>sp</sub>l<sub>aye</sub>d <sub>compar</sub>i<sub>sons.</sub>

Th<sub>e</sub> <sub>upper</sub> b<sub>oun</sub>d $U _ { t }$ <sub>appears</sub> i<sub>n</sub> th<sub>e</sub> <sub>su</sub>bt<sub>rac</sub>t<sub>e</sub>d b<sub>ase</sub>li<sub>ne</sub> t<sub>erm</sub> <sub>o</sub>f th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>es</sub>t<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d $L _ { t }$ d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e</sub> <sub>guaran</sub>t<sub>ee</sub>d <sub>cre</sub>dit f<sub>rom</sub> <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>.</sub> Th<sub>e</sub> <sub>un</sub>k<sub>nown</sub> <sub>quan</sub>tit<sub>y</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } )$ i<sub>s</sub> <sub>no</sub>t <sub>use</sub>d b<sub>y</sub> thi<sub>s</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>ru</sub>l<sub>e.</sub>

L<sub>e</sub>t $A _ { T } , B _ { T }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s o</sub>f thi<sub>s var</sub>i<sub>an</sub>t<sub>, an</sub>d <sub>wr</sub>it<sub>e</sub> $M _ { T } = | A _ { T } |$ <sub>an</sub>d $N _ { T } = | \boldsymbol { B _ { T } } |$ <sub>.</sub> Use $\beta _ { t }$ from (6), with model-confidence parameter δ, and G and $N _ { \mathrm { c o n s } }$ from (15).

Theorem A.1 (Guarantees with baseline-score bounds) Suppose the assumptions above hold, <sub>w</sub>ith $\alpha , \delta \in ( 0 , 1 ) , \lambda > 0$ <sub>,</sub> <sub>an</sub>d $\delta + \delta _ { b } < 1$ . Run the variant (48)– (50). With <sub>p</sub>robabilit<sub>y</sub> at least $1 - \delta - \delta _ { b } ,$ , simultaneously for every $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \operatorname* { m a x } \{ s _ { t } ( \mathbf { x } _ { t } ) , s _ { t } ( \mathbf { y } _ { t } ) \} \geq ( 1 - \alpha ) \sum _ { t = 1 } ^ { T } s _ { t } ( \mathbf { x } _ { b _ { t } } ) ,\tag{51}
$$

$$
N _ { T } \leq \operatorname* { m i n } \{ T , N _ { \mathrm { c o n s } } \} ,\tag{52}
$$

$$
\mathsf { R e g } ( T ) \leq G ( T ) + \frac { \Delta _ { h } } { 2 } N _ { \mathrm { c o n s } } .\tag{53}
$$

In particular, deterministically valid bounds preserve the exact guarantees and confidence level of Th<sub>eorem</sub> 3<sub>.</sub>1<sub>.</sub>

Proof We first establish the relevant model-confidence event. The proof of Lemma 3.1 applies to any <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub>l<sub>y se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>compar</sub>i<sub>son vec</sub>t<sub>ors</sub> th<sub>a</sub>t <sub>are measura</sub>bl<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub>i<sub>r</sub> f<sub>ee</sub>db<sub>ac</sub>k i<sub>s o</sub>b<sub>serve</sub>d<sub>.</sub> Th<sub>e</sub> <sub>mo</sub>difi<sub>e</sub>d <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> t<sub>es</sub>t <sub>preserves</sub> thi<sub>s</sub> <sub>proper</sub>t<sub>y.</sub> H<sub>ence</sub>

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { E } ) \geq 1 - \delta , \qquad \mathcal { E } : = \left\{ \| \hat { { \boldsymbol \theta } } _ { t } - { \boldsymbol \theta } ^ { \star } \| _ { V _ { t } } \leq \beta _ { t } \mathrm { f o r e v e r y } t \geq 1 \right\} . } \end{array}
$$

Th<sub>e</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { P } ( \mathcal { E } \cap \mathcal { E } _ { b } ) \geq 1 - \delta - \delta _ { b } .\tag{54}
$$

W<sub>e prove a</sub>ll th<sub>ree conc</sub>l<sub>us</sub>i<sub>ons on</sub> thi<sub>s s</sub>i<sub>ng</sub>l<sub>e even</sub>t<sub>.</sub>

Certificates and their cumulative envelope. On E, Lemma 3.2 gives

$$
f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } ) \leq c _ { t } \qquad { \mathrm { f o r ~ e v e r y ~ } } \mathbf { y } \in { \mathcal { C } } _ { t } .\tag{55}
$$

I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>pa</sub>i<sub>r</sub> h<sub>as</sub> i<sub>ns</sub>t<sub>an</sub>t<sub>aneous</sub> <sub>regre</sub>t <sub>a</sub>t <sub>mos</sub>t $c _ { t } .$

W<sub>r</sub>it<sub>e</sub>

$$
v _ { t } : = \left. \mathbf { z } _ { t } \right. _ { V _ { t } ^ { - 1 } } , \qquad \bar { \beta } _ { t } : = \operatorname* { m a x } \{ B / 2 , \beta _ { t } \} ,
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { z } _ { t }$ i<sub>s</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>vec</sub>t<sub>or</sub> <sub>ac</sub>t<sub>ua</sub>ll<sub>y</sub> di<sub>sp</sub>l<sub>aye</sub>d<sub>.</sub> O<sub>n</sub> <sub>a</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>roun</sub>d<sub>,</sub> $w _ { t } = v _ { t }$ <sub>.</sub> O<sub>n a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>,</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f ${ \bf y } _ { t } ^ { \mathrm { f b } }$ <sub>g</sub><sup>i</sup>ve

$$
\begin{array} { r l } & { w _ { t } = \| \tilde { { \mathbf x } } _ { t } - \tilde { { \mathbf y } } _ { t } \| _ { V _ { t } ^ { - 1 } } } \\ & { \quad \leq \| \tilde { { \mathbf x } } _ { t } - { \mathbf x } _ { b _ { t } } \| _ { V _ { t } ^ { - 1 } } + \| { \mathbf x } _ { b _ { t } } - \tilde { { \mathbf y } } _ { t } \| _ { V _ { t } ^ { - 1 } } } \\ & { \quad \leq 2 \| { \mathbf x } _ { b _ { t } } - { \mathbf y } _ { t } ^ { \mathrm { f b } } \| _ { V _ { t } ^ { - 1 } } = 2 v _ { t } . } \end{array}
$$

Th<sub>us, on e</sub>ith<sub>er</sub> t<sub>ype o</sub>f <sub>roun</sub>d<sub>,</sub>

$$
c _ { t } \leq \operatorname* { m i n } \{ B , 4 \beta _ { t } v _ { t } \} \leq 4 \bar { \beta } _ { t } \operatorname* { m i n } \{ 1 , v _ { t } \} .\tag{56}
$$

F<sub>or</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y,</sub> if $v _ { t } \leq 1$ , use $c _ { t } \leq 4 \beta _ { t } v _ { t }$ <sub>.</sub> If $v _ { t } > 1$ , use $c _ { t } \leq B \leq 2 \bar { \beta } _ { t }$

Th<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t l<sub>emma</sub> <sub>g</sub>i<sub>ves</sub>

$$
\operatorname* { d e t } ( V _ { t + 1 } ) = \operatorname* { d e t } ( V _ { t } ) ( 1 + v _ { t } ^ { 2 } ) .
$$

Since min $\{ 1 , u \} \le 2 \log ( 1 + u )$ <sup>f</sup>or ever<sub>y</sub> $u \geq 0$

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \{ 1 , v _ { t } ^ { 2 } \} \leq 2 \displaystyle \sum _ { t = 1 } ^ { T } \log ( 1 + v _ { t } ^ { 2 } ) } \\ { = 2 \log \frac { \operatorname* { d e t } ( V _ { T + 1 } ) } { \operatorname* { d e t } ( V _ { 1 } ) } } \\ { \leq 2 d \log \left( 1 + \frac { 4 L ^ { 2 } T } { \lambda d } \right) . } \end{array}\tag{57}
$$

Th<sub>e</sub> fi<sub>na</sub>l i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> <sub>ar</sub>ith<sub>me</sub>ti<sub>c–geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>mean</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> <sub>e</sub>i<sub>genva</sub>l<sub>ues</sub> <sub>o</sub>f $V _ { T + 1 }$ <sub>,</sub> t<sub>oge</sub>th<sub>er w</sub>ith

$$
\mathrm { t r } ( V _ { T + 1 } ) = \lambda d + \sum _ { t = 1 } ^ { T } \Vert \mathbf { z } _ { t } \Vert _ { 2 } ^ { 2 } \leq \lambda d + 4 L ^ { 2 } T .
$$

B<sub>ecause</sub> $\bar { \beta } _ { t }$ is nondecreasing, Cauchy–Schwarz and min $\{ 1 , v \} ^ { 2 } = \operatorname* { m i n } \{ 1 , v ^ { 2 } \}$ <sub>y</sub>i<sub>e</sub>ld

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } c _ { t } \leq 4 \bar { \beta } _ { T } \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \{ 1 , v _ { t } \} } \\ { \leq 4 \bar { \beta } _ { T } \sqrt { T \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \{ 1 , v _ { t } ^ { 2 } \} } } \\ { \leq 4 \bar { \beta } _ { T } \sqrt { 2 d T \log \left( 1 + \frac { 4 L ^ { 2 } T } { \lambda d } \right) } = G ( T ) . } \end{array}\tag{58}
$$

Thi<sub>s</sub> b<sub>oun</sub>d i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> $c _ { t }$ on fallback rounds, even though it is the certificate of the rejected candidate. Th<sub>e</sub> <sub>w</sub>idth<sub>-</sub>t<sub>rans</sub>f<sub>er</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>a</sub>t <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> th<sub>roug</sub>h th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>vec</sub>t<sub>or</sub> <sub>ac</sub>t<sub>ua</sub>ll<sub>y</sub> di<sub>sp</sub>l<sub>aye</sub>d<sub>.</sub>

Safety. Define the true cumulative surplus by

$$
\Sigma _ { t } : = \sum _ { i = 1 } ^ { t } \left[ \operatorname* { m a x } \{ s _ { i } ( \mathbf { x } _ { i } ) , s _ { i } ( \mathbf { y } _ { i } ) \} - ( 1 - \alpha ) s _ { i } ( \mathbf { x } _ { b _ { i } } ) \right] , \quad \quad \Sigma _ { 0 } = 0 .\tag{59}
$$

W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t

$$
\Sigma _ { t } \geq \Lambda _ { t } \geq 0 \qquad \mathrm { f o r e v e r y } t \geq 0 .\tag{60}
$$

O<sub>n a can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>,</sub> L<sub>emmas</sub> 3<sub>.</sub>2 <sub>an</sub>d 3<sub>.</sub>3 i<sub>mp</sub>l<sub>y</sub>

$$
\operatorname* { m a x } \{ s _ { t } ( \mathbf { x } _ { t } ) , s _ { t } ( \mathbf { y } _ { t } ) \} \geq \frac { 1 } { 2 } - \frac { c _ { t } } { 4 } .
$$

Si<sub>nce</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } ) \leq U _ { t }$ on $\mathcal { E } _ { b }$

$$
\begin{array} { l } { \displaystyle \Sigma _ { t } - \Sigma _ { t - 1 } \geq \frac 1 2 - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) s _ { t } ( \mathbf { x } _ { b _ { t } } ) } \\ { \displaystyle \qquad \geq \frac 1 2 - \frac { c _ { t } } { 4 } - ( 1 - \alpha ) U _ { t } } \\ { \displaystyle \qquad = q _ { t } = \Lambda _ { t } - \Lambda _ { t - 1 } . } \end{array}
$$

Th<sub>e accep</sub>t<sub>a</sub>n<sub>ce</sub> t<sub>es</sub>t <sub>e</sub>n<sub>su</sub>r<sub>es</sub> $\Lambda _ { t } \geq 0$

O<sub>n</sub> <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>,</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> $\mathbf { x } _ { b _ { t } }$ , so

$$
\begin{array} { r l } & { \sum _ { t } - \sum _ { t - 1 } \geq s _ { t } \bigl ( \mathbf { x } _ { b _ { t } } \bigr ) - ( 1 - \alpha ) s _ { t } \bigl ( \mathbf { x } _ { b _ { t } } \bigr ) } \\ & { \qquad = \alpha s _ { t } \bigl ( \mathbf { x } _ { b _ { t } } \bigr ) } \\ & { \qquad \geq \alpha L _ { t } = \Lambda _ { t } - \Lambda _ { t - 1 } . } \end{array}
$$

M<sub>oreover,</sub> $L _ { t } \ \ge \ s > \ 0$ <sub>,</sub> <sub>so</sub> <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>preserves</sub> <sub>nonnega</sub>ti<sub>v</sub>it<sub>y</sub> <sub>o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> St<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> $\Sigma _ { 0 } = \Lambda _ { 0 } = 0$ , induction <sub>p</sub>roves (60). Ex<sub>p</sub>andin<sub>g</sub> $\Sigma _ { T } \geq 0$ <sub>g</sub>ives (51).

Fallback count. Expanding the modified budget gives

$$
\Lambda _ { t } = \sum _ { i \in \mathcal { A } _ { t } } \left[ \frac { 1 } { 2 } - \frac { c _ { i } } { 4 } - ( 1 - \alpha ) U _ { i } \right] + \sum _ { i \in \mathcal { B } _ { t } } \alpha L _ { i } .\tag{61}
$$

L<sub>e</sub>t $\tau$ be a fallback round. By the rejection condition,

$$
\Lambda _ { \tau - 1 } + \frac { 1 } { 2 } - \frac { c _ { \tau } } { 4 } - ( 1 - \alpha ) U _ { \tau } < 0 .
$$

Substitutin<sub>g</sub> (61) and rearran<sub>g</sub>in<sub>g</sub> <sub>y</sub>ields

$$
\begin{array} { c } { \displaystyle \sum _ { i \in \mathcal { A } _ { \tau - 1 } } \left[ \frac { 1 } { 2 } - ( 1 - \alpha ) U _ { i } \right] + \displaystyle \sum _ { i \in \mathcal { B } _ { \tau - 1 } } \alpha L _ { i } + \frac { 1 } { 2 } - ( 1 - \alpha ) U _ { \tau } } \\ { \displaystyle < \frac { 1 } { 4 } \left( \displaystyle \sum _ { i \in \mathcal { A } _ { \tau - 1 } } c _ { i } + c _ { \tau } \right) . } \end{array}\tag{62}
$$

The end<sub>p</sub>oint restrictions (46) im<sub>p</sub>l<sub>y</sub>

$$
\frac { 1 } { 2 } - ( 1 - \alpha ) U _ { i } \geq \frac { \alpha } { 2 } , \qquad \alpha L _ { i } \geq \alpha \underline { { { s } } } .
$$

Since all certificates are nonne<sub>g</sub>ative, (58) <sub>g</sub>ives

$$
\sum _ { i \in \mathcal { A } _ { \tau - 1 } } c _ { i } + c _ { \tau } \leq \sum _ { i = 1 } ^ { \tau } c _ { i } \leq G ( \tau ) .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\frac { \alpha } { 2 } M _ { \tau - 1 } + \alpha \underline { { s } } N _ { \tau - 1 } + \frac { \alpha } { 2 } < \frac { 1 } { 4 } G ( \tau ) .\tag{63}
$$

Usin<sub>g</sub> $\underline { { s } } \le \frac { 1 } { 2 }$ <sub>an</sub>d $M _ { \tau - 1 } + { \cal N } _ { \tau - 1 } = \tau - 1$ , the left-hand side is at least αsτ. Hence every fallback ti<sub>me sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
4 \alpha \underline { s } \tau < G ( \tau ) .
$$

B<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $N _ { \mathrm { c o n s } }$ in (15), ever<sub>y</sub> fallback time is at most $N _ { \mathrm { c o n s } }$ <sub>.</sub> B<sub>ecause</sub> f<sub>a</sub>llb<sub>ac</sub>k ti<sub>mes are</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>pos</sub>iti<sub>ve</sub> i<sub>n</sub>t<sub>egers,</sub>

$$
N _ { T } \leq \operatorname* { m i n } \{ T , N _ { \mathrm { c o n s } } \} .
$$

F<sub>o</sub>r <sub>co</sub>m<sub>p</sub>l<sub>e</sub>t<sub>e</sub>n<sub>ess,</sub> $N _ { \mathrm { c o n s } }$ is finite. For fixed <sub>p</sub>roblem <sub>p</sub>arameters and δ, (6) <sub>g</sub>ives $\bar { \beta } _ { t } = O ( \sqrt { \log ( e t ) } )$ , an<sup>d</sup> conse<sub>q</sub>uent<sup>l</sup><sub>y</sub>

$$
G ( t ) = O \bigl ( { \sqrt { t } } \log ( e t ) \bigr ) = o ( t ) .
$$

Si<sub>nce</sub> $\alpha s > 0$ , the inequality 4αst $< G ( t )$ can hold only for bounded t.

Regret. On a candidate round, Lemma 3.2 gives $R _ { t } \leq c _ { t }$ <sub>.</sub> O<sub>n a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>,</sub> ${ \bf y } _ { t } ^ { \mathrm { f b } } \in \mathcal { C } _ { t }$ , so (55) <sub>an</sub>d A<sub>ssump</sub>ti<sub>on</sub> 3 <sub>g</sub>i<sub>ve</sub>

$$
\begin{array} { l l l c } { { \displaystyle R _ { t } = \frac { 1 } { 2 } \left[ f ( { \bf x } _ { t } ^ { \star } ) - f ( { \bf x } _ { b _ { t } } ) + f ( { \bf x } _ { t } ^ { \star } ) - f ( { \bf y } _ { t } ^ { \mathrm { f b } } ) \right] } } \\ { { \displaystyle \quad \le \frac { 1 } { 2 } ( \Delta _ { h } + c _ { t } ) . } } \end{array}
$$

It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
\begin{array} { r l } & { { \tt R e g } ( T ) \leq \displaystyle \sum _ { t \in { \cal A } _ { T } } c _ { t } + \frac { 1 } { 2 } \sum _ { t \in { \cal B } _ { T } } c _ { t } + \frac { \Delta _ { h } } { 2 } N _ { T } } \\ & { \quad \quad \quad \leq \displaystyle \sum _ { t = 1 } ^ { T } c _ { t } + \frac { \Delta _ { h } } { 2 } N _ { T } } \\ & { \quad \quad \quad \leq G ( T ) + \frac { \Delta _ { h } } { 2 } N _ { \mathrm { c o n s } } . } \end{array}
$$

All conclusions hold simultaneously for every T on $\mathcal { E } \cap \mathcal { E } _ { b }$ . Equation (54) therefore <sub>p</sub>roves the stated <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>guaran</sub>t<sub>ee.</sub>

A variant requiring no baseline-score estimates. Assumption 3 and the definition of the preference <sub>score g</sub>i<sub>ve</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> b<sub>oun</sub>d<sub>s</sub>

$$
\begin{array} { r } { \underline { { s } } \le s _ { t } ( { \mathbf { x } } _ { b _ { t } } ) \le \frac { 1 } { 2 } . } \end{array}
$$

Th<sub>us</sub> <sub>one</sub> <sub>may</sub> <sub>se</sub>t

$$
L _ { t } = \underline { { s } } , \qquad U _ { t } = { \textstyle \frac { 1 } { 2 } } \qquad \mathrm { f o r e v e r y } t .\tag{64}
$$

I<sub>n</sub> thi<sub>s case,</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>s accep</sub>t<sub>e</sub>d <sub>prec</sub>i<sub>se</sub>l<sub>y w</sub>h<sub>en</sub>

$$
\Lambda _ { t - 1 } + \frac { \alpha } { 2 } - \frac { c _ { t } } { 4 } \geq 0 ,
$$

<sub>an</sub>d th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>ncremen</sub>t<sub>s</sub> <sub>are</sub>

$$
\Lambda _ { t } - \Lambda _ { t - 1 } = { \left\{ \begin{array} { l l } { \displaystyle { \frac { \alpha } { 2 } } - { \frac { c _ { t } } { 4 } } , } & { t \in { \mathcal { A } } _ { T } , } \\ { \displaystyle { \alpha } \underline { { s } } , } & { t \in B _ { T } , } \end{array} \right. } \qquad t \leq T .\tag{65}
$$

H<sub>ere</sub> $\mathcal { E } _ { b }$ h<sub>o</sub>ld<sub>s sure</sub>l<sub>y, so</sub> $\delta _ { b } = 0$ <sub>an</sub>d Th<sub>eorem</sub> A<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> th<sub>e same exac</sub>t b<sub>oun</sub>d<sub>s an</sub>d <sub>con</sub>fid<sub>ence</sub> l<sub>eve</sub>l <sub>as</sub> Th<sub>eorem</sub> 3<sub>.</sub>1<sub>.</sub> Th<sub>e</sub> l<sub>earner</sub> <sub>nee</sub>d<sub>s</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> it<sub>em</sub> <sub>an</sub>d th<sub>e</sub> k<sub>nown</sub> <sub>gap</sub> b<sub>oun</sub>d $\Delta _ { h }$ <sub>,</sub> b<sub>u</sub>t <sub>no</sub> <sub>exac</sub>t <sub>or</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d b<sub>ase</sub>li<sub>ne score.</sub>

If th<sub>e</sub> l<sub>ower</sub> <sub>gap</sub> b<sub>oun</sub>d $\Delta _ { l }$ i<sub>s</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e,</sub> th<sub>e</sub> ti<sub>g</sub>ht<sub>er</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> <sub>upper</sub> b<sub>oun</sub>d $U _ { t } = \sigma ( - \Delta _ { l } )$ <sup>ma</sup>y i<sub>ns</sub>t<sub>ea</sub>d b<sub>e use</sub>d<sub>.</sub> It <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\begin{array} { r } { U _ { t } \le \frac { 1 } { 2 } } \end{array}$ <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> <sub>preserves</sub> th<sub>e</sub> <sub>same</sub> <sub>guaran</sub>t<sub>ees.</sub>

Statistical bounds and confidence accounting. Statistical intervals must cover the actual contextd<sub>epen</sub>d<sub>en</sub>t <sub>scores</sub> $s _ { t } ( \mathbf { x } _ { b _ { t } } )$ <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y over</sub> ti<sub>me.</sub> A <sub>con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l f<sub>or a marg</sub>i<sub>na</sub>l <sub>average w</sub>i<sub>n</sub> rate does not b<sub>y</sub> itself establish (47).

F<sub>o</sub>r <sub>exa</sub>m<sub>p</sub>l<sub>e,</sub> <sub>suppose</sub> <sub>a</sub> <sub>p</sub>r<sub>oce</sub>d<sub>u</sub>r<sub>e</sub> <sub>p</sub>r<sub>o</sub>d<sub>uces</sub> int<sub>e</sub>r<sub>va</sub>l<sub>s</sub> $[ \widetilde { L } _ { t } , \widetilde { U } _ { t } ]$ before the round-t decision and <sub>guaran</sub>t<sub>ees, un</sub>d<sub>er</sub> th<sub>e a</sub>d<sub>ap</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on pro</sub>t<sub>oco</sub>l<sub>,</sub>

$$
\mathbb { P } \left( s _ { t } ( \mathbf { x } _ { b _ { t } } ) \notin [ \widetilde { L } _ { t } , \widetilde { U } _ { t } ] \right) \le \delta _ { b , t } , \qquad \sum _ { t = 1 } ^ { \infty } \delta _ { b , t } \le \delta _ { b } .
$$

Th<sub>e</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d th<sub>en</sub> <sub>g</sub>i<sub>ves</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> <sub>coverage</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta _ { b }$ <sub>.</sub> O<sub>ne</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\delta _ { b , t } = \frac { 6 \delta _ { b } } { \pi ^ { 2 } t ^ { 2 } } .
$$

Alt<sub>erna</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>a</sub> <sub>proce</sub>d<sub>ure</sub> <sub>may</sub> <sub>prov</sub>id<sub>e</sub> <sub>a</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> <sub>con</sub>fid<sub>ence</sub> <sub>sequence</sub> di<sub>rec</sub>tl<sub>y.</sub>

To enforce (46), intersect each su<sub>pp</sub>lied interval with $[ \underline { { s } } , \frac { 1 } { 2 } ]$ <sub>.</sub> If th<sub>e</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on</sub> i<sub>s</sub> <sub>emp</sub>t<sub>y,</sub> <sub>use</sub> $[ \underline { { s } } , \frac { 1 } { 2 } ]$ i<sub>ns</sub>t<sub>ea</sub>d<sub>.</sub> Wh<sub>enever</sub> th<sub>e supp</sub>li<sub>e</sub>d i<sub>n</sub>t<sub>erva</sub>l <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> t<sub>rue score,</sub> it<sub>s</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on w</sub>ith thi<sub>s</sub> k<sub>nown range</sub> i<sub>s</sub> <sub>nonemp</sub>t<sub>y</sub> <sub>an</sub>d <sub>s</sub>till <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>a</sub>t <sub>score.</sub> Th<sub>us</sub> thi<sub>s</sub> <sub>opera</sub>ti<sub>on</sub> <sub>preserves</sub> th<sub>e</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> <sub>coverage</sub> <sub>guaran</sub>t<sub>ee</sub> <sub>w</sub>hil<sub>e</sub> <sub>ensur</sub>i<sub>ng</sub> th<sub>a</sub>t th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>s</sub> <sub>we</sub>ll<sub>-</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>on</sub> <sub>every</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on.</sub>

Th<sub>e mo</sub>d<sub>e</sub>l<sub>-con</sub>fid<sub>ence an</sub>d b<sub>ase</sub>li<sub>ne-coverage</sub> f<sub>a</sub>il<sub>ure pro</sub>b<sub>a</sub>biliti<sub>es a</sub>dd<sub>:</sub>

$$
\mathbb { P } ( \mathcal { E } \cap \mathcal { E } _ { b } ) \geq 1 - \delta - \delta _ { b } .
$$

F<sub>or</sub> <sub>a</sub> d<sub>es</sub>i<sub>re</sub>d t<sub>o</sub>t<sub>a</sub>l f<sub>a</sub>il<sub>ure</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $\delta _ { \mathrm { t o t } } \in ( 0 , 1 )$ , <sup>one ma</sup>y <sup>use</sup> $\delta = \delta _ { b } = \delta _ { \mathrm { t o t } } / 2$ <sub>.</sub> I<sub>n</sub> th<sub>a</sub>t <sub>case,</sub> $\beta _ { t } , G ,$ <sub>an</sub>d $N _ { \mathrm { c o n s } }$ <sub>are</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-con</sub>fid<sub>ence</sub> <sub>parame</sub>t<sub>er</sub> $\delta _ { \mathrm { t o t } } / 2$ <sub>.</sub> F<sub>or</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c c</sub>h<sub>o</sub>i<sub>ce</sub> (64), no s<sub>p</sub>lit is necessar<sub>y</sub>: take $\delta = \delta _ { \mathrm { t o t } }$ <sub>an</sub>d $\delta _ { b } = 0$

## Appendix B. Analysis for General Utility Functions

Thi<sub>s appen</sub>di<sub>x proves</sub> Th<sub>eorem</sub> 4<sub>.</sub>1<sub>, cons</sub>t<sub>ruc</sub>t<sub>s a</sub> fi<sub>n</sub>it<sub>e-c</sub>l<sub>ass orac</sub>l<sub>e, an</sub>d t<sub>rea</sub>t<sub>s</sub> b<sub>ase</sub>li<sub>ne-score</sub> b<sub>oun</sub>d<sub>s.</sub> Th<sub>roug</sub>h<sub>ou</sub>t<sub>,</sub> $B > 0 , K \geq 2$ <sub>,</sub> <sub>an</sub>d $\kappa = \sigma ( B ) ( 1 - \sigma ( B ) )$ ). All contexts and baseline information are <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> <sub>pa</sub>i<sub>r</sub> <sub>samp</sub>li<sub>ng,</sub> <sub>an</sub>d th<sub>e</sub> BTL l<sub>aw</sub> i<sub>s</sub> <sub>con</sub>diti<sub>one</sub>d <sub>on</sub> <sub>a</sub>ll i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>ou</sub>t<sub>come.</sub> Th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>a</sub>ll<sub>ows</sub> <sub>ar</sub>bit<sub>rary</sub> <sub>pre</sub>di<sub>c</sub>t<sub>a</sub>bl<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t <sub>se</sub>t<sub>s,</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e</sub> i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> <sub>mo</sub>d<sub>e</sub>l <sub>o</sub>f th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>paper.</sub> N<sub>o</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>s</sub> <sub>use</sub>d<sub>.</sub>

## B.1. Adapting C-SquareCB to preference feedback

Thi<sub>s</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>on</sub> <sub>exp</sub>l<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> <sub>c</sub>h<sub>anges</sub> t<sub>o</sub> <sub>cer</sub>tifi<sub>ca</sub>ti<sub>on</sub> <sub>an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>samp</sub>li<sub>ng</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> C<sub>-</sub>S<sub>quare</sub>CB (Deb et al., 2025a). The concentration bounds and <sub>p</sub>roof of Theorem 4.1 are <sub>g</sub>iven in the subsequent <sub>su</sub>b<sub>sec</sub>ti<sub>ons.</sub>

The quantity controlled by the oracle changes. C-SquareCB observes noisy cardinal costs and <sub>uses a square-</sub>l<sub>oss orac</sub>l<sub>e</sub> t<sub>o ca</sub>lib<sub>ra</sub>t<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve pre</sub>di<sub>c</sub>ti<sub>ons o</sub>f th<sub>ose cos</sub>t<sub>s;</sub> it<sub>s sa</sub>f<sub>e</sub>t<sub>y</sub> t<sub>es</sub>t <sub>a</sub>dd<sub>s a</sub>

<sub>correc</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub>i<sub>r cumu</sub>l<sub>a</sub>ti<sub>ve es</sub>ti<sub>ma</sub>ti<sub>on error.</sub> I<sub>n our se</sub>tti<sub>ng,</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>mean o</sub>f th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l i<sub>s</sub> $\sigma ( f ( \mathbf { x } _ { t } ) - f ( \mathbf { y } _ { t } ) )$ <sub>.</sub> A<sub>ccor</sub>di<sub>ng</sub>l<sub>y,</sub> th<sub>e</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c orac</sub>l<sub>e con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>e marg</sub>i<sub>n error</sub>

$$
\varepsilon _ { t } ( \mathbf { x } _ { t } , \mathbf { y } _ { t } ) = \left[ \hat { f } _ { t } ( \mathbf { x } _ { t } ) - \hat { f } _ { t } ( \mathbf { y } _ { t } ) \right] - \left[ f ( \mathbf { x } _ { t } ) - f ( \mathbf { y } _ { t } ) \right] ,\tag{66}
$$

<sub>as</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>3<sub>.</sub> Thi<sub>s</sub> d<sub>oes</sub> <sub>no</sub>t <sub>con</sub>t<sub>ro</sub>l i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>u</sub>tilit<sub>y</sub> <sub>errors:</sub> <sub>a</sub>ddi<sub>ng</sub> th<sub>e</sub> <sub>same</sub> constant to ever<sub>y p</sub>rediction leaves both the lo<sub>g</sub>istic loss and (66) unchan<sub>g</sub>ed. Thus C-SquareCB’s <sub>car</sub>di<sub>na</sub>l <sub>ca</sub>lib<sub>ra</sub>ti<sub>on canno</sub>t b<sub>e</sub> i<sub>nvo</sub>k<sub>e</sub>d <sub>w</sub>ith th<sub>ese pre</sub>di<sub>c</sub>ti<sub>ons.</sub> S<sub>cores</sub> th<sub>emse</sub>l<sub>ves are s</sub>hift i<sub>nvar</sub>i<sub>an</sub>t<sub>,</sub> h<sub>owever, so</sub> thi<sub>s o</sub>b<sub>serva</sub>ti<sub>on</sub> i<sub>s no</sub>t <sub>an</sub> i<sub>mposs</sub>ibilit<sub>y o</sub>f <sub>score cer</sub>tifi<sub>ca</sub>ti<sub>on.</sub>

Pairwise fit alone does not calibrate plug-in scores. A separate example shows why simply <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>scores</sub> i<sub>n</sub>t<sub>o</sub> <sub>a</sub> <sub>car</sub>di<sub>na</sub>l t<sub>es</sub>t <sub>a</sub>l<sub>so</sub> <sub>requ</sub>i<sub>res</sub> <sub>an</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>argumen</sub>t<sub>.</sub> F<sub>or</sub> <sub>any</sub> <sub>u</sub>tilit<sub>y</sub> f<sub>unc</sub>ti<sub>on</sub> $^ { g , }$ , define its <sub>p</sub>lu<sub>g</sub>-in score on a context set S b<sub>y</sub>

$$
s ^ { g } ( \mathbf { x } ) : = \sigma \bigg ( g ( \mathbf { x } ) - \underset { \mathbf { y } \in S } { \operatorname* { m a x } } g ( \mathbf { y } ) \bigg ) .
$$

C<sub>ons</sub>id<sub>er a rea</sub>li<sub>ze</sub>d <sub>con</sub>t<sub>ex</sub>t <sub>se</sub>t <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> f<sub>our</sub> di<sub>s</sub>ti<sub>nc</sub>t it<sub>ems</sub> $\boldsymbol { S } = \{ \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { ( 2 ) } , \mathbf { x } _ { b } , \mathbf { x } ^ { \star } \}$ <sub>an</sub>d t<sub>wo</sub> <sub>u</sub>tilit<sub>y</sub> f<sub>unc</sub>ti<sub>ons w</sub>ith <sub>va</sub>l<sub>ues</sub>

$$
\frac { \mathbf { x } ^ { ( 1 ) } \quad \mathbf { x } ^ { ( 2 ) } \quad \mathbf { x } _ { b } ^ { \star } } { f } \frac { 0 } { \log 2 \quad \log 3 }
$$

B<sub>o</sub>th h<sub>ave u</sub>tilit<sub>y range a</sub>t <sub>mos</sub>t $B = \log 3$ <sub>,</sub> th<sub>e same</sub> b<sub>es</sub>t it<sub>em, an</sub>d th<sub>e same</sub> b<sub>ase</sub>li<sub>ne score</sub> $s ^ { f } ( { \bf x } _ { b } ) = s ^ { g } ( { \bf x } _ { b } ) = 2 / 5$ <sub>.</sub> O<sub>n</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>r</sub> $( \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { ( 2 ) } )$ <sub>,</sub> b<sub>o</sub>th <sub>marg</sub>i<sub>ns are zero.</sub> H<sub>ence</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>an</sub>d t<sub>rue compar</sub>i<sub>son pro</sub>b<sub>a</sub>biliti<sub>es co</sub>i<sub>nc</sub>id<sub>e,</sub> th<sub>e marg</sub>i<sub>n error van</sub>i<sub>s</sub>h<sub>es, an</sub>d th<sub>e</sub> t<sub>wo</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> l<sub>osses agree</sub> f<sub>or e</sub>ith<sub>er ou</sub>t<sub>come.</sub> N<sub>ever</sub>th<sub>e</sub>l<sub>ess,</sub>

$$
\operatorname* { m a x } _ { j \in \{ 1 , 2 \} } s ^ { f } ( \mathbf { x } ^ { ( j ) } ) = \frac { 1 } { 4 } , \qquad \operatorname* { m a x } _ { j \in \{ 1 , 2 \} } s ^ { g } ( \mathbf { x } ^ { ( j ) } ) = \frac { 2 } { 5 } .\tag{67}
$$

F<sub>or</sub> $\alpha = 1 / 4$ <sub>,</sub> th<sub>e</sub> <sub>requ</sub>i<sub>re</sub>d b<sub>ase</sub>li<sub>ne</sub> f<sub>rac</sub>ti<sub>on</sub> i<sub>s</sub> $3 / 1 0 .$ <sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>p</sub>l<sub>ug-</sub>i<sub>n</sub> <sub>score</sub> <sub>excee</sub>d<sub>s</sub> thi<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>w</sub>hil<sub>e</sub> the true pair score falls below it. Repeating this pair n times leaves the observed margin errors zero b<sub>u</sub>t <sub>g</sub>i<sub>ves a cumu</sub>l<sub>a</sub>ti<sub>ve score overes</sub>ti<sub>ma</sub>t<sub>e o</sub>f $3 n / 2 0$ <sub>.</sub> Thi<sub>s</sub> i<sub>s an a</sub>l<sub>ge</sub>b<sub>ra</sub>i<sub>c</sub> f<sub>a</sub>il<sub>ure o</sub>f <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> f<sub>rom</sub> <sub>o</sub>b<sub>serve</sub>d <sub>marg</sub>i<sub>n errors a</sub>l<sub>one, con</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>pa</sub>i<sub>rs;</sub> it i<sub>s no</sub>t <sub>a regre</sub>t l<sub>ower</sub> b<sub>oun</sub>d f<sub>or</sub> <sub>an</sub> IGW <sub>a</sub>l<sub>gor</sub>ith<sub>m or a c</sub>l<sub>a</sub>i<sub>m a</sub>b<sub>ou</sub>t <sub>every poss</sub>ibl<sub>e compar</sub>i<sub>son.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>, compar</sub>i<sub>ng</sub> $\mathbf { x } ^ { ( 1 ) }$ a<sub>g</sub>a<sup>i</sup>nst th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> th<sub>e mo</sub>d<sub>e</sub>l<sub>s:</sub> it<sub>s w</sub>i<sub>n pro</sub>b<sub>a</sub>biliti<sub>es are</sub> $1 / 3$ <sub>un</sub>d<sub>er</sub> $f$ <sub>an</sub>d $1 / 2$ <sub>un</sub>d<sub>er</sub> $g .$ Th<sub>e</sub> <sub>examp</sub>l<sub>e s</sub>h<sub>ows w</sub>h<sub>y a score guaran</sub>t<sub>ee mus</sub>t <sub>a</sub>l<sub>so accoun</sub>t f<sub>or w</sub>hi<sub>c</sub>h <sub>compar</sub>i<sub>sons</sub> th<sub>e</sub> l<sub>earner co</sub>ll<sub>ec</sub>t<sub>s.</sub>

The replacement couples certification to the sampling rule. Algorithm 2 uses $p _ { t } \otimes p _ { t }$ <sub>on can</sub>did<sub>a</sub>t<sub>e</sub> <sub>roun</sub>d<sub>s an</sub>d <sub>a</sub> b<sub>ase</sub>li<sub>ne pa</sub>i<sub>re</sub>d <sub>w</sub>ith <sub>one</sub> d<sub>raw</sub> f<sub>rom</sub> $p _ { t }$ <sub>on</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s.</sub> Th<sub>e</sub> b<sub>ranc</sub>h i<sub>s se</sub>l<sub>ec</sub>t<sub>e</sub>d b<sub>e</sub>f<sub>ore</sub> <sub>samp</sub>li<sub>ng, so</sub> th<sub>ese are</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>samp</sub>li<sub>ng</sub> l<sub>aws.</sub> L<sub>e</sub>t $\mathbb { E } _ { t }$ <sub>con</sub>diti<sub>on on a</sub>ll i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>e</sub>f<sub>ore</sub> <sub>pa</sub>i<sub>r</sub> <sub>samp</sub>li<sub>ng,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> b<sub>ranc</sub>h d<sub>ec</sub>i<sub>s</sub>i<sub>on,</sub> <sub>an</sub>d <sub>pu</sub>t $v _ { t } : = \mathbb { E } _ { t } [ \varepsilon _ { t } ( { \bf x } _ { t } , { \bf y } _ { t } ) ^ { 2 } ]$ <sub>.</sub> F<sub>or</sub> $r _ { t } ^ { \circ }$ defined in (80), L<sub>emma</sub> 4<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> <sub>on</sub> b<sub>o</sub>th b<sub>ranc</sub>h<sub>es</sub>

$$
\mathbb { E } _ { t } [ r _ { t } ^ { \circ } ] \leq \frac { 2 K } { \gamma _ { t } } + \frac { \gamma _ { t } } { 4 } v _ { t } .\tag{68}
$$

O<sub>n can</sub>did<sub>a</sub>t<sub>es,</sub> th<sub>e</sub> fi<sub>xe</sub>d<sub>-par</sub>t<sub>ner</sub> i<sub>nequa</sub>lit<sub>y</sub> i<sub>s average</sub>d <sub>over an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t IGW <sub>par</sub>t<sub>ner; on</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>s,</sub> th<sub>e par</sub>t<sub>ner</sub> i<sub>s</sub> fi<sub>xe</sub>d t<sub>o</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub> I<sub>n</sub> b<sub>o</sub>th <sub>cases</sub> it<sub>s error</sub> t<sub>erm</sub> i<sub>s</sub> th<sub>a</sub>t <sub>o</sub>f th<sub>e compar</sub>i<sub>son ac</sub>t<sub>ua</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serve</sub>d<sub>.</sub> Th<sub>e</sub> <sub>cen</sub>t<sub>er</sub>i<sub>ng</sub> <sub>argumen</sub>t <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>s</sub>t<sub>ep</sub> <sub>appears</sub> i<sub>n</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f L<sub>emma</sub> 12 <sub>o</sub>f S<sub>e</sub>kh<sub>ar</sub>i et al. (2023); A<sub>pp</sub>endix B.2 verifies the constants used here. Our construction a<sub>pp</sub>lies it to both <sub>conserva</sub>ti<sub>ve</sub> b<sub>ranc</sub>h<sub>es</sub> <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>e</sub> <sub>orac</sub>l<sub>e</sub> <sub>w</sub>ith b<sub>o</sub>th t<sub>ypes</sub> <sub>o</sub>f <sub>compar</sub>i<sub>son.</sub> A f<sub>a</sub>llb<sub>ac</sub>k <sub>may</sub> <sub>s</sub>till b<sub>e</sub> <sub>a se</sub>lf<sub>-</sub>d<sub>ue</sub>l<sub>;</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> f<sub>ee</sub>db<sub>ac</sub>k i<sub>s no</sub>t <sub>guaran</sub>t<sub>ee</sub>d <sub>on every</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>roun</sub>d<sub>.</sub>

Cumulative regret replaces cardinal calibration. Lemma B.3 and $\mathsf { A p } _ { \mathsf { I } }$ <sub>p</sub>endix B.5 combine (68) with the oracle <sub>g</sub>uarantee to establish the envelo<sub>p</sub>e (85). The score inequalit<sub>y</sub> (90) then <sub>y</sub>ields

$$
\sum _ { i = 1 } ^ { t } \left[ \operatorname* { m a x } \{ s _ { i } ( \mathbf { x } _ { i } ) , s _ { i } ( \mathbf { y } _ { i } ) \} - ( 1 - \alpha ) s _ { i } ^ { b } \right] \geq Q _ { t } - \frac { 1 } { 4 } \sum _ { i \in \mathcal { A } _ { t } } R _ { i } ,\tag{69}
$$

<sub>w</sub>h<sub>ere</sub> $\boldsymbol { A } _ { t }$ <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>s an</sub>d $Q _ { t }$ is the credit total in (89). This ex<sub>p</sub>lains the char<sub>g</sub>e $G _ { \mathrm { I G W } } ( t ) / 4$ <sub>w</sub>h<sub>en cons</sub>id<sub>er</sub>i<sub>ng a can</sub>did<sub>a</sub>t<sub>e.</sub> I<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne ma</sub>k<sub>es eac</sub>h f<sub>a</sub>llb<sub>ac</sub>k <sub>sa</sub>f<sub>e</sub> di<sub>rec</sub>tl<sub>y,</sub> while (68) also controls its sam<sub>p</sub>led <sub>p</sub>artner’s <sub>g</sub>a<sub>p</sub>. A<sub>pp</sub>endix B.6 <sub>p</sub>roves <sub>p</sub>refix safet<sub>y</sub>, bounds the f<sub>a</sub>llb<sub>ac</sub>k <sub>coun</sub>t<sub>, an</sub>d d<sub>er</sub>i<sub>ves</sub> th<sub>e regre</sub>t <sub>guaran</sub>t<sub>ee.</sub> Th<sub>ese are c</sub>h<sub>anges</sub> t<sub>o</sub> th<sub>e cer</sub>tifi<sub>ca</sub>t<sub>e an</sub>d it<sub>s connec</sub>ti<sub>on</sub> t<sub>o o</sub>b<sub>serve</sub>d f<sub>ee</sub>db<sub>ac</sub>k<sub>;</sub> th<sub>ey</sub> d<sub>o no</sub>t <sub>asser</sub>t th<sub>a</sub>t <sub>a regre</sub>t <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> i<sub>s</sub> th<sub>e on</sub>l<sub>y poss</sub>ibl<sub>e way</sub> t<sub>o en</sub>f<sub>orce</sub> th<sub>e score cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub>

## B.2. The fixed-partner IGW inequality

W<sub>e</sub> fi<sub>rs</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>an</sub> <sub>e</sub>l<sub>emen</sub>t<sub>ary</sub> IGW i<sub>nequa</sub>lit<sub>y,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s</sub> <sub>cons</sub>t<sub>an</sub>t<sub>s.</sub>

Lemma B.1 (Single-action IGW inequality) Let $v , \hat { v } \in \mathbb { R } ^ { K } , a ^ { \star } \in \arg$ ma $\mathfrak { c } _ { a } v _ { a } , \hat { a } \in$ arg ma $\mathrm { x } _ { a } \hat { v } _ { a } ,$ <sub>an</sub>d $\gamma > 0$ . Define $p _ { a } = ( K + \gamma ( \hat { v } _ { \hat { a } } - \hat { v } _ { a } ) ) ^ { - 1 }$ <sup>1</sup> for a $\neq \hat { a }$ <sub>an</sub>d <sub>pu</sub>t th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>mass</sub> <sub>on</sub> ${ \hat { a } } .$ Th<sub>en</sub> $p$ i<sub>s</sub> <sub>a pro</sub>b<sub>a</sub>bilit<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> $p _ { \hat { a } } \geq 1 / K$ <sub>,</sub> <sub>an</sub>d

$$
\sum _ { a } p _ { a } ( v _ { a ^ { \star } } - v _ { a } ) \leq \frac { 2 K } { \gamma } + \frac { \gamma } { 4 } \sum _ { a } p _ { a } ( \hat { v } _ { a } - v _ { a } ) ^ { 2 } .\tag{70}
$$

Proof Every action other than aˆ has probability in $( 0 , 1 / K ]$ <sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>res</sub>id<sub>ua</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $1 / K$ <sub>.</sub> L<sub>e</sub>t $e _ { a } = \hat { v } _ { a } - v _ { a } , u _ { a } = \hat { v } _ { \hat { a } } - \hat { v } _ { a } \geq 0 , \eta = \gamma / 4$ <sub>,</sub> <sub>an</sub>d $p _ { \star } = p _ { a ^ { \star } } > 0$ <sub>.</sub> S<sub>u</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>qua</sub>d<sub>ra</sub>ti<sub>c</sub> term in (70), and usin<sub>g</sub> $e - \eta e ^ { 2 } \le 1 / ( 4 \eta )$ f<sub>or ac</sub>ti<sub>ons o</sub>th<sub>er</sub> th<sub>an</sub> $a ^ { \star }$ , <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r l } & { D : = \displaystyle \sum _ { a } p _ { a } \big ( v _ { a ^ { \star } } - v _ { a } \big ) - \eta \displaystyle \sum _ { a } p _ { a } e _ { a } ^ { 2 } } \\ & { \quad \le \displaystyle \sum _ { a \ne a ^ { \star } } p _ { a } \big ( \hat { v } _ { a ^ { \star } } - \hat { v } _ { a } \big ) - ( 1 - p _ { \star } ) e _ { a ^ { \star } } - \eta p _ { \star } e _ { a ^ { \star } } ^ { 2 } + \frac { 1 - p _ { \star } } { 4 \eta } . } \end{array}\tag{71}
$$

C<sub>omp</sub>l<sub>e</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>square</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> $- ( 1 - p _ { \star } ) e - \eta p _ { \star } e ^ { 2 } \le ( 1 - p _ { \star } ) ^ { 2 } / ( 4 \eta p _ { \star } )$ <sub>.</sub> H<sub>ence</sub>

$$
D \leq \sum _ { a } p _ { a } u _ { a } - u _ { a ^ { \star } } + \frac { 1 - p _ { \star } } { \gamma p _ { \star } } \leq \sum _ { a } p _ { a } u _ { a } - u _ { a ^ { \star } } + \frac { 1 } { \gamma p _ { \star } } .
$$

Al<sub>so,</sub>

$$
\sum _ { a } p _ { a } u _ { a } = \sum _ { a \ne \hat { a } } \frac { u _ { a } } { K + \gamma u _ { a } } \le \frac { K - 1 } { \gamma } .
$$

If $\boldsymbol { a } ^ { \star } = \boldsymbol { \hat { a } }$ <sub>,</sub> th<sub>en</sub> $u _ { a ^ { \star } } = 0$ <sub>an</sub>d $p _ { \star } \geq 1 / K$ <sub>.</sub> Oth<sub>erw</sub>i<sub>se,</sub> $p _ { \star } ^ { - 1 } = K + \gamma u _ { a ^ { \star } }$ <sub>.</sub> I<sub>n</sub> <sub>e</sub>ith<sub>er</sub> <sub>case,</sub> $- u _ { a ^ { \star } } +$ $1 / ( \gamma p _ { \star } ) \leq K / \gamma$ <sub>.</sub> Th<sub>us</sub> $D \le ( 2 K - 1 ) / \gamma \le 2 K / \gamma$ <sub>, prov</sub>i<sub>ng</sub> th<sub>e c</sub>l<sub>a</sub>i<sub>m.</sub> Th<sub>e argumen</sub>t <sub>requ</sub>i<sub>res no</sub> restriction on the absolute levels of v or vˆ. ■

Proof [Proof of Lemma 4.1] Fix $\mathbf { u _ { \theta } } \in \mathcal S _ { t }$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub> L<sub>emma</sub> B<sub>.</sub>1 <sub>w</sub>ith $v _ { a } = f ( \mathbf { x } _ { t , a } ) - f ( \mathbf { u } )$ <sub>an</sub>d $\hat { v } _ { a } = \hat { f } _ { t } ( \mathbf { x } _ { t , a } ) - \hat { f } _ { t } ( \mathbf { u } )$ <sub>.</sub> S<sub>u</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>par</sub>t<sub>ner</sub>’<sub>s va</sub>l<sub>ue</sub> l<sub>eaves</sub> b<sub>o</sub>th th<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>max</sub>i<sub>m</sub>i<sub>zer</sub> <sub>an</sub>d <sub>every</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>gap</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> $p _ { t }$ <sub>.</sub> Th<sub>e</sub> t<sub>rue</sub> <sub>regre</sub>t i<sub>s</sub> $v _ { a ^ { \star } } - v _ { a } = \Delta _ { t } ( \mathbf { x } _ { t , a } )$ <sub>,</sub> <sub>an</sub>d $\hat { v } _ { a } - v _ { a } = \varepsilon _ { t } ( \mathbf { x } _ { t , a } , \mathbf { u } )$ . Equation (70) <sub>g</sub>ives the bound in the lemma.

## B.3. From logistic regret to predictable margin error

Fix an epoch of length n and failure budget $\delta ^ { \prime } \in ( 0 , 1 )$ <sub>, an</sub>d <sub>re</sub>i<sub>n</sub>d<sub>ex</sub> it<sub>s roun</sub>d<sub>s</sub> b<sub>y</sub> $m = 1 , \ldots , n$ <sub>.</sub> Th<sub>e</sub> <sub>orac</sub>l<sub>e</sub> i<sub>s</sub> f<sub>res</sub>h <sub>a</sub>t th<sub>e s</sub>t<sub>ar</sub>t <sub>o</sub>f thi<sub>s epoc</sub>h<sub>.</sub> All <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> b<sub>oun</sub>d<sub>s</sub> b<sub>e</sub>l<sub>ow a</sub>l<sub>so</sub> h<sub>o</sub>ld <sub>con</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e epoc</sub>h<sub>.</sub> L<sub>e</sub>t $\mathbb { E } _ { m }$ d<sub>eno</sub>t<sub>e con</sub>diti<sub>ona</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on g</sub>i<sub>ven a</sub>ll i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> b<sub>e</sub>f<sub>ore</sub> <sub>samp</sub>li<sub>ng</sub> th<sub>e</sub> <sub>curren</sub>t <sub>pa</sub>i<sub>r,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> <sub>curren</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>,</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> <sub>orac</sub>l<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub>/f<sub>a</sub>llb<sub>ac</sub>k d<sub>ec</sub>i<sub>s</sub>i<sub>on.</sub> Th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l l<sub>aw, a</sub>ft<sub>er con</sub>diti<sub>on</sub>i<sub>ng a</sub>dditi<sub>ona</sub>ll<sub>y on</sub> th<sub>e samp</sub>l<sub>e</sub>d <sub>pa</sub>i<sub>r,</sub> i<sub>s</sub> BTL<sub>.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
\Lambda = \log \frac { 3 n } { \delta ^ { \prime } } , \qquad E = \frac { 8 \mathcal { R } _ { \log } ( n ) } { \kappa } + \frac { 5 \Lambda } { \kappa ^ { 2 } } ,
$$

$$
d _ { m } = f ( \mathbf { x } _ { m } ) - f ( \mathbf { y } _ { m } ) , \hat { d } _ { m } = \hat { f } _ { m } ( \mathbf { x } _ { m } ) - \hat { f } _ { m } ( \mathbf { y } _ { m } ) ,\tag{72}
$$

$$
e _ { m } = \hat { d } _ { m } - d _ { m } , \qquad X _ { m } = e _ { m } ^ { 2 } , \qquad v _ { m } = \mathbb { E } _ { m } [ X _ { m } ] .\tag{73}
$$

Th<sub>e</sub> t<sub>rue</sub> <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>marg</sub>i<sub>ns</sub> li<sub>e</sub> i<sub>n</sub> $[ - B , B ]$ , so $0 \leq X _ { m } \leq 4 B ^ { 2 }$ <sub>.</sub> A <sub>use</sub>f<sub>u</sub>l <sub>e</sub>l<sub>emen</sub>t<sub>ary</sub> b<sub>oun</sub>d i<sub>s</sub>

$$
\kappa ^ { - 1 } = 2 + 2 \cosh B \geq 4 + B ^ { 2 } \geq 4 B , \qquad B \kappa \leq \frac { 1 } { 4 } .\tag{74}
$$

Lemma B.2 (Uniform predictable error bound) For any predictable comparison-sampling rule, <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - 2 \delta ^ { \prime } / 3 ,$ , simultaneouslyfor every $j \leq n ,$

$$
\sum _ { m = 1 } ^ { j } v _ { m } \leq E .\tag{75}
$$

Proof Let $\Phi ( d ) = \log ( 1 + e ^ { d } ) , \pi _ { m } = \sigma ( d _ { m } )$ <sub>,</sub> <sub>an</sub>d $\xi _ { m } = e _ { m } ( o _ { m } - \pi _ { m } )$ <sub>.</sub> Th<sub>e</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> l<sub>oss</sub> dif<sub>erence</sub> <sup>d</sup>ecom<sub>p</sub>oses as

$$
\ell _ { m } ( \hat { f } _ { m } ) - \ell _ { m } ( f ) = D _ { m } - \xi _ { m } , \qquad D _ { m } : = \Phi ( \hat { d } _ { m } ) - \Phi ( d _ { m } ) - \Phi ^ { \prime } ( d _ { m } ) e _ { m } .\tag{76}
$$

Si<sub>nce</sub> th<sub>e</sub> <sub>segmen</sub>t b<sub>e</sub>t<sub>ween</sub> $d _ { m }$ <sub>an</sub>d $\hat { d } _ { m }$ li<sub>es</sub> i<sub>n</sub> $[ - B , B ]$ <sub>an</sub>d $\Phi ^ { \prime \prime } = \sigma ^ { \prime } \ge \kappa$ th<sub>ere,</sub>

$$
D _ { m } = e _ { m } ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - u ) \Phi ^ { \prime \prime } ( d _ { m } + u e _ { m } ) d u \geq \frac { \kappa } { 2 } X _ { m } .
$$

R<sub>ea</sub>li<sub>za</sub>bilit<sub>y</sub> <sub>an</sub>d th<sub>e</sub> <sub>pre</sub>fi<sub>x</sub> <sub>orac</sub>l<sub>e</sub> <sub>guaran</sub>t<sub>ee</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>mp</sub>l<sub>y,</sub> f<sub>or</sub> <sub>every</sub> $j \leq n ,$

$$
\frac { \kappa } { 2 } \sum _ { m = 1 } ^ { j } X _ { m } \leq \mathcal { R } _ { \log } ( n ) + \sum _ { m = 1 } ^ { j } \xi _ { m } .\tag{77}
$$

L<sub>a</sub>b<sub>e</sub>l <sub>no</sub>i<sub>se.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>,</sub> $o _ { m }$ i<sub>s</sub> B<sub>ernou</sub>lli <sub>w</sub>ith <sub>mean</sub> $\pi _ { m } .$ <sub>, an</sub>d th<sub>e</sub> t<sub>wo poss</sub>ibl<sub>e va</sub>l<sub>ues o</sub>f $\xi _ { m }$ li<sub>e</sub> i<sub>n an</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>o</sub>f l<sub>eng</sub>th $| e _ { m } |$ <sub>.</sub> C<sub>on</sub>diti<sub>ona</sub>l H<sub>oe</sub>fdi<sub>ng</sub>’<sub>s</sub> l<sub>emma g</sub>i<sub>ves</sub>

$$
{ \mathbb E } \bigg [ \exp \bigg ( a \xi _ { m } - \frac { a ^ { 2 } X _ { m } } { 8 } \bigg ) \bigg | \operatorname { i n f o r m a t i o n } \mathrm { b e f o r e ~ t h e ~ l a b e l } \bigg ] \le 1 \qquad ( a \in { \mathbb R } ) .
$$

T<sub>a</sub>ki<sub>ng a</sub> f<sub>ur</sub>th<sub>er con</sub>diti<sub>ona</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>f<sub>ore pa</sub>i<sub>r samp</sub>li<sub>ng s</sub>h<sub>ows</sub> th<sub>a</sub>t $\begin{array} { r } { \exp ( a \sum _ { m \leq j } \xi _ { m } - } \end{array}$ $a ^ { 2 } \sum _ { m \leq j } X _ { m } / 8 )$ i<sub>s</sub> <sub>a</sub> <sub>nonnega</sub>ti<sub>ve</sub> <sub>supermar</sub>ti<sub>nga</sub>l<sub>e</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> <sub>a</sub>t <sub>one.</sub> Vill<sub>e</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>w</sub>ith $a = 2 r$ i<sub>mp</sub>li<sub>es, ou</sub>t<sub>s</sub>id<sub>e an even</sub>t <sub>o</sub>f <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t <sub>mos</sub>t $e ^ { - \Lambda }$ <sub>, s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>or every</sub> $j \leq n ,$

$$
\sum _ { m = 1 } ^ { j } \xi _ { m } \le \frac \kappa 4 \sum _ { m = 1 } ^ { j } X _ { m } + \frac \Lambda { 2 \kappa } .
$$

Combinin<sub>g</sub> this with (77) <sub>g</sub>ives

$$
\sum _ { m = 1 } ^ { j } X _ { m } \leq \frac { 4 \mathcal { R } _ { \log } ( n ) } { \kappa } + \frac { 2 \Lambda } { \kappa ^ { 2 } } .\tag{78}
$$

P<sub>a</sub>i<sub>r samp</sub>li<sub>ng.</sub> P<sub>u</sub>t $c = ( 1 - e ^ { - 1 } ) / ( 4 B ^ { 2 } )$ <sub>.</sub> C<sub>onvex</sub>it<sub>y</sub> <sub>o</sub>f $x \mapsto e ^ { - x / ( 4 B ^ { 2 } ) }$ i<sub>mp</sub>li<sub>es</sub> th<sub>e c</sub>h<sub>or</sub>d b<sub>oun</sub>d $e ^ { - x / ( 4 B ^ { 2 } ) } \leq \mathrm { { i } } - c x$ on [0, 4B<sup>2</sup>]. Thus

$$
\begin{array} { r } { \mathbb { E } _ { m } [ e ^ { - X _ { m } / ( 4 B ^ { 2 } ) } ] \le 1 - c v _ { m } \le e ^ { - c v _ { m } } . } \end{array}
$$

It follows that exp $\textstyle ( c \sum _ { m \leq j } v _ { m } - \sum _ { m \leq j } X _ { m } / ( 4 B ^ { 2 } ) )$ i<sub>s ano</sub>th<sub>er nonnega</sub>ti<sub>ve supermar</sub>ti<sub>nga</sub>l<sub>e s</sub>t<sub>ar</sub>ti<sub>ng</sub> <sub>a</sub>t <sub>one.</sub> A <sub>secon</sub>d <sub>app</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>o</sub>f Vill<sub>e</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves,</sub> <sub>w</sub>ith <sub>a</sub>dditi<sub>ona</sub>l f<sub>a</sub>il<sub>ure</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t <sub>mos</sub>t $e ^ { - \Lambda }$

$$
\begin{array} { c l l } { \displaystyle \sum _ { m = 1 } ^ { j } v _ { m } \leq \frac { 1 } { 1 - e ^ { - 1 } } \sum _ { m = 1 } ^ { j } X _ { m } + \frac { 4 B ^ { 2 } \Lambda } { 1 - e ^ { - 1 } } } \\ { \leq 2 \displaystyle \sum _ { m = 1 } ^ { j } X _ { m } + 7 B ^ { 2 } \Lambda . } \end{array}\tag{79}
$$

On the intersection of the two events, (78), (79), and (74) <sub>y</sub>ield

$$
\sum _ { m = 1 } ^ { j } v _ { m } \le \frac { 8 \mathcal { R } _ { \mathrm { l o g } } ( n ) } { \kappa } + \frac { 4 \Lambda } { \kappa ^ { 2 } } + 7 B ^ { 2 } \Lambda \le \frac { 8 \mathcal { R } _ { \mathrm { l o g } } ( n ) } { \kappa } + \frac { 5 \Lambda } { \kappa ^ { 2 } } = E .
$$

The total failure probability is at most 2e $^ { - \Lambda } = 2 \delta ^ { \prime } / ( 3 n ) \leq 2 \delta ^ { \prime } / 3$ <sub>.</sub> N<sub>o</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> <sub>c</sub>h<sub>osen pa</sub>i<sub>rs across roun</sub>d<sub>s</sub> i<sub>s requ</sub>i<sub>re</sub>d<sub>.</sub> ■

## B.4. An epoch envelope including fallback comparisons

At <sub>eac</sub>h <sub>roun</sub>d<sub>,</sub> th<sub>e</sub> b<sub>ranc</sub>h i<sub>s</sub> <sub>c</sub>h<sub>osen</sub> b<sub>e</sub>f<sub>ore</sub> <sub>samp</sub>li<sub>ng.</sub> A <sub>can</sub>did<sub>a</sub>t<sub>e</sub> di<sub>sp</sub>l<sub>ays</sub> t<sub>wo</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t d<sub>raws</sub> f<sub>rom</sub> $p _ { m } ;$ <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k di<sub>sp</sub>l<sub>ays</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne an</sub>d <sub>one</sub> d<sub>raw</sub> f<sub>rom</sub> $p _ { m }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
r _ { m } ^ { \circ } = \left\{ \begin{array} { l l } { R _ { m } , } & { \mathrm { c a n d i d a t e } , } \\ { f ( \mathbf { x } _ { m } ^ { \star } ) - f ( \mathbf { y } _ { m } ) , } & { \mathrm { f a l l b a c k } . } \end{array} \right.\tag{80}
$$

Th<sub>us</sub> $0 \leq r _ { m } ^ { \circ } \leq B$ <sub>.</sub> Thi<sub>s</sub> <sub>quan</sub>tit<sub>y</sub> i<sub>s</sub> <sub>use</sub>d <sub>on</sub>l<sub>y</sub> i<sub>n</sub> th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub>

Lemma B.3 (Uniform epoch envelope) With E, Λ from (72), use $\gamma _ { m } = \sqrt { 8 K m / E }$ and define

$$
C ( n , \delta ^ { \prime } ) : = \frac { 3 } { \sqrt { 2 } } \sqrt { K E } + B \sqrt { 2 \Lambda } .\tag{81}
$$

For any predictable sequence of candidate/fallback decisions, with probability at least $1 - \delta ^ { \prime }$ simultaneously for every $j \leq n ,$

$$
\sum _ { m = 1 } ^ { j } r _ { m } ^ { \circ } \leq C ( n , \delta ^ { \prime } ) \sqrt { j } .\tag{82}
$$

Proof On a candidate round, independence and identical distribution of the two draws give $\mathbb E _ { m } [ R _ { m } ] = \mathbb E _ { a \sim p _ { m } } \Delta _ { m } ( \mathbf x _ { m , a } )$ <sub>.</sub> A<sub>verage</sub> L<sub>emma</sub> 4<sub>.</sub>1 <sub>over an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>par</sub>t<sub>ner</sub> f<sub>rom</sub> $p _ { m }$ t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathbb { E } _ { m } [ r _ { m } ^ { \circ } ] \leq \frac { 2 K } { \gamma _ { m } } + \frac { \gamma _ { m } } { 4 } v _ { m } .
$$

O<sub>n</sub> <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>,</sub> <sub>app</sub>l<sub>y</sub> th<sub>e</sub> <sub>same</sub> l<sub>emma</sub> <sub>w</sub>ith th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>as</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>par</sub>t<sub>ner.</sub> Th<sub>e</sub> <sub>square</sub>d <sub>error</sub> <sub>o</sub>f th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>compar</sub>i<sub>son</sub> <sub>equa</sub>l<sub>s</sub> th<sub>e</sub> <sub>square</sub>d <sub>error</sub> <sub>w</sub>ith th<sub>e</sub> <sub>pa</sub>i<sub>r</sub> <sub>reverse</sub>d<sub>,</sub> <sub>so</sub> thi<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>aga</sub>i<sub>n</sub> h<sub>o</sub>ld<sub>s, now w</sub>ith $r _ { m } ^ { \circ } = \Delta _ { m } (  { \mathbf { y } } _ { m } )$ <sub>.</sub> Th<sub>e error</sub> $v _ { m }$ i<sub>s a</sub>l<sub>ways</sub> th<sub>a</sub>t <sub>o</sub>f th<sub>e compar</sub>i<sub>son ac</sub>t<sub>ua</sub>ll<sub>y</sub> di<sub>sp</sub>l<sub>aye</sub>d<sub>.</sub>

O<sub>n</sub> th<sub>e</sub> <sub>even</sub>t <sub>o</sub>f L<sub>emma</sub> B<sub>.</sub>2<sub>,</sub> <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y</sub> <sub>o</sub>f $\gamma _ { m }$ <sub>an</sub>d $\begin{array} { r } { \sum _ { m = 1 } ^ { j } m ^ { - 1 / 2 } \leq 2 \sqrt { j } } \end{array}$ <sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub>

$$
\begin{array} { r l r } {  { \sum _ { m = 1 } ^ { j } \mathbb { E } _ { m } [ r _ { m } ^ { \circ } ] \leq 2 K \sum _ { m = 1 } ^ { j } \frac { 1 } { \gamma _ { m } } + \frac { \gamma _ { j } } { 4 } \sum _ { m = 1 } ^ { j } v _ { m } } }  \\ & { } & { \leq \sqrt { 2 K j E } + \frac { 1 } { \sqrt { 2 } } \sqrt { K j E } = \frac { 3 } { \sqrt { 2 } } \sqrt { K j E } . } \end{array}\tag{83}
$$

Th<sub>e</sub> dif<sub>erences</sub> $r _ { m } ^ { \circ } - \mathbb { E } _ { m } [ r _ { m } ^ { \circ } ]$ h<sub>ave con</sub>diti<sub>ona</sub>l <sub>mean zero an</sub>d <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e va</sub>l<sub>ue a</sub>t <sub>mos</sub>t $B .$ <sub>.</sub> F<sub>or eac</sub>h $j ,$ A<sub>zuma</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> b<sub>oun</sub>d<sub>s</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t th<sub>e</sub>i<sub>r sum excee</sub>d<sub>s</sub> $B \sqrt { 2 j \Lambda }$ <sup>b</sup><sub>y</sub> $e ^ { - \Lambda }$ <sub>.</sub> A <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d over $j \leq n$ <sub>g</sub>i<sub>ves a</sub>dditi<sub>ona</sub>l f<sub>a</sub>il<sub>ure pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t <sub>mos</sub>t $n e ^ { - \Lambda } = \delta ^ { \prime } / 3$ <sub>.</sub> C<sub>om</sub>bi<sub>n</sub>i<sub>ng w</sub>ith L<sub>emma</sub> B<sub>.</sub>2 <sub>p</sub>roves (82).

Lemma B.4 (Envelope constants) The constant in (81) satisfies $\begin{array} { r } { C ( n , \delta ^ { \prime } ) \leq \frac { 9 } { 4 } \sqrt { K E } . } \end{array}$

Proof Since $E \ge 5 \Lambda / \kappa ^ { 2 } , K \ge 2$ <sub>,</sub> <sub>an</sub>d $B \kappa \leq 1 / 4$

$$
\frac { B \sqrt { 2 \Lambda } } { \sqrt { K E } } \le B \kappa \sqrt \frac { 2 } { 5 K } \le \frac { 1 } { 4 \sqrt { 5 } } < \frac { 1 } { 8 } .
$$

Use $3 / \sqrt { 2 } + 1 / 8 < 9 / 4$ in (81).

## B.5. Combining epochs and controlling growth

Epoch e comprises rounds $2 ^ { e } , \ldots , 2 ^ { e + 1 } - 1$ <sub>.</sub> L<sub>e</sub>t $\mathcal { E } _ { e }$ b<sub>e</sub> it<sub>s s</sub>i<sub>mu</sub>lt<sub>aneous even</sub>t f<sub>rom</sub> L<sub>emma</sub> B<sub>.</sub>3<sub>, w</sub>ith l<sub>eng</sub>th $n _ { e } = 2 ^ { e }$ <sub>an</sub>d f<sub>a</sub>il<sub>ure</sub> b<sub>u</sub>d<sub>ge</sub>t $\delta _ { e } = \delta / [ ( e + 1 ) ( e + 2 ) ]$ ]. The oracle is restarted, while the credit t<sub>o</sub>t<sub>a</sub>l i<sub>s preserve</sub>d<sub>.</sub> Th<sub>e con</sub>diti<sub>ona</sub>l <sub>guaran</sub>t<sub>ees o</sub>f th<sub>a</sub>t l<sub>emma app</sub>l<sub>y</sub> d<sub>esp</sub>it<sub>e</sub> d<sub>epen</sub>d<sub>ence on a</sub>ll <sub>prev</sub>i<sub>ous</sub> <sub>epoc</sub>h<sub>s.</sub> Si<sub>nce</sub> $\begin{array} { r } { \sum _ { e > 0 } \delta _ { e } = \delta } \end{array}$

$$
\mathbb { P } ( \mathcal { E } _ { \mathrm { I G W } } ) \geq 1 - \delta , \qquad \mathcal { E } _ { \mathrm { I G W } } : = \bigcap _ { e \geq 0 } \mathcal { E } _ { e } .\tag{84}
$$

S<sub>umm</sub>i<sub>ng</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d<sub>-epoc</sub>h b<sub>oun</sub>d<sub>s</sub> <sub>an</sub>d th<sub>e</sub> <sub>curren</sub>t <sub>par</sub>ti<sub>a</sub>l<sub>-epoc</sub>h b<sub>oun</sub>d <sub>g</sub>i<sub>ves,</sub> <sub>on</sub> thi<sub>s</sub> <sub>even</sub>t<sub>,</sub>

$$
\sum _ { i = 1 } ^ { t } r _ { i } ^ { \circ } \leq G _ { \mathrm { I G W } } ( t ) \qquad { \mathrm { f o r ~ e v e r y ~ } } t \geq 1 .\tag{85}
$$

Thi<sub>s</sub> <sub>exp</sub>l<sub>a</sub>i<sub>ns</sub> <sub>w</sub>h<sub>y</sub> f<sub>a</sub>llb<sub>ac</sub>k l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>en</sub>t<sub>er</sub> th<sub>e</sub> <sub>orac</sub>l<sub>e:</sub> th<sub>e</sub>i<sub>r</sub> <sub>errors</sub> <sub>con</sub>t<sub>ro</sub>l th<sub>e</sub>i<sub>r</sub> <sub>par</sub>t<sub>ners</sub>’ <sub>gaps</sub> th<sub>roug</sub>h th<sub>e same</sub> i<sub>nequa</sub>lit<sub>y as on can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>s.</sub>

T<sub>o ma</sub>k<sub>e</sub> th<sub>e grow</sub>th <sub>ra</sub>t<sub>e exp</sub>li<sub>c</sub>it<sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> f<sub>or</sub> $t \geq 1$

$$
\begin{array} { r l } & { \mathcal { L } ( t ) : = \displaystyle \log \frac { 3 t ( 1 + \log _ { 2 } t ) ( 2 + \log _ { 2 } t ) } { \delta } , \qquad \quad \overline { { E } } ( t ) : = \frac { 8 \mathcal { R } _ { \log } ( t ) } { \kappa } + \frac { 5 \mathcal { L } ( t ) } { \kappa ^ { 2 } } , } \\ & { \mathit { c _ { G } } : = \displaystyle \frac { 9 ( 2 + \sqrt { 2 } ) } { 4 } . } \end{array}\tag{86}
$$

B<sub>o</sub>th $E _ { e }$ <sub>an</sub>d $C _ { e }$ are nondecreasing in e. For $e = e ( t )$ <sub>, we</sub> h<sub>ave</sub> $2 ^ { e } \leq t , E _ { e } \leq \overline { { E } } ( t )$ <sub>,</sub> <sub>an</sub>d

$$
\begin{array} { l } { { \displaystyle { G _ { \mathrm { I G W } } ( t ) \leq C _ { e } \left( \sum _ { j < e } \sqrt { 2 ^ { j } } + \sqrt { m ( t ) } \right) } } } \\ { { \leq ( 2 + \sqrt { 2 } ) C _ { e } \sqrt { t } \leq c _ { G } \sqrt { K t \overline { { E } } ( t ) } . } } \end{array}\tag{87}
$$

H<sub>ere</sub> $\textstyle \sum _ { i < e } 2 ^ { j / 2 } \leq ( 1 + { \sqrt { 2 } } ) 2 ^ { e / 2 }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> l<sub>as</sub>t <sub>s</sub>t<sub>ep</sub> <sub>uses</sub> L<sub>emma</sub> B<sub>.</sub>4<sub>.</sub> Th<sub>e</sub> <sub>enve</sub>l<sub>ope</sub> i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>a</sub>t <sub>epoc</sub>h b<sub>oun</sub>d<sub>ar</sub>i<sub>es.</sub> Si<sub>nce</sub> $\mathcal { R } _ { \mathrm { l o g } } ( t ) = o ( t )$ <sub>an</sub>d $\mathcal { L } ( t ) = O ( \log ( t / \delta ) + \log \log ( t + 2 ) )$ ), (87) i<sub>mp</sub>li<sub>es</sub> $G _ { \mathrm { I G W } } ( t ) = o ( t )$

## B.6. Safety, fallback count, and regret

L<sub>e</sub>t $\boldsymbol { A } _ { t }$ <sub>an</sub>d $B _ { t }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>s</sub> th<sub>roug</sub>h $t ,$ <sub>an</sub>d l<sub>e</sub>t $N _ { t } = | B _ { t } |$ <sub>.</sub> F<sub>or</sub> th<sub>e un</sub>if<sub>orm</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>num</sub>b<sub>er o</sub>f f<sub>a</sub>llb<sub>ac</sub>k<sub>s,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
N _ { \mathrm { I G W } } : = \operatorname* { s u p } \big \{ n \in \mathbb { N } _ { \geq 1 } : 4 \alpha \underline { { s } } n < G _ { \mathrm { I G W } } ( n ) \big \} ,\tag{88}
$$

with sup $\emptyset : = 0$ <sub>.</sub> Thi<sub>s quan</sub>tit<sub>y</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e</sub> b<sub>ecause</sub> $G _ { \mathrm { I G W } } ( n ) = o ( n )$ <sub>.</sub> It d<sub>epen</sub>d<sub>s on</sub> th<sub>e pro</sub>bl<sub>em</sub> <sub>parame</sub>t<sub>ers</sub> <sub>an</sub>d th<sub>e</sub> <sub>orac</sub>l<sub>e-regre</sub>t f<sub>unc</sub>ti<sub>on,</sub> b<sub>u</sub>t <sub>no</sub>t <sub>on</sub> th<sub>e</sub> <sub>s</sub>t<sub>opp</sub>i<sub>ng</sub> h<sub>or</sub>i<sub>zon,</sub> <sub>an</sub>d i<sub>s</sub> <sub>no</sub>t <sub>nee</sub>d<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m.</sub> Th<sub>e cre</sub>dit t<sub>o</sub>t<sub>a</sub>l <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
Q _ { t } = \sum _ { i \in \mathcal { A } _ { t } } \left[ \frac { 1 } { 2 } - ( 1 - \alpha ) s _ { i } ^ { b } \right] + \sum _ { i \in \mathcal { B } _ { t } } \alpha s _ { i } ^ { b } .\tag{89}
$$

Proof [Proof of Theorem 4.1] Work on $\mathcal { E } _ { \mathrm { I G W } }$ from (84).

Safety. Put $g _ { t , \mathbf { x } } = f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { x } _ { t } )$ <sub>an</sub>d $g _ { t , \mathbf { y } } = f ( \mathbf { x } _ { t } ^ { \star } ) - f ( \mathbf { y } _ { t } )$ <sub>.</sub> B<sub>ecause</sub> $R _ { t } = ( g _ { t , \mathbf { x } } + g _ { t , \mathbf { y } } ) / 2$

$$
\begin{array} { r } { \operatorname* { m a x } \{ s _ { t } ( \mathbf { x } _ { t } ) , s _ { t } ( \mathbf { y } _ { t } ) \} = \sigma ( - \operatorname* { m i n } \{ g _ { t , \mathbf { x } } , g _ { t , \mathbf { y } } \} ) \geq \sigma ( - R _ { t } ) \geq \frac { 1 } { 2 } - \frac { 1 } { 4 } R _ { t } , } \end{array}\tag{90}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>na</sub>l i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $\sigma ^ { \prime } \leq 1 / 4$ <sub>.</sub> A f<sub>a</sub>llb<sub>ac</sub>k <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne, so</sub> it<sub>s</sub> t<sub>rue surp</sub>l<sub>us</sub> i<sub>s a</sub>t l<sub>eas</sub>t $\alpha s _ { t } ^ { b }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>rue surp</sub>l<sub>us sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\begin{array} { l } { { \displaystyle S _ { t } : = \sum _ { i = 1 } ^ { t } \left[ \operatorname* { m a x } \{ s _ { i } ( { \bf x } _ { i } ) , s _ { i } ( { \bf y } _ { i } ) \} - ( 1 - \alpha ) s _ { i } ^ { b } \right] } } \\ { { \displaystyle ~ \geq Q _ { t } - \frac { 1 } { 4 } \sum _ { i \in \cal A _ { t } } R _ { i } . } } \end{array}\tag{91}
$$

L<sub>e</sub>t $\tau _ { t } =$ max $( { \cal A } _ { t } \cup \{ 0 \} )$ b<sub>e</sub> th<sub>e</sub> l<sub>as</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e</sub> ti<sub>me, us</sub>i<sub>ng zero</sub> if th<sub>ere</sub> h<sub>as</sub> b<sub>een no can</sub>did<sub>a</sub>t<sub>e.</sub> N<sub>onnega</sub>ti<sub>v</sub>it<sub>y</sub> <sub>o</sub>f $r _ { i } ^ { \circ }$ and (85) <sub>g</sub>ive

$$
\sum _ { i \in \mathcal { A } _ { t } } R _ { i } \leq \sum _ { i = 1 } ^ { \tau _ { t } } r _ { i } ^ { \circ } \leq G _ { \mathrm { I G W } } ( \tau _ { t } ) ,
$$

w<sup>h</sup>ere em<sub>p</sub>t<sub>y</sub> sums are zero an<sup>d</sup> $G _ { \mathrm { I G W } } ( 0 ) = 0$ <sub>.</sub> Th<sub>us</sub> $\Lambda _ { t } : = Q _ { t } - G _ { \mathrm { I G W } } ( \tau _ { t } ) / 4$ i<sub>s a</sub> l<sub>ower cer</sub>tifi<sub>ca</sub>t<sub>e</sub> f<sub>or</sub> $S _ { t }$ <sub>.</sub> I<sub>n</sub>iti<sub>a</sub>ll $\Lambda _ { 0 } = 0$ <sub>.</sub> O<sub>n a can</sub>did<sub>a</sub>t<sub>e roun</sub>d<sub>,</sub> $\tau _ { t } = t$ <sub>an</sub>d th<sub>e accep</sub>t<sub>ance</sub> t<sub>es</sub>t <sub>ensures</sub> $\Lambda _ { t } \geq 0$ <sub>.</sub> On <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>roun</sub>d<sub>,</sub> $\tau _ { t } = \tau _ { t - 1 }$ <sub>an</sub>d $\Lambda _ { t } = \Lambda _ { t - 1 } + \alpha s _ { t } ^ { b } \geq 0$ <sub>.</sub> Thi<sub>s rema</sub>i<sub>ns</sub> t<sub>rue across orac</sub>l<sub>e res</sub>t<sub>ar</sub>t<sub>s</sub> b<sub>ecause</sub> th<sub>e</sub> <sub>cre</sub>dit<sub>s</sub> <sub>are</sub> <sub>preserve</sub>d<sub>.</sub> H<sub>ence</sub> $S _ { t } \geq 0$ at ever<sub>y</sub> <sub>p</sub>re<sup>fi</sup>x.

F<sub>a</sub>llb<sub>ac</sub>k<sub>s.</sub> Si<sub>nce</sub> $s _ { t } ^ { b } \leq 1 / 2$ <sub>,</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e cre</sub>dit <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $h _ { t } \geq \alpha / 2 \geq \alpha \underline { { s } }$ <sub>.</sub> Th<sub>e</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>cre</sub>dit <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\alpha s _ { t } ^ { b } \geq \alpha s$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, $Q _ { t - 1 } \geq \alpha \underline { { s } } ( t - 1 )$ , and rejection at time t implies

$$
\alpha \underline { { s } } t \leq Q _ { t - 1 } + h _ { t } < G _ { \mathrm { I G W } } ( t ) / 4 .\tag{92}
$$

Ever<sub>y</sub> fallback time therefore belon<sub>g</sub>s to the finite set in (88). It follows that $N _ { T } \leq$ min $\{ T , N _ { \mathrm { I G W } } \}$ <sup>f</sup>or ever<sub>y</sub> $T ,$ <sub>an</sub>d th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>num</sub>b<sub>er o</sub>f f<sub>a</sub>llb<sub>ac</sub>k<sub>s</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e.</sub>

F<sub>or</sub> <sub>an</sub> <sub>exp</sub>li<sub>c</sub>it b<sub>oun</sub>d i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f th<sub>e</sub> <sub>orac</sub>l<sub>e</sub> <sub>regre</sub>t <sub>a</sub>t $T ,$ <sup>s</sup>upp<sup>ose</sup> $N _ { T } > 0$ <sub>an</sub>d l<sub>e</sub>t $\tau = \operatorname* { m a x } B _ { T }$ b<sub>e</sub> th<sub>e</sub> last fallback throu<sub>g</sub>h time T. Combinin<sub>g</sub> (92) with (87) <sub>y</sub>ields

$$
4 \alpha \underline { { s } } \tau < G _ { \mathrm { I G W } } ( \tau ) \leq c _ { G } \sqrt { K \tau \overline { { E } } ( \tau ) } .
$$

S<sub>quar</sub>i<sub>ng</sub> <sub>an</sub>d di<sub>v</sub>idi<sub>ng</sub> b<sub>y</sub> $\tau > 0$ <sub>g</sub><sup>i</sup>ves

$$
\tau < \frac { c _ { G } ^ { 2 } K \overline { { E } } ( \tau ) } { 1 6 \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \leq \frac { c _ { G } ^ { 2 } K \overline { { E } } ( T ) } { 1 6 \alpha ^ { 2 } \underline { { s } } ^ { 2 } } ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> <sub>uses</sub> $\tau \leq T$ <sub>an</sub>d <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y</sub> <sub>o</sub>f ${ \overline { { E } } } .$ Si<sub>nce</sub> $N _ { T } \le \tau$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
N _ { T } \leq \mathrm { m i n } \left\{ T , N _ { \mathrm { I G W } } , \frac { c _ { G } ^ { 2 } K \overline { { E } } ( T ) } { 1 6 { \alpha ^ { 2 } } { \underline { { s } } } ^ { 2 } } \right\} .\tag{93}
$$

Th<sub>e same</sub> b<sub>oun</sub>d i<sub>s</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e w</sub>h<sub>en</sub> $N _ { T } = 0$

R<sub>egre</sub>t<sub>.</sub> O<sub>n</sub> <sub>can</sub>did<sub>a</sub>t<sub>es,</sub> $R _ { t } = r _ { t } ^ { \circ }$ <sub>.</sub> O<sub>n</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>s,</sub>

$$
\begin{array} { r } { R _ { t } = \frac 1 2 \left[ \Delta _ { t , b _ { t } } + \Delta _ { t } ( \mathbf y _ { t } ) \right] \leq \frac 1 2 ( \Delta _ { h } + r _ { t } ^ { \circ } ) . } \end{array}
$$

Summin<sub>g</sub> and usin<sub>g</sub> (85) <sub>g</sub>ives

$$
\begin{array} { r l } & { { \tt R e g } ( T ) \leq \displaystyle \sum _ { t \in { \mathcal A } _ { T } } r _ { t } ^ { \circ } + \frac { 1 } { 2 } \sum _ { t \in { \mathcal B } _ { T } } r _ { t } ^ { \circ } + \frac { 1 } { 2 } \Delta _ { h } N _ { T } } \\ & { \quad \quad \leq G _ { \mathrm { I G W } } ( T ) + \frac { 1 } { 2 } \Delta _ { h } N _ { T } . } \end{array}\tag{94}
$$

Equations (87) and (93) therefore im<sub>p</sub>l<sub>y</sub> the ex<sub>p</sub>licit bound

$$
\mathtt { R e g } ( T ) \le c _ { G } \sqrt { K T \overline { { { E } } } ( T ) } + \frac { c _ { G } ^ { 2 } \Delta _ { h } K \overline { { { E } } } ( T ) } { 3 2 \alpha ^ { 2 } \underline { { { s } } } ^ { 2 } } .\tag{95}
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $\overline { { E } } ( T ) = 8 \mathcal { R } _ { \log } ( T ) / \kappa + 5 \mathcal { L } ( T ) / \kappa ^ { 2 }$ from (86) and su<sub>pp</sub>ressin<sub>g</sub> lo<sub>g</sub>arithmic factors in $T$ <sub>an</sub>d $1 / \delta$ <sub>g</sub><sup>i</sup>ves

$$
\mathtt { R e g } ( T ) = \tilde { \mathcal { O } } \left( \sqrt { K T \left[ \frac { \mathcal { R } _ { \mathrm { l o g } } ( T ) } { \kappa } + \frac { 1 } { \kappa ^ { 2 } } \right] } + \frac { \Delta _ { h } K } { \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \left[ \frac { \mathcal { R } _ { \mathrm { l o g } } ( T ) } { \kappa } + \frac { 1 } { \kappa ^ { 2 } } \right] \right) ,
$$

which is (22). In <sub>p</sub>articular, $\overline { { E } } ( T ) = o ( T )$ i<sub>mp</sub>li<sub>es</sub> $\mathsf { R e g } ( T ) = o ( T )$ directl<sub>y</sub> from (95). We also retain th<sub>e un</sub>if<sub>orm</sub> b<sub>oun</sub>d

$$
\begin{array} { r } { \mathsf { R e g } ( T ) \leq G _ { \mathrm { I G W } } ( T ) + \frac 1 2 \Delta _ { h } N _ { \mathrm { I G W } } , } \end{array}\tag{96}
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>use</sub>f<sub>u</sub>l <sub>w</sub>h<sub>en</sub> <sub>a</sub> <sub>grow</sub>th b<sub>oun</sub>d <sub>on</sub> $\mathcal { R } _ { \mathrm { l o g } }$ <sub>a</sub>ll<sub>ows</sub> <sub>an</sub> <sub>exp</sub>li<sub>c</sub>it h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>o</sub>f $N _ { \mathrm { I G W } }$ <sub>.</sub> All <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s</sub> h<sub>o</sub>ld <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>or every</sub> $T$ <sub>on</sub> th<sub>e same even</sub>t <sub>o</sub>f <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta$

## B.7. A finite-class oracle and explicit specialization

<sup>S</sup>upp<sup>ose</sup> $1 \leq | \mathcal { F } | < \infty$ and each function has range diameter at most B. Within each epoch initialize $w _ { 1 } ( g ) = 1 / \vert \mathcal { F } \vert , \ker \eta = e ^ { - B }$ <sub>, an</sub>d <sub>pre</sub>di<sub>c</sub>t <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e accor</sub>di<sub>ng</sub> t<sub>o</sub>

$$
\hat { f } _ { m } ( \mathbf { x } ) = \sum _ { g \in \mathcal { F } } w _ { m } ( g ) g ( \mathbf { x } ) ,\tag{97}
$$

$$
w _ { m + 1 } ( g ) = \frac { w _ { m } ( g ) e ^ { - \eta \ell _ { m } ( g ) } } { \sum _ { h \in \mathcal { F } } w _ { m } ( h ) e ^ { - \eta \ell _ { m } ( h ) } } .\tag{98}
$$

Th<sub>e common we</sub>i<sub>g</sub>ht<sub>s pro</sub>d<sub>uce one pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>u</sub>tilit<sub>y</sub> f<sub>unc</sub>ti<sub>on, an</sub>d $\begin{array} { r } { | \hat { f } _ { m } ( \mathbf { x } ) - \hat { f } _ { m } ( \mathbf { y } ) | \leq \sum _ { g } w _ { m } ( g ) B = } \end{array}$ $B .$

Proposition B.1 (Finite-class logistic oracle) The oracle in (97)–(98) satisfies Assumption 5 with $\mathcal { R } _ { \mathrm { l o g } } ( n ) = e ^ { B } \log | \mathcal { F } |$ for every n.

Proof For $\ell ( d , o ) = \log ( 1 + e ^ { d } ) - o d ,$

$$
\frac { \partial _ { d } ^ { 2 } \ell ( d , o ) } { ( \partial _ { d } \ell ( d , o ) ) ^ { 2 } } = \left\{ e ^ { d } , \quad o = 1 , \quad \quad | d | \leq B . \right.
$$

Thi<sub>s ra</sub>ti<sub>o</sub> i<sub>s a</sub>t l<sub>eas</sub>t $e ^ { - B } = \eta , \mathrm { s o } d \mapsto e ^ { - \eta \ell ( d , o ) }$ i<sub>s concave on</sub> $[ - B , B ]$ <sub>.</sub> B<sub>ecause</sub> th<sub>e pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>marg</sub>i<sub>n</sub> i<sub>s</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>average o</sub>f th<sub>e</sub> f<sub>unc</sub>ti<sub>ons</sub>’ <sub>marg</sub>i<sub>ns,</sub> J<sub>ensen</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
e ^ { - \eta \ell _ { m } ( \hat { f } _ { m } ) } \geq Z _ { m } , \qquad Z _ { m } : = \sum _ { g } w _ { m } ( g ) e ^ { - \eta \ell _ { m } ( g ) } .
$$

Th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>g</sub>i<sub>ves</sub> $\begin{array} { r } { \prod _ { m = 1 } ^ { n } Z _ { m } = | \mathcal { F } | ^ { - 1 } \sum _ { g } \exp ( - \eta \sum _ { m = 1 } ^ { n } \ell _ { m } ( g ) ) } \end{array}$ <sub>.</sub> T<sub>a</sub>ki<sub>ng</sub> l<sub>ogar</sub>ith<sub>ms</sub> <sub>an</sub>d <sub>compar</sub>i<sub>ng</sub> th<sub>e</sub> <sub>sum</sub> <sub>w</sub>ith <sub>any</sub> <sub>one</sub> <sub>o</sub>f it<sub>s</sub> t<sub>erms</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>g</sub>i<sub>ves</sub>

$$
\sum _ { m = 1 } ^ { n } \ell _ { m } ( \hat { f } _ { m } ) \leq \operatorname* { m i n } _ { g \in \mathcal { F } } \sum _ { m = 1 } ^ { n } \ell _ { m } ( g ) + \frac { \log | \mathcal { F } | } { \eta } .
$$

Th<sub>e</sub> b<sub>oun</sub>d i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> f<sub>or</sub> <sub>every</sub> <sub>sequence</sub> <sub>an</sub>d <sub>every</sub> <sub>pre</sub>fi<sub>x,</sub> <sub>an</sub>d <sub>so</sub> <sub>perm</sub>it<sub>s</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>con</sub>t<sub>ex</sub>t<sub>s</sub> <sub>an</sub>d <sub>compar</sub>i<sub>sons.</sub> Th<sub>e range an</sub>d ti<sub>m</sub>i<sub>ng requ</sub>i<sub>remen</sub>t<sub>s a</sub>l<sub>so</sub> h<sub>o</sub>ld b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

F<sub>or</sub> <sub>an</sub> <sub>exp</sub>li<sub>c</sub>it f<sub>a</sub>llb<sub>ac</sub>k b<sub>oun</sub>d<sub>,</sub> l<sub>e</sub>t

$$
a _ { 0 } : = \frac { c _ { G } ^ { 2 } K } { 1 6 \alpha ^ { 2 } { \underline { { s } } } ^ { 2 } } \left[ \frac { 8 e ^ { B } \log | \mathcal { F } | } { \kappa } + \frac { 5 } { \kappa ^ { 2 } } \log \frac { 1 2 } { \delta } \right] ,\tag{99}
$$

$$
a _ { 1 } : = \frac { 1 5 c _ { G } ^ { 2 } K } { 1 6 \alpha ^ { 2 } { \underline { { s } } } ^ { 2 } \kappa ^ { 2 } } .\tag{100}
$$

F<sub>or</sub> $t \geq 1 , 1 + \log _ { 2 } t \leq 2 t$ <sub>an</sub>d $2 + \log _ { 2 } t \leq 2 t$ , so $\mathcal { L } ( t ) \leq \log ( 1 2 / \delta ) + 3 \log t$ . Combinin<sub>g</sub> (87) <sub>w</sub>ith th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>or</sub> $N _ { \mathrm { I G W } }$ <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>eac</sub>h i<sub>n</sub>t<sub>eger</sub> i<sub>n</sub> th<sub>a</sub>t <sub>se</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $t < a _ { 0 } + a _ { 1 }$ log t. Concavity of the logarithm gives log $t \leq t / ( 2 a _ { 1 } ) + \log ( 2 a _ { 1 } ) - 1$ <sub>,</sub> <sub>w</sub>h<sub>ence</sub>

$$
N _ { \mathrm { I G W } } \leq 2 a _ { 0 } + 2 a _ { 1 } \log \bigl ( \operatorname* { m a x } \{ e , 2 a _ { 1 } \} \bigr ) .\tag{101}
$$

Combinin<sub>g</sub> (101) with (96), and writin<sub>g</sub> $A _ { \mathcal { F } } : = e ^ { B } \log | \mathcal { F } | / \kappa + 1 / \kappa ^ { 2 }$ , <sub>g</sub><sup>i</sup>ves

$$
\mathsf { R e g } ( T ) = \tilde { \mathcal { O } } \biggl ( \sqrt { K A _ { \mathcal { F } } T } + \frac { \Delta _ { h } K A _ { \mathcal { F } } } { \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \biggr ) .\tag{102}
$$

H<sub>ere</sub> th<sub>e suppresse</sub>d f<sub>ac</sub>t<sub>ors are</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> i<sub>n</sub> $T , 1 / \delta , K , A _ { \mathcal { F } } $ <sub>,</sub> <sub>an</sub>d $1 / ( \alpha \underline { { s } } )$ <sub>.</sub> Th<sub>e a</sub>dditi<sub>ve con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> uses (101) and can be bounded inde<sub>p</sub>endentl<sub>y</sub> of the sto<sub>pp</sub>in<sub>g</sub> horizon. For $B = 1$ , κ is a numerical <sub>cons</sub>t<sub>an</sub>t <sub>an</sub>d $A _ { \mathcal { F } } = \mathcal { O } ( \log | \mathcal { F } | + 1 )$ . <sup>C</sup>om<sub>p</sub>ut<sup>i</sup>n<sub>g</sub> $p _ { t }$ t<sub>a</sub>k<sub>es</sub> $\mathcal O ( K )$ <sub>ar</sub>ith<sub>me</sub>ti<sub>c</sub> <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> $K$ score <sub>eva</sub>l<sub>ua</sub>ti<sub>ons;</sub> th<sub>e orac</sub>l<sub>e</sub>’<sub>s own compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>cos</sub>t i<sub>s separa</sub>t<sub>e.</sub>

## B.8. Relaxing baseline-score knowledge

<sup>S</sup>upp<sup>ose</sup> $L _ { t } , U _ { t }$ <sub>are ava</sub>il<sub>a</sub>bl<sub>e</sub> b<sub>e</sub>f<sub>ore pa</sub>i<sub>r samp</sub>li<sub>ng an</sub>d <sub>a</sub>l<sub>ways sa</sub>ti<sub>s</sub>f<sub>y</sub> $\underline { { s } } \le L _ { t } \le U _ { t } \le 1 / 2$ <sub>.</sub> A<sub>ssume</sub> th<sub>a</sub>t<sub>,</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta _ { b }$ <sub>,</sub> th<sub>ey</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> <sub>cover</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>scores</sub> <sub>a</sub>t <sub>every</sub> <sub>roun</sub>d<sub>:</sub> $L _ { t } \leq s _ { t } ^ { b } \leq U _ { t }$ <sub>.</sub> R<sub>ep</sub>l<sub>ace</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e cre</sub>dit <sub>an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>cre</sub>dit<sub>, respec</sub>ti<sub>ve</sub>l<sub>y,</sub> b<sub>y</sub>

$$
\begin{array} { r } { h _ { t } = \frac { 1 } { 2 } - ( 1 - \alpha ) U _ { t } , \qquad \mathrm { f a l l b a c k ~ c r e d i t } = \alpha L _ { t } . } \end{array}\tag{103}
$$

U<sub>se</sub> th<sub>ese cre</sub>dit<sub>s</sub> i<sub>n</sub> th<sub>e same accep</sub>t<sub>ance</sub> t<sub>es</sub>t <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es.</sub> O<sub>n</sub> th<sub>e coverage even</sub>t<sub>, a can</sub>did<sub>a</sub>t<sub>e</sub>’<sub>s</sub> t<sub>rue</sub> <sub>surp</sub>l<sub>us</sub> i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $h _ { t } - R _ { t } / 4 .$ <sub>,</sub> <sub>an</sub>d <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k’<sub>s</sub> t<sub>rue</sub> <sub>surp</sub>l<sub>us</sub> i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $\alpha L _ { t }$ <sub>.</sub> Th<sub>us</sub> th<sub>e</sub> <sub>same</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y</sub> <sub>argumen</sub>t <sub>app</sub>li<sub>es w</sub>ith th<sub>ese cre</sub>dit<sub>s.</sub> M<sub>oreover,</sub> $h _ { t } \geq \alpha / 2$ and α $L _ { t } \geq \alpha s .$ <sub>, so</sub> th<sub>e</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>-coun</sub>t <sub>argumen</sub>t <sub>an</sub>d <sub>a</sub>ll <sub>regre</sub>t b<sub>oun</sub>d<sub>s are unc</sub>h<sub>ange</sub>d<sub>.</sub> Th<sub>e overa</sub>ll <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> b<sub>ecomes a</sub>t l<sub>eas</sub>t $1 - \delta - \delta _ { b }$ <sub>, w</sub>ith<sub>ou</sub>t <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence requ</sub>i<sub>remen</sub>t<sub>.</sub> D<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> b<sub>oun</sub>d<sub>s</sub> h<sub>ave</sub> $\delta _ { b } = 0 ;$ i<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> $\left( L _ { t } , U _ { t } \right) = \left( \underline { { s } } , 1 / 2 \right)$ <sub>removes exac</sub>t b<sub>ase</sub>li<sub>ne-score</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge en</sub>ti<sub>re</sub>l<sub>y w</sub>hil<sub>e re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e exac</sub>t <sub>guaran</sub>t<sub>ees o</sub>f Th<sub>eorem</sub> 4<sub>.</sub>1<sub>.</sub>

## B.9. A known-horizon version without oracle restarts

If a horizon T is <sub>g</sub>iven, use one oracle instance, the <sub>p</sub>arameters in (72) with $( n , \delta ^ { \prime } ) \ = \ ( T , \delta )$ $\gamma _ { t } = \sqrt { 8 K t / E }$ <sub>, an</sub>d th<sub>e enve</sub>l<sub>ope</sub> $G _ { T } ( t ) = C ( T , \delta ) \sqrt { t }$ <sub>.</sub> Th<sub>e same</sub> t<sub>es</sub>t <sub>an</sub>d <sub>samp</sub>li<sub>ng ru</sub>l<sub>es</sub> th<sub>en</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta$ <sub>, s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>or every</sub> $t \leq T$ <sub>,</sub> th<sub>e max-score cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>an</sub>d

$$
N _ { t } \leq \operatorname* { m i n } \left\{ t , \frac { C ( T , \delta ) ^ { 2 } } { 1 6 \alpha ^ { 2 } \underline { { s } } ^ { 2 } } \right\} ,\tag{104}
$$

$$
\begin{array} { r } { \mathtt { R e g } ( t ) \le C ( T , \delta ) \sqrt { t } + \frac { 1 } { 2 } \Delta _ { h } N _ { t } . } \end{array}\tag{105}
$$

Indeed, Lemma B.3 <sub>g</sub>ives the required envelo<sub>p</sub>e on the entire run, and (92) forces ever<sub>y</sub> fallback ti<sub>me</sub> t<sub>o</sub> b<sub>e sma</sub>ll<sub>er</sub> th<sub>an</sub> $C ( T , \delta ) ^ { 2 } / ( 1 6 \alpha ^ { 2 } \underline { { s } } ^ { 2 } )$ <sub>.</sub> Th<sub>e sa</sub>f<sub>e</sub>t<sub>y an</sub>d <sub>regre</sub>t <sub>argumen</sub>t<sub>s are o</sub>th<sub>erw</sub>i<sub>se</sub> id<sub>en</sub>ti<sub>ca</sub>l<sub>.</sub> L<sub>emma</sub> B<sub>.</sub>4 <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e</sub> <sub>exp</sub>li<sub>c</sub>it t<sub>erm</sub>i<sub>na</sub>l b<sub>oun</sub>d

$$
\mathsf { R e g } ( T ) \leq \frac { 9 } { 4 } \sqrt { K T E } + \frac { 8 1 \Delta _ { h } K E } { 5 1 2 \alpha ^ { 2 } \underline { { s } } ^ { 2 } } .\tag{106}
$$

Thi<sub>s vers</sub>i<sub>on uses</sub> th<sub>e</sub> k<sub>nown</sub> h<sub>or</sub>i<sub>zon</sub> i<sub>n</sub> it<sub>s parame</sub>t<sub>ers an</sub>d <sub>avo</sub>id<sub>s orac</sub>l<sub>e res</sub>t<sub>ar</sub>t<sub>s.</sub> Th<sub>e epoc</sub>h construction additionall<sub>y</sub> admits the uniform total-fallback bound (88).