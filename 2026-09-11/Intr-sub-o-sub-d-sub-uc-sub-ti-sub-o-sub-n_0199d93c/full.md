# Or<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>p</sub>rin<sub>c</sub>i<sub>p</sub>l<sub>es e</sub>n<sub>a</sub>bl<sub>e co</sub>ll<sub>ec</sub>ti<sub>ve</sub> int<sub>e</sub>lli<sub>ge</sub>n<sub>ce</sub> in <sub>e</sub>mb<sub>o</sub>di<sub>e</sub>d AI

Zh<sub>e</sub>n<sub>g</sub>r<sub>a</sub>n Ji<sup>1</sup><sub>,</sub> J<sub>o</sub>n<sub>a</sub>th<sub>a</sub>n H<sub>yu</sub>n<sup>1</sup><sub>,</sub> B<sub>oyua</sub>n Ch<sub>e</sub>n<sup>1,2,3∗</sup>

<sup>1</sup>D<sub>epa</sub>rtm<sub>e</sub>nt <sub>o</sub>f C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r S<sub>c</sub>i<sub>e</sub>n<sub>ce,</sub> D<sub>u</sub>k<sub>e</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y,</sub> D<sub>u</sub>rh<sub>a</sub>m<sub>,</sub> USA<sub>.</sub>

<sup>2</sup>D<sub>epa</sub>rtm<sub>e</sub>nt <sub>o</sub>f El<sub>ec</sub>tri<sub>ca</sub>l <sub>a</sub>nd C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r En<sub>g</sub>in<sub>ee</sub>rin<sub>g,</sub> D<sub>u</sub>k<sub>e</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y,</sub> D<sub>u</sub>rh<sub>a</sub>m<sub>,</sub> USA<sub>.</sub>

<sup>3</sup>D<sub>epar</sub>t<sub>men</sub>t <sub>o</sub>f M<sub>ec</sub>h<sub>an</sub>i<sub>ca</sub>l E<sub>ng</sub>i<sub>neer</sub>i<sub>ng</sub> <sub>an</sub>d M<sub>a</sub>t<sub>er</sub>i<sub>a</sub>l<sub>s</sub> S<sub>c</sub>i<sub>ence,</sub> D<sub>u</sub>k<sub>e</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y,</sub> D<sub>ur</sub>h<sub>am,</sub> USA<sub>.</sub>

<sup>∗</sup>C<sub>orrespon</sub>di<sub>ng</sub> <sub>au</sub>th<sub>or.</sub> E<sub>ma</sub>il<sub>:</sub> b<sub>oyuan.c</sub>h<sub>en</sub>@d<sub>u</sub>k<sub>e.e</sub>d<sub>u.</sub>

https://generalroboticslab.com/ORCH

C<sub>o</sub>ll<sub>ec</sub>ti<sub>ve</sub> int<sub>e</sub>lli<sub>ge</sub>n<sub>ce</sub> d<sub>epe</sub>nd<sub>s</sub> n<sub>o</sub>t <sub>o</sub>nl<sub>y</sub> <sub>o</sub>n th<sub>e</sub> <sub>capa</sub>biliti<sub>es</sub> <sub>o</sub>f indi<sub>v</sub>id<sub>ua</sub>l m<sub>e</sub>m<sub>-</sub> b<sub>e</sub>r<sub>s,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so o</sub>n h<sub>ow</sub> th<sub>ose</sub> m<sub>e</sub>mb<sub>e</sub>r<sub>s a</sub>r<sub>e o</sub>r<sub>ga</sub>niz<sub>e</sub>d<sub>.</sub> Y<sub>e</sub>t <sub>a</sub>rtifi<sub>c</sub>i<sub>a</sub>l m<sub>u</sub>lti-<sub>age</sub>nt <sub>sys</sub>t<sub>e</sub>m<sub>s a</sub>r<sub>e</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y asse</sub>mbl<sub>e</sub>d <sub>us</sub>in<sub>g</sub> fix<sub>e</sub>d <sub>o</sub>r<sub>ga</sub>niz<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>s</sub>tr<sub>uc</sub>t<sub>u</sub>r<sub>es, eve</sub>n <sub>w</sub>h<sub>e</sub>n th<sub>e</sub> <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l t<sub>as</sub>k<sub>s</sub> th<sub>ey</sub> <sub>pe</sub>rf<sub>o</sub>rm im<sub>pose</sub> f<sub>u</sub>nd<sub>a</sub>m<sub>e</sub>nt<sub>a</sub>ll<sub>y</sub> dif<sub>e</sub>r<sub>e</sub>nt <sub>coo</sub>rdin<sub>a</sub>ti<sub>o</sub>n r<sub>e</sub>- <sub>qu</sub>i<sub>remen</sub>t<sub>s.</sub> H<sub>ere</sub> <sub>we</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> f<sub>rom</sub> h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory</sub> <sub>can</sub> be o<sub>p</sub>erationalized to or<sub>g</sub>anize lar<sub>g</sub>e<sub>,</sub> hetero<sub>g</sub>eneo<sub>u</sub>s collecti<sub>v</sub>es of embodied artificial a<sub>g</sub>ents. We introduce ORCH (Or<sub>g</sub>anizin<sub>g</sub> Roles and Coordination Hierarchies), which constructs task-specific hierarchical or<sub>g</sub>anizations b<sub>y</sub> combinin<sub>g</sub> <sub>poo</sub>l<sub>e</sub>d int<sub>e</sub>rd<sub>epe</sub>nd<sub>e</sub>n<sub>ce</sub> f<sub>o</sub>r <sub>wo</sub>rk th<sub>a</sub>t <sub>ca</sub>n <sub>p</sub>r<sub>ocee</sub>d <sub>co</sub>n<sub>cu</sub>rr<sub>e</sub>ntl<sub>y w</sub>ith <sub>seque</sub>nti<sub>a</sub>l int<sub>e</sub>rd<sub>epe</sub>nd<sub>e</sub>n<sub>ce</sub> f<sub>o</sub>r <sub>wo</sub>rk <sub>gove</sub>rn<sub>e</sub>d b<sub>y</sub> <sub>p</sub>r<sub>e</sub>r<sub>equ</sub>i<sub>s</sub>it<sub>e</sub> r<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>hi<sub>ps.</sub> A<sub>c</sub>r<sub>oss</sub> 25 wildfire-res<sub>p</sub>onse missions s<sub>p</sub>annin<sub>g</sub> reconnaissance<sub>,</sub> rescue<sub>,</sub> trans<sub>p</sub>ortation<sub>,</sub> resource mana<sub>g</sub>ement<sub>,</sub> containment and su<sub>pp</sub>ression<sub>,</sub> we evaluated teams of u<sub>p</sub> to 50 hetero<sub>g</sub>eneous a<sub>g</sub>ents usin<sub>g</sub> ei<sub>g</sub>ht lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models. Or<sub>g</sub>anizations constructed usin<sub>g</sub> these <sub>p</sub>rinci<sub>p</sub>les consistentl<sub>y</sub> out<sub>p</sub>erformed four re<sub>p</sub>resentative embodied m<sub>u</sub>lti-a<sub>g</sub>ent a<sub>pp</sub>roaches across mission o<sub>u</sub>tcome<sub>,</sub> exec<sub>u</sub>tion eficienc<sub>y,</sub>

<sub>exp</sub>l<sub>o</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd <sub>co</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l r<sub>esou</sub>r<sub>ce use.</sub> H<sub>u</sub>m<sub>a</sub>n<sub>-</sub>d<sub>es</sub>i<sub>g</sub>n<sub>e</sub>d ORCH <sub>o</sub>r<sub>ga</sub>ni<sub>-</sub> <sub>za</sub>ti<sub>o</sub>n<sub>s</sub> im<sub>p</sub>r<sub>ove</sub>d fin<sub>a</sub>l <sub>sco</sub>r<sub>e</sub> b<sub>y</sub> 63<sub>.</sub>97% <sub>a</sub>nd <sub>execu</sub>ti<sub>o</sub>n <sub>e</sub>fi<sub>c</sub>i<sub>e</sub>n<sub>cy</sub> b<sub>y</sub> $7 4 . 2 9 \%$ on a<sub>v</sub>era<sub>g</sub>e relati<sub>v</sub>e to the fo<sub>u</sub>r <sub>p</sub>rior frame<sub>w</sub>orks<sub>.</sub> Or<sub>g</sub>anizations <sub>g</sub>enerated a<sub>u</sub>tom<sub>a</sub>ti<sub>ca</sub>ll<sub>y</sub> b<sub>y</sub> l<sub>a</sub>n<sub>guage</sub> m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> im<sub>p</sub>r<sub>ove</sub>d th<sub>ese</sub> m<sub>easu</sub>r<sub>es</sub> b<sub>y</sub> $4 3 . 6 3 \%$ <sub>a</sub>nd 52<sub>.</sub>53%<sub>,</sub> res<sub>p</sub>ecti<sub>v</sub>el<sub>y.</sub> These ad<sub>v</sub>anta<sub>g</sub>es <sub>p</sub>ersisted across missions and <sub>u</sub>nderl<sub>y</sub>in<sub>g</sub> lan<sub>gu</sub>a<sub>g</sub>e m<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> N<sub>o</sub>t<sub>a</sub>bl<sub>y, co</sub>ll<sub>ec</sub>ti<sub>ve pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce was</sub> n<sub>o</sub>t m<sub>o</sub>n<sub>o</sub>t<sub>o</sub>ni<sub>ca</sub>ll<sub>y</sub> d<sub>e</sub>t<sub>e</sub>rmin<sub>e</sub>d b<sub>y</sub> m<sub>o</sub>d<sub>e</sub>l <sub>sca</sub>l<sub>e.</sub> An<sub>a</sub>l<sub>ys</sub>i<sub>s o</sub>f l<sub>o</sub>n<sub>g-</sub>h<sub>o</sub>ri<sub>zo</sub>n mi<sub>ss</sub>i<sub>o</sub>n<sub>s s</sub>h<sub>owe</sub>d th<sub>a</sub>t hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>hi<sub>ca</sub>l <sub>o</sub>r<sub>ga</sub>ni<sub>za-</sub> tion enabled teams to <sub>p</sub>reser<sub>v</sub>e conc<sub>u</sub>rrent acti<sub>v</sub>it<sub>y</sub> <sub>w</sub>ithin s<sub>p</sub>ecialized <sub>g</sub>ro<sub>up</sub>s <sub>w</sub>hile <sub>coo</sub>rdin<sub>a</sub>tin<sub>g o</sub>rd<sub>e</sub>r<sub>e</sub>d tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>s</sub> b<sub>e</sub>t<sub>wee</sub>n mi<sub>ss</sub>i<sub>o</sub>n <sub>p</sub>h<sub>ases.</sub> Th<sub>ese</sub> r<sub>esu</sub>lt<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>o</sub>r<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l d<sub>es</sub>i<sub>g</sub>n <sub>as a</sub> f<sub>u</sub>nd<sub>a</sub>m<sub>e</sub>nt<sub>a</sub>l dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n <sub>o</sub>f <sub>a</sub>rtifi<sub>c</sub>i<sub>a</sub>l <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> int<sub>e</sub>lli<sub>-</sub> <sub>g</sub>ence and su<sub>gg</sub>est that <sub>p</sub>rinci<sub>p</sub>les develo<sub>p</sub>ed to understand human or<sub>g</sub>anizations <sub>ca</sub>n <sub>gu</sub>id<sub>e</sub> th<sub>e co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>o</sub>f <sub>sca</sub>l<sub>a</sub>bl<sub>e e</sub>mb<sub>o</sub>di<sub>e</sub>d AI <sub>co</sub>ll<sub>ec</sub>ti<sub>ves.</sub>

## Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

I<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub> i<sub>n groups ar</sub>i<sub>ses no</sub>t <sub>on</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e a</sub>biliti<sub>es o</sub>f th<sub>e</sub>i<sub>r mem</sub>b<sub>ers,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> f<sub>rom</sub> th<sub>e way</sub> th<sub>ose mem</sub>b<sub>ers are organ</sub>i<sub>ze</sub>d<sub>.</sub> H<sub>uman soc</sub>i<sub>e</sub>ti<sub>es coor</sub>di<sub>na</sub>t<sub>e ac</sub>ti<sub>v</sub>iti<sub>es o</sub>f <sub>ex</sub>t<sub>raor</sub>di<sub>nary sca</sub>l<sub>e</sub> b<sub>y</sub> di<sub>v</sub>idi<sub>ng</sub> <sub>respons</sub>ibiliti<sub>es,</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> <sub>au</sub>th<sub>or</sub>it<sub>y</sub> <sub>an</sub>d <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ures,</sub> <sub>an</sub>d <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> f<sub>orms</sub> <sub>o</sub>f <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps</sub> <sub>among</sub> t<sub>as</sub>k<sub>s.</sub> E<sub>mergency</sub> <sub>response</sub> t<sub>eams,</sub> f<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> <sub>com</sub>bi<sub>ne</sub> <sub>ac</sub>ti<sub>v</sub>iti<sub>es</sub> th<sub>a</sub>t <sub>can procee</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y, suc</sub>h <sub>as searc</sub>hi<sub>ng</sub> dif<sub>eren</sub>t <sub>reg</sub>i<sub>ons, w</sub>ith <sub>ac</sub>ti<sub>v</sub>iti<sub>es</sub> th<sub>a</sub>t <sub>mus</sub>t <sub>occur</sub> i<sub>n</sub> <sub>a</sub> <sub>prescr</sub>ib<sub>e</sub>d <sub>or</sub>d<sub>er,</sub> <sub>suc</sub>h <sub>as</sub> l<sub>oca</sub>ti<sub>ng</sub> <sub>a</sub> h<sub>azar</sub>d b<sub>e</sub>f<sub>ore</sub> d<sub>ep</sub>l<sub>oy</sub>i<sub>ng</sub> <sub>personne</sub>l <sub>an</sub>d <sub>resources.</sub> A<sub>n</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>a</sub>t f<sub>a</sub>il<sub>s</sub> t<sub>o</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h th<sub>ese</sub> f<sub>orms</sub> <sub>o</sub>f <sub>wor</sub>k <sub>can</sub> <sub>per</sub>f<sub>orm</sub> <sub>poor</sub>l<sub>y</sub> <sub>even</sub> <sub>w</sub>h<sub>en</sub> <sub>every</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l i<sub>s</sub> <sub>capa</sub>bl<sub>e.</sub> O<sub>rgan</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>no</sub>t <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>an</sub> <sub>a</sub>d<sub>m</sub>i<sub>n</sub>i<sub>s</sub>t<sub>ra</sub>ti<sub>ve</sub> l<sub>ayer</sub> <sub>p</sub>l<sub>ace</sub>d <sub>a</sub>b<sub>ove</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence,</sub> b<sub>u</sub>t <sub>one</sub> <sub>o</sub>f th<sub>e</sub> <sub>core</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sms</sub> th<sub>roug</sub>h <sub>w</sub>hi<sub>c</sub>h i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>capa</sub>biliti<sub>es</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> t<sub>o</sub> the greater collective intelligence (1–3).

B<sub>u</sub>ildi<sub>ng ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>co</sub>ll<sub>ec</sub>ti<sub>ves poses a correspon</sub>di<sub>ng c</sub>h<sub>a</sub>ll<sub>enge.</sub> P<sub>rogress</sub> i<sub>n re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn-</sub> i<sub>ng,</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>li<sub>ng,</sub> <sub>an</sub>d <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d i<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub> h<sub>as</sub> <sub>pro</sub>d<sub>uce</sub>d <sub>agen</sub>t<sub>s</sub> th<sub>a</sub>t <sub>can</sub> <sub>compe</sub>t<sub>e,</sub> <sub>coop-</sub> erate, negotiate, and act over increasingly long horizons (4–11). Large language models further provide a general interface for decomposing objectives, assigning tasks, communicating observations, and revising plans (12–15). These capabilities have motivated systems in which multiple l<sub>anguage-mo</sub>d<sub>e</sub>l<sub>-</sub>b<sub>ase</sub>d <sub>agen</sub>t<sub>s</sub> <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>t<sub>e</sub> <sub>on</sub> <sub>pro</sub>bl<sub>ems</sub> th<sub>a</sub>t <sub>excee</sub>d th<sub>e</sub> <sub>scope</sub> <sub>o</sub>f <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>agen</sub>t<sub>.</sub> H<sub>owever,</sub> i<sub>ncreas</sub>i<sub>ng</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>an</sub>d di<sub>vers</sub>it<sub>y</sub> <sub>o</sub>f <sub>agen</sub>t<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>crea</sub>t<sub>es</sub> <sub>a</sub> <sub>new</sub> <sub>source</sub> <sub>o</sub>f <sub>comp</sub>l<sub>ex</sub>it<sub>y.</sub> E<sub>ac</sub>h <sub>a</sub>dditi<sub>ona</sub>l <sub>wor</sub>k<sub>er</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> <sub>new</sub> <sub>capa</sub>biliti<sub>es</sub> th<sub>a</sub>t <sub>can</sub> b<sub>e</sub> <sub>exp</sub>l<sub>o</sub>it<sub>e</sub>d<sub>,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> <sub>commun</sub>i<sub>ca-</sub> ti<sub>on,</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>requ</sub>i<sub>remen</sub>t<sub>s</sub> th<sub>a</sub>t <sub>mus</sub>t b<sub>e</sub> <sub>manage</sub>d<sub>.</sub> A <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> <sub>o</sub>f i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>ll<sub>y</sub> <sub>capa</sub>bl<sub>e</sub> <sub>agen</sub>t<sub>s</sub> d<sub>oes</sub> <sub>no</sub>t<sub>,</sub> b<sub>y</sub> it<sub>se</sub>lf<sub>,</sub> <sub>cons</sub>tit<sub>u</sub>t<sub>e</sub> <sub>a</sub> <sub>capa</sub>bl<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve.</sub>

![](images/ea00fc146a630345f0664d5db349f6b6fb2cb2b022344b9b36634ccf986f62ce.jpg)  
Figure 1: Method Overall. (A) The team hierarch<sub>y</sub> was desi<sub>g</sub>ned based on the task descri<sub>p</sub>tions and available worker a<sub>g</sub>ents, either b<sub>y</sub> a human ex<sub>p</sub>ert or an LLM with a critic a<sub>g</sub>ent. (B) Durin<sub>g</sub> <sub>execu</sub>ti<sub>on,</sub> th<sub>e manager genera</sub>t<sub>es</sub> th<sub>e p</sub>l<sub>an</sub> f<sub>or</sub> it<sub>s wor</sub>k<sub>er agen</sub>t<sub>s;</sub> th<sub>e wor</sub>k<sub>er agen</sub>t<sub>s execu</sub>t<sub>e</sub> th<sub>e</sub> <sub>p</sub>lan and <sub>p</sub>rovide feedback for the mana<sub>g</sub>er to u<sub>p</sub>date the <sub>p</sub>lan. (C) Pooled interde<sub>p</sub>endenc<sub>y</sub> refers t<sub>o</sub> <sub>a</sub> <sub>group</sub> <sub>o</sub>f <sub>agen</sub>t<sub>s</sub> <sub>wor</sub>ki<sub>ng</sub> <sub>concurren</sub>tl<sub>y</sub> <sub>an</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> t<sub>o</sub> <sub>ac</sub>hi<sub>eve</sub> <sub>a</sub> <sub>goa</sub>l<sub>.</sub> F<sub>or</sub> i<sub>ns</sub>t<sub>ance,</sub> 3 firefi<sub>g</sub>hters each cut s<sub>p</sub>ecific tar<sub>g</sub>et trees (<sub>y</sub>ellow color) to achieve the <sub>g</sub>oal of cuttin<sub>g</sub> a <sub>g</sub>rou<sub>p</sub> of tar<sub>g</sub>et trees. (D) Sequential interde<sub>p</sub>endenc<sub>y</sub> refers to a <sub>g</sub>rou<sub>p</sub> of a<sub>g</sub>ents workin<sub>g</sub> sequentiall<sub>y</sub> to achieve a <sub>goa</sub>l<sub>.</sub> F<sub>or</sub> i<sub>ns</sub>t<sub>ance,</sub> t<sub>o</sub> <sub>ex</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>a</sub> fi<sub>re,</sub> th<sub>e</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>mus</sub>t fi<sub>rs</sub>t <sub>scou</sub>t th<sub>e</sub> l<sub>oca</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> fi<sub>re;</sub> th<sub>en</sub> th<sub>e</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>er p</sub>i<sub>c</sub>k<sub>s up an</sub>d d<sub>rops o</sub>f th<sub>e</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers near</sub> th<sub>e</sub> fi<sub>re;</sub> fi<sub>na</sub>ll<sub>y,</sub> th<sub>e</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers ex</sub>ti<sub>ngu</sub>i<sub>s</sub>h the fire. (E) ORCH enforces h<sub>y</sub>brid collaboration mode (<sub>p</sub>ooled and sequential interde<sub>p</sub>endenc<sub>y</sub>).

M<sub>os</sub>t <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>mu</sub>lti<sub>-agen</sub>t <sub>sys</sub>t<sub>ems</sub> <sub>a</sub>dd<sub>ress</sub> thi<sub>s</sub> <sub>pro</sub>bl<sub>em</sub> b<sub>y</sub> d<sub>ec</sub>idi<sub>ng</sub> <sub>w</sub>h<sub>a</sub>t <sub>eac</sub>h <sub>agen</sub>t <sub>s</sub>h<sub>ou</sub>ld d<sub>o w</sub>ithi<sub>n an organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure c</sub>h<sub>osen</sub> i<sub>n a</sub>d<sub>vance.</sub> A<sub>gen</sub>t<sub>s may commun</sub>i<sub>ca</sub>t<sub>e</sub> di<sub>rec</sub>tl<sub>y w</sub>ith one another (16, 17), report to a central planner (18), or occupy predefined manager and worker roles (19). These approaches have enabled cooperation among embodied agents, but generally t<sub>rea</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>as</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>proper</sub>t<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>sys</sub>t<sub>em.</sub> Th<sub>e</sub> <sub>same</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> t<sub>opo</sub>l<sub>ogy</sub> <sub>an</sub>d <sub>manager</sub>i<sub>a</sub>l <sub>arrangemen</sub>t <sub>may</sub> <sub>consequen</sub>tl<sub>y</sub> b<sub>e</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> th<sub>a</sub>t dif<sub>er</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> i<sub>n</sub> <sub>sca</sub>l<sub>e,</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>personne</sub>l<sub>,</sub> t<sub>empora</sub>l d<sub>epen</sub>d<sub>enc</sub>i<sub>es,</sub> <sub>an</sub>d <sub>requ</sub>i<sub>re</sub>d <sub>spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on.</sub> Thi<sub>s</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>on</sub> b<sub>ecomes</sub> i<sub>ncreas</sub>i<sub>ng</sub>l<sub>y</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>as</sub> <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l t<sub>eams</sub> <sub>grow</sub> l<sub>arger,</sub> <sub>more</sub> h<sub>e</sub>t<sub>erogeneous,</sub> <sub>an</sub>d <sub>opera</sub>t<sub>e</sub> <sub>over</sub> l<sub>onger</sub> h<sub>or</sub>i<sub>zons.</sub> F<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>ng</sub> <sub>severa</sub>l id<sub>en</sub>ti<sub>ca</sub>l <sub>agen</sub>t<sub>s</sub> <sub>searc</sub>hi<sub>ng</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t l<sub>oca</sub>ti<sub>ons</sub> i<sub>s</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>ll<sub>y</sub> dif<sub>eren</sub>t f<sub>rom</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>ng</sub> d<sub>rones,</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>ers,</sub> <sub>groun</sub>d <sub>ve</sub>hi<sub>c</sub>l<sub>es,</sub> <sub>an</sub>d <sub>personne</sub>l <sub>w</sub>h<sub>ose</sub> <sub>ac</sub>ti<sub>ons</sub> f<sub>orm</sub> <sub>a</sub> <sub>sequence</sub> <sub>o</sub>f i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>ga</sub>th<sub>er</sub>i<sub>ng,</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>prepara</sub>ti<sub>on,</sub> <sub>an</sub>d i<sub>n</sub>t<sub>erven</sub>ti<sub>on.</sub> A<sub>n</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>we</sub>ll <sub>su</sub>it<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> f<sub>ormer</sub> <sub>may</sub> b<sub>e</sub> <sub>poor</sub>l<sub>y</sub> <sub>su</sub>it<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er.</sub>

H<sub>uman organ</sub>i<sub>za</sub>ti<sub>ons a</sub>dd<sub>ress</sub> th<sub>ese</sub> dif<sub>erences</sub> b<sub>y a</sub>d<sub>op</sub>ti<sub>ng</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>coor</sub>di<sub>na</sub>ti<sub>on pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> th<sub>a</sub>t <sub>re</sub>fl<sub>ec</sub>t h<sub>ow wor</sub>k i<sub>s</sub> i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>.</sub> O<sub>rgan</sub>i<sub>za</sub>ti<sub>ona</sub>l th<sub>eory c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>zes wor</sub>k i<sub>n par</sub>t th<sub>roug</sub>h interdependence: the way in which the activity of one unit depends on the activity of others (1,20). Under pooled interdependence, members contribute independently to a shared outcome and can generally work in parallel. Under sequential interdependence, the output of one activity becomes th<sub>e</sub> <sub>prerequ</sub>i<sub>s</sub>it<sub>e</sub> f<sub>or</sub> <sub>ano</sub>th<sub>er,</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> <sub>or</sub>d<sub>ere</sub>d <sub>execu</sub>ti<sub>on</sub> <sub>an</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d h<sub>an</sub>d<sub>o</sub>f<sub>s.</sub> Th<sub>ese</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> h<sub>e</sub>l<sub>p</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> h<sub>ow respons</sub>ibiliti<sub>es,</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on, an</sub>d <sub>au</sub>th<sub>or</sub>it<sub>y s</sub>h<sub>ou</sub>ld b<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>w</sub>ithi<sub>n</sub> h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons.</sub> Wh<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>same</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> <sub>can</sub> <sub>a</sub>l<sub>so</sub> <sub>prov</sub>id<sub>e</sub> <sub>an</sub> <sub>opera</sub>ti<sub>ona</sub>l b<sub>as</sub>i<sub>s</sub> f<sub>or</sub> <sub>organ</sub>i<sub>z</sub>i<sub>ng</sub> <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>agen</sub>t<sub>s</sub> h<sub>as</sub> <sub>rema</sub>i<sub>ne</sub>d <sub>unc</sub>l<sub>ear.</sub>

E<sub>x</sub>i<sub>s</sub>ti<sub>ng mu</sub>lti<sub>-agen</sub>t <sub>s</sub>t<sub>u</sub>di<sub>es</sub> h<sub>ave</sub> i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub>d <sub>commun</sub>i<sub>ca</sub>ti<sub>on s</sub>t<sub>ruc</sub>t<sub>ures, manager-wor</sub>k<sub>er ar-</sub> rangements, and alternative organizational forms (21–23). However, past focus has been on how <sub>agen</sub>t<sub>s s</sub>h<sub>ou</sub>ld <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>t<sub>e w</sub>ithi<sub>n a g</sub>i<sub>ven s</sub>t<sub>ruc</sub>t<sub>ure ra</sub>th<sub>er</sub> th<sub>an w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e s</sub>t<sub>ruc</sub>t<sub>ure</sub> it<sub>se</sub>lf <sub>s</sub>h<sub>ou</sub>ld d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>wor</sub>kf<sub>orce.</sub> A<sub>s</sub> <sub>a</sub> <sub>resu</sub>lt<sub>,</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l t<sub>opo</sub>l<sub>ogy</sub> <sub>an</sub>d <sub>coor-</sub> di<sub>na</sub>ti<sub>on</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> <sub>are</sub> <sub>o</sub>ft<sub>en</sub> <sub>separa</sub>t<sub>e</sub>d<sub>:</sub> <sub>one</sub> <sub>componen</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> <sub>w</sub>h<sub>o</sub> <sub>commun</sub>i<sub>ca</sub>t<sub>es</sub> <sub>w</sub>ith <sub>w</sub>h<sub>om</sub> <sub>un</sub>d<sub>er</sub> <sub>w</sub>h<sub>a</sub>t <sub>respons</sub>ibilit<sub>y</sub> <sub>an</sub>d <sub>au</sub>th<sub>or</sub>it<sub>y,</sub> <sub>w</sub>hil<sub>e</sub> <sub>ano</sub>th<sub>er</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> <sub>w</sub>h<sub>a</sub>t th<sub>e</sub> <sub>agen</sub>t<sub>s</sub> <sub>s</sub>h<sub>ou</sub>ld d<sub>o.</sub> N<sub>ev-</sub> <sub>er</sub>th<sub>e</sub>l<sub>ess,</sub> i<sub>n</sub> <sub>comp</sub>l<sub>ex</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>ons,</sub> th<sub>ese</sub> <sub>ques</sub>ti<sub>ons</sub> <sub>are</sub> i<sub>nsepara</sub>bl<sub>e.</sub> Th<sub>e</sub> <sub>capa</sub>biliti<sub>es</sub> <sub>o</sub>f th<sub>e</sub> <sub>agen</sub>t<sub>s,</sub> th<sub>e</sub> <sub>amoun</sub>t <sub>o</sub>f <sub>resources,</sub> th<sub>e</sub> <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l d<sub>epen</sub>d<sub>enc</sub>i<sub>es</sub> <sub>among</sub> <sub>su</sub>b<sub>-</sub>t<sub>as</sub>k<sub>s,</sub> <sub>an</sub>d th<sub>e</sub> t<sub>empora</sub>l progression of the mission jointly determine both how the team should be organized and how <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>an</sub>d <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>occur</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>a</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>on.</sub>

H<sub>ere</sub> <sub>we</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> <sub>can</sub> <sub>ena</sub>bl<sub>e</sub> <sub>sca</sub>l<sub>a</sub>bl<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub> i<sub>n</sub> <sub>em-</sub> bodied artificial a<sub>g</sub>ents. We develo<sub>p</sub>ed ORCH (Or<sub>g</sub>anizin<sub>g</sub> Roles and Coordination Hierarchies), a framework that constructs a task-specific organizational hierarchy from a mission objective, the <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s,</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>capa</sub>biliti<sub>es.</sub> Th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> t<sub>wo</sub> <sub>com-</sub> <sub>p</sub>l<sub>emen</sub>t<sub>ary</sub> f<sub>orms</sub> <sub>o</sub>f <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ures.</sub> H<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>poo</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>e-</sub> <sub>pen</sub>d<sub>ence</sub> b<sub>y</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>ng</sub> <sub>wor</sub>k th<sub>a</sub>t <sub>can</sub> b<sub>e</sub> <sub>per</sub>f<sub>orme</sub>d <sub>concurren</sub>tl<sub>y</sub> <sub>an</sub>d i<sub>n</sub>t<sub>egra</sub>ti<sub>ng</sub> <sub>progress</sub> <sub>across</sub> th<sub>e</sub>i<sub>r mem</sub>b<sub>er agen</sub>t<sub>s.</sub> V<sub>er</sub>ti<sub>ca</sub>l <sub>managers coor</sub>di<sub>na</sub>t<sub>e sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence</sub> b<sub>y</sub> d<sub>ecompos</sub>i<sub>ng a</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> <sub>or</sub>d<sub>ere</sub>d <sub>p</sub>h<sub>ases,</sub> <sub>ac</sub>ti<sub>va</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>wor</sub>k <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase,</sub> <sub>an</sub>d <sub>a</sub>d<sub>vanc</sub>i<sub>ng</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> <sub>w</sub>h<sub>en</sub> it<sub>s</sub> <sub>prerequ</sub>i<sub>s</sub>it<sub>es</sub> h<sub>ave</sub> b<sub>een</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d<sub>.</sub> Th<sub>ese</sub> <sub>manager</sub> t<sub>ypes</sub> <sub>can</sub> b<sub>e</sub> <sub>compose</sub>d i<sub>n</sub> <sub>a</sub> h<sub>y</sub>b<sub>r</sub>id <sub>manner</sub> <sub>an</sub>d <sub>recurs</sub>i<sub>ve</sub>l<sub>y,</sub> <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> t<sub>o</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> <sub>s</sub>h<sub>a</sub>ll<sub>ow</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons,</sub> <sub>w</sub>hil<sub>e</sub> enabling more complex missions to form multilevel “teams of teams” (24).

W<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d ORCH <sub>across</sub> 25 <sub>w</sub>ildfi<sub>re-response</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> i<sub>nvo</sub>l<sub>v</sub>i<sub>ng</sub> <sub>reconna</sub>i<sub>ssance,</sub> <sub>rescue,</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>resource</sub> <sub>managemen</sub>t<sub>,</sub> <sub>con</sub>t<sub>a</sub>i<sub>nmen</sub>t<sub>,</sub> <sub>an</sub>d <sub>suppress</sub>i<sub>on.</sub> Th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> <sub>requ</sub>i<sub>re</sub>d h<sub>e</sub>t<sub>-</sub> <sub>erogeneous</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>agen</sub>t<sub>s</sub> <sub>an</sub>d <sub>range</sub>d f<sub>rom</sub> <sub>sma</sub>ll <sub>an</sub>d l<sub>oca</sub>li<sub>ze</sub>d t<sub>as</sub>k<sub>s</sub> t<sub>o</sub> l<sub>ong-</sub>h<sub>or</sub>i<sub>zon</sub> <sub>opera-</sub> ti<sub>ons</sub> i<sub>nvo</sub>l<sub>v</sub>i<sub>ng</sub> <sub>as</sub> <sub>many</sub> <sub>as</sub> 50 <sub>agen</sub>t<sub>s.</sub> W<sub>e</sub> <sub>assesse</sub>d b<sub>o</sub>th h<sub>uman-</sub>d<sub>er</sub>i<sub>ve</sub>d <sub>an</sub>d <sub>au</sub>t<sub>oma</sub>t<sub>e</sub>d l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-genera</sub>t<sub>e</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> f<sub>rom</sub> ORCH’<sub>s</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> <sub>on</sub> <sub>poo</sub>l<sub>e</sub>d <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence.</sub> W<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons</sub> <sub>us</sub>i<sub>ng</sub> <sub>e</sub>i<sub>g</sub>ht l<sub>arge</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>-</sub> els (LLMs) with various model sizes and ca<sub>p</sub>abilities, and com<sub>p</sub>ared them with four re<sub>p</sub>resentative <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>mu</sub>lti<sub>-agen</sub>t <sub>sys</sub>t<sub>em</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms.</sub> O<sub>rgan</sub>i<sub>za</sub>ti<sub>ons</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>aroun</sub>d ORCH’<sub>s</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> i<sub>mprove</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>per</sub>f<sub>ormance,</sub> <sub>execu</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>resource</sub> <sub>use</sub> across the majority of mission scenarios. The gains persisted across missions and underlying lan-<sub>guage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> M<sub>os</sub>t i<sub>mpor</sub>t<sub>an</sub>tl<sub>y,</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>per</sub>f<sub>ormance</sub> <sub>was</sub> <sub>no</sub>t <sub>mono</sub>t<sub>on</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>o</sub> <sub>mo</sub>d<sub>e</sub>l <sub>sca</sub>l<sub>e.</sub> O<sub>ur</sub> fi<sub>n</sub>di<sub>ngs</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>es</sub>i<sub>gn</sub> <sub>as</sub> <sub>a</sub> <sub>core</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>an</sub>d <sub>sc</sub>i<sub>en</sub>tifi<sub>c</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>n</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l i<sub>n</sub>t<sub>e</sub>lli<sub>gence.</sub>

## Th<sub>e</sub> <sub>c</sub>h<sub>a</sub>ll<sub>e</sub>n<sub>ge</sub> <sub>o</sub>f <sub>o</sub>r<sub>ga</sub>ni<sub>z</sub>in<sub>g</sub> <sub>e</sub>mb<sub>o</sub>di<sub>e</sub>d <sub>a</sub>rtifi<sub>c</sub>i<sub>a</sub>l <sub>co</sub>ll<sub>ec</sub>ti<sub>ves</sub>

C<sub>oor</sub>di<sub>na</sub>ti<sub>ng em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>agen</sub>t<sub>s</sub> b<sub>ecomes</sub> i<sub>ncreas</sub>i<sub>ng</sub>l<sub>y</sub> difi<sub>cu</sub>lt <sub>as</sub> t<sub>eams grow</sub> i<sub>n s</sub>i<sub>ze,</sub> di<sub>vers</sub>it<sub>y,</sub> and mission duration (25–27). Each agent contributes a distinct set of capabilities, executions, <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> <sub>a</sub>dditi<sub>ona</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>a</sub>b<sub>ou</sub>t t<sub>as</sub>k <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on,</sub> ti<sub>m</sub>i<sub>ng,</sub> <sub>an</sub>d <sub>resource</sub> <sub>use.</sub> I<sub>n</sub> <sub>a</sub> <sub>sma</sub>ll h<sub>omogeneous</sub> t<sub>eam,</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>manager</sub> <sub>may</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>ass</sub>i<sub>gn</sub> <sub>wor</sub>k t<sub>o</sub> <sub>every</sub> <sub>agen</sub>t<sub>.</sub> H<sub>owever,</sub> i<sub>n</sub> <sub>a</sub> l<sub>arge</sub> h<sub>e</sub>t<sub>erogeneous</sub> t<sub>eam,</sub> th<sub>e</sub> <sub>manager</sub> <sub>mus</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> t<sub>rac</sub>k th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>o</sub>f <sub>many</sub> <sub>wor</sub>k<sub>ers,</sub> <sub>reason</sub> <sub>over</sub> th<sub>e</sub>i<sub>r</sub> <sub>capa</sub>biliti<sub>es,</sub> <sub>preserve</sub> t<sub>empora</sub>l d<sub>epen</sub>d<sub>enc</sub>i<sub>es,</sub> <sub>an</sub>d <sub>rev</sub>i<sub>se</sub> <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>as</sub> th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t <sub>or</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>c</sub>h<sub>anges.</sub>

Additi<sub>ona</sub>ll<sub>y,</sub> th<sub>ese</sub> d<sub>eman</sub>d<sub>s</sub> <sub>are</sub> <sub>no</sub>t <sub>cap</sub>t<sub>ure</sub>d b<sub>y</sub> t<sub>eam</sub> <sub>s</sub>i<sub>ze</sub> <sub>a</sub>l<sub>one.</sub> Th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>o</sub>f th<sub>e</sub> <sub>wor</sub>k i<sub>s equa</sub>ll<sub>y</sub> i<sub>mpor</sub>t<sub>an</sub>t<sub>.</sub> S<sub>ome</sub> t<sub>as</sub>k<sub>s are na</sub>t<sub>ura</sub>ll<sub>y</sub> di<sub>v</sub>i<sub>s</sub>ibl<sub>e.</sub> M<sub>u</sub>lti<sub>p</sub>l<sub>e</sub> d<sub>rones can searc</sub>h dif<sub>eren</sub>t <sub>reg</sub>i<sub>ons or mu</sub>lti<sub>p</sub>l<sub>e</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers can remove</sub> dif<sub>eren</sub>t <sub>o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>es concurren</sub>tl<sub>y.</sub> Oth<sub>er</sub> t<sub>as</sub>k<sub>s con</sub>t<sub>a</sub>i<sub>n</sub> <sub>prerequ</sub>i<sub>s</sub>it<sub>e re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps.</sub> A fi<sub>re mus</sub>t b<sub>e</sub> l<sub>oca</sub>t<sub>e</sub>d b<sub>e</sub>f<sub>ore crews can</sub> b<sub>e</sub> t<sub>ranspor</sub>t<sub>e</sub>d<sub>,</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>ers mus</sub>t fi<sub>n</sub>i<sub>s</sub>h th<sub>e</sub>i<sub>r</sub> <sub>prev</sub>i<sub>ous</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on</sub> t<sub>as</sub>k t<sub>o</sub> <sub>emp</sub>t<sub>y</sub> th<sub>e</sub> l<sub>oa</sub>d b<sub>e</sub>f<sub>ore</sub> t<sub>a</sub>ki<sub>ng</sub> <sub>more</sub> <sub>ass</sub>i<sub>gnmen</sub>t<sub>s,</sub> <sub>an</sub>d <sub>access</sub> <sub>rou</sub>t<sub>es</sub> <sub>nee</sub>d t<sub>o</sub> b<sub>e</sub> id<sub>en</sub>tifi<sub>e</sub>d b<sub>e</sub>f<sub>ore</sub> <sub>groun</sub>d <sub>un</sub>it<sub>s</sub> <sub>can</sub> b<sub>e</sub> d<sub>ep</sub>l<sub>oye</sub>d <sub>sa</sub>f<sub>e</sub>l<sub>y.</sub> C<sub>omp</sub>l<sub>ex</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> <sub>o</sub>ft<sub>en com</sub>bi<sub>ne</sub> b<sub>o</sub>th f<sub>orms.</sub> M<sub>oreover, comp</sub>l<sub>ex m</sub>i<sub>ss</sub>i<sub>ons o</sub>ft<sub>en</sub> i<sub>nvo</sub>l<sub>ve</sub> d<sub>ynam</sub>i<sub>c c</sub>h<sub>anges suc</sub>h <sub>as</sub> <sub>unexpec</sub>t<sub>e</sub>d <sub>env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>c</sub>h<sub>anges,</sub> <sub>agen</sub>t <sub>or</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> l<sub>oss,</sub> <sub>or</sub> <sub>new</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>ecom</sub>i<sub>ng</sub> <sub>gra</sub>d<sub>ua</sub>ll<sub>y ava</sub>il<sub>a</sub>bl<sub>e.</sub> A<sub>n ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>organ</sub>i<sub>za</sub>ti<sub>on mus</sub>t th<sub>ere</sub>f<sub>ore coor</sub>di<sub>na</sub>t<sub>e para</sub>ll<sub>e</sub>l <sub>wor</sub>k <sub>w</sub>ithi<sub>n s</sub>t<sub>ages</sub> <sub>w</sub>hil<sub>e</sub> <sub>preserv</sub>i<sub>ng</sub> d<sub>epen</sub>d<sub>enc</sub>i<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> <sub>s</sub>t<sub>ages</sub> <sub>an</sub>d <sub>a</sub>d<sub>ap</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>overa</sub>ll <sub>coor</sub>di<sub>na</sub>ti<sub>on.</sub>

Fl<sub>a</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>ons an</sub>d <sub>s</sub>i<sub>ng</sub>l<sub>e-manager sys</sub>t<sub>ems</sub> f<sub>ace comp</sub>l<sub>emen</sub>t<sub>ary</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>ons.</sub> I<sub>n a</sub> fl<sub>a</sub>t <sub>organ</sub>i<sub>-</sub> <sub>za</sub>ti<sub>on, every agen</sub>t <sub>may nee</sub>d t<sub>o commun</sub>i<sub>ca</sub>t<sub>e w</sub>ith <sub>many o</sub>th<sub>ers, crea</sub>ti<sub>ng re</sub>d<sub>un</sub>d<sub>an</sub>t <sub>exc</sub>h<sub>anges an</sub>d making it dificult to maintain a coherent global plan (25,28,29). In a single-manager organization, <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> i<sub>s</sub> <sub>cen</sub>t<sub>ra</sub>li<sub>ze</sub>d<sub>,</sub> b<sub>u</sub>t th<sub>e</sub> <sub>manager</sub> <sub>mus</sub>t <sub>reason</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>over</sub> <sub>every</sub> <sub>wor</sub>k<sub>er</sub> <sub>an</sub>d <sub>every</sub> <sub>su</sub>b<sub>-</sub>t<sub>as</sub>k<sub>.</sub> A<sub>s</sub> th<sub>e</sub> t<sub>eam grows,</sub> th<sub>e cen</sub>t<sub>ra</sub>l <sub>manager</sub> b<sub>ecomes respons</sub>ibl<sub>e</sub> f<sub>or an</sub> i<sub>ncreas</sub>i<sub>ng</sub>l<sub>y</sub> l<sub>arge</sub> context, a larger action-allocation problem, and more simultaneous feedback (30–32). A multilevel <sub>organ</sub>i<sub>za</sub>ti<sub>on can re</sub>d<sub>uce</sub> thi<sub>s</sub> b<sub>ur</sub>d<sub>en</sub> b<sub>y</sub> f<sub>orm</sub>i<sub>ng</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e un</sub>it<sub>s</sub> th<sub>a</sub>t i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub> l<sub>oca</sub>l <sub>s</sub>t<sub>a</sub>t<sub>us an</sub>d translate global objectives into tractable sub-goals.

Th<sub>e</sub> <sub>c</sub>h<sub>a</sub>ll<sub>enge</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>no</sub>t <sub>on</sub>l<sub>y</sub> t<sub>o</sub> <sub>pro</sub>d<sub>uce</sub> <sub>a</sub> b<sub>e</sub>tt<sub>er</sub> <sub>p</sub>l<sub>an.</sub> It i<sub>s</sub> t<sub>o</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>an</sub> <sub>organ</sub>i<sub>za-</sub> ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> th<sub>a</sub>t <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on,</sub> th<sub>e</sub> <sub>wor</sub>kf<sub>orce,</sub> <sub>an</sub>d th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps</sub> <sub>among</sub> th<sub>e</sub> <sub>requ</sub>i<sub>re</sub>d <sub>ac</sub>ti<sub>v</sub>iti<sub>es.</sub> Thi<sub>s</sub> <sub>requ</sub>i<sub>res</sub> <sub>reason</sub>i<sub>ng</sub> <sub>a</sub>b<sub>ou</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>agen</sub>t<sub>s</sub> <sub>s</sub>h<sub>ou</sub>ld b<sub>e</sub> <sub>groupe</sub>d<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>managers</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>superv</sub>i<sub>se</sub> th<sub>em,</sub> h<sub>ow</sub> <sub>many</sub> l<sub>eve</sub>l<sub>s</sub> <sub>o</sub>f <sub>managemen</sub>t <sub>are</sub> <sub>warran</sub>t<sub>e</sub>d<sub>,</sub> <sub>an</sub>d <sub>w</sub>h<sub>a</sub>t f<sub>orm</sub> <sub>o</sub>f <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>govern</sub> <sub>eac</sub>h <sub>group.</sub>

## A t<sub>es</sub>tb<sub>e</sub>d f<sub>o</sub>r <sub>o</sub>r<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l int<sub>e</sub>lli<sub>ge</sub>n<sub>ce</sub>

W<sub>e s</sub>t<sub>u</sub>di<sub>e</sub>d <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n an ex</sub>t<sub>en</sub>d<sub>e</sub>d <sub>vers</sub>i<sub>on o</sub>f CREW<sub>-</sub>Wildfi<sub>re, an em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>en-</sub> <sub>v</sub>i<sub>ronmen</sub>t <sub>or</sub>i<sub>g</sub>i<sub>na</sub>ll<sub>y</sub> d<sub>eve</sub>l<sub>ope</sub>d t<sub>o</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> l<sub>arge-sca</sub>l<sub>e</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>among</sub> l<sub>anguage-</sub> model-based agents (33, 34). The original environment introduced procedurally generated wildfire <sub>env</sub>i<sub>ronmen</sub>t<sub>s</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith 12 t<sub>as</sub>k l<sub>eve</sub>l<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> t<sub>ree</sub> <sub>remova</sub>l<sub>,</sub> <sub>scou</sub>ti<sub>ng,</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>c</sub>i<sub>v</sub>ili<sub>an</sub> <sub>rescue,</sub> fi<sub>re suppress</sub>i<sub>on, an</sub>d i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub>d <sub>w</sub>ildfi<sub>re response, w</sub>ith <sub>sma</sub>ll <sub>an</sub>d l<sub>arge var</sub>i<sub>an</sub>t<sub>s y</sub>i<sub>e</sub>ldi<sub>ng</sub> 16 b<sub>enc</sub>h<sub>mar</sub>k <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub> W<sub>e</sub> <sub>expan</sub>d<sub>e</sub>d thi<sub>s</sub> b<sub>enc</sub>h<sub>mar</sub>k t<sub>o</sub> 25 <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> d<sub>es</sub>i<sub>gne</sub>d t<sub>o</sub> <sub>span</sub> <sub>a</sub> b<sub>roa</sub>d<sub>er</sub> <sub>range</sub> <sub>o</sub>f <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>eman</sub>d<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> l<sub>arger</sub> t<sub>eams,</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>wor</sub>kf<sub>orce</sub> <sub>compos</sub>iti<sub>ons,</sub> longer planning horizons, multi-stage objectives, and unexpected changes during execution.

CREW<sub>-</sub>Wildfi<sub>re opera</sub>t<sub>es</sub> i<sub>n proce</sub>d<sub>ura</sub>ll<sub>y genera</sub>t<sub>e</sub>d l<sub>an</sub>d<sub>scapes w</sub>ith <sub>see</sub>d <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> f<sub>or repro</sub> d<sub>uc</sub>ibilit<sub>y.</sub> C<sub>on</sub>ti<sub>nuous</sub> fi<sub>e</sub>ld<sub>s</sub> d<sub>escr</sub>ibi<sub>ng</sub> <sub>e</sub>l<sub>eva</sub>ti<sub>on,</sub> <sub>mo</sub>i<sub>s</sub>t<sub>ure,</sub> <sub>an</sub>d <sub>w</sub>i<sub>n</sub>d <sub>are</sub> <sub>genera</sub>t<sub>e</sub>d t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith di<sub>scre</sub>t<sub>e</sub> t<sub>erra</sub>i<sub>n an</sub>d l<sub>an</sub>d<sub>-cover</sub> t<sub>ypes suc</sub>h <sub>as</sub> f<sub>ores</sub>t<sub>,</sub> b<sub>rus</sub>h<sub>,</sub> l<sub>an</sub>d<sub>, an</sub>d <sub>wa</sub>t<sub>er.</sub> S<sub>e</sub>ttl<sub>emen</sub>t<sub>s an</sub>d <sub>o</sub>th<sub>er</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on-re</sub>l<sub>evan</sub>t f<sub>ea</sub>t<sub>ures</sub> <sub>are</sub> <sub>p</sub>l<sub>ace</sub>d <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> <sub>scenar</sub>i<sub>o</sub> <sub>see</sub>d<sub>.</sub> Th<sub>ese</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> <sub>a</sub>lt<sub>er</sub> th<sub>e</sub> <sub>spa-</sub> ti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f h<sub>azar</sub>d<sub>s an</sub>d <sub>resources an</sub>d <sub>a</sub>f<sub>ec</sub>t th<sub>e evo</sub>l<sub>u</sub>ti<sub>on o</sub>f th<sub>e</sub> fi<sub>re, suc</sub>h th<sub>a</sub>t th<sub>e same</sub> high-level objective can require diferent search, transportation, and intervention strategies.

Th<sub>e</sub> <sub>w</sub>ildfi<sub>re</sub> <sub>evo</sub>l<sub>ves</sub> d<sub>ynam</sub>i<sub>ca</sub>ll<sub>y</sub> th<sub>roug</sub>h <sub>a</sub> <sub>ce</sub>ll<sub>u</sub>l<sub>ar-au</sub>t<sub>oma</sub>t<sub>a</sub> <sub>process</sub> <sub>w</sub>h<sub>ose</sub> l<sub>oca</sub>l <sub>propaga</sub>ti<sub>on</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>con</sub>diti<sub>ons</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> t<sub>erra</sub>i<sub>n</sub> <sub>s</sub>l<sub>ope,</sub> <sub>vege</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>mo</sub>i<sub>s</sub>t<sub>ure,</sub> <sub>an</sub>d <sub>w</sub>i<sub>n</sub>d<sub>.</sub> Fi<sub>re</sub> <sub>can</sub> th<sub>ere</sub>f<sub>ore expan</sub>d i<sub>n</sub> dif<sub>eren</sub>t di<sub>rec</sub>ti<sub>ons an</sub>d <sub>a</sub>t dif<sub>eren</sub>t <sub>ra</sub>t<sub>es as a m</sub>i<sub>ss</sub>i<sub>on un</sub>f<sub>o</sub>ld<sub>s.</sub> F<sub>or</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pro</sub>bl<sub>em</sub> <sub>s</sub>t<sub>u</sub>di<sub>e</sub>d i<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k<sub>,</sub> thi<sub>s</sub> d<sub>ynam</sub>i<sub>c</sub> i<sub>s</sub> <sub>consequen</sub>ti<sub>a</sub>l<sub>.</sub> Ob<sub>serva</sub>ti<sub>ons</sub> <sub>co</sub>ll<sub>ec</sub>t<sub>e</sub>d <sub>ear</sub>l<sub>y</sub> i<sub>n</sub> <sub>an</sub> <sub>ep</sub>i<sub>so</sub>d<sub>e</sub> <sub>can</sub> b<sub>ecome</sub> <sub>ou</sub>td<sub>a</sub>t<sub>e</sub>d<sub>,</sub> <sub>access</sub> <sub>rou</sub>t<sub>es</sub> <sub>can</sub> b<sub>ecome</sub> <sub>unsa</sub>f<sub>e,</sub> <sub>an</sub>d th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>urgency</sub> <sub>o</sub>f <sub>reconna</sub>i<sub>ssance,</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>rescue,</sub> <sub>con</sub>t<sub>a</sub>i<sub>nmen</sub>t<sub>,</sub> <sub>an</sub>d di<sub>rec</sub>t <sub>suppress</sub>i<sub>on</sub> <sub>can</sub> <sub>c</sub>h<sub>ange</sub> <sub>over</sub> ti<sub>me.</sub> T<sub>eams</sub> <sub>nee</sub>d t<sub>o</sub> <sub>consequen</sub>tl<sub>y</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>no</sub>t <sub>on</sub>l<sub>y</sub> <sub>across</sub> <sub>space</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> <sub>across</sub> <sub>an</sub> <sub>evo</sub>l<sub>v</sub>i<sub>ng</sub> <sub>p</sub><sup>h</sup><sub>y</sub>s<sup>i</sup>ca<sup>l</sup> <sub>p</sub>rocess.

CREW<sub>-</sub>Wildfi<sub>re</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> f<sub>our</sub> <sub>wor</sub>k<sub>er</sub> t<sub>ypes</sub> <sub>w</sub>ith <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>capa</sub>biliti<sub>es.</sub> Fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>are</sub> <sub>genera</sub>l<sub>-purpose</sub> <sub>groun</sub>d <sub>agen</sub>t<sub>s</sub> th<sub>a</sub>t <sub>can</sub> <sub>remove</sub> <sub>vege</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>rescue</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans,</sub> <sub>re</sub>fill <sub>wa</sub>t<sub>er,</sub> <sub>an</sub>d di<sub>rec</sub>tl<sub>y</sub> <sub>suppress</sub> fi<sub>re.</sub> B<sub>u</sub>lld<sub>ozers</sub> <sub>c</sub>l<sub>ear</sub> <sub>vege</sub>t<sub>a</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> b<sub>u</sub>t <sub>canno</sub>t <sub>rescue</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans</sub> <sub>or</sub> <sub>ex</sub>ti<sub>ngu</sub>i<sub>s</sub>h fi<sub>res.</sub> D<sub>rones</sub> <sub>prov</sub>id<sub>e</sub> <sub>rap</sub>id <sub>w</sub>id<sub>e-area</sub> <sub>reconna</sub>i<sub>ssance</sub> b<sub>u</sub>t l<sub>ac</sub>k <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>capa</sub>biliti<sub>es.</sub> H<sub>e</sub>li<sub>-</sub> <sub>cop</sub>t<sub>ers</sub> <sub>prov</sub>id<sub>e</sub> l<sub>ong-range</sub> <sub>mo</sub>bilit<sub>y,</sub> <sub>can</sub> t<sub>ranspor</sub>t fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers,</sub> <sub>an</sub>d <sub>suppor</sub>t <sub>wa</sub>t<sub>er-</sub>b<sub>ase</sub>d <sub>opera</sub>ti<sub>ons.</sub>

W<sub>e</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>ex</sub>t<sub>en</sub>d<sub>e</sub>d th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l CREW<sub>-</sub>Wildfi<sub>re</sub> t<sub>as</sub>k <sub>su</sub>it<sub>e</sub> t<sub>o</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>a</sub> t<sub>o</sub>t<sub>a</sub>l <sub>o</sub>f 25 <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> <sub>rang</sub>i<sub>ng</sub> f<sub>rom</sub> th<sub>ree-agen</sub>t<sub>,</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-ro</sub>l<sub>e</sub> t<sub>as</sub>k<sub>s</sub> t<sub>o</sub> l<sub>arge-sca</sub>l<sub>e</sub> <sub>scenar</sub>i<sub>os</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> 50 <sub>wor</sub>k<sub>ers.</sub> Thi<sub>s pro</sub>d<sub>uces a progress</sub>i<sub>on</sub> f<sub>rom</sub> t<sub>as</sub>k<sub>s</sub> th<sub>a</sub>t <sub>can</sub> l<sub>arge</sub>l<sub>y</sub> b<sub>e so</sub>l<sub>ve</sub>d b<sub>y</sub> di<sub>v</sub>idi<sub>ng</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>wor</sub>k t<sub>o</sub> t<sub>as</sub>k<sub>s</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> <sub>severa</sub>l <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>groups</sub> t<sub>o</sub> <sub>exc</sub>h<sub>ange</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> <sub>resources,</sub> <sub>an</sub>d <sub>con</sub>t<sub>ro</sub>l <sub>over</sub> l<sub>ong</sub> t<sub>empora</sub>l h<sub>or</sub>i<sub>zons.</sub> Th<sub>e ex</sub>t<sub>en</sub>d<sub>e</sub>d <sub>su</sub>it<sub>e a</sub>l<sub>so</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces exp</sub>li<sub>c</sub>it d<sub>ynam</sub>i<sub>ca</sub>l <sub>c</sub>h<sub>anges</sub> d<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> t<sub>o eva</sub>l<sub>ua</sub>t<sub>e w</sub>h<sub>e</sub>th<sub>er a ro</sub>b<sub>us</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>on can respon</sub>d <sub>w</sub>h<sub>en</sub> it<sub>s or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>p</sub>l<sub>ans</sub> b<sub>ecome</sub> <sub>o</sub>b<sub>so</sub>l<sub>e</sub>t<sub>e.</sub> I<sub>n</sub> dif<sub>eren</sub>t <sub>m</sub>i<sub>ss</sub>i<sub>ons, a reconna</sub>i<sub>ssance</sub> d<sub>rone can</sub> b<sub>ecome unava</sub>il<sub>a</sub>bl<sub>e, a</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>er can</sub> f<sub>a</sub>il d<sub>ur</sub>i<sub>ng</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>prev</sub>i<sub>ous</sub>l<sub>y</sub> <sub>un</sub>k<sub>nown</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans</sub> <sub>can</sub> <sub>appear,</sub> <sub>a</sub> <sub>secon</sub>d fi<sub>re</sub> <sub>can</sub> b<sub>rea</sub>k <sub>ou</sub>t<sub>,</sub> <sub>a</sub> <sub>wa</sub>t<sub>er</sub> <sub>source</sub> <sub>can</sub> b<sub>ecome</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>a</sub>ft<sub>er</sub> <sub>execu</sub>ti<sub>on</sub> h<sub>as</sub> b<sub>egun,</sub> <sub>or</sub> th<sub>e</sub> fi<sub>re</sub> <sub>can</sub> <sub>en</sub>t<sub>er</sub> <sub>a</sub> <sub>per</sub>i<sub>o</sub>d <sub>o</sub>f rapid growth. Other missions deliberately chain multiple objectives. Agents may need to search for <sub>c</sub>i<sub>v</sub>ili<sub>ans</sub> b<sub>e</sub>f<sub>ore</sub> <sub>rescu</sub>i<sub>ng</sub> <sub>an</sub>d t<sub>ranspor</sub>ti<sub>ng</sub> th<sub>em,</sub> <sub>or</sub> l<sub>oca</sub>t<sub>e</sub> <sub>a</sub> fi<sub>re</sub> b<sub>e</sub>f<sub>ore</sub> d<sub>ep</sub>l<sub>oy</sub>i<sub>ng</sub> h<sub>e</sub>t<sub>erogeneous</sub> units for containment and suppression. The most complex condition combines all major wildfireresponse objectives within a single mission. The detailed information for each task is provided in S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> T<sub>a</sub>bl<sub>e</sub> S1 <sub>an</sub>d P<sub>romp</sub>t S1 i<sub>n</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> T<sub>ex</sub>t<sub>.</sub>

Th<sub>ese ex</sub>t<sub>ens</sub>i<sub>ons were</sub> d<sub>es</sub>i<sub>gne</sub>d t<sub>o ma</sub>k<sub>e</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or s</sub>t<sub>u</sub>d<sub>y</sub>i<sub>ng orga-</sub> <sub>n</sub>i<sub>za</sub>ti<sub>on.</sub> F<sub>or examp</sub>l<sub>e, a sparse</sub> t<sub>ree-remova</sub>l <sub>m</sub>i<sub>ss</sub>i<sub>on pr</sub>i<sub>mar</sub>il<sub>y</sub> t<sub>es</sub>t<sub>s w</sub>h<sub>e</sub>th<sub>er wor</sub>k <sub>can</sub> b<sub>e par</sub>titi<sub>one</sub>d <sub>among</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>agen</sub>t<sub>s.</sub> A h<sub>e</sub>li<sub>cop</sub>t<sub>er-</sub>fi<sub>re</sub>fi<sub>g</sub>ht<sub>er</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>requ</sub>i<sub>res</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>ro</sub>l<sub>es</sub> <sub>an</sub>d <sub>sync</sub>h<sub>ron</sub>i<sub>ze</sub>d h<sub>an</sub>d<sub>o</sub>f<sub>s.</sub> S<sub>earc</sub>h<sub>-an</sub>d<sub>-rescue</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> <sub>ma</sub>k<sub>e</sub> l<sub>a</sub>t<sub>er</sub> <sub>ac</sub>ti<sub>ons</sub> d<sub>epen</sub>d <sub>on</sub> i<sub>n</sub>f<sub>or-</sub> <sub>ma</sub>ti<sub>on</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d d<sub>ur</sub>i<sub>ng</sub> <sub>reconna</sub>i<sub>ssance.</sub> Fi<sub>re</sub> <sub>con</sub>t<sub>a</sub>i<sub>nmen</sub>t <sub>com</sub>bi<sub>nes</sub> <sub>groun</sub>d <sub>wor</sub>k<sub>ers</sub> <sub>w</sub>ith dif<sub>eren</sub>t i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>capa</sub>biliti<sub>es.</sub> Th<sub>e</sub> <sub>mos</sub>t <sub>comp</sub>l<sub>ex</sub> <sub>suppress</sub>i<sub>on</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> <sub>requ</sub>i<sub>re</sub> <sub>reconna</sub>i<sub>ssance,</sub> <sub>resource</sub> <sub>acqu</sub>i<sub>s</sub>iti<sub>on,</sub> <sub>rescue,</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> d<sub>ep</sub>l<sub>oymen</sub>t<sub>,</sub> fi<sub>re</sub>b<sub>rea</sub>k <sub>cons</sub>t<sub>ruc</sub>ti<sub>on,</sub> <sub>an</sub>d di<sub>rec</sub>t <sub>suppress</sub>i<sub>on</sub> t<sub>o</sub> b<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d <sub>over</sub> ti<sub>me.</sub> Th<sub>us,</sub> th<sub>e</sub> 25 t<sub>as</sub>k<sub>s</sub> <sub>vary</sub> <sub>no</sub>t <sub>on</sub>l<sub>y</sub> i<sub>n</sub> <sub>conven</sub>ti<sub>ona</sub>l difi<sub>cu</sub>lt<sub>y</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> i<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure o</sub>f <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> th<sub>ey</sub> d<sub>eman</sub>d<sub>.</sub>

## Or<sub>g</sub>anizin<sub>g</sub> a<sub>g</sub>ents thro<sub>ug</sub>h <sub>p</sub>ooled and se<sub>qu</sub>ential interde<sub>p</sub>endence

H<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> dif<sub>eren</sub>t f<sub>orms</sub> <sub>o</sub>f <sub>wor</sub>k i<sub>n</sub> dif<sub>eren</sub>t <sub>ways.</sub> T<sub>as</sub>k<sub>s</sub> th<sub>a</sub>t <sub>can</sub> <sub>procee</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>are</sub> <sub>o</sub>ft<sub>en</sub> <sub>manage</sub>d i<sub>n</sub> <sub>para</sub>ll<sub>e</sub>l<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> t<sub>as</sub>k<sub>s</sub> <sub>w</sub>ith <sub>prerequ</sub>i<sub>s</sub>it<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps</sub> <sub>requ</sub>i<sub>re</sub> <sub>or</sub>d<sub>ere</sub>d <sub>coor</sub>di<sub>na</sub>ti<sub>on across s</sub>t<sub>ages.</sub> ORCH i<sub>s</sub> b<sub>ase</sub>d <sub>on</sub> thi<sub>s prem</sub>i<sub>se</sub> th<sub>a</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>re</sub>fl<sub>ec</sub>t th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps</sub> <sub>among</sub> <sub>ac</sub>ti<sub>v</sub>iti<sub>es.</sub> W<sub>e</sub> <sub>opera</sub>ti<sub>ona</sub>li<sub>ze</sub>d t<sub>wo</sub> f<sub>orms</sub> <sub>o</sub>f i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence</sub> f<sub>rom</sub> h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory:</sub> <sub>poo</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence.</sub>

P<sub>oo</sub>l<sub>e</sub>d int<sub>e</sub>rd<sub>epe</sub>nd<sub>e</sub>n<sub>ce.</sub> U<sub>n</sub>d<sub>er</sub> <sub>poo</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence,</sub> <sub>mem</sub>b<sub>ers</sub> <sub>ma</sub>k<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u-</sub> ti<sub>ons</sub> t<sub>o a s</sub>h<sub>are</sub>d <sub>ou</sub>t<sub>come.</sub> Th<sub>e</sub>i<sub>r wor</sub>k <sub>can procee</sub>d <sub>concurren</sub>tl<sub>y, an</sub>d th<sub>e organ</sub>i<sub>za</sub>ti<sub>on mus</sub>t di<sub>v</sub>id<sub>e</sub> the objective, prevent unnecessary duplication, and integrate progress across members. In a tree-<sub>cu</sub>tti<sub>ng</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on,</sub> f<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> dif<sub>eren</sub>t fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>or</sub> b<sub>u</sub>lld<sub>ozers</sub> <sub>can</sub> b<sub>e</sub> <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> dif<sub>eren</sub>t t<sub>arge</sub>t<sub>s</sub> (Fi<sub>g</sub>ure 1C). In a reconnaissance mission, drones can search diferent re<sub>g</sub>ions of the ma<sub>p</sub>. ORCH <sub>represen</sub>t<sub>s</sub> thi<sub>s</sub> f<sub>orm</sub> <sub>o</sub>f <sub>wor</sub>k <sub>w</sub>ith <sub>a</sub> “h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub>”<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> <sub>concurren</sub>t <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>across</sub> it<sub>s c</sub>hild<sub>ren an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>ose ass</sub>i<sub>gnmen</sub>t<sub>s</sub> f<sub>rom</sub> th<sub>e</sub>i<sub>r repor</sub>t<sub>s.</sub>

For a horizontal mana<sub>g</sub>er with mission �, children C, estimated <sub>p</sub>ro<sub>g</sub>ress �, and re<sub>p</sub>orts �, the t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>are</sub> <sub>represen</sub>t<sub>e</sub>d <sub>as</sub>

$$
{ \mathcal { T } } _ { h } = \{ ( c _ { i } , t _ { i } ) \mid c _ { i } \in C \} = f _ { h } ( M , P , R ) ,\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> $t _ { i }$ i<sub>s</sub> th<sub>e</sub> <sub>ass</sub>i<sub>gnmen</sub>t <sub>g</sub>i<sub>ven</sub> t<sub>o</sub> <sub>c</sub>hild $c _ { i } .$ <sub>.</sub> Th<sub>e</sub> <sub>c</sub>hild<sub>ren</sub> <sub>may</sub> b<sub>e</sub> <sub>wor</sub>k<sub>ers</sub> <sub>or</sub> l<sub>ower-</sub>l<sub>eve</sub>l <sub>managers,</sub> <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>poo</sub>l<sub>e</sub>d <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> t<sub>o</sub> <sub>opera</sub>t<sub>e</sub> <sub>a</sub>t <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>sca</sub>l<sub>es.</sub>

S<sub>eque</sub>nti<sub>a</sub>l int<sub>e</sub>rd<sub>epe</sub>nd<sub>e</sub>n<sub>ce.</sub> U<sub>n</sub>d<sub>er</sub> <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence,</sub> th<sub>e</sub> <sub>ou</sub>t<sub>pu</sub>t <sub>o</sub>f <sub>one</sub> <sub>ac</sub>ti<sub>v</sub>it<sub>y</sub> b<sub>e-</sub> <sub>comes</sub> <sub>a</sub> <sub>prerequ</sub>i<sub>s</sub>it<sub>e</sub> f<sub>or</sub> th<sub>e</sub> <sub>nex</sub>t<sub>.</sub> A<sub>gen</sub>t<sub>s</sub> <sub>canno</sub>t <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>per</sub>f<sub>orm</sub> <sub>a</sub>ll <sub>su</sub>b<sub>-</sub>t<sub>as</sub>k<sub>s</sub> i<sub>n</sub> <sub>para</sub>ll<sub>e</sub>l<sub>.</sub> Th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on nee</sub>d<sub>s</sub> t<sub>o represen</sub>t <sub>m</sub>il<sub>es</sub>t<sub>ones,</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne w</sub>h<sub>en eac</sub>h h<sub>as</sub> b<sub>een ac</sub>hi<sub>eve</sub>d<sub>, an</sub>d <sub>coor</sub>di<sub>-</sub> <sub>na</sub>t<sub>e</sub> h<sub>an</sub>d<sub>o</sub>f<sub>s.</sub> I<sub>n</sub> <sub>w</sub>ildfi<sub>re</sub> <sub>response,</sub> d<sub>rones</sub> <sub>or</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>may</sub> fi<sub>rs</sub>t l<sub>oca</sub>t<sub>e</sub> th<sub>e</sub> fi<sub>re</sub> <sub>an</sub>d id<sub>en</sub>tif<sub>y</sub> <sub>sa</sub>f<sub>e</sub> <sub>access</sub> <sub>rou</sub>t<sub>es.</sub> H<sub>e</sub>li<sub>cop</sub>t<sub>ers</sub> <sub>can</sub> th<sub>en</sub> <sub>co</sub>ll<sub>ec</sub>t <sub>wa</sub>t<sub>er</sub> <sub>or</sub> t<sub>ranspor</sub>t <sub>personne</sub>l<sub>,</sub> <sub>a</sub>ft<sub>er</sub> <sub>w</sub>hi<sub>c</sub>h fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>an</sub>d bulldozers can construct firebreaks and su<sub>pp</sub>ress the fire (Fi<sub>g</sub>ure 1D).

ORCH <sub>represen</sub>t<sub>s</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>wor</sub>k <sub>w</sub>ith <sub>a</sub> “<sub>ver</sub>ti<sub>ca</sub>l <sub>manager</sub>”<sub>.</sub> R<sub>a</sub>th<sub>er</sub> th<sub>an</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> <sub>ass</sub>i<sub>gn</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>wor</sub>k<sub>,</sub> <sub>a</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>manager</sub> d<sub>ecomposes</sub> it<sub>s</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> <sub>an</sub> <sub>or</sub>d<sub>ere</sub>d <sub>sequence</sub> <sub>o</sub>f <sub>p</sub>h<sub>ases,</sub>

$$
\Phi = \{ \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { K } \} = f _ { \nu } ( M , P _ { m } , R ) ,\tag{2}
$$

<sub>an</sub>d <sub>genera</sub>t<sub>es ass</sub>i<sub>gnmen</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e ac</sub>ti<sub>ve p</sub>h<sub>ase</sub> $\phi _ { k }$

$$
\mathcal { T } _ { \nu } ^ { ( k ) } = \{ ( c _ { i } , t _ { i } ^ { ( k ) } ) ~ | ~ c _ { i } \in C \} = g _ { \nu } ( \phi _ { k } , P _ { p } , R ) ,\tag{3}
$$

<sub>w</sub>h<sub>ere</sub> $P _ { m }$ <sub>an</sub>d $P _ { p }$ d<sub>eno</sub>t<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>an</sub>d <sub>p</sub>h<sub>ase</sub> <sub>progress.</sub> Th<sub>e</sub> <sub>manager</sub> <sub>a</sub>d<sub>vances</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>con</sub>diti<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> <sub>are</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d<sub>.</sub>

Th<sub>e</sub> t<sub>wo manager</sub> t<sub>ypes are comp</sub>l<sub>emen</sub>t<sub>ary.</sub> A <sub>ver</sub>ti<sub>ca</sub>l <sub>manager can coor</sub>di<sub>na</sub>t<sub>e</sub> th<sub>e or</sub>d<sub>ere</sub>d <sub>progress</sub>i<sub>on o</sub>f <sub>a m</sub>i<sub>ss</sub>i<sub>on w</sub>hil<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers organ</sub>i<sub>ze concurren</sub>t <sub>wor</sub>k <sub>w</sub>ithi<sub>n eac</sub>h <sub>p</sub>h<sub>ase.</sub>

C<sub>onverse</sub>l<sub>y,</sub> <sub>a</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub> <sub>may</sub> <sub>superv</sub>i<sub>se</sub> <sub>severa</sub>l <sub>su</sub>b<sub>-</sub>t<sub>eams,</sub> <sub>some</sub> <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h <sub>con</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub>i<sub>r</sub> own vertical and horizontal mana<sub>g</sub>ers (Fi<sub>g</sub>ure 1E). Therefore, ORCH’s concrete or<sub>g</sub>anizational <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>can</sub> <sub>ena</sub>bl<sub>e</sub> <sub>sca</sub>l<sub>a</sub>bl<sub>e</sub> <sub>an</sub>d <sub>comp</sub>l<sub>ex</sub> <sub>recurs</sub>i<sub>ve</sub> <sub>compos</sub>iti<sub>on</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>ows</sub> th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> t<sub>o</sub> <sub>express</sub> th<sub>e</sub> <sub>m</sub>i<sub>x</sub>t<sub>ure</sub> <sub>o</sub>f <sub>para</sub>ll<sub>e</sub>l <sub>an</sub>d <sub>or</sub>d<sub>ere</sub>d <sub>wor</sub>k <sub>common</sub>l<sub>y</sub> f<sub>oun</sub>d i<sub>n</sub> <sub>comp</sub>l<sub>ex</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons.</sub>

## T<sub>as</sub>k<sub>-spec</sub>ifi<sub>c o</sub>r<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>hi<sub>es</sub>

Gi<sub>ven</sub> <sub>a</sub> t<sub>as</sub>k d<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub> �<sub>,</sub> <sub>a</sub> <sub>se</sub>t <sub>o</sub>f <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>wor</sub>k<sub>ers</sub> �<sub>,</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>capa</sub>biliti<sub>es</sub> �<sub>,</sub> ORCH <sub>cons</sub>t<sub>ruc</sub>t<sub>s</sub> <sub>a</sub> <sub>roo</sub>t<sub>e</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l hi<sub>erarc</sub>h<sub>y,</sub> <sub>represen</sub>t<sub>e</sub>d b<sub>y</sub> <sub>a</sub> t<sub>ree,</sub>

$$
G = { \mathcal { H } } ( T , W , C ) ,\tag{4}
$$

<sub>w</sub>h<sub>ose</sub> l<sub>ea</sub>f <sub>no</sub>d<sub>es correspon</sub>d t<sub>o</sub> th<sub>e ava</sub>il<sub>a</sub>bl<sub>e em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>wor</sub>k<sub>ers an</sub>d <sub>o</sub>th<sub>er no</sub>d<sub>es correspon</sub>d t<sub>o</sub> <sub>managers.</sub> Th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> <sub>can</sub> b<sub>e</sub> d<sub>es</sub>i<sub>gne</sub>d b<sub>y</sub> <sub>a</sub> h<sub>uman</sub> <sub>exper</sub>t <sub>or</sub> <sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> <sub>a</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l (Fi<sub>g</sub>ure 1A).

E<sub>ac</sub>h hi<sub>erarc</sub>hi<sub>ca</sub>l t<sub>eam organ</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>s</sub> t<sub>as</sub>k<sub>-spec</sub>ifi<sub>c.</sub> Si<sub>mp</sub>l<sub>e m</sub>i<sub>ss</sub>i<sub>ons w</sub>ith <sub>a sma</sub>ll <sub>num</sub>b<sub>er o</sub>f <sub>wor</sub>k<sub>ers can re</sub>t<sub>a</sub>i<sub>n a s</sub>h<sub>a</sub>ll<sub>ow s</sub>t<sub>ruc</sub>t<sub>ure</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>one manager superv</sub>i<sub>ses a</sub>ll <sub>wor</sub>k<sub>ers.</sub> L<sub>arger m</sub>i<sub>ss</sub>i<sub>ons</sub> <sub>can pro</sub>d<sub>uce a</sub> t<sub>eam o</sub>f t<sub>eams</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>a roo</sub>t <sub>manager superv</sub>i<sub>ses severa</sub>l i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e managers, eac</sub>h <sub>respons</sub>ibl<sub>e</sub> f<sub>or</sub> <sub>a</sub> <sub>co</sub>h<sub>eren</sub>t <sub>group</sub> <sub>o</sub>f <sub>su</sub>b<sub>-</sub>t<sub>eams</sub> <sub>or</sub> <sub>su</sub>b<sub>-goa</sub>l<sub>s.</sub> Th<sub>e</sub> <sub>genera</sub>ti<sub>on</sub> <sub>process</sub> i<sub>s</sub> <sub>encourage</sub>d t<sub>o</sub> k<sub>eep</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> <sub>as</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> <sub>as</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>w</sub>hil<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>a</sub>dditi<sub>ona</sub>l d<sub>ecompos</sub>iti<sub>on</sub> <sub>w</sub>h<sub>en</sub> <sub>one</sub> <sub>manager</sub> <sub>wou</sub>ld <sub>o</sub>th<sub>erw</sub>i<sub>se</sub> <sub>nee</sub>d t<sub>o</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> t<sub>oo</sub> <sub>many</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>wor</sub>k<sub>ers</sub> <sub>or</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> contains distinct interde<sub>p</sub>endent sub-<sub>g</sub>oals (Prom<sub>p</sub>t S2 in Su<sub>pp</sub>lementar<sub>y</sub> Text).

F<sub>or</sub> <sub>au</sub>t<sub>oma</sub>t<sub>e</sub>d <sub>genera</sub>ti<sub>on,</sub> <sub>a</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l fi<sub>rs</sub>t <sub>proposes</sub> th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>an</sub>d <sub>wor</sub>kf<sub>orce.</sub> A <sub>cr</sub>iti<sub>c mo</sub>d<sub>e</sub>l <sub>eva</sub>l<sub>ua</sub>t<sub>es w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e s</sub>t<sub>ruc</sub>t<sub>ure con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e correc</sub>t <sub>num</sub>b<sub>er o</sub>f <sub>wor</sub>k<sub>ers,</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>agen</sub>t <sub>capa</sub>biliti<sub>es</sub> <sub>are</sub> <sub>ma</sub>t<sub>c</sub>h<sub>e</sub>d t<sub>o</sub> <sub>respons</sub>ibiliti<sub>es,</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>unnecessary</sub> l<sub>ayers</sub> <sub>or</sub> <sub>groups</sub> have been introduced, and whether the proposed groups correspond to sub-goals that require joint <sub>coor</sub>di<sub>na</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>proposa</sub>l <sub>w</sub>ill th<sub>en</sub> b<sub>e</sub> <sub>rev</sub>i<sub>se</sub>d i<sub>n</sub> <sub>response</sub> t<sub>o</sub> thi<sub>s</sub> <sub>cr</sub>iti<sub>que</sub> <sub>un</sub>til <sub>convergence</sub> <sub>or</sub> ti<sub>me ou</sub>t<sub>.</sub> Thi<sub>s cr</sub>iti<sub>c-gu</sub>id<sub>e</sub>d d<sub>es</sub>i<sub>gn can cons</sub>t<sub>ra</sub>i<sub>n or prune</sub> th<sub>e</sub> LLM<sub>-propose</sub>d t<sub>eams.</sub> Thi<sub>s</sub> i<sub>s</sub> b<sub>ecause</sub> h<sub>av</sub>i<sub>ng</sub> LLM<sub>s</sub> <sub>propose</sub> t<sub>eam</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons</sub> f<sub>or</sub> <sub>comp</sub>l<sub>ex</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> it<sub>se</sub>lf <sub>can</sub> b<sub>e</sub> <sub>c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng.</sub> I<sub>n</sub> <sub>suc</sub>h <sub>cases,</sub> <sub>we</sub> f<sub>oun</sub>d th<sub>a</sub>t th<sub>e</sub> <sub>secon</sub>d<sub>ary</sub> <sub>cr</sub>iti<sub>c</sub> d<sub>es</sub>i<sub>gn</sub> <sub>w</sub>ith it<sub>era</sub>ti<sub>ve</sub> <sub>reason</sub>i<sub>ng</sub> <sub>an</sub>d <sub>re</sub>fi<sub>nemen</sub>t h<sub>e</sub>l<sub>ps</sub> <sub>grea</sub>tl<sub>y</sub> im<sub>p</sub>rove the correctness, eficienc<sub>y</sub>, and efectiveness of the LLM-<sub>g</sub>enerated teams (Prom<sub>p</sub>t S3–S4 in Su<sub>pp</sub>lementar<sub>y</sub> Text and Fi<sub>g</sub>ure 5A).

O<sub>nce</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d<sub>,</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>rema</sub>i<sub>ns</sub> <sub>a</sub> <sub>sca</sub>f<sub>o</sub>ld f<sub>or</sub> <sub>execu</sub>ti<sub>on.</sub> O<sub>ne</sub> <sub>can</sub> <sub>po-</sub> t<sub>en</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>exp</sub>l<sub>ore</sub> d<sub>ynam</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>c</sub>h<sub>ang</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> d<sub>ur</sub>i<sub>ng</sub> t<sub>as</sub>k <sub>progress</sub>i<sub>on.</sub> Thi<sub>s</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> t<sub>a</sub>k<sub>es</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>s</sub>t<sub>ep</sub> t<sub>owar</sub>d b<sub>a</sub>ki<sub>ng</sub> h<sub>uman organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory, so we</sub> l<sub>eave suc</sub>h <sub>exp</sub>l<sub>ora</sub>ti<sub>on as</sub> f<sub>u</sub>t<sub>ure wor</sub>k<sub>.</sub> Th<sub>oug</sub>h th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> th<sub>e</sub> <sub>g</sub>i<sub>ven</sub> t<sub>as</sub>k <sub>upon</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on,</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>ve</sub> <sub>p</sub>l<sub>an,</sub> <sub>p</sub>h<sub>ase,</sub> <sub>an</sub>d <sub>as-</sub> <sub>s</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>can</sub> <sub>c</sub>h<sub>ange</sub> <sub>as</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>progresses.</sub> Th<sub>us,</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>execu</sub>ti<sub>on</sub> <sub>can</sub> <sub>s</sub>till <sub>occur</sub> th<sub>roug</sub>h th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons, an</sub>d <sub>consequences</sub> th<sub>a</sub>t <sub>move</sub> th<sub>roug</sub>h th<sub>e organ</sub>i<sub>za</sub>ti<sub>on.</sub>

## Cl<sub>ose</sub>d<sub>-</sub>l<sub>oop execu</sub>ti<sub>on</sub> th<sub>roug</sub>h hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub>

T<sub>o</sub> k<sub>eep</sub> <sub>a</sub> t<sub>eam</sub> <sub>o</sub>f t<sub>eams</sub> <sub>a</sub>li<sub>gne</sub>d <sub>as</sub> th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t <sub>an</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>evo</sub>l<sub>ve,</sub> ORCH <sub>con</sub>ti<sub>nuous</sub>l<sub>y</sub> <sub>exc</sub>h<sub>anges</sub> l<sub>oca</sub>l <sub>progress</sub> <sub>upwar</sub>d <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub>d d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> d<sub>ownwar</sub>d th<sub>roug</sub>h th<sub>e</sub> hi<sub>erarc</sub>h<sub>y.</sub> It <sub>a</sub>lt<sub>er-</sub> <sub>na</sub>t<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> <sub>a</sub> b<sub>o</sub>tt<sub>om-up</sub> <sub>s</sub>t<sub>a</sub>t<sub>us</sub> <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d <sub>a</sub> t<sub>op-</sub>d<sub>own</sub> <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>p</sub>h<sub>ase</sub> <sub>a</sub>t <sub>eac</sub>h <sub>env</sub>i<sub>ronmen</sub>t <sub>s</sub>t<sub>ep</sub> (Fi<sub>g</sub>ure 1B).

D<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>o</sub>tt<sub>om-up</sub> <sub>p</sub>h<sub>ase,</sub> <sub>eac</sub>h <sub>wor</sub>k<sub>er</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>s</sub> it<sub>s</sub> l<sub>oca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>an</sub>d <sub>ass</sub>i<sub>gnmen</sub>t <sub>an</sub>d <sub>pro</sub>d<sub>uces</sub> <sub>a</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>repor</sub>t <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d <sub>wor</sub>k<sub>,</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>us,</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>comp</sub>l<sub>e</sub>ti<sub>on,</sub> and ur<sub>g</sub>ent information (Prom<sub>p</sub>t S6 in Su<sub>pp</sub>lementar<sub>y</sub> Text). Re<sub>p</sub>orts are then a<sub>gg</sub>re<sub>g</sub>ated u<sub>p</sub>ward th<sub>roug</sub>h th<sub>e</sub> hi<sub>erarc</sub>h<sub>y.</sub> E<sub>ac</sub>h <sub>manager com</sub>bi<sub>nes</sub> th<sub>e repor</sub>t<sub>s o</sub>f it<sub>s c</sub>hild<sub>ren</sub> t<sub>o es</sub>ti<sub>ma</sub>t<sub>e progress a</sub>t it<sub>s</sub> <sub>own</sub> l<sub>eve</sub>l <sub>an</sub>d t<sub>o</sub> d<sub>ec</sub>id<sub>e</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>l<sub>an</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>con</sub>ti<sub>nue,</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>s</sub>h<sub>ou</sub>ld b<sub>e</sub> revised, whether an additional <sub>p</sub>hase is needed (for vertical mana<sub>g</sub>ers), or whether the mission is com<sub>p</sub>lete (Prom<sub>p</sub>t S5 in Su<sub>pp</sub>lementar<sub>y</sub> Text).

D<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>op-</sub>d<sub>own</sub> <sub>p</sub>h<sub>ase,</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>propaga</sub>t<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> <sub>roo</sub>t t<sub>owar</sub>d th<sub>e</sub> l<sub>eaves.</sub> H<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> di<sub>rec</sub>tl<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> <sub>concurren</sub>t <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>among</sub> th<sub>e</sub>i<sub>r</sub> <sub>c</sub>hild<sub>ren.</sub> V<sub>er</sub>ti<sub>ca</sub>l <sub>managers</sub> fi<sub>rs</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>ve</sub> <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d th<sub>e</sub> <sub>sequence</sub> <sub>o</sub>f f<sub>u</sub>t<sub>ure</sub> <sub>p</sub>h<sub>ases,</sub> <sub>an</sub>d th<sub>en</sub> <sub>ass</sub>i<sub>gn</sub> <sub>wor</sub>k <sub>nee</sub>d<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase.</sub> B<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>p</sub>l<sub>an</sub> i<sub>s</sub> fi<sub>na</sub>li<sub>ze</sub>d<sub>,</sub> <sub>c</sub>hild<sub>ren</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub>i<sub>r</sub> <sub>ass</sub>i<sub>gne</sub>d <sub>ro</sub>l<sub>es</sub> <sub>are</sub> feasible and consistent with their capabilities. Actionable objections can be returned to the manager, <sub>w</sub>h<sub>o</sub> <sub>rev</sub>i<sub>ses</sub> th<sub>e</sub> <sub>p</sub>l<sub>an</sub> <sub>un</sub>til <sub>no</sub> f<sub>ur</sub>th<sub>er</sub> <sub>con</sub>fli<sub>c</sub>t i<sub>s</sub> id<sub>en</sub>tifi<sub>e</sub>d<sub>.</sub> W<sub>or</sub>k<sub>ers</sub> th<sub>en</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> <sub>accep</sub>t<sub>e</sub>d <sub>ass</sub>i<sub>gnmen</sub>t<sub>s an</sub>d <sub>curren</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> i<sub>n</sub>t<sub>o env</sub>i<sub>ronmen</sub>t <sub>ac</sub>ti<sub>ons, w</sub>hi<sub>c</sub>h <sub>are execu</sub>t<sub>e</sub>d <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y.</sub>

Thi<sub>s</sub> bi<sub>-</sub>di<sub>rec</sub>ti<sub>ona</sub>l <sub>pro</sub>t<sub>oco</sub>l <sub>prov</sub>id<sub>es</sub> <sub>eac</sub>h <sub>manager</sub> <sub>w</sub>ith <sub>a</sub> <sub>compresse</sub>d <sub>v</sub>i<sub>ew</sub> <sub>o</sub>f th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>re</sub>l<sub>evan</sub>t t<sub>o</sub> it<sub>s respons</sub>ibiliti<sub>es.</sub> W<sub>or</sub>k<sub>ers</sub> d<sub>o no</sub>t <sub>nee</sub>d t<sub>o reason over</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> hi<sub>s</sub>t<sub>ory o</sub>f th<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve,</sub> <sub>an</sub>d th<sub>e</sub> <sub>roo</sub>t <sub>manager</sub> d<sub>oes</sub> <sub>no</sub>t <sub>nee</sub>d t<sub>o</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>con</sub>t<sub>ro</sub>l <sub>every</sub> <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>ac</sub>ti<sub>on.</sub> Th<sub>e</sub> hi<sub>er-</sub> <sub>arc</sub>h<sub>y</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>ac</sub>t<sub>s</sub> b<sub>o</sub>th <sub>as</sub> <sub>a</sub> di<sub>v</sub>i<sub>s</sub>i<sub>on</sub> <sub>o</sub>f d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>ki<sub>ng</sub> l<sub>a</sub>b<sub>or</sub> <sub>an</sub>d <sub>as</sub> <sub>an</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on-process</sub>i<sub>ng</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub> ORCH <sub>preserves</sub> th<sub>e</sub> <sub>se</sub>lf<sub>-agency</sub> <sub>o</sub>f <sub>eac</sub>h <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t <sub>an</sub>d <sub>su</sub>b<sub>-</sub>t<sub>eam</sub> <sub>w</sub>hil<sub>e</sub> <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>an</sub>d <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub> l<sub>arge</sub> t<sub>eam</sub> f<sub>or</sub> <sub>comp</sub>l<sub>ex</sub> t<sub>as</sub>k<sub>s.</sub>

## R<sub>esu</sub>lt<sub>s</sub>

## Or<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>p</sub>rin<sub>c</sub>i<sub>p</sub>l<sub>es</sub> im<sub>p</sub>r<sub>ove</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub>

W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d ORCH <sub>on a</sub>ll 25 CREW<sub>-</sub>Wildfi<sub>re m</sub>i<sub>ss</sub>i<sub>ons us</sub>i<sub>ng</sub> fi<sub>ve ran</sub>d<sub>om see</sub>d<sub>s per</sub> t<sub>as</sub>k <sub>an</sub>d <sub>e</sub>i<sub>g</sub>ht LLMs as the base model for each agent including ChatGPT-5.4 (35), Gemma-4-it (36), Llama-4- Scout-16E-it (37), ERNIE-4.5 (38), Qwen-3.6 (39), Nemotron-3-Super (40), GLM-5.1 (41), and DeepSeek-V4-Pro (42). These models range in size and capabilities. They also include both propri-<sub>e</sub>t<sub>a</sub>r<sub>y</sub> <sub>a</sub>nd <sub>ope</sub>n<sub>-we</sub>i<sub>g</sub>ht m<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> W<sub>e</sub> <sub>co</sub>m<sub>pa</sub>r<sub>e</sub>d ORCH <sub>aga</sub>in<sub>s</sub>t f<sub>ou</sub>r r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>ve</sub> r<sub>ece</sub>nt <sub>app</sub>r<sub>oac</sub>h<sub>es</sub> to embodied LLM-based multi-agent collaboration that span the major coordination architectures used in the field. CAMON (16) uses decentralized agents with communication-triggered dynamic leadership, allowing leadership to emerge when coordination is required. COELA (18) is another de-<sub>cen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>me</sub>th<sub>o</sub>d i<sub>n w</sub>hi<sub>c</sub>h <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>agen</sub>t<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y p</sub>l<sub>an, commun</sub>i<sub>ca</sub>t<sub>e, ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n memory,</sub> and act through a modular cognitive architecture. HMAS-2 (19) represents a hybrid centralizedd<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d d<sub>es</sub>i<sub>gn,</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h <sub>a</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>p</sub>l<sub>anner</sub> <sub>proposes</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d <sub>ac</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>are</sub> <sub>su</sub>b<sub>sequen</sub>tl<sub>y</sub> refined through feedback from individual agents. Embodied (17) imposes prompt-based organi-<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ures on commun</sub>i<sub>ca</sub>ti<sub>ng agen</sub>t<sub>s an</sub>d <sub>s</sub>t<sub>u</sub>di<sub>es</sub> h<sub>ow</sub> l<sub>ea</sub>d<sub>ers</sub>hi<sub>p an</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>on a</sub>f<sub>ec</sub>t <sub>coor</sub>di<sub>na</sub>ti<sub>on e</sub>fi<sub>c</sub>i<sub>ency.</sub> O<sub>vera</sub>ll<sub>,</sub> th<sub>ese</sub> b<sub>ase</sub>li<sub>nes cover recen</sub>t d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d<sub>,</sub> d<sub>ynam</sub>i<sub>ca</sub>ll<sub>y</sub> l<sub>e</sub>d <sub>an</sub>d h<sub>y</sub>b<sub>r</sub>id <sub>cen</sub>t<sub>ra</sub>li<sub>ze</sub>d<sub>-</sub>d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d<sub>, an</sub>d <sub>exp</sub>li<sub>c</sub>itl<sub>y organ</sub>i<sub>ze</sub>d <sub>approac</sub>h<sub>es, an</sub>d i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>e mos</sub>t di<sub>rec</sub>tl<sub>y</sub> <sub>compara</sub>bl<sub>e</sub> <sub>pr</sub>i<sub>or</sub> <sub>sys</sub>t<sub>ems</sub> f<sub>or</sub> l<sub>anguage-mo</sub>d<sub>e</sub>l<sub>-</sub>b<sub>ase</sub>d <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>among</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>agen</sub>t<sub>s.</sub>

R<sub>a</sub>th<sub>er</sub> th<sub>an so</sub>l<sub>e</sub>l<sub>y us</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>score as</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on me</sub>t<sub>r</sub>i<sub>c, we a</sub>d<sub>op</sub>t<sub>e</sub>d <sub>a mu</sub>lti<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>pro</sub>t<sub>oco</sub>l t<sub>o</sub> <sub>prov</sub>id<sub>e</sub> <sub>a</sub> <sub>more</sub> <sub>compre</sub>h<sub>ens</sub>i<sub>ve</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> <sub>we</sub> <sub>a</sub>d<sub>op</sub>t<sub>e</sub>d <sub>s</sub>i<sub>x</sub> <sub>com-</sub> <sub>p</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>measures.</sub> Fi<sub>na</sub>l <sub>score</sub> <sub>quan</sub>tifi<sub>e</sub>d th<sub>e</sub> t<sub>as</sub>k <sub>ou</sub>t<sub>come</sub> <sub>a</sub>t th<sub>e</sub> <sub>en</sub>d <sub>o</sub>f <sub>an</sub> <sub>ep</sub>i<sub>so</sub>d<sub>e.</sub> E<sub>xecu</sub>ti<sub>on</sub> eficienc<sub>y</sub> was the area under the task <sub>p</sub>erformance versus task <sub>p</sub>ro<sub>g</sub>ress curve (AUC (43)) throu<sub>g</sub>h <sub>ou</sub>t th<sub>e</sub> t<sub>as</sub>k<sub>,</sub> <sub>rewar</sub>di<sub>ng</sub> t<sub>eams</sub> th<sub>a</sub>t <sub>accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>progress</sub> <sub>ear</sub>li<sub>er</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>on</sub>l<sub>y</sub> <sub>a</sub>t th<sub>e</sub> fi<sub>na</sub>l <sub>s</sub>t<sub>ep.</sub> E<sub>xp</sub>l<sub>ora</sub>ti<sub>on</sub> <sub>measure</sub>d th<sub>e</sub> <sub>area</sub> <sub>covere</sub>d b<sub>y</sub> th<sub>e</sub> t<sub>eam.</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>an</sub>d <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> d<sub>eman</sub>d<sub>s</sub> <sub>were</sub> <sub>cap</sub>t<sub>ure</sub>d th<sub>roug</sub>h th<sub>e</sub> <sub>average</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l API <sub>ca</sub>ll<sub>s,</sub> i<sub>npu</sub>t t<sub>o</sub>k<sub>ens,</sub> <sub>an</sub>d <sub>ou</sub>t<sub>pu</sub>t t<sub>o</sub>k<sub>ens</sub> <sub>per</sub> <sub>env</sub>i<sub>ronmen</sub>t <sub>s</sub>t<sub>ep.</sub> T<sub>as</sub>k<sub>-</sub>l<sub>eve</sub>l <sub>resu</sub>lt<sub>s</sub> <sub>were</sub> <sub>aggrega</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k difi<sub>cu</sub>lt<sub>y</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s</sub> <sub>an</sub>d then avera<sub>g</sub>ed across the ei<sub>g</sub>ht lan<sub>g</sub>ua<sub>g</sub>e models (Table S1–S3). The dificult<sub>y</sub> wei<sub>g</sub>ht assi<sub>g</sub>ned to <sub>eac</sub>h t<sub>as</sub>k i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>par</sub>ti<sub>c</sub>i<sub>pa</sub>ti<sub>ng agen</sub>t<sub>s,</sub> th<sub>e</sub> di<sub>vers</sub>it<sub>y o</sub>f <sub>agen</sub>t t<sub>ypes, an</sub>d th<sub>e</sub> task duration (Section S2 in Su<sub>pp</sub>lementar<sub>y</sub> Text).

C  
A  
![](images/26d129fb2c98f1ce3c5e1e8aa2e427a5a8ca0414a680058d28689e9b3cbbb411.jpg)  
B Rank-1 Percentage across All LLM-Task-Seed Combinations

Aggregated Performance with 25 CREW-Wildfire Tasks and 8 LLMs  
![](images/3155fe745cc50316c0af5c222a3986db9920917860fef6fb1097a59a7206c176.jpg)  
Type-II ANOVA Test Results

![](images/8fc130071efb2f08bdc7dd11585797cba74ed557da6a7c5988c7f342acf3e9f7.jpg)

<table><tr><td>Factor</td><td>Final Score</td><td>Efficiency (AUC)</td></tr><tr><td>Algorithm</td><td>F(df1, df2) Curve F(4, 936) = 9.21</td><td>F(df1, df2) Curve F(4, 936) = 6.68</td><td>p-value p &lt; 0.01</td></tr><tr><td>LLM</td><td>F(8, 936) = 13.89</td><td>F(8, 936) = 9.23</td><td>p &lt; 0.01</td></tr><tr><td>Task</td><td>F(24, 936) = 29.11</td><td>F(24, 936) = 22.80</td><td>p &lt; 0.01</td></tr><tr><td>Algorithm x LLM</td><td>F(32, 936) = 0.90</td><td>F(32, 936) = 0.63</td><td>p = 0.93</td></tr></table>

Fi<sub>gu</sub>r<sub>e</sub> 2<sub>:</sub> N<sub>o</sub>rm<sub>a</sub>li<sub>ze</sub>d A<sub>gg</sub>r<sub>ega</sub>t<sub>e</sub>d P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> <sub>o</sub>n 25 CREW<sub>-</sub>Wildfir<sub>e</sub> T<sub>as</sub>k<sub>s,</sub> R<sub>a</sub>nkin<sub>g</sub> An<sub>a</sub>l<sub>y-</sub> <sub>s</sub>i<sub>s, a</sub>nd ANOVA T<sub>es</sub>t R<sub>esu</sub>lt<sub>s.</sub> Th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>norma</sub>li<sub>za</sub>ti<sub>on process can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> S3<sub>–</sub>S4 i<sub>n</sub> Su<sub>pp</sub>lementar<sub>y</sub> Text. (A) ORCH with human-ex<sub>p</sub>ert-desi<sub>g</sub>ned and LLM-<sub>g</sub>enerated team hierarchies b<sub>o</sub>th <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orm</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> f<sub>our</sub> b<sub>ase</sub>li<sub>nes</sub> i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f <sub>ou</sub>t<sub>pu</sub>t t<sub>o</sub>k<sub>en</sub> <sub>usage,</sub> fi<sub>na</sub>l <sub>score,</sub> <sub>an</sub>d <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> <sub>an</sub>d th<sub>ey</sub> <sub>ran</sub>k t<sub>op-</sub>3 i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f <sub>exp</sub>l<sub>ora</sub>ti<sub>on,</sub> i<sub>npu</sub>t t<sub>o</sub>k<sub>en</sub> <sub>usage,</sub> <sub>an</sub>d th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f API calls. (B) ORCH has the hi<sub>g</sub>hest <sub>p</sub>ro<sub>p</sub>ortion of first-<sub>p</sub>lace rankin<sub>g</sub>s across all the LLM-task-seed combinations for both final score and eficienc<sub>y</sub> metric. (C) The ANOVA test results showed that th<sub>e</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erm</sub> b<sub>e</sub>t<sub>ween a</sub>l<sub>gor</sub>ith<sub>m an</sub>d LLM i<sub>s no</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y s</sub>i<sub>gn</sub>ifi<sub>can</sub>t<sub>, w</sub>hil<sub>e</sub> th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> terms are.

B<sub>o</sub>th ORCH <sub>var</sub>i<sub>an</sub>t<sub>s, us</sub>i<sub>ng</sub> h<sub>uman-exper</sub>t<sub>-</sub>d<sub>es</sub>i<sub>gne</sub>d <sub>an</sub>d LLM<sub>-genera</sub>t<sub>e</sub>d t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es, su</sub>b<sub>-</sub> <sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orme</sub>d <sub>a</sub>ll b<sub>ase</sub>li<sub>ne</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>n</sub> fi<sub>na</sub>l <sub>score,</sub> <sub>execu</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> <sub>an</sub>d <sub>ou</sub>t<sub>pu</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> <sub>w</sub>hil<sub>e</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>among</sub> th<sub>e</sub> t<sub>op</sub> th<sub>ree</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>n</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on,</sub> i<sub>npu</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> <sub>an</sub>d API-call eficienc<sub>y</sub> (Fi<sub>g</sub>ure 2A). This advanta<sub>g</sub>e was also evident throu<sub>g</sub>hout task execution. ORCH <sub>accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>progress</sub> <sub>mar</sub>k<sub>e</sub>dl<sub>y</sub> f<sub>as</sub>t<sub>er</sub> th<sub>an</sub> th<sub>e</sub> <sub>compe</sub>ti<sub>ng</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>an</sub>d <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>a</sub> <sub>c</sub>l<sub>ear</sub> <sub>per</sub>f<sub>or-</sub> <sub>mance</sub> <sub>marg</sub>i<sub>n</sub> th<sub>roug</sub>h th<sub>e</sub> <sub>en</sub>d <sub>o</sub>f th<sub>e</sub> <sub>ep</sub>i<sub>so</sub>d<sub>es.</sub> I<sub>mpor</sub>t<sub>an</sub>tl<sub>y,</sub> th<sub>ese</sub> <sub>ga</sub>i<sub>ns</sub> <sub>were</sub> <sub>no</sub>t <sub>con</sub>fi<sub>ne</sub>d t<sub>o</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> t<sub>ype</sub> <sub>o</sub>f <sub>m</sub>i<sub>ss</sub>i<sub>on.</sub> A<sub>cross</sub> t<sub>as</sub>k<sub>s</sub> th<sub>a</sub>t <sub>requ</sub>i<sub>re</sub>d <sub>scou</sub>ti<sub>ng,</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>rescue,</sub> t<sub>ree</sub> <sub>cu</sub>tti<sub>ng,</sub> <sub>con-</sub> t<sub>a</sub>i<sub>nmen</sub>t<sub>,</sub> <sub>suppress</sub>i<sub>on,</sub> <sub>an</sub>d <sub>com</sub>bi<sub>na</sub>ti<sub>ons</sub> <sub>o</sub>f th<sub>ese</sub> <sub>ac</sub>ti<sub>v</sub>iti<sub>es,</sub> t<sub>as</sub>k<sub>-spec</sub>ifi<sub>c</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l hi<sub>erarc</sub>h<sub>y</sub> <sub>genera</sub>ll<sub>y</sub> <sub>ena</sub>bl<sub>e</sub>d <sub>agen</sub>t<sub>s</sub> t<sub>o</sub> <sub>ma</sub>k<sub>e</sub> <sub>progress</sub> <sub>more</sub> <sub>rap</sub>idl<sub>y</sub> <sub>an</sub>d <sub>reac</sub>h <sub>s</sub>t<sub>ronger</sub> fi<sub>na</sub>l <sub>ou</sub>t<sub>comes.</sub>

Thi<sub>s</sub> i<sub>s an</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>resu</sub>lt <sub>s</sub>i<sub>nce</sub> th<sub>e s</sub>i<sub>x measures expose</sub> dif<sub>eren</sub>t <sub>poss</sub>ibl<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f<sub>s.</sub> A <sub>sys</sub>t<sub>em</sub> <sub>cou</sub>ld i<sub>ncrease</sub> fi<sub>na</sub>l <sub>per</sub>f<sub>ormance</sub> b<sub>y</sub> i<sub>ssu</sub>i<sub>ng more mo</sub>d<sub>e</sub>l <sub>quer</sub>i<sub>es, genera</sub>ti<sub>ng</sub> l<sub>onger messages, or</sub> <sub>exp</sub>l<sub>or</sub>i<sub>ng</sub> i<sub>ne</sub>fi<sub>c</sub>i<sub>ency.</sub> ORCH did <sub>no</sub>t <sub>ex</sub>hibit thi<sub>s</sub> t<sub>ra</sub>d<sub>e-o</sub>f<sub>.</sub> ORCH’<sub>s</sub> <sub>ga</sub>i<sub>ns</sub> i<sub>n</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>ou</sub>t<sub>come</sub> <sub>an</sub>d <sub>execu</sub>ti<sub>on</sub> <sub>spee</sub>d <sub>were</sub> <sub>ac</sub>hi<sub>eve</sub>d <sub>w</sub>hil<sub>e</sub> <sub>surpass</sub>i<sub>ng</sub> <sub>or</sub> <sub>approac</sub>hi<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>n</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>an</sub>d <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency.</sub> Th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>mprove</sub>d <sub>no</sub>t <sub>on</sub>l<sub>y</sub> <sub>w</sub>h<sub>a</sub>t th<sub>e</sub> t<sub>eam</sub> <sub>accomp</sub>li<sub>s</sub>h<sub>e</sub>d b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> h<sub>ow</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>were</sub> <sub>pro</sub>d<sub>uce</sub>d<sub>.</sub>

## Th<sub>e</sub> <sub>a</sub>d<sub>va</sub>nt<sub>age</sub> <sub>o</sub>f <sub>o</sub>r<sub>ga</sub>niz<sub>a</sub>ti<sub>o</sub>n <sub>pe</sub>r<sub>s</sub>i<sub>s</sub>t<sub>s</sub> <sub>ac</sub>r<sub>oss</sub> m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> <sub>a</sub>nd mi<sub>ss</sub>i<sub>o</sub>n<sub>s</sub>

T<sub>o</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> ORCH’<sub>s</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> d<sub>epen</sub>d<sub>e</sub>d <sub>on</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>com</sub>bi<sub>na</sub>ti<sub>ons</sub> <sub>o</sub>f t<sub>as</sub>k <sub>an</sub>d l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>, we ran</sub>k<sub>e</sub>d th<sub>e s</sub>i<sub>x a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or every</sub> l<sub>anguage-mo</sub>d<sub>e</sub>l<sub>–</sub>t<sub>as</sub>k<sub>–see</sub>d <sub>com</sub>bi<sub>na</sub>ti<sub>on.</sub> ORCH <sub>w</sub>ith h<sub>uman-exper</sub>t<sub>-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>h<sub>y rece</sub>i<sub>ve</sub>d th<sub>e</sub> fi<sub>rs</sub>t<sub>-p</sub>l<sub>ace ran</sub>k i<sub>n</sub> 32<sub>.</sub>6% <sub>o</sub>f <sub>com</sub>bi<sub>na</sub>ti<sub>ons</sub> f<sub>or</sub> fi<sub>na</sub>l <sub>score an</sub>d 31<sub>.</sub>1% f<sub>or execu</sub>ti<sub>on e</sub>fi<sub>c</sub>i<sub>ency an</sub>d ORCH <sub>w</sub>ith LLM<sub>-genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>h<sub>y rece</sub>i<sub>ve</sub>d th<sub>e</sub> fi<sub>rs</sub>t<sub>-p</sub>l<sub>ace ran</sub>k i<sub>n</sub> 26<sub>.</sub>2% <sub>o</sub>f <sub>com</sub>bi<sub>na</sub>ti<sub>ons</sub> f<sub>or</sub> fi<sub>na</sub>l <sub>score an</sub>d 25<sub>.</sub>0% f<sub>or execu</sub>ti<sub>on e</sub>fi<sub>c</sub>i<sub>ency an</sub>d Th<sub>e correspon</sub>di<sub>ng</sub> fi<sub>rs</sub>t<sub>-p</sub>l<sub>ace</sub> f<sub>requenc</sub>i<sub>es were</sub> 16<sub>.</sub>2% <sub>an</sub>d 19<sub>.</sub>5% f<sub>or</sub> CAMON<sub>,</sub> 7<sub>.</sub>2% <sub>an</sub>d 4<sub>.</sub>8% for COELA, 5.8% and 6.3% for HMAS-2, and 12.1% and 13.3% for Embodied (Fi<sub>g</sub>ure 2B). The <sub>resu</sub>lt<sub>s</sub> <sub>s</sub>h<sub>owe</sub>d th<sub>a</sub>t ORCH l<sub>arge</sub>l<sub>y</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orms</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>nes</sub> <sub>across</sub> <sub>a</sub> <sub>w</sub>id<sub>e</sub> <sub>range</sub> <sub>o</sub>f t<sub>as</sub>k<sub>s,</sub> b<sub>ase</sub> LLM <sub>mo</sub>d<sub>e</sub>l<sub>s, an</sub>d <sub>see</sub>d <sub>com</sub>bi<sub>na</sub>ti<sub>ons.</sub>

We further performed Type-II ANOVA tests (44) to assess the efectiveness of the algorithm, LLM<sub>,</sub> t<sub>as</sub>k<sub>, an</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>m-</sub>LLM i<sub>n</sub>t<sub>erac</sub>ti<sub>on on</sub> fi<sub>na</sub>l <sub>score an</sub>d <sub>e</sub>fi<sub>c</sub>i<sub>ency, respec</sub>ti<sub>ve</sub>l<sub>y.</sub> A<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 2C<sub>,</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> h<sub>a</sub>d <sub>a</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>ma</sub>i<sub>n</sub> <sub>e</sub>f<sub>ec</sub>t <sub>on</sub> fi<sub>na</sub>l <sub>score</sub> $( F _ { 4 , 9 3 6 } = 9 . 2 1 , p < 0 . 0 1 )$ <sub>an</sub>d <sub>execu</sub>ti<sub>on e</sub>fi<sub>c</sub>i<sub>ency</sub> $( F _ { 4 , 8 9 3 6 } = 6 . 6 8 , p < 0 . 0 1 )$ ). Similarl<sub>y</sub>, lan<sub>g</sub>ua<sub>g</sub>e model and task also had <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>e</sub>f<sub>ec</sub>t<sub>s on</sub> b<sub>o</sub>th <sub>measures.</sub> F<sub>or</sub> fi<sub>na</sub>l <sub>score,</sub> th<sub>e repor</sub>t<sub>e</sub>d <sub>e</sub>f<sub>ec</sub>t<sub>s were</sub> $F _ { 8 , 9 3 6 } = 1 3 . 8 9$ f<sub>or</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d $F _ { 2 4 , 9 1 6 } = 2 9 . 1 1 $ f<sub>or</sub> t<sub>as</sub>k $( p < 0 . 0 1$ for both). For execution eficienc<sub>y</sub>, the<sub>y</sub> were $F _ { 8 , 9 3 6 } = 9 . 2 3$ <sub>an</sub>d $F _ { 2 4 , 9 3 6 } = 2 2 . 8 0$ , res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub> $( p < 0 . 0 1$ for both). B<sub>y</sub> contrast, the al<sub>g</sub>orithm-b<sub>y</sub>- <sub>mo</sub>d<sub>e</sub>l i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> <sub>was</sub> <sub>no</sub>t <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t f<sub>or</sub> <sub>e</sub>ith<sub>er</sub> fi<sub>na</sub>l <sub>score</sub> $( F _ { 3 2 , 9 3 6 } = 0 . 9 0 , p = 0 . 6 2 )$ <sub>or execu</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency</sub> $( F _ { 3 2 , 9 3 6 } = 0 . 6 3 , p = 0 . 9 3 )$ <sub>.</sub> Th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> <sub>o</sub>f ORCH <sub>was</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>across</sub> th<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> b<sub>e</sub>i<sub>ng</sub> d<sub>r</sub>i<sub>ven</sub> b<sub>y</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l th<sub>a</sub>t h<sub>appene</sub>d t<sub>o</sub> i<sub>n</sub>t<sub>erac</sub>t f<sub>avora</sub>bl<sub>y w</sub>ith th<sub>e</sub> f<sub>ramewor</sub>k<sub>.</sub>

## C<sub>o</sub>ll<sub>ec</sub>ti<sub>ve pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> i<sub>s</sub> n<sub>o</sub>t d<sub>e</sub>t<sub>e</sub>rmin<sub>e</sub>d b<sub>y</sub> m<sub>o</sub>d<sub>e</sub>l <sub>sca</sub>l<sub>e</sub>

O<sub>ur</sub> <sub>compar</sub>i<sub>son</sub> <sub>across</sub> <sub>e</sub>i<sub>g</sub>ht l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>revea</sub>l<sub>e</sub>d th<sub>a</sub>t <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>per</sub>f<sub>ormance</sub> did not increase monotonicall<sub>y</sub> with nominal model scale. Gemma-4-it and Qwen-3.6 produced the <sub>s</sub>t<sub>ronges</sub>t <sub>aggrega</sub>t<sub>e</sub> <sub>per</sub>f<sub>ormance</sub> <sub>w</sub>ithi<sub>n</sub> ORCH<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> l<sub>arger</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> Ch<sub>a</sub>tGPT<sub>-</sub>5<sub>.</sub>4<sub>,</sub> GLM<sub>-</sub>5<sub>.</sub>1<sub>,</sub> <sub>an</sub>d D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>V4<sub>-</sub>P<sub>ro,</sub> <sub>ac</sub>hi<sub>eve</sub>d <sub>more</sub> <sub>mo</sub>d<sub>era</sub>t<sub>e</sub> <sub>aggrega</sub>t<sub>e</sub> <sub>resu</sub>lt<sub>s,</sub> <sub>even</sub> th<sub>oug</sub>h th<sub>ey</sub> achieve state-of-the-art performance on many popular benchmarks, such as coding (45–47) and mathematical reasoning (48, 49) (Figure 3). This finding does not imply that model capability i<sub>s</sub> i<sub>rre</sub>l<sub>evan</sub>t<sub>.</sub> Th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f <sub>var</sub>i<sub>ance</sub> <sub>a</sub>b<sub>ove</sub> <sub>s</sub>h<sub>owe</sub>d <sub>a</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>e</sub>f<sub>ec</sub>t <sub>o</sub>f th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> R<sub>a</sub>th<sub>er,</sub> it <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>sca</sub>l<sub>e</sub> <sub>an</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>capa</sub>bilit<sub>y</sub> <sub>are</sub> <sub>no</sub>t i<sub>n</sub>t<sub>erc</sub>h<sub>angea</sub>bl<sub>e.</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub>d <sub>or</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d f<sub>or</sub> <sub>co</sub>di<sub>ng,</sub> f<sub>ac</sub>t<sub>ua</sub>l <sub>reca</sub>ll<sub>,</sub> <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>reason</sub>i<sub>ng,</sub> <sub>or</sub> <sub>s</sub>h<sub>or</sub>t<sub>-</sub>f<sub>orm</sub> l<sub>anguage</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> <sub>may</sub> dif<sub>er</sub> i<sub>n</sub> th<sub>e</sub> <sub>proper</sub>ti<sub>es</sub> <sub>nee</sub>d<sub>e</sub>d t<sub>o</sub> <sub>sus</sub>t<sub>a</sub>i<sub>n</sub> l<sub>ong-</sub>h<sub>or</sub>i<sub>zon</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on.</sub> S<sub>uc</sub>h <sub>proper</sub>ti<sub>es</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>ro</sub>l<sub>es,</sub> <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>conc</sub>i<sub>se</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>summar</sub>i<sub>es,</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>hi<sub>ng</sub> <sub>para</sub>ll<sub>e</sub>l f<sub>rom sequen</sub>ti<sub>a</sub>l <sub>wor</sub>k<sub>, a</sub>d<sub>ap</sub>ti<sub>ng ass</sub>i<sub>gnmen</sub>t<sub>s</sub> f<sub>rom</sub> l<sub>oca</sub>l <sub>repor</sub>t<sub>s, an</sub>d t<sub>rans</sub>l<sub>a</sub>ti<sub>ng organ</sub>i<sub>za</sub>ti<sub>ona</sub>l i<sub>ns</sub>t<sub>ruc</sub>ti<sub>ons</sub> i<sub>n</sub>t<sub>o</sub> f<sub>eas</sub>ibl<sub>e ac</sub>ti<sub>ons.</sub>

Th<sub>e</sub> <sub>a</sub>b<sub>sence</sub> <sub>o</sub>f <sub>a</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>m-</sub>b<sub>y-mo</sub>d<sub>e</sub>l i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> <sub>prov</sub>id<sub>es</sub> <sub>a</sub> <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>resu</sub>lt<sub>.</sub> ORCH did <sub>no</sub>t <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>compensa</sub>t<sub>e</sub> f<sub>or</sub> <sub>one</sub> <sub>wea</sub>k <sub>mo</sub>d<sub>e</sub>l <sub>or</sub> <sub>exp</sub>l<sub>o</sub>it <sub>one</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>promp</sub>ti<sub>ng</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or.</sub> I<sub>ns</sub>t<sub>ea</sub>d<sub>,</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> i<sub>mprove</sub>d <sub>per</sub>f<sub>ormance</sub> <sub>w</sub>ith b<sub>roa</sub>dl<sub>y</sub> <sub>s</sub>i<sub>m</sub>il<sub>ar</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s across</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>se</sub>t<sub>.</sub> C<sub>o</sub>ll<sub>ec</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub> th<sub>ere</sub>f<sub>ore re</sub>fl<sub>ec</sub>t<sub>e</sub>d b<sub>o</sub>th th<sub>e capa</sub>biliti<sub>es o</sub>f th<sub>e</sub> <sub>mem</sub>b<sub>ers</sub> <sub>an</sub>d th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> th<sub>a</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d th<sub>em.</sub> Th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>per</sub>f<sub>ormance</sub> <sub>o</sub>f <sub>eac</sub>h t<sub>as</sub>k <sub>w</sub>ith <sub>eac</sub>h LLM <sub>was</sub> <sub>p</sub>r<sub>ese</sub>nt<sub>e</sub>d in th<sub>e</sub> <sub>supp</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>r<sub>y</sub> Fi<sub>gu</sub>r<sub>es</sub> S1<sub>–</sub>S8<sub>.</sub>

![](images/2c919532bbaaaf41dd4667f54c745753f0ceecefbcd1410f86c93b765e509f29.jpg)

![](images/877e6acff3a56b5c3e15af83a53ef4b84d71621918278027ffbc404d09569792.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 3<sub>:</sub> N<sub>o</sub>rm<sub>a</sub>li<sub>ze</sub>d A<sub>gg</sub>r<sub>ega</sub>t<sub>e</sub>d P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> <sub>o</sub>n 25 CREW<sub>-</sub>Wildfir<sub>e</sub> T<sub>as</sub>k<sub>s</sub> b<sub>y</sub> L<sub>a</sub>n<sub>guage</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s.</sub> Th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>process</sub> <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> S3<sub>–</sub>S4 i<sub>n</sub> th<sub>e</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> Text. (A) A<sub>gg</sub>re<sub>g</sub>ated LLM <sub>p</sub>erformance on the 25 CREW-Wildfire tasks with ORCH with humanex<sub>p</sub>ert-desi<sub>g</sub>ned team hierarchies. (B) A<sub>gg</sub>re<sub>g</sub>ated LLM <sub>p</sub>erformance on the 25 CREW-Wildfire t<sub>as</sub>k<sub>s</sub> <sub>w</sub>ith ORCH <sub>w</sub>ith LLM<sub>-genera</sub>t<sub>e</sub>d t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es.</sub> S<sub>urpr</sub>i<sub>s</sub>i<sub>ng</sub>l<sub>y,</sub> th<sub>e</sub> <sub>m</sub>iddl<sub>e-s</sub>i<sub>ze</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>per</sub>f<sub>orm</sub> b<sub>e</sub>tt<sub>er</sub> th<sub>an</sub> l<sub>arge-s</sub>i<sub>ze</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>w</sub>ith ORCH <sub>un</sub>d<sub>er</sub> b<sub>o</sub>th <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>es</sub>i<sub>gns.</sub>

## ORCH’<sub>s</sub> <sub>o</sub>r<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>p</sub>rin<sub>c</sub>i<sub>p</sub>l<sub>e</sub> <sub>suppo</sub>rt<sub>s</sub> <sub>sca</sub>l<sub>e</sub> <sub>a</sub>nd h<sub>e</sub>t<sub>e</sub>r<sub>oge</sub>n<sub>e</sub>it<sub>y</sub>

ORCH’<sub>s</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>ro</sub>l<sub>e</sub> <sub>o</sub>f hi<sub>erarc</sub>h<sub>y</sub> b<sub>ecomes</sub> <sub>prom</sub>i<sub>nen</sub>t i<sub>n</sub> <sub>more</sub> <sub>comp</sub>l<sub>ex</sub> t<sub>as</sub>k<sub>s.</sub> W<sub>e</sub> <sub>v</sub>i<sub>sua</sub>li<sub>ze</sub>d th<sub>e</sub> <sub>mos</sub>t difi<sub>cu</sub>lt “S<sub>ca</sub>l<sub>e</sub> L<sub>eve</sub>l C<sub>omp</sub>l<sub>ex</sub>” t<sub>as</sub>k<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>res</sub> th<sub>e</sub> t<sub>eam</sub> t<sub>o</sub> l<sub>oca</sub>t<sub>e</sub> <sub>a</sub> fi<sub>re</sub> <sub>an</sub>d <sub>a</sub> <sub>wa</sub>t<sub>er</sub> source on a $2 0 0 \times 2 0 0$ <sub>map,</sub> <sub>suppress</sub> th<sub>e</sub> fi<sub>re,</sub> <sub>an</sub>d <sub>per</sub>f<sub>orm</sub> <sub>searc</sub>h <sub>an</sub>d <sub>rescue</sub> <sub>o</sub>f th<sub>e</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans.</sub> Th<sub>e</sub> <sub>wor</sub>kf<sub>orce</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 50 <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>agen</sub>t<sub>s:</sub> 25 fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers,</sub> 10 d<sub>rones,</sub> 10 h<sub>e</sub>li<sub>cop</sub>t<sub>ers,</sub> <sub>an</sub>d 5 b<sub>u</sub>lld<sub>ozers.</sub> Th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gne</sub>d b<sub>y</sub> th<sub>e</sub> h<sub>uman</sub> <sub>exper</sub>t f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> ORCH <sub>cons</sub>i<sub>s</sub>t<sub>e</sub>d <sub>o</sub>f <sub>one</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>manager</sub> <sub>superv</sub>i<sub>s</sub>i<sub>ng</sub> fi<sub>ve</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers,</sub> <sub>eac</sub>h <sub>o</sub>f <sub>w</sub>h<sub>om</sub> <sub>oversaw</sub> <sub>a</sub> dif<sub>eren</sub>t <sub>group</sub> <sub>o</sub>f <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s.</sub> A<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub> Fi<sub>gure</sub> 4<sub>,</sub> th<sub>e</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>manager</sub> d<sub>ecompose</sub>d th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> f<sub>our</sub> <sub>p</sub>h<sub>ases.</sub> Fi<sub>rs</sub>t<sub>,</sub> th<sub>e</sub> t<sub>eam</sub> <sub>scou</sub>t<sub>e</sub>d th<sub>e</sub> <sub>map</sub> t<sub>o</sub> l<sub>oca</sub>t<sub>e</sub> th<sub>e</sub> fi<sub>re,</sub> <sub>wa</sub>t<sub>er</sub> <sub>source,</sub> <sub>an</sub>d <sub>c</sub>i<sub>v</sub>ili<sub>ans.</sub> S<sub>econ</sub>d<sub>,</sub> <sub>aer</sub>i<sub>a</sub>l <sub>un</sub>it<sub>s</sub> <sub>con</sub>fi<sub>rme</sub>d <sub>an</sub>d <sub>mon</sub>it<sub>ore</sub>d th<sub>e</sub> fi<sub>re</sub> <sub>w</sub>hil<sub>e</sub> id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> <sub>sa</sub>f<sub>e</sub> <sub>access</sub> <sub>rou</sub>t<sub>es.</sub> Thi<sub>r</sub>d<sub>,</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>ers</sub> <sub>re</sub>fill<sub>e</sub>d <sub>a</sub>t th<sub>e</sub> <sub>wa</sub>t<sub>er</sub> <sub>source,</sub> <sub>an</sub>d th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> <sub>pos</sub>iti<sub>one</sub>d fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>an</sub>d b<sub>u</sub>lld<sub>ozers</sub> <sub>aroun</sub>d th<sub>e</sub> fi<sub>re.</sub> F<sub>our</sub>th<sub>,</sub> <sub>groun</sub>d <sub>an</sub>d <sub>aer</sub>i<sub>a</sub>l t<sub>eams</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d fi<sub>re</sub>b<sub>rea</sub>k<sub>s</sub> <sub>an</sub>d <sub>suppresse</sub>d <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> fl<sub>ames</sub> <sub>w</sub>hil<sub>e</sub> li<sub>m</sub>iti<sub>ng</sub> <sub>exposure</sub> <sub>o</sub>f <sub>groun</sub>d <sub>crews.</sub>

Withi<sub>n</sub> <sub>eac</sub>h <sub>p</sub>h<sub>ase,</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>concurren</sub>t <sub>wor</sub>k<sub>.</sub> D<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>p</sub>h<sub>ase,</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> 1<sub>-</sub>3 <sub>ass</sub>i<sub>gne</sub>d th<sub>e</sub>i<sub>r wor</sub>k<sub>er agen</sub>t<sub>s</sub> t<sub>o exp</sub>l<sub>ore</sub> dif<sub>eren</sub>t <sub>reg</sub>i<sub>ons o</sub>f th<sub>e map</sub> t<sub>o</sub> l<sub>oca</sub>t<sub>e</sub> th<sub>e</sub> fi<sub>re</sub> <sub>an</sub>d <sub>wa</sub>t<sub>er</sub> <sub>source,</sub> <sub>w</sub>hil<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub> 4<sub>,</sub> <sub>w</sub>h<sub>o</sub> <sub>oversees</sub> <sub>a</sub> t<sub>eam</sub> <sub>o</sub>f fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers,</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d it<sub>s c</sub>hild <sub>agen</sub>t<sub>s</sub> t<sub>o rescue</sub> th<sub>e c</sub>i<sub>v</sub>ili<sub>ans.</sub> O<sub>nce</sub> th<sub>e c</sub>i<sub>v</sub>ili<sub>ans were secure</sub>d<sub>,</sub> th<sub>e</sub> fi<sub>re was</sub> <sub>scou</sub>t<sub>e</sub>d<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>wa</sub>t<sub>er</sub> <sub>source</sub> <sub>was</sub> f<sub>oun</sub>d<sub>,</sub> th<sub>e</sub> t<sub>eam</sub> t<sub>rans</sub>iti<sub>one</sub>d t<sub>o</sub> th<sub>e</sub> <sub>secon</sub>d <sub>p</sub>h<sub>ase,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager,</sub> i<sub>n</sub> <sub>c</sub>h<sub>arge</sub> <sub>o</sub>f th<sub>e</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>ers,</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d <sub>aer</sub>i<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>aroun</sub>d th<sub>e</sub> fi<sub>re.</sub> I<sub>n</sub> th<sub>e</sub> thi<sub>r</sub>d <sub>p</sub>h<sub>ase,</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub> <sub>respons</sub>ibl<sub>e</sub> f<sub>or</sub> th<sub>e</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>ers</sub> di<sub>rec</sub>t<sub>e</sub>d th<sub>em</sub> t<sub>o</sub> th<sub>e</sub> di<sub>scovere</sub>d <sub>wa</sub>t<sub>er</sub> <sub>source</sub> t<sub>o</sub> <sub>re</sub>fill b<sub>e</sub>f<sub>ore</sub> d<sub>ep</sub>l<sub>oy</sub>i<sub>ng</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>near</sub> th<sub>e</sub> fi<sub>re.</sub> Fi<sub>na</sub>ll<sub>y,</sub> i<sub>n</sub> <sub>p</sub>h<sub>ase</sub> f<sub>our,</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub>d th<sub>e</sub>i<sub>r</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s</sub> t<sub>o</sub> <sub>cons</sub>t<sub>ruc</sub>t fi<sub>re</sub>b<sub>rea</sub>k<sub>s</sub> <sub>an</sub>d <sub>suppress</sub> th<sub>e</sub> fi<sub>re</sub> <sub>un</sub>til th<sub>e</sub> t<sub>as</sub>k <sub>was</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d<sub>.</sub> O<sub>vera</sub>ll<sub>,</sub> thi<sub>s</sub> <sub>examp</sub>l<sub>e</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> h<sub>ow</sub> th<sub>e</sub> <sub>propose</sub>d hi<sub>erarc</sub>h<sub>y</sub> <sub>ena</sub>bl<sub>es</sub>

![](images/f7f770079838c3e5bac00552bb6c7e746480524ee95eb5691521c2445504de5b.jpg)

Fi<sub>g</sub>ure 4: Visualization of ORCH on Scale Level Complex (Most Dificult Task). The vertical <sub>manager</sub> di<sub>v</sub>id<sub>es</sub> th<sub>e</sub> t<sub>as</sub>k i<sub>n</sub>t<sub>o</sub> f<sub>our</sub> <sub>p</sub>h<sub>ases</sub> <sub>an</sub>d t<sub>rac</sub>k<sub>s</sub> <sub>eac</sub>h <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d th<sub>e</sub> <sub>overa</sub>ll <sub>progress.</sub> D<sub>ur</sub>i<sub>ng</sub> <sub>eac</sub>h <sub>p</sub>h<sub>ase,</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> <sub>genera</sub>t<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> f<sub>or</sub> th<sub>e</sub>i<sub>r</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s</sub> <sub>an</sub>d t<sub>rac</sub>k th<sub>e</sub> <sub>progress.</sub>

th<sub>e</sub> t<sub>eam</sub> t<sub>o</sub> d<sub>ecompose</sub> <sub>a</sub> <sub>comp</sub>l<sub>ex</sub> l<sub>ong-</sub>h<sub>or</sub>i<sub>zon</sub> t<sub>as</sub>k i<sub>n</sub>t<sub>o</sub> <sub>managea</sub>bl<sub>e</sub> <sub>s</sub>t<sub>ages</sub> <sub>w</sub>hil<sub>e</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>ng</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>groups</sub> <sub>o</sub>f h<sub>e</sub>t<sub>erogeneous</sub> <sub>agen</sub>t<sub>s</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>w</sub>ith <sub>a</sub> h<sub>y</sub>b<sub>r</sub>id <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e.</sub> B<sub>y</sub> <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>w</sub>ith h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>coor</sub>di<sub>na</sub>ti<sub>on,</sub> ORCH <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ns</sub> <sub>c</sub>l<sub>ear</sub> t<sub>as</sub>k <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> <sub>con</sub>ti<sub>nuous</sub> <sub>progress</sub> t<sub>rac</sub>ki<sub>ng,</sub> <sub>an</sub>d <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> th<sub>roug</sub>h<sub>ou</sub>t <sub>execu</sub>ti<sub>on.</sub> Th<sub>ere</sub>f<sub>ore,</sub> <sub>sca</sub>li<sub>ng</sub> i<sub>s</sub> <sub>ena</sub>bl<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e</sub> “t<sub>eams</sub> <sub>o</sub>f t<sub>eams</sub>” <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> th<sub>a</sub>t <sub>preserve</sub> l<sub>oca</sub>l <sub>concurrency</sub> <sub>an</sub>d <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>or</sub>d<sub>er.</sub>

## H<sub>u</sub>m<sub>a</sub>n <sub>a</sub>nd <sub>a</sub>rtifi<sub>c</sub>i<sub>a</sub>l <sub>o</sub>r<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l d<sub>es</sub>i<sub>g</sub>n

W<sub>e</sub> <sub>compare</sub>d th<sub>ree</sub> <sub>ways</sub> <sub>o</sub>f <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> th<sub>e</sub> hi<sub>erarc</sub>h<sub>y,</sub> <sub>a</sub>ll f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> ORCH’<sub>s</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> <sub>o</sub>f b<sub>a</sub>ki<sub>ng</sub> <sub>poo</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ency</sub> <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ency</sub> i<sub>n</sub>t<sub>o</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gn,</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>-</sub> i<sub>ng</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gns</sub> b<sub>y</sub> <sub>a</sub> h<sub>uman</sub> <sub>exper</sub>t<sub>,</sub> <sub>genera</sub>ti<sub>on</sub> b<sub>y</sub> <sub>a</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l <sub>w</sub>ith <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on,</sub> <sub>an</sub>d <sub>genera</sub>ti<sub>on</sub> b<sub>y</sub> <sub>a</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub> <sub>cr</sub>iti<sub>c.</sub> Th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>promp</sub>t<sub>s</sub> <sub>are</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d i<sub>n</sub> P<sub>romp</sub>t S2<sub>.</sub> Human-desi<sub>g</sub>ned hierarchies <sub>p</sub>roduced the stron<sub>g</sub>est a<sub>gg</sub>re<sub>g</sub>ate <sub>p</sub>erformance (Fi<sub>g</sub>ure 5A). This result <sub>sugges</sub>t<sub>s</sub> th<sub>a</sub>t <sub>peop</sub>l<sub>e</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> <sub>an</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> i<sub>n</sub> <sub>recogn</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>o</sub>f <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>wor</sub>k <sub>an</sub>d i<sub>n cons</sub>t<sub>ruc</sub>ti<sub>ng conc</sub>i<sub>se</sub> di<sub>v</sub>i<sub>s</sub>i<sub>ons o</sub>f <sub>respons</sub>ibilit<sub>y.</sub> H<sub>umans may a</sub>l<sub>so a</sub>l<sub>rea</sub>d<sub>y</sub> h<sub>ave un-</sub> derstandings of organizational hierarchy (50, 51). Another key possible reason could be due to the strong cognitive skills (52,53) such as reasoning and memory capabilities that humans have, which <sub>rema</sub>i<sub>n</sub> k<sub>ey areas o</sub>f <sub>researc</sub>h t<sub>o</sub> i<sub>mprove</sub> th<sub>e curren</sub>t f<sub>oun</sub>d<sub>a</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s an</sub>d <sub>agen</sub>ti<sub>c</sub> AI <sub>sys</sub>t<sub>ems.</sub> Thi<sub>s</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> d<sub>oes</sub> <sub>no</sub>t f<sub>ocus</sub> <sub>on</sub> <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> h<sub>uman</sub> <sub>per</sub>f<sub>ormance.</sub> H<sub>ence,</sub> <sub>we</sub> l<sub>eave</sub> <sub>r</sub>i<sub>c</sub>h<sub>er</sub> h<sub>uman</sub> <sub>s</sub>t<sub>u</sub>di<sub>es as</sub> f<sub>u</sub>t<sub>ure wor</sub>k<sub>.</sub>

Amon<sub>g</sub> automaticall<sub>y g</sub>enerated or<sub>g</sub>anizations, both hierarchies (with and without critic su-<sub>p</sub>ervision) out<sub>p</sub>erformed the stron<sub>g</sub>est baseline s<sub>y</sub>stems, althou<sub>g</sub>h the<sub>y</sub> remained below the humand<sub>es</sub>i<sub>gne</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ons.</sub> R<sub>emov</sub>i<sub>ng</sub> th<sub>e</sub> <sub>cr</sub>iti<sub>c</sub> <sub>cause</sub>d <sub>a</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on.</sub> I<sub>nspec</sub>ti<sub>on</sub> <sub>o</sub>f <sub>genera</sub>t<sub>e</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub>d th<sub>a</sub>t l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>cr</sub>iti<sub>que</sub> t<sub>en</sub>d<sub>e</sub>d t<sub>o</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>unnec-</sub> essar<sub>y</sub> mana<sub>g</sub>ers, la<sub>y</sub>ers, or <sub>g</sub>rou<sub>p</sub>s (Fi<sub>g</sub>ure 6). These overcom<sub>p</sub>licated or<sub>g</sub>anizations increased

Different Method Generated Team Hierarchy Analysis

B  
![](images/f1a6373ce1987f8bef2733fa83b4c87343afe3155b6dcb4385d866b3482fcee4.jpg)

![](images/65432c84fff11e0d4c27090142d8a2e84940c3995796f28bf425c6b0e73e7b66.jpg)

![](images/f8a898f68328eb13dbeada449b7c1c6b648f9adf20742b5b671b6ee73e6651ec.jpg)

![](images/cdbd6a575b45bf34f6c7d215686a8e7ff5bc1efc8d29d760bbd12cd2cdb299cb.jpg)

![](images/f9c9ee17a1240cb8357da93756072bdd49827b52ae433100b8abf3c73d3023fb.jpg)

![](images/87b1cf8025ab0bb29dc78fbc841e46f336ffe88bfb0b5f5b3b7b837416030781.jpg)

![](images/e1192e6ab2799da1a51d8dd0bf0377bca61865cf17cd7a56c8aa4399987154df.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 5<sub>:</sub> Abl<sub>a</sub>ti<sub>o</sub>n <sub>s</sub>t<sub>u</sub>d<sub>y o</sub>n Or<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n D<sub>es</sub>i<sub>g</sub>n<sub>,</sub> T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> An<sub>a</sub>l<sub>ys</sub>i<sub>s, a</sub>nd F<sub>a</sub>il<sub>u</sub>r<sub>e</sub> Reasons Analysis through Chat History. (A) ORCH consistentl<sub>y</sub> out<sub>p</sub>erformed <sub>p</sub>rior baselines <sub>across</sub> h<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d <sub>an</sub>d LLM<sub>-genera</sub>t<sub>e</sub>d t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>ose genera</sub>t<sub>e</sub>d <sub>w</sub>ith<sub>ou</sub>t critic su<sub>p</sub>ervision. (B) We anal<sub>y</sub>zed the team hierarchies <sub>g</sub>enerated b<sub>y</sub> diferent methods. The results <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t h<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>ex</sub>hibit th<sub>e</sub> <sub>mos</sub>t b<sub>a</sub>l<sub>ance</sub>d t<sub>eam</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ures,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> LLM<sub>-</sub> <sub>genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>w</sub>ith <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on,</sub> <sub>w</sub>hil<sub>e</sub> hi<sub>erarc</sub>hi<sub>es</sub> <sub>genera</sub>t<sub>e</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> are less balanced. (C) We classified the failure modes of diferent al<sub>g</sub>orithms into five cate<sub>g</sub>ories. F<sub>or</sub> th<sub>e</sub> f<sub>our</sub> <sub>pr</sub>i<sub>or</sub> <sub>me</sub>th<sub>o</sub>d<sub>s,</sub> f<sub>a</sub>il<sub>ures</sub> <sub>were</sub> <sub>pre</sub>d<sub>om</sub>i<sub>nan</sub>tl<sub>y</sub> <sub>a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d t<sub>o</sub> <sub>wor</sub>k<sub>er-agen</sub>t <sub>execu</sub>ti<sub>on.</sub> I<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> f<sub>a</sub>il<sub>ures</sub> i<sub>n</sub> ORCH <sub>were</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>across wor</sub>k<sub>er execu</sub>ti<sub>on,</sub> i<sub>ncorrec</sub>t t<sub>as</sub>k <sub>a</sub>ll<sub>oca</sub>ti<sub>on, an</sub>d i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>rep</sub>l<sub>ann</sub>i<sub>ng,</sub> i<sub>n</sub>di<sub>ca</sub>ti<sub>ng a re</sub>d<sub>uce</sub>d <sub>concen</sub>t<sub>ra</sub>ti<sub>on o</sub>f f<sub>a</sub>il<sub>ures a</sub>t th<sub>e wor</sub>k<sub>er-execu</sub>ti<sub>on</sub> l<sub>eve</sub>l <sub>w</sub>ith th<sub>e pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e o</sub>f <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>es</sub>i<sub>gn.</sub> 20

Team Hierarchy Generation Process for Scale\_Level\_Complex Task with DeepSeek-V4-Pro

![](images/2c1f54cb9f530802dcedcb3da740625ed9f9f5d2086a10c8a866c538484f6fff.jpg)  
Fi<sub>gure</sub> 6<sub>:</sub> T<sub>eam</sub> Hi<sub>erarc</sub>h<sub>y</sub> G<sub>enera</sub>ti<sub>on</sub> f<sub>or</sub> S<sub>ca</sub>l<sub>e</sub> L<sub>eve</sub>l C<sub>omp</sub>l<sub>ex</sub> T<sub>as</sub>k <sub>w</sub>ith H<sub>uman</sub> E<sub>xper</sub>t <sub>a</sub>nd D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>V4<sub>-</sub>Pr<sub>o.</sub> W<sub>e o</sub>b<sub>serve</sub>d th<sub>a</sub>t th<sub>e</sub> h<sub>uman exper</sub>t d<sub>es</sub>i<sub>gne</sub>d th<sub>e mos</sub>t <sub>reasona</sub>bl<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y,</sub> <sub>w</sub>h<sub>ere</sub> <sub>eac</sub>h <sub>su</sub>b<sub>-</sub>t<sub>eam</sub> <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d i<sub>n</sub> <sub>a</sub> <sub>cer</sub>t<sub>a</sub>i<sub>n</sub> <sub>aspec</sub>t <sub>o</sub>f th<sub>e</sub> t<sub>as</sub>k<sub>.</sub> Wh<sub>en</sub> <sub>us</sub>i<sub>ng</sub> <sub>an</sub> LLM t<sub>o</sub> <sub>genera</sub>t<sub>e</sub> th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y,</sub> <sub>w</sub>ith<sub>ou</sub>t th<sub>e</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> <sub>o</sub>f <sub>a</sub> <sub>cr</sub>iti<sub>c</sub> <sub>agen</sub>t<sub>,</sub> th<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> b<sub>e</sub> <sub>overcomp</sub>li<sub>ca</sub>t<sub>e</sub>d<sub>.</sub> With th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>o</sub>f th<sub>e</sub> <sub>cr</sub>iti<sub>c</sub> <sub>agen</sub>t<sub>,</sub> th<sub>e</sub> LLM <sub>can</sub> <sub>genera</sub>t<sub>e</sub> <sub>a</sub> <sub>more</sub> <sub>reasona</sub>bl<sub>e</sub> <sub>an</sub>d b<sub>a</sub>l<sub>ance</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>on.</sub>

<sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> <sub>an</sub>d <sub>execu</sub>ti<sub>on</sub> b<sub>ur</sub>d<sub>en</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub>ddi<sub>ng</sub> <sub>use</sub>f<sub>u</sub>l <sub>coor</sub>di<sub>na</sub>ti<sub>on.</sub> Th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>genera</sub>t<sub>e</sub>d t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y</sub> f<sub>or eac</sub>h LLM <sub>on eac</sub>h t<sub>as</sub>k <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> Fi<sub>gure</sub> S9<sub>–</sub>S16<sub>, an</sub>d th<sub>e correspon</sub>di<sub>ng</sub> t<sub>as</sub>k <sub>per</sub>f<sub>ormance</sub> i<sub>s</sub> <sub>presen</sub>t<sub>e</sub>d i<sub>n</sub> Fi<sub>gure</sub> S17<sub>–</sub>S24<sub>.</sub>

Th<sub>ese</sub> <sub>resu</sub>lt<sub>s</sub> <sub>revea</sub>l t<sub>wo</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>ons.</sub> Fi<sub>rs</sub>t<sub>,</sub> ORCH’<sub>s</sub> k<sub>ey</sub> id<sub>ea</sub> <sub>o</sub>f b<sub>a</sub>ki<sub>ng</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory</sub> i<sub>n</sub>t<sub>o</sub> hi<sub>erarc</sub>hi<sub>ca</sub>l t<sub>eam</sub> d<sub>es</sub>i<sub>gn</sub> b<sub>r</sub>i<sub>ngs</sub> <sub>a</sub> <sub>c</sub>l<sub>ear</sub> <sub>a</sub>d<sub>van</sub>t<sub>age,</sub> <sub>as</sub> th<sub>e</sub> <sub>per</sub>f<sub>ormance</sub> i<sub>mprovemen</sub>t<sub>s</sub> <sub>are</sub> <sub>seen</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub>ll ORCH’<sub>s</sub> t<sub>eams.</sub> S<sub>econ</sub>d<sub>,</sub> <sub>au</sub>t<sub>oma</sub>t<sub>e</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>es</sub>i<sub>gn</sub> i<sub>s</sub> <sub>prom</sub>i<sub>s</sub>i<sub>ng</sub> b<sub>u</sub>t <sub>s</sub>till l<sub>eaves</sub> <sub>wor</sub>k t<sub>o</sub> d<sub>o.</sub> A <sub>cr</sub>iti<sub>c</sub> <sub>can</sub> i<sub>mprove</sub> <sub>genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>hi<sub>es</sub> b<sub>y</sub> <sub>en</sub>f<sub>orc</sub>i<sub>ng</sub> f<sub>eas</sub>ibilit<sub>y</sub> <sub>an</sub>d <sub>s</sub>i<sub>mp</sub>li<sub>c</sub>it<sub>y.</sub> H<sub>owever,</sub> <sub>a</sub> <sub>gap</sub> <sub>s</sub>till <sub>rema</sub>i<sub>ns</sub> b<sub>e</sub>t<sub>ween</sub> LLM<sub>-genera</sub>t<sub>e</sub>d <sub>an</sub>d <sub>exper</sub>t<sub>-</sub>d<sub>es</sub>i<sub>gne</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ons.</sub> Thi<sub>s</sub> <sub>gap</sub> d<sub>e</sub>fi<sub>nes</sub> <sub>a</sub> <sub>concre</sub>t<sub>e</sub> f<sub>ron</sub>ti<sub>er</sub> f<sub>or</sub> f<sub>u</sub>t<sub>ure</sub> <sub>researc</sub>h i<sub>n</sub> <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence.</sub>

## A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f dif<sub>eren</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gns</sub>

T<sub>o</sub> f<sub>ur</sub>th<sub>er</sub> <sub>quan</sub>tit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>ana</sub>l<sub>yze</sub> th<sub>e</sub> dif<sub>erences</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es</sub> <sub>genera</sub>t<sub>e</sub>d <sub>w</sub>ith dif<sub>eren</sub>t <sub>approac</sub>h<sub>es,</sub> <sub>we</sub> <sub>per</sub>f<sub>orm</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>on</sub> <sub>eac</sub>h t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y.</sub> B<sub>ecause</sub> th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es</sub> <sub>are</sub> <sub>represen</sub>t<sub>e</sub>d <sub>as</sub> <sub>roo</sub>t<sub>e</sub>d t<sub>rees,</sub> <sub>we</sub> <sub>quan</sub>tit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>compare</sub> th<sub>e</sub>i<sub>r</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> usin<sub>g</sub> six tree-based metrics. 1) Tree de<sub>p</sub>th measures the maximum number of hierarchical levels from the root mana<sub>g</sub>er to a leaf worker. 2) Avera<sub>g</sub>e worker de<sub>p</sub>th measures the avera<sub>g</sub>e hierarchical distance between workers and the root mana<sub>g</sub>er. 3) Avera<sub>g</sub>e s<sub>p</sub>an of control measures the avera<sub>g</sub>e number of direct children su<sub>p</sub>ervised b<sub>y</sub> each mana<sub>g</sub>er. 4) The variance of it ca<sub>p</sub>tures how evenl<sub>y</sub> su<sub>p</sub>ervisor<sub>y</sub> res<sub>p</sub>onsibilit<sub>y</sub> is distributed across mana<sub>g</sub>ers. 5) Mana<sub>g</sub>er count measures the number of mana<sub>g</sub>erial a<sub>g</sub>ents in the hierarch<sub>y</sub>. 6) Worker-de<sub>p</sub>th variance characterizes the consistenc<sub>y</sub> of <sub>wor</sub>k<sub>ers</sub>’ hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>pos</sub>iti<sub>ons.</sub> A<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 5B<sub>,</sub> th<sub>e</sub> h<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> h<sub>a</sub>d th<sub>e</sub> lowest avera<sub>g</sub>e tree de<sub>p</sub>th (1.20), com<sub>p</sub>ared with LLM-<sub>g</sub>enerated hierarchies with critic su<sub>p</sub>ervision (1.31) and without critic su<sub>p</sub>ervision (1.56). A similar <sub>p</sub>attern was observed for avera<sub>g</sub>e worker d<sub>ep</sub>th<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>ncrease</sub>d f<sub>rom</sub> 1<sub>.</sub>20 t<sub>o</sub> 1<sub>.</sub>26 <sub>an</sub>d 1<sub>.</sub>52<sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> H<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> also exhibited the lowest s<sub>p</sub>an-of-control variance (1.48), followed b<sub>y</sub> LLM-<sub>g</sub>enerated hierarchies with critic su<sub>p</sub>ervision (5.08) and those without critic su<sub>p</sub>ervision (7.48). Worker-de<sub>p</sub>th variance followed the same orderin<sub>g</sub> (0.00, 0.025, and 0.031, res<sub>p</sub>ectivel<sub>y</sub>). Hierarchies <sub>g</sub>enerated without critic su<sub>p</sub>ervision also contained the lar<sub>g</sub>est avera<sub>g</sub>e number of mana<sub>g</sub>ers (2.08), com<sub>p</sub>ared with 1<sub>.</sub>68 f<sub>or</sub> h<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>an</sub>d 1<sub>.</sub>60 f<sub>or</sub> <sub>cr</sub>iti<sub>c-superv</sub>i<sub>se</sub>d LLM<sub>-genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>hi<sub>es.</sub>

O<sub>vera</sub>ll<sub>,</sub> th<sub>ese s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>me</sub>t<sub>r</sub>i<sub>cs revea</sub>l <sub>a cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>n</sub> hi<sub>erarc</sub>h<sub>y qua</sub>lit<sub>y.</sub> H<sub>uman-</sub> d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>ex</sub>hibit th<sub>e</sub> <sub>mos</sub>t <sub>compac</sub>t <sub>an</sub>d b<sub>a</sub>l<sub>ance</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ures,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> LLM<sub>-genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>w</sub>ith <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on,</sub> <sub>w</sub>h<sub>ereas</sub> hi<sub>erarc</sub>hi<sub>es</sub> <sub>genera</sub>t<sub>e</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> <sub>are</sub> <sub>genera</sub>ll<sub>y</sub> d<sub>eeper</sub> <sub>an</sub>d <sub>more</sub> <sub>uneven.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> th<sub>e</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> l<sub>ower</sub> <sub>span-o</sub>f<sub>-</sub> <sub>con</sub>t<sub>ro</sub>l <sub>an</sub>d <sub>wor</sub>k<sub>er-</sub>d<sub>ep</sub>th <sub>var</sub>i<sub>ances o</sub>f th<sub>e</sub> h<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e a more un</sub>if<sub>orm</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f <sub>superv</sub>i<sub>sory</sub> <sub>respons</sub>ibilit<sub>y</sub> <sub>an</sub>d <sub>more</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>p</sub>l<sub>acemen</sub>t <sub>o</sub>f <sub>wor</sub>k<sub>ers</sub> <sub>across</sub> hi<sub>er-</sub> <sub>arc</sub>hi<sub>ca</sub>l l<sub>eve</sub>l<sub>s.</sub> C<sub>r</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on</sub> <sub>moves</sub> th<sub>e</sub> LLM<sub>-genera</sub>t<sub>e</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> <sub>c</sub>l<sub>oser</sub> t<sub>o</sub> th<sub>ese</sub> h<sub>uman-</sub> d<sub>es</sub>i<sub>gne</sub>d <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> b<sub>y</sub> <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> hi<sub>erarc</sub>h<sub>y</sub> d<sub>ep</sub>th<sub>,</sub> <sub>wor</sub>k<sub>er</sub> d<sub>ep</sub>th<sub>,</sub> <sub>an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l i<sub>m</sub>b<sub>a</sub>l<sub>ance</sub> <sub>com-</sub> <sub>pare</sub>d <sub>w</sub>ith <sub>genera</sub>ti<sub>on</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on.</sub> Th<sub>ese</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l dif<sub>erences</sub> <sub>are</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith th<sub>e</sub> <sub>per</sub>f<sub>ormance</sub> <sub>resu</sub>lt<sub>s,</sub> <sub>w</sub>h<sub>ere</sub> h<sub>uman-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>ac</sub>hi<sub>eve</sub> th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t <sub>per</sub>f<sub>ormance</sub> <sub>an</sub>d <sub>cr</sub>iti<sub>c-superv</sub>i<sub>se</sub>d LLM hi<sub>erarc</sub>hi<sub>es</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orm</sub> th<sub>ose</sub> <sub>genera</sub>t<sub>e</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>cr</sub>iti<sub>c</sub> <sub>superv</sub>i<sub>s</sub>i<sub>on,</sub> <sub>sugges</sub>t<sub>-</sub> i<sub>ng</sub> th<sub>a</sub>t <sub>compac</sub>t <sub>an</sub>d b<sub>a</sub>l<sub>ance</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> t<sub>o</sub> <sub>more</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>mu</sub>lti<sub>-agen</sub>t <sub>coor</sub>di<sub>na</sub>ti<sub>on.</sub>

## C<sub>o</sub>mm<sub>u</sub>ni<sub>ca</sub>ti<sub>o</sub>n Hi<sub>s</sub>t<sub>o</sub>r<sub>y a</sub>nd F<sub>a</sub>il<sub>u</sub>r<sub>e</sub> M<sub>o</sub>d<sub>e</sub> An<sub>a</sub>l<sub>ys</sub>i<sub>s</sub>

W<sub>e</sub> <sub>quan</sub>tit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>ana</sub>l<sub>yze</sub>d th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>es</sub> <sub>o</sub>f th<sub>e</sub> <sub>mu</sub>lti<sub>-agen</sub>t <sub>sys</sub>t<sub>ems</sub> t<sub>o</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> th<sub>e</sub> f<sub>a</sub>il<sub>ure mo</sub>d<sub>es o</sub>f dif<sub>eren</sub>t <sub>coor</sub>di<sub>na</sub>ti<sub>on me</sub>th<sub>o</sub>d<sub>s.</sub> W<sub>e use</sub>d D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>V4<sub>-</sub>P<sub>ro, one</sub> <sub>o</sub>f th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>open-we</sub>i<sub>g</sub>ht <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>a</sub>t th<sub>e</sub> ti<sub>me</sub> <sub>o</sub>f th<sub>e</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s,</sub> t<sub>o</sub> <sub>ana</sub>l<sub>yze</sub> <sub>unsuccess</sub>f<sub>u</sub>l <sub>ep</sub>i<sub>so</sub>d<sub>es</sub> b<sub>y</sub> <sub>prov</sub>idi<sub>ng</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> hi<sub>s</sub>t<sub>ory</sub> <sub>an</sub>d <sub>as</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> <sub>c</sub>l<sub>ass</sub>if<sub>y</sub> th<sub>e</sub> <sub>pr</sub>i<sub>mary</sub> f<sub>a</sub>il<sub>ure</sub> i<sub>n</sub>t<sub>o</sub> <sub>one</sub> <sub>o</sub>f fi<sub>ve</sub> <sub>ca</sub>t<sub>egor</sub>i<sub>es:</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>rep</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>or</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>manager</sub> f<sub>a</sub>il<sub>ure</sub> t<sub>o</sub> <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>d th<sub>e</sub> t<sub>as</sub>k<sub>,</sub> <sub>uneven</sub> <sub>or</sub> i<sub>ncorrec</sub>t t<sub>as</sub>k <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> <sub>wor</sub>k<sub>er</sub> <sub>execu</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ure,</sub> <sub>an</sub>d <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>or commun</sub>i<sub>ca</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ure.</sub> A<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 5C<sub>, across</sub> 1<sub>,</sub>000 <sub>runs per me</sub>th<sub>o</sub>d<sub>,</sub> ORCH <sub>ac</sub>hi<sub>eve</sub>d <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> hi<sub>g</sub>h<sub>er</sub> <sub>success</sub> <sub>ra</sub>t<sub>es</sub> th<sub>an</sub> th<sub>e</sub> f<sub>our</sub> b<sub>ase</sub>li<sub>nes.</sub> ORCH <sub>w</sub>ith h<sub>uman-exper</sub>t<sub>-</sub>d<sub>es</sub>i<sub>gne</sub>d<sub>,</sub> LLM<sub>-genera</sub>t<sub>e</sub>d<sub>,</sub> <sub>an</sub>d LLM<sub>-genera</sub>t<sub>e</sub>d<sub>-w</sub>ith<sub>ou</sub>t<sub>-cr</sub>iti<sub>c</sub> hi<sub>erarc</sub>hi<sub>es</sub> <sub>ac</sub>hi<sub>eve</sub>d <sub>success</sub> <sub>ra</sub>t<sub>es</sub> <sub>o</sub>f 53<sub>.</sub>1%<sub>,</sub> 42<sub>.</sub>2%<sub>,</sub> <sub>an</sub>d 35<sub>.</sub>2%<sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>compare</sub>d <sub>w</sub>ith <sub>on</sub>l<sub>y</sub> 6<sub>.</sub>9<sub>–</sub>13<sub>.</sub>5% f<sub>or</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>nes.</sub> B<sub>ase</sub>li<sub>ne</sub> f<sub>a</sub>il<sub>ures</sub> were dominated b<sub>y</sub> worker execution (51.2–64.5%) and incorrect task allocation (14.7–28.1%), <sub>w</sub>h<sub>ereas</sub> ORCH <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>re</sub>d<sub>uce</sub>d <sub>wor</sub>k<sub>er</sub> <sub>execu</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ures</sub> t<sub>o</sub> 13<sub>.</sub>1<sub>–</sub>20<sub>.</sub>7%<sub>.</sub>

C<sub>ompare</sub>d <sub>w</sub>ith th<sub>e</sub> f<sub>our</sub> b<sub>ase</sub>li<sub>nes,</sub> <sub>a</sub>ll th<sub>ree</sub> ORCH <sub>var</sub>i<sub>an</sub>t<sub>s</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>re</sub>d<sub>uce</sub>d <sub>wor</sub>k<sub>er</sub> <sub>exe-</sub> <sub>cu</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ures</sub> f<sub>rom</sub> 51<sub>.</sub>2<sub>–</sub>64<sub>.</sub>5% t<sub>o</sub> 13<sub>.</sub>1<sub>–</sub>20<sub>.</sub>7%<sub>,</sub> i<sub>n</sub>di<sub>ca</sub>ti<sub>ng</sub> th<sub>a</sub>t hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>coor</sub>di<sub>na</sub>ti<sub>on prov</sub>id<sub>es</sub> <sub>wor</sub>k<sub>ers</sub> <sub>w</sub>ith b<sub>e</sub>tt<sub>er-s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>an</sub>d <sub>more</sub> <sub>ac</sub>ti<sub>ona</sub>bl<sub>e</sub> t<sub>as</sub>k<sub>s</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>execu</sub>ti<sub>on.</sub> I<sub>ns</sub>t<sub>ea</sub>d<sub>,</sub> f<sub>a</sub>il<sub>ures</sub> i<sub>n</sub> ORCH <sub>were</sub> <sub>more</sub> <sub>concen</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub> hi<sub>g</sub>h<sub>er-</sub>l<sub>eve</sub>l t<sub>as</sub>k <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>an</sub>d <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>sugges</sub>ti<sub>ng</sub> th<sub>a</sub>t it<sub>s</sub> <sub>pr</sub>i<sub>mary</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>ons</sub> <sub>ar</sub>i<sub>se</sub> f<sub>rom</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>over</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>wor</sub>k<sub>er</sub> <sub>execu</sub>ti<sub>on.</sub> C<sub>oor</sub>di<sub>-</sub> nation and communication failures were also consistentl<sub>y</sub> low across all ORCH variants (0.7%), <sub>compare</sub>d <sub>w</sub>ith <sub>up</sub> t<sub>o</sub> 7<sub>.</sub>1% f<sub>or</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>nes.</sub> Th<sub>ese</sub> <sub>resu</sub>lt<sub>s</sub> <sub>sugges</sub>t th<sub>a</sub>t ORCH’<sub>s</sub> hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>organ</sub>i<sub>-</sub> <sub>za</sub>ti<sub>on</sub> i<sub>mproves</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> fl<sub>ow,</sub> t<sub>as</sub>k d<sub>e</sub>l<sub>ega</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>wor</sub>k<sub>er</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on,</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>wor</sub>k<sub>er-</sub>l<sub>eve</sub>l <sub>execu</sub>ti<sub>on</sub> <sub>an</sub>d <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ures</sub> th<sub>a</sub>t d<sub>om</sub>i<sub>na</sub>t<sub>e</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>nes.</sub>

Withi<sub>n</sub> ORCH<sub>,</sub> th<sub>e</sub> th<sub>ree var</sub>i<sub>an</sub>t<sub>s</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t hi<sub>erarc</sub>h<sub>y</sub> d<sub>es</sub>i<sub>gn</sub> i<sub>s cr</sub>iti<sub>ca</sub>l i<sub>n</sub> d<sub>e</sub>t<sub>er-</sub> <sub>m</sub>i<sub>n</sub>i<sub>ng</sub> f<sub>a</sub>il<sub>ure</sub> <sub>mo</sub>d<sub>es.</sub> T<sub>as</sub>k<sub>-a</sub>ll<sub>oca</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ures</sub> i<sub>ncrease</sub>d <sub>s</sub>h<sub>arp</sub>l<sub>y</sub> f<sub>rom</sub> 8<sub>.</sub>4% <sub>w</sub>ith th<sub>e</sub> h<sub>uman-exper</sub>t hi<sub>erarc</sub>h<sub>y</sub> t<sub>o</sub> 21<sub>.</sub>0% <sub>w</sub>ith th<sub>e</sub> LLM<sub>-genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>h<sub>y</sub> <sub>an</sub>d 29<sub>.</sub>6% <sub>w</sub>ith<sub>ou</sub>t th<sub>e</sub> <sub>cr</sub>iti<sub>c,</sub> <sub>w</sub>hil<sub>e</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a-</sub> ti<sub>on</sub> f<sub>a</sub>il<sub>ures</sub> i<sub>ncrease</sub>d f<sub>rom</sub> 13<sub>.</sub>6% t<sub>o</sub> 17<sub>.</sub>6% <sub>an</sub>d 18<sub>.</sub>8%<sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> Thi<sub>s</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t d<sub>egra</sub>d<sub>a</sub>ti<sub>on</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t ORCH d<sub>epen</sub>d<sub>s</sub> <sub>s</sub>t<sub>rong</sub>l<sub>y</sub> <sub>on</sub> hi<sub>erarc</sub>h<sub>y</sub> <sub>qua</sub>lit<sub>y,</sub> <sub>w</sub>ith b<sub>e</sub>tt<sub>er-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> <sub>ena</sub>bli<sub>ng</sub> <sub>more</sub> <sub>re</sub>li<sub>a</sub>bl<sub>e</sub> t<sub>as</sub>k d<sub>ecompos</sub>iti<sub>on,</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>mu</sub>lti<sub>-agen</sub>t <sub>execu</sub>ti<sub>on.</sub>

## Di<sub>scuss</sub>i<sub>ons</sub>

O<sub>ur</sub> <sub>resu</sub>lt<sub>s</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>rom</sub> h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory</sub> <sub>can</sub> i<sub>mprove</sub> th<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub> <sub>o</sub>f <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>agen</sub>t<sub>s.</sub> A<sub>cross</sub> 25 <sub>m</sub>i<sub>ss</sub>i<sub>ons,</sub> <sub>e</sub>i<sub>g</sub>ht l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>repea</sub>t<sub>e</sub>d <sub>see</sub>d<sub>s,</sub> <sub>an</sub>d f<sub>our</sub> <sub>compar</sub>i<sub>son</sub> <sub>sys</sub>t<sub>ems,</sub> t<sub>eams</sub> <sub>organ</sub>i<sub>ze</sub>d <sub>aroun</sub>d <sub>poo</sub>l<sub>e</sub>d <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>ac</sub>hi<sub>eve</sub>d <sub>s</sub>t<sub>ronger</sub> <sub>ou</sub>t<sub>comes,</sub> <sub>accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>progress</sub> <sub>more</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y,</sub> <sub>exp</sub>l<sub>ore</sub>d <sub>more</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>an</sub>d <sub>use</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>resources</sub> <sub>compe</sub>titi<sub>ve</sub>l<sub>y.</sub> Th<sub>e</sub> <sub>a</sub>d<sub>van</sub>t<sub>ages</sub> <sub>pers</sub>i<sub>s</sub>t<sub>e</sub>d <sub>across</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d t<sub>as</sub>k <sub>var</sub>i<sub>a</sub>ti<sub>ons,</sub> <sub>an</sub>d b<sub>ecame</sub> <sub>espec</sub>i<sub>a</sub>ll<sub>y</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>n</sub> <sub>comp</sub>l<sub>ex</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons</sub> th<sub>a</sub>t <sub>requ</sub>i<sub>re</sub>d l<sub>arge</sub> h<sub>e</sub>t<sub>erogeneous</sub> t<sub>eams</sub> t<sub>o coor</sub>di<sub>na</sub>t<sub>e para</sub>ll<sub>e</sub>l <sub>an</sub>d <sub>or</sub>d<sub>ere</sub>d <sub>wor</sub>k<sub>.</sub>

Th<sub>ese</sub> fi<sub>n</sub>di<sub>ngs</sub> f<sub>ur</sub>th<sub>er</sub> <sub>revea</sub>l <sub>an</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>aspec</sub>t <sub>o</sub>f th<sub>e</sub> <sub>un</sub>it <sub>o</sub>f <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> i<sub>n</sub> l<sub>arge-</sub>l<sub>anguage-mo</sub>d<sub>e</sub>l<sub>-</sub> d<sub>r</sub>i<sub>ven mu</sub>lti<sub>-agen</sub>t <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>sys</sub>t<sub>ems.</sub> Th<sub>e per</sub>f<sub>ormance o</sub>f <sub>a co</sub>ll<sub>ec</sub>ti<sub>ve</sub> i<sub>s o</sub>ft<sub>en a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence o</sub>f it<sub>s mem</sub>b<sub>ers,</sub> th<sub>e qua</sub>lit<sub>y o</sub>f th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>po</sub>li<sub>c</sub>i<sub>es, or</sub> th<sub>e amoun</sub>t <sub>o</sub>f <sub>commun</sub>i<sub>ca</sub>ti<sub>on</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> t<sub>o</sub> th<sub>em.</sub> ORCH <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> <sub>arrangemen</sub>t <sub>o</sub>f <sub>au</sub>th<sub>or</sub>it<sub>y,</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>respons</sub>ibilit<sub>y</sub> i<sub>s</sub> <sub>a</sub>l<sub>so an</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e.</sub> Th<sub>e same wor</sub>k<sub>ers an</sub>d <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> l<sub>anguage mo</sub>d<sub>e</sub>l <sub>can pro</sub>d<sub>uce</sub> dif<sub>eren</sub>t <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>ou</sub>t<sub>comes</sub> <sub>w</sub>h<sub>en</sub> <sub>p</sub>l<sub>ace</sub>d <sub>un</sub>d<sub>er</sub> dif<sub>eren</sub>t <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es.</sub>

Th<sub>e</sub> <sub>emp</sub>h<sub>as</sub>i<sub>s</sub> <sub>on</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory</sub> i<sub>s</sub> <sub>an</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> <sub>use</sub>d t<sub>o</sub> <sub>gu</sub>id<sub>e</sub> h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons</sub> <sub>can</sub> <sub>o</sub>f<sub>er</sub> <sub>s</sub>t<sub>rong</sub> <sub>prom</sub>i<sub>ses</sub> t<sub>o</sub> <sub>s</sub>t<sub>reng</sub>th<sub>en</sub> <sub>mu</sub>lti<sub>-agen</sub>t AI t<sub>eams.</sub> W<sub>e</sub> t<sub>urn</sub> <sub>poo</sub>l<sub>e</sub>d <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence</sub> i<sub>n</sub>t<sub>o</sub> <sub>execu</sub>t<sub>a</sub>bl<sub>e</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sms.</sub> P<sub>oo</sub>l<sub>e</sub>d <sub>wor</sub>k i<sub>s</sub> <sub>repre-</sub> <sub>sen</sub>t<sub>e</sub>d b<sub>y</sub> <sub>managers</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub> <sub>concurren</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>wor</sub>k i<sub>s</sub> <sub>represen</sub>t<sub>e</sub>d b<sub>y</sub> <sub>managers</sub> th<sub>a</sub>t <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub> <sub>or</sub>d<sub>ere</sub>d <sub>p</sub>h<sub>ases</sub> <sub>an</sub>d <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>con</sub>diti<sub>ons.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>com</sub>bi<sub>na</sub>ti<sub>on</sub> <sub>a</sub>ll<sub>ows</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> t<sub>o</sub> <sub>m</sub>i<sub>rror</sub> th<sub>e</sub> <sub>m</sub>i<sub>xe</sub>d d<sub>epen</sub>d<sub>ency</sub> <sub>pa</sub>tt<sub>erns</sub> <sub>o</sub>f <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>ons.</sub> Th<sub>oug</sub>h th<sub>e</sub> <sub>govern</sub>i<sub>ng</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> i<sub>s</sub> <sub>concre</sub>t<sub>e,</sub> th<sub>e</sub> hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>an</sub>d “t<sub>eams</sub> <sub>o</sub>f t<sub>eams</sub>” <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>a</sub>ll<sub>ows</sub> <sub>recurs</sub>i<sub>ve</sub> <sub>an</sub>d <sub>comp</sub>l<sub>ex</sub> <sub>compos</sub>iti<sub>ons</sub> <sub>o</sub>f l<sub>arge</sub> h<sub>e</sub>t<sub>erogeneous</sub> t<sub>eams</sub> t<sub>o</sub> <sub>sca</sub>l<sub>e</sub> t<sub>o</sub> <sub>comp</sub>li<sub>ca</sub>t<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>ons.</sub>

W<sub>e</sub> fi<sub>n</sub>d <sub>an</sub> i<sub>n</sub>t<sub>eres</sub>ti<sub>ng</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>p</sub> b<sub>e</sub>t<sub>ween</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>an</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>capa</sub>bilit<sub>y.</sub> Th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l h<sub>a</sub>d <sub>a</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>e</sub>f<sub>ec</sub>t <sub>on</sub> <sub>per</sub>f<sub>ormance,</sub> b<sub>u</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>sca</sub>l<sub>e</sub> did <sub>no</sub>t <sub>mono</sub>t<sub>on</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>pre</sub>di<sub>c</sub>t th<sub>e</sub> <sub>qua</sub>lit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve.</sub> M<sub>o</sub>d<sub>era</sub>t<sub>e</sub>l<sub>y</sub> <sub>s</sub>i<sub>ze</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>cou</sub>ld <sub>ou</sub>t<sub>per</sub>f<sub>orm</sub> l<sub>arger</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>un</sub>d<sub>er</sub> <sub>o</sub>th<sub>erw</sub>i<sub>se</sub> th<sub>e same con</sub>diti<sub>ons w</sub>h<sub>en</sub> b<sub>e</sub>i<sub>ng u</sub>tili<sub>ze</sub>d <sub>as a co</sub>ll<sub>ec</sub>ti<sub>ve w</sub>ith <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>gu</sub>id<sub>ance.</sub> Thi<sub>s</sub> <sub>sugges</sub>t<sub>s</sub> th<sub>a</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>o</sub>f l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>as</sub> i<sub>so</sub>l<sub>a</sub>t<sub>e</sub>d <sub>reasoners</sub> <sub>may</sub> <sub>no</sub>t f<sub>u</sub>ll<sub>y</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> th<sub>e</sub>i<sub>r</sub> <sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> <sub>par</sub>ti<sub>c</sub>i<sub>pa</sub>t<sub>e</sub> i<sub>n</sub> <sub>organ</sub>i<sub>ze</sub>d <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>sys</sub>t<sub>ems.</sub> C<sub>o</sub>ll<sub>ec</sub>ti<sub>ve</sub> <sub>per</sub>f<sub>ormance</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>proper</sub>ti<sub>es</sub> <sub>suc</sub>h <sub>as</sub> <sub>ro</sub>l<sub>e</sub> <sub>a</sub>dh<sub>erence,</sub> <sub>conc</sub>i<sub>se</sub> <sub>repor</sub>ti<sub>ng,</sub> <sub>p</sub>l<sub>an</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>over</sub> <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l

d<sub>epen</sub>d<sub>enc</sub>i<sub>es.</sub>

Th<sub>ere are severa</sub>l f<sub>u</sub>t<sub>ure oppor</sub>t<sub>un</sub>iti<sub>es</sub> t<sub>o</sub> i<sub>mprove our wor</sub>k<sub>.</sub> Alth<sub>oug</sub>h th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 25 t<sub>as</sub>k<sub>s</sub> <sub>w</sub>ith <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> <sub>sca</sub>l<sub>e,</sub> <sub>wor</sub>kf<sub>orce,</sub> <sub>an</sub>d d<sub>epen</sub>d<sub>ency</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure,</sub> th<sub>e</sub> <sub>ex</sub>t<sub>en</sub>t t<sub>o</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>same</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> <sub>may</sub> t<sub>rans</sub>f<sub>er</sub> t<sub>o</sub> <sub>o</sub>th<sub>er</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d d<sub>oma</sub>i<sub>ns</sub> <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> b<sub>e</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d<sub>.</sub> Wh<sub>en more</sub> l<sub>arge-sca</sub>l<sub>e</sub> l<sub>anguage-</sub>b<sub>ase</sub>d <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d t<sub>es</sub>tb<sub>e</sub>d<sub>s</sub> b<sub>ecome ava</sub>il<sub>a</sub>bl<sub>e, eva</sub>l<sub>ua</sub>ti<sub>ons across</sub> <sub>more scenar</sub>i<sub>os w</sub>ill b<sub>e va</sub>l<sub>ua</sub>bl<sub>e.</sub> T<sub>o suppor</sub>t thi<sub>s e</sub>f<sub>or</sub>t<sub>, we</sub> h<sub>ave pu</sub>bli<sub>c</sub>l<sub>y re</sub>l<sub>ease</sub>d <sub>our so</sub>ft<sub>ware</sub> <sub>on</sub>li<sub>ne.</sub> M<sub>oreover,</sub> th<sub>e</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l hi<sub>erarc</sub>h<sub>y</sub> i<sub>s</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> b<sub>e</sub>f<sub>ore</sub> <sub>execu</sub>ti<sub>on.</sub> Th<sub>oug</sub>h <sub>p</sub>l<sub>ans, ass</sub>i<sub>gnmen</sub>t<sub>s, an</sub>d <sub>ac</sub>ti<sub>ve p</sub>h<sub>ases a</sub>d<sub>ap</sub>t <sub>on</sub>li<sub>ne,</sub> th<sub>e presen</sub>t f<sub>ramewor</sub>k d<sub>oes no</sub>t <sub>ye</sub>t <sub>res</sub>t<sub>ruc</sub>t<sub>ure</sub> th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> it<sub>se</sub>lf <sub>a</sub>l<sub>ong</sub> th<sub>e m</sub>i<sub>ss</sub>i<sub>on progress</sub>i<sub>on.</sub> Th<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on a</sub>l<sub>so ma</sub>i<sub>n</sub>l<sub>y</sub> f<sub>ocuses</sub> <sub>on</sub> <sub>poo</sub>l<sub>e</sub>d <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence.</sub> Oth<sub>er</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps,</sub> <sub>suc</sub>h <sub>as</sub> <sub>rec</sub>i<sub>proca</sub>l <sub>an</sub>d <sub>con</sub>ti<sub>nuous</sub>l<sub>y</sub> <sub>nego</sub>ti<sub>a</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ence,</sub> <sub>may</sub> <sub>requ</sub>i<sub>re</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>manager</sub> t<sub>ypes</sub> <sub>or commun</sub>i<sub>ca</sub>ti<sub>on pro</sub>t<sub>oco</sub>l<sub>s</sub> i<sub>n</sub> f<sub>u</sub>t<sub>ure s</sub>t<sub>u</sub>di<sub>es.</sub> F<sub>ur</sub>th<sub>ermore,</sub> h<sub>uman-exper</sub>t<sub>-</sub>d<sub>es</sub>i<sub>gne</sub>d hi<sub>erarc</sub>hi<sub>es</sub> r<sub>e</sub>m<sub>a</sub>in <sub>s</sub>tr<sub>o</sub>n<sub>ge</sub>r th<sub>a</sub>n LLM<sub>-ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d <sub>o</sub>n<sub>es</sub> <sub>u</sub>nd<sub>e</sub>r th<sub>e</sub> <sub>sa</sub>m<sub>e</sub> ORCH <sub>p</sub>rin<sub>c</sub>i<sub>p</sub>l<sub>e,</sub> <sub>a</sub>lth<sub>oug</sub>h th<sub>e</sub> LLM <sub>genera</sub>t<sub>e</sub>d <sub>ones</sub> h<sub>ave a</sub>l<sub>rea</sub>d<sub>y ou</sub>t<sub>per</sub>f<sub>orme</sub>d <sub>o</sub>th<sub>er</sub> b<sub>ase</sub>li<sub>nes.</sub> A<sub>u</sub>t<sub>oma</sub>t<sub>e</sub>d <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l d<sub>es</sub>i<sub>gn un</sub>d<sub>er</sub> ORCH’<sub>s</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>prom</sub>i<sub>s</sub>i<sub>ng</sub> b<sub>u</sub>t l<sub>eaves</sub> <sub>space</sub> f<sub>or</sub> f<sub>u</sub>t<sub>ure</sub> i<sub>mprovemen</sub>t<sub>s.</sub> Fi<sub>na</sub>ll<sub>y,</sub> <sub>our</sub> f<sub>a</sub>il<sub>ure</sub> t<sub>axonomy</sub> <sub>was</sub> <sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> <sub>a</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>may</sub> <sub>nee</sub>d t<sub>o</sub> b<sub>e</sub> <sub>va</sub>lid<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>ra</sub>t<sub>ers</sub> t<sub>o</sub> f<sub>ur</sub>th<sub>er</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> <sub>an</sub>d i<sub>mprove</sub> th<sub>e</sub> <sub>sys</sub>t<sub>em</sub> f<sub>or</sub> hi<sub>g</sub>h<sub>-s</sub>t<sub>a</sub>k<sub>es</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>ki<sub>ng.</sub>

D<sub>esp</sub>it<sub>e</sub> th<sub>ese</sub> f<sub>u</sub>t<sub>ure</sub> di<sub>rec</sub>ti<sub>ons,</sub> th<sub>e sca</sub>l<sub>e an</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>ency o</sub>f <sub>our resu</sub>lt<sub>s suppor</sub>t <sub>a c</sub>l<sub>ear v</sub>i<sub>s</sub>i<sub>on:</sub> <sub>a</sub>d<sub>vances</sub> i<sub>n</sub> <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub> <sub>w</sub>ill <sub>requ</sub>i<sub>re</sub> <sub>progress</sub> <sub>no</sub>t <sub>on</sub>l<sub>y</sub> i<sub>n</sub> <sub>ma</sub>ki<sub>ng</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>agen</sub>t<sub>s</sub> <sub>more</sub> <sub>capa</sub>bl<sub>e,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> i<sub>n</sub> di<sub>scover</sub>i<sub>ng</sub> h<sub>ow</sub> th<sub>ose</sub> <sub>agen</sub>t<sub>s</sub> <sub>s</sub>h<sub>ou</sub>ld b<sub>e</sub> <sub>organ</sub>i<sub>ze</sub>d<sub>.</sub> O<sub>ur</sub> <sub>wor</sub>k <sub>sugges</sub>t<sub>s</sub> th<sub>a</sub>t h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons</sub> <sub>can</sub> <sub>o</sub>f<sub>er</sub> <sub>va</sub>l<sub>ua</sub>bl<sub>e</sub> i<sub>ns</sub>i<sub>g</sub>ht<sub>s.</sub> A<sub>s</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>sys</sub>t<sub>ems</sub> <sub>expan</sub>d f<sub>rom</sub> <sub>sma</sub>ll h<sub>omogeneous</sub> <sub>groups</sub> t<sub>o</sub> l<sub>arge</sub> t<sub>eams</sub> <sub>o</sub>f <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>agen</sub>t<sub>s,</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gn</sub> <sub>may</sub> b<sub>ecome</sub> <sub>as</sub> <sub>essen</sub>ti<sub>a</sub>l <sub>as</sub> <sub>mo</sub>d<sub>e</sub>l <sub>arc</sub>hit<sub>ec</sub>t<sub>ure</sub> <sub>or</sub> <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m.</sub> P<sub>r</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> d<sub>eve</sub>l<sub>ope</sub>d t<sub>o</sub> <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>d h<sub>uman</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ons</sub> <sub>can</sub> <sub>prov</sub>id<sub>e</sub> <sub>a</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t f<sub>or</sub> thi<sub>s</sub> <sub>emerg</sub>i<sub>ng</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>sc</sub>i<sub>ence</sub> <sub>o</sub>f <sub>ar</sub>tifi<sub>c</sub>i<sub>a</sub>l <sub>organ</sub>i<sub>za</sub>ti<sub>on.</sub>

## References and Notes

1. J. D. Thompson, Organizations in action: Social science bases ofadministrative theory (Routled<sub>g</sub>e) (2017).

2<sub>.</sub> N<sub>.</sub> W<sub>e</sub>ll<sub>man,</sub> J<sub>.</sub> A<sub>pp</sub>l<sub>ega</sub>t<sub>e,</sub> J<sub>.</sub> H<sub>ar</sub>l<sub>ow,</sub> E<sub>.</sub> W<sub>.</sub> J<sub>o</sub>h<sub>ns</sub>t<sub>on,</sub> B<sub>eyon</sub>d th<sub>e</sub> <sub>pyram</sub>id<sub>:</sub> Alt<sub>erna</sub>ti<sub>ve</sub> f<sub>orma</sub>l hierarchical structures and team performance. Academy of Management Journal 63 (4), 997– 1027 (2020).

3<sub>.</sub> J<sub>.</sub> G<sub>.</sub> M<sub>a</sub>t<sub>us</sub>ik<sub>,</sub> R<sub>.</sub> L<sub>.</sub> Mit<sub>c</sub>h<sub>e</sub>ll<sub>,</sub> N<sub>.</sub> A<sub>.</sub> H<sub>ays,</sub> S<sub>.</sub> F<sub>a</sub>th<sub>,</sub> J<sub>.</sub> R<sub>.</sub> H<sub>o</sub>ll<sub>en</sub>b<sub>ec</sub>k<sub>,</sub> Th<sub>e</sub> hi<sub>g</sub>h<sub>s an</sub>d l<sub>ows o</sub>f hierarchy in multiteam systems. Academy of Management Journal 65 (5), 1571–1592 (2022).

4. J. Ferber, G. Weiss, Multi-agent systems: an introduction to distributed artificial intelligence, vol. 1 (Addison-wesle<sub>y</sub> Readin<sub>g</sub>) (1999).

5. L. Panait, S. Luke, Cooperative multi-agent learning: The state of the art. Autonomous agents and multi-agent systems 11 (3), 387–434 (2005).

6. G. Dudek, M. R. Jenkin, E. Milios, D. Wilkes, A taxonomy for multi-agent robotics. Autonomous Robots 3 (4), 375–397 (1996).

7. M. Jaderberg, et al., Human-level performance in 3D multiplayer games with population-based reinforcement learning. Science 364 (6443), 859–865 (2019).

8. O. Vinyals, et al., Grandmaster level in StarCraft II using multi-agent reinforcement learning. nature 575 (7782), 350–354 (2019).

9. M. F. A. R. D. T. (FAIR)†<sub>,</sub> et al.<sub>,</sub> Human-level <sub>p</sub>la<sub>y</sub> in the <sub>g</sub>ame of Di<sub>p</sub>lomac<sub>y</sub> b<sub>y</sub> combinin<sub>g</sub> language models with strategic reasoning. Science 378 (6624), 1067–1074 (2022).

10. Z. Ji, L. Zhang, P. Sajda, B. Chen, Enabling Multi-Robot Collaboration from Single-Human Guidance, in 2025 IEEE International Conference on Robotics and Automation (ICRA) (2025), <sub>pp.</sub> 4272<sub>–</sub>4279<sub>,</sub> d<sub>o</sub>i<sub>:</sub>10<sub>.</sub>1109/ICRA55743<sub>.</sub>2025<sub>.</sub>11127982<sub>.</sub>

11. G. Wang, et al., Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023).

12. J. S. Park, et al., Generative agents: Interactive simulacra of human behavior, in Proceedings of the 36th annual acm symposium on user interface software and technology (2023), pp. 1–22.

13. S. Hong, et al., MetaGPT: Meta programming for a multi-agent collaborative framework, in International Conference on Learning Representations, vol. 2024 (2024), pp. 23247–23275.

14. Q. Wu, et al., Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155 (2023).

15<sub>.</sub> G<sub>.</sub> Li<sub>,</sub> H<sub>.</sub> H<sub>ammou</sub>d<sub>,</sub> H<sub>.</sub> It<sub>an</sub>i<sub>,</sub> D<sub>.</sub> Khi<sub>z</sub>b<sub>u</sub>lli<sub>n,</sub> B<sub>.</sub> Gh<sub>anem,</sub> C<sub>ame</sub>l<sub>:</sub> C<sub>ommun</sub>i<sub>ca</sub>ti<sub>ve</sub> <sub>agen</sub>t<sub>s</sub> f<sub>or</sub>” mind” exploration of large language model society. Advances in neural information processing systems 36, 51991–52008 (2023).

16. P. Wu, et al., Camon: Cooperative agents for multi-object navigation with llm-based conversations. arXiv preprint arXiv:2407.00632 (2024).

17. X. Guo, et al., Embodied llm agents learn to cooperate in organized teams. IEEE Transactions on Computational Social Systems (2026).

18. H. Zhang, et al., Building cooperative embodied agents modularly with large language models, in International Conference on Learning Representations, vol. 2024 (2024), pp. 19373–19401.

19<sub>.</sub> Y<sub>.</sub> Ch<sub>en,</sub> J<sub>.</sub> A<sub>r</sub>ki<sub>n,</sub> Y<sub>.</sub> Zh<sub>ang,</sub> N<sub>.</sub> R<sub>oy,</sub> C<sub>.</sub> F<sub>an,</sub> S<sub>ca</sub>l<sub>a</sub>bl<sub>e</sub> <sub>mu</sub>lti<sub>-ro</sub>b<sub>o</sub>t <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>w</sub>ith l<sub>arge</sub> l<sub>an-</sub> guage models: Centralized or decentralized systems?, in 2024 IEEE International Conference on Robotics and Automation (ICRA) (IEEE) (2024), pp. 4311–4317.

20. J. Thompson, W. Scott, M. Zald, Organizations in Action: Social Science Bases of Administrative Theory, Organization and Business (Transaction Publishers) (2011), https: //books.google.com/books?id=YhHo7aHmBGMC.

21. B. HORLING, V. LESSER, A survey of multi-agent organizational paradigms. The Knowledge Engineering Review 19 (4), 281–316 (2004), doi:10.1017/S0269888905000317.

22<sub>.</sub> J<sub>.</sub> F<sub>er</sub>b<sub>er,</sub> O<sub>.</sub> G<sub>u</sub>tk<sub>nec</sub>ht<sub>,</sub> F<sub>.</sub> Mi<sub>c</sub>h<sub>e</sub>l<sub>,</sub> F<sub>rom</sub> A<sub>gen</sub>t<sub>s</sub> t<sub>o</sub> O<sub>rgan</sub>i<sub>za</sub>ti<sub>ons: an</sub> O<sub>rgan</sub>i<sub>za</sub>ti<sub>ona</sub>l Vi<sub>ew o</sub>f Multi-Agent Systems, in Agent-Oriented Software Engineering IV, P. Giorgini, J. P. M¨uller,

J. Odell, Eds. (Springer Berlin Heidelberg), vol. 2935 of Lecture Notes in Computer Science, <sub>pp</sub>. 214–230 (2004), doi:htt<sub>p</sub>s://doi.or<sub>g</sub>/10.1007/978-3-540-24620-6 15.

23. Q. Wu, et al., AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (2023), https://arxiv.org/abs/2308.08155.

24. S. A. McChrystal, T. Collins, D. Silverman, C. Fussell, P. Michael, Team of teams: New rules of engagement for a complex world (Portfolio/Penguin New York) (2015).

25. A. Dorri, S. S. Kanhere, R. Jurdak, Multi-Agent Systems: A Survey. IEEE Access 6, 28573– 28593 (2018), doi:10.1109/ACCESS.2018.2831228.

26. T. Guo, et al., Large Language Model based Multi-Agents: A Survey of Progress and Challenges (2024), https://arxiv.org/abs/2402.01680.

27. C. Zhang, et al., ProAgent: Building Proactive Cooperative Agents with Large Language Models (2024), https://arxiv.org/abs/2308.11339.

28<sub>.</sub> L<sub>.</sub> L<sub>ee,</sub> H<sub>.</sub> N<sub>wana,</sub> D<sub>.</sub> Nd<sub>umu,</sub> P<sub>.</sub> D<sub>e</sub> Wild<sub>e,</sub> Th<sub>e</sub> St<sub>a</sub>bilit<sub>y,</sub> S<sub>ca</sub>l<sub>a</sub>bilit<sub>y an</sub>d P<sub>er</sub>f<sub>ormance o</sub>f M<sub>u</sub>lti<sub>-</sub> agent Systems. BT Technology Journal 16, 94–103 (1998), doi:10.1023/A:1009686016775.

29. O. Shehor<sub>y</sub>, Architectural Pro<sub>p</sub>erties of Multi-A<sub>g</sub>ent S<sub>y</sub>stems (1999).

30. Y. Yang, et al., AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-A<sub>g</sub>ent S<sub>y</sub>stems (2025), https://arxiv.org/abs/2504.00587.

31. Y. Mao, A. Mirhoseini, Decentralized Multi-A<sub>g</sub>ent S<sub>y</sub>stems with Shared Context (2026), https://arxiv.org/abs/2606.10662.

32. L. Sander, et al., Scaling LLM-Driven Multi-Agent Systems: Design Principles and Architectural Scalabilit<sub>y</sub> Anal<sub>y</sub>sis (2026), https://arxiv.org/abs/2607.27942.

33<sub>.</sub> J<sub>.</sub> H<sub>yu</sub>n<sub>,</sub> N<sub>.</sub> R<sub>.</sub> W<sub>ay</sub>t<sub>ow</sub>i<sub>c</sub>h<sub>,</sub> B<sub>.</sub> Ch<sub>e</sub>n<sub>,</sub> CREW<sub>-</sub>Wildfir<sub>e:</sub> B<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rkin<sub>g</sub> A<sub>ge</sub>nti<sub>c</sub> M<sub>u</sub>lti<sub>-</sub>A<sub>ge</sub>nt Collaborations at Scale. Transactions on Machine Learning Research .

34. L. Zhang, Z. Ji, B. Chen, CREW: Facilitating Human-AI Teaming Research. Transactions on Machine Learning Research .

35. A. Sin<sub>g</sub>h, et al., O<sub>p</sub>enAI GPT-5 S<sub>y</sub>stem Card (2026), https://arxiv.org/abs/2601. 03267.

36. G. Team, et al., Gemma 4 Technical Re<sub>p</sub>ort (2026), https://arxiv.org/abs/2607.02770.

37<sub>.</sub> R<sub>.</sub> b<sub>y</sub> <sub>ar</sub>Xi<sub>v,</sub> Th<sub>e</sub> Ll<sub>ama</sub> 4 H<sub>er</sub>d<sub>:</sub> A<sub>rc</sub>hit<sub>ec</sub>t<sub>ure,</sub> T<sub>ra</sub>i<sub>n</sub>i<sub>ng,</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on,</sub> <sub>an</sub>d D<sub>ep</sub>l<sub>oymen</sub>t N<sub>o</sub>t<sub>es</sub> (2026), https://arxiv.org/abs/2601.11659.

38. Baidu-ERNIE-Team, ERNIE 4.5 Technical Re<sub>p</sub>ort (2025).

39. A. Yan<sub>g</sub>, et al., Qwen3 Technical Re<sub>p</sub>ort (2025), https://arxiv.org/abs/2505.09388.

40. A. Blakeman, et al., Nvidia nemotron 3: Eficient and open intelligence. arXiv preprint arXiv:2512.20856 (2025).

41. GLM-5-Team, et al., GLM-5: from Vibe Codin<sub>g</sub> to A<sub>g</sub>entic En<sub>g</sub>ineerin<sub>g</sub> (2026), https: //arxiv.org/abs/2602.15763.

42. DeepSeek-AI, et al., DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelli-<sub>g</sub>ence (2026), https://arxiv.org/abs/2606.19348.

43<sub>.</sub> A<sub>.</sub> P<sub>.</sub> B<sub>ra</sub>dl<sub>ey,</sub> Th<sub>e use o</sub>f th<sub>e area un</sub>d<sub>er</sub> th<sub>e</sub> ROC <sub>curve</sub> i<sub>n</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on o</sub>f <sub>ma-</sub> chine learning algorithms. Pattern Recognition 30 (7), 1145–1159 (1997), doi:https:// doi.or<sub>g</sub>/10.1016/S0031-3203(96)00142-2, https://www.sciencedirect.com/science/ article/pii/S0031320396001422.

44<sub>.</sub> R<sub>.</sub> A<sub>.</sub> Fi<sub>s</sub>h<sub>er,</sub> St<sub>u</sub>di<sub>es</sub> i<sub>n</sub> <sub>crop</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on.</sub> I<sub>.</sub> A<sub>n</sub> <sub>exam</sub>i<sub>na</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>y</sub>i<sub>e</sub>ld <sub>o</sub>f d<sub>resse</sub>d <sub>gra</sub>i<sub>n</sub> from Broadbalk. The Journal of Agricultural Science 11, 107 – 135 (1921), https: //api.semanticscholar.org/CorpusID:86029217.

45. M. Chen, et al., Evaluating Large Language Models Trained on Code (2021).

46. T. Y. Zhuo, et al., BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions. arXiv preprint arXiv:2406.15877 (2024).

47. M. Tian, et al., Scicode: A research coding benchmark curated by scientists. Advances in Neural Information Processing Systems 37, 30624–30650 (2024).

48. F. Shi, et al., Lan<sub>g</sub>ua<sub>g</sub>e Models are Multilin<sub>g</sub>ual Chain-of-Thou<sub>g</sub>ht Reasoners (2022), https: //arxiv.org/abs/2210.03057.

49. J. Dekoninck, et al., Beyond Benchmarks: MathArena as an Evaluation Platform for Mathe matics with LLMs (2026), https://arxiv.org/abs/2605.00674.

50<sub>.</sub> S<sub>.</sub> Y<sub>u,</sub> L<sub>.</sub> L<sub>.</sub> Gr<sub>ee</sub>r<sub>,</sub> N<sub>.</sub> H<sub>a</sub>l<sub>evy,</sub> L<sub>.</sub> V<sub>a</sub>n B<sub>u</sub>nd<sub>e</sub>r<sub>e</sub>n<sub>,</sub> On l<sub>a</sub>dd<sub>e</sub>r<sub>s</sub> <sub>a</sub>nd <sub>py</sub>r<sub>a</sub>mid<sub>s:</sub> Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub>’<sub>s</sub> shape determines relationships and performance in groups. Personality and Social Psychology Bulletin 45 (12), 1717–1733 (2019).

51. O. Mascaro, et al., Human and animal dominance hierarchies show a pyramidal structure guiding adult and infant social inferences. Nature Human Behaviour 7 (8), 1294–1306 (2023).

52<sub>.</sub> L<sub>.</sub> Zh<sub>a</sub>n<sub>g,</sub> Z<sub>.</sub> Ji<sub>,</sub> N<sub>.</sub> R<sub>.</sub> W<sub>ay</sub>t<sub>ow</sub>i<sub>c</sub>h<sub>,</sub> B<sub>.</sub> Ch<sub>e</sub>n<sub>,</sub> GUIDE<sub>:</sub> R<sub>ea</sub>l<sub>-</sub>Tim<sub>e</sub> H<sub>u</sub>m<sub>a</sub>n<sub>-</sub>Sh<sub>ape</sub>d Agents, in Advances in Neural Information Processing Systems, A. Globerson, et al., Eds. (Curran Associates, Inc.), vol. 37 (2024), <sub>pp</sub>. 138959–138980, doi:10.52202/ 079017-4409, https://proceedings.neurips.cc/paper\_files/paper/2024/file/ facf3192e99ce60c0ef5ed4067b72f68-Paper-Conference.pdf.

53<sub>.</sub> Z<sub>.</sub> Ji<sub>,</sub> B<sub>.</sub> Ch<sub>e</sub>n<sub>,</sub> Pr<sub>e</sub>f<sub>-</sub>GUIDE<sub>:</sub> C<sub>o</sub>ntin<sub>ua</sub>l P<sub>o</sub>li<sub>cy</sub> L<sub>ea</sub>rnin<sub>g</sub> fr<sub>o</sub>m R<sub>ea</sub>l<sub>-</sub>Tim<sub>e</sub> H<sub>u</sub>m<sub>a</sub>n F<sub>ee</sub>db<sub>ac</sub>k <sub>v</sub>i<sub>a</sub> Preference-Based Learning. Transactions on Machine Learning Research .

54. W. Kwon, et al., Eficient Memory Management for Large Language Model Serving with Pa<sub>g</sub>edAttention (2023), https://arxiv.org/abs/2309.06180.

## A<sub>c</sub>kn<sub>ow</sub>l<sub>e</sub>d<sub>g</sub>m<sub>e</sub>nt<sub>s</sub>

W<sub>e</sub> th<sub>an</sub>k <sub>mem</sub>b<sub>ers</sub> <sub>o</sub>f th<sub>e</sub> G<sub>enera</sub>l R<sub>o</sub>b<sub>o</sub>ti<sub>cs</sub> L<sub>a</sub>b <sub>a</sub>t D<sub>u</sub>k<sub>e</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub> f<sub>or</sub> h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l di<sub>scuss</sub>i<sub>ons</sub> <sub>an</sub>d <sup>s</sup>ugg<sup>estions</sup>.

F<sub>u</sub>ndin<sub>g:</sub> Thi<sub>s</sub> <sub>wor</sub>k i<sub>s</sub> <sub>suppor</sub>t<sub>e</sub>d b<sub>y</sub> ARL STRONG <sub>program</sub> <sub>un</sub>d<sub>er</sub> <sub>awar</sub>d<sub>s</sub> W911NF2320182<sub>,</sub> W911NF2220113<sub>,</sub> <sub>an</sub>d W911NF242021<sub>,</sub> DARPA TIAMAT <sub>program</sub> <sub>un</sub>d<sub>er</sub> <sub>awar</sub>d HR00112490419<sub>,</sub> <sub>an</sub>d ARO <sub>un</sub>d<sub>er</sub> <sub>awar</sub>d W911NF2410405<sub>.</sub>

A<sub>u</sub>thor contrib<sub>u</sub>tions<sub>:</sub> B<sub>.</sub>C<sub>.,</sub> Z<sub>.</sub>J<sub>.,</sub> <sub>a</sub>nd J<sub>.</sub>H<sub>.</sub> <sub>co</sub>n<sub>ce</sub>i<sub>ve</sub>d <sub>a</sub>nd d<sub>es</sub>i<sub>g</sub>n<sub>e</sub>d th<sub>e</sub> r<sub>esea</sub>r<sub>c</sub>h<sub>.</sub> Z<sub>.</sub>J<sub>.</sub> <sub>a</sub>nd J<sub>.</sub>H<sub>.</sub> d<sub>es</sub>i<sub>gne</sub>d <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d <sub>per</sub>f<sub>orme</sub>d <sub>exper</sub>i<sub>men</sub>t<sub>s.</sub> Z<sub>.</sub>J<sub>.,</sub> J<sub>.</sub>H<sub>.,</sub> <sub>an</sub>d B<sub>.</sub>C<sub>.</sub> <sub>ana</sub>l<sub>yze</sub>d d<sub>a</sub>t<sub>a.</sub> Z<sub>.</sub>J <sub>an</sub>d B<sub>.</sub>C<sub>.</sub> <sub>wro</sub>t<sub>e</sub> th<sub>e manuscr</sub>i<sub>p</sub>t<sub>.</sub> All <sub>au</sub>th<sub>ors prov</sub>id<sub>e</sub>d f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub>

C<sub>o</sub>m<sub>pe</sub>tin<sub>g</sub> int<sub>e</sub>r<sub>es</sub>t<sub>s:</sub> Th<sub>e au</sub>th<sub>ors</sub> d<sub>ec</sub>l<sub>are no compe</sub>ti<sub>ng</sub> i<sub>n</sub>t<sub>eres</sub>t<sub>s.</sub>

Data, code and materials availability: The com<sub>p</sub>lete code and data is available at: https: //github.com/generalroboticslab/ORCH.

S<sub>upp</sub>lim<sub>e</sub>nt<sub>a</sub>r<sub>y</sub> M<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l<sub>s:</sub>

M<sub>e</sub>th<sub>o</sub>d

Fi<sub>gures</sub> S1 t<sub>o</sub> S24

T<sub>a</sub>bl<sub>es</sub> S1 t<sub>o</sub> S3

<sup>S</sup>u<sub>pp</sub><sup>l</sup>ementar<sub>y</sub> <sup>T</sup>ext

S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> M<sub>ov</sub>i<sub>e</sub> S1

# S<sub>upp</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>r<sub>y</sub> M<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l<sub>s</sub> f<sub>o</sub>r Or<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>p</sub>rin<sub>c</sub>i<sub>p</sub>l<sub>es</sub> <sub>e</sub>n<sub>a</sub>bl<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub> int<sub>e</sub>lli<sub>ge</sub>n<sub>ce</sub> in <sub>e</sub>mb<sub>o</sub>di<sub>e</sub>d AI

Zh<sub>e</sub>n<sub>g</sub>r<sub>a</sub>n Ji<sub>,</sub> J<sub>o</sub>n<sub>a</sub>th<sub>a</sub>n H<sub>yu</sub>n<sub>,</sub> B<sub>oyua</sub>n Ch<sub>e</sub>n <sup>∗</sup>C<sub>orrespon</sub>di<sub>ng</sub> <sub>au</sub>th<sub>or.</sub> E<sub>ma</sub>il<sub>:</sub> b<sub>oyuan.c</sub>h<sub>en</sub>@d<sub>u</sub>k<sub>e.e</sub>d<sub>u</sub>

Thi<sub>s</sub> PDF fil<sub>e</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>es:</sub>

M<sub>e</sub>th<sub>o</sub>d

Fi<sub>gures</sub> S1 t<sub>o</sub> S24

T<sub>a</sub>bl<sub>es</sub> S1 t<sub>o</sub> S3

<sup>S</sup>u<sub>pp</sub><sup>l</sup>ementar<sub>y</sub> <sup>T</sup>ext

## M<sub>e</sub>th<sub>o</sub>d

## Ad<sub>ap</sub>ti<sub>ve</sub> <sub>o</sub>r<sub>ga</sub>ni<sub>za</sub>ti<sub>o</sub>n<sub>a</sub>l hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> d<sub>es</sub>i<sub>g</sub>n

P<sub>rev</sub>i<sub>ous</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d LLM<sub>-</sub>b<sub>ase</sub>d <sub>mu</sub>lti<sub>-agen</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>re</sub>li<sub>e</sub>d <sub>on</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>na</sub>i<sub>ve</sub> t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es,</sub> typically following an orchestral style in which a single manager supervises all worker agents (16– 19). In ORCH, we propose adaptive organizational hierarchies that are tailored to the task dificulty <sub>an</sub>d th<sub>e capa</sub>biliti<sub>es o</sub>f th<sub>e ava</sub>il<sub>a</sub>bl<sub>e wor</sub>k<sub>er agen</sub>t<sub>s.</sub> Gi<sub>ven a</sub> t<sub>as</sub>k d<sub>escr</sub>i<sub>p</sub>ti<sub>on an</sub>d th<sub>e ava</sub>il<sub>a</sub>bl<sub>e wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s,</sub> th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y</sub> <sub>can</sub> <sub>e</sub>ith<sub>er</sub> b<sub>e</sub> d<sub>es</sub>i<sub>gne</sub>d b<sub>y</sub> <sub>a</sub> h<sub>uman</sub> <sub>exper</sub>t <sub>or</sub> <sub>genera</sub>t<sub>e</sub>d <sub>au</sub>t<sub>oma</sub>ti<sub>ca</sub>ll<sub>y</sub> b<sub>y</sub> LLM<sub>s</sub> <sub>w</sub>ith <sub>a</sub> <sub>cr</sub>iti<sub>c,</sub> <sub>as</sub> <sub>wr</sub>itt<sub>en</sub> i<sub>n</sub> E<sub>qua</sub>ti<sub>on</sub> S1<sub>.</sub> Th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y</sub> i<sub>s</sub> <sub>represen</sub>t<sub>e</sub>d b<sub>y</sub> <sub>a</sub> <sub>roo</sub>t<sub>e</sub>d t<sub>ree</sub> <sub>w</sub>ith th<sub>e num</sub>b<sub>er o</sub>f l<sub>ea</sub>f <sub>no</sub>d<sub>es equa</sub>l t<sub>o</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>wor</sub>k<sub>er agen</sub>t<sub>s prov</sub>id<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> t<sub>as</sub>k<sub>.</sub>

$$
G = { \mathcal { H } } ( T , W , C )\tag{S1}
$$

<sub>w</sub>h<sub>ere</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e genera</sub>ti<sub>on</sub> f<sub>unc</sub>ti<sub>on,</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> t<sub>as</sub>k d<sub>escr</sub>i<sub>p</sub>ti<sub>on,</sub> � i<sub>s</sub> th<sub>e se</sub>t <sub>o</sub>f <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s,</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub>i<sub>r</sub> <sub>capa</sub>biliti<sub>es,</sub> <sub>an</sub>d � i<sub>s</sub> th<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d <sub>roo</sub>t<sub>e</sub>d t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y.</sub> Th<sub>e</sub> <sub>promp</sub>t f<sub>or a</sub> h<sub>uman exper</sub>t <sub>or</sub> LLM t<sub>o genera</sub>t<sub>e</sub> th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> th<sub>e supp</sub>l<sub>emen</sub>t<sub>ary</sub> t<sub>ex</sub>t <sub>sec</sub>ti<sub>on</sub> S2<sub>.</sub> F<sub>or</sub> t<sub>as</sub>k<sub>s</sub> i<sub>nvo</sub>l<sub>v</sub>i<sub>ng</sub> <sub>a</sub> l<sub>arge</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>agen</sub>t<sub>s,</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>ona</sub>l hi<sub>erarc</sub>hi<sub>es</sub> <sub>ena</sub>bl<sub>e</sub> <sub>more sop</sub>hi<sub>s</sub>ti<sub>ca</sub>t<sub>e</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ures, suc</sub>h <sub>as</sub> hi<sub>erarc</sub>hi<sub>ca</sub>l t<sub>eams-o</sub>f<sub>-</sub>t<sub>eams,</sub> t<sub>o</sub> b<sub>e</sub>tt<sub>er organ</sub>i<sub>ze an</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> th<sub>e</sub> <sub>wor</sub>kf<sub>orce.</sub> M<sub>eanw</sub>hil<sub>e,</sub> th<sub>ey</sub> <sub>a</sub>l<sub>so</sub> <sub>a</sub>ll<sub>ow</sub> th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> t<sub>o</sub> <sub>rema</sub>i<sub>n</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> <sub>w</sub>h<sub>en</sub> <sub>on</sub>l<sub>y</sub> <sub>a</sub> f<sub>ew</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s</sub> <sub>are</sub> <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> <sub>a</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> t<sub>as</sub>k<sub>.</sub> C<sub>ompare</sub>d t<sub>o</sub> th<sub>e</sub> t<sub>ra</sub>diti<sub>ona</sub>l <sub>orc</sub>h<sub>es</sub>t<sub>ra</sub>l <sub>s</sub>t<sub>y</sub>l<sub>e,</sub> <sub>our</sub> <sub>approac</sub>h <sub>prov</sub>id<sub>es</sub> <sub>grea</sub>t<sub>er</sub> fl<sub>ex</sub>ibilit<sub>y</sub> <sub>an</sub>d i<sub>mproves</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>execu</sub>ti<sub>on.</sub>

## H<sub>y</sub>brid interde<sub>p</sub>endence bet<sub>w</sub>een <sub>w</sub>orker a<sub>g</sub>ents

Pooled interdependency (horizontal manager). Inspired b<sub>y</sub> organization theor<sub>y</sub> (20), we propose t<sub>wo</sub> t<sub>ypes</sub> <sub>o</sub>f <sub>managers</sub> t<sub>o</sub> <sub>organ</sub>i<sub>ze</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y:</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>man-</sub> <sub>agers an</sub>d <sub>ver</sub>ti<sub>ca</sub>l <sub>managers.</sub> I<sub>n organ</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>eory, co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on para</sub>di<sub>gms can</sub> b<sub>e c</sub>l<sub>ass</sub>ifi<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> <sub>poo</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ency</sub> <sub>an</sub>d <sub>sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ency.</sub> P<sub>oo</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ency</sub> i<sub>s</sub> <sub>a</sub> <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>para</sub>di<sub>gm</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h <sub>a</sub> <sub>group</sub> <sub>o</sub>f <sub>agen</sub>t<sub>s</sub> <sub>wor</sub>k<sub>s</sub> <sub>concurren</sub>tl<sub>y</sub> <sub>on</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>su</sub>bt<sub>as</sub>k<sub>s,</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>com-</sub> bined out<sub>p</sub>uts constitute the team’s outcome. For exam<sub>p</sub>le, in the Cut Trees Sparse Small task, <sub>a</sub>ll <sub>agen</sub>t<sub>s</sub> <sub>are</sub> <sub>requ</sub>i<sub>re</sub>d t<sub>o</sub> <sub>cu</sub>t d<sub>es</sub>i<sub>gna</sub>t<sub>e</sub>d t<sub>rees</sub> <sub>a</sub>t dif<sub>eren</sub>t l<sub>oca</sub>ti<sub>ons.</sub> Th<sub>e</sub> <sub>wor</sub>k<sub>er</sub> <sub>agen</sub>t<sub>s</sub> <sub>can</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>an</sub>d <sub>concurren</sub>tl<sub>y</sub> <sub>cu</sub>t th<sub>e</sub> t<sub>rees</sub> <sub>ass</sub>i<sub>gne</sub>d b<sub>y</sub> th<sub>e</sub>i<sub>r</sub> <sub>manager.</sub> I<sub>n</sub> <sub>prac</sub>ti<sub>ce,</sub> <sub>we</sub> d<sub>es</sub>i<sub>gne</sub>d the horizontal mana<sub>g</sub>er to assi<sub>g</sub>n concurrent tasks to its children(can be either worker a<sub>g</sub>ents or mana<sub>g</sub>ers) b<sub>y</sub> considerin<sub>g</sub> the current <sub>g</sub>iven mission, estimated mission <sub>p</sub>ro<sub>g</sub>ress, and re<sub>p</sub>orts from it<sub>s c</sub>hild<sub>ren,</sub> d<sub>eno</sub>t<sub>e</sub>d i<sub>n</sub> E<sub>qua</sub>ti<sub>on</sub> S2<sub>.</sub>

$$
{ \mathcal { T } } _ { h } = \{ ( c _ { i } , t _ { i } ) \mid c _ { i } \in C \} = f _ { h } ( M , P , R ) ,\tag{S2}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { T } _ { h }$ is the set of task assi<sub>g</sub>nments <sub>g</sub>enerated b<sub>y</sub> the horizontal mana<sub>g</sub>er, C denotes the <sub>se</sub>t <sub>o</sub>f <sub>c</sub>hild <sub>agen</sub>t<sub>s</sub> <sub>superv</sub>i<sub>se</sub>d b<sub>y</sub> th<sub>e</sub> <sub>manager,</sub> $c _ { i }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> �<sub>-</sub>th <sub>c</sub>hild <sub>agen</sub>t<sub>,</sub> $t _ { i }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> t<sub>as</sub>k <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> <sub>c</sub>hild <sub>agen</sub>t $c _ { i } ,$ � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>curren</sub>t <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> th<sub>e</sub> <sub>manager,</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>progress,</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>repor</sub>t<sub>s</sub> <sub>rece</sub>i<sub>ve</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>c</sub>hild <sub>agen</sub>t<sub>s,</sub> <sub>an</sub>d $f _ { h } ( \cdot )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub>’<sub>s</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> f<sub>unc</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>promp</sub>t <sub>we</sub> <sub>use</sub>d f<sub>or</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub> <sub>can</sub> b<sub>e</sub> f<sub>ou</sub>nd in th<sub>e</sub> S<sub>upp</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>r<sub>y</sub> T<sub>ex</sub>t Pr<sub>o</sub>m<sub>p</sub>t S5<sub>.</sub>

Sequential interdependenc<sub>y</sub> (vertical mana<sub>g</sub>er). In contrast, se<sub>q</sub>uential interde<sub>p</sub>endenc<sub>y</sub> refers t<sub>o</sub> <sub>a</sub> <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>para</sub>di<sub>gm</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h <sub>agen</sub>t<sub>s</sub> <sub>mus</sub>t <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> t<sub>as</sub>k<sub>s</sub> i<sub>n</sub> <sub>a</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>or</sub>d<sub>er,</sub> <sub>suc</sub>h th<sub>a</sub>t th<sub>e</sub> <sub>ou</sub>t<sub>pu</sub>t <sub>o</sub>f <sub>one</sub> <sub>s</sub>t<sub>age</sub> b<sub>ecomes</sub> th<sub>e</sub> <sub>prerequ</sub>i<sub>s</sub>it<sub>e</sub> f<sub>or</sub> th<sub>e</sub> <sub>nex</sub>t <sub>one.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> <sub>agen</sub>t<sub>s</sub> <sub>canno</sub>t <sub>execu</sub>t<sub>e a</sub>ll t<sub>as</sub>k<sub>s concurren</sub>tl<sub>y an</sub>d <sub>mus</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r ac</sub>ti<sub>v</sub>iti<sub>es accor</sub>di<sub>ng</sub> t<sub>o</sub> t<sub>as</sub>k d<sub>epen</sub>d<sub>enc</sub>i<sub>es.</sub> For exam<sub>p</sub>le, in the Scale Level Complex task, drone a<sub>g</sub>ents must first scout the environment to l<sub>oca</sub>t<sub>e</sub> th<sub>e</sub> fi<sub>re an</sub>d id<sub>en</sub>tif<sub>y sa</sub>f<sub>e access rou</sub>t<sub>es.</sub> O<sub>n</sub>l<sub>y a</sub>ft<sub>er</sub> thi<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>ecomes ava</sub>il<sub>a</sub>bl<sub>e can</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>er</sub> <sub>an</sub>d b<sub>u</sub>lld<sub>ozer</sub> <sub>agen</sub>t<sub>s</sub> <sub>nav</sub>i<sub>ga</sub>t<sub>e</sub> t<sub>o</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>area</sub> t<sub>o</sub> <sub>cons</sub>t<sub>ruc</sub>t fi<sub>re</sub>b<sub>rea</sub>k<sub>s</sub> <sub>an</sub>d <sub>suppress</sub> th<sub>e</sub> fi<sub>re.</sub> T<sub>o suppor</sub>t thi<sub>s</sub> f<sub>orm o</sub>f <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on, we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d th<sub>e ver</sub>ti<sub>ca</sub>l <sub>manager, w</sub>hi<sub>c</sub>h <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s sequen</sub>ti<sub>a</sub>l i<sub>n</sub>t<sub>er</sub>d<sub>epen</sub>d<sub>ency.</sub> R<sub>a</sub>th<sub>er</sub> th<sub>an</sub> di<sub>rec</sub>tl<sub>y ass</sub>i<sub>gn</sub>i<sub>ng</sub> t<sub>as</sub>k<sub>s</sub> t<sub>o</sub> it<sub>s c</sub>hild<sub>ren,</sub> th<sub>e ver</sub>ti<sub>ca</sub>l <sub>manager</sub> fi<sub>rs</sub>t d<sub>ecomposes</sub> th<sub>e</sub> <sub>overa</sub>ll <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> <sub>a</sub> <sub>sequence</sub> <sub>o</sub>f <sub>p</sub>h<sub>ases,</sub> <sub>w</sub>ith <sub>eac</sub>h <sub>p</sub>h<sub>ase</sub> <sub>represen</sub>t<sub>-</sub> i<sub>ng</sub> <sub>a</sub> <sub>m</sub>il<sub>es</sub>t<sub>one</sub> <sub>requ</sub>i<sub>re</sub>d t<sub>o</sub> <sub>accomp</sub>li<sub>s</sub>h th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on.</sub> Th<sub>e</sub> <sub>manager</sub> <sub>ac</sub>ti<sub>va</sub>t<sub>es</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d <sub>genera</sub>t<sub>es</sub> t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> f<sub>or</sub> it<sub>s</sub> <sub>c</sub>hild<sub>ren</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>a</sub>t <sub>p</sub>h<sub>ase.</sub> O<sub>nce</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> h<sub>as</sub> b<sub>een</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d<sub>,</sub> th<sub>e</sub> <sub>manager</sub> <sub>a</sub>d<sub>vances</sub> t<sub>o</sub> th<sub>e</sub> <sub>nex</sub>t <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d i<sub>ssues</sub> <sub>a</sub> <sub>new</sub> <sub>se</sub>t <sub>o</sub>f t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s.</sub> Thi<sub>s</sub> <sub>p</sub>h<sub>ase</sub>d <sub>execu</sub>ti<sub>on ena</sub>bl<sub>es</sub> th<sub>e</sub> t<sub>eam</sub> t<sub>o coor</sub>di<sub>na</sub>t<sub>e</sub> l<sub>ong-</sub>h<sub>or</sub>i<sub>zon</sub> t<sub>as</sub>k<sub>s w</sub>hil<sub>e respec</sub>ti<sub>ng</sub> d<sub>epen</sub>d<sub>enc</sub>i<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> dif<sub>eren</sub>t <sub>s</sub>t<sub>ages</sub> <sub>o</sub>f th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on,</sub> <sub>as</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> E<sub>qua</sub>ti<sub>on</sub> S3<sub>.</sub>

$$
\begin{array} { c } { \Phi = \bigl \{ \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { K } \bigr \} = f _ { \nu } ( M , P _ { m } , R ) , } \\ { \mathcal { T } _ { \nu } ^ { ( k ) } = \bigl \{ ( c _ { i } , t _ { i } ^ { ( k ) } ) \mid c _ { i } \in C \bigr \} = g _ { \nu } ( \phi _ { k } , P _ { p } , R ) } \end{array}\tag{S3}
$$

where Φ denotes the ordered se<sub>q</sub>uence of mission <sub>p</sub>hases <sub>g</sub>enerated b<sub>y</sub> the vertical mana<sub>g</sub>er, $\phi _ { k }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> �<sub>-</sub>th <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>p</sub>h<sub>ase,</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>p</sub>h<sub>ases,</sub> $\mathcal { T } _ { \nu } ^ { ( k ) }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>genera</sub>t<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> $\phi _ { k }$ , C denotes the set of child a<sub>g</sub>ents su<sub>p</sub>ervised b<sub>y</sub> th<sub>e ver</sub>ti<sub>ca</sub>l <sub>manager,</sub> $c _ { i }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> �<sub>-</sub>th <sub>c</sub>hild <sub>agen</sub>t<sub>,</sub> $t _ { i } ^ { ( k ) }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> t<sub>as</sub>k <sub>ass</sub>i<sub>gne</sub>d t<sub>o c</sub>hild <sub>agen</sub>t $c _ { i }$ d<sub>ur</sub>i<sub>ng</sub> <sub>p</sub>h<sub>ase</sub> $\phi _ { k }$ <sub>,</sub> � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>curren</sub>t <sub>m</sub>i<sub>ss</sub>i<sub>on,</sub> $P _ { m }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>progress</sub> <sub>an</sub>d $P _ { p }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>p</sub>h<sub>ase</sub> <sub>progress,</sub> <sub>an</sub>d � d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>repor</sub>t<sub>s</sub> <sub>rece</sub>i<sub>ve</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>c</sub>hild <sub>agen</sub>t<sub>s.</sub> Th<sub>e promp</sub>t <sub>we use</sub>d f<sub>or</sub> th<sub>e ver</sub>ti<sub>ca</sub>l <sub>manager can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> T<sub>ex</sub>t P<sub>romp</sub>t S5<sub>.</sub>

## C<sub>o</sub>mm<sub>u</sub>ni<sub>ca</sub>ti<sub>o</sub>n <sub>pa</sub>r<sub>a</sub>di<sub>g</sub>m

I<sub>n</sub> ORCH<sub>, commun</sub>i<sub>ca</sub>ti<sub>on</sub> f<sub>o</sub>ll<sub>ows a</sub> bidi<sub>rec</sub>ti<sub>ona</sub>l hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>pro</sub>t<sub>oco</sub>l th<sub>a</sub>t <sub>a</sub>lt<sub>erna</sub>t<sub>es</sub> b<sub>e</sub>t<sub>ween a</sub> b<sub>o</sub>tt<sub>om-up</sub> <sub>s</sub>t<sub>a</sub>t<sub>us</sub> <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d <sub>a</sub> t<sub>op-</sub>d<sub>own</sub> <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>p</sub>h<sub>ase</sub> <sub>a</sub>t <sub>every</sub> <sub>env</sub>i<sub>ronmen</sub>t ti<sub>mes</sub>t<sub>ep.</sub>

Bottom-<sub>up</sub> stat<sub>u</sub>s <sub>p</sub>hase<sub>.</sub> D<sub>u</sub>rin<sub>g</sub> th<sub>e</sub> b<sub>o</sub>tt<sub>o</sub>m<sub>-up</sub> <sub>s</sub>t<sub>a</sub>t<sub>us</sub> <sub>p</sub>h<sub>ase,</sub> <sub>eac</sub>h <sub>wo</sub>rk<sub>e</sub>r <sub>age</sub>nt fir<sub>s</sub>t int<sub>e</sub>r<sub>p</sub>r<sub>e</sub>t<sub>s</sub> it<sub>s</sub> l<sub>oca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>an</sub>d <sub>ass</sub>i<sub>gne</sub>d t<sub>as</sub>k t<sub>o</sub> <sub>genera</sub>t<sub>e</sub> <sub>a</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>s</sub>t<sub>a</sub>t<sub>us</sub> <sub>repor</sub>t <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> it<sub>s</sub> <sub>curren</sub>t <sub>progress,</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d t<sub>as</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>percen</sub>t<sub>age,</sub> <sub>an</sub>d <sub>any</sub> <sub>urgen</sub>t i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on.</sub> Th<sub>ese</sub> r<sub>epo</sub>rt<sub>s</sub> <sub>a</sub>r<sub>e</sub> <sub>p</sub>r<sub>opaga</sub>t<sub>e</sub>d <sub>upwa</sub>rd thr<sub>oug</sub>h th<sub>e</sub> t<sub>ea</sub>m hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y,</sub> <sub>w</sub>h<sub>e</sub>r<sub>e</sub> <sub>eac</sub>h m<sub>a</sub>n<sub>age</sub>r <sub>agg</sub>r<sub>ega</sub>t<sub>es</sub> th<sub>e repor</sub>t<sub>s rece</sub>i<sub>ve</sub>d f<sub>rom</sub> it<sub>s c</sub>hild<sub>ren</sub> t<sub>o es</sub>ti<sub>ma</sub>t<sub>e</sub> th<sub>e overa</sub>ll <sub>m</sub>i<sub>ss</sub>i<sub>on an</sub>d <sub>p</sub>h<sub>ase progress.</sub> B<sub>ase</sub>d <sub>on</sub> thi<sub>s</sub> <sub>aggrega</sub>t<sub>e</sub>d i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> <sub>managers</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>l<sub>an</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>con</sub>ti<sub>nue</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>,</sub> t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>requ</sub>i<sub>re</sub> <sub>rev</sub>i<sub>s</sub>i<sub>on,</sub> <sub>or</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>s</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>e</sub>d<sub>.</sub>

T<sub>op-</sub>d<sub>ow</sub>n <sub>p</sub>l<sub>a</sub>nnin<sub>g</sub> <sub>p</sub>h<sub>ase.</sub> D<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>op-</sub>d<sub>own</sub> <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>p</sub>h<sub>ase,</sub> <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> fl<sub>ow</sub> f<sub>rom</sub> th<sub>e roo</sub>t <sub>manager</sub> t<sub>owar</sub>d th<sub>e</sub> l<sub>ea</sub>f <sub>no</sub>d<sub>es.</sub> H<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers</sub> di<sub>rec</sub>tl<sub>y genera</sub>t<sub>e concurren</sub>t t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e</sub>i<sub>r</sub> <sub>c</sub>hild<sub>ren,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>managers</sub> fi<sub>rs</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> d<sub>ecompos</sub>iti<sub>on</sub> b<sub>y</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>ng</sub> <sub>or</sub> <sub>mo</sub>dif<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>ve</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> <sub>p</sub>h<sub>ase</sub> b<sub>e</sub>f<sub>ore</sub> <sub>genera</sub>ti<sub>ng</sub> t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> f<sub>or</sub> th<sub>a</sub>t <sub>p</sub>h<sub>ase.</sub> B<sub>e</sub>f<sub>ore</sub> <sub>a</sub> <sub>new</sub> <sub>p</sub>l<sub>an</sub> i<sub>s</sub> fi<sub>na</sub>li<sub>ze</sub>d<sub>,</sub> <sub>managers</sub> <sub>co</sub>ll<sub>ec</sub>t f<sub>ee</sub>db<sub>ac</sub>k f<sub>rom</sub> th<sub>e</sub>i<sub>r</sub> <sub>c</sub>hild <sub>agen</sub>t<sub>s</sub> t<sub>o</sub> <sub>ver</sub>if<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> assigned tasks are feasible and consistent with each agent’s capabilities. If actionable objections <sub>are</sub> <sub>ra</sub>i<sub>se</sub>d<sub>,</sub> th<sub>e</sub> <sub>manager</sub> <sub>rev</sub>i<sub>ses</sub> th<sub>e</sub> <sub>p</sub>l<sub>an</sub> <sub>an</sub>d <sub>repea</sub>t<sub>s</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>process</sub> <sub>un</sub>til <sub>no</sub> f<sub>ur</sub>th<sub>er</sub> <sub>con</sub>fli<sub>c</sub>t<sub>s</sub> <sub>rema</sub>i<sub>n.</sub> O<sub>nce approve</sub>d<sub>,</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>ons, p</sub>h<sub>ases, an</sub>d t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s are propaga</sub>t<sub>e</sub>d d<sub>own</sub> th<sub>e</sub> hi<sub>erarc</sub>h<sub>y</sub> t<sub>o a</sub>ll <sub>c</sub>hild <sub>agen</sub>t<sub>s.</sub> Fi<sub>na</sub>ll<sub>y, eac</sub>h <sub>wor</sub>k<sub>er agen</sub>t <sub>conver</sub>t<sub>s</sub> it<sub>s ass</sub>i<sub>gne</sub>d t<sub>as</sub>k <sub>an</sub>d <sub>curren</sub>t <sub>percep</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>o</sub> <sub>execu</sub>t<sub>a</sub>bl<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t <sub>ac</sub>ti<sub>ons,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>are</sub> <sub>execu</sub>t<sub>e</sub>d <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> t<sub>o</sub> <sub>a</sub>d<sub>vance</sub> th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t t<sub>o</sub> th<sub>e</sub> <sub>nex</sub>t ti<sub>mes</sub>t<sub>ep.</sub>

![](images/ad4c6cdf64949ac8f0cf5ae89408ce4eb3c28ef65a1782c2ca291027c53d2a30.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S1<sub>:</sub> Ch<sub>a</sub>tGPT<sub>-</sub>5<sub>.</sub>4 P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/472f088db647e6f4b65d26a65ad8279e65078b84538ab7aead188fb6e75cecc8.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S2<sub>:</sub> D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>V4<sub>-</sub>Pr<sub>o</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/c2e5e24b60dce81122e09a46b155bdcd606af0bb78cbf15d7109e4ac98539f6b.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S3<sub>:</sub> Ll<sub>a</sub>m<sub>a-</sub>4<sub>-</sub>S<sub>cou</sub>t<sub>-</sub>17B<sub>-</sub>16E<sub>-</sub>In<sub>s</sub>tr<sub>uc</sub>t P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/c6a49aa121ab9941aaa755d01c518b76a301c39709440398592d19fb997231a6.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S4<sub>:</sub> G<sub>e</sub>mm<sub>a-</sub>4<sub>-</sub>it P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/5cae8bae6b6b7cbaf9844af6da6128c4b6e5808c11c5ab3650f2bf44f125d592.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S5<sub>:</sub> ERNIE<sub>-</sub>4<sub>.</sub>5<sub>-</sub>21B<sub>-</sub>A3B P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/820b892dd31ca25e39b9dab334707c15ee86b365c3a27779f39483837a831f20.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S6<sub>:</sub> GLM<sub>-</sub>5<sub>.</sub>1 P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/0d6a889d90762b9a85088ab375a4f585ca35a8ede5651e2483719fff8e560a2b.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S7<sub>:</sub> N<sub>e</sub>m<sub>o</sub>tr<sub>o</sub>n<sub>-</sub>3<sub>-</sub>S<sub>upe</sub>r<sub>-</sub>120B<sub>-</sub>A12B P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/410b7c948a7c998aa2a5343b460dd8372662b704521b6cf605e8d4588363bddd.jpg)  
Figure S8: Qwen-3.6-35B-A3B Performance Across 25 CREW-Wildfire tasks.

![](images/3f4ddabcccde53fbc33d54d135f78c322284f2f0d7069dccf3d93f9f49a62c6b.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S9<sub>:</sub> Ch<sub>a</sub>tGPT<sub>-</sub>5<sub>.</sub>4 G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> tasks.

![](images/18a2cf1cf7b4a7a8876aa968622f14cf5e7849126c087b5a42d80db639d0f587.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S10<sub>:</sub> D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>V4<sub>-</sub>Pr<sub>o</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub> Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/a36734228292e09bb3d8f0ca6d1751f979e62817622031727fc5037a61764ccc.jpg)  
Fi<sub>gure</sub> S11<sub>:</sub> Ll<sub>ama-</sub>4<sub>-</sub>S<sub>cou</sub>t<sub>-</sub>17B<sub>-</sub>16E<sub>-</sub>I<sub>ns</sub>t<sub>ruc</sub>t G<sub>enera</sub>t<sub>e</sub>d T<sub>eam</sub> Hi<sub>erarc</sub>h<sub>y</sub> P<sub>er</sub>f<sub>ormance</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/0d711d1c7534b3d98d49430f9c5636bb5dc964fe5ffd71f9e5da64e2eeb8b6fa.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S12<sub>:</sub> G<sub>e</sub>mm<sub>a-</sub>4<sub>-</sub>it G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> tasks.

Different Ways Generated Team Hierarchy Performance Across 25 Tasks (ERNIE-4.5-21B-A3B) Cut Trees Sparse Small (Easy) Cut Trees Lines Small (Easy) Scout Fire Small (Easy)  
![](images/c9cd2057db98401fa5f28badca75869c9f4fc507082ca8d7be294fe7910dcf2c.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S13<sub>:</sub> ERNIE<sub>-</sub>4<sub>.</sub>5<sub>-</sub>21B<sub>-</sub>A3B G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub> Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/a8a42d3fe0440f7cb48b1e975bdc1405795ce1816295f45154992ea99b1508d2.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S14<sub>:</sub> GLM<sub>-</sub>5<sub>.</sub>1 G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> tasks.

![](images/30b020a893d8b6e4831e26b52a6b43f9cbf5ef9f641041597eacadf5a28899db.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S15<sub>:</sub> N<sub>e</sub>m<sub>o</sub>tr<sub>o</sub>n<sub>-</sub>3<sub>-</sub>S<sub>upe</sub>r<sub>-</sub>120B<sub>-</sub>A12B G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/749fcc6c1bc1dc58a2ea084e434511c3ec3e791a7eb59cb8c1255fdacc8ce233.jpg)  
Figure S16: Qwen-3.6-35B-A3B Generated Team Hierarchy Performance Across 25 CREW-Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/9ba66aba449332d6274327a8ee388ef5679ee7ca076698df84aa36699c90173f.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S17<sub>:</sub> Ch<sub>a</sub>tGPT<sub>-</sub>5<sub>.</sub>4 G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> tasks.

![](images/1c274fb7e02bbc32f54941b3f36b432ffed7da48b7040d899e53fb24495b0f0d.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S18<sub>:</sub> D<sub>eep</sub>S<sub>ee</sub>k<sub>-</sub>V4<sub>-</sub>Pr<sub>o</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub> Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/9c779271b1f13ab1c88db0d1664499006ff39c90a63c8f31962c8938379da656.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S19<sub>:</sub> Ll<sub>a</sub>m<sub>a-</sub>4<sub>-</sub>S<sub>cou</sub>t<sub>-</sub>17B<sub>-</sub>16E<sub>-</sub>In<sub>s</sub>tr<sub>uc</sub>t G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/99051e0179e00b0e715e15a95155d2bbddb0d29b09f8523c713d586df1040ac4.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S20<sub>:</sub> G<sub>e</sub>mm<sub>a-</sub>4<sub>-</sub>it G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> tasks.

![](images/a435207d770b88ccb4c6430f02046763e2f2498e8d21cb0cb282661249de448b.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S21<sub>:</sub> ERNIE<sub>-</sub>4<sub>.</sub>5<sub>-</sub>21B<sub>-</sub>A3B G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub> Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/063b6d88a5af4ca32555c6f0ef06c4e2d939ecc0014a3c75bf7bb798888408aa.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S22<sub>:</sub> GLM<sub>-</sub>5<sub>.</sub>1 G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> tasks.

![](images/7b4705a6d8f1a46443177d1235e4ffe06dbaf5828a5c55b25484a2e2f6a6b949.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> S23<sub>:</sub> N<sub>e</sub>m<sub>o</sub>tr<sub>o</sub>n<sub>-</sub>3<sub>-</sub>S<sub>upe</sub>r<sub>-</sub>120B<sub>-</sub>A12B G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d T<sub>ea</sub>m Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n A<sub>c</sub>r<sub>oss</sub> 25 CREW<sub>-</sub>Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

![](images/5e6b9d89fafbfdcddfd049918a56340b3797ba3d35958833911127b67a128b64.jpg)  
Figure S24: Qwen-3.6-35B-A3B Generated Team Hierarchy Comparison Across 25 CREW-Wildfir<sub>e</sub> t<sub>as</sub>k<sub>s.</sub>

T<sub>a</sub>bl<sub>e</sub> S1<sub>:</sub> Wildfir<sub>e</sub> b<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk t<sub>as</sub>k <sub>co</sub>nfi<sub>gu</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> Th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>cons</sub>i<sub>s</sub>t<sub>s o</sub>f 25 t<sub>as</sub>k<sub>s w</sub>ith <sub>vary</sub>i<sub>ng</sub> difi<sub>cu</sub>lt<sub>y</sub> l<sub>eve</sub>l<sub>s,</sub> <sub>agen</sub>t <sub>compos</sub>iti<sub>ons,</sub> <sub>an</sub>d ti<sub>me</sub> h<sub>or</sub>i<sub>zons.</sub> Th<sub>e</sub> difi<sub>cu</sub>lt<sub>y</sub> <sub>score</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> in th<sub>e</sub> S<sub>upp</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>r<sub>y</sub> T<sub>ex</sub>t S<sub>ec</sub>ti<sub>o</sub>n S2<sub>.</sub>
<table><tr><td>Task</td><td>Difficulty Level</td><td>Difficulty Score</td><td>Number of Agents</td><td>Types of Agents</td><td>Firefighter</td><td>Bulldozer</td><td>Drone</td><td>Helicopter</td><td>Total Timestep</td></tr><tr><td>Rescue_Civilians_Known_Location_Small</td><td>Easy</td><td>0.0189</td><td>3</td><td>1</td><td>3</td><td>0</td><td>0</td><td>0</td><td>25</td></tr><tr><td>Cut_Trees_Sparse_Small</td><td>Easy</td><td>0.0284</td><td>3</td><td>1</td><td>3</td><td>0</td><td>0</td><td>0</td><td>30</td></tr><tr><td>Scout_Fire_Small</td><td>Easy</td><td>0.0284</td><td>3</td><td>1</td><td>0</td><td>0</td><td>3</td><td>0</td><td>30</td></tr><tr><td>Scout_Fire_Drone_Lost</td><td>Easy</td><td>0.0378</td><td>3</td><td>1</td><td>0</td><td>0</td><td>3</td><td>0</td><td>35</td></tr><tr><td>Scout_Fire_Large</td><td>Easy</td><td>0.0454</td><td>5</td><td>1</td><td>0</td><td>0</td><td>5</td><td>0</td><td>30</td></tr><tr><td>Rescue_Civilians_Known_Location_Large</td><td>Easy</td><td>0.0549</td><td>5</td><td>1</td><td>5</td><td>0</td><td>0</td><td>0</td><td>35</td></tr><tr><td>Rescue_Civilians_Surprise</td><td>Easy</td><td>0.1022</td><td>5</td><td>1</td><td>5</td><td>0</td><td>0</td><td>0</td><td>60</td></tr><tr><td>Cut_Trees_Lines_Small</td><td>Medium</td><td>0.1117</td><td>3</td><td>2</td><td>2</td><td>1</td><td>0</td><td>0</td><td>30</td></tr><tr><td>Transport_Firefighters_Small</td><td>Medium</td><td>0.1174</td><td>7</td><td>2</td><td>6</td><td>0</td><td>0</td><td>1</td><td>15</td></tr><tr><td>Cut_Trees_Sparse_Large</td><td>Medium</td><td>0.1258</td><td>10</td><td>1</td><td>10</td><td>0</td><td>0</td><td>0</td><td>50</td></tr><tr><td>Cut_Trees_Lines_Large</td><td>Medium</td><td>0.1458</td><td>7</td><td>2</td><td>4</td><td>3</td><td>0</td><td>0</td><td>30</td></tr><tr><td>Rescue_Civilians_Search_And_Rescue</td><td>Medium</td><td>0.1647</td><td>7</td><td>2</td><td>5</td><td>0</td><td>2</td><td>0</td><td>40</td></tr><tr><td>Suppress_Fire_Contain_Water_Source</td><td>Medium</td><td>0.1778</td><td>5</td><td>1</td><td>5</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>Transport_Helicopter_Down</td><td>Hard</td><td>0.1978</td><td>12</td><td>2</td><td>10</td><td>0</td><td>0</td><td>2</td><td>35</td></tr><tr><td>Suppress_Fire_Extinguish</td><td>Hard</td><td>0.2034</td><td>8</td><td>1</td><td>8</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>Suppress_Fire_Extinguish_Second_Fire</td><td>Hard</td><td>0.2034</td><td>8</td><td>1</td><td>8</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>Transport_FirefightersLarge</td><td>Hard</td><td>0.2148</td><td>14</td><td>2</td><td>12</td><td>0</td><td>0</td><td>2</td><td>35</td></tr><tr><td>Suppress_Fire_Contain</td><td>Hard</td><td>0.2782</td><td>7</td><td>2</td><td>5</td><td>2</td><td>0</td><td>0</td><td>100</td></tr><tr><td>Rescue_Civilians_Search_Rescue_Transport</td><td>Hard</td><td>0.3454</td><td>14</td><td>3</td><td>10</td><td>0</td><td>2</td><td>2</td><td>60</td></tr><tr><td>Suppress_Fire_Extinguish_Rapid_Growth</td><td>Extreme</td><td>0.3615</td><td>7</td><td>3</td><td>5</td><td>0</td><td>1</td><td>1</td><td>100</td></tr><tr><td>Suppress_Fire_Locate_And_Suppress</td><td>Extreme</td><td>0.3700</td><td>8</td><td>3</td><td>5</td><td>1</td><td>2</td><td>0</td><td>100</td></tr><tr><td>Suppress_Fire_Locate_Deploy_Suppress</td><td>Extreme</td><td>0.4589</td><td>14</td><td>3</td><td>10</td><td>0</td><td>2</td><td>2</td><td>120</td></tr><tr><td>Full_Game</td><td>Extreme</td><td>0.5508</td><td>15</td><td>4</td><td>10</td><td>1</td><td>2</td><td>2</td><td>120</td></tr><tr><td>Scale_Level_Simple</td><td>Extreme</td><td>0.5608</td><td>50</td><td>1</td><td>50</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>Scale_Level_Complex</td><td>Extreme</td><td>1.0000</td><td>50</td><td>4</td><td>25</td><td>5</td><td>10</td><td>10</td><td>200</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> S2<sub>:</sub> S<sub>co</sub>r<sub>e</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r <sub>eac</sub>h <sub>w</sub>ildfir<sub>e</sub> t<sub>as</sub>k<sub>.</sub> F<sub>or</sub> <sub>non-suppress</sub>i<sub>on</sub> t<sub>as</sub>k<sub>s,</sub> th<sub>e</sub> <sub>score</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>va</sub>l<sub>ue o</sub>f th<sub>e pr</sub>i<sub>mary</sub> t<sub>as</sub>k <sub>me</sub>t<sub>r</sub>i<sub>c.</sub> F<sub>or suppress</sub>i<sub>on,</sub> f<sub>u</sub>ll<sub>-game, an</sub>d <sub>comp</sub>l<sub>ex-</sub> <sub>sca</sub>l<sub>e</sub> t<sub>as</sub>k<sub>s,</sub> th<sub>e score uses</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d f<sub>ormu</sub>l<sub>a s</sub>h<sub>own</sub> i<sub>n</sub> th<sub>e</sub> t<sub>a</sub>bl<sub>e.</sub>
<table><tr><td>Task</td><td>Score calculation</td><td>Max score</td></tr><tr><td>Cut_Trees_Sparse_small</td><td>correct_trees_cut</td><td>18</td></tr><tr><td>Cut_Trees_Sparse_large</td><td>correct_trees_cut</td><td>75</td></tr><tr><td>Cut_Trees_Lines_small</td><td>correct_trees_cut</td><td>30</td></tr><tr><td>Cut_Trees_Lines_large</td><td>correct_trees_cut</td><td>105</td></tr><tr><td>Scale_Level_Simple</td><td>correct_trees_cut</td><td>300</td></tr><tr><td>Scout_Fire_small</td><td>fire_scouted</td><td>2</td></tr></table>

Continued on the next page

T<sub>a</sub>bl<sub>e</sub> S2 <sub>co</sub>ntin<sub>ue</sub>d fr<sub>o</sub>m th<sub>e</sub> <sub>p</sub>r<sub>ev</sub>i<sub>ous</sub> <sub>page</sub>
<table><tr><td>Task</td><td>Score calculation</td><td>Max score</td></tr><tr><td>Scout_Fire_large</td><td>fire_scouted</td><td>2</td></tr><tr><td>Scout_Fire_Drone_Lost</td><td>fire_scouted</td><td>2</td></tr><tr><td>Transport_Firefighters_small</td><td>firefighters_transported</td><td>6</td></tr><tr><td>Transport_Firefighters_large</td><td>firefighters_transported</td><td>12</td></tr><tr><td>Transport_Helicopter_Down</td><td>firefighters_transported</td><td>10</td></tr><tr><td>Rescue_Civilians_Known_Location_small</td><td>civilians_rescued</td><td>3</td></tr><tr><td>Rescue_Civilians_Known_Location_large</td><td>civilians_rescued</td><td>9</td></tr><tr><td>Rescue_Civilians_Search_and_Rescue</td><td>civilians_rescued</td><td>5</td></tr><tr><td>Rescue_Civilians_Search_Rescue_Transport</td><td>civilians_rescued</td><td>10</td></tr><tr><td>Rescue_Civilians_Surprise</td><td>civilians_rescued</td><td>10</td></tr><tr><td>Suppress_Fire_Contain</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Suppress_Fire_Contain_Water_Source</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Suppress_Fire_Extinguish</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Suppress_Fire_Extinguish_Second_Fire</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Suppress_Fire_Extinguish_Rapid_Growth</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Suppress_Fire_Locate_and_Suppress</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Suppress_Fire_Locate_Deploy_Suppress</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + R _ { t }$ </td><td></td></tr><tr><td>Full_Game</td><td> $0 . 5 \Delta T + 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } + 1 0 C _ { r } +$ </td><td></td></tr><tr><td rowspan="3">Scale_Level_Complex</td><td> $5 C _ { s } - 5 C _ { d } + R _ { t }$ </td><td></td></tr><tr><td> $0 . 5 \Delta T \mathrm { ~ + ~ } 1 0 F _ { e } + 5 F _ { s } + 5 W _ { s } - 5 A _ { d } +$ </td><td></td></tr><tr><td> $+ 1 0 C _ { r } + 5 C _ { s } - 5 C _ { d } + R _ { t }$ </td><td></td></tr></table>

T<sub>a</sub>bl<sub>e</sub> S3<sub>:</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>ons o</sub>f th<sub>e score-ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on sym</sub>b<sub>o</sub>l<sub>s.</sub>
<table><tr><td>Symbol Definition</td><td></td><td>Symbol Definition</td><td></td></tr><tr><td> $\Delta T$ </td><td>trees.  $\mathbf { f i r e } _ { \mathrm { n o o p } } - \mathbf { t r e e s } \mathbf { . } \mathbf { f i r e } _ { \mathrm { r u n } }$ </td><td> $F _ { e }$ </td><td>fire_extinguished</td></tr><tr><td> $F _ { s }$ </td><td>fire_scouted</td><td> $W _ { s }$ </td><td>water_scouted</td></tr><tr><td> $A _ { d }$ </td><td>agents_destroyed</td><td> $C _ { r }$ </td><td>civilians_rescued</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> S3 <sub>co</sub>ntin<sub>ue</sub>d fr<sub>o</sub>m th<sub>e</sub> <sub>p</sub>r<sub>ev</sub>i<sub>ous</sub> <sub>page</sub>
<table><tr><td>Symbol Definition</td><td></td><td>Symbol Definition</td><td></td></tr><tr><td> $C _ { s }$ </td><td>civilians_scouted</td><td> $C _ { d }$ </td><td>civilians_destroyed</td></tr><tr><td> $R _ { t }$ </td><td> $\left( 1 - \frac { t _ { \mathrm { e n d } } } { T } \right)$  1000</td><td></td><td></td></tr></table>

## Su<sub>pp</sub>lementar<sub>y</sub> Text

## S1 Al<sub>go</sub>rithm

Al<sub>gor</sub>ith<sub>m</sub> 1<sub>:</sub> WILDFIRE hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>mu</sub>lti<sub>-agen</sub>t <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m.</sub> Th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>ga</sub>th<sub>ers</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>o</sub>tt<sub>om-up</sub> th<sub>roug</sub>h th<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d hi<sub>erarc</sub>h<sub>y</sub> <sub>an</sub>d <sub>propaga</sub>t<sub>es</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons,</sub> <sub>p</sub>h<sub>ases,</sub> <sub>an</sub>d t<sub>as</sub>k<sub>s</sub> t<sub>op-</sub>d<sub>own</sub> b<sub>e</sub>f<sub>ore</sub> <sub>wor</sub>k<sub>ers</sub> <sub>ac</sub>t i<sub>n</sub> th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t<sub>.</sub>

Re<sub>q</sub>uire: Mission descri<sub>p</sub>tion �, worker set W, maximum timeste<sub>p</sub>s $T _ { \mathrm { m a x } }$ <sub>,</sub> <sub>an</sub>d LLM $\mathcal { L }$

E<sub>nsure:</sub> Fi<sub>na</sub>l <sub>m</sub>i<sub>ss</sub>i<sub>on s</sub>t<sub>a</sub>t<sub>e, per</sub>f<sub>ormance score,</sub> l<sub>ogs, an</sub>d <sub>agen</sub>t hi<sub>s</sub>t<sub>or</sub>i<sub>es</sub>

1<sub>:</sub> G<sub>enera</sub>t<sub>e</sub> t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y</sub> � <sub>us</sub>i<sub>ng</sub> $\mathcal { L }$ <sub>or</sub> f<sub>rom</sub> h<sub>uman</sub> <sub>exper</sub>t

2<sub>:</sub> A<sub>ss</sub>i<sub>gn</sub> <sub>env</sub>i<sub>ronmen</sub>t <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> � t<sub>o</sub> <sub>roo</sub>t <sub>manager</sub> <sub>�</sub>

3<sub>:</sub> f<sub>or</sub> $t = 0$ to $T _ { \mathrm { m a x } } - 1$ do

4<sub>:</sub> R<sub>ece</sub>i<sub>ve env</sub>i<sub>ronmen</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { t }$

5<sub>:</sub> U<sub>p</sub>d<sub>a</sub>t<sub>e</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> f<sub>or</sub> <sub>a</sub>ll li<sub>v</sub>i<sub>ng</sub> <sub>agen</sub>t<sub>s</sub>

6<sub>:</sub> if <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>s comp</sub>l<sub>e</sub>t<sub>e or env</sub>i<sub>ronmen</sub>t h<sub>as</sub> t<sub>erm</sub>i<sub>na</sub>t<sub>e</sub>d th<sub>en</sub>

$$
\mathbf { i f } \ i \in \mathcal { W }
$$

$$
 \mathcal { L } ( P _ { i }
$$

17<sub>:</sub> mission and <sub>p</sub>hase <sub>p</sub>ro<sub>g</sub>ress ← L(child re<sub>p</sub>orts, mana<sub>g</sub>er context)

$$
d _ { i }
$$

22<sub>:</sub> <sub>en</sub>d if   
23<sub>:</sub> if $d _ { i }$ <sub>c</sub>h<sub>anges</sub> th<sub>e curren</sub>t <sub>p</sub>l<sub>an</sub> th<sub>e</sub>n   
24<sub>:</sub> $d _ { i }$ ← ConfirmDecision $( d _ { i } .$ , standing orders, operator feedback)   
25<sub>:</sub> <sub>en</sub>d if   
26<sub>:</sub> <sub>en</sub>d if   
27<sub>:</sub> <sub>e</sub>nd f<sub>o</sub>r   
28<sub>:</sub> <sub>e</sub>nd f<sub>o</sub>r   
29<sub>:</sub> To<sub>p</sub>-do<sub>w</sub>n action <sub>p</sub>hase   
30<sub>:</sub> f<sub>or a</sub>ll hi<sub>erarc</sub>h<sub>y</sub> l<sub>eve</sub>l<sub>s</sub> ℓ<sub>,</sub> f<sub>rom roo</sub>t t<sub>o wor</sub>k<sub>ers</sub> d<sub>o</sub>   
31<sub>:</sub> f<sub>o</sub>r <sub>a</sub>ll li<sub>v</sub>i<sub>ng agen</sub>t<sub>s</sub> � <sub>a</sub>t l<sub>eve</sub>l ℓ in <sub>pa</sub>r<sub>a</sub>ll<sub>e</sub>l d<sub>o</sub>   
32<sub>:</sub> if � i<sub>s a ver</sub>ti<sub>ca</sub>l <sub>manager</sub> th<sub>en</sub>   
33<sub>:</sub> $\Pi _ { i }$ ← UpdateVerticalPlan $( d _ { i } .$ , mission, current phase, future phases)   
34<sub>:</sub> <sub>e</sub>l<sub>se</sub> if � i<sub>s a</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager</sub> th<sub>en</sub>   
35<sub>:</sub> $\Pi _ { i }$ ← UpdateHorizontalPlan $( d _ { i }$ , mission, child assignments)   
36<sub>:</sub> <sub>en</sub>d if   
37<sub>:</sub> if � i<sub>s a manager an</sub>d $\Pi _ { i }$ i<sub>s a new p</sub>l<sub>an</sub> th<sub>en</sub>   
38<sub>:</sub> repeat   
39<sub>:</sub> $F _ { i }$ ← CollectChildFeedback $( \Pi _ { i } )$   
40<sub>:</sub> if $F _ { i }$ contains actionable objections then   
41<sub>:</sub> R<sub>ev</sub>i<sub>se</sub> $\Pi _ { i }$ us<sup>i</sup>n<sub>g</sub> $F _ { i }$   
42<sub>:</sub> <sub>en</sub>d if   
43<sub>:</sub> <sub>un</sub>til $F _ { i }$ contains no actionable objections   
44<sub>:</sub> A<sub>ss</sub>i<sub>gn approve</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>ons, p</sub>h<sub>ases, an</sub>d t<sub>as</sub>k<sub>s</sub> t<sub>o c</sub>hild<sub>ren</sub>   
45<sub>:</sub> <sub>en</sub>d if   
46<sub>:</sub> if $i \in \mathcal W$ <sub>an</sub>d � <sub>requ</sub>i<sub>res</sub> <sub>new</sub> <sub>ac</sub>ti<sub>ons</sub> th<sub>e</sub>n   
47<sub>:</sub> $a _ { i }$ ← GenerateWorkerAction $( P _ { i }$ <sub>,</sub> t<sub>as</sub>k<sub>,</sub> $s _ { t } )$   
48<sub>:</sub> if $a _ { i }$ i<sub>s</sub> i<sub>nva</sub>lid <sub>or unava</sub>il<sub>a</sub>bl<sub>e</sub> th<sub>en</sub>   
49<sub>:</sub> $a _ { i }$ ← Idle   
50<sub>:</sub> <sub>en</sub>d if

51<sub>:</sub> <sub>en</sub>d if

52<sub>:</sub> <sub>e</sub>nd f<sub>o</sub>r

53<sub>:</sub> <sub>e</sub>nd f<sub>o</sub>r

54: Construct joint worker action

55<sub>:</sub> $\mathbf { a } _ { t } \gets \{ a _ { i } \ | \ i \in \mathcal { W } \}$

56<sub>:</sub> E<sub>xecu</sub>t<sub>e env</sub>i<sub>ronmen</sub>t t<sub>rans</sub>iti<sub>on</sub>

57<sub>:</sub> $s _ { t + 1 }$ ← EnvironmentStep(�<sub>�</sub>, a<sub>�</sub>)

58<sub>:</sub> R<sub>ecor</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons,</sub> <sub>ac</sub>ti<sub>ons,</sub> <sub>scores,</sub> <sub>an</sub>d LLM <sub>cos</sub>t<sub>s</sub>

59<sub>:</sub> <sub>e</sub>nd f<sub>o</sub>r

60<sub>: re</sub>t<sub>urn</sub> fi<sub>na</sub>l <sub>env</sub>i<sub>ronmen</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e, score,</sub> l<sub>ogs, an</sub>d <sub>agen</sub>t hi<sub>s</sub>t<sub>or</sub>i<sub>es</sub>

61: function VerticalDecision(mission, <sub>p</sub>hase, <sub>p</sub>ro<sub>g</sub>ress, re<sub>p</sub>orts)

62<sub>:</sub> if <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>s comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>en</sub>

63<sub>:</sub> return N<sub>ew</sub>M<sub>ission</sub>

64<sub>: e</sub>l<sub>se</sub> if <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> i<sub>s comp</sub>l<sub>e</sub>t<sub>e an</sub>d f<sub>u</sub>t<sub>ure p</sub>h<sub>ases ex</sub>i<sub>s</sub>t th<sub>en</sub>

65<sub>:</sub> return N<sub>ext</sub>P<sub>h</sub>a<sub>se</sub>

66<sub>:</sub> <sub>e</sub>l<sub>se</sub> if <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> i<sub>s</sub> <sub>unsu</sub>it<sub>a</sub>bl<sub>e</sub> <sub>or</sub> <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> <sub>are</sub> i<sub>ne</sub>f<sub>ec</sub>ti<sub>ve</sub> th<sub>en</sub>

67<sub>:</sub> r<sub>e</sub>t<sub>u</sub>rn R<sub>ewrite</sub>T<sub>asks</sub>

68<sub>:</sub> <sub>e</sub>l<sub>se</sub>

69<sub>:</sub> r<sub>e</sub>t<sub>u</sub>rn C<sub>ontinue</sub>P<sub>hase</sub>

70<sub>:</sub> <sub>en</sub>d if

71<sub>: e</sub>nd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

72: function HorizontalDecision(mission, <sub>p</sub>ro<sub>g</sub>ress, re<sub>p</sub>orts)

73<sub>:</sub> if <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>s comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>en</sub>

74<sub>:</sub> return N<sub>ew</sub>M<sub>ission</sub>

75<sub>: e</sub>l<sub>se</sub> if <sub>para</sub>ll<sub>e</sub>l <sub>ass</sub>i<sub>gnmen</sub>t<sub>s are unsu</sub>it<sub>a</sub>bl<sub>e or</sub> i<sub>ne</sub>f<sub>ec</sub>ti<sub>ve</sub> th<sub>en</sub>

76<sub>:</sub> r<sub>e</sub>t<sub>u</sub>rn R<sub>ewrite</sub>T<sub>asks</sub>

77<sub>:</sub> <sub>e</sub>l<sub>se</sub>

78<sub>:</sub> r<sub>e</sub>t<sub>u</sub>rn C<sub>ontinue</sub>M<sub>ission</sub>

## 79<sub>:</sub> <sub>en</sub>d if 80<sub>: e</sub>nd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

81: function UpdateVerticalPlan(decision, mission, <sub>p</sub>hase, future <sub>p</sub>hases)

82: if �������� = NewMission then

83<sub>:</sub> D<sub>ecompose m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o sequen</sub>ti<sub>a</sub>l <sub>p</sub>h<sub>ases</sub>

84<sub>:</sub> S<sub>e</sub>t th<sub>e</sub> fi<sub>rs</sub>t <sub>p</sub>h<sub>ase as</sub> th<sub>e curren</sub>t <sub>p</sub>h<sub>ase</sub>

85<sub>:</sub> G<sub>enera</sub>t<sub>e one</sub> t<sub>as</sub>k f<sub>or eac</sub>h <sub>c</sub>hild

86: else if �������� = NextPhase then

87<sub>:</sub> M<sub>ove</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d <sub>p</sub>h<sub>ase</sub> i<sub>n</sub>t<sub>o</sub> <sub>p</sub>h<sub>ase</sub> hi<sub>s</sub>t<sub>ory</sub>

88<sub>:</sub> S<sub>e</sub>l<sub>ec</sub>t th<sub>e</sub> <sub>nex</sub>t f<sub>u</sub>t<sub>ure</sub> <sub>p</sub>h<sub>ase</sub>

89<sub>:</sub> G<sub>enera</sub>t<sub>e c</sub>hild t<sub>as</sub>k<sub>s</sub> f<sub>or</sub> th<sub>e new p</sub>h<sub>ase</sub>

90: else if �������� = RewriteTasks then

91<sub>:</sub> R<sub>ewr</sub>it<sub>e</sub> <sub>c</sub>hild t<sub>as</sub>k<sub>s</sub> f<sub>or</sub> th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub>

92<sub>:</sub> <sub>e</sub>l<sub>se</sub>

93<sub>:</sub> P<sub>reserve</sub> th<sub>e curren</sub>t <sub>p</sub>h<sub>ase an</sub>d <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub>

94<sub>:</sub> <sub>en</sub>d if

95<sub>:</sub> return <sub>up</sub>dated <sub>p</sub>lan

96<sub>:</sub> <sub>e</sub>nd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

97: function UpdateHorizontalPlan(decision, mission, assi<sub>g</sub>nments)

98: if �������� = NewMission then

99: D<sub>ecompose</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> <sub>para</sub>ll<sub>e</sub>l <sub>c</sub>hild t<sub>as</sub>k<sub>s</sub>

100: else if �������� = RewriteTasks then

101<sub>:</sub> R<sub>ewr</sub>it<sub>e</sub> <sub>para</sub>ll<sub>e</sub>l <sub>c</sub>hild<sub>-</sub>t<sub>as</sub>k <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub>

102<sub>:</sub> <sub>e</sub>l<sub>se</sub>

103<sub>:</sub> P<sub>reserve</sub> th<sub>e curren</sub>t <sub>m</sub>i<sub>ss</sub>i<sub>on an</sub>d <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub>

104<sub>:</sub> <sub>en</sub>d if

105<sub>:</sub> return <sub>up</sub>dated <sub>p</sub>lan

106<sub>: e</sub>nd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

107: function CollectChildFeedback(<sub>p</sub>lan)

108: for all living children � in parallel do

109<sub>:</sub> Ask � to evaluate its assigned role in the plan

110<sub>:</sub> if <sub>ass</sub>i<sub>gnmen</sub>t i<sub>s</sub> f<sub>eas</sub>ibl<sub>e</sub> <sub>an</sub>d <sub>capa</sub>bilit<sub>y-cons</sub>i<sub>s</sub>t<sub>en</sub>t th<sub>en</sub>

111<sub>:</sub> Record approval from �

112<sub>:</sub> <sub>e</sub>l<sub>se</sub>

113<sub>:</sub> Record objection and suggested revision from �

114<sub>:</sub> <sub>en</sub>d if

115<sub>:</sub> <sub>e</sub>nd f<sub>o</sub>r

116<sub>: re</sub>t<sub>urn co</sub>ll<sub>ec</sub>t<sub>e</sub>d <sub>c</sub>hild f<sub>ee</sub>db<sub>ac</sub>k

117<sub>: e</sub>nd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

118: function GenerateWorkerAction(<sub>p</sub>erce<sub>p</sub>tion, task, state)

119: � ← L (worker <sub>p</sub>lanner, <sub>p</sub>erce<sub>p</sub>tion, task, state)

120<sub>:</sub> P<sub>arse or</sub>d<sub>ere</sub>d t<sub>ex</sub>t<sub>ua</sub>l <sub>ac</sub>ti<sub>on op</sub>ti<sub>ons</sub> f<sub>rom</sub> �

121: � ← L (worker translator, �)

122<sub>:</sub> V<sub>a</sub>lid<sub>a</sub>t<sub>e</sub> <sub>�</sub> <sub>aga</sub>i<sub>ns</sub>t <sub>wor</sub>k<sub>er</sub> <sub>capa</sub>biliti<sub>es</sub>

123<sub>:</sub> return �

124<sub>: e</sub>nd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n

## S2 T<sub>as</sub>k Difi<sub>cu</sub>lt<sub>y</sub> S<sub>core</sub> C<sub>a</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub>

Th<sub>e</sub> t<sub>as</sub>k difi<sub>cu</sub>lt<sub>y</sub> <sub>score</sub> i<sub>s</sub> <sub>compu</sub>t<sub>e</sub>d <sub>as</sub> <sub>a</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>sum</sub> <sub>o</sub>f <sub>m</sub>i<sub>n–max</sub> <sub>norma</sub>li<sub>ze</sub>d t<sub>as</sub>k f<sub>ea</sub>t<sub>ures:</sub>

$$
D _ { t } = 0 . 4 0 \times \widetilde { N } _ { \mathrm { a g e n t s } } + 0 . 2 5 \times \widetilde { N } _ { \mathrm { t y p e s } } + 0 . 3 5 \times \widetilde { N } _ { \mathrm { s t e p s } } ,
$$

$$
\widetilde { x } = \frac { x - x _ { \mathrm { m i n } } } { x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } } .
$$

<sub>w</sub>h<sub>ere</sub>

## S3 A<sub>gg</sub>r<sub>ega</sub>t<sub>e</sub>d P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> C<sub>a</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r th<sub>e</sub> Lin<sub>e</sub> Pl<sub>o</sub>t

L<sub>e</sub>t $m \in { \mathcal { M } }$ d<sub>eno</sub>t<sub>e an</sub> LLM <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> $a \in { \mathcal { A } }$ <sub>an</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m,</sub> $t \in \mathcal { T }$ <sub>a</sub> t<sub>as</sub>k<sub>, � a ran</sub>d<sub>om see</sub>d<sub>, an</sub>d $p \in [ 0 , 1 ]$ <sub>norma</sub>li<sub>ze</sub>d t<sub>as</sub>k <sub>progress.</sub>

For each run, the raw score trajectory is interpolated onto a common progress grid:

$$
p \in \{ 0 , 0 . 0 1 , 0 . 0 2 , \ldots , 1 . 0 0 \} .
$$

L<sub>e</sub>t $x _ { m , a , t , s } ( p )$ b<sub>e</sub> th<sub>e</sub> <sub>score</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l <sub>�,</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>�,</sub> t<sub>as</sub>k �<sub>,</sub> <sub>an</sub>d <sub>see</sub>d <sub>�</sub> <sub>a</sub>t <sub>progress</sub> <sub>�.</sub> S<sub>cores</sub> <sub>are</sub> <sub>norma</sub>li<sub>ze</sub>d t<sub>as</sub>k<sub>-w</sub>i<sub>se</sub> b<sub>y</sub> th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>b<sub>serve</sub>d <sub>score</sub> <sub>a</sub>t th<sub>e</sub> <sub>same</sub> <sub>progress</sub> <sub>po</sub>i<sub>n</sub>t<sub>:</sub>

$$
\tilde { x } _ { m , a , t , s } ( p ) = \frac { x _ { m , a , t , s } ( p ) } { \displaystyle \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } , s ^ { \prime } } x _ { m , a ^ { \prime } , t , s ^ { \prime } } ( p ) } .
$$

Th<sub>e</sub> <sub>see</sub>d<sub>-</sub>l<sub>eve</sub>l <sub>norma</sub>li<sub>ze</sub>d <sub>curves</sub> <sub>are</sub> th<sub>en</sub> <sub>average</sub>d <sub>w</sub>ithi<sub>n</sub> <sub>eac</sub>h <sub>mo</sub>d<sub>e</sub>l<sub>–a</sub>l<sub>gor</sub>ith<sub>m–</sub>t<sub>as</sub>k<sub>:</sub>

$$
\bar { x } _ { m , a , t } ( p ) = \frac { 1 } { | S _ { m , a , t } | } \sum _ { s \in S _ { m , a , t } } \tilde { x } _ { m , a , t , s } ( p ) .
$$

F<sub>or</sub> <sub>eac</sub>h <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>m,</sub> t<sub>as</sub>k <sub>curves</sub> <sub>are</sub> <sub>com</sub>bi<sub>ne</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>as</sub>k<sub>-</sub>difi<sub>cu</sub>lt<sub>y-score-we</sub>i<sub>g</sub>ht<sub>e</sub>d avera<sub>g</sub>e (the calculation can be found in section S2):

$$
y _ { m , a } ( p ) = \frac { \displaystyle \sum _ { t \in \mathcal { T } _ { m , a } } w _ { t } \bar { x } _ { m , a , t } ( p ) } { \displaystyle \sum _ { t \in \mathcal { T } _ { m , a } } w _ { t } } .
$$

Finall<sub>y</sub>, we avera<sub>g</sub>e these curves across the ei<sub>g</sub>ht LLM models (if a<sub>pp</sub>licable):

$$
\mu _ { a } ( \boldsymbol { p } ) = \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } y _ { m , a } ( \boldsymbol { p } ) .
$$

<sub>w</sub>h<sub>ere</sub>

$$
\begin{array} { r } { \mathcal { M } = \{ \mathrm { G P T } , \mathrm { E R N I E , G e m m a } , \mathrm { Q w e n } , \mathrm { L l a m a } , \mathrm { D e e p S e e k } , \mathrm { G L M } , \mathrm { N e m o t r o n } \} . } \end{array}
$$

Th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>error</sub> i<sub>s compu</sub>t<sub>e</sub>d <sub>across mo</sub>d<sub>e</sub>l<sub>s:</sub>

$$
\mathrm { S E } _ { a } ( p ) = \frac { \mathrm { s d } \left( \{ y _ { m , a } ( p ) : m \in M \} \right) } { \sqrt { | \mathcal { M } | } } .
$$

## S4 R<sub>a</sub>d<sub>ar</sub> Pl<sub>o</sub>t M<sub>e</sub>t<sub>r</sub>i<sub>c</sub> C<sub>a</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub>

Sin<sub>g</sub>l<sub>e-</sub>m<sub>o</sub>d<sub>e</sub>l<sub>, s</sub>in<sub>g</sub>l<sub>e-</sub>t<sub>as</sub>k r<sub>a</sub>d<sub>a</sub>r <sub>p</sub>l<sub>o</sub>t<sub>.</sub> F<sub>or a</sub> fi<sub>xe</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>�,</sub> t<sub>as</sub>k �<sub>, a</sub>l<sub>gor</sub>ith<sub>m �, an</sub>d <sub>ra</sub>d<sub>ar me</sub>t<sub>r</sub>i<sub>c</sub> �<sub>,</sub> th<sub>e raw me</sub>t<sub>r</sub>i<sub>c</sub> i<sub>s average</sub>d <sub>across a</sub>ll <sub>see</sub>d<sub>s</sub> $R _ { m , t , a }$

$$
\mu _ { m , t , a , k } = \frac { 1 } { | R _ { m , t , a } | } \sum _ { r \in R _ { m , t , a } } S _ { m , t , a , r , k } .
$$

Th<sub>e see</sub>d<sub>-average</sub>d <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> th<sub>en m</sub>i<sub>n–max norma</sub>li<sub>ze</sub>d <sub>across</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>ms</sub> $\mathcal { A }$ <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> <sub>same mo</sub>d<sub>e</sub>l <sub>an</sub>d t<sub>as</sub>k<sub>:</sub>

$$
N _ { m , t , a , k } = \left\{ \begin{array} { l l } { \frac { \mu _ { m , t , a , k } - \operatorname* { m i n } _ { b \in \mathcal { R } } \mu _ { m , t , b , k } } { \operatorname* { m a x } _ { b \in \mathcal { R } } \mu _ { m , t , b , k } - \operatorname* { m i n } _ { b \in \mathcal { R } } \mu _ { m , t , b , k } } , } & { k \mathrm { ~ i s ~ a ~ b e n e f i t ~ m e t r i c ~ ( t h e ~ h i g h e r ~ t h e ~ b e t t e r ) } , } \\ { \frac { \operatorname* { m a x } _ { b \in \mathcal { R } } \mu _ { m , t , b , k } - \mu _ { m , t , a , k } } { \operatorname* { m a x } _ { b \in \mathcal { R } } \mu _ { m , t , b , k } - \operatorname* { m i n } _ { b \in \mathcal { R } } \mu _ { m , t , b , k } } , } & { k \mathrm { ~ i s ~ a ~ c o s t ~ m e t r i c ~ ( t h e ~ l o w e r ~ t h e ~ b e t t e r ) } . } \end{array} \right.
$$

H<sub>ere,</sub> fi<sub>na</sub>l <sub>score,</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> <sub>an</sub>d <sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub> <sub>per</sub> <sub>s</sub>t<sub>ep</sub> <sub>are</sub> b<sub>ene</sub>fit <sub>me</sub>t<sub>r</sub>i<sub>cs,</sub> <sub>w</sub>h<sub>ereas</sub> API <sub>ca</sub>ll<sub>s,</sub> i<sub>npu</sub>t t<sub>o</sub>k<sub>ens,</sub> <sub>an</sub>d <sub>ou</sub>t<sub>pu</sub>t t<sub>o</sub>k<sub>ens</sub> <sub>per</sub> <sub>s</sub>t<sub>ep</sub> <sub>are</sub> <sub>cos</sub>t <sub>me</sub>t<sub>r</sub>i<sub>cs;</sub> <sub>consequen</sub>tl<sub>y,</sub> <sub>a</sub> l<sub>arger</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>ra</sub>d<sub>ar</sub> <sub>va</sub>l<sub>ue</sub> <sub>a</sub>l<sub>ways</sub> <sub>represen</sub>t<sub>s</sub> b<sub>e</sub>tt<sub>er</sub> <sub>per</sub>f<sub>ormance.</sub> If <sub>a</sub>ll <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> h<sub>ave</sub> th<sub>e</sub> <sub>same</sub> <sub>va</sub>l<sub>ue</sub> f<sub>or</sub> <sub>a</sub> <sub>me</sub>t<sub>r</sub>i<sub>c,</sub> th<sub>e</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> $N _ { m , t , a , k } = 1$

Dific<sub>u</sub>lt<sub>y</sub>-<sub>w</sub>ei<sub>g</sub>hted radar a<sub>gg</sub>re<sub>g</sub>ated across tasks and ei<sub>g</sub>ht models<sub>.</sub> U<sub>s</sub>in<sub>g</sub> th<sub>e</sub> <sub>s</sub>in<sub>g</sub>l<sub>e-</sub>t<sub>as</sub>k <sub>norma</sub>li<sub>ze</sub>d <sub>me</sub>t<sub>r</sub>i<sub>c</sub> $N _ { m , t , a , k }$ d<sub>e</sub>fi<sub>ne</sub>d <sub>a</sub>b<sub>ove,</sub> th<sub>e</sub> t<sub>as</sub>k<sub>-</sub>l<sub>eve</sub>l <sub>resu</sub>lt<sub>s</sub> f<sub>or</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>�</sub> <sub>are</sub> fi<sub>rs</sub>t <sub>aggrega</sub>t<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> f<sub>or</sub> <sub>eac</sub>h <sub>mo</sub>d<sub>e</sub>l <sub>�</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>as</sub>k difi<sub>cu</sub>lt<sub>y</sub> <sub>scores</sub> $D _ { t }$ :

$$
W _ { m , a , k } = \frac { \sum _ { t \in \mathcal { T } _ { m , a } } D _ { t } N _ { m , t , a , k } } { \sum _ { t \in \mathcal { T } _ { m , a } } D _ { t } } .
$$

Th<sub>e</sub> fi<sub>na</sub>l <sub>ra</sub>d<sub>ar</sub> <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> th<sub>en</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>averag</sub>i<sub>ng</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-</sub>l<sub>eve</sub>l difi<sub>cu</sub>lt<sub>y-we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>va</sub>l<sub>ues</sub> equall<sub>y</sub> across the available LLMs (if a<sub>pp</sub>licable):

$$
\mathrm { R a d a r } _ { a , k } = \frac { 1 } { | \mathcal { M } _ { a } | } \sum _ { m \in \mathcal { M } _ { a } } W _ { m , a , k } ,
$$

<sub>w</sub>h<sub>ere</sub>

$$
\begin{array} { r } { \mathcal { M } = \{ \mathrm { G P T } , \mathrm { E R N I E , G e m m a } , \mathrm { Q w e n } , \mathrm { L l a m a } , \mathrm { D e e p S e e k } , \mathrm { G L M } , \mathrm { N e m o t r o n } \} . } \end{array}
$$

Th<sub>us,</sub> t<sub>as</sub>k difi<sub>cu</sub>lt<sub>y</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>eac</sub>h t<sub>as</sub>k <sub>w</sub>ithi<sub>n a mo</sub>d<sub>e</sub>l<sub>, w</sub>hil<sub>e eac</sub>h <sub>o</sub>f th<sub>e</sub> <sub>e</sub>i<sub>g</sub>ht <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> <sub>equa</sub>ll<sub>y</sub> t<sub>o</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>aggrega</sub>t<sub>e</sub> <sub>ra</sub>d<sub>ar</sub> <sub>p</sub>l<sub>o</sub>t<sub>.</sub>

## S5 C<sub>o</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l R<sub>esou</sub>r<sub>ces</sub>

We use servers with 8× NVIDIA A100 80GB<sub>,</sub> 8× NVIDIA H200<sub>,</sub> and 8× NVIDIA L40S GPUs to <sub>con</sub>d<sub>uc</sub>t th<sub>e</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s.</sub> O<sub>pen-source</sub> l<sub>arge</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>were</sub> h<sub>os</sub>t<sub>e</sub>d l<sub>oca</sub>ll<sub>y</sub> f<sub>rom</sub> H<sub>ugg</sub>i<sub>ng</sub> Face using the vLLM (54) inference engine.

S6 Pr<sub>o</sub>m<sub>p</sub>t f<sub>o</sub>r th<sub>e</sub> A<sub>ge</sub>nt<sub>s</sub>

## P<sub>romp</sub>t S1<sub>:</sub> T<sub>as</sub>k D<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub>

<table><tr><td>Cut_Trees_Sparse_small</td></tr><tr><td>Small map with sparse trees to cut down.</td></tr><tr><td>Cut_Trees_Sparse_large</td></tr><tr><td>Large map with many sparse trees to cut down.</td></tr><tr><td>Cut_Trees_Lines_small</td></tr><tr><td>Small map with trees arranged in lines to create firebreaks.</td></tr><tr><td>Cut_Trees_Lines_large</td></tr><tr><td>Large map with multiple tree lines to cut.</td></tr><tr><td>Scout_Fire_small</td></tr><tr><td>Scout for fires in a small area using drones.</td></tr><tr><td>Scout_Fire_large</td></tr><tr><td>Scout for fires across a large area.</td></tr><tr><td>Transport_Firefighters_small</td></tr><tr><td>Use helicopters to transport firefighters.</td></tr><tr><td>Transport_Firefighters_large</td></tr><tr><td>Conduct large-scale firefighter transportation operations.</td></tr><tr><td>Rescue_Civilians_Known_Location_small</td></tr><tr><td>Rescue civilians from known locations.</td></tr><tr><td>Rescue_Civilians_Known_Location_large</td></tr><tr><td>Conduct large-scale civilian rescue operations at known locations.</td></tr><tr><td>Suppress_Fire_Contain</td></tr><tr><td>Contain the fire using firefighters and bulldozers.</td></tr><tr><td>Suppress_Fire_Extinguish</td></tr><tr><td>Extinguish the fire using water carried by firefighters.</td></tr></table>

S<sub>earc</sub>h f<sub>or</sub> <sub>an</sub>d <sub>rescue</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans</sub> <sub>us</sub>i<sub>ng</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>agen</sub>t t<sub>ypes.</sub>

Suppress Fire Locate and Suppress L<sub>oca</sub>t<sub>e</sub> fi<sub>res</sub> <sub>an</sub>d <sub>suppress</sub> th<sub>em</sub> <sub>us</sub>i<sub>ng</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>un</sub>it<sub>s.</sub>

Suppress Fire Locate Deploy Suppress P<sub>er</sub>f<sub>orm</sub> <sub>comp</sub>l<sub>ex</sub> fi<sub>re</sub> <sub>suppress</sub>i<sub>on</sub> <sub>us</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>un</sub>it t<sub>ypes.</sub>

Rescue Civilians Search Rescue Transport S<sub>earc</sub>h f<sub>or</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans,</sub> <sub>rescue</sub> th<sub>em,</sub> <sub>an</sub>d t<sub>ranspor</sub>t th<sub>em</sub> t<sub>o</sub> <sub>sa</sub>f<sub>e</sub>t<sub>y.</sub>

Full Game Complete a wildfire-response scenario involving all objectives.

Scout Fire Drone Lost   
A d<sub>rone</sub> b<sub>ecomes</sub> <sub>unava</sub>il<sub>a</sub>bl<sub>e</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>execu</sub>ti<sub>on,</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>eam</sub> t<sub>o</sub> <sub>a</sub>d<sub>ap</sub>t <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>re-</sub>   
<sub>ma</sub>i<sub>n</sub>i<sub>ng</sub> <sub>scou</sub>t<sub>s.</sub>   
Transport Helicopter Down   
A h<sub>e</sub>li<sub>cop</sub>t<sub>er</sub> b<sub>ecomes unava</sub>il<sub>a</sub>bl<sub>e</sub> d<sub>ur</sub>i<sub>ng</sub> t<sub>ranspor</sub>t<sub>a</sub>ti<sub>on, requ</sub>i<sub>r</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>eam</sub> t<sub>o rea</sub>ll<sub>oca</sub>t<sub>e</sub> th<sub>e</sub>   
<sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> h<sub>e</sub>li<sub>cop</sub>t<sub>ers.</sub>

Rescue Civilians Surprise Additi<sub>ona</sub>l <sub>c</sub>i<sub>v</sub>ili<sub>ans</sub> <sub>appear</sub> i<sub>n</sub> <sub>a</sub> <sub>new</sub> <sub>area</sub> d<sub>ur</sub>i<sub>ng</sub> t<sub>as</sub>k <sub>execu</sub>ti<sub>on.</sub>

## Suppress Fire Extinguish Second Fire

A <sub>secon</sub>d fi<sub>re</sub> b<sub>rea</sub>k<sub>s ou</sub>t d<sub>ur</sub>i<sub>ng execu</sub>ti<sub>on, requ</sub>i<sub>r</sub>i<sub>ng resources</sub> t<sub>o</sub> b<sub>e</sub> di<sub>v</sub>id<sub>e</sub>d b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> fi<sub>res.</sub>

Suppress Fire Contain Water Source Th<sub>e</sub> t<sub>eam</sub> b<sub>eg</sub>i<sub>ns</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>wa</sub>t<sub>er,</sub> <sub>an</sub>d <sub>a</sub> <sub>wa</sub>t<sub>er</sub> <sub>source</sub> b<sub>ecomes</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> d<sub>ur</sub>i<sub>ng</sub> t<sub>as</sub>k <sub>execu</sub>ti<sub>on.</sub>

Suppress Fire Extinguish Rapid Growth Th<sub>e</sub> fi<sub>re</sub> <sub>su</sub>dd<sub>en</sub>l<sub>y</sub> <sub>sprea</sub>d<sub>s</sub> <sub>rap</sub>idl<sub>y</sub> b<sub>ecause</sub> <sub>o</sub>f <sub>worsen</sub>i<sub>ng</sub> <sub>env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>con</sub>diti<sub>ons.</sub>

## Scale Level Simple

L<sub>arge</sub> <sub>map</sub> <sub>w</sub>ith <sub>many</sub> t<sub>rees</sub> <sub>an</sub>d fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> i<sub>n</sub> <sub>a</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>scenar</sub>i<sub>o.</sub>

Y<sub>ou are an exper</sub>t <sub>mu</sub>lti<sub>-agen</sub>t <sub>coor</sub>di<sub>na</sub>ti<sub>on s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>s</sub>t<sub>.</sub> Y<sub>ou</sub> d<sub>es</sub>i<sub>gn op</sub>ti<sub>ma</sub>l t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es</sub> f<sub>or</sub> <sub>comp</sub>l<sub>ex</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons.</sub> Al<sub>ways</sub> <sub>respon</sub>d <sub>w</sub>ith <sub>va</sub>lid JSON <sub>on</sub>l<sub>y</sub>

Y<sub>ou</sub> <sub>are</sub> <sub>an</sub> <sub>exper</sub>t i<sub>n</sub> <sub>mu</sub>lti<sub>-agen</sub>t t<sub>eam</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> f<sub>or</sub> <sub>w</sub>ildfi<sub>re</sub> <sub>response,</sub> <sub>searc</sub>h <sub>an</sub>d <sub>rescue,</sub> <sub>scou</sub>ti<sub>ng,</sub> <sub>an</sub>d fi<sub>re</sub>b<sub>rea</sub>k <sub>crea</sub>ti<sub>on</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons.</sub> D<sub>es</sub>i<sub>gn</sub> <sub>an</sub> <sub>op</sub>ti<sub>ma</sub>l t<sub>eam</sub> hi<sub>erarc</sub>h<sub>y</sub> f<sub>or</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>scenar</sub>i<sub>o:</sub>

IMPORTANT<sub>:</sub> O<sub>n</sub>l<sub>y</sub> <sub>expec</sub>t <sub>w</sub>h<sub>a</sub>t th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> d<sub>eman</sub>d<sub>s.</sub> D<sub>o</sub> <sub>no</sub>t <sub>expec</sub>t <sub>poss</sub>ibl<sub>e</sub> <sub>s</sub>it<sub>ua</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>are</sub> <sub>no</sub>t <sub>men</sub>ti<sub>one</sub>d<sub>;</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>s</sub> f<sub>u</sub>ll<sub>y</sub> <sub>compre</sub>h<sub>ens</sub>i<sub>ve</sub> <sub>o</sub>f th<sub>e</sub> t<sub>as</sub>k t<sub>o</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e.</sub>

• Firefi<sub>g</sub>hters: {firefi<sub>g</sub>hter count}

• Bulldozers: {bulldozer count}

• Drones: {drone count}

• Helico<sub>p</sub>ters: {helico<sub>p</sub>ter count}

• Firefi<sub>g</sub>hter: Can cut trees slowl<sub>y,</sub> <sub>p</sub>ick u<sub>p</sub> and dro<sub>p</sub> of civilians<sub>,</sub> s<sub>p</sub>ra<sub>y</sub> water to extin-<sub>gu</sub>i<sub>s</sub>h fi<sub>res,</sub> <sub>an</sub>d <sub>re</sub>fill <sub>wa</sub>t<sub>er.</sub>

<sub>–</sub> NOTE<sub>:</sub> Fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub> <sub>canno</sub>t <sub>supp</sub>l<sub>y</sub> <sub>wa</sub>t<sub>er</sub> t<sub>o</sub> <sub>eac</sub>h <sub>o</sub>th<sub>er,</sub> <sub>so</sub> d<sub>o</sub> <sub>no</sub>t <sub>crea</sub>t<sub>e</sub> <sub>separa</sub>t<sub>e</sub> <sub>wa</sub>t<sub>er-re</sub>fill <sub>an</sub>d fi<sub>re-suppress</sub>i<sub>on</sub> t<sub>eams.</sub>

• Bulldozer: Can cut trees <sub>q</sub>uickl<sub>y,</sub> but cannot interact with fires or civilians.

• Drone: Can scout and observe areas<sub>,</sub> but cannot mani<sub>p</sub>ulate the environment.

• Helico<sub>p</sub>ter: Can trans<sub>p</sub>ort u<sub>p</sub> to five firefi<sub>g</sub>hters to diferent locations and can refill <sub>an</sub>d d<sub>ep</sub>l<sub>oy</sub> <sub>wa</sub>t<sub>er</sub> <sub>on</sub> fi<sub>res.</sub>

## MANAGER TYPES <sub>-</sub> CHOOSE CAREFULLY<sub>:</sub>

## 1. Vertical Mana<sub>g</sub>er (Temporal Coordination)

• Use when this mana<sub>g</sub>er’s direct children re<sub>q</sub>uire se<sub>q</sub>uential or real-time coordinat<sup>i</sup>on, suc<sup>h</sup> as trans<sub>p</sub>ort<sup>i</sup>n<sub>g</sub> a<sub>g</sub>ents.

• Children work throu<sub>g</sub>h <sub>p</sub>hases to<sub>g</sub>ether in lockste<sub>p</sub>.

• Question to ask: “Do my direct children need to wait for each other?”

## 2. Horizontal Mana<sub>g</sub>er (Parallel Coordination)

• Use when this mana<sub>g</sub>er’s direct children work on inde<sub>p</sub>endent subtasks.

• Children execute simultaneousl<sub>y</sub> without interde<sub>p</sub>endencies.

• Question to ask: “Can my direct children work independently?”

## TEAM NAMES<sub>:</sub>

• Team names should be descri<sub>p</sub>tive and role-s<sub>p</sub>ecific.

• Exam<sub>p</sub>les: ”FIRE SUPPRESS ALPHA”<sub>,</sub> ”RECON TEAM”<sub>,</sub> ”HYBRID TEAM”<sub>,</sub> <sub>a</sub>nd ”GROUND TEAM”<sub>.</sub>

• A team name should communicate the team’s <sub>p</sub>ur<sub>p</sub>ose and sco<sub>p</sub>e.

• A meanin<sub>g</sub>ful team name should hel<sub>p</sub> team members understand their mission context i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y.</sub>

## MIXED CHILDREN<sub>:</sub>

M<sub>anagers</sub> <sub>can</sub> h<sub>ave</sub> b<sub>o</sub>th <sub>wor</sub>k<sub>er</sub> <sub>c</sub>hild<sub>ren</sub> <sub>an</sub>d <sub>su</sub>b<sub>-manager</sub> <sub>c</sub>hild<sub>ren.</sub>

• Exam<sub>p</sub>le: A coordination mana<sub>g</sub>er ma<sub>y</sub> directl<sub>y</sub> su<sub>p</sub>ervise scout drones and two firefi<sub>g</sub>hti<sub>ng</sub> <sub>su</sub>bt<sub>eams.</sub>

## CONSTRAINTS AND NOTES<sub>:</sub>

• Exactl<sub>y</sub> one root mana<sub>g</sub>er is re<sub>q</sub>uired.

• Ever<sub>y</sub> mana<sub>g</sub>er must have at least two children.

• Use all workers exactl<sub>y</sub>: no more and no fewer.

• The hierarch<sub>y</sub> must not contain c<sub>y</sub>cles.

• A mana<sub>g</sub>er can have workers<sub>,</sub> sub-mana<sub>g</sub>ers<sub>,</sub> or both as children.

• Mana<sub>g</sub>ers with onl<sub>y</sub> one child are redundant.

• Worker a<sub>g</sub>ents alread<sub>y</sub> have some internal <sub>p</sub>lannin<sub>g</sub> ca<sub>p</sub>abilities.

• Teams ma<sub>y</sub> be hetero<sub>g</sub>eneous and contain mixed a<sub>g</sub>ent t<sub>yp</sub>es.

• Kee<sub>p</sub> the structure as minimal and sim<sub>p</sub>le as <sub>p</sub>ossible.

## OUTPUT FORMAT EXAMPLE (JSON ONLY):

{   
"explanation": "Brief explanation of the strategic reasoning behind   
this team structure, including why you chose vertical versus horizonta   
managers and how the hierarchy supports the mission",   
"root\_manager": {   
"manager\_type": "vertical\_manager",   
"team\_name": "DESCRIPTIVE\_NAME",   
"children": [

{   
"type": "manager",   
"manager\_type": "vertical\_manager",   
"team\_name": "DESCRIPTIVE\_NAME",   
"children": [   
{   
"type": "worker",   
"worker\_type": "firefighter",   
"count": 1   
},   
{   
"type": "worker",   
"worker\_type": "bulldozer",   
"count": 1   
},   
{   
"type": "worker",   
"worker\_type": "helicopter",   
"count": 1   
}   
]   
},   
{   
"type": "manager",   
"manager\_type": "horizontal\_manager",   
"team\_name": "DESCRIPTIVE\_NAME",   
"children": [   
{   
"type": "worker",

"worker\_type": "drone",   
"count": 1   
},   
{   
"type": "worker",   
"worker\_type": "firefighter",   
"count": 1   
}   
]   
},   
{   
"type": "worker",   
"worker\_type": "firefighter",   
"count": 1   
}   
]   
}   
}   
IMPORTANT<sub>:</sub>   
• Think strate<sub>g</sub>icall<sub>y</sub> about the mission’s coordination needs.   
• Choose mana<sub>g</sub>er t<sub>yp</sub>es based on whether their direct children re<sub>q</sub>uire se<sub>q</sub>uential coor  
di<sub>na</sub>ti<sub>on, para</sub>ll<sub>e</sub>l <sub>coor</sub>di<sub>na</sub>ti<sub>on, or</sub> b<sub>o</sub>th<sub>.</sub>   
• Create meanin<sub>g</sub>ful team names that communicate each team’s <sub>p</sub>ur<sub>p</sub>ose.   
• Ensure that all {total workers} workers are assi<sub>g</sub>ned.   
• Include a clear ex<sub>p</sub>lanation of the desi<sub>g</sub>n choices.

• Mana<sub>g</sub>ers with onl<sub>y</sub> one worker are redundant. Workers can handle some <sub>p</sub>lannin<sub>g</sub> and <sub>reason</sub>i<sub>ng</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y.</sub>

• The count value must be an inte<sub>g</sub>er onl<sub>y</sub>. It must not contain decimals<sub>,</sub> strin<sub>g</sub>s<sub>,</sub> or t<sub>ex</sub>t<sub>ua</sub>l d<sub>escr</sub>i<sub>p</sub>ti<sub>ons.</sub>

• The count value must satisf<sub>y</sub> count ≥ 1.

• The count value must never be:

<sub>–</sub> A fl<sub>oa</sub>t<sub>, suc</sub>h <sub>as</sub> 1<sub>.</sub>5<sub>.</sub>

<sub>–</sub> A <sub>s</sub>t<sub>r</sub>i<sub>ng,</sub> <sub>suc</sub>h <sub>as</sub> 3<sub>.</sub>

<sub>–</sub> N<sub>u</sub>ll<sub>.</sub>

<sub>–</sub> A t<sub>ex</sub>t<sub>ua</sub>l d<sub>escr</sub>i<sub>p</sub>ti<sub>on,</sub> <sub>suc</sub>h <sub>as</sub> ”<sub>a</sub>ll <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>ers</sub>”<sub>.</sub>

## RULE OF THUMB<sub>:</sub>

K<sub>eep</sub> t<sub>eams</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> <sub>w</sub>h<sub>enever</sub> <sub>poss</sub>ibl<sub>e.</sub>

• Start with the sim<sub>p</sub>lest <sub>p</sub>ossible team containin<sub>g</sub> onl<sub>y</sub> the root mana<sub>g</sub>er.

• Then add subteams and additional hierarch<sub>y</sub> onl<sub>y</sub> when necessar<sub>y</sub>.

• Onl<sub>y</sub> make subteams when there is a critical need for:

<sub>–</sub> S<sub>y</sub>n<sub>c</sub>hr<sub>o</sub>ni<sub>ze</sub>d t<sub>as</sub>k d<sub>eco</sub>m<sub>pos</sub>iti<sub>o</sub>n<sub>,</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h t<sub>eam</sub> <sub>mem</sub>b<sub>ers</sub> <sub>mus</sub>t <sub>wor</sub>k th<sub>roug</sub>h <sub>p</sub>h<sub>ases</sub> t<sub>oge</sub>th<sub>er un</sub>d<sub>er a ver</sub>ti<sub>ca</sub>l <sub>manager.</sub>

<sub>–</sub> S<sub>ca</sub>lin<sub>g</sub> d<sub>eco</sub>m<sub>pos</sub>iti<sub>o</sub>n<sub>,</sub> i<sub>n w</sub>hi<sub>c</sub>h th<sub>ere are</sub> t<sub>oo many</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>agen</sub>t<sub>s</sub> f<sub>or one</sub> <sub>manager</sub> t<sub>o</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager.</sub>

• Think less about dividin<sub>g</sub> teams u<sub>p</sub> b<sub>y</sub> t<sub>yp</sub>e or function<sub>,</sub> but rather b<sub>y</sub> what <sub>g</sub>rou<sub>p</sub>s are <sub>necessary</sub> t<sub>o</sub> <sub>wor</sub>k t<sub>oge</sub>th<sub>er</sub> t<sub>o</sub> <sub>ac</sub>hi<sub>eve</sub> <sub>su</sub>b<sub>goa</sub>l<sub>s.</sub>

• Kee<sub>p</sub> teams as sim<sub>p</sub>le as <sub>p</sub>ossible.

R<sub>espon</sub>d <sub>w</sub>ith <sub>on</sub>l<sub>y</sub> th<sub>e</sub> JSON <sub>s</sub>t<sub>ruc</sub>t<sub>ure,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> <sub>exp</sub>l<sub>ana</sub>ti<sub>on</sub> fi<sub>e</sub>ld<sub>.</sub> D<sub>o</sub> <sub>no</sub>t i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>any</sub> <sub>a</sub>dditi<sub>ona</sub>l t<sub>ex</sub>t<sub>.</sub>

## Pr<sub>o</sub>m<sub>p</sub>t S3<sub>:</sub>In<sub>va</sub>lid A<sub>ge</sub>nt C<sub>ou</sub>nt Criti<sub>que</sub> Pr<sub>o</sub>m<sub>p</sub>t

System Message:

Y<sub>ou</sub> <sub>are</sub> <sub>an</sub> <sub>exper</sub>t <sub>cr</sub>iti<sub>c</sub> <sub>o</sub>f <sub>mu</sub>lti<sub>-agen</sub>t t<sub>eam</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ures.</sub> Y<sub>ou</sub> <sub>prov</sub>id<sub>e</sub> h<sub>ones</sub>t<sub>,</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>ve</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>on</sub> t<sub>eam</sub> hi<sub>erarc</sub>hi<sub>es.</sub>

User Message:

Y<sub>ou</sub> <sub>are</sub> <sub>an</sub> <sub>exper</sub>t <sub>cr</sub>iti<sub>c</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> t<sub>eam</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> f<sub>or</sub> <sub>mu</sub>lti<sub>-agen</sub>t <sub>w</sub>ildfi<sub>re</sub> <sub>response</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ons.</sub>

MISSION: {mission description}

AVAILABLE AGENTS: {worker summary str}

• Firefi<sub>g</sub>hters: {firefi<sub>g</sub>hter count}

• Bulldozers: {bulldozer count}

• Drones: {drone count}

• Helico<sub>p</sub>ters: {helico<sub>p</sub>ter count}

## PROPOSED TEAM STRUCTURE<sub>:</sub>

{structure\_json}

## CRITICAL ERROR <sub>–</sub> INVALID AGENT COUNTS<sub>:</sub>

[TOO MANY {agent type}s: assigned {actual}, only have {expected}] or [NOT ENOUGH {agent type}s: assigned {actual}, need to assign all {expected}]

Th<sub>e</sub> <sub>propose</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> h<sub>as</sub> INCORRECT <sub>agen</sub>t <sub>coun</sub>t<sub>s.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>a</sub> CRITICAL ERROR th<sub>a</sub>t must be rejected.

YOUR EVALUATION<sub>:</sub>

• Set approved: false because the a<sub>g</sub>ent counts are invalid

• In <sub>y</sub>our explanation<sub>,</sub> clearl<sub>y</sub> state that the a<sub>g</sub>ent counts are wron<sub>g</sub> and s<sub>p</sub>ecif<sub>y</sub> the errors

<table><tr><td>System Message:</td></tr><tr><td>You are an expert critic of multi-agent team coordination structures. You provide honest, constructive feedback on team hierarchies.</td></tr><tr><td>User Message:</td></tr><tr><td>You are an expert critic evaluating team structures for multi-agent missions. MISSION: {mission_description}</td></tr><tr><td>IMPORTANT: Only expect what the mission demands. Do not expect possible situations</td></tr><tr><td>that are not mentioned; the mission is fully comprehensive of the task to complete.</td></tr><tr><td>AVAILABLE AGENTS: {worker_summary_str}</td></tr><tr><td>[Agent counts have been validated and are correct.]</td></tr><tr><td>PROPOSED TEAM STRUCTURE:</td></tr><tr><td>{structure_json}</td></tr><tr><td>AGENT CAPABILITIES:</td></tr><tr><td>• Firefighter: Can cut trees (slow), pick up/drop off civilians, spray water to extinguish</td></tr><tr><td>fires, refill water</td></tr><tr><td>• Bulldozer: Can cut trees (fast), cannot interact with fire or civilians</td></tr><tr><td>• Drone: Can scout/observe areas, cannot manipulate environment</td></tr><tr><td>• Helicopter: Can transport (up to 5) firefighters to different locations, can refill and deploy water on fire.</td></tr></table>

• In <sub>y</sub>our suggestions<sub>,</sub> <sub>p</sub>rovide s<sub>p</sub>ecific <sub>g</sub>uidance on how to fix the a<sub>g</sub>ent count issues

You MUST reject this structure due to invalid agent counts.

## Pr<sub>o</sub>m<sub>p</sub>t S4<sub>:</sub>V<sub>a</sub>lid A<sub>ge</sub>nt C<sub>ou</sub>nt T<sub>ea</sub>m Str<sub>uc</sub>t<sub>u</sub>r<sub>e</sub> Criti<sub>que</sub> Pr<sub>o</sub>m<sub>p</sub>t

## MANAGER TYPES<sub>:</sub>

## 1. Vertical Mana<sub>g</sub>er (Temporal Coordination)

• Use when this mana<sub>g</sub>er’s direct children re<sub>q</sub>uire se<sub>q</sub>uential or real-time coordinat<sup>i</sup>on, suc<sup>h</sup> as trans<sub>p</sub>ort<sup>i</sup>n<sub>g</sub> a<sub>g</sub>ents.

• Children work throu<sub>g</sub>h <sub>p</sub>hases to<sub>g</sub>ether in lockste<sub>p</sub>.

• Question to ask: “Do my direct children need to wait for each other?”

## 2. Horizontal Mana<sub>g</sub>er (Parallel Coordination)

• Use when this mana<sub>g</sub>er’s direct children work on inde<sub>p</sub>endent subtasks.

• Children execute simultaneousl<sub>y</sub> without interde<sub>p</sub>endencies.

• Question to ask: “Can my direct children work independently?”

Y<sub>our</sub> t<sub>as</sub>k i<sub>s</sub> t<sub>o cr</sub>iti<sub>ca</sub>ll<sub>y eva</sub>l<sub>ua</sub>t<sub>e</sub> thi<sub>s</sub> t<sub>eam s</sub>t<sub>ruc</sub>t<sub>ure spec</sub>ifi<sub>ca</sub>ll<sub>y</sub> f<sub>or</sub> th<sub>ese po</sub>i<sub>n</sub>t<sub>s:</sub>

## EVALUATION CRITERIA<sub>:</sub>

1<sub>.</sub> Th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> i<sub>s</sub> <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l<sub>:</sub> Th<sub>ere</sub> <sub>are</sub> <sub>no</sub> <sub>unnecessary</sub> <sub>managers,</sub> <sub>an</sub>d <sub>a</sub>ll f<sub>u</sub>ll <sub>managers</sub> <sub>canno</sub>t b<sub>e</sub> <sub>rep</sub>l<sub>ace</sub>d b<sub>y</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>managers.</sub>

2<sub>.</sub> Th<sub>ere</sub> <sub>are</sub> <sub>no</sub> <sub>managers</sub> <sub>w</sub>ith <sub>on</sub>l<sub>y</sub> 1 <sub>wor</sub>k<sub>ers</sub>/<sub>su</sub>bt<sub>eams.</sub>

3<sub>.</sub> All <sub>agen</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> <sub>same</sub> t<sub>ype</sub> <sub>an</sub>d i<sub>n</sub> th<sub>e</sub> <sub>same</sub> di<sub>rec</sub>t t<sub>eam</sub> <sub>are</sub> <sub>groupe</sub>d t<sub>oge</sub>th<sub>er</sub> <sub>as</sub> <sub>coun</sub>t<sub>:</sub> N<sub>.</sub>

4<sub>.</sub> W<sub>or</sub>k<sub>er agen</sub>t<sub>s can p</sub>l<sub>an mu</sub>lti<sub>p</sub>l<sub>e s</sub>t<sub>eps</sub>/<sub>ac</sub>ti<sub>ons</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y, so</sub> th<sub>ey s</sub>h<sub>ou</sub>ld <sub>no</sub>t b<sub>e</sub> <sub>m</sub>i<sub>cromanage</sub>d t<sub>oo</sub> <sub>muc</sub>h<sub>.</sub>

5<sub>.</sub> Th<sub>e</sub> <sub>p</sub>l<sub>an</sub> i<sub>sn</sub>’t b<sub>u</sub>ilt <sub>aroun</sub>d <sub>poss</sub>ibiliti<sub>es</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> <sub>o</sub>f <sub>w</sub>h<sub>a</sub>t i<sub>s</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on.</sub>

6<sub>.</sub> Th<sub>e</sub> <sub>ru</sub>l<sub>e</sub> <sub>o</sub>f th<sub>um</sub>b i<sub>s</sub> b<sub>e</sub>i<sub>ng</sub> f<sub>o</sub>ll<sub>owe</sub>d<sub>:</sub>

## RULE OF THUMB

R<sub>u</sub>l<sub>e</sub> <sub>o</sub>f th<sub>um</sub>b<sub>:</sub>

• Start with the sim<sub>p</sub>lest team <sub>p</sub>ossible with just the root mana<sub>g</sub>er

• Then add subteams and com<sub>p</sub>lexit<sub>y</sub> from there.

• Onl<sub>y</sub> make subteams when there is a critical need for:

<sub>–</sub> S<sub>y</sub>n<sub>c</sub>hr<sub>o</sub>ni<sub>ze</sub>d t<sub>as</sub>k d<sub>eco</sub>m<sub>pos</sub>iti<sub>o</sub>n<sub>,</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h t<sub>eam</sub> <sub>mem</sub>b<sub>ers</sub> <sub>mus</sub>t <sub>wor</sub>k th<sub>roug</sub>h <sub>p</sub>h<sub>ases</sub> t<sub>oge</sub>th<sub>er</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>manager.</sub>

<sub>–</sub> S<sub>ca</sub>lin<sub>g</sub> d<sub>eco</sub>m<sub>pos</sub>iti<sub>o</sub>n<sub>,</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>ere</sub> <sub>are</sub> t<sub>oo</sub> <sub>many</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>agen</sub>t<sub>s</sub> f<sub>or</sub> <sub>one</sub> <sub>manager</sub> t<sub>o</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager.</sub>

R<sub>e</sub>t<sub>urn an overa</sub>ll <sub>consensus, a s</sub>h<sub>or</sub>t <sub>exp</sub>l<sub>ana</sub>ti<sub>on</sub> i<sub>n one sen</sub>t<sub>ence</sub> f<sub>or</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on, an</sub>d <sub>a s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>s</sub>h<sub>or</sub>t <sub>sugges</sub>ti<sub>on</sub> f<sub>or</sub> i<sub>mprovemen</sub>t<sub>.</sub>

## Prompt S5:Mana<sub>g</sub>er Prompt (All the word ”phase” will be replaced b<sub>y</sub> ”mission” for h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>manager.</sub>

## System Message:

You are {AGENT ID}, the Team Manager of {self.team name}, part of a broader team of embodied agents within a grid world. The map is made up of {map size} by {map size} cells/grids with coordinates in the range of [0 to {map size-1}, 0 to map size-1}] with the to<sub>p</sub> left corner of the ma<sub>p</sub> bein<sub>g</sub> (0,0).

## T<sub>erm</sub>i<sub>no</sub>l<sub>ogy:</sub>

• Mission: The overall <sub>g</sub>oal for <sub>y</sub>our team (e.<sub>g</sub>.<sub>,</sub> “Locate and su<sub>pp</sub>ress the fire”)

• Phase: A major ste<sub>p</sub> toward the mission (e.<sub>g</sub>.<sub>,</sub> “Scout for the fire”<sub>,</sub> “Build firebreaks”)

• Task: The s<sub>p</sub>ecific assi<sub>g</sub>nment for each a<sub>g</sub>ent in <sub>y</sub>our team for the current <sub>p</sub>hase

• U<sub>pp</sub>er team: The team mana<sub>g</sub>ed b<sub>y</sub> <sub>y</sub>our mana<sub>g</sub>er (<sub>y</sub>our <sub>p</sub>arent in the hierarch<sub>y</sub>)

Your team’s mission: {mission}   
Your team’s current phase: {current phase}   
Your team’s phase progress: {phase progress}%   
Your upper team’s phase: {upperteam phase}   
Your upper team’s mission: {upperteam mission}   
Y<sub>our</sub> <sub>ro</sub>l<sub>e</sub> i<sub>s</sub> t<sub>o:</sub>   
1<sub>.</sub> C<sub>o</sub>ll<sub>ec</sub>t <sub>an</sub>d <sub>summar</sub>i<sub>ze o</sub>b<sub>serva</sub>ti<sub>ons</sub> f<sub>rom your su</sub>bt<sub>eam</sub>   
2<sub>.</sub> A<sub>ssess</sub> <sub>progress</sub> <sub>on</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> <sub>an</sub>d <sub>overa</sub>ll <sub>m</sub>i<sub>ss</sub>i<sub>on</sub>   
3. Make decisions about phase transitions and task adjustments   
Al<sub>ways respon</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e requ</sub>i<sub>re</sub>d t<sub>ag s</sub>t<sub>ruc</sub>t<sub>ure an</sub>d <sub>examp</sub>l<sub>e</sub> f<sub>orma</sub>t<sub>.</sub>   
CRITICAL FORMATTING RULE<sub>:</sub> E<sub>very response</sub> MUST <sub>use</sub> th<sub>e exac</sub>t XML t<sub>ags</sub>   
re<sub>qu</sub>ested<sub>.</sub> Ne<sub>v</sub>er omit or rename a ta<sub>g.</sub> Al<sub>w</sub>a<sub>y</sub>s start <sub>y</sub>o<sub>u</sub>r res<sub>p</sub>onse <sub>w</sub>ith the o<sub>p</sub>enin<sub>g</sub> ta<sub>g.</sub>   
User Message:   
Perception:   
Given <sub>y</sub>our team’s observations, summarize <sub>y</sub>our team’s collective <sub>p</sub>erce<sub>p</sub>tion in ≤ 50 words.   
{team context}   
Y<sub>ou</sub> MUST <sub>respon</sub>d <sub>us</sub>i<sub>ng</sub> ONLY th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> XML t<sub>ags—no</sub> <sub>prose,</sub> <sub>no</sub> <sub>pream</sub>bl<sub>e,</sub> <sub>an</sub>d   
<sub>no</sub> t<sub>ex</sub>t <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> t<sub>ags.</sub> D<sub>o</sub> <sub>no</sub>t <sub>re</sub>f<sub>erence</sub> <sub>any</sub> <sub>agen</sub>t<sub>s</sub> b<sub>y</sub> <sub>name,</sub> b<sub>u</sub>t <sub>ra</sub>th<sub>er</sub> <sub>re</sub>f<sub>er</sub> t<sub>o</sub> th<sub>em</sub>   
<sub>co</sub>ll<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>as</sub> “<sub>my</sub> t<sub>eam</sub>”<sub>:</sub>   
<<sub>percep</sub>ti<sub>on</sub>>A <sub>conc</sub>i<sub>se</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>summary o</sub>f <sub>your</sub> t<sub>eam</sub>’<sub>s co</sub>ll<sub>ec</sub>ti<sub>ve o</sub>b<sub>serva</sub>ti<sub>ons.</sub> I<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>   
<sub>w</sub>h<sub>a</sub>t <sub>your</sub> t<sub>eam</sub> h<sub>as</sub> di<sub>scovere</sub>d<sub>,</sub> th<sub>e</sub>i<sub>r</sub> <sub>overa</sub>ll <sub>pos</sub>iti<sub>ons,</sub> <sub>an</sub>d <sub>any</sub> i<sub>mpor</sub>t<sub>an</sub>t fi<sub>n</sub>di<sub>ngs.</sub> D<sub>o</sub> <sub>no</sub>t   
<sup>refer to an</sup>y <sup>a</sup>g<sup>ents b</sup>y <sup>name</sup>. <sup></</sup>p<sup>erce</sup>p<sup>tion></sup>   
DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET<sub>.</sub> O<sub>n</sub>l<sub>y</sub> <sub>prov</sub>id<sub>e</sub>   
<sup>a</sup> p<sup>erce</sup>p<sup>tion</sup> <sup>s</sup>u<sup>mmar</sup>y.

<table><tr><td>Your response MUST begin with &lt;perception&gt; and end with &lt;/perception&gt;. Do not</td></tr><tr><td>write anything before &lt;perception&gt; or after &lt;/perception&gt;.</td></tr><tr><td>Phase Planning (only for vertical manager):</td></tr><tr><td>Your mission: &#x27;{mission}&#x27;</td></tr><tr><td>Upper team phase: {upperteam_phase}</td></tr><tr><td>Upper team mission: {upperteam_mission}</td></tr><tr><td>{feedback_section}</td></tr><tr><td>First, break your mission into phases using the following tags:</td></tr><tr><td>&lt;phases&gt;</td></tr><tr><td>List each phase on a separate line, numbered or with bullet points.</td></tr><tr><td>&lt;/phases&gt;</td></tr><tr><td>Example response:</td></tr><tr><td>&lt;phases&gt;</td></tr><tr><td>1. Scout for the fire</td></tr><tr><td>2. Move to the fire and suppress it</td></tr><tr><td>3. Check if the fire is fully contained</td></tr><tr><td>&lt;/phases&gt;</td></tr><tr><td>Phases should be concise, simple, and built directly on the exact abilities of the team.</td></tr><tr><td>You may only need one/a few phases. Do not include unnecessary phases, or phases that are</td></tr><tr><td>outside of agent capabilities. However, look at specifically how many agents you have and what needs to be done for the mission to be fully complete. You may have to repeat groups</td></tr><tr><td>of phases to complete the whole mission.</td></tr><tr><td>DO NOT BREAK DOWN ANY PHASES INTO TASKS YET. KEEP THEM CON-</td></tr><tr><td>CISE.</td></tr><tr><td>Additional Phase (only for vertical manager):</td></tr><tr><td>Now, generate additional phase(s) to continue the mission. Phases should be concise, simple,</td></tr><tr><td>and built directly on the exact abilities of the team.</td></tr></table>

## Y<sub>ou</sub>r t<sub>ea</sub>m <sub>co</sub>m<sub>pos</sub>iti<sub>o</sub>n <sub>a</sub>nd t<sub>ea</sub>m <sub>a</sub>biliti<sub>es:</sub>

O<sub>n</sub>l<sub>y</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>necessary</sub> <sub>p</sub>h<sub>ases.</sub> DO NOT i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>o</sub>th<sub>er</sub> li<sub>nes</sub>/i<sub>n</sub>f<sub>o</sub> i<sub>n</sub> b<sub>e</sub>t<sub>ween;</sub> it <sub>s</sub>h<sub>ou</sub>ld b<sub>e</sub> exactl<sub>y</sub> n lines.

1<sub>.</sub> A <sub>s</sub>t<sub>a</sub>t<sub>us</sub> <sub>summary</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>ags:</sub> <summary> A concise summar<sub>y</sub> of <sub>y</sub>our team’s <sub>p</sub>ro<sub>g</sub>ress and situation, includin<sub>g</sub> what h<sub>as</sub> b<sub>een accomp</sub>li<sub>s</sub>h<sub>e</sub>d <sub>an</sub>d <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>us.</sub> D<sub>o no</sub>t <sub>c</sub>it<sub>e spec</sub>ifi<sub>c agen</sub>t<sub>s, on</sub>l<sub>y</sub> th<sub>e</sub> t<sub>eam</sub> as a whole. </summary> <phase percent complete> A number from 0 to 100 re<sub>p</sub>resentin<sub>g</sub> <sub>y</sub>our team’s estimated percent complete on the current phase {current phase}. </phase percent complete>

<mission percent complete> A number from 0 to 100 re<sub>p</sub>resentin<sub>g</sub>   
your team’s estimated percent complete on the current mission {mission}.   
</mission percent complete>   
<urgent> An<sub>y</sub> ur<sub>g</sub>ent information for <sub>y</sub>our u<sub>pp</sub>er team, such as unex<sub>p</sub>ected fires,   
civilians, or other issues. If none, write ”None”. </urgent>   
2<sub>.</sub> A d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>ag:</sub>   
<reasoning> Concise reasonin<sub>g</sub> for <sub>y</sub>our decision. </reasoning>   
<decision> One of: {decision options} </decision>   
{decision descriptions}   
<sup>E</sup>xam<sub>p</sub><sup>l</sup>e res<sub>p</sub>onse:   
<summary> {example summary} </summary>   
<phase percent complete> {example pct} </phase percent complete>   
<mission percent complete> 20 </mission percent complete>   
<urgent> None </urgent>   
<reasoning> The team has com<sub>p</sub>leted all <sub>p</sub>lanned <sub>p</sub>hases but the fire is not full<sub>y</sub> contained.   
Additional work is needed. </reasoning>   
<decision> {example decision} </decision>   
Plan Checking:   
You are a member of a team structure {team structure}, including yourself: {AGENT ID}.   
{context line}   
Y<sub>ou</sub>r <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>s:</sub>   
{perception summary}   
{self.knowledge base}   
Here is the proposed plan for the team by your Team Manager {plan label}:   
{plan}

![](images/417e551b9466307b8ece7d8c0da0b7a2d5f32bee405965c93c9f2db5eaf557fb.jpg)

1<sub>.</sub> P<sub>rov</sub>id<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>on</sub> <sub>propose</sub>d t<sub>eam</sub> <sub>p</sub>l<sub>ans.</sub>

2<sub>.</sub> A<sub>ssess</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>p</sub>l<sub>ans</sub> <sub>are</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>an</sub>d f<sub>eas</sub>ibl<sub>e</sub> f<sub>or</sub> <sub>your</sub> <sub>su</sub>bt<sub>eam</sub>’<sub>s</sub> <sub>capa</sub>biliti<sub>es.</sub>

Wh<sub>en</sub> <sub>prov</sub>idi<sub>ng</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>:</sub>

• Focus on <sub>y</sub>our subteam’s s<sub>p</sub>ecific role and ca<sub>p</sub>abilities.

• Consider <sub>y</sub>our subteam’s current <sub>p</sub>osition and observations.

• Assess whether the assi<sub>g</sub>ned tasks are a<sub>pp</sub>ro<sub>p</sub>riate and achievable.

• Provide constructive su<sub>gg</sub>estions for im<sub>p</sub>rovement.

• Res<sub>p</sub>ond with ’YES’ if the <sub>p</sub>lan is satisfactor<sub>y</sub> for <sub>y</sub>our subteam.

## Feedback Receiving:

{format feedback}

IMPORTANT<sub>:</sub> St<sub>an</sub>di<sub>ng</sub> <sub>or</sub>d<sub>ers</sub> ALWAYS t<sub>a</sub>k<sub>e</sub> <sub>pr</sub>i<sub>or</sub>it<sub>y.</sub> Y<sub>ou</sub> <sub>mus</sub>t NOT <sub>overr</sub>id<sub>e</sub> <sub>or</sub> <sub>con</sub>t<sub>ra-</sub> di<sub>c</sub>t<sub>.</sub> If <sub>your</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>con</sub>fli<sub>c</sub>t<sub>s</sub> <sub>w</sub>ith <sub>any</sub> <sub>s</sub>t<sub>an</sub>di<sub>ng</sub> <sub>or</sub>d<sub>er</sub> <sub>or</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>,</sub> <sub>you</sub> MUST CANCEL<sub>.</sub>

Pl<sub>ease</sub> <sub>con</sub>fi<sub>rm</sub> <sub>your</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>ag:</sub>

CONFIRM: Proceed with the decision to {decision}

CANCEL<sub>:</sub> C<sub>on</sub>ti<sub>nue</sub> <sub>w</sub>ith th<sub>e</sub> <sub>curren</sub>t <sub>p</sub>h<sub>ase</sub> i<sub>ns</sub>t<sub>ea</sub>d

## <sup>E</sup>xam<sub>p</sub><sup>l</sup>e res<sub>p</sub>onse:

<reasoning>

Th<sub>e</sub> t<sub>eam</sub> i<sub>s near</sub>l<sub>y comp</sub>l<sub>e</sub>t<sub>e w</sub>ith th<sub>e curren</sub>t <sub>m</sub>i<sub>ss</sub>i<sub>on,</sub> b<sub>u</sub>t <sub>a</sub> f<sub>ew agen</sub>t<sub>s nee</sub>d t<sub>o</sub> fi<sub>n</sub>i<sub>s</sub>h th<sub>e</sub>i<sub>r</sub>   
l<sub>as</sub>t <sub>ac</sub>ti<sub>ons.</sub>   
</reasoning>   
<confirmation>   
CANCEL   
</confirmation>   
G<sub>rea</sub>t<sub>, now</sub> l<sub>e</sub>t’<sub>s move</sub> t<sub>o</sub> th<sub>e nex</sub>t <sub>p</sub>h<sub>ase.</sub>   
Timeline: Past Phases: {phase history}, New Current Phase: ’{current phase}’, Future   
Phases: {future phases}   
{feedback section}   
{announcement section}

## Y<sub>ou</sub>r t<sub>ea</sub>m <sub>co</sub>m<sub>pos</sub>iti<sub>o</sub>n <sub>a</sub>nd t<sub>ea</sub>m <sub>a</sub>biliti<sub>es:</sub>

B<sub>rea</sub>k d<sub>own</sub> th<sub>e</sub> <sub>nex</sub>t <sub>p</sub>h<sub>ase</sub> i<sub>n</sub>t<sub>o</sub> t<sub>as</sub>k<sub>s</sub> f<sub>or</sub> <sub>eac</sub>h t<sub>eam</sub> <sub>mem</sub>b<sub>er</sub> <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> <sub>a</sub> <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>con</sub>diti<sub>on</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>ags:</sub>

<completion condition>   
A <sub>s</sub>h<sub>or</sub>t<sub>, spec</sub>ifi<sub>c con</sub>diti<sub>on</sub> th<sub>a</sub>t d<sub>e</sub>fi<sub>nes w</sub>h<sub>en</sub> thi<sub>s p</sub>h<sub>ase</sub> i<sub>s comp</sub>l<sub>e</sub>t<sub>e.</sub> B<sub>e concre</sub>t<sub>e an</sub>d <sub>mea-</sub>   
<sub>sura</sub>bl<sub>e.</sub>   
</completion condition>

K<sub>eep</sub> <sub>eac</sub>h t<sub>as</sub>k <sub>conc</sub>i<sub>se—one</sub> <sub>or</sub> t<sub>wo</sub> <sub>s</sub>h<sub>or</sub>t <sub>sen</sub>t<sub>ences.</sub> B<sub>e</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>w</sub>ith <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> b<sub>u</sub>t <sub>s</sub>ki<sub>p</sub> <sub>unnecessary</sub> d<sub>e</sub>t<sub>a</sub>il<sub>,</sub> <sub>suc</sub>h <sub>as</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on,</sub> <sub>s</sub>t<sub>ay</sub>i<sub>ng</sub> <sub>a</sub>l<sub>er</sub>t<sub>,</sub> <sub>repor</sub>ti<sub>ng</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> <sub>e</sub>t<sub>c.</sub> H<sub>owever,</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub>ll t<sub>as</sub>k<sub>s</sub> SHOULD <sub>resu</sub>lt i<sub>n</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>p</sub>h<sub>ase,</sub> <sub>so</sub> <sub>ma</sub>k<sub>e</sub> th<sub>em</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e.</sub>

## ONLY ASSIGN ONE TASK PER AGENT<sub>.</sub>

## Y<sub>ou</sub>r t<sub>ea</sub>m <sub>co</sub>m<sub>pos</sub>iti<sub>o</sub>n <sub>a</sub>nd t<sub>ea</sub>m <sub>a</sub>biliti<sub>es:</sub>

E<sub>xp</sub>l<sub>a</sub>i<sub>n</sub> <sub>w</sub>h<sub>a</sub>t <sub>c</sub>h<sub>ange</sub>d <sub>an</sub>d <sub>w</sub>h<sub>y</sub> <sub>a</sub> <sub>new</sub> <sub>p</sub>l<sub>an</sub> f<sub>or</sub> thi<sub>s</sub> <sub>p</sub>h<sub>ase</sub> i<sub>s</sub> <sub>necessary.</sub>

{agent tags}

<completion condition>

A <sub>s</sub>h<sub>or</sub>t<sub>, spec</sub>ifi<sub>c con</sub>diti<sub>on</sub> th<sub>a</sub>t d<sub>e</sub>fi<sub>nes w</sub>h<sub>en</sub> th<sub>e curren</sub>t <sub>p</sub>h<sub>ase</sub> i<sub>s comp</sub>l<sub>e</sub>t<sub>e.</sub> B<sub>e concre</sub>t<sub>e an</sub>d <sub>measura</sub>bl<sub>e.</sub>

</completion condition>

K<sub>eep eac</sub>h t<sub>as</sub>k <sub>conc</sub>i<sub>se—one or</sub> t<sub>wo s</sub>h<sub>or</sub>t <sub>sen</sub>t<sub>ences.</sub> B<sub>e spec</sub>ifi<sub>c w</sub>ith <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> b<sub>u</sub>t <sub>s</sub>ki<sub>p</sub> <sub>unnecessary</sub> d<sub>e</sub>t<sub>a</sub>il<sub>,</sub> <sub>suc</sub>h <sub>as</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>on,</sub> <sub>s</sub>t<sub>ay</sub>i<sub>ng</sub> <sub>a</sub>l<sub>er</sub>t<sub>,</sub> <sub>repor</sub>ti<sub>ng</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> <sub>e</sub>t<sub>c.</sub> H<sub>owever,</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub>ll t<sub>as</sub>k<sub>s</sub> SHOULD <sub>resu</sub>lt i<sub>n</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>p</sub>h<sub>ase,</sub> <sub>so</sub> <sub>ma</sub>k<sub>e</sub> th<sub>em</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e.</sub>

## <sup>E</sup>xam<sub>p</sub><sup>l</sup>e res<sub>p</sub>onse:

<explanation>

Fi<sub>re</sub> <sub>spo</sub>tt<sub>e</sub>d i<sub>n</sub> <sub>nor</sub>th<sub>eas</sub>t<sub>.</sub> R<sub>eass</sub>i<sub>gn</sub>i<sub>ng</sub> t<sub>as</sub>k<sub>s</sub> t<sub>o</sub> <sub>respon</sub>d<sub>.</sub>

</explanation>

<AGENT 1>

Scout northeast re<sub>g</sub>ion around (45,12) for fire outbreaks.

</AGENT 1>

<AGENT 2>

E<sub>x</sub>ti<sub>ngu</sub>i<sub>s</sub>h fi<sub>re</sub> <sub>a</sub>t <sub>sou</sub>th<sub>ern</sub> b<sub>or</sub>d<sub>er</sub> <sub>a</sub>l<sub>ong</sub> <sub>y=</sub>400<sub>.</sub>

</AGENT 2>

Cut all trees at (30,15), (31,15), (32,15).

</AGENT 3>

<completion condition>

N<sub>or</sub>th<sub>eas</sub>t <sub>scou</sub>t<sub>e</sub>d<sub>,</sub> <sub>sou</sub>th<sub>ern</sub> fi<sub>re</sub> <sub>ex</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>e</sub>d<sub>.</sub>

</completion condition>

ONLY ASSIGN ONE TASK PER AGENT<sub>.</sub>

Prom<sub>p</sub>t S6:Worker A<sub>g</sub>ent Prom<sub>p</sub>t   
System Message:   
You are {AGENT ID}, a {Agent type} agent within a forest grid world.   
T<sub>erm</sub>i<sub>no</sub>l<sub>ogy:</sub>   
• Mission: The overall <sub>g</sub>oal for <sub>y</sub>our team (e.<sub>g</sub>.<sub>,</sub> “Su<sub>pp</sub>ress the fire in sector 7”)   
• Phase: A major ste<sub>p</sub> toward the mission (e.<sub>g</sub>.<sub>,</sub> “Scout for the fire”<sub>,</sub> “Build firebreaks”)   
• Task: Your s<sub>p</sub>ecific assi<sub>g</sub>nment for the current <sub>p</sub>hase (e.<sub>g</sub>.<sub>,</sub> “Go to coordinates (5<sub>,</sub>10)   
and scout for fire”)   
• U<sub>pp</sub>er team: The team mana<sub>g</sub>ed b<sub>y</sub> <sub>y</sub>our mana<sub>g</sub>er (<sub>y</sub>our <sub>p</sub>arent in the hierarch<sub>y</sub>)   
Example: Your current task is “Go to coordinates (5,10) and scout for fire”. Your team’s   
<sub>p</sub>h<sub>ase</sub> i<sub>s</sub> “C<sub>oor</sub>di<sub>na</sub>t<sub>e</sub> <sub>reg</sub>i<sub>ona</sub>l <sub>suppress</sub>i<sub>on</sub>”<sub>.</sub> Y<sub>our</sub> t<sub>eam</sub>’<sub>s</sub> <sub>m</sub>i<sub>ss</sub>i<sub>on</sub> i<sub>s</sub> “S<sub>uppress</sub> th<sub>e</sub> fi<sub>re</sub> i<sub>n</sub>   
<sub>sec</sub>t<sub>or</sub> 7”<sub>.</sub>   
Your task: {mission}   
Your team’s phase: {upperteam phase}   
Your team’s mission: {upperteam mission}   
Your job is to analyze observations and provide status updates with structured output using   
XML t<sub>ags.</sub>   
CRITICAL FORMATTING RULE<sub>:</sub> E<sub>very</sub> <sub>response</sub> MUST <sub>use</sub> th<sub>e</sub> <sub>exac</sub>t XML t<sub>ags</sub>   
re<sub>qu</sub>ested<sub>.</sub> Ne<sub>v</sub>er omit or rename a ta<sub>g.</sub> Al<sub>w</sub>a<sub>y</sub>s start <sub>y</sub>o<sub>u</sub>r res<sub>p</sub>onse <sub>w</sub>ith the o<sub>p</sub>enin<sub>g</sub> ta<sub>g.</sub>   
Perception:   
Here are your observations: {observation string}   
Create a concise, detailed <sub>p</sub>erce<sub>p</sub>tion summar<sub>y</sub> (≤50 words). The KEY FEATURES section   
i<sub>s au</sub>th<sub>or</sub>it<sub>a</sub>ti<sub>ve—use</sub> it <sub>as your pr</sub>i<sub>mary source</sub> f<sub>or</sub> fi<sub>res, c</sub>i<sub>v</sub>ili<sub>ans, an</sub>d <sub>wa</sub>t<sub>er</sub> l<sub>oca</sub>ti<sub>ons.</sub> Y<sub>ou</sub>   
MUST <sub>respon</sub>d <sub>us</sub>i<sub>ng</sub> ONLY th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> XML t<sub>ags—no</sub> <sub>prose,</sub> <sub>no</sub> <sub>pream</sub>bl<sub>e,</sub> <sub>an</sub>d <sub>no</sub> t<sub>ex</sub>t   
<sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> t<sub>ags:</sub>

<table><tr><td>&lt;perception&gt;</td></tr><tr><td>A detailed summary of what you observe, including your location, surroundings, any fires,</td></tr><tr><td>and civilians from the KEY FEATURES section. Note other agents in your vicinity, and</td></tr><tr><td>your current status (carrying capacity, etc.). Focus on information relevant to your current</td></tr><tr><td>task.</td></tr><tr><td>&lt;/perception&gt;</td></tr><tr><td>DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. Only provide</td></tr><tr><td>a perception summary.</td></tr><tr><td>Your response MUST begin with &lt;perception&gt; and end with &lt;/perception&gt;. Do not</td></tr><tr><td>write anything before &lt;perception&gt; or after &lt;/perception&gt;.</td></tr><tr><td>Example response:</td></tr><tr><td>&lt;perception&gt;</td></tr><tr><td>I am currently at coordinates (5,10) in a medium forest cell. I can see dense forest to the north,</td></tr><tr><td>a water source to the east at (7,10), and no signs of fire. There are no civilians nearby. I am</td></tr><tr><td>not carrying any civilians and have 3/5 water remaining. I can see AGENT_2 at coordinates (6,9) to the northeast.</td></tr><tr><td>&lt;/perception&gt;</td></tr><tr><td>Summary:</td></tr><tr><td>Now, given your current task: &#x27;{mission}</td></tr><tr><td>Your timeline: Past/Completed actions: {past_action}, Current/Ongoing actions:</td></tr><tr><td>{ongoing_action}, Planned future actions: {future_action}</td></tr><tr><td>Provide your status summary using the following tags:</td></tr><tr><td>&lt;summary&gt;</td></tr><tr><td>A concise summary (≤30 words) of your progress and situation, including what you&#x27;ve</td></tr><tr><td>accomplished and what you’re currently doing</td></tr><tr><td>&lt;/summary&gt;</td></tr><tr><td>&lt;percent_complete&gt;</td></tr></table>

A <sub>num</sub>b<sub>er</sub> f<sub>rom</sub> 0 t<sub>o</sub> 100 <sub>represen</sub>ti<sub>ng</sub> <sub>your</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>percen</sub>t <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>on</sub> <sub>your</sub>

current task ({mission}).

</percent complete>

<urgent>

A<sub>ny</sub> i<sub>mpor</sub>t<sub>an</sub>t i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> f<sub>or</sub> <sub>your</sub> <sub>manager,</sub> <sub>suc</sub>h <sub>as</sub> <sub>unexpec</sub>t<sub>e</sub>d fi<sub>res,</sub> <sub>c</sub>i<sub>v</sub>ili<sub>ans,</sub> <sub>or</sub> i<sub>ssues.</sub> If <sub>none, wr</sub>it<sub>e</sub> “N<sub>one</sub>”<sub>.</sub>

</urgent>

## <sup>E</sup>xam<sub>p</sub><sup>l</sup>e res<sub>p</sub>onse:

<summary>

I have moved to coordinates (5,10) and com<sub>p</sub>leted the scoutin<sub>g</sub> of the northern area. I found <sub>no</sub> fi<sub>re</sub> i<sub>n</sub> thi<sub>s</sub> <sub>sec</sub>t<sub>or.</sub>

</summary>

<percent complete>

75

</percent complete>

<urgent>

I have located the missin<sub>g g</sub>rou<sub>p</sub> of civilians at coordinates (5,10). I su<sub>gg</sub>est we move to the <sub>nex</sub>t <sub>p</sub>h<sub>ase</sub> <sub>o</sub>f t<sub>ranspor</sub>ti<sub>ng</sub> th<sub>em.</sub>

</urgent>

## Action Planning:

Y<sub>ou</sub> <sub>are</sub> <sub>a</sub> b<sub>u</sub>lld<sub>ozer</sub> fi<sub>re</sub>fi<sub>g</sub>ht<sub>er</sub> <sub>agen</sub>t <sub>w</sub>ithi<sub>n</sub> <sub>a</sub> f<sub>ores</sub>t <sub>ce</sub>ll <sub>gr</sub>id <sub>wor</sub>ld<sub>.</sub> Th<sub>e</sub> <sub>map</sub> i<sub>s</sub> <sub>ma</sub>d<sub>e</sub> <sub>up</sub> <sub>o</sub>f {map size} by {map size} cells/grids with coordinates in the range of (0 to {map size-1}, 0 to {map size-1}) inclusive. The origin of the map is at the northwest corner, so � increases <sub>eas</sub>t<sub>war</sub>d <sub>an</sub>d <sub>�</sub> i<sub>ncreases</sub> <sub>sou</sub>th<sub>war</sub>d<sub>.</sub>

E<sub>ac</sub>h <sub>ce</sub>ll i<sub>s</sub> <sub>one</sub> <sub>o</sub>f th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng:</sub>

• Brush (0 trees)

• Li<sub>g</sub>ht forest (1 tree)

• Medium forest (2 trees)

• Dense forest (3 trees)

• I<sub>g</sub>nited

• On fire

• Extin<sub>g</sub>uishin<sub>g</sub>

• Full<sub>y</sub> Extin<sub>g</sub>uished

Your current location is {position} and your current cell is {current cell}. Here are your direct observations: {observation string}

If the agent is bulldozer:

Th<sub>ere</sub> <sub>are</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> th<sub>ree</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>you</sub> <sub>can</sub> <sub>per</sub>f<sub>orm.</sub> N<sub>o</sub>thi<sub>ng</sub> <sub>e</sub>l<sub>se</sub> i<sub>s</sub> <sub>a</sub>ll<sub>owe</sub>d<sub>:</sub>

• Move to an<sub>y</sub> coordinate location not cuttin<sub>g</sub> an<sub>y</sub> trees on the wa<sub>y</sub> there. Distance does <sub>no</sub>t <sub>ma</sub>tt<sub>er;</sub> <sub>you</sub> <sub>w</sub>ill <sub>reac</sub>h th<sub>a</sub>t l<sub>oca</sub>ti<sub>on</sub> i<sub>n</sub> <sub>one</sub> <sub>move.</sub> Y<sub>ou</sub> <sub>on</sub>l<sub>y</sub> <sub>nee</sub>d <sub>one</sub> <sub>move</sub> <sub>ac</sub>ti<sub>on</sub> to move an<sub>y</sub>where (so do not <sub>p</sub>lan in-between destinations, onl<sub>y</sub> final destinations).

• Move to an<sub>y</sub> coordinate location cuttin<sub>g</sub> all trees on the wa<sub>y</sub> there<sub>,</sub> creatin<sub>g</sub> a line <sub>o</sub>f <sub>c</sub>l<sub>eare</sub>d t<sub>rees</sub> f<sub>rom your curren</sub>t l<sub>oca</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> l<sub>oca</sub>ti<sub>on.</sub> Di<sub>s</sub>t<sub>ance</sub> d<sub>oes</sub> <sub>no</sub>t <sub>ma</sub>tt<sub>er;</sub> <sub>you</sub> <sub>w</sub>ill <sub>reac</sub>h th<sub>a</sub>t l<sub>oca</sub>ti<sub>on</sub> i<sub>n</sub> <sub>one</sub> <sub>move.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>s</sub>l<sub>ower</sub> <sub>an</sub>d <sub>more</sub> <sub>cos</sub>tl<sub>y</sub> th<sub>an</sub> <sub>no</sub>t <sub>cu</sub>tti<sub>ng,</sub> <sub>so</sub> <sub>on</sub>l<sub>y</sub> d<sub>o</sub> thi<sub>s</sub> <sub>w</sub>h<sub>en</sub> <sub>you</sub> NEED t<sub>o</sub> <sub>crea</sub>t<sub>e</sub> <sub>a</sub> fi<sub>re</sub>b<sub>rea</sub>k <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>move</sub> <sub>somew</sub>h<sub>ere.</sub>

• Do nothin<sub>g,</sub> remainin<sub>g</sub> on standb<sub>y</sub> and conservin<sub>g</sub> ener<sub>gy</sub>. This remains until further <sub>c</sub>h<sub>ange, so</sub> d<sub>o no</sub>t <sub>queue</sub> it <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> ti<sub>mes.</sub>

If the agent is drone:

Th<sub>ere</sub> <sub>are</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> t<sub>wo</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>you</sub> <sub>can</sub> <sub>per</sub>f<sub>orm.</sub> N<sub>o</sub>thi<sub>ng</sub> <sub>e</sub>l<sub>se</sub> i<sub>s</sub> <sub>a</sub>ll<sub>owe</sub>d<sub>:</sub>

• Move in the direction of an<sub>y</sub> tar<sub>g</sub>et coordinate location<sub>,</sub> re<sub>g</sub>ardless of distance<sub>,</sub> observi<sub>ng</sub> <sub>a</sub>l<sub>ong</sub> th<sub>e</sub> <sub>way.</sub> Y<sub>ou</sub> <sub>w</sub>ill <sub>arr</sub>i<sub>ve</sub> th<sub>ere</sub> i<sub>n</sub> <sub>one</sub> <sub>move.</sub> Y<sub>ou</sub> <sub>on</sub>l<sub>y</sub> <sub>nee</sub>d <sub>one</sub> <sub>move</sub> <sub>ac</sub>ti<sub>on</sub> to move an<sub>y</sub>where (so do not <sub>p</sub>lan in-between destinations, onl<sub>y</sub> final destinations).

• Do nothin<sub>g,</sub> remainin<sub>g</sub> on standb<sub>y</sub> and conservin<sub>g</sub> ener<sub>gy</sub>. This remains until further <sub>c</sub>h<sub>ange,</sub> <sub>so</sub> d<sub>o</sub> <sub>no</sub>t <sub>queue</sub> it <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> ti<sub>mes.</sub>

If the agent is firefighter:

Your current status:

• Carr<sub>y</sub>in<sub>g</sub> civilian: {CARRYING CIVILIAN}

• Water level: {WATER LEVEL}/5

Th<sub>ere</sub> <sub>are</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>seven</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>you</sub> <sub>can</sub> <sub>per</sub>f<sub>orm.</sub> N<sub>o</sub>thi<sub>ng</sub> <sub>e</sub>l<sub>se</sub> i<sub>s</sub> <sub>a</sub>ll<sub>owe</sub>d<sub>:</sub>

• Move to an<sub>y</sub> coordinate location. DISTANCE DOES NOT MATTER<sub>;</sub> <sub>y</sub>ou will reach that location in one move. You onl<sub>y</sub> need one move action to move an<sub>y</sub>where (so do not <sub>p</sub>lan in-between destinations, onl<sub>y</sub> final destinations).

• Cut down some or all trees within <sub>y</sub>our current cell. Define how man<sub>y,</sub> or if all.

• Pick u<sub>p</sub> a nearb<sub>y</sub> civilian.

• Dro<sub>p</sub> of <sub>y</sub>our current civilian.

• S<sub>p</sub>ra<sub>y</sub> a wide cone of water in the direction of a tar<sub>g</sub>et coordinate location. Uses 1 water.

• Refill water. You must be on a water source.

• Do nothin<sub>g,</sub> remainin<sub>g</sub> on standb<sub>y</sub> and conservin<sub>g</sub> ener<sub>gy</sub>. This remains until further <sub>c</sub>h<sub>ange,</sub> <sub>so</sub> d<sub>o</sub> <sub>no</sub>t <sub>queue</sub> it <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> ti<sub>mes.</sub>

If the agent is helicopter:

Your current status:

• Carr<sub>y</sub>in<sub>g</sub> firefi<sub>g</sub>hters: {Carr<sub>y</sub>in<sub>g</sub> firefi<sub>g</sub>hters}/5

• Water level: {water level}/5

Th<sub>ere are exp</sub>li<sub>c</sub>itl<sub>y</sub> fi<sub>ve poss</sub>ibl<sub>e ac</sub>ti<sub>ons you can per</sub>f<sub>orm.</sub> N<sub>o</sub>thi<sub>ng e</sub>l<sub>se</sub> i<sub>s a</sub>ll<sub>owe</sub>d<sub>:</sub>

• Move to an<sub>y</sub> coordinate location. Distance does not matter<sub>;</sub> <sub>y</sub>ou will reach that location in one move. You onl<sub>y</sub> need one move action to move an<sub>y</sub>where (so do not <sub>p</sub>lan in-between destinations, onl<sub>y</sub> final destinations).

• Pick u<sub>p</sub> ALL nearb<sub>y</sub> Firefi<sub>g</sub>hter A<sub>g</sub>ents<sub>,</sub> u<sub>p</sub> to 5 in total. (You onl<sub>y</sub> need to do this once.)

• Dro<sub>p</sub> of all carried Firefi<sub>g</sub>hter A<sub>g</sub>ents.

• Refill water stora<sub>g</sub>e. You must be above a water source.

• De<sub>p</sub>lo<sub>y</sub> 1 unit of water directl<sub>y</sub> below.

Now your task is the following:{task}

Your job is to break down, if needed, your high-level task into a sequence of actions that you <sub>can</sub> <sub>per</sub>f<sub>orm.</sub> Y<sub>ou</sub> <sub>may</sub> <sub>ma</sub>k<sub>e</sub> <sub>up</sub> t<sub>o</sub> 20 <sub>or</sub> <sub>as</sub> f<sub>ew</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>as</sub> <sub>you</sub> <sub>nee</sub>d<sub>.</sub> Th<sub>ey</sub> <sub>mus</sub>t <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> b<sub>e</sub> <sub>one</sub> of the allowed action t<sub>yp</sub>es; nothin<sub>g</sub> else is <sub>p</sub>ermitted (scannin<sub>g</sub>, observin<sub>g</sub>, communicatin<sub>g</sub>, re<sub>p</sub>ortin<sub>g</sub>, and waitin<sub>g</sub> are all automaticall<sub>y</sub> done).

N<sub>o</sub>t<sub>e:</sub> O<sub>n</sub>l<sub>y</sub> <sub>use</sub> “d<sub>o</sub> <sub>no</sub>thi<sub>ng</sub>” if <sub>you</sub> h<sub>ave</sub> <sub>no</sub>thi<sub>ng</sub> t<sub>o</sub> d<sub>o</sub> f<sub>or</sub> thi<sub>s</sub> t<sub>as</sub>k<sub>.</sub> D<sub>o</sub> <sub>no</sub>t <sub>use</sub> it t<sub>o</sub> <sub>wa</sub>it i<sub>n</sub> b<sub>e</sub>t<sub>ween</sub> <sub>ac</sub>ti<sub>ons.</sub> D<sub>o</sub> <sub>no</sub>t <sub>use</sub> it t<sub>o</sub> <sub>en</sub>d <sub>your</sub> <sub>ac</sub>ti<sub>ons,</sub> <sub>s</sub>i<sub>nce</sub> <sub>you</sub> <sub>au</sub>t<sub>oma</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>rema</sub>i<sub>n</sub> <sub>on</sub> <sub>s</sub>t<sub>an</sub>db<sub>y</sub> <sub>w</sub>h<sub>en</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>e</sub>d<sub>.</sub> D<sub>o</sub> <sub>no</sub>t <sub>use</sub> it t<sub>o</sub> <sub>o</sub>b<sub>serve,</sub> <sub>as</sub> <sub>you</sub> <sub>are</sub> <sub>a</sub>l<sub>ways</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> <sub>regar</sub>dl<sub>ess</sub> <sub>o</sub>f <sub>ac</sub>ti<sub>on.</sub> O<sub>n</sub>l<sub>y</sub> <sub>use</sub> it if <sub>an</sub>d <sub>on</sub>l<sub>y</sub> if it i<sub>s</sub> th<sub>e</sub> <sub>on</sub>l<sub>y</sub> <sub>ac</sub>ti<sub>on.</sub>

Y<sub>ou</sub> <sub>are</sub> t<sub>o</sub> <sub>respon</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>ags:</sub>

## <actions>

1<sub>.</sub> Fi<sub>rs</sub>t <sub>ac</sub>ti<sub>on</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub>

2<sub>.</sub> S<sub>econ</sub>d <sub>ac</sub>ti<sub>on</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub>

## </actions>

K<sub>eep eac</sub>h <sub>ac</sub>ti<sub>on</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub> t<sub>o</sub> ONE <sub>s</sub>h<sub>or</sub>t <sub>sen</sub>t<sub>ence.</sub> N<sub>o e</sub>l<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> i<sub>s nee</sub>d<sub>e</sub>d<sub>.</sub> LESS THAN   
21 ITEMS<sub>.</sub>   
R<sub>emem</sub>b<sub>er,</sub> <sub>scann</sub>i<sub>ng,</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng,</sub> <sub>commun</sub>i<sub>ca</sub>ti<sub>ng,</sub> <sub>repor</sub>ti<sub>ng,</sub> <sub>an</sub>d <sub>wa</sub>iti<sub>ng</sub> <sub>are</sub> <sub>a</sub>ll <sub>au</sub>t<sub>oma</sub>ti<sub>ca</sub>ll<sub>y</sub>   
d<sub>one,</sub> <sub>so</sub> DO NOT i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>em.</sub> O<sub>n</sub>l<sub>y</sub> <sub>per</sub>f<sub>orm</sub> <sub>ac</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>are</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>a</sub>ll<sub>owe</sub>d<sub>.</sub>   
Feedback Providing:   
You are {agent id}, a {Bulldozer/Drone/Firefighter/Helicopter} agent within a forest grid   
world. You are currently at {location}.   
Y<sub>ou</sub>r <sub>capa</sub>biliti<sub>es</sub> in<sub>c</sub>l<sub>u</sub>d<sub>e:</sub>   
If the agent is bulldozer:   
• Movin<sub>g</sub> to an<sub>y</sub> location on the ma<sub>p</sub>.   
• Creatin<sub>g</sub> fire breaks b<sub>y</sub> clearin<sub>g</sub> ve<sub>g</sub>etation.   
If the agent is drone:   
• Movin<sub>g</sub> to an<sub>y</sub> location on the ma<sub>p</sub>.   
• Scoutin<sub>g</sub> lar<sub>g</sub>e areas for fires<sub>,</sub> civilians<sub>,</sub> and other im<sub>p</sub>ortant objects.   
• Providin<sub>g</sub> aerial reconnaissance and surveillance.   
• Coverin<sub>g</sub> lar<sub>g</sub>e distances <sub>q</sub>uickl<sub>y</sub>.   
If the agent is firefighter:   
• Movin<sub>g</sub> to an<sub>y</sub> location on the ma<sub>p</sub>.   
• Cuttin<sub>g</sub> down trees in <sub>y</sub>our current cell.   
• Pickin<sub>g</sub> u<sub>p</sub> and dro<sub>pp</sub>in<sub>g</sub> of a civilian.   
• Usin<sub>g</sub> water to extin<sub>g</sub>uish fires.   
If the agent is helicopter:

• Movin<sub>g</sub> to an<sub>y</sub> location on the ma<sub>p</sub>.

• Trans<sub>p</sub>ortin<sub>g</sub> firefi<sub>g</sub>hters to diferent locations.

• Pickin<sub>g</sub> u<sub>p</sub> and de<sub>p</sub>lo<sub>y</sub>in<sub>g</sub> water for fire su<sub>pp</sub>ression.

• Providin<sub>g</sub> aerial su<sub>pp</sub>ort and coordination.

• Coverin<sub>g</sub> lar<sub>g</sub>e areas <sub>q</sub>uickl<sub>y</sub>.

Your current observations:{observation}.

Wh<sub>en</sub> <sub>prov</sub>idi<sub>ng</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>on</sub> t<sub>eam</sub> <sub>p</sub>l<sub>ans:</sub>

• Assess whether <sub>y</sub>our assi<sub>g</sub>ned task is within <sub>y</sub>our ca<sub>p</sub>abilities.

• Consider <sub>y</sub>our current <sub>p</sub>osition and the eficienc<sub>y</sub> of the <sub>p</sub>ro<sub>p</sub>osed movements.

• Evaluate whether <sub>y</sub>our task is a<sub>pp</sub>ro<sub>p</sub>riate for a Bulldozer a<sub>g</sub>ent.

• Provide feedback on task feasibilit<sub>y</sub> and <sub>p</sub>ositionin<sub>g</sub>.

• Res<sub>p</sub>ond with ’YES’ if the <sub>p</sub>lan is satisfactor<sub>y</sub> for <sub>y</sub>ou.

F<sub>ocus</sub> <sub>on</sub> <sub>your</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>ro</sub>l<sub>e</sub> <sub>as</sub> <sub>a</sub> B<sub>u</sub>lld<sub>ozer</sub> <sub>an</sub>d <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>propose</sub>d t<sub>as</sub>k<sub>s</sub> <sub>a</sub>li<sub>gn</sub> <sub>w</sub>ith <sub>your</sub> <sub>capa</sub>biliti<sub>es</sub> <sub>an</sub>d <sub>curren</sub>t <sub>s</sub>it<sub>ua</sub>ti<sub>on.</sub>