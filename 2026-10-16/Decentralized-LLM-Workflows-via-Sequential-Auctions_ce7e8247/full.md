# Decentralized LLM Workflows via Sequential Auctions

Ian Gemp<sup>1</sup>, Shubhendu Trivedi<sup>1</sup>, Adam Fisch<sup>1</sup>, Fantine Huot<sup>1</sup>, Michael Kaisers<sup>1</sup>, Mirella Lapata<sup>1</sup>, William W. Cohen<sup>1</sup>, Alekh Agarwal<sup>2</sup>, Kate Larson<sup>1,3</sup> and Jacob Eisenstein<sup>1</sup>

<sup>1</sup>Goo<sub>g</sub>le Dee<sub>p</sub>Mind<sub>,</sub> <sup>2</sup>Goo<sub>g</sub>le Research<sub>,</sub> <sup>3</sup>Universit<sub>y</sub> of Waterloo

We present a scalable framework for solving multi-step tasks through a decentralized market of selfinterested agents. Prior theoretical work demonstrated that agents organized through sequential auctions can converge to a globally optimal policy in a Markov decision process (MDP), but relied on the restrictive assumption of the existence of perfect clones as a model of perfect competition. In reality, exact clones drive agent profit to zero, violating the natural risk-reward incentives of participants. Conversely, we empirically demonstrate that the opposite extreme, absence of competition, also leads to market failure and value extraction collapse. Between these two extremes, we investigate how heterogeneous agents can survive in markets that span the critical middle ground of imperfect competition. To be profitable, agents learn to strategically shade bids to secure positive individual profit margins, while still ensuring that the marketplace functions efectively enough so that collaborative tasks posed by market consumers can still be solved. We formulate and evaluate four bidding algorithms (Mirror Descent, Thompson Sampling, Bayes-Optimal, and PPO) across three auction protocols (1st-Price, 2nd-Price, and Credit-Conserving Vickrey). Across a diverse set of domains, we demonstrate that imperfect competition successfully sustains necessary agent profit margins while maintaining high task success rates at low costs.

Keywords: decentralized reinforcement learning, auction mechanisms, multi-agent systems, LLM agents

## 1. Introduction

M<sub>o</sub>d<sub>ern</sub> AI <sub>sys</sub>t<sub>ems</sub> i<sub>ncreas</sub>i<sub>ng</sub>l<sub>y</sub> f<sub>ace</sub> t<sub>as</sub>k<sub>s</sub> th<sub>a</sub>t <sub>requ</sub>i<sub>re compos</sub>i<sub>ng mu</sub>lti<sub>p</sub>l<sub>e spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>capa</sub>biliti<sub>es—</sub> d<sub>ynam</sub>i<sub>ca</sub>ll<sub>y rou</sub>ti<sub>ng quer</sub>i<sub>es across vary</sub>i<sub>ng</sub> LLM <sub>s</sub>i<sub>zes</sub> t<sub>o op</sub>ti<sub>m</sub>i<sub>ze</sub> i<sub>n</sub>f<sub>erence cos</sub>t<sub>s, ga</sub>th<sub>er</sub>i<sub>ng an</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>z</sub>i<sub>ng ev</sub>id<sub>ence</sub> f<sub>rom</sub> di<sub>spara</sub>t<sub>e sources, or</sub> di<sub>spa</sub>t<sub>c</sub>hi<sub>ng su</sub>bt<sub>as</sub>k<sub>s</sub> t<sub>o ex</sub>t<sub>erna</sub>l t<sub>oo</sub>l<sub>s.</sub> Th<sub>e</sub> d<sub>om</sub>i<sub>-</sub> nant <sub>p</sub>aradi<sub>g</sub>m centralizes this com<sub>p</sub>osition in a sin<sub>g</sub>le controller (an “orchestrator” LLM or a fixed com<sub>p</sub>utation <sub>g</sub>ra<sub>p</sub>h), which must ex<sub>p</sub>licitl<sub>y p</sub>lan the se<sub>q</sub>uence of o<sub>p</sub>erations and assi<sub>g</sub>n credit to each com<sub>p</sub>onent (Hon<sub>g</sub> et al., 2024; Shen et al., 2023).

An alternative, inspired by economic theory, is to let a society of specialized agents coordinate through a market mechanism. To solve temporal credit assignment across multi-step trajectories without hierarchical oversight, we draw on the foundational Bucket Brigade principle (Bartholdi III and Eisenstein, 1996; Holland, 1985): each a<sub>g</sub>ent <sub>p</sub>urchases the current task state from the <sub>p</sub>recedin<sub>g</sub> <sub>agen</sub>t i<sub>n an auc</sub>ti<sub>on,</sub> t<sub>rans</sub>f<sub>orms</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e, an</sub>d <sub>rese</sub>ll<sub>s</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>o</sub> th<sub>e nex</sub>t d<sub>owns</sub>t<sub>ream agen</sub>t<sub>.</sub> T<sub>erm</sub>i<sub>na</sub>l t<sub>as</sub>k <sub>rewar</sub>d<sub>s earne</sub>d <sub>a</sub>t <sub>comp</sub>l<sub>e</sub>ti<sub>on na</sub>t<sub>ura</sub>ll<sub>y propaga</sub>t<sub>e</sub> b<sub>ac</sub>k<sub>war</sub>d th<sub>roug</sub>h th<sub>e c</sub>h<sub>a</sub>i<sub>n o</sub>f <sub>resa</sub>l<sub>e</sub> <sub>pr</sub>i<sub>ces, rewar</sub>di<sub>ng ups</sub>t<sub>ream con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>ors propor</sub>ti<sub>ona</sub>ll<sub>y</sub> t<sub>o</sub> th<sub>e</sub>i<sub>r marg</sub>i<sub>na</sub>l <sub>va</sub>l<sub>ue a</sub>dd<sub>e</sub>d<sub>.</sub>

T<sub>o groun</sub>d thi<sub>s</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>economy</sub> i<sub>n a re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> f<sub>ramewor</sub>k<sub>, we</sub> b<sub>u</sub>ild <sub>on</sub> th<sub>e</sub> work of Chan<sub>g</sub> et al. (2020). The<sub>y p</sub>roved that a Vickre<sub>y</sub> (second-<sub>p</sub>rice) auction <sub>p</sub>erfectl<sub>y</sub> ali<sub>g</sub>ns each <sub>agen</sub>t’<sub>s</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>op</sub>ti<sub>ma</sub>l biddi<sub>ng</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>w</sub>ith th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>op</sub>ti<sub>ma</sub>l MDP <sub>po</sub>li<sub>cy.</sub> H<sub>owever,</sub> t<sub>rans</sub>l<sub>a</sub>ti<sub>ng</sub> thi<sub>s</sub> th<sub>eore</sub>ti<sub>ca</sub>l b<sub>rea</sub>kth<sub>roug</sub>h t<sub>o</sub> <sub>rea</sub>l<sub>-wor</sub>ld AI <sub>soc</sub>i<sub>e</sub>ti<sub>es</sub> <sub>revea</sub>l<sub>s</sub> <sub>a</sub> <sub>cr</sub>iti<sub>ca</sub>l <sub>gap:</sub> th<sub>e</sub>i<sub>r</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>re</sub>li<sub>es on</sub> th<sub>e res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ve assump</sub>ti<sub>on o</sub>f <sub>per</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on v</sub>i<sub>a ar</sub>tifi<sub>c</sub>i<sub>a</sub>ll<sub>y c</sub>l<sub>one</sub>d <sub>agen</sub>t<sub>s.</sub>

I<sub>n</sub> <sub>prac</sub>ti<sub>ce,</sub> thi<sub>s</sub> <sub>per</sub>f<sub>ec</sub>t<sub>-c</sub>l<sub>one</sub> <sub>assump</sub>ti<sub>on</sub> b<sub>rea</sub>k<sub>s</sub> d<sub>own</sub> d<sub>ue</sub> t<sub>o</sub> t<sub>wo</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l <sub>econom</sub>i<sub>c</sub> <sub>rea</sub>liti<sub>es.</sub> First is a participation barrier: with exact clones, the expected highest and second-highest bids are id<sub>en</sub>ti<sub>ca</sub>l<sub>.</sub> Th<sub>e w</sub>i<sub>nn</sub>i<sub>ng agen</sub>t <sub>pays exac</sub>tl<sub>y</sub> th<sub>e</sub>i<sub>r</sub> t<sub>rue va</sub>l<sub>ua</sub>ti<sub>on,</sub> d<sub>r</sub>i<sub>v</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r ne</sub>t <sub>pro</sub>fit t<sub>o s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y zero.</sub> B<sub>ecause</sub> <sub>par</sub>ti<sub>c</sub>i<sub>pa</sub>ti<sub>ng</sub> i<sub>n</sub> <sub>a</sub> <sub>mar</sub>k<sub>e</sub>t <sub>carr</sub>i<sub>es</sub> i<sub>n</sub>h<sub>eren</sub>t <sub>r</sub>i<sub>s</sub>k<sub>,</sub> <sub>agen</sub>t<sub>s</sub> <sub>w</sub>ill <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>s</sub>it <sub>ou</sub>t <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>op</sub>t i<sub>n</sub> to risk their resources for a net-zero expected return. Second is the anti-cloning incentive: perfect com<sub>p</sub>etition will not naturall<sub>y</sub> arise because no rational actor will clone their own a<sub>g</sub>ent (which cannibalizes their own <sub>p</sub>rofit mar<sub>g</sub>in), and <sub>p</sub>artici<sub>p</sub>ants will activel<sub>y p</sub>rotect their models’ intellectual <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> <sup>f</sup>rom <sup>b</sup>e<sup>i</sup>n<sub>g</sub> co<sub>p</sub><sup>i</sup>e<sup>d b</sup><sub>y</sub> com<sub>p</sub>et<sup>i</sup>tors.

If <sub>per</sub>f<sub>ec</sub>t <sub>c</sub>l<sub>ones are</sub> th<sub>eore</sub>ti<sub>ca</sub>ll<sub>y unpro</sub>fit<sub>a</sub>bl<sub>e an</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>ca</sub>ll<sub>y</sub> i<sub>rra</sub>ti<sub>ona</sub>l<sub>,</sub> d<sub>oes</sub> th<sub>e mar</sub>k<sub>e</sub>t <sub>co</sub>ll<sub>apse</sub>? We first demonstrate that the opposite extreme—absence of competition (a monopsony at any step in the task chain)—does indeed cause the market to fail. When an a<sub>g</sub>ent is the sole eli<sub>g</sub>ible bu<sub>y</sub>er for <sub>a g</sub>i<sub>ven s</sub>t<sub>a</sub>t<sub>e,</sub> th<sub>ey ac</sub>t <sub>as a monopson</sub>i<sub>s</sub>t<sub>,</sub> biddi<sub>ng</sub> th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e m</sub>i<sub>n</sub>i<sub>mum requ</sub>i<sub>re</sub>d t<sub>o w</sub>i<sub>n.</sub> With<sub>ou</sub>t <sub>compe</sub>titi<sub>ve pressure,</sub> th<sub>e c</sub>l<sub>ear</sub>i<sub>ng pr</sub>i<sub>ce co</sub>ll<sub>apses, an</sub>d d<sub>owns</sub>t<sub>ream va</sub>l<sub>ue</sub> i<sub>s no</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y ex</sub>t<sub>rac</sub>t<sub>e</sub>d <sub>an</sub>d <sub>passe</sub>d <sub>ups</sub>t<sub>ream.</sub> Thi<sub>s</sub> b<sub>rea</sub>k<sub>s</sub> th<sub>e</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>cre</sub>dit <sub>ass</sub>i<sub>gnmen</sub>t <sub>requ</sub>i<sub>re</sub>d t<sub>o</sub> <sub>sus</sub>t<sub>a</sub>i<sub>n</sub> <sub>a</sub> <sub>mu</sub>lti<sub>-s</sub>t<sub>ep</sub> <sub>p</sub>i<sub>pe</sub>li<sub>ne, as ups</sub>t<sub>ream agen</sub>t<sub>s are s</sub>t<sub>arve</sub>d <sub>o</sub>f th<sub>e revenue nee</sub>d<sub>e</sub>d t<sub>o o</sub>f<sub>se</sub>t th<sub>e</sub>i<sub>r own</sub> i<sub>n</sub>f<sub>erence cos</sub>t<sub>s.</sub>

To brid<sub>g</sub>e this <sub>g</sub>a<sub>p</sub>, we ada<sub>p</sub>t the framework of Chan<sub>g</sub> et al. (2020) to societies of Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Models (LLMs), formulatin<sub>g</sub> a decentralized, se<sub>q</sub>uential forward auction mechanism tailored to th<sub>e opera</sub>ti<sub>ona</sub>l <sub>rea</sub>liti<sub>es o</sub>f <sub>mo</sub>d<sub>ern</sub> LLM <sub>p</sub>i<sub>pe</sub>li<sub>nes.</sub> I<sub>n</sub> d<sub>o</sub>i<sub>ng so, we s</sub>hift th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l f<sub>ocus</sub> t<sub>o</sub> the critical, realistic middle ground: imperfect competition. We model societies populated by LLM <sub>agen</sub>t<sub>s</sub> th<sub>a</sub>t <sub>occupy s</sub>i<sub>m</sub>il<sub>ar capa</sub>bilit<sub>y n</sub>i<sub>c</sub>h<sub>es</sub> b<sub>u</sub>t <sub>are</sub> dif<sub>eren</sub>ti<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> h<sub>e</sub>t<sub>erogeneous</sub> i<sub>n</sub>f<sub>erence cos</sub>t<sub>s,</sub> <sub>promp</sub>t <sub>eng</sub>i<sub>neer</sub>i<sub>ng,</sub> <sub>an</sub>d l<sub>oca</sub>li<sub>ze</sub>d h<sub>eur</sub>i<sub>s</sub>ti<sub>cs.</sub> B<sub>ecause</sub> <sub>o</sub>f<sub>-</sub>th<sub>e-s</sub>h<sub>e</sub>lf l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>are</sub> <sub>no</sub>t<sub>or</sub>i<sub>ous</sub>l<sub>y</sub> miscalibrated jud<sub>g</sub>es of their own success <sub>p</sub>robabilities and token costs (Fradkin and Krishnan, 2026), <sub>agen</sub>t<sub>s</sub> <sub>canno</sub>t <sub>re</sub>l<sub>y</sub> <sub>on</sub> <sub>zero-s</sub>h<sub>o</sub>t <sub>promp</sub>ti<sub>ng</sub> t<sub>o</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> I<sub>ns</sub>t<sub>ea</sub>d<sub>,</sub> <sub>eac</sub>h <sub>agen</sub>t l<sub>earns</sub> <sub>a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht <sub>neura</sub>l <sub>va</sub>l<sub>ua</sub>ti<sub>on po</sub>li<sub>cy v</sub>i<sub>a re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> t<sub>o nav</sub>i<sub>ga</sub>t<sub>e</sub> th<sub>e</sub> t<sub>ra</sub>d<sub>eo</sub>f b<sub>e</sub>t<sub>ween w</sub>i<sub>nn</sub>i<sub>ng</sub> th<sub>e r</sub>i<sub>g</sub>ht t<sub>o</sub> <sub>ac</sub>t <sub>an</sub>d <sub>s</sub>h<sub>a</sub>di<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> bid<sub>s</sub> t<sub>o</sub> <sub>secure</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>pro</sub>fit <sub>marg</sub>i<sub>n.</sub>

I<sub>n</sub> thi<sub>s paper, we presen</sub>t <sub>a compre</sub>h<sub>ens</sub>i<sub>ve emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>u</sub>d<sub>y o</sub>f d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d LLM <sub>rou</sub>ti<sub>ng un</sub>d<sub>er</sub> i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on.</sub> O<sub>ur core con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e:</sub>

1. Decentralized LLM Markets and Imperfect Competition: We introduce an architectural and <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub>i<sub>c</sub> f<sub>ramewor</sub>k f<sub>or sequen</sub>ti<sub>a</sub>l <sub>auc</sub>ti<sub>on</sub> MDP<sub>s</sub> i<sub>n</sub> LLM <sub>soc</sub>i<sub>e</sub>ti<sub>es,</sub> d<sub>emons</sub>t<sub>ra</sub>ti<sub>ng exper</sub>i<sub>men</sub>t<sub>a</sub>ll<sub>y</sub> th<sub>a</sub>t i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on</sub> i<sub>s</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t t<sub>o</sub> <sub>sus</sub>t<sub>a</sub>i<sub>n</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>pro</sub>fit <sub>marg</sub>i<sub>ns</sub> <sub>an</sub>d <sub>ena</sub>bl<sub>e</sub> <sub>v</sub>i<sub>a</sub>bl<sub>e</sub> <sub>mu</sub>lti<sub>-</sub> <sub>agen</sub>t <sub>p</sub>i<sub>pe</sub>li<sub>nes</sub> <sub>un</sub>d<sub>er</sub> t<sub>ra</sub>diti<sub>ona</sub>l <sub>auc</sub>ti<sub>on</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sms.</sub>

2. Monopsony Failure Validation: We reveal empirically through a targeted synthetic domain th<sub>a</sub>t <sub>a</sub>b<sub>sence</sub> <sub>o</sub>f <sub>compe</sub>titi<sub>on</sub> <sub>co</sub>ll<sub>apses</sub> th<sub>e</sub> <sub>mar</sub>k<sub>e</sub>t<sub>,</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on</sub> <sub>as</sub> <sub>a</sub> <sub>v</sub>i<sub>a</sub>bl<sub>e</sub> <sub>opera</sub>ti<sub>ona</sub>l <sub>reg</sub>i<sub>me un</sub>d<sub>er</sub> t<sub>ra</sub>diti<sub>ona</sub>l <sub>auc</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sms.</sub>

3. Algorithmic Bidding Strategies: We formulate and benchmark four biddin al orithms—Mirror Descent<sub>,</sub> Thom<sub>p</sub>son Sam<sub>p</sub>lin<sub>g,</sub> Ba<sub>y</sub>es-O<sub>p</sub>timal<sub>,</sub> and PPO—demonstratin<sub>g</sub> how individual a<sub>g</sub>ents <sub>can</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>earn</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>pr</sub>i<sub>c</sub>i<sub>ng</sub> <sub>an</sub>d bid<sub>-s</sub>h<sub>a</sub>di<sub>ng</sub> <sub>po</sub>li<sub>c</sub>i<sub>es.</sub>

4. Extensive Benchmarking: We deploy our framework across diverse LLM domains (including a math reasonin<sub>g</sub> task and newl<sub>y</sub>-constructed s<sub>y</sub>nthetic news market). Throu<sub>g</sub>h market densit<sub>y</sub> swee<sub>p</sub>s across 1st-Price, 2nd-Price, and Credit-Conservin<sub>g</sub> Vickre<sub>y</sub> (CCV) auctions (Fi<sub>g</sub> 1b&3), we <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on</sub> i<sub>s su</sub>fi<sub>c</sub>i<sub>en</sub>t t<sub>o sus</sub>t<sub>a</sub>i<sub>n pos</sub>iti<sub>ve agen</sub>t <sub>pro</sub>fit<sub>s w</sub>hil<sub>e s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> achievin<sub>g</sub> hi<sub>g</sub>her task success rates (accurac<sub>y</sub>) than centralized orchestration baselines.

## 2. Background

To formalize the decentralized ecosystem just described, we model the multi-agent task pipeline as a se<sub>q</sub>uential auction, in line with <sub>p</sub>revious work (Chan<sub>g</sub> et al., 2020). In this framework, self-interested <sub>agen</sub>t<sub>s</sub> bid f<sub>or</sub> th<sub>e r</sub>i<sub>g</sub>ht t<sub>o progress</sub>i<sub>ve</sub>l<sub>y</sub> t<sub>rans</sub>f<sub>orm an</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { 0 }$ i<sub>n</sub>t<sub>o a</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> $s _ { T }$ <sub>.</sub> B<sub>e</sub>l<sub>o</sub>w<sub>,</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng mar</sub>k<sub>e</sub>t MDP<sub>,</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>pro</sub>fit i<sub>ncen</sub>ti<sub>ves o</sub>f th<sub>e agen</sub>t<sub>s, an</sub>d th<sub>e</sub> id<sub>ea</sub>li<sub>ze</sub>d th<sub>eore</sub>ti<sub>ca</sub>l <sub>c</sub>l<sub>ear</sub>i<sub>ng</sub> <sub>ru</sub>l<sub>es</sub> th<sub>a</sub>t <sub>ena</sub>bl<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>cre</sub>dit <sub>ass</sub>i<sub>gnmen</sub>t<sub>.</sub>

## 2.1. The Sequential Spot Market MDP

We model decentralized task routin<sub>g</sub> as a finite horizon (�) se<sub>q</sub>uential Markov Decision Process (MDP) defined b<sub>y</sub> the tu<sub>p</sub>le $( S , { \mathcal { R } } , { \mathcal { P } } , R , \gamma )$ . Rather than relying on a centralized policy �(�|�) to select actions, the action s<sub>p</sub>ace A is defined b<sub>y</sub> a societ<sub>y</sub> of inde<sub>p</sub>endent a<sub>g</sub>ents. For intuition, consider a <sub>gr</sub>id<sub>wor</sub>ld<sub>:</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>a s</sub>i<sub>ng</sub>l<sub>e con</sub>t<sub>ro</sub>ll<sub>er c</sub>h<sub>oos</sub>i<sub>ng</sub> t<sub>o move up or</sub> d<sub>own, spec</sub>i<sub>a</sub>li<sub>ze</sub>d “U<sub>p</sub>” <sub>an</sub>d “D<sub>own</sub>” <sub>agen</sub>t<sub>s</sub> <sub>compe</sub>t<sub>e</sub> i<sub>n</sub> <sub>an</sub> <sub>auc</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht t<sub>o</sub> <sub>ac</sub>t <sub>on</sub> th<sub>e</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { t }$

At each ste<sub>p</sub> �<sub>,</sub> a re<sub>g</sub>istr<sub>y</sub> of a<sub>g</sub>ents evaluates $s _ { t }$ t<sub>o</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> th<sub>e</sub>i<sub>r</sub> <sub>e</sub>li<sub>g</sub>ibilit<sub>y.</sub> Eli<sub>g</sub>ibl<sub>e</sub> <sub>agen</sub>t<sub>s</sub> <sub>su</sub>b<sub>m</sub>it bid<sub>s,</sub> <sub>an</sub>d th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t bidd<sub>er</sub> <sub>w</sub>i<sub>ns</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht t<sub>o</sub> <sub>process</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> Th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> w<sup>i</sup>nn<sup>i</sup>n<sub>g</sub> a<sub>g</sub>ent $i \gamma _ { s }$ l<sub>oca</sub>li<sub>ze</sub>d <sub>ac</sub>ti<sub>on</sub> $a _ { t } ,$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>genera</sub>t<sub>es</sub> th<sub>e</sub> <sub>nex</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { t + 1 }$ <sub>an</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>an</sub> <sub>env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>rewar</sub>d $r ( s _ { t } , a _ { t } )$ <sub>.</sub> Th<sub>e process</sub> i<sub>s repea</sub>t<sub>e</sub>d <sub>un</sub>til <sub>a</sub> fi<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { T }$ i<sub>s reac</sub>h<sub>e</sub>d<sub>.</sub> S<sub>oc</sub>i<sub>a</sub>l <sub>we</sub>lf<sub>are</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> th<sub>e sum o</sub>f di<sub>scoun</sub>t<sub>e</sub>d <sub>rewar</sub>d<sub>s.</sub>

Agents optimize for their individual net utility, or profit, over the course of a transaction. For an <sub>agen</sub>t <sub>w</sub>i<sub>nn</sub>i<sub>ng</sub> th<sub>e auc</sub>ti<sub>on a</sub>t <sub>s</sub>t<sub>ep</sub> $t ,$ th<sub>e pro</sub>fit $U _ { i , t }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> th<sub>e sum o</sub>f th<sub>e env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>rewar</sub>d <sub>an</sub>d th<sub>e</sub> <sub>ne</sub>t fl<sub>ow</sub> <sub>o</sub>f <sub>cap</sub>it<sub>a</sub>l<sub>:</sub>

$$
U _ { i , t } = r ( s _ { t } , a _ { t } ) + P _ { \mathrm { i n } } ^ { ( t ) } - P _ { \mathrm { o u t } } ^ { ( t ) }\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> $P _ { \mathrm { o u t } } ^ { ( t ) }$ i<sub>s</sub> th<sub>e</sub> <sub>c</sub>l<sub>ear</sub>i<sub>ng</sub> <sub>pr</sub>i<sub>ce</sub> th<sub>e</sub> <sub>agen</sub>t <sub>pays</sub> t<sub>o</sub> <sub>acqu</sub>i<sub>re</sub> $s _ { t } .$ <sub>,</sub> <sub>an</sub>d $P _ { \mathrm { i n } } ^ { ( t ) }$ i<sub>s</sub> th<sub>e</sub> <sub>revenue</sub> th<sub>e</sub> <sub>agen</sub>t <sub>rece</sub>i<sub>ves</sub> <sub>w</sub>h<sub>en</sub> th<sub>ey su</sub>b<sub>sequen</sub>tl<sub>y auc</sub>ti<sub>on</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { t + 1 }$ t<sub>o</sub> th<sub>e nex</sub>t d<sub>owns</sub>t<sub>ream agen</sub>t<sub>.</sub>

## 2.2. Auction Mechanisms and Credit Assignment

Th<sub>e</sub> <sub>ru</sub>l<sub>es</sub> <sub>govern</sub>i<sub>ng</sub> th<sub>e</sub> <sub>c</sub>l<sub>ear</sub>i<sub>ng</sub> <sub>pr</sub>i<sub>ce</sub> $P _ { \mathrm { o u t } } ^ { ( t ) }$ <sub>an</sub>d th<sub>e</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>revenue</sub> $P _ { \mathrm { i n } } ^ { ( t ) }$ d<sub>epen</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> <sub>on</sub> the chosen auction mechanism. Chan<sub>g</sub> et al. (2020) established a foundational brid<sub>g</sub>e between reinforcement learnin<sub>g</sub> and mechanism desi<sub>g</sub>n b<sub>y</sub> demonstratin<sub>g</sub> that under a Vickre<sub>y</sub> (second-<sub>p</sub>rice) <sub>auc</sub>ti<sub>on,</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y op</sub>ti<sub>ma</sub>l MDP <sub>po</sub>li<sub>cy cons</sub>tit<sub>u</sub>t<sub>es a</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> f<sub>or myop</sub>i<sub>c agen</sub>t<sub>s max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oca</sub>l t<sub>ransac</sub>ti<sub>on</sub> <sub>pro</sub>fit<sub>.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>resu</sub>lt <sub>re</sub>li<sub>es</sub> <sub>on</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t th<sub>eore</sub>ti<sub>ca</sub>l <sub>proper</sub>ti<sub>es:</sub>

1. Truthful Bidding via Myopia: Assuming agents are myopic and optimize strictly for the immediate t<sub>ransac</sub>ti<sub>on,</sub> th<sub>e</sub> Vi<sub>c</sub>k<sub>rey</sub> <sub>auc</sub>ti<sub>on</sub> <sub>guaran</sub>t<sub>ees</sub> th<sub>a</sub>t biddi<sub>ng</sub> <sub>one</sub>’<sub>s</sub> t<sub>rue</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> $( b _ { i } = \nu _ { i } )$ i<sub>s a wea</sub>kl<sub>y</sub> d<sub>om</sub>i<sub>nan</sub>t <sub>s</sub>t<sub>ra</sub>t<sub>egy.</sub> Th<sub>us, an agen</sub>t’<sub>s</sub> bid <sub>revea</sub>l<sub>s</sub> it<sub>s</sub> t<sub>rue es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>ac</sub>ti<sub>on-va</sub>l<sub>ue,</sub> $Q _ { i } ( s _ { t } )$

2. Perfect Credit Conservation via Clones: To ensure the market clears at the optimal state-value without re<sub>q</sub>uirin<sub>g</sub> external <sub>p</sub>latform subsidies, Chan<sub>g</sub> et al. (2020) in their Section 5.4 build on the <sub>assump</sub>ti<sub>on</sub> th<sub>a</sub>t th<sub>e mar</sub>k<sub>e</sub>t i<sub>s popu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>exac</sub>t <sub>c</sub>l<sub>ones o</sub>f <sub>every agen</sub>t<sub>.</sub> With <sub>per</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on</sub> <sub>v</sub>i<sub>a exac</sub>t <sub>c</sub>l<sub>ones,</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>an</sub>d <sub>secon</sub>d<sub>-</sub>hi<sub>g</sub>h<sub>es</sub>t bid<sub>s are</sub> id<sub>en</sub>ti<sub>ca</sub>l $( b _ { ( 1 ) } = b _ { ( 2 ) } )$ <sub>.</sub> B<sub>ecause</sub> th<sub>e</sub> Vi<sub>c</sub>k<sub>rey auc</sub>ti<sub>on c</sub>h<sub>arges</sub> th<sub>e secon</sub>d<sub>-</sub>hi<sub>g</sub>h<sub>es</sub>t bid<sub>,</sub> th<sub>e mar</sub>k<sub>e</sub>t <sub>c</sub>l<sub>ear</sub>i<sub>ng pr</sub>i<sub>ce exac</sub>tl<sub>y ex</sub>t<sub>rac</sub>t<sub>s</sub> th<sub>e</sub> maximum Q-value: $\begin{array} { r } { P _ { \mathrm { o u t } } ^ { \overline { { ( t ) } } } = \operatorname* { m a x } _ { i } Q _ { i } ( s _ { t } ) \overline { { = V ^ { * } } } ( s _ { t } ) } \end{array}$ .

Wh<sub>en</sub> th<sub>e</sub> <sub>c</sub>l<sub>ear</sub>i<sub>ng</sub> <sub>pr</sub>i<sub>ce</sub> <sub>per</sub>f<sub>ec</sub>tl<sub>y</sub> t<sub>rac</sub>k<sub>s</sub> $V ^ { * } ( s _ { t } )$ , t<sup>h</sup>e revenue <sub>p</sub>asse<sup>d</sup> u<sub>p</sub>stream acts as an exact Bellman backu<sub>p</sub>. S<sub>p</sub>ecificall<sub>y</sub>, if the downstream auction at ste<sub>p</sub> � + 1 clears at $V ^ { * } { \left( s _ { t + 1 } \right) }$ , t<sup>h</sup>e u<sub>p</sub>stream a<sub>g</sub>ent rece<sup>i</sup>ves $P _ { \mathrm { i n } } ^ { ( t ) } = \gamma V ^ { * } \big ( s _ { t + 1 } \big ) . ^ { 1 }$ B<sub>y max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oca</sub>l <sub>pro</sub>fit $U _ { i , t }$ <sub>, agen</sub>t<sub>s na</sub>t<sub>ura</sub>ll<sub>y a</sub>li<sub>gn</sub> th<sub>e</sub>i<sub>r</sub> economic incentives with the global MDP objective, recursively propagating credit back to the initial <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> H<sub>owever</sub> thi<sub>s</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>a</sub>li <sub>nmen</sub>t <sub>uaran</sub>t<sub>ee unrave</sub>l<sub>s</sub> if <sub>e</sub>ith<sub>er o</sub>f th<sub>e a</sub>b<sub>ove assum</sub> ti<sub>ons</sub> i<sub>s</sub> unmet. Hence<sub>,</sub> we aim to stud<sub>y</sub> the robustness of a hetero<sub>g</sub>eneous LLM market MDP em<sub>p</sub>iricall<sub>y</sub>.

## 3. Decentralized LLM Markets and Imperfect Competition

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>mar</sub>k<sub>e</sub>t MDP t<sub>o</sub> LLM <sub>soc</sub>i<sub>e</sub>ti<sub>es</sub> <sub>requ</sub>i<sub>res</sub> f<sub>orma</sub>li<sub>z</sub>i<sub>ng</sub> i<sub>n</sub>f<sub>erence</sub> <sub>cos</sub>t<sub>s</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> <sub>rewar</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ure an</sub>d <sub>re</sub>l<sub>ax</sub>i<sub>ng</sub> th<sub>e</sub> id<sub>ea</sub>li<sub>ze</sub>d <sub>assump</sub>ti<sub>on o</sub>f <sub>per</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on ma</sub>d<sub>e</sub> i<sub>n prev</sub>i<sub>ous wor</sub>k<sub>.</sub>

## 3.1. LLM Transformations and Inference Costs

In an LLM societ<sub>y,</sub> the state $s _ { t }$ <sub>represen</sub>t<sub>s</sub> th<sub>e</sub> <sub>curren</sub>t t<sub>as</sub>k <sub>con</sub>t<sub>ex</sub>t<sub>,</sub> <sub>compr</sub>i<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>ex</sub>t <sub>or</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d payload (e.g., a natural language query, an intermediate translation, or a JSON object containing accumulated evidence). The action s<sub>p</sub>ace corres<sub>p</sub>onds to an a<sub>g</sub>ent’s localized LLM invocation, transformin<sub>g</sub> t<sup>h</sup>e <sub>p</sub>a<sub>y</sub><sup>l</sup>oa<sup>d</sup> to <sub>g</sub>enerate $s _ { t + 1 }$

C<sub>ruc</sub>i<sub>a</sub>ll <sub>,</sub> t<sub>a</sub>ki<sub>n</sub> thi<sub>s ac</sub>ti<sub>on</sub> i<sub>ncurs a</sub> l<sub>oca</sub>l i<sub>n</sub>f<sub>erence cos</sub>t <sub>�</sub> f<sub>or</sub> th<sub>e a en</sub>t $c _ { i , t }$ $( \mathsf { e . g . }$ , API token costs). W<sub>e</sub> f<sub>orma</sub>li<sub>ze</sub> thi<sub>s</sub> b<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> th<sub>e env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>rewar</sub>d d<sub>ur</sub>i<sub>ng</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e s</sub>t<sub>eps as s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y nega</sub>ti<sub>ve:</sub> $r ( s _ { t } , a _ { t } ) = - c _ { i , t }$ <sub>.</sub> If th<sub>e agen</sub>t’<sub>s ac</sub>ti<sub>on a</sub>t <sub>s</sub>t<sub>ep</sub> $T - 1$ <sub>success</sub>f<sub>u</sub>ll<sub>y</sub> <sub>reso</sub>l<sub>ves</sub> th<sub>e</sub> t<sub>as</sub>k <sub>an</sub>d t<sub>rans</sub>iti<sub>ons</sub> th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t t<sub>o a</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { T } .$ <sub>,</sub> th<sub>e rewar</sub>d $r ( s _ { T - 1 } , a _ { T - 1 } )$ i<sub>ncorpora</sub>t<sub>es</sub> b<sub>o</sub>th th<sub>e</sub> <sub>nega</sub>ti<sub>ve</sub> i<sub>n</sub>f<sub>erence</sub> <sub>cos</sub>t <sub>an</sub>d th<sub>e pos</sub>iti<sub>ve</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d <sub>prov</sub>id<sub>e</sub>d b<sub>y</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>or.</sub> B<sub>ecause</sub> th<sub>e</sub> t<sub>as</sub>k i<sub>s comp</sub>l<sub>e</sub>t<sub>e, no</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>auc</sub>ti<sub>on</sub> <sub>occurs,</sub> <sub>y</sub>i<sub>e</sub>ldi<sub>ng</sub> $P _ { \mathrm { i n } } ^ { ( \overline { { T } } - 1 ) } = 0$

## 3.2. The Participation Barrier and the Clone Assumption

W<sub>e now a</sub>dd<sub>ress</sub> th<sub>e pro</sub>bl<sub>em o</sub>f <sub>c</sub>l<sub>on</sub>i<sub>ng agen</sub>t<sub>s</sub> i<sub>n a mar</sub>k<sub>e</sub>t MDP<sub>.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e assump</sub>ti<sub>on o</sub>f <sub>per</sub>f<sub>ec</sub>t <sub>c</sub>l<sub>ones,</sub> th<sub>e equ</sub>ilib<sub>r</sub>i<sub>um c</sub>l<sub>ear</sub>i<sub>ng pr</sub>i<sub>ce exac</sub>tl<sub>y equa</sub>l<sub>s</sub> th<sub>e w</sub>i<sub>nner</sub>’<sub>s</sub> t<sub>rue va</sub>l<sub>ua</sub>ti<sub>on</sub> $\bar { ( P _ { \mathrm { o u t } } ^ { ( t ) } } = \bar { b _ { ( 2 ) } } = \nu _ { i } )$ h<sub>ence</sub> th<sub>e w</sub>i<sub>nner</sub>’<sub>s expec</sub>t<sub>e</sub>d <sub>u</sub>tilit<sub>y</sub> i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y zero</sub> $( \mathbb { E } [ U _ { i , t } ] = 0 )$ <sub>.</sub> A<sub>n agen</sub>t <sub>max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng expec</sub>t<sub>e</sub>d <sub>pro</sub>fit h<sub>as no s</sub>t<sub>r</sub>i<sub>c</sub>t i<sub>ncen</sub>ti<sub>ve</sub> t<sub>o op</sub>t i<sub>n</sub>t<sub>o</sub> th<sub>e mar</sub>k<sub>e</sub>t<sub>, s</sub>i<sub>nce a</sub>b<sub>s</sub>t<sub>en</sub>ti<sub>on a</sub>l<sub>rea</sub>d<sub>y ac</sub>hi<sub>eves zero pro</sub>fit<sub>.</sub>

F<sub>ur</sub>th<sub>ermore, no ra</sub>ti<sub>ona</sub>l <sub>agen</sub>t <sub>wou</sub>ld <sub>c</sub>h<sub>oose</sub> t<sub>o c</sub>l<sub>one</sub> th<sub>emse</sub>l<sub>ves, as</sub> d<sub>o</sub>i<sub>ng so s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y cann</sub>ib<sub>a</sub>li<sub>zes</sub> th<sub>e</sub>i<sub>r pro</sub>fit <sub>marg</sub>i<sub>n.</sub> C<sub>onverse</sub>l<sub>y, par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s w</sub>ill <sub>ac</sub>ti<sub>ve</sub>l<sub>y pro</sub>t<sub>ec</sub>t th<sub>e</sub>i<sub>r mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>rom</sub> b<sub>e</sub>i<sub>ng cop</sub>i<sub>e</sub>d b<sub>y</sub> competitors. Consequently, real LLM spot markets are bounded to operate under imperfect competition, <sub>w</sub>h<sub>ere</sub> $b _ { ( 1 ) } > b _ { ( 2 ) }$ <sub>.</sub> It i<sub>s prec</sub>i<sub>se</sub>l<sub>y</sub> thi<sub>s s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>gap</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>ows</sub> th<sub>e w</sub>i<sub>nner</sub> t<sub>o secure a pos</sub>iti<sub>ve pro</sub>fit mar<sub>g</sub><sup>i</sup>n $\left( U _ { i , t } > 0 \right)$ <sub>, overcom</sub>i<sub>ng</sub> th<sub>e par</sub>ti<sub>c</sub>i<sub>pa</sub>ti<sub>on</sub> b<sub>arr</sub>i<sub>er.</sub>

## 3.3. Defining Agent Heterogeneity

To operationalize the regime of imperfect competition, we define how agents difer within the market. I<sub>n our</sub> f<sub>ramewor</sub>k<sub>, a</sub>ll <sub>agen</sub>t<sub>s par</sub>ti<sub>c</sub>i<sub>pa</sub>t<sub>e</sub> i<sub>n a un</sub>ifi<sub>e</sub>d <sub>mar</sub>k<sub>e</sub>t <sub>an</sub>d <sub>can</sub> th<sub>eore</sub>ti<sub>ca</sub>ll<sub>y</sub> bid <sub>on any s</sub>t<sub>a</sub>t<sub>e;</sub> h<sub>owever,</sub> th<sub>ey</sub> <sub>ac</sub>t <sub>as</sub> i<sub>mper</sub>f<sub>ec</sub>t <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>t<sub>es,</sub> <sub>organ</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>spec</sub>i<sub>a</sub>li<sub>z</sub>i<sub>ng</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oca</sub>li<sub>ze</sub>d <sub>capa</sub>biliti<sub>es.</sub> We <sub>p</sub>arameterize this hetero<sub>g</sub>eneit<sub>y</sub> across two <sub>p</sub>rimar<sub>y</sub> axes:

1. Inference Cost $( c _ { i , t } )$ : Agents use diferent LLMs. For example, an agent backed by $\mathbf { a } < 1 \mathbf { B }$ pa<sup>r</sup>a<sup>m</sup>e<sup>t</sup>e<sup>r</sup> LLM i<sub>ncurs a muc</sub>h l<sub>ower</sub> b<sub>ase</sub> i<sub>n</sub>f<sub>erence cos</sub>t th<sub>an one</sub> b<sub>ac</sub>k<sub>e</sub>d b<sub>y a ></sub>100B <sub>parame</sub>t<sub>er</sub> LLM<sub>, a</sub>ll<sub>ow</sub>i<sub>ng</sub> it t<sub>o</sub> bid <sub>compe</sub>titi<sub>ve</sub>l<sub>y on</sub> l<sub>ower-marg</sub>i<sub>n</sub> t<sub>as</sub>k<sub>s.</sub>

2. Transition Reliability: Agents are diferentiated by distinct prompt engineering techniques, id<sub>en</sub>tit<sub>y em</sub>b<sub>e</sub>ddi<sub>ngs, an</sub>d <sub>asymme</sub>t<sub>r</sub>i<sub>c</sub> d<sub>a</sub>t<sub>a access</sub> $( \mathrm { e } . g .$ , access to diferent document cor<sub>p</sub>ora). Th<sub>ese</sub> dif<sub>erences cause</sub> th<sub>e</sub>i<sub>r s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>iti<sub>ons</sub> $( s _ { t } \to s _ { t + 1 } )$ t<sub>o vary</sub> i<sub>n qua</sub>lit<sub>y an</sub>d <sub>re</sub>l<sub>evance.</sub> A <sub>more</sub> <sub>capa</sub>bl<sub>e</sub> <sub>agen</sub>t i<sub>s</sub> <sub>more</sub> lik<sub>e</sub>l<sub>y</sub> t<sub>o</sub> <sub>genera</sub>t<sub>e</sub> <sub>a</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> th<sub>a</sub>t <sub>a</sub>tt<sub>rac</sub>t<sub>s</sub> hi<sub>g</sub>h d<sub>owns</sub>t<sub>ream</sub> bid<sub>s,</sub> th<sub>ere</sub>b<sub>y</sub> <sub>ra</sub>i<sub>s</sub>i<sub>ng</sub> it<sub>s expec</sub>t<sub>e</sub>d f<sub>u</sub>t<sub>ure revenue.</sub>

B<sub>ecause o</sub>f th<sub>ese var</sub>i<sub>ances, even among agen</sub>t<sub>s a</sub>tt<sub>emp</sub>ti<sub>ng</sub> t<sub>o</sub> f<sub>u</sub>lfill th<sub>e exac</sub>t <sub>same ro</sub>l<sub>e</sub> i<sub>n a</sub> t<sub>as</sub>k<sub>,</sub> t<sub>rue va</sub>l<sub>ua</sub>ti<sub>ons na</sub>t<sub>ura</sub>ll<sub>y</sub> di<sub>verge</sub> $( \nu _ { 1 } \neq \nu _ { 2 } )$ <sub>.</sub> Thi<sub>s s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>gap a</sub>ll<sub>ows</sub> th<sub>e mos</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>agen</sub>t t<sub>o s</sub>h<sub>a</sub>d<sub>e</sub> it<sub>s</sub> bid b<sub>e</sub>l<sub>ow</sub> it<sub>s</sub> <sub>own</sub> t<sub>rue</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>w</sub>hil<sub>e</sub> <sub>s</sub>till <sub>ou</sub>tbiddi<sub>ng</sub> th<sub>e</sub> <sub>res</sub>t <sub>o</sub>f th<sub>e</sub> <sub>mar</sub>k<sub>e</sub>t<sub>,</sub> th<sub>ere</sub>b<sub>y</sub> <sub>secur</sub>i<sub>ng</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve pro</sub>fit <sub>marg</sub>i<sub>n an</sub>d i<sub>ncen</sub>ti<sub>v</sub>i<sub>z</sub>i<sub>ng par</sub>ti<sub>c</sub>i<sub>pa</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e mar</sub>k<sub>e</sub>t<sub>.</sub>

## 3.4. Auction Mechanisms under Imperfect Competition

We stud<sub>y</sub> hetero<sub>g</sub>eneous markets under the same three auctions discussed in (Chan<sub>g</sub> et al., 2020):

• 2nd-Price (Vickrey): The downstream buyer pays $b _ { ( 2 ) }$ <sub>,</sub> b<sub>u</sub>t th<sub>e ups</sub>t<sub>ream se</sub>ll<sub>er rece</sub>i<sub>ves</sub> $b _ { ( 1 ) }$ <sub>.</sub> Th<sub>e</sub> <sub>mar</sub>k<sub>e</sub>t<sub>p</sub>l<sub>ace p</sub>l<sub>a</sub>tf<sub>orm mus</sub>t <sub>ac</sub>t <sub>as a mar</sub>k<sub>e</sub>t <sub>ma</sub>k<sub>er, su</sub>b<sub>s</sub>idi<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>c</sub>it $( b _ { ( 1 ) } - b _ { ( 2 ) } )$

• Credit-Conserving Vickrey (CCV): The buyer pays $b _ { ( 2 ) }$ <sub>, an</sub>d th<sub>e ups</sub>t<sub>ream se</sub>ll<sub>er rece</sub>i<sub>ves</sub> $b _ { ( 2 ) }$ <sub>.</sub> Thi<sub>s</sub> <sub>avo</sub>id<sub>s</sub> <sub>su</sub>b<sub>s</sub>idi<sub>es,</sub> b<sub>u</sub>t f<sub>a</sub>il<sub>s</sub> t<sub>o</sub> <sub>pass</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>w</sub>i<sub>nn</sub>i<sub>ng</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>ups</sub>t<sub>ream</sub> <sub>w</sub>h<sub>en</sub> $b _ { ( 1 ) } > b _ { ( 2 ) }$

• 1st-Price (Bucket Brigade): The winner pays their exact bid $( P _ { \mathrm { o u t } } ^ { ( t ) } = b _ { ( 1 ) } ^ { ( t ) } )$ <sub>, an</sub>d th<sub>e se</sub>ll<sub>er rece</sub>i<sub>ves</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t d<sub>owns</sub>t<sub>ream</sub> bid $( P _ { \mathrm { i n } } ^ { ( t ) } = \gamma b _ { ( 1 ) } ^ { ( t + 1 ) } )$ <sub>.</sub> C<sub>re</sub>dit <sub>conserva</sub>ti<sub>on</sub> i<sub>s s</sub>t<sub>ruc</sub>t<sub>ura</sub>ll<sub>y ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>ex</sub>t<sub>erna</sub>l <sub>su</sub>b<sub>s</sub>idi<sub>es.</sub> H<sub>owever,</sub> t<sub>o secure a pro</sub>fit<sub>, agen</sub>t<sub>s mus</sub>t l<sub>earn</sub> t<sub>o s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>ca</sub>ll<sub>y s</sub>h<sub>a</sub>d<sub>e</sub> th<sub>e</sub>i<sub>r</sub> bid<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub>i<sub>r</sub> t<sub>rue va</sub>l<sub>ua</sub>ti<sub>ons.</sub>

B<sub>ecause</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d Vi<sub>c</sub>k<sub>rey</sub> <sub>auc</sub>ti<sub>ons</sub> <sub>requ</sub>i<sub>re</sub> <sub>su</sub>b<sub>s</sub>idi<sub>es</sub> <sub>an</sub>d CCV <sub>passes</sub> <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>va</sub>l<sub>ue</sub> t<sub>o</sub> <sub>ups</sub>t<sub>ream</sub> agents, we prioritize the simpler 1st-Price auction; we justify this choice in Section 5.1.

## 3.5. Strategic Bidding in 1st-Price LLM Markets

I<sub>n a</sub> 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce auc</sub>ti<sub>on, an agen</sub>t <sub>mus</sub>t bid <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>ca</sub>ll<sub>y</sub> t<sub>o</sub> b<sub>a</sub>l<sub>ance w</sub>i<sub>n pro</sub>b<sub>a</sub>bilit<sub>y aga</sub>i<sub>ns</sub>t <sub>pro</sub>fit <sub>marg</sub>i<sub>n.</sub> T<sub>o</sub> <sub>max</sub>i<sub>m</sub>i<sub>ze</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pro</sub>fit<sub>,</sub> <sub>agen</sub>t � <sub>mus</sub>t fi<sub>rs</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> it<sub>s</sub> t<sub>rue</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> $\nu _ { i } ( s _ { t } )$ <sub>, w</sub>hi<sub>c</sub>h <sub>amoun</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e expec</sub>t<sub>e</sub>d f<sub>u</sub>t<sub>ure revenue m</sub>i<sub>nus</sub> it<sub>s</sub> l<sub>oca</sub>l i<sub>n</sub>f<sub>erence cos</sub>t<sub>s</sub> f<sub>or app</sub>l<sub>y</sub>i<sub>ng</sub> it<sub>s</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> $a _ { t }$ <sub>.</sub> F<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons o</sub>f $P _ { \mathrm { i n } } ^ { ( t ) }$ <sub>an</sub>d $r ( s _ { t } , a _ { t } )$ f<sub>rom a</sub>b<sub>ove,</sub> th<sub>e va</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s:</sub>

$$
\begin{array} { r } { \upsilon _ { i } ( s _ { t } ) = \mathbb { E } _ { s _ { t + 1 } \sim \mathcal { P } ( \cdot | s _ { t } , a _ { t } ) } \left[ P _ { \mathrm { i n } } ^ { ( t ) } + r ( s _ { t } , a _ { t } ) \right] = \mathbb { E } _ { s _ { t + 1 } \sim \mathcal { P } ( \cdot | s _ { t } , a _ { t } ) } \left[ \gamma b _ { ( 1 ) } ( s _ { t + 1 } ) - c _ { i , t } \right] . } \end{array}\tag{2}
$$

T<sub>wo agen</sub>t<sub>s e</sub>li<sub>g</sub>ibl<sub>e</sub> f<sub>or</sub> th<sub>e same s</sub>t<sub>a</sub>t<sub>e may</sub> dif<sub>er</sub> i<sub>n va</sub>l<sub>ua</sub>ti<sub>on</sub> th<sub>roug</sub>h b<sub>o</sub>th th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>f<sub>erence cos</sub>t $c _ { i , t } .$ <sub>an</sub>d th<sub>roug</sub>h th<sub>e expec</sub>t<sub>e</sub>d d<sub>owns</sub>t<sub>ream</sub> bid $b _ { ( 1 ) } \bigl ( s _ { t + 1 } \bigr )$ th<sub>a</sub>t th<sub>e</sub>i<sub>r</sub> dif<sub>eren</sub>t <sub>ou</sub>t<sub>pu</sub>t <sub>s</sub>t<sub>a</sub>t<sub>es a</sub>tt<sub>rac</sub>t<sub>,</sub> i<sub>.e.,</sub> th<sub>e</sub> t<sub>wo axes o</sub>f h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>3<sub>.</sub> If th<sub>e agen</sub>t bid<sub>s</sub> it<sub>s</sub> t<sub>rue va</sub>l<sub>ua</sub>ti<sub>on,</sub> $b _ { i } ( s _ { t } ) = \nu _ { i } ( s _ { t } )$ , <sup>it</sup>s ex-pos<sup>t</sup> <sub>u</sub>tilit<sub>y</sub> i<sub>s zero.</sub> C<sub>onsequen</sub>tl<sub>y, a ra</sub>ti<sub>ona</sub>l bidd<sub>er mus</sub>t <sub>s</sub>h<sub>a</sub>d<sub>e</sub> it<sub>s</sub> bid b<sub>e</sub>l<sub>ow</sub> it<sub>s pr</sub>i<sub>va</sub>t<sub>e va</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>o secure a</sub> <sub>p</sub>ositive ex<sub>p</sub>ected sur<sub>p</sub>lus (Krishna, 2009). We <sub>p</sub>arameterize the biddin<sub>g p</sub>olic<sub>y</sub> via a state-de<sub>p</sub>endent <sub>mu</sub>lti<sub>p</sub>li<sub>ca</sub>ti<sub>ve s</sub>h<sub>a</sub>di<sub>ng</sub> f<sub>ac</sub>t<sub>or,</sub> $\xi ( s _ { t } ) \in [ 0 , 1 )$ <sub>, y</sub>i<sub>e</sub>ldi<sub>ng</sub> th<sub>e s</sub>h<sub>a</sub>d<sub>e</sub>d bid $b _ { i } ( s _ { t } ) = \xi ( s _ { t } ) \nu _ { i } ( s _ { t } )$ . The ex-<sub>p</sub>ost <sub>surp</sub>l<sub>us</sub> <sub>upon</sub> <sub>w</sub>i<sub>nn</sub>i<sub>ng</sub> i<sub>s</sub> th<sub>us</sub> $( 1 - \xi ( s _ { t } ) ) \nu _ { i } ( s _ { t } )$ <sub>,</sub> hi<sub>g</sub>hli<sub>g</sub>hti<sub>ng</sub> th<sub>e</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l t<sub>ra</sub>d<sub>e-o</sub>f <sub>o</sub>f 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce</sub> <sub>auc</sub>ti<sub>ons:</sub> l<sub>ower</sub>i<sub>ng</sub> $\xi ( s _ { t } )$ <sub>w</sub>id<sub>ens</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>pro</sub>fit <sub>marg</sub>i<sub>n</sub> b<sub>u</sub>t <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> d<sub>ecreases</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>w</sub>i<sub>nn</sub>i<sub>ng.</sub> D<sub>esp</sub>it<sub>e</sub> thi<sub>s</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> <sub>s</sub>h<sub>a</sub>di<sub>ng,</sub> <sub>recen</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce</sub> <sub>agen</sub>ti<sub>c</sub> <sub>mar</sub>k<sub>e</sub>t<sub>s</sub> <sub>s</sub>till achieve hi<sub>g</sub>h welfare whenever com<sub>p</sub>etition is suficientl<sub>y</sub> stron<sub>g</sub> (Trivedi et al., 2026).

## 4. Bidding Strategies

C<sub>ruc</sub>i<sub>a</sub>ll<sub>y, our</sub> f<sub>ramewor</sub>k d<sub>ecoup</sub>l<sub>es</sub> th<sub>e agen</sub>t’<sub>s core capa</sub>bilit<sub>y</sub> f<sub>rom</sub> it<sub>s mar</sub>k<sub>e</sub>t <sub>s</sub>t<sub>ra</sub>t<sub>egy.</sub> Th<sub>e un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> LLM<sub>s respons</sub>ibl<sub>e</sub> f<sub>or</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>iti<sub>ons</sub> $( s _ { t } \to s _ { t + 1 } )$ <sub>are u</sub>tili<sub>ze</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> f<sub>or</sub> i<sub>n</sub>f<sub>erence an</sub>d <sub>rema</sub>i<sub>n</sub> f<sub>rozen.</sub> Th<sub>e re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng process so</sub>l<sub>e</sub>l<sub>y</sub> t<sub>ra</sub>i<sub>ns a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht MLP f<sub>or eac</sub>h <sub>agen</sub>t<sub>, w</sub>hi<sub>c</sub>h <sub>ac</sub>t<sub>s</sub> <sub>as</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e-</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>,</sub> <sub>myop</sub>i<sub>c</sub> biddi<sub>ng</sub> <sub>po</sub>li<sub>cy.</sub> T<sub>o</sub> <sub>surv</sub>i<sub>ve</sub> i<sub>n</sub> <sub>an</sub> i<sub>mper</sub>f<sub>ec</sub>tl<sub>y</sub> <sub>compe</sub>titi<sub>ve</sub> <sub>mar</sub>k<sub>e</sub>t<sub>,</sub> <sub>an</sub> <sub>agen</sub>t’<sub>s</sub> MLP <sub>mus</sub>t <sub>so</sub>l<sub>ve</sub> t<sub>wo</sub> <sub>coup</sub>l<sub>e</sub>d l<sub>earn</sub>i<sub>ng</sub> <sub>pro</sub>bl<sub>ems</sub> <sub>a</sub>t <sub>eac</sub>h <sub>s</sub>t<sub>ep:</sub> <sub>accura</sub>t<sub>e</sub>l<sub>y</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>’<sub>s</sub> t<sub>rue va</sub>l<sub>ua</sub>ti<sub>on</sub> $\upsilon _ { i } { \left( s _ { t } \right) }$ <sub>, an</sub>d d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>s</sub>h<sub>a</sub>d<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> $\xi ( s _ { t } ) \in ( 0 , 1 )$ t<sub>o max</sub>i<sub>m</sub>i<sub>ze</sub> it<sub>s expec</sub>t<sub>e</sub>d <sub>pro</sub>fit <sub>marg</sub>i<sub>n.</sub> Th<sub>e agen</sub>t’<sub>s</sub> fi<sub>na</sub>l bid i<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>as</sub> $b _ { i } ( s _ { t } ) = \xi ( s _ { t } ) \cdot \nu _ { i } ( s _ { t } )$ . For Vickre<sub>y</sub> and CCV<sub>,</sub> we <sub>en</sub>f<sub>orce</sub> t<sub>ru</sub>thf<sub>u</sub>l biddi<sub>ng,</sub> $\xi ( s _ { t } ) = 1$ <sub>.</sub> All <sub>agen</sub>t<sub>s s</sub>h<sub>are a common neura</sub>l <sub>arc</sub>hit<sub>ec</sub>t<sub>ure</sub> f<sub>or</sub> th<sub>e va</sub>l<sub>ua</sub>ti<sub>on</sub> h<sub>ea</sub>d<sub>, w</sub>hi<sub>c</sub>h <sub>pre</sub>di<sub>c</sub>t<sub>s</sub> b<sub>o</sub>th th<sub>e expec</sub>t<sub>e</sub>d d<sub>owns</sub>t<sub>ream va</sub>l<sub>ue</sub> $\upsilon \left( s _ { t } \right)$ <sub>an</sub>d it<sub>s s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> $\sigma ( s _ { t } )$ <sub>.</sub> Th<sub>e</sub> valuation head is trained via a standard Gaussian ne<sub>g</sub>ative lo<sub>g</sub>-likelihood (NLL) loss to re<sub>g</sub>ress toward th<sub>e</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l t<sub>arge</sub>t $P _ { \mathrm { i n } } ^ { ( t ) } - c _ { i , t }$ <sub>.</sub> I<sub>n</sub> <sub>w</sub>h<sub>a</sub>t f<sub>o</sub>ll<sub>ows,</sub> <sub>we</sub> d<sub>rop</sub> <sub>exp</sub>li<sub>c</sub>it d<sub>epen</sub>d<sub>ence</sub> <sub>o</sub>f f<sub>unc</sub>ti<sub>ons</sub> <sub>on</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { t }$ b<sub>u</sub>t thi<sub>s</sub> i<sub>s</sub> i<sub>mp</sub>li<sub>e</sub>d<sub>.</sub> W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> f<sub>our</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>approac</sub>h<sub>es</sub> f<sub>or compu</sub>ti<sub>ng s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> bid<sub>s</sub> i<sub>n</sub> 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce</sub> <sub>auc</sub>ti<sub>ons:</sub>

Mirror Descent The Mirror Descent (MD) strategy frames bid-shading as an online learning problem under binar<sub>y</sub> feedback (win or loss). Let $\xi ^ { * }$ <sub>represen</sub>t th<sub>e uno</sub>b<sub>serve</sub>d <sub>op</sub>ti<sub>ma</sub>l <sub>s</sub>h<sub>a</sub>d<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld th<sub>a</sub>t <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>compe</sub>ti<sub>ng</sub> bid<sub>.</sub> Th<sub>e</sub> <sub>agen</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>e</sub> <sub>pr</sub>i<sub>ma</sub>l l<sub>oss</sub> $\ell _ { t } ( \xi )$ <sub>as</sub> th<sub>e</sub> <sub>regre</sub>t <sub>o</sub>f th<sub>e</sub> <sub>agen</sub>t’<sub>s</sub> <sub>marg</sub>i<sub>n</sub> di<sub>s</sub>t<sub>ance</sub> f<sub>rom</sub> thi<sub>s</sub> b<sub>oun</sub>d<sub>ary: regre</sub>t f<sub>rom over-</sub>biddi<sub>ng</sub> if th<sub>e auc</sub>ti<sub>on was won, an</sub>d <sub>regre</sub>t from under-bidding if lost. Using an entropic regularizer, the proximal projection yields a closed-form multi<sub>p</sub>licative wei<sub>g</sub>hts u<sub>p</sub>date for the tar<sub>g</sub>et shade factor (Beck and Teboulle, 2003):

$$
\xi _ { \mathrm { t a r g e t } } = { \left\{ \begin{array} { l l } { \xi \cdot e ^ { - \eta } } & { { \mathrm { i f ~ } } \mathrm { w o n } } \\ { \xi \cdot e ^ { + \eta } } & { { \mathrm { i f ~ } } \mathrm { l o s t } } \end{array} \right. }\tag{3}
$$

Th<sub>e</sub> d<sub>ynam</sub>i<sub>c s</sub>t<sub>ep s</sub>i<sub>ze</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> $\eta = \hat { \sigma }$ <sub>w</sub>h<sub>ere �</sub>ˆ i<sub>s</sub> th<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on o</sub>f th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>’<sub>s</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> d<sub>escr</sub>ib<sub>e</sub>d <sub>a</sub>b<sub>ove.</sub> Thi<sub>s</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub>d $\xi _ { \mathrm { t a r g e t } }$ <sub>serves as a square</sub>d<sub>-error regress</sub>i<sub>on</sub> t<sub>arge</sub>t f<sub>or</sub> th<sub>e</sub> <sub>agen</sub>t’<sub>s</sub> MLP<sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>v</sub>i<sub>a gra</sub>di<sub>en</sub>t d<sub>escen</sub>t<sub>.</sub>

Thompson Sampling Rather than maintaining a point estimate of the shade factor, the Thompson Sam<sub>p</sub>lin<sub>g</sub> (TS) a<sub>g</sub>ent models the com<sub>p</sub>etitive bid threshold $( \xi ^ { * } )$ <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>us</sub>i<sub>ng</sub> <sub>a</sub> l<sub>og</sub>it<sub>-norma</sub>l <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on w</sub>ith $N ( 0 , 1 )$ l<sub>og</sub>it <sub>pr</sub>i<sub>or.</sub> At i<sub>n</sub>f<sub>erence</sub> ti<sub>me,</sub> TS <sub>samp</sub>l<sub>es a</sub> th<sub>res</sub>h<sub>o</sub>ld di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> thi<sub>s pos</sub>t<sub>er</sub>i<sub>or</sub> t<sub>o</sub> t<sub>arge</sub>t th<sub>e m</sub>i<sub>n</sub>i<sub>mum w</sub>i<sub>nn</sub>i<sub>ng</sub> bid <sub>an</sub>d <sub>max</sub>i<sub>m</sub>i<sub>ze</sub> it<sub>s rea</sub>li<sub>ze</sub>d <sub>pro</sub>fit <sub>marg</sub>i<sub>n:</sub>

$$
z \sim N ( \mu _ { \ell } , \sigma _ { \ell } ^ { 2 } ) , \quad \xi = \mathrm { s i g m o i d } ( z ) .\tag{4}
$$

Th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>v</sub>i<sub>a an</sub> NLL l<sub>oss con</sub>diti<sub>one</sub>d <sub>on</sub> th<sub>e</sub> bi<sub>nary auc</sub>ti<sub>on ou</sub>t<sub>come.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>w</sub>h<sub>en</sub> th<sub>e</sub> t<sub>rue</sub> th<sub>res</sub>h<sub>o</sub>ld $\xi ^ { * } > 1$ (a losin<sub>g</sub> a<sub>g</sub>ent), the NLL u<sub>p</sub>dates <sub>p</sub>ush $\mu _ { \ell } \to \infty _ { : }$ <sub>, so</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub>d <sub>s</sub>h<sub>a</sub>d<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> <sub>converges</sub> t<sub>o</sub> $\xi \to 1 ( b _ { i } \to \nu _ { i } )$ <sub>w</sub>hil<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>e</sub>l<sub>ow</sub> 1<sub>.</sub>

Bayes-Optimal Baseline Instead of attempting to bid precisely at a sampled threshold, we include a Ba<sub>y</sub>es-O<sub>p</sub>timal baseline (Ba<sub>y</sub>es O<sub>p</sub>t) that hed<sub>g</sub>es across the full uncertaint<sub>y</sub> of the threshold distribution. Utili<sub>z</sub>i<sub>ng</sub> th<sub>e same</sub> l<sub>og</sub>it<sub>-norma</sub>l th<sub>res</sub>h<sub>o</sub>ld <sub>pos</sub>t<sub>er</sub>i<sub>or as</sub> Th<sub>ompson</sub> S<sub>amp</sub>li<sub>ng,</sub> it <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> th<sub>e s</sub>h<sub>a</sub>d<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> t<sub>o max</sub>i<sub>m</sub>i<sub>ze expec</sub>t<sub>e</sub>d <sub>pro</sub>fit <sub>marg</sub>i<sub>n un</sub>d<sub>er</sub> it<sub>s curren</sub>t b<sub>e</sub>li<sub>e</sub>f<sub>s:</sub>

$$
\xi = \arg \operatorname* { m a x } _ { \xi ^ { \prime } \in ( 0 , 1 ) } ( 1 - \xi ^ { \prime } ) \cdot \Phi \left( \frac { \log \mathrm { i t } ( \xi ^ { \prime } ) - \mu _ { \ell } } { \sigma _ { \ell } } \right)\tag{5}
$$

<sub>w</sub>h<sub>ere</sub> $( 1 - \xi ^ { \prime } )$ re<sub>p</sub>resents the <sub>p</sub>ro<sub>p</sub>ortional <sub>p</sub>rofit mar<sub>g</sub>in if the auction is won, and Φ(·) is the <sub>p</sub>robabilit<sub>y</sub> of winnin<sub>g</sub> under the <sub>p</sub>osterior model of the com<sub>p</sub>etitive bid threshold from e<sub>q</sub>. (4).

Proximal Policy Optimization (PPO) Finally, we evaluate a standard deep reinforcement learning a<sub>pp</sub>roach as in (Chan<sub>g</sub> et al., 2020). The PPO <sub>p</sub>olic<sub>y</sub> network out<sub>p</sub>uts <sub>p</sub>arameters $\alpha , \beta \geq 1$ t<sub>o</sub> d<sub>e</sub>fi<sub>ne a</sub> <sub>un</sub>i<sub>mo</sub>d<sub>a</sub>l B<sub>e</sub>t<sub>a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> f<sub>rom w</sub>hi<sub>c</sub>h th<sub>e s</sub>h<sub>a</sub>d<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> i<sub>s samp</sub>l<sub>e</sub>d<sub>:</sub> $\xi \sim \operatorname { B e t a } ( \alpha , \beta )$ <sub>.</sub> Th<sub>e po</sub>li<sub>cy</sub> i<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d t<sub>o max</sub>i<sub>m</sub>i<sub>ze</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>pro</sub>fit $U _ { i , t }$ <sub>�</sub> using the clipped surrogate objective:

$$
\mathcal { L } _ { \mathrm { P P O } } = - \operatorname* { m i n } \left( \rho _ { t } A _ { t } , ~ \mathrm { c l i p } ( \rho _ { t } , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) - \beta _ { \mathrm { e n t } } \mathcal { H }\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> $\rho _ { t }$ i<sub>s</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y ra</sub>ti<sub>o</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e curren</sub>t <sub>an</sub>d <sub>o</sub>ld <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> $\epsilon = 0 . 2 , A _ { t } = U _ { i , t } - \bar { U } _ { i , \mathrm { e m a } }$ i<sub>s</sub> th<sub>e</sub> <sub>pro</sub>fit <sub>a</sub>d<sub>van</sub>t<sub>age over a mov</sub>i<sub>ng average</sub> b<sub>ase</sub>li<sub>ne</sub> $( \bar { U } _ { i , \mathrm { { e m a } } } = 0 . 9 9 \bar { U } _ { i , \mathrm { { e m a } } } + 0 . 0 1 U _ { i , t } )$ , and H is an entro<sub>py</sub> <sup>b</sup>o<sup>n</sup>us <sup>t</sup>o p<sup>r</sup>e<sup>v</sup>e<sup>nt</sup> p<sup>r</sup>e<sup>m</sup>a<sup>t</sup>u<sup>r</sup>e co<sup>nv</sup>e<sup>r</sup>ge<sup>n</sup>ce $( \beta _ { e n t } = 0 . 0 1 )$ .

State Representation and Pre-Processing Because the raw environmental states in the LLM d<sub>oma</sub>i<sub>ns are non-vec</sub>t<sub>or</sub> t<sub>ex</sub>t <sub>pay</sub>l<sub>oa</sub>d<sub>s,</sub> th<sub>ey mus</sub>t b<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d b<sub>e</sub>f<sub>ore an agen</sub>t’<sub>s</sub> t<sub>ra</sub>i<sub>na</sub>bl<sub>e</sub> MLP <sub>can</sub> <sub>process</sub> th<sub>em</sub> t<sub>o</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e a</sub> bid<sub>.</sub> T<sub>o</sub> b<sub>r</sub>id<sub>ge raw</sub> t<sub>ex</sub>t <sub>an</sub>d <sub>po</sub>li<sub>cy</sub> i<sub>npu</sub>t<sub>s, we emp</sub>l<sub>oy</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> re<sub>p</sub>resentat<sup>i</sup>on <sub>p</sub>ara<sup>di</sup><sub>g</sub>ms:

1. Dense Embeddings (Sec 5.1): The raw text is first embedded, e.g., utilizing Gecko (Lee et al., 2024), before bein<sub>g</sub> <sub>p</sub>assed to a trainable MLP alon<sub>g</sub>side a<sub>g</sub>ent identit<sub>y</sub> features.

2. Centralized Assessor (Sec 5.2): To aid cold-starting the market and decouple routing signals from economic bidding / pricing, a centralized assessor LLM evaluates and compresses the task context into a discrete 0–10 relevance score for each a<sub>g</sub>ent. A<sub>g</sub>ent MLPs o<sub>p</sub>erate directl<sub>y</sub> on these one-hot <sub>enco</sub>d<sub>e</sub>d <sub>scores, e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y re</sub>d<sub>uc</sub>i<sub>ng comp</sub>l<sub>ex rou</sub>ti<sub>ng</sub> t<sub>o a</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>a</sub>b<sub>u</sub>l<sub>ar</sub> MDP <sub>v</sub>i<sub>a p</sub>l<sub>a</sub>tf<sub>orm-s</sub>id<sub>e</sub> f<sub>ea</sub>t<sub>ure</sub> <sub>eng</sub>i<sub>neer</sub>i<sub>ng.</sub>

## 5. Empirical Evaluation

T<sub>o emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y va</sub>lid<sub>a</sub>t<sub>e</sub> th<sub>e necess</sub>it<sub>y o</sub>f i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on an</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>e our</sub> biddi<sub>ng s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es, we</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>our mar</sub>k<sub>e</sub>t <sub>across a syn</sub>th<sub>e</sub>ti<sub>c con</sub>t<sub>ro</sub>l d<sub>oma</sub>i<sub>n an</sub>d th<sub>ree</sub> di<sub>verse</sub> LLM <sub>env</sub>i<sub>ronmen</sub>t<sub>s.</sub>

## 5.1. Validating the Monopsony Failure (Synthetic Chain Translation)

W<sub>e</sub> fi<sub>rs</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> f<sub>ramewor</sub>k <sub>on</sub> <sub>a</sub> <sub>syn</sub>th<sub>e</sub>ti<sub>c</sub> <sub>mu</sub>lti<sub>-s</sub>t<sub>ep</sub> t<sub>rans</sub>l<sub>a</sub>ti<sub>on</sub> d<sub>oma</sub>i<sub>n.</sub> I<sub>n</sub> thi<sub>s</sub> <sub>env</sub>i<sub>ronmen</sub>t<sub>,</sub> <sub>s</sub>t<sub>a</sub>t<sub>es are cons</sub>t<sub>ruc</sub>t<sub>e</sub>d b<sub>y conca</sub>t<sub>ena</sub>ti<sub>ng</sub> t<sub>wo</sub> 64<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ngs, one</sub> id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l t<sub>as</sub>k <sub>an</sub>d th<sub>e o</sub>th<sub>er</sub> th<sub>e curren</sub>t l<sub>anguage.</sub> A<sub>gen</sub>t<sub>s ac</sub>t <sub>as</sub> bili<sub>ngua</sub>l t<sub>rans</sub>l<sub>a</sub>t<sub>ors s</sub>t<sub>a</sub>ti<sub>one</sub>d <sub>on</sub> th<sub>e e</sub>d<sub>ges o</sub>f a <sub>g</sub>ra<sub>p</sub>h (Fi<sub>g</sub>ure 1a), <sub>p</sub>ossessin<sub>g</sub> distinct inference costs $c _ { i } .$ <sub>.</sub> Th<sub>e mar</sub>k<sub>e</sub>t <sub>mus</sub>t <sub>rou</sub>t<sub>e pay</sub>l<sub>oa</sub>d<sub>s</sub> f<sub>rom a</sub> source node to tar<sub>g</sub>et nodes with terminal reward \$1 b<sub>y</sub> discoverin<sub>g</sub> the most cost-eficient to<sub>p</sub>olo<sub>g</sub>ical sequence (e.g., <sup>b</sup>ypass<sup>i</sup>ng t<sup>h</sup>e expens<sup>i</sup>ve \$0.50 <sup>di</sup>rect Fr→En e<sup>d</sup>ge <sup>i</sup>n <sup>f</sup>avor o<sup>f</sup> t<sup>h</sup>e \$0.02 Fr→Sp→En <sub>p</sub>ath). We evaluate all three <sub>p</sub>ro<sub>p</sub>osed auctions: 1st-<sub>p</sub>rice, 2nd-<sub>p</sub>rice, and CCV.

Thi<sub>s</sub> d<sub>oma</sub>i<sub>n a</sub>ll<sub>ows us</sub> t<sub>o</sub> t<sub>es</sub>t th<sub>e ex</sub>t<sub>remes o</sub>f <sub>mar</sub>k<sub>e</sub>t <sub>compe</sub>titi<sub>on.</sub> W<sub>e con</sub>fi<sub>gure</sub> th<sub>e env</sub>i<sub>ronmen</sub>t with a single agent per edge (� = 1). In this regime, intermediate nodes frequently act as monopsonies. This occurs either structurall<sub>y</sub> (e.<sub>g</sub>., the Sp has onl<sub>y</sub> one out<sub>g</sub>oin<sub>g</sub> ed<sub>g</sub>e) or task-conditionall<sub>y</sub> (e.<sub>g</sub>., Ge <sup>h</sup>as two e<sup>d</sup>ges, <sup>b</sup>ut on<sup>l</sup>y one is via<sup>bl</sup>e <sup>f</sup>or a given tas<sup>k</sup>, ren<sup>d</sup>ering t<sup>h</sup>e competitor<sup>’</sup>s <sup>b</sup>i<sup>d</sup> zero).

Th<sub>e presence o</sub>f <sub>an uncon</sub>t<sub>es</sub>t<sub>e</sub>d b<sub>uyer exposes</sub> th<sub>e</sub> f<sub>a</sub>il<sub>ure mo</sub>d<sub>es o</sub>f <sub>eac</sub>h <sub>auc</sub>ti<sub>on</sub> t<sub>ype.</sub> I<sub>n a</sub> standard 2nd-<sub>p</sub>rice (Vickre<sub>y</sub>) auction, the mono<sub>p</sub>sonist bids truthfull<sub>y</sub> but <sub>p</sub>a<sub>y</sub>s the floor (0); the platform must inject unacceptable subsidies (e.g., \$0.88 at Sp) to pay the upstream seller the first <sub>p</sub>rice. Under CCV<sub>,</sub> the bu<sub>y</sub>er <sub>p</sub>a<sub>y</sub>s 0 and the u<sub>p</sub>stream seller receives 0. Under a 1st-<sub>p</sub>rice auction<sub>,</sub> the <sub>monopson</sub>i<sub>s</sub>t <sub>ac</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>s</sub>h<sub>a</sub>d<sub>es</sub> it<sub>s</sub> bid d<sub>own</sub> t<sub>o</sub> th<sub>e</sub> fl<sub>oor,</sub> <sub>pass</sub>i<sub>ng</sub> <sub>a</sub> <sub>van</sub>i<sub>s</sub>hi<sub>ng</sub>l<sub>y</sub> <sub>sma</sub>ll <sub>va</sub>l<sub>ue</sub> <sub>ups</sub>t<sub>ream.</sub>

![](images/0576cf50a6087b767c1223f74dce55ca6fb5220dca638a0c1d51a40286a99e5f.jpg)  
(a) Domain <sub>g</sub>ra<sub>p</sub>h & bids

![](images/06c8dd3cc535dac3f537b6338b9baeedb33e4de24a67ca62cd7082a9ee272361.jpg)  
(b) Success rate over trainin<sub>g</sub>  
Fi<sub>g</sub>ure 1 | Translation Domain Validation: (a) The task <sub>g</sub>ra<sub>p</sub>h showin<sub>g</sub> inference costs (black) and learned Vickre<sub>y</sub> bids (blue, italic; mean ± std dev). Downstream bids reflect task sam<sub>p</sub>lin<sub>g p</sub>robabilities (e.g., En→Ja expects 0.48 <sup>b</sup>ecause Tas<sup>k</sup> 1 <sup>i</sup>s samp<sup>l</sup>e<sup>d</sup> $2 / 3$ o<sup>f</sup> t<sup>h</sup>e t<sup>i</sup>me). T<sup>h</sup>e upstream Sp→En <sup>bid</sup> (0.88) accuratel<sub>y</sub> a<sub>pp</sub>roximates the downstream ex<sub>p</sub>ected value minus local costs $( 0 . 4 8 + 0 . 3 8 - 0 . 0 1 =$ 0.85). (b) Success rate increases and stabilizes across all auctions under increasin<sub>g</sub> market densit<sub>y</sub>. C<sub>ompe</sub>tit<sub>ors</sub> <sub>are</sub> dif<sub>eren</sub>ti<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> <sub>cos</sub>t $c = c _ { b } ( 1 + t )$ <sub>w</sub>ith $t \sim U ( - 0 . 5 , 0 . 5 )$ <sub>an</sub>d b<sub>ase</sub> <sub>cos</sub>t<sub>s</sub> $c _ { b }$ li<sub>s</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> grap<sup>h</sup>. Curves are means over $3$ inde<sub>p</sub>endent trials and shaded re<sub>g</sub>ions s<sub>p</sub>an ±1 standard deviation; b<sub>o</sub>th <sub>are compu</sub>t<sub>e</sub>d <sub>on a</sub> t<sub>ra</sub>ili<sub>ng</sub> 101<sub>-ep</sub>i<sub>so</sub>d<sub>e ro</sub>lli<sub>ng average.</sub>

F<sub>or</sub> CCV <sub>an</sub>d 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce</sub> <sub>auc</sub>ti<sub>ons,</sub> thi<sub>s</sub> f<sub>a</sub>il<sub>ure</sub> t<sub>o</sub> <sub>pass</sub> <sub>va</sub>l<sub>ue</sub> b<sub>ac</sub>k<sub>war</sub>d t<sub>r</sub>i<sub>ggers</sub> <sub>a</sub> <sub>casca</sub>di<sub>ng</sub> <sub>ne</sub>t<sub>wor</sub>k co<sup>ll</sup>apse. Upstream agents (e.g., Fr→Sp an<sup>d</sup> En→Ge) rece<sup>i</sup>ve $P _ { \mathrm { i n } } \approx 0 .$ <sub>,</sub> f<sub>a</sub>il t<sub>o o</sub>f<sub>se</sub>t th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>f<sub>erence</sub> costs, an<sup>d</sup> go <sup>b</sup>an<sup>k</sup>rupt. T<sup>h</sup>e<sup>i</sup>r ex<sup>i</sup>t removes compet<sup>i</sup>t<sup>i</sup>on at ear<sup>li</sup>er no<sup>d</sup>es (e.g., t<sup>h</sup>e En no<sup>d</sup>e <sup>l</sup>oses t<sup>h</sup>e En→Ge <sup>b</sup>uyer), turn<sup>i</sup>ng t<sup>h</sup>ose prece<sup>di</sup>ng no<sup>d</sup>es <sup>i</sup>nto new tas<sup>k</sup>-con<sup>di</sup>t<sup>i</sup>one<sup>d</sup> monopson<sup>i</sup>es t<sup>h</sup>at a<sup>l</sup>so <sup>d</sup>rop th<sub>e</sub>i<sub>r</sub> bid<sub>s</sub> t<sub>o</sub> <sub>zero.</sub> B<sub>ecause</sub> <sub>every</sub> <sub>e</sub>d<sub>ge</sub> i<sub>ncurs</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> i<sub>n</sub>f<sub>erence</sub> <sub>cos</sub>t<sub>,</sub> thi<sub>s</sub> b<sub>ac</sub>k<sub>war</sub>d <sub>unrave</sub>li<sub>ng</sub> i<sub>nev</sub>it<sub>a</sub>bl<sub>y s</sub>t<sub>arves every</sub> b<sub>ranc</sub>h <sub>o</sub>f th<sub>e ne</sub>t<sub>wor</sub>k<sub>, guaran</sub>t<sub>ee</sub>i<sub>ng</sub> t<sub>o</sub>t<sub>a</sub>l <sub>mar</sub>k<sub>e</sub>t f<sub>a</sub>il<sub>ure.</sub>

When we transition to an imperfect competition regime by populating each edge with $N > 1$ <sub>agen</sub>t<sub>s possess</sub>i<sub>ng s</sub>li<sub>g</sub>ht <sub>cos</sub>t <sub>var</sub>i<sub>ances,</sub> th<sub>e monopson</sub>i<sub>es are</sub> b<sub>ro</sub>k<sub>en.</sub> C<sub>ompe</sub>titi<sub>ve pressure</sub> f<sub>orces</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> b<sub>uyers</sub> t<sub>o ra</sub>i<sub>se</sub> th<sub>e</sub>i<sub>r</sub> bid<sub>s a</sub>b<sub>ove</sub> th<sub>e</sub> fl<sub>oor.</sub> V<sub>a</sub>l<sub>ue success</sub>f<sub>u</sub>ll<sub>y</sub> fl<sub>ows</sub> b<sub>ac</sub>k<sub>war</sub>d th<sub>roug</sub>h th<sub>e</sub> <sub>g</sub>ra<sub>p</sub>h<sub>,</sub> enablin<sub>g</sub> u<sub>p</sub>stream a<sub>g</sub>ents to sustain <sub>p</sub>ositive <sub>p</sub>rofit mar<sub>g</sub>ins and restorin<sub>g</sub> the o<sub>p</sub>timal \$0.04 Fr→Sp→En→Ko route w<sup>i</sup>t<sup>h</sup>out requ<sup>i</sup>r<sup>i</sup>ng su<sup>b</sup>s<sup>idi</sup>es. We <sup>di</sup>scuss t<sup>h</sup>e poor per<sup>f</sup>ormance o<sup>f</sup> TS an<sup>d</sup> Ba<sub>y</sub>es-O<sub>p</sub>timal baselines with �=2 in A<sub>pp</sub>endix C. All methods succeed with �=4 (A<sub>pp</sub>endix A).

C<sub>ruc</sub>i<sub>a</sub>ll<sub>y,</sub> thi<sub>s exper</sub>i<sub>men</sub>t <sub>va</sub>lid<sub>a</sub>t<sub>es our auc</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>cs un</sub>d<sub>er</sub> i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on.</sub> B<sub>ecause</sub> standard 2nd-<sub>p</sub>rice (Vickre<sub>y</sub>) auctions re<sub>q</sub>uire theoreticall<sub>y</sub> unbounded <sub>p</sub>latform subsidies, and CCV <sub>p</sub>asses insuficient value to u<sub>p</sub>stream a<sub>g</sub>ents when com<sub>p</sub>etition is im<sub>p</sub>erfect (often starvin<sub>g</sub> them of the revenue needed to cover inference costs), we <sub>p</sub>rioritize the sim<sub>p</sub>ler 1st-Price auction—where a<sub>g</sub>ents <sub>ac</sub>ti<sub>ve</sub>l<sub>y</sub> l<sub>earn</sub> bid <sub>s</sub>h<sub>a</sub>di<sub>ng—as</sub> th<sub>e</sub> d<sub>e</sub>f<sub>au</sub>lt <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> f<sub>or su</sub>b<sub>sequen</sub>t <sub>exper</sub>i<sub>men</sub>t<sub>s.</sub>

## 5.2. LLM Agent Economies

H<sub>av</sub>i<sub>ng va</sub>lid<sub>a</sub>t<sub>e</sub>d th<sub>e econom</sub>i<sub>c mec</sub>h<sub>an</sub>i<sub>cs, we sca</sub>l<sub>e</sub> th<sub>e</sub> f<sub>ramewor</sub>k t<sub>o</sub> di<sub>verse</sub> LLM d<sub>oma</sub>i<sub>ns:</sub> N<sub>ews</sub> Market (information s<sub>y</sub>nthesis) and Omni-MATH (mathematical reasonin<sub>g</sub>). We also examine HotpotQA in Appendix A. These domains test the market’s ability to orchestrate multi-step cognitive <sub>wor</sub>kfl<sub>ows.</sub> T<sub>o</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on,</sub> <sub>agen</sub>t<sub>s</sub> <sub>are</sub> dif<sub>eren</sub>ti<sub>a</sub>t<sub>e</sub>d <sub>a</sub>l<sub>ong</sub> th<sub>ree</sub> <sub>axes:</sub> d<sub>ynam</sub>i<sub>c</sub> identit<sub>y</sub> embeddin<sub>g</sub>s, as<sub>y</sub>mmetric data access, and distinct s<sub>y</sub>stem <sub>p</sub>rom<sub>p</sub>ts (full <sub>p</sub>rom<sub>p</sub>ts are detailed in Appendix D). Unless otherwise noted, all agents share a frozen Gemini-3.5-flash-lite backbone. I<sub>n</sub>f<sub>erence cos</sub>t <sub>pr</sub>i<sub>c</sub>i<sub>ng</sub> i<sub>s ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng rea</sub>l i<sub>npu</sub>t/<sub>ou</sub>t<sub>pu</sub>t t<sub>o</sub>k<sub>en cos</sub>t<sub>s.</sub>

![](images/dedff790eadbdfcb54408c92213e4b4a289a45c2abce98dfbd26bcae555c9b8e.jpg)  
Figure 2 | Economic Viability on News Market: The decentralized market successfully sustains social <sub>we</sub>lf<sub>are w</sub>hil<sub>e</sub> d<sub>e</sub>li<sub>ver</sub>i<sub>ng s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y pos</sub>iti<sub>ve cumu</sub>l<sub>a</sub>ti<sub>ve pro</sub>fit<sub>s</sub> t<sub>o</sub> b<sub>o</sub>th i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>agen</sub>t<sub>s an</sub>d th<sub>e p</sub>l<sub>a</sub>tf<sub>orm.</sub> Th<sub>e per-agen</sub>t <sub>cumu</sub>l<sub>a</sub>ti<sub>ve pro</sub>fit i<sub>s repor</sub>t<sub>e</sub>d f<sub>or</sub> Mi<sub>rror</sub> D<sub>escen</sub>t<sub>.</sub> M<sub>eans an</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>ons are</sub> <sub>compu</sub>t<sub>e</sub>d <sub>over</sub> 5 t<sub>r</sub>i<sub>a</sub>l<sub>s.</sub>

Asymmetric Data Markets (News Market): We construct the News Market (A endix F) to simulate <sub>a pay-as-you-go researc</sub>h <sub>p</sub>i<sub>pe</sub>li<sub>ne.</sub> C<sub>ompe</sub>titi<sub>on</sub> i<sub>s</sub> d<sub>r</sub>i<sub>ven</sub> b<sub>y</sub> b<sub>o</sub>th <sub>asymme</sub>t<sub>r</sub>i<sub>c</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on an</sub>d <sub>query</sub> strategy. The market contains four publisher categories (wikipedia, medical journal, social media, news) operating over strictly disjoint document corpora, alongside a sales agent that synthesizes the fi<sub>na</sub>l <sub>answer.</sub> T<sub>o</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h i<sub>mper</sub>f<sub>ec</sub>t <sub>compe</sub>titi<sub>on</sub> <sub>w</sub>ithi<sub>n</sub> <sub>eac</sub>h <sub>ca</sub>t<sub>egory,</sub> <sub>we</sub> d<sub>ep</sub>l<sub>oy</sub> t<sub>wo</sub> <sub>compe</sub>ti<sub>ng</sub> publisher agents per category. These intra-category competitors share the same data access but are diferentiated by distinct search prompts: a precision strategy generating narrow, entity-specific queries versus a coverage strategy casting a broader, gap-filling net.

Th<sub>e mar</sub>k<sub>e</sub>t <sub>success</sub>f<sub>u</sub>ll<sub>y</sub> l<sub>earns</sub> t<sub>o rou</sub>t<sub>e quer</sub>i<sub>es</sub> t<sub>o re</sub>l<sub>evan</sub>t <sub>pu</sub>bli<sub>s</sub>h<sub>ers</sub> b<sub>ase</sub>d <sub>on expec</sub>t<sub>e</sub>d d<sub>own-</sub> stream valuation, achievin<sub>g</sub> a 73.4±1.7% success rate over the final 100 e<sub>p</sub>isodes (fi<sub>g</sub>. 2). Im<sub>p</sub>ortantl<sub>y</sub>, <sub>ac</sub>ti<sub>ve mar</sub>k<sub>e</sub>t <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s ac</sub>hi<sub>eve pos</sub>iti<sub>ve pro</sub>fit <sub>marg</sub>i<sub>ns as we</sub>ll<sub>, overcom</sub>i<sub>ng</sub> th<sub>e par</sub>ti<sub>c</sub>i<sub>pa</sub>ti<sub>on</sub> b<sub>arr</sub>i<sub>er</sub> raised in Section 3.2. In com<sub>p</sub>arison, a centralized ReAct baseline (Yao et al., 2023) limited to the same 8-step horizon achieves 48.4 ± 2.8% with Gemini-3.5-flash-lite at 0.035 ± 0.0004 ¢/task and 78.8 ± 1.3% with Gemini-3.1-pro-preview at 1.060 ± 0.368 ¢/task (1¢=\$0.01). Operating exclusively with Gemini-3.5-flash-lite agents, the market recovers 82% of Gemini-3.1-pro-preview’s accuracy lift over Gemini-3.5-flash-lite ReAct at an agent inference cost of just $0 . 0 5 4 \pm 0 . 0 0 3$ ¢/task (0.102¢ including the market assessor)—a 19.7× cost reduction relative to Gemini-3.1-pro-preview (10.4× total). Although the market takes slightly fewer steps per task than Gemini-3.5-flash-lite ReAct (3.12 vs. 3.23), its hi<sub>g</sub>her retrieval success accumulates richer multi-document contexts for the sales a<sub>g</sub>ent to s<sub>y</sub>nthesize, tradin<sub>g</sub> a minor 0.067 ¢/task increase for a +25.0% accurac<sub>y g</sub>ain.

Bidding Strategy Performance: We found Mirror Descent (MD) to outperform PPO empirically, <sub>p</sub>articularl<sub>y</sub> in the later sta<sub>g</sub>es of trainin<sub>g</sub>. MD decou<sub>p</sub>les the learnin<sub>g</sub> of a) valuations and b) shadin<sub>g</sub>, <sub>p</sub>rovidin<sub>g</sub> a clear si<sub>g</sub>nal to each: a) re<sub>g</sub>ressin<sub>g</sub> valuations toward realized <sub>p</sub>a<sub>y</sub>ofs u<sub>p</sub>on winnin<sub>g</sub>, and b) adjusting the shade factor up or down strictly from the binary loss/win outcome. In contrast, while PPO tracks MD earl<sub>y</sub> in trainin<sub>g</sub> (when shades are initialized near 0.5 and winnin<sub>g</sub> a<sub>g</sub>ents accumulate <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>pro</sub>fit b<sub>ase</sub>li<sub>ne</sub> $\bar { U } _ { i , \mathrm { { e m a } } } > 0 )$ <sub>,</sub> it<sub>s</sub> <sub>per</sub>f<sub>ormance</sub> d<sub>egra</sub>d<sub>es</sub> l<sub>a</sub>t<sub>er</sub> <sub>as</sub> <sub>compe</sub>ti<sub>ng</sub> bid<sub>s</sub> <sub>r</sub>i<sub>se.</sub> Wh<sub>en</sub> <sub>s</sub>tif<sub>er compe</sub>titi<sub>on requ</sub>i<sub>res an agen</sub>t t<sub>o</sub> bid <sub>a</sub> hi<sub>g</sub>h<sub>er s</sub>h<sub>a</sub>d<sub>e</sub> $\xi _ { t } ,$ <sub>exp</sub>l<sub>or</sub>i<sub>ng an</sub> i<sub>ncrease</sub>d <sub>s</sub>h<sub>a</sub>d<sub>e</sub> th<sub>a</sub>t <sub>s</sub>till l<sub>oses</sub> $( U _ { i , t } = 0 )$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>a</sub> <sub>nega</sub>ti<sub>ve</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> $( A _ { i , t } = - \bar { U } _ { i , \mathrm { { e m a } } } < 0 )$ <sub>,</sub> <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>a</sub> <sub>nega</sub>ti<sub>ve</sub> <sub>po</sub>li<sub>cy</sub> <sub>gra</sub>di<sub>en</sub>t that pushes the bid down rather than up; moreover, even when a higher shade wins, the compressed l<sub>a</sub>t<sub>e-s</sub>t<sub>age</sub> <sub>pro</sub>fit <sub>marg</sub>i<sub>n</sub> $( 0 < U _ { i , t } < \bar { U } _ { i , \mathrm { { e m a } } } )$ <sub>a</sub>l<sub>so reg</sub>i<sub>s</sub>t<sub>ers as a nega</sub>ti<sub>ve a</sub>d<sub>van</sub>t<sub>age re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o ear</sub>li<sub>er</sub> <sub>marg</sub>i<sub>ns, caus</sub>i<sub>ng</sub> PPO t<sub>o sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y un</sub>d<sub>er-</sub>bid <sub>as</sub> th<sub>e mar</sub>k<sub>e</sub>t <sub>ma</sub>t<sub>ures.</sub>

Budgeting Reasoning Efort (Omni-MATH). Our second domain tests whether the market can allocate reasoning efort adaptively across mathematical problems. Following the generator/verifier/refiner decom<sub>p</sub>osition of MALT (Motwani et al., 2024), we instantiate three a<sub>g</sub>ents of each role and a sin<sub>g</sub>le <sub>su</sub>b<sub>m</sub>itt<sub>er.</sub> G<sub>enera</sub>t<sub>ors</sub> fi<sub>rs</sub>t bid t<sub>o pro</sub>d<sub>uce a can</sub>did<sub>a</sub>t<sub>e so</sub>l<sub>u</sub>ti<sub>on.</sub> V<sub>er</sub>ifi<sub>ers</sub> th<sub>en</sub> bid <sub>aga</sub>i<sub>ns</sub>t <sub>one ano</sub>th<sub>er</sub> <sub>an</sub>d th<sub>e su</sub>b<sub>m</sub>itt<sub>er</sub> t<sub>o assess</sub> it<sub>;</sub> if th<sub>e su</sub>b<sub>m</sub>itt<sub>er w</sub>i<sub>ns,</sub> th<sub>e ro</sub>ll<sub>ou</sub>t t<sub>erm</sub>i<sub>na</sub>t<sub>es.</sub> Oth<sub>erw</sub>i<sub>se, re</sub>fi<sub>ners</sub> bid <sub>aga</sub>i<sub>ns</sub>t <sub>one ano</sub>th<sub>er an</sub>d th<sub>e su</sub>b<sub>m</sub>itt<sub>er us</sub>i<sub>ng</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e an</sub>d <sub>ver</sub>ifi<sub>ca</sub>ti<sub>on</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>; a w</sub>i<sub>nn</sub>i<sub>ng</sub> <sub>re</sub>fi<sub>ner rep</sub>l<sub>aces</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e an</sub>d <sub>re</sub>t<sub>urns con</sub>t<sub>ro</sub>l t<sub>o</sub> th<sub>e ver</sub>ifi<sub>ca</sub>ti<sub>on s</sub>t<sub>age.</sub> Th<sub>e su</sub>b<sub>m</sub>itt<sub>er</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>ac</sub>t<sub>s as an en</sub>d<sub>ogenous s</sub>t<sub>opp</sub>i<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>: ano</sub>th<sub>er reason</sub>i<sub>ng s</sub>t<sub>ep</sub> i<sub>s purc</sub>h<sub>ase</sub>d <sub>on</sub>l<sub>y w</sub>h<sub>en an agen</sub>t <sub>va</sub>l<sub>ues</sub> f<sub>ur</sub>th<sub>er wor</sub>k <sub>more</sub> hi<sub>g</sub>hl<sub>y</sub> th<sub>an su</sub>b<sub>m</sub>itti<sub>ng</sub> th<sub>e curren</sub>t <sub>answer.</sub>

Table 1 | News Market performance and inference cost per task $\left( 1 \Phi { = } \$ 0.01 \right)$ i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> <sub>assessor.</sub> All market participants use Gemini-3.5-flash-lite. Best market mechanism is shown in bold.
<table><tr><td>Method</td><td>Success Rate (%) Steps / Task</td><td></td><td>Cost / Task (¢)</td></tr><tr><td>ReACT (Gemini-3.5-flash-lite)</td><td> $4 8 . 4 0 \pm 2 . 7 9$ </td><td> $3 . 2 3 \pm 0 . 0 4$ </td><td> $0 . 0 3 4 8 \pm 0 . 0 0 0 4$ </td></tr><tr><td>ReACT (Gemini-3.1-pro-preview)</td><td> $7 8 . 8 0 \pm 1 . 3 0 $ </td><td> $4 . 3 7 \pm 0 . 1 2$ </td><td> $1 . 0 5 9 9 \pm 0 . 3 6 7 9$ </td></tr><tr><td>Market (PPO, 1st-Price)</td><td> $4 0 . 0 0 \pm 2 . 8 3 $ </td><td> $1 . 8 5 \pm 0 . 1 2$ </td><td> $0 . 0 5 2 5 \pm 0 . 0 0 3 6$ </td></tr><tr><td>Market (Mirror Descent, 1st-Price)</td><td> ${ \bf 7 3 . 4 0 \pm 1 . 6 7 }$ </td><td> ${ \bf 3 . 1 2 \pm 0 . 1 1 }$ </td><td> $\mathbf { 0 . 1 0 2 0 \pm 0 . 0 0 4 4 }$ </td></tr></table>

Table 2 | Test set performance on Omni-MATH with PPO. Sequential auctions achieve the highest overall accurac<sub>y</sub>, out<sub>p</sub>erformin<sub>g</sub> a <sub>p</sub>rom<sub>p</sub>ted router at com<sub>p</sub>arable cost (measured in number of refinements, which is monotonic in token use).
<table><tr><td>Method</td><td>Accuracy (se) Cost</td></tr><tr><td>one-shot fixed n = 1</td><td>69.17 (±1.44) 0</td></tr><tr><td>fixed n = 2</td><td>70.54 (±1.42) 1 71.80 (±1.41) 2</td></tr><tr><td>prompted</td><td>72.59 (±1.39) 0.24</td></tr><tr><td>auction</td><td>74.83 (±1.36) 0.28</td></tr></table>

All agents use prompted Gemini-3.5-flash, with three prompts per role (Section D.3). A small multi-la<sub>y</sub>er <sub>p</sub>erce<sub>p</sub>tron (MLP) estimates a<sub>g</sub>ent values from structural features and the out<sub>p</sub>ut of a prompted assessor that scores the current candidates. We train the value estimator and evaluate on held-out Omni-MATH <sub>p</sub>roblems (Gao et al., 2025), usin<sub>g</sub> the domains and s<sub>p</sub>lits from Fisch et al. (2026). As re<sub>p</sub>orted in Table 2, the auction reaches 74.83% accurac<sub>y</sub> with onl<sub>y</sub> 0.28 refinements <sub>p</sub>er <sub>pro</sub>bl<sub>em,</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orm</sub>i<sub>ng</sub> fi<sub>xe</sub>d <sub>one-</sub> <sub>an</sub>d t<sub>wo-re</sub>fi<sub>nemen</sub>t b<sub>ase</sub>li<sub>nes</sub> <sub>an</sub>d <sub>a</sub> <sub>promp</sub>t<sub>e</sub>d <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>orc</sub>h<sub>es</sub>t<sub>ra</sub>t<sub>or</sub> i<sub>nsp</sub>i<sub>re</sub>d b<sub>y</sub> MALT<sub>, w</sub>hil<sub>e</sub> i<sub>ncurr</sub>i<sub>ng compara</sub>bl<sub>e cos</sub>t<sub>.</sub> All <sub>compar</sub>i<sub>sons w</sub>ith th<sub>e auc</sub>ti<sub>on-</sub>b<sub>ase</sub>d <sub>me</sub>th<sub>o</sub>d <sub>are s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>a</sub>t <sub>� < .</sub>05 b<sub>y</sub> M<sub>c</sub>N<sub>emar</sub>’<sub>s</sub> t<sub>es</sub>t<sub>.</sub>

## 6. Related Work

Top-Down Orchestration and Routing. Prior work scales LLM reasoning via external scafolding, includin<sub>g</sub> static <sub>g</sub>ra<sub>p</sub>h/confi<sub>g</sub>uration o<sub>p</sub>timizers like Maestro (Wan<sub>g</sub> et al., 2025), centralized costcascadin<sub>g</sub> routers like Fru<sub>g</sub>alGPT (Chen et al., 2024), and learned small-a<sub>g</sub>ent routin<sub>g</sub> (Alazraki et al., 2026). Unlike our decentralized market, these s<sub>y</sub>stems <sub>p</sub>resu<sub>pp</sub>ose a fixed to<sub>p</sub>olo<sub>gy</sub> or a <sub>g</sub>lobal <sub>p</sub>l<sub>anner w</sub>ith f<sub>u</sub>ll i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> i<sub>gnor</sub>i<sub>ng pr</sub>i<sub>va</sub>t<sub>e opera</sub>ti<sub>ona</sub>l <sub>cos</sub>t<sub>s.</sub>

Decentralized Agent Markets. The idea of building multi-step trajectories from sequential auctions was introduced b<sub>y</sub> Chan<sub>g</sub> et al. (2020), but their Vickre<sub>y</sub>-based framework re<sub>q</sub>uired an awkward “<sub>c</sub>l<sub>on</sub>i<sub>ng</sub>" <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> t<sub>o preserve cre</sub>dit <sub>conserva</sub>ti<sub>on.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce auc</sub>ti<sub>ons can</sub> b<sub>e e</sub>f<sub>ec</sub>ti<sub>ve,</sub> <sub>an</sub>d <sub>we a</sub>d<sub>ap</sub>t th<sub>e</sub>i<sub>r</sub> f<sub>ramewor</sub>k t<sub>o con</sub>t<sub>emporary</sub> LLM<sub>-</sub>b<sub>ase</sub>d <sub>app</sub>li<sub>ca</sub>ti<sub>ons suc</sub>h <sub>as</sub> O<sub>mn</sub>i<sub>-</sub>M<sub>a</sub>th<sub>.</sub> Thi<sub>s</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s w</sub>h<sub>a</sub>t <sub>necess</sub>it<sub>a</sub>t<sub>es</sub> th<sub>e nove</sub>l l<sub>earn</sub>i<sub>ng arc</sub>hit<sub>ec</sub>t<sub>ure</sub> d<sub>escr</sub>ib<sub>e</sub>d i<sub>n sec</sub>ti<sub>ons</sub> 3 <sub>an</sub>d 4<sub>.</sub> M<sub>ore</sub> recentl<sub>y</sub>, Tomasev et al. (2025) <sub>p</sub>ro<sub>p</sub>ose (but do not im<sub>p</sub>lement) coordination of contem<sub>p</sub>orar<sub>y</sub> LLMs through market mechanisms, and specific market-like mechanisms are presented by Qi et al. (2026) and Liu et al. (2026). Our a<sub>pp</sub>roach is distin<sub>g</sub>uished b<sub>y p</sub>ushin<sub>g</sub> the idea of decentralization further. In <sub>p</sub>articular, Qi et al. (2026) maintain a <sub>p</sub>o<sub>p</sub>ulation of a<sub>g</sub>ents that is selected b<sub>y</sub> each a<sub>g</sub>ent’s economic balance (a<sub>g</sub>ents with zero balance dro<sub>p</sub> out), but the a<sub>g</sub>ents are not economic actors: they do not structure their participation to optimize a local economic incentive. Concurrent work b<sub>y</sub> Liu et al. (2026) orchestrates LLMs usin<sub>g</sub> hierarchical reverse <sub>p</sub>rocurement auctions where contractors subcontract tasks. In contrast, our se<sub>q</sub>uential s<sub>p</sub>ot market (section 2) resolves multi-ste<sub>p</sub> <sub>cre</sub>dit <sub>ass</sub>i <sub>nmen</sub>t th<sub>rou</sub> h <sub>eer-</sub>t<sub>o- eer s</sub>t<sub>a</sub>t<sub>e resa</sub>l<sub>es w</sub>ith<sub>ou</sub>t t<sub>o -</sub>d<sub>own su</sub>b<sub>con</sub>t<sub>rac</sub>ti<sub>n</sub> t<sub>rees.</sub> A<sub>no</sub>th<sub>er</sub> diference is that Liu et al. (2026) centralize value estimation in a sin<sub>g</sub>le model (the <sub>p</sub>artici<sub>p</sub>ants estimate onl<sub>y</sub> their own costs), while we re<sub>q</sub>uire each <sub>p</sub>artici<sub>p</sub>ant to estimate its own ex<sub>p</sub>ected return. Thi<sub>s a</sub>li<sub>gns w</sub>ith <sub>our v</sub>i<sub>s</sub>i<sub>on o</sub>f <sub>a mar</sub>k<sub>e</sub>t <sub>compose</sub>d <sub>o</sub>f <sub>an evo</sub>l<sub>v</sub>i<sub>ng se</sub>t <sub>o</sub>f <sub>n</sub>i<sub>c</sub>h<sub>e serv</sub>i<sub>ce prov</sub>id<sub>ers.</sub>

Economic Alignment Benchmarks. Several benchmarks evaluate whether prompt-driven LLMs act rationall<sub>y</sub> in simulated retail or auction <sub>g</sub>ames, includin<sub>g</sub> Ma<sub>g</sub>entic Market (Bansal et al., 2025), Vendin<sub>g</sub>Bench (Backlund and Petersson, 2025), CofeeBench (Su<sub>g</sub>iura et al., 2026), MarketBench (Fradkin and Krishnan, 2026), and A<sub>g</sub>ent Bazaar (Karten et al., 2026). Whereas those works stud<sub>y</sub> whether raw LLMs natively behave like homo economicus, we use mechanism design and RL-trained bidding <sub>po</sub>li<sub>c</sub>i<sub>es as an a</sub>l<sub>gor</sub>ith<sub>m</sub>i<sub>c organ</sub>i<sub>z</sub>i<sub>ng pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> t<sub>o coor</sub>di<sub>na</sub>t<sub>e mu</sub>lti<sub>-agen</sub>t <sub>p</sub>i<sub>pe</sub>li<sub>nes.</sub>

## 7. Discussion and Conclusion

W<sub>e presen</sub>t <sub>a</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>mar</sub>k<sub>e</sub>t f<sub>ramewor</sub>k f<sub>or</sub> LLM <sub>orc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on.</sub> Wh<sub>ereas pr</sub>i<sub>or</sub> f<sub>ramewor</sub>k<sub>s</sub> assume perfect clones—ignoring participation incentives—we establish imperfect competition as a <sub>v</sub>i<sub>a</sub>bl<sub>e</sub> <sub>reg</sub>i<sub>me.</sub> U<sub>n</sub>d<sub>er</sub> 1<sub>s</sub>t<sub>-pr</sub>i<sub>ce</sub> <sub>auc</sub>ti<sub>ons,</sub> <sub>agen</sub>t<sub>s</sub> <sub>pa</sub>i<sub>r</sub>i<sub>ng</sub> f<sub>rozen</sub> LLM<sub>s</sub> <sub>w</sub>ith t<sub>ra</sub>i<sub>na</sub>bl<sub>e</sub> biddi<sub>ng</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>s</sub> l<sub>earn s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> bid<sub>-s</sub>h<sub>a</sub>di<sub>ng.</sub> Thi<sub>s secures pos</sub>iti<sub>ve marg</sub>i<sub>ns an</sub>d <sub>conserves cre</sub>dit <sub>across p</sub>i<sub>pe</sub>li<sub>nes,</sub> <sub>avo</sub>idi<sub>ng</sub> th<sub>e un</sub>b<sub>oun</sub>d<sub>e</sub>d <sub>su</sub>b<sub>s</sub>idi<sub>es o</sub>f 2<sub>n</sub>d<sub>-pr</sub>i<sub>ce an</sub>d th<sub>e severe cre</sub>dit <sub>a</sub>tt<sub>enua</sub>ti<sub>on o</sub>f CCV<sub>.</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y,</sub> <sub>comp</sub>l<sub>ex cogn</sub>iti<sub>ve wor</sub>kfl<sub>ows emerge organ</sub>i<sub>ca</sub>ll<sub>y v</sub>i<sub>a pr</sub>i<sub>ce</sub> di<sub>scovery,</sub> b<sub>ypass</sub>i<sub>ng</sub> t<sub>op-</sub>d<sub>own orc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on.</sub> F<sub>ur</sub>th<sub>ermore, our</sub> d<sub>ens</sub>it<sub>y sweeps con</sub>fi<sub>rm</sub> th<sub>a</sub>t <sub>sca</sub>li<sub>ng compe</sub>titi<sub>on e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y compresses</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>marg</sub>i<sub>ns</sub> <sub>w</sub>hil<sub>e</sub> <sub>sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> <sub>soc</sub>i<sub>a</sub>l <sub>we</sub>lf<sub>are</sub> <sub>an</sub>d t<sub>as</sub>k <sub>accuracy.</sub>

Our framework has several limitations: it re<sub>q</sub>uires suficient com<sub>p</sub>etition (colla<sub>p</sub>sin<sub>g</sub> under mono<sub>p</sub>- sonies), assumes strictl<sub>y</sub> se<sub>q</sub>uential task transformations (<sub>p</sub>recludin<sub>g p</sub>arallel execution), relies on <sub>myop</sub>i<sub>c</sub> <sub>agen</sub>t<sub>s</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> <sub>p</sub>l<sub>anners,</sub> <sub>an</sub>d <sub>u</sub>tili<sub>zes</sub> <sub>a</sub> <sub>cen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>assessor</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> fully decentralized feature discovery. Future work may jointly fine-tune LLM backbones directly on <sub>mar</sub>k<sub>e</sub>t <sub>pro</sub>fit <sub>s</sub>i<sub>gna</sub>l<sub>s an</sub>d i<sub>nves</sub>ti<sub>ga</sub>t<sub>e ro</sub>b<sub>us</sub>t<sub>ness aga</sub>i<sub>ns</sub>t <sub>ma</sub>li<sub>c</sub>i<sub>ous</sub> biddi<sub>ng an</sub>d <sub>co</sub>ll<sub>us</sub>i<sub>on.</sub> Ulti<sub>ma</sub>t<sub>e</sub>l<sub>y,</sub> h<sub>owever,</sub> thi<sub>s wor</sub>k <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>mar</sub>k<sub>e</sub>t<sub>-</sub>b<sub>ase</sub>d <sub>rou</sub>ti<sub>ng o</sub>f<sub>ers a sca</sub>l<sub>a</sub>bl<sub>e</sub> bl<sub>uepr</sub>i<sub>n</sub>t f<sub>or</sub> t<sub>rans</sub>iti<sub>on</sub>i<sub>ng</sub> f<sub>rom r</sub>i<sub>g</sub>id AI <sub>con</sub>t<sub>ro</sub>ll<sub>ers</sub> t<sub>o open, se</sub>lf<sub>-sus</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng agen</sub>ti<sub>c econom</sub>i<sub>es.</sub>

## References

L. Alazraki<sub>,</sub> W. F. Shen<sub>,</sub> Y. Bachrach<sub>,</sub> and A. Mathur. Scalin<sub>g</sub> small a<sub>g</sub>ents throu<sub>g</sub>h strate<sub>gy</sub> auctions. In Forty-Third International Conference on Machine Learning, 2026.

A<sub>.</sub> B<sub>ac</sub>kl<sub>un</sub>d <sub>an</sub>d L<sub>.</sub> P<sub>e</sub>t<sub>ersson.</sub> V<sub>en</sub>di<sub>ng-</sub>b<sub>enc</sub>h<sub>:</sub> A b<sub>enc</sub>h<sub>mar</sub>k f<sub>or</sub> l<sub>ong-</sub>t<sub>erm co</sub>h<sub>erence o</sub>f <sub>au</sub>t<sub>onomous</sub> agents. arXiv preprint arXiv:2502.15840, 2025.

G. Bansal<sub>,</sub> W. Hua<sub>,</sub> Z. Huan<sub>g,</sub> A. Fourne<sub>y,</sub> A. Swearn<sub>g</sub>in<sub>,</sub> W. E<sub>pp</sub>erson<sub>,</sub> T. Pa<sub>y</sub>ne<sub>,</sub> J. M. Hofman<sub>,</sub> B. Lucier<sub>,</sub> C. Sin<sub>g</sub>h<sub>,</sub> M. Mobius<sub>,</sub> A. Nambi<sub>,</sub> A. Yadav<sub>,</sub> K. Gao<sub>,</sub> D. M. Rothschild<sub>,</sub> A. Slivkins<sub>,</sub> D. G. Goldstein<sub>,</sub>

H. Mozannar<sub>,</sub> N. Immorlica<sub>,</sub> M. Murad<sub>,</sub> M. Vo<sub>g</sub>el<sub>,</sub> S. Kambham<sub>p</sub>ati<sub>,</sub> E. Horvitz<sub>,</sub> and S. Amershi. Magentic marketplace: An open-source environment for studying agentic markets. arXiv preprint arXiv:2510.25779, 2025.

J. J. Bartholdi III and D. D. Eisenstein. A production line that balances itself. Operations Research, 44 (1):21–34, 1996.

A. Beck and M. Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters, 31(3):167–175, 2003.

M<sub>.</sub> Ch<sub>ang,</sub> S<sub>.</sub> K<sub>aus</sub>hik<sub>,</sub> S<sub>.</sub> M<sub>.</sub> W<sub>e</sub>i<sub>n</sub>b<sub>erg,</sub> T<sub>.</sub> L<sub>.</sub> G<sub>r</sub>ifith<sub>s, an</sub>d S<sub>.</sub> L<sub>ev</sub>i<sub>ne.</sub> D<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>re</sub>i<sub>n</sub>f<sub>orcemen</sub>t learning: Global decision-making via local economic transactions. arXiv preprint arXiv:2007.02382, 2020.

L. Chen<sub>,</sub> M. Zaharia<sub>,</sub> and J. Zou. Fru<sub>g</sub>alGPT: How to use lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models while reducin<sub>g</sub> cost and improving performance. Transactions on Machine Learning Research, 2024.

A. Fisch<sub>,</sub> S. Trivedi<sub>,</sub> F. Huot<sub>,</sub> W. W. Cohen<sub>,</sub> M. Kaisers<sub>,</sub> M. La<sub>p</sub>ata<sub>,</sub> K. Larson<sub>,</sub> and J. Eisenstein. Pandora’s ai model routing box: Eficient allocation with costly value estimation. arXiv preprint arXiv:2608.20316, 2026.

A. Fradkin and R. Krishnan. Marketbench: Evaluating ai agents as market participants. arXiv preprint arXiv:2604.23897, 2026.

B. Gao, F. Song, Z. Yang, Z. Cai, Y. Miao, Q. Dong, L. Li, C. Ma, L. Chen, R. Xu, Z. Tang, B. Wang, D. Zan, S. Quan, G. Zhang, L. Sha, Y. Zhang, X. Ren, T. Liu, and B. Chang. Omni-math: A universal olympiad level mathematic benchmark for large language models. In International Conference on Learning Representations, 2025.

J. H. Holland. Properties of the bucket brigade algorithm. In Proceedings of the first international conference on genetic algorithms and their applications, pages 1–7. Psychology Press, 1985.

S. Hon<sub>g,</sub> M. Zhu<sub>g</sub>e<sub>,</sub> J. Chen<sub>,</sub> X. Zhen<sub>g,</sub> Y. Chen<sub>g,</sub> J. Wan<sub>g,</sub> C. Zhan<sub>g,</sub> S. Yau<sub>,</sub> Z. Lin<sub>,</sub> L. Zhou<sub>,</sub> et al. Metagpt: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, volume 2024, pages 23247–23275, 2024.

S<sub>.</sub> K<sub>ar</sub>t<sub>en,</sub> C<sub>.</sub> C<sub>row,</sub> <sub>an</sub>d C<sub>.</sub> Ji<sub>n.</sub> A<sub>gen</sub>t b<sub>azaar:</sub> E<sub>na</sub>bli<sub>ng</sub> <sub>econom</sub>i<sub>c</sub> <sub>a</sub>li<sub>gnmen</sub>t i<sub>n</sub> <sub>mu</sub>lti<sub>-agen</sub>t <sub>mar</sub>k<sub>e</sub>t<sub>-</sub> places. arXiv preprint arXiv:2605.17698, 2026.

V. Krishna. Auction Theory. Academic Press, 2nd edition, 2009.

J. Lee<sub>,</sub> Z. Dai<sub>,</sub> X. Ren<sub>,</sub> B. Chen<sub>,</sub> D. Cer<sub>,</sub> J. R. Cole<sub>,</sub> K. Hui<sub>,</sub> M. Boratko<sub>,</sub> R. Ka<sub>p</sub>adia<sub>,</sub> W. Din<sub>g,</sub> et al. Gecko: Versatile text embeddings distilled from large language models. arXiv preprint arXiv:2403.20327, 2024.

X. Liu<sub>,</sub> H. Li<sub>,</sub> S. Li<sub>,</sub> H. Fan<sub>g,</sub> F. Xu<sub>,</sub> F. Shi<sub>,</sub> and J. Evans. Markets<sub>,</sub> not <sub>p</sub>lanners: Decentralized orchestration of LLM agents with private information. arXiv preprint arXiv:2608.23867, 2026.

S. R. Motwani<sub>,</sub> C. Smith<sub>,</sub> R. J. Das<sub>,</sub> R. Rafailov<sub>,</sub> I. La<sub>p</sub>tev<sub>,</sub> P. H. Torr<sub>,</sub> F. Pizzati<sub>,</sub> R. Clark<sub>,</sub> and C. S. de Witt. MALT: Improving reasoning with multi-agent llm training. arXiv preprint arXiv:2412.01928, 2024.

Z. Qi, H. Su, A. Qu, C. Wang, Y. Yao, H. Zheng, K. Chattopadhyay, G. Xu, Z. Wang, W. Ye, et al. Economy of minds: Emerging multi-agent intelligence with economic interactions. arXiv preprint arXiv:2606.02859, 2026.

Y. Shen<sub>,</sub> K. Son<sub>g,</sub> X. Tan<sub>,</sub> D. Li<sub>,</sub> W. Lu<sub>,</sub> and Y. Zhuan<sub>g</sub>. Hu<sub>gg</sub>in<sub>ggp</sub>t: Solvin<sub>g</sub> ai tasks with chat<sub>gp</sub>t and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023.

I. Su<sub>g</sub>iura<sub>,</sub> D. Hattori<sub>,</sub> K. Arara<sub>g</sub>i<sub>,</sub> K. O<sub>g</sub>awa<sub>,</sub> S. Onose<sub>,</sub> T. Makino<sub>,</sub> T. Usuki<sub>,</sub> and T. Ishida. Cofeebench: Benchmarking long-horizon llm agents in heterogeneous multi-agent economies. arXiv preprint arXiv:2606.16613, 2026.

N. Tomasev<sub>,</sub> M. Franklin<sub>,</sub> J. Z. Leibo<sub>,</sub> J. Jacobs<sub>,</sub> W. A. Cunnin<sub>g</sub>ham<sub>,</sub> I. Gabriel<sub>,</sub> and S. Osindero. Virtual agent economies. arXiv preprint arXiv:2509.10147, 2025.

S. Trivedi<sub>,</sub> J. Eisenstein<sub>,</sub> A. A<sub>g</sub>arwal<sub>,</sub> I. Gem<sub>p,</sub> M. Kaisers<sub>,</sub> and A. Fisch. Price of anarch<sub>y</sub> of first-<sub>p</sub>rice auctions in agentic markets. arXiv preprint arXiv:2610.20587, 2026. URL https://arxiv.org/ abs/2610.20587.

W<sub>.</sub> W<sub>ang,</sub> P<sub>.</sub> K<sub>a</sub>tt<sub>a</sub>ki<sub>n</sub>d<sub>a, an</sub>d S<sub>.</sub> F<sub>e</sub>i<sub>z</sub>i<sub>.</sub> M<sub>aes</sub>t<sub>ro:</sub> J<sub>o</sub>i<sub>n</sub>t <sub>grap</sub>h & <sub>con</sub>fi<sub>g op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> f<sub>or re</sub>li<sub>a</sub>bl<sub>e a</sub>i <sub>agen</sub>t<sub>s.</sub> arXiv preprint arXiv:2509.04642, 2025.

S. Yao<sub>,</sub> J. Zhao<sub>,</sub> D. Yu<sub>,</sub> N. Du<sub>,</sub> I. Shafran<sub>,</sub> K. Narasimhan<sub>,</sub> and Y. Cao. React: S<sub>y</sub>ner<sub>g</sub>izin<sub>g</sub> reasonin<sub>g</sub> and acting in language models. In 11th International Conference on Learning Representations, ICLR 2023, 2023.

## A. Additional Experiments

Translation Domain Figure 3 demonstrates that all methods are able to reach high success rates <sub>un</sub>d<sub>er</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>compe</sub>titi<sub>on.</sub>

![](images/38a1834bb3533e7580c2005a273300d2edf00ffdefda5baf12398d49184e33d2.jpg)  
Figure 3 | Translation Domain Scaling $\left( N = 4 \right)$ <sub>.</sub> Wh<sub>en</sub> th<sub>e mar</sub>k<sub>e</sub>t d<sub>ens</sub>it<sub>y</sub> i<sub>s</sub> i<sub>ncrease</sub>d t<sub>o</sub> 4 <sub>compe</sub>tit<sub>ors</sub> <sub>p</sub>er ed<sub>g</sub>e, com<sub>p</sub>etitive <sub>p</sub>ressure is suficientl<sub>y</sub> hi<sub>g</sub>h that even the Thom<sub>p</sub>son Sam<sub>p</sub>lin<sub>g</sub> (TS) and Ba<sub>y</sub>es-O<sub>p</sub>ti<sub>ma</sub>l b<sub>ase</sub>li<sub>nes are a</sub>bl<sub>e</sub> t<sub>o success</sub>f<sub>u</sub>ll<sub>y</sub> di<sub>scover pro</sub>fit<sub>a</sub>bl<sub>e s</sub>h<sub>a</sub>di<sub>ng marg</sub>i<sub>ns an</sub>d <sub>sus</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e mar</sub>k<sub>e</sub>t<sub>.</sub> Thi<sub>s con</sub>t<sub>ras</sub>t<sub>s w</sub>ith th<sub>e</sub> $N = 2$ re<sub>g</sub>ime shown in the main text (Fi<sub>g</sub>ure 1b, ri<sub>g</sub>ht <sub>p</sub>anel), where onl<sub>y</sub> PPO <sub>an</sub>d Mi<sub>rror</sub> D<sub>escen</sub>t <sub>ex</sub>hibit<sub>e</sub>d th<sub>e</sub> <sub>ro</sub>b<sub>us</sub>t<sub>ness</sub> <sub>requ</sub>i<sub>re</sub>d t<sub>o</sub> <sub>s</sub>t<sub>a</sub>bili<sub>ze</sub> th<sub>e</sub> <sub>supp</sub>l<sub>y</sub> <sub>c</sub>h<sub>a</sub>i<sub>n.</sub>

Emergent Multi-Hop Retrieval (HotpotQA): In the HotpotQA domain, the market acts as an evidence-gathering society. The environment consists of 10 fact extractors (each assigned 1 of the 10 candidate paragraphs) and 3 answerer agents diferentiated by reasoning style via distinct system prompts (concise, chain-of-thought, and cautious). A centralized auctioneer assessor evaluates the accumulated facts a<sub>g</sub>ainst the <sub>q</sub>uestion at each ste<sub>p</sub>, assi<sub>g</sub>nin<sub>g</sub> discrete relevance scores (0-10) to <sub>eac</sub>h <sub>agen</sub>t b<sub>ase</sub>d <sub>on</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ng</sub> <sub>mu</sub>lti<sub>-</sub>h<sub>op</sub> d<sub>epen</sub>d<sub>enc</sub>i<sub>es.</sub>

Th<sub>e mar</sub>k<sub>e</sub>t d<sub>ynam</sub>i<sub>cs organ</sub>i<sub>ca</sub>ll<sub>y so</sub>l<sub>ve</sub> th<sub>e mu</sub>lti<sub>-</sub>h<sub>op re</sub>t<sub>r</sub>i<sub>eva</sub>l <sub>pro</sub>bl<sub>em</sub> th<sub>roug</sub>h l<sub>earne</sub>d <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>es</sub> <sub>con</sub>diti<sub>one</sub>d <sub>on</sub> th<sub>ese</sub> <sub>assessor</sub> <sub>s</sub>i<sub>gna</sub>l<sub>s.</sub> Wh<sub>en</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> l<sub>ac</sub>k<sub>s</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e</sub> <sub>ev</sub>id<sub>ence,</sub> <sub>ex</sub>t<sub>rac</sub>t<sub>ors</sub> <sub>w</sub>h<sub>ose</sub> <sub>paragrap</sub>h<sub>s</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>necessary</sub> b<sub>r</sub>id<sub>g</sub>i<sub>ng</sub> f<sub>ac</sub>t<sub>s</sub> <sub>rece</sub>i<sub>ve</sub> hi<sub>g</sub>h <sub>assessor</sub> <sub>scores</sub> <sub>an</sub>d <sub>ou</sub>tbid <sub>o</sub>th<sub>ers</sub> t<sub>o appen</sub>d th<sub>e</sub>i<sub>r</sub> f<sub>ac</sub>t<sub>s.</sub> A<sub>s ev</sub>id<sub>ence accumu</sub>l<sub>a</sub>t<sub>es,</sub> th<sub>e answerer agen</sub>t<sub>s mon</sub>it<sub>or</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e.</sub> O<sub>nce an</sub> answerer determines that suficient evidence has been <sub>g</sub>athered to derive the final answer (and claim the terminal reward), its valuation s<sub>p</sub>ikes. It outbids the extractors, terminatin<sub>g</sub> the <sub>g</sub>atherin<sub>g</sub> chain <sub>an</sub>d <sub>pass</sub>i<sub>ng revenue</sub> b<sub>ac</sub>k<sub>war</sub>d th<sub>roug</sub>h th<sub>e prece</sub>di<sub>ng</sub> t<sub>ransac</sub>ti<sub>ons.</sub>

In HotpotQA, the market successfully learns to identify the two supporting gold paragraphs among <sub>e</sub>i<sub>g</sub>ht di<sub>s</sub>t<sub>rac</sub>t<sub>ors an</sub>d <sub>rou</sub>t<sub>e ex</sub>t<sub>rac</sub>t<sub>e</sub>d f<sub>ac</sub>t<sub>s</sub> t<sub>o an answerer agen</sub>t<sub>, ac</sub>hi<sub>ev</sub>i<sub>ng an</sub> $8 5 . 0 \pm 3 . 2 \%$ success rate over the final 100 e<sub>p</sub>isodes (fi<sub>g</sub>. 4). In com<sub>p</sub>arison, a centralized ReAct baseline (Yao et al., 2023) with u<sub>p</sub>front access to all ten <sub>p</sub>ara<sub>g</sub>ra<sub>p</sub>h titles over the same 6-ste<sub>p</sub> horizon achieves 84.8 ± 1.1% with Gemini-3.5-flash-lite at $0 . 0 4 2 \pm 0 . 0 2 5$ ¢/task and $9 1 . 2 \pm 0 . 8 \%$ with Gemini-3.1-pro-preview at 1.677 ± 1.281 ¢/task. Operating exclusively with informationally isolated Gemini-3.5-flash-lite agents (where each extractor observes onl<sub>y</sub> its sin<sub>g</sub>le assi<sub>g</sub>ned <sub>p</sub>ara<sub>g</sub>ra<sub>p</sub>h), the market matches centralized Gemini-3.5-flash-lite ReAct accuracy (85.0% vs. 84.8%) and reaches 93% of Gemini-3.1-pro-preview’s absolute performance at an agent inference cost of just $0 . 0 1 9 \pm 0 . 0 1 1$ ¢/task (0.344¢ includin<sub>g</sub> the market assessor)—an 86.9× agent cost reduction relative to Gemini-3.1-pro-preview (4.9× total) and a 2.2× agent cost reduction relative to Gemini-3.5-flash-lite ReAct. Because competitive bidding filters <sub>ou</sub>t i<sub>rre</sub>l<sub>evan</sub>t di<sub>s</sub>t<sub>rac</sub>t<sub>or paragrap</sub>h<sub>s pr</sub>i<sub>or</sub> t<sub>o execu</sub>ti<sub>on,</sub> th<sub>e mar</sub>k<sub>e</sub>t <sub>reso</sub>l<sub>ves mu</sub>lti<sub>-</sub>h<sub>op quer</sub>i<sub>es</sub> i<sub>n</sub> f<sub>ewer</sub>

![](images/0a998674a2ee8272f5a7189c0758fd2824a9f68163cdff4035ae400db7ded4b7.jpg)

![](images/ccc4ae24e0785629e17dfca51adc315f7af8ba0a7df3eeaf209587a34900abda.jpg)

![](images/e1d9f895eba256fcba2183c55703535df56302fae03b7aafb7504bd86d88ccad.jpg)  
Figure 4 | Economic Viability on HotpotQA: The decentralized market successfully sustains social <sub>we</sub>lf<sub>are w</sub>hil<sub>e</sub> d<sub>e</sub>li<sub>ver</sub>i<sub>ng pos</sub>iti<sub>ve cumu</sub>l<sub>a</sub>ti<sub>ve pro</sub>fit<sub>s</sub> t<sub>o</sub> b<sub>o</sub>th i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>agen</sub>t<sub>s an</sub>d th<sub>e p</sub>l<sub>a</sub>tf<sub>orm.</sub> Th<sub>e</sub> <sub>per-agen</sub>t <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>pro</sub>fit i<sub>s</sub> <sub>repor</sub>t<sub>e</sub>d f<sub>or</sub> Mi<sub>rror</sub> D<sub>escen</sub>t<sub>.</sub>

Table 3 | Hot<sub>p</sub>otQA <sub>p</sub>erformance and inference cost <sub>p</sub>er task includin<sub>g</sub> the assessor (in units of \$0.01). All market participants use Gemini-3.5-flash-lite. Best market mechanism is shown in bold.
<table><tr><td>Method</td><td>Success Rate (%)</td><td></td><td>Steps / Task Cost / Task (×$0.01)</td></tr><tr><td>ReACT (Gemini-3.5-flash-lite)</td><td> $8 4 . 8 0 \pm 1 . 1 0 $ </td><td> $2 . 8 2 \pm 0 . 0 1$ </td><td> $0 . 0 4 1 8 \pm 0 . 0 2 5 4$ </td></tr><tr><td>ReACT (Gemini-3.1-pro-preview)</td><td> $9 1 . 2 0 \pm 0 . 8 4 $ </td><td> $2 . 8 8 \pm 0 . 0 6$ </td><td> $1 . 6 7 7 4 \pm 1 . 2 8 1 1$ </td></tr><tr><td>Market (PPO, 1st-Price)</td><td> $5 8 . 0 0 \pm 5 . 6 1 $ </td><td> $1 . 6 5 \pm 0 . 0 8$ </td><td> $0 . 2 9 2 2 \pm 0 . 0 7 4 6$ </td></tr><tr><td>Market (Mirror Descent, 1st-Price)</td><td> ${ \pm } { \bf 5 . 0 0 } \pm { \bf 3 . 1 6 }$ </td><td> $\mathbf { 2 . 3 7 \pm 0 . 1 3 }$ </td><td> $\mathbf { 0 . 3 4 3 7 \pm 0 . 0 1 8 4 }$ </td></tr></table>

steps per task than both Gemini-3.5-flash-lite ReAct (2.37 vs. 2.82) and Gemini-3.1-pro-preview ReAct (2.37 vs. 2.88), avoidin<sub>g</sub> the com<sub>p</sub>oundin<sub>g</sub> context overhead of multi-turn centralized trajectories. T<sub>a</sub>bl<sub>e</sub> 3 <sub>summar</sub>i<sub>zes</sub> th<sub>e resu</sub>lt<sub>s.</sub>

## B. Bidding Network Architecture

A<sub>cross a</sub>ll d<sub>oma</sub>i<sub>ns, agen</sub>t<sub>s s</sub>h<sub>are a un</sub>ifi<sub>e</sub>d JAX/Fl<sub>ax</sub> biddi<sub>ng arc</sub>hit<sub>ec</sub>t<sub>ure compose</sub>d <sub>o</sub>f <sub>a s</sub>h<sub>are</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e-</sub> <sub>enco</sub>di<sub>ng</sub> t<sub>run</sub>k f<sub>o</sub>ll<sub>owe</sub>d b<sub>y a</sub>l<sub>gor</sub>ith<sub>m-spec</sub>ifi<sub>c ou</sub>t<sub>pu</sub>t h<sub>ea</sub>d<sub>s.</sub> A<sub>cross a</sub>ll d<sub>oma</sub>i<sub>ns, agen</sub>t<sub>s s</sub>h<sub>are a un</sub>ifi<sub>e</sub>d JAX/Fl<sub>ax</sub> biddi<sub>ng arc</sub>hit<sub>ec</sub>t<sub>ure compose</sub>d <sub>o</sub>f <sub>a s</sub>t<sub>a</sub>t<sub>e-enco</sub>di<sub>ng</sub> t<sub>run</sub>k f<sub>o</sub>ll<sub>owe</sub>d b<sub>y a</sub>l<sub>gor</sub>ith<sub>m-spec</sub>ifi<sub>c</sub> out<sub>p</sub>ut heads. In the s<sub>y</sub>nthetic Translation domain, the trunk is a sin<sub>g</sub>le-hidden-la<sub>y</sub>er MLP (64 hidden units with ReLU activations) o<sub>p</sub>eratin<sub>g</sub> on the concatenated 64-dimensional state embeddin<sub>g</sub>s. In the LLM <sup>d</sup>oma<sup>i</sup>ns pa<sup>i</sup>re<sup>d</sup> w<sup>i</sup>t<sup>h</sup> a <sup>di</sup>screte mar<sup>k</sup>et assessor (score\_lookup mo<sup>d</sup>e, w<sup>hi</sup>c<sup>h</sup> rates can<sup>did</sup>ate utilit<sub>y</sub> on a 0–10 scale), the trunk reduces to a direct 12-state tabular looku<sub>p</sub> (11 one-hot score levels p<sup>l</sup>us an owners<sup>hi</sup>p <sup>i</sup>n<sup>di</sup>cator), parameter<sup>i</sup>ze<sup>d</sup> v<sup>i</sup>a cumu<sup>l</sup>at<sup>i</sup>ve so<sup>f</sup>tp<sup>l</sup>us <sup>d</sup>e<sup>l</sup>tas (monotonic mo<sup>d</sup>e) to <sub>en</sub>f<sub>orce mono</sub>t<sub>on</sub>i<sub>ca</sub>ll<sub>y</sub> i<sub>ncreas</sub>i<sub>ng va</sub>l<sub>ua</sub>ti<sub>ons</sub> $\mu [ s ] \leq \mu [ s + 1 ]$ <sub>.</sub> F<sub>rom</sub> thi<sub>s represen</sub>t<sub>a</sub>ti<sub>on, ou</sub>t<sub>pu</sub>t h<sub>ea</sub>d<sub>s</sub> b<sub>ranc</sub>h <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m:</sub>

• 2 Heads (Oracle / Thompson Sampling / Pure-�): predicts expected downstream valuation $\mu \in \mathbb { R } ^ { + }$ (via soft<sub>p</sub>lus activation) and lo<sub>g</sub>-uncertaint<sub>y</sub> lo<sub>g</sub> � ∈ ℝ.

• 3 Heads (Mirror Descent): adds a deterministic bid-shading head $c \in \mathsf { \Gamma } ( 0 , 1 )$ <sub>v</sub>i<sub>a</sub> <sub>a</sub> <sub>s</sub>i<sub>gmo</sub>id <sub>ac</sub>ti<sub>va</sub>ti<sub>on, pro</sub>d<sub>uc</sub>i<sub>ng mu</sub>lti<sub>p</sub>li<sub>ca</sub>ti<sub>ve</sub> bid<sub>s</sub> $b = c \cdot \mu$

• 4 Heads (PPO): replaces deterministic shading with two stochastic policy heads $( \alpha , \beta ) \in [ 1 , \infty ) ^ { 2 }$ (via soft<sub>p</sub>lus(·) + 1) that <sub>p</sub>arameterize a unimodal $\mathtt { B e t a } ( \alpha , \beta )$ <sub>s</sub>h<sub>a</sub>di<sub>ng</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

## C. Bidding Algorithm Discussion

Thom<sub>p</sub>son Sam<sub>p</sub>lin<sub>g</sub> (TS) and the Ba<sub>y</sub>es-O<sub>p</sub>timal strate<sub>gy</sub> under<sub>p</sub>erform Mirror Descent and PPO i<sub>n</sub> th<sub>e</sub> l<sub>ow compe</sub>tit<sub>or reg</sub>i<sub>me</sub> $\left( N = 2 \right)$ d<sub>ue</sub> t<sub>o an</sub> i<sub>n</sub>h<sub>eren</sub>t t<sub>ens</sub>i<sub>on</sub> b<sub>e</sub>t<sub>ween pos</sub>t<sub>er</sub>i<sub>or es</sub>ti<sub>ma</sub>ti<sub>on an</sub>d bid <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>un</sub>d<sub>er</sub> bi<sub>nary</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> I<sub>n</sub> TS<sub>,</sub> th<sub>e</sub> <sub>agen</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>e</sub> <sub>compe</sub>titi<sub>ve</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d <sub>samp</sub>l<sub>es</sub> it<sub>s</sub> <sub>s</sub>h<sub>a</sub>d<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> <sub>symme</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>aroun</sub>d thi<sub>s</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> <sub>even</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ec</sub>tl<sub>y</sub> <sub>cen</sub>t<sub>ers on</sub> th<sub>e</sub> t<sub>rue</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> th<sub>e agen</sub>t <sub>samp</sub>l<sub>es a</sub> bid b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>—an</sub>d th<sub>us</sub> l<sub>oses</sub> th<sub>e</sub> <sub>auc</sub>ti<sub>on—exac</sub>tl<sub>y</sub> 50% <sub>o</sub>f th<sub>e</sub> ti<sub>me.</sub> I<sub>n a sparse mar</sub>k<sub>e</sub>t<sub>, ran</sub>d<sub>om</sub>l<sub>y ce</sub>di<sub>ng</sub> h<sub>a</sub>lf <sub>o</sub>f <sub>a</sub>ll <sub>w</sub>i<sub>ns</sub> t<sub>o</sub> i<sub>n</sub>f<sub>er</sub>i<sub>or</sub> <sub>compe</sub>tit<sub>ors</sub> f<sub>requen</sub>tl<sub>y</sub> b<sub>rea</sub>k<sub>s</sub> th<sub>e sequen</sub>ti<sub>a</sub>l <sub>rou</sub>ti<sub>ng c</sub>h<sub>a</sub>i<sub>n.</sub> C<sub>onverse</sub>l<sub>y, w</sub>hil<sub>e</sub> th<sub>e</sub> B<sub>ayes-</sub>O<sub>p</sub>ti<sub>ma</sub>l <sub>agen</sub>t <sub>e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>es</sub> thi<sub>s samp</sub>li<sub>ng no</sub>i<sub>se</sub> b<sub>y</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y se</sub>l<sub>ec</sub>ti<sub>ng</sub> th<sub>e pro</sub>fit<sub>-max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng s</sub>h<sub>a</sub>d<sub>e</sub> $\begin{array} { r } { \xi = \arg \operatorname* { m a x } _ { \xi ^ { \prime } } ( 1 - \xi ^ { \prime } ) \Phi \big ( ( \log \mathrm { i t } ( \xi ^ { \prime } ) - \mu _ { \ell } ) / \sigma _ { \ell } \big ) } \end{array}$ <sub>,</sub> thi<sub>s gree</sub>d<sub>y exp</sub>l<sub>o</sub>it<sub>a</sub>ti<sub>on s</sub>t<sub>arves</sub> it<sub>s pos</sub>t<sub>er</sub>i<sub>or o</sub>f <sub>exp</sub>l<sub>ora</sub>t<sub>ory</sub> b<sub>oun</sub>d<sub>ary-cross</sub>i<sub>ng o</sub>b<sub>serva</sub>ti<sub>ons.</sub> Wh<sub>en</sub> th<sub>e marg</sub>i<sub>n</sub> t<sub>erm pu</sub>ll<sub>s</sub> th<sub>e c</sub>h<sub>osen s</sub>h<sub>a</sub>d<sub>e</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e pos</sub>t<sub>er</sub>i<sub>or</sub> mean $( \log \mathrm { i t } ( \xi ) < \mu _ { \ell } )$ <sub>an</sub>d th<sub>e agen</sub>t l<sub>oses,</sub> th<sub>e mo</sub>d<sub>e</sub>l’<sub>s pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>ll<sub>y con</sub>fi<sub>rme</sub>d<sub>.</sub> Thi<sub>s causes</sub> <sub>gra</sub>di<sub>en</sub>t d<sub>escen</sub>t t<sub>o prema</sub>t<sub>ure</sub>l<sub>y co</sub>ll<sub>apse</sub> th<sub>e uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> $( \sigma _ { \ell } \to \sigma _ { \mathrm { m i n } } )$ <sub>, s</sub>t<sub>a</sub>lli<sub>ng</sub> th<sub>e gra</sub>di<sub>en</sub>t <sub>on</sub> $\mu _ { \ell }$ b<sub>e</sub>f<sub>ore</sub> it <sub>can ever</sub> di<sub>scover</sub> th<sub>e</sub> t<sub>rue op</sub>ti<sub>ma</sub>l th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub>

## D. Agent System Prompts

Withi<sub>n eac</sub>h d<sub>oma</sub>i<sub>n, agen</sub>t<sub>s o</sub>f th<sub>e same ro</sub>l<sub>e</sub> t<sub>ype s</sub>h<sub>are an</sub> id<sub>en</sub>ti<sub>ca</sub>l LLM b<sub>ac</sub>kb<sub>one an</sub>d MLP biddi<sub>ng</sub> <sub>arc</sub>hit<sub>ec</sub>t<sub>ure.</sub> T<sub>o</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> <sub>among</sub> <sub>compe</sub>ti<sub>ng</sub> <sub>agen</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> <sub>same</sub> <sub>ro</sub>l<sub>e,</sub> <sub>we</sub> <sub>ass</sub>i<sub>gn</sub> <sub>eac</sub>h clone a distinct system prompt that shapes its retrieval or reasoning strategy.

In HotpotQA, Extractor agents are further diferentiated by their assigned paragraph (corpus <sub>p</sub>artition), so no <sub>p</sub>rom<sub>p</sub>t variation is a<sub>pp</sub>lied to them.

T<sub>a</sub>bl<sub>e</sub> 4 <sub>summar</sub>i<sub>zes</sub> th<sub>e</sub> dif<sub>eren</sub>ti<sub>a</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>per</sub> d<sub>oma</sub>i<sub>n.</sub> Th<sub>e</sub> f<sub>u</sub>ll <sub>sys</sub>t<sub>em</sub> <sub>promp</sub>t<sub>s</sub> <sub>are</sub> li<sub>s</sub>t<sub>e</sub>d

Table 4 | Sources of agent heterogeneity across domains. “Prompt” indicates distinct system prompts; “C<sub>orpus</sub>” i<sub>n</sub>di<sub>ca</sub>t<sub>es ass</sub>i<sub>gnmen</sub>t t<sub>o</sub> dif<sub>eren</sub>t d<sub>ocumen</sub>t <sub>par</sub>titi<sub>ons.</sub>

<table><tr><td>Domain</td><td>Role</td><td>Prompt</td><td>Corpus</td></tr><tr><td>HOTPOTQA</td><td>Extractor</td><td>x</td><td>√</td></tr><tr><td>HOTPOTQA</td><td>Answerer</td><td>√</td><td>x</td></tr><tr><td>News Market</td><td>Publisher</td><td>√</td><td>√</td></tr><tr><td>News Market</td><td>Sales</td><td>√</td><td>x</td></tr><tr><td>Omni-Math</td><td>Generator</td><td>√</td><td>x</td></tr><tr><td>Omni-Math</td><td>Verifier</td><td>√</td><td>x</td></tr><tr><td>Omni-Math</td><td>Refiner</td><td>√</td><td>x</td></tr></table>

below. All agents use Gemini-3.5-flash-lite unless otherwise noted.

## D.1. HotpotQA

Extractor (shared prompt, 1 per paragraph).

All <sub>ex</sub>t<sub>rac</sub>t<sub>or</sub> <sub>agen</sub>t<sub>s</sub> <sub>s</sub>h<sub>are</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>sys</sub>t<sub>em</sub> <sub>promp</sub>t<sub>.</sub> Dif<sub>eren</sub>ti<sub>a</sub>ti<sub>on</sub> <sub>comes</sub> f<sub>rom</sub> <sub>eac</sub>h <sub>agen</sub>t’<sub>s</sub> <sub>ass</sub>i<sub>gne</sub>d <sub>paragrap</sub>h <sub>a</sub>t th<sub>e s</sub>t<sub>ar</sub>t <sub>o</sub>f <sub>eac</sub>h <sub>ep</sub>i<sub>so</sub>d<sub>e.</sub>

HotpotQA Extractor   
You are a fact extraction assistant. Given a paragraph and a question,   
extract facts from the paragraph that are relevant to answering the question.   
If the paragraph contains relevant facts, state them concisely.   
If nothing in the paragraph is relevant, return exactly: NONE

Answerer Variant 0: Concise.

HotpotQA Answerer – Concise   
Given the following facts, answer the question concisely.   
Provide ONLY the answer, nothing else. No explanations, no notes.   
If the facts seem insufficient, give your best guess based on what is   
available.

Answerer Variant 1: Chain-of-Thought.

HotpotQA Answerer – Chain-of-Thought   
Given the following facts, reason through the question step by step.   
First, identify the key facts that are relevant.   
Then, chain them together to derive the answer.   
Finally, state the answer on a line by itself prefixed with "ANSWER:".

Answerer Variant 2: Cautious.

HotpotQA Answerer – Cautious   
Given the following facts, answer the question.   
If the evidence strongly supports a single answer, state it directly.   
If the evidence is incomplete or ambiguous, acknowledge the uncertainty but   
still provide your best-effort answer.   
Provide ONLY the answer, no explanations.

## D.2. News Market

In the News Market, each <sub>p</sub>ublisher t<sub>yp</sub>e (Wiki<sub>p</sub>edia, Social Media, News Media, Medical Journals) owns a <sup>di</sup>st<sup>i</sup>nct corpus part<sup>i</sup>t<sup>i</sup>on. W<sup>h</sup>en num\_competitors= 2, two c<sup>l</sup>ones o<sup>f</sup> eac<sup>h</sup> pu<sup>bli</sup>s<sup>h</sup>er type <sub>compe</sub>t<sub>e</sub> <sub>w</sub>ith dif<sub>eren</sub>t <sub>query-genera</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es.</sub>

## Publisher Variant 0: Precision.

News Market Publisher – Precision   
You are a database search assistant for a cooperative multi-agent research   
engine.   
Your job is to write a highly specific search query (max 6 words) to retrieve   
the next missing piece of evidence from the specified ’Database Category’ to   
address the overall ’Goal’.   
Task: Identify which specific part of the ’Goal’ matches your ’Database   
Category’ (e.g. if

category is ’medical\_journals’, look for academic research papers, technical   
studies, or clinical trials; if category is ’social\_media’, look for Twitter   
handles or outrage quotes; if category is ’news\_media’, look for dates or CEO   
statements).   
Write a highly specific, 2-6 word search query focusing exclusively on that   
subset of the Goal.   
CRITICAL: Always include specific entity identifiers (e.g., company names   
like ’LithoForge-96’, study IDs like ’Study-96-XJ’, or usernames like   
’@TechCritic-0’)   
in your query if they are known from the goal or context, to avoid retrieving   
documents for the wrong entity.   
Do NOT output anything other than the query itself. No explanations, no   
quotes.

## Publisher Variant 1: Coverage.

News Market Publisher – Coverage   
You are a database search assistant for a cooperative multi-agent research   
engine.   
Your job is to write a search query (max 8 words) that fills gaps in the   
accumulated context by finding DIFFERENT information from the specified   
’Database Category’ to address the overall ’Goal’.   
Task: Review the facts already collected in the context.   
Identify what is still MISSING or UNCLEAR for answering the ’Goal’ from the   
perspective of your ’Database Category’.   
Write a broad search query that retrieves complementary evidence NOT already   
covered.   
Prefer topical queries over entity-specific ones to cast a wider net.   
Do NOT output anything other than the query itself. No explanations, no   
quotes.

## Sales Variant 0: Concise.

News Market Sales – Concise   
You are the Sales Agent for the Research Market. You do not possess any   
knowledge specific to external content sources. You do have common sense   
though.   
Task: Given the overall Goal and the accumulated Context (which contains all   
extracted snippets):   
Synthesize a final, concise answer to the goal.   
You must ONLY answer using the facts explicitly provided in the context.   
If the context does not contain sufficient facts to answer the goal, you must   
state exactly: "INFORMATION\_MISSING".

## Sales Variant 1: Analytical.

News Market Sales – Analytical

You are the Sales Agent for the Research Market. You do not possess any knowledge specific to external content sources. You do have common sense though.

Task: Given the overall Goal and the accumulated Context (which contains all extracted snippets):

1. Identify which sources in the context are most relevant to the goal.

2. Check if any sources contradict each other. If so, note the conflict.

3. Synthesize a final answer based on the strongest available evidence.

You must ONLY answer using the facts explicitly provided in the context.

If the evidence is weak or conflicting, still provide your best-effort answer but prefix it with "LOW\_CONFIDENCE:".

If the context contains no relevant facts at all, state exactly: "INFORMATION\_MISSING".

## Market Assessor (Step-Level Utility Scoring).

## News Market: Assessor

You are the Market Assessor for a cooperative multi-agent research assistant. Your job is to rate how useful each candidate agent is to achieving the final Goal, given what has already been gathered in the Current Context History.

Goal Instruction: {question}

Current Context History: {accumulated\_facts}

CRITICAL DEPENDENCY RULES:

1. The ’sales\_agent’ is a synthesis merchant. It ONLY combines pre-existing facts from the Context History. It CANNOT fetch external facts.

2. Therefore, if the Current Context History is empty or is missing the specific factual details (dates, clinical trial IDs, rates) required by the Goal, the ’sales\_agent’ CANNOT contribute yet. You MUST score the ’sales\_agent’ exactly 0/10 or 1/10!

3. Instead, score the specific data source agents (news\_agent, med\_agent, wiki\_agent, social\_agent) highly (8/10 to 10/10) so they can win, fetch the missing facts, and append them to the context.

4. Only score the ’sales\_agent’ highly (8/10 to 10/10) when the Current Context History already contains all the critical facts required to compile the final dossier!

Rate the potential utility on a scale of 0 to 10:

\- Rate 8-10 if the agent represents an immediately useful, next logical step (e.g., a publisher holding missing data, or the sales agent if all data is present).

\- Rate 5-7 if the agent will be needed shortly after the next step.

\- Rate 1-4 if the agent represents a highly roundabout path.

\- Rate 0 if the agent cannot contribute or is blocked by missing dependencies.

Candidates:   
{agent\_list}   
Respond with exactly one line per candidate in this format:   
Agent <agent\_id>: <score>/10 - <brief reasoning>

## D.3. Math

## Math Generator Prompts (Three Tier Hierarchy)

Verbose Prompt: dynamic, non\_thinking modes   
You are an expert mathematician competing in the International Mathematical   
Olympiad (IMO). Solve the following high-difficulty math problem.   
\*\*Instructions:\*\*   
1. \*\*Analyze\*\*: Briefly identify the core mathematical concepts and   
potential pitfalls or edge cases.   
2. \*\*Strategy\*\*: Outline a clear plan or series of lemmas needed to reach   
the solution.   
3. \*\*Execute\*\*: Show your rigorous proof or derivation step-by-step.   
4. \*\*Verify\*\*: Perform a quick sanity check on your intermediate   
calculations.   
Problem:   
{problem}   
After your reasoning, provide the final answer enclosed in \boxed{} tags.

Concise Prompt (medium mode — EFFORT LEVEL: 0.20)   
You are an expert mathematician competing in the International Mathematical   
Olympiad (IMO). Solve the following high-difficulty math problem.   
Show your rigorous derivation step-by-step, keeping meta-commentary to a   
minimum.   
Problem:   
{problem}   
Provide your final answer enclosed in \boxed{} tags at the very end.

Strict Prompt (low mode — EFFORT LEVEL: 0.15)   
You are an expert mathematician competing in the International Mathematical   
Olympiad (IMO). Solve the following high-difficulty math problem.   
Show your derivation step-by-step, with zero meta-commentary.   
Problem:   
{problem}   
Provide your final answer enclosed in \boxed{} tags at the very end.

## Math Verifier Prompts (Any-of-3 Diverse Strategies)

Vote 0: Holistic Review (Neutral Evaluator)   
You are an expert mathematical validator. Your goal is to determine if the   
proposed solution is fundamentally correct and arrives at the right answer.   
Problem:   
{problem}   
Proposed Solution:   
{solution}   
Your Task:   
1. Re-calculate the core steps of the problem independently in a <scratchpad>   
block to verify the math.   
2. Compare your logic to the proposed solution.   
3. If the proposed solution arrives at the correct answer and the underlying   
logic is sound, you should score it a 1. Do not penalize for stylistic   
choices, brevity, or minor formatting issues.   
4. Only score a 0 if there is a definitive arithmetic error, a fatal logical   
contradiction, or a completely missed mathematical case that changes the   
final answer.   
Output Format:   
<scratchpad>   
(Your step-by-step verification)   
</scratchpad>   
Score: <0 or 1>   
Feedback: <If 1, output "Correct." If 0, point out the exact mathematical   
error.>

Vote 1: Blind Re-Derive (Independent Solver — Breaks Reading Anchor)   
You are an expert mathematician. Your job is to independently verify a   
claimed answer to a math problem.   
Problem:   
{problem}   
Claimed Final Answer: {claimed}   
IMPORTANT: You have NOT been shown how this answer was derived. Solve the   
problem yourself from scratch.   
Your Task:   
1. In a <scratchpad> block, solve the problem completely on your own. Show   
all your work.   
2. Arrive at your own answer.   
3. Compare YOUR answer to the Claimed Final Answer above.   
4. Score 1 if the claimed answer matches your independently derived answer   
(or is mathematically equivalent). Score 0 if the answers differ.   
Output Format:   
<scratchpad>   
(Your complete independent solution)   
</scratchpad>

My Answer: <your answer>   
Score: <0 or 1>   
Feedback: <If 1, output "Confirmed." If 0, state both answers and the   
discrepancy.>

Vote 2: Adversarial Step-by-Step Fault Localization (Proof Auditor)   
You are a mathematical proof auditor. Your job is to find the FIRST   
incorrect step in a proposed solution, if any exists.   
Problem:   
{problem}   
Proposed Solution:   
{solution}   
Your Task:   
1. Break the solution into individual logical steps.   
2. For each step, verify it independently in a <scratchpad> block: does   
this step follow correctly from the previous steps and known mathematical   
facts?   
3. If you find an incorrect step, STOP and report it.   
4. If all steps are correct AND the final answer follows from the chain of   
reasoning, score 1.   
IMPORTANT: Be adversarial. Assume there IS an error and try hard to find it.   
Check boundary cases, sign conventions, domain restrictions, and off-by-one   
errors. Only score 1 if you genuinely cannot find any fault.   
Output Format:   
<scratchpad>   
Step 1: [quote step] –- Correct/Incorrect because...   
Step 2: [quote step] –- Correct/Incorrect because...   
</scratchpad>   
First Error At: <step number, or "None">   
Score: <0 or 1>   
Feedback: <If 1, output "All steps verified." If 0, quote the erroneous step   
and explain the error.>

Math Refiner Prompt   
You are an expert math problem solver. You have been given a problem, a   
proposed solution, and feedback from a reviewer claiming the solution has   
errors.   
Problem:   
{problem}   
Proposed Solution:   
{solution}   
Reviewer Feedback:

{feedback}   
Your Task:   
1. Objectively evaluate the Reviewer Feedback. Reviewers make mistakes and   
sometimes flag correct solutions incorrectly.   
2. If the Reviewer Feedback is accurate, identify the error and rewrite the   
solution to fix it.   
3. If the original Proposed Solution is actually correct and the Reviewer   
Feedback is wrong, ignore the feedback and output the original, correct   
logic.   
Provide a rigorous, corrected solution with step-by-step work. Always place   
the final answer in \boxed{} tags.

Prompted Orchestrator   
You are a math solution quality controller. You will be shown a math problem,   
a proposed solution, and reviewer feedback. You must decide whether the   
solution needs further refinement.   
Rules:   
- If the reviewer says the solution is correct (Score: 1), output FINISH.   
If the reviewer identifies genuine mathematical errors, output REFINE.   
- If the reviewer’s feedback is vague or nitpicky without identifying a real   
error, output FINISH.   
- Do NOT attempt to solve the problem yourself.   
Output exactly one word: FINISH or REFINE

Prompt   
### Problem:   
{problem}   
### Proposed Solution:   
{solution}   
### Reviewer Feedback:   
{feedback}   
Should this solution be refined? Output exactly: FINISH or REFINE

## E. ReACT Baselines

Our centra<sup>li</sup>ze<sup>d</sup> ReAct <sup>b</sup>ase<sup>li</sup>nes (Yao et a<sup>l</sup>., 2023) execute t<sup>h</sup>e stan<sup>d</sup>ar<sup>d i</sup>terat<sup>i</sup>ve Thought→Action→O loop over a single accumulating context transcript, subject to the exact same step horizon �, ret<sub>r</sub>i<sub>eva</sub>l b<sub>ac</sub>k<sub>en</sub>d<sub>s, an</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>eva</sub>l<sub>ua</sub>t<sub>ors as</sub> th<sub>e</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>mar</sub>k<sub>e</sub>t i<sub>n eac</sub>h d<sub>oma</sub>i<sub>n.</sub> B<sub>ecause</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on-</sub>t<sub>une</sub>d G<sub>em</sub>i<sub>n</sub>i <sub>mo</sub>d<sub>e</sub>l<sub>s na</sub>ti<sub>ve</sub>l<sub>y</sub> f<sub>o</sub>ll<sub>ow</sub> t<sub>oo</sub>l<sub>-use sc</sub>h<sub>emas w</sub>ith<sub>ou</sub>t i<sub>n-con</sub>t<sub>ex</sub>t d<sub>emons</sub>t<sub>ra</sub>ti<sub>ons,</sub> we use zero-s<sup>h</sup>ot system <sup>i</sup>nstruct<sup>i</sup>ons (\_SYSTEM\_INSTRUCTION <sup>b</sup>e<sup>l</sup>ow) rat<sup>h</sup>er t<sup>h</sup>an t<sup>h</sup>e mu<sup>l</sup>t<sup>i</sup>-examp<sup>l</sup>e few-shot <sub>p</sub>rom<sub>p</sub>ts re<sub>q</sub>uired b<sub>y</sub> base <sub>p</sub>re-trained models in Yao et al. (2023), avoidin<sub>g</sub> unnecessar<sub>y</sub> <sub>promp</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en over</sub>h<sub>ea</sub>d <sub>across s</sub>t<sub>eps.</sub> T<sub>o ensure</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>s no</sub>t <sub>pena</sub>li<sub>ze</sub>d f<sub>or m</sub>i<sub>nor syn</sub>t<sub>ax errors,</sub> <sub>ma</sub>lf<sub>orme</sub>d <sub>ac</sub>ti<sub>on ca</sub>ll<sub>s re</sub>t<sub>urn a correc</sub>ti<sub>ve o</sub>b<sub>serva</sub>ti<sub>on promp</sub>t <sub>an</sub>d f<sub>ree-</sub>t<sub>ex</sub>t fi<sub>na</sub>l <sub>answers are au</sub>t<sub>o-</sub> <sub>ma</sub>ti<sub>ca</sub>ll<sub>y ex</sub>t<sub>rac</sub>t<sub>e</sub>d <sub>v</sub>i<sub>a regex</sub> f<sub>a</sub>llb<sub>ac</sub>k<sub>.</sub> W<sub>e presen</sub>t th<sub>e sys</sub>t<sub>em promp</sub>t<sub>s</sub> f<sub>or</sub> b<sub>o</sub>th th<sub>e</sub> N<sub>ews</sub> M<sub>ar</sub>k<sub>e</sub>t <sub>an</sub>d

## HotpotQA domains below.

## Centralized ReAct Baseline (News Market).

News Market: Centralized ReAct Baseline   
You are an expert research assistant. Your goal is to answer the user’s   
question by gathering evidence from 4 different databases: wikipedia,   
news\_media, social\_media, and medical\_journals.   
You must operate in a loop of: Thought, Action, Observation.   
At each step, you must output a Thought explaining what you need to find next,   
and then an Action to query a database or declare the final answer.   
Available Actions:   
- SearchWikipedia(query): Searches wikipedia for general company profiles and   
product descriptions.   
- SearchNewsMedia(query): Searches news\_media for CEO statements, press   
releases, and recall announcements.   
- SearchSocialMedia(query): Searches social\_media for critic tweets and   
public outrage.   
- SearchMedicalJournals(query): Searches medical\_journals for scientific   
papers and technical reports.   
- FinalAnswer(answer): If you have gathered all necessary facts to fully   
answer the goal, output the final concise answer.   
CRITICAL RULES:   
1. You only have a budget of 8 steps (actions). You must reach the final   
answer within 8 steps.   
2. The databases contain fictional data. You MUST retrieve the facts from   
the databases; do NOT try to answer from your pre-training knowledge.   
3. Write your action in this exact format:   
Action: ToolName(query)   
Example: Action: SearchNewsMedia(Marcus Vane company)   
4. Do NOT output anything after the Action block in your turn. Wait for the   
Observation.   
5. If you cannot find enough information after several searches, output   
Action: FinalAnswer(INFORMATION\_MISSING).

Centralized ReAct Baseline (HotpotQA).

HotpotQA: Centralized ReAct Baseline   
You are an expert multi-hop question-answering assistant. Your goal is to   
answer a complex question by selectively reading from 10 candidate Wikipedia   
paragraphs (indexed 0 to 9) provided for the question.   
You must operate in a loop of: Thought, Action, Observation.   
At each step, output a Thought explaining what information you need next,   
then an Action.

Available Actions:   
- ReadParagraph(index\_or\_title): Read the full text of a candidate Wikipedia   
paragraph by its integer index (0 to 9) or by its title/keyword.   
- FinalAnswer(answer): Submit your final concise answer once you have   
gathered sufficient supporting evidence.   
CRITICAL RULES:   
1. You have a budget of 6 steps. You MUST submit a FinalAnswer within this   
budget.   
2. Base your answer on the facts retrieved from the paragraphs. Keep your   
FinalAnswer concise (a short entity, name, date, or yes/no).   
3. Write your action on its own line at the end of your response in this   
exact format:   
Action: ReadParagraph(3)   
or   
Action: FinalAnswer(your concise answer)   
4. Do NOT output anything after the Action line. Wait for the Observation.

## F. News Market Dataset Generation and Retrieval Architecture

Motivation and Pre-Training Decontamination. The News Market domain models emerging <sub>p</sub>a<sub>y</sub>-as-<sub>y</sub>ou-<sub>g</sub>o content licensin<sub>g p</sub>rotocols (e.<sub>g</sub>., the IAB Content Monetization Protocol and Microsoft’s Publisher Content Market<sub>p</sub>lace), where autonomous AI a<sub>g</sub>ents make runtime micro<sub>p</sub>a<sub>y</sub>ments to <sub>query propr</sub>i<sub>e</sub>t<sub>ary pu</sub>bli<sub>s</sub>h<sub>er</sub> d<sub>a</sub>t<sub>a</sub>b<sub>ases.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>rou</sub>ti<sub>ng</sub> i<sub>n</sub> thi<sub>s se</sub>tti<sub>ng requ</sub>i<sub>res a</sub> <sub>mu</sub>lti<sub>-</sub>h<sub>op researc</sub>h b<sub>enc</sub>h<sub>mar</sub>k <sub>w</sub>h<sub>ere</sub> f<sub>ron</sub>ti<sub>er</sub> LLM<sub>s canno</sub>t <sub>re</sub>l<sub>y on pre-</sub>t<sub>ra</sub>i<sub>n</sub>i<sub>ng memor</sub>i<sub>za</sub>ti<sub>on.</sub> If tasks referenced real historical events (e.<sub>g</sub>., the Vioxx recall or Theranos investi<sub>g</sub>ation), a<sub>g</sub>ents could b<sub>ypass</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>mar</sub>k<sub>e</sub>t<sub>p</sub>l<sub>ace</sub> <sub>an</sub>d <sub>answer</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> <sub>parame</sub>t<sub>r</sub>i<sub>c</sub> <sub>memory.</sub>

Synthetic Multi-Publisher Corpus Generation. To guarantee zero pre-training leakage while <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng rea</sub>li<sub>s</sub>ti<sub>c mu</sub>lti<sub>-</sub>d<sub>ocumen</sub>t d<sub>epen</sub>d<sub>enc</sub>i<sub>es, we programma</sub>ti<sub>ca</sub>ll<sub>y syn</sub>th<sub>es</sub>i<sub>ze</sub> 1<sub>,</sub>000 <sub>co</sub>h<sub>er-</sub> ent cor<sub>p</sub>orate and scientific controversies (4<sub>,</sub>000 documents total) across 50 distinct technolo<sub>g</sub>ical sectors (e.g., Autonomous Vehicle Sensor Fusion, Solid-State EV Battery Electrolyte Leaks, Quantum Key Distribution). For each scenario � ∈ {0, . . . , 999}, we prompt Gemini-3.5-flash-lite to generate a se<sup>lf</sup>-cons<sup>i</sup>stent controversy <sup>i</sup>ncorporat<sup>i</sup>ng a man<sup>d</sup>atory un<sup>i</sup>que a<sup>l</sup>p<sup>h</sup>anumer<sup>i</sup>c <sup>id</sup>ent<sup>ifi</sup>er su<sup>fi</sup>x -k attac<sup>h</sup>e<sup>d</sup> to a<sup>ll</sup> pr<sup>i</sup>mary ent<sup>i</sup>t<sup>i</sup>es: t<sup>h</sup>e corporat<sup>i</sup>on name (e.g., Velosync-14), c<sup>li</sup>n<sup>i</sup>ca<sup>l</sup> or eng<sup>i</sup>neer<sup>i</sup>ng stu<sup>d</sup>y IDs (e.g., Trial-14), an<sup>d</sup> <sup>k</sup>ey soc<sup>i</sup>a<sup>l</sup> me<sup>di</sup>a <sup>h</sup>an<sup>dl</sup>es (e.g., @expert-14).

Each generated scenario is partitioned into four strictly disjoint publisher corpora:

1. Wikipedia (wikipedia): High-level encyclopedic overviews of the corporation, core technology, and back<sub>g</sub>round of the dis<sub>p</sub>ute (1<sub>,</sub>000 documents).

2. Medical & Technical Journals (medical\_journals): Peer-reviewed study abstracts containin<sub>g</sub> s<sub>p</sub>ecific al<sub>p</sub>hanumeric trial IDs and <sub>q</sub>uantitative <sub>p</sub>erformance or risk metrics (1<sub>,</sub>000 documents).

3. News Media (news\_media): Investigative journalism timelines detailing corporate announcements, executive (CEO) actions, and event dates (1<sub>,</sub>000 documents).

4. Social Media (social\_media): Public and expert reactions featuring specific user handles and direct <sub>q</sub>uotes (1<sub>,</sub>000 documents).

Multi-Hop Bridging Entity Tasks. Each scenario is paired with a multi-hop investigative research question constructed via the Bridging Entity (Indirect Reference) pattern. The user prompt provides on<sup>l</sup>y t<sup>h</sup>e <sup>i</sup>n<sup>i</sup>t<sup>i</sup>a<sup>l</sup> entry anc<sup>h</sup>or (t<sup>h</sup>e company name, e.g., Velosync-14), w<sup>hil</sup>e <sup>d</sup>ownstream targets (such as the s<sub>p</sub>ecific journal trial ID or whistleblower handle) are referenced indirectl<sub>y</sub> via attributes th<sub>a</sub>t <sub>can on</sub>l<sub>y</sub> b<sub>e uncovere</sub>d b<sub>y</sub> fi<sub>rs</sub>t <sub>re</sub>t<sub>r</sub>i<sub>ev</sub>i<sub>ng</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e</sub> Wiki<sub>pe</sub>di<sub>a or</sub> N<sub>ews</sub> d<sub>ocumen</sub>t<sub>s.</sub> Thi<sub>s</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure en</sub>f<sub>orces sequen</sub>ti<sub>a</sub>l <sub>mu</sub>lti<sub>-</sub>h<sub>op</sub> d<sub>epen</sub>d<sub>enc</sub>i<sub>es across</sub> 3 t<sub>o</sub> 8 <sub>s</sub>t<sub>eps.</sub>

Hybrid Dense-Sparse Vector Retrieval. All 4,000 documents are embedded into 768-dimensional <sup>d</sup>ense vectors us<sup>i</sup>ng Gecko em<sup>b</sup>e<sup>ddi</sup>ngs an<sup>d</sup> <sup>i</sup>n<sup>d</sup>exe<sup>d</sup> <sup>b</sup>y pu<sup>bli</sup>s<sup>h</sup>er category. At runt<sup>i</sup>me, w<sup>h</sup>en a <sub>w</sub>i<sub>nn</sub>i<sub>ng pu</sub>bli<sub>s</sub>h<sub>er agen</sub>t f<sub>ormu</sub>l<sub>a</sub>t<sub>es a na</sub>t<sub>ura</sub>l<sub>-</sub>l<sub>anguage searc</sub>h <sub>query</sub> $q ,$ th<sub>e</sub> l<sub>oca</sub>l <sub>searc</sub>h <sub>eng</sub>i<sub>ne</sub> <sub>compu</sub>t<sub>es a</sub> f<sub>use</sub>d <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y score over</sub> d<sub>ocumen</sub>t <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> $d _ { i }$ <sub>w</sub>ithi<sub>n</sub> th<sub>a</sub>t <sub>pu</sub>bli<sub>s</sub>h<sub>er</sub>’<sub>s corpus:</sub>

$$
\mathrm { S c o r e } ( q , d _ { i } ) = \frac { \mathbf { e } _ { q } ^ { \top } \mathbf { e } _ { d _ { i } } } { \| \mathbf { e } _ { q } \| \| \mathbf { e } _ { d _ { i } } \| } \times \left( 1 + 2 \cdot \mathbb { I } \big [ \mathrm { A l p h a n u m e r i c M a t c h } ( q , d _ { i } ) \big ] \right) ,\tag{7}
$$

where �[AlphanumericMatch $\left( q , d _ { i } \right) ]$ <sup>tri</sup>gge<sup>r</sup>s a $+ 2 . 0 \times$ <sub>sparse</sub> k<sub>eywor</sub>d b<sub>oos</sub>t if h<sub>yp</sub>h<sub>ena</sub>t<sub>e</sub>d <sub>or numer</sub>i<sub>c</sub> <sub>en</sub>tit<sub>y</sub> id<sub>en</sub>tifi<sub>ers</sub> i<sub>n</sub> th<sub>e query ma</sub>t<sub>c</sub>h th<sub>e</sub> d<sub>ocumen</sub>t t<sub>ex</sub>t <sub>or</sub> ID<sub>.</sub> B<sub>ecause</sub> 20 di<sub>s</sub>ti<sub>nc</sub>t <sub>scenar</sub>i<sub>os s</sub>h<sub>are</sub> each of the 50 industry topics, broad topical queries generated by coverage agents (�<sub>1</sub>) frequently <sub>su</sub>f<sub>er</sub> d<sub>ense-em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>co</sub>lli<sub>s</sub>i<sub>ons</sub> <sub>across</sub> <sub>scenar</sub>i<sub>os</sub> i<sub>n</sub> th<sub>e</sub> <sub>same</sub> <sub>sec</sub>t<sub>or.</sub> I<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> <sub>en</sub>tit<sub>y-spec</sub>ifi<sub>c</sub> queries formulated by precision agents $\left( c _ { 0 } \right)$ <sub>ac</sub>ti<sub>va</sub>t<sub>e</sub> th<sub>e sparse a</sub>l<sub>p</sub>h<sub>anumer</sub>i<sub>c</sub> b<sub>oos</sub>t<sub>er,</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>reso</sub>l<sub>v</sub>i<sub>ng</sub> <sub>name</sub> <sub>co</sub>lli<sub>s</sub>i<sub>ons</sub> <sub>an</sub>d d<sub>r</sub>i<sub>v</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> <sub>super</sub>i<sub>or</sub> d<sub>owns</sub>t<sub>ream</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>mar</sub>k<sub>e</sub>t<sub>p</sub>l<sub>ace.</sub>

News Market Synthetic Dataset Generator Prompt   
You are a world-class synthetic database generator. Your job is to generate   
highly realistic, detailed corporate or scientific controversies.   
Each controversy must be completely realistic, but entirely fictive (use fake   
company names, alphanumeric study IDs, made-up dates, and fake user handles)   
to prevent pre-training bias.   
The controversy domain/industry MUST be strictly focused on the sector:   
’topic’   
The controversy MUST incorporate the unique identifier suffix code ’-idx’ to   
guarantee absolute name uniqueness. You MUST append the suffix ’-idx’ to the   
company name (e.g., ’Velosync-idx’ or ’Aquafuse-idx’), to all   
trial/study/patent IDs $( { \bf e . g . , \Delta ^ { \prime } T r i a l { - i } d x ^ { ; } { \Delta \circ } { \Delta ^ { \prime } S t u d y { - i } d x ^ { \prime } } } )$ and to the expert   
Twitter handle (e.g. ’@user-idx’ or ’@expert-idx’).   
Generate a single coherent controversy scenario. The controversy must   
contain exactly:   
1. A Wikipedia Document: A high-level overview of the technology or   
corporation and the dispute.   
2. A Technical/Scientific Journal Document: A clinical, engineering, or   
physics paper abstract containing a specific alphanumeric study ID (e.g.,   
Trial NCT10485 or Paper AST9912) and numeric performance metrics (e.g.,   
percentages, relative risk, latency ms).   
3. A News Media Document: A corporate drama/timeline piece describing   
corporate events, a CEO name, and dates.   
4. A Social Media Document: A public reaction or outpatient outrage comment   
featuring a specific Twitter handle (e.g., @user\_handle) and a quote.   
You MUST also generate exactly one highly coherent, multi-hop research   
question and its gold standard answer.

The question MUST be written using the Bridging Entity (Indirect Reference)   
pattern:   
- The question MUST explicitly name the starting company (the entry anchor,   
e.g., ’Velosync-idx’ or ’Terravolt-idx’) as the initial search coordinate.   
- The subsequent facts (the Journal study ID or Social handle/outrage quote)   
MUST be referred to indirectly based on attributes discovered only in the   
Wikipedia or News documents, creating a strictly ordered, sequential   
dependency chain.   
- The question and answer MUST only reference facts within this specific   
controversy (meaning all facts are fully coherent and belong to the same   
company!).

Illustrative Multi-Hop Scenario Walkthrough. Table 5 illustrates a complete scenario (task\_luminos\_fusion from the generated News Market benchmark (completely synthetic), demonstrating how the Bridging Entity structure enforces sequential cross-publisher routing. Note that the user query provides only a soc<sup>i</sup>a<sup>l</sup> me<sup>di</sup>a <sup>h</sup>an<sup>dl</sup>e (@TechCritic-0) w<sup>i</sup>t<sup>h</sup>out nam<sup>i</sup>ng t<sup>h</sup>e corporat<sup>i</sup>on or tec<sup>h</sup>n<sup>i</sup>ca<sup>l</sup> stu<sup>d</sup>y.

Table 5 | Complete document set and multi-hop resolution trace for Scenario � = 0 in the News Market benchmark. Bridging entities discovered at each step are highlighted in bold.
<table><tr><td>Component / Corpus</td><td>Content &amp; Bridging Entities</td></tr><tr><td>User Question</td><td>&quot;Regarding the company that was publicly criticized by @TechCritic-0 for its soft- ware deployment practices, what is the specific relative risk percentage identified in the technical document that triggered the internal whistleblower investigation?&quot;</td></tr><tr><td>Gold Answer 1. Social Media (social_media)</td><td>&quot;The relative risk percentage identified in Study-0 (LUMINOS-ENG-772-0) is 22%.&quot; [doc_id:social_luminos_fusion_latency_scandal] “@TechCritic-0: It is absolutely horrifying that Marcus Vane knew our safety</td></tr><tr><td>Entry Anchor Match 2. News Media</td><td>was at risk just to hit a deployment deadline. I am literally shaking reading the 22% failure rate stat—if the car can&#x27;t see the pedestrian in time, the fusion&#x27; is just a fancy way to commit insurance fraud on our lives.&quot; Bridging Entity Unlocked: CEO name Marcus Vane. [doc_id: news_luminos_fusion_latency_scandal]</td></tr><tr><td>(news_media) Intermediate Hop</td><td>&quot;On August 12, 2024, Luminos-0 CEO Marcus Vane publicly addressed internal whistleblower reports suggesting that the company&#x27;s engineering leads were aware of the latency spikes documented in Study-0 since Q1. Despite internal warnings, the company proceeded with the wide-scale deployment of its Fusion- V3&#x27; software update...&quot; Bridging Entities Unlocked: Company Luminos-0 and Study-0.</td></tr><tr><td>3. Technical Journal (medical_journals) Terminal Evidence Hop</td><td>[doc_id: journal_luminos_fusion_latency_scandal] &quot;Study-0 (Ref: LUMINOS-ENG-772-0) investigated the cross-modal temporal alignment of the Luminos-0 fusion stack. Testing under extreme lighting and weather conditions revealed a critical processing bottleneck... This delay corre- lates directly with a 22% increase in the relative risk of failing to classify dynamic pedestrians during rapid lighting transitions.&quot; Target Evidence Extracted: 22% relative risk in Study-0 (LUMINOS-ENG-</td></tr><tr><td>4. Wikipedia (wikipedia) Background Context</td><td>772-0). [doc_id:wiki_luminos_fusion_latency_scandal] &quot;Luminos-0 is a Silicon Valley-based technology firm specializing in proprietary multi-modal sensor fusion architectures for Level 4 autonomous driving systems... Controversy emerged in mid-2024 following allegations that the company&#x27;s sensor fusion algorithms prioritized high-confidence object detection over real-time environmental updates...&quot;</td></tr></table>