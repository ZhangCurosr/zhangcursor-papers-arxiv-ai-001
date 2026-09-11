# CryptoL: Towards Scale Dominance and Physics Constraints Mitigation in Financial Multivariate Time Series Forecasting

Yalda Taheri<sup>1∗</sup>, Mohammad Hassan Heydari<sup>2∗</sup>, Armon Rasooli<sup>3†</sup>, Maryam Amirshahkarami<sup>2†</sup>, Mohammad Ebrahim Mahdavi<sup>2†</sup>, Hossein Karshenas<sup>2</sup>

<sup>1</sup>F<sub>acu</sub>lt<sub>y</sub> <sub>o</sub>f E<sub>ng</sub>i<sub>neer</sub>i<sub>ng,</sub> A<sub>za</sub>d U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub>

<sup>2</sup>F<sub>acu</sub>lt<sub>y</sub> <sub>o</sub>f C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r En<sub>g</sub>in<sub>ee</sub>rin<sub>g,</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f I<sub>s</sub>f<sub>a</sub>h<sub>a</sub>n

<sup>3</sup>D<sub>epar</sub>t<sub>men</sub>t <sub>o</sub>f El<sub>ec</sub>t<sub>r</sub>i<sub>ca</sub>l E<sub>ng</sub>i<sub>neer</sub>i<sub>ng,</sub> I<sub>ran</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub> <sub>o</sub>f S<sub>c</sub>i<sub>ence</sub> <sub>an</sub>d T<sub>ec</sub>h<sub>no</sub>l<sub>ogy</sub>

y.jaliltaheri@iau.ir, m.heydari@mehr.ui.ac.ir, a.rasouli@ec.iut.ac.ir

<sub>maryamam</sub>i<sub>rs</sub>h<sub>a</sub>hk<sub>aram</sub>i@<sub>me</sub>h<sub>r.u</sub>i<sub>.ac.</sub>i<sub>r, m</sub>h<sub>m.e</sub>b<sub>r</sub>h<sub>.ma</sub>hd<sub>av</sub>i@<sub>gma</sub>il<sub>.com,</sub> h<sub>.</sub>k<sub>ars</sub>h<sub>enas</sub>@<sub>eng.u</sub>i<sub>.ac.</sub>i<sub>r</sub>

## Abstract

C<sub>ryp</sub>t<sub>ocurrency</sub> f<sub>orecas</sub>ti<sub>ng presen</sub>t<sub>s a</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>ve com</sub>bi<sub>-</sub> <sub>na</sub>ti<sub>on o</sub>f <sub>ex</sub>t<sub>reme cross-asse</sub>t <sub>sca</sub>l<sub>e</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y, non-</sub> <sub>s</sub>t<sub>a</sub>ti<sub>onary</sub> d<sub>ynam</sub>i<sub>cs,</sub> <sub>an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l d<sub>epen</sub>d<sub>enc</sub>i<sub>es</sub> <sub>among</sub> O<sub>p</sub>en–Hi<sub>g</sub>h–Low–Close (OHLC) variables. We <sub>p</sub>resent Cr<sub>yp</sub>- t<sub>o</sub>L<sub>, a un</sub>ifi<sub>e</sub>d f<sub>ramewor</sub>k d<sub>es</sub>i<sub>gne</sub>d t<sub>o a</sub>dd<sub>ress</sub> th<sub>ese c</sub>h<sub>a</sub>ll<sub>enges</sub> <sub>w</sub>ithi<sub>n mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> ti<sub>me-ser</sub>i<sub>es</sub> f<sub>orecas</sub>ti<sub>ng.</sub> C<sub>ryp</sub>t<sub>o</sub>L <sub>eva</sub>l<sub>ua</sub>t<sub>es</sub> f<sub>orecas</sub>ti<sub>ng error</sub> i<sub>n con</sub>t<sub>ex</sub>t<sub>-norma</sub>li<sub>ze</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es w</sub>ithi<sub>n</sub> th<sub>e</sub> R<sub>ev</sub>IN <sub>p</sub>i<sub>pe</sub>li<sub>ne, preven</sub>ti<sub>ng</sub> i<sub>nverse norma</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>rom</sub> i<sub>n</sub>t<sub>ro-</sub> d<sub>uc</sub>i<sub>ng an a</sub>dditi<sub>ona</sub>l <sub>square</sub>d<sub>-sca</sub>l<sub>e we</sub>i<sub>g</sub>hti<sub>ng</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> MSE objective. We formally characterize this efect through the <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>r</sub>i<sub>s</sub>k <sub>an</sub>d <sub>parame</sub>t<sub>er-gra</sub>di<sub>en</sub>t <sub>geome</sub>t<sub>ry, es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> th<sub>e con</sub>diti<sub>ons un</sub>d<sub>er w</sub>hi<sub>c</sub>h l<sub>arge-sca</sub>l<sub>e asse</sub>t<sub>s can</sub> di<sub>spropor-</sub> ti<sub>ona</sub>t<sub>e</sub>l<sub>y</sub> i<sub>n</sub>fl<sub>uence s</sub>h<sub>are</sub>d<sub>-mo</sub>d<sub>e</sub>l <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on.</sub> B<sub>eyon</sub>d l<sub>oss-</sub> <sub>space norma</sub>li<sub>za</sub>ti<sub>on,</sub> C<sub>ryp</sub>t<sub>o</sub>L <sub>exam</sub>i<sub>nes c</sub>h<sub>anne</sub>l<sub>-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>an</sub>d <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or</sub> OHLC d<sub>a</sub>t<sub>a, s</sub>h<sub>ow-</sub> i<sub>ng</sub> th<sub>a</sub>t <sub>a s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on pre-</sub> <sub>serves can</sub>dl<sub>e-or</sub>d<sub>er re</sub>l<sub>a</sub>ti<sub>ons</sub> th<sub>a</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>c</sub>h<sub>anne</sub>l t<sub>rans-</sub> f<sub>orma</sub>ti<sub>ons nee</sub>d <sub>no</sub>t <sub>preserve.</sub> Th<sub>e</sub> f<sub>ramewor</sub>k f<sub>ur</sub>th<sub>er</sub> i<sub>ncor-</sub> <sub>pora</sub>t<sub>es sca</sub>l<sub>e-a</sub>d<sub>ap</sub>ti<sub>ve numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>bili<sub>za</sub>ti<sub>on</sub> t<sub>o re</sub>d<sub>uce</sub> di<sub>s-</sub> t<sub>or</sub>ti<sub>ons cause</sub>d b<sub>y a</sub> fi<sub>xe</sub>d <sub>norma</sub>li<sub>za</sub>ti<sub>on cons</sub>t<sub>an</sub>t <sub>across asse</sub>t<sub>s</sub> <sub>spann</sub>i<sub>ng many or</sub>d<sub>ers o</sub>f <sub>magn</sub>it<sub>u</sub>d<sub>e,</sub> t<sub>oge</sub>th<sub>er w</sub>ith <sub>a so</sub>ft f<sub>ea-</sub> <sub>s</sub>ibilit<sub>y</sub> l<sub>oss</sub> th<sub>a</sub>t <sub>pena</sub>li<sub>zes v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ons o</sub>f th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> OHLC i<sub>n-</sub> e<sub>q</sub>ua<sup>li</sup>t<sup>i</sup>es. <sup>E</sup>x<sub>p</sub>er<sup>i</sup>ments across <sup>h</sup>etero<sub>g</sub>eneous cr<sub>yp</sub>tocurrenc<sub>y</sub> <sub>asse</sub>t<sub>s eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>ese componen</sub>t<sub>s</sub> th<sub>roug</sub>h <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d <sub>a</sub>bl<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> i<sub>mprovemen</sub>t<sub>s</sub> i<sub>n</sub> f<sub>orecas</sub>ti<sub>ng accuracy,</sub> t<sub>ra</sub>i<sub>n-</sub> i<sub>ng s</sub>t<sub>a</sub>bilit<sub>y, an</sub>d th<sub>e</sub> f<sub>requency o</sub>f fi<sub>nanc</sub>i<sub>a</sub>ll<sub>y va</sub>lid OHLC <sub>pre-</sub> di<sub>c</sub>ti<sub>ons re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e cons</sub>id<sub>ere</sub>d b<sub>ase</sub>li<sub>nes.</sub> C<sub>ryp</sub>t<sub>o</sub>L th<sub>ere-</sub> f<sub>ore prov</sub>id<sub>es an</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub>d <sub>approac</sub>h t<sub>o sca</sub>l<sub>e-</sub>b<sub>a</sub>l<sub>ance</sub>d <sub>op</sub>ti<sub>-</sub> <sub>m</sub>i<sub>za</sub>ti<sub>on, s</sub>t<sub>ruc</sub>t<sub>ure-preserv</sub>i<sub>ng norma</sub>li<sub>za</sub>ti<sub>on, numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a-</sub> bili<sub>za</sub>ti<sub>on, an</sub>d <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>-aware cryp</sub>t<sub>ocurrency</sub> f<sub>orecas</sub>ti<sub>ng.</sub>

## Introduction

M<sub>u</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> fi<sub>nanc</sub>i<sub>a</sub>l f<sub>orecas</sub>ti<sub>ng</sub> us<sup>i</sup>n<sub>g</sub> o<sub>p</sub>en–hi<sub>g</sub>h–low–close (OHLC) data <sub>p</sub>rovides a detailed <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f <sub>pr</sub>i<sub>ce</sub> <sub>evo</sub>l<sub>u</sub>ti<sub>on,</sub> b<sub>u</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>ng</sub> th<sub>ese</sub> hi<sub>g</sub>hl<sub>y</sub> coupled variables jointly is challenging due to abrupt regime chan<sub>g</sub>es and strict <sub>p</sub>h<sub>y</sub>sical candlestick rules (Das, Go<sub>y</sub>al, <sub>an</sub>d Y<sub>a</sub>d<sub>av</sub> 2026<sub>;</sub> S<sub>er</sub>d<sub>ar e</sub>t <sub>a</sub>l<sub>.</sub> 2021<sub>;</sub> Si<sub>m</sub>th<sub>ara</sub>k<sub>ao</sub> 2022<sub>;</sub> Wan<sub>g</sub>, Huan<sub>g</sub>, and Wan<sub>g</sub> 2021). This task is es<sub>p</sub>eciall<sub>y</sub> d<sub>eman</sub>di<sub>ng</sub> i<sub>n cryp</sub>t<sub>ocurrency mar</sub>k<sub>e</sub>t<sub>s, w</sub>hi<sub>c</sub>h <sub>ex</sub>hibit <sub>ex</sub>t<sub>reme non-s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y an</sub>d <sub>cross-asse</sub>t <sub>sca</sub>l<sub>e</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> (Simtharakao 2022; Kim et al. 2021; Berthelier et al. 2026). M<sub>o</sub>d<sub>e</sub>l<sub>s mus</sub>t l<sub>earn s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>rom asse</sub>t<sub>s rang</sub>i<sub>ng</sub> f<sub>rom</sub> f<sub>rac</sub>ti<sub>ons o</sub>f <sub>a cen</sub>t t<sub>o</sub> t<sub>ens o</sub>f th<sub>ousan</sub>d<sub>s o</sub>f d<sub>o</sub>ll<sub>ars, ma</sub>ki<sub>ng</sub> joint optimization dificult when using standard forecasting b<sub>ac</sub>kb<sub>ones</sub> th<sub>a</sub>t <sub>s</sub>t<sub>rugg</sub>l<sub>e w</sub>ith t<sub>empora</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on s</sub>hift<sub>s an</sub>d scale diferences (Kim et al. 2021; Liu et al. 2023; Ye et al. 2024<sub>;</sub> B<sub>er</sub>th<sub>e</sub>li<sub>er e</sub>t <sub>a</sub>l<sub>.</sub> 2026<sub>;</sub> F<sub>eng,</sub> H<sub>uang, an</sub>d K<sub>rompass</sub> 2024).

While Reversible Instance Normalization (RevIN) is com-<sub>mon</sub>l<sub>y</sub> <sub>use</sub>d t<sub>o</sub> <sub>m</sub>iti<sub>ga</sub>t<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>s</sub>hift<sub>s,</sub> it<sub>s</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a-</sub> ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> h<sub>ave s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>an</sub>d <sub>numer</sub>i<sub>ca</sub>l <sub>con-</sub> se uences for OHLC data (Kim et al. 2021; Simtharakao 2022; Te<sub>p</sub>el<sub>y</sub>an 2025). Channel-inde<sub>p</sub>endent standardization <sub>can</sub> di<sub>srup</sub>t th<sub>e re</sub>l<sub>a</sub>ti<sub>ve or</sub>d<sub>er</sub>i<sub>ng o</sub>f <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>c</sub>h<sub>anne</sub>l<sub>s,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>preserves</sub> th<sub>ese</sub> <sub>essen</sub>ti<sub>a</sub>l <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps.</sub> Additi<sub>ona</sub>ll<sub>y, s</sub>t<sub>an</sub>d<sub>ar</sub>d R<sub>ev</sub>IN <sub>eva</sub>l<sub>u-</sub> <sub>a</sub>t<sub>es</sub> f<sub>orecas</sub>ti<sub>ng errors</sub> i<sub>n</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>sca</sub>l<sub>e, w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>c</sub>itl<sub>y pr</sub>i<sub>or</sub>iti<sub>zes</sub> hi<sub>g</sub>h<sub>-va</sub>l<sub>ue asse</sub>t<sub>s</sub> d<sub>ur</sub>i<sub>ng s</sub>h<sub>are</sub>d <sub>mo</sub>d<sub>e</sub>l o<sub>p</sub>timization (Kim et al. 2021; Berthelier et al. 2026; Liu <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2023<sub>;</sub> Y<sub>e e</sub>t <sub>a</sub>l<sub>.</sub> 2024<sub>;</sub> F<sub>eng,</sub> H<sub>uang, an</sub>d K<sub>rompass</sub> 2024). Trainin<sub>g</sub> directl<sub>y</sub> in normalized tar<sub>g</sub>et s<sub>p</sub>ace which is referred to as Two-Phase RevIN (TP-RevIN), removes this <sub>sca</sub>l<sub>e-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>we</sub>i<sub>g</sub>hti<sub>ng, w</sub>hil<sub>e sca</sub>l<sub>e-a</sub>d<sub>ap</sub>ti<sub>ve s</sub>t<sub>a</sub>bili<sub>za-</sub> ti<sub>on preven</sub>t<sub>s</sub> fi<sub>xe</sub>d <sub>norma</sub>li<sub>za</sub>ti<sub>on cons</sub>t<sub>an</sub>t<sub>s</sub> f<sub>rom</sub> di<sub>s</sub>t<sub>or</sub>ti<sub>ng</sub> low-valued series (Berthelier et al. 2026; Fen<sub>g</sub>, Huan<sub>g</sub>, and Krom<sub>p</sub>ass 2024).

E<sub>ven w</sub>ith <sub>proper norma</sub>li<sub>za</sub>ti<sub>on, mo</sub>d<sub>e</sub>l<sub>s can s</sub>till <sub>pro-</sub> d<sub>uce</sub> fi<sub>nanc</sub>i<sub>a</sub>ll<sub>y</sub> i<sub>na</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e pre</sub>di<sub>c</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>e core</sub> candlestick inequalities (Wan<sub>g</sub>, Huan<sub>g</sub>, and Wan<sub>g</sub> 2021; Simtharakao 2022). Previous a<sub>pp</sub>roaches addressed this <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> th<sub>roug</sub>h <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>arc</sub>hit<sub>ec</sub>t<sub>ures or</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons,</sub> b<sub>u</sub>t th<sub>ese</sub> <sub>are</sub> <sub>o</sub>ft<sub>en</sub> difi<sub>cu</sub>lt t<sub>o</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub> <sub>w</sub>ith <sub>mo</sub>d<sub>ern</sub> ti<sub>me-ser</sub>i<sub>es</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> T<sub>o</sub> <sub>a</sub>dd<sub>ress</sub> th<sub>ese</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>ons, we presen</sub>t C<sub>rypto</sub>L<sub>, a un</sub>ifi<sub>e</sub>d f<sub>ramewor</sub>k th<sub>a</sub>t <sub>com</sub>bi<sub>nes</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure-aware</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>sca</sub>l<sub>e-</sub>b<sub>a</sub>l<sub>ance</sub>d <sub>op-</sub> ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>v</sub>i<sub>a</sub> TP<sub>-</sub>R<sub>ev</sub>IN<sub>,</sub> <sub>an</sub>d <sub>an</sub> <sub>aux</sub>ili<sub>ary</sub> <sub>p</sub>h<sub>ys</sub>i<sub>cs-</sub>i<sub>n</sub>f<sub>orme</sub>d l<sub>oss.</sub> Thi<sub>s</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub>d <sub>approac</sub>h di<sub>scourages</sub> i<sub>nva</sub>lid <sub>can</sub>dl<sub>e-</sub> <sub>s</sub>ti<sub>c</sub>k <sub>s</sub>t<sub>ruc</sub>t<sub>ures w</sub>hil<sub>e ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>bilit<sub>y an</sub>d f<sub>orecas</sub>ti<sub>ng accuracy across</sub> di<sub>verse asse</sub>t <sub>sca</sub>l<sub>es.</sub>

![](images/5cba1456b5979a1910323e0afda08da4178f9049d7f5c065a35df817a8f70532.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 1<sub>:</sub> Cr<sub>yp</sub>t<sub>o</sub>L fr<sub>a</sub>m<sub>ewo</sub>rk

Th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons o</sub>f thi<sub>s wor</sub>k <sub>are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• We s<sub>y</sub>stematicall<sub>y</sub> anal<sub>y</sub>ze channel-inde<sub>p</sub>endent and <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t R<sub>ev</sub>IN f<sub>or mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> OHLC f<sub>ore-</sub> X̂ <sub>cas</sub>ti<sub>ng,</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> dif<sub>eren</sub>t <sub>e</sub>f<sub>ec</sub>t<sub>s on can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>s</sub>t<sub>ruc</sub>t<sub>ure, an</sub>d i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>sca</sub>l<sub>e-a</sub>d<sub>ap</sub>ti<sub>ve eps</sub>il<sub>on</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> f<sub>or s</sub>t<sub>a</sub>bl<sub>e norma</sub>li<sub>za</sub>ti<sub>on across asse</sub>t<sub>s w</sub>ith <sub>ex</sub>t<sub>reme numer</sub>i<sub>ca</sub>l <sub>ranges.</sub>

• We <sub>p</sub>rovide a theoretical and extensive em<sub>p</sub>irical anal<sub>y</sub>sis <sub>o</sub>f TP<sub>-</sub>R<sub>ev</sub>IN<sub>, equ</sub>i<sub>va</sub>l<sub>e</sub>ntl<sub>y</sub> n<sub>o</sub>rm<sub>a</sub>li<sub>ze</sub>d<sub>-space</sub> MSE tr<sub>a</sub>in<sub>-</sub> ing, showing how the location of the training objective <sub>a</sub>f<sub>ec</sub>t<sub>s sca</sub>l<sub>e-</sub>i<sub>n</sub>d<sub>uce</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>m</sub>b<sub>a</sub>l<sub>ance</sub> i<sub>n</sub> h<sub>e</sub>t<sub>ero-</sub> g<sup>eneo</sup>u<sup>s</sup> <sup>cr</sup>yp<sup>toc</sup>u<sup>rrenc</sup>y <sup>forecastin</sup>g.

• We introduce an auxiliar<sub>y p</sub>h<sub>y</sub>sics-informed loss that <sub>p</sub>e-<sub>na</sub>li<sub>zes v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ons o</sub>f th<sub>e</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c</sub> OHLC <sub>or</sub>d<sub>er</sub>i<sub>ng con-</sub> <sub>s</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s an</sub>d <sub>compare</sub> th<sub>e resu</sub>lti<sub>ng</sub> f<sub>ramewor</sub>k <sub>w</sub>ith <sub>un-</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>ne</sub>d f<sub>orecas</sub>ti<sub>ng, conven</sub>ti<sub>ona</sub>l R<sub>ev</sub>IN <sub>var</sub>i<sub>an</sub>t<sub>s, an</sub>d <sub>a</sub>lt<sub>erna</sub>ti<sub>ve norma</sub>li<sub>za</sub>ti<sub>on approac</sub>h<sub>es.</sub>

## Related Work

M<sub>u</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> fi<sub>nanc</sub>i<sub>a</sub>l f<sub>orecas</sub>ti<sub>ng a</sub>i<sub>ms</sub> t<sub>o mo</sub>d<sub>e</sub>l t<sub>empo-</sub> ral, cross-variable, and cross-asset dependencies jointly (Simtharakao 2022; Das, Go<sub>y</sub>al, and Yadav 2026; Serdar et al. 2021; Te<sub>p</sub>el<sub>y</sub>an 2025). Classical a<sub>pp</sub>roaches have used <sub>s</sub>t<sub>a</sub>t<sub>e-space mo</sub>d<sub>e</sub>l<sub>s,</sub> K<sub>a</sub>l<sub>man</sub> filt<sub>er</sub>i<sub>ng, an</sub>d d<sub>ynam</sub>i<sub>c con</sub>di<sub>-</sub> ti<sub>ona</sub>l <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>w</sub>hil<sub>e</sub> <sub>recen</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>emp</sub>l<sub>oy</sub> <sub>re-</sub> <sub>curren</sub>t <sub>ne</sub>t<sub>wor</sub>k<sub>s, au</sub>t<sub>oenco</sub>d<sub>ers,</sub> T<sub>rans</sub>f<sub>ormers, an</sub>d MLP<sub>-</sub> b<sub>ase</sub>d <sub>arc</sub>hit<sub>ec</sub>t<sub>ures.</sub> St<sub>oc</sub>kMi<sub>xer</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>or,</sub> t<sub>empora</sub>l<sub>,</sub> <sub>an</sub>d <sub>s</sub>t<sub>oc</sub>k<sub>-</sub>l<sub>eve</sub>l i<sub>n</sub>t<sub>erac</sub>ti<sub>ons, w</sub>h<sub>ereas</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on-mo</sub>d<sub>e</sub>l <sub>s</sub>t<sub>u</sub>d<sub>-</sub> i<sub>es</sub> h<sub>ave</sub> <sub>s</sub>h<sub>own</sub> th<sub>a</sub>t <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> i<sub>npu</sub>t<sub>s</sub> <sub>can</sub> i<sub>mprove</sub> financial forecastin<sub>g</sub> (Fan and Shen 2024). DPP instead learns <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> d<sub>ecompose</sub>d <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>c</sub>h<sub>ar</sub>t<sub>s</sub> to <sub>p</sub>redict future <sub>p</sub>rice movements (Hun<sub>g</sub> and Chen 2021).

OHLC f<sub>orecas</sub>ti<sub>n</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces a</sub>dditi<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>re-</sub> <sub>qu</sub>i<sub>remen</sub>t<sub>s</sub> b<sub>ecause va</sub>lid <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k<sub>s mus</sub>t <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> fi<sub>xe</sub>d <sup>Ŷ Ŷ</sup><sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>ps among</sub> th<sub>e open,</sub> hi<sub>g</sub>h<sub>,</sub> l<sub>ow, an</sub>d <sub>c</sub>l<sub>ose pr</sub>i<sub>ces</sub> (Simtharakao 2022; Te<sub>p</sub>el<sub>y</sub>an 2025; Wan<sub>g</sub>, Huan<sub>g</sub>, and Wan<sub>g</sub> 2021). Existin<sub>g</sub> studies have addressed these constraints th<sub>roug</sub>h i<sub>nver</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>guaran</sub>t<sub>ee va</sub>lid <sub>re-</sub> constructed out<sub>p</sub>uts (Wan<sub>g</sub>, Huan<sub>g</sub>, and Wan<sub>g</sub> 2021), as well <sub>as</sub> h<sub>y</sub>b<sub>r</sub>id <sub>au</sub>t<sub>oenco</sub>d<sub>er an</sub>d <sub>mu</sub>ltit<sub>as</sub>k <sub>arc</sub>hit<sub>ec</sub>t<sub>ures</sub> d<sub>es</sub>i<sub>gne</sub>d to model channel de<sub>p</sub>endencies (Chakrabort<sub>y</sub>, Ghosh, and Ghosh 2022). Other work has incor<sub>p</sub>orated the timestam<sub>p</sub>s of OHLC events to enrich the bar re<sub>p</sub>resentation (Das, Go<sub>y</sub>al, <sub>an</sub>d Y<sub>a</sub>d<sub>av</sub> 2026<sub>;</sub> S<sub>er</sub>d<sub>ar e</sub>t <sub>a</sub>l<sub>.</sub> 2021<sub>;</sub> Si<sub>m</sub>th<sub>ara</sub>k<sub>ao</sub> 2022<sub>;</sub> Fiszeder, Fałdziński, and Molnár 2023). These studies indi<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t OHLC f<sub>orecas</sub>ti<sub>ng</sub> i<sub>s no</sub>t <sub>a s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> <sub>regress</sub>i<sub>on pro</sub>bl<sub>em, s</sub>i<sub>nce pre</sub>di<sub>c</sub>ti<sub>ons mus</sub>t <sub>preserve</sub> th<sub>e</sub> i<sub>n-</sub> t<sub>erna</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure o</sub>f <sub>eac</sub>h <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k<sub>.</sub>

N<sub>orma</sub>li<sub>za</sub>ti<sub>on me</sub>th<sub>o</sub>d<sub>s</sub> h<sub>ave</sub> b<sub>een w</sub>id<sub>e</sub>l<sub>y s</sub>t<sub>u</sub>di<sub>e</sub>d f<sub>or</sub> h<sub>an-</sub> dli<sub>ng non-s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y an</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on s</sub>hift i<sub>n</sub> ti<sub>me-ser</sub>i<sub>es</sub> f<sub>orecas</sub>ti<sub>ng.</sub> R<sub>ev</sub>IN <sub>removes</sub> i<sub>ns</sub>t<sub>ance-spec</sub>ifi<sub>c s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs</sub> b<sub>e</sub>f<sub>ore</sub> forecastin<sub>g</sub> and restores them afterward (Kim et al. 2021). SAN <sub>ex</sub>t<sub>en</sub>d<sub>s</sub> thi<sub>s pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> th<sub>roug</sub>h l<sub>oca</sub>l t<sub>empora</sub>l<sub>-s</sub>li<sub>ce nor-</sub> malization (Liu et al. 2023), while FAN uses dominant fre-<sub>quency</sub> <sub>componen</sub>t<sub>s</sub> t<sub>o</sub> <sub>a</sub>dd<sub>ress</sub> b<sub>o</sub>th t<sub>ren</sub>d <sub>an</sub>d <sub>seasona</sub>l <sub>non-</sub> stationarit<sub>y</sub> (Ye et al. 2024). However, most normalization <sub>s</sub>t<sub>u</sub>di<sub>es</sub> f<sub>ocus</sub> <sub>on</sub> t<sub>empora</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>c</sub>h<sub>anges</sub> <sub>an</sub>d <sub>g</sub>i<sub>ve</sub> l<sub>ess</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> t<sub>o</sub> h<sub>ow</sub> th<sub>e norma</sub>li<sub>za</sub>ti<sub>on ax</sub>i<sub>s a</sub>f<sub>ec</sub>t<sub>s s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e ou</sub>t <sub>u</sub>t<sub>s suc</sub>h <sub>as</sub> OHLC <sub>c</sub>h<sub>anne</sub>l<sub>s.</sub>

R<sub>ecen</sub>t <sub>wor</sub>k h<sub>as</sub> f<sub>ur</sub>th<sub>er</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>e</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>nor-</sub> <sub>ma</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e sys</sub>t<sub>em</sub> i<sub>n w</sub>hi<sub>c</sub>h th<sub>e</sub> t<sub>ra</sub>i<sub>n-</sub> i<sub>ng</sub> l<sub>oss</sub> i<sub>s</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d<sub>.</sub> GTT t<sub>ra</sub>i<sub>ns</sub> <sub>on</sub> t<sub>arge</sub>t<sub>s</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>us</sub>i<sub>ng</sub> i<sub>npu</sub>t<sub>-con</sub>t<sub>ex</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs,</sub> th<sub>ere</sub>b<sub>y</sub> l<sub>earn</sub>i<sub>ng</sub> di<sub>rec</sub>tl<sub>y</sub> i<sub>n</sub> a unified curve-sha<sub>p</sub>e s<sub>p</sub>ace (Fen<sub>g</sub>, Huan<sub>g</sub>, and Krom<sub>p</sub>ass 2024). A com<sub>p</sub>arative stud<sub>y</sub> of normalization in foundation <sub>mo</sub>d<sub>e</sub>l<sub>s s</sub>h<sub>owe</sub>d th<sub>a</sub>t MSE <sub>an</sub>d MAE <sub>rema</sub>i<sub>n sca</sub>l<sub>e sens</sub>iti<sub>ve</sub> <sub>w</sub>h<sub>en pre</sub>di<sub>c</sub>ti<sub>ons are</sub> d<sub>enorma</sub>li<sub>ze</sub>d b<sub>e</sub>f<sub>ore</sub> l<sub>oss ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> (Ahmed et al. 2026). Another recent anal<sub>y</sub>sis of RevIN com-<sub>pare</sub>d <sub>conven</sub>ti<sub>ona</sub>l <sub>an</sub>d <sub>norma</sub>li<sub>ze</sub>d b<sub>ac</sub>k<sub>propaga</sub>ti<sub>on</sub> th<sub>roug</sub>h <sub>norma</sub>li<sub>ze</sub>d MSE <sub>an</sub>d <sub>repor</sub>t<sub>e</sub>d b<sub>ene</sub>fit<sub>s</sub> f<sub>or</sub> h<sub>e</sub>t<sub>erogeneous-</sub> scale data (Berthelier et al. 2026). Buildin<sub>g</sub> on these directions, our work jointly studies OHLC-aware normalizati<sub>on, norma</sub>li<sub>ze</sub>d<sub>-space op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on, a</sub>d<sub>ap</sub>ti<sub>ve numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a-</sub> bili<sub>za</sub>ti<sub>on, an</sub>d <sub>aux</sub>ili<sub>ary en</sub>f<sub>orcemen</sub>t <sub>o</sub>f <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>va</sub>lidit<sub>y.</sub>

## Methodology

## Preliminaries and Problem Definition

L<sub>e</sub>t

$$
\begin{array} { r } { \begin{array} { c } { \mathcal { D } = \{ ( \mathbf { X } _ { n } , \mathbf { Y } _ { n } ) \} _ { n = 1 } ^ { N } , } \\ { \mathbf { X } _ { n } \in \mathbb { R } ^ { L \times 4 } , \qquad \mathbf { Y } _ { n } \in \mathbb { R } ^ { H \times 4 } , } \end{array} } \end{array}\tag{1}
$$

d<sub>eno</sub>t<sub>e</sub> <sub>a</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> <sub>o</sub>f <sub>cryp</sub>t<sub>ocurrency</sub> f<sub>orecas</sub>ti<sub>ng</sub> <sub>samp</sub>l<sub>es,</sub> <sub>w</sub>h<sub>ere</sub> ${ \bf X } _ { n }$ is an observed context window of length L and ${ \bf Y } _ { n }$ is the corresponding forecast horizon of length H. The f<sub>our c</sub>h<sub>anne</sub>l<sub>s are or</sub>d<sub>ere</sub>d <sub>as</sub>

$$
{ \bf X } _ { n , t , : } = \left( O _ { n , t } , H _ { n , t } , L _ { n , t } , C _ { n , t } \right) ,\tag{2}
$$

<sub>represen</sub>ti<sub>ng</sub> th<sub>e open,</sub> hi<sub>g</sub>h<sub>,</sub> l<sub>ow, an</sub>d <sub>c</sub>l<sub>ose pr</sub>i<sub>ces, respec-</sub> ti<sub>ve</sub>l<sub>y.</sub> A <sub>va</sub>lid OHLC <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> set

$$
\begin{array} { r } { \Omega _ { \mathrm { O H L C } } = \{ ( O , H , L , C ) \in \mathbb { R } ^ { 4 } : H \geq \operatorname* { m a x } ( O , C ) , } \\ { L \leq \operatorname* { m i n } ( O , C ) \} . } \end{array}\tag{3}
$$

Gi<sub>ven</sub> <sub>a</sub> f<sub>orecas</sub>ti<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l $f _ { \theta }$ , the objective is to estimate

$$
{ \widehat { \mathbf { Y } } } _ { n } = f _ { \pmb { \theta } } ( \mathbf { X } _ { n } )\tag{4}
$$

<sub>across cryp</sub>t<sub>ocurrenc</sub>i<sub>es w</sub>h<sub>ose numer</sub>i<sub>ca</sub>l <sub>sca</sub>l<sub>es may</sub> dif<sub>er</sub> b<sub>y severa</sub>l <sub>or</sub>d<sub>ers o</sub>f <sub>magn</sub>it<sub>u</sub>d<sub>e, w</sub>hil<sub>e ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng s</sub>t<sub>a</sub>bl<sub>e</sub> <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on an</sub>d <sub>re</sub>d<sub>uc</sub>i<sub>ng v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ons o</sub>f th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> i<sub>n</sub> Equation (3). All normalization statistics are com<sub>p</sub>uted ex-<sub>c</sub>l<sub>us</sub>i<sub>ve</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>con</sub>t<sub>ex</sub>t $\mathbf { X } _ { n }$ <sub>an</sub>d <sub>are reuse</sub>d t<sub>o</sub> <sub>norma</sub>li<sub>ze</sub> th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d t<sub>arge</sub>t <sub>an</sub>d <sub>res</sub>t<sub>ore</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>on</sub> t<sub>o</sub> it<sub>s or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>sca</sub>l<sub>e.</sub>

## Channel-wise and Dynamic-Epsilon Normalization

W<sub>e</sub> i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub> t<sub>wo c</sub>h<sub>o</sub>i<sub>ces</sub> f<sub>or</sub> th<sub>e axes over w</sub>hi<sub>c</sub>h R<sub>ev</sub>IN statistics are computed. In channel-independent (CI) nor-<sub>ma</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>eac</sub>h OHLC <sub>c</sub>h<sub>anne</sub>l i<sub>s</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>us</sub>i<sub>ng</sub> it<sub>s</sub> <sub>own</sub> <sub>con</sub>t<sub>ex</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs.</sub> F<sub>or c</sub>h<sub>anne</sub>l $c \in \{ O , H , L , C \}$ <sub>,</sub> th<sub>ese</sub> <sub>s</sub>t<sub>a</sub>ti<sub>s-</sub> ti<sub>cs</sub> <sub>are</sub>

$$
\begin{array} { r l } & { \displaystyle \mu _ { n , c } ^ { \mathrm { { C I } } } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } X _ { n , t , c } , } \\ & { \displaystyle v _ { n , c } ^ { \mathrm { { C I } } } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \left( X _ { n , t , c } - \mu _ { n , c } ^ { \mathrm { { C I } } } \right) ^ { 2 } , } \end{array}\tag{5}
$$

<sub>w</sub>ith <sub>e</sub>f<sub>ec</sub>ti<sub>ve sca</sub>l<sub>e</sub>

$$
s _ { n , c } ^ { \mathrm { C I } } = \sqrt { v _ { n , c } ^ { \mathrm { C I } } + \epsilon _ { n , c } } .\tag{6}
$$

Th<sub>e con</sub>t<sub>ex</sub>t <sub>an</sub>d t<sub>arge</sub>t <sub>are</sub> th<sub>en</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>as</sub>

$$
\begin{array} { r } { \widetilde { X } _ { n , t , c } ^ { \mathrm { C I } } = \frac { X _ { n , t , c } - \mu _ { n , c } ^ { \mathrm { C I } } } { s _ { n , c } ^ { \mathrm { C I } } } , } \\ { \widetilde { Y } _ { n , \tau , c } ^ { \mathrm { C I } } = \frac { Y _ { n , \tau , c } - \mu _ { n , c } ^ { \mathrm { C I } } } { s _ { n , c } ^ { \mathrm { C I } } } . } \end{array}\tag{7}
$$

CI <sub>norma</sub>li<sub>za</sub>ti<sub>on s</sub>t<sub>an</sub>d<sub>ar</sub>di<sub>zes open,</sub> hi<sub>g</sub>h<sub>,</sub> l<sub>ow, an</sub>d <sub>c</sub>l<sub>ose</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y.</sub> B<sub>ecause</sub> th<sub>e</sub> f<sub>our c</sub>h<sub>anne</sub>l<sub>s un</sub>d<sub>ergo</sub> dif<sub>eren</sub>t <sub>a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons,</sub> th<sub>e</sub>i<sub>r</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> <sub>no</sub>t <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>guaran</sub>t<sub>ee</sub>d t<sub>o rema</sub>i<sub>n unc</sub>h<sub>ange</sub>d i<sub>n norma</sub>li<sub>ze</sub>d <sub>space.</sub>

In channel-dependent (CD) normalization, one shared mean and variance are computed jointly over the temporal <sub>an</sub>d <sub>c</sub>h<sub>anne</sub>l di<sub>mens</sub>i<sub>ons:</sub>

$$
\begin{array} { l } { \displaystyle \mu _ { n } ^ { \mathrm { { C D } } } = \frac { 1 } { 4 L } \sum _ { t = 1 } ^ { L } \sum _ { c = 1 } ^ { 4 } X _ { n , t , c } , } \\ { \displaystyle v _ { n } ^ { \mathrm { { C D } } } = \frac { 1 } { 4 L } \sum _ { t = 1 } ^ { L } \sum _ { c = 1 } ^ { 4 } \left( X _ { n , t , c } - \mu _ { n } ^ { \mathrm { { C D } } } \right) ^ { 2 } . } \end{array}\tag{8}
$$

Th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>sca</sub>l<sub>e</sub> i<sub>s</sub>

$$
s _ { n } ^ { \mathrm { C D } } = \sqrt { v _ { n } ^ { \mathrm { C D } } + \epsilon _ { n } } ,\tag{9}
$$

<sub>an</sub>d th<sub>e</sub> <sub>same</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> <sub>a</sub>ll OHLC <sub>c</sub>h<sub>anne</sub>l<sub>s:</sub>

$$
\begin{array} { r } { \widetilde { X } _ { n , t , c } ^ { \mathrm { C D } } = \frac { X _ { n , t , c } - \mu _ { n } ^ { \mathrm { C D } } } { s _ { n } ^ { \mathrm { C D } } } , } \\ { \widetilde { Y } _ { n , \tau , c } ^ { \mathrm { C D } } = \frac { Y _ { n , \tau , c } - \mu _ { n } ^ { \mathrm { C D } } } { s _ { n } ^ { \mathrm { C D } } } . } \end{array}\tag{10}
$$

U<sub>n</sub>lik<sub>e</sub> CI<sub>,</sub> CD <sub>app</sub>li<sub>es</sub> <sub>one</sub> <sub>common</sub> <sub>a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e comp</sub>l<sub>e</sub>t<sub>e</sub> OHLC <sub>samp</sub>l<sub>e an</sub>d th<sub>ere</sub>f<sub>ore re</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>organ</sub>i<sub>za</sub>ti<sub>on o</sub>f th<sub>e can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>c</sub>h<sub>anne</sub>l<sub>s.</sub> F<sub>ur</sub>th<sub>er ma</sub>th<sub>ema</sub>t<sub>-</sub> i<sub>ca</sub>l <sub>proo</sub>f<sub>s an</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s o</sub>f <sub>or</sub>d<sub>er preserva</sub>ti<sub>on un</sub>d<sub>er</sub> CD <sub>an</sub>d CI <sub>norma</sub>li<sub>za</sub>ti<sub>on are prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>

W<sub>e a</sub>dditi<sub>ona</sub>ll<sub>y exam</sub>i<sub>ne</sub> fi<sub>xe</sub>d <sub>an</sub>d d<sub>ynam</sub>i<sub>c</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> of the stabilizin<sub>g</sub> term used in Equations (6) and (9). The fi<sub>xe</sub>d f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\epsilon ^ { \mathrm { f i x } } = 1 0 ^ { - 5 } ,\tag{11}
$$

<sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> d<sub>ynam</sub>i<sub>c</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>a</sub>d<sub>ap</sub>t<sub>s</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>bili<sub>zer</sub> t<sub>o</sub> th<sub>e</sub> <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> <sub>o</sub>f th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t <sub>mean:</sub>

$$
\epsilon ^ { \mathrm { d y n } } = 1 0 ^ { - 5 } \left( \mu ^ { 2 } + 1 0 ^ { - 1 2 } \right) .\tag{12}
$$

F<sub>or</sub> CI <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> $\mu$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>c</sub>h<sub>an-</sub> <sub>ne</sub>l<sub>w</sub>i<sub>se mean;</sub> f<sub>or</sub> CD <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> it d<sub>eno</sub>t<sub>es</sub> th<sub>e s</sub>h<sub>are</sub>d <sub>samp</sub>l<sub>e-</sub>l<sub>eve</sub>l <sub>mean.</sub> D<sub>ynam</sub>i<sub>c eps</sub>il<sub>on</sub> i<sub>s</sub> d<sub>es</sub>i<sub>gne</sub>d t<sub>o preven</sub>t <sub>a</sub> fi<sub>xe</sub>d <sub>numer</sub>i<sub>ca</sub>l <sub>cons</sub>t<sub>an</sub>t f<sub>rom</sub> h<sub>av</sub>i<sub>ng a</sub> di<sub>spropor</sub>ti<sub>ona</sub>t<sub>e</sub>l<sub>y</sub> l<sub>arge e</sub>f<sub>ec</sub>t <sub>on cryp</sub>t<sub>ocurrenc</sub>i<sub>es w</sub>ith <sub>very sma</sub>ll <sub>quo</sub>t<sub>e</sub>d <sub>va</sub>l<sub>-</sub> <sub>ues.</sub> Additi<sub>ona</sub>l <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s o</sub>f th<sub>e</sub> fi<sub>xe</sub>d <sub>an</sub>d d<sub>y-</sub> <sub>nam</sub>i<sub>c eps</sub>il<sub>on</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> i<sub>s prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>

## Two-Phase RevIN and Scale-Balanced Optimization

L<sub>e</sub>t th<sub>e norma</sub>li<sub>za</sub>ti<sub>on s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs se</sub>l<sub>ec</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e prev</sub>i<sub>ous su</sub>b<sub>-</sub> <sub>sec</sub>ti<sub>on</sub> b<sub>e</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $\mu _ { n } \in \mathbb { R } ^ { 4 }$ <sub>an</sub>d $\mathbf { S } _ { n } \in \mathbb { R } ^ { 4 \times 4 }$ <sub>, w</sub>h<sub>ere</sub> $\mathbf { S } _ { n }$ i<sub>s a pos</sub>iti<sub>ve</sub> di<sub>agona</sub>l <sub>c</sub>h<sub>anne</sub>l<sub>-sca</sub>l<sub>e ma</sub>t<sub>r</sub>i<sub>x.</sub> U<sub>n</sub>d<sub>er</sub> CD <sub>nor-</sub> <sub>ma</sub>li<sub>za</sub>ti<sub>on,</sub> ${ \bf S } _ { n } = s _ { n } { \bf I } _ { 4 } ;$ <sub>un</sub>d<sub>er</sub> CI <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> it<sub>s</sub> di<sub>agona</sub>l <sub>en</sub>t<sub>r</sub>i<sub>es con</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e c</sub>h<sub>anne</sub>l<sub>w</sub>i<sub>se sca</sub>l<sub>es.</sub> Aft<sub>er norma</sub>li<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>con</sub>t<sub>ex</sub>t <sub>an</sub>d t<sub>arge</sub>t<sub>,</sub> <sub>we</sub> <sub>wr</sub>it<sub>e</sub>

<table><tr><td rowspan="2">Time Frame</td><td rowspan="2">Horizon</td><td rowspan="2">Norm Technique</td><td colspan="3">Time-MoE</td><td colspan="3">Timer-XL</td><td colspan="3">Timer</td></tr><tr><td>MSE</td><td>MAE</td><td>PHY</td><td>MSE</td><td>MAE</td><td>PHY</td><td>MSE</td><td>MAE</td><td>PHY</td></tr><tr><td rowspan="9">5m</td><td rowspan="9">5</td><td>w/o RevIN</td><td>5.7e+8</td><td>6325.3</td><td>1.312</td><td>5.2e+8</td><td>6571.64</td><td>5.536</td><td>6.15e+8</td><td>6548.55</td><td>15.154</td></tr><tr><td>CI RevIN</td><td>9890.9</td><td>19.03</td><td>0.0040</td><td>4387.64</td><td>11.66</td><td>0.012</td><td>4722.96</td><td>12.37</td><td>0.439</td></tr><tr><td>CD RevIN</td><td>9868.58</td><td>18.93</td><td>0.0076</td><td>4485.7</td><td>11.81</td><td>0.0050</td><td>5013.64</td><td>12.70</td><td>1.71</td></tr><tr><td>Two-Phase CI RevIN</td><td>6610.0</td><td>15.25</td><td>0.001</td><td>4584.75</td><td>12.34</td><td>0.0005</td><td>4510.90</td><td>12.14</td><td>0.278</td></tr><tr><td>Two-Phase CD RevIN</td><td>7480.2</td><td>16.31</td><td>0.00049</td><td>4811.0</td><td>12.72</td><td>0.0004</td><td>4632.05</td><td>12.30</td><td>1.09</td></tr><tr><td>w/o RevIN</td><td>5.7e+8</td><td>6328.50</td><td>1.827</td><td>5.5e+8</td><td>6572.12</td><td>5.532</td><td>6.15e+8</td><td>6548.99</td><td>9.523</td></tr><tr><td>CI RevIN</td><td>2.20e+4</td><td>28.63</td><td>0.0046</td><td>1.05e+4</td><td>18.54</td><td>0.0159</td><td>1.06e+4</td><td>18.69</td><td>0.915</td></tr><tr><td>CD RevIN</td><td>2.23e+4</td><td>28.62</td><td>0.0025</td><td>1.07e+4</td><td>18.71</td><td>0.0061</td><td>1.19e+4</td><td>19.73</td><td>6.285</td></tr><tr><td>Two-Phase CI RevIN Two-Phase CD RevIN</td><td>1.53e+4 1.95e+4</td><td>23.55 26.83</td><td>0.012 0.0010</td><td>1.02e+4 1.01e+4</td><td>18.39 18.32</td><td>0.0003 0.0002</td><td>9949.94 1.04e+4</td><td>18.09 18.54</td><td>0.390 4.36</td></tr><tr><td rowspan="9"></td><td rowspan="9"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o RevIN</td><td>7.73e+8</td><td>8438.67</td><td>1.271</td><td>7.66e+8</td><td>8367.45</td><td>0.501</td><td>7.71e+8</td><td>8408.11</td><td>5.717</td></tr><tr><td>CI RevIN</td><td>6.47e+4</td><td>54.60</td><td>0.0685</td><td>3.35e+4</td><td>37.58</td><td>0.063</td><td>3.22e+4</td><td>36.13</td><td>1.012</td></tr><tr><td>CD RevIN</td><td>6.33e+4</td><td>54.20</td><td>0.0307</td><td>3.38e+4</td><td>37.97</td><td>0.081</td><td>3.24e+4</td><td>36.39</td><td>3.957</td></tr><tr><td>Two-Phase CI RevIN Two-Phase CD RevIN</td><td>4.67e+4 5.41e+4</td><td>45.45 49.12</td><td>0.011</td><td>3.11e+4</td><td>36.04 36.26</td><td>0.003 0.00074</td><td>3.03e+4 1.4e+4</td><td>34.93 18.54</td><td>0.678</td></tr><tr><td></td><td></td><td></td><td>0.0076</td><td>3.11e+4</td><td></td><td></td><td></td><td></td><td>4.365</td></tr><tr><td>w/o RevIN</td><td>7.73e+8 1.56e+5</td><td>8442.13 86.77</td><td>1.553</td><td>7.68e+8</td><td>8383.54 59.11</td><td>0.606</td><td>7.71e+8</td><td>8412.84</td><td>5.542</td></tr><tr><td>CI RevIN CD RevIN</td><td>1.55e+5</td><td>87.05</td><td>0.0701 0.0443</td><td>8.29e+4 8.49e+4</td><td>60.007</td><td>0.228 0.163</td><td>7.66e+4 8.51e+4</td><td>57.00 59.75</td><td>3.238</td></tr><tr><td>Two-Phase CI RevIN</td><td>1.14e+5</td><td>72.48</td><td>0.12</td><td>7.34e+4</td><td>55.10</td><td>0.026</td><td>6.99e+4</td><td>53.38</td><td>20.67 1.123</td></tr><tr><td></td><td>Two-Phase CD RevIN</td><td>1.37e+5</td><td>80.14</td><td>0.0078</td><td>7.24e+4</td><td>54.71</td><td>0.00052</td><td>7.36e+4</td><td>55.11</td><td>15.002</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> C<sub>ompar</sub>i<sub>son o</sub>f Dif<sub>eren</sub>t R<sub>ev</sub>IN t<sub>ec</sub>h<sub>n</sub>i<sub>ques.</sub> L<sub>ower num</sub>b<sub>ers</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> l<sub>ower errors an</sub>d b<sub>e</sub>tt<sub>er per</sub>f<sub>ormances.</sub> All <sub>o</sub>f th<sub>e</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> i<sub>n</sub> thi<sub>s</sub> t<sub>a</sub>bl<sub>e are</sub> d<sub>one us</sub>i<sub>ng</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng w</sub>ith MSE <sub>on</sub>l<sub>y.</sub> B<sub>es</sub>t <sub>me</sub>t<sub>r</sub>i<sub>cs</sub> i<sub>n eac</sub>h <sub>ca</sub>t<sub>egory are co</sub>l<sub>ore</sub>d <sub>w</sub>ith R<sub>e</sub>d <sub>an</sub>d S<sub>econ</sub>d b<sub>es</sub>t <sub>me</sub>t<sub>r</sub>i<sub>cs are co</sub>l<sub>ore</sub>d <sub>w</sub>ith Bl<sub>ue.</sub>
<table><tr><td rowspan="2">Model</td><td rowspan="2">Horizon</td><td colspan="3">FAN</td><td colspan="3">SAN</td><td colspan="3">TP-RevIN</td></tr><tr><td>MAE</td><td>MAPE</td><td>PHY</td><td>MAE</td><td>MAPE</td><td>PHY</td><td>MAE</td><td>MAPE</td><td>PHY</td></tr><tr><td rowspan="3">Time-MoE</td><td>5</td><td>284.19</td><td>1.63e+4</td><td>181.12</td><td>34.99</td><td>1.45e+4</td><td>0.32</td><td>68.70</td><td>1.31</td><td>0.068</td></tr><tr><td>15</td><td>285.92</td><td>1.92e+3</td><td>187.11</td><td>55.89</td><td>1.52e+3</td><td>0.15</td><td>108.05</td><td>2.07</td><td>0.91</td></tr><tr><td>30</td><td>286.15</td><td>2.6e+4</td><td>183.43</td><td>83.23</td><td>5.21e+3</td><td>0.50</td><td>131.45</td><td>2.52</td><td>2.68</td></tr><tr><td rowspan="3">Timer-XL</td><td>5</td><td>113.85</td><td>1.67e+4</td><td>12.68</td><td>36.03</td><td>2.40e+4</td><td>0.84</td><td>50.93</td><td>0.95</td><td>0.01</td></tr><tr><td>15</td><td>123.66</td><td>8.0e+4</td><td>11.03</td><td>56.75</td><td>4.84e+3</td><td>1.46</td><td>80.36</td><td>1.49</td><td>0.016</td></tr><tr><td>30</td><td>125.46</td><td>2.6e+5</td><td>11.04</td><td>81.05</td><td>9.58e+3</td><td>1.29</td><td>106.21</td><td>1.97</td><td>0.0097</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> C<sub>ompar</sub>i<sub>son o</sub>f M<sub>o</sub>d<sub>e</sub>l<sub>s w</sub>ith FAN<sub>,</sub> SAN<sub>, an</sub>d TP<sub>-</sub>R<sub>ev</sub>IN N<sub>orma</sub>li<sub>za</sub>ti<sub>on</sub> T<sub>ec</sub>h<sub>n</sub>i<sub>ques.</sub> E<sub>xper</sub>i<sub>men</sub>t<sub>s are</sub> d<sub>one un</sub>d<sub>er</sub> 1h tim<sub>e</sub> fr<sub>a</sub>m<sub>e.</sub> F<sub>o</sub>r TP<sub>-</sub>R<sub>ev</sub>IN<sub>,</sub> h<sub>e</sub>r<sub>e we use</sub> d<sub>y</sub>n<sub>a</sub>mi<sub>c eps</sub>il<sub>o</sub>n<sub>.</sub>

$$
\begin{array} { r } { \widetilde { { \mathbf { X } } } _ { n } = \left( { \mathbf { X } } _ { n } - { \mathbf { 1 } } _ { L } { \pmb { \mu } } _ { n } ^ { \top } \right) { \mathbf { S } } _ { n } ^ { - 1 } , \qquad { \mathbf { Z } } _ { n } = \left( { \mathbf { Y } } _ { n } - { \mathbf { 1 } } _ { H } { \pmb { \mu } } _ { n } ^ { \top } \right) { \mathbf { S } } _ { n } ^ { - 1 } , } \\ { ( 1 3 ) ^ { \top } { \mathbf { \Sigma } } } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs are compu</sub>t<sub>e</sub>d <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub>l<sub>y</sub> f<sub>rom</sub> $\mathbf { X } _ { n } .$ <sub>.</sub> Th<sub>e</sub> f<sub>orecas</sub>ti<sub>ng</sub> b<sub>ac</sub>kb<sub>one</sub> <sub>pro</sub>d<sub>uces</sub> <sub>a</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub>

$$
\widehat { \mathbf { Z } } _ { n } = f _ { \pmb { \theta } } \left( \widetilde { \mathbf { X } } _ { n } \right) ,\tag{14}
$$

<sub>an</sub>d th<sub>e correspon</sub>di<sub>ng p</sub>h<sub>ys</sub>i<sub>ca</sub>l<sub>-sca</sub>l<sub>e pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s recovere</sub>d as

$$
\begin{array} { r } { \widehat { { \bf Y } } _ { n } = { \bf 1 } _ { H } \pmb { \mu } _ { n } ^ { \top } + \widehat { { \bf Z } } _ { n } { \bf S } _ { n } . } \end{array}\tag{15}
$$

C<sub>onven</sub>ti<sub>ona</sub>l R<sub>ev</sub>IN<sub>-</sub>b<sub>ase</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>common</sub>l<sub>y</sub> <sub>res</sub>t<sub>ores</sub> the <sub>p</sub>rediction throu<sub>g</sub>h Equation (15) before evaluatin<sub>g</sub> MSE.

We refer to this objective as original-space RevIN training:

$$
\mathcal { L } _ { \mathrm { R } } \left( \pmb { \theta } \right) = \frac { 1 } { 2 N } \sum _ { n = 1 } ^ { N } \left\| \widehat { \mathbf { Y } } _ { n } - \mathbf { Y } _ { n } \right\| _ { F } ^ { 2 } .\tag{16}
$$

In contrast, Two-Phase RevIN (TP-RevIN) separates the nor-<sub>ma</sub>li<sub>ze</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng p</sub>h<sub>ase</sub> f<sub>rom</sub> th<sub>e p</sub>h<sub>ys</sub>i<sub>ca</sub>l<sub>-sca</sub>l<sub>e</sub> i<sub>n</sub>f<sub>erence</sub> phase. During training, the objective is evaluated directly b<sub>e</sub>t<sub>ween</sub> th<sub>e norma</sub>li<sub>ze</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on an</sub>d <sub>norma</sub>li<sub>ze</sub>d t<sub>arge</sub>t<sub>:</sub>

$$
\mathcal { L } _ { \mathrm { T P } } \left( \pmb { \theta } \right) = \frac { 1 } { 2 N } \sum _ { n = 1 } ^ { N } \left. \widehat { \mathbf { Z } } _ { n } - \mathbf { Z } _ { n } \right. _ { F } ^ { 2 } .\tag{17}
$$

I<sub>nverse norma</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore requ</sub>i<sub>re</sub>d f<sub>or</sub> i<sub>n</sub>f<sub>erence an</sub>d reporting, but it does not participate in the training objective.

Scale-induced reweighting in original-space RevIN. Let rvec(·) stack the rows of an $H \times 4$ <sub>ma</sub>t<sub>r</sub>i<sub>x, an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
\mathbf { D } _ { n } = \mathbf { I } _ { H } \otimes \mathbf { S } _ { n } \in \mathbb { R } ^ { 4 H \times 4 H } .\tag{18}
$$

D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>vec</sub>t<sub>or</sub>i<sub>ze</sub>d <sub>norma</sub>li<sub>ze</sub>d <sub>res</sub>id<sub>ua</sub>l <sub>an</sub>d it<sub>s</sub> <sub>parame</sub>t<sub>er</sub> J<sub>aco</sub>bi<sub>an</sub> <sub>as</sub>

$$
\mathbf { e } _ { n } \left( \theta \right) = \mathrm { r v e c } \left( \widehat { \mathbf { Z } } _ { n } - \mathbf { Z } _ { n } \right) , \qquad \mathbf { J } _ { n } \left( \theta \right) = \frac { \partial \mathrm { r v e c } \left( \widehat { \mathbf { Z } } _ { n } \right) } { \partial \theta } \in \mathbb { R } ^ { 4 H \times }\tag{19}
$$

Proposition 1. Assume that $\pmb { \mu } _ { n }$ <sub>an</sub>d ${ \bf D } _ { n }$ <sub>are</sub> <sub>compu</sub>t<sub>e</sub>d f<sub>rom</sub> the observed context and are inde endent of θ. Then ori inal-<sub>space</sub> R<sub>ev</sub>IN MSE <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\mathcal { L } _ { \mathrm { R } } \left( \pmb { \theta } \right) = \frac { 1 } { 2 N } \sum _ { n = 1 } ^ { N } \mathbf { e } _ { n } ^ { \top } \mathbf { D } _ { n } ^ { 2 } \mathbf { e } _ { n } ,\tag{20}
$$

whereas TP-RevIN satisfies Equation (17). Their <sub>p</sub>arameter <sub>gra</sub>di<sub>en</sub>t<sub>s</sub> <sub>are</sub>

$$
\nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { R } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { J } _ { n } ^ { \top } \mathbf { D } _ { n } ^ { 2 } \mathbf { e } _ { n } ,\tag{21}
$$

<sub>an</sub>d

$$
\nabla _ { \pmb \theta } \mathcal { L } _ { \mathrm { T P } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { J } _ { n } ^ { \top } \mathbf { e } _ { n } .\tag{22}
$$

C<sub>onsequen</sub>tl<sub>y, or</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space</sub> R<sub>ev</sub>IN i<sub>n</sub>t<sub>ro</sub>d<sub>uces an exp</sub>li<sub>c</sub>it scale-dependent weighting into both the empirical objective <sub>an</sub>d th<sub>e</sub> <sub>s</sub>h<sub>are</sub>d <sub>parame</sub>t<sub>er</sub> <sub>up</sub>d<sub>a</sub>t<sub>e,</sub> <sub>w</sub>hil<sub>e</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>removes</sub> thi<sub>s</sub> <sub>we</sub>i<sub>g</sub>hti<sub>ng.</sub>

Proof. From Equations (13) and (15),

$$
\begin{array} { c } { \operatorname { r v e c } \left( \widehat { \mathbf { Y } } _ { n } - { \mathbf { Y } } _ { n } \right) = { \mathbf { D } } _ { n } \operatorname { r v e c } \left( \widehat { \mathbf { Z } } _ { n } - { \mathbf { Z } } _ { n } \right) } \\ { = { \mathbf { D } } _ { n } \mathbf { e } _ { n } . } \end{array}\tag{23}
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\left\| { \widehat { \mathbf { Y } } } _ { n } - \mathbf { Y } _ { n } \right\| _ { F } ^ { 2 } = \mathbf { e } _ { n } ^ { \top } \mathbf { D } _ { n } ^ { \top } \mathbf { D } _ { n } \mathbf { e } _ { n } .\tag{24}
$$

B<sub>ecause</sub> ${ \bf D } _ { n }$ i<sub>s</sub> di<sub>agona</sub>l <sub>an</sub>d <sub>pos</sub>iti<sub>ve,</sub> ${ \bf D } _ { n } ^ { \top } { \bf D } _ { n } = { \bf D } _ { n } ^ { 2 }$ <sub>, w</sub>hi<sub>c</sub>h <sub>p</sub>roves Equation (20).

Diferentiatin<sub>g</sub> the <sub>p</sub>er-sam<sub>p</sub>le term in Equation (20) and treat<sup>i</sup>n<sub>g</sub> ${ \bf D } _ { n }$ as constant with respect to θ gives

$$
\nabla _ { \pmb { \theta } } \left( \frac { 1 } { 2 } \mathbf { e } _ { n } ^ { \top } \mathbf { D } _ { n } ^ { 2 } \mathbf { e } _ { n } \right) = \mathbf { J } _ { n } ^ { \top } \mathbf { D } _ { n } ^ { 2 } \mathbf { e } _ { n } .\tag{25}
$$

Summin<sub>g</sub> over all sam<sub>p</sub>les <sub>y</sub>ields Equation (21). Similarl<sub>y</sub>, dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng</sub> th<sub>e</sub> TP<sub>-</sub>R<sub>ev</sub>IN t<sub>erm</sub> $\frac { 1 } { 2 } \mathbf { e } _ { n } ^ { \top } \mathbf { e } _ { n }$ <sub>g</sub><sup>i</sup>ves $\mathbf { J } _ { n } ^ { \top } \mathbf { e } _ { n } .$ , establishin<sub>g</sub> Equation (22). □

F<sub>or</sub> CD <sub>norma</sub>li<sub>za</sub>ti<sub>on, w</sub>h<sub>ere</sub> ${ \bf D } _ { n } = s _ { n } { \bf I } _ { 4 H }$ <sub>,</sub> P<sub>ropos</sub>iti<sub>on</sub> 1 <sub>re</sub>d<sub>uces</sub> t<sub>o</sub>

$$
\mathcal { L } _ { \mathrm { R } } = \frac { 1 } { 2 N } \sum _ { n = 1 } ^ { N } s _ { n } ^ { 2 } \left\| \mathbf { e } _ { n } \right\| _ { 2 } ^ { 2 } ,\tag{26}
$$

<sub>an</sub>d

$$
\nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { R } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } ^ { 2 } \mathbf { J } _ { n } ^ { \top } \mathbf { e } _ { n } .\tag{27}
$$

Th<sub>us</sub> t<sub>wo samp</sub>l<sub>es w</sub>ith <sub>compara</sub>bl<sub>e norma</sub>li<sub>ze</sub>d <sub>res</sub>id<sub>ua</sub>l<sub>s an</sub>d <sub>compara</sub>bl<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> t<sub>o</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>n</sub> <sub>propor</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> <sub>squares</sub> <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> <sub>con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>es.</sub> Thi<sub>s</sub> i<sub>s</sub> th<sub>e prec</sub>i<sub>se sense</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>sca</sub>l<sub>e</sub> d<sub>om</sub>i<sub>nance ar</sub>i<sub>ses</sub> i<sub>n</sub> R<sub>ev</sub>IN<sub>-</sub>b<sub>ase</sub>d MSE t<sub>ra</sub>i<sub>n</sub>i<sub>ng.</sub> L<sub>arge numer</sub>i<sub>ca</sub>l <sub>sca</sub>l<sub>e a</sub>l<sub>one</sub> d<sub>oes no</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> th<sub>e comp</sub>l<sub>e</sub>t<sub>e gra</sub>di<sub>en</sub>t<sub>, s</sub>i<sub>nce</sub> th<sub>e con-</sub> t<sub>r</sub>ib<sub>u</sub>ti<sub>on a</sub>l<sub>so</sub> d<sub>epen</sub>d<sub>s on</sub> ${ \bf J } _ { n } ^ { \top } { \bf e } _ { n } ;$ h<sub>owever,</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> $s _ { n } ^ { 2 }$ i<sub>s</sub> an unavoidable multiplicative component of the objective .<sub>whenever the loss is evaluated after inverse normalization.</sub>

Th<sub>e same resu</sub>lt <sub>can</sub> b<sub>e expresse</sub>d <sub>a</sub>t th<sub>e asse</sub>t l<sub>eve</sub>l<sub>.</sub> L<sub>e</sub>t $\pi _ { a }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e samp</sub>li<sub>ng pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>asse</sub>t $^ { a , }$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> it<sub>s</sub> <sub>norma</sub>li<sub>ze</sub>d f<sub>orecas</sub>ti<sub>ng r</sub>i<sub>s</sub>k <sub>as</sub>

$$
R _ { a } ( \pmb \theta ) = \frac { 1 } { 2 } \mathbb { E } [ \| \mathbf { e } ( \pmb \theta ) \| _ { 2 } ^ { 2 } | a ] .\tag{28}
$$

If asset a has approximately constant context scale $s _ { a }$ <sub>,</sub> th<sub>en</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space</sub> R<sub>ev</sub>IN <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub>

$$
R _ { \mathrm { R } } \left( \pmb { \theta } \right) = \sum _ { a } \pi _ { a } s _ { a } ^ { 2 } R _ { a } \left( \pmb { \theta } \right) ,\tag{29}
$$

<sub>w</sub>h<sub>ereas</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub>

$$
R _ { \mathrm { T P } } \left( \pmb { \theta } \right) = \sum _ { a } \pi _ { a } R _ { a } \left( \pmb { \theta } \right) .\tag{30}
$$

U<sub>p</sub> to a <sub>p</sub>ositive <sub>g</sub>lobal constant, Equation (29) is equivalent t<sub>o rep</sub>l<sub>ac</sub>i<sub>ng</sub> th<sub>e</sub> d<sub>ec</sub>l<sub>are</sub>d <sub>asse</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\pi _ { a }$ <sub>w</sub>ith

$$
q _ { a } = \frac { \pi _ { a } s _ { a } ^ { 2 } } { \sum _ { b } \pi _ { b } s _ { b } ^ { 2 } } .\tag{31}
$$

O<sub>r</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space</sub> R<sub>ev</sub>IN th<sub>ere</sub>f<sub>ore c</sub>h<sub>anges</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>ve op</sub>ti<sub>-</sub> <sub>m</sub>i<sub>za</sub>ti<sub>on pr</sub>i<sub>or</sub>it<sub>y</sub> t<sub>owar</sub>d <sub>asse</sub>t<sub>s w</sub>ith l<sub>arger con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>es.</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>preserves</sub> th<sub>e</sub> d<sub>ec</sub>l<sub>are</sub>d <sub>samp</sub>li<sub>ng we</sub>i<sub>g</sub>ht<sub>s an</sub>d <sub>re-</sub> <sub>moves</sub> thi<sub>s</sub> i<sub>mp</sub>li<sub>c</sub>it <sub>sca</sub>l<sub>e-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>re</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

Thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on</sub> i<sub>s par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y</sub> i<sub>mpor</sub>t<sub>an</sub>t f<sub>or cryp</sub>t<sub>ocur-</sub> renc<sub>y p</sub>ane<sup>l</sup>s, w<sup>h</sup>ere assets can s<sub>p</sub>an man<sub>y</sub> or<sup>d</sup>ers o<sup>f</sup> ma<sub>g</sub>n<sup>i</sup>- t<sub>u</sub>d<sub>e.</sub> U<sub>n</sub>d<sub>er compara</sub>bl<sub>e norma</sub>li<sub>ze</sub>d f<sub>orecas</sub>ti<sub>ng errors an</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>sens</sub>iti<sub>v</sub>iti<sub>es, a</sub> hi<sub>g</sub>h<sub>-sca</sub>l<sub>e asse</sub>t <sub>can</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub> th<sub>e ag-</sub> <sub>g</sub>re<sub>g</sub>ate <sub>p</sub>arameter u<sub>p</sub><sup>d</sup>ate even w<sup>h</sup>en <sup>l</sup>ow-sca<sup>l</sup>e assets are <sub>samp</sub>l<sub>e</sub>d <sub>equa</sub>ll<sub>y o</sub>ft<sub>en.</sub> TP<sub>-</sub>R<sub>ev</sub>IN d<sub>oes no</sub>t <sub>ma</sub>k<sub>e a</sub>ll <sub>as-</sub> <sub>se</sub>t<sub>s equa</sub>ll<sub>y</sub> difi<sub>cu</sub>lt <sub>or guaran</sub>t<sub>ee equa</sub>l <sub>gra</sub>di<sub>en</sub>t <sub>norms;</sub> dif<sub>-</sub> f<sub>erences</sub> i<sub>n</sub> f<sub>orecas</sub>t<sub>a</sub>bilit<sub>y, res</sub>id<sub>ua</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure, samp</sub>li<sub>ng</sub> f<sub>re-</sub> <sub>quency, an</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y rema</sub>i<sub>n.</sub> It <sub>spec</sub>ifi<sub>ca</sub>ll<sub>y removes</sub> th<sub>e mu</sub>lti<sub>p</sub>li<sub>ca</sub>ti<sub>ve we</sub>i<sub>g</sub>hti<sub>ng</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d <sub>so</sub>l<sub>e</sub>l<sub>y</sub> b<sub>y</sub> th<sub>e</sub> R<sub>ev</sub>IN scale and thereby yields a training objective that is neutral to <sub>pos</sub>iti<sub>ve</sub> <sub>a</sub>fi<sub>ne</sub> <sub>c</sub>h<sub>anges</sub> i<sub>n</sub> th<sub>e</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>un</sub>it<sub>s</sub> <sub>o</sub>f <sub>eac</sub>h <sub>asse</sub>t<sub>.</sub> F<sub>ur</sub>th<sub>er proo</sub>f<sub>s</sub> b<sub>ase</sub>d <sub>on</sub> J<sub>aco</sub>bi<sub>an an</sub>d H<sub>ess</sub>i<sub>an ana</sub>l<sub>yses are</sub> <sub>prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>

## Normalized-Space OHLC Constraint Loss

Under TP-RevIN, both the forecasting objective and the auxili<sub>ary</sub> OHLC <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t l<sub>oss are eva</sub>l<sub>ua</sub>t<sub>e</sub>d i<sub>n norma</sub>li<sub>ze</sub>d <sub>co-</sub> <sub>or</sub>di<sub>na</sub>t<sub>es.</sub> L<sub>e</sub>t $\widehat { \mathbf { Z } } _ { n }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> <sub>w</sub>ith <sup>com</sup>p<sup>onents</sup> $\widehat { O } _ { n , \tau } , \widehat { H } _ { n , \tau } , \widehat { L } _ { n , \tau }$ <sub>,</sub> <sub>an</sub>d ${ \widehat { C } } _ { n , \tau }$ at forecast step τ . D<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> $[ x ] _ { + } = \operatorname* { m a x } ( 0 , x )$ <sub>,</sub> th<sub>e norma</sub>li<sub>ze</sub>d<sub>-space cons</sub>t<sub>ra</sub>i<sub>n</sub>t

<table><tr><td rowspan="2">Model</td><td rowspan="2">Coin</td><td colspan="2">RevIN</td><td colspan="2">TP-RevIN</td></tr><tr><td>Fixed Epsilon</td><td>Dynamic Epsilon</td><td>Fixed Epsilon</td><td>Dynamic Epsilon</td></tr><tr><td rowspan="5">Time-MoE</td><td>BTC</td><td>1.15 / 2.38</td><td>1.16 / 2.38</td><td>1.12 / 2.23</td><td>1.16 / 2.27</td></tr><tr><td>ETH</td><td>1.94 / 4.20</td><td>1.96 / 4.16</td><td>1.85 / 3.58</td><td>1.94 / 3.59</td></tr><tr><td>SHIB</td><td>17362.6 / 12894.9</td><td>1.89 / 4.12</td><td>67.24 / 73.13</td><td>1.87 / 3.51</td></tr><tr><td>DOGE</td><td>2.19 / 4.71</td><td>2.21 / 4.68</td><td>2.07 / 4.26</td><td>2.18 / 4.17</td></tr><tr><td>ADA</td><td>2.47 / 5.61</td><td>2.48 / 5.61</td><td>2.37 / 4.93</td><td>2.47 / 4.95</td></tr><tr><td rowspan="5">Timer-XL</td><td>BTC</td><td>0.93 / 1.85</td><td>0.94 / 1.83</td><td>0.93 / 1.72</td><td>0.91 / 1.73</td></tr><tr><td>ETH</td><td>1.61 / 3.19</td><td>1.65 / 3.26</td><td>1.67 / 2.81</td><td>1.61 / 2.88</td></tr><tr><td>SHIB</td><td>181.5 / 3003.3</td><td>1.48 / 3.20</td><td>71.2 / 146.3</td><td>1.40 / 2.74</td></tr><tr><td>DOGE</td><td>1.72 / 3.74</td><td>1.71 / 3.69</td><td>1.73 / 3.31</td><td>1.67 / 3.25</td></tr><tr><td>ADA</td><td>1.85 / 4.06</td><td>1.86 / 4.13</td><td>1.82 / 3.69</td><td>1.81 / 3.81</td></tr><tr><td rowspan="5">Timer</td><td>BTC</td><td>0.88 / 1.77</td><td>0.88 / 1.80</td><td>0.85 / 1.68</td><td>0.85 / 1.65</td></tr><tr><td>ETH</td><td>1.44 / 3.02</td><td>1.45 / 3.00</td><td>1.40 / 2.70</td><td>1.39 / 2.65</td></tr><tr><td>SHIB</td><td>471.4 / 5346.9</td><td>1.45 / 2.97</td><td>179.0 / 355.4</td><td>1.38 / 2.72</td></tr><tr><td>DOGE</td><td>1.63 / 3.48</td><td>1.64 / 3.42</td><td>1.58 / 3.20</td><td>1.56 / 3.12</td></tr><tr><td>ADA</td><td>1.89 / 4.13</td><td>1.90 / 3.94</td><td>1.87 / 3.74</td><td>1.86 / 3.80</td></tr></table>

Table 3: Com<sub>p</sub>arison of Fixed vs. D<sub>y</sub>namic E<sub>p</sub>silon Normalization (Values re<sub>p</sub>ort Horizon 5 / 30 for MAPE).

l<sub>oss</sub> i<sub>s</sub>

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { p h y } } = \frac { 1 } { N H } \sum _ { n = 1 } ^ { N } \sum _ { \tau = 1 } ^ { H } \Bigl ( [ \widehat { O } _ { n , \tau } - \widehat { H } _ { n , \tau } ] _ { + } + [ \widehat { C } _ { n , \tau } - \widehat { H } _ { n , \tau } ] _ { + } } } \\ & { } & { + [ \widehat { L } _ { n , \tau } - \widehat { O } _ { n , \tau } ] _ { + } + [ \widehat { L } _ { n , \tau } - \widehat { C } _ { n , \tau } ] _ { + } } \\ & { } & { + [ \widehat { L } _ { n , \tau } - \widehat { H } _ { n , \tau } ] _ { + } \Bigr ) . } \end{array}\tag{32}
$$

The complete TP-RevIN training objective is therefore

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { M S E } } \mathcal { L } _ { \mathrm { T P } } + \lambda _ { \mathrm { p h y } } \mathcal { L } _ { \mathrm { p h y } } ,\tag{33}
$$

<sub>w</sub>h<sub>ere</sub> b<sub>o</sub>th t<sub>erms are compu</sub>t<sub>e</sub>d b<sub>e</sub>f<sub>ore</sub> i<sub>nverse norma</sub>li<sub>za</sub>ti<sub>on.</sub> Thi<sub>s</sub> d<sub>es</sub>i<sub>gn preven</sub>t<sub>s</sub> th<sub>e aux</sub>ili<sub>ary</sub> l<sub>oss</sub> f<sub>rom re</sub>i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng</sub> th<sub>e</sub> <sub>sca</sub>l<sub>e-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>we</sub>i<sub>g</sub>hti<sub>ng</sub> <sub>remove</sub>d b<sub>y</sub> TP<sub>-</sub>R<sub>ev</sub>IN<sub>.</sub> Ad<sub>-</sub> diti<sub>ona</sub>l <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>resu</sub>lt<sub>s concern</sub>i<sub>ng norma</sub>li<sub>ze</sub>d<sub>-space</sub> OHLC <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s are prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>

## CryptoL Framework

C<sub>ryp</sub>t<sub>o</sub>L <sub>com</sub>bi<sub>nes</sub> d<sub>ynam</sub>i<sub>c-eps</sub>il<sub>on norma</sub>li<sub>za</sub>ti<sub>on,</sub> TP<sub>-</sub> R<sub>ev</sub>IN<sub>,</sub> <sub>a</sub>nd th<sub>e</sub> n<sub>o</sub>rm<sub>a</sub>li<sub>ze</sub>d<sub>-space</sub> OHLC <sub>co</sub>n<sub>s</sub>tr<sub>a</sub>int l<sub>oss</sub> <sub>w</sub>ithi<sub>n</sub> <sub>a</sub> <sub>un</sub>ifi<sub>e</sub>d <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> <sub>cryp</sub>t<sub>ocurrency</sub> f<sub>orecas</sub>ti<sub>ng</sub> f<sub>ramewor</sub>k<sub>.</sub> Th<sub>e</sub> i<sub>npu</sub>t <sub>con</sub>t<sub>ex</sub>t <sub>an</sub>d t<sub>arge</sub>t <sub>are</sub> fi<sub>rs</sub>t t<sub>rans</sub>f<sub>orme</sub>d <sub>us</sub>i<sub>ng e</sub>ith<sub>er</sub> CI <sub>or</sub> CD <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs</sub> t<sub>oge</sub>th<sub>er w</sub>ith th<sub>e</sub> d<sub>ynam</sub>i<sub>c</sub> e<sub>p</sub>silon in Equation (12). The forecastin<sub>g</sub> backbone <sub>p</sub>redicts th<sub>e</sub> f<sub>u</sub>t<sub>ure sequence</sub> i<sub>n norma</sub>li<sub>ze</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es, w</sub>h<sub>ere</sub> b<sub>o</sub>th the MSE objective and the auxiliary constraint loss are evaluated accordin<sub>g</sub> to Equation (33). Inverse normalization is <sub>app</sub>li<sub>e</sub>d <sub>on</sub>l<sub>y a</sub>ft<sub>er op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> t<sub>o recover pre</sub>di<sub>c</sub>ti<sub>ons</sub> i<sub>n</sub> th<sub>e</sub>i<sub>r</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>pr</sub>i<sub>ce</sub> <sub>un</sub>it<sub>s.</sub>

Th<sub>e se</sub>l<sub>ec</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> CI <sub>an</sub>d CD <sub>rema</sub>i<sub>ns con</sub>fi<sub>gura</sub>bl<sub>e</sub> b<sub>ecause</sub> th<sub>e</sub> t<sub>wo</sub> <sub>sc</sub>h<sub>emes</sub> <sub>prov</sub>id<sub>e</sub> dif<sub>eren</sub>t <sub>prac</sub>ti<sub>ca</sub>l t<sub>ra</sub>d<sub>e-</sub> <sub>o</sub>f<sub>s.</sub> CD <sub>p</sub>r<sub>ese</sub>r<sub>ves</sub> OHLC <sub>o</sub>rd<sub>e</sub>rin<sub>g u</sub>nd<sub>e</sub>r n<sub>o</sub>rm<sub>a</sub>li<sub>za</sub>ti<sub>o</sub>n<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> CI <sub>can prov</sub>id<sub>e s</sub>t<sub>ronger c</sub>h<sub>anne</sub>l<sub>w</sub>i<sub>se</sub> f<sub>orecas</sub>ti<sub>ng</sub> <sub>accu</sub>r<sub>acy.</sub> A<sub>cco</sub>rdin<sub>g</sub>l<sub>y,</sub> b<sub>o</sub>th CI <sub>a</sub>nd CD <sub>va</sub>ri<sub>a</sub>nt<sub>s o</sub>f Cr<sub>yp-</sub> t<sub>o</sub>L <sub>ac</sub>hi<sub>eve</sub> l<sub>ow p</sub>h<sub>ys</sub>i<sub>ca</sub>l<sub>-cons</sub>i<sub>s</sub>t<sub>ency errors w</sub>h<sub>en com</sub>bi<sub>ne</sub>d <sub>w</sub>ith th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t l<sub>oss, a</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e norma</sub>li<sub>za</sub>ti<sub>on ax</sub>i<sub>s</sub> t<sub>o</sub> b<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> b<sub>ac</sub>kb<sub>one, an</sub>d t<sub>arge</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on cr</sub>it<sub>er</sub>i<sub>on.</sub>

## Experiments

## Dataset

We constr<sub>u</sub>ct a lar<sub>g</sub>e-scale cr<sub>yp</sub>toc<sub>u</sub>rrenc<sub>y</sub> forecastin<sub>g</sub> dataset containing approximately 15.5 million historical OHLC observations from 16 assets. The data were collected through th<sub>e</sub> Bi<sub>nance</sub> API <sub>a</sub>t <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> t<sub>empora</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>ons,</sub> <sub>ena</sub>bli<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on across</sub> dif<sub>eren</sub>t f<sub>orecas</sub>ti<sub>ng</sub> ti<sub>me</sub> f<sub>rames an</sub>d h<sub>or</sub>i<sub>-</sub> <sub>zons.</sub> Th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>asse</sub>t<sub>s</sub> <sub>ex</sub>hibit <sub>ex</sub>t<sub>reme</sub> <sub>sca</sub>l<sub>e</sub> h<sub>e</sub>t<sub>erogene-</sub> it<sub>y,</sub> <sub>w</sub>ith <sub>quo</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ues</sub> <sub>rang</sub>i<sub>ng</sub> f<sub>rom</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub> $\mathrm { \overline { { 1 0 } } } ^ { - 7 }$ f<sub>or</sub> l<sub>ow-va</sub>l<sub>ue</sub>d <sub>asse</sub>t<sub>s suc</sub>h <sub>as</sub> PEPE t<sub>o a</sub>b<sub>ove</sub> $1 0 ^ { 5 }$ f<sub>or</sub> hi<sub>g</sub>h<sub>-</sub> <sub>va</sub>l<sub>ue</sub>d <sub>asse</sub>t<sub>s</sub> <sub>suc</sub>h <sub>as</sub> BTC<sub>.</sub> Thi<sub>s</sub> b<sub>roa</sub>d d<sub>ynam</sub>i<sub>c</sub> <sub>range</sub> <sub>pro-</sub> <sub>v</sub>id<sub>es</sub> <sub>a</sub> <sub>su</sub>it<sub>a</sub>bl<sub>e</sub> <sub>se</sub>tti<sub>ng</sub> f<sub>or</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>e</sub>h<sub>av-</sub> i<sub>or,</sub> <sub>cross-asse</sub>t l<sub>earn</sub>i<sub>ng,</sub> <sub>an</sub>d th<sub>e</sub> <sub>sca</sub>l<sub>e-</sub>b<sub>a</sub>l<sub>anc</sub>i<sub>ng</sub> <sub>proper</sub>ti<sub>es</sub> <sub>o</sub>f TP<sub>-</sub>R<sub>ev</sub>IN<sub>.</sub> C<sub>omp</sub>l<sub>e</sub>t<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t d<sub>e</sub>t<sub>a</sub>il<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> t<sub>empora</sub>l <sub>coverage, row coun</sub>t<sub>s, an</sub>d <sub>asse</sub>t<sub>-spec</sub>ifi<sub>c vo</sub>l<sub>a</sub>tilit<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u-</sub> ti<sub>ons, are prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>

## Training Configuration

W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> C<sub>ryp</sub>t<sub>o</sub>L <sub>us</sub>i<sub>ng</sub> th<sub>ree</sub> d<sub>eco</sub>d<sub>er-on</sub>l<sub>y</sub> ti<sub>me-ser</sub>i<sub>es</sub> forecastin<sub>g</sub> backbones: Timer (Liu et al. 2024) , Timer-XL (Liu et al. 2025), and Time-MoE (Shi et al. 2025). All mod-<sub>e</sub>l<sub>s are op</sub>ti<sub>m</sub>i<sub>ze</sub>d <sub>us</sub>i<sub>ng</sub> Ad<sub>am</sub>W <sub>w</sub>ith <sub>a</sub> li<sub>near</sub>l<sub>y sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub>d l<sub>earn</sub>i<sub>ng ra</sub>t<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>ze</sub>d <sub>a</sub>t $1 0 ^ { - 5 }$ <sub>.</sub> E<sub>ac</sub>h <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d f<sub>or</sub> one epoch using 90% of the available data, while the remaining 10% is reserved for evaluation. All reported forecasting <sub>me</sub>t<sub>r</sub>i<sub>cs are compu</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>er res</sub>t<sub>or</sub>i<sub>ng pre</sub>di<sub>c</sub>ti<sub>ons</sub> t<sub>o</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>-</sub> <sub>na</sub>l <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>space, ensur</sub>i<sub>ng a cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>an</sub>d f<sub>a</sub>i<sub>r compar</sub>i<sub>son</sub> <sub>among</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub>

<table><tr><td rowspan="2">Model</td><td rowspan="2">Time Frame</td><td rowspan="2">Horizon</td><td colspan="2">RevIN (CI)</td><td colspan="2">TP-RevIN (CI)</td><td colspan="2">TP-RevIN (CD)</td><td colspan="2">Unconstrained Space</td></tr><tr><td>MAE</td><td>PHY</td><td>MAE</td><td>PHY</td><td>MAE</td><td>PHY</td><td>MAE</td><td>PHY</td></tr><tr><td rowspan="7">Time-MoE</td><td rowspan="2">30m</td><td>5</td><td>47.08</td><td>0.047</td><td>45.90</td><td>1.0e-6</td><td>49.82</td><td>1.6e-3</td><td>459.66</td><td>0.0</td></tr><tr><td>15</td><td>78.63</td><td>2.20</td><td>72.64</td><td>5.0e-7</td><td>80.99</td><td>1.5e-3</td><td>452.61</td><td>0.0</td></tr><tr><td rowspan="2"></td><td>30</td><td>100.07</td><td>6.08</td><td>91.81</td><td>3.6e-5</td><td>101.23</td><td>7.9e-4</td><td>517.42</td><td>0.0</td></tr><tr><td>5</td><td>68.28</td><td>0.161</td><td>66.91</td><td>1.0e-6</td><td>69.25</td><td>2.7e-3</td><td>723.18</td><td>0.0</td></tr><tr><td rowspan="2">1h</td><td>15</td><td>112.06</td><td>1.14</td><td>109.06</td><td>8.0e-6</td><td>117.69</td><td>7.8e-3</td><td>764.56</td><td>0.0</td></tr><tr><td>30</td><td>117.92</td><td>6.14</td><td>130.28</td><td>4.3e-5</td><td>141.20</td><td>8.9e-3</td><td>768.84</td><td>0.0</td></tr><tr><td rowspan="6">Timer-XL</td><td rowspan="2">30m</td><td>5</td><td>34.26</td><td>5.4e-2</td><td>34.77</td><td>2.0e-6</td><td>37.09</td><td>2.9e-5</td><td>3307.53</td><td>0.0</td></tr><tr><td>15</td><td>55.07</td><td>2.4e-1</td><td>54.04</td><td>1.9e-3</td><td>55.71</td><td>2.0e-6</td><td>1285.78</td><td>0.0</td></tr><tr><td rowspan="4"></td><td>30</td><td>73.50</td><td>1.6e-1</td><td>71.19</td><td>6.9e-4</td><td>106.98</td><td>2.0e-4</td><td>1520.53</td><td>0.0</td></tr><tr><td>5</td><td>52.15</td><td>8.7e-2</td><td>52.10</td><td>1.0e-7</td><td>54.91</td><td>7.4e-5</td><td>1492.80</td><td>0.0</td></tr><tr><td>15</td><td>85.10</td><td>4.4e-1</td><td>82.53</td><td>9.6e-5</td><td>84.85</td><td>4.5e-4</td><td>801.78</td><td>0.0</td></tr><tr><td>30</td><td>110.69</td><td>3.1e-1</td><td>105.5</td><td>5.0e-7</td><td>106.98</td><td>2.0e-4</td><td>1247.36</td><td>0.0</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 4<sub>:</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s o</sub>f t<sub>ra</sub>i<sub>n</sub>i<sub>ng mo</sub>d<sub>e</sub>l<sub>s us</sub>i<sub>ng aux</sub>ili<sub>ary p</sub>h<sub>ys</sub>i<sub>cs</sub> l<sub>oss on</sub> dif<sub>eren</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s</sub>

E<sub>ac</sub>h <sub>exper</sub>i<sub>men</sub>t i<sub>s con</sub>d<sub>uc</sub>t<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> f<sub>or a spec</sub>ifi<sub>c</sub> ti<sub>me</sub> f<sub>rame an</sub>d f<sub>orecas</sub>t h<sub>or</sub>i<sub>zon.</sub> Withi<sub>n eac</sub>h <sub>con</sub>fi<sub>gura</sub>ti<sub>on,</sub> h<sub>ow-</sub> ever, the model is trained jointly on all 16 assets rather than i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> f<sub>or eac</sub>h <sub>cryp</sub>t<sub>ocurrency.</sub> Thi<sub>s pro</sub>t<sub>oco</sub>l <sub>eva</sub>l<sub>u-</sub> <sub>a</sub>t<sub>es w</sub>h<sub>e</sub>th<sub>er a s</sub>h<sub>are</sub>d f<sub>orecas</sub>ti<sub>ng</sub> b<sub>ac</sub>kb<sub>one can</sub> l<sub>earn</sub> t<sub>rans-</sub> f<sub>era</sub>bl<sub>e</sub> t<sub>empora</sub>l <sub>an</sub>d <sub>cross-asse</sub>t <sub>pa</sub>tt<sub>erns</sub> d<sub>esp</sub>it<sub>e su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub> dif<sub>erences</sub> i<sub>n asse</sub>t <sub>sca</sub>l<sub>e, vo</sub>l<sub>a</sub>tilit<sub>y, an</sub>d <sub>mar</sub>k<sub>e</sub>t b<sub>e</sub>h<sub>av</sub>i<sub>or.</sub>

## Hardware and Software Environment

All <sub>exper</sub>i<sub>men</sub>t<sub>s are con</sub>d<sub>uc</sub>t<sub>e</sub>d <sub>on e</sub>i<sub>g</sub>ht TPU <sub>v</sub>5<sub>e acce</sub>l<sub>er-</sub> ators, each providing 16 GB of accelerator memory, for a total distributed memory capacity of 128 GB. The implementation is develo<sub>p</sub>ed usin<sub>g</sub> JAX (Bradbur<sub>y</sub> et al. 2018) and Flax (Heek et al. 2024).<sup>1</sup> Both model <sub>p</sub>arameters and i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e ac</sub>ti<sub>va</sub>ti<sub>ons are represen</sub>t<sub>e</sub>d i<sub>n</sub> FP32 th<sub>roug</sub>h<sub>ou</sub>t t<sub>ra</sub>i<sub>n</sub>i<sub>ng.</sub>

T<sub>o</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> th<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng wor</sub>kl<sub>oa</sub>d<sub>, we emp</sub>l<sub>oy</sub> f<sub>u</sub>ll<sub>y</sub> <sub>s</sub>h<sub>ar</sub>d<sub>e</sub>d d<sub>a</sub>t<sub>a-para</sub>ll<sub>e</sub>l t<sub>ra</sub>i<sub>n</sub>i<sub>ng across a</sub>ll <sub>e</sub>i<sub>g</sub>ht TPU<sub>s.</sub> Th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>parame</sub>t<sub>ers, op</sub>ti<sub>m</sub>i<sub>zer s</sub>t<sub>a</sub>t<sub>es, an</sub>d i<sub>npu</sub>t d<sub>a</sub>t<sub>a are</sub> <sub>s</sub>h<sub>ar</sub>d<sub>e</sub>d <sub>over</sub> th<sub>e comp</sub>l<sub>e</sub>t<sub>e acce</sub>l<sub>era</sub>t<sub>or mes</sub>h<sub>, re</sub>d<sub>uc</sub>i<sub>ng</sub> th<sub>e</sub> <sub>memory a</sub>ll<sub>oca</sub>t<sub>e</sub>d t<sub>o eac</sub>h d<sub>ev</sub>i<sub>ce an</sub>d <sub>ena</sub>bli<sub>ng e</sub>fi<sub>c</sub>i<sub>en</sub>t t<sub>ra</sub>i<sub>n-</sub> i<sub>ng o</sub>f th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d f<sub>orecas</sub>ti<sub>ng</sub> b<sub>ac</sub>kb<sub>ones.</sub>

## Results and Analysis

W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> b<sub>y eva</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>ve per</sub>f<sub>ormance o</sub>f dif<sub>er-</sub> <sub>en</sub>t <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es,</sub> <sub>as</sub> <sub>summar</sub>i<sub>ze</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>an</sub>d Table 2. Without an<sub>y</sub> RevIN (Kim et al. 2021), the models <sub>s</sub>t<sub>rugg</sub>l<sub>e</sub> t<sub>o</sub> <sub>converge</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>resu</sub>lti<sub>ng</sub> i<sub>n</sub> <sub>ex</sub>t<sub>reme</sub>l<sub>y</sub> hi<sub>g</sub>h

MSE <sub>va</sub>l<sub>ues</sub> d<sub>ue</sub> t<sub>o</sub> th<sub>e</sub> i<sub>mmense sca</sub>l<sub>e</sub> di<sub>screpanc</sub>i<sub>es among</sub> dif<sub>eren</sub>t <sub>cryp</sub>t<sub>ocurrency asse</sub>t<sub>s.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng s</sub>t<sub>an</sub>d<sub>ar</sub>d CI <sub>an</sub>d CD R<sub>ev</sub>IN <sub>s</sub>i<sub>g</sub>nifi<sub>ca</sub>ntl<sub>y</sub> miti<sub>ga</sub>t<sub>es</sub> thi<sub>s p</sub>r<sub>o</sub>bl<sub>e</sub>m<sub>.</sub> H<sub>oweve</sub>r<sub>,</sub> TP<sub>-</sub> RevIN (Fen<sub>g</sub>, Huan<sub>g</sub>, and Krom<sub>p</sub>ass 2024; Berthelier et al. 2026) formulations consistentl<sub>y y</sub>ield further reductions in b<sub>o</sub>th MSE <sub>an</sub>d MAE <sub>across var</sub>i<sub>ous</sub> f<sub>orecas</sub>ti<sub>ng</sub> h<sub>or</sub>i<sub>zons an</sub>d <sub>mo</sub>d<sub>e</sub>l b<sub>ac</sub>kb<sub>ones.</sub> Wh<sub>en compar</sub>i<sub>ng</sub> TP<sub>-</sub>R<sub>ev</sub>IN t<sub>o a</sub>lt<sub>erna</sub>ti<sub>ve</sub> tem<sub>p</sub>oral normalization schemes such as FAN (Ye et al. 2024) and SAN (Liu et al. 2023) in Table 2, we observe that both FAN <sub>an</sub>d SAN <sub>y</sub>i<sub>e</sub>ld <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> hi<sub>g</sub>h<sub>er</sub> MAE <sub>an</sub>d M<sub>ean</sub> Ab<sub>-</sub> solute Percenta<sub>g</sub>e Error (MAPE) values. This indicates that <sub>w</sub>hil<sub>e</sub> th<sub>ose me</sub>th<sub>o</sub>d<sub>s are</sub> d<sub>es</sub>i<sub>gne</sub>d t<sub>o</sub> h<sub>an</sub>dl<sub>e non-s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y,</sub> th<sub>ey s</sub>t<sub>rugg</sub>l<sub>e</sub> t<sub>o</sub> b<sub>a</sub>l<sub>ance</sub> th<sub>e ex</sub>t<sub>reme cross-asse</sub>t <sub>sca</sub>l<sub>e</sub> dif<sub>-</sub> ferences present in joint cryptocurrency panels. In contrast, TP<sub>-</sub>R<sub>ev</sub>IN <sub>p</sub>r<sub>ese</sub>r<sub>ves</sub> r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n <sub>s</sub>t<sub>a</sub>bilit<sub>y ac</sub>r<sub>oss</sub> di<sub>ve</sub>r<sub>se</sub> scales and maintains lower <sub>p</sub>h<sub>y</sub>sical violation (PHY) rates.

T<sub>o eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> i<sub>mpac</sub>t <sub>o</sub>f <sub>numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>bili<sub>za</sub>ti<sub>on across</sub> <sub>asse</sub>t<sub>s o</sub>f <sub>vary</sub>i<sub>ng pr</sub>i<sub>ce sca</sub>l<sub>es, we ana</sub>l<sub>yze</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>versus</sub> d<sub>ynam</sub>i<sub>c eps</sub>il<sub>on</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> F<sub>or</sub> hi<sub>g</sub>h<sub>-va</sub>l<sub>ue as-</sub> sets such as Bitcoin (BTC) and Ethereum (ETH), the choice between a fixed and d<sub>y</sub>namic stabilizer (ϵ) has ne<sub>g</sub>li<sub>g</sub>ible i<sub>mpac</sub>t <sub>on</sub> th<sub>e</sub> f<sub>orecas</sub>ti<sub>ng per</sub>f<sub>ormance.</sub> H<sub>owever,</sub> f<sub>or</sub> l<sub>ow-</sub> value assets like Shiba Inu (SHIB), which has an avera<sub>g</sub>e <sub>pr</sub>i<sub>ce sca</sub>l<sub>e on</sub> th<sub>e or</sub>d<sub>er o</sub>f $1 0 ^ { - 5 }$ <sub>,</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>eps</sub>il<sub>on s</sub>t<sub>a</sub>bili<sub>zer</sub> $\dot { ( \epsilon ^ { \mathrm { { f i x } } } } = 1 0 ^ { - 5 } )$ i<sub>n</sub>t<sub>ro</sub>d<sub>uces severe numer</sub>i<sub>ca</sub>l di<sub>s</sub>t<sub>or</sub>ti<sub>ons</sub> d<sub>ur-</sub> i<sub>ng norma</sub>li<sub>za</sub>ti<sub>on.</sub> Thi<sub>s</sub> di<sub>s</sub>t<sub>or</sub>ti<sub>on resu</sub>lt<sub>s</sub> i<sub>n</sub> hi<sub>g</sub>hl<sub>y e</sub>l<sub>eva</sub>t<sub>e</sub>d MAPE values, such as 17362.6 for Time-MoE under stand<sub>ar</sub>d R<sub>ev</sub>IN<sub>.</sub> Utili<sub>z</sub>i<sub>ng</sub> th<sub>e sca</sub>l<sub>e-a</sub>d<sub>ap</sub>ti<sub>ve</sub> d<sub>ynam</sub>i<sub>c eps</sub>il<sub>on</sub> $( \epsilon ^ { \mathrm { d y n } } )$ <sub>reso</sub>l<sub>ves</sub> thi<sub>s</sub> <sub>numer</sub>i<sub>ca</sub>l i<sub>ns</sub>t<sub>a</sub>bilit<sub>y,</sub> <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> th<sub>e</sub> SHIB MAPE to 1.87 / 3.51 under TP-RevIN and stabilizing the <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on process w</sub>ith<sub>ou</sub>t <sub>comprom</sub>i<sub>s</sub>i<sub>ng</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>accuracy o</sub>f th<sub>e</sub> hi<sub>g</sub>h<sub>-va</sub>l<sub>ue asse</sub>t<sub>s.</sub>

Fi<sub>na</sub>ll<sub>y, we ana</sub>l<sub>yze</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f th<sub>e aux</sub>ili<sub>ary</sub> <sub>p</sub>h<sub>ys</sub>i<sub>cs-</sub>i<sub>n</sub>f<sub>orme</sub>d <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t l<sub>oss, as presen</sub>t<sub>e</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 4<sub>.</sub> Th<sub>e</sub> "Unconstrained S<sub>p</sub>ace" method (Wan<sub>g</sub>, Huan<sub>g</sub>, and Wan<sub>g</sub> 2021) mathematicall<sub>y</sub> <sub>g</sub>uarantees zero <sub>p</sub>h<sub>y</sub>sical violations (PHY = 0.0) through its structural design. However, this <sub>r</sub>i<sub>g</sub>id <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>comes a</sub>t <sub>a severe cos</sub>t t<sub>o overa</sub>ll <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>accuracy,</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>o</sub> MAE <sub>va</sub>l<sub>ues</sub> th<sub>a</sub>t <sub>are</sub> <sub>or</sub>d<sub>ers</sub> <sub>o</sub>f <sub>mag-</sub> nitude hi<sub>g</sub>her than those of alternative a<sub>pp</sub>roaches (for example, an MAE of 3307.53 for Timer-XL on a 30m timeframe at horizon 5, compared to 34.77 for TP-RevIN (CI)). C<sub>onverse</sub>l<sub>y,</sub> th<sub>e propose</sub>d TP<sub>-</sub>R<sub>ev</sub>IN <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>aux</sub>ili<sub>ary p</sub>h<sub>ys</sub>i<sub>cs</sub> l<sub>oss</sub> fi<sub>n</sub>d <sub>a more</sub> b<sub>a</sub>l<sub>ance</sub>d t<sub>ra</sub>d<sub>e-o</sub>f<sub>.</sub> Th<sub>ey re-</sub> d<sub>uce p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ons</sub> t<sub>o near-zero</sub> l<sub>eve</sub>l<sub>s w</sub>hil<sub>e</sub> k<sub>eep</sub>i<sub>ng</sub> th<sub>e</sub> MAE hi<sub>g</sub>hl<sub>y compe</sub>titi<sub>ve.</sub>

## Conclusion

I<sub>n</sub> thi<sub>s wor</sub>k<sub>, we presen</sub>t<sub>e</sub>d C<sub>ryp</sub>t<sub>o</sub>L<sub>, a</sub> f<sub>ramewor</sub>k f<sub>or mu</sub>l<sub>-</sub> ti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> <sub>cryp</sub>t<sub>ocurrency</sub> f<sub>orecas</sub>ti<sub>ng</sub> th<sub>a</sub>t <sub>a</sub>dd<sub>resses</sub> <sub>cross-</sub> <sub>asse</sub>t <sub>sca</sub>l<sub>e</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y,</sub> <sub>numer</sub>i<sub>ca</sub>l i<sub>ns</sub>t<sub>a</sub>bilit<sub>y,</sub> <sub>an</sub>d <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ons.</sub> B<sub>y</sub> <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> T<sub>wo-</sub>Ph<sub>ase</sub> R<sub>ev</sub>IN<sub>,</sub> d<sub>ynam</sub>i<sub>c eps</sub>il<sub>on s</sub>t<sub>a</sub>bili<sub>za</sub>ti<sub>on, an</sub>d <sub>a norma</sub>li<sub>ze</sub>d<sub>-</sub> <sub>space</sub> <sub>aux</sub>ili<sub>ary</sub> <sub>p</sub>h<sub>ys</sub>i<sub>cs</sub> l<sub>oss,</sub> th<sub>e</sub> f<sub>ramewor</sub>k b<sub>a</sub>l<sub>ances</sub> <sub>op</sub>ti<sub>-</sub> <sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>across</sub> hi<sub>g</sub>hl<sub>y</sub> di<sub>verse</sub> <sub>pr</sub>i<sub>ce</sub> <sub>sca</sub>l<sub>es</sub> <sub>w</sub>hil<sub>e</sub> <sub>encourag-</sub> i<sub>ng s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>va</sub>lidit<sub>y.</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t thi<sub>s</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub>d <sub>approac</sub>h <sub>o</sub>f<sub>ers a prac</sub>ti<sub>ca</sub>l <sub>comprom</sub>i<sub>se</sub> b<sub>e</sub>t<sub>ween</sub> f<sub>orecas</sub>ti<sub>ng</sub> <sub>accuracy</sub> <sub>an</sub>d <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>ency,</sub> <sub>prov</sub>idi<sub>ng</sub> <sub>a</sub> <sub>s</sub>t<sub>a</sub>bl<sub>e</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng s</sub>h<sub>are</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s on</sub> h<sub>e</sub>t<sub>eroge-</sub> <sub>neous</sub> fi<sub>nanc</sub>i<sub>a</sub>l ti<sub>me ser</sub>i<sub>es.</sub>

## References

Ah<sub>me</sub>d<sub>,</sub> I<sub>.;</sub> K<sub>rompa</sub>ß<sub>,</sub> D<sub>.;</sub> F<sub>eng,</sub> C<sub>.; an</sub>d T<sub>resp,</sub> V<sub>.</sub> 2026<sub>.</sub> A C<sub>ompara</sub>ti<sub>ve</sub> St<sub>u</sub>d<sub>y on</sub> h<sub>ow</sub> D<sub>a</sub>t<sub>a</sub> N<sub>orma</sub>li<sub>za</sub>ti<sub>on</sub> Af<sub>ec</sub>t<sub>s</sub> Z<sub>ero-</sub>Sh<sub>o</sub>t G<sub>enera</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>n</sub> Ti<sub>me</sub> S<sub>er</sub>i<sub>es</sub> F<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> M<sub>o</sub>d<sub>-</sub> els. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 3436– 3440<sub>.</sub> IEEE<sub>.</sub>

B<sub>e</sub>rth<sub>e</sub>li<sub>e</sub>r<sub>,</sub> G<sub>.;</sub> N<sub>a</sub>bil<sub>,</sub> T<sub>.;</sub> N<sub>aou</sub>r<sub>,</sub> E<sub>.</sub> L<sub>.;</sub> Ni<sub>a</sub>mk<sub>e,</sub> R<sub>.;</sub> P<sub>e</sub>rl<sub>aza,</sub> S<sub>.; an</sub>d N<sub>eg</sub>li<sub>a,</sub> G<sub>.</sub> 2026<sub>.</sub> O<sub>n</sub> th<sub>e</sub> R<sub>o</sub>l<sub>e o</sub>f R<sub>evers</sub>ibl<sub>e</sub> I<sub>ns</sub>t<sub>ance</sub> Normalization. arXiv preprint arXiv:2603.11869.

Br<sub>a</sub>db<sub>u</sub>r<sub>y,</sub> J<sub>.;</sub> Fr<sub>os</sub>ti<sub>g,</sub> R<sub>.;</sub> H<sub>aw</sub>kin<sub>s,</sub> P<sub>.;</sub> J<sub>o</sub>hn<sub>so</sub>n<sub>,</sub> M<sub>.</sub> J<sub>.;</sub> K<sub>a</sub>t<sub>ar</sub>i<sub>ya,</sub> Y<sub>.;</sub> L<sub>eary,</sub> C<sub>.;</sub> M<sub>ac</sub>l<sub>aur</sub>i<sub>n,</sub> D<sub>.;</sub> N<sub>ecu</sub>l<sub>a,</sub> G<sub>.;</sub> P<sub>asz</sub>k<sub>e,</sub> A.; VanderPlas, J.; Wanderman-Milne, S.; and Zhan<sub>g</sub>, Q. 2018<sub>.</sub> JAX<sub>: composa</sub>bl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons o</sub>f P<sub>y</sub>th<sub>on</sub>+N<sub>um</sub>P<sub>y</sub> p<sup>ro</sup>g<sup>rams</sup>.

Ch<sub>a</sub>k<sub>ra</sub>b<sub>or</sub>t<sub>y,</sub> D<sub>.;</sub> Gh<sub>os</sub>h<sub>,</sub> S<sub>.; an</sub>d Gh<sub>os</sub>h<sub>,</sub> A<sub>.</sub> 2022<sub>.</sub> A<sub>u-</sub> t<sub>oenco</sub>d<sub>er</sub> b<sub>ase</sub>d h<sub>y</sub>b<sub>r</sub>id <sub>mu</sub>lti<sub>-</sub>t<sub>as</sub>k <sub>pre</sub>di<sub>c</sub>t<sub>or ne</sub>t<sub>wor</sub>k f<sub>or</sub> d<sub>a</sub>il<sub>y open-</sub>hi<sub>g</sub>h<sub>-</sub>l<sub>ow-c</sub>l<sub>ose pr</sub>i<sub>ces pre</sub>di<sub>c</sub>ti<sub>on o</sub>f I<sub>n</sub>di<sub>an s</sub>t<sub>oc</sub>k<sub>s.</sub> arXiv preprint arXiv:2204.13422.

D<sub>as,</sub> S<sub>.</sub> R<sub>.;</sub> G<sub>oya</sub>l<sub>,</sub> T<sub>.; a</sub>nd Y<sub>a</sub>d<sub>av,</sub> M<sub>.</sub> 2026<sub>.</sub> M<sub>u</sub>lti<sub>va</sub>ri<sub>a</sub>t<sub>e</sub> Fi<sub>nanc</sub>i<sub>a</sub>l F<sub>orecas</sub>ti<sub>ng us</sub>i<sub>ng</sub> th<sub>e</sub> Ch<sub>ronos</sub> Ti<sub>me</sub> S<sub>er</sub>i<sub>es</sub> F<sub>oun-</sub> dation Models. arXiv preprint arXiv:2605.21504.

F<sub>a</sub>n<sub>,</sub> J<sub>.; a</sub>nd Sh<sub>e</sub>n<sub>,</sub> Y<sub>.</sub> 2024<sub>.</sub> St<sub>oc</sub>kMi<sub>xe</sub>r<sub>: a s</sub>im<sub>p</sub>l<sub>e ye</sub>t <sub>s</sub>tr<sub>o</sub>n<sub>g</sub> MLP-based architecture for stock price forecasting. In Proceedings of the AAAI conference on artificial intelligence, <sub>vo</sub>l<sub>u</sub>m<sub>e</sub> 38<sub>,</sub> 8389<sub>–</sub>8397<sub>.</sub>

F<sub>e</sub>n<sub>g,</sub> C<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> L<sub>.;</sub> <sub>a</sub>nd Kr<sub>o</sub>m<sub>pass,</sub> D<sub>.</sub> 2024<sub>.</sub> Onl<sub>y</sub> th<sub>e</sub> <sub>curve s</sub>h<sub>ape ma</sub>tt<sub>ers:</sub> T<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>or zero-</sub> <sub>s</sub>h<sub>o</sub>t <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> ti<sub>me</sub> <sub>ser</sub>i<sub>es</sub> f<sub>orecas</sub>ti<sub>ng</sub> th<sub>roug</sub>h <sub>nex</sub>t <sub>curve</sub> shape prediction. arXiv preprint arXiv:2402.07570.

Fi<sub>sze</sub>d<sub>er,</sub> P<sub>.;</sub> F<sub>a</sub>łd<sub>z</sub>iń<sub>s</sub>ki<sub>,</sub> M<sub>.; an</sub>d M<sub>o</sub>l<sub>n</sub>á<sub>r,</sub> P<sub>.</sub> 2023<sub>.</sub> M<sub>o</sub>d<sub>e</sub>l<sub>-</sub> i<sub>ng an</sub>d f<sub>orecas</sub>ti<sub>ng</sub> d<sub>ynam</sub>i<sub>c con</sub>diti<sub>ona</sub>l <sub>corre</sub>l<sub>a</sub>ti<sub>ons w</sub>ith opening, high, low, and closing prices. Journal of Empirical Finance, 70: 308–321.

H<sub>ee</sub>k<sub>,</sub> J<sub>.;</sub> L<sub>evs</sub>k<sub>aya,</sub> A<sub>.;</sub> Oli<sub>ver,</sub> A<sub>.;</sub> Ritt<sub>er,</sub> M<sub>.;</sub> R<sub>on</sub>d<sub>ep</sub>i<sub>erre,</sub> B<sub>.;</sub> St<sub>e</sub>i<sub>ner,</sub> A<sub>.; an</sub>d <sub>van</sub> Z<sub>ee,</sub> M<sub>.</sub> 2024<sub>.</sub> Fl<sub>ax:</sub> A <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k lib<sub>rary an</sub>d <sub>ecosys</sub>t<sub>em</sub> f<sub>or</sub> JAX<sub>.</sub>

H<sub>ung,</sub> C<sub>.-</sub>C<sub>.; an</sub>d Ch<sub>en,</sub> Y<sub>.-</sub>J<sub>.</sub> 2021<sub>.</sub> DPP<sub>:</sub> D<sub>eep pre</sub>di<sub>c</sub>t<sub>or</sub> f<sub>or</sub> price movement from candlestick charts. Plos one, 16(6): <sub>e</sub>0252404<sub>.</sub>

Kim<sub>,</sub> T<sub>.;</sub> Kim<sub>,</sub> J<sub>.;</sub> T<sub>ae,</sub> Y<sub>.;</sub> P<sub>a</sub>rk<sub>,</sub> C<sub>.;</sub> Ch<sub>o</sub>i<sub>,</sub> J<sub>.-</sub>H<sub>.; a</sub>nd Ch<sub>oo,</sub> J<sub>.</sub> 2021<sub>.</sub> R<sub>evers</sub>ibl<sub>e</sub> i<sub>ns</sub>t<sub>ance norma</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or accura</sub>t<sub>e</sub> ti<sub>me-</sub> series forecasting against distribution shift. In International conference on learning representations.

Liu, Y.; Qin, G.; Huan<sub>g</sub>, X.; Wan<sub>g</sub>, J.; and Lon<sub>g</sub>, M. 2025. Ti<sub>mer-x</sub>l<sub>:</sub> L<sub>ong-con</sub>t<sub>ex</sub>t t<sub>rans</sub>f<sub>ormers</sub> f<sub>or un</sub>ifi<sub>e</sub>d ti<sub>me ser</sub>i<sub>es</sub> forecasting. In International Conference on Learning Representations, volume 2025, 83982–84006.

Li<sub>u,</sub> Y<sub>.;</sub> Zh<sub>a</sub>n<sub>g,</sub> H<sub>.;</sub> Li<sub>,</sub> C<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> X<sub>.;</sub> W<sub>a</sub>n<sub>g,</sub> J<sub>.; a</sub>nd L<sub>o</sub>n<sub>g,</sub> M<sub>.</sub> 2024<sub>.</sub> Ti<sub>mer:</sub> G<sub>enera</sub>ti<sub>ve pre-</sub>t<sub>ra</sub>i<sub>ne</sub>d t<sub>rans</sub>f<sub>ormers are</sub> l<sub>arge</sub> time series models. arXiv preprint arXiv:2402.02368.

Liu, Z.; Chen<sub>g</sub>, M.; Li, Z.; Huan<sub>g</sub>, Z.; Liu, Q.; Xie, Y<sub>.; an</sub>d Ch<sub>en,</sub> E<sub>.</sub> 2023<sub>.</sub> Ad<sub>ap</sub>ti<sub>ve norma</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or non-</sub> <sub>s</sub>t<sub>a</sub>ti<sub>onary</sub> ti<sub>me ser</sub>i<sub>es</sub> f<sub>orecas</sub>ti<sub>ng:</sub> A t<sub>empora</sub>l <sub>s</sub>li<sub>ce perspec-</sub> tive. Advances in Neural Information Processing Systems, 36<sub>:</sub> 14273<sub>–</sub>14292<sub>.</sub>

S<sub>er</sub>d<sub>ar,</sub> N<sub>.;</sub> St<sub>e</sub>li<sub>os,</sub> B<sub>.;</sub> M<sub>c</sub>C<sub>o</sub>ll<sub>,</sub> J<sub>.; an</sub>d L<sub>ee,</sub> D<sub>.</sub> 2021<sub>.</sub> M<sub>u</sub>lti<sub>-</sub> <sub>var</sub>i<sub>a</sub>t<sub>e</sub> ti<sub>me-vary</sub>i<sub>ng</sub> <sub>parame</sub>t<sub>er</sub> <sub>mo</sub>d<sub>e</sub>lli<sub>ng</sub> f<sub>or</sub> <sub>s</sub>t<sub>oc</sub>k <sub>mar</sub>k<sub>e</sub>t<sub>s.</sub> Empirical Economics, 61(2): 947–972.

Shi, X.; Wan<sub>g</sub>, S.; Nie, Y.; Li, D.; Ye, Z.; Wen, Q.; and Jin, M<sub>.</sub> 2025<sub>.</sub> Ti<sub>me-moe:</sub> Billi<sub>on-sca</sub>l<sub>e</sub> ti<sub>me ser</sub>i<sub>es</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> models with mixture of experts. In International conference on learning representations, volume 2025, 34635–34667.

Si<sub>m</sub>th<sub>ara</sub>k<sub>ao,</sub> S<sub>.</sub> 2022<sub>.</sub> Bit<sub>co</sub>i<sub>n</sub> <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>pr</sub>i<sub>ce</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>recurren</sub>t <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>.</sub>

T<sub>epe</sub>l<sub>ya</sub>n<sub>,</sub> R<sub>.</sub> 2025<sub>.</sub> Enh<sub>a</sub>n<sub>c</sub>in<sub>g</sub> OHLC D<sub>a</sub>t<sub>a w</sub>ith Timin<sub>g</sub> Features: A Machine Learning Evaluation. arXiv preprint arXiv:2509.16137.

W<sub>a</sub>n<sub>g,</sub> H<sub>.;</sub> H<sub>ua</sub>n<sub>g,</sub> W<sub>.;</sub> <sub>a</sub>nd W<sub>a</sub>n<sub>g,</sub> S<sub>.</sub> 2021<sub>.</sub> F<sub>o</sub>r<sub>ecas</sub>tin<sub>g</sub> <sub>ope</sub>n<sub>-</sub> high-low-close data contained in candlestick chart. arXiv preprint arXiv:2104.00581.

Ye, W.; Den<sub>g</sub>, S.; Zou, Q.; and Gui, N. 2024. Frequenc<sub>y</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve norma</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or non-s</sub>t<sub>a</sub>ti<sub>onary</sub> ti<sub>me ser</sub>i<sub>es</sub> f<sub>ore-</sub> casting. Advances in Neural Information Processing Systems, 37<sub>:</sub> 31350<sub>–</sub>31379<sub>.</sub>

## A Discussion

C<sub>ryp</sub>t<sub>ocurrency</sub> OHLC f<sub>orecas</sub>ti<sub>ng</sub> <sub>com</sub>bi<sub>nes</sub> <sub>severa</sub>l difi<sub>cu</sub>lti<sub>es</sub> th<sub>a</sub>t <sub>are</sub> <sub>usua</sub>ll<sub>y</sub> <sub>s</sub>t<sub>u</sub>di<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> i<sub>n</sub> <sub>genera</sub>l<sub>-purpose</sub> ti<sub>me-</sub> <sub>ser</sub>i<sub>es</sub> f<sub>orecas</sub>ti<sub>ng.</sub> Fi<sub>rs</sub>t<sub>, cryp</sub>t<sub>ocurrency asse</sub>t<sub>s ex</sub>hibit <sub>ex</sub>t<sub>reme numer</sub>i<sub>ca</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y:</sub> th<sub>e quo</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ue o</sub>f <sub>one asse</sub>t <sub>may</sub> b<sub>e</sub> b<sub>e</sub>l<sub>ow</sub> $1 0 ^ { - 7 }$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> <sub>ano</sub>th<sub>er</sub> <sub>may</sub> <sub>excee</sub>d $\mathrm { { \bar { 1 0 ^ { 5 } } } }$ <sub>.</sub> A <sub>s</sub>h<sub>are</sub>d f<sub>orecas</sub>ti<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l <sub>mus</sub>t th<sub>ere</sub>f<sub>ore</sub> l<sub>earn</sub> f<sub>rom</sub> <sub>ser</sub>i<sub>es</sub> <sub>w</sub>h<sub>ose</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>sca</sub>l<sub>es</sub> dif<sub>er</sub> b<sub>y more</sub> th<sub>an</sub> t<sub>we</sub>l<sub>ve or</sub>d<sub>ers o</sub>f <sub>magn</sub>it<sub>u</sub>d<sub>e.</sub> S<sub>econ</sub>d<sub>,</sub> OHLC <sub>o</sub>b<sub>serva</sub>ti<sub>ons are s</sub>t<sub>ruc</sub>t<sub>ura</sub>ll<sub>y cons</sub>t<sub>ra</sub>i<sub>ne</sub>d<sub>.</sub> A <sub>va</sub>lid <sub>can</sub>dl<sub>es</sub>ti<sub>c</sub>k <sub>mus</sub>t <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
H \geq \operatorname* { m a x } ( O , C ) , \qquad L \leq \operatorname* { m i n } ( O , C ) ,\tag{34}
$$

and a model may produce an invalid financial object even when its aggregate pointwise forecasting error is small.

R<sub>evers</sub>ibl<sub>e</sub> i<sub>ns</sub>t<sub>ance norma</sub>li<sub>za</sub>ti<sub>on re</sub>d<sub>uces</sub> t<sub>empora</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on s</sub>hift b<sub>y</sub> t<sub>rans</sub>f<sub>orm</sub>i<sub>ng eac</sub>h <sub>con</sub>t<sub>ex</sub>t i<sub>n</sub>t<sub>o norma</sub>li<sub>ze</sub>d <sub>coor-</sub> di<sub>na</sub>t<sub>es</sub> b<sub>e</sub>f<sub>ore</sub> it i<sub>s processe</sub>d b<sub>y</sub> th<sub>e</sub> f<sub>orecas</sub>ti<sub>ng</sub> b<sub>ac</sub>kb<sub>one.</sub> H<sub>owever, norma</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>s no</sub>t <sub>a s</sub>i<sub>ng</sub>l<sub>e unam</sub>bi<sub>guous opera</sub>ti<sub>on</sub> i<sub>n</sub> <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> OHLC f<sub>orecas</sub>ti<sub>ng.</sub> Wh<sub>en eac</sub>h <sub>c</sub>h<sub>anne</sub>l i<sub>s norma</sub>li<sub>ze</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y,</sub> th<sub>e open,</sub> hi<sub>g</sub>h<sub>,</sub> l<sub>ow, an</sub>d <sub>c</sub>l<sub>ose c</sub>h<sub>anne</sub>l<sub>s un</sub>d<sub>ergo</sub> dif<sub>eren</sub>t <sub>a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>cross-c</sub>h<sub>anne</sub>l <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>no</sub>t <sub>guaran</sub>t<sub>ee</sub>d t<sub>o</sub> <sub>rema</sub>i<sub>n</sub> <sub>va</sub>lid i<sub>n</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> I<sub>n con</sub>t<sub>ras</sub>t<sub>, w</sub>h<sub>en a</sub>ll f<sub>our c</sub>h<sub>anne</sub>l<sub>s s</sub>h<sub>are one pos</sub>iti<sub>ve a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on,</sub> th<sub>e</sub>i<sub>r pa</sub>i<sub>rw</sub>i<sub>se or</sub>d<sub>er</sub> i<sub>s preserve</sub>d <sub>exac</sub>tl<sub>y.</sub> Thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> th<sub>e</sub> <sub>con</sub>fi<sub>gura</sub>bl<sub>e</sub> <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>an</sub>d <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>componen</sub>t<sub>s</sub> <sub>o</sub>f C<sub>ryp</sub>t<sub>o</sub>L<sub>.</sub>

A <sub>secon</sub>d i<sub>ssue concerns</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e sys</sub>t<sub>em</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>s per</sub>f<sub>orme</sub>d<sub>.</sub> U<sub>n</sub>d<sub>er conven</sub>ti<sub>ona</sub>l R<sub>ev</sub>IN<sub>-s</sub>t<sub>y</sub>l<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng,</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>pre</sub>di<sub>c</sub>t<sub>s</sub> i<sub>n norma</sub>li<sub>ze</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> b<sub>u</sub>t th<sub>e pre</sub>di<sub>c</sub>ti<sub>on may</sub> b<sub>e</sub> i<sub>nverse norma</sub>li<sub>ze</sub>d b<sub>e</sub>f<sub>ore mean-square</sub>d <sub>error</sub> i<sub>s</sub> evaluated. This operation reintroduces the context scale into the empirical objective. For a sample with scalar context scale $s _ { n } ,$ <sub>,</sub> th<sub>e raw-space res</sub>id<sub>ua</sub>l i<sub>s</sub> $s _ { n }$ ti<sub>mes</sub> th<sub>e norma</sub>li<sub>ze</sub>d <sub>res</sub>id<sub>ua</sub>l<sub>, an</sub>d it<sub>s square</sub>d l<sub>oss</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore we</sub>i<sub>g</sub>ht<sub>e</sub>d b<sub>y</sub> $s _ { n } ^ { 2 }$ <sub>.</sub> I<sub>n a s</sub>h<sub>are</sub>d <sub>mu</sub>lti<sub>-asse</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> thi<sub>s we</sub>i<sub>g</sub>hti<sub>ng can a</sub>lt<sub>er</sub> th<sub>e aggrega</sub>t<sub>e gra</sub>di<sub>en</sub>t di<sub>rec</sub>ti<sub>on an</sub>d th<sub>e</sub> l<sub>oca</sub>l <sub>curva</sub>t<sub>ure o</sub>f th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em.</sub> TP-RevIN avoids this efect by evaluating the forecasting objective directly in normalized target coordinates.

C<sub>ryp</sub>t<sub>o</sub>L <sub>a</sub>dditi<sub>ona</sub>ll<sub>y eva</sub>l<sub>ua</sub>t<sub>es</sub> it<sub>s</sub> OHLC <sub>cons</sub>i<sub>s</sub>t<sub>ency pena</sub>lt<sub>y</sub> i<sub>n norma</sub>li<sub>ze</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> b<sub>o</sub>th th<sub>e</sub> f<sub>orecas</sub>ti<sub>ng</sub> objective and the auxiliary structural objective are prevented from inheriting raw numerical scale. Under CD normalization, <sub>norma</sub>li<sub>ze</sub>d<sub>-space</sub> OHLC <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s are exac</sub>tl<sub>y equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> b<sub>ecause a</sub>ll <sub>c</sub>h<sub>anne</sub>l<sub>s s</sub>h<sub>are</sub> th<sub>e same</sub> <sub>os</sub>iti<sub>ve a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> U<sub>n</sub>d<sub>er</sub> CI <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> thi<sub>s exac</sub>t <sub>e u</sub>i<sub>va</sub>l<sub>ence</sub> d<sub>oes no</sub>t h<sub>o</sub>ld<sub>;</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t t<sub>erm s</sub>h<sub>ou</sub>ld i<sub>ns</sub>t<sub>ea</sub>d b<sub>e</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>e</sub>d <sub>as a represen</sub>t<sub>a</sub>ti<sub>on-space regu</sub>l<sub>ar</sub>i<sub>zer, an</sub>d fi<sub>nanc</sub>i<sub>a</sub>l <sub>va</sub>lidit<sub>y mus</sub>t <sub>s</sub>till b<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>er</sub> i<sub>nverse norma</sub>li<sub>za</sub>ti<sub>on.</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y,</sub> th<sub>e aux</sub>ili<sub>ary</sub> l<sub>oss su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y re</sub>d<sub>uces or</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ons un</sub>d<sub>er</sub> b<sub>o</sub>th <sub>norma</sub>li<sub>za</sub>ti<sub>on sc</sub>h<sub>emes, a</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on ax</sub>i<sub>s</sub> t<sub>o rema</sub>i<sub>n a con</sub>fi<sub>gura</sub>bl<sub>e</sub> d<sub>es</sub>i<sub>gn c</sub>h<sub>o</sub>i<sub>ce.</sub>

Th<sub>e</sub> fi<sub>na</sub>l <sub>componen</sub>t i<sub>s</sub> th<sub>e s</sub>t<sub>a</sub>bili<sub>z</sub>i<sub>ng</sub> t<sub>erm use</sub>d i<sub>n</sub> th<sub>e norma</sub>li<sub>za</sub>ti<sub>on</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or.</sub> A fi<sub>xe</sub>d <sub>eps</sub>il<sub>on</sub> h<sub>as an a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e numer</sub>i<sub>ca</sub> <sub>sca</sub>l<sub>e an</sub>d <sub>may</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>var</sub>i<sub>ance o</sub>f <sub>ex</sub>t<sub>reme</sub>l<sub>y</sub> l<sub>ow-va</sub>l<sub>ue</sub>d <sub>asse</sub>t<sub>s w</sub>hil<sub>e</sub> b<sub>e</sub>i<sub>ng neg</sub>li<sub>g</sub>ibl<sub>e</sub> f<sub>or</sub> hi<sub>g</sub>h<sub>-va</sub>l<sub>ue</sub>d <sub>asse</sub>t<sub>s.</sub> C<sub>ryp</sub>t<sub>o</sub>L th<sub>ere</sub>f<sub>ore s</sub>t<sub>u</sub>di<sub>es a</sub> d<sub>ynam</sub>i<sub>c eps</sub>il<sub>on propor</sub>ti<sub>ona</sub>l t<sub>o</sub> th<sub>e square</sub>d <sub>con</sub>t<sub>ex</sub>t <sub>mean.</sub> Thi<sub>s c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y equ</sub>i<sub>var</sub>i<sub>an</sub>t <sub>un</sub>d<sub>er pos</sub>iti<sub>ve c</sub>h<sub>anges o</sub>f <sub>numer</sub>i<sub>ca</sub>l <sub>un</sub>it<sub>s an</sub>d <sub>ma</sub>k<sub>es</sub> th<sub>e regu</sub>l<sub>ar</sub>i<sub>za</sub>ti<sub>on s</sub>t<sub>reng</sub>th d<sub>epen</sub>d <sub>pr</sub>i<sub>mar</sub>il<sub>y on re</sub>l<sub>a</sub>ti<sub>ve ra</sub>th<sub>er</sub> th<sub>an a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>sca</sub>l<sub>e.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng sec</sub>ti<sub>ons</sub> f<sub>orma</sub>li<sub>ze</sub> th<sub>ese c</sub>l<sub>a</sub>i<sub>ms an</sub>d d<sub>e</sub>li<sub>m</sub>it th<sub>e</sub>i<sub>r assump</sub>ti<sub>ons.</sub>

## B Mathematical Analysis

L<sub>e</sub>t

$$
\mathcal { D } = \left\{ ( \mathbf { X } _ { n } , \mathbf { Y } _ { n } ) \right\} _ { n = 1 } ^ { N } , \qquad \mathbf { X } _ { n } \in \mathbb { R } ^ { L \times 4 } , \qquad \mathbf { Y } _ { n } \in \mathbb { R } ^ { H \times 4 } ,\tag{35}
$$

d<sub>eno</sub>t<sub>e a se</sub>t <sub>o</sub>f OHLC f<sub>orecas</sub>ti<sub>ng samp</sub>l<sub>es.</sub> Th<sub>e</sub> f<sub>our c</sub>h<sub>anne</sub>l<sub>s are or</sub>d<sub>ere</sub>d <sub>as</sub>

$$
{ \bf X } _ { n , t , : } = ( O _ { n , t } , H _ { n , t } , L _ { n , t } , C _ { n , t } ) .\tag{36}
$$

All <sub>norma</sub>li<sub>za</sub>ti<sub>on s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs are assume</sub>d t<sub>o</sub> b<sub>e measura</sub>bl<sub>e</sub> f<sub>unc</sub>ti<sub>ons o</sub>f th<sub>e o</sub>b<sub>serve</sub>d <sub>con</sub>t<sub>ex</sub>t ${ \bf X } _ { n }$ <sub>on</sub>l<sub>y.</sub> Th<sub>ey are</sub> t<sub>rea</sub>t<sub>e</sub>d <sub>as</sub> constants with respect to the forecasting parameters θ.

F<sub>or</sub> th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on ana</sub>l<sub>ys</sub>i<sub>s,</sub> th<sub>e</sub> f<sub>orecas</sub>t h<sub>or</sub>i<sub>zon an</sub>d <sub>c</sub>h<sub>anne</sub>l di<sub>mens</sub>i<sub>ons are vec</sub>t<sub>or</sub>i<sub>ze</sub>d i<sub>n</sub>t<sub>o</sub>

$$
d = 4 H , \qquad \mathbf { z } _ { n } , \widehat { \mathbf { z } } _ { n } \in \mathbb { R } ^ { d } .\tag{37}
$$

## Order Preservation Under CD and CI Normalization

## Channel-dependent normalization

U<sub>n</sub>d<sub>er</sub> CD <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>one</sub> <sub>s</sub>h<sub>are</sub>d l<sub>oca</sub>ti<sub>on</sub> <sub>an</sub>d <sub>one</sub> <sub>s</sub>h<sub>are</sub>d <sub>pos</sub>iti<sub>ve</sub> <sub>sca</sub>l<sub>e</sub> <sub>are</sub> <sub>compu</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>:</sub>

$$
\mu _ { n } ^ { \mathrm { C D } } = \frac { 1 } { 4 L } \sum _ { t = 1 } ^ { L } \sum _ { c = 1 } ^ { 4 } X _ { n , t , c } ,\tag{38}
$$

$$
\boldsymbol { v } _ { n } ^ { \mathrm { { C D } } } = \frac { 1 } { 4 L } \sum _ { t = 1 } ^ { L } \sum _ { c = 1 } ^ { 4 } \left( \boldsymbol { X } _ { n , t , c } - \boldsymbol { \mu } _ { n } ^ { \mathrm { { C D } } } \right) ^ { 2 } ,\tag{39}
$$

<sub>an</sub>d

$$
s _ { n } ^ { \mathrm { C D } } = \sqrt { v _ { n } ^ { \mathrm { C D } } + \epsilon _ { n } } > 0 .\tag{40}
$$

Th<sub>e s</sub>h<sub>are</sub>d <sub>a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub>

$$
T _ { n } ^ { \mathrm { C D } } ( x ) = { \frac { x - \mu _ { n } ^ { \mathrm { C D } } } { s _ { n } ^ { \mathrm { C D } } } } .\tag{41}
$$

Theorem B.1 (Order preservation under CD normalization). Let $x _ { 1 } , x _ { 2 } \in \mathbb { R }$ , and let $T _ { n } ^ { \mathrm { C D } }$ be defined by Equation (41) with $s _ { n } ^ { \mathrm { C D } } > 0 .$ . Then

$$
x _ { 1 } \geq x _ { 2 } \quad \iff \quad T _ { n } ^ { \mathrm { C D } } ( x _ { 1 } ) \geq T _ { n } ^ { \mathrm { C D } } ( x _ { 2 } ) .\tag{42}
$$

Consequently, every valid OHLC observation remains valid after CD normalization.

Proof. Subtracting the two transformed values gives

$$
T _ { n } ^ { \mathrm { C D } } ( x _ { 1 } ) - T _ { n } ^ { \mathrm { C D } } ( x _ { 2 } ) = \frac { x _ { 1 } - x _ { 2 } } { s _ { n } ^ { \mathrm { C D } } } .\tag{43}
$$

B<sub>ecause</sub> $s _ { n } ^ { \mathrm { C D } } > 0 ,$

$$
\mathrm { s i g n } \left( T _ { n } ^ { \mathrm { C D } } ( x _ { 1 } ) - T _ { n } ^ { \mathrm { C D } } ( x _ { 2 } ) \right) = \mathrm { s i g n } ( x _ { 1 } - x _ { 2 } ) .\tag{44}
$$

Th<sub>ere</sub>f<sub>ore</sub> $x _ { 1 } - x _ { 2 } \geq 0$ if and onl<sub>y</sub> if the transformed diference is nonne<sub>g</sub>ative, <sub>p</sub>rovin<sub>g</sub> Equation (42).

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>resu</sub>lt t<sub>o</sub> th<sub>e</sub> OHLC i<sub>nequa</sub>liti<sub>es</sub> <sub>g</sub>i<sub>ves</sub>

$$
H \geq { \cal O } \Longrightarrow \widetilde { H } ^ { \mathrm { C D } } \geq \widetilde { { \cal O } } ^ { \mathrm { C D } } ,\tag{45}
$$

$$
H \geq C \Longrightarrow \widetilde { H } ^ { \mathrm { C D } } \geq \widetilde { C } ^ { \mathrm { C D } } ,\tag{46}
$$

$$
{ \cal L } \le { \cal O } \Longrightarrow \widetilde { \cal L } ^ { \mathrm { C D } } \le \widetilde { \cal O } ^ { \mathrm { C D } } ,\tag{47}
$$

<sub>an</sub>d

$$
{ \cal L } \le { \cal C } \Longrightarrow { \cal \widetilde L } ^ { \mathrm { C D } } \le { \cal \widetilde C } ^ { \mathrm { C D } } .\tag{48}
$$

H<sub>ence mem</sub>b<sub>ers</sub>hi<sub>p</sub> i<sub>n</sub> th<sub>e</sub> OHLC f<sub>eas</sub>ibl<sub>e se</sub>t i<sub>s preserve</sub>d<sub>.</sub> Th<sub>e</sub> fi<sub>gure</sub> 2 <sub>s</sub>h<sub>ows</sub> h<sub>ow</sub> CD <sub>an</sub>d CI <sub>per</sub>f<sub>orm</sub> dif<sub>eren</sub>t <sub>on or</sub>d<sub>er</sub> <sub>p</sub>r<sub>ese</sub>r<sub>va</sub>ti<sub>o</sub>n f<sub>o</sub>r OHLC d<sub>a</sub>t<sub>a.</sub> □

Corollary B.2 (Equivalence of CD normalized and raw constraints). For every $s _ { n } ^ { \mathrm { C D } } > 0 ,$

$$
\left[ \widetilde { O } ^ { \mathrm { C D } } - \widetilde { H } ^ { \mathrm { C D } } \right] _ { + } = \frac { 1 } { s _ { n } ^ { \mathrm { C D } } } \left[ O - H \right] _ { + } .\tag{49}
$$

Analogous identities hold for every pairwise OHLC constraint. Therefore, the zero set of the normalized-space CD physics loss is identical to the zero set ofthe corresponding raw-space OHLC constraint loss.

Proof. Using the shared afine transformation,

$$
\widetilde { O } ^ { \mathrm { C D } } - \widetilde { H } ^ { \mathrm { C D } } = \frac { O - H } { s _ { n } ^ { \mathrm { C D } } } .\tag{50}
$$

F<sub>or</sub> $a > 0$ <sub>,</sub> th<sub>e</sub> <sub>pos</sub>iti<sub>ve-par</sub>t f<sub>unc</sub>ti<sub>on</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
[ a x ] _ { + } = a \left[ x \right] _ { + } .\tag{51}
$$

S<sub>e</sub>ttin<sub>g</sub> $a = 1 / s _ { n } ^ { \mathrm { C D } }$ <sub>p</sub>roves Equation (49).

Channel-independent normalization

U<sub>n</sub>d<sub>er</sub> CI <sub>norma</sub>li<sub>za</sub>ti<sub>on, eac</sub>h <sub>c</sub>h<sub>anne</sub>l $c \in \{ O , H , L , C \}$ h<sub>as</sub> it<sub>s own s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs:</sub>

$$
\mu _ { n , c } ^ { \mathrm { C I } } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } X _ { n , t , c } ,\tag{52}
$$

$$
\boldsymbol { v } _ { n , c } ^ { \mathrm { C I } } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \left( \boldsymbol { X } _ { n , t , c } - \boldsymbol { \mu } _ { n , c } ^ { \mathrm { C I } } \right) ^ { 2 } ,\tag{53}
$$

<sub>an</sub>d

$$
s _ { n , c } ^ { \mathrm { C I } } = \sqrt { v _ { n , c } ^ { \mathrm { C I } } + \epsilon _ { n , c } } > 0 .\tag{54}
$$

Th<sub>e c</sub>h<sub>anne</sub>l<sub>-spec</sub>ifi<sub>c</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub>

$$
T _ { n , c } ^ { \mathrm { C I } } ( x ) = \frac { x - \mu _ { n , c } ^ { \mathrm { C I } } } { s _ { n , c } ^ { \mathrm { C I } } } .\tag{55}
$$

Proposition B.3 (CI normalization does not guarantee OHLC order). There exist valid OHLC values satisfying $H > O$ for which

$$
T _ { n , H } ^ { \mathrm { C I } } ( H ) < T _ { n , O } ^ { \mathrm { C I } } ( O ) .\tag{56}
$$

Therefore, validity in original coordinates does not imply validity in CI-normalized coordinates.

Proof. Consider

$$
O = 1 0 , \qquad H = 1 1 ,\tag{57}
$$

<sub>so</sub> th<sub>a</sub>t $H > O$ <sub>.</sub> L<sub>e</sub>t th<sub>e c</sub>h<sub>anne</sub>l <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs</sub> b<sub>e</sub>

$$
\mu _ { O } = 9 , \qquad s _ { O } = 0 . 5 , \qquad \mu _ { H } = 1 0 . 8 , \qquad s _ { H } = 0 . 2 .\tag{58}
$$

Th<sub>e norma</sub>li<sub>ze</sub>d <sub>va</sub>l<sub>ues are</sub>

$$
\widetilde { O } ^ { \mathrm { C I } } = \frac { 1 0 - 9 } { 0 . 5 } = 2 ,\tag{59}
$$

<sub>an</sub>d

$$
\widetilde { H } ^ { \mathrm { C I } } = \frac { 1 1 - 1 0 . 8 } { 0 . 2 } = 1 .\tag{60}
$$

H<sub>ence</sub>

$$
H > { \cal O } \qquad \mathrm { b u t } \qquad \widetilde { H } ^ { \mathrm { C I } } < \widetilde { { \cal O } } ^ { \mathrm { C I } } .\tag{61}
$$

Th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore no</sub>t <sub>guaran</sub>t<sub>ee</sub>d<sub>.</sub>

□

Remark B.4. The proposition does not state that CI normalization always destroys OHLC ordering. It states only that no universal i<sub>mp</sub>li<sub>ca</sub>ti<sub>on o</sub>f th<sub>e</sub> f<sub>orm</sub>

$$
H \geq O \Longrightarrow \widetilde { H } ^ { \mathrm { C I } } \geq \widetilde { O } ^ { \mathrm { C I } }\tag{62}
$$

<sub>can</sub> b<sub>e es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub>dditi<sub>ona</sub>l <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ons on</sub> th<sub>e c</sub>h<sub>anne</sub>l<sub>-spec</sub>ifi<sub>c s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs.</sub>

Remark B.5. When the physics loss is evaluated in CI-normalized coordinates, it enforces a representation-space ordering <sub>ra</sub>th<sub>er</sub> th<sub>an an exac</sub>tl<sub>y equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>raw-space or</sub>d<sub>er</sub>i<sub>ng.</sub> O<sub>r</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space</sub> OHLC <sub>va</sub>lidit<sub>y mus</sub>t th<sub>ere</sub>f<sub>ore</sub> b<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>er</sub> i<sub>nverse</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on.</sub>

## Mathematical Analysis of Dynamic Epsilon

L<sub>e</sub>t $x _ { 1 } , \ldots , x _ { L }$ d<sub>eno</sub>t<sub>e</sub> <sub>one</sub> <sub>con</sub>t<sub>ex</sub>t <sub>sequence</sub> <sub>w</sub>ith <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>mean</sub> <sub>an</sub>d <sub>var</sub>i<sub>ance</sub>

$$
\mu = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } x _ { t } ,\tag{63}
$$

$$
v = { \frac { 1 } { L } } \sum _ { t = 1 } ^ { L } ( x _ { t } - \mu ) ^ { 2 } .\tag{64}
$$

Th<sub>e</sub> fi<sub>xe</sub>d<sub>-eps</sub>il<sub>on</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\widetilde { x } _ { t } ^ { \mathrm { f i x } } = \frac { x _ { t } - \mu } { \sqrt { v + \epsilon _ { 0 } } } , \qquad \epsilon _ { 0 } = 1 0 ^ { - 5 } .\tag{65}
$$

<sup>C</sup>r<sub>yp</sub>to<sup>L</sup> uses

$$
\epsilon ^ { \mathrm { d y n } } = \lambda \left( \mu ^ { 2 } + \delta \right) , \qquad \lambda = 1 0 ^ { - 5 } , \qquad \delta = 1 0 ^ { - 1 2 } ,\tag{66}
$$

<sub>an</sub>d

$$
\widetilde { x } _ { t } ^ { \mathrm { d y n } } = \frac { x _ { t } - \mu } { \sqrt { v + \lambda ( \mu ^ { 2 } + \delta ) } } .\tag{67}
$$

Proposition B.6 (Non-equivariance of fixed epsilon). Let

$$
x _ { t } ^ { \prime } = a x _ { t } , \qquad a > 0 .\tag{68}
$$

Then fixed-epsilon normalization satisfies

$$
\widetilde { x } _ { t } ^ { \prime \mathrm { f i x } } = \frac { x _ { t } - \mu } { \sqrt { v + \epsilon _ { 0 } / a ^ { 2 } } } .\tag{69}
$$

Consequently,

$$
\widetilde { x } _ { t } ^ { \prime \mathrm { f i x } } \neq \widetilde { x } _ { t } ^ { \mathrm { f i x } }\tag{70}
$$

in general.

Proof. Under Equation (68),

$$
\mu ^ { \prime } = a \mu , \qquad v ^ { \prime } = a ^ { 2 } v .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

(71)

$$
\begin{array} { c } { { \widetilde { x } _ { t } ^ { \prime \mathrm { f i x } } = \displaystyle \frac { a x _ { t } - a \mu } { \sqrt { a ^ { 2 } v + \epsilon _ { 0 } } } } } \\ { { = \displaystyle \frac { a ( x _ { t } - \mu ) } { a \sqrt { v + \epsilon _ { 0 } / a ^ { 2 } } } } } \\ { { = \displaystyle \frac { x _ { t } - \mu } { \sqrt { v + \epsilon _ { 0 } / a ^ { 2 } } } , } } \end{array}\tag{72}
$$

<sub>w</sub>h<sub>ere</sub> $a > 0$ was used. The denominator depends on a, proving the claim.

Theorem B.7 (Approximate scale equivariance of dynamic epsilon). Under the rescaling in Equation (68), dynamic-epsilon normalization satisfies

$$
\widetilde { x } _ { t } ^ { \prime \mathrm { d y n } } = \frac { x _ { t } - \mu } { \sqrt { v + \lambda \mu ^ { 2 } + \lambda \delta / a ^ { 2 } } } .\tag{73}
$$

If $\begin{array} { r } { \dot { { \boldsymbol { \cdot } } } \delta = 0 , } \end{array}$ then

$$
\widetilde { x } _ { t } ^ { \prime \mathrm { d y n } } = \widetilde { x } _ { t } ^ { \mathrm { d y n } }\tag{74}
$$

for every $a > 0 .$ For $\delta > 0 ,$ , the transformation is approximately equivariant whenever

$$
\delta \ll \mu ^ { 2 } \qquad a n d \qquad { \frac { \delta } { a ^ { 2 } } } \ll v + \lambda \mu ^ { 2 } .\tag{75}
$$

Proof. Using Equation (71),

$$
\begin{array} { r } { \widetilde { x } _ { t } ^ { \prime { \mathrm { d y n } } } = \frac { a \left( x _ { t } - \mu \right) } { \sqrt { a ^ { 2 } v + \lambda ( a ^ { 2 } \mu ^ { 2 } + \delta ) } } } \\ { = \frac { a \left( x _ { t } - \mu \right) } { a \sqrt { v + \lambda \mu ^ { 2 } + \lambda \delta / a ^ { 2 } } } } \\ { = \frac { x _ { t } - \mu } { \sqrt { v + \lambda \mu ^ { 2 } + \lambda \delta / a ^ { 2 } } } , } \end{array}\tag{76}
$$

which <sub>p</sub>roves Equation (73). If $\delta = 0 ,$ the ri<sub>g</sub>ht-hand side equals Equation (67). For $\delta > 0$ <sub>,</sub> th<sub>e on</sub>l<sub>y sca</sub>l<sub>e-</sub>d<sub>epen</sub>d<sub>en</sub>t dif<sub>erence</sub> i<sub>s</sub> th<sub>e</sub> t<sub>erm</sub> $\lambda \delta / \dot { a ^ { 2 } }$ , which is ne<sub>g</sub>li<sub>g</sub>ible under Equation (75). □

Proposition B.8 (Dependence on relative variability). Assume $\mu$ ̸= 0, and define the squared coeficient of variation as

$$
\mathrm { C V } ^ { 2 } = { \frac { v } { \mu ^ { 2 } } } .\tag{77}
$$

Ignoring the negligible δ term, the ratio between the dynamic regularizer and the empirical variance is

$$
{ \frac { \epsilon ^ { \mathrm { d y n } } } { v } } = { \frac { \lambda } { \mathrm { C V ^ { 2 } } } } .\tag{78}
$$

Thus,for assets with comparable relative volatility, the efect ofdynamic epsilon is approximately independent oftheir absolute quoted price.

Proof. From Equation (66), with δ omitted,

$$
{ \frac { \epsilon ^ { \mathrm { d y n } } } { v } } = { \frac { \lambda \mu ^ { 2 } } { v } } = { \frac { \lambda } { v / \mu ^ { 2 } } } = { \frac { \lambda } { \mathrm { C V ^ { 2 } } } } .\tag{79}
$$

Remark B.9. Dynamic epsilon is not universally preferable. If

$$
v \ll \lambda \mu ^ { 2 } ,\tag{80}
$$

th<sub>en</sub> th<sub>e</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>za</sub>ti<sub>on</sub> t<sub>erm</sub> d<sub>om</sub>i<sub>na</sub>t<sub>es</sub> th<sub>e</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>var</sub>i<sub>ance</sub> <sub>an</sub>d <sub>may</sub> <sub>compress</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ons</sub> i<sub>n</sub> <sub>near</sub>l<sub>y</sub> <sub>cons</sub>t<sub>an</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>s.</sub> It<sub>s</sub> <sub>va</sub>l<sub>ue</sub> <sub>s</sub>h<sub>ou</sub>ld th<sub>ere</sub>f<sub>ore</sub> b<sub>e</sub> <sub>exam</sub>i<sub>ne</sub>d <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>across</sub> <sub>asse</sub>t<sub>s</sub> <sub>an</sub>d <sub>vo</sub>l<sub>a</sub>tilit<sub>y</sub> <sub>reg</sub>i<sub>mes.</sub>

Jacobian Analysis of Scale-Dominance Mitigation

L<sub>e</sub>t

$$
\widetilde { \mathbf { X } } _ { n } = { \mathcal { N } } _ { n } ( \mathbf { X } _ { n } )\tag{81}
$$

d<sub>eno</sub>t<sub>e</sub> th<sub>e norma</sub>li<sub>ze</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>, an</sub>d l<sub>e</sub>t th<sub>e</sub> f<sub>orecas</sub>ti<sub>ng</sub> b<sub>ac</sub>kb<sub>one pro</sub>d<sub>uce</sub>

$$
\widehat { \mathbf { z } } _ { n } = f _ { \pmb { \theta } } \left( \widetilde { \mathbf { X } } _ { n } \right) .\tag{82}
$$

Th<sub>e norma</sub>li<sub>ze</sub>d t<sub>arge</sub>t <sub>an</sub>d <sub>norma</sub>li<sub>ze</sub>d <sub>res</sub>id<sub>ua</sub>l <sub>are</sub>

$$
{ \bf z } _ { n } = { \bf D } _ { n } ^ { - 1 } \left( { \bf y } _ { n } - { \pmb \mu } _ { n } \right) , \qquad { \bf e } _ { n } ( \pmb \theta ) = \widehat { { \bf z } } _ { n } - { \bf z } _ { n } ,\tag{83}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { D } _ { n } \succ 0$ i<sub>s</sub> di<sub>agona</sub>l<sub>.</sub> Th<sub>e recons</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\widehat { \mathbf { y } } _ { n } = \pmb { \mu } _ { n } + \mathbf { D } _ { n } \widehat { \mathbf { z } } _ { n } .\tag{84}
$$

Th<sub>e</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> J<sub>aco</sub>bi<sub>an</sub> i<sub>s</sub>

$$
\mathbf { J } _ { n } = \frac { \partial \widehat { \mathbf { z } } _ { n } } { \partial \pmb { \theta } } \in \mathbb { R } ^ { d \times p } .\tag{85}
$$

Theorem B.10 (Residual-Jacobian scaling under original-space RevIN). Assume that $\pmb { \mu } _ { n }$ and ${ \bf D } _ { n }$ are independent of θ. Define the original-space residual

$$
\mathbf { r } _ { n } ^ { \mathrm { R a w } } = \widehat { \mathbf { y } } _ { n } - \mathbf { y } _ { n } .\tag{86}
$$

Then

$$
{ \bf r } _ { n } ^ { \mathrm { R a w } } = { \bf D } _ { n } { \bf e } _ { n } ,\tag{87}
$$

and the Jacobian ofthe original-space residual is

$$
{ \bf J } _ { n } ^ { \mathrm { R a w } } = \frac { \partial { \bf r } _ { n } ^ { \mathrm { R a w } } } { \partial \pmb { \theta } } = { \bf D } _ { n } { \bf J } _ { n } .\tag{88}
$$

Consequently, the original-space MSE gradient contribution is

$$
\mathbf { g } _ { n } ^ { \mathrm { R a w } } = \left( \mathbf { J } _ { n } ^ { \mathrm { R a w } } \right) ^ { \top } \mathbf { r } _ { n } ^ { \mathrm { R a w } } = \mathbf { J } _ { n } ^ { \top } \mathbf { D } _ { n } ^ { 2 } \mathbf { e } _ { n } .\tag{89}
$$

In contrast, the TP-RevIN gradient contribution is

$$
\mathbf { g } _ { n } ^ { \mathrm { T P } } = \mathbf { J } _ { n } ^ { \top } \mathbf { e } _ { n } .\tag{90}
$$

Proof. From Equation (84) and the identity

$$
{ \bf y } _ { n } = { \pmb { \mu } } _ { n } + { \bf D } _ { n } { \bf z } _ { n } ,\tag{91}
$$

<sub>we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { { \bf r } _ { n } ^ { \mathrm { R a w } } = \pmb { \mu } _ { n } + { \bf D } _ { n } \pmb { \widehat { \bf z } } _ { n } - \pmb { \mu } _ { n } - { \bf D } _ { n } { \bf z } _ { n } } \\ & { \quad \quad = { \bf D } _ { n } \left( \widehat { \bf z } _ { n } - { \bf z } _ { n } \right) } \\ & { \quad \quad = { \bf D } _ { n } \mathbf { e } _ { n } , } \end{array}\tag{92}
$$

<sub>p</sub>rovin<sub>g</sub> Equation (87). Since ${ \bf D } _ { n }$ is independent of θ,

$$
{ \bf J } _ { n } ^ { \mathrm { R a w } } = \frac { \partial ( { \bf D } _ { n } { \bf e } _ { n } ) } { \partial \pmb { \theta } } = { \bf D } _ { n } \frac { \partial { \bf e } _ { n } } { \partial \pmb { \theta } } = { \bf D } _ { n } { \bf J } _ { n } ,\tag{93}
$$

b<sub>ecause</sub> th<sub>e</sub> t<sub>arge</sub>t $\mathbf { z } _ { n }$ is also independent of θ.

For a least-squares objective, the per-sample gradient is the transpose of the residual Jacobian multiplied by the residual:

$$
\begin{array} { r l } & { \mathbf { g } _ { n } ^ { \mathrm { R a w } } = \left( { \bf D } _ { n } { \bf J } _ { n } \right) ^ { \top } \left( { \bf D } _ { n } { \bf e } _ { n } \right) } \\ & { \quad \quad \quad = { \bf J } _ { n } ^ { \top } { \bf D } _ { n } ^ { \top } { \bf D } _ { n } { \bf e } _ { n } } \\ & { \quad \quad \quad = { \bf J } _ { n } ^ { \top } { \bf D } _ { n } ^ { 2 } { \bf e } _ { n } , } \end{array}\tag{94}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>ecause</sub> ${ \bf D } _ { n }$ i<sub>s</sub> di<sub>agona</sub>l <sub>an</sub>d <sub>pos</sub>iti<sub>ve.</sub> Th<sub>e</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>resu</sub>lt f<sub>o</sub>ll<sub>ows</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>res</sub>id<sub>ua</sub>l $\mathbf { e } _ { n }$ <sub>an</sub>d J<sub>aco</sub>bi<sub>an</sub> ${ \bf J } _ { n }$ □

Corollary B.11 (Scalar-scale gradient weighting). Under CD normalization,

$$
\mathbf { D } _ { n } = s _ { n } \mathbf { I } _ { d } ,\tag{95}
$$

and therefore

$$
\begin{array} { r } { \mathbf { g } _ { n } ^ { \mathrm { R a w } } = s _ { n } ^ { 2 } \mathbf { J } _ { n } ^ { \top } \mathbf { e } _ { n } = s _ { n } ^ { 2 } \mathbf { g } _ { n } ^ { \mathrm { T P } } . } \end{array}\tag{96}
$$

Thus, for equal normalized residual-Jacobian products, original-space RevIN weights the sample contributions in proportion $t o \ s _ { n } ^ { 2 }$

Remark B.12. The relation in Equation (96) does not imply that every high-scale sample necessarily has a larger gradient. The <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>gra</sub>di<sub>en</sub>t <sub>a</sub>l<sub>so</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>

$$
{ \mathbf { J } } _ { n } ^ { \top } { \mathbf { e } } _ { n } .\tag{97}
$$

TP<sub>-</sub>R<sub>ev</sub>IN <sub>removes</sub> th<sub>e</sub> <sub>exp</sub>li<sub>c</sub>it <sub>sca</sub>l<sub>e</sub> <sub>mu</sub>lti<sub>p</sub>li<sub>er</sub> b<sub>u</sub>t d<sub>oes</sub> <sub>no</sub>t <sub>equa</sub>li<sub>ze</sub> <sub>res</sub>id<sub>ua</sub>l<sub>s,</sub> J<sub>aco</sub>bi<sub>ans,</sub> <sub>gra</sub>di<sub>en</sub>t di<sub>rec</sub>ti<sub>ons,</sub> <sub>samp</sub>l<sub>e</sub> f<sub>requen-</sub> <sub>c</sub>i<sub>es, or</sub> t<sub>as</sub>k difi<sub>cu</sub>lt<sub>y.</sub>

## Stacked Jacobian geometry

St<sub>ac</sub>k th<sub>e norma</sub>li<sub>ze</sub>d <sub>res</sub>id<sub>ua</sub>l<sub>s an</sub>d J<sub>aco</sub>bi<sub>ans as</sub>

$$
\mathbf { e } = \mathrm { c o l } ( \mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { N } ) , \qquad \mathbf { J } = \mathrm { c o l } ( \mathbf { J } _ { 1 } , \ldots , \mathbf { J } _ { N } ) .\tag{98}
$$

D<sub>e</sub>fi<sub>ne</sub>

$$
\mathbf { S } = \mathrm { d i a g } ( \mathbf { D } _ { 1 } , \dots , \mathbf { D } _ { N } ) .\tag{99}
$$

Th<sub>en</sub>

$$
{ \bf r } ^ { \mathrm { R a w } } = { \bf S } { \bf e } , \qquad { \bf J } ^ { \mathrm { R a w } } = { \bf S } { \bf J } ,\tag{100}
$$

<sub>w</sub>h<sub>ereas</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>uses</sub>

$$
\begin{array} { r } { { \bf r } ^ { \mathrm { T P } } = { \bf e } , \qquad { \bf J } ^ { \mathrm { T P } } = { \bf J } . } \end{array}\tag{101}
$$

Theorem B.13 (Conditional Jacobian conditioning result). Assume CD normalization and an orthogonal decomposition

$$
\mathbb { R } ^ { p } = V _ { 1 } \oplus \cdots \oplus V _ { A } .\tag{102}
$$

Suppose the normalized Jacobian of asset a satisfies

$$
\mathbf { J } _ { a } ^ { \top } \mathbf { J } _ { a } = \mathbf { P } _ { a } ,\tag{103}
$$

where ${ \mathbf { P } } _ { a }$ is the orthogonal projector onto $V _ { a } ,$ , and

$$
\sum _ { a = 1 } ^ { A } \mathbf { P } _ { a } = \mathbf { I } _ { p } .\tag{104}
$$

Ifasset a has scalar scale $s _ { a } > 0 ,$ , then

$$
\begin{array} { r } { \kappa _ { 2 } \left( { \bf J } ^ { \mathrm { T P } } \right) = 1 , } \end{array}\tag{105}
$$

whereas

$$
\kappa _ { 2 } \left( { \bf J } ^ { \mathrm { R a w } } \right) = \frac { s _ { \mathrm { m a x } } } { s _ { \mathrm { m i n } } } .\tag{106}
$$

Proof. Under the stated assumptions,

$$
\left( \mathbf { J } ^ { \mathrm { T P } } \right) ^ { \top } \mathbf { J } ^ { \mathrm { T P } } = \sum _ { a = 1 } ^ { A } \mathbf { P } _ { a } = \mathbf { I } _ { p } .\tag{107}
$$

All <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>va</sub>l<sub>ues</sub> <sub>o</sub>f $\mathbf { J } ^ { \mathrm { T P } }$ <sub>are</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>equa</sub>l t<sub>o</sub> <sub>one.</sub>

F<sub>o</sub>r <sub>o</sub>ri<sub>g</sub>in<sub>a</sub>l<sub>-space</sub> R<sub>ev</sub>IN<sub>,</sub>

$$
\left( \mathbf { J } ^ { \mathrm { R a w } } \right) ^ { \top } \mathbf { J } ^ { \mathrm { R a w } } = \sum _ { a = 1 } ^ { A } s _ { a } ^ { 2 } \mathbf { P } _ { a } .\tag{108}
$$

<sup>F</sup>or an<sub>y</sub> $\mathbf { u } \in V _ { a }$

$$
\left( \sum _ { b = 1 } ^ { A } s _ { b } ^ { 2 } \mathbf { P } _ { b } \right) \mathbf { u } = s _ { a } ^ { 2 } \mathbf { u } .\tag{109}
$$

Th<sub>us</sub> th<sub>e s</sub>i<sub>ngu</sub>l<sub>ar va</sub>l<sub>ues o</sub>f $\mathbf { J } ^ { \mathrm { R a w } }$ <sub>are</sub> th<sub>e va</sub>l<sub>ues</sub> $s _ { a } .$ , with multiplicities determined by dim $\left( V _ { a } \right)$ <sub>.</sub> Th<sub>e</sub>i<sub>r ra</sub>ti<sub>o</sub> i<sub>s</sub> $s _ { \operatorname* { m a x } } / s _ { \operatorname* { m i n } }$ □

Remark B.14. Theorem B.13 is conditional. TP-RevIN removes the row scaling induced by the RevIN inverse map, but it does <sub>no</sub>t <sub>un</sub>i<sub>versa</sub>ll<sub>y guaran</sub>t<sub>ee a sma</sub>ll<sub>er</sub> J<sub>aco</sub>bi<sub>an con</sub>diti<sub>on num</sub>b<sub>er.</sub> I<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c an</sub>i<sub>so</sub>t<sub>ropy</sub> i<sub>n</sub> th<sub>e norma</sub>li<sub>ze</sub>d J<sub>aco</sub>bi<sub>ans may rema</sub>i<sub>n.</sub>

## Hessian Analysis of Scale-Dominance Mitigation

For sample n, define the normalized-space MSE

$$
\ell _ { n } ^ { \mathrm { T P } } = \frac { 1 } { 2 } \left. \mathbf { e } _ { n } \right. _ { 2 } ^ { 2 } .\tag{110}
$$

Th<sub>e</sub> <sub>o</sub>ri<sub>g</sub>in<sub>a</sub>l<sub>-space</sub> R<sub>ev</sub>IN MSE i<sub>s</sub>

$$
\boldsymbol { \ell } _ { n } ^ { \mathrm { R a w } } = \frac { 1 } { 2 } \mathbf { e } _ { n } ^ { \top } \mathbf { W } _ { n } \mathbf { e } _ { n } , \qquad \mathbf { W } _ { n } = \mathbf { D } _ { n } ^ { \top } \mathbf { D } _ { n } = \mathbf { D } _ { n } ^ { 2 } .\tag{111}
$$

L<sub>e</sub>t

$$
\widehat { z } _ { n , k }\tag{112}
$$

denote output component k, and define its parameter Hessian as

$$
\mathbf { H } _ { n , k } ^ { f } = \nabla _ { \pmb { \theta } } ^ { 2 } \widehat { z } _ { n , k } .\tag{113}
$$

Theorem B.15 (Exact TP-RevIN and original-space Hessians). Assume $\mathbf { W } _ { n }$ is independent of θ. Then the exact TP-RevIN Hessian is

$$
\mathbf { H } _ { n } ^ { \mathrm { T P } } = \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n } + \sum _ { k = 1 } ^ { d } e _ { n , k } \mathbf { H } _ { n , k } ^ { f } .\tag{114}
$$

The exact original-space RevIN Hessian is

$$
\mathbf { H } _ { n } ^ { \mathrm { R a w } } = \mathbf { J } _ { n } ^ { \top } \mathbf { W } _ { n } \mathbf { J } _ { n } + \sum _ { k = 1 } ^ { d } ( \mathbf { W } _ { n } \mathbf { e } _ { n } ) _ { k } \mathbf { H } _ { n , k } ^ { f } .\tag{115}
$$

Under CD normalization, where

$$
{ \mathbf W } _ { n } = s _ { n } ^ { 2 } { \mathbf I } _ { d } ,\tag{116}
$$

the complete per-sample Hessians satisfy

$$
\mathbf { H } _ { n } ^ { \mathrm { R a w } } = s _ { n } ^ { 2 } \mathbf { H } _ { n } ^ { \mathrm { T P } } .\tag{117}
$$

Proof. The TP-RevIN gradient is

$$
\nabla _ { \pmb { \theta } } \ell _ { n } ^ { \mathrm { T P } } = \mathbf { J } _ { n } ^ { \top } \mathbf { e } _ { n } .\tag{118}
$$

Dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng</sub> b<sub>y</sub> th<sub>e pro</sub>d<sub>uc</sub>t <sub>ru</sub>l<sub>e g</sub>i<sub>ves</sub>

$$
\nabla _ { \pmb { \theta } } ^ { 2 } \ell _ { n } ^ { \mathrm { T P } } = \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n } + \sum _ { k = 1 } ^ { d } e _ { n , k } \nabla _ { \pmb { \theta } } ^ { 2 } \widehat { z } _ { n , k } ,\tag{119}
$$

which <sub>p</sub>roves Equation (114).

F<sub>o</sub>r <sub>o</sub>ri<sub>g</sub>in<sub>a</sub>l<sub>-space</sub> R<sub>ev</sub>IN<sub>,</sub>

$$
\nabla _ { \pmb { \theta } } \ell _ { n } ^ { \mathrm { R a w } } = { \bf J } _ { n } ^ { \top } { \bf W } _ { n } { \bf e } _ { n } .\tag{120}
$$

Dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng aga</sub>i<sub>n g</sub>i<sub>ves a</sub> t<sub>erm</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve o</sub>f th<sub>e res</sub>id<sub>ua</sub>l <sub>an</sub>d <sub>a</sub> t<sub>erm</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve o</sub>f th<sub>e</sub> J<sub>aco</sub>bi<sub>an:</sub>

$$
\nabla _ { \pmb { \theta } } ^ { 2 } \ell _ { n } ^ { \mathrm { R a w } } = \mathbf { J } _ { n } ^ { \top } \mathbf { W } _ { n } \mathbf { J } _ { n } + \sum _ { k = 1 } ^ { d } ( \mathbf { W } _ { n } \mathbf { e } _ { n } ) _ { k } \mathbf { H } _ { n , k } ^ { f } ,\tag{121}
$$

<sub>p</sub>rovin<sub>g</sub> Equation (115).

If ${ \mathbf W } _ { n } = s _ { n } ^ { 2 } { \mathbf I } _ { d }$ <sub>,</sub> th<sub>en</sub>

$$
\begin{array} { r l } {  { \mathbf { H } _ { n } ^ { \mathrm { R a w } } = s _ { n } ^ { 2 } \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n } + s _ { n } ^ { 2 } \sum _ { k = 1 } ^ { d } e _ { n , k } \mathbf { H } _ { n , k } ^ { f } } } \\ & { = s _ { n } ^ { 2 } \mathbf { H } _ { n } ^ { \mathrm { T P } } , } \end{array}\tag{122}
$$

which <sub>p</sub>roves Equation (117).

Remark B.16. For CI normalization, $\mathbf { W } _ { n }$ i<sub>s</sub> <sub>genera</sub>ll<sub>y</sub> di<sub>agona</sub>l b<sub>u</sub>t <sub>no</sub>t <sub>a</sub> <sub>sca</sub>l<sub>ar</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>o</sub>f th<sub>e</sub> id<sub>en</sub>tit<sub>y.</sub> I<sub>n</sub> th<sub>a</sub>t <sub>case,</sub> th<sub>e</sub> <sub>exac</sub>t Hessian is <sub>g</sub>iven b<sub>y</sub> Equation (115); it is not <sub>g</sub>enerall<sub>y</sub> valid to write $\mathbf { H } _ { n } ^ { \mathrm { R a w } } = s _ { n } ^ { 2 } \mathbf { H } _ { n } ^ { \mathrm { T P } }$ <sub>us</sub>i<sub>ng one sca</sub>l<sub>ar</sub> $s _ { n }$

## Gauss–Newton curvature

Th<sub>e</sub> G<sub>auss–</sub>N<sub>ew</sub>t<sub>on ma</sub>t<sub>r</sub>i<sub>ces re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e pos</sub>iti<sub>ve-sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> fi<sub>rs</sub>t t<sub>erm o</sub>f th<sub>e exac</sub>t H<sub>ess</sub>i<sub>ans:</sub>

$$
\mathbf { G } ^ { \mathrm { T P } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n } ,\tag{123}
$$

<sub>an</sub>d

$$
\mathbf { G } ^ { \mathrm { R a w } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { J } _ { n } ^ { \top } \mathbf { W } _ { n } \mathbf { J } _ { n } .\tag{124}
$$

U<sub>n</sub>d<sub>er sca</sub>l<sub>ar</sub> CD <sub>sca</sub>l<sub>es,</sub>

$$
\mathbf { G } ^ { \mathrm { R a w } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } ^ { 2 } \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n } .\tag{125}
$$

Theorem B.17 (Loewner and eigenvalue bounds). Assume scalar scales satisfying

$$
0 < s _ { \operatorname* { m i n } } \le s _ { n } \le s _ { \operatorname* { m a x } } < \infty .\tag{126}
$$

Then

$$
s _ { \mathrm { m i n } } ^ { 2 } \mathbf { G } ^ { \mathrm { T P } } \preceq \mathbf { G } ^ { \mathrm { R a w } } \preceq s _ { \mathrm { m a x } } ^ { 2 } \mathbf { G } ^ { \mathrm { T P } } .\tag{127}
$$

On any common parameter subspace on which both matrices are positive definite,

$$
\lambda _ { \operatorname* { m i n } } \left( \mathbf { G } ^ { \mathrm { R a w } } \right) \geq s _ { \operatorname* { m i n } } ^ { 2 } \lambda _ { \operatorname* { m i n } } \left( \mathbf { G } ^ { \mathrm { T P } } \right) ,\tag{128}
$$

$$
\lambda _ { \operatorname* { m a x } } \left( \mathbf { G } ^ { \mathrm { R a w } } \right) \leq s _ { \operatorname* { m a x } } ^ { 2 } \lambda _ { \operatorname* { m a x } } \left( \mathbf { G } ^ { \mathrm { T P } } \right) ,\tag{129}
$$

and

$$
\kappa \left( { \bf G } ^ { \mathrm { R a w } } \right) \leq \left( \frac { s _ { \operatorname* { m a x } } } { s _ { \operatorname* { m i n } } } \right) ^ { 2 } \kappa \left( { \bf G } ^ { \mathrm { T P } } \right) .\tag{130}
$$

Proof. For any $\mathbf { u } \in \mathbb { R } ^ { p }$

$$
\mathbf { u } ^ { \top } \mathbf { G } ^ { \mathrm { R a w } } \mathbf { u } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } ^ { 2 } \left\| \mathbf { J } _ { n } \mathbf { u } \right\| _ { 2 } ^ { 2 } .\tag{131}
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Equation (126) termwise <sub>g</sub>ives

$$
s _ { \mathrm { m i n } } ^ { 2 } \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left. \mathbf { J } _ { n } \mathbf { u } \right. _ { 2 } ^ { 2 } \leq \mathbf { u } ^ { \top } \mathbf { G } ^ { \mathrm { R a w } } \mathbf { u }
$$

$$
\leq s _ { \operatorname* { m a x } } ^ { 2 } \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left\| \mathbf { J } _ { n } \mathbf { u } \right\| _ { 2 } ^ { 2 } .\tag{132}
$$

Th<sub>e</sub> <sub>ou</sub>t<sub>er</sub> <sub>express</sub>i<sub>ons</sub> <sub>are</sub>

$$
s _ { \mathrm { m i n } } ^ { 2 } \mathbf { u } ^ { \top } \mathbf { G } ^ { \mathrm { T P } } \mathbf { u }\tag{133}
$$

<sub>an</sub>d

$$
s _ { \mathrm { m a x } } ^ { 2 } \mathbf { u } ^ { \top } \mathbf { G } ^ { \mathrm { T P } } \mathbf { u } ,\tag{134}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>proves</sub> th<sub>e</sub> L<sub>oewner</sub> b<sub>oun</sub>d<sub>s.</sub> Th<sub>e</sub> <sub>e</sub>i<sub>genva</sub>l<sub>ue</sub> i<sub>nequa</sub>liti<sub>es</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> th<sub>e</sub> R<sub>ay</sub>l<sub>e</sub>i<sub>g</sub>h <sub>quo</sub>ti<sub>en</sub>t<sub>,</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>ra</sub>ti<sub>o</sub> <sub>g</sub>i<sub>ves</sub> E<sub>qua-</sub> tion (130). □

Theorem B.18 (Exact scale-induced conditioning under orthogonal asset subspaces). Assume the parameter space decomposes into mutually orthogonal subspaces $V _ { 1 } , \dots , V _ { A }$ , and suppose the normalized Gauss–Newton operatorfor asset a is

$$
{ \bf G } _ { a } = \lambda _ { a } { \bf P } _ { a } , \qquad \lambda _ { a } > 0 ,\tag{135}
$$

where ${ \mathbf { P } } _ { a }$ projects onto $V _ { a }$ . Let the asset sampling probabilities be $\pi _ { a } > 0 .$ . Then

$$
{ \bf G } ^ { \mathrm { T P } } = \bigoplus _ { a = 1 } ^ { A } \pi _ { a } \lambda _ { a } { \bf I } _ { V _ { a } } ,\tag{136}
$$

and

$$
\mathbf { G } ^ { \mathrm { R a w } } = \bigoplus _ { a = 1 } ^ { A } \pi _ { a } s _ { a } ^ { 2 } \lambda _ { a } \mathbf { I } _ { V _ { a } } .\tag{137}
$$

Consequently,

$$
\kappa \left( { { \bf G } ^ { \mathrm { T P } } } \right) = \frac { { \operatorname* { m a x } _ { a } } \pi _ { a } \lambda _ { a } } { { \operatorname* { m i n } _ { a } } \pi _ { a } \lambda _ { a } } ,\tag{138}
$$

whereas

$$
\kappa \left( { \bf G } ^ { \mathrm { R a w } } \right) = \frac { \operatorname* { m a x } _ { a } \pi _ { a } s _ { a } ^ { 2 } \lambda _ { a } } { \operatorname* { m i n } _ { a } \pi _ { a } s _ { a } ^ { 2 } \lambda _ { a } } .\tag{139}
$$

${ \cal I } f \pi _ { a }$ and $\lambda _ { a }$ are equal across assets, then

$$
\kappa \left( { \bf G } ^ { \mathrm { R a w } } \right) = \left( \frac { s _ { \mathrm { m a x } } } { s _ { \mathrm { m i n } } } \right) ^ { 2 } , \quad \quad \kappa \left( { \bf G } ^ { \mathrm { T P } } \right) = 1 .\tag{140}
$$

Proof. Because the subspaces are mutually orthogonal, the aggregate Gauss–Newton matrices are block diagonal. The eigenvalue <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith $V _ { a } \operatorname { i s } \pi _ { a } \lambda _ { a }$ <sub>un</sub>d<sub>er</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>an</sub>d $\overset { \cdot } { \pi } _ { a } s _ { a } ^ { 2 } \lambda _ { a }$ <sub>un</sub>d<sub>er or</sub>i<sub>g</sub>i<sub>na</sub>l<sub>-space</sub> R<sub>ev</sub>IN<sub>.</sub> Th<sub>e con</sub>diti<sub>on num</sub>b<sub>ers are</sub> th<sub>ere</sub>f<sub>ore</sub> th<sub>e</sub> ratios of the lar<sub>g</sub>est and smallest block ei<sub>g</sub>envalues, <sub>p</sub>rovin<sub>g</sub> Equations (138) and (139). Equal $\pi _ { a }$ <sub>an</sub>d $\lambda _ { a }$ <sub>re</sub>d<sub>uce</sub> th<sub>ese express</sub>i<sub>ons</sub> to Equation (140). □

Remark B.19 (Scope of the curvature claim). TP-RevIN removes the explicit scale weighting

$$
s _ { n } ^ { 2 } \mathbf { J } _ { n } ^ { \top } \mathbf { J } _ { n }\tag{141}
$$

f<sub>rom</sub> th<sub>e</sub> G<sub>auss–</sub>N<sub>ew</sub>t<sub>on geome</sub>t<sub>ry.</sub> It d<sub>oes no</sub>t <sub>un</sub>i<sub>versa</sub>ll<sub>y m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> th<sub>e con</sub>diti<sub>on num</sub>b<sub>er o</sub>f <sub>every non</sub>li<sub>near</sub> f<sub>orecas</sub>ti<sub>ng pro</sub>bl<sub>em.</sub> S<sub>ca</sub>l<sub>e we</sub>i<sub>g</sub>hti<sub>ng may occas</sub>i<sub>ona</sub>ll<sub>y compensa</sub>t<sub>e</sub> f<sub>or pre-ex</sub>i<sub>s</sub>ti<sub>ng curva</sub>t<sub>ure an</sub>i<sub>so</sub>t<sub>ropy.</sub> Th<sub>e conc</sub>l<sub>us</sub>i<sub>on</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> th<sub>a</sub>t TP<sub>-</sub>R<sub>ev</sub>IN <sub>removes</sub> th<sub>e componen</sub>t <sub>o</sub>f <sub>curva</sub>t<sub>ure</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>so</sub>l<sub>e</sub>l<sub>y</sub> b<sub>y</sub> th<sub>e</sub> R<sub>ev</sub>IN <sub>con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>es.</sub>

Example B.20 (Scale weighting may leave conditioning unchanged). Let

$$
\mathbf { G } _ { 1 } = \mathbf { G } _ { 2 } = \mathbf { I } _ { p } .\tag{142}
$$

Th<sub>en</sub>

$$
\begin{array} { r } { \mathbf { G } ^ { \mathrm { T P } } = 2 \mathbf { I } _ { p } , \qquad \mathbf { G } ^ { \mathrm { R a w } } = ( s _ { 1 } ^ { 2 } + s _ { 2 } ^ { 2 } ) \mathbf { I } _ { p } . } \end{array}\tag{143}
$$

B<sub>o</sub>th <sub>ma</sub>t<sub>r</sub>i<sub>ces</sub> h<sub>ave con</sub>diti<sub>on num</sub>b<sub>er one even w</sub>h<sub>en</sub> $s _ { 1 } \neq s _ { 2 }$ <sub>.</sub> Th<sub>us, sca</sub>l<sub>e</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> d<sub>oes no</sub>t <sub>necessar</sub>il<sub>y worsen con</sub>di ti<sub>on</sub>i<sub>ng w</sub>h<sub>en a</sub>ll <sub>samp</sub>l<sub>es exc</sub>it<sub>e</sub> id<sub>en</sub>ti<sub>ca</sub>l i<sub>so</sub>t<sub>rop</sub>i<sub>c parame</sub>t<sub>er</sub> di<sub>rec</sub>ti<sub>ons.</sub>

Example B.21 (Scale weighting can compensate for intrinsic curvature). Let

$$
{ \bf G } _ { 1 } = \left[ { \bf { 1 } } \quad 0 \right] , \qquad { \bf G } _ { 2 } = \left[ { \bf { 0 } } \quad { \bf { 0 } } 0 \right] .\tag{144}
$$

Th<sub>en</sub>

$$
\mathbf { G } ^ { \mathrm { T P } } = \left[ \begin{array} { c c } { 1 } & { 0 } \\ { 0 } & { 1 0 0 } \end{array} \right] , \qquad \kappa \left( \mathbf { G } ^ { \mathrm { T P } } \right) = 1 0 0 .\tag{145}
$$

Ch<sub>oos</sub>i<sub>ng</sub>

$$
s _ { 1 } ^ { 2 } = 1 0 0 , \qquad s _ { 2 } ^ { 2 } = 1\tag{146}
$$

<sub>g</sub><sup>i</sup>ves

$$
\mathbf { G } ^ { \mathrm { R a w } } = \left[ \begin{array} { l l } { 1 0 0 } & { \phantom { - } 0 } \\ { 0 } & { 1 0 0 } \end{array} \right] , \qquad \kappa \left( \mathbf { G } ^ { \mathrm { R a w } } \right) = 1 .\tag{147}
$$

Thi<sub>s</sub> <sub>examp</sub>l<sub>e</sub> <sub>con</sub>fi<sub>rms</sub> th<sub>a</sub>t <sub>un</sub>i<sub>versa</sub>l <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>super</sub>i<sub>or</sub>it<sub>y</sub> <sub>canno</sub>t b<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>me</sub>d<sub>.</sub> TP<sub>-</sub>R<sub>ev</sub>IN <sub>prov</sub>id<sub>es</sub> <sub>sca</sub>l<sub>e</sub> <sub>neu</sub>t<sub>ra</sub>lit<sub>y,</sub> <sub>no</sub>t <sub>an</sub> <sub>uncon</sub>diti<sub>ona</sub>l <sub>m</sub>i<sub>n</sub>i<sub>mum-con</sub>diti<sub>on-num</sub>b<sub>er guaran</sub>t<sub>ee.</sub>

## C Dataset Details and Volatility Analysis

Thi<sub>s sec</sub>ti<sub>on rov</sub>id<sub>es a</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d b<sub>rea</sub>kd<sub>own o</sub>f th<sub>e</sub> l<sub>ar e-sca</sub>l<sub>e cr</sub> t<sub>ocurrenc</sub> f<sub>orecas</sub>ti<sub>n</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>u</sub>tili<sub>ze</sub>d i<sub>n our em</sub> i<sub>r</sub>i<sub>ca</sub> evaluations. The dataset consists of historical O<sub>p</sub>en-Hi<sub>g</sub>h-Low-Close (OHLC) observations collected via the Binance API. In total, the dataset contains 15, 490, 300 data rows spanning 16 heterogeneous assets across 7 distinct temporal resolutions (5-minute, 15-minute, 30-minute, 1-hour, 2-hour, 4-hour, and 1-day intervals), yielding a total of 112 distinct time-series panels.

T<sub>a</sub>bl<sub>e</sub> 5<sub>:</sub> T<sub>e</sub>m<sub>po</sub>r<sub>a</sub>l <sub>cove</sub>r<sub>age</sub> <sub>pe</sub>r <sub>c</sub>r<sub>yp</sub>t<sub>ocu</sub>rr<sub>e</sub>n<sub>cy</sub> <sub>asse</sub>t in th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub>
<table><tr><td>Coin</td><td>Start Date</td><td>End Date</td></tr><tr><td>ADAUSDT</td><td>2018-04-17 03:30:00</td><td>2025-06-01 03:25:00</td></tr><tr><td>BCHUSDT BNBUSDT</td><td>2019-11-28 03:30:00 2017-11-06 03:30:00</td><td>2025-06-01 03:25:00 2025-06-01 03:25:00</td></tr><tr><td>BTCUSDT DOGEUSDT ETHUSDT</td><td>2017-08-17 03:30:00</td><td>2025-06-01 03:15:00</td></tr><tr><td>LINKUSDT LTCUSDT</td><td>2019-07-05 03:30:00 2017-08-17 03:30:00</td><td>2025-06-01 03:25:00</td></tr><tr><td>PEPEUSDT SHIBUSDT</td><td>2019-01-16 03:30:00 2017-12-13 03:30:00 2023-05-05 03:30:00</td><td>2025-06-01 03:25:00 2025-06-01 03:25:00 2025-06-01 03:25:00</td></tr></table>

## Dataset Overview and Coverage

The temporal coverage and active periods for each of the 16 selected cryptocurrency assets are detailed in Table 5. The assets cover hi<sub>g</sub>hl<sub>y</sub> mature coins such as Bitcoin (BTC) and Ethereum (ETH) with histories datin<sub>g</sub> back to 2017, alon<sub>g</sub>side more <sub>recen</sub>tl<sub>y</sub> l<sub>aunc</sub>h<sub>e</sub>d t<sub>o</sub>k<sub>ens suc</sub>h <sub>as</sub> SUI <sub>an</sub>d TON<sub>, w</sub>hi<sub>c</sub>h <sub>prov</sub>id<sub>e eva</sub>l<sub>ua</sub>ti<sub>on pa</sub>th<sub>s</sub> f<sub>or</sub> l<sub>ower-</sub>hi<sub>s</sub>t<sub>ory,</sub> hi<sub>g</sub>h<sub>-vo</sub>l<sub>a</sub>tilit<sub>y reg</sub>i<sub>mes.</sub>

Th<sub>e samp</sub>l<sub>e</sub> d<sub>ens</sub>it<sub>y var</sub>i<sub>es across</sub> ti<sub>me</sub>f<sub>rames.</sub> T<sub>a</sub>bl<sub>e</sub> 6 <sub>ou</sub>tli<sub>nes</sub> th<sub>e exac</sub>t <sub>num</sub>b<sub>er o</sub>f <sub>recor</sub>d<sub>e</sub>d d<sub>a</sub>t<sub>a rows</sub> f<sub>or eac</sub>h <sub>asse</sub>t <sub>across</sub> th<sub>e seven</sub> t<sub>empora</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>ons.</sub>

## Empirical Volatility Analysis and Optimization Dificulty

T<sub>o un</sub>d<sub>ers</sub>t<sub>an</sub>d th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on c</sub>h<sub>a</sub>ll<sub>enges across</sub> dif<sub>eren</sub>t <sub>se</sub>tti<sub>ngs, we ana</sub>l<sub>yze</sub> th<sub>e vo</sub>l<sub>a</sub>tilit<sub>y pro</sub>fil<sub>es o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub> V<sub>o</sub>l<sub>a</sub>tilit<sub>y</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d h<sub>ere as</sub> th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e percen</sub>t<sub>age c</sub>h<sub>ange</sub> i<sub>n</sub> th<sub>e c</sub>l<sub>ose pr</sub>i<sub>ce o</sub>f <sub>a can</sub>dl<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e prece</sub>di<sub>ng can</sub>dl<sub>e:</sub>

$$
v _ { t } = \frac { | C _ { t } - C _ { t - 1 } | } { C _ { t - 1 } } \times 1 0 0 \%\tag{148}
$$

W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e propor</sub>ti<sub>on o</sub>f <sub>can</sub>dl<sub>es excee</sub>di<sub>ng spec</sub>ifi<sub>e</sub>d th<sub>res</sub>h<sub>o</sub>ld l<sub>eve</sub>l<sub>s</sub> $( > 1 \% > 2 \% , > 3 \% , > 4 \% , \mathrm { a n d } > 5 \% )$

T<sub>a</sub>bl<sub>e</sub> 7 <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> <sub>aggrega</sub>t<sub>e</sub> <sub>vo</sub>l<sub>a</sub>tilit<sub>y</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> <sub>across</sub> <sub>a</sub>ll <sub>s</sub>i<sub>x</sub>t<sub>een</sub> <sub>asse</sub>t<sub>s.</sub> A<sub>s</sub> th<sub>e</sub> ti<sub>me</sub> f<sub>rame</sub> i<sub>ncreases,</sub> th<sub>e</sub> lik<sub>e</sub>lih<sub>oo</sub>d <sub>o</sub>f substantial price changes scales non-linearly. For instance, only 0.99% of candles in the 5-minute resolution experience a price deviation greater than 1%, and a mere 0.01% exceed 5%. Conversel , in the dail (1-da ) timeframe, 37.46% of the candle exceed a 1% change, and 11.33% experience fluctuations larger than 5%.

Thi<sub>s</sub> di<sub>spar</sub>it<sub>y</sub> di<sub>rec</sub>tl<sub>y a</sub>li<sub>gns w</sub>ith th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>ons presen</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n paper.</sub> A<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n our exper</sub>i<sub>men</sub>t<sub>a</sub>l results, forecastin<sub>g</sub> backbones ex<sub>p</sub>erience a reduction in <sub>p</sub>redictive accurac<sub>y</sub> (hi<sub>g</sub>her MAE and MAPE metrics) when o<sub>p</sub>eratin<sub>g</sub> <sub>on</sub> l<sub>arger</sub> ti<sub>me</sub>f<sub>rames.</sub> Th<sub>e</sub> hi<sub>g</sub>h d<sub>ens</sub>it<sub>y o</sub>f <sub>ex</sub>t<sub>reme re</sub>t<sub>urns on</sub> l<sub>arger</sub> t<sub>empora</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>ons represen</sub>t<sub>s a</sub> hi<sub>g</sub>hl<sub>y vo</sub>l<sub>a</sub>til<sub>e</sub> d<sub>ynam</sub>i<sub>ca</sub>l regime with increased variance in the target distributions, presenting a fundamentally more challenging forecasting objective.

## Asset-Specific Volatility Distributions

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>a</sub>bl<sub>es repor</sub>t th<sub>e granu</sub>l<sub>ar, asse</sub>t<sub>-spec</sub>ifi<sub>c vo</sub>l<sub>a</sub>tilit<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons across a</sub>ll <sub>exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>ons.</sub> T<sub>o accommo-</sub> d<sub>a</sub>t<sub>e</sub> th<sub>e s</sub>i<sub>ze o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>w</sub>ith<sub>ou</sub>t <sub>comp</sub>il<sub>a</sub>ti<sub>on</sub> i<sub>ssues,</sub> th<sub>e recor</sub>d<sub>s are</sub> di<sub>v</sub>id<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>par</sub>t<sub>s:</sub> T<sub>a</sub>bl<sub>e</sub> 8 <sub>covers asse</sub>t<sub>s</sub> f<sub>rom</sub> ADAUSDT t<sub>o</sub> LTCUSDT <sub>an</sub>d T<sub>a</sub>bl<sub>e</sub> 9 <sub>covers asse</sub>t<sub>s</sub> f<sub>rom</sub> PEPEUSDT t<sub>o</sub> XRPUSDT<sub>.</sub>

## D Experimental Details

T<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e</sub> f<sub>u</sub>ll <sub>repro</sub>d<sub>uc</sub>ibilit<sub>y,</sub> thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> th<sub>e</sub> <sub>concre</sub>t<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons,</sub> h<sub>yperparame</sub>t<sub>ers,</sub> <sub>an</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>pro</sub>t<sub>oco</sub>l<sub>s emp</sub>l<sub>oye</sub>d <sub>across</sub> th<sub>e</sub> f<sub>our pr</sub>i<sub>mary exper</sub>i<sub>men</sub>t<sub>a</sub>l t<sub>a</sub>bl<sub>es o</sub>f th<sub>e manuscr</sub>i<sub>p</sub>t<sub>.</sub>

For the evaluation presented in Table 1 (comparing models without normalization, standard RevIN, and our proposed TP RevIN), all backbones are trained strictl<sub>y</sub> usin<sub>g</sub> the standard Mean Squared Error (MSE) forecastin<sub>g</sub> objective without the <sub>aux</sub>ili<sub>ar</sub> h <sub>s</sub>i<sub>cs cons</sub>t<sub>ra</sub>i<sub>n</sub>t l<sub>oss</sub> $( \lambda _ { \mathrm { p h y } } = 0 )$ <sub>.</sub> Th<sub>e</sub> i<sub>npu</sub>t <sub>con</sub>t<sub>ex</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> i<sub>s</sub> fi<sub>xe</sub>d t<sub>o</sub> $L = 4 8 0$ <sub>s</sub>t<sub>e s.</sub> A <sub>se ara</sub>t<sub>e,</sub> i<sub>n</sub>d<sub>e en</sub>d<sub>en</sub>t experiment is conducted for each timeframe (5m, 30m) and target forecast horizon $( H \in \{ 5 , 1 5 \} )$ <sub>.</sub> All <sub>con</sub>fi<sub>gura</sub>ti<sub>ons u</sub>tili<sub>ze a</sub> fi<sub>xe</sub>d <sub>numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>bili<sub>za</sub>ti<sub>on cons</sub>t<sub>an</sub>t <sub>o</sub>f $\epsilon = 1 0 ^ { - 5 }$ i<sub>n</sub> th<sub>e</sub>i<sub>r norma</sub>li<sub>za</sub>ti<sub>on</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>ors.</sub> Th<sub>e</sub> F<sub>u</sub>ll <sub>resu</sub>lt<sub>s o</sub>f T<sub>a</sub>bl<sub>e</sub> 1 i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> <sub>paper are presen</sub>t<sub>e</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 10<sub>.</sub>

T<sub>a</sub>bl<sub>e</sub> 6<sub>:</sub> Ob<sub>serve</sub>d d<sub>a</sub>t<sub>a row coun</sub>t<sub>s par</sub>titi<sub>one</sub>d b<sub>y co</sub>i<sub>n an</sub>d ti<sub>me</sub> f<sub>rame.</sub>
<table><tr><td>Coin</td><td>5m</td><td>15m</td><td>30m</td><td>1h</td><td>2h</td><td>4h</td><td>1d</td></tr><tr><td>ADAUSDT</td><td>748,156</td><td>249,388</td><td>124,699</td><td>62,357</td><td>31,188</td><td>15,602</td><td>2,602</td></tr><tr><td>BCHUSDT</td><td>578,874</td><td>192,960</td><td>96,483</td><td>48,247</td><td>24,130</td><td>12,069</td><td>2,012</td></tr><tr><td>BNBUSDT</td><td>794,367</td><td>264,795</td><td>132,404</td><td>66,213</td><td>33,118</td><td>16,568</td><td>2,764</td></tr><tr><td>BTCUSDT</td><td>647,474</td><td>272,542</td><td>136,277</td><td>68,149</td><td>34,086</td><td>17,052</td><td>2,845</td></tr><tr><td>DOGEUSDT</td><td>620,750</td><td>206,919</td><td>103,463</td><td>51,737</td><td>25,875</td><td>12,943</td><td>2,158</td></tr><tr><td>ETHUSDT</td><td>817,609</td><td>272,542</td><td>134,789</td><td>68,149</td><td>34,086</td><td>17,052</td><td>2,845</td></tr><tr><td>LINKUSDT</td><td>669,530</td><td>223,179</td><td>111,594</td><td>55,803</td><td>27,909</td><td>13,961</td><td>2,328</td></tr><tr><td>LTCUSDT</td><td>783,714</td><td>261,243</td><td>130,628</td><td>65,325</td><td>32,674</td><td>16,346</td><td>2,727</td></tr><tr><td>PEPEUSDT</td><td>218,088</td><td>72,696</td><td>36,348</td><td>18,174</td><td>9,087</td><td>4,544</td><td>758</td></tr><tr><td>SHIBUSDT</td><td>426,878</td><td>142,293</td><td>71,147</td><td>35,574</td><td>17,789</td><td>8,896</td><td>1,483</td></tr><tr><td>SOLUSDT</td><td>505,085</td><td>168,363</td><td>84,184</td><td>42,095</td><td>21,052</td><td>10,529</td><td>1,755</td></tr><tr><td>SUIUSDT</td><td>218,736</td><td>72,912</td><td>36,456</td><td>18,228</td><td>9,114</td><td>4,557</td><td>760</td></tr><tr><td>TONUSDT</td><td>85,416</td><td>28,472</td><td>14,236</td><td>7,118</td><td>3,559</td><td>1,780</td><td>297</td></tr><tr><td>TRXUSDT</td><td>732,226</td><td>244,078</td><td>122,044</td><td>61,030</td><td>30,525</td><td>15,271</td><td>2,547</td></tr><tr><td>XLMUSDT</td><td>735,418</td><td>245,142</td><td>122,576</td><td>61,296</td><td>30,658</td><td>15,337</td><td>2,558</td></tr><tr><td>XRPUSDT</td><td>743,210</td><td>247,740</td><td>123,875</td><td>61,945</td><td>30,982</td><td>15,499</td><td>2,585</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 7<sub>:</sub> A<sub>ggrega</sub>t<sub>e vo</sub>l<sub>a</sub>tilit<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on across a</sub>ll <sub>asse</sub>t<sub>s par</sub>titi<sub>one</sub>d b<sub>y</sub> ti<sub>me</sub>f<sub>rame.</sub>
<table><tr><td>Timeframe</td><td> $> 1 \%$ </td><td> $> 2 \%$ </td><td> $> 3 \%$ </td><td> $> 4 \%$ </td><td> $> 5 \%$ </td></tr><tr><td>5m</td><td>0.99%</td><td>0.15%</td><td>0.05%</td><td>0.02%</td><td>0.01%</td></tr><tr><td>15m</td><td>3.10%</td><td>0.65%</td><td>0.22%</td><td>0.10%</td><td>0.05%</td></tr><tr><td>30m</td><td>5.89%</td><td>1.49%</td><td>0.56%</td><td>0.26%</td><td>0.14%</td></tr><tr><td>1h</td><td>9.94%</td><td>3.07%</td><td>1.29%</td><td>0.64%</td><td>0.35%</td></tr><tr><td>2h</td><td>15.45%</td><td>5.88%</td><td>2.73%</td><td>1.47%</td><td>0.86%</td></tr><tr><td>4h</td><td>21.75%</td><td>10.29%</td><td>5.43%</td><td>3.16%</td><td>1.97%</td></tr><tr><td>1d</td><td>37.46%</td><td>27.27%</td><td>20.20%</td><td>15.04%</td><td>11.33%</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 8<sub>:</sub> G<sub>ranu</sub>l<sub>ar vo</sub>l<sub>a</sub>tilit<sub>y</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> f<sub>or asse</sub>t<sub>s</sub> ADAUSDT th<sub>roug</sub>h LTCUSDT <sub>across exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>ons.</sub>
<table><tr><td>Coin</td><td>Timeframe</td><td>&gt; 1%</td><td>&gt; 2%</td><td>&gt; 3%</td><td>&gt; 4%</td><td>&gt; 5%</td></tr><tr><td>ADAUSDT</td><td>5m</td><td>0.90%</td><td>0.11%</td><td>0.03%</td><td>0.01%</td><td>0.01%</td></tr><tr><td>ADAUSDT</td><td>15m</td><td>3.35%</td><td>0.60%</td><td>0.18%</td><td>0.07%</td><td>0.03%</td></tr><tr><td>ADAUSDT</td><td>30m</td><td>6.63%</td><td>1.56%</td><td>0.51%</td><td>0.21%</td><td>0.10%</td></tr><tr><td>ADAUSDT</td><td>1h</td><td>11.38%</td><td>3.40%</td><td>1.30%</td><td>0.59%</td><td>0.30%</td></tr><tr><td>ADAUSDT</td><td>2h</td><td>17.65%</td><td>6.67%</td><td>3.03%</td><td>1.55%</td><td>0.85%</td></tr><tr><td>ADAUSDT</td><td>4h</td><td>23.81%</td><td>11.88%</td><td>6.14%</td><td>3.41%</td><td>2.02%</td></tr><tr><td>ADAUSDT</td><td>1d</td><td>39.64%</td><td>29.10%</td><td>21.88%</td><td>16.57%</td><td>12.61%</td></tr><tr><td>BCHUSDT</td><td>5m</td><td>0.74%</td><td>0.10%</td><td>0.03%</td><td>0.01%</td><td>0.01%</td></tr><tr><td>BCHUSDT</td><td>15m</td><td>2.61%</td><td>0.50%</td><td>0.17%</td><td>0.06%</td><td>0.03%</td></tr><tr><td>BCHUSDT</td><td>30m</td><td>5.27%</td><td>1.20%</td><td>0.46%</td><td>0.19%</td><td>0.10%</td></tr><tr><td>BCHUSDT</td><td>1h</td><td>9.10%</td><td>2.68%</td><td>1.06%</td><td>0.52%</td><td>0.27%</td></tr><tr><td>BCHUSDT</td><td>2h</td><td>14.70%</td><td>5.22%</td><td>2.35%</td><td>1.21%</td><td>0.70%</td></tr><tr><td>BCHUSDT</td><td>4h</td><td>21.52%</td><td>9.56%</td><td>4.89%</td><td>2.77%</td><td>1.64%</td></tr><tr><td>BCHUSDT</td><td>1d</td><td>37.69%</td><td>27.00%</td><td>19.69%</td><td>14.42%</td><td>10.89%</td></tr><tr><td>BNBUSDT</td><td>5m</td><td>0.92%</td><td>0.18%</td><td>0.06%</td><td>0.03%</td><td>0.01%</td></tr><tr><td>BNBUSDT</td><td>15m</td><td>2.72%</td><td>0.62%</td><td>0.24%</td><td>0.11%</td><td>0.06%</td></tr><tr><td>BNBUSDT</td><td>30m</td><td>5.16%</td><td>1.34%</td><td>0.55%</td><td>0.27%</td><td>0.16%</td></tr><tr><td>BNBUSDT</td><td>1h</td><td>8.85%</td><td>2.72%</td><td>1.22%</td><td>0.61%</td><td>0.36%</td></tr><tr><td>BNBUSDT</td><td>2h</td><td>14.09%</td><td>5.25%</td><td>2.45%</td><td>1.36%</td><td>0.82%</td></tr><tr><td>BNBUSDT BNBUSDT</td><td>4h 1d</td><td>20.60%</td><td>9.31%</td><td>4.80% 19.44%</td><td>2.73%</td><td>1.73%</td></tr><tr><td></td><td></td><td>36.81%</td><td>26.28%</td><td></td><td>14.12%</td><td>10.46%</td></tr><tr><td>BTCUSDT</td><td>5m</td><td>0.24%</td><td>0.03%</td><td>0.01%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>BTCUSDT</td><td>15m</td><td>1.46%</td><td>0.25%</td><td>0.09%</td><td>0.04%</td><td>0.02%</td></tr><tr><td>BTCUSDT</td><td>30m</td><td>3.01%</td><td>0.66%</td><td>0.21%</td><td>0.09%</td><td>0.04%</td></tr><tr><td>BTCUSDT BTCUSDT</td><td>1h 2h</td><td>5.64%</td><td>1.48%</td><td>0.52% 1.26%</td><td>0.24% 0.58%</td><td>0.12%</td></tr><tr><td>BTCUSDT</td><td>4h</td><td>9.74% 15.48%</td><td>3.23% 6.19%</td><td>2.87%</td><td>1.46%</td><td>0.31%</td></tr><tr><td>BTCUSDT</td><td>1d</td><td>33.90%</td><td>22.15%</td><td>15.12%</td><td>10.41%</td><td>0.74% 6.93%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DOGEUSDT DOGEUSDT</td><td>5m</td><td>1.30%</td><td>0.30%</td><td>0.12%</td><td>0.07%</td><td>0.04%</td></tr><tr><td>DOGEUSDT</td><td>15m</td><td>3.53%</td><td>0.94%</td><td>0.41%</td><td>0.23%</td><td>0.14%</td></tr><tr><td></td><td>30m</td><td>6.24%</td><td>1.89%</td><td>0.84% 1.69%</td><td>0.47% 0.93%</td><td>0.30% 0.59%</td></tr><tr><td>DOGEUSDT DOGEUSDT</td><td>1h 2h</td><td>10.03% 15.30%</td><td>3.48% 6.15%</td><td>3.11%</td><td>1.87%</td><td>1.23%</td></tr><tr><td>DOGEUSDT</td><td>4h</td><td>21.45%</td><td>10.33%</td><td>5.59%</td><td>3.38%</td><td>2.29%</td></tr><tr><td>DOGEUSDT</td><td>1d</td><td>34.96%</td><td>25.17%</td><td>18.96%</td><td>14.51%</td><td>10.66%</td></tr><tr><td>ETHUSDT</td><td>5m</td><td>0.66%</td><td>0.09%</td><td>0.03%</td><td>0.01%</td><td>0.01%</td></tr><tr><td>ETHUSDT</td><td>15m</td><td>2.25%</td><td>0.42%</td><td>0.12%</td><td>0.05%</td><td>0.03%</td></tr><tr><td>ETHUSDT</td><td>30m</td><td>4.70%</td><td>1.06%</td><td>0.35%</td><td>0.15%</td><td>0.07%</td></tr><tr><td>ETHUSDT</td><td>1h</td><td>8.41%</td><td>2.34%</td><td>0.89%</td><td>0.41%</td><td>0.18%</td></tr><tr><td>ETHUSDT</td><td>2h</td><td>13.89%</td><td>4.88%</td><td>2.12%</td><td>1.05%</td><td>0.51%</td></tr><tr><td>ETHUSDT</td><td>4h</td><td>20.27%</td><td>9.23%</td><td>4.53%</td><td>2.41%</td><td>1.33%</td></tr><tr><td>ETHUSDT</td><td>1d</td><td>37.62%</td><td>27.67%</td><td>20.50%</td><td>14.94%</td><td>10.79%</td></tr><tr><td>LINKUSDT</td><td>5m</td><td>1.22%</td><td>0.15%</td><td>0.04%</td><td>0.02%</td><td>0.01%</td></tr><tr><td>LINKUSDT</td><td>15m</td><td>4.22%</td><td>0.82%</td><td>0.23%</td><td>0.09%</td><td>0.04%</td></tr><tr><td>LINKUSDT</td><td>30m</td><td>8.09%</td><td>1.94%</td><td>0.67%</td><td>0.27%</td><td>0.12%</td></tr><tr><td>LINKUSDT</td><td>1h</td><td>13.38%</td><td>4.17%</td><td>1.64%</td><td>0.76%</td><td>0.37%</td></tr><tr><td>LINKUSDT</td><td>2h</td><td>20.06%</td><td>8.23%</td><td>3.64%</td><td>1.85%</td><td>1.00%</td></tr><tr><td>LINKUSDT</td><td>4h</td><td>26.78%</td><td>13.93%</td><td>7.54%</td><td>4.23%</td><td>2.44%</td></tr><tr><td>LINKUSDT</td><td>1d</td><td>42.16%</td><td>32.87%</td><td>25.91%</td><td>19.55%</td><td>15.13%</td></tr><tr><td>LTCUSDT</td><td>5m</td><td>0.85%</td><td>0.12%</td><td>0.04%</td><td>0.02%</td><td>0.01%</td></tr><tr><td>LTCUSDT</td><td>15m</td><td>2.90%</td><td>0.58%</td><td>0.18%</td><td>0.07%</td><td>0.04%</td></tr><tr><td>LTCUSDT</td><td>30m</td><td>5.73%</td><td>1.39%</td><td>0.48%</td><td>0.21%</td><td>0.10%</td></tr><tr><td>LTCUSDT</td><td>1h</td><td>9.99%</td><td>3.01%</td><td>1.15%</td><td>0.52%</td><td>0.27%</td></tr><tr><td>LTCUSDT</td><td>2h</td><td>15.61%</td><td>5.77%</td><td>2.56%</td><td>1.29%</td><td>0.80%</td></tr><tr><td>LTCUSDT</td><td>4h</td><td>22.10%</td><td>10.31%</td><td>5.26%</td><td>2.99% 14.82%</td><td>1.79% 11.30%</td></tr><tr><td>LTCUSDT</td><td>1d</td><td>38.59%</td><td>28.47%</td><td>20.80%</td></table>

T<sub>a</sub>bl<sub>e</sub> 9<sub>:</sub> Gr<sub>a</sub>n<sub>u</sub>l<sub>a</sub>r <sub>vo</sub>l<sub>a</sub>tilit<sub>y</sub> thr<sub>es</sub>h<sub>o</sub>ld<sub>s</sub> f<sub>o</sub>r <sub>asse</sub>t<sub>s</sub> PEPEUSDT thr<sub>oug</sub>h XRPUSDT <sub>ac</sub>r<sub>oss expe</sub>rim<sub>e</sub>nt<sub>a</sub>l r<sub>eso</sub>l<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub>
<table><tr><td>Coin</td><td>Timeframe</td><td>&gt; 1%</td><td>&gt; 2%</td><td>&gt; 3%</td><td>&gt; 4%</td><td>&gt; 5%</td></tr><tr><td>PEPEUSDT</td><td>5m</td><td>5.22%</td><td>0.52%</td><td>0.14%</td><td>0.05%</td><td>0.03%</td></tr><tr><td>PEPEUSDT</td><td>15m</td><td>9.48%</td><td>1.84%</td><td>0.63%</td><td>0.28%</td><td>0.14%</td></tr><tr><td>PEPEUSDT</td><td>30m</td><td>14.31%</td><td>3.95%</td><td>1.54%</td><td>0.72%</td><td>0.39%</td></tr><tr><td>PEPEUSDT</td><td>1h</td><td>19.11%</td><td>7.02%</td><td>3.26%</td><td>1.77%</td><td>1.11%</td></tr><tr><td>PEPEUSDT</td><td>2h</td><td>25.26%</td><td>11.63%</td><td>6.28%</td><td>3.64%</td><td>2.18%</td></tr><tr><td>PEPEUSDT</td><td>4h</td><td>30.97%</td><td>17.96%</td><td>11.14%</td><td>7.07%</td><td>4.75%</td></tr><tr><td>PEPEUSDT</td><td>1d</td><td>41.08%</td><td>34.35%</td><td>29.46%</td><td>24.70%</td><td>20.87%</td></tr><tr><td>SHIBUSDT</td><td>5m</td><td>1.36%</td><td>0.30%</td><td>0.11%</td><td>0.05%</td><td>0.02%</td></tr><tr><td>SHIBUSDT</td><td>15m</td><td>3.75%</td><td>0.99%</td><td>0.43%</td><td>0.22%</td><td>0.12%</td></tr><tr><td>SHIBUSDT</td><td>30m</td><td>6.71%</td><td>1.97%</td><td>0.93%</td><td>0.50%</td><td>0.29%</td></tr><tr><td>SHIBUSDT</td><td>1h</td><td>10.88%</td><td>3.62%</td><td>1.70%</td><td>1.01%</td><td>0.64%</td></tr><tr><td>SHIBUSDT</td><td>2h</td><td>16.34%</td><td>6.54%</td><td>3.28%</td><td>1.98%</td><td>1.36%</td></tr><tr><td>SHIBUSDT</td><td>4h</td><td>22.25%</td><td>10.87%</td><td>5.90%</td><td>3.87%</td><td>2.70%</td></tr><tr><td>SHIBUSDT</td><td>1d</td><td>37.25%</td><td>27.53%</td><td>19.16%</td><td>14.30%</td><td>10.59%</td></tr><tr><td>SOLUSDT</td><td>5m</td><td>1.49%</td><td>0.25%</td><td>0.07%</td><td>0.03%</td><td>0.02%</td></tr><tr><td>SOLUSDT</td><td>15m</td><td>4.84%</td><td>1.10%</td><td>0.40%</td><td>0.16%</td><td>0.08%</td></tr><tr><td>SOLUSDT</td><td>30m</td><td>8.98%</td><td>2.58%</td><td>1.04%</td><td>0.48%</td><td>0.23%</td></tr><tr><td>SOLUSDT</td><td>1h</td><td>14.58%</td><td>5.16%</td><td>2.25%</td><td>1.10%</td><td>0.63%</td></tr><tr><td>SOLUSDT</td><td>2h</td><td>21.18%</td><td>9.40%</td><td>4.77%</td><td>2.74%</td><td>1.62%</td></tr><tr><td>SOLUSDT SOLUSDT</td><td>4h 1d</td><td>28.27%</td><td>15.44%</td><td>9.10%</td><td>5.66%</td><td>3.92%</td></tr><tr><td></td><td></td><td>40.99%</td><td>33.41%</td><td>26.97%</td><td>21.78%</td><td>17.45%</td></tr><tr><td>SUIUSDT</td><td>5m</td><td>1.04%</td><td>0.11%</td><td>0.02%</td><td>0.01%</td><td>0.01%</td></tr><tr><td>SUIUSDT</td><td>15m</td><td>4.55%</td><td>0.71%</td><td>0.18%</td><td>0.05%</td><td>0.02%</td></tr><tr><td>SUIUSDT</td><td>30m</td><td>9.10%</td><td>2.02%</td><td>0.63%</td><td>0.21%</td><td>0.09%</td></tr><tr><td>SUIUSDT</td><td>1h 2h</td><td>15.37%</td><td>4.78%</td><td>1.91%</td><td>0.78%</td><td>0.34%</td></tr><tr><td>SUIUSDT SUIUSDT</td><td>4h</td><td>21.73%</td><td>9.34% 15.25%</td><td>4.15%</td><td>2.16%</td><td>1.04%</td></tr><tr><td>SUIUSDT</td><td>1d</td><td>27.94% 38.60%</td><td>29.78%</td><td>8.65% 24.37%</td><td>4.98% 20.82%</td><td>3.14% 16.86%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TONUSDT</td><td>5m</td><td>0.30%</td><td>0.05%</td><td>0.01%</td><td>0.01%</td><td>0.00%</td></tr><tr><td>TONUSDT</td><td>15m</td><td>1.43%</td><td>0.23%</td><td>0.06%</td><td>0.03%</td><td>0.01%</td></tr><tr><td>TONUSDT</td><td>30m</td><td>3.39%</td><td>0.53%</td><td>0.18%</td><td>0.07%</td><td>0.04%</td></tr><tr><td>TONUSDT TONUSDT</td><td>1h 2h</td><td>7.80%</td><td>1.25%</td><td>0.46%</td><td>0.22%</td><td>0.08%</td></tr><tr><td>TONUSDT</td><td>4h</td><td>12.87% 20.40%</td><td>3.20% 6.63%</td><td>0.93% 2.53%</td><td>0.48% 1.24%</td><td>0.28% 0.56%</td></tr><tr><td>TONUSDT</td><td>1d</td><td>35.14%</td><td>23.99%</td><td>16.55%</td><td>11.15%</td><td>7.43%</td></tr><tr><td>TRXUSDT</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TRXUSDT</td><td>5m 15m</td><td>0.57% 2.10%</td><td>0.08% 0.40%</td><td>0.02% 0.12%</td><td>0.01% 0.05%</td><td>0.00% 0.02%</td></tr><tr><td>TRXUSDT</td><td>30m</td><td>4.10%</td><td>1.01%</td><td>0.35%</td><td>0.15%</td><td>0.09%</td></tr><tr><td>TRXUSDT</td><td>1h</td><td>7.22%</td><td>2.06%</td><td>0.91%</td><td>0.43%</td><td>0.22%</td></tr><tr><td>TRXUSDT</td><td>2h</td><td>11.92%</td><td>4.24%</td><td>1.93%</td><td>1.03%</td><td>0.60%</td></tr><tr><td>TRXUSDT</td><td>4h</td><td>17.73%</td><td>7.60%</td><td>3.94%</td><td>2.21%</td><td>1.32%</td></tr><tr><td>TRXUSDT</td><td>1d</td><td>35.15%</td><td>23.64%</td><td>16.34%</td><td>11.55%</td><td>8.25%</td></tr><tr><td>XLMUSDT</td><td>5m</td><td>0.86%</td><td>0.14%</td><td>0.04%</td><td>0.02%</td><td>0.01%</td></tr><tr><td>XLMUSDT</td><td>15m</td><td>2.92%</td><td>0.60%</td><td>0.21%</td><td>0.10%</td><td>0.05%</td></tr><tr><td>XLMUSDT</td><td>30m</td><td>5.65%</td><td>1.38%</td><td>0.52%</td><td>0.26%</td><td>0.13%</td></tr><tr><td>XLMUSDT</td><td>1h</td><td>9.74%</td><td>2.89%</td><td>1.25%</td><td>0.61%</td><td>0.35%</td></tr><tr><td>XLMUSDT</td><td>2h</td><td>15.29%</td><td>5.49%</td><td>2.49%</td><td>1.37%</td><td>0.81%</td></tr><tr><td>XLMUSDT</td><td>4h</td><td>21.71%</td><td>9.71%</td><td>4.97%</td><td>3.05%</td><td>1.87%</td></tr><tr><td>XLMUSDT</td><td>1d</td><td>37.19%</td><td>26.44%</td><td>19.12%</td><td>14.12%</td><td>10.29%</td></tr><tr><td>XRPUSDT</td><td>5m</td><td>0.90%</td><td>0.15%</td><td>0.05%</td><td>0.02%</td><td>0.01%</td></tr><tr><td>XRPUSDT</td><td>15m</td><td>2.71%</td><td>0.64%</td><td>0.23%</td><td>0.10%</td><td>0.06%</td></tr><tr><td>XRPUSDT</td><td>30m</td><td>5.05%</td><td>1.38%</td><td>0.57%</td><td>0.28%</td><td>0.16%</td></tr><tr><td>XRPUSDT</td><td>1h</td><td>8.63%</td><td>2.73%</td><td>1.19%</td><td>0.62%</td><td>0.37%</td></tr><tr><td>XRPUSDT</td><td>2h</td><td>13.86%</td><td>4.95%</td><td>2.43%</td><td>1.35%</td><td>0.79%</td></tr><tr><td>XRPUSDT</td><td>4h</td><td>19.87%</td><td>8.76%</td><td>4.67%</td><td>2.91% 12.85%</td><td>1.93% 10.26%</td></tr><tr><td>XRPUSDT</td><td>1d</td><td>35.33%</td><td>24.96%</td><td>17.61%</td></table>

The comparison against alternative temporal normalization baselines in Table 2 (FAN and SAN) is conducted using the 1h timeframe with a lookback window of $L = 4 8 0$ . All models in this evaluation are trained using the MSE objective only, <sub>w</sub>ith<sub>ou</sub>t <sub>any aux</sub>ili<sub>ary p</sub>h<sub>ys</sub>i<sub>cs-</sub>i<sub>n</sub>f<sub>orme</sub>d l<sub>osses.</sub> T<sub>o ensure a</sub> f<sub>a</sub>i<sub>r an</sub>d <sub>op</sub>ti<sub>ma</sub>l i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on o</sub>f <sub>eac</sub>h b<sub>ase</sub>li<sub>ne, we a</sub>d<sub>op</sub>t th<sub>e</sub>i<sub>r</sub> <sub>recommen</sub>d<sub>e</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>es:</sub>

• FAN: The forecastin<sub>g</sub> backbone and the fre<sub>q</sub>uenc<sub>y</sub> ada<sub>p</sub>tive <sub>p</sub>rojection module are o<sub>p</sub>timized jointl<sub>y</sub> for 3 e<sub>p</sub>ochs <sub>p</sub>er ex<sub>p</sub>er<sup>i</sup>ment.

• SAN: A decou<sub>p</sub>led<sub>,</sub> two-sta<sub>g</sub>e trainin<sub>g</sub> strate<sub>gy</sub> is ado<sub>p</sub>ted. First<sub>,</sub> the non-stationar<sub>y</sub> slice <sub>p</sub>rojection module is o<sub>p</sub>timized independently for 3 epochs until convergence. This projection module is subsequently frozen, and the main forecasting backbone is trained for 1 epoch.

• TP-RevIN: The model is trained for 1 e<sub>p</sub>och usin<sub>g</sub> the scale-ada<sub>p</sub>tive d<sub>y</sub>namic e<sub>p</sub>silon formulation $( \epsilon ^ { \mathrm { d y n } } = 1 0 ^ { - 5 } ( \mu ^ { 2 } { + } 1 0 ^ { - 1 2 } ) )$ t<sub>o</sub> h<sub>an</sub>dl<sub>e cross-asse</sub>t h<sub>e</sub>t<sub>erogene</sub>it<sub>y.</sub>

For the numerical stabilization analysis in Table 3 (comparing fixed versus dynamic epsilon), all configurations are evaluated on the 2h timeframe with an input context length of $L = 4 8 0$ . The models are trained strictly under the MSE objective to isolate the structural im<sub>p</sub>act of denominator re<sub>g</sub>ularization on low-valued assets (e.<sub>g</sub>., SHIB) versus hi<sub>g</sub>h-valued assets (e.<sub>g</sub>., BTC).

For the physical consistency evaluations in Table 4, all models are trained with a lookback window of $L = 4 8 0$ <sub>across</sub> dif<sub>eren</sub>t historical sampling intervals (30m and 1h). Under the standard RevIN configuration, the auxiliary candlestick constraint loss is <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>e space, w</sub>hi<sub>c</sub>h <sub>re</sub>i<sub>n</sub>t<sub>ro</sub>d<sub>uces a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>es.</sub> All <sub>o</sub>f th<sub>e eva</sub>l<sub>ua</sub>ti<sub>ons are</sub> d<sub>one</sub> <sub>un</sub>d<sub>er or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>space</sub> f<sub>or</sub> f<sub>a</sub>i<sub>r compar</sub>i<sub>son.</sub>

Actual Prices  
![](images/7f51984a08c14f753b24f807b3acee786cec3c828a35b7730f41cfe5543440ef.jpg)

Channel-Dependent Normalized Prices  
![](images/0749a99d62de9b1465423cd43e42bfa67471a3d4215bd71b12cc755f16497cbf.jpg)

Channel-Independent Normalized Prices  
![](images/6b491c2ee3457424a4a2b975688e4d0150832bf2d41db597596f705148da959b.jpg)  
Fi<sub>gure</sub> 2<sub>:</sub> Dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>an</sub>d <sub>c</sub>h<sub>anne</sub>l<sub>-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>on</sub> OHLC d<sub>a</sub>t<sub>a</sub> <sub>an</sub>d h<sub>ow</sub> th<sub>ey</sub> <sub>per</sub>f<sub>orm</sub> <sub>on</sub> <sub>or</sub>d<sub>er</sub> <sub>preserva</sub>ti<sub>on.</sub>

<table><tr><td rowspan=2 colspan=4>TimeFrame Horizon Norm Technique</td><td rowspan=1 colspan=3>Time-MoE                    Timer-XL                      Timer</td></tr><tr><td rowspan=1 colspan=1>MSE   MAE   PHY</td><td rowspan=1 colspan=1>MSE   MAE   PHY</td><td rowspan=1 colspan=1>MSE   MAE  PHY</td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=2 colspan=1>w/o RevINCI RevIN</td><td rowspan=1 colspan=1>5.7e+8  6325.3   1.312</td><td rowspan=1 colspan=1>5.2e+8  6571.64  5.536</td><td rowspan=1 colspan=1>6.15e+8 6548.55 15.154</td></tr><tr><td rowspan=1 colspan=1>9890.9   19.03   0.0040</td><td rowspan=1 colspan=1>4387.64   11.66   0.012</td><td rowspan=1 colspan=1>4722.96  12.37   0.439</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5</td><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>9868.58  18.93   0.0076</td><td rowspan=1 colspan=1>4485.7   11.81   0.0050</td><td rowspan=1 colspan=1>5013.64  12.70   1.71</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>6610.0   15.25   0.001</td><td rowspan=1 colspan=1>4584.75   12.34   0.0005</td><td rowspan=1 colspan=1>4510.90   12.14   0.278</td></tr><tr><td rowspan=1 colspan=1>7480.2   16.31   0.00049</td><td rowspan=1 colspan=1>4811.0   12.72   0.0004</td><td rowspan=1 colspan=1>4632.05   12.30   1.09</td></tr><tr><td rowspan=3 colspan=1>5m</td><td rowspan=3 colspan=2>15</td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>5.7e+8  6328.50  1.827</td><td rowspan=1 colspan=1>5.5e+8  6572.12  5.532</td><td rowspan=1 colspan=1>6.15e+8  6548.99  9.523</td></tr><tr><td rowspan=1 colspan=1>CI RevIN</td><td rowspan=1 colspan=1>2.20e+4  28.63   0.0046</td><td rowspan=1 colspan=1>1.05e+4  18.54   0.0159</td><td rowspan=1 colspan=1>1.06e+4  18.69   0.915</td></tr><tr><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>2.23e+4  28.62   0.0025</td><td rowspan=1 colspan=1>1.07e+4  18.71   0.0061</td><td rowspan=1 colspan=1>1.19e+4  19.73   6.285</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>1.53e+4   23.55   0.012</td><td rowspan=1 colspan=1>1.02e+4   18.39   0.0003</td><td rowspan=1 colspan=1>9949.94   18.09   0.390</td></tr><tr><td rowspan=1 colspan=1>1.95e+4  26.83   0.0010</td><td rowspan=1 colspan=1>1.01e+4   18.32   0.0002</td><td rowspan=1 colspan=1>1.04e+4   18.54   4.36</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>w/o RevINCI RevIN</td><td rowspan=1 colspan=1>5.76e+8 6327.02  2.189</td><td rowspan=1 colspan=1>5.9e+8  6748.77   2.32</td><td rowspan=1 colspan=1>6.15e+8 6547.06  9.84</td></tr><tr><td rowspan=1 colspan=1>3.37e+4  34.82   0.0016</td><td rowspan=1 colspan=1>1.74e+4  24.07   0.011</td><td rowspan=1 colspan=1>1.74e+4  23.94   1.38</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>30</td><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>3.36e+4  34.81   0.0024</td><td rowspan=1 colspan=1>1.73e+4  23.95   0.0071</td><td rowspan=1 colspan=1>1.99e+4  25.81   12.70</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>2.41e+4   29.36   0.015</td><td rowspan=1 colspan=1>1.69e+4  23.70   0.0003</td><td rowspan=1 colspan=1>1.60e+4  23.10   0.555</td></tr><tr><td rowspan=1 colspan=1>3.14e+4  33.64  0.00072</td><td rowspan=1 colspan=1>1.65e+4  23.44  0.00003</td><td rowspan=1 colspan=1>1.73e+4  23.94   8.90</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>7.73e+8 8438.67  1.271</td><td rowspan=1 colspan=1>7.66e+8 8367.45  0.501</td><td rowspan=1 colspan=1>7.71e+8 8408.11  5.717</td></tr><tr><td rowspan=1 colspan=1>CI RevIN</td><td rowspan=1 colspan=1>6.47e+4  54.60   0.0685</td><td rowspan=1 colspan=1>3.35e+4  37.58   0.063</td><td rowspan=1 colspan=1>3.22e+4  36.13   1.012</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5</td><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>6.33e+4  54.20   0.0307</td><td rowspan=1 colspan=1>3.38e+4  37.97   0.081</td><td rowspan=1 colspan=1>3.24e+4  36.39   3.957</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>4.67e+4  45.45   0.011</td><td rowspan=1 colspan=1>3.11e+4  36.04   0.003</td><td rowspan=1 colspan=1>3.03e+4  34.93   0.678</td></tr><tr><td rowspan=1 colspan=1>5.41e+4  49.12   0.0076</td><td rowspan=1 colspan=1>3.11e+4  36.26  0.00074</td><td rowspan=1 colspan=1>1.4e+4   18.54   4.365</td></tr><tr><td rowspan=3 colspan=1>30m</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>7.73e+8 8442.13  1.553</td><td rowspan=1 colspan=1>7.68e+8 8383.54  0.606</td><td rowspan=1 colspan=1>7.71e+8  8412.84  5.542</td></tr><tr><td rowspan=2 colspan=2>15</td><td rowspan=2 colspan=1>CI RevINCD RevIN</td><td rowspan=1 colspan=1>1.56e+5  86.77   0.0701</td><td rowspan=1 colspan=1>8.29e+4  59.11   0.228</td><td rowspan=1 colspan=1>7.66e+4  57.00   3.238</td></tr><tr><td rowspan=1 colspan=1>1.55e+5  87.05   0.0443</td><td rowspan=1 colspan=1>8.49e+4  60.007   0.163</td><td rowspan=1 colspan=1>8.51e+4  59.75   20.67</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>1.14e+5  72.48    0.12</td><td rowspan=1 colspan=1>7.34e+4  55.10   0.026</td><td rowspan=1 colspan=1>6.99e+4  53.38   1.123</td></tr><tr><td rowspan=1 colspan=1>1.37e+5  80.14   0.0078</td><td rowspan=1 colspan=1>7.24e+4  54.71   0.00052</td><td rowspan=1 colspan=1>7.36e+4  55.11   15.002</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=2>30</td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>7.74e+8 8449.87   1.643</td><td rowspan=1 colspan=1>7.69e+8 8392.46  0.522</td><td rowspan=1 colspan=1>7.72e+8 8424.90  6.923</td></tr><tr><td rowspan=2 colspan=1>CI RevINCD RevIN</td><td rowspan=1 colspan=1>2.29e+5  106.78   0.129</td><td rowspan=1 colspan=1>1.39e+5  78.05   0.082</td><td rowspan=1 colspan=1>1.34e+5  75.79   6.036</td></tr><tr><td rowspan=1 colspan=1>2.34e+5  107.85   0.214</td><td rowspan=1 colspan=1>1.43e+5  78.81    0.153</td><td rowspan=1 colspan=1>1.48e+5  80.23  39.168</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>1.79e+5  91.87   0.562</td><td rowspan=1 colspan=1>1.21e+5  71.71    0.001</td><td rowspan=1 colspan=1>1.16e+5  70.08   1.928</td></tr><tr><td rowspan=1 colspan=1>2.19e+5  101.75  0.0180</td><td rowspan=1 colspan=1>1.22e+5  71.92   0.001</td><td rowspan=1 colspan=1>1.26e+5  73.74  34.545</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>7.8e+8  8649.4   1.36</td><td rowspan=1 colspan=1>7.80e+8 8610.74  0.136</td><td rowspan=1 colspan=1>7.82e+8 8643.78  2.214</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CI RevIN</td><td rowspan=1 colspan=1>1.27e+5  79.73   0.139</td><td rowspan=1 colspan=1>6.84e+4  56.16   0.146</td><td rowspan=1 colspan=1>6.46e+4  52.80   1.571</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5</td><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>1.25e+5  79.68   0.084</td><td rowspan=1 colspan=1>6.90e+4  56.82   0.122</td><td rowspan=1 colspan=1>6.57e+4  52.94   4.97</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>9.38e+4  65.91    0.02</td><td rowspan=1 colspan=1>6.48e+4  54.44    0.008</td><td rowspan=1 colspan=1>6.16e+4  51.25   1.275</td></tr><tr><td rowspan=1 colspan=1>1.08e+5  71.86    0.042</td><td rowspan=1 colspan=1>6.16e+4  52.55   0.0032</td><td rowspan=1 colspan=1>6.34e+4  51.83   5.247</td></tr><tr><td rowspan=3 colspan=1>1h</td><td rowspan=3 colspan=2>15</td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>7.85e+8 8680.34  1.532</td><td rowspan=1 colspan=1>7.81e+8 8629.43  0.302</td><td rowspan=1 colspan=1>7.83e+8 8646.39  1.833</td></tr><tr><td rowspan=1 colspan=1>CI RevIN</td><td rowspan=1 colspan=1>3.05e+5  128.34   0.296</td><td rowspan=1 colspan=1>1.73e+5  90.42   0.455</td><td rowspan=1 colspan=1>1.62e+5  86.06   6.20</td></tr><tr><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>3.05e+5  127.82   0.234</td><td rowspan=1 colspan=1>1.77e+5  90.61    0.520</td><td rowspan=1 colspan=1>1.71e+5  88.88   29.87</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>2.31e+5  108.28   0.571</td><td rowspan=1 colspan=1>1.48e+5  82.80   0.055</td><td rowspan=1 colspan=1>1.39e+5  78.92   2.365</td></tr><tr><td rowspan=1 colspan=1>2.75e+5  119.51   0.063</td><td rowspan=1 colspan=1>1.47e+5  82.38   0.0042</td><td rowspan=1 colspan=1>1.48e+5  82.25   23.58</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>w/o RevIN</td><td rowspan=1 colspan=1>7.85e+8 8691.64  1.728</td><td rowspan=1 colspan=1>7.82e+8  8647.22  0.290</td><td rowspan=1 colspan=1>7.84e+8 8670.49  2.358</td></tr><tr><td rowspan=1 colspan=1>CI RevIN</td><td rowspan=1 colspan=1>4.21e+5  151.30   0.509</td><td rowspan=1 colspan=1>2.80e+5  116.25   0.299</td><td rowspan=1 colspan=1>2.58e+5  110.42  10.22</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>30</td><td rowspan=1 colspan=1>CD RevIN</td><td rowspan=1 colspan=1>4.32e+5  155.56   0.743</td><td rowspan=1 colspan=1>2.79e+5  116.94   0.476</td><td rowspan=1 colspan=1>2.95e+5  118.23  57.194</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Two-Phase CI RevINTwo-Phase CD RevIN</td><td rowspan=1 colspan=1>3.26e+5  130.22   2.123</td><td rowspan=1 colspan=1>2.35e+5  105.72   0.017</td><td rowspan=2 colspan=1>2.40e+5  107.54  52.350</td></tr><tr><td rowspan=1 colspan=1>3.77e+5  141.53   0.109</td><td rowspan=1 colspan=1>2.33e+5  105.22  0.0039</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 10<sub>:</sub> C<sub>ompar</sub>i<sub>son o</sub>f M<sub>o</sub>d<sub>e</sub>l<sub>s w</sub>ith dif<sub>eren</sub>t N<sub>orma</sub>li<sub>za</sub>ti<sub>on</sub> T<sub>ec</sub>h<sub>n</sub>i<sub>ques.</sub> L<sub>ower num</sub>b<sub>ers</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> l<sub>ower errors an</sub>d b<sub>e</sub>tt<sub>er</sub> <sub>p</sub>er<sup>f</sup>ormances. B<sub>es</sub>t <sub>me</sub>t<sub>r</sub>i<sub>cs</sub> i<sub>n eac</sub>h <sub>ca</sub>t<sub>egory are co</sub>l<sub>ore</sub>d <sub>w</sub>ith R<sub>e</sub>d <sub>an</sub>d S<sub>econ</sub>d b<sub>es</sub>t <sub>me</sub>t<sub>r</sub>i<sub>cs are co</sub>l<sub>ore</sub>d <sub>w</sub>ith Bl<sub>ue</sub>