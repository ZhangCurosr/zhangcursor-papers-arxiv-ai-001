# Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

XIAOBO CHEN, University of Massachusetts Amherst, USA

GENLIANG ZHU, Accentrust, Canada and Georgia Institute of Technology, USA

ZIYUAN WANG, Duke University, USA

CHU WANG, Accentrust, Canada and University of Illinois Urbana-Champaign, USA

After a runtime or API mi<sub>g</sub>ration<sub>,</sub> an LLM-re<sub>p</sub>aired <sub>p</sub>rocedure can run cleanl<sub>y y</sub>et return wron<sub>g</sub> results<sub>,</sub> and a release check on t<sub>yp</sub>ical test in<sub>p</sub>uts shi<sub>p</sub>s it. We re<sub>p</sub>ort a <sub>p</sub>rere<sub>g</sub>istered stud<sub>y</sub> of 73 tasks coverin<sub>g</sub> 60 documented changes in CPython, Node.js and SQLite. Every candidate is executed on the real runtimes against withheld contracts. Over the com<sub>p</sub>lete re<sub>p</sub>air s<sub>p</sub>ace<sub>,</sub> the all-assertions <sub>g</sub>ate on t<sub>yp</sub>ical in<sub>p</sub>uts released 0.410 of wron<sub>g</sub> re<sub>p</sub>airs. One ed<sub>g</sub>e in<sub>p</sub>ut written from the release note cut this to 0.065<sub>,</sub> and to 0.061 on 15 unseen clusters. Th<sub>ree</sub> i<sub>n u</sub>t<sub>s</sub> f<sub>rom a</sub> bli<sub>n</sub>d GPT<sub>-</sub>5<sub>.</sub>6<sub>-</sub>l<sub>una wr</sub>it<sub>er</sub> th<sub>a</sub>t <sub>never saw</sub> th<sub>e ro ram reac</sub>h<sub>e</sub>d 0<sub>.</sub>070<sub>.</sub> T<sub>wo</sub> f<sub>ur</sub>th<sub>er</sub> f<sub>rozen</sub> <sub>s</sub>t<sub>ages en</sub>l<sub>arge</sub>d th<sub>e menu</sub> t<sub>o</sub> 1491 f<sub>ragmen</sub>t<sub>s an</sub>d th<sub>en a</sub>dd<sub>e</sub>d 233 h<sub>ar</sub>d di<sub>s</sub>t<sub>rac</sub>t<sub>ors</sub> th<sub>a</sub>t <sub>ma</sub>t<sub>c</sub>h th<sub>e</sub> l<sub>egacy ou</sub>t<sub>pu</sub>t on t<sub>yp</sub>ical in<sub>p</sub>uts. The ed<sub>g</sub>e in<sub>p</sub>ut (0.058) and the frontier-model writer (0.058) <sub>p</sub>assed their <sub>p</sub>rere<sub>g</sub>istered tests. Y<sub>e</sub>t 0<sub>.</sub>168 <sub>an</sub>d 0<sub>.</sub>153 <sub>o</sub>f th<sub>e</sub> h<sub>ar</sub>d <sub>wrong</sub> <sub>programs</sub> <sub>s</sub>till <sub>passe</sub>d<sub>.</sub> T<sub>wo</sub> l<sub>oca</sub>l<sub>-mo</sub>d<sub>e</sub>l <sub>wr</sub>it<sub>ers</sub> f<sub>a</sub>il<sub>e</sub>d th<sub>e</sub>i<sub>r</sub> <sub>prereg</sub>i<sub>s</sub>t<sub>ere</sub>d t<sub>es</sub>t<sub>s.</sub> A<sub>c</sub>r<sub>oss</sub> f<sub>ou</sub>r m<sub>o</sub>d<sub>e</sub>l f<sub>a</sub>mili<sub>es,</sub> th<sub>e e</sub>d<sub>ge</sub> in<sub>pu</sub>t <sub>cu</sub>t <sub>w</sub>r<sub>o</sub>n<sub>g</sub> r<sub>e</sub>l<sub>eases pe</sub>r <sub>a</sub>tt<sub>e</sub>m<sub>p</sub>t in <sub>a</sub>ll thr<sub>ee ope</sub>n-<sub>we</sub>i<sub>g</sub>ht families on all three menus, for exam<sub>p</sub>le 0.233→0.057, without lowerin<sub>g</sub> correct com<sub>p</sub>letion. Self-tests hel<sub>p</sub>ed l<sub>ess.</sub> A blind AI <sub>au</sub>dit fl<sub>agge</sub>d 34 <sub>co</sub>ntr<sub>ac</sub>t<sub>s as s</sub>tri<sub>c</sub>t<sub>e</sub>r th<sub>a</sub>n th<sub>e</sub>ir r<sub>equ</sub>ir<sub>e</sub>m<sub>e</sub>nt<sub>s.</sub> Dr<sub>opp</sub>in<sub>g</sub> th<sub>e</sub> di<sub>spu</sub>t<sub>e</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> <sub>c</sub>h<sub>ange</sub>d 0 <sub>ver</sub>di<sub>c</sub>t<sub>s,</sub> b<sub>u</sub>t <sub>c</sub>l<sub>ass-</sub>l<sub>a</sub>b<sub>e</sub>l di<sub>spu</sub>t<sub>es rema</sub>i<sub>n open.</sub> N<sub>o mo</sub>d<sub>e</sub>l<sub>-wr</sub>itt<sub>en co</sub>d<sub>e was execu</sub>t<sub>e</sub>d<sub>.</sub> A<sub>r</sub>tif<sub>ac</sub>t<sub>s are</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> <sub>au</sub>th<sub>ors</sub> <sub>on</sub> <sub>reques</sub>t<sub>.</sub>

Additi<sub>o</sub>n<sub>a</sub>l K<sub>ey</sub> W<sub>o</sub>rd<sub>s</sub> <sub>a</sub>nd Phr<sub>ases</sub>: LLM <sub>p</sub>r<sub>og</sub>r<sub>a</sub>m r<sub>epa</sub>ir<sub>,</sub> API mi<sub>g</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> r<sub>u</sub>ntim<sub>e</sub> <sub>upg</sub>r<sub>a</sub>d<sub>e,</sub> r<sub>e</sub>l<sub>ease</sub> n<sub>o</sub>t<sub>es,</sub> ed<sub>g</sub>e-case test in<sub>p</sub>uts<sub>,</sub> test oracle ade<sub>q</sub>uac<sub>y,</sub> release <sub>g</sub>ate<sub>,</sub> LLM test <sub>g</sub>eneration<sub>,</sub> acce<sub>p</sub>ted-but-wron<sub>g</sub> re<sub>p</sub>airs silent behavioral breakin<sub>g</sub> chan<sub>g</sub>es<sub>, p</sub>atch overfittin<sub>g,</sub> diferential testin<sub>g, p</sub>rere<sub>g</sub>istered em<sub>p</sub>irical stud<sub>y,</sub> P<sub>y</sub>thon 3.12, Node.js 24, SQLite

## 1 Introduction

When the runtime or API under a business <sub>p</sub>rocedure evolves<sub>,</sub> an a<sub>pp</sub>roved mi<sub>g</sub>ration must re<sub>p</sub>air th<sub>e</sub> <sub>proce</sub>d<sub>ure</sub> <sub>so</sub> th<sub>a</sub>t it <sub>s</sub>till <sub>mee</sub>t<sub>s</sub> th<sub>e</sub> <sub>same</sub> <sub>o</sub>bli<sub>ga</sub>ti<sub>on.</sub> S<sub>ome</sub> b<sub>rea</sub>k<sub>ages</sub> <sub>are</sub> l<sub>ou</sub>d<sub>:</sub> <sub>a</sub> <sub>remove</sub>d f<sub>unc</sub>ti<sub>on</sub> raises an error. Others are <sub>q</sub>uiet: ever<sub>y</sub> call succeeds<sub>,</sub> <sub>y</sub>et the result chan<sub>g</sub>es. For exam<sub>p</sub>le:

• an enumeration renders as 3 instead of Priority.HIGH;

• sum() returns a more accurate float that no lon<sub>g</sub>er matches stored led<sub>g</sub>er totals;

• Buffer.poolSize <sub>g</sub>rows ei<sub>g</sub>htfold and silentl<sub>y</sub> chan<sub>g</sub>es the chunkin<sub>g</sub> of ever<sub>y</sub> u<sub>p</sub>load.

A valid call can still have an invalid efect [19]. A<sub>g</sub>ents notice <sub>p</sub>lausible wron<sub>g</sub> values far less often than ex<sub>p</sub>licit errors [17]. In <sub>p</sub>ro<sub>g</sub>ram re<sub>p</sub>air, weak tests let incorrect <sub>p</sub>atches throu<sub>g</sub>h: the overfittin<sub>g</sub> <sub>p</sub>roblem [31, 35, 39].

Our earlier study (Section 6) found that a pre-release check fails mainly through its inputs. Checks <sub>on</sub> th<sub>e</sub> <sub>o</sub>bli<sub>ga</sub>ti<sub>on</sub> <sub>re</sub>l<sub>ease</sub>d f<sub>ew</sub> <sub>wrong</sub> <sub>repa</sub>i<sub>rs</sub> <sub>once</sub> <sub>one</sub> <sub>e</sub>d<sub>ge</sub> i<sub>npu</sub>t <sub>wr</sub>itt<sub>en</sub> f<sub>rom</sub> th<sub>e</sub> <sub>re</sub>l<sub>ease</sub> <sub>no</sub>t<sub>e</sub> was visi<sup>bl</sup>e. Reviewers raise<sup>d f</sup>our o<sup>b</sup>jections:

• the ed<sub>g</sub>e in<sub>p</sub>uts were author-written and close to the hidden in<sub>p</sub>uts;

• the re<sub>p</sub>air s<sub>p</sub>ace is a finite menu;

• the human contract si<sub>g</sub>n-of never ha<sub>pp</sub>ened;

• the test <sub>p</sub>artition had onl<sub>y</sub> 34 clusters.

Thi<sub>s paper answers</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>an</sub>d f<sub>our</sub>th <sub>w</sub>ith <sub>new, prereg</sub>i<sub>s</sub>t<sub>ere</sub>d <sub>exper</sub>i<sub>men</sub>t<sub>s.</sub> It <sub>answers</sub> th<sub>e</sub> thi<sub>r</sub>d onl<sub>y</sub> with an AI audit<sub>,</sub> not a human review. It answers the second onl<sub>y p</sub>artl<sub>y</sub>: a second sta<sub>g</sub>e <sub>en</sub>l<sub>arges</sub> th<sub>e menu</sub> t<sub>o a me</sub>di<sub>an o</sub>f 27 <sub>programs per</sub> t<sub>as</sub>k<sub>,</sub> b<sub>u</sub>t 25 t<sub>as</sub>k<sub>s s</sub>till h<sub>ave</sub> f<sub>ewer</sub> th<sub>an</sub> 15 <sub>op</sub>ti<sub>ons.</sub> R<sub>epa</sub>i<sub>rs rema</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ces among au</sub>th<sub>or-wr</sub>itt<sub>en</sub> f<sub>ragmen</sub>t<sub>s, an</sub>d th<sub>a</sub>t <sub>s</sub>t<sub>age s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e s</sub>h<sub>are o</sub>f wron<sub>g</sub> <sub>p</sub>ro<sub>g</sub>rams a <sub>g</sub>ate releases de<sub>p</sub>ends on how the menu is built (LM3).

Setting. We built 73 tasks with 66 afected tasks in 60 documented-change clusters, 18 of them <sub>new,</sub> <sub>on</sub> th<sub>ree</sub> i<sub>ns</sub>t<sub>a</sub>ll<sub>e</sub>d <sub>open-source</sub> li<sub>neages:</sub>

• CP<sub>y</sub>thon 3.9→3.12;

• Node.js 20→24;

• the SQLite shell 3.51→3.54.

Ever<sub>y</sub> chan<sub>g</sub>e is <sub>g</sub>rounded in <sub>p</sub>inned release-note text. Ever<sub>y</sub> candidate is executed on the real runtimes. Hidden contracts are never shown to an<sub>y</sub> model. We com<sub>p</sub>are the visible-in<sub>p</sub>ut sets that a re<sup>l</sup>ease <sub>g</sub>ate can use:

• t<sub>yp</sub>ical in<sub>p</sub>uts onl<sub>y</sub>;

• <sub>p</sub>lus the author’s release-note ed<sub>g</sub>e in<sub>p</sub>ut;

• plus up to three inputs from a blind writer, an isolated LLM that sees only the release note and th<sub>e v</sub>i<sub>s</sub>ibl<sub>e con</sub>t<sub>rac</sub>t<sub>;</sub>

• plus self-tests that the repairing model writes for itself before it repairs.

Fo<sub>u</sub>r model families re<sub>p</sub>air the <sub>p</sub>roced<sub>u</sub>res <sub>u</sub>nder e<sub>qu</sub>al information: GPT-5.6-l<sub>u</sub>na <sub>v</sub>ia the Codex CLI, Qwen3-8B, gpt-oss-20b and qwen3.6-35b-a3b. A fifth, gemma-4-12b, failed the unchanged <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>i<sub>on c</sub>h<sub>ec</sub>k <sub>an</sub>d i<sub>s repor</sub>t<sub>e</sub>d <sub>as no</sub>t <sub>a</sub>d<sub>m</sub>itt<sub>e</sub>d<sub>.</sub>

Contributions.

(1) Exhaustive gate adequacy, replicated on unseen changes. Over the complete repair space <sub>o</sub>f 52 <sub>a</sub>f<sub>ec</sub>t<sub>e</sub>d t<sub>es</sub>t t<sub>as</sub>k<sub>s:</sub>

• the all-assertions <sub>g</sub>ate on t<sub>yp</sub>ical in<sub>p</sub>uts released 0.410 of wron<sub>g</sub> re<sub>p</sub>airs, and one release-note <sub>e</sub>d<sub>ge</sub> i<sub>npu</sub>t l<sub>owere</sub>d thi<sub>s</sub> t<sub>o</sub> 0<sub>.</sub>065<sub>;</sub>

• on the 15 new test clusters, built after the earlier stud<sub>y</sub>, the dro<sub>p</sub> was 0.471→0.061 (Holm <sub>�</sub> 0.002);

• false rejection sta<sub>y</sub>ed at 0.010.

(2) A blind test writer comes close to the author (descriptive), and a preregistered guard fails.

• Three blind-writer in<sub>p</sub>uts <sub>p</sub>er task reached 0.070 (0.090 without the 3 in<sub>p</sub>uts that e<sub>q</sub>ual a hidden input), with no increase in false rejection. This result is descriptive: the unguarded curve was not a <sub>p</sub>rere<sub>g</sub>istered h<sub>yp</sub>othesis<sub>,</sub> it was known before an<sub>y</sub> re<sub>p</sub>air call<sub>,</sub> and there is one writer.

• The <sub>p</sub>rere<sub>g</sub>istered covera<sub>g</sub>e-aware <sub>g</sub>ate ke<sub>p</sub>t onl<sub>y</sub> in<sub>p</sub>uts the writer declared le<sub>g</sub>ac<sub>y</sub>-bindin<sub>g</sub>. It reached onl<sub>y</sub> 0.157<sub>,</sub> so its <sub>p</sub>rere<sub>g</sub>istered test is not <sub>p</sub>ositive.

• Writers mislabel chan<sub>g</sub>e-revealin<sub>g</sub> in<sub>p</sub>uts as non-bindin<sub>g</sub>: the label the <sub>g</sub>uard relies on is the l<sub>a</sub>b<sub>e</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>ge</sub>t <sub>wrong.</sub>

(3) Edge inputs replicate across model families; self-tests do not reach significance.

• The ed<sub>g</sub>e in<sub>p</sub>ut cut wron<sub>g</sub> releases <sub>p</sub>er attem<sub>p</sub>t in all three o<sub>p</sub>en-wei<sub>g</sub>ht families (Table 4), <sub>w</sub>ith <sub>no</sub> l<sub>oss o</sub>f <sub>correc</sub>t <sub>comp</sub>l<sub>e</sub>ti<sub>on.</sub>

• The <sub>p</sub>rere<sub>g</sub>istered re<sub>p</sub>lication criterion is met.

• The frontier model released few wron<sub>g</sub> re<sub>p</sub>airs in an<sub>y</sub> arm.

• Self-tests lowered wron<sub>g</sub> releases in ever<sub>y</sub> famil<sub>y</sub>, but no self-test contrast survived Holm correction. A <sub>p</sub>ost-hoc writer × re<sub>p</sub>airer matrix, restricted to the 21 tasks where ever<sub>y</sub> writer has tests<sub>,</sub> is consistent with writer ca<sub>p</sub>abilit<sub>y</sub> matterin<sub>g</sub> more than inde<sub>p</sub>endence. Tests from th<sub>e s</sub>t<sub>ronges</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>ga</sub>t<sub>e</sub>d b<sub>es</sub>t<sub>, an</sub>d <sub>same-mo</sub>d<sub>e</sub>l <sub>ce</sub>ll<sub>s were no</sub>t <sub>worse</sub> th<sub>an cross-mo</sub>d<sub>e</sub>l <sub>ce</sub>ll<sub>s.</sub> This is descri<sub>p</sub>tive<sub>,</sub> with four families and one <sub>p</sub>rom<sub>p</sub>t.

(4) A larger repair menu. A second frozen stage enlarged the menu from 405 to 1491 documented fra<sub>g</sub>ments. Over that s<sub>p</sub>ace the ed<sub>g</sub>e in<sub>p</sub>ut (0.045) and the blind writer (0.053) both <sub>p</sub>assed their <sub>p</sub>rere<sub>g</sub>istered tests. The ed<sub>g</sub>e-in<sub>p</sub>ut efect on model re<sub>p</sub>airs re<sub>p</sub>licated in the same three o<sub>p</sub>en-wei<sub>g</sub>ht families.

(5) Hard distractors and more writers. A third frozen stage added 233 wrong-looking-right distractors that match the le<sub>g</sub>ac<sub>y</sub> out<sub>p</sub>ut on t<sub>yp</sub>ical in<sub>p</sub>uts. The ed<sub>g</sub>e in<sub>p</sub>ut (0.058) and the GPT-5.6-luna writer (0.058) still <sub>p</sub>assed their <sub>p</sub>rere<sub>g</sub>istered tests, and the ed<sub>g</sub>e-in<sub>p</sub>ut efect <sub>rep</sub>li<sub>ca</sub>t<sub>e</sub>d i<sub>n</sub> <sub>a</sub>ll th<sub>ree</sub> <sub>open-we</sub>i<sub>g</sub>ht f<sub>am</sub>ili<sub>es.</sub> T<sub>wo</sub> <sub>wea</sub>k<sub>er</sub> bli<sub>n</sub>d <sub>wr</sub>it<sub>ers</sub> did <sub>no</sub>t <sub>pass.</sub>

(6) Independent AI contract audit.

• Two blind GPT-5.6-luna audits <sub>p</sub>er contract a<sub>g</sub>reed on 0.91 of items (� = 0.73).

• The<sub>y</sub> fla<sub>gg</sub>ed 34 of 73 contracts, mostl<sub>y</sub> because the file-<sub>p</sub>reservation check is stricter than th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>requ</sub>i<sub>remen</sub>t<sub>.</sub>

• Removin<sub>g</sub> that check, and le<sub>g</sub>ac<sub>y</sub> e<sub>q</sub>ualit<sub>y</sub> as well, chan<sub>g</sub>es 0 hidden verdicts; the class-label (R2) objections remain o<sub>p</sub>en.

• We re<sub>p</sub>ort this as an AI audit, not as a human si<sub>g</sub>n-of.

Scope. Repairs are choices among documented fragments. A free-form arm, which would execute <sub>mo</sub>d<sub>e</sub>l<sub>-wr</sub>itt<sub>en</sub> <sub>co</sub>d<sub>e,</sub> <sub>was</sub> <sub>approve</sub>d <sub>on</sub>l<sub>y</sub> <sub>un</sub>d<sub>er</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>san</sub>db<sub>ox</sub>i<sub>ng.</sub> Th<sub>e</sub> <sub>san</sub>db<sub>ox</sub> <sub>cou</sub>ld <sub>no</sub>t b<sub>e</sub> b<sub>u</sub>ilt within the <sub>p</sub>ermissions of the stud<sub>y</sub> machine, so the arm was not run (Section 8). No model-written <sub>co</sub>d<sub>e was execu</sub>t<sub>e</sub>d <sub>anyw</sub>h<sub>ere.</sub>

## 2 Related Work

## 2.1 Repairing clients after API and dependency evolution

Behavioural breakin<sub>g</sub> chan<sub>g</sub>es occur even without si<sub>g</sub>nature chan<sub>g</sub>es or major releases [14, 38, 43]. Mi<sub>g</sub>ration recommenders recover old-to-new API relations [9, 37], and web-service clients have been re<sub>p</sub>aired from mi<sub>g</sub>ration <sub>p</sub>atterns [28].

Th<sub>e</sub> <sub>c</sub>l<sub>oses</sub>t <sub>recen</sub>t <sub>wor</sub>k <sub>repa</sub>i<sub>rs</sub> <sub>c</sub>li<sub>en</sub>t <sub>co</sub>d<sub>e</sub> <sub>a</sub>ft<sub>er</sub> d<sub>epen</sub>d<sub>ency</sub> <sub>up</sub>d<sub>a</sub>t<sub>es:</sub>

• Fruntke and Krinke fix de<sub>p</sub>endenc<sub>y</sub> breakin<sub>g</sub> chan<sub>g</sub>es automaticall<sub>y</sub> [11].

• B<sub>y</sub>am combines com<sub>p</sub>ilation errors, error localisation and API diferences with an LLM to fix breakin<sub>g</sub> de<sub>p</sub>endenc<sub>y</sub> u<sub>p</sub>dates [27].

• De<sub>p</sub>Re<sub>p</sub>air observes that the decisive evidence often lies u<sub>p</sub>stream (release notes, API difs) and that no failin<sub>g</sub> test localises the break [41].

The closest is BBCFixer [5]. It targets behavioural dependency breaks (BBCBench: 100 Python and JavaScri t cases). It runs the failin test under the old and new librar versions, ranks the calls <sub>w</sub>h<sub>ose re</sub>t<sub>urn va</sub>l<sub>ues</sub> dif<sub>er</sub> t<sub>o</sub> l<sub>oca</sub>t<sub>e</sub> th<sub>e roo</sub>t API<sub>, an</sub>d filt<sub>ers</sub> th<sub>e</sub> lib<sub>rary</sub> dif <sub>accor</sub>di<sub>ng</sub>l<sub>y.</sub> Gil<sub>es</sub>i records API-interaction sna<sub>p</sub>shots durin<sub>g</sub> client tests and com<sub>p</sub>ares them across librar<sub>y</sub> versions to fla<sub>g</sub> behavioural breakin<sub>g</sub> chan<sub>g</sub>es [24]. In efect, it a<sub>pp</sub>lies diferential testin<sub>g</sub> at the API boundar<sub>y</sub>.

These s<sub>y</sub>stems acce<sub>p</sub>t a re<sub>p</sub>air when the client com<sub>p</sub>iles or its own tests <sub>p</sub>ass. BBCFixer is the exception: it uses old-versus-new execution as repairevidence, not as a release criterion. BBCBench is the closest benchmark: it contains 100 real behavioural de<sub>p</sub>endenc<sub>y</sub> breaks in P<sub>y</sub>thon and JavaScri<sub>p</sub>t projects, eac<sup>h</sup> wit<sup>h</sup> t<sup>h</sup>e <sup>f</sup>ai<sup>l</sup>ing c<sup>l</sup>ient test. We <sup>d</sup>i<sup>d</sup> not eva<sup>l</sup>uate our re<sup>l</sup>ease gates on it, <sup>f</sup>or t<sup>h</sup>ree reasons:

• its cases re<sub>q</sub>uire installin<sub>g</sub> <sub>p</sub>roject de<sub>p</sub>endencies, which our ofline, no-install <sub>p</sub>rotocol forbids;

• its oracle is the <sub>p</sub>roject’s own failin<sub>g</sub> test, so the oracle cannot be varied inde<sub>p</sub>endentl<sub>y</sub> of the re<sub>p</sub>a<sup>i</sup>r;

• its re<sub>p</sub>airs are free-form code, whose s<sub>p</sub>ace cannot be enumerated.

The two desi<sub>g</sub>ns are com<sub>p</sub>lementar<sub>y</sub>. BBCBench measures whether re<sub>p</sub>air succeeds a<sub>g</sub>ainst the project<sup>’</sup>s test. Our <sup>d</sup>esign as<sup>k</sup>s <sup>h</sup>ow o<sup>f</sup>ten a given re<sup>l</sup>ease c<sup>h</sup>ec<sup>k</sup> wou<sup>ld</sup> <sup>l</sup>et a wrong repair t<sup>h</sup>roug<sup>h</sup>, <sub>w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>res a</sub> fi<sub>n</sub>it<sub>e, enumera</sub>bl<sub>e repa</sub>i<sub>r space an</sub>d <sub>orac</sub>l<sub>es</sub> th<sub>a</sub>t <sub>can</sub> b<sub>e swappe</sub>d<sub>.</sub>

O<sub>u</sub>r in<sub>c</sub>r<sub>e</sub>m<sub>e</sub>nt r<sub>e</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> BBCFix<sub>e</sub>r<sub>,</sub> Gil<sub>es</sub>i <sub>a</sub>nd B<sub>ya</sub>m i<sub>s</sub> f<sub>ou</sub>rf<sub>o</sub>ld:

• we hold the re<sub>p</sub>airer’s information fixed and var<sub>y</sub> onl<sub>y</sub> the release <sub>g</sub>ate;

• we measure each <sub>g</sub>ate’s false-acce<sub>p</sub>tance and false-rejection rates over the com<sub>p</sub>lete s<sub>p</sub>ace of <sub>poss</sub>ibl<sub>e</sub> <sub>repa</sub>i<sub>rs</sub> <sub>an</sub>d <sub>over</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-pro</sub>d<sub>uce</sub>d <sub>repa</sub>i<sub>rs;</sub>

• we treat “no safe re<sub>p</sub>lacement exists” as a re<sub>g</sub>istered outcome;

• we com<sub>p</sub>are the sources of the in<sub>p</sub>uts that make a <sub>g</sub>ate ade<sub>q</sub>uate: author-written, a blind LLM <sub>wr</sub>it<sub>er, an</sub>d th<sub>e repa</sub>i<sub>r</sub>i<sub>ng mo</sub>d<sub>e</sub>l it<sub>se</sub>lf<sub>.</sub>

Th<sub>e</sub> <sub>re</sub>l<sub>ease</sub> <sub>ga</sub>t<sub>e</sub> i<sub>s</sub> <sub>measure</sub>d <sub>aga</sub>i<sub>ns</sub>t <sub>accep</sub>t<sub>ance</sub> <sub>an</sub>d <sub>preserva</sub>ti<sub>on</sub> <sub>con</sub>t<sub>rac</sub>t<sub>s</sub> th<sub>a</sub>t <sub>are</sub> <sub>w</sub>ithh<sub>e</sub>ld from the re<sub>p</sub>airer. B<sub>y</sub>am’s si<sub>g</sub>nal (com<sub>p</sub>ile errors <sub>p</sub>lus API difs) is the natural analo<sub>g</sub>ue of our runtime/schema <sub>g</sub>ate and dif view. We therefore do not <sub>p</sub>ort its Java-s<sub>p</sub>ecific <sub>p</sub>i<sub>p</sub>eline as a baseline; we com<sub>p</sub>are the <sub>g</sub>ate classes it relies on.

Version-aware generation <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup>s are a<sup>d</sup>jacent:

• u<sub>p</sub>datin<sub>g</sub> model knowled<sub>g</sub>e of API chan<sub>g</sub>es or detectin<sub>g</sub> de<sub>p</sub>recated-API usa<sub>g</sub>e [21, 33];

• <sub>g</sub>eneratin<sub>g</sub> code for a re<sub>q</sub>uested librar<sub>y</sub> version [18, 23];

• mi<sub>g</sub>ratin<sub>g</sub> re<sub>p</sub>ositories [20, 22].

T<sup>h</sup>ey eva<sup>l</sup>uate generation or w<sup>h</sup>o<sup>l</sup>e-project migration wit<sup>h</sup> tests as t<sup>h</sup>e orac<sup>l</sup>e. T<sup>h</sup>ey <sup>d</sup>o not as<sup>k h</sup>ow <sub>muc</sub>h <sub>a wea</sub>k<sub>er orac</sub>l<sub>e wou</sub>ld h<sub>ave</sub> l<sub>e</sub>t th<sub>roug</sub>h<sub>.</sub>

## 2.2 Tools that evolve under agents

• ToolEVO learns to use evolvin<sub>g</sub> tools from environment feedback and reflection [4].

• Continual documentation ada<sub>p</sub>tation revises tool descri<sub>p</sub>tions as tools chan<sub>g</sub>e [36].

• Drift benchmarks inject schema, <sub>p</sub>olic<sub>y</sub> and tool-name chan<sub>g</sub>es into stateful environments and score the final state [1, 25].

• D<sub>y</sub>namicMCPBench scores efects on live MCP servers [16].

• Kali<sub>y</sub>ev and Mar<sub>y</sub>ansk<sub>yy</sub> re<sub>p</sub>ort a <sub>g</sub>a<sub>p</sub> between in-session verification and held-out conformance for tools that a<sub>g</sub>ents s<sub>y</sub>nthesise themselves [15].

• ContractRL re<sub>p</sub>airs individual JSON tool calls with contract-masked minimal <sub>p</sub>atches and com<sub>p</sub>ares them with full re<sub>g</sub>eneration [13].

B<sub>enc</sub>h2R<sub>o</sub>b<sub>us</sub>t <sub>augmen</sub>t<sub>s</sub> <sub>agen</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> <sub>w</sub>ith <sub>recovera</sub>bl<sub>e</sub> <sub>an</sub>d <sub>unrecovera</sub>bl<sub>e</sub> f<sub>a</sub>il<sub>ures,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> drift and e<sub>p</sub>isodes in which no viable <sub>p</sub>ath remains [3]. Kraishan injects t<sub>yp</sub>ed tool faults and finds that a<sub>g</sub>ents fla<sub>g</sub> <sub>p</sub>lausible wron<sub>g</sub> values far less often than ex<sub>p</sub>licit errors [17]. Our settin<sub>g</sub> is dif<sub>eren</sub>t<sub>: ups</sub>t<sub>ream evo</sub>l<sub>u</sub>ti<sub>on un</sub>d<sub>er a</sub> fi<sub>xe</sub>d <sub>o</sub>bli<sub>ga</sub>ti<sub>on, w</sub>ith th<sub>e re</sub>l<sub>ease orac</sub>l<sub>e as</sub> th<sub>e man</sub>i<sub>pu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>e.</sub> W<sub>e</sub> d<sub>o no</sub>t t<sub>ra</sub>i<sub>n any mo</sub>d<sub>e</sub>l<sub>.</sub>

## 2.3 Oracles, contracts and patch overfiting

T<sub>es</sub>t<sub>-</sub>b<sub>ase</sub>d <sub>repa</sub>i<sub>r pro</sub>d<sub>uces pa</sub>t<sub>c</sub>h<sub>es</sub> th<sub>a</sub>t <sub>pass</sub> th<sub>e ava</sub>il<sub>a</sub>bl<sub>e</sub> t<sub>es</sub>t<sub>s ye</sub>t <sub>are</sub> i<sub>ncorrec</sub>t<sub>.</sub> Th<sub>e pro</sub>bl<sub>em</sub> is known as overfittin<sub>g</sub> [31], and it motivates <sub>p</sub>atch-correctness assessment be<sub>y</sub>ond the <sub>g</sub>iven tests [39]. It recurs for LLM a<sub>g</sub>ents, whose “solved” issues often are not [35, 42]. Our release <sub>g</sub>ates <sub>p</sub>l<sub>ay</sub> th<sub>e ro</sub>l<sub>e o</sub>f th<sub>e g</sub>i<sub>ven</sub> t<sub>es</sub>t<sub>s.</sub> Th<sub>e m</sub>i<sub>gra</sub>ti<sub>on se</sub>tti<sub>ng a</sub>dd<sub>s</sub> t<sub>wo</sub> thi<sub>ngs: a</sub> fi<sub>xe</sub>d l<sub>egacy</sub> b<sub>e</sub>h<sub>av</sub>i<sub>our</sub> th<sub>a</sub>t <sub>can serve as an orac</sub>l<sub>e, an</sub>d i<sub>npu</sub>t<sub>s w</sub>h<sub>ose re</sub>l<sub>evance</sub> i<sub>s name</sub>d b<sub>y</sub> th<sub>e c</sub>h<sub>ange no</sub>t<sub>e.</sub>

E<sub>xecu</sub>t<sub>a</sub>bl<sub>e</sub> <sub>proper</sub>ti<sub>es</sub> <sub>an</sub>d <sub>con</sub>t<sub>rac</sub>t<sub>s</sub> <sub>are</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>ways</sub> t<sub>o</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>our</sub> <sub>w</sub>h<sub>en</sub> <sub>a</sub> f<sub>u</sub>ll <sub>ou</sub>t<sub>pu</sub>t <sub>orac</sub>l<sub>e</sub> i<sub>s unava</sub>il<sub>a</sub>bl<sub>e:</sub>

• <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub>-based and metamor<sub>p</sub>hic testin<sub>g</sub> [8, 30];

• consumer-driven contract testin<sub>g</sub> [29];

• diferential re<sub>g</sub>ression testin<sub>g</sub> of API versions [12].

LLMs now write <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub>-based tests and other tests whose fault-revealin<sub>g</sub> <sub>p</sub>ower is assessed b<sub>y</sub> mutation anal<sub>y</sub>sis [10, 32, 34]. BreakGuard uses LLM-<sub>g</sub>enerated tests to detect behavioural breakin<sub>g</sub> chan<sub>g</sub>es in client code [26].

Generated tests as filters, and self-tests. CodeT generates tests with the same model and uses a<sub>g</sub>reement with them to rank sam<sub>p</sub>led <sub>p</sub>ro<sub>g</sub>rams [2]. Self-debu<sub>gg</sub>in<sub>g</sub> lets a model ex<sub>p</sub>lain and revise its own code from execution feedback [6]. Both treat model-written tests as a selection si<sub>g</sub>nal, <sub>an</sub>d b<sub>o</sub>th <sub>measure</sub> <sub>success</sub> <sub>aga</sub>i<sub>ns</sub>t h<sub>e</sub>ld<sub>-ou</sub>t t<sub>es</sub>t<sub>s.</sub> W<sub>e</sub> <sub>as</sub>k <sub>a</sub> <sub>narrower</sub> <sub>ques</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>ne</sub>ith<sub>er</sub> i<sub>so</sub>l<sub>a</sub>t<sub>es.</sub> When a model writes test inputs from a release note before repairing, how many of its own wrong re<sub>p</sub>airs does a <sub>g</sub>ate built on those in<sub>p</sub>uts still release? How does that com<sub>p</sub>are with an inde<sub>p</sub>endent blind writer and with an author-written ed<sub>g</sub>e in<sub>p</sub>ut? Our tests are data onl<sub>y</sub>. Ex<sub>p</sub>ected out<sub>p</sub>uts come f<sub>rom</sub> th<sub>e</sub> l<sub>egacy</sub> <sub>program</sub> <sub>on</sub> th<sub>e</sub> l<sub>egacy</sub> <sub>run</sub>ti<sub>me,</sub> <sub>so</sub> <sub>no</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-wr</sub>itt<sub>en</sub> <sub>co</sub>d<sub>e</sub> <sub>or</sub> <sub>orac</sub>l<sub>e</sub> i<sub>s</sub> <sub>execu</sub>t<sub>e</sub>d<sub>.</sub>

LLM judges as auditors. LLM judges agree with human preferences on open-ended answers but show <sub>p</sub>osition and self-<sub>p</sub>reference biases [44]. We use an LLM onl<sub>y</sub> to audit contracts a<sub>g</sub>ainst re<sup>l</sup>ease notes, wit<sup>h</sup> two <sup>bl</sup>in<sup>d</sup> rep<sup>l</sup>icates an<sup>d</sup> a <sup>f</sup>rozen a<sup>d</sup>ju<sup>d</sup>ication ru<sup>l</sup>e. We report it as an AI au<sup>d</sup>it, <sub>no</sub>t <sub>as a</sub> h<sub>uman rev</sub>i<sub>ew.</sub>

W<sub>e use</sub> th<sub>ese orac</sub>l<sub>e</sub> f<sub>am</sub>ili<sub>es as re</sub>l<sub>ease ga</sub>t<sub>es</sub> f<sub>or repa</sub>i<sub>rs.</sub> W<sub>e</sub> th<sub>en repor</sub>t<sub>,</sub> b<sub>o</sub>th <sub>over</sub> th<sub>e comp</sub>l<sub>e</sub>t<sub>e</sub> finite re<sub>p</sub>air s<sub>p</sub>ace and over model-<sub>p</sub>roduced re<sub>p</sub>airs<sub>,</sub> how man<sub>y</sub> wron<sub>g</sub> re<sub>p</sub>airs each <sub>g</sub>ate lets th<sub>roug</sub>h <sub>an</sub>d h<sub>ow many correc</sub>t <sub>ones</sub> it bl<sub>oc</sub>k<sub>s.</sub> Th<sub>e mo</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree measuremen</sub>t i<sub>s</sub> th<sub>e comp</sub>l<sub>emen</sub>t <sub>o</sub>f a mutation score com<sub>p</sub>uted over the s<sub>p</sub>ace of <sub>p</sub>ossible re<sub>p</sub>airs.

## 2.4 Edit formats and over-editing

Models over-edit when asked for minimal chan<sub>g</sub>es<sub>,</sub> and <sub>p</sub>reservation-aware trainin<sub>g</sub> or instructions reduce this [40, 45]. Dif and full-rewrite out<sub>p</sub>ut formats trade of fra<sub>g</sub>ilit<sub>y</sub> a<sub>g</sub>ainst cost [7].

O<sub>ur</sub> <sub>ear</sub>li<sub>er</sub> <sub>exp</sub>l<sub>ora</sub>t<sub>ory</sub> <sub>s</sub>t<sub>age</sub> <sub>s</sub>h<sub>owe</sub>d th<sub>a</sub>t <sub>a</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d t<sub>wo-</sub>fi<sub>e</sub>ld b<sub>u</sub>d<sub>ge</sub>t <sub>can</sub> <sub>manu</sub>f<sub>ac</sub>t<sub>ure</sub> <sub>a</sub> <sub>represen-</sub> tation efect. Here re<sub>p</sub>resentation is a secondar<sub>y,</sub> e<sub>q</sub>ual-information factor<sub>,</sub> the bud<sub>g</sub>et is unlimited<sub>,</sub> <sub>an</sub>d <sub>e</sub>dit <sub>s</sub>i<sub>ze</sub> i<sub>s recor</sub>d<sub>e</sub>d <sub>as an ou</sub>t<sub>come.</sub>

## 3 Problem and Terminology

A task has five parts:

• a consumer re<sub>q</sub>uirement $R ;$

• an old <sub>p</sub>rocedure $\scriptstyle { \mathcal { P } } 0$ th<sub>a</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> � <sub>on</sub> th<sub>e</sub> <sub>o</sub>ld <sub>run</sub>ti<sub>me</sub> $A _ { 0 }$ ;

• a finite re<sub>p</sub>air domain � (each <sub>p</sub>rocedure is a choice of documented fra<sub>g</sub>ments for the tem<sub>p</sub>late’s fields);

• a visible in<sub>p</sub>ut set $X _ { v } ;$

• a hidden in<sub>p</sub>ut set $X _ { h }$ , <sup>d</sup>isjoint <sup>f</sup>rom $X _ { v }$ <sub>excep</sub>t <sub>w</sub>h<sub>ere</sub> <sub>no</sub>t<sub>e</sub>d<sub>.</sub>

The business obli<sub>g</sub>ation sta<sub>y</sub>s fixed across the mi<sub>g</sub>ration.

Definition 1 (Hidden correctness). A procedure $p \in D$ is correct on the new runtime $A _ { 1 } \ i f ,$ for every $x \in X _ { h }$ , two conditions hold:

• its output satisfies the acceptance assertions derived from �, together with equality to $\scriptstyle { p _ { 0 } }$ on $A _ { 0 }$ for the outputs that � constrains;

• the files it leaves behind equal those left by �<sub>0</sub> (preservation).

Hidden contracts are never shown to a repairer.

Definition 2 (Release gate and outcomes). A release gate (or release oracle) maps a candidate’s observations on $X _ { v }$ under $A _ { 1 }$ to release or block. Each attempt ends in exactly one of four outcomes:

• correct completion: released and correct;

• false acceptance: released and not correct;

• false rejection: blocked although correct;

• safe non-release: blocked or abstained, and not correct.

A false acceptance is error-free when the hidden executions raised no error.

A no-safe-repair task is one in which no $\pmb { \mathscr { p } } \in D$ i<sub>s correc</sub>t<sub>; ex</sub>h<sub>aus</sub>ti<sub>ve searc</sub>h <sub>con</sub>fi<sub>rms</sub> thi<sub>s.</sub> F<sub>or</sub> such a task the correct dis<sub>p</sub>osition is non-release<sub>,</sub> ideall<sub>y</sub> an ex<sub>p</sub>licit abstention.

Th<sub>e</sub> “<sub>no sa</sub>f<sub>e repa</sub>i<sub>r</sub>” l<sub>a</sub>b<sub>e</sub>l d<sub>epen</sub>d<sub>s on</sub> th<sub>e o</sub>bli<sub>ga</sub>ti<sub>on.</sub> I<sub>n</sub> th<sub>e ear</sub>li<sub>er s</sub>t<sub>u</sub>d<sub>y,</sub> 4 <sub>o</sub>f it<sub>s no-sa</sub>f<sub>e-</sub> re<sub>p</sub>air tasks <sub>g</sub>ain correct <sub>p</sub>ro<sub>g</sub>rams once e<sub>q</sub>ualit<sub>y</sub> with the le<sub>g</sub>ac<sub>y</sub> out<sub>p</sub>ut is dro<sub>pp</sub>ed and onl<sub>y</sub> re<sub>q</sub>uirement-level assertions are ke<sub>p</sub>t: node-createcipher (4), py-binascii-hqx (5), py-distutils-looseversion (2), py-shuffle-random-param (5) correct <sub>p</sub>ro<sub>g</sub>rams (Section 6).

Th<sub>e ear</sub>li<sub>er s</sub>t<sub>u</sub>d<sub>y compare</sub>d f<sub>our ga</sub>t<sub>es.</sub> All <sub>excep</sub>t “<sub>none</sub>” <sub>execu</sub>t<sub>e</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e on</sub> $X _ { v }$ <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> h<sub>ave</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>execu</sub>ti<sub>on cos</sub>t<sub>:</sub>

• none;

• runtime/schema: the program exits normally and prints output of the declared shape and types;

• diferential: runtime/schema plus equality with $\scriptstyle { \mathcal { P } } 0$ on $A _ { 0 }$ for ever<sub>y</sub> visible in<sub>p</sub>ut<sub>;</sub>

• all consumer assertions: runtime/schema plus every requirement assertion on every visible input.

Visible-input coverage is the composition of $X _ { v } .$ . The gate on each set is all consumer assertions on <sub>eve</sub>r<sub>y</sub> in<sub>pu</sub>t in it:

• T (t<sub>yp</sub>ical): the task’s t<sub>yp</sub>ical in<sub>p</sub>ut or in<sub>p</sub>uts;

• E (ed<sub>g</sub>e in<sub>p</sub>ut): T <sub>p</sub>lus one author-written ed<sub>g</sub>e in<sub>p</sub>ut derived from the release note;

$W _ { k }$ (blind writer): T <sub>p</sub>lus the first $k \leq 3$ <sub>va</sub>lid i<sub>npu</sub>t<sub>s</sub> <sub>o</sub>f <sub>a</sub> bli<sub>n</sub>d LLM <sub>wr</sub>it<sub>er</sub> th<sub>a</sub>t <sub>sees</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>re</sub>l<sub>ease no</sub>t<sub>e an</sub>d th<sub>e v</sub>i<sub>s</sub>ibl<sub>e con</sub>t<sub>rac</sub>t<sub>;</sub>

$G _ { k }$ (<sub>g</sub>uarded writer): as $\mathrm { W } _ { k }$ , but only inputs the writer declares legacy-binding, that is, inputs on <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>o</sub>ld <sub>ou</sub>t<sub>pu</sub>t i<sub>s</sub> <sub>s</sub>till th<sub>e</sub> <sub>requ</sub>i<sub>re</sub>d <sub>ou</sub>t<sub>pu</sub>t<sub>;</sub>

• S (self-test): T <sub>p</sub>lus the re<sub>p</sub>airin<sub>g</sub> model’s own valid le<sub>g</sub>ac<sub>y</sub>-bindin<sub>g</sub> in<sub>p</sub>uts, written before it re<sub>p</sub>a<sup>i</sup>rs;

• ES: E plus S.

E<sub>xpec</sub>t<sub>e</sub>d <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> f<sub>or</sub> <sub>every</sub> <sub>a</sub>dd<sub>e</sub>d i<sub>npu</sub>t <sub>come</sub> f<sub>rom</sub> $\mathbf { \nabla } \mathcal { P } 0$ on $A _ { 0 }$ <sub>.</sub> N<sub>o</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-wr</sub>itt<sub>en</sub> <sub>co</sub>d<sub>e</sub> <sub>or</sub> <sub>expec</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ue</sub> i<sub>s ever execu</sub>t<sub>e</sub>d <sub>or</sub> t<sub>rus</sub>t<sub>e</sub>d<sub>.</sub>

Throughout, silent refers to the class of the change: calls succeed and the result changes. Error-free <sub>re</sub>f<sub>ers</sub> t<sub>o an ou</sub>t<sub>come.</sub>

## 4 Study Design

The desi<sub>g</sub>n was <sub>p</sub>rere<sub>g</sub>istered and frozen twice b<sub>y</sub> SHA-256 hash sets.

• F0 came before any model call of this study.

• F-test came before any repair, self-test or probe call.

E<sub>ac</sub>h f<sub>reeze</sub> <sub>s</sub>t<sub>ores</sub> <sub>a</sub> <sub>ver</sub>b<sub>a</sub>ti<sub>m</sub> <sub>copy</sub> <sub>o</sub>f th<sub>e</sub> d<sub>es</sub>i<sub>gn</sub> t<sub>ex</sub>t <sub>an</sub>d h<sub>as</sub>h<sub>es</sub> it<sub>s</sub> <sub>own</sub> f<sub>reeze</sub> <sub>scr</sub>i<sub>p</sub>t<sub>.</sub> Th<sub>ese</sub> <sub>are</sub> t<sub>wo</sub> l<sub>essons</sub> f<sub>rom</sub> th<sub>e</sub> <sub>ear</sub>li<sub>er</sub> <sub>s</sub>t<sub>u</sub>d<sub>y,</sub> <sub>w</sub>h<sub>ose</sub> F0 d<sub>es</sub>i<sub>gn</sub> t<sub>ex</sub>t h<sub>a</sub>d t<sub>o</sub> b<sub>e</sub> <sub>recons</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>erwar</sub>d<sub>s</sub> (A<sub>pp</sub>endix B).

## 4.1 Substrate

• Runtimes. Three installed lineages, executed ofline and unchanged since the earlier stud<sub>y</sub>: CP<sub>y</sub>thon 3.9.6→3.12.15, Node.js 20.20.2→24.21.0 and the SQLite command-line shell 3.51.0→3.54.0. SQLite 3.54.0 was released on 2026-10-09, after the models’ likel<sub>y</sub> trainin<sub>g</sub> cutof.

• Tasks. The 73 tasks are the 55 earlier tasks, imported unchanged, plus 18 new ones. The new t<sub>as</sub>k<sub>s</sub> <sub>are</sub> <sub>groun</sub>d<sub>e</sub>d i<sub>n</sub> <sub>ver</sub>b<sub>a</sub>ti<sub>m</sub> t<sub>ex</sub>t f<sub>rom</sub> <sub>a</sub>l<sub>rea</sub>d<sub>y-p</sub>i<sub>nne</sub>d <sub>o</sub>fi<sub>c</sub>i<sub>a</sub>l <sub>pages:</sub>

– Python What’s New 3.10–3.12;

– the Node.js deprecation registry and Bufer reference;

– the SQLite 3.53/3.54 release logs.

Exam<sub>p</sub>les are the random.seed t<sub>yp</sub>e restriction, NaN hashin<sub>g</sub> b<sub>y</sub> identit<sub>y</sub>, the PEP 701 tokenizer chan<sub>g</sub>e, the ipaddress is\_private fix, Buffer.poolSize, and SQLite’s ri<sub>g</sub>ht-justified numbers <sub>an</sub>d <sub>emp</sub>t<sub>y-resu</sub>lt h<sub>ea</sub>d<sub>ers.</sub>

• Counts. 66 tasks are afected, in 60 clusters (CPython 45, Node.js 13, SQLite 8). The other 7 are <sub>unc</sub>h<sub>ange</sub>d <sub>con</sub>t<sub>ro</sub>l<sub>s.</sub> B<sub>y</sub> <sub>c</sub>l<sub>ass</sub> th<sub>ere</sub> <sub>are</sub> 25 <sub>s</sub>il<sub>en</sub>t<sub>,</sub> 34 <sub>exp</sub>li<sub>c</sub>it <sub>an</sub>d 7 <sub>no-sa</sub>f<sub>e-repa</sub>i<sub>r</sub> t<sub>as</sub>k<sub>s.</sub>

• Authoring rules.

– Each new task passes the earlier study’s gates unchanged: quote verbatim, determinism, stale failure on hidden in<sub>p</sub>uts, at least one correct fra<sub>g</sub>ment choice (none for no-safe-re<sub>p</sub>air tasks), and static safet<sub>y</sub>.

– Each has one author edge input that must make the stale procedure fail a visible check: the edge-input QA rule.

– Two candidate tasks failed these gates in dry builds and were dropped before F0.

– No isomorphic variants were added.

• Split. Earlier clusters keep their earlier partition. New clusters go 20% per lineage to development, b<sub>y</sub> h<sub>as</sub>h<sub>.</sub> Th<sub>e</sub> t<sub>es</sub>t <sub>par</sub>titi<sub>on</sub> h<sub>as</sub> 52 <sub>a</sub>f<sub>ec</sub>t<sub>e</sub>d t<sub>as</sub>k<sub>s</sub> i<sub>n</sub> 49 <sub>c</sub>l<sub>us</sub>t<sub>ers,</sub> 15 <sub>o</sub>f th<sub>em new.</sub>

## 4.2 Visible inputs and gates

## Blind writer.

• Calls. One isolated GPT-5.6-luna call per task (73 calls) through the hardened Codex client:

– all MCP servers, plugins, apps and web search are disabled;

– the sandbox is read-only, ephemeral, with an empty working directory;

– an output schema is enforced;

– any use of the model’s own tools voids the call.

• What the writer saw. The requirement, the pinned excerpts, the input keys, the typical input <sub>w</sub>ith it<sub>s</sub> <sub>o</sub>ld <sub>ou</sub>t<sub>pu</sub>t<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>ou</sub>t<sub>pu</sub>t <sub>c</sub>h<sub>ec</sub>k<sub>s.</sub> It <sub>never</sub> <sub>saw</sub> th<sub>e</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e,</sub> th<sub>e</sub> f<sub>ragmen</sub>t<sub>s,</sub> th<sub>e</sub> <sub>o</sub>ld <sub>p</sub>ro<sub>g</sub>ram<sub>,</sub> the hidden in<sub>p</sub>uts or an<sub>y</sub> result.

• Output. Up to three inputs, each with a legacy-binding flag.

• Validation (fail-closed, data only). An input must have exactly the typical keys and value types. It <sub>mus</sub>t <sub>a</sub>l<sub>so</sub> <sub>run</sub> <sub>c</sub>l<sub>ean</sub>l<sub>y</sub> <sub>an</sub>d d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>o</sub>ld <sub>proce</sub>d<sub>ure</sub> <sub>on</sub> th<sub>e</sub> <sub>o</sub>ld <sub>run</sub>ti<sub>me,</sub> <sub>w</sub>hi<sub>c</sub>h su<sub>pp</sub><sup>li</sup>es t<sup>h</sup>e ex<sub>p</sub>ecte<sup>d</sup> out<sub>p</sub>ut.

• Yield. 176 inputs were kept in 69 tasks, and 35 of them were flagged non-binding. 27 were d<sub>roppe</sub>d<sub>:</sub> 13 b<sub>ecause</sub> th<sub>e</sub> <sub>o</sub>ld <sub>run</sub>ti<sub>me</sub> <sub>ra</sub>i<sub>se</sub>d <sub>an</sub> <sub>error,</sub> 12 <sub>as</sub> d<sub>up</sub>li<sub>ca</sub>t<sub>es,</sub> <sub>an</sub>d 2 <sub>as</sub> <sub>non</sub>d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c.</sub> 3 ke<sub>p</sub>t in<sub>p</sub>uts e<sub>q</sub>ual a hidden in<sub>p</sub>ut<sub>;</sub> the<sub>y</sub> are ke<sub>p</sub>t<sub>,</sub> as <sub>p</sub>rere<sub>g</sub>istered.

Coverage-aware gate. $\mathrm { W } _ { k }$ <sub>a</sub>dd<sub>s</sub> th<sub>e</sub> fi<sub>rs</sub>t � k<sub>ep</sub>t i<sub>npu</sub>t<sub>s.</sub> Th<sub>e prereg</sub>i<sub>s</sub>t<sub>ere</sub>d <sub>ga</sub>t<sub>e</sub> ${ \mathrm { G } } _ { k }$ <sub>a</sub>dd<sub>s</sub> th<sub>e</sub> fi<sub>rs</sub>t � ke<sub>p</sub>t in<sub>p</sub>uts fla<sub>gg</sub>ed le<sub>g</sub>ac<sub>y</sub>-bindin<sub>g</sub>. The <sub>g</sub>uard was motivated b<sub>y</sub> the earlier stud<sub>y</sub>: there<sub>,</sub> LLM inputs over-rejecte<sup>d</sup> on runtime <sup>f</sup>acts t<sup>h</sup>at no repair may preserve, suc<sup>h</sup> as a mo<sup>d</sup>u<sup>l</sup>e t<sup>h</sup>e migration <sub>remove</sub>d<sub>.</sub>

Self-tests. Inside each family’s run, the repairing model first writes its own inputs with the same <sub>p</sub>rom<sub>p</sub>t and validation (ste<sub>p</sub> 0, one call <sub>p</sub>er task). The S arm uses those fla<sub>gg</sub>ed le<sub>g</sub>ac<sub>y</sub>-bindin<sub>g</sub>. The <sub>unguar</sub>d<sub>e</sub>d <sub>se</sub>t i<sub>s</sub> <sub>ana</sub>l<sub>yse</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>ca</sub>ll<sub>s.</sub>

## 4.3 Repair protocol, arms and models

• Equal information. The repair prompt, parser and feedback are the earlier study’s, unchanged. <sup>E</sup>ver<sub>y</sub> arm sees:

– the requirement, the excerpts and the template;

– the field domains and the old procedure;

– the arm’s visible inputs;

– “EDIT BUDGET: unlimited” and permission to abstain.

• Arms and cells. There are four arms: T, S, E and ES. Each task and arm has eight cells, crossing <sub>pa</sub>t<sub>c</sub>h <sub>or</sub> f<sub>u</sub>ll <sub>represen</sub>t<sub>a</sub>ti<sub>on,</sub> dif <sub>v</sub>i<sub>ew</sub> <sub>o</sub>f <sub>or</sub> <sub>on,</sub> <sub>an</sub>d li<sub>s</sub>t<sub>e</sub>d <sub>or</sub> <sub>reverse</sub>d f<sub>ragmen</sub>t <sub>or</sub>d<sub>er.</sub>

• Steps. Step 1 is shared by two branches, no gate and all consumer assertions. The gated branch issues one revision if and onl<sub>y</sub> if its <sub>g</sub>ate fails<sub>,</sub> with <sub>p</sub>ass/FAIL feedback and no ex<sub>p</sub>ected values.

• Ofline gates. Every gate $\mathrm { W } _ { 1 } \mathrm { - } \mathrm { W } _ { 3 } , \mathrm { G } _ { 1 } \mathrm { - } \mathrm { G } _ { 3 }$ and S/Sraw is also a<sub>pp</sub>lied ofline to each T-arm <sub>s</sub>t<sub>ep-</sub>1 <sub>can</sub>did<sub>a</sub>t<sub>e.</sub>

• Models.

– GPT-5.6-luna via the Codex CLI (reasonin<sub>g</sub> efort hi<sub>g</sub>h);

– Qwen3-8B (4-bit, MLX, greedy);

– gpt-oss-20b, qwen3.6-35b-a3b with thinking of, and gemma-4-12b-qat, all in LM Studio at <sup>tem</sup>p<sup>erature 0.</sup>

• Admission. Local families must first pass the unchanged 12-call qualification: 12/12 exact out<sub>p</sub>uts, <sub>p</sub>5 <sub>p</sub>refill ≥500 tok/s, <sub>p</sub>5 decode ≥32 tok/s, and overhead ≤0.5 s.

– Qwen3-8B (12/12, decode 38.9), <sub>gp</sub>t-oss-20b (12/12, 61.4) and <sub>q</sub>wen3.6-35b-a3b (12/12, 104.0) <sub>were a</sub>d<sub>m</sub>itt<sub>e</sub>d <sub>a</sub>t th<sub>e</sub> fi<sub>rs</sub>t <sub>a</sub>tt<sub>emp</sub>t<sub>.</sub>

– gemma-4-12b-qat was not admitted: 0/12 exact, decode 20.7. Its answers began with leaked reasonin<sub>g</sub>-channel text (<|channel>thought), a servin<sub>g</sub>-confi<sub>g</sub>uration <sub>p</sub>roblem rather than <sub>ev</sub>id<sub>ence</sub> <sub>a</sub>b<sub>ou</sub>t th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>repa</sub>i<sub>r</sub> <sub>a</sub>bilit<sub>y.</sub> Th<sub>e</sub> <sub>spee</sub>d<sub>-on</sub>l<sub>y</sub> <sub>re</sub>t<sub>ry</sub> <sub>po</sub>li<sub>cy</sub> d<sub>oes</sub> <sub>no</sub>t <sub>cover</sub> <sub>con</sub>t<sub>en</sub>t f<sub>a</sub>il<sub>ures, so</sub> th<sub>e</sub> f<sub>am</sub>il<sub>y was no</sub>t <sub>re</sub>t<sub>r</sub>i<sub>e</sub>d<sub>, an</sub>d <sub>we</sub> d<sub>raw no conc</sub>l<sub>us</sub>i<sub>on a</sub>b<sub>ou</sub>t <sub>gemma.</sub>

• Calls. GPT-5.6-luna made 2626 calls, Qwen3-8B 3266, gpt-oss-20b 2965 and qwen3.6-35b-a3b 2885<sub>.</sub> N<sub>one was vo</sub>id<sub>e</sub>d<sub>.</sub>

• Truncations. Answers cut of at the output-token limit are never parsed and count as rejected replies: 6 for Qwen3-8B, 7 for gpt-oss-20b and 1 for qwen3.6-35b-a3b. The Codex CLI reports no fi<sub>n</sub>i<sub>s</sub>h <sub>reason.</sub>

• Contamination probe. Each model answered one step-1 prompt per afected task with the <sub>re</sub>l<sub>ease-no</sub>t<sub>e</sub> <sub>excerp</sub>t<sub>s</sub> <sub>an</sub>d dif <sub>v</sub>i<sub>ew</sub> <sub>remove</sub>d<sub>.</sub>

## 4.4 Hypotheses, decision rules and power

St<sub>a</sub>ti<sub>s</sub>ti<sub>cs are</sub> th<sub>e ear</sub>li<sub>er s</sub>t<sub>u</sub>d<sub>y</sub>’<sub>s, unc</sub>h<sub>ange</sub>d<sub>:</sub>

• the unit is the documented-chan<sub>g</sub>e cluster, with tasks avera<sub>g</sub>ed within clusters;

• si<sub>g</sub>n-fli<sub>p</sub> <sub>p</sub>ermutation <sub>�</sub>-values, exact for ≤16 clusters and otherwise 200,000 Monte Carlo fli<sub>p</sub>s;

• whole-cluster bootstra<sub>p</sub> 95% intervals;

• Holm adjustment within each famil<sub>y</sub>.

Rule thresholds use task means. Holm and zero discordance. Our frozen analysis code gave a h<sub>ypo</sub>th<sub>es</sub>i<sub>s w</sub>ith<sub>ou</sub>t di<sub>scor</sub>d<sub>an</sub>t <sub>c</sub>l<sub>us</sub>t<sub>ers no �-va</sub>l<sub>ue an</sub>d l<sub>e</sub>ft it <sub>ou</sub>t <sub>o</sub>f th<sub>e</sub> H<sub>o</sub>l<sub>m coun</sub>t<sub>.</sub> Th<sub>e</sub> f<sub>rozen</sub> d<sub>es</sub>i<sub>gn</sub> t<sub>ex</sub>t <sub>says</sub> “H<sub>o</sub>l<sub>m</sub> <sub>over</sub> 3” <sub>an</sub>d “<sub>over</sub> 4”<sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>repor</sub>t th<sub>e</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>va</sub>l<sub>ues,</sub> <sub>w</sub>ith <sub>every</sub> h<sub>yp</sub>othesis in � and a zero-discordance h<sub>yp</sub>othesis enterin<sub>g</sub> with <sub>�</sub> = 1: M4c then has Holm <sub>�</sub> 0.002 (code: <0.001) and <sub>gp</sub>t-oss-20b’s H2 0.132 (code: 0.088). No decision chan<sub>g</sub>es; tables show the <sub>co</sub>d<sub>e</sub>’<sub>s va</sub>l<sub>ues.</sub>

Family M4 (model-free, complete repair space, Holm over 3).

$\mathrm { \mathbf { M 4 a } } , \mathrm { F A ( G _ { 3 } ) - F A ( T ) }$ <sub>.</sub> P<sub>os</sub>iti<sub>ve</sub> if H<sub>o</sub>lm $p < 0 . 0 5 , \mathrm { F A } ( \mathrm { G } _ { 3 } ) \leq 0 . 1 0$ an<sup>d</sup> t<sup>h</sup>e rise in <sup>f</sup>a<sup>l</sup>se rejection is $\leq 0 . 0 5 .$

$\mathbf { M 4 b } , \mathrm { F R } ( \mathrm { G } _ { 3 } ) { \mathrm { - F R } } ( \mathrm { W } _ { 3 } )$ . Positive i<sup>f</sup> t<sup>h</sup>e guar<sup>d</sup> signi<sup>fi</sup>cant<sup>l</sup>y <sup>l</sup>owers <sup>f</sup>a<sup>l</sup>se rejection.

• M4c, FA(E)−FA(T) on the 15 new test clusters onl<sub>y</sub>. This is a fresh re<sub>p</sub>lication of the earlier HM1 <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> <sub>w</sub>ith th<sub>e</sub> <sub>same</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> <sub>as</sub> M4<sub>a.</sub>

## Family A4 (per model, Holm over 4; per-attempt rates averaged over the eight cells).

• H1: FA(E)−FA(T);

• H2: FA(S)−FA(T);

• H3: FA(ES)−FA(E);

• H4: CC(S)−CC(T).

H1–H3 are positive ifHolm $\textstyle p < 0 . 0 5$ <sub>,</sub> the diference is ne<sub>g</sub>ative<sub>,</sub> and the matchin<sub>g</sub> correct-com<sub>p</sub>letion dif<sub>erence</sub> i<sub>s</sub> $\ge - 0 . 0 5$

## Power and confirmatory status.

• A <sub>p</sub>rere<sub>g</sub>istered simulation resam<sub>p</sub>led the earlier <sub>p</sub>er-cluster rates with ei<sub>g</sub>ht attem<sub>p</sub>ts and the <sub>coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>e</sub>f<sub>ec</sub>t $r = 0 . 3$ <sub>.</sub> Th<sub>e</sub> f<sub>rozen</sub> d<sub>es</sub>i<sub>gn</sub> t<sub>ex</sub>t <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>e</sub> 400<sub>-s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on p</sub>il<sub>o</sub>t <sub>va</sub>l<sub>ues</sub> 0<sub>.</sub>99<sub>,</sub> 0.91 and 0.03 for Qwen3-8B, gpt-oss-20b and GPT-5.6-luna; the confirmatory status follows th<sub>ese.</sub> Th<sub>e</sub> 1<sub>,</sub>000<sub>-s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>recor</sub>d <sub>o</sub>f th<sub>e</sub> <sub>same</sub> <sub>scr</sub>i<sub>p</sub>t <sub>g</sub>i<sub>ves</sub> 0<sub>.</sub>99<sub>,</sub> 0<sub>.</sub>94 <sub>an</sub>d 0<sub>.</sub>06<sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t t<sub>wo</sub> f<sub>am</sub>ili<sub>es</sub> <sub>are</sub> <sub>con</sub>fi<sub>rma</sub>t<sub>ory;</sub> th<sub>e</sub> f<sub>ron</sub>ti<sub>er</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>near</sub> th<sub>e</sub> <sub>ce</sub>ili<sub>ng,</sub> i<sub>s</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ve.</sub>

• The two new families were confirmator<sub>y</sub> onl<sub>y</sub> if their T-arm false-acce<sub>p</sub>tance rate on the development partition was at least 0.15. qwen3.6-35b-a3b had 0.148 and is therefore descriptive.

• The 1,000-simulation <sub>p</sub>ower record was <sub>p</sub>roduced after F-test from the frozen scri<sub>p</sub>t. This is lo<sub>gg</sub>ed as a deviation.

## Cross-family criteria.

• Ed<sub>g</sub>e-in<sub>p</sub>ut re<sub>p</sub>lication: H1 <sub>p</sub>ositive in at least three families and in ever<sub>y</sub> confirmator<sub>y</sub> famil<sub>y</sub>.

• Self-test re<sub>p</sub>lication: H2 <sub>p</sub>ositive in at least three families.

• Model-scale anal<sub>y</sub>sis (descri<sub>p</sub>tive): S<sub>p</sub>earman <sub>�</sub> between ca<sub>p</sub>abilit<sub>y</sub> (T-arm correct com<sub>p</sub>letion) <sub>an</sub>d th<sub>e</sub> H1 <sub>e</sub>f<sub>ec</sub>t<sub>,</sub> <sub>w</sub>ith <sub>an</sub> <sub>exac</sub>t <sub>permu</sub>t<sub>a</sub>ti<sub>on</sub> <sub>�.</sub>

• The thresholds of three of five families were fixed when five families were <sub>p</sub>lanned. With <sub>g</sub>emma <sub>no</sub>t <sub>a</sub>d<sub>m</sub>itt<sub>e</sub>d<sub>,</sub> th<sub>ey</sub> <sub>are</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> f<sub>our.</sub>

## AI contract audit.

• Calls. Two blind GPT-5.6-luna calls per task (146 calls).

• What the auditor saw. The requirement, the excerpts, the template, the fragments, the old <sub>proce</sub>d<sub>ure</sub> <sub>an</sub>d th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>me</sub>d <sub>c</sub>l<sub>ass.</sub> It <sub>a</sub>l<sub>so</sub> <sub>saw</sub> th<sub>e</sub> <sub>con</sub>t<sub>rac</sub>t <sub>ru</sub>l<sub>es</sub> i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e</sub> t<sub>yp</sub>i<sub>ca</sub>l i<sub>npu</sub>t<sub>.</sub> It never saw hidden in<sub>p</sub>uts<sub>,</sub> out<sub>p</sub>uts or results.

## • Rubric.

– R1: the material documents the change;

– R2: the class is right;

– R3: legacy equality is a reasonable reading;

– R4: no assertion is stricter than the requirement;

– R5: the contract is complete.

• Adjudication. An item counts as a “no” only if both replicates say no. Contracts are never <sub>c</sub>h<sub>ange</sub>d <sub>a</sub>ft<sub>er</sub> th<sub>e au</sub>dit<sub>.</sub>

## 4.5 Second stage: a large repair menu

T<sup>h</sup>e main externa<sup>l</sup>-va<sup>l</sup>i<sup>d</sup>ity o<sup>b</sup>jection is t<sup>h</sup>e <sup>fi</sup>nite menu: wit<sup>h</sup> a<sup>b</sup>out <sup>fi</sup>ve <sup>f</sup>ragments per <sup>fi</sup>e<sup>ld</sup>, a model can often eliminate wron<sub>g</sub> o<sub>p</sub>tions b<sub>y</sub> ins<sub>p</sub>ection. A second, se<sub>p</sub>aratel<sub>y</sub> frozen sta<sub>g</sub>e (F0 2c95616bd805, F-test db2f355dba79) therefore enlar<sub>g</sub>es ever<sub>y</sub> field domain.

• Fragments. The menu grows from the 405 earlier fragments to 1491. Every added fragment <sub>comes</sub> f<sub>rom</sub> <sub>an</sub> <sub>ear</sub>li<sub>er</sub> f<sub>ragmen</sub>t b<sub>y</sub> <sub>one</sub> <sub>or</sub> t<sub>wo</sub> <sub>au</sub>th<sub>or-wr</sub>itt<sub>en</sub> <sub>opera</sub>t<sub>ors:</sub>

– a documented sibling of a module attribute (for example another inspect or crypto function);

– a swap among documented builtins, string methods, operators, SQLite functions or CLI <sub>mo</sub>d<sub>es;</sub>

– an integer-literal edit.

Ever<sub>y</sub> o<sub>p</sub>erator is checked a<sub>g</sub>ainst one of 63 newl<sub>y p</sub>inned oficial <sub>p</sub>a<sub>g</sub>es. Siblin<sub>g</sub>s that could act <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> <sub>san</sub>db<sub>ox,</sub> <sub>s</sub>t<sub>ar</sub>t <sub>processes</sub> <sub>or</sub> i<sub>mpor</sub>t b<sub>y</sub> <sub>name</sub> <sub>are</sub> <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d<sub>.</sub>

• Filters. A fragment is kept only if the program runs cleanly on the typical input on either runtime (a visible-onl<sub>y</sub> filter). The ste<sub>p</sub>-1 <sub>p</sub>rom<sub>p</sub>t must sta<sub>y</sub> within the frozen 2,048-token in<sub>p</sub>ut <sup>ca</sup>p<sup>.</sup>

• Coverage. 25 tasks still have fewer than 15 options, because few documented operators apply t<sub>o</sub> th<sub>e</sub>i<sub>r</sub> f<sub>ragmen</sub>t<sub>s.</sub> All 1986 <sub>proce</sub>d<sub>ures were eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>ex</sub>h<sub>aus</sub>ti<sub>ve</sub>l<sub>y.</sub>

• Runs. Arms T and E, eight cells each, with the same models, prompts and gates. The writer’s inputs and each family’s own self-tests are applied ofline. GPT-5.6-luna made 1242 calls, Qwen3- 8B 1599<sub>, gp</sub>t<sub>-oss-</sub>20b 1403 <sub>an</sub>d <sub>qwen</sub>3<sub>.</sub>6<sub>-</sub>35b<sub>-a</sub>3b 1387<sub>.</sub> E<sub>ac</sub>h l<sub>oca</sub>l f<sub>am</sub>il<sub>y passe</sub>d th<sub>e unc</sub>h<sub>ange</sub>d qualification again at the first attempt: Qwen3-8B 12/12, decode 40.7; gpt-oss-20b 12/12, 55.9; <sub>qwen</sub>3<sub>.</sub>6<sub>-</sub>35b<sub>-a</sub>3b 12/12<sub>,</sub> 74<sub>.</sub>3<sub>.</sub> T<sub>runca</sub>t<sub>e</sub>d <sub>answers, never parse</sub>d<sub>:</sub> 5<sub>,</sub> 8 <sub>an</sub>d 3<sub>.</sub>

• Hypotheses (strict Holm over 3 per family, fixed at F0).

– Model-free: LM1 FA(E)−FA(T); LM2 FA(W<sub>3</sub>)−FA(T), which makes the earlier descri<sub>p</sub>tive writer result a <sub>p</sub>rere<sub>g</sub>istered test; LM3 FA(T) on the lar<sub>g</sub>e menu minus FA(T) on the earlier menu.

– Per famil<sub>y</sub>: LH1 wron<sub>g</sub> releases <sub>p</sub>er attem<sub>p</sub>t E−T; LH2 and LH3, the chan<sub>g</sub>e in wron<sub>g</sub> releases and in correct com<sub>p</sub>letion with t<sub>yp</sub>ical in<sub>p</sub>uts relative to the earlier menu.

– Thresholds and confirmatory status are as before. The model-free family was computed b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> f<sub>reezes,</sub> b<sub>e</sub>f<sub>ore</sub> <sub>any</sub> <sub>mo</sub>d<sub>e</sub>l <sub>ca</sub>ll<sub>.</sub>

## 4.6 Third stage: hard distractors and two more writers

A third frozen sta<sub>g</sub>e (F0 bf681a0b7291, F-test 5f4ca8417951) addresses the menu-com<sub>p</sub>osition <sub>p</sub>roblem that LM3 ex<sub>p</sub>osed (Section 5.3).

• Hard distractors. The same documented operators generate candidates from every large-menu fragment. A candidate is kept as a hard distractor only if, on the typical inputs, the program’s <sub>ou</sub>t<sub>pu</sub>t <sub>on</sub> th<sub>e new run</sub>ti<sub>me equa</sub>l<sub>s</sub> th<sub>e o</sub>ld <sub>program</sub>’<sub>s ou</sub>t<sub>pu</sub>t <sub>on</sub> th<sub>e o</sub>ld <sub>run</sub>ti<sub>me.</sub> Thi<sub>s</sub> i<sub>s v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on on</sub>l<sub>y:</sub> th<sub>e</sub> f<sub>ragmen</sub>t <sub>passes</sub> th<sub>e v</sub>i<sub>s</sub>ibl<sub>e</sub> dif<sub>eren</sub>ti<sub>a</sub>l <sub>c</sub>h<sub>ec</sub>k<sub>.</sub>

– 233 hard fragments were added in 26 tasks, giving 1689 fragments and 2310 procedures, all <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>ex</sub>h<sub>aus</sub>ti<sub>ve</sub>l<sub>y.</sub>

– 323 of the 359 hard-origin programs turned out hidden-wrong.

• Writers v2. qwen3.6-35b-a3b and Qwen3-8B write up to three inputs each with a new prompt. It h<sub>as no</sub> l<sub>egacy-</sub>bi<sub>n</sub>di<sub>ng</sub> fl<sub>ag an</sub>d <sub>as</sub>k<sub>s</sub> f<sub>or</sub> i<sub>npu</sub>t<sub>s</sub> th<sub>a</sub>t <sub>exerc</sub>i<sub>se</sub> th<sub>e</sub> d<sub>ocumen</sub>t<sub>e</sub>d <sub>c</sub>h<sub>ange an</sub>d it<sub>s</sub> boundar<sub>y</sub> values. The<sub>y</sub> see the same information as before. Both <sub>p</sub>assed the <sub>q</sub>ualification (12/12, decode 65.1; 12/12, 43.8).

Table 1. Model-free release-gate adequacy over the complete finite repair space of the 52 afected test tasks (49 clusters), all-consumer-assertions gate. Added: mean visible inputs added to the typical ones. FA: share of hidden-wrong procedures released; FR: share of hidden-correct procedures blocked (task means). Stale: tasks in which the unrepaired procedure fails visibly.
<table><tr><td>Visible inputs</td><td>Added</td><td>FA</td><td>FR</td><td>Stale</td></tr><tr><td>typical only</td><td>0.00</td><td>0.410</td><td>0.010</td><td>34/52</td></tr><tr><td>+ blind writer, k=1</td><td>0.96</td><td>0.130</td><td>0.010</td><td>49/52</td></tr><tr><td>+ blind writer, k=2</td><td>1.85</td><td>0.075</td><td>0.010</td><td>50/52</td></tr><tr><td>+ blind writer, k=3</td><td>2.46</td><td>0.070</td><td>0.010</td><td>50/52</td></tr><tr><td>+ guarded writer, k=1</td><td>0.73</td><td>0.206</td><td>0.010</td><td>42/52</td></tr><tr><td>+ guarded writer, k=2</td><td>1.40</td><td>0.161</td><td>0.010</td><td>42/52</td></tr><tr><td>+ guarded writer, k=3</td><td>1.85</td><td>0.157</td><td>0.010</td><td>42/52</td></tr><tr><td>+ author edge input</td><td>1.00</td><td>0.065</td><td>0.010</td><td>52/52</td></tr></table>

• Runs. Arms T and E, eight cells each. GPT-5.6-luna made 1238 calls, Qwen3-8B 1593 (decode 48.4), <sub>gp</sub>t-oss-20b 1404 (62.0) and <sub>q</sub>wen3.6-35b-a3b 1391 (94.7). Truncated answers: 4, 4 and 4.

• Hypotheses (strict Holm, fixed at F0).

– HD1 FA(E)−FA(T), HD2 FA(W<sub>3</sub>)−FA(T) and HD3 FA(T, hard)−FA(T, lar<sub>g</sub>e), over the com<sub>p</sub>lete h<sub>ar</sub>d <sub>menu.</sub>

– HA1 E−T and HA2 FA<sub>�</sub>(hard)−FA<sub>�</sub>(lar<sub>g</sub>e), <sub>p</sub>er re<sub>p</sub>air famil<sub>y</sub>.

– WR1 and WR2, each new writer’s X<sub>3</sub> gate against T on the hard menu, with the same th<sub>res</sub>h<sub>o</sub>ld<sub>s as</sub> HD1<sub>.</sub> WR3 <sub>compares</sub> th<sub>e qwen</sub>3<sub>.</sub>6 <sub>wr</sub>it<sub>er w</sub>ith <sub>qwen</sub>3<sub>.</sub>6’<sub>s own ear</sub>li<sub>er se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s</sub> <sub>on</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>menu.</sub>

Th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree</sub> f<sub>am</sub>il<sub>y</sub> HD <sub>was</sub> <sub>compu</sub>t<sub>e</sub>d b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> f<sub>reezes,</sub> b<sub>e</sub>f<sub>ore</sub> <sub>any</sub> <sub>mo</sub>d<sub>e</sub>l <sub>ca</sub>ll<sub>.</sub>

What was known at each freeze.

• M4c de<sub>p</sub>ends onl<sub>y</sub> on the frozen build. It was com<sub>p</sub>uted durin<sub>g</sub> the mock-<sub>p</sub>i<sub>p</sub>eline test before F0 <sub>an</sub>d i<sub>s repor</sub>t<sub>e</sub>d <sub>as</sub> k<sub>nown a</sub>t F0<sub>.</sub>

• M4a, M4b and the audit were com<sub>p</sub>uted between F0 and F-test, as desi<sub>g</sub>ned.

• One amendment followed, before an<sub>y</sub> re<sub>p</sub>air call. It added ofline, descri<sub>p</sub>tive self-test <sub>g</sub>ates and chan<sub>g</sub>ed no <sub>p</sub>rimar<sub>y</sub> rule (A<sub>pp</sub>endix B).

## 5 Results

## 5.1 Model-free adequacy over the complete repair space

Table 1 and Fi<sub>g</sub>ure 1a a<sub>pp</sub>l<sub>y</sub> the all-assertions <sub>g</sub>ate to ever<sub>y p</sub>rocedure in the finite domain of <sub>eac</sub>h <sub>a</sub>f<sub>ec</sub>t<sub>e</sub>d t<sub>es</sub>t t<sub>as</sub>k<sub>.</sub> Th<sub>ey use</sub> th<sub>e v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>npu</sub>t<sub>s o</sub>f <sub>eac</sub>h l<sub>eve</sub>l<sub>.</sub> T<sub>a</sub>bl<sub>es repor</sub>t t<sub>as</sub>k <sub>means.</sub> Th<sub>e</sub> <sub>prereg</sub>i<sub>s</sub>t<sub>ere</sub>d t<sub>es</sub>t<sub>s</sub> <sub>use</sub> <sub>c</sub>l<sub>us</sub>t<sub>er</sub> <sub>means,</sub> f<sub>or</sub> <sub>examp</sub>l<sub>e</sub> 0<sub>.</sub>406 <sub>an</sub>d 0<sub>.</sub>064 f<sub>or</sub> T <sub>an</sub>d E<sub>;</sub> th<sub>e</sub> t<sub>wo</sub> <sub>never</sub> dif<sub>er</sub> b<sub>y more</sub> th<sub>an</sub> 0<sub>.</sub>005<sub>.</sub>

Author edge input.

• One release-note ed<sub>g</sub>e in<sub>p</sub>ut lowered the share of wron<sub>g</sub> <sub>p</sub>rocedures released from 0.410 to 0.065. The cluster-level diference is −0.342, with 36 discordant clusters. False rejection sta<sub>y</sub>ed at 0.010.

• M4c restricts the contrast to the 15 new test clusters, which did not exist when the efect was first observed. There the rate fell from 0.471 to 0.061 (diference −0.410, 95% CI [−0.589, −0.238], strict Holm � 0.002). The decision is positive.

• M4c is a deterministic function of the build and was known at F0 (Section 4).

![](images/acb720b0f35fd0813c197d43a601372c247bdf6ce0258c1675b1e411aa6f4d56.jpg)

![](images/b0a2a806cf92d82f485d50dbccf4047bf410ed959094bc6472f51963b1f6371b.jpg)  
Fig. 1. Headline result. (a) Share of wrong procedures released over the complete finite repair space as blind-writer inputs are added $( k = 0$ is typical inputs only; in parentheses, mean inputs actually added per task). The dashed line marks the author edge input. Unguarded writer inputs $( \mathsf { W } _ { k } )$ reach the author level at no cost in false rejection; the preregistered guard $( \mathrm { G } _ { k } )$ does not. (b) The same gates applied ofline to each model’s own hidden-wrong first-step repairs (arm T, test partition; � wrong candidates). Writer cost: one call per task.

Blind writer (descriptive). The unguarded writer curve was not a preregistered hypothesis, its <sub>va</sub>l<sub>ues</sub> <sub>were</sub> k<sub>nown</sub> b<sub>e</sub>f<sub>ore</sub> <sub>any</sub> <sub>repa</sub>i<sub>r</sub> <sub>ca</sub>ll<sub>,</sub> <sub>an</sub>d <sub>one</sub> <sub>mo</sub>d<sub>e</sub>l <sub>wro</sub>t<sub>e</sub> <sub>a</sub>ll i<sub>npu</sub>t<sub>s.</sub>

• Without seein<sub>g</sub> the <sub>p</sub>ro<sub>g</sub>ram, the fra<sub>g</sub>ments or the hidden in<sub>p</sub>uts, the writer’s in<sub>p</sub>uts lowered the rate to 0.130 with one in<sub>p</sub>ut and 0.070 with u<sub>p</sub> to three. On avera<sub>g</sub>e the<sub>y</sub> added 2.46 in<sub>p</sub>uts <sub>p</sub>er tas<sup>k</sup>.

• This is close to the author ed<sub>g</sub>e in<sub>p</sub>ut (0.065), and false rejection did not rise at an<sub>y</sub> �.

• Overlap and similarity (post hoc). 3 kept writer inputs equal a hidden input, in 3 tasks. With<sub>ou</sub>t th<sub>em,</sub> $\mathrm { W } _ { 1 } { - } \mathrm { W } _ { 3 }$ reach 0.149<sub>,</sub> 0.094 and 0.090. The overla<sub>pp</sub>in<sub>g</sub> in<sub>p</sub>uts carried <sub>p</sub>art of the <sub>e</sub>f<sub>ec</sub>t<sub>, an</sub>d th<sub>e unguar</sub>d<sub>e</sub>d <sub>wr</sub>it<sub>er w</sub>ith<sub>ou</sub>t th<sub>em rema</sub>i<sub>ns</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> 0<sub>.</sub>10 <sub>ce</sub>ili<sub>ng</sub> b<sub>u</sub>t <sub>a</sub>b<sub>ove</sub> th<sub>e</sub> author in<sub>p</sub>ut. The mean hi<sub>g</sub>hest strin<sub>g</sub> similarit<sub>y</sub> of the 128 ke<sub>p</sub>t writer in<sub>p</sub>uts of the test tasks to a hidden in ut of the same task is 0.70 (median 0.72; 20 at least 0.9), close to that of the author ed<sub>g</sub>e in<sub>p</sub>uts of the earlier stud<sub>y</sub> (0.71).

• The stale <sub>p</sub>rocedure fails visibl<sub>y</sub> in 50 of 52 tasks at $\mathrm { { W } } _ { 3 }$ . The writer was not bound by the QA <sub>ru</sub>l<sub>e</sub> th<sub>a</sub>t f<sub>orce</sub>d 52/52 f<sub>or</sub> th<sub>e</sub> <sub>au</sub>th<sub>or</sub> i<sub>npu</sub>t<sub>.</sub>

## The preregistered guard backfired.

${ \mathrm { G } } _ { 3 }$ k<sub>ep</sub>t <sub>on</sub>l<sub>y</sub> i<sub>npu</sub>t<sub>s</sub> th<sub>e</sub> <sub>wr</sub>it<sub>er</sub> d<sub>ec</sub>l<sub>are</sub>d l<sub>egacy-</sub>bi<sub>n</sub>di<sub>ng,</sub> <sub>an</sub>d <sub>reac</sub>h<sub>e</sub>d 0<sub>.</sub>157<sub>.</sub> Th<sub>e</sub> <sub>con</sub>t<sub>ras</sub>t <sub>w</sub>ith T is lar<sub>g</sub>e (−0.245, Holm $\textstyle p < 0 . 0 0 1 )$ . But the rate exceeds the preregistered ceiling of 0.10, so M4a is not positive.

• M4b has 0 discordant clusters: the guard removed no false rejection, because the unguarded in<sub>p</sub>uts caused none.

• The writer marked 35 of its 176 in<sub>p</sub>uts as non-bindin<sub>g</sub>. These were dis<sub>p</sub>ro<sub>p</sub>ortionatel<sub>y</sub> the <sub>c</sub>h<sub>ange-revea</sub>li<sub>ng</sub> <sub>ones:</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>l<sub>e</sub> <sub>proce</sub>d<sub>ure</sub> f<sub>a</sub>il<sub>s</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>y</sub> i<sub>n</sub> 42 t<sub>as</sub>k<sub>s</sub> <sub>un</sub>d<sub>er</sub> ${ \mathrm { G } } _ { 3 }$ a<sub>g</sub>ainst 50 under $\mathrm { W } _ { 3 }$

• The jud<sub>g</sub>ement “the old out<sub>p</sub>ut is no lon<sub>g</sub>er re<sub>q</sub>uired here” is exactl<sub>y</sub> where the release note is <sub>mos</sub>t <sub>re</sub>l<sub>evan</sub>t<sub>.</sub> It i<sub>s a</sub>l<sub>so w</sub>h<sub>ere</sub> th<sub>e wr</sub>it<sub>er erre</sub>d<sub>.</sub>

Table 2. Post-hoc robustness of the model-free efects (afected test tasks; FA task means; cluster-level diferences with unadjusted sign-flip �).
<table><tr><td>Subset</td><td>FA T</td><td>FA E</td><td>FA  $\mathrm { W } _ { 3 }$ </td><td>E-T</td><td> $\mathrm { W } _ { 3 } \mathrm { - T }$ </td></tr><tr><td>class S (23 tasks)</td><td>0.571</td><td>0.080</td><td>0.091</td><td>−0.503 (&lt;0.001)</td><td>−0.491 (&lt;0.001)</td></tr><tr><td>class E (25 tasks)</td><td>0.297</td><td>0.053</td><td>0.055</td><td>−0.235 (&lt;0.001)</td><td>−0.233 (&lt;0.001)</td></tr><tr><td>class B (4 tasks)</td><td>0.192</td><td>0.050</td><td>0.050</td><td>-0.142 (0.500)</td><td>-0.142 (0.500)</td></tr><tr><td>without CPython (16 tasks)</td><td></td><td>一</td><td></td><td>−0.328 (0.002)</td><td>-0.368 (0.002)</td></tr><tr><td>without Node.js (42 tasks)</td><td>1</td><td>一</td><td></td><td>−0.321 (&lt;0.001)</td><td>−0.313 (&lt;0.001)</td></tr><tr><td>without SQLite (46 tasks)</td><td></td><td></td><td></td><td>-0.367 (&lt;0.001)</td><td>−0.346 (&lt;0.001)</td></tr></table>

Table 3. Model repairs on the test partition under the all-consumer-assertions gate: correct completions / false acceptances / false rejections per arm (8 cells per task and arm; 456 records per arm). T typical inputs; S + the model’s own self-tests; E + the author edge input; ES both. Last column: false acceptances with no gate (arm T).
<table><tr><td>Model</td><td>T</td><td>S</td><td>E</td><td>ES</td><td>no gate</td></tr><tr><td>GPT-5.6-luna</td><td>398/22/0</td><td>403/9/0</td><td>408/7/0</td><td>409/0/0</td><td>45</td></tr><tr><td>Qwen3-8B</td><td>268/98/0</td><td>271/80/0</td><td>270/24/0</td><td>268/20/0</td><td>182</td></tr><tr><td>gpt-oss-20b</td><td>320/71/0</td><td>327/47/1</td><td>353/8/0</td><td>351/8/0</td><td>111</td></tr><tr><td>qwen3.6-35b-a3b</td><td>354/49/0</td><td>363/30/0</td><td>368/7/0</td><td>379/8/0</td><td>97</td></tr></table>

Table 4. Family A4 per model (test clusters; cluster means of per-atempt rates): diference and Holm-adjusted �. H1 FA(E)−FA(T); H2 FA(S)−FA(T); H3 FA(ES)−FA(E); H4 CC(S)−CC(T).
<table><tr><td>Model</td><td>Status</td><td>H1</td><td>H2</td><td>H3</td><td>H4</td></tr><tr><td>GPT-5.6-luna</td><td>descriptive</td><td>−0.038 (0.499)</td><td>−0.033 (0.749)</td><td>−0.018 (1.000)</td><td>0.013 (1.000)</td></tr><tr><td>Qwen3-8B</td><td>confirmatory</td><td>−0.176 (&lt;0.001)</td><td>-0.046 (0.749)</td><td>-0.010 (1.000)</td><td>0.010 (0.998)</td></tr><tr><td>gpt-oss-20b</td><td>confirmatory</td><td>−0.161 (&lt;0.001)</td><td>-0.064 (0.088)</td><td>0.000 (−)</td><td>0.023 (0.385)</td></tr><tr><td>qwen3.6-35b-a3b</td><td>descriptive</td><td>−0.102 (0.004)</td><td>-0.048 (0.256)</td><td>0.003 (1.000)</td><td>0.023 (0.256)</td></tr></table>

Robustness by class and lineage (post hoc). Table 2 splits the model-free contrasts by class <sub>an</sub>d l<sub>eaves</sub> <sub>one</sub> li<sub>neage</sub> <sub>ou</sub>t<sub>.</sub> Th<sub>e</sub> <sub>e</sub>d<sub>ge-</sub>i<sub>npu</sub>t <sub>an</sub>d <sub>wr</sub>it<sub>er</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s</sub> h<sub>o</sub>ld f<sub>or</sub> <sub>s</sub>il<sub>en</sub>t <sub>an</sub>d <sub>exp</sub>li<sub>c</sub>it <sub>c</sub>h<sub>anges</sub> <sub>an</sub>d i<sub>n</sub> <sub>every</sub> l<sub>eave-one-</sub>li<sub>neage-ou</sub>t <sub>su</sub>b<sub>se</sub>t<sub>.</sub> Th<sub>e</sub> f<sub>our</sub> <sub>no-sa</sub>f<sub>e-repa</sub>i<sub>r</sub> t<sub>es</sub>t t<sub>as</sub>k<sub>s</sub> <sub>are</sub> t<sub>oo</sub> f<sub>ew</sub> f<sub>or</sub> <sub>a</sub> t<sub>es</sub>t<sub>.</sub>

## 5.2 Repairs by four model families

Tables 3 and 4 re<sub>p</sub>ort the test-<sub>p</sub>artition re<sub>p</sub>airs (456 attem<sub>p</sub>ts <sub>p</sub>er arm and model).

• Wron<sub>g</sub> releases <sub>p</sub>er attem<sub>p</sub>t under all assertions fell as follows:

– Qwen3-8B: 0.233→0.057 (CI [−0.267, −0.094], Holm � <0.001);

– <sub>gp</sub>t-oss-20b: 0.181→0.020 (Holm � <0.001);

– <sub>q</sub>wen3.6-35b-a3b: 0.120→0.018 (Holm � 0.004).

• All three decisions are <sub>p</sub>ositive. For the two confirmator<sub>y</sub> families the<sub>y</sub> are confirmator<sub>y</sub>, and f<sub>or</sub> <sub>qwen</sub>3<sub>.</sub>6<sub>-</sub>35b<sub>-a</sub>3b d<sub>escr</sub>i<sub>p</sub>ti<sub>ve.</sub>

• Correct com<sub>p</sub>letion did not fall: the E−T diference was ≥ 0 in ever<sub>y</sub> famil<sub>y</sub>.

Table 5. Self-tests writen by each model from the release note and the visible contract: tasks with at least one valid test, valid tests, tests declared legacy-binding, tests identical to a hidden input, mean highest similarity to a hidden input, and the model-free FA of the gate built from the guarded (S) and unguarded (raw) self-tests, with the unguarded gate’s FR.
<table><tr><td>Model</td><td>Tasks</td><td>Tests</td><td>Binding</td><td>=hidden</td><td>Sim.</td><td>FA (S)</td><td>FA (raw)</td><td>FR (raw)</td></tr><tr><td>GPT-5.6-luna</td><td>71/73</td><td>187</td><td>153</td><td>5</td><td>0.69</td><td>0.153</td><td>0.054</td><td>0.031</td></tr><tr><td>Qwen3-8B</td><td>30/73</td><td>49</td><td>27</td><td>2</td><td>0.71</td><td>0.355</td><td>0.240</td><td>0.031</td></tr><tr><td>gpt-oss-20b</td><td>57/73</td><td>128</td><td>93</td><td>1</td><td>0.69</td><td>0.235</td><td>0.149</td><td>0.031</td></tr><tr><td>qwen3.6-35b-a3b</td><td>67/73</td><td>168</td><td>122</td><td>3</td><td>0.68</td><td>0.215</td><td>0.100</td><td>0.031</td></tr></table>

Table 6. Ofline gates applied to the hidden-wrong first-step candidates of arm T (test partition): share released by the blind-writer gate with 1 and 3 inputs (W1, W3), the guarded writer gate (G3), and the model’s own unguarded (Sraw) and guarded (S) self-tests; last column: share of hidden-correct candidates blocked by Sraw. All other gates blocked no correct candidate.
<table><tr><td>Model</td><td>wrong</td><td>W1</td><td>W3</td><td>G3</td><td>Sraw</td><td>S</td><td>FR Sraw</td></tr><tr><td>GPT-5.6-luna</td><td>45</td><td>0.178</td><td>0.044</td><td>0.178</td><td>0.044</td><td>0.178</td><td>0.023</td></tr><tr><td>Qwen3-8B</td><td>176</td><td>0.261</td><td>0.170</td><td>0.261</td><td>0.330</td><td>0.455</td><td>0.038</td></tr><tr><td>gpt-oss-20b</td><td>109</td><td>0.156</td><td>0.083</td><td>0.266</td><td>0.339</td><td>0.376</td><td>0.030</td></tr><tr><td>qwen3.6-35b-a3b</td><td>92</td><td>0.098</td><td>0.054</td><td>0.337</td><td>0.326</td><td>0.337</td><td>0.026</td></tr></table>

• GPT-5.6-luna released 22 wron<sub>g</sub> re<sub>p</sub>airs in arm T and 7 in arm E. This contrast is descri<sub>p</sub>tive, <sub>w</sub>ith H<sub>o</sub>l<sub>m</sub> <sub>�</sub> 0<sub>.</sub>499<sub>.</sub>

• The preregistered edge-input replication criterion is therefore met.

• Self-tests lowered wron<sub>g</sub> releases in ever<sub>y</sub> famil<sub>y</sub>: 22→9, 98→80, 71→47 and 49→30 out of 456 attem<sub>p</sub>ts eac<sup>h</sup>.

• None of these reductions survived Holm adjustment. The smallest Holm � was 0.132 (strict; code 0.088), for <sub>gp</sub>t-oss-20b.

• The self-test replication criterion is not met.

• Addin<sub>g</sub> self-tests to the ed<sub>g</sub>e in<sub>p</sub>ut (H3) chan<sub>g</sub>ed little: ES released 20, 8 and 8 wron<sub>g</sub> re<sub>p</sub>airs in the o<sub>p</sub>en-wei<sub>g</sub>ht families<sub>,</sub> a<sub>g</sub>ainst 24<sub>,</sub> 8 and 7 in E.

• Self-tests cost one extra call <sub>p</sub>er task. The<sub>y</sub> did not reduce correct com<sub>p</sub>letion: H4 diferences were 0.010 to 0.023.

Why self-tests helped less.

• Fewer valid tests from smaller models. The models difered most in how many valid tests the<sub>y</sub> wrote (Table 5). GPT-5.6-luna wrote valid tests for 71 of 73 tasks; Qwen3-8B mana<sub>g</sub>ed onl<sub>y</sub> 30.

• Weaker gates from those tests. Over the complete repair space, a gate built from Qwen3-8B’s <sub>unguar</sub>d<sub>e</sub>d <sub>se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s</sub> <sub>re</sub>l<sub>ease</sub>d 0<sub>.</sub>240 <sub>o</sub>f <sub>wrong</sub> <sub>proce</sub>d<sub>ures,</sub> <sub>aga</sub>i<sub>ns</sub>t 0<sub>.</sub>070 f<sub>or</sub> th<sub>e</sub> bli<sub>n</sub>d <sub>wr</sub>it<sub>er.</sub>

• Weakest exactly where needed. The weakness concentrates on each model’s own mistakes (Table 6). Gates were a<sub>pp</sub>lied ofline to the hidden-wron<sub>g</sub> first-ste<sub>p</sub> candidates of arm T:

– the blind writer’s gate $\mathrm { W } _ { 3 }$ released 0.170 of Qwen3-8B’s wrong candidates and 0.083 of <sub>gp</sub>t-oss-20<sup>b’</sup>s;

– the models’ own unguarded tests released 0.330 and 0.339.

Table 7. Post-hoc writer × repairer matrix (zero calls). Each family’s self-test step used exactly the blind-writer prompt and validation, so its unguarded self-tests are that family’s blind-writer inputs. Cells: share of the repairer’s hidden-wrong first-step candidates (arm T, test partition) released by the all-assertions gate on typical inputs plus the writer’s inputs. Tasks: afected test tasks with ≥1 valid input. Last column: model-free FA over the complete repair space on the 21 tasks where every writer has inputs (typical only: 0.510).
<table><tr><td>Writer ↓ / repairer →</td><td>Tasks</td><td>GPT-5.6-luna</td><td>Qwen3-8B</td><td>gpt-oss-20b</td><td>qwen3.6-35b-a3b</td><td>model-free</td></tr><tr><td>GPT-5.6-luna (writer-001)</td><td>50</td><td>0.044</td><td>0.170</td><td>0.083</td><td>0.054</td><td>0.069</td></tr><tr><td>GPT-5.6-luna</td><td>52</td><td>0.044</td><td>0.176</td><td>0.128</td><td>0.065</td><td>0.075</td></tr><tr><td>Qwen3-8B</td><td>23</td><td>0.356</td><td>0.330</td><td>0.385</td><td>0.337</td><td>0.109</td></tr><tr><td>gpt-oss-20b</td><td>42</td><td>0.178</td><td>0.330</td><td>0.339</td><td>0.272</td><td>0.088</td></tr><tr><td>qwen3.6-35b-a3b</td><td>48</td><td>0.311</td><td>0.210</td><td>0.202</td><td>0.326</td><td>0.078</td></tr></table>

Table 8. Post-hoc writer × repairer matrix restricted to the 21 afected test tasks where every writer has at least one valid input (removes the coverage diference; 30–70 wrong first-step repairs per repairer). Cells: share of the repairer’s hidden-wrong first-step candidates (arm T) released by typical inputs plus the writer’s inputs. Tasks: tasks with ≥1 valid input overall. Descriptive.
<table><tr><td>Writer\repairer</td><td>Tasks</td><td>GPT-5.6-luna</td><td>Qwen3-8B</td><td>gpt-oss-20b</td><td>qwen3.6-35b-a3b</td></tr><tr><td>GPT-5.6-luna (writer-001)</td><td>50</td><td>0.000</td><td>0.029</td><td>0.000</td><td>0.000</td></tr><tr><td>GPT-5.6-luna</td><td>52</td><td>0.000</td><td>0.157</td><td>0.125</td><td>0.021</td></tr><tr><td>Qwen3-8B</td><td>23</td><td>0.400</td><td>0.143</td><td>0.179</td><td>0.106</td></tr><tr><td>gpt-oss-20b</td><td>42</td><td>0.200</td><td>0.257</td><td>0.268</td><td>0.021</td></tr><tr><td>qwen3.6-35b-a3b</td><td>48</td><td>0.400</td><td>0.157</td><td>0.214</td><td>0.234</td></tr></table>

A <sub>mo</sub>d<sub>e</sub>l th<sub>a</sub>t <sub>m</sub>i<sub>srea</sub>d<sub>s a re</sub>l<sub>ease no</sub>t<sub>e</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o wr</sub>it<sub>e</sub> t<sub>es</sub>t<sub>s</sub> th<sub>a</sub>t <sub>agree w</sub>ith it<sub>s m</sub>i<sub>srea</sub>di<sub>ng.</sub>

• Matched on test volume (post hoc). Restricting to tasks where the family wrote at least one valid self-test removes the volume diference. On Qwen3-8B’s 23 such tasks its own unguarded t<sub>es</sub>t<sub>s re</sub>l<sub>ease</sub>d 0<sub>.</sub>132 <sub>o</sub>f it<sub>s</sub> 76 <sub>wrong</sub> fi<sub>rs</sub>t<sub>-s</sub>t<sub>ep repa</sub>i<sub>rs, aga</sub>i<sub>ns</sub>t 0<sub>.</sub>026 f<sub>or</sub> th<sub>e</sub> bli<sub>n</sub>d <sub>wr</sub>it<sub>er.</sub> F<sub>or</sub> <sub>gp</sub>t-oss-20b the fi<sub>g</sub>ures are 0.292 a<sub>g</sub>ainst 0.021<sub>,</sub> and for <sub>q</sub>wen3.6-35b-a3b 0.271 a<sub>g</sub>ainst 0.000. For GPT-5.6-luna, which is also the blind writer, the<sub>y</sub> are e<sub>q</sub>ual (0.044).

• Writer or independence? (post hoc, descriptive). Each family’s self-test step used exactly th<sub>e</sub> bli<sub>n</sub>d<sub>-wr</sub>it<sub>er</sub> <sub>promp</sub>t <sub>an</sub>d <sub>va</sub>lid<sub>a</sub>ti<sub>on.</sub> It<sub>s</sub> <sub>unguar</sub>d<sub>e</sub>d <sub>se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s</sub> <sub>are</sub> th<sub>ere</sub>f<sub>ore</sub> th<sub>a</sub>t f<sub>am</sub>il<sub>y</sub>’<sub>s</sub> blind-writer in<sub>p</sub>uts<sub>,</sub> and ever<sub>y</sub> famil<sub>y</sub>’s tests can <sub>g</sub>ate ever<sub>y</sub> famil<sub>y</sub>’s re<sub>p</sub>airs without new calls.

– Coverage difers. The number of tasks with at least one valid test difers by writer, from 23 for Qwen3-8B to 52 for GPT-5.6-luna, and tasks without tests fall back to typical inputs. The full matrix (Table 7) therefore mixes covera<sub>g</sub>e with test <sub>q</sub>ualit<sub>y</sub>.

– Common tasks. Table 8 restricts the matrix to the 21 tasks where every writer has tests. There the GPT-5.6-luna writer run <sub>g</sub>ates ever<sub>y</sub> famil<sub>y</sub> best or tied for best. The mean same-<sub>mo</sub>d<sub>e</sub>l <sub>ce</sub>ll i<sub>s</sub> 0<sub>.</sub>161 <sub>aga</sub>i<sub>ns</sub>t 0<sub>.</sub>187 f<sub>or cross-mo</sub>d<sub>e</sub>l <sub>ce</sub>ll<sub>s.</sub>

– Reading. This is consistent with writer capability mattering more than independence, and <sub>g</sub>i<sub>ves</sub> <sub>no</sub> <sub>s</sub>i<sub>gn</sub> th<sub>a</sub>t <sub>a</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>own</sub> t<sub>es</sub>t<sub>s</sub> <sub>are</sub> <sub>spec</sub>ifi<sub>ca</sub>ll<sub>y</sub> bli<sub>n</sub>d t<sub>o</sub> it<sub>s</sub> <sub>own</sub> <sub>errors.</sub> It <sub>res</sub>t<sub>s</sub> <sub>on</sub> f<sub>our</sub> families<sub>,</sub> one <sub>p</sub>rom<sub>p</sub>t and few wron<sub>g</sub> re<sub>p</sub>airs <sub>p</sub>er cell<sub>,</sub> so we treat it as a h<sub>yp</sub>othesis.

• The guard hurt again. The guard $( \mathsf { S } , \mathsf { G } _ { 3 } )$ <sub>aga</sub>i<sub>n</sub> l<sub>e</sub>t <sub>more</sub> <sub>wrong</sub> <sub>can</sub>did<sub>a</sub>t<sub>es</sub> th<sub>roug</sub>h th<sub>an</sub> it<sub>s</sub> un<sub>g</sub>uarded counter<sub>p</sub>art in ever<sub>y</sub> o<sub>p</sub>en-wei<sub>g</sub>ht famil<sub>y</sub>. Model scale (Figure 2).

• Across the four admitted families, the absolute ed<sub>g</sub>e-in<sub>p</sub>ut efect shrank as ca<sub>p</sub>abilit<sub>y</sub> <sub>g</sub>rew: <sup>S</sup><sub>p</sub>earman $\rho = - 1 . 0 0$ , exact $\mathnormal { p } = 0 . 0 8 3$ with � = 4.

![](images/7a6e2278b8eaa25ffe7de3d4a1a64fea1ac2ff3fdfa14c3f7b409f5519c09e5c.jpg)  
Fig. 2. Wrong releases per atempt under all assertions with typical inputs (filled) and with the author edge input (open), against each family’s capability (test partition, cluster means).

• The relative efect did not $( \rho = - 0 . 4 0 )$ <sub>.</sub> Th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> i<sub>npu</sub>t <sub>remove</sub>d <sub>mos</sub>t <sub>o</sub>f <sub>w</sub>h<sub>a</sub>t<sub>ever</sub> <sub>wrong</sub> <sub>re</sub>l<sub>eases</sub> <sub>a</sub> f<sub>am</sub>il<sub>y</sub> <sub>pro</sub>d<sub>uce</sub>d<sub>.</sub> St<sub>ronger</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>pro</sub>d<sub>uce</sub>d f<sub>ewer.</sub>

• With four families this is descri<sub>p</sub>tive.

## 5.3 Large-menu replication

## Model-free (Table 9).

• On the lar<sub>g</sub>e menu, the ed<sub>g</sub>e in<sub>p</sub>ut lowered the share of wron<sub>g</sub> <sub>p</sub>rocedures released from 0.278 to 0.045 (LM1 positive, strict Holm � <0.001).

• The unguarded blind writer lowered it to 0.053, with false rejection 0.043 against 0.014 (LM2 positive, Holm $\textstyle p < 0 . 0 0 1 )$ . The writer result thus <sub>p</sub>asses a <sub>p</sub>rere<sub>g</sub>istered test on a lar<sub>g</sub>er s<sub>p</sub>ace<sub>,</sub> for a sin<sub>g</sub>le writer model (GPT-5.6-luna).

• The <sub>g</sub>uarded writer a<sub>g</sub>ain la<sub>gg</sub>ed (0.106).

• LM3 is not positive and goes the other way. The typical-input gate released a smaller share of wron<sub>g</sub> <sub>p</sub>rocedures on the lar<sub>g</sub>e menu than on the earlier one (0.406→0.271, cluster means). Most added wrong fragments fail visibly on typical inputs. The share of wrong programs released therefore de<sub>p</sub>ends on how the menu is built. The headline shares (0.410, 0.278) hold onl<sub>y</sub> for their own menus<sub>,</sub> and are not <sub>p</sub>ro<sub>p</sub>erties of a <sub>g</sub>ate.

Table 9. Large-menu stage, model-free: share of hidden-wrong (FA) and hidden-correct blocked (FR) procedures over the complete large-menu space of the afected test tasks (1986 procedures in all tasks), allconsumer-assertions gate, task means.
<table><tr><td>Visible inputs</td><td>FA</td></tr><tr><td>typical only</td><td>0.278 0.014</td></tr><tr><td>+ author edge input</td><td>0.045 0.014</td></tr><tr><td>+ blind writer, k=1</td><td>0.089 0.036</td></tr><tr><td>+ blind writer, k=2</td><td>0.054 0.043</td></tr><tr><td>+ blind writer, k=3</td><td>0.053 0.043</td></tr><tr><td>+ guarded writer, k=1</td><td>0.135 0.036</td></tr><tr><td>+ guarded writer, k=2</td><td>0.107 0.043</td></tr><tr><td>+ guarded writer, k=3</td><td>0.106 0.043</td></tr></table>

Table 10. Large-menu stage, model repairs (test partition, 456 atempts per arm): correct completions / wrong releases under all consumer assertions for arms T and E; LH1 wrong releases per atempt T→E (strict Holm �); LH2 change in wrong releases per atempt with typical inputs relative to the v4 menu (strict Holm �); LH3 change in correct completion relative to the v4 menu.
<table><tr><td>Model</td><td>Status</td><td>T</td><td>E</td><td>LH1</td><td>LH2</td><td>LH3</td></tr><tr><td>GPT-5.6-luna</td><td>descriptive</td><td>391/29</td><td>401/12</td><td>0.074→0.031 (0.378)</td><td>0.018 (0.378)</td><td>-0.018</td></tr><tr><td>Qwen3-8B</td><td>confirmatory</td><td>241/108</td><td>244/27</td><td>0.253→0.062 (&lt;0.001)</td><td>0.019 (0.634)</td><td>-0.047</td></tr><tr><td>gpt-oss-20b</td><td>confirmatory</td><td>310/85</td><td>334/15</td><td>0.212→0.038 (&lt;0.001)</td><td>0.031 (0.335)</td><td>-0.026</td></tr><tr><td>qwen3.6-35b-a3b</td><td>descriptive</td><td>327/72</td><td>344/9</td><td>0.177→0.023 (&lt;0.001)</td><td>0.057 (0.047)</td><td>-0.069</td></tr></table>

• Menu-independent view (post hoc, Table 11). We count, per task, the wrong programs that the visible inputs cannot tell apart from a correct one. With typical inputs this number grows <sub>w</sub>ith th<sub>e menu,</sub> f<sub>rom</sub> 113 t<sub>o</sub> 445 <sub>over</sub> th<sub>e</sub> t<sub>es</sub>t t<sub>as</sub>k<sub>s, an</sub>d 41 <sub>or</sub> 42 t<sub>as</sub>k<sub>s</sub> h<sub>ave a</sub>t l<sub>eas</sub>t <sub>one suc</sub>h <sub>p</sub>ro<sub>g</sub>ram. With the ed<sub>g</sub>e in<sub>p</sub>ut it is 14 and 46<sub>,</sub> in 11 and 17 tasks<sub>;</sub> with the blind writer 15 and 62. The task-level view of model repairs agrees. For example, Qwen3-8B released a wrong repair in 19 tasks with t<sub>yp</sub>ical in<sub>p</sub>uts and 5 with the ed<sub>g</sub>e in<sub>p</sub>ut on the earlier menu<sub>,</sub> and in 20 and 5 on th<sub>e</sub> l<sub>arge</sub> <sub>menu.</sub>

## Model repairs (Table 10).

• LH1. The edge input cut wrong releases per attempt for Qwen3-8B (0.253→0.062), gpt-oss-20b (0.212→0.038) and <sub>q</sub>wen3.6-35b-a3b (0.177→0.023). All three decisions are <sub>p</sub>ositive, both confirmatory families included, so the preregistered cross-family criterion is met. GPT-5.6-luna a<sub>g</sub>ain sta<sub>y</sub>ed near the ceilin<sub>g</sub> (0.074→0.031, Holm � 0.378).

• LH2 and LH3. Unlike the model-free space, the models’ own typical-input wrong releases <sub>rose</sub> <sub>s</sub>li<sub>g</sub>htl<sub>y</sub> <sub>on</sub> th<sub>e</sub> l<sub>arge</sub> <sub>menu</sub> i<sub>n</sub> <sub>every</sub> f<sub>am</sub>il<sub>y,</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>correc</sub>t <sub>comp</sub>l<sub>e</sub>ti<sub>on</sub> f<sub>e</sub>ll <sub>s</sub>li<sub>g</sub>htl<sub>y.</sub> O<sub>n</sub>l<sub>y</sub> <sub>qwen</sub>3<sub>.</sub>6<sub>-</sub>35b<sub>-a</sub>3b<sub>,</sub> <sub>a</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ve</sub> f<sub>am</sub>il<sub>y,</sub> <sub>reac</sub>h<sub>e</sub>d H<sub>o</sub>l<sub>m</sub> $\textstyle p < 0 . 0 5$ (LH2 0.057, � 0.047). These are d<sub>escr</sub>i<sub>p</sub>ti<sub>ve</sub> <sub>c</sub>h<sub>anges,</sub> <sub>no</sub>t <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t i<sub>n</sub> th<sub>e</sub> <sub>con</sub>fi<sub>rma</sub>t<sub>ory</sub> f<sub>am</sub>ili<sub>es.</sub> Th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> i<sub>npu</sub>t <sub>s</sub>till <sub>c</sub>l<sub>ose</sub>d th<sub>e</sub> g<sup>ate.</sup>

• Ofline gates. The blind writer’s gate $\mathrm { W } _ { 3 }$ <sub>re</sub>l<sub>ease</sub>d 0<sub>.</sub>161<sub>,</sub> 0<sub>.</sub>089 <sub>an</sub>d 0<sub>.</sub>034 <sub>o</sub>f th<sub>e</sub> <sub>open-we</sub>i<sub>g</sub>ht f<sub>am</sub>ili<sub>es</sub>’ <sub>wrong</sub> fi<sub>rs</sub>t<sub>-s</sub>t<sub>ep</sub> <sub>repa</sub>i<sub>rs.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>own</sub> <sub>unguar</sub>d<sub>e</sub>d <sub>se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s</sub> <sub>re</sub>l<sub>ease</sub>d 0<sub>.</sub>382<sub>,</sub> 0<sub>.</sub>379 <sub>an</sub>d 0<sub>.</sub>297<sub>.</sub>

## 5.4 Hard distractors and two more writers

## Model-free (Table 12).

Table 11. Post-hoc, menu-independent view (52 afected test tasks). Top: wrong programs that pass the all-assertions gate on the visible inputs, i.e. that the visible inputs cannot tell apart from a correct one (total, mean per task in parentheses), and tasks with at least one such program. Botom: tasks in which a model released at least one wrong repair in any of its 8 cells under all assertions, on the earlier and the large menu.
<table><tr><td>Menu</td><td rowspan="2">Programs</td><td colspan="2">T</td><td colspan="2">E</td><td colspan="2"> $\mathrm { W } _ { 3 }$ </td></tr><tr><td>median</td><td>wrong</td><td>tasks</td><td>wrong</td><td>tasks</td><td>wrong</td><td>tasks</td></tr><tr><td>earlier menu</td><td>5</td><td>113 (2.17)</td><td>41</td><td>14 (0.27)</td><td>11</td><td>15 (0.29)</td><td>12</td></tr><tr><td>large menu</td><td>30</td><td>445 (8.56)</td><td>42</td><td>46 (0.88)</td><td>17</td><td>62 (1.19)</td><td>15</td></tr></table>

<table><tr><td colspan="3">earlier menu</td><td rowspan="2">large menu T E</td></tr><tr><td>Model</td><td>T</td><td>E</td></tr><tr><td>GPT-5.6-luna</td><td>5</td><td>1</td><td>7 3</td></tr><tr><td>Qwen3-8B</td><td>19</td><td>5</td><td>20 5</td></tr><tr><td>gpt-oss-20b</td><td>16</td><td>1</td><td>16 2</td></tr><tr><td>qwen3.6-35b-a3b</td><td>11</td><td>2</td><td>17 2</td></tr></table>

Table 12. Hard-distractor stage, model-free over the complete hard-menu space of the afected test tasks (2310 procedures in all tasks): FA and FR task means under all consumer assertions, and FA restricted to hard-origin programs (wrong options that match the legacy output on the typical inputs).
<table><tr><td>Visible inputs</td><td>FA</td><td>FR</td><td>FA hard</td></tr><tr><td>typical only</td><td>0.337</td><td>0.010</td><td>0.990</td></tr><tr><td>+ blind writer (GPT-5.6-luna), k=3</td><td>0.058</td><td>0.042</td><td>0.153</td></tr><tr><td>+ guarded writer, k=3</td><td>0.135</td><td>0.042</td><td>0.492</td></tr><tr><td>+ author edge input</td><td>0.058</td><td>0.010</td><td>0.168</td></tr><tr><td>+ blind writer v2 (qwen3.6), k=3</td><td>0.094</td><td>0.083</td><td>一</td></tr><tr><td>+ blind writer v2 (Qwen3-8B), k=3</td><td>0.182</td><td>0.049</td><td>一</td></tr></table>

Table 13. Hard-distractor stage, model repairs (test partition, 456 atempts per arm): correct completions / wrong releases under all consumer assertions in arms T and E; HA1 wrong releases per atempt T→E (strict Holm �); HA2 change of typical-input wrong releases relative to the large menu. Last four columns (descriptive, ofline): share of each model’s hidden-wrong first-step repairs released by the typical inputs plus three inputs of the GPT-5.6-luna writer, the qwen3.6 and Qwen3-8B writers (prompt v2), and the model’s own self-tests. <sup>�</sup> confirmatory family.
<table><tr><td>Model</td><td>T</td><td>E</td><td>HA1</td><td>HA2</td><td>luna</td><td>qwen3.6</td><td>Qwen3-8B</td><td>own</td></tr><tr><td>GPT-5.6-luna</td><td>385/42</td><td>394/12</td><td>0.107→0.031 (0.063)</td><td>0.033 (0.094)</td><td>0.117</td><td>0.433</td><td>0.300</td><td>0.117</td></tr><tr><td>Qwen3-8Bc</td><td>245/105</td><td>244/22</td><td>0.250→0.050 (&lt;0.001)</td><td>-0.003 (1.000)</td><td>0.173</td><td>0.303</td><td>0.378</td><td>0.395</td></tr><tr><td>gpt-oss-20bc</td><td>306/88</td><td>339/17</td><td>0.218→0.043 (&lt;0.001)</td><td>0.006 (0.757)</td><td>0.104</td><td>0.392</td><td>0.384</td><td>0.416</td></tr><tr><td>qwen3.6-35b-a3b</td><td>317/80</td><td>347/11</td><td>0.198→0.028 (&lt;0.001)</td><td>0.020 (0.225)</td><td>0.048</td><td>0.373</td><td>0.349</td><td>0.317</td></tr></table>

• The menu is harder. The hard distractors raised the typical-input gate’s FA above the large menu’s (HD3 positive, 0.271→0.328 cluster means). Typical inputs released 0.990 of the hard <sup>wron</sup>g p<sup>ro</sup>g<sup>rams</sup>, <sup>as constructed.</sup>

• The edge input reduces but does not close the gap. It released 0.058 of all wrong programs (HD1 positive, Holm � <0.001), but 0.168 of the hard wrong programs still passed, above the 0.10 ceilin<sub>g</sub> that the overall rate meets. HD1 <sub>p</sub>asses because hard <sub>p</sub>ro<sub>g</sub>rams are a minorit<sub>y</sub> of the <sub>wrong</sub> <sub>ones.</sub> I<sub>n</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> t<sub>erms,</sub> th<sub>e</sub> <sub>wrong</sub> <sub>programs</sub> th<sub>a</sub>t th<sub>e</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>npu</sub>t<sub>s</sub> <sub>canno</sub>t t<sub>e</sub>ll <sub>apar</sub>t f<sub>rom</sub> a correct one number 709 with t<sub>yp</sub>ical in<sub>p</sub>uts<sub>,</sub> 77 with the ed<sub>g</sub>e in<sub>p</sub>ut and 82 with the writer’s i<sub>npu</sub>t<sub>s.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> i<sub>npu</sub>t thi<sub>s</sub> <sub>coun</sub>t <sub>grows</sub> <sub>w</sub>ith th<sub>e</sub> <sub>menu:</sub> 14<sub>,</sub> 46 <sub>an</sub>d 77 <sub>on</sub> th<sub>e</sub> th<sub>ree</sub> <sub>menus.</sub>

• The GPT-5.6-luna writer behaves alike: 0.058 overall and 0.153 of the hard wrong programs (HD2 positive).

• The two v2 writers are weaker. The qwen3.6 writer reached 0.094, below the 0.10 ceiling. But its inputs raised false rejection to 0.083 against 0.010, beyond the 0.05 tolerance, so WR1 is not positive. The Qwen3-8B writer reached only 0.182 (WR2 not positive).

• WR3. We found no evidence that the v2 prompt helps: on the first menu, the qwen3.6 writer with the new <sub>p</sub>rom<sub>p</sub>t released 0.138 a<sub>g</sub>ainst 0.097 for <sub>q</sub>wen3.6’s own earlier self-tests (twosided, Holm � 0.196). With the earlier <sub>p</sub>rom<sub>p</sub>t, the GPT-5.6-luna writer out<sub>p</sub>erformed <sub>q</sub>wen3.6’s <sub>se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s.</sub>

• Writer <sub>y</sub>ield: the <sub>q</sub>wen3.6 writer ke<sub>p</sub>t 171 in<sub>p</sub>uts in 61 tasks (8 un<sub>p</sub>arseable answers); the Qwen3-8B writer ke<sub>p</sub>t 105 in 44 (21 un<sub>p</sub>arseable).

## Model repairs (Table 13).

• HA1. The edge input cut wrong releases per attempt for Qwen3-8B (0.250→0.050), gpt-oss-20b (0.218→0.043) and <sub>q</sub>wen3.6-35b-a3b (0.198→0.028), all with strict Holm � < 0.001. The cross-family criterion is met. GPT-5.6-luna: 0.107→0.031 (Holm � 0.063, descriptive).

• HA2. Typical-input wrong releases did not change detectably relative to the large menu in any f<sub>am</sub>il<sub>y.</sub>

• Ofline. Applied ofline to the models’ own wrong first-step repairs, the GPT-5.6-luna writer’s i<sub>npu</sub>t<sub>s aga</sub>i<sub>n</sub> l<sub>e</sub>t th<sub>roug</sub>h th<sub>e</sub> f<sub>ewes</sub>t<sub>.</sub> Th<sub>e</sub> t<sub>wo v</sub>2 <sub>wr</sub>it<sub>ers an</sub>d th<sub>e mo</sub>d<sub>e</sub>l<sub>s</sub>’ <sub>own se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s</sub> l<sub>e</sub>t th<sub>roug</sub>h <sub>severa</sub>l ti<sub>mes</sub> <sub>more.</sub> A<sub>cross</sub> <sub>a</sub>ll <sub>wr</sub>it<sub>ers</sub> <sub>we</sub> t<sub>es</sub>t<sub>e</sub>d<sub>,</sub> th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t <sub>wr</sub>it<sub>er</sub> <sub>ga</sub>t<sub>e</sub>d b<sub>es</sub>t<sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ve,</sub> <sub>an</sub>d it <sub>means</sub> th<sub>a</sub>t th<sub>e</sub> <sub>wr</sub>it<sub>er</sub> <sub>resu</sub>lt d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>a</sub> f<sub>ron</sub>ti<sub>er</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

## 5.5 AI contract audit

• The two blind re<sub>p</sub>licates a<sub>g</sub>reed on 0.91 of item verdicts (� = 0.73).

• The <sub>y</sub>es-rates were R1 0.98, R2 0.92, R3 0.81, R4 0.51 and R5 0.82.

• Under the frozen rule, 34 of 73 tasks have at least one item on which both re<sub>p</sub>licates said no. Th<sub>e</sub> it<sub>ems were</sub> R4 f<sub>or</sub> 20 t<sub>as</sub>k<sub>s,</sub> R3 f<sub>or</sub> 10<sub>,</sub> R5 f<sub>or</sub> 7 <sub>an</sub>d R2 f<sub>or</sub> 3<sub>.</sub>

• R4 objections. Most R4 objections concern the file-preservation check p0, which the require-<sub>men</sub>t<sub>s</sub> d<sub>o no</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e.</sub>

• R2 objections. The R2 objections are substantive. For two Node.js tasks the quoted entry d<sub>ocumen</sub>t<sub>s a</sub> d<sub>epreca</sub>ti<sub>on</sub> hi<sub>s</sub>t<sub>ory ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e remova</sub>l it<sub>se</sub>lf<sub>.</sub>

• Contracts unchanged. Contracts were not changed, as preregistered.

• Post-hoc check of the objections. Redefining hidden correctness without p0 changes 0 of th<sub>e ex</sub>h<sub>aus</sub>ti<sub>ve</sub> hidd<sub>en ver</sub>di<sub>c</sub>t<sub>s o</sub>f th<sub>e</sub> t<sub>es</sub>t t<sub>as</sub>k<sub>s.</sub> S<sub>o</sub> d<sub>oes</sub> d<sub>ropp</sub>i<sub>ng p</sub>0 <sub>an</sub>d l<sub>egacy equa</sub>lit<sub>y</sub> d0 t<sub>oge</sub>th<sub>er:</sub> 0 <sub>c</sub>h<sub>anges.</sub> Th<sub>e c</sub>h<sub>ec</sub>k<sub>s p</sub>0 <sub>an</sub>d d0 <sub>never</sub> d<sub>ec</sub>id<sub>e a ver</sub>di<sub>c</sub>t th<sub>a</sub>t th<sub>e consumer asser</sub>ti<sub>ons</sub> d<sub>o no</sub>t <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> d<sub>ec</sub>id<sub>e.</sub>

• Sensitivity analysis. Excluding the 24 flagged test tasks entirely leaves the edge-input contrast at −0.370 (� <0.001, 25 clusters).

• Strong-evidence criterion. Our predeclared strong-evidence criterion fails on two counts: M4<sub>a</sub> i<sub>s</sub> <sub>no</sub>t <sub>pos</sub>iti<sub>ve,</sub> <sub>an</sub>d <sub>more</sub> th<sub>an</sub> 10% <sub>o</sub>f t<sub>as</sub>k<sub>s</sub> <sub>are</sub> AI<sub>-</sub>fl<sub>agge</sub>d<sub>.</sub> Th<sub>e</sub> h<sub>uman</sub> <sub>s</sub>i<sub>gn-o</sub>f <sub>o</sub>f th<sub>e</sub> <sub>ear</sub>li<sub>er</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> i<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>no</sub>t <sub>c</sub>l<sub>a</sub>i<sub>me</sub>d<sub>.</sub>

Table 14. Contamination probe: hidden-correct first-step repairs without any release-note text (in parentheses: with the excerpts, typical inputs, full representation, no dif view, listed order). Uniform random choice from each finite domain would be expected to be hidden-correct in 9.9/45, 3.8/13, 0.8/4 and 1.0/4 tasks.
<table><tr><td>Model</td><td>CPython</td><td>Node.js</td><td>SQLite 3.53</td><td>SQLite 3.54</td></tr><tr><td>GPT-5.6-luna</td><td>37/45 (37)</td><td>10/13 (10)</td><td>3/4 (2)</td><td>3/4 (3)</td></tr><tr><td>Qwen3-8B</td><td>19/45 (24)</td><td>6/13 (9)</td><td>1/4 (0)</td><td>1/4 (0)</td></tr><tr><td>gpt-oss-20b</td><td>29/45 (28)</td><td>7/13 (8)</td><td>2/4 (1)</td><td>3/4 (3)</td></tr><tr><td>qwen3.6-35b-a3b</td><td>32/45 (34)</td><td>10/13 (6)</td><td>2/4 (3)</td><td>2/4 (2)</td></tr></table>

## 5.6 Contamination probe

Witho<sub>u</sub>t an<sub>y</sub> release-note text<sub>,</sub> GPT-5.6-l<sub>u</sub>na <sub>p</sub>rod<sub>u</sub>ced hidden-correct first-ste<sub>p</sub> re<sub>p</sub>airs in 37/45 CPython tasks. It solved 3/4 of the post-cutof SQLite 3.54 tasks. The open-weight families solved fewer, but ever<sub>y</sub> famil<sub>y</sub> was well above uniform random choice (9.9/45 for CP<sub>y</sub>thon). The tem<sub>p</sub>late, <sub>requ</sub>i<sub>remen</sub>t <sub>an</sub>d <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>npu</sub>t<sub>s o</sub>ft<sub>en su</sub>fi<sub>ce</sub> t<sub>o</sub> i<sub>n</sub>f<sub>er</sub> th<sub>e repa</sub>i<sub>r.</sub> Th<sub>e pro</sub>b<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> b<sub>oun</sub>d<sub>s w</sub>h<sub>a</sub>t release notes add to repair. It does not bear on the gate results, which concern what reaches release.

## 6 Prior Evidence

Four ear<sup>l</sup>ier <sup>l</sup>ayers o<sup>f</sup> t<sup>h</sup>is project motivate<sup>d</sup> t<sup>h</sup>e present <sup>d</sup>esign. T<sup>h</sup>ey <sup>k</sup>eep t<sup>h</sup>eir own <sup>d</sup>enominators <sub>an</sub>d <sub>are never poo</sub>l<sub>e</sub>d <sub>w</sub>ith it<sub>.</sub>

## 6.1 Preregistered coverage study (the direct predecessor)

Th<sub>e</sub> di<sub>rec</sub>t <sub>pre</sub>d<sub>ecessor</sub> <sub>use</sub>d th<sub>e</sub> fi<sub>rs</sub>t 55 <sub>o</sub>f th<sub>e</sub> <sub>presen</sub>t t<sub>as</sub>k<sub>s,</sub> th<sub>ree</sub> <sub>mo</sub>d<sub>e</sub>l f<sub>am</sub>ili<sub>es</sub> <sub>an</sub>d th<sub>ree</sub> <sub>coverage</sub> levels: t<sub>yp</sub>ical<sub>,</sub> author ed<sub>g</sub>e in<sub>p</sub>ut<sub>,</sub> and LLM-written in<sub>p</sub>uts.

• Model-free. Over the complete repair space, the author edge input lowered the all-assertions false-acce<sub>p</sub>tance rate from 0.377 to 0.065 (cluster means; Holm � <0.001).

• LLM inputs. LLM inputs went further, to 0.034, but raised false rejection by 0.061.

• Model families. Wrong releases per attempt fell for Qwen3-8B (0.228→0.085) and gpt-oss-20b (0.169→0.029).

Its freeze history, edge-input QA rule and contract review were open points that the present study addresses (Section 4 and A<sub>pp</sub>endix B).

## 6.2 Typical-input study on two lineages

T<sup>h</sup>e <sup>fi</sup>rst, exp<sup>l</sup>oratory stu<sup>d</sup>y use<sup>d</sup> 33 tas<sup>k</sup>s on CPyt<sup>h</sup>on an<sup>d</sup> No<sup>d</sup>e.js, wit<sup>h</sup> one typica<sup>l</sup> visi<sup>bl</sup>e input <sub>p</sub>er task and six release <sub>g</sub>ates. Its main findin<sub>g</sub>s were:

• Exhaustive gate adequacy. Over the complete repair space, runtime/schema checks released 0.80 of wron<sub>g</sub> re<sub>p</sub>airs (task mean), and all consumer assertions and diferential testin<sub>g</sub> released 0<sub>.</sub>36<sub>.</sub> Th<sub>e</sub> t<sub>wo</sub> <sub>co</sub>i<sub>nc</sub>id<sub>e</sub>d b<sub>ecause</sub> <sub>eac</sub>h t<sub>as</sub>k h<sub>a</sub>d <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>npu</sub>t<sub>.</sub>

• GPT-5.6-luna. It was near the ceiling, and no preregistered contrast was significant.

• Qwen3-8B. It released 31 wrong repairs under runtime checks and 20 under all assertions, out of 108 attem<sub>p</sub>ts. This diference was descri<sub>p</sub>tive: Holm <sub>�</sub> = 0.375.

T<sub>wo</sub> <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> <sub>ana</sub>l<sub>yses</sub> f<sub>o</sub>ll<sub>owe</sub>d<sub>,</sub> <sub>ne</sub>ith<sub>er</sub> <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>re</sub>d <sub>any</sub> <sub>mo</sub>d<sub>e</sub>l <sub>ca</sub>ll<sub>:</sub>

• Requirement-only sensitivity. When only requirement-derived assertions are kept, the <sub>g</sub>ate orderin<sub>g</sub> is unchan<sub>g</sub>ed. Diferential testin<sub>g</sub> then over-rejects (0.26), and the no-safe-re<sub>p</sub>air labels shift: node-createcipher (4), py-binascii-hqx (5), py-distutils-looseversion (2), py-shuffle-random-param (5) correct <sub>p</sub>ro<sub>g</sub>rams a<sub>pp</sub>ear.

Table 15. Prior evidence: sealed deterministic study, held-out test split (20 lineages, 100 task units). Scopeneutral reconstruction B1R and local repair B4 share the finite candidate base and the 24-probe ceiling; B1R also reads the complete synthetic new-version document. Resource columns are the only resources this layer measured (means per unit); tokens, latency and graph-construction efort were not measured. The injected B1/B1C rows are in Appendix C.
<table><tr><td>Condition</td><td>Joint</td><td>affected</td><td>controls</td><td>silent</td><td>abstain</td><td>probes</td><td>doc. bytes</td><td>edits</td></tr><tr><td>stale procedure (B0)</td><td>24/100</td><td>一</td><td>一</td><td>48</td><td>0</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>scope-neutral reconstruction (B1R)</td><td>96/100</td><td>72</td><td>24</td><td>0</td><td>4</td><td>1.04</td><td>119.83</td><td>1.00</td></tr><tr><td>contract-guided local repair (B4)</td><td>96/100</td><td>72</td><td>24</td><td>0</td><td>4</td><td>2.60</td><td>11.16</td><td>1.00</td></tr></table>

• Counterfactual. Moving one hidden edge input into the visible set and re-judging the existing can<sup>d</sup>i<sup>d</sup>ates <sup>l</sup>owere<sup>d</sup> t<sup>h</sup>e a<sup>ll</sup>-assertions <sup>f</sup>a<sup>l</sup>se-acceptance rate to 0.054, wit<sup>h</sup> no <sup>f</sup>a<sup>l</sup>se rejection.

The counterfactual reuses in<sub>p</sub>uts that also define correctness<sub>,</sub> so it is o<sub>p</sub>timistic. The covera<sub>g</sub>e stud<sub>y</sub> above and the <sub>p</sub>resent stud<sub>y</sub> test the same mechanism <sub>p</sub>ros<sub>p</sub>ectivel<sub>y,</sub> with new in<sub>p</sub>uts that are never hidd<sub>en.</sub>

## 6.3 Sealed deterministic study and visible-completion failures

Table 15 re<sub>p</sub>roduces the sco<sub>p</sub>e-neutral com<sub>p</sub>arison from the sealed deterministic stud<sub>y</sub> (24 linea<sub>g</sub>es, 120 task units):

• the stale <sub>p</sub>rocedure com<sub>p</sub>leted 24/100 held-out units, and a further 48 were acce<sub>p</sub>ted but wron<sub>g</sub>;

• sco<sub>p</sub>e-neutral reconstruction (B1R) and local re<sub>p</sub>air (B4) each com<sub>p</sub>leted 96/100;

• the injected B1/B1C evaluator-<sub>q</sub>ualification rows are listed in A<sub>pp</sub>endix Table 19.

In <sub>a</sub> blind <sub>sa</sub>m<sub>p</sub>l<sub>e o</sub>f <sub>s</sub>ix <sub>pu</sub>bli<sub>c</sub> APIFl<sub>ow</sub>-B<sub>e</sub>n<sub>c</sub>h t<sub>as</sub>k<sub>s,</sub> thr<sub>ee</sub> r<sub>epa</sub>ir<sub>s</sub> m<sub>e</sub>t th<sub>e v</sub>i<sub>s</sub>ibl<sub>e</sub>-<sub>co</sub>m<sub>p</sub>l<sub>e</sub>ti<sub>o</sub>n <sub>ru</sub>l<sub>e</sub> b<sub>u</sub>t f<sub>a</sub>il<sub>e</sub>d th<sub>e ups</sub>t<sub>ream va</sub>lid<sub>a</sub>t<sub>or;</sub> T<sub>a</sub>bl<sub>e</sub> 16 <sub>rea</sub>d<sub>s</sub> th<sub>em mec</sub>h<sub>an</sub>i<sub>ca</sub>ll<sub>y</sub> f<sub>rom</sub> th<sub>e re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>recor</sub>d<sub>s.</sub> I<sub>n</sub> <sub>eac</sub>h <sub>case</sub> th<sub>e</sub> <sub>requ</sub>i<sub>remen</sub>t <sub>as</sub>k<sub>e</sub>d f<sub>or</sub> <sub>re-au</sub>th<sub>en</sub>ti<sub>ca</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>reuse</sub>d <sub>a</sub> <sub>s</sub>t<sub>a</sub>l<sub>e</sub> t<sub>o</sub>k<sub>en</sub> th<sub>a</sub>t th<sub>e</sub> <sub>moc</sub>k <sub>accep</sub>t<sub>e</sub>d<sub>.</sub> Th<sub>e</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> <sub>c</sub>h<sub>ec</sub>k <sub>ver</sub>ifi<sub>e</sub>d f<sub>orm</sub> <sub>an</sub>d <sub>ava</sub>il<sub>a</sub>bilit<sub>y,</sub> <sub>no</sub>t th<sub>e</sub> <sub>o</sub>bli<sub>ga</sub>ti<sub>on.</sub>

## 6.4 Edit budget

O<sub>n</sub> t<sub>en o</sub>fi<sub>c</sub>i<sub>a</sub>l <sub>m</sub>i<sub>gra</sub>ti<sub>on sources, a s</sub>t<sub>a</sub>t<sub>e</sub>d t<sub>wo-</sub>fi<sub>e</sub>ld <sub>e</sub>dit b<sub>u</sub>d<sub>ge</sub>t <sub>pro</sub>d<sub>uce</sub>d <sub>a</sub> l<sub>oca</sub>l<sub>-repa</sub>i<sub>r a</sub>d<sub>van</sub>t<sub>age</sub> th<sub>a</sub>t di<sub>sappeare</sub>d <sub>un</sub>d<sub>er</sub> <sub>a</sub> th<sub>ree-</sub>fi<sub>e</sub>ld b<sub>u</sub>d<sub>ge</sub>t<sub>:</sub> 21/24 <sub>eac</sub>h i<sub>n</sub> <sub>a</sub> <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> <sub>rep</sub>l<sub>ay.</sub> A <sub>prospec</sub>ti<sub>ve</sub> <sub>rep</sub>li<sub>-</sub> cation on a 12-source bank (A<sub>pp</sub>endix Table 20) is consistent with this readin<sub>g</sub>. With the bud<sub>g</sub>et unlimited, full and <sub>p</sub>atch out<sub>p</sub>ut were indistin<sub>g</sub>uishable for GPT-5.6-luna (48/48 versus 47/48). U<sub>n</sub>d<sub>er</sub> th<sub>e</sub> t<sub>wo-</sub>fi<sub>e</sub>ld b<sub>u</sub>d<sub>ge</sub>t th<sub>e mo</sub>d<sub>e</sub>l <sub>a</sub>b<sub>s</sub>t<sub>a</sub>i<sub>ne</sub>d i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>excee</sub>di<sub>ng</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> Th<sub>e presen</sub>t <sub>s</sub>t<sub>u</sub>d<sub>y</sub> th<sub>ere</sub>f<sub>ore s</sub>t<sub>a</sub>t<sub>es an un</sub>li<sub>m</sub>it<sub>e</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub>

## 7 Discussion

What makes a release gate adequate here is its inputs, not its assertion family. On typical in<sub>p</sub>uts<sub>,</sub> the all-assertions <sub>g</sub>ate released 0.410 of wron<sub>g</sub> re<sub>p</sub>airs. It released 0.065 once one in<sub>p</sub>ut <sub>exerc</sub>i<sub>se</sub>d th<sub>e</sub> d<sub>ocumen</sub>t<sub>e</sub>d <sub>c</sub>h<sub>ange.</sub> Th<sub>e e</sub>f<sub>ec</sub>t <sub>rep</sub>li<sub>ca</sub>t<sub>e</sub>d <sub>on</sub> 15 <sub>c</sub>l<sub>us</sub>t<sub>ers</sub> b<sub>u</sub>ilt <sub>a</sub>ft<sub>er</sub> it <sub>was</sub> fi<sub>rs</sub>t <sub>o</sub>b<sub>serve</sub>d<sub>.</sub> It held in ever<sub>y</sub> o<sub>p</sub>en-wei<sub>g</sub>ht famil<sub>y,</sub> and it cost no correct com<sub>p</sub>letions.

The inputs can come from a blind frontier-model writer. The objection to the earlier study <sub>was</sub> th<sub>a</sub>t th<sub>e</sub> <sub>au</sub>th<sub>ors</sub> <sub>wro</sub>t<sub>e</sub> b<sub>o</sub>th th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> i<sub>npu</sub>t<sub>s</sub> <sub>an</sub>d th<sub>e</sub> hidd<sub>en</sub> i<sub>npu</sub>t<sub>s</sub> f<sub>rom</sub> th<sub>e</sub> <sub>same</sub> <sub>no</sub>t<sub>e.</sub> A <sub>wr</sub>it<sub>er</sub> th<sub>a</sub>t <sub>saw</sub> <sub>ne</sub>ith<sub>er</sub> th<sub>e</sub> <sub>program</sub> <sub>nor</sub> th<sub>e</sub> hidd<sub>en</sub> i<sub>npu</sub>t<sub>s</sub> <sub>reac</sub>h<sub>e</sub>d 0<sub>.</sub>070 <sub>w</sub>ith th<sub>ree</sub> i<sub>npu</sub>t<sub>s,</sub> <sub>a</sub>t <sub>one</sub> <sub>ca</sub>ll <sub>per</sub> task. For <sub>p</sub>ractitioners this su<sub>gg</sub>ests a chea<sub>p</sub> <sub>p</sub>re-release ste<sub>p</sub>:

(1) ask an isolated model for a few release-note ed<sub>g</sub>e in<sub>p</sub>uts;

(2) com<sub>p</sub>ute their ex<sub>p</sub>ected out<sub>p</sub>uts on the old runtime;

Table 16. The three public tasks that met the frozen visible-completion rule (“the main request completes without an error status and every declared output field can be derived”) but failed the upstream validator. Columns are read mechanically from retained records; the validator’s per-predicate reasons were stored only as hashes, so which predicates failed is unknown and left blank. Exploratory, post hoc.
<table><tr><td></td><td></td><td></td><td>Task predicates partial score candidate operations verdicts</td><td></td><td>decidable missing obligation</td><td></td></tr><tr><td>w02</td><td>4</td><td></td><td>reused; edits 0</td><td>0.75 1 call(s); auth header visible: yes; upstream: fail</td><td>1 re-authentication via /v3/oauth/token requested but not performed(stale</td><td></td></tr><tr><td>w08</td><td>3</td><td></td><td>reused; edits 0</td><td>0.67 1 call(s); auth header visible: yes; upstream: fail re-authentication via</td><td>/v3/oauth/token requested but not performed(stale</td><td>the mock)</td></tr><tr><td>w14</td><td>2</td><td></td><td>reused; edits 0</td><td>0.50 1 call(s); auth header visible: yes; upstream: fail</td><td>the mock) /v3/oauth/token performed(stale bearer accepted by</td><td>1 re-authentication via requested but not</td></tr></table>

(3) <sub>g</sub>ate the re<sub>p</sub>air on them.

Do not let the writer decide which old outputs still bind. Our preregistered guard asked the writer to fla<sub>g</sub> in<sub>p</sub>uts on which the le<sub>g</sub>ac<sub>y</sub> out<sub>p</sub>ut is no lon<sub>g</sub>er re<sub>q</sub>uired. It discarded exactl<sub>y</sub> the inputs t<sup>h</sup>at revea<sup>l</sup>e<sup>d</sup> t<sup>h</sup>e c<sup>h</sup>ange. It a<sup>l</sup>so prevente<sup>d</sup> no <sup>f</sup>a<sup>l</sup>se rejection, <sup>b</sup>ecause t<sup>h</sup>e unguar<sup>d</sup>e<sup>d</sup> inputs cause<sup>d</sup> none. In t<sup>h</sup>e ear<sup>l</sup>ier stu<sup>d</sup>y, LLM inputs <sup>d</sup>i<sup>d</sup> over-reject. T<sup>h</sup>ere t<sup>h</sup>e cause was runtime <sup>f</sup>acts th<sub>a</sub>t <sub>no repa</sub>i<sub>r may preserve, w</sub>hi<sub>c</sub>h <sub>a s</sub>t<sub>r</sub>i<sub>c</sub>t<sub>er va</sub>lidit<sub>y c</sub>h<sub>ec</sub>k <sub>on</sub> th<sub>e o</sub>ld <sub>run</sub>ti<sub>me may</sub> h<sub>an</sub>dl<sub>e</sub> b<sub>e</sub>tt<sub>er</sub> t<sup>h</sup>an a mo<sup>d</sup>e<sup>l’</sup>s ju<sup>d</sup>gement.

Who writes the tests may matter more than whether they are independent. A natural h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> i<sub>s</sub> th<sub>a</sub>t <sub>a</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>own</sub> t<sub>es</sub>t<sub>s</sub> <sub>s</sub>h<sub>are</sub> it<sub>s</sub> <sub>m</sub>i<sub>srea</sub>di<sub>ng</sub> <sub>o</sub>f th<sub>e</sub> <sub>re</sub>l<sub>ease</sub> <sub>no</sub>t<sub>e.</sub> O<sub>ur</sub> <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> d<sub>a</sub>t<sub>a</sub> do not su<sub>pp</sub>ort it. On the 21 tasks where ever<sub>y</sub> writer has tests<sub>,</sub> the stron<sub>g</sub>est writer <sub>g</sub>ates best<sub>,</sub> and same-model cells are not worse than cross-model cells (0.161 versus 0.187). CodeT ranks <sub>p</sub>ro<sub>g</sub>rams by agreement with tests from the same model [2]. Our results are consistent with that design workin<sub>g</sub> when the model is stron<sub>g</sub>. For release <sub>g</sub>atin<sub>g,</sub> a weak re<sub>p</sub>airer <sub>g</sub>ains most from a stron<sub>g</sub> test writer<sub>,</sub> whichever model re<sub>p</sub>airs. This matrix is <sub>p</sub>ost hoc and uses one <sub>p</sub>rom<sub>p</sub>t<sub>;</sub> it is a h<sub>yp</sub>othesis for <sub>a prereg</sub>i<sub>s</sub>t<sub>ere</sub>d t<sub>es</sub>t<sub>, no</sub>t <sub>a con</sub>fi<sub>rme</sub>d <sub>resu</sub>lt<sub>.</sub>

Stronger models need the gate less, but not never. The absolute benefit of the edge input <sub>s</sub>h<sub>ran</sub>k <sub>w</sub>ith <sub>capa</sub>bilit<sub>y.</sub> GPT<sub>-</sub>5<sub>.</sub>6<sub>-</sub>l<sub>una</sub> <sub>pro</sub>d<sub>uce</sub>d f<sub>ew</sub> <sub>wrong</sub> <sub>re</sub>l<sub>eases</sub> i<sub>n</sub> <sub>any</sub> <sub>arm.</sub> Th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> b<sub>ene</sub>fit sta<sub>y</sub>ed hi<sub>g</sub>h<sub>,</sub> and the frontier model still shi<sub>pp</sub>ed 22 wron<sub>g</sub> re<sub>p</sub>airs on t<sub>yp</sub>ical in<sub>p</sub>uts versus 0 with b<sub>o</sub>th i<sub>npu</sub>t <sub>sources.</sub>

Table 17. Threats to validity and how they are addressed.
<table><tr><td>Threat</td><td>What we did</td><td>What remains</td></tr><tr><td>Author-written edge and hidden inputs</td><td>blind writer that never sees program or hidden inputs; 3 of 176 inputs equal a hidden input</td><td>hidden inputs are still author-written</td></tr><tr><td>Effect seen before it was tested</td><td>fresh replication on 15 new clusters; freeze copies of the design text</td><td>M4c computable from the build before F0 (disclosed)</td></tr><tr><td>Contracts stricter than requirements</td><td>two blind AI audits per contract; post-hoc removal of p0 and do changes 0 verdicts</td><td>no human review; R2 class-label objections unresolved</td></tr><tr><td>One model in three roles</td><td>writer × repairer matrix (post hoc); two further preregistered writers (qwen3.6, Qwen3-8B)</td><td>only GPT-5.6-luna&#x27;s writer passed; auditor and writer are the same model</td></tr><tr><td>Menu composition</td><td>hard-distractor stage (233 legacy-matching distractors); menu-independent counts</td><td>distractors remain author-generated by documented operators</td></tr><tr><td>Strong-evidence criterion</td><td>reported against its frozen definition</td><td>not met: M4a not positive; audit flags &gt;10%</td></tr><tr><td>Finite repair menu</td><td>second frozen stage with 1491 documented fragments; exhaustive model-free measurement</td><td>still author-written fragments; 25 tasks below 15 options; free-form repair not run</td></tr><tr><td>Model coverage</td><td>four admitted families from three providers; power analysis</td><td>frontier model at ceiling; gemma not admitted</td></tr><tr><td>Contamination</td><td>post-cutoff SQLite 3.54 subset; no-documentation probe</td><td>public Python and Node.js notes</td></tr><tr><td>Ecosystem breadth</td><td>three lineages, 60 documented changes</td><td>no web APIs; no further lineages</td></tr></table>

## 8 Threats to Validity

T<sub>a</sub>bl<sub>e</sub> 17 <sub>summar</sub>i<sub>zes</sub> th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> th<sub>rea</sub>t<sub>s</sub> <sub>an</sub>d <sub>w</sub>h<sub>a</sub>t thi<sub>s</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> d<sub>oes</sub> <sub>a</sub>b<sub>ou</sub>t <sub>eac</sub>h<sub>.</sub> Construct validity.

• Hidden correctness treats le<sub>g</sub>ac<sub>y</sub> out<sub>p</sub>ut as <sub>p</sub>art of the obli<sub>g</sub>ation. The AI audit jud<sub>g</sub>ed several contracts stricter than their re<sub>q</sub>uirements<sub>,</sub> mainl<sub>y</sub> throu<sub>g</sub>h the file-<sub>p</sub>reservation check. Removin<sub>g</sub> it<sub>, an</sub>d l<sub>egacy equa</sub>lit<sub>y as we</sub>ll<sub>, c</sub>h<sub>ange</sub>d <sub>no ver</sub>di<sub>c</sub>t<sub>.</sub>

• The R2 objections, that a <sub>q</sub>uote documents a de<sub>p</sub>recation histor<sub>y</sub> rather than the removal, are <sub>no</sub>t <sub>reso</sub>l<sub>ve</sub>d b<sub>y</sub> th<sub>a</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub> W<sub>e repor</sub>t th<sub>e</sub> fl<sub>agge</sub>d t<sub>as</sub>k<sub>s</sub> i<sub>n</sub> th<sub>e ar</sub>tif<sub>ac</sub>t<sub>s.</sub>

• No human reviewed the contracts.

• Onl<sub>y</sub> the frontier-model writer (GPT-5.6-luna) <sub>p</sub>assed its tests; both local-model writers failed th<sub>e</sub>i<sub>rs, so</sub> th<sub>e wr</sub>it<sub>er resu</sub>lt i<sub>s spec</sub>ifi<sub>c</sub> t<sub>o a</sub> f<sub>ron</sub>ti<sub>er mo</sub>d<sub>e</sub>l<sub>.</sub>

• GPT-5.6-luna is the blind writer, the auditor and one of the re<sub>p</sub>airers. The audit ma<sub>y</sub> share the writer’s readin<sub>g</sub> of the release notes. The writer’s advanta<sub>g</sub>e on its own re<sub>p</sub>airs is not an i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t t<sub>es</sub>t<sub>, an</sub>d th<sub>e ma</sub>t<sub>r</sub>i<sub>x s</sub>h<sub>ows</sub> th<sub>e wr</sub>it<sub>er</sub>’<sub>s</sub> l<sub>ea</sub>d h<sub>o</sub>ld<sub>s on</sub> th<sub>e o</sub>th<sub>er</sub> th<sub>ree</sub> f<sub>am</sub>ili<sub>es.</sub>

Internal validity.

• The authors wrote the tasks, t<sub>yp</sub>ical in<sub>p</sub>uts, author ed<sub>g</sub>e in<sub>p</sub>uts and hidden in<sub>p</sub>uts. Under the QA <sub>ru</sub>l<sub>e,</sub> th<sub>e au</sub>th<sub>or e</sub>d<sub>ge</sub> i<sub>npu</sub>t <sub>mus</sub>t <sub>ma</sub>k<sub>e</sub> th<sub>e s</sub>t<sub>a</sub>l<sub>e proce</sub>d<sub>ure</sub> f<sub>a</sub>il <sub>v</sub>i<sub>s</sub>ibl<sub>y.</sub> Th<sub>e</sub> bli<sub>n</sub>d <sub>wr</sub>it<sub>er was no</sub>t b<sub>oun</sub>d b<sub>y</sub> th<sub>a</sub>t <sub>ru</sub>l<sub>e an</sub>d <sub>reac</sub>h<sub>e</sub>d <sub>a s</sub>i<sub>m</sub>il<sub>ar</sub> l<sub>eve</sub>l<sub>, w</sub>hi<sub>c</sub>h li<sub>m</sub>it<sub>s</sub> b<sub>u</sub>t d<sub>oes no</sub>t <sub>remove</sub> thi<sub>s concern.</sub>

• Model-free M4c was known at F0. M4a, M4b and the audit were known before an<sub>y</sub> re<sub>p</sub>air call. On<sub>e</sub> d<sub>esc</sub>ri<sub>p</sub>ti<sub>ve</sub> <sub>a</sub>m<sub>e</sub>ndm<sub>e</sub>nt f<sub>o</sub>ll<sub>owe</sub>d<sub>.</sub> Th<sub>e</sub> f<sub>u</sub>ll tim<sub>e</sub>lin<sub>e</sub> i<sub>s</sub> in A<sub>ppe</sub>ndix B<sub>.</sub>

## Statistical conclusion validity.

• The test <sub>p</sub>artition has 49 clusters. The frontier model and <sub>q</sub>wen3.6-35b-a3b are descri<sub>p</sub>tive b<sub>y</sub> <sub>p</sub>rere<sub>g</sub>istered rules.

• The scale anal<sub>y</sub>sis has four families.

• The cross-famil<sub>y</sub> thresholds were fixed for five families and a<sub>pp</sub>lied to the four admitted.

## External validity.

• Re<sub>p</sub>airs are choices amon<sub>g</sub> documented fra<sub>g</sub>ments. The free-form arm, in which models write <sub>co</sub>d<sub>e</sub> th<sub>a</sub>t i<sub>s</sub> th<sub>en</sub> <sub>execu</sub>t<sub>e</sub>d<sub>,</sub> <sub>was</sub> <sub>approve</sub>d <sub>on</sub>l<sub>y</sub> <sub>un</sub>d<sub>er</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>non-</sub>d<sub>es</sub>t<sub>ruc</sub>ti<sub>ve</sub> <sub>san</sub>db<sub>ox</sub>i<sub>ng.</sub> Th<sub>e</sub> <sub>san</sub>db<sub>ox</sub> <sub>runner</sub> <sub>cou</sub>ld <sub>no</sub>t b<sub>e</sub> <sub>crea</sub>t<sub>e</sub>d <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> <sub>mac</sub>hi<sub>ne</sub>’<sub>s</sub> <sub>perm</sub>i<sub>ss</sub>i<sub>ons,</sub> <sub>an</sub>d <sub>no</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ve</sub> h<sub>os</sub>t <sub>was use</sub>d<sub>.</sub> Th<sub>e arm was no</sub>t <sub>run, an</sub>d <sub>we</sub> d<sub>o no</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m</sub> th<sub>a</sub>t th<sub>e e</sub>f<sub>ec</sub>t<sub>s</sub> t<sub>rans</sub>f<sub>er</sub> t<sub>o</sub> f<sub>ree-</sub>f<sub>orm</sub> re<sub>p</sub>a<sup>i</sup>r.

• Three runtime linea<sub>g</sub>es share release ancestries within a linea<sub>g</sub>e. We make no claim about web API<sub>s o</sub>r <sub>o</sub>th<sub>e</sub>r <sub>ecosys</sub>t<sub>e</sub>m<sub>s.</sub>

Contamination. Python and Node.js release notes are public. The SQLite 3.54.0 subset post-dates the models’ likel<sub>y</sub> trainin<sub>g</sub>. The no-documentation <sub>p</sub>robe is descri<sub>p</sub>tive.

## Run conditions.

• Local runs <sub>p</sub>assed their admission checks.

• The launcher records onl<sub>y</sub> <sub>p</sub>oint readin<sub>g</sub>s of machine load.

• <sub>g</sub>emma-4-12b-<sub>q</sub>at was not admitted.

## 9 Conclusion

O<sub>n</sub> 60 d<sub>ocumen</sub>t<sub>e</sub>d <sub>c</sub>h<sub>anges</sub> i<sub>n</sub> th<sub>ree rea</sub>l <sub>run</sub>ti<sub>mes,</sub> th<sub>e re</sub>l<sub>ease c</sub>h<sub>ec</sub>k’<sub>s</sub> i<sub>npu</sub>t<sub>s</sub> d<sub>ec</sub>id<sub>e</sub>d <sub>w</sub>h<sub>e</sub>th<sub>er</sub> wron<sub>g</sub> LLM re<sub>p</sub>airs shi<sub>pp</sub>ed:

• one release-note ed<sub>g</sub>e in<sub>p</sub>ut cut the share of wron<sub>g</sub> re<sub>p</sub>airs released from 0.410 to 0.065 over the <sup>com</sup>p<sup>lete</sup> <sup>re</sup>p<sup>air</sup> <sup>s</sup>p<sup>ace</sup>;

• the efect re<sub>p</sub>licated on unseen chan<sub>g</sub>es, in three o<sub>p</sub>en-wei<sub>g</sub>ht families, on a re<sub>p</sub>air menu enlar<sub>g</sub>ed t<sub>o</sub> 1491 d<sub>ocumen</sub>t<sub>e</sub>d f<sub>ragmen</sub>t<sub>s,</sub> <sub>an</sub>d <sub>w</sub>ith 233 h<sub>ar</sub>d di<sub>s</sub>t<sub>rac</sub>t<sub>ors</sub> <sub>a</sub>dd<sub>e</sub>d<sub>;</sub>

• a blind frontier-model writer su<sub>pp</sub>lied in<sub>p</sub>uts of nearl<sub>y</sub> the same value without seein<sub>g</sub> the <sub>program, w</sub>hil<sub>e</sub> t<sub>wo</sub> l<sub>oca</sub>l<sub>-mo</sub>d<sub>e</sub>l <sub>wr</sub>it<sub>ers</sub> did <sub>no</sub>t <sub>pass;</sub>

• lettin<sub>g</sub> the writer decide which le<sub>g</sub>ac<sub>y</sub> out<sub>p</sub>uts still bind hel<sub>p</sub>ed less, and so did weaker models’ <sub>se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s;</sub> <sub>a</sub> <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> <sub>compar</sub>i<sub>son</sub> <sub>po</sub>i<sub>n</sub>t<sub>s</sub> t<sub>o</sub> <sub>wr</sub>it<sub>er</sub> <sub>capa</sub>bilit<sub>y</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> l<sub>ac</sub>k <sub>o</sub>f i<sub>n</sub>d<sub>epen</sub>d<sub>ence.</sub>

The contracts <sub>p</sub>assed an AI audit onl<sub>y</sub> in <sub>p</sub>art. The dis<sub>p</sub>uted <sub>p</sub>reservation and le<sub>g</sub>ac<sub>y</sub>-e<sub>q</sub>ualit<sub>y</sub> checks <sub>c</sub>h<sub>ange no ver</sub>di<sub>c</sub>t<sub>,</sub> th<sub>e c</sub>l<sub>ass-</sub>l<sub>a</sub>b<sub>e</sub>l di<sub>spu</sub>t<sub>es rema</sub>i<sub>n open, an</sub>d <sub>our pre</sub>d<sub>ec</sub>l<sub>are</sub>d <sub>s</sub>t<sub>rong-ev</sub>id<sub>ence</sub> <sub>c</sub>rit<sub>e</sub>ri<sub>o</sub>n i<sub>s</sub> n<sub>o</sub>t m<sub>e</sub>t<sub>.</sub> A fr<sub>ee</sub>-f<sub>o</sub>rm r<sub>epa</sub>ir <sub>a</sub>rm r<sub>e</sub>m<sub>a</sub>in<sub>s</sub> f<sub>u</sub>t<sub>u</sub>r<sub>e</sub> <sub>wo</sub>rk<sub>.</sub>

## Data Availability

Task s<sub>p</sub>ecifications, build records (tasks, visible oracles, hidden contracts and exhaustive verdicts), <sub>p</sub>inned release-note <sub>p</sub>a<sub>g</sub>es<sub>, p</sub>rom<sub>p</sub>ts<sub>,</sub> ever<sub>y</sub> model call with its res<sub>p</sub>onse<sub>,</sub> run recei<sub>p</sub>ts<sub>,</sub> freeze hash sets with verbatim desi<sub>g</sub>n co<sub>p</sub>ies<sub>,</sub> anal<sub>y</sub>sis out<sub>p</sub>uts and the scri<sub>p</sub>ts that render ever<sub>y</sub> number and fi<sub>gure</sub> <sub>are</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> <sub>au</sub>th<sub>ors</sub> <sub>on</sub> <sub>reques</sub>t<sub>.</sub> Th<sub>ey</sub> <sub>are</sub> <sub>no</sub>t <sub>pos</sub>t<sub>e</sub>d <sub>on</sub> <sub>a</sub> <sub>pu</sub>bli<sub>c</sub> <sub>co</sub>d<sub>e</sub> h<sub>os</sub>t<sub>.</sub> Th<sub>e</sub> hidd<sub>en</sub> <sub>con</sub>t<sub>rac</sub>t<sub>s</sub> <sub>an</sub>d hidd<sub>en</sub> i<sub>npu</sub>t<sub>s</sub> <sub>are</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> k<sub>ey</sub> <sub>o</sub>f th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>,</sub> <sub>an</sub>d <sub>pu</sub>bli<sub>s</sub>hi<sub>ng</sub> th<sub>em</sub> <sub>wou</sub>ld l<sub>e</sub>t f<sub>u</sub>t<sub>ure mo</sub>d<sub>e</sub>l<sub>s</sub> b<sub>e</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>or</sub> t<sub>une</sub>d <sub>on</sub> th<sub>em.</sub> Th<sub>e recor</sub>d<sub>s a</sub>l<sub>so con</sub>t<sub>a</sub>i<sub>n ver</sub>b<sub>a</sub>ti<sub>m responses</sub> <sub>o</sub>f <sub>a commerc</sub>i<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>serv</sub>i<sub>ce w</sub>h<sub>ose re</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> t<sub>erms we</sub> h<sub>ave no</sub>t <sub>c</sub>l<sub>eare</sub>d<sub>.</sub> W<sub>e w</sub>ill <sub>s</sub>h<sub>are</sub> th<sub>e</sub> com<sub>p</sub>lete <sub>p</sub>acka<sub>g</sub>e with reviewers and with researchers who re<sub>q</sub>uest it for re<sub>p</sub>roduction.

## Use of AI Tools

The stud<sub>y</sub> desi<sub>g</sub>n<sub>,</sub> <sub>p</sub>rere<sub>g</sub>istration decisions and inter<sub>p</sub>retation of results were made b<sub>y</sub> the authors. LLM-based codin<sub>g</sub> and writin<sub>g</sub> assistants were used to im<sub>p</sub>lement and test ex<sub>p</sub>eriment and anal<sub>y</sub>sis <sub>scr</sub>i<sub>p</sub>t<sub>s</sub> <sub>an</sub>d t<sub>o</sub> d<sub>ra</sub>ft <sub>an</sub>d <sub>e</sub>dit <sub>par</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> <sub>manuscr</sub>i<sub>p</sub>t<sub>;</sub> th<sub>e</sub> <sub>au</sub>th<sub>ors</sub> <sub>rev</sub>i<sub>ewe</sub>d <sub>a</sub>ll <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> <sub>an</sub>d t<sub>a</sub>k<sub>e</sub> f<sub>u</sub>ll <sub>respons</sub>ibilit<sub>y</sub> f<sub>or</sub> th<sub>e</sub> <sub>con</sub>t<sub>en</sub>t<sub>.</sub> AI t<sub>oo</sub>l<sub>s</sub> <sub>are</sub> <sub>no</sub>t <sub>au</sub>th<sub>ors.</sub> Th<sub>e</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d i<sub>n</sub> t<sup>h</sup>e experiments are o<sup>b</sup>jects o<sup>f</sup> stu<sup>d</sup>y, not aut<sup>h</sup>ors or assistants.

## References

[1] Mohammad Hasbi Assidi<sub>q</sub>i, Dani<sub>y</sub>al Al<sub>g</sub>hazzawi, Suaad Alarifi, and Li Chen<sub>g</sub>. 2026. Benchmarkin<sub>g</sub> Reference-Free LLM Agent Robustness Under Schema, Policy, and Toolset Drift. IEEE Access 14 (2026), 79662–79672. doi:10.1109 ACCESS.2026.3696096

[2] Bei Chen, Fen<sub>g</sub>ji Zhan<sub>g</sub>, Anh N<sub>g</sub>u<sub>y</sub>en, Dao<sub>g</sub>uan<sub>g</sub> Zan, Ze<sub>q</sub>i Lin, Jian-Guan<sub>g</sub> Lou, and Weizhu Chen. 2023. CodeT: Code Generation with Generated Tests. In International Conference on Learning Representations (ICLR). arXiv:2207.10397.

[3] Chaoran Chen, V<sub>y</sub> N<sub>g</sub>u<sub>y</sub>en, Ziji Zhan<sub>g</sub>, Abhinav Gulla<sub>p</sub>alli, Zi<sub>y</sub>i Wan<sub>g</sub>, Yuxuan Lu, Dakuo Wan<sub>g</sub>, and Jin<sub>g</sub> Huan<sub>g</sub>. 2026. Bench2Rob<sub>u</sub>st: Scenario-Controlled Reco<sub>v</sub>er<sub>y</sub> for Tool-Usin<sub>g</sub> A<sub>g</sub>ents. arXi<sub>v</sub>:2608.11977. arXi<sub>v</sub>:2608.11977

[4] Guoxin Chen, Zhon<sub>g</sub> Zhan<sub>g</sub>, Xin Con<sub>g</sub>, Fan<sub>g</sub>da Guo, Yesai Wu, Yankai Lin, Wenzhen<sub>g</sub> Fen<sub>g</sub>, and Yashen<sub>g</sub> Wan<sub>g</sub>. 2025. Learning Evolving Tools for Large Language Models. In International Conference on Learning Representations, Vol. 2025. OpenReview.net, Singapore, 87140–87169. https://proceedings.iclr.cc/paper\_files/paper/2025/hash d8d1d325f822<sub>e</sub>07d16866780<sub>cc</sub>23d<sub>e</sub>b2<sub>-</sub>Ab<sub>s</sub>t<sub>rac</sub>t<sub>-</sub>C<sub>on</sub>f<sub>erence.</sub>ht<sub>m</sub>l

[5] Juntao Chen, Tin<sub>g</sub>tin<sub>g</sub> Bi, Yanlin Wan<sub>g</sub>, and Patanamon Thon<sub>g</sub>tanunam. 2026. Who Broke Me? Execution-Guided Re<sub>p</sub>air of Behavioral De<sub>p</sub>endenc<sub>y</sub> Breaks. arXiv:2610.09267. arXiv:2610.09267 [cs.SE]

[6] Xin<sub>y</sub>un Chen, Maxwell Lin, Nathanael Schärli, and Denn<sub>y</sub> Zhou. 2024. Teachin<sub>g</sub> Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Models to Self-Debu<sub>g</sub>. In International Conference on Learning Representations (ICLR). arXiv:2304.05128.

[7] Wei Chen<sub>g</sub>, Yon<sub>g</sub>chan<sub>g</sub> Cao, Chen Shen, Binhua Li, Jue Chen, Yon<sub>g</sub>bin Li, and Wei Hu. 2026. To Dif or Not t<sub>o</sub> Dif? Str<sub>uc</sub>t<sub>u</sub>r<sub>e</sub>-A<sub>wa</sub>r<sub>e a</sub>nd Ad<sub>ap</sub>ti<sub>ve</sub> O<sub>u</sub>t<sub>pu</sub>t F<sub>o</sub>rm<sub>a</sub>t<sub>s</sub> f<sub>o</sub>r Efi<sub>c</sub>i<sub>e</sub>nt LLM-b<sub>ase</sub>d C<sub>o</sub>d<sub>e</sub> Editin<sub>g. a</sub>rXi<sub>v</sub>:2604<sub>.</sub>27296<sub>.</sub> arXiv:2604.27296 [cs.SE]

[8] Koen Claessen and John Hu<sub>g</sub>hes. 2000. QuickCheck: A Li<sub>g</sub>htwei<sub>g</sub>ht Tool for Random Testin<sub>g</sub> of Haskell Pro<sub>g</sub>rams. In Proceedings of the Fifth ACM SIGPLAN International Conference on Functional Programming. Association for Computing M<sub>ac</sub>hin<sub>e</sub>r<sub>y,</sub> N<sub>ew</sub> Y<sub>o</sub>rk<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 268–279<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/351240<sub>.</sub>351266

[9] Barthélem<sub>y</sub> Da<sub>g</sub>enais and Martin P. Robillard. 2009. SemDif: Anal<sub>y</sub>sis and Recommendation Su<sub>pp</sub>ort for API Evolution. In 2009 IEEE 31st International Conference on Software Engineering (ICSE). IEEE, Piscatawa , NJ, USA, 599–602<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1109/ICSE<sub>.</sub>2009<sub>.</sub>5070565

[10] Arghavan Moradi Dakhel, Amin Nikanjam, Vahid Majdinasab, Foutse Khomh, and Michel C. Desmarais. 2024. Efective test generation using pre-trained Large Language Models and mutation testing. Information and Software Technology 171 (2024), 107468. doi:10.1016/j.infsof.2024.107468

[11] Lukas Fruntke and Jens Krinke. 2025. Automatically Fixing Dependency Breaking Changes. Proceedings ofthe ACM on Software Engineering 2, FSE (2025), 2146–2168. doi:10.1145/3729366

[12] Patrice Godefroid, Daniel Lehmann, and Marina Polishchuk. 2020. Diferential Re<sub>g</sub>ression Testin<sub>g</sub> for REST APIs. In Proceedings of the 29th ACM SIGSOFT International Symposium on Software Testing and Analysis. Association for C<sub>o</sub>m<sub>pu</sub>tin<sub>g</sub> M<sub>ac</sub>hin<sub>e</sub>r<sub>y,</sub> N<sub>ew</sub> Y<sub>o</sub>rk<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 312–323<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/3395363<sub>.</sub>3397374

[13] Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wan<sub>g</sub>, Bokun Wan<sub>g</sub>, Yina Sa, Daren Zha, and Jun Xiao. 2026. ContractRL: Shielded Grou<sub>p</sub>-Relative Polic<sub>y</sub> O<sub>p</sub>timization for Auditable Tool-Call Re<sub>p</sub>air. arXiv:2610.00328. arXiv:2610.00328 [cs.AI]

[14] Dhanushka Ja<sub>y</sub>asuri<sub>y</sub>a, Valerio Terra<sub>g</sub>ni, Jens Dietrich, and Kell<sub>y</sub> Blincoe. 2024. Understandin<sub>g</sub> the Im<sub>p</sub>act of APIs Behavioral Breaking Changes on Client Applications. Proceedings of the ACM on Software Engineering 1, FSE (2024), 1238–1261. doi:10.1145/3643782

[15] Alibek Kali<sub>y</sub>ev and Artem Mar<sub>y</sub>ansk<sub>yy</sub>. 2026. Be<sub>y</sub>ond Task Com<sub>p</sub>letion: A Verification-vs.-Conformance Ga<sub>p</sub> in Tool-Evolvin<sub>g</sub> A<sub>g</sub>ents. arXiv:2604.00392. arXiv:2604.00392 [cs.SE]

[16] Jerz<sub>y</sub> Kamiński, Il<sub>y</sub>a Gal<sub>y</sub>ukshev, Artem Kuznetsov, Ser<sub>g</sub>e<sub>y</sub> Chu<sub>p</sub>rin, Kirill Redko, Aidar Shumbalov, and Anna K<sub>a</sub>l <sub>u</sub>zhn<sub>a a.</sub> 2026<sub>.</sub> D n<sub>a</sub>mi<sub>c</sub>MCPB<sub>e</sub>n<sub>c</sub>h: A Tr<sub>ace</sub>-Gr<sub>ou</sub>nd<sub>e</sub>d Ef<sub>ec</sub>t-S<sub>co</sub>r<sub>e</sub>d B<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk f<sub>o</sub>r LLM A <sub>e</sub>nt<sub>s ove</sub>r Li<sub>ve</sub> MCP Servers. arXiv:2607.20531. arXiv:2607.20531 [cs.AI]

[17] Obada Kraishan. 2026. Loud Failures, Quiet Failures: Fault Detection and Recover<sub>y</sub> in Tool-Usin<sub>g</sub> Lan<sub>g</sub>ua<sub>g</sub>e Model A<sub>ge</sub>nt<sub>s. a</sub>rXi<sub>v</sub>:2610<sub>.</sub>10062<sub>. a</sub>rXi<sub>v</sub>:2610<sub>.</sub>10062

[18] Sachit Kuhar, Wasi Uddin Ahmad, Zijian Wang, Nihal Jain, Haifeng Qian, Baishakhi Ray, Murali Krishna Ramanathan Xi<sub>ao</sub>f<sub>e</sub>i M<sub>a, a</sub>nd An<sub>oop</sub> D<sub>eo</sub>r<sub>as.</sub> 2025<sub>.</sub> LibE<sub>vo</sub>l<sub>u</sub>ti<sub>o</sub>nE<sub>va</sub>l: A B<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk <sub>a</sub>nd St<sub>u</sub>d<sub>y</sub> f<sub>o</sub>r V<sub>e</sub>r<sub>s</sub>i<sub>o</sub>n-S<sub>pec</sub>ifi<sub>c</sub> C<sub>o</sub>d<sub>e</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> In Proceedings of NAACL-HLT 2025 (Volume 1: Long Papers). 6826–6840. doi:10.18653/v1/2025.naacl-long.348

[19] Scott Lee. 2026. A Valid Tool Call Can Still Be an Invalid Efect. Queue 24, 5, Article 3839117 (2026). doi:10.1145/3839117

[20] Linbo Liu, Xinle Liu, Qian<sub>g</sub> Zhou, Lin Chen, Yihan Liu, Hoan N<sub>g</sub>u<sub>y</sub>en, Behrooz Omidvar-Tehrani, Xi Shen, Jun Huan, Omer Tri<sub>pp</sub>, and Anoo<sub>p</sub> Deoras. 2026. Mi<sub>g</sub>rationBench: Re<sub>p</sub>ositor<sub>y</sub>-Level Code Mi<sub>g</sub>ration Benchmark from Java 8. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2. doi:10.1145/3770855. 3817526

[21] Ze<sub>y</sub>u Leo Liu, Shre<sub>y</sub> Pandit, Xi Ye, Eunsol Choi, and Gre<sub>g</sub> Durrett. 2024. CodeU<sub>p</sub>dateArena: Benchmarkin<sub>g</sub> Knowled<sub>g</sub>e Editin<sub>g</sub> on API U<sub>p</sub>dates. arXiv:2407.06249. arXiv:2407.06249 [cs.CL]

[22] Victor May, Diganta Misra, Yanqi Luo, Anjali Sridhar, Justine Gehring, and Silvio Soares Ribeiro Junior. 2026. FreshBrew: A Benchmark for Evaluating AI Agents on Java Code Migration. In Proceedings of the IEEE/ACM 48th International Conference on Software Engineering (ICSE). doi:10.1145/3744916.3773214

[23] Diganta Misra, Nizar Islah, Victor May, Brice Rauby, Zihan Wang, Justine Gehring, Antonio Orvieto, Muawiz Sajjad Ch<sub>au</sub>dh<sub>a</sub>r<sub>y,</sub> EilifB<sub>.</sub> M<sub>u</sub>ll<sub>e</sub>r<sub>,</sub> Irin<sub>a</sub> Ri<sub>s</sub>h<sub>,</sub> S<sub>a</sub>mir<sub>a</sub> Ebr<sub>a</sub>himi K<sub>a</sub>h<sub>ou, a</sub>nd M<sub>ass</sub>im<sub>o</sub> C<sub>acc</sub>i<sub>a.</sub> 2026<sub>.</sub> GitCh<sub>a</sub>m<sub>e</sub>l<sub>eo</sub>n 2<sub>.</sub>0: E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> AI Code Generation Against Python Library Version Incompatibilities. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, San Die<sub>g</sub>o<sub>,</sub> California<sub>,</sub> United States<sub>,</sub> 46792–46831. doi:10.18653/v1/2026.acl-lon<sub>g</sub>.2170

[24] Gustave Monce, Thomas De<sub>g</sub>ueule, Jean-Rém<sub>y</sub> Falleri, and Romain Robbes. 2025. Client–Librar<sub>y</sub> Com<sub>p</sub>atibilit<sub>y</sub> Testin<sub>g</sub> with API Interaction Sna<sub>p</sub>shots. arXiv:2507.20814. arXiv:2507.20814 [cs.SE]

[25] Sunil Kumar Paidi and Venkata Ratna Kumar Bona<sub>g</sub>iri. 2026. ToolDriftBench: Detectin<sub>g</sub> and Recoverin<sub>g</sub> from MCP Tool Index Staleness in A entic S stems. In 2026 IEEE International Conference on Smart Internet ofThings (SmartIoT). IEEE<sub>,</sub> Sh<sub>e</sub>n <sub>a</sub>n <sub>,</sub> Chin<sub>a,</sub> 357–362<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1109/Sm<sub>a</sub>rtI<sub>o</sub>T70864<sub>.</sub>2026<sub>.</sub>00056

[26] Rachna Raj, Benoit Baudr<sub>y</sub>, and Die<sub>g</sub>o Elias Costa. 2026. BreakGuard: Towards Detectin<sub>g</sub> De<sub>p</sub>endenc<sub>y</sub> Breakin<sub>g</sub> Chan<sub>g</sub>es with LLM-Generated Tests. arXiv:2608.20167. arXiv:2608.20167 [cs.SE]

[27] Frank Re<sub>y</sub>es, Ma<sub>y</sub> Mahmoud, Federico Bono, Sarah Nadi, Benoit Baudr<sub>y</sub>, and Martin Mon<sub>p</sub>errus. 2026. B<sub>y</sub>am: Fixin<sub>g</sub> Breaking Dependency Updates with Large Language Models. Empirical Software Engineering 31, 4 (2026), 113. doi:10.1007/s10664-026-10835-1

[28] Paul Schmiedma<sub>y</sub>er, Andreas Bauer, and Bernd Brue<sub>gg</sub>e. 2023. Reducin<sub>g</sub> the Im<sub>p</sub>act of Breakin<sub>g</sub> Chan<sub>g</sub>es to Web Service Clients During Web API Evolution. In 2023 IEEE/ACM 10th International Conference on Mobile Software Engineering and Systems (MOBILESoft). IEEE, Piscataway, NJ, USA, 1–11. doi:10.1109/MOBILSoft59058.2023.00008

[29] Geor<sub>g</sub>-Daniel Schwarz, Felix Quast, and Dirk Riehle. 2025. Ensurin<sub>g</sub> S<sub>y</sub>ntactic Intero<sub>p</sub>erabilit<sub>y</sub> Usin<sub>g</sub> Consumer-Driven Contract Testing. Software Testing, Verification and Reliability 35, 5 (2025), e70006. doi:10.1002/stvr.70006

[30] Ser<sub>g</sub>io Se<sub>g</sub>ura, Gordon Fraser, Ana B. Sanchez, and Antonio Ruiz-Cortes. 2016. A Surve<sub>y</sub> on Metamor<sub>p</sub>hic Testin<sub>g</sub>. IEEE Transactions on Software Engineering 42, 9 (2016), 805–824. doi:10.1109/TSE.2016.2532875

[31] Edward K. Smith, Earl T. Barr, Claire Le Goues, and Yuri<sub>y</sub> Brun. 2015. Is the Cure Worse than the Disease? Overfittin<sub>g</sub> in Automated Program Repair. In Proceedings ofthe 10th Joint Meeting on Foundations ofSoftware Engineering (ESEC/FSE). 532–543. doi:10.1145/2786805.2786825

[32] Vasudev Vikram, Caroline Lemieux, Joshua Sunshine, and Rohan Padh<sub>y</sub>e. 2023. Can Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Models Write Good Pro<sub>p</sub>ert<sub>y</sub>-Based Tests? arXiv:2307.04346. arXiv:2307.04346 [cs.SE]

[33] Chon<sub>g</sub> Wan<sub>g</sub>, Kaifen<sub>g</sub> Huan<sub>g</sub>, Jian Zhan<sub>g</sub>, Yebo Fen<sub>g</sub>, L<sub>y</sub>u<sub>y</sub>e Zhan<sub>g</sub>, Yan<sub>g</sub> Liu, and Xin Pen<sub>g</sub>. 2025. LLMs Meet Librar<sub>y</sub> Evolution: Evaluating Deprecated API Usage in LLM-Based Code Completion. In Proceedings ofthe 47th IEEE/ACM International Conference on Software Engineering (ICSE). doi:10.1109/ICSE55347.2025.00245

[34] Guanchen<sub>g</sub> Wan<sub>g</sub>, Qin<sub>g</sub>hua Xu, Lionel Briand, and Kui Liu. 2026. Mutation-Guided Unit Test Generation With a Lar<sub>g</sub>e Language Model. IEEE Transactions on Software Engineering (2026). doi:10.1109/TSE.2026.3682975

[35] You Wan<sub>g</sub>, Michael Pradel, and Zhon<sub>g</sub>xin Liu. 2026. Are “Solved Issues” in SWE-bench Reall<sub>y</sub> Solved Correctl<sub>y</sub>? An Empirical Study. In Proceedings ofthe IEEE/ACM 48th International Conference on Software Engineering (ICSE). doi:10.1145/3744916.3764576

[36] Bin Wu, Ed<sub>g</sub>ar Meij, and Emine Yilmaz. 2026. Be<sub>y</sub>ond Static Toolsets: Self-Evolvin<sub>g</sub> LLM Tool A<sub>g</sub>ents via Continual Documentation Adaptation. In Findings ofthe Association for Computational Linguistics: ACL 2026. Association for Com<sub>pu</sub>tational Lin<sub>gu</sub>istics<sub>,</sub> San Die<sub>g</sub>o<sub>,</sub> California<sub>,</sub> United States<sub>,</sub> 21519–21539. doi:10.18653/<sub>v</sub>1/2026.findin<sub>g</sub>s-acl.1082

[37] Wei Wu, Yann-Gaël Guéhéneuc, Giuliano Antoniol, and Mir<sub>y</sub>un<sub>g</sub> Kim. 2010. AURA: A H<sub>y</sub>brid A<sub>pp</sub>roach to Identify Framework Evolution. In Proceedings ofthe 32nd ACM/IEEE International Conference on Software Engineering. A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>m<sub>pu</sub>tin<sub>g</sub> M<sub>ac</sub>hin<sub>e</sub>r<sub>y,</sub> N<sub>ew</sub> Y<sub>o</sub>rk<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 325–334<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/1806799<sub>.</sub>1806848

[38] Laerte Xavier, Aline Brito, André Hora, and Marco Túlio Valente. 2017. Historical and Im<sub>p</sub>act Anal<sub>y</sub>sis of API Breaking Changes: A Large-Scale Study. In 2017 IEEE 24th International Conference on Software Analysis, Evolution and Reengineering (SANER). IEEE, Piscataway, NJ, USA, 138–147. doi:10.1109/SANER.2017.7884616

[39] Yin<sub>g</sub>fei Xion<sub>g</sub>, Xin<sub>y</sub>uan Liu, Muhan Zen<sub>g</sub>, Lu Zhan<sub>g</sub>, and Gan<sub>g</sub> Huan<sub>g</sub>. 2018. Identif<sub>y</sub>in<sub>g</sub> Patch Correctness in Test-Based Program Repair. In Proceedings ofthe 40th International Conference on Software Engineering (ICSE). 789–799. doi:10.1145/3180155.3180182

[40] Bo<sub>y</sub>an<sub>g</sub> Yan<sub>g</sub>, Zijian Cai, Shunfu Jin, and Hao<sub>y</sub>e Tian. 2026. PAFT: Preservation Aware Fine-Tunin<sub>g</sub> for Minimal-Edit Pro<sub>g</sub>ram Re<sub>p</sub>air. arXiv:2604.03113. arXiv:2604.03113 [cs.SE]

[41] Shenghao Yang, Bo Lu, Yaochen Liu, Yu Kang, Qiongfang Zhang, Chetan Bansal, Saravan Rajmohan, and Minghua Ma. 2026. De<sub>p</sub>Re<sub>p</sub>air: LLM-Based So<sub>u</sub>rce-Code Re<sub>p</sub>air for De<sub>p</sub>endenc<sub>y</sub> Breakin<sub>g</sub> Chan<sub>g</sub>es. arXi<sub>v</sub>:2607.17957. arXiv:2607.17957 [cs.SE]

[42] Boxi Yu, Yuxuan Zhu, Pinjia He, and Daniel Kan<sub>g</sub>. 2025. UTBoost: Ri<sub>g</sub>orous Evaluation of Codin<sub>g</sub> A<sub>g</sub>ents on SWE-Bench. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 3762–3774. doi:10.18653/v1/2025.acl-long.189

[43] L<sub>y</sub>u<sub>y</sub>e Zhan<sub>g</sub>, Chen<sub>g</sub>wei Liu, Zhen<sub>g</sub>zi Xu, Sen Chen, Lin<sub>g</sub>lin<sub>g</sub> Fan, Bihuan Chen, and Yan<sub>g</sub> Liu. 2022. Has M<sub>y</sub> Release Disobeyed Semantic Versioning? Static Detection Based on Semantic Diferencing. In Proceedings ofthe 37th IEEE/ACM International Conference on Automated Software Engineering. Association for Computing Machinery, New York, NY, USA<sub>,</sub> 1–12<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/3551349<sub>.</sub>3556956

[44] Lianmin Zhen<sub>g</sub>, Wei-Lin Chian<sub>g</sub>, Yin<sub>g</sub> Shen<sub>g</sub>, Si<sub>y</sub>uan Zhuan<sub>g</sub>, Zhan<sub>g</sub>hao Wu, Yon<sub>g</sub>hao Zhuan<sub>g</sub>, Zi Lin, Zhuohan Li, Dachen<sub>g</sub> Li, Eric P. Xin<sub>g</sub>, Hao Zhan<sub>g</sub>, Jose<sub>p</sub>h E. Gonzalez, and Ion Stoica. 2023. Jud<sub>g</sub>in<sub>g</sub> LLM-as-a-Jud<sub>g</sub>e with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track. arXiv:2306.05685.

[45] Ton<sub>gy</sub>ao Zhu, Wei Hern Lim, and Min-Yen Kan. 2026. When Models Edit Too Much: On the Fidelit<sub>y</sub> of Minimal Code Edits. arXiv:2609.04061. arXiv:2609.04061 [cs.SE]

## A Tasks

Table 18. The 73 tasks. Class: S silent change, E explicit error, B no-safe-repair task, C unchanged control. Correct: hidden-correct procedures in the finite domain. Last two columns: whether the stale procedure fails a visible all-assertions check with typical inputs and with the author edge input added (by the edge-input QA rule always yes for afected tasks).
<table><tr><td>Task</td><td>Runtime</td><td>Origin</td><td>Class</td><td>Split</td><td>Correct</td><td>typical</td><td>edge</td></tr><tr><td>py-binascii-hqx</td><td>cpython</td><td>v2</td><td>B</td><td>test</td><td>0/6</td><td>yes</td><td>yes</td></tr><tr><td>py-distutils-looseversion</td><td>cpython</td><td>v2</td><td>B</td><td>test</td><td>0/18</td><td>yes</td><td>yes</td></tr><tr><td>py-shuffle-random-param</td><td>cpython</td><td>v2</td><td>B</td><td>test</td><td>0/6</td><td>yes</td><td>yes</td></tr><tr><td>py-control-csv</td><td>cpython</td><td>v2</td><td>C</td><td>development</td><td>1/4</td><td>no</td><td>no</td></tr><tr><td>py-control-json-canonical</td><td>cpython</td><td>v2</td><td>C</td><td>development</td><td>1/4</td><td>no</td><td>no</td></tr><tr><td>py-control-sha256</td><td>cpython</td><td>v2</td><td>C</td><td>test</td><td>2/4</td><td>no</td><td>no</td></tr><tr><td>py-collections-abc-alias</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>3/6</td><td>yes</td><td>yes</td></tr><tr><td>py-imp-removed</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>2/10</td><td>yes</td><td>yes</td></tr><tr><td>py-inspect-getargspec</td><td>cpython</td><td>v2</td><td>E</td><td>development</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-int-max-str-digits</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-locale-format</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>2/6</td><td>yes</td><td>yes</td></tr><tr><td>py-random-sample-set</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>2/6</td><td>yes</td><td>yes</td></tr><tr><td>py-randrange-float</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-safeconfigparser</td><td>cpython</td><td>v2</td><td>E</td><td>development</td><td>1/9</td><td>yes</td><td>yes</td></tr><tr><td>py-unittest-assertequals</td><td>cpython</td><td>v2</td><td>E</td><td>test</td><td>2/6</td><td>yes</td><td>yes</td></tr><tr><td>py-asyncio-timeouterror</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>11/32</td><td>no</td><td>yes</td></tr><tr><td>py-enum-contains-value</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>2/6</td><td>no</td><td>yes</td></tr><tr><td>py-intenum-str</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>2/7</td><td>yes</td><td>yes</td></tr><tr><td>py-intflag-invert</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>1/6</td><td>no</td><td>yes</td></tr><tr><td>py-mkdtemp-absolute</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>1/6</td><td>yes</td><td>yes</td></tr><tr><td>py-strenum-format</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>2/6</td><td>yes</td><td>yes</td></tr><tr><td>py-sum-neumaier</td><td>cpython</td><td>v2</td><td>S</td><td>test</td><td>1/6</td><td>no</td><td>yes</td></tr><tr><td>py-ssl-match-hostname</td><td>cpython</td><td>v3</td><td>B</td><td>development</td><td>0/5</td><td>yes</td><td>yes</td></tr><tr><td>py-unicodedata-15</td><td>cpython</td><td>v3</td><td>B</td><td>test</td><td>0/5</td><td>no</td><td>yes</td></tr><tr><td>py-asyncio-coroutine</td><td>cpython</td><td>v3</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-asyncio-loop-param</td><td>cpython</td><td>v3</td><td>E</td><td>development</td><td>3/15</td><td>yes</td><td>yes</td></tr><tr><td>py-configparser-readfp</td><td>cpython</td><td>v3</td><td>E</td><td>development</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-gettext-lgettext</td><td>cpython</td><td>v3</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-gzip-filename</td><td>cpython</td><td>v3</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-inspect-formatargspec</td><td>cpython</td><td>v3</td><td>E</td><td>development</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-parser-module</td><td>cpython</td><td>v3</td><td>E</td><td>test</td><td>3/15</td><td>yes</td><td>yes</td></tr><tr><td>py-re-inline-flags</td><td>cpython</td><td>v3</td><td>E</td><td>test</td><td>6/20</td><td>yes</td><td>yes</td></tr><tr><td>py-unittest-assertdictcontainssubset</td><td>cpython</td><td>v3</td><td>E</td><td>test</td><td>2/5</td><td>yes</td><td>yes</td></tr><tr><td>py-flag-iteration</td><td>cpython</td><td>v3</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>py-asynchat-producer</td><td>cpython</td><td>v4</td><td>E</td><td>test</td><td>2/15</td><td>yes</td><td>yes</td></tr><tr><td>py-chr-decimal-index</td><td>cpython</td><td>v4</td><td>E</td><td>development</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-formatter-dumbwriter</td><td>cpython</td><td>v4</td><td>E</td><td>development</td><td>1/10</td><td>yes</td><td>yes</td></tr><tr><td>py-os-bytearray-path</td><td>cpython</td><td>v4</td><td>E</td><td>development</td><td>2/5</td><td>yes</td><td>yes</td></tr><tr><td>py-pkgutil-impimporter</td><td>cpython</td><td>v4</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-random-seed-tuple</td><td>cpython</td><td>v4</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-sqlite3-optimizedunicode</td><td>cpython</td><td>V4</td><td>E</td><td>test</td><td>2/5</td><td>yes</td><td>yes</td></tr><tr><td>py-traceback-etype</td><td>cpython</td><td>v4</td><td>E</td><td>test</td><td>2/5</td><td>yes</td><td>yes</td></tr><tr><td>py-zipimport-find-module</td><td>cpython</td><td>v4</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>py-fromisoformat-legacy</td><td>cpython</td><td>v4</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>py-ipaddress-private</td><td>cpython</td><td>v4</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>py-math-pow-inf</td><td>cpython</td><td>v4</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>py-nan-hash-buckets</td><td>cpython</td><td>v4</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>py-tokenize-fstring</td><td>cpython</td><td>v4</td><td>S</td><td>test</td><td>2/5</td><td>no</td><td>yes</td></tr><tr><td>node-createcipher</td><td>node</td><td>v2</td><td>B</td><td>development</td><td>0/5</td><td>yes</td><td>yes</td></tr><tr><td>node-control-path-posix</td><td>node</td><td>v2</td><td>C</td><td>test</td><td>2/4</td><td>no</td><td>no</td></tr><tr><td>node-control-sha256</td><td>node</td><td>v2</td><td>C</td><td>test</td><td>1/4</td><td>no</td><td>no</td></tr><tr><td>node-fs-truncate-fd</td><td>node</td><td>v2</td><td>E</td><td>test</td><td>2/5</td><td>yes</td><td>yes</td></tr><tr><td>node-process-assert</td><td>node</td><td>v2</td><td>E</td><td>test</td><td>4/15</td><td>yes</td><td>yes</td></tr><tr><td>node-util-isdate</td><td>node</td><td>v2</td><td>E</td><td>test</td><td>3/6</td><td>yes</td><td>yes</td></tr><tr><td>node-buffer-max-length</td><td>node</td><td>v2</td><td>S</td><td>test</td><td>1/6</td><td>no</td><td>yes</td></tr><tr><td>node-dirent-path</td><td>node</td><td>v2</td><td>S</td><td>test</td><td>2/6</td><td>yes</td><td>yes</td></tr><tr><td>node-global-navigator</td><td>node</td><td>v2</td><td>S</td><td>test</td><td>3/6</td><td>yes</td><td>yes</td></tr><tr><td>node-stream-highwatermark</td><td>node</td><td>v2</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>node-zlib-bytesread</td><td>node</td><td>v2</td><td>S</td><td>development</td><td>2/5</td><td>yes</td><td>yes</td></tr><tr><td>node-createdecipher</td><td>node</td><td>v3</td><td>B</td><td>development</td><td>0/4</td><td>yes</td><td>yes</td></tr><tr><td>node-writable-highwatermark node-timers-active</td><td>node</td><td>v3</td><td>S</td><td>test</td><td>1/5</td><td>no</td><td>yes</td></tr><tr><td>node-buffer-poolsize</td><td>node</td><td>v4</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td></td><td>node</td><td>v4</td><td>S</td><td>test</td><td>3/5</td><td>no</td><td>yes</td></tr><tr><td>sqlite-control-group-sum</td><td>sqlite</td><td>v3</td><td>C</td><td>test</td><td>1/4</td><td>no</td><td>no</td></tr><tr><td>sqlite-control-json-extract</td><td>sqlite</td><td>v3</td><td>C</td><td>test</td><td>2/4</td><td>no</td><td>no</td></tr><tr><td>sqlite-dotcmd-semicolon</td><td>sqlite</td><td>v3</td><td>E</td><td>development</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>sqlite-go-terminator</td><td>sqlite</td><td>v3</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>sqlite-rowid-alias-column</td><td>sqlite</td><td>v3</td><td>E</td><td>test</td><td>1/5</td><td>yes</td><td>yes</td></tr><tr><td>sqlite-blob-zero-bytes</td><td>sqlite</td><td>v3</td><td>S</td><td>test</td><td>2/5</td><td>no</td><td>yes</td></tr><tr><td>sqlite-fp-17-digits</td><td>sqlite</td><td>v3</td><td>S</td><td>test</td><td>1/6</td><td>no</td><td>yes</td></tr><tr><td>sqlite-indexes-pattern sqlite-column-numeric-align</td><td>sqlite sqlite</td><td>v3 V4</td><td>S S</td><td>development test</td><td>1/5 1/5</td><td>no yes</td><td>yes yes</td></tr></table>

## B Freeze timeline and deviations

This study. All times are UTC on 2026-10-17.

<table><tr><td>Time</td><td>Event</td></tr><tr><td>03:10</td><td>F0 hash set written (351 files; design copy stored)</td></tr><tr><td>03:15</td><td>blind writer finished (73 calls)</td></tr><tr><td>03:24</td><td>AI audit finished (146 calls)</td></tr><tr><td>03:43</td><td>writer levels attached; model-free family M4 computed</td></tr><tr><td>03:48</td><td>F-test hash set written (794 files; design copy stored)</td></tr><tr><td>04:38</td><td>GPT-5.6-luna run finished</td></tr><tr><td>later</td><td>local windows: Qwen3-8B, gpt-oss-20b, qwen3.6-35b-a3b admitted; gemma-4-12b-qat not admitted</td></tr></table>

B<sub>e</sub>t<sub>ween</sub> F0 <sub>an</sub>d F<sub>-</sub>t<sub>es</sub>t<sub>,</sub> t<sub>wo</sub> h<sub>as</sub>h<sub>e</sub>d fil<sub>es</sub> <sub>c</sub>h<sub>ange</sub>d<sub>:</sub>

• the desi<sub>g</sub>n document <sub>g</sub>ained a disclosed amendment re<sub>p</sub>ortin<sub>g</sub> the writer and audit results;

• the evaluator <sub>g</sub>ained an ofline, descri<sub>p</sub>tive self-test <sub>g</sub>ate.

No h<sub>yp</sub>othesis<sub>,</sub> arm or decision rule chan<sub>g</sub>ed. Deviations:

• the 1,000-simulation <sub>p</sub>ower record was <sub>p</sub>roduced after F-test from the frozen scri<sub>p</sub>t;

• the cross-famil<sub>y</sub> thresholds are a<sub>pp</sub>lied to four instead of five families;

• the contract-strictness anal<sub>y</sub>sis without <sub>p</sub>0 and d0 is <sub>p</sub>ost hoc.

The large-menu stage. F0 (2c95616bd805; 192 files) was written on 2026-10-18 at 03:27 UTC, before the build and before an<sub>y</sub> call. F-test (db2f355dba79; 923 files) was written at 03:48 UTC, after th<sub>e</sub> b<sub>u</sub>ild <sub>an</sub>d th<sub>e mo</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree</sub> f<sub>am</sub>il<sub>y</sub> LM <sub>an</sub>d b<sub>e</sub>f<sub>ore any mo</sub>d<sub>e</sub>l <sub>ca</sub>ll<sub>.</sub> N<sub>o</sub> h<sub>as</sub>h<sub>e</sub>d fil<sub>e c</sub>h<sub>ange</sub>d b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo; on</sub>l<sub>y</sub> th<sub>e</sub> b<sub>u</sub>ild fil<sub>es were a</sub>dd<sub>e</sub>d<sub>.</sub> Th<sub>e mo</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree resu</sub>lt fil<sub>e</sub> it<sub>se</sub>lf li<sub>es ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> F<sub>-</sub>t<sub>es</sub>t h<sub>as</sub>h set; its in<sub>p</sub>uts (the build) are hashed. Both freezes store a co<sub>py</sub> of the addendum.

The hard-distractor stage. F0 (bf681a0b7291; 197 files) was written at 2026-10-19 02:22 UTC, before the build and before an<sub>y</sub> call. F-test (5f4ca8417951; 928 files) was written at 2026-10-19 02:54 UTC<sub>,</sub> <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> b<sub>u</sub>ild <sub>an</sub>d f<sub>am</sub>il<sub>y</sub> HD <sub>an</sub>d b<sub>e</sub>f<sub>ore</sub> <sub>any</sub> <sub>ca</sub>ll<sub>.</sub> N<sub>o</sub> h<sub>as</sub>h<sub>e</sub>d fil<sub>e</sub> <sub>c</sub>h<sub>ange</sub>d b<sub>e</sub>t<sub>ween</sub> th<sub>em.</sub>

The predecessor study. Between its two freezes, the design file and run driver changed. The thi<sub>r</sub>d f<sub>am</sub>il<sub>y</sub> <sub>an</sub>d it<sub>s</sub> <sub>s</sub>t<sub>rong-ev</sub>id<sub>ence</sub> <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> <sub>were</sub> <sub>a</sub>dd<sub>e</sub>d <sub>a</sub>ft<sub>er</sub> it<sub>s</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree</sub> f<sub>am</sub>il<sub>y</sub> h<sub>a</sub>d b<sub>een</sub> <sub>compu</sub>t<sub>e</sub>d<sub>.</sub> It<sub>s</sub> F0 d<sub>es</sub>i<sub>gn</sub> t<sub>ex</sub>t<sub>, no</sub>t <sub>arc</sub>hi<sub>ve</sub>d <sub>a</sub>t th<sub>e</sub> ti<sub>me, was recons</sub>t<sub>ruc</sub>t<sub>e</sub>d b<sub>y</sub>t<sub>e-exac</sub>tl<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> h<sub>as</sub>h<sub>es.</sub> Th<sub>e presen</sub>t <sub>s</sub>t<sub>u</sub>d<sub>y arc</sub>hi<sub>ves</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gn</sub> t<sub>ex</sub>t <sub>a</sub>t <sub>every</sub> f<sub>reeze an</sub>d h<sub>as</sub>h<sub>es</sub> th<sub>e</sub> f<sub>reeze scr</sub>i<sub>p</sub>t it<sub>se</sub>lf<sub>.</sub>

## C Evaluator qualification (deterministic study)

Table 19. Evaluator qualification (not a method comparison): the sealed B1 row and the post-audit B1C row contain an injected, scope-marker-triggered preservation perturbation. Their rejections show that the hidden preservation evaluator detects it.
<table><tr><td>Row</td><td>Joint</td><td>silent</td><td>discordant with B4</td></tr><tr><td>B1 (sealed)</td><td>88/100</td><td>12</td><td>8</td></tr><tr><td>B1C (post-audit)</td><td>88/100</td><td>一</td><td>8</td></tr></table>

T<sup>h</sup>e sea<sup>l</sup>e<sup>d</sup> B1 row an<sup>d</sup> t<sup>h</sup>e post-au<sup>d</sup>it B1C row contain an injecte<sup>d</sup>, scope-mar<sup>k</sup>er-triggere<sup>d</sup> preservation pertur<sup>b</sup>ation. T<sup>h</sup>eir rejections s<sup>h</sup>ow t<sup>h</sup>at t<sup>h</sup>e <sup>h</sup>i<sup>dd</sup>en preservation eva<sup>l</sup>uator <sup>d</sup>etects it. Th<sub>ey</sub> d<sub>o</sub> <sub>no</sub>t <sub>measure</sub> <sub>a</sub> <sub>repa</sub>i<sub>r-me</sub>th<sub>o</sub>d <sub>e</sub>f<sub>ec</sub>t<sub>.</sub>

## D Edit-budget replication (negative result)

B<sub>e</sub>h<sub>av</sub>i<sub>oura</sub>l f<sub>ee</sub>db<sub>ac</sub>k <sub>never</sub> t<sub>r</sub>i<sub>ggere</sub>d <sub>a rev</sub>i<sub>s</sub>i<sub>on.</sub> With <sub>an un</sub>li<sub>m</sub>it<sub>e</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> f<sub>u</sub>ll <sub>an</sub>d <sub>pa</sub>t<sub>c</sub>h <sub>ou</sub>t<sub>pu</sub>t <sub>were</sub> i<sub>n</sub>di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>a</sub>bl<sub>e;</sub> <sub>w</sub>ith <sub>a</sub> t<sub>wo-</sub>fi<sub>e</sub>ld b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> <sub>s</sub>h<sub>or</sub>tf<sub>a</sub>ll<sub>s</sub> <sub>appear</sub> <sub>as</sub> <sub>a</sub>b<sub>s</sub>t<sub>en</sub>ti<sub>ons.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith th<sub>e ear</sub>li<sub>er</sub> “l<sub>oca</sub>l <sub>w</sub>i<sub>ns</sub>” dif<sub>erence</sub> b<sub>e</sub>i<sub>ng a</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>ar</sub>tif<sub>ac</sub>t<sub>.</sub>

Table 20. Study B (appendix, descriptive): stated edit budget × representation on the 12-source minimalreproduction bank, Codex. Joint: hidden acceptance, preservation and forbidden-efect contracts all hold.
<table><tr><td>Budget</td><td>Repr.</td><td>Feedback</td><td>Joint</td><td>silent</td><td>abstain</td><td>mean edits</td></tr><tr><td>2</td><td>full</td><td>behavioral</td><td>45/48</td><td>0</td><td>3</td><td>1.24</td></tr><tr><td>2</td><td>full</td><td>none</td><td>45/48</td><td>0</td><td>3</td><td>1.24</td></tr><tr><td>2</td><td>patch</td><td>behavioral</td><td>46/48</td><td>0</td><td>2</td><td>1.28</td></tr><tr><td>2</td><td>patch</td><td>none</td><td>46/48</td><td>0</td><td>2</td><td>1.28</td></tr><tr><td>∞</td><td>full</td><td>behavioral</td><td>48/48</td><td>0</td><td>0</td><td>1.40</td></tr><tr><td>∞</td><td>full</td><td>none</td><td>48/48</td><td>0</td><td>0</td><td>1.40</td></tr><tr><td>∞</td><td>patch</td><td>behavioral</td><td>47/48</td><td>0</td><td>1</td><td>1.36</td></tr><tr><td>∞</td><td>patch</td><td>none</td><td>47/48</td><td>0</td><td>1</td><td>1.36</td></tr></table>

## E Reproducibility

E<sub>very</sub>thi<sub>ng</sub> b<sub>e</sub>l<sub>ow</sub> i<sub>s</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> <sub>au</sub>th<sub>ors</sub> <sub>on</sub> <sub>reques</sub>t<sub>.</sub> N<sub>o</sub>thi<sub>ng</sub> i<sub>s</sub> <sub>pu</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>on</sub> <sub>a</sub> <sub>pu</sub>bli<sub>c</sub> <sub>co</sub>d<sub>e</sub> h<sub>os</sub>t<sub>.</sub>

• Runtimes. CPython 3.9.6 (/usr/bin/python3) and 3.12.15 (Homebrew); Node.js 20.20.2 and 24.21.0; SQLite CLI 3.51.0 (s<sub>y</sub>stem) and 3.54.0 (Homebrew). Binar<sub>y</sub> hashes are recorded in each f<sub>reeze.</sub>

• Frozen materials. Task specifications, build records (tasks, visible oracles, hidden contracts, exhaustive verdicts), <sub>p</sub>inned release-note <sub>p</sub>a<sub>g</sub>es with retrieval time and SHA-256, <sub>p</sub>rom<sub>p</sub>ts, <sub>parser,</sub> <sub>ga</sub>t<sub>es,</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>co</sub>d<sub>e</sub> <sub>an</sub>d b<sub>o</sub>th f<sub>reeze</sub> h<sub>as</sub>h <sub>se</sub>t<sub>s</sub> <sub>w</sub>ith <sub>ver</sub>b<sub>a</sub>ti<sub>m</sub> d<sub>es</sub>i<sub>gn</sub> <sub>cop</sub>i<sub>es.</sub>

• Model records. Every call with prompt hash, response text, usage and timing. Also the qualifi-<sub>ca</sub>ti<sub>on</sub> <sub>recor</sub>d<sub>s</sub> <sub>o</sub>f <sub>a</sub>ll fi<sub>ve</sub> f<sub>am</sub>ili<sub>es,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> <sub>non-a</sub>d<sub>m</sub>itt<sub>e</sub>d <sub>one,</sub> th<sub>e</sub> <sub>wr</sub>it<sub>er</sub> <sub>an</sub>d <sub>au</sub>dit <sub>ou</sub>t<sub>pu</sub>t<sub>s,</sub> <sub>an</sub>d th<sub>e se</sub>lf<sub>-</sub>t<sub>es</sub>t<sub>s o</sub>f <sub>eac</sub>h f<sub>am</sub>il<sub>y.</sub>

• Derived records. The hidden evaluation, the independent checker’s receipts, per-run analyses, th<sub>e cross-</sub>f<sub>am</sub>il<sub>y ana</sub>l<sub>ys</sub>i<sub>s,</sub> th<sub>e pos</sub>t<sub>-</sub>h<sub>oc con</sub>t<sub>rac</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s, an</sub>d th<sub>e ren</sub>d<sub>er man</sub>if<sub>es</sub>t<sub>s</sub> th<sub>a</sub>t h<sub>as</sub>h ever<sub>y</sub> in<sub>p</sub>ut of ever<sub>y</sub> number and fi<sub>g</sub>ure in this <sub>p</sub>a<sub>p</sub>er.

• Replay without model access. All model-free results, the evaluation, the check and the figures <sub>can</sub> b<sub>e recompu</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e recor</sub>d<sub>s on</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>run</sub>ti<sub>mes w</sub>ith<sub>ou</sub>t <sub>any mo</sub>d<sub>e</sub>l <sub>ca</sub>ll<sub>.</sub>

## F Artifacts

R<sub>u</sub>n directories are create-onl<sub>y</sub>. An inde<sub>p</sub>endent checker recom<sub>pu</sub>tes e<sub>v</sub>er<sub>y</sub> release decision and hidd<sub>en</sub> <sub>ver</sub>di<sub>c</sub>t <sub>w</sub>ith <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>e</sub>d <sub>asser</sub>ti<sub>on</sub> <sub>seman</sub>ti<sub>cs;</sub> <sub>a</sub>ll f<sub>our</sub> <sub>a</sub>d<sub>m</sub>itt<sub>e</sub>d <sub>runs</sub> <sub>pass.</sub> All <sub>num</sub>b<sub>ers</sub> i<sub>n</sub> thi<sub>s paper are ren</sub>d<sub>ere</sub>d b<sub>y scr</sub>i<sub>p</sub>t<sub>s</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>rozen recor</sub>d<sub>s.</sub> T<sub>as</sub>k <sub>recor</sub>d<sub>s, con</sub>t<sub>rac</sub>t<sub>s,</sub> <sub>wr</sub>it<sub>er</sub> <sub>an</sub>d <sub>au</sub>dit <sub>ou</sub>t<sub>pu</sub>t<sub>s,</sub> <sub>run</sub> <sub>recor</sub>d<sub>s</sub> <sub>an</sub>d <sub>rece</sub>i<sub>p</sub>t<sub>s</sub> <sub>are</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> <sub>au</sub>th<sub>ors</sub> <sub>on</sub> <sub>reques</sub>t<sub>.</sub>