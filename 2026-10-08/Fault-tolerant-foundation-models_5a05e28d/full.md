# Fault-tolerant foundation models

Trevor McCourt,<sup>1,</sup> <sup>∗</sup> Ila R. Fiete,<sup>2</sup> and Isaac L. Chuang<sup>1,</sup> <sup>3</sup>

<sup>1</sup>Department of Electrical Engineering and Computer Science, MIT

<sup>2</sup>Department of Brain and Cognitive Sciences, McGovern Institute, MIT

<sup>3</sup>Department of Physics, MIT

(Dated: October 8, 2026)

Emerging computer hardware often trades reliability for energy eficiency; here we show that large-language models (LLMs) can be trained to tolerate this unreliability, and that rather than degrading, their error resilience actually increases as they grow. Modified neural scaling laws inferred from 40,000 GPU-hours of training runs on simulated faulty digital hardware quantify this trend and suggest that models learn to compute within “good” error-correcting codes, whose relative overhead remains finite no matter how large the model gets. This finding leads us to conjecture that appropriately trained LLMs may be formally fault-tolerant; if true, running AI inference on low energy, faulty hardware may be a path to substantial energy savings over the status quo.

Today’s AI systems demand perfectly reliable hardware, and that reliability costs energy. The chips that run commercial AI models [1, 2] suppress variability between individual logic gates by operating at high voltage [3]; the energy required to toggle a logic gate’s state scales with voltage squared. This energy–reliability tradeof is generic to any physical computing device and central to the thermodynamics of computation [4].

Fortunately, biology shows that this reliance on pristine components (and its energy penalty) is not fundamental: the brain, built from heterogeneous, noisy neurons [5, 6], leverages fault-tolerant circuit architectures to perform complex reasoning and control with orders of magnitude less energy than any existing artificial neural network. Grid cells, for example, compute within an error-correcting code: position is represented redundantly across modules of diferent spatial periods, so noise in any one module is inconsistent with the rest and can be corrected [7]. Downstream, the grid–hippocampal loop feeds back corrections that cancel slowly accumulating error [8–10].

AI could be far more energy eficient if run on faulty hardware: analog [11–13], low-voltage digital [14–17], spiking neuromorphic [18, 19], photonic [20, 21], and probabilistic [22–24] processors all trade absolute reliability for drastic gains in energy eficiency.

However, energy-eficient emerging hardware cannot yet be leveraged for AI at scale because commercial models are fragile: the advanced capabilities [25] of large language models (LLMs) often do not survive even quantization (rounding weights to fewer bits) [26–28]. Quantization, which perturbs every weight in a deterministic and bounded way, is far gentler than what real low-energy, fault-prone hardware presents (random errors of potentially arbitrary magnitude).

With energy a main bottleneck on AI deployment, developing models that can tolerate hardware errors is among the most important unsolved problems in AI. Computing reliably with unreliable components goes back to von Neumann [29], but was largely set aside once digital hardware became essentially error-free. Energyconstrained AI has revived it: by taking cues from the brain’s grid code, Zlokapa et al. [30] recently proved that neural networks can be constructed to compute any function reliably in the presence of errors (although not necessarily eficiently).

![](images/7cc73fa945ada5ae6888e4cb5f5472b233f530b30beed5e56fbcf38e6cea5226.jpg)  
FIG. 1. The emergence of fault tolerance in AI models as they grow. Each curve shows how much useful capacity a language model retains when trained and run on hardware that introduces errors into arithmetic operations with probability p (darker, more faults). Such a fault-hardened model with N weights performs as well as a conventional model with η weights run on clean hardware, so the useful capacity η/N is the fraction of the model doing useful work rather than correcting errors $( \mathrm { E q . ~ 1 ; }$ unlimited training data). Capacity first falls with size, then recovers beyond a critical size $N ^ { * }$ driven by a term in the scaling law that fault-free models lack $\left( \operatorname { E q . 4 } \right)$ . This is reminiscent of “good” error-correcting codes, whose overhead stays a fixed fraction at any scale, and we conjecture by analogy that $\eta / N$ approaches a nonzero constant whenever p lies below a threshold (Eq. 3). Solid curves and shaded bands show measured scaling-law fits and their bootstrap interquartile ranges; dashed curves show possible extrapolations, which depend on how much faults raise the best performance a model can ever attain (Eqs. 9 and 10).

The most pressing question concerning fault-tolerant neural network utility is whether models can be eficiently adapted to be insensitive to hardware errors at any size. AI models tend to get larger and more complicated over time, so it is critical to develop techniques that imbue deep learning systems with fault tolerance that improves (or at least does not worsen) as models grow. Encouragingly, some fault tolerance has been built into particular billion-weight models [31, 32], but only at a single scale: we do not know whether that behavior is part of a persistent trend or sits at the edge of a clif beyond which errors consume models.

Coding theory suggests that the scaling behavior of fault-tolerance in AI systems ought to be sharply defined. Nature protects information in two ways. The crude way is repetition: keep many copies of a value, as a cell does with duplicated genes, or as simple codes in the brain’s sensory periphery do by spreading information over large populations. The powerful way is a distributed, “good” code [33–35] such as what is used in the brain’s grid cells, in which the redundancy needed stays a fixed fraction of the whole no matter how large it grows [7]. If models can learn good codes, a finite fraction of the model performs useful computation at any scale; if they can only manage repetition, that fraction shrinks to zero as models grow. It is not obvious a priori that good codes can be built into transformer models by design or learned by training.

Here we take a critical first step toward running large language models on energy-eficient, unreliable hardware: we show that, unlike conventionally trained models, LLMs trained with the same faults they will later encounter at inference become increasingly error-resilient as they scale (Fig. 1). These results are consistent with models adopting “good” codes rather than simple repetition-based schemes; fault tolerance may therefore persist beyond the scales accessible to our experiments.

The trends in Fig. 1 arise because fault-hardened models (trained with the same faults they will meet in use) obey slightly diferent neural scaling laws than conventionally trained, fault-blind ones. Neural scaling laws [36–38] are the allometry of AI: just as metabolic rate follows a power law in body mass, a model’s prediction error falls as a power law in its number of weights and in its amount of training data. As we detail below, the scaling law of fault-hardened models gains an extra term that lets them improve faster than fault-blind ones at large scale.

We measure computational capacity by η, the size of a fault-blind model (trained and evaluated without faults) that is required to match the performance of a given fault-hardened model (trained and evaluated with

faults),

$$
L \left( 0 , \eta , D \right) = L \left( p , N , D \right)\tag{1}
$$

where $L \left( p , N , D \right)$ is the loss (prediction error on held-out text) of a model with N parameters whose components fault at rate $p ,$ trained on D tokens (word fragments).

The relative capacity $\eta / N$ is the neural network analog of a code’s rate: the fraction of the fault-hardened model’s parameters doing useful work rather than absorbing errors. The curves in Fig. 1 fit performance data from the thousands of fault-hardened models we trained, indicating $\eta / N$ in the converged limit $( D \to \infty )$ . The shape of these curves is derived from our newly identified scaling laws,

$$
\begin{array} { c } { { \displaystyle \ln \displaystyle \frac { \eta } { N } = - c _ { 0 } \left( p \right) - c _ { 1 } \left( p \right) n + c _ { 2 } \left( p \right) n ^ { 2 } - A ( p , N ) } } \\ { { n = \displaystyle \ln \displaystyle \frac { N } { N _ { 0 } } , N ^ { \ast } = N _ { 0 } \exp \left( \displaystyle \frac { c _ { 1 } \left( p \right) } { 2 c _ { 2 } \left( p \right) } \right) } } \end{array}\tag{2}
$$

where $N _ { 0 }$ is an arbitrary size scale and the $c _ { k }$ are simple functions of our scaling law coeficients (Eqs. 4 and 11). ln η/N is a parabola in log model size with its minimum at $N ^ { * }$ ; the quadratic term is what allows capacity to recover. $\mathcal { A }$ is an asymptotic term that controls behavior beyond our experimental range and is indistinguishable from 0 within it.

If models were learning to protect themselves from errors using repetition like schemes, we would expect $\eta / N$ to decrease smoothly to 0 with $N ;$ the fact that it instead increases past some critical size $N ^ { * }$ strongly suggests that models are learning “good” codes comparable to the brain’s grid code. If this is true, for suficiently weak errors, $\eta / N$ should asymptotically approach a fixed threshold. We return to this limiting behavior later in the manuscript.

It may be surprising that language models could so readily discover good codes without external guidance; however, this is exactly what the principle of errorcorrection-enhanced evolvability [39] predicts: adaptive processes (such as learning or evolution) are drawn towards fault-tolerant solutions because fault tolerance allows for more configurations to be explored without risking catastrophic damage. This work is the largest-scale evidence to date for that principle.

Because fault-hardened models show signs of learning to operate within good codes, we conjecture that they may be formally fault-tolerant, retaining finite capacity in the limit of infinite N,

$$
\operatorname* { l i m } _ { N , D \to \infty } \frac { \eta } { N } \ge c , \forall p < p _ { \mathrm { t h } }\tag{3}
$$

for $p _ { \mathrm { t h } } > 0$ and $0 < c \leq 1$ , where $p _ { \mathrm { t h } }$ is the fault-tolerance threshold (an analog of Eigen’s error threshold for replicating genomes [40]) and c is the asymptotic capacity retained.

![](images/6a47e478847481530c32c0b75e552c193d0f89f57d195e19b28d74e74b466668.jpg)

![](images/4a06b4d347105d48e42fd3eef024b03d56ffa6fdf770422a902021213269f619.jpg)  
FIG. 2. Training under faults changes how language models scale. Fault-hardened transformers obey a scaling law $\mathrm { ( E q . ~ 4 ) }$ with an extra coeficient compared to the standard form $( \alpha _ { 2 } ) ;$ each point fits all models trained at a fault rate $p$ (Methods). a, Model-size coeficients. $\alpha _ { 1 } .$ the typical model-size coeficient, shrinks as $p$ increases. $\alpha _ { 2 } ,$ the curvature term absent from standard laws (pinned at zero for $p = 0 )$ , grows with $p$ and drives the recovery of capacity beyond $N ^ { * }$ in ${ \mathrm { F i g . } }$ 1. b, Data coeficients. $\beta _ { 1 } ,$ the rate at which loss falls with training data, decreases as $p$ increases. c, Cost of robustness. Tokens $_ \mathrm { ~ D ~ a ~ }$ fault-hardened model needs to match a fault-free model trained on $\delta$ tokens (Eq. 6); gray shading marks extrapolation beyond our largest runs. Error bars, 95% confidence intervals; bands in $\mathbf { c } ,$ interquartile ranges; both from 1,500 bootstrap resamples per p cohort.

Our experiments reached $N \approx 1 0 ^ { 9 }$ , which was not enough to resolve A at commercial scales: establishing whether fault-hardened models satisfy Eq. 3 would require experiments orders of magnitude more expensive, which our results strongly suggest are worth doing.

By contrast, fault-blind models are strikingly fragile: on imperfect hardware their scaling behavior is ill-defined and divergent. This is consistent with recent findings that conventional models depend on a handful of outsized “super weights”[26], single points of failure whose number grows the longer a model is trained[41].

## The scaling behavior of fault-hardened transformers

Whether LLMs can tolerate faulty hardware at scale is, like an allometric law, a property of a whole population of models rather than of any single specimen: we established new scaling laws for fault-hardened models by training thousands of them with diferent N, D, and $p$ on a slice of FineWeb, a corpus of web text [42].

Our experiments emulated faulty digital hardware that randomly drops blocks of weights out of matrix products. For every forward pass during training or inference, random blocks of k matrix elements within the attention and feed-forward operations of our Llama2-style models were zeroed with probability p [43]. We fixed $k = 4$ throughout and varied $p$ to simulate more or less faulty hardware. Because the faults changed with every query, as synaptic failures change with every spike, fault-hardened models had to learn representations robust to a wide range of failure modes, which is distinct from and harder than calibrating a model to the fixed idiosyncrasies of a specific device [11, 44], the analog of adapting to a small local lesion. We describe the processor architecture behind this error model below.

Our critical finding was that, under this error model, fault-hardened transformers obey a modified form of the standard scaling law [38], with a new curvature coeficient $\alpha _ { 2 }$ that lets them close the performance gap to their faultfree counterparts as $N$ grows,

$$
\begin{array} { c } { { { \cal L } ( p , N , D ) = E \left( p \right) + { \cal L } _ { N } + { \cal L } _ { D } } } \\ { { { \cal L } _ { N } \left( p , N \right) = e ^ { \alpha _ { 0 } ( p ) - \alpha _ { 1 } ( p ) n - \alpha _ { 2 } ( p ) n ^ { 2 } } } } \\ { { { \cal L } _ { D } \left( p , D \right) = e ^ { \beta _ { 0 } ( p ) - \beta _ { 1 } ( p ) d } } } \\ { { { \displaystyle n = \ln \frac { N } { N _ { 0 } } , d = \ln \frac { D } { D _ { 0 } } } } } \end{array}\tag{4}
$$

Here $E$ is the irreducible loss floor, $\alpha _ { k }$ the model-size coeficients, and $\beta _ { k }$ the data coeficients. $N _ { 0 }$ and $D _ { 0 }$ are arbitrary scales that render the log arguments dimensionless, both fixed at $4 \times 1 0 ^ { 7 }$

The restoration of model capacity with size displayed in Fig. 1 is driven by the increase of $\alpha _ { 2 }$ with p (Fig. 2a), through the relations connecting the $\alpha _ { k }$ and $c _ { k }$ coeficients (Eq. 11, Methods).

The p-dependence of $\alpha _ { k }$ in Fig. 2a hints at two regimes of fault-hardened scaling, reminiscent of crossing a faulttolerance threshold. At small $p ,$ decreases in $\alpha _ { 1 }$ are mirrored by increases in $\alpha _ { 2 } .$ , whereas at large $p , \alpha _ { 2 }$ plateaus while $\alpha _ { 1 }$ continues to shrink. Fault-hardened models appear to learn to correct increasingly severe errors up to a point, before being overwhelmed, much as an induced stress response strengthens with the insult until the response itself is overwhelmed.

The asymptotic behavior of $\eta / N$ is controlled by ${ \mathcal { A } } ,$ which contributed negligibly at the scales accessible to our experiments: shifts in $E$ afect efective capacity only once they are of similar order to $L _ { N }$ , and $L _ { N } / E$ fell no lower than ∼ 0.5 even for our largest models. We return to the limiting behavior in the final section.

The data coeficients in Fig. 2b reflect an additional fact about fault-hardened scaling: models get more expensive to train as $p$ increases (Fig. 2c). Robustness in artificial neural networks, as in biology, is not free [45]. Let δ be the efective training budget of a faulty model, satisfying,

![](images/32c3d5850edf91e247734bae2b3613f253facd6832b795674d44a1f926f40c80.jpg)

![](images/17376a4e71501e17cafe4f1226eb738714c0c41ebf707c43712b88868d9034ee.jpg)

![](images/9a6ba409ffecaef9a5efa527d0b6afc8ce2193b74575aa844d8d0b9948d2e0eb.jpg)  
FIG. 3. Inference-time faults are fatal to models trained without them. ${ \mathbf { a } } ,$ The divergence of fault-blind model scaling under inference-time errors. Conventionally, we expect language model performance to increase monotonically with D. However, when $p _ { \mathrm { { e v a l } } } > 0$ , fault-blind models do not always exhibit this crucial behavior: performance scaling with $D$ weakens as $p _ { \mathrm { e v a l } }$ increases, and past a threshold, performance can actually decrease with $D .$ . Shaded regions indicate the standard error of the reported values, estimated from 8 independently trained models for each value of D. $\mathbf { b } ,$ Error-induced heating of model logits. Part of the degradation caused to models by inference-time errors is heating of the output logits: fault-blind mode predictions rise in temperature rapidly with $p _ { \mathrm { e v a l } }$ , more so at large D, while fault-hardened models resist the heating. $\mathbf { c } ,$ Errorinduced scrambling of model logits. Even after removing the temperature contribution, fault-blind models lose capacity rapidly as $p _ { \mathrm { e v a l } }$ rises above $0 ,$ faster at larger $D ;$ fault-hardened models maintain almost all capacity for $p _ { \mathrm { e v a l } } < p _ { \mathrm { t r a i n } }$ , independent of $D .$ Bands in (b) and (c) indicate the IQR of the reported values, taken over the diferent trained models and contexts. $\mathbf { d } ,$ Faulted predictions plotted against fault-free predictions, overlaid for many validation-set inputs. A temperature shift skews the logit blob: the $p _ { \mathrm { t r a i n } } = 0$ heating is obvious, along with substantial randomization, while the $p _ { \mathrm { t r a i n } } = p _ { \mathrm { e v a l } }$ logits are barely displaced.

$$
L _ { D } ( p , D ) = L _ { D } ( 0 , \delta )\tag{5}
$$

Under $\mathrm { E q . ~ 4 , ~ } D / \delta$ takes the form,

$$
\ln \frac { D } { \delta } = \frac { \beta _ { 0 } \left( p \right) - \beta _ { 0 } \left( 0 \right) } { \beta _ { 1 } \left( 0 \right) } + \left( 1 - \frac { \beta _ { 1 } \left( p \right) } { \beta _ { 1 } \left( 0 \right) } \right) d\tag{6}
$$

Since $\beta _ { 1 } \left( p \right)$ decreases with $p ,$ the second term’s coeficient is positive and increases with $p .$

Since our experiments were capped at $N \approx 1 0 ^ { 9 }$ , we did not achieve losses close to the irreducible floor and could not uniquely identify $E$ against the other fit parameters; we therefore fixed $E \approx 1 . 5$ in all fits (Methods), a choice with no qualitative impact on our results.

Indeed, while $\operatorname { E q }$ . 4 is useful for smoothing and interpreting our experiments, it is not strictly necessary for our main conclusions: the trends in Fig. 1 are fitindependent and can be seen in the (appropriately processed) raw data; see Supplementary Information Sec. $\mathrm { A } .$

## Fault-blind models collapse under inference-time errors

In contrast to fault-hardened models, fault-blind models, trained in a sterile, fault-free environment, learn fragile internal representations that fall apart under hardware errors. Fault-hardened models are trained at the fault rate they see at inference, $p = p _ { \mathrm { t r a i n } } = p _ { \mathrm { e v a l } } ;$ faultblind ones see $p _ { \mathrm { t r a i n } } = 0$ but can still be evaluated with $p _ { \mathrm { { e v a l } } } > 0$

The efect of inference-time faults on fault-blind models is not subtle: at suficiently large $p _ { \mathrm { e v a l } }$ , fault-blind models lack convergent scaling laws altogether. As shown in Fig. 3a, when $p _ { \mathrm { { e v a l } } } > 0$ , fault-blind transformer performance can actually decrease with $D _ { : }$ , and the value of D at which performance stops increasing seems to fall with p<sub>eval</sub>.

This catastrophic degradation is consistent with prior observations of super weights/activations in conventional transformers (features of outsized magnitude critical to a model’s functionality) [26]. From the perspective of error-resilience, super-features are single points of failure, the antithesis of the degeneracy that makes biologi cal networks robust [46]. A fault hitting a super-weight, or landing in the path of a super-activation, can degrade a model’s outputs to complete nonsense. Curiously, superfeature counts have been shown to grow with training duration [41]. This may be directly linked to our degraded scaling laws.

Part of the reason fault-blind models degrade as $p _ { \mathrm { e v a l } }$ increases is that the efective temperature (τ) of their output distribution rises. The term is borrowed from statistical mechanics and measures randomness, not heat:

at high $\tau ,$ predictions spread difusely over possible next tokens $x _ { t }$ rather than concentrating on the most likely one,

$$
\mathbb { P } \left( x _ { t } = k | x _ { t - 1 } = k _ { t - 1 } , \ldots , x _ { 0 } = k _ { 0 } \right) = \frac { e ^ { a _ { k } / \tau } } { \sum _ { j } e ^ { a _ { j } / \tau } }\tag{7}
$$

where the logits $a _ { k }$ depend explicitly on the context $( k _ { t - 1 } , \ldots , k _ { 0 } )$ . Eq. 7 is the divisive normalization familiar from sensory neuroscience [47], with $1 / \tau$ playing the role of neural gain.

The efective temperature of fault-blind models rises rapidly as $p _ { \mathrm { e v a l } }$ is raised from 0 (Fig. 3b), while fault-hardened ones hold a constant temperature until p<sub>eval</sub> exceeds $p _ { \mathrm { t r a i n } } \mathrm { . }$ : fault-hardened models are climatecontrolled, preserving the precision of their predictions under noise. The heating worsens for both model classes as D increases.

If the only efect of faults were to raise the temperature, the solution would be simple gain control: rescale the logits as a function of $p$ to restore τ to 1.

However, the temperature shift is accompanied by a significant irreversible scrambling of the logits. To be precise, the KL-divergence $D \left( \cdot | | \cdot \right)$ between the clean logit distribution $P$ and the faulted distribution $Q$ can be broken into two components,

$$
D \left( Q | P \right) = D \left( Q | P ^ { * } \right) + D \left( P ^ { * } | P \right)\tag{8}
$$

where $P ^ { * }$ is the nearest distribution to $Q$ that is a strict temperature shift of P (Methods).

The first term in Eq. 8 is the part of the gap between $P$ and $Q$ that no rescaling can close; Fig. 3c plots the capacity lost to this term as $p _ { \mathrm { e v a l } }$ increases. Notably, fault-hardened models lose almost no capacity until p<sub>eval</sub> exceeds $p _ { \mathrm { t r a i n } }$ , while fault-blind ones deteriorate rapidly from $p _ { \mathrm { e v a l } } = 0$ . Fig. 3d plots clean logits against their faulted counterparts, clarifying the distinct contributions of temperature rescaling vs. scrambling.

## Low-energy neural network accelerators

Our fault model emulated a Low Energy Neural Network Accelerator (LENNA) [17]: a chip that runs its arithmetic at voltages too low for perfect reliability, but detects its own errors and converts each one into a dropped connection (the artificial counterpart of a synapse that fails to transmit).

Our experiments were consistent with a hardware system in which LENNA cores execute the primary transformer matrix products (in the attention and feedforward layers), while all the supporting operations (nonlinearities, norms, positional encoding, and embed ding/readout layers) are run on some other fault-free sub strate; see Fig. 4 (a). As shown in Fig. 4 (b), matrix products overwhelmingly dominate the FLOP budget of scaled transformer models.

![](images/5002cae7a1c1d40f344c42cf221969cee788aa52ab891e31d72ea9854302a537.jpg)  
FIG. 4. Error-detecting hardware turns arbitrary arithmetic faults into dropped connections. a, The operations that faults targeted in our experiments. We injected errors into the matrix multiplications that drive the attention and feed-forward operations (orange) in a Llama-2-style transformer. b, The fraction of fault-targeted operations vs model size. Matrix multiplications (matmuls) account for almost all of the floating-point operations (FLOPs) required to run transformers; for large models, more than 99% of the FLOP budget. $\mathbf { c } ,$ Matrix products reduce to dot products. In hardware, each matmul breaks down into a series of dot products $u \cdot v .$ d, The error model. A chain of fused multiply–add (FMA) cells accumulates each dot product. Operands carry a redundant residue (their remainder modulo small bases), and after every $k = 4$ cells a cheap modulus check (%) tests the running sum for consistency. A failed check discards those $k$ products and carries the previous partial sum forward, equivalent to zeroing a block of k elements in one operand; blocks fail independently with probability p. Detection thus converts hardware errors of arbitrary magnitude into well-behaved block drop-connect, leaving correction to the model.

LENNAs build error detection into matrix products by encoding operands within a redundant residue number system (RRNS) [48, 49], where extra “check moduli” are carried through a matrix product so the correctness of results can be verified [50]. This is the same principle by which grid cells encode position: each module reports location modulo its spatial period, and redundancy across modules exposes errors [7]. In hardware, matrix–vector products are broken into dot products (Fig. 4c) computed by chains of fused multiply–add (FMA) circuits. In a LENNA, each FMA also processes the residues, and after every k FMAs, a check compares the residues with the primary result; if they disagree, those k operations are discarded, and the previous intermediate result is carried forward (Fig. 4d).

Importantly, LENNAs only detect errors; correcting them is left to the model it runs. This division of labor is eficient because, for RRNS arithmetic, detection is far cheaper than correction. An N-bit (array-style) FMA uses ∝ $N ^ { 2 }$ full adders; in a 12-bit example with $k = 4$ , carrying an extra 3-bit residue adds only a few percent, and the consistency check needs only residue-sized adders, negligible beside the FMAs. Full correction in hardware requires an algorithm based on the Chinese remainder theorem that could easily exceed the cost of the FMAs themselves. The arrangement echoes the brain’s, where grid modules expose errors cheaply, and correction is delegated to the downstream grid–hippocampal loop [7, 8].

## Towards formally fault-tolerant AI

The ultimate objective of fault-hardened neural network research should be to establish that Eq. 3 holds for realistic hardware fault models: that fault-hardened models running on actual faulty hardware retain capacity at arbitrary scale, as the brain evidently does. Practically, it would sufice to show that at some large enough scale $( \mathrm { e . g . , 1 0 ^ { 1 1 } }$ parameters), $\eta / N$ stays large enough for a fault-hardened model on low-energy faulty hardware to beat a fault-blind one on an existing accelerator by a margin that justifies the change of hardware.

If our scaling law holds asymptotically, we expect one of two behaviors, with p controlling a smooth transition between them. If the floor-gap $\Delta$ is much larger than the remaining size-loss $L _ { N }$ at scale, the capacity retained by the fault-hardened model collapses to 0,

$$
\ln { \frac { \eta } { N } } \propto - n\tag{9}
$$

Alternatively, if $\eta / N$ approaches 1 while the gap remains small compared to $L _ { N }$ , then ${ \mathcal { A } } ( p , N )$ becomes a negative correction that slows the growth of $\eta ,$ leading to saturation near $N _ { : }$

$$
\ A \left( p , N \right) \approx \frac { \Delta \left( p \right) } { \alpha _ { 1 } \left( 0 \right) L _ { N } \left( p , N \right) }\tag{10}
$$

The value of p at the boundary between collapse and saturation may be operationally defined as the faulttolerance threshold at a given experimental scale, much as the error threshold of a replicating genome separates heritable information from mutational meltdown [40].

## DISCUSSION

We have identified additional terms in the scaling laws for fault-hardened transformers that lead to large models performing surprisingly well (relative to what would be expected via simple error-correcting schemes like repetition).

To determine whether this behavior persists at the $\sim 1 0 ^ { 1 1 }$ weight scale, as our work hints is possible, models must be trained in that regime. We estimate that a reasonable extension of our experiment would cost around $1 0 ^ { 2 6 }$ FLOPs, or ∼ 10<sup>8</sup> GPU-hours (similar to the total training budget of Llama-3 405B [51]), which is intractable within academia.

Smaller experiments could de-risk the large one. Simple natural or synthetic datasets, where the asymptotic regime is easily reached, can serve as model organisms: fault-hardened transformers trained on them could be studied as a function of the dataset’s irreducible floor E. Fault tolerance that exists at small scale and persists as E increases would be strong evidence that commercialscale experiments are worth doing.

Further evidence that fault-hardened models learn good codes may be found in the structure of their weights and activations. Understanding how large neural networks work is notoriously hard [52, 53]. Still, there is precedent: networks trained to navigate spontaneously develop grid-cell-like codes [54, 55], and the stark con trast between how fault-blind and fault-hardened models respond to faults (Fig. 3) suggests an equally detectable diference in their internals. Our trained models, multiple terabytes of weights and optimizer states, are available to anyone who wants to look (see Data availability).

The brain is the existence proof that fault tolerance, energy eficiency, and intelligence are compatible, and the principle of error-correction-enhanced evolvability [39] holds that fault tolerance is an expected outcome of adaptation in a noisy environment. Our results suggest that AI trained under the constraints that shaped nervous systems may come to share their robustness. Additionally, hardware built with error-detection codes, as found in the brain, may provide the substrate for bounded-error, low-energy accelerator operation.

Establishing the commercial viability of fault-hardened networks will require substantial further investment, but the reward is commensurate: on future low-energy hardware, they could be orders of magnitude more energy eficient than any inference system existing today.

## METHODS

## Dataset

Every model was trained on a 350-billion-token slice of FineWeb [42], tokenized with an 8192-token bytepair-encoding vocabulary and packed into 1024-token sequences. The corpus was shufled in large blocks, so any contiguous slice is statistically equivalent to any other. Performance was measured on a held-out split of two billion tokens, drawn from the same web crawls but sharing no documents with the training data: near-duplicates were removed by a normalized-content hash, and the tokenizer was fit on training documents alone.

## Model ladder

All models are Llama2-style decoder-only transformers (rotary position embeddings, RMSNorm, and SwiGLU feed-forward blocks), following a standard reference implementation [56]. The ladder holds every architectural ratio fixed and varies only overall size: the attention head dimension is 32 at every rung (head count is width over 32, no grouped-query attention), the width-to-depth ratio is held near 32, the context length is 1024 tokens, and the embeddings are untied. Sixteen logarithmically spaced (width, depth) pairs range from $7 . 9 \times 1 0 ^ { 6 }$ to $9 . 3 \times 1 0 ^ { 8 }$ total parameters; throughout, N refers to total parameters, embeddings included.

One architecture (width 512, depth 13, $N = 5 . 0 \times 1 0 ^ { 7 } )$ was additionally trained at eight token budgets spanning two orders of magnitude, with eight random seeds per condition and six training fault rates. These runs provide the fixed-N measurements in Fig. 3.

## Fault injection

Faults were applied to the operands of the matrix products that dominate a transformer’s FLOP budget: the query, key, value, and output projections of the attention block, the two context products $Q K ^ { \mathsf { T } }$ and AV (K and V being the faulted operands), and the gate, up, and down projections of the feed-forward block. Each faulted operand was masked along its contraction axis in contiguous blocks of k elements, every block zeroed independently with probability $p ,$ which is exactly the efect of an error-detecting FMA chain discarding its last k accumulations (Fig. 4 (d)); blocks fail independently for every row of the product. The embedding lookup, readout, norms, rotary embeddings, nonlinearities, and residual additions were left fault-free. A fresh mask was drawn for every sequence on every forward pass and held fixed across positions within it, so a model trains against the whole fault distribution rather than one realization of it. Throughout, k was fixed at 4 and p took seventeen values between 0 and 0.2.

## Training

We used a typical training procedure and “conservative” hyperparameter choices lifted from a standard reference implementation [56]. More elegant tuning schemes [57–60] slightly improved the performance of large models as expected, but introduced systematic deviations from typical scaling laws that confounded our analysis; since all our measurements are diferential across p-values rather than concerned with absolute performance, the added complexity was not worthwhile.

The number of training tokens was set per run as a fixed multiple of the parameter count, giving token budgets from $7 . 9 \times 1 0 ^ { 7 }$ to $3 . 5 \times 1 0 ^ { 1 0 }$ . The two axes were not fully crossed: the ten architectures at or below $1 . 1 \times 1 0 ^ { 8 }$ parameters were each trained at 10, 20, 80, and 320 tokens per parameter, while the six larger ones were trained only at 20 and below, since completing the ladder at the top would have cost an order of magnitude more compute than the entire sweep. Every combination of size, duration, and fault rate was trained at two random seeds.

## Scaling-law fits

Equation 4 was fit to each p cohort separately, so no assumption about how the coeficients vary with p is built into the estimates. Fits were performed on log L, which weights runs evenly across the orders of magnitude the sweep spans, with a Huber loss to prevent a few oftrend points from dominating, and from multiple starting points because the objective is not convex. The pivots $N _ { 0 }$ and $D _ { 0 }$ are a gauge choice rather than fitted quantities, both fixed at $\bar { 4 } \times \bar { 1 } 0 ^ { 7 }$

Two quantities were held rather than fitted: the curvature $\alpha _ { 2 } .$ , pinned to zero for the fault-free reference cohort and fit freely for every other cohort, and the floor $E ,$ fixed at 1.55 for every cohort, the value preferred by the $p = 0$ cohort. The choice of E mainly rescales $\alpha _ { 1 } { : }$ fixing $E = 1 . 8 5$ instead gives $\alpha _ { 1 } \approx 0 . 3$ at $p = 0$ with no meaningful change to the other coeficients, consistent with the literature [38, 61, 62]. Deviations from published values are expected given the simpler dataset and smaller tokenizer used here.

Confidence intervals come from bootstrapping: the fitted runs were resampled with replacement 1500 times per cohort, and the surface was refit on each resample. Because the (N, D) grid is a design rather than a sample, a resample that drops the largest models loses the only constraint on the long-N end of the surface and reports its absence as uncertainty in the exponents; the intervals we quote are conservative for that reason.

The capacity relations of Eq. 2 follow from solving ${ \cal L } ( 0 , \eta , D ) = { \cal L } ( p , N , D )$ (Eq. 1) for η in the converged limit $D \to \infty$ . Expressed through the fitted coeficients

of Eq. 4,

$$
\begin{array} { c } { { c _ { 0 } = \displaystyle \frac { \alpha _ { 0 } \left( p \right) - \alpha _ { 0 } \left( 0 \right) } { \alpha _ { 1 } \left( 0 \right) } , ~ c _ { 1 } = 1 - \displaystyle \frac { \alpha _ { 1 } \left( p \right) } { \alpha _ { 1 } \left( 0 \right) } , ~ c _ { 2 } = \displaystyle \frac { \alpha _ { 2 } \left( p \right) } { \alpha _ { 1 } \left( 0 \right) } } } \\ { { A ( p , N ) = \displaystyle \frac { 1 } { \alpha _ { 1 } \left( 0 \right) } \ln \left( \displaystyle \frac { \Delta \left( p \right) } { L _ { N } \left( p , N \right) } + 1 \right) } } \\ { { \Delta \left( p \right) = E \left( p \right) - E \left( 0 \right) } } \end{array}\tag{11}
$$

The asymptotic term A depends on the floor gap $\Delta .$ which is indistinguishable from zero within our experimental range.

## Logit-space analysis

The measurements in Fig. 3 were made on a fixed set of 256 contexts, each the final token position of a held-out sequence. For each context and fault rate, we averaged the model’s next-token distribution over 1000 independent fault draws, giving the fault-averaged predictive $Q ;$ the fault-free distribution $P$ was recorded for the same contexts. This was repeated at 81 fault rates between 0.002 and 0.2.

A single efective temperature was fit per model and fault rate by minimizing $\begin{array} { r l } { ~ } & { { } \sum _ { c } D ( Q _ { c } | P _ { c } ^ { * } ) } \end{array}$ over $\tau _ { \mathrm { { i } } }$ , the sum running over contexts and $P ^ { * }$ being $P$ rescaled by $\tau$ as in $\operatorname { E q . 7 }$ . Since the rescaled distributions form a oneparameter exponential family in $1 / \tau$ , the minimizer is the temperature matching the mean logit under $P ^ { * }$ to its mean under $Q ,$ , and the split of Eq. 8 into explained and residual parts is exact for the sum over contexts.

## DECLARATIONS

## Data and Code Availability

The complete codebase used to produce the results presented in this manuscript is publicly available [63]. In particular, this repository contains a set of Jupyter notebooks that generate this manuscript’s figures, along with many other supplementary plots that help support our conclusions. These notebooks rely on summary data [64] which records the performance of our trained models, but omits their weights. Model weights are also available; this dataset is multiple terabytes and may be accessed by running the command aws s3 ls s3://ft-experiment-weights/ --request-payer requester through the AWS CLI.

## Author Contributions

TM conceptualized, designed, executed, and interpreted the results of the experiments conducted in this

work. ILC and IF provided advice throughout the project and helped compose the manuscript.

## Competing Interests

All authors declare no competing interests.

[1] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, Eficient processing of deep neural networks, Vol. 51 (Springer, 2020).

[2] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, et al., In-datacenter performance analysis of a tensor processing unit, in Proceedings of the $\mathit { 4 4 } t h$ annual international symposium on computer architecture (2017) pp. 1–12.

[3] M. J. Pelgrom, A. C. Duinmaijer, and A. P. Welbers, Matching properties of MOS transistors, IEEE Journal of solid-state circuits 24, 1433–1439 (1989).

[4] J. M. Parrondo, J. M. Horowitz, and T. Sagawa, Thermodynamics of information, Nature physics 11, 131–139 (2015).

[5] Z. F. Mainen and T. J. Sejnowski, Reliability of spike timing in neocortical neurons, Science 268, 1503–1506 (1995).

[6] A. A. Faisal, L. P. Selen, and D. M. Wolpert, Noise in the nervous system, Nature reviews neuroscience 9, 292–303 (2008).

[7] S. Sreenivasan and I. Fiete, Grid cells generate an analog error-correcting code for singularly precise neural computation, Nature neuroscience 14, 1330–1337 (2011).

[8] Y. Burak and I. R. Fiete, Accurate path integration in continuous attractor network models of grid cells, PLoS computational biology 5, e1000291 (2009).

[9] H. Agmon and Y. Burak, A theory of joint attractor dynamics in the hippocampus and the entorhinal cortex accounts for artificial remapping and grid cell field-to-field variability, Elife 9, e56894 (2020).

[10] S. Chandra, S. Sharma, R. Chaudhuri, and I. Fiete, Episodic and associative memory from spatial scafolds in the hippocampus, Nature 638, 739–751 (2025).

[11] P. Yao, H. Wu, B. Gao, J. Tang, Q. Zhang, W. Zhang, J. J. Yang, and H. Qian, Fully hardware-implemented memristor convolutional neural network, Nature 577, 641–646 (2020).

[12] M. Rao, H. Tang, J. Wu, W. Song, M. Zhang, W. Yin, Y. Zhuo, F. Kiani, B. Chen, X. Jiang, et al., Thousands of conductance levels in memristors integrated on cmos, Nature 615, 823–829 (2023).

[13] S. Ambrogio, P. Narayanan, A. Okazaki, A. Fasoli, C. Mackin, K. Hosokawa, A. Nomura, T. Yasuda, A. Chen, A. Friz, et al., An analog-ai chip for energyeficient speech recognition and transcription, Nature 620, 768–775 (2023).

[14] C. Mead, Analog VLSI and neural systems (1989).

[15] C. C. Enz and E. A. Vittoz, Charge-based MOS transistor modeling: the EKV model for low-power and RF IC design (John Wiley & Sons, 2006).

[16] A. Wang, B. H. Calhoun, and A. P. Chandrakasan, Subthreshold design for ultra low-power systems (Springer, 2006).

[17] T. McCourt, I. Fiete, N. Gershenfeld, and I. Chuang, Reliable arithmetic circuits using less reliable (sub-digital) logic (2026), submitted for publication.

[18] D. Kudithipudi, C. Schuman, C. M. Vineyard, T. Pandit, C. Merkel, R. Kubendran, J. B. Aimone, G. Orchard, C. Mayr, R. Benosman, et al., Neuromorphic computing at scale, Nature 637, 801–812 (2025).

[19] S. Ornes, Can neuromorphic computing help reduce AI’s high energy cost?, Proceedings of the National Academy of Sciences 122, e2528654122 (2025).

[20] S. R. Ahmed, R. Baghdadi, M. Bernadskiy, N. Bowman, R. Braid, J. Carr, C. Chen, P. Ciccarella, M. Cole, J. Cooke, et al., Universal photonic artificial intelligence acceleration, Nature 640, 368–374 (2025).

[21] Z. Xu, T. Zhou, M. Ma, C. Deng, Q. Dai, and L. Fang, Large-scale photonic chiplet taichi empowers 160-tops/w artificial general intelligence, Science 384, 202–209 (2024).

[22] W. A. Borders, A. Z. Pervaiz, S. Fukami, K. Y. Camsari, H. Ohno, and S. Datta, Integer factorization using stochastic magnetic tunnel junctions, Nature 573, 390– 393 (2019).

[23] A. Jelinˇciˇc, O. Lockwood, A. Garlapati, P. Schillinger, I. L. Chuang, G. Verdon, and T. McCourt, An eficient probabilistic hardware architecture for difusion-like models, npj Unconventional Computing 3, 30 (2026).

[24] N. Freitas, G. Massarelli, J. Rothschild, D. Keane, E. Dawe, S. Hwang, A. Garlapati, and T. McCourt, Taming nonequilibrium thermal fluctuations in subthreshold cmos circuits, Physical Review Applied 25, 034061 (2026).

[25] T. Eloundou, S. Manning, P. Mishkin, and D. Rock, Gpts are gpts: Labor market impact potential of llms, Science 384, 1306–1308 (2024).

[26] M. Yu, D. Wang, Q. Shan, C. J. Reed, and A. Wan, The super weight in large language models (2025), arXiv:2411.07191 [cs.CL].

[27] T. Dettmers and L. Zettlemoyer, The case for 4-bit precision: k-bit inference scaling laws, in International Conference on Machine Learning (PMLR, 2023) pp. 7750– 7774.

[28] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale, Advances in neural information processing systems 35, 30318–30332 (2022).

[29] J. von Neumann, Probabilistic logics and the synthesis of reliable organisms from unreliable components, in Automata Studies. (AM-34), Volume 34 , edited by C. E. Shannon and J. McCarthy (Princeton University Press, Princeton, 1956) pp. 43–98.

[30] A. Zlokapa, A. K. Tan, J. M. Martyn, I. R. Fiete, M. Tegmark, and I. L. Chuang, Fault-tolerant neural networks from biological error correction codes, Physical Review E 110, 054303 (2024).

[31] M. J. Rasch, C. Mackin, M. Le Gallo, A. Chen, A. Fasoli, F. Odermatt, N. Li, S. Nandakumar, P. Narayanan, H. Tsai, et al., Hardware-aware training for large-scale and diverse deep learning inference workloads using inmemory computing-based accelerators, Nature communications 14, 5282 (2023).

[32] J. B¨uchel, I. Chalas, G. Acampa, A. Chen, O. Fagbohungbe, H. Tsai, K. El Maghraoui, M. Le Gallo, A. Rahimi, and A. Sebastian, Analog foundation models, Advances in Neural Information Processing Systems 38, 61179–61228 (2026).

[33] R. Gallager, Low-density parity-check codes, IRE Transactions on information theory 8, 21–28 (1962).

[34] M. Sipser and D. A. Spielman, Expander codes, IEEE transactions on Information Theory 42, 1710–1722 (1996).

[35] E. Arikan, Channel polarization: A method for constructing capacity-achieving codes for symmetric binaryinput memoryless channels, IEEE Transactions on information Theory 55, 3051–3073 (2009).

[36] J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou, Deep learning scaling is predictable, empirically, arXiv preprint arXiv:1712.00409 (2017).

[37] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, Scaling laws for neural language models, arXiv preprint arXiv:2001.08361 (2020).

[38] J. Hofmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al., Training compute-optimal large language models, arXiv preprint arXiv:2203.15556 (2022).

[39] T. McCourt, I. R. Fiete, and I. L. Chuang, Noisy dynamical systems evolve error correcting codes and modularity, PRX Life 4, 023002 (2026).

[40] M. Eigen, Selforganization of matter and the evolution of biological macromolecules, Naturwissenschaften 58, 465– 523 (1971).

[41] J. Gallego-Feliciano, S. A. McClendon, J. Morinelli, S. Zervoudakis, and A. Saravanos, Hidden dynamics of massive activations in transformer training, arXiv preprint arXiv:2508.03616 (2025).

[42] G. Penedo, H. Kydl´ıˇcek, A. Lozhkov, M. Mitchell, C. Raffel, L. Von Werra, T. Wolf, et al., The fineweb datasets: Decanting the web for the finest text data at scale, Advances in Neural Information Processing Systems 37, 30811–30849 (2024).

[43] L. Wan, M. Zeiler, S. Zhang, Y. Le Cun, and R. Fergus, Regularization of neural networks using dropconnect, in Proceedings of the 30th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 28, edited by S. Dasgupta and D. McAllester (PMLR, Atlanta, Georgia, USA, 2013) pp. 1058–1066.

[44] L. G. Wright, T. Onodera, M. M. Stein, T. Wang, D. T. Schachter, Z. Hu, and P. L. McMahon, Deep physical neural networks trained with backpropagation, Nature 601, 549–555 (2022).

[45] A. Wagner, Robustness and Evolvability in Living Systems (Princeton University Press, Princeton, NJ, 2005).

[46] G. M. Edelman and J. A. Gally, Degeneracy and complexity in biological systems, Proceedings of the National Academy of Sciences 98, 13763–13768 (2001).

[47] M. Carandini and D. J. Heeger, Normalization as a canonical neural computation, Nature Reviews Neuroscience 13, 51–62 (2012).

[48] T. R. Rao, Error coding for arithmetic processors (Elsevier, 1974).

[49] F. Barsi and P. Maestrini, Error correcting properties of redundant residue number systems, IEEE Transactions

on Computers 100, 307–315 (2006).

[50] B. Parhami, Computer arithmetic, Vol. 20 (Oxford university press Oxford, 1999) p. 455.

[51] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al., The llama 3 herd of models, arXiv preprint arXiv:2407.21783 (2024).

[52] E. Jonas and K. P. Kording, Could a neuroscientist understand a microprocessor?, PLoS computational biology 13, e1005268 (2017).

[53] Z. C. Lipton, The mythos of model interpretability, Communications of the ACM 61, 36–43 (2018).

[54] C. J. Cueva and X.-X. Wei, Emergence of grid-like representations by training recurrent neural networks to perform spatial localization, in International Conference on Learning Representations (2018).

[55] A. Banino, C. Barry, B. Uria, C. Blundell, T. Lillicrap, P. Mirowski, A. Pritzel, M. J. Chadwick, T. Degris, J. Modayil, G. Wayne, H. Soyer, F. Viola, B. Green, T. Cain, R. Gonz´alez, D. Hassabis, D. Kumaran, et al., Vector-based navigation using grid-like representations in artificial agents, Nature 557, 429–433 (2018).

[56] A. Karpathy, llama2.c: Inference Llama 2 in one file of pure C (2023).

[57] G. Yang and E. J. Hu, Tensor programs iv: Feature learning in infinite-width neural networks, in International Conference on Machine Learning (PMLR, 2021) pp. 11727–11737.

[58] G. Yang, E. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao, Tuning large neural networks via zero-shot hyperparameter transfer, Advances in Neural Information Processing Systems 34, 17084–17097 (2021).

[59] N. Dey, B. Zhang, L. Noci, M. Li, B. Bordelon, S. Bergsma, C. Pehlevan, B. Hanin, and J. Hestness, Don’t be lazy: Completep enables compute-eficient deep transformers, Advances in Neural Information Processing Systems 38, 137707–137739 (2026).

[60] B. Mlodozeniec, P. Ablin, L. B´ethune, D. Busbridge, M. Klein, J. Ramapuram, et al., Completed hyperparameter transfer across modules, width, depth, batch and duration, in International Conference on Learning Representations, Vol. 2026 (2026) pp. 45300–45325.

[61] T. Besiroglu, E. Erdil, M. Barnett, and J. You, Chinchilla scaling: A replication attempt, arXiv preprint arXiv:2404.10102 (2024).

[62] N. Muennighof, A. Rush, B. Barak, T. Le Scao, N. Tazi, A. Piktus, S. Pyysalo, T. Wolf, and C. A. Rafel, Scaling data-constrained language models, Advances in Neural Information Processing Systems 36, 50358–50376 (2023).

[63] Fault hardened transformers, https://github.com/ reserach-12344321/fault\_hardened\_transformers, please see the codebase associted with this manuscript for full detail on how the data shown in this paper was produced and analyzed. In particular, this repository contains several analysis notebooks containing plots that provide supplemental support for our findings.

[64] Summary data for “Fault-tolerant foundation models”, Google Drive (2026), a summary dataset that contains performance data for all the models trained in our experiments.

## SUPPLEMENTARY MATERIALS

A. N-curvature in the raw data S1   
B. Residuals of the scaling law fit S2   
C. The necessity of the curvature term S2   
D. Correlations between scaling law parameters S3

## A. N-CURVATURE IN THE RAW DATA

The specific functional form chosen for the N-dependence of Eq. 4 has no bearing on the conclusions presented in Fig. 1. The recovery of performance at large scales is visible in the raw data after subtraction of the D-dependence.

To reveal the trends shown in Fig. 1 in the raw data, we fit the scaling laws to each p-cohort as usual to identify the parameters $\beta _ { 0 } \left( p \right)$ and $\beta _ { 1 } \left( p \right)$ , and then subtract $L _ { D }$ from every point in the original dataset to efectively bring each one to convergence. This transformation is shown in Fig. S1 (a): the leftmost plot shows the experimentally collected $( L , N , D )$ data for a particular value of $p ,$ and the rightmost plot shows the $( L - L _ { D } , N )$ data produced by the subtraction. As can be seen from the flatness of the contours in the rightmost plot, the subtraction entirely removes the D-dependence from the data (within the bounds of run-to-run variation).

![](images/c648def6308e46c5efddca1e60062045b7f3222c7f95d5332655301c5a6d6c78.jpg)

![](images/83d80bd7f349ad011242f9898dc9fc5b0ce097817e1c7e8f5a008a3581eee3be.jpg)  
b

![](images/2299fe295632d2ad7611845ecf06c6e77f35c7507e1b54b21f3ebccff724a38d.jpg)  
FIG. S1. Retained-capacity signals are visible in the raw data. (a) Subtracting the D-dependence from the raw data. The left panel shows the raw experimental data; the right panel shows the same data after $L _ { D }$ has been subtracted from each loss value. Essentially all D-dependence is removed from the data, as far as is visible within run-to-run variation. We show the subtraction for a single value of $p ;$ the efective datasets for all values of $p$ are given in a notebook included in the code accompanying this manuscript [63]. ${ \bf ( b ) }$ Retained-capacity signals in the L<sub>D</sub>-subtracted dataset. Once every datapoint has been brought to convergence by removing the D-dependence, the standard Chinchilla law fitted to the $p = 0$ cohort can be used to compute η for each point in the data.

After the subtraction is complete, we can use the standard Chinchilla scaling law identified for the $p = 0$ cohort to assign a value of η to each point in the efective dataset. Fig. S1 (b) shows the results of this projection: exactly the same curvature shown in Fig. 1 is shown in this figure.

## B. RESIDUALS OF THE SCALING LAW FIT

The ultimate reason that the trends shown in Fig. 1 are not corrupted by the choice of the N-dependence in the scaling law is that the scaling law fits the data well. The goodness-of-fit of Eq. 4 to our data is demonstrated by the plots shown in Fig. S2.

![](images/71ac38c13b4e858f30e0239c8e330baa7942d303cd1bc150eab8e709fd70fd6a.jpg)

![](images/304c46bca5c403100f2e2c3903b42ee033782af97bf1455cbc0318843270c332.jpg)  
FIG. S2. Residuals of the scaling law fit (a) The log-residuals of our scaling law fit (L<sup>ˆ</sup>) vs measured loss (L), for all measured values of p. The scale of the residuals is near what would be expected due to run-run variation alone (indicated by the gray band). (b) The loss predicted by our model vs the measured loss. The black line indicates a perfect fit.

## C. THE NECESSITY OF THE CURVATURE TERM

In contrast to the precise fit of our new law demonstrated in Fig. S2, fits of the standard Chinchilla form (without the new curvature term) to $p > 0$ cohorts fail miserably.

Fig. S3 shows the log residual of the curvature-free fit vs N, averaged over diferent values of D and independent training runs. For $p > 0$ , there is a clear curvature in the residuals which is ultimately what inspired our addition of $\alpha _ { 2 }$ to the law.

![](images/10190fa192497e8779f651e5659718ea74b3e217a5c2c9820b80e70f7ce6b535.jpg)  
FIG. S3. Residuals in the scaling law fit without the curvature term Residuals in a fit of the standard Chinchilla form to our datasets. The curvature term is clearly needed to represent the trends in the raw data.

## D. CORRELATIONS BETWEEN SCALING LAW PARAMETERS

We quantified the uncertainty in our scaling law fit parameter values via bootstrapping. We drew many synthetic datasets (1500) by resampling the experimental runs with replacement and refit the scaling law to each one. This procedure left us with many samples of possible scaling law parameter sets, which approximate the true distribution of fit parameters given the data. The error bars shown in Fig. 2 were generated by estimating 95% confidence intervals from our bootstrap samples.

The parameter samples generated by our bootstrapping procedure are useful for revealing correlations between the diferent fit parameters, as shown in Fig. S4.

![](images/7b9846f7569de39fd8b5dc5c65cbe99406484f577724607539c87988773998f4.jpg)  
FIG. S4. Pairwise correlations between scaling law parameters Pairwise scatter plots that identify correlations between scaling law parameters. A small subset of the complete bootstrap dataset is shown to avoid overcrowding the plots with points.

While there are weak correlations between the lower order N and D coeficients, α is essentially entirely uncorrelated with any other fit parameter. This makes sense, as no other parameter in Eq. 4 is capable of producing the curvature shown clearly in Fig. S3.