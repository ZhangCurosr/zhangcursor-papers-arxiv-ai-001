# A Tutorial on Quantum Amplitude Embedding with Data Re-Uploading

Hyojun Ahn

Department of Electrical & Computer Engineering

Korea University

Seoul, Republic of Korea

hyojun@korea.ac.kr

Joongheon Kim

Department of Electrical & Computer Engineering

Korea University

Seoul, Republic of Korea

joongheon@korea.ac.kr

Abstract—Amplitude embedding lets a variational quantum model read many classical features with few qubits, but the model output is then only a quadratic form in the normalized input. Data re-uploading, which encodes the input repeatedly between trainable blocks, can be used to enrich such a model. Because every additional upload of an amplitude embedding repeats a costly state preparation, the gain that an upload buys has to be known before the gates are spent. This tutorial thus addresses the problem of explaining what data re-uploading adds to an amplitude-embedded model and how large the gain is. We express sequential re-uploading and parallel re-uploading with two copies in one notation and show that each upload raises the reachable polynomial order by two. Under one trainable circuit family and one budget of trainable parameters, simulation results show that re-uploading raises the test coefficient of determination from 0.689 to between 0.883 and 0.975 on targets of order four and from 0.337 to between 0.642 and 0.783 on targets of order six, on all ten seeds. On a quadratic target and on handwritten digits, one upload remains the best choice, and every additional upload lowers the gate error that the classifier tolerates. These results help designers match the number of uploads to the polynomial order of the task at hand.

Index Terms—Quantum machine learning, variational quantum circuit, amplitude embedding, data re-uploading, state preparation, quantum data encoding

## I. INTRODUCTION

Learning-based models now control aerial vehicles and schedule satellites [1], [2]. Variational quantum models are parameterized circuits trained as functions of classical inputs [3]– [5]. Such models are candidate learners for these systems [6]– [8]. They are also studied for visual recognition and distributed learning [9]–[12]. In all of these settings the classical input first has to be written into a quantum state, and this encoding step decides which functions of the input the model can express [13]. Amplitude embedding stores the features in the amplitudes of a quantum state, so the number of qubits grows only logarithmically with the number of features. This makes it a common encoding pattern for high-dimensional inputs [14], [15]. The price is a rigid function class, because the output of an amplitude-embedded model is only a quadratic form in the normalized input vector, that is, a weighted sum of products of two features [16].

Data re-uploading encodes the input several times between trainable blocks, each encoding being one upload, and it is the standard remedy for a similar rigidity in angle-encoded models [17]. The theory of re-uploading is written for angle encodings [13], [18]. Hands-on tutorials on re-uploading also use angle encodings [19]. For an amplitude embedding, each additional upload repeats a state preparation whose gate count is exponential in the qubit count [20], [21]. Studies of generalization also warn that encoding layers are not free [22]. Empirical comparisons of encodings change the preprocessing, the qubit count, or the learning model together with the encoding [23], [24]. A recent empirical study applies re-uploading to amplitude-encoded classifiers on two small tabular datasets and reports gains that depend on the dataset [25]. Which targets make re-uploading pay off for an amplitude-embedded model, and by how much, thus remains unexplained.

We treat the state preparation of the amplitude embedding as the unit that is uploaded again. A sequential scheme reapplies this state preparation between trainable blocks on the same qubits, and a parallel scheme prepares several copies of the embedded state side by side. Viewed in the angles of the state preparation, every upload adds frequencies to the model, which raises by two the polynomial order of the input that the model can reach. To attribute a gain to the uploads alone, all schemes share one trainable circuit family and one budget of trainable parameters, so that they differ only in the position at which the data enters the circuit.

The main contribution of this paper is a tutorial on data re-uploading for amplitude embedding that explains its fundamentals and quantifies its performance gain.

• We express amplitude embedding with sequential and with parallel re-uploading in one notation. We show that a sequential upload acts as an angle encoding in the statepreparation angles, so that each additional upload raises the reachable polynomial order by two and costs one more state preparation.

• We measure the gain under one trainable circuit family, one budget of trainable parameters, and one training procedure, on synthetic targets of order 2, 4, and 6.

• We delimit the gain. On a quadratic target and on handwritten digits, one upload remains the best choice, and every additional upload lowers the two-qubit gate error that the classifier tolerates.

![](images/e0a7abbd2f46a72c4774ae6de089d738a3494587bd160e05d28ca4adb56edb84.jpg)  
Fig. 1. Amplitude-embedded variational quantum model with one upload. S(x) writes the normalized image into six qubits, and an optimizer trains θ.

## II. DATA ENCODING IN VARIATIONAL QUANTUM MODELS

This section reviews how classical data enters a variational quantum model, which Fig. 1 illustrates for an image. A register of n qubits has $2 ^ { n }$ basis states, and its state is a unit vector of $2 ^ { n }$ amplitudes. The model proceeds as follows.

1) The encoding normalizes the input and writes it into the amplitudes of the qubits with a state preparation S(x).

2) The trainable block $W ( \pmb \theta )$ applies V layers, each of which rotates every qubit and entangles the qubits with a ring of controlled-Z gates.

3) The measurement returns outcome probabilities, from which the model output is computed.

4) A classical optimizer updates the trainable parameters from the loss, and steps 1) to 3) are repeated.

A variational quantum model on n qubits applies a circuit $U ( { \pmb x } , { \pmb \theta } )$ that depends on the input $\pmb { x } \in \mathbb { R } ^ { d }$ and on trainable parameters θ. The model output is the expectation of an observable O, which can be expressed as,

$$
f _ { \pmb \theta } ( \pmb x ) = \langle 0 | U ^ { \dagger } ( \pmb x , \pmb \theta ) O U ( \pmb x , \pmb \theta ) | 0 \rangle ,\tag{1}
$$

where |0⟩ denotes the all-zero state of the n qubits. A classical optimizer trains θ on a loss that is computed from $f _ { \theta }$ [3]. The circuit structure can also be searched, and such models have been combined with language models [26], [27].

Amplitude embedding writes $d \ = \ 2 ^ { n }$ features into the amplitudes of n qubits, and the encoded state is defined as,

$$
| \psi ( { \pmb x } ) \rangle = \sum _ { i = 0 } ^ { d - 1 } \tilde { x } _ { i } | i \rangle , \qquad \tilde { \pmb x } = { \pmb x } / \| { \pmb x } \| _ { 2 } ,\tag{2}
$$

where |i⟩ denotes the i-th computational basis state and $\tilde { x } _ { i }$ denotes the i-th entry of x˜. A state-preparation unitary $S ( { \pmb x } )$ maps |0⟩ to $| \psi ( { \pmb x } ) \rangle$ , and a trainable circuit $W ( \pmb \theta )$ follows it, so that $U = W ( \pmb \theta ) S ( \pmb x )$ . For a real input, the model output in (1) can then be expressed as,

$$
f _ { \pmb \theta } ( { \pmb x } ) = \tilde { \pmb x } ^ { \top } M _ { \pmb \theta } \tilde { \pmb x } , \qquad M _ { \pmb \theta } = \mathrm { R e } \big [ W ^ { \dagger } ( \pmb \theta ) { \cal O } W ( \pmb \theta ) \big ] ,\tag{3}
$$

which is a quadratic form in the normalized input. With two features, for example, the output can contain $\tilde { x } _ { 0 } ^ { 2 } , \tilde { x } _ { 0 } \tilde { x } _ { 1 }$ , and $\tilde { x } _ { 1 } ^ { 2 }$ but no term of higher order. Training changes the matrix $M _ { \theta }$ but not the degree, so a deeper W adds no nonlinearity in x.

![](images/649ee647a1ac6a57c07c6a524f88b47273e112774703dedc828c1c444528b1c5.jpg)  
Fig. 2. One upload of four features. Both encodings use rotations whose angles depend on the data. Angle encoding turns one qubit per feature, and each angle $\alpha _ { k , c } ( \pmb { x } )$ of amplitude embedding splits the probability mass.

Angle encoding writes each feature as the angle of a singlequbit rotation, and Fig. 2 compares one upload of four features in the two encodings. A model of this kind is a partial Fourier series in the data, that is, a sum of terms cos(mx) and sin(mx) with integer frequencies m, and the encoding gates determine which frequencies appear [13], [28]. Repeating the encoding gates between trainable blocks, which is called data re-uploading, enlarges the frequency set with every repetition. The repetitions are not free, because generalization bounds grow with the number of encoding gates and more expressive circuits have smaller training gradients [22], [29].

For angle encoding, one upload is a layer of single-qubit gates. For amplitude embedding, one upload is a state preparation. An $R _ { y } ( \alpha )$ gate rotates one qubit by the angle α, and a CNOT gate flips one qubit when another qubit is one. The standard construction is a cascade of uniformly controlled $R _ { y }$ rotations, in which the rotation angle on qubit k is selected by the k preceding qubits [20]. The rotation on qubit k has one angle $\alpha _ { k , c }$ for every setting $c = 0 , \ldots , 2 ^ { k } - 1$ of these qubits,

![](images/a8df85b240bdd137b93e9de669bbcfa8040d5ee544dad428de5cd525011bff58.jpg)  
Fig. 3. Re-uploading schemes for amplitude embedding. Orange blocks upload the input, and blue blocks are trainable.

TABLE I  
ENCODING SCHEMES ON $d = 2 ^ { n }$ FEATURES
<table><tr><td>Scheme</td><td></td><td>Qubits Enc. CNOTs Model output</td><td></td></tr><tr><td>Amplitude embedding</td><td>n</td><td> $2 ^ { n } - 2$ </td><td>Quadratic form in x</td></tr><tr><td>Sequential, L uploads</td><td>n</td><td> $L ( 2 ^ { n } - 2 )$ </td><td>Frequencies  $\leq L$  in  $\alpha _ { k , c }$ </td></tr><tr><td>Parallel, k copies</td><td>kn</td><td> $k ( 2 ^ { n } - 2 )$ </td><td>Degree-2k polynomial in x</td></tr><tr><td>Angle, L uploads</td><td>d</td><td>0</td><td>Frequencies ≤ L per feature</td></tr></table>

and the angle is defined as,

$$
\alpha _ { k , c } ( \pmb { x } ) = 2 \arctan \sqrt { p _ { k + 1 , 2 c + 1 } / p _ { k + 1 , 2 c } } ,\tag{4}
$$

where $p _ { k , c }$ denotes the sum of $\tilde { x } _ { i } ^ { 2 }$ over block c when x˜ is cut into $2 ^ { k }$ blocks of consecutive entries, counted from zero. The first rotation splits the total probability, which is the sum of the squared amplitudes, between the two halves of x˜, and each later rotation splits one of these parts again, until the parts in Fig. 2 are the squared amplitudes. Each amplitude is therefore a product of n factors co $\scriptstyle \{ \alpha _ { k , c } / 2 )$ or sin $( \alpha _ { k , c } / 2 )$ , one for every split. For example, $\tilde { x } _ { 2 } = \sin ( \alpha _ { 0 , 0 } / 2 ) \cos ( \alpha _ { 1 , 1 } / 2 )$ . A uniformly controlled rotation with k control qubits takes $2 ^ { k }$ CNOT gates, as Fig. 2 shows for one control [30]. Preparing a real nonnegative vector thus takes $2 ^ { n } - 2$ CNOT gates, and a general complex vector takes about twice as many [21]. This count is linear in the number of features d and exponential in the number of qubits, a known caveat of loading classical data [31]. In both encodings an upload is thus a set of rotations whose angles depend on the data, and the encodings differ in these angles and in the gate count.

## III. RE-UPLOADING AN AMPLITUDE EMBEDDING

This section derives what a second upload adds to (3) and what it costs, as Table I summarizes. Fig. 3 shows the two ways of repeating the upload of Fig. 1, each drawn as a block view above the corresponding circuit.

Sequential re-uploading. The sequential scheme in Fig. 3(a) re-applies $S ( { \pmb x } )$ between L trainable blocks on the same n qubits, and the circuit can be expressed as,

$$
U _ { L } ( { \pmb x } , { \pmb \theta } ) = W _ { L } ( { \pmb \theta } _ { L } ) S ( { \pmb x } ) \cdots W _ { 2 } ( { \pmb \theta } _ { 2 } ) S ( { \pmb x } ) W _ { 1 } ( { \pmb \theta } _ { 1 } ) S ( { \pmb x } ) ,\tag{5}
$$

where $W _ { l } ( \pmb { \theta } _ { l } )$ denotes the l-th trainable block. Only the first upload prepares $| \psi ( { \pmb x } ) \rangle$ . The later uploads act on a state that is no longer |0⟩, so they are data-dependent unitaries with no simple expression in x˜. The cascade that implements $S ( { \pmb x } )$ depends on the input only through the $R _ { y }$ angles $\alpha _ { k , c } ( \pmb { x } )$ in (4). The circuit in (5) is therefore an angle-encoded model in these $2 ^ { n } - 1$ angles, and the Fourier analysis of re-uploading applies to it [13]. One upload produces frequencies of at most one in each angle, which is the quadratic form in (3). With L uploads the frequencies reach L. A polynomial of degree 2L in x˜ generally contains the frequency L in these angles. A target of order 2L in the input thus needs at least L uploads, whatever the depth of the trainable blocks. The case $ { n _ { \mathrm { ~  ~ } } } =  { \mathrm { ~ \boldsymbol ~ 1 ~ } }$ makes this concrete. Here $S ( { \pmb x } ) \ = \ R _ { y } ( \alpha )$ with $\tilde { \pmb { x } } = ( \cos ( \alpha / 2 ) , \sin ( \alpha / 2 ) )$ ), so one upload gives $f _ { \pmb { \theta } } =$ $a _ { 0 } + a _ { 1 }$ cos $\alpha + b _ { 1 }$ sin α with coefficients that depend on θ. A second upload adds cos 2α and sin 2α, and cos $2 \alpha = 1 - 8 \tilde { x } _ { 0 } ^ { 2 } \tilde { x } _ { 1 } ^ { 2 }$ is a quartic polynomial in x˜.

Parallel re-uploading. The parallel scheme in Fig. 3(b) prepares k copies of $| \psi ( { \pmb x } ) \rangle$ ⟩ on kn qubits and applies one trainable circuit $W ( \pmb \theta )$ to all of them. The copies are prepared from the same classical input, so no quantum state has to be cloned. For two copies, the model output can be expressed as,

$$
f _ { \pmb { \theta } } ( \pmb { x } ) = \langle \psi ( \pmb { x } ) | \langle \psi ( \pmb { x } ) | ~ W ^ { \dagger } ( \pmb { \theta } ) ~ O ~ W ( \pmb { \theta } ) ~ | \psi ( \pmb { x } ) \rangle | \psi ( \pmb { x } ) \rangle ,\tag{6}
$$

which is a quadratic form in x˜ ⊗ x˜ and hence a homogeneous polynomial of degree four in x˜. With k copies the degree is 2k. This is the circuit counterpart of the polynomial kernel that copies of an embedded state provide [16]. Two copies thus reach the same order as two sequential uploads at a different cost in qubits and depth.

Cost. Table I sets the function class of each scheme against its resources, namely the number of qubits (Qubits) and the CNOT gates spent on encoding nonnegative inputs (Enc. CNOTs). The sequential scheme keeps n qubits and multiplies the encoding count by L. The preparations run in series, so the circuit depth grows by one state preparation per upload. The parallel scheme multiplies the count by k as well but spreads the gates over k registers that are prepared side by side, at the price of kn qubits. With $n = 6 ,$ , one upload costs 62 CNOT gates, whereas one trainable layer of the circuits in Section IV costs only six two-qubit gates. These counts decide whether a quantum learner fits platforms on which learned controllers operate, such as aerial and underwater vehicles [32]–[34].

TABLE II EXPERIMENTAL SETTINGS OF THE TWO TASKS
<table><tr><td>Setting</td><td>Synthetic targets</td><td>Digits</td></tr><tr><td>Features d / qubits n</td><td>8 / 3</td><td>64 / 6</td></tr><tr><td>Trainable layers V</td><td>36</td><td>24</td></tr><tr><td>Trainable parameters 3nV</td><td>324</td><td>432</td></tr><tr><td>Training / test samples</td><td>4,000 / 2,000</td><td>100 to 1,297 / 500</td></tr><tr><td>Training steps</td><td>2,400</td><td>600</td></tr><tr><td>Learning rates searched</td><td>0.003, 0.01, 0.03, 0.1</td><td></td></tr><tr><td>Reported / pilot seeds</td><td>10 / 2</td><td></td></tr><tr><td>Metric</td><td>Test R²</td><td>Test accuracy</td></tr></table>

![](images/bed480c5b266c2e62900be2a17d46cbb9b1343677603c672c63ef500f17e2f33.jpg)  
Fig. 4. Training loss on the order-4 target (ten-seed mean).

## IV. PERFORMANCE EVALUATION

The evaluation measures how much re-uploading improves an amplitude-embedded model and where the improvement ends. Section III predicts a gain only for targets of higher order than two, so the first task controls the order of the target. The order of a target is its polynomial degree in the input. The second task is an image benchmark, and a final analysis prices the additional state preparations under gate errors.

## A. Experimental Setup

Models. Table II lists the settings. All models use one trainable circuit family, that is, one ansatz. A layer applies a general single-qubit rotation with three parameters to every qubit and then a ring of controlled-Z gates, as in Fig. 1. We compare amplitude embedding with one upload $( L = 1 )$ , sequential reuploading with $L \in \{ 2 , 3 , 4 \}$ , and parallel re-uploading with two copies. The total number of trainable layers V is fixed and is split evenly over the L blocks, and the two-copy model uses $V / 2$ layers on 2n qubits. Every model therefore has the same 3nV trainable parameters, and the models differ only in where the data enters. The state preparation $S ( { \pmb x } )$ is the cascade of uniformly controlled $R _ { y }$ rotations, which is exact for nonnegative inputs.

Tasks. The synthetic task isolates the order of the target. Inputs are uniformly distributed on the nonnegative part of the unit sphere. The target is a product of one, two, or three standardized random quadratic forms in ${ \tilde { \mathbf { x } } } ,$ which gives the orders $2 , 4 ,$ and 6 that Fig. 5 compares. We set $V = 3 6$ , so that every trainable block of the three-qubit models has at least 81 parameters, more than the 63 of a general three-qubit unitary, and the trainable part is not the bottleneck. The model output is a trained linear combination of the eight outcome probabilities. The metric is the test coefficient of determination $R ^ { 2 }$ , which is one for a perfect fit and zero for predicting the mean. The image task uses the $8 \times 8$ handwritten digits distributed with scikit-learn, with 1,797 images in ten classes [35]. The 64 pixel intensities are embedded in six qubits, and $V = 2 4$ leaves six layers per block with four uploads. Four qubits are measured, and the predicted class is the most probable of their first ten outcomes [36]. The training loss is their cross-entropy scaled by one trained factor.

![](images/c00f575d0067efa71adb54d2cd60482b425a672b39ffe40e0e2120ee566f5eeb.jpg)  
Fig. 5. Test $R ^ { 2 }$ on synthetic targets of order 2, 4, and 6 (mean and standard deviation over ten seeds). The dashed line is the best quadratic form.

Training. All models are trained with the Adam optimizer. Training is full-batch with a cosine learning-rate schedule. Runs take 2,400 steps on the synthetic task, by which point the training loss in Fig. 4 has leveled off. On the digits, 600 steps suffice for every model to fit its training set. Initial parameters are uniform in $[ - \pi , \pi ]$ . The learning rate of each model is selected on held-out training data of two pilot seeds, which are not among the ten reported seeds. A seed fixes the data split or target and the initial parameters, and all models see the same data for a given seed. We report means and standard deviations over the ten seeds, together with the number of seeds on which each model beats $L = 1$

## B. Convergence

Fig. 4 shows that the gain of re-uploading is already present in the training loss and is not an effect of the stopping point. The loss is the mean squared error on the standardized target. One upload levels off at 0.312, whereas two, three, and four uploads level off at 0.109, 0.073, and 0.064, and two copies reach 0.022. The models with $L = 1$ to $L = 4$ are within 5% of their final loss after at most 1,400 of the $2 , 4 0 0$ steps, so more steps would not let one upload catch up.

## C. Gain on Targets of Higher Order

Fig. 5 shows that re-uploading lifts an amplitude-embedded model above a limit that one upload cannot pass. The dashed line is the best quadratic form in x˜, obtained by least squares, and (3) makes it the upper limit of $L = 1$ . On the order-4 target, one upload reaches an $R ^ { 2 }$ of 0.689 against a limit of 0.694, so further training of $L = 1$ has nothing left to gain. Two uploads raise $R ^ { 2 }$ to 0.883, three and four uploads reach 0.916 and 0.918, and two copies reach 0.975. The paired gains over $L = 1$ are 0.194, 0.227, 0.229, and 0.286, and each model is ahead on all ten seeds. Measured against the gap that one upload leaves to a perfect fit, two uploads close 62% of the gap and two copies close 92%. The two-copy model leads because its function class is the set of degree-4 polynomials in Table I, which contains the target.

TABLE III  
TEST ACCURACY (%) ON THE DIGITS (MEAN ± STANDARD DEVIATION)
<table><tr><td>Model</td><td> $2 \mathrm { q }$ </td><td></td><td></td><td>gates 100 images 400 images 1,297 images</td></tr><tr><td>Amplitude,  $L = 1$ </td><td>206</td><td> ${ \bf 8 7 . 4 \pm 1 . 9 }$ </td><td> ${ \bf 9 4 . 0 \pm 1 . 1 }$ </td><td> ${ \bf 9 5 . 9 \pm 1 . 1 }$ </td></tr><tr><td>Sequential,  $L = 2$ </td><td>268</td><td> $8 4 . 2 \pm 2 . 6$ </td><td> $9 1 . 8 \pm 0 . 9$ </td><td> $9 4 . 8 \pm 0 . 8$ </td></tr><tr><td>Sequential,  $L = 3$ </td><td>330</td><td> $8 1 . 7 \pm 1 . 9$ </td><td> $8 9 . 4 \pm 1 . 4$ </td><td> $\overline { { 9 2 . 3 \pm 1 . 2 } }$ </td></tr><tr><td>Sequential,  $L = 4$ </td><td>392</td><td> $6 8 . 9 \pm 2 . 4$ </td><td> $8 4 . 2 \pm 2 . 0$ </td><td> $8 8 . 5 \pm 1 . 4$ </td></tr><tr><td>Parallel, two copies</td><td>268</td><td> $8 2 . 2 \pm 2 . 6$ </td><td> $9 0 . 7 \pm 2 . 8$ </td><td> $9 4 . 3 \pm 0 . 7$ </td></tr></table>

![](images/15d9706616e9f4809922f5441164c52eeb2568e821e987ee8767da2d73513168.jpg)  
Fig. 6. Digit accuracy with 1,024 shots against the two-qubit gate error (mean and standard deviation over ten seeds).

On the order-6 target, the gain follows the number of uploads as Section III predicts. One upload gives 0.337, which again sits at the limit of a quadratic form. Two uploads give 0.642, three give 0.745, and four give 0.783, which are gains of 0.305, 0.408, and 0.446. Every model is again ahead of $L = 1$ on all ten seeds, and four uploads close 67% of the gap that one upload leaves. The gain thus keeps growing with every upload when the order of the target is high, whereas it saturates after three uploads on order 4. Two copies give 0.771 and are no longer the best model, because their function class ends at degree four. On the order-2 target, one upload already reaches 0.999, and re-uploading has nothing to add. The sequential models are slightly lower on all ten seeds, between 0.975 and 0.977, and two copies reach 0.997. The model classes are not nested, because the later uploads act on states other than |0⟩. A pure quadratic form is thus harder to realize once uploads interrupt the trainable layers.

## D. Limits of the Gain

Table III shows that the gain does not carry over to a task that one upload already fits, where bold and underline mark the best and the second-best model. With all 1,297 images, one upload reaches 95.9%, two uploads reach 94.8%, three reach 92.3%, and four reach 88.5%. The deficit grows as the training set shrinks, and four uploads lose 7.4 percentage points with 1,297 images and 18.5 points with 100 images. Two copies lose less, 1.6 points with 1,297 images and 5.2 points with 100 images. The cause is generalization and not the fit, since every model classifies at least 99.9% of its training images correctly, which agrees with bounds that grow with the number of encoding gates [22]. The digits thus behave like the order-2 target, on which additional uploads add capacity that the task does not need. The ranking of the models does not depend on the selected learning rates. With a common rate of 0.01, four uploads reach 75.1% with 100 images and 88.8% with 1,297 images and remain below one to three uploads.

The second limit is the hardware cost of the additional state preparations, which Fig. 6 measures. Under a global depolarizing channel, we evaluate the classifiers that are trained on 1,297 images, and the circuit fidelity is calculated as,

$$
F = ( 1 - \epsilon ) ^ { G } ,\tag{7}
$$

where ϵ denotes the two-qubit gate error rate and G denotes the two-qubit gate count in Table III. The outcome distribution is then the ideal one with weight F and the uniform one with weight $1 - F$ . A shot is one run of the circuit followed by a measurement. We draw 1,024 shots per image, repeat the sampling 20 times, and vary ϵ from 0 to 2%. This range covers the 0.62% reported for two-qubit gates on a 53-qubit processor [37]. Training is noiseless, so the results measure how much error a trained circuit tolerates. Fig. 6 shows that each upload lowers the gate error that the classifier tolerates, and that the two-copy model is limited by shot noise before gate errors. Without gate errors, 1,024 shots lower $L = 1$ from 95.9% to 93.7% and $L = 2$ from 94.8% to 93.8%, so both models are level. One upload loses more because its margin between the two most probable classes is 0.058 on average, against 0.085 for $L = 2 . \mathrm { A t } \epsilon = 1 \%$ the 62 additional CNOT gates of the second upload dominate, and $L = 1$ keeps 47.6% against 36.7% for $L \ = \ 2 .$ . Four uploads fall off first, from 87.2% without gate errors to 58.0% at $\epsilon = 0 . 5 \%$ , where their fidelity is 0.14 against 0.36 for $L = 1$ . The two-copy model reaches only 69.2% with 1,024 shots, far below its 94.3% in exact evaluation. Its average margin is 0.013, less than a quarter of that of $L = 1 ,$ , so its classes are resolved only with many more shots. The number of uploads should therefore follow the polynomial order of the task, and each upload beyond that order costs both accuracy and two-qubit gates.

## V. CONCLUSION

In this paper, we have presented the fundamentals of data reuploading for amplitude embedding and quantified its performance gain under one trainable circuit family and one budget of trainable parameters. We have shown that a sequential upload acts as an angle encoding in the state-preparation angles. A target of order 2L in the input therefore needs at least L uploads, each at $2 ^ { n } - 2 \thinspace \mathrm { C N O T }$ gates. On synthetic targets of order 4 and 6, re-uploading has raised the test $R ^ { 2 }$ from 0.689 to as much as 0.975 and from 0.337 to as much as 0.783, on all ten seeds. On a quadratic target and on handwritten digits, one upload has remained the best choice, and every additional upload has lowered the tolerated gate error. These results come from simulated circuits of three to twelve qubits with nonnegative inputs and noiseless training. They still give designers a rule, namely to match the number of uploads to the polynomial order of the task.

## ACKNOWLEDGEMENTS

This research was supported by MSIT (Ministry of Science and ICT), Korea, under ITRC (Information Technology Research Center) support program (IITP-2026-RS-2024- 00436887) supervised by IITP (Institute for Information & Communications Technology Planning & Evaluation); and this work was also supported by the National Research Foundation (NRF) by the Korea Ministry of Science and ICT (Information and Communications Technology) (MSIT) (RS-2025- 00561377). (Corresponding author: Joongheon Kim)

## REFERENCES

[1] H. Ahn, S. Oh, G. S. Kim, S. Park, S. Jung, and J. Kim, “Hallucinationaware hierarchical LLM for autonomous UAV mobility control: A safe reinforcement learning approach,” IEEE Trans. Netw., vol. 34, pp. 6827– 6842, Aug. 2026.

[2] H. Ahn, G. S. Kim, I.-S. Cho, S. Jung, and J. Kim, “Hybrid large language models and reinforcement learning for energy-efficient multisatellite scheduling: Boosting the performance from scratch,” IEEE Internet Things J., vol. 13, no. 3, pp. 5379–5392, Feb. 2026.

[3] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, “Quantum circuit learning,” Phys. Rev. A, vol. 98, no. 3, Sep. 2018, Art. no. 032309.

[4] M. Cerezo et al., “Variational quantum algorithms,” Nat. Rev. Phys., vol. 3, no. 9, pp. 625–644, Sep. 2021.

[5] S. Oh, C. Im, S. Park, and J. Kim, “Understanding qubit trajectories in parametrized quantum circuits,” Phys. Rev. A, vol. 113, no. 2, Feb. 2026, Art. no. 022609.

[6] S. Park and J. Kim, “Trends in quantum reinforcement learning: Stateof-the-arts and the road ahead,” ETRI J., vol. 46, no. 5, pp. 748–758, Oct. 2024.

[7] G. S. Kim, S. Lee, I.-S. Cho, S. Park, and J. Kim, “Quantum reinforcement learning for lightweight LEO satellite routing,” IEEE Internet Things J., vol. 12, no. 14, pp. 28 986–29 004, Jul. 2025.

[8] G. S. Kim et al., “Quantum multi-agent reinforcement learning for cooperative mobile access in space-air-ground integrated networks,” IEEE Trans. Mobile Comput., vol. 25, no. 1, pp. 1200–1218, Jan. 2026.

[9] Y. Kwak et al., “Quantum distributed deep learning architectures: Models, discussions, and applications,” ICT Express, vol. 9, no. 3, pp. 486–491, Jun. 2023.

[10] E. J. Roh, H. Baek, D. Kim, and J. Kim, “Fast quantum convolutional neural networks for low-complexity object detection in autonomous driving applications,” IEEE Trans. Mobile Comput., vol. 24, no. 2, pp. 1031–1042, Feb. 2025.

[11] J. Kim, E. J. Roh, C. Im, and S. Park, “Quantum-Eyes: Scalable quantum convolutional neural networks for low-overhead object detection,” IEEE Comput. Intell. Mag., vol. 20, no. 3, pp. 63–74, Aug. 2025.

[12] S. Park, S. B. Son, S. Jung, and J. Kim, “Dynamic quantum federated learning for UAV-based autonomous surveillance,” IEEE Trans. Veh. Technol., vol. 74, no. 5, pp. 8158–8170, May 2025.

[13] M. Schuld, R. Sweke, and J. J. Meyer, “Effect of data encoding on the expressive power of variational quantum-machine-learning models,” Phys. Rev. A, vol. 103, no. 3, Mar. 2021, Art. no. 032430.

[14] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, “Circuitcentric quantum classifiers,” Phys. Rev. A, vol. 101, no. 3, Mar. 2020, Art. no. 032308.

[15] E. J. Roh and J. Kim, “Quantum-amplitude embedded adaptation for parameter-efficient fine-tuning in large language models,” in Proc. ACM Int. Conf. Inf. Knowl. Manage. (CIKM), Seoul, Republic of Korea, Nov. 2025, pp. 5171–5175.

[16] M. Schuld and N. Killoran, “Quantum machine learning in feature Hilbert spaces,” Phys. Rev. Lett., vol. 122, no. 4, Feb. 2019, Art. no. 040504.

[17] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, “Data re-uploading for a universal quantum classifier,” Quantum, vol. 4, Feb. 2020, Art. no. 226.

[18] S. Jerbi, L. J. Fiderer, H. Poulsen Nautrup, J. M. Kübler, H. J. Briegel, and V. Dunjko, “Quantum machine learning beyond kernel methods,” Nat. Commun., vol. 14, no. 1, Jan. 2023, Art. no. 517.

[19] E. Peña Tapia, G. Scarpa, and A. Pozas-Kerstjens, “A didactic approach to quantum machine learning with a single qubit,” Phys. Scr., vol. 98, no. 5, May 2023, Art. no. 054001.

[20] M. Möttönen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa, “Transformation of quantum states using uniformly controlled rotations,” Quantum Inf. Comput., vol. 5, no. 6, pp. 467–473, Sep. 2005.

[21] V. V. Shende, S. S. Bullock, and I. L. Markov, “Synthesis of quantumlogic circuits,” IEEE Trans. Comput.-Aided Design Integr. Circuits Syst., vol. 25, no. 6, pp. 1000–1010, Jun. 2006.

[22] M. C. Caro, E. Gil-Fuster, J. J. Meyer, J. Eisert, and R. Sweke, “Encoding-dependent generalization bounds for parametrized quantum circuits,” Quantum, vol. 5, Nov. 2021, Art. no. 582.

[23] T. Hur, L. Kim, and D. K. Park, “Quantum convolutional neural network for classical data classification,” Quantum Mach. Intell., vol. 4, no. 1, Jun. 2022, Art. no. 3.

[24] M. Rath and H. Date, “Quantum data encoding: A comparative analysis of classical-to-quantum mapping techniques and their impact on machine learning accuracy,” EPJ Quantum Technol., vol. 11, no. 1, Dec. 2024, Art. no. 72.

[25] A. Tudisco, A. Marchesin, M. Zamboni, M. Graziano, and G. Turvani, “Evaluating angle and amplitude encoding strategies for variational quantum machine learning: Their impact on model’s accuracy,” Adv. Quantum Technol., vol. 9, no. 4, Apr. 2026, Art. no. e00611.

[26] S. B. Son, H. Ahn, S. Y.-C. Chen, S. Park, and J. Kim, “Efficient quantum neural architecture search with filtered supernet for mobile learning systems,” IEEE Trans. Mobile Comput., vol. 25, no. 11, pp. 19 358–19 375, Nov. 2026.

[27] E. J. Roh, H. Ahn, S. Y.-C. Chen, S. Park, and J. Kim, “How can quantum deep learning improve large language models?” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Barcelona, Spain, May 2026, pp. 22 412–22 416.

[28] S. Oh, E. J. Roh, A. V. Vasilakos, S. Park, and J. Kim, “Fourier analysis perspective on quantum neural networks,” Commun. Phys., vol. 9, no. 1, May 2026, Art. no. 176.

[29] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, “Barren plateaus in quantum neural network training landscapes,” Nat. Commun., vol. 9, no. 1, Nov. 2018, Art. no. 4812.

[30] V. Bergholm, J. J. Vartiainen, M. Möttönen, and M. M. Salomaa, “Quantum circuits with uniformly controlled one-qubit gates,” Phys. Rev. A, vol. 71, no. 5, May 2005, Art. no. 052330.

[31] S. Aaronson, “Read the fine print,” Nat. Phys., vol. 11, no. 4, pp. 291– 293, Apr. 2015.

[32] H. Ahn, S. Oh, G. S. Kim, S. Jung, S. Park, and J. Kim, “Hallucinationaware generative pretrained transformer for cooperative aerial mobility control,” in Proc. IEEE Global Commun. Conf. (GLOBECOM), Taipei, Taiwan, Dec. 2025, pp. 5369–5374.

[33] H. Ahn, H. Lee, B. Kang, J. Choi, Y. Yoon, and J. Kim, “Transformerbased offline reinforcement learning for intelligent torpedo evasion in autonomous underwater vehicles,” IEEE Access, vol. 14, pp. 135 721– 135 736, Aug. 2026.

[34] G. S. Kim, J. Chung, S. Park, S. Jung, T. Q. Duong, and J. Kim, “Quantum learning for autonomous aircraft control: A reinforcement learning perspective,” IEEE Commun. Mag., vol. 64, no. 10, pp. 46–53, Oct. 2026.

[35] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, Oct. 2011.

[36] H. Baek, S. Park, and J. Kim, “Logarithmic dimension reduction for quantum neural networks,” in Proc. ACM Int. Conf. Inf. Knowl. Manage. (CIKM), Birmingham, United Kingdom, Oct. 2023, pp. 3738–3742.

[37] F. Arute et al., “Quantum supremacy using a programmable superconducting processor,” Nature, vol. 574, no. 7779, pp. 505–510, Oct. 2019.