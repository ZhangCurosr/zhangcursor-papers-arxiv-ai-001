# Look It Up: Multi-bit LLM Watermarking via Codebook Search

Jarosław Janas<sup>1</sup>, Paweł Morawiecki<sup>1</sup>, and Josef Pieprzyk<sup>1,2</sup>

<sup>1</sup>Institute of Computer Science, Polish Academy of Sciences, 01-248 Warsaw, Poland <sup>2</sup>CSIRO, Marsfield, NSW 2122, Australia

Corresponding e-mail: jaroslawjanas.dev@gmail.com Code: https://github.com/jaroslawjanas/llmw-codebook

This work was supported in part by the National Science Centre, Poland, under Grant OPUS 2023/49/B/ST6/02580. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract—Attributing short LLM-generated texts to a specific source remains challenging for existing multi-bit watermarking methods, which often require long generations or costly decoding to recover payloads reliably. We propose a codebook-based watermarking scheme that maps an n-bit payload to a longer u-bit codeword and embeds it during generation through tokenlevel logit biasing. During extraction, the received bit sequence is matched to the nearest codeword in a precomputed secret codebook using Hamming or Levenshtein distance. Experiments show that our method achieves near-perfect recovery for 16–24 bit payloads on 200-token generations and strong performance on 127-token generations for base models. Results are weaker on some instruction-tuned models with very short texts, where lowentropy output distributions reduce embedding reliability. We also study a BCH-based variant that avoids codebook storage but generally reduces robustness. The Levenshtein variant improves resilience to deletions and light paraphrasing, while insertions and stronger rewrites remain challenging. We analyze collision probability, codeword separation, extraction time, tokenization consistency, and text quality, highlighting both the practicality and limitations of codebook-based watermarking for short-form attribution.

Index Terms—Large language models, text watermarking, multi-bit watermarking, content provenance, attribution, errorcorrecting codes, AI-generated text detection.

## I. INTRODUCTION

L <sup>ARGE</sup> <sup>Language</sup> <sup>Models</sup> <sup>(LLMs)</sup> <sup>have</sup> <sup>become</sup> <sup>widely</sup>adopted for tasks ranging from content drafting and adopted for tasks ranging from content drafting and summarisation to dialogue and code generation. As the quality of machine-generated text has improved, so has the difficulty of distinguishing it from human writing—raising concerns about misuse in misinformation, plagiarism, and large-scale deceptive content such as fake product reviews. Early watermarking methods addressed the detection problem: zero-bit schemes [1], [2] embed a statistical signal that allows a verifier to determine whether a given text was produced by an LLM, but they do not carry any identifying information about the source.

Multi-bit watermarking goes further by embedding a payload—such as a user or session identifier—directly into the generated token sequence, enabling attribution to a specific source. Several such schemes have since been proposed (see Section II), yet they are predominantly evaluated on texts of approximately 200 tokens, as longer sequences provide more redundancy for reliable extraction. This leaves the sub-200 token space largely unaddressed. Product reviews, social media posts, and short comments—all common targets for AIgenerated deception—routinely fall well below this threshold. Studies estimate that 16–33% of online reviews are inauthentic [3], with recent analyses putting this figure at roughly 30% as of 2025, a problem projected to cost hundreds of billions of dollars globally [4]. Reliably watermarking these short-form texts remains an open and practically important challenge.

## II. CURRENT APPROACHES AND LIMITATIONS

Several recent works have explored multi-bit watermarking schemes for LLM-generated text beyond the simple zero-bit (detector) watermarks [2], [5]–[8]. Table I summarizes the performance of representative methods from the literature. Overall, there is a clear trade-off between accuracy and efficiency in existing approaches.

On one end of the spectrum, enumeration-based methods treat message extraction as a search problem, examining many possible bit sequences to find which one was embedded. Fernandez et al. [7] propose an enumeration approach that achieves very high extraction accuracy (nearly 99% success on 16-bit payloads) by exhaustively searching for the hidden message. However, this comes at the cost of extremely slow decoding: extracting a 32-bit watermark from a single short text was reported to take on the order of 8 hours, making it impractical for large-scale deployment. Another multi-bit method by Wang et al. [2] improves upon this with a more efficient coding strategy, managing to extract 32-bit messages in about 2.3 hours per text in experiments. While faster, even a two-hour extraction for one short document is far too slow for real-time or mass detection scenarios. Moreover, both of these methods exhibit exponentially growing extraction time as the message length increases, due to their reliance on combinatorial search. This lack of scalability means that although they can reach high accuracy, they are not suitable for embedding more than a handful of bits in short texts.

At the other end of the design spectrum are approaches that prioritize efficiency by avoiding expensive search. A notable example is the method of Yoo et al. [5], which uses a position-based token allocation technique to encode bits on the fly during text generation. This approach enables nearly instantaneous extraction as it does not require iterating over candidate messages. However, the drawback is significantly lower accuracy in recovering the full message, especially as the payload size grows. For instance, Yoo et al.’s watermark could only recover a 16-bit message from a short text about 74% of the time, dropping to around 30% success for 24- bit payloads. Such error rates are far from acceptable for reliable tracing—losing even a single bit can corrupt the entire extracted identifier. In summary, purely efficient schemes tend to underperform on short texts, because a short generated passage provides limited opportunities to redundantly encode each bit and ensure error-free decoding.

Table I. Comparison of multi-bit watermarking methods on accuracy (match rate). The match rate is the percentage of watermarked texts from which the full hidden message is correctly extracted. Texts were generated with Llama 2 7B and embedded message was 24 bits long.
<table><tr><td>Method</td><td>Match Rate</td><td>Limitation</td></tr><tr><td>Fernandez et al. [7]</td><td>99.6%</td><td>very long or infeasible extraction time</td></tr><tr><td>Wang et al. [2]</td><td>98.8%</td><td>very long or infeasible extraction time</td></tr><tr><td>Yoo et al. [5]</td><td>30.4%</td><td>much below 98–100% match rate</td></tr><tr><td>Lin et al. [8]</td><td>90.0%</td><td>below 98-100% match rate</td></tr><tr><td>Qu et al. [6]</td><td>96.0%</td><td>below 98-100% match rate</td></tr></table>

A key observation from Table I is that no existing method meets the >98% extraction success criterion on short texts without incurring prohibitive extraction time. The enumeration-based watermark of Fernandez et al. [7] achieves near-perfect accuracy on short passages, but only by tolerating an infeasible decoding latency. The fast method of Yoo et al. [5] sacrifices too much accuracy to be useful for robust tracing. Wang et al.’s [2] balanced coding strikes a middle ground, but still does not reach the reliability needed for high-stakes use cases. Lin et al. [8] propose DERMARK, which uses dynamic segment allocation to improve capacity on short texts; in roughly 100-token outputs it reports about 90% success for 24-bit payloads, reducing tokens-per-bit compared to prior art, yet still falling short of the >98% target in this regime. Very recently, Qu et al. [6] introduced a watermarking scheme that combines ideas from both camps—achieving 96– 98% extraction success on 20–24 bit messages in ∼200- token texts, while keeping extraction time around 1 second. This represents the state of the art to date, yet even this approach sees the success rate dip to 94% for 32-bit payloads in a paragraph of text, and the performance on extremely short texts (e.g., single-sentence inputs) has not been fully demonstrated. In other words, the problem is not completely solved: there remains a practical gap in being able to robustly watermark and decode very short AI-generated texts at scale.

As shown above, existing techniques each fail to fully satisfy the needs of short-text watermarking. An ideal solution would combine the near 100% reliability of exhaustive methods with the speed of lightweight methods. This is precisely the challenge our work tackles.

## A. Contributions

Our main contributions are as follows:

• We developed a novel watermarking scheme with very high extraction success on multi-bit watermarks, even for short AI-generated texts, without sacrificing efficiency. We achieved a very high match rate with a 24-bit payload capacity, even on texts as short as 127 tokens in most settings. Our method also performs well at the standard 200-token evaluation length, matching or outperforming existing approaches.

• We showed that our main variant with Levenshtein distance provides a meaningful level of robustness and for some attacks is more robust than the state-of-the-art solutions.

• We investigated variants with Hamming and Levenshtein distances, analyzing their codeword collision and separation features. Additionally, we proposed the BCH variant, which does not require a pre-generated codebook. The BCH variant offers longer watermarks at the expense of lower accuracy and robustness.

## III. MULTI-BIT WATERMARKING SCHEME WITH CODEBOOK

To overcome the trade-off between accuracy and efficiency, we introduce a novel scheme centered around a pre-computed codebook. Enumeration-based methods can be prohibitively slow not necessarily because they consider more candidate watermarks, but because evaluating each candidate may require expensive decoding. Our method still searches over all 2<sup>n</sup> watermark identities, but it reduces extraction to a nearestneighbor lookup against a precomputed codebook of u-bit codewords, where distance computations are very cheap and the codewords are well separated. This approach combines the robustness of having a long, redundant signal with the search efficiency, making it ideal for short texts where every token counts.

Our method consists of three stages: an offline Codebook Generation step, and online Watermark Embedding and Watermark Extraction steps.

## A. Codebook Generation

The first step, performed offline, is to generate a codebook C that serves as a shared secret between the embedding and extraction processes. As detailed in Algorithm 1, the codebook is a map from an n-bit watermark space to a u-bit codeword space, where u > n.

![](images/2fb7c3b5432d48ba7c5bff94d4647fc1573105cdc2b0238dbe4605344dc16782.jpg)

Fig. 1. Our watermarking scheme. A codebook is pre-generated offline and later used by the embedding and extraction algorithms. The embedding phase embeds a long codeword (e.g., 127 bits) into the generated text. During the extraction, a noisy version of codeword is first obtained and then the closest codeword is searched in the codebook. Once the closest codeword is identified, the corresponding watermark is extracted. The detection decision (watermarked vs. non-watermarked) is based on the distance threshold. A distance is measured by the Hamming or Levenshtein distance.  
Algorithm 1 Codebook Generation   
1: Input: secret key ${ \overline { { \kappa , } } }$ parameters $n , u \in \mathbb { N }$   
2: Output: codebook $\mathcal { C } : \{ 0 , \dotsc , 2 ^ { n } - 1 \}  \{ 0 , 1 \} ^ { u }$   
3: G, r ← DeriveParams $( K , n , u )$ $\triangleright n \times u$ binary matrix and u-bit offset (Appendix C)   
4: $c \gets$ empty map   
5: for $i \gets 0$ to $2 ^ { n } - 1$ do   
6: $W  i$   
7: $L \gets$ bits(W) · G mod $2 \oplus \mathbf { r }$ ▷ random linear code with affine offset   
8: $\mathcal { C } [ W ]  L$   
9: end for   
10: return C

Specifically, we generate a table containing $2 ^ { n }$ entries. Each entry maps a potential watermark W (an integer from 0 to $2 ^ { n } - 1 )$ to a unique binary vector $L _ { W }$ of length u. To create this mapping, we use a random linear code over $\mathrm { G F } ( 2 )$ : we first derive a random $n \times u$ binary generator matrix G and a u-bit offset vector r from the secret key K (see Appendix C for details). Each codeword is then computed as $L _ { W } = \mathrm { b i t s } ( W ) \cdot \mathbf { G }$ mod $2 \oplus \mathbf { r } ,$ where $\mathrm { b i t s } ( W )$ is the n-bit binary representation of W. This process ensures that each message W corresponds to a unique codeword, and without the secret key K, the codebook cannot be reconstructed.

The redundancy introduced by making the codeword length u greater than the message length n is the key to our scheme’s error resilience and effectiveness. The difference, $u - n$ , can be seen as providing redundancy analogous to parity or check bits in classical error-correcting codes. By expanding each n-bit message into a u-bit codeword, we ensure that the valid codewords are, with high probability, far apart from each other in terms of the Hamming or Levenshtein distance. This separation allows us to correctly identify the original watermark even if several bits of the codewords are not encoded properly during generation. The greater the difference $u - n$ , the more errors can be tolerated, improving the chances of perfect watermark recovery. However, the choice of these parameters presents a trade-off: choosing too small of an n limits the total number of unique watermarks that can be embedded, while choosing too large of a u requires longer generated texts.

The generation and storage of the codebook both have a time and memory complexity of $O ( 2 ^ { n } \cdot u )$ , as the process must iterate through all $2 ^ { n }$ watermarks to generate a u-bit codeword for each. This is a one-time, offline cost. For the parameters used in our testing $( n ~ = ~ 2 4 , ~ u ~ = ~ 1 2 7 )$ , the codebook requires storing $2 ^ { 2 4 } \times 1 2 7$ bits, which amounts to a manageable 256 MB of memory. However, scaling to a 32-bit message space $( n = 3 2 )$ , which can support over 4 billion unique watermarks, would increase this requirement to $2 ^ { 3 2 } \times 1 2 7$ bits, or a substantial 64 GB.

Algorithm 2 Watermark Embedding   
1: Input: large language model $L L M ,$ vocabulary V, bias $\delta ,$ prompt $\overline { { s ^ { ( - N _ { p } ) } , \ldots , s ^ { ( - 1 ) } } }$ , secret key $\kappa ,$ hash hash, codebook   
$\mathcal { C } : \{ 0 , \dotsc , 2 ^ { n } - 1 \}  \{ 0 , 1 \} ^ { u }$   
2: Output: generated tokens $\tau$ with embedded watermark   
3: Choose a watermark $W \in$ domain(C) ▷ select a watermark from the codebook C   
4: $L _ { W }  { \mathcal { C } } [ W ] ; \quad u  | L _ { W } | ;$ unpack $L _ { W }$ as $\left( b _ { 0 } , \ldots , b _ { u - 1 } \right)$   
5: $\mathcal { T }  [ ] , \quad t  0$ ▷ start embedding at the first generated token   
6: while $t < \tau$ u do   
hash(K), if $t = 0$   
7: seed $ \{ _ { \mathrm { h a s h } ( s ^ { ( t - 1 ) } , K ) }  $ ▷ use hash(K) only for the very first token   
, otherwise   
8: Using seed, deterministically split $\nu$ into two equal sets $\mathcal { V } _ { 0 }$ and $\nu _ { 1 }$   
9: $v  \mathit { L L M } ( s ^ { ( - N _ { p } ) } , \ldots , s ^ { ( t - 1 ) } )$ ▷ generate logits   
10: if $b _ { t } = 0$ then   
11: $\forall w \in \mathcal { V } _ { 0 } : \ \hat { v } [ w ]  { v } [ w ] + \delta$   
12: else   
13: $\forall w \in \mathcal { V } _ { 1 } : \ \hat { v } [ w ]  { v } [ w ] + \delta$   
14: end if   
15: s<sup>(t)</sup> ← argmax(ˆv) ▷ greedy sampling after bias   
16: Append $s ^ { ( t ) }$ to $\begin{array} { r l } { \mathcal { T } ; } & { { } ~ t \gets t + 1 } \end{array}$   
17: end while   
18: return $\tau$

## B. Watermark Embedding

The embedding process, shown in Algorithm 2, encodes the watermark during the LLM’s text generation phase. It modifies the token selection process to encode the bits of a chosen codeword $L _ { W }$

Given a watermark $W$ to embed, we first retrieve its corresponding u-bit codeword $L _ { W }$ from the codebook C. Then, for the first u tokens to be generated, we embed bits from $L _ { W }$ . At each token generation step t, we follow the standard practice from Kirchenbauer et al. [1], we use the secret key K and the previously generated token $s ^ { ( t - 1 ) }$ to deterministically partition the model’s vocabulary V into two sets, which we call $\mathcal { V } _ { 0 }$ and V<sub>1</sub>.

The t-th bit of the codeword, $b _ { t } .$ , determines which vocabulary set is favoured. If $b _ { t } = 0 .$ , we add a large positive bias δ to the logits of all tokens in $\mathcal { V } _ { 0 }$ . Likewise, if $b _ { t } = 1$ , we bias the logits of tokens in $\mathcal { V } _ { 1 }$ . Having modified the logits, we use greedy sampling to select the next token $s ^ { ( t ) }$ . The strong bias $( \delta = 6 )$ makes it likely that the selected token $s ^ { ( t ) }$ will be a member of the favoured vocabulary set, therefore encoding the bit $b _ { t } .$ . This procedure is repeated for all bits of the codeword, embedding the entire sequence $L _ { W }$ into the tokens of the generated text $\tau$ . To mitigate tokenization inconsistency during embedding, we apply a stepwise consistency check (Appendix F).

## C. Watermark Extraction

The extraction process, detailed in Algorithm 3, consists of two main phases: reconstructing the embedded codeword and finding the closest match in the codebook.

To reconstruct the embedded codeword, we iterate over the first u tokens of the given text $\tau$ . For each token $s ^ { ( t ) }$ the vocabulary partitions $\mathcal { V } _ { 0 }$ and $\nu _ { 1 }$ are regenerated based on the previous token $s ^ { ( t - 1 ) }$ and the secret key K. We then check which group the token $s ^ { ( t ) }$ belongs to, if $s ^ { ( t ) } \in \mathcal { V } _ { 1 }$ it is decoded as 1, likewise if $s ^ { ( t ) } \in \mathcal { V } _ { 0 }$ it is decoded as 0. This decoding continues until a sequence of u bits is recovered, which now acts as a codeword candidate ${ \widehat { L } } .$

In the second phase, we find the closest match for this candidate codeword $\widehat { L }$ within the pre-computed codebook C. We iterate through every watermark-codeword pair $( W , L _ { W } )$ in the codebook and compute the distance, $d ( \widehat { L } , L _ { W } )$ . Our main and recommended variant uses the Levenshtein distance but we also examine the Hamming distance. The watermark $W ^ { \star }$ corresponding to the codeword $L _ { W } .$ ⋆ that has the minimum distance to $\widehat { L }$ is selected as the most likely embedded message.

The extraction algorithm returns this best-match watermark $W ^ { \star }$ and its corresponding minimum distance, $d _ { \mathrm { m i n } }$ . This distance score serves as a confidence metric. A user can set a threshold η for this distance. If $d _ { \operatorname* { m i n } } \leq \eta .$ , the watermark $W ^ { \star }$ is accepted. If the distance exceeds the threshold, the text can be classified as not containing a valid watermark or as having a signal that is too corrupted for reliable decoding. This errorcorrection capability stems directly from the redundancy in our codebook design $( u > n )$ . Because the valid codewords $L _ { W }$ are generally some distance apart from each other in the Hamming and Levenshtein space, the system can tolerate a number of bit errors during generation (e.g., when low-entropy, peaky logits make the bias insufficient to steer selection into the intended partition) and still correctly identify the original watermark.

Algorithm 3 Watermark Extraction   
1: Input: tokens $\overline { { \mathcal { T } = \left( s ^ { ( 0 ) } , s ^ { ( 1 ) } , \dots \right) } }$ , vocabulary V, secret key K, hash hash, codebook C, codeword length u   
2: Output: $( W ^ { \star } , d _ { \operatorname* { m i n } } )$ where $W ^ { \star } \in$ domain(C) and $d _ { \operatorname* { m i n } } = d ( \widehat { L } , L _ { W ^ { \star } } )$   
3: for $t \gets 0$ to $\underline { { u } } - 1$ do   
hash(K), if $t = 0$   
4: seed ←   
hash(s<sup>(t−1)</sup>, K), otherwise   
5: Using seed, deterministically split V into two equal sets $\mathcal { V } _ { 0 }$ and $\nu _ { 1 }$   
$\widehat { b } _ { t } \gets \{ { 1 , }$ if $s ^ { ( t ) } \in \mathcal { V } _ { 1 }$   
6:   
otherwise   
7: end for   
8: $\widehat { L } \gets ( \widehat { b } _ { 0 } , \dots , \widehat { b } _ { u - 1 } )$   
▷ Search the stored codebook for the closest codeword   
9: $( W ^ { \star } , d _ { \mathrm { m i n } } ) \gets ( \bot , \infty )$   
10: for $( W , L _ { W } )$ in C do   
11: $d  d ( \widehat { L } , L _ { W } )$ ▷ calculate the distance   
12: if $d < d _ { \operatorname* { m i n } }$ then   
13: $( W ^ { \star } , d _ { \operatorname* { m i n } } ) \gets ( W , d )$   
14: end if   
15: end for   
16: return $( W ^ { \star } , d _ { \operatorname* { m i n } } )$ ▷ a distance threshold may be applied as desired

## D. Codeword Collision and Separation

Because each codeword is computed as $L _ { W } = \mathrm { b i t s } ( W )$ G mod $2 \ \oplus \ \mathbf { r }$ (see (4)), two distinct watermarks $W _ { i } \neq W _ { j }$ collide if and only if $( \mathrm { b i t s } ( i ) \oplus \mathrm { b i t s } ( j ) ) \cdot \mathbf { G } = \mathbf { 0 }$ (mod 2). Since G is drawn uniformly at random, every nonzero linear combination of its rows is a uniformly random u-bit vector, so each pairwise collision occurs with probability $2 ^ { - u }$ . A union bound over all ${ \binom { 2 ^ { n } } { 2 } } < 2 ^ { 2 n - 1 }$ pairs gives:

$$
\mathrm { P r } [ \mathrm { a n y ~ c o l l i s i o n } ] \ \leq \ 2 ^ { 2 n - u - 1 } .\tag{1}
$$

For our parameters $n = 2 4 , u = 1 2 7$ , this evaluates to $2 ^ { - 8 0 } \approx$ $1 0 ^ { - 2 4 }$ , making collisions negligible in practice.

Beyond collisions, the minimum Hamming distance $d _ { \mathrm { m i n } }$ between any two codewords determines the scheme’s error tolerance: nearest-neighbour decoding correctly identifies a watermark as long as at most $t = \lfloor ( d _ { \operatorname* { m i n } } - 1 ) / 2 \rfloor$ bits are corrupted. For a random binary linear $[ u , n ]$ code, the Gilbert– Varshamov (GV) bound [9] guarantees the existence of a code with $d _ { \operatorname* { m i n } } ~ \geq ~ d _ { \mathrm { G V } }$ satisfying $H _ { 2 } ( d _ { \mathrm { G V } } / u ) = 1 - n / u ,$ where $H _ { 2 }$ is the binary entropy function. Evaluating this for our parameters yields $d _ { \mathrm { G V } } ~ \approx ~ 3 2$ for $u \ = \ 1 2 7$ and $d _ { \mathrm { G V } } \approx 6 0$ for $u = 2 0 0$ . In our experiments (Appendix E), the measured minimum distances are $d _ { \operatorname* { m i n } } = 3 5$ and $d _ { \operatorname* { m i n } } = 6 2$ respectively—both exceeding the GV bound. This translates to an error tolerance of $t = 1 7$ bit errors for $u = 1 2 7$ and t = 30 for $u = 2 0 0$

Finally, the linear construction offers a structural advantage over independent random (RNG) codewords. In a linear code, $d _ { \mathrm { m i n } }$ is determined by the minimum Hamming weight among only $2 ^ { n } - 1$ nonzero codewords, whereas RNG codes must take the minimum over all ${ \binom { 2 ^ { n } } { 2 } } \approx 2 ^ { 2 n }$ pairwise distances—a much larger set subject to a stronger birthday-type effect. At $n = 2 4$ and $u = 1 2 7$ , this yields $d _ { \operatorname* { m i n } } = 3 5$ for the linear code versus a sampled minimum of 25 for RNG, a 40% advantage. A detailed experimental comparison across all values of n is provided in Appendix E.

## IV. VARIANT WITH BCH CORRECTION CODE

The codebook-based watermarking scheme described in Section III offers high accuracy and robustness for short texts, but it has notable limitations when scaling to larger watermark sizes. For example, with a 32-bit watermark (supporting over 4 billion unique identifiers), the codebook would require storing $2 ^ { 3 2 }$ entries, which is computationally and memoryintensive during extraction. This makes the codebook approach impractical for very large watermark spaces, as discussed in Section VI.

To address this, we introduce a variant that uses a Bose– Chaudhuri–Hocquenghem (BCH) error-correcting code instead of a codebook. The overall workflow remains similar: during encoding, a binary watermark is transformed into a binary codeword using BCH encoding, and this codeword is embedded into the text using the same token-level biasing strategy as in Algorithm 2. During extraction, the received codeword is decoded using the BCH algorithm to recover the original watermark. Unlike the codebook method, where the Hamming/Levenshtein distance to the closest codeword serves as a confidence metric, here the number of bit flips (i.e., the difference between the retrieved codeword and the codeword reconstructed from the decoded watermark) provides a measure of reliability. This approach avoids the need for a large codebook and enables efficient watermarking for larger payload sizes, at the cost of potentially lower error tolerance compared to the codebook method.

Table II. Comparison of methods on match rate (%) for different watermark lengths (in bits) on 200 tokens. Evaluated on the OpenGen dataset [10]. <sup>†</sup>BCH uses 127 tokens (native BCH codeword length) for 16–24-bit watermarks to maximize the error correction rate; the 32-bit watermark uses 200 tokens with code shortening (see Appendix D). NA denotes that a method was too computationally expensive for the whole dataset.
<table><tr><td>Watermark size</td><td>16</td><td>20</td><td>24</td><td>32</td></tr><tr><td>Fernandez et al. [7]</td><td>99.6</td><td>99.2</td><td>98.0</td><td>NA</td></tr><tr><td>Wang et al. [2]</td><td>98.8</td><td>98.4</td><td>97.2</td><td>NA</td></tr><tr><td>Yoo et al. [5]</td><td>73.6</td><td>49.2</td><td>30.4</td><td>8.4</td></tr><tr><td>Cohen et al. [11]</td><td>88.8</td><td>78.4</td><td>65.6</td><td>27.2</td></tr><tr><td>Qu et al. [6]</td><td>98.0</td><td>97.6</td><td>96.0</td><td>94.0</td></tr><tr><td>Ours (BCH†)</td><td>99.1</td><td>98.6</td><td>97.6</td><td>91.1</td></tr><tr><td>Ours (Hamming)</td><td>99.9</td><td>99.8</td><td>99.9</td><td>NA</td></tr><tr><td>Ours (Levenshtein)</td><td>99.8</td><td>99.8</td><td>99.8</td><td>NA</td></tr></table>

## V. EXPERIMENTAL RESULTS

We compared our approach with the other top five methods. The results of the experiments are given in Table II. Baseline numbers are taken from [6] and the experiment setting we followed is identical to the one used in that work. Watermarks of 16 to 32 bits are embedded in generated texts of 200 tokens. Texts were generated with Llama 2 7B and prompts were taken from the OpenGen dataset [10]. All experiments used a consistent logit bias of δ = 6. We used greedy sampling, the same as [6] in their experiments.

The primary metric is match rate, defined as the percentage of texts for which the full multi-bit watermark is correctly extracted.

Our codebook-based schemes achieve nearly a 100% match rate across most settings, outperforming all other methods. For the 32-bit watermark, we obtain the second-best result (BCH variant), since the codebook-based algorithms were too computationally expensive (see Section VI).

We also evaluated our schemes in a more demanding setting with shorter 127-token texts embedding longer, 24- bit watermark. As shown in Table III, the codebook methods achieve a high match rate across several models when using both Hamming and Levenshtein distance metrics. The table also compares against BCH variant. The codebook method consistently provides a higher match rate. The advantage is especially significant on instruction-tuned models, where the BCH scheme’s performance degrades rapidly while our method remains robust. This highlights the superior error resilience of our pseudorandom codebook approach over the structured constraints of BCH codes, particularly in the challenging low-entropy settings of aligned models.

Generally, performance is comparatively lower on instruction-tuned models. We attribute this to the low-entropy, peaky logit distributions produced by these models after fine-tuning with methods like RLHF, which can overpower the logit-biasing mechanism.

## A. Extraction Time

Extraction consists of decoding the bit sequence from text and searching the codebook for the closest match. Decoding is negligible (tokenization and per-token classification); the bottleneck is the brute-force search over all $2 ^ { n }$ entries. The BCH variant bypasses this search entirely via algebraic decoding. Table IV reports the extraction time for n=24 (≈16.8M entries), averaged over 10,000 queries. The Hamming variant uses the Faiss library on CPU with 8 OpenMP threads; the Levenshtein variant uses a custom Triton kernel on the GPU.

Table III. Match rate (%) at $\mathrm { F P R } \leq 1 \%$ for a 24-bit watermark. Evaluated on Essays-With-Instructions dataset [12] and HC3 dataset [13].
<table><tr><td>Model</td><td>Hamming</td><td>Levenshtein</td><td>BCH</td></tr><tr><td colspan="4">127 tokens</td></tr><tr><td>Llama 2 7B</td><td>99.3</td><td>97.5</td><td>98.3</td></tr><tr><td>Llama 3.1 8B</td><td>100</td><td>100</td><td>99.7</td></tr><tr><td>Llama 3.2 3B</td><td>100</td><td>100</td><td>99.6</td></tr><tr><td>Llama 3.2 3B Inst.</td><td>90.2</td><td>61.6</td><td>56.5</td></tr><tr><td>Mistral 7B</td><td>99.7</td><td>99.4</td><td>99.4</td></tr><tr><td>Mistral 7B Inst.</td><td>96.3</td><td>76.4</td><td>64.5</td></tr><tr><td colspan="4">200 tokens</td></tr><tr><td>Llama 2 7B</td><td>100</td><td>99.6</td><td>99.3</td></tr><tr><td>Llama 3.1 8B</td><td>99.6</td><td>99.6</td><td>99.6</td></tr><tr><td>Llama 3.2 3B</td><td>100</td><td>100</td><td>99.6</td></tr><tr><td>Llama 3.2 3B Inst.</td><td>99.6</td><td>90.6</td><td>39.6</td></tr><tr><td>Mistral 7B</td><td>100</td><td>99.7</td><td>98.7</td></tr><tr><td>Mistral 7B Inst.</td><td>100</td><td>96.7</td><td>43.8</td></tr></table>

Table IV. Average extraction time per query for n=24 (≈16.8M codebook entries). Hamming and Levenshtein benchmarked on an AMD Epyc 7302 (16C/32T, 3.0 GHz) with a single Nvidia RTX 5090 (32 GB).
<table><tr><td>Variant</td><td>u=127</td><td>u=200</td></tr><tr><td>Levenshtein (GPU, Triton)</td><td>5.8 ms</td><td>75.5 ms</td></tr><tr><td>Hamming (CPU, Faiss)</td><td>4.1 ms</td><td>13.3 ms</td></tr><tr><td>BCH (CPU, algebraic)</td><td>2.3 ms</td><td>3.5 ms</td></tr></table>

Even in the slowest setting, extraction takes under 0.1 seconds—practical for real-time use. For comparison, other methods that reach comparable match rates are orders of magnitude slower for watermarks upwards of 20 bits [6].

## B. Quality of Watermarked Texts

Our quality evaluation (Appendix G) indicates that the watermark has only a modest effect on perceived text quality. In an LLM-as-a-Judge study on review-style and social-post prompts, watermarked outputs were rated similarly to nonwatermarked ones and even received a small authenticity preference. Perplexity increases under watermarking, but the values remain moderate. Lexical-diversity analysis further shows that watermarked text is consistently more diverse and less repetitive, suggesting that the bias introduces phrasing variation rather than obvious degradation. Overall, these results support that the watermark preserves readable, socially plausible text while slightly increasing diversity.

## C. Attack Robustness

We consider an attacker whose aim is to remove the watermark and alter the multi-bit payload. The attacker can edit the watermarked text while preserving the semantics and readability.

![](images/ecbd508a590f401e8488b411aea550d03445f2841c8aac4238085ab5b54ffe3b.jpg)  
Fig. 2. Match rate under attacks, evaluated on generations from Llama 2 7B with the OpenGen dataset [10]. The attacker’s budget is 10% of a total number of tokens for a given text. Baseline values are estimated from Figure 6 of Qu et al. [6].

Because we focus on short texts generated at scale, we exclude attacks that require substantial human effort (e.g., manual rewriting) or heavy paraphrasing by other models. In our regime, a strong global rewrite often replaces most of the surface form and becomes very close to generating a new text with the same intent. An attacker capable of doing this can often bypass watermarking by generating a fresh unwatermarked text directly.

We apply four common attacks: word insertion, word deletion, substitution with synonyms and paraphrasing (details in Appendix B). Following a common setting, we modify 10% tokens. Figure 2 shows a comparison between our recommended variant (Levenshtein) and baselines [5], [6], [11].

Our method offers a meaningful level of robustness, particularly against the word deletion and light paraphrasing attacks (10% modified tokens). The word insertion attack is more challenging because it increases the Levenshtein distance to the correct codeword and the extraction algorithm recovers an incorrect watermark more often. Yet, it still achieves nearly 60% match rate.

Hamming and BCH variants are brittle against the considered attacks. This is because they compare codewords bitby-bit at fixed positions without considering positional shifts. Even a single insertion or deletion causes a frameshift that desynchronizes all subsequent bit comparisons. Therefore, for applications where robustness matters, we recommend a scheme which uses the Levenshtein distance.

## VI. LIMITATIONS

A main limitation of our proposed scheme is the size of the codebook, which directly impacts extraction efficiency. While the watermark space, determined by n, can theoretically be made large (e.g., n = 32 to support over 4 billion unique watermarks), this creates a practical bottleneck during the extraction phase. The extraction process requires computing a distance between the candidate codeword $\widehat { L }$ and every codeword $L _ { W }$ in the codebook C. For a large n, this computation becomes prohibitively expensive, undermining the goal of a fast and scalable detection system. Therefore, a trade-off exists between the watermark’s capacity (2<sup>n</sup>) and the computational cost of extraction.

The limitation can be circumvented by requiring longer texts for embedding a watermark. For example, instead of common 200-token text, we would use 256 tokens. Then, we can easily embed 32-bit watermark by dividing the text into two parts and embedding 16-bit watermark in each part.

## VII. CONCLUSION

We proposed a novel multi-bit watermarking scheme for LLM-generated texts. We showed that for watermark lengths up to 24 bits, our approach can achieve both near-100% match rates and practical extraction speed. For most models, this stays true even for very short, 127-token texts. This makes the method well suited to watermarking short comments, product reviews or social media posts — all potential AI-generated fake content. Our recommended Levenshtein variant offers solid robustness against common attacks. We also investigated other variants that may be preferable when speed is the primary concern (Hamming) or when longer watermarks are required (BCH).

## APPENDIX A AUROC

Figure 3 shows the combined ROC curves for our three watermarking variants. All three methods achieve near-perfect discrimination between watermarked and non-watermarked text, with AUROC values above 0.998. This evaluation was performed on Llama 2 7B generations from the OpenGen dataset [10] with a 24-bit watermark (n = 24). The Hamming and Levenshtein variants are evaluated on 200 tokens; the BCH variant uses u = 127 tokens, its native BCH codeword length, to maximize the error correction rate (see Appendix D).

## APPENDIX B

## ATTACK ROBUSTNESS SETUP

The attack robustness experiments in Section V-C apply adversarial modifications at the word level to the watermarked text itself. This reflects a realistic threat model in which an attacker edits the generated text while attempting to preserve its meaning and readability.

All four attacks are governed by an attack budget, a fraction of tokens to be affected. Given a watermarked text of length T tokens (measured by the model’s tokenizer without special tokens), the target number of tokens to modify is B = ⌊attack\_budget · T⌋. In our experiments we set the attack budget to 0.1 (10% of tokens).

![](images/f2e99a38e707ede34b21f47e350b7a12c7323ac7a7a7e0805923f5d191f1c574.jpg)  
Fig. 3. ROC curve with reported AUROC. Marked points have 1% False Positive Rate.

1) Word deletion: The text is split into words by whitespace. The word indices are shuffled uniformly at random and words are deleted one by one, accumulating the token count of each removed word, until the cumulative count reaches or exceeds B. Because tokens are removed, the decoded bit sequence becomes shorter than u, causing positional shifts in the remaining bits.

2) Word insertion: Random English words are drawn from the NLTK words corpus (approximately 235,000 entries) and inserted at uniformly random positions in the text until the cumulative token count of inserted words reaches B. Inserted tokens lengthen the decoded bit sequence beyond u and shift all subsequent bit positions, making this attack particularly challenging for fixed-position distance metrics.

3) Synonym replacement: Words are part-of-speech tagged using the Penn Treebank tagset and mapped to WordNet categories (noun, verb, adjective, adverb). For each word with at least one single-word synonym in WordNet, we record it as a replacement candidate. Candidates are shuffled and replaced one by one with a randomly chosen synonym until the cumulative token count of the original words reaches $B .$ Unlike deletion and insertion, synonym replacement largely preserves text length, so its primary effect is to alter the decoded bit values at the affected positions without shifting neighbouring bits.

4) Paraphrasing: A contiguous region of words is selected by choosing a random centre position and expanding alternately left and right until the region spans at least B tokens. The selected region is sent to a large language model (GPT-oss-120B via an OpenAI-compatible API) with the instruction: “Rewrite the following text while preserving its meaning. Output only the rewritten text, nothing else.” (temperature $\ c = ~ 0 . 7 )$ . The paraphrased region replaces the original, while the surrounding prefix and suffix remain unchanged. This attack can simultaneously alter bit values, insert new tokens, and delete existing ones within the affected region.

5) Evaluation procedure: After each attack, the modified text is re-tokenized and decoded using the same green/red token classification as during embedding (Section III). If the decoded bit sequence exceeds u bits (which can occur after word insertion, synonym replacement, or paraphrasing), it is truncated to exactly u bits. The resulting bit sequence is searched against the codebook using the Levenshtein distance, and the watermark identity of the closest codeword is returned. We report the match rate: the fraction of samples for which the recovered watermark equals the originally embedded one.

6) Experimental parameters: The experiments in Figure 2 use watermarked texts generated by Llama 2 7B on the OpenGen dataset [10], with parameters $n = 2 0 , u = 2 0 0$ logit bias $\delta \ = \ 6 ,$ , and a linear codebook searched with the Levenshtein distance. The attack budget is 0.1. (These are the same attack settings as in Qu et al. [6].)

## APPENDIX C CODEBOOK CONSTRUCTION

The DeriveParams procedure in Algorithm 1 deterministically derives the generator matrix $\mathbf { G } \in \{ 0 , 1 \} ^ { n \times u }$ and offset vector $\mathbf { r } \in \{ 0 , 1 \} ^ { u }$ from the secret key K as follows.

Two independent seeds are obtained by hashing the key with distinct domain-separation tags:

$$
s e e d _ { \mathbf { G } } = { \mathrm { S H A 2 5 6 } } ( K \parallel " 1 \mathsf { n e a r } - \mathsf { G } ^ { \mathbf { u } } ) [ : 6 4 ]\tag{2}
$$

$$
s e e d _ { \mathbf { r } } = { \mathrm { S H A 2 5 6 } } ( K \parallel " { \textrm { l i n e a r } } - \mathbf { r } " ) [ : 6 4 ]\tag{3}
$$

where ∥ denotes string concatenation and [:64] selects the first 64 bits of the digest. Each seed initializes a pseudorandom number generator that fills its respective structure with i.i.d. Bernoulli(0.5) bits: $s e e d _ { \mathbf { G } }$ produces all n·u entries of $\mathbf { G } ,$ and $s e e d _ { \mathbf { r } }$ produces the u entries of r.

Given G and $\mathbf { r } ,$ the codeword for watermark W is:

$$
L _ { W } = \mathrm { b i t s } ( W ) \cdot \mathbf { G } \ \mathrm { m o d } \ 2 \ \oplus \ \mathbf { r }\tag{4}
$$

where bits $( W ) \in \{ 0 , 1 \} ^ { n }$ is the big-endian binary representation of W. The set of all codewords forms a coset of a random binary linear code, shifted by the offset r.

An alternative construction generates codewords independently at random. For each watermark identity i, a seed is derived as SH $\mathrm { A 2 5 6 } ( K \parallel i )$ and used to initialize a PRNG that samples u i.i.d. Bernoulli(0.5) bits. We refer to this as the RNG method. Because each codeword is drawn independently, the pairwise distances between codewords are random variables with no algebraic structure linking them.

## APPENDIX D

## BCH TOKEN LENGTH SELECTION

In Table II, the BCH variant uses only 127 tokens for 16–24- bit watermarks, seemingly giving up the advantage of the extra 73 bits available to all other methods. This section explains why: BCH codes have fixed native codeword lengths of the form $2 ^ { m } - 1$ , and adapting them to non-native lengths requires code shortening that reduces the relative error correction rate, making the longer codeword counterproductively less robust.

1) Native BCH lengths and code shortening: A BCH code of field degree m has a native codeword length $n _ { 0 } = 2 ^ { m } - 1$ For our experiments the relevant values are $n _ { 0 } = 1 2 7 \left( m = 7 \right)$ and $n _ { 0 } = 2 5 5 \ ( m = 8 )$ . When the desired output length u does not equal $n _ { 0 } .$ , the standard approach is code shortening: a BCH code with $n _ { 0 } > u$ is used, but $n _ { 0 } - u$ message positions are fixed to known zeros and omitted from the transmitted codeword. Shortening preserves the absolute correction capability t, but because the codeword shrinks while t stays fixed, the relative correction rate $t / u$ decreases compared to using the code at its native length.

2) Parameter comparison: At $u \ = \ 1 2 7$ the BCH code operates at its native length $( n _ { 0 } = 1 2 7 ) – - \mathbf { a }$ perfect fit requiring no shortening. $\mathrm { { A t } } \ u = 2 0 0 .$ , the nearest larger native length is $n _ { 0 } = 2 5 5$ , requiring 55 bits of shortening. Table V shows the resulting parameters for $n = 2 4 \colon$ although t rises from 21 to 27 in absolute terms, the correction rate $t / \tau$ u drops from 16.5% to 13.5%.

Table V. BCH parameters for a 24-bit watermark $( n = 2 4 )$ at two codeword lengths. At $u = 1 2 7$ the code operates at its native length with no shortening, yielding a higher relative error correction rate.
<table><tr><td></td><td> $u = 1 2 7$ </td><td> $u = 2 0 0$ </td></tr><tr><td>Native length no</td><td>127</td><td>255</td></tr><tr><td>Field degree m</td><td>7</td><td>8</td></tr><tr><td>Message capacity k</td><td>29</td><td>79</td></tr><tr><td>Shortening</td><td>0</td><td>55 bits</td></tr><tr><td>Correction capability t</td><td>21</td><td>27</td></tr><tr><td>Correction rate  $t / u$ </td><td>16.5%</td><td>13.5%</td></tr></table>

The same pattern holds for $n ~ = ~ 1 6$ and $n \ = \ 2 0 !$ the native $u = 1 2 7$ configuration yields correction rates of 18.1% compared to 14.5% and 13.5%, respectively, at $u = 2 0 0$

3) Empirical evidence: The instruction-tuned models in Table III showcase this effect. These models produce lowentropy logit distributions that resist the embedding bias, pushing per-bit error rates above 14%. At $u \ = \ 1 2 7$ the expected number of bit errors (≈ 20) falls just within the correction capability $\left( t \ \right) \ = \ 2 1 )$ , whereas at $u \ = \ 2 0 0$ the expected errors (≈ 29) exceed the limit $( t = 2 7 )$ . Concretely, for $n = 2 4 .$ , Llama 3.2 3B Instruct achieves a 56.5% match rate at $u \ = \ 1 2 7$ but only 39.6% at $u \ = \ 2 0 0 ;$ Mistral 7B Instruct shows a similar gap of 64.5% versus 43.8%. On base models, where per-bit error rates are lower (≈ 6%), both configurations succeed comfortably.

4) Why $n = 3 2$ uses $u = 2 0 0 \colon$ For a 32-bit message at $u = 1 2 7$ , the BCH code requires $k = 3 6$ , leaving fewer parity bits and reducing correction to $t = 1 5 \ ( t / u = 1 1 . 8 \% )$ . At $u = 2 0 0$ , the code uses a BCH(255, 87) code with shortening: the 32-bit watermark occupies the first 32 of the 87 message positions, the remaining 55 are set to known zeros, the full 255-bit codeword is encoded, and the 55 known-zero positions are removed to produce a 200-bit transmitted sequence. This yields $t = 2 6 ~ ( t / u = 1 3 . 0 \% )$ , reversing the advantage: for $n = 3 2$ , the shortened 200-bit code provides both better error resilience and a direct comparison with the other methods in Table II, which all use 200 tokens.

## APPENDIX E CODEBOOK SPREAD AND SEPARATION

To characterise how tightly packed the codebook becomes as the number of watermark identities grows, we measure the spread: the nearest-neighbour distance distribution across sampled codewords. For each value of n from 4 to 24, we sample $z \ = \ 5 0 , 0 0 0$ codewords (or all $2 ^ { n }$ if fewer exist) from the codebook and compute each codeword’s distance to its nearest neighbour, excluding itself. We then report the minimum, mean, and maximum of these nearest-neighbour distances. As n increases and the codebook grows denser in the binary space, we expect nearest-neighbour distances to decrease; the rate of this decrease and the spread between minimum and maximum reveal how uniformly the code distributes its codewords.

1) Results: Figures 4 and 5 compare the Linear and RNG construction methods under both Hamming and Levenshtein distance for codeword lengths $u \ = \ 1 2 7$ and $u \ = \ 2 0 0 .$ respectively. Under Levenshtein distance, both methods behave nearly identically: the mean curves overlap and the min–max bands are comparable. Under Hamming distance, however, a striking difference emerges. The Linear construction produces zero spread: the minimum, mean, and maximum nearestneighbour distances are all equal at every n, so no shaded band appears. The RNG method, by contrast, shows substantial spread, with the gap between minimum and maximum widening as n grows.

2) Why Linear codes have zero Hamming spread: This is a fundamental property of linear codes. For any two codewords $L _ { i }$ and $L _ { j }$ :

$$
L _ { i } \oplus L _ { j } = \left( \mathrm { b i t s } ( i ) \oplus \mathrm { b i t s } ( j ) \right) \cdot \mathbf { G } \mathrm { ~ m o d ~ 2 ~ }
$$

since the offset r cancels under XOR. The Hamming distance between $L _ { i }$ and $L _ { j }$ therefore equals the Hamming weight of the corresponding codeword in the underlying linear code. A well-known property of linear codes is that the minimum distance equals the minimum Hamming weight of any nonzero codeword. Because every codeword in the code can appear as a difference $L _ { i } \oplus L _ { j }$ for some pair, the nearest neighbour of every entry achieves this minimum weight. Hence the nearest-neighbour distance is identical for all codewords.

3) Why Levenshtein distance breaks this uniformity: Levenshtein (edit) distance permits insertions and deletions, which shift bit positions and break the algebraic structure that Hamming distance preserves. Two codeword pairs with the same Hamming distance can have different edit distances depending on where the differing bits are located — clustered differences can be bridged by fewer edit operations than scattered ones. Since the linear code’s generator matrix G has no control over the spatial distribution of differing bits, the nearestneighbour edit distances vary across codewords even though their Hamming-distance neighbours are all equidistant.

4) Choice of construction: We adopt linear codebook construction for its deterministic guarantee of uniform minimum distance in Hamming space. Every codeword enjoys the same nearest-neighbour distance, providing uniform error tolerance across all watermark identities. The RNG method, while producing comparable mean distances, cannot offer this guarantee — some identities may be unluckily close to a neighbour, reducing robustness for those watermarks.

![](images/5ba1da527f1894db6af4e3db024321eb9e59a26829bf536a4ed963e4af36039a.jpg)  
Fig. 4. Nearest-neighbour distance statistics for Linear and RNG codebook construction with codeword length $u = 1 2 7$ . Lines show the mean; shaded regions show the min–max range. The Linear method has no visible band under Hamming distance because min = mean = max at every n.

## APPENDIX F

## TOKENIZATION INCONSISTENCY CORRECTION

Subword tokenizers used by modern LLMs are not roundtrip stable: decoding a token sequence into text and reencoding that text can produce a token list of different length [14]. This phenomenon, known as tokenization inconsistency (TI), arises because the greedy merge rules applied during encoding may segment a string differently when surrounding context changes. Yan and Murawaki [14] report that while token-level inconsistency rates are typically below 0.3% for Llama 2 7B, the text-level rate—the probability that at least one token in a generated text is affected—reaches 33.6% for 200-token sequences.

1) Impact on our scheme: In our watermarking pipeline, the extractor receives text rather than the original token IDs and must retokenize it to recover the embedded codeword. Because each token encodes exactly one bit, a single token insertion or deletion during retokenization shifts all subsequent bit positions. For Hamming distance and BCH decoding, which rely on strict positional alignment, this causes complete desynchronization of the extracted codeword, typically resulting in extraction failure. Levenshtein distance is only marginally affected, as it natively handles insertions and deletions, but the extra edit operations still reduce the effective codeword separation.

![](images/5944a4e07cba16f89de7e09def9f3bb10e5ad600eafcdf05448699caadb7d637.jpg)  
Fig. 5. Same as Figure 4 but for codeword length $u = 2 0 0$

2) Stepwise consistency verification: To eliminate TI at its source, we augment the embedding procedure (Algorithm 2) with a per-token consistency check. After sampling each token $s ^ { ( t ) }$ , we verify that the generated sequence $( s ^ { ( 0 ) } , \ldots , s ^ { ( t ) } )$ survives a decode–encode round-trip with the same length. If the check fails, $s ^ { ( t ) }$ is excluded by setting its logit to −∞, and a new token is sampled from the remaining distribution. This process repeats until a consistent token is found or a retry limit is reached.

The key properties of this approach are:

• No additional model calls. The model’s logits are computed once per token position; retries only mask and resample from the same logit vector.

• Length-only check. Verifying that the round-trip preserves the sequence length is sufficient, because length preservation guarantees correct bit-position alignment during extraction.

• Negligible overhead. Each consistency check requires one tokenizer decode and one encode call. On Llama 2 7B the cumulative check time over a 200- token generation averages 139 ms (0.70 ms per token), roughly 1.2% of the total generation time.

![](images/e43b44d51e89f5ba31bc030bcc2a557b65d593cd075215325660c0c07d84607d.jpg)  
Fig. 6. Lexical diversity distributions for 200-token Llama 2 7B generations on the OpenGen dataset [10].

![](images/d351beb0a81c9a7b35276ae6873c61f34e6052d8beac591d7c9441a912312e8e.jpg)  
Fig. 7. Lexical diversity distributions for 200-token Mistral 7B Instruct generations on the SAR Prompts dataset [15].

Our method is similar in spirit to the TI mitigation methods proposed by Yan and Murawaki [14]. Since inconsistent tokens are infrequent resampling rarely occurs and the impact on text quality is negligible.

## APPENDIX G

## TEXT QUALITY EVALUATION

We conduct text quality evaluation comparing generated text with and without an embedded watermark. We assess quality along three dimensions: LLM-as-a-Judge pairwise comparison, perplexity, and lexical diversity via word-level distinctness and repetition metrics.

## A. LLM-as-a-Judge Pairwise Evaluation

For this evaluation, texts are generated with Mistral 7B Instruct. The non-watermarked baseline uses nucleus sampling (temperature = 0.6, top\_p = 0.9) with no logit bias, while the watermarked output uses greedy decoding (temperature = 0) with a logit bias of δ = 6. For each prompt, the judge receives two anonymised responses side-by-side: one watermarked and one non-watermarked. The positions are randomised per-pair to mitigate position bias. The judge model, Gemma 4 31B IT, is instructed to rate which response is better on the dimension of social authenticity—how believable the text is as a genuine humanwritten product review or social media post. The LLM judge decides a winner (or a tie), preceded by brief reasoning. Texts were generated using SAR Prompts dataset [15], which contains review-style and social-post prompts.

Table VI reports the results for two codeword lengths (u = 127 and u = 200). The two conditions receive very similar ratings, with a slight bias toward the watermarked text at both lengths. The margin is modest—roughly 9 percentage points at u = 127 and 10 percentage points at u = 200—and the tie rate is non-negligible, especially at u = 127.

A plausible explanation for this small difference is that the logit bias pushes the model away from its most probable tokens, introducing slightly more variation in phrasing that the judge occasionally reads as marginally less recognisably AI-generated.

Manual inspection of the judge’s reasoning supports this interpretation. In the majority of comparisons the judge describes the two responses as broadly equivalent, often labelling the winner only “slightly” better. Across both conditions, the most common criticism is that both texts read like “generic marketing copy” or “AI-generated product descriptions” rather than genuine human reviews. The watermarked text occasionally wins by adopting a more conversational or first-person tone, while the non-watermarked text is sometimes faulted for overly formal or “corporate” phrasing. Overall, the judge perceives the two conditions as close in quality, with the watermark producing a small but consistent edge in perceived authenticity.

Table VI. LLM-as-a-Judge pairwise comparison on social authenticity. Generation: Mistral 7B Instruct; judge: Gemma 4 31B IT. Position randomised per pair.
<table><tr><td>Length</td><td>Not WM</td><td>Watermarked</td><td>Tie</td></tr><tr><td>u = 127</td><td>39.7%</td><td>49.0%</td><td>11.3%</td></tr><tr><td>u = 200</td><td>43.4%</td><td>53.7%</td><td>2.9%</td></tr></table>

## B. Perplexity Evaluation

![](images/4b6d04df4a60e0d76e5411b049a3e73ff7a8354823e8247cf0fa4fadb5bc5b73.jpg)  
Fig. 8. Perplexity distribution for 200-token Llama 2 7B generations on the OpenGen dataset. Evaluator: Gemma 4 31B.

We report perplexity only because it is customary in watermarking papers, but we complement it with other quality metrics, which are more reliable and better suited. Prior work has shown that likelihood-oriented decoding can produce bland or repetitive text, and that decoding strategy alone can strongly affect text quality. More directly, [16] argue that perplexity from pretrained language models is unreliable for evaluating text quality, especially for short texts.

For this evaluation, texts are generated with Llama 2 7B. We evaluate perplexity on 200-token texts on the Open-Gen dataset [10]. Both conditions produced approximately

7,000 generations. Token-level perplexity is computed with a sliding-window approach using the Gemma 4 31B evaluator model. Figure 8 shows the distributions. Non-watermarked text achieves a median perplexity of 4.22 (mean 4.48), while watermarked text reaches 9.58 (mean 9.91)—roughly a two-fold increase. This increase is expected: the logit bias deliberately pushes the generator away from its most probable tokens, which raises perplexity. The absolute values remain moderate, and the relative increase is comparable to that reported by Qu et al. [6] for the same bias.

## C. Diversity Evaluation

We measure diversity with word-level lexical diversity metrics: Distinct-1 (unique unigrams over total unigrams), Distinct-2 (unique bigrams over total bigrams), and Trigram Repetition Rate (1 − unique/total trigrams). Higher Distinct scores indicate more varied vocabulary; lower Trigram Repetition Rate indicates less repetition. All metrics are computed with a lightweight CPU script that tokenises text at word boundaries and requires no model or GPU.

Figures 6 and 7 show the distributions for Llama 2 7B (≈7,000 texts per variant) and Mistral 7B Instruct (≈150 texts per variant), respectively. In both cases the watermarked text is consistently more diverse and less repetitive.

These results confirm that the logit bias does not merely raise perplexity—it actively increases lexical variety. By steering the model away from its most probable tokens, the watermark produces less repetitive, more idiosyncratic phrasing. This explains both the higher perplexity reported above and the slight authenticity edge observed in the LLM-as-a-Judge evaluation.

Randomly selected examples of watermarked generations, together with the prompts that produced them, are provided in Appendix H.

## APPENDIX H GENERATED TEXT EXAMPLES

Below are randomly selected examples of watermarked generations, together with the prompts that produced them. Generated text length is 200 tokens.

## A. SAR Prompts

Texts generated with Mistral 7B Instruct on the SAR PROMPTS dataset [15] (review-style and social-post prompts).

Prompt:

Compose a review of the Shimano GRX gravel bike groupset for mixed-terrain riding.

## Generation:

The Shimano GRX gravel bike groupset has been designed for those who enjoy mixed-terrain cycling. The GRX groupset features Shimano’s latest technology, including their new hydraulic disk brakes, which offer excellent power and accuracy. The groupset also includes Shimano’s new 48-11 cassette and cranks, which provide the perfect range of gear ratios for tackling a variety of terrains. The GRX groupset also includes Shimano’s Di2 shifting, which offers smooth shifting, even on the most technical terrains.

One of the standout features of the Shimano GRX groupset is its versatility. It is suitable for cycling both gravel paths and pavement roads, making it a great choice for those looking to explore a variety of terrain. Additionally, it is also suitable for cycling in a wide range of weather conditions

## Prompt:

Write a review of the Anker 737 Power Bank evaluating its 140W output for charging laptops on the go.

Generation:

The Anker 737 Power Bank is a portable charger that is great for charging laptops on the go. With its 140W output power, you are sure that your laptop will be charged to your liking. The power bank is small and convenient to carry, making it perfect for travelers, students and anyone who wants to keep their laptop charged on-thego. One of the great things about this power bank is its compatibility with a wide variety of laptops. It is designed to work with laptops that are compatible with USB-C chargers, which includes most of modern laptops today. Additionally, it also has two other ports that can be used to charge two devices at the same time, making it perfect for charging other devices such as your smartphone. Another thing I like about this power bank is its durability. The power bank is equipped with an advanced battery technology and is rated for 1,500 charge

## Prompt:

Write a review of the KitchenAid Artisan stand mixer for baking bread and making pasta attachments.

## Generation:

The KitchenAid Artisan stand mixer is a high-quality and reliable option for both baking bread and making pasta. The mixer’s sturdy and powerful design makes it easy to handle a variety of tasks, from kneading dough to whipping cream. The bowl-lift design also makes it easy to access the bowl and ingredients without the mixer being on, making it a safer option for those who prefer to work with the mixer unplugged.

One great aspect of using the Artisan for baking bread is that you can use any recipe that calls for a bread machine. The Artisan’s versatility means you can make everything from simple white bread to complex sourdough bread. The kneading capabilities of the mixer are also impressive. It’s easy to switch between dough and bread modes and adjust the time and speed to suit your needs.

In addition to baking bread, the KitchenAid also has many

Prompt:

Draft a review of the Bang and Olufsen Beoplay HX headphones for luxury design and soundstage.

## Generation:

The Bang and Olufsen Beoplay HX are an exquisite pair of headphones that combine luxury and soundstage in an unrivaled way. The headphone’s sleek and elegant design exude a feeling of opulence and sophistication. From the premium leather earpads and headband to the meticulously engineered aluminum frame, the build quality and attention to detail in the Beoplay HX is truly remarkable. The headphones come in several eye-catching colors, including black, gray, and gold. The gold version, in particular, is a true statement piece, with a stunning gold finish and a luxurious texture to the touch.

One of the standout features of the Beoplay HX is its expansive and immersive soundstage. The headphones are powered by a high-performance Bang & Olufsen proprietary driver, which delivers crystal-clear and detailed sound with a wide

## Prompt:

Create a review ofthe Apple AirPods Pro 2for commuting on noisy subway systems.

Generation:

The AirPods Pro 2 are a top-of-the-line option when it comes to noise-cancelling earphones for commuters. The ActiveNoise Cancellation is particularly useful for those looking to block out unwanted sounds on noisy subway systems.

The earbuds are comfortable and secure, making them a good choice for use during long subway rides or commutes. They stay in your ears well, even with all the noise and motion of a busy subway system.

One great feature of these headphones is that they offer a transparency mode that allows you to hear ambient sounds while still benefiting from some degree of active noise cancellation. This is particularly helpful when it is important to be aware of your surroundings, such as when crossing the street or boarding a train.

Another plus is that you can use the AirPods to take calls and listen to music, all while still enjoying the active noise

## B. OpenGen Dataset

Texts generated with Llama 2 7B on the OpenGen dataset [10].

## Prompt:

British army forces in North America were primarily tied up with the Siege of Boston in 1775. Seeking bases of operations where they had more control, the British planned an expedition to the southern colonies. Major General Henry Clinton, then in Boston, was to travel to Cape Fear, North Carolina, where he would join with largely Scottish Loyalists raised in the North Carolina backcountry, and a force of 2,000 men from Ireland under the command of Major General Charles Cornwallis. The plan was beset by difficulties from the start. The Irish expedition, originally supposed to depart at the beginning of December 1775, was delayed by logistical difficulties, and its 2,500 troops did not depart until February 13, 1776, escorted by 11 warships under the command of Admiral Sir Peter Parker. Clinton left Boston on January 20 with two companies of light infantry, and first stopped at New York City to confer with William Tryon, New York’s royal governor. Major General Charles Lee, sent by Major General George Washington to see to the defense of New York, coincidentally arrived there the same day as Clinton. New York was at that time extremely tense; Patriot forces were beginning to disarm and evict Loyalists, and the Britishfleet anchored there was having difficulty acquiring provisions. Despite this, Clinton made no secret that his final target was in the south. Lee observed that this was "certainly a droll way of proceeding; to communicate his full plan to the enemy is too novel to be credited." This was not even the first notice of the expedition to the colonists; a letter intercepted in December had already provided intelligence that the British were planning to go to the South. Clinton arrived at Cape Fear on March 12, expecting to find the European convoy already there. He met with the royal governors ofNorth and South Carolina, Josiah Martin and William Campbell, and learned that the recruited Scottish Loyalists had been defeated at Moore’s Creek Bridge two weeks earlier. Clinton also received pleas for assistance from the royal governor of Georgia, James Wright, who had been arrested, and then escaped to a navy ship. Parker’s fleet had an extremely difficult crossing. Battered by storms and high seas, the first ships of the fleet did not arrive at Cape Fear until April 18, and Cornwallis did not arrive until May 3.

## Generation:

Clinton, meanwhile, was forced to wait until he received word that Cornwallis was safely in port before departing Cape Fear on April 20, in an unsuccessful effort to capture the Patriot garrison in Charlestown (now called Charles Town) in Charleston, S. C. Clinton’s expedition to South Carolina was a complete success for British forces, but was a disaster in the long run. The Patriotic forces were unable to defend Charlestown, the British took over the town without a fight and the Patriotic government fled to the safety of the countryside. The Patriotic forces were able, in the end, to recapture the city in the Battle of Great Swamp, but the loss of control over South Carolina was a major setback for the colonist cause. In the short term, the British gained control over a significant portion of South and North Carolina, and they were able,

## Prompt:

Attending school became difficult for Radcliffe after the release of the first Harry Potter film. Some fellow pupils became hostile, though he says they were just trying to "have a crack at the kid that plays Harry Potter" rather than acting out ofjealousy. As his acting career began to consume his schedule, Radcliffe continued his education through on-set tutors. He admitted he was not a very good student, considering school useless and finding the work "really difficult". He achieved A grades in the three ASlevel exams that he took in 2006, but decided to take a break from education and did not go to college or university. Part of his reasoning was that he already knew he wanted to act and write, and that it would be difficult to have a normal college experience. "The paparazzi, they’d love it", he told Details magazine in 2007. "If there were any parties going on, they’d be tipped off as to where they were." Radcliffe first expressed a desire to act at age five, and, aged ten, he made his acting debut in BBC One’s televised two-part adaptation ofthe Charles Dickens novel David Copperfield (1999), portraying the title character as a young boy. Radcliffe made his film debut in The Tailor of Panama (2001), an American film based on John le Carré’s 1996 spy novel, and a moderate commercial success. In 2000, producer David Heyman asked Radcliffe to audition for the role of Harry Potter for the film adaptation of Harry Potter and the Philosopher’s Stone, the best-selling book by British author J. K. Rowling. Rowling had been searching for an unknown British actor to personify the character, and the film’s director Chris Columbus recalled thinking, "This is what I want. This is Harry Potter" after he saw a video of the young actor in David Copperfield. Eight months later, following several auditions, Radcliffe was selected to play the part.

## Generation:

The Harry Potter film franchise has been one of Hollywood’s most successful series of all time. It was the first time in history that three consecutive movies had earned more than US \$300 million in box office revenue. In addition to the Harry Potter series of films, Radcliffe starred as the title character of the 2006 film version of the BBC television serial adaptation, The Woman in White. He has starred as the young witch-hunter Jonah in The Sorcerer’s Apprentice with actor and comedic actor Jonah Hill and Nicolas Cage. In December 2006, Radcliffe made his Broadway debut as the titular character of Peter Shaffer’s one-man show Equus, for which he received positive notices from The New York Times critic, Charles Isherwood who called him a "fearless young actor". In 2006 he won

## Prompt:

On the first night Lead Belly and his wife Martha spent in the city, they encountered racially discriminatory Jim Crow laws similar to thosefound in their native Louisiana: most hotels refused to rent rooms to African Americans and the few that would were either full or refused to serve him because he was with a white man (Lomax). Lomax, in some versions of the story described as an unnamed "white friend", offered to let the couple stay for the night in his apartment near the Supreme Court Building. The next morning, Lead Belly awoke to Lomax arguing with his landlord about the presence of a black man, with the landlord threatening to call the police. While in Washington, Lead Belly encountered several other incidents of segregation that are believed to have contributed to the impetus of the song. For instance, when Lead Belly, Lomax, and their wives wanted to go out to dinner together, they discovered that it was impossible for the mixed race group to find a restaurant that would serve them. Lead Belly was told that if he returned later without Lomax, he would be served. In response to one of these incidents, a friend of Lead Belly’s, variously identified as either Lomax or Mary Elizabeth Barnicle joked that Washington was a "bourgeois town." Though Lead Belly did not know what the word "bourgeois" meant, he was fascinated by the sound of it, and after its meaning was explained to him he decided to incorporate it into a song about the trip. The song came together quickly; one account claims that it only took afew hoursfor Lead Belly to write it. Lomax liked it because it was partly based on what happened in his apartment. "The Bourgeois Blues" is a blues-style protest song that criticizes the culture of Washington, DC. It protests against both the city’s Jim Crow laws and the racism of its white population. Its structure includes several verses and a refrain that declares that the speaker is going to "spread the news all around" about the racial issues plaguing the city. The song, particularly in the refrain, conflates race and economics by referring to Caucasians as "bourgeois".

## Generation:

The lyrical structure of the refrain is based upon that of "The House of the Rising Sun" by English musician Eric B. King (1898–?). Lead Belly’s version is the only known instance of the song being performed with a refrain, and it has since been performed with the lyrics of "Birmingham Sunday", a song written by Hudie Ledbetter (Lead Belly) in the 1860s. In addition to its lyrical structure, "The Bourgeois Blues" is also musically similar to "The Rising of the Sun" (or the "Rising Sun" Blues) written in 1899. Both songs have a twelve bar form, a similar melancholy feel in the verses that contrast with the more optimistic refrains, a descending chromatic bass line, a descending chromatic bass pattern, and a similar instrumentation consisting only of

## Prompt:

After two straight second-place finishes at Charlotte and Riverside, Richmond started the Pocono event in third place inside the second row. That race saw a caution for rain with five laps left before the halfway point. NASCAR wanted the cars to get to the halfway point to make the race official, so the sanctioning body had the drivers slowly circle the track. It took the drivers 26 minutes to complete the laps, and the rain was so heavy that some drivers had to look out their side windows because they could not see out their windshields. Two hours later, the track had dried and the race resumed with Richmond in third. After Richmond’s car was adjusted to remove the "push", the car was more to his liking. Because his radio did not work, he was unable to communicate with his crew chief, Hyde, and he made his final pit stop with 37 laps left. Hyde worried that Richmond had stopped a lap too early to ensure that he would have enough fuel to make it to the end. After Richmond took the lead with 30 laps left in the race, Dale Earnhardt made up three seconds on Richmond’s five-second lead. With four laps to go, Buddy Arrington spun in a three-car accident. The remaining laps of the race where completed slowly under caution and Richmond took the checkered flag for the victory. He had led 97 laps, including the final 30, taking his first victory in a Rick Hendrick car. The tour returned to Pocono a month later, and Richmond battled for another victory in a fog-shortened event. In the final 8-lap sprint, Richmond competed in a three-car battle with Geoff Bodine and Ricky Rudd. Richmond crossed the finish line beside Rudd, winning the race by 0.05 seconds. He notched four more victories that season, and over a span of twelve races, Richmond earned three secondplace finishes, and six wins. The National Motorsports Press Association named him Co-Driver of the Year with Earnhardt after Richmond accumulated 13 top 5 finishes and 16 in the top 10. He had a career-best third-place finish in points after winning seven events in 1986, in what was his last full NASCAR season.

## Generation:

He was the first driver to win the Winston Millions, a 34-race series, in the 30-race schedule.

\## 1993 and later years

Richmond was named the Grand Marshal for the inaugural 1993 Daytona 500. In his final race as the driver for RCR, Richmond started the race in 12th, but had to pit with a broken axel on lap 35. His car returned on Lap 54, and he had to make up 30 positions to get to a position in which he was able to race for positions at the front. He had a strong run, and finished in the top 5, but was disallowed a top five finish after the crew chief made an unapproved pitstop. He was released from RCR after that race, and he was hired as a commentator for TNN. In

## Prompt:

As a result, it was slowed down. The gameplay originated with a tech demo by Naka, who developed an algorithm allowing a sprite to move smoothly on a curve by determining its position with a dot matrix. Naka’s prototype was a platform game with a fast-moving character rolling in a ball through a long, winding tube, and this concept was fleshed out with Ohshima’s character designs and levels by Yasuhara. Yasuhara originally intended to work on the game for three months due to the delay of his planned move to the United States by the outbreak of the Gulf War, but was engrossed in the project for nearly a year. His designs for levels were intended to attract both hardcore and casual gamers by integrating occasional challenging set pieces into the mostly accessible level design. The color scheme was influenced by the work of pop artist Eizin Suzuki, and the aesthetics of Green Hill were influenced by the geography ofCalifornia. In designing the gameplay, Naka was inspired by Mario creator Shigeru Miyamoto, whose games he had enjoyed playing years earlier. Admiring the simplicity of Miyamoto’s mechanics in complex environments, Naka decided that Sonic would be controlled with only a directional pad for movement and a single button for jumping. He also wanted his creation to be more action-oriented than the Mario series; while playing Super Mario Bros., he had wondered why the levels could not be cleared more quickly. Naka, Ohshima, and Yasuhara worked 19 hours a day on the project for several months. Due to the need to demonstrate the Genesis’ technological prowess, the game underwent extensive testing and redesign, which took over six months. According to Naka, the game had the fastestever character speed in a video game and a rotation effect in the special stages that had been considered impossible on the console. The team intended to add a two-player mode displayed via split-screen, but Naka’s programming knowledge was insufficient to implement it. A two-player mode appeared in Sonic the Hedgehog 2 (1992), whereby the second player controls Sonic’s sidekick Miles "Tails" Prower.

## Generation:

The soundtrack, which was composed and produced over a period of three weeks, was composed by Masato Nakamura, who was given a month to complete it after the game went gold. The music was inspired in part by rock musician Billy Corgan and the band he founded in the United States. Nakamura’s work was recorded in two weeks with a 200-man orchestra, a 56-man choir, and several solo musicians. Nakamura’s compositions were influenced in style by the music of video game soundtracks he had heard in the past; his favorite was Super Castlevania IV’s (1989). The sound effects were created in two weeks with the use of the SGI Indigo workstation, a computer with 16 megabytes of RAM, 160 megabytes of disk space, a CD-ROM, and a hard drive, which was unhe

## ACKNOWLEDGMENT

In accordance with IEEE policy, the authors disclose the use of AI assistants from the GPT (OpenAI) and Claude (Anthropic) model families for software implementation and editing prose. All content was reviewed and verified by the authors.

## REFERENCES

[1] J. Kirchenbauer, J. Geiping, Y. Wen, J. Katz, I. Miers, and T. Goldstein, “A watermark for large language models,” in Proceedings of the 40th International Conference on Machine Learning (ICML). PMLR, 2023, pp. 17 061–17 084.

[2] L. Wang, W. Yang, D. Chen, H. Zhou, Y. Lin, F. Meng, J. Zhou, and X. Sun, “Towards codable watermarking for injecting multi-bits information to LLMs,” in Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.

[3] J.-M. Sahut, M. Laroche, and E. Braune, “Antecedents and consequences of fake reviews in a marketing approach: An overview and synthesis,” Journal of Business Research, vol. 175, p. 114572, 2024.

[4] Capital One Shopping, “Fake review statistics (2025),” Online Report (accessed Aug 2025), March 2025. [Online]. Available: https://capitaloneshopping.com/research/fake-review-statistics/

[5] K. Yoo, W. Ahn, and N. Kwak, “Advancing beyond identification: Multibit watermark for large language models via position allocation,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL). Association for Computational Linguistics, 2024, pp. 4031–4055.

[6] W. Qu, W. Zheng, T. Tao, D. Yin, Y. Jiang, Z. Tian, W. Zou, J. Jia, and J. Zhang, “Provably robust multi-bit watermarking for AI-generated text,” in Proceedings of the 34th USENIX Security Symposium, 2025, arXiv:2401.16820.

[7] P. Fernandez, A. Chaffin, K. Tit, V. Chappelier, and T. Furon, “Three bricks to consolidate watermarks for large language models,” in Proceedings of the IEEE International Workshop on Information Forensics and Security (WIFS). Nürnberg, Germany: IEEE, Dec. 2023, pp. 1–6.

[8] Q. Lin, C. Tang, L. Zhang, J. Zhang, and X. Li, “DERMARK: A dynamic, efficient and robust multi-bit watermark for large language models,” 2025. [Online]. Available: https://arxiv.org/abs/2502.05213

[9] F. J. MacWilliams and N. J. A. Sloane, The Theory of Error-Correcting Codes. North-Holland, 1977.

[10] K. Krishna, Y. Song, M. Karpinska, J. Wieting, and M. Iyyer, “Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense,” 2023, dataset: https://huggingface.co/datasets/ jaroslawjanas/open-generation-data. [Online]. Available: https://arxiv. org/abs/2303.13408

[11] A. Cohen, A. Hoover, and G. Schoenbach, “Watermarking language models for many adaptive users,” in 2025 IEEE Symposium on Security and Privacy (SP). IEEE, 2025.

[12] C. Schuhmann, “essays-with-instructions: Essays paired with instructions (Hugging Face dataset),” Hugging Face, Dataset at ChristophSchuhmann/essays-with-instructions, 2023, accessed Aug 4, 2025. [Online]. Available: https://huggingface.co/datasets/ ChristophSchuhmann/essays-with-instructions

[13] B. Guo, X. Zhang, Z. Wang, M. Jiang, J. Nie, Y. Ding, J. Yue, and Y. Wu, “How close is ChatGPT to human experts? comparison corpus, evaluation, and detection,” arXiv preprint arxiv:2301.07597, 2023.

[14] R. Yan and Y. Murawaki, “Addressing tokenization inconsistency in steganography and watermarking based on large language models,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2025, pp. 7076–7098.

[15] J. Janas, “SAR Prompts,” 2026. [Online]. Available: https://huggingface. co/datasets/jaroslawjanas/sar-prompts

[16] Y. Wang, J. Deng, A. Sun, and X. Meng, “Perplexity from PLM is unreliable for evaluating text quality,” 2023. [Online]. Available: https://arxiv.org/abs/2210.05892