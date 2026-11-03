# A Cross-Domain Matrix Engine for Signal Processing and Deep Neural Net Inference

Mehdi Safarpour

lastname.firstname[at]oulu.fi

Abstract—Matrix multiplication dominates both PHY signal processing and Deep Neural Network (DNN) inference, yet these domains place opposing demands on the matrix engine. Signal processing requires low-latency execution of relatively small matrix operations, whereas DNN inference, particularly at large batch sizes, prioritizes high throughput and energy efficiency for large matrix multiplications. While offering efficient memory access profile, a Systolic Array (SA) adds fill and drain delays per matrix. Those can be amortized for large matrices but not for small matrices. Conversely, a tensor unit completes the kernel in few cycles but would require repeated reads of same operand tiles over large matrices. This makes SA too slow for signal processing and tensor too inefficient for AI. We show that this need not be a choice between systolic and tensor structuring of Multiply–Accumulate (MAC) units. We propose a single engine that operates primarily as a SA and, through low-cost reconfiguration, transforms into a tensor-style engine on a per-operation basis. MUXes create both the spatial reduction adder tree and temporal reduction accumulators from the existing MACs and registers within the systolic array. The proposed reconfigurable SA simulated using modified SCALE-Sim simulator, verified on an Zynq 702 FPGA and synthesized using ASAP 7nm FinFET library. SCALE-Sim simulations shows up to 7.5× lower latency for small complex value PHY kernels. Even on DNN workloads, it achieves up to 1.65× speedups over vanilla SA execution with trivial increase of memory accesses, retaining the systolic array’s memory access advantage. Synthesis results estimate reconfigurable fabric costs only 8% power and 13.6% area overheads.

Index Terms—memory access, matrix engine, AI accelerator.

## I. INTRODUCTION

Systolic Arrays (SA) are one of the most popular structures for implementation of matrix engines, especially for AI accelerators [1]. In mobile devices, both physical-layer baseband processing and neural-network inference are dominated by matrix arithmetic. Most DNN computations, including convolutional, fully connected, and attention layers, are commonly lowered to GEMM [1]. Similarly, wireless operations such as MIMO detection, rely heavily on matrix operations. Hence, supporting both domains on a shared engine becomes increasingly attractive<sup>1</sup>.

The two domains, however, place rather different demands on matrix engines. Wireless algorithms often involve serially processed relatively small matrices, under tight realtime constraints, e.g., iterative MIMO decoding composes of successive matrix operations whose results depend on previous computations. DNN workloads, on the other hand, generally involve larger matrices, where high throughput and efficient

2D systolic array

![](images/5fdaa9fd524a15ca20e45d8f894e0c193ad97c54184d8bdfe80e235c714947f0.jpg)  
GEMM size (M = N = K)

![](images/f98eac385158b50090e1ee86ad27e03950c0a08b0a34d73674cc6b091eb88dcf.jpg)  
GEMM size (M = N = K)  
Fig. 1. Compute cycles and memory accesses of tensor and systolic engines for small and large GEMMs. Tensor is with full memory bandwidth.

operand reuse are essential, as a large energy portion is spent on moving operands between the memory and compute engines [1]. A 2D systolic array [1] addresses the latter requirement well through its large matrix tile and substantial data reuse. However, its fill-and-drain delays and low utilization become significant for small or irregular matrices, where these overheads cannot be amortized. Tensor units make the opposite trade-off. By organizing MAC units to perform reductions in parallel [3], they substantially reduce the latency of small matrix operations, but sacrifice operand reuse when processing large matrices. Repeated fetching of same tiles increases memory energy consumption [1], as memory access can be up to O(10)× more expensive than arithmetic [1].

As illustrated in Fig. 1, the tensor engine offers substantial latency advantages for small GEMMs, while its advantage diminishes for larger matrices despite retaining considerably higher memory traffic. Neither organization is therefore efficient across both workload domains.

This paper proposes a reconfigurable matrix engine, depicted in Fig. 2, in which the same MAC units can operate either as a systolic array or as a tensor-style engine, without changing the peripheral memory system or interfaces. Unlike conventional systolic reshaping, the proposed organization changes how the reduction dimension is realized, allowing peroperation selection between low-latency tensor execution and high-reuse systolic execution.

## II. BACKGROUND AND RELATED WORK

In this section, a brief background on targeted applications, the utilized matrix engines, and related works is presented.

## A. Application Domains

Wireless physical layer: Many physical-layer algorithms reduce to small dense matrix operations. For example, linear massive MIMO detection [4] forms the Gram matrix $( \mathbf G \ = \ \mathbf H ^ { H } \mathbf H )$ , matched-filter vector $( \mathbf { H } ^ { H } \mathbf { y } )$ , followed by inversion or an approximation such as a Neumann series. All relatively small GEMM operations that are serialized. Precoding and channel estimation similarly rely on small sequential GEMMs. Complex-valued kernels can be mapped to realvalued GEMMs through the $[ \mathrm { R e } , - \mathrm { I m } ; \mathrm { I m } , \mathrm { R e } ]$ embedding. Compared with DNN workloads, wireless domain matrices are typically smaller, but must often require much tighter per operation latency constraints.

![](images/f96dacebebc8be14ab8f89284d7352c3667fd8e77457c32adf8a306a1040203f.jpg)  
Fig. 2. The MAC units can be configured in tensor-style or and in 2D SA.

DNN inference: DNN inference is also dominated by matrix multiplication. Convolutional, fully connected layers and attention are commonly lowered to GEMM [1]. Matrix dimensions can vary substantially across layers and models. Batch size further changes the matrix shape, usually for benefit of higher throughput at the cost of higher latency. Thus, DNN workloads contain both large reuse-dominated GEMMs and smaller operations for which array utilization and latency become important.

## B. Matrix Engines

Tensor units: Tensor engines such as NVIDIA Tensor Cores and the Ascend Cube [3] execute small dense matrix multiplications with parallelism across the reduction dimension. Instead of propagating operands through a 2-D wavefront as in SAs, multiple products along K are generated concurrently and reduced through an adder network. This removes most of the systolic fill-and-drain latency, at the cost of smaller tile size (see Fig. 1). Total multiplications and adders count is equal to that of an SA. Adder performs spatial reduction, and temporal reduction based an an output-stationary (OS) dataflow.

Systolic Arrays: In this work we consider OS systolic array, while the results are applicable to weight stationary and input stationary flavors as well. In an $R \times C$ output-stationary systolic array, each processing element accumulates one output while operands propagate along the rows and columns. Each GEMM tile incurs a fill and drain overhead in addition to computation. This is amortized for large GEMMs, i.e., with large reduction, but becomes significant when the mapped dimensions are small relative to the array.

## C. Related works: Reconfigurable Systolic Arrays

Several architectures have addressed the utilization loss of fixed-size systolic arrays by reconfiguring their shape or dataflow. FlexSA [5] partitions a large systolic array into smaller parallel sub-arrays, while ReDas [6] enables finergrained reshaping and multiple systolic dataflows through additional PE routing. However, these approaches are less effective for small, sequentially dependent GEMMs, common in wireless algorithms, where limited parallelism leaves many MAC units underutilized. Our approach reconfigures the reduction dimension itself, enabling low-latency execution of individual small GEMMs, not just improve utilization.

![](images/e88ff49c2cdf72cebb2642295e33125e4a8b964cec43e690d27dbbb9c46cd82b.jpg)  
Fig. 3. The internal routing in PE of SA (a) requires just few reconfiguration fabric (as MUXes or transmission gates) for adder tree of a Tensor core (b).

## III. RECONFIGURABLE MATRIX ENGINE

The same MAC units of a systolic array (SA) can either form a large 2D systolic plane or be reorganized into smaller planes with parallelism along the reduction dimension K, forming a tensor-style so called 3D engine [3]. Unlike conventional systolic reshaping, the proposed reconfiguration changes how the reduction dimension is realized by forming a spatial reduction tree from existing SA resources.

In our configuration, a 64 × 64 SA is reconfigured into a $1 6 \times 1 6 \times 1 6$ tensor engine using the same 4096 MACs. The SA accepts 64 elements per operand per cycle, whereas a tensor operation requires two $1 6 \times 1 6$ operand tiles, supplied over four cycles through the shared interface. The SA’s skew buffer is reused for tensor operand buffering, while existing accumulators on SA form the spatial and temporal reduction tree. For complex-valued PHY operations, local complex-toreal expansion further reduces operand loading to two cycles.

Restricting tensor mode to the SA bandwidth requires minimal hardware modifications but limits MAC utilization compared to an unrestricted tensor engine. Nonetheless, even with this restriction, tensor mode substantially reduces the latency of small kernels, particularly complex-valued operations.

1) Organization: Figures 2,3 and 4 depict the two organizations using an 8×8 SA reconfigurable into a $4 \times 4 \times 4$ tensor engine for illustration simplicity. In the actual implementation, the $6 4 \times 6 4 ~ \mathrm { S A }$ is reorganized into sixteen $1 6 \times 1 6$ planes along the reduction dimension K. Each plane receives the corresponding column of A and row of B, generating products $a _ { i k } b _ { k j }$ that are summed through a spatial reduction tree.

As illustrated in Figs. 3 and 4, reconfiguration is achieved through three sets of routing fabrics, shown in Fig. 4 top right. Notice the depiction assumes a N × N SA is reconfigured to $K \times K \times K$ tensor. Notice the N and K here refer to physical array’s dimensions not matrix sizes. Our implementation restricts both modes to SA’s interface. Routing fabrics are mainly 2:1 multiplexers here. The first set controls operand routing, switching between systolic streaming and tensor-style broadcasting. The second changes the accumulation structure, i.e., in systolic mode, each PE accumulates partial results over time using its adder-register pair, whereas in tensor mode, the existing adders are interconnected to form a spatial reduction tree across the planes, see Fig 3. The third controls output routing, as systolic results leave the array row by row, while tensor mode produces a complete $1 6 \times 1 6$ output tile that is transferred through the shared output interface over multiple cycles. All routing configurations are controlled by a single mode bit distributed across the array.

![](images/e58c640923a4b75301eb74e96210365205da87034da94f9e9d6b5e641030ce11.jpg)  
Fig. 4. Top view of integration of systolic and tensor cores as reconfigurable engine (64 MAC example) with MUXes shows as reconfiguration fabric.

2) Timing, memory access and shared interface: Both modes share the same memory interface, with bandwidth determined by the systolic array. Consequently, tensor mode must load its operand tiles over multiple cycles rather than in a single cycle, limiting its performance compared to an unrestricted tensor engine. Nevertheless, by exploiting parallelism along the reduction dimension, tensor mode substantially reduces the execution latency of small GEMMs despite the shared bandwidth constraints. For an $M \times N \times K$ GEMM, cycles each engine take are:

$$
\begin{array} { r } { T _ { \mathrm { s y s } } = \left\lceil \frac { M } { 6 4 } \right\rceil \left\lceil \frac { N } { 6 4 } \right\rceil ( K + 1 2 6 ) , } \end{array}\tag{1}
$$

$$
\begin{array} { r } { T _ { \mathrm { t e n - l i m i t e d } } = \biggl \lceil \frac { M } { 1 6 } \biggr \rceil \biggl \lceil \frac { N } { 1 6 } \biggr \rceil \biggl ( 4 \biggl \lceil \frac { K } { 1 6 } \biggr \rceil + 3 + 4 \biggr ) . } \end{array}\tag{2}
$$

3) Operand traffic: Each engine reads valid operand elements only so edge padding is not fetched and invalid outputs are not written. The SA reads every A element once per column fold and every B element once per row fold and tensor mode with 16-element granularity:

$$
\begin{array} { r } { A _ { \mathrm { s y s } } = M K \Big \lceil \frac { N } { 6 4 } \Big \rceil + K N \Big \lceil \frac { M } { 6 4 } \Big \rceil + M N , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { A _ { \mathrm { t e n } } = M K \Big [ \frac { N } { 1 6 } \Big ] + K N \Big [ \frac { M } { 1 6 } \Big ] + M N . } \end{array}\tag{4}
$$

Since $\left\lceil x / 1 6 \right\rceil ~ \geq ~ \left\lceil x / 6 4 \right\rceil$ , tensor mode never moves fewer operands than systolic mode. Tensor approaches 4× when both dimensions are large. Equations (1) to (5) show why one structure cannot serve both application domains.

4) Mode selection: The compiler or scheduler selects the mode per operations known at compile time or at dispatch. The selection policy in this paper is to take the mode with fewer cycles and if the engine is within 10%, take whichever has fewer accesses by Eq. (3) and (4) and the assume one cycle per transition after pipeline completion.

![](images/a29c22d687bb44fde8cc72286ea98ed7c15a4cb976f221993567b57eaa439848.jpg)  
Fig. 5. Complex wireless PHY matrix operations. The lowest latency mode is selected for reconfigurable. SA does not perfom well for PHY.

Finally, for complex matrices, the restricted bandwidth of the SA in tensor mode can be mitigated, by generating the larger real matrices locally from the real and imaginary components of the complex inputs. This small modification is particularly useful for wireless PHY processing, which predominantly involves complex-valued matrices. This reduces number of cycles to load a new matrix to halve:

$$
\begin{array} { r } { T _ { \mathrm { t e n - l i m i t e d } } = \biggl \lceil \frac { M } { 1 6 } \biggr \rceil \biggl \lceil \frac { N } { 1 6 } \biggr \rceil \biggl ( 2 \biggl \lceil \frac { K } { 1 6 } \biggr \rceil + 3 + 2 \biggr ) . } \end{array}\tag{5}
$$

## IV. METHODOLOGY

To measure performance across different workloads, the SCALE-Sim simulator [7] was used. Workloads consists of common telecom jobs and various DNN models. Also, to estimate the routing overhead introduced by reconfiguration, a conventional tensor core, a systolic array, and the proposed reconfigurable array were described in HDL and synthesized and functionality verified.

All cycle and access counts are produced by SCALE-Sim. Systolic mode uses the stock output-stationary SA model with a 64×64 array, 64KB operand and output scratchpads, and the default bandwidth estimation memory model [7]. Tensor mode is added as a new compute engine such that it consumes matrices for a $1 6 \times 1 6 \times 1 6$ tensor with a four cycle pipeline, while the unmodified scratchpad, DRAM prefetch and stall machinery serves all investigated organizations [3], hence, memory interconnect bandwidth is assumed to be limited to that of SA. Valid SRAM operand reads and output writes are counted and register accesses are ignored. In systolic mode, edge tiles do not retain their full cycle cost and idle lanes do not issue null requests as it is common in practice. This emulates behavior of mentioned reconfigurable SAs, e.g., [5]. Analytical cycles may match with SCALE-Sim’s report when there are no memory stalls due to small size of problem.

The selected wireless and AI workloads represent different requirements on the matrix engine. Wireless kernels assume

![](images/d20e984f2b7a4389840ea9c8e772be51ad7cacd7a9c6073be16d3f9ca2b517fd.jpg)  
Fig. 6. AI GEMM workloads on different engines. Share of SA and tensor operation are noted as Systolic/Tenser.

64 antennas and 8 users unless otherwise stated. Various DNN workloads, lowered according to SCALE-Sim’s convention (M output pixels, N filters, and K receptive field), were selected to demonstrate that the solution also benefits conventional AI workloads while retaining its efficiency for large matrix operations.

## V. EVALUATION AND DISCUSSION

Performance and memory accesses were evaluated using SCALE-Sim. The design was first functionally verified on an FPGA with results of synthesis in a 7-nm process reported.

1) Wireless kernels: Figure 5 presents per-kernel results under the same memory-interface constraint. For latency-critical wireless PHY operations, tensor mode reduces latency by 2.15–7.52×. However, for larger matrices, systolic execution remains faster which of course is due to memory-interface restriction on tensor. The reconfigurable is the lowest of two modes. However, limited bandwidth tensor remain competitive to unlimited bandwidth tensor version for complex-valued operations, as local complex-to-real expansion reduces input and output transfers from four to two cycles each, further improving tensor-mode efficiency under limited bandwidth. The reported results cover only matrix multiplications.

2) Neural networks: Figure 6 presents topology-level results. Although the primary motivation of this work is to enable SAs to meet the latency demands of wireless kernels, the results interestingly show that per-operation reconfiguration also improves DNN performance over conventional SAs, with only a small increase in SRAM accesses for most workloads. The benefits are particularly evident in MobileNet, YOLOtiny, and low batch Fully Connected (FC). For large batch size SA turns out to benefit over tensor and hence reconfigurable array is kept configured to SA mode. For larger, convolutiondominated workloads, such as ResNet-50, the policy retains SA execution, preserving performance and memory efficiency.

3) Synthesis Results: The design first was verified functionally on a Zynq SoC FPGA then synthesized for 8-bit integer

<table><tr><td>B-tile operand regs</td><td>SA</td><td>Tensor</td><td>Reconf.</td></tr><tr><td>16-PE column (b)</td><td>Technology Supply voltage</td><td>ASAP7 7-nm FinFET</td><td></td></tr><tr><td>MACs</td><td></td><td>0.70 V 4096</td><td></td></tr><tr><td>Freq.</td><td></td><td>500 MHz</td><td></td></tr><tr><td>A-tile operand regs</td><td>Area (mm²)</td><td></td><td></td></tr><tr><td></td><td>0.308</td><td>0.167</td><td>0.350</td></tr><tr><td>Power (mW)</td><td>848</td><td>417</td><td>915</td></tr><tr><td>Overhead(Area)</td><td>8</td><td></td><td>13.6%</td></tr><tr><td>output mux (fabric 3)</td><td>Overhead(Power)⊗</td><td></td><td>8%</td></tr><tr><td>(a)</td><td>Tile size</td><td>16×16</td><td>Variable</td></tr><tr><td></td><td>Mem. BW*</td><td>768</td><td>192</td></tr></table>

(a) 64 × 4 grid of 16-PE (c) Synthesis reports. Elements per cycle. <sup>⊗</sup>Reconfigurable SA-mode versus base SA. columns. (b) one 16-PE △ With unlimited full memory BW. column (rotated).

Fig. 7. Synthesis results of the reconfigurable engine.

[8] using OpenROAD tool (see Fig. 7). Note, the numbers only include datapath and not memory access.

4) Discussion: The results in Fig. 5 and Fig. 6 reflect an example mode selection policy that can be tuned to favor performance or energy efficiency. For overheads, note all engines use INT8 arithmetic; higher-precision and multiple data type support substantially reduce the relative overheads. Assuming 5 pJ/byte access cost to a 64KB SRAM [1], SA consumes ≈ 2× less total energy than the tensor per ResNet-50 inference. Reconfigurable maintains this advantage while enabling low latency of the tensor.

## VI. CONCLUSION

The proposed reconfigurable engine combines the advantages of both organizations, retaining the SA’s memory efficiency for large AI workloads while providing low-latency tensor execution for small PHY operations. With modest hardware overhead, it enables systolic-array-based accelerators in mobile devices to support both AI inference and wireless baseband processing on a shared compute engine. It also benefits low-batch-size AI inference, extending the applicability to a wider range of edge AI applications.

## REFERENCES

[1] T. Krishna, H. Kwon, A. Parashar, M. Pellauer, and A. Samajdar, “Data orchestration in deep learning accelerators,” 2020.

[2] M. Safarpour and H. Dicander, “Reconfigurable processing elements for matrix multiplication,” Patent, 2026, filed Feb. 2026. (Pending patent).

[3] Y. Tang and C.-l. Wang, “Performance modeling on davinci ai core,” Journal of Parallel and Distributed Computing, vol. 175, pp. 134–149, 2023.

[4] J. Hoydis, F. A. Aoudia, A. Alkhateeb, S. Cammerer et al., “Toward a 6G AI-native air interface,” IEEE Commun. Mag., vol. 59, no. 5, pp. 76–81, 2021.

[5] S. Lym and M. Erez, “Flexsa: Flexible systolic array architecture for efficient pruned dnn model training,” arXiv preprint arXiv:2004.13027, 2020.

[6] M. Han, L. Wang, L. Xiao, T. Cai, Z. Wang, X. Xu, and C. Zhang, “Redas: A lightweight architecture for supporting fine-grained reshaping and multiple dataflows on systolic array,” IEEE Transactions on Computers, vol. 73, no. 8, pp. 1997–2011, 2024.

[7] R. Raj, S. Banerjee, N. Chandra, Z. Wan, J. Tong, A. Samajdhar, and T. Krishna, “Scale-sim v3: A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis,” in 2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 2025, pp. 186–200.

[8] L. T. Clark, V. Vashishtha, L. Shifren, A. Gujja, S. Sinha, B. Cline, C. Ramamurthy, and G. Yeric, “Asap7: A 7-nm finfet predictive process design kit,” Microelectronics Journal, vol. 53, pp. 105–115, 2016.