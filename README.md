# ACM MM 2025 Rebuttal

## 1、创新性

(VdQz): 1. The novelty is somewhat limited and leans more toward engineering optimizations. The proposed HOM and ESQ modules resemble practical heuristics rather than being grounded in deeper theoretical insights. 

(JA2J): Overall, this article is effective, but it lacks innovation and is a Diffusion version of the existing Hadamard quantification method. The author could compare with these Hadamard based quantization methods and clarify the differences from them: [1] Sun, Yuxuan, et al. "Flatquant: Flatness matters for llm quantization." arxiv preprint arxiv:2410.09426 (2024). [2] Ashkboos, Saleh, et al. "Quarot: Outlier-free 4-bit inference in rotated llms." Advances in Neural Information Processing Systems 37 (2024): 100213-100240.

(3VWG): Hadamard-based quantization methods are not novel, as some LLM quantization methods [1] and DiT quantization methods [2] have already been used. The difference between HOM and them is not obvious, it looks more like some hardware adaptation improvements.

(CV2U): The HOM proposed in the paper appears to be an application of Quarot[1] to the DiT models, with low overall innovation.

没有扎实的理论基础：引用之前大模型的论文，关于Hardmard变换的理论基础。

其他主要就是说，Hardmard变换大模型那边已经并不新鲜，HOM我的创新点主要在于，将hardmard变换与DiT模型的结构相适配，并加上自己观察：transformer中的多头本就是为了增加注意力的多样性，捕获更多的复杂关系，因此其分布差异比较大，通过头间的尺度缩放，进一步拉平了头之间的分布差异。

ESQ部分不仅符合softmax的输出分布，并且其指数特性使得它的计算更硬件友好。相比于log量化相关方法，我们的方法没有引入额外的非线性算子，并且更加灵活（和均匀量化无异）。

## 2、是否是针对扩散模型相关

(VdQz): 2. Both HOM and ESQ are closely tied to standard components of Multi-Head Attention and quantization techniques. They do not appear to be specifically designed with the unique characteristics of Diffusion Transformers in mind. (不针对扩散模型？在引入hardmard变换的过程中，针对DiT模型的结构进行了适配（入adanorm）)

(JA2J): Do outliers exist with all kinds of current diffusion models? The authors need to discuss the prevalence of the phenomenon to support the proposed method. (扩散模型是否都存在异常值问题，transformer-base的存在)

## 3、推理性能/时延

(JA2J): Due to the overhead of online reasoning, it is still unknown whether the author provides an accelerating operator for the online transformation. Therefore, the author could explain whether the proposed method can achieve acceleration. (在线计算部分效率问题)

(3VWG): The paper did not report any information about inference efficiency, which makes the effectiveness of online or offline transform, ESQ, and static quantization confusing.

(CV2U): ESQ appears to be a hardware optimization method for log quantizers, but there is no report on actual hardware acceleration.

(CV2U): The paper does not compare the hardware costs between online and offline transform, nor does it compare the efficiency with baseline methods.

(xEyP): Real-world hardware efficiency validation (e.g., speedups, latency).

补充量化模块和非量化模块的推理时延对比，补充oneline变换带来的额外推理时延，补充ESQ带来的推理加速分析。

## 4、动态量化对比

(3VWG): The paper did not provide a detailed explanation for why static quantization was chosen, and I did not see any conflict between the method proposed with dynamic quantization. Directly not comparing with dynamic quantization will lack many baseline methods.

(CV2U): The paper’s explanation of not comparing dynamic activation quantization is not enough. For widely used and proven effective methods [2][3][4], the article should provide sufficient explanations for actual latency and other reasons, or directly change the static quantization method to dynamic to conduct certain ablation experiments to verify its effectiveness in the dynamic framework.

(xEyP): Comparative discussion with dynamic quantization methods.

补充动态量化与静态量化的对比相关的引用，补充动态量化和静态量化模块，推理时延的对比，和上面的表格合并。

## n、其他

(VdQz): 3. The proposed methods show clear benefits in the 4-bit PTQ setting but offer minimal improvements at higher bit-widths. (低bit性能优秀但高bit提升有限，因为高bit本身性能损失微弱。)

(JA2J): Although W4A4 has achieved a milestone improvement, its current accuracy is still unacceptable. Could the authors analyze the challenges of W4A4 and further research work? (分析未来的方向)

(JA2J): In Table 1, why are Online(Ours) and Online+ESQ missing in some settings(8/8,4/8)? (8/8是因为offline性能提升足够，已经和全精度的模型相当，因此不需要引入额外的开销，4/8是因为缺少ESQ是因为ESQ使用于低bit，高bit由于刚性分辨率问题，反而效果不佳。)

(JA2J): The ablation experiment table lacks citations in Section 4.3.

(JA2J): Figure 4 can be visually presented using the same vertical coordinate. 

(3VWG): The difference between ESQ and common log quantizers is not obvious, and I think the paper does not detail explain the improvements and advantages of ESQ. (ESQ与其他log非均匀量化的差异与优化)

(3VWG): The comparison in experiments is not sufficient. The effectiveness of Hadamard matrix has been proven in [1][2], but are not compared. (实验部分，未与大模型的hadmard变换相关的进行对比，为什么没有对比：Quarot等方法有针对llm模型结构的适配，比如将RMSnorm的可学习参数的调整，使之变换可融入后续的权重中等，上述改动并不能直接搬用于DiT模型中，我们的研究针对DiT的结构进行了适配（adanorm），并且针对于qk注意力分数纯在的普遍分布差异的现象，提出进一步的head-wise scaling，在不引入额外计算的前提下，实现了注意力分数的有效平滑（这一段在创新点回复也可以说。），比如有点难补这个实验)

(CV2U): The color selection in the right figure of Fig. 7 can be improved appropriately, it is now difficult to understand.

(xEyP): Sensitivity analysis of hyper parameters (e.g., the impact of α). (这个审稿人评价其实都还可以，小分也给的挺高，不知道为啥最后只给4分。)

