# LLM101n: Let's build a Storyteller

## 1.简介

在本课程中，将构建一个故事讲述者 AI 大型语言模型（LLM）。与 AI 一起创建、完善和插图小故事。 将从基础开始，逐步构建，从头到尾开发出类似 ChatGPT 的功能性 Web 应用。

本课程将使用 Python、C 和 CUDA，并且对计算机科学的前置知识要求很少。到课程结束时，你应该对 AI、LLM 和深度学习有相对深刻的理解。

数据集：
- 英文故事数据集：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- 中文故事数据集：[TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)

## 2.在线阅读

## 3.目录

**目录**

- Chapter 01 Bigram 语言模型（语言建模）
- Chapter 02 Micrograd（机器学习，反向传播）
- Chapter 03 N-gram模型（多层感知器，matmul，gelu）
- Chapter 04 注意力机制（注意力机制、softmax、位置编码器）
- Chapter 05 Transformer (Transformer，残差，layernorm, GPT-2)
- Chapter 06 词嵌入模型 (minBPE，字节对编码)
- Chapter 07 优化器 (initialization, optimization, AdamW)
- Chapter 08 Need for Speed I: 设备 (设备、CPU、GPU)
- Chapter 09 Need for Speed II: 精度 (mixed precision training, fp16, bf16, fp8, ...)
- Chapter 10 Need for Speed III: 分布式 (distributed optimization, DDP, ZeRO)
- Chapter 11 数据集(数据集、数据加载、合成数据生成)
- Chapter 12 模型推理一:kv-cache (kv-cache)
- Chapter 13 模型推理二:量化(量化)
- Chapter 14 微调一: SFT (supervised finetuning SFT, PEFT, LoRA, chat)
- Chapter 15 微调二: RL (reinforcement learning, RLHF, PPO, DPO)
- Chapter 16 部署 (API, Web 应用)
- Chapter 17 多模态 (VQVAE, diffusion transformer)


**Appendix 附录**

在上述进展中需要进一步研究的主题:

- 编程语言:汇编、C、Python  
- 数据类型:整数、浮点数、字符串(ASCII、Unicode、UTF-8)  
- 张量: shapes, views, strides, contiguous  
- 深度学习框架:PyTorch, JAX  
- 神经网络架构:GPT (1,2,3,4)， Llama (RoPE, RMSNorm, GQA)， MoE  
- 多模态:图像，音频，视频，VQVAE, VQGAN，扩散  
