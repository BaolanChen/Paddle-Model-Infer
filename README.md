# Paddle-Mono-InternVL
基于飞桨Paddle深度学习框架的推理对齐，Mono-InternVL模型、jina-clip-v2模型

------------------
## 模型列表

| 模型| 单位 | 时间 | 地址 |
| --- | --- | --- |  --- |
| Mono-InternVL | 上海人工智能实验室 | 2024 | https://huggingface.co/OpenGVLab/Mono-InternVL-2B |
| jina-clip-v2 | jina-ai | 2024 | https://huggingface.co/jinaai/jina-clip-v2 |


## Mono-InternVL


新的单体MLLM：它通过多模态专家混合架构无缝集成了一组视觉专家。这种架构能够有效地将预训练的大型语言模型（LLM）扩展到单体MLLM，同时保留预训练的知识。

内源性视觉预训练（EViP）：一种新颖的视觉预训练方法，一个三阶段的渐进学习过程，包括概念学习、语义学习和对齐学习，以逐步提升模型的视觉知识和多模态能力，鼓励Mono-InternVL的视觉专家从嘈杂的数据到高质量的数据不断掌握视觉知识。


## jina-clip-v2

多语言多模态的文本图像向量模型

多模态向量通过统一的数据表示，实现了不同模态数据的搜索和理解，是神经检索和多模态生成式 AI 应用的基石。今天，我们推出了全新的通用多语言多模态向量模型 —— jina-clip-v2。该模型基于 jina-clip-v1 和 jina-embeddings-3 构建，并实现了多项关键改进：

- 性能提升：v2 在文本-图像和文本-文本检索任务中，性能较 v1 提升了 3%。此外，与 v1 类似，v2 的文本编码器也能高效地应用于多语言长文本密集检索索，其性能可与我们目前最先进的模型 —— 参数量低于 1B 的最佳多语言向量模型 jina-embeddings-v3（基于 MTEB 排行榜）—— 相媲美。

- 多语言支持：以 jina-embeddings-v3 作为文本塔，jina-clip-v2 支持 89 种语言的多语言图像检索，并在该任务上的性能相比 nllb-clip-large-siglip 提升了 4%。

- 更高图像分辨率：v2 支持 512x512 像素的输入图像分辨率，相比 v1 的 224x224 有了大幅提升。能够更好地捕捉图像细节，提升特征提取的精度，并更准确地识别细粒度视觉元素。

- 可变维度输出：jina-clip-v2 引入了俄罗斯套娃表示学习（Matryoshka Representation Learning，MRL）技术，只需设置 dimensions 参数，即可获取指定维度的向量输出，且在减少存储成本的同时，保持强大的性能。
