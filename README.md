# Paddle-Model-Infer
基于飞桨Paddle深度学习框架的推理对齐，Mono-InternVL模型、jina-clip-v2模型


------------------
## 模型列表

| 模型| 单位 | 时间 | 地址 |
| --- | --- | --- |  --- |
| jina-clip-v2 | jina-ai | 2024 | https://huggingface.co/jinaai/jina-clip-v2 |



## jina-clip-v2

多语言多模态的文本图像向量模型

API快速上手使用： https://jina.ai/?sui=&model=jina-clip-v2

多模态向量通过统一的数据表示，实现了不同模态数据的搜索和理解，是神经检索和多模态生成式 AI 应用的基石。今天，我们推出了全新的通用多语言多模态向量模型 —— jina-clip-v2。该模型基于 jina-clip-v1 和 jina-embeddings-3 构建，并实现了多项关键改进：

- 性能提升：v2 在文本-图像和文本-文本检索任务中，性能较 v1 提升了 3%。此外，与 v1 类似，v2 的文本编码器也能高效地应用于多语言长文本密集检索索，其性能可与我们目前最先进的模型 —— 参数量低于 1B 的最佳多语言向量模型 jina-embeddings-v3（基于 MTEB 排行榜）—— 相媲美。

- 多语言支持：以 jina-embeddings-v3 作为文本塔，jina-clip-v2 支持 89 种语言的多语言图像检索，并在该任务上的性能相比 nllb-clip-large-siglip 提升了 4%。
- 
Jina CLIP v2 支持 89 种语言，在包括中文、英语、法语、德语、日语、俄语、阿拉伯语和西班牙语在内的主要语种中都表现优异。

在多语言图像检索基准测试中，8.65 亿参数的jina-clip-v2 的性能比目前最先进的 CLIP 模型 NLLB-CLIP-SigLIP 相当甚至更好。

Jina CLIP v2 的参数量介于 NLLB-CLIP-SigLIP 的两个版本之间：其 base 版本参数量为 5.07 亿，比 Jina CLIP v2 小 41%，large 版本参数量则高达 12 亿，比 Jina CLIP v2 大 39%。

------------------
## MiniCPM-V-2_6 PyTorch转Paddle


### 0. 模型概述

MiniCPM-V-2_6是一个多模态理解模型，主要由两个核心组件构成：
- 视觉编码器(Vision Encoder): 使用SigLip模型
- 语言模型(LLM): 使用Qwen2模型

模型权重下载地址：https://huggingface.co/openbmb/MiniCPM-V-2_6

### 1. 权重转换

通过分析模型组网代码和model.safetensors.index.json文件，我们可以清晰地识别出需要转换的权重层。转换过程中需要注意以下几点：

#### 1.1 PyTorch与Paddle的差异处理

主要差异在于线性层的权重矩阵需要进行转置操作。需要转置的层包括：

```python
need_transpose = {
    # 语言模型部分
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "lm_head.weight",
    
    # 视觉模型部分
    "self_attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    
    # 重采样层部分
    "resampler.attn.in_proj_weight",
    "resampler.attn.out_proj.weight",
    "resampler.attn.kv_proj.weight",
    "resampler.kv_proj.weight",
}
```

#### 1.2 模型命名调整
由于在PaddleNLP中Qwen2模型的命名规则不同，需要对模型权重的key进行重命名：
* 将 llm.model. 替换为 llm.qwen2.

#### 1.3 完整转换脚本
这个脚本实现了完整的safetensors转换（PyTorch → Paddle）功能：

1. 权重转换：
  * 处理线性层的权重转置
  * 调整模型命名
  * 保持其他层的权重不变
2. 配置文件调整：
  * 将torch_dtype替换为dtype
  * 移除transformers_version相关信息
3. 文件处理：
  * 自动处理分片权重文件
  * 复制并调整相关配置文件
完整的转换脚本见：



```python
import json
import os
import shutil
import copy
import paddle
import torch
from safetensors.torch import load_file
from safetensors.numpy import save_file
import logging as logger

model_path = "MiniCPM-V-2_6"
dst_path = model_path + "_pd"

# # 这里不修改，xxxxx代表随机名称，完全不会匹配到对应的key
src_prefix_key = "xxxxx."
dst_prefix_key = "xxxxx."

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

need_transpose = {
    # language_model
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "lm_head.weight",

    # vision_model
    "self_attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",

    # resampler
    "resampler.attn.in_proj_weight",
    "resampler.attn.out_proj.weight",
    "resampler.attn.kv_proj.weight",
    "resampler.kv_proj.weight",
}


def check_trans(key):
    for x in need_transpose:
        if x in key:
            return True

    return False


def translate_one_safetensors(file_name):
    tensors = load_file(os.path.join(model_path, file_name))
    for key in list(tensors.keys()):
        dst_key = key.replace(src_prefix_key, dst_prefix_key)
        dst_key = dst_key.replace("llm.model.", "llm.qwen2.") ###
        logger.info("{} {}".format(key, tensors[key].shape))
        shape_ = tensors[key].shape
        if check_trans(key) and len(shape_) == 2:
            t = tensors.pop(key).cuda().t().contiguous()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()
        else:
            t = tensors.pop(key).cuda()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()
        logger.info("{} {}".format(dst_key, tensors[dst_key].shape))

    save_file(tensors, os.path.join(dst_path, file_name), metadata={"format": "np"})


def execute_cmd(cmd, file_path):
    cmd = cmd + " " + file_path
    os.system(cmd)


if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
    index = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))

    dst_index = copy.deepcopy(index)
    for key in list(dst_index["weight_map"].keys()):
        dst_key = key.replace(src_prefix_key, dst_prefix_key)
        dst_index["weight_map"][dst_key] = dst_index["weight_map"].pop(key)

    files = set(index["weight_map"].values())
    logger.info(files)

    for file_name in sorted(os.listdir(model_path)):
        # skip hidden files
        if file_name.startswith("."):
            continue

        logger.info(file_name)
        if file_name in files:
            # convert safetensors to safetensors(paddle)
            translate_one_safetensors(file_name)
        else:
            # copy config.json and other files
            shutil.copy(os.path.join(model_path, file_name), os.path.join(dst_path, file_name))

    json.dump(dst_index, open(os.path.join(dst_path, "model.safetensors.index.json"), "w"), indent=2)

else:
    for file_name in sorted(os.listdir(model_path)):
        # skip hidden files
        if file_name.startswith("."):
            continue

        logger.info(file_name)
        if file_name == "model.safetensors":
            # convert safetensors to safetensors(paddle)
            translate_one_safetensors(file_name)
        else:
            # copy config.json and other files
            shutil.copy(os.path.join(model_path, file_name), os.path.join(dst_path, file_name))

execute_cmd(cmd="sed -i -e  's/torch_dtype/dtype/g' ",
            file_path=os.path.join(dst_path, "config.json"))

execute_cmd(cmd="sed -i /transformers_version/d ",
            file_path=os.path.join(dst_path, "config.json"))


logger.info(model_path)
logger.info(dst_path)
```

### 2. 代码转换注意事项

#### 2.1 转换工具使用
推荐使用官方转换工具：[PaConvert](https://github.com/PaddlePaddle/PaConvert)
注意，paddle不完全兼容一些函数，要手动改写（可借助大语言模型）。



#### 2.2 转换注意事项
代码组织优化
* 删除使用工具转换生成的utils处理包，保持代码结构简洁
* 按照paddlemix规范组织代码结构：
  * 模型代码：paddlemix/models/
  * 预测脚本：paddlemix/examples/
  * 图像预处理：paddlemix/processors/


#### 2.3 API差异处理

##### 2.3.1 基础操作转换：
```python
# PyTorch -> Paddle 常见替换
# view -> reshape
x = x.reshape(shape)

# permute -> transpose (注意：paddle需要明确指定所有维度)
x = x.transpose([0, 2, 1, 3])

# 设备相关代码可以移除，Paddle支持自动转换
# 删除 .to(self.device)
```

##### 2.3.2 特殊情况处理
位置编码处理
在LLaMA和Qwen2模型中需要手动添加position_ids：

```python
batch_size, seq_length = attention_mask.shape
position_ids = paddle.arange(seq_length).expand((batch_size, seq_length))
```
Flash Attention适配
* 使用PaddleNLP 3.0版本的实现
* flash_attention函数参数差异：

```python
# PyTorch
# flash_attn_varlen_func(..., softmax_scale=...)
# flash_attn_func(..., softmax_scale=...)

# Paddle
# flash_attn_varlen_func(..., scale=...)  # 替换softmax_scale参数
# flash_attn_func(...)  # 移除softmax_scale参数
```
* 注意获取输出时添加索引：output[0]

#### 2.4 模型组件替换

1. 优先使用现有实现：
   - 全局搜索检查PaddleNLP和paddlemix是否已有实现
   - 参考已合入模型（如qwen2vl）的写法
2. 注意Qwen2模型替换


3. MultiHeadAttention适配：
   - PyTorch和Paddle实现存在差异
   - 需要手动实现注意力机制
   - 注意参数初始化和加载的对应关系

### 3. 代码组织优化

建议按照paddlemix规范组织代码结构：

```
paddlemix/
├── models/          # 模型代码
├── examples/        # 预测脚本
└── processors/      # 图像预处理
```



- 更高图像分辨率：v2 支持 512x512 像素的输入图像分辨率，相比 v1 的 224x224 有了大幅提升。能够更好地捕捉图像细节，提升特征提取的精度，并更准确地识别细粒度视觉元素。

- 可变维度输出：jina-clip-v2 引入了俄罗斯套娃表示学习（Matryoshka Representation Learning，MRL）技术，只需设置 dimensions 参数，即可获取指定维度的向量输出，且在减少存储成本的同时，保持强大的性能。
