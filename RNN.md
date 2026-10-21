## 第一部分：文本预处理与概率基础

### 1. 语言模型与 N-gram

语言模型的目标是计算一个句子（词序列）的联合概率：

$$
P(x_1, x_2, \dots, x_T) = P(x_1) P(x_2 \mid x_1) P(x_3 \mid x_1, x_2) \dots = \prod_{t=1}^T P(x_t \mid x_1, \dots, x_{t-1})
$$

为了降低计算难度，引入**马尔可夫假设（Markov Assumption）**，当前词只依赖前 \(N-1\) 个词：

- **Bigram (N=2)**：\(P(x_t \mid x_1, \dots, x_{t-1}) \approx P(x_t \mid x_{t-1})\)
- **Trigram (N=3)**：\(P(x_t \mid x_1, \dots, x_{t-1}) \approx P(x_t \mid x_{t-1}, x_{t-2})\)

**参数爆炸与数据稀疏**：
如果词表大小是 \(V\)，N-gram 组合的理论参数数量为 \(V^N\)。
如果采用拉普拉斯平滑（Laplace Smoothing）解决零概率问题，公式为：

$$
P_{\text{Laplace}}(x_t \mid x_{t-1}) = \frac{\text{count}(x_{t-1}, x_t) + 1}{\text{count}(x_{t-1}) + V}
$$

*缺陷*：分母直接加上 \(V\) 会导致高频词概率被严重稀释，破坏了齐普夫定律的长尾分布。

### 2. 齐普夫定律与词表构建

**齐普夫定律（Zipf's Law）**：词频 \(n_i\) 与排名 \(i\) 成反比：

$$
n_i \propto \frac{1}{i^\alpha}
$$

两边取对数，在双对数坐标系下呈一条直线：

$$
\log n_i = -\alpha \log i + c
$$

*影响*：决定了词表必须截断，也催生了子词（BPE）算法。

---

## 第二部分：RNN 的核心机制

### 1. 两种自回归建模范式

- **自回归模型（AR）**：直接基于历史观测值预测未来。
- **隐变量自回归模型**：引入隐状态（记忆） \(h_t\) 压缩历史。RNN 属于此类。

### 2. RNN 的结构与公式

在时间步 \(t\)，RNN 接收当前输入 \(x_t\) 和上一时刻的隐藏状态 \(h_{t-1}\)，更新当前隐藏状态 \(h_t\)：

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

如果是语言模型，最终输出预测下一个词的概率分布：

$$
\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

- **\(W_{xh}\)**：输入到隐藏层的权重矩阵。
- **\(W_{hh}\)**：隐藏层到隐藏层的循环权重矩阵（**参数共享**）。
- **\(\tanh\)**：激活函数，将值压缩到 \([-1, 1]\) 区间。

### 3. RNN 的输入输出模式

- **多对一**：文本分类（最后一步输出）。
- **一对多**：图像描述生成。
- **多对多（等长）**：词性标注。
- **多对多（不等长）**：机器翻译（Seq2Seq）。

---

## 第三部分：RNN 的训练、评估与反向传播

### 1. 损失函数与困惑度（Perplexity, PPL）

语言模型通常使用交叉熵损失（Cross-Entropy Loss）。对于整个序列，总损失为：

$$
L = \sum_{t=1}^T L_t = -\sum_{t=1}^T \log P(x_t \mid x_1, \dots, x_{t-1})
$$

**困惑度（PPL）**是语言模型的标准评估指标：

$$
\text{PPL} = \exp\left(-\frac{1}{N} \sum_{t=1}^N \log P(x_t \mid x_{<t})\right)
$$

*PPL 越小，模型预测越准确。*

### 2. BPTT（随时间反向传播）

参数共享导致在计算梯度时，必须沿时间轴将所有时间步的梯度累加。以 \(W_{hh}\) 为例：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L_t}{\partial W_{hh}}
$$

利用链式法则展开 \(t\) 时刻的梯度：

$$
\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}
$$



$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^t \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^t W_{hh}^T \text{diag}(1 - \tanh^2(\dots))
$$

*如果连乘的值持续大于 1，导致**梯度爆炸**；持续小于 1，导致**梯度消失**。*

### 3. 梯度裁剪（Gradient Clipping）

**全局范数**计算：

$$
\text{norm} = \sqrt{\sum_{p \in \text{params}} \sum_{i} (\nabla_p^{(i)})^2}
$$

**裁剪操作**（当 \(\text{norm} > \theta\) 时）：

$$
\nabla_p \leftarrow \nabla_p \times \frac{\theta}{\text{norm}}
$$

### 4. 跨 Batch 的梯度截断（Detach）

保留数值，切断梯度图：

$$
h_{\text{init}} \leftarrow \text{detach}(h_{\text{last}})
$$
