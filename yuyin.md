# 人体行为预测处理实验报告


## 使用MATLAB实现人体行为预测处理实验报告

### 一、实验内容

本次实验旨在基于智能手机传感器信号，构建一个能够自动识别6种不同人类活动的分类系统。实验内容主要包括：

1. 对原始加速度信号进行预处理，包括去除重力分量等；
2. 从预处理后的信号中提取时域、频域和自相关特征；
3. 使用多层感知机(MLP)神经网络对提取的特征进行训练；
4. 比较不同网络结构配置下的分类性能差异；
5. 通过混淆矩阵、准确率和召回率等指标对模型进行评估。

### 二、实验原理

#### 2.1 数据采集与预处理

实验使用的数据集来自UCI机器学习库，由30名受试者在进行6种不同活动（行走、上楼、下楼、坐、站、躺）时佩戴智能手机采集的加速度数据和角速度数据组成。

数据预处理阶段主要进行以下操作：

（1）使用高通滤波器分离身体运动引起的加速度和重力加速度；

（2）将信号分割成固定长度的窗口（缓冲区），每个窗口对应一个活动标签。

#### 2.2 特征提取

特征提取是本实验的关键环节，从每个信号窗口中提取了66个特征，分为以下几类：

（1）时域特征：包括各轴加速度的均值和RMS值；

（2）频域特征：功率谱密度(PSD)中的峰值位置和高度；特定频率波段内的总功率；

（3）自相关特征：信号自相关函数中的峰值位置和高度，用于检测信号的周期性。

#### 2.3 分类模型

使用模式识别神经网络（patternnet）作为分类器，这是一种前馈多层感知机，包含输入层、隐藏层和输出层。实验中在一个隐藏层的基础上进行扩展，测试了以下几种网络结构配置：

（1）单隐藏层结构：18个隐藏神经元；

（2）双隐藏层结构：18个和12个隐藏神经元；

（3）三隐藏层结构：18、12和8个隐藏神经元；

（4）四隐藏层结构：24、18、12和8个隐藏神经元。

#### 2.4 评估指标

（1）准确率：所有预测正确的样本占总样本的比例（TP/（TP+FP））；

（2）召回率：某类活动被正确预测的比例（TP/（TP+FN））；

（3）混淆矩阵：展示各类活动被正确分类和错误分类的情况。

以下结果展示中，混淆矩阵的最下方一行的红色字体分别表示各个行为预测的准确率；最右边一列的红色字体分别表示各个行为预测的召回率。

### 三、实验结果与分析

#### 3.1 单隐藏层（18）

网络结构：单隐藏层，18个隐藏神经元；

![1-单隐藏层网络结构图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E5%9B%BE.png "1-单隐藏层网络结构图")

其中，图中显示网络结构，时间等，以及数据划分是随机；训练使用缩放共轭梯度；性能利用交叉熵函数；计算MEX；训练了Epoch=142 iterations。查看结果图如下：

性能Performance，向最好的效果靠近：

![2-交叉熵曲线图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/2-%E4%BA%A4%E5%8F%89%E7%86%B5%E6%9B%B2%E7%BA%BF%E5%9B%BE.png "2-交叉熵曲线图")

训练状态Training State，观察梯度变化和验证集错误分布：

![3-梯度曲线和验证集错误分布图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/3-%E6%A2%AF%E5%BA%A6%E6%9B%B2%E7%BA%BF%E5%92%8C%E9%AA%8C%E8%AF%81%E9%9B%86%E9%94%99%E8%AF%AF%E5%88%86%E5%B8%83%E5%9B%BE.png "3-梯度曲线和验证集错误分布图")

误差直方图Error Histogram，数据集划分训练集，验证集，测试集比例0.7，0.15，0.15：

![4-误差直方图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/4-%E8%AF%AF%E5%B7%AE%E7%9B%B4%E6%96%B9%E5%9B%BE.png "4-误差直方图")

混淆矩阵Confusion，包括训练集、测试集、验证集以及结果的混淆矩阵：

![5-训练集混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/5-%E8%AE%AD%E7%BB%83%E9%9B%86%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "5-训练集混淆矩阵")

![6-验证集混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/6-%E9%AA%8C%E8%AF%81%E9%9B%86%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "6-验证集混淆矩阵")

![7-测试集混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/7-%E6%B5%8B%E8%AF%95%E9%9B%86%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "7-测试集混淆矩阵")

![8-估计混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/8-%E4%BC%B0%E8%AE%A1%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "8-估计混淆矩阵")

Receiver Operating Characteristic，都向1靠近:

![9-训练ROC](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/9-%E8%AE%AD%E7%BB%83ROC.png "9-训练ROC")

![10-验证ROC](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/10-%E9%AA%8C%E8%AF%81ROC.png "10-验证ROC")

![11-测试ROC](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/11-%E6%B5%8B%E8%AF%95ROC.png "11-测试ROC")

![12-估计ROC](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/12-%E4%BC%B0%E8%AE%A1ROC.png "12-估计ROC")

估计错误情况信号，实际是下楼，但是却估计为上楼，信号显示如下：

![13-估计错误情况信号图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/13-%E4%BC%B0%E8%AE%A1%E9%94%99%E8%AF%AF%E6%83%85%E5%86%B5%E4%BF%A1%E5%8F%B7%E5%9B%BE.png "13-估计错误情况信号图")

最终的混淆矩阵输出：

![14-最终混淆矩阵输出](https://github.com/Wansui233/zhineng/blob/main/matlab_images/1-%E5%8D%95%E9%9A%90%E8%97%8F%E5%B1%82/14-%E6%9C%80%E7%BB%88%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E8%BE%93%E5%87%BA.png "14-最终混淆矩阵输出")

可知，全局准确率为92%，其中行为4和5也就是上、下楼分类预测效果一般。

#### 3.2 双隐藏层（18,12）

网络结构：双隐藏层，18和12个隐藏神经元；

![1-双隐藏层网络结构图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/2-%E5%8F%8C%E9%9A%90%E8%97%8F%E5%B1%82/1-%E5%8F%8C%E9%9A%90%E8%97%8F%E5%B1%82%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E5%9B%BE.png "1-双隐藏层网络结构图")

可知，该网络结构，epoch，时间和性能等，以及混淆矩阵如下：

![2-MSE曲线图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/2-%E5%8F%8C%E9%9A%90%E8%97%8F%E5%B1%82/2-MSE%E6%9B%B2%E7%BA%BF%E5%9B%BE.png "2-MSE曲线图")

![3-训练和测试集混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/matlab_images/2-%E5%8F%8C%E9%9A%90%E8%97%8F%E5%B1%82/3-%E8%AE%AD%E7%BB%83%E9%9B%86%E5%92%8C%E6%B5%8B%E8%AF%95%E9%9B%86%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "3-训练和测试集混淆矩阵")

![4-验证集和估计混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/matlab_images/2-%E5%8F%8C%E9%9A%90%E8%97%8F%E5%B1%82/4-%E9%AA%8C%E8%AF%81%E9%9B%86%E5%92%8C%E4%BC%B0%E8%AE%A1%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "4-验证集和估计混淆矩阵")

![5-最终混淆矩阵输出](https://github.com/Wansui233/zhineng/blob/main/matlab_images/2-%E5%8F%8C%E9%9A%90%E8%97%8F%E5%B1%82/5-%E6%9C%80%E7%BB%88%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E8%BE%93%E5%87%BA.png "5-最终混淆矩阵输出")

由该混淆矩阵知，全局准确率为93.3%，相较于单隐藏层效果略微提高，但是行为4、5也就是坐、站，分类结果仍然一般。

#### 3.3 三隐藏层 (18,12,8)

网络结构：三隐藏层，18、12和8个隐藏神经元；

![1-三隐藏层网络结构图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/3-%E4%B8%89%E9%9A%90%E8%97%8F%E5%B1%82/1-%E4%B8%89%E9%9A%90%E8%97%8F%E5%B1%82%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E5%9B%BE.png "1-三隐藏层网络结构图")

![2-三隐藏层最终混淆矩阵输出](https://github.com/Wansui233/zhineng/blob/main/matlab_images/3-%E4%B8%89%E9%9A%90%E8%97%8F%E5%B1%82/2-%E4%B8%89%E9%9A%90%E8%97%8F%E5%B1%82%E6%9C%80%E7%BB%88%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E8%BE%93%E5%87%BA.png "2-三隐藏层最终混淆矩阵输出")

由该结果得，全局准确率为90.2%，相较前两种模型，结果降低。

#### 3.4 四隐藏层 (24,18,12,8)

网络结构：四隐藏层，24、18、12和8个隐藏神经元；

![1-四隐藏层网络结构图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/4-%E5%9B%9B%E9%9A%90%E8%97%8F%E5%B1%82/1-%E5%9B%9B%E9%9A%90%E8%97%8F%E5%B1%82%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E5%9B%BE.png "1-四隐藏层网络结构图")

![2-四隐藏层交叉熵变化曲线](https://github.com/Wansui233/zhineng/blob/main/matlab_images/4-%E5%9B%9B%E9%9A%90%E8%97%8F%E5%B1%82/2-%E5%9B%9B%E9%9A%90%E8%97%8F%E5%B1%82%E4%BA%A4%E5%8F%89%E7%86%B5%E5%8F%98%E5%8C%96%E6%9B%B2%E7%BA%BF.png "2-四隐藏层交叉熵变化曲线")

![3-四隐藏层最终混淆矩阵输出](https://github.com/Wansui233/zhineng/blob/main/matlab_images/4-%E5%9B%9B%E9%9A%90%E8%97%8F%E5%B1%82/3-%E5%9B%9B%E9%9A%90%E8%97%8F%E5%B1%82%E6%9C%80%E7%BB%88%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E8%BE%93%E5%87%BA.png "3-四隐藏层最终混淆矩阵输出")

由该结果知，全局准确率为90.7%，同样较低。
总之，适当地增加隐藏层数量和神经元数量可以提高模型的准确率，双隐藏层结构 (18,12) 取得了最高的准确率；更深的网络 (三隐藏层和四隐藏层) 并没有带来明显的性能提升。对于行为6，也就是躺，分类预测结果的准确率和召回率几乎都是100%；但是对于行为4、5，也就是坐、站准确率和召回率有待进一步提高。

#### 3.5 志愿者时域和频域结果

随机选择志愿者编号：19

![输出19](https://github.com/Wansui233/zhineng/blob/main/matlab_images/%E5%BF%97%E6%84%BF%E8%80%85%E6%97%B6%E5%9F%9F%2B%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C%E5%9B%BE/%E8%BE%93%E5%87%BA19.png "输出19")

该志愿者的六种行为的时域和频域结果如下：

![1-时域图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/%E5%BF%97%E6%84%BF%E8%80%85%E6%97%B6%E5%9F%9F%2B%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C%E5%9B%BE/1-%E6%97%B6%E5%9F%9F%E5%9B%BE.png "1-时域图")

![2-频域图](https://github.com/Wansui233/zhineng/blob/main/matlab_images/%E5%BF%97%E6%84%BF%E8%80%85%E6%97%B6%E5%9F%9F%2B%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C%E5%9B%BE/2-%E9%A2%91%E5%9F%9F%E5%9B%BE.png "2-频域图")

通过时域、频域结果，可知不同人类活动在时域和频域呈现显著差异，这些差异是机器学习模型提取特征、实现分类的核心依据。

**（1）时域分析**：对于时域，走、上下楼可通过周期性波动的有无、频率、幅度区分；坐、站、躺可通过波动强度、基线稳定性区分；时域波形的“周期性”和“平稳性”是识别活动类型的关键

**（2）频域分析**：对于频域，走、上下楼可通过主频率范围、高频能量占比区分（上楼>下楼>平地步行）；坐、站、躺可通过能量集中的频率区间、高频能量强度区分（躺<坐<站）；频域的“能量分布区间”是补充时域特征、区分相似活动（如上/下楼）的关键。

### 四、心得体会

本次实验的主要步骤包括数据预处理、特征提取、模型训练与测试以及结果分析。在特征提取阶段，从每个信号缓冲区中提取了多种时域和频域特征，如信号均值、均方根值、自相关特征、频谱峰值特征和频谱功率特征等，形成特征向量。然后使用神经网络对这些特征向量进行分类，以识别相应的活动类型。还通过修改神经网络的隐藏层结构来观察对分类性能的影响，并利用准确率、召回率、混淆矩阵等量化指标对不同网络结构下的预测结果进行评估比较。

适当增加隐藏层的深度使得网络能够学习到更加复杂的特征表示，从而提高了对不同活动的区分性能，在训练过程中观察到，具有两个隐藏层的网络在训练集上的分类准确率有所提高，同时在验证集上的表现也更为稳定，表明其泛化性能得到了提升。但是仍有部分样本被错误地归为其他活动，准确率一般。为获得更好的预测结果，之后可以尝试更换其他的深度学习模型，或者更好的技术。



## 使用Python实现人体行为预测处理实验报告

### 一、实验内容

人体行为预测在许多领域都有广泛的应用，如智能家居、健康监测、安防监控等。本实验旨在利用深度学习方法使用Python构建基于深度学习的人体行为预测模型，对UCI Human Activity Recognition（UCI HAR）数据集进行处理，实现对人体行为的准确预测。并随机从数据集中选取一名志愿者的六种行为数据，显示其时域和频域的结果。

### 二、实验环境

#### 2.1 实验环境

- Python
- Numpy
- Scikit-learn
- Seaborn
- Matplotlib
- Random

#### 2.2 数据集

本实验使用UCI HAR数据集，该数据集包含来自30个不同个体的加速度计和陀螺仪数据，这些个体执行了6种不同的活动（步行、上楼、下楼、坐、站、躺）。数据集分为训练集和测试集，分别包含7352和2947个样本。

### 三、实验方法

#### 3.1 模型选择

（1）VGG1D：基于 VGG 架构的一维卷积神经网络，用于提取时间序列数据的特征。以及加深的模型VGG1D_shen。

（2）ResNet1D_18：基于 ResNet 架构的一维卷积神经网络，通过残差连接解决了深度神经网络的梯度消失问题。以及加深的模型ResNet1D_50。

#### 3.2 模型训练

（1）损失函数：使用交叉熵损失函数，并绘制训练和测试集的损失曲线；

（2）优化器：使用Adam优化器，初始学习率为0.001；

（3）学习率调度器：使用余弦退火学习率调度器，最大训练轮数为100，最小学习率为1e-6；

（4）训练轮数：设置为100轮。

#### 3.3 模型评估

（1）准确率：计算模型在测试集上的准确率；

（2）分类报告：生成详细的分类报告，包括精确率、召回率、F1 值等指标；

（3）混淆矩阵：计算混淆矩阵，并绘制可视化的混淆矩阵；

### 四、实验结果

#### 4.1 模型VGG1D实验结果

分类报告如下图所示：准确率为94.94%：

![1-VGG1D分类报告结果](https://github.com/Wansui233/zhineng/blob/main/python_images/1-VGG1D%E5%88%86%E7%B1%BB%E6%8A%A5%E5%91%8A%E7%BB%93%E6%9E%9C.png "1-VGG1D分类报告结果")

训练和测试损失曲线如下：

![2-VGG1D训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/python_images/2-VGG1D%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "2-VGG1D训练和测试损失曲线")

混淆矩阵如下：

![3-VGG1D混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/python_images/3-VGG1D%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "3-VGG1D混淆矩阵")

#### 4.2 模型VGG1D_shen实验结果

分类报告如下图所示：准确率为95.86%：

![4-VGG1D_shen分类报告结果](https://github.com/Wansui233/zhineng/blob/main/python_images/4-VGG1D_shen%E5%88%86%E7%B1%BB%E6%8A%A5%E5%91%8A%E7%BB%93%E6%9E%9C.png "4-VGG1D_shen分类报告结果")

训练和测试损失曲线如下：

![5-VGG1D_shen训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/python_images/5-VGG1D_shen%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "5-VGG1D_shen训练和测试损失曲线")

混淆矩阵如下：

![6-VGG1D_shen混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/python_images/6-VGG1D_shen%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "6-VGG1D_shen混淆矩阵")

#### 4.3 模型 ResNet1D_18 实验结果

分类报告如下图所示：准确率为 95.42%：

![7-ResNet1D_18分类报告结果](https://github.com/Wansui233/zhineng/blob/main/python_images/7-ResNet1D_18%E5%88%86%E7%B1%BB%E6%8A%A5%E5%91%8A%E7%BB%93%E6%9E%9C.png "7-ResNet1D_18分类报告结果")

训练和测试损失曲线如下：

![8-ResNet1D_18训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/python_images/8-ResNet1D_18%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "8-ResNet1D_18训练和测试损失曲线")

混淆矩阵如下：

![9-ResNet1D_18混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/python_images/9-ResNet1D_18%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "9-ResNet1D_18混淆矩阵")

#### 4.4 模型 ResNet1D_50 实验结果

分类报告如下图所示：准确率为 94.33%：

![10-ResNet1D_50分类报告结果](https://github.com/Wansui233/zhineng/blob/main/python_images/10-ResNet1D_50%E5%88%86%E7%B1%BB%E6%8A%A5%E5%91%8A%E7%BB%93%E6%9E%9C.png "10-ResNet1D_50分类报告结果")

训练和测试损失曲线如下：

![11-ResNet1D_50训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/python_images/11-ResNet1D_50%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "11-ResNet1D_50训练和测试损失曲线")

混淆矩阵如下：

![12-ResNet1D_50混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/python_images/12-ResNet1D_50%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "12-ResNet1D_50混淆矩阵")

综合观察以上结果可知，获得较好的准确率，而且没有过拟合，考虑计算复杂度，可以选择模型 ResNet1D_18，如果要进一步提高准确度，可以使用更深层次或更复杂的模型架构，如 Transformer 或注意力机制模型，这些模型在处理序列数据时表现出色，能够捕捉长期依赖关系。调整超参数，加入数据增强技术，或训练更长的轮数等等技术。

#### 4.5 志愿者时域和频域结果

随机选择的志愿者编号: 9；

该志愿者的六种行为的时域和频域结果如下：

![13-行为1-时域频域结果](https://github.com/Wansui233/zhineng/blob/main/python_images/13-%E8%A1%8C%E4%B8%BA1-%E6%97%B6%E5%9F%9F%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "13-行为1-时域频域结果")

![14-行为2-时域频域结果](https://github.com/Wansui233/zhineng/blob/main/python_images/14-%E8%A1%8C%E4%B8%BA2-%E6%97%B6%E5%9F%9F%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "14-行为2-时域频域结果")

![15-行为3-时域频域结果](https://github.com/Wansui233/zhineng/blob/main/python_images/15-%E8%A1%8C%E4%B8%BA3-%E6%97%B6%E5%9F%9F%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "15-行为3-时域频域结果")

![16-行为4-时域频域结果](https://github.com/Wansui233/zhineng/blob/main/python_images/16-%E8%A1%8C%E4%B8%BA4-%E6%97%B6%E5%9F%9F%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "16-行为4-时域频域结果")

![17-行为5-时域频域结果](https://github.com/Wansui233/zhineng/blob/main/python_images/17-%E8%A1%8C%E4%B8%BA5-%E6%97%B6%E5%9F%9F%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "17-行为5-时域频域结果")

![18-行为6-时域频域结果](https://github.com/Wansui233/zhineng/blob/main/python_images/18-%E8%A1%8C%E4%B8%BA6-%E6%97%B6%E5%9F%9F%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "18-行为6-时域频域结果")

### 五、心得体会

使用 Python 实现人体行为预测，其中 VGG1D_shen 和 ResNet1D_18 模型在 UCI HAR 数据集上取得较好的性能，未来可以进一步探索更复杂的模型架构和数据增强方法，以提高人体行为预测的准确率。同时，可以考虑将模型应用到实际场景中，如智能家居、健康监测等。



## 使用MWORKS实现人体行为预测处理实验报告

### 一、实验内容

本实验旨在使用深度学习方法对UCI HAR(人类活动识别)数据集进行分类，识别六种不同的人类活动。实验实现了两种不同的模型架构：一维残差网络(ResNet1D)和多层感知机(MLP)，并对两种模型的性能进行了对比分析。

### 二、实验原理

#### 2.1 数据集介绍

UCI HAR数据集包含从30名受试者收集的智能手机传感器数据，受试者进行了六种不同的活动：步行(Walking)，步行上楼(Walking Upstairs)，步行下楼(Walking Downstairs)，坐着(Sitting)，站立(Standing) 和躺着(Lying)。数据通过智能手机的加速度计和陀螺仪采集，采样频率为50Hz，包含时域和频域特征。数据集分为训练集和测试集，其中训练集包含7352个样本，测试集包含2947个样本。每个样本有561个特征，这些特征是由原始传感器信号经过预处理（如滤波、分段等）以及特征提取（如均值、标准差等统计特征）得到的。

#### 2.2 数据预处理

从文本文件中读取数据，并验证数据与标签的样本数量一致性；接着使用均值和标准差对数据进行标准化处理，确保各特征具有相同的尺度；然后进行格式转换，对于ResNet1D模型，将数据重塑为(特征，通道，样本)的三维格式，适应CNN输入要求，对于MLP模型，将数据重塑为(特征，样本) 的二维格式，适应全连接网络输入要求；最后将标签范围从1-6调整为0-5，并转换为one-hot编码格式。

#### 2.3 模型介绍

（1）ResNet1D

ResNet1D是一维残差神经网络，首先通过一个初始的卷积层和批量归一化层对输入数据进行预处理，然后依次构建三个残差层，每个残差层包含两个基本残差块，用于提取不同层次的时间序列特征。最后通过全局平均池化层将特征映射降维，接着通过全连接层将特征映射映射到类别空间，实现对六种活动的分类。使用ADAM优化器对模型进行训练，以交叉熵作为损失函数，训练100个周期，批量大小为32，学习率为0.001。

（2）MLP

MLP是多层感知机，属于全连接神经网络，包含一个输入层、一个隐藏层和一个输出层。输入层的神经元数量等于特征数量（561），隐藏层有18个神经元，采用ReLU激活函数，输出层有6个神经元（对应六种活动类别）。同样使用ADAM优化器进行训练，训练100个周期，批量大小为32，学习率为0.001。在实验中还使用了不同结构的隐藏层。

#### 2.4 训练与评估

（1）损失函数：使用交叉熵损失函数，适合多分类任务；

（2）优化器：使用ADAM优化器，结合了动量和自适应学习率；

（3）评估指标：

- **准确率**：正确预测的样本比例；
- **混淆矩阵**：展示各类别的预测情况；
- **精确率**：正确预测为某类的样本占该类所有预测的比例；
- **召回率**：正确预测为某类的样本占该类所有实际样本的比例；
- **F1 分数**：精确率和召回率的调和平均。

### 三、实验结果及分析

#### 3.1 ResNet1D模型结果

使用ResNet1D模型，获得数据和训练过程，最终结果如下：

![1-ResNet1D训练开始](https://github.com/Wansui233/zhineng/blob/main/mworks_images/1-ResNet1D%E8%AE%AD%E7%BB%83%E5%BC%80%E5%A7%8B.png "1-ResNet1D训练开始")

![2-ResNet1D训练结束和评价结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/2-ResNet1D%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9D%9F%E5%92%8C%E8%AF%84%E4%BB%B7%E7%BB%93%E6%9E%9C.png "2-ResNet1D训练结束和评价结果")

可知：

- 训练 100 轮准确率约为 80.45%；
- 精确率约为 82.11%；
- 召回率约为 80.57%；
- F1 分数约为 80.05%；

训练和测试过程中损失曲线如下：

![3-ResNet1D训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/mworks_images/3-ResNet1D%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "3-ResNet1D训练和测试损失曲线")

混淆矩阵如下：

![4-ResNet1D混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/mworks_images/4-ResNet1D%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "4-ResNet1D混淆矩阵")

由上图知，训练损失逐渐下降，模型在不断学习，测试损失也呈现下降趋势，但下降速度通常慢于训练损失，测试准确率逐步提高，最终达到较高水平。对于混淆矩阵，对角线元素（正确分类）的值较大，非对角线元素（错误分类）的值较小，各类别之间存在混淆。为提高准确度，可能需要调整参数，比如增加训练轮数等。

#### 3.2 MLP模型结果

##### （1）单隐藏层 [18]

使用MLP模型，单隐藏层 [18]时，结果如下：

![5-单层MLP训练开始](https://github.com/Wansui233/zhineng/blob/main/mworks_images/5-%E5%8D%95%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E5%BC%80%E5%A7%8B.png "5-单层MLP训练开始")

![6-单层MLP训练结束和评价结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/6-%E5%8D%95%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9D%9F%E5%92%8C%E8%AF%84%E4%BB%B7%E7%BB%93%E6%9E%9C.png "6-单层MLP训练结束和评价结果")

可知：

- 训练 100 轮准确率约为 93.93%；
- 精确率约为 94.09%；
- 召回率约为 93.89%；
- F1 分数约为 93.96%；

训练和测试过程中损失曲线如下：

![7-单层MLP训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/mworks_images/7-%E5%8D%95%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "7-单层MLP训练和测试损失曲线")

混淆矩阵如下：

![8-单层MLP混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/mworks_images/8-%E5%8D%95%E5%B1%82MLP%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "8-单层MLP混淆矩阵")

可知测试过程有些过拟合。

##### （2）双隐藏层 [18, 12]

双隐藏层 [18, 12] 时，结果如下：

![9-双层MLP训练开始](https://github.com/Wansui233/zhineng/blob/main/mworks_images/9-%E5%8F%8C%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E5%BC%80%E5%A7%8B.png "9-双层MLP训练开始")

![10-双层MLP训练结束和评价结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/10-%E5%8F%8C%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9D%9F%E5%92%8C%E8%AF%84%E4%BB%B7%E7%BB%93%E6%9E%9C.png "10-双层MLP训练结束和评价结果")

可知：

- 训练 100 轮准确率约为 94.47%；
- 精确率约为 94.63%；
- 召回率约为 94.49%；
- F1 分数约为 94.53%；

训练和测试过程中损失曲线如下：

![11-双层MLP训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/mworks_images/11-%E5%8F%8C%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "11-双层MLP训练和测试损失曲线")

混淆矩阵如下：

![12-双层MLP混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/mworks_images/12-%E5%8F%8C%E5%B1%82MLP%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "12-双层MLP混淆矩阵")

可知测试过程有些过拟合。

##### （3）三隐藏层 [128, 64, 32]

三隐藏层 [128, 64, 32] 时，结果如下：

![13-三层MLP训练开始](https://github.com/Wansui233/zhineng/blob/main/mworks_images/13-%E4%B8%89%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E5%BC%80%E5%A7%8B.png "13-三层MLP训练开始")

![14-三层MLP训练结束和评价结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/14-%E4%B8%89%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9D%9F%E5%92%8C%E8%AF%84%E4%BB%B7%E7%BB%93%E6%9E%9C.png "14-三层MLP训练结束和评价结果")

可知：

- 训练 100 轮准确率约为 94.98%；
- 精确率约为 95.00%；
- 召回率约为 94.96%；
- F1 分数约为 94.96%；

训练和测试过程中损失曲线如下：

![15-三层MLP训练和测试损失曲线](https://github.com/Wansui233/zhineng/blob/main/mworks_images/15-%E4%B8%89%E5%B1%82MLP%E8%AE%AD%E7%BB%83%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF.png "15-三层MLP训练和测试损失曲线")

混淆矩阵如下：

![16-三层MLP混淆矩阵](https://github.com/Wansui233/zhineng/blob/main/mworks_images/16-%E4%B8%89%E5%B1%82MLP%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png "16-三层MLP混淆矩阵")

可知训练过程优秀，但是测试过程有些过拟合。

总体来看，MLP模型训练速度比ResNet1D快，训练损失下降趋势明显，训练损失逐渐下降，但是测试损失存在波动，最终结果比ResNet1D有较高的准确率。对于混淆矩阵，对角线元素（正确分类）的值较大，非对角线元素（错误分类）的值较小，各类别之间的混淆较少，但是容易过拟合。

#### 3.3 志愿者时域和频域结果

随机选择的志愿者编号：10

![17-志愿者编号输出](https://github.com/Wansui233/zhineng/blob/main/mworks_images/17-%E5%BF%97%E6%84%BF%E8%80%85%E7%BC%96%E5%8F%B7%E8%BE%93%E5%87%BA.png "17-志愿者编号输出")

该志愿者的六种行为的时域和频域结果如下：

![18-行为1-时域和频域结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/18-%E8%A1%8C%E4%B8%BA1-%E6%97%B6%E5%9F%9F%E5%92%8C%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "18-行为1-时域和频域结果")

![19-行为2-时域和频域结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/19-%E8%A1%8C%E4%B8%BA2-%E6%97%B6%E5%9F%9F%E5%92%8C%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "19-行为2-时域和频域结果")

![20-行为3-时域和频域结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/20-%E8%A1%8C%E4%B8%BA3-%E6%97%B6%E5%9F%9F%E5%92%8C%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "20-行为3-时域和频域结果")

![21-行为4-时域和频域结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/21-%E8%A1%8C%E4%B8%BA4-%E6%97%B6%E5%9F%9F%E5%92%8C%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "21-行为4-时域和频域结果")

![22-行为5-时域和频域结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/22-%E8%A1%8C%E4%B8%BA5-%E6%97%B6%E5%9F%9F%E5%92%8C%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "22-行为5-时域和频域结果")

![23-行为6-时域和频域结果](https://github.com/Wansui233/zhineng/blob/main/mworks_images/23-%E8%A1%8C%E4%B8%BA6-%E6%97%B6%E5%9F%9F%E5%92%8C%E9%A2%91%E5%9F%9F%E7%BB%93%E6%9E%9C.png "23-行为6-时域和频域结果")

### 四、心得体会

通过本次实验，深刻体会到了深度学习模型在处理复杂时间序列数据分类任务中的强大能力。ResNet1D模型的设计巧妙地解决了深层网络训练中的梯度消失问题，使其能够有效地学习到时间序列中的深层次特征，对于具有时间依赖关系的数据具有很好的适用性，但是训练速度慢。相比之下，MLP 模型结构简单、易于实现，训练速度快，但是容易过拟合。因此，我们在面对不同类型的数据和任务时，需要根据数据的特点选择合适的模型架构。

通过仔细分析混淆矩阵，我们可以发现模型在哪些类别上存在不足，进而有针对性地对模型进行改进，例如增加特定类别的样本数量、调整模型结构或优化训练策略等，以进一步提升模型的性能。

未来，可以通过尝试将MLP和CNN结合，或者更复杂的网络架构，如LSTM、GRU等循环神经网络，等各种先进技术优化模型，以及超参数调优等方法进行改进，以获得更好的结果。



## MATLAB实现代码

```matlab
net = patternnet(18);

%net = patternnet([18, 12]);

%net = patternnet([18, 12, 8]);

%net = patternnet([24, 18, 12, 8]);

以及从数据集中随机取一名志愿者的六种行为数据，显示其时域和频域的结果，plot_subject_acceleration代码如下：

![plot_subject_acceleration链接](https://github.com/Wansui233/zhineng/blob/main/matlab/plot_subject_acceleration "plot_subject_acceleration链接")

%% Load data for a random subject
% 获取所有志愿者的ID
subjectIDs = unique(subid);

% 输出所有志愿者编号
disp('All Subject IDs:');
disp(subjectIDs);

% 随机选择一名志愿者
randomSubjectID = datasample(subjectIDs, 1);

% 输出志愿者编号
disp(['Selected Subject ID: ', num2str(randomSubjectID)])

% Load data for all acceleration components
[accX, actidX, actlabels, t, fs] = getRawAcceleration('SubjectID', randomSubjectID, 'Component', 'x');
[accY, ~, ~, ~, ~] = getRawAcceleration('SubjectID', randomSubjectID, 'Component', 'y');
[accZ, ~, ~, ~, ~] = getRawAcceleration('SubjectID', randomSubjectID, 'Component', 'z');

%% Plot time-domain signals for each activity
activities = unique(actidX);

figure(1);
for i = 1:length(activities)
    activity = activities(i);
    activityName = actlabels{activity};
    
    % Select data for the current activity
    sel = actidX == activity;
    accX_act = accX(sel);
    accY_act = accY(sel);
    accZ_act = accZ(sel);
    t_act = t(sel);
    
    % Plot time-domain signals
    subplot(length(activities), 3, (i-1)*3 + 1);
    plot(t_act, accX_act);
    title(['Time Domain - ', activityName, ' (X-axis)']);
    xlabel('Time (s)');
    ylabel('Acceleration (m/s^2)');
    grid on;
    hold on;
    
    subplot(length(activities), 3, (i-1)*3 + 2);
    plot(t_act, accY_act);
    title(['Time Domain - ', activityName, ' (Y-axis)']);
    xlabel('Time (s)');
    ylabel('Acceleration (m/s^2)');
    grid on;
    hold on;
    
    subplot(length(activities), 3, (i-1)*3 + 3);
    plot(t_act, accZ_act);
    title(['Time Domain - ', activityName, ' (Z-axis)']);
    xlabel('Time (s)');
    ylabel('Acceleration (m/s^2)');
    grid on;
end

%% Plot frequency-domain signals for each activity
figure(2);
for i = 1:length(activities)
    activity = activities(i);
    activityName = actlabels{activity};
    
    % Select data for the current activity
    sel = actidX == activity;
    accX_act = accX(sel);
    accY_act = accY(sel);
    accZ_act = accZ(sel);
    
    % Compute and plot frequency-domain signals
    subplot(length(activities), 3, (i-1)*3 + 1);
    pwelch(accX_act, [], [], [], fs);
    title(['Frequency Domain - ', activityName, ' (X-axis)']);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    grid on;
    hold on;
    
    subplot(length(activities), 3, (i-1)*3 + 2);
    pwelch(accY_act, [], [], [], fs);
    title(['Frequency Domain - ', activityName, ' (Y-axis)']);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    grid on;
    hold on;
    
    subplot(length(activities), 3, (i-1)*3 + 3);
    pwelch(accZ_act, [], [], [], fs);
    title(['Frequency Domain - ', activityName, ' (Z-axis)']);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    grid on;
end

```



## Python实现代码

```python
utils.py对数据处理：

# ![utils.py链接](https://github.com/Wansui233/zhineng/blob/main/python/utils.py "utils.py链接")

import numpy as np

class UCIHARDataset:
    def __init__(self, data_path, label_path=None, transform=None):
        self.data = np.loadtxt(data_path)
        self.labels = np.loadtxt(label_path) if label_path else None
        self.transform = transform  # 添加对 transform 的支持

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将数据 reshape 成 1D 形式 (1, 561)
        img = self.data[idx].reshape(1, -1).astype(np.float32)
        # 手动归一化数据到 [-1, 1] 范围
        img = (img - np.mean(img)) / np.std(img)
        label = int(self.labels[idx]) - 1 if self.labels is not None else 0  # 将标签转换为从 0 开始

        # 应用变换（如果存在）
        if self.transform:
            img = self.transform(img)
            
        return img, label

```

```python
model.py建立模型：

# ![model.py链接](https://github.com/Wansui233/zhineng/blob/main/python/model.py "model.py链接")

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG1D(nn.Module):
    def __init__(self, num_classes):
        super(VGG1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 计算特征提取后的大小
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet1D_18(num_classes=6):
    return ResNet1D(BasicBlock1D, [2, 2, 2], num_classes=num_classes)

```

```python
model_co.py对模型扩展：

# ![model_co.py链接](https://github.com/Wansui233/zhineng/blob/main/python/model_co.py "model_co.py链接")

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG1D_shen(nn.Module):
    def __init__(self, num_classes):
        super(VGG1D_shen, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 计算特征提取后的大小
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D_shen(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6):
        super(ResNet1D_shen, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet1D_50(num_classes=6):
    return ResNet1D_shen(Bottleneck1D, [3, 4, 6, 3], num_classes=num_classes)

```

```python
train.py训练模型：

# ![train.py链接](https://github.com/Wansui233/zhineng/blob/main/python/train.py "train.py链接")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils import UCIHARDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from model import VGG1D
from model import ResNet1D_18

from model_co import VGG1D_shen
from model_co import ResNet1D_50
'''
# 定义数据增强变换
class TimeSeriesAugmentation:
    def __init__(self, noise_std=0.005, scale_factor=0.05, shift_range=5, mask_prob=0.1, mask_length=10):
        self.noise_std = noise_std
        self.scale_factor = scale_factor
        self.shift_range = shift_range
        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def __call__(self, x):
        # 添加噪声
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise

        # 随机缩放
        scale = 1.0 + torch.randn(1) * self.scale_factor
        x = x * scale

        # 时间平移
        shift = torch.randint(-self.shift_range, self.shift_range, (1,))
        x = torch.roll(x, shifts=shift.item(), dims=1)

        # 随机反转
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])

        # 时间掩蔽
        if torch.rand(1) < self.mask_prob:
            start = torch.randint(0, x.shape[1] - self.mask_length, (1,))
            x[:, start:start + self.mask_length] = 0.0

        return x
    
# 数据预处理和加载
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x)),  # 将 numpy 数组转换为 Tensor
    TimeSeriesAugmentation(noise_std=0.005, scale_factor=0.05, shift_range=5, mask_prob=0.1, mask_length=10),  # 数据增强
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x)),  # 将 numpy 数组转换为 Tensor
])
'''

train_dataset = UCIHARDataset(
    data_path='data/train/X_train.txt',
    label_path='data/train/y_train.txt',
    #transform=train_transform
)

test_dataset = UCIHARDataset(
    data_path='data/test/X_test.txt',
    label_path='data/test/y_test.txt',
    #transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 初始化模型、损失函数和优化器
#model = VGG1D(num_classes=6)
#model = ResNet1D_18(num_classes=6)
#model = VGG1D_shen(num_classes=6)
model = ResNet1D_50(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 使用余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)


# 训练模型
num_epochs = 100
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 记录训练和测试损失
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # 更新学习率
    scheduler.step()

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        epoch_test_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_test_loss += criterion(outputs, labels).item()
        epoch_test_loss /= len(test_loader)
        test_losses.append(epoch_test_loss)
        current_lr = scheduler.get_last_lr()[0]


        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        print(f'Test Loss: {epoch_test_loss:.4f}')
        print(f'Current Learning Rate: {current_lr:.6f}')


    # 保存模型
    torch.save(model.state_dict(), f'models_50/model_{epoch+1}.pth')

# 绘制最终的训练和测试损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.savefig('models_50/final_loss_curve.png')
plt.close()

```

```python
evaluate.py评估模块：

# ![evaluate.py链接](https://github.com/Wansui233/zhineng/blob/main/python/evaluate.py "evaluate.py链接")

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import VGG1D
from model import ResNet1D_18
from utils import UCIHARDataset
from torch.utils.data import DataLoader

from model_co import VGG1D_shen
from model_co import ResNet1D_50


test_dataset = UCIHARDataset(
    data_path='data/test/X_test.txt',
    label_path='data/test/y_test.txt'
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
model = ResNet1D_50(num_classes=6)
model.load_state_dict(torch.load('models_50/model_100.pth'))

# 测试模型
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    # 打印分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.set_style("darkgrid")  # 设置背景为灰色网格
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('models_50/final_Matrix.png')
    plt.show()

```

```python
fenxi.py志愿者的时域和频域结果显示：

# ![fenxi.py链接](https://github.com/Wansui233/zhineng/blob/main/python/fenxi.py "fenxi.py链接")

import numpy as np

from utils import UCIHARDataset

import matplotlib.pyplot as plt
import random

# 假设每个志愿者有固定数量的样本，这里需要根据实际数据集调整
# UCI HAR数据集共有30个志愿者，每个志愿者的数据分布在训练集和测试集中
# 这里简单假设数据集中每个志愿者的样本是连续排列的
# 你可能需要根据实际数据集的结构进行调整
NUM_VOLUNTEERS = 30

# 加载数据集
train_dataset = UCIHARDataset(
    data_path='data/train/X_train.txt',
    label_path='data/train/y_train.txt'
)
test_dataset = UCIHARDataset(
    data_path='data/test/X_test.txt',
    label_path='data/test/y_test.txt'
)

# 合并训练集和测试集
all_dataset = train_dataset.data.tolist() + test_dataset.data.tolist()
all_labels = train_dataset.labels.tolist() + test_dataset.labels.tolist()

# 随机选择一名志愿者
volunteer_id = random.randint(1, NUM_VOLUNTEERS)
print(f"随机选择的志愿者编号: {volunteer_id}")

# 假设每个志愿者的样本数量大致相同，这里简单计算每个志愿者的样本范围
samples_per_volunteer = len(all_dataset) // NUM_VOLUNTEERS
start_index = (volunteer_id - 1) * samples_per_volunteer
end_index = start_index + samples_per_volunteer

# 获取该志愿者的所有数据和标签
volunteer_data = all_dataset[start_index:end_index]
volunteer_labels = all_labels[start_index:end_index]

# 每种行为的编号从1到6
for activity_id in range(1, 7):
    # 找到该行为的所有样本
    activity_indices = [i for i, label in enumerate(volunteer_labels) if label == activity_id]
    if activity_indices:
        # 随机选择一个样本
        sample_index = random.choice(activity_indices)
        sample_data = volunteer_data[sample_index]

        # 时域可视化
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(sample_data)
        plt.title(f'Volunteer {volunteer_id}, Activity {activity_id} - Time Domain')
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')

        # 频域可视化
        fft_data = np.fft.fft(sample_data)
        frequencies = np.fft.fftfreq(len(sample_data))
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, np.abs(fft_data))
        plt.title(f'Volunteer {volunteer_id}, Activity {activity_id} - Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

        plt.tight_layout()
        # 保存图像
        plt.savefig(f'volunteer_{volunteer_id}_activity_{activity_id}.png')
        plt.close()

```



## MWORKS实现代码

```mworks
Unnamed.jl是基于Resnet_18实现人体行为预测，包括预处理，模型训练、评估等完整过程：

# ![Resnet_18实现人体行为预测链接](https://github.com/Wansui233/zhineng/blob/main/mworks/Unnamed.jl "Resnet_18实现人体行为预测链接")

using TyPlot
using BenchmarkTools
using DelimitedFiles
using Flux
using Flux: onehotbatch, onecold, crossentropy, softmax, params, relu
using MLJ
using MLJBase
using StatsBase
# 在代码开头添加以下行
using Pkg

using BSON: @save  # 显式导入 BSON 的 @save 宏

# 数据加载与预处理
struct UCIHARDataset
    data::Array{Float32, 3}
    labels::Array{Int, 1}
end

function UCIHARDataset(data_path::String, label_path::String)
    data = Float32.(readdlm(data_path))
    labels = Int.(readdlm(label_path))
    
    # 确保标签是一维向量
    if ndims(labels) > 1
        labels = vec(labels)
    end
    
    # 检查数据的样本数量与标签数量是否一致
    @assert size(data, 1) == length(labels) "数据和标签的样本数量不一致: $(size(data, 1)) vs $(length(labels))"
    
    # 标准化数据
    data = (data .- mean(data, dims = 1)) ./ std(data, dims = 1)
    
    # 重塑数据为CNN输入格式: (特征, 通道, 样本)
    data = reshape(data', size(data, 2), 1, size(data, 1))
    
    # 调整标签范围从1-6到0-5
    labels = labels .- 1
    
    return UCIHARDataset(data, labels)
end

# ResNet1D基础块定义
struct BasicBlock1D
    conv1::Conv
    bn1::BatchNorm
    conv2::Conv
    bn2::BatchNorm
    shortcut::Any
end

function BasicBlock1D(in_channels::Int, out_channels::Int, stride::Int = 1)
    conv1 = Conv((3,), in_channels => out_channels, stride = stride, pad = 1, bias = false)
    bn1 = BatchNorm(out_channels)
    conv2 = Conv((3,), out_channels => out_channels, stride = 1, pad = 1, bias = false)
    bn2 = BatchNorm(out_channels)
    shortcut = if stride != 1 || in_channels != out_channels
        Chain(Conv((1,), in_channels => out_channels, stride = stride, bias = false), BatchNorm(out_channels))
    else
        identity
    end
    return BasicBlock1D(conv1, bn1, conv2, bn2, shortcut)
end

function (block::BasicBlock1D)(x)
    out = relu.(block.bn1(block.conv1(x)))
    out = block.bn2(block.conv2(out))
    out += block.shortcut(x)
    out = relu.(out)
    return out
end

# ResNet1D模型定义
function ResNet1D(block, layers::Vector{Int}, num_classes::Int = 6)
    # 初始卷积层
    conv1 = Conv((3,), 1 => 64, stride = 1, pad = 1, bias = false)
    bn1 = BatchNorm(64)
    
    # 构建残差层
    in_channels = 64
    layer1, in_channels = make_layer(block, 64, layers[1], 1, in_channels)
    layer2, in_channels = make_layer(block, 128, layers[2], 2, in_channels)
    layer3, _ = make_layer(block, 256, layers[3], 2, in_channels)
    
    # 全局平均池化
    avg_pool = GlobalMeanPool()
    
    # 全连接层
    linear = Dense(256, num_classes)
    
    return Chain(
        conv1,
        bn1,
        relu,
        layer1,
        layer2,
        layer3,
        avg_pool,
        x -> reshape(x, :, size(x, 3)),
        linear
    )
end

function make_layer(block, planes::Int, blocks::Int, stride::Int, in_channels::Int)
    layers = []
    # 第一个残差块可能需要下采样
    push!(layers, block(in_channels, planes, stride))
    
    # 更新输入通道数
    in_channels = planes
    
    # 添加剩余的残差块
    for _ in 1:(blocks-1)
        push!(layers, block(in_channels, planes))
    end
    
    return Chain(layers...), planes
end

function ResNet1D_18(num_classes::Int = 6)
    return ResNet1D(BasicBlock1D, [2, 2, 2], num_classes)
end

# 训练函数
function train(model, train_dataset, test_dataset; num_epochs = 100, batch_size = 32, lr = 0.001)
    # 再次检查数据和标签的样本数量
    println("训练数据维度: ", size(train_dataset.data))
    println("训练标签数量: ", length(train_dataset.labels))
    println("测试数据维度: ", size(test_dataset.data))
    println("测试标签数量: ", length(test_dataset.labels))
    
    # 转换标签为one-hot编码
    train_labels_onehot = onehotbatch(train_dataset.labels, 0:5)
    test_labels_onehot = onehotbatch(test_dataset.labels, 0:5)
    
    # 检查one-hot编码后的标签维度
    println("训练one-hot标签维度: ", size(train_labels_onehot))
    println("测试one-hot标签维度: ", size(test_labels_onehot))
    
    # 创建数据加载器
    train_data = Flux.DataLoader((train_dataset.data, train_labels_onehot), batchsize = batch_size, shuffle = true)
    test_data = Flux.DataLoader((test_dataset.data, test_labels_onehot), batchsize = batch_size, shuffle = false)
    
    # 使用显式优化器API
    opt = ADAM(lr)
    ps = params(model)  # 获取可训练参数
    
    train_losses = Float64[]
    test_losses = Float64[]
    
    for epoch in 1:num_epochs
        epoch_train_loss = 0.0
        for (x, y) in train_data
            # 计算损失和梯度
            loss, grads = Flux.withgradient(ps) do
                ŷ = softmax(model(x))
                crossentropy(ŷ, y)
            end
            
            # 更新参数
            Flux.update!(opt, ps, grads)
            
            epoch_train_loss += loss
        end
        epoch_train_loss /= length(train_data)
        push!(train_losses, epoch_train_loss)

        # 测试模型
        epoch_test_loss = 0.0
        correct = 0
        total = 0
        for (x, y) in test_data
            ŷ = softmax(model(x))
            epoch_test_loss += crossentropy(ŷ, y)
            predicted = onecold(ŷ, 0:5)
            labels = onecold(y, 0:5)
            correct += sum(predicted .== labels)
            total += length(labels)
        end
        epoch_test_loss /= length(test_data)
        push!(test_losses, epoch_test_loss)
        accuracy = correct / total
        println("Epoch $epoch: Train Loss = $epoch_train_loss, Test Loss = $epoch_test_loss, Test Accuracy = $accuracy")

        # 保存模型
        try
            mkpath("models")
            @save "models/model_$epoch.bson" model  # 直接使用@save，因为已经导入
            println("模型保存成功: models/model_$epoch.bson")
        catch e
            println("保存模型失败: $e")
        end
    end
    return train_losses, test_losses
end

# 评估函数
function evaluate(model, test_dataset)
    test_data = Flux.DataLoader((test_dataset.data, onehotbatch(test_dataset.labels, 0:5)), batchsize = 32, shuffle = false)
    y_true = Int[]
    y_pred = Int[]
    for (x, y) in test_data
        ŷ = softmax(model(x))
        predicted = onecold(ŷ, 0:5)
        labels = onecold(y, 0:5)
        append!(y_true, labels)
        append!(y_pred, predicted)
    end
    
    # 转换为 MLJ 兼容的格式
    y_true_mlj = categorical(y_true)
    y_pred_mlj = categorical(y_pred)
    
    # 计算准确率
    accuracy = MLJBase.accuracy(y_pred_mlj, y_true_mlj)
    println("Test Accuracy: $accuracy")
    
    # 计算混淆矩阵
    cm = MLJBase.confusion_matrix(y_pred_mlj, y_true_mlj)
    println("Confusion Matrix:")
    println(cm)
    
    # 计算精确率、召回率和F1分数
    precision = MLJBase.Precision(y_pred_mlj, y_true_mlj)
    recall = MLJBase.recall(y_pred_mlj, y_true_mlj)
    f1 = MLJBase.f1score(y_pred_mlj, y_true_mlj)
    
    println("\nClassification Report:")
    println("Precision: $precision")
    println("Recall: $recall")
    println("F1 Score: $f1")
    
    # 绘制混淆矩阵热图
    plt = heatmap(cm, xlabel = "Predicted", ylabel = "Truth", title = "Confusion Matrix", color = :viridis)
    TyPlot.ty_savefig("models/final_Matrix.png")
    return plt
end

# 主程序
train_dataset = UCIHARDataset("C:/zhineng/pythonshi/pythonshi/data/train/X_train.txt", "C:/zhineng/pythonshi/pythonshi/data/train/y_train.txt")
test_dataset = UCIHARDataset("C:/zhineng/pythonshi/pythonshi/data/test/X_test.txt", "C:/zhineng/pythonshi/pythonshi/data/test/y_test.txt")

# 选择模型
# model = VGG1D(6)
model = ResNet1D_18(6)

# 训练模型
train_losses, test_losses = train(model, train_dataset, test_dataset)

# 绘制损失曲线
plot(1:length(train_losses), train_losses, label = "Train Loss", xlabel = "Epoch", ylabel = "Loss", title = "Train and Test Loss")
plot!(1:length(test_losses), test_losses, label = "Test Loss")
TyPlot.ty_savefig("models/final_loss_curve.png")

# 评估模型
evaluate(model, test_dataset)

```

```mworks
ganzhi.jl是基于MLP实现人体行为预测，包括预处理，模型训练、评估等完整过程：

# [ganzhi.jl链接](https://github.com/Wansui233/zhineng/blob/main/mworks/ganzhi.jl "ganzhi.jl链接")

using TySystemIdentification
using TyControlSystems
using TyImageProcessing
using PyPlot  # 使用PyPlot替代Plots

using BenchmarkTools
using DelimitedFiles
using Flux
using Flux: onehotbatch, onecold, crossentropy as flux_crossentropy, softmax, params as flux_params, relu
using MLJ
using MLJBase
using StatsBase
# 在代码开头添加以下行
using Pkg

using BSON: @save  # 显式导入BSON的@save宏

# 数据加载与预处理
struct UCIHARDataset
    data::Array{Float32, 2}  # 修改为2维数组，适合MLP
    labels::Array{Int, 1}
end

function UCIHARDataset(data_path::String, label_path::String)
    data = Float32.(readdlm(data_path))
    labels = Int.(readdlm(label_path))
    
    # 确保标签是一维向量
    if ndims(labels) > 1
        labels = vec(labels)
    end
    
    # 检查数据的样本数量与标签数量是否一致
    @assert size(data, 1) == length(labels) "数据和标签的样本数量不一致: $(size(data, 1)) vs $(length(labels))"
    
    # 标准化数据
    data = (data .- mean(data, dims = 1)) ./ std(data, dims = 1)
    
    # 重塑数据为MLP输入格式: (特征, 样本)
    data = reshape(data', size(data, 2), size(data, 1))
    
    # 调整标签范围从1-6到0-5
    labels = labels .- 1
    
    return UCIHARDataset(collect(data), labels)  # 确保data是正确的Array类型
end

# 多层感知机模型定义
function MLP(input_size::Int, hidden_sizes::Vector{Int}, num_classes::Int = 6)
    layers = []
    
    # 输入层到第一个隐藏层
    push!(layers, Dense(input_size, hidden_sizes[1], relu))
    
    # 添加隐藏层
    for i in 1:(length(hidden_sizes)-1)
        push!(layers, Dense(hidden_sizes[i], hidden_sizes[i+1], relu))
    end
    
    # 输出层
    push!(layers, Dense(hidden_sizes[end], num_classes))
    
    return Chain(layers...)
end

# 训练函数
function train(model, train_dataset, test_dataset; num_epochs = 100, batch_size = 32, lr = 0.001)
    # 再次检查数据和标签的样本数量
    println("训练数据维度: ", size(train_dataset.data))
    println("训练标签数量: ", length(train_dataset.labels))
    println("测试数据维度: ", size(test_dataset.data))
    println("测试标签数量: ", length(test_dataset.labels))
    
    # 转换标签为one-hot编码
    train_labels_onehot = onehotbatch(train_dataset.labels, 0:5)
    test_labels_onehot = onehotbatch(test_dataset.labels, 0:5)
    
    # 检查one-hot编码后的标签维度
    println("训练one-hot标签维度: ", size(train_labels_onehot))
    println("测试one-hot标签维度: ", size(test_labels_onehot))
    
    # 创建数据加载器
    train_data = Flux.DataLoader((train_dataset.data, train_labels_onehot), batchsize = batch_size, shuffle = true)
    test_data = Flux.DataLoader((test_dataset.data, test_labels_onehot), batchsize = batch_size, shuffle = false)
    
    # 使用显式优化器API
    opt = ADAM(lr)
    ps = flux_params(model)  # 使用Flux的params函数
    
    train_losses = Float64[]
    test_losses = Float64[]
    
    for epoch in 1:num_epochs
        epoch_train_loss = 0.0
        for (x, y) in train_data
            # 计算损失和梯度
            loss, grads = Flux.withgradient(ps) do
                ŷ = softmax(model(x))
                flux_crossentropy(ŷ, y)  # 使用Flux的crossentropy函数
            end
            
            # 更新参数
            Flux.update!(opt, ps, grads)
            
            epoch_train_loss += loss
        end
        epoch_train_loss /= length(train_data)
        push!(train_losses, epoch_train_loss)

        # 测试模型
        epoch_test_loss = 0.0
        correct = 0
        total = 0
        for (x, y) in test_data
            ŷ = softmax(model(x))
            epoch_test_loss += flux_crossentropy(ŷ, y)  # 使用Flux的crossentropy函数
            predicted = onecold(ŷ, 0:5)
            labels = onecold(y, 0:5)
            correct += sum(predicted .== labels)
            total += length(labels)
        end
        epoch_test_loss /= length(test_data)
        push!(test_losses, epoch_test_loss)
        accuracy = correct / total
        println("Epoch $epoch: Train Loss = $epoch_train_loss, Test Loss = $epoch_test_loss, Test Accuracy = $accuracy")

        # 保存模型
        try
            mkpath("models3mlp")
            @save "models3mlp/model_$epoch.bson" model  # 直接使用@save，因为已经导入
            println("模型保存成功: models3mlp/model_$epoch.bson")
        catch e
            println("保存模型失败: $e")
        end
    end
    return train_losses, test_losses
end

# 手动计算混淆矩阵的函数
function calculate_confusion_matrix(y_pred::Vector{Int}, y_true::Vector{Int}, num_classes::Int=6)
    cm = zeros(Int, num_classes, num_classes)
    for i in 1:length(y_true)
        cm[y_true[i]+1, y_pred[i]+1] += 1  # 标签从0开始，矩阵从1开始
    end
    return cm
end

# 评估函数
function evaluate(model, test_dataset)
    test_data = Flux.DataLoader((test_dataset.data, onehotbatch(test_dataset.labels, 0:5)), batchsize = 32, shuffle = false)
    y_true = Int[]
    y_pred = Int[]
    for (x, y) in test_data
        ŷ = softmax(model(x))
        predicted = onecold(ŷ, 0:5)
        labels = onecold(y, 0:5)
        append!(y_true, labels)
        append!(y_pred, predicted)
    end
    
    # 转换为 MLJ 兼容的格式
    y_true_mlj = categorical(y_true)
    y_pred_mlj = categorical(y_pred)
    
    # 计算准确率
    # 计算准确率 - 直接计算避免依赖
    accuracy = sum(y_pred .== y_true) / length(y_true)
    println("Test Accuracy: $accuracy")
    
    
    # 计算混淆矩阵 - 手动实现
    cm = calculate_confusion_matrix(y_pred, y_true)
    println("Confusion Matrix:")
    println(cm)
    
     # 计算精确率、召回率和F1分数 - 手动实现
     precision = zeros(Float64, 6)
     recall = zeros(Float64, 6)
     f1 = zeros(Float64, 6)
     
     for i in 1:6
         true_positives = cm[i, i]
         false_positives = sum(cm[:, i]) - true_positives
         false_negatives = sum(cm[i, :]) - true_positives
         
         precision[i] = true_positives > 0 ? true_positives / (true_positives + false_positives) : 0.0
         recall[i] = true_positives > 0 ? true_positives / (true_positives + false_negatives) : 0.0
         f1[i] = precision[i] + recall[i] > 0 ? 2 * precision[i] * recall[i] / (precision[i] + recall[i]) : 0.0
     end
     
     avg_precision = mean(precision)
     avg_recall = mean(recall)
     avg_f1 = mean(f1)
     
     println("\nClassification Report:")
     println("Average Precision: $avg_precision")
     println("Average Recall: $avg_recall")
     println("Average F1 Score: $avg_f1")
    
    
    # 绘制混淆矩阵热图
    fig = PyPlot.figure(figsize=(8, 6))  # 显式调用 PyPlot.figure
    PyPlot.imshow(cm, cmap="viridis")    # 显式调用 PyPlot.imshow
    PyPlot.colorbar(label="Count")
    PyPlot.xticks(0:5, 1:6)
    PyPlot.yticks(0:5, 1:6)
    PyPlot.xlabel("Predicted")
    PyPlot.ylabel("Truth")
    PyPlot.title("Confusion Matrix")

    # 添加数值标签
    for i in 1:6
        for j in 1:6
            PyPlot.text(j-1, i-1, cm[i, j], ha="center", va="center", color="white")
        end
    end

    PyPlot.savefig("models3mlp/final_Matrix.png")
    PyPlot.close(fig)

    # 绘制损失曲线
    fig = PyPlot.figure(figsize=(8, 6))
    PyPlot.plot(1:length(train_losses), train_losses, label="Train Loss", linewidth=2)
    PyPlot.plot(1:length(test_losses), test_losses, label="Test Loss", linewidth=2)
    PyPlot.xlabel("Epoch")
    PyPlot.ylabel("Loss")
    PyPlot.title("Train and Test Loss")
    PyPlot.legend()
    PyPlot.grid(true)

    PyPlot.savefig("models3mlp/final_loss_curve.png")
    PyPlot.close(fig)
    
    return nothing
end

# 主程序
train_dataset = UCIHARDataset("C:/zhineng/pythonshi/pythonshi/data/train/X_train.txt", "C:/zhineng/pythonshi/pythonshi/data/train/y_train.txt")
test_dataset = UCIHARDataset("C:/zhineng/pythonshi/pythonshi/data/test/X_test.txt", "C:/zhineng/pythonshi/pythonshi/data/test/y_test.txt")

# 获取输入特征大小
input_size = size(train_dataset.data, 1)

# 选择模型 - 定义一个3层MLP
model = MLP(input_size, [128, 64, 32], 6)
#model = MLP(input_size, [18,12,8], 6)
#model = MLP(input_size, [18], 6)
#model = MLP(input_size, [18,12], 6)

# 训练模型
train_losses, test_losses = train(model, train_dataset, test_dataset)

# 评估模型
evaluate(model, test_dataset)

```

```mworks
fenxi.jl志愿者的时域和频域结果显示：

# [fenxi.jl链接](https://github.com/Wansui233/zhineng/blob/main/mworks/fenxi.jl "fenxi.jl链接")

using FFTW
using PyPlot
using DelimitedFiles
using Random

# 假设每个志愿者有固定数量的样本，这里需要根据实际数据集调整
# UCI HAR数据集共有30个志愿者，每个志愿者的数据分布在训练集和测试集中
# 这里简单假设数据集中每个志愿者的样本是连续排列的
# 你可能需要根据实际数据集的结构进行调整
NUM_VOLUNTEERS = 30

# 加载数据集
function load_dataset(data_path, label_path)
    data = readdlm(data_path)
    labels = vec(readdlm(label_path))
    return data, labels
end

train_data, train_labels = load_dataset("C:/zhineng/pythonshi/pythonshi/data/train/X_train.txt", "C:/zhineng/pythonshi/pythonshi/data/train/y_train.txt")
test_data, test_labels = load_dataset("C:/zhineng/pythonshi/pythonshi/data/test/X_test.txt", "C:/zhineng/pythonshi/pythonshi/data/test/y_test.txt")

# 合并训练集和测试集
all_dataset = vcat(train_data, test_data)
all_labels = vcat(train_labels, test_labels)

# 随机选择一名志愿者
volunteer_id = rand(1:NUM_VOLUNTEERS)
println("随机选择的志愿者编号: $volunteer_id")

# 假设每个志愿者的样本数量大致相同，这里简单计算每个志愿者的样本范围
samples_per_volunteer = size(all_dataset, 1) ÷ NUM_VOLUNTEERS
start_index = (volunteer_id - 1) * samples_per_volunteer + 1
end_index = start_index + samples_per_volunteer - 1

# 获取该志愿者的所有数据和标签
volunteer_data = all_dataset[start_index:end_index, :]
volunteer_labels = all_labels[start_index:end_index]

# 每种行为的编号从1到6
for activity_id in 1:6
    # 找到该行为的所有样本
    activity_indices = findall(x -> x == activity_id, volunteer_labels)
    if !isempty(activity_indices)
        # 随机选择一个样本
        sample_index = rand(activity_indices)
        sample_data = volunteer_data[sample_index, :]

        # 创建一个新的图形
        fig, (ax1, ax2) = PyPlot.subplots(2, 1)

        # 时域可视化
        ax1.plot(sample_data)
        ax1.set_title("Volunteer $volunteer_id, Activity $activity_id - Time Domain")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Amplitude")

        # 频域可视化
        fft_data = fft(sample_data)
        frequencies = fftfreq(length(sample_data))
        ax2.plot(frequencies, abs.(fft_data))
        ax2.set_title("Volunteer $volunteer_id, Activity $activity_id - Frequency Domain")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")

        # 调整子图间距
        PyPlot.tight_layout()

        # 保存图像
        PyPlot.savefig("volunteer_$(volunteer_id)_activity_$(activity_id).png")

        # 关闭图窗口
        PyPlot.close(fig)
    end
end

```


