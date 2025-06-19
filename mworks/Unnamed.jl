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