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