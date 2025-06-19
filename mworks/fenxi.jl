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