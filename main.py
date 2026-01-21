# 导入必要的库
import os  # 操作系统接口
import numpy as np  # 数值计算库
import matplotlib  # 绘图库
import warnings  # 警告处理
from cryptography.utils import CryptographyDeprecationWarning

# 忽略Paramiko的弃用警告
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning, module="paramiko")
import matplotlib.pyplot as plt  # 绘图接口
import wave  # WAV文件处理
import librosa  # 音频处理库
import pyaudio  # 实时音频处理
import time
from sklearn.model_selection import train_test_split  # 数据集划分工具
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU设备: {torch.cuda.current_device()}")

    # 设置GPU内存使用策略
    torch.backends.cudnn.benchmark = True
else:
    print("警告: 未检测到GPU，将使用CPU运行")
    print("请确保已安装CUDA和cuDNN，并且版本与PyTorch兼容")

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 数据预处理模块
def extract_mfcc(wav_path: str, n_mfcc: int = 40, max_frames: int = 100) -> np.ndarray:
    """
    提取MFCC特征并标准化
    :param wav_path: 音频文件路径
    :param n_mfcc: MFCC系数数量（默认40）
    :param max_frames: 最大帧数（默认100）
    :return: 标准化后的MFCC特征
    """
    # 加载音频文件并固定采样率为16kHz
    try:
        y, sr = librosa.load(wav_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file {wav_path}: {e}")
        return None

    # 计算MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 固定帧数
    if mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]  # 截断
    else:
        # 填充到最大帧数
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # 计算均值和标准差
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)

    # Z-score标准化
    mfcc = (mfcc - mean) / (std + 1e-8)

    # 转置以适应CNN输入 (PyTorch使用 [batch, channel, height, width] 格式)
    return mfcc.T[np.newaxis, np.newaxis, ...]


# 自定义数据集类
class SpeechDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def build_dataset(data_dir: str, classes: list = None, test_size: float = 0.2, val_size: float = 0.1):
    """
    构建数据集（支持自动扫描文件夹）
    :param data_dir: 数据根目录
    :param classes: 目标类别列表(默认自动获取所有类别)
    :param test_size: 测试集比例
    :param val_size: 验证集比例
    :return: 训练集/验证集/测试集的DataLoader
    """
    # 自动获取所有类别（排除_background_noise_）
    if classes is None:
        classes = [cls for cls in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, cls)) and cls != '_background_noise_']
        classes.sort()  # 排序确保顺序一致

    X, y = [], []  # 初始化特征和标签列表

    # 遍历每个类别目录
    for label, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)  # 获取类别目录路径

        # 遍历目录中的每个WAV文件
        for file in os.listdir(cls_path):
            if file.endswith('.wav'):
                # 提取MFCC特征并添加到数据集
                wav_path = os.path.join(cls_path, file)
                feat = extract_mfcc(wav_path)
                if feat is not None: # Check if MFCC extraction was successful
                    X.append(feat)
                    y.append(label)  # 添加对应标签

    # 合并所有特征和标签
    if X:  # Only concatenate if X is not empty
        X = np.concatenate(X, axis=0)
        y = np.array(y)
    else:
        print("错误：没有有效的数据被加载。请检查你的数据集。")
        return None, None, None, classes

    # 分层划分数据集：先分割训练+验证集与测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # 从训练+验证集中分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), stratify=y_train_val, random_state=42
    )

    # 确保标签为1D整数数组
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    # 创建PyTorch数据集
    train_dataset = SpeechDataset(X_train, y_train)
    val_dataset = SpeechDataset(X_val, y_val)
    test_dataset = SpeechDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0) # num_workers=4会导致windows报错
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, classes


# 定义CNN模型
class SpeechCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeechCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层的输入特征数量
        # 原始输入: [batch, 1, 100, 40]
        # 经过3次池化后: [batch, 128, 12, 5]
        self.fc1 = nn.Linear(128 * 12 * 5, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 模型训练模块
def train_model(train_loader, val_loader, num_classes, epochs=40):
    """
    训练CNN模型并保存最佳权重
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param num_classes: 类别数量
    :param epochs: 训练轮数
    :return: 训练好的模型和训练历史
    """
    # 确保模型目录存在
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    # 确保日志目录存在
    log_dir = "logs/train"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 创建模型并移至GPU（如果可用）
    model = SpeechCNN(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.001)

    # 用于保存最佳模型
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算训练集平均损失和准确率
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证集平均损失和准确率
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_pytorch.pth')
            print(f'模型已保存，验证准确率: {val_acc:.4f}')

    return model, history


# 可视化模块
def plot_training_history(history):
    """
    绘制训练过程曲线
    :param history: 训练历史记录
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_pytorch.png')
    plt.show()


def plot_loss_curve(history):
    """
    单独绘制损失曲线
    :param history: 训练历史记录
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], 'b-', linewidth=2, label='训练损失')
    plt.plot(history['val_loss'], 'r-', linewidth=2, label='验证损失')
    plt.title('模型训练损失曲线', fontsize=16)
    plt.xlabel('训练轮次', fontsize=14)
    plt.ylabel('损失值', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_pytorch.png')
    plt.show()


def plot_accuracy_curve(history):
    """
    单独绘制准确率曲线
    :param history: 训练历史记录
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], 'b-', linewidth=2, label='训练准确率')
    plt.plot(history['val_acc'], 'r-', linewidth=2, label='验证准确率')
    plt.title('模型训练准确率曲线', fontsize=16)
    plt.xlabel('训练轮次', fontsize=14)
    plt.ylabel('准确率', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_curve_pytorch.png')
    plt.show()


# 模型评估函数
def evaluate_model(model, test_loader, classes):
    """
    评估模型性能
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器
    :param classes: 类别列表
    """
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集预测和标签用于混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算测试集平均损失和准确率
    test_loss = test_loss / total
    test_acc = correct / total

    print(f'测试集性能:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_acc:.4f}')

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(15, 15))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix_pytorch.png')
    plt.show()

    return test_loss, test_acc


# 主函数
def main():
    """
    语音识别系统主函数
    """
    print("正在检查环境...")
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("未检测到GPU，将使用CPU运行")

    # 检查数据目录
    data_dir = "./data_dir/classes"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"警告: 数据目录 {data_dir} 为空或不存在")
        print("请先准备训练数据，或者使用以下命令创建示例数据:")
        print("1. 创建类别目录，例如: mkdir -p data_dir/classes/yes data_dir/classes/no")
        print("2. 添加WAV格式的音频文件到对应类别目录")
        print("\n运行测试模式以验证环境配置...")
        test_pytorch()
        return

    print("正在准备数据集...")
    train_loader, val_loader, test_loader, classes = build_dataset(data_dir)

    if train_loader is None or val_loader is None or test_loader is None:
        print("数据集构建失败，请检查数据目录。")
        return

    # 获取类别数量
    num_classes = len(classes)
    print(f"类别数量: {num_classes}")

    print("开始训练模型...")
    model, history = train_model(train_loader, val_loader, num_classes, epochs=20)

    print("正在评估模型...")
    test_loss, test_acc = evaluate_model(model, test_loader, classes)

    print("生成训练历史图表...")
    plot_training_history(history)

    # 额外生成单独的损失和准确率图表
    plot_loss_curve(history)
    plot_accuracy_curve(history)

    # 保存类别列表
    with open('classes_pytorch.txt', 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(cls + '\n')
    print("类别列表已保存为 classes_pytorch.txt")


# 添加测试函数
def test_pytorch():
    """
    测试PyTorch环境配置
    """
    print("\n===== PyTorch环境测试 =====")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}")
    print(f"Librosa版本: {librosa.__version__}")

    # 创建一个简单的模型
    print("\n创建测试模型...")
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    ).to(device)

    # 生成随机数据
    print("生成测试数据...")
    x = torch.randn(100, 5).to(device)
    y = torch.randint(0, 2, (100, 1)).float().to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    print("测试模型训练...")
    model.train()
    for _ in range(1):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print("\n===== 测试完成 =====")
    print("环境配置正常，PyTorch可以正常使用")
    print("请准备训练数据后再次运行程序进行实际训练")


def main2():
    """
    实时语音识别功能
    通过麦克风输入实时识别语音命令
    """
    import time
    start_time = time.time()

    # 加载训练好的模型
    print("开始加载模型...")
    model_load_start = time.time()

    # 加载类别列表
    if os.path.exists('classes_pytorch.txt'):
        with open('classes_pytorch.txt', 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f]
        print(f"加载类别列表: {classes}")
    else:
        # 从数据目录自动加载类别（排除_background_noise_）
        data_classes_dir = "./data_dir/classes"
        classes = [cls for cls in os.listdir(data_classes_dir)
                   if os.path.isdir(os.path.join(data_classes_dir, cls)) and cls != '_background_noise_']
        classes.sort()
        print(f"警告: 未找到 classes_pytorch.txt 文件, 从数据目录加载类别列表: {classes}")

        # 保存类别列表供下次使用
        with open('classes_pytorch.txt', 'w', encoding='utf-8') as f:
            for cls in classes:
                f.write(cls + '\n')
        print("类别列表已保存为 classes_pytorch.txt")

    # 创建模型实例
    num_classes = len(classes)
    model = SpeechCNN(num_classes).to(device)

    # 加载模型权重
    if os.path.exists('best_model_pytorch.pth'):
        model.load_state_dict(torch.load('best_model_pytorch.pth'))
        model.eval()  # 设置为评估模式
        model_load_time = time.time() - model_load_start
        print(f"模型加载完成，耗时: {model_load_time:.2f}秒")
    else:
        print("错误: 未找到模型文件 'best_model_pytorch.pth'")
        print("请先运行训练过程生成模型文件")
        return

    # 添加索引越界保护
    def safe_get_class(index):
        if 0 <= index < len(classes):
            return classes[index]
        return f"未知类别({index})"

    # 音频参数设置
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 2  # 增加每次录音时长
    SILENCE_THRESHOLD = 50  # 进一步降低静音阈值，提高灵敏度

    # 初始化PyAudio
    print("初始化PyAudio...")
    pyaudio_init_start = time.time()
    p = pyaudio.PyAudio()
    pyaudio_init_time = time.time() - pyaudio_init_start
    print(f"PyAudio初始化完成，耗时: {pyaudio_init_time:.2f}秒")

    # 打开音频流
    stream_start = time.time()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"音频流打开失败: {e}")
        p.terminate()
        return
    stream_open_time = time.time() - stream_start
    print(f"音频流打开完成，耗时: {stream_open_time:.2f}秒")

    print(f"总初始化时间: {time.time() - start_time:.2f}秒")

    print("开始实时语音识别，按Ctrl+C停止...")

    try:
        while True:
            print("请说话...")
            frames = []

            # 检测是否有声音输入（添加超时机制）
            max_silent_seconds = 5  # 最大静音等待时间（秒）
            max_silent_frames = int(RATE / CHUNK * max_silent_seconds)
            silent_frame_count = 0
            detected_sound = False

            while silent_frame_count < max_silent_frames:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    # 检测是否有声音（使用峰值检测更灵敏）
                    current_volume = np.abs(audio_data).max()  # 使用峰值而不是平均值
                    if current_volume > SILENCE_THRESHOLD:
                        print(f"检测到声音 (音量: {current_volume})，开始录音")
                        detected_sound = True
                        break
                    else:
                        # 打印当前音量用于调试
                        print(f"当前音量: {current_volume} (阈值: {SILENCE_THRESHOLD})", end='\r')
                    silent_frame_count += 1
                except OSError as e:
                    print(f"音频流读取错误: {e}, 重新初始化音频设备")
                    # 重新初始化音频设备
                    stream.close()
                    p.terminate()
                    p = pyaudio.PyAudio()
                    stream = p.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK)
                    print("音频设备重新初始化完成")
                    silent_frame_count = 0  # 重置计数器
                    continue
                except Exception as e:
                     print(f"声音检测过程中出错: {e}")
                     break # 发生其他异常，退出循环

            if not detected_sound:
                print("等待超时，未检测到声音")
                continue  # 跳过本次录音，重新开始

            # 开始录音
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except OSError as e:
                    print(f"录音过程中出错: {e}, 重新初始化音频设备")
                    # 重新初始化音频设备
                    stream.close()
                    p.terminate()
                    p = pyaudio.PyAudio()
                    try:
                         stream = p.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)
                         print("音频设备重新初始化完成")
                         frames = []  # 清空当前录音
                         break  # 跳出录音循环重新开始
                    except Exception as e:
                         print(f"重试打开音频流失败: {e}")
                         break # 无法重新打开音频流，跳出录音
                except Exception as e:
                    print(f"录音过程中出错: {e}")
                    break # 其他错误直接跳出

            # 将音频数据转换为numpy数组
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

            # 保存临时文件用于特征提取
            temp_file = "temp.wav"
            try:
                wf = wave.open(temp_file, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
            except Exception as e:
                print(f"保存临时文件失败: {e}")
                continue


            # 提取MFCC特征
            mfcc = extract_mfcc(temp_file)

            # 检查mfcc是否有效
            if mfcc is None:
                print("提取MFCC特征失败，跳过本次识别。")
                os.remove(temp_file) # 删除临时文件
                continue
            # 转换为PyTorch张量并移至GPU
            mfcc_tensor = torch.FloatTensor(mfcc).to(device)

            # 进行预测
            with torch.no_grad():
                outputs = model(mfcc_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # 输出结果（使用安全函数避免索引越界）
            if confidence > 0.4:
                result = safe_get_class(predicted_class)
                print(f"识别结果: {result} (置信度: {confidence:.2f})")
            else:
                print(f"未识别到有效命令 (置信度: {confidence:.2f})")

            # 删除临时文件
            os.remove(temp_file)

    except KeyboardInterrupt:
        print("停止识别")
    finally:
        try:
            stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass


# 程序入口
if __name__ == "__main__":
    # 设置环境变量，解决 librosa mp3 decode 报错
    os.environ['PATH'] = os.environ['PATH'] + (';C:\\ffmpeg\\bin' if os.name == 'nt' else ':/usr/bin')
    #main()  # 执行训练和评估
    main2()  # 执行实时语音识别