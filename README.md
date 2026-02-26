```markdown
# 行人重识别 (Person Re-ID) 项目

基于 `torchreid` 库的行人重识别系统，包含模型训练和图形界面演示。

## 📁 项目结构

```
reid/
├── train.py          # 模型训练脚本
├── gui_demo.py       # 图形界面演示程序
└── README.md         # 项目说明文档
```

## 🔧 环境要求

- Python 3.7+
- PyTorch 1.9+
- torchreid
- PyQt5
- PIL
- NumPy

### 安装依赖

```bash
pip install torch torchreid pyqt5 pillow numpy
```

## 🚀 train.py - 模型训练

训练一个基于 ResNet50 的行人重识别模型。

### 功能特点
- 使用 Market1501 数据集
- ResNet50 骨干网络
- Softmax 分类损失
- Adam 优化器

### 使用方法

1. **准备数据集**
   - 确保 Market1501 数据集存放在 `E:/pylearning/reid-data/` 目录下
   - 数据集结构应为：
     ```
     reid-data/
     └── market1501/
         ├── bounding_box_train/  # 训练集
         ├── bounding_box_test/   # 测试集
         └── query/                # 查询集
     ```

2. **修改配置**（可选）
   ```python
   # 在 train.py 中调整以下参数
   root='E:/pylearning/reid-data',     # 数据集路径
   height=256,                          # 图片高度
   width=128,                            # 图片宽度
   batch_size_train=32,                  # 训练批次大小
   batch_size_test=64,                    # 测试批次大小
   max_epoch=60,                          # 训练轮数
   lr=0.00035                             # 学习率
   ```

3. **开始训练**
   ```bash
   python train.py
   ```

4. **训练输出**
   - 模型会保存在 `logs/resnet50/model/` 目录下
   - 每轮训练后会保存 checkpoint 文件（如 `model.pth.tar-60`）
   - 训练日志会显示损失值和准确率

## 🖥️ gui_demo.py - 图形界面演示

使用训练好的模型进行行人检索的图形界面程序。

### 功能特点
- 支持选择查询图片
- 自动提取特征并检索相似图片
- 显示 Top-10 检索结果
- 特征缓存机制，加速重复检索

### 使用方法

1. **准备模型**
   - 确保已有训练好的模型文件
   - 默认路径：`E:/pylearning/reid_project/logs/resnet50/model/model.pth.tar-60`
   - 如需使用其他模型，请修改 `checkpoint_path` 变量

2. **准备图库**
   - 确保测试集图片在 `E:/pylearning/reid_data/market1501/bounding_box_test/` 目录下
   - 程序会自动缓存特征到 `gallery_feats.npy`

3. **启动程序**
   ```bash
   python gui_demo.py
   ```

4. **使用界面**
   - 点击"选择图片"按钮，选择一张行人图片
   - 点击"开始检索"按钮，等待检索结果
   - 右侧会显示最相似的10张图库图片

### 界面说明

```
┌──────────────────────────────────────┐
│  行人重识别系统 Demo                   │
├──────────────┬───────────────────────┤
│              │  Top-1  Top-2  ...    │
│  查询图片     │  Top-3  Top-4  ...    │
│              │  Top-5  Top-6  ...    │
│  [选择图片]   │  Top-7  Top-8  ...    │
│  [开始检索]   │  Top-9  Top-10        │
└──────────────┴───────────────────────┘
```

## ⚙️ 配置说明

### train.py 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `root` | 数据集根目录 | `E:/pylearning/reid-data` |
| `height` | 输入图片高度 | 256 |
| `width` | 输入图片宽度 | 128 |
| `batch_size_train` | 训练批次大小 | 32 |
| `max_epoch` | 训练轮数 | 60 |
| `lr` | 学习率 | 0.00035 |

### gui_demo.py 主要配置

| 配置项 | 说明 | 默认路径 |
|--------|------|----------|
| `checkpoint_path` | 模型文件路径 | `E:/pylearning/reid_project/logs/resnet50/model/model.pth.tar-60` |
| `gallery_dir` | 图库目录 | `E:/pylearning/reid_data/market1501/bounding_box_test` |

## 📊 数据集说明

本项目使用 **Market1501** 数据集：
- 训练集：12,936 张图片，751 个行人ID
- 测试集：19,732 张图片，750 个行人ID
- 查询集：3,368 张图片

## 💡 注意事项

1. **Windows 系统**：`train.py` 中已设置 `workers=0`，避免多进程问题
2. **GPU 要求**：建议使用 GPU 训练，CPU 训练会很慢
3. **缓存文件**：`gui_demo.py` 会生成 `gallery_feats.npy` 缓存文件，如需重新提取特征请删除该文件
4. **路径修改**：请根据实际路径修改代码中的文件路径

## 🐛 常见问题

### Q: 训练时提示找不到数据集
A: 检查 `root` 参数是否正确指向包含 market1501 的目录

### Q: gui_demo.py 启动后无法显示图片
A: 确保图片路径中包含中文字符？建议使用纯英文路径

### Q: 检索速度慢
A: 第一次检索会提取特征并缓存，第二次开始会快很多

## 👥 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！
```
