# GroundingDINO物体检测和抓取点计算使用说明

## 概述

本程序使用GroundingDINO模型进行物体检测和抓取点计算，相比传统的YOLO模型，GroundingDINO具有以下优势：

1. **支持自然语言描述**：可以使用文本描述来检测任意物体，不限于预训练类别
2. **更高的灵活性**：支持复杂的物体描述，如"red apple"、"metal bottle"等
3. **零样本检测**：无需重新训练即可检测新的物体类别

## 环境安装

### 方法1：使用安装脚本（推荐）

```bash
chmod +x install_grounding_dino_env.sh
./install_grounding_dino_env.sh
```

### 方法2：手动安装

```bash
# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio

# 安装transformers
pip install transformers

# 安装其他依赖
pip install -r requirements_grounding_dino.txt
```

## 使用方法

### 基本使用

```bash
python3 grounding_dino_detection_and_grasping.py
```

运行后程序会提示输入：
1. 目标物体描述（如：bottle、cup、phone等）
2. 置信度阈值（0.0-1.0，默认0.5）

### 在代码中使用

```python
from grounding_dino_detection_and_grasping import GroundingDINODetectionAndGrasping

# 创建检测器实例
detector = GroundingDINODetectionAndGrasping(target_description="bottle")

# 运行检测和抓取点计算
success = detector.run_detection_and_grasping(
    text_prompt="bottle",
    confidence_threshold=0.5
)
```

## 支持的物体描述示例

### 基本物体
- `bottle` - 瓶子
- `cup` - 杯子
- `phone` - 手机
- `apple` - 苹果
- `book` - 书本
- `keys` - 钥匙

### 带颜色的物体
- `red apple` - 红色苹果
- `blue bottle` - 蓝色瓶子
- `white cup` - 白色杯子

### 带材质的物体
- `metal bottle` - 金属瓶子
- `plastic cup` - 塑料杯子
- `glass bottle` - 玻璃瓶子

### 带形状的物体
- `round bottle` - 圆形瓶子
- `square box` - 方形盒子

## 输出文件

程序会生成以下文件：

1. **点坐标文件** (`/path/to/point.txt`)：
   ```
   320        # x坐标
   240        # y坐标
   450.5      # 深度值(mm)
   ```

2. **检测结果图像** (`grounding_dino_detection_result.png`)：
   - 显示检测框
   - 显示置信度
   - 显示抓取点位置

3. **原始图像** (`rgb.png`)：
   - 保存的RGB图像

4. **深度数据** (`depth.npy`)：
   - 原始深度数据

5. **深度可视化** (`depth_visualization.png`)：
   - 深度图像的彩色可视化

## 参数调整

### 置信度阈值
- **默认值**：0.5
- **范围**：0.0-1.0
- **建议**：
  - 0.3-0.5：检测更多物体，可能有误检
  - 0.5-0.7：平衡检测率和准确率
  - 0.7-0.9：只检测高置信度物体

### 深度计算参数
可以在代码中调整以下参数：
- `num_frames`：用于平均的帧数（默认20）
- `skip_frames`：跳过的初始帧数（默认50）
- `window_size`：深度计算窗口大小（默认15）

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案：
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 清理pip缓存：pip cache purge
   ```

2. **CUDA内存不足**
   ```
   解决方案：
   - 使用CPU模式：在代码中设置device="cpu"
   - 减少batch size
   - 关闭其他GPU程序
   ```

3. **检测结果不准确**
   ```
   解决方案：
   - 调整置信度阈值
   - 改进文本描述
   - 检查光照条件
   - 确保物体在相机视野内
   ```

4. **深度信息无效**
   ```
   解决方案：
   - 检查相机连接
   - 确保物体在深度相机范围内
   - 调整深度计算窗口大小
   ```

### 性能优化建议

1. **使用GPU加速**：
   - 安装CUDA版本的PyTorch
   - 确保GPU内存充足

2. **模型优化**：
   - 使用ONNX运行时
   - 量化模型（FP16）

3. **图像预处理**：
   - 调整图像分辨率
   - 优化相机参数

## 与YOLO的对比

| 特性 | YOLO | GroundingDINO |
|------|------|---------------|
| 检测类别 | 固定80类 | 任意文本描述 |
| 灵活性 | 低 | 高 |
| 速度 | 快 | 中等 |
| 准确率 | 高（预训练类别） | 高（任意类别） |
| 内存使用 | 低 | 中等 |

## 扩展功能

### 1. 批量检测
```python
# 检测多个物体
detections = detector.detect_objects_with_grounding_dino(
    rgb_image, 
    text_prompt="bottle . cup . phone",  # 用.分隔多个物体
    confidence_threshold=0.5
)
```

### 2. 区域检测
```python
# 在指定区域内检测
# 可以通过修改图像ROI来实现
```

### 3. 实时检测
```python
# 连续检测模式
while True:
    success = detector.run_detection_and_grasping()
    if success:
        break
```

## 技术支持

如果遇到问题，请检查：
1. 所有依赖是否正确安装
2. 相机是否正常工作
3. 模型文件是否下载完整
4. 系统资源是否充足

更多技术细节请参考：
- [GroundingDINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Transformers文档](https://huggingface.co/transformers/)
- [PyTorch文档](https://pytorch.org/docs/)
