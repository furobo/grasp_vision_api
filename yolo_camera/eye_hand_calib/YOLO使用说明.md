# YOLO检测功能使用说明

## 功能概述

本系统已成功集成YOLOv8目标检测功能，支持在机器人标定GUI中进行实时目标检测和抓取定位。

## 主要功能

### 1. 实时检测模式
- **功能**: 在相机画面中实时显示YOLO检测结果
- **使用方法**: 勾选"实时检测"复选框
- **显示效果**: 
  - 绿色边界框标识检测到的目标
  - 红色圆点标识目标中心点
  - 白色文字显示类别、置信度和3D位置信息

### 2. 单次检测模式
- **功能**: 对当前帧进行一次性检测
- **使用方法**: 点击"YOLO检测"按钮
- **结果显示**: 弹出对话框显示检测结果统计

### 3. 检测结果保存
- **功能**: 保存检测结果到文件夹
- **使用方法**: 点击"保存检测结果"按钮
- **保存内容**:
  - 原始图像 (`original_YYYYMMDD_HHMMSS.jpg`)
  - 检测结果图像 (`detection_YYYYMMDD_HHMMSS.jpg`)
  - 详细信息文本 (`results_YYYYMMDD_HHMMSS.txt`)

## 启动方法

### 方法1: 使用启动脚本（推荐）
```bash
# 给脚本执行权限
chmod +x launch_yolo_gui.sh

# 运行脚本
./launch_yolo_gui.sh
```

### 方法2: 手动启动
```bash
# 激活conda环境
conda activate camera_yolo

# 设置X11显示
export DISPLAY=localhost:10.0

# 运行GUI
python calibration_viewer.py
```

### 方法3: 独立测试
```bash
# 运行YOLO检测测试程序
python test_yolo_gui.py
```

## 配置说明

### config.yaml 配置文件
```yaml
yolo:
  model_path: "yolov8n.pt"          # YOLO模型路径
  class_names: ["bottle", "cup"]     # 目标类别
  conf_threshold: 0.25              # 置信度阈值
  nms_threshold: 0.4                # NMS阈值
```

### 支持的模型
- `yolov8n.pt` - YOLOv8 Nano (快速，精度一般)
- `yolov8s.pt` - YOLOv8 Small (平衡)
- `yolov8m.pt` - YOLOv8 Medium (准确，较慢)
- `yolov8l.pt` - YOLOv8 Large (高精度)
- `yolov8x.pt` - YOLOv8 Extra Large (最高精度)

## 检测结果说明

### 3D位置计算
系统使用相机内参矩阵和深度信息计算目标的3D位置：
- **X**: 水平位置 (米)
- **Y**: 垂直位置 (米)  
- **Z**: 深度距离 (米)

### 置信度阈值
- **默认值**: 0.25
- **调整范围**: 0.1 - 0.9
- **建议值**: 
  - 快速检测: 0.2
  - 准确检测: 0.5
  - 严格检测: 0.7

## 常见问题

### 1. YOLO检测器初始化失败
**原因**: ultralytics包未安装
**解决方法**:
```bash
pip install ultralytics
```

### 2. 检测结果不准确
**原因**: 
- 置信度阈值过低
- 光照条件不佳
- 目标距离过远

**解决方法**:
- 调整config.yaml中的conf_threshold值
- 改善光照条件
- 调整相机位置

### 3. 实时检测卡顿
**原因**: 
- 模型过大
- 计算资源不足

**解决方法**:
- 使用更小的模型 (如yolov8n.pt)
- 降低检测频率
- 关闭不必要的应用程序

### 4. 3D位置计算错误
**原因**:
- 相机内参矩阵不准确
- 深度信息缺失

**解决方法**:
- 重新进行相机标定
- 检查深度相机连接
- 确保目标在有效深度范围内

## 性能优化建议

### 1. 模型选择
- **开发阶段**: 使用yolov8n.pt快速测试
- **生产环境**: 根据精度需求选择合适模型

### 2. 检测频率
- **实时检测**: 建议10-15 FPS
- **单次检测**: 按需触发

### 3. 内存管理
- 定期清理检测结果文件
- 限制保存的历史记录数量

## 技术支持

如遇到问题，请查看以下日志：
- 终端输出信息
- `detection_results/` 文件夹中的保存结果
- Python错误堆栈信息

## 扩展功能

### 自定义模型训练
可以使用自己的数据集训练YOLO模型：
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练自定义模型
model.train(data='path/to/your/dataset.yaml', epochs=100)
```

### 添加新的检测类别
1. 修改config.yaml中的class_names
2. 使用包含新类别的模型
3. 重新启动GUI

---

**版本**: v1.0  
**更新日期**: 2025-01-25  
**兼容性**: Python 3.10+, PyQt5, OpenCV, ultralytics 