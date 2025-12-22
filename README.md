# 🌊 FFT 频域水印工具

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![DearPyGui](https://img.shields.io/badge/GUI-DearPyGui-orange.svg)

**一个基于傅里叶变换的图像频域水印嵌入工具**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [使用说明](#-使用说明) • [技术原理](#-技术原理)

</div>

---

## 📖 项目简介

FFT 频域水印工具是一款基于**快速傅里叶变换（FFT）**技术的图像水印嵌入应用程序。与传统的空域水印不同，本工具将水印信息嵌入到图像的**频率域**中，使水印更加隐蔽且具有更好的鲁棒性。

### 工作原理

```
原始图像 → FFT变换 → 频域嵌入水印 → IFFT逆变换 → 带水印图像
```

## ✨ 功能特性

- 🎨 **三通道处理** - 分别对 RGB 三个通道进行独立的 FFT 变换和水印嵌入
- 📍 **灵活的位置选择** - 支持5种水印位置：左上、右上、左下、右下、居中
- 🔧 **可调参数** - 可自定义水印文字、字体大小、嵌入强度
- 👁️ **实时预览** - 同时显示原图、频谱图和处理结果
- 📊 **频谱可视化** - 支持查看和保存带水印的频谱图像
- 💾 **多格式支持** - 支持 PNG、JPG、JPEG、BMP 等常见图像格式
- 🔄 **FFT Only 模式** - 仅进行FFT变换，用于分析频谱特征

## 🚀 快速开始

### 环境要求

- Python 3.13+
- Windows / Linux / macOS

### 安装依赖

本项目使用 `uv` 进行包管理，推荐使用以下方式安装：

```bash
# 使用 uv 安装依赖
uv sync
```

或者使用 pip 安装：

```bash
pip install dearpygui numpy opencv-python scipy
```

### 运行应用

```bash
# 使用 uv 运行
uv run python fft_watermark_app.py

# 或直接运行
python fft_watermark_app.py
```

## 📖 使用说明

### 界面概览

应用程序界面分为以下几个部分：

1. **图像输入区** - 输入或浏览选择图像文件
2. **水印设置区** - 配置水印文字、位置、大小和强度
3. **图像预览区** - 显示原图、频谱图和处理结果
4. **操作按钮区** - 处理图像、保存结果

### 操作步骤

1. **加载图像**
   - 在输入框中输入图像路径，或点击 `Browse` 按钮选择图像
   - 点击 `Load Image` 按钮加载图像

2. **设置水印参数**
   - **Watermark Text**: 输入要嵌入的水印文字
   - **Position**: 选择水印在频域中的位置
   - **Font Size**: 调整水印字体大小（10-100）
   - **Strength**: 调整水印嵌入强度（1.0-200.0）
   - **FFT Only Mode**: 勾选后仅进行FFT变换，不进行逆变换

3. **处理图像**
   - 点击 `Process Image (Add Watermark)` 按钮开始处理
   - 处理完成后可在预览区查看结果

4. **保存结果**
   - 点击 `Save Result` 保存处理后的图像
   - 点击 `Save Spectrum` 保存频谱可视化图像

### 参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| Font Size | 10-100 | 水印文字大小，越大在频谱中越明显 |
| Strength | 1.0-200.0 | 嵌入强度，值越大水印越明显但可能影响图像质量 |

## 🔬 技术原理

### 傅里叶变换

快速傅里叶变换（FFT）将图像从空间域转换到频率域：
- **低频分量**（中心区域）：代表图像的整体亮度和缓慢变化的区域
- **高频分量**（边缘区域）：代表图像的细节、边缘和噪声

### 频域水印嵌入

本工具通过以下步骤嵌入水印：

1. **FFT 变换**：对图像的 R、G、B 三个通道分别进行 2D FFT
2. **频移**：使用 `fftshift` 将零频分量移到频谱中心
3. **水印生成**：根据用户设置创建文字水印掩码
4. **水印嵌入**：将水印掩码按指定强度叠加到频谱上
5. **逆 FFT**：进行 `ifftshift` 和 `ifft2` 还原到空间域

### 水印特点

频域水印相比空域水印具有以下优势：
- **隐蔽性**：水印不直接可见于图像中
- **鲁棒性**：对某些图像处理操作（如裁剪、缩放）有一定的抵抗能力
- **可验证性**：通过 FFT 可以检测水印是否存在

## 📁 项目结构

```
fft_watermark/
├── fft_watermark_app.py    # 主应用程序
├── pyproject.toml          # 项目配置文件
├── uv.lock                 # 依赖锁定文件
├── README.md               # 项目说明文档
├── .gitignore              # Git 忽略配置
└── .python-version         # Python 版本配置
```

## 🔧 依赖项

| 包名 | 用途 |
|------|------|
| `dearpygui` | GUI 框架，用于构建用户界面 |
| `numpy` | 数值计算，FFT 变换核心 |
| `opencv-python` | 图像读取、处理和保存 |
| `scipy` | 科学计算支持 |

## 📝 开发计划

- [ ] 支持批量处理多张图像
- [ ] 添加水印检测/提取功能
- [ ] 支持图片水印（不仅限于文字）
- [ ] 添加更多水印嵌入算法
- [ ] 支持视频水印

## 🔨 构建与发布

### 手动构建

如需本地打包，可执行：

```bash
# 安装 PyInstaller
pip install pyinstaller

# 打包为单文件 exe（无控制台窗口）
pyinstaller --onefile --noconsole --name "FFT_Watermark" fft_watermark_app.py
```

生成的可执行文件位于 `dist/FFT_Watermark.exe`

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件 