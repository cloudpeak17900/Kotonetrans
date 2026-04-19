#  Kotonetrans

> 日语音声/广播剧翻译工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-API-purple.svg)](https://deepseek.com)


## 说明
run.bat启动

默认先降噪预处理。支持逐句、批量翻译，内置两个whisper提词模型（推荐faster）和一个翻译模型，如后续自行添加model请按相应格式，可能存在识别问题。

可选择接入网络大模型并能api预检是否正常

有预设关键词功能（如：”你是一名日语音声翻译者”，请将人名xxx翻译为xxx）

支持浏览全部字幕和拖动进度条

| 测试项 | 数据 |
|--------|------|
| 音频时长 | 9分44秒 |
| 短句数量 | 120句 |
| 文件大小 | 8.7 MB |
| 处理耗时 | ≈5分钟 |
| API成本 | <¥0.01 (DeepSeek) |

<img width="1920" height="1080" alt="aeced7b8a0a8e2702c41de95ed17ff7d" src="https://github.com/user-attachments/assets/e6a87bae-b91c-41d3-9edf-21803c10680a" />




### 快速开始

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 下载模型

由于模型文件较大，首次运行时需要下载：

```bash
# 自动下载所有推荐模型（推荐）
python download_models.py --all

# 或手动选择下载
python download_models.py --asr faster-whisper-large-v3
python download_models.py --translate nllb-200-distilled-600M
```

详细说明请查看 [模型下载指南](MODELS.md)

#### 3. 运行程序

```bash
run.bat
```

### 环境要求

- Python 3.8+
- FFmpeg（音频处理）
- 5GB+ 磁盘空间（模型文件）



