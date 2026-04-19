# 模型下载指南

本项目使用深度学习模型进行语音识别和翻译。由于模型文件较大（每个约2-5GB），未包含在仓库中。

## 快速开始

### 方法一：自动下载（推荐）

首次运行程序时，如果检测到模型未下载，会自动提示下载。

或者手动运行：

```bash
# 下载所有推荐模型
python download_models.py --all

# 只下载ASR模型
python download_models.py --asr faster-whisper-large-v3

# 只下载翻译模型
python download_models.py --translate nllb-200-distilled-600M

# 查看已下载的模型
python download_models.py --list
```

### 方法二：手动下载

如果自动下载失败，可以从 HuggingFace 手动下载：

#### ASR模型
- **Faster Whisper Large V3**: [guillaumekln/faster-whisper-large-v3](https://huggingface.co/guillaumekln/faster-whisper-large-v3)
- **Faster Whisper Medium**: [guillaumekln/faster-whisper-medium](https://huggingface.co/guillaumekln/faster-whisper-medium)

下载后请解压到以下目录结构：
```
models/
└── asr/
    └── faster-whisper-large-v3/
        ├── config.json
        ├── model.bin
        ├── tokenizer.json
        └── vocabulary.txt
```

#### 翻译模型
- **NLLB-200 600M**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **NLLB-200 1.3B**: [facebook/nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)

下载后请解压到以下目录结构：
```
models/
└── translate/
    └── nllb-200-distilled-600M/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── ...
```

## 可用模型列表

### ASR模型
| 模型名称 | 描述 | 大小 | 精度 | 速度 |
|---------|------|------|------|------|
| faster-whisper-large-v3 | Large V3 模型 | ~3GB | 高 | 中等 |
| faster-whisper-medium | Medium 模型 | ~1.5GB | 中 | 快 |

### 翻译模型
| 模型名称 | 描述 | 大小 | 语言 |
|---------|------|------|------|
| nllb-200-distilled-600M | NLLB 600M | ~2.4GB | 200+ 语言 |
| nllb-200-distilled-1.3B | NLLB 1.3B | ~5GB | 200+ 语言 |

## 网络问题解决

如果在国内无法访问 HuggingFace，可以：

1. **使用镜像站**：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python download_models.py --all
   ```

2. **手动下载后放到指定目录**（见上面的手动下载说明）

## 验证模型完整性

下载完成后，可以运行以下命令验证：

```bash
python download_models.py --list
```

确保显示"✓ 完整"状态。
