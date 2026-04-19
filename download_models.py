"""
模型自动下载脚本
确保模型目录结构符合项目要求
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("请安装依赖: pip install requests tqdm")
    sys.exit(1)


# 模型配置：定义所有支持的模型及其下载信息
MODELS_CONFIG = {
    "asr": {
        "faster-whisper-large-v3": {
            "repo_id": "guillaumekln/faster-whisper-large-v3",
            "files": [
                "config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.txt",
                "vocabulary.json"
            ],
            "description": "Faster Whisper Large V3 - ASR模型",
            "size_gb": 3.0
        },
        "faster-whisper-medium": {
            "repo_id": "guillaumekln/faster-whisper-medium",
            "files": [
                "config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.txt",
                "vocabulary.json"
            ],
            "description": "Faster Whisper Medium - ASR模型(较小)",
            "size_gb": 1.5
        }
    },
    "translate": {
        "nllb-200-distilled-600M": {
            "repo_id": "facebook/nllb-200-distilled-600M",
            "files": [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "generation_config.json"
            ],
            "description": "NLLB-200 Distilled 600M - 翻译模型",
            "size_gb": 2.4
        },
        "nllb-200-distilled-1.3B": {
            "repo_id": "facebook/nllb-200-distilled-1.3B",
            "files": [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "generation_config.json"
            ],
            "description": "NLLB-200 Distilled 1.3B - 翻译模型(更大)",
            "size_gb": 5.0
        }
    }
}


def get_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """计算文件的哈希值"""
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url: str, dest: Path, expected_hash: Optional[str] = None) -> bool:
    """
    下载文件到指定路径

    Args:
        url: 下载URL
        dest: 目标路径
        expected_hash: 期望的哈希值（用于校验）

    Returns:
        bool: 下载是否成功
    """
    try:
        # 检查文件是否已存在
        if dest.exists():
            if expected_hash:
                actual_hash = get_file_hash(dest)
                if actual_hash == expected_hash:
                    print(f"  ✓ {dest.name} 已存在且校验通过，跳过")
                    return True
                else:
                    print(f"  ⚠ {dest.name} 已存在但校验失败，重新下载")
                    dest.unlink()
            else:
                print(f"  ✓ {dest.name} 已存在，跳过")
                return True

        # 下载文件
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # 显示进度条
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # 校验文件
        if expected_hash:
            actual_hash = get_file_hash(dest)
            if actual_hash != expected_hash:
                dest.unlink()
                print(f"  ✗ {dest.name} 校验失败")
                return False

        print(f"  ✓ {dest.name} 下载成功")
        return True

    except Exception as e:
        print(f"  ✗ {dest.name} 下载失败: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_model(model_type: str, model_name: str, models_dir: Path) -> bool:
    """
    下载单个模型

    Args:
        model_type: 模型类型 ("asr" 或 "translate")
        model_name: 模型名称
        models_dir: 模型根目录

    Returns:
        bool: 是否下载成功
    """
    if model_type not in MODELS_CONFIG:
        print(f"✗ 未知的模型类型: {model_type}")
        return False

    if model_name not in MODELS_CONFIG[model_type]:
        print(f"✗ 未知的模型: {model_name}")
        return False

    config = MODELS_CONFIG[model_type][model_name]
    model_dir = models_dir / model_type / model_name

    # 创建模型目录
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 下载 {config['description']}")
    print(f"   仓库: {config['repo_id']}")
    print(f"   目标: {model_dir}")
    print(f"   大小: ~{config['size_gb']} GB")

    # 下载所有必需文件
    base_url = f"https://huggingface.co/{config['repo_id']}/resolve/main/"
    success_count = 0

    for filename in config['files']:
        url = urljoin(base_url, filename)
        dest_path = model_dir / filename

        if download_file(url, dest_path):
            success_count += 1

    # 检查是否所有文件都下载成功
    if success_count == len(config['files']):
        print(f"✓ {model_name} 下载完成 ({success_count}/{len(config['files'])} 文件)")
        return True
    else:
        print(f"✗ {model_name} 下载不完整 ({success_count}/{len(config['files'])} 文件)")
        # 清理不完整的下载
        if model_dir.exists():
            shutil.rmtree(model_dir)
        return False


def verify_model(model_type: str, model_name: str, models_dir: Path) -> bool:
    """
    验证模型是否完整

    Args:
        model_type: 模型类型
        model_name: 模型名称
        models_dir: 模型根目录

    Returns:
        bool: 模型是否完整
    """
    if model_type not in MODELS_CONFIG or model_name not in MODELS_CONFIG[model_type]:
        return False

    config = MODELS_CONFIG[model_type][model_name]
    model_dir = models_dir / model_type / model_name

    if not model_dir.exists():
        return False

    # 检查所有必需文件是否存在
    for filename in config['files']:
        if not (model_dir / filename).exists():
            return False

    return True


def list_models(models_dir: Path):
    """列出当前已下载的模型"""
    print("\n📋 已下载的模型:")
    print("-" * 50)

    for model_type in ["asr", "translate"]:
        type_dir = models_dir / model_type
        if not type_dir.exists():
            continue

        for model_dir in type_dir.iterdir():
            if model_dir.is_dir():
                # 检查模型是否完整
                for available_name in MODELS_CONFIG[model_type].keys():
                    if available_name in model_dir.name:
                        is_complete = verify_model(model_type, available_name, models_dir)
                        status = "✓ 完整" if is_complete else "✗ 不完整"
                        desc = MODELS_CONFIG[model_type][available_name]["description"]
                        print(f"  [{model_type.upper()}] {available_name}")
                        print(f"    状态: {status}")
                        print(f"    描述: {desc}")
                        print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="模型下载管理工具")
    parser.add_argument("--list", action="store_true", help="列出已下载的模型")
    parser.add_argument("--asr", metavar="MODEL", help="下载ASR模型 (可选项: faster-whisper-large-v3, faster-whisper-medium)")
    parser.add_argument("--translate", metavar="MODEL", help="下载翻译模型 (可选项: nllb-200-distilled-600M, nllb-200-distilled-1.3B)")
    parser.add_argument("--all", action="store_true", help="下载所有推荐的模型")
    parser.add_argument("--models-dir", type=str, default="./models", help="模型存储目录")

    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    # 列出已下载的模型
    if args.list:
        list_models(models_dir)
        return

    # 下载指定的ASR模型
    if args.asr:
        download_model("asr", args.asr, models_dir)

    # 下载指定的翻译模型
    if args.translate:
        download_model("translate", args.translate, models_dir)

    # 下载所有推荐的模型
    if args.all:
        print("🚀 开始下载所有推荐模型...")
        recommended = [
            ("asr", "faster-whisper-large-v3"),
            ("translate", "nllb-200-distilled-600M")
        ]

        for model_type, model_name in recommended:
            download_model(model_type, model_name, models_dir)

    # 如果没有指定任何操作，显示帮助
    if not any([args.list, args.asr, args.translate, args.all]):
        parser.print_help()
        print("\n💡 快速开始:")
        print("  python download_models.py --all          # 下载所有推荐模型")
        print("  python download_models.py --list         # 查看已下载的模型")
        print("  python download_models.py --asr faster-whisper-medium  # 下载指定模型")


if __name__ == "__main__":
    main()
