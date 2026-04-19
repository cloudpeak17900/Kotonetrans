"""
模型管理模块
负责模型的发现、加载、缓存和内存管理
"""

import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List


class ModelManager:
    """模型管理器（单例模式）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 模型存储目录
        self.models_dir = Path("./models")
        self.asr_dir = self.models_dir / "asr"
        self.translate_dir = self.models_dir / "translate"
        
        # 创建目录
        self.models_dir.mkdir(exist_ok=True)
        self.asr_dir.mkdir(exist_ok=True)
        self.translate_dir.mkdir(exist_ok=True)
        
        # 已加载的模型
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_used: Dict[str, float] = {}
        
        # 模型锁
        self.model_locks: Dict[str, threading.Lock] = {}
        
        # 内存管理
        self.max_cached_models = 2
        self.model_ttl = 300
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._auto_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        self._initialized = True
    
    def scan_asr_models(self) -> List[str]:
        """
        扫描asr目录，发现可用的ASR模型
        返回: 模型名称列表
        """
        models = []
        if not self.asr_dir.exists():
            return models
        
        for item in self.asr_dir.iterdir():
            if item.is_dir():
                # Whisper模型目录格式: models--openai--whisper-tiny
                if "whisper" in item.name:
                    # 提取模型大小
                    for size in ["tiny", "base", "small", "medium", "large"]:
                        if size in item.name:
                            models.append(f"whisper-{size}")
                            break
                # 其他ASR模型可以在这里添加识别逻辑
        
        return models
    
    def scan_translate_models(self) -> List[str]:
        """
        扫描translate目录，发现可用的翻译模型
        返回: 模型名称列表
        """
        models = []
        if not self.translate_dir.exists():
            return models
        
        for item in self.translate_dir.iterdir():
            if item.is_dir():
                # HuggingFace缓存目录格式: models--Helsinki-NLP--opus-mt-ja-zh
                if "opus-mt-ja-zh" in item.name:
                    models.append("opus-mt-ja-zh")
                elif "nllb" in item.name:
                    models.append("nllb-200-600M")
                # 其他翻译模型可以在这里添加识别逻辑
        
        return models
    
    def get_model_path(self, model_name: str, model_type: str) -> Optional[Path]:
        """
        获取模型的本地路径
        
        Args:
            model_name: 模型名称
            model_type: "asr" 或 "translate"
        """
        if model_type == "asr":
            # Whisper模型
            if model_name.startswith("whisper-"):
                size = model_name.replace("whisper-", "")
                # Whisper的缓存目录格式
                cache_name = f"models--openai--whisper-{size}"
                return self.asr_dir / cache_name
        else:
            # 翻译模型
            if model_name == "opus-mt-ja-zh":
                return self.translate_dir / "models--Helsinki-NLP--opus-mt-ja-zh"
            elif model_name == "nllb-200-600M":
                return self.translate_dir / "models--facebook--nllb-200-distilled-600M"
        
        return None
    
    def is_model_available(self, model_name: str, model_type: str) -> bool:
        """检查模型是否已下载"""
        path = self.get_model_path(model_name, model_type)
        if path is None or not path.exists():
            return False

        # 检查必需文件是否存在
        required_files = self._get_model_required_files(model_name, model_type)
        return all((path / f).exists() for f in required_files)

    def _get_model_required_files(self, model_name: str, model_type: str) -> List[str]:
        """获取模型必需的文件列表"""
        if model_type == "asr":
            if "faster-whisper" in model_name:
                return ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
            else:  # whisper
                return ["config.json", "model.bin"]
        else:  # translate
            if "nllb" in model_name:
                return ["config.json", "pytorch_model.bin", "tokenizer.json",
                       "tokenizer_config.json", "sentencepiece.bpe.model"]
            else:  # opus-mt
                return ["config.json", "pytorch_model.bin", "tokenizer.json"]

    def ensure_model_downloaded(self, model_name: str, model_type: str) -> bool:
        """
        确保模型已下载，如果未下载则自动下载

        Args:
            model_name: 模型名称
            model_type: "asr" 或 "translate"

        Returns:
            bool: 是否模型可用
        """
        if self.is_model_available(model_name, model_type):
            return True

        print(f"⚠ 模型 {model_name} 未找到，尝试自动下载...")

        try:
            # 导入下载模块
            import sys
            import subprocess

            # 获取download_models.py路径
            download_script = Path(__file__).parent / "download_models.py"

            if not download_script.exists():
                print(f"✗ 找不到下载脚本: {download_script}")
                print(f"  请手动下载模型或运行: python download_models.py --{model_type} {model_name}")
                return False

            # 构建下载命令
            cmd = [sys.executable, str(download_script), f"--{model_type}", model_name, "--models-dir", str(self.models_dir)]

            print(f"📦 正在下载模型 {model_name}...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ 模型 {model_name} 下载成功")
                return self.is_model_available(model_name, model_type)
            else:
                print(f"✗ 模型 {model_name} 下载失败:")
                print(result.stderr)
                return False

        except Exception as e:
            print(f"✗ 自动下载失败: {e}")
            print(f"  请手动运行: python download_models.py --{model_type} {model_name}")
            return False
    
    def get_model(self, model_name: str, model_type: str, force_reload: bool = False) -> Optional[Any]:
        """
        获取模型实例（懒加载）
        
        Args:
            model_name: 模型名称
            model_type: "asr" 或 "translate"
            force_reload: 是否强制重新加载
        """
        model_key = f"{model_type}_{model_name}"
        
        # 获取锁
        if model_key not in self.model_locks:
            self.model_locks[model_key] = threading.Lock()
        
        with self.model_locks[model_key]:
            # 检查是否已加载
            if not force_reload and model_key in self.loaded_models:
                self.model_last_used[model_key] = time.time()
                return self.loaded_models[model_key]
            
            # 卸载旧模型（如果存在）
            if model_key in self.loaded_models:
                self._unload_model(model_key)
            
            # 加载新模型
            model = self._load_model(model_name, model_type)
            if model:
                self.loaded_models[model_key] = model
                self.model_last_used[model_key] = time.time()
                self._check_cache_limit()
                return model
        
        return None
    
    def _load_model(self, model_name: str, model_type: str) -> Optional[Any]:
        """实际加载模型"""
        try:
            if model_type == "asr":
                return self._load_asr_model(model_name)
            else:
                return self._load_translate_model(model_name)
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            return None
    
    def _load_asr_model(self, model_name: str):
        """加载ASR模型"""
        # 确保模型已下载
        if not self.ensure_model_downloaded(model_name, "asr"):
            raise RuntimeError(f"无法加载ASR模型 {model_name}，请确保模型已正确下载")

        import whisper

        if not model_name.startswith("whisper-"):
            raise ValueError(f"不支持的ASR模型: {model_name}")

        size = model_name.replace("whisper-", "")

        model = whisper.load_model(
            size,
            device="cpu",
            download_root=str(self.asr_dir),
            in_memory=False
        )

        return model
    
    def _load_translate_model(self, model_name: str):
        """加载翻译模型"""
        # 确保模型已下载
        if not self.ensure_model_downloaded(model_name, "translate"):
            raise RuntimeError(f"无法加载翻译模型 {model_name}，请确保模型已正确下载")

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_map = {
            "opus-mt-ja-zh": "Helsinki-NLP/opus-mt-ja-zh",
            "nllb-200-600M": "facebook/nllb-200-distilled-600M"
        }

        if model_name not in model_map:
            raise ValueError(f"不支持的翻译模型: {model_name}")

        model_id = model_map[model_name]
        cache_dir = str(self.translate_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True  # 强制使用本地文件
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            local_files_only=True  # 强制使用本地文件
        )

        return {
            "model": model,
            "tokenizer": tokenizer
        }
    
    def _unload_model(self, model_key: str):
        """卸载模型"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            del self.model_last_used[model_key]
            import gc
            gc.collect()
    
    def _check_cache_limit(self):
        """检查缓存限制"""
        if len(self.loaded_models) <= self.max_cached_models:
            return
        
        oldest = min(self.model_last_used.items(), key=lambda x: x[1])
        self._unload_model(oldest[0])
    
    def _auto_cleanup(self):
        """自动清理长时间未使用的模型"""
        while True:
            time.sleep(60)
            current_time = time.time()
            to_unload = []
            
            for key, last_used in self.model_last_used.items():
                if current_time - last_used > self.model_ttl:
                    to_unload.append(key)
            
            for key in to_unload:
                self._unload_model(key)
    
    def unload_all(self):
        """卸载所有模型"""
        for key in list(self.loaded_models.keys()):
            self._unload_model(key)
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self.loaded_models.keys())
    
    def get_cache_status(self) -> Dict:
        """获取缓存状态"""
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "loaded_count": len(self.loaded_models),
            "max_cache": self.max_cached_models
        }


# 全局实例
model_manager = ModelManager()