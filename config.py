import json
import os
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置管理类（单例模式）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 配置文件路径
        self.config_dir = Path.home() / ".asmr_translator"
        self.config_file = self.config_dir / "config.json"
        
        # 默认配置
        self.default_config = {
            # 音频处理配置
            "audio": {
                "enable_denoise": True,
                "denoise_strength": 0.8,
                "silence_thresh": -40,
                "min_silence_len": 500,
            },
            
            # ASR模型配置
            "asr": {
                "model_name": "whisper-base",
                "device": "cpu",
            },
            
            # 翻译模型配置
            "translate": {
                "model_name": "opus-mt-ja-zh",
                "context_window": 2,
            },
            
            # GUI配置
            "gui": {
                "window_width": 1000,
                "window_height": 700,
                "font_size": 14,
            },
            
            # 播放器配置
            "player": {
                "auto_play": False,
                "volume": 0.8,
            },
            
            # 性能配置
            "performance": {
                "max_workers": 2,
                "lazy_load_models": True,
            },
        }
        
        # 当前配置
        self.config = self.load_config()
        self._initialized = True
    
    def load_config(self) -> Dict:
        """从文件加载配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                return self._merge_config(self.default_config, user_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
                return self.default_config.copy()
        else:
            self._ensure_config_dir()
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: Dict = None) -> bool:
        """保存配置到文件"""
        if config is None:
            config = self.config
        
        self._ensure_config_dir()
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
    
    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的键名
        例如: config.get("audio.enable_denoise")
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值，支持点号分隔的键名
        例如: config.set("audio.enable_denoise", False)
        """
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
        return self.save_config()
    
    def get_all(self) -> Dict:
        """获取所有配置"""
        return self.config.copy()
    
    def reset_to_default(self) -> Dict:
        """重置为默认配置"""
        self.config = self.default_config.copy()
        self.save_config()
        return self.config