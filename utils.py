"""
工具函数模块
"""

import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import sys


# ==================== 文件工具 ====================

def get_file_hash(filepath: str) -> str:
    """计算文件MD5"""
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_temp_file(suffix: str = ".wav") -> str:
    """创建临时文件"""
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name


def cleanup_temp_file(filepath: str):
    """删除临时文件"""
    try:
        Path(filepath).unlink()
    except:
        pass


def ensure_dir(directory: str) -> Path:
    """确保目录存在"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==================== 时间工具 ====================

def seconds_to_time_str(seconds: float) -> str:
    """将秒数转换为 mm:ss 格式"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def time_str_to_seconds(time_str: str) -> float:
    """将 mm:ss 格式转换为秒数"""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(time_str)


# ==================== 文本工具 ====================

def split_sentences(text: str, max_length: int = 100) -> List[str]:
    """将长文本按标点分割"""
    if len(text) <= max_length:
        return [text]
    
    delimiters = ['。', '！', '？', '；', '、', '，', '.', '!', '?', ';', ',']
    
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in delimiters and len(current) >= max_length * 0.5:
            sentences.append(current.strip())
            current = ""
    
    if current:
        sentences.append(current.strip())
    
    return sentences if sentences else [text]


def add_brackets(text: str) -> str:
    """添加方括号"""
    return f"[{text}]"


def remove_brackets(text: str) -> str:
    """移除方括号"""
    return text.strip('[]')


# ==================== 音频工具 ====================

def get_audio_duration(filepath: str) -> Optional[float]:
    """获取音频时长（秒）"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(filepath)
        return len(audio) / 1000.0
    except:
        return None


def is_audio_file(filepath: str) -> bool:
    """检查是否为音频文件"""
    suffix = Path(filepath).suffix.lower()
    return suffix in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']


# ==================== 系统工具 ====================

def check_ffmpeg() -> bool:
    """检查ffmpeg是否安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def get_system_info() -> Dict:
    """获取系统信息"""
    import platform
    import psutil
    
    return {
        "os": platform.system(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "has_ffmpeg": check_ffmpeg()
    }


# ==================== 字幕导出 ====================

def export_to_srt(subtitles: List[Dict], filepath: str):
    """
    导出为SRT字幕文件
    
    Args:
        subtitles: [{"start": 0.5, "end": 2.3, "jp": "...", "zh": "..."}]
        filepath: 输出路径
    """
    def _to_srt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{_to_srt_time(sub['start'])} --> {_to_srt_time(sub['end'])}\n")
            f.write(f"{sub['jp']}\n")
            f.write(f"{sub['zh']}\n\n")