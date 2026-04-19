import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.effects import normalize
import noisereduce as nr  # 降噪库
import shutil

class AudioPipeline:
    """音频预处理：降噪 + 静音检测分段"""
    
    def __init__(self, enable_denoise=True):
        self.enable_denoise = enable_denoise
        # 超过该时长自动跳过降噪，避免长音频预处理过慢
        self.max_denoise_duration_sec = 600
    
    def process(self, audio_path):
        """
        完整流程：
        1. 加载音频
        2. 降噪（可选）
        3. 检测静音分段（用于句子边界）
        4. 返回处理后的音频和分段信息
        """
        self._ensure_ffmpeg()
        try:
            audio = AudioSegment.from_file(audio_path)
        except FileNotFoundError as e:
            raise RuntimeError("读取音频失败：系统找不到音频文件或 ffmpeg/ffprobe。") from e
        
        duration_sec = len(audio) / 1000.0

        # 降噪（长音频自动跳过，避免预处理卡顿）
        if self.enable_denoise and duration_sec <= self.max_denoise_duration_sec:
            audio = self._reduce_noise(audio)
        
        # 归一化音量
        audio = normalize(audio)
        
        # 检测句子边界（静音分段）
        segments = self._detect_speech_segments(audio)
        
        return audio, segments

    def _ensure_ffmpeg(self):
        """pydub 读取多数格式依赖 ffmpeg/ffprobe"""
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if not ffmpeg_path or not ffprobe_path:
            raise RuntimeError("未找到 ffmpeg/ffprobe。请安装 ffmpeg 并加入 PATH 后重试。")
    
    def _reduce_noise(self, audio):
        """
        降噪策略：
        - 取前2秒作为噪声样本（ASMR通常开头有空白）
        - 使用谱减法降噪
        """
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # 转为float32
        samples_float = samples.astype(np.float32) / 32768.0
        
        # 取前2秒作为噪声样本
        noise_sample_len = min(int(2 * sample_rate), len(samples_float))
        noise_sample = samples_float[:noise_sample_len]
        
        # 应用降噪
        reduced = nr.reduce_noise(
            y=samples_float,
            sr=sample_rate,
            y_noise=noise_sample,
            prop_decrease=0.8,  # 降噪强度
            stationary=True      # ASMR背景音相对稳定
        )
        
        # 转回int16
        reduced_int16 = (reduced * 32768).astype(np.int16)
        return AudioSegment(
            reduced_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
    
    def _detect_speech_segments(self, audio, silence_thresh=-40, min_silence_len=500):
        """
        检测静音来分段（ASMR/广播剧的句子边界明显）
        返回：[(start_ms, end_ms), ...]
        """
        # 使用pydub的静音检测
        from pydub.silence import detect_nonsilent
        
        segments = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,  # 最小静音长度(ms)
            silence_thresh=silence_thresh      # 静音阈值(dB)
        )
        
        # 合并过短的片段（<0.5秒）
        merged = []
        for start, end in segments:
            if end - start < 500:  # 小于0.5秒的片段
                if merged:
                    # 合并到上一个片段
                    merged[-1] = (merged[-1][0], end)
                continue
            merged.append((start, end))
        
        return merged