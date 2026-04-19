from pathlib import Path
import shutil
import threading
import queue
import time

import whisper

class ASREngine:
    """支持多模型切换的ASR封装"""

    @classmethod
    def discover_models(cls):
        """扫描 models/asr 中已下载模型，返回可选模型名列表"""
        model_root = Path("./models/asr")
        if not model_root.exists():
            return []

        models = []
        for model_file in model_root.glob("*.pt"):
            stem = model_file.stem
            if stem:
                models.append(f"whisper-{stem}")

        # faster-whisper 本地目录（包含 model.bin）
        for item in model_root.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith(".") or item.name.startswith("models--"):
                continue
            if (item / "model.bin").exists():
                models.append(item.name)

        models.sort()
        return models

    def __init__(self, model_name=None):
        available = self.discover_models()
        if model_name is None:
            model_name = available[0] if available else "whisper-base"

        self.model_name = model_name
        self.model = None
        self.device = "cuda" if self._has_cuda() else "cpu"

    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _ensure_ffmpeg(self):
        """Whisper 依赖 ffmpeg，可提前给出清晰报错"""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError(
                "未找到 ffmpeg。请安装 ffmpeg 并加入 PATH 后重试。"
            )
    
    def _load_model(self):
        """懒加载，首次调用时加载"""
        if self.model is None:
            if self.model_name.startswith("whisper-"):
                model_size = self.model_name.replace("whisper-", "", 1)
                try:
                    self.model = whisper.load_model(
                        model_size,
                        device=self.device,
                        download_root="./models/asr/"
                    )
                except Exception:
                    self.device = "cpu"
                    self.model = whisper.load_model(
                        model_size,
                        device="cpu",
                        download_root="./models/asr/"
                    )
            else:
                # faster-whisper 目录模型
                from faster_whisper import WhisperModel
                model_dir = Path("./models/asr") / self.model_name
                if not model_dir.exists():
                    raise RuntimeError(f"未找到 faster-whisper 模型目录: {model_dir}")
                try:
                    if self.device == "cuda":
                        self.model = WhisperModel(
                            str(model_dir),
                            device="cuda",
                            compute_type="float16"
                        )
                    else:
                        self.model = WhisperModel(
                            str(model_dir),
                            device="cpu",
                            compute_type="int8"
                        )
                except Exception:
                    self.device = "cpu"
                    self.model = WhisperModel(
                        str(model_dir),
                        device="cpu",
                        compute_type="int8"
                    )
    
    def _ensure_model(self):
        """确保模型已加载"""
        if self.model is None:
            self._load_model()
    
    def transcribe(self, audio_path, segments=None):
        self._ensure_ffmpeg()
        self._ensure_model()
        """
        转录音频
        segments: 预分段的边界（可选），如果不提供则自动检测
        """
        self._load_model()

        # 直接整体识别，避免按片段反复调用 ffmpeg 导致极慢和窗口闪烁
        if self.model_name.startswith("whisper-"):
            result = self.model.transcribe(
                audio_path,
                language="ja",
                task="transcribe",
                verbose=False
            )
            return [{
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            } for seg in result["segments"]]

        segments, _ = self.model.transcribe(
            audio_path,
            language="ja",
            beam_size=1,
            vad_filter=True
        )
        return [{
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        } for seg in segments]
    
    def _extract_segment(self, audio_path, start_ms, end_ms):
        """提取音频片段（临时文件）"""
        from pydub import AudioSegment
        import tempfile

        audio = AudioSegment.from_file(audio_path)
        segment = audio[start_ms:end_ms]

        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        segment.export(temp.name, format="wav")
        return temp.name

    def transcribe_streaming(self, audio_path, segments=None, callback=None, chunk_size=3):
        """
        流式转录，识别到一定数量的句子后就通过回调返回

        Args:
            audio_path: 音频文件路径
            segments: 预分段的边界（可选）
            callback: 回调函数 callback(results_chunk, current_index, total_estimated)
            chunk_size: 每识别多少句后触发一次回调

        Returns:
            完整的转录结果列表
        """
        self._ensure_ffmpeg()
        self._ensure_model()

        # 用于 faster-whisper 的流式处理
        if not self.model_name.startswith("whisper-"):
            return self._transcribe_faster_whisper_streaming(
                audio_path, segments, callback, chunk_size
            )

        # 标准 Whisper 暂不支持真正的流式，回退到普通模式
        results = self.transcribe(audio_path, segments)
        if callback:
            callback(results, len(results), len(results))
        return results

    def _transcribe_faster_whisper_streaming(self, audio_path, segments, callback, chunk_size):
        """faster-whisper 的流式处理"""
        results = []
        count = 0

        # faster-whisper 的 transcribe 返回生成器
        segments_gen, info = self.model.transcribe(
            audio_path,
            language="ja",
            beam_size=1,
            vad_filter=True,
            word_timestamps=False  # 不需要词级时间戳，加快速度
        )

        for seg in segments_gen:
            result = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            }
            results.append(result)
            count += 1

            # 每识别到 chunk_size 句就触发回调
            if callback and count % chunk_size == 0:
                # 估算总数（基于已识别的时长比例）
                estimated_total = len(results)  # 简化处理
                callback(list(results), len(results), estimated_total)

        # 最后再回调一次，确保所有结果都被处理
        if callback and results:
            callback(results, len(results), len(results))

        return results

    def transcribe_with_queue(self, audio_path, segments=None, result_queue=None, status_callback=None):
        """
        将转录结果放入队列，供翻译线程消费（生产者-消费者模式）

        Args:
            audio_path: 音频文件路径
            segments: 预分段的边界（可选）
            result_queue: 结果队列
            status_callback: 状态回调 callback(status_message)

        Returns:
            完整的转录结果列表
        """
        if result_queue is None:
            result_queue = queue.Queue()

        self._ensure_ffmpeg()
        self._ensure_model()

        results = []

        if status_callback:
            status_callback("开始语音识别...")

        if not self.model_name.startswith("whisper-"):
            # faster-whisper 支持流式
            segments_gen, info = self.model.transcribe(
                audio_path,
                language="ja",
                beam_size=1,
                vad_filter=True
            )

            for seg in segments_gen:
                result = {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                }
                results.append(result)
                # 将结果放入队列，每识别一句就放一次
                result_queue.put(("asr", result))

                if status_callback and len(results) % 5 == 0:
                    status_callback(f"已识别 {len(results)} 句...")
        else:
            # 标准 Whisper 一次性识别
            if status_callback:
                status_callback("正在进行语音识别（Whisper模式）...")

            transcribe_result = self.model.transcribe(
                audio_path,
                language="ja",
                task="transcribe",
                verbose=False
            )

            for seg in transcribe_result["segments"]:
                result = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }
                results.append(result)
                result_queue.put(("asr", result))

        # 放入结束标记
        result_queue.put(("asr_done", None))

        if status_callback:
            status_callback(f"语音识别完成，共 {len(results)} 句")

        return results