import time
from collections import deque

class SubtitleTimeline:
    """管理字幕时间轴，实现自检对齐"""
    
    def __init__(self):
        self.subtitles = []      # [{start, end, jp, zh}]
        self.current_index = -1
        self.alignment_history = deque(maxlen=10)  # 记录最近的对齐偏差
        
    def load_subtitles(self, asr_results, translations):
        """加载ASR和翻译结果"""
        self.subtitles = []
        for i, (asr_seg, zh_text) in enumerate(zip(asr_results, translations)):
            self.subtitles.append({
                "index": i,
                "start": asr_seg["start"],
                "end": asr_seg["end"],
                "jp": asr_seg["text"],
                "zh": zh_text,
                "duration": asr_seg["end"] - asr_seg["start"]
            })

        # 调试输出
        print(f"DEBUG: load_subtitles called with {len(asr_results)} ASR results and {len(translations)} translations")
        if self.subtitles:
            print(f"DEBUG: First subtitle after load: {self.subtitles[0]}")
    
    def get_subtitle_at_time(self, playback_time):
        """根据播放时间获取字幕"""
        for i, sub in enumerate(self.subtitles):
            if sub["start"] <= playback_time <= sub["end"]:
                self.current_index = i
                return sub
        return None
    
    def get_next_subtitle(self):
        """获取下一句"""
        if self.current_index + 1 < len(self.subtitles):
            return self.subtitles[self.current_index + 1]
        return None
    
    def get_prev_subtitle(self):
        """获取上一句"""
        if self.current_index - 1 >= 0:
            return self.subtitles[self.current_index - 1]
        return None
    
    def align_check(self, current_time, audio_feature_callback=None):
        """
        自检对齐：检测当前播放位置是否偏离预期
        返回：是否需要重新对齐
        """
        if self.current_index < 0:
            return False
        
        current_sub = self.subtitles[self.current_index]
        expected_time = current_sub["start"]
        deviation = current_time - expected_time
        
        self.alignment_history.append(deviation)
        
        # 检测连续偏离超过0.3秒
        if len(self.alignment_history) >= 3:
            avg_deviation = sum(self.alignment_history) / len(self.alignment_history)
            if abs(avg_deviation) > 0.3:
                # 需要重新对齐
                return True
        
        return False
    
    def realign(self, current_time):
        """重新对齐：找到最接近的字幕"""
        min_diff = float('inf')
        best_index = -1
        
        for i, sub in enumerate(self.subtitles):
            # 找最接近开始时间的
            diff = abs(current_time - sub["start"])
            if diff < min_diff and diff < 1.0:  # 1秒内
                min_diff = diff
                best_index = i
        
        if best_index >= 0:
            self.current_index = best_index
            return self.subtitles[best_index]
        
        return None
    
    def jump_to_subtitle(self, index):
        """跳转到指定字幕"""
        if 0 <= index < len(self.subtitles):
            self.current_index = index
            return self.subtitles[index]
        return None