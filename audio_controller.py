import pygame
import threading
import time

class AudioController:
    """音频播放控制"""
    
    def __init__(self, callback_update=None):
        """
        callback_update: 每帧回调，用于更新UI进度条
        """
        pygame.mixer.init()
        self.callback = callback_update
        self.current_file = None
        self.is_playing = False
        self.pause_pos = 0
        self.audio_length = 0
        self.thread = None
    
    def load(self, audio_path):
        """加载音频"""
        pygame.mixer.music.load(audio_path)
        self.current_file = audio_path
        # 获取时长（秒）
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        self.audio_length = len(audio) / 1000.0
    
    def play(self):
        """播放"""
        if self.pause_pos > 0:
            pygame.mixer.music.play(start=self.pause_pos)
        else:
            pygame.mixer.music.play()
        self.is_playing = True
        
        # 启动进度更新线程
        self.thread = threading.Thread(target=self._update_progress, daemon=True)
        self.thread.start()
    
    def pause(self):
        """暂停"""
        pygame.mixer.music.pause()
        self.is_playing = False
        self.pause_pos = self.get_current_pos()
    
    def stop(self):
        """停止"""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.pause_pos = 0
    
    def seek(self, position):
        """跳转到指定位置（秒）"""
        self.pause_pos = position
        if self.is_playing:
            pygame.mixer.music.play(start=position)
    
    def get_current_pos(self):
        """获取当前播放位置"""
        if self.is_playing:
            return pygame.mixer.music.get_pos() / 1000.0 + self.pause_pos
        return self.pause_pos
    
    def _update_progress(self):
        """更新进度回调"""
        while self.is_playing:
            if self.callback:
                self.callback(self.get_current_pos())
            time.sleep(0.1)  # 100ms更新一次，足够流畅