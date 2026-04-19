import sys
import threading
import os
import subprocess
import time
import queue
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from audio_pipeline import AudioPipeline
from asr_engine import ASREngine
from translator_engine import TranslatorEngine
from subtitle_timeline import SubtitleTimeline
from audio_controller import AudioController
from config import Config


def _hide_child_console_on_windows():
    """在 Windows 下隐藏 ffmpeg 等子进程弹出的控制台窗口。"""
    if os.name != "nt":
        return
    if getattr(subprocess, "_kotonetrans_patched", False):
        return

    original_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        creationflags = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = creationflags | 0x08000000  # CREATE_NO_WINDOW
        return original_popen(*args, **kwargs)

    subprocess.Popen = patched_popen
    subprocess._kotonetrans_patched = True


_hide_child_console_on_windows()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("日语ASMR/广播剧翻译工具")
        self.setMinimumSize(900, 500)
        
        # 初始化组件（但不加载模型）
        self.audio_pipeline = AudioPipeline(enable_denoise=True)
        self.asr_engine = None  # 改：不要直接创建，设为None
        self.translator = None  # 改：不要直接创建，设为None
        self.timeline = SubtitleTimeline()
        self.audio_ctrl = AudioController(callback_update=self.on_progress_update)
        
        self.current_audio_path = None
        self.config = Config()
        self.last_display_index = None
        self.setup_ui()
        
        # 添加状态提示
        self.status_bar.showMessage("就绪，先选音频与模型，再点击开始翻译")
    
    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f1115; }
            QLabel { color: #e6eaf2; font-size: 13px; }
            QPushButton {
                background-color: #2d6cdf;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #3a78e8; }
            QPushButton:pressed { background-color: #2157be; }
            QComboBox, QLineEdit, QPlainTextEdit {
                background-color: #171a21;
                border: 1px solid #2b3240;
                border-radius: 8px;
                color: #e6eaf2;
                padding: 6px 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #171a21;
                color: #e6eaf2;
                border: 1px solid #2b3240;
                selection-background-color: #2d6cdf;
                selection-color: #ffffff;
                outline: 0;
            }
            QGroupBox {
                border: 1px solid #2b3240;
                border-radius: 10px;
                margin-top: 8px;
                padding-top: 10px;
                color: #b9c1d0;
            }
            QStatusBar { background-color: #11141b; color: #b9c1d0; }
            QSlider::groove:horizontal {
                height: 8px;
                background: #1f2430;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -4px 0;
                background: #4b8dff;
                border-radius: 8px;
            }
        """)

        central = QWidget()
        layout = QVBoxLayout()
        
        # 1. 文件选择区域
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.select_btn = QPushButton("选择音频文件")
        self.select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_btn)
        layout.addLayout(file_layout)
        
        # 2. 模型选择
        model_layout = QGridLayout()
        model_layout.addWidget(QLabel("ASR模型:"), 0, 0)
        self.asr_combo = QComboBox()
        model_layout.addWidget(self.asr_combo, 0, 1)

        model_layout.addWidget(QLabel("翻译模型:"), 1, 0)
        self.trans_combo = QComboBox()
        self.trans_combo.currentTextChanged.connect(self.on_translate_model_changed)
        model_layout.addWidget(self.trans_combo, 1, 1)

        # 翻译模式选择
        model_layout.addWidget(QLabel("翻译方式:"), 0, 2)
        self.translate_mode_combo = QComboBox()
        self.translate_mode_combo.addItems(["批量翻译（推荐）", "并发翻译", "逐句翻译（原版）"])
        self.translate_mode_combo.setCurrentIndex(0)
        model_layout.addWidget(self.translate_mode_combo, 0, 3)

        # 流式处理选项
        self.streaming_checkbox = QCheckBox("启用流式处理（推荐）")
        self.streaming_checkbox.setChecked(True)
        self.streaming_checkbox.setStyleSheet("color: #b9c1d0;")
        self.streaming_checkbox.setToolTip("ASR识别的同时进行翻译，大幅提升速度")
        model_layout.addWidget(self.streaming_checkbox, 1, 2, 1, 2)

        self.apply_model_btn = QPushButton("应用模型")
        self.apply_model_btn.clicked.connect(self.change_model)
        model_layout.addWidget(self.apply_model_btn, 1, 2, 1, 1)
        self.quick_toggle_btn = QPushButton("一键切换翻译模式")
        self.quick_toggle_btn.clicked.connect(self.quick_toggle_translate_mode)
        model_layout.addWidget(self.quick_toggle_btn, 1, 3, 1, 1)
        layout.addLayout(model_layout)

        # 2.1 API 翻译配置（仅 api-http 显示）
        self.api_group = QGroupBox("翻译 API 设置（仅 api-http）")
        api_form = QFormLayout()
        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText("例如 http://127.0.0.1:8000/translate")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("可选：Bearer Token")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_provider_combo = QComboBox()
        self.api_provider_combo.addItems(TranslatorEngine.API_PRESETS.keys())
        self.api_provider_combo.currentTextChanged.connect(self.on_api_provider_changed)
        self.api_model_edit = QLineEdit()
        self.api_model_edit.setPlaceholderText("模型名，例如 deepseek-chat")
        self.api_check_btn = QPushButton("API 预检")
        self.api_check_btn.clicked.connect(self.check_api_now)
        self.api_prompt_edit = QPlainTextEdit()
        self.api_prompt_edit.setPlaceholderText("系统提示词（可选）")
        self.api_prompt_edit.setMaximumHeight(72)
        saved_provider = self.config.get("translate.api_provider", "DeepSeek")
        if saved_provider in TranslatorEngine.API_PRESETS:
            self.api_provider_combo.setCurrentText(saved_provider)
        self.api_url_edit.setText(self.config.get("translate.api_url", "") or "")
        self.api_key_edit.setText(self.config.get("translate.api_key", "") or "")
        self.api_model_edit.setText(
            self.config.get("translate.api_model", "deepseek-chat") or "deepseek-chat"
        )
        self.api_prompt_edit.setPlainText(
            self.config.get(
                "translate.api_system_prompt",
                "你是一名日本二次元广播剧和音声翻译者。保持口语自然，忠实原意，必要时保留语气词。仅输出中文翻译，不要解释。"
            ) or ""
        )
        api_form.addRow("服务商:", self.api_provider_combo)
        api_form.addRow("URL:", self.api_url_edit)
        api_form.addRow("Key:", self.api_key_edit)
        api_form.addRow("模型:", self.api_model_edit)
        api_form.addRow("", self.api_check_btn)
        api_form.addRow("提示词:", self.api_prompt_edit)
        self.api_group.setLayout(api_form)
        layout.addWidget(self.api_group)
        self.on_api_provider_changed(self.api_provider_combo.currentText())
        
        # 3. 进度条区域
        progress_layout = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderMoved.connect(self.on_seek)
        progress_layout.addWidget(self.time_label)
        progress_layout.addWidget(self.progress_slider)
        layout.addLayout(progress_layout)
        
        # 4. 控制按钮
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.start_btn = QPushButton("开始翻译")
        self.start_btn.clicked.connect(self.start_translate)
        self.prev_btn = QPushButton("上一句")
        self.prev_btn.clicked.connect(self.prev_subtitle)
        self.next_btn = QPushButton("下一句")
        self.next_btn.clicked.connect(self.next_subtitle)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        layout.addLayout(control_layout)
        
        # 5. 字幕显示区域（全量可滚动列表）
        self.subtitle_frame = QFrame()
        self.subtitle_frame.setObjectName("subtitleFrame")
        self.subtitle_frame.setStyleSheet("""
            #subtitleFrame {
                background-color: #161a22;
                border: 1px solid #2a3345;
                border-radius: 14px;
            }
        """)
        subtitle_layout = QVBoxLayout()
        subtitle_layout.setSpacing(8)
        subtitle_layout.setContentsMargins(14, 14, 14, 14)

        self.subtitle_title = QLabel("字幕总览（滚轮上下查看，不影响音频进度）")
        self.subtitle_title.setStyleSheet("color:#8cb4ff; font-weight:700;")
        self.subtitle_table = QTableWidget(0, 3)
        self.subtitle_table.setHorizontalHeaderLabels(["#", "日语", "中文"])
        self.subtitle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subtitle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subtitle_table.setAlternatingRowColors(True)
        self.subtitle_table.verticalHeader().setVisible(False)
        self.subtitle_table.horizontalHeader().setStretchLastSection(True)
        self.subtitle_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.subtitle_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.subtitle_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.subtitle_table.setStyleSheet("""
            QTableWidget {
                background-color: #111723;
                border: 1px solid #273249;
                border-radius: 10px;
                gridline-color: #1f2a3a;
                alternate-background-color: #0f1520;
                color: #e6eaf2;
            }
            QHeaderView::section {
                background-color: #172033;
                color: #b9c1d0;
                border: none;
                padding: 6px;
                font-weight: 700;
            }
            QTableWidget::item:selected {
                background-color: #2d6cdf;
                color: #ffffff;
            }
        """)

        subtitle_layout.addWidget(self.subtitle_title)
        subtitle_layout.addWidget(self.subtitle_table)
        self.subtitle_frame.setLayout(subtitle_layout)
        layout.addWidget(self.subtitle_frame)
        
        # 6. 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.refresh_model_lists()

    def refresh_model_lists(self):
        """扫描本地模型目录并刷新下拉列表"""
        asr_models = ASREngine.discover_models()
        trans_models = TranslatorEngine.discover_models()
        current_asr = self.asr_combo.currentText().strip()
        current_trans = self.trans_combo.currentText().strip()

        self.asr_combo.clear()
        self.trans_combo.clear()

        self.asr_combo.addItems(asr_models)
        self.trans_combo.addItems(trans_models)

        if current_asr and current_asr in asr_models:
            self.asr_combo.setCurrentText(current_asr)
        if current_trans and current_trans in trans_models:
            self.trans_combo.setCurrentText(current_trans)

        has_model = bool(asr_models and trans_models)
        self.apply_model_btn.setEnabled(has_model)

        if not has_model:
            self.status_bar.showMessage("未检测到完整模型，请检查 models/asr 和 models/translate", 4000)

        self.on_translate_model_changed(self.trans_combo.currentText())

    def on_translate_model_changed(self, model_name: str):
        """根据翻译模型选择，显示/隐藏 API 配置区域"""
        use_api = (model_name or "").strip() == TranslatorEngine.API_MODEL_NAME
        self.api_group.setVisible(use_api)

    def on_api_provider_changed(self, provider_name: str):
        preset = TranslatorEngine.API_PRESETS.get(provider_name, TranslatorEngine.API_PRESETS["自定义"])
        if provider_name != "自定义":
            if preset.get("base_url"):
                self.api_url_edit.setText(preset["base_url"])
            if preset.get("model"):
                self.api_model_edit.setText(preset["model"])

    def check_api_now(self):
        """手动执行 API 预检，不触发完整处理流程"""
        api_url = self.api_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        api_model = self.api_model_edit.text().strip()
        api_prompt = self.api_prompt_edit.toPlainText().strip()
        if not api_url:
            self.status_bar.showMessage("API URL 为空", 4000)
            return
        if not api_key:
            self.status_bar.showMessage("API Key 为空", 4000)
            return

        checker = TranslatorEngine(
            model_name=TranslatorEngine.API_MODEL_NAME,
            api_url=api_url,
            api_key=api_key,
            api_system_prompt=api_prompt,
            api_model=api_model
        )
        self.status_bar.showMessage("正在进行 API 预检...")
        ok, msg = checker.validate_api()
        if ok:
            self.status_bar.showMessage("API 预检通过", 3000)
        else:
            self.status_bar.showMessage(msg, 5000)

    def quick_toggle_translate_mode(self):
        """在本地翻译模型和 API 翻译之间一键切换并自动应用"""
        all_models = [self.trans_combo.itemText(i) for i in range(self.trans_combo.count())]
        if not all_models:
            self.status_bar.showMessage("未检测到翻译模型，无法切换", 3000)
            return

        current = self.trans_combo.currentText().strip()
        api_name = TranslatorEngine.API_MODEL_NAME
        local_candidates = [m for m in all_models if m != api_name]
        if not local_candidates:
            self.status_bar.showMessage("未检测到本地翻译模型，无法切换", 3000)
            return

        if current == api_name:
            target = local_candidates[0]
        else:
            target = api_name if api_name in all_models else local_candidates[0]

        self.trans_combo.setCurrentText(target)
        self.change_model()
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", "", "音频文件 (*.mp3 *.wav *.m4a *.flac)"
        )
        if file_path:
            self.current_audio_path = file_path
            self.file_label.setText(f"已选择: {file_path.split('/')[-1]}")
            self.status_bar.showMessage("文件已选择，点击“开始翻译”执行处理", 3000)

    def start_translate(self):
        """手动开始翻译流程"""
        if not self.current_audio_path:
            self.status_bar.showMessage("请先选择音频文件", 4000)
            return
        self.last_display_index = None
        self.subtitle_table.setRowCount(0)
        if self.load_models():
            self.process_audio()

    def load_models(self):
        """延迟加载模型"""
        asr_name = self.asr_combo.currentText().strip()
        trans_name = self.trans_combo.currentText().strip()

        if not asr_name or not trans_name:
            self.status_bar.showMessage("模型目录为空，无法加载模型", 4000)
            return False

        api_url = self.api_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        api_model = self.api_model_edit.text().strip()
        api_prompt = self.api_prompt_edit.toPlainText().strip()
        if trans_name == TranslatorEngine.API_MODEL_NAME:
            if not api_url:
                self.status_bar.showMessage("请选择 api-http 后需要填写 URL", 4000)
                return False
            self.config.set("translate.api_url", api_url)
            self.config.set("translate.api_key", api_key)
            self.config.set("translate.api_model", api_model)
            self.config.set("translate.api_provider", self.api_provider_combo.currentText())
            self.config.set("translate.api_system_prompt", api_prompt)

        if self.asr_engine is None:
            self.status_bar.showMessage("正在加载ASR模型...")
            self.asr_engine = ASREngine(asr_name)
        
        if self.translator is None:
            self.status_bar.showMessage("正在加载翻译模型...")
            self.translator = TranslatorEngine(
                trans_name,
                api_url=api_url,
                api_key=api_key,
                api_system_prompt=api_prompt,
                api_model=api_model
            )

        if trans_name == TranslatorEngine.API_MODEL_NAME:
            self.status_bar.showMessage("正在预检翻译API...")
            ok, msg = self.translator.validate_api()
            if not ok:
                self.status_bar.showMessage(msg, 5000)
                return False
        
        self.status_bar.showMessage("模型加载完成", 2000)
        return True
    
    def process_audio(self):
        """处理音频：降噪 → ASR → 翻译（支持流式并行）"""
        if self.asr_engine is None or self.translator is None:
            self.status_bar.showMessage("模型尚未加载，请先检查模型目录", 4000)
            return

        self.status_bar.showMessage("正在处理音频...")

        def set_status(text, timeout=0):
            QMetaObject.invokeMethod(
                self.status_bar,
                "showMessage",
                Qt.QueuedConnection,
                Q_ARG(str, text),
                Q_ARG(int, int(timeout))
            )

        def process():
            try:
                # 1. 音频预处理
                set_status("正在预处理音频...")
                processed_audio, segments = self.audio_pipeline.process(self.current_audio_path)

                # 检查是否使用流式处理
                use_streaming = self.streaming_checkbox.isChecked()

                if use_streaming:
                    # 流式并行处理：ASR 和翻译同时进行
                    self._process_streaming(set_status, segments)
                else:
                    # 传统串行处理
                    self._process_sequential(set_status, segments)

            except Exception as e:
                import traceback
                traceback.print_exc()
                set_status(f"处理失败: {e}", 6000)

        threading.Thread(target=process, daemon=True).start()

    def _process_streaming(self, set_status, segments):
        """流式并行处理：ASR识别的同时进行翻译"""
        result_queue = queue.Queue()

        # 使用可变对象存储结果，确保跨线程可见
        shared_data = {
            "asr_results": [],
            "translations": [],
            "asr_done": False,
            "trans_done": False
        }
        asr_lock = threading.Lock()
        trans_batch_size = 5  # 每累积5句翻译一次，保证上下文充足

        # 状态更新
        asr_count = {"value": 0}
        trans_count = {"value": 0}

        def update_status():
            if not shared_data["asr_done"]:
                set_status(f"ASR识别中: {asr_count['value']}句 | 翻译中: {trans_count['value']}句")
            else:
                with asr_lock:
                    total = len(shared_data["asr_results"])
                set_status(f"ASR完成 | 翻译中: {trans_count['value']}/{total}句")

        # 定义翻译辅助函数（在使用前定义）
        def _translate_batch_with_context(all_texts, batch_indices):
            """批量翻译，保持上下文"""
            results = {}
            for idx in batch_indices:
                results[idx] = self.translator.translate_with_context(all_texts, idx)
            return [results[i] for i in batch_indices]

        def _translate_concurrent_with_context(all_texts, batch_indices):
            """并发翻译，保持上下文"""
            from concurrent.futures import ThreadPoolExecutor

            results = {}

            def translate_one(idx):
                results[idx] = self.translator.translate_with_context(all_texts, idx)

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(translate_one, batch_indices))

            return [results[i] for i in batch_indices]

        # ASR 识别线程（生产者）
        def asr_worker():
            try:
                # 不使用 transcribe_with_queue 的返回值，完全依赖队列
                self.asr_engine.transcribe_with_queue(
                    self.current_audio_path,
                    segments,
                    result_queue=result_queue,
                    status_callback=lambda msg: set_status(msg)
                )
                shared_data["asr_done"] = True
                update_status()
            except Exception as e:
                import traceback
                traceback.print_exc()
                result_queue.put(("error", str(e)))

        # 翻译线程（消费者）
        def translator_worker():
            pending_translations = {}  # {index: translation}
            pending_asr = []  # 待翻译的ASR结果 [(index, result), ...]

            while True:
                try:
                    item = result_queue.get(timeout=1)
                    item_type, data = item

                    if item_type == "error":
                        set_status(f"错误: {data}", 5000)
                        break

                    if item_type == "asr":
                        # 收集ASR结果
                        idx = len(shared_data["asr_results"])
                        with asr_lock:
                            shared_data["asr_results"].append(data)
                        asr_count["value"] = idx + 1
                        pending_asr.append((idx, data))

                        # 当累积到一定数量或ASR完成时，批量翻译
                        if len(pending_asr) >= trans_batch_size:
                            # 批量翻译
                            batch_indices = [i for i, _ in pending_asr]

                            # 获取完整上下文
                            with asr_lock:
                                all_texts = [r["text"] for r in shared_data["asr_results"]]

                            # 根据翻译模式选择方法
                            translate_mode = self.translate_mode_combo.currentIndex()
                            if translate_mode == 0:  # 批量翻译
                                batch_trans = _translate_batch_with_context(all_texts, batch_indices)
                            elif translate_mode == 1:  # 并发翻译
                                batch_trans = _translate_concurrent_with_context(all_texts, batch_indices)
                            else:  # 逐句翻译
                                batch_trans = [
                                    self.translator.translate_with_context(all_texts, idx)
                                    for idx in batch_indices
                                ]

                            # 保存翻译结果
                            for idx, trans in zip(batch_indices, batch_trans):
                                pending_translations[idx] = trans
                            trans_count["value"] = len(pending_translations)
                            pending_asr.clear()
                            update_status()

                    elif item_type == "asr_done":
                        # ASR完成，处理剩余的翻译
                        shared_data["asr_done"] = True
                        if pending_asr:
                            batch_indices = [i for i, _ in pending_asr]
                            with asr_lock:
                                all_texts = [r["text"] for r in shared_data["asr_results"]]

                            translate_mode = self.translate_mode_combo.currentIndex()
                            if translate_mode == 0:
                                batch_trans = _translate_batch_with_context(all_texts, batch_indices)
                            elif translate_mode == 1:
                                batch_trans = _translate_concurrent_with_context(all_texts, batch_indices)
                            else:
                                batch_trans = [
                                    self.translator.translate_with_context(all_texts, idx)
                                    for idx in batch_indices
                                ]

                            for idx, trans in zip(batch_indices, batch_trans):
                                pending_translations[idx] = trans
                            trans_count["value"] = len(pending_translations)
                            pending_asr.clear()
                            update_status()
                        break

                except queue.Empty:
                    # 队列空了，检查ASR是否完成
                    if shared_data["asr_done"]:
                        break
                    continue

            # 确保所有翻译都按顺序排列
            with asr_lock:
                total = len(shared_data["asr_results"])
            translations_list = [pending_translations.get(i, "（缺失）") for i in range(total)]
            # 直接修改共享对象中的列表
            with asr_lock:
                shared_data["translations"] = translations_list
            shared_data["trans_done"] = True
            update_status()

        # 启动并行线程
        asr_thread = threading.Thread(target=asr_worker, daemon=True)
        trans_thread = threading.Thread(target=translator_worker, daemon=True)

        asr_thread.start()
        trans_thread.start()

        # 等待完成
        asr_thread.join()
        trans_thread.join()

        # 从共享对象中获取最终结果
        asr_results = shared_data["asr_results"]
        translations = shared_data["translations"]

        # 调试输出
        print(f"DEBUG: asr_results count = {len(asr_results)}")
        print(f"DEBUG: translations count = {len(translations)}")
        if asr_results:
            print(f"DEBUG: Sample ASR: {asr_results[0]}")
        if translations:
            print(f"DEBUG: Sample Trans: {translations[0]}")
        print(f"DEBUG: timeline.subtitles before load = {len(self.timeline.subtitles)}")

        # 检查翻译是否包含错误信息
        failed_trans = sum(1 for t in translations if t.startswith("（") and t.endswith("）"))
        if failed_trans > 0:
            print(f"WARNING: {failed_trans}/{len(translations)} translations may have failed!")
            print(f"DEBUG: Sample translations: {translations[:3]}")
        else:
            print(f"DEBUG: All translations look OK")

        # 先加载到时间轴（在子线程中执行）
        self.timeline.load_subtitles(asr_results, translations)

        print(f"DEBUG: timeline.subtitles after load = {len(self.timeline.subtitles)}")
        if self.timeline.subtitles:
            print(f"DEBUG: First subtitle = {self.timeline.subtitles[0]}")

        # 确保UI更新在主线程中执行
        QMetaObject.invokeMethod(self, "update_first_subtitle")
        set_status("处理完成", 3000)

    def _process_sequential(self, set_status, segments):
        """传统串行处理"""
        # ASR识别
        asr_start = time.time()
        set_status(f"正在进行语音识别...（模型: {self.asr_engine.model_name}）")

        done_flag = {"done": False}

        def asr_ticker():
            while not done_flag["done"]:
                elapsed = int(time.time() - asr_start)
                set_status(f"正在进行语音识别...（已用时 {elapsed}s，模型: {self.asr_engine.model_name}）")
                time.sleep(1.5)

        threading.Thread(target=asr_ticker, daemon=True).start()
        asr_results = self.asr_engine.transcribe(self.current_audio_path, segments)
        done_flag["done"] = True

        # 翻译
        jp_texts = [seg["text"] for seg in asr_results]

        def progress_callback(current, total):
            set_status(f"翻译进度: {current}/{total}")

        translate_mode = self.translate_mode_combo.currentIndex()
        if translate_mode == 0:  # 批量翻译
            translations = self.translator.translate_batch(
                jp_texts,
                batch_size=10,
                progress_callback=progress_callback
            )
        elif translate_mode == 1:  # 并发翻译
            translations = self.translator.translate_concurrent(
                jp_texts,
                max_workers=6,
                progress_callback=progress_callback
            )
        else:  # 逐句翻译
            translations = []
            for i in range(len(jp_texts)):
                zh = self.translator.translate_with_context(jp_texts, i)
                translations.append(zh)
                progress_callback(i + 1, len(jp_texts))

        # 调试输出
        print(f"DEBUG: asr_results count = {len(asr_results)}")
        print(f"DEBUG: translations count = {len(translations)}")
        if asr_results:
            print(f"DEBUG: Sample ASR: {asr_results[0]}")
        if translations:
            print(f"DEBUG: Sample Trans: {translations[0]}")
        print(f"DEBUG: timeline.subtitles before load = {len(self.timeline.subtitles)}")

        # 检查翻译是否包含错误信息
        failed_trans = sum(1 for t in translations if t.startswith("（") and t.endswith("）"))
        if failed_trans > 0:
            print(f"WARNING: {failed_trans}/{len(translations)} translations may have failed!")
            print(f"DEBUG: Sample translations: {translations[:3]}")
        else:
            print(f"DEBUG: All translations look OK")

        # 加载到时间轴
        self.timeline.load_subtitles(asr_results, translations)

        print(f"DEBUG: timeline.subtitles after load = {len(self.timeline.subtitles)}")
        if self.timeline.subtitles:
            print(f"DEBUG: First subtitle = {self.timeline.subtitles[0]}")

        # 确保UI更新在主线程中执行
        QMetaObject.invokeMethod(self, "update_first_subtitle")
        set_status("处理完成", 3000)
    
    @Slot()
    def update_first_subtitle(self):
        """处理完成后填充并定位到第一句"""
        print(f"DEBUG: update_first_subtitle called, subtitles count = {len(self.timeline.subtitles)}")

        # 加载音频到播放器（在主线程中执行）
        if self.current_audio_path:
            self.audio_ctrl.load(self.current_audio_path)
            self.progress_slider.setMaximum(int(self.audio_ctrl.audio_length))

        # 填充字幕表
        self._populate_subtitle_table()

        # 强制刷新UI
        self.subtitle_table.repaint()
        self.subtitle_table.update()

        if self.timeline.subtitles:
            print(f"DEBUG: Highlighting row 0")
            self._highlight_subtitle_row(0)
        else:
            print(f"DEBUG: No subtitles to highlight")

    def _populate_subtitle_table(self):
        subs = self.timeline.subtitles
        print(f"DEBUG: _populate_subtitle_table called with {len(subs)} subtitles")

        self.subtitle_table.setRowCount(len(subs))
        for i, sub in enumerate(subs):
            idx_item = QTableWidgetItem(str(i + 1))
            jp_item = QTableWidgetItem(sub.get("jp", ""))
            zh_item = QTableWidgetItem(sub.get("zh", ""))

            print(f"DEBUG: Row {i}: JP='{sub.get('jp', '')}', ZH='{sub.get('zh', '')}'")

            self.subtitle_table.setItem(i, 0, idx_item)
            self.subtitle_table.setItem(i, 1, jp_item)
            self.subtitle_table.setItem(i, 2, zh_item)

        print(f"DEBUG: Table row count set to {self.subtitle_table.rowCount()}")

    def _highlight_subtitle_row(self, idx):
        if idx is None or idx < 0 or idx >= self.subtitle_table.rowCount():
            return
        if idx == self.last_display_index:
            return
        self.last_display_index = idx
        self.subtitle_table.selectRow(idx)
        item = self.subtitle_table.item(idx, 0)
        if item:
            self.subtitle_table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
    
    def toggle_play(self):
        if self.audio_ctrl.is_playing:
            self.audio_ctrl.pause()
            self.play_btn.setText("播放")
        else:
            self.audio_ctrl.play()
            self.play_btn.setText("暂停")
    
    def on_progress_update(self, current_time):
        """进度更新回调"""
        # 统一转回主线程，避免音频回调线程直接操作 Qt 控件
        QMetaObject.invokeMethod(
            self,
            "_on_progress_update_ui",
            Qt.QueuedConnection,
            Q_ARG(float, float(current_time))
        )

    @Slot(float)
    def _on_progress_update_ui(self, current_time):
        self.progress_slider.setValue(int(current_time))

        current_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
        total_str = f"{int(self.audio_ctrl.audio_length//60):02d}:{int(self.audio_ctrl.audio_length%60):02d}"
        self.time_label.setText(f"{current_str} / {total_str}")

        sub = self.timeline.get_subtitle_at_time(current_time)
        if sub and sub.get("index") != self.last_display_index:
            self._highlight_subtitle_row(sub.get("index"))

        if self.timeline.align_check(current_time):
            new_sub = self.timeline.realign(current_time)
            if new_sub and new_sub.get("index") != self.last_display_index:
                self._highlight_subtitle_row(new_sub.get("index"))
    
    def on_seek(self, position):
        """拖动进度条"""
        self.audio_ctrl.seek(position)
        # 手动更新字幕
        sub = self.timeline.get_subtitle_at_time(position)
        if sub:
            self._highlight_subtitle_row(sub.get("index"))
    
    def prev_subtitle(self):
        """上一句"""
        sub = self.timeline.get_prev_subtitle()
        if sub:
            self.audio_ctrl.seek(sub["start"])
            self._highlight_subtitle_row(sub.get("index"))
    
    def next_subtitle(self):
        """下一句"""
        sub = self.timeline.get_next_subtitle()
        if sub:
            self.audio_ctrl.seek(sub["start"])
            self._highlight_subtitle_row(sub.get("index"))
    
    def change_model(self):
        """切换模型"""
        new_asr = self.asr_combo.currentText()
        new_trans = self.trans_combo.currentText()

        if not new_asr or not new_trans:
            self.status_bar.showMessage("未检测到可用模型，无法切换", 3000)
            return

        api_url = self.api_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        api_model = self.api_model_edit.text().strip()
        api_prompt = self.api_prompt_edit.toPlainText().strip()
        if new_trans == TranslatorEngine.API_MODEL_NAME:
            if not api_url:
                self.status_bar.showMessage("请选择 api-http 后需要填写 URL", 3000)
                return
            self.config.set("translate.api_url", api_url)
            self.config.set("translate.api_key", api_key)
            self.config.set("translate.api_model", api_model)
            self.config.set("translate.api_provider", self.api_provider_combo.currentText())
            self.config.set("translate.api_system_prompt", api_prompt)
        
        # 重新初始化引擎
        self.asr_engine = ASREngine(new_asr)
        self.translator = TranslatorEngine(
            new_trans,
            api_url=api_url,
            api_key=api_key,
            api_system_prompt=api_prompt,
            api_model=api_model
        )

        if new_trans == TranslatorEngine.API_MODEL_NAME:
            self.status_bar.showMessage("正在预检翻译API...")
            ok, msg = self.translator.validate_api()
            if not ok:
                self.status_bar.showMessage(msg, 5000)
                return
        
        self.status_bar.showMessage(f"已切换模型: ASR={new_asr}, 翻译={new_trans}", 2000)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()