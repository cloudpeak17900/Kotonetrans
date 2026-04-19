from pathlib import Path
import json
import os
import re
from urllib import error, request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class APIAuthError(RuntimeError):
    """翻译API鉴权失败异常"""


class TranslatorEngine:
    """支持上下文感知的翻译"""

    KNOWN_MODELS = {
        "opus-mt-ja-zh": "Helsinki-NLP/opus-mt-ja-zh",
        "nllb-200-distilled-600M": "facebook/nllb-200-distilled-600M",
    }
    API_MODEL_NAME = "api-http"
    GLOSSARY_FILE = Path("./glossary.json")
    API_PRESETS = {
        "DeepSeek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        },
        "OpenAI": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
        "DashScope(阿里兼容)": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
        },
        "自定义": {
            "base_url": "",
            "model": "",
        },
    }

    @classmethod
    def discover_models(cls):
        """扫描 models/translate 中已下载模型，返回可选模型名列表"""
        model_root = Path("./models/translate")
        if not model_root.exists():
            return []

        models = []
        for item in model_root.iterdir():
            if not item.is_dir():
                continue

            model_name = None
            if item.name.startswith("models--"):
                parts = item.name.split("--")
                if len(parts) >= 3:
                    repo = "--".join(parts[2:])
                    model_name = repo
            elif "--" in item.name:
                model_name = item.name.split("--")[-1]
            else:
                model_name = item.name

            if model_name and model_name not in models:
                models.append(model_name)

        if cls.API_MODEL_NAME not in models:
            models.append(cls.API_MODEL_NAME)
        models.sort()
        return models

    def __init__(
        self,
        model_name=None,
        api_url=None,
        api_key=None,
        api_timeout=None,
        api_system_prompt=None,
        api_model=None
    ):
        available = self.discover_models()
        if model_name is None:
            model_name = available[0] if available else "opus-mt-ja-zh"

        self.model_name = model_name
        self.use_api = model_name == self.API_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.model_path = self._resolve_model_path(model_name)
        self.context_window = 3 if "nllb" in model_name.lower() else 2
        env_url = os.getenv("TRANSLATE_API_URL", "").strip()
        env_key = os.getenv("TRANSLATE_API_KEY", "").strip()
        env_timeout = os.getenv("TRANSLATE_API_TIMEOUT", "30")

        self.api_url = (api_url if api_url is not None else env_url).strip()
        self.api_key = (api_key if api_key is not None else env_key).strip()
        self.api_timeout = int(api_timeout if api_timeout is not None else env_timeout)
        env_model = os.getenv("TRANSLATE_API_MODEL", "deepseek-chat").strip()
        self.api_model = (api_model if api_model is not None else env_model).strip()
        self.api_system_prompt = (
            api_system_prompt
            if api_system_prompt is not None
            else "你是一名日本二次元广播剧和音声翻译者。保持口语自然，忠实原意，必要时保留语气词。仅输出中文翻译，不要解释。"
        )
        self.api_consecutive_failures = 0
        self.glossary = self._load_glossary()

    def _load_glossary(self):
        if not self.GLOSSARY_FILE.exists():
            default_glossary = {
                "先輩": "前辈",
                "お兄ちゃん": "哥哥",
                "お姉ちゃん": "姐姐",
            }
            self.GLOSSARY_FILE.write_text(
                json.dumps(default_glossary, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            return default_glossary

        try:
            data = json.loads(self.GLOSSARY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _clean_japanese_text(self, text):
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"(ー)\1{2,}", r"\1\1", cleaned)
        cleaned = re.sub(r"([。！？!?、,])\1{1,}", r"\1", cleaned)
        return cleaned

    def _dedupe_repeated_translation(self, text):
        if not text:
            return text

        half = len(text) // 2
        if len(text) >= 8 and text[:half] == text[half:half * 2]:
            return text[:half].strip()

        parts = [p.strip() for p in re.split(r"(。|！|？|\!|\?)", text) if p and p.strip()]
        rebuilt = []
        i = 0
        while i < len(parts):
            seg = parts[i]
            if i + 1 < len(parts) and re.fullmatch(r"(。|！|？|\!|\?)", parts[i + 1]):
                seg += parts[i + 1]
                i += 2
            else:
                i += 1
            if not rebuilt or rebuilt[-1] != seg:
                rebuilt.append(seg)
        return "".join(rebuilt).strip()

    def _apply_glossary_to_output(self, text):
        if not text:
            return text
        for jp, zh in self.glossary.items():
            if jp in text:
                text = text.replace(jp, zh)
        return text

    def _resolve_model_path(self, model_name):
        if model_name in self.KNOWN_MODELS:
            return self.KNOWN_MODELS[model_name]
        return model_name
    
    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir="./models/translate/"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                cache_dir="./models/translate/",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

    def _ensure_model(self):
        """确保模型已加载"""
        if self.use_api:
            return
        if self.model is None:
            self._load_model()
    
    def translate_with_context(self, jp_texts, current_idx):
        """
        带上下文翻译，解决省略问题
        jp_texts: 所有日文文本列表
        current_idx: 当前要翻译的句子索引
        """
        context_size = self.context_window
        
        # 构建上下文
        start = max(0, current_idx - context_size)
        end = min(len(jp_texts), current_idx + context_size + 1)
        
        context_texts = [self._clean_japanese_text(t) for t in jp_texts[start:end]]
        
        if self.use_api:
            # API大模型使用上下文提示，帮助处理省略和代词指代
            prompt = self._build_prompt(context_texts, current_idx - start)
            return self._translate_via_api(
                source_text=self._clean_japanese_text(jp_texts[current_idx]),
                prompt=prompt,
                context_texts=context_texts,
                current_pos=current_idx - start
            )

        self._ensure_model()

        # 本地机器翻译模型（NLLB/Opus）不适合提示词格式，直接翻当前句质量更稳定
        source_text = self._clean_japanese_text(jp_texts[current_idx])
        inputs = self.tokenizer(source_text, return_tensors="pt", max_length=512, truncation=True)
        generate_kwargs = {
            "max_length": 200,
            "num_beams": 3,
            "temperature": 0.7,
            "early_stopping": True,
        }

        # NLLB 需要显式指定源/目标语言，否则可能跑到错误语种
        if "nllb" in (self.model_name or "").lower():
            self.tokenizer.src_lang = "jpn_Jpan"
            zh_token_id = self.tokenizer.convert_tokens_to_ids("zho_Hans")
            if zh_token_id is not None and zh_token_id >= 0:
                generate_kwargs["forced_bos_token_id"] = zh_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs
            )
        
        zh_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        zh_text = self._dedupe_repeated_translation(zh_text)
        zh_text = self._apply_glossary_to_output(zh_text)
        return zh_text
    
    def _build_prompt(self, context_texts, current_pos):
        """构建带标记的翻译提示"""
        prompt_parts = []
        for i, text in enumerate(context_texts):
            if i == current_pos:
                prompt_parts.append(f"【要翻译】{text}")
            else:
                prompt_parts.append(text)
        
        return " | ".join(prompt_parts)

    def _translate_via_api(self, source_text, prompt, context_texts, current_pos):
        """通过HTTP API进行翻译"""
        if not self.api_url:
            return "（API未配置或已停用）"

        endpoint = self._normalize_api_endpoint(self.api_url)
        payload = {
            "text": source_text,
            "source_lang": "ja",
            "target_lang": "zh",
            "prompt": prompt,
            "context_texts": context_texts,
            "current_pos": current_pos,
        }
        if self._is_openai_compatible_endpoint(endpoint):
            payload = self._build_openai_compatible_payload(source_text, context_texts, current_pos)

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        response_data = None
        last_error = None
        # 每句最多重试一次，避免偶发网络抖动导致整段质量崩掉
        for _ in range(2):
            try:
                with request.urlopen(req, timeout=self.api_timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))
                break
            except error.HTTPError as e:
                if e.code == 401:
                    raise APIAuthError("API鉴权失败(401)，请检查URL和Key是否正确") from e
                last_error = e
                continue
            except (error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                continue

        if response_data is None:
            self.api_consecutive_failures += 1
            return f"（API翻译失败: {self._format_api_error(last_error)}）"

        translated = self._extract_translation(response_data)
        translated = self._dedupe_repeated_translation(translated)
        translated = self._apply_glossary_to_output(translated)
        self.api_consecutive_failures = 0
        return translated

    def validate_api(self):
        """快速预检 API 配置，避免整段音频跑完才发现 401。"""
        if not self.use_api:
            return True, ""
        if not self.api_url:
            return False, "API URL 为空"
        if not self.api_key:
            return False, "API Key 为空"

        endpoint = self._normalize_api_endpoint(self.api_url)
        if self._is_openai_compatible_endpoint(endpoint):
            payload = {
                "model": self.api_model or "deepseek-chat",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
            }
        else:
            payload = {"text": "ping", "source_lang": "ja", "target_lang": "zh"}

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=min(self.api_timeout, 15)) as resp:
                _ = resp.read()
            return True, ""
        except error.HTTPError as e:
            if e.code == 401:
                return False, "API鉴权失败(401)：请检查 URL / Key / 模型名"
            return False, f"API请求失败 HTTP {e.code}"
        except Exception as e:
            return False, f"API预检失败: {e}"

    def _normalize_api_endpoint(self, raw_url):
        url = (raw_url or "").strip().rstrip("/")
        if not url:
            return ""
        if "api.deepseek.com" in url and not url.endswith("/v1/chat/completions"):
            return f"{url}/v1/chat/completions"
        return url

    def _is_openai_compatible_endpoint(self, endpoint):
        lower = endpoint.lower()
        return lower.endswith("/v1/chat/completions") or "openai" in lower

    def _build_openai_compatible_payload(self, source_text, context_texts, current_pos):
        context_joined = "\n".join(
            [f"{'[TARGET]' if i == current_pos else '[CTX]'} {text}" for i, text in enumerate(context_texts)]
        )
        return {
            "model": self.api_model or "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": self.api_system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"请翻译这句日语为中文：\n{source_text}\n\n"
                        f"上下文：\n{context_joined}\n\n"
                        f"术语表（优先使用）：\n{self._format_glossary_for_prompt()}"
                    ),
                },
            ],
            "temperature": 0.2,
        }

    def _format_glossary_for_prompt(self):
        if not self.glossary:
            return "（无）"
        lines = []
        for jp, zh in list(self.glossary.items())[:50]:
            lines.append(f"- {jp} => {zh}")
        return "\n".join(lines)

    def _extract_translation(self, response_data):
        if isinstance(response_data, dict):
            choices = response_data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

            return (
                response_data.get("translation")
                or response_data.get("translated_text")
                or response_data.get("text")
                or response_data.get("result")
                or "（API返回为空）"
            )
        return "（API返回格式不支持）"

    def translate_batch(self, jp_texts, batch_size=10, progress_callback=None):
        """
        批量翻译多句，减少API调用次数，大幅提升速度

        Args:
            jp_texts: 日文文本列表
            batch_size: 每批翻译的句数
            progress_callback: 进度回调函数 callback(current, total)

        Returns:
            翻译结果列表
        """
        if not jp_texts:
            return []

        # 本地模型也使用批量翻译
        if not self.use_api:
            self._ensure_model()
            return self._translate_batch_local(jp_texts, progress_callback)

        # API模式：分批处理
        translations = [None] * len(jp_texts)
        total_batches = (len(jp_texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(jp_texts))
            batch_texts = jp_texts[start:end]

            try:
                batch_results = self._translate_batch_via_api(batch_texts, start, len(jp_texts))
                translations[start:end] = batch_results
            except Exception as e:
                # 批量失败时降级为逐句翻译
                for i, text in enumerate(batch_texts):
                    global_idx = start + i
                    context_texts = self._get_context_texts(jp_texts, global_idx)
                    translations[global_idx] = self._translate_via_api(
                        source_text=text,
                        prompt=self._build_prompt(context_texts, i),
                        context_texts=context_texts,
                        current_pos=i
                    )

            if progress_callback:
                progress_callback(end, len(jp_texts))

        return translations

    def _get_context_texts(self, jp_texts, current_idx):
        """获取当前句的上下文"""
        context_size = self.context_window
        start = max(0, current_idx - context_size)
        end = min(len(jp_texts), current_idx + context_size + 1)
        return [self._clean_japanese_text(t) for t in jp_texts[start:end]]

    def _translate_batch_local(self, jp_texts, progress_callback):
        """本地模型的批量翻译"""
        translations = []
        for i, text in enumerate(jp_texts):
            result = self.translate_with_context(jp_texts, i)
            translations.append(result)
            if progress_callback:
                progress_callback(i + 1, len(jp_texts))
        return translations

    def _translate_batch_via_api(self, batch_texts, batch_start, total_count):
        """通过API批量翻译一批文本"""
        if not self.api_url:
            return ["（API未配置）"] * len(batch_texts)

        endpoint = self._normalize_api_endpoint(self.api_url)

        if self._is_openai_compatible_endpoint(endpoint):
            return self._batch_translate_openai(batch_texts, batch_start, total_count)

        # 非OpenAI兼容接口，逐句处理
        results = []
        for i, text in enumerate(batch_texts):
            context = self._get_context_texts(batch_texts, i)
            results.append(self._translate_via_api(
                source_text=text,
                prompt=self._build_prompt(context, i),
                context_texts=context,
                current_pos=i
            ))
        return results

    def _batch_translate_openai(self, batch_texts, batch_start, total_count):
        """使用OpenAI兼容接口批量翻译"""
        # 构建批量翻译提示
        lines = []
        for i, text in enumerate(batch_texts):
            lines.append(f"{i + batch_start + 1}. {text}")

        payload = {
            "model": self.api_model or "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": f"{self.api_system_prompt}\n\n请批量翻译以下日语为中文，每行一句，按序号输出，不要添加任何解释。",
                },
                {
                    "role": "user",
                    "content": f"术语表：\n{self._format_glossary_for_prompt()}\n\n待翻译文本：\n" + "\n".join(lines),
                },
            ],
            "temperature": 0.2,
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(2):
            try:
                req = request.Request(endpoint=endpoint, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.api_timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))

                content = self._extract_translation(response_data)
                return self._parse_batch_response(content, len(batch_texts), batch_start + 1)

            except error.HTTPError as e:
                if e.code == 401:
                    raise APIAuthError("API鉴权失败(401)") from e
                if attempt == 0:
                    continue
                return [f"（API错误: {e.code}）"] * len(batch_texts)
            except Exception as e:
                if attempt == 0:
                    continue
                return [f"（API失败: {e}）"] * len(batch_texts)

        return [f"（API失败）"] * len(batch_texts)

    def _parse_batch_response(self, content, expected_count, start_number):
        """解析批量翻译的响应"""
        lines = content.strip().split("\n")
        results = []

        # 尝试解析 "1. 翻译" 或 "1.翻译" 格式
        for i in range(expected_count):
            target_num = start_number + i
            found = False

            for line in lines:
                line = line.strip()
                # 匹配 "1. xxx" 或 "1.xxx"
                match = re.match(rf"^{target_num}\.\s*(.+)", line)
                if match:
                    results.append(match.group(1))
                    found = True
                    break

            if not found:
                # 解析失败，回退到逐行对应
                if i < len(lines):
                    clean_line = re.sub(r"^\d+\.\s*", "", lines[i].strip())
                    results.append(clean_line if clean_line else "（解析失败）")
                else:
                    results.append("（缺失）")

        return results

    def _format_api_error(self, err):
        if err is None:
            return "未知错误"
        if isinstance(err, error.HTTPError):
            return f"HTTP {err.code}"
        if isinstance(err, TimeoutError):
            return "超时"
        if isinstance(err, error.URLError):
            return "网络错误"
        if isinstance(err, json.JSONDecodeError):
            return "返回非JSON"
        return "请求异常"

    def translate_concurrent(self, jp_texts, max_workers=4, progress_callback=None):
        """
        并发翻译，通过多线程同时处理多个翻译请求

        Args:
            jp_texts: 日文文本列表
            max_workers: 最大并发线程数
            progress_callback: 进度回调函数 callback(current, total)

        Returns:
            翻译结果列表
        """
        if not jp_texts:
            return []

        if not self.use_api:
            # 本地模型不支持并发（GPU锁），回退到批量翻译
            return self.translate_batch(jp_texts, progress_callback=progress_callback)

        translations = [None] * len(jp_texts)
        completed = 0
        lock = threading.Lock()

        def translate_one(idx):
            nonlocal completed
            try:
                context_texts = self._get_context_texts(jp_texts, idx)
                result = self._translate_via_api(
                    source_text=jp_texts[idx],
                    prompt=self._build_prompt(context_texts, idx - max(0, idx - self.context_window)),
                    context_texts=context_texts,
                    current_pos=idx - max(0, idx - self.context_window)
                )
                translations[idx] = result
            except Exception as e:
                translations[idx] = f"（翻译失败: {e}）"
            finally:
                with lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(jp_texts))

        # 使用线程池并发翻译
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(translate_one, i) for i in range(len(jp_texts))]
            for future in as_completed(futures):
                future.result()  # 等待所有任务完成

        return translations