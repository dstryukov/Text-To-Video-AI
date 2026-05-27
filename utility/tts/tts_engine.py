import os
import re
import wave
import torch
import shutil
import base64
import requests
import tempfile
import platform
import subprocess
from utility.config import get_config

# Локальный кэш для модели Silero
_silero_model = None

# ================= 1. РУССКАЯ НОРМАЛИЗАЦИЯ ЧИСЕЛ =================

def number_to_text_ru(num: int, ordinal: bool = False) -> str:
    if num == 0:
        return "нулевой" if ordinal else "ноль"
        
    units = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
    tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
    
    ord_units = ["", "первый", "второй", "третий", "четвертый", "пятый", "шестой", "седьмой", "восьмой", "девятый"]
    ord_teens = ["десять", "одиннадцатый", "двенадцать", "тринадцатый", "четырнадцатый", "пятнадцатый", "шестнадцатый", "семнадцатый", "восемнадцатый", "девятнадцатый"]
    ord_teens = ["десятый", "одиннадцатый", "двенадцатый", "тринадцатый", "четырнадцатый", "пятнадцатый", "шестнадцатый", "семнадцатый", "восемнадцатый", "девятнадцатый"]
    ord_tens = ["", "", "двадцатый", "тридцатый", "сороковой", "пятидесятый", "шестидесятый", "семидесятый", "восьмидесятый", "девяностый"]
    ord_hundreds = ["", "сотый", "двухсотый", "трехсотый", "четырехсотый", "пятисотый", "шестисотый", "семисотый", "восьмисотый", "девятисотый"]
    
    parts = []
    
    # Тысячи
    th = num // 1000
    rem = num % 1000
    if th > 0:
        if ordinal and rem == 0:
            if th == 1:
                parts.append("тысячный")
            elif th == 2:
                parts.append("двухтысячный")
            else:
                th_prefixes = ["", "", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"]
                parts.append(th_prefixes[th] + "тысячный")
        else:
            if th == 1:
                parts.append("тысяча")
            elif th == 2:
                parts.append("две тысячи")
            elif th in [3, 4]:
                parts.append(units[th] + " тысячи")
            else:
                parts.append(units[th] + " тысяч")
                
    # Сотни
    h = rem // 100
    rem = rem % 100
    if h > 0:
        if ordinal and rem == 0:
            parts.append(ord_hundreds[h])
        else:
            parts.append(hundreds[h])
            
    # Десятки и единицы
    t = rem // 10
    u = rem % 10
    
    if t == 1:
        if ordinal:
            parts.append(ord_teens[u])
        else:
            parts.append(teens[u])
    else:
        if t > 1:
            if ordinal and u == 0:
                parts.append(ord_tens[t])
            else:
                parts.append(tens[t])
        if u > 0:
            if ordinal:
                parts.append(ord_units[u])
            else:
                parts.append(units[u])
                
    return " ".join(p for p in parts if p)

def replace_numbers_ru(text: str) -> str:
    # Заменяем шаблоны года, например "в 2026 году" или "в 2026 г."
    def year_repl(match):
        num = int(match.group(1))
        suffix = match.group(2) or ""
        val = number_to_text_ru(num, ordinal=True)
        return f"{val} {suffix}" if suffix else val
        
    text = re.sub(r'\b(19\d{2}|20\d{2})\b\s*(г\.?|год[а-я]*)', year_repl, text)
    
    # Заменяем остальные числа
    def card_repl(match):
        num = int(match.group(1))
        return number_to_text_ru(num, ordinal=False)
        
    text = re.sub(r'\b(\d+)\b', card_repl, text)
    return text

# ================= 2. НОРМАЛИЗАЦИЯ И ЧАНКИНГ ТЕКСТА =================

def clean_markdown(text):
    """Очистка текста от разметки Markdown."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text_for_tts(text: str, config_dict: dict = None) -> str:
    """Улучшенная нормализация русского текста для озвучки."""
    if config_dict is None:
        config_dict = {}
        
    text = clean_markdown(text)
    
    # 1. Применяем словарь замен пользователя из конфигурации (если есть)
    replacements = config_dict.get('replacements', {})
    for src, dst in replacements.items():
        # Используем границы слов для точной замены аббревиатур
        text = re.sub(rf'\b{re.escape(src)}\b', dst, text)
        
    # 2. Базовая замена спецсимволов словами
    symbol_replacements = {
        "%": " процентов ",
        "+": " плюс ",
        "=": " равно ",
        "$": " долларов ",
        "€": " евро ",
        "№": " номер ",
        "&": " и ",
        "@": " собака ",
        "/": " дробь "
    }
    
    for symbol, word in symbol_replacements.items():
        text = text.replace(symbol, word)
        
    # 3. Нормализация чисел и дат на русском
    text = replace_numbers_ru(text)
    
    # 4. Уборка спецсимволов и дубликатов
    text = re.sub(r'\-{2,}', ' — ', text)
    text = re.sub(r'\.{2,}', '... ', text)
    text = text.replace("<", " ").replace(">", " ")
    
    # 5. Очистка лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_text_into_chunks(text: str, max_chars: int = 350) -> list:
    """Разбивка текста на фрагменты, не превышающие max_chars символов."""
    if len(text) <= max_chars:
        return [text]
        
    # Разбиваем по границам предложений
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if current_len + len(s) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_len = len(s)
        else:
            current_chunk.append(s)
            current_len += len(s) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# ================= 3. ПОСТОБРАБОТКА АУДИО =================

def shorten_silence(sound, max_silence_ms=500, silence_thresh=-45):
    """Сжатие пауз тишины до max_silence_ms в pydub AudioSegment."""
    try:
        from pydub.silence import detect_silence
    except ImportError:
        print("Warning: pydub.silence not importable. Skipping silence shortening.")
        return sound
        
    silences = detect_silence(sound, min_silence_len=max_silence_ms, silence_thresh=silence_thresh)
    if not silences:
        return sound
        
    chunks = []
    last_end = 0
    for start, end in silences:
        chunks.append(sound[last_end:start])
        chunks.append(sound[start:start + max_silence_ms])
        last_end = end
    chunks.append(sound[last_end:])
    
    combined = sound[:0]
    for chunk in chunks:
        combined += chunk
    return combined

def postprocess_audio(input_path: str, output_path: str, pp_config: dict) -> str:
    """Постобработка звукового файла: LUFS-громкость, паузы, компрессия."""
    if not pp_config.get('enabled', True):
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return output_path
        
    try:
        from pydub import AudioSegment
    except ImportError:
        print("Warning: pydub is not installed. Skipping audio post-processing.")
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return output_path
        
    print(f"Applying audio post-processing to {input_path}...")
    try:
        sound = AudioSegment.from_file(input_path)
        
        # 1. Удаление/сжатие длинных пауз
        if pp_config.get('remove_silence', True):
            max_silence = int(pp_config.get('max_silence_ms', 500))
            sound = shorten_silence(sound, max_silence_ms=max_silence)
            
        # 2. Нормализация громкости к LUFS/dBFS
        if pp_config.get('normalize_loudness', True):
            target_dbfs = float(pp_config.get('target_lufs', -14))
            change_in_dBFS = target_dbfs - sound.dBFS
            sound = sound.apply_gain(change_in_dBFS)
            
            # Пиковый лимитер для предотвращения клиппинга
            if sound.max_dBFS > -1.0:
                sound = sound.apply_gain(-1.0 - sound.max_dBFS)
                
        # 3. Экспорт результата
        sound.export(output_path, format="wav")
        print(f"Post-processed audio saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Warning: Audio post-processing failed: {e}")
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return output_path

# ================= 4. РЕАЛИЗАЦИЯ БЭКЕНДОВ TTS =================

def generate_silence(output_path, text, sample_rate=24000):
    """Генерация пустого (тихого) аудиофайла на основе объема текста."""
    words_count = len(text.split())
    # Примерно 2.5 слова в секунду, минимум 2 секунды
    duration = max(2.0, words_count / 2.5)
    
    temp_wav = output_path if output_path.endswith('.wav') else output_path + ".temp.wav"
    with wave.open(temp_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        num_frames = int(duration * sample_rate)
        wav_file.writeframes(b'\x00' * (num_frames * 2))
        
    if output_path.endswith('.mp3'):
        try:
            from moviepy.editor import AudioFileClip
        except ImportError:
            from moviepy import AudioFileClip
        clip = AudioFileClip(temp_wav)
        clip.write_audiofile(output_path, logger=None)
        clip.close()
        if temp_wav != output_path:
            os.remove(temp_wav)
    elif temp_wav != output_path:
        os.rename(temp_wav, output_path)

def generate_f5_tts(text: str, output_path: str, config: dict):
    """Синтез через F5-TTS с клонированием голоса."""
    f5_cfg = config.get('f5_tts', {})
    ref_audio = config.get('reference_audio_path', 'voices/reference.wav')
    ref_text = config.get('reference_text', '')
    use_clone = config.get('use_voice_clone', True)
    speed = float(f5_cfg.get('speed', 1.0))
    cfg = float(f5_cfg.get('cfg_strength', 2.0))
    
    # Priority resolution for F5 nfe_step:
    # Backend default -> Quality Preset -> Explicit Config / CLI Override
    nfe = 32
    quality_preset_name = config.get('quality_preset', 'balanced')
    quality_presets = get_config().get_tts_quality_presets()
    preset_cfg = quality_presets.get(quality_preset_name, {}).get('f5_tts', {})
    if 'nfe_step' in preset_cfg:
        nfe = int(preset_cfg['nfe_step'])
        
    user_nfe = get_config().yaml_config.get('tts', {}).get('f5_tts', {}).get('nfe_step')
    if user_nfe is not None:
        nfe = int(user_nfe)
    device = f5_cfg.get('device', 'cuda')
    dtype = f5_cfg.get('dtype', 'float16')
    
    # 1. Попытка импорта библиотеки python API
    try:
        from f5_tts.api import F5TTS
        print("F5-TTS: Инициализация python API...")
        # Локальный импорт torch для dtype
        import torch
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        
        # Загружаем модель
        f5tts = F5TTS(model_type=f5_cfg.get('model', 'F5-TTS'), device=device, dtype=torch_dtype)
        
        if use_clone and os.path.exists(ref_audio):
            print(f"F5-TTS: Voice cloning с файлом {ref_audio}")
            f5tts.infer(
                ref_audio_v4=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                file_name=output_path,
                speed=speed,
                cfg_strength=cfg,
                nfe_step=nfe
            )
            return output_path
        else:
            raise FileNotFoundError(f"Reference audio not found or cloning disabled. Path: {ref_audio}")
            
    except Exception as e:
        print(f"F5-TTS: Python API failed or not installed ({e}). Trying CLI...")
        
        # 2. Попытка вызова CLI как запасной вариант
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found at {ref_audio} for F5-TTS voice cloning.")
            
        cmd = [
            "f5-tts_infer-cli",
            "-r", ref_audio,
            "-s", ref_text,
            "-t", text,
            "-o", output_path,
            "--nfe_step", str(nfe),
            "--cfg_strength", str(cfg),
            "--speed", str(speed)
        ]
        
        try:
            print(f"F5-TTS: running CLI command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("F5-TTS CLI ran successfully.")
            return output_path
        except Exception as cli_err:
            print(f"F5-TTS CLI failed: {cli_err}")
            raise RuntimeError(
                "F5-TTS is not installed. Install it or switch tts.backend to silero/local_tts_api."
            )

def generate_fish_speech(text: str, output_path: str, config: dict):
    """Синтез через Fish Speech."""
    fish_cfg = config.get('fish_speech', {})
    ref_audio = config.get('reference_audio_path', 'voices/reference.wav')
    use_clone = config.get('use_voice_clone', True)
    temp = float(fish_cfg.get('temperature', 0.7))
    top_p = float(fish_cfg.get('top_p', 0.8))
    rep_penalty = float(fish_cfg.get('repetition_penalty', 1.1))
    
    print("Fish-Speech: Запуск генерации через CLI...")
    cmd = [
        "fish-speech-cli",
        "--text", text,
        "--output", output_path,
        "--temperature", str(temp),
        "--top_p", str(top_p),
        "--repetition_penalty", str(rep_penalty)
    ]
    if use_clone and os.path.exists(ref_audio):
        cmd.extend(["--reference-audio", ref_audio])
        
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Fish-Speech CLI ran successfully.")
        return output_path
    except Exception as e:
        print(f"Fish-Speech CLI failed: {e}")
        raise RuntimeError(
            "Fish-Speech is not installed or checkpoints are missing. Falling back."
        )

def generate_cosyvoice(text: str, output_path: str, config: dict):
    """Синтез через CosyVoice."""
    cosy_cfg = config.get('cosyvoice', {})
    ref_audio = config.get('reference_audio_path', 'voices/reference.wav')
    use_clone = config.get('use_voice_clone', True)
    model_path = cosy_cfg.get('model_path', 'checkpoints/cosyvoice')
    
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice
        import soundfile as sf
        
        print("CosyVoice: Инициализация python API...")
        cosyvoice = CosyVoice(model_path)
        
        if use_clone and os.path.exists(ref_audio):
            ref_text = config.get('reference_text', '')
            print(f"CosyVoice Zero-Shot cloning с референсом {ref_audio}...")
            # Получаем генератор аудио
            output = cosyvoice.inference_zero_shot(text, ref_text, ref_audio)
            # Извлекаем первый чанк
            for r in output:
                # r['tts_speech'] это факельный тензор
                audio_data = r['tts_speech'].numpy()
                sf.write(output_path, audio_data, cosy_cfg.get('sample_rate', 22050))
                break
            print("CosyVoice: аудио успешно сгенерировано.")
            return output_path
        else:
            raise FileNotFoundError(f"Reference audio not found or clone disabled for CosyVoice. Path: {ref_audio}")
    except Exception as e:
        print(f"CosyVoice python inference failed: {e}")
        raise RuntimeError("CosyVoice is not installed. Falling back.")

def generate_local_tts_api(text: str, output_path: str, config: dict):
    """Синтез через универсальный POST HTTP-сервис."""
    api_url = config.get('local_tts_api_url', 'http://127.0.0.1:8020/tts')
    payload = {
        "text": text,
        "language": config.get('language', 'ru'),
        "voice": config.get('voice', 'default'),
        "speed": float(config.get('speed', 1.0)),
        "emotion": config.get('emotion', 'neutral'),
        "reference_audio_path": config.get('reference_audio_path', 'voices/reference.wav'),
        "reference_text": config.get('reference_text', ''),
        "sample_rate": int(config.get('sample_rate', 24000)),
        "format": config.get('format', 'wav')
    }
    
    print(f"Sending POST request to Local TTS API: {api_url}...")
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Could not connect to Local TTS API at {api_url}: {e}")
        
    content_type = response.headers.get('content-type', '').lower()
    if 'audio/' in content_type or (len(response.content) > 2000 and not content_type.startswith('application/json')):
        # Ответ содержит бинарный аудио-поток
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
        
    # Пробуем распарсить JSON
    try:
        data = response.json()
        
        # Поиск base64
        b64_data = None
        for key in ['audio', 'audio_b64', 'base64', 'b64']:
            if key in data:
                val = data[key]
                if isinstance(val, list) and len(val) > 0:
                    b64_data = val[0]
                elif isinstance(val, str):
                    b64_data = val
                break
                
        if b64_data:
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            audio_bytes = base64.b64decode(b64_data)
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            return output_path
            
        # Поиск URL
        audio_url = None
        for key in ['url', 'audio_url', 'link']:
            if key in data:
                audio_url = data[key]
                break
                
        if audio_url:
            print(f"Downloading API audio from URL: {audio_url}")
            img_resp = requests.get(audio_url, timeout=30)
            img_resp.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(img_resp.content)
            return output_path
            
    except Exception as e:
        raise RuntimeError(f"Failed to parse Local TTS API JSON response: {e}")
        
    raise RuntimeError("Could not retrieve audio data from Local TTS API response.")

def use_existing_audio_file(output_path: str, config: dict):
    """Использование заранее подготовленного аудиофайла."""
    audio_cfg = config.get('audio_file', {})
    source_path = audio_cfg.get('path', 'input_audio/voiceover.wav')
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Audio file not found: {source_path}")
    shutil.copy2(source_path, output_path)
    print(f"Copied pre-existing audio file from {source_path} to {output_path}")
    return output_path

def generate_silero_tts(text: str, output_path: str, config: dict):
    """Синтез через Silero TTS."""
    global _silero_model
    device = torch.device('cpu')
    torch.set_num_threads(4)
    
    if _silero_model is None:
        print("Loading Silero TTS model...")
        model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru'
        )
        _silero_model = model
        _silero_model.to(device)
    
    sil_cfg = config.get('silero', {})
    voice = sil_cfg.get('voice', 'baya')
    sample_rate = int(sil_cfg.get('sample_rate', 24000))
    speed = sil_cfg.get('speed', 'medium')
    
    # Регулировка темпа
    rate_preset = "medium"
    if isinstance(speed, str):
        rate_preset = speed
    elif isinstance(speed, (int, float)):
        if speed <= 0.6:
            rate_preset = "x-slow"
        elif speed <= 0.85:
            rate_preset = "slow"
        elif speed <= 1.15:
            rate_preset = "medium"
        elif speed <= 1.5:
            rate_preset = "fast"
        else:
            rate_preset = "x-fast"
            
    ssml_text = f'<speak><prosody rate="{rate_preset}">{text}</prosody></speak>'
    
    temp_wav = output_path if output_path.endswith('.wav') else output_path + ".temp.wav"
    _silero_model.save_wav(
        text=ssml_text,
        speaker=voice,
        sample_rate=sample_rate,
        audio_path=temp_wav
    )
    
    if output_path.endswith('.mp3'):
        try:
            from moviepy.editor import AudioFileClip
        except ImportError:
            from moviepy import AudioFileClip
        clip = AudioFileClip(temp_wav)
        clip.write_audiofile(output_path, logger=None)
        clip.close()
        if temp_wav != output_path:
            os.remove(temp_wav)
    elif temp_wav != output_path:
        os.rename(temp_wav, output_path)
        
    return output_path

# ================= 5. ЕДИНЫЙ ИНТЕРФЕЙС / ROUTING =================

def generate_tts(text: str, output_path: str, config_dict: dict = None) -> str:
    """
    Единый интерфейс генерации озвучки с многоуровневым Fallback и постобработкой.
    """
    if config_dict is None:
        config = get_config()
        config_dict = config.get_tts_config()
        
    backend = config_dict.get('backend', 'silero').lower()
    fallback_backend = config_dict.get('fallback_backend', 'silero').lower()
    normalize = config_dict.get('normalize_text', True)
    
    # 1. Нормализация текста
    if normalize:
        processed_text = normalize_text_for_tts(text, config_dict)
    else:
        processed_text = text.strip()
        
    if not processed_text:
        processed_text = "Пустой текст"
        
    print("\n" + "="*70)
    print("STARTING TTS GENERATION:")
    print(f"Selected TTS backend: {backend}")
    print(f"Fallback TTS backend: {fallback_backend}")
    print(f"Voice cloning: {'enabled' if config_dict.get('use_voice_clone') else 'disabled'}")
    print(f"Reference audio: {config_dict.get('reference_audio_path')}")
    print(f"Normalize text: {normalize}")
    print(f"Normalized text preview: '{processed_text[:60]}...'")
    print("="*70 + "\n")
    
    # 2. Выполнение генерации (с разбиением на чанки при необходимости)
    max_chars = int(config_dict.get('max_chars_per_chunk', 350))
    pause_ms = int(config_dict.get('pause_between_chunks_ms', 250))
    sample_rate = int(config_dict.get('sample_rate', 24000))
    
    # Вспомогательная функция запуска выбранного бэкенда
    def run_backend(selected_backend, t_text, o_path):
        if selected_backend == "f5_tts":
            return generate_f5_tts(t_text, o_path, config_dict)
        elif selected_backend == "fish_speech":
            return generate_fish_speech(t_text, o_path, config_dict)
        elif selected_backend == "cosyvoice":
            return generate_cosyvoice(t_text, o_path, config_dict)
        elif selected_backend == "local_tts_api":
            return generate_local_tts_api(t_text, o_path, config_dict)
        elif selected_backend == "silero":
            return generate_silero_tts(t_text, o_path, config_dict)
        elif selected_backend == "audio_file":
            return use_existing_audio_file(o_path, config_dict)
        elif selected_backend == "none":
            return generate_silence(o_path, t_text, sample_rate)
        else:
            raise ValueError(f"Unknown TTS backend: {selected_backend}")
            
    # Запуск с Fallback
    def execute_with_fallback(t_text, final_out_path):
        try:
            return run_backend(backend, t_text, final_out_path)
        except Exception as e:
            print(f"Primary TTS backend '{backend}' failed with error: {e}")
            if fallback_backend and fallback_backend != backend:
                print(f"Primary TTS backend failed. Trying fallback backend: {fallback_backend}.")
                try:
                    return run_backend(fallback_backend, t_text, final_out_path)
                except Exception as fallback_err:
                    print(f"Fallback TTS backend '{fallback_backend}' also failed: {fallback_err}")
                    raise fallback_err
            else:
                raise e

    # Реализация чанкинга (только для текстовых бэкендов, пропускаем для готового аудиофайла)
    if backend == "audio_file":
        try:
            execute_with_fallback(processed_text, output_path)
        except Exception as e:
            print(f"Error copying audio file: {e}. Generating silent placeholder.")
            generate_silence(output_path, processed_text, sample_rate)
    else:
        chunks = split_text_into_chunks(processed_text, max_chars)
        if len(chunks) == 1:
            try:
                execute_with_fallback(processed_text, output_path)
            except Exception as e:
                print(f"Error synthesizing audio: {e}. Generating silent placeholder.")
                generate_silence(output_path, processed_text, sample_rate)
        else:
            print(f"Splitting text into {len(chunks)} chunks due to max_chars_per_chunk limit ({max_chars}).")
            temp_dir = tempfile.mkdtemp()
            temp_paths = []
            
            try:
                for idx, chunk in enumerate(chunks):
                    chunk_path = os.path.join(temp_dir, f"chunk_{idx}.wav")
                    print(f"Synthesizing chunk {idx + 1}/{len(chunks)}: '{chunk[:40]}...'")
                    execute_with_fallback(chunk, chunk_path)
                    temp_paths.append(chunk_path)
                    
                # Соединяем чанки через pydub
                from pydub import AudioSegment
                combined = AudioSegment.empty()
                silence_segment = AudioSegment.silent(duration=pause_ms, frame_rate=sample_rate)
                
                for idx, path in enumerate(temp_paths):
                    segment = AudioSegment.from_file(path)
                    if idx > 0:
                        combined += silence_segment
                    combined += segment
                    
                combined.export(output_path, format="wav")
                print("All chunks concatenated successfully.")
                
            except Exception as e:
                print(f"Failed to synthesize or concatenate text chunks: {e}. Generating silent placeholder.")
                generate_silence(output_path, processed_text, sample_rate)
            finally:
                # Очистка временных файлов
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                    
    # 3. Применяем аудио-постобработку к итоговому файлу
    try:
        pp_config = get_config().get_audio_postprocess_config()
    except Exception:
        pp_config = {
            'enabled': True,
            'normalize_loudness': True,
            'target_lufs': -14,
            'remove_silence': True,
            'max_silence_ms': 500,
            'add_compressor': True,
            'noise_reduction': False
        }
        
    postprocess_audio(output_path, output_path, pp_config)
    
    if config_dict.get('unload_tts_model_after_generation', True):
        global _silero_model
        if _silero_model is not None:
            print("Unloading Silero TTS model from cache...")
            from utility.memory import clear_memory
            _silero_model = None
            clear_memory()
            
    return output_path
