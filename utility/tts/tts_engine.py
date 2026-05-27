import os
import re
import wave
import torch
from utility.config import get_config

# Локальный кэш для модели Silero
_silero_model = None
_silero_symbols = None

def clean_markdown(text):
    """Очистка текста от разметки Markdown, чтобы не ломать озвучку."""
    # Удаление жирного шрифта (**текст**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Удаление курсива (*текст* или _текст_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Удаление кода (`текст` или ```текст```)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Удаление заголовков (# текст)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Удаление ссылок [текст](url) -> текст
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Очистка лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text_for_tts(text):
    """Нормализация текста для озвучки (замена спецсимволов словами и очистка)."""
    text = clean_markdown(text)
    
    # Замена популярных знаков словами на русском
    replacements = {
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
    
    for symbol, word in replacements.items():
        text = text.replace(symbol, word)
        
    # Заменяем повторяющиеся тире или точки на пробелы / паузы
    text = re.sub(r'\-{2,}', ' — ', text)
    text = re.sub(r'\.{2,}', '... ', text)
    
    # Убираем недопустимые SSML-символы (<, >, &) вне тегов
    # Мы сами обернем в speak, поэтому сырой текст очищаем от угловых скобок
    text = text.replace("<", " ").replace(">", " ")
    
    return text.strip()

def map_speed_to_ssml_rate(speed):
    """Преобразование численного темпа речи в строковые пресеты Silero SSML."""
    if isinstance(speed, str):
        speed_lower = speed.lower()
        valid_presets = ["x-slow", "slow", "medium", "fast", "x-fast"]
        if speed_lower in valid_presets:
            return speed_lower
        try:
            speed = float(speed)
        except ValueError:
            return "medium"
            
    if speed <= 0.6:
        return "x-slow"
    elif speed <= 0.85:
        return "slow"
    elif speed <= 1.15:
        return "medium"
    elif speed <= 1.5:
        return "fast"
    else:
        return "x-fast"

def generate_silence(output_path, text, sample_rate=24000):
    """Генерация пустого (тихого) аудиофайла на основе объема текста."""
    words_count = len(text.split())
    # Примерно 2.5 слова в секунду, минимум 2 секунды
    duration = max(2.0, words_count / 2.5)
    
    # Создаем временный wav-файл
    temp_wav = output_path if output_path.endswith('.wav') else output_path + ".temp.wav"
    
    with wave.open(temp_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        num_frames = int(duration * sample_rate)
        wav_file.writeframes(b'\x00' * (num_frames * 2))
        
    # Если на выходе нужен mp3, конвертируем
    if output_path.endswith('.mp3'):
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(temp_wav)
            clip.write_audiofile(output_path, logger=None)
            clip.close()
            if temp_wav != output_path:
                os.remove(temp_wav)
        except Exception as e:
            print(f"Warning: Failed to convert silent wav to mp3: {e}. Keeping wav name as fallback.")
            if temp_wav != output_path:
                os.rename(temp_wav, output_path)

def generate_tts(text: str, output_path: str, config_dict: dict = None):
    """
    Единый интерфейс генерации озвучки.
    
    Args:
        text: Текст для озвучки
        output_path: Путь для сохранения результата (.wav или .mp3)
        config_dict: Словарь с настройками (voice, speed, sample_rate, backend, и т.д.)
    """
    if config_dict is None:
        config = get_config()
        config_dict = config.get_tts_config()
        
    backend = config_dict.get('backend', 'silero').lower()
    voice = config_dict.get('voice', 'xenia')
    speed = config_dict.get('speed', 'medium')
    sample_rate = int(config_dict.get('sample_rate', 24000))
    normalize_text = config_dict.get('normalize_text', True)
    local_api_url = config_dict.get('local_api_url', 'http://localhost:8010/tts')
    
    if normalize_text:
        processed_text = normalize_text_for_tts(text)
    else:
        processed_text = text.strip()
        
    if not processed_text:
        processed_text = "Пустой текст"

    print(f"Generating TTS using backend: {backend}, voice: {voice}, output: {output_path}")

    # Бэкенд 1: SILERO (локальный)
    if backend == 'silero':
        global _silero_model, _silero_symbols
        try:
            device = torch.device('cpu')
            torch.set_num_threads(4)
            
            # Загрузка модели, если она не загружена
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
            
            # Применение темпа речи через SSML
            rate_preset = map_speed_to_ssml_rate(speed)
            ssml_text = f'<speak><prosody rate="{rate_preset}">{processed_text}</prosody></speak>'
            
            # Путь для сохранения wav-файла
            temp_wav = output_path if output_path.endswith('.wav') else output_path + ".temp.wav"
            
            # Синтезируем wav напрямую в файл
            _silero_model.save_wav(
                text=ssml_text,
                speaker=voice,
                sample_rate=sample_rate,
                audio_path=temp_wav
            )
            
            # Если нужен mp3, конвертируем
            if output_path.endswith('.mp3'):
                from moviepy.editor import AudioFileClip
                clip = AudioFileClip(temp_wav)
                clip.write_audiofile(output_path, logger=None)
                clip.close()
                if temp_wav != output_path:
                    os.remove(temp_wav)
                    
            print("Silero TTS audio generated successfully.")
            return output_path
            
        except Exception as e:
            print(f"Error in Silero TTS: {e}")
            print("Falling back to silent audio generation...")
            generate_silence(output_path, processed_text, sample_rate)
            return output_path

    # Бэкенд 2: LOCAL_API (внешний HTTP сервис)
    elif backend == 'local_api':
        try:
            import requests
            payload = {
                "text": processed_text,
                "voice": voice,
                "speed": speed,
                "sample_rate": sample_rate,
                "format": "mp3" if output_path.endswith('.mp3') else "wav"
            }
            response = requests.post(local_api_url, json=payload, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print("Local API TTS audio generated successfully.")
                return output_path
            else:
                raise Exception(f"Local API returned status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error in local_api TTS: {e}")
            print("Falling back to silent audio generation...")
            generate_silence(output_path, processed_text, sample_rate)
            return output_path

    # Бэкенд 3: NONE / MOCK
    elif backend == 'none':
        generate_silence(output_path, processed_text, sample_rate)
        print("Silent/Mock audio generated successfully.")
        return output_path

    else:
        # Устаревшие провайдеры для обратной совместимости (elevenlabs)
        try:
            import asyncio
            if backend == 'elevenlabs':
                from utility.tts.elevenlabs_tts import generate_audio as elevenlabs_audio
                asyncio.run(elevenlabs_audio(processed_text, output_path, voice))
            else:
                raise ValueError(f"Unknown TTS backend: {backend}")
            print(f"Legacy TTS {backend} audio generated successfully.")
            return output_path
        except Exception as e:
            print(f"Error in legacy TTS: {e}")
            print("Falling back to silent audio generation...")
            generate_silence(output_path, processed_text, sample_rate)
            return output_path
