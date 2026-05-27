import os
import time
import json
import base64
import requests
import tempfile
import platform
import subprocess
import math

try:
    from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip, ImageClip,
                                   TextClip, ColorClip, VideoFileClip)
except ImportError:
    from moviepy import (AudioFileClip, CompositeVideoClip, CompositeAudioClip, ImageClip,
                                   TextClip, ColorClip, VideoFileClip)

from utility.config import get_config

# ----------------- СОВМЕСТИМОСТЬ MOVIEPY V1 / V2 -----------------

def set_clip_duration(clip, duration):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)

def set_clip_start(clip, start):
    if hasattr(clip, "with_start"):
        return clip.with_start(start)
    return clip.set_start(start)

def set_clip_end(clip, end):
    if hasattr(clip, "with_end"):
        return clip.with_end(end)
    return clip.set_end(end)

def set_clip_position(clip, position):
    if hasattr(clip, "with_position"):
        return clip.with_position(position)
    return clip.set_position(position)

def set_clip_audio(clip, audio):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip.set_audio(audio)

def resize_clip(clip, newsize=None, width=None, height=None):
    if hasattr(clip, "resized"):
        if width is not None or height is not None:
            return clip.resized(width=width, height=height)
        return clip.resized(newsize)
    if hasattr(clip, "resize"):
        if width is not None or height is not None:
            return clip.resize(width=width, height=height)
        return clip.resize(newsize)
    raise AttributeError("Clip has no resize or resized method")

def crop_clip(clip, x1=None, y1=None, x2=None, y2=None):
    if hasattr(clip, "cropped"):
        return clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    try:
        from moviepy.video.fx.crop import crop as v1_crop
        return v1_crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)
    except ImportError:
        if hasattr(clip, "crop"):
            return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
        raise

def create_text_clip(text, font, font_size, color, stroke_width, stroke_color, size, method="caption"):
    # We want a list of font fallbacks based on what was requested
    fonts_to_try = [font]
    
    # Try mapping common names to actual font files on Windows
    font_lower = font.lower()
    if platform.system() == "Windows":
        if "arial" in font_lower:
            if "bold" in font_lower:
                fonts_to_try.append("arialbd.ttf")
            elif "italic" in font_lower:
                fonts_to_try.append("ariali.ttf")
            else:
                fonts_to_try.append("arial.ttf")
        elif "times" in font_lower:
            if "bold" in font_lower:
                fonts_to_try.append("timesbd.ttf")
            elif "italic" in font_lower:
                fonts_to_try.append("timesi.ttf")
            else:
                fonts_to_try.append("times.ttf")
        elif "courier" in font_lower:
            if "bold" in font_lower:
                fonts_to_try.append("courbd.ttf")
            elif "italic" in font_lower:
                fonts_to_try.append("couri.ttf")
            else:
                fonts_to_try.append("cour.ttf")
    
    # Add general fallbacks at the end
    fallbacks = ["arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "Arial", "sans-serif"]
    for f in fallbacks:
        if f not in fonts_to_try:
            fonts_to_try.append(f)
            
    # Try creating TextClip with each font in sequence until one succeeds
    last_err = None
    for f in fonts_to_try:
        try:
            # Try moviepy v2 first
            return TextClip(
                text=text,
                font=f,
                font_size=font_size,
                color=color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                size=size,
                method=method
            )
        except TypeError:
            # If it's a TypeError, maybe it's moviepy v1 signature (arguments: txt, fontsize)
            try:
                return TextClip(
                    txt=text,
                    font=f,
                    fontsize=font_size,
                    color=color,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    size=size,
                    method=method
                )
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
            
    if last_err:
        raise last_err
    raise RuntimeError("Failed to create TextClip with all font fallbacks.")

# ----------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------

def download_file(url, filename):
    """Скачивание файла по URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    with open(filename, 'wb') as f:
        f.write(response.content)

def search_program(program_name):
    """Поиск установленной в системе программы."""
    try: 
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def get_program_path(program_name):
    return search_program(program_name)

def clear_cuda_memory():
    """Очистка кэша CUDA и вызов сборщика мусора для предотвращения OOM."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def resolve_generation_params(config):
    """
    Разрешение параметров генерации на основе приоритетов:
    1. CLI overrides / render.* explicit values
    2. acceleration_preset
    3. model_preset
    4. default values
    """
    render_cfg = config.get_render_config()
    explicit_render = config.yaml_config.get('render', {})
    
    default_render = {
        'steps': 4,
        'guidance_scale': 0.0,
        'sampler': 'euler',
        'scheduler': 'simple'
    }
    
    model_preset_name = render_cfg.get('model_preset')
    model_preset = config.get_model_preset(model_preset_name) if model_preset_name else {}
    
    acc_mode = render_cfg.get('acceleration_mode')
    acc_preset = {}
    if acc_mode:
        preset_name = f"{acc_mode}_4step"  # дефолтное предположение
        rec_steps = model_preset.get('recommended_steps')
        if acc_mode == 'turbo':
            if rec_steps == 6:
                preset_name = "turbo_6step"
            else:
                preset_name = "turbo_4step"
        elif acc_mode == 'schnell':
            preset_name = "schnell_4step"
        elif acc_mode == 'distilled':
            preset_name = "distilled_6step"
        elif acc_mode == 'lcm':
            preset_name = "lcm_6step"
        elif acc_mode == 'lightning':
            preset_name = "lightning_4step"
        elif acc_mode == 'hyper':
            preset_name = "hyper_8step"
            
        acc_preset = config.get_acceleration_preset(preset_name)
        if not acc_preset:
            acc_preset = {}

    def get_param(key, default_val):
        # 1 & 2. CLI / Explicit render section in yaml
        if key in explicit_render and explicit_render[key] is not None:
            return explicit_render[key]
        # 3. Acceleration preset
        if key in acc_preset and acc_preset[key] is not None:
            return acc_preset[key]
        # 4. Model preset
        model_key = key
        if key == 'steps' and 'recommended_steps' in model_preset:
            model_key = 'recommended_steps'
        if key == 'guidance_scale' and 'guidance_scale' in model_preset:
            model_key = 'guidance_scale'
            
        if model_key in model_preset and model_preset[model_key] is not None:
            return model_preset[model_key]
        # 5. Default
        return default_val

    resolved = {
        'steps': get_param('steps', default_render['steps']),
        'guidance_scale': get_param('guidance_scale', default_render['guidance_scale']),
        'sampler': get_param('sampler', default_render['sampler']),
        'scheduler': get_param('scheduler', default_render['scheduler']),
        'model_preset': model_preset_name,
        'acceleration_mode': acc_mode
    }
    return resolved

# ----------------- БЭКЕНДЫ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ -----------------

def generate_comfyui_image(prompt, negative_prompt, output_path, width, height, config):
    """
    Генерация изображения через REST API ComfyUI с динамическим маппингом параметров.
    """
    comfyui_cfg = config.get_comfyui_config()
    render_cfg = config.get_render_config()
    
    comfyui_url = comfyui_cfg.get('url', 'http://127.0.0.1:8188').rstrip('/')
    
    model_preset_name = render_cfg.get('model_preset')
    model_preset = config.get_model_preset(model_preset_name) if model_preset_name else {}
    
    if model_preset_name:
        if model_preset.get('enabled') is False:
            raise ValueError(f"Model preset {model_preset_name} is disabled or workflow is missing.")
            
    # 1. Определение файла воркфлоу
    workflow_path = model_preset.get('workflow') or render_cfg.get('comfyui_workflow_path') or comfyui_cfg.get('workflow_path')
    if not workflow_path:
        raise RuntimeError("Workflow path is not specified in config.")
        
    full_workflow_path = os.path.join(os.getcwd(), workflow_path)
    if not os.path.exists(full_workflow_path):
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
        
    print(f"Loading workflow: {workflow_path}")
    with open(full_workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
        
    node_map = comfyui_cfg.get('node_map', {})
    
    # 2. Вычисление параметров генерации
    resolved = resolve_generation_params(config)
    steps = resolved.get('steps')
    guidance = resolved.get('guidance_scale')
    sampler = resolved.get('sampler')
    scheduler = resolved.get('scheduler')
    
    seed = render_cfg.get('seed', -1)
    if seed == -1:
        import random
        seed = random.randint(1, 1000000000000000)
        
    # Формируем словарь параметров для подстановки
    params = {
        'positive_prompt': prompt,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'seed': seed,
        'steps': steps,
        'cfg': guidance,
        'sampler': sampler,
        'scheduler': scheduler,
        'save_image_prefix': f"t2v_{int(time.time())}"
    }
    
    # Обработка LoRA
    lora_preset_name = render_cfg.get('lora_preset', 'none')
    if lora_preset_name and lora_preset_name != 'none':
        lora_preset = config.get_lora_preset(lora_preset_name)
        if lora_preset and lora_preset.get('enabled', False):
            params['lora_name'] = lora_preset.get('lora_name')
            lora_strength = render_cfg.get('lora_strength')
            if lora_strength is not None:
                params['lora_strength_model'] = lora_strength
                params['lora_strength_clip'] = lora_strength
            else:
                params['lora_strength_model'] = lora_preset.get('strength_model', 0.7)
                params['lora_strength_clip'] = lora_preset.get('strength_clip', 0.7)
        else:
            params['lora_strength_model'] = 0.0
            params['lora_strength_clip'] = 0.0
    else:
        params['lora_strength_model'] = 0.0
        params['lora_strength_clip'] = 0.0

    def set_workflow_value(wf, path, val):
        try:
            parts = path.split('.')
            current = wf
            for part in parts[:-1]:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            last = parts[-1]
            if last in ['width', 'height', 'steps', 'seed']:
                try:
                    val = int(val)
                except:
                    pass
            elif last in ['cfg', 'strength_model', 'strength_clip']:
                try:
                    val = float(val)
                except:
                    pass
            current[last] = val
            return True
        except Exception as e:
            print(f"Warning: Could not set path {path} to {val}: {e}")
            return False

    # 3. Подстановка параметров в воркфлоу по нод-мапе
    for key, val in params.items():
        if key in node_map:
            path = node_map[key]
            set_workflow_value(workflow, path, val)
            
    # 4. Отправка POST запроса на запуск
    p = {"prompt": workflow}
    try:
        response = requests.post(f"{comfyui_url}/prompt", json=p, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Could not connect to ComfyUI at {comfyui_url}")
        
    resp_data = response.json()
    prompt_id = resp_data.get('prompt_id')
    if not prompt_id:
        raise RuntimeError("ComfyUI response did not contain prompt_id")
        
    print(f"ComfyUI prompt queued: {prompt_id}")
    
    # 5. Опрос готовности
    timeout_sec = comfyui_cfg.get('timeout_sec', 300)
    poll_interval_sec = comfyui_cfg.get('poll_interval_sec', 2)
    start_time = time.time()
    
    history_data = None
    while time.time() - start_time < timeout_sec:
        try:
            history_resp = requests.get(f"{comfyui_url}/history/{prompt_id}", timeout=10)
            if history_resp.status_code == 200:
                history_json = history_resp.json()
                if prompt_id in history_json:
                    history_data = history_json[prompt_id]
                    break
        except Exception:
            pass
        time.sleep(poll_interval_sec)
        
    if not history_data:
        raise TimeoutError(f"ComfyUI generation timed out after {timeout_sec}s for prompt_id: {prompt_id}")
        
    status = history_data.get('status', {})
    if status.get('completed') is not True:
        messages = status.get('messages', [])
        raise RuntimeError(f"ComfyUI prompt run was not completed successfully. Messages: {messages}")
        
    # 6. Извлечение результатов
    outputs = history_data.get('outputs', {})
    image_info = None
    for node_id, output in outputs.items():
        if 'images' in output:
            for img in output['images']:
                image_info = img
                break
        if image_info:
            break
            
    if not image_info:
        raise RuntimeError(f"ComfyUI history did not contain generated images.")
        
    filename = image_info.get('filename')
    subfolder = image_info.get('subfolder', '')
    img_type = image_info.get('type', 'output')
    
    # 7. Скачивание картинки
    view_url = f"{comfyui_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
    try:
        img_resp = requests.get(view_url, timeout=30)
        img_resp.raise_for_status()
    except Exception:
        raise ConnectionError(f"Could not connect to ComfyUI at {comfyui_url}")
        
    with open(output_path, 'wb') as f:
        f.write(img_resp.content)
        
    print(f"Generated image saved to: {output_path}")
    return output_path


def generate_local_api_image(prompt, negative_prompt, output_path, width=512, height=512, url="http://127.0.0.1:8000/txt2img", params=None):
    """
    Генерация через кастомный локальный API. 
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height
    }
    if params:
        payload.update(params)
        if 'guidance_scale' in params:
            payload['cfg'] = params['guidance_scale']
            
    try:
        print(f"Sending generation request to local API: {url}...")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        # 1. Проверяем Content-Type на картинку
        content_type = response.headers.get('content-type', '').lower()
        if 'image/' in content_type:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
            
        # 2. Иначе пробуем распарсить JSON
        try:
            r = response.json()
            b64_data = None
            for key in ['image', 'images', 'base64', 'b64']:
                if key in r:
                    val = r[key]
                    if isinstance(val, list) and len(val) > 0:
                        b64_data = val[0]
                    elif isinstance(val, str):
                        b64_data = val
                    break
            
            if b64_data:
                if ',' in b64_data:
                    b64_data = b64_data.split(',')[1]
                image_bytes = base64.b64decode(b64_data)
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                return output_path
                
            img_url = None
            for key in ['url', 'image_url', 'link']:
                if key in r:
                    img_url = r[key]
                    break
            
            if img_url:
                print(f"Downloading image from returned URL: {img_url}")
                download_file(img_url, output_path)
                return output_path
                
        except json.JSONDecodeError:
            if len(response.content) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
                
        print("Could not parse image data from local API response.")
        return None
    except Exception as e:
        print(f"Failed to generate image via local API: {e}")
        return None


def generate_a1111(prompt, negative_prompt, output_path, width=512, height=512, url="http://127.0.0.1:7860/sdapi/v1/txt2img"):
    """Генерация изображения через API Automatic1111."""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 20,
        "width": width,
        "height": height,
        "cfg_scale": 7.0
    }
    try:
        print(f"Sending A1111 generation request to {url}...")
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            r = response.json()
            image_data = base64.b64decode(r['images'][0])
            with open(output_path, 'wb') as f:
                f.write(image_data)
            return output_path
        else:
            print(f"A1111 API returned error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Failed to connect to A1111 WebUI: {e}")
        return None


def get_image_from_folder(folder_path, scene_id, index):
    """Поиск изображения в папке по ID сцены или по порядку."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return None
        
    patterns = [
        f"scene_{scene_id}.png", f"scene_{scene_id}.jpg", f"scene_{scene_id}.jpeg",
        f"{scene_id}.png", f"{scene_id}.jpg", f"{scene_id}.jpeg",
        f"scene_{scene_id}.mp4", f"{scene_id}.mp4",
        f"scene_{scene_id}.mov", f"{scene_id}.mov",
        f"scene_{scene_id}.avi", f"{scene_id}.avi",
        f"scene_{scene_id}.mkv", f"{scene_id}.mkv"
    ]
    for pattern in patterns:
        path = os.path.join(folder_path, pattern)
        if os.path.exists(path):
            return path
            
    # Если точного совпадения нет, ищем медиафайлы по списку расширений
    valid_exts = ('.png', '.jpg', '.jpeg', '.mp4', '.mov', '.avi', '.mkv')
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_exts)])
    if files:
        filename = files[index % len(files)]
        path = os.path.join(folder_path, filename)
        print(f"Using local media file: {path}")
        return path
    return None


def is_video_file(path):
    return path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


def download_pexels_video(query, output_path, orientation_landscape=True):
    """Поиск и скачивание видео с Pexels."""
    pexels_key = os.getenv('PEXELS_API_KEY')
    if not pexels_key:
        print("Warning: PEXELS_API_KEY not found. stock_video backend unavailable. Trying fallback backend.")
        return None
        
    try:
        from utility.video.background_video_generator import getBestVideo
        print(f"Searching Pexels video for query: {query}...")
        url = getBestVideo(query, orientation_landscape=orientation_landscape)
        if url:
            print(f"Downloading Pexels video: {url}...")
            download_file(url, output_path)
            return output_path
        else:
            print(f"No video found on Pexels for query: {query}")
        return None
    except Exception as e:
        print(f"Pexels video download failed: {e}")
        return None


def generate_image_by_backend(backend, prompt, negative_prompt, output_path, width, height, scene_id, index, config):
    """Маршрутизация генерации на основе выбранного бэкенда."""
    render_cfg = config.get_render_config()
    local_image_api_url = render_cfg.get('local_image_api_url')
    image_folder_path = render_cfg.get('image_folder_path', 'input_images')
    aspect_ratio = config.get_aspect_ratio()
    
    if backend == 'comfyui':
        return generate_comfyui_image(prompt, negative_prompt, output_path, width, height, config)
    elif backend == 'local_api':
        params = resolve_generation_params(config)
        return generate_local_api_image(prompt, negative_prompt, output_path, width, height, local_image_api_url, params)
    elif backend == 'a1111':
        return generate_a1111(prompt, negative_prompt, output_path, width, height, local_image_api_url)
    elif backend == 'image_folder':
        img_path = get_image_from_folder(image_folder_path, scene_id, index)
        if img_path:
            if is_video_file(img_path):
                return img_path
            import shutil
            root, ext = os.path.splitext(output_path)
            image_output_path = output_path if ext.lower() in ('.png', '.jpg', '.jpeg') else root + ".png"
            shutil.copy2(img_path, image_output_path)
            return image_output_path
        return None
    elif backend == 'stock_video':
        return download_pexels_video(prompt, output_path, orientation_landscape=(aspect_ratio == "16:9"))
    elif backend == 'none':
        return None
    else:
        print(f"Unknown backend requested: {backend}")
        return None

# ----------------- ОБРАБОТКА РАЗМЕРОВ И ЭФФЕКТОВ -----------------

def resize_and_crop_clip(clip, target_w, target_h):
    """Масштабирование и кадрирование клипа под целевое разрешение без искажений и черных полос."""
    clip_w, clip_h = clip.size
    
    scale_w = target_w / clip_w
    scale_h = target_h / clip_h
    scale = max(scale_w, scale_h)
    
    resized = resize_clip(clip, scale)
    resized_w, resized_h = resized.size
    
    x1 = (resized_w - target_w) / 2
    y1 = (resized_h - target_h) / 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    
    cropped = crop_clip(resized, x1=x1, y1=y1, x2=x2, y2=y2)
    return cropped

def apply_motion_preset(clip, motion_preset, target_w, target_h):
    """Применение Ken Burns эффектов к ImageClip."""
    if not isinstance(clip, ImageClip):
        return set_clip_position(clip, "center")
        
    duration = clip.duration
    
    if motion_preset == 'slow_zoom_in':
        zoom_clip = resize_clip(clip, lambda t: 1.0 + 0.08 * (t / duration))
        return set_clip_position(zoom_clip, "center")
        
    elif motion_preset == 'slow_zoom_out':
        zoom_clip = resize_clip(clip, lambda t: 1.08 - 0.08 * (t / duration))
        return set_clip_position(zoom_clip, "center")
        
    elif motion_preset == 'pan_left':
        panned = resize_clip(clip, width=int(target_w * 1.2))
        return set_clip_position(panned, lambda t: (int(-0.2 * target_w * (1.0 - t / duration)), "center"))
        
    elif motion_preset == 'pan_right':
        panned = resize_clip(clip, width=int(target_w * 1.2))
        return set_clip_position(panned, lambda t: (int(-0.2 * target_w * (t / duration)), "center"))
        
    elif motion_preset == 'handheld_light':
        shaked = resize_clip(clip, width=int(target_w * 1.1))
        def shake_pos(t):
            x = int(-0.05 * target_w + 0.015 * target_w * math.sin(2 * math.pi * t * 1.2))
            y = int(-0.05 * target_h + 0.015 * target_h * math.cos(2 * math.pi * t * 0.9))
            return (x, y)
        return set_clip_position(shaked, shake_pos)
        
    else:
        return set_clip_position(clip, "center")

# ----------------- ОСНОВНОЙ РЕНДЕР -----------------

def render_video_from_scenes(audio_file_path, scenes_json_path, output_mp4_path):
    config = get_config()
    aspect_ratio = config.get_aspect_ratio()
    
    render_cfg = config.get_render_config()
    visual_backend = render_cfg.get('visual_backend', 'none').lower()
    fallback_backend = render_cfg.get('fallback_backend', 'image_folder').lower()
    fallback_to_black = render_cfg.get('fallback_to_black', True)
    default_motion_preset = render_cfg.get('motion_preset', 'slow_zoom_in')
    
    # 1. Разрешение итогового видео
    target_w = int(render_cfg.get("final_width", 0))
    target_h = int(render_cfg.get("final_height", 0))

    if not target_w or not target_h:
        if aspect_ratio == "9:16":
            target_w, target_h = 1080, 1920
        elif aspect_ratio == "16:9":
            target_w, target_h = 1920, 1080
        elif aspect_ratio == "1:1":
            target_w, target_h = 1080, 1080
        else:
            target_w, target_h = 1080, 1920

    # 2. Вычисление параметров генерации по приоритетам
    resolved = resolve_generation_params(config)
    steps = resolved.get('steps')
    guidance = resolved.get('guidance_scale')
    sampler = resolved.get('sampler')
    scheduler = resolved.get('scheduler')
    model_preset_name = resolved.get('model_preset', 'none')
    
    model_preset = config.get_model_preset(model_preset_name) if model_preset_name else {}
    quantization = model_preset.get('quantization', 'none')
    
    # Разрешение генерации картинок
    image_width = render_cfg.get('image_width', 576)
    image_height = render_cfg.get('image_height', 1024)
    
    # Автоматическая корректировка разрешения генерации под соотношение сторон, если используются дефолтные значения
    if aspect_ratio == "16:9" and image_width == 576 and image_height == 1024:
        image_width, image_height = 1024, 576
    elif aspect_ratio == "1:1" and image_width == 576 and image_height == 1024:
        image_width, image_height = 768, 768

    # Выводим подробное стартовое логирование параметров
    print("\n" + "="*80)
    print("STARTING RENDER ENGINE WITH PARAMETERS:")
    print(f"Selected visual backend: {visual_backend}")
    print(f"Selected model preset: {model_preset_name}")
    print(f"Model quantization: {quantization}")
    print(f"Acceleration mode: {resolved.get('acceleration_mode', 'none')}")
    print(f"Generation size: {image_width}x{image_height}")
    print(f"Final video size: {target_w}x{target_h}")
    print(f"Steps: {steps}")
    print(f"Guidance scale: {guidance}")
    print(f"Sampler: {sampler}")
    print(f"Scheduler: {scheduler}")
    print(f"LoRA preset: {render_cfg.get('lora_preset', 'none')}")
    print(f"Fallback backend: {fallback_backend}")
    
    pexels_status = "enabled" if os.getenv('PEXELS_API_KEY') else "disabled"
    print(f"Stock video backend: {pexels_status}")
    print("="*80 + "\n")

    magick_path = get_program_path("magick")
    if magick_path:
        os.environ['IMAGEMAGICK_BINARY'] = magick_path
    else:
        os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'
        
    with open(scenes_json_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
        
    project_dir = os.path.dirname(scenes_json_path)
    images_dir = os.path.join(project_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)
    
    visual_clips = []
    current_time = 0.0
    
    for idx, scene in enumerate(scenes):
        scene_id = scene.get('scene_id', idx + 1)
        duration = float(scene.get('duration', 4.0))
        prompt = scene.get('prompt', '')
        neg_prompt = scene.get('negative_prompt', 'blurry, low quality')
        motion_preset = scene.get('camera_motion', default_motion_preset)
        
        # Определяем промпт/запрос для стоков
        stock_query = scene.get('stock_query') or prompt
        backend_prompt = stock_query if visual_backend == 'stock_video' else prompt
        
        visual_path = None
        out_img = os.path.join(images_dir, f"scene_{scene_id}.png")
        
        # --- Tier 1: Primary Backend ---
        if visual_backend != 'none':
            try:
                print(f"\n--- Scene {scene_id}: Generating image with primary backend '{visual_backend}' ---")
                if visual_backend == 'stock_video':
                    print(f"Using stock query: {stock_query}")
                    actual_out_path = os.path.join(images_dir, f"scene_{scene_id}.mp4")
                else:
                    actual_out_path = out_img
                    
                visual_path = generate_image_by_backend(
                    visual_backend, backend_prompt, neg_prompt, actual_out_path, 
                    image_width, image_height, scene_id, idx, config
                )
            except Exception as e:
                err_msg = str(e)
                print(f"Error on primary backend '{visual_backend}': {err_msg}")
                if any(marker in err_msg for marker in ["CUDA out of memory", "out of memory", "CUDA error", "CUBLAS"]):
                    print("\n" + "="*80)
                    print("CUDA OOM detected. Reduce image_width/image_height, use GGUF/FP8 workflow, or switch to image_folder/local_api/stock_video fallback.")
                    print("="*80 + "\n")
                visual_path = None
                
        # --- Tier 2: Fallback Backend ---
        if (not visual_path or not os.path.exists(visual_path)) and visual_backend != 'none':
            clear_cuda_memory()
            if fallback_backend and fallback_backend != visual_backend and fallback_backend != 'none':
                try:
                    print(f"Attempting fallback backend '{fallback_backend}'...")
                    
                    fallback_out_path = os.path.join(images_dir, f"scene_{scene_id}_fallback.png")
                    fallback_prompt = prompt
                    if fallback_backend == 'stock_video':
                        fallback_out_path = os.path.join(images_dir, f"scene_{scene_id}_fallback.mp4")
                        fallback_prompt = stock_query
                        print(f"Using stock query: {stock_query}")
                        
                    visual_path = generate_image_by_backend(
                        fallback_backend, fallback_prompt, neg_prompt, fallback_out_path,
                        image_width, image_height, scene_id, idx, config
                    )
                except Exception as e:
                    err_msg = str(e)
                    print(f"Error on fallback backend '{fallback_backend}': {err_msg}")
                    if any(marker in err_msg for marker in ["CUDA out of memory", "out of memory", "CUDA error", "CUBLAS"]):
                        print("\n" + "="*80)
                        print("CUDA OOM detected. Reduce image_width/image_height, use GGUF/FP8 workflow, or switch to image_folder/local_api/stock_video fallback.")
                        print("="*80 + "\n")
                    visual_path = None
                    
        # --- Post generation cleanup / garbage collection ---
        if render_cfg.get('clear_cuda_cache_between_scenes', True):
            clear_cuda_memory()
            
        # --- Tier 3: Black Screen Fallback ---
        scene_clip = None
        if visual_path and os.path.exists(visual_path):
            if visual_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                try:
                    video_clip = VideoFileClip(visual_path)
                    if video_clip.duration < duration:
                        from moviepy.video.fx.loop import loop
                        video_clip = loop(video_clip, duration=duration)
                    else:
                        video_clip = video_clip.subclip(0, duration)
                    scene_clip = resize_and_crop_clip(video_clip, target_w, target_h)
                except Exception as e:
                    print(f"Error loading video clip {visual_path}: {e}")
                    scene_clip = None
            else:
                try:
                    img_clip = set_clip_duration(ImageClip(visual_path), duration)
                    cropped_img = resize_and_crop_clip(img_clip, target_w, target_h)
                    scene_clip = apply_motion_preset(cropped_img, motion_preset, target_w, target_h)
                except Exception as e:
                    print(f"Error loading image clip {visual_path}: {e}")
                    scene_clip = None
                    
        if scene_clip is None:
            if fallback_to_black or visual_backend == 'none':
                print(f"Rendering black screen for scene {scene_id} as fallback.")
                scene_clip = set_clip_duration(ColorClip(size=(target_w, target_h), color=(0, 0, 0)), duration)
                scene_clip = set_clip_position(scene_clip, "center")
            else:
                raise RuntimeError(f"Visual asset generation failed for scene {scene_id} and fallback_to_black is disabled.")
            
        scene_clip = set_clip_start(scene_clip, current_time)
        visual_clips.append(scene_clip)
        
        # Subtitles / Captions processing
        if config.get_captions_enabled():
            subtitle_text = scene.get('subtitle', '')
            if subtitle_text:
                font_size = config.get_caption_font_size()
                font_color = config.get_caption_font_color()
                stroke_width = config.get_caption_stroke_width()
                stroke_color = config.get_caption_stroke_color()
                font_face = config.get_caption_font_face()
                caption_position = config.get_caption_position()
                
                if caption_position == 'bottom_center':
                    position = ("center", int(target_h * 0.8))
                elif caption_position == 'bottom_left':
                    position = (int(target_w * 0.05), int(target_h * 0.8))
                elif caption_position == 'bottom_right':
                    position = ("right", int(target_h * 0.8))
                elif caption_position == 'top':
                    position = ("center", int(target_h * 0.15))
                elif caption_position == 'center':
                    position = ("center", "center")
                else:
                    position = ("center", int(target_h * 0.8))
                    
                try:
                    text_clip = create_text_clip(
                        text=subtitle_text,
                        font=font_face,
                        font_size=font_size,
                        color=font_color,
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        size=(int(target_w * 0.9), None),
                        method="caption"
                    )
                    text_clip = set_clip_start(text_clip, current_time)
                    text_clip = set_clip_duration(text_clip, duration)
                    text_clip = set_clip_position(text_clip, position)
                    visual_clips.append(text_clip)
                except Exception as e:
                    print(f"Error creating subtitle TextClip: {e}")
                    
        current_time += duration
        
    final_video = CompositeVideoClip(visual_clips, size=(target_w, target_h))
    
    audio_clips = []
    if os.path.exists(audio_file_path):
        audio_file_clip = AudioFileClip(audio_file_path)
        audio_clips.append(audio_file_clip)
        
        final_audio = CompositeAudioClip(audio_clips)
        final_video = set_clip_audio(final_video, final_audio)
        final_video.duration = final_audio.duration
    else:
        print(f"Warning: Audio file not found at {audio_file_path}. Video will be silent.")
        final_video.duration = current_time
        
    print(f"Rendering final video to: {output_mp4_path}...")
    final_video.write_videofile(
        output_mp4_path,
        codec='libx264',
        audio_codec='aac',
        fps=25,
        preset='veryfast'
    )
    
    final_video.close()
    if audio_clips:
        audio_clips[0].close()
        
    print(f"Rendering completed successfully. Final video saved at {output_mp4_path}")
    return output_mp4_path


def get_output_media(audio_file_path, timed_captions, background_video_data, video_server):
    config = get_config()
    project_name = config.get_project_name()
    output_dir = os.path.join("output", project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    scenes = []
    for idx, ((t1, t2), text) in enumerate(timed_captions):
        prompt = ""
        for (vt1, vt2), url in background_video_data:
            if abs(vt1 - t1) < 0.5:
                prompt = url
                break
                
        scenes.append({
            "scene_id": idx + 1,
            "text": text,
            "duration": t2 - t1,
            "prompt": prompt or "stock video",
            "negative_prompt": "",
            "camera_motion": "none",
            "subtitle": text,
            "visual_type": "stock_keywords",
            "style_preset": ""
        })
        
    scenes_path = os.path.join(output_dir, "legacy_scenes.json")
    with open(scenes_path, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
        
    final_output = os.path.join(output_dir, "rendered_video.mp4")
    config.yaml_config.setdefault('render', {})['visual_backend'] = 'image_folder'
    
    render_video_from_scenes(audio_file_path, scenes_path, final_output)
    
    import shutil
    try:
        shutil.copy2(final_output, "rendered_video.mp4")
    except Exception as e:
        print(f"Could not copy rendered video to root: {e}")
        
    return "rendered_video.mp4"
