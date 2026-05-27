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

# Локальный кэш для пайплайна SDXL Turbo
_sdxl_pipeline = None

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

# ----------------- БЭКЕНДЫ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ -----------------

def generate_sdxl_turbo(prompt, negative_prompt, output_path, width=512, height=512):
    """Генерация изображения локально через SDXL Turbo."""
    global _sdxl_pipeline
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if _sdxl_pipeline is None:
            print("Loading SDXL Turbo pipeline on device:", device)
            _sdxl_pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None
            )
            _sdxl_pipeline.to(device)
            
        print(f"Generating image via SDXL Turbo for prompt: {prompt[:40]}...")
        image = _sdxl_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            width=width,
            height=height
        ).images[0]
        image.save(output_path)
        return output_path
    except Exception as e:
        print(f"SDXL Turbo generation failed: {e}")
        print("Please ensure 'diffusers', 'transformers' and 'accelerate' are installed.")
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

def generate_comfyui(prompt, negative_prompt, output_path, width=512, height=512, url="http://127.0.0.1:8188"):
    """Генерация изображения через API ComfyUI."""
    workflow_path = os.path.join(os.getcwd(), 'comfyui_workflow.json')
    workflow = None
    if os.path.exists(workflow_path):
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except Exception as e:
            print(f"Failed to read comfyui_workflow.json: {e}")
            
    if workflow is None:
        print("comfyui_workflow.json not found. ComfyUI request requires a configured workflow template.")
        return None
        
    try:
        for node_id, node in workflow.items():
            class_type = node.get('class_type', '')
            if class_type == 'CLIPTextEncode':
                inputs = node.get('inputs', {})
                if 'text' in inputs:
                    if 'negative' in str(node_id) or 'bad' in str(inputs.get('text', '')).lower():
                        node['inputs']['text'] = negative_prompt
                    else:
                        node['inputs']['text'] = prompt
            elif class_type == 'EmptyLatentImage':
                node['inputs']['width'] = width
                node['inputs']['height'] = height
                
        p = {"prompt": workflow}
        response = requests.post(f"{url}/prompt", json=p, timeout=20)
        if response.status_code == 200:
            print("ComfyUI prompt queued successfully. Waiting for output...")
            time.sleep(5)
            return None
        else:
            print(f"ComfyUI returned error: {response.text}")
            return None
    except Exception as e:
        print(f"Failed to connect to ComfyUI: {e}")
        return None

def generate_local_api_image(prompt, negative_prompt, output_path, width=512, height=512, url="http://127.0.0.1:8000/txt2img"):
    """Генерация через кастомный локальный API."""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        else:
            print(f"Local Image API returned error {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to connect to local image API: {e}")
        return None

def get_image_from_folder(folder_path, scene_id, index):
    """Поиск изображения в папке по ID сцены или по порядку."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return None
        
    patterns = [
        f"scene_{scene_id}.png", f"scene_{scene_id}.jpg", f"scene_{scene_id}.jpeg",
        f"{scene_id}.png", f"{scene_id}.jpg", f"{scene_id}.jpeg",
        f"scene_{scene_id}.mp4", f"{scene_id}.mp4"
    ]
    for pattern in patterns:
        path = os.path.join(folder_path, pattern)
        if os.path.exists(path):
            return path
            
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    valid_exts = ('.png', '.jpg', '.jpeg', '.mp4')
    media_files = [f for f in files if f.lower().endswith(valid_exts)]
    if media_files:
        filename = media_files[index % len(media_files)]
        return os.path.join(folder_path, filename)
    return None

def download_pexels_video(query, output_path, orientation_landscape=True):
    """Поиск и скачивание видео с Pexels."""
    try:
        from utility.video.background_video_generator import getBestVideo
        print(f"Searching Pexels video for query: {query}...")
        url = getBestVideo(query, orientation_landscape=orientation_landscape)
        if url:
            print(f"Downloading Pexels video: {url}...")
            download_file(url, output_path)
            return output_path
        return None
    except Exception as e:
        print(f"Pexels video download failed: {e}")
        return None

# ----------------- ОБРАБОТКА РАЗМЕРОВ И ЭФФЕКТОВ -----------------

def resize_and_crop_clip(clip, target_w, target_h):
    """Масштабирование и кадрирование клипа под целевое разрешение без искажений и черных полос."""
    clip_w, clip_h = clip.size
    
    scale_w = target_w / clip_w
    scale_h = target_h / clip_h
    scale = max(scale_w, scale_h)
    
    # Ресайз с сохранением пропорций
    resized = resize_clip(clip, scale)
    resized_w, resized_h = resized.size
    
    # Обрезаем из центра
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
    
    if aspect_ratio == "9:16":
        target_w, target_h = 1080, 1920
    elif aspect_ratio == "16:9":
        target_w, target_h = 1920, 1080
    elif aspect_ratio == "1:1":
        target_w, target_h = 1080, 1080
    else:
        target_w, target_h = 1080, 1920
        
    render_cfg = config.get_render_config()
    visual_backend = render_cfg.get('visual_backend', 'none').lower()
    default_motion_preset = render_cfg.get('motion_preset', 'slow_zoom_in')
    local_image_api_url = render_cfg.get('local_image_api_url')
    image_folder_path = render_cfg.get('image_folder_path', 'input_images')
    
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
        
        visual_path = None
        
        if visual_backend == 'sdxl_turbo':
            out_img = os.path.join(images_dir, f"scene_{scene_id}.png")
            visual_path = generate_sdxl_turbo(prompt, neg_prompt, out_img, target_w, target_h)
        elif visual_backend == 'a1111':
            out_img = os.path.join(images_dir, f"scene_{scene_id}.png")
            visual_path = generate_a1111(prompt, neg_prompt, out_img, target_w, target_h, local_image_api_url)
        elif visual_backend == 'comfyui':
            out_img = os.path.join(images_dir, f"scene_{scene_id}.png")
            visual_path = generate_comfyui(prompt, neg_prompt, out_img, target_w, target_h)
        elif visual_backend == 'local_api':
            out_img = os.path.join(images_dir, f"scene_{scene_id}.png")
            visual_path = generate_local_api_image(prompt, neg_prompt, out_img, target_w, target_h, local_image_api_url)
        elif visual_backend == 'image_folder':
            visual_path = get_image_from_folder(image_folder_path, scene_id, idx)
        elif visual_backend == 'stock_video':
            out_video = os.path.join(images_dir, f"scene_{scene_id}.mp4")
            visual_path = download_pexels_video(prompt, out_video, orientation_landscape=(aspect_ratio == "16:9"))
            
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
            scene_clip = set_clip_duration(ColorClip(size=(target_w, target_h), color=(0, 0, 0)), duration)
            scene_clip = set_clip_position(scene_clip, "center")
            
        scene_clip = set_clip_start(scene_clip, current_time)
        visual_clips.append(scene_clip)
        
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
                    text_clip = TextClip(
                        txt=subtitle_text,
                        font=font_face,
                        fontsize=font_size,
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
