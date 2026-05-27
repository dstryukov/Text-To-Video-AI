import os
import argparse
import json
import wave
import shutil
try:
    from moviepy.editor import AudioFileClip, concatenate_audioclips
except ImportError:
    from moviepy import AudioFileClip, concatenate_audioclips
from utility.config import get_config
from utility.script.script_generator import generate_script
from utility.tts.tts_engine import generate_tts
from utility.video.visual_prompt_generator import generate_visual_prompts
from utility.render.render_engine import render_video_from_scenes
from utility.memory import clear_memory

def main():
    parser = argparse.ArgumentParser(description="Universal AI Video Generation Pipeline")
    parser.add_argument("topic_or_file", type=str, nargs='?', default=None, 
                        help="Topic for script generation OR path to script text/JSON scenes file")
    parser.add_argument("--script_file", type=str, default=None, 
                        help="Explicit path to a text file containing the script or JSON containing scenes")
    parser.add_argument("--preset", type=str, default=None, 
                        help="Override style preset (cinematic_realistic, documentary, etc.)")
    parser.add_argument("--backend", type=str, default=None, 
                        help="Override visual backend (none, comfyui, local_api, image_folder)")
    parser.add_argument("--aspect", type=str, default=None, 
                        help="Override aspect ratio (9:16, 16:9, 1:1)")
    
    # Новые CLI-аргументы
    parser.add_argument("--model-preset", type=str, default=None, 
                        help="Override model preset (flux_schnell_fp8, z_image_turbo_gguf, etc.)")
    parser.add_argument("--acceleration", type=str, default=None, 
                        help="Override acceleration mode (schnell, turbo, distilled, etc.)")
    parser.add_argument("--image-width", type=int, default=None, 
                        help="Override image generation width (e.g. 576)")
    parser.add_argument("--image-height", type=int, default=None, 
                        help="Override image generation height (e.g. 1024)")
    parser.add_argument("--steps", type=int, default=None, 
                        help="Override generation steps")
    parser.add_argument("--guidance", type=float, default=None, 
                        help="Override guidance scale")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Override seed (-1 for random)")
    parser.add_argument("--sampler", type=str, default=None, 
                        help="Override sampler name (e.g. euler)")
    parser.add_argument("--scheduler", type=str, default=None, 
                        help="Override scheduler name (e.g. simple)")
    parser.add_argument("--lora-preset", type=str, default=None, 
                        help="Override LoRA preset (none, cinematic, photoreal, anime)")
    parser.add_argument("--lora-strength", type=float, default=None, 
                        help="Override LoRA strength")
    parser.add_argument("--hf-token-env", type=str, default=None, 
                        help="Override Hugging Face token environment variable name")
    parser.add_argument("--fallback-backend", type=str, default=None, 
                        help="Override fallback visual backend (e.g. image_folder)")
    parser.add_argument("--final-width", type=int, default=None, 
                        help="Override final video width (e.g. 1080)")
    parser.add_argument("--final-height", type=int, default=None, 
                        help="Override final video height (e.g. 1920)")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to custom config YAML file to load")
    parser.add_argument("--tts-backend", type=str, default=None, 
                        help="Override TTS backend (f5_tts, fish_speech, cosyvoice, local_tts_api, silero, audio_file, none)")
    parser.add_argument("--tts-mode", type=str, default=None, 
                        help="Override TTS mode (per_scene, full_script)")
    parser.add_argument("--voice", type=str, default=None, 
                        help="Override TTS voice name or path")
    parser.add_argument("--reference-audio", type=str, default=None, 
                        help="Override reference audio path for voice cloning")
    parser.add_argument("--reference-text", type=str, default=None, 
                        help="Override reference audio text transcription")
    parser.add_argument("--tts-speed", type=str, default=None, 
                        help="Override TTS speed (float or string preset)")
    parser.add_argument("--emotion", type=str, default=None, 
                        help="Override TTS emotion (neutral, energetic, calm, etc.)")
    parser.add_argument("--tts-quality", type=str, default=None,
                        choices=["draft", "balanced", "high"],
                        help="Override TTS quality preset (draft, balanced, high)")
    parser.add_argument("--clear-memory-after-tts", dest="clear_memory_after_tts",
                        action="store_true", default=None,
                        help="Clear GPU/system memory after TTS generation")
    parser.add_argument("--no-clear-memory-after-tts", dest="clear_memory_after_tts",
                        action="store_false",
                        help="Do not clear memory after TTS generation")
    parser.add_argument("--unload-tts-model", dest="unload_tts_model",
                        action="store_true", default=None,
                        help="Unload TTS model from memory after generation")
    parser.add_argument("--no-unload-tts-model", dest="unload_tts_model",
                        action="store_false",
                        help="Keep TTS model in memory after generation")
    
    args = parser.parse_args()
    
    # 1. Загрузка конфигурации
    config = get_config()
    
    # Если передан кастомный файл конфигурации, загружаем его поверх дефолтного
    if args.config:
        print(f"Loading custom configuration file: {args.config}")
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f) or {}
                config.yaml_config.update(custom_config)
        except Exception as e:
            print(f"Error loading custom config {args.config}: {e}")
    
    # Применение переопределений командной строки в кэш конфигурации
    if args.preset:
        config.yaml_config.setdefault('visual_generator', {})['style_preset'] = args.preset
    if args.backend:
        config.yaml_config.setdefault('render', {})['visual_backend'] = args.backend
    if args.aspect:
        config.yaml_config['aspect_ratio'] = args.aspect
    if args.model_preset:
        config.yaml_config.setdefault('render', {})['model_preset'] = args.model_preset
    if args.acceleration:
        config.yaml_config.setdefault('render', {})['acceleration_mode'] = args.acceleration
    if args.image_width is not None:
        config.yaml_config.setdefault('render', {})['image_width'] = args.image_width
    if args.image_height is not None:
        config.yaml_config.setdefault('render', {})['image_height'] = args.image_height
    if args.steps is not None:
        config.yaml_config.setdefault('render', {})['steps'] = args.steps
    if args.guidance is not None:
        config.yaml_config.setdefault('render', {})['guidance_scale'] = args.guidance
    if args.seed is not None:
        config.yaml_config.setdefault('render', {})['seed'] = args.seed
    if args.sampler:
        config.yaml_config.setdefault('render', {})['sampler'] = args.sampler
    if args.scheduler:
        config.yaml_config.setdefault('render', {})['scheduler'] = args.scheduler
    if args.lora_preset:
        config.yaml_config.setdefault('render', {})['lora_preset'] = args.lora_preset
    if args.lora_strength is not None:
        config.yaml_config.setdefault('render', {})['lora_strength'] = args.lora_strength
    if args.hf_token_env:
        config.yaml_config.setdefault('huggingface', {})['token_env'] = args.hf_token_env
    if args.fallback_backend:
        config.yaml_config.setdefault('render', {})['fallback_backend'] = args.fallback_backend
    if args.final_width is not None:
        config.yaml_config.setdefault('render', {})['final_width'] = args.final_width
    if args.final_height is not None:
        config.yaml_config.setdefault('render', {})['final_height'] = args.final_height
        
    # CLI Overrides for TTS
    if args.tts_backend:
        config.yaml_config.setdefault('tts', {})['backend'] = args.tts_backend
    if args.tts_mode:
        config.yaml_config.setdefault('tts', {})['mode'] = args.tts_mode
    if args.voice:
        config.yaml_config.setdefault('tts', {})['voice'] = args.voice
    if args.reference_audio:
        config.yaml_config.setdefault('tts', {})['reference_audio_path'] = args.reference_audio
    if args.reference_text:
        config.yaml_config.setdefault('tts', {})['reference_text'] = args.reference_text
    if args.tts_speed is not None:
        try:
            speed_val = float(args.tts_speed)
        except ValueError:
            speed_val = args.tts_speed
        config.yaml_config.setdefault('tts', {})['speed'] = speed_val
    if args.emotion:
        config.yaml_config.setdefault('tts', {})['emotion'] = args.emotion
    if args.tts_quality:
        config.yaml_config.setdefault('tts', {})['quality_preset'] = args.tts_quality
    if args.clear_memory_after_tts is not None:
        config.yaml_config.setdefault('tts', {})['clear_memory_after_tts'] = args.clear_memory_after_tts
    if args.unload_tts_model is not None:
        config.yaml_config.setdefault('tts', {})['unload_tts_model_after_generation'] = args.unload_tts_model
        
    project_name = config.get_project_name()
    output_dir = os.path.join("output", project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Starting Pipeline for Project: {project_name} ---")
    print(f"Output directory: {output_dir}")
    print(f"Aspect ratio: {config.get_aspect_ratio()}")
    
    render_cfg = config.get_render_config()
    print(f"Visual backend: {render_cfg.get('visual_backend')}")
    print(f"Model preset: {render_cfg.get('model_preset')}")
    print(f"Style preset: {config.get_visual_generator_config().get('style_preset')}")
    
    # 2. Определение входных данных (сценарий или тема)
    input_source = args.topic_or_file
    script_file_path = args.script_file
    
    script_content = ""
    predefined_scenes = None
    
    # Если первый аргумент является существующим файлом
    if input_source and os.path.exists(input_source):
        script_file_path = input_source
        input_source = None
        
    if script_file_path:
        print(f"Reading script from file: {script_file_path}")
        if script_file_path.endswith('.json'):
            with open(script_file_path, 'r', encoding='utf-8') as f:
                predefined_scenes = json.load(f)
        else:
            with open(script_file_path, 'r', encoding='utf-8') as f:
                script_content = f.read().strip()
    elif input_source:
        print(f"Generating script for topic: '{input_source}'...")
        script_content = generate_script(input_source)
        # Сохраняем сгенерированный текст сценария
        script_txt_path = os.path.join(output_dir, "generated_script.txt")
        with open(script_txt_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"Script saved to: {script_txt_path}")
    else:
        # Дефолтная тема, если ничего не передано
        default_topic = "Интересные факты о космосе"
        print(f"No topic or file provided. Using default topic: '{default_topic}'")
        script_content = generate_script(default_topic)
        script_txt_path = os.path.join(output_dir, "generated_script.txt")
        with open(script_txt_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
    # 3. Генерация scenes.json (разбивка на сцены и промпты)
    scenes_json_path = os.path.join(output_dir, "scenes.json")
    if predefined_scenes:
        print("Using predefined scenes JSON structure.")
        scenes = generate_visual_prompts(predefined_scenes, scenes_json_path)
    else:
        scenes = generate_visual_prompts(script_content, scenes_json_path)
        
    # 4. Генерация озвучки
    print("\n--- Generating Audio (TTS) ---")
    tts_config = config.get_tts_config()
    audio_dir = os.path.join(output_dir, "audio_scenes")
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_ext = tts_config.get('format', 'wav')
    sample_rate = int(tts_config.get('sample_rate', 24000))
    tts_mode = tts_config.get('mode', 'per_scene')
    backend = tts_config.get('backend', 'silero').lower()
    
    # Режим audio_file автоматически переходит в full_script логику
    if backend == 'audio_file':
        tts_mode = 'full_script'
        
    master_audio_path = os.path.join(output_dir, f"audio_tts.{audio_ext}")
    
    if tts_mode == 'full_script':
        print(f"TTS mode: full_script (Backend: {backend})")
        # 1. Объединяем весь текст сценария
        combined_text = " ".join(scene.get('text', '') for scene in scenes)
        
        # 2. Генерируем единый мастер-трек
        generate_tts(combined_text, master_audio_path, tts_config)
        
        # 3. Измеряем его длительность
        total_duration = 0.0
        try:
            if audio_ext == 'wav':
                with wave.open(master_audio_path, 'rb') as wav_f:
                    frames = wav_f.getnframes()
                    rate = wav_f.getframerate()
                    total_duration = frames / float(rate)
            else:
                audio_clip = AudioFileClip(master_audio_path)
                total_duration = audio_clip.duration
                audio_clip.close()
        except Exception as e:
            print(f"Warning: Could not read precise audio duration for master track: {e}. Estimating.")
            total_duration = max(5.0, len(combined_text.split()) / 2.5)
            
        print(f"Master audio track ready. Total duration: {total_duration:.2f}s")
        
        # 4. Распределяем длительность сцен пропорционально количеству слов
        total_words = sum(len(scene.get('text', '').split()) for scene in scenes)
        if total_words == 0:
            total_words = 1
            
        for idx, scene in enumerate(scenes):
            scene_text = scene.get('text', '')
            words = len(scene_text.split())
            ratio = words / total_words
            duration = total_duration * ratio
            if duration < 1.0:
                duration = 1.0
            scene['duration'] = duration
            print(f"Scene {scene.get('scene_id', idx+1)} word count: {words}, proportional duration: {duration:.2f}s")
            
        # Записываем уточненные длительности в scenes.json
        with open(scenes_json_path, 'w', encoding='utf-8') as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)
            
    else:
        print(f"TTS mode: per_scene (Backend: {backend})")
        audio_paths = []
        for idx, scene in enumerate(scenes):
            scene_id = scene.get('scene_id', idx + 1)
            scene_text = scene.get('text', '')
            
            scene_audio_filename = f"scene_{scene_id}.{audio_ext}"
            scene_audio_path = os.path.join(audio_dir, scene_audio_filename)
            
            # Синтезируем аудио для каждой сцены
            generate_tts(scene_text, scene_audio_path, tts_config)
            audio_paths.append(scene_audio_path)
            
            # Измеряем длительность чанка
            duration = 4.0
            try:
                if audio_ext == 'wav':
                    with wave.open(scene_audio_path, 'rb') as wav_f:
                        frames = wav_f.getnframes()
                        rate = wav_f.getframerate()
                        duration = frames / float(rate)
                else:
                    audio_clip = AudioFileClip(scene_audio_path)
                    duration = audio_clip.duration
                    audio_clip.close()
            except Exception as e:
                print(f"Warning: Could not read precise audio duration for {scene_audio_filename}: {e}. Estimating.")
                duration = max(2.5, len(scene_text.split()) / 2.5)
                
            print(f"Scene {scene_id} audio duration: {duration:.2f}s")
            scene['duration'] = duration
            
        # Записываем уточненные длительности в scenes.json
        with open(scenes_json_path, 'w', encoding='utf-8') as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)
            
        # Объединяем отдельные аудиофайлы в один мастер-трек
        print(f"\nConcatenating audio clips into master track: {master_audio_path}")
        try:
            clips = [AudioFileClip(ap) for ap in audio_paths]
            final_audio = concatenate_audioclips(clips)
            final_audio.write_audiofile(master_audio_path, fps=sample_rate, logger=None)
            
            final_audio.close()
            for c in clips:
                c.close()
        except Exception as e:
            print(f"Error concatenating audio clips: {e}")
            if audio_paths:
                shutil.copy2(audio_paths[0], master_audio_path)

    # 5. Очистка памяти после TTS
    if tts_config.get('clear_memory_after_tts', True):
        print("\n--- Clearing memory after TTS stage ---")
        clear_memory()
            
    # 6. Запуск сборки видео (Рендеринг)
    print("\n--- Starting Video Assembly (Render Engine) ---")
    final_video_path = os.path.join(output_dir, "rendered_video.mp4")
    
    # Очищаем память перед визуальным рендерингом для освобождения VRAM
    clear_memory()
    
    render_video_from_scenes(master_audio_path, scenes_json_path, final_video_path)
    
    # Дублируем ролик в корень проекта для обратной совместимости
    try:
        shutil.copy2(final_video_path, "rendered_video.mp4")
        print("Final video copied to root directory as 'rendered_video.mp4'")
    except Exception as e:
        print(f"Warning: Could not copy video to root directory: {e}")
        
    print("\n--- Pipeline Execution Finished Successfully! ---")
    print(f"Final video: {final_video_path}")

if __name__ == "__main__":
    main()
