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

def main():
    parser = argparse.ArgumentParser(description="Universal AI Video Generation Pipeline")
    parser.add_argument("topic_or_file", type=str, nargs='?', default=None, 
                        help="Topic for script generation OR path to script text/JSON scenes file")
    parser.add_argument("--script_file", type=str, default=None, 
                        help="Explicit path to a text file containing the script or JSON containing scenes")
    parser.add_argument("--preset", type=str, default=None, 
                        help="Override style preset (cinematic_realistic, documentary, etc.)")
    parser.add_argument("--backend", type=str, default=None, 
                        help="Override visual backend (none, sdxl_turbo, a1111, stock_video, etc.)")
    parser.add_argument("--aspect", type=str, default=None, 
                        help="Override aspect ratio (9:16, 16:9, 1:1)")
    
    args = parser.parse_args()
    
    # 1. Загрузка конфигурации
    config = get_config()
    
    # Применение переопределений командной строки в кэш конфигурации
    if args.preset:
        config.yaml_config.setdefault('visual_generator', {})['style_preset'] = args.preset
    if args.backend:
        config.yaml_config.setdefault('render', {})['visual_backend'] = args.backend
    if args.aspect:
        config.yaml_config['aspect_ratio'] = args.aspect
        
    project_name = config.get_project_name()
    output_dir = os.path.join("output", project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Starting Pipeline for Project: {project_name} ---")
    print(f"Output directory: {output_dir}")
    print(f"Aspect ratio: {config.get_aspect_ratio()}")
    print(f"Visual backend: {config.get_render_config().get('visual_backend')}")
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
        
    # 4. Генерация озвучки для каждой сцены
    # Мы генерируем аудио отдельно для каждой сцены, чтобы точно знать длительность звука.
    # Это гарантирует 100% совпадение звука и видео без использования Whisper/STT.
    print("\n--- Generating Audio (TTS) for scenes ---")
    tts_config = config.get_tts_config()
    audio_dir = os.path.join(output_dir, "audio_scenes")
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_paths = []
    sample_rate = int(tts_config.get('sample_rate', 24000))
    audio_ext = tts_config.get('format', 'wav')
    
    for idx, scene in enumerate(scenes):
        scene_id = scene.get('scene_id', idx + 1)
        scene_text = scene.get('text', '')
        
        scene_audio_filename = f"scene_{scene_id}.{audio_ext}"
        scene_audio_path = os.path.join(audio_dir, scene_audio_filename)
        
        # Синтез речи для сцены
        generate_tts(scene_text, scene_audio_path, tts_config)
        audio_paths.append(scene_audio_path)
        
        # Получаем реальную длительность сгенерированного файла
        duration = 4.0
        try:
            if audio_ext == 'wav':
                with wave.open(scene_audio_path, 'rb') as wav_f:
                    frames = wav_f.getnframes()
                    rate = wav_f.getframerate()
                    duration = frames / float(rate)
            else:
                # Для mp3 используем MoviePy
                audio_clip = AudioFileClip(scene_audio_path)
                duration = audio_clip.duration
                audio_clip.close()
        except Exception as e:
            print(f"Warning: Could not read precise audio duration for {scene_audio_filename}: {e}. Using words estimation.")
            duration = max(2.5, len(scene_text.split()) / 2.5)
            
        print(f"Scene {scene_id} audio duration: {duration:.2f}s")
        scene['duration'] = duration
        
    # Перезаписываем scenes.json с уточненными длительностями
    with open(scenes_json_path, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
        
    # 5. Объединение аудиофайлов в один мастер-трек
    master_audio_path = os.path.join(output_dir, f"audio_tts.{audio_ext}")
    print(f"\nConcatenating audio clips into master track: {master_audio_path}")
    
    try:
        clips = [AudioFileClip(ap) for ap in audio_paths]
        final_audio = concatenate_audioclips(clips)
        final_audio.write_audiofile(master_audio_path, fps=sample_rate, logger=None)
        
        # Закрываем ресурсы
        final_audio.close()
        for c in clips:
            c.close()
    except Exception as e:
        print(f"Error concatenating audio clips: {e}")
        # В случае сбоя берем первый попавшийся или создаем пустышку
        if audio_paths:
            shutil.copy2(audio_paths[0], master_audio_path)
            
    # 6. Запуск сборки видео (Рендеринг)
    print("\n--- Starting Video Assembly (Render Engine) ---")
    final_video_path = os.path.join(output_dir, "rendered_video.mp4")
    
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
