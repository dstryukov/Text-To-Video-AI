import json
import os
import re
from utility.config import get_config
from utility.utils import log_response, LOG_TYPE_GPT

def generate_visual_prompts(script_or_scenes, output_path=None):
    """
    Генерирует scenes.json на основе сценария или списка сцен с использованием LLM.
    
    Args:
        script_or_scenes: Строка (сырой сценарий) или список словарей/сцен.
        output_path: Путь для сохранения результата (по умолчанию output/{project_name}/scenes.json)
    """
    config = get_config()
    vg_config = config.get_visual_generator_config()
    mode = vg_config.get('mode', 'image_prompts')
    style_preset_name = vg_config.get('style_preset', 'cinematic_realistic')
    style_preset_description = config.get_style_preset(style_preset_name)
    
    project_name = config.get_project_name()
    if output_path is None:
        output_dir = os.path.join("output", project_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "scenes.json")
        
    if isinstance(script_or_scenes, list):
        # Если передан готовый список сцен, просто форматируем его и пишем на диск без вызова LLM
        print("Using provided structured scenes list. Bypassing LLM.")
        scenes = []
        for idx, s in enumerate(script_or_scenes):
            scene = {}
            scene['scene_id'] = s.get('scene_id', idx + 1)
            scene['text'] = s.get('text', '')
            scene['duration'] = float(s.get('duration', 4.0))
            scene['prompt'] = s.get('prompt', 'default visualization')
            scene['negative_prompt'] = s.get('negative_prompt', 'blurry, low quality')
            scene['camera_motion'] = s.get('camera_motion', 'slow_zoom_in')
            if scene['camera_motion'] not in ['none', 'slow_zoom_in', 'slow_zoom_out', 'pan_left', 'pan_right', 'handheld_light']:
                scene['camera_motion'] = 'slow_zoom_in'
            scene['subtitle'] = s.get('subtitle', scene['text'])
            scene['visual_type'] = s.get('visual_type', mode)
            scene['style_preset'] = s.get('style_preset', style_preset_name)
            
            # Добавляем stock_query
            scene['stock_query'] = s.get('stock_query', '')
            if not scene['stock_query']:
                if mode == 'stock_keywords':
                    scene['stock_query'] = scene['prompt']
                else:
                    scene['stock_query'] = " ".join(scene['prompt'].split()[:4])
                    
            scenes.append(scene)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)
        return scenes

    client = config.get_llm_client()
    model = config.get_llm_model()
    provider = config.get_llm_provider()
    
    # Формируем системные инструкции
    system_prompt = f"""You are an expert AI video director. Your task is to analyze the provided video script (or structured scenes) and divide/refine it into sequential visual scenes.
For each scene, you must generate a structured JSON object containing visual details.

The style preset to use is: {style_preset_name}
The style description is: {style_preset_description}

The generation mode is: {mode}
(Instructions for mode:
- image_prompts: generate detailed, visually concrete English prompts optimized for image generation models (like Stable Diffusion).
- video_prompts: generate English prompts describing motion, action, and transitions optimized for video generation models (like Runway, Sora).
- stock_keywords: generate 3-4 descriptive, visually concrete English search terms/keywords (like "rainy street city night", "business meeting close up") to search for stock videos.)

Each scene in the JSON array must contain exactly these fields:
1. "scene_id": integer, starting from 1.
2. "text": string, the portion of the script spoken during this scene.
3. "duration": float, estimated duration in seconds (aim for ~2.5 to 3 words per second, minimum 2.0 seconds). The sum of all scene durations should cover the full script length.
4. "prompt": string, the generated visual prompt or keywords in English. It MUST incorporate the style description and be highly concrete.
5. "negative_prompt": string, negative prompt in English (e.g. "blurry, low quality, distorted" for image/video prompts, or empty for stock keywords).
6. "camera_motion": string, the camera motion preset to apply. Choose ONLY one from this list: none, slow_zoom_in, slow_zoom_out, pan_left, pan_right, handheld_light.
7. "subtitle": string, the subtitle text to display during this scene (should be matching or summarizing the text, short and readable).
8. "visual_type": string, matching the generation mode (e.g. "{mode}").
9. "style_preset": string, matching the style preset name (e.g. "{style_preset_name}").
10. "stock_query": string, a short 2-4 word visually concrete English search query for stock video search engines (e.g. "red planet surface", "smiling student"). If mode is stock_keywords, stock_query must match the prompt.

All scenes must be strictly consecutive and cover the entire script without overlaps or gaps.
Your response must be a single, valid JSON array containing the list of scene objects. Do not wrap it in any formatting other than standard json code block (or return raw json). Do not add any text before or after the JSON array.
"""

    user_content = f"Video Script:\n{script_or_scenes}"

    print(f"Generating visual prompts for style: {style_preset_name}, mode: {mode}...")
    
    max_retries = 3
    retry_count = 0
    raw_response = ""
    
    while retry_count < max_retries:
        try:
            if provider == 'gemini':
                response = client.generate_content(
                    contents=[
                        {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_content}"}]}
                    ],
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "max_output_tokens": 8192,
                    }
                )
                raw_response = response.text.strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                )
                raw_response = response.choices[0].message.content.strip()
            
            # Очистка markdown блоков
            cleaned_text = raw_response
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # Парсинг JSON
            scenes = json.loads(cleaned_text)
            
            # Валидация полей
            if not isinstance(scenes, list):
                raise ValueError("LLM response is not a JSON array")
                
            for idx, scene in enumerate(scenes):
                # Проверка и установка значений по умолчанию
                scene['scene_id'] = scene.get('scene_id', idx + 1)
                scene['text'] = scene.get('text', '')
                scene['duration'] = float(scene.get('duration', 4.0))
                scene['prompt'] = scene.get('prompt', 'default visualization')
                scene['negative_prompt'] = scene.get('negative_prompt', 'blurry, low quality')
                scene['camera_motion'] = scene.get('camera_motion', 'slow_zoom_in')
                if scene['camera_motion'] not in ['none', 'slow_zoom_in', 'slow_zoom_out', 'pan_left', 'pan_right', 'handheld_light']:
                    scene['camera_motion'] = 'slow_zoom_in'
                scene['subtitle'] = scene.get('subtitle', scene['text'])
                scene['visual_type'] = scene.get('visual_type', mode)
                scene['style_preset'] = scene.get('style_preset', style_preset_name)
                
                # Добавляем/проверяем stock_query
                scene['stock_query'] = scene.get('stock_query', '')
                if not scene['stock_query']:
                    if mode == 'stock_keywords':
                        scene['stock_query'] = scene['prompt']
                    else:
                        scene['stock_query'] = " ".join(scene['prompt'].split()[:4])
            
            # Сохранение в файл
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scenes, f, ensure_ascii=False, indent=2)
                
            log_response(LOG_TYPE_GPT, user_content, cleaned_text)
            print(f"Successfully generated visual prompts: {output_path}")
            return scenes
            
        except Exception as e:
            print(f"Error parsing LLM response for scenes (attempt {retry_count+1}/{max_retries}): {e}")
            retry_count += 1
            
    # Резервная заглушка, если LLM не смогла сгенерировать JSON после 3 попыток
    print("Warning: Using fallback scenes JSON structure.")
    fallback_scenes = []
    
    if isinstance(script_or_scenes, list):
        for idx, s in enumerate(script_or_scenes):
            prompt = s.get('prompt', 'abstract visual background')
            fallback_scenes.append({
                "scene_id": s.get('scene_id', idx + 1),
                "text": s.get('text', ''),
                "duration": float(s.get('duration', 5.0)),
                "prompt": prompt,
                "negative_prompt": "blurry, low quality",
                "camera_motion": "slow_zoom_in",
                "subtitle": s.get('subtitle', s.get('text', '')),
                "visual_type": mode,
                "style_preset": style_preset_name,
                "stock_query": s.get('stock_query') or " ".join(prompt.split()[:4])
            })
    else:
        # Разбиваем текст по предложениям
        sentences = re.split(r'(?<=[.!?])\s+', script_or_scenes)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            sentences = [script_or_scenes]
            
        for idx, sentence in enumerate(sentences):
            words = len(sentence.split())
            duration = max(2.5, words / 2.5)
            prompt = f"Scenic visual representing: {sentence[:30]}"
            fallback_scenes.append({
                "scene_id": idx + 1,
                "text": sentence,
                "duration": duration,
                "prompt": prompt,
                "negative_prompt": "blurry, low quality",
                "camera_motion": "slow_zoom_in",
                "subtitle": sentence,
                "visual_type": mode,
                "style_preset": style_preset_name,
                "stock_query": " ".join(prompt.split()[:4])
            })
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fallback_scenes, f, ensure_ascii=False, indent=2)
        
    return fallback_scenes
