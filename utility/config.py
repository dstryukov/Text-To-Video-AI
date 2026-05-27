import os
from dotenv import load_dotenv
from typing import Optional, Literal
from openai import OpenAI

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ConfigurationError(Exception):
    pass


class Config:
    _instance: Optional['Config'] = None
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        load_dotenv()
        
        self.yaml_config = {}
        yaml_path = os.path.join(os.getcwd(), 'config.yaml')
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    self.yaml_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config.yaml: {e}")
        
        self._validate_env_file()
        self._validate_configuration()
        
        self._llm_client = None
        self._initialized = True
    
    def _validate_env_file(self) -> None:
        env_path = os.path.join(os.getcwd(), '.env')
        if not os.path.exists(env_path):
            yaml_path = os.path.join(os.getcwd(), 'config.yaml')
            if os.path.exists(yaml_path):
                print(f"Warning: .env file not found at {env_path}, but config.yaml is present. Continuing...")
                return
            raise ConfigurationError(
                ".env file not found. Please create a .env file based on .env.example\n"
                f"Expected location: {env_path}"
            )
    
    def _validate_configuration(self) -> None:
        errors = []
        
        llm_provider = self.get_llm_provider()
        if llm_provider not in ['openai', 'groq', 'gemini']:
            errors.append(
                f"Invalid LLM_PROVIDER: '{llm_provider}'. Must be one of: openai, groq, gemini"
            )
        
        render_cfg = self.get_render_config()
        vg_cfg = self.get_visual_generator_config()
        needs_pexels = (render_cfg.get('visual_backend') == 'stock_video' or vg_cfg.get('mode') == 'stock_keywords')
        if needs_pexels and not os.getenv('PEXELS_API_KEY'):
            errors.append("Missing required API key: PEXELS_API_KEY (required when visual_backend=stock_video or mode=stock_keywords)")
        
        stt_provider = self.get_stt_provider()
        if stt_provider not in ['whisper', 'deepgram']:
            errors.append(
                f"Invalid STT_PROVIDER: '{stt_provider}'. Must be one of: whisper, deepgram"
            )
        elif stt_provider == 'deepgram':
            if not os.getenv('DEEPGRAM_API_KEY'):
                errors.append("Missing required API key: DEEPGRAM_API_KEY (required for STT_PROVIDER=deepgram)")
        
        if errors:
            error_message = "Configuration validation failed:\n\n"
            for error in errors:
                error_message += f"  - {error}\n"
            error_message += "\nPlease check your config.yaml and .env file and ensure all required keys are set."
            raise ConfigurationError(error_message)
    
    def get_project_name(self) -> str:
        return self.yaml_config.get('project_name', 'space_facts')

    def get_aspect_ratio(self) -> str:
        return self.yaml_config.get('aspect_ratio', '9:16')

    def get_tts_config(self) -> dict:
        default_tts = {
            'backend': 'silero',
            'language': 'ru',
            'voice': 'xenia',
            'speed': 'medium',
            'sample_rate': 24000,
            'format': 'wav',
            'normalize_text': True,
            'local_api_url': 'http://localhost:8010/tts'
        }
        user_tts = self.yaml_config.get('tts', {})
        merged = {**default_tts, **user_tts}
        
        # Override with env variables if present
        if os.getenv('TTS_PROVIDER'):
            merged['backend'] = os.getenv('TTS_PROVIDER')
        if os.getenv('EDGETTS_VOICE') and merged['backend'] == 'edgetts':
            merged['voice'] = os.getenv('EDGETTS_VOICE')
        elif os.getenv('ELEVENLABS_VOICE_ID') and merged['backend'] == 'elevenlabs':
            merged['voice'] = os.getenv('ELEVENLABS_VOICE_ID')
            
        return merged

    def get_visual_generator_config(self) -> dict:
        default_vg = {
            'mode': 'image_prompts',
            'style_preset': 'cinematic_realistic'
        }
        return {**default_vg, **self.yaml_config.get('visual_generator', {})}

    def get_render_config(self) -> dict:
        default_render = {
            'visual_backend': 'comfyui',
            'model_preset': 'flux_schnell_fp8',
            'acceleration_mode': 'schnell',
            'image_width': 576,
            'image_height': 1024,
            'final_width': 1080,
            'final_height': 1920,
            'steps': 4,
            'guidance_scale': 0.0,
            'seed': -1,
            'sampler': 'euler',
            'scheduler': 'simple',
            'lora_preset': 'none',
            'lora_strength': 0.7,
            'motion_preset': 'slow_zoom_in',
            'captions_enabled': True,
            'comfyui_url': 'http://127.0.0.1:8188',
            'comfyui_workflow_path': 'workflows/flux_schnell_fp8_4step.json',
            'local_image_api_url': 'http://127.0.0.1:8000/txt2img',
            'image_folder_path': 'input_images',
            'fallback_backend': 'image_folder',
            'fallback_to_black': True,
            'clear_cuda_cache_between_scenes': True
        }
        return {**default_render, **self.yaml_config.get('render', {})}

    def get_model_presets(self) -> dict:
        return self.yaml_config.get('model_presets', {})

    def get_model_preset(self, name: str) -> dict:
        presets = self.get_model_presets()
        return presets.get(name, {})

    def get_acceleration_presets(self) -> dict:
        return self.yaml_config.get('acceleration_presets', {})

    def get_acceleration_preset(self, name: str) -> dict:
        presets = self.get_acceleration_presets()
        return presets.get(name, {})

    def get_lora_presets(self) -> dict:
        return self.yaml_config.get('lora_presets', {})

    def get_lora_preset(self, name: str) -> dict:
        presets = self.get_lora_presets()
        return presets.get(name, {})

    def get_comfyui_config(self) -> dict:
        default_comfy = {
            'url': 'http://127.0.0.1:8188',
            'workflow_path': 'workflows/flux_schnell_fp8_4step.json',
            'timeout_sec': 300,
            'poll_interval_sec': 2,
            'node_map': {}
        }
        return {**default_comfy, **self.yaml_config.get('comfyui', {})}

    def get_huggingface_config(self) -> dict:
        default_hf = {
            'token_env': 'HF_TOKEN',
            'use_auth_token': True
        }
        return {**default_hf, **self.yaml_config.get('huggingface', {})}

    def get_style_presets(self) -> dict:
        return self.yaml_config.get('style_presets', {})

    def get_style_preset(self, name: str) -> str:
        presets = self.get_style_presets()
        return presets.get(name, "")

    def get_llm_provider(self) -> Literal['openai', 'groq', 'gemini']:
        val = os.getenv('LLM_PROVIDER') or self.yaml_config.get('llm_provider') or 'openai'
        return val.lower()
    
    def get_llm_model(self) -> str:
        provider = self.get_llm_provider()
        env_model = os.getenv(f"{provider.upper()}_MODEL")
        if env_model:
            return env_model
        yaml_model = self.yaml_config.get('llm_model')
        if yaml_model:
            return yaml_model
        if provider == 'openai':
            return 'gpt-4o'
        elif provider == 'groq':
            return 'llama3-70b-8192'
        elif provider == 'gemini':
            return 'gemini-2.5-flash'
        raise ConfigurationError(f"Unknown LLM provider: {provider}")
    
    def get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        
        provider = self.get_llm_provider()
        
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ConfigurationError("Missing required API key: OPENAI_API_KEY (required for LLM_PROVIDER=openai)")
            self._llm_client = OpenAI(api_key=api_key)
        elif provider == 'groq':
            if not GROQ_AVAILABLE:
                raise ConfigurationError("Groq library not installed. Run: pip install groq")
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ConfigurationError("Missing required API key: GROQ_API_KEY (required for LLM_PROVIDER=groq)")
            self._llm_client = Groq(api_key=api_key)
        elif provider == 'gemini':
            if not GEMINI_AVAILABLE:
                raise ConfigurationError("Gemini library not installed. Run: pip install google-generativeai")
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ConfigurationError("Missing required API key: GEMINI_API_KEY (required for LLM_PROVIDER=gemini)")
            genai.configure(api_key=api_key)
            model_name = self.get_llm_model()
            self._llm_client = genai.GenerativeModel(model_name)
        
        return self._llm_client
    
    def get_stt_provider(self) -> Literal['whisper', 'deepgram']:
        val = os.getenv('STT_PROVIDER') or self.yaml_config.get('stt_provider') or 'whisper'
        return val.lower()
    
    def get_tts_provider(self) -> str:
        tts_cfg = self.get_tts_config()
        return tts_cfg['backend']
    
    def get_tts_voice(self) -> str:
        tts_cfg = self.get_tts_config()
        return tts_cfg['voice']
    
    def get_pexels_api_key(self) -> str:
        key = os.getenv('PEXELS_API_KEY')
        if not key:
            return ""
        return key
    
    def get_video_orientation(self) -> bool:
        ratio = self.get_aspect_ratio()
        if ratio == "16:9":
            return True
        elif ratio == "9:16":
            return False
        orientation = os.getenv('VIDEO_ORIENTATION', 'portrait').lower()
        return orientation == 'landscape'

    def get_deepgram_api_key(self) -> str:
        key = os.getenv('DEEPGRAM_API_KEY')
        if not key:
            raise ConfigurationError("DEEPGRAM_API_KEY not found in .env file")
        return key
    
    def get_elevenlabs_api_key(self) -> str:
        key = os.getenv('ELEVENLABS_API_KEY')
        if not key:
            raise ConfigurationError("ELEVENLABS_API_KEY not found in .env file")
        return key
    
    def get_captions_enabled(self) -> bool:
        caps = self.yaml_config.get('captions', {})
        if 'enabled' in caps:
            return bool(caps['enabled'])
        render_caps = self.yaml_config.get('render', {})
        if 'captions_enabled' in render_caps:
            return bool(render_caps['captions_enabled'])
        return os.getenv('CAPTIONS_ENABLED', 'true').lower() == 'true'

    def get_caption_font_size(self) -> int:
        caps = self.yaml_config.get('captions', {})
        if 'font_size' in caps:
            return int(caps['font_size'])
        return int(os.getenv('CAPTION_FONT_SIZE', '100'))
    
    def get_caption_font_color(self) -> str:
        caps = self.yaml_config.get('captions', {})
        if 'font_color' in caps:
            return str(caps['font_color']).lower()
        return os.getenv('CAPTION_FONT_COLOR', 'white').lower()
    
    def get_caption_stroke_width(self) -> int:
        caps = self.yaml_config.get('captions', {})
        if 'stroke_width' in caps:
            return int(caps['stroke_width'])
        return int(os.getenv('CAPTION_STROKE_WIDTH', '3'))
    
    def get_caption_stroke_color(self) -> str:
        caps = self.yaml_config.get('captions', {})
        if 'stroke_color' in caps:
            return str(caps['stroke_color']).lower()
        return os.getenv('CAPTION_STROKE_COLOR', 'black').lower()
    
    def get_caption_position(self) -> str:
        caps = self.yaml_config.get('captions', {})
        position = caps.get('position') or os.getenv('CAPTION_POSITION', 'bottom_center')
        position = str(position).lower()
        valid_positions = ['center', 'top', 'bottom', 'bottom_center', 'bottom_left', 'bottom_right']
        if position not in valid_positions:
            raise ConfigurationError(
                f"Invalid CAPTION_POSITION: '{position}'. Must be one of: {', '.join(valid_positions)}"
            )
        return position
    
    def get_caption_font_face(self) -> str:
        caps = self.yaml_config.get('captions', {})
        if 'font_face' in caps:
            return str(caps['font_face'])
        return os.getenv('CAPTION_FONT_FACE', 'Arial-Bold')


def get_config() -> Config:
    try:
        return Config()
    except ConfigurationError as e:
        print(f"\n{'='*70}")
        print("ERROR: Configuration Failed")
        print('='*70)
        print(f"\n{str(e)}\n")
        print("Please fix these issues and try again.")
        print('='*70 + '\n')
        raise
