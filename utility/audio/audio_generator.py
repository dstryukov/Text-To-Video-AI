from utility.config import get_config
from utility.tts.tts_engine import generate_tts

async def generate_audio(text, outputFilename):
    """
    Асинхронная обертка для обратной совместимости, которая вызывает 
    синхронный метод генерации озвучки из tts_engine.py.
    """
    config = get_config()
    tts_config = config.get_tts_config()
    generate_tts(text, outputFilename, tts_config)
