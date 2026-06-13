# === ARQUIVO: backend/core/tts_manager.py ===
"""
TTSManager — converte texto em audio via ElevenLabs API.
Retorna audio em base64 para o frontend reproduzir diretamente.
"""
import logging
import base64
import os
import httpx

logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam (pt-BR compatível)
ELEVENLABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"


async def text_to_speech(text: str) -> str | None:
    """
    Converte texto em audio via ElevenLabs.
    Retorna string base64 do audio MP3, ou None em caso de falha.
    Limita a 500 caracteres para economizar cota do tier gratuito.
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY nao configurada — TTS desabilitado.")
        return None

    # Limitar tamanho para economizar cota (10k chars/mes no tier free)
    text_trimmed = text[:500] if len(text) > 500 else text

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                ELEVENLABS_URL,
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text_trimmed,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.3,
                        "use_speaker_boost": True,
                    },
                },
            )

            if resp.status_code == 200:
                audio_b64 = base64.b64encode(resp.content).decode("utf-8")
                logger.info("TTS gerado com sucesso (%d bytes).", len(resp.content))
                return audio_b64
            else:
                logger.warning("ElevenLabs erro %s: %s", resp.status_code, resp.text[:200])
                return None

    except Exception as e:
        logger.warning("TTS falhou: %s", e)
        return None
