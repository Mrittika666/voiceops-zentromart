import re
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

NO_AUDIO = "Sorry, I didn't catch that. Could you say it again?"
_stt = _tts = None


def listen(audio_path, language="en-IN"):
    global _stt
    if _stt is None:
        _stt = GnaniSTTClient()
    try:
        return (_stt.transcribe(audio_path, language_code=language).get("transcript") or "").strip()
    except Exception as e:
        print("STT error:", str(e)[:200])
        return ""


def for_ear(text):
    """Reply ko bolne layak banao: markdown hatao, order numbers ko groups mein todo."""
    t = re.sub(r"[*#`_]+", "", text).replace("₹", "rupees ")
    t = re.sub(r"(?<!\d)(\d{4})(\d{4})(?!\d)", r"\1 \2", t)
    t = re.sub(r" ending [0-9a-f]{6}", "", t)
    return re.sub(r"\s+", " ", t).strip()


def speak(text, out_path, voice="Kaveri", model="timbre-v2.5", language="en-IN"):
    global _tts
    if _tts is None:
        _tts = GnaniTTSClient()
    _tts.synthesize(text, voice=voice, model=model, language=language, output_file=out_path)
