from gtts import gTTS
import tempfile

def text_to_speech(text, lang='en'):
    """
    Converts text to speech using gTTS and returns the path to the audio file.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None
