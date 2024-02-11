from gtts import gTTS
tts = gTTS(text='नमस्ते आप कैसे हैं? मैं ठीक हूँ।', lang='hi')
tts.save('hello.mp3')

