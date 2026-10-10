"""
StudyTime Bot - Telegram chatbot powered by a local Ollama model (llama3.2:1b).

Flow:  Telegram user -> this script -> Ollama (local LLM) -> reply to the user

Voice (Task 1):
  * Voice input : user sends a voice message -> PyAV decodes it -> Google Speech Recognition -> text
  * Voice output: answer -> gTTS -> voice message sent back to the user
"""

import asyncio
import io
import logging
import os
import re
import tempfile
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv
import av
import speech_recognition as sr
from gtts import gTTS
from ollama import AsyncClient, ResponseError
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, TelegramError
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
HISTORY_LENGTH = 10          # how many previous messages the bot remembers per chat
TELEGRAM_LIMIT = 4096        # max characters in one Telegram message
VOICE_LANGUAGES = ("en-US", "ru-RU")   # languages tried for voice messages
TTS_MAX_CHARS = 1500         # longer answers are read only partially

SYSTEM_PROMPT = (
    "You are StudyTime Bot, a friendly study assistant for university students. "
    "Explain concepts clearly and simply, give short examples, help with study plans, "
    "time management and exam preparation. Keep answers concise. "
    "Reply in the same language the user writes in."
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)   # hide noisy polling logs
log = logging.getLogger("StudyTimeBot")

ollama = AsyncClient(host=OLLAMA_HOST)

# chat_id -> last N messages, so the bot remembers the conversation
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

# chats that asked for voice replies to text messages too (/voice command)
voice_mode: set[int] = set()

recognizer = sr.Recognizer()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
async def ask_ollama(chat_id: int, user_text: str) -> str:
    """Send the conversation to the local model and return its answer."""
    history[chat_id].append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[chat_id]]

    try:
        response = await ollama.chat(model=MODEL, messages=messages)
        answer = response["message"]["content"].strip() or "I couldn't generate an answer, please rephrase."
        history[chat_id].append({"role": "assistant", "content": answer})
        return answer

    except (httpx.ConnectError, ConnectionError):
        log.error("Cannot connect to Ollama at %s", OLLAMA_HOST)
        history[chat_id].pop()
        return "⚠️ The AI model is offline. Please make sure Ollama is running and try again."
    except ResponseError as e:
        log.error("Ollama error: %s", e)
        history[chat_id].pop()
        if e.status_code == 404:
            return f"⚠️ Model '{MODEL}' is not installed. Run: ollama pull {MODEL}"
        return "⚠️ The AI model returned an error. Please try again."
    except Exception:
        log.exception("Unexpected error while calling Ollama")
        history[chat_id].pop()
        return "⚠️ Something went wrong while generating the answer."


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Telegram rejects messages longer than 4096 chars, so split long answers."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip()
    parts.append(text)
    return parts


# ---------------------------------------------------------------------------
# Voice: speech-to-text and text-to-speech
# ---------------------------------------------------------------------------
def ogg_to_pcm(path: str) -> bytes:
    """Decode a Telegram voice file (OGG/Opus) into 16 kHz mono 16-bit PCM."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    pcm = bytearray()
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                pcm += bytes(out.planes[0])[: out.samples * 2]
    for out in resampler.resample(None):          # flush the resampler
        pcm += bytes(out.planes[0])[: out.samples * 2]
    return bytes(pcm)


class FlacAudioData(sr.AudioData):
    """AudioData that creates FLAC with PyAV instead of an external 'flac' program
    (the one bundled with SpeechRecognition does not run on every Mac)."""

    def get_flac_data(self, convert_rate=None, convert_width=None):
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="flac") as out:
            stream = out.add_stream("flac", rate=self.sample_rate)
            stream.layout = "mono"
            frame = av.AudioFrame(format="s16", layout="mono",
                                  samples=len(self.frame_data) // 2)
            frame.planes[0].update(self.frame_data)
            frame.sample_rate = self.sample_rate
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
        return buf.getvalue()


def transcribe(path: str) -> tuple[str, str]:
    """Voice file -> (text, language). Tries English and Russian, keeps the most confident result."""
    audio = FlacAudioData(ogg_to_pcm(path), 16000, 2)
    best_text, best_lang, best_conf = "", "en", -1.0
    for language in VOICE_LANGUAGES:
        result = recognizer.recognize_google(audio, language=language, show_all=True)
        if result and result.get("alternative"):
            alt = result["alternative"][0]
            confidence = alt.get("confidence", 0.5)
            if confidence > best_conf:
                best_text, best_lang, best_conf = alt["transcript"], language[:2], confidence
    return best_text.strip(), best_lang


def detect_language(text: str) -> str:
    """Simple language guess for the voice reply: Cyrillic -> Russian, else English."""
    return "ru" if re.search(r"[а-яА-ЯёЁ]", text) else "en"


def clean_for_speech(text: str) -> str:
    """Remove markdown and symbols that sound bad when read aloud."""
    text = re.sub(r"`{3}.*?`{3}", " code example omitted. ", text, flags=re.S)
    text = re.sub(r"[*_#`>|~]", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"https?://\S+", "link", text)
    return text[:TTS_MAX_CHARS]


def synthesize(text: str, lang: str) -> str:
    """Text -> path to an MP3 file (Google Text-to-Speech, needs internet)."""
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    gTTS(text=clean_for_speech(text), lang=lang).save(path)
    return path


async def send_voice_reply(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           text: str, lang: str) -> None:
    """Send the answer as a Telegram voice message. Text is already sent, so failures are soft."""
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.RECORD_VOICE)
    path = None
    try:
        path = await asyncio.to_thread(synthesize, text, lang)
        with open(path, "rb") as audio:
            await update.message.reply_voice(voice=audio)
    except Exception as e:
        log.error("Text-to-speech failed: %s", e)
        await update.message.reply_text("🔇 Couldn't create the voice reply (check internet connection).")
    finally:
        if path and os.path.exists(path):
            os.remove(path)


async def reply_with_answer(update: Update, context: ContextTypes.DEFAULT_TYPE,
                            user_text: str, voice: bool, lang: str | None = None) -> None:
    """Ask Ollama, send the text answer and (optionally) a voice answer."""
    chat = update.effective_chat
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    answer = await ask_ollama(chat.id, user_text)

    for part in split_message(answer):
        await update.message.reply_text(part)

    if voice and not answer.startswith("⚠️"):
        await send_voice_reply(update, context, answer, detect_language(answer) or lang or "en")


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        f"Hi {name}! 👋 I'm StudyTime Bot.\n"
        "Ask me anything about your studies: explanations, study plans, exam tips.\n"
        "🎤 You can also send me a voice message – I'll answer with text and voice.\n\n"
        "Commands:\n/help – show help\n/voice – voice replies to text messages on/off\n"
        "/reset – forget our conversation\n/id – show your chat ID"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Just send me a question, for example:\n"
        "• Explain recursion simply\n"
        "• Make a 1-week study plan for my database exam\n"
        "• How can I stop procrastinating?\n\n"
        "🎤 Hold the microphone button in Telegram to send a voice question.\n"
        "/voice – turn voice replies to text messages on/off"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history.pop(update.effective_chat.id, None)
    await update.message.reply_text("🧹 Conversation cleared. Let's start fresh!")


async def chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the chat ID – needed for sending alerts (see send_alert.py)."""
    await update.message.reply_text(f"Your chat ID is: {update.effective_chat.id}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    text = update.message.text
    log.info("Message from %s (%s): %s", update.effective_user.username, chat.id, text)

    await reply_with_answer(update, context, text, voice=chat.id in voice_mode)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Voice message -> transcribe -> Ollama -> text + voice reply."""
    chat = update.effective_chat
    voice = update.message.voice or update.message.audio
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    fd, path = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    try:
        tg_file = await voice.get_file()
        await tg_file.download_to_drive(path)
        text, lang = await asyncio.to_thread(transcribe, path)
    except sr.RequestError as e:
        log.error("Speech service unavailable: %s", e)
        await update.message.reply_text("⚠️ Speech recognition service is unavailable. Check your internet connection.")
        return
    except Exception as e:
        log.error("Speech recognition failed: %s", e)
        await update.message.reply_text("⚠️ I couldn't process this voice message. Please try again.")
        return
    finally:
        if os.path.exists(path):
            os.remove(path)

    if not text:
        await update.message.reply_text("🤔 I couldn't hear any words. Please speak a bit louder and try again.")
        return

    log.info("Voice from %s (%s) [%s]: %s", update.effective_user.username, chat.id, lang, text)
    await update.message.reply_text(f"🎤 You said: {text}")
    await reply_with_answer(update, context, text, voice=True, lang=lang)


async def toggle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id_ = update.effective_chat.id
    if chat_id_ in voice_mode:
        voice_mode.discard(chat_id_)
        await update.message.reply_text("🔇 Voice replies to text messages are OFF. (Voice messages still get voice replies.)")
    else:
        voice_mode.add(chat_id_)
        await update.message.reply_text("🔊 Voice replies are ON – I'll also answer your text messages with voice.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Telegram/network errors instead of crashing."""
    err = context.error
    if isinstance(err, NetworkError):
        log.warning("Network problem with Telegram, retrying automatically: %s", err)
    elif isinstance(err, TelegramError):
        log.error("Telegram API error: %s", err)
    else:
        log.exception("Unhandled error", exc_info=err)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
async def on_startup(app: Application) -> None:
    log.info("Voice is enabled: speech recognition (Google) + text-to-speech (gTTS).")
    await check_ollama(app)


async def check_ollama(app: Application) -> None:
    """Runs once at startup: warn early if Ollama or the model is missing."""
    try:
        models = await ollama.list()
        names = [m.get("model") or m.get("name") for m in models["models"]]
        if not any(n and n.startswith(MODEL) for n in names):
            log.warning("Model %s not found. Run: ollama pull %s", MODEL, MODEL)
        else:
            log.info("Ollama is running, model %s is ready.", MODEL)
    except Exception:
        log.warning("Ollama is not reachable at %s. Start it with: ollama serve", OLLAMA_HOST)


def main() -> None:
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing. Add it to the .env file.")

    app = Application.builder().token(TOKEN).post_init(on_startup).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("id", chat_id))
    app.add_handler(CommandHandler("voice", toggle_voice))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    log.info("StudyTime Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)  # listens continuously


if __name__ == "__main__":
    main()