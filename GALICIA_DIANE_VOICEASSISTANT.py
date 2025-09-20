import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import string
import threading

# ------------------ SETUP ------------------
recognizer = sr.Recognizer()

# Load DialoGPT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# ------------------ FUNCTIONS ------------------
def speak(text):
    """Speak text in a separate thread to avoid blocking."""
    def run_speech(t):
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # Female voice
        engine.say(t)
        engine.runAndWait()
        engine.stop()
    t = threading.Thread(target=run_speech, args=(text,))
    t.start()
    t.join()  # Wait until speech finishes before continuing

def listen():
    """Listen to microphone input and return recognized text."""
    with sr.Microphone() as source:
        print("■ Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            command = recognizer.recognize_google(audio)
            print(f"■ You said: {command}")
            return command
        except sr.UnknownValueError:
            print("■ Sorry, I didn't understand that.")
            return ""
        except sr.RequestError:
            print("■■ Could not request results from Google Speech Recognition.")
            return ""

def chatbot_response(user_message, chat_history_ids=None):
    """Generate AI response using DialoGPT and attention mask."""
    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    if chat_history_ids.shape[-1] > 1000:
        chat_history_ids = chat_history_ids[:, -1000:]

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# ------------------ MAIN PROGRAM ------------------
def main():
    speak("Hello! I am your AI voice assistant. How can I help you today?")
    chat_history_ids = None

    while True:
        user_command = listen()
        if user_command == "":
            continue

        normalized_command = user_command.lower().strip()
        normalized_command = normalized_command.translate(str.maketrans('', '', string.punctuation))

        # Exit words
        if any(exit_word in normalized_command for exit_word in ["stop", "exit", "quit", "bye"]):
            print("■ Exiting program...")
            speak("Goodbye! Have a great day.")  # TTS works now
            break

        # AI response
        ai_reply, chat_history_ids = chatbot_response(user_command, chat_history_ids)
        print(f"■ AI: {ai_reply}")
        speak(ai_reply)  # TTS works now

if __name__ == "__main__":
    main()
