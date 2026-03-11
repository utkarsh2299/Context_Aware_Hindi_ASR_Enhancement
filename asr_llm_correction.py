import os
import json
import whisper
from jiwer import wer
from openai import OpenAI

"""
Pipeline:
Audio -> Whisper -> Raw Transcript -> LLM Correction -> Final Transcript
"""

# -------------------------------
# Load Whisper Model
# -------------------------------

def load_asr(model_size="small"):
    model = whisper.load_model(model_size)
    return model


# -------------------------------
# Transcribe Audio
# -------------------------------

def transcribe_audio(model, audio_path):

    result = model.transcribe(
        audio_path,
        language="hi"
    )

    transcript = result["text"]
    segments = result.get("segments", [])

    return transcript, segments


# -------------------------------
# Confidence Score (approx)
# -------------------------------

def compute_avg_logprob(segments):

    if len(segments) == 0:
        return None

    scores = [s["avg_logprob"] for s in segments if "avg_logprob" in s]

    if len(scores) == 0:
        return None

    return sum(scores) / len(scores)


# -------------------------------
# LLM Correction
# -------------------------------

class LLMCorrector:

    def __init__(self, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("#-###########"))
        self.model = model

        self.system_prompt = """
You are working ass a Hindi ASR post-processing module.

Your task is to correct transcription errors produced by speech recognition.
mainly Focus on:
- spelling mistakes
- grammatical errors
- missing punctuation
- context-level corrections

Rules:
Do not translate the sentence.
Preserve the original meaning.
Return only the corrected Hindi sentence.
"""

    def correct(self, text):

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
        )

        return response.choices[0].message.content.strip()


# -------------------------------
# Pipeline
# -------------------------------

def run_pipeline(audio_path):

    asr_model = load_asr()
    corrector = LLMCorrector()

    transcript, segments = transcribe_audio(asr_model, audio_path)

    confidence = compute_avg_logprob(segments)

    #  skip correction if confidence is high
    if confidence is not None and confidence > -0.4:
        corrected = transcript
    else:
        corrected = corrector.correct(transcript)

    return transcript, corrected


# -------------------------------
# WER Evaluation
# -------------------------------

# def evaluate(reference, hypothesis):

#     return wer(reference, hypothesis)


# -------------------------------
# Example
# -------------------------------

if __name__ == "__main__":

    audio = "test_hindi_male_00048.wav"

    raw, corrected = run_pipeline(audio)

    print("\nRaw ASR Output:")
    print(raw)

    print("\nCorrected Output:")
    print(corrected)
