import os
import json
# import whisper
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from openai import OpenAI

"""
Pipeline:
Audio -> Whisper -> Raw Transcript -> LLM Correction -> Final Transcript
"""

# -------------------------------
# Load Whisper Model
# -------------------------------

MODEL_PATH = "./whisper-small-hi"   #


def load_asr():

    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

    model.eval()

    return processor, model


# -------------------------------
# Transcribe Audio
# -------------------------------

def transcribe_audio(asr, audio_path):

    processor, model = asr

    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        predicted_ids = model.generate(
            inputs.input_features,
            language="hi",
            task="transcribe"
        )

    transcript = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    segments = []   # huggingface whisper does not return segments like openai whisper

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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
