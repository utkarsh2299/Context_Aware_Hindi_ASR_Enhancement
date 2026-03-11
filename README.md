# Context_Aware_Hindi_ASR_Enhancement
Some project work I did during my ms tenure to improve Hindi ASR output by finetuning Whisper small which was trained on several lakhs of hours of data. And passing it through GPT-3.5 Turbo LLM for improving transcription further.

This project explores a simple post-processing approach for improving Hindi automatic speech recognition (ASR) output using a large language model (LLM). The system uses OpenAI Whisper for speech transcription and applies an LLM-based correction layer to refine the raw transcripts using sentence-level context.

The motivation behind this project is that ASR systems often produce grammatically incorrect or slightly misrecognized words, especially in low-resource languages or noisy conditions. While Whisper performs well for multilingual speech recognition, some transcription errors remain. This project investigates whether a language model can correct such errors using contextual understanding.

The pipeline consists of three main stages:

1. Audio transcription using Whisper  
2. confidence estimation from ASR segments  
3. Context-aware correction using an LLM

The final output is a cleaner Hindi transcript that preserves the original meaning while correcting spelling, grammar, and punctuation.

---

## Project Overview

Traditional ASR systems optimize for acoustic matching and language modeling, but they may still produce minor lexical or grammatical errors. Instead of retraining a large ASR model, this project adds a lightweight post-processing stage using a language model to improve the final transcript.

The approach is particularly useful for:

- correcting spelling mistakes
- restoring punctuation
- fixing grammatical inconsistencies
- correcting contextually incorrect words produced by ASR

The goal is to evaluate whether contextual correction can reduce transcription errors measured using Word Error Rate (WER).

---

## Pipeline

Audio Input  
↓  
Whisper ASR  
↓  
Raw Hindi Transcript  
↓  
LLM Context Correction  
↓  
Final Corrected Transcript

---


---

## Installation

Clone the repository:

```

git clone https://github.com/utkarsh2299/Context_Aware_Hindi_ASR_Enhancement.git
cd Context_Aware_Hindi_ASR_Enhancement

```

Install dependencies:

```

pip install openai whisper jiwer

```

You will also need **ffmpeg** installed for Whisper audio processing.

---

## API Setup

Set your OpenAI API key as an environment variable.

Linux / macOS:

```

export OPENAI_API_KEY="your_api_key"

```

Windows:

```

setx OPENAI_API_KEY "your_api_key"

```

---

## Usage

Train the whisper model as per your language of choice, I have used Hindi from Mozilla dataset.

use the `Finetune_whisper_hindi.ipynb` notebook to train and generate the checkpoint.


Run the ASR pipeline on an audio file:

```

python asr_llm_correction.py

```

The script will:

1. Transcribe the audio using Whisper
2. Estimate average log probability from ASR segments
3. Apply LLM correction when confidence is low
4. Output both the raw and corrected transcripts

---

## Example

Raw Whisper Output:

```

मुझे आज कॉलेज जाना है लेकिन बारिश बहुत हो रही है तो शायद में नहीं जागा

```

Corrected Output:

```

मुझे आज कॉलेज जाना है, लेकिन बारिश बहुत हो रही है, तो शायद मैं नहीं जाऊँगा।

```

---


## References

Radford et al., 2022. Whisper: Robust Speech Recognition via Large-Scale Weak Supervision.

OpenAI Whisper Repository  
https://github.com/openai/whisper

JiWER: Word Error Rate Evaluation  
https://github.com/jitsi/jiwer

---

## License

This project is released under the MIT License.

```
