# VoiceOps 2026 - ZentroMart voice support agent

Pipeline: Gnani STT -> agent (Groq LLM + RAG over kb/*.md + lookup_order/create_ticket) -> Gnani TTS.

## Setup
1. pip install -r requirements.txt
2. Copy .env.example to .env and fill GROQ_API_KEY and GNANI_API_KEY
3. Extract the capstone dataset and set DATASET_DIR to its folder (or pass --data)

## Run (batch over a folder of audio files)
python run.py <audio_folder> --out sample_outputs --data <dataset_folder>

Files named like PC002_1.wav, PC002_2.wav are treated as turns of one conversation.
Any other file name is treated as a separate single question.

Outputs in the --out folder: <name>_reply.wav, results.json and results.csv
(heard text, reply text, retrieved chunk IDs, tool calls).

## Links
- Report: https://github.com/Mrittika666/voiceops-zentromart/blob/main/report.pdf
- Demo video (2 min): https://drive.google.com/file/d/12F0X-VCh-bkDZXXL8XtTa-RX8ic0tCe1/view?usp=sharing
