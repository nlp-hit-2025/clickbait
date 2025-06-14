🧠 Clickbait Detection and Tactics Attribution (NLP-HIT 2025)
This repository contains the code, data, and experiments for our Clickbait Detection Project, developed as part of the NLP course at HIT (2025).

📌 Project Goals
- Detect clickbait headlines using NLP techniques and pre-trained models
- Attribute clickbait tactics (e.g., curiosity gap, emotional triggers) to each headline
- Compare pipeline approaches: single-step GPT prompting vs two-step classification using BERT

📁 Repository Structure

👥 Team 
- Lihi Nofar
- Aviv Elbaz
- Tomer Portal
The lecturer: Sasha Apartsin 



Project description: Using LLMs to detection clickbait titles and explain the atention-grabbing tactics used in the titles.
We use both a single-step pipeline of detection + tactic attribution with a single model (using GPT-4o mini and Gemini-2.0 Flash),
and a two-step pipeline where detection and attribution are done seperatly (fine-tuned BERT and fine-tuned RoBERTa)

![תמונה של WhatsApp‏ 2025-06-03 בשעה 22 23 21_ece96b0a](https://github.com/user-attachments/assets/829488ad-b4f3-4612-b1e1-d89958cc2558)
