🧠 **"Bait Buster"** - Clickbait Detection and Tactics Attribution (NLP-HIT 2025)

This repository contains the code, data, and experiments for our Clickbait Detection Project, developed as part of the NLP course at HIT (2025)


🎯 **Problem Statement**

Deceptive headlines use psychological manipulation to drive engagement. This is not a trivial issue: 72% of users report feeling tricked by misleading headlines (Pew Research), and 68% say clickbait reduces overall trust in the media. Beyond perception, the financial toll is massive – an estimated $7 billion per year is wasted on misplaced ad revenue driven by misleading content.
Detecting clickbait is challenging due to stylistic ambiguity between legitimate teasers and manipulative hooks, cultural variability in humor and sarcasm, and the adaptive nature of clickbait tactics which evolve continuously to bypass detection and exploit user attention.


📌 **Project Goals**

- Detect clickbait headlines using NLP techniques and pre-trained models
- Attribute clickbait tactics (e.g., curiosity gap, emotional triggers) to each headline
- Compare pipeline approaches: single-step GPT prompting (using GPT-4o mini and Gemini-2.0 Flash) vs two-step classification using BERT
  
  
🧾 **Formal Task Specification**

Input: a short news headline (original or modified to appear as clickbait), typically between 15–20 words.
Output: binary classification – Clickbait (1) or Not Clickbait (0), if classified as clickbait: perform multi-label classification to identify the specific clickbait tactics or stylistic patterns used
Metrics: for clickbait detection: Accuracy, Precision, Recall, F1-score. For tactic attribution (multi-label classification): Macro / Micro F1-scores


📦 **Dataset Generation**

We generate the dataset using a custom Python script. First, real news headlines are loaded from a CSV file named news_data.csv, which contains original, non-clickbait titles. Then, we define 10 common clickbait tactics:
1.Curiosity Gap
2.Exaggeration
3.Emotional Triggers
4.Sensationalism
5.Lists / Superlatives
6.Ambiguous References
7.Direct Appeals
8.Unfinished Narratives
9.Unexpected Associations
10.Provocative Questions

For each real headline, we randomly select several tactics and use GPT to rewrite the headline into a clickbait version — preserving the original factual content but changing the style to reflect the selected clickbait techniques.


📁 **Repository Structure**

![image](https://github.com/user-attachments/assets/6081f4b6-2759-4b81-919e-1f0f11f93ecf)



👥 **Team**

- Lihi Nofar
- Aviv Elbaz
- Tomer Portal
  
👨‍🏫 **Lecturer:**
Sasha Apartsin 


🖼️ **Graphical Abstract**

![תמונה של WhatsApp‏ 2025-06-03 בשעה 22 23 21_ece96b0a](https://github.com/user-attachments/assets/829488ad-b4f3-4612-b1e1-d89958cc2558)
