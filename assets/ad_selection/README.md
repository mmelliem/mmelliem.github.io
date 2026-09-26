# Context-Aware Selection of AI-generated Audio Descriptions for Blind and Low Vision Users
Melanie Medina (2026)

## Purpose

This project explored whether lightweight supervised learning models can select the best AI-generated audio descriptions (AD) from a pool of candidates for a given scene from a TV show, using only textual features extracted from those candidate descriptions.

## Context

Audio description (AD) narrates visual events in video for blind and low vision audiences. With the advent of newer vision-language models, AI-generated AD has become feasible. However, the technology and research for AD lags far behind that of auto-captions, focusing mostly on generation and not selection. This project targets the selection layer, where once multiple candidate ADs have already been generated, the best one is selected based on textual features. While the generation step is computationally expensive, the selection layer is lightweight and designed to run on a standard laptop.

The main question for this project is, which features are most important for selection, and what lightweight models are best for this purpose?


## Methodology

I used the Shot-by-Shot pipeline ([Xie et al., 2025](https://github.com/Jyxarthur/shot-by-shot)) on Quest High-Performance Computing Cluster to generate five candidate ADs per scene across 2,983 scenes from the TV-AD dataset (Friends, The Big Bang Theory). I used Qwen2-VL-7B (VLM) and LLaMA-3-8B (LLM) for generation.

For each candidate I extracted textual features using spaCy and sentence-transformers:
- `n_tokens`, `TTS_speech_dur`
- `n_action_verbs`, `pronoun_to_noun_ratio`, 
- `noun_ratio` (nouns / tokens)
- `verb_density` (action verbs / tokens)
- `type_token_ratio` (unique words / total words)
- `len_diff_from_mean` (AD length - mean length of all five ADs)
- `consensus_score` (mean cosine similarity between candidates)

I then evaluated two different quality metrics as my response variables:
- **Q1**: cosine similarity between candidate and human AD embeddings
- **Q2**: action score from Shot-by-Shot authors, combining semantic similarity and verb lemma matching (shown to correlate with human judgments)

I trained classification models (logistic regression, random forest, XGBoost) and regression models (Ridge, Lasso, KNN, RF, Gradient Boosting) using 5-fold GroupKFold cross-validation, then evaluated each as a selector.

[Watch video demo of the results](AD_selection.mp4) - in this video, ground-truth human AD is labeled as "GT AD", and the highest ranked candidate description is labeled as "AI AD". The dubbed audio matches "AI AD", and the logistic regression Q2 rankings. 


## Results
```
Model                    Q1 acc.    Q2 acc.    
─────────────────────────────────────────────────────────
Baseline: random          20.1%      20.1%     
Baseline: shortest        21.9%      18.0%     
Baseline: first           23.2%      22.2%     
─────────────────────────────────────────────────────────
                Classification
Logistic Regression       25.0%      33.7% ✓   
Random Forest             23.8%      31.0%     
XGBoost                   25.5%      31.8%     
─────────────────────────────────────────────────────────
				Regression
Ridge                     20.7%      29.3%     
Lasso                     20.5%      30.0% ✓   
KNN                       21.3%      24.5%     
Random Forest             20.7%      29.7%     
XGBoost                   19.7%      29.3%   
```

Logistic regression with action score (Q2) achieved the best classification result (33.7%, +13.6pp over random baseline). Regression-based selection (30.0%) nearly matched classification. All regression models converged to similar accuracy, indicating the bottleneck is feature set size rather than model architecture. Q1 (embedding similarity) performed near chance across all models, confirming it is not an effective quality metric for 
this task.

The most important features under Q2 were action verb count, pronoun-to-noun ratio, and consensus score.


As shown in the video demo, the selector model can pick descriptions that slightly follow GT AD, but it remains that the generation step is still very computationally expensive. In future work, this is the main issue I want to address.

For additional information, read my research proposal and final report, which also include citations.[^1]<br>


![Project Proposal](AD_selection_proposal.pdf)<br>
![Final Report](AD_selection_final_report.pdf)<br>





[^1]: The study resulting in this repository was assisted by a grant from the BPF Undergraduate Research Grant Award, which is administered by Northwestern University's Department of Statistics and Data Science. However, the conclusions, opinions, and other statements in this repository are the author's and not necessarily those of the department or the sponsoring institution.
