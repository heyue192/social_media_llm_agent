# Social Media Sentiment Analysis and Response Generation LLM Agent

## Project Overview
This project builds a social media LLM agent that can:
1. classify the sentiment of a social media post,
2. map the sentiment to a response policy,
3. generate a natural response with a Qwen-based language model,
4. provide an interactive Streamlit demo for testing and presentation.

The system is designed for social-media-style text and combines sentiment analysis with response generation in a modular pipeline.

---

## Project Motivation
Social media posts are often short, emotional, and context-sensitive.  
This project explores how an LLM agent can first identify sentiment and then generate a context-aware reply that matches the emotional tone of the input.

---

## Main Functions
- **Sentiment Analysis**
  - Predicts one of three sentiment classes:
    - `negative`
    - `neutral`
    - `positive`

- **Policy Mapping**
  - Maps sentiment to a response strategy.

- **Response Generation**
  - Uses a Qwen-based model to generate a natural reply.

- **Interactive Demo**
  - Provides a Streamlit front end for:
    - online input prediction,
    - response generation,
    - result visualization.

---

## Datasets
This project mainly uses:

- **TweetEval / sentiment**
  - for 3-class sentiment classification

- **EmpatheticDialogues**
  - for building response guidance and improving empathetic reply generation

---

## Model Pipeline

```text
Input social media post
→ sentiment classification
→ response policy mapping
→ guidance retrieval
→ Qwen-based response generation
→ final reply
```

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- LangChain
- Streamlit
- scikit-learn
- matplotlib

## Repository Structure

```bush
social-media-llm-agent/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
│   ├── demo.png
│   └── confusion_matrix.png
├── docs/
│   └── index.html
├── knowledge/
│   └── combined_knowledge.txt
├── outputs/
│   └── eval/
│       ├── metrics.txt
│       └── confusion_matrix.png
├── prompts/
│   └── response_prompt.txt
└── src/
    ├── agent_chain.py
    ├── config.py
    ├── generator.py
    ├── retriever.py
    ├── sentiment_model.py
    └── text_utils.py
```

## Local Run

### 1. Install dependencies

```bush
pip install -r requirements.txt
```

### 2. Prepare local model path

This project uses a locally available Qwen model path in `deployment/runtime`.

Make sure the model path is correctly set in `src/config.py` or through environment variables.

### 3. Run the Streamlit app

```bush
streamlit run app.py
```

## Evaluation Results

The classification module is evaluated using:

- Accuracy
- Macro-F1
- Confusion Matrix

Evaluation outputs are stored in:

- `outputs/eval/metrics.txt`
- `outputs/eval/confusion_matrix.png`

## Demo Screenshots

- **Streamlit Demo**  
- **Confusion Matrix**

## Notes

- The repository is mainly for code review and project presentation.
- Large model weights are not uploaded to GitHub.
- The deployed demo may run on a rented GPU server, while this repository contains the source code, prompts, evaluation outputs, and documentation.

## Future Improvements

- Improve multilingual sentiment classification
- Improve response relevance and robustness
- Add better retrieval and filtering for response guidance
- Support cleaner online deployment workflows

## Course Information

- **Course:** CDS547 – Introduction to Large Language Models  
- **Project Title:** Social Media Sentiment Analysis and Response Generation LLM Agent

## License

This repository is for academic/course project use.
