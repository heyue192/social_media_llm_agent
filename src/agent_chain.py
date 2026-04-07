from __future__ import annotations
from langchain_core.runnables import RunnableLambda

from src.config import CONFIG
from src.generator import QwenResponseGenerator
from src.prompts import load_response_prompt
from src.retriever import TfidfRetriever
from src.sentiment_model import SentimentClassifier
from src.text_utils import normalize_for_sentiment, keyword_sentiment_zh


def map_policy(sentiment: str) -> str:
    if sentiment == "negative":
        return "third_party_comforting_and_supportive"
    if sentiment == "positive":
        return "third_party_warm_and_congratulatory"
    return "third_party_calm_and_friendly"


def stringify_guidance(guidance) -> str:
    if isinstance(guidance, str):
        return guidance
    if isinstance(guidance, list):
        lines = []
        for i, item in enumerate(guidance, start=1):
            if isinstance(item, str):
                lines.append(f"- [Example {i}] {item}")
            elif hasattr(item, "__dict__"):
                data = item.__dict__
                text = data.get("text", str(item))
                lines.append(f"- [Example {i}] {text}")
            else:
                lines.append(f"- [Example {i}] {str(item)}")
        return "\n".join(lines)
    return str(guidance)


def get_fixed_guidance(language: str, sentiment: str, policy: str) -> str:
    if language == "zh":
        if sentiment == "positive":
            return (
                "请用中文回复。先识别并回应用户的开心、兴奋或喜悦情绪。"
                "然后以第三方普通网友的身份，给出自然、温暖、真诚的回应。"
                "回复可以稍微展开一点，但不要冗长，控制在2到4句。"
                "不要像客服，不要像官方公告，不要引入政治、宗教、选举、社会立场等无关内容。"
            )
        if sentiment == "negative":
            return (
                "请用中文回复。先识别并回应用户的失望、难过、生气或沮丧情绪。"
                "然后以第三方普通网友的身份，给出自然、平和、支持性的回应。"
                "回复可以稍微展开一点，但不要冗长，控制在2到4句。"
                "不要像客服，不要像官方公告，不要引入政治、宗教、选举、社会立场等无关内容。"
            )
        return (
            "请用中文回复。先判断用户整体语气较平稳，再以第三方普通网友的身份给出自然、友好的回应。"
            "回复控制在2到4句，不要太短。"
            "不要像客服，不要像官方公告，不要引入政治、宗教、选举、社会立场等无关内容。"
        )
    return (
        f"Reply in English. The sentiment is {sentiment}. "
        "Act like a normal third-party social media user. "
        "Acknowledge the user's emotion first, then give a natural and slightly fuller reply in 2-4 sentences. "
        "Do not sound like customer service or an official statement. "
        "Do not introduce unrelated topics such as politics, religion, elections, or ideology."
    )

def fallback_response(post: str, language: str, sentiment: str) -> str:
    if language == "zh":
        if sentiment == "positive":
            return "恭喜你，这真是个令人开心的好消息！希望这份好运也能给你带来更多美好的计划和收获。"
        if sentiment == "negative":
            return "听起来这件事确实让人不太舒服，很能理解你的感受。希望问题能尽快顺利解决，也谢谢你愿意说出来。"
        return "谢谢你的分享，我已经收到你的信息了。希望这条消息对你有所帮助。"
    if sentiment == "positive":
        return "Congratulations, that’s wonderful news! I’m glad to hear it and hope this good luck brings you even more great opportunities."
    if sentiment == "negative":
        return "I’m sorry you’re dealing with this, and your feelings are completely understandable. I hope things get resolved smoothly soon."
    return "Thanks for sharing your message. I’ve noted it and hope this response is helpful."


def is_off_topic_response(response: str) -> bool:
    bad_keywords = [
        "选举", "政党", "民主党", "共和党", "总统", "投票", "政治",
        "election", "democrat", "republican", "president", "politics", "vote"
    ]
    return any(k.lower() in response.lower() for k in bad_keywords)


def build_agent_chain(classifier_dir: str):
    classifier = SentimentClassifier(model_dir=classifier_dir, device=-1)
    retriever = TfidfRetriever(CONFIG.knowledge_path)
    generator = QwenResponseGenerator(
        model_name=CONFIG.qwen_model_name,
        max_new_tokens=CONFIG.max_new_tokens,
        temperature=CONFIG.temperature,
        top_p=CONFIG.top_p,
    )
    prompt_template = load_response_prompt()

    def init_state(inputs: dict) -> dict:
        original_post = inputs["post"]
        language, post_for_classification = normalize_for_sentiment(original_post)
        return {
            "post": original_post,
            "input_language": language,
            "post_for_classification": post_for_classification,
        }

    def classify(state: dict) -> dict:
        if state["input_language"] == "zh":
            zh_sentiment = keyword_sentiment_zh(state["post"])
            if zh_sentiment is not None:
                state["sentiment"] = zh_sentiment
                if zh_sentiment == "positive":
                    state["sentiment_scores"] = {
                        "negative": 0.05,
                        "neutral": 0.10,
                        "positive": 0.85,
                    }
                elif zh_sentiment == "negative":
                    state["sentiment_scores"] = {
                        "negative": 0.85,
                        "neutral": 0.10,
                        "positive": 0.05,
                    }
                else:
                    state["sentiment_scores"] = {
                        "negative": 0.10,
                        "neutral": 0.80,
                        "positive": 0.10,
                    }
                state["policy"] = map_policy(state["sentiment"])
                return state

        result = classifier.predict(state["post_for_classification"])
        state["sentiment"] = result["label"]
        state["sentiment_scores"] = result["scores"]
        state["policy"] = map_policy(state["sentiment"])
        return state

    def retrieve(state: dict) -> dict:
        if state["input_language"] == "zh":
            state["retrieved_guidance"] = get_fixed_guidance(
                language=state["input_language"],
                sentiment=state["sentiment"],
                policy=state["policy"],
            )
            return state

        query = f"{state['sentiment']} {state['policy']} {state['post']}"
        try:
            guidance = retriever.retrieve(query=query, top_k=2)
            guidance_text = stringify_guidance(guidance)
            fixed = get_fixed_guidance(
                language=state["input_language"],
                sentiment=state["sentiment"],
                policy=state["policy"],
            )
            state["retrieved_guidance"] = fixed + "\n\n" + guidance_text
        except Exception:
            state["retrieved_guidance"] = get_fixed_guidance(
                language=state["input_language"],
                sentiment=state["sentiment"],
                policy=state["policy"],
            )
        return state

    def build_prompt(state: dict) -> dict:
        reply_language = "Chinese" if state["input_language"] == "zh" else "English"
        emotion_hint = {
            "positive": "The user is feeling happy, excited, satisfied, or pleasantly surprised.",
            "negative": "The user is feeling upset, disappointed, frustrated, sad, or emotionally uncomfortable.",
            "neutral": "The user is sharing information or asking something in a relatively calm tone.",
            }.get(state["sentiment"], "The user's emotion is unclear.")
        
        prompt = prompt_template.format(
            post=state["post"],
            sentiment=state["sentiment"],
            policy=state["policy"],
            guidance=state["retrieved_guidance"],
            reply_language=reply_language,
            emotion_hint=emotion_hint,
        )
        state["prompt"] = prompt
        return state

    def generate(state: dict) -> dict:
        response = generator.generate(state["prompt"])
        if is_off_topic_response(response):
            response = fallback_response(
                post=state["post"],
                language=state["input_language"],
                sentiment=state["sentiment"],
            )
        state["response"] = response
        return state

    return (
        RunnableLambda(init_state)
        | RunnableLambda(classify)
        | RunnableLambda(retrieve)
        | RunnableLambda(build_prompt)
        | RunnableLambda(generate)
    )