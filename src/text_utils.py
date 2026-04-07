from __future__ import annotations
import re


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def detect_input_language(text: str) -> str:
    return "zh" if contains_chinese(text) else "en"


POSITIVE_ZH_WORDS = [
    "高兴", "开心", "快乐", "幸福", "激动", "满意", "喜欢", "太棒", "真棒",
    "中奖了", "中奖", "成功", "顺利", "感谢", "感动", "惊喜", "兴奋", "喜悦",
    "开心死了", "太开心", "好开心", "好高兴", "棒极了"
]

NEGATIVE_ZH_WORDS = [
    "生气", "愤怒", "失望", "难过", "伤心", "糟糕", "投诉", "烦", "崩溃", "痛苦",
    "迟到", "晚到", "延迟", "没人回复", "没有回复", "差", "问题", "无语", "气死了",
    "很烦", "很差", "太差", "不满意", "伤心", "郁闷"
]


def keyword_sentiment_zh(text: str) -> str | None:
    pos_score = sum(1 for w in POSITIVE_ZH_WORDS if w in text)
    neg_score = sum(1 for w in NEGATIVE_ZH_WORDS if w in text)
    if pos_score > neg_score and pos_score > 0:
        return "positive"
    if neg_score > pos_score and neg_score > 0:
        return "negative"
    return None


def normalize_for_sentiment(text: str) -> tuple[str, str]:
    """
    返回:
    - language: zh / en
    - text_for_classification
    """
    lang = detect_input_language(text)
    return lang, text