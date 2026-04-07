from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.agent_chain import build_agent_chain
from src.config import CONFIG
from src.sentiment_model import SentimentClassifier

import os
from pathlib import Path

# =============================
# 🌍 Language Dictionary
# =============================
LANG = {
    "en": {
        "sidebar_title": "⚙️ System Configuration",
        "classifier_dir": "Sentiment Classifier Directory",
        "knowledge_path": "Knowledge Base Path",
        "show_prompt": "Show Full Prompt",
        "show_guidance": "Show Retrieved Guidance",
        "show_scores": "Show Sentiment Scores",
        "current_config": "### Current Configuration",
        "qwen_model": "Qwen Model",
        "knowledge_base": "Knowledge Base",

        "desc": 
        """
        This system demonstrates an LLM agent for social media:
        
        - Sentiment Classification  
        - Response Policy Mapping  
        - Response Generation (Qwen)  
        - Knowledge Retrieval  
        """,

        "model_found": "Classifier directory detected",
        "model_not_found": "Classifier directory NOT found",
        "kb_found": "Knowledge file detected",
        "kb_not_found": "Knowledge file NOT found",

        "tabs": ["📊 Training Results", "🧪 Live Prediction", "📝 Case Study", "ℹ️ System Info"],

        "metrics": "Classification Metrics",
        "confusion": "Confusion Matrix",
        "pipeline": "Training & Inference Pipeline",

        "input_label": "Enter a social media post",
        "run": "Run Model",
        "running": "Loading model and generating response...",
        "done": "Completed",

        "result": "Sentiment Analysis Result",
        "response": "Generated Response",
        "guidance": "Retrieved Guidance",
        "prompt": "Full Prompt",
        "json": "Raw JSON Output",

        "case_title": "Case Study",
        "case_desc": "Click a sample below to test the model.",
        "select_case": "Select a sample case",
        "run_case": "Run Sample",

        "system": "System Description",

        "error_model": "Classifier directory does not exist.",
        "error_kb": "Knowledge file does not exist.",

        "case_input": "Input Text",
        "case_response": "Model Response",
        "case_running": "Running sample...",
        "error_model_short": "Classifier directory not found.",
        "error_kb_short": "Knowledge file not found.",
        "metric_sentiment": "Sentiment",
        "metric_policy": "Policy",
        "metric_length": "Response Length",

        "system_detail": 
        """
        **Design Overview**  
        - Sentiment Classification: Fine-tuned BERT model based on `tweet_eval_sentiment`  
        - Policy Mapping: Select response strategy based on sentiment (positive/negative/neutral)  
        - Knowledge Retrieval: Retrieve relevant few-shot examples and guidance from knowledge base  
        - Response Generation: Generate final reply using Qwen2.5-1.5B-Instruct  
        
        **Project Structure**  
        - `outputs/tweet_eval_sentiment/`: classification model  
        - `outputs/eval/`: evaluation results (metrics.txt, confusion_matrix.png)  
        - `knowledge/combined_knowledge.txt`: knowledge base  
        - `src/agent_chain.py`: agent pipeline  
        - `src/sentiment_model.py`: sentiment classifier  
        
        **Requirements**  
        - Python 3.10+  
        - Dependencies: `streamlit`, `transformers`, `torch`, `qwen`  
        - Run `merge_knowledge.py` to build knowledge base  
        - Train or download the sentiment model in advance  
        """
    },

    "zh": {
        "sidebar_title": "⚙️ 系统配置",
        "classifier_dir": "情感分类模型目录",
        "knowledge_path": "知识库路径",
        "show_prompt": "显示完整 Prompt",
        "show_guidance": "显示检索到的 Guidance",
        "show_scores": "显示情感分数",
        "current_config": "### 当前配置",
        "qwen_model": "Qwen 模型",
        "knowledge_base": "知识库",

        "desc": """
        本系统用于演示一个面向社交媒体的 LLM Agent：
        - 情感分类  
        - 回复策略映射  
        - 回复生成（Qwen）  
        - 知识检索  
        """,

        "model_found": "已检测到分类模型目录",
        "model_not_found": "未找到分类模型目录",
        "kb_found": "已检测到知识库文件",
        "kb_not_found": "未找到知识库文件",

        "tabs": ["📊 训练结果展示", "🧪 在线输入预测", "📝 案例分析", "ℹ️ 系统说明"],

        "metrics": "分类评测指标",
        "confusion": "混淆矩阵",
        "pipeline": "项目训练与推理流程",

        "input_label": "请输入一条社交媒体文本",
        "run": "运行模型",
        "running": "正在加载模型并生成结果...",
        "done": "运行完成",

        "result": "情感分析结果",
        "response": "生成回复",
        "guidance": "检索到的 Guidance",
        "prompt": "完整 Prompt",
        "json": "原始 JSON 输出",

        "case_title": "案例分析",
        "case_desc": "点击下面的样例，可快速测试模型效果。",
        "select_case": "选择一个测试样例",
        "run_case": "运行样例",

        "system": "系统说明",

        "error_model": "分类模型目录不存在",
        "error_kb": "知识库文件不存在",

        "case_input": "输入文本",
        "case_response": "模型回复",
        "case_running": "正在运行样例...",
        "error_model_short": "分类模型目录不存在。",
        "error_kb_short": "知识库文件不存在。",
        "metric_sentiment": "情感",
        "metric_policy": "策略",
        "metric_length": "回复长度",

        "system_detail": 
        """
        **设计说明**  
        - 情感分类：基于 `tweet_eval_sentiment` 微调的 BERT 模型  
        - 策略映射：根据情感标签（positive/negative/neutral）选择回复策略  
        - 知识检索：从文本知识库中检索相关的 few-shot 示例和指导  
        - 回复生成：使用 Qwen2.5-1.5B-Instruct 生成最终回复  

        **文件结构**  
        - `outputs/tweet_eval_sentiment/`：分类模型  
        - `outputs/eval/`：评测结果（metrics.txt, confusion_matrix.png）  
        - `knowledge/combined_knowledge.txt`：知识库文件  
        - `src/agent_chain.py`：Agent 链实现 
        - `src/sentiment_model.py`：情感分类器  

        **运行要求**  
        - Python 3.10+  
        - 安装依赖：`streamlit`, `transformers`, `torch`, `qwen` 等  
        - 预先运行 `merge_knowledge.py` 生成知识库  
        - 预先训练或下载情感分类模型 
        """
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

lang = st.sidebar.radio(
    "Language / 语言",
    ["en", "zh"],
    index=0,
    format_func=lambda x: "English" if x == "en" else "中文",
)

T = LANG[lang]


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Social Media Sentiment Analysis and Response Generation Agent",
    page_icon="🤖",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CLASSIFIER_DIR = Path(
    os.getenv("CLASSIFIER_DIR", str(PROJECT_ROOT / "outputs" / "tweet_eval_sentiment"))
)
DEFAULT_EVAL_DIR = Path(
    os.getenv("EVAL_DIR", str(PROJECT_ROOT / "outputs" / "eval"))
)
DEFAULT_KNOWLEDGE_PATH = Path(
    os.getenv("KNOWLEDGE_PATH", str(PROJECT_ROOT / "knowledge" / "combined_knowledge.txt"))
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_classifier(model_dir: str):
    return SentimentClassifier(model_dir=model_dir, device=-1)


@st.cache_resource
def load_agent_chain_cached(model_dir: str, knowledge_path: str):
    CONFIG.knowledge_path = knowledge_path
    return build_agent_chain(classifier_dir=model_dir)


def read_text_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def render_metric_file(metrics_path: Path):
    if metrics_path.exists():
        content = metrics_path.read_text(encoding="utf-8")
        st.code(content, language="text")
    else:
        st.warning(f"未找到评测结果文件：{metrics_path}")


def render_confusion_matrix(conf_mat_path: Path):
    if conf_mat_path.exists():
        st.image(str(conf_mat_path), caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning(f"未找到混淆矩阵图片：{conf_mat_path}")


def get_sample_cases():
    return [
        "I’m really upset because my order is late and nobody replied to my message.",
        "Thanks so much! The support team solved my problem really quickly.",
        "Can someone tell me when the update will be released?",
        "I’m disappointed with the service. This is the second time this happened.",
        "The product works fine, but I need more information about the warranty.",
        "I love the new feature. It makes everything much easier.",
    ]

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # 优先处理常见自定义对象
    if hasattr(obj, "__dict__"):
        try:
            return {k: make_json_safe(v) for k, v in obj.__dict__.items()}
        except Exception:
            return str(obj)

    return str(obj)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title(T["sidebar_title"])

classifier_dir = st.sidebar.text_input(T["classifier_dir"], value=str(DEFAULT_CLASSIFIER_DIR))
knowledge_path = st.sidebar.text_input(T["knowledge_path"], value=str(DEFAULT_KNOWLEDGE_PATH))

show_prompt = st.sidebar.checkbox(T["show_prompt"], value=True)
show_guidance = st.sidebar.checkbox(T["show_guidance"], value=True)
show_scores = st.sidebar.checkbox(T["show_scores"], value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(T["current_config"])
st.sidebar.write(f"**{T['qwen_model']}**: `{CONFIG.qwen_model_name}`")
st.sidebar.write(f"**{T['knowledge_base']}**: `{knowledge_path}`")


# -----------------------------
# Title
# -----------------------------
st.title("🤖 Social Media Sentiment Analysis and Response Generation Agent")
st.markdown(T["desc"])

# Path check
classifier_path_obj = Path(classifier_dir).resolve()
knowledge_path_obj = Path(knowledge_path).resolve()

col_status1, col_status2 = st.columns(2)

with col_status1:
    if classifier_path_obj.exists():
        st.success(f"{T['model_found']}: {classifier_path_obj}")
    else:
        st.error(f"{T['model_not_found']}: {classifier_path_obj}")

with col_status2:
    if knowledge_path_obj.exists():
        st.success(f"{T['kb_found']}: {knowledge_path_obj}")
    else:
        st.error(f"{T['kb_not_found']}: {knowledge_path_obj}")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(T["tabs"])


# -----------------------------
# Tab 1: Evaluation results
# -----------------------------
with tab1:
    st.subheader(T["tabs"][0])

    eval_dir = DEFAULT_EVAL_DIR
    metrics_path = eval_dir / "metrics.txt"
    conf_mat_path = eval_dir / "confusion_matrix.png"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"#### {T['metrics']}")
        render_metric_file(metrics_path)

    with col2:
        st.markdown(f"#### {T['confusion']}")
        render_confusion_matrix(conf_mat_path)

    st.markdown(f"#### {T['pipeline']}")
    st.markdown(
        """
        ```text
        Input social media post
        → sentiment classifier
        → response policy mapping
        → retrieve guidance/examples
        → Qwen2.5-1.5B-Instruct
        → final response
        """
    )

# =========================================================
# Tab 2: Live demo
# =========================================================

with tab2:
    st.subheader(T["tabs"][1])

    default_text = "I’m really upset because my order is late and nobody replied to my message."
    user_input = st.text_area(
        T["input_label"],
        value=default_text,
        height=140,
    )

    run_button = st.button(T["run"], type="primary", use_container_width=True)

    if run_button:
        if not classifier_path_obj.exists():
            st.error(T["error_model"])
        elif not knowledge_path_obj.exists():
            st.error(T["error_kb"])
        else:
            try:
                with st.spinner(T["running"]):
                    agent = load_agent_chain_cached(
                        model_dir=str(classifier_path_obj),
                        knowledge_path=str(knowledge_path_obj),
                    )
                    result = agent.invoke({"post": user_input})

                st.success(T["done"])

                result_col1, result_col2 = st.columns([1, 1])

                with result_col1:
                    st.markdown(f"### {T['result']}")
                    st.write(f"**Sentiment**: `{result['sentiment']}`")
                    st.write(f"**Policy**: `{result['policy']}`")

                    if show_scores:
                        st.markdown("#### Sentiment Scores")
                        st.json(result.get("sentiment_scores", {}))

                with result_col2:
                    st.markdown(f"### {T['response']}")
                    st.text_area(
                        "Response",
                        value=result.get("response", ""),
                        height=180,
                    )

                if show_guidance:
                    st.markdown(f"### {T['guidance']}")
                    st.code(result.get("retrieved_guidance", ""), language="text")

                if show_prompt:
                    st.markdown(f"### {T['prompt']}")
                    st.code(result.get("prompt", ""), language="text")

                st.markdown(f"### {T['json']}")
                safe_result = make_json_safe(result)
                st.code(json.dumps(safe_result, ensure_ascii=False, indent=2), language="json")

            except Exception as e:
                st.exception(e)

# =========================================================
# Tab 3: Sample cases
# =========================================================

with tab3:
    st.subheader(T["case_title"])
    st.markdown(T["case_desc"])

    sample_cases = get_sample_cases()
    selected_case = st.selectbox(T["select_case"], sample_cases)

    if st.button(T["run_case"], use_container_width=True):
        if not classifier_path_obj.exists():
            st.error(T["error_model_short"])
        elif not knowledge_path_obj.exists():
            st.error(T["error_kb_short"])
        else:
            try:
                with st.spinner(T["case_running"]):
                    agent = load_agent_chain_cached(
                        model_dir=str(classifier_path_obj),
                        knowledge_path=str(knowledge_path_obj),
                    )
                    result = agent.invoke({"post": selected_case})

                st.markdown(f"#### {T['case_input']}")
                st.write(selected_case)

                c1, c2, c3 = st.columns(3)
                c1.metric(T["metric_sentiment"], result["sentiment"])
                c2.metric(T["metric_policy"], result["policy"])
                c3.metric(T["metric_length"], len(result.get("response", "").split()))

                st.markdown(f"#### {T['case_response']}")
                st.info(result.get("response", ""))

                if show_guidance:
                    st.markdown(f"#### {T['guidance']}")
                    st.code(result.get("retrieved_guidance", ""), language="text")

                if show_prompt:
                    st.markdown(f"#### {T['prompt']}")
                    st.code(result.get("prompt", ""), language="text")

            except Exception as e:
                st.exception(e)

# =========================================================
# Tab 4: System description
# =========================================================

with tab4:
    st.markdown(T["system_detail"])
