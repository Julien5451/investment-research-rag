"""Chinese research QA demo. Run: streamlit run streamlit_app.py."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import pymupdf
import streamlit as st

from research_qa.demo_client import DemoAPIError, QAClient

ROOT = Path(__file__).resolve().parent
PHASE3_SUMMARY = ROOT / "eval" / "phase3" / "optimized" / "summary.json"
SUMMARY_PATH = (
    PHASE3_SUMMARY
    if PHASE3_SUMMARY.is_file()
    else ROOT / "eval" / "results" / "latest" / "summary.json"
)


@st.cache_data(max_entries=8, show_spinner=False)
def source_page_png(data: bytes, page: int) -> bytes:
    """Render only browser-session bytes, never a model-supplied filesystem path."""
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if not 1 <= page <= len(pdf):
            raise ValueError("Citation page outside PDF")
        source = pdf[page - 1]
        # Bound preview pixel dimensions even for PDFs with enormous page boxes.
        scale = min(1.4, 1500 / max(source.rect.width, 1), 2000 / max(source.rect.height, 1))
        return source.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")


def show_answer(turn: dict, turn_index: int) -> None:
    response = turn["response"]
    if response["abstained"]:
        st.warning("证据不足，系统选择拒答。")
        st.write(response["answer"])
        st.caption("原因：" + (response.get("reason") or "insufficient_evidence"))
    else:
        st.write(response["answer"])
    st.caption(f"本次请求耗时 {turn['elapsed_seconds']:.2f} 秒 · 引用页码为 PDF 物理页码")
    for index, citation in enumerate(response["citations"]):
        source_id = citation["source_id"]
        name = citation["document_name"]
        page = citation["page"]
        with st.expander(f"[{source_id}] {name} · 第 {page} 页"):
            st.text(citation["quote"])
            st.caption(f"语义相似度 {citation['relevance_score']:.3f}；该分数不代表正确率。")
            local = st.session_state.uploaded_documents.get(citation["document_id"])
            if local is None:
                st.caption("该 PDF 由其他会话或评测流程上传；当前会话仅展示引用摘录。")
                continue
            prefix = f"source_{turn_index}_{index}"
            st.download_button(
                "下载原始 PDF",
                data=local["data"],
                file_name=local["document_name"],
                mime="application/pdf",
                key=prefix + "_download",
            )
            if st.checkbox("预览引用页面", key=prefix + "_preview"):
                try:
                    st.image(source_page_png(local["data"], page), width="stretch")
                except (RuntimeError, ValueError):
                    st.info("当前 PDF 页面无法预览，请下载原文核对。")


def show_evaluation() -> None:
    st.subheader("真实模型评测")
    if not SUMMARY_PATH.is_file():
        st.info("尚无评测结果。完成真实模型评测后，此处将读取保存的 summary.json。")
        st.caption("不会使用演示数据填充指标。")
        return
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("Expected summary object")
    except (OSError, ValueError):
        st.warning("评测结果文件无法读取，请重新生成 summary.json。")
        return
    run = summary.get("run", {})
    if not isinstance(run, dict) or run.get("mode") != "live" or run.get("status") != "complete":
        st.info("尚无完整的真实模型评测结果。模拟运行或未完成的运行不会展示为真实指标。")
        return
    try:
        summary_label = str(SUMMARY_PATH.relative_to(ROOT))
    except ValueError:
        summary_label = SUMMARY_PATH.name
    st.caption(f"数据来源：{summary_label}；只展示已保存的真实运行结果。")
    counts = summary.get("counts", {})
    trials = counts.get("questions", "未记录")
    unique = counts.get("unique_questions", trials)
    repeats = len(run.get("repeat_ids", []))
    sample_label = (
        f"{unique} 题 × {repeats} 轮 = {trials} 次答题" if repeats else f"样本数：{trials}"
    )
    st.caption(
        f"运行时间（上海）：{run.get('started_at_shanghai', '未记录')} · "
        f"Chat：{run.get('chat_model', '未记录')} · "
        f"Embedding：{run.get('embedding_model', '未记录')} · "
        f"{sample_label}"
    )
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        st.warning("评测指标格式无法识别，请检查 summary.json。")
        return

    def percentage(value) -> str:
        if isinstance(value, (int, float)) and math.isfinite(value):
            return f"{value:.1%}"
        return "暂无判定"

    columns = st.columns(4)
    evidence_recall = metrics.get("evidence_recall_at_k")
    if evidence_recall is not None:
        columns[0].metric(f"证据 Recall@{run.get('top_k', 5)}", percentage(evidence_recall))
    else:
        columns[0].metric("页级 Recall@3", percentage(metrics.get("recall_at_3")))
    accuracy_label = (
        "答案正确率（规则初筛）"
        if metrics.get("answer_accuracy_method") == "regex_preliminary"
        else "答案正确率"
    )
    columns[1].metric(accuracy_label, percentage(metrics.get("answer_accuracy")))
    columns[2].metric("正确拒答率", percentage(metrics.get("correct_refusal_rate")))
    latency = metrics.get("latency_seconds", {}).get("ask", {}).get("p95")
    columns[3].metric(
        "问答延迟 P95", f"{latency:.2f} 秒" if isinstance(latency, (int, float)) else "暂无记录"
    )
    method = metrics.get("answer_accuracy_method")
    if method:
        st.caption("答案评分方式：" + str(method))
    if evidence_recall is not None:
        st.caption(
            f"Dense Top-{run.get('top_k', 5)} · 标注页 Recall@3 "
            + percentage(metrics.get("recall_at_3"))
            + "；证据召回检查已核验的内容、年份与单位组合，不以同页命中代替。"
        )
    else:
        st.caption("Recall@3 按标注的文档与物理页统计召回，未校验召回片段是否完整包含所需证据。")
    st.caption(
        "误拒答率："
        + percentage(metrics.get("false_refusal_rate"))
        + " · 拒答精确率："
        + percentage(metrics.get("abstention_precision"))
    )
    st.info("这是小样本开发集结果。重复运行仍是相同问题，项目复核不等同于独立人工盲评。")
    with st.expander("运行配置、样本数与指标定义"):
        st.json(summary, expanded=False)


def main() -> None:
    st.set_page_config(page_title="研报问答 | Research QA", page_icon="📑", layout="wide")
    st.session_state.setdefault("uploaded_documents", {})
    st.session_state.setdefault("upload_checksums", {})
    st.session_state.setdefault("turns", [])
    st.title("研报问答")
    st.caption("上传公开年报，用原文证据回答问题，并回到具体页面核对。")
    try:
        client = QAClient(os.environ.get("QA_API_URL", "http://127.0.0.1:8000"))
    except DemoAPIError as exc:
        st.error(str(exc))
        st.stop()
    connected = False
    with st.sidebar:
        st.subheader("服务状态")
        st.caption("FastAPI：" + client.base_url)
        st.button("刷新连接", key="refresh_health")
        try:
            health = client.health()
            connected = health.status == "ok"
            if connected:
                st.success("后端已连接")
            else:
                st.warning("后端尚未就绪")
            st.metric("已索引文本块", health.indexed_chunks)
        except DemoAPIError as exc:
            st.error(str(exc))
        st.divider()
        top_k = st.slider("检索 Top-k", 1, 20, 5, help="先检索，再过滤低相关性或可疑指令片段。")
        st.caption("检索当前后端集合的全部文档。每个问题独立检索，聊天记录仅用于展示。")
        if st.button("清空聊天记录"):
            st.session_state.turns = []
        st.divider()
        st.caption("模型配置和凭据由后端管理；此界面只调用文档与问答 API。")

    question_tab, evaluation_tab = st.tabs(["文档与问答", "评测结果"])
    with evaluation_tab:
        show_evaluation()
    with question_tab:
        with st.container(border=True):
            st.subheader("1 · 上传研究材料")
            st.caption("支持含可提取文字的 PDF。后端负责校验大小、类型与页数；扫描件暂不支持 OCR。")
            notice = st.session_state.pop("upload_notice", None)
            if notice:
                st.success(notice)
            pdf = st.file_uploader("选择 PDF", type=["pdf"], accept_multiple_files=False)
            if st.button("上传并建立索引", disabled=pdf is None or not connected, type="primary"):
                data = pdf.getvalue()
                checksum = hashlib.sha256(data).hexdigest()
                if checksum in st.session_state.upload_checksums:
                    st.info("这份 PDF 已在当前会话上传，无需重复建库。")
                else:
                    try:
                        with st.spinner("正在上传、提取文本并建立向量索引…"):
                            result = client.upload(pdf.name, data)
                        st.session_state.uploaded_documents[result.document_id] = {
                            **result.model_dump(),
                            "data": data,
                        }
                        st.session_state.upload_checksums[checksum] = result.document_id
                        st.session_state.upload_notice = (
                            f"已完成：{result.document_name} · "
                            f"{result.pages} 页 · {result.chunks} 块"
                        )
                        st.rerun()
                    except DemoAPIError as exc:
                        st.error(str(exc))
            if st.session_state.uploaded_documents:
                with st.expander("本会话已上传文档", expanded=False):
                    for document in st.session_state.uploaded_documents.values():
                        st.text(
                            f"{document['document_name']} · {document['pages']} 页 · "
                            f"{document['chunks']} 块"
                        )

        st.subheader("2 · 提问并核验证据")
        if not st.session_state.turns:
            st.info("例如：Apple 在 2025 财年的总净销售额是多少？请保留金额单位。")
        for turn_index, turn in enumerate(st.session_state.turns):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                if "error" in turn:
                    st.error(turn["error"])
                else:
                    show_answer(turn, turn_index)
        question = st.chat_input(
            "输入关于已上传文档的问题…", max_chars=4000, disabled=not connected
        )
        if question and question.strip():
            started = time.perf_counter()
            turn = {"question": question.strip()}
            try:
                with st.spinner("检索证据并生成回答…"):
                    response = client.ask(question.strip(), top_k)
                turn["response"] = response.model_dump()
                turn["elapsed_seconds"] = time.perf_counter() - started
            except DemoAPIError as exc:
                turn["error"] = str(exc)
            st.session_state.turns.append(turn)
            st.rerun()


if __name__ == "__main__":
    main()
