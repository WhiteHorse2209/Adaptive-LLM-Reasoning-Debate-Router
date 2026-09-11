"""Interactive Streamlit dashboard for Adaptive LLM Reasoning & Debate Router."""

import time
import streamlit as st

from src.config.config import load_config
from src.provider.factory import get_provider
from src.reasoning.consistency import SelfConsistencyReasoner
from src.reasoning.direct import DirectReasoner
from src.reasoning.judge import DebateWithJudgePipeline
from src.router.models import DifficultyLevel, ReasoningStrategy
from src.router.router import AdaptiveRouter

# Set page configuration
st.set_page_config(
    page_title="Adaptive LLM Reasoning & Debate Router",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark-mode modern glassmorphism aesthetic)
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .answer-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    .judge-card {
        background: linear-gradient(135deg, rgba(20, 83, 45, 0.4), rgba(15, 23, 42, 0.9));
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
        margin-bottom: 20px;
    }
    .agent-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-revised {
        background-color: #f59e0b;
        color: #000;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }
    .badge-winner {
        background-color: #10b981;
        color: #fff;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load configuration
config = load_config()

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.caption("Adaptive LLM Reasoning & Debate Router")

    profile_names = list(config.profiles.keys())
    active_prof = st.selectbox(
        "Active Model Profile",
        options=profile_names,
        index=profile_names.index(config.active_profile) if config.active_profile in profile_names else 0,
    )

    profile_data = config.profiles.get(active_prof)
    st.info(f"**Model**: `{profile_data.model}`\n\n**Provider**: `{profile_data.provider}`\n\n**Timeout**: `{profile_data.timeout}s`")

    st.subheader("Reasoning Mode")
    mode_selection = st.radio(
        "Strategy Dispatch:",
        options=["Adaptive (Autonomous)", "Direct (Fast)", "Self-Consistency (Consensus)", "Multi-Agent Debate (Deliberative)"],
        index=0,
    )

    st.subheader("Router Hyperparameters")
    conf_high = st.slider("High Confidence Threshold (Easy)", 0.50, 0.95, float(config.raw_config.get("router", {}).get("confidence_threshold_high", 0.80)), 0.05)
    conf_low = st.slider("Low Confidence Threshold (Hard)", 0.20, 0.70, float(config.raw_config.get("router", {}).get("confidence_threshold_low", 0.50)), 0.05)
    sc_samples = st.slider("Self-Consistency Samples (N)", 2, 7, int(config.raw_config.get("router", {}).get("consistency_samples", 3)))
    debate_rounds = st.slider("Debate Rounds", 1, 4, int(config.raw_config.get("debate", {}).get("num_rounds", 2)))

# Header
st.title("🧠 Adaptive LLM Reasoning & Debate Router")
st.markdown(
    """
    Intelligently allocating LLM computational effort between **Direct Zero-Shot**, **Self-Consistency Sampling**, 
    and **Multi-Agent Deliberative Debate with Impartial Adjudication**.
    """
)

# Example presets
col_ex1, col_ex2, col_ex3 = st.columns(3)
selected_prompt = ""
with col_ex1:
    if st.button("🟢 Easy Math (Direct)", use_container_width=True):
        selected_prompt = "What is 45 * 12?"
with col_ex2:
    if st.button("🟡 Tricky Logic (Self-Consistency)", use_container_width=True):
        selected_prompt = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?"
with col_ex3:
    if st.button("🔴 Ethical Dilemma (Debate)", use_container_width=True):
        selected_prompt = "Should autonomous vehicles prioritize passenger safety or pedestrian safety in an unavoidable collision? Resolve the ethical conflict."

# Input Query
query_input = st.text_area(
    "Enter Question / Logic Problem:",
    value=selected_prompt or "Which number is larger: 9.11 or 9.9?",
    height=100,
)

if st.button("🚀 Run Reasoning Engine", type="primary", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a question to reason through.")
        st.stop()

    with st.spinner("Executing adaptive reasoning pipeline..."):
        provider = get_provider(profile_data)
        
        router_cfg = {
            "confidence_threshold_high": conf_high,
            "confidence_threshold_low": conf_low,
            "consistency_samples": sc_samples,
            "sample_temperature": 0.7,
        }
        debate_cfg = dict(config.raw_config.get("debate", {}))
        debate_cfg["num_rounds"] = debate_rounds

        start_wall = time.perf_counter()

        if mode_selection == "Direct (Fast)":
            reasoner = DirectReasoner(provider)
            res = reasoner.answer(query_input)
            duration = time.perf_counter() - start_wall

            st.markdown("### 📊 Execution Telemetry")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Strategy", "DIRECT")
            m2.metric("Difficulty", "MANUAL")
            m3.metric("Calls", "1")
            m4.metric("Tokens", f"{res.token_usage.total_tokens}")
            m5.metric("Latency", f"{duration:.2f}s")
            m6.metric("API Cost", "$0.0000")

            st.markdown(
                f"""
                <div class="answer-card">
                    <h3 style="color: #6366f1; margin-top:0;">Final Answer:</h3>
                    <h2 style="color: #fff; margin-bottom: 12px;">{res.answer}</h2>
                    <h4 style="color: #94a3b8; margin-top: 16px;">Reasoning:</h4>
                    <p style="color: #cbd5e1; line-height: 1.6;">{res.explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif mode_selection == "Self-Consistency (Consensus)":
            reasoner = SelfConsistencyReasoner(provider)
            sc_res = reasoner.sample_and_vote(query_input, num_samples=sc_samples, temperature=0.7)
            duration = time.perf_counter() - start_wall

            st.markdown("### 📊 Execution Telemetry")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Strategy", "SELF_CONSISTENCY")
            m2.metric("Agreement", f"{sc_res.agreement_score * 100:.1f}%")
            m3.metric("Samples (Calls)", f"{sc_res.num_samples}")
            m4.metric("Tokens", f"{sc_res.token_usage.total_tokens}")
            m5.metric("Latency", f"{duration:.2f}s")
            m6.metric("API Cost", "$0.0000")

            st.markdown(
                f"""
                <div class="answer-card">
                    <h3 style="color: #6366f1; margin-top:0;">Consensus Answer (Majority Vote):</h3>
                    <h2 style="color: #fff; margin-bottom: 12px;">{sc_res.final_answer}</h2>
                    <h4 style="color: #94a3b8; margin-top: 16px;">Consensus Explanation:</h4>
                    <p style="color: #cbd5e1; line-height: 1.6;">{sc_res.final_explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Candidate Distributions")
            st.bar_chart(sc_res.agreement_distribution)

            with st.expander("🔍 View Individual Candidate Reasoning Chains"):
                for cand in sc_res.candidates:
                    st.markdown(f"**Sample {cand.candidate_id}:** `{cand.answer}`")
                    st.caption(cand.explanation)

        elif mode_selection == "Multi-Agent Debate (Deliberative)":
            pipeline = DebateWithJudgePipeline(provider, debate_config=debate_cfg)
            deb_res = pipeline.run(query_input)
            duration = time.perf_counter() - start_wall

            st.markdown("### 📊 Execution Telemetry")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Strategy", "DEBATE + JUDGE")
            m2.metric("Verdict Conf", f"{deb_res.verdict.confidence_in_verdict * 100:.0f}%")
            m3.metric("Total Calls", f"{deb_res.total_calls}")
            m4.metric("Tokens", f"{deb_res.total_token_usage.total_tokens}")
            m5.metric("Latency", f"{duration:.2f}s")
            m6.metric("API Cost", "$0.0000")

            st.markdown(
                f"""
                <div class="judge-card">
                    <span class="badge-winner">WINNER: {deb_res.verdict.winning_agent}</span>
                    <h3 style="color: #10b981; margin-top:8px;">Impartial Supreme Judge Verdict:</h3>
                    <h2 style="color: #fff; margin-bottom: 8px;">{deb_res.final_answer}</h2>
                    <p style="color: #cbd5e1; line-height: 1.6;"><strong>Adjudication Summary:</strong> {deb_res.verdict.evaluation_summary}</p>
                    <p style="color: #f87171; line-height: 1.6;"><strong>Identified Argument Flaws:</strong> {', '.join(deb_res.verdict.identified_flaws) if deb_res.verdict.identified_flaws else 'None'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Cross-Examination Debate Transcript")
            for rnd in deb_res.transcript.rounds:
                st.markdown(f"#### 🥊 Round {rnd.round_number}")
                cols = st.columns(len(rnd.turns))
                for idx, turn in enumerate(rnd.turns):
                    with cols[idx]:
                        rev_badge = '<span class="badge-revised">REVISED ANSWER</span> ' if turn.revised else ''
                        st.markdown(
                            f"""
                            <div class="agent-card">
                                <strong>{turn.agent_name}</strong> {rev_badge}<br>
                                <p style="font-size: 13px; color: #94a3b8; margin-top: 6px;">{turn.argument}</p>
                                <hr style="border-color: rgba(255,255,255,0.05);">
                                <strong>Proposed Answer:</strong> <code>{turn.answer}</code>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            # Autonomous Adaptive Router
            router = AdaptiveRouter(provider, router_config=router_cfg, debate_config=debate_cfg)
            routed = router.route_and_solve(query_input)
            duration = time.perf_counter() - start_wall

            st.markdown("### 📊 Autonomous Routing Telemetry")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Selected Strategy", routed.strategy.value)
            m2.metric("Difficulty", routed.difficulty.value)
            m3.metric("Confidence", f"{routed.confidence.score:.2f}")
            m4.metric("Calls Made", f"{routed.call_count}")
            m5.metric("Total Tokens", f"{routed.token_usage.total_tokens}")
            m6.metric("Latency", f"{duration:.2f}s")

            st.markdown(
                f"""
                <div class="answer-card">
                    <h4 style="color: #818cf8; margin-top:0;">Route Justification:</h4>
                    <p style="color: #94a3b8; font-style: italic;">{routed.confidence.reasoning}</p>
                    <hr style="border-color: rgba(99,102,241,0.2);">
                    <h3 style="color: #6366f1;">Authoritative Answer:</h3>
                    <h2 style="color: #fff; margin-bottom: 12px;">{routed.answer}</h2>
                    <h4 style="color: #94a3b8; margin-top: 16px;">Detailed Explanation:</h4>
                    <p style="color: #cbd5e1; line-height: 1.6;">{routed.explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if routed.verdict:
                st.markdown(
                    f"""
                    <div class="judge-card">
                        <span class="badge-winner">WINNER: {routed.verdict.winning_agent}</span>
                        <h4 style="color: #10b981; margin-top:8px;">Impartial Judge Adjudication:</h4>
                        <p style="color: #cbd5e1;">{routed.verdict.evaluation_summary}</p>
                        <p style="color: #f87171;"><strong>Critique / Flaws:</strong> {', '.join(routed.verdict.identified_flaws) if routed.verdict.identified_flaws else 'None'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if routed.transcript:
                st.subheader("🥊 Deliberation Transcript")
                for rnd in routed.transcript.rounds:
                    st.markdown(f"**Round {rnd.round_number}:**")
                    t_cols = st.columns(len(rnd.turns))
                    for idx, turn in enumerate(rnd.turns):
                        with t_cols[idx]:
                            rev_badge = '<span class="badge-revised">REVISED</span> ' if turn.revised else ''
                            st.markdown(
                                f"""
                                <div class="agent-card">
                                    <strong>{turn.agent_name}</strong> {rev_badge}<br>
                                    <p style="font-size: 13px; color: #94a3b8; margin-top: 6px;">{turn.argument}</p>
                                    <strong>Answer:</strong> <code>{turn.answer}</code>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            if routed.self_consistency_result:
                st.subheader("Consensus Voting")
                st.bar_chart(routed.self_consistency_result.agreement_distribution)

            with st.expander("🔍 View Complete Raw Telemetry JSON"):
                st.json(routed.metadata)
