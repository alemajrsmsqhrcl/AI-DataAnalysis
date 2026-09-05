from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    CASE_LABELS,
    DERIVED_CASES,
    group_summary,
    parse_participant_spec,
    participant_summary,
    prepare_data,
    safe_export,
)

RATE_LABELS = {"성과": "성과율", "수용": "수용률", "유지": "유지율"}


st.set_page_config(page_title="AI 개입 실험 분석", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f3f5f8; color: #172033; }
    .block-container { max-width: 1180px; padding-top: 1.25rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #d9e0e8; border-radius: 10px;
        padding: 10px 12px; box-shadow: 0 2px 8px rgba(15, 23, 42, .04);
    }
    [data-testid="stMetricLabel"] { font-size: .82rem; font-weight: 650; color: #465469; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; color: #172033; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d9e0e8; }
    [data-testid="stPlotlyChart"] { background: white; border: 1px solid #dfe5ec; border-radius: 10px; padding: 4px; }
    h1 { font-size: 2rem !important; margin-bottom: .15rem !important; }
    h2, h3 { color: #172033; letter-spacing: -0.02em; margin-top: 1.1rem !important; }
    h3 { font-size: 1.25rem !important; }
    hr { margin: 1rem 0 !important; }
    .filter-note { color:#465469; font-size:.86rem; padding:.45rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI 개입 실험 분석")
st.caption("도메인 지식 수준과 AI 제시 방식에 따른 응답 결과를 탐색합니다.")

uploaded = st.file_uploader("실험 CSV 파일", type=["csv"])
if uploaded is None:
    st.info("분석을 시작하려면 CSV 파일을 올려주세요. 파일은 현재 로컬 세션에서만 처리됩니다.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 문자 인코딩을 확인할 수 없습니다.")


try:
    raw = load_csv(uploaded.getvalue())
    prepared, audit = prepare_data(raw)
except Exception as exc:
    st.error(f"파일을 분석할 수 없습니다. {exc}")
    st.stop()

with st.sidebar:
    st.header("분석 설정")
    analysis_mode = st.radio(
        "분석 방식", ["한 집단 내에서의 분석", "서로 다른 집단 간의 분석"]
    )
    if analysis_mode == "한 집단 내에서의 분석":
        condition = st.radio("실험 조건", ["Human First", "AI First"], horizontal=True)
        knowledge = st.multiselect(
            "전문지식 수준", ["저지식", "고지식", "학년 미분류"], default=["저지식", "고지식"]
        )

        available_numbers = prepared["참가순번_num"].dropna().astype(int)
        min_available = int(available_numbers.min()) if len(available_numbers) else 0
        max_available = int(available_numbers.max()) if len(available_numbers) else 0
        start_no = st.number_input("시작 참가순번", value=min_available, step=1)
        end_no = st.number_input("종료 참가순번", value=max_available, step=1)
        exclude_text = st.text_input("제외할 참가순번", placeholder="예: 45, 51, 58-60")
    else:
        st.caption("다른 필터 없이 입력한 참가순번만으로 두 그룹을 비교합니다.")
        group_a_text = st.text_input(
            "그룹 A 참가순번", value="", placeholder="예: 26-30, 35", key="group_a"
        )
        group_b_text = st.text_input(
            "그룹 B 참가순번", value="", placeholder="예: 36-40, 45", key="group_b"
        )

    st.divider()
    selected_difficulties = st.multiselect("난이도", ["하", "중", "상"], default=["하", "중", "상"])

if analysis_mode == "서로 다른 집단 간의 분석":
    st.markdown('<div class="filter-note">분석 방식: <b>서로 다른 집단 간의 분석</b></div>', unsafe_allow_html=True)
    if not group_a_text.strip() or not group_b_text.strip():
        st.info("사이드바의 그룹 A와 그룹 B에 비교할 참가순번을 입력해주세요.")
        st.stop()
    try:
        group_specs = {
            "그룹 A": parse_participant_spec(group_a_text),
            "그룹 B": parse_participant_spec(group_b_text),
        }
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    overlap = group_specs["그룹 A"] & group_specs["그룹 B"]
    if overlap:
        st.error("두 그룹에 중복된 참가순번이 있습니다: " + ", ".join(map(str, sorted(overlap))))
        st.stop()

    comparison_base = prepared[prepared["난이도"].isin(selected_difficulties)].copy()
    selected_comparison = comparison_base[
        comparison_base["참가순번_num"].isin(group_specs["그룹 A"] | group_specs["그룹 B"])
    ]
    conditions_in_comparison = set(selected_comparison["조건"].dropna().unique())
    comparison_metrics = ["성과", "수용"]
    if conditions_in_comparison == {"Human First"}:
        comparison_metrics.append("유지")

    custom_rows = []
    detail_rows = []
    response_groups = {}
    participant_groups = []
    for name, numbers in group_specs.items():
        subset = comparison_base[comparison_base["참가순번_num"].isin(numbers)]
        response_groups[name] = subset
        found = set(subset["참가순번_num"].dropna().astype(int).unique())
        missing = sorted(numbers - found)
        p = participant_summary(subset, comparison_metrics)
        p["그룹"] = name
        participant_groups.append(p)
        detail_rows.append(
            {
                "그룹": name,
                "입력 인원": len(numbers),
                "분석 인원": len(p),
                "실험 조건": ", ".join(sorted(subset["조건"].dropna().unique())) or "–",
                "지식 수준": ", ".join(sorted(subset["지식수준"].dropna().unique())) or "–",
                "분석 제외 순번": ", ".join(map(str, missing)) if missing else "없음",
            }
        )
        for metric in comparison_metrics:
            custom_rows.append(
                {
                    "그룹": name,
                    "지표": metric,
                    "평균": p[metric].mean() if not p.empty else float("nan"),
                    "참가자 수": len(p),
                }
            )

    st.subheader("그룹 A vs 그룹 B")
    st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")
    custom = pd.DataFrame(custom_rows)
    if custom["참가자 수"].eq(0).any():
        st.warning("한쪽 그룹에 분석 가능한 12문항 완주자가 없습니다.")
        st.stop()

    fig = px.bar(
        custom, x="지표", y="평균", color="그룹", barmode="group", text_auto=".1%",
        color_discrete_map={"그룹 A": "#4F7CAC", "그룹 B": "#E5824B"}, hover_data=["참가자 수"],
    )
    fig.update_layout(
        height=290, yaxis_tickformat=".0%", yaxis_range=[0, 1.05], yaxis_title="참가자별 비율 평균",
        xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white", bargap=.42,
        margin=dict(l=10, r=10, t=12, b=10), font=dict(size=12, color="#344054"),
    )
    fig.update_traces(width=.28)
    st.plotly_chart(fig, width="stretch")
    comparison_table = custom.pivot(index="지표", columns="그룹", values="평균").reset_index()
    comparison_table["차이(A-B)"] = comparison_table["그룹 A"] - comparison_table["그룹 B"]
    for col in ["그룹 A", "그룹 B", "차이(A-B)"]:
        comparison_table[col] = comparison_table[col].map(lambda value: f"{value:.1%}")
    st.dataframe(comparison_table, hide_index=True, width="stretch")
    if conditions_in_comparison != {"Human First"}:
        st.info("AI First가 포함된 비교에서는 두 조건에 공통인 성과율과 수용률만 표시합니다.")

    st.subheader("참가자별 분포")
    comparison_metric = st.selectbox("표시 지표", comparison_metrics, key="between_metric")
    participant_comparison = pd.concat(participant_groups, ignore_index=True)
    fig = px.strip(
        participant_comparison, x="그룹", y=comparison_metric, color="그룹",
        color_discrete_map={"그룹 A": "#4F7CAC", "그룹 B": "#E5824B"},
        hover_data={"참가순번_num": True, comparison_metric: ":.1%"},
    )
    fig.update_traces(jitter=0.18, marker=dict(size=10, opacity=.75))
    fig.update_layout(
        height=270, showlegend=False, yaxis_tickformat=".0%", yaxis_range=[-0.05, 1.05],
        xaxis_title=None, yaxis_title=f"참가자별 {RATE_LABELS[comparison_metric]}",
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=12, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    if conditions_in_comparison == {"Human First"}:
        st.subheader("8개 케이스")
        chosen_cases_between = st.multiselect(
            "표시할 케이스", list(CASE_LABELS), default=list(CASE_LABELS), key="between_cases",
            format_func=lambda n: f"Case {n} · {CASE_LABELS[n]}",
        )
        case_rows = []
        for name, subset in response_groups.items():
            denominator = len(subset)
            counts = subset[subset["case"].isin(chosen_cases_between)]["case"].value_counts()
            for case_no, count in counts.items():
                case_rows.append(
                    {"그룹": name, "케이스": f"Case {int(case_no)}", "응답 비율": count / denominator, "응답 수": count}
                )
        if case_rows:
            case_comparison = pd.DataFrame(case_rows)
            fig = px.bar(
                case_comparison, x="케이스", y="응답 비율", color="그룹", barmode="group",
                text_auto=".1%", color_discrete_map={"그룹 A": "#4F7CAC", "그룹 B": "#E5824B"},
                hover_data=["응답 수"],
            )
            fig.update_layout(height=300, yaxis_tickformat=".0%", yaxis_title="그룹 내 응답 비율",
                              xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white", bargap=.38,
                              margin=dict(l=10, r=10, t=12, b=10))
            fig.update_traces(width=.3)
            st.plotly_chart(fig, width="stretch")

        st.subheader("의존·불신·고착·수정")
        between_category = st.selectbox("분석 관점", ["의존", "불신", "고착", "수정"], key="between_category")
        between_details = st.toggle("자세히 보기", value=False, key="between_details")
        if between_details:
            derived_metrics = [
                f"올바른 {between_category}", f"올바르지 않은 {between_category}", f"{between_category} X"
            ]
        else:
            derived_metrics = [f"{between_category} O", f"{between_category} X"]
        derived_rows = []
        for name, subset in response_groups.items():
            p = participant_summary(subset, derived_metrics)
            for metric in derived_metrics:
                derived_rows.append({"그룹": name, "구분": metric, "평균": p[metric].mean()})
        fig = px.bar(
            pd.DataFrame(derived_rows), x="구분", y="평균", color="그룹", barmode="group", text_auto=".1%",
            color_discrete_map={"그룹 A": "#4F7CAC", "그룹 B": "#E5824B"},
        )
        fig.update_layout(height=290, yaxis_tickformat=".0%", yaxis_title="참가자별 발생률 평균",
                          xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white", bargap=.42,
                          margin=dict(l=10, r=10, t=12, b=10))
        fig.update_traces(width=.28)
        st.plotly_chart(fig, width="stretch")

    st.subheader("난이도별 결과")
    difficulty_metric = st.selectbox("난이도별 표시 지표", comparison_metrics, key="between_difficulty_metric")
    difficulty_rows = []
    for name, subset in response_groups.items():
        summary_by_difficulty = subset.groupby("난이도", dropna=False)[difficulty_metric].mean()
        for difficulty, value in summary_by_difficulty.items():
            difficulty_rows.append({"그룹": name, "난이도": difficulty, "비율": value})
    difficulty_comparison = pd.DataFrame(difficulty_rows)
    difficulty_comparison["난이도"] = pd.Categorical(
        difficulty_comparison["난이도"], categories=["하", "중", "상"], ordered=True
    )
    difficulty_comparison = difficulty_comparison.sort_values("난이도")
    fig = px.bar(
        difficulty_comparison, x="난이도", y="비율", color="그룹", barmode="group", text_auto=".1%",
        color_discrete_map={"그룹 A": "#4F7CAC", "그룹 B": "#E5824B"},
    )
    fig.update_layout(height=290, yaxis_tickformat=".0%", yaxis_range=[0, 1.05], bargap=.42,
                      xaxis_title="난이도", yaxis_title=RATE_LABELS[difficulty_metric],
                      plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=12, b=10))
    fig.update_traces(width=.28)
    st.plotly_chart(fig, width="stretch")
    st.stop()

try:
    excluded = parse_participant_spec(exclude_text)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()

filtered = prepared[
    prepared["조건"].eq(condition)
    & prepared["지식수준"].isin(knowledge)
    & prepared["참가순번_num"].between(start_no, end_no)
    & ~prepared["참가순번_num"].isin(excluded)
    & prepared["난이도"].isin(selected_difficulties)
].copy()

participant_count = filtered["participant_id"].nunique()
response_count = len(filtered)
st.markdown(
    f'<div class="filter-note">현재 조건: <b>{condition}</b> · '
    f'전문지식 수준: <b>{", ".join(knowledge) or "선택 없음"}</b> · '
    f'참가순번: <b>{int(start_no)}–{int(end_no)}</b> · '
    f'난이도: <b>{", ".join(selected_difficulties) or "선택 없음"}</b></div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("현재 필터에 해당하는 완주자 응답이 없습니다.")
    st.stop()

metrics = ["성과", "수용"] if condition == "AI First" else ["성과", "수용", "유지"]
participant = participant_summary(filtered, metrics)

metric_cols = st.columns(2 + len(metrics))
metric_cols[0].metric("분석 참가자", f"{participant_count}명")
metric_cols[1].metric("분석 응답", f"{response_count}건")
for col, metric in zip(metric_cols[2:], metrics):
    col.metric(RATE_LABELS[metric], f"{filtered[metric].mean():.1%}")

st.divider()
st.subheader("전문지식 수준별 결과")
selected_metric = st.selectbox("표시 지표", metrics, label_visibility="collapsed")
summary = group_summary(participant, selected_metric, "지식수준")

left, right = st.columns([1.7, 1])
with left:
    chart = go.Figure()
    colors = {"저지식": "#4F7CAC", "고지식": "#12A594", "학년 미분류": "#7A8493"}
    for _, row in summary.iterrows():
        error_plus = row["신뢰구간 상한"] - row["평균"] if pd.notna(row["신뢰구간 상한"]) else 0
        error_minus = row["평균"] - row["신뢰구간 하한"] if pd.notna(row["신뢰구간 하한"]) else 0
        chart.add_trace(
            go.Bar(
                x=[row["지식수준"]], y=[row["평균"]], name=row["지식수준"],
                marker_color=colors.get(row["지식수준"], "#7A8493"),
                text=[f'{row["평균"]:.1%}'], textposition="outside",
                error_y=dict(type="data", array=[error_plus], arrayminus=[error_minus]),
                hovertemplate="%{x}<br>평균 %{y:.1%}<extra></extra>",
            )
        )
    chart.update_layout(
        height=290, showlegend=False, margin=dict(l=10, r=10, t=12, b=10), bargap=.48,
        yaxis=dict(tickformat=".0%", range=[0, 1.08], title=f"참가자별 {RATE_LABELS[selected_metric]} 평균"),
        xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white",
    )
    chart.update_traces(width=.42)
    st.plotly_chart(chart, width="stretch")
with right:
    display_summary = summary.copy()
    for col in ["평균", "표준편차", "신뢰구간 하한", "신뢰구간 상한"]:
        display_summary[col] = display_summary[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "–")
    st.dataframe(display_summary, hide_index=True, width="stretch")
    st.caption("오차선은 참가자별 비율 평균의 95% 신뢰구간입니다.")

st.subheader("참가자별 분포")
dist = participant.rename(columns={selected_metric: "비율"})
fig = px.strip(
    dist, x="지식수준", y="비율", color="지식수준",
    color_discrete_map={"저지식": "#4F7CAC", "고지식": "#12A594", "학년 미분류": "#7A8493"},
    hover_data={"참가순번_num": True, "비율": ":.1%"},
)
fig.update_traces(jitter=0.18, marker=dict(size=10, opacity=.75))
fig.update_layout(height=270, showlegend=False, yaxis_tickformat=".0%", yaxis_range=[-0.05, 1.05],
                  xaxis_title=None, yaxis_title=f"참가자별 {RATE_LABELS[selected_metric]}",
                  plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=12, b=10))
st.plotly_chart(fig, width="stretch")

if condition == "Human First":
    st.divider()
    st.subheader("8개 케이스")
    chosen_cases = st.multiselect(
        "표시할 케이스", list(CASE_LABELS), default=list(CASE_LABELS),
        format_func=lambda n: f"Case {n} · {CASE_LABELS[n]}",
    )
    case_counts = (
        filtered[filtered["case"].isin(chosen_cases)]
        .groupby(["지식수준", "case"], observed=True)
        .size().rename("응답 수").reset_index()
    )
    denominators = filtered.groupby("지식수준").size()
    if not case_counts.empty:
        case_counts["응답 비율"] = case_counts.apply(
            lambda row: row["응답 수"] / denominators[row["지식수준"]], axis=1
        )
        case_counts["케이스"] = case_counts["case"].map(lambda n: f"Case {int(n)}")
        fig = px.bar(
            case_counts, x="케이스", y="응답 비율", color="지식수준", barmode="group",
            color_discrete_map={"저지식": "#4F7CAC", "고지식": "#12A594"},
            text_auto=".1%", hover_data={"응답 수": True, "응답 비율": ":.1%"},
        )
        fig.update_layout(height=300, yaxis_tickformat=".0%", yaxis_title="전체 응답 중 비율",
                          xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white", bargap=.38,
                          margin=dict(l=10, r=10, t=12, b=10))
        fig.update_traces(width=.3)
        st.plotly_chart(fig, width="stretch")
        with st.expander("케이스 정의 보기"):
            st.dataframe(
                pd.DataFrame([{"케이스": f"Case {n}", "정의": label} for n, label in CASE_LABELS.items()]),
                hide_index=True, width="stretch",
            )

    st.subheader("의존·불신·고착·수정")
    category = st.selectbox("분석 관점", ["의존", "불신", "고착", "수정"])
    show_details = st.toggle("자세히 보기", value=False, help=f"{category} O를 올바른 결과와 올바르지 않은 결과로 나눕니다.")
    if show_details:
        category_metrics = [f"올바른 {category}", f"올바르지 않은 {category}", f"{category} X"]
    else:
        category_metrics = [f"{category} O", f"{category} X"]
    long_rows = []
    for metric in category_metrics:
        p = participant_summary(filtered, [metric])
        for _, row in group_summary(p, metric, "지식수준").iterrows():
            long_rows.append({"지식수준": row["지식수준"], "구분": metric, "평균": row["평균"], "참가자 수": row["참가자 수"]})
    derived_summary = pd.DataFrame(long_rows)
    fig = px.bar(
        derived_summary, x="구분", y="평균", color="지식수준", barmode="group",
        color_discrete_map={"저지식": "#4F7CAC", "고지식": "#12A594"}, text_auto=".1%",
    )
    fig.update_layout(height=290, yaxis_tickformat=".0%", yaxis_title="참가자별 발생률 평균",
                      xaxis_title=None, plot_bgcolor="white", paper_bgcolor="white", bargap=.42,
                      margin=dict(l=10, r=10, t=12, b=10))
    fig.update_traces(width=.28)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("AI First는 독립적인 1차 답변이 없어 성과와 AI 제안 일치율만 분석합니다.")

st.divider()
st.subheader("난이도별 결과")
difficulty_summary = filtered.groupby("난이도", dropna=False)[metrics].mean().reset_index()
difficulty_summary["난이도"] = pd.Categorical(
    difficulty_summary["난이도"], categories=["하", "중", "상"], ordered=True
)
difficulty_summary = difficulty_summary.sort_values("난이도")
difficulty_long = difficulty_summary.melt(
    id_vars=["난이도"], value_vars=metrics, var_name="지표", value_name="비율"
)
fig = px.bar(
    difficulty_long, x="난이도", y="비율", color="지표", barmode="group", text_auto=".1%",
    color_discrete_map={"성과": "#4F7CAC", "수용": "#E5824B", "유지": "#12A594"},
)
fig.update_layout(height=290, yaxis_tickformat=".0%", yaxis_range=[0, 1.05], bargap=.42,
                  xaxis_title="난이도", yaxis_title="응답 비율", plot_bgcolor="white", paper_bgcolor="white",
                  margin=dict(l=10, r=10, t=12, b=10))
fig.update_traces(width=.28)
st.plotly_chart(fig, width="stretch")

with st.expander("데이터 품질과 다운로드"):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("원본 참가자", f"{audit.raw_participants}명")
    q2.metric("12문항 완주자", f"{audit.eligible_participants}명")
    q3.metric("제외된 미완주자", f"{audit.incomplete_participants}명")
    q4.metric("중복 의심 응답행", f"{audit.duplicate_response_rows}건")
    if audit.unclassified_grade_participants:
        st.warning(f"학년을 분류하지 못한 완주자가 {audit.unclassified_grade_participants}명 있습니다.")
    sanitized = safe_export(filtered)
    st.download_button(
        "현재 분석 데이터 CSV 다운로드",
        sanitized.to_csv(index=False, encoding="utf-8-sig"),
        file_name="filtered_analysis_data.csv",
        mime="text/csv",
    )
    st.caption("이름, 전화번호, 내부 참가자 ID는 다운로드 파일에서 제외됩니다.")
