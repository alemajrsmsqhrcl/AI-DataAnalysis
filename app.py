from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import streamlit as st
from analysis import CASE_LABELS, parse_participant_spec, participant_summary, prepare_data, safe_export

GROUPS = ["저지식 · Human First", "저지식 · AI First", "고지식 · Human First", "고지식 · AI First"]
COLORS = {
    "저지식 · Human First": "#2563B8",
    "저지식 · AI First": "#70A7E8",
    "고지식 · Human First": "#C94F24",
    "고지식 · AI First": "#F29A62",
}

st.set_page_config(page_title="AI 개입 실험 분석", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp{background:#f5f7fa;color:#172033}.block-container{max-width:1180px;padding-top:1.25rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #dfe4ea}
[data-testid="stMetric"]{background:#fff;border:1px solid #dfe4ea;border-radius:10px;padding:10px 12px}
[data-testid="stMetricLabel"]{font-size:.82rem;font-weight:650;color:#536174}[data-testid="stMetricValue"]{font-size:1.45rem}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid #d9e0e8;border-radius:12px;padding:8px 10px;box-shadow:0 2px 8px rgba(15,23,42,.04)}
h1{font-size:2rem!important;margin-bottom:.1rem!important}h2{font-size:1.45rem!important}h3{font-size:1.15rem!important}
.summary-note{background:#fff;border:1px solid #dfe4ea;border-radius:9px;padding:.65rem .8rem;color:#465469;margin:.4rem 0 .8rem}
</style>""", unsafe_allow_html=True)

st.title("AI 개입 실험 분석")
st.caption("선택한 참가자를 전문지식 수준과 AI 제시 방식에 따라 자동 분류해 비교합니다.")
uploaded = st.file_uploader("실험 CSV 파일", type=["csv"])
if uploaded is None:
    st.info("분석을 시작하려면 CSV 파일을 올려주세요. 파일은 현재 로컬 세션에서만 처리됩니다.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try: return pd.read_csv(io.BytesIO(data), encoding=encoding, dtype=str)
        except UnicodeDecodeError: pass
    raise ValueError("CSV 문자 인코딩을 확인할 수 없습니다.")

try:
    raw = load_csv(uploaded.getvalue())
    prepared, audit = prepare_data(raw)
except Exception as exc:
    st.error(f"파일을 분석할 수 없습니다. {exc}")
    st.stop()

# Streamlit 개발 서버가 변경 전 analysis 모듈을 잠시 유지해도 난이도를 안전하게 생성합니다.
if "난이도" not in prepared.columns:
    order = pd.to_numeric(prepared["제시순서"], errors="coerce")
    prepared["난이도"] = pd.cut(
        order, bins=[0, 4, 8, 12], labels=["하", "중", "상"]
    ).astype("string")

prepared["자동분류"] = prepared["지식수준"] + " · " + prepared["조건"]
available = set(prepared["참가순번_num"].dropna().astype(int))

with st.sidebar:
    st.header("분석 설정")
    st.subheader("참가자 선택")
    selection_mode = st.radio("선택 방식", ["전체", "범위 지정", "직접 입력"], horizontal=True)
    requested, input_error = set(), None
    if selection_mode == "범위 지정" and available:
        start = st.number_input("시작 순번", min_value=0, value=min(available), step=1)
        end = st.number_input("종료 순번", min_value=0, value=max(available), step=1)
        requested = set(range(min(start, end), max(start, end) + 1))
    elif selection_mode == "직접 입력":
        spec = st.text_area("참가순번", placeholder="예: 2, 6, 14, 70-73", height=80)
        try: requested = parse_participant_spec(spec)
        except ValueError as exc: input_error = str(exc)
    else: requested = available.copy()

    exclude_spec = st.text_input("제외할 참가순번", placeholder="예: 5, 12, 20-24")
    try:
        excluded_numbers = parse_participant_spec(exclude_spec)
    except ValueError as exc:
        excluded_numbers = set()
        input_error = str(exc)

    st.divider(); st.subheader("분석 범위")
    ai_answer_filter = st.radio("AI 답변 유형", ["전체", "AI 정답", "AI 오답"], horizontal=True)

    selected_numbers = (requested & available) - excluded_numbers
    missing_numbers = requested - available
    applied_exclusions = (requested & available) & excluded_numbers
    selected = prepared[prepared["참가순번_num"].isin(selected_numbers)].copy()
    participant_meta = selected.drop_duplicates("participant_id")
    st.divider(); st.subheader("표시할 집단")
    selected_groups = []
    for group in GROUPS:
        group_participants = participant_meta[participant_meta["자동분류"].eq(group)]
        numbers = sorted(group_participants["참가순번_num"].dropna().astype(int).unique())
        n = len(numbers)
        if st.checkbox(f"{group} ({n}명)", value=True, key=f"group_{group}"): selected_groups.append(group)
        with st.popover("참가순번 보기", use_container_width=True):
            st.caption(", ".join(map(str, numbers)) if numbers else "해당 참가자 없음")
    st.divider(); st.subheader("분석 항목")
    show_basic = st.checkbox("기본 지표 (성과·수용·유지)", value=True)
    with st.expander("기본 지표 개별 선택"):
        individual_metrics = [m for m in ["성과", "수용", "유지"] if st.checkbox(m, key=f"metric_{m}")]
    show_ai_answer_comparison = st.checkbox("AI 정답 여부별 비교", value=False)
    show_difficulty_comparison = st.checkbox("난이도별 비교", value=False)
    show_cases = st.checkbox("8개 케이스", value=False)
    show_derived = {name: st.checkbox(f"{name}도", value=False, key=f"derived_{name}") for name in ["의존", "불신", "고착", "수정"]}

if input_error: st.error(input_error); st.stop()
if selection_mode == "직접 입력" and not requested: st.info("사이드바에 분석할 참가순번을 입력해주세요."); st.stop()
if missing_numbers:
    preview = ", ".join(map(str, sorted(missing_numbers)[:20]))
    st.warning(f"12문항 완주 데이터가 없어 제외된 순번: {preview}{' …' if len(missing_numbers)>20 else ''}")
if applied_exclusions:
    st.caption("직접 제외된 참가순번: " + ", ".join(map(str, sorted(applied_exclusions))))

group_filtered = selected[selected["자동분류"].isin(selected_groups)].copy()
filtered = group_filtered.copy()
if ai_answer_filter != "전체":
    filtered = filtered[filtered["AI답변유형"].eq(ai_answer_filter)].copy()
if filtered.empty: st.warning("현재 선택에 해당하는 12문항 완주자가 없습니다."); st.stop()
participants = filtered.drop_duplicates("participant_id")
counts = participants["자동분류"].value_counts().reindex(GROUPS, fill_value=0)
st.markdown(f'<div class="summary-note"><b>{len(participants)}명</b> · {len(filtered)}건 응답 · AI 답변 유형: <b>{ai_answer_filter}</b> &nbsp;|&nbsp; '+" &nbsp;·&nbsp; ".join(f"{g} {counts[g]}명" for g in GROUPS)+"</div>", unsafe_allow_html=True)

def draw_bar(data, x, y, y_title):
    if data.empty: return
    present_groups = [group for group in GROUPS if group in set(data["자동분류"])]
    max_value = pd.to_numeric(data[y], errors="coerce").max()
    upper = min(1.16, max(0.35, float(max_value) + 0.14))
    fig = px.bar(
        data, x=x, y=y, color="자동분류", barmode="group", text_auto=".1%",
        category_orders={"자동분류": GROUPS}, color_discrete_map=COLORS,
    )
    fig.update_traces(
        width=max(.13, .44 / max(1, len(present_groups))),
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13, color="#172033"),
        marker_line=dict(color="rgba(255,255,255,.9)", width=1),
        hovertemplate="%{fullData.name}<br>%{x}<br><b>%{y:.1%}</b><extra></extra>",
    )
    fig.update_layout(
        height=305,
        yaxis=dict(
            tickformat=".0%", range=[0, upper], title=y_title,
            tickfont=dict(size=12, color="#536174"), title_font=dict(size=13, color="#344054"),
            gridcolor="#E7EBF0", gridwidth=1, zerolinecolor="#BFC8D4",
        ),
        xaxis=dict(
            title=None, tickfont=dict(size=14, color="#243247"),
            showline=True, linecolor="#C9D1DC",
        ),
        legend=dict(
            title=None, orientation="h", yanchor="bottom", y=1.03,
            xanchor="left", x=0, font=dict(size=12, color="#344054"),
            bgcolor="rgba(0,0,0,0)", itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=14, r=14, t=58, b=12), bargap=.30, hoverlabel=dict(font_size=13),
    )
    st.plotly_chart(fig, width="stretch")

metrics = (["성과","수용","유지"] if show_basic else []) + [m for m in individual_metrics if m not in (["성과","수용","유지"] if show_basic else [])]
if metrics:
    st.subheader("기본 지표")
    rows=[]
    for group in selected_groups:
        subset=filtered[filtered["자동분류"].eq(group)]
        p=participant_summary(subset, metrics) if not subset.empty else pd.DataFrame()
        for metric in metrics:
            unsupported=metric=="유지" and group.endswith("AI First")
            rows.append({"자동분류":group,"지표":metric,"평균":pd.NA if unsupported or p.empty else p[metric].mean(),"참가자 수":0 if p.empty else len(p)})
    basic=pd.DataFrame(rows)
    draw_bar(basic.dropna(subset=["평균"]),"지표","평균","참가자별 비율 평균")
    if "유지" in metrics and any(g.endswith("AI First") for g in selected_groups): st.caption("AI First에는 독립적인 1차 답변이 없으므로 유지율은 ‘해당 없음’입니다.")

if show_ai_answer_comparison:
    st.subheader("AI 정답 여부별 비교")
    comparison_rows = []
    for group in selected_groups:
        for answer_type in ["AI 정답", "AI 오답"]:
            subset = group_filtered[
                group_filtered["자동분류"].eq(group)
                & group_filtered["AI답변유형"].eq(answer_type)
            ]
            comparison_metrics = ["성과", "수용", "유지"]
            p = participant_summary(subset, comparison_metrics) if not subset.empty else pd.DataFrame()
            for metric in comparison_metrics:
                if metric == "유지" and group.endswith("AI First"):
                    continue
                comparison_rows.append({
                    "자동분류": group,
                    "비교 항목": f"{metric} · {answer_type}",
                    "평균": pd.NA if p.empty else p[metric].mean(),
                })
    comparison_data = pd.DataFrame(comparison_rows).dropna(subset=["평균"])
    draw_bar(comparison_data, "비교 항목", "평균", "참가자별 비율 평균")
    st.caption("이 비교 그래프는 위의 AI 답변 유형 필터와 관계없이 AI 정답과 AI 오답 응답을 함께 비교합니다.")

if show_difficulty_comparison:
    st.subheader("난이도별 비교")
    difficulty_rows = []
    for group in selected_groups:
        for difficulty in ["하", "중", "상"]:
            subset = group_filtered[
                group_filtered["자동분류"].eq(group)
                & group_filtered["난이도"].eq(difficulty)
            ]
            difficulty_metrics = ["성과", "수용", "유지"]
            p = participant_summary(subset, difficulty_metrics) if not subset.empty else pd.DataFrame()
            for metric in difficulty_metrics:
                if metric == "유지" and group.endswith("AI First"):
                    continue
                difficulty_rows.append({
                    "자동분류": group,
                    "난이도": difficulty,
                    "지표": metric,
                    "평균": pd.NA if p.empty else p[metric].mean(),
                })
    difficulty_data = pd.DataFrame(difficulty_rows).dropna(subset=["평균"])
    if not difficulty_data.empty:
        fig = px.bar(
            difficulty_data, x="난이도", y="평균", color="자동분류",
            facet_col="지표", barmode="group", text_auto=".1%",
            category_orders={"자동분류": GROUPS, "난이도": ["하", "중", "상"], "지표": ["성과", "수용", "유지"]},
            color_discrete_map=COLORS,
        )
        fig.update_traces(
            textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color="#172033"),
            marker_line=dict(color="rgba(255,255,255,.9)", width=1),
            hovertemplate="%{fullData.name}<br>난이도 %{x}<br><b>%{y:.1%}</b><extra></extra>",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1.14], gridcolor="#E7EBF0", title=None)
        fig.update_xaxes(title=None, showline=True, linecolor="#C9D1DC")
        fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
        fig.update_layout(
            height=340, legend=dict(title=None, orientation="h", yanchor="bottom", y=1.08, x=0),
            plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=12, r=12, t=72, b=12),
            bargap=.28, hoverlabel=dict(font_size=13),
        )
        st.plotly_chart(fig, width="stretch")
    st.caption("난이도는 제시순서 1–4=하, 5–8=중, 9–12=상으로 구분하며 AI 답변 유형 필터와 독립적으로 계산합니다.")

human=filtered[filtered["조건"].eq("Human First")].copy()
human_groups=[g for g in selected_groups if g.endswith("Human First") and counts[g]>0]
if show_cases:
    st.subheader("8개 케이스")
    chosen=st.multiselect("표시할 케이스",list(CASE_LABELS),default=list(CASE_LABELS),format_func=lambda n:f"Case {n} · {CASE_LABELS[n]}")
    rows=[]
    for group in human_groups:
        subset=human[human["자동분류"].eq(group)]; denominator=len(subset)
        for case_no in chosen:
            count=int(subset["case"].eq(case_no).sum()); rows.append({"자동분류":group,"케이스":f"Case {case_no}","응답 비율":count/denominator,"응답 수":count})
    data=pd.DataFrame(rows)
    if data.empty: st.info("8개 케이스는 Human First 집단에서만 분석할 수 있습니다.")
    else:
        draw_bar(data,"케이스","응답 비율","그룹 내 응답 비율")
    with st.expander("케이스 정의 보기"):
        st.dataframe(pd.DataFrame([{"케이스":f"Case {n}","정의":v} for n,v in CASE_LABELS.items()]),hide_index=True,width="stretch")

for category,enabled in show_derived.items():
    if not enabled: continue
    st.subheader(f"{category}도")
    details=st.toggle("자세히 보기",value=False,key=f"detail_{category}",help=f"{category} O를 올바른 {category}과 올바르지 않은 {category}으로 나눕니다.")
    category_metrics=[f"올바른 {category}",f"올바르지 않은 {category}",f"{category} X"] if details else [f"{category} O",f"{category} X"]
    rows=[]
    for group in human_groups:
        subset=human[human["자동분류"].eq(group)]; p=participant_summary(subset,category_metrics)
        for metric in category_metrics: rows.append({"자동분류":group,"구분":metric,"평균":p[metric].mean(),"참가자 수":len(p)})
    data=pd.DataFrame(rows)
    if data.empty: st.info(f"{category}도는 Human First 집단에서만 분석할 수 있습니다.")
    else:
        draw_bar(data,"구분","평균","참가자별 발생률 평균")

if not metrics and not show_cases and not show_ai_answer_comparison and not show_difficulty_comparison and not any(show_derived.values()): st.info("사이드바에서 보고 싶은 분석 항목을 선택해주세요.")
with st.expander("데이터 품질과 다운로드"):
    q1,q2,q3,q4=st.columns(4)
    q1.metric("원본 참가자",f"{audit.raw_participants}명"); q2.metric("12문항 완주자",f"{audit.eligible_participants}명"); q3.metric("제외된 미완주자",f"{audit.incomplete_participants}명"); q4.metric("중복 의심 응답행",f"{audit.duplicate_response_rows}건")
    st.download_button("현재 분석 데이터 CSV 다운로드",safe_export(filtered).to_csv(index=False,encoding="utf-8-sig"),file_name="filtered_analysis_data.csv",mime="text/csv")
    st.caption("이름, 전화번호, 내부 참가자 ID는 다운로드 파일에서 제외됩니다.")
