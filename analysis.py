from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_COLUMNS = {
    "participant_id",
    "참가순번",
    "조건",
    "학년",
    "문제번호",
    "정답",
    "1차답",
    "AI제안",
    "AI정답여부",
    "최종답",
    "제시순서",
}

PRIVATE_COLUMNS = ["이름", "전화번호"]

CASE_LABELS = {
    1: "성과 O · 수용 O · 유지 O",
    2: "성과 X · 수용 X · 유지 X",
    3: "성과 O · 수용 X · 유지 O",
    4: "성과 X · 수용 O · 유지 X",
    5: "성과 O · 수용 O · 유지 X",
    6: "성과 X · 수용 X · 유지 O",
    7: "성과 X · 수용 O · 유지 O",
    8: "성과 O · 수용 X · 유지 X",
}

DERIVED_CASES = {
    "올바른 의존": {1, 5},
    "올바르지 않은 의존": {4, 7},
    "의존 O": {1, 4, 5, 7},
    "의존 X": {2, 3, 6, 8},
    "올바른 불신": {3, 8},
    "올바르지 않은 불신": {2, 6},
    "불신 O": {2, 3, 6, 8},
    "불신 X": {1, 4, 5, 7},
    "올바른 고착": {1, 3},
    "올바르지 않은 고착": {6, 7},
    "고착 O": {1, 3, 6, 7},
    "고착 X": {2, 4, 5, 8},
    "올바른 수정": {5, 8},
    "올바르지 않은 수정": {2, 4},
    "수정 O": {2, 4, 5, 8},
    "수정 X": {1, 3, 6, 7},
}


@dataclass
class DataAudit:
    raw_rows: int
    raw_participants: int
    eligible_participants: int
    incomplete_participants: int
    duplicate_response_rows: int
    unclassified_grade_participants: int


def _normalize_answer(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def knowledge_group(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"\d+", text)
    if not match:
        if "대학원" in text:
            return "고지식"
        return "학년 미분류"
    grade = int(match.group())
    if grade in (1, 2):
        return "저지식"
    if grade >= 3:
        return "고지식"
    return "학년 미분류"


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, DataAudit]:
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(sorted(missing)))

    df = raw.copy()
    for col in ["참가순번", "문제번호", "학년", "정답", "1차답", "AI제안", "AI정답여부", "최종답", "제시순서"]:
        df[col] = _normalize_answer(df[col])

    df["참가순번_num"] = pd.to_numeric(df["참가순번"], errors="coerce")
    df["문제번호_num"] = pd.to_numeric(df["문제번호"], errors="coerce")
    df["제시순서_num"] = pd.to_numeric(df["제시순서"], errors="coerce")
    df["난이도"] = pd.cut(
        df["제시순서_num"], bins=[0, 4, 8, 12], labels=["하", "중", "상"]
    ).astype("string")
    df["지식수준"] = df["학년"].map(knowledge_group)

    valid = df[df["문제번호_num"].notna()].copy()
    duplicate_mask = valid.duplicated(["participant_id", "문제번호_num"], keep=False)
    duplicate_response_rows = int(duplicate_mask.sum())

    completion = (
        valid[valid["최종답"].notna()]
        .groupby("participant_id")["문제번호_num"]
        .nunique()
    )
    eligible_ids = completion[completion == 12].index
    eligible = valid[valid["participant_id"].isin(eligible_ids)].copy()

    eligible["성과"] = eligible["최종답"].eq(eligible["정답"])
    eligible["수용"] = eligible["최종답"].eq(eligible["AI제안"])
    eligible["유지"] = eligible["최종답"].eq(eligible["1차답"])
    eligible["AI답변유형"] = eligible["AI정답여부"].map(
        {"정답": "AI 정답", "오답": "AI 오답"}
    ).fillna("미분류")

    human = eligible["조건"].eq("Human First")
    case = pd.Series(pd.NA, index=eligible.index, dtype="Int64")
    mapping = {
        (True, True, True): 1,
        (False, False, False): 2,
        (True, False, True): 3,
        (False, True, False): 4,
        (True, True, False): 5,
        (False, False, True): 6,
        (False, True, True): 7,
        (True, False, False): 8,
    }
    for combo, case_no in mapping.items():
        mask = human.copy()
        for col, value in zip(["성과", "수용", "유지"], combo):
            mask &= eligible[col].eq(value)
        case.loc[mask] = case_no
    eligible["case"] = case

    for label, case_numbers in DERIVED_CASES.items():
        eligible[label] = eligible["case"].isin(case_numbers).where(human, pd.NA)

    p_meta = eligible.drop_duplicates("participant_id")
    audit = DataAudit(
        raw_rows=len(raw),
        raw_participants=raw["participant_id"].nunique(),
        eligible_participants=len(eligible_ids),
        incomplete_participants=raw["participant_id"].nunique() - len(eligible_ids),
        duplicate_response_rows=duplicate_response_rows,
        unclassified_grade_participants=int(
            p_meta.loc[p_meta["지식수준"].eq("학년 미분류"), "participant_id"].nunique()
        ),
    )
    return eligible, audit


def parse_participant_spec(text: str) -> set[int]:
    result: set[int] = set()
    if not text.strip():
        return result
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[-~]\s*(\d+)", token)
        if match:
            start, end = map(int, match.groups())
            result.update(range(min(start, end), max(start, end) + 1))
        elif token.isdigit():
            result.add(int(token))
        else:
            raise ValueError(f"해석할 수 없는 참가순번: {token}")
    return result


def participant_summary(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    keys = ["participant_id", "참가순번_num", "조건", "지식수준"]
    metrics = list(metrics)
    grouped = df.groupby(keys, as_index=False, dropna=False)[metrics].agg(
        lambda values: pd.to_numeric(values, errors="coerce").mean()
    )
    return grouped


def group_summary(participant_df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    rows = []
    for group, values in participant_df.groupby(group_col, dropna=False)[metric]:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        n = len(clean)
        mean = clean.mean() if n else np.nan
        sd = clean.std(ddof=1) if n > 1 else np.nan
        if n > 1:
            margin = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        else:
            margin = np.nan
        rows.append(
            {
                group_col: group,
                "참가자 수": n,
                "평균": mean,
                "표준편차": sd,
                "신뢰구간 하한": max(0.0, mean - margin) if n > 1 else np.nan,
                "신뢰구간 상한": min(1.0, mean + margin) if n > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def safe_export(df: pd.DataFrame) -> pd.DataFrame:
    dropped = [c for c in PRIVATE_COLUMNS if c in df.columns]
    return df.drop(columns=dropped + ["participant_id"], errors="ignore")
