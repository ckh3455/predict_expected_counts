# predict_expected_counts.py
# 목적: "거래요약" 시트의 '예상건수' 행을 매일 업데이트 (전국, 서울)
# - 한국 공휴일 반영
# - 등록지연 규칙(주말/공휴일 제외, 다음영업일 등록) 반영
# - 과거 12개월 누적완성도 곡선으로 현재 누적→최종치 추정
# 실행: python predict_expected_counts.py

import os, re, math, datetime as dt
from dataclasses import dataclass
import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import holidays

# ====== 환경변수 ======
SHEET_ID    = os.getenv("SHEET_ID")  # 구글 시트 ID
SA_PATH     = os.getenv("SA_PATH", "sa.json")  # 서비스계정 키 경로
TARGET_YEAR = 2024
TARGET_MONTH = 10   # 예: 2024년 10월 예측

# ====== 지역 매핑 (필요 시 보강) ======
REGION_MAP = {
    "전국": ["전국"],
    "서울": ["서울", "서울특별시", "서울시"],
}
# 거래요약 탭의 지역 라벨 → 우리가 계산할 지역 키
SUMMARY_REGION_KEYS = ["전국", "서울"]

# ====== 유틸: 한국 공휴일 + 주말 제외한 '등록가능일' ======
KR_HOL = holidays.KR()

def is_posting_day(d: dt.date) -> bool:
    if d.weekday() >= 5:  # 토(5), 일(6) 제외
        return False
    if d in KR_HOL:
        return False
    return True

def posting_days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    days = []
    cur = start
    while cur <= end:
        if is_posting_day(cur):
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days

def posting_day_index(target_date: dt.date, month_year: tuple[int,int]) -> int:
    """해당 월의 시작일부터 target_date까지의 '등록가능일' 카운트(1-based).
       month_year=(YYYY,MM)"""
    y, m = month_year
    first = dt.date(y, m, 1)
    # '등록'은 다음 영업일에 반영되므로, 실제 등록일 지표는 그대로 사용 (자료가 '등록일 기준'이라는 가정)
    idx = 0
    cur = first
    while cur <= target_date:
        if is_posting_day(cur):
            idx += 1
        cur += dt.timedelta(days=1)
    return idx

# ====== 구글시트 연결 ======
def open_sheet(sheet_id: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    # ① 시크릿을 ENV로 직접 받기 (권장)
    json_str = os.getenv("GOOGLE_SA_JSON")
    if json_str:
        from google.oauth2.service_account import Credentials
        import json
        info = json.loads(json_str)  # 여기서 에러 나면 시크릿 내용이 잘못 붙여진 것
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    else:
        # ② (fallback) 여전히 파일에서 읽고 싶다면 SA_PATH 사용
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(SA_PATH, scopes=scopes)

    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)



# ====== 탭 파싱 ======
TAB_RE = re.compile(r"^(?P<region>전국|서울)\s+(?P<yy>\d{2})년\s+(?P<m>\d{1,2})월$")

def list_region_month_tabs(sh):
    """('전국', 2025, 9) -> worksheet 객체 매핑"""
    tabs = {}
    for ws in sh.worksheets():
        m = TAB_RE.match(ws.title.strip())
        if m:
            region = m.group("region")
            yy = int(m.group("yy"))
            year = 2000 + yy
            month = int(m.group("m"))
            tabs[(region, year, month)] = ws
    return tabs

# ====== 일간 데이터 읽기 (각 탭: 날짜별 등록 건수) ======
def read_daily(ws) -> pd.DataFrame:
    values = ws.get_all_values()
    df = pd.DataFrame(values)
    if df.empty:
        return pd.DataFrame(columns=["date","count","region"])
    # 열 헤더 추정: '등록일' '건수' 같은 패턴을 탐색
    header_row = 0
    cols = df.iloc[header_row].tolist()
    df = df.iloc[header_row+1:].reset_index(drop=True)
    df.columns = cols
    # 후보 컬럼명
    date_col = next((c for c in df.columns if "등록" in c or "날짜" in c or "Date" in c), None)
    cnt_col  = next((c for c in df.columns if "건수" in c or "수" in c or "Count" in c), None)
    if not date_col or not cnt_col:
        # 안전장치: 첫 두 열 가정
        date_col = df.columns[0]
        cnt_col  = df.columns[1]
    # 파싱
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.date,
        "count": pd.to_numeric(df[cnt_col].replace("", np.nan), errors="coerce")
    })
    out = out.dropna(subset=["date"])
    out["count"] = out["count"].fillna(0).astype(int)
    return out

# ====== 월별 누적완성도 곡선 산출 ======
def month_cutoff(year:int, month:int) -> dt.date:
    """해당 월의 공식 컷오프(다음 달 30일)"""
    if month == 12:
        return dt.date(year+1, 1, 30)
    return dt.date(year, month+1, 30)

def build_completion_curve(daily_df: pd.DataFrame, year:int, month:int) -> pd.Series:
    """
    과거 12개월 데이터로 d(등록가능일 index)별 누적/최종 중앙값 곡선을 만든다.
    daily_df: columns [date, count] (여러 월이 포함되어도 됨)
    반환: r(d) (index: d=1..Dmax, value in (0,1])
    """
    # 대상: 현재 (year,month) 기준으로 이전 12개월
    base = dt.date(year, month, 1)
    start = (base - pd.DateOffset(months=12)).date()
    end   = (base - pd.DateOffset(days=1)).date()

    # 월별 group
    daily_df["y"] = pd.to_datetime(daily_df["date"]).dt.year
    daily_df["m"] = pd.to_datetime(daily_df["date"]).dt.month

    curves = []
    for (y,m), g in daily_df.groupby(["y","m"]):
        first = dt.date(y,m,1)
        cut   = month_cutoff(y,m)
        if g["date"].min() < first or g["date"].max() > (cut + dt.timedelta(days=90)):
            # 범위 밖 데이터가 섞여도 무시
            pass
        # 컷오프까지 누적
        g2 = g[(g["date"]>=first) & (g["date"]<=cut)].sort_values("date")
        if g2.empty:
            continue
        final_total = int(g2["count"].sum())
        if final_total == 0:
            continue
        # d index별 누적
        g2["d"] = [posting_day_index(d, (y,m)) for d in g2["date"]]
        cum = g2.groupby("d")["count"].sum().cumsum()
        # d 결손 채우기
        Dmax = posting_day_index(cut, (y,m))
        s = pd.Series(index=range(1, Dmax+1), dtype=float)
        s.loc[cum.index] = cum.values
        s = s.fillna(method="ffill").fillna(0.0)
        r = s / final_total
        curves.append(r)

    if not curves:
        return pd.Series(dtype=float)

    # 다른 월 길이를 맞추기 위해 공통 d축(최대 Dmax)에 맞춤
    D = max(c.index.max() for c in curves)
    aligned = []
    for c in curves:
        c2 = c.copy()
        c2.index = c2.index.astype(int)
        c2 = c2.reindex(range(1, D+1)).fillna(method="ffill").fillna(1.0)
        aligned.append(c2)

    # 중앙값 곡선
    R = pd.concat(aligned, axis=1).median(axis=1).clip(upper=1.0)
    return R

def late_tail_factor(daily_df: pd.DataFrame) -> float:
    """컷오프(+30) 이후 +60 시점까지 추가 등록 평균 비율(중앙값).
       자료가 없다면 1.00으로."""
    ratios = []
    daily_df["y"] = pd.to_datetime(daily_df["date"]).dt.year
    daily_df["m"] = pd.to_datetime(daily_df["date"]).dt.month
    for (y,m), g in daily_df.groupby(["y","m"]):
        first = dt.date(y,m,1)
        cut   = month_cutoff(y,m)              # +30일 기준
        late  = cut + dt.timedelta(days=30)    # +60일 기준(대략)
        g_cut  = g[(g["date"]>=first) & (g["date"]<=cut)]
        g_late = g[(g["date"]>=first) & (g["date"]<=late)]
        a = int(g_cut["count"].sum())
        b = int(g_late["count"].sum())
        if a>0 and b>=a:
            ratios.append(b / a)
    if ratios:
        return float(np.median(ratios))
    return 1.0

# ====== 현재 월 예측 ======
def predict_month_final(daily_df: pd.DataFrame, year:int, month:int, today:dt.date) -> int:
    first = dt.date(year, month, 1)
    cut   = month_cutoff(year, month)

    # 오늘까지 누적
    cur = daily_df[(daily_df["date"]>=first) & (daily_df["date"]<=min(today, cut))]
    C_today = int(cur["count"].sum())

    # 완료도 곡선
    R = build_completion_curve(daily_df, year, month)
    if R.empty:
        # 데이터 부족: 보수적으로 추정 (현재 평균*남은 등록가능일)
        d_today = posting_day_index(min(today, cut), (year,month))
        avg = (cur["count"].tail(10).mean() if len(cur)>=3 else 0)
        remain_days = max(0, posting_day_index(cut,(year,month)) - d_today)
        base_pred = int(round(C_today + max(0, avg) * remain_days))
        return base_pred

    d_today = posting_day_index(min(today, cut), (year,month))
    d_today = min(d_today, R.index.max())
    r = float(R.loc[d_today])
    r = max(r, 0.1)  # 방어

    base = C_today / r
    tail = late_tail_factor(daily_df)
    return int(round(base * tail))

# ====== '거래요약' 쓰기 ======
def write_summary(ws_summary, month_col_label:str, region_key:str, value:int):
    """거래요약 시트에서 month_col_label(예:'25/10') 열, '예상건수' 행에 value 기록.
       region_key('전국'/'서울') 위치는 시트 구조에 맞게 보정."""
    # 전체 시트 읽기
    values = ws_summary.get_all_values()
    df = pd.DataFrame(values)
    # 헤더(열 라벨) 찾기
    header_row = 0
    col_names = df.iloc[header_row].tolist()
    # '25/10' 같은 열 찾기
    try:
        col_idx = col_names.index(month_col_label)
    except ValueError:
        raise RuntimeError(f"거래요약에 '{month_col_label}' 열을 찾을 수 없습니다.")
    # '예상건수' 행 찾기 (또는 지역별 블록 내 '예상건수')
    row_idx = None
    for i in range(1, len(df)):
        label = str(df.iloc[i, 0]).strip()
        if label == "예상건수" or label == f"{region_key}_예상건수":
            row_idx = i; break
    if row_idx is None:
        # 없으면 '예상건수' 행을 새로 만들거나 정책에 맞게 조정 필요
        raise RuntimeError("거래요약에서 '예상건수' 행을 찾을 수 없습니다.")

    # 기록
    rng = gspread.utils.rowcol_to_a1(row_idx+1, col_idx+1)
    ws_summary.update(rng, [[int(value)]])

def main():
    print(f"[INFO] target: {TARGET_YEAR}-{TARGET_MONTH:02d}")
    sh = open_sheet(SHEET_ID)

    # 탭 수집
    tab_map = list_region_month_tabs(sh)

    # 대상 월 일간 데이터 모으기(지역별)
    today = dt.date.today()
    month_col_label = f"{str(TARGET_YEAR)[2:]}/{TARGET_MONTH}"

    # 거래요약 시트
    ws_summary = sh.worksheet("거래요약")

    for region_key in SUMMARY_REGION_KEYS:
        # 과거 12개월 + 현재월까지의 일간 데이터 합치기
        frames = []
        for (region, y, m), ws in tab_map.items():
            if region not in REGION_MAP.get(region_key, []):
                continue
            df = read_daily(ws)
            if df.empty: 
                continue
            df["region"] = region_key
            frames.append(df)
        if not frames:
            print(f"[WARN] '{region_key}' 데이터 탭이 없습니다. 건너뜀.")
            continue

        daily = pd.concat(frames, ignore_index=True)
        est = predict_month_final(
            daily_df=daily,
            year=TARGET_YEAR,
            month=TARGET_MONTH,
            today=today
        )
        print(f"[EST] {region_key} {TARGET_YEAR}/{TARGET_MONTH}: {est:,}")

        # 기록
        try:
            write_summary(ws_summary, month_col_label, region_key, est)
        except Exception as e:
            print(f"[ERROR] 거래요약 기록 실패: {e}")

if __name__ == "__main__":
    main()
