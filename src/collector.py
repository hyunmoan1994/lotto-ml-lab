from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import pandas as pd
import requests

from .utils import lotto_columns


OFFICIAL_API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
MIRROR_RESULT_URL = "https://smok95.github.io/lotto/results/{draw_no}.json"
MIRROR_LATEST_URL = "https://smok95.github.io/lotto/results/latest.json"
MIRROR_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LottoMLLab/1.0)",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}


def _row_from_official(data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("returnValue") != "success":
        return None
    return {
        "draw_no": int(data["drwNo"]),
        "draw_date": str(data.get("drwNoDate", ""))[:10],
        "n1": int(data["drwtNo1"]),
        "n2": int(data["drwtNo2"]),
        "n3": int(data["drwtNo3"]),
        "n4": int(data["drwtNo4"]),
        "n5": int(data["drwtNo5"]),
        "n6": int(data["drwtNo6"]),
        "bonus": int(data["bnusNo"]),
    }


def _row_from_mirror(data: dict[str, Any]) -> dict[str, Any] | None:
    numbers = data.get("numbers") or []
    if len(numbers) != 6:
        return None
    return {
        "draw_no": int(data["draw_no"]),
        "draw_date": str(data.get("date", ""))[:10],
        "n1": int(numbers[0]),
        "n2": int(numbers[1]),
        "n3": int(numbers[2]),
        "n4": int(numbers[3]),
        "n5": int(numbers[4]),
        "n6": int(numbers[5]),
        "bonus": int(data["bonus_no"]),
    }


def _get_json(url: str, timeout: int = 8) -> dict[str, Any]:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type.lower() and not response.text.lstrip().startswith("{"):
        raise ValueError("JSON 응답이 아닙니다.")
    return response.json()


def _fetch_draw_official(draw_no: int) -> dict[str, Any] | None:
    return _row_from_official(_get_json(OFFICIAL_API_URL.format(draw_no=draw_no)))


def _fetch_draw_mirror(draw_no: int) -> dict[str, Any] | None:
    return _row_from_mirror(_get_json(MIRROR_RESULT_URL.format(draw_no=draw_no)))


def _fetch_draw(draw_no: int, source: Literal["official", "mirror"] = "official") -> dict[str, Any] | None:
    if source == "mirror":
        return _fetch_draw_mirror(draw_no)
    return _fetch_draw_official(draw_no)


def _find_latest_official(max_draws: int) -> int:
    misses = 0
    errors = 0
    for draw_no in range(max_draws, 0, -1):
        try:
            if _fetch_draw_official(draw_no) is not None:
                return draw_no
            misses += 1
        except Exception:
            errors += 1
        if errors >= 5 or misses >= 250:
            return 0
    return 0


def _find_latest_mirror() -> int:
    latest = _row_from_mirror(_get_json(MIRROR_LATEST_URL))
    if latest is None:
        return 0
    return int(latest["draw_no"])


def _fetch_all_mirror() -> pd.DataFrame:
    response = requests.get(MIRROR_ALL_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()
    data = response.json()
    rows = [_row_from_mirror(item) for item in data]
    return normalize_lotto_frame(pd.DataFrame([row for row in rows if row is not None]))


def collect_lotto_data(max_draws: int = 1300, min_draws: int = 300, recent_draws: int = 500) -> pd.DataFrame:
    """Collect Korean Lotto 6/45 draws without writing any local CSV files."""
    latest = _find_latest_official(max_draws)
    source: Literal["official", "mirror"] = "official"
    if latest == 0:
        all_draws = _fetch_all_mirror()
        all_draws = all_draws[all_draws["draw_no"] <= max_draws]
        draw_count = min(len(all_draws), max(min_draws, recent_draws))
        if draw_count < min_draws:
            raise RuntimeError(f"데이터 수집 실패: {draw_count}회차만 수집되었습니다. 네트워크 상태를 확인하세요.")
        return all_draws.tail(draw_count).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    if latest:
        draw_count = min(latest, max(min_draws, recent_draws))
        first_draw = max(1, latest - draw_count + 1)
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(_fetch_draw, draw_no, source): draw_no
                for draw_no in range(first_draw, latest + 1)
            }
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    row = None
                if row is not None:
                    rows.append(row)

    if len(rows) < min_draws:
        raise RuntimeError(f"데이터 수집 실패: {len(rows)}회차만 수집되었습니다. 네트워크 상태를 확인하세요.")

    return normalize_lotto_frame(pd.DataFrame(rows))


def load_or_collect() -> pd.DataFrame:
    return collect_lotto_data()


def normalize_lotto_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[lotto_columns()]
    for col in ["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna().astype(
        {"draw_no": int, "n1": int, "n2": int, "n3": int, "n4": int, "n5": int, "n6": int, "bonus": int}
    )
    df["draw_date"] = df["draw_date"].astype(str).str[:10]
    return df.sort_values("draw_no").drop_duplicates("draw_no").reset_index(drop=True)
