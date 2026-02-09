# app.py
# Streamlit: AI 습관 트래커 (📊)
# 실행: streamlit run app.py

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")

with st.sidebar:
    st.header("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="OWM key...")
    st.caption("※ 키는 브라우저 세션(session_state)에만 저장됩니다.")

# -----------------------------
# 상수/세션 초기화
# -----------------------------
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

# “영문 도시명만” 쓰면 OWM에서 동명이인 도시로 꼬이거나 못 찾는 경우가 있어
# 한국 도시용으로 KR 컨텍스트를 강하게 주도록, 내부적으로는 (표시명, geocode_query) 형태로 둡니다.
CITIES: list[tuple[str, str]] = [
    ("Seoul", "Seoul,KR"),
    ("Busan", "Busan,KR"),
    ("Incheon", "Incheon,KR"),
    ("Daegu", "Daegu,KR"),
    ("Daejeon", "Daejeon,KR"),
    ("Gwangju", "Gwangju,KR"),
    ("Suwon", "Suwon,KR"),
    ("Ulsan", "Ulsan,KR"),
    ("Jeju", "Jeju,KR"),
    ("Changwon", "Changwon,KR"),
]

COACH_STYLES = {
    "스파르타 코치": "너는 매우 엄격하고 직설적인 스파르타 코치다. 핑계는 허용하지 말고, 행동 지침을 강하게 요구하라. 다만 모욕은 금지.",
    "따뜻한 멘토": "너는 따뜻하고 공감적인 멘토다. 사용자의 노력을 인정하고, 작은 성공을 축하하며, 실행 가능한 다음 քայլ을 부드럽게 제안하라.",
    "게임 마스터": "너는 RPG 세계관의 게임 마스터다. 사용자를 모험가로 설정하고, 오늘의 상태를 스탯/퀘스트/보상처럼 묘사하며, 내일 미션을 퀘스트로 제시하라.",
}


def _today_str() -> str:
    return date.today().isoformat()


def _ensure_history():
    """데모 6일 + 오늘(7일) 기본 데이터 생성 (최초 1회)."""
    if "history" not in st.session_state:
        demo = []
        for i in range(6, 0, -1):
            d = (date.today() - timedelta(days=i)).isoformat()
            checks = {
                "기상 미션": i % 2 == 0,
                "물 마시기": True,
                "공부/독서": i % 3 != 0,
                "운동하기": i % 2 != 0,
                "수면": i % 4 != 0,
            }
            mood = max(1, min(10, 5 + (3 - i % 7)))
            demo.append(
                {
                    "date": d,
                    "habits": checks,
                    "mood": mood,
                    "rate": round(sum(checks.values()) / len(HABITS) * 100, 1),
                    "city": "Seoul",
                    "coach_style": "따뜻한 멘토",
                }
            )
        st.session_state.history = demo

    if "weather_debug" not in st.session_state:
        st.session_state.weather_debug = None


def _upsert_today(record: dict):
    """오늘 날짜 기준으로 history에 insert/update."""
    today = _today_str()
    replaced = False
    for i, r in enumerate(st.session_state.history):
        if r["date"] == today:
            st.session_state.history[i] = record
            replaced = True
            break
    if not replaced:
        st.session_state.history.append(record)

    st.session_state.history = sorted(st.session_state.history, key=lambda x: x["date"])


def _last_7_df() -> pd.DataFrame:
    hist = sorted(st.session_state.history, key=lambda x: x["date"])[-7:]
    rows = []
    for r in hist:
        rows.append(
            {
                "date": r["date"][5:],  # MM-DD
                "달성률(%)": r.get("rate", 0),
                "기분(1~10)": r.get("mood", 0),
                "달성 습관 수": sum(bool(v) for v in r.get("habits", {}).values()),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# API 연동
# -----------------------------
def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}


def _geocode_city(query: str, api_key: str) -> Optional[Tuple[float, float, str]]:
    """
    OpenWeatherMap 지오코딩으로 도시 -> (lat, lon, resolved_name)
    실패 시 None
    """
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": query, "limit": 1, "appid": api_key}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Geocode HTTP {r.status_code}: {_safe_json(r)}")

    arr = _safe_json(r)
    if not isinstance(arr, list) or not arr:
        return None

    item = arr[0]
    lat = item.get("lat")
    lon = item.get("lon")
    name = item.get("name") or query
    country = item.get("country")
    state = item.get("state")
    resolved = f"{name}" + (f", {state}" if state else "") + (f", {country}" if country else "")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon), resolved


def get_weather(city_display: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap에서 날씨 가져오기 (한국어, 섭씨)
    - 지오코딩(도시->lat/lon) 후 weather 호출로 안정성 개선
    - 실패 시 None 반환
    - timeout=10
    """
    st.session_state.weather_debug = None

    if not api_key:
        st.session_state.weather_debug = {"reason": "missing_api_key"}
        return None

    # display -> query("Seoul,KR") 매핑
    query = next((q for (disp, q) in CITIES if disp == city_display), f"{city_display},KR")

    try:
        geo = _geocode_city(query, api_key)
        if not geo:
            st.session_state.weather_debug = {"reason": "geocode_not_found", "query": query}
            return None

        lat, lon, resolved_name = geo

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            st.session_state.weather_debug = {
                "reason": "weather_http_error",
                "status_code": r.status_code,
                "body": _safe_json(r),
                "query": query,
                "lat": lat,
                "lon": lon,
            }
            return None

        data = _safe_json(r)
        weather0 = (data.get("weather") or [{}])[0]
        main = data.get("main") or {}
        desc = weather0.get("description")
        icon = weather0.get("icon")

        return {
            "city": resolved_name,
            "description": desc,
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "icon_url": f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None,
        }
    except requests.Timeout:
        st.session_state.weather_debug = {"reason": "timeout", "query": query}
        return None
    except Exception as e:
        st.session_state.weather_debug = {"reason": "exception", "query": query, "error": str(e)}
        return None


def _breed_from_dog_url(url: str) -> Optional[str]:
    try:
        m = re.search(r"/breeds/([^/]+)/", url)
        if not m:
            return None
        raw = m.group(1)
        parts = [p.capitalize() for p in raw.split("-") if p]
        if len(parts) >= 2:
            parts = parts[::-1]
        return " ".join(parts)
    except Exception:
        return None


def get_dog_image() -> Optional[Dict[str, str]]:
    """Dog CEO 랜덤 강아지 이미지 URL + 품종(추정). 실패 시 None, timeout=10"""
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = _safe_json(r)
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
        if not img_url:
            return None
        breed = _breed_from_dog_url(img_url) or "Unknown"
        return {"image_url": img_url, "breed": breed}
    except Exception:
        return None


def generate_report(
    *,
    openai_key: str,
    coach_style: str,
    habits: dict,
    mood: int,
    weather: Optional[dict],
    dog: Optional[dict],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    - 코치 스타일별 시스템 프롬프트
    - 출력 형식 고정
    - 모델: gpt-5-mini
    실패 시 None
    """
    if not openai_key:
        return None

    sys_prompt = COACH_STYLES.get(coach_style, COACH_STYLES["따뜻한 멘토"])

    habit_lines = []
    for _, name in HABITS:
        habit_lines.append(f"- {name}: {'완료' if habits.get(name) else '미완료'}")
    habit_text = "\n".join(habit_lines)

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('description')}, "
            f"{weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C), "
            f"습도 {weather.get('humidity')}%"
        )

    dog_text = "강아지 정보 없음"
    if dog:
        dog_text = f"품종: {dog.get('breed')}"

    user_prompt = f"""
[오늘 체크인]
날짜: {_today_str()}
기분(1~10): {mood}

[습관]
{habit_text}

[날씨]
{weather_text}

[오늘의 강아지]
{dog_text}

아래 형식을 정확히 지켜 한국어로 작성해줘.

형식:
1) 컨디션 등급: (S/A/B/C/D 중 하나) - 한 줄 이유
2) 습관 분석: (잘한 점 2개 + 개선점 2개, 각 불릿)
3) 날씨 코멘트: (날씨가 없으면 대체 코멘트)
4) 내일 미션: (3개, 구체적이고 체크 가능하게)
5) 오늘의 한마디: (짧고 임팩트 있게)
""".strip()

    try:
        # 신형 SDK 우선 시도
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=openai_key)
            if hasattr(client, "responses"):
                resp = client.responses.create(
                    model="gpt-5-mini",
                    input=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = getattr(resp, "output_text", None)
                if text:
                    return text
        except Exception:
            pass

        # 구형 SDK fallback
        import openai  # type: ignore

        openai.api_key = openai_key
        if hasattr(openai, "ChatCompletion"):
            cc = openai.ChatCompletion.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return cc["choices"][0]["message"]["content"]
    except Exception:
        return None

    return None


# -----------------------------
# UI: 체크인
# -----------------------------
_ensure_history()

st.subheader("✅ 오늘의 체크인")

# 오늘 기존 기록(있으면 불러오기)
today_existing = next((r for r in st.session_state.history if r["date"] == _today_str()), None)
existing_habits = (today_existing or {}).get("habits", {})

left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.markdown("**습관 체크(2열)**")
    c1, c2 = st.columns(2, gap="medium")
    habit_state: dict[str, bool] = {}

    for idx, (emoji, name) in enumerate(HABITS):
        target = c1 if idx % 2 == 0 else c2
        with target:
            habit_state[name] = st.checkbox(
                f"{emoji} {name}",
                value=bool(existing_habits.get(name, False)),
            )

    mood_default = int((today_existing or {}).get("mood", 6))
    mood = st.slider("🙂 오늘 기분은 어떤가요? (1~10)", 1, 10, mood_default)

with right:
    st.markdown("**환경 설정**")
    city_default = (today_existing or {}).get("city", "Seoul")
    city_options = [d for (d, _) in CITIES]
    city = st.selectbox(
        "🌍 도시 선택",
        options=city_options,
        index=city_options.index(city_default) if city_default in city_options else 0,
    )

    coach_default = (today_existing or {}).get("coach_style", "따뜻한 멘토")
    coach_style = st.radio(
        "🧑‍🏫 코치 스타일",
        options=list(COACH_STYLES.keys()),
        index=list(COACH_STYLES.keys()).index(coach_default) if coach_default in COACH_STYLES else 1,
    )

# -----------------------------
# 달성률 + 메트릭 + 차트 + 저장
# -----------------------------
checked_count = sum(bool(v) for v in habit_state.values())
rate = round((checked_count / len(HABITS)) * 100, 1)

m1, m2, m3 = st.columns(3, gap="medium")
m1.metric("달성률", f"{rate} %")
m2.metric("달성 습관", f"{checked_count} / {len(HABITS)}")
m3.metric("기분", f"{mood} / 10")

# 오늘 기록 저장
today_record = {
    "date": _today_str(),
    "habits": habit_state,
    "mood": mood,
    "rate": rate,
    "city": city,
    "coach_style": coach_style,
}
_upsert_today(today_record)

st.divider()
st.subheader("📈 최근 7일 추세")
df7 = _last_7_df()
st.bar_chart(df7.set_index("date")["달성률(%)"])

# -----------------------------
# 결과 표시: 날씨+강아지 카드 + AI 리포트 + 공유 텍스트
# -----------------------------
st.divider()
st.subheader("🧠 AI 코치 리포트")

gen = st.button("컨디션 리포트 생성", type="primary")

if gen:
    with st.spinner("날씨와 강아지를 불러오고, AI 리포트를 생성 중..."):
        weather_data = get_weather(city, weather_api_key)
        dog_data = get_dog_image()

        report = generate_report(
            openai_key=openai_api_key,
            coach_style=coach_style,
            habits=habit_state,
            mood=mood,
            weather=weather_data,
            dog=dog_data,
        )

    wcol, dcol = st.columns(2, gap="large")

    with wcol:
        st.markdown("#### 🌦️ 오늘의 날씨")
        if weather_data:
            if weather_data.get("icon_url"):
                st.image(weather_data["icon_url"], width=80)
            st.write(f"**도시:** {weather_data.get('city')}")
            st.write(f"**상태:** {weather_data.get('description')}")
            st.write(f"**기온:** {weather_data.get('temp_c')}°C (체감 {weather_data.get('feels_like_c')}°C)")
            st.write(f"**습도:** {weather_data.get('humidity')}%")
        else:
            st.warning("날씨 정보를 불러오지 못했어요. (API Key/도시/네트워크를 확인해 주세요)")
            with st.expander("날씨 오류 상세(디버그)", expanded=False):
                st.json(st.session_state.weather_debug or {"debug": "no_debug_info"})

    with dcol:
        st.markdown("#### 🐶 오늘의 강아지")
        if dog_data and dog_data.get("image_url"):
            st.image(dog_data["image_url"], use_container_width=True)
            st.caption(f"품종(추정): {dog_data.get('breed', 'Unknown')}")
        else:
            st.warning("강아지 이미지를 불러오지 못했어요.")

    st.markdown("#### 📝 리포트")
    if report:
        st.markdown(report)

        share_lines = [
            f"📊 AI 습관 트래커 | {_today_str()}",
            f"도시: {city} | 코치: {coach_style}",
            f"달성률: {rate}% ({checked_count}/{len(HABITS)}) | 기분: {mood}/10",
            "",
            "✅ 오늘의 습관",
        ]
        for emoji, name in HABITS:
            share_lines.append(f"- {emoji} {name}: {'완료' if habit_state.get(name) else '미완료'}")

        if weather_data:
            share_lines += [
                "",
                "🌦️ 날씨",
                f"- {weather_data.get('description')} / {weather_data.get('temp_c')}°C (체감 {weather_data.get('feels_like_c')}°C)",
            ]
        if dog_data:
            share_lines += ["", "🐶 오늘의 강아지", f"- {dog_data.get('breed', 'Unknown')}"]

        share_lines += ["", "🧠 AI 리포트", report.strip()]

        st.markdown("#### 🔗 공유용 텍스트")
        st.code("\n".join(share_lines), language="text")
    else:
        st.error("AI 리포트를 생성하지 못했어요. (OpenAI API Key/네트워크/SDK 설치 여부를 확인해 주세요)")

# -----------------------------
# 하단: API 안내
# -----------------------------
with st.expander("ℹ️ API 안내 / 설정 방법", expanded=False):
    st.markdown(
        """
**OpenAI API Key**
- OpenAI에서 발급받은 키를 사이드바에 붙여넣으세요.
- 모델은 `gpt-5-mini`로 호출합니다.

**OpenWeatherMap API Key**
- OpenWeatherMap에서 API 키를 발급받고 사이드바에 입력하세요.
- 이 앱은 도시명을 바로 쓰지 않고, 먼저 지오코딩(`/geo/1.0/direct`)으로 **위경도(lat/lon)** 를 얻은 뒤
  현재 날씨(`/data/2.5/weather`)를 **한국어(lang=kr)**, **섭씨(units=metric)** 로 요청합니다.
- 그래도 안 되면 expander의 **날씨 오류 상세(디버그)** 에서 HTTP 코드/메시지를 확인하세요.

**필수 패키지**
- `streamlit`, `requests`, `pandas`
- OpenAI SDK는 환경에 따라 신형/구형 모두 시도(fallback)합니다.
"""
    )
