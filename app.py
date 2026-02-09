# app.py
# Streamlit: AI 습관 트래커 (📊)
# 실행: streamlit run app.py
import os
import re
from datetime import date, timedelta

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
    st.caption("※ 키는 브라우저 세션에만 저장됩니다(session_state).")

# -----------------------------
# 유틸/세션 초기화
# -----------------------------
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Suwon", "Ulsan", "Jeju", "Changwon"
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
        # 최근 6일 샘플
        for i in range(6, 0, -1):
            d = (date.today() - timedelta(days=i)).isoformat()
            # 보기 좋게 패턴 생성 (데모)
            checks = {
                "기상 미션": i % 2 == 0,
                "물 마시기": True,
                "공부/독서": i % 3 != 0,
                "운동하기": i % 2 != 0,
                "수면": True if i % 4 != 0 else False,
            }
            mood = max(1, min(10, 5 + (3 - i % 7)))
            demo.append({
                "date": d,
                "habits": checks,
                "mood": mood,
                "rate": round(sum(checks.values()) / len(HABITS) * 100, 1),
            })
        st.session_state.history = demo

def _upsert_today(record: dict):
    """오늘 날짜 기준으로 history에 insert/update."""
    today = _today_str()
    found = False
    for i, r in enumerate(st.session_state.history):
        if r["date"] == today:
            st.session_state.history[i] = record
            found = True
            break
    if not found:
        st.session_state.history.append(record)
    # 날짜 오름차순 정렬
    st.session_state.history = sorted(st.session_state.history, key=lambda x: x["date"])

def _last_7_df():
    """history에서 마지막 7일 데이터프레임 생성(부족하면 있는 만큼)."""
    hist = sorted(st.session_state.history, key=lambda x: x["date"])
    hist = hist[-7:]
    rows = []
    for r in hist:
        rows.append({
            "date": r["date"][5:],  # MM-DD
            "달성률(%)": r.get("rate", 0),
            "기분(1~10)": r.get("mood", 0),
            "달성 습관 수": sum(bool(v) for v in r.get("habits", {}).values()),
        })
    return pd.DataFrame(rows)

# -----------------------------
# API 연동
# -----------------------------
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap에서 현재 날씨 가져오기 (한국어, 섭씨).
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        desc = (data.get("weather") or [{}])[0].get("description")
        icon = (data.get("weather") or [{}])[0].get("icon")
        main = data.get("main") or {}
        return {
            "city": city,
            "description": desc,
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "icon_url": f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None,
        }
    except Exception:
        return None

def _breed_from_dog_url(url: str):
    # Dog CEO 이미지 URL 패턴: .../breeds/<breed>/xxxx.jpg
    # <breed>가 "hound-afghan"처럼 하이픈 포함 가능
    try:
        m = re.search(r"/breeds/([^/]+)/", url)
        if not m:
            return None
        raw = m.group(1)
        # 표기 정리: hound-afghan -> Afghan Hound / bulldog-french -> French Bulldog
        parts = raw.split("-")
        parts = [p.capitalize() for p in parts if p]
        # 흔한 형태는 [종, 서브]가 아니라 [그룹, 서브]일 수 있어 뒤집어 보기 좋게
        if len(parts) >= 2:
            parts = parts[::-1]
        return " ".join(parts)
    except Exception:
        return None

def get_dog_image():
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
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
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    - 코치 스타일별 시스템 프롬프트
    - 출력 형식: 컨디션 등급(S~D), 습관 분석, 날씨 코멘트, 내일 미션, 오늘의 한마디
    - 모델: gpt-5-mini
    실패 시 None 반환
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
        # OpenAI SDK (신형/구형 호환 시도)
        # 1) 신형: from openai import OpenAI; client.responses.create(...)
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
                # SDK 반환 형태 대비
                text = getattr(resp, "output_text", None)
                if text:
                    return text
        except Exception:
            pass

        # 2) 구형: openai.ChatCompletion.create(...)
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
# 습관 체크인 UI
# -----------------------------
_ensure_history()

st.subheader("✅ 오늘의 체크인")

col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.markdown("**습관 체크(2열)**")
    c1, c2 = st.columns(2, gap="medium")

    # 기본값: 오늘 기록이 있으면 불러오기
    today_existing = next((r for r in st.session_state.history if r["date"] == _today_str()), None)
    existing_habits = (today_existing or {}).get("habits", {})

    habit_state = {}

    # 5개를 2열로 배치 (3/2)
    for idx, (emoji, name) in enumerate(HABITS):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            habit_state[name] = st.checkbox(f"{emoji} {name}", value=bool(existing_habits.get(name, False)))

    mood_default = int((today_existing or {}).get("mood", 6))
    mood = st.slider("🙂 오늘 기분은 어떤가요? (1~10)", min_value=1, max_value=10, value=mood_default)

with col_right:
    st.markdown("**환경 설정**")
    city_default = (today_existing or {}).get("city", "Seoul")
    city = st.selectbox("🌍 도시 선택", options=CITIES, index=CITIES.index(city_default) if city_default in CITIES else 0)

    coach_default = (today_existing or {}).get("coach_style", "따뜻한 멘토")
    coach_style = st.radio("🧑‍🏫 코치 스타일", options=list(COACH_STYLES.keys()),
                           index=list(COACH_STYLES.keys()).index(coach_default) if coach_default in COACH_STYLES else 1)

# -----------------------------
# 달성률 + 메트릭 + 차트
# -----------------------------
checked_count = sum(bool(v) for v in habit_state.values())
rate = round((checked_count / len(HABITS)) * 100, 1)

m1, m2, m3 = st.columns(3, gap="medium")
m1.metric("달성률", f"{rate} %")
m2.metric("달성 습관", f"{checked_count} / {len(HABITS)}")
m3.metric("기분", f"{mood} / 10")

# session_state에 오늘 기록 저장(항상 최신 유지)
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
# 결과 표시 (날씨 + 강아지 + AI 리포트)
# -----------------------------
st.divider()
st.subheader("🧠 AI 코치 리포트")

gen = st.button("컨디션 리포트 생성", type="primary")

weather_data = None
dog_data = None

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

    # 카드 2열: 날씨 / 강아지
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

        # 공유용 텍스트(간단 템플릿)
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
- 이 앱은 현재 날씨(`/data/2.5/weather`)를 **한국어(lang=kr)**, **섭씨(units=metric)**로 요청합니다.

**네트워크/설치 체크**
- `requests`, `pandas`, `streamlit` 필요
- OpenAI SDK는 환경에 따라 신형/구형 모두 시도합니다.
  - 신형 예: `pip install openai`
  - 구형 환경에서도 동작하도록 fallback 포함
"""
    )
