from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "agri_knowledge.json"
MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "model_labels.json"
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="A-R-Green | مساعد المزارع الذكي", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stTextInput, .stFileUploader { font-family: 'Cairo', sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg,#f4fbf4 0%,#edf7ef 55%,#e7f3e9 100%); direction:rtl; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d3b2e,#145c40); }
[data-testid="stSidebar"] * { color:#f4fff7 !important; }
.hero { background:linear-gradient(120deg,#0d3b2e,#238b5b); color:white; padding:2.2rem; border-radius:22px; box-shadow:0 10px 25px #0d3b2e26; margin-bottom:1.4rem; }
.hero h1 { font-size:2.4rem; margin:0; font-weight:800; }
.hero p { font-size:1.1rem; margin:.5rem 0 0; }
.card { background:#ffffffcc; border:1px solid #d8ebdc; border-radius:18px; padding:1.2rem; margin:.5rem 0; box-shadow:0 5px 16px #164e3520; }
.badge { display:inline-block; background:#d9f2df; color:#11643c; padding:.25rem .8rem; border-radius:20px; font-weight:700; }
.alert { border-right:5px solid #dc8b27; background:#fff7e8; padding:1rem; border-radius:12px; }
.small-note { color:#557064; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


KB = load_json(KB_PATH, {"crops": {}, "diseases": {}})


def configured_weather_key() -> str:
    try:
        return os.getenv("OPENWEATHER_API_KEY", "") or st.secrets.get("OPENWEATHER_API_KEY", "")
    except Exception:
        return os.getenv("OPENWEATHER_API_KEY", "")


def disease_card(disease_id: str, confidence: float | None = None) -> None:
    disease = KB.get("diseases", {}).get(disease_id, KB.get("diseases", {}).get("healthy", {}))
    title = disease.get("name", "نتيجة غير معروفة")
    st.markdown(f"<div class='card'><span class='badge'>التشخيص</span><h2>{title}</h2>", unsafe_allow_html=True)
    if confidence is not None:
        st.progress(float(max(0.0, min(confidence, 1.0))), text=f"ثقة النموذج: {confidence:.1%}")
    st.markdown(f"**الأعراض:** {disease.get('symptoms', '—')}  \n**العلاج/الإجراء المقترح:** {disease.get('treatment', '—')}  \n**نصائح الري:** {disease.get('irrigation', '—')}</div>", unsafe_allow_html=True)
    st.caption("تنبيه: التشخيص مساعد لاتخاذ القرار وليس بديلاً عن مهندس زراعي. تحقّق من ملصق المبيد والقوانين المحلية وفترة الأمان قبل الاستخدام.")


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        labels = load_json(LABELS_PATH, [])
        return model, labels
    except Exception as exc:
        return None, f"تعذر تحميل النموذج: {exc}"


def mock_diagnosis(image_bytes: bytes) -> str:
    keys = ["tomato_late_blight", "powdery_mildew", "nitrogen_deficiency", "whitefly"]
    return keys[int(hashlib.sha256(image_bytes).hexdigest(), 16) % len(keys)]


def page_home() -> None:
    st.markdown("<section class='hero'><h1>🌿 A-R-Green</h1><p>مساعد المزارع الذكي — قرارات زراعية أوضح، محصول أفضل.</p></section>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("المحاصيل المدعومة", "4")
    c2.metric("أدوات ذكية", "5")
    c3.metric("وضع التشخيص", "AI + تجريبي")
    st.markdown("### ابدأ من القائمة الجانبية")
    st.info("ارفع صورة في طبيب النباتات، أو اختر محصولًا لعرض تقويمه، أو أدخل حالة الطقس للحصول على توصية فورية.")
    st.markdown("<div class='card'><b>ملاحظة تشغيلية:</b> بعد تدريب النموذج ضع الملف <code>model.h5</code> وملف <code>model_labels.json</code> بجوار <code>app.py</code>. إذا لم يتوفرا، يعمل التطبيق بوضع Mock واضح للتجربة فقط.</div>", unsafe_allow_html=True)


def page_doctor() -> None:
    st.title("🩺 طبيب النباتات")
    st.write("ارفع صورة واضحة للورقة. سيستخدم التطبيق النموذج الحقيقي إن كان موجودًا، وإلا سيعرض نتيجة تجريبية ثابتة للصورة.")
    uploaded = st.file_uploader("رفع صورة الورقة", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", width=420)
    model, labels_or_error = load_model_bundle()
    if model is not None and isinstance(labels_or_error, list):
        arr = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32) / 255.0
        probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        label = labels_or_error[idx] if idx < len(labels_or_error) else "healthy"
        disease_card(label, float(probs[idx]))
    else:
        label = mock_diagnosis(uploaded.getvalue())
        st.warning("النموذج المدرب غير موجود أو تعذر تحميله؛ هذه نتيجة تجريبية وليست تشخيصًا حقيقيًا.")
        if isinstance(labels_or_error, str):
            st.caption(labels_or_error)
        disease_card(label)


def page_calendar() -> None:
    st.title("📅 التقويم الزراعي")
    crops = KB.get("crops", {})
    crop_names = {v.get("name", k): k for k, v in crops.items()}
    selected_name = st.selectbox("اختر المحصول", list(crop_names) or ["لا توجد بيانات"])
    crop = crops.get(crop_names.get(selected_name, ""), {})
    if not crop:
        st.error("لم يتم العثور على قاعدة المعرفة.")
        return
    a, b, c = st.columns(3)
    a.metric("ميعاد الزراعة", crop.get("planting", "—"))
    b.metric("ميعاد الحصاد", crop.get("harvest", "—"))
    c.metric("جدول الرش", crop.get("spraying", "—"))
    st.markdown(f"<div class='card'><b>جدول الري:</b> {crop.get('irrigation', '—')}<br><b>ملاحظات:</b> {crop.get('notes', '—')}</div>", unsafe_allow_html=True)
    st.subheader("الأمراض الشائعة")
    for disease_id in crop.get("diseases", []):
        disease = KB.get("diseases", {}).get(disease_id, {})
        with st.expander(disease.get("name", disease_id)):
            st.write(f"**المبيد/المادة الفعالة المقترحة:** {disease.get('pesticide', '—')}")
            st.write(disease.get("treatment", "—"))


def page_weather() -> None:
    st.title("🌦️ مساعد الري والرش الذكي")
    st.write("يمكنك اختيار الحالة يدويًا، أو جلب حالة الطقس الحالية من OpenWeatherMap عند توفر مفتاح API.")
    key = st.text_input("مفتاح OpenWeatherMap (اختياري)", type="password", value=configured_weather_key())
    city = st.text_input("المدينة (اختياري لجلب الطقس)", value="")
    condition = st.selectbox("حالة الطقس", ["مشمس", "ممطر", "غائم", "رياح قوية", "حرارة مرتفعة"])
    if st.button("جلب الطقس الحالي") and key and city:
        try:
            response = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"q": city, "appid": key, "units": "metric", "lang": "ar"}, timeout=10)
            response.raise_for_status()
            data = response.json()
            condition = data.get("weather", [{}])[0].get("description", condition)
            temp = data.get("main", {}).get("temp")
            st.success(f"الطقس في {city}: {condition} — الحرارة {temp}°م")
            if temp is not None and temp >= 35:
                condition = "حرارة مرتفعة"
            elif data.get("rain") or "مطر" in condition:
                condition = "ممطر"
        except requests.RequestException as exc:
            st.error(f"تعذر جلب الطقس: {exc}")
    advice = {
        "ممطر": ("⛔ أوقف الري والرش اليوم.", "الأمطار قد تغسل المبيد وتزيد رطوبة الأوراق."),
        "حرارة مرتفعة": ("💧 زد الري تدريجيًا وراقب إجهاد النبات.", "يفضل الري صباحًا أو مساءً وتجنب الرش وقت الظهيرة."),
        "رياح قوية": ("⛔ أجّل الرش.", "الرياح تسبب انجراف الرذاذ وتوزيعًا غير متجانس."),
        "غائم": ("✅ افحص رطوبة التربة قبل الري.", "راقب الأمراض الفطرية مع استمرار رطوبة الأوراق."),
        "مشمس": ("✅ الري حسب رطوبة التربة.", "تجنب رش بعض المنتجات في الشمس القوية؛ اتبع ملصق المنتج."),
    }
    title, detail = advice.get(condition, ("⚠️ راقب الحقل قبل القرار.", "لم يتم التعرف على الحالة بدقة."))
    st.markdown(f"<div class='alert'><h3>{title}</h3><p>{detail}</p></div>", unsafe_allow_html=True)


def page_academy() -> None:
    st.title("🎓 أكاديمية التدريب")
    lessons = {
        "كيفية زراعة الطماطم": "ابدأ بشتلات سليمة، جهّز تربة جيدة الصرف، واترك مسافات مناسبة. ثبّت النباتات مبكرًا، وراقب الري دون إغراق التربة.",
        "الأخطاء الشائعة": "الري الزائد، تبليل الأوراق مساءً، الزراعة في تربة سيئة الصرف، واستخدام مبيد دون قراءة الملصق أو احترام فترة الأمان.",
        "الفحص الأسبوعي": "افحص ظهر الأوراق والسيقان، صوّر الأعراض مبكرًا، وسجّل تاريخ الري والرش والظروف الجوية.",
    }
    for title, body in lessons.items():
        with st.expander(title, expanded=title == "كيفية زراعة الطماطم"):
            st.write(body)


with st.sidebar:
    st.markdown("# 🌿 A-R-Green")
    page = st.radio("القائمة الرئيسية", ["الرئيسية", "طبيب النباتات", "التقويم الزراعي", "مساعد الري والرش", "أكاديمية التدريب"])
    st.divider()
    st.caption("نسخة تجريبية قابلة للربط بنموذج CNN وOpenWeatherMap")

{"الرئيسية": page_home, "طبيب النباتات": page_doctor, "التقويم الزراعي": page_calendar, "مساعد الري والرش": page_weather, "أكاديمية التدريب": page_academy}[page]()
