# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh   # auto-refresh helper

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="Krishi Sewa",
    page_icon="🌱",
    layout="wide"
)

# ----------------------
# Session State Initialization
# ----------------------
# These hold the live sensor values in memory
if 'temp_val' not in st.session_state: st.session_state['temp_val'] = 25.0
if 'hum_val' not in st.session_state:  st.session_state['hum_val'] = 70.0
if 'ph_val'  not in st.session_state: st.session_state['ph_val']  = 6.5

# ----------------------
# Data & Model Loading
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'dataset.csv' is in the folder.")
        st.stop()
        
    df.dropna(subset=['soil_texture', 'label'], inplace=True)
    
    soil_encoder = LabelEncoder()
    df['soil_texture_encoded'] = soil_encoder.fit_transform(df['soil_texture'])
    
    crop_encoder = LabelEncoder()
    df['crop_encoded'] = crop_encoder.fit_transform(df['label'])
    
    return df, soil_encoder, crop_encoder

df, soil_encoder, crop_encoder = load_data()

@st.cache_resource
def train_model():
    X = df[['temperature', 'humidity', 'ph', 'soil_texture_encoded']]
    y = df['crop_encoded']
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

rf_model = train_model()

@st.cache_data
def get_crop_profiles():
    # Mean env + NPK per crop
    profiles = df.groupby("label")[['temperature', 'humidity', 'ph', 'N', 'P', 'K']].mean()
    # Most common soil texture
    profiles['soil_texture'] = df.groupby("label")['soil_texture'].agg(lambda x: x.mode()[0])
    return profiles

crop_profiles = get_crop_profiles()

# Min/max env ranges from dataset (for caption/context)
@st.cache_data
def get_env_ranges():
    return df.groupby("label")[['temperature', 'humidity', 'ph']].agg(['min', 'max'])

env_ranges = get_env_ranges()

# Top 3 soil textures per crop (best + alternatives)
@st.cache_data
def get_soil_alternatives():
    return df.groupby('label')['soil_texture'].agg(
        lambda x: x.value_counts().head(3).index.tolist()
    )

soil_alternatives = get_soil_alternatives()

# ----------------------
# HELPER: SMART FERTILIZER SCHEDULE
# ----------------------
def get_fertilizer_schedule(crop_name, n_needed, p_needed, k_needed):
    """
    Returns a custom schedule based on crop type / life-cycle.
    Uses simple heuristic groupings for:
    - Trees/orchards
    - Short-duration pulses/melons
    - Long-duration crops
    - Vegetables (fruiting, leafy, root/tuber)
    - Default cereals / others
    """
    crop_name = crop_name.lower()

    # 1. TREES / ORCHARDS (Seasonal Logic)
    trees = [
        'apple', 'banana', 'coconut', 'coffee', 'grapes',
        'mango', 'orange', 'papaya', 'pomegranate'
    ]
    if crop_name in trees:
        return pd.DataFrame({
            "Stage": ["Spring (Pre-Bloom)", "Post-Harvest (Recovery)"],
            "Nitrogen (N)": [f"{n_needed*0.7:.1f} kg", f"{n_needed*0.3:.1f} kg"],
            "Phosphorus (P)": [f"{p_needed:.1f} kg", "0 kg"],
            "Potassium (K)": [f"{k_needed*0.5:.1f} kg", f"{k_needed*0.5:.1f} kg"]
        })

    # 2. SHORT DURATION PULSES / MELONS (< ~70 days)
    short_crops = [
        'mungbean', 'blackgram', 'lentil', 'mothbeans',
        'watermelon', 'muskmelon', 'kidneybeans', 'chickpea'
    ]
    if crop_name in short_crops:
        return pd.DataFrame({
            "Stage": ["Basal (At Sowing)", "Flowering (Day 25)"],
            "Nitrogen (N)": [f"{n_needed*0.5:.1f} kg", f"{n_needed*0.5:.1f} kg"],
            "Phosphorus (P)": [f"{p_needed:.1f} kg", "0 kg"],
            "Potassium (K)": [f"{k_needed:.1f} kg", "0 kg"]
        })

    # 3. LONG DURATION CROPS (> ~150 days)
    long_crops = ['sugarcane', 'pigeonpeas', 'cotton', 'tea']
    if crop_name in long_crops:
        return pd.DataFrame({
            "Stage": ["Basal", "Tillering / Mid-Season", "Late Growth"],
            "Nitrogen (N)": [
                f"{n_needed*0.33:.1f} kg",
                f"{n_needed*0.33:.1f} kg",
                f"{n_needed*0.33:.1f} kg"
            ],
            "Phosphorus (P)": [f"{p_needed:.1f} kg", "0 kg", "0 kg"],
            "Potassium (K)": [
                f"{k_needed*0.5:.1f} kg",
                f"{k_needed*0.5:.1f} kg",
                "0 kg"
            ]
        })

    # 4. VEGETABLES – FRUITING (tomato, chili, cucurbits)
    veg_fruiting = ['tomato', 'chili', 'cucumber', 'pumpkin']
    if crop_name in veg_fruiting:
        return pd.DataFrame({
            "Stage": ["Basal (Transplanting)", "Vegetative (Day 20-25)", "Flowering/Fruiting"],
            "Nitrogen (N)": [
                f"{n_needed*0.4:.1f} kg",
                f"{n_needed*0.3:.1f} kg",
                f"{n_needed*0.3:.1f} kg"
            ],
            "Phosphorus (P)": [f"{p_needed*0.7:.1f} kg", f"{p_needed*0.3:.1f} kg", "0 kg"],
            "Potassium (K)": [
                f"{k_needed*0.3:.1f} kg",
                f"{k_needed*0.3:.1f} kg",
                f"{k_needed*0.4:.1f} kg"
            ]
        })

    # 5. VEGETABLES – LEAFY / COLE (cabbage, cauliflower)
    veg_leafy = ['cabbage', 'cauliflower']
    if crop_name in veg_leafy:
        return pd.DataFrame({
            "Stage": ["Basal (Transplanting)", "Head Formation (Day 25-30)"],
            "Nitrogen (N)": [f"{n_needed*0.6:.1f} kg", f"{n_needed*0.4:.1f} kg"],
            "Phosphorus (P)": [f"{p_needed:.1f} kg", "0 kg"],
            "Potassium (K)": [f"{k_needed*0.7:.1f} kg", f"{k_needed*0.3:.1f} kg"]
        })

    # 6. VEGETABLES – ROOT / TUBER / BULB (potato, onion, garlic, carrot, radish, ginger)
    veg_root = ['potato', 'onion', 'garlic', 'carrot', 'radish', 'ginger']
    if crop_name in veg_root:
        return pd.DataFrame({
            "Stage": ["Basal (Planting)", "Bulking / Root Development"],
            "Nitrogen (N)": [f"{n_needed*0.5:.1f} kg", f"{n_needed*0.5:.1f} kg"],
            "Phosphorus (P)": [f"{p_needed:.1f} kg", "0 kg"],
            "Potassium (K)": [f"{k_needed*0.5:.1f} kg", f"{k_needed*0.5:.1f} kg"]
        })

    # 7. DEFAULT CEREALS / OTHER (rice, maize, wheat, barley, millet, jute, mustard, etc.)
    return pd.DataFrame({
        "Stage": ["Basal (At Sowing)", "Tillering (Day 25-30)", "Panicle/Flower Init (Day 55-60)"],
        "Nitrogen (N)": [
            f"{n_needed*0.5:.1f} kg",
            f"{n_needed*0.25:.1f} kg",
            f"{n_needed*0.25:.1f} kg"
        ],
        "Phosphorus (P)": [f"{p_needed:.1f} kg", "0 kg", "0 kg"],
        "Potassium (K)": [f"{k_needed:.1f} kg", "0 kg", "0 kg"]
    })

# ----------------------
# SIDEBAR: IOT CONNECTION
# ----------------------
st.sidebar.header("📡 IoT Connection")
st.sidebar.markdown("Sync with ESP32 sensors.")

if st.sidebar.button("🔄 Sync Live Data", type="primary"):
    try:
        with open("sensor_data.json", "r") as f:
            data = json.load(f)
        st.session_state['temp_val'] = float(data.get('temperature', 25.0))
        st.session_state['hum_val']  = float(data.get('humidity', 70.0))
        st.session_state['ph_val']   = float(data.get('ph', 6.5))
        st.sidebar.success("Synced!")
    except Exception:
        st.sidebar.error("ESP32 Not Connected")

st.sidebar.markdown("---")
st.sidebar.metric("🌡️ Temp", f"{st.session_state['temp_val']:.2f} °C")
st.sidebar.metric("💧 Humidity", f"{st.session_state['hum_val']:.2f} %")
st.sidebar.metric("🧪 Soil pH", f"{st.session_state['ph_val']:.2f}")

# ===========================
# 🔥 ADDED: SIDEBAR MOISTURE
# ===========================
try:
    with open("sensor_data.json", "r") as f:
        side_data = json.load(f)
    sidebar_moisture = float(side_data.get("moisture", 0))
    sidebar_pump = side_data.get("pump", False)
except:
    sidebar_moisture = 0
    sidebar_pump = False

st.sidebar.metric("🌱 Moisture", f"{sidebar_moisture:.2f} %")
st.sidebar.metric("⚙️ Pump", "ON" if sidebar_pump else "OFF")


# ----------------------
# MAIN DASHBOARD
# ----------------------
st.title("🌱 Krishi Sewa")
tab1, tab2, tab3 = st.tabs(["🧠 Predict", "🛡️ Monitor", "📖 Guide & Plan"])

# ==========================================
# TAB 1: PREDICTION (Hybrid Mode)
# ==========================================
with tab1:
    st.header("Crop Recommendation")
    
    input_mode = st.radio(
        "Select Data Source:",
        ["📡 Live IoT Sensors", "✍️ Manual Entry (Simulation)"],
        horizontal=True
    )
    st.write("---")

    col1, col2, col3 = st.columns(3)

    if input_mode == "📡 Live IoT Sensors":
        st.info("Using live data from ESP32 (last synced values).")
        t = col1.number_input("Temperature (°C)", value=st.session_state['temp_val'], disabled=True)
        h = col2.number_input("Humidity (%)",    value=st.session_state['hum_val'],  disabled=True)
        p = col3.number_input("Soil pH",         value=st.session_state['ph_val'],   disabled=True)
    else:
        st.warning("Simulation Mode Enabled.")
        t = col1.number_input("Temperature (°C)", 0.0, 60.0,  st.session_state['temp_val'])
        h = col2.number_input("Humidity (%)",    0.0, 100.0, st.session_state['hum_val'])
        p = col3.number_input("Soil pH",         0.0, 14.0,  st.session_state['ph_val'])

    st.write("") 
    soil_options = sorted(df['soil_texture'].dropna().unique().tolist())
    soil_type = st.selectbox("Select Soil Texture", soil_options)

    if st.button("🧠 Get Recommendation", type="primary"):
        soil_enc = soil_encoder.transform([soil_type])[0]
        feats = np.array([[t, h, p, soil_enc]])
        
        probs = rf_model.predict_proba(feats)[0]
        top_indices = np.argsort(probs)[-3:][::-1]
        top_crops = crop_encoder.inverse_transform(top_indices)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        # c1.success(f"🥇 1st Choice: **{top_crops[0]}**")
        # c2.info(   f"🥈 2nd Choice: **{top_crops[1]}**")
        # c3.info(   f"🥉 3rd Choice: **{top_crops[2]}**")
        c1.success(f"🥇 1st Choice: **{top_crops[0]}** ({probs[top_indices[0]] * 100:.1f}% confidence)")
        c2.info(f"🥈 2nd Choice: **{top_crops[1]}** ({probs[top_indices[1]] * 100:.1f}% confidence)")
        c3.info(f"🥉 3rd Choice: **{top_crops[2]}** ({probs[top_indices[2]] * 100:.1f}% confidence)")

        # --- NPK SUMMARY FOR RECOMMENDED CROPS ---
        st.subheader("NPK Summary for Recommended Crops")
        st.caption("Average seasonal N, P, K per hectare (from dataset). For detailed soil-test-based adjustment, see Phase 3.")

        for crop in top_crops:
            if crop in crop_profiles.index:
                req = crop_profiles.loc[crop]
                with st.expander(f"{crop} – View NPK (kg/ha)"):
                    n_col, p_col, k_col = st.columns(3)
                    n_col.metric("Nitrogen (N)",   f"{req['N']:.0f}")
                    p_col.metric("Phosphorus (P)", f"{req['P']:.0f}")
                    k_col.metric("Potassium (K)",  f"{req['K']:.0f}")
            else:
                st.warning(f"No NPK data available for {crop}.")
# ==========================================
# TAB 2: MONITORING (Live Alerts)
# ==========================================
with tab2:
    st.header("Active Field Monitoring")
    st.markdown("Select the crop currently growing to check if conditions are safe.")
    
    auto_refresh = st.checkbox("Auto-refresh sensor data", value=False)
    if auto_refresh:
        st_autorefresh(interval=10000, limit=None, key="monitor_autorefresh")

    active_crop = st.selectbox("Current Crop in Field:", crop_profiles.index.sort_values())
    ideal = crop_profiles.loc[active_crop]

    # Start from last synced snapshot
    curr_temp = st.session_state['temp_val']
    curr_hum  = st.session_state['hum_val']
    curr_ph   = st.session_state['ph_val']

    # Load live sensor data
    try:
        with open("sensor_data.json", "r") as f:
            data = json.load(f)
        curr_temp = float(data.get('temperature', curr_temp))
        curr_hum  = float(data.get('humidity', curr_hum))
        curr_ph   = float(data.get('ph', curr_ph))

        # ===========================
        # 🔥 ADDED: MOISTURE + PUMP
        # ===========================
        moisture = float(data.get("moisture", 0))
        pump = data.get("pump", False)

    except:
        moisture = 0
        pump = False

    # Show current live readings
    st.subheader("Current Live Sensor Readings")
    c_live1, c_live2, c_live3, c_live4, c_live5 = st.columns(5)

    c_live1.metric("Temperature (°C)", f"{curr_temp:.1f}")
    c_live2.metric("Humidity (%)",    f"{curr_hum:.1f}")
    c_live3.metric("Soil pH",         f"{curr_ph:.1f}")

    # ===========================
    # 🔥 ADDED DISPLAY
    # ===========================
    c_live4.metric("🌱 Moisture (%)", f"{moisture:.1f}")

    st.markdown("---")

    # Base tolerances
    TOLERANCE_TEMP = 5.0
    TOLERANCE_HUM  = 15.0
    TOLERANCE_PH   = 0.8

    t_center = ideal['temperature']
    h_center = ideal['humidity']
    p_center = ideal['ph']

    low_t  = t_center - TOLERANCE_TEMP
    high_t = t_center + TOLERANCE_TEMP

    low_h  = max(0.0, h_center - TOLERANCE_HUM)
    high_h = min(100.0, h_center + TOLERANCE_HUM)

    low_p  = max(0.0, p_center - TOLERANCE_PH)
    high_p = min(14.0, p_center + TOLERANCE_PH)

    col_m1, col_m2, col_m3 = st.columns(3)

    # Temp Logic
    with col_m1:
        if low_t <= curr_temp <= high_t:
            st.success(
                f"🌡️ Temperature OK\n"
                f"Comfort range: {low_t:.1f}–{high_t:.1f}°C\n"
                f"Current: {curr_temp:.1f}°C"
            )
        elif curr_temp > high_t:
            st.error(
                f"🔥 HEAT STRESS\n"
                f"Comfort range: {low_t:.1f}–{high_t:.1f}°C\n"
                f"Current: {curr_temp:.1f}°C"
            )
        else:
            st.warning(
                f"❄️ COLD STRESS\n"
                f"Comfort range: {low_t:.1f}–{high_t:.1f}°C\n"
                f"Current: {curr_temp:.1f}°C"
            )

    # Humidity Logic
    with col_m2:
        if low_h <= curr_hum <= high_h:
            st.success(
                f"💧 Humidity OK\n"
                f"Comfort range: {low_h:.1f}–{high_h:.1f}%\n"
                f"Current: {curr_hum:.1f}%"
            )
        elif curr_hum > high_h:
            st.warning(
                f"🍄 FUNGUS RISK (Too Humid)\n"
                f"Comfort range: {low_h:.1f}–{high_h:.1f}%\n"
                f"Current: {curr_hum:.1f}%"
            )
        else:
            st.error(
                f"🌵 DROUGHT RISK (Too Dry)\n"
                f"Comfort range: {low_h:.1f}–{high_h:.1f}%\n"
                f"Current: {curr_hum:.1f}%"
            )

    # pH Logic
    with col_m3:
        if low_p <= curr_ph <= high_p:
            st.success(
                f"🧪 pH Optimal\n"
                f"Comfort range: {low_p:.1f}–{high_p:.1f}\n"
                f"Current: {curr_ph:.1f}"
            )
        else:
            st.warning(
                f"⚠️ pH Imbalance\n"
                f"Comfort range: {low_p:.1f}–{high_p:.1f}\n"
                f"Current: {curr_ph:.1f}"
            )

    # ===========================
    # 🔥 ADDED: MOISTURE WARNING + PUMP LOGIC
    # ===========================
    st.markdown("---")
    st.subheader("🌱 Irrigation Status")

    if moisture < 30:
        st.error("🌵 DRY SOIL → Pump SHOULD BE ON")
    elif moisture >= 40:
        st.success("💧 GOOD SOIL → Pump SHOULD BE OFF")
    else:
        st.info("🌱 STABLE ZONE")

    if pump:
        st.warning("⚙️ PUMP IS RUNNING")
    else:
        st.info("⚙️ PUMP IS OFF")

# ==========================================
# TAB 3: GUIDE & SCHEDULE (The Expert System)
# ==========================================
with tab3:
    st.header("Crop Guide & Fertilizer Plan")
    
    # 1. Select Crop
    target = st.selectbox("Crop Planning:", crop_profiles.index.sort_values(), key="sched_crop")
    req = crop_profiles.loc[target]

    # Observed min/max from dataset
    ranges = env_ranges.loc[target]
    t_min, t_max = ranges[('temperature', 'min')], ranges[('temperature', 'max')]
    h_min, h_max = ranges[('humidity',   'min')], ranges[('humidity',   'max')]
    p_min, p_max = ranges[('ph',         'min')], ranges[('ph',         'max')]

    # Same tolerances as Phase 2
    TOLERANCE_TEMP = 5.0
    TOLERANCE_HUM  = 15.0
    TOLERANCE_PH   = 0.8

    t_center = req['temperature']
    h_center = req['humidity']
    p_center = req['ph']

    # Comfort bands: mean ± tolerance (clamped)
    t_low, t_high = t_center - TOLERANCE_TEMP, t_center + TOLERANCE_TEMP

    h_low, h_high = h_center - TOLERANCE_HUM, h_center + TOLERANCE_HUM
    h_low  = max(0.0,  h_low)
    h_high = min(100.0, h_high)

    p_low, p_high = p_center - TOLERANCE_PH, p_center + TOLERANCE_PH
    p_low  = max(0.0,  p_low)
    p_high = min(14.0, p_high)

    # Soil alternatives (top 3)
    soil_list = soil_alternatives.loc[target] if target in soil_alternatives.index else [req['soil_texture']]
    best_soil = soil_list[0] if len(soil_list) > 0 else req['soil_texture']
    alt_soils = soil_list[1:] if len(soil_list) > 1 else []

    # --- PART A: THE CROP GUIDE ---
    st.subheader(f"📖 Reference Guide for {target}")
    st.markdown("Standard environmental requirements (from dataset means and tolerances):")
    
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Typical Temp",     f"{t_center:.1f} °C", f"{t_low:.1f}–{t_high:.1f}")
    g2.metric("Typical Humidity", f"{h_center:.1f} %", f"{h_low:.1f}–{h_high:.1f}")
    g3.metric("Typical pH",       f"{p_center:.1f}",   f"{p_low:.1f}–{p_high:.1f}")
    g4.metric("Optimal Soil",      best_soil, ", ".join(alt_soils) if alt_soils else "")
    
    st.divider()
    
    # --- PART B: FERTILIZER CALCULATOR ---
    st.subheader(f"💊 Fertilizer Calculator")
    st.write("Enter your lab test results to calculate exact dosage.")
    
    c1, c2, c3 = st.columns(3)
    cur_n = c1.number_input("Current Nitrogen (N)", 0, 200, 0, help="Value from Soil Test Report")
    cur_p = c2.number_input("Current Phosphorus (P)", 0, 200, 0)
    cur_k = c3.number_input("Current Potassium (K)", 0, 200, 0)
    
    # Calculate Deficit
    need_n = max(0, req['N'] - cur_n)
    need_p = max(0, req['P'] - cur_p)
    need_k = max(0, req['K'] - cur_k)
    
    # Display Deficit
    k1, k2, k3 = st.columns(3)
    k1.metric("Add Nitrogen",   f"{need_n:.0f} kg/ha", delta="-Deficit" if need_n > 0 else "OK", delta_color="inverse")
    k2.metric("Add Phosphorus", f"{need_p:.0f} kg/ha", delta="-Deficit" if need_p > 0 else "OK", delta_color="inverse")
    k3.metric("Add Potassium",  f"{need_k:.0f} kg/ha", delta="-Deficit" if need_k > 0 else "OK", delta_color="inverse")
    
    st.divider()

    # --- PART C: SMART SCHEDULE ---
    st.subheader(f"📅 Lifecycle Application Schedule")
    
    if need_n == 0 and need_p == 0 and need_k == 0:
        st.success("🎉 Your soil is perfectly balanced! No fertilizer needed.")
    else:
        schedule_df = get_fertilizer_schedule(target, need_n, need_p, need_k)
        st.table(schedule_df)
        
        trees = ['apple', 'banana', 'coconut', 'coffee', 'grapes', 'mango', 'orange', 'papaya', 'pomegranate']
        short_crops = ['mungbean', 'blackgram', 'lentil', 'mothbeans', 'watermelon', 'muskmelon']
        
        if target.lower() in trees:
            st.caption("ℹ️ **Logic:** Trees feed seasonally (Spring/Harvest), not by days.")
        elif target.lower() in short_crops:
            st.caption("ℹ️ **Logic:** Short crops need nutrients front-loaded.")
        else:
            st.caption("ℹ️ **Logic:** Cereals and others use a split schedule across key growth stages.")