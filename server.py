#server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import uvicorn
import os
from datetime import datetime

app = FastAPI()

# ---------------- SHARED STORAGE FILE ----------------
DATA_FILE = "sensor_data.json"

# ---------------- INITIALIZE DATA FILE ----------------
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump({
            "temperature": 25.0,
            "humidity": 50.0,
            "ph": 6.5,
            "moisture": 50,  # Safe default baseline
            "pump": False,
            "timestamp": ""
        }, f, indent=4)


# ================= ROOT ENDPOINT =================
@app.get("/")
async def home():
    return {"message": "Smart Agriculture Monitoring Server Running"}


# ================= GET SENSOR DATA (READ BY STREAMLIT) =================
@app.get("/data")
async def get_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ================= RECEIVE TELEMETRY FROM ESP32 =================
@app.post("/update")
async def receive_data(request: Request):
    try:
        data = await request.json()

        # 1. Sanitize incoming payload to clean out sensor wire dropouts or NaN values
        for key in ["temperature", "humidity", "ph", "moisture"]:
            val = data.get(key)
            if val is None or str(val).lower() == "nan":
                if key == "moisture": data[key] = 50
                elif key == "temperature": data[key] = 25.0
                elif key == "humidity": data[key] = 50.0
                elif key == "ph": data[key] = 6.5

        # 2. Append processing timestamp
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. Ensure pump status is explicitly recorded as a boolean fallback
        if "pump" not in data:
            data["pump"] = False

        # 4. Save sanitized configuration state straight to file
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)

        # 5. Output live stream trace directly to terminal console
        print("\n========== LIVE MONITORING STREAM ==========")
        print(f"Temperature : {data.get('temperature')} °C")
        print(f"Humidity    : {data.get('humidity')} %")
        print(f"Soil pH     : {data.get('ph')}")
        print(f"Moisture    : {data.get('moisture')} %")
        print(f"Pump Relay  : {'ON' if data.get('pump') else 'OFF'}")
        print(f"Timestamp   : {data.get('timestamp')}")
        print("============================================\n")

        # 6. Return a basic success response confirmation to the ESP32
        return JSONResponse({
            "status": "success",
            "message": "Telemetry updated successfully"
        })

    except Exception as e:
        print("🚨 INSTANCE RUNTIME EXCEPTION:", e)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    # 0.0.0.0 lets any local network device connect to your app
    uvicorn.run(app, host="0.0.0.0", port=8000)
