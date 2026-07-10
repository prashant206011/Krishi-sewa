# 🌱 Krishi Sewa

An AI-powered Smart Agriculture System that integrates IoT and Machine Learning to monitor soil conditions, automate irrigation, and recommend suitable crops and fertilizers for sustainable farming.

## 📖 Overview

Krishi Sewa is a final-year engineering project developed to help farmers make data-driven agricultural decisions. The system uses an ESP32 microcontroller with soil moisture, temperature, humidity, and pH sensors to collect real-time environmental data. Based on these inputs, it automates irrigation and provides crop and fertilizer recommendations using a Random Forest machine learning model.

## ✨ Features

- 🌾 Machine Learning-based Crop Recommendation
- 🌱 Intelligent Fertilizer Recommendation
- 💧 Automatic Irrigation using Soil Moisture Sensor
- 🌡️ Real-time Temperature and Humidity Monitoring
- 🧪 Soil pH Analysis
- 📊 Streamlit Dashboard for Live Monitoring
- ⚡ ESP32-based IoT System
- 🔗 FastAPI Backend for Communication

## 🛠️ Technologies Used

- Python
- ESP32
- Streamlit
- FastAPI
- Random Forest
- PlatformIO
- Embedded C++
- Pandas
- NumPy

##  Hardware Components

- ESP32 Development Board
- Soil Moisture Sensor
- DHT11 Temperature & Humidity Sensor
- pH Sensor
- Relay Module
- Water Pump

## 📂 Repository Structure

```
Krishi-sewa/
│── app.py
│── server.py
│── main.cpp
│── dataset.csv
│── README.md
```

## ⚙️ System Workflow

1. ESP32 collects real-time sensor data.
2. Soil moisture controls automatic irrigation.
3. Temperature, humidity, and pH values are processed.
4. The Random Forest model predicts suitable crops.
5. Fertilizer recommendations are generated.
6. Results are displayed on the Streamlit dashboard.

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/prashant206011/Krishi-sewa.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Start the FastAPI server:

```bash
python server.py
```


## 🔮 Future Enhancements

- Weather API Integration
- Cloud Database Support
- Mobile Application
- Pest and Disease Detection
- Advanced AI Models

## 👨‍💻 Team Members

- Bhupendra Singh
- Prashanta Upadhyaya
- Rohit Kumar Shah
- Sagar Joshi

## 🎓 Academic Information

**Institute:** Institute of Engineering (IOE), Pashchimanchal Campus, Pokhara

**Department:** Electronics and Computer Engineering

**Project:** Major Project

## 📄 License

This project is intended for educational and research purposes.
