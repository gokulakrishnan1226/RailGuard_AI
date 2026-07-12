# Installation Guide

Follow these steps to set up the RailGuard AI environment on Windows.

## 1. Prerequisites
- Python 3.12 (ensure Python is added to environmental PATH).
- Git.
- (Optional) MySQL Server.
- C++ Build Tools (required for some compiled dependencies like `pyttsx3` or `insightface` if compiling from source).

## 2. Virtual Environment Setup
Open PowerShell in the project workspace and run:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies
Install Python libraries using:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

*Note: If pyttsx3 fails to compile or speak on Windows, ensure the `SAPI5` speech subsystem is active or install `pywin32`.*

## 4. Configuration Check
Open [config.json](file:///d:/Projects/Railguard%20ai/config.json) and verify setting variables:
- To run with MySQL, set `"use_mock": false` and add username/password details.
- Otherwise, leave `"use_mock": true` for local in-memory fallback.
- Configure camera indexes under `"inference"`.

## 5. Launch the Server
To spin up the web interface, execute:
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
Navigate to `http://127.0.0.1:8000/login` in your web browser. Login credentials:
- Username: `admin`
- Password: `admin123`
