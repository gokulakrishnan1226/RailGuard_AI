# Production Deployment Guide

Follow these guidelines to deploy RailGuard AI to a production server environment.

## 1. Production Web Server (Uvicorn / Gunicorn)
For production environments, run FastAPI with multiple worker processes using Gunicorn:
```bash
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 2. Nginx Reverse Proxy
It is recommended to place Nginx in front of the application server to handle SSL termination, rate limiting, and serve static assets:

```nginx
server {
    listen 80;
    server_name railguard.infra.local;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Serve static assets directly
    location /static/ {
        alias /path/to/railguard_ai/website/static/;
        expires 7d;
    }
}
```

## 3. Database Migration
To migrate from the mock/offline mode to a production MySQL Database:
1. Connect to your MySQL server instance:
   ```bash
   mysql -u root -p < database/schema.sql
   ```
2. In `config.json`, configure the `"database"` block:
   ```json
   "database": {
       "host": "production-db-instance-ip",
       "user": "railguard_admin",
       "password": "secure_password",
       "database": "railguard_db",
       "port": 3306,
       "use_mock": false
   }
   ```

## 4. Hardware Deployment
For a physical patrolling cart:
1. Mount the ESP32 with battery pack, H-bridge motor drivers, vibration sensor, and SG90 pan-tilt servo.
2. Compile and upload `esp32/railguard_esp32/railguard_esp32.ino` using the Arduino IDE.
3. Establish a local WiFi network (hotspot) matching the credentials in the Arduino sketch.
4. Mount the camera to the pan-tilt servo and connect it to the server system.
5. In `config.json`, set `"enabled": true` and configure the ESP32's allocated IP address.
