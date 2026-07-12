import traceback
import sys
import os

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Request
from api.main import app, get_login

try:
    print("Creating mock request...")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/login",
        "headers": [],
        "query_string": b"",
        "server": ("127.0.0.1", 8000),
    }
    req = Request(scope=scope)
    
    print("Calling get_login...")
    response = get_login(req)
    print("Render successful! Response type:", type(response))
    print("Body length:", len(response.body))
except Exception as e:
    print("An exception occurred:")
    traceback.print_exc()
