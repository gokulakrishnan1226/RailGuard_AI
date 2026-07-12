import traceback
import sys
import os

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth import verify_password, get_password_hash

test_hash = "$2b$12$R.S2uU61eZgG6z9.qWc2UuU7p60i1nNlV0J0c2K/pM/P7rE.VlyjK"

try:
    print("Verifying password 'admin123' against stored hash...")
    result = verify_password("admin123", test_hash)
    print("Verification result:", result)
    
    # Generate new hash
    new_hash = get_password_hash("admin123")
    print("New hash generated:", new_hash)
    print("Verifying new hash:", verify_password("admin123", new_hash))
except Exception as e:
    print("Verification failed:")
    traceback.print_exc()
