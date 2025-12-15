import subprocess
import sys
import os

if __name__ == "__main__":
    # Clear previous logs
    log_file = os.path.join("examples", "run_logs.jsonl")
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Run tests
    subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
    print("Tests complete. Check examples/run_logs.jsonl for outputs.")

