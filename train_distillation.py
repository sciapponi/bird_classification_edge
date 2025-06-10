#!/usr/bin/env python3
"""
Convenience script to run knowledge distillation training.
This script calls the actual implementation in distillation/scripts/train_distillation.py
"""

import subprocess
import sys
import os

def main():
    # Get the script path
    script_path = os.path.join("distillation", "scripts", "train_distillation.py")
    
    # Forward all arguments to the actual script
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    # Run the script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 