#!/usr/bin/env python3
"""Simple script to check if training is complete."""

import os
import time
from datetime import datetime

def check_training_status():
    """Check if training process is still running."""
    
    # Check if final model exists
    final_model_path = "models/checkpoints/final_model.pth"
    best_model_path = "models/checkpoints/best_model.pth"
    
    print(f"🔍 Training Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check if process is running
    try:
        import subprocess
        result = subprocess.run(['python', '-c', 'import psutil; print("psutil available")'], 
                              capture_output=True, text=True)
        psutil_available = result.returncode == 0
    except:
        psutil_available = False
    
    if os.path.exists(final_model_path):
        print("✅ TRAINING COMPLETE!")
        print(f"   Final model saved: {final_model_path}")
        
        # Get file modification time
        mod_time = os.path.getmtime(final_model_path)
        mod_datetime = datetime.fromtimestamp(mod_time)
        print(f"   Completed at: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
    elif os.path.exists(best_model_path):
        print("🔄 TRAINING IN PROGRESS...")
        print(f"   Best model so far: {best_model_path}")
        
        # Get file modification time
        mod_time = os.path.getmtime(best_model_path)
        mod_datetime = datetime.fromtimestamp(mod_time)
        print(f"   Last updated: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check how recent the update was
        time_diff = time.time() - mod_time
        if time_diff < 600:  # Less than 10 minutes
            print(f"   🟢 Recently active ({int(time_diff/60)} minutes ago)")
        else:
            print(f"   🟡 Last activity: {int(time_diff/60)} minutes ago")
        
    else:
        print("❓ No model files found - training may not have started")
    
    print("=" * 60)
    print("💡 Run this script anytime to check status:")
    print("   python check_training.py")

if __name__ == "__main__":
    check_training_status()