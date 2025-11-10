import subprocess
import os
import sys
import yaml
import datetime
import torch
import time

# --- HELPER FUNCTIONS ---
def check_gpu():
    """Checks for GPU availability and prints status"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🔥 GPU DETECTED: {gpu_name}")
        return True
    else:
        print("\n⚠️ NO GPU DETECTED. Running in CPU mode.")
        return False

def setup_data_paths():
    """Interactive prompt to set data paths in config.yaml"""
    print(f"\n{'='*50}")
    print("📂 DATASET CONFIGURATION")
    print(f"{'='*50}")

    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_paths = config['data']['raw']
    print("\nCurrent Data Paths:")
    print(f"  [1] RAVDESS Speech: {raw_paths.get('ravdess_speech', 'Not Set')}")
    print(f"  [2] RAVDESS Song:   {raw_paths.get('ravdess_song', 'Not Set')}")
    print(f"  [3] CREMA-D:        {raw_paths.get('crema_d', 'Not Set')}")
    
    use_defaults = input("\nKeep these paths? (y/n): ").lower().strip()
    if use_defaults == 'y' or use_defaults == '':
        print("✅ Using default paths.")
        return

    print("\nENTER NEW PATHS (leave blank to keep current):")
    
    for key, pretty_name in [
        ('ravdess_speech', 'RAVDESS Speech'), 
        ('ravdess_song', 'RAVDESS Song'), 
        ('crema_d', 'CREMA-D')
    ]:
        current = raw_paths.get(key, '')
        new_path = input(f"{pretty_name} [{current}]: ").strip()
        
        if new_path:
            if os.path.exists(new_path):
                 config['data']['raw'][key] = new_path
            else:
                print(f"⚠️ WARNING: Path not found: {new_path}")
                retry = input("Use it anyway? (y/n): ").lower().strip()
                if retry == 'y':
                     config['data']['raw'][key] = new_path
                else:
                    print(f"Keeping old path for {pretty_name}")

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print("\n✅ config.yaml updated with new paths.")

def run_command(command, step_name):
    print(f"\n{'='*50}\n🚀 STARTING: {step_name}\n{'='*50}\n")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"\n✅ COMPLETED: {step_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {step_name}\nError: {e}")
        if "Docker" not in step_name: sys.exit(1)

def generate_report(gpu_active):
    print("\n📝 Generating REPORT.md...")
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    
    if os.path.exists('results.yaml'):
        with open('results.yaml', 'r') as f: results = yaml.safe_load(f)
    else:
        results = {'best_val_loss': 'N/A', 'final_accuracy': 0, 'final_f1_score': 0}

    docker_cmd = "docker run --gpus all -p 8000:8000 emotitune-api" if gpu_active else "docker run -p 8000:8000 emotitune-api"

    md = f"""# 🎵 Emotitune Report
**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} | **Hardware:** `{'GPU 🚀' if gpu_active else 'CPU 🐌'}`

## 📊 Results
| Accuracy | F1-Score | Best Loss |
| :--- | :--- | :--- |
| `{results['final_accuracy']*100:.2f}%` | `{results['final_f1_score']:.4f}` | `{results['best_val_loss']}` |

## 🐳 Deployment
```bash
{docker_cmd}

```
"""
    with open('REPORT.md', 'w') as f: 
        f.write(md) 
        print("✨ REPORT.md created!")

def main(): 
    gpu_active = check_gpu()
    # 1. Interactive Setup
    setup_data_paths()

    # 2. Run Pipeline Steps
    py = sys.executable
    run_command(f"{py} src/audio/preprocess.py", "Feature Extraction")
    run_command(f"{py} src/audio/cnn_baseline.py", "Model Training")
    run_command("docker build -t emotitune-api .", "Docker Build")

    generate_report(gpu_active)
    print("\n🎉 PIPELINE FINISHED!")

if __name__ == "__main__":
    main()