#!/usr/bin/env python3
import subprocess
import os
import time
import json

STATE_FILE = "rl_state.json"

def save_state(iteration):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_iteration": iteration}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_iteration", -1)
    return -1

def run_step(cmd):
    print(f"Executing: {' '.join(cmd)}")
    # Add repo/python to PYTHONPATH
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    python_path = os.path.join(repo_root, "python")
    
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = python_path
        
    process = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    if process.returncode != 0:
        print(f"Error executing step: {cmd}")
        return False
    return True

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(repo_root, "checkpoints_rl")
    output_dir = os.path.join(repo_root, "data/selfplay_rl")
    engine_path = os.path.join(repo_root, "build/liquid_chess")
    
    # "Zero" scaling parameters
    num_iterations = 1000
    games_per_iteration = 10 
    training_epochs = 1
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    last_completed = load_state()
    start_iter = last_completed + 1

    print("=== STARTING FLUXFISH ZERO RL LOOP ===")
    if start_iter > 0:
        print(f"Resuming from Iteration {start_iter+1}...")
    
    import multiprocessing
    max_workers = multiprocessing.cpu_count()
    # Use 75% of cores for chess engines, leaving 25% for OS/Model management
    num_workers = max(1, int(max_workers * 0.75))

    for i in range(start_iter, num_iterations):
        # Scale simulations over time
        simulations = 50 + (i * 2) 
        if simulations > 800: simulations = 800        

        print(f"\n--- Iteration {i+1} (Sims: {simulations}) ---")
        
        # 1. Generate self-play data
        generate_cmd = [
            "python", "generate_selfplay.py",
            "--num-games", str(games_per_iteration),
            "--simulations", str(simulations),
            "--output-dir", output_dir,
            "--hidden-dim", "512",
            "--workers", str(num_workers)
        ]
        
        # Prefer the ultra-fast C++ engine for massive data generation
        if os.path.exists(engine_path):
            generate_cmd += ["--engine", engine_path]
        elif os.path.exists(os.path.join(checkpoint_dir, "checkpoint_0")) or any(os.listdir(checkpoint_dir)):
            # Fallback to model-guided search if C++ engine is missing
            generate_cmd += ["--checkpoint-dir", checkpoint_dir]
            
        if not run_step(generate_cmd): break
        
        # 2. Train on Replay Buffer (all npz files in output_dir)
        train_cmd = [
            "python", "train_rl.py",
            "--data-path", os.path.join(output_dir, "selfplay_*.npz"),
            "--epochs", str(training_epochs),
            "--checkpoint-dir", checkpoint_dir,
            "--hidden-dim", "512",
            "--batch-size", "32" # More stable for 512-dim model
        ]
        if i > 0 or os.path.exists(checkpoint_dir):
            train_cmd += ["--resume"]
            
        if not run_step(train_cmd): break
        
        save_state(i)
        print(f"Iteration {i+1} complete. Model updated.")

if __name__ == "__main__":
    main()
