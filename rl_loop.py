#!/usr/bin/env python3
import subprocess
import os
import time

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
    checkpoint_dir = "checkpoints_rl"
    output_dir = "data/selfplay_rl"
    engine_path = "build/liquid_chess_mcts" # Use our new C++ MCTS engine
    num_iterations = 10
    games_per_iteration = 5
    simulations_per_move = 50
    training_epochs = 5
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=== STARTING REINFORCEMENT LEARNING LOOP ===")
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        # 1. Generate self-play data
        generate_cmd = [
            "python", "generate_selfplay.py",
            "--num-games", str(games_per_iteration),
            "--simulations", str(simulations_per_move),
            "--output-dir", output_dir
        ]
        
        # Prefer C++ engine for generation if it exists
        if os.path.exists(engine_path):
            generate_cmd += ["--engine", engine_path]
        elif i > 0:
            generate_cmd += ["--checkpoint-dir", checkpoint_dir]
            
        if not run_step(generate_cmd): break
        
        # Find latest selfplay file
        files = [f for f in os.listdir(output_dir) if f.startswith("selfplay_")]
        latest_file = os.path.join(output_dir, sorted(files)[-1])
        
        # 2. Train on the new data
        train_cmd = [
            "python", "train_rl.py",
            "--data-path", latest_file,
            "--epochs", str(training_epochs),
            "--checkpoint-dir", checkpoint_dir
        ]
        if i > 0:
            train_cmd += ["--resume"]
            
        if not run_step(train_cmd): break
        
        print(f"Iteration {i+1} complete. Model updated.")

if __name__ == "__main__":
    main()
