# Copyright (c) OpenMMLab. All rights reserved.
import optuna
import os
import subprocess
import multiprocessing
from optuna.samplers import GridSampler

def objective(trial):
    # Use the trial object to get parameter values.
    # (With a grid sampler, each trial is guaranteed to be unique.)
    res = trial.suggest_categorical("resolution", [384])
    augmentation_index = trial.suggest_categorical("augmentation_index", [0, 1])
    batch_size = trial.suggest_categorical("batch_size", [64])
    repeat_times = trial.suggest_categorical("repeat_times", [2, 3])
    resnet_depth = trial.suggest_categorical("resnet_depth", [50])
    
    # Retrieve the GPU assigned for this process.
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    try:
        process = subprocess.Popen(
            [
                "python", "objective.py",
                "--res", str(res),
                "--augmentation_index", str(augmentation_index),
                "--batch_size", str(batch_size),
                "--repeat_times", str(repeat_times),
                "--resnet_depth", str(resnet_depth),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream the subprocess output in real time.
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.stdout.close()
        process.wait()

    except Exception as e:
        print(f"Error: {e}")
        return float("inf")
    
    # Replace 0 with your actual metric (for maximization or minimization).
    return 0

def run_study(gpu_id, n_trials):
    # Set the GPU for this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Define your search space as a grid.
    search_space = {
        "resolution": [384, 512],
        "augmentation_index": [0, 1],
        "batch_size": [64],
        "repeat_times": [2, 3],
        "resnet_depth": [18, 50],
    }
    sampler = GridSampler(search_space)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="mmpose_optimization",
        storage="sqlite:///mmpose_optimization.db",
        load_if_exists=True,
        sampler=sampler,
    )
    
    study.optimize(objective, n_trials=n_trials)

def main():
    # For grid search, the total unique combinations are 16.
    total_trials = 16  
    trials_per_gpu = total_trials // 2  # e.g. 8 trials per GPU

    # Launch two processes—one per GPU.
    process_gpu0 = multiprocessing.Process(target=run_study, args=(0, trials_per_gpu))
    process_gpu1 = multiprocessing.Process(target=run_study, args=(1, trials_per_gpu))
    
    process_gpu0.start()
    process_gpu1.start()
    
    process_gpu0.join()
    process_gpu1.join()

if __name__ == "__main__":
    main()
