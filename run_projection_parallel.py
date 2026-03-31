import os
import time
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import sys
import datetime

# Import from your customized optimization module
from vendi_projection import optimize_q_vne_torch, _vne_and_grad_wrt_q_torch
# Import the evaluation logic created previously
from evaluation import evaluate_weighted
from conditional_evaluation import ConditionalEvaluation


def vendi_weighted(K_tensor, q_tensor):
    """Helper to compute Vendi score = exp(VNE) directly on GPU tensors."""
    with torch.no_grad():
        vne, _ = _vne_and_grad_wrt_q_torch(K_tensor, q_tensor)
        return float(np.exp(vne))


def process_batches_worker(rank, gpu_id, batches_to_process, fake_data, args, run_dir, return_dict):
    """
    Worker function executed by each GPU.
    Processes assigned batches and saves individual batch weight files.
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[Worker {rank} on GPU {gpu_id}] Starting processing {len(batches_to_process)} batches.")
    
    local_results = []
    local_indices = []
    local_weights = []

    for batch_idx, idxs in batches_to_process:
        print(f"[Worker {rank}] === Batch {batch_idx+1} | size={len(idxs)} ===")
        batch_start = time.time()

        # Load specific batch features
        X = fake_data[idxs]
        
        # Compute Kernel Matrix using the ConditionalEvaluation model
        eval_model = ConditionalEvaluation(sigma=(args.sigma, args.sigma))
        K_tensor = eval_model.gaussian_kernel(X, sigma=args.sigma, batchsize=32, normalize=True)
        K_tensor = K_tensor.to(device, dtype=torch.float32)

        # Uniform Vendi
        q0 = torch.full((len(idxs),), 1.0 / len(idxs), device=device, dtype=torch.float32)
        v0 = vendi_weighted(K_tensor, q0)
        print(f"[Worker {rank}]  Uniform Vendi: {v0:.6f}")

        # Optimize q using numpy interface (explicitly pass the device if your func allows, else map after)
        q_star_np, _, _ = optimize_q_vne_torch(K_np=K_tensor.cpu().numpy(), lambda_=args.lambda_val, device=f"cuda:{gpu_id}")
        
        # Optimized Vendi
        q_star_tensor = torch.tensor(q_star_np, device=device, dtype=torch.float32)
        v_star = vendi_weighted(K_tensor, q_star_tensor)
        print(f"[Worker {rank}]  Optimized Vendi: {v_star:.6f}")

        # Save indices + weights for this batch
        out_path = os.path.join(run_dir, f"batch_{batch_idx+1:02d}_weights.npz")
        np.savez(
            out_path,
            indices=idxs,          
            weights=q_star_np,     
            uniform_vendi=np.float32(v0),
            optimized_vendi=np.float32(v_star),
            sigma=np.float32(args.sigma),
            lam=np.float32(args.lambda_val),
        )
        print(f"[Worker {rank}]  Saved: {out_path}")

        local_results.append((batch_idx, v0, v_star))
        local_indices.append(idxs)
        local_weights.append(q_star_np)

        # Free GPU memory
        del X, K_tensor, q0, q_star_tensor, eval_model
        torch.cuda.empty_cache()
            
        print(f"[Worker {rank}]  Batch time: {time.time() - batch_start:.2f} seconds")

    # Pass results back to main process via thread-safe dict
    return_dict[rank] = {
        'results': local_results,
        'indices': local_indices,
        'weights': local_weights
    }


def main():
    _GLOBAL_START = time.time()
    
    # Required for multiprocessing with PyTorch
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Batched Projected-Vendi Weights Calculation and Evaluation")
    
    # File Paths
    parser.add_argument("--fake_features", type=str, required=True, 
                        help="Path to the fake features .npz file")
    parser.add_argument("--real_features", type=str, required=False, 
                        help="Path to the real/reference features .npz file (Required if evaluate is True)")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Base folder path to save the batched weights and plots")
    
    # Evaluation flag (default True, use --no_evaluate to disable)
    parser.add_argument("--no_evaluate", action="store_true", 
                        help="If passed, skips the evaluation and scores.txt generation")
    
    # Batching and Sampling Parameters
    parser.add_argument("--total_samples", type=int, default=50000, 
                        help="Total number of samples to process from the dataset")
    parser.add_argument("--batch_size", type=int, default=8000, 
                        help="Size of each batch")
    parser.add_argument("--use_first_samples", action="store_true", 
                        help="If passed, sequentially uses the first N samples instead of random shuffling")
    
    # Optimization Hyperparameters
    parser.add_argument("--lambda_val", type=float, default=0.01, 
                        help="Lambda penalty for the VNE regularization")
    parser.add_argument("--sigma", type=float, default=35.0, 
                        help="Bandwidth parameter for the Gaussian Kernel")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random base seed")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), 
                        help="Number of GPUs to use")
    
    args = parser.parse_args()

    # 1. Setup Environment & Unique Directory
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Extract base filename without extension
    fake_basename = os.path.splitext(os.path.basename(args.fake_features))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M")
    
    # Create unique run directory
    run_dir = os.path.join(args.save_dir, f"{fake_basename}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Run directory created at: {run_dir}")
    print(f"Loading fake features from {args.fake_features}...")
    fake_data = np.load(args.fake_features)['dino_features']
    n_total_available = len(fake_data)
    
    n_total = min(args.total_samples, n_total_available)
    
    # 2. Select and Sequence Indices
    if args.use_first_samples:
        print(f"Using the FIRST {n_total} samples sequentially.")
        all_indices = np.arange(n_total)
    else:
        print(f"Randomly sampling {n_total} samples from the dataset.")
        all_indices = np.random.choice(n_total_available, n_total, replace=False)
        
    n_batches = int(np.ceil(n_total / args.batch_size))
    batches = np.array_split(all_indices, n_batches)
    
    print(f"\nSplit {n_total} samples into {n_batches} batches (Max Batch Size: {args.batch_size}).")
    print(f"Distributing workload across {args.num_gpus} GPUs.")

    # 3. Distribute batches among GPUs
    indexed_batches = list(enumerate(batches)) # [(0, batch0_idxs), (1, batch1_idxs), ...]
    batches_per_gpu = [indexed_batches[i::args.num_gpus] for i in range(args.num_gpus)]
    
    # 4. Spawn worker processes
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(args.num_gpus):
        if not batches_per_gpu[rank]: 
            continue # Skip if no batches for this GPU
        
        p = mp.Process(
            target=process_batches_worker,
            args=(rank, rank, batches_per_gpu[rank], fake_data, args, run_dir, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll GPU workers finished. Aggregating results...")

    # 5. Reconstruct global data safely from workers
    batch_results = []
    global_indices = []
    global_weights = []

    for rank in range(args.num_gpus):
        if rank in return_dict:
            batch_results.extend(return_dict[rank]['results'])
            global_indices.extend(return_dict[rank]['indices'])
            global_weights.extend(return_dict[rank]['weights'])

    # Sort results to ensure correct sequential batch order for plotting/printing
    batch_results.sort(key=lambda x: x[0])

    # 6. Save Combined Weights
    global_indices = np.concatenate(global_indices)
    # Average out the weights so the global sum remains 1
    global_weights = np.concatenate(global_weights) / n_batches
    
    combined_path = os.path.join(run_dir, "combined_weights.npz")
    np.savez(
        combined_path,
        indices=global_indices,
        weights=global_weights,
        sigma=np.float32(args.sigma),
        lam=np.float32(args.lambda_val),
    )
    print(f"\nSaved combined global weights across all batches: {combined_path}")

    # 7. Summary Output
    print("\n" + "="*45)
    print(f"{'Batch':>6} | {'Uniform Vendi':>14} | {'Optimized Vendi':>16}")
    print("-" * 45)
    for batch_idx, v0, v_star in batch_results:
        print(f"  {batch_idx+1:4d} | {v0:14.6f} | {v_star:16.6f}")

    # 8. Plotting
    batch_ids = [r[0] + 1 for r in batch_results]
    y_uni = [r[1] for r in batch_results]
    y_opt = [r[2] for r in batch_results]

    plt.figure(figsize=(10, 5))
    plt.plot(batch_ids, y_uni, marker="o", label="Uniform q = 1/n")
    plt.plot(batch_ids, y_opt, marker="o", label=f"Optimized q ($\lambda$={args.lambda_val})")
    plt.xlabel("Batch index")
    plt.ylabel(f"Vendi = exp(VNE) (Gaussian $\sigma$={args.sigma})")
    plt.title("Weighted Vendi per batch after EG reweighting")
    plt.xticks(batch_ids)
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(run_dir, "vendi_per_batch.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    # 9. Final Evaluation and TXT Dump
    evaluate = not args.no_evaluate
    if evaluate:
        print("\n" + "="*45)
        print("Starting Global Evaluation...")
        if not args.real_features:
            raise ValueError("Evaluation is enabled, but --real_features path was not provided.")
            
        print(f"Loading reference features from {args.real_features}...")
        real_data = np.load(args.real_features)['dino_features']
        n_real_available = len(real_data)
        
        # Match real feature selection logic
        if args.use_first_samples:
            real_idxs = np.arange(min(n_total, n_real_available))
        else:
            real_idxs = np.random.choice(n_real_available, min(n_total, n_real_available), replace=False)
            
        real_feats = real_data[real_idxs]
        
        # Ensure fake features order matches the global combined weights order
        fake_feats_eval = fake_data[global_indices]

        # Lock evaluation to default device (cuda:0)
        eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Evaluate Uniform
        print("Evaluating Uniform Metrics...")
        metrics_unif = evaluate_features(real_feats, fake_feats_eval, q_weights=None, device=eval_device)
        
        # Evaluate Optimized
        print("Evaluating Optimized Metrics...")
        metrics_opt = evaluate_features(real_feats, fake_feats_eval, q_weights=global_weights, device=eval_device)

        # Dump to scores.txt
        scores_file = os.path.join(run_dir, "scores.txt")
        with open(scores_file, "w") as f:
            f.write(f"Evaluation Results for: {fake_basename}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Total Samples: {n_total}\n")
            f.write(f"Sampling Mode: {'Sequential (First N)' if args.use_first_samples else 'Random'}\n")
            f.write("="*50 + "\n\n")
            
            f.write("--- UNIFORM WEIGHTS ---\n")
            for k, v in metrics_unif.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        f.write(f"{k.upper()}_{sub_k.upper():<15s}: {float(sub_v):.6f}\n")
                else:
                    f.write(f"{k.upper():<15s}: {float(v):.6f}\n")
                
            f.write("\n--- PROJECTED-VENDI WEIGHTS ---\n")
            for k, v in metrics_opt.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        f.write(f"{k.upper()}_{sub_k.upper():<15s}: {float(sub_v):.6f}\n")
                else:
                    f.write(f"{k.upper():<15s}: {float(v):.6f}\n")
                
        print(f"Evaluation finished. Metrics dumped to: {scores_file}")

    print(f"\nAll operations completed. Results available in: {run_dir}")
    print(f"Total elapsed time: {time.time() - _GLOBAL_START:.2f} seconds")

if __name__ == "__main__":
    main()
