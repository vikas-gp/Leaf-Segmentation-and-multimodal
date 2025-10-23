import re
import numpy as np
import pandas as pd

results = {}

with open("train_logs.txt", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines[:-1]):
    # Match the epoch line with fold, seed, val loss
    m = re.match(
        r"\[Fold (\d+) \| Seed (\d+)\] Epoch \d+/500 \| Train [\d.]+ \| Val ([\d.]+) \| LR [\deE\-.]+", line)
    if m:
        fold = int(m.group(1))
        seed = int(m.group(2))
        val_loss = float(m.group(3))

        next_line = lines[i + 1].strip()
        # Check if next line says best checkpoint saved
        if next_line.startswith("✅ Saved best checkpoint"):
            if seed not in results:
                results[seed] = {}
            # Save the minimum val loss per fold (in case multiple best checkpoints)
            if fold not in results[seed] or val_loss < results[seed][fold]:
                results[seed][fold] = val_loss

# Print results summary
print("\n" + "=" * 80)
print("K-FOLD CROSS-VALIDATION RESULTS")
print("=" * 80 + "\n")

all_losses = []
for seed in sorted(results.keys()):
    print(f"\n{'SEED ' + str(seed):^80}")
    print("-" * 80)
    print(f"{'Fold':<10} {'Best Val Loss':<20}")
    print("-" * 80)

    fold_losses = []
    for fold in sorted(results[seed].keys()):
        loss = results[seed][fold]
        fold_losses.append(loss)
        all_losses.append(loss)
        print(f"{fold:<10} {loss:<20.4f}")

    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses, ddof=1)

    print("-" * 80)
    print(f"{'Mean:':<10} {mean_loss:<20.4f}")
    print(f"{'Std Dev:':<10} {std_loss:<20.4f}")
    print(f"{'Min:':<10} {min(fold_losses):<20.4f}")
    print(f"{'Max:':<10} {max(fold_losses):<20.4f}")

# Overall summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY (All Seeds & Folds)")
print("=" * 80)
overall_mean = np.mean(all_losses)
overall_std = np.std(all_losses, ddof=1)
print(f"Mean Val Loss:   {overall_mean:.4f}")
print(f"Std Dev:         {overall_std:.4f}")
print(f"Min Val Loss:    {min(all_losses):.4f}")
print(f"Max Val Loss:    {max(all_losses):.4f}")
print(f"Total Folds:     {len(all_losses)}")
print("=" * 80 + "\n")

# Save CSV
df_data = []
for seed in sorted(results.keys()):
    for fold in sorted(results[seed].keys()):
        df_data.append({
            'Seed': seed,
            'Fold': fold,
            'Best_Val_Loss': results[seed][fold]
        })

df = pd.DataFrame(df_data)
df.to_csv("kfold_results_summary.csv", index=False)
print("✅ Results saved to 'kfold_results_summary.csv'")

# Best checkpoint info
best_idx = np.argmin(all_losses)
best_entry = df_data[best_idx]
print(f"\n🏆 BEST CHECKPOINT:")
print(f"   Seed {best_entry['Seed']}, Fold {best_entry['Fold']}")
print(f"   Val Loss: {best_entry['Best_Val_Loss']:.4f}")
print(f"   Checkpoint: ckpt_unet_fold{best_entry['Fold']}_bs16_s{best_entry['Seed']}.pth\n")
