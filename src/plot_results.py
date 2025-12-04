import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, '../results')
    csv_path = os.path.join(results_dir, 'experiment_results.csv')
    df = pd.read_csv(csv_path)
    models = list(df['Model'])
    x = np.arange(len(models))
    w = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1 = axes[0, 0]
    ax1.bar(x - w/2, df['Train Accuracy'], width=w, label='Train')
    ax1.bar(x + w/2, df['Test Accuracy'], width=w, label='Test')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Train vs Test Accuracy by Model')
    ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2 = axes[0, 1]
    ax2.bar(x, df['Gap'], width=0.6, color='tab:orange')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha='right')
    ax2.set_ylabel('Gap (Train - Test)')
    ax2.set_title('Generalization Gap by Model')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax3 = axes[1, 0]
    ax3.bar(x, df['F1 Score'], width=0.6, color='tab:green')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=30, ha='right')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score by Model')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax4 = axes[1, 1]
    dp_df = df[df['Model'].str.startswith('DP')]
    ax4.plot(dp_df['Epsilon'], dp_df['Test Accuracy'], marker='o', linestyle='-', label='DP Models')
    base_acc = df[df['Model'] == 'Baseline (Non-DP)']['Test Accuracy'].values
    if len(base_acc) > 0:
        ax4.axhline(y=base_acc[0], color='r', linestyle='--', label='Baseline')
    ax4.set_xlabel('Privacy Budget (Epsilon)')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('DP: Test Accuracy vs Epsilon')
    ax4.legend()
    ax4.grid(True)
    fig.tight_layout()
    out_path = os.path.join(results_dir, 'summary_all.png')
    plt.savefig(out_path)
    print(f'Summary plot saved to {out_path}')

if __name__ == '__main__':
    main()
