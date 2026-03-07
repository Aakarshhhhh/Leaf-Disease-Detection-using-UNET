import matplotlib.pyplot as plt
import numpy as np
import os

def generate_metric_plot(save_dir="outputs"):
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Data to plot
    metrics = ["Dice Score", "IoU Score"]
    values = [82.19, 71.72]
    
    # Modern color palette setup
    colors = ['#2E7D32', '#4CAF50']  # Dark Green, Lighter Green
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bars
    bars = ax.bar(metrics, values, color=colors, width=0.5, 
                  edgecolor='none', alpha=0.9)
    
    # Stylize the plot
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', color='#1B5E20')
    ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', color='#1B5E20', pad=20)
    
    # Clean up the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    
    # Customize ticks
    ax.tick_params(axis='x', colors='#2E7D32', labelsize=11)
    ax.tick_params(axis='y', colors='#2E7D32', labelsize=10)
    
    # Add horizontal gridlines behind bars
    ax.yaxis.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'{height:.2f}%',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#1B5E20')
    
    plt.tight_layout()
    
    # Save chart
    save_path = os.path.join(save_dir, "metrics_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully at: {save_path}")
    
    # Display plot interactively
    plt.show()

if __name__ == "__main__":
    generate_metric_plot()
