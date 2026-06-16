import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_temporal_split(output_file='temporal_split.pdf'):
    """
    Create a timeline figure illustrating the temporal hold-out validation split.
    """
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(1998, 2023)
    ax.set_ylim(-0.5, 1.5)
    
    # Timeline axis
    ax.axhline(y=0, color='black', linewidth=2, zorder=0)
    
    # Training + Validation period (2000–2017)
    ax.broken_barh([(2000, 18)], (-0.1, 0.2), facecolors='lightblue', edgecolor='black', linewidth=1, alpha=0.7)
    ax.broken_barh([(2000, 18)], (0.8, 0.2), facecolors='lightblue', edgecolor='black', linewidth=1, alpha=0.7)
    
    # Test (Hold‑out) period (2018–2021)
    ax.broken_barh([(2018, 4)], (-0.1, 0.2), facecolors='lightcoral', edgecolor='black', linewidth=1, alpha=0.7)
    ax.broken_barh([(2018, 4)], (0.8, 0.2), facecolors='lightcoral', edgecolor='black', linewidth=1, alpha=0.7)
    
    # Annotate year ticks
    years = [2000, 2005, 2010, 2015, 2018, 2021]
    for y in years:
        ax.plot([y, y], [-0.05, 0.05], color='black', linewidth=1)
        ax.text(y, -0.15, str(y), ha='center', va='top', fontsize=10)
    
    # Add arrows / brackets for periods
    # Using annotate with arrows or just text with horizontal lines – simpler with text and braces
    # For publication, we can use text with horizontal lines as a clean alternative
    
    # Training + Validation label
    ax.annotate('', xy=(2000, 0.25), xytext=(2017, 0.25),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((2000+2017)/2, 0.35, 'Training + Validation (2000–2017)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Test label
    ax.annotate('', xy=(2018, 0.25), xytext=(2021, 0.25),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((2018+2021)/2, 0.35, 'Test / Hold‑out (2018–2021)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Remove axes spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    
    # Title
    ax.set_title('Temporal Hold‑Out Validation Split', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved as {output_file}")

# Run the function
plot_temporal_split('temporal_split.pdf')