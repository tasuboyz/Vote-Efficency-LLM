import matplotlib.pyplot as plt
import seaborn as sns

def configure_plotting_style():
    """Configure the default plotting style for the application."""
    # Set style using a valid matplotlib style
    plt.style.use('default')  # Reset to default style first
    
    # Configure seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = [10, 6]
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Set color palette
    sns.set_palette("husl")
    
    # Additional customization
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False