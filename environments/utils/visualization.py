
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import io

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_map(m: np.ndarray, localx: Optional[List[float]] = None, 
            localy: Optional[List[float]] = None, 
            robot_pos: Optional[Tuple[float, float]] = None,
            frontiers_x: Optional[List[int]] = None,
            frontiers_y: Optional[List[int]] = None,
            title: Optional[str] = None) -> Optional[plt.Figure]:
   
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Cannot create plot.")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.imshow(np.subtract(1, m), cmap='gray', vmin=0, vmax=1, origin='lower')
    
    if localx is not None and localy is not None and len(localx) > 0:
        ax.plot(localx, localy, 'r-', linewidth=2, label='Path')
        ax.plot(localx[0], localy[0], 'go', markersize=8, label='Start')
        ax.plot(localx[-1], localy[-1], 'bo', markersize=8, label='Current')
    

    if robot_pos is not None:
        ax.plot(robot_pos[1], robot_pos[0], 'ro', markersize=10, label='Robot')
    
    if frontiers_x is not None and frontiers_y is not None:
        ax.plot(frontiers_x, frontiers_y, 'co', markersize=6, label='Frontiers')
    
    if title:
        ax.set_title(title)
    
    ax.legend(loc='upper right')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig


def plot_map_with_pheromones(m: np.ndarray, pher_map: np.ndarray, 
                            localx: Optional[List[float]] = None, 
                            localy: Optional[List[float]] = None, 
                            robot_pos: Optional[Tuple[float, float]] = None) -> Optional[plt.Figure]:
    
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Cannot create plot.")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(np.subtract(1, m), cmap='gray', vmin=0, vmax=1, origin='lower')
    pheromone_plot = axes[1].imshow(pher_map, cmap='hot', origin='lower')
    plt.colorbar(pheromone_plot, ax=axes[1], label='Pheromone Intensity')
    if localx is not None and localy is not None and len(localx) > 0:
        axes[0].plot(localx, localy, 'r-', linewidth=2)
        axes[1].plot(localx, localy, 'c-', linewidth=2)
        axes[0].plot(localx[0], localy[0], 'go', markersize=8)
        axes[1].plot(localx[0], localy[0], 'go', markersize=8)
        axes[0].plot(localx[-1], localy[-1], 'bo', markersize=8)
        axes[1].plot(localx[-1], localy[-1], 'bo', markersize=8)
    

    if robot_pos is not None:
        axes[0].plot(robot_pos[1], robot_pos[0], 'ro', markersize=10)
        axes[1].plot(robot_pos[1], robot_pos[0], 'ro', markersize=10)
    

    axes[0].set_title('Map Representation')
    axes[1].set_title('Pheromone Distribution')
    

    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig


def generate_animation(map_frames: List[np.ndarray], 
                      path_x: List[List[float]], 
                      path_y: List[List[float]], 
                      output_filename: str,
                      robot_number: int = 1, 
                      fps: int = 100) -> Optional[str]:
    
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Cannot create animation.")
        return None
    
    writer = PillowWriter(fps=fps)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    color_list = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    
    max_path_len = 0
    for i in range(robot_number):
        if i < len(path_x) and len(path_x[i]) > max_path_len:
            max_path_len = len(path_x[i])
    
    with writer.saving(fig, output_filename, fps):
        for frame in range(max_path_len):
            ax.clear()
            
            if frame < len(map_frames):
                ax.imshow(np.subtract(1, map_frames[frame]), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.subtract(1, map_frames[-1]), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
            
            for i in range(min(robot_number, len(path_x))):
                color = color_list[i % len(color_list)]
                
                if frame < len(path_x[i]):
                    ax.plot(path_x[i][:frame+1], path_y[i][:frame+1], '-', color=color, linewidth=2)
                    ax.plot(path_x[i][frame], path_y[i][frame], 'o', color=color, markersize=8)
                else:
                    ax.plot(path_x[i], path_y[i], '-', color=color, linewidth=2)
                    ax.plot(path_x[i][-1], path_y[i][-1], 'o', color=color, markersize=8)
                    
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Frame {frame+1}/{max_path_len}')
            
            writer.grab_frame()
    
    return output_filename


def render_map_to_rgb(m: np.ndarray, 
                     localx: Optional[List[float]] = None, 
                     localy: Optional[List[float]] = None, 
                     robot_pos: Optional[Tuple[float, float]] = None,
                     frontiers_x: Optional[List[int]] = None,
                     frontiers_y: Optional[List[int]] = None,
                     dpi: int = 100) -> Optional[np.ndarray]:
    
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Cannot render map.")
        return None
    
    fig = plot_map(m, localx, localy, robot_pos, frontiers_x, frontiers_y)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    
    img = np.array(Image.open(buf))
    
    plt.close(fig)
    
    return img


def find_max_len(arrays: List[List[Any]]) -> int:
    max_len = 0
    max_idx = 0
    
    for i, arr in enumerate(arrays):
        if len(arr) > max_len:
            max_len = len(arr)
            max_idx = i
            
    return max_idx