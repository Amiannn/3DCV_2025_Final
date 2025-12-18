import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import shutil

def get_image_path(csv_filename):
    """
    Converts the CSV filename to the corresponding image path.
    Input example: track_visuals_v2/s00493...rot50.4.json_pose_results.json
    Output example: track_visuals_v2/s00493...rot50.4.json_conf_heat_pair_0000_0001.png
    """
    base_name = csv_filename.replace('_pose_results.json', '')
    image_name = base_name + '_conf_heat_pair_0000_0001.png'
    return image_name

def visualize_cases_grid(csv_file, num_success=6, num_error=6, cols=4):
    """
    Visualizes success and error cases in a compact grid layout.
    
    Args:
        csv_file (str): Path to the CSV file.
        num_success (int): Number of success cases to sample.
        num_error (int): Number of error cases to sample.
        cols (int): Number of columns in the grid.
    """
    # 1. Read Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    # 2. Filter Data
    df = df[df['method'] == 'corr'].copy()
    success_df = df[df['max_error'] < 5].copy()
    error_df = df[df['max_error'] > 20].copy()
    
    success_df['type'] = 'Success'
    error_df['type'] = 'Error'

    # 3. Sample Data
    n_s = min(num_success, len(success_df))
    n_e = min(num_error, len(error_df))
    
    if n_s == 0 and n_e == 0:
        print("No data found matching criteria.")
        return

    # Combine samples into one list for plotting
    samples = pd.concat([
        success_df.sample(n=n_s, random_state=42), 
        error_df.sample(n=n_e, random_state=42)
    ])

    samples = samples.sort_values(by='max_error', ascending=True)

    # 4. Setup Grid
    total_plots = len(samples)
    rows = math.ceil(total_plots / cols)
    
    # Create figure (adjust height based on rows)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
    
    # Flatten axes array for easy iteration
    if total_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    fig.suptitle(f'Visualization Results ({n_s} Success, {n_e} Error)', fontsize=20)

    # 5. Plot Images
    for i, (idx, row) in enumerate(samples.iterrows()):
        ax = axes_flat[i]
        
        # Construct path
        img_path = get_image_path(row['filename'])

        print(img_path)
        
        # Determine color and label
        is_success = row['type'] == 'Success'
        color = 'green' if is_success else 'red'
        title_text = f"[{row['type']}] Err: {row['max_error']:.1f}°\n{os.path.basename(img_path)[:30]}..." # Truncate long names

        # track_visuals_v2/s00523_bin_90-135_i309_j579_rot125.3.json_conf_heat_pair_0000_0001.png
        # pairs/s00460/bin_45-90
        filename = img_path.split('/')[-1]
        sid = filename.split('_')[0]
        rot = filename.split('_i')[0].split('_')[-1]
        name = filename.split('.json')[0]
        original_img_path = os.path.join(f'/media/ai2lab/storage1/experiments/mapfree/pairs', sid, f'bin_{rot}', f'{name}.jpg')

        shutil.copy(original_img_path, os.path.join('error_study', f'{i}_{is_success}_{name}.jpg'))
        

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            # Add filename inside the plot or as a smaller title
            ax.set_title(title_text, color=color, fontsize=9, fontweight='bold')
            
            # Add full filename at the bottom of the image for reference
            ax.set_xlabel(row['filename'].split('/')[-1], fontsize=7) 
        else:
            ax.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', color='gray')
            ax.set_title(title_text, color=color, fontsize=9)
            print(f"Missing: {img_path}")

        ax.set_xticks([])
        ax.set_yticks([])

    # 6. Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'vis.png')

# --- Run the Code ---
# You can adjust 'cols' to make rows shorter or longer
# You can adjust num_success and num_error to show more or fewer images
visualize_cases_grid('track_visuals_v2/pair_errors.csv', num_success=24, num_error=24, cols=6)