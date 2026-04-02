import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os


#-------------------------------

def read_frames(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Read the first line to get the number of rows and columns
    rows, cols = map(int, lines[0].split())
   
	# number of matrices is in the second line
    num_matrices = int(lines[1].strip())  
    data = [list(map(float, l.split())) for l in lines[2:] if l.strip()]
    
    # Convert the list of lists to a flat array
    data = np.array(data).flatten()

    # Verify if the number of data points matches the expected size
    if len(data) != num_matrices * rows * cols:
        print(f"Error: the file does not contain the correct number of data points for {num_matrices} matrices of {rows}x{cols}, it contains {len(data)}.")
        sys.exit()

    # Cretae a list of matrices
    matrices = [data[i * rows * cols : (i + 1) * rows * cols].reshape(rows, cols) for i in range(num_matrices)]
    
    print(f"{num_matrices} matrices with dimensions {rows} x {cols} have been loaded")

    terrain = matrices[0]  # First matrix is the terrain
    return rows, cols, terrain, matrices[1:]


#------------- 
    
def animate_matrices(rows, cols, terrain, matrices, filename):
    """Creates an animation showing the evolution of water over a terrain."""
   
    fig, ax = plt.subplots(figsize=(30, 30))  # ajust the figure size to 30x30 inches
    ax.set_aspect('equal')  # maintain aspect ratio


    # Draw the terrain as a background
    img_terrain = ax.imshow(terrain, cmap="YlOrBr_r", interpolation="nearest", extent=[0, 30, 0, 30])

	# Wait for a key press to start the animation
    plt.waitforbuttonpress()

    # Create a colormap for the water
    # Using a blue colormap to represent water
    cmap = plt.cm.Blues

    # Replace 0 values with NaN so the terrain is visible where there is no water
    matrices = [np.where(m == 0, np.nan, m) for m in matrices]

    # Initialize the water image with transparency over the terrain
    img_water = ax.imshow(matrices[0], cmap=cmap, interpolation='nearest', extent=[0,30,0,30], vmin=0, vmax=np.nanmax(matrices), alpha=0.7)

    ax.set_title("Water Evolution")
    cbar = plt.colorbar(img_water, ax=ax)
    cbar.set_label("Water Level")

    def update_frame(frame):
        """Updates the plot with the matrix corresponding to the current frame."""
        img_water.set_array(matrices[frame])
        ax.set_title(f"Time: {frame + 1}")
        return [img_water]  # Return the updated image for the animation

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(matrices), interval=200, repeat=True)

    # Save the animation as a video file
    base, _ = os.path.splitext(filename)
    output_file = base + ".mp4"
    print("Saving file ...")
    ani.save(output_file, writer="ffmpeg", fps=5)
    print("file saved")
    # Show the animation
    #plt.show()


# Main 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python animation.py <file_name>")
        sys.exit(1)
    
    filename = sys.argv[1]
    rows, cols, terrain, frames = read_frames(filename)

    animate_matrices( rows, cols, terrain, frames, filename)

