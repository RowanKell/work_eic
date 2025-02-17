'''THIS FILE EXISTS TO HOLD CODE THAT I WROTE IN A NOTEBOOK LIKE GNN_Energy.ipynb
BUT DELETED FOR READABILITY, BUT MAYBE WANTED LATER'''

'''Data related code:'''
# Plotting data input features
# Use after reading data from csv files - data should be one big df
new_fig, new_axes = plot.subplots(3,3, figsize = (10,10))
new_axes[0,0].hist(data["P"],bins = 100);
new_axes[0,0].set_title("P")
new_axes[0,1].hist(data["layer_idx"],bins = 100);
new_axes[0,1].set_title("layer_idx")
new_charge0 = data["Charge1"]
new_charge0 = new_charge0[new_charge0 < 30]
new_axes[0,2].hist(new_charge0,bins = 100);
new_axes[0,2].set_title("Charge0")
new_axes[0,2].set_xlim(0,20)
new_axes[1,0].hist(data["Time1"],bins = 100);
new_axes[1,0].set_title("Time0")
new_axes[1,1].hist(data["Theta"],bins = 100);
new_axes[1,1].set_title("Theta")
new_axes[1,2].hist(data["KMU_endpoint_y"],bins = 100);
new_axes[1,2].set_title("KMU_endpoint_y")
new_axes[2,0].hist(data["strip_x"],bins = 100);
new_axes[2,0].set_title("strip_x")
new_axes[2,1].hist(data["strip_y"],bins = 100);
new_axes[2,1].set_title("strip_y")
new_fig.tight_layout()

'''GIF STUFF'''
# This creates the gif and saves it
# Use sometime after running the training loop
from PIL import Image
import imageio

# List of JPEG files (make sure they are in order)
jpeg_files = []
for i in range(27):
    jpeg_files.append(f"plots/training_gif_frames/sipm_connected/frame{i}.jpeg")

# Load images
images = [Image.open(f) for f in jpeg_files]

# Save as a GIF
output_gif = "plots/GNN/rmse_sipmconnected.gif"
imageio.mimsave(output_gif, images, format="GIF", duration=5)  # duration in seconds

print(f"GIF saved as {output_gif}")
