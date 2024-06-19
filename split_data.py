import splitfolders

# Define the input folder path and output folder path
input_folder = r"C:\Users\ADMIN\Segmentation1\tmp"
output_folder = "split"

# Call the splitfolders.ratio function
splitfolders.ratio(input_folder, 
                   output=output_folder,
                   seed=1, 
                   ratio=(.8, .2), 
                   group_prefix=None, 
                   move=False)
