# LiuWang2024
Code used in the manuscript "Flexible gating between subspaces by a disinhibitory motif: a neural network model of internally guided task switching" (https://www.biorxiv.org/content/10.1101/2023.08.15.553375v5)

# Reproducing the results in the paper

## Accessing the pre-trained networks
To start, download the pre-trained networks from ... Note that models starting with '2023-05-01' were trained with sparse SST to E cell connections (c.f. Figure 7), whereas those starting with '2023-05-10' were trained with SST cells fully connected to the E cells.
## Run the trained network on a sequence of trials and save the test data
Next, please run all the blocks in generate_data.ipynb to test the model on a series of trials and save the model activity as well as trial information. This takes about 30 secs per model on an M1 max Macbook.
## Run the analyses
Then, run the corresponding Jupyter notebook to reproduce the figures in the manuscript. Remember to replace the directories for model and test data with yours.
# Software version
This code has been tested on Python 3.10.8 and Pytorch 1.13.1
