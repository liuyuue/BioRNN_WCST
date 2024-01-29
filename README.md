Code used in the manuscript "Flexible gating between subspaces by a disinhibitory motif: a neural network model of internally guided task switching" (https://www.biorxiv.org/content/10.1101/2023.08.15.553375v5)

# Reproducing the results in the paper

## Accessing the pre-trained networks
To start, download the pre-trained networks from https://drive.google.com/drive/folders/17LqvmcBynX0a4OtUgnkODGJKK_JFRB2Q?usp=drive_link. Note that models whose name starting with '2023-05-01' were trained with sparse SST to E cell connections (c.f. Figure 7), whereas those starting with '2023-05-10' were trained with SST cells fully connected to the E cells.
## Run the trained network on a sequence of trials and save the test data
Next, please run all the blocks in generate_data.ipynb to test the model on a series of trials and save the model activity as well as trial information. This takes about 30 secs per model on an M1 max Macbook.
## Run the analyses
Then, run the corresponding Jupyter notebook to reproduce the figures in the manuscript. Remember to replace the directories for model and test data with your own.
# Software version
This code has been tested on Python 3.10.8 and Pytorch 1.13.1
# License information
Copyright 2024 Yue Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
