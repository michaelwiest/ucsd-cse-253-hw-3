# ucsd-cse-253-hw-3
HW3 for CSE 253 (Winter 2018) at UCSD

## Getting to and activating computing instance
`ssh your_id@ieng6.ucsd.edu`

Put in your password.

To activate the computing environment (i don't get what this does)

`cs253w`

Then depending on whether or not you want to use a GPU.

`launch-pytorch.sh`

or

`launch-pytorch-gpu.sh`

You should now have a Docker container where all of your stuff is accessible.


## Git
You can clone this repo into either the container started from:
`launch-pytorch-gpu.sh`
Or just more generally into your position on the `ieng6` cluster.


## Running scripts
### Problem 1

`python network_1.py`

`python network_2.py`

`python Network_3.py`

These files will produce 2 figures each: percent accuracy and class accuracy.

### Problem 2
'TransferLearningFinal.py'

Performs transfer learning; will produce two plots of accuracy and loss, as well as 4 data files for accuracy and loss of training and testing data

'FeatureExtractionFinal.py'

Performs feature extraction after 3rd and 4th layers; will produce 4 plots of accuracy and loss for each of the 3rd and 4th layers, as well as 8 data files for accuracy and loss of training and testing data

'PlotActivations.py'

Plots 5 image inputs as original images, after 1st conv layer, and after last conv layer. Also plots 1st layer weights. Saves all images to a folder (must change directory to run)

### Credit

Credit to [this repo](https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb)
for the validation data loader.
