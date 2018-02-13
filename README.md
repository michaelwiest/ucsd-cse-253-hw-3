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


### Credit

Credit to [this repo](https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb)
for the validation data loader.
