# Speaker Verification

> The goal of this project is to create a neural network capable of encoding the speaker's identity into 
an embedding vector, which can be used for speaker verification. Many loss functions can be used (e.g. triplet loss 
[torch.nn.TripletMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html),
Angular Softmax Loss).

## Development setup

To create a virtual environment with all the necessary dependencies, run `make venv`. See [Makefile](Makefile) 
for other helpful targets.

## Dataset

For training and evaluation, we are using the VoxCeleb dataset. This dataset takes around 30GB and can be downloaded
manually using `make download-voxceleb` target. Because it needs to be unzipped first, you should have at least 70GB of 
free space available before downloading. Note that the dataset is downloaded from the unofficial Hugging face 
repo, as it is not possible anymore to download it from [the official website](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) anymore.

For code, see `datasets/` directory.

## License

Files `datasets/voxceleb1_metadata.csv` and `datasets/voxceleb1_veri_test.txt` contain VoxCeleb metadata,
which are licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
