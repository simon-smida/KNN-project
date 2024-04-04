# Speaker Verification

> The goal of this project is to create a neural network capable of encoding the speaker's identity into 
an embedding vector, which can be used for speaker verification. Many loss functions can be used (e.g. triplet loss 
[torch.nn.TripletMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html),
Angular Softmax Loss).

## Development setup

To create a virtual environment with all the necessary dependencies, run `make venv`. To use it,
run `source venv/bin/activate`.

For other helpful targets (such as formatting, launching scripts), see [Makefile](Makefile).

## Dataset

For training and evaluation, we are using the VoxCeleb dataset. This dataset takes around 30GB and can be downloaded
manually using `make download-voxceleb` target. This will save the dataset into `voxceleb1` directory relative 
to project root. Because it needs to be unzipped first, you should have at least 70GB of free space available before 
downloading. Note that the dataset is downloaded from the unofficial Hugging face repo, as it is not possible anymore 
to download it from [the official website](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) anymore.

For code, see `datasets/` directory.

## Model evaluation

Model evaluation can be executed using `make evaluate` target. This will launches [evaluate.py](evaluate.py) script,
which evaluates the model on the VoxCeleb test set. Check this script to see which model is being evaluated, and also
if the testing dataset split is not limited to first few samples (which can be done via `FIRST_N` variable - helpful
for local development).

The script will create two files: `scores.txt` with scores for each pair of recordings and `det_curve.png` with DET 
curve. It will also print the EER (Equal Error Rate) value to stdout.

## License

Files `datasets/voxceleb1_metadata.csv` and `datasets/voxceleb1_veri_test.txt` contain VoxCeleb metadata,
which are licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.

The rest of the code is licensed under MIT license, if not in collision with the VoxCeleb licensing.
