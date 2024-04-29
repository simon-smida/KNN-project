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
using `make download-voxceleb` target. This will save the dataset into `voxceleb1` directory relative
to project root (the path can be changed by setting `KNN_DATASET_DIR` environment variable).

On Metacentrum, I have pre-downloaded this dataset into my home directory at
`/storage/brno12-cerit/home/tichavskym/voxceleb1`.

## Model evaluation

Model evaluation can be executed using `make evaluate` target. This will launches [evaluate.py](evaluate.py) script,
which evaluates the model on the VoxCeleb test set. Check this script to see which model is being evaluated, and also
if the testing dataset split is not limited to first few samples (which can be done via `FIRST_N` variable - helpful
for local development).

The script will create two files: `scores.txt` with scores for each pair of recordings and `det_curve.png` with DET 
curve. It will also print the EER (Equal Error Rate) value to stdout.

If you want to evaluate (or perform any other computation) on Metacentrum infrastructure, check out scripts 
in [metacentrum/](metacentrum/) directory.

### Results

| Model                             | EER    | minCDF   |
|-----------------------------------|--------|----------|
| speechbrain/spkrec-ecapa-voxceleb | 1.04 % | 0.0036 * |

`*`: this might be wrong, as I'd expect numbers around 0.06

#### DET curve:

![speechbrain/spkrec-ecapa-voxceleb](docs/speechbrain-spkrec-ecapa-voxceleb-det.png)

## Acknowledgements

Computational resources were provided by the e-INFRA CZ project (ID:90254),
supported by the Ministry of Education, Youth and Sports of the Czech Republic.
