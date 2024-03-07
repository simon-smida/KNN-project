# Identifikace řečníka

Cílem je vytvořit neuronovou síť schopnou zakódovat identitu řečníka do embedding vektoru, který je možné použít pro identifikaci řečníka z nahrávky jen na základě výpočtu vzdálenosti těchto vektorů. Je možné použít mnoho chybových funkcí (například triplet loss [torch.nn.TripletMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html), Angular Softmax Loss, ...). Key words můžou být "speaker verification".

- http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad
