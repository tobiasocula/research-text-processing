### Neural network for image text processing

An ongoing small research project, in collaboration with one of my professors.
The goal is to train a neutral network for recognizing vertical line fragments given an image containing text.

#### Model experimentations

##### U-net model

My first attempt for the project was to implement a U-net architecture.

The idea here is that the model first abstracts high-level data from the image, and then after passing through the bottleneck builds up the image again, having extracted the patterns of the text.

##### clDice model

I then tried implementing the U-net design again but using a custom loss function based on a paper discussed in this repository:

https://github.com/jacobkoenig/clDice-Loss/blob/master/README.md

This approach aims to more accurately detect thin structures within a given image.

##### Dual U-Net model

This architecture consists of two parallel U-net branches that merge at the bottleneck. The goal is for the model to abstract different types of aspects from the image (one is for a more general high-level overview of the image whilst the other detects more local patterns) and then combine then at the end.

##### Coordinate-based learning model

I then trained a model that learns from the positions of the bar structures, instead of the bar masks themselves.

The model outputs a heatmap with probabilities where it thinks the bars are located. The argmax function can then be used for extracing the most probable bar positions.

