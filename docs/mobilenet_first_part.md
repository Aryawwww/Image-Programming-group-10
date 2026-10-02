# MobileNetV3-Small: Initial Transfer-Learning Stage

## Data and preprocessing

The MobileNetV3-Small pipeline used the same five-class tomato-leaf subset and
fixed directory splits as the custom CNN. The selected classes represented one
healthy tomato-leaf class and four disease classes. Following the custom CNN
preprocessing procedure, 550 valid original images were sampled from each class
using random seed 42. Each class was divided into 400 training images, 75
validation images, and 75 test images. Augmentation was applied only to the
training folders until each class contained 600 training images. The validation
and test folders remained unchanged.

Rather than sampling or splitting the PlantVillage images a second time, the
MobileNet implementation loaded the CNN-generated `train`, `validation`, and
`test` directories directly. This ensured that both neural pipelines used the same
files, labels, and training-only augmented images. All images were decoded as RGB
and resized to 224 x 224 pixels in batches of 32. Training data were shuffled with
seed 42, while validation and test data were kept in deterministic order. The test
split was excluded from training and model selection.

## Frozen-backbone baseline

MobileNetV3-Small was selected because it combines a compact architecture with
features learned from ImageNet and was designed for resource-constrained devices
(Howard et al., 2019). The ImageNet-pretrained convolutional backbone was loaded
without its original classification layer. A task-specific classification head was
then added using global average pooling, dropout with a rate of 0.2, and a dense
softmax layer with five outputs.

During the initial transfer-learning stage, every layer in the pretrained feature
extractor was frozen. Only the new classification head was updated. Freezing the
backbone reduced the number of trainable parameters and provided a controlled
baseline for determining how useful the transferred ImageNet representations were
for tomato-leaf disease classification before partial fine-tuning (Yosinski et al.,
2014).

The model used the preprocessing operation built into TensorFlow's
MobileNetV3-Small implementation, so the directory loader retained pixel values in
their decoded range instead of applying the custom CNN's `1/255` rescaling. This
model-specific numerical preprocessing does not change which images belong to
each split. The classifier was optimised with Adam using an initial learning rate of
0.001 and sparse categorical cross-entropy loss. Accuracy was recorded for the
training and validation sets.

Training was configured for a maximum of 50 epochs. Validation loss was used for
both early stopping and checkpoint selection, with a patience of 10 epochs. The
weights from the epoch with the lowest validation loss were retained in the saved
`.keras` checkpoint. The pipeline also recorded the training duration, class-index
mapping, random seed, hyperparameters, epoch-level loss and accuracy, and the
training and validation learning curves. These artifacts provide the starting point
for the later partial fine-tuning and final evaluation stages.

## Outputs from this stage

The implementation produces the following artifacts under `outputs/frozen`:

- `best_frozen_model.keras`: best validation-loss checkpoint;
- `training_history.csv`: epoch-level training and validation metrics;
- `training_history.png`: accuracy and loss learning curves; and
- `training_config.json`: reproducibility settings and measured training time.

Numerical results should be added to the final report only after the pipeline has
been run in the agreed TensorFlow environment. Test accuracy, macro-averaged
precision, recall and F1-score, the confusion matrix, and robustness experiments
belong to the later shared evaluation stage and are intentionally not reported here.

## Individual contribution statement

Shuhui Wang implemented the initial MobileNetV3-Small transfer-learning pipeline.
This work included aligning the MobileNet input pipeline with the custom CNN's
shared dataset, loading the ImageNet-pretrained architecture, replacing its original
classifier with a five-class head, freezing the feature extractor, implementing
training and validation with early stopping and checkpointing, and recording the
configuration, training time, and learning curves for the frozen-backbone baseline.

## References used in this section

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu,
Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for
MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer
Vision*, 1314-1324.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are
features in deep neural networks? *Advances in Neural Information Processing
Systems, 27*.
