# MobileNetV3-Small Plant Disease Classification

## Task

This branch contains the MobileNetV3-Small pipeline for Assessment Task 2 of 42177 Image Processing and Pattern Recognition. The pipeline will classify tomato leaf images from the PlantVillage dataset into five selected classes using ImageNet transfer learning.

The MobileNet implementation must use the same dataset subset, leaf-grouped data split, random seed, evaluation metrics, and robustness-test settings as the HOG-SVM and custom CNN pipelines so that the three approaches can be compared fairly.

## Dataset and Experimental Protocol

- Dataset: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)
- Configuration: original RGB images (`color`)
- Scope: one healthy tomato class and four tomato disease classes
- Shared source subset: 550 original images per class
- Split before augmentation: 400 training, 75 validation, and 75 testing images per class
- Training set after CNN preprocessing: 600 images per class, including training-only augmented images
- Split implementation: reuse the Custom CNN's generated `train`, `validation`, and `test` folders directly
- Random seed: `42`
- Data augmentation: training split only
- Model selection: validation split only
- Final evaluation: untouched test split

The final model will be evaluated with accuracy, macro precision, macro recall, macro F1-score, per-class metrics, a confusion matrix, training time, and inference time. Robustness will also be tested using Gaussian blur, Gaussian noise, and brightness-adjusted versions of the test images.

## Current Progress

- [x] Confirmed the PlantVillage data source
- [x] Selected MobileNetV3-Small as the transfer-learning architecture
- [x] Reused the Custom CNN sampling and train/validation/test procedure
- [x] Defined the responsibilities of the two MobileNet team members
- [x] Created the `mobileNet` development branch
- [x] Confirmed the five tomato classes used by the Custom CNN pipeline
- [x] Matched the Custom CNN's 550-original-image-per-class sampling protocol
- [x] Reused the Custom CNN's generated train, validation, and test folders
- [x] Kept augmentation restricted to the shared training folder
- [x] Implement frozen-backbone training
- [ ] Implement partial fine-tuning
- [ ] Implement the shared evaluation functions
- [ ] Run the original and robustness test sets
- [ ] Generate figures, tables, and report material

No model-training results are available yet. Results will be added after the MobileNetV3-Small training environment and model settings have been finalised.

## Frozen-backbone baseline

The baseline reads the exact directory dataset produced by the Custom CNN's
`subset_preprocess.py`. This avoids creating a second random split and includes
the same saved training-only augmented images. MobileNetV3-Small's built-in
preprocessing remains enabled, the complete ImageNet feature extractor is frozen,
and only a global-average-pooling/dropout/five-class head is trained. The shared
dataset is stored in this repository under `dataset/` with `train`, `validation`,
and `test` folders. Run from the project directory:

```powershell
python train_frozen.py
```

Use `python train_frozen.py --help` to override the dataset, output, or training
settings. If `--dataset-dir` is omitted, it defaults to the repository's `dataset`
folder. Artifacts are written to `outputs/frozen`: the best model,
training history, learning curves, and a JSON record of the configuration and
training time. The test dataset is loaded only to verify the shared class mapping;
it is not passed to model training or checkpoint selection.

## Team Responsibilities

### Shuhui Wang

Shuhui is responsible for the initial MobileNetV3-Small transfer-learning pipeline:

- Load and preprocess the shared PlantVillage subset
- Apply augmentation to the training split only
- Load the ImageNet-pretrained MobileNetV3-Small model
- Replace the original classifier with a five-class classification head
- Freeze the feature extractor and train the new classification head
- Implement training and validation loops, checkpointing, and early stopping
- Save the best frozen-backbone model
- Produce the initial training and validation accuracy/loss curves
- Record the initial training configuration and training time

### Alicia Wee

Alicia is responsible for fine-tuning, final evaluation, and result analysis:

- Load the best frozen-backbone checkpoint
- Unfreeze a limited number of final feature blocks
- Fine-tune the model using a smaller learning rate
- Select the final model using validation macro F1-score
- Calculate overall and per-class evaluation metrics
- Generate the confusion matrix
- Measure training and inference time
- Evaluate the final model on Gaussian blur, Gaussian noise, and brightness-adjusted test sets
- Analyse misclassifications, robustness, computational cost, and the effect of fine-tuning

### Joint Responsibilities

Shuhui and Alicia will jointly:

- Verify that the implementation uses the shared dataset split and experimental protocol
- Review the training and evaluation code
- Compare frozen-backbone and fine-tuned performance
- Validate the final figures and result tables
- Document the MobileNet methodology, results, discussion, and code appendix

## Planned Outputs

- Reproducible MobileNetV3-Small training code
- Saved best-model checkpoint
- Training and validation learning curves
- Test-set classification metrics
- Confusion matrix
- Robustness comparison table
- Training and inference time measurements
- Clearly commented code for the report appendix

## Reproducibility Rules

1. Do not create an independent random split for MobileNet.
2. Reuse the Custom CNN's generated split folders and files directly.
3. Do not augment validation or test images.
4. Do not use test results to select hyperparameters or checkpoints.
5. Run robustness testing only after the final model configuration has been selected.
6. Record all final parameters, random seeds, and software dependencies.
