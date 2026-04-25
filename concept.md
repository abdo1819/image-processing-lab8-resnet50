# Lab 8 concepts and required functions

This file summarizes the functions students are expected to complete in the root skeleton files.
The finished reference implementations are available in `solution/inference.py` and `solution/finetune.py`.

## `inference.py`

- `get_device()`
  - Chooses the execution device.
  - Should return `cuda` when a GPU is available, otherwise `cpu`.

- `load_pretrained_model(device)`
  - Loads pretrained ResNet50 ImageNet weights.
  - Should return the model in evaluation mode and the weights metadata object.

- `build_preprocess()`
  - Builds the input transformation pipeline.
  - Should resize, crop, convert to tensor, and normalize images with ImageNet statistics.

- `select_test_images(data_dir, classes, num_images, seed)`
  - Creates the small evaluation set used in Part 1.
  - Should sample a balanced set of image paths from the four class folders.

- `map_imagenet_to_class(name)`
  - Converts one ImageNet class name into one of the lab labels.
  - Should use keyword matching and return `cats`, `dogs`, `horses`, `Humans`, or `unknown`.

- `run_inference(model, preprocess, imagenet_classes, device, test_images)`
  - Runs prediction on the selected images.
  - Should preprocess each image, compute top-5 predictions, map them to the lab classes, and collect detailed output rows.

- `save_confusion_matrix(true_labels, pred_labels, output_dir)`
  - Visualizes prediction quality.
  - Should create and save a heatmap image for the 4-class confusion matrix.

- `save_classification_report(true_labels, pred_labels, detail_rows, output_dir)`
  - Produces the text summary for Part 1.
  - Should save precision, recall, F1-score, and per-image details into a text file.

- `main()`
  - Orchestrates the full workflow.
  - Should call the helper functions in order and write the final outputs into `output/`.

## `finetune.py`

- `SplitDataset`
  - Wraps a subset of `ImageFolder` samples.
  - Should apply one transform for training data and another for validation data while keeping access to the original labels.

- `get_device()`
  - Chooses the training device.
  - Should return `cuda` when available, otherwise `cpu`.

- `build_transforms()`
  - Defines the image pipelines for both splits.
  - Training should include augmentation; validation should stay deterministic.

- `create_dataloaders(data_dir, train_tf, val_tf)`
  - Prepares the dataset and loaders.
  - Should build the `ImageFolder` dataset, split it reproducibly, and return train/validation loaders plus class names.

- `build_model(device)`
  - Prepares the transfer-learning model.
  - Should load ResNet50, freeze the backbone, replace `model.fc`, and move the model to the selected device.

- `train_model(model, train_loader, val_loader, device)`
  - Handles training and validation across epochs.
  - Should define the loss, optimizer, scheduler, history tracking, and best-model selection.

- `save_model(model, output_path)`
  - Stores the best weights.
  - Should save the trained state dictionary into `trained_models/`.

- `save_training_curves(history, output_dir)`
  - Visualizes learning progress.
  - Should save a figure for training/validation loss and accuracy.

- `evaluate_model(model, val_loader, device)`
  - Runs the final validation pass.
  - Should collect all predicted labels and true labels for reporting.

- `save_confusion_matrix(all_labels, all_preds, class_names, output_dir)`
  - Produces the validation confusion matrix.
  - Should save a heatmap for the fine-tuned model.

- `save_classification_report(all_labels, all_preds, class_names, best_acc, output_dir)`
  - Writes the Part 2 metrics to disk.
  - Should save the classification report and include the best validation accuracy.

- `main()`
  - Orchestrates the full fine-tuning pipeline.
  - Should connect data preparation, model creation, training, evaluation, and output saving.
