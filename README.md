Overview

This project investigates the vulnerability of a deep learning classifier to one-pixel adversarial perturbations and evaluates multiple passive defense techniques designed to improve robustness. The study focuses on a binary Xception-based model trained for casting defect detection.

One-Pixel Attack

The one-pixel attack is an evasion technique where only a single pixel in the input image is modified. Despite being visually imperceptible, such perturbations can cause misclassification due to model sensitivity.

Attack Method

A custom random-search attack is implemented:

A pixel location is sampled uniformly at random.

A random RGB value is assigned to that pixel.

The modified image is evaluated by the classifier.

The procedure repeats for a fixed number of iterations.

An attack is considered successful if the predicted class changes.

This implementation operates independently of gradient information and does not rely on external libraries such as ART.

Defense Techniques

Multiple passive defenses are applied at the input level. These techniques aim to remove, smooth, or neutralize very small perturbations.

1. JPEG Compression

Simulates re-encoding the input image using 60–80% quality to attenuate high-frequency noise.

2. Median and Bilateral Filtering

Median filtering removes localized impulse noise (effective for isolated pixel changes).

Bilateral filtering smooths regions while preserving edges.

3. Bit Depth Reduction (Quantization)

Reduces the image to a lower number of intensity levels, suppressing subtle perturbations.

4. Gaussian Smoothing

Applies mild blur to reduce sensitivity to fine-grained pixel artifacts.

5. Feature Squeezing

Combines quantization and median filtering to remove unnecessary input complexity.

6. Randomized Smoothing with Ensemble Prediction

Randomized preprocessing pipelines are applied repeatedly, and predictions are averaged. This reduces the chance that a single perturbation survives all random transformations. A confidence-drop heuristic is used for adversarial detection.

Robust Model Loading

The model is stored in an HDF5 file generated with a different Keras version. A custom compatibility layer for SeparableConv2D is included to prevent deserialization errors. If full reconstruction fails, the code rebuilds an Xception-like architecture and loads weights by name.

Evaluation Pipeline

Clean images are loaded from the dataset.

The one-pixel attack is applied to generate adversarial variants.

The defense method is applied to both clean and adversarial inputs.

Metrics recorded include:

Attack success before defense

Whether the defense restores the correct label

Whether the defense flags the sample as suspicious

Summary statistics and plots are generated.

Results Summary

The base classifier exhibits substantial vulnerability to one-pixel perturbations.

Median filtering, JPEG compression, and quantization significantly reduce attack effectiveness.

Randomized smoothing with ensemble prediction yields the strongest robustness, frequently restoring correct classifications.

Confidence-based detection identifies many adversarial samples with minimal impact on clean inputs.
