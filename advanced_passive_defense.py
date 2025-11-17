import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from scipy import ndimage
from skimage import restoration, filters
import os
import random
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedPassiveDefense:
    """Advanced passive defense methods using various techniques"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_image(self, image):
        """Get model prediction for an image"""
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0, predicted_class].item()
        return predicted_class.item(), confidence
    
    def randomized_smoothing(self, image, methods: List[str] = None, num_samples: int = 5):
        """Apply randomized smoothing defense by combining multiple methods"""
        if methods is None:
            methods = ['gaussian', 'median', 'bilateral']
        
        # Randomly select a defense method
        selected_method = random.choice(methods)
        
        # Apply with random parameters
        if selected_method == 'gaussian':
            sigma = random.uniform(0.5, 2.0)
            return self.gaussian_filter_defense(image, sigma=sigma)
        elif selected_method == 'median':
            kernel_size = random.choice([3, 5])
            return self.median_filter_defense(image, kernel_size=kernel_size)
        elif selected_method == 'bilateral':
            d = random.choice([5, 9, 15])
            sigma_color = random.uniform(50, 100)
            sigma_space = random.uniform(50, 100)
            return self.bilateral_filter_defense(image, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
        else:
            return image
    
    def wavelet_denoising_defense(self, image, wavelet='db4', sigma=0.1):
        """Apply wavelet denoising defense"""
        try:
            import pywt
        except ImportError:
            print("PyWavelets not installed. Using Gaussian filter instead.")
            return self.gaussian_filter_defense(image, sigma=1.0)
        
        # Convert tensor to numpy
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Apply wavelet denoising to each channel
        filtered_np = np.zeros_like(image_np)
        for c in range(3):
            # Wavelet transform
            coeffs = pywt.wavedec2(image_np[:, :, c], wavelet, level=2)
            
            # Threshold coefficients (VisuShrink)
            sigma_est = self.estimate_sigma(coeffs[-1])
            threshold = sigma_est * np.sqrt(2 * np.log(image_np[:, :, c].size))
            
            # Apply soft thresholding
            coeffs_thresh = [coeffs[0]]  # Keep approximation
            for detail in coeffs[1:]:
                coeffs_thresh.append(
                    tuple(pywt.threshold(c, threshold, mode='soft') for c in detail)
                )
            
            # Inverse wavelet transform
            filtered_np[:, :, c] = pywt.waverec2(coeffs_thresh, wavelet)
        
        # Ensure same shape
        if filtered_np.shape != image_np.shape:
            filtered_np = filtered_np[:image_np.shape[0], :image_np.shape[1], :]
        
        # Convert back to tensor
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def estimate_sigma(self, coeffs):
        """Estimate noise standard deviation from wavelet coefficients"""
        if isinstance(coeffs, tuple):
            # For detail coefficients
            detail_coeffs = np.concatenate([c.flatten() for c in coeffs])
            return np.median(np.abs(detail_coeffs)) / 0.6745
        else:
            return np.median(np.abs(coeffs)) / 0.6745
    
    def jpeg_compression_defense(self, image, quality=75):
        """Apply JPEG compression defense"""
        # Convert tensor to PIL Image
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Save as JPEG and reload (simulating compression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pil_image.save('temp_jpeg.jpg', 'JPEG', quality=quality)
            compressed_image = Image.open('temp_jpeg.jpg')
        
        # Clean up
        if os.path.exists('temp_jpeg.jpg'):
            os.remove('temp_jpeg.jpg')
        
        # Convert back to tensor
        compressed_np = np.array(compressed_image).astype(np.float32) / 255.0
        compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1).unsqueeze(0).float()
        return compressed_tensor.to(self.device)
    
    def quantization_defense(self, image, levels=8):
        """Apply quantization defense"""
        # Convert tensor to numpy
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Apply quantization
        quantized_np = np.round(image_np * (levels - 1)) / (levels - 1)
        
        # Convert back to tensor
        quantized_tensor = torch.from_numpy(quantized_np).permute(2, 0, 1).unsqueeze(0).float()
        return quantized_tensor.to(self.device)
    
    def feature_squeezing_defense(self, image, methods: List[str] = None):
        """Apply feature squeezing with multiple methods"""
        if methods is None:
            methods = ['quantization', 'smoothing']
        
        squeezed_image = image.clone()
        
        for method in methods:
            if method == 'quantization':
                squeezed_image = self.quantization_defense(squeezed_image, levels=8)
            elif method == 'smoothing':
                squeezed_image = self.gaussian_filter_defense(squeezed_image, sigma=0.5)
        
        return squeezed_image
    
    def geometric_transformation_defense(self, image, max_rotation=5, max_shift=2):
        """Apply random geometric transformations"""
        # Convert tensor to numpy
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Random rotation
        angle = random.uniform(-max_rotation, max_rotation)
        rotated = ndimage.rotate(image_np, angle, reshape=False, mode='reflect')
        
        # Random shift
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        shifted = ndimage.shift(rotated, (shift_y, shift_x, 0), mode='reflect')
        
        # Convert back to tensor
        transformed_tensor = torch.from_numpy(shifted).permute(2, 0, 1).unsqueeze(0).float()
        return transformed_tensor.to(self.device)
    
    def adaptive_smoothing_defense(self, image, edge_preserving=True):
        """Apply adaptive smoothing based on local image characteristics"""
        if edge_preserving:
            # Use bilateral filter for edge preservation
            return self.bilateral_filter_defense(image, d=9, sigma_color=75, sigma_space=75)
        else:
            # Use anisotropic diffusion
            return self.anisotropic_diffusion_defense(image)
    
    def anisotropic_diffusion_defense(self, image, iterations=10, delta=0.14, kappa=50):
        """Apply Perona-Malik anisotropic diffusion"""
        # Convert tensor to numpy
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Apply anisotropic diffusion to each channel
        filtered_np = np.zeros_like(image_np)
        for c in range(3):
            filtered_np[:, :, c] = filters.rank.median(image_np[:, :, c], 
                                                     np.ones((3, 3)))  # Simple alternative
        
        # Convert back to tensor
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def ensemble_defense(self, image, methods: List[str] = None):
        """Apply ensemble of defense methods"""
        if methods is None:
            methods = ['gaussian', 'median', 'jpeg']
        
        # Apply each method and average results
        defended_images = []
        for method in methods:
            if method == 'gaussian':
                defended_images.append(self.gaussian_filter_defense(image, sigma=1.0))
            elif method == 'median':
                defended_images.append(self.median_filter_defense(image, kernel_size=3))
            elif method == 'jpeg':
                defended_images.append(self.jpeg_compression_defense(image, quality=75))
            elif method == 'bilateral':
                defended_images.append(self.bilateral_filter_defense(image))
        
        # Average the results
        ensemble_result = torch.mean(torch.stack(defended_images), dim=0)
        return ensemble_result

    # Keep original methods for compatibility
    def gaussian_filter_defense(self, image, sigma=1.0):
        """Apply Gaussian filtering defense"""
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        filtered_np = np.zeros_like(image_np)
        for c in range(3):
            filtered_np[:, :, c] = ndimage.gaussian_filter(image_np[:, :, c], sigma=sigma)
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def median_filter_defense(self, image, kernel_size=3):
        """Apply median filtering defense"""
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        filtered_np = cv2.medianBlur(image_np, kernel_size)
        filtered_np = filtered_np.astype(np.float32) / 255.0
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def bilateral_filter_defense(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filtering defense"""
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        filtered_np = cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)
        filtered_np = filtered_np.astype(np.float32) / 255.0
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def non_local_means_defense(self, image, h=10, template_window_size=7, search_window_size=21):
        """Apply Non-local Means denoising defense"""
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        filtered_np = cv2.fastNlMeansDenoisingColored(
            image_np, None, h, h, template_window_size, search_window_size
        )
        filtered_np = filtered_np.astype(np.float32) / 255.0
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def total_variation_defense(self, image, weight=0.1, num_iterations=10):
        """Apply Total Variation denoising defense"""
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        filtered_np = np.zeros_like(image_np)
        for c in range(3):
            filtered_np[:, :, c] = restoration.denoise_tv_chambolle(
                image_np[:, :, c], weight=weight, max_num_iter=num_iterations
            )
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered_tensor.to(self.device)
    
    def apply_defense(self, image, method, **kwargs):
        """Apply specified defense method"""
        defense_methods = {
            'gaussian': self.gaussian_filter_defense,
            'median': self.median_filter_defense,
            'bilateral': self.bilateral_filter_defense,
            'nlm': self.non_local_means_defense,
            'tv': self.total_variation_defense,
            'wavelet': self.wavelet_denoising_defense,
            'jpeg': self.jpeg_compression_defense,
            'quantization': self.quantization_defense,
            'feature_squeezing': self.feature_squeezing_defense,
            'geometric': self.geometric_transformation_defense,
            'adaptive': self.adaptive_smoothing_defense,
            'randomized': self.randomized_smoothing,
            'ensemble': self.ensemble_defense
        }
        
        if method in defense_methods:
            return defense_methods[method](image, **kwargs)
        else:
            raise ValueError(f"Unknown defense method: {method}")

def load_adversarial_examples():
    """Load adversarial examples from previous experiments"""
    print("Loading target model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from torchvision import models
    import torch.nn as nn
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    try:
        model.load_state_dict(torch.load('best_transfer_model.pth', map_location=device))
        print("Loaded target model weights from best_transfer_model.pth")
    except:
        print("Using randomly initialized target model")

    model.to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder("./casting_512x512/", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, val_size])
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Generate adversarial examples using multiple attack methods
    adversarial_examples = []
    clean_examples = []
    
    print("Generating adversarial examples...")
    for i, (image, label) in enumerate(test_loader):
        if i >= 15:  # Generate more examples for better evaluation
            break
        
        image = image.to(device)
        clean_examples.append((image.detach(), label.item()))
        
        # Generate multiple types of adversarial examples
        adv_images = generate_multiple_attacks(model, image, label, device)
        adversarial_examples.extend(adv_images)
    
    return adversarial_examples, clean_examples, model, device

def generate_multiple_attacks(model, image, label, device, epsilon=0.1):
    """Generate multiple types of adversarial attacks"""
    adversarial_examples = []
    
    # FGSM Attack
    adv_fgsm = fgsm_attack(model, image, label, epsilon, device)
    adversarial_examples.append((adv_fgsm, label.item()))
    
    # PGD Attack
    adv_pgd = pgd_attack(model, image, label, epsilon=epsilon, alpha=0.01, num_iter=10, device=device)
    adversarial_examples.append((adv_pgd, label.item()))
    
    # Simple noise attack
    noise = torch.randn_like(image) * 0.1
    adv_noise = torch.clamp(image + noise, 0, 1)
    adversarial_examples.append((adv_noise, label.item()))
    
    return adversarial_examples

def fgsm_attack(model, image, label, epsilon, device):
    """Fast Gradient Sign Method attack"""
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label.to(device))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image.detach()

def pgd_attack(model, image, label, epsilon=0.1, alpha=0.01, num_iter=10, device='cpu'):
    """Projected Gradient Descent attack"""
    original_image = image.data
    
    for i in range(num_iter):
        image.requires_grad = True
        output = model(image)
        loss = F.cross_entropy(output, label.to(device))
        model.zero_grad()
        loss.backward()
        
        # Update image
        adv_image = image + alpha * image.grad.sign()
        eta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, 0, 1).detach()
    
    return image

def evaluate_defense_robustness(defense, adversarial_examples, clean_examples, defense_methods):
    """Evaluate the robustness of different defense methods"""
    results = {}

    print("Evaluating defense methods...")
    print("=" * 60)

    for method_name, method_params in defense_methods.items():
        print(f"\nTesting {method_name} defense...")

        # Test on adversarial examples
        adv_correct = 0
        adv_total = len(adversarial_examples)

        # Test on clean examples
        clean_correct = 0
        clean_total = len(clean_examples)
        
        # Track confidence scores
        adv_confidences = []
        clean_confidences = []

        for adv_img, true_label in adversarial_examples:
            defended_img = defense.apply_defense(adv_img, method_params['method'], **method_params['params'])
            pred_class, confidence = defense.predict_image(defended_img)
            adv_confidences.append(confidence)
            
            if pred_class == true_label:
                adv_correct += 1

        for clean_img, true_label in clean_examples:
            defended_img = defense.apply_defense(clean_img, method_params['method'], **method_params['params'])
            pred_class, confidence = defense.predict_image(defended_img)
            clean_confidences.append(confidence)
            
            if pred_class == true_label:
                clean_correct += 1

        # Calculate metrics
        adv_accuracy = adv_correct / adv_total
        clean_accuracy = clean_correct / clean_total
        avg_adv_confidence = np.mean(adv_confidences)
        avg_clean_confidence = np.mean(clean_confidences)

        results[method_name] = {
            'adversarial_accuracy': adv_accuracy,
            'clean_accuracy': clean_accuracy,
            'avg_adversarial_confidence': avg_adv_confidence,
            'avg_clean_confidence': avg_clean_confidence,
            'defense_gap': clean_accuracy - adv_accuracy,  # Smaller is better
            'robustness_score': (adv_accuracy + clean_accuracy) / 2  # Overall performance
        }

        print(f"  Adversarial accuracy: {adv_accuracy:.3f} ({adv_correct}/{adv_total})")
        print(f"  Clean accuracy: {clean_accuracy:.3f} ({clean_correct}/{clean_total})")
        print(f"  Avg adversarial confidence: {avg_adv_confidence:.3f}")
        print(f"  Avg clean confidence: {avg_clean_confidence:.3f}")
        print(f"  Defense gap: {results[method_name]['defense_gap']:.3f}")

    return results

def plot_advanced_comparison(results):
    """Plot advanced comparison of defense methods"""
    methods = list(results.keys())
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy comparison
    adv_acc = [results[m]['adversarial_accuracy'] for m in methods]
    clean_acc = [results[m]['clean_accuracy'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, adv_acc, width, label='Adversarial', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width/2, clean_acc, width, label='Clean', color='blue', alpha=0.7)
    ax1.set_xlabel('Defense Methods')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence scores
    adv_conf = [results[m]['avg_adversarial_confidence'] for m in methods]
    clean_conf = [results[m]['avg_clean_confidence'] for m in methods]
    
    ax2.plot(methods, adv_conf, 'ro-', label='Adversarial Confidence', linewidth=2, markersize=8)
    ax2.plot(methods, clean_conf, 'bo-', label='Clean Confidence', linewidth=2, markersize=8)
    ax2.set_xlabel('Defense Methods')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Prediction Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Defense gap (smaller is better)
    defense_gaps = [results[m]['defense_gap'] for m in methods]
    ax3.bar(methods, defense_gaps, color='green', alpha=0.7)
    ax3.set_xlabel('Defense Methods')
    ax3.set_ylabel('Clean - Adversarial Accuracy')
    ax3.set_title('Defense Gap (Smaller is Better)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Robustness score
    robustness_scores = [results[m]['robustness_score'] for m in methods]
    ax4.bar(methods, robustness_scores, color='purple', alpha=0.7)
    ax4.set_xlabel('Defense Methods')
    ax4.set_ylabel('Robustness Score')
    ax4.set_title('Overall Robustness (Higher is Better)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_11_advanced_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run advanced passive defense experiment"""
    print("Experiment 11: Advanced Passive Defense Against Adversarial Attacks")
    print("=" * 70)

    # Load adversarial examples and model
    print("Loading adversarial examples and model...")
    adversarial_examples, clean_examples, model, device = load_adversarial_examples()

    # Initialize advanced defense system
    defense = AdvancedPassiveDefense(model, device)

    # Define advanced defense methods to test
    defense_methods = {
        'Randomized Smoothing': {
            'method': 'randomized',
            'params': {'methods': ['gaussian', 'median', 'bilateral'], 'num_samples': 5}
        },
        'Wavelet Denoising': {
            'method': 'wavelet',
            'params': {'wavelet': 'db4', 'sigma': 0.1}
        },
        'JPEG Compression': {
            'method': 'jpeg',
            'params': {'quality': 75}
        },
        'Feature Squeezing': {
            'method': 'feature_squeezing',
            'params': {'methods': ['quantization', 'smoothing']}
        },
        'Geometric Transformation': {
            'method': 'geometric',
            'params': {'max_rotation': 5, 'max_shift': 2}
        },
        'Adaptive Smoothing': {
            'method': 'adaptive',
            'params': {'edge_preserving': True}
        },
        'Ensemble Defense': {
            'method': 'ensemble',
            'params': {'methods': ['gaussian', 'median', 'jpeg']}
        },
        'Bilateral Filter': {
            'method': 'bilateral',
            'params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}
        }
    }

    # Test baseline
    print("\nTesting baseline (no defense)...")
    adv_correct_baseline = 0
    clean_correct_baseline = 0
    adv_confidences_baseline = []
    clean_confidences_baseline = []

    for adv_img, true_label in adversarial_examples:
        pred_class, confidence = defense.predict_image(adv_img)
        adv_confidences_baseline.append(confidence)
        if pred_class == true_label:
            adv_correct_baseline += 1

    for clean_img, true_label in clean_examples:
        pred_class, confidence = defense.predict_image(clean_img)
        clean_confidences_baseline.append(confidence)
        if pred_class == true_label:
            clean_correct_baseline += 1

    baseline_adv_acc = adv_correct_baseline / len(adversarial_examples)
    baseline_clean_acc = clean_correct_baseline / len(clean_examples)
    baseline_adv_conf = np.mean(adv_confidences_baseline)
    baseline_clean_conf = np.mean(clean_confidences_baseline)

    print(f"Baseline adversarial accuracy: {baseline_adv_acc:.3f}")
    print(f"Baseline clean accuracy: {baseline_clean_acc:.3f}")
    print(f"Baseline adversarial confidence: {baseline_adv_conf:.3f}")
    print(f"Baseline clean confidence: {baseline_clean_conf:.3f}")

    # Evaluate defense methods
    results = evaluate_defense_robustness(defense, adversarial_examples, clean_examples, defense_methods)

    # Add baseline to results
    results['No Defense'] = {
        'adversarial_accuracy': baseline_adv_acc,
        'clean_accuracy': baseline_clean_acc,
        'avg_adversarial_confidence': baseline_adv_conf,
        'avg_clean_confidence': baseline_clean_conf,
        'defense_gap': baseline_clean_acc - baseline_adv_acc,
        'robustness_score': (baseline_adv_acc + baseline_clean_acc) / 2
    }

    # Generate visualizations
    print("\nGenerating advanced visualizations...")
    plot_advanced_comparison(results)

    # Print comprehensive analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE DEFENSE ANALYSIS")
    print("="*70)

    # Find best methods for different criteria
    best_defense = max(results.keys(), key=lambda x: results[x]['adversarial_accuracy'])
    best_preservation = max(results.keys(), key=lambda x: results[x]['clean_accuracy'])
    best_robustness = max(results.keys(), key=lambda x: results[x]['robustness_score'])
    smallest_gap = min(results.keys(), key=lambda x: results[x]['defense_gap'])

    print(f"\n🏆 Best Defense Against Adversarial Attacks: {best_defense}")
    print(f"   Adversarial Accuracy: {results[best_defense]['adversarial_accuracy']:.3f}")
    print(f"   Clean Accuracy: {results[best_defense]['clean_accuracy']:.3f}")
    print(f"   Robustness Score: {results[best_defense]['robustness_score']:.3f}")

    print(f"\n🛡️  Best Clean Image Preservation: {best_preservation}")
    print(f"   Clean Accuracy: {results[best_preservation]['clean_accuracy']:.3f}")
    print(f"   Adversarial Accuracy: {results[best_preservation]['adversarial_accuracy']:.3f}")

    print(f"\n⚖️  Most Balanced Defense (Smallest Gap): {smallest_gap}")
    print(f"   Defense Gap: {results[smallest_gap]['defense_gap']:.3f}")
    print(f"   Clean Accuracy: {results[smallest_gap]['clean_accuracy']:.3f}")
    print(f"   Adversarial Accuracy: {results[smallest_gap]['adversarial_accuracy']:.3f}")

    print(f"\n🌟 Overall Most Robust: {best_robustness}")
    print(f"   Robustness Score: {results[best_robustness]['robustness_score']:.3f}")

    # Improvement analysis
    print(f"\n📈 Improvement Analysis (vs No Defense):")
    print("-" * 50)
    for method, result in results.items():
        if method != 'No Defense':
            adv_improvement = result['adversarial_accuracy'] - baseline_adv_acc
            clean_change = result['clean_accuracy'] - baseline_clean_acc
            robustness_improvement = result['robustness_score'] - results['No Defense']['robustness_score']
            
            print(f"\n  {method}:")
            print(f"    ✓ Adversarial improvement: {adv_improvement:+.3f}")
            print(f"    ✓ Clean accuracy change: {clean_change:+.3f}")
            print(f"    ✓ Robustness improvement: {robustness_improvement:+.3f}")
            print(f"    ✓ Defense gap: {result['defense_gap']:.3f}")

    print("\n💡 Recommendations:")
    if results[best_robustness]['robustness_score'] > 0.7:
        print("  ✅ Excellent robustness achieved! Consider deploying in production.")
    elif results[best_robustness]['robustness_score'] > 0.5:
        print("  ⚠️  Moderate robustness. Consider combining multiple defenses.")
    else:
        print("  ❌ Low robustness. Explore more advanced defense strategies.")

    print("\n📊 Files generated:")
    print("  - experiment_11_advanced_comparison.png")

if __name__ == "__main__":
    main()