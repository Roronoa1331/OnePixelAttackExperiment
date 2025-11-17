import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from art.attacks.evasion import OnePixel
from art.estimators.classification import KerasClassifier
import os
from tensorflow.keras.preprocessing import image

def one_pixel_attack_on_model(model_path, data_dir, num_samples=5):
    # Load model
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Create ART classifier
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    
    # Create attack
    attack = OnePixel(classifier=classifier, max_iter=20, num_pixels=1, verbose=False)
    
    # Test on samples from both classes
    classes = ['def_front', 'ok_front']
    results = []
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        image_files = os.listdir(class_dir)[:num_samples]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Get original prediction
                original_pred = model.predict(img_array, verbose=0)[0][0]
                original_class = 'Defective' if original_pred > 0.5 else 'OK'
                
                # Generate adversarial example
                x_adv = attack.generate(x=img_array)
                
                # Get adversarial prediction
                adv_pred = model.predict(x_adv, verbose=0)[0][0]
                adv_class = 'Defective' if adv_pred > 0.5 else 'OK'
                
                # Check if attack was successful
                success = original_class != adv_class
                
                results.append({
                    'image': img_file,
                    'class': class_name,
                    'original_pred': original_pred,
                    'adversarial_pred': adv_pred,
                    'original_class': original_class,
                    'adversarial_class': adv_class,
                    'success': success
                })
                
                print(f\"Image: {img_file} | Original: {original_class} ({original_pred:.3f}) | "
                      f\"Adversarial: {adv_class} ({adv_pred:.3f}) | Success: {success}\")
                      
            except Exception as e:
                print(f\"Error processing {img_file}: {e}\")
    
    # Print summary
    success_rate = sum(1 for r in results if r['success']) / len(results) if results else 0
    print(f\"\\nAttack Success Rate: {success_rate:.2%}\")
    
    return results

# Run the attack
if __name__ == "__main__":
    results = one_pixel_attack_on_model(
        model_path='./xception_transfer_model.h5',
        data_dir='./casting_512x512/',
        num_samples=3
    )