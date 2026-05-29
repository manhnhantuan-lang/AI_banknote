import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import load_img
from AI_week3_banknotes import preprocess_banknote

img_width, img_height = 200, 200
train_dir = 'banknotes_split/train'
model_path = 'banknote_reg/banknot_reg.keras'

def predict_real_image(image_path):
    if not os.path.exists(image_path):
        print(f"[!] Test image '{image_path}' not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"[!] Model file '{model_path}' not found.")
        return

    print(f"Loading model and predicting on: {image_path}...")
    model = load_model(model_path)
    
    if not os.path.exists(train_dir):
        print(f"[!] Train directory '{train_dir}' not found.")
        return
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = np.array(img)
    preprocessed = preprocess_banknote(img_array, apply_crop=True)
    img_input = (preprocessed / 255.0)[np.newaxis, ...]
    
    predictions = model.predict(img_input)[0]
    prediction_idx = np.argmax(predictions)
    predicted_name = class_names[prediction_idx]
    confidence = predictions[prediction_idx] * 100
    
    # Format denomination output beautifully (e.g., 500,000 VND)
    try:
        formatted_name = f"{int(predicted_name):,} VND"
    except ValueError:
        formatted_name = predicted_name
        
    print(f"\n=> Prediction: {formatted_name} ({confidence:.2f}% confidence)")
    
    # Thiet ke giao dien GUI cục bộ (Currency & Emerald theme)
    fig = plt.figure(figsize=(14, 6), facecolor='#08100d')

    # Khung anh ben trai
    ax_img = fig.add_axes([0.04, 0.06, 0.42, 0.84])
    ax_img.imshow(preprocessed.astype(np.uint8))
    ax_img.set_title('Anh to polymer quet duoc', color='#ffeaa7', fontsize=12, fontweight='bold', pad=10)
    ax_img.set_facecolor('#111f1a')
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_color('#223d33')
        spine.set_linewidth(1.5)

    # Bieu do xac suat ben phai
    ax_bar = fig.add_axes([0.50, 0.06, 0.44, 0.84])
    
    # Dinh dang ten cac nhan de hien thi tren bieu do
    formatted_labels = []
    for name in class_names:
        try:
            formatted_labels.append(f"{int(name):,} VND")
        except ValueError:
            formatted_labels.append(name)
            
    probs  = [float(p)*100 for p in predictions]
    
    # Sap xep de menh gia cao nhat nam tren cung
    sorted_indices = np.argsort(probs)
    formatted_labels = [formatted_labels[i] for i in sorted_indices]
    probs  = [probs[i] for i in sorted_indices]
    
    # To mau noi bat cho to tien duoc nhan dien
    colors = ['#1e332a'] * len(formatted_labels)
    colors[-1] = '#55efc4' # To tien duoc du doan co mau xanh polymer dac trung

    bars = ax_bar.barh(formatted_labels, probs, color=colors, height=0.45)
    for bar, p in zip(bars, probs):
        if p > 0.05:
            ax_bar.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                        f'{p:.2f}%', va='center', fontsize=9, color='#e0e0ea', fontweight='bold')

    ax_bar.set_facecolor('#111f1a')
    ax_bar.tick_params(colors='#e0e0ea', labelsize=10)
    ax_bar.set_xlim(0, 115)
    ax_bar.set_title(f'Nhan dien: {formatted_name} ({confidence:.2f}%)', color='#ffeaa7', fontsize=13, fontweight='bold', pad=12)
    
    # An khung duong bao quanh bieu do
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.set_xticks([])

    plt.suptitle('Nhan Dien Menh Gia Polymer', fontsize=15, fontweight='bold', color='#ffeaa7', y=0.97)
    plt.show()

if __name__ == '__main__':
    predict_real_image('banknote_reg/50.jpg')

