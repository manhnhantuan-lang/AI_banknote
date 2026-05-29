import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

# Config paths and image size
train_dir = 'banknotes_split/train'
validation_dir = 'banknotes_split/valid'
img_width, img_height = 200, 200
batch_size = 32

def preprocess_banknote(image, apply_crop=True):
    if image.max() <= 1.0:
        image = image * 255.0
        
    img_uint8 = image.astype(np.uint8)
    h, w, c = img_uint8.shape
    cropped_done = False
    
    if apply_crop:
        try:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Otsu thresholding
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
            
            contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter banknote contour by size (15% - 85% of image area)
            def get_banknote_contour(contours):
                valid = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if (h * w * 0.15) < area < (h * w * 0.85):
                        valid.append((c, area))
                if valid:
                    return max(valid, key=lambda item: item[1])
                return None, 0
                
            contour1, area1 = get_banknote_contour(contours1)
            contour2, area2 = get_banknote_contour(contours2)
            
            best_contour = None
            best_area = 0
            
            if area1 > 0 or area2 > 0:
                if area1 > area2:
                    best_contour = contour1
                    best_area = area1
                else:
                    best_contour = contour2
                    best_area = area2
            else:
                max_c1 = max(contours1, key=cv2.contourArea) if contours1 else None
                max_c2 = max(contours2, key=cv2.contourArea) if contours2 else None
                area_c1 = cv2.contourArea(max_c1) if max_c1 is not None else 0
                area_c2 = cv2.contourArea(max_c2) if max_c2 is not None else 0
                if area_c1 > area_c2:
                    best_contour = max_c1
                    best_area = area_c1
                else:
                    best_contour = max_c2
                    best_area = area_c2
                    
            if best_contour and (h * w * 0.15) < best_area < (h * w * 0.98):
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [best_contour], -1, 255, -1)
                
                # Check if mask is inverted
                corners_val = (int(mask[0, 0]) + 
                               int(mask[0, w-1]) + 
                               int(mask[h-1, 0]) + 
                               int(mask[h-1, w-1]))
                
                if corners_val > 500:
                    mask = cv2.bitwise_not(mask)
                
                # Dilate mask slightly
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.03) + 1, int(h * 0.03) + 1))
                mask_dilated = cv2.dilate(mask, kernel_dilate)
                
                masked_img = cv2.bitwise_and(img_uint8, img_uint8, mask=mask_dilated)
                
                # Crop bounding box
                x, y, cw, ch = cv2.boundingRect(best_contour)
                pad_y = int(ch * 0.03)
                pad_x = int(cw * 0.03)
                
                ymin = max(0, y - pad_y)
                ymax = min(h, y + ch + pad_y)
                xmin = max(0, x - pad_x)
                xmax = min(w, x + cw + pad_x)
                
                cropped = masked_img[ymin:ymax, xmin:xmax]
                resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                cropped_done = True
        except Exception:
            pass
            
    # Fallback to 15% center crop to minimize background noise
    if not cropped_done:
        if apply_crop:
            masked_img = np.zeros_like(img_uint8)
            ymin, ymax = int(h * 0.15), int(h * 0.85)
            xmin, xmax = int(w * 0.15), int(w * 0.85)
            masked_img[ymin:ymax, xmin:xmax] = img_uint8[ymin:ymax, xmin:xmax]
            
            cropped = masked_img[ymin:ymax, xmin:xmax]
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img_uint8
            
    # LAB CLAHE
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return enhanced.astype(np.float32)

if __name__ == '__main__':
    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.6, 1.4],
        brightness_range=[0.5, 1.5],
        channel_shift_range=5.0,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_banknote
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_banknote
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Custom CNN Architecture
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(1e-4), input_shape=(img_width, img_height, 3)),
        BatchNormalization(momentum=0.9),
        MaxPooling2D(2,2),
        Dropout(0.2),

        Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.9),
        MaxPooling2D(2,2),
        Dropout(0.2),

        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.9),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.9),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.9),
        MaxPooling2D(2,2),
        Dropout(0.3),

        GlobalAveragePooling2D(),

        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(momentum=0.9),
        Dropout(0.5),

        Dense(6, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'banknot_reg.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train model
    epochs = 30
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Plot training results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()

    plt.tight_layout()
    plt.show()
