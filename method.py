import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_cnn_model(input_shape, num_classes):
    """Create CNN model based on the proposed architecture."""
    model = Sequential()
    
    # Convolutional Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    # Convolutional Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Convolutional Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Convolutional Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Convolutional Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
def train_cnn_model(train_dir, val_dir, input_shape, batch_size, epochs, num_classes):
    """Train the CNN model with the given data."""
    # Data generators for training and validation datasets
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(train_dir, 
                                                  target_size=input_shape[:2], 
                                                  batch_size=batch_size, 
                                                  class_mode='categorical', 
                                                  subset='training')

    val_generator = datagen.flow_from_directory(val_dir, 
                                                target_size=input_shape[:2], 
                                                batch_size=batch_size, 
                                                class_mode='categorical', 
                                                subset='validation')

    # Create and compile the model
    model = create_cnn_model(input_shape, num_classes)
    
    # Train the model
    model.fit(train_generator, 
              validation_data=val_generator, 
              epochs=epochs, 
              verbose=1)

    return model

# Example usage
train_directory = "./segmented_images/k2"
val_directory = "./dataset"
input_shape = (224, 224, 3)
batch_size = 32
epochs = 25
num_classes = 3  # Assuming 3 classes for corn leaf diseases

trained_model = train_cnn_model(train_directory, val_directory, input_shape, batch_size, epochs, num_classes)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_directory, input_shape, batch_size):
    """Evaluate the trained model and print classification report and confusion matrix."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(test_directory, 
                                                      target_size=input_shape[:2], 
                                                      batch_size=batch_size, 
                                                      class_mode='categorical', 
                                                      shuffle=False)

    # Predict the classes
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Print the classification report and confusion matrix
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    print("Confusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

# Example usage
test_directory = "./dataset"
evaluate_model(trained_model, test_directory, input_shape, batch_size)
