import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def create_vgg16_model(input_shape, num_classes):
    """Create VGG16 model without adding custom fully connected layers."""
    base_model = VGG16(weights='imagenet', include_top=True)  # Keep the fully connected layers
    
    # Adjust the last layer to fit the number of classes
    model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.layers[-2].output)
    )
    
    # Compile the model with an appropriate optimizer and loss function
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_vgg16_model(train_dir, val_dir, input_shape, batch_size, epochs, num_classes):
    """Train the VGG16 model with the given data."""
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
    model = create_vgg16_model(input_shape, num_classes)
    
    # Train the model
    model.fit(train_generator, 
              validation_data=val_generator, 
              epochs=epochs, 
              verbose=1)

    return model

# Example usage
train_directory = "./segmented_images_tomato/train/k3"
val_directory = "./segmented_images_tomato/val"
input_shape = (224, 224, 3)
batch_size = 32
epochs = 25
num_classes = 11  # Assuming 3 classes for corn leaf diseases

trained_model = train_vgg16_model(train_directory, val_directory, input_shape, batch_size, epochs, num_classes)

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
test_directory = "./segmented_images_tomato/val"
evaluate_model(trained_model, test_directory, input_shape, batch_size)
