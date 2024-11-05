import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def create_vgg16_model(input_shape, num_classes):
    """Create VGG16 model with ImageNet weights."""
    # Load the VGG16 model pre-trained on ImageNet, keeping the top layers
    base_model = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Adjust the output layer to fit the number of classes
    x = base_model.layers[-2].output  # Get the output from the second to last layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # New output layer for your classes
    
    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    
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
train_directory = "./segmented_images/train/k2"
val_directory = "./segmented_images/val"
input_shape = (224, 224, 3)  # Input shape for VGG16
batch_size = 32
epochs = 10
num_classes = 3  # Assuming 3 classes for corn leaf diseases

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
test_directory = "./segmented_images/val"
evaluate_model(trained_model, test_directory, input_shape, batch_size)
