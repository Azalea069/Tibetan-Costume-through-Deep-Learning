import numpy as np # Import Numpy
import pandas as pd # Import Pandas
import os # Import OS to read files
import cv2 # Import OpenCV toolbox
# Randomly display several pictures
import matplotlib.pyplot as plt  # Import matplotlib
# Randomly display several pictures
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font to SimHei
# Path to store the images read

print(os.listdir('C:/Users/20721/Desktop/Tibetans'))

# Path to the corresponding ethnic group
# Path to the corresponding ethnic group, using raw string
adz = r"C:\Users\20721\Desktop\Tibetans\Amdo"
kbz = r"C:\Users\20721\Desktop\Tibetans\Kangba"
wz = r"C:\Users\20721\Desktop\Tibetans\Wei Zang"

# Store image data and labels
X = []
y_label = []
# Image size
imgsize = 150

# Define a function to read images
def training_data(label, data_dir):
    print("Reading:", data_dir)  # Output the reading path
    for img in os.listdir(data_dir):  # Loop through all files in the directory
        path = os.path.join(data_dir, img)  # Join the full path
        print(f"Reading image: {path}")  # Output the path of the image being read

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # Read the image
        if img is None:  # Check if the image is read successfully
            print(f"Failed to read image: {path}, skipping...")
            continue  # If image reading fails, skip this image

        # Check if the image size is invalid
        if img.size == 0:
            print(f"Invalid image size: {path}, skipping...")
            continue  # If the image size is invalid, skip this image

        try:
            img = cv2.resize(img, (imgsize, imgsize))  # Resize the image
        except Exception as e:
            print(f"Failed to resize image: {path}, error: {e}")
            continue  # If resizing fails, skip this image

        X.append(np.array(img))  # Convert the image to numpy format and store in the list
        y_label.append(str(label))  # Store the corresponding label

# Read images from each directory
training_data('Amdo', adz)
training_data('Kangba', kbz)
training_data('Wei Zang', wz)


print("Image reading complete.")

# Randomly display several pictures
fig, ax = plt.subplots(5, 2) # Create a 5x2 image grid
fig.set_size_inches(15, 15) # Set the overall size of the image to 15x15 inches
for i in range(5): # Loop for 5 rows
    for j in range(2): # Loop for 2 columns
        r = np.random.randint(0, len(X)) # Randomly generate an integer within the range of the number of images
        ax[i, j].imshow(cv2.cvtColor(X[r], cv2.COLOR_BGR2RGB)) # Display the randomly selected image and convert the color format from BGR to RGB
        ax[i, j].set_title('Category: ' + y_label[r]) # Set the title of the image
plt.tight_layout() # Adjust the layout of subplots to avoid overlap
plt.show() # Display the image


from sklearn.preprocessing import LabelEncoder # Import label encoding tool
from keras.utils import to_categorical # Import one-hot encoding tool
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label) # Label encoding
y = to_categorical(y,3) # Convert labels to one-hot encoding,
# Call to_categorical to convert y according to 3 categories
X = np.array(X) # Convert X from list to tensor array
X = X/255 # Normalize X
import matplotlib.pyplot as plt # Import matplotlib
import random as rdm # Import random number tool
# Randomly display images of ethnic clothing after normalization
fig,ax=plt.subplots(5,2)# Draw frame
fig.set_size_inches(15,15)# Image size
for i in range(5):# Loop for 5 rows
    for j in range (2):# Loop for two columns
        r=rdm.randint(0,len(X))# Set loop
        ax[i,j].imshow(X[r])# Display image
        ax[i,j].set_title('Category: '+y_label[r])# Title
plt.tight_layout()# Draw

from sklearn.model_selection import train_test_split # Import split tool
# Split training and test sets, ratio 8:2, test set 20%
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=1)
from keras import layers # Import all layers
from keras import models # Import all models
cnn = models.Sequential() # Sequential model
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', # Convolution
                        input_shape=(150, 150, 3)))
cnn.add(layers.MaxPooling2D((2, 2))) # Max pooling
cnn.add(layers.Conv2D(64, (3, 3), activation='relu')) # Convolution
cnn.add(layers.MaxPooling2D((2, 2))) # Max pooling
cnn.add(layers.Conv2D(128, (3, 3), activation='relu')) # Convolution
cnn.add(layers.MaxPooling2D((2, 2))) # Max pooling
cnn.add(layers.Conv2D(128, (3, 3), activation='relu')) # Convolution
cnn.add(layers.MaxPooling2D((2, 2))) # Max pooling
cnn.add(layers.Flatten()) # Flatten
cnn.add(layers.Dense(512, activation='relu')) # Fully connected
cnn.add(layers.Dense(3, activation='softmax')) # Classification output
cnn.compile(loss='categorical_crossentropy', # Loss function
            optimizer='RMSprop', # Optimizer
            metrics=['acc']) # Evaluation metric
# Define a function to display learning curves during training
def show_history(history):
    # Get training loss from training history
    loss = history.history['loss']
    # Get validation loss from training history
    val_loss = history.history['val_loss']
    # Generate sequence of training epochs, starting from 1 to the length of loss list plus 1
    epochs = range(1, len(loss) + 1)
    # Create a figure window, set size to 12x4 inches
    plt.figure(figsize=(12,4))
    # Add first subplot, 1 row 2 columns, first one
    plt.subplot(1, 2, 1)
    # Plot training loss curve
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # Plot validation loss curve
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # Set title of subplot
    plt.title('Training and validation loss')
    # Set x-axis label
    plt.xlabel('Epochs')
    # Set y-axis label
    plt.ylabel('Loss')
    # Show legend
    plt.legend()

    # Get training accuracy from training history
    acc = history.history['acc']
    # Get validation accuracy from training history
    val_acc = history.history['val_acc']
    # Add second subplot, 1 row 2 columns, second one
    plt.subplot(1, 2, 2)
    # Plot training accuracy curve
    plt.plot(epochs, acc, 'bo', label='Training acc')
    # Plot validation accuracy curve
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # Set title of subplot
    plt.title('Training and validation accuracy')
    # Set x-axis label
    plt.xlabel('Epochs')
    # Set y-axis label
    plt.ylabel('Accuracy')
    # Show legend
    plt.legend()
    # Show figure
    plt.show()

# Fit CNN model to training data, set training epochs to 30, validation split to 0.2
history = cnn.fit(X_train, y_train, epochs=100, validation_split=0.2)
# Call show_history function to display learning curves during training
show_history(history)

# Define a function to predict images
def predict_f(image_path):
    # Read image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Resize image to predefined size
    img = cv2.resize(img, (imgsize, imgsize))
    # Convert image to numpy array
    img = np.array(img)
    # Convert image data type to float32 and normalize
    img = img.astype('float32') / 255.0
    # Add a dimension to match the input shape expected by the model (assuming the model expects batched data)
    img = np.expand_dims(img, axis=0)
    # Predict using CNN model
    prediction = cnn.predict(img)
    # Define list of clothing categories
    clothes_classes = ['Amdo', 'Kangba', 'Wei Zang']
    # Get the index of the highest probability category from the prediction result and convert to the corresponding ethnic name
    predicted_class = clothes_classes[np.argmax(prediction)]
    # Return the predicted ethnic category
    return predicted_class

# Randomly select a path to an image and predict its clothing category
image_path = 'C:/Users/20721/Desktop/Tibetans/Amdo/18.webp'
predicted_class = predict_f(image_path)
# Print the predicted clothing category
print('Predicted clothing category:', predicted_class)

# Display the predicted image and its result
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# Convert the image from BGR color space to RGB color space to display correctly in matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Display the image using matplotlib
plt.imshow(img)
# Set the title of the image to the predicted clothing category
plt.title('Predicted Category: ' + predicted_class)
# Turn off axis display
plt.axis('off')
# Show the image
plt.show()