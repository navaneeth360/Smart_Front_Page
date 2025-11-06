# # train_emnist.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # --- CNN model ---
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)
#         self.fc2 = nn.Linear(128, 47)  # 0–9 + A–Z

#     def forward(self, x):
#         x = nn.ReLU()(self.conv1(x))
#         x = nn.MaxPool2d(2)(x)
#         x = nn.ReLU()(self.conv2(x))
#         x = nn.MaxPool2d(2)(x)
#         x = x.view(-1, 64 * 5 * 5)
#         x = nn.ReLU()(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # --- Preprocessing (matching EMNIST) ---
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# # --- Load EMNIST Balanced Dataset ---
# train_dataset = datasets.EMNIST(
#     root="./data",
#     split="balanced",
#     train=True,
#     download=True,
#     transform=transform
# )
# test_dataset = datasets.EMNIST(
#     root="./data",
#     split="balanced",
#     train=False,
#     download=True,
#     transform=transform
# )

# # --- Dataloaders ---
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# # --- Device setup ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN().to(device)

# # --- Loss and Optimizer ---
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # --- Training loop ---
# epochs = 5
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         # EMNIST images are rotated by 90°, fix that
#         images = images.transpose(2, 3).flip(2)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# # --- Evaluation ---
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         images = images.transpose(2, 3).flip(2)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# # --- Save model ---
# torch.save(model.state_dict(), "emnist_cnn2.pth")
# print("Model saved as emnist_cnn2.pth")
# train_emnist.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
# import src.constants as constants

# 1. Load EMNIST Balanced dataset (47 classes)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features['label'].num_classes  # should be 47

# 2. Preprocess function
def preprocess(image, label):
    # Convert to float and normalize
    image = tf.cast(image, tf.float32) / 255.0

    # EMNIST images are transposed and upside-down by default, fix that:
    image = tf.image.transpose(image)
    image = tf.image.flip_left_right(image)

    # Add channel dimension for Conv2D
    image = tf.expand_dims(image, -1)
    return image, label

# 3. Prepare datasets
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

ds_train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache().shuffle(10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

# 4. Build a CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# 5. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train
model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test
)

# 7. Evaluate
loss, acc = model.evaluate(ds_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# 8. Save the model
model.save('emnist_model.h5')
print(f"💾 Model saved at {'emnist_model.h5'}")
