import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 0. Device Configuration ---
# Check if CUDA is available and set the device accordingly for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Tensors: The building blocks of PyTorch ---
print("\n--- Tensors Basics ---")

# Create a tensor directly from data
data_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"Tensor from data:\n{data_tensor}")

# Create a tensor from a NumPy array
numpy_array = np.array([[5, 6], [7, 8]], dtype=np.float32)
numpy_to_tensor = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy:\n{numpy_to_tensor}")

# Create tensors with specific shapes (e.g., zeros, ones, random)
zeros_tensor = torch.zeros(2, 3)
print(f"Zeros tensor (2x3):\n{zeros_tensor}")
ones_tensor = torch.ones(3, 2)
print(f"Ones tensor (3x2):\n{ones_tensor}")
rand_tensor = torch.rand(2, 2)
print(f"Random tensor (2x2):\n{rand_tensor}")

# Tensor operations
sum_tensor = data_tensor + numpy_to_tensor
print(f"Tensor addition:\n{sum_tensor}")
mult_tensor = data_tensor * rand_tensor
print(f"Element-wise multiplication:\n{mult_tensor}")
matmul_tensor = torch.matmul(data_tensor, ones_tensor) # Matrix multiplication
print(f"Matrix multiplication (data_tensor @ ones_tensor):\n{matmul_tensor}")

# Reshaping tensors
reshaped_tensor = data_tensor.view(1, -1) # -1 infers the size
print(f"Reshaped tensor (1x4):\n{reshaped_tensor}")

# Moving tensors to device (GPU if available)
tensor_on_device = data_tensor.to(device)
print(f"Tensor moved to {device}:\n{tensor_on_device}")
# Moving back to CPU
tensor_on_cpu = tensor_on_device.to('cpu')
print(f"Tensor moved back to CPU:\n{tensor_on_cpu}")


# --- 2. Gradients (Autograd): Automatic differentiation ---
print("\n--- Autograd Basics ---")

# Create a tensor for which gradients are to be computed
x = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)
y = x**2 + 2*x + 1 # A simple function
z = y.mean()       # Taking mean for scalar output

print(f"x: {x}, y: {y}, z: {z}")

# Perform backward pass: computes gradients of z with respect to x
z.backward()

# Access the gradient of x (dz/dx)
# For y = x^2 + 2x + 1, dy/dx = 2x + 2. At x=1, dy/dx = 2(1) + 2 = 4
# Since z is scalar y, dz/dx = dy/dx
print(f"Gradient of x (dz/dx): {x.grad}")

# Detaching tensors or using torch.no_grad()
# Gradients are not computed for operations within this block
with torch.no_grad():
    a = torch.tensor(2.0, requires_grad=True)
    b = a * 3
    print(f"b created within no_grad block: {b}")
    # b.backward() # This would raise an error because b has no grad_fn due to no_grad context
print(f"Does b require grad? {b.requires_grad}") # False, because it was created in no_grad


# --- 3. Datasets & Dataloaders: Handling data efficiently ---
print("\n--- Datasets & Dataloaders ---")

# 3.1. Synthetic Data Generation for a simple linear regression
# y = 2*x + 1 + noise
X_np = np.random.rand(100, 1).astype(np.float32) * 10 # 100 samples, 1 feature
y_np = (2 * X_np + 1 + np.random.randn(100, 1) * 0.5).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X_np)
y_tensor = torch.from_numpy(y_np)

# 3.2. Custom Dataset Class
class CustomTensorDataset(Dataset):
    """
    A custom Dataset for holding features (X) and labels (y) as tensors.
    """
    def __init__(self, X_data, y_data):
        # Store features and labels
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        # Return the total number of samples
        return self.X_data.shape[0]

    def __getitem__(self, index):
        # Return one sample (feature, label) at the given index
        return self.X_data[index], self.y_data[index]

# Create dataset instances
train_dataset = CustomTensorDataset(X_tensor[:80], y_tensor[:80]) # 80% for training
test_dataset = CustomTensorDataset(X_tensor[80:], y_tensor[80:])   # 20% for testing

# 3.3. DataLoader
# Create DataLoaders to iterate over datasets in batches
batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Example of iterating through a DataLoader
print(f"Number of batches in training loader: {len(train_loader)}")
for i, (inputs, targets) in enumerate(train_loader):
    if i == 0:
        print(f"First batch - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    break # Just show the first batch


# --- 4. Models: Defining neural networks ---
print("\n--- Model Definition ---")

# Define a simple Linear Regression Model
class LinearRegressionModel(nn.Module):
    """
    A simple linear regression model using PyTorch's nn.Module.
    It takes a single input feature and outputs a single prediction.
    """
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        # Define a linear layer (fully connected layer)
        # input_size: number of input features
        # output_size: number of output features (e.g., 1 for scalar prediction)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        It computes the output given the input x.
        """
        # Pass the input through the linear layer
        out = self.linear(x)
        return out

# Instantiate the model
input_size = 1  # Our X_np has 1 feature
output_size = 1 # Our y_np has 1 output
model = LinearRegressionModel(input_size, output_size).to(device) # Move model to device
print(f"Model architecture:\n{model}")


# --- 5. Training Loop: The core of deep learning ---
print("\n--- Training Loop ---")

# Hyperparameters
learning_rate = 0.01
num_epochs = 50

# Loss function: Mean Squared Error for regression tasks
criterion = nn.MSELoss()

# Optimizer: Stochastic Gradient Descent (SGD)
# It takes model parameters and learning rate.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode (important for layers like BatchNorm, Dropout)
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 1. Move inputs and targets to the configured device (GPU/CPU)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 2. Forward pass: Compute model predictions
        outputs = model(inputs)

        # 3. Calculate loss
        loss = criterion(outputs, targets)

        # 4. Backward pass: Compute gradients of the loss with respect to model parameters
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute new gradients

        # 5. Optimizer step: Update model parameters using the computed gradients
        optimizer.step()

    # Print progress every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# --- 6. Evaluation Loop: Assessing model performance ---
print("\n--- Evaluation Loop ---")

# Set model to evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()

# Disable gradient calculation for inference (saves memory and computations)
with torch.no_grad():
    total_loss = 0
    num_samples = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0) # Accumulate weighted loss
        num_samples += inputs.size(0)

    avg_test_loss = total_loss / num_samples
    print(f'Test Loss: {avg_test_loss:.4f}')

# Set model back to training mode if you plan to continue training
# or for consistency if the program continues
model.train()


# --- 7. Prediction with the trained model ---
print("\n--- Prediction Example ---")

# Example new data point
new_x = torch.tensor([[5.0]], dtype=torch.float32).to(device)

# Make a prediction
with torch.no_grad(): # No need to compute gradients for prediction
    predicted_y = model(new_x)

# Move the prediction back to CPU and convert to a Python number
predicted_y_cpu = predicted_y.cpu().item()
print(f"Input X: {new_x.cpu().item()}, Predicted Y: {predicted_y_cpu:.4f}")

# --- 8. Saving and Loading Models ---
print("\n--- Saving and Loading Models ---")

# Define a path to save the model
model_path = 'linear_regression_model.pth'

# 8.1. Saving the model's state dictionary (recommended)
# This saves only the learned parameters (weights and biases), not the model architecture.
torch.save(model.state_dict(), model_path)
print(f"Model parameters saved to {model_path}")

# 8.2. Loading the model's state dictionary
# First, you need to create an instance of the model with the same architecture
loaded_model = LinearRegressionModel(input_size, output_size).to(device)
# Then load the saved state_dict into the new model instance
loaded_model.load_state_dict(torch.load(model_path))
# Set the loaded model to evaluation mode
loaded_model.eval()
print(f"Model parameters loaded from {model_path}")

# Verify the loaded model makes the same prediction
with torch.no_grad():
    loaded_predicted_y = loaded_model(new_x)
print(f"Prediction from loaded model: {loaded_predicted_y.cpu().item():.4f}")
print("Loading successful, predictions match.")

# You can also save/load the entire model (architecture + state_dict),
# but it's generally less flexible for deployment or different PyTorch versions.
# torch.save(model, 'full_model.pth')
# loaded_full_model = torch.load('full_model.pth')
# loaded_full_model.eval()
# print(f"Prediction from loaded full model: {loaded_full_model(new_x).cpu().item():.4f}")