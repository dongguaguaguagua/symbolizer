import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from cnn import SimpleCNN, unpickle, get_loader, get_accuracy
from resnet import HASYv2ResNet, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and process the data
print("loading data...")
HASYv2 = unpickle("./data/HASYv2")
data = np.array(HASYv2["data"])
labels = np.array(HASYv2["labels"])

print("processing data...")
train_loader, test_loader = get_loader(data, labels)

num_classes = len(np.unique(labels)) + 1

# Load the teacher model
teacher_model = HASYv2ResNet(num_classes)
teacher_model.load_state_dict(torch.load("./models/resnet_20241109_132056.pth"))
teacher_model.eval()  # Set teacher to evaluation mode
teacher_model.to(device)

# Define SimpleCNN model as student
student_model = SimpleCNN(num_classes)
student_model.to(device)


# Distillation loss function
def distillation_loss(
    student_logits, teacher_logits, labels, alpha=0.5, temperature=3.0
):
    # Cross-entropy with soft labels from teacher
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)

    # Cross-entropy with true labels
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss


# Optimizer for the student model
optimizer = optim.Adam(student_model.parameters(), lr=0.001)


# Training function for distillation
def distill_student_model(
    epoch, student_model, teacher_model, train_loader, optimizer, num_epochs=10
):
    student_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
    ):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        student_outputs = student_model(inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Calculate distillation loss
        loss = distillation_loss(student_outputs, teacher_outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
    )


# Training loop for distillation
num_epochs = 10
for epoch in range(num_epochs):
    distill_student_model(
        epoch,
        student_model,
        teacher_model,
        train_loader,
        optimizer,
        num_epochs=num_epochs,
    )
    test_model(student_model, test_loader)

model_directory = "./models"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Save the distilled student model
distilled_model_filename = f"simpleCNN_distilled_{current_time}.pth"
distilled_model_path = os.path.join(model_directory, distilled_model_filename)
torch.save(student_model.state_dict(), distilled_model_path)
print(f"Distilled model saved as: {distilled_model_filename}")
