import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from cnn import StudentCNN
from common import load_data, get_loader, test_model, save_model, device
from resnet import HASYv2ResNet


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


if __name__ == "__main__":
    print("loading data...")
    data, labels = load_data("./data/HASYv2")

    print("processing data...")
    train_loader, test_loader = get_loader(data, labels)
    num_classes = len(np.unique(labels)) + 1

    # Load the teacher model
    teacher_model = HASYv2ResNet(num_classes)
    teacher_model.load_state_dict(torch.load("./models/resnet_20241110_041804.pth"))
    teacher_model.eval()  # Set teacher to evaluation mode
    teacher_model.to(device)

    # Define SimpleCNN model as student
    student_model = StudentCNN(num_classes)
    student_model.to(device)

    # Optimizer for the student model
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop for distillation
    num_epochs = 15
    for epoch in range(num_epochs):
        distill_student_model(
            epoch,
            student_model,
            teacher_model,
            train_loader,
            optimizer,
            num_epochs=num_epochs,
        )
        test_model(student_model, test_loader, criterion)

    save_model(student_model, "simpleCNN_distilled")
