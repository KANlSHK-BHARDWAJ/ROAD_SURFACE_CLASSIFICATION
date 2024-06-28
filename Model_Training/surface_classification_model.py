import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def train_and_validate_model():

    data_dir = r#change data directory path as per your saved location

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # Creating dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Spliting dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Creating data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Defining the model
    num_classes = 27 
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Using DirectML device for tensor operations
    device = torch_directml.device()

    # Moving the model to DirectML device
    model.to(device)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  

    # Defining a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Variables to store training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training and validation loop
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 2  
    patience_counter = 0 

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  
            
            outputs = model(inputs) 
            loss = criterion(outputs, labels)  
            loss.backward() 
            optimizer.step()  
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss / val_size
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

        # Step the scheduler
        scheduler.step()

    # Loading best model weights
    model.load_state_dict(best_model_wts)

    # Saving the trained model
    torch.save(model.state_dict(), 'road_surface_model.pth')

    # Plotling and saving the graphs
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Plotting validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Validation Accuracy')

    plt.savefig('training_results.png')
    plt.show()

if __name__ == '__main__':
    train_and_validate_model()
