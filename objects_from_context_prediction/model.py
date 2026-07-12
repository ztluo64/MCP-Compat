import torch
import torch.nn as nn
import torch.nn.functional as F

# Classification network
    
class ClassificationNet(nn.Module):
    def __init__(self, input_dim, num_classes=80):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 512)  
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 128)  
        self.fc4 = nn.Linear(128, num_classes) 
        
        self.dropout = nn.Dropout(0.3)  # Dropout layer to prevent overfitting
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate inputs along the feature dimension
        x = self.relu(self.fc1(x)) 
        x = self.dropout(x)  
        x = self.relu(self.fc2(x))  
        x = self.dropout(x)  
        x = self.relu(self.fc3(x))  
        x = self.dropout(x) 
        x = self.fc4(x) # Output results
        return x


def load_classification_model(filepath, input_dim, num_classes, device):
    model = ClassificationNet(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

if __name__ == "__main__":
    # Example usage
    input_dim = 4 * 64 * 64  # Adjust based on your VAE latent space size
    num_classes = 80
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = ClassificationNet(input_dim, num_classes).to(device)
    print("Model created with input dimension:", input_dim)
