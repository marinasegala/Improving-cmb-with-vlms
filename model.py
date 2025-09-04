import torch.nn as nn

# ================================
# MODEL DEF
# ================================

class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts=6, use_fc=False, input_channels=3):
        """
        The ConceptPredictor takes an image (X) as input and predicts the concepts (C).
        Args:
            num_concepts (int): Number of concepts to predict from the image.
            input_channels (int): Number of input channels (3 for RGB).
        """
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_concepts),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.float()
        return self.backbone(x)

class LabelPredictor(nn.Module):
    def __init__(self, input_dim=6, num_classes=4):
        """
        The LabelPredictor takes concepts (C) as input and predicts the labels (Y).
        Args:
            input_dim (int): Number of concepts.
            num_classes (int): Number of possible labels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, c):
        return self.net(c)

class End2EndModel(nn.Module):
    def __init__(self, concept_predictor, label_predictor):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.label_predictor = label_predictor
    
    def forward(self, x):
        concepts = self.concept_predictor(x)
        preds = self.label_predictor(concepts)
        return concepts, preds

def ModelXtoCtoY(num_concepts=14, num_classes=2):
    concept_predictor = ConceptPredictor(num_concepts=num_concepts)
    label_predictor = LabelPredictor(input_dim=num_concepts, num_classes=num_classes)
    return End2EndModel(concept_predictor, label_predictor)