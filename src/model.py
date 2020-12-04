import torch
import torch.nn as nn
import timm

model_arch = 'gluon_seresnext50_32x4d'

model = timm.create_model(model_arch, pretrained=True)
print(dir(model.fc))

# class OcrModel_v0(nn.Module):
#     def __init__(self,model_arch, num_characters, pretrained=False):
#         super().__init__()
#         self.model = timm.create_model(model_arch, pretrained=pretrained)
#         n_features = self.model.classifier.in_features
#         self.model.classifier = nn.Linear(n_features, num_characters)
#         '''
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
#             nn.Linear(n_features, n_class, bias=True)
#         )
#         '''
#     def forward(self, x):
#         x = self.model(x)
#         return x



# class CassvaImgClassifier(nn.Module):
#     def __init__(self, model_arch, n_class, pretrained=False):
#         super().__init__()
#         self.model = timm.create_model(model_arch, pretrained=pretrained)
#         n_features = self.model.classifier.in_features
#         self.model.classifier = nn.Linear(n_features, n_class)
#         '''
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
#             nn.Linear(n_features, n_class, bias=True)
#         )
#         '''
#     def forward(self, x):
#         x = self.model(x)
#         return x

# if __name__ == '__main__':
#     model = OcrModel_v0(model_arch='gluon_seresnext50_32x4d', num_characters=19, pretrained=True)
#     img = torch.rand(1,3,75,300)
#     target = torch.randint(1,20,(1,5))
#     x, loss = model(img, target)