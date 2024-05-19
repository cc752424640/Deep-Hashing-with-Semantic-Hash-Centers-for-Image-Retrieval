import os.path

import torch.nn as nn
from torchvision import models
import torch


class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet34"):
        super(ResNet, self).__init__()
        if os.path.exists('./save/resnet34-b627a593.pth'):
            model_resnet = resnet_dict[res_model](pretrained=False)
            pre = torch.load('./models_ckpt/resnet34-b627a593.pth')  # 进行加载
            model_resnet.load_state_dict(pre)
        else:
            model_resnet = resnet_dict[res_model](pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.tanh = nn.Tanh()
        # self.layer_hash = nn.Linear(model_resnet.fc.in_features, hash_bit)
        # self.layer_hash.weight.data.normal_(0, 0.01)
        # self.layer_hash.bias.data.fill_(0.0)
        # self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(hash_bit, momentum=0.1))

    def forward(self, x):

        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        # x = self.tanh(x)
        return x

class NewNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(NewNet, self).__init__()
        self.m = 0.9
        self.encoder_q = ResNet(hash_bit)
        self.encoder_k = ResNet(hash_bit)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        encode_x = self.encoder_q(x)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2 = self.encoder_k(x)
        return encode_x, encode_x2

class ClassifyNet(nn.Module):
    def __init__(self, n_class, res_model="ResNet34"):
        super(ClassifyNet, self).__init__()
        if os.path.exists('./save/resnet34-b627a593.pth'):
            model_resnet = resnet_dict[res_model](pretrained=False)
            pre = torch.load('./models_ckpt/resnet34-b627a593.pth')  # 进行加载
            model_resnet.load_state_dict(pre)
        else:
            model_resnet = resnet_dict[res_model](pretrained=True)


        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.classify_layer = nn.Linear(model_resnet.fc.in_features, n_class)
        self.softmax = nn.Softmax(dim=1)
        self.classify_layer.weight.data.normal_(0, 0.01)
        self.classify_layer.bias.data.fill_(0.0)

        # self.tanh = nn.Tanh()
        # self.layer_hash = nn.Linear(model_resnet.fc.in_features, hash_bit)
        # self.layer_hash.weight.data.normal_(0, 0.01)
        # self.layer_hash.bias.data.fill_(0.0)
        # self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(hash_bit, momentum=0.1))

    def forward(self, x):

        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classify_layer(x)
        y = self.softmax(x)
        # x = self.tanh(x)
        return x, y


class LTHNet(nn.Module):
    def __init__(self, origin_model, feature_dim=2000, code_length=64, num_classes=100, num_prototypes=100):
        super(LTHNet, self).__init__()
        # self.dynamic_meta_embedding = dynamic_meta_embedding
        self.feature_dim = feature_dim
        self.code_length = code_length
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # direct features
        self.features = nn.Sequential(*list(origin_model.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

        # dynamic meta-embedding
        self.fc_hallucinator = nn.Linear(feature_dim, num_prototypes)
        self.fc_selector = nn.Linear(feature_dim, feature_dim)
        self.attention = nn.Softmax(dim=1)

        # hash layer and classifier
        self.hash_layer = nn.Linear(feature_dim, code_length)

        self.classifier = nn.Linear(code_length, num_classes)
        self.assignments = nn.Softmax(dim=1)

    def forward(self, x, dynamic_meta_embedding, prototypes):
        # generate the feature
        #print(x.shape)
        #data = x
        #Image.open(data)


        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)

        # storing direct feature
        direct_feature = x

        if dynamic_meta_embedding:
            # visual memory: consisted of prototypes, each of which represents a center of one semantic structure.
            # visual_memory = prototypes, sized by [num_prototypes, feature_dim].
            if prototypes.size(0) != self.num_prototypes or prototypes.size(1) != self.feature_dim:
                print(prototypes.size(0))
                print(prototypes.size(1))
                print(prototypes.size(0) != self.num_prototypes)
                print(prototypes.size(1) != self.feature_dim)
                print('prototypes error')
                return

            # computing memory_feature by querying and associating visual memory (prototypes)
            attention = self.fc_hallucinator(x)
            attention = self.attention(attention)
            memory_feature = torch.matmul(attention, prototypes)

            # computing concept selector
            concept_selector = self.fc_selector(x)
            concept_selector = nn.Tanh()(concept_selector)

            # infused feature
            x_meta = direct_feature + concept_selector * memory_feature

            # generate hashing
            x = self.hash_layer(x_meta)
            hash_codes = nn.Tanh()(x)

            # class assignments
            assignments = self.classifier(hash_codes)
            assignments = self.assignments(assignments)

        else:
            x_meta = direct_feature  # no dynamic meta-embedding

            # generate hashing
            x = self.hash_layer(x_meta)
            hash_codes = nn.Tanh()(x)

            # class assignments
            assignments = self.classifier(hash_codes)
            assignments = self.assignments(assignments)

        return hash_codes, assignments, direct_feature


class orthohashNet(nn.Module):
    def __init__(self, hash_bit, nclass, res_model="ResNet34"):
        super(orthohashNet, self).__init__()
        if os.path.exists('./save/resnet34-b627a593.pth'):
            model_resnet = resnet_dict[res_model](pretrained=False)
            pre = torch.load('./models_ckpt/resnet34-b627a593.pth')  # 进行加载
            model_resnet.load_state_dict(pre)
        else:
            model_resnet = resnet_dict[res_model](pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.ce_fc = nn.Linear(hash_bit, nclass)
        # self.tanh = nn.Tanh()
        # self.layer_hash = nn.Linear(model_resnet.fc.in_features, hash_bit)
        # self.layer_hash.weight.data.normal_(0, 0.01)
        # self.layer_hash.bias.data.fill_(0.0)
        # self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(hash_bit, momentum=0.1))

    def forward(self, x):

        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        y = self.ce_fc(x)
        # x = self.tanh(x)
        return y, x

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_layer.parameters())

    def get_backbone_params(self):
        return list(self.feature_layers.parameters())