import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x):  # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B*T, 512)
        feat = self.fc(feat)  # (B*T, out_dim)
        feat = feat.view(B, T, -1)  # (B, T, out_dim)
        return feat


class NumericEncoder(nn.Module):
    def __init__(self, in_dim=4, out_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, loc, distance, height):
        x = torch.cat([loc, distance, height], dim=-1)  # (B,T,4)
        x = self.mlp(x)  # (B,T,out_dim)
        return x




class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]




class MultiModalTransformer(nn.Module):
    def __init__(self, img_dim=128, num_dim=32, hidden_dim=160, nhead=4, num_layers=2,
                 out_seq_len=5, num_beams=64, seq_len=8):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=img_dim)
        self.num_encoder = NumericEncoder(in_dim=4, out_dim=num_dim)
        self.input_fc = nn.Linear(img_dim + num_dim, hidden_dim)

        self.pos_encoder = LearnablePositionalEncoding(seq_len=seq_len, hidden_dim=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, out_seq_len * num_beams)
        self.out_seq_len = out_seq_len
        self.num_beams = num_beams

    def forward(self, images, loc, distance, height):
        img_feat = self.img_encoder(images)  # (B,T,img_dim)
        num_feat = self.num_encoder(loc, distance, height)  # (B,T,num_dim)
        x = torch.cat([img_feat, num_feat], dim=-1)  # (B,T,img_dim+num_dim)
        x = self.input_fc(x)  # (B,T,hidden_dim)
        x = self.pos_encoder(x)  
        x = self.transformer(x)  # (B,T,hidden_dim)

        x_last = x[:, -1, :]  # (B, hidden_dim)
        out = self.fc_out(x_last)  # (B, out_seq_len*num_beams)
        out = out.view(-1, self.out_seq_len, self.num_beams)  # (B,5,num_beams)

        return out


class MultiModalGRU(nn.Module):
    def __init__(self, img_dim=128, num_dim=32, hidden_dim=128, num_layers=2,
                 dropout=0.5, out_seq_len=5, num_beams=64):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=img_dim)
        self.num_encoder = NumericEncoder(in_dim=4, out_dim=num_dim)
        self.gru = nn.GRU(
            input_size=img_dim + num_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, out_seq_len * num_beams)
        self.out_seq_len = out_seq_len
        self.num_beams = num_beams

    def forward(self, images, loc, distance, height):
        img_feat = self.img_encoder(images)  # (B, T, img_dim)
        num_feat = self.num_encoder(loc, distance, height)  # (B, T, num_dim)
        x = torch.cat([img_feat, num_feat], dim=-1)  # (B, T, img_dim + num_dim)
        _, hidden = self.gru(x)
        x_last = hidden[-1]  # (B, hidden_dim)
        out = self.fc_out(x_last)
        out = out.view(-1, self.out_seq_len, self.num_beams)
        return out


class LocationOnlyTransformer(nn.Module):
    def __init__(self, num_dim=32, hidden_dim=160, nhead=4, num_layers=2,
                 out_seq_len=5, num_beams=64, seq_len=8):
        super().__init__()
        self.num_encoder = NumericEncoder(in_dim=4, out_dim=num_dim)
        
        self.input_fc = nn.Linear(num_dim, hidden_dim)

        self.pos_encoder = LearnablePositionalEncoding(seq_len=seq_len, hidden_dim=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, out_seq_len * num_beams)
        self.out_seq_len = out_seq_len
        self.num_beams = num_beams

    def forward(self, images, loc, distance, height):
        
        num_feat = self.num_encoder(loc, distance, height)  # (B, T, num_dim)
        
        x = self.input_fc(num_feat)  # (B, T, hidden_dim)
        x = self.pos_encoder(x)      
        x = self.transformer(x)      # (B, T, hidden_dim)

        x_last = x[:, -1, :]         # (B, hidden_dim)
        out = self.fc_out(x_last)    # (B, out_seq_len * num_beams)
        out = out.view(-1, self.out_seq_len, self.num_beams)

        return out


class ImageOnlyTransformer(nn.Module):
    def __init__(self, img_dim=128, hidden_dim=160, nhead=4, num_layers=2,
                 out_seq_len=5, num_beams=64, seq_len=8):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=img_dim)
        
        self.input_fc = nn.Linear(img_dim, hidden_dim)

        self.pos_encoder = LearnablePositionalEncoding(seq_len=seq_len, hidden_dim=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, out_seq_len * num_beams)
        self.out_seq_len = out_seq_len
        self.num_beams = num_beams

    def forward(self, images, loc, distance, height):
        img_feat = self.img_encoder(images)  # (B, T, img_dim)

        x = self.input_fc(img_feat)  # (B, T, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)      # (B, T, hidden_dim)

        x_last = x[:, -1, :]         # (B, hidden_dim)
        out = self.fc_out(x_last)    # (B, out_seq_len * num_beams)
        out = out.view(-1, self.out_seq_len, self.num_beams)

        return out


# from torchinfo import summary
# import torch
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # batch_size=16, T=8, C=3, H=W=256
# B, T, C, H, W = 16, 8, 3, 256, 256
# images = torch.randn(B, T, C, H, W).to(device)
# loc = torch.randn(B, T, 2).to(device)
# distance = torch.randn(B, T, 1).to(device)
# height = torch.randn(B, T, 1).to(device)
#
# model = MultiModalTransformer(img_dim=128, num_dim=32, hidden_dim=160,
#                               nhead=4, num_layers=2, out_seq_len=5, num_beams=64).to(device)
#
# summary(model, input_data=(images, loc, distance, height))
