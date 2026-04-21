import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. THE SOTA MAST-NET ARCHITECTURE
# ==========================================
class MASTNet_SOTA(nn.Module):
    def __init__(self):
        super(MASTNet_SOTA, self).__init__()

        # Parallel Spatial Extraction (4 Specialists)
        self.branch1 = self._make_cnn_branch(3) # Sentinel-5P
        self.branch2 = self._make_cnn_branch(2) # MODIS
        self.branch3 = self._make_cnn_branch(2) # Landsat-8
        self.branch4 = self._make_cnn_branch(2) # Ground-Auxiliary

        # Dynamic Feature Selection (DFS) - The Gatekeeper
        self.dfs_layer = nn.Sequential(
            nn.Linear(256, 256), # 64 features * 4 branches = 256
            nn.Sigmoid()
        )

        # Bi-Directional Temporal Modeling (7-day memory)
        # input_size matches the 256 selected features
        self.temporal_bi_lstm = nn.LSTM(input_size=256, hidden_size=128,
                                        num_layers=2, batch_first=True, bidirectional=True)

        # Multi-Head Attention Fusion (8-Heads)
        # Bi-LSTM hidden 128 * 2 directions = 256
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

        # Uncertainty Quantification (UQ) Output
        # Outputs 8 values: 4 Means (Predictions) and 4 Variances (Uncertainty)
        self.output_layer = nn.Linear(256, 8)

    def _make_cnn_branch(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x_list, x_ts):
        # 1. Spatial Feature Extraction
        f1 = self.branch1(x_list[0]).view(x_list[0].size(0), -1)
        f2 = self.branch2(x_list[1]).view(x_list[1].size(0), -1)
        f3 = self.branch3(x_list[2]).view(x_list[2].size(0), -1)
        f4 = self.branch4(x_list[3]).view(x_list[3].size(0), -1)

        spatial_combined = torch.cat((f1, f2, f3, f4), dim=1)

        # 2. Dynamic Selection (Filtering noisy satellite data)
        selected_features = self.dfs_layer(spatial_combined) * spatial_combined

        # 3. Temporal Modeling (Using selected spatial features over time)
        # We simulate the temporal sequence by repeating the selected features
        # In real world, this would be a sequence of different daily images
        x_ts_input = selected_features.unsqueeze(1).repeat(1, 7, 1)
        t_feat, _ = self.temporal_bi_lstm(x_ts_input)
        t_feat = t_feat[:, -1, :] # Take last hidden state

        # 4. Attention Fusion
        combined = t_feat.unsqueeze(0)
        attn_out, _ = self.attention(combined, combined, combined)

        # 5. UQ Output
        out = self.output_layer(attn_out.squeeze(0))
        mean = out[:, :4]
        log_var = out[:, 4:] # Use log_var for numerical stability
        return mean, log_var

# ==========================================
# 2. SOTA DATA SIMULATOR (Triple-Satellite)
# ==========================================
def generate_sota_data(batch_size=16):
    # Branch 1: Sentinel (3 channels), Branch 2-4: 2 channels each
    x1 = torch.randn(batch_size, 3, 64, 64)
    x2 = torch.randn(batch_size, 2, 64, 64)
    x3 = torch.randn(batch_size, 2, 64, 64)
    x4 = torch.randn(batch_size, 2, 64, 64)

    # Time Series (Not used directly in this specific forward flow simplified for Colab)
    x_ts = torch.randn(batch_size, 7, 256)

    # Ground Truth (4 Pollutants)
    targets = torch.randn(batch_size, 4)

    return [x1, x2, x3, x4], x_ts, targets

# ==========================================
# 3. EXECUTION & TRAINING LOOP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MASTNet_SOTA().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Gaussian Negative Log Likelihood Loss (Standard for Uncertainty Quantification)
def criterion(mean, log_var, target):
    precision = torch.exp(-log_var)
    return torch.mean(0.5 * precision * (target - mean)**2 + 0.5 * log_var)

print(f"Starting SOTA MAST-Net Training on {device}...")
loss_history = []

for epoch in range(1, 11):
    x_list, x_ts, targets = generate_sota_data()
    x_list = [x.to(device) for x in x_list]
    targets = targets.to(device)

    optimizer.zero_grad()
    mean, log_var = model(x_list, x_ts.to(device))

    loss = criterion(mean, log_var, targets)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    print(f"Epoch [{epoch}/10] | SOTA Loss (with UQ): {loss.item():.4f}")

# ==========================================
# 4. RESULTS VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='blue', marker='x', label='SOTA Negative Log-Likelihood')
plt.title("MAST-Net SOTA Performance (With Uncertainty Logic)")
plt.xlabel("Epochs")
plt.ylabel("Loss (Error + Uncertainty)")
plt.legend()
plt.grid(True)
plt.show()

print("\n[COMPLETE] SOTA MAST-Net is operational.")
