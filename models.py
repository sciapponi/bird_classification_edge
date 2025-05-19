from modules import HighwayGRU, MatchboxNetSkip, AttentionLayer, StatefulRNNLayer, FocusedAttention, StatefulGRU, LightConsonantEnhancer #, FocusedAttention, StatefulGRU, LightConsonantEnhancer # Temporaneamente commentate perchè mancanti in modules.py
import torch
import torch.nn as nn
import torchaudio
import numpy as np # Added for linspace in filterbank
import sys
sys.path.append('.')
from differentiable_spec_torch import DifferentiableSpectrogram

# Helper function for Linear Triangular Filterbank (Hz-based)
def _create_triangular_filterbank_hz(num_filters, n_fft, sample_rate, min_freq_hz=0.0, max_freq_hz=None):
    if max_freq_hz is None:
        max_freq_hz = sample_rate / 2.0
    
    # Frequencies of the STFT bins
    stft_freq_bins_hz = torch.linspace(0, sample_rate / 2.0, n_fft // 2 + 1)
    
    # Points defining filter edges and peaks (num_filters + 2 points for num_filters)
    points_hz = torch.tensor(np.linspace(min_freq_hz, max_freq_hz, num_filters + 2))
    
    filterbank = torch.zeros((num_filters, n_fft // 2 + 1)) # (num_filters, num_stft_bins)
    
    for i in range(num_filters):
        left_hz = points_hz[i]
        center_hz = points_hz[i+1] # Peak of the i-th filter
        right_hz = points_hz[i+2]
        
        # Rising slope
        mask_rising = (stft_freq_bins_hz >= left_hz) & (stft_freq_bins_hz <= center_hz)
        if center_hz > left_hz:
            filterbank[i, mask_rising] = (stft_freq_bins_hz[mask_rising] - left_hz) / (center_hz - left_hz)
        elif center_hz == left_hz and torch.sum(mask_rising) > 0:
             filterbank[i, mask_rising] = 1.0 # Handle case where left_hz == center_hz (point filter)

        # Falling slope
        mask_falling = (stft_freq_bins_hz > center_hz) & (stft_freq_bins_hz <= right_hz)
        if right_hz > center_hz:
            filterbank[i, mask_falling] = (right_hz - stft_freq_bins_hz[mask_falling]) / (right_hz - center_hz)
        # No special handling for right_hz == center_hz, as it means falling slope is zero (already init)
            
    return filterbank


class Improved_Phi_GRU_ATT(nn.Module):
    def __init__(self, num_classes=10, spectrogram_type="mel", sample_rate=32000, 
                 n_mel_bins=64, n_linear_filters=64, # Added n_linear_filters
                 f_min=0.0, f_max=None,  # mel/dataset specific
                 hidden_dim=32, n_fft=400, hop_length=160, matchbox={}, 
                 breakpoint=4000, transition_width=100, **kwargs):
        super(Improved_Phi_GRU_ATT, self).__init__()

        self.spectrogram_type = spectrogram_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.hidden_dim = hidden_dim
        self.n_mel_bins = n_mel_bins 
        self.n_linear_filters = n_linear_filters # Store n_linear_filters
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else self.sample_rate / 2.0
        self.breakpoint = breakpoint
        self.transition_width = transition_width

        current_n_input_features = 0
        self.mel_transform = None
        self.stft_transform = None
        self.linear_filterbank = None
        self.differentiable_spec = None
        self.combined_log_linear_spec = None

        if self.spectrogram_type == "mel":
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mel_bins,
                f_min=self.f_min,
                f_max=self.f_max,
                power=2.0
            )
            current_n_input_features = self.n_mel_bins
        elif self.spectrogram_type == "linear_stft":
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0
            )
            current_n_input_features = self.n_fft // 2 + 1
        elif self.spectrogram_type == "linear_triangular":
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0
            )
            self.linear_filterbank = _create_triangular_filterbank_hz(
                num_filters=self.n_linear_filters,
                n_fft=self.n_fft,
                sample_rate=self.sample_rate,
                min_freq_hz=self.f_min,
                max_freq_hz=self.f_max
            )
            current_n_input_features = self.n_linear_filters
        elif self.spectrogram_type == "combined_log_linear":
            # Nuova modalità: filtro log-lineare apprendibile
            self.combined_log_linear_spec = DifferentiableSpectrogram(
                n_filters=self.n_linear_filters,
                f_min=self.f_min,
                f_max=self.f_max,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                initial_breakpoint=self.breakpoint,
                initial_transition_width=self.transition_width
            )
            current_n_input_features = self.n_linear_filters
        else:
            raise ValueError(f"Unsupported spectrogram_type: {self.spectrogram_type}. Choose 'mel', 'linear_stft', 'linear_triangular', or 'combined_log_linear'.")

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        matchbox_cfg = matchbox.copy()
        matchbox_cfg['input_features'] = current_n_input_features 

        self.phi = MatchboxNetSkip(cfg=matchbox_cfg)
        self.gru = nn.GRU(
            input_size=matchbox.get('base_filters', 32),
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.keyword_attention = AttentionLayer(self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
            if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.size(1) == 1:
            x = x.squeeze(1) 

        if self.spectrogram_type == "mel":
            x = self.mel_transform(x)
        elif self.spectrogram_type == "linear_stft":
            x = self.stft_transform(x)
        elif self.spectrogram_type == "linear_triangular":
            x = self.stft_transform(x) # Raw power spectrogram: (batch, freq_raw, time)
            x = x.permute(0, 2, 1) # -> (batch, time, freq_raw)
            x = torch.matmul(x, self.linear_filterbank.T.to(x.device))
            x = x.permute(0, 2, 1) # -> (batch, freq_filt, time)
            x = self.amplitude_to_db(x)
        elif self.spectrogram_type == "combined_log_linear":
            # Nuova modalità: usa il filtro log-lineare apprendibile
            x = self.combined_log_linear_spec(x)
        
        x = self.amplitude_to_db(x) if self.spectrogram_type not in ["linear_triangular", "combined_log_linear"] else x

        mean = x.mean(dim=(1, 2), keepdim=True) 
        std = x.std(dim=(1, 2), keepdim=True) + 1e-5 
        x = (x - mean) / std

        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.gru(x)
        x = self.projection(x)
        x, attention_weights = self.keyword_attention(x)
        x = self.fc(x)

        return x


class Improved_Phi_GRU_ATT_Spec(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=40, hidden_dim=32, matchbox={}):
        super(Improved_Phi_GRU_ATT_Spec, self).__init__()

        # Remove the mel_spec and amplitude_to_db transformations
        # as we now receive spectrograms directly from the dataset

        self.phi = MatchboxNetSkip(matchbox)
        self.gru = nn.GRU(
            input_size=matchbox.get('base_filters', 32),
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input x is now a spectrogram from the dataset
        # with shape [batch_size, 1, n_mel_bins, time]

        # Normalize the input
        if x.dim() == 4 and x.size(1) == 1:
            # Input is [batch_size, 1, n_mel_bins, time]
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-5
            x = (x - mean) / std
            x = x.squeeze(1)  # Remove channel dimension -> [batch_size, n_mel_bins, time]
        elif x.dim() == 3:
            # Input is already [batch_size, n_mel_bins, time]
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True) + 1e-5
            x = (x - mean) / std

        # Process through MatchboxNet
        x = self.phi(x)

        # Prepare for GRU: convert from [batch_size, channels, time] to [batch_size, time, channels]
        x = x.permute(0, 2, 1).contiguous()

        # GRU processing
        x, _ = self.gru(x)

        # Projection
        x = self.projection(x)

        # Attention mechanism
        x, attention_weights = self.keyword_attention(x)

        # Final classification
        x = self.fc(x)

        return x

class Improved_Phi_FC(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, matchbox=None):
        super(Improved_Phi_FC, self).__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.phi = MatchboxNetSkip(matchbox)  # Expects n_mel_bins channels

        self.gru = nn.GRU(
            input_size=matchbox.base_filters,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        self.projection = nn.Linear(hidden_dim, hidden_dim)

        # Removed keyword_attention
        # self.keyword_attention = nn.MultiheadAttention(...)

        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)
        self.attention = FocusedAttention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std
        x = x.squeeze(1)

        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()

        x, _ = self.gru(x)
        x = self.projection(x)

        # Removed keyword_attention and residual connection
        # You could keep a residual or pass x directly

        x = self.consonant_enhancer(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Improved_Phi_FC_Hybrid(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, matchbox=None):
        super(Improved_Phi_FC_Hybrid, self).__init__()
        # Keep your original mel spectrogram components
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # The CNN backbone

        self.phi = MatchboxNetSkip(matchbox)  # Expects n_mel_bins channels

        # First use a single GRU layer (more parameter efficient than multiple)
        self.gru = nn.GRU(
            input_size=matchbox.base_filters,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        self.projection = nn.Linear(hidden_dim, hidden_dim)


        # Small self-attention to focus on keywords
        self.keyword_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,  # Single head for efficiency
            batch_first=True
        )

        # Add consonant enhancer (minimal parameters)
        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)
        # Keep your original attention
        self.attention = FocusedAttention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)
        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std
        # Remove the dimension we added
        x = x.squeeze(1)  # [batch, n_mels, time]
        # Now x should be correctly shaped for phi: [batch, n_mels, time]
        x = self.phi(x)  # Output shape: [batch, 64, time]
        x = x.permute(0, 2, 1).contiguous()  # [batch, time, 64]
        # print(x.shape)
        # exit()
        # GRU layer for sequential processing
        x, _ = self.gru(x)  # [batch, time, hidden_dim]
        x = self.projection(x)  # [batch, time, hidden_dim]

        # Small attention layer focused on keywords
        residual = x
        attn_output, _ = self.keyword_attention(
            query=x,
            key=x,
            value=x
        )
        x = residual + attn_output  # Residual connection

        # Enhance consonant features
        x = self.consonant_enhancer(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Improved_Phi_FC_Attention(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_heads=2):
        super(Improved_Phi_FC_Attention, self).__init__()
        # Keep your original mel spectrogram components
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # The CNN backbone
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Expects n_mel_bins channels

        # Replace RNN with lightweight Multi-head Attention
        self.input_projection = nn.Linear(64, hidden_dim)

        self.hidden_dim = hidden_dim
        # Reduced number of heads and smaller feed-forward dim
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Simplified feed-forward network with smaller expansion factor
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # Reduced from 4x to 2x expansion
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Single combined normalization layer
        self.norm = nn.LayerNorm(hidden_dim)

        # Add consonant enhancer (minimal parameters)
        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)
        # Keep your original attention
        self.attention = FocusedAttention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)
        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std
        # Remove the dimension we added
        x = x.squeeze(1)  # [batch, n_mels, time]
        # Now x should be correctly shaped for phi: [batch, n_mels, time]
        x = self.phi(x)  # Output shape: [batch, 64, time]
        x = x.permute(0, 2, 1).contiguous()  # [batch, time, 64]

        # Project to hidden dimension
        x = self.input_projection(x)  # [batch, time, hidden_dim]

        # Add positional information using relative position
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        positions = positions.unsqueeze(-1).float() / seq_len  # Simple normalized position [0-1]
        x = torch.cat([x, positions], dim=-1)  # Append position as feature
        x = nn.Linear(self.hidden_dim + 1, self.hidden_dim).to(x.device)(x)  # Project back to hidden_dim

        # Multi-head Attention (lighter weight)
        residual = x
        attn_output, _ = self.multihead_attention(
            query=x,
            key=x,
            value=x
        )
        x = residual + attn_output

        # Feed-forward block with smaller expansion
        residual = x
        x = self.feed_forward(x)
        x = residual + x

        # Single normalization at the end
        x = self.norm(x)

        # Enhance consonant features
        x = self.consonant_enhancer(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
class Improved_Phi_FC_Recurrent(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Improved_Phi_FC_Recurrent, self).__init__()

        # Keep your original mel spectrogram components
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # The CNN backbone
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Expects n_mel_bins channels

        # Use your original RNN implementation for now
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = 64 if i == 0 else hidden_dim
            self.rnn_layers.append(StatefulGRU(input_dim, hidden_dim))

        # Add consonant enhancer (minimal parameters)
        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)

        # Keep your original attention
        self.attention = FocusedAttention(hidden_dim)

        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)

        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)

        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        # Remove the dimension we added
        x = x.squeeze(1)  # [batch, n_mels, time]

        # print(x.shape)  # Debugging line to check shape after normalization
        # Now x should be correctly shaped for phi: [batch, n_mels, time]
        x = self.phi(x)  # Output shape: [batch, 64, time]
        x = x.permute(0, 2, 1).contiguous()  # [batch, time, 64]

        # Rest of your code is the same
        h_t = None
        for rnn in self.rnn_layers:
            x, h_t = rnn(x, h_t)

        # Enhance consonant features
        x = self.consonant_enhancer(x)

        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Phi_HGRU(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Phi_HGRU, self).__init__()

        # Mel spectrogram transformation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN Feature Extractor (MatchboxNet or equivalent small CNN)
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Ensure it returns (batch, num_filters, seq_len)

        # GRU instead of SRNN
        self.gru = HighwayGRU(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)
        # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialize GRU weights
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization for stability."""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:  # Apply Xavier Uniform to weights
                if len(param.shape) == 2:  # Ensure it's a weight matrix (2D)
                    nn.init.xavier_uniform_(param)
                else:
                    print(f"Skipping xavier_uniform for {name} due to non-2D shape {param.shape}")
            elif 'bias' in name:  # Apply zero initialization to biases
                nn.init.zeros_(param)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)

        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)

        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        # CNN Feature extraction
        x = x.squeeze(1)  # (batch, mel_bins, time)
        x = self.phi(x)   # (batch, num_filters, seq_len)

        # Reshape for GRU (batch, seq_len, features)
        x = x.permute(0, 2, 1).contiguous()

        # GRU forward pass
        x, _ = self.gru(x)  # Output shape: (batch, seq_len, hidden_dim)
        x, _ = self.attention(x)  # Apply attention to GRU output
        # Take last time step's output (for classification)
        # x = x[:, -1, :]  # (batch, hidden_dim)

        # Fully connected layers
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)

        return x

class Phi_FC_Recurrent(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Phi_FC_Recurrent, self).__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)

        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = 64 if i == 0 else hidden_dim
            self.rnn_layers.append(StatefulRNNLayer(input_dim, hidden_dim))

        # self.attention = AttentionLayer(hidden_dim)
        self.attention = FocusedAttention(hidden_size=hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        x = x.squeeze(1)
        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, seq_len, features)

        # Initialize hidden state
        h_t = None

        # Pass through stacked RNN layers with stateful connections
        for rnn in self.rnn_layers:
            x, h_t = rnn(x, h_t)

        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




class Phi_GRU(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Phi_GRU, self).__init__()

        # Mel spectrogram transformation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN Feature Extractor (MatchboxNet or equivalent small CNN)
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Ensure it returns (batch, num_filters, seq_len)

        # GRU instead of SRNN
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)
        # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialize GRU weights
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization for stability."""
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)

        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)

        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        # CNN Feature extraction
        x = x.squeeze(1)  # (batch, mel_bins, time)
        x = self.phi(x)   # (batch, num_filters, seq_len)

        # Reshape for GRU (batch, seq_len, features)
        x = x.permute(0, 2, 1).contiguous()

        # GRU forward pass
        x, _ = self.gru(x)  # Output shape: (batch, seq_len, hidden_dim)
        x, _ = self.attention(x)  # Apply attention to GRU output
        # Take last time step's output (for classification)
        # x = x[:, -1, :]  # (batch, hidden_dim)

        # Fully connected layers
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)

        return x


