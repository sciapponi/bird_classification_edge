import torch
import torchaudio
import torchaudio.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# Load audio file
def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    # Convert to mono if stereo
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze(), sr

# Parameters
n_filters = 40  # Number of filters
f_min = 1000  # Minimum frequency in Hz
f_max = 8000  # Maximum frequency in Hz
n_fft = 512  # FFT size
hop_length = 128  # Hop length for STFT
breakpoint = 500  # Frequency where we switch from log to linear
transition_width = 300  # Smooth transition width for differentiable log-linear

def compute_stft(waveform, n_fft, hop_length):
    """
    Compute STFT magnitude spectrogram
    
    Args:
        waveform: [n_samples]
        n_fft: FFT size
        hop_length: Hop size
    
    Returns:
        magnitude: [n_freq, n_time]
    """
    # Ensure waveform is 1D
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    if waveform.dim() == 0:
        waveform = waveform.unsqueeze(0)
        
    stft = torch.stft(waveform, 
                     n_fft=n_fft, 
                     hop_length=hop_length,
                     window=torch.hann_window(n_fft).to(waveform.device),
                     return_complex=True)
    
    # Make sure the output is 2D [n_freq, n_time]
    return torch.abs(stft)

def get_mel_filter_bank(sr, n_fft, n_filters, f_min, f_max):
    """
    Create mel filter bank
    
    Args:
        sr: sample rate
        n_fft: FFT size
        n_filters: number of mel bands
        f_min: minimum frequency
        f_max: maximum frequency
    
    Returns:
        filter_bank: [n_filters, n_freq] where n_freq = n_fft//2 + 1
    """
    # Get the mel filter bank
    filter_bank = F.melscale_fbanks(
        n_freqs=n_fft//2 + 1,  # This must match STFT output
        f_min=f_min,
        f_max=f_max,
        n_mels=n_filters,
        sample_rate=sr,
        norm='slaney',
        mel_scale='htk'
    )
    
    # Transpose if needed to ensure shape is [n_filters, n_freq]
    if filter_bank.shape[0] != n_filters:
        filter_bank = filter_bank.T
        
    return filter_bank

# Compute log-spaced center frequencies
def get_log_filter_bank(n_filters, f_min, f_max, n_fft, sr):
    log_center_freqs = torch.logspace(torch.log10(torch.tensor(f_min)), 
                                      torch.log10(torch.tensor(f_max)), 
                                      steps=n_filters)
    log_fft_bins = torch.round((log_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    log_fft_bins = torch.clamp(log_fft_bins, 1, (n_fft // 2) - 2)
    return generate_filter_bank(log_fft_bins, n_filters, n_fft)

# Compute piecewise log-linear center frequencies
def get_piecewise_filter_bank(n_filters, f_min, f_max, breakpoint, n_fft, sr):
    n_low = n_filters // 2
    n_high = n_filters - n_low
    low_freqs = torch.logspace(torch.log10(torch.tensor(f_min)), 
                               torch.log10(torch.tensor(breakpoint)), 
                               steps=n_low)
    high_freqs = torch.linspace(breakpoint, f_max, steps=n_high)
    piecewise_center_freqs = torch.cat([low_freqs, high_freqs])
    piecewise_fft_bins = torch.round((piecewise_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    piecewise_fft_bins = torch.clamp(piecewise_fft_bins, 1, (n_fft // 2) - 2)
    return generate_filter_bank(piecewise_fft_bins, n_filters, n_fft)

# Compute differentiable piecewise log-linear center frequencies
def get_differentiable_filter_bank(n_filters, f_min, f_max, breakpoint, transition_width, n_fft, sr):
    x = torch.linspace(0, 1, steps=n_filters)
    log_part = f_min * (f_max / f_min) ** x
    linear_part = f_min + x * (f_max - f_min)
    S = torch.sigmoid((x - (breakpoint - f_min) / (f_max - f_min)) * transition_width)
    differentiable_center_freqs = (1 - S) * log_part + S * linear_part
    differentiable_fft_bins = torch.round((differentiable_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    differentiable_fft_bins = torch.clamp(differentiable_fft_bins, 1, (n_fft // 2) - 2)
    return generate_filter_bank(differentiable_fft_bins, n_filters, n_fft)

# Function to generate triangular filters
def generate_filter_bank(fft_bins, n_filters, n_fft):
    filter_bank = torch.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left = fft_bins[i - 1] if i > 0 else 0
        center = fft_bins[i]
        right = fft_bins[i + 1] if i < n_filters - 1 else (n_fft // 2)

        if center > left and center - left > 1:
            filter_bank[i, left:center] = torch.linspace(0, 1, steps=center - left)
        if right > center and right - center > 1:
            filter_bank[i, center:right] = torch.linspace(1, 0, steps=right - center)
    return filter_bank

def apply_filter_bank(filter_bank, magnitude):
    """
    Apply filter bank to STFT magnitude spectrogram
    
    Args:
        filter_bank: [n_filters, n_freq]
        magnitude: [n_freq, n_time]
    
    Returns:
        filtered: [n_filters, n_time]
    """
    # Ensure both tensors are on same device and dtype
    filter_bank = filter_bank.to(magnitude.device).float()
    magnitude = magnitude.float()
    
    # Check dimensions
    assert filter_bank.shape[1] == magnitude.shape[0], \
        f"Frequency dimension mismatch: filter_bank {filter_bank.shape} vs magnitude {magnitude.shape}"
    
    # [n_filters, n_freq] @ [n_freq, n_time] = [n_filters, n_time]
    return torch.matmul(filter_bank, magnitude)

# Function to convert to dB scale and normalize
def to_db_scale(S, ref=None):
    if ref is None:
        ref = torch.max(S)
    return 20 * torch.log10(torch.clamp(S, min=1e-10) / ref)

# Differentiability check function
def check_differentiability():
    # Test parameters
    n_filters = 10
    f_min = 500.0
    f_max = 8000.0
    breakpoint = 1000.0
    transition_width = 50.0
    n_fft = 512
    sr = 22050
    
    # Create input with requires_grad
    x = torch.linspace(0, 1, steps=n_filters, requires_grad=True)
    
    # Compute center frequencies
    log_part = f_min * (f_max / f_min) ** x
    linear_part = f_min + x * (f_max - f_min)
    S = torch.sigmoid((x - (breakpoint - f_min) / (f_max - f_min)) * transition_width)
    freqs = (1 - S) * log_part + S * linear_part
    
    # Test backpropagation
    try:
        loss = freqs.sum()
        loss.backward()
        if x.grad is not None:
            print("Differentiability check PASSED")
            print("Sample gradients:", x.grad[:3])
            return True
        else:
            print("Differentiability check FAILED (no gradients)")
            return False
    except Exception as e:
        print(f"Differentiability check FAILED: {str(e)}")
        return False

def get_differentiable_filter_bank(n_filters, f_min, f_max, breakpoint, transition_width, n_fft, sr):
    # Create parameter space
    x = torch.linspace(0, 1, steps=n_filters)
    
    # Compute in log domain directly
    log_min = torch.log(torch.tensor(f_min))
    log_max = torch.log(torch.tensor(f_max))
    log_break = torch.log(torch.tensor(breakpoint))
    
    # Compute log-spaced frequencies
    log_freqs = torch.exp(log_min + x * (log_max - log_min))
    
    # Compute linear-spaced frequencies
    lin_freqs = f_min + x * (f_max - f_min)
    
    # Create a smoother transition using sigmoid
    normalized_break = (log_break - log_min) / (log_max - log_min)
    S = torch.sigmoid((x - normalized_break) * transition_width)
    
    # Combine log and linear parts
    differentiable_center_freqs = (1 - S) * log_freqs + S * lin_freqs
    
    # Convert to FFT bins
    differentiable_fft_bins = torch.round((differentiable_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    differentiable_fft_bins = torch.clamp(differentiable_fft_bins, 1, (n_fft // 2) - 2)
    
    return generate_filter_bank(differentiable_fft_bins, n_filters, n_fft)

# Revised differentiability check function
def check_differentiability():
    # Test parameters
    n_filters = 10
    f_min = 500.0
    f_max = 8000.0
    breakpoint = 1000.0
    transition_width = 10.0  # Reduced from 50.0 to decrease gradient magnitude
    n_fft = 512
    sr = 22050
    
    # Create input with requires_grad
    x = torch.linspace(0, 1, steps=n_filters, requires_grad=True)
    
    # Compute center frequencies using a more stable approach
    # Using log domain directly
    log_min = torch.tensor(torch.log(torch.tensor(f_min)))
    log_max = torch.tensor(torch.log(torch.tensor(f_max)))
    log_break = torch.tensor(torch.log(torch.tensor(breakpoint)))
    
    # Compute log-spaced frequencies
    log_freqs = torch.exp(log_min + x * (log_max - log_min))
    
    # Compute linear-spaced frequencies
    lin_freqs = f_min + x * (f_max - f_min)
    
    # Create a smoother transition using sigmoid
    normalized_break = (torch.log(torch.tensor(breakpoint)) - log_min) / (log_max - log_min)
    S = torch.sigmoid((x - normalized_break) * transition_width)
    
    # Combine log and linear parts
    freqs = (1 - S) * log_freqs + S * lin_freqs
    
    # Apply scaling factor to reduce gradient magnitude
    scaling_factor = 0.01
    freqs_scaled = freqs * scaling_factor
    
    # Test backpropagation
    try:
        loss = freqs_scaled.sum()
        loss.backward()
        if x.grad is not None:
            print("Differentiability check PASSED")
            print("Sample gradients:", x.grad[:3])
            return True
        else:
            print("Differentiability check FAILED (no gradients)")
            return False
    except Exception as e:
        print(f"Differentiability check FAILED: {str(e)}")
        return False

class DifferentiableSpectrogram(nn.Module):
    def __init__(self, n_filters=40, f_min=1000, f_max=8000, n_fft=512, 
                 hop_length=128, initial_breakpoint=500, initial_transition_width=300):
        super().__init__()
        
        # Register learnable parameters
        self.breakpoint = nn.Parameter(torch.tensor(float(initial_breakpoint)))
        self.transition_width = nn.Parameter(torch.tensor(float(initial_transition_width)))
        
        # Store other parameters
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize filter bank
        self.register_buffer('filter_bank', self._create_filter_bank())
        
    def _create_filter_bank(self):
        """Create the filter bank based on current parameters"""
        # Determine the target device from one of the module's parameters
        target_device = self.breakpoint.device

        # Create parameter space
        x = torch.linspace(0, 1, steps=self.n_filters).to(target_device)
        
        # Compute in log domain directly
        log_min = torch.log(torch.tensor(self.f_min, device=target_device))
        log_max = torch.log(torch.tensor(self.f_max, device=target_device))
        # self.breakpoint is already a tensor on the target_device (it's an nn.Parameter)
        log_break = torch.log(self.breakpoint) 
        
        # Compute log-spaced frequencies
        log_freqs = torch.exp(log_min + x * (log_max - log_min))
        
        # Compute linear-spaced frequencies
        lin_freqs = self.f_min + x * (self.f_max - self.f_min) # self.f_min and self.f_max are scalars, result will be on x's device
        
        # Create a smoother transition using sigmoid
        # Ensure normalized_break is on the correct device, or its components are
        # (log_break - log_min) will be on target_device
        # (log_max - log_min) will be on target_device
        normalized_break = (log_break - log_min) / (log_max - log_min)
        
        # self.transition_width is an nn.Parameter, already on target_device
        # x is on target_device, normalized_break is on target_device
        S = torch.sigmoid((x - normalized_break) * self.transition_width)
        
        # Combine log and linear parts
        differentiable_center_freqs = (1 - S) * log_freqs + S * lin_freqs
        
        # Convert to FFT bins
        # Ensure calculations leading to fft_bins are on the correct device
        # Example: (self.f_max / 2) could be problematic if self.f_max is a Python float
        # However, the tensor operations should keep things on target_device if inputs are there.
        differentiable_fft_bins = torch.round((differentiable_center_freqs / (torch.tensor(self.f_max, device=target_device) / 2)) * (self.n_fft // 2)).long()
        differentiable_fft_bins = torch.clamp(differentiable_fft_bins, 1, (self.n_fft // 2) - 2)
        
        # _generate_filter_bank will also need to use target_device
        return self._generate_filter_bank(differentiable_fft_bins, target_device)
    
    def _generate_filter_bank(self, fft_bins, device):
        """Generate triangular filter bank"""
        filter_bank = torch.zeros((self.n_filters, self.n_fft // 2 + 1), device=device)
        for i in range(self.n_filters):
            left = fft_bins[i - 1] if i > 0 else 0
            center = fft_bins[i]
            right = fft_bins[i + 1] if i < self.n_filters - 1 else (self.n_fft // 2)

            # Ensure linspace is created on the correct device
            if center > left and center - left > 1:
                filter_bank[i, left:center] = torch.linspace(0, 1, steps=center - left, device=device)
            if right > center and right - center > 1:
                filter_bank[i, center:right] = torch.linspace(1, 0, steps=right - center, device=device)
        return filter_bank
    
    def forward(self, waveform):
        """
        Forward pass to compute the differentiable spectrogram
        
        Args:
            waveform: [batch_size, n_samples] or [n_samples]
            
        Returns:
            spectrogram: [batch_size, n_filters, n_time] or [n_filters, n_time]
        """
        # Ensure waveform is 2D [batch_size, n_samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Compute STFT
        stft = torch.stft(waveform, 
                         n_fft=self.n_fft, 
                         hop_length=self.hop_length,
                         window=torch.hann_window(self.n_fft).to(waveform.device),
                         return_complex=True)
        
        # Get magnitude
        magnitude = torch.abs(stft)
        
        # Update filter bank if parameters have changed
        self.filter_bank = self._create_filter_bank().to(waveform.device)
        
        # Apply filter bank
        # [batch_size, n_freq, n_time] -> [batch_size, n_filters, n_time]
        spectrogram = torch.matmul(self.filter_bank, magnitude)
        
        # Convert to dB scale
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-10))
        
        return spectrogram

# Example usage
def test_differentiable_spectrogram():
    # Create model
    model = DifferentiableSpectrogram()
    
    # Create dummy input
    x = torch.randn(1, 16000)  # 1 second of audio at 16kHz
    
    # Forward pass
    spec = model(x)
    
    # Check gradients
    loss = spec.sum()
    loss.backward()
    
    print("Breakpoint gradient:", model.breakpoint.grad)
    print("Transition width gradient:", model.transition_width.grad)
    print("Spectrogram shape:", spec.shape)

# Main function
def analyze_robin_sound(file_path="robin.wav"):
    # Run differentiability check
    print("Running differentiability check...")
    check_differentiability()
    
    # Load audio
    print(f"Loading audio file: {file_path}")
    y, sr = load_audio(file_path)
    
    # Adjust f_max if needed
    global f_max
    if sr < 2 * f_max:
        f_max = sr // 2 - 1000
    
    # Compute STFT
    print("Computing STFT...")
    stft_magnitude = compute_stft(y, n_fft, hop_length)
    
    # Compute filter banks
    print("Computing filter banks...")
    mel_filter_bank = get_mel_filter_bank(sr, n_fft, n_filters, f_min, f_max)
    log_filter_bank = get_log_filter_bank(n_filters, f_min, f_max, n_fft, sr)
    piecewise_filter_bank = get_piecewise_filter_bank(n_filters, f_min, f_max, breakpoint, n_fft, sr)
    differentiable_filter_bank = get_differentiable_filter_bank(n_filters, f_min, f_max, breakpoint, transition_width, n_fft, sr)
    print("DFF.shape: ",differentiable_filter_bank.shape)
    # Apply filter banks
    print("Applying filter banks...")
    mel_spectrogram = apply_filter_bank(mel_filter_bank, stft_magnitude)
    log_spectrogram = apply_filter_bank(log_filter_bank, stft_magnitude)
    piecewise_spectrogram = apply_filter_bank(piecewise_filter_bank, stft_magnitude)
    differentiable_spectrogram = apply_filter_bank(differentiable_filter_bank, stft_magnitude)
    
    # Convert to dB
    stft_db = to_db_scale(stft_magnitude)
    mel_db = to_db_scale(mel_spectrogram)
    log_db = to_db_scale(log_spectrogram)
    piecewise_db = to_db_scale(piecewise_spectrogram)
    differentiable_db = to_db_scale(differentiable_spectrogram)
    
    # Plot results
    print("Plotting results...")
    fig, ax = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # STFT Spectrogram
    img = ax[0].imshow(stft_db.numpy(), aspect='auto', origin='lower', 
                      extent=[0, stft_db.shape[1] * hop_length / sr, 0, sr // 2])
    ax[0].set_title('STFT Magnitude Spectrogram')
    ax[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax[0], format='%+2.0f dB')
    
    # Time axis
    time_points = torch.arange(mel_db.shape[1]) * hop_length / sr
    freq_points = torch.arange(n_filters)
    
    # Mel Spectrogram
    img = ax[1].pcolormesh(time_points.numpy(), freq_points.numpy(), mel_db.numpy(), 
                          shading='auto', cmap='viridis')
    ax[1].set_title('Mel Filter Bank Spectrogram')
    ax[1].set_ylabel('Mel Filter Index')
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    
    # Log Spectrogram
    img = ax[2].pcolormesh(time_points.numpy(), freq_points.numpy(), log_db.numpy(), 
                          shading='auto', cmap='viridis')
    ax[2].set_title('Log-Spaced Filter Bank Spectrogram')
    ax[2].set_ylabel('Filter Index')
    fig.colorbar(img, ax=ax[2], format='%+2.0f dB')
    
    # Piecewise Spectrogram
    img = ax[3].pcolormesh(time_points.numpy(), freq_points.numpy(), piecewise_db.numpy(), 
                          shading='auto', cmap='viridis')
    ax[3].set_title('Piecewise Log-Linear Filter Bank Spectrogram')
    ax[3].set_ylabel('Filter Index')
    fig.colorbar(img, ax=ax[3], format='%+2.0f dB')
    
    # Differentiable Spectrogram
    img = ax[4].pcolormesh(time_points.numpy(), freq_points.numpy(), differentiable_db.numpy(), 
                          shading='auto', cmap='viridis')
    ax[4].set_title('Differentiable Piecewise Log-Linear Filter Bank Spectrogram')
    ax[4].set_xlabel('Time (s)')
    ax[4].set_ylabel('Filter Index')
    fig.colorbar(img, ax=ax[4], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis complete!")
    return {
        'stft': stft_db,
        'mel': mel_db,
        'log': log_db,
        'piecewise': piecewise_db,
        'differentiable': differentiable_db
    }

if __name__ == "__main__":
    analyze_robin_sound("toucan.wav")