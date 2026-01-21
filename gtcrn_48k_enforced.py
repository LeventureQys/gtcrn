"""
GTCRN 48kHz Inference Script (优化版)
=====================================================================
用于语音增强的推理脚本

48kHz 配置说明:
- n_fft=1536, hop_len=768, win_len=1536 (32ms 帧长)
- 频带数: 769 (n_fft/2 + 1)
- ERB 压缩: 4kHz 以下不压缩 (erb_subband_1=128)
- ERB 压缩频带数: 256

使用方法:
    # 处理单个文件
    python gtcrn_48k_inference.py -m model.tar -i audio.wav -o enhanced.wav
    
    # 批量处理文件夹
    python gtcrn_48k_inference.py -m model.tar --input_dir ./noisy_wavs --output_dir ./enhanced_wavs
    
    # 使用 GPU
    python gtcrn_48k_inference.py -m model.tar -i audio.wav -o enhanced.wav -d cuda
=====================================================================
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import soundfile as sf
from glob import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ======================= 模型定义部分 =======================

class ERB(nn.Module):
    """
    ERB (Equivalent Rectangular Bandwidth) 频带压缩模块
    
    48kHz 配置:
    - erb_subband_1: 4kHz 以下的频带数 (不压缩)
      计算: 4000 / (48000/1536) = 4000 / 31.25 ≈ 128
    - erb_subband_2: ERB 压缩后的频带数
    """
    def __init__(self, erb_subband_1, erb_subband_2, nfft=1536, high_lim=24000, fs=48000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10 ** (erb_f / 21.4) - 1) / 0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=1536, high_lim=24000, fs=48000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                          / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                                                      / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                                                          / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """Band Merge: x: (B,C,T,F) -> (B,C,T,F_erb)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """Band Split: x: (B,C,T,F_erb) -> (B,C,T,F)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride), 
                                padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)
        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, 
                 dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)

        self.tra = TRA(in_channels // 2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)
        x = self.shuffle(h1, x2)

        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, 
                           batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, 
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h is None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, dropout=0.1, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.intra_dropout = nn.Dropout(dropout)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
        self.inter_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)  # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)  # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_x = self.intra_dropout(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)  # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)  # (B,F,T,C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_x = self.inter_dropout(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return x


class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    """
    GTCRN 48kHz 模型
    
    48kHz 配置参数计算:
    - n_fft=1536 -> 频带数 nfreqs = 769
    - 帧长: 1536/48000 = 32ms (满足 >= 20ms 要求)
    - ERB 压缩起始频率: 4kHz (作者建议)
    - erb_subband_1 = round(4000 / (48000/1536)) = round(4000/31.25) = 128
    - erb_subband_2 = 256 (ERB 压缩后的频带数)
    - Total ERB bins: 128 + 256 = 384
    - After 2x freq downsampling: 384/4 = 96
    """
    def __init__(
        self,
        n_fft=1536,
        hop_len=768,
        win_len=1536,
        dropout=0.1,
        # ERB 参数 (可配置以匹配训练时的设置)
        erb_subband_1=128,  # 4kHz 以下不压缩 (推荐)
        erb_subband_2=256,  # ERB 压缩频带数
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len

        # 计算参数
        nfreqs = n_fft // 2 + 1  # 769 for n_fft=1536
        total_erb_bins = erb_subband_1 + erb_subband_2  # 128 + 256 = 384
        bottleneck_width = total_erb_bins // 4  # 384 / 4 = 96
        
        print(f"[GTCRN] 48kHz Configuration:")
        print(f"  - n_fft: {n_fft}, hop_len: {hop_len}, win_len: {win_len}")
        print(f"  - Frame length: {win_len/48000*1000:.1f}ms")
        print(f"  - Frequency bins: {nfreqs}")
        print(f"  - ERB: {erb_subband_1} (no compress) + {erb_subband_2} (compressed) = {total_erb_bins}")
        print(f"  - Bottleneck width: {bottleneck_width}")

        self.erb = ERB(erb_subband_1, erb_subband_2, nfft=n_fft, high_lim=24000, fs=48000)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()

        # Bottleneck: DPGRNN width 需要匹配 encoder 输出的频率维度
        self.dpgrnn1 = DPGRNN(16, bottleneck_width, 16, dropout=dropout)
        self.dpgrnn2 = DPGRNN(16, bottleneck_width, 16, dropout=dropout)

        self.decoder = Decoder()
        self.mask = Mask()

    def forward(self, x):
        """
        x: (B, L) - raw waveform at 48kHz
        """
        device = x.device
        n_samples = x.shape[1]

        stft_kwargs = {
            'n_fft': self.n_fft, 
            'hop_length': self.hop_len, 
            'win_length': self.win_len,
            'window': torch.hann_window(self.win_len).to(device), 
            'onesided': True
        }

        spec = torch.stft(x, **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)  # (B, F, T, 2), F=769

        spec_real = spec[..., 0].permute(0, 2, 1)  # (B, T, F)
        spec_imag = spec[..., 1].permute(0, 2, 1)  # (B, T, F)
        spec_mag = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,769)

        spec = spec.permute(0, 3, 2, 1)  # (B,2,T,F)

        feat = self.erb.bm(feat)  # (B,3,T,384) - ERB compression
        feat = self.sfe(feat)  # (B,9,T,384)

        feat, en_outs = self.encoder(feat)  # -> (B,16,T,96)

        feat = self.dpgrnn1(feat)  # (B,16,T,96)
        feat = self.dpgrnn2(feat)  # (B,16,T,96)

        m_feat = self.decoder(feat, en_outs)

        m = self.erb.bs(m_feat)  # ERB expansion back to 769

        spec_enh = self.mask(m, spec)  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        spec_enh = torch.complex(spec_enh[..., 0], spec_enh[..., 1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples - output.shape[1]))

        return output


# ======================= 推理工具类 =======================

class GTCRNInference:
    """GTCRN 48kHz 推理类"""
    
    SUPPORTED_FORMATS = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.wma', '.aac']
    
    def __init__(self, model_path: str, device: str = None, 
                 erb_subband_1: int = 128, erb_subband_2: int = 256):
        """
        初始化推理器
        
        Args:
            model_path: 预训练模型路径 (.tar, .pth, .pt)
            device: 设备 ('cuda', 'cpu', 或 None 自动选择)
            erb_subband_1: ERB 不压缩频带数 (需要匹配训练配置)
            erb_subband_2: ERB 压缩频带数 (需要匹配训练配置)
        """
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = GTCRN(
            n_fft=1536,
            hop_len=768,
            win_len=1536,
            erb_subband_1=erb_subband_1,
            erb_subband_2=erb_subband_2
        )
        
        # 加载权重
        self._load_checkpoint(model_path)
        
        # 设置为评估模式
        self.model.eval()
        self.model.to(self.device)
        
        # 采样率
        self.sample_rate = 48000
    
    def _load_checkpoint(self, model_path: str):
        """加载模型权重"""
        print(f"正在加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载 checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 尝试不同的权重格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 处理可能的 'module.' 前缀 (DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 加载权重 (strict=False 允许部分匹配)
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            print(f"  警告: 缺少的键: {missing}")
        if unexpected:
            print(f"  警告: 多余的键: {unexpected}")
        
        print("模型加载成功!")
        
        # 打印额外信息
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'best_score' in checkpoint:
                print(f"  Best Score: {checkpoint['best_score']:.4f}")
    
    def load_audio(self, audio_path: str) -> tuple:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            audio: 音频数据 (numpy array)
            sr: 采样率
        """
        audio, sr = sf.read(audio_path)
        
        # 立体声转单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 重采样到 48kHz
        if sr != self.sample_rate:
            print(f"  警告: 输入采样率 {sr}Hz，正在重采样到 {self.sample_rate}Hz")
            try:
                from scipy import signal
                gcd = np.gcd(sr, self.sample_rate)
                up = self.sample_rate // gcd
                down = sr // gcd
                audio = signal.resample_poly(audio, up, down).astype(np.float32)
            except Exception:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                except ImportError:
                    raise ImportError("需要安装 scipy 或 librosa 进行重采样")
        
        return audio.astype(np.float32), self.sample_rate
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = None):
        """保存音频文件"""
        if sr is None:
            sr = self.sample_rate
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        sf.write(output_path, audio, sr)
    
    @torch.no_grad()
    def enhance(self, audio: np.ndarray, chunk_size: int = None) -> np.ndarray:
        """
        对音频进行增强
        
        Args:
            audio: 输入音频 (numpy array, 单声道)
            chunk_size: 分块处理大小（秒）。None 表示不分块
            
        Returns:
            enhanced_audio: 增强后的音频
        """
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, L)
        audio_tensor = audio_tensor.to(self.device)
        
        if chunk_size is None or len(audio) <= chunk_size * self.sample_rate:
            enhanced = self.model(audio_tensor)
            enhanced = enhanced.squeeze(0).cpu().numpy()
        else:
            enhanced = self._process_chunks(audio_tensor, chunk_size)
        
        return enhanced
    
    def _process_chunks(self, audio_tensor: torch.Tensor, chunk_size: int) -> np.ndarray:
        """分块处理长音频，带重叠和渐变融合"""
        chunk_samples = int(chunk_size * self.sample_rate)
        hop_samples = chunk_samples // 2  # 50% overlap
        
        total_samples = audio_tensor.shape[1]
        enhanced_audio = np.zeros(total_samples, dtype=np.float32)
        overlap_count = np.zeros(total_samples, dtype=np.float32)
        
        # 创建渐变窗口
        fade_len = hop_samples // 4
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio_tensor[:, start:end]
            
            # Padding
            if chunk.shape[1] < chunk_samples:
                pad_len = chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            
            # 处理
            enhanced_chunk = self.model(chunk)
            enhanced_chunk = enhanced_chunk.squeeze(0).cpu().numpy()
            
            # 截取有效部分
            valid_len = min(chunk_samples, total_samples - start)
            chunk_data = enhanced_chunk[:valid_len]
            
            # 应用渐变窗口
            if start > 0 and valid_len > fade_len:
                chunk_data[:fade_len] *= fade_in
            if start + chunk_samples < total_samples and valid_len > fade_len:
                chunk_data[-fade_len:] *= fade_out
            
            enhanced_audio[start:start + valid_len] += chunk_data
            overlap_count[start:start + valid_len] += 1
            
            start += hop_samples
        
        # 平均重叠区域
        enhanced_audio = enhanced_audio / np.maximum(overlap_count, 1)
        
        return enhanced_audio
    
    def process_file(self, input_path: str, output_path: str, chunk_size: int = None):
        """处理单个音频文件"""
        audio, sr = self.load_audio(input_path)
        enhanced = self.enhance(audio, chunk_size)
        self.save_audio(enhanced, output_path, sr)
    
    def process_directory(self, input_dir: str, output_dir: str, chunk_size: int = None, 
                          suffix: str = "", keep_structure: bool = True):
        """批量处理文件夹中的音频文件"""
        if not os.path.isdir(input_dir):
            raise ValueError(f"输入路径不是文件夹: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集音频文件
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            if keep_structure:
                audio_files.extend(glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
                audio_files.extend(glob(os.path.join(input_dir, '**', f'*{ext.upper()}'), recursive=True))
            else:
                audio_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
                audio_files.extend(glob(os.path.join(input_dir, f'*{ext.upper()}')))
        
        audio_files = sorted(set(audio_files))
        
        if not audio_files:
            print(f"在 {input_dir} 中未找到音频文件")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print("-" * 50)
        
        success_count = 0
        fail_count = 0
        
        for input_path in tqdm(audio_files, desc="处理进度"):
            try:
                if keep_structure:
                    rel_path = os.path.relpath(input_path, input_dir)
                else:
                    rel_path = os.path.basename(input_path)
                
                name, ext = os.path.splitext(rel_path)
                output_filename = f"{name}{suffix}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                output_subdir = os.path.dirname(output_path)
                if output_subdir:
                    os.makedirs(output_subdir, exist_ok=True)
                
                self.process_file(input_path, output_path, chunk_size)
                success_count += 1
                
            except Exception as e:
                print(f"\n处理失败: {input_path}")
                print(f"  错误: {e}")
                fail_count += 1
        
        print("-" * 50)
        print(f"处理完成! 成功: {success_count}, 失败: {fail_count}")


def main():
    parser = argparse.ArgumentParser(
        description='GTCRN 48kHz 语音增强推理 (优化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python gtcrn_48k_inference.py -m model.tar -i noisy.wav -o enhanced.wav
  
  # 使用 GPU
  python gtcrn_48k_inference.py -m model.tar -i noisy.wav -o enhanced.wav -d cuda
  
  # 批量处理文件夹
  python gtcrn_48k_inference.py -m model.tar --input_dir ./noisy --output_dir ./enhanced
  
  # 批量处理并添加后缀
  python gtcrn_48k_inference.py -m model.tar --input_dir ./noisy --output_dir ./enhanced --suffix "_clean"
  
  # 指定 ERB 参数 (需要匹配训练配置)
  python gtcrn_48k_inference.py -m model.tar -i noisy.wav -o enhanced.wav --erb1 128 --erb2 256

注意:
  ERB 参数 (--erb1, --erb2) 需要与训练时的配置一致，否则模型权重无法正确加载。
  默认配置: erb_subband_1=128 (4kHz), erb_subband_2=256
        """
    )
    
    # 模型参数
    parser.add_argument('--model_path', '-m', type=str, 
                        default="./onnx_models/best_model_035.tar",
                        help='预训练模型路径 (.tar, .pth, .pt)')
    
    # ERB 参数 (用于匹配不同训练配置)
    parser.add_argument('--erb1', type=int, default=128,
                        help='ERB 不压缩频带数 (4kHz=128, 3kHz=96)')
    parser.add_argument('--erb2', type=int, default=256,
                        help='ERB 压缩频带数')
    
    # 单文件模式
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入音频文件路径 (单文件模式)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出音频文件路径 (单文件模式)')
    
    # 批量处理模式
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入文件夹路径 (批量模式)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出文件夹路径 (批量模式)')
    parser.add_argument('--suffix', type=str, default="",
                        help='输出文件名后缀，如 "_enhanced"')
    parser.add_argument('--no_keep_structure', action='store_true',
                        help='不保持子文件夹结构')
    
    # 通用参数
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='计算设备 (默认: cpu)')
    parser.add_argument('--chunk_size', '-c', type=int, default=5,
                        help='分块处理大小（秒），用于处理长音频 (默认: 5)')
    
    args = parser.parse_args()
    
    # 检查参数
    single_file_mode = args.input is not None or args.output is not None
    batch_mode = args.input_dir is not None or args.output_dir is not None
    
    if single_file_mode and batch_mode:
        parser.error("不能同时使用单文件模式 (-i/-o) 和批量模式 (--input_dir/--output_dir)")
    
    if single_file_mode:
        if args.input is None or args.output is None:
            parser.error("单文件模式需要同时指定 -i 和 -o")
    
    if batch_mode:
        if args.input_dir is None or args.output_dir is None:
            parser.error("批量模式需要同时指定 --input_dir 和 --output_dir")
    
    if not single_file_mode and not batch_mode:
        parser.error("请指定输入输出: 单文件模式 (-i/-o) 或批量模式 (--input_dir/--output_dir)")
    
    # 创建推理器
    inferencer = GTCRNInference(
        args.model_path, 
        args.device,
        erb_subband_1=args.erb1,
        erb_subband_2=args.erb2
    )
    
    # 处理
    if single_file_mode:
        print(f"正在处理: {args.input}")
        audio, sr = inferencer.load_audio(args.input)
        print(f"  音频长度: {len(audio)/sr:.2f}秒")
        inferencer.process_file(args.input, args.output, args.chunk_size)
        print(f"已保存到: {args.output}")
    else:
        inferencer.process_directory(
            args.input_dir, 
            args.output_dir, 
            args.chunk_size,
            suffix=args.suffix,
            keep_structure=not args.no_keep_structure
        )
    
    print("处理完成!")


if __name__ == "__main__":
    main()