"""
GTCRN 16kHz Inference Script with Evaluation Metrics
用于语音增强的推理脚本 - 16kHz模型版本（带评估指标）

特点:
    - 使用16kHz的GTCRN模型
    - 支持48kHz输入音频（自动降采样到16kHz）
    - 降采样后的文件会缓存，避免重复处理
    - 输出音频会上采样回原始采样率
    - 支持DNS-MOS, PESQ, SI-SNR评估指标
    - 支持多线程并行处理

使用方法:
    # 处理单个文件
    python gtcrn_16khz.py --model_path model.tar --input audio.wav --output enhanced.wav

    # 批量处理文件夹
    python gtcrn_16khz.py --model_path model.tar --input_dir ./noisy_wavs --output_dir ./enhanced_wavs

    # 带评估的批量处理（需要clean参考）
    python gtcrn_16khz.py --model_path model.tar --input_dir ./noisy --output_dir ./enhanced --clean_dir ./clean --evaluate

    python .\gtcrn_16khz.py --model_path .\onnx_models\model_trained_on_dns3.tar --input_dir .\test_wavs\noisy_16k\ --output_dir .\test_wavs\output_16khz --clean_dir .\test_wavs\clean_16khz\ --evaluate
"""
import os
import hashlib
import argparse
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import soundfile as sf
from glob import glob
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import urllib.request
import warnings
warnings.filterwarnings('ignore')


# ======================= 评估指标工具类 =======================

class MetricsEvaluator:
    """评估指标计算器"""

    def __init__(self, dns_mos_model_path=None):
        """
        初始化评估器

        Args:
            dns_mos_model_path: DNS-MOS ONNX模型路径，None则自动下载
        """
        self.dns_mos_model_path = dns_mos_model_path
        self.onnx_session = None

        # 尝试导入评估库
        try:
            from pesq import pesq
            self.pesq_func = pesq
            self.has_pesq = True
        except ImportError:
            print("警告: 未安装pesq库，PESQ指标将不可用。安装: pip install pesq")
            self.has_pesq = False

        # 初始化DNS-MOS
        if dns_mos_model_path is not None or self._check_dns_mos_available():
            self._init_dns_mos()

    def _check_dns_mos_available(self):
        """检查是否可以使用DNS-MOS"""
        try:
            import onnxruntime
            return True
        except ImportError:
            print("警告: 未安装onnxruntime，DNS-MOS指标将不可用。安装: pip install onnxruntime")
            return False

    def _download_dns_mos_model(self, save_path):
        """下载DNS-MOS ONNX模型"""
        # DNS-MOS模型URL (DNSMOS P.835)
        model_url = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"

        print(f"正在下载DNS-MOS模型...")
        print(f"URL: {model_url}")

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            urllib.request.urlretrieve(model_url, save_path)
            print(f"DNS-MOS模型已下载到: {save_path}")
            return True
        except Exception as e:
            print(f"下载DNS-MOS模型失败: {e}")
            print("将跳过DNS-MOS评估")
            return False

    def _init_dns_mos(self):
        """初始化DNS-MOS模型"""
        try:
            import onnxruntime as ort

            # 确定模型路径
            if self.dns_mos_model_path is None:
                self.dns_mos_model_path = "./dns_mos_models/sig_bak_ovr.onnx"

            # 如果模型不存在，尝试下载
            if not os.path.exists(self.dns_mos_model_path):
                if not self._download_dns_mos_model(self.dns_mos_model_path):
                    self.onnx_session = None
                    return

            # 加载ONNX模型
            self.onnx_session = ort.InferenceSession(self.dns_mos_model_path)
            print(f"DNS-MOS模型已加载: {self.dns_mos_model_path}")

        except Exception as e:
            print(f"初始化DNS-MOS模型失败: {e}")
            self.onnx_session = None

    def calculate_si_snr(self, reference, estimate):
        """
        计算SI-SNR (Scale-Invariant Signal-to-Noise Ratio)

        Args:
            reference: 参考信号 (numpy array)
            estimate: 估计信号 (numpy array)

        Returns:
            SI-SNR值 (dB)
        """
        eps = 1e-8

        # 确保长度一致
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]

        # 去均值
        reference = reference - np.mean(reference)
        estimate = estimate - np.mean(estimate)

        # 计算投影
        reference_energy = np.sum(reference ** 2) + eps
        optimal_scaling = np.sum(reference * estimate) / reference_energy
        projection = optimal_scaling * reference

        # 计算噪声
        noise = estimate - projection

        # 计算SI-SNR
        si_snr = 10 * np.log10(np.sum(projection ** 2) / (np.sum(noise ** 2) + eps) + eps)

        return si_snr

    def calculate_pesq(self, reference, estimate, sr=16000):
        """
        计算PESQ (Perceptual Evaluation of Speech Quality)

        Args:
            reference: 参考信号
            estimate: 估计信号
            sr: 采样率 (8000 or 16000)

        Returns:
            PESQ值 (1.0-4.5)
        """
        if not self.has_pesq:
            return None

        try:
            # 确保长度一致
            min_len = min(len(reference), len(estimate))
            reference = reference[:min_len]
            estimate = estimate[:min_len]

            # PESQ只支持8kHz和16kHz
            if sr not in [8000, 16000]:
                print(f"警告: PESQ不支持{sr}Hz采样率，跳过")
                return None

            mode = 'wb' if sr == 16000 else 'nb'
            score = self.pesq_func(sr, reference, estimate, mode)
            return score
        except Exception as e:
            print(f"计算PESQ失败: {e}")
            return None

    def calculate_dns_mos(self, audio, sr=16000):
        """
        计算DNS-MOS (Deep Noise Suppression Mean Opinion Score)

        Args:
            audio: 音频信号
            sr: 采样率

        Returns:
            dict: {'OVRL': overall_mos, 'SIG': signal_mos, 'BAK': background_mos}
        """
        if self.onnx_session is None:
            return {'OVRL': None, 'SIG': None, 'BAK': None}

        try:
            # DNS-MOS需要16kHz
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # DNS-MOS模型需要固定长度输入 (9.01秒 = 144160 samples at 16kHz)
            target_length = 144160
            audio = audio.astype(np.float32)

            # 如果音频太短，进行padding
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            # 如果音频太长，分段处理并取平均
            elif len(audio) > target_length:
                # 分段处理
                num_segments = int(np.ceil(len(audio) / target_length))
                scores = {'SIG': [], 'BAK': [], 'OVRL': []}

                for i in range(num_segments):
                    start = i * target_length
                    end = min((i + 1) * target_length, len(audio))
                    segment = audio[start:end]

                    # Padding if needed
                    if len(segment) < target_length:
                        segment = np.pad(segment, (0, target_length - len(segment)), mode='constant')

                    segment = segment.reshape(1, -1)

                    # 运行推理
                    ort_inputs = {self.onnx_session.get_inputs()[0].name: segment}
                    ort_outs = self.onnx_session.run(None, ort_inputs)

                    scores['SIG'].append(float(ort_outs[0][0][0]))
                    scores['BAK'].append(float(ort_outs[0][0][1]))
                    scores['OVRL'].append(float(ort_outs[0][0][2]))

                # 返回平均值
                return {
                    'SIG': np.mean(scores['SIG']),
                    'BAK': np.mean(scores['BAK']),
                    'OVRL': np.mean(scores['OVRL'])
                }

            # 正常长度，直接处理
            audio = audio.reshape(1, -1)

            # 运行推理
            ort_inputs = {self.onnx_session.get_inputs()[0].name: audio}
            ort_outs = self.onnx_session.run(None, ort_inputs)

            # 解析输出 [SIG, BAK, OVRL]
            return {
                'SIG': float(ort_outs[0][0][0]),
                'BAK': float(ort_outs[0][0][1]),
                'OVRL': float(ort_outs[0][0][2])
            }
        except Exception as e:
            print(f"计算DNS-MOS失败: {e}")
            return {'OVRL': None, 'SIG': None, 'BAK': None}

    def evaluate(self, clean_audio, enhanced_audio, noisy_audio=None, sr=16000):
        """
        综合评估

        Args:
            clean_audio: 干净参考音频
            enhanced_audio: 增强后音频
            noisy_audio: 噪声音频（可选）
            sr: 采样率

        Returns:
            dict: 包含所有指标的字典
        """
        results = {}

        # SI-SNR
        results['SI-SNR'] = self.calculate_si_snr(clean_audio, enhanced_audio)

        # PESQ
        results['PESQ'] = self.calculate_pesq(clean_audio, enhanced_audio, sr)

        # DNS-MOS
        dns_mos = self.calculate_dns_mos(enhanced_audio, sr)
        results['OVRL'] = dns_mos['OVRL']
        results['SIG'] = dns_mos['SIG']
        results['BAK'] = dns_mos['BAK']

        # P808_MOS (使用OVRL作为近似)
        results['P808_MOS'] = dns_mos['OVRL']

        return results


# ======================= 模型定义部分 (16kHz) =======================

class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)

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

        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
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
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
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
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    """GTCRN 16kHz 模型"""
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        # 16kHz配置:
        # n_fft=512 -> nfreqs=257
        # ERB: erb_subband_1=65, erb_subband_2=64
        # Total ERB bins: 65 + 64 = 129
        # After 2x freq downsampling in encoder: 129/4 = 32.25 -> 33
        self.erb = ERB(65, 64, nfft=n_fft, high_lim=8000, fs=16000)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()
        
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        
        self.decoder = Decoder()

        self.mask = Mask()

    def forward(self, x):
        """
        x: (B, L) - raw waveform at 16kHz
        """
        device = x.device
        n_samples = x.shape[1]
        
        stft_kwargs = {'n_fft': self.n_fft, 'hop_length': self.hop_len, 'win_length': self.win_len,
                       'window': torch.hann_window(self.win_len).pow(0.5).to(device), 'onesided': True}
        
        spec = torch.stft(x, **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)  # (B, F, T, 2), F=257
        
        spec_real = spec[..., 0].permute(0,2,1)  # (B, T, F)
        spec_imag = spec[..., 1].permute(0,2,1)  # (B, T, F)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
        
        spec = spec.permute(0,3,2,1)  # (B,2,T,F)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs = self.encoder(feat)
        
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output


# ======================= 推理工具类 =======================

class GTCRN16kInference:
    """GTCRN 16kHz 推理类"""

    # 支持的音频格式
    SUPPORTED_FORMATS = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.wma', '.aac']

    # 模型采样率
    MODEL_SAMPLE_RATE = 16000

    def __init__(self, model_path: str, device: str = None, cache_dir: str = "./cache_16k",
                 enable_evaluation: bool = False, dns_mos_model_path: str = None):
        """
        初始化推理器

        Args:
            model_path: 预训练模型路径 (.tar, .pth, .pt)
            device: 设备 ('cuda', 'cpu', 或 None 自动选择)
            cache_dir: 降采样缓存目录
            enable_evaluation: 是否启用评估功能
            dns_mos_model_path: DNS-MOS模型路径
        """
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 缓存目录
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"缓存目录: {cache_dir}")

        # 创建模型
        self.model = GTCRN()

        # 加载权重
        self._load_checkpoint(model_path)

        # 设置为评估模式
        self.model.eval()
        self.model.to(self.device)

        # 评估器
        self.enable_evaluation = enable_evaluation
        self.evaluator = None
        if enable_evaluation:
            self.evaluator = MetricsEvaluator(dns_mos_model_path)
            print("评估功能已启用")

        # 线程锁（用于多线程写入CSV）
        self.csv_lock = Lock()
    
    def _load_checkpoint(self, model_path: str):
        """加载模型权重"""
        print(f"正在加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 尝试不同的权重格式
        if isinstance(checkpoint, dict):
            # 常见的checkpoint格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            else:
                # 假设整个dict就是state_dict
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
        
        # 加载权重
        self.model.load_state_dict(new_state_dict, strict=True)
        print("模型加载成功!")
        
        # 打印额外信息（如果有）
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"  Loss: {checkpoint['loss']:.6f}")
    
    def _get_cache_path(self, audio_path: str) -> str:
        """
        获取缓存文件路径
        基于原始文件路径生成唯一的缓存文件名
        """
        # 使用文件路径的MD5哈希作为缓存文件名
        path_hash = hashlib.md5(os.path.abspath(audio_path).encode()).hexdigest()
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        cache_filename = f"{original_name}_{path_hash}_16k.wav"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样音频"""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            raise ImportError("需要安装 librosa 进行重采样: pip install librosa")
    
    def _ensure_16k_cached(self, audio_path: str) -> tuple:
        """
        确保16kHz缓存存在
        
        Returns:
            cache_path: 缓存文件路径
            original_sr: 原始采样率
        """
        cache_path = self._get_cache_path(audio_path)
        
        # 读取原始音频获取采样率
        audio_info = sf.info(audio_path)
        original_sr = audio_info.samplerate
        
        # 检查缓存是否存在
        if os.path.exists(cache_path):
            print(f"  [缓存命中] {os.path.basename(cache_path)}")
            return cache_path, original_sr
        
        # 缓存不存在，需要降采样
        print(f"  [降采样] {original_sr}Hz -> {self.MODEL_SAMPLE_RATE}Hz")
        
        # 加载原始音频
        audio, sr = sf.read(audio_path)
        
        # 如果是立体声，转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 降采样到16kHz
        if sr != self.MODEL_SAMPLE_RATE:
            audio_16k = self._resample_audio(audio, sr, self.MODEL_SAMPLE_RATE)
        else:
            audio_16k = audio
        
        # 保存缓存
        sf.write(cache_path, audio_16k, self.MODEL_SAMPLE_RATE)
        print(f"  [缓存保存] {os.path.basename(cache_path)}")
        
        return cache_path, original_sr
    
    def load_audio_16k(self, audio_path: str) -> tuple:
        """
        加载音频文件（16kHz版本）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            audio_16k: 16kHz音频数据 (numpy array)
            original_sr: 原始采样率
        """
        # 确保缓存存在
        cache_path, original_sr = self._ensure_16k_cached(audio_path)
        
        # 从缓存加载16kHz音频
        audio_16k, _ = sf.read(cache_path)
        
        return audio_16k, original_sr
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: int):
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            output_path: 输出路径
            sr: 采样率
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        sf.write(output_path, audio, sr)
    
    @torch.no_grad()
    def enhance(self, audio: np.ndarray, chunk_size: int = None) -> np.ndarray:
        """
        对音频进行增强
        
        Args:
            audio: 输入音频 (numpy array, 单声道, 16kHz)
            chunk_size: 分块处理大小（秒）。None表示不分块
            
        Returns:
            enhanced_audio: 增强后的音频 (16kHz)
        """
        # 转换为tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, L)
        audio_tensor = audio_tensor.to(self.device)
        
        if chunk_size is None or len(audio) <= chunk_size * self.MODEL_SAMPLE_RATE:
            # 直接处理整个音频
            enhanced = self.model(audio_tensor)
            enhanced = enhanced.squeeze(0).cpu().numpy()
        else:
            # 分块处理长音频
            enhanced = self._process_chunks(audio_tensor, chunk_size)
        
        return enhanced
    
    def _process_chunks(self, audio_tensor: torch.Tensor, chunk_size: int) -> np.ndarray:
        """分块处理长音频"""
        chunk_samples = int(chunk_size * self.MODEL_SAMPLE_RATE)
        hop_samples = chunk_samples // 2  # 50% overlap
        
        total_samples = audio_tensor.shape[1]
        enhanced_audio = np.zeros(total_samples)
        overlap_count = np.zeros(total_samples)
        
        # 滑动窗口处理
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio_tensor[:, start:end]
            
            # 如果chunk太短，进行padding
            if chunk.shape[1] < chunk_samples:
                pad_len = chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            
            # 处理
            enhanced_chunk = self.model(chunk)
            enhanced_chunk = enhanced_chunk.squeeze(0).cpu().numpy()
            
            # 截取有效部分
            valid_len = min(chunk_samples, total_samples - start)
            enhanced_audio[start:start+valid_len] += enhanced_chunk[:valid_len]
            overlap_count[start:start+valid_len] += 1
            
            start += hop_samples
        
        # 平均重叠区域
        enhanced_audio = enhanced_audio / np.maximum(overlap_count, 1)
        
        return enhanced_audio
    
    def process_file(self, input_path: str, output_path: str, chunk_size: int = None,
                     output_sr: int = None, clean_path: str = None):
        """
        处理单个音频文件

        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            chunk_size: 分块大小（秒）
            output_sr: 输出采样率，None表示与输入相同
            clean_path: 干净参考音频路径（用于评估）

        Returns:
            dict: 评估结果（如果启用评估）
        """
        # 加载音频（会自动处理缓存）
        audio_16k, original_sr = self.load_audio_16k(input_path)

        # 增强
        enhanced_16k = self.enhance(audio_16k, chunk_size)

        # 确定输出采样率
        if output_sr is None:
            output_sr = original_sr

        # 如果需要，上采样回原始采样率
        if output_sr != self.MODEL_SAMPLE_RATE:
            print(f"  [上采样] {self.MODEL_SAMPLE_RATE}Hz -> {output_sr}Hz")
            enhanced = self._resample_audio(enhanced_16k, self.MODEL_SAMPLE_RATE, output_sr)
        else:
            enhanced = enhanced_16k

        # 保存
        self.save_audio(enhanced, output_path, output_sr)

        # 评估（如果启用且提供了clean参考）
        eval_results = None
        if self.enable_evaluation and clean_path is not None and self.evaluator is not None:
            try:
                # 加载clean音频
                clean_16k, _ = self.load_audio_16k(clean_path)

                # 评估
                eval_results = self.evaluator.evaluate(clean_16k, enhanced_16k, audio_16k, self.MODEL_SAMPLE_RATE)
                eval_results['filename'] = os.path.basename(input_path)

            except Exception as e:
                print(f"  [评估失败] {e}")

        return eval_results

    def _process_single_file_worker(self, args):
        """多线程工作函数"""
        input_path, output_path, chunk_size, output_sr, clean_path = args

        try:
            eval_results = self.process_file(input_path, output_path, chunk_size, output_sr, clean_path)
            return (True, input_path, eval_results)
        except Exception as e:
            return (False, input_path, str(e))

    def process_directory(self, input_dir: str, output_dir: str, chunk_size: int = None,
                          suffix: str = "", keep_structure: bool = True, output_sr: int = None,
                          clean_dir: str = None, num_workers: int = 4, csv_output: str = None):
        """
        批量处理文件夹中的音频文件（支持多线程）

        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径
            chunk_size: 分块大小（秒）
            suffix: 输出文件名后缀 (如 "_enhanced")
            keep_structure: 是否保持子文件夹结构
            output_sr: 输出采样率，None表示与输入相同
            clean_dir: 干净参考音频文件夹（用于评估）
            num_workers: 线程数
            csv_output: CSV输出路径
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"输入路径不是文件夹: {input_dir}")

        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)

        # 收集所有音频文件
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            if keep_structure:
                # 递归搜索
                audio_files.extend(glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
                audio_files.extend(glob(os.path.join(input_dir, '**', f'*{ext.upper()}'), recursive=True))
            else:
                # 仅搜索顶层目录
                audio_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
                audio_files.extend(glob(os.path.join(input_dir, f'*{ext.upper()}')))

        # 去重并排序
        audio_files = sorted(set(audio_files))

        if not audio_files:
            print(f"在 {input_dir} 中未找到音频文件")
            return

        print(f"找到 {len(audio_files)} 个音频文件")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"缓存目录: {self.cache_dir}")
        if clean_dir:
            print(f"参考目录: {clean_dir}")
        print(f"线程数: {num_workers}")
        print("-" * 50)

        # 准备任务列表
        tasks = []
        for input_path in audio_files:
            # 计算输出路径
            if keep_structure:
                rel_path = os.path.relpath(input_path, input_dir)
            else:
                rel_path = os.path.basename(input_path)

            # 添加后缀
            name, ext = os.path.splitext(rel_path)
            output_filename = f"{name}{suffix}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            # 确保输出子目录存在
            output_subdir = os.path.dirname(output_path)
            if output_subdir:
                os.makedirs(output_subdir, exist_ok=True)

            # 查找对应的clean文件
            clean_path = None
            if clean_dir and self.enable_evaluation:
                # 尝试多种匹配方式
                base_name = os.path.basename(input_path)
                # 替换 noisy -> clean
                clean_name = base_name.replace('_noisy', '_clean').replace('noisy', 'clean')
                potential_clean_path = os.path.join(clean_dir, clean_name)

                if os.path.exists(potential_clean_path):
                    clean_path = potential_clean_path
                else:
                    # 尝试直接使用相同文件名
                    potential_clean_path = os.path.join(clean_dir, base_name)
                    if os.path.exists(potential_clean_path):
                        clean_path = potential_clean_path

            tasks.append((input_path, output_path, chunk_size, output_sr, clean_path))

        # 多线程处理
        success_count = 0
        fail_count = 0
        all_eval_results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self._process_single_file_worker, task) for task in tasks]

            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
                success, input_path, result = future.result()

                if success:
                    success_count += 1
                    if result is not None:
                        all_eval_results.append(result)
                else:
                    fail_count += 1
                    print(f"\n处理失败: {input_path}")
                    print(f"  错误: {result}")

        print("-" * 50)
        print(f"处理完成! 成功: {success_count}, 失败: {fail_count}")

        # 保存评估结果到CSV
        if all_eval_results and csv_output:
            self._save_evaluation_csv(all_eval_results, csv_output)

    def _save_evaluation_csv(self, results, csv_path):
        """保存评估结果到CSV"""
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                # 写入表头
                fieldnames = ['filename', 'SI-SNR', 'PESQ', 'OVRL', 'SIG', 'BAK', 'P808_MOS']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # 写入数据
                for result in results:
                    row = {
                        'filename': result.get('filename', ''),
                        'SI-SNR': f"{result.get('SI-SNR', 0):.2f}" if result.get('SI-SNR') is not None else '',
                        'PESQ': f"{result.get('PESQ', 0):.3f}" if result.get('PESQ') is not None else '',
                        'OVRL': f"{result.get('OVRL', 0):.3f}" if result.get('OVRL') is not None else '',
                        'SIG': f"{result.get('SIG', 0):.3f}" if result.get('SIG') is not None else '',
                        'BAK': f"{result.get('BAK', 0):.3f}" if result.get('BAK') is not None else '',
                        'P808_MOS': f"{result.get('P808_MOS', 0):.3f}" if result.get('P808_MOS') is not None else ''
                    }
                    writer.writerow(row)

            print(f"\n评估结果已保存到: {csv_path}")

            # 计算平均值
            if results:
                avg_si_snr = np.mean([r['SI-SNR'] for r in results if r.get('SI-SNR') is not None])
                avg_pesq = np.mean([r['PESQ'] for r in results if r.get('PESQ') is not None])
                avg_ovrl = np.mean([r['OVRL'] for r in results if r.get('OVRL') is not None])
                avg_sig = np.mean([r['SIG'] for r in results if r.get('SIG') is not None])
                avg_bak = np.mean([r['BAK'] for r in results if r.get('BAK') is not None])

                print("\n平均评估指标:")
                print(f"  SI-SNR: {avg_si_snr:.2f} dB")
                print(f"  PESQ: {avg_pesq:.3f}")
                print(f"  OVRL: {avg_ovrl:.3f}")
                print(f"  SIG: {avg_sig:.3f}")
                print(f"  BAK: {avg_bak:.3f}")

        except Exception as e:
            print(f"保存CSV失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='GTCRN 16kHz 语音增强推理（支持评估和多线程）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python gtcrn_16khz.py -m model.tar -i noisy.wav -o enhanced.wav

  # 批量处理文件夹
  python gtcrn_16khz.py -m model.tar --input_dir ./noisy --output_dir ./enhanced

  # 批量处理并评估（需要clean参考）
  python gtcrn_16khz.py -m model.tar --input_dir ./test_wavs/noisy_16k --output_dir ./test_wavs/output --clean_dir ./test_wavs/clean_16khz --evaluate --csv_output ./test_wavs/output/evaluation_results.csv

  # 使用多线程加速处理
  python gtcrn_16khz.py -m model.tar --input_dir ./noisy --output_dir ./enhanced --num_workers 8

注意:
  - 输入音频可以是任意采样率（如48kHz），会自动降采样到16kHz处理
  - 降采样后的文件会缓存到指定目录，重复处理时无需重新降采样
  - 默认输出采样率与输入相同（会自动上采样）
  - 评估功能需要安装: pip install pesq onnxruntime
        """
    )

    # 模型参数
    parser.add_argument('--model_path', '-m', type=str,
                        required=True,
                        help='预训练模型路径 (.tar, .pth, .pt)')

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
                        help='不保持子文件夹结构，所有输出文件放在同一目录')

    # 评估参数
    parser.add_argument('--evaluate', action='store_true',
                        help='启用评估功能（需要提供clean_dir）')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='干净参考音频文件夹路径（用于评估）')
    parser.add_argument('--csv_output', type=str, default=None,
                        help='评估结果CSV输出路径')
    parser.add_argument('--dns_mos_model', type=str, default=None,
                        help='DNS-MOS ONNX模型路径（默认自动下载）')

    # 缓存参数
    parser.add_argument('--cache_dir', type=str, default="./cache_16k",
                        help='16kHz降采样缓存目录 (默认: ./cache_16k)')

    # 通用参数
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='计算设备 (默认: cpu)')
    parser.add_argument('--chunk_size', '-c', type=int, default=5,
                        help='分块处理大小（秒），用于处理长音频 (默认: 5)')
    parser.add_argument('--output_sr', type=int, default=None,
                        help='输出采样率，默认与输入相同 (设为16000可跳过上采样)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='多线程处理的线程数 (默认: 4)')

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

    # 评估参数检查
    if args.evaluate and args.clean_dir is None:
        parser.error("启用评估功能需要指定 --clean_dir")

    # 创建推理器
    inferencer = GTCRN16kInference(
        args.model_path,
        args.device,
        args.cache_dir,
        enable_evaluation=args.evaluate,
        dns_mos_model_path=args.dns_mos_model
    )

    # 处理
    if single_file_mode:
        # 单文件模式
        print(f"正在处理: {args.input}")
        inferencer.process_file(args.input, args.output, args.chunk_size, args.output_sr)
        print(f"已保存到: {args.output}")
    else:
        # 批量处理模式
        # 如果启用评估但未指定CSV输出路径，使用默认路径
        csv_output = args.csv_output
        if args.evaluate and csv_output is None:
            csv_output = os.path.join(args.output_dir, "evaluation_results.csv")

        inferencer.process_directory(
            args.input_dir,
            args.output_dir,
            args.chunk_size,
            suffix=args.suffix,
            keep_structure=not args.no_keep_structure,
            output_sr=args.output_sr,
            clean_dir=args.clean_dir,
            num_workers=args.num_workers,
            csv_output=csv_output
        )

    print("处理完成!")


if __name__ == "__main__":
    main()