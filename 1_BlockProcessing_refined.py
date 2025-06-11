import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig


# 1. Generate test input (1 s, 440 Hz @16 kHz)
fs   = 16000
t    = np.linspace(0, 1, fs, endpoint=False)
x_n  = 0.5*np.sin(2*np.pi*440*t)

# 2. Create (or pad) a 128-tap filter
#    Here we just make an identity filter for sanity-check
#h_n  = np.zeros(128); h_n[0] = 1.0
h_n = sig.firwin(numtaps=128, cutoff=3000, fs=fs)

# 3. OLA parameters from CATSE
M    = len(h_n)    # 128
L    = 64          # hop size
N    = 256         # FFT size (>= L+M-1)

# 4. Pad filter and precompute its FFT
h_pad = np.pad(h_n, (0, N - M))
Hf    = rfft(h_pad)

# 5. Prepare output buffer (zeroed)
y     = np.zeros(len(x_n) + M - 1)

# 6. Block-by-block FFT–filter–iFFT
num_blocks = int(np.ceil(len(x_n)/L))
for i in range(num_blocks):
    start = i*L
    chunk = x_n[start:start+L]
    # pad final chunk to exactly L
    if len(chunk) < L:
        chunk = np.pad(chunk, (0, L-len(chunk)))
    # pad to FFT length
    x_pad = np.pad(chunk, (0, N-L))

    # STFT → mask/filter → iSTFT
    Xf     = rfft(x_pad)
    Yf     = Xf * Hf            # replace "*Hf" with "*mask" for SE
    y_block= irfft(Yf)

    # overlap-add into y[], trimming the last block
    end = min(start+N, len(y))
    y[start:end] += y_block[:end-start]

# 7. Truncate to true convolution length
y = y[:len(x_n) + M - 1]

# 8. Verify against direct convolution
y_direct = np.convolve(x_n, h_n)
print("Max abs difference:", np.max(np.abs(y - y_direct)))
# → should print something like 4.4e-16
