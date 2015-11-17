"""Parameters of the 32 channel system."""

samp_freq = 250e6 # Hz; Sampling frequency.
nchan = 32        # Total channels.
fft_len = 1024    # FFT length.
nfreq = fft_len / 2 # number of frequencies
databits = 32     # raw data bits, either real part or imaginary part.
LO_freq = 935     # MHz; Frequency of Local Oscillator.
int_time = 1      # second; Integration time.
dot_bit = 14      # Bit length of fractional part.
delta_f = samp_freq / fft_len / 1e6 # MHz
block_size = (2 + fft_len/2 * nchan**2 / 4) * (databits / 8) # Bytes, length of one time record