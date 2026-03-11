import numpy as np
from scipy.io import wavfile
import struct
import concurrent.futures
import os

ENCODED_FILE = "encoded.pyflac"

def bytes_to_bits(byte_data):
    '''
    Converts bytes to a list of bits.
    '''
    bits = []
    for byte in byte_data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def zigzag_decode(x):
    '''
    Zigzag decoding.
    '''
    x = int(x)
    if x % 2 == 0:
        return x // 2
    else:
        return -(x + 1) // 2


def rice_decode(byte_data, k, num_samples):
    '''
    Rice decoding.
    '''
    decoded = []

    bits = bytes_to_bits(byte_data)
    i = 0
    
    while len(decoded) < num_samples and i < len(bits):
        # Read unary: count zeros until a one is found
        q = 0
        while i < len(bits) and bits[i] == 0:
            q += 1
            i += 1
        
        if i >= len(bits):
            break
        
        # Skip the one from unary
        i += 1
        
        # Read k bits for the remainder
        if i + k > len(bits):
            break
        
        r = 0
        for j in range(k):
            r = (r << 1) | bits[i]
            i += 1
        
        # Reconstruct folded value
        folded = (q << k) | r
        
        # Apply zigzag decoding
        sample = zigzag_decode(folded)
        decoded.append(sample)
    
    return np.array(decoded, dtype=np.int64)


def lpc_synthesis(residual, coefs):
    '''
    Reconstructs the signal from the residual and LPC coefficients.
    '''
    order = len(coefs)
    N = len(residual)
    reconstructed = np.zeros(N, dtype=np.float64)
    
    # Copy the first 'order' samples
    if order > 0:
        reconstructed[:order] = residual[:order].astype(np.float64)
    
    for n in range(order, N):
        prediction = np.dot(coefs, reconstructed[n-order:n][::-1])
        
        if not np.isfinite(prediction):
            prediction = 0.0
        
        # Round the prediction before adding the residual
        prediction = np.round(prediction)

        # Reconstruct the sample
        reconstructed[n] = residual[n] + prediction
    
    return reconstructed


def mid_side_decode(mid, side, bits_per_sample):
    '''
    Converts Mid-Side back to L/R
    r = |Side| % 2 (residual lost in division)
    '''
    mid = mid.astype(np.int64)
    side = side.astype(np.int64)
    
    r = np.abs(side) % 2
    
    left = mid + (side + r) // 2
    right = mid + (r - side) // 2
    
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    return left.astype(dtype), right.astype(dtype)


def process_single_frame_decode(args):
    '''
    Helper function for ProcessPoolExecutor.
    '''
    byte_data, k, length, coefs, bits_per_sample = args
    residual = rice_decode(byte_data, k, length).astype(np.int64)
    reconstructed = lpc_synthesis(residual.astype(np.float64), coefs.astype(np.float64))
    
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    samples = np.round(reconstructed).astype(dtype)
    return samples[:length]


def decode_frames(frames_data, use_mid_side, bits_per_sample, leave_one_core=False):
    '''
    Decodes the processed frames, applying LPC synthesis and Rice decoding.
    Parallel execution using threads or processes.
    '''
    num_channels = len(frames_data)
    all_channels = []
    
    for ch in range(num_channels):
        ch_frames = frames_data[ch]
        all_samples = []
        num_frames = len(ch_frames)
        channel_name = "Mid" if (use_mid_side and ch == 0) else "Side" if (use_mid_side and ch == 1) else f"Channel {ch+1}"
        
        print(f"\nDecoding {num_frames} frames in parallel for {channel_name}...")
        
        # Prepare data for parallel processing
        frames_to_process = []
        for frame in ch_frames:
            frames_to_process.append((frame['bytes'], frame['k'], frame['length'], frame['coefs'], bits_per_sample))
        
        # Decide number of workers based on CPU
        max_workers = os.cpu_count() - 1 if leave_one_core else os.cpu_count()
        max_workers = min(max_workers, num_frames)
        
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, samples in enumerate(executor.map(process_single_frame_decode, frames_to_process), start=1):
                    all_samples.append(samples)
                    print(f"\rFrame {i}/{num_frames} decoded", end='', flush=True)
        except Exception as e:
            # Fallback to threads if process execution fails
            print(f"Error executing with processes, using threads instead. Error: {e}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, samples in enumerate(executor.map(process_single_frame_decode, frames_to_process), start=1):
                    all_samples.append(samples)
                    print(f"\rFrame {i}/{num_frames} decoded", end='', flush=True)
        
        print()
        all_channels.append(np.concatenate(all_samples))
    
    if use_mid_side and num_channels == 2:
        print("\nConverting Mid-Side to L/R...")
        left, right = mid_side_decode(all_channels[0], all_channels[1], bits_per_sample)
        return np.stack([left, right], axis=-1)
    elif num_channels == 1:
        return all_channels[0]
    else:
        # Pack channels into matrix (N, 2)
        min_len = min(len(ch) for ch in all_channels)
        stacked = np.stack([ch[:min_len] for ch in all_channels], axis=-1)
        return stacked


def load_audio_encoded(filename):
    '''
    Loads the encoded binary file (supports mono and stereo).
    '''
    with open(filename, 'rb') as f:
        # Read header
        sample_rate = struct.unpack('<I', f.read(4))[0]
        predictor_order = struct.unpack('<H', f.read(2))[0]
        frame_size = struct.unpack('<H', f.read(2))[0]
        num_channels = struct.unpack('<B', f.read(1))[0]
        bits_per_sample = struct.unpack('<B', f.read(1))[0]
        use_mid_side = struct.unpack('<B', f.read(1))[0]
        num_frames = struct.unpack('<I', f.read(4))[0]

        print(f"Sample rate: {sample_rate} Hz")
        print(f"Bits per sample: {bits_per_sample}")
        print(f"Predictor order: {predictor_order}")
        print(f"Frame size: {frame_size}")
        print(f"Channels: {num_channels}")

        # frames_data[channel][frame]
        frames_data = [[] for _ in range(num_channels)]

        for i in range(num_frames):
            for ch in range(num_channels):
                k = struct.unpack('<B', f.read(1))[0]
                padding = struct.unpack('<B', f.read(1))[0]
                length = struct.unpack('<H', f.read(2))[0]
                byte_length = struct.unpack('<H', f.read(2))[0]
                coefs = np.zeros(predictor_order, dtype=np.float32)
                for j in range(predictor_order):
                    coefs[j] = struct.unpack('<f', f.read(4))[0]
                byte_data = f.read(byte_length)
                frames_data[ch].append({
                    'k': k,
                    'padding': padding,
                    'length': length,
                    'bytes': byte_data,
                    'coefs': coefs
                })
            print(f"\rLoading frames {i+1}/{num_frames}", end='', flush=True)
        print()
        return sample_rate, predictor_order, frame_size, frames_data, num_channels, use_mid_side, bits_per_sample


if __name__ == "__main__":
    try:
        # Load encoded file
        fs, order, frame_size, frames_data, num_channels, use_mid_side, bits_per_sample = load_audio_encoded(ENCODED_FILE)
        # Decode
        audio_data = decode_frames(frames_data, use_mid_side, bits_per_sample)
        # Save WAV
        wavfile.write("Decoded_Audio.wav", fs, audio_data)
        print("\nDecoding completed")
    except FileNotFoundError:
        print(f"Error: File '{ENCODED_FILE}' not found")
    except Exception as e:
        print(f"Error during decoding: {e}")