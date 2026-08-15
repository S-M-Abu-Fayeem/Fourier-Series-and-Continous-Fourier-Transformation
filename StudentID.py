import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    """Object-Oriented framework for generating and manipulating signals."""
    def __init__(self, t):
        self.t = t
        self.signal = np.zeros_like(t, dtype=complex)

    def square(self):
        """Generates a standard square wave (rect function): 1 for |t| <= 0.5"""
        return np.where(np.abs(self.t) <= 0.5, 1.0 + 0j, 0.0 + 0j)

    def triangle(self):
        """Generates a standard triangle wave: 1 - |t| for |t| <= 1"""
        return np.where(np.abs(self.t) <= 1.0, 1.0 - np.abs(self.t), 0.0 + 0j)

    def set_signal(self, sig):
        self.signal = sig

    def time_compress(self, a):
        """
        Compresses the time axis of the signal by a factor of a: x(a*t).
        Uses interpolation to maintain the OOP framework rules.
        """
        if self.signal is None:
            raise ValueError("Generate a signal first.")
        
        # Interpolate real and imaginary parts separately
        real_part = np.interp(a * self.t, self.t, np.real(self.signal), left=0, right=0)
        imag_part = np.interp(a * self.t, self.t, np.imag(self.signal), left=0, right=0)
        
        compressed_gen = SignalGenerator(self.t)
        compressed_gen.set_signal(real_part + 1j * imag_part)
        return compressed_gen

    def phase_shift(self, f0):
        """
        Shifts the phase of the signal by 2*pi*f0*t: x(t) * exp(j * 2*pi * f0 * t)
        """
        if self.signal is None:
            raise ValueError("Generate a signal first.")
            
        shifted_signal = self.signal * np.exp(1j * 2 * np.pi * f0 * self.t)
        
        shifted_gen = SignalGenerator(self.t)
        shifted_gen.set_signal(shifted_signal)
        return shifted_gen

class CFTAnalyzer:
    """Object-Oriented framework for Continuous Fourier Transform analysis."""
    def __init__(self, t, f):
        self.t = t
        self.f = f

    def compute_cft(self, signal):
        """
        Computes the Continuous Fourier Transform of the given signal
        using numerical integration (np.trapz).
        """
        X = np.zeros(len(self.f), dtype=complex)
        for i, freq in enumerate(self.f):
            integrand = signal * np.exp(-1j * 2 * np.pi * freq * self.t)
            X[i] = np.trapz(integrand, self.t)
        return X

def main():
    # ---------------------------------------------------------
    # 1. Define Time Axis
    # ---------------------------------------------------------
    t = np.linspace(-5, 5, 5000)  # Using 5000 samples for high accuracy
    
    # ---------------------------------------------------------
    # 2. Construct Original Signal x(t) = Square(t) + Triangle(t)
    # ---------------------------------------------------------
    sig_gen = SignalGenerator(t)
    sq = sig_gen.square()
    tri = sig_gen.triangle()
    
    # Set combined signal
    x = sq + tri
    sig_gen.set_signal(x)
    
    # ---------------------------------------------------------
    # 3. Construct Modified Signal y(t)
    # ---------------------------------------------------------
    f0 = 10
    a = 10
    
    # y(t) = x(a*t) * exp(j * 2*pi * f0 * t)
    # Operation (ii) then (i)
    compressed_gen = sig_gen.time_compress(a)
    y_gen = compressed_gen.phase_shift(f0)
    y = y_gen.signal
    
    # ---------------------------------------------------------
    # 4. Compute CFTs
    # ---------------------------------------------------------
    f = np.linspace(-10, 10, 2000)
    analyzer = CFTAnalyzer(t, f)
    
    print("Computing CFT of x(t) and y(t)...")
    X = analyzer.compute_cft(x)
    Y = analyzer.compute_cft(y)
    
    # ---------------------------------------------------------
    # 5. Theoretical Prediction
    # ---------------------------------------------------------
    # The property states: Y(f) = 1/|a| * X((f - f0)/a)
    # We must evaluate X at f_target = (f - f0) / a
    f_target = (f - f0) / a
    
    # Interpolate the real and imaginary parts of X(f) to find X at f_target
    X_target_real = np.interp(f_target, f, np.real(X))
    X_target_imag = np.interp(f_target, f, np.imag(X))
    X_target = X_target_real + 1j * X_target_imag
    
    # Apply the 1/|a| scaling factor
    Y_pred = (1 / np.abs(a)) * X_target
    
    # ---------------------------------------------------------
    # 6. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))
    
    # Magnitude Plot
    plt.subplot(1, 2, 1)
    plt.plot(f, np.abs(Y), label='|Y(f)| (Empirical)', color='red', linewidth=3, alpha=0.6)
    plt.plot(f, np.abs(Y_pred), label='1/|a| * |X((f-f0)/a)| (Theoretical)', color='black', linestyle='dashed', linewidth=2)
    plt.title("Magnitude Spectrum Verification")
    plt.xlabel("Frequency (f)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    
    # Phase Plot
    plt.subplot(1, 2, 2)
    # Mask out phase where magnitude is very small to avoid random noise
    threshold = 0.05 * np.max(np.abs(Y))
    mask = np.abs(Y) > threshold
    
    plt.plot(f[mask], np.angle(Y)[mask], label='∠Y(f)', color='red', marker='o', linestyle='none', markersize=6, alpha=0.6)
    plt.plot(f[mask], np.angle(Y_pred)[mask], label='∠X((f-f0)/a)', color='black', marker='x', linestyle='none', markersize=6)
    plt.title("Phase Spectrum Verification (At Peaks)")
    plt.xlabel("Frequency (f)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("Online_Verification.png")
    plt.close()
    print("Saved plots to Online_Verification.png")
    
    # ---------------------------------------------------------
    # 7. Error Analysis
    # ---------------------------------------------------------
    print("\nError Analysis:")
    print("-" * 50)
    
    # MSE Magnitude
    mse_mag = np.mean((np.abs(Y) - np.abs(Y_pred))**2)
    print(f"MSE Magnitude: {mse_mag:.6e}")
    
    # MSE Phase
    # Phase difference must be wrapped to [-pi, pi] before squaring
    phase_diff = np.angle(Y) - np.angle(Y_pred)
    phase_diff_wrapped = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    
    mse_phase_all = np.mean(phase_diff_wrapped**2)
    mse_phase_sig = np.mean(phase_diff_wrapped[mask]**2)
    
    print(f"MSE Phase (All Frequencies): {mse_phase_all:.6e}")
    print(f"MSE Phase (Significant Magnitudes Only): {mse_phase_sig:.6e}")
    
    if mse_mag < 1e-4 and mse_phase_sig < 1e-4:
        print("\nConclusion: Both MSE values are within the acceptable tolerance range.")
        print("The Time Scaling and Frequency Shifting properties are verified.")
    else:
        print("\nConclusion: Errors are large. Check implementation.")

if __name__ == "__main__":
    main()
