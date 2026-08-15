import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    """Object-Oriented framework for generating and manipulating signals."""
    def __init__(self, t):
        self.t = t
        self.signal = None

    def gaussian(self, a):
        """Part 1: Generates the Gaussian signal x(t) = exp(-a * t^2)"""
        self.signal = np.exp(-a * self.t**2)
        return self.signal

    def time_shift(self, t0):
        """
        Part 3: Constructs the shifted signal y(t) = x(t - t0).
        Important: Uses interpolation for the OOP framework to shift
        the signal without manual array rolling.
        """
        if self.signal is None:
            raise ValueError("Generate a signal first.")
            
        # x(t - t0) means evaluating the original signal at t - t0
        shifted_signal = np.interp(self.t - t0, self.t, self.signal, left=0, right=0)
        
        # Return a new SignalGenerator instance containing the shifted signal
        shifted_gen = SignalGenerator(self.t)
        shifted_gen.signal = shifted_signal
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
            # Use np.trapz for numerical integration as required
            X[i] = np.trapz(integrand, self.t)
        return X

def main():
    # ---------------------------------------------------------
    # Part 2: Constructing the Original Signal
    # ---------------------------------------------------------
    # 1. Define the time axis t in [-5, 5] using at least 2000 samples
    t = np.linspace(-5, 5, 2000)
    
    # 2. Generate the signal x(t) = exp(-t^2) by setting a = 1
    sig_gen = SignalGenerator(t)
    x = sig_gen.gaussian(a=1)

    # ---------------------------------------------------------
    # Part 3: Time-Shifting the Signal
    # ---------------------------------------------------------
    # Let the time shift be t0 = 1, y(t) = x(t - 1)
    t0 = 1
    shifted_gen = sig_gen.time_shift(t0)
    y = shifted_gen.signal

    # ---------------------------------------------------------
    # Part 4: Continuous Fourier Transform
    # ---------------------------------------------------------
    # Frequency axis f in [-10, 10] with at least 1000 samples
    f = np.linspace(-10, 10, 1000)
    analyzer = CFTAnalyzer(t, f)
    
    print("Computing CFTs (this may take a few seconds)...")
    X = analyzer.compute_cft(x)
    Y = analyzer.compute_cft(y)

    # ---------------------------------------------------------
    # Part 5: Numerical Verification (Plots)
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))
    
    # 1. Plot magnitude spectra |X(f)| and |Y(f)|
    plt.subplot(1, 2, 1)
    plt.plot(f, np.abs(X), label='|X(f)| (Original)', color='blue', linewidth=2)
    plt.plot(f, np.abs(Y), label='|Y(f)| (Shifted)', color='red', linestyle='dashed', linewidth=2)
    plt.title("Magnitude Spectra")
    plt.xlabel("Frequency (f)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    # 2. Plot phase spectra
    plt.subplot(1, 2, 2)
    
    # To avoid chaotic phase plots where magnitude is effectively zero, 
    # we'll mask out the negligible frequency components.
    threshold = 0.01 * np.max(np.abs(X))
    mask = np.abs(X) > threshold
    
    # Predicted phase: ∠Y(f) = ∠X(f) - 2*pi*f*t0
    predicted_phase = np.angle(X) - 2 * np.pi * f * t0
    
    plt.plot(f[mask], np.angle(X)[mask], label='∠X(f)', color='blue', marker='o', linestyle='none', markersize=4)
    plt.plot(f[mask], np.angle(Y)[mask], label='∠Y(f) (Measured)', color='red', marker='x', linestyle='none', markersize=4)
    
    # Wrap predicted phase to [-pi, pi] for plotting comparison
    predicted_phase_wrapped = (predicted_phase + np.pi) % (2 * np.pi) - np.pi
    plt.plot(f[mask], predicted_phase_wrapped[mask], label='∠Y(f) (Predicted)', color='green', marker='.', linestyle='none', markersize=2)
    
    plt.title("Phase Spectra (Significant Magnitudes Only)")
    plt.xlabel("Frequency (f)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = "Online03_Spectra.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved spectra plot to {plot_filename}\n")

    # ---------------------------------------------------------
    # Part 6: Error Analysis (MSE and Phase Difference)
    # ---------------------------------------------------------
    print("Error Analysis:")
    print("-" * 50)
    
    # (a) Mean Squared Error (MSE) of Magnitude
    mse_mag = np.mean((np.abs(X) - np.abs(Y))**2)
    print(f"(a) MSE of Magnitude: {mse_mag:.6e}")
    if mse_mag < 1e-4:
        print("    -> Comment: The magnitude MSE is extremely small, confirming the time-shift property that |X(f)| = |Y(f)|.")
    else:
        print("    -> Comment: The magnitude MSE is large, indicating an error.")

    # (b) Phase Difference Error
    # ∠Y(f) = ∠X(f) - 2*pi*f*t0
    # We must wrap the phase difference to [-pi, pi] because phase is circular (e.g. pi and -pi are the same)
    phase_diff = np.angle(Y) - predicted_phase
    phase_diff_wrapped = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    
    # The noise in phase at frequencies where the magnitude is near zero will dominate the MSE.
    # Therefore, we calculate both the overall MSE and the MSE for the significant frequencies.
    mse_phase_all = np.mean(phase_diff_wrapped**2)
    mse_phase_sig = np.mean(phase_diff_wrapped[mask]**2)
    
    print(f"\n(b) MSE of Phase (All frequencies): {mse_phase_all:.6e}")
    print(f"    MSE of Phase (Significant magnitudes): {mse_phase_sig:.6e}")
    
    if mse_phase_sig < 1e-4:
        print("    -> Comment: In the regions where the signal exists, the phase MSE is extremely small.")
        print("                This confirms the time-shift property ∠Y(f) = ∠X(f) - 2πft_0.")
        print("                (The overall MSE is larger purely due to numerical noise at near-zero magnitudes).")
    else:
        print("    -> Comment: The phase MSE is large, indicating an error.")

if __name__ == "__main__":
    main()
