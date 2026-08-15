from matplotlib.backend_bases import key_press_handler
import numpy as np

from svg_utils import load_svg_path
from epicycle_animation import save_outputs


class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        """
        Step 1: Store the sampled signal and set up everything the other
        methods will need.

        Parameters
        ----------
        t : 1D numpy array, shape (M,)
            Uniformly spaced sample times covering ONE FULL PERIOD of the
            signal, as a *closed* interval: t[0] == 0 and t[-1] == T (the
            period). This is exactly what svg_utils.load_svg_path(...)
            returns.
        signal : 1D complex numpy array, shape (M,)
            signal[i] = f(t[i]) = x(t[i]) + 1j * y(t[i]). Periodic, so
            signal[-1] == signal[0].
        n_harmonics : int (call it N)
            The series will use every integer harmonic n with
            -N <= n <= N (i.e. 2N+1 terms in total -- do not forget the
            negative harmonics).

        You must set at least the following attributes, since the rest of
        this class (and the provided plotting/animation code) expects
        them to exist:
            self.t, self.signal, self.N
            self.T      -- the period (a float)
            self.omega  -- the fundamental angular frequency, 2*pi/T
            self.coeffs -- an (initially empty) dict that will map
                           n -> c_n once calculate_all_coefficients() has
                           been called
        """
        # TODO: implement this method
        self.t = t
        self.signal = signal
        #self.T = t[-1] - t[0]
        self.T = t[-1]
        self.omega = 2*np.pi/self.T
        self.N = n_harmonics
        self.coeffs = {}
        #raise NotImplementedError("Implement __init__")

    def calculate_cn(self, n):
        """
        Step 2: Compute a single complex Fourier coefficient c_n using
        numerical integration (np.trapezoid) over the stored samples
        self.t, self.signal.

            c_n = (1/T) * integral_0^T  f(t) * exp(-j*n*omega*t)  dt

        n may be zero, positive, or negative.
        """
        # TODO: implement this method
        integrand = self.signal * np.exp(-1j * n * self.omega * self.t)
        return_value = (1/self.T) * np.trapz(integrand,self.t)
        return return_value
        #raise NotImplementedError("Implement calculate_cn")

    def calculate_all_coefficients(self):
        """
        Step 3: Populate self.coeffs with c_n for every harmonic
        n = -N, ..., -1, 0, 1, ..., N by repeatedly calling calculate_cn(n).
        """
        # TODO: implement this method
        for n in range (-1*self.N, self.N + 1):
            self.coeffs[n] = self.calculate_cn(n)

        #raise NotImplementedError("Implement calculate_all_coefficients")

    def approximate(self, t):
        """
        Step 4: Reconstruct (an approximation of) the signal at time(s) t
        from the coefficients already stored in self.coeffs:

            f_hat(t) = sum_{n=-N}^{N} c_n * exp(j*n*omega*t)

        t may be a single number or a numpy array of times -- your
        implementation must support both, since the provided
        plotting/animation code calls this both ways.
        """
        # TODO: implement this method
        f_hat = 0j
        for n in range (-1*self.N , self.N + 1):
            f_hat+=self.coeffs[n] * np.exp(1j*n*self.omega*t)
        return f_hat
        #raise NotImplementedError("Implement approximate")


        #python3 task1/fs_redrawer.py task1/svgs/heart.svg 150 heart_comparison.png heart_epicycles.gif
        #python3 task1/fs_redrawer.py task1/svgs/star.svg 150 star_comparison.png star_epicycles.gif
        #python3 task1/fs_redrawer.py task1/svgs/infinity.svg 150 infinity_comparison.png infinity_epicycles.gif
        #python3 task1/fs_redrawer.py task1/svgs/circle.svg 150 circle_comparison.png circle_epicycles.gif

    def prune_harmonics_by_energy(self, r):
        #total_energy = sum(abs(c)**2 for c in self.coeffs.values())

        total_energy = 0.0
        for key, values in self.coeffs.items():
            total_energy+=abs(values)**2

        # Sort harmonics by |c_n|^2 descending
        sorted_harmonics = sorted(self.coeffs.keys(),
                                  key=lambda n: abs(self.coeffs[n])**2,
                                  reverse=True)
        print(type(sorted_harmonics[0]))

        cum_energy = 0.0
        retained = set()
        for n in sorted_harmonics:
            cum_energy += abs(self.coeffs[n])**2
            retained.add(n)
            if cum_energy >= r * total_energy:
                break

        # Zero out discarded harmonics
        for n in self.coeffs:
            if n not in retained:
                self.coeffs[n] = 0.0

        actual_ratio = cum_energy / total_energy
        return len(retained), actual_ratio

    def evaluate_reconstruction_error(self):
        f_hat = self.approximate(self.t)
        mse = np.mean(np.abs(self.signal - f_hat)**2)
        return mse

    def verify_parseval(self):
        """
        Verifies Parseval's Theorem: Total energy in the time domain equals 
        total energy in the frequency domain.
        Formula: (1/T) ∫ |x(t)|^2 dt = Σ |c_n|^2
        """
        # Time-domain energy using trapezoidal integration
        E_time = (1 / self.T) * np.trapz(np.abs(self.signal)**2, self.t)
        # Frequency-domain energy by summing squared magnitudes of coefficients
        E_freq = sum(np.abs(c)**2 for c in self.coeffs.values())
        
        # Calculate relative error to check how close they are
        rel_error = np.abs(E_time - E_freq) / E_time
        print(f"[Parseval] E_time: {E_time:.6e}, E_freq: {E_freq:.6e}, Rel Error: {rel_error:.6e}")
        return rel_error

    def verify_low_pass(self, cutoff):
        """
        Verifies Low-Pass Filtering by truncating the series at a cutoff frequency.
        This demonstrates how discarding high frequencies (sharp corners/details) 
        increases the Mean Squared Error (MSE).
        """
        original_coeffs = self.coeffs.copy()
        
        # Zero out coefficients beyond the cutoff frequency K
        for n in list(self.coeffs.keys()):
            if abs(n) > cutoff:
                self.coeffs[n] = 0.0
                
        # Reconstruct the signal using only the low-frequency components
        z_hat = self.approximate(self.t)
        # Compute MSE between original signal and low-pass filtered signal
        mse = np.mean(np.abs(self.signal - z_hat)**2)
        print(f"[Low-Pass K={cutoff}] MSE: {mse:.6e}")
        
        # Restore the original coefficients
        self.coeffs = original_coeffs
        return mse

    def verify_time_shift(self, tau):
        """
        Verifies the Time-Shift property: 
        x(t - tau) <=> c_n * exp(-j * n * omega * tau)
        Shifting the signal in time only rotates the phase of the coefficients 
        without changing their magnitude (energy is perfectly preserved).
        """
        # Apply the phase shift formula to each coefficient
        shifted_coeffs = {n: c * np.exp(-1j * n * self.omega * tau) for n, c in self.coeffs.items()}
        
        # Reconstruct the shifted signal manually using the modified coefficients
        z_hat = 0j
        for n, c in shifted_coeffs.items():
            z_hat += c * np.exp(1j * n * self.omega * self.t)
            
        # Create the true shifted signal in the time domain using periodic interpolation
        t_shifted = (self.t - tau) % self.T
        signal_real_shifted = np.interp(t_shifted, self.t, np.real(self.signal))
        signal_imag_shifted = np.interp(t_shifted, self.t, np.imag(self.signal))
        signal_shifted_true = signal_real_shifted + 1j * signal_imag_shifted
        
        # MSE against the true shifted signal (should be very small)
        mse_shifted = np.mean(np.abs(signal_shifted_true - z_hat)**2)
        
        # Verify energy preservation: Σ |shifted_c_n|^2 == Σ |original_c_n|^2
        E_original = sum(np.abs(c)**2 for c in self.coeffs.values())
        E_shifted = sum(np.abs(c)**2 for c in shifted_coeffs.values())
        
        print(f"[Time Shift tau={tau:.2f}] MSE vs true shifted signal: {mse_shifted:.6e}, Energy Preserved Ratio: {E_shifted/E_original:.6f}")
        return mse_shifted

    def verify_time_reversal(self):
        """
        Verifies the Time-Reversal property: 
        x(-t) <=> c_{-n}
        Time reversal in the time domain corresponds to swapping the positive 
        and negative harmonic coefficients.
        """
        # Swap c_n with c_{-n}
        reversed_coeffs = {n: self.coeffs[-n] for n in self.coeffs}
        
        # Reconstruct the time-reversed signal
        z_hat = 0j
        for n, c in reversed_coeffs.items():
            z_hat += c * np.exp(1j * n * self.omega * self.t)
            
        # Create the true time-reversed signal in the time domain
        t_reversed = (-self.t) % self.T
        signal_real_reversed = np.interp(t_reversed, self.t, np.real(self.signal))
        signal_imag_reversed = np.interp(t_reversed, self.t, np.imag(self.signal))
        signal_reversed_true = signal_real_reversed + 1j * signal_imag_reversed
            
        # Calculate MSE against the true reversed signal (should be very small)
        mse_reversed = np.mean(np.abs(signal_reversed_true - z_hat)**2)
        print(f"[Time Reversal] MSE vs true reversed signal: {mse_reversed:.6e}")
        return mse_reversed

    def verify_differentiation(self):
        """
        Verifies the Differentiation property: 
        d/dt x(t) <=> j * n * omega * c_n
        The derivative of a signal can be computed by scaling its Fourier 
        coefficients by j*n*ω.
        """
        # Apply the differentiation scaling factor in the frequency domain
        diff_coeffs = {n: 1j * n * self.omega * c for n, c in self.coeffs.items()}
        
        # Reconstruct the derivative signal from the scaled coefficients
        z_prime_hat = 0j
        for n, c in diff_coeffs.items():
            z_prime_hat += c * np.exp(1j * n * self.omega * self.t)
            
        # Calculate the empirical derivative in the time domain using finite differences (np.gradient)
        dt = self.t[1] - self.t[0]
        z_prime_true = np.gradient(self.signal, dt)
        
        # Compare the frequency-domain derivative with the time-domain derivative
        mse = np.mean(np.abs(z_prime_true - z_prime_hat)**2)
        print(f"[Differentiation] MSE between time-domain gradient and freq-domain derivative: {mse:.6e}")
        return mse

    def verify_lanczos_smoothing(self):
        """
        Verifies Lanczos Sigma-factor smoothing.
        By multiplying coefficients by a sinc-like factor (σ_n = sin(nπ/N) / (nπ/N)),
        we can reduce the ringing artifacts (Gibbs phenomenon) near sharp corners.
        """
        original_coeffs = self.coeffs.copy()
        
        # Apply the Lanczos sigma factors
        for n in self.coeffs:
            if n == 0:
                sigma = 1.0
            else:
                x_val = np.pi * n / self.N
                sigma = np.sin(x_val) / x_val
            self.coeffs[n] *= sigma
            
        # Reconstruct the smoothed signal
        z_hat = self.approximate(self.t)
        # Compute MSE to see how much the smoothing altered the signal's geometry
        mse = np.mean(np.abs(self.signal - z_hat)**2)
        print(f"[Lanczos Smoothing] MSE: {mse:.6e}")
        
        # Restore the original coefficients
        self.coeffs = original_coeffs
        return mse

    def run_all_verifications(self):
        """
        Helper method to sequentially execute all the property verifications.
        """
        print("\n--- Running All Fourier Properties Verifications ---")
        self.verify_parseval()
        self.verify_low_pass(cutoff=self.N // 2)
        self.verify_time_shift(tau=self.T / 4)
        self.verify_time_reversal()
        self.verify_differentiation()
        self.verify_lanczos_smoothing()
        print("--------------------------------------------------\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python3 fs_redrawer.py <path_to_svg> [n_harmonics]")
        sys.exit(1)

    svg_path = sys.argv[1]
    N_HARMONICS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    stem = Path(svg_path).stem

    print(f"Target Ratio | Harmonics Retained | Actual Energy Ratio | MSE")
    print(f"------------------------------------------------------------------")

    for r in [0.96, 0.98, 0.99, 1.00]:
        t, z = load_svg_path(svg_path, num_points=1000)
        fs = FourierEpicycles(t, z, n_harmonics=N_HARMONICS)
        fs.calculate_all_coefficients()
        
        if r == 1.00:
            fs.run_all_verifications()

        retained, actual_ratio = fs.prune_harmonics_by_energy(r)
        mse = fs.evaluate_reconstruction_error()

        print(f"{r:.2f}         | {retained:<18d} | {actual_ratio:.4f}              | {mse:.6e}")

        comparison_path = f"{stem}_pruned_{r}.png"
        gif_path = f"{stem}_epicycles_{r}.gif"
        save_outputs(fs, z, comparison_path, gif_path, num_frames=240)

