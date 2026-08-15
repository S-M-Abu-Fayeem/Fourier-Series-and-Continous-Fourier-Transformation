import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread


class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)

        # Continuous spatial coordinate vectors, both spanning [-1, 1]
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


class CFT2D:
    """Computes the 2D Continuous Fourier Transform of a ContinuousImage
    using separable numerical (trapezoidal) integration."""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

        # Frequency axes conjugate to x and y (given), spanning the full
        # Nyquist range implied by the sample spacing (dx, dy). This is
        # what lets the transform represent fine, edge-scale spatial
        # detail instead of only very coarse (near-DC) variation.
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        """
        Compute the real and imaginary parts of the 2D Continuous Fourier
        Transform of self.I, using SEPARABLE trapezoidal integration:

            Re{F(u,v)} =  Integral Integral I(x,y) cos(2*pi*(u*x + v*y)) dx dy
            Im{F(u,v)} = -Integral Integral I(x,y) sin(2*pi*(u*x + v*y)) dx dy

        Do NOT evaluate this as a direct 4-nested-loop double integral over
        (x, y, u, v) -- that is O(N^4) and will not finish in reasonable
        time. Instead exploit separability: expand cos(2*pi*(ux+vy)) and
        sin(2*pi*(ux+vy)) with the angle-sum identities, first integrate
        over x for every (y, u) pair, then integrate the result over y for
        every (u, v) pair. Each of the two stages is an O(N^3) operation
        (an O(N) numerical integral, repeated over an N x N grid), which is
        what makes this tractable.

        Use self.u and self.v (NOT self.x/self.y) as the frequency axes --
        they were already computed for you in __init__.

        Use np.trapz(..., axis=...) for the integration -- no built-in
        FFT/DFT routine (np.fft, scipy.fft, ...) may be used anywhere in
        this method.

        Returns
        -------
        real, imag : two 2D numpy arrays, each of shape self.I.shape
        """
        # TODO: implement this method
        Ny , Nx = self.I.shape
        Nu = len(self.u)
        Nv = len(self.v)
        Cx = np.zeros((Ny,Nu),dtype=float)
        
        Sx = np.zeros((Ny,Nu), dtype=float)

        for idx_u in range(Nu):
            cos_ux = np.cos(2*np.pi * self.u[idx_u]*self.x)
            sin_ux = np.sin(2*np.pi * self.u[idx_u]*self.x)

            for idx_y in range(Ny):
                Cx[idx_y, idx_u] = np.trapz(self.I[idx_y,:] * cos_ux, self.x)
                Sx[idx_y, idx_u] = np.trapz(self.I[idx_y,:] * sin_ux, self.x)

        real = np.zeros_like(self.I)
        imag = np.zeros_like(self.I)

        for idx_v in range(Nv):
            cos_vy = np.cos(2*np.pi * self.v[idx_v]*self.y)
            sin_vy = np.sin(2*np.pi * self.v[idx_v]*self.y)

            for idx_u in range(Nu):
                
                real[idx_v, idx_u] = (np.trapz( Cx[:,idx_u] * cos_vy, self.y)) - (np.trapz(Sx[:,idx_u] * sin_vy, self.y))
                imag[idx_v, idx_u] = -(np.trapz( Sx[:,idx_u] * cos_vy, self.y)) - (np.trapz(Cx[:,idx_u] * sin_vy, self.y))
            
            
        return real, imag

    def plot_magnitude(self):
        """
        Plot the log-scaled magnitude spectrum of the 2D CFT computed by
        compute_cft(), i.e. plt.imshow(np.log(1 + magnitude), ...) where
        magnitude = sqrt(real**2 + imag**2). Purely for your own visual
        debugging -- not called by the command-line entry point below.
        """
        # TODO: implement this method
        real , imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        image_in_log = np.log(1+magnitude)
        plt.imshow(image_in_log, cmap='gray')
        plt.title("2D CFT Magnitude Spectrum (log-scaled)")
        plt.axis('off')
        plt.show()

    def verify_parseval(self, real, imag):
        """
        Verifies Parseval's Theorem for 2D CFT:
        Integral |I(x,y)|^2 dx dy = Integral |F(u,v)|^2 du dv
        """
        E_x = np.trapz(self.I**2, self.x, axis=1)
        spatial_energy = np.trapz(E_x, self.y, axis=0)

        magnitude_sq = real**2 + imag**2
        E_u = np.trapz(magnitude_sq, self.u, axis=1)
        freq_energy = np.trapz(E_u, self.v, axis=0)
        
        rel_error = np.abs(spatial_energy - freq_energy) / spatial_energy
        print(f"[Parseval 2D] E_spatial: {spatial_energy:.6e}, E_freq: {freq_energy:.6e}, Rel Error: {rel_error:.6e}")
        return rel_error

    def verify_differentiation_x(self, real, imag):
        """
        Verifies the differentiation property: dI/dx <=> j*2*pi*u * F(u,v)
        """
        U, _ = np.meshgrid(self.u, self.v)
        
        diff_real = -2 * np.pi * U * imag
        diff_imag = 2 * np.pi * U * real
        
        icft = InverseCFT2D(diff_real, diff_imag, self.u, self.v, self.x, self.y)
        I_prime_hat = icft.reconstruct()
        
        dx = self.x[1] - self.x[0]
        I_prime_true = np.gradient(self.I, dx, axis=1)
        
        mse = np.mean(np.abs(I_prime_true - I_prime_hat)**2)
        print(f"[Differentiation X] MSE between spatial gradient and freq-domain derivative: {mse:.6e}")
        return mse

    def verify_spatial_shift(self, real, imag, x0, y0):
        """
        Verifies the spatial shift property:
        I(x-x0, y-y0) <=> F(u,v) * exp(-j*2*pi*(u*x0 + v*y0))
        """
        U, V = np.meshgrid(self.u, self.v)
        phase_shift = np.exp(-1j * 2 * np.pi * (U * x0 + V * y0))
        
        F = real + 1j * imag
        F_shifted = F * phase_shift
        
        icft = InverseCFT2D(np.real(F_shifted), np.imag(F_shifted), self.u, self.v, self.x, self.y)
        I_shifted_hat = icft.reconstruct()
        
        mse = np.mean(np.abs(self.I - I_shifted_hat)**2)
        
        E_original = np.trapz(np.trapz(np.abs(F)**2, self.u, axis=1), self.v, axis=0)
        E_shifted = np.trapz(np.trapz(np.abs(F_shifted)**2, self.u, axis=1), self.v, axis=0)
        
        print(f"[Spatial Shift x0={x0}, y0={y0}] MSE vs unshifted: {mse:.6e}, Energy Preserved Ratio: {E_shifted/E_original:.6f}")
        return mse

    def run_all_verifications(self, real, imag):
        print("\n--- Running All CFT Properties Verifications ---")
        self.verify_parseval(real, imag)
        self.verify_differentiation_x(real, imag)
        self.verify_spatial_shift(real, imag, x0=0.1, y0=0.1)
        print("------------------------------------------------\n")

class FrequencyFilter:
    """Applies frequency-domain filtering operations. (Given)"""

    def high_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2

        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag


class InverseCFT2D:
    """Reconstructs the spatial-domain image from a (filtered) 2D frequency
    spectrum using separable numerical integration."""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform the inverse 2D Continuous Fourier Transform:

            I(x,y) = Integral Integral F(u,v) exp(j*2*pi*(u*x + v*y)) du dv

        using the same separable-integration strategy as compute_cft():
        expand the complex exponential into cos/sin via Euler's identity,
        integrate over v first (for every (y, u) pair), then integrate
        that result over u (for every (x, y) pair). Use np.trapz.

        self.real, self.imag are the (possibly filtered) frequency-domain
        components; self.u, self.v are the frequency axes they were
        computed on; self.x, self.y are the spatial axes to reconstruct
        onto.

        Returns
        -------
        image : 2D numpy array of shape (len(self.y), len(self.x))
            The reconstructed real-valued spatial-domain signal. Note
            that after a high-pass filter this is NOT a valid image on
            its own (it will contain negative values, since the DC/
            low-frequency component that carried the average brightness
            has been removed) -- see the command-line entry point below
            for how it gets turned into a displayable edge map.
        """
        # TODO: implement this method
        Ny = len(self.y)
        Nx = len(self.x)

        Nu = len(self.u)
        Nv = len(self.v)

        P = np.zeros((Ny,Nu),dtype=float)
        Q = np.zeros((Ny,Nu),dtype=float)

        for idx_y in range(Ny):
            cos_vy = np.cos(2*np.pi*self.v*self.y[idx_y])
            sin_vy = np.sin(2*np.pi*self.v*self.y[idx_y])
            for idx_u in range(Nu):
                A = np.trapz(self.real[:,idx_u] * cos_vy, self.v)
                B = np.trapz(self.imag[:,idx_u] * sin_vy, self.v)
                P[idx_y,idx_u] = A - B

                C = np.trapz(self.real[:,idx_u]*sin_vy, self.v)
                D = np.trapz(self.imag[:,idx_u]*cos_vy, self.v)
                Q[idx_y,idx_u] = C +D

        image = np.zeros((Ny,Nx))
        for idx_x in range(Nx):
            cos_ux = np.cos(2* np.pi * self.u * self.x[idx_x])
            sin_ux = np.sin(2* np.pi * self.u * self.x[idx_x])
            for idx_y in range(Ny):
                image[idx_y,idx_x] = np.trapz( P[idx_y,:] * cos_ux, self.u) - np.trapz(Q[idx_y,:] * sin_ux, self.u) 
        
        return image
               


# =====================================================
# Command-line entry point (given -- do not modify)
# Usage: python3 cft_edge_detector.py <input_image_path> <output_image_path> [cutoff]
# =====================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 cft_edge_detector.py <input_image_path> <output_image_path> [cutoff]")
        print("Example: python3 cft_edge_detector.py pikachu.png pikachu_edges.png 15")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 15

    img = ContinuousImage(input_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    
    # Run properties verifications
    cft2d.run_all_verifications(real, imag)

    filt = FrequencyFilter()
    real_f, imag_f = filt.high_pass(real, imag, cutoff)

    icft2d = InverseCFT2D(real_f, imag_f, cft2d.u, cft2d.v, img.x, img.y)
    edges = icft2d.reconstruct()

    edge_map = np.abs(edges)
    if edge_map.max() > 0:
        edge_map = edge_map / edge_map.max()
    edge_map = 1 - edge_map  # invert: edges black, background white

    plt.imsave(output_path, edge_map, cmap='gray')
    print(f"Saved edge map to {output_path}")
