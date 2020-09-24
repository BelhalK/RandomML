import matplotlib.pyplot as plt
import numpy as np

def detect_blur_fft(image, size=60, thresh=10, vis=False):
	# (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	# compute the FFT to find the frequency transform 
	# shift the zero frequency component
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	if vis:
		# Magnitude spectrum of the transform
		magnitude = 20 * np.log(np.abs(fftShift))

		# display the original input image
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])

		# display the magnitude image
		ax[1].imshow(magnitude, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])

		plt.show()

	# Remove low frequencies
	# Apply the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	# compute the magnitude spectrum of the reconstructed image
	# Compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	
	# Blurry if magnitudes is less than the threshold value (Bool mean <= thresh)
	return (mean, mean <= thresh)