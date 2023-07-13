# ---------------------- Spectrum Plot -------------------------------

def spectrum_plot(A):

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            u_fft[i,j,:] = abs(np.fft.fft(A[i,j,:]))

    u_fft_mean = np.mean(u_fft, axis = [0,1])

    # Display grid
    plot.grid(True, which ="both")
      
    # Linear X axis, Logarithmic Y axis
    plot.semilogy(range(100), u_fft_mean[0:15] )
      
    # Provide the title for the semilogy plot
    plot.title('Prediction Spectra')
      
    # Give x axis label for the semilogy plot
    plot.xlabel('k')
      
    # Give y axis label for the semilogy plot
    plot.ylabel('Spectra')
      
    # Display the semilogy plot
    plot.savefig('Spectra.png')

