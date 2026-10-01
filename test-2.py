import numpy
import matplotlib.pyplot as plt
from modFreq import chirpMod

nFFTs = 1024

#chirp
A = 1.0
ipp = 400.0e-6
dc = 10.0 #in percent (%)
sr_tx = 20.0e6
sr_rx = 2.5e6
# The central frequency will define the Chirp sweep (ascending or descending)
fc = 0.0e6
bw = 1.0e6
td_ = 0.0 # 5.2 us
window_ = 'R'
mode_f_ = 0.0
phi_ = 0.0
rep_ = 250.0

chirp, full_chirp = chirpMod(A, 
                            ipp, 
                            dc, 
                            sr_tx, 
                            sr_rx, 
                            fc, 
                            bw, 
                            t_d = td_, 
                            window = window_, 
                            mode_f = mode_f_, 
                            phi = phi_)

t = [i for i in range(len(chirp))]

plt.plot(t,numpy.real(chirp))
plt.plot(t,numpy.imag(chirp))
plt.plot(t,numpy.abs(chirp))
plt.show()

#step_frequency = int(fs/nFFTs)
freq = numpy.fft.fftfreq(nFFTs, 1/sr_tx) #(nFFTs samples, steps)
freq = numpy.fft.fftshift(freq)

fft_volt = numpy.fft.fft(chirp, n=nFFTs)
fft_volt = fft_volt.astype(numpy.dtype('complex'))
fft_volt = numpy.fft.fftshift(fft_volt) 



plt.plot(freq,fft_volt)
plt.plot(freq,numpy.real(fft_volt))
plt.plot(freq,numpy.imag(fft_volt))
plt.plot(freq,numpy.abs(fft_volt))
plt.show()

spc = fft_volt*numpy.conjugate(fft_volt)

plt.plot(freq,numpy.abs(spc))
plt.show()


full_c = chirp
#full_c = c
#full_c = numpy.hstack((c, numpy.zeros(int(2*n))))


#full_c_real = numpy.real(full_c)
#full_c_imag = numpy.imag(full_c)

#plt.plot(time,full_c_real[0])
#plt.plot(time,full_c_imag[0])
#plt.show()

