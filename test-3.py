import numpy
import matplotlib.pyplot as plt
from scipy import signal
from modFreq import chirpMod

fs = 10.0e7
T = 5.0e-6
n = int(fs*T) 
nFFTs = 4096

nProfiles = 128 #24

time = numpy.linspace(0, T, 128)

#sin wave
A = 1.0
frequency = 0.5e7
phase = 2*numpy.pi*frequency
c = A*numpy.exp(1j*phase*time)



#c = c*B

plt.plot(time,c)
plt.show()

#chirp
A = 1.0
ipp = 400.0e-6
dc = 10.0
sr_tx = 20.0e6
sr_rx = 2.5e6
# The central frequency will define the Chirp sweep (ascending or descending)
fc = 0.0e6
bw = 1.0e6
td_ = 0.0 # 5.2 us
window_ = 'B'
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

plt.plot(t,chirp)
plt.show()

#step_frequency = int(fs/nFFTs)
freq = numpy.fft.fftfreq(nFFTs, 1/sr_tx) #(nFFTs samples, steps)
freq = numpy.fft.fftshift(freq)

full_c = chirp
#full_c = full_chirp
#full_c = c
#full_c = numpy.hstack((c, numpy.zeros(int(2*n))))



full_c = numpy.tile(full_c, (nProfiles, 1))     #(24,50) <> (nProfiles,nHeights)
full_c = full_c.transpose()                     #(50,24) <> (nHeights,nProfiles)

sin_add = numpy.zeros(full_c.shape)

rep = 250
sines = numpy.tile(c, (rep,1)) # (200,128)

#window to sin
alpha = 1.0
B = A*signal.windows.tukey(rep, alpha)

sines = numpy.multiply(sines, B[:, numpy.newaxis])

print('full_c.shape',full_c.shape)
print('sin_add.shape',sin_add.shape)
print('sines.shape',sines.shape)

index = 400 #half of heights
sin_add[int(index-rep/2):int(index+rep/2), :] = sines
print('sin_add.shape',sin_add.shape)

full_c = full_c + sin_add


#full_c_real = numpy.real(full_c)
#full_c_imag = numpy.imag(full_c)

#plt.plot(time,full_c_real[0])
#plt.plot(time,full_c_imag[0])
#plt.show()

fft_volt = numpy.fft.fft(full_c, n=nFFTs, axis=1)   #axis=0 ->(256,24)
                                                    #axis=1 ->(50,256)
fft_volt = fft_volt.astype(numpy.dtype('complex'))
fft_volt = numpy.fft.fftshift(fft_volt, axes = (1,)) 
spc = fft_volt * numpy.conjugate(fft_volt) # (800, 128)

# print('spc.shape',spc.shape)



#spc = spc.real #(nHeights,nFFTs)
spc = numpy.abs(spc) #(nHeights,nFFTs)


#power profile
spc = numpy.where(numpy.isfinite(spc), spc, numpy.nan)
# avg = numpy.nanmean(spc, axis=1)
avg = numpy.nanmean(spc, axis=1)

#print('avg.shape',avg.shape)
#print('spc.shape',spc.shape)

'''plt.pcolormesh(x[:z[n].T.shape[1]], #x, # Change for adjust index in DP experiment spectraPlot 
                                       y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap),
                                       )'''

#doppler spectrum
x = freq                                                #x axis: frequency
y = numpy.linspace(0, len(spc[:,0]), len(spc[:,0]))     #y axis: heights 
#linspace -> (start, end, step)


#power profile
plt.plot(y, avg)
plt.show()


plt.pcolormesh(x, y, spc, cmap=plt.get_cmap('jet'))
plt.show()
