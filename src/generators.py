from numpy.random import default_rng
rng = default_rng()

def generate_wavy_circle_contour(x0,y0,radius,amplitude,smoothness,length):
    """Generate a awvy circle contour"""
    t = np.linspace(0,2*math.pi,length).reshape((length,1))
    f = np.exp(-smoothness*length*np.abs(np.fft.fftfreq(length))).reshape((length,1))
    circle = radius * np.cos(t) + 1j * radius * np.sin(t)
    s = circle + x0 + 1j * y0
    s = s + amplitude * rng.normal(0,0.1,size=(length,1)) * circle
    s = np.fft.ifftn(f*np.fft.fftn(s))
    return np.real(s),np.imag(s)

x, y = generate_wavy_circle_contour(1,1,1,3,0.5,128)
plt.plot(x,y)
