import numpy as np
from finder import Finder
from streak import model


class Simulator:
    """
    The Simulator class creates images with "realistic" streaks in them:
    streaks have width according to the PSF size, and white noise background added.

    The key parameters to input before making a streak are:
    -x1,x2,y1,y2: the streak start and end points (normalized units!) e.g, x1=0.1, x2=0.2, y1=0, y2=1
    -im_size: the image size as a scalar (makes square images only, for now). Use powers of 2. Default is 512.
    -bg_noise_var: noise variance (counts^2 per pixel). Default is 1.
    -psf_sigma: width parameter of the PSF. default is 2.
    -intensity: counts/unit length (diagonal lines have slightly larger value per pixel)

    You can also turn on/off the source noise (use_source_noise) or background by
    setting bg_noise_var to zero.

    The streaks are automatically input to the Finder object, that returns (hopefully)
    with the right streak parameters.
    """

    def __init__(self):

        # objects
        self.finder = Finder()

        # outputs
        self.image_clean = []
        self.image = []
        self.psf = []

        # switches
        self.im_size = 512  # assume we want to make square images
        self.intensity = 10.0
        self.bg_noise_var = 1.0
        self.use_source_noise = True
        self.psf_sigma = 2.0

        # these parameters are in units of image size
        self.x1 = 0.0
        self.x2 = 1.0
        self.y1 = 0.0
        self.y2 = 1.0

        self.verbosity = 1

    @property
    def L(self):
        # return math.sqrt((self.x2-self.x1)**2+(self.y2-self.y1)**2)
        x1 = min(max(self.x1, 0), 1)
        x2 = min(max(self.x2, 0), 1)
        y1 = min(max(self.y1, 0), 1)
        y2 = min(max(self.y2, 0), 1)

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @property
    def th(self):
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def a(self):
        if self.x1 == self.x2:
            return np.nan
        else:
            return (self.y2 - self.y1) / (self.x2 - self.x1)

    @property
    def b(self):
        return self.y2 - self.a * self.x2

    @property
    def x0(self):
        if self.x1 == self.x2:
            return self.x1
        elif self.a == 0:
            return np.nan
        else:
            return -self.b / self.a

    @property
    def midpoint_x(self):
        return (self.x2 + self.x1) / 2

    @property
    def midpoint_y(self):
        return (self.y2 + self.y1) / 2

    @property
    def trig_factor(self):
        return max(abs(np.cos(np.radians(self.th))), abs(np.sin(np.radians(self.th))))

    def clear(self):
        self.image_clean = []
        self.image = []
        self.psf = []

    def is_vertical(self):

        val1 = 45 <= self.th <= 135
        val2 = -135 <= self.th <= -45

        return val1 or val2

    def calc_snr(self):

        snr = self.intensity * np.sqrt(self.L * self.im_size / self.bg_noise_var)

        snr = snr / np.sqrt(2 * np.sqrt(np.pi) * self.psf_sigma)

        return abs(snr)

    def make_image(self):

        if self.verbosity > 1:
            print("make_mage()")

        self.clear()
        self.make_clean()
        self.add_noise()

    def make_clean(self):
        self.image_clean = self.intensity * model(
            self.im_size,
            self.x1 * self.im_size,
            self.x2 * self.im_size,
            self.y1 * self.im_size,
            self.y2 * self.im_size,
            self.psf_sigma,
        )

    def add_noise(self):

        if self.verbosity > 2:
            print("add_noise()")

        if np.isscalar(self.im_size):
            size = (self.im_size, self.im_size)
        bg = np.ones(size, dtype=np.float32) * self.bg_noise_var

        if self.use_source_noise:
            var = bg + np.abs(self.image_clean)
        else:
            var = bg

        self.image = np.random.normal(self.image_clean, np.sqrt(var)).astype(np.float32)

    def find(self):

        if self.verbosity > 1:
            print("find()")

        self.finder.input(self.image, psf=self.psf_sigma, variance=self.bg_noise_var)

    def run(self):
        if self.verbosity > 1:
            print("run()")
        self.clear()
        self.make_image()
        self.find()

        if len(self.finder.streaks) == 0:
            print(f"No streaks found. Maximal S/N= {self.finder.data.best_snr}")
        else:
            s = self.finder.streaks[0]

            if self.verbosity:
                print(
                    f"SIMULATED : S/N= {self.calc_snr():.2f} | "
                    f"I= {self.intensity:.2f} | "
                    f"L= {self.L*self.im_size:.1f} | "
                    f"th= {self.th:.2f} | "
                    f"x0= {self.x0*self.im_size:.2f}"
                )
                print(
                    f"CALCULATED: S/N= {s.snr:.2f} | "
                    f"I= {s.I:.2f} | L= {s.L:.1f} | "
                    f"th= {s.th:.2f} | x0= {s.x0:.2f} "
                )

            if self.verbosity > 1:
                input_xy = (self.x1, self.x2, self.y1, self.y2)
                input_xy = tuple(int(round(x * self.im_size)) for x in input_xy)
                print(
                    f"INPUT: x1= {self.x1:>4d} | x2= {self.x2:>4d} | "
                    f"y1= {self.y1:>4d} | y2= {self.y2:>4d}"
                )
                print(
                    f"FOUND: x1= {s.x1:>4d} | x2= {s.x2:>4d} | "
                    f"y1= {s.y1:>4d} | y2= {s.y2:>4d}"
                )

    def randomly(self):
        x = np.random.rand(2)
        y = np.random.rand(2)
        self.x1 = min(x)
        self.x2 = max(x)
        self.y1 = min(y)
        self.y2 = max(y)


# test (reload object)
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("This is a test for Simulator and Finder...")
    s = Simulator()
    s.verbosity = 1

    # s.x1 = 3.0 / 8
    # s.x2 = 0.5
    # s.y1 = 1 / 3
    # s.y2 = 1.0 / 2

    # s.x1 = 0.2
    # s.y1 = 0.01
    # s.x2 = 1
    # s.y2 = 1.5
    s.randomly()

    s.finder.use_subtract_mean = False
    s.run()

    fig, ax = plt.subplots()
    ax.imshow(s.image)
    if s.finder.streaks:
        for st in s.finder.streaks:
            st.plot_lines(ax)
