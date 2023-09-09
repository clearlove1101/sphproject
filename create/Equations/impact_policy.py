from pysph.sph.equation import Equation


class ExplosionByTime(Equation):
    def __init__(self, dest, sources, time, dt=1e-4, e=1e3):
        super(ExplosionByTime, self).__init__(dest, sources)

        self.time = time

        self.dt = dt

        self.e = e

    def initialize(self, d_e, d_idx, d_t):

        if self.dt >= (d_t[d_idx] - self.time) >= -self.dt:

            d_e[d_idx] += self.e



