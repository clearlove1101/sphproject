from pysph.sph.equation import Equation
from math import exp
import config


class TaitEOS(Equation):
    def __init__(self,
                 dest, sources, rho0=config.tillotson_rho0, c0=config.tillotson_c0, gamma=config.tillotson_gamma,
                 p0=0.0, cv=config.tillotson_cv
                 ):
        self.rho0 = rho0
        self.rho01 = 1.0 / rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5 * (gamma - 1.0)
        self.B = rho0 * c0 * c0 / gamma
        self.p0 = p0
        self.cv = cv

        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p, d_cs, d_T, d_e):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        d_p[d_idx] = self.p0 + self.B * (tmp - 1.0)
        d_cs[d_idx] = self.c0 * pow(ratio, self.gamma1)
        d_T[d_idx] = d_e[d_idx] / self.cv


class TillotsonEOS(Equation):

    def __init__(self,
                 dest, sources, rho0=config.tillotson_rho0, us=config.tillotson_us, a=config.tillotson_a,
                 b=config.tillotson_b, e0=config.tillotson_e0, A=config.tillotson_A, B=config.tillotson_B,
                 us_=config.tillotson_us_, alpha=config.tillotson_alpha, beta=config.tillotson_beta,
                 c0=config.tillotson_c0, cv=config.tillotson_cv
                 ):
        super(TillotsonEOS, self).__init__(dest, sources)
        self.rho0 = rho0
        self.us = us
        self.a = a
        self.b = b
        self.e0 = e0
        self.A = A
        self.B = B
        self.us_ = us_
        self.alpha = alpha
        self.beta = beta
        self.c0 = c0
        self.cv = cv
        self.min_cs = (self.A / 4 / self.rho0 * 1e-4) ** 0.5

    def loop(self, d_p, d_idx, d_rho, d_e, d_cs, d_T, d_G):

        rho = d_rho[d_idx]

        eta = rho / self.rho0

        miu = eta - 1

        e = d_e[d_idx]

        if e < self.us or eta >= 1:

            w0 = e / self.e0 / (eta ** 2) + 1.

            abedw = (self.a + self.b / w0) * e
            apbm = self.A + self.B * miu

            pc = (abedw * rho + apbm * miu)

            d_p[d_idx] = pc

            dpdu = (self.a + self.b / w0 ** 2) * rho

            dpdrho = (self.a + self.b * (3. * w0 - 2.) / w0 ** 2) * e + (apbm + self.B * miu) / self.rho0
            c2 = dpdrho + dpdu * pc / (rho * rho) + 4 / 3 * d_G[d_idx] / d_rho[d_idx]

            d_cs[d_idx] = c2 ** 0.5

        elif e > self.us_:
            z = 1 / eta - 1
            ea = exp(-min(self.alpha * z, 70.))
            betaz = self.beta * z
            eb = exp(-min(betaz * z, 70.))
            w0 = e / self.e0 / eta ** 2 + 1.
            berdw = self.b * e * rho / w0

            pe = (self.a * e * rho + (berdw + self.A * miu * ea) * eb)
            d_p[d_idx] = pe

            dpdu = (self.a + ea * self.b / w0 ** 2) * rho

            dpdrho = self.a * e + ea * (self.b * e * (3. * w0 - 2.) / w0 ** 2) + \
                ea * (self.b * e * rho / w0) * self.rho0 * (2. * self.alpha * z) / rho ** 2 + \
                ea * self.A * eb * (1. / self.rho0 + self.rho0 * miu / rho ** 2 * (2. * self.alpha * z + self.beta))

            c2 = dpdrho + dpdu * pe / (rho * rho) + 4 / 3 * d_G[d_idx] / d_rho[d_idx]

            d_cs[d_idx] = c2 ** 0.5

        else:
            z = 1 / eta - 1
            w0 = e / self.e0 / eta ** 2 + 1.
            betaz = self.beta * z
            ea = exp(-self.alpha * z)
            eb = exp(-betaz * z)
            abedw = (self.a + self.b / w0) * e
            apbm = self.A + self.B * miu
            berdw = self.b * e * rho / w0

            pe = (self.a * e * rho + (berdw + self.A * miu * ea) * eb)
            pc = (abedw * rho + apbm * miu)

            d_p[d_idx] = ((pe * (e - self.us) + pc * (self.us_ - e)) / (self.us_ - self.us))

            dpdu = self.a * rho + self.b * rho / w0 ** 2
            dpdrho = self.a * e + self.b * e * (
                        3. * w0 - 2.) / w0 ** 2 + self.A / self.rho0 + 2. * self.B * miu / self.rho0
            c2_1 = dpdrho + dpdu * pc / (rho * rho)

            dpdu = self.a * rho + ea * self.b * rho / w0 ** 2
            dpdrho = self.a * e + ea * (self.b * e * (3. * w0 - 2.) / w0 ** 2) + \
                     ea * (self.b * e * rho / w0) * self.rho0 * (2. * self.alpha * z) / rho ** 2 + \
                     ea * self.A * eb * (
                                 1. / self.rho0 + self.rho0 * miu / rho ** 2 * (2. * self.alpha * z + self.beta))
            c2_2 = dpdrho + dpdu * pe / (rho * rho)
            c2 = ((c2_2 * (e - self.us) + c2_1 * (self.us_ - e)) / (self.us_ - self.us)) + 4 / 3 * d_G[d_idx] / d_rho[d_idx]
            d_cs[d_idx] = c2 ** 0.5

        d_cs[d_idx] = max(d_cs[d_idx], self.min_cs)
        d_T[d_idx] = e / self.cv


class TillotsonEOSWithPorosity(Equation):
    def __init__(self,
                  dest, sources, rho0=config.tillotson_rho0, us=config.tillotson_us, a=config.tillotson_a,
                  b=config.tillotson_b, e0=config.tillotson_e0, A=config.tillotson_A, B=config.tillotson_B,
                  us_=config.tillotson_us_, alpha=config.tillotson_alpha, beta=config.tillotson_beta,
                  c0=config.tillotson_c0, cv=config.tillotson_cv,
                  p_e=config.tillotson_p_e, p_t=config.tillotson_p_t, p_s=config.tillotson_p_s,
                  alpha_0=config.tillotson_alpha_0, alpha_e=config.tillotson_alpha_e, alpha_t=config.tillotson_alpha_t,
                  n1=config.tillotson_n1, n2=config.tillotson_n2, crushcurve_style=config.tillotson_crushcurve_style
                  ):
        super(TillotsonEOSWithPorosity, self).__init__(dest, sources)
        self.rho0 = rho0
        self.us = us
        self.a = a
        self.b = b
        self.e0 = e0
        self.A = A
        self.B = B
        self.us_ = us_
        self.alpha = alpha
        self.beta = beta
        self.c0 = c0
        self.cv = cv
        self.p_e = p_e
        self.p_t = p_t
        self.p_s = p_s
        self.alpha_0 = alpha_0
        self.alpha_e = alpha_e
        self.alpha_t = alpha_t
        self.n1 = n1
        self.n2 = n2
        self.crushcurve_style = crushcurve_style

    def loop(self, d_p, d_idx, d_rho, d_e, d_cs, d_T, d_dpde, d_dpdrho, d_alpha, d_dalphadp, d_G):

        # 1erg/g = 1e4J/kg
        e = d_e[d_idx]

        # 1g/cm3 = 1e3kg/m3
        rho = d_rho[d_idx]

        eta = rho / self.rho0

        miu = eta - 1

        if e < self.us or eta >= 1:

            w0 = e / self.e0 / eta ** 2 + 1.

            abedw = (self.a + self.b / w0) * e
            apbm = self.A + self.B * miu

            d_dpde[d_idx] = self.a * rho * d_alpha[d_idx] + rho * d_alpha[d_idx] * self.b / (
                                pow(e / (self.e0 * eta * eta) + 1.0, 2))\

            d_dpdrho[d_idx] = self.a * e + e * self.b * (1.0 + 3.0 * e / (self.e0 * eta * eta)) / (
                pow(e / (self.e0 * eta * eta) + 1.0, 2)) + self.A / self.rho0 + 2.0 * self.B / self.rho0 * (eta - 1.0)

            d_p[d_idx] = ((self.a + self.b / (
                    e / self.e0 * eta ** 2 + 1)) * e * d_alpha[d_idx] * rho + self.A * miu + self.B * miu ** 2)

            d_cs[d_idx] = (d_dpdrho[d_idx] + d_dpde[d_idx] * d_p[d_idx] / d_rho[d_idx] ** 2 + 4 / 3 * d_G[d_idx] / d_rho[d_idx]) ** 0.5

        elif e > self.us_:
            z = 1 / eta - 1
            ea = exp(-self.alpha * z)
            betaz = self.beta * z
            eb = exp(-betaz * z)
            w0 = e / self.e0 / eta ** 2 + 1.
            berdw = self.b * e * rho / w0

            d_dpde[d_idx] = self.a * rho * d_alpha[d_idx] + rho * d_alpha[d_idx] * self.b / (
                                pow(e / (self.e0 * eta * eta) + 1.0, 2)) * exp(
                                self.alpha * (pow(self.rho0 / (rho * d_alpha[d_idx]) - 1.0, 2)))

            d_p[d_idx] = (self.a * e * rho * d_alpha[d_idx] + (self.b * e * rho * d_alpha[d_idx] / (
                                    e / self.e0 / eta ** 2 + 1.0) + self.A * miu * ea) * eb)

            d_dpdrho[d_idx] = self.a * e + exp(-self.alpha * (pow(self.rho0 / (rho * d_alpha[d_idx]) - 1.0, 2))) * (
                    2.0 * self.alpha * self.rho0 / (rho * rho * d_alpha[d_idx] * d_alpha[d_idx]) * (self.rho0 / (
                    rho * d_alpha[d_idx]) - 1.0) * (self.b * rho * d_alpha[d_idx] * e / (e / (self.e0 * eta * eta) + 1.0)
                                                    + self.A * miu * exp(
                        -self.beta * (self.rho0 / (rho * d_alpha[d_idx]) - 1.0)))
                    + self.b * e * (1.0 + 3.0 * e / (self.e0 * eta * eta)) / (pow(e / (self.e0 * eta * eta) + 1.0, 2))
                    + self.A * exp(-self.beta * (self.rho0 / (rho * d_alpha[d_idx]) - 1.0))
                    * (1.0 / self.rho0 + self.beta / (rho * d_alpha[d_idx])
                       - self.beta * self.rho0 / (rho * rho * d_alpha[d_idx] * d_alpha[d_idx])))

            d_cs[d_idx] = (d_dpdrho[d_idx] + d_dpde[d_idx] * d_p[d_idx] / d_rho[d_idx] ** 2 + 4 / 3 * d_G[d_idx] / d_rho[d_idx]) ** 0.5

        else:
            z = 1 / eta - 1
            w0 = e / self.e0 / eta ** 2 + 1.
            betaz = self.beta * z
            ea = exp(-self.alpha * z)
            eb = exp(-betaz * z)
            abedw = (self.a + self.b / w0) * e
            apbm = self.A + self.B * miu
            berdw = self.b * e * rho / w0

            pe = self.a * e * rho * d_alpha[d_idx] + (self.b * e * rho * d_alpha[d_idx] / (
                                    e / self.e0 / eta ** 2 + 1.0) + self.A * miu * ea) * eb
            pc = (self.a + self.b / (
                    e / self.e0 * eta ** 2 + 1)) * e * d_alpha[d_idx] * rho + self.A * miu + self.B * miu ** 2

            d_p[d_idx] = ((pe * (e - self.us) + pc * (self.us_ - e)) / (self.us_ - self.us))

            d_dpde[d_idx] = ((pe - pc) + (e - self.us) * self.a * rho * d_alpha[d_idx]
                          + rho * d_alpha[d_idx] * self.b / (pow(e / (self.e0 * eta * eta) + 1.0, 2))
                          * exp(-self.alpha * (pow(self.rho0 / (rho * d_alpha[d_idx]) - 1.0, 2)))
                          + (self.us_ - e) * self.a * rho * d_alpha[d_idx] + rho * d_alpha[d_idx]
                          * self.b / (pow(e / (self.e0 * eta * eta) + 1.0, 2))) / (self.us_ - self.us)

            d_dpdrho[d_idx] = ((self.a * e + exp(-self.alpha * (pow(self.rho0 / (rho * d_alpha[d_idx]) - 1.0, 2)))
                                * (2.0 * self.alpha * self.rho0 / (rho * rho * d_alpha[d_idx] * d_alpha[d_idx]) * (
                            self.rho0 / (rho * d_alpha[d_idx]) - 1.0)
                                   * (self.b * rho * d_alpha[d_idx] * e / (e / (self.e0 * eta * eta) + 1.0)
                                      + self.A * miu * exp(-self.beta * (self.rho0 / (rho * d_alpha[d_idx]) - 1.0)))
                                   + self.b * e * (1.0 + 3.0 * e / (self.e0 * eta * eta)) / (
                                       pow(e / (self.e0 * eta * eta) + 1.0, 2))
                                   + self.A * exp(-self.beta * (self.rho0 / (rho * d_alpha[d_idx]) - 1.0))
                                   * (1.0 / self.rho0 + self.beta / (rho * d_alpha[d_idx]) - self.beta * self.rho0
                                      / (rho * rho * d_alpha[d_idx] * d_alpha[d_idx])))) * (e - self.us)
                               + (self.a * e + e * self.b * (1.0 + 3.0 * e / (self.e0 * eta * eta))
                                  / (pow(e / (self.e0 * eta * eta) + 1.0, 2))
                                  + self.A / self.rho0 + 2.0 * self.B / self.rho0 * (eta - 1.0))
                               * (self.us_ - e)) / (self.us_ - self.us)

            d_cs[d_idx] = (d_dpdrho[d_idx] + d_dpde[d_idx] * d_p[d_idx] / d_rho[d_idx] ** 2 + 4 / 3 * d_G[d_idx] / d_rho[d_idx]) ** 0.5

        d_T[d_idx] = e / self.cv

        pressure = d_p[d_idx] / d_alpha[d_idx]

        dalphadp_elastic = 0.0
        d_dalphadp[d_idx] = 0.0

        if self.crushcurve_style == 0:
            if pressure <= self.p_e:
                d_dalphadp[d_idx] = dalphadp_elastic
            elif self.p_e < pressure < self.p_s:
                d_dalphadp[d_idx] = - 2.0 * (self.alpha_0 - 1.0) * (self.p_s - pressure) / (
                    pow((self.p_s - self.p_e), 2))
            elif pressure >= self.p_s:
                d_dalphadp[d_idx] = 0.0
        elif self.crushcurve_style == 1:
            if pressure <= self.p_e:
                d_dalphadp[d_idx] = dalphadp_elastic
            elif self.p_e < pressure < self.p_t:
                d_dalphadp[d_idx] = - ((self.alpha_0 - 1.0) / (self.alpha_e - 1.0)) * (
                        self.alpha_e - self.alpha_t) * self.n1 * (
                                            pow(self.p_t - pressure, self.n1 - 1.0) / pow(self.p_t - self.p_e,
                                                                                          self.n1)) - (
                                            (self.alpha_0 - 1.0) / (self.alpha_e - 1.0)) * (
                                            self.alpha_t - 1.0) * self.n2 * (
                                            pow(self.p_s - pressure, self.n2 - 1.0) / pow(self.p_s - self.p_e,
                                                                                          self.n2))
            elif self.p_t <= pressure < self.p_s:
                d_dalphadp[d_idx] = - ((self.alpha_0 - 1.0) / (self.alpha_e - 1.0)) * (self.alpha_t - 1.0) * self.n2 * (
                        pow(self.p_s - pressure, self.n2 - 1.0) / pow(self.p_s - self.p_e, self.n2))
            elif pressure >= self.p_s:
                d_dalphadp[d_idx] = 0.0
