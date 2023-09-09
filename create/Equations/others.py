from pysph.sph.equation import Equation
import config


class EnergyEquationWithStressAndViscosity(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0, eta=0.01):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eta = float(eta)
        super(EnergyEquationWithStressAndViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ae):
        d_ae[d_idx] = 0.0


    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p, d_cs, s_cs, d_ae,
             XIJ, VIJ, DWIJ, HIJ, R2IJ, RHOIJ1):

        rhoa = d_rho[d_idx]
        ca = d_cs[d_idx]
        pa = d_p[d_idx]

        rhob = s_rho[s_idx]
        cb = s_cs[s_idx]
        pb = s_p[s_idx]
        mb = s_m[s_idx]

        rhoa2 = 1. / (rhoa * rhoa)
        rhob2 = 1. / (rhob * rhob)

        # artificial viscosity
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij) / (R2IJ + self.eta * self.eta * HIJ * HIJ)

            piij = -self.alpha * cij * muij + self.beta * muij * muij
            piij = piij * RHOIJ1

        vijdotdwij = VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] + VIJ[2] * DWIJ[2]

        # thermal energy contribution
        d_ae[d_idx] += 0.5 * mb * (pa * rhoa2 + pb * rhob2)


    def post_loop(self, d_idx, d_rho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_v00, d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21, d_sr00, d_sr01, d_sr02, d_sr11, d_sr12, d_sr22,
                  d_v22, d_ae):

        # particle density
        rhoa = d_rho[d_idx]

        # deviatoric stress rate (symmetric)
        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        # particle density
        rhoa = d_rho[d_idx]

        # strain rate tensor (symmetric)
        eps00 = d_sr00[d_idx]
        eps01 = d_sr01[d_idx]
        eps02 = d_sr02[d_idx]

        eps10 = eps01
        eps11 = d_sr11[d_idx]
        eps12 = d_sr12[d_idx]

        eps20 = eps02
        eps21 = eps12
        eps22 = d_sr22[d_idx]

        # energy accelerations
        #sdoteij = s00a*eps00 +  s01a*eps01 + s10a*eps10 + s11a*eps11
        sdoteij = (s00a * eps00 + s01a * eps01 + s02a * eps02 + s10a * eps10 +
                   s11a * eps11 + s12a * eps12 + s20a * eps20 + s21a * eps21 +
                   s22a * eps22)

        d_ae[d_idx] += 1. / rhoa * sdoteij


class VelocityGradient3D(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_v00, d_v01, d_v02,
             d_v10, d_v11, d_v12,
             d_v20, d_v21, d_v22,
             DWIJ, VIJ):

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -VIJ[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -VIJ[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -VIJ[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -VIJ[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -VIJ[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -VIJ[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -VIJ[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -VIJ[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -VIJ[2] * DWIJ[2]

    def post_loop(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22, d_sr00, d_sr01, d_sr02, d_sr11, d_sr12, d_sr22,
                   d_rr00, d_rr01, d_rr02, d_rr11, d_rr12, d_rr22):
        # sr_ij = sr_ji
        # rr_ij = -rr_ji

        d_sr00[d_idx] = d_v00[d_idx]
        d_sr01[d_idx] = 0.5 * (d_v01[d_idx] + d_v10[d_idx])
        d_sr02[d_idx] = 0.5 * (d_v02[d_idx] + d_v20[d_idx])

        d_sr11[d_idx] = d_v11[d_idx]
        d_sr12[d_idx] = 0.5 * (d_v12[d_idx] + d_v21[d_idx])

        d_sr22[d_idx] = d_v22[d_idx]

        d_rr00[d_idx] = 0.
        d_rr01[d_idx] = 0.5 * (d_v01[d_idx] - d_v10[d_idx])
        d_rr02[d_idx] = 0.5 * (d_v02[d_idx] - d_v20[d_idx])

        d_rr11[d_idx] = 0.
        d_rr12[d_idx] = 0.5 * (d_v12[d_idx] - d_v21[d_idx])

        d_rr22[d_idx] = 0.


class ContinuityEquation(Equation):

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class MomentumEquationByStress(Equation):

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_s00, d_s01,
             d_s02, d_s11, d_s12, d_s22, s_s00, s_s01, s_s02, s_s11, s_s12,
             s_s22, d_au, d_av, d_aw,
             DWIJ):

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        rhoa21 = 1. / (rhoa * rhoa)
        rhob21 = 1. / (rhob * rhob)

        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        s00b = s_s00[s_idx]
        s01b = s_s01[s_idx]
        s02b = s_s02[s_idx]

        s10b = s_s01[s_idx]
        s11b = s_s11[s_idx]
        s12b = s_s12[s_idx]

        s20b = s_s02[s_idx]
        s21b = s_s12[s_idx]
        s22b = s_s22[s_idx]

        s00a = s00a
        s00b = s00b

        s11a = s11a
        s11b = s11b

        s22a = s22a
        s22b = s22b

        mb = s_m[s_idx]

        d_au[d_idx] += (
            mb * (s00a * rhoa21 + s00b * rhob21) * DWIJ[0] +
            mb * (s01a * rhoa21 + s01b * rhob21) * DWIJ[1] +
            mb * (s02a * rhoa21 + s02b * rhob21) * DWIJ[2])

        d_av[d_idx] += (
            mb * (s10a * rhoa21 + s10b * rhob21) * DWIJ[0] +
            mb * (s11a * rhoa21 + s11b * rhob21) * DWIJ[1] +
            mb * (s12a * rhoa21 + s12b * rhob21) * DWIJ[2])

        d_aw[d_idx] += (
            mb * (s20a * rhoa21 + s20b * rhob21) * DWIJ[0] +
            mb * (s21a * rhoa21 + s21b * rhob21) * DWIJ[1] +
            mb * (s22a * rhoa21 + s22b * rhob21) * DWIJ[2])


class EnergyEquationWithStress(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0,
                 eta=0.01):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eta = float(eta)
        super(EnergyEquationWithStress,self).__init__(dest, sources)

    def loop(self, d_idx, d_rho,
                  d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_v00, d_sr00, d_sr01, d_sr02, d_sr11, d_sr12, d_sr22,
                  d_v11, d_p, d_v22,
                  d_ae):

        # deviatoric stress rate (symmetric)
        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        # particle density
        rhoa = d_rho[d_idx]

        # strain rate tensor (symmetric)
        eps00 = d_sr00[d_idx]
        eps01 = d_sr01[d_idx]
        eps02 = d_sr02[d_idx]

        eps10 = eps01
        eps11 = d_sr11[d_idx]
        eps12 = d_sr12[d_idx]

        eps20 = eps02
        eps21 = eps12
        eps22 = d_sr22[d_idx]

        # energy acclerations
        #sdoteij = s00a*eps00 +  s01a*eps01 + s10a*eps10 + s11a*eps11
        sdoteij = s00a*eps00 + s01a*eps01 + s02a*eps02 + \
                  s10a*eps10 + s11a*eps11 + s12a*eps12 + \
                  s20a*eps20 + s21a*eps21 + s22a*eps22

        d_ae[d_idx] = 1./rhoa * (sdoteij - d_p[d_idx] * (d_v00[d_idx] + d_v11[d_idx] + d_v22[d_idx]))



class MomentumEquation3D(Equation):

    def __init__(self, dest, sources, c0=config.tillotson_c0,
                 alpha=1.0, beta=1.0, gx=0.0, gy=0.0, gz=0.0,
                 tensile_correction=False):

        self.alpha = alpha
        self.beta = beta
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0

        self.tensile_correction = tensile_correction

        super(MomentumEquation3D, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl):

        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        # compute the CFL time step factor
        _dt_cfl = 0.0
        if R2IJ > 1e-12:
            _dt_cfl = abs(HIJ * vijdotxij/R2IJ) + self.c0
            d_dt_cfl[d_idx] = max(_dt_cfl, d_dt_cfl[d_idx])

        tmpi = d_p[d_idx]*rhoi21
        tmpj = s_p[s_idx]*rhoj21

        fij = WIJ/WDP
        Ri = 0.0
        Rj = 0.0

        # tensile instability correction
        if self.tensile_correction:
            fij = fij*fij
            fij = fij*fij

            if d_p[d_idx] > 0:
                Ri = 0.01 * tmpi
            else:
                Ri = 0.2*abs(tmpi)

            if s_p[s_idx] > 0:
                Rj = 0.01 * tmpj
            else:
                Rj = 0.2 * abs(tmpj)

        # gradient and correction terms
        tmp = (tmpi + tmpj) + (Ri + Rj)*fij

        d_au[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz

        acc2 = (d_au[d_idx]*d_au[d_idx] +
                d_av[d_idx]*d_av[d_idx] +
                d_aw[d_idx]*d_aw[d_idx])

        # store the square of the max acceleration
        d_dt_force[d_idx] = acc2\


class HookesDeviatoricStressRate(Equation):

    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12,
                   d_as22):
        d_as00[d_idx] = 0.0
        d_as01[d_idx] = 0.0
        d_as02[d_idx] = 0.0

        d_as11[d_idx] = 0.0
        d_as12[d_idx] = 0.0

        d_as22[d_idx] = 0.0


    def loop(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_v00,
             d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21, d_v22, d_as00,
             d_as01, d_as02, d_as11, d_as12, d_as22, d_G, d_sr00, d_sr01, d_sr02, d_sr11, d_sr12, d_sr22,
             d_rr00, d_rr01, d_rr02, d_rr11, d_rr12, d_rr22):

        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v02 = d_v02[d_idx]

        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]
        v12 = d_v12[d_idx]

        v20 = d_v20[d_idx]
        v21 = d_v21[d_idx]
        v22 = d_v22[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s02 = d_s02[d_idx]

        s10 = d_s01[d_idx]
        s11 = d_s11[d_idx]
        s12 = d_s12[d_idx]

        s20 = d_s02[d_idx]
        s21 = d_s12[d_idx]
        s22 = d_s22[d_idx]

        # strain rate tensor is symmetric
        eps00 = d_sr00[d_idx]
        eps01 = d_sr01[d_idx]
        eps02 = d_sr02[d_idx]

        eps10 = eps01
        eps11 = d_sr11[d_idx]
        eps12 = d_sr12[d_idx]

        eps20 = eps02
        eps21 = eps12
        eps22 = d_sr22[d_idx]

        # rotation tensor is asymmetric
        omega00 = d_rr00[d_idx]
        omega01 = d_rr01[d_idx]
        omega02 = d_rr02[d_idx]

        omega10 = -omega01
        omega11 = d_rr11[d_idx]
        omega12 = d_rr12[d_idx]

        omega20 = -omega02
        omega21 = -omega12
        omega22 = d_rr22[d_idx]

        tmp = 2.0 * d_G[d_idx]
        trace = 1.0 / 3.0 * (eps00 + eps11 + eps22)

        # S_00
        d_as00[d_idx] = tmp*( eps00 - trace ) + \
                        ( s00*omega00 + s01*omega01 + s02*omega02) + \
                        ( s00*omega00 + s10*omega01 + s20*omega02)

        # S_01
        d_as01[d_idx] = tmp*(eps01) + \
                        ( s00*omega10 + s01*omega11 + s02*omega12) + \
                        ( s01*omega00 + s11*omega01 + s21*omega02)

        # S_02
        d_as02[d_idx] = tmp*eps02 + \
                        (s00*omega20 + s01*omega21 + s02*omega22) + \
                        (s02*omega00 + s12*omega01 + s22*omega02)

        # S_11
        d_as11[d_idx] = tmp*( eps11 - trace ) + \
                        (s10*omega10 + s11*omega11 + s12*omega12) + \
                        (s01*omega10 + s11*omega11 + s21*omega12)

        # S_12
        d_as12[d_idx] = tmp*eps12 + \
                        (s10*omega20 + s11*omega21 + s12*omega22) + \
                        (s02*omega10 + s12*omega11 + s22*omega12)

        # S_22
        d_as22[d_idx] = tmp*(eps22 - trace) + \
                        (s20*omega20 + s21*omega21 + s22*omega22) + \
                        (s02*omega20 + s12*omega21 + s22*omega22)


class Porosity(Equation):
    def __init__(self, dest, sources, alpha_0=config.tillotson_alpha_0):
        super(Porosity, self).__init__(dest, sources)
        self.alpha_0 = alpha_0

    def initialize(self, d_idx, d_alpha):
        d_alpha[d_idx] = self.alpha_0

    def loop(self, d_idx, d_alpha, d_dalphadt, d_ae, d_dpde, d_arho, d_dpdrho, d_dalphadp, d_p, d_rho):
        if d_alpha[d_idx] <= 1.0:
            d_dalphadt[d_idx] = 0.0
            d_alpha[d_idx] = 1.0
        else:
            d_dalphadt[d_idx] = ((d_ae[d_idx] * d_dpde[d_idx] + d_alpha[d_idx] * d_arho[d_idx] * d_dpdrho[d_idx])
                                 * d_dalphadp[d_idx]) / (d_alpha[d_idx] + d_dalphadp[d_idx] * (
                    d_p[d_idx] - d_rho[d_idx] * d_dpdrho[d_idx]))
            if d_dalphadt[d_idx] > 0.0:
                d_dalphadt[d_idx] = 0.0


class IdealInternalEnergy(Equation):

    def post_loop(self, d_p, d_rho, d_arho, d_ae, d_idx):
        d_ae[d_idx] = d_p[d_idx] * d_arho[d_idx] / d_rho[d_idx] ** 2