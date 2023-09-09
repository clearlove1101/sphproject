from pysph.sph.equation import Equation
from math import tanh
import config


class ROCK(Equation):
    def __init__(self, dest, sources):
        # kw = rock_parameters()
        self.Tm0 = config.rock_Tm0
        self.a = config.rock_a
        self.c = config.rock_c
        self.Xi = config.rock_Xi
        self.Epsilonfb = config.rock_Epsilonfb
        self.Pc = config.rock_Pc
        self.B = config.rock_B
        self.Yi0 = config.rock_Yi0
        self.miui = config.rock_miui
        self.Yim = config.rock_Yim
        self.Yd0 = config.rock_Yd0
        self.miud = config.rock_miud
        self.Ydm = config.rock_Ydm
        super(ROCK, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_Y, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             d_T, d_p, d_eps, d_D, d_sr00, d_sr01, d_sr02, d_sr11, d_sr12, d_sr22, d_aeps
             ):

        if d_p[d_idx] < 0:
            d_p[d_idx] *= (1 - d_D[d_idx])

        dam = d_eps[d_idx]
        p = d_p[d_idx]
        Tm = self.Tm0 * (p / self.a + 1) ** (1 / self.c)
        YtDY = tanh(self.Xi * (Tm / d_T[d_idx] - 1))
        Epsilonf_ = self.B * (p - self.Pc)
        Epsilonf = Epsilonf_ if Epsilonf_ > self.Epsilonfb else self.Epsilonfb
        D_ = dam / Epsilonf
        if 0 < D_ < 1:
            D = D_
        elif D_ > 0:
            D = 1
        else:
            D = 0
        d_D[d_idx] = D
        tmp = self.miui * p
        Yi = self.Yi0 + 1 / (1 / tmp + 1 / (self.Yim - self.Yi0))
        Yd_ = self.Yd0 + self.miud * p
        Yd = Yd_ if Yd_ < self.Ydm else self.Ydm
        d_Y[d_idx] = (D * Yd + (1 - D) * Yi) * YtDY

        d_aeps[d_idx] = (
                0.5 * (d_sr00[d_idx] * d_sr00[d_idx] + d_sr11[d_idx] * d_sr11[d_idx] + d_sr22[d_idx] * d_sr22[d_idx]) +
                d_sr01[d_idx] * d_sr01[d_idx] + d_sr02[d_idx] * d_sr02[d_idx] + d_sr12[d_idx] * d_sr12[d_idx]
        ) ** 0.5

        J2 = 0.5 * (d_s00[d_idx] * d_s00[d_idx] + d_s11[d_idx] * d_s11[d_idx] + d_s22[d_idx] * d_s22[d_idx]) + \
             d_s01[d_idx] * d_s01[d_idx] + d_s02[d_idx] * d_s02[d_idx] + d_s12[d_idx] * d_s12[d_idx]

        J2_sqrt = J2 ** 0.5

        fy = min(d_Y[d_idx] / J2_sqrt, 1.)

        d_s00[d_idx] *= fy
        d_s01[d_idx] *= fy
        d_s02[d_idx] *= fy
        d_s11[d_idx] *= fy
        d_s12[d_idx] *= fy
        d_s22[d_idx] *= fy

        if fy == 1:
            d_aeps[d_idx] = 0.
        # if fy < 1.:
        #     d_eps[d_idx] += (J2_sqrt - d_Y[d_idx]) / (3 * d_Y[d_idx])




class VonMises(Equation):
    def post_loop(self, d_idx, d_Y, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22):
        J2 = 0.5 * (d_s00[d_idx] * d_s00[d_idx] + d_s11[d_idx] * d_s11[d_idx] + d_s22[d_idx] * d_s22[d_idx]) + \
             d_s01[d_idx] * d_s01[d_idx] + d_s02[d_idx] * d_s02[d_idx] + d_s12[d_idx] * d_s12[d_idx]

        fy = min(d_Y[d_idx] / (3. * J2) ** 0.5, 1)

        d_s00[d_idx] *= fy
        d_s01[d_idx] *= fy
        d_s02[d_idx] *= fy
        d_s11[d_idx] *= fy
        d_s12[d_idx] *= fy
        d_s22[d_idx] *= fy


class EnergyEquationWithStress(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0,
                 eta=0.01):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eta = float(eta)
        super(EnergyEquationWithStress,self).__init__(dest, sources)

    def initialize(self, d_idx, d_ae):
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p,
             d_cs, s_cs, d_ae, XIJ, VIJ, DWIJ, HIJ, R2IJ, RHOIJ1):

        rhoa = d_rho[d_idx]
        ca = d_cs[d_idx]
        pa = d_p[d_idx]

        rhob = s_rho[s_idx]
        cb = s_cs[s_idx]
        pb = s_p[s_idx]
        mb = s_m[s_idx]

        rhoa2 = 1./(rhoa*rhoa)
        rhob2 = 1./(rhob*rhob)

        # artificial viscosity
        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + self.eta*self.eta*HIJ*HIJ)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1


        vijdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]

        # thermal energy contribution
        d_ae[d_idx] += 0.5 * mb * (pa*rhoa2 + pb*rhob2 + piij) * vijdotdwij

    def post_loop(self, d_idx, d_rho,
                  d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_v00, d_v01, d_v02,
                  d_v10, d_v11, d_v12,
                  d_v20, d_v21, d_v22,
                  d_ae):

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

        # strain rate tensor (symmetric)
        eps00 = d_v00[d_idx]
        eps01 = 0.5 * (d_v01[d_idx] + d_v10[d_idx])
        eps02 = 0.5 * (d_v02[d_idx] + d_v20[d_idx])

        eps10 = eps01
        eps11 = d_v11[d_idx]
        eps12 = 0.5 * (d_v12[d_idx] + d_v21[d_idx])

        eps20 = eps02
        eps21 = eps12
        eps22 = d_v22[d_idx]

        # energy acclerations
        #sdoteij = s00a*eps00 +  s01a*eps01 + s10a*eps10 + s11a*eps11
        sdoteij = s00a*eps00 + s01a*eps01 + s02a*eps02 + \
                  s10a*eps10 + s11a*eps11 + s12a*eps12 + \
                  s20a*eps20 + s21a*eps21 + s22a*eps22

        d_ae[d_idx] += 1./rhoa * sdoteij

