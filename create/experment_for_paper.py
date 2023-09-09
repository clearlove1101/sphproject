from pysph.base.utils import get_particle_array_wcsph
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.equation import Group
from pysph.base.kernels import CubicSpline
from pysph.sph.integrator import EulerIntegrator
from integrator import step
from pysph.tools import geometry as G
from Equations import EOS, solid, others, impact_policy
import config as c
from visualization import visual2d, visual
from tools import tools
from math import sin, cos, pi
import numpy as np
import glob


""" 实验超参数 """
out_put_file = r"main"

A = 50.
B = 30.
C = 30.
omg = 10.
c.V_impactor = 2e3
angle = 45

c.dx_impactor = 2.9
c.dx_asteroid = 2.9

tf = 0.1
dt = 2e-5

""" 预处理 """
if A >= B:
    omg_ = [0., 0., omg]
else:
    omg_ = [omg, 0., 0.]

c.direction_impactor = [abs(cos(angle / 180 * pi)), 0., -abs(sin(angle / 180 * pi))]


class App(Application):

    def create_particles(self):

        # x, y, z = G.get_3d_sphere(dx=dx_asteroid, center=center_asteroid, r=r_asteroid)
        # x, y, z = tools.get_random_rubble_pile(dx=dx_asteroid, center=center_asteroid, max_radius=r_asteroid)
        x, y, z = tools.get_ellipsoid(A, B, C, c.dx_asteroid)

        u, v, w = tools.add_omg(x, y, z, omg_)
        asteroid = get_particle_array_wcsph(
            name=c.name_asteroid, x=x, y=y, z=z, h=c.h_asteroid, m=c.rho_asteroid * c.dx_asteroid ** 3,
            rho=c.rho_asteroid,
            as00=c.as00_asteroid, as01=c.as01_asteroid, as11=c.as11_asteroid, sr00=0, sr01=0, sr02=0, sr11=0, sr12=0,
            sr22=0,
            s00=c.s00_asteroid, s01=c.s01_asteroid, s11=c.s11_asteroid, rr00=0, rr01=0, rr02=0, rr11=0, rr12=0, rr22=0,
            G=c.G_asteroid, T=c.T_asteroid, Y=c.Y_asteroid, as12=0, as02=0, as22=0, eps=0, aeps=0, u=u, v=v, w=w,
            e=c.e_asteroid, ae=c.ae_asteroid, v20=0, s02=0, v12=0, v22=0, v21=0, s12=0, v02=0, s22=0, t=0.,
            v00=c.v00_asteroid, v01=c.v01_asteroid, v10=c.v10_asteroid, v11=c.v11_asteroid, g_av=0., g_aw=0., g_au=0.,
            D=c.D_asteroid, cwij=c.cwij_asteroid, alpha=c.tillotson_alpha_0, dalphadt=0., dalphade=0., dalphadp=0.,
            dpde=0., dpdrho=0.
        )
        asteroid.add_output_arrays(["T", "s01", "v01", "as01", "av", "eps", "e", "D", "cs", "g_av", "ae", "sr00", "t"])

        # v
        x, y, z = G.get_3d_sphere(dx=c.dx_impactor, center=c.center_impactor, r=c.r_impactor)
        # x, y, z = tools.get_ellipsoid(a, b, b, dx_asteroid)

        u, v, w = tools.get_velocity(c.V_impactor, c.direction_impactor)
        # u_, v_, w_ = tools.add_omg(x, y, z, omg_impactor)
        # u = u_ + u
        # v = v_ + v
        # w = w_ + w

        impactor = get_particle_array_wcsph(
            name=c.name_impactor, x=x, y=y, z=z, h=c.h_impactor, m=c.rho_impactor * c.dx_impactor ** 3,
            rho=c.rho_impactor,
            as00=c.as00_impactor, as01=c.as01_impactor, as11=c.as11_impactor,
            s00=c.s00_impactor, s01=c.s01_impactor, s11=c.s11_impactor, sr00=0, sr01=0, sr02=0, sr11=0, sr12=0, sr22=0,
            G=c.G_impactor, T=c.T_impactor, Y=c.Y_impactor, as12=0, as02=0, as22=0, eps=0, aeps=0, t=0.,
            e=c.e_impactor, ae=c.ae_impactor, v20=0, s02=0, v12=0, v22=0, v21=0, s12=0, v02=0, s22=0,
            v00=c.v00_impactor, v01=c.v01_impactor, v10=c.v10_impactor, v11=c.v11_impactor,
            D=c.D_impactor, cwij=c.cwij_impactor, g_av=0., g_au=0., g_aw=0., rr00=0, rr01=0, rr02=0, rr11=0, rr12=0,
            rr22=0,
            u=u, v=v, w=w, alpha=c.tillotson_alpha_0, dalphadt=0., dalphade=0., dalphadp=0., dpde=0., dpdrho=0.
        )
        impactor.add_output_arrays(["T", "s01", "v01", "as01", "av", "eps", "e", "D", "cs", "g_av", "ae", "sr00", "t"])

        return [asteroid, impactor]

    def create_equations(self):

        equations = [Group(equations=[

            # impact_policy.ExplosionByTime(dest=c.name_impactor, sources=None, time=1e3, dt=2e-5, e=1e5),

            others.VelocityGradient3D(dest=c.name_impactor, sources=[c.name_asteroid, c.name_impactor]),
            others.VelocityGradient3D(dest=c.name_asteroid, sources=[c.name_asteroid, c.name_impactor]),

            others.HookesDeviatoricStressRate(dest=c.name_impactor, sources=[c.name_asteroid, c.name_impactor]),
            others.HookesDeviatoricStressRate(dest=c.name_asteroid, sources=[c.name_asteroid, c.name_impactor]),

            others.ContinuityEquation(dest=c.name_impactor, sources=[c.name_asteroid, c.name_impactor]),
            others.ContinuityEquation(dest=c.name_asteroid, sources=[c.name_asteroid, c.name_impactor]),

            # others.EnergyEquationWithStress(dest=name_impactor, sources=[name_asteroid, name_impactor]),
            # others.EnergyEquationWithStress(dest=name_asteroid, sources=[name_asteroid, name_impactor]),

            solid.EnergyEquationWithStress(dest=c.name_impactor, sources=[c.name_asteroid, c.name_impactor], alpha=1.5, beta=3.),
            solid.EnergyEquationWithStress(dest=c.name_asteroid, sources=[c.name_asteroid, c.name_impactor], alpha=1.5, beta=3.),

            # use any one of the solid equation below
            solid.VonMises(dest=c.name_impactor, sources=None),
            solid.ROCK(dest=c.name_asteroid, sources=None),
            # solid.VonMises(dest=name_impactor, sources=None),
            # solid.VonMises(dest=name_asteroid, sources=None),

            # use any one of the Tillotson below
            # EOS.TillotsonEOS(dest=name_impactor, sources=None),
            # EOS.TillotsonEOS(dest=name_asteroid, sources=None),

            # if TillotsonEOSWithPorosity is chosen, Porosity should be used together
            EOS.TillotsonEOSWithPorosity(dest=c.name_impactor, sources=None),
            EOS.TillotsonEOSWithPorosity(dest=c.name_asteroid, sources=None),
            others.Porosity(dest=c.name_impactor, sources=None),
            others.Porosity(dest=c.name_asteroid, sources=None),

            others.MomentumEquation3D(dest=c.name_impactor, sources=[c.name_asteroid, c.name_impactor], alpha=1.5, beta=3.),
            others.MomentumEquation3D(dest=c.name_asteroid, sources=[c.name_asteroid, c.name_impactor], alpha=1.5, beta=3.),

            others.MomentumEquationByStress(dest=c.name_impactor, sources=[c.name_asteroid, c.name_impactor]),
            others.MomentumEquationByStress(dest=c.name_asteroid, sources=[c.name_asteroid, c.name_impactor]),

        ])]
        return equations

    def create_solver(self):

        integrator = EulerIntegrator(asteroid=step.EulerStep3D(), impactor=step.EulerStep3D())

        solver = Solver(dim=3, kernel=CubicSpline(dim=3), tf=tf, dt=dt, integrator=integrator)  # 参数还得细调
        return solver


print("-------------" * 5)
print("Experiment parameters:")
print("-------------" * 5)
print("asteroid size (a, b, c): ({}, {}, {})".format(A, B, C))
print("asteroid rotation: {}".format(omg))
print("asteroid rotation axis: {}".format(omg_))
print("_ _ _ _ _ " * 5)
print("impactor velocity: {}".format(c.V_impactor))
print("impactor direction: {}".format(c.direction_impactor))
print("-------------" * 5)


app = App(fname=out_put_file)

app.run()

print("")

visual2d.show_all_process(
    r"./main_output", s=1, xlimit=[-1.5 * max(A, B), 1.5 * max(A, B)], ylimit=[-1.5 * max(A, B), 1.5 * max(A, B)],
    property_name="D"
)

lst = glob.glob(out_put_file + r"_output/*.npz")
lst.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))
pth = lst[-1]

x = visual2d.load_npz(pth)
arrays = x['asteroid']['arrays']

print("-------------" * 5)
print("Results in {}".format(pth))
print("-------------" * 5)
print("Temperature range: ({}, {})".format(np.min(arrays['T']), np.max(arrays['T'])))
print("Pressure range: ({}, {})".format(np.min(np.abs(arrays['p'])), np.max(arrays['p'])))
print("Density range: ({}, {})".format(np.min(arrays['rho']), np.max(arrays['rho'])))
print("Sound Speed range: ({}, {})".format(np.min(arrays['cs']), np.max(arrays['cs'])))
print("Max Velocity in z: {}".format(np.abs(np.min(arrays['w']))))
print("Dammage range: ({}, {})".format(np.min(np.abs(arrays['D'])), np.max(arrays['D'])))
print("-------------" * 5)




