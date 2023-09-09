from pysph.base.utils import get_particle_array_wcsph
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.equation import Group
from pysph.base.kernels import CubicSpline
from pysph.sph.integrator import EulerIntegrator
from integrator import step
from pysph.tools import geometry as G
from Equations import EOS, solid, others, impact_policy
from config import *
from visualization import visual2d, visual
from tools import tools
import random

dx_asteroid = 0.1
dx_impactor = 0.1
len_asteroid = random.uniform(40,80)
ra = random.uniform(10, 15)
r_asteroid = len_asteroid / ra
r_impactor = random.uniform(0.2, 0.9)
V_impactor = random.randint(200, 300)
angle = 90
tf = 0.1
dt = 2e-5
print(dx_asteroid, dx_impactor, r_asteroid, ra, len_asteroid, r_impactor)

#
# dx_asteroid = 0.1
# dx_impactor = 0.1
# r_asteroid = 2.
# # ra = random.uniform(10, 15)
# len_asteroid = 30.
# r_impactor = 0.5
# V_impactor = 200
# angle = 90
# tf = 0.1
# dt = 2e-5
# print(dx_asteroid, dx_impactor, r_asteroid, len_asteroid, r_impactor)


class App(Application):

    def create_particles(self):

        x, y, z = G.get_3d_hollow_cylinder(dx=dx_asteroid, r=r_asteroid, length=len_asteroid,
                           center=center_asteroid,
                           num_layers=2, inside=True)

        asteroid = get_particle_array_wcsph(
            name=name_asteroid, x=x, y=y, z=z, h=h_asteroid, m=rho_asteroid * dx_asteroid ** 2, rho=rho_asteroid,
            as00=as00_asteroid, as01=as01_asteroid, as11=as11_asteroid, sr00=0, sr01=0, sr02=0, sr11=0, sr12=0, sr22=0,
            s00=s00_asteroid, s01=s01_asteroid, s11=s11_asteroid, rr00=0, rr01=0, rr02=0, rr11=0, rr12=0, rr22=0,
            G=G_asteroid, T=T_asteroid, Y=Y_asteroid, as12=0, as02=0, as22=0, eps=0, aeps=0,
            e=e_asteroid, ae=ae_asteroid, v20=0, s02=0, v12=0, v22=0, v21=0, s12=0, v02=0, s22=0, pi=0, t=0.,
            v00=v00_asteroid, v01=v01_asteroid, v10=v10_asteroid, v11=v11_asteroid, g_av=0., g_aw=0., g_au=0.,
            D=D_asteroid, cwij=cwij_asteroid, alpha=tillotson_alpha_0, dalphadt=0., dalphade=0., dalphadp=0., dpde=0., dpdrho=0.
        )
        asteroid.add_output_arrays(["T", "s01", "v01", "as01", "av", "eps", "e", "D", "cs", "g_av", "ae", "sr00", "t"])

        x, y, z = G.get_3d_sphere(dx=dx_impactor, center=center_impactor, r=r_impactor)

        impactor = get_particle_array_wcsph(
            name=name_impactor, x=x, y=y, z=z, h=h_impactor, m=rho_impactor * dx_impactor ** 2, rho=rho_impactor,
            as00=as00_impactor, as01=as01_impactor, as11=as11_impactor,
            s00=s00_impactor, s01=s01_impactor, s11=s11_impactor, sr00=0, sr01=0, sr02=0, sr11=0, sr12=0, sr22=0,
            G=G_impactor, T=T_impactor, Y=Y_impactor, as12=0, as02=0, as22=0, eps=0, aeps=0,
            e=e_impactor, ae=ae_impactor, v20=0, s02=0, v12=0, v22=0, v21=0, s12=0, v02=0, s22=0,
            v00=v00_impactor, v01=v01_impactor, v10=v10_impactor, v11=v11_impactor, pi=0, t=0.,
            D=D_impactor, cwij=cwij_impactor, g_av=0., g_au=0., g_aw=0., rr00=0, rr01=0, rr02=0, rr11=0, rr12=0, rr22=0,
             alpha=tillotson_alpha_0, dalphadt=0., dalphade=0., dalphadp=0., dpde=0., dpdrho=0.
        )
        impactor.add_output_arrays(["T", "s01", "v01", "as01", "av", "eps", "e", "D", "cs", "g_av", "ae", "sr00", "t"])

        return [asteroid, impactor]

    def create_equations(self):

        equations = [Group(equations=[

            impact_policy.ExplosionByTime(dest=name_impactor, sources=None, time=0.01, dt=2e-5, e=1e5),

            others.VelocityGradient3D(dest=name_impactor, sources=[name_asteroid, name_impactor]),
            others.VelocityGradient3D(dest=name_asteroid, sources=[name_asteroid, name_impactor]),

            others.HookesDeviatoricStressRate(dest=name_impactor, sources=[name_asteroid, name_impactor]),
            others.HookesDeviatoricStressRate(dest=name_asteroid, sources=[name_asteroid, name_impactor]),

            others.ContinuityEquation(dest=name_impactor, sources=[name_asteroid, name_impactor]),
            others.ContinuityEquation(dest=name_asteroid, sources=[name_asteroid, name_impactor]),

            # others.EnergyEquationWithStress(dest=name_impactor, sources=[name_asteroid, name_impactor]),
            # others.EnergyEquationWithStress(dest=name_asteroid, sources=[name_asteroid, name_impactor]),

            solid.EnergyEquationWithStress(dest=name_impactor, sources=[name_asteroid, name_impactor], alpha=1.5, beta=3.),
            solid.EnergyEquationWithStress(dest=name_asteroid, sources=[name_asteroid, name_impactor], alpha=1.5, beta=3.),

            # use any one of the solid equation below
            solid.ROCK(dest=name_impactor, sources=None),
            solid.ROCK(dest=name_asteroid, sources=None),
            # solid.VonMises(dest=name_impactor, sources=None),
            # solid.VonMises(dest=name_asteroid, sources=None),

            # use any one of the Tillotson below
            # EOS.TillotsonEOS(dest=name_impactor, sources=None),
            # EOS.TillotsonEOS(dest=name_asteroid, sources=None),

            # if TillotsonEOSWithPorosity is chosen, Porosity should be used together
            EOS.TillotsonEOSWithPorosity(dest=name_impactor, sources=None),
            EOS.TillotsonEOSWithPorosity(dest=name_asteroid, sources=None),
            others.Porosity(dest=name_impactor, sources=None),
            others.Porosity(dest=name_asteroid, sources=None),

            others.MomentumEquation3D(dest=name_impactor, sources=[name_asteroid, name_impactor], alpha=1.5, beta=3.),
            others.MomentumEquation3D(dest=name_asteroid, sources=[name_asteroid, name_impactor], alpha=1.5, beta=3.),

            others.MomentumEquationByStress(dest=name_impactor, sources=[name_asteroid, name_impactor]),
            others.MomentumEquationByStress(dest=name_asteroid, sources=[name_asteroid, name_impactor]),

        ])]
        return equations

    def create_solver(self):

        integrator = EulerIntegrator(asteroid=step.EulerStep3D(), impactor=step.EulerStep3D())

        solver = Solver(dim=3, kernel=CubicSpline(dim=3), tf=0.04, dt=2e-5, integrator=integrator)  # 参数还得细调
        return solver


for i in range(1):

    # tillotson_c0 = 2e3
    # dx_asteroid= 0.1
    # dx_impactor= 0.1
    # r_asteroid = random.uniform(1, 15)
    # ra= random.uniform(10,15)
    # len_asteroid = r_asteroid/ra
    # r_impactor = random.uniform(0.2,0.9)
    # V_impactor = random.randint(200,300)
    # angle = 90
    # tf = 0.1
    # dt = 2e-5
    # print(dx_asteroid,dx_impactor,r_asteroid ,ra,len_asteroid,r_impactor)
    app = App(fname=r"./main"+str(i))

    app.run()

    visual2d.show_all_process(
        r"./main"+str(i)+"_output", s=1, xlimit=[-2 * r_asteroid, 2 * r_asteroid], ylimit=[-2 * r_asteroid, 2 * r_asteroid],
        property_name="D"
    )
