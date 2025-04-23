import numpy as np
import scipy as sp
import sympy as sm
import sympy.physics.mechanics as me


class SerialRobot:
    """
    This robot has 3 dynamic symbols theta1, theta2, and x.
    """
    def __init__(self):
        th1, th2 = me.dynamicsymbols('theta_1 theta_2', real=True)
        x = me.dynamicsymbols('x', real=True, positive=True)

        dth1 = th1.diff()
        dth2 = th2.diff()
        dx = x.diff()

        ddth1 = dth1.diff()
        ddth2 = dth2.diff()
        ddx = dx.diff()

        uth2, ux = sm.symbols('u_theta_2 u_x', real=True)

        self.th1, self.th2, self.x = th1, th2, x
        self.uth2, self.ux = uth2, ux
        self.dth1, self.dth2, self.dx = dth1, dth2, dx
        self.ddth1, self.ddth2, self.ddx = ddth1, ddth2, ddx

        self.state_vec = [th1, th2, x, dth1, dth2, dx]
        
        # Three reference frames are involved for this configuration.
        # N is the world frame.
        # A and B are attached to robot links.
        N = me.ReferenceFrame('N')
        NA = me.ReferenceFrame('A')
        NB = me.ReferenceFrame('B')

        NA.orient_axis(N, N.y, th1)
        NB.orient_axis(NA, NA.y, th2)

        self.N, self.NA, self.NB = N, NA, NB

        # Some relevant physical parameters.
        # Link lengths, masses, gravity, moment of inertias.
        la, lb, ma, mb, g = sm.symbols('l_a l_b m_a m_b g',
                                       real=True, positive=True)
        Ia, Ib = sm.symbols('I_a I_b', real=True, positive=True)

        self.la, self.lb, self.ma, self.mb, self.g = la, lb, ma, mb, g
        self.Ia, self.Ib = Ia, Ib

        # Define the origin O in the frame N.
        O = me.Point('O')
        O.set_vel(N, 0 * N.x)

        # These points are the centers of mass of two links.
        Pa = O.locatenew('P_a', (la / 2 + x) * NA.z)
        Pb = O.locatenew('P_b', (la + x) * NA.z + lb / 2 * NB.z)

        Ba = me.RigidBody('B_a', Pa, N, ma,
                          (me.inertia(NA, Ia, Ia, 0), Pa))
        Bb = me.RigidBody('B_b', Pb, N, mb,
                          (me.inertia(NB, Ib, Ib, 0), Pb))

        self.O, self.Pa, self.Pb = O, Pa, Pb

        # Gravitational potential energy.
        Ba.potential_energy = Pa.pos_from(O).dot(N.z) * ma * g
        Bb.potential_energy = Pb.pos_from(O).dot(N.z) * mb * g

        L = me.Lagrangian(N, Ba, Bb)

        LM = me.LagrangesMethod(L, [th1, th2, x])
        eom = LM.form_lagranges_equations()

        # Actuation u(t), which is a dynamic symbol,
        # is added to the equations of motion.
        eom -= sm.Matrix([[0.], [uth2], [ux]])
        self.eom = eom


class FlyingRobot:
    """
    4 degrees of freedom.

    x is a dynamic symbol but it is kinematic in this stage of motions.
    """
    def __init__(self):
        q1, q2 = me.dynamicsymbols('q_1 q_2 ', real=True)
        th1, th2 = me.dynamicsymbols('theta_1 theta_2', real=True)
        x = me.dynamicsymbols('x', real=True, positive=True)

        dq1, dq2 = q1.diff(), q2.diff()
        dth1, dth2 = th1.diff(), th2.diff()
        dx = x.diff()

        ddq1, ddq2 = dq1.diff(), dq2.diff()
        ddth1, ddth2 = dth1.diff(), dth2.diff()
        ddx = dx.diff()

        self.state_vec = [q1, q2, th1, th2, dq1, dq2, dth1, dth2]

        self.q1, self.q2 = q1, q2
        self.th1, self.th2 = th1, th2
        self.x = x

        self.dq1, self.dq2 = dq1, dq2
        self.dth1, self.dth2 = dth1, dth2
        self.dx = dx

        self.ddq1, self.ddq2 = ddq1, ddq2
        self.ddth1, self.ddth2 = ddth1, ddth2
        self.ddx = ddx

        # Only actuation.
        uth2 = sm.symbols('u_theta_2', real=True)
        self.uth2 = uth2

        # Three reference frames.
        # A and B are attached to robot links.
        N = me.ReferenceFrame('N')
        NA = me.ReferenceFrame('A')
        NB = me.ReferenceFrame('B')

        NA.orient_axis(N, N.y, th1)
        NB.orient_axis(NA, NA.y, th2)

        self.N, self.NA, self.NB = N, NA, NB

        # Physical parameters.
        la, lb, ma, mb, g = sm.symbols('l_a l_b m_a m_b g',
                                       real=True, positive=True)
        Ia, Ib = sm.symbols('I_a I_b', real=True, positive=True)

        self.la, self.lb, self.ma, self.mb, self.g = la, lb, ma, mb, g
        self.Ia, self.Ib = Ia, Ib

        # Define the origin in the frame N.
        O = me.Point('O')
        O.set_vel(N, 0 * N.x)
        self.O = O

        # These points are the centers of mass of two links.
        Pa = O.locatenew('P_a', q1 * N.x + q2 * N.z)
        Pb = Pa.locatenew('P_b', la / 2 * NA.z + lb / 2 * NB.z)
        Pe = Pa.locatenew('P_e', -(la / 2 + x) * NA.z)

        self.Pa, self.Pb, self.Pe = Pa, Pb, Pe

        Ba = me.RigidBody('B_a', Pa, N, ma, (me.inertia(NA, Ia, Ia, 0), Pa))
        Bb = me.RigidBody('B_b', Pb, N, mb, (me.inertia(NB, Ib, Ib, 0), Pb))

        # Gravitational potential energy.
        Ba.potential_energy = Pa.pos_from(O).dot(N.z) * ma * g
        Bb.potential_energy = Pb.pos_from(O).dot(N.z) * mb * g

        L = me.Lagrangian(N, Ba, Bb)

        LM = me.LagrangesMethod(L, [q1, q2, th1, th2])
        eom = LM.form_lagranges_equations()

        # Actuation u(t), which is a dynamic symbol, is added to the equations of motion.
        # eom -= sm.Matrix([[sm.core.numbers.Zero()],
        #                   [sm.core.numbers.Zero()],
        #                   [sm.core.numbers.Zero()],
        #                   [uth2]])
        eom -= sm.Matrix([[0],
                          [0],
                          [0],
                          [uth2]])
        
        self.eom = eom
