import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from collections import namedtuple

ROCKET_MASS = 4.0
FUEL_MASS = 4.0
BURN_RATE = 0.4

KM = 700.0  # exhaust speed
C = 0.05  # drag coefficient
G = np.array([0, -9.82])  # gravitational acceleration

TIME = 4  # simulation time
INIT_LAUNCH_HEIGHT = 20  # min height until steering

START_POS = np.array([0, 0])
START_VEL = np.array([0, 0])
GOAL_POS = np.array([80, 60])

H = 0.01  # default step size


def m(t):  # mass function
    return ROCKET_MASS + max(FUEL_MASS - BURN_RATE * t, 0)


def dm(t):  # mass derivative
    return -BURN_RATE if t * BURN_RATE <= FUEL_MASS else 0


def u(theta):  # exhaust vector
    return KM * np.array([np.cos(theta), np.sin(theta)])


def F(t, v):  # extrernal forces
    return m(t) * G - C * np.linalg.norm(v) * v


def a(t, v, u):  # acceleration
    return (F(t, v) + dm(t) * u) / m(t)


def system(theta):  # ode using steering function
    def fun(t, y):
        # y = [pos_x, pos_y, vel_x, vel_y]
        # y' = [vel_x, vel_y, acc_x, acc_y]
        pos, vel = np.split(y, 2)
        acc = a(t, vel, u(theta(t, pos, vel)))
        return np.concatenate([vel, acc])
    return fun


OdeResult = namedtuple("OdeResult", ["t", "y"])


def rk4(f, tspan, y0, h=H):
    steps = round((tspan[1] - tspan[0]) / h)
    t = np.linspace(tspan[0], tspan[1], steps + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2)*k1)
        k3 = f(t[i] + h/2, y[i] + (h/2)*k2)
        k4 = f(t[i+1], y[i] + h*k3)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    # transpose and put in OdeResult to match solve_ivp
    return OdeResult(t, np.transpose(y))


SOLVER = rk4  # solver to use


def theta_const(angle):  # theta function with a constant angle
    def theta(t, p, v):
        # don't steer before reaching height requirement
        if p[1] < INIT_LAUNCH_HEIGHT:
            return -np.pi/2
        else:
            return angle
    return theta


def argument(vec):  # find the ccw angle between the positive x axis and a vector
    unit = vec / np.linalg.norm(vec)
    return np.arctan2(unit[1], unit[0])


def apply_dot(arr1, arr2):  # dot product of array of vectors
    return np.sum(arr1 * arr2, axis=1)


def min_dist(angle):  # closest distance from goal using constant theta
    sol = SOLVER(system(theta_const(angle[0])), [0, TIME], np.concatenate([START_POS, START_VEL]))
    points = np.transpose(sol.y[:2])

    a = points[:-1]  # line segment start
    b = points[1:]  # line segment end
    ab = b - a  # line segment vector

    # project target point onto line
    t = apply_dot(GOAL_POS - a, ab) / apply_dot(ab, ab)
    # clamp projection onto line segment
    t = np.clip(t, 0, 1)
    # closest point of each line segment
    closest = a + t[:, None] * ab
    # distance of each line segment
    dist = np.linalg.norm(closest - GOAL_POS, axis=1)
    # minimum distance
    return np.min(dist)


# initial guess is to thrust in the opposte direction of the goal
angle = argument(START_POS - GOAL_POS)
# sovle for what angle reaches goal
angle = fsolve(min_dist, [angle])[0]
# simulate using that angle
sol = SOLVER(system(theta_const(angle)), [0, TIME], np.concatenate([START_POS, START_VEL]))
# print that angle
print(np.rad2deg(angle))

# plot path
plt.title("Rocket path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.plot(START_POS[0], START_POS[1], "or", label="Start position")
plt.plot(GOAL_POS[0], GOAL_POS[1], "og", label="Goal position")
plt.plot(sol.y[0], sol.y[1], "--", label="Rocket path")
plt.legend()

plt.grid()
plt.show()
