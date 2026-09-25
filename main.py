import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import namedtuple

ROCKET_MASS = 4.0
FUEL_MASS = 4.0
BURN_RATE = 0.4

KM = 700.0
C = 0.05
G = np.array([0, -9.82])

TIME = 10

INIT_LAUNCH_HEIGHT = 20

START_POS = np.array([0, 0])
START_VEL = np.array([0, 0])
GOAL_POS = np.array([80, 60])

X_AXIS = np.array([1, 0])

H = 0.01


def angle_between(p1, p2):
    vec = p1 - p2
    unit = vec / np.linalg.norm(vec)
    return np.arctan2(unit[1], unit[0])


def theta_toward(t, p, v):
    if p[1] < INIT_LAUNCH_HEIGHT:
        return -np.pi/2
    else:
        return angle_between(p, GOAL_POS)


def theta_constant(angle):
    def theta(t, p, v):
        if p[1] < INIT_LAUNCH_HEIGHT:
            return -np.pi/2
        else:
            return angle
    return theta


def m(t):
    return ROCKET_MASS + max(FUEL_MASS - BURN_RATE * t, 0)


def dm(t):
    return -BURN_RATE if t * BURN_RATE <= FUEL_MASS else 0


def u(theta):
    return KM * np.array([np.cos(theta), np.sin(theta)])


def F(t, v):
    return m(t) * G - C * np.linalg.norm(v) * v


def a(t, v, u):
    return (F(t, v) + dm(t) * u) / m(t)


def fun(theta, t, y):
    pos, vel = np.split(y, 2)
    acc = a(t, vel, u(theta(t, pos, vel)))
    return np.concatenate([vel, acc])


def system(theta):
    return lambda t, y: fun(theta, t, y)


OdeResult = namedtuple("OdeResult", ["t", "y"])


def rk4(f, tspan, y0, h):
    steps = round((tspan[1] - tspan[0]) / h)
    t = np.linspace(tspan[0], tspan[1], steps + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2)*k1)
        k3 = f(t[i] + h/2, y[i] + (h/2)*k2)
        k4 = f(t[i+1], y[i] + h*k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) * (h/6)

    return OdeResult(t, np.transpose(y))


# sol = solve_ivp(system(theta_toward), [0, TIME], np.concatenate([START_POS, START_VEL]))
# sol = rk4(system(theta_toward), [0, TIME], np.concatenate([START_POS, START_VEL]), H)

def find_angle(angle):
    angle, = angle
    sol = rk4(system(theta_constant(angle)), [0, TIME], np.concatenate([START_POS, START_VEL]), H)
    dist = cdist(np.transpose(sol.y[:2]), np.array([GOAL_POS]))
    return dist.min()


angle = angle_between(START_POS, GOAL_POS)
angle, = fsolve(find_angle, angle)

print(angle)

sol = rk4(system(theta_constant(angle)), [0, TIME], np.concatenate([START_POS, START_VEL]), H)

plt.plot(sol.y[0], sol.y[1], "--")
plt.plot(START_POS[0], START_POS[1], "or")
plt.plot(GOAL_POS[0], GOAL_POS[1], "og")

plt.grid()
plt.show()
