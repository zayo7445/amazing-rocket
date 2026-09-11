import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from collections import namedtuple

ROCKET_MASS = 4.0
FUEL_MASS = 4.0
BURN_RATE = 0.4

KM = 700.0
C = 0.05
G = np.array([0, -9.82])

TIME = 20

INIT_LAUNCH_HEIGHT = 20

START_POS = np.array([0, 0])
START_VEL = np.array([0, 0])
GOAL_POS = np.array([80, 60])

X_AXIS = np.array([1, 0])

def theta(t, v, pos):
    if pos[1] < INIT_LAUNCH_HEIGHT:
        return -np.pi/2
    else:
        thrust = pos - GOAL_POS
        unit = thrust / np.linalg.norm(thrust) 
        return np.arctan2(unit[1], unit[0])

def m(t):
    return ROCKET_MASS + max(FUEL_MASS - BURN_RATE * t, 0)

def dm(t):
    return -BURN_RATE if t * BURN_RATE <= FUEL_MASS else 0

def u(t, v, pos):
    alpha = theta(t, v, pos)
    return KM * np.array([np.cos(alpha), np.sin(alpha)])

def F(t, v):
    return m(t) * G - C * np.linalg.norm(v) * v

def a(t, v, pos):
    return (F(t, v) + dm(t) * u(t, v, pos)) / m(t)

def fun(t, y):
    posx, posy, vx, vy = y
    v = np.array([vx, vy])
    pos = np.array([posx, posy])
    ax, ay = a(t, v, pos)
    print(vx, vy, ax, ay)
    return np.array([vx, vy, ax, ay])

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




# sol = solve_ivp(fun, [0, TIME], [*START_POS, *START_VEL])
sol = rk4(fun, [0, TIME], [*START_POS, *START_VEL], 0.01)

plt.plot(sol.y[0], sol.y[1], "--")
plt.plot(START_POS[0], START_POS[1], "or")
plt.plot(GOAL_POS[0], GOAL_POS[1], "og")

plt.grid()
plt.show()
