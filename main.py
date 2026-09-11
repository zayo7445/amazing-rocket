import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ROCKET_MASS = 4.0
FUEL_MASS = 4.0
BURN_RATE = 0.4

KM = 700.0
C = 0.05
G = np.array([0, -9.82])

START_POS = [0, 0]
START_VEL = [0, 0]
GOAL_POS = [80, 60]

def theta(t):
    return -np.pi/2

def m(t):
    return ROCKET_MASS + max(FUEL_MASS - BURN_RATE * t, 0)

def dm(t):
    return -BURN_RATE if t * BURN_RATE <= FUEL_MASS else 0

def u(t):
    return KM * np.array([np.cos(theta(t)), np.sin(theta(t))])

def F(t, v):
    return m(t) * G - C * np.linalg.norm(v) * v

def a(t, v):
    return (F(t, v) + dm(t) * u(t)) / m(t)


def fun(t, y):
    posx, posy, vx, vy = y
    v = np.array([vx, vy])
    ax, ay = a(t, v)
    return np.array([vx, vy, ax, ay])


sol = solve_ivp(fun, [0, 20], [*START_POS, *START_VEL])

plt.plot(sol.y[0], sol.y[1], "--")
plt.plot(START_POS[0], START_POS[1], "or")
plt.plot(GOAL_POS[0], GOAL_POS[1], "og")

plt.grid()
plt.show()
