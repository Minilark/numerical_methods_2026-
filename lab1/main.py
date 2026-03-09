import numpy as np
import matplotlib.pyplot as plt

# 1. Дані з прикладу для забезпечення ідентичності результатів [cite: 447]
lats = [48.164214, 48.164983, 48.165605, 48.166228, 48.166777, 48.167326, 48.167011, 48.166053, 48.166655, 48.166497,
        48.166128, 48.165416, 48.164546, 48.163412, 48.162331, 48.162015, 48.162147, 48.161751, 48.161197, 48.160580,
        48.160250]
lons = [24.536044, 24.534836, 24.534068, 24.532915, 24.531927, 24.530884, 24.530061, 24.528039, 24.526064, 24.523574,
        24.520214, 24.517170, 24.514640, 24.512980, 24.511715, 24.509462, 24.506932, 24.504244, 24.501793, 24.500537,
        24.500106]
elevations = [1264.0, 1285.0, 1285.0, 1333.0, 1310.0, 1318.0, 1318.0, 1339.0, 1375.0, 1417.0, 1486.0, 1524.0, 1553.0,
              1630.0, 1757.0, 1794.0, 1828.0, 1887.0, 1975.0, 1975.0, 2031.0]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах [cite: 372]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def solve_tridiagonal(alpha, beta, gamma, delta):
    n = len(delta)
    A = np.zeros(n);
    B = np.zeros(n)
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, n - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom
    x = np.zeros(n)
    x[n - 1] = (delta[n - 1] - alpha[n - 1] * B[n - 2]) / (alpha[n - 1] * A[n - 2] + beta[n - 1])
    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]
    return x


def build_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)
    alpha = np.zeros(n + 1);
    beta = np.ones(n + 1);
    gamma = np.zeros(n + 1);
    delta = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    alpha[n] = h[n - 1]
    beta[n] = 2 * h[n - 1]
    c = solve_tridiagonal(alpha, beta, gamma, delta)
    a = y[:-1]
    d = np.zeros(n);
    b = np.zeros(n)
    for i in range(n):
        if i < n - 1:
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        else:
            d[i] = -c[i] / (3 * h[i])
            b[i] = (y[i + 1] - y[i]) / h[i] - 2 / 3 * h[i] * c[i]
    return a, b, c[:-1], d


def eval_spline(x_nodes, a, b, c, d, x_val):
    idx = np.searchsorted(x_nodes, x_val) - 1
    idx = max(0, min(idx, len(a) - 1))
    dx = x_val - x_nodes[idx]
    return a[idx] + b[idx] * dx + c[idx] * (dx ** 2) + d[idx] * (dx ** 3)


# Підготовка дистанцій [cite: 382-384]
dist = [0.0]
for i in range(1, len(lats)):
    dist.append(dist[-1] + haversine(lats[i - 1], lons[i - 1], lats[i], lons[i]))
dist = np.array(dist)

# Вивід табуляції вузлів [cite: 445-447]
print(f"Кількість вузлів: {len(elevations)}")
print("\nТабуляція вузлів:")
print(" № |  Latitude  |  Longitude | Elevation (m)")
for i in range(len(lats)):
    print(f"{i:2d} | {lats[i]:.6f} | {lons[i]:.6f} | {elevations[i]:.2f}")

print("\nТабуляція (відстань, висота):")
print(" № | Distance (m) | Elevation (m)")
for i in range(len(dist)):
    print(f"{i:2d} | {dist[i]:10.2f} | {elevations[i]:8.2f}")

# Розрахунок похибок для 10, 15, 20 вузлів [cite: 451-458]
node_counts = [10, 15, 20]
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

for k in node_counts:
    indices = np.linspace(0, len(dist) - 1, k, dtype=int)
    xk, yk = dist[indices], np.array(elevations)[indices]
    a, b, c, d = build_spline(xk, yk)

    # Похибка на всіх 21 оригінальних вузлах
    y_spline_at_nodes = np.array([eval_spline(xk, a, b, c, d, d_val) for d_val in dist])
    errors = np.abs(np.array(elevations) - y_spline_at_nodes)

    print(f"\n===== {k} вузлів =====")
    print(f"Максимальна похибка: {np.max(errors)}")
    print(f"Середня похибка: {np.mean(errors)}")

    # Графіки
    x_fine = np.linspace(dist[0], dist[-1], 300)
    y_fine = [eval_spline(xk, a, b, c, d, x) for x in x_fine]
    ax1.plot(x_fine, y_fine, label=f"{k} вузлів")

    err_fine = [np.abs(np.interp(x, dist, elevations) - eval_spline(xk, a, b, c, d, x)) for x in x_fine]
    ax2.plot(x_fine, err_fine, label=f"{k} вузлів")

# Оформлення графіків
ax1.set_title("Вплив кількості вузлів")
ax1.plot(dist, elevations, 'o-', label="21 вузол (еталон)", alpha=0.4)
ax1.legend();
ax1.grid(True)

ax2.set_title("Похибка апроксимації")
ax2.legend();
ax2.grid(True)

plt.show()