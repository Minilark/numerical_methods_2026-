import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

results_folder = "lab_results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"Створено папку: {results_folder}")
else:
    print(f"Папка {results_folder} вже існує")

def f(x, y):
    return y - x ** 2 + 1


def exact_solution(x):
    return x ** 2 + 2 * x + np.exp(x) - 1


def runge_kutta_2_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h, y + h * k1)
    return y + h / 2 * (k1 + k2)


def adams_predictor_corrector_fixed(f, a, b, y0, h):
    x_values = [a]
    y_values = [y0]

    x1 = a + h
    y1 = runge_kutta_2_step(f, a, y0, h)
    x_values.append(x1)
    y_values.append(y1)

    while x_values[-1] < b - 1e-10:
        xn = x_values[-1]
        yn = y_values[-1]
        xnm1 = x_values[-2]
        ynm1 = y_values[-2]

        yn_pred = yn + h / 2 * (3 * f(xn, yn) - f(xnm1, ynm1))

        yn_corr = yn + h / 2 * (f(xn + h, yn_pred) + f(xn, yn))

        x_new = xn + h
        if x_new > b:
            x_new = b
            h = b - xn
            yn_pred = yn + h / 2 * (3 * f(xn, yn) - f(xnm1, ynm1))
            yn_corr = yn + h / 2 * (f(xn + h, yn_pred) + f(xn, yn))

        x_values.append(x_new)
        y_values.append(yn_corr)

        if x_new >= b - 1e-10:
            break

    return np.array(x_values), np.array(y_values)


def adams_adaptive(f, a, b, y0, h0, eps):
    x_values = [a]
    y_values = [y0]
    h_values = [h0]
    h = h0

    x1 = a + h
    y1 = runge_kutta_2_step(f, a, y0, h)
    x_values.append(x1)
    y_values.append(y1)
    h_values.append(h)

    while x_values[-1] < b - 1e-10:
        xn = x_values[-1]
        yn = y_values[-1]
        xnm1 = x_values[-2]
        ynm1 = y_values[-2]

        yn_pred = yn + h / 2 * (3 * f(xn, yn) - f(xnm1, ynm1))
        yn_corr = yn + h / 2 * (f(xn + h, yn_pred) + f(xn, yn))

        local_error = abs(yn_corr - yn_pred)

        if local_error > eps and h > 1e-8:
            h = h / 2
            x_values.pop()
            y_values.pop()
            h_values.pop()
            x1 = x_values[-1] + h
            y1 = runge_kutta_2_step(f, x_values[-1], y_values[-1], h)
            x_values.append(x1)
            y_values.append(y1)
            h_values.append(h)
            continue
        elif local_error < eps / 10 and h < (b - a) / 5:
            h = min(h * 2, b - x_values[-1])

        x_new = xn + h
        if x_new > b:
            x_new = b
            h = b - xn
            yn_pred = yn + h / 2 * (3 * f(xn, yn) - f(xnm1, ynm1))
            yn_corr = yn + h / 2 * (f(xn + h, yn_pred) + f(xn, yn))

        x_values.append(x_new)
        y_values.append(yn_corr)
        h_values.append(h)

        if x_new >= b - 1e-10:
            break

    return np.array(x_values), np.array(y_values), np.array(h_values)


def runge_kutta_4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def runge_kutta_4_fixed(f, a, b, y0, h):
    x_values = [a]
    y_values = [y0]

    x = a
    y = y0
    while x < b - 1e-10:
        if x + h > b:
            h = b - x
        y = runge_kutta_4_step(f, x, y, h)
        x = x + h
        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)


def runge_kutta_4_adaptive(f, a, b, y0, h0, eps):
    x_values = [a]
    y_values = [y0]
    h_values = [h0]
    h = h0

    x = a
    y = y0

    while x < b - 1e-10:
        if x + h > b:
            h = b - x

        y1 = runge_kutta_4_step(f, x, y, h)
        y_mid = runge_kutta_4_step(f, x, y, h / 2)
        y2 = runge_kutta_4_step(f, x + h / 2, y_mid, h / 2)

        # Оцінка похибки методом Рунге
        error_est = abs(y2 - y1) * 16 / 15

        if error_est > eps and h > 1e-8:
            h = h / 2
            continue
        elif error_est < eps / 10 and h < (b - a) / 5:
            h = min(h * 2, b - x)

        x = x + h
        y = y2
        x_values.append(x)
        y_values.append(y)
        h_values.append(h)

    return np.array(x_values), np.array(y_values), np.array(h_values)

a, b = 0.0, 2.0
y0 = 0.0
h_fixed = 0.1
eps = 1e-4

print("\n" + "=" * 70)
print("Метод прогнозу та корекції Адамса 2-го порядку")
print("=" * 70)

# Розв'язок з фіксованим кроком
x_adams, y_adams = adams_predictor_corrector_fixed(f, a, b, y0, h_fixed)
y_exact_adams = exact_solution(x_adams)
error_adams = y_exact_adams - y_adams

print(f"\nРезультати з фіксованим кроком h = {h_fixed}:")
print("    x         y_наближене      y_точне          Похибка")
print("    " + "-" * 60)
for i in range(0, len(x_adams), max(1, len(x_adams) // 8)):
    print(f"    {x_adams[i]:.3f}    {y_adams[i]:.10f}    {y_exact_adams[i]:.10f}    {error_adams[i]:.2e}")

# Пункт 3: графік локальної похибки
plt.figure(figsize=(10, 6))
plt.plot(x_adams, error_adams, 'b-o', markersize=4, linewidth=1.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('Похибка φ(x)', fontsize=12)
plt.title('Локальна похибка методу Адамса (фіксований крок)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{results_folder}/part1_error_fixed.png", dpi=150)
plt.close()

error_estimate = []
x_est = []
for i in range(1, len(x_adams) - 1):
    xn, yn = x_adams[i], y_adams[i]
    xnm1, ynm1 = x_adams[i - 1], y_adams[i - 1]
    h_cur = x_adams[i + 1] - x_adams[i]
    yn_pred = yn + h_cur / 2 * (3 * f(xn, yn) - f(xnm1, ynm1))
    yn_corr = yn + h_cur / 2 * (f(xn + h_cur, yn_pred) + f(xn, yn))
    error_estimate.append(yn_corr - yn_pred)
    x_est.append(x_adams[i + 1])

plt.figure(figsize=(10, 6))
plt.plot(x_est, error_estimate, 'r-o', markersize=4, linewidth=1.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('Оцінка похибки (y_corr - y_pred)', fontsize=12)
plt.title('Оцінка локальної похибки методу Адамса', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='b', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{results_folder}/part1_error_estimate.png", dpi=150)
plt.close()

x_adams_auto, y_adams_auto, h_adams_auto = adams_adaptive(f, a, b, y0, h_fixed, eps)
y_exact_auto = exact_solution(x_adams_auto)
error_auto = y_exact_auto - y_adams_auto

print(f"\nРезультати з автоматичним вибором кроку (eps = {eps}):")
print("    x         y_наближене      y_точне          Похибка        Крок")
print("    " + "-" * 75)
for i in range(0, len(x_adams_auto), max(1, len(x_adams_auto) // 8)):
    h_val = h_adams_auto[i] if i < len(h_adams_auto) else h_adams_auto[-1]
    print(f"    {x_adams_auto[i]:.3f}    {y_adams_auto[i]:.10f}    {y_exact_auto[i]:.10f}    "
          f"{error_auto[i]:.2e}    {h_val:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x_adams_auto, h_adams_auto, 'g-o', markersize=4, linewidth=1.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('Крок h(x)', fontsize=12)
plt.title(f'Автоматичний вибір кроку (eps={eps})', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_folder}/part1_adaptive_step.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("ЧАСТИНА 2. Метод Рунге-Кутта 4-го порядку")
print("=" * 70)

# Розв'язок з фіксованим кроком
h_rk4 = 0.01
x_rk4, y_rk4 = runge_kutta_4_fixed(f, a, b, y0, h_rk4)
y_exact_rk4 = exact_solution(x_rk4)
error_rk4 = y_exact_rk4 - y_rk4

print(f"\nРезультати з фіксованим кроком h = {h_rk4}:")
print("    x         y_наближене      y_точне          Похибка")
print("    " + "-" * 60)
for i in range(0, len(x_rk4), max(1, len(x_rk4) // 8)):
    print(f"    {x_rk4[i]:.3f}    {y_rk4[i]:.10f}    {y_exact_rk4[i]:.10f}    {error_rk4[i]:.2e}")

plt.figure(figsize=(10, 6))
plt.plot(x_rk4, error_rk4, 'b-', linewidth=1)
plt.xlabel('x', fontsize=12)
plt.ylabel('Похибка φ(x)', fontsize=12)
plt.title('Локальна похибка методу Рунге-Кутта 4-го порядку', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{results_folder}/part2_error_rk4.png", dpi=150)
plt.close()

h_test = [0.1, 0.05, 0.025, 0.01, 0.005]
max_errors = []
for hh in h_test:
    xx, yy = runge_kutta_4_fixed(f, a, b, y0, hh)
    yy_ex = exact_solution(xx)
    max_err = np.max(np.abs(yy_ex - yy))
    max_errors.append(max_err)

print(f"\nДослідження залежності похибки від кроку:")
print("    h          Максимальна похибка")
print("    " + "-" * 35)
for hh, err in zip(h_test, max_errors):
    print(f"    {hh:.4f}      {err:.2e}")

h_runge = 0.1
x_h, y_h = runge_kutta_4_fixed(f, a, b, y0, h_runge)
x_h2, y_h2 = runge_kutta_4_fixed(f, a, b, y0, h_runge / 2)
y_h2_interp = np.interp(x_h, x_h2, y_h2)
error_runge_est = 16 / 15 * np.abs(y_h2_interp - y_h)

plt.figure(figsize=(10, 6))
plt.plot(x_h, error_runge_est, 'g-', linewidth=1.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('Оцінка похибки', fontsize=12)
plt.title(f'Оцінка похибки методом Рунге (h={h_runge})', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_folder}/part2_runge_estimate.png", dpi=150)
plt.close()

x_rk4_auto, y_rk4_auto, h_rk4_auto = runge_kutta_4_adaptive(f, a, b, y0, h_fixed, eps)
y_exact_rk4_auto = exact_solution(x_rk4_auto)
error_rk4_auto = y_exact_rk4_auto - y_rk4_auto

print(f"\nРезультати з автоматичним вибором кроку (eps = {eps}):")
print("    x         y_наближене      y_точне          Похибка        Крок")
print("    " + "-" * 75)
for i in range(0, len(x_rk4_auto), max(1, len(x_rk4_auto) // 8)):
    h_val = h_rk4_auto[i] if i < len(h_rk4_auto) else h_rk4_auto[-1]
    print(f"    {x_rk4_auto[i]:.3f}    {y_rk4_auto[i]:.10f}    {y_exact_rk4_auto[i]:.10f}    "
          f"{error_rk4_auto[i]:.2e}    {h_val:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x_rk4_auto, h_rk4_auto, 'm-o', markersize=4, linewidth=1.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('Крок h(x)', fontsize=12)
plt.title(f'Автоматичний вибір кроку (eps={eps})', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_folder}/part2_adaptive_step.png", dpi=150)
plt.close()

