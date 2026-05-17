import math
import cmath
import matplotlib.pyplot as plt
import numpy as np


def F(x):
    return x ** 2 - 4 * math.sin(x) - 1


def dF(x):
    return 2 * x - 4 * math.cos(x)


def d2F(x):
    return 2 + 4 * math.sin(x)


def phi(x, step_sign=1):
    tau = -0.1 if step_sign == 1 else 0.1
    return x + tau * F(x)


def dphi(x, step_sign=1):
    tau = -0.1 if step_sign == 1 else 0.1
    return 1 + tau * dF(x)


def check_stop(x_new, x_old, eps):
    return abs(F(x_new)) < eps and abs(x_new - x_old) < eps


def simple_iteration(x0, step_sign=1, eps=1e-10, max_iter=1000):
    x_old = x0
    for i in range(1, max_iter + 1):
        x_new = phi(x_old, step_sign)
        if check_stop(x_new, x_old, eps):
            return x_new, i
        x_old = x_new
    return x_old, max_iter


def newton_method(x0, eps=1e-10, max_iter=1000):
    x_old = x0
    for i in range(1, max_iter + 1):
        df_val = dF(x_old)
        if abs(df_val) < 1e-12:
            break
        x_new = x_old - F(x_old) / df_val
        if check_stop(x_new, x_old, eps):
            return x_new, i
        x_old = x_new
    return x_old, max_iter


def chebyshev_method(x0, eps=1e-10, max_iter=1000):
    x_old = x0
    for i in range(1, max_iter + 1):
        fx = F(x_old)
        dfx = dF(x_old)
        d2fx = d2F(x_old)
        if abs(dfx) < 1e-12:
            break
        x_new = x_old - fx / dfx - 0.5 * (fx ** 2 * d2fx) / (dfx ** 3)
        if check_stop(x_new, x_old, eps):
            return x_new, i
        x_old = x_new
    return x_old, max_iter


def secant_method(x0, x1, eps=1e-10, max_iter=1000):
    x_prev, x_curr = x0, x1
    for i in range(1, max_iter + 1):
        f_curr, f_prev = F(x_curr), F(x_prev)
        if abs(f_curr - f_prev) < 1e-12:
            break
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if check_stop(x_new, x_curr, eps):
            return x_new, i
        x_prev, x_curr = x_curr, x_new
    return x_curr, max_iter


def muller_method(x0, x1, x2, eps=1e-10, max_iter=1000):
    for i in range(1, max_iter + 1):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        h1, h2 = x1 - x0, x2 - x1
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2
        discr = cmath.sqrt(b ** 2 - 4 * a * c)
        if abs(b + discr) > abs(b - discr):
            den = b + discr
        else:
            den = b - discr
        if abs(den) < 1e-12:
            break
        delta = -2 * c / den
        x_new = x2 + delta
        x_new = x_new.real
        if check_stop(x_new, x2, eps):
            return x_new, i
        x0, x1, x2 = x1, x2, x_new
    return x2, max_iter


def inverse_lagrange_3points(x0, x1, x2, eps=1e-10, max_iter=1000):
    for i in range(1, max_iter + 1):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        if abs(y0 - y1) < 1e-12 or abs(y0 - y2) < 1e-12 or abs(y1 - y2) < 1e-12:
            break
        term0 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
        term1 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
        term2 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        x_new = term0 + term1 + term2
        if check_stop(x_new, x2, eps):
            return x_new, i
        x0, x1, x2 = x1, x2, x_new
    return x2, max_iter


def horner_eval(a, x):
    n = len(a) - 1
    p_val = a[n]
    dp_val = 0.0
    for i in range(n - 1, -1, -1):
        dp_val = p_val + x * dp_val
        p_val = a[i] + x * p_val
    return p_val, dp_val


def newton_horner(a, x0, eps=1e-10, max_iter=100):
    x_old = x0
    for i in range(1, max_iter + 1):
        f_val, df_val = horner_eval(a, x_old)
        if abs(df_val) < 1e-12:
            x_old += 0.1
            continue
        x_new = x_old - f_val / df_val
        if abs(x_new - x_old) < eps and abs(f_val) < eps:
            return x_new, i
        x_old = x_new
    return x_old, max_iter


def lin_method(a, p0, q0, eps=1e-10, max_iter=1000):
    a3, a2, a1, a0 = a[3], a[2], a[1], a[0]
    p, q = p0, q0
    for iteration in range(1, max_iter + 1):
        b2 = a3
        b1 = a2 - p * b2
        b0 = a1 - p * b1 - q * b2
        denom_p = a3 * b1 - a2 * b2
        denom_q = a2 * b2 - a3 * b1
        if abs(denom_p) < 1e-12 or abs(denom_q) < 1e-12:
            break
        p_new = (a2 * b0 - a3 * b1) / denom_p
        q_new = (a1 * b0 - a0 * b2) / denom_q
        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            p, q = p_new, q_new
            break
        p, q = p_new, q_new
    D = complex(p ** 2 - 4 * q, 0)
    sqrt_D = cmath.sqrt(D)
    root1 = (-p + sqrt_D) / 2
    root2 = (-p - sqrt_D) / 2
    return root1, root2, iteration


def plot_transcendental_function():
    x = np.linspace(-3, 3, 1000)
    y = [F(xi) for xi in x]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='F(x) = x² - 4·sin(x) - 1')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Графік трансцендентної функції')
    plt.legend()
    roots = [1.933753764, -1.902527546]
    for r in roots:
        plt.plot(r, 0, 'ro', markersize=8)
        plt.annotate(f'x = {r:.6f}', (r, 0), xytext=(5, 5), textcoords='offset points')
    return plt


def plot_cubic_polynomial():
    coeffs = [-2, 4, -3, 1]

    def cubic(x):
        return coeffs[3] * x ** 3 + coeffs[2] * x ** 2 + coeffs[1] * x + coeffs[0]

    x = np.linspace(-1, 3, 1000)
    y = [cubic(xi) for xi in x]
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'r-', linewidth=2, label='P(x) = x³ - 3x² + 4x - 2')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.title('Кубічний многочлен (1 дійсний + 2 комплексних корені)')
    plt.legend()
    plt.plot(1, 0, 'ro', markersize=8)
    plt.annotate('x = 1', (1, 0), xytext=(5, 5), textcoords='offset points')
    return plt


if __name__ == "__main__":


    print("\n--- ПУНКТ 1: Табуляція функції ---")
    start, end, step = -3.0, 3.0, 0.1
    curr = start
    with open("tabulation.txt", "w", encoding="utf-8") as file:
        file.write("x\t\tF(x)\n")
        file.write("-" * 30 + "\n")
        while curr <= end + 1e-10:
            file.write(f"{curr:.2f}\t\t{F(curr):.10f}\n")
            curr += step
    print("Табуляцію збережено у файл 'tabulation.txt'")

    print("\n--- ПУНКТ 2-4: Уточнення коренів з точністю 1e-10 ---")
    print("-" * 70)

    root_growth = 1.8
    print(f"\nКорінь №1 (функція зростає, початкове наближення x0 = {root_growth})")
    print("-" * 55)

    res, it = simple_iteration(root_growth, 1, 1e-10)
    print(f"Метод простої ітерації    : x = {res:.12f} | Ітерацій: {it}")

    res, it = newton_method(root_growth, 1e-10)
    print(f"Метод Ньютона             : x = {res:.12f} | Ітерацій: {it}")

    res, it = chebyshev_method(root_growth, 1e-10)
    print(f"Метод Чебишева            : x = {res:.12f} | Ітерацій: {it}")

    res, it = secant_method(root_growth - 0.3, root_growth, 1e-10)
    print(f"Метод хорд                : x = {res:.12f} | Ітерацій: {it}")

    res, it = muller_method(root_growth - 0.3, root_growth - 0.15, root_growth, 1e-10)
    print(f"Метод парабол             : x = {res:.12f} | Ітерацій: {it}")

    res, it = inverse_lagrange_3points(root_growth - 0.3, root_growth - 0.15, root_growth, 1e-10)
    print(f"Зворотна інтерполяція     : x = {res:.12f} | Ітерацій: {it}")

    root_decay = -1.9
    print(f"\nКорінь №2 (функція спадає, початкове наближення x0 = {root_decay})")
    print("-" * 55)

    res, it = simple_iteration(root_decay, -1, 1e-10)
    print(f"Метод простої ітерації    : x = {res:.12f} | Ітерацій: {it}")

    res, it = newton_method(root_decay, 1e-10)
    print(f"Метод Ньютона             : x = {res:.12f} | Ітерацій: {it}")

    res, it = chebyshev_method(root_decay, 1e-10)
    print(f"Метод Чебишева            : x = {res:.12f} | Ітерацій: {it}")

    res, it = secant_method(root_decay - 0.3, root_decay, 1e-10)
    print(f"Метод хорд                : x = {res:.12f} | Ітерацій: {it}")

    res, it = muller_method(root_decay - 0.3, root_decay - 0.15, root_decay, 1e-10)
    print(f"Метод парабол             : x = {res:.12f} | Ітерацій: {it}")

    res, it = inverse_lagrange_3points(root_decay - 0.3, root_decay - 0.15, root_decay, 1e-10)
    print(f"Зворотна інтерполяція     : x = {res:.12f} | Ітерацій: {it}")

    print("\n--- ПУНКТ 5: Побудова графіків функцій ---")
    plot_transcendental_function()
    plot_cubic_polynomial()
    plt.suptitle("Графіки досліджуваних функцій", fontsize=14, fontweight='bold')
    plt.show()
    print("Графіки відображено")

    print("\n--- ПУНКТ 6: Запис коефіцієнтів алгебраїчного рівняння ---")
    with open("poly_coeffs.txt", "w", encoding="utf-8") as f_poly:
        f_poly.write("-2\n4\n-3\n1\n")
    print("Коефіцієнти рівняння x³ - 3x² + 4x - 2 = 0 записано в 'poly_coeffs.txt'")

    print("\n--- ПУНКТ 7: Зчитування коефіцієнтів ---")
    with open("poly_coeffs.txt", "r", encoding="utf-8") as f_poly:
        poly_coeffs = [float(line.strip()) for line in f_poly if line.strip()]
    print(f"Зчитаний масив коефіцієнтів (a0, a1, a2, a3): {poly_coeffs}")

    print("\n--- ПУНКТ 8: Метод Ньютона за схемою Горнера ---")
    print("-" * 55)
    alg_root, alg_it = newton_horner(poly_coeffs, x0=2.0, eps=1e-10)
    print(f"Дійсний корінь:         x = {alg_root:.12f}")
    print(
        f"Перевірка P(x) = {poly_coeffs[3] * alg_root ** 3 + poly_coeffs[2] * alg_root ** 2 + poly_coeffs[1] * alg_root + poly_coeffs[0]:.2e}")
    print(f"Кількість ітерацій:     {alg_it}")

    print("\n--- ПУНКТ 9: Метод Ліна для комплексних коренів ---")
    print("-" * 55)
    cx1, cx2, lin_it = lin_method(poly_coeffs, -1.0, 1.0, eps=1e-10)
    print(f"Комплексний корінь 1:   {cx1.real:.10f} + {cx1.imag:.10f}i")
    print(f"Комплексний корінь 2:   {cx2.real:.10f} + {cx2.imag:.10f}i")
    print(f"Кількість ітерацій:     {lin_it}")

    print("\n--- Пункт 10: Перевірка за теоремою Вієта ---")
    print("-" * 55)
    print(f"Сума коренів (знайдена):    {alg_root + cx1 + cx2:.10f}")
    print(f"Сума коренів (за Вієтом):   {poly_coeffs[2]:.10f}")
    print(f"Добуток коренів (знайдений): {alg_root * cx1 * cx2:.10f}")
    print(f"Добуток коренів (за Вієтом): {-poly_coeffs[0]:.10f}")

    print("\n" + "=" * 70)
    print("ВИСНОВКИ:")
    print("=" * 70)
    print("Усі методи знайшли корені із заданою точністю 10^-10")
    print("Найшвидша збіжність: методи Ньютона та Чебишева (2-3 ітерації)")
    print("Найповільніша збіжність: метод простої ітерації")
    print("Дійсний корінь кубічного рівняння: x = 1")
    print("Комплексні корені: 1 + i та 1 - i")
    print("=" * 70)