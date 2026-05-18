import numpy as np
import matplotlib.pyplot as plt

def f1(x1, x2):
    return x1 ** 2 + x2 ** 2 - 4


def f2(x1, x2):
    return x1 * x2 - 1


def system(X):
    return np.array([f1(X[0], X[1]), f2(X[0], X[1])])


def Phi(X):
    F = system(X)
    return F[0] ** 2 + F[1] ** 2


def rosenbrock(X):
    return 100 * (X[1] - X[0] ** 2) ** 2 + (1 - X[0]) ** 2

def exploratory_search(func, X_base, delta, reduce_step_if_failed=False, q=2, eps1=1e-6):

    n = len(X_base)
    X_curr = np.array(X_base, dtype=float)
    delta_curr = np.array(delta, dtype=float)
    f_base = func(X_curr)

    for i in range(n):
        # Спроба додати крок
        X_test = X_curr.copy()
        X_test[i] += delta_curr[i]
        if func(X_test) < f_base:
            X_curr = X_test.copy()
            f_base = func(X_curr)
            continue

        X_test = X_curr.copy()
        X_test[i] -= delta_curr[i]
        if func(X_test) < f_base:
            X_curr = X_test.copy()
            f_base = func(X_curr)
            continue

        if reduce_step_if_failed:
            while True:
                delta_curr[i] /= q
                if delta_curr[i] < eps1:
                    break

                X_test = X_curr.copy()
                X_test[i] += delta_curr[i]
                if func(X_test) < f_base:
                    X_curr = X_test.copy()
                    f_base = func(X_curr)
                    break

                X_test = X_curr.copy()
                X_test[i] -= delta_curr[i]
                if func(X_test) < f_base:
                    X_curr = X_test.copy()
                    f_base = func(X_curr)
                    break

    return X_curr, delta_curr


def hooke_jeeves(func, X0, delta0, eps1, eps2, q=2, p=2, max_iter=10000):

    X0 = np.array(X0, dtype=float)
    delta = np.array(delta0, dtype=float)

    X_base = X0.copy()
    trajectory = [X_base.copy()]
    f_base = func(X_base)
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1

        X_new, delta = exploratory_search(func, X_base, delta, reduce_step_if_failed=True, q=q, eps1=eps1)
        f_new = func(X_new)

        if np.linalg.norm(delta) < eps1 and abs(f_new - f_base) < eps2:
            X_base = X_new.copy()
            trajectory.append(X_base.copy())
            break

        if f_new < f_base:
            X_old = X_base.copy()
            X_base = X_new.copy()
            trajectory.append(X_base.copy())

            while True:

                X_pattern = X_base + p * (X_base - X_old)

                X_test, _ = exploratory_search(func, X_pattern, delta, reduce_step_if_failed=False)
                f_test = func(X_test)

                if f_test < func(X_base):
                    X_old = X_base.copy()
                    X_base = X_test.copy()
                    trajectory.append(X_base.copy())
                else:
                    break
        else:
            delta = delta / q

            if np.linalg.norm(delta) < eps1:
                trajectory.append(X_base.copy())
                break

    return X_base, trajectory, iter_count


def plot_equations():
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 8))

    # Тільки графіки рівнянь (без точок)
    c1 = plt.contour(X, Y, f1(X, Y), levels=[0], colors='r', linewidths=2)
    c2 = plt.contour(X, Y, f2(X, Y), levels=[0], colors='b', linewidths=2)

    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    plt.legend([h1[0], h2[0]], ['x₁² + x₂² = 4', 'x₁·x₂ = 1'], fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Графіки системи рівнянь', fontsize=14)
    plt.savefig('system_equations.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_program():
    print("\n" + "=" * 60)
    print("ТЕСТУВАННЯ ПРОГРАМИ")
    print("=" * 60)

    # Початкові параметри
    X0 = np.array([-1.2, 1.0])
    delta = np.array([0.1, 0.1])
    eps1, eps2 = 1e-5, 1e-5

    print(f"\nТестування на функції Розенброка:")
    print(f"  Формула: f(x₁,x₂) = 100·(x₂ - x₁²)² + (1 - x₁)²")
    print(f"  Початкова точка: {X0}")
    print(f"  Початковий крок: {delta}")
    print(f"  Критерії: ε₁={eps1}, ε₂={eps2}")

    # Пошук мінімуму
    solution, trajectory, steps = hooke_jeeves(rosenbrock, X0, delta, eps1, eps2, q=2, p=2)

    print(f"\n  Результат:")
    print(f"  Мінімум знайдено в точці: ({solution[0]:.8f}, {solution[1]:.8f})")
    print(f"  Значення функції в мінімумі: {rosenbrock(solution):.2e}")
    print(f"  Кількість ітерацій: {steps}")
    print(f"  Кількість точок траєкторії: {len(trajectory)}")

    exact_solution = np.array([1.0, 1.0])
    error = np.linalg.norm(solution - exact_solution)
    print(f"  Похибка відносно точного розв'язку (1,1): {error:.2e}")


def solve_system():
    print("\n" + "=" * 60)
    print("РОЗВ'ЯЗАННЯ СИСТЕМИ РІВНЯНЬ")
    print("=" * 60)

    X0 = np.array([2.0, 1.0])
    delta0 = np.array([0.1, 0.1])
    eps1 = 1e-6
    eps2 = 1e-6
    q = 2.0
    p = 2.0

    print(f"\nПараметри методу:")
    print(f"  X⁰ = {X0}")
    print(f"  ΔX = {delta0}")
    print(f"  ε₁ = {eps1}")
    print(f"  ε₂ = {eps2}")
    print(f"  q = {q}")
    print(f"  p = {p}")

    # Знаходження розв'язку
    solution, trajectory, steps = hooke_jeeves(Phi, X0, delta0, eps1, eps2, q, p)

    print(f"\nРезультати пошуку:")
    print(f"  Розв'язок: x₁ = {solution[0]:.10f}, x₂ = {solution[1]:.10f}")
    print(f"  Перевірка розв'язку:")
    print(f"    f₁ = {f1(solution[0], solution[1]):.2e} (має дорівнювати 0)")
    print(f"    f₂ = {f2(solution[0], solution[1]):.2e} (має дорівнювати 0)")
    print(f"  Цільова функція Φ(X) = {Phi(solution):.2e}")

    print(f"\nЗБЕРЕЖЕННЯ ТРАЄКТОРІЇ У ФАЙЛ")
    print("-" * 60)

    # Виведення координат точок траєкторії спуску в файл
    with open("trajectory.txt", "w", encoding="utf-8") as f:


        f.write("Система нелінійних рівнянь:\n")
        f.write("  f₁(x₁,x₂) = x₁² + x₂² - 4 = 0\n")
        f.write("  f₂(x₁,x₂) = x₁·x₂ - 1 = 0\n\n")

        f.write("Цільова функція:\n")
        f.write("  Φ(x₁,x₂) = f₁² + f₂²\n\n")

        f.write("Параметри методу:\n")
        f.write(f"  X⁰ = ({X0[0]}, {X0[1]})\n")
        f.write(f"  ΔX = ({delta0[0]}, {delta0[1]})\n")
        f.write(f"  ε₁ = {eps1}\n")
        f.write(f"  ε₂ = {eps2}\n")
        f.write(f"  q = {q}\n")
        f.write(f"  p = {p}\n\n")

        f.write("ТРАЄКТОРІЯ СПУСКУ:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'№ кроку':<10} {'x₁':<20} {'x₂':<20} {'Φ(X)':<20}\n")
        f.write("-" * 70 + "\n")

        for i, pt in enumerate(trajectory):
            f.write(f"{i:<10} {pt[0]:<20.10f} {pt[1]:<20.10f} {Phi(pt):<20.2e}\n")

        f.write("-" * 70 + "\n")
        f.write(f"\nКількість кроків на траєкторії спуску: {len(trajectory)}\n")
        f.write(f"Кількість ітерацій пошуку: {steps}\n\n")

        f.write("РЕЗУЛЬТАТ:\n")
        f.write(f"  Розв'язок: x₁ = {solution[0]:.10f}, x₂ = {solution[1]:.10f}\n")
        f.write(f"  Φ(X*) = {Phi(solution):.2e}\n")
        f.write(f"  Похибка: |f₁| = {abs(f1(solution[0], solution[1])):.2e}, ")
        f.write(f"|f₂| = {abs(f2(solution[0], solution[1])):.2e}\n")

    print(f"  Траєкторію спуску збережено у файл 'trajectory.txt'")
    print(f"  Кількість кроків на траєкторії: {len(trajectory)}")
    print(f"  Кількість ітерацій пошуку: {steps}")

    return solution, trajectory


def main():


    print("\nПОБУДОВА ГРАФІКІВ РІВНЯНЬ")
    print("-" * 60)
    plot_equations()
    print("  Графіки збережено у файл 'system_equations.png'")

    test_program()

    solve_system()


# Запуск програми
if __name__ == "__main__":
    main()