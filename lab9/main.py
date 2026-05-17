import numpy as np
import matplotlib.pyplot as plt


# ========== 1. Система рівнянь ==========
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2 - 4


def f2(x1, x2):
    return x1 * x2 - 1


def system(X):
    return np.array([f1(X[0], X[1]), f2(X[0], X[1])])


# ========== 2. Цільова функція Φ(Χ) ==========
def Phi(X):
    F = system(X)
    return F[0] ** 2 + F[1] ** 2


# Вдосконалений допоміжний досліджуючий пошук (п. 4 методички)
def exploratory_search(func, X_base, delta, reduce_step_if_failed=False, q=2, eps1=1e-6):
    n = len(X_base)
    X_curr = np.array(X_base, dtype=float)
    delta_curr = np.array(delta, dtype=float)

    for i in range(n):
        # Крок вперед
        X_test = X_curr.copy()
        X_test[i] += delta_curr[i]
        if func(X_test) < func(X_curr):
            X_curr = X_test.copy()
            continue

        # Крок назад
        X_test[i] -= 2 * delta_curr[i]
        if func(X_test) < func(X_curr):
            X_curr = X_test.copy()
            continue

        # Якщо обидва кроки невдалі і увімкнено режим зменшення кроку
        if reduce_step_if_failed:
            while True:
                delta_curr[i] /= q
                if delta_curr[i] < eps1:
                    # Якщо крок став меншим за eps1, залишаємо попереднє значення
                    break

                # Спробуємо знову з меншим кроком
                X_test = X_curr.copy()
                X_test[i] += delta_curr[i]
                if func(X_test) < func(X_curr):
                    X_curr = X_test.copy()
                    break
                X_test[i] -= 2 * delta_curr[i]
                if func(X_test) < func(X_curr):
                    X_curr = X_test.copy()
                    break

    return X_curr, delta_curr


# ========== 3. Метод Хука-Дживса (За методичкою) ==========
def hooke_jeeves(func, X0, delta0, eps1, eps2, q=2, p=2):
    X0 = np.array(X0, dtype=float)
    delta = np.array(delta0, dtype=float)

    X_base = X0.copy()  # X^(0)
    trajectory = [X_base.copy()]
    steps_count = 0

    while True:
        # 4. Досліджуючий пошук з базової точки
        X_new, delta = exploratory_search(func, X_base, delta, reduce_step_if_failed=True, q=q, eps1=eps1)
        steps_count += 1

        # 5. Перевірка зміни базисної точки
        if not np.array_equal(X_new, X_base):
            # Перевірка умов закінчення пошуку
            if np.linalg.norm(delta) < eps1 and abs(func(X_new) - func(X_base)) < eps2:
                X_base = X_new.copy()
                trajectory.append(X_base.copy())
                break

            # 6. Перехід до пошуку по зразку
            X_old = X_base.copy()  # Колишній X^(0)
            X_base = X_new.copy()  # Новий X^(1)
            trajectory.append(X_base.copy())

            # Пошук по зразку
            while True:
                # 7. Знаходимо точку екстраполяції (по зразку)
                X_pattern = X_base + p * (X_base - X_old)

                # 8. Досліджуючий пошук з точки зразка (БЕЗ зменшення кроку)
                X_test, _ = exploratory_search(func, X_pattern, delta, reduce_step_if_failed=False)
                steps_count += 1

                # 9. Перевірка покращення
                if func(X_test) < func(X_base):
                    X_old = X_base.copy()
                    X_base = X_test.copy()
                    trajectory.append(X_base.copy())
                    # Повторюємо пошук по зразку (п. 6)
                else:
                    # Якщо умови не виконуються, повертаємось до досліджуючого пошуку
                    X_old = X_base.copy()
                    break
        else:
            # Якщо X^(1) == X^(0), то це точка мінімуму (або крок зменшився до мінімуму)
            break

    return X_base, trajectory, steps_count


# ========== 4. Побудова графіків (Виправлено помилку з label) ==========
def plot_equations():
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 8))
    c1 = plt.contour(X, Y, f1(X, Y), levels=[0], colors='r', linewidths=2)
    c2 = plt.contour(X, Y, f2(X, Y), levels=[0], colors='b', linewidths=2)

    # Створення кастомної легенди для контурів
    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    plt.legend([h1[0], h2[0]], ['x₁² + x₂² = 4', 'x₁ · x₂ = 1'])

    plt.grid(True)
    plt.axis('equal')
    plt.title('Графіки системи рівнянь')
    plt.show()


# ========== 5. Тестування ==========
def test_rosenbrock():
    def rosenbrock(X):
        return 100 * (X[0] ** 2 - X[1]) ** 2 + (X[0] - 1) ** 2

    X0 = [-1.2, 0.0]
    delta = [0.1, 0.1]

    res, traj, steps = hooke_jeeves(rosenbrock, X0, delta, 1e-5, 1e-5)
    print(f"\nТест на функції Розенброка:")
    print(f"Мінімум знайдено в точці: {res}")
    print(f"Значення функції в мінімумі: {rosenbrock(res):.2e}")


# ========== 6. Головна програма ==========
if __name__ == "__main__":
    plot_equations()
    test_rosenbrock()

    # Розв'язання системи
    X0 = [2.0, 1.0]  # Початкове наближення
    delta = [0.1, 0.1]  # Початковий крок
    eps1, eps2 = 1e-6, 1e-6
    q, p = 2.0, 2.0

    print(f"\nРозв'язання системи рівнянь:")
    print(f"X⁽⁰⁾ = {X0}, ΔX = {delta}, ε₁={eps1}, ε₂={eps2}, q={q}, p={p}")

    solution, trajectory, steps = hooke_jeeves(Phi, X0, delta, eps1, eps2, q, p)

    print(f"\nРозв'язок: x₁ = {solution[0]:.8f}, x₂ = {solution[1]:.8f}")
    print(f"Перевірка: f₁={f1(solution[0], solution[1]):.2e}, f₂={f2(solution[0], solution[1]):.2e}")
    print(f"Значення цільової функції Φ(X) = {Phi(solution):.2e}")
    print(f"Кількість ітерацій пошуку: {steps}")
    print(f"Кількість точок траєкторії: {len(trajectory)}")

    # Запис у файл
    with open("trajectory.txt", "w", encoding="utf-8") as f:
        f.write("Траєкторія спуску:\n")
        f.write("Крок\tx1\t\tx2\t\tΦ(X)\n")
        for i, pt in enumerate(trajectory):
            f.write(f"{i}\t{pt[0]:.8f}\t{pt[1]:.8f}\t{Phi(pt):.2e}\n")
        f.write(f"\nВсього обчислень кроків пошуку: {steps}\n")

    print("\nТраєкторію успішно збережено у файл 'trajectory.txt'")