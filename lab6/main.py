import numpy as np
import matplotlib.pyplot as plt
import os

def generate_system(n, x_true_value=2.5, filename_A="matrix_A.txt", filename_B="vector_B.txt"):
    np.random.seed(42)
    A = np.random.rand(n, n) * 100
    x_true = np.full(n, x_true_value)
    b = A @ x_true

    np.savetxt(filename_A, A, fmt="%.6f")
    np.savetxt(filename_B, b, fmt="%.6f")
    print(f"Матрицю A збережено у {filename_A}")
    print(f"Вектор B збережено у {filename_B}")
    return A, b, x_true

def read_matrix(filename):
    return np.loadtxt(filename)


def read_vector(filename):
    return np.loadtxt(filename)

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - s

        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U


def save_lu_to_file(L, U, filename="LU_decomposition.txt"):
    with open(filename, 'w') as f:
        f.write("L matrix (lower triangular with ones on diagonal):\n")
        np.savetxt(f, L, fmt="%.6f")
        f.write("\nU matrix (upper triangular):\n")
        np.savetxt(f, U, fmt="%.6f")
    print(f"LU-розклад збережено у {filename}")

def solve_lu(L, U, b):
    n = L.shape[0]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


def mat_vec_mul(A, x):
    return A @ x


def vector_norm(v):
    return np.max(np.abs(v))


def compute_accuracy(A, x, b):
    return np.max(np.abs(A @ x - b))


def iterative_refinement(A, L, U, b, x0, eps=1e-14, max_iter=100):
    errors = []
    x = x0.copy()

    for it in range(max_iter):
        r = b - mat_vec_mul(A, x)
        dx = solve_lu(L, U, r)
        x_new = x + dx
        err = vector_norm(dx)
        errors.append(err)

        if err < eps:
            print(f"Досягнуто заданої точності на ітерації {it + 1}")
            break

        x = x_new

    return x_new, it + 1, errors


def main():
    output_dir = "lab6_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Створено папку: {output_dir}")

    os.chdir(output_dir)

    n = 100
    eps0 = 1e-14

    print("=" * 60)
    print("ЧАСТИНА 1: Генерація системи")
    print("=" * 60)
    A, b, x_true = generate_system(n, x_true_value=2.5)

    print("\n" + "=" * 60)
    print("ЧАСТИНА 2: LU-розклад")
    print("=" * 60)
    L, U = lu_decomposition(A)
    save_lu_to_file(L, U)

    print("\n" + "=" * 60)
    print("ЧАСТИНА 3: Розв'язання СЛАР")
    print("=" * 60)
    x_lu = solve_lu(L, U, b)
    acc = compute_accuracy(A, x_lu, b)
    print(f"Точність початкового розв'язку (max |Ax-b|): {acc:.2e}")

    print("\n" + "=" * 60)
    print("ЧАСТИНА 4: Ітераційне уточнення")
    print("=" * 60)
    x_refined, num_iters, errors = iterative_refinement(A, L, U, b, x_lu, eps=eps0)

    acc_refined = compute_accuracy(A, x_refined, b)
    print(f"\nТочність уточненого розв'язку: {acc_refined:.2e}")
    print(f"Кількість ітерацій: {num_iters}")
    print(f"Досягнута точність: {eps0}")

    error_to_true = vector_norm(x_refined - x_true)
    print(f"\nВідхилення від точного розв'язку (x_i=2.5): {error_to_true:.2e}")

    print("\n" + "=" * 60)
    print("ЧАСТИНА 5: Візуалізація")
    print("=" * 60)

    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(errors) + 1), errors, 'bo-', linewidth=2, markersize=6)
    plt.xlabel("Номер ітерації", fontsize=12)
    plt.ylabel("||Δx||", fontsize=12)
    plt.title("Збіжність ітераційного уточнення", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=eps0, color='r', linestyle='--', label=f'tolerance = {eps0}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot1_convergence.png", dpi=150)
    plt.close()
    print("Збережено: plot1_convergence.png")

    residuals_initial = np.abs(A @ x_lu - b)
    residuals_refined = np.abs(A @ x_refined - b)

    plt.figure(figsize=(10, 6))
    plt.plot(range(n), residuals_initial, 'r.', alpha=0.5, label="Початковий розв'язок")
    plt.plot(range(n), residuals_refined, 'g.', alpha=0.5, label="Уточнений розв'язок")
    plt.yscale('log')
    plt.xlabel("Індекс рівняння", fontsize=12)
    plt.ylabel("|Ax - b|", fontsize=12)
    plt.title("Нев'язка для кожного рівняння", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot2_residuals.png", dpi=150)
    plt.close()
    print("Збережено: plot2_residuals.png")

    # Графік 3: Порівняння розв'язків
    plt.figure(figsize=(10, 6))
    plt.plot(range(20), x_true[:20], 'k-', linewidth=2, label="Точний розв'язок (2.5)")
    plt.plot(range(20), x_lu[:20], 'ro', markersize=4, label="LU розв'язок")
    plt.plot(range(20), x_refined[:20], 'gx', markersize=4, label="Уточнений розв'язок")
    plt.xlabel("Індекс компоненти", fontsize=12)
    plt.ylabel("Значення x_i", fontsize=12)
    plt.title("Порівняння розв'язків (перші 20 компонент)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot3_solutions_comparison.png", dpi=150)
    plt.close()
    print("Збережено: plot3_solutions_comparison.png")

    errors_distribution = np.abs(x_refined - x_true)

    plt.figure(figsize=(10, 6))
    plt.hist(errors_distribution, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Похибка |x_refined - x_true|", fontsize=12)
    plt.ylabel("Кількість компонент", fontsize=12)
    plt.title("Розподіл похибок уточненого розв'язку", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=np.mean(errors_distribution), color='r', linestyle='--',
                label=f'Середня: {np.mean(errors_distribution):.2e}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot4_error_distribution.png", dpi=150)
    plt.close()
    print("Збережено: plot4_error_distribution.png")

    print("\n" + "=" * 60)
    print("ПІДСУМКИ")
    print("=" * 60)
    print(f"Розмір матриці: {n}×{n}")
    print(f"Початкова точність: {acc:.2e}")
    print(f"Кінцева точність: {acc_refined:.2e}")
    print(f"Покращення: {acc / acc_refined:.2e} разів")
    print(f"Кількість ітерацій уточнення: {num_iters}")
    print(f"\nВсі результати збережено у папці: {os.path.abspath('.')}")

if __name__ == "__main__":
    main()