import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_matrix_to_file(matrix, filename):
    np.savetxt(filename, matrix, fmt="%.10f")


def save_vector_to_file(vector, filename):
    np.savetxt(filename, vector.reshape(-1, 1), fmt="%.10f")


def read_matrix_from_file(filename):
    return np.loadtxt(filename)


def read_vector_from_file(filename):
    return np.loadtxt(filename)


def matrix_vector_product(matrix, vector):
    return matrix @ vector


def vector_norm(vector):
    return np.max(np.abs(vector))


def matrix_norm(matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))


def generate_diagonally_dominant_matrix(n, seed=42):
    np.random.seed(seed)
    A = np.random.randint(-3, 4, size=(n, n)).astype(float)
    A = np.triu(A, 1) + np.triu(A, 1).T
    row_sums = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, row_sums + np.random.randint(20, 31, size=n))
    return A


def simple_iteration_method(A, b, x0, eps, max_iter, x_exact):
    eigvals = np.linalg.eigvalsh(A)
    tau = 2.0 / (np.min(eigvals) + np.max(eigvals))
    x = x0.copy()
    hist = {"residual": [], "error": []}

    for i in range(1, max_iter + 1):
        x_new = x - tau * (A @ x - b)
        resid = vector_norm(A @ x_new - b)
        hist["residual"].append(resid)
        hist["error"].append(vector_norm(x_new - x_exact))
        if resid <= eps:
            return x_new, i, tau, hist
        x = x_new
    return x, max_iter, tau, hist


def jacobi_method(A, b, x0, eps, max_iter, x_exact):
    D = np.diag(A)
    R = A - np.diag(D)
    x = x0.copy()
    hist = {"residual": [], "error": []}

    for i in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        resid = vector_norm(A @ x_new - b)
        hist["residual"].append(resid)
        hist["error"].append(vector_norm(x_new - x_exact))
        if resid <= eps:
            return x_new, i, hist
        x = x_new
    return x, max_iter, hist


def seidel_method(A, b, x0, eps, max_iter, x_exact):
    n = len(A)
    x = x0.copy()
    hist = {"residual": [], "error": []}

    for i in range(1, max_iter + 1):
        x_new = x.copy()
        for j in range(n):
            x_new[j] = (b[j] - np.dot(A[j, :j], x_new[:j]) - np.dot(A[j, j + 1:], x[j + 1:])) / A[j, j]
        resid = vector_norm(A @ x_new - b)
        hist["residual"].append(resid)
        hist["error"].append(vector_norm(x_new - x_exact))
        if resid <= eps:
            return x_new, i, hist
        x = x_new
    return x, max_iter, hist


def plot_results(matrix, diag_dominance, histories, solutions, x_exact, output_folder):
    plt.figure(figsize=(12, 8))
    plt.imshow(matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Значення')
    plt.title("Теплова карта матриці A")
    plt.tight_layout()
    plt.savefig(output_folder / "01_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(diag_dominance, 'b-o', markersize=2)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Запас діагонального переважання")
    plt.xlabel("Номер рядка")
    plt.ylabel("a_ii - Σ|a_ij|")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "02_diagonal_dominance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.semilogy(hist["residual"], label=name, linewidth=1.5)
    plt.title("Норма нев'язки ||Ax - b||_∞")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма нев'язки")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "03_residual_norms.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.semilogy(hist["error"], label=name, linewidth=1.5)
    plt.title("Норма похибки ||x - x*||_∞")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма похибки")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "04_error_norms.png", dpi=150)
    plt.close()


def main():
    n = 100
    eps = 1e-14
    max_iter = 50000

    output_folder = Path("lab7_output")
    output_folder.mkdir(exist_ok=True)

    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №7 - Розв'язання СЛАР ітераційними методами")
    print("=" * 70)

    print("\n[1] Генерація матриці A (n=100) з діагональним переважанням...")
    print("   Використовується np.random.randint() для генерації випадкових чисел")
    A = generate_diagonally_dominant_matrix(n)

    print("[2] Задання точного розв'язку x_i = 2.5 та обчислення вектора b...")
    x_exact = np.full(n, 2.5)
    b = A @ x_exact

    print("[3] Запис матриці A та вектора b у файли...")
    save_matrix_to_file(A, output_folder / "matrix_A.txt")
    save_vector_to_file(b, output_folder / "vector_B.txt")

    print("[4] Читання даних з файлів...")
    A_read = read_matrix_from_file(output_folder / "matrix_A.txt")
    b_read = read_vector_from_file(output_folder / "vector_B.txt")

    print("[5] Початкове наближення x0 = 1.0...")
    x0 = np.ones(n)

    diag_dominance = np.diag(A) - (np.sum(np.abs(A), axis=1) - np.abs(np.diag(A)))

    print(f"\nПараметри системи:")
    print(f"  Розмірність: n = {n}")
    print(f"  Точність: ε = {eps:.1e}")
    print(f"  Норма матриці ||A||_∞ = {matrix_norm(A):.2f}")
    print(f"  Діагональне переважання: min = {np.min(diag_dominance):.2f}, max = {np.max(diag_dominance):.2f}")

    print("\n[6] Розв'язання СЛАР ітераційними методами...")
    print("-" * 70)

    print("\n▶ Метод простої ітерації:")
    x_simple, iter_simple, tau, hist_simple = simple_iteration_method(A, b, x0, eps, max_iter, x_exact)
    print(f"   Ітерацій: {iter_simple}, τ = {tau:.6e}")
    print(f"   Нев'язка: {hist_simple['residual'][-1]:.2e}, Похибка: {hist_simple['error'][-1]:.2e}")

    print("\n▶ Метод Якобі:")
    x_jacobi, iter_jacobi, hist_jacobi = jacobi_method(A, b, x0, eps, max_iter, x_exact)
    print(f"   Ітерацій: {iter_jacobi}")
    print(f"   Нев'язка: {hist_jacobi['residual'][-1]:.2e}, Похибка: {hist_jacobi['error'][-1]:.2e}")

    print("\n▶ Метод Зейделя:")
    x_seidel, iter_seidel, hist_seidel = seidel_method(A, b, x0, eps, max_iter, x_exact)
    print(f"   Ітерацій: {iter_seidel}")
    print(f"   Нев'язка: {hist_seidel['residual'][-1]:.2e}, Похибка: {hist_seidel['error'][-1]:.2e}")

    histories = {
        "Проста ітерація": hist_simple,
        "Якобі": hist_jacobi,
        "Зейдель": hist_seidel
    }

    solutions = {
        "Проста ітерація": x_simple,
        "Якобі": x_jacobi,
        "Зейдель": x_seidel
    }

    print("\n[7] Порівняння методів:")
    print("-" * 70)
    print(f"{'Метод':<20} {'Ітерації':<12} {'Макс. похибка':<20}")
    print("-" * 70)
    print(f"{'Проста ітерація':<20} {iter_simple:<12} {np.max(np.abs(x_simple - x_exact)):.4e}")
    print(f"{'Якобі':<20} {iter_jacobi:<12} {np.max(np.abs(x_jacobi - x_exact)):.4e}")
    print(f"{'Зейдель':<20} {iter_seidel:<12} {np.max(np.abs(x_seidel - x_exact)):.4e}")

    print("\n[8] Побудова графіків...")
    plot_results(A, diag_dominance, histories, solutions, x_exact, output_folder)

    print("\n[9] Створені файли:")
    print("-" * 70)
    print(f"   {output_folder / 'matrix_A.txt'}")
    print(f"   {output_folder / 'vector_B.txt'}")
    for img in sorted(output_folder.glob("*.png")):
        print(f"   {img.name}")

    print("\n" + "=" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()