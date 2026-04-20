import os
import numpy as np
import matplotlib.pyplot as plt

x0 = 1.0
h_fixed = 1e-3
plots_dir = "lab5_plots"

os.makedirs(plots_dir, exist_ok=True)

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_diff(step, x=x0):
    return (M(x + step) - M(x - step)) / (2 * step)

with np.errstate(all="ignore"):
    # Багато значень h, щоб дослідити, як змінюється похибка
    h_values = np.logspace(-20, 3, 2000)
    y_num_values = central_diff(h_values)

y_exact = dM_exact(x0)

finite_mask = np.isfinite(h_values) & np.isfinite(y_num_values)
h_values = h_values[finite_mask]
y_num_values = y_num_values[finite_mask]

# Абсолютна похибка для кожного h
R_values = np.abs(y_num_values - y_exact)
deviation_values = y_num_values - y_exact
best_idx = np.argmin(R_values)
h0 = h_values[best_idx]
R0 = R_values[best_idx]
y0_h0 = y_num_values[best_idx]

h = h_fixed

y0_h = central_diff(h)
y0_2h = central_diff(2 * h)
y0_4h = central_diff(4 * h)

R1 = abs(y0_h - y_exact)

# Метод Рунге-Ромберга:
y_R = y0_h + (y0_h - y0_2h) / 3
R2 = abs(y_R - y_exact)

denominator_eitken = y0_4h - 2 * y0_2h + y0_h
denominator_p = y0_2h - y0_h

if np.isclose(denominator_eitken, 0.0) or np.isclose(denominator_p, 0.0):
    y_E = np.nan
    p = np.nan
    R3 = np.nan
else:
    # Метод Ейткена:
    y_E = ((y0_2h ** 2) - y0_4h * y0_h) / (2 * y0_2h - (y0_4h + y0_h))
    p = np.log(np.abs((y0_4h - y0_2h) / (y0_2h - y0_h))) / np.log(2)
    R3 = abs(y_E - y_exact)

with np.errstate(all="ignore"):
    h_step_change = np.logspace(-20, 2, 1800)
    step_change_values = np.abs(
        central_diff(h_step_change) - central_diff(2 * h_step_change)
    )

step_mask = (
    np.isfinite(h_step_change)
    & np.isfinite(step_change_values)
    & (step_change_values > 0)
)
h_step_change = h_step_change[step_mask]
step_change_values = step_change_values[step_mask]

t_plot = np.linspace(0, 20, 4000)
M_plot = M(t_plot)
dM_plot = dM_exact(t_plot)

# Найшвидше висихання: там, де похідна найменша
idx_fastest_drying = np.argmin(dM_plot)
t_fastest_drying = t_plot[idx_fastest_drying]
d_fastest_drying = dM_plot[idx_fastest_drying]

# Де похідна від’ємна — там вологість зменшується
negative_mask = dM_plot < 0
drying_intervals = []
inside = False
start_interval = None

for i in range(len(t_plot)):
    if negative_mask[i] and not inside:
        start_interval = t_plot[i]
        inside = True
    elif not negative_mask[i] and inside:
        drying_intervals.append((start_interval, t_plot[i - 1]))
        inside = False

if inside:
    drying_intervals.append((start_interval, t_plot[-1]))

print("=" * 72)
print("ЛАБОРАТОРНА РОБОТА №5")
print("=" * 72)

print("\n1. АНАЛІТИЧНЕ РОЗВ'ЯЗАННЯ")
print("M(t)  = 50*exp(-0.1*t) + 5*sin(t)")
print("M'(t) = -5*exp(-0.1*t) + 5*cos(t)")
print(f"Точне значення M'({x0}) = {y_exact:.15f}")

print("\n2. ДОСЛІДЖЕННЯ ПОХИБКИ ВІД h")
print(f"Оптимальний крок h0 ≈ {h0:.15e}")
print(f"Найкраще наближене значення y0'(h0) = {y0_h0:.15f}")
print(f"Досягнута точність R0 = |y0'(h0) - y'(x0)| = {R0:.15e}")

print("\n3. ПРИЙНЯТИЙ КРОК")
print(f"h = {h:.15e}")

print("\n4. ЗНАЧЕННЯ ПОХІДНОЇ ДЛЯ h І 2h")
print(f"y0'(h)   = {y0_h:.15f}")
print(f"y0'(2h)  = {y0_2h:.15f}")
print(f"y0'(4h)  = {y0_4h:.15f}")

print("\n5. ПОХИБКА ПРИ КРОЦІ h")
print(f"R1 = |y0'(h) - y'(x0)| = {R1:.15e}")

print("\n6. МЕТОД РУНГЕ-РОМБЕРГА")
print(f"y_R = {y_R:.15f}")
print(f"R2  = |y_R - y'(x0)| = {R2:.15e}")
print(f"Зменшення похибки R1/R2 ≈ {R1 / R2:.6f} разів")

print("\n7. МЕТОД ЕЙТКЕНА")
print(f"y_E = {y_E:.15f}")
print(f"p   = {p:.15f}")
print(f"R3  = |y_E - y'(x0)| = {R3:.15e}")
print(f"Зменшення похибки R1/R3 ≈ {R1 / R3:.6f} разів")

print("\nДОДАТКОВИЙ АНАЛІЗ РЕЖИМІВ ПОЛИВУ")
print(f"Найшвидше висихання спостерігається приблизно при t ≈ {t_fastest_drying:.6f}")
print(f"У цей момент M'(t) ≈ {d_fastest_drying:.15f}")

print("\nІнтервали, де M'(t) < 0 (вологість зменшується):")
for i, (a, b) in enumerate(drying_intervals, start=1):
    print(f"{i}) [{a:.6f}; {b:.6f}]")
    print("\nBackend matplotlib:", plt.get_backend())

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, M_plot, linewidth=2)
    plt.axvline(x=x0, linestyle="--", linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("M(t)")
    plt.title("Графік функції вологості ґрунту M(t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_function_M_t.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, dM_plot, linewidth=2)
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.axvline(t_fastest_drying, linestyle="--", linewidth=1.2)
    for a, b in drying_intervals:
        plt.axvspan(a, b, alpha=0.12)
    plt.xlabel("t")
    plt.ylabel("M'(t)")
    plt.title("Швидкість зміни вологості ґрунту")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_derivative_over_time.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, R_values, linewidth=2)
    plt.scatter([h0], [R0], zorder=5)
    plt.xlabel("h")
    plt.ylabel("R = |y0'(h) - y'(x0)|")
    plt.title("Залежність похибки чисельного диференціювання від кроку h")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_error_vs_h.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(h_values, deviation_values, linewidth=1.5)
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-16)
    plt.xlabel("h")
    plt.ylabel("Δ(h) = y0'(h) - y'(x0)")
    plt.title("Відхилення чисельної похідної від точної")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_deviation_vs_h.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.loglog(h_step_change, step_change_values, linewidth=2)
    plt.xlabel("h")
    plt.ylabel("|y0'(h) - y0'(2h)|")
    plt.title("Зміна оцінки похідної при зміні кроку")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "05_step_change_vs_h.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(h_values, y_num_values, linewidth=1.5, label="y0'(h)")
    plt.axhline(y_exact, linestyle="--", linewidth=1.2, label="Точне значення")
    plt.xscale("log")
    plt.xlabel("h")
    plt.ylabel("Значення похідної")
    plt.title("Чисельна похідна залежно від кроку h")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "06_derivative_vs_h.png"), dpi=300)

    method_names = ["R1: центральна різниця", "R2: Рунге-Ромберг", "R3: Ейткен"]
    method_errors = [R1, R2, R3]

    plt.figure(figsize=(10, 6))
    plt.bar(method_names, method_errors)
    plt.yscale("log")
    plt.ylabel("Абсолютна похибка")
    plt.title("Порівняння похибок чисельних методів")
    plt.grid(True, axis="y", which="both", alpha=0.3)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "07_methods_error_compare.png"), dpi=300)

    value_names = ["Точне", "y0'(h)", "y0'(2h)", "y0'(4h)", "y_R", "y_E"]
    value_data = [y_exact, y0_h, y0_2h, y0_4h, y_R, y_E]

    plt.figure(figsize=(11, 6))
    plt.plot(value_names, value_data, marker="o", linewidth=2)
    plt.ylabel("Значення похідної")
    plt.title("Порівняння точного та чисельних значень похідної")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "08_derivative_values_compare.png"), dpi=300)

    print("\nУсі графіки збережено в папку:", plots_dir)
    print("Список файлів:")
    for filename in sorted(os.listdir(plots_dir)):
        print(" -", filename)

    plt.show(block=True)