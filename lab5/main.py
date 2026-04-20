import math
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_plots_folder():
    """Створює папку для збереження графіків з часовою міткою"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"simpson_plots_{timestamp}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"\nСтворено папку для графіків: {folder_name}/")
    return folder_name

def save_plot(folder_name, filename, dpi=300):
    """Зберігає поточний графік у вказану папку"""
    full_path = os.path.join(folder_name, filename)
    plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    print(f"Збережено: {full_path}")


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a = 0.0
b = 24.0
eps_required = 1e-12


def exact_integral():
    return 1200.0 + 5.0 * math.sqrt(math.pi / 0.2) * math.erf(12.0 * math.sqrt(0.2))


I0 = exact_integral()

def simpson_integral(func, a, b, N):
    if N % 2 != 0:
        raise ValueError("Для формули Сімпсона N має бути парним.")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)

    odd_sum = np.sum(y[1:N:2])
    even_sum = np.sum(y[2:N - 1:2])

    return (h / 3.0) * (y[0] + 4.0 * odd_sum + 2.0 * even_sum + y[N])


N_values = list(range(10, 1001, 2))
I_values = []
eps_values = []

for N in N_values:
    IN = simpson_integral(f, a, b, N)
    epsN = abs(IN - I0)
    I_values.append(IN)
    eps_values.append(epsN)

Nopt = None
epsopt = None
I_Nopt = None

for N, IN, epsN in zip(N_values, I_values, eps_values):
    if epsN <= eps_required:
        Nopt = N
        I_Nopt = IN
        epsopt = epsN
        break


def choose_N0(Nopt):
    if Nopt is None:
        return 8, False
    limit = Nopt / 10.0
    candidates = [n for n in range(8, 1001, 8) if n < limit]
    if len(candidates) > 0:
        return candidates[-1], True
    return 8, False


N0, N0_condition_ok = choose_N0(Nopt)

I_N0 = simpson_integral(f, a, b, N0)
eps0 = abs(I_N0 - I0)

I_N0_half = simpson_integral(f, a, b, N0 // 2)
I_RR = I_N0 + (I_N0 - I_N0_half) / 15.0
epsR = abs(I_RR - I0)

I_N0_quarter = simpson_integral(f, a, b, N0 // 4)
numerator = abs(I_N0_half - I_N0_quarter)
denominator = abs(I_N0 - I_N0_half)

if denominator > 0 and numerator > 0:
    p_Aitken = math.log(numerator / denominator, 2.0)
else:
    p_Aitken = 1.0

I_Aitken = I_N0 + (I_N0 - I_N0_half) / (2.0 ** p_Aitken - 1.0)
epsA = abs(I_Aitken - I0)


def adaptive_simpson(func, a, b, eps):
    cache = {}
    eval_count = 0
    final_intervals = []

    def f_cached(x):
        nonlocal eval_count
        key = round(x, 12)
        if key not in cache:
            cache[key] = func(x)
            eval_count += 1
        return cache[key]

    def simpson_on_interval(left, right, f_left, f_mid, f_right):
        return (right - left) * (f_left + 4.0 * f_mid + f_right) / 6.0

    def recurse(left, right, eps_local, f_left, f_mid, f_right, S):
        mid = (left + right) / 2.0
        left_mid = (left + mid) / 2.0
        right_mid = (mid + right) / 2.0

        f_left_mid = f_cached(left_mid)
        f_right_mid = f_cached(right_mid)

        S_left = simpson_on_interval(left, mid, f_left, f_left_mid, f_mid)
        S_right = simpson_on_interval(mid, right, f_mid, f_right_mid, f_right)
        S2 = S_left + S_right

        if abs(S2 - S) <= 15.0 * eps_local:
            final_intervals.append((left, right))
            return S2 + (S2 - S) / 15.0
        else:
            left_value = recurse(left, mid, eps_local / 2.0, f_left, f_left_mid, f_mid, S_left)
            right_value = recurse(mid, right, eps_local / 2.0, f_mid, f_right_mid, f_right, S_right)
            return left_value + right_value

    fa = f_cached(a)
    fb = f_cached(b)
    m = (a + b) / 2.0
    fm = f_cached(m)

    S_initial = (b - a) * (fa + 4.0 * fm + fb) / 6.0
    result = recurse(a, b, eps, fa, fm, fb, S_initial)

    return result, eval_count, final_intervals


adaptive_eps_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
adaptive_results = []

for eps_ad in adaptive_eps_values:
    I_ad, eval_count, intervals = adaptive_simpson(f, a, b, eps_ad)
    eps_fact = abs(I_ad - I0)
    adaptive_results.append({
        "eps": eps_ad,
        "I_ad": I_ad,
        "eps_fact": eps_fact,
        "eval_count": eval_count,
        "intervals": intervals
    })

plots_folder = create_plots_folder()

print("=" * 70)
print("ЛАБОРАТОРНА РОБОТА №5")
print("Складова квадратурна формула Сімпсона")
print("=" * 70)

print("\n1. Задана функція:")
print("f(x) = 50 + 20*sin(pi*x/12) + 5*exp(-0.2*(x-12)^2), x in [0,24]")

print("\n2. Точне значення інтегралу:")
print(f"I0 = {I0:.15f}")

if Nopt is not None:
    print("\n3. Дослідження похибки для складової формули Сімпсона")
    print(f"Nopt = {Nopt}")
    print(f"I(Nopt) = {I_Nopt:.15f}")
    print(f"epsopt = |I(Nopt)-I0| = {epsopt:.15e}")
else:
    print("\n3. Дослідження похибки для складової формули Сімпсона")
    print(f"Не досягнуто потрібної точності eps_required = {eps_required}")
    print(f"Максимальна точність: {min(eps_values):.15e} при N={N_values[np.argmin(eps_values)]}")

print("\n4. Обчислення для N0")
if N0_condition_ok:
    print(f"Умова N0 < Nopt/10 виконується. Взято N0 = {N0}")
else:
    print("Умова N0 < Nopt/10 для кратного 8 числа не виконується.")
    print(f"Тому взято мінімальне кратне 8: N0 = {N0}")

print(f"I(N0) = {I_N0:.15f}")
print(f"eps0 = |I(N0)-I0| = {eps0:.15e}")

print("\n5. Метод Рунге-Ромберга")
print(f"I(N0/2) = {I_N0_half:.15f}")
print(f"I_RR = {I_RR:.15f}")
print(f"epsR = |I_RR - I0| = {epsR:.15e}")

print("\n6. Метод Ейткена")
print(f"I(N0/4) = {I_N0_quarter:.15f}")
print(f"Порядок p = {p_Aitken:.15f}")
print(f"I_A = {I_Aitken:.15f}")
print(f"epsA = |I_A - I0| = {epsA:.15e}")

print("\n7. Адаптивний алгоритм")
for row in adaptive_results:
    print(
        f"eps={row['eps']:.0e}, "
        f"I_ad={row['I_ad']:.15f}, "
        f"eps_fact={row['eps_fact']:.15e}, "
        f"eval_count={row['eval_count']}"
    )

x_plot = np.linspace(a, b, 2000)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label=r"$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$")
plt.title("Графік функції навантаження на сервер")
plt.xlabel("Час, x (год)")
plt.ylabel("Навантаження, f(x)")
plt.grid(True)
plt.legend()
save_plot(plots_folder, "1_function_plot.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label="f(x)")
plt.fill_between(x_plot, y_plot, alpha=0.25, label="Площа, що відповідає інтегралу")
plt.title("Візуалізація означеного інтегралу")
plt.xlabel("Час, x (год)")
plt.ylabel("Навантаження, f(x)")
plt.grid(True)
plt.legend()
save_plot(plots_folder, "2_integral_visualization.png")
plt.show()

x_acc = np.linspace(a, b, 4000)
y_acc = f(x_acc)

F_acc = np.zeros_like(x_acc)
for i in range(1, len(x_acc)):
    dx = x_acc[i] - x_acc[i - 1]
    F_acc[i] = F_acc[i - 1] + (y_acc[i - 1] + y_acc[i]) * dx / 2.0

plt.figure(figsize=(10, 6))
plt.plot(x_acc, F_acc)
plt.title("Накопичений інтеграл")
plt.xlabel("Час, x (год)")
plt.ylabel(r"$\int_0^x f(t)\,dt$")
plt.grid(True)
save_plot(plots_folder, "3_cumulative_integral.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(N_values, I_values, label="I(N)")
plt.axhline(I0, linestyle="--", label="Точне значення I0")
plt.title("Залежність наближеного інтегралу від числа розбиттів N")
plt.xlabel("N")
plt.ylabel("I(N)")
plt.grid(True)
plt.legend()
save_plot(plots_folder, "4_I_vs_N.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(N_values, eps_values, label=r"$\varepsilon(N)=|I(N)-I_0|$")
plt.axhline(eps_required, linestyle="--", label=r"$\varepsilon=10^{-12}$")

if Nopt is not None:
    plt.axvline(Nopt, linestyle="--", label=f"Nopt = {Nopt}")

plt.title("Залежність похибки від числа розбиттів N")
plt.xlabel("N")
plt.ylabel("Похибка")
plt.grid(True, which="both")
plt.legend()
save_plot(plots_folder, "5_error_vs_N.png")
plt.show()

sample_N = [10, 20, 50, 100, 200, 500, 1000]
sample_I = [simpson_integral(f, a, b, N) for N in sample_N]

x_bar = np.arange(len(sample_N) + 1)
labels = [f"N={N}" for N in sample_N] + ["I0"]
values = sample_I + [I0]

plt.figure(figsize=(11, 6))
plt.bar(x_bar, values)
plt.xticks(x_bar, labels, rotation=30)
plt.title("Порівняння наближених значень інтегралу з точним")
plt.ylabel("Значення інтегралу")
plt.grid(True, axis="y")
save_plot(plots_folder, "6_comparison_bar.png")
plt.show()

method_names = ["Сімпсон N0", "Рунге-Ромберг", "Ейткен"]
method_errors = [eps0, epsR, epsA]

plt.figure(figsize=(9, 6))
plt.bar(method_names, method_errors)
plt.yscale("log")
plt.title("Порівняння похибок різних методів уточнення")
plt.ylabel("Похибка")
plt.grid(True, axis="y", which="both")
save_plot(plots_folder, "7_methods_comparison.png")
plt.show()

adaptive_eps_plot = [row["eps"] for row in adaptive_results]
adaptive_err_plot = [row["eps_fact"] for row in adaptive_results]
plt.figure(figsize=(10, 6))
plt.loglog(adaptive_eps_plot, adaptive_err_plot, marker="o", label="Фактична похибка")
plt.loglog(adaptive_eps_plot, adaptive_eps_plot, linestyle="--", label="Лінія y=x")
plt.title("Залежність фактичної похибки адаптивного алгоритму від eps")
plt.xlabel("Задане eps")
plt.ylabel("Фактична похибка")
plt.grid(True, which="both")
plt.legend()
save_plot(plots_folder, "8_adaptive_error.png")
plt.show()

adaptive_eval_plot = [row["eval_count"] for row in adaptive_results]

plt.figure(figsize=(10, 6))
plt.semilogx(adaptive_eps_plot, adaptive_eval_plot, marker="o")
plt.title("Залежність кількості обчислень f(x) від eps")
plt.xlabel("Задане eps")
plt.ylabel("Кількість обчислень f(x)")
plt.grid(True, which="both")
save_plot(plots_folder, "9_adaptive_evaluations.png")
plt.show()

if len(adaptive_results) > 0:
    best_adaptive = adaptive_results[-1]
    intervals = best_adaptive["intervals"]

    centers = [(left + right) / 2.0 for left, right in intervals]
    widths = [right - left for left, right in intervals]

    plt.figure(figsize=(11, 6))
    plt.bar(centers, widths, width=widths, align="center")
    plt.title("Ширини кінцевих підвідрізків адаптивного алгоритму")
    plt.xlabel("Положення підвідрізка")
    plt.ylabel("Довжина підвідрізка")
    plt.grid(True)
    save_plot(plots_folder, "10_adaptive_intervals.png")
    plt.show()

if N0 is not None:
    x_N0 = np.linspace(a, b, N0 + 1)
    y_N0 = f(x_N0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label="f(x)")
    plt.plot(x_N0, y_N0, "o", label=f"Вузли Сімпсона для N0={N0}")
    plt.title(f"Вузли складової формули Сімпсона при N0={N0}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    save_plot(plots_folder, "11_simpson_nodes.png")
    plt.show()

print("\n" + "=" * 70)
print("ПІДСУМКОВІ РЕЗУЛЬТАТИ")
print("=" * 70)
print(f"{'Точне значення I0':35s} = {I0:.15f}")
if Nopt is not None:
    print(f"{'Nopt':35s} = {Nopt}")
    print(f"{'I(Nopt)':35s} = {I_Nopt:.15f}")
    print(f"{'epsopt':35s} = {epsopt:.15e}")
else:
    print(f"{'Nopt':35s} = Не досягнуто")
print(f"{'N0':35s} = {N0}")
print(f"{'I(N0)':35s} = {I_N0:.15f}")
print(f"{'eps0':35s} = {eps0:.15e}")
print(f"{'I_RR':35s} = {I_RR:.15f}")
print(f"{'epsR':35s} = {epsR:.15e}")
print(f"{'p_Aitken':35s} = {p_Aitken:.15f}")
print(f"{'I_Aitken':35s} = {I_Aitken:.15f}")
print(f"{'epsA':35s} = {epsA:.15e}")
