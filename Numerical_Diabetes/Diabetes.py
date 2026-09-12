import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================
# Model Parameters (from Schiesser Chapter 2)
# ============================================
Ex = 15000   # Extracellular space (ml)
Cg = 150     # Glucose capacitance = Ex/100
Ci = 150     # Insulin capacitance = Ex/100
Q  = 8400    # Liver glucose release rate (mg/hr)
Dd = 24.7    # First-order glucose loss rate
Gg = 13.9    # Controlled glucose loss rate
Gk = 250     # Renal threshold (mg glucose/100ml)
Mu = 72      # Renal loss rate
G0 = 51      # Pancreas threshold (mg glucose/100ml)
Aa = 76      # First-order insulin reduction rate

# ============================================
# Initial Conditions
# ============================================
G_init = 81.14    # Initial glucose concentration
I_init = 5.671    # Initial insulin concentration

# ============================================
# ODE System Definition
# ============================================
def glucose_insulin_ode(t, y, Bb, Gt):
    """
    Defines the glucose-insulin ODE system.

    Parameters:
        t  : current time (hours)
        y  : state vector [G, I]
        Bb : pancreas insulin release rate (varies per case)
        Gt : glucose infusion rate (mg/hr)

    Returns:
        [dGdt, dIdt] : derivatives
    """
    G, I = y

    # Glucose infusion function: active only for t <= 0.5 hr
    In = Gt if t <= 0.5 else 0.0

    # --- Glucose ODE ---
    # Below renal threshold: no kidney removal
    if G < Gk:
        dGdt = (1/Cg) * (Q + In - Gg*I*G - Dd*G)
    # Above renal threshold: kidney removes excess glucose
    else:
        dGdt = (1/Cg) * (Q + In - Gg*I*G - Dd*G - Mu*(G - Gk))

    # --- Insulin ODE ---
    # Below pancreas threshold: no insulin production
    if G < G0:
        dIdt = (1/Ci) * (-Aa*I)
    # Above pancreas threshold: pancreas releases insulin
    else:
        dIdt = (1/Ci) * (-Aa*I + Bb*(G - G0))

    return [dGdt, dIdt]

# ============================================
# Four Test Cases (matching Schiesser Table 2.1)
# ============================================
cases = [
    {
        "ncase" : 1,
        "Bb"    : 14.3,
        "Gt"    : 0,
        "label" : "Case 1: No Infusion (Baseline)"
    },
    {
        "ncase" : 2,
        "Bb"    : 14.3,
        "Gt"    : 80000,
        "label" : "Case 2: Normal Pancreatic Sensitivity"
    },
    {
        "ncase" : 3,
        "Bb"    : 0.2 * 14.3,
        "Gt"    : 80000,
        "label" : "Case 3: Reduced Sensitivity (Type I Diabetes)"
    },
    {
        "ncase" : 4,
        "Bb"    : 2.0 * 14.3,
        "Gt"    : 80000,
        "label" : "Case 4: Elevated Pancreatic Sensitivity"
    },
]

# ============================================
# Time Setup
# ============================================
t_start = 0
t_end   = 12                              # 12 hours total
t_eval  = np.linspace(t_start, t_end, 500)
y0      = [G_init, I_init]

# Time points to print in numerical output
print_times = [0, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12]

# ============================================
# Solve ODEs and Store Results
# ============================================
results = {}

for i, case in enumerate(cases):
    sol = solve_ivp(
        fun     = lambda t, y, Bb=case["Bb"], Gt=case["Gt"]:
                      glucose_insulin_ode(t, y, Bb, Gt),
        t_span  = (t_start, t_end),
        y0      = y0,
        t_eval  = t_eval,
        method  = 'RK45',
        rtol    = 1e-8,
        atol    = 1e-8
    )
    results[i] = sol

# ============================================
# Compute Derivatives for All Cases
# ============================================
# Re-evaluate the ODE at each solution point to get dG/dt and dI/dt
derivatives = {}

for i, case in enumerate(cases):
    sol = results[i]
    dGdt_arr = np.zeros(len(sol.t))
    dIdt_arr = np.zeros(len(sol.t))

    for j in range(len(sol.t)):
        dydt = glucose_insulin_ode(
            sol.t[j],
            [sol.y[0][j], sol.y[1][j]],
            case["Bb"],
            case["Gt"]
        )
        dGdt_arr[j] = dydt[0]
        dIdt_arr[j] = dydt[1]

    derivatives[i] = {"dGdt": dGdt_arr, "dIdt": dIdt_arr}

# ============================================
# Figure 1: Glucose G(t) - All Four Cases
# ============================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()

for i, case in enumerate(cases):
    sol = results[i]
    axes1[i].plot(sol.t, sol.y[0], 'b-', linewidth=2)
    axes1[i].set_title(case["label"] + "\nGlucose G(t)", fontsize=11)
    axes1[i].set_xlabel("Time (hours)")
    axes1[i].set_ylabel("G(t)  [mg glucose / 100 ml]")
    axes1[i].set_xlim([0, 12])
    axes1[i].set_ylim([0, 300])
    axes1[i].grid(True, alpha=0.3)
    axes1[i].axvline(x=0.5, color='gray', linestyle='--',
                     linewidth=1, label='Infusion ends')
    axes1[i].legend(fontsize=8)

fig1.suptitle("Extracellular Glucose G(t) — Cases 1 to 4",
              fontsize=14, fontweight='bold')
fig1.tight_layout()

# ============================================
# Figure 2: Insulin I(t) - All Four Cases
# ============================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for i, case in enumerate(cases):
    sol = results[i]
    axes2[i].plot(sol.t, sol.y[1], 'r-', linewidth=2)
    axes2[i].set_title(case["label"] + "\nInsulin I(t)", fontsize=11)
    axes2[i].set_xlabel("Time (hours)")
    axes2[i].set_ylabel("I(t)  [mg insulin / 100 ml]")
    axes2[i].set_xlim([0, 12])
    axes2[i].grid(True, alpha=0.3)
    axes2[i].axvline(x=0.5, color='gray', linestyle='--',
                     linewidth=1, label='Infusion ends')
    axes2[i].legend(fontsize=8)

fig2.suptitle("Extracellular Insulin I(t) — Cases 1 to 4",
              fontsize=14, fontweight='bold')
fig2.tight_layout()

# ============================================
# Figure 3: Combined Plot (G and I on same axes)
# ============================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
colors = ['blue', 'green', 'red', 'orange']
labels = [f"Case {c['ncase']}: Bb={c['Bb']:.2f}" for c in cases]

for i, case in enumerate(cases):
    sol = results[i]
    axes3[0].plot(sol.t, sol.y[0], color=colors[i],
                  linewidth=2, label=labels[i])
    axes3[1].plot(sol.t, sol.y[1], color=colors[i],
                  linewidth=2, label=labels[i])

axes3[0].set_title("Glucose G(t) — All Cases", fontsize=12)
axes3[0].set_xlabel("Time (hours)")
axes3[0].set_ylabel("G(t)  [mg glucose / 100 ml]")
axes3[0].set_xlim([0, 12])
axes3[0].set_ylim([0, 300])
axes3[0].legend(fontsize=9)
axes3[0].grid(True, alpha=0.3)

axes3[1].set_title("Insulin I(t) — All Cases", fontsize=12)
axes3[1].set_xlabel("Time (hours)")
axes3[1].set_ylabel("I(t)  [mg insulin / 100 ml]")
axes3[1].set_xlim([0, 12])
axes3[1].legend(fontsize=9)
axes3[1].grid(True, alpha=0.3)

fig3.suptitle("Glucose-Insulin ODE Model — Schiesser Chapter 2",
              fontsize=13, fontweight='bold')
fig3.tight_layout()

# ============================================
# Figure 4: dG/dt — All Four Cases (subplots)
# ============================================
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
axes4 = axes4.flatten()

for i, case in enumerate(cases):
    sol  = results[i]
    dGdt = derivatives[i]["dGdt"]
    axes4[i].plot(sol.t, dGdt, color='darkblue', linewidth=2)
    axes4[i].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes4[i].axvline(x=0.5, color='gray', linestyle='--',
                     linewidth=1, label='Infusion ends')
    axes4[i].set_title(case["label"] + "\ndG/dt", fontsize=11)
    axes4[i].set_xlabel("Time (hours)")
    axes4[i].set_ylabel("dG/dt  [mg glucose / (100 ml · hr)]")
    axes4[i].set_xlim([0, 12])
    axes4[i].grid(True, alpha=0.3)
    axes4[i].legend(fontsize=8)

fig4.suptitle("Glucose Rate of Change dG/dt — Cases 1 to 4",
              fontsize=14, fontweight='bold')
fig4.tight_layout()

# ============================================
# Figure 5: dI/dt — All Four Cases (subplots)
# ============================================
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
axes5 = axes5.flatten()

for i, case in enumerate(cases):
    sol  = results[i]
    dIdt = derivatives[i]["dIdt"]
    axes5[i].plot(sol.t, dIdt, color='darkred', linewidth=2)
    axes5[i].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes5[i].axvline(x=0.5, color='gray', linestyle='--',
                     linewidth=1, label='Infusion ends')
    axes5[i].set_title(case["label"] + "\ndI/dt", fontsize=11)
    axes5[i].set_xlabel("Time (hours)")
    axes5[i].set_ylabel("dI/dt  [mg insulin / (100 ml · hr)]")
    axes5[i].set_xlim([0, 12])
    axes5[i].grid(True, alpha=0.3)
    axes5[i].legend(fontsize=8)

fig5.suptitle("Insulin Rate of Change dI/dt — Cases 1 to 4",
              fontsize=14, fontweight='bold')
fig5.tight_layout()

# ============================================
# Figure 6: Combined Derivative Plot (overlay)
# ============================================
fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))

for i, case in enumerate(cases):
    sol  = results[i]
    dGdt = derivatives[i]["dGdt"]
    dIdt = derivatives[i]["dIdt"]
    axes6[0].plot(sol.t, dGdt, color=colors[i],
                  linewidth=2, label=labels[i])
    axes6[1].plot(sol.t, dIdt, color=colors[i],
                  linewidth=2, label=labels[i])

axes6[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes6[0].set_title("dG/dt — All Cases", fontsize=12)
axes6[0].set_xlabel("Time (hours)")
axes6[0].set_ylabel("dG/dt  [mg glucose / (100 ml · hr)]")
axes6[0].set_xlim([0, 12])
axes6[0].legend(fontsize=9)
axes6[0].grid(True, alpha=0.3)

axes6[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes6[1].set_title("dI/dt — All Cases", fontsize=12)
axes6[1].set_xlabel("Time (hours)")
axes6[1].set_ylabel("dI/dt  [mg insulin / (100 ml · hr)]")
axes6[1].set_xlim([0, 12])
axes6[1].legend(fontsize=9)
axes6[1].grid(True, alpha=0.3)

fig6.suptitle("Derivative Rates dG/dt and dI/dt — All Cases Overlaid",
              fontsize=13, fontweight='bold')
fig6.tight_layout()

plt.show()

# ============================================
# Numerical Output Table (matches book Table 2.1)
# ============================================
print("\n" + "=" * 65)
print("   NUMERICAL OUTPUT — Schiesser Chapter 2 Reproduction")
print("=" * 65)

for i, case in enumerate(cases):
    sol = results[i]
    print(f"\nCase {case['ncase']}:  Bb = {case['Bb']:.3f},  "
          f"Gt = {case['Gt']}")
    print(f"  {'t (hr)':>8}  {'G(t)':>12}  {'I(t)':>12}")
    print("  " + "-" * 38)

    for t_val in print_times:
        idx = np.argmin(np.abs(sol.t - t_val))
        print(f"  {sol.t[idx]:>8.2f}  "
              f"{sol.y[0][idx]:>12.4f}  "
              f"{sol.y[1][idx]:>12.4f}")

# ============================================
# Validation: Compare final values with book
# ============================================
print("\n" + "=" * 65)
print("   VALIDATION — Final Values at t = 12 hr")
print("=" * 65)

expected = {
    1: (81.14,  5.671),   # no change expected
    2: (81.03,  5.664),
    3: (129.22, 2.915),
    4: (69.49,  6.926),
}

print(f"\n  {'Case':<6} {'G computed':>12} {'G expected':>12} "
      f"{'I computed':>12} {'I expected':>12}")
print("  " + "-" * 58)

for i, case in enumerate(cases):
    sol   = results[i]
    G_end = sol.y[0][-1]
    I_end = sol.y[1][-1]
    G_exp, I_exp = expected[case["ncase"]]
    print(f"  {case['ncase']:<6} {G_end:>12.4f} {G_exp:>12.4f} "
          f"{I_end:>12.4f} {I_exp:>12.4f}")

print("\nDone. All figures displayed.")