#!/usr/bin/env python3
"""
ZOF_CLI.py
Zero of Functions (ZOF) Solver - CLI
Supports: Bisection, Regula Falsi, Secant, Newton-Raphson, Fixed Point, Modified Secant
Requires: sympy, numpy
"""

import sys
import math
from sympy import sympify, Symbol, lambdify, diff
import numpy as np

x = Symbol('x')

def parse_function(expr_str):
    try:
        expr = sympify(expr_str)
        f = lambdify(x, expr, 'numpy')
        return expr, f
    except Exception as e:
        print("Error parsing function:", e)
        return None, None

def print_iter_header():
    print(f"{'k':>3} | {'x_k':>18} | {'f(x_k)':>18} | {'error':>12} | extra")
    print("-"*70)

def bisection(f, a, b, tol, maxit):
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Function returned NaN on interval endpoints.")
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection.")
    print_iter_header()
    c = a
    for k in range(1, maxit+1):
        c = (a + b)/2.0
        fc = f(c)
        err = abs(b-a)/2.0
        print(f"{k:3d} | {c:18.10f} | {fc:18.10e} | {err:12.4e} | interval=[{a:.6g},{b:.6g}]")
        if abs(fc) == 0 or err < tol:
            return c, fc, k
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, fc, maxit

def regula_falsi(f, a, b, tol, maxit):
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    print_iter_header()
    c = a
    for k in range(1, maxit+1):
        c = (a*fb - b*fa) / (fb - fa)
        fc = f(c)
        err = abs(fc)
        print(f"{k:3d} | {c:18.10f} | {fc:18.10e} | {err:12.4e} | a={a:.6g}, b={b:.6g}")
        if abs(fc) < tol:
            return c, fc, k
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, fc, maxit

def secant(f, x0, x1, tol, maxit):
    print_iter_header()
    f0, f1 = f(x0), f(x1)
    for k in range(1, maxit+1):
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Division by zero in Secant method.")
        x2 = x1 - f1*(x1-x0)/(f1 - f0)
        err = abs(x2 - x1)
        fx2 = f(x2)
        print(f"{k:3d} | {x2:18.10f} | {fx2:18.10e} | {err:12.4e} | x0={x0:.6g}, x1={x1:.6g}")
        if err < tol or abs(fx2) < tol:
            return x2, fx2, k
        x0, f0 = x1, f1
        x1, f1 = x2, fx2
    return x2, fx2, maxit

def newton_raphson(f, df, x0, tol, maxit):
    print_iter_header()
    xi = x0
    for k in range(1, maxit+1):
        fxi = f(xi)
        dfxi = df(xi)
        if dfxi == 0:
            raise ZeroDivisionError("Zero derivative encountered in Newton-Raphson.")
        x_next = xi - fxi/dfxi
        err = abs(x_next - xi)
        print(f"{k:3d} | {x_next:18.10f} | {f(x_next):18.10e} | {err:12.4e} | f'={dfxi:.6g}")
        if err < tol or abs(f(x_next)) < tol:
            return x_next, f(x_next), k
        xi = x_next
    return xi, f(xi), maxit

def fixed_point(g, x0, tol, maxit):
    print_iter_header()
    xi = x0
    for k in range(1, maxit+1):
        x_next = g(xi)
        err = abs(x_next - xi)
        try:
            gx = g(x_next)
        except Exception:
            gx = float('nan')
        print(f"{k:3d} | {x_next:18.10f} | {gx:18.10e} | {err:12.4e} | g(x) used")
        if err < tol:
            return x_next, gx, k
        xi = x_next
    return xi, g(xi), maxit

def modified_secant(f, x0, delta, tol, maxit):
    print_iter_header()
    xi = x0
    for k in range(1, maxit+1):
        perturb = x0 * delta if x0 != 0 else delta
        denom = f(xi + perturb) - f(xi)
        if denom == 0:
            raise ZeroDivisionError("Denominator zero in Modified Secant.")
        x_next = xi - f(xi) * perturb / denom
        err = abs(x_next - xi)
        print(f"{k:3d} | {x_next:18.10f} | {f(x_next):18.10e} | {err:12.4e} | delta={delta}")
        if err < tol or abs(f(x_next)) < tol:
            return x_next, f(x_next), k
        xi = x_next
    return xi, f(xi), maxit

def get_float(prompt, default=None):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            return float(s)
        except:
            print("Enter a valid number.")

def main():
    print("ZOF_CLI - Zero of Functions Solver")
    print("Enter function in variable x (example: x**3 - 2*x - 5). Use Python syntax.")
    func_str = input("f(x) = ").strip()
    expr, f = parse_function(func_str)
    if f is None:
        return
    # derivative for Newton
    df_expr = diff(expr, x)
    df = lambdify(x, df_expr, 'numpy')

    print("\nChoose method:")
    print("1) Bisection")
    print("2) Regula Falsi (False Position)")
    print("3) Secant")
    print("4) Newton-Raphson")
    print("5) Fixed Point Iteration (requires g(x))")
    print("6) Modified Secant")
    choice = input("Choice (1-6): ").strip()
    try:
        choice = int(choice)
    except:
        print("Invalid choice."); return

    tol = get_float("Tolerance (default 1e-8): ", 1e-8)
    maxit = int(get_float("Max iterations (default 50): ", 50))

    try:
        if choice == 1:
            a = get_float("a = ")
            b = get_float("b = ")
            root, froot, iterations = bisection(f, a, b, tol, maxit)
        elif choice == 2:
            a = get_float("a = ")
            b = get_float("b = ")
            root, froot, iterations = regula_falsi(f, a, b, tol, maxit)
        elif choice == 3:
            x0 = get_float("x0 = ")
            x1 = get_float("x1 = ")
            root, froot, iterations = secant(f, x0, x1, tol, maxit)
        elif choice == 4:
            x0 = get_float("initial x0 = ")
            root, froot, iterations = newton_raphson(f, df, x0, tol, maxit)
        elif choice == 5:
            print("Enter g(x) for Fixed Point iteration (x_{n+1} = g(x_n)).")
            g_str = input("g(x) = ").strip()
            g_expr, g_func = parse_function(g_str)
            if g_func is None:
                return
            x0 = get_float("initial x0 = ")
            root, froot, iterations = fixed_point(g_func, x0, tol, maxit)
        elif choice == 6:
            x0 = get_float("initial x0 = ")
            delta = get_float("delta (perturbation fraction, e.g. 1e-3): ", 1e-3)
            root, froot, iterations = modified_secant(f, x0, delta, tol, maxit)
        else:
            print("Invalid choice."); return
    except Exception as e:
        print("Method failed:", e)
        return

    print("\nFinal result:")
    print(f"Estimated root: {root:.12f}")
    print(f"f(root) = {froot:.4e}")
    print(f"Iterations: {iterations}")

if __name__ == "__main__":
    main()
