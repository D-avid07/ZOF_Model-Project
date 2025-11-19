from flask import Flask, render_template, request, redirect, url_for
from sympy import sympify, Symbol, lambdify, diff
import numpy as np

app = Flask(__name__)
x = Symbol('x')

def parse_function(expr_str):
    expr = sympify(expr_str)
    f = lambdify(x, expr, 'numpy')
    return expr, f

# (Re-use the same numerical methods as CLI but return iteration list)
def bisection_list(f, a, b, tol, maxit):
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection.")
    iters = []
    for k in range(1, maxit+1):
        c = (a + b)/2.0
        fc = f(c)
        err = abs(b-a)/2.0
        iters.append((k, c, fc, err, f"[{a},{b}]"))
        if abs(fc) == 0 or err < tol:
            return iters, c, fc, k
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return iters, c, fc, maxit

def regula_falsi_list(f, a, b, tol, maxit):
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    iters = []
    for k in range(1, maxit+1):
        c = (a*fb - b*fa) / (fb - fa)
        fc = f(c)
        err = abs(fc)
        iters.append((k, c, fc, err, f"a={a}, b={b}"))
        if abs(fc) < tol:
            return iters, c, fc, k
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return iters, c, fc, maxit

def secant_list(f, x0, x1, tol, maxit):
    iters = []
    f0, f1 = f(x0), f(x1)
    for k in range(1, maxit+1):
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Division by zero in Secant method.")
        x2 = x1 - f1*(x1-x0)/(f1 - f0)
        err = abs(x2 - x1)
        fx2 = f(x2)
        iters.append((k, x2, fx2, err, f"x0={x0}, x1={x1}"))
        if err < tol or abs(fx2) < tol:
            return iters, x2, fx2, k
        x0, f0 = x1, f1
        x1, f1 = x2, fx2
    return iters, x2, fx2, maxit

def newton_list(f, df, x0, tol, maxit):
    iters = []
    xi = x0
    for k in range(1, maxit+1):
        fxi = f(xi)
        dfxi = df(xi)
        if dfxi == 0:
            raise ZeroDivisionError("Zero derivative in Newton.")
        x_next = xi - fxi/dfxi
        err = abs(x_next - xi)
        iters.append((k, x_next, f(x_next), err, f"f'={dfxi}"))
        if err < tol or abs(f(x_next)) < tol:
            return iters, x_next, f(x_next), k
        xi = x_next
    return iters, xi, f(xi), maxit

def fixed_point_list(g, x0, tol, maxit):
    iters = []
    xi = x0
    for k in range(1, maxit+1):
        x_next = g(xi)
        err = abs(x_next - xi)
        iters.append((k, x_next, g(x_next) if callable(g) else None, err, "g(x)"))
        if err < tol:
            return iters, x_next, g(x_next), k
        xi = x_next
    return iters, xi, g(xi), maxit

def modified_secant_list(f, x0, delta, tol, maxit):
    iters = []
    xi = x0
    for k in range(1, maxit+1):
        perturb = x0 * delta if x0 != 0 else delta
        denom = f(xi + perturb) - f(xi)
        if denom == 0:
            raise ZeroDivisionError("Denominator zero in Modified Secant.")
        x_next = xi - f(xi) * perturb / denom
        err = abs(x_next - xi)
        iters.append((k, x_next, f(x_next), err, f"delta={delta}"))
        if err < tol or abs(f(x_next)) < tol:
            return iters, x_next, f(x_next), k
        xi = x_next
    return iters, xi, f(xi), maxit

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        func_str = request.form.get('function')
        method = request.form.get('method')
        tol = float(request.form.get('tolerance') or 1e-8)
        maxit = int(request.form.get('maxit') or 50)
        try:
            expr, f = parse_function(func_str)
            if method in ['bisection', 'regulafalsi']:
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                if method == 'bisection':
                    iters, root, froot, its = bisection_list(f, a, b, tol, maxit)
                else:
                    iters, root, froot, its = regula_falsi_list(f, a, b, tol, maxit)
            elif method == 'secant':
                x0 = float(request.form.get('x0'))
                x1 = float(request.form.get('x1'))
                iters, root, froot, its = secant_list(f, x0, x1, tol, maxit)
            elif method == 'newton':
                x0 = float(request.form.get('x0'))
                df_expr = diff(expr, x)
                df = lambdify(x, df_expr, 'numpy')
                iters, root, froot, its = newton_list(f, df, x0, tol, maxit)
            elif method == 'fixed':
                g_str = request.form.get('gfunction')
                g_expr, g = parse_function(g_str)
                x0 = float(request.form.get('x0'))
                iters, root, froot, its = fixed_point_list(g, x0, tol, maxit)
            elif method == 'modified_secant':
                x0 = float(request.form.get('x0'))
                delta = float(request.form.get('delta') or 1e-3)
                iters, root, froot, its = modified_secant_list(f, x0, delta, tol, maxit)
            else:
                raise ValueError("Unknown method")
            result = {
                'iters': iters,
                'root': root,
                'froot': froot,
                'its': its,
                'function': func_str,
                'method': method
            }
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
