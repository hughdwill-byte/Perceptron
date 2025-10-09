''
# import sys
# sys.set_int_max_str_digits(1000000)  # allow up to 1M digits


def fibonacci(n: int) -> int:
    a, b = 0, 1
    
    for _ in range(n):
        next_val = a + b
        # shift window
        a, b = b, next_val
   
    return a

# import sys
# sys.set_int_max_str_digits(1000000)  # allow big Fibonacci numbers

import time, math

def fibonacci_for_time(seconds: float):
    a, b = 0, 1
    start = time.time()
    count = 0
    while time.time() - start < seconds:
        a, b = b, a + b
        count += 1
    return a, count

def sci_notation(num: int, sig_digits: int = 4) -> str:
    # Number of digits ≈ floor(log10(num)) + 1
    digits = int(num.bit_length() * math.log10(2)) + 1
    # Scale down to get the mantissa
    scale = 10 ** (digits - sig_digits)
    mantissa = num // scale
    s = str(mantissa)
    return f"{s[0]}.{s[1:]}e+{digits-1}"

# Example: run for 3 seconds
largest, terms = fibonacci_for_time(3)
print(f"Largest Fibonacci number ≈ {sci_notation(largest)}")
print(f"Terms generated: {terms}")


'''
# --------------------------------------
# field_ext
'''
import time, math

class Zrt5:
    """Represents numbers of the form a + b√5 with integer coefficients."""
    def __init__(self, a, b):
        self.a = int(a)
        self.b = int(b)

    def __mul__(self, other):
        ac = self.a * other.a
        bd = self.b * other.b
        five_bd = (bd << 2) + bd  # *5
        new_a = ac + five_bd
        new_b = self.a * other.b + self.b * other.a
        return Zrt5(new_a, new_b)

    def __imul__(self, other):
        result = self * other
        self.a, self.b = result.a, result.b
        return self

    def __irshift__(self, n):
        self.a >>= n
        self.b >>= n
        return self


def fibonacci_timed(seconds: float):
    """Run Fibonacci calculation for a time limit, return last value reached."""
    start = time.time()
    n = 1
    fib_val = 1

    while time.time() - start < seconds:
        fib_val = fibonacci(n)
        n += 1

    return fib_val, n - 1


def fibonacci(n: int) -> int:
    """Efficient Fibonacci using algebraic method (fast-doubling style)."""
    if n == 0:
        return 0

    step = Zrt5(1, 1)
    fib = Zrt5(1, 1)
    n -= 1

    while n > 0:
        if n & 1:
            fib *= step
            fib >>= 1
        step *= step
        step >>= 1
        n >>= 1

    return fib.b


def sci_notation(num: int, sig_digits: int = 4) -> str:
    """Format big integer into scientific notation safely."""
    digits = int(num.bit_length() * math.log10(2)) + 1
    scale = 10 ** (digits - sig_digits)
    mantissa = num // scale
    s = str(mantissa)
    return f"{s[0]}.{s[1:]}e+{digits-1}"


# Example: run Fibonacci until 3 seconds have passed
val, terms = fibonacci_timed(3)
print(f"Largest Fibonacci number found (n={terms}) ≈ {sci_notation(val)}")
'''
# --------------------------------------
# matmul_fastexp
'''
import time, math

class M2x2:
    """2x2 matrix for Fibonacci fast exponentiation."""
    def __init__(self, e00, e01, e10, e11):
        self.e00, self.e01, self.e10, self.e11 = int(e00), int(e01), int(e10), int(e11)

    def __mul__(self, other):
        return M2x2(
            self.e00*other.e00 + self.e01*other.e10,
            self.e00*other.e01 + self.e01*other.e11,
            self.e10*other.e00 + self.e11*other.e10,
            self.e10*other.e01 + self.e11*other.e11
        )

    def __imul__(self, other):
        result = self * other
        self.e00, self.e01, self.e10, self.e11 = result.e00, result.e01, result.e10, result.e11
        return self


def fibonacci(n: int) -> int:
    """Compute F(n) using matrix exponentiation."""
    step = M2x2(0, 1, 1, 1)
    fib = M2x2(step.e00, step.e01, step.e10, step.e11)

    while n > 0:
        if n & 1:
            fib *= step
        step *= step
        n >>= 1

    return fib.e00


def fibonacci_timed(seconds: float):
    """Run Fibonacci generation for a given time limit."""
    start = time.time()
    n, val = 1, 1
    while time.time() - start < seconds:
        val = fibonacci(n)
        n += 1
    return val, n - 1


def sci_notation(num: int, sig_digits: int = 4) -> str:
    """Format big int safely into scientific notation."""
    digits = int(num.bit_length() * math.log10(2)) + 1
    scale = 10 ** (digits - sig_digits)
    mantissa = num // scale
    s = str(mantissa)
    return f"{s[0]}.{s[1:]}e+{digits-1}"


# Example: run Fibonacci until 3 seconds have passed
val, terms = fibonacci_timed(3)
print(f"Largest Fibonacci number found (n={terms}) ≈ {sci_notation(val)}")
'''
# --------------------------------------
# matmul_strussen
'''
import time, math

class M2x2:
    """2x2 matrix for Fibonacci fast exponentiation (Strassen multiplication)."""
    def __init__(self, e00, e01, e10, e11):
        self.e00, self.e01, self.e10, self.e11 = int(e00), int(e01), int(e10), int(e11)

    def __mul__(self, other):
        # Strassen multiplication
        m0 = (self.e00 + self.e11) * (other.e00 + other.e11)
        m1 = (self.e10 + self.e11) * other.e00
        m2 = self.e00 * (other.e01 - other.e11)
        m3 = self.e11 * (other.e10 - other.e00)
        m4 = (self.e00 + self.e01) * other.e11
        m5 = (self.e10 - self.e00) * (other.e00 + other.e01)
        m6 = (self.e01 - self.e11) * (other.e10 + other.e11)
        return M2x2(
            m0 + m3 - m4 + m6, m2 + m4,
            m1 + m3, m0 - m1 + m2 + m5
        )

    def __imul__(self, other):
        result = self * other
        self.e00, self.e01, self.e10, self.e11 = result.e00, result.e01, result.e10, result.e11
        return self


def fibonacci(n: int) -> int:
    """Compute F(n) using fast matrix exponentiation."""
    step = M2x2(0, 1, 1, 1)
    fib = M2x2(step.e00, step.e01, step.e10, step.e11)

    while n > 0:
        if n & 1:
            fib *= step
        step *= step
        n >>= 1

    return fib.e00


def fibonacci_timed(seconds: float):
    """Run Fibonacci until time expires, return last result and index."""
    start = time.time()
    n, val = 1, 1
    while time.time() - start < seconds:
        val = fibonacci(n)
        n += 1
    return val, n - 1


def sci_notation(num: int, sig_digits: int = 4) -> str:
    """Format a huge integer into scientific notation safely."""
    digits = int(num.bit_length() * math.log10(2)) + 1
    scale = 10 ** (digits - sig_digits)
    mantissa = num // scale
    s = str(mantissa)
    return f"{s[0]}.{s[1:]}e+{digits-1}"


# Example: run Fibonacci for 3 seconds
val, terms = fibonacci_timed(3)
print(f"Largest Fibonacci number (n={terms}) ≈ {sci_notation(val)}")
'''
# --------------------------------------
'''
# matmul_fft
import time, math

class M2x2:
    """2x2 matrix for Fibonacci fast exponentiation."""
    def __init__(self, e00, e01, e10, e11):
        self.e00, self.e01, self.e10, self.e11 = int(e00), int(e01), int(e10), int(e11)

    def __mul__(self, other):
        return M2x2(
            self.e00*other.e00 + self.e01*other.e10,
            self.e00*other.e01 + self.e01*other.e11,
            self.e10*other.e00 + self.e11*other.e10,
            self.e10*other.e01 + self.e11*other.e11
        )

    def __imul__(self, other):
        result = self * other
        self.e00, self.e01, self.e10, self.e11 = result.e00, result.e01, result.e10, result.e11
        return self


def fibonacci(n: int) -> int:
    """Compute Fibonacci using matrix exponentiation."""
    step = M2x2(0, 1, 1, 1)
    fib = M2x2(step.e00, step.e01, step.e10, step.e11)

    while n > 0:
        if n & 1:
            fib *= step
        step *= step
        n >>= 1

    return fib.e00


def fibonacci_timed(seconds: float = 3.0):
    """Run Fibonacci until a time limit (default 3 seconds)."""
    start = time.time()
    n, val = 1, 1
    while time.time() - start < seconds:
        val = fibonacci(n)
        n += 1
    return val, n - 1


def sci_notation(num: int, sig_digits: int = 4) -> str:
    """Format huge int safely in scientific notation."""
    digits = int(num.bit_length() * math.log10(2)) + 1
    scale = 10 ** (digits - sig_digits)
    mantissa = num // scale
    s = str(mantissa)
    return f"{s[0]}.{s[1:]}e+{digits-1}"


# Example: run Fibonacci for 3 seconds
val, terms = fibonacci_timed(3)
print(f"Largest Fibonacci number (n={terms}) ≈ {sci_notation(val)}")
''''''