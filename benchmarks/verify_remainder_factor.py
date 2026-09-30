"""Exact rational verification of the degree-four residual sampling factors.

The quadratic interpolation residual vanishes at vertices and edge midpoints.
For a polynomial of degree <= 4, its other degree-four lattice values determine
it. Bound the sum of absolute values of those Lagrange cardinal polynomials by
2**dimension, using Bernstein convex hulls on an exact partition. This also
bounds matrix residuals in any norm. No dense sampling or floating point is
used in the proof. For nonpolynomial Hamiltonians the allowance stays sampled.
"""

from fractions import Fraction as F
from itertools import product
from math import comb, factorial, prod
import json


def multiindices(vertices):
    return [a for a in product(range(5), repeat=vertices) if sum(a) == 4]


def cardinal_controls(alpha, indices):
    vertices = len(alpha)
    monomials = {(0,) * vertices: F(1)}
    # L_alpha(lambda) = product_i product_(j<alpha_i) (4 lambda_i-j)/(j+1).
    for axis, degree in enumerate(alpha):
        for j in range(degree):
            expanded = {}
            for powers, coefficient in monomials.items():
                raised = list(powers)
                raised[axis] += 1
                raised = tuple(raised)
                expanded[raised] = expanded.get(raised, 0) + coefficient * F(4, j + 1)
                expanded[powers] = expanded.get(powers, 0) - coefficient * F(j, j + 1)
            monomials = expanded
    controls = []
    for gamma in indices:
        value = F(0)
        for beta, coefficient in monomials.items():
            if all(b <= g for b, g in zip(beta, gamma)):
                falling = prod(
                    factorial(g) // factorial(g - b) for b, g in zip(beta, gamma)
                )
                value += coefficient * F(
                    falling * factorial(4 - sum(beta)), factorial(4)
                )
        controls.append(value)
    return controls


def verify(dimension):
    v = dimension + 1
    indices = multiindices(v)
    lookup = {a: i for i, a in enumerate(indices)}
    probes = [a for a in indices if any(x % 2 for x in a)]
    columns = [cardinal_controls(a, indices) for a in probes]
    controls = list(map(tuple, zip(*columns)))
    points = [tuple(F(i == j) for i in range(v)) for j in range(v)]
    factors = {}
    for replaced in range(v):
        for other in range(v):
            if replaced == other:
                continue
            rows = []
            for alpha in indices:
                row = []
                degree = alpha[replaced]
                for k in range(degree + 1):
                    beta = list(alpha)
                    beta[replaced] -= k
                    beta[other] += k
                    row.append((lookup[tuple(beta)], F(comb(degree, k), 2**degree)))
                rows.append(row)
            factors[replaced, other] = rows

    limit = 2**dimension
    stack = [(controls, points, 0)]
    maximum = F(0)
    leaves = max_depth = 0
    while stack:
        controls, points, depth = stack.pop()
        bound = max(sum(abs(x) for x in row) for row in controls)
        if bound <= limit:
            leaves += 1
            max_depth = max(max_depth, depth)
            maximum = max(maximum, bound)
            continue
        assert depth < 16, (dimension, depth, bound)
        _, left, right = max(
            (sum((x - y) ** 2 for x, y in zip(points[i], points[j])), i, j)
            for i in range(v)
            for j in range(i + 1, v)
        )
        midpoint = tuple((a + b) / 2 for a, b in zip(points[left], points[right]))
        for replaced, other in ((left, right), (right, left)):
            child = [
                tuple(
                    sum(weight * controls[index][p] for index, weight in row)
                    for p in range(len(probes))
                )
                for row in factors[replaced, other]
            ]
            child_points = points.copy()
            child_points[replaced] = midpoint
            stack.append((child, child_points, depth + 1))
    return dict(
        dimension=dimension,
        probes=len(probes),
        factor=limit,
        certified_upper_bound=str(maximum),
        leaves=leaves,
        depth=max_depth,
    )


if __name__ == "__main__":
    print(json.dumps([verify(d) for d in (1, 2, 3)], indent=2))
