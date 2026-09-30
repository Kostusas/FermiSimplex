"""Exact rational verification of dimension-general residual sampling factors.

The quadratic interpolation residual vanishes at vertices and edge midpoints.
For a polynomial through the chosen degree, its remaining lattice values determine
it. Bound the sum of absolute values of those Lagrange cardinal polynomials by
min(2**dimension,32), using exact Bernstein subdivision through dimension four
and a universal coefficient bound in all higher dimensions. This also
bounds matrix residuals in any norm. No dense sampling or floating point is
used in the proof. For nonpolynomial Hamiltonians the allowance stays sampled.
"""

from fractions import Fraction as F
import argparse
from math import comb, factorial, prod
import json


def multiindices(vertices, degree=4):
    if vertices == 1:
        return [(degree,)]
    return [
        (first, *tail)
        for first in range(degree + 1)
        for tail in multiindices(vertices - 1, degree - first)
    ]


def interpolation_node(alpha, degree):
    return sum(a != 0 for a in alpha) == 1 or all(
        a == 0 or 2 * a == degree for a in alpha
    )


def cardinal_controls(alpha, indices):
    degree = sum(alpha)
    vertices = len(alpha)
    monomials = {(0,) * vertices: F(1)}
    # L_alpha(lambda) = product_i product_(j<alpha_i) (degree lambda_i-j)/(j+1).
    for axis, axis_degree in enumerate(alpha):
        for j in range(axis_degree):
            expanded = {}
            for powers, coefficient in monomials.items():
                raised = list(powers)
                raised[axis] += 1
                raised = tuple(raised)
                expanded[raised] = expanded.get(raised, 0) + coefficient * F(
                    degree, j + 1
                )
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
                    falling * factorial(degree - sum(beta)), factorial(degree)
                )
        controls.append(value)
    return controls


def verify(dimension, degree=4):
    v = dimension + 1
    indices = multiindices(v, degree)
    lookup = {a: i for i, a in enumerate(indices)}
    probes = [a for a in indices if not interpolation_node(a, degree)]
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
                axis_degree = alpha[replaced]
                for k in range(axis_degree + 1):
                    beta = list(alpha)
                    beta[replaced] -= k
                    beta[other] += k
                    row.append(
                        (lookup[tuple(beta)], F(comb(axis_degree, k), 2**axis_degree))
                    )
                rows.append(row)
            factors[replaced, other] = rows

    limit = min(2**dimension, 32)
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
        degree=degree,
        dimension=dimension,
        probes=len(probes),
        factor=limit,
        certified_upper_bound=str(maximum),
        leaves=leaves,
        depth=max_depth,
    )


def universal_bound(degree):
    # A degree-m Bernstein index gamma uses at most m vertices. Cardinal
    # polynomials with support outside gamma have zero gamma coefficient,
    # since they contain a factor lambda_i for every node support index i.
    # Therefore every row in arbitrary dimension occurs already with m
    # vertices, after adding zero coordinates. This finite maximum proves
    # the bound for every dimension, not just the ones tested explicitly.
    indices = multiindices(degree, degree)
    probes = [a for a in indices if not interpolation_node(a, degree)]
    columns = [cardinal_controls(a, indices) for a in probes]
    return max(sum(abs(x) for x in row) for row in zip(*columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=(3, 4), default=4)
    args = parser.parse_args()
    universal = universal_bound(args.degree)
    assert universal <= 32
    print(
        json.dumps(
            dict(
                degree=args.degree,
                finite_dimension_checks=[verify(d, args.degree) for d in (1, 2, 3, 4)],
                universal_bernstein_bound=str(universal),
                conclusion="min(2**dimension,32) bounds the sampled polynomial residual in every dimension",
            ),
            indent=2,
        )
    )
