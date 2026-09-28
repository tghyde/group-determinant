#!/usr/bin/env sage
# Generates data.js for the group determinant explorer.
#
# For each group we record: element names (LaTeX), the multiplication table,
# a presentation, and the factorization of the group determinant
#     Theta(G) = det( x_{g h} )_{g,h in G}
# into irreducible factors Phi_rho = det( sum_g x_g rho(g) ), one for each
# irreducible representation rho, each appearing with multiplicity deg rho.
# The factorization is verified numerically by random substitution.

from sage.all import *
import json, itertools, sys, math

# ---------------------------------------------------------------------------
# Group models.  Each returns (names, mult, info) where
#   names : list of LaTeX strings for the elements (index = position)
#   mult  : function (i, j) -> index of  elems[i] * elems[j]
#   info  : dict with presentation etc.
# ---------------------------------------------------------------------------

def perm_compose_rtl(p, q):
    """(p q)(x) = p(q(x)): apply q first, then p.  Perms are tuples of images of 1..n (0-based)."""
    return tuple(p[q[x]] for x in range(len(p)))

def cycle_notation(p):
    n = len(p); seen = [False]*n; cycles = []
    for i in range(n):
        if seen[i] or p[i] == i:
            seen[i] = True; continue
        c = []; j = i
        while not seen[j]:
            seen[j] = True; c.append(j+1); j = p[j]
        cycles.append(c)
    if not cycles: return "e"
    return "".join("(" + "".join(str(a) for a in c) + ")" for c in cycles)

def cycle_type(p):
    n = len(p); seen = [False]*n; ct = []
    for i in range(n):
        if seen[i]: continue
        L = 0; j = i
        while not seen[j]:
            seen[j] = True; L += 1; j = p[j]
        ct.append(L)
    return tuple(sorted(ct, reverse=True))

def perm_sign(p):
    return (-1)**(len(p) - len(cycle_type(p)))

def symmetric_model(n, even_only=False):
    perms = [tuple(p) for p in itertools.permutations(range(n))]
    if even_only:
        perms = [p for p in perms if perm_sign(p) == 1]
    # order: identity, then by cycle type (fewer moved points first), then lexicographically
    def key(p):
        ct = cycle_type(p)
        moved = sum(c for c in ct if c > 1)
        return (moved, tuple(-c for c in ct), cycle_notation(p))
    perms.sort(key=key)
    idx = {p: i for i, p in enumerate(perms)}
    names = [cycle_notation(p) for p in perms]
    def mult(i, j):
        return idx[perm_compose_rtl(perms[i], perms[j])]
    return names, mult

def cyclic_model(n):
    names = ["e"] + ["g"] + ["g^{%d}" % k for k in range(2, n)]
    return names, (lambda i, j: (i + j) % n)

def power_name(g, k):
    if k == 0: return ""
    if k == 1: return g
    return "%s^{%d}" % (g, k)

def abelian_model(orders, letters):
    # elements are tuples (k_1, ..., k_r); ordered with the LAST coordinate varying fastest
    ranges = [range(m) for m in orders]
    elems = list(itertools.product(*ranges))
    idx = {e: i for i, e in enumerate(elems)}
    def name(e):
        s = "".join(power_name(L, k) for L, k in zip(letters, e))
        return s if s else "e"
    names = [name(e) for e in elems]
    def mult(i, j):
        return idx[tuple((a + b) % m for a, b, m in zip(elems[i], elems[j], orders))]
    return names, mult

def dihedral_model(n):
    # elements r^i s^j, i mod n, j in {0,1};  s r = r^{-1} s
    elems = [(i, 0) for i in range(n)] + [(i, 1) for i in range(n)]
    idx = {e: k for k, e in enumerate(elems)}
    def name(e):
        i, j = e
        s = power_name("r", i) + ("s" if j else "")
        return s if s else "e"
    names = [name(e) for e in elems]
    def mult(a, b):
        i, j = elems[a]; k, l = elems[b]
        # r^i s^j r^k s^l = r^{i + (-1)^j k} s^{j+l}
        return idx[((i + (k if j == 0 else -k)) % n, (j + l) % 2)]
    return names, mult

def quaternion_model():
    # units: 1, i, j, k with signs
    names_base = ["1", "i", "j", "k"]
    elems = [(1, u) for u in range(4)] + [(-1, u) for u in range(4)]
    idx = {e: k for k, e in enumerate(elems)}
    names = [("" if s == 1 else "-") + names_base[u] for s, u in elems]
    # multiplication table of units: (sign, unit)
    tab = {}
    def setm(a, b, s, c): tab[(a, b)] = (s, c)
    for u in range(4): setm(0, u, 1, u); setm(u, 0, 1, u)
    setm(1, 1, -1, 0); setm(2, 2, -1, 0); setm(3, 3, -1, 0)
    setm(1, 2, 1, 3); setm(2, 1, -1, 3)
    setm(2, 3, 1, 1); setm(3, 2, -1, 1)
    setm(3, 1, 1, 2); setm(1, 3, -1, 2)
    def mult(a, b):
        s1, u1 = elems[a]; s2, u2 = elems[b]
        s, u = tab[(u1, u2)]
        return idx[(s1 * s2 * s, u)]
    return names, mult

def dicyclic_model(n):
    # Dic_n of order 4n: a^i x^j, a^{2n} = e, x^2 = a^n, x a x^{-1} = a^{-1}
    m = 2 * n
    elems = [(i, 0) for i in range(m)] + [(i, 1) for i in range(m)]
    idx = {e: k for k, e in enumerate(elems)}
    def name(e):
        i, j = e
        s = power_name("a", i) + ("x" if j else "")
        return s if s else "e"
    names = [name(e) for e in elems]
    def mult(p, q):
        i, j = elems[p]; k, l = elems[q]
        if j == 0:
            return idx[((i + k) % m, l)]
        # a^i x a^k x^l = a^{i-k} x^{1+l}
        if l == 0:
            return idx[((i - k) % m, 1)]
        return idx[((i - k + n) % m, 0)]
    return names, mult

def metacyclic_model(p, q, t, letters=("a", "b")):
    # C_p x| C_q : a^p = b^q = e, b a b^{-1} = a^t  (t has order q mod p)
    # elements a^i b^j ;  b^j a^k = a^{t^j k} b^j
    A, B = letters
    elems = [(i, j) for j in range(q) for i in range(p)]  # b-power outer, a-power inner
    idx = {e: k for k, e in enumerate(elems)}
    def name(e):
        i, j = e
        s = power_name(A, i) + power_name(B, j)
        return s if s else "e"
    names = [name(e) for e in elems]
    def mult(x, y):
        i, j = elems[x]; k, l = elems[y]
        return idx[((i + pow(t, j, p) * k) % p, (j + l) % q)]
    return names, mult

def heisenberg_model():
    # upper unitriangular 3x3 over F_3: a = I+E12, b = I+E23, c = I+E13 = [a,b] (central)
    # elements a^i b^j c^k.  Multiply via matrices.
    def mat(i, j, k):
        return ((1, i % 3, k % 3), (0, 1, j % 3), (0, 0, 1))
    def mmul(M, N):
        return tuple(tuple(sum(M[r][s] * N[s][c] for s in range(3)) % 3 for c in range(3)) for r in range(3))
    def matpow(M, e):
        R = mat(0, 0, 0)
        for _ in range(e): R = mmul(R, M)
        return R
    a, b, c = mat(1, 0, 0), mat(0, 1, 0), mat(0, 0, 1)
    # check c = a b a^{-1} b^{-1}
    ainv, binv = matpow(a, 2), matpow(b, 2)
    comm = mmul(mmul(a, b), mmul(ainv, binv))
    assert comm == c, comm
    triples = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    elems = [mmul(mmul(matpow(a, i), matpow(b, j)), matpow(c, k)) for (i, j, k) in triples]
    assert len(set(elems)) == 27
    idx = {e: n for n, e in enumerate(elems)}
    def name(t):
        i, j, k = t
        s = power_name("a", i) + power_name("b", j) + power_name("c", k)
        return s if s else "e"
    names = [name(t) for t in triples]
    def mult(x, y):
        return idx[mmul(elems[x], elems[y])]
    return names, mult

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

GROUPS = []
def add(gid, label, latex, category, model, presentation, note, permgroup=False):
    names, mult = model
    GROUPS.append(dict(id=gid, label=label, latex=latex, category=category,
                       names=names, mult=mult, presentation=presentation,
                       note=note, permgroup=permgroup))

for n in range(2, 9):
    add("C%d" % n, "C_%d (order %d)" % (n, n), "C_{%d}" % n, "Cyclic groups", cyclic_model(n),
        r"\langle\, g \mid g^{%d} = e \,\rangle" % n,
        "Elements are the powers of the generator $g$.")

add("C2xC2", "C_2 × C_2 (order 4)", r"C_2 \times C_2", "Abelian groups", abelian_model([2, 2], ["a", "b"]),
    r"\langle\, a, b \mid a^{2} = b^{2} = e,\ ab = ba \,\rangle",
    "The Klein four-group. Elements are written in the form $a^i b^j$.")
add("C2xC4", "C_2 × C_4 (order 8)", r"C_2 \times C_4", "Abelian groups", abelian_model([2, 4], ["a", "b"]),
    r"\langle\, a, b \mid a^{2} = b^{4} = e,\ ab = ba \,\rangle",
    "Elements are written in the form $a^i b^j$.")
add("C2xC2xC2", "C_2 × C_2 × C_2 (order 8)", r"C_2 \times C_2 \times C_2", "Abelian groups", abelian_model([2, 2, 2], ["a", "b", "c"]),
    r"\langle\, a, b, c \mid a^{2} = b^{2} = c^{2} = e,\ ab = ba,\ ac = ca,\ bc = cb \,\rangle",
    "Elements are written in the form $a^i b^j c^k$.")
add("C3xC3", "C_3 × C_3 (order 9)", r"C_3 \times C_3", "Abelian groups", abelian_model([3, 3], ["a", "b"]),
    r"\langle\, a, b \mid a^{3} = b^{3} = e,\ ab = ba \,\rangle",
    "Elements are written in the form $a^i b^j$.")

POLY = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon'}
for n in range(3, 8):
    add("D%d" % n, "D_%d (order %d)" % (n, 2 * n), "D_{%d}" % n, "Dihedral groups", dihedral_model(n),
        r"\langle\, r, s \mid r^{%d} = s^{2} = e,\ srs = r^{-1} \,\rangle" % n,
        "Symmetries of a regular %s: $r$ is a rotation by $2\\pi/%d$, $s$ a reflection. Elements are written in the form $r^i s^j$." % (POLY[n], n))

add("S3", "S_3 (order 6)", "S_3", "Symmetric groups", symmetric_model(3),
    r"\text{permutations of } \{1,2,3\}",
    "Elements are written in cycle notation.", permgroup=True)
add("S4", "S_4 (order 24)", "S_4", "Symmetric groups", symmetric_model(4),
    r"\text{permutations of } \{1,2,3,4\}",
    "Elements are written in cycle notation.", permgroup=True)
add("A4", "A_4 (order 12)", "A_4", "Alternating groups", symmetric_model(4, even_only=True),
    r"\text{even permutations of } \{1,2,3,4\}",
    "Elements are written in cycle notation.", permgroup=True)

add("Q8", "Q_8 (order 8)", "Q_8", "Other groups", quaternion_model(),
    r"\langle\, i, j, k \mid i^{2} = j^{2} = k^{2} = ijk = -1 \,\rangle",
    "The quaternion group $\\{\\pm 1, \\pm i, \\pm j, \\pm k\\}$ with $ij = k$.")
add("Dic3", "Dic_3 (order 12)", r"\mathrm{Dic}_3", "Other groups", dicyclic_model(3),
    r"\langle\, a, x \mid a^{6} = e,\ x^{2} = a^{3},\ xax^{-1} = a^{-1} \,\rangle",
    "The dicyclic group of order 12 (also $C_3 \\rtimes C_4$). Elements are written in the form $a^i x^j$.")
add("F20", "C_5 ⋊ C_4 (order 20)", r"C_5 \rtimes C_4", "Other groups", metacyclic_model(5, 4, 2),
    r"\langle\, a, b \mid a^{5} = b^{4} = e,\ bab^{-1} = a^{2} \,\rangle",
    "The Frobenius group of order 20, the affine maps $t \\mapsto ut + v$ of $\\mathbb{Z}/5$. Elements are written in the form $a^i b^j$.")
add("F21", "C_7 ⋊ C_3 (order 21)", r"C_7 \rtimes C_3", "Other groups", metacyclic_model(7, 3, 2),
    r"\langle\, a, b \mid a^{7} = b^{3} = e,\ bab^{-1} = a^{2} \,\rangle",
    "The Frobenius group of order 21, the smallest nonabelian group of odd order. Elements are written in the form $a^i b^j$.")
add("Heis3", "Heis(F_3) (order 27)", r"\mathrm{Heis}(\mathbb{F}_3)", "Other groups", heisenberg_model(),
    r"\langle\, a, b, c \mid a^{3} = b^{3} = c^{3} = e,\ c = aba^{-1}b^{-1},\ ac = ca,\ bc = cb \,\rangle",
    "The Heisenberg group of $3\\times 3$ upper unitriangular matrices over $\\mathbb{F}_3$; $c$ is central. Elements are written in the form $a^i b^j c^k$.")

# ---------------------------------------------------------------------------
# Nice expressions for elements of Z[zeta_m]
# ---------------------------------------------------------------------------

_expr_cache = {}
def expression_table(m):
    """Map field element -> shortest expression sum c_k zeta_m^k (as dict k->c)."""
    if m in _expr_cache: return _expr_cache[m]
    K = CyclotomicField(m); z = K.gen()
    zp = [z**k for k in range(m)]
    best = {}
    def rank(expr):
        terms = len(expr)
        l1 = sum(abs(c) for c in expr.values())
        nonconst = sum(1 for k in expr if k != 0)
        allneg = terms > 1 and all(c < 0 for c in expr.values())
        neg = 0 if allneg else sum(1 for c in expr.values() if c < 0)
        sumexp = sum(expr.keys())
        return (terms, nonconst, neg, l1, sumexp)
    def consider(expr):
        val = sum(c * zp[k] for k, c in expr.items())
        r = rank(expr)
        if val not in best or r < best[val][0]:
            best[val] = (r, expr)
    consider({})
    bounds = {1: 30, 2: 8, 3: 4, 4: 3}
    for s in range(1, 4 if m > 8 else 5):
        B = bounds[s]
        cs = [c for c in range(-B, B + 1) if c != 0]
        for S in itertools.combinations(range(m), s):
            for C in itertools.product(cs, repeat=s):
                consider(dict(zip(S, C)))
    _expr_cache[m] = (K, best)
    return _expr_cache[m]

def in_cyclotomic(a, K_big, m):
    """Return image of a in CyclotomicField(m) if it lies there, else None."""
    Km = CyclotomicField(m)
    try:
        return Km(a)
    except (TypeError, ValueError):
        return None

def conductor_of(coeffs, K_big, mbig):
    for d in divisors(mbig):
        ok = True
        for a in coeffs:
            if in_cyclotomic(a, K_big, d) is None:
                ok = False; break
        if ok: return d
    return mbig

def zeta_latex(m, k, c):
    """LaTeX of c * zeta_m^k (c a nonzero integer), without a leading sign (abs value)."""
    c = abs(c)
    if k == 0:
        return str(c)
    base = r"\zeta_{%d}" % m if k == 1 else r"\zeta_{%d}^{%d}" % (m, k)
    return base if c == 1 else "%d%s" % (c, base)

def coeff_latex(expr, m):
    """Return (sign, body) where sign is +1/-1 and body is LaTeX of the absolute coefficient,
    body == '' means 1.  Multi-term coefficients are parenthesised."""
    items = sorted(expr.items())
    if len(items) == 0:
        return (1, "0")
    if len(items) == 1:
        k, c = items[0]
        sgn = 1 if c > 0 else -1
        body = zeta_latex(m, k, c)
        if body == "1": body = ""
        return (sgn, body)
    # multi-term: if all negative, factor out -1
    if all(c < 0 for k, c in items):
        sgn = -1; items = [(k, -c) for k, c in items]
    else:
        sgn = 1
    parts = []
    for n, (k, c) in enumerate(items):
        t = zeta_latex(m, k, c)
        if n == 0:
            parts.append(("-" if c < 0 else "") + t)
        else:
            parts.append((" - " if c < 0 else " + ") + t)
    return (sgn, "(" + "".join(parts) + ")")

def monomial_latex(exps, names):
    parts = []
    for i, e in enumerate(exps):
        if e == 0: continue
        v = "x_{%s}" % names[i]
        parts.append(v if e == 1 else v + "^{%d}" % e)
    return "".join(parts) if parts else "1"

def poly_terms_latex(P, names, m, K_big):
    """List of LaTeX term strings (with signs) for polynomial P over K_big, coefficients in Q(zeta_m)."""
    Km, table = expression_table(m)
    mons = sorted(P.dict().items(), key=lambda kv: tuple(kv[0]), reverse=True)
    terms = []
    for exps, a in mons:
        am = Km(a)
        if am not in table:
            # fall back to power basis expression
            vec = am.list()
            expr = {k: ZZ(c) for k, c in enumerate(vec) if c != 0}
            # (all coefficients are algebraic integers, so integer entries)
        else:
            expr = table[am][1]
        sgn, body = coeff_latex(expr, m)
        mon = monomial_latex(exps, names)
        if body == "" :
            core = mon
        elif mon == "1":
            core = body
        else:
            core = body + " " + mon
        terms.append(dict(sign=sgn, tex=core))
    return terms

def linear_form_latex(row_entries, names, m):
    """LaTeX for sum_g c_g x_g (a matrix entry)."""
    Km, table = expression_table(m)
    parts = []
    for i, a in enumerate(row_entries):
        if a == 0: continue
        am = Km(a)
        expr = table[am][1] if am in table else {k: ZZ(c) for k, c in enumerate(am.list()) if c != 0}
        sgn, body = coeff_latex(expr, m)
        core = ("x_{%s}" % names[i]) if body == "" else body + " x_{%s}" % names[i]
        parts.append((sgn, core))
    if not parts: return "0"
    s = ""
    for n, (sgn, core) in enumerate(parts):
        if n == 0: s += ("-" if sgn < 0 else "") + core
        else: s += (" - " if sgn < 0 else " + ") + core
    return s

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def process(G):
    names, mult = G["names"], G["mult"]
    N = len(names)
    table = [[mult(i, j) for j in range(N)] for i in range(N)]
    # identity index
    e = [i for i in range(N) if all(table[i][j] == j for j in range(N))]
    assert len(e) == 1, G["id"]
    e = e[0]
    # right-regular permutation representation R_g : h -> h g   (a homomorphism for Sage/GAP's left-to-right products)
    perms = [libgap.PermList([table[h][g] + 1 for h in range(N)]) for g in range(N)]
    gapG = libgap.Group(perms)
    assert int(gapG.Size()) == N
    reps = gapG.IrreducibleRepresentations()
    exponent = int(gapG.Exponent())
    Kbig = CyclotomicField(exponent)
    R = PolynomialRing(Kbig, N, "x")
    xs = R.gens()
    factors = []
    for rho in reps:
        mats = [matrix(Kbig, rho.Image(perms[g]).sage()) for g in range(N)]
        d = mats[0].nrows()
        S = sum(xs[g] * mats[g].change_ring(R) for g in range(N))
        Phi = S.det()
        coeffs = list(Phi.coefficients())
        entries = [S[i, j] for i in range(d) for j in range(d)]
        entry_vals = [c for E in entries for c in E.coefficients()]
        # m_nat: order of the group of roots of unity generated by the matrix entries
        m_nat = 1
        for c in entry_vals:
            o = c.multiplicative_order()
            assert o != infinity, ("non-root-of-unity entry", G["id"], c)
            m_nat = lcm(m_nat, int(o))
        # m for the expanded polynomial: conductor of its coefficients, but if that is odd
        # while the representation involves roots of unity of twice that order (e.g. an
        # order-6 character of C_6), use the even index so ζ_6 appears rather than -ζ_3^2.
        m = conductor_of(coeffs, Kbig, exponent)
        if m % 2 == 1 and m_nat == 2 * m:
            m = m_nat
        # same rule for the matrix entries
        m_mat = conductor_of(entry_vals, Kbig, exponent)
        if m_mat % 2 == 1 and m_nat == 2 * m_mat:
            m_mat = m_nat
        chi = [mats[g].trace() for g in range(N)]
        factors.append(dict(dim=d, Phi=Phi, S=S, m=m, m_mat=m_mat, chi=chi, mats=mats))
    # order factors: by dimension, then by the character values read as complex numbers
    # (argument in [0, 2pi) first, so trivial < zeta < zeta^2 < ... for cyclic groups)
    def angle_key(c):
        z = CC(c.complex_embedding()) if c != 0 else CC(0)
        if abs(z) < 1e-9: return (0.0, 0.0)
        a = float(z.argument())
        if a < -1e-9: a += 2 * math.pi
        if abs(a) < 1e-9 or abs(a - 2 * math.pi) < 1e-9: a = 0.0
        return (round(a, 6), -round(float(abs(z)), 6))
    def fkey(f):
        return (f["dim"], [angle_key(c) for c in f["chi"]])
    factors.sort(key=fkey)
    # harmonise zeta indices within the group: if zeta_m and zeta_{m'} both occur with m | m',
    # write everything in terms of zeta_{m'}  (so C_8 uses zeta_8 throughout, not zeta_4 and zeta_8)
    def harmonise(key):
        ms = set(f[key] for f in factors if f[key] > 2)
        for f in factors:
            if f[key] > 2:
                bigger = [mm for mm in ms if mm % f[key] == 0]
                f[key] = max(bigger)
    harmonise("m"); harmonise("m_mat")
    # sign epsilon
    n_inv = sum(1 for g in range(N) if table[g][g] == e)
    eps = (-1)**((N - n_inv) // 2)
    # numeric verification
    for trial in range(2):
        v = [ZZ.random_element(-5, 6) for _ in range(N)]
        M = matrix(ZZ, N, N, lambda i, j: v[table[i][j]])
        lhs = M.det()
        rhs = Kbig(eps)
        for f in factors:
            rhs *= f["Phi"](*v)**f["dim"]
        assert lhs == rhs, ("verification failed", G["id"], lhs, rhs)
        # also check the transpose convention (left-to-right) gives the same determinant
        assert M.transpose().det() == lhs
    # build output
    out_factors = []
    for n, f in enumerate(factors):
        d, m = f["dim"], f["m"]
        terms = poly_terms_latex(f["Phi"], names, m, Kbig)
        mat = [[linear_form_latex([f["S"][i, j].coefficient(xs[g]) for g in range(N)], names, f["m_mat"])
                for j in range(d)] for i in range(d)]
        out_factors.append(dict(dim=d, zeta=m, zetaMat=f["m_mat"], nterms=len(terms), terms=terms, matrix=mat,
                                character=[str(c) for c in f["chi"]]))
    nz = sorted(set([f["zeta"] for f in out_factors if f["zeta"] > 2] + [f["zetaMat"] for f in out_factors if f["zetaMat"] > 2]))
    return dict(id=G["id"], label=G["label"], latex=G["latex"], category=G["category"],
                order=N, names=names, table=table, identity=e, presentation=G["presentation"],
                note=G["note"], permgroup=G["permgroup"], sign=eps, factors=out_factors, zetas=nz)

results = []
for G in GROUPS:
    t0 = walltime()
    r = process(G)
    results.append(r)
    print("%-10s order %2d  factors %s  sign %+d  zetas %s  (%.1fs)" % (
        r["id"], r["order"],
        ",".join("%d:%d:z%d/%d" % (f["dim"], f["nterms"], f["zeta"], f["zetaMat"]) for f in r["factors"]),
        r["sign"], r["zetas"], walltime(t0)))
    sys.stdout.flush()

out = "// Generated by gen.sage — do not edit by hand.\nwindow.GROUP_DATA = " + json.dumps(results, separators=(",", ":"), default=lambda o: int(o)) + ";\n"
with open("data.js", "w") as fh:
    fh.write(out)
print("wrote data.js (%d bytes)" % len(out))
