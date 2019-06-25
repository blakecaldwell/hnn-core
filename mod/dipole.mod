: dipole.mod - mod file for range variable dipole
:
: v 2.0.0
: rev 2019-06-24 (BC)
: last rev: (BC: made thread-safe by removing POINTERs)

NEURON {
    THREADSAFE
    SUFFIX dipole
    RANGE ri, ia, Q, ztan, pv

    : for density. sums into Dipole at section position 1
    RANGE Qsum
}

UNITS {
    (nA)   = (nanoamp)
    (mV)   = (millivolt)
    (Mohm) = (megaohm)
    (um)   = (micrometer)
    (Am)   = (amp meter)
    (fAm)  = (femto amp meter)
}

ASSIGNED {
    ia (nA)
    ri (Mohm)
    pv (mV)
    v (mV)
    ztan (um)
    Q (fAm)

    : human dipole order of 10 nAm
    Qsum (fAm)
}

: solve for v's first then use them
AFTER SOLVE {
    ia = (pv - v) / ri
    Q = ia * ztan
    Qsum = Qsum + Q
}

AFTER INITIAL {
    ia = (pv - v) / ri
    Q = ia * ztan
    Qsum = Qsum + Q
}
