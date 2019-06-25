: dipole_pp.mod - creates point process mechanism Dipole
:
: v 2.0.0
: rev 2019-06-24 (BC)
: last rev: (BC: made thread-safe by removing POINTERs)

NEURON {
    THREADSAFE
    POINT_PROCESS Dipole
    RANGE ri, ia, Q, ztan
    RANGE pv
    RANGE Qtotal

    : for POINT_PROCESS. Gets additions from dipole
    RANGE Qin
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
    Qin (fAm)
    Qtotal (fAm)
}

: solve for v's first then use them
AFTER SOLVE {
    ia = (pv - v) / ri
    Q = ia * ztan
    Qtotal = Qtotal + Q + Qin
}

AFTER INITIAL {
    ia = (pv - v) / ri
    Q = ia * ztan
    Qtotal = Qtotal + Q + Qin
}

: following needed for POINT_PROCESS only but will work if also in SUFFIX
BEFORE INITIAL {
    Qin = 0
    Qtotal = 0
}

BEFORE BREAKPOINT {
    Qin = 0
    Qtotal = 0
}
