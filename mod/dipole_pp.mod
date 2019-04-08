: dipole_pp.mod - creates point process mechanism Dipole
:
: v 1.9.1m0
: rev 2015-12-15 (SL: minor)
: last rev: (SL: added Qtotal back, used for par calc)

NEURON {
    THREADSAFE
    POINT_PROCESS Dipole
    RANGE ri, ia, Q, ztan
    BBCOREPOINTER pv

    : for POINT_PROCESS. Gets additions from dipole
    RANGE Qsum
    BBCOREPOINTER Qtotal
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
    Qsum (fAm)
    Qtotal (fAm)
}

: solve for v's first then use them
AFTER SOLVE {
VERBATIM
    ia = (*(double *)_p_pv - v) / ri;
    Q = ia * ztan;
    Qsum = Qsum + Q;
    *(double *)_p_Qtotal = *(double *)_p_Qtotal + Q;
ENDVERBATIM
}

AFTER INITIAL {
VERBATIM
    ia = (*(double *)_p_pv - v) / ri;
    Q = ia * ztan;
    Qsum = Qsum + Q;
    *(double *)_p_Qtotal = *(double *)_p_Qtotal + Q;
ENDVERBATIM
}

: following needed for POINT_PROCESS only but will work if also in SUFFIX
BEFORE INITIAL {
VERBATIM
    Qsum = 0;
    *(double *)_p_Qtotal = 0;
ENDVERBATIM
}

BEFORE BREAKPOINT {
VERBATIM
    Qsum = 0;
    *(double *)_p_Qtotal = 0;
ENDVERBATIM
}

VERBATIM
static void bbcore_write(double* x, int* d, int* xx, int *offset, _threadargsproto_){}
static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_){}
ENDVERBATIM
