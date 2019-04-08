: dipole.mod - mod file for range variable dipole
:
: v 1.9.1m0
: rev 2015-12-15 (SL: minor)
: last rev: (SL: Added back Qtotal, which WAS used in par version)

NEURON {
    THREADSAFE
       SUFFIX dipole
       RANGE ri, ia, Q, ztan
       BBCOREPOINTER pv

       : for density. sums into Dipole at section position 1
       BBCOREPOINTER Qsum
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

    : human dipole order of 10 nAm
    Qsum (fAm)
    Qtotal (fAm)
}

: solve for v's first then use them
AFTER SOLVE {
VERBATIM
    ia = (*(double *)_p_pv - v) / ri;
    Q = ia * ztan;
    *(double *)_p_Qsum = *(double *)_p_Qsum + Q;
    *(double *)_p_Qtotal = *(double *)_p_Qtotal + Q;
ENDVERBATIM
}

AFTER INITIAL {
VERBATIM
    ia = (*(double *)_p_pv - v) / ri;
    Q = ia * ztan;
    *(double *)_p_Qsum = *(double *)_p_Qsum + Q;
    *(double *)_p_Qtotal = *(double *)_p_Qtotal + Q;
ENDVERBATIM
}

VERBATIM
static void bbcore_write(double* x, int* d, int* xx, int *offset, _threadargsproto_){}
static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_){}
ENDVERBATIM
