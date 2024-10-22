const real TEMP = T[i]*tconv;
const real PRES = P[i]*pconv;
const real SMALL = FLT_MIN;

real SUM, ctmp;

SUM = 0.0f;

C(1)  = ctmp = Y(1) * (real)4.96046521e-1;
SUM  += ctmp;
C(2)  = ctmp = Y(2) * (real)9.92093043e-1;
SUM  += ctmp;
C(3)  = ctmp = Y(3) * (real)6.25023433e-2;
SUM  += ctmp;
C(4)  = ctmp = Y(4) * (real)3.12511716e-2;
SUM  += ctmp;
C(5)  = ctmp = Y(5) * (real)5.87980383e-2;
SUM  += ctmp;
C(6)  = ctmp = Y(6) * (real)5.55082499e-2;
SUM  += ctmp;
C(7)  = ctmp = Y(7) * (real)3.02968146e-2;
SUM  += ctmp;
C(8)  = ctmp = Y(8) * (real)2.93990192e-2;
SUM  += ctmp;
C(9)  = ctmp = Y(9) * (real)6.65112065e-2;
SUM  += ctmp;
C(10) = ctmp = Y(10)* (real)6.23323639e-2;
SUM  += ctmp;
C(11) = ctmp = Y(11)* (real)3.57008335e-2;
SUM  += ctmp;
C(12) = ctmp = Y(12)* (real)2.27221341e-2;
SUM  += ctmp;
C(13) = ctmp = Y(13)* (real)3.33039255e-2;
SUM  += ctmp;
C(14) = ctmp = Y(14)* (real)3.84050525e-2;
SUM  += ctmp;
C(15) = ctmp = Y(15)* (real)3.56453112e-2;
SUM  += ctmp;
C(16) = ctmp = Y(16)* (real)3.32556033e-2;
SUM  += ctmp;
C(17) = ctmp = Y(17)* (real)2.4372606e-2;
SUM  += ctmp;
C(18) = ctmp = Y(18)* (real)2.37882046e-2;
SUM  += ctmp;
C(19) = ctmp = Y(19)* (real)2.26996304e-2;
SUM  += ctmp;
C(20) = ctmp = Y(20)* (real)2.43467162e-2;
SUM  += ctmp;
C(21) = ctmp = Y(21)* (real)2.37635408e-2;
SUM  += ctmp;
C(22) = ctmp = Y(22)* (real)3.56972032e-2;
SUM  += ctmp;

SUM = DIV (PRES, (SUM * (TEMP) * (real)8.314510e7));

#pragma unroll 22
for (unsigned k=1; k<=22; k++) {
C(k) = MAX(C(k), SMALL) * SUM;
}

