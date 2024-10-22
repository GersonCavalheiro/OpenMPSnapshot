const real TEMP = T[i]*tconv;
real CTOT = (real)0.0;
register real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
register real CPRLOG, FLOG, FC, SQR;
const real SMALL = FLT_MIN;

#pragma unroll 22
for (unsigned int k=1; k<=22; k++)
{
CTOT += C(k);
}

real CTB_5  = CTOT - C(1) - C(6) + C(10) - C(12) + (real)2.e0*C(16)
+ (real)2.e0*C(14) + (real)2.e0*C(15) ;
real CTB_9  = CTOT - (real)2.7e-1*C(1) + (real)2.65e0*C(6) + C(10) + (real)2.e0*C(16)
+ (real)2.e0*C(14) + (real)2.e0*C(15) ;
real CTB_10 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
+ (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15);
real CTB_11 = CTOT + (real)1.4e0*C(1) + (real)1.44e1*C(6) + C(10) + (real)7.5e-1*C(11)
+ (real)2.6e0*C(12) + (real)2.e0*C(16) + (real)2.e0*C(14)
+ (real)2.e0*C(15) ;
real CTB_12 = CTOT - C(4) - C(6) - (real)2.5e-1*C(11) + (real)5.e-1*C(12)
+ (real)5.e-1*C(16) - C(22) + (real)2.e0*C(14) + (real)2.e0*C(15) ;
real CTB_29 = CTOT + C(1) + (real)5.e0*C(4) + (real)5.e0*C(6) + C(10)
+ (real)5.e-1*C(11) + (real)2.5e0*C(12) + (real)2.e0*C(16)
+ (real)2.e0*C(14) + (real)2.e0*C(15) ;
real CTB_190= CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11)
+ C(12) + (real)2.e0*C(16) ;

RF(5) = RF(5)*CTB_5*C(2)*C(2);
RB(5) = RB(5)*CTB_5*C(1);
RF(9) = RF(9)*CTB_9*C(2)*C(5);
RB(9) = RB(9)*CTB_9*C(6);
RF(10) = RF(10)*CTB_10*C(3)*C(2);
RB(10) = RB(10)*CTB_10*C(5);
RF(11) = RF(11)*CTB_11*C(3)*C(3);
RB(11) = RB(11)*CTB_11*C(4);
RF(12) = RF(12)*CTB_12*C(2)*C(4);
RB(12) = RB(12)*CTB_12*C(7);
RF(29) = RF(29)*CTB_29*C(11)*C(3);
RB(29) = RB(29)*CTB_29*C(12);
RF(46) = RF(46)*CTB_10;
RB(46) = RB(46)*CTB_10*C(11)*C(2);
RF(121) = RF(121)*CTOT*C(14)*C(9);
RB(121) = RB(121)*CTOT*C(20);


PR = RKLOW(13) * DIV(CTB_10, RF(126));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)6.63e-1*EXP(DIV(-TEMP,(real)1.707e3)) + (real)3.37e-1*EXP(DIV(-TEMP,(real)3.2e3))
+ EXP(DIV((real)-4.131e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(126) = RF(126) * PCOR;
RB(126) = RB(126) * PCOR;

PR = RKLOW(14) * DIV(CTB_10, RF(132));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)2.18e-1*EXP(DIV(-TEMP,(real)2.075e2)) + (real)7.82e-1*EXP(DIV(-TEMP,(real)2.663e3))
+ EXP(DIV((real)-6.095e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(132) = RF(132) * PCOR;
RB(132) = RB(132) * PCOR;

PR = RKLOW(15) * DIV(CTB_10, RF(145));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)8.25e-1*EXP(DIV(-TEMP,(real)1.3406e3)) + (real)1.75e-1*EXP(DIV(-TEMP,(real)6.e4))
+ EXP(DIV((real)-1.01398e4,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(145) = RF(145) * PCOR;
RB(145) = RB(145) * PCOR;

PR = RKLOW(16) * DIV(CTB_10, RF(148));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)4.5e-1*EXP(DIV(-TEMP,(real)8.9e3)) + (real)5.5e-1*EXP(DIV(-TEMP,(real)4.35e3))
+ EXP(DIV((real)-7.244e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(148) = RF(148) * PCOR;
RB(148) = RB(148) * PCOR;

PR = RKLOW(17) * DIV(CTB_10, RF(155));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)2.655e-1*EXP(DIV(-TEMP,(real)1.8e2)) + (real)7.345e-1*EXP(DIV(-TEMP,(real)1.035e3))
+ EXP(DIV((real)-5.417e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(155) = RF(155) * PCOR;
RB(155) = RB(155) * PCOR;

PR = RKLOW(18) * DIV(CTB_10, RF(156));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)2.47e-2*EXP(DIV(-TEMP,(real)2.1e2)) + (real)9.753e-1*EXP(DIV(-TEMP,(real)9.84e2))
+ EXP(DIV((real)-4.374e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(156) = RF(156) * PCOR;
RB(156) = RB(156) * PCOR;

PR = RKLOW(19) * DIV(CTB_10, RF(170));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)1.578e-1*EXP(DIV(-TEMP,(real)1.25e2)) + (real)8.422e-1*EXP(DIV(-TEMP,(real)2.219e3))
+ EXP(DIV((real)-6.882e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(170) = RF(170) * PCOR;
RB(170) = RB(170) * PCOR;

PR = RKLOW(20) * DIV(CTB_10, RF(185));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)9.8e-1*EXP(DIV(-TEMP,(real)1.0966e3)) + (real)2.e-2*EXP(DIV(-TEMP,(real)1.0966e3))
+ EXP(DIV((real)-6.8595e3,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(185) = RF(185) * PCOR;
RB(185) = RB(185) * PCOR;

PR = RKLOW(21) * DIV(CTB_190, RF(190));
PCOR = DIV(PR, ((real)1.0 + PR));
PRLOG = LOG10(MAX(PR,SMALL));
FCENT = (real)0.e0*EXP(DIV(-TEMP,(real)1.e3)) + (real)1.e0*EXP(DIV(-TEMP,(real)1.31e3))
+ EXP(DIV((real)-4.8097e4,TEMP));
FCLOG = LOG10(MAX(FCENT,SMALL));
XN    = (real)0.75 - (real)1.27*FCLOG;
CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
FC = EXP10(FLOG);
PCOR = FC * PCOR;
RF(190) = RF(190) * PCOR;
RB(190) = RB(190) * PCOR;

