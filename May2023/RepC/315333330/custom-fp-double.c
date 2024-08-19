#include <stdio.h> 
#include <math.h>
#pragma GCC target ("custom-frdxhi=40")
#pragma GCC target ("custom-frdxlo=41")
#pragma GCC target ("custom-frdy=42")
#pragma GCC target ("custom-fwrx=43")
#pragma GCC target ("custom-fwry=44")
#pragma GCC target ("custom-fabsd=100")
#pragma GCC target ("custom-faddd=101")
#pragma GCC target ("custom-fatand=102")
#pragma GCC target ("custom-fcosd=103")
#pragma GCC target ("custom-fdivd=104")
#pragma GCC target ("custom-fexpd=105")
#pragma GCC target ("custom-flogd=106")
#pragma GCC target ("custom-fmaxd=107")
#pragma GCC target ("custom-fmind=108")
#pragma GCC target ("custom-fmuld=109")
#pragma GCC target ("custom-fnegd=110")
#pragma GCC target ("custom-fsind=111")
#pragma GCC target ("custom-fsqrtd=112")
#pragma GCC target ("custom-fsubd=113")
#pragma GCC target ("custom-ftand=114")
#pragma GCC target ("custom-fcmpeqd=200")
#pragma GCC target ("custom-fcmpged=201")
#pragma GCC target ("custom-fcmpgtd=202")
#pragma GCC target ("custom-fcmpled=203")
#pragma GCC target ("custom-fcmpltd=204")
#pragma GCC target ("custom-fcmpned=205")
void
custom_fp (double a, double b, double *fp, int *ip)
{
fp[0] = fabs (a);
fp[1] = a + b;
fp[2] = atan (a);
fp[3] = cos (a);
fp[4] = a / b;
fp[5] = exp (a);
fp[6] = log (a);
fp[7] = fmax (a, b);
fp[8] = fmin (a, b);
fp[9] = a * b;
fp[10] = -b;
fp[11] = sin (b);
fp[12] = sqrt (a);
fp[13] = a - b;
fp[14] = tan (a);
ip[0] = (a == fp[0]);
ip[1] = (a >= fp[1]);
ip[2] = (a > fp[2]);
ip[3] = (a <= fp[3]);
ip[4] = (a < fp[4]);
ip[5] = (a != fp[5]);
}
