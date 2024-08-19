

#include "master.c"
#include <math.h>


void 
embedded_fehlberg_7_8(
fp timeinst,
fp h,
fp *initvalu,
fp *finavalu,
fp *params,
fp *com,
fp *error,
double *timecopyin,
double *timecopykernel,
double *timecopyout) 
{


static const fp c_1_11 = 41.0 / 840.0;
static const fp c6 = 34.0 / 105.0;
static const fp c_7_8= 9.0 / 35.0;
static const fp c_9_10 = 9.0 / 280.0;

static const fp a2 = 2.0 / 27.0;
static const fp a3 = 1.0 / 9.0;
static const fp a4 = 1.0 / 6.0;
static const fp a5 = 5.0 / 12.0;
static const fp a6 = 1.0 / 2.0;
static const fp a7 = 5.0 / 6.0;
static const fp a8 = 1.0 / 6.0;
static const fp a9 = 2.0 / 3.0;
static const fp a10 = 1.0 / 3.0;

static const fp b31 = 1.0 / 36.0;
static const fp b32 = 3.0 / 36.0;
static const fp b41 = 1.0 / 24.0;
static const fp b43 = 3.0 / 24.0;
static const fp b51 = 20.0 / 48.0;
static const fp b53 = -75.0 / 48.0;
static const fp b54 = 75.0 / 48.0;
static const fp b61 = 1.0 / 20.0;
static const fp b64 = 5.0 / 20.0;
static const fp b65 = 4.0 / 20.0;
static const fp b71 = -25.0 / 108.0;
static const fp b74 =  125.0 / 108.0;
static const fp b75 = -260.0 / 108.0;
static const fp b76 =  250.0 / 108.0;
static const fp b81 = 31.0/300.0;
static const fp b85 = 61.0/225.0;
static const fp b86 = -2.0/9.0;
static const fp b87 = 13.0/900.0;
static const fp b91 = 2.0;
static const fp b94 = -53.0/6.0;
static const fp b95 = 704.0 / 45.0;
static const fp b96 = -107.0 / 9.0;
static const fp b97 = 67.0 / 90.0;
static const fp b98 = 3.0;
static const fp b10_1 = -91.0 / 108.0;
static const fp b10_4 = 23.0 / 108.0;
static const fp b10_5 = -976.0 / 135.0;
static const fp b10_6 = 311.0 / 54.0;
static const fp b10_7 = -19.0 / 60.0;
static const fp b10_8 = 17.0 / 6.0;
static const fp b10_9 = -1.0 / 12.0;
static const fp b11_1 = 2383.0 / 4100.0;
static const fp b11_4 = -341.0 / 164.0;
static const fp b11_5 = 4496.0 / 1025.0;
static const fp b11_6 = -301.0 / 82.0;
static const fp b11_7 = 2133.0 / 4100.0;
static const fp b11_8 = 45.0 / 82.0;
static const fp b11_9 = 45.0 / 164.0;
static const fp b11_10 = 18.0 / 41.0;
static const fp b12_1 = 3.0 / 205.0;
static const fp b12_6 = - 6.0 / 41.0;
static const fp b12_7 = - 3.0 / 205.0;
static const fp b12_8 = - 3.0 / 41.0;
static const fp b12_9 = 3.0 / 41.0;
static const fp b12_10 = 6.0 / 41.0;
static const fp b13_1 = -1777.0 / 4100.0;
static const fp b13_4 = -341.0 / 164.0;
static const fp b13_5 = 4496.0 / 1025.0;
static const fp b13_6 = -289.0 / 82.0;
static const fp b13_7 = 2193.0 / 4100.0;
static const fp b13_8 = 51.0 / 82.0;
static const fp b13_9 = 33.0 / 164.0;
static const fp b13_10 = 12.0 / 41.0;

static const fp err_factor  = -41.0 / 840.0;

fp h2_7 = a2 * h;

fp timeinst_temp;
fp* initvalu_temp;
fp** finavalu_temp;

int i;


initvalu_temp= (fp *) malloc(EQUATIONS* sizeof(fp));

finavalu_temp= (fp **) malloc(13* sizeof(fp *));
for (i= 0; i<13; i++){
finavalu_temp[i]= (fp *) malloc(EQUATIONS* sizeof(fp));
}



timeinst_temp = timeinst;
for(i=0; i<EQUATIONS; i++){
initvalu_temp[i] = initvalu[i] ;
}

#pragma omp target data map (alloc: initvalu[0:EQUATIONS], finavalu[0:EQUATIONS])
{

#ifdef DEBUG
printf("master 1\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);

memcpy(finavalu_temp[0], finavalu, sizeof(fp)*EQUATIONS);

timeinst_temp = timeinst+h2_7;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h2_7 * (finavalu_temp[0][i]);
}
#ifdef DEBUG
printf("master 2\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);

memcpy(finavalu_temp[1], finavalu, sizeof(fp)*EQUATIONS);

timeinst_temp = timeinst+a3*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b31*finavalu_temp[0][i] + b32*finavalu_temp[1][i]);
}

#ifdef DEBUG
printf("master 3\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);

memcpy(finavalu_temp[2], finavalu, sizeof(fp)*EQUATIONS);

timeinst_temp = timeinst+a4*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b41*finavalu_temp[0][i] + b43*finavalu_temp[2][i]) ;
}

#ifdef DEBUG
printf("master 4\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);

memcpy(finavalu_temp[3], finavalu, sizeof(fp)*EQUATIONS);

timeinst_temp = timeinst+a5*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b51*finavalu_temp[0][i] + b53*finavalu_temp[2][i] + b54*finavalu_temp[3][i]) ;
}

#ifdef DEBUG
printf("master 5\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);

memcpy(finavalu_temp[4], finavalu, sizeof(fp)*EQUATIONS);

timeinst_temp = timeinst+a6*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b61*finavalu_temp[0][i] + b64*finavalu_temp[3][i] + b65*finavalu_temp[4][i]) ;
}

#ifdef DEBUG
printf("master 6\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[5], finavalu, sizeof(fp)*EQUATIONS);


#ifdef DEBUG
printf("master 7\n");
#endif
timeinst_temp = timeinst+a7*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b71*finavalu_temp[0][i] + b74*finavalu_temp[3][i] + b75*finavalu_temp[4][i] + b76*finavalu_temp[5][i]) ;
}

master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[6], finavalu, sizeof(fp)*EQUATIONS);



timeinst_temp = timeinst+a8*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b81*finavalu_temp[0][i] + b85*finavalu_temp[4][i] + b86*finavalu_temp[5][i] + b87*finavalu_temp[6][i]);
}

#ifdef DEBUG
printf("master 8\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[7], finavalu, sizeof(fp)*EQUATIONS);


timeinst_temp = timeinst+a9*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b91*finavalu_temp[0][i] + b94*finavalu_temp[3][i] + b95*finavalu_temp[4][i] + b96*finavalu_temp[5][i] + b97*finavalu_temp[6][i]+ b98*finavalu_temp[7][i]) ;
}

#ifdef DEBUG
printf("master 9\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[8], finavalu, sizeof(fp)*EQUATIONS);


timeinst_temp = timeinst+a10*h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b10_1*finavalu_temp[0][i] + b10_4*finavalu_temp[3][i] + b10_5*finavalu_temp[4][i] + b10_6*finavalu_temp[5][i] + b10_7*finavalu_temp[6][i] + b10_8*finavalu_temp[7][i] + b10_9*finavalu_temp[8] [i]) ;
}

#ifdef DEBUG
printf("master 10\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[9], finavalu, sizeof(fp)*EQUATIONS);


timeinst_temp = timeinst+h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b11_1*finavalu_temp[0][i] + b11_4*finavalu_temp[3][i] + b11_5*finavalu_temp[4][i] + b11_6*finavalu_temp[5][i] + b11_7*finavalu_temp[6][i] + b11_8*finavalu_temp[7][i] + b11_9*finavalu_temp[8][i]+ b11_10 * finavalu_temp[9][i]);
}

#ifdef DEBUG
printf("master 11\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[10], finavalu, sizeof(fp)*EQUATIONS);


timeinst_temp = timeinst;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b12_1*finavalu_temp[0][i] + b12_6*finavalu_temp[5][i] + b12_7*finavalu_temp[6][i] + b12_8*finavalu_temp[7][i] + b12_9*finavalu_temp[8][i] + b12_10 * finavalu_temp[9][i]) ;
}

#ifdef DEBUG
printf("master 12\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);
memcpy(finavalu_temp[11], finavalu, sizeof(fp)*EQUATIONS);


timeinst_temp = timeinst+h;
for(i=0; i<EQUATIONS; i++){
initvalu[i] = initvalu_temp[i] + h * ( b13_1*finavalu_temp[0][i] + b13_4*finavalu_temp[3][i] + b13_5*finavalu_temp[4][i] + b13_6*finavalu_temp[5][i] + b13_7*finavalu_temp[6][i] + b13_8*finavalu_temp[7][i] + b13_9*finavalu_temp[8][i] + b13_10*finavalu_temp[9][i] + finavalu_temp[11][i]) ;
}

#ifdef DEBUG
printf("master 13\n");
#endif
master(  timeinst_temp,
initvalu,
params,
finavalu,
com,
timecopyin,
timecopykernel,
timecopyout);

memcpy(finavalu_temp[12], finavalu, sizeof(fp)*EQUATIONS);

} 


for(i=0; i<EQUATIONS; i++){
finavalu[i]= initvalu_temp[i] +  h * (c_1_11 * (finavalu_temp[0][i] + finavalu_temp[10][i])  + c6 * finavalu_temp[5][i] + c_7_8 * (finavalu_temp[6][i] + finavalu_temp[7][i]) + c_9_10 * (finavalu_temp[8][i] + finavalu_temp[9][i]) );
}


for(i=0; i<EQUATIONS; i++){
error[i] = fabs(err_factor * (finavalu_temp[0][i] + finavalu_temp[10][i] - finavalu_temp[11][i] - finavalu_temp[12][i]));
}


free(initvalu_temp);
free(finavalu_temp);

}

