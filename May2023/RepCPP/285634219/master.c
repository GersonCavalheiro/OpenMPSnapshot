#include <stdio.h>
#include <cmath>
#include "../common.h"
#include "kernel_fin.c"
#include "kernel_ecc.h"
#include "kernel_cam.h"

void master(
fp timeinst,
fp *initvalu,
fp *params,
fp *finavalu,
fp *com,
double *timecopyin,
double *timekernel,
double *timecopyout)
{


int i;

int initvalu_offset_ecc;                                
int initvalu_offset_Dyad;                              
int initvalu_offset_SL;                                
int initvalu_offset_Cyt;                                

auto time0 = std::chrono::steady_clock::now();



#ifdef DEBUG
for (int i = 0; i < EQUATIONS; i++)
printf("initvalu %d %f\n", i, initvalu[i]);
for (int i = 0; i < PARAMETERS; i++)
printf("params %d %f\n", i, params[i]);
printf("\n");
#endif

#pragma omp target update to (initvalu[0:EQUATIONS]) 
#pragma omp target update to (params[0:PARAMETERS]) 

auto time1 = std::chrono::steady_clock::now();


#pragma omp target teams num_teams(2) thread_limit(NUMBER_THREADS)
{
#pragma omp parallel
{
int bx = omp_get_team_num();
int tx = omp_get_thread_num();

int valu_offset;                                  
int params_offset;                                  
int com_offset;                                    

fp CaDyad;                                      
fp CaSL;                                      
fp CaCyt;                                      


if(bx == 0){                                    

if(tx == 0){                                  

valu_offset = 0;                              
kernel_ecc(  timeinst,
initvalu,
finavalu,
valu_offset,
params);

}

}


else if(bx == 1){                                  

if(tx == 0){                                  

valu_offset = 46;
params_offset = 0;
com_offset = 0;
CaDyad = initvalu[35]*1e3;                        
kernel_cam(  timeinst,
initvalu,
finavalu,
valu_offset,
params,
params_offset,
com,
com_offset,
CaDyad);

valu_offset = 61;
params_offset = 5;
com_offset = 1;
CaSL = initvalu[36]*1e3;                          
kernel_cam(  timeinst,
initvalu,
finavalu,
valu_offset,
params,
params_offset,
com,
com_offset,
CaSL);

valu_offset = 76;
params_offset = 10;
com_offset = 2;
CaCyt = initvalu[37]*1e3;                    
kernel_cam(  timeinst,
initvalu,
finavalu,
valu_offset,
params,
params_offset,
com,
com_offset,
CaCyt);
}
}
}
}

auto time2 = std::chrono::steady_clock::now();

#pragma omp target update from (finavalu[0:EQUATIONS]) 
#pragma omp target update from (com[0:3]) 

#ifdef DEBUG
for (int i = 0; i < EQUATIONS; i++)
printf("finavalu %d %f\n", i, finavalu[i]);
for (int i = 0; i < 3; i++)
printf("%f ", com[i]);
printf("\n");

#endif

auto time3 = std::chrono::steady_clock::now();

*timecopyin += std::chrono::duration_cast<std::chrono::nanoseconds>(time1-time0).count();
*timekernel += std::chrono::duration_cast<std::chrono::nanoseconds>(time2-time1).count();
*timecopyout += std::chrono::duration_cast<std::chrono::nanoseconds>(time3-time2).count();


initvalu_offset_ecc = 0;
initvalu_offset_Dyad = 46;
initvalu_offset_SL = 61;
initvalu_offset_Cyt = 76;

kernel_fin(
initvalu,
initvalu_offset_ecc,
initvalu_offset_Dyad,
initvalu_offset_SL,
initvalu_offset_Cyt,
params,
finavalu,
com[0],
com[1],
com[2]);


for(i=0; i<EQUATIONS; i++){
if (std::isnan(finavalu[i])){ 
finavalu[i] = 0.0001;                        
}
else if (std::isinf(finavalu[i])){ 
finavalu[i] = 0.0001;                        
}
}
}
