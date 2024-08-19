



#ifndef _gauss_seidel_h_
#define _gauss_seidel_h_

#include "dimensions.h"
#include "simulation.h"




class GS : public Simulation
{
public:





GS() {
printf("Default Constructor. DO NOT USE. This does not pass along simulation dimensions!\n\n");
exit(0);
}  

GS(Dimensions d);


void init();



void run();


void print();

private:



void solve(int nu, int ncellx, int ncelly, 
real* __restrict__ vo, 
const real* __restrict__ vi, 
const real* __restrict__ an, 
const real* __restrict__ as, 
const real* __restrict__ ae, 
const real* __restrict__ aw);




#pragma acc routine vector
void process_cell(int ix, int iy, int nu, int ncellx, int ncelly,
real* __restrict__ vo,
const real* __restrict__ vi,
const real* __restrict__ an,
const real* __restrict__ as,
const real* __restrict__ ae,
const real* __restrict__ aw);






int v_size, a_size, ncellx, ncelly, nu;


real* __restrict__ v1;
real* __restrict__ v2;
real* __restrict__ an;
real* __restrict__ as;
real* __restrict__ ae;
real* __restrict__ aw;


real* __restrict__ vfinal;
};

#endif 


