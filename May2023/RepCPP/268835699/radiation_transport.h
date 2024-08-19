



#ifndef _radiation_transport_h_
#define _radiation_transport_h_

#include "dimensions.h"
#include "simulation.h"




class RT : public Simulation
{
public:





RT() {
printf("Default Constructor. DO NOT USE. This does not pass along simulation dimensions!\n\n");
exit(0);
}  

RT(Dimensions d);


void init();



void run();


void print();

private:






#pragma acc routine vector
void solve(int ix,
int iy,
int iz,
int octant);


#pragma acc routine seq
void compute(int ix, int iy, int iz, int ie, int ia, int octant);


#pragma acc routine seq
real Quantities_init_face(int ia, int ie, int iu, int scalefactor_space, int octant);


#pragma acc routine seq
int Quantities_scalefactor_space(int ix, int iy, int iz);

#pragma acc routine seq
int Quantities_scalefactor_energy(int ic);






int NE;
int NU = 4;
int NM = 4;
int NA = 32;
int NOCTANT = 8;


real* __restrict__ facexy;
real* __restrict__ facexz;
real* __restrict__ faceyz;
real* __restrict__ local;
real* __restrict__ input;
real* __restrict__ m_from_a;
};

#endif 


