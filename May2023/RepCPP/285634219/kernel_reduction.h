#ifndef _REDUCTION_H_
#define _REDUCTION_H_



template <typename T>
void kernel_redenergy(const int *s, int L, T *out, const int *H, float h)
{
float energy = 0.f; 
#pragma omp target teams distribute parallel for collapse(3) reduction(+:energy)
for (int z = 0; z < L; z++)
for (int y = 0; y < L; y++)
for (int x = 0; x < L; x++) {
int id = C(x,y,z,L);

float sum = -(float)(s[id] * ((float)(s[C((x+1) >=  L? 0: x+1, y, z, L)] + 
s[C(x, (y+1) >= L? 0 : y+1, z, L)] + s[C(x, y, (z+1) >= L? 0 : z+1, L)]) + h*H[id]));

energy += sum;
}

*out = energy;
}


#endif
