#ifndef LIBQQC_QMP2_ENERGY_H
#define LIBQQC_QMP2_ENERGY_H

#include <stddef.h> 
#include <iostream> 

using namespace std;

namespace libqqc {

class Qmp2_energy {
private:
size_t &m1Dnpts; 
size_t &m3Dnpts; 
size_t &mnocc; 
size_t &mnvirt; 

double *mmo = NULL; 
double *mmv = NULL; 
double *mc_c = NULL; 
double *mm1Deps_o = NULL; 
double *mm1Deps_v = NULL; 
double *mm1Deps_ov = NULL; 

double *mvf = NULL; 
double *mv1Dpts = NULL; 
double *mv1Dwts = NULL; 
double *mv3Dwts = NULL; 

size_t &moffset; 
size_t &mnpts_to_proc; 

public: 
Qmp2_energy (size_t &p1Dnpts, size_t &p3Dnpts, size_t &nocc, 
size_t &nvirt, double *mo, double *mv, double *c_c,
double *m1Deps_o, double *m1Deps_v, double *m1Deps_ov, 
double *vf, double *v1Dpts, double *v1Dwts, double *v3Dwts, 
size_t &offset, 
size_t &npts_to_proc) : m1Dnpts(p1Dnpts), 
m3Dnpts(p3Dnpts), mnocc(nocc), mnvirt(nvirt), mmo(mo),
mmv(mv), mc_c(c_c), mm1Deps_o(m1Deps_o), mm1Deps_v(m1Deps_v),
mm1Deps_ov(m1Deps_ov), mvf(vf), mv1Dpts(v1Dpts), 
mv1Dwts(v1Dwts), mv3Dwts (v3Dwts), moffset(offset), mnpts_to_proc(npts_to_proc)
{
#pragma omp parallel for schedule(static) default(none)\
shared(m3Dnpts, mnocc, mnvirt, mc_c, mv3Dwts)\
collapse(2)
for (size_t p = 0; p < m3Dnpts; p++){
for (size_t i = 0; i < mnocc; i++){
for (size_t a = 0; a < mnvirt; a++){
mc_c [p * mnvirt * mnocc + i * mnvirt + a] *=
mv3Dwts[p];
}
}
}
};

double compute (){


double energy = 0;
#pragma omp parallel for reduction(+:energy) schedule(dynamic) default(none)\
shared(moffset, mnpts_to_proc, m1Dnpts, m3Dnpts, mnocc, mnvirt, mv1Dwts, mmo, mm1Deps_o, mmv, mm1Deps_v, mc_c, mm1Deps_ov)\
collapse(2)
for (size_t k = 0; k < m1Dnpts; k++){
for (size_t p = moffset; p < moffset+mnpts_to_proc; p++){
double v_o_p[mnocc];
double v_v_p[mnvirt];
double m_c_p[mnocc][mnvirt];
for (size_t i = 0; i < mnocc; i++) 
v_o_p[i] = mmo[p * mnocc + i] 
* mm1Deps_o[k * mnocc + i]; 
for (size_t a = 0; a < mnvirt; a++) 
v_v_p[a] = mmv[p * mnvirt + a] 
* mm1Deps_v[k * mnvirt + a]; 
for (size_t i =0; i < mnocc; i++) {
for (size_t a = 0; a < mnvirt; a++){
m_c_p[i][a] = 
mc_c [p * mnocc * mnvirt + i * mnvirt + a] *
mm1Deps_ov[k * mnocc * mnvirt + i * mnvirt + a]; 
}
}

for (size_t q = 0; q <= p; q++){

double jo = 0;
for (size_t a = 0; a < mnvirt; a++){
double tmp1 = 0;
double tmp2 = 0;
for (size_t i = 0; i < mnocc; i++){
tmp1 += v_o_p[i] *
mc_c[q * mnocc * mnvirt + i * mnvirt + a];
tmp2 += mmo[q*mnocc+i] * m_c_p[i][a];
}
jo += tmp1 * tmp2;
}

double j = 0;
for (size_t i = 0; i < mnocc; i++){
for (size_t a = 0; a < mnvirt; a++){
j += m_c_p[i][a] 
* mc_c[q * mnocc * mnvirt + i * mnvirt+a];
}
}

double o = 0;
for (size_t i = 0; i < mnocc; i++){
o += v_o_p[i] *  mmo[q * mnocc + i];
}

double v = 0;
for (size_t a = 0; a < mnvirt; a++){
v += v_v_p[a] * mmv[q * mnvirt + a];
}

double sum = (jo - 2 * j * o) * v;

if (p != q){
sum *= 2.0;
}

energy += sum * mv1Dwts[k];
}
}
}
return energy;
};
};

}

#endif 
