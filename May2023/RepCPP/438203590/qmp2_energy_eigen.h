#ifndef LIBQQC_QMP2_ENERGY_H
#define LIBQQC_QMP2_ENERGY_H

#include <stddef.h> 
#include <Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

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
mv1Dwts(v1Dwts), mv3Dwts (v3Dwts), moffset(offset), 
mnpts_to_proc(npts_to_proc){
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
using MatMap = Map<Matrix <double, Dynamic, Dynamic, RowMajor>>;
using VecMap = Map<VectorXd>;

MatMap map_mmo(mmo, m3Dnpts, mnocc); 
MatMap map_mmv(mmv, m3Dnpts, mnvirt); 
MatMap map_mm1Deps_o(mm1Deps_o, m1Dnpts, mnocc); 
MatMap map_mm1Deps_v(mm1Deps_v, m1Dnpts, mnvirt); 

vector<MatMap> map_mm1Deps_ov; 
map_mm1Deps_ov.reserve(m1Dnpts);
for (size_t i = 0; i < m1Dnpts; i++){
map_mm1Deps_ov.push_back(
MatMap(mm1Deps_ov + i * mnocc * mnvirt, mnocc, mnvirt));
}

VecMap map_mvf(mvf, mnocc+mnvirt); 
VecMap map_mv1Dpts(mv1Dpts, m1Dnpts); 
VecMap map_mv1Dwts(mv1Dwts, m1Dnpts); 

VectorXd o_p; 
VectorXd v_p; 
MatrixXd c2_p;

double energy = 0;
#pragma omp parallel for reduction(+:energy) schedule(dynamic) default(none)\
private(o_p, v_p, c2_p)\
shared(moffset, mnpts_to_proc, m1Dnpts, m3Dnpts, mnocc, mnvirt, \
map_mv1Dwts, map_mmo, map_mm1Deps_o, map_mmv, \
map_mm1Deps_v, mc_c, map_mm1Deps_ov)\
collapse(2)
for (int k = 0; k < m1Dnpts; k++){
for (int p = moffset; p < moffset + mnpts_to_proc; p++){
o_p = map_mmo.row(p)
.cwiseProduct(map_mm1Deps_o.row(k));
v_p = map_mmv.row(p)
.cwiseProduct(map_mm1Deps_v.row(k));
c2_p = MatMap(mc_c + p * mnocc * mnvirt, mnocc, mnvirt)
.cwiseProduct(map_mm1Deps_ov[k]);

for (int q = 0; q <= p; q++){
VectorXd o_q = map_mmo.row(q).transpose();
VectorXd v_q = map_mmv.row(q).transpose();
MatMap map_mc_c_q(mc_c + q * mnocc * mnvirt, 
mnocc, mnvirt);

double jo = 0;
for (int a = 0; a < mnvirt; a++){
double tmp1 = o_p.dot(map_mc_c_q.col(a));
double tmp2 = o_q.dot(c2_p.col(a));
jo += tmp1 * tmp2;
}

double j = (c2_p.cwiseProduct(map_mc_c_q)).sum();

double o = (o_p).dot(o_q);

double v = (v_p).dot(v_q);

double sum = (jo - 2 * j * o) * v;

if (p != q){
sum *= 2.0;
}

energy += map_mv1Dwts(k) * sum; 
}
}
}

return energy;
};
};

}

#endif 
