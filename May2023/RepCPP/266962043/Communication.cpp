#include "Communication.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <strings.h>

void Distrib::createDecomposition(Config *conf) {
Domain &dom = conf->dom_;

ulist_.resize(nbp_);
for(int id=0; id<nbp_; id++)
ulist_[id].pid_ = -1;

ulist_[0].pid_ = 0;
ulist_[0].nbp_ = nbp_;

for(int j=0; j<DIMENSION; j++)
ulist_[0].xmin_[j] = 0;

for(int j=0; j<DIMENSION; j++) {
ulist_[0].xmax_[j] = dom.nxmax_[j] - 1;
nxmax_[j] = dom.nxmax_[j];
}

for(int count = 1; count < nbp_; count++) {
const float64 cutbound = 10000000000.;
float64 cutbest = cutbound;
int cutpart = -1;
int cutid = -1;
int cutwidth = -1;
int cutdir = -1;

for(int direction = 0; direction < NB_DIMS_DECOMPOSITION; direction++) {
for(int id = 0; id < count; id++) {
int nbp = ulist_[id].nbp_;
if(nbp > 1) {
int center = floor(nbp * 0.5);
for(int part = center; part < nbp; part++) {
const int width = ulist_[id].xmax_[direction] - ulist_[id].xmin_[direction] + 1;
const float64 fract = static_cast<float64>(part) / static_cast<float64>(nbp);
const float64 splitscore = 1. - width / static_cast<float64>(dom.nxmax_[direction]);
const int tmpwidth = round(fract * width);
const float64 ratioscore = fabs(fract * width - tmpwidth);
const float64 cutscore = ratioscore + splitscore + cutbound * (tmpwidth < MMAX + 2) + cutbound * ((width - tmpwidth) < (MMAX + 2));

if(cutscore < cutbest) {
cutid = id;
cutbest = cutscore;
cutpart = part;
cutwidth = tmpwidth;
cutdir = direction;
}
}
}
}
}
if(cutbest == cutbound) {
printf("No cut found to create domain decomposition, reduce nb of processes\n");
}

for(int j=0; j<DIMENSION; j++) {
ulist_[count].xmax_[j] = ulist_[cutid].xmax_[j];
ulist_[count].xmin_[j] = ulist_[cutid].xmin_[j];
}
ulist_[cutid].nbp_          -= cutpart;
ulist_[cutid].xmin_[cutdir] += cutwidth;
ulist_[count].nbp_          = cutpart;
ulist_[count].xmax_[cutdir] = ulist_[cutid].xmin_[cutdir] - 1;
}

uint64 msum = 0;
for(int id = 0; id < nbp_; id++) {
const uint64 mcontrib = (ulist_[id].xmax_[3] - ulist_[id].xmin_[3] + 1) * (ulist_[id].xmax_[2] - ulist_[id].xmin_[2] + 1)
* (ulist_[id].xmax_[1] - ulist_[id].xmin_[1] + 1) * (ulist_[id].xmax_[0] - ulist_[id].xmin_[0] + 1);

if(master())
printf("[%d] local domain [%4u:%4u,%4u:%4u,%4u:%4u,%4u:%4u] cost %lu\n", id, ulist_[id].xmin_[0], ulist_[id].xmax_[0], ulist_[id].xmin_[1],
ulist_[id].xmax_[1], ulist_[id].xmin_[2], ulist_[id].xmax_[2], ulist_[id].xmin_[3], ulist_[id].xmax_[3], mcontrib);

msum += mcontrib;
}
uint64 mref = static_cast<uint64>(dom.nxmax_[3]) * static_cast<uint64>(dom.nxmax_[2]) * static_cast<uint64>(dom.nxmax_[1]) * static_cast<uint64>(dom.nxmax_[0]);

if(mref != msum) {
printf("Problem: sum check mref %lf mcalc %lu \n", mref, msum);

MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
exit(1);
}

node_ = &(ulist_[pid_]);

for(int j=0; j<DIMENSION; j++) {
dom.local_nx_[j]    = node_->xmax_[j] - node_->xmin_[j] + 1;
dom.local_nxmin_[j] = node_->xmin_[j];
dom.local_nxmax_[j] = node_->xmax_[j];
}
}


void Distrib::neighboursList(Config *conf, RealView4D &halo_fn) {
const Domain& dom = conf->dom_;
const int s_nxmax  = dom.nxmax_[0];
const int s_nymax  = dom.nxmax_[1];
const int s_nvxmax = dom.nxmax_[2];
const int s_nvymax = dom.nxmax_[3];

Urbnode &mynode = ulist_[pid_];

for(int ivy = mynode.xmin_[3] - HALO_PTS; ivy < mynode.xmax_[3] + HALO_PTS + 1; ivy++) {
for(int ivx = mynode.xmin_[2] - HALO_PTS; ivx < mynode.xmax_[2] + HALO_PTS + 1; ivx++) {
const int jvy = (s_nvymax + ivy) % s_nvymax;
const int jvx = (s_nvxmax + ivx) % s_nvxmax;

for(int iy = mynode.xmin_[1] - HALO_PTS; iy < mynode.xmax_[1] + HALO_PTS + 1; iy++) {
for(int ix = mynode.xmin_[0] - HALO_PTS; ix < mynode.xmax_[0] + HALO_PTS + 1; ix++) {
const int jy = (s_nymax + iy) % s_nymax;
const int jx = (s_nxmax + ix) % s_nxmax;

int id = 0;
bool notfound = true;
while(notfound && (id<nbp_)) {
const Urbnode &node = ulist_[id];
if(    node.xmin_[0] <= jx  && jx  <= node.xmax_[0]
&& node.xmin_[1] <= jy  && jy  <= node.xmax_[1] 
&& node.xmin_[2] <= jvx && jvx <= node.xmax_[2]
&& node.xmin_[3] <= jvy && jvy <= node.xmax_[3]
) {
halo_fn(ix, iy, ivx, ivy) = static_cast<float64>(id);
notfound = 0;
}
id++;
}
assert(!notfound);
}
}
}
}

int id = pid_;
Urbnode &node = ulist_[id];
std::vector<Halo> hlist;
int count = 0;
int jv[4];

for(jv[0] = -1; jv[0] < 2; jv[0]++) {
for(jv[1] = -1; jv[1] < 2; jv[1]++) {
for(jv[2] = -1; jv[2] < 2; jv[2]++) {
for(jv[3] = -1; jv[3] < 2; jv[3]++) {
int face[8] = {
node.xmin_[0],
node.xmax_[0] + 1,
node.xmin_[1],
node.xmax_[1] + 1,
node.xmin_[2],
node.xmax_[2] + 1,
node.xmin_[3],
node.xmax_[3] + 1,
};

for(int k = 0; k < 4; k++) {
if(jv[k] == -1) {
face[2 * k + 0] = node.xmin_[k] - HALO_PTS;
face[2 * k + 1] = node.xmin_[k] - 1;
}

if(jv[k] == 1) {
face[2 * k + 0] = node.xmax_[k] + 1;
face[2 * k + 1] = node.xmax_[k] + HALO_PTS;
}
}

if(jv[0] != 0 || jv[1] != 0 || jv[2] != 0 || jv[3] != 0)
getNeighbours(conf, halo_fn, face, hlist, mynode.xmin_, mynode.xmax_, count);

count++;
}
}
}
}

assert(dom.nxmax_[0] < 2010);
assert(dom.nxmax_[1] < 2010);
assert(dom.nxmax_[2] < 2010);
assert(dom.nxmax_[3] < 2010);
for(int k = 0; k < 4; k++) {
int k0 = k;
int k1 = (k + 1) % 4;
int k2 = (k + 2) % 4;
int k3 = (k + 3) % 4;

auto larger = [k0, k1, k2, k3](const Halo &a, const Halo &b) {
if(a.pid_ == b.pid_) {
if(a.xmin_[k0] == b.xmin_[k0]) {
if(a.xmin_[k1] == b.xmin_[k1]) {
if(a.xmin_[k2] == b.xmin_[k2]) {
return (a.xmin_[k3] < b.xmin_[k3]);
} else {
return (a.xmin_[k2] < b.xmin_[k2]);
}
} else {
return (a.xmin_[k1] < b.xmin_[k1]);
}
} else {
return (a.xmin_[k0] < b.xmin_[k0]);
}
} else {
return (a.pid_ < b.pid_);
}
};
std::sort(hlist.begin(), hlist.end(), larger);
int cursize = hlist.size();
int oldsize = cursize + 1;
while(oldsize > cursize) {
for(auto it = hlist.begin(); it != hlist.end(); ++it) {
auto next = it + 1;
if(next != hlist.end())
mergeElts(hlist, it, next);
}
oldsize = cursize;
cursize = hlist.size();
}
}

recv_list_.assign(hlist.begin(), hlist.end());
}


void Distrib::bookHalo(Config *conf) {
int size_halo = sizeof(Halo);
std::vector<unsigned short> vectp(nbp_, 0);
std::vector<unsigned short> redp(nbp_, 0);
std::vector<MPI_Request> req;
std::vector<MPI_Status>  stat;
int tag = 111;
int nbreq = 0;

for(auto it = recv_list_.begin(); it != recv_list_.end(); ++it) {
vectp[(*it).pid_]++;
}

MPI_Allreduce(vectp.data(), redp.data(), nbp_, MPI_UNSIGNED_SHORT, MPI_SUM, MPI_COMM_WORLD);

nbreq = redp[pid_];
send_list_.resize(nbreq);
req.resize(nbreq + recv_list_.size());
stat.resize(nbreq + recv_list_.size());

for(size_t i = 0; i < send_list_.size(); i++) {
Halo *sender = &(send_list_[i]);
MPI_Irecv(sender, size_halo, MPI_BYTE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &(req[i]));
}
for(size_t i = 0; i < recv_list_.size(); i++) {
Halo *recver = &(recv_list_[i]);
int dest = recver->pid_;
MPI_Isend(recver, size_halo, MPI_BYTE, dest, tag, MPI_COMM_WORLD, &(req[nbreq]));
nbreq++;
}
MPI_Waitall(nbreq, req.data(), stat.data());

for(size_t i = 0; i < send_list_.size(); i++) {
Halo *halo = &(send_list_[i]);
int32 size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1) * (halo->xmax_[2] - halo->xmin_[2] + 1)
* (halo->xmax_[3] - halo->xmin_[3] + 1);
halo->size_ = size;
}

for(size_t i = 0; i < recv_list_.size(); i++) {
Halo *halo = &(recv_list_[i]);
int32 size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1) * (halo->xmax_[2] - halo->xmin_[2] + 1)
* (halo->xmax_[3] - halo->xmin_[3] + 1);
halo->size_ = size;
}

for(size_t i = 0; i < send_list_.size(); i++) {
Halo *sender = &(send_list_[i]);
sender->pid_ = stat[i].MPI_SOURCE;
}

send_buffers_ = new Halos();
recv_buffers_ = new Halos();
send_buffers_->set(conf, send_list_, "send", nbp_, pid_);
recv_buffers_->set(conf, recv_list_, "recv", nbp_, pid_);

fprintf(stderr, "[%d] Number of halo blocs = %lu\n", pid_, send_list_.size());
}


void Distrib::getNeighbours(const Config *conf, const RealView4D &halo_fn, int xrange[8],
std::vector<Halo> &hlist, int lxmin[4], int lxmax[4], int count) {
std::vector<Halo> vhalo;
uint8 neighbours[nbp_];
uint32 nb_neib = 0;

for(uint32 j=0; j<nbp_; j++)
neighbours[j] = 255;

vhalo.clear();
for(int ivy = xrange[6]; ivy <= xrange[7]; ivy++) {
for(int ivx = xrange[4]; ivx <= xrange[5]; ivx++) {
for(int iy = xrange[2]; iy <= xrange[3]; iy++) {
for(int ix = xrange[0]; ix <= xrange[1]; ix++) {
const uint32 neibid = round(halo_fn(ix, iy, ivx, ivy));

if(neighbours[neibid] == 255) {
Halo myneib;
neighbours[neibid] = nb_neib;
myneib.pid_     = neibid;
myneib.xmin_[0] = ix;
myneib.xmin_[1] = iy;
myneib.xmin_[2] = ivx;
myneib.xmin_[3] = ivy;
myneib.xmax_[0] = ix;
myneib.xmax_[1] = iy;
myneib.xmax_[2] = ivx;
myneib.xmax_[3] = ivy;
myneib.tag_ = count;
for(int k = 0; k < 4; k++) {
myneib.lxmin_[k] = lxmin[k];
myneib.lxmax_[k] = lxmax[k];
}
vhalo.push_back(myneib);
nb_neib++;
}
uint8 neighbour = neighbours[neibid];
vhalo[neighbour].xmax_[0] = ix;
vhalo[neighbour].xmax_[1] = iy;
vhalo[neighbour].xmax_[2] = ivx;
vhalo[neighbour].xmax_[3] = ivy;
}
}
}
}

hlist.insert(hlist.end(), vhalo.begin(), vhalo.end());
}


int Distrib::mergeElts(std::vector<Halo> &v, std::vector<Halo>::iterator &f, std::vector<Halo>::iterator &g) {
if((*f).pid_ == (*g).pid_) {
for(uint32 i = 0; i < DIMENSION; i++) {
bool equal = true;
int retcode = 0;
for(uint32 j = 0; j < DIMENSION; j++) {
if(j != i) {
equal = equal && ((*f).xmin_[j] == (*g).xmin_[j]) && ((*f).xmax_[j] == (*g).xmax_[j]);
}
}

if(equal && ((*f).xmin_[i] == (*g).xmax_[i] + 1)) {
(*f).xmin_[i] = (*g).xmin_[i];
retcode = 1;
}

if(equal && ((*f).xmax_[i] + 1 == (*g).xmin_[i])) {
(*f).xmax_[i] = (*g).xmax_[i];
retcode = 2;
}

if(retcode != 0) {
v.erase(g);
return retcode;
}
}
}
return 0;
}

void Distrib::Isend(int &creq, std::vector<MPI_Request> &req) {
int nb_halos = send_buffers_->nb_merged_halos_;
int size_local_copy = send_buffers_->merged_size(pid_);
for(int i = 0; i < nb_halos; i++) {
if(i == pid_) {
local_copy(send_buffers_, recv_buffers_);
} else {
float64 *head  = send_buffers_->head(i);
const int size = send_buffers_->merged_size(i);
const int pid  = send_buffers_->merged_pid(i);
const int tag  = send_buffers_->merged_tag(i);
if(size != 0 ) {
#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target data use_device_ptr(head)
#endif
MPI_Isend(head, size, MPI_DOUBLE, pid, tag, MPI_COMM_WORLD, &(req[creq++]));
}
}
}
}

void Distrib::Irecv(int &creq, std::vector<MPI_Request> &req) {
int nb_halos = recv_buffers_->nb_merged_halos_;
int size_local_copy = recv_buffers_->merged_size(pid_);
for(int i = 0; i < nb_halos; i++) {
if(i != pid_) {
float64 *head  = recv_buffers_->head(i);
const int size = recv_buffers_->merged_size(i);
const int pid  = recv_buffers_->merged_pid(i);
const int tag  = recv_buffers_->merged_tag(i);
if(size != 0 ) {
#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target data use_device_ptr(head)
#endif
MPI_Irecv(head, size, MPI_DOUBLE, pid, tag, MPI_COMM_WORLD, &(req[creq++]));
}
}
}
}


void Distrib::packAndBoundary(Config *conf, RealView4D &halo_fn, Halos *send_buffers) {
if(spline_) {
pack(halo_fn, send_buffers);
boundary_condition(halo_fn, send_buffers);
}
}


void Distrib::exchangeHalo(Config *conf, RealView4D &halo_fn, std::vector<Timer*> &timers) {
std::vector<MPI_Request> req;
std::vector<MPI_Status>  stat;
int nbreq = 0, creq = 0;
nbreq = recv_buffers_->nb_reqs() + send_buffers_->nb_reqs();
creq  = 0;
req.resize(nbreq);
stat.resize(nbreq);

timers[TimerEnum::pack]->begin();
Irecv(creq, req);

packAndBoundary(conf, halo_fn, send_buffers_);
timers[TimerEnum::pack]->end();
timers[TimerEnum::comm]->begin();

Isend(creq, req); 
Waitall(creq, req, stat);

std::vector<MPI_Request>().swap(req);
std::vector<MPI_Status>().swap(stat);
timers[TimerEnum::comm]->end();
timers[TimerEnum::unpack]->begin();

unpack(halo_fn, recv_buffers_);
timers[TimerEnum::unpack]->end();
}
