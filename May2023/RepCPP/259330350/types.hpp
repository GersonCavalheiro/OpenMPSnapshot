
#pragma once

#include "decs.hpp"
#include "mpi.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

using parthenon::MeshBlockData;



#define V1 0
#define V2 1
#define V3 2

enum ReconstructionType{donor_cell=0, linear_mc, linear_vl, ppm, mp5, weno5, weno5_lower_poles};

enum InversionStatus{success=0, neg_input, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};

typedef struct {
Real ucon[GR_DIM];
Real ucov[GR_DIM];
Real bcon[GR_DIM];
Real bcov[GR_DIM];
} FourVectors;


class VarMap {
public:
int8_t RHO, UU, U1, U2, U3, B1, B2, B3;
int8_t RHO_ADDED, UU_ADDED, PASSIVE;
int8_t KTOT, K_CONSTANT, K_HOWES, K_KAWAZURA, K_WERNER, K_ROWAN, K_SHARMA;
int8_t PSI, Q, DP;

VarMap(parthenon::PackIndexMap& name_map, bool is_cons)
{
if (is_cons) {
RHO = name_map["cons.rho"].first;
UU = name_map["cons.u"].first;
U1 = name_map["cons.uvec"].first;
B1 = name_map["cons.B"].first;
PSI = name_map["cons.psi_cd"].first;
RHO_ADDED = name_map["cons.rho_added"].first;
UU_ADDED = name_map["cons.u_added"].first;
KTOT = name_map["cons.Ktot"].first;
K_CONSTANT = name_map["cons.Kel_Constant"].first;
K_HOWES = name_map["cons.Kel_Howes"].first;
K_KAWAZURA = name_map["cons.Kel_Kawazura"].first;
K_WERNER = name_map["cons.Kel_Werner"].first;
K_ROWAN = name_map["cons.Kel_Rowan"].first;
K_SHARMA = name_map["cons.Kel_Sharma"].first;
Q = name_map["cons.q"].first;
DP = name_map["cons.dP"].first;
} else {
RHO = name_map["prims.rho"].first;
UU = name_map["prims.u"].first;
U1 = name_map["prims.uvec"].first;
B1 = name_map["prims.B"].first;
PSI = name_map["prims.psi_cd"].first;
RHO_ADDED = name_map["prims.rho_added"].first;
UU_ADDED = name_map["prims.u_added"].first;
KTOT = name_map["prims.Ktot"].first;
K_CONSTANT = name_map["prims.Kel_Constant"].first;
K_HOWES = name_map["prims.Kel_Howes"].first;
K_KAWAZURA = name_map["prims.Kel_Kawazura"].first;
K_WERNER = name_map["prims.Kel_Werner"].first;
K_ROWAN = name_map["prims.Kel_Rowan"].first;
K_SHARMA = name_map["prims.Kel_Sharma"].first;
Q = name_map["prims.q"].first;
DP = name_map["prims.dP"].first;
}
U2 = U1 + 1;
U3 = U1 + 2;
B2 = B1 + 1;
B3 = B1 + 2;
}
};


KOKKOS_INLINE_FUNCTION bool inside(const int& k, const int& j, const int& i,
const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
return (i >= ib.s) && (i <= ib.e) && (j >= jb.s) && (j <= jb.e) && (k >= kb.s) && (k <= kb.e);
}
KOKKOS_INLINE_FUNCTION bool outside(const int& k, const int& j, const int& i,
const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
return (i < ib.s) || (i > ib.e) || (j < jb.s) || (j > jb.e) || (k < kb.s) || (k > kb.e);
}


inline bool IsDomainBound(MeshBlock *pmb, BoundaryFace face)
{
return (pmb->boundary_flag[face] != BoundaryFlag::block &&
pmb->boundary_flag[face] != BoundaryFlag::periodic);
}


#if TRACE
#define PRINTCORNERS 0
#define PRINTZONE 0
inline void PrintCorner(MeshBlockData<Real> *rc)
{
auto rhop = rc->Get("prims.rho").data.GetHostMirrorAndCopy();
auto up = rc->Get("prims.u").data.GetHostMirrorAndCopy();
auto uvecp = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
auto Bp = rc->Get("prims.B").data.GetHostMirrorAndCopy();
auto rhoc = rc->Get("cons.rho").data.GetHostMirrorAndCopy();
auto uc = rc->Get("cons.u").data.GetHostMirrorAndCopy();
auto uvecc = rc->Get("cons.uvec").data.GetHostMirrorAndCopy();
auto Bu = rc->Get("cons.B").data.GetHostMirrorAndCopy();
auto pflag = rc->Get("pflag").data.GetHostMirrorAndCopy();
const IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
const IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
const IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
std::cerr << "p:";
for (int j=0; j<8; j++) {
std::cerr << std::endl;
for (int i=0; i<8; i++) {
fprintf(stderr, "%.5g\t", pflag(kb.s, j, i));
}
}
std::cerr << std::endl << std::endl;
}

inline void PrintZone(MeshBlockData<Real> *rc)
{
auto rhop = rc->Get("prims.rho").data.GetHostMirrorAndCopy();
auto up = rc->Get("prims.u").data.GetHostMirrorAndCopy();
auto uvecp = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
auto Bp = rc->Get("prims.B").data.GetHostMirrorAndCopy();
auto q = rc->Get("prims.q").data.GetHostMirrorAndCopy();
auto dP = rc->Get("prims.dP").data.GetHostMirrorAndCopy();
std::cerr << "RHO: " << rhop(0,0,100)
<< " UU: "  << up(0,0,100)
<< " U: "   << uvecp(0, 0,0,100) << " " << uvecp(1, 0,0,100)<< " " << uvecp(2, 0,0,100)
<< " B: "   << Bp(0, 0,0,100) << " " << Bp(1, 0,0,100) << " " << Bp(2, 0,0,100)
<< " q: "   << q(0,0,100) 
<< " dP: "  << dP(0,0,100) << std::endl;
}

inline void Flag(std::string label)
{
if(MPIRank0()) std::cerr << label << std::endl;
}

inline void Flag(MeshBlockData<Real> *rc, std::string label)
{
if(MPIRank0()) {
std::cerr << label << std::endl;
if(PRINTCORNERS) PrintCorner(rc);
if(PRINTZONE) PrintZone(rc);
}
}

inline void Flag(MeshData<Real> *md, std::string label)
{
if(MPIRank0()) {
std::cerr << label << std::endl;
if(PRINTCORNERS || PRINTZONE) {
auto rc = md->GetBlockData(0).get();
if(PRINTCORNERS) PrintCorner(rc);
if(PRINTZONE) PrintZone(rc);
}
}
}

#else
inline void Flag(std::string label) {}
inline void Flag(MeshBlockData<Real> *rc, std::string label) {}
inline void Flag(MeshData<Real> *md, std::string label) {}
#endif
