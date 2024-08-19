
#pragma once

#include "decs.hpp"

#include "reconstruct/dc_inline.hpp"
#include "reconstruct/plm_inline.hpp"

using namespace parthenon;

#define EPS 1.e-26


namespace KReconstruction
{

KOKKOS_INLINE_FUNCTION Real mc(const Real dm, const Real dp)
{
const Real r = (m::abs(dp) > 0. ? dm/dp : 2.0);
return m::max(0.0, m::min(2.0, m::min(2*r,0.5*(1+r))));
}

template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX1(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real dql = q(p, k, j, i) - q(p, k, j, i - 1);
Real dqr = q(p, k, j, i + 1) - q(p, k, j, i);
Real dq = mc(dql, dqr)*dqr;
ql(p, i+1) = q(p, k, j, i) + 0.5*dq;
qr(p, i) = q(p, k, j, i) - 0.5*dq;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX2(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real dql = q(p, k, j, i) - q(p, k, j - 1, i);
Real dqr = q(p, k, j + 1, i) - q(p, k, j, i);
Real dq = mc(dql, dqr)*dqr;
ql(p, i) = q(p, k, j, i) + 0.5*dq;
qr(p, i) = q(p, k, j, i) - 0.5*dq;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX3(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T& q, ScratchPad2D<Real> &ql,
ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real dql = q(p, k, j, i) - q(p, k - 1, j, i);
Real dqr = q(p, k + 1, j, i) - q(p, k, j, i);
Real dq = mc(dql, dqr)*dqr;
ql(p, i) = q(p, k, j, i) + 0.5*dq;
qr(p, i) = q(p, k, j, i) - 0.5*dq;
}
);
}
}


KOKKOS_INLINE_FUNCTION void weno5(const Real& x1, const Real& x2, const Real& x3, const Real& x4, const Real& x5,
Real &lout, Real &rout)
{
Real beta[3], c1, c2;
c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

Real wtr[3] = {(1./16.)/den[0], (5./8. )/den[1], (5./16.)/den[2]};
Real Wr = wtr[0] + wtr[1] + wtr[2];

Real wtl[3] = {(1./16.)/den[2], (5./8. )/den[1], (5./16.)/den[0]};
Real Wl = wtl[0] + wtl[1] + wtl[2];

lout = ((3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3)*(wtl[0] / Wl) +
((-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2)*(wtl[1] / Wl) +
((3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1)*(wtl[2] / Wl);
rout = ((3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3)*(wtr[0] / Wr) +
((-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4)*(wtr[1] / Wr) +
((3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5)*(wtr[2] / Wr);
}
KOKKOS_INLINE_FUNCTION void weno5l(const Real x1, const Real& x2, const Real& x3, const Real x4, const Real& x5,
Real &lout)
{
Real beta[3], c1, c2;
c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

Real wtl[3] = {(1./16.)/den[2], (5./8. )/den[1], (5./16.)/den[0]};
Real Wl = wtl[0] + wtl[1] + wtl[2];

lout = ((3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3)*(wtl[0] / Wl) +
((-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2)*(wtl[1] / Wl) +
((3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1)*(wtl[2] / Wl);
}
KOKKOS_INLINE_FUNCTION void weno5r(const Real& x1, const Real& x2, const Real& x3, const Real x4, const Real& x5,
Real &rout)
{
Real beta[3], c1, c2;
c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

Real wtr[3] = {(1./16.)/den[0], (5./8. )/den[1], (5./16.)/den[2]};
Real Wr = wtr[0] + wtr[1] + wtr[2];

rout = ((3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3)*(wtr[0] / Wr) +
((-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4)*(wtr[1] / Wr) +
((3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5)*(wtr[2] / Wr);
}

template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X1(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real lout, rout;
weno5(q(p, k, j, i - 2),
q(p, k, j, i - 1),
q(p, k, j, i),
q(p, k, j, i + 1),
q(p, k, j, i + 2), lout, rout);
ql(p, i+1) = rout;
qr(p, i) = lout;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real lout, rout;
weno5(q(p, k, j - 2, i),
q(p, k, j - 1, i),
q(p, k, j, i),
q(p, k, j + 1, i),
q(p, k, j + 2, i), lout, rout);
ql(p, i) = rout;
qr(p, i) = lout;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2l(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real rout;
weno5r(q(p, k, j - 2, i),
q(p, k, j - 1, i),
q(p, k, j, i),
q(p, k, j + 1, i),
q(p, k, j + 2, i), rout);
ql(p, i) = rout;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2r(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real lout;
weno5l(q(p, k, j - 2, i),
q(p, k, j - 1, i),
q(p, k, j, i),
q(p, k, j + 1, i),
q(p, k, j + 2, i), lout);
qr(p, i) = lout;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real lout, rout;
weno5(q(p, k - 2, j, i),
q(p, k - 1, j, i),
q(p, k, j, i),
q(p, k + 1, j, i),
q(p, k + 2, j, i), lout, rout);
ql(p, i) = rout;
qr(p, i) = lout;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3l(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real rout;
weno5r(q(p, k - 2, j, i),
q(p, k - 1, j, i),
q(p, k, j, i),
q(p, k + 1, j, i),
q(p, k + 2, j, i), rout);
ql(p, i) = rout;
}
);
}
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3r(parthenon::team_mbr_t const &member, const int& k, const int& j,
const int& il, const int& iu, const T &q, ScratchPad2D<Real> &qr)
{
const int nu = q.GetDim(4) - 1;
for (int p = 0; p <= nu; ++p) {
parthenon::par_for_inner(member, il, iu,
KOKKOS_LAMBDA_1D {
Real lout;
weno5l(q(p, k - 2, j, i),
q(p, k - 1, j, i),
q(p, k, j, i),
q(p, k + 1, j, i),
q(p, k + 2, j, i), lout);
qr(p, i) = lout;
}
);
}
}


template <ReconstructionType Recon, int dir>
KOKKOS_INLINE_FUNCTION void reconstruct(parthenon::team_mbr_t& member, const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr) {}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::donor_cell, X1DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
DonorCellX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::donor_cell, X2DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
DonorCellX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::donor_cell, X3DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
DonorCellX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
DonorCellX3(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_vl, X1DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
PiecewiseLinearX1(member, k, j, is_l, ie_l, G, P, ql, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_vl, X2DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, G, P, ql, q_u, qc, dql, dqr, dqm);
PiecewiseLinearX2(member, k, j, is_l, ie_l, G, P, q_u, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_vl, X3DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, G, P, ql, q_u, qc, dql, dqr, dqm);
PiecewiseLinearX3(member, k, j, is_l, ie_l, G, P, q_u, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_mc, X1DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
KReconstruction::PiecewiseLinearX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_mc, X2DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_mc, X3DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
KReconstruction::PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
KReconstruction::PiecewiseLinearX3(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5, X1DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5, X2DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5, X3DIR>(parthenon::team_mbr_t& member,
const GRCoordinates& G, const VariablePack<Real> &P,
const int& k, const int& j, const int& is_l, const int& ie_l, 
ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
}

} 
