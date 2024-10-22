
#ifdef HAVE_CONFIG_H
#include "config.h"  
#endif               

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "CnError.h"
#include "DfCalcGridX.h"
#include "DfGenerateGrid.h"
#include "Fl_Geometry.h"
#include "TlFile.h"
#include "TlPrdctbl.h"
#include "TlUtils.h"
#include "tl_dense_symmetric_matrix_lapack.h"


const double DfCalcGridX::TOOBIG = 30.0;
const double DfCalcGridX::EPS = std::numeric_limits<double>::epsilon();
const double DfCalcGridX::INV_SQRT3 = 1.0 / std::sqrt(3.0);
const double DfCalcGridX::INV_SQRT12 = 1.0 / std::sqrt(12.0);

DfCalcGridX::DfCalcGridX(TlSerializeData* pPdfParam)
: DfObject(pPdfParam),
m_tlOrbInfo((*pPdfParam)["coordinates"], (*pPdfParam)["basis_set"]) {
const TlSerializeData& pdfParam = *pPdfParam;

DfXCFunctional dfXcFunc(pPdfParam);
this->functionalType_ = dfXcFunc.getFunctionalType();

this->inputtedDensityCutoffValue_ = 1.0E-16;
if (!(pdfParam["xc-density-threshold"].getStr().empty())) {
this->inputtedDensityCutoffValue_ =
pdfParam["xc-density-threshold"].getDouble();
}
this->m_densityCutOffValueA = this->inputtedDensityCutoffValue_;
this->m_densityCutOffValueB = this->inputtedDensityCutoffValue_;

this->m_inputedCutoffThreshold = pdfParam["cut_value"].getDouble();

this->isDebugOutPhiTable_ =
(TlUtils::toUpper(pdfParam["debug_out_phi_table"].getStr()) == "YES")
? true
: false;
this->isSaveGrad_ =
(TlUtils::toUpper(pdfParam["DfCalcGridX::save_grad"].getStr()) == "YES")
? true
: false;
}

DfCalcGridX::~DfCalcGridX() {}

void DfCalcGridX::defineCutOffValues(const TlDenseSymmetricMatrix_Lapack& P) {
const double maxValueOfP = std::max(P.getMaxAbsoluteElement(), 1.0E-16);
if (maxValueOfP < 1.0) {
this->m_densityCutOffValueA /= maxValueOfP;
}
this->log_.info(TlUtils::format(" density cutoff value = %e",
this->m_densityCutOffValueA));
}

void DfCalcGridX::defineCutOffValues(const TlDenseSymmetricMatrix_Lapack& PA,
const TlDenseSymmetricMatrix_Lapack& PB) {
const double maxValueOfPA = std::max(PA.getMaxAbsoluteElement(), 1.0E-16);
const double maxValueOfPB = std::max(PB.getMaxAbsoluteElement(), 1.0E-16);
if (maxValueOfPA < 1.0) {
this->m_densityCutOffValueA /= maxValueOfPA;
}
if (maxValueOfPB < 1.0) {
this->m_densityCutOffValueB /= maxValueOfPB;
}
this->log_.info(TlUtils::format("density cutoff value(alpha) = %e",
this->m_densityCutOffValueA));
this->log_.info(TlUtils::format("density cutoff value(beta ) = %e",
this->m_densityCutOffValueB));
}

void DfCalcGridX::getPrefactor(const int nType, const TlPosition& pos,
double* pPrefactor) {
*pPrefactor = 1.0;

switch (nType) {
case 0:  
break;
case 1:  
*pPrefactor = pos.x();
break;
case 2:  
*pPrefactor = pos.y();
break;
case 3:  
*pPrefactor = pos.z();
break;
case 4:  
*pPrefactor = pos.x() * pos.y();
break;
case 5:  
*pPrefactor = pos.z() * pos.x();
break;
case 6:  
*pPrefactor = pos.y() * pos.z();
break;
case 7:  
*pPrefactor = 0.5 * (pos.x() * pos.x() - pos.y() * pos.y());
break;
case 8:  
*pPrefactor =
INV_SQRT3 * (pos.z() * pos.z() -
0.5 * (pos.x() * pos.x() + pos.y() * pos.y()));
break;
default:
std::cout << "Basis Type is Wrong." << std::endl;
break;
}
}

void DfCalcGridX::getPrefactorForDerivative(const int nType, const double alpha,
const TlPosition& pos,
double* pPrefactorX,
double* pPrefactorY,
double* pPrefactorZ) {
assert(pPrefactorX != NULL);
assert(pPrefactorY != NULL);
assert(pPrefactorZ != NULL);

const double alpha2 = 2.0 * alpha;
const double x = pos.x();
const double y = pos.y();
const double z = pos.z();

switch (nType) {
case 0: {
*pPrefactorX = -alpha2 * x;
*pPrefactorY = -alpha2 * y;
*pPrefactorZ = -alpha2 * z;
} break;
case 1: {
*pPrefactorX = -alpha2 * x * x + 1.0;
*pPrefactorY = -alpha2 * x * y;
*pPrefactorZ = -alpha2 * x * z;
} break;
case 2: {
*pPrefactorX = -alpha2 * y * x;
*pPrefactorY = -alpha2 * y * y + 1.0;
*pPrefactorZ = -alpha2 * y * z;
} break;
case 3: {
*pPrefactorX = -alpha2 * z * x;
*pPrefactorY = -alpha2 * z * y;
*pPrefactorZ = -alpha2 * z * z + 1.0;
} break;
case 4: {
const double xy = x * y;
*pPrefactorX = -alpha2 * xy * x + y;
*pPrefactorY = -alpha2 * xy * y + x;
*pPrefactorZ = -alpha2 * xy * z;
} break;
case 5: {
const double xz = x * z;
*pPrefactorX = -alpha2 * xz * x + z;
*pPrefactorY = -alpha2 * xz * y;
*pPrefactorZ = -alpha2 * xz * z + x;
} break;
case 6: {
const double yz = y * z;
*pPrefactorX = -alpha2 * yz * x;
*pPrefactorY = -alpha2 * yz * y + z;
*pPrefactorZ = -alpha2 * yz * z + y;
} break;
case 7: {
const double xx = x * x;
const double xx_X = -alpha2 * xx * x + 2.0 * x;
const double xx_Y = -alpha2 * xx * y;
const double xx_Z = -alpha2 * xx * z;

const double yy = y * y;
const double yy_X = -alpha2 * yy * x;
const double yy_Y = -alpha2 * yy * y + 2.0 * y;
const double yy_Z = -alpha2 * yy * z;

*pPrefactorX = 0.5 * (xx_X - yy_X);
*pPrefactorY = 0.5 * (xx_Y - yy_Y);
*pPrefactorZ = 0.5 * (xx_Z - yy_Z);
} break;
case 8: {
const double xx = x * x;
const double xx_X = -alpha2 * xx * x + 2.0 * x;
const double xx_Y = -alpha2 * xx * y;
const double xx_Z = -alpha2 * xx * z;

const double yy = y * y;
const double yy_X = -alpha2 * yy * x;
const double yy_Y = -alpha2 * yy * y + 2.0 * y;
const double yy_Z = -alpha2 * yy * z;

const double zz = z * z;
const double zz_X = -alpha2 * zz * x;
const double zz_Y = -alpha2 * zz * y;
const double zz_Z = -alpha2 * zz * z + 2.0 * z;

*pPrefactorX = INV_SQRT3 * (zz_X - 0.5 * (xx_X + yy_X));
*pPrefactorY = INV_SQRT3 * (zz_Y - 0.5 * (xx_Y + yy_Y));
*pPrefactorZ = INV_SQRT3 * (zz_Z - 0.5 * (xx_Z + yy_Z));
} break;
default:
std::cout << "Basis Type is Wrong." << std::endl;
break;
}
}

void DfCalcGridX::getPrefactorForSecondDerivative(
const int nType, const double a, const TlPosition& pos, double* pXX,
double* pXY, double* pXZ, double* pYY, double* pYZ, double* pZZ) {
assert(pXX != NULL);
assert(pXY != NULL);
assert(pXZ != NULL);
assert(pYY != NULL);
assert(pYZ != NULL);
assert(pZZ != NULL);

const double aa = a * a;
const double x = pos.x();
const double y = pos.y();
const double z = pos.z();

switch (nType) {
case 0:  
{
*pXX = 4.0 * aa * x * x - 2.0 * a;
*pXY = 4.0 * aa * x * y;
*pXZ = 4.0 * aa * x * z;
*pYY = 4.0 * aa * y * y - 2.0 * a;
*pYZ = 4.0 * aa * y * z;
*pZZ = 4.0 * aa * z * z - 2.0 * a;
} break;
case 1:  
{
*pXX = 4.0 * aa * x * x * x - 6.0 * a * x;
*pXY = 4.0 * aa * x * x * y - 2.0 * a * y;
*pXZ = 4.0 * aa * x * x * z - 2.0 * a * z;
*pYY = 4.0 * aa * x * y * y - 2.0 * a * x;
*pYZ = 4.0 * aa * x * y * z;
*pZZ = 4.0 * aa * x * z * z - 2.0 * a * x;
} break;
case 2:  
{
*pXX = 4.0 * aa * x * x * y - 2.0 * a * y;
*pXY = 4.0 * aa * x * y * y - 2.0 * a * x;
*pXZ = 4.0 * aa * x * y * z;
*pYY = 4.0 * aa * y * y * y - 6.0 * a * y;
*pYZ = 4.0 * aa * y * y * z - 2.0 * a * z;
*pZZ = 4.0 * aa * y * z * z - 2.0 * a * y;
} break;
case 3:  
{
*pXX = 4.0 * aa * x * x * z - 2.0 * a * z;
*pXY = 4.0 * aa * x * y * z;
*pXZ = 4.0 * aa * x * z * z - 2.0 * a * x;
*pYY = 4.0 * aa * y * y * z - 2.0 * a * z;
*pYZ = 4.0 * aa * y * z * z - 2.0 * a * y;
*pZZ = 4.0 * aa * z * z * z - 6.0 * a * z;
} break;
case 4: {  
*pXX = 4.0 * aa * x * x * x * y - 6.0 * a * x * y;
*pXY = 4.0 * aa * x * x * y * y - 2.0 * a * x * x -
2.0 * a * y * y + 1.0;
*pXZ = 4.0 * aa * x * x * y * z - 2.0 * a * y * z;
*pYY = 4.0 * aa * x * y * y * y - 6.0 * a * x * y;
*pYZ = 4.0 * aa * x * y * y * z - 2.0 * a * x * z;
*pZZ = 4.0 * aa * x * y * z * z - 2.0 * a * x * y;
} break;
case 5:  
{
*pXX = 4.0 * aa * x * x * x * z - 6.0 * a * x * z;
*pXY = 4.0 * aa * x * x * y * z - 2.0 * a * y * z;
*pXZ = 4.0 * aa * x * x * z * z - 2.0 * a * x * x -
2.0 * a * z * z + 1.0;
*pYY = 4.0 * aa * x * y * y * z - 2.0 * a * x * z;
*pYZ = 4.0 * aa * x * y * z * z - 2.0 * a * x * y;
*pZZ = 4.0 * aa * x * z * z * z - 6.0 * a * x * z;
} break;
case 6:  
{
*pXX = 4.0 * aa * x * x * y * z - 2.0 * a * y * z;
*pXY = 4.0 * aa * x * y * y * z - 2.0 * a * x * z;
*pXZ = 4.0 * aa * x * y * z * z - 2.0 * a * x * y;
*pYY = 4.0 * aa * y * y * y * z - 6.0 * a * y * z;
*pYZ = 4.0 * aa * y * y * z * z - 2.0 * a * y * y -
2.0 * a * z * z + 1.0;
*pZZ = 4.0 * aa * y * z * z * z - 6.0 * a * y * z;
} break;
case 7:  
{
const double xx_xx =
4.0 * aa * x * x * x * x - 10.0 * a * x * x + 2.0;
const double xx_xy = 4.0 * aa * x * x * x * y - 4.0 * a * x * y;
const double xx_xz = 4.0 * aa * x * x * x * z - 4.0 * a * x * z;
const double xx_yy = 4.0 * aa * x * x * y * y - 2.0 * a * x * x;
const double xx_yz = 4.0 * aa * x * x * y * z;
const double xx_zz = 4.0 * aa * x * x * z * z - 2.0 * a * x * x;

const double yy_xx = 4.0 * aa * x * x * y * y - 2.0 * a * y * y;
const double yy_xy = 4.0 * aa * x * y * y * y - 4.0 * a * x * y;
const double yy_xz = 4.0 * aa * x * y * y * z;
const double yy_yy =
4.0 * aa * y * y * y * y - 10.0 * a * y * y + 2.0;
const double yy_yz = 4.0 * aa * y * y * y * z - 4.0 * a * y * z;
const double yy_zz = 4.0 * aa * y * y * z * z - 2.0 * a * y * y;

*pXX = 0.5 * (xx_xx - yy_xx);
*pXY = 0.5 * (xx_xy - yy_xy);
*pXZ = 0.5 * (xx_xz - yy_xz);
*pYY = 0.5 * (xx_yy - yy_yy);
*pYZ = 0.5 * (xx_yz - yy_yz);
*pZZ = 0.5 * (xx_zz - yy_zz);
} break;
case 8:  
{
const double xx_xx =
4.0 * aa * x * x * x * x - 10.0 * a * x * x + 2.0;
const double xx_xy = 4.0 * aa * x * x * x * y - 4.0 * a * x * y;
const double xx_xz = 4.0 * aa * x * x * x * z - 4.0 * a * x * z;
const double xx_yy = 4.0 * aa * x * x * y * y - 2.0 * a * x * x;
const double xx_yz = 4.0 * aa * x * x * y * z;
const double xx_zz = 4.0 * aa * x * x * z * z - 2.0 * a * x * x;

const double yy_xx = 4.0 * aa * x * x * y * y - 2.0 * a * y * y;
const double yy_xy = 4.0 * aa * x * y * y * y - 4.0 * a * x * y;
const double yy_xz = 4.0 * aa * x * y * y * z;
const double yy_yy =
4.0 * aa * y * y * y * y - 10.0 * a * y * y + 2.0;
const double yy_yz = 4.0 * aa * y * y * y * z - 4.0 * a * y * z;
const double yy_zz = 4.0 * aa * y * y * z * z - 2.0 * a * y * y;

const double zz_xx = 4.0 * aa * x * x * z * z - 2.0 * a * z * z;
const double zz_xy = 4.0 * aa * x * y * z * z;
const double zz_xz = 4.0 * aa * x * z * z * z - 4.0 * a * x * z;
const double zz_yy = 4.0 * aa * y * y * z * z - 2.0 * a * z * z;
const double zz_yz = 4.0 * aa * y * z * z * z - 4.0 * a * y * z;
const double zz_zz =
4.0 * aa * z * z * z * z - 10.0 * a * z * z + 2.0;

*pXX = INV_SQRT3 * (zz_xx - 0.5 * (xx_xx + yy_xx));
*pXY = INV_SQRT3 * (zz_xy - 0.5 * (xx_xy + yy_xy));
*pXZ = INV_SQRT3 * (zz_xz - 0.5 * (xx_xz + yy_xz));
*pYY = INV_SQRT3 * (zz_yy - 0.5 * (xx_yy + yy_yy));
*pYZ = INV_SQRT3 * (zz_yz - 0.5 * (xx_yz + yy_yz));
*pZZ = INV_SQRT3 * (zz_zz - 0.5 * (xx_zz + yy_zz));
} break;
default:
std::cout << "Basis Type is Wrong." << std::endl;
break;
}
}








































void DfCalcGridX::getAOs(const TlPosition& gridPosition,
std::vector<double>* pAO_values) {
const int numOfAOs = this->m_nNumOfAOs;

std::vector<index_type> AO_indeces(numOfAOs);
for (int i = 0; i < numOfAOs; ++i) {
AO_indeces[i] = i;
}

this->getAOs_core(gridPosition, AO_indeces, pAO_values);
}

void DfCalcGridX::getAOs_core(const TlPosition& gridPosition,
const std::vector<index_type>& AO_indeces,
std::vector<double>* pAO_values) {
assert(pAO_values != NULL);
const int numOfIndeces = AO_indeces.size();
pAO_values->resize(numOfIndeces);

for (int i = 0; i < numOfIndeces; ++i) {
const int AO_index = AO_indeces[i];

double AO = 0.0;

const TlPosition pos =
gridPosition - this->m_tlOrbInfo.getPosition(AO_index);
const double distance2 = pos.squareDistanceFrom();
const int basisType = this->m_tlOrbInfo.getBasisType(AO_index);

const int contraction = this->m_tlOrbInfo.getCgtoContraction(AO_index);
for (int PGTO = 0; PGTO < contraction; ++PGTO) {
const double alpha = this->m_tlOrbInfo.getExponent(AO_index, PGTO);
const double shoulder = alpha * distance2;

if (shoulder <= TOOBIG) {
const double g =
this->m_tlOrbInfo.getCoefficient(AO_index, PGTO) *
std::exp(-shoulder);
double coef = 0.0;
this->getPrefactor(basisType, pos, &coef);
AO += coef * g;
}
}

(*pAO_values)[i] = AO;
}
}

void DfCalcGridX::getDAOs(const TlPosition& gridPosition,
std::vector<double>* p_dAO_dx_values,
std::vector<double>* p_dAO_dy_values,
std::vector<double>* p_dAO_dz_values) {
const int numOfAOs = this->m_nNumOfAOs;

std::vector<index_type> AO_indeces(numOfAOs);
for (int i = 0; i < numOfAOs; ++i) {
AO_indeces[i] = i;
}

this->getDAOs_core(gridPosition, AO_indeces, p_dAO_dx_values,
p_dAO_dy_values, p_dAO_dz_values);
}

void DfCalcGridX::getDAOs_core(const TlPosition& gridPosition,
const std::vector<index_type>& AO_indeces,
std::vector<double>* p_dAO_dx_values,
std::vector<double>* p_dAO_dy_values,
std::vector<double>* p_dAO_dz_values) {
assert(p_dAO_dx_values != NULL);
assert(p_dAO_dy_values != NULL);
assert(p_dAO_dz_values != NULL);

const int numOfIndeces = AO_indeces.size();
p_dAO_dx_values->resize(numOfIndeces);
p_dAO_dy_values->resize(numOfIndeces);
p_dAO_dz_values->resize(numOfIndeces);

for (int i = 0; i < numOfIndeces; ++i) {
const int AO_index = AO_indeces[i];
double dAO_dx = 0.0;
double dAO_dy = 0.0;
double dAO_dz = 0.0;

const TlPosition pos =
gridPosition - this->m_tlOrbInfo.getPosition(AO_index);
const double distance2 = pos.squareDistanceFrom();
const int basisType = this->m_tlOrbInfo.getBasisType(AO_index);

const int contraction = this->m_tlOrbInfo.getCgtoContraction(AO_index);
for (int PGTO = 0; PGTO < contraction; ++PGTO) {
double pX = 0.0;
double pY = 0.0;
double pZ = 0.0;
const double alpha = this->m_tlOrbInfo.getExponent(AO_index, PGTO);
const double shoulder = alpha * distance2;

if (shoulder <= TOOBIG) {
const double g =
this->m_tlOrbInfo.getCoefficient(AO_index, PGTO) *
std::exp(-shoulder);
this->getPrefactorForDerivative(basisType, alpha, pos, &pX, &pY,
&pZ);
dAO_dx += pX * g;
dAO_dy += pY * g;
dAO_dz += pZ * g;
}
}

(*p_dAO_dx_values)[i] = dAO_dx;
(*p_dAO_dy_values)[i] = dAO_dy;
(*p_dAO_dz_values)[i] = dAO_dz;
}
}

void DfCalcGridX::getD2AOs(const TlPosition& gridPosition,
std::vector<double>* p_d2AO_dxdx_values,
std::vector<double>* p_d2AO_dxdy_values,
std::vector<double>* p_d2AO_dxdz_values,
std::vector<double>* p_d2AO_dydy_values,
std::vector<double>* p_d2AO_dydz_values,
std::vector<double>* p_d2AO_dzdz_values) {
const int numOfAOs = this->m_nNumOfAOs;

std::vector<index_type> AO_indeces(numOfAOs);
for (int i = 0; i < numOfAOs; ++i) {
AO_indeces[i] = i;
}

this->getD2AOs_core(gridPosition, AO_indeces, p_d2AO_dxdx_values,
p_d2AO_dxdy_values, p_d2AO_dxdz_values,
p_d2AO_dydy_values, p_d2AO_dydz_values,
p_d2AO_dzdz_values);
}

void DfCalcGridX::getD2AOs_core(const TlPosition& gridPosition,
const std::vector<index_type>& AO_indeces,
std::vector<double>* p_d2AO_dxdx_values,
std::vector<double>* p_d2AO_dxdy_values,
std::vector<double>* p_d2AO_dxdz_values,
std::vector<double>* p_d2AO_dydy_values,
std::vector<double>* p_d2AO_dydz_values,
std::vector<double>* p_d2AO_dzdz_values) {
assert(p_d2AO_dxdx_values != NULL);
assert(p_d2AO_dxdy_values != NULL);
assert(p_d2AO_dxdz_values != NULL);
assert(p_d2AO_dydy_values != NULL);
assert(p_d2AO_dydz_values != NULL);
assert(p_d2AO_dzdz_values != NULL);

const int numOfIndeces = AO_indeces.size();
p_d2AO_dxdx_values->resize(numOfIndeces);
p_d2AO_dxdy_values->resize(numOfIndeces);
p_d2AO_dxdz_values->resize(numOfIndeces);
p_d2AO_dydy_values->resize(numOfIndeces);
p_d2AO_dydz_values->resize(numOfIndeces);
p_d2AO_dzdz_values->resize(numOfIndeces);

for (int i = 0; i < numOfIndeces; ++i) {
const int AO_index = AO_indeces[i];
double d2AO_dxdx = 0.0;
double d2AO_dxdy = 0.0;
double d2AO_dxdz = 0.0;
double d2AO_dydy = 0.0;
double d2AO_dydz = 0.0;
double d2AO_dzdz = 0.0;

const TlPosition pos =
gridPosition - this->m_tlOrbInfo.getPosition(AO_index);
const double distance2 = pos.squareDistanceFrom();
const int basisType = this->m_tlOrbInfo.getBasisType(AO_index);

const int contraction = this->m_tlOrbInfo.getCgtoContraction(AO_index);
for (int PGTO = 0; PGTO < contraction; ++PGTO) {
double pXX = 0.0;
double pXY = 0.0;
double pXZ = 0.0;
double pYY = 0.0;
double pYZ = 0.0;
double pZZ = 0.0;
const double alpha = this->m_tlOrbInfo.getExponent(AO_index, PGTO);
const double shoulder = alpha * distance2;

if (shoulder <= TOOBIG) {
const double g =
this->m_tlOrbInfo.getCoefficient(AO_index, PGTO) *
std::exp(-shoulder);
this->getPrefactorForSecondDerivative(
basisType, alpha, pos, &pXX, &pXY, &pXZ, &pYY, &pYZ, &pZZ);
d2AO_dxdx += pXX * g;
d2AO_dxdy += pXY * g;
d2AO_dxdz += pXZ * g;
d2AO_dydy += pYY * g;
d2AO_dydz += pYZ * g;
d2AO_dzdz += pZZ * g;
}
}

(*p_d2AO_dxdx_values)[i] = d2AO_dxdx;
(*p_d2AO_dxdy_values)[i] = d2AO_dxdy;
(*p_d2AO_dxdz_values)[i] = d2AO_dxdz;
(*p_d2AO_dydy_values)[i] = d2AO_dydy;
(*p_d2AO_dydz_values)[i] = d2AO_dydz;
(*p_d2AO_dzdz_values)[i] = d2AO_dzdz;
}
}

void DfCalcGridX::getRhoAtGridPoint(const TlMatrixObject& PA,
const std::vector<double>& AO_values,
double* pRhoA) {
const int numOfIndeces = AO_values.size();

double rho = 0.0;
for (int i = 0; i < numOfIndeces; ++i) {
const double AO_i = AO_values[i];

for (int j = 0; j < i; ++j) {
rho += 2.0 * PA.get(i, j) * AO_i * AO_values[j];
}
rho += PA.get(i, i) * AO_i * AO_i;
}

*pRhoA = rho;
}

void DfCalcGridX::getRhoAtGridPoint(const TlMatrixObject& PA,
const std::vector<double>& row_AO_values,
const std::vector<double>& col_AO_values,
double* pRhoA) {
const int numOfRowIndeces = row_AO_values.size();
const int numOfColIndeces = col_AO_values.size();

double rho = 0.0;
for (int i = 0; i < numOfRowIndeces; ++i) {
const double AO_i = row_AO_values[i];

for (int j = 0; j < numOfColIndeces; ++j) {
rho += PA.get(i, j) * AO_i * col_AO_values[j];
}
}

*pRhoA = rho;
}

void DfCalcGridX::getGradRhoAtGridPoint(
const TlMatrixObject& PA, const std::vector<double>& AO_values,
const std::vector<double>& dAO_dx_values,
const std::vector<double>& dAO_dy_values,
const std::vector<double>& dAO_dz_values, double* pGradRhoAX,
double* pGradRhoAY, double* pGradRhoAZ) {
const int numOfIndeces = AO_values.size();
assert(numOfIndeces == static_cast<int>(dAO_dx_values.size()));
assert(numOfIndeces == static_cast<int>(dAO_dy_values.size()));
assert(numOfIndeces == static_cast<int>(dAO_dz_values.size()));

double grx = 0.0;
double gry = 0.0;
double grz = 0.0;
for (int i = 0; i < numOfIndeces; ++i) {
const double AO_i = AO_values[i];

for (int j = 0; j < numOfIndeces; ++j) {
const double Pij = PA.get(i, j);

grx += Pij * AO_i * dAO_dx_values[j];
gry += Pij * AO_i * dAO_dy_values[j];
grz += Pij * AO_i * dAO_dz_values[j];
}
}

*pGradRhoAX = 2.0 * grx;
*pGradRhoAY = 2.0 * gry;
*pGradRhoAZ = 2.0 * grz;
}

void DfCalcGridX::getGradRhoAtGridPoint(
const TlMatrixObject& PA, const std::vector<double>& row_AO_values,
const std::vector<double>& row_dAO_dx_values,
const std::vector<double>& row_dAO_dy_values,
const std::vector<double>& row_dAO_dz_values,
const std::vector<double>& col_AO_values,
const std::vector<double>& col_dAO_dx_values,
const std::vector<double>& col_dAO_dy_values,
const std::vector<double>& col_dAO_dz_values, double* pGradRhoAX,
double* pGradRhoAY, double* pGradRhoAZ) {
const int numOfRowIndeces = row_AO_values.size();
assert(numOfRowIndeces == static_cast<int>(row_dAO_dx_values.size()));
assert(numOfRowIndeces == static_cast<int>(row_dAO_dy_values.size()));
assert(numOfRowIndeces == static_cast<int>(row_dAO_dz_values.size()));
const int numOfColIndeces = col_AO_values.size();
assert(numOfColIndeces == static_cast<int>(col_dAO_dx_values.size()));
assert(numOfColIndeces == static_cast<int>(col_dAO_dy_values.size()));
assert(numOfColIndeces == static_cast<int>(col_dAO_dz_values.size()));

double grx = 0.0;
double gry = 0.0;
double grz = 0.0;
for (int i = 0; i < numOfRowIndeces; ++i) {
const double AO_i = row_AO_values[i];
const double dAO_dx_i = row_dAO_dx_values[i];
const double dAO_dy_i = row_dAO_dy_values[i];
const double dAO_dz_i = row_dAO_dz_values[i];

for (int j = 0; j < numOfColIndeces; ++j) {
const double AO_j = col_AO_values[j];
const double dAO_dx_j = col_dAO_dx_values[j];
const double dAO_dy_j = col_dAO_dy_values[j];
const double dAO_dz_j = col_dAO_dz_values[j];

const double Pij = PA.get(i, j);

grx += Pij * AO_i * dAO_dx_j;
gry += Pij * AO_i * dAO_dy_j;
grz += Pij * AO_i * dAO_dz_j;

grx += Pij * AO_j * dAO_dx_i;
gry += Pij * AO_j * dAO_dy_i;
grz += Pij * AO_j * dAO_dz_i;
}
}

*pGradRhoAX = grx;
*pGradRhoAY = gry;
*pGradRhoAZ = grz;
}


void DfCalcGridX::buildFock(std::vector<WFGrid>::const_iterator pBegin,
std::vector<WFGrid>::const_iterator pEnd,
const double coef, const double cutoffValue,
TlMatrixObject* pF) {
assert(pF != NULL);

for (std::vector<WFGrid>::const_iterator p = pBegin; p != pEnd; ++p) {
const int u_index = p->index;
const double u_value = p->value;
const double tmp1A = coef * u_value;

pF->add(u_index, u_index, tmp1A * u_value);

std::vector<WFGrid>::const_iterator qEnd =
std::upper_bound(p + 1, pEnd, WFGrid(0, cutoffValue / tmp1A),
WFGrid_sort_functional());
for (std::vector<WFGrid>::const_iterator q = p + 1; q != qEnd; ++q) {
const int v_index = q->index;
const double v_value = q->value;

pF->add(u_index, v_index, tmp1A * v_value);
}
}
}

void DfCalcGridX::buildFock(std::vector<WFGrid>::const_iterator pBegin,
std::vector<WFGrid>::const_iterator pEnd,
std::vector<WFGrid>::const_iterator qBegin,
std::vector<WFGrid>::const_iterator qEnd,
const double coef, const double cutoffValue,
TlMatrixObject* pF) {
assert(pF != NULL);

for (std::vector<WFGrid>::const_iterator p = pBegin; p != pEnd; ++p) {
const int u_index = p->index;
const double u_value = p->value;
const double tmp2AX = coef * u_value;

for (std::vector<WFGrid>::const_iterator q = qBegin; q != qEnd; ++q) {
const int v_index = q->index;
const double v_value = q->value;

if (u_index >= v_index) {
pF->add(u_index, v_index, tmp2AX * v_value);
}
}
}
}

void DfCalcGridX::gridDensity(const TlDenseSymmetricMatrix_Lapack& PA,
const TlPosition& gridPosition, double* pRhoA) {
std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

this->getRhoAtGridPoint(PA, AO_values, pRhoA);
}

TlDenseGeneralMatrix_Lapack DfCalcGridX::selectGridMatrixByAtom(
const TlDenseGeneralMatrix_Lapack& globalGridMat, const int atomIndex) {
const index_type numOfGrids = globalGridMat.getNumOfRows();
const double atomIndex_real = double(atomIndex);
std::vector<index_type> finder;
for (index_type i = 0; i < numOfGrids; ++i) {
const double validation =
std::fabs(globalGridMat.get(i, GM_ATOM_INDEX) - atomIndex_real);
if (validation < 1.0E-5) {
finder.push_back(i);
}
}

const index_type numOfFinds = finder.size();
const index_type numOfCols = globalGridMat.getNumOfCols();
TlDenseGeneralMatrix_Lapack answer(numOfFinds, numOfCols);
for (index_type i = 0; i < numOfFinds; ++i) {
const index_type globalRow = finder[i];
for (index_type col = 0; col < numOfCols; ++col) {
const double value = globalGridMat.get(globalRow, col);
answer.set(i, col, value);
}
}

return answer;
}

TlDenseGeneralMatrix_Lapack DfCalcGridX::energyGradient(
const TlDenseSymmetricMatrix_Lapack& P_A, DfFunctional_LDA* pFunctional) {
this->log_.info("pure DFT XC energy (LDA) gradient by grid method");
const int numOfAOs = this->m_nNumOfAOs;
const int numOfAtoms = this->m_nNumOfAtoms;

TlDenseGeneralMatrix_Lapack gammaX(numOfAOs, numOfAOs);
TlDenseGeneralMatrix_Lapack gammaY(numOfAOs, numOfAOs);
TlDenseGeneralMatrix_Lapack gammaZ(numOfAOs, numOfAOs);

TlDenseGeneralMatrix_Lapack Fxc_f(numOfAtoms, 3);  
TlDenseGeneralMatrix_Lapack Fxc_w(numOfAtoms, 3);  
const double ene_xc = this->energyGradient_part(P_A, pFunctional, 0,
numOfAtoms, &Fxc_f, &Fxc_w);

this->log_.info(TlUtils::format("XC ene = % 16.10f", ene_xc));

Fxc_f *= 2.0;  

if (this->isSaveGrad_) {
Fxc_w.save("Fxc_w.mat");
Fxc_f.save("Fxc_f.mat");
}

return Fxc_w + Fxc_f;
}

double DfCalcGridX::energyGradient_part(
const TlDenseSymmetricMatrix_Lapack& P_A, DfFunctional_LDA* pFunctional,
const int startAtomIndex, const int endAtomIndex,
TlDenseGeneralMatrix_Lapack* pFxc_f, TlDenseGeneralMatrix_Lapack* pFxc_w) {
assert(pFxc_w != NULL);
assert(pFxc_f != NULL);

const int numOfAOs = this->m_nNumOfAOs;
const int numOfAtoms = this->m_nNumOfAtoms;
const double densityCutOffValue = this->m_densityCutOffValueA;
const TlOrbitalInfo orbitalInfo((*this->pPdfParam_)["coordinates"],
(*this->pPdfParam_)["basis_set"]);
DfGenerateGrid dfGenGrid(this->pPdfParam_);
double ene_xc = 0.0;
for (int atomIndexA = startAtomIndex; atomIndexA < endAtomIndex;
++atomIndexA) {
std::vector<TlPosition> grids;
std::vector<double> singleCenterWeights;
std::vector<double> partitioningWeights;
dfGenGrid.getGrids(atomIndexA, &grids, &singleCenterWeights,
&partitioningWeights);

const int numOfGrids = grids.size();
assert(static_cast<int>(singleCenterWeights.size()) == numOfGrids);
assert(static_cast<int>(partitioningWeights.size()) == numOfGrids);

for (int gridIndex = 0; gridIndex < numOfGrids; ++gridIndex) {
const TlPosition gridPosition = grids[gridIndex];
const double singleCenterWeight = singleCenterWeights[gridIndex];
const double partitioningWeight = partitioningWeights[gridIndex];

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

std::vector<double> dAO_dx_values;
std::vector<double> dAO_dy_values;
std::vector<double> dAO_dz_values;
this->getDAOs(gridPosition, &dAO_dx_values, &dAO_dy_values,
&dAO_dz_values);

double rhoA = 0.0;
this->getRhoAtGridPoint(P_A, AO_values, &rhoA);
const double& rhoB = rhoA;

if (rhoA > densityCutOffValue) {
const double w = partitioningWeight * singleCenterWeight;

const double ene = pFunctional->getFunctional(rhoA, rhoA);
ene_xc += w * ene;

double dF_dRhoA = 0.0;
double dF_dRhoB = 0.0;
pFunctional->getDerivativeFunctional(rhoA, rhoB, &dF_dRhoA,
&dF_dRhoB);

for (int i = 0; i < numOfAOs; ++i) {
const double dx = dAO_dx_values[i];
const double dy = dAO_dy_values[i];
const double dz = dAO_dz_values[i];

double rao = 0.0;
for (int j = 0; j < numOfAOs; ++j) {
const double Pij = P_A.get(i, j);
rao += Pij * AO_values[j];
}

double gradx = dF_dRhoA * dx * rao;
double grady = dF_dRhoA * dy * rao;
double gradz = dF_dRhoA * dz * rao;

gradx *= -2.0 * w;
grady *= -2.0 * w;
gradz *= -2.0 * w;

const int atomIndexB = orbitalInfo.getAtomIndex(i);
pFxc_f->add(atomIndexB, 0, gradx);
pFxc_f->add(atomIndexA, 0, -gradx);
pFxc_f->add(atomIndexB, 1, grady);
pFxc_f->add(atomIndexA, 1, -grady);
pFxc_f->add(atomIndexB, 2, gradz);
pFxc_f->add(atomIndexA, 2, -gradz);
}

for (int atomIndexB = 0; atomIndexB < numOfAtoms;
++atomIndexB) {
if (atomIndexA != atomIndexB) {
const TlDenseVector_Lapack nablaB_omegaA =
dfGenGrid.JGP_nablaB_omegaA(atomIndexA, atomIndexB,
gridPosition);
const TlDenseVector_Lapack E_nablaB_omegaAi =
nablaB_omegaA * singleCenterWeight * ene;

for (int i = 0; i < 3; ++i) {
pFxc_w->add(atomIndexB, i, E_nablaB_omegaAi.get(i));
pFxc_w->add(
atomIndexA, i,
-E_nablaB_omegaAi.get(i));  
}
}
}
}
}
}

return ene_xc;
}

TlDenseGeneralMatrix_Lapack DfCalcGridX::energyGradient(
const TlDenseSymmetricMatrix_Lapack& P_A, DfFunctional_GGA* pFunctional) {
this->log_.info("pure DFT XC energy (GGA) gradient by grid method");
const int numOfAOs = this->m_nNumOfAOs;
const int numOfAtoms = this->m_nNumOfAtoms;

TlDenseGeneralMatrix_Lapack gammaX(numOfAOs, numOfAOs);
TlDenseGeneralMatrix_Lapack gammaY(numOfAOs, numOfAOs);
TlDenseGeneralMatrix_Lapack gammaZ(numOfAOs, numOfAOs);

TlDenseGeneralMatrix_Lapack Fxc_w(numOfAtoms, 3);  
TlDenseGeneralMatrix_Lapack Fxc_f(numOfAtoms, 3);  
const double ene_xc = this->energyGradient_part(P_A, pFunctional, 0,
numOfAtoms, &Fxc_f, &Fxc_w);

this->log_.info(TlUtils::format("XC ene = % 16.10f", ene_xc));

Fxc_f *= 2.0;  

if (this->isSaveGrad_) {
Fxc_w.save("Fxc_w.mat");
Fxc_f.save("Fxc_f.mat");
}

return Fxc_w + Fxc_f;  
}

double DfCalcGridX::energyGradient_part(
const TlDenseSymmetricMatrix_Lapack& P_A, DfFunctional_GGA* pFunctional,
const int startAtomIndex, const int endAtomIndex,
TlDenseGeneralMatrix_Lapack* pFxc_f, TlDenseGeneralMatrix_Lapack* pFxc_w) {
assert(pFxc_w != NULL);
assert(pFxc_f != NULL);

const int numOfAOs = this->m_nNumOfAOs;
const int numOfAtoms = this->m_nNumOfAtoms;
const double densityCutOffValue = this->m_densityCutOffValueA;
const TlOrbitalInfo orbitalInfo((*this->pPdfParam_)["coordinates"],
(*this->pPdfParam_)["basis_set"]);
DfGenerateGrid dfGenGrid(this->pPdfParam_);
double ene_xc = 0.0;
for (int atomIndexA = startAtomIndex; atomIndexA < endAtomIndex;
++atomIndexA) {
std::vector<TlPosition> grids;
std::vector<double> singleCenterWeights;
std::vector<double> partitioningWeights;
dfGenGrid.getGrids(atomIndexA, &grids, &singleCenterWeights,
&partitioningWeights);

const int numOfGrids = grids.size();
assert(static_cast<int>(singleCenterWeights.size()) == numOfGrids);
assert(static_cast<int>(partitioningWeights.size()) == numOfGrids);

for (int gridIndex = 0; gridIndex < numOfGrids; ++gridIndex) {
const TlPosition gridPosition = grids[gridIndex];
const double singleCenterWeight = singleCenterWeights[gridIndex];
const double partitioningWeight = partitioningWeights[gridIndex];

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

std::vector<double> dAO_dx_values;
std::vector<double> dAO_dy_values;
std::vector<double> dAO_dz_values;
this->getDAOs(gridPosition, &dAO_dx_values, &dAO_dy_values,
&dAO_dz_values);

std::vector<double> d2AO_dxdx_values;
std::vector<double> d2AO_dxdy_values;
std::vector<double> d2AO_dxdz_values;
std::vector<double> d2AO_dydy_values;
std::vector<double> d2AO_dydz_values;
std::vector<double> d2AO_dzdz_values;
this->getD2AOs(gridPosition, &d2AO_dxdx_values, &d2AO_dxdy_values,
&d2AO_dxdz_values, &d2AO_dydy_values,
&d2AO_dydz_values, &d2AO_dzdz_values);

double rhoA = 0.0;
this->getRhoAtGridPoint(P_A, AO_values, &rhoA);
const double& rhoB = rhoA;

double gradRhoAX = 0.0;
double gradRhoAY = 0.0;
double gradRhoAZ = 0.0;
this->getGradRhoAtGridPoint(P_A, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values,
&gradRhoAX, &gradRhoAY, &gradRhoAZ);
const double& gradRhoBX = gradRhoAX;
const double& gradRhoBY = gradRhoAY;
const double& gradRhoBZ = gradRhoAZ;

if (rhoA > densityCutOffValue) {
const double w = partitioningWeight * singleCenterWeight;
const double gammaAA = gradRhoAX * gradRhoAX +
gradRhoAY * gradRhoAY +
gradRhoAZ * gradRhoAZ;
const double& gammaAB = gammaAA;
const double& gammaBB = gammaAA;

const double ene = pFunctional->getFunctional(
rhoA, rhoB, gammaAA, gammaAB, gammaBB);
ene_xc += w * ene;

double dF_dRhoA = 0.0;
double dF_dRhoB = 0.0;
double dF_dGammaAA = 0.0;
double dF_dGammaAB = 0.0;
double dF_dGammaBB = 0.0;
pFunctional->getDerivativeFunctional(
rhoA, rhoB, gammaAA, gammaAB, gammaBB, &dF_dRhoA, &dF_dRhoB,
&dF_dGammaAA, &dF_dGammaAB, &dF_dGammaBB);

const double coef_ax =
2.0 * dF_dGammaAA * gradRhoAX + dF_dGammaAB * gradRhoBX;
const double coef_ay =
2.0 * dF_dGammaAA * gradRhoAY + dF_dGammaAB * gradRhoBY;
const double coef_az =
2.0 * dF_dGammaAA * gradRhoAZ + dF_dGammaAB * gradRhoBZ;

for (int i = 0; i < numOfAOs; ++i) {
const double dx = dAO_dx_values[i];
const double dy = dAO_dy_values[i];
const double dz = dAO_dz_values[i];
const double dxdx = d2AO_dxdx_values[i];
const double dxdy = d2AO_dxdy_values[i];
const double dxdz = d2AO_dxdz_values[i];
const double dydy = d2AO_dydy_values[i];
const double dydz = d2AO_dydz_values[i];
const double dzdz = d2AO_dzdz_values[i];

double rao = 0.0;
double rdx = 0.0;
double rdy = 0.0;
double rdz = 0.0;
for (int j = 0; j < numOfAOs; ++j) {
const double Pij = P_A.get(i, j);
rao += Pij * AO_values[j];
rdx += Pij * dAO_dx_values[j];
rdy += Pij * dAO_dy_values[j];
rdz += Pij * dAO_dz_values[j];
}

double gradx = dF_dRhoA * dx * rao;
double grady = dF_dRhoA * dy * rao;
double gradz = dF_dRhoA * dz * rao;

gradx += coef_ax * (dxdx * rao + dx * rdx) +
coef_ay * (dxdy * rao + dx * rdy) +
coef_az * (dxdz * rao + dx * rdz);
grady += coef_ax * (dxdy * rao + dy * rdx) +
coef_ay * (dydy * rao + dy * rdy) +
coef_az * (dydz * rao + dy * rdz);
gradz += coef_ax * (dxdz * rao + dz * rdx) +
coef_ay * (dydz * rao + dz * rdy) +
coef_az * (dzdz * rao + dz * rdz);

gradx *= -2.0 * w;
grady *= -2.0 * w;
gradz *= -2.0 * w;

const int atomIndexB = orbitalInfo.getAtomIndex(i);
pFxc_f->add(atomIndexB, 0, gradx);
pFxc_f->add(atomIndexA, 0, -gradx);
pFxc_f->add(atomIndexB, 1, grady);
pFxc_f->add(atomIndexA, 1, -grady);
pFxc_f->add(atomIndexB, 2, gradz);
pFxc_f->add(atomIndexA, 2, -gradz);
}

for (int atomIndexB = 0; atomIndexB < numOfAtoms;
++atomIndexB) {
if (atomIndexA != atomIndexB) {
const TlDenseVector_Lapack nablaB_omegaA =
dfGenGrid.JGP_nablaB_omegaA(atomIndexA, atomIndexB,
gridPosition);
const TlDenseVector_Lapack E_nablaB_omegaAi =
nablaB_omegaA * singleCenterWeight * ene;

for (int i = 0; i < 3; ++i) {
pFxc_w->add(atomIndexB, i, E_nablaB_omegaAi.get(i));
pFxc_w->add(
atomIndexA, i,
-E_nablaB_omegaAi.get(i));  
}
}
}
}
}
}

return ene_xc;
}














































void DfCalcGridX::calcRho_LDA(const TlDenseSymmetricMatrix_Lapack& P_A) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration - 1);
assert(gridMat.getNumOfCols() == GM_LDA_RHO_ALPHA + 1);

this->calcRho_LDA_part(P_A, &gridMat);

DfObject::saveGridMatrix(this->m_nIteration, gridMat);
}

void DfCalcGridX::calcRho_LDA(const TlDenseSymmetricMatrix_Lapack& P_A,
const TlDenseSymmetricMatrix_Lapack& P_B) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration - 1);
assert(gridMat.getNumOfCols() == GM_LDA_RHO_BETA + 1);

this->calcRho_LDA_part(P_A, P_B, &gridMat);

DfObject::saveGridMatrix(this->m_nIteration, gridMat);
}

void DfCalcGridX::calcRho_GGA(const TlDenseSymmetricMatrix_Lapack& P_A) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration - 1);
assert(gridMat.getNumOfCols() == GM_GGA_GRAD_RHO_Z_ALPHA + 1);

this->calcRho_GGA_part(P_A, &gridMat);

DfObject::saveGridMatrix(this->m_nIteration, gridMat);
}

void DfCalcGridX::calcRho_GGA(const TlDenseSymmetricMatrix_Lapack& P_A,
const TlDenseSymmetricMatrix_Lapack& P_B) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration - 1);
assert(gridMat.getNumOfCols() == GM_GGA_GRAD_RHO_Z_BETA + 1);

this->calcRho_GGA_part(P_A, P_B, &gridMat);

DfObject::saveGridMatrix(this->m_nIteration, gridMat);
}

void DfCalcGridX::calcRho_LDA_part(const TlDenseSymmetricMatrix_Lapack& P_A,
TlDenseGeneralMatrix_Lapack* pGridMat) {
assert(pGridMat != NULL);

if ((this->m_nIteration == 1) || (this->m_bIsUpdateXC != true)) {
TlDenseGeneralMatrix_Lapack zero(pGridMat->getNumOfRows(), 1);
pGridMat->block(0, GM_LDA_RHO_ALPHA, zero);
}

const index_type numOfGrids = pGridMat->getNumOfRows();
#pragma omp parallel for schedule(runtime)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(pGridMat->get(grid, GM_X),
pGridMat->get(grid, GM_Y),
pGridMat->get(grid, GM_Z));

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

double rhoA = 0.0;
this->getRhoAtGridPoint(P_A, AO_values, &rhoA);

assert((0 <= grid) && (grid < pGridMat->getNumOfRows()));
pGridMat->add(grid, GM_LDA_RHO_ALPHA, rhoA);
}
}

void DfCalcGridX::calcRho_LDA_part(const TlDenseSymmetricMatrix_Lapack& P_A,
const TlDenseSymmetricMatrix_Lapack& P_B,
TlDenseGeneralMatrix_Lapack* pGridMat) {
if ((this->m_nIteration == 1) || (this->m_bIsUpdateXC != true)) {
TlDenseGeneralMatrix_Lapack zero(pGridMat->getNumOfRows(), 2);
pGridMat->block(0, GM_LDA_RHO_ALPHA, zero);
}

const index_type numOfGrids = pGridMat->getNumOfRows();
#pragma omp parallel for schedule(runtime)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(pGridMat->get(grid, GM_X),
pGridMat->get(grid, GM_Y),
pGridMat->get(grid, GM_Z));

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

double rhoA = 0.0;
double rhoB = 0.0;
this->getRhoAtGridPoint(P_A, AO_values, &rhoA);
this->getRhoAtGridPoint(P_B, AO_values, &rhoB);

pGridMat->add(grid, GM_LDA_RHO_ALPHA, rhoA);
pGridMat->add(grid, GM_LDA_RHO_BETA, rhoB);
}
}

void DfCalcGridX::calcRho_GGA_part(const TlDenseSymmetricMatrix_Lapack& P_A,
TlDenseGeneralMatrix_Lapack* pGridMat) {
if ((this->m_nIteration == 1) || (this->m_bIsUpdateXC != true)) {
TlDenseGeneralMatrix_Lapack zero(pGridMat->getNumOfRows(), 4);
pGridMat->block(0, GM_GGA_RHO_ALPHA, zero);
}

const index_type numOfGrids = pGridMat->getNumOfRows();
#pragma omp parallel for schedule(runtime)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(pGridMat->get(grid, GM_X),
pGridMat->get(grid, GM_Y),
pGridMat->get(grid, GM_Z));

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);
std::vector<double> dAO_dx_values;
std::vector<double> dAO_dy_values;
std::vector<double> dAO_dz_values;
this->getDAOs(gridPosition, &dAO_dx_values, &dAO_dy_values,
&dAO_dz_values);

double rhoA = 0.0;
double gradRhoXA = 0.0;
double gradRhoYA = 0.0;
double gradRhoZA = 0.0;
this->getRhoAtGridPoint(P_A, AO_values, &rhoA);
this->getGradRhoAtGridPoint(P_A, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values, &gradRhoXA,
&gradRhoYA, &gradRhoZA);

pGridMat->add(grid, GM_GGA_RHO_ALPHA, rhoA);
pGridMat->add(grid, GM_GGA_GRAD_RHO_X_ALPHA, gradRhoXA);
pGridMat->add(grid, GM_GGA_GRAD_RHO_Y_ALPHA, gradRhoYA);
pGridMat->add(grid, GM_GGA_GRAD_RHO_Z_ALPHA, gradRhoZA);
}
}

void DfCalcGridX::calcRho_GGA_part(const TlDenseSymmetricMatrix_Lapack& P_A,
const TlDenseSymmetricMatrix_Lapack& P_B,
TlDenseGeneralMatrix_Lapack* pGridMat) {
if ((this->m_nIteration == 1) || (this->m_bIsUpdateXC != true)) {
TlDenseGeneralMatrix_Lapack zero(pGridMat->getNumOfRows(), 8);
pGridMat->block(0, GM_GGA_RHO_ALPHA, zero);
}

const index_type numOfGrids = pGridMat->getNumOfRows();
#pragma omp parallel for schedule(runtime)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(pGridMat->get(grid, GM_X),
pGridMat->get(grid, GM_Y),
pGridMat->get(grid, GM_Z));

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);
std::vector<double> dAO_dx_values;
std::vector<double> dAO_dy_values;
std::vector<double> dAO_dz_values;
this->getDAOs(gridPosition, &dAO_dx_values, &dAO_dy_values,
&dAO_dz_values);

double rhoA = 0.0;
double gradRhoXA = 0.0;
double gradRhoYA = 0.0;
double gradRhoZA = 0.0;
double rhoB = 0.0;
double gradRhoXB = 0.0;
double gradRhoYB = 0.0;
double gradRhoZB = 0.0;
this->getRhoAtGridPoint(P_A, AO_values, &rhoA);
this->getGradRhoAtGridPoint(P_A, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values, &gradRhoXA,
&gradRhoYA, &gradRhoZA);
this->getRhoAtGridPoint(P_B, AO_values, &rhoB);
this->getGradRhoAtGridPoint(P_B, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values, &gradRhoXB,
&gradRhoYB, &gradRhoZB);

pGridMat->add(grid, GM_GGA_RHO_ALPHA, rhoA);
pGridMat->add(grid, GM_GGA_GRAD_RHO_X_ALPHA, gradRhoXA);
pGridMat->add(grid, GM_GGA_GRAD_RHO_Y_ALPHA, gradRhoYA);
pGridMat->add(grid, GM_GGA_GRAD_RHO_Z_ALPHA, gradRhoZA);
pGridMat->add(grid, GM_GGA_RHO_BETA, rhoB);
pGridMat->add(grid, GM_GGA_GRAD_RHO_X_BETA, gradRhoXB);
pGridMat->add(grid, GM_GGA_GRAD_RHO_Y_BETA, gradRhoYB);
pGridMat->add(grid, GM_GGA_GRAD_RHO_Z_BETA, gradRhoZB);
}
}

double DfCalcGridX::buildVxc(DfFunctional_LDA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration);
double energy = this->buildVxc(gridMat, pFunctional, pF_A);
return energy;
}

double DfCalcGridX::buildVxc(DfFunctional_LDA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A,
TlDenseSymmetricMatrix_Lapack* pF_B) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration);
double energy = this->buildVxc(gridMat, pFunctional, pF_A, pF_B);
return energy;
}

double DfCalcGridX::buildVxc(DfFunctional_GGA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration);
double energy = this->buildVxc(gridMat, pFunctional, pF_A);
return energy;
}

double DfCalcGridX::buildVxc(DfFunctional_GGA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A,
TlDenseSymmetricMatrix_Lapack* pF_B) {
TlDenseGeneralMatrix_Lapack gridMat =
DfObject::getGridMatrix<TlDenseGeneralMatrix_Lapack>(
this->m_nIteration);
double energy = this->buildVxc(gridMat, pFunctional, pF_A, pF_B);
return energy;
}

double DfCalcGridX::buildVxc(const TlDenseGeneralMatrix_Lapack& gridMatrix,
DfFunctional_LDA* pFunctional,
TlMatrixObject* pF_A) {
double energy = 0.0;
const double densityCutOffValue = this->m_densityCutOffValueA;
const index_type numOfGrids = gridMatrix.getNumOfRows();

#pragma omp parallel for schedule(runtime) reduction(+ : energy)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(gridMatrix.get(grid, GM_X),
gridMatrix.get(grid, GM_Y),
gridMatrix.get(grid, GM_Z));
const double weight = gridMatrix.get(grid, GM_WEIGHT);

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

const double rhoA = gridMatrix.get(grid, GM_LDA_RHO_ALPHA);
double roundF_roundRhoA = 0.0;

if (rhoA > densityCutOffValue) {
pFunctional->getDerivativeFunctional(rhoA, &roundF_roundRhoA);

this->build_XC_Matrix(roundF_roundRhoA, AO_values, pFunctional,
weight, pF_A);
energy += weight * pFunctional->getFunctional(rhoA, rhoA);
}
}

return energy;
}

double DfCalcGridX::buildVxc(const TlDenseGeneralMatrix_Lapack& gridMatrix,
DfFunctional_LDA* pFunctional,
TlMatrixObject* pF_A, TlMatrixObject* pF_B) {
double energy = 0.0;
const double densityCutOffValue =
std::min(this->m_densityCutOffValueA, this->m_densityCutOffValueB);
const index_type numOfGrids = gridMatrix.getNumOfRows();

#pragma omp parallel for schedule(runtime) reduction(+ : energy)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(gridMatrix.get(grid, GM_X),
gridMatrix.get(grid, GM_Y),
gridMatrix.get(grid, GM_Z));
const double weight = gridMatrix.get(grid, GM_WEIGHT);

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

const double rhoA = gridMatrix.get(grid, GM_LDA_RHO_ALPHA);
const double rhoB = gridMatrix.get(grid, GM_LDA_RHO_BETA);
double roundF_roundRhoA = 0.0;
double roundF_roundRhoB = 0.0;

if ((rhoA > densityCutOffValue) || (rhoB > densityCutOffValue)) {
pFunctional->getDerivativeFunctional(rhoA, rhoB, &roundF_roundRhoA,
&roundF_roundRhoB);

this->build_XC_Matrix(roundF_roundRhoA, AO_values, pFunctional,
weight, pF_A);
this->build_XC_Matrix(roundF_roundRhoB, AO_values, pFunctional,
weight, pF_B);
energy += weight * pFunctional->getFunctional(rhoA, rhoB);
}
}

return energy;
}

double DfCalcGridX::buildVxc(const TlDenseGeneralMatrix_Lapack& gridMatrix,
DfFunctional_GGA* pFunctional,
TlMatrixObject* pF_A) {
double energy = 0.0;
const double densityCutOffValue = this->m_densityCutOffValueA;
const index_type numOfGrids = gridMatrix.getNumOfRows();

#pragma omp parallel for schedule(runtime) reduction(+ : energy)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(gridMatrix.get(grid, GM_X),
gridMatrix.get(grid, GM_Y),
gridMatrix.get(grid, GM_Z));
const double weight = gridMatrix.get(grid, GM_WEIGHT);

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

std::vector<double> dAO_dx_values;
std::vector<double> dAO_dy_values;
std::vector<double> dAO_dz_values;
this->getDAOs(gridPosition, &dAO_dx_values, &dAO_dy_values,
&dAO_dz_values);

double rhoA = gridMatrix.get(grid, GM_GGA_RHO_ALPHA);
double gradRhoXA = gridMatrix.get(grid, GM_GGA_GRAD_RHO_X_ALPHA);
double gradRhoYA = gridMatrix.get(grid, GM_GGA_GRAD_RHO_Y_ALPHA);
double gradRhoZA = gridMatrix.get(grid, GM_GGA_GRAD_RHO_Z_ALPHA);

if (rhoA > densityCutOffValue) {
const double gammaAA = gradRhoXA * gradRhoXA +
gradRhoYA * gradRhoYA +
gradRhoZA * gradRhoZA;
double roundF_roundRhoA;
double roundF_roundGammaAA;
double roundF_roundGammaAB;
pFunctional->getDerivativeFunctional(
rhoA, gammaAA, &roundF_roundRhoA, &roundF_roundGammaAA,
&roundF_roundGammaAB);

this->build_XC_Matrix(
roundF_roundRhoA, roundF_roundGammaAA, roundF_roundGammaAB,
gradRhoXA, gradRhoYA, gradRhoZA, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values, pFunctional, weight, pF_A);
energy += weight * pFunctional->getFunctional(rhoA, gammaAA);
}
}

return energy;
}

double DfCalcGridX::buildVxc(const TlDenseGeneralMatrix_Lapack& gridMatrix,
DfFunctional_GGA* pFunctional,
TlMatrixObject* pF_A, TlMatrixObject* pF_B) {
double energy = 0.0;
const double densityCutOffValue =
std::min(this->m_densityCutOffValueA, this->m_densityCutOffValueB);
const index_type numOfGrids = gridMatrix.getNumOfRows();

#pragma omp parallel for schedule(runtime) reduction(+ : energy)
for (index_type grid = 0; grid < numOfGrids; ++grid) {
const TlPosition gridPosition(gridMatrix.get(grid, GM_X),
gridMatrix.get(grid, GM_Y),
gridMatrix.get(grid, GM_Z));
const double weight = gridMatrix.get(grid, GM_WEIGHT);

std::vector<double> AO_values;
this->getAOs(gridPosition, &AO_values);

std::vector<double> dAO_dx_values;
std::vector<double> dAO_dy_values;
std::vector<double> dAO_dz_values;
this->getDAOs(gridPosition, &dAO_dx_values, &dAO_dy_values,
&dAO_dz_values);

double rhoA = gridMatrix.get(grid, GM_GGA_RHO_ALPHA);
double gradRhoXA = gridMatrix.get(grid, GM_GGA_GRAD_RHO_X_ALPHA);
double gradRhoYA = gridMatrix.get(grid, GM_GGA_GRAD_RHO_Y_ALPHA);
double gradRhoZA = gridMatrix.get(grid, GM_GGA_GRAD_RHO_Z_ALPHA);
double rhoB = gridMatrix.get(grid, GM_GGA_RHO_BETA);
double gradRhoXB = gridMatrix.get(grid, GM_GGA_GRAD_RHO_X_BETA);
double gradRhoYB = gridMatrix.get(grid, GM_GGA_GRAD_RHO_Y_BETA);
double gradRhoZB = gridMatrix.get(grid, GM_GGA_GRAD_RHO_Z_BETA);

if ((rhoA > densityCutOffValue) || ((rhoB > densityCutOffValue))) {
const double gammaAA = gradRhoXA * gradRhoXA +
gradRhoYA * gradRhoYA +
gradRhoZA * gradRhoZA;
const double gammaAB = gradRhoXA * gradRhoXB +
gradRhoYA * gradRhoYB +
gradRhoZA * gradRhoZB;
const double gammaBB = gradRhoXB * gradRhoXB +
gradRhoYB * gradRhoYB +
gradRhoZB * gradRhoZB;

double roundF_roundRhoA, roundF_roundRhoB;
double roundF_roundGammaAA, roundF_roundGammaAB,
roundF_roundGammaBB;
pFunctional->getDerivativeFunctional(
rhoA, rhoB, gammaAA, gammaAB, gammaBB, &roundF_roundRhoA,
&roundF_roundRhoB, &roundF_roundGammaAA, &roundF_roundGammaAB,
&roundF_roundGammaBB);

this->build_XC_Matrix(
roundF_roundRhoA, roundF_roundGammaAA, roundF_roundGammaAB,
gradRhoXA, gradRhoYA, gradRhoZA, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values, pFunctional, weight, pF_A);
this->build_XC_Matrix(
roundF_roundRhoB, roundF_roundGammaBB, roundF_roundGammaAB,
gradRhoXB, gradRhoYB, gradRhoZB, AO_values, dAO_dx_values,
dAO_dy_values, dAO_dz_values, pFunctional, weight, pF_B);
energy += weight * pFunctional->getFunctional(rhoA, rhoB, gammaAA,
gammaAB, gammaBB);
}
}

return energy;
}

double DfCalcGridX::calcXCIntegForFockAndEnergy(
const TlDenseSymmetricMatrix_Lapack& P_A, DfFunctional_LDA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A) {
assert(pFunctional != NULL);
assert(pF_A != NULL);

this->calcRho_LDA(P_A);
double energy = this->buildVxc(pFunctional, pF_A);

return energy;
}

double DfCalcGridX::calcXCIntegForFockAndEnergy(
const TlDenseSymmetricMatrix_Lapack& P_A,
const TlDenseSymmetricMatrix_Lapack& P_B, DfFunctional_LDA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A, TlDenseSymmetricMatrix_Lapack* pF_B) {
assert(pFunctional != NULL);
assert(pF_A != NULL);
assert(pF_B != NULL);

this->calcRho_LDA(P_A, P_B);
double energy = this->buildVxc(pFunctional, pF_A, pF_B);
return energy;
}

double DfCalcGridX::calcXCIntegForFockAndEnergy(
const TlDenseSymmetricMatrix_Lapack& P_A, DfFunctional_GGA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A) {
assert(pFunctional != NULL);
assert(pF_A != NULL);

this->calcRho_GGA(P_A);
double energy = this->buildVxc(pFunctional, pF_A);

return energy;
}

double DfCalcGridX::calcXCIntegForFockAndEnergy(
const TlDenseSymmetricMatrix_Lapack& P_A,
const TlDenseSymmetricMatrix_Lapack& P_B, DfFunctional_GGA* pFunctional,
TlDenseSymmetricMatrix_Lapack* pF_A, TlDenseSymmetricMatrix_Lapack* pF_B) {
assert(pFunctional != NULL);
assert(pF_A != NULL);
assert(pF_B != NULL);

this->calcRho_GGA(P_A, P_B);
double energy = this->buildVxc(pFunctional, pF_A, pF_B);
return energy;
}



void DfCalcGridX::build_XC_Matrix(const double roundF_roundRhoA,
const std::vector<double>& AO_values,
DfFunctional_LDA* pFunctional,
const double weight, TlMatrixObject* pF_A) {
const double coef1_A = weight * roundF_roundRhoA;

const int numOfAOs = this->m_nNumOfAOs;
for (int i = 0; i < numOfAOs; ++i) {
const double aoi = AO_values[i];

pF_A->add(i, i, coef1_A * aoi * aoi);
for (int j = 0; j < i; ++j) {
pF_A->add(i, j, coef1_A * aoi * AO_values[j]);
}
}
}




void DfCalcGridX::build_XC_Matrix(
const double roundF_roundRhoA, const double roundF_roundGammaAA,
const double roundF_roundGammaAB, const double gradRhoAX,
const double gradRhoAY, const double gradRhoAZ,
const std::vector<double>& AO_values,
const std::vector<double>& dAO_dx_values,
const std::vector<double>& dAO_dy_values,
const std::vector<double>& dAO_dz_values, DfFunctional_GGA* pFunctional,
const double weight, TlMatrixObject* pF_A) {
const double coef1_A = weight * roundF_roundRhoA;
const double roundF_roundGammaAA2 = 2.0 * roundF_roundGammaAA;
const double coef2_AX = weight * (roundF_roundGammaAA2 * gradRhoAX +
roundF_roundGammaAB * gradRhoAX);
const double coef2_AY = weight * (roundF_roundGammaAA2 * gradRhoAY +
roundF_roundGammaAB * gradRhoAY);
const double coef2_AZ = weight * (roundF_roundGammaAA2 * gradRhoAZ +
roundF_roundGammaAB * gradRhoAZ);

const int numOfAOs = this->m_nNumOfAOs;
for (int i = 0; i < numOfAOs; ++i) {
const double aoi = AO_values[i];
const double dxi = dAO_dx_values[i];
const double dyi = dAO_dy_values[i];
const double dzi = dAO_dz_values[i];

{
double v = coef1_A * aoi * aoi;
v += aoi * (coef2_AX * dxi + coef2_AY * dyi + coef2_AZ * dzi) * 2.0;
pF_A->add(i, i, v);
}

for (int j = 0; j < i; ++j) {
const double aoj = AO_values[j];
const double dxj = dAO_dx_values[j];
const double dyj = dAO_dy_values[j];
const double dzj = dAO_dz_values[j];

double v = coef1_A * aoi * aoj;
v += aoi * (coef2_AX * dxj + coef2_AY * dyj + coef2_AZ * dzj);
v += aoj * (coef2_AX * dxi + coef2_AY * dyi + coef2_AZ * dzi);
pF_A->add(i, j, v);
}
}
}

void DfCalcGridX::getWholeDensity(double* pRhoA, double* pRhoB) const {
TlDenseGeneralMatrix_Lapack gridMat;
gridMat.load(this->getGridMatrixPath(this->m_nIteration));
const index_type numOfGrids = gridMat.getNumOfRows();

TlDenseGeneralMatrix_Lapack weightMat;
gridMat.block(0, GM_WEIGHT, numOfGrids, 1, &weightMat);

const DfXCFunctional dfXcFunc(this->pPdfParam_);
if (this->m_nMethodType == METHOD_RKS) {
TlDenseGeneralMatrix_Lapack rhoMat_A;
if (dfXcFunc.getFunctionalType() == DfXCFunctional::LDA) {
gridMat.block(0, GM_LDA_RHO_ALPHA, numOfGrids, 1, &rhoMat_A);
} else {
gridMat.block(0, GM_GGA_RHO_ALPHA, numOfGrids, 1, &rhoMat_A);
}

assert(pRhoA != NULL);
*pRhoA = weightMat.dotInPlace(rhoMat_A).sum();
} else {
TlDenseGeneralMatrix_Lapack rhoMat_A;
TlDenseGeneralMatrix_Lapack rhoMat_B;
if (dfXcFunc.getFunctionalType() == DfXCFunctional::LDA) {
gridMat.block(0, GM_LDA_RHO_ALPHA, numOfGrids, 1, &rhoMat_A);
gridMat.block(0, GM_LDA_RHO_BETA, numOfGrids, 1, &rhoMat_B);
} else {
gridMat.block(0, GM_GGA_RHO_ALPHA, numOfGrids, 1, &rhoMat_A);
gridMat.block(0, GM_GGA_RHO_BETA, numOfGrids, 1, &rhoMat_B);
}

assert(pRhoA != NULL);
assert(pRhoB != NULL);
*pRhoA = weightMat.dotInPlace(rhoMat_A).sum();
*pRhoB = weightMat.dotInPlace(rhoMat_B).sum();
}
}
