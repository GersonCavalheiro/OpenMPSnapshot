
#include "DfGenerateGrid.h"

#include <cmath>

#include "CnError.h"
#include "DfXCFunctional.h"
#include "TlMath.h"
#include "TlOrbitalInfo_Density.h"
#include "TlPosition.h"
#include "TlPrdctbl.h"
#include "TlTime.h"
#include "TlUtils.h"

#define SQ2 1.414213562373095049
#define SQ1_2 0.707106781186547524
#define SQ1_3 0.577350269189625765
#define SMALL 1
#define TOOBIG 30.0
#define BOHR 0.52917706

DfGenerateGrid::DfGenerateGrid(TlSerializeData* pPdfParam)
: DfObject(pPdfParam),
flGeometry_((*pPdfParam)["coordinates"]),
radialGridType_(RG_EularMaclaurin),
GC_mappingType_(GC_TA),
partitioningMethod_(Partitioning_SSWeight),
isAtomicSizeAdjustments_(true),
isPruning_(true) {
const TlSerializeData& pdfParam = *pPdfParam;

this->xctype = pdfParam["xc_functional"].getStr();

this->weightCutoff_ = 1.0E-16;
if (pdfParam["grid/weight_cutoff"].getStr() != "") {
this->weightCutoff_ = pdfParam["grid/weight_cutoff"].getDouble();
}

const std::string sGridType = pdfParam["xc-potential/grid-type"].getStr();
if (TlUtils::toUpper(sGridType) == "COARSE") {
this->m_gridType = COARSE;
this->nrgrid = 35;
this->nOgrid = 110;
} else if (TlUtils::toUpper(sGridType) == "MEDIUM") {
this->m_gridType = MEDIUM;
this->nrgrid = 32;
this->nOgrid = 146;
} else if ((TlUtils::toUpper(sGridType) == "MEDIUM-FINE") ||
(TlUtils::toUpper(sGridType) == "MEDIUM_FINE")) {
this->m_gridType = MEDIUM_FINE;
this->nrgrid = 64;
this->nOgrid = 146;
} else if (TlUtils::toUpper(sGridType) == "FINE") {
this->m_gridType = FINE;
this->nrgrid = 75;
this->nOgrid = 302;
this->partitioningMethod_ = Partitioning_SSWeight;
} else if ((TlUtils::toUpper(sGridType) == "ULTRA-FINE") ||
(TlUtils::toUpper(sGridType) == "ULTRA_FINE")) {
this->m_gridType = ULTRA_FINE;
this->nrgrid = 99;
this->nOgrid = 590;
this->partitioningMethod_ = Partitioning_SSWeight;
} else if (TlUtils::toUpper(sGridType) == "USER") {
this->log_.info("grid type =  UserDefined");
this->m_gridType = USER;

this->nrgrid = 75;
if (pdfParam["grid/num_of_radial_shells"].getStr() != "") {
this->nrgrid = pdfParam["grid/num_of_radial_shells"].getInt();
}

this->nOgrid = 302;
if (pdfParam["grid/num_of_angular_points"].getStr() != "") {
this->nOgrid = pdfParam["grid/num_of_angular_points"].getInt();
}

this->radialGridType_ = RG_EularMaclaurin;
if (pdfParam["grid/radial_quadorature_method"].getStr() != "") {
const std::string method = TlUtils::toUpper(
pdfParam["grid/radial_quadorature_method"].getStr());
if ((method == "GC") || (method == "Gauss-Chebyshev")) {
this->radialGridType_ = RG_GaussChebyshev;
} else if ((method == "EM") || (method == "Eular-Maclaurin")) {
this->radialGridType_ = RG_EularMaclaurin;
}
}

this->GC_mappingType_ = GC_TA;
if (this->radialGridType_ == RG_GaussChebyshev) {
if (pdfParam["grid/GC_mapping_type"].getStr() != "") {
const std::string mappingType =
TlUtils::toUpper(pdfParam["grid/GC_mapping_type"].getStr());
if (mappingType == "BECKE") {
this->GC_mappingType_ = GC_BECKE;
} else if (mappingType == "TA") {
this->GC_mappingType_ = GC_TA;
} else if (mappingType == "KK") {
this->GC_mappingType_ = GC_KK;
}
}
}

this->isPruning_ = true;
if (pdfParam["grid/pruning"].getStr() != "") {
this->isPruning_ = pdfParam["grid/pruning"].getBoolean();
}

this->isAtomicSizeAdjustments_ = true;
} else if ((TlUtils::toUpper(sGridType) == "SG-1") ||
(TlUtils::toUpper(sGridType) == "SG1")) {
this->log_.info("grid type = SG-1");
this->m_gridType = SG_1;
this->nrgrid = 50;
this->nOgrid = 194;
this->radialGridType_ = RG_EularMaclaurin;
this->isAtomicSizeAdjustments_ = false;
} else {
this->log_.critical(
TlUtils::format("unsupported grid type: %s", sGridType.c_str()));
CnErr.abort();
}

if (pdfParam["grid/partitioning_method"].getStr() != "") {
const std::string partitioningMethod =
TlUtils::toUpper(pdfParam["grid/partitioning_method"].getStr());
if (partitioningMethod == "BECKE") {
this->partitioningMethod_ = Paritioning_Becke;
} else if (partitioningMethod == "SSWEIGHT") {
this->partitioningMethod_ = Partitioning_SSWeight;
} else {
this->log_.warn(TlUtils::format(
"unknown parameter: grid/partitioning_method = %s",
pdfParam["grid/partitioning_method"].getStr().c_str()));
}
}

if (pdfParam["grid/atomic_size_adjustments"].getStr() != "") {
this->isAtomicSizeAdjustments_ =
pdfParam["grid/atomic_size_adjustments"].getBoolean();
}

this->log_.info(TlUtils::format("# radial grids  = %d", this->nrgrid));
this->log_.info(TlUtils::format("# lebedev grids = %d", this->nOgrid));
switch (this->radialGridType_) {
case RG_EularMaclaurin:
this->log_.info("ragial quadrature type: Eular-Maclaurin");
break;

case RG_GaussChebyshev:
this->log_.info("ragial quadrature type: Gauss-Chebyshev");
switch (this->GC_mappingType_) {
case GC_BECKE:
this->log_.info("Gauss-Chebyshev mapping type: Becke");
break;
case GC_TA:
this->log_.info("Gauss-Chebyshev mapping type: TA");
break;
case GC_KK:
this->log_.info("Gauss-Chebyshev mapping type: KK");
break;
}
break;
}
switch (this->partitioningMethod_) {
case Paritioning_Becke:
this->log_.info("partitioning method: Becke");
if (this->isAtomicSizeAdjustments_) {
this->log_.info("atomic size adjustments: yes");
} else {
this->log_.info("atomic size adjustments: no");
}
break;

case Partitioning_SSWeight:
this->log_.info("partitioning method: SSWeight");
break;

default:
this->log_.critical("wrong partitioning method.");
break;
}
if (this->isPruning_) {
this->log_.info("pruning grids: yes");
} else {
this->log_.info("pruning grids: no");
}

const int dNumOfAtoms = this->m_nNumOfAtoms;
this->log_.info(TlUtils::format("# atoms: %d", dNumOfAtoms));
this->coord_.resize(dNumOfAtoms);
for (int i = 0; i < dNumOfAtoms; ++i) {
this->coord_[i] = this->flGeometry_.getCoordinate(i);
}

this->distanceMatrix_.resize(dNumOfAtoms);
this->invDistanceMatrix_.resize(dNumOfAtoms);
for (int mc = 0; mc < dNumOfAtoms; ++mc) {
const TlPosition pos_mc = this->coord_[mc];
for (int nc = 0; nc < mc; ++nc) {
const double dist = pos_mc.distanceFrom(this->coord_[nc]);
if (dist < 1E-10) {
std::cerr << " distance < 1E-10: " << mc << " th atom and "
<< nc << " th atom." << std::endl;
}
this->distanceMatrix_.set(mc, nc, dist);
this->invDistanceMatrix_.set(mc, nc, 1.0 / dist);
}
}

this->numOfColsOfGrdMat_ = 5;  
{
const int coef = (this->m_nMethodType == METHOD_RKS) ? 1 : 2;
DfXCFunctional dfXcFunctional(this->pPdfParam_);
if (dfXcFunctional.getFunctionalType() == DfXCFunctional::LDA) {
this->numOfColsOfGrdMat_ += coef * 1;  
} else if (dfXcFunctional.getFunctionalType() == DfXCFunctional::GGA) {
this->numOfColsOfGrdMat_ +=
coef * 4;  
}
}

this->logger("make table");
this->makeTable();

this->logger("set Cell Para");
this->setCellPara();

this->log_.info("calc grid origin");
this->O_ = this->getOMatrix();
}

DfGenerateGrid::~DfGenerateGrid() {}

int DfGenerateGrid::dfGrdMain() {
this->logger("start");


this->logger("generateGrid");
this->generateGrid(this->O_);

this->logger("end");

return 0;
}

void DfGenerateGrid::makeTable() {
TlOrbitalInfo_Density orbInfoAuxCD((*this->pPdfParam_)["coordinates"],
(*this->pPdfParam_)["basis_set_j"]);
const int maxNumOfAuxCDs = orbInfoAuxCD.getNumOfOrbitals();

double dMaxExpAlpha = 0.0;
for (int i = 0; i < maxNumOfAuxCDs; ++i) {
dMaxExpAlpha = std::max(dMaxExpAlpha, orbInfoAuxCD.getExponent(i, 0));
}

const double r = std::max((TOOBIG / dMaxExpAlpha), 200.0);
this->maxRadii_ = std::sqrt(r);
}

void DfGenerateGrid::setCellPara() {
const int maxKindOfAtoms = 110;
this->radiusList_.resize(maxKindOfAtoms);  
this->xGL_.resize(100);
this->wGL_.resize(100);

if ((this->m_gridType == COARSE) || (this->m_gridType == MEDIUM) ||
(this->m_gridType == MEDIUM_FINE) || (this->m_gridType == FINE)) {
this->radiusList_[1] = 0.50;
this->radiusList_[2] = 2.00;
this->radiusList_[3] = 1.45;
this->radiusList_[4] = 1.05;
this->radiusList_[5] = 0.85;
this->radiusList_[6] = 0.70;
this->radiusList_[7] = 0.65;
this->radiusList_[8] = 0.60;
this->radiusList_[9] = 0.50;
this->radiusList_[10] = 2.25;

this->radiusList_[11] = 1.80;
this->radiusList_[12] = 1.50;
this->radiusList_[13] = 1.25;
this->radiusList_[14] = 1.10;
this->radiusList_[15] = 1.00;
this->radiusList_[16] = 1.00;
this->radiusList_[17] = 1.00;
this->radiusList_[18] = 2.50;
this->radiusList_[19] = 2.20;
this->radiusList_[20] = 1.80;

this->radiusList_[21] = 1.60;
this->radiusList_[22] = 1.40;
this->radiusList_[23] = 1.35;
this->radiusList_[24] = 1.40;
this->radiusList_[25] = 1.40;
this->radiusList_[26] = 1.40;
this->radiusList_[27] = 1.35;
this->radiusList_[28] = 1.35;
this->radiusList_[29] = 1.35;
this->radiusList_[30] = 1.35;

this->radiusList_[31] = 1.30;
this->radiusList_[32] = 1.25;
this->radiusList_[33] = 1.15;
this->radiusList_[34] = 1.15;
this->radiusList_[35] = 1.15;
this->radiusList_[36] = 2.75;
this->radiusList_[37] = 2.35;
this->radiusList_[38] = 2.00;
this->radiusList_[39] = 1.80;
this->radiusList_[40] = 1.55;

this->radiusList_[41] = 1.45;
this->radiusList_[42] = 1.45;
this->radiusList_[43] = 1.35;
this->radiusList_[44] = 1.30;
this->radiusList_[45] = 1.35;
this->radiusList_[46] = 1.40;
this->radiusList_[47] = 1.60;
this->radiusList_[48] = 1.55;
this->radiusList_[49] = 1.55;
this->radiusList_[50] = 1.45;

this->radiusList_[51] = 1.45;
this->radiusList_[52] = 1.40;
this->radiusList_[53] = 1.40;
this->radiusList_[54] = 3.00;
this->radiusList_[55] = 2.60;
this->radiusList_[56] = 2.15;
this->radiusList_[57] = 1.95;
this->radiusList_[58] = 1.85;
this->radiusList_[59] = 1.85;
this->radiusList_[60] = 1.85;

this->radiusList_[61] = 1.85;
this->radiusList_[62] = 1.85;
this->radiusList_[63] = 1.85;
this->radiusList_[64] = 1.80;
this->radiusList_[65] = 1.75;
this->radiusList_[66] = 1.75;
this->radiusList_[67] = 1.75;
this->radiusList_[68] = 1.75;
this->radiusList_[69] = 1.75;
this->radiusList_[70] = 1.75;

this->radiusList_[71] = 1.75;
this->radiusList_[72] = 1.55;
this->radiusList_[73] = 1.45;
this->radiusList_[74] = 1.35;
this->radiusList_[75] = 1.35;
this->radiusList_[76] = 1.30;
this->radiusList_[77] = 1.35;
this->radiusList_[78] = 1.35;
this->radiusList_[79] = 1.35;
this->radiusList_[80] = 1.50;

this->radiusList_[81] = 1.90;
this->radiusList_[82] = 1.80;
this->radiusList_[83] = 1.60;
this->radiusList_[84] = 1.90;
this->radiusList_[85] = 1.65;
this->radiusList_[86] = 3.25;
this->radiusList_[87] = 2.80;
this->radiusList_[88] = 2.15;
this->radiusList_[89] = 1.95;
this->radiusList_[90] = 1.80;

this->radiusList_[91] = 1.80;
this->radiusList_[92] = 1.75;
this->radiusList_[93] = 1.75;
this->radiusList_[94] = 1.75;
this->radiusList_[95] = 1.75;
this->radiusList_[96] = 1.75;
this->radiusList_[97] = 1.75;
this->radiusList_[98] = 1.75;
this->radiusList_[99] = 1.75;
this->radiusList_[100] = 1.75;

this->radiusList_[101] = 1.75;
this->radiusList_[102] = 1.75;
this->radiusList_[103] = 1.75;
this->radiusList_[104] = 1.55;
this->radiusList_[105] = 1.55;

} else if ((this->m_gridType == SG_1) || (this->m_gridType == USER)) {
this->radiusList_[1] = 1.0000;   
this->radiusList_[2] = 0.5882;   
this->radiusList_[3] = 3.0769;   
this->radiusList_[4] = 2.0513;   
this->radiusList_[5] = 1.5385;   
this->radiusList_[6] = 1.2308;   
this->radiusList_[7] = 1.0256;   
this->radiusList_[8] = 0.8791;   
this->radiusList_[9] = 0.7692;   
this->radiusList_[10] = 0.6838;  

this->radiusList_[11] = 4.0909;  
this->radiusList_[12] = 3.1579;  
this->radiusList_[13] = 2.5714;  
this->radiusList_[14] = 2.1687;  
this->radiusList_[15] = 1.8750;  
this->radiusList_[16] = 1.6514;  
this->radiusList_[17] = 1.4754;  
this->radiusList_[18] = 1.3333;  
this->radiusList_[19] = 2.20;    
this->radiusList_[20] = 2.03;    

this->radiusList_[21] = 3.023563;  
this->radiusList_[22] = 2.645618;  
this->radiusList_[23] = 2.551131;  
this->radiusList_[24] = 2.645618;
this->radiusList_[25] = 2.645618;
this->radiusList_[26] = 2.645618;
this->radiusList_[27] = 2.551131;
this->radiusList_[28] = 2.551131;
this->radiusList_[29] = 2.551131;
this->radiusList_[30] = 2.551131;

this->radiusList_[31] = 2.456645;  
this->radiusList_[32] = 2.362159;  
this->radiusList_[33] = 2.173186;  
this->radiusList_[34] = 2.173186;
this->radiusList_[35] = 2.173186;
this->radiusList_[36] = 5.196749;  
this->radiusList_[37] = 4.440858;  
this->radiusList_[38] = 3.779454;  
this->radiusList_[39] = 3.401508;  
this->radiusList_[40] = 2.551131;  

this->radiusList_[41] = 2.702309;  
this->radiusList_[42] = 2.740104;  
this->radiusList_[43] = 2.551131;  
this->radiusList_[44] = 2.456645;  
this->radiusList_[45] = 2.551131;  
this->radiusList_[46] = 2.645618;  
this->radiusList_[47] = 3.023563;  
this->radiusList_[48] = 2.929077;  
this->radiusList_[49] = 2.929077;  
this->radiusList_[50] = 2.740104;  

this->radiusList_[51] = 2.740104;  
this->radiusList_[52] = 2.645618;  
this->radiusList_[53] = 2.645618;  
this->radiusList_[54] = 5.669181;  
this->radiusList_[55] = 4.913290;  
this->radiusList_[56] = 4.062913;  
this->radiusList_[57] = 3.684967;  
this->radiusList_[58] = 3.495995;  
this->radiusList_[59] = 3.495995;  
this->radiusList_[60] = 3.495995;  

this->radiusList_[61] = 3.458200;  
this->radiusList_[62] = 3.495995;  
this->radiusList_[63] = 3.495995;  
this->radiusList_[64] = 3.401508;  
this->radiusList_[65] = 3.307022;  
this->radiusList_[66] = 3.307022;  
this->radiusList_[67] = 3.307022;  
this->radiusList_[68] = 3.307022;  
this->radiusList_[69] = 3.307022;  
this->radiusList_[70] = 3.307022;  

this->radiusList_[71] = 3.307022;  
this->radiusList_[72] = 2.929077;  
this->radiusList_[73] = 2.740104;  
this->radiusList_[74] = 2.551131;  
this->radiusList_[75] = 2.551131;  
this->radiusList_[76] = 2.456645;  
this->radiusList_[77] = 2.551131;  
this->radiusList_[78] = 2.551131;  
this->radiusList_[79] = 2.551131;  
this->radiusList_[80] = 2.834590;  

this->radiusList_[81] = 3.590481;  
this->radiusList_[82] = 3.401508;  
this->radiusList_[83] = 3.023563;  
this->radiusList_[84] = 3.590481;  
this->radiusList_[85] = 3.118049;  
this->radiusList_[86] = 6.141612;  
this->radiusList_[87] = 5.291235;  
this->radiusList_[88] = 4.062913;  
this->radiusList_[89] = 3.684967;  
this->radiusList_[90] = 3.401508;  

this->radiusList_[91] = 3.401508;   
this->radiusList_[92] = 3.307022;   
this->radiusList_[93] = 3.307022;   
this->radiusList_[94] = 3.307022;   
this->radiusList_[95] = 3.307022;   
this->radiusList_[96] = 3.307022;   
this->radiusList_[97] = 3.307022;   
this->radiusList_[98] = 3.307022;   
this->radiusList_[99] = 3.307022;   
this->radiusList_[100] = 3.307022;  

this->radiusList_[101] = 3.307022;  
this->radiusList_[102] = 3.307022;  
this->radiusList_[103] = 3.307022;  
this->radiusList_[104] = 2.929077;  
this->radiusList_[105] = 2.929077;  
}

if ((this->m_gridType == COARSE) || (this->m_gridType == MEDIUM)) {
this->xGL_[0] = -9.9726386184948157e-01;
this->xGL_[1] = -9.8561151154526838e-01;
this->xGL_[2] = -9.6476225558750639e-01;
this->xGL_[3] = -9.3490607593773967e-01;
this->xGL_[4] = -8.9632115576605209e-01;
this->xGL_[5] = -8.4936761373256997e-01;
this->xGL_[6] = -7.9448379596794239e-01;
this->xGL_[7] = -7.3218211874028971e-01;
this->xGL_[8] = -6.6304426693021523e-01;
this->xGL_[9] = -5.8771575724076230e-01;
this->xGL_[10] = -5.0689990893222936e-01;
this->xGL_[11] = -4.2135127613063533e-01;
this->xGL_[12] = -3.3186860228212767e-01;
this->xGL_[13] = -2.3928736225213706e-01;
this->xGL_[14] = -1.4447196158279649e-01;
this->xGL_[15] = -4.8307665687738310e-02;
this->xGL_[16] = 4.8307665687738310e-02;
this->xGL_[17] = 1.4447196158279649e-01;
this->xGL_[18] = 2.3928736225213706e-01;
this->xGL_[19] = 3.3186860228212767e-01;
this->xGL_[20] = 4.2135127613063533e-01;
this->xGL_[21] = 5.0689990893222936e-01;
this->xGL_[22] = 5.8771575724076230e-01;
this->xGL_[23] = 6.6304426693021523e-01;
this->xGL_[24] = 7.3218211874028971e-01;
this->xGL_[25] = 7.9448379596794239e-01;
this->xGL_[26] = 8.4936761373256997e-01;
this->xGL_[27] = 8.9632115576605209e-01;
this->xGL_[28] = 9.3490607593773967e-01;
this->xGL_[29] = 9.6476225558750639e-01;
this->xGL_[30] = 9.8561151154526838e-01;
this->xGL_[31] = 9.9726386184948157e-01;

this->wGL_[0] = 7.0186100094695213e-03;
this->wGL_[1] = 1.6274394730905709e-02;
this->wGL_[2] = 2.5392065309262139e-02;
this->wGL_[3] = 3.4273862913021411e-02;
this->wGL_[4] = 4.2835898022226704e-02;
this->wGL_[5] = 5.0998059262376154e-02;
this->wGL_[6] = 5.8684093478535128e-02;
this->wGL_[7] = 6.5822222776361808e-02;
this->wGL_[8] = 7.2345794108848616e-02;
this->wGL_[9] = 7.8193895787070436e-02;
this->wGL_[10] = 8.3311924226946721e-02;
this->wGL_[11] = 8.7652093004403742e-02;
this->wGL_[12] = 9.1173878695763905e-02;
this->wGL_[13] = 9.3844399080804414e-02;
this->wGL_[14] = 9.5638720079274847e-02;
this->wGL_[15] = 9.6540088514727854e-02;
this->wGL_[16] = 9.6540088514727854e-02;
this->wGL_[17] = 9.5638720079274847e-02;
this->wGL_[18] = 9.3844399080804414e-02;
this->wGL_[19] = 9.1173878695763905e-02;
this->wGL_[20] = 8.7652093004403742e-02;
this->wGL_[21] = 8.3311924226946721e-02;
this->wGL_[22] = 7.8193895787070436e-02;
this->wGL_[23] = 7.2345794108848616e-02;
this->wGL_[24] = 6.5822222776361808e-02;
this->wGL_[25] = 5.8684093478535128e-02;
this->wGL_[26] = 5.0998059262376154e-02;
this->wGL_[27] = 4.2835898022226704e-02;
this->wGL_[28] = 3.4273862913021411e-02;
this->wGL_[29] = 2.5392065309262139e-02;
this->wGL_[30] = 1.6274394730905709e-02;
this->wGL_[31] = 7.0186100094695213e-03;

} else if ((this->m_gridType == MEDIUM_FINE) ||
(this->m_gridType == FINE) || (this->m_gridType == SG_1)) {
this->xGL_[0] = -9.9930504173577217e-01;
this->xGL_[1] = -9.9634011677195522e-01;
this->xGL_[2] = -9.9101337147674429e-01;
this->xGL_[3] = -9.8333625388462598e-01;
this->xGL_[4] = -9.7332682778991098e-01;
this->xGL_[5] = -9.6100879965205377e-01;
this->xGL_[6] = -9.4641137485840277e-01;
this->xGL_[7] = -9.2956917213193957e-01;
this->xGL_[8] = -9.1052213707850282e-01;
this->xGL_[9] = -8.8931544599511414e-01;
this->xGL_[10] = -8.6599939815409288e-01;
this->xGL_[11] = -8.4062929625258043e-01;
this->xGL_[12] = -8.1326531512279754e-01;
this->xGL_[13] = -7.8397235894334139e-01;
this->xGL_[14] = -7.5281990726053194e-01;
this->xGL_[15] = -7.1988185017161088e-01;
this->xGL_[16] = -6.8523631305423327e-01;
this->xGL_[17] = -6.4896547125465731e-01;
this->xGL_[18] = -6.1115535517239328e-01;
this->xGL_[19] = -5.7189564620263400e-01;
this->xGL_[20] = -5.3127946401989457e-01;
this->xGL_[21] = -4.8940314570705296e-01;
this->xGL_[22] = -4.4636601725346409e-01;
this->xGL_[23] = -4.0227015796399163e-01;
this->xGL_[24] = -3.5722015833766813e-01;
this->xGL_[25] = -3.1132287199021097e-01;
this->xGL_[26] = -2.6468716220876742e-01;
this->xGL_[27] = -2.1742364374000708e-01;
this->xGL_[28] = -1.6964442042399280e-01;
this->xGL_[29] = -1.2146281929612056e-01;
this->xGL_[30] = -7.2993121787799042e-02;
this->xGL_[31] = -2.4350292663424429e-02;
this->xGL_[32] = 2.4350292663424429e-02;
this->xGL_[33] = 7.2993121787799042e-02;
this->xGL_[34] = 1.2146281929612056e-01;
this->xGL_[35] = 1.6964442042399280e-01;
this->xGL_[36] = 2.1742364374000708e-01;
this->xGL_[37] = 2.6468716220876742e-01;
this->xGL_[38] = 3.1132287199021097e-01;
this->xGL_[39] = 3.5722015833766813e-01;
this->xGL_[40] = 4.0227015796399163e-01;
this->xGL_[41] = 4.4636601725346409e-01;
this->xGL_[42] = 4.8940314570705296e-01;
this->xGL_[43] = 5.3127946401989457e-01;
this->xGL_[44] = 5.7189564620263400e-01;
this->xGL_[45] = 6.1115535517239328e-01;
this->xGL_[46] = 6.4896547125465731e-01;
this->xGL_[47] = 6.8523631305423327e-01;
this->xGL_[48] = 7.1988185017161088e-01;
this->xGL_[49] = 7.5281990726053194e-01;
this->xGL_[50] = 7.8397235894334139e-01;
this->xGL_[51] = 8.1326531512279754e-01;
this->xGL_[52] = 8.4062929625258043e-01;
this->xGL_[53] = 8.6599939815409288e-01;
this->xGL_[54] = 8.8931544599511414e-01;
this->xGL_[55] = 9.1052213707850282e-01;
this->xGL_[56] = 9.2956917213193957e-01;
this->xGL_[57] = 9.4641137485840277e-01;
this->xGL_[58] = 9.6100879965205377e-01;
this->xGL_[59] = 9.7332682778991098e-01;
this->xGL_[60] = 9.8333625388462598e-01;
this->xGL_[61] = 9.9101337147674429e-01;
this->xGL_[62] = 9.9634011677195522e-01;
this->xGL_[63] = 9.9930504173577217e-01;

this->wGL_[0] = 1.7832807216962678e-03;
this->wGL_[1] = 4.1470332605625208e-03;
this->wGL_[2] = 6.5044579689783680e-03;
this->wGL_[3] = 8.8467598263639348e-03;
this->wGL_[4] = 1.1168139460131076e-02;
this->wGL_[5] = 1.3463047896718736e-02;
this->wGL_[6] = 1.5726030476024472e-02;
this->wGL_[7] = 1.7951715775697225e-02;
this->wGL_[8] = 2.0134823153530181e-02;
this->wGL_[9] = 2.2270173808383212e-02;
this->wGL_[10] = 2.4352702568710933e-02;
this->wGL_[11] = 2.6377469715054645e-02;
this->wGL_[12] = 2.8339672614259459e-02;
this->wGL_[13] = 3.0234657072402426e-02;
this->wGL_[14] = 3.2057928354851606e-02;
this->wGL_[15] = 3.3805161837141613e-02;
this->wGL_[16] = 3.5472213256882358e-02;
this->wGL_[17] = 3.7055128540240068e-02;
this->wGL_[18] = 3.8550153178615598e-02;
this->wGL_[19] = 3.9953741132720363e-02;
this->wGL_[20] = 4.1262563242623548e-02;
this->wGL_[21] = 4.2473515123653625e-02;
this->wGL_[22] = 4.3583724529323471e-02;
this->wGL_[23] = 4.4590558163756580e-02;
this->wGL_[24] = 4.5491627927418121e-02;
this->wGL_[25] = 4.6284796581314396e-02;
this->wGL_[26] = 4.6968182816209993e-02;
this->wGL_[27] = 4.7540165714830315e-02;
this->wGL_[28] = 4.7999388596458331e-02;
this->wGL_[29] = 4.8344762234802899e-02;
this->wGL_[30] = 4.8575467441503394e-02;
this->wGL_[31] = 4.8690957009139731e-02;
this->wGL_[32] = 4.8690957009139731e-02;
this->wGL_[33] = 4.8575467441503394e-02;
this->wGL_[34] = 4.8344762234802899e-02;
this->wGL_[35] = 4.7999388596458331e-02;
this->wGL_[36] = 4.7540165714830315e-02;
this->wGL_[37] = 4.6968182816209993e-02;
this->wGL_[38] = 4.6284796581314396e-02;
this->wGL_[39] = 4.5491627927418121e-02;
this->wGL_[40] = 4.4590558163756580e-02;
this->wGL_[41] = 4.3583724529323471e-02;
this->wGL_[42] = 4.2473515123653625e-02;
this->wGL_[43] = 4.1262563242623548e-02;
this->wGL_[44] = 3.9953741132720363e-02;
this->wGL_[45] = 3.8550153178615598e-02;
this->wGL_[46] = 3.7055128540240068e-02;
this->wGL_[47] = 3.5472213256882358e-02;
this->wGL_[48] = 3.3805161837141613e-02;
this->wGL_[49] = 3.2057928354851606e-02;
this->wGL_[50] = 3.0234657072402426e-02;
this->wGL_[51] = 2.8339672614259459e-02;
this->wGL_[52] = 2.6377469715054645e-02;
this->wGL_[53] = 2.4352702568710933e-02;
this->wGL_[54] = 2.2270173808383212e-02;
this->wGL_[55] = 2.0134823153530181e-02;
this->wGL_[56] = 1.7951715775697225e-02;
this->wGL_[57] = 1.5726030476024472e-02;
this->wGL_[58] = 1.3463047896718736e-02;
this->wGL_[59] = 1.1168139460131076e-02;
this->wGL_[60] = 8.8467598263639348e-03;
this->wGL_[61] = 6.5044579689783680e-03;
this->wGL_[62] = 4.1470332605625208e-03;
this->wGL_[63] = 1.7832807216962678e-03;
}
}

void DfGenerateGrid::generateGrid(const TlDenseGeneralMatrix_Lapack& O) {
std::size_t numOfGrids = 0;
const int endAtom = this->m_nNumOfAtoms;

#pragma omp parallel for schedule(runtime)
for (int atom = 0; atom < endAtom; ++atom) {
std::vector<double> coordX;
std::vector<double> coordY;
std::vector<double> coordZ;
std::vector<double> weight;

this->generateGrid_atom(O, atom, &coordX, &coordY, &coordZ, &weight);

const std::size_t numOfAtomGrids = weight.size();
if (numOfAtomGrids > 0) {
#pragma omp critical(DfGenerateGrid__generateGrid)
{
this->grdMat_.resize(numOfGrids + numOfAtomGrids,
this->numOfColsOfGrdMat_);
for (std::size_t i = 0; i < numOfAtomGrids; ++i) {
this->grdMat_.set(numOfGrids, 0, coordX[i]);
this->grdMat_.set(numOfGrids, 1, coordY[i]);
this->grdMat_.set(numOfGrids, 2, coordZ[i]);
this->grdMat_.set(numOfGrids, 3, weight[i]);
this->grdMat_.set(numOfGrids, 4, atom);
++numOfGrids;
}
}
}
}

this->saveGridMatrix(0, this->grdMat_);
}
























void DfGenerateGrid::generateGrid_atom(const TlDenseGeneralMatrix_Lapack& O,
const int iAtom,
std::vector<double>* pCoordX,
std::vector<double>* pCoordY,
std::vector<double>* pCoordZ,
std::vector<double>* pWeight) {
assert(pCoordX != NULL);
assert(pCoordY != NULL);
assert(pCoordZ != NULL);
assert(pWeight != NULL);

std::vector<TlPosition> crdpoint;
std::vector<double> weightvec;

int numOfGrids = 0;
const int atomnum =
TlPrdctbl::getAtomicNumber(this->flGeometry_.getAtomSymbol(iAtom));
if (atomnum != 0) {
double rM = 0.0;
if (this->m_gridType == SG_1) {
rM = this->radiusList_[atomnum];
} else {
rM = TlPrdctbl::getBraggSlaterRadii(atomnum);
}
const double inv_rM = 1.0 / rM;

std::vector<double> alpha(5);
alpha[0] = 0.0;
alpha[1] = 0.0;
alpha[2] = 0.0;
alpha[3] = 10000.0;
if ((atomnum == 1) || (atomnum == 2)) {
alpha[0] = 0.2500;
alpha[1] = 0.5000;
alpha[2] = 1.0000;
alpha[3] = 4.5000;
} else if ((3 <= atomnum) && (atomnum <= 10)) {
alpha[0] = 0.1667;
alpha[1] = 0.5000;
alpha[2] = 0.9000;
alpha[3] = 3.5000;
} else if ((11 <= atomnum) && (atomnum <= 18)) {
alpha[0] = 0.1000;
alpha[1] = 0.4000;
alpha[2] = 0.8000;
alpha[3] = 2.5000;
}

const int radvec_max = this->nrgrid;
const int Nr = this->nrgrid;
for (int radvec = 0; radvec < radvec_max; ++radvec) {
const int i = radvec + 1;

double ri = 0.0;
double wr = 0.0;
switch (this->radialGridType_) {
case RG_GaussChebyshev:
this->getRadialAbscissaAndWeight_GaussChebyshev(rM, Nr, i,
&ri, &wr);
break;

case RG_EularMaclaurin:
this->getRadialAbscissaAndWeight_EulerMaclaurin(rM, Nr, i,
&ri, &wr);
break;

default:
this->log_.critical("unknown radial grid type.");
break;
}

if (ri > 30.0) {
continue;
}

int Ogrid = 0;
if (this->m_gridType == SG_1) {
Ogrid = this->getNumOfPrunedAnglarPoints_SG1(ri, inv_rM, alpha);
} else {
Ogrid =
this->getNumOfPrunedAnglarPoints(ri, this->nOgrid, atomnum);
}


std::vector<TlPosition> grid(Ogrid);
std::vector<double> lebWeight(Ogrid);
this->getSphericalGrids(Ogrid, ri, wr, this->coord_[iAtom], O,
&grid, &lebWeight);
assert(grid.size() == static_cast<std::size_t>(Ogrid));
assert(lebWeight.size() == static_cast<std::size_t>(Ogrid));

switch (this->partitioningMethod_) {
case Paritioning_Becke:
this->calcMultiCenterWeight_Becke(iAtom, Ogrid, grid,
&lebWeight);
break;

case Partitioning_SSWeight:
this->calcMultiCenterWeight_SS(iAtom, Ogrid, grid,
&lebWeight);
break;

default:
this->log_.critical("unknown partitioning method type.");
break;
}

crdpoint.resize(numOfGrids + Ogrid);
weightvec.resize(numOfGrids + Ogrid);
for (int i = 0; i < Ogrid; ++i) {
crdpoint[numOfGrids + i] = grid[i];
weightvec[numOfGrids + i] = lebWeight[i];
}
numOfGrids += Ogrid;
}

{
const int numOfGrids_orig = numOfGrids;
this->screeningGridsByWeight0(&crdpoint, &weightvec);
numOfGrids = crdpoint.size();
assert(weightvec.size() == static_cast<std::size_t>(numOfGrids));
{
const int diff = numOfGrids_orig - numOfGrids;
const double ratio =
double(diff) / double(numOfGrids_orig) * 100.0;
this->log_.info(
TlUtils::format("screened grids: %d -> %d; (%d; %3.2f%%)",
numOfGrids_orig, numOfGrids, diff, ratio));
}
}
}

pCoordX->resize(numOfGrids);
pCoordY->resize(numOfGrids);
pCoordZ->resize(numOfGrids);
for (int i = 0; i < numOfGrids; ++i) {
(*pCoordX)[i] = crdpoint[i].x();
(*pCoordY)[i] = crdpoint[i].y();
(*pCoordZ)[i] = crdpoint[i].z();
}
*pWeight = weightvec;
}

void DfGenerateGrid::points2(const int nOgrid, const double r0,
const TlPosition& core, const double weight,
const TlDenseGeneralMatrix_Lapack& O,
std::vector<TlPosition>& Ogrid,
std::vector<double>& w) {
double bm[4];
double bl[4];

Ogrid.resize(nOgrid);
w.resize(nOgrid);
for (int i = 0; i < nOgrid; ++i) {
w[i] = weight;
Ogrid[i] = TlPosition(0.0, 0.0, 0.0);
}

if ((this->m_gridType == COARSE) || (this->m_gridType == FINE)) {
std::cout << "Sorry, do not support now." << std::endl;
} else if (this->m_gridType == SG_1) {
if (nOgrid == 6) {
for (int i = 0; i < nOgrid; ++i) {
w[i] *= 0.166666666666667;
}

Ogrid[0] = TlPosition(1.0, 0.0, 0.0);
Ogrid[1] = TlPosition(0.0, 1.0, 0.0);
Ogrid[2] = TlPosition(0.0, 0.0, 1.0);
Ogrid[3] = TlPosition(-1.0, 0.0, 0.0);
Ogrid[4] = TlPosition(0.0, -1.0, 0.0);
Ogrid[5] = TlPosition(0.0, 0.0, -1.0);
} else if (nOgrid == 38) {
const double A1 = 0.00952380952387;
const double A3 = 0.0321428571429;
const double C1 = 0.0285714285714;
const double p1 = 0.888073833977;
const double q1 = 0.459700843381;

for (int i = 0; i < 6; ++i) {
w[i] *= A1;
}
for (int i = 6; i < 14; ++i) {
w[i] *= A3;
}
for (int i = 14; i < 38; ++i) {
w[i] *= C1;
}

Ogrid[0] = TlPosition(1.0, 0.0, 0.0);
Ogrid[1] = TlPosition(0.0, 1.0, 0.0);
Ogrid[2] = TlPosition(0.0, 0.0, 1.0);
Ogrid[3] = TlPosition(-1.0, 0.0, 0.0);
Ogrid[4] = TlPosition(0.0, -1.0, 0.0);
Ogrid[5] = TlPosition(0.0, 0.0, -1.0);

Ogrid[6] = TlPosition(SQ1_3, SQ1_3, SQ1_3);
Ogrid[7] = TlPosition(-SQ1_3, SQ1_3, SQ1_3);
Ogrid[8] = TlPosition(SQ1_3, -SQ1_3, SQ1_3);
Ogrid[9] = TlPosition(SQ1_3, SQ1_3, -SQ1_3);
Ogrid[10] = TlPosition(-SQ1_3, -SQ1_3, SQ1_3);
Ogrid[11] = TlPosition(-SQ1_3, SQ1_3, -SQ1_3);
Ogrid[12] = TlPosition(SQ1_3, -SQ1_3, -SQ1_3);
Ogrid[13] = TlPosition(-SQ1_3, -SQ1_3, -SQ1_3);

Ogrid[14] = TlPosition(p1, q1, 0.0);
Ogrid[15] = TlPosition(p1, -q1, 0.0);
Ogrid[16] = TlPosition(-p1, q1, 0.0);
Ogrid[17] = TlPosition(-p1, -q1, 0.0);
Ogrid[18] = TlPosition(p1, 0.0, q1);
Ogrid[19] = TlPosition(p1, 0.0, -q1);
Ogrid[20] = TlPosition(-p1, 0.0, q1);
Ogrid[21] = TlPosition(-p1, 0.0, -q1);
Ogrid[22] = TlPosition(0.0, p1, q1);
Ogrid[23] = TlPosition(0.0, p1, -q1);
Ogrid[24] = TlPosition(0.0, -p1, q1);
Ogrid[25] = TlPosition(0.0, -p1, -q1);

Ogrid[26] = TlPosition(q1, p1, 0.0);
Ogrid[27] = TlPosition(q1, -p1, 0.0);
Ogrid[28] = TlPosition(-q1, p1, 0.0);
Ogrid[29] = TlPosition(-q1, -p1, 0.0);
Ogrid[30] = TlPosition(q1, 0.0, p1);
Ogrid[31] = TlPosition(q1, 0.0, -p1);
Ogrid[32] = TlPosition(-q1, 0.0, p1);
Ogrid[33] = TlPosition(-q1, 0.0, -p1);
Ogrid[34] = TlPosition(0.0, q1, p1);
Ogrid[35] = TlPosition(0.0, q1, -p1);
Ogrid[36] = TlPosition(0.0, -q1, p1);
Ogrid[37] = TlPosition(0.0, -q1, -p1);
} else if (nOgrid == 86) {

const double A1 = 0.0115440115441;
const double A3 = 0.0119439090859;
const double B1 = 0.0111105557106;
const double B2 = 0.0118765012945;
const double C1 = 0.0118123037469;
bm[0] = 0.852518311701;
bm[1] = 0.189063552885;
bl[0] = 0.369602846454;
bl[1] = 0.694354006603;
const double p1 = 0.927330657151;
const double q1 = 0.374243039090;

for (int i = 0; i < 6; ++i) {
w[i] *= A1;
}
for (int i = 6; i < 14; ++i) {
w[i] *= A3;
}
for (int i = 14; i < 38; ++i) {
w[i] *= B1;
}
for (int i = 38; i < 62; ++i) {
w[i] *= B2;
}
for (int i = 62; i < 86; ++i) {
w[i] *= C1;
}

Ogrid[0] = TlPosition(1.0, 0.0, 0.0);
Ogrid[1] = TlPosition(0.0, 1.0, 0.0);
Ogrid[2] = TlPosition(0.0, 0.0, 1.0);
Ogrid[3] = TlPosition(-1.0, 0.0, 0.0);
Ogrid[4] = TlPosition(0.0, -1.0, 0.0);
Ogrid[5] = TlPosition(0.0, 0.0, -1.0);

Ogrid[6] = TlPosition(SQ1_3, SQ1_3, SQ1_3);
Ogrid[7] = TlPosition(-SQ1_3, SQ1_3, SQ1_3);
Ogrid[8] = TlPosition(SQ1_3, -SQ1_3, SQ1_3);
Ogrid[9] = TlPosition(SQ1_3, SQ1_3, -SQ1_3);
Ogrid[10] = TlPosition(-SQ1_3, -SQ1_3, SQ1_3);
Ogrid[11] = TlPosition(-SQ1_3, SQ1_3, -SQ1_3);
Ogrid[12] = TlPosition(SQ1_3, -SQ1_3, -SQ1_3);
Ogrid[13] = TlPosition(-SQ1_3, -SQ1_3, -SQ1_3);

for (int i = 0; i < 2; ++i) {
Ogrid[14 + 24 * i + 0] = TlPosition(bl[i], bl[i], bm[i]);
Ogrid[14 + 24 * i + 1] = TlPosition(-bl[i], bl[i], bm[i]);
Ogrid[14 + 24 * i + 2] = TlPosition(bl[i], -bl[i], bm[i]);
Ogrid[14 + 24 * i + 3] = TlPosition(bl[i], bl[i], -bm[i]);
Ogrid[14 + 24 * i + 4] = TlPosition(-bl[i], -bl[i], bm[i]);
Ogrid[14 + 24 * i + 5] = TlPosition(-bl[i], bl[i], -bm[i]);
Ogrid[14 + 24 * i + 6] = TlPosition(bl[i], -bl[i], -bm[i]);
Ogrid[14 + 24 * i + 7] = TlPosition(-bl[i], -bl[i], -bm[i]);
}
for (int i = 0; i < 2; ++i) {
for (int j = 0; j < 8; ++j) {
TlPosition tmp = Ogrid[14 + 24 * i + j];
Ogrid[14 + 24 * i + 8 + j] =
TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[14 + 24 * i + 16 + j] =
TlPosition(tmp.z(), tmp.x(), tmp.y());
}
}

Ogrid[62] = TlPosition(p1, q1, 0.0);
Ogrid[63] = TlPosition(p1, -q1, 0.0);
Ogrid[64] = TlPosition(-p1, q1, 0.0);
Ogrid[65] = TlPosition(-p1, -q1, 0.0);

Ogrid[66] = TlPosition(p1, 0.0, q1);
Ogrid[67] = TlPosition(p1, 0.0, -q1);
Ogrid[68] = TlPosition(-p1, 0.0, q1);
Ogrid[69] = TlPosition(-p1, 0.0, -q1);

Ogrid[70] = TlPosition(0.0, p1, q1);
Ogrid[71] = TlPosition(0.0, p1, -q1);
Ogrid[72] = TlPosition(0.0, -p1, q1);
Ogrid[73] = TlPosition(0.0, -p1, -q1);

Ogrid[74] = TlPosition(q1, p1, 0.0);
Ogrid[75] = TlPosition(q1, -p1, 0.0);
Ogrid[76] = TlPosition(-q1, p1, 0.0);
Ogrid[77] = TlPosition(-q1, -p1, 0.0);

Ogrid[78] = TlPosition(q1, 0.0, p1);
Ogrid[79] = TlPosition(q1, 0.0, -p1);
Ogrid[80] = TlPosition(-q1, 0.0, p1);
Ogrid[81] = TlPosition(-q1, 0.0, -p1);

Ogrid[82] = TlPosition(0.0, q1, p1);
Ogrid[83] = TlPosition(0.0, q1, -p1);
Ogrid[84] = TlPosition(0.0, -q1, p1);
Ogrid[85] = TlPosition(0.0, -q1, -p1);
} else if (nOgrid == 194) {
const double A1 = 0.00178234044724;
const double A2 = 0.00571690594998;
const double A3 = 0.00557338317884;
const double B1 = 0.00551877146727;
const double B2 = 0.00515823771181;
const double B3 = 0.00560870408259;
const double B4 = 0.00410677702817;
const double C1 = 0.00505184606462;
const double D1 = 0.00553024891623;
bm[0] = 0.777493219315;
bm[1] = 0.912509096867;
bm[2] = 0.314196994183;
bm[3] = 0.982972302707;
bl[0] = 0.444693317871;
bl[1] = 0.289246562758;
bl[2] = 0.671297344270;
bl[3] = 0.129933544765;

const double p1 = 0.938319218138;
const double q1 = 0.345770219761;
const double dr = 0.836036015482;
const double du = 0.159041710538;
const double dw = 0.525118572443;

for (int i = 0; i < 6; ++i) {
w[i] *= A1;
}
for (int i = 6; i < 18; ++i) {
w[i] *= A2;
}
for (int i = 18; i < 26; ++i) {
w[i] *= A3;
}
for (int i = 26; i < 50; ++i) {
w[i] *= B1;
}
for (int i = 50; i < 74; ++i) {
w[i] *= B2;
}
for (int i = 74; i < 98; ++i) {
w[i] *= B3;
}
for (int i = 98; i < 122; ++i) {
w[i] *= B4;
}
for (int i = 122; i < 146; ++i) {
w[i] *= C1;
}
for (int i = 146; i < 194; ++i) {
w[i] *= D1;
}

Ogrid[0] = TlPosition(1.0, 0.0, 0.0);
Ogrid[1] = TlPosition(0.0, 1.0, 0.0);
Ogrid[2] = TlPosition(0.0, 0.0, 1.0);
Ogrid[3] = TlPosition(-1.0, 0.0, 0.0);
Ogrid[4] = TlPosition(0.0, -1.0, 0.0);
Ogrid[5] = TlPosition(0.0, 0.0, -1.0);

Ogrid[6] = TlPosition(SQ1_2, SQ1_2, 0.0);
Ogrid[7] = TlPosition(-SQ1_2, SQ1_2, 0.0);
Ogrid[8] = TlPosition(SQ1_2, -SQ1_2, 0.0);
Ogrid[9] = TlPosition(-SQ1_2, -SQ1_2, 0.0);

for (int i = 0; i < 4; ++i) {
const TlPosition tmp = Ogrid[6 + i];
Ogrid[6 + 4 + i] = TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[6 + 8 + i] = TlPosition(tmp.z(), tmp.x(), tmp.y());
}

Ogrid[18] = TlPosition(SQ1_3, SQ1_3, SQ1_3);
Ogrid[19] = TlPosition(-SQ1_3, SQ1_3, SQ1_3);
Ogrid[20] = TlPosition(SQ1_3, -SQ1_3, SQ1_3);
Ogrid[21] = TlPosition(SQ1_3, SQ1_3, -SQ1_3);
Ogrid[22] = TlPosition(-SQ1_3, -SQ1_3, SQ1_3);
Ogrid[23] = TlPosition(-SQ1_3, SQ1_3, -SQ1_3);
Ogrid[24] = TlPosition(SQ1_3, -SQ1_3, -SQ1_3);
Ogrid[25] = TlPosition(-SQ1_3, -SQ1_3, -SQ1_3);

for (int i = 0; i < 4; ++i) {
Ogrid[26 + 24 * i + 0] = TlPosition(bl[i], bl[i], bm[i]);
Ogrid[26 + 24 * i + 1] = TlPosition(-bl[i], bl[i], bm[i]);
Ogrid[26 + 24 * i + 2] = TlPosition(bl[i], -bl[i], bm[i]);
Ogrid[26 + 24 * i + 3] = TlPosition(bl[i], bl[i], -bm[i]);
Ogrid[26 + 24 * i + 4] = TlPosition(-bl[i], -bl[i], bm[i]);
Ogrid[26 + 24 * i + 5] = TlPosition(-bl[i], bl[i], -bm[i]);
Ogrid[26 + 24 * i + 6] = TlPosition(bl[i], -bl[i], -bm[i]);
Ogrid[26 + 24 * i + 7] = TlPosition(-bl[i], -bl[i], -bm[i]);
}

for (int i = 0; i < 4; ++i) {
for (int j = 0; j < 8; ++j) {
const TlPosition tmp = Ogrid[26 + 24 * i + j];
Ogrid[26 + 24 * i + 8 + j] =
TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[26 + 24 * i + 16 + j] =
TlPosition(tmp.z(), tmp.x(), tmp.y());
}
}

Ogrid[122] = TlPosition(p1, q1, 0.0);
Ogrid[123] = TlPosition(p1, -q1, 0.0);
Ogrid[124] = TlPosition(-p1, q1, 0.0);
Ogrid[125] = TlPosition(-p1, -q1, 0.0);

Ogrid[126] = TlPosition(p1, 0.0, q1);
Ogrid[127] = TlPosition(p1, 0.0, -q1);
Ogrid[128] = TlPosition(-p1, 0.0, q1);
Ogrid[129] = TlPosition(-p1, 0.0, -q1);

Ogrid[130] = TlPosition(0.0, p1, q1);
Ogrid[131] = TlPosition(0.0, p1, -q1);
Ogrid[132] = TlPosition(0.0, -p1, q1);
Ogrid[133] = TlPosition(0.0, -p1, -q1);

Ogrid[134] = TlPosition(q1, p1, 0.0);
Ogrid[135] = TlPosition(q1, -p1, 0.0);
Ogrid[136] = TlPosition(-q1, p1, 0.0);
Ogrid[137] = TlPosition(-q1, -p1, 0.0);

Ogrid[138] = TlPosition(q1, 0.0, p1);
Ogrid[139] = TlPosition(q1, 0.0, -p1);
Ogrid[140] = TlPosition(-q1, 0.0, p1);
Ogrid[141] = TlPosition(-q1, 0.0, -p1);

Ogrid[142] = TlPosition(0.0, q1, p1);
Ogrid[143] = TlPosition(0.0, q1, -p1);
Ogrid[144] = TlPosition(0.0, -q1, p1);
Ogrid[145] = TlPosition(0.0, -q1, -p1);

Ogrid[146 + 0] = TlPosition(dr, du, dw);
Ogrid[146 + 1] = TlPosition(-dr, du, dw);
Ogrid[146 + 2] = TlPosition(dr, -du, dw);
Ogrid[146 + 3] = TlPosition(dr, du, -dw);
Ogrid[146 + 4] = TlPosition(-dr, -du, dw);
Ogrid[146 + 5] = TlPosition(-dr, du, -dw);
Ogrid[146 + 6] = TlPosition(dr, -du, -dw);
Ogrid[146 + 7] = TlPosition(-dr, -du, -dw);

for (int i = 0; i < 8; ++i) {
const TlPosition tmp = Ogrid[146 + i];
Ogrid[146 + 8 + i] = TlPosition(tmp.x(), tmp.z(), tmp.y());
Ogrid[146 + 16 + i] = TlPosition(tmp.y(), tmp.x(), tmp.z());
Ogrid[146 + 24 + i] = TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[146 + 32 + i] = TlPosition(tmp.z(), tmp.y(), tmp.x());
Ogrid[146 + 40 + i] = TlPosition(tmp.z(), tmp.x(), tmp.y());
}
}
} else if ((this->m_gridType == MEDIUM) ||
(this->m_gridType == MEDIUM_FINE)) {
const double A1 = 0.000599631368862;
const double A2 = 0.00737299971862;
const double A3 = 0.00721051536014;
const double B1 = 0.00757439415905;
const double B2 = 0.00675382948631;
const double B3 = 0.00711635549312;
const double D1 = 0.00699108735330;

const double dr = 0.882270011260;
const double du = 0.140355381171;
const double dw = 0.449332832327;

const double CORRP = 1.00;
const double invCORRP = 1.0 / CORRP;

bm[0] = 0.974888643677;
bm[1] = 0.807089818360;
bm[2] = 0.291298882210;

for (int i = 0; i < 6; ++i) {
w[i] *= A1 * 146.0 * invCORRP;
}
for (int i = 6; i < 18; ++i) {
w[i] *= A2 * 146.0 * invCORRP;
}
for (int i = 18; i < 26; ++i) {
w[i] *= A3 * 146.0 * invCORRP;
}
for (int i = 26; i < 50; ++i) {
w[i] *= B1 * 146.0 * invCORRP;
}
for (int i = 50; i < 74; ++i) {
w[i] *= B2 * 146.0 * invCORRP;
}
for (int i = 74; i < 98; ++i) {
w[i] *= B3 * 146.0 * invCORRP;
}
for (int i = 98; i < 146; ++i) {
w[i] *= D1 * 146.0 * invCORRP;
}

Ogrid[0] = TlPosition(1.0, 0.0, 0.0);
Ogrid[1] = TlPosition(0.0, 1.0, 0.0);
Ogrid[2] = TlPosition(0.0, 0.0, 1.0);
Ogrid[3] = TlPosition(-1.0, 0.0, 0.0);
Ogrid[4] = TlPosition(0.0, -1.0, 0.0);
Ogrid[5] = TlPosition(0.0, 0.0, -1.0);

Ogrid[6] = TlPosition(SQ1_2, SQ1_2, 0.0);
Ogrid[7] = TlPosition(-SQ1_2, SQ1_2, 0.0);
Ogrid[8] = TlPosition(SQ1_2, -SQ1_2, 0.0);
Ogrid[9] = TlPosition(-SQ1_2, -SQ1_2, 0.0);

for (int i = 0; i < 4; ++i) {
const TlPosition tmp = Ogrid[6 + i];
Ogrid[6 + 4 + i] = TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[6 + 8 + i] = TlPosition(tmp.z(), tmp.x(), tmp.y());
}

Ogrid[18] = TlPosition(SQ1_3, SQ1_3, SQ1_3);
Ogrid[19] = TlPosition(-SQ1_3, SQ1_3, SQ1_3);
Ogrid[20] = TlPosition(SQ1_3, -SQ1_3, SQ1_3);
Ogrid[21] = TlPosition(SQ1_3, SQ1_3, -SQ1_3);
Ogrid[22] = TlPosition(-SQ1_3, -SQ1_3, SQ1_3);
Ogrid[23] = TlPosition(-SQ1_3, SQ1_3, -SQ1_3);
Ogrid[24] = TlPosition(SQ1_3, -SQ1_3, -SQ1_3);
Ogrid[25] = TlPosition(-SQ1_3, -SQ1_3, -SQ1_3);

for (int i = 0; i < 3; ++i) {
bl[i] = sqrt(1.0 - bm[i] * bm[i]) / SQ2;

Ogrid[26 + 24 * i + 0] = TlPosition(bl[i], bl[i], bm[i]);
Ogrid[26 + 24 * i + 1] = TlPosition(-bl[i], bl[i], bm[i]);
Ogrid[26 + 24 * i + 2] = TlPosition(bl[i], -bl[i], bm[i]);
Ogrid[26 + 24 * i + 3] = TlPosition(bl[i], bl[i], -bm[i]);
Ogrid[26 + 24 * i + 4] = TlPosition(-bl[i], -bl[i], bm[i]);
Ogrid[26 + 24 * i + 5] = TlPosition(-bl[i], bl[i], -bm[i]);
Ogrid[26 + 24 * i + 6] = TlPosition(bl[i], -bl[i], -bm[i]);
Ogrid[26 + 24 * i + 7] = TlPosition(-bl[i], -bl[i], -bm[i]);
}

for (int i = 0; i < 3; ++i) {
for (int j = 0; j < 8; ++j) {
const TlPosition tmp = Ogrid[26 + 24 * i + j];
Ogrid[26 + 24 * i + 8 + j] =
TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[26 + 24 * i + 16 + j] =
TlPosition(tmp.z(), tmp.x(), tmp.y());
}
}

Ogrid[98 + 0] = TlPosition(dr, du, dw);
Ogrid[98 + 1] = TlPosition(-dr, du, dw);
Ogrid[98 + 2] = TlPosition(dr, -du, dw);
Ogrid[98 + 3] = TlPosition(dr, du, -dw);
Ogrid[98 + 4] = TlPosition(-dr, -du, dw);
Ogrid[98 + 5] = TlPosition(-dr, du, -dw);
Ogrid[98 + 6] = TlPosition(dr, -du, -dw);
Ogrid[98 + 7] = TlPosition(-dr, -du, -dw);

for (int i = 0; i < 8; ++i) {
const TlPosition tmp = Ogrid[98 + i];
Ogrid[98 + 8 + i] = TlPosition(tmp.x(), tmp.z(), tmp.y());
Ogrid[98 + 16 + i] = TlPosition(tmp.y(), tmp.x(), tmp.z());
Ogrid[98 + 24 + i] = TlPosition(tmp.y(), tmp.z(), tmp.x());
Ogrid[98 + 32 + i] = TlPosition(tmp.z(), tmp.y(), tmp.x());
Ogrid[98 + 40 + i] = TlPosition(tmp.z(), tmp.x(), tmp.y());
}
}

for (int i = 0; i < nOgrid; ++i) {
TlDenseVector_Lapack v(3);
v.set(0, Ogrid[i][0]);
v.set(1, Ogrid[i][1]);
v.set(2, Ogrid[i][2]);

v = O * v;

Ogrid[i][0] = v.get(0);
Ogrid[i][1] = v.get(1);
Ogrid[i][2] = v.get(2);

Ogrid[i] *= r0;
Ogrid[i] += core;
}
}

void DfGenerateGrid::getRadialAbscissaAndWeight_EulerMaclaurin(
const double R, const double Nr, const int i, double* p_ri,
double* pWeight) {
assert((1 <= i) && (i <= Nr));
const double Nr1i = double(Nr + 1 - i);
const double inv_Nr1i = 1.0 / Nr1i;

const double ri = R * double(i * i) * inv_Nr1i * inv_Nr1i;
*p_ri = ri;

*pWeight = 2.0 * R * R * R * double(Nr + 1) * double(i * i * i * i * i) *
inv_Nr1i * inv_Nr1i * inv_Nr1i * inv_Nr1i * inv_Nr1i * inv_Nr1i *
inv_Nr1i * 4.0 * M_PI;
}

void DfGenerateGrid::getRadialAbscissaAndWeight_GaussChebyshev(
const double R, const int Nr, const int i, double* p_ri, double* pWeight) {
assert((1 <= i) && (i <= Nr));
static const double invLn2 = 1.0 / std::log(2.0);

const double Nr1 = Nr + 1;
const double tmp = double(i) * M_PI / Nr1;
const double s = std::sin(tmp);
const double ss = s * s;
const double c = std::cos(tmp);

const double t1 = (Nr1 - 2.0 * double(i)) / Nr1;
const double t2 = (2.0 / M_PI) * (1.0 + (2.0 / 3.0) * ss) * c * s;

const double x = t1 + t2;

double ri = 0.0;
double jacobian = 0.0;
const double inv_x1 = 1.0 / (x - 1.0);
switch (this->GC_mappingType_) {
case GC_BECKE:
ri = R * (1.0 + x) / (1.0 - x);
jacobian = R * ((x + 1.0) * inv_x1 * inv_x1 - inv_x1);
break;

case GC_TA:
ri = invLn2 * std::log(2.0 * (-inv_x1)) * R;
jacobian = R * invLn2 / (1.0 - x);
break;

case GC_KK:
ri = invLn2 * std::log(2.0 * (-inv_x1));
jacobian = invLn2 / (1.0 - x);
break;

default:
this->log_.critical("unknown GC_MAPPING_TYPE");
break;
}
*p_ri = ri;

*pWeight = (16.0 / (3.0 * Nr1)) * ss * ss * jacobian * ri * ri * 4.0 * M_PI;
}

int DfGenerateGrid::getNumOfPrunedAnglarPoints(const double r,
const int maxNumOfAngGrids,
const int atomicNumber) {
int n_theta = maxNumOfAngGrids;
if (this->isPruning_) {
static const double K_theta = 5.0;  
const double r_Bragg = TlPrdctbl::getBraggSlaterRadii(atomicNumber);

n_theta = std::min<int>((K_theta * r / r_Bragg) * maxNumOfAngGrids,
maxNumOfAngGrids);
}

std::vector<int> supportedGrids = this->lebGrd_.getSupportedGridNumber();
std::vector<int>::const_iterator it =
std::upper_bound(supportedGrids.begin(), supportedGrids.end(), n_theta);
int numOfGrid = supportedGrids[0];
if (it != supportedGrids.begin()) {
numOfGrid = *(--it);
}

if (maxNumOfAngGrids != n_theta) {
this->log_.debug(
TlUtils::format("pruned: %d -> %d", n_theta, numOfGrid));
}
return numOfGrid;
}

int DfGenerateGrid::getNumOfPrunedAnglarPoints_SG1(
const double r, const double inv_R, const std::vector<double>& alpha) {
const double judge = r * inv_R;  

int numOfGrid = 86;
if (judge < alpha[1]) {
if (judge < alpha[0]) {
numOfGrid = 6;
} else {
numOfGrid = 38;
}
} else {
if (judge < alpha[2]) {
numOfGrid = 86;
} else {
if (judge < alpha[3]) {
numOfGrid = 194;
}
}
}

return numOfGrid;
}

void DfGenerateGrid::getSphericalGrids(const int numOfGrids, const double r,
const double radial_weight,
const TlPosition& center,
const TlDenseGeneralMatrix_Lapack& O,
std::vector<TlPosition>* pGrids,
std::vector<double>* pWeights) {
this->lebGrd_.getGrids(numOfGrids, pGrids, pWeights);
assert(static_cast<int>(pGrids->size()) == numOfGrids);
assert(static_cast<int>(pWeights->size()) == numOfGrids);

for (int i = 0; i < numOfGrids; ++i) {
(*pGrids)[i] = O * (*pGrids)[i];
(*pGrids)[i] *= r;
(*pGrids)[i] += center;

(*pWeights)[i] *= radial_weight;
}
}

TlDenseGeneralMatrix_Lapack DfGenerateGrid::getOMatrix() {
const int numOfAtoms = this->flGeometry_.getNumOfAtoms();

TlPosition sum_zr(0.0, 0.0, 0.0);
double sum_z = 0.0;
for (int atom = 0; atom < numOfAtoms; ++atom) {
const std::string symbol = this->flGeometry_.getAtomSymbol(atom);
if (symbol == "X") {
continue;
}

const double z = this->flGeometry_.getCharge(atom);
const TlPosition r = this->flGeometry_.getCoordinate(atom);
sum_zr += z * r;
sum_z += z;
}
sum_zr /= sum_z;
TlDenseVector_Lapack T(3);
T.set(0, sum_zr[0]);
T.set(1, sum_zr[1]);
T.set(2, sum_zr[2]);

TlDenseSymmetricMatrix_Lapack M(3);
TlDenseSymmetricMatrix_Lapack I(3);
I.set(0, 0, 1.0);
I.set(1, 1, 1.0);
I.set(2, 2, 1.0);
#pragma omp parallel for
for (int atom = 0; atom < numOfAtoms; ++atom) {
const std::string symbol = this->flGeometry_.getAtomSymbol(atom);
if (symbol == "X") {
continue;
}

const double z = this->flGeometry_.getCharge(atom);
const TlPosition p = this->flGeometry_.getCoordinate(atom);
TlDenseVector_Lapack R(3);
R.set(0, p[0]);
R.set(1, p[1]);
R.set(2, p[2]);

const TlDenseVector_Lapack RT = R - T;
const double RT2 = RT.norm2();

TlDenseGeneralMatrix_Lapack mRT(3, 1);
mRT.set(0, 0, RT.get(0));
mRT.set(1, 0, RT.get(1));
mRT.set(2, 0, RT.get(2));
TlDenseGeneralMatrix_Lapack mRTt = mRT;
mRTt.transposeInPlace();
const TlDenseSymmetricMatrix_Lapack RTRT = mRT * mRTt;
assert(RTRT.getNumOfRows() == 3);
assert(RTRT.getNumOfCols() == 3);

M += z * (RT2 * I - RTRT);
}

TlDenseGeneralMatrix_Lapack O;
TlDenseVector_Lapack lambda;
M.eig(&lambda, &O);


return O;
}

void DfGenerateGrid::calcMultiCenterWeight_Becke(
const int iAtom, const int Ogrid, const std::vector<TlPosition>& grids,
std::vector<double>* pWeights) {
const int numOfAtoms = this->m_nNumOfAtoms;
const int numOfGrids = grids.size();
assert(numOfGrids == static_cast<int>(pWeights->size()));

for (int Omega = 0; Omega < numOfGrids; ++Omega) {
const TlPosition pos_O = grids[Omega];


std::vector<double> Ps(numOfAtoms);
for (int m = 0; m < numOfAtoms; ++m) {







Ps[m] = this->Ps_uij(m, pos_O);
}

const double Ps_A = Ps[iAtom];
double Ps_total = 0.0;
for (int i = 0; i < numOfAtoms; ++i) {
Ps_total += Ps[i];
}

(*pWeights)[Omega] *= (Ps_A / Ps_total);
}
}

void DfGenerateGrid::calcMultiCenterWeight_SS(
const int iAtom, const int Ogrid, const std::vector<TlPosition>& grids,
std::vector<double>* pWeights) {
const int numOfAtoms = this->m_nNumOfAtoms;
const int numOfGrids = grids.size();
assert(numOfGrids == static_cast<int>(pWeights->size()));

for (int Omega = 0; Omega < numOfGrids; ++Omega) {
const TlPosition pos_O = grids[Omega];

std::vector<double> rr(numOfAtoms);
for (int p = 0; p < numOfAtoms; ++p) {
rr[p] = pos_O.distanceFrom(this->coord_[p]);
}

std::vector<double> Ps(numOfAtoms);
for (int m = 0; m < numOfAtoms; ++m) {
double Psuij = 1.0;

for (int n = 0; n < numOfAtoms; ++n) {
if (m != n) {

double u_ij =
(rr[m] - rr[n]) * this->invDistanceMatrix_.get(m, n);



double g = 0.0;
static const double a = 0.64;
if (u_ij < -a) {
g = -1.0;
} else if (u_ij > a) {
g = 1.0;
} else {
const double ua = u_ij / a;
const double ua2 = ua * ua;
const double ua3 = ua2 * ua;
const double ua5 = ua2 * ua3;
g = (1.0 / 16.0) * (35.0 * ua - 35.0 * ua3 +
21.0 * ua5 - 5.0 * ua5 * ua2);
}
const double s = 0.5 * (1.0 - g);

Psuij *= s;
}
}

Ps[m] = Psuij;
}

const double Ps_A = Ps[iAtom];
double Ps_total = 0.0;
for (int i = 0; i < numOfAtoms; ++i) {
Ps_total += Ps[i];
}

(*pWeights)[Omega] *= (Ps_A / Ps_total);
}
}

double DfGenerateGrid::getCovalentRadiiForBecke(const int atomicNumber) {
double answer = 0.0;
if (atomicNumber == 1) {
answer = 0.35;
} else {
answer = TlPrdctbl::getBraggSlaterRadii(atomicNumber);
}

return answer;
}

double DfGenerateGrid::Becke_f1(const double x) {
const double ans = 0.5 * x * (3.0 - x * x);
return ans;
}

double DfGenerateGrid::Becke_f3(const double x) {
return this->Becke_f1(this->Becke_f1(this->Becke_f1(x)));
}

void DfGenerateGrid::screeningGridsByWeight0(std::vector<TlPosition>* pGrids,
std::vector<double>* pWeights) {
const double threshold = this->weightCutoff_;
const int numOfGrids = pGrids->size();
assert(numOfGrids == static_cast<int>(pWeights->size()));

std::vector<TlPosition> tmpGrids(numOfGrids);
std::vector<double> tmpWeights(numOfGrids);
int count = 0;
for (int i = 0; i < numOfGrids; ++i) {
const double weight = (*pWeights)[i];
if (std::fabs(weight) > threshold) {
tmpGrids[count] = (*pGrids)[i];
tmpWeights[count] = weight;
++count;
}
}

tmpGrids.resize(count);
tmpWeights.resize(count);
std::vector<TlPosition>(tmpGrids).swap(tmpGrids);
std::vector<double>(tmpWeights).swap(tmpWeights);

*pGrids = tmpGrids;
*pWeights = tmpWeights;
}

void DfGenerateGrid::screeningGridsByWeight(
std::vector<TlPosition>* pGrids, std::vector<double>* pSingleCenterWeights,
std::vector<double>* pPartitioningWeights) {
const double threshold = this->weightCutoff_;
const int numOfGrids = pGrids->size();
assert(numOfGrids == static_cast<int>(pSingleCenterWeights->size()));
assert(numOfGrids == static_cast<int>(pPartitioningWeights->size()));

std::vector<TlPosition> tmpGrids(numOfGrids);
std::vector<double> tmpSingleCenterWeights(numOfGrids);
std::vector<double> tmpPartitioningWeights(numOfGrids);
int count = 0;
for (int i = 0; i < numOfGrids; ++i) {
const double singleCenterWeight = (*pSingleCenterWeights)[i];
const double partitioningWeight = (*pPartitioningWeights)[i];
const double weight = singleCenterWeight * partitioningWeight;
if (std::fabs(weight) > threshold) {
tmpGrids[count] = (*pGrids)[i];
tmpSingleCenterWeights[count] = singleCenterWeight;
tmpPartitioningWeights[count] = partitioningWeight;
++count;
}
}

tmpGrids.resize(count);
tmpSingleCenterWeights.resize(count);
tmpPartitioningWeights.resize(count);
std::vector<TlPosition>(tmpGrids).swap(tmpGrids);
std::vector<double>(tmpSingleCenterWeights).swap(tmpSingleCenterWeights);
std::vector<double>(tmpPartitioningWeights).swap(tmpPartitioningWeights);

*pGrids = tmpGrids;
*pSingleCenterWeights = tmpSingleCenterWeights;
*pPartitioningWeights = tmpPartitioningWeights;
}

double DfGenerateGrid::Ps_uij(const int atomIndex_m,
const TlPosition& gridpoint) {
const int numOfAtoms = this->m_nNumOfAtoms;
const int atomicNumber_m =
TlAtom::getElementNumber(this->flGeometry_.getAtomSymbol(atomIndex_m));
const double R_m = this->getCovalentRadiiForBecke(atomicNumber_m);
const double rr_m = gridpoint.distanceFrom(this->coord_[atomIndex_m]);

double Ps_uij = 1.0;
for (int atomIndex_n = 0; atomIndex_n < numOfAtoms; ++atomIndex_n) {
if (atomIndex_m != atomIndex_n) {
const int atomicNumber_n = TlAtom::getElementNumber(
this->flGeometry_.getAtomSymbol(atomIndex_n));
if (atomicNumber_n > 0) {
const double R_n =
this->getCovalentRadiiForBecke(atomicNumber_n);
const double rr_n =
gridpoint.distanceFrom(this->coord_[atomIndex_n]);

double u_ij = (rr_m - rr_n) * this->invDistanceMatrix_.get(
atomIndex_m, atomIndex_n);

if (this->isAtomicSizeAdjustments_) {
double au_ij = (R_m - R_n) / (R_m + R_n);
double a = au_ij / (au_ij * au_ij - 1.0);

if (a < -0.50) {
a = -0.50;
} else if (a > 0.50) {
a = 0.50;
}
u_ij = u_ij + a * (1.0 - u_ij * u_ij);
}

const double f3 = this->Becke_f3(u_ij);
const double s = 0.5 * (1.0 - f3);

Ps_uij *= s;
}
}
}

return Ps_uij;
}

void DfGenerateGrid::getGrids(const int atomIndex,
std::vector<TlPosition>* pGrids,
std::vector<double>* pSingleCenterWeights,
std::vector<double>* pPartitioningWeights) {
const int gridType = SG_1;
const int numOfRadialGrids = 50;
const int radialGridType = RG_EularMaclaurin;
const int maxAngularGrids = 196;
const PARTITIONING_METHOD partitioningMethod = Paritioning_Becke;
this->getGrids_sub(atomIndex, gridType, numOfRadialGrids, radialGridType,
maxAngularGrids, partitioningMethod, pGrids,
pSingleCenterWeights, pPartitioningWeights);
}

void DfGenerateGrid::getGrids_sub(const int atomIndex, const int gridType,
const int numOfRadialGrids,
const int radialGridType,
const int maxAngularGrids,
const PARTITIONING_METHOD partitioningMethod,
std::vector<TlPosition>* pGrids,
std::vector<double>* pSingleCenterWeights,
std::vector<double>* pPartitioningWeights) {
assert(pGrids != NULL);
assert(pSingleCenterWeights != NULL);
assert(pPartitioningWeights != NULL);

const int atomicNumber =
TlPrdctbl::getAtomicNumber(this->flGeometry_.getAtomSymbol(atomIndex));
if (atomicNumber != 0) {
int numOfGrids = 0;
{
const int maxGrids = numOfRadialGrids * maxAngularGrids;
pGrids->resize(maxGrids);
pSingleCenterWeights->resize(maxGrids);
pPartitioningWeights->resize(maxGrids);
}

double rM = 0.0;
if (gridType == SG_1) {
rM = this->radiusList_[atomicNumber];
} else {
rM = TlPrdctbl::getBraggSlaterRadii(atomicNumber);
}
const double inv_rM = 1.0 / rM;

std::vector<double> alpha(5);
alpha[0] = 0.0;
alpha[1] = 0.0;
alpha[2] = 0.0;
alpha[3] = 10000.0;
if ((atomicNumber == 1) || (atomicNumber == 2)) {
alpha[0] = 0.2500;
alpha[1] = 0.5000;
alpha[2] = 1.0000;
alpha[3] = 4.5000;
} else if ((3 <= atomicNumber) && (atomicNumber <= 10)) {
alpha[0] = 0.1667;
alpha[1] = 0.5000;
alpha[2] = 0.9000;
alpha[3] = 3.5000;
} else if ((11 <= atomicNumber) && (atomicNumber <= 18)) {
alpha[0] = 0.1000;
alpha[1] = 0.4000;
alpha[2] = 0.8000;
alpha[3] = 2.5000;
}

for (int radvec = 0; radvec < numOfRadialGrids; ++radvec) {
const int i = radvec + 1;

double ri = 0.0;
double wr = 0.0;
switch (radialGridType) {
case RG_GaussChebyshev:
this->getRadialAbscissaAndWeight_GaussChebyshev(
rM, numOfRadialGrids, i, &ri, &wr);
break;

case RG_EularMaclaurin:
this->getRadialAbscissaAndWeight_EulerMaclaurin(
rM, numOfRadialGrids, i, &ri, &wr);
break;

default:
this->log_.critical("unknown radial grid type.");
break;
}

if (ri > 30.0) {
continue;
}

int numOfAngularGrids = 0;
if (gridType == SG_1) {
numOfAngularGrids =
this->getNumOfPrunedAnglarPoints_SG1(ri, inv_rM, alpha);
} else {
numOfAngularGrids = this->getNumOfPrunedAnglarPoints(
ri, maxAngularGrids, atomicNumber);
}

std::vector<TlPosition> angularGrids(numOfAngularGrids);
std::vector<double> lebWeights(numOfAngularGrids);
this->getSphericalGrids(numOfAngularGrids, ri, wr,
this->coord_[atomIndex], this->O_,
&angularGrids, &lebWeights);
assert(angularGrids.size() ==
static_cast<std::size_t>(numOfAngularGrids));
assert(lebWeights.size() ==
static_cast<std::size_t>(numOfAngularGrids));

std::vector<double> partWeights(numOfAngularGrids, 1.0);
switch (partitioningMethod) {
case Paritioning_Becke:
this->calcMultiCenterWeight_Becke(
atomIndex, numOfAngularGrids, angularGrids,
&partWeights);
break;

case Partitioning_SSWeight:
this->calcMultiCenterWeight_SS(atomIndex, numOfAngularGrids,
angularGrids, &partWeights);
break;

default:
this->log_.critical("unknown partitioning method type.");
break;
}

std::copy(angularGrids.begin(), angularGrids.end(),
pGrids->begin() + numOfGrids);
std::copy(lebWeights.begin(), lebWeights.end(),
pSingleCenterWeights->begin() + numOfGrids);
std::copy(partWeights.begin(), partWeights.end(),
pPartitioningWeights->begin() + numOfGrids);
numOfGrids += numOfAngularGrids;
}

{
const int numOfGrids_orig = numOfGrids;
this->screeningGridsByWeight(pGrids, pSingleCenterWeights,
pPartitioningWeights);
numOfGrids = pGrids->size();
assert(pSingleCenterWeights->size() ==
static_cast<std::size_t>(numOfGrids));
assert(pPartitioningWeights->size() ==
static_cast<std::size_t>(numOfGrids));
{
const int diff = numOfGrids_orig - numOfGrids;
const double ratio =
double(diff) / double(numOfGrids_orig) * 100.0;
this->log_.info(
TlUtils::format("screened grids: %d -> %d; (%d; %3.2f%%)",
numOfGrids_orig, numOfGrids, diff, ratio));
}
}
}
}

TlDenseVector_Lapack DfGenerateGrid::JGP_nablaB_omegaA(
const int atomIndexA, const int atomIndexB, const TlPosition& gridpoint) {
TlDenseVector_Lapack nablaB_omegaA(3);
if (atomIndexA != atomIndexB) {
const int numOfAtoms = this->m_nNumOfAtoms;
double PA = 0.0;
double Z = 0.0;
for (int i = 0; i < numOfAtoms; ++i) {
const double Ps = this->Ps_uij(i, gridpoint);
if (i == atomIndexA) {
PA = Ps;
}
Z += Ps;
}
const double invZ = 1.0 / Z;

const TlDenseVector_Lapack nablaB_PA =
this->JGP_nablaB_PA(atomIndexA, atomIndexB, gridpoint);

TlDenseVector_Lapack nablaB_Z(3);
for (int i = 0; i < numOfAtoms; ++i) {
TlDenseVector_Lapack nablaB_Pi(3);
if (i == atomIndexB) {
nablaB_Pi = this->JGP_nablaA_PA(atomIndexB, gridpoint);
} else {
nablaB_Pi = this->JGP_nablaB_PA(i, atomIndexB, gridpoint);
}
nablaB_Z += nablaB_Pi;
}

nablaB_omegaA = (nablaB_PA - PA * invZ * nablaB_Z) * invZ;
}

return nablaB_omegaA;
}

TlDenseVector_Lapack DfGenerateGrid::JGP_nablaA_PA(
const int atomIndexA, const TlPosition& gridpoint) {
TlDenseVector_Lapack nablaA_PA(3);
const double PA = this->Ps_uij(atomIndexA, gridpoint);

if (PA > 1.0E-10) {
const int numOfAtoms = this->m_nNumOfAtoms;
for (int atomIndexB = 0; atomIndexB < numOfAtoms; ++atomIndexB) {
if (atomIndexB != atomIndexA) {
const TlDenseVector_Lapack nablaA_myuAB =
this->JGP_nablaA_myuAB(atomIndexA, atomIndexB, gridpoint);

const double rr_A =
this->coord_[atomIndexA].distanceFrom(gridpoint);
const double rr_B =
this->coord_[atomIndexB].distanceFrom(gridpoint);
const double myu_AB =
(rr_A - rr_B) *
this->invDistanceMatrix_.get(atomIndexA, atomIndexB);
const double t = this->JGP_t(myu_AB);

nablaA_PA += t * nablaA_myuAB;
}
}

nablaA_PA *= PA;
}

return nablaA_PA;
}

TlDenseVector_Lapack DfGenerateGrid::JGP_nablaB_PA(
const int atomIndexA, const int atomIndexB, const TlPosition& gridpoint) {
assert(atomIndexA != atomIndexB);
TlDenseVector_Lapack nablaB_PA(3);

const double PA = this->Ps_uij(atomIndexA, gridpoint);
if (PA > 1.0E-10) {
const TlDenseVector_Lapack nablaB_myuBA =
this->JGP_nablaA_myuAB(atomIndexB, atomIndexA, gridpoint);

const double rr_A = this->coord_[atomIndexA].distanceFrom(gridpoint);
const double rr_B = this->coord_[atomIndexB].distanceFrom(gridpoint);
const double myu_AB = (rr_A - rr_B) * this->invDistanceMatrix_.get(
atomIndexA, atomIndexB);
const double t = this->JGP_t(myu_AB);

nablaB_PA = -PA * t * nablaB_myuBA;
}

return nablaB_PA;
}

double DfGenerateGrid::JGP_t(const double myu) {
static const double coef = -27.0 / 16.0;
double answer = 0.0;

const double p1 = this->Becke_f1(myu);
const double p2 = this->Becke_f1(p1);
const double p3 = this->Becke_f1(p2);
const double s = 0.5 * (1.0 - p3);  

assert(std::fabs(s) > 1.0E-16);
answer = coef * (1.0 - p2 * p2) * (1.0 - p1 * p1) * (1 - myu * myu) / s;

return answer;
}

TlDenseVector_Lapack DfGenerateGrid::JGP_nablaA_myuAB(
const int atomIndexA, const int atomIndexB, const TlPosition& gridpoint) {
const double inv_R_AB =
this->invDistanceMatrix_.get(atomIndexA, atomIndexB);

const TlPosition v_rA = this->coord_[atomIndexA] - gridpoint;
const double rA = v_rA.distanceFrom();
const TlPosition u_A = v_rA / rA;

const double rB = this->coord_[atomIndexB].distanceFrom(gridpoint);
TlPosition u_AB = this->coord_[atomIndexB] - this->coord_[atomIndexA];
u_AB.unit();

TlPosition nablaA_myuAB = inv_R_AB * (u_A - ((rA - rB) * inv_R_AB) * u_AB);
TlDenseVector_Lapack answer(3);
for (int i = 0; i < 3; ++i) {
answer.set(i, nablaA_myuAB[i]);
}

return answer;
}
