#include "Nutrient_Transformation.h"

#include "text.h"

Nutrient_Transformation::Nutrient_Transformation() :
m_cellWth(-1.), m_nCells(-1), m_cellAreaFr(NODATA_VALUE),
m_nSoilLyrs(nullptr), m_maxSoilLyrs(-1), m_cbnModel(0), m_solP_model(0),
m_phpApldDays(nullptr), m_phpDefDays(nullptr),
m_tillSwitch(nullptr), m_tillDepth(nullptr), m_tillDays(nullptr), m_tillFactor(nullptr),
m_minrlCoef(-1.), m_orgNFrActN(-1.), m_denitThres(-1.), m_phpSorpIdxBsn(-1.),
m_phpSorpIdx(nullptr), m_psp_store(nullptr), m_ssp_store(nullptr), m_denitCoef(-1.), m_landCover(nullptr),
m_pltRsdDecCoef(nullptr), m_rsdCovSoil(nullptr), m_rsdInitSoil(nullptr), m_soilTemp(nullptr),
m_soilBD(nullptr), m_soilMass(nullptr), m_soilCbn(nullptr),
m_soilWtrSto(nullptr), m_soilFC(nullptr), m_soilDepth(nullptr),
m_soilClay(nullptr), m_soilRock(nullptr), m_soilThk(nullptr),
m_soilActvOrgN(nullptr), m_soilFrshOrgN(nullptr), m_soilFrshOrgP(nullptr),
m_soilActvMinP(nullptr), m_soilStabMinP(nullptr), m_soilSat(nullptr), m_soilPor(nullptr),
m_soilSand(nullptr), m_sol_WOC(nullptr),
m_sol_WON(nullptr), m_sol_BM(nullptr), m_sol_BMC(nullptr),
m_sol_BMN(nullptr), m_sol_HP(nullptr), m_sol_HS(nullptr), m_sol_HSC(nullptr), m_sol_HSN(nullptr),
m_sol_HPC(nullptr), m_sol_HPN(nullptr), m_sol_LM(nullptr), m_sol_LMC(nullptr), m_sol_LMN(nullptr),
m_sol_LSC(nullptr), m_sol_LSN(nullptr), m_sol_LS(nullptr), m_sol_LSL(nullptr), m_sol_LSLC(nullptr),
m_sol_LSLNC(nullptr), m_sol_RNMN(nullptr),
m_sol_RSPC(nullptr), m_hmntl(nullptr), m_hmptl(nullptr), m_rmn2tl(nullptr),
m_rmptl(nullptr), m_rwntl(nullptr), m_wdntl(nullptr), m_rmp1tl(nullptr), m_roctl(nullptr),
m_soilNO3(nullptr), m_soilStabOrgN(nullptr), m_soilHumOrgP(nullptr), m_soilRsd(nullptr), m_soilSolP(nullptr),
m_soilNH4(nullptr), m_soilWP(nullptr), m_wshd_dnit(-1.), m_wshd_hmn(-1.), m_wshd_hmp(-1.),
m_wshd_rmn(-1.), m_wshd_rmp(-1.), m_wshd_rwn(-1.), m_wshd_nitn(-1.), m_wshd_voln(-1.),
m_wshd_pal(-1.), m_wshd_pas(-1.),
m_conv_wt(nullptr), m_conv_wt_reverse(nullptr) {
}

Nutrient_Transformation::~Nutrient_Transformation() {
if (m_hmntl != nullptr) Release1DArray(m_hmntl);
if (m_hmptl != nullptr) Release1DArray(m_hmptl);
if (m_rmn2tl != nullptr) Release1DArray(m_rmn2tl);
if (m_rmptl != nullptr) Release1DArray(m_rmptl);
if (m_rwntl != nullptr) Release1DArray(m_rwntl);
if (m_wdntl != nullptr) Release1DArray(m_wdntl);
if (m_rmp1tl != nullptr) Release1DArray(m_rmp1tl);
if (m_roctl != nullptr) Release1DArray(m_roctl);
if (m_phpApldDays != nullptr) Release1DArray(m_phpApldDays);
if (m_phpDefDays != nullptr) Release1DArray(m_phpDefDays);
if (m_soilMass != nullptr) Release2DArray(m_soilMass);
if (m_sol_WOC != nullptr) Release2DArray(m_sol_WOC);
if (m_sol_WON != nullptr) Release2DArray(m_sol_WON);
if (m_sol_BM != nullptr) Release2DArray(m_sol_BM);
if (m_sol_BMC != nullptr) Release2DArray(m_sol_BMC);
if (m_sol_BMN != nullptr) Release2DArray(m_sol_BMN);
if (m_sol_HP != nullptr) Release2DArray(m_sol_HP);
if (m_sol_HS != nullptr) Release2DArray(m_sol_HS);
if (m_sol_HSC != nullptr) Release2DArray(m_sol_HSC);
if (m_sol_HSN != nullptr) Release2DArray(m_sol_HSN);
if (m_sol_HPC != nullptr) Release2DArray(m_sol_HPC);
if (m_sol_HPN != nullptr) Release2DArray(m_sol_HPN);
if (m_sol_LM != nullptr) Release2DArray(m_sol_LM);
if (m_sol_LMC != nullptr) Release2DArray(m_sol_LMC);
if (m_sol_LMN != nullptr) Release2DArray(m_sol_LMN);
if (m_sol_LSC != nullptr) Release2DArray(m_sol_LSC);
if (m_sol_LSN != nullptr) Release2DArray(m_sol_LSN);
if (m_sol_LS != nullptr) Release2DArray(m_sol_LS);
if (m_sol_LSL != nullptr) Release2DArray(m_sol_LSL);
if (m_sol_LSLC != nullptr) Release2DArray(m_sol_LSLC);
if (m_sol_LSLNC != nullptr) Release2DArray(m_sol_LSLNC);
if (m_sol_RNMN != nullptr) Release2DArray(m_sol_RNMN);
if (m_sol_RSPC != nullptr) Release2DArray(m_sol_RSPC);

if (m_conv_wt != nullptr) Release2DArray(m_conv_wt);
if (m_conv_wt_reverse != nullptr) Release2DArray(m_conv_wt_reverse);
}

bool Nutrient_Transformation::CheckInputData() {
CHECK_POSITIVE(M_NUTR_TF[0], m_nCells);
CHECK_POSITIVE(M_NUTR_TF[0], m_cellAreaFr);
CHECK_POSITIVE(M_NUTR_TF[0], m_maxSoilLyrs);
CHECK_POSITIVE(M_NUTR_TF[0], m_cellWth);
CHECK_POINTER(M_NUTR_TF[0], m_nSoilLyrs);
CHECK_POSITIVE(M_NUTR_TF[0], m_minrlCoef);
CHECK_POSITIVE(M_NUTR_TF[0], m_denitCoef);
CHECK_POSITIVE(M_NUTR_TF[0], m_orgNFrActN);
CHECK_POSITIVE(M_NUTR_TF[0], m_denitThres);
CHECK_POSITIVE(M_NUTR_TF[0], m_phpSorpIdxBsn);
CHECK_POINTER(M_NUTR_TF[0], m_landCover);
CHECK_POINTER(M_NUTR_TF[0], m_soilClay);
CHECK_POINTER(M_NUTR_TF[0], m_soilDepth);
CHECK_POINTER(M_NUTR_TF[0], m_rsdInitSoil);
CHECK_POINTER(M_NUTR_TF[0], m_soilThk);
CHECK_POINTER(M_NUTR_TF[0], m_soilBD);
CHECK_POINTER(M_NUTR_TF[0], m_pltRsdDecCoef);
CHECK_POINTER(M_NUTR_TF[0], m_soilCbn);
CHECK_POINTER(M_NUTR_TF[0], m_soilFC);
CHECK_POINTER(M_NUTR_TF[0], m_soilWP);
CHECK_POINTER(M_NUTR_TF[0], m_soilNO3);
CHECK_POINTER(M_NUTR_TF[0], m_soilNH4);
CHECK_POINTER(M_NUTR_TF[0], m_soilStabOrgN);
CHECK_POINTER(M_NUTR_TF[0], m_soilHumOrgP);
CHECK_POINTER(M_NUTR_TF[0], m_soilSolP);
CHECK_POINTER(M_NUTR_TF[0], m_soilWtrSto);
CHECK_POINTER(M_NUTR_TF[0], m_soilTemp);
CHECK_POINTER(M_NUTR_TF[0], m_soilSat);

if (!(m_cbnModel == 0 || m_cbnModel == 1 || m_cbnModel == 2)) {
throw ModelException(M_NUTR_TF[0], "CheckInputData",
"Carbon modeling method must be 0, 1, or 2.");
}
return true;
}

void Nutrient_Transformation::SetValue(const char* key, const FLTPT value) {
string sk(key);
if (StringMatch(sk, Tag_CellWidth[0])) {
m_cellWth = value;
} else if (StringMatch(sk, VAR_NACTFR[0])) {
m_orgNFrActN = value;
} else if (StringMatch(sk, VAR_SDNCO[0])) {
m_denitThres = value;
} else if (StringMatch(sk, VAR_CMN[0])) {
m_minrlCoef = value;
} else if (StringMatch(sk, VAR_CDN[0])) {
m_denitCoef = value;
} else if (StringMatch(sk, VAR_PSP[0])) {
m_phpSorpIdxBsn = value;
} else {
throw ModelException(M_NUTR_TF[0], "SetValue",
"Parameter " + sk + " does not exist.");
}
}

void Nutrient_Transformation::SetValue(const char* key, const int value) {
string sk(key);
if (StringMatch(sk, VAR_CSWAT[0])) {
m_cbnModel = value;
} else {
throw ModelException(M_NUTR_TF[0], "SetValue",
"Integer Parameter " + sk + " does not exist.");
}
}

void Nutrient_Transformation::Set1DData(const char* key, const int n, FLTPT* data) {
CheckInputSize(M_NUTR_TF[0], key, n, m_nCells);
string sk(key);
if (StringMatch(sk, VAR_PL_RSDCO[0])) {
m_pltRsdDecCoef = data;
} else if (StringMatch(sk, VAR_SOL_RSDIN[0])) {
m_rsdInitSoil = data;
} else if (StringMatch(sk, VAR_SOL_COV[0])) {
m_rsdCovSoil = data;
} else if (StringMatch(sk, VAR_SOTE[0])) {
m_soilTemp = data;
}
else if (StringMatch(sk, VAR_TILLAGE_DAYS[0])) {
m_tillDays = data;
} else if (StringMatch(sk, VAR_TILLAGE_DEPTH[0])) {
m_tillDepth = data;
} else if (StringMatch(sk, VAR_TILLAGE_FACTOR[0])) {
m_tillFactor = data;
} else {
throw ModelException(M_NUTR_TF[0], "Set1DData",
"Parameter " + sk + " does not exist.");
}
}
void Nutrient_Transformation::Set1DData(const char* key, const int n, int* data) {
CheckInputSize(M_NUTR_TF[0], key, n, m_nCells);
string sk(key);
if (StringMatch(sk, VAR_LANDCOVER[0])) {
m_landCover = data;
} else if (StringMatch(sk, VAR_SOILLAYERS[0])) {
m_nSoilLyrs = data;
} else if (StringMatch(sk, VAR_TILLAGE_SWITCH[0])) {
m_tillSwitch = data;
} else {
throw ModelException(M_NUTR_TF[0], "Set1DData",
"Integer Parameter " + sk + " does not exist.");
}
}
void Nutrient_Transformation::Set2DData(const char* key, const int nrows, const int ncols, FLTPT** data) {
CheckInputSize2D(M_NUTR_TF[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
string sk(key);
if (StringMatch(sk, VAR_SOL_CBN[0])) {
m_soilCbn = data;
} else if (StringMatch(sk, VAR_SOL_BD[0])) {
m_soilBD = data;
} else if (StringMatch(sk, VAR_CLAY[0])) {
m_soilClay = data;
} else if (StringMatch(sk, VAR_ROCK[0])) {
m_soilRock = data;
} else if (StringMatch(sk, VAR_SOL_ST[0])) {
m_soilWtrSto = data;
} else if (StringMatch(sk, VAR_SOL_AWC[0])) {
m_soilFC = data;
} else if (StringMatch(sk, VAR_SOL_NO3[0])) {
m_soilNO3 = data;
} else if (StringMatch(sk, VAR_SOL_NH4[0])) {
m_soilNH4 = data;
} else if (StringMatch(sk, VAR_SOL_SORGN[0])) {
m_soilStabOrgN = data;
} else if (StringMatch(sk, VAR_SOL_HORGP[0])) {
m_soilHumOrgP = data;
} else if (StringMatch(sk, VAR_SOL_SOLP[0])) {
m_soilSolP = data;
} else if (StringMatch(sk, VAR_SOL_WPMM[0])) {
m_soilWP = data;
} else if (StringMatch(sk, VAR_SOILDEPTH[0])) {
m_soilDepth = data;
} else if (StringMatch(sk, VAR_SOILTHICK[0])) {
m_soilThk = data;
} else if (StringMatch(sk, VAR_SOL_RSD[0])) {
m_soilRsd = data;
} else if (StringMatch(sk, VAR_SOL_UL[0])) {
m_soilSat = data;
} else if (StringMatch(sk, VAR_POROST[0])) {
m_soilPor = data;
} else if (StringMatch(sk, VAR_SAND[0])) {
m_soilSand = data;
} else {
throw ModelException(M_NUTR_TF[0], "Set2DData",
"Parameter " + sk + " does not exist.");
}
}

void Nutrient_Transformation::InitialOutputs() {
CHECK_POSITIVE(M_NUTR_TF[0], m_nCells);
if (m_cellAreaFr < 0.) m_cellAreaFr = 1. / m_nCells;
CHECK_POSITIVE(M_NUTR_TF[0], m_maxSoilLyrs);
if (m_hmntl == nullptr) Initialize1DArray(m_nCells, m_hmntl, 0.);
if (m_hmptl == nullptr) Initialize1DArray(m_nCells, m_hmptl, 0.);
if (m_rmn2tl == nullptr) Initialize1DArray(m_nCells, m_rmn2tl, 0.);
if (m_rmptl == nullptr) Initialize1DArray(m_nCells, m_rmptl, 0.);
if (m_rwntl == nullptr) Initialize1DArray(m_nCells, m_rwntl, 0.);
if (m_wdntl == nullptr) Initialize1DArray(m_nCells, m_wdntl, 0.);
if (m_rmp1tl == nullptr) Initialize1DArray(m_nCells, m_rmp1tl, 0.);
if (m_roctl == nullptr) Initialize1DArray(m_nCells, m_roctl, 0.);
if (m_phpApldDays == nullptr) Initialize1DArray(m_nCells, m_phpApldDays, 0.);
if (m_phpDefDays == nullptr) Initialize1DArray(m_nCells, m_phpDefDays, 0.);
if (m_rsdCovSoil == nullptr || m_soilRsd == nullptr) {
Initialize1DArray(m_nCells, m_rsdCovSoil, m_rsdInitSoil);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilRsd, 0.);
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
m_soilRsd[i][0] = m_rsdCovSoil[i];
}
}
if (m_conv_wt == nullptr) {
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_conv_wt, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_conv_wt_reverse, 0.);
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
FLTPT wt1 = 0.;
FLTPT conv_wt = 0.;
wt1 = m_soilBD[i][k] * m_soilThk[i][k] * 0.01; 
conv_wt = 1.e6 * wt1;                          
m_conv_wt[i][k] = conv_wt;
m_conv_wt_reverse[i][k] = 1. / conv_wt;
}
}
}
if (m_soilMass == nullptr) {
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilMass, 0.);
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
m_soilMass[i][k] = 10000. * m_soilThk[i][k] *
m_soilBD[i][k] * (1. - m_soilRock[i][k] * 0.01);
}
}
}

if (m_phpSorpIdxBsn <= 0.) m_phpSorpIdxBsn = 0.4;
if (nullptr == m_phpSorpIdx) Initialize1DArray(m_nCells, m_phpSorpIdx, m_phpSorpIdxBsn);
if (nullptr == m_psp_store) Initialize2DArray(m_nCells, m_maxSoilLyrs, m_psp_store, 0.);
if (nullptr == m_ssp_store) Initialize2DArray(m_nCells, m_maxSoilLyrs, m_ssp_store, 0.);

if (m_soilNO3 == nullptr) Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilNO3, 0.);
if (m_soilFrshOrgN == nullptr || m_soilFrshOrgP == nullptr || m_soilActvOrgN == nullptr ||
m_soilActvMinP == nullptr || m_soilStabMinP == nullptr) {
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilFrshOrgN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilFrshOrgP, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilActvOrgN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilActvMinP, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilStabMinP, 0.);

#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
m_soilFrshOrgP[i][0] = m_rsdCovSoil[i] * .0010;
m_soilFrshOrgN[i][0] = m_rsdCovSoil[i] * .0055;
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
FLTPT wt1 = m_conv_wt[i][k] * 1.e-6;
if (m_soilNO3[i][k] <= 0.) {
m_soilNO3[i][k] = 0.;
FLTPT zdst = 0.;
zdst = CalExp(-m_soilDepth[i][k] * 0.001);
m_soilNO3[i][k] = 10. * zdst * 0.7;
m_soilNO3[i][k] *= wt1; 
}
if (m_soilStabOrgN[i][k] <= 0.) {
m_soilStabOrgN[i][k] = 10000. * (m_soilCbn[i][k] * 0.07142857) * wt1; 
}
FLTPT nactfr = .02;
m_soilActvOrgN[i][k] = m_soilStabOrgN[i][k] * nactfr;
m_soilStabOrgN[i][k] *= 1. - nactfr;


if (m_soilHumOrgP[i][k] <= 0.) {
m_soilHumOrgP[i][k] = 0.125 * m_soilStabOrgN[i][k];
}

if (m_soilSolP[i][k] <= 0.) {
m_soilSolP[i][k] = 5. * wt1;
}
FLTPT solp = 0.;
FLTPT actp = 0.;
if (m_solP_model == 0) {
if (!FloatEqual(m_conv_wt[i][k], 0.)) {
solp = m_soilSolP[i][k] * m_conv_wt_reverse[i][k] * 1000000.;
}
if (m_soilClay[i][k] > 0.) {
m_phpSorpIdx[i] = -0.045 * CalLn(m_soilClay[i][k]) + 0.001 * solp;
m_phpSorpIdx[i] = m_phpSorpIdx[i] - 0.035 * m_soilCbn[i][k] + 0.43;
}
if (m_phpSorpIdx[i] < 0.1) {
m_phpSorpIdx[i] = 0.1;
} else if (m_phpSorpIdx[i] > 0.7) {
m_phpSorpIdx[i] = 0.7;
}
if (m_psp_store[i][k] > 0.) {
m_phpSorpIdx[i] = (m_psp_store[i][k] * 29. + m_phpSorpIdx[i]) * 0.033333333;
}
m_psp_store[i][k] = m_phpSorpIdx[i];
}

m_soilActvMinP[i][k] = m_soilSolP[i][k] * (1. - m_phpSorpIdx[i]) / m_phpSorpIdx[i];

if (m_solP_model == 0) {
actp = m_soilActvMinP[i][k] * m_conv_wt_reverse[i][k] * 1000000.;
solp = m_soilSolP[i][k] * m_conv_wt_reverse[i][k] * 1000000.;
FLTPT ssp = 25.044 * CalPow(actp + solp, -0.3833);
if (ssp > 7.) ssp = 7.;
if (ssp < 1.) ssp = 1.;
m_soilStabMinP[i][k] = ssp * (m_soilActvMinP[i][k] + m_soilSolP[i][k]);
} else {
m_soilStabMinP[i][k] = 4. * m_soilActvMinP[i][k];
}
}
}
}


if (m_cbnModel == 2) {
FLTPT FBM = 0.;
FLTPT FHP = 0.;
FLTPT x1 = 0.;
FLTPT RTO = 0.;
if (m_sol_WOC == nullptr) {
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_WOC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_WON, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_BM, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_BMC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_BMN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_HP, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_HS, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_HSC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_HSN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_HPC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_HPN, 0.);

Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LM, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LMC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LMN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LSC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LSN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LS, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LSL, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LSLC, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_LSLNC, 0.);

Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_RNMN, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_sol_RSPC, 0.);

#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
for (int k = 0; k < m_nSoilLyrs[i]; k++) {
m_sol_WOC[i][k] = m_soilMass[i][k] * m_soilCbn[i][k] * 0.01;
m_sol_WON[i][k] = m_soilActvOrgN[i][k] + m_soilStabOrgN[i][k];
if (FBM < 1.e-10) FBM = 0.04;
if (FHP < 1.e-10) FHP = 0.6999999999996266; 
m_sol_BM[i][k] = FBM * m_sol_WOC[i][k];
m_sol_BMC[i][k] = m_sol_BM[i][k];
RTO = m_sol_WON[i][k] / m_sol_WOC[i][k];
m_sol_BMN[i][k] = RTO * m_sol_BMC[i][k];
m_sol_HP[i][k] = FHP * (m_sol_WOC[i][k] - m_sol_BM[i][k]);
m_sol_HS[i][k] = m_sol_WOC[i][k] - m_sol_BM[i][k] - m_sol_HP[i][k];
m_sol_HSC[i][k] = m_sol_HS[i][k];
m_sol_HSN[i][k] = RTO * m_sol_HSC[i][k];
m_sol_HPC[i][k] = m_sol_HP[i][k];        
m_sol_HPN[i][k] = RTO * m_sol_HPC[i][k]; 
x1 = m_soilRsd[i][k] * 0.001;
m_sol_LM[i][k] = 500. * x1;
m_sol_LS[i][k] = m_sol_LM[i][k];
m_sol_LSL[i][k] = 0.8 * m_sol_LS[i][k];
m_sol_LMC[i][k] = 0.42 * m_sol_LM[i][k];

m_sol_LMN[i][k] = 0.1 * m_sol_LMC[i][k];
m_sol_LSC[i][k] = 0.42 * m_sol_LS[i][k];
m_sol_LSLC[i][k] = 0.8 * m_sol_LSC[i][k];
m_sol_LSLNC[i][k] = 0.2 * m_sol_LSC[i][k];
m_sol_LSN[i][k] = m_sol_LSC[i][k] * 0.006666666666666667; 

m_sol_WOC[i][k] += m_sol_LSC[i][k] + m_sol_LMC[i][k];
m_sol_WON[i][k] += m_sol_LSN[i][k] + m_sol_LMN[i][k];

m_soilStabOrgN[i][k] = m_sol_HPN[i][k];
m_soilActvOrgN[i][k] = m_sol_HSN[i][k];
m_soilFrshOrgN[i][k] = m_sol_LMN[i][k] + m_sol_LSN[i][k];
}
}
}
}
if (!FloatEqual(m_wshd_dnit, 0.)) {
m_wshd_dnit = 0.;
m_wshd_hmn = 0.;
m_wshd_hmp = 0.;
m_wshd_rmn = 0.;
m_wshd_rmp = 0.;
m_wshd_rwn = 0.;
m_wshd_nitn = 0.;
m_wshd_voln = 0.;
m_wshd_pal = 0.;
m_wshd_pas = 0.;
}
}

int Nutrient_Transformation::Execute() {
CheckInputData();
InitialOutputs();
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
if (m_cbnModel == 0) {
MineralizationStaticCarbonMethod(i);
} else if (m_cbnModel == 1) {
MineralizationCfarmOneCarbonModel(i);
} else if (m_cbnModel == 2) {
MineralizationCenturyModel(i);
}
Volatilization(i);
CalculatePflux(i);
}
return 0;
}

void Nutrient_Transformation::MineralizationStaticCarbonMethod(const int i) {
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
int kk = k == 0 ? 1 : k;
FLTPT sut = 0.;
if (m_soilTemp[i] > 0) {
if (m_soilWtrSto[i][kk] < 0) {
m_soilWtrSto[i][kk] = 0.0000001;
}
sut = 0.1 + 0.9 * CalSqrt(m_soilWtrSto[i][kk] / m_soilFC[i][kk]);
sut = Max(0.05, sut);


FLTPT xx = 0.;
FLTPT cdg = 0.;
xx = m_soilTemp[i];
cdg = 0.9 * xx / (xx + CalExp(9.93 - 0.312 * xx)) + 0.1;
cdg = Max(0.1, cdg);

xx = 0.;
FLTPT csf = 0.;
xx = cdg * sut;
if (xx < 0) {
xx = 0.;
}
if (xx > 1000000) {
xx = 1000000.;
}
csf = CalSqrt(xx);

FLTPT rwn = 0.;
rwn = 0.1e-4 * (m_soilActvOrgN[i][k] * (1. / m_orgNFrActN - 1.) - m_soilStabOrgN[i][k]);
if (rwn > 0.) {
rwn = Min(rwn, m_soilActvOrgN[i][k]);
} else {
rwn = -(Min(Abs(rwn), m_soilStabOrgN[i][k]));
}
m_soilStabOrgN[i][k] = Max(1.e-6, m_soilStabOrgN[i][k] + rwn);
m_soilActvOrgN[i][k] = Max(1.e-6, m_soilActvOrgN[i][k] - rwn);

FLTPT hmn = 0.;
hmn = m_minrlCoef * csf * m_soilActvOrgN[i][k];
hmn = Min(hmn, m_soilActvOrgN[i][k]);
xx = 0.;
FLTPT hmp = 0.;
xx = m_soilStabOrgN[i][k] + m_soilActvOrgN[i][k];
if (xx > 1.e-6) {
hmp = 1.4 * hmn * m_soilHumOrgP[i][k] / xx;
} else {
hmp = 0.;
}
hmp = Min(hmp, m_soilHumOrgP[i][k]);

m_soilActvOrgN[i][k] = Max(1.e-6, m_soilActvOrgN[i][k] - hmn);
m_soilNO3[i][k] = m_soilNO3[i][k] + hmn;
m_soilHumOrgP[i][k] = m_soilHumOrgP[i][k] - hmp;
m_soilSolP[i][k] = m_soilSolP[i][k] + hmp;

FLTPT rmn1 = 0.;
FLTPT rmp = 0.;
if (k <= 2) {
FLTPT cnr = 0.;
FLTPT cpr = 0.;
FLTPT ca = 0.;
FLTPT cnrf = 0.;
FLTPT cprf = 0.;
if (m_soilFrshOrgN[i][k] + m_soilNO3[i][k] > 1e-4) {
cnr = 0.58 * m_soilRsd[i][k] / (m_soilFrshOrgN[i][k] + m_soilNO3[i][k]);
if (cnr > 500) {
cnr = 500.;
}
cnrf = CalExp(-0.693 * (cnr - 25.f) * 0.04);
} else {
cnrf = 1.f;
}
if (m_soilFrshOrgP[i][k] + m_soilSolP[i][k] > 1e-4) {
cpr = 0.58 * m_soilRsd[i][k] / (m_soilFrshOrgP[i][k] + m_soilSolP[i][k]);
if (cpr > 5000) {
cpr = 5000.;
}
cprf = 0.;
cprf = CalExp(-0.693 * (cpr - 200.) * 0.005);
} else {
cprf = 1.;
}
FLTPT decr = 0.;

FLTPT rdc = 0.;
ca = Min(Min(cnrf, cprf), 1.);
if (m_pltRsdDecCoef[i] < 0.) {
decr = 0.05;
} else {
decr = m_pltRsdDecCoef[i] * ca * csf;
}
decr = Min(decr, 1.);
m_soilRsd[i][k] = Max(1.e-6, m_soilRsd[i][k]);
rdc = decr * m_soilRsd[i][k];
m_soilRsd[i][k] = m_soilRsd[i][k] - rdc;
if (m_soilRsd[i][k] < 0)m_soilRsd[i][k] = 0.;
rmn1 = decr * m_soilFrshOrgN[i][k];
m_soilFrshOrgP[i][k] = Max(1.e-6, m_soilFrshOrgP[i][k]);
rmp = decr * m_soilFrshOrgP[i][k];

m_soilFrshOrgP[i][k] = m_soilFrshOrgP[i][k] - rmp;
m_soilFrshOrgN[i][k] = Max(1.e-6, m_soilFrshOrgN[i][k]) - rmn1;

m_soilNO3[i][k] = m_soilNO3[i][k] + 0.8 * rmn1;
m_soilActvOrgN[i][k] = m_soilActvOrgN[i][k] + 0.2 * rmn1;
m_soilSolP[i][k] = m_soilSolP[i][k] + 0.8 * rmp;
m_soilHumOrgP[i][k] = m_soilHumOrgP[i][k] + 0.2 * rmp;
}
FLTPT wdn = 0.;
if (sut >= m_denitThres) {
wdn = m_soilNO3[i][k] * (1. - CalExp(-m_denitCoef * cdg * m_soilCbn[i][k]));
} else {
wdn = 0.;
}
m_soilNO3[i][k] = m_soilNO3[i][k] - wdn;

m_wshd_hmn = m_wshd_hmn + hmn * m_cellAreaFr;
m_wshd_rwn = m_wshd_rwn + rwn * m_cellAreaFr;
m_wshd_hmp = m_wshd_hmp + hmp * m_cellAreaFr;
m_wshd_rmn = m_wshd_rmn + rmn1 * m_cellAreaFr;
m_wshd_rmp = m_wshd_rmp + rmp * m_cellAreaFr;
m_wshd_dnit = m_wshd_dnit + wdn * m_cellAreaFr;
m_hmntl[i] = hmn;
m_rwntl[i] = rwn;
m_hmptl[i] = hmp;
m_rmn2tl[i] = rmn1;
m_rmptl[i] = rmp;
m_wdntl[i] = wdn;
}
}
}

void Nutrient_Transformation::MineralizationCfarmOneCarbonModel(const int i) {
}

void Nutrient_Transformation::Volatilization(const int i) {
FLTPT kk = 0.;
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
FLTPT nvtf = 0.;
nvtf = 0.41 * (m_soilTemp[i] - 5.) * 0.1;
if (m_soilNH4[i][k] > 0. && nvtf >= 0.001) {
FLTPT sw25 = 0.;
FLTPT swwp = 0.;
FLTPT swf = 0.;
sw25 = m_soilWP[i][k] + 0.25 * m_soilFC[i][k];
swwp = m_soilWP[i][k] + m_soilWtrSto[i][k];
if (swwp < sw25) {
swf = (swwp - m_soilWP[i][k]) / (sw25 - m_soilWP[i][k]);
} else {
swf = 1.;
}
kk = k == 0 ? 0. : m_soilDepth[i][k - 1];
FLTPT dmidl = 0.;
FLTPT dpf = 0.;
FLTPT akn = 0.;
FLTPT akv = 0.;
FLTPT rnv = 0.;
FLTPT rnit = 0.;
FLTPT rvol = 0.;
FLTPT cecf = 0.15;
dmidl = (m_soilDepth[i][k] + kk) * 0.5;
dpf = 1. - dmidl / (dmidl + CalExp(4.706 - 0.0305 * dmidl));
akn = nvtf * swf;
akv = nvtf * dpf * cecf;
rnv = m_soilNH4[i][k] * (1. - CalExp(-akn - akv));
rnit = 1. - CalExp(-akn);
rvol = 1. - CalExp(-akv);

if (rvol + rnit > 1.e-6) {
rvol = rnv * rvol / (rvol + rnit);
rnit = rnv - rvol;
if (rnit < 0)rnit = 0.;
m_soilNH4[i][k] = Max(1e-6, m_soilNH4[i][k] - rnit);
}
if (m_soilNH4[i][k] < 0.) {
rnit = rnit + m_soilNH4[i][k];
m_soilNH4[i][k] = 0.;
}
m_soilNO3[i][k] = m_soilNO3[i][k] + rnit;
m_soilNH4[i][k] = Max(1e-6, m_soilNH4[i][k] - rvol);
if (m_soilNH4[i][k] < 0.) {
rvol = rvol + m_soilNH4[i][k];
m_soilNH4[i][k] = 0.;
}
m_wshd_voln += rvol * m_cellAreaFr;
m_wshd_nitn += rnit * m_cellAreaFr;
}
}
}

void Nutrient_Transformation::CalculatePflux(const int i) {
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
if (m_soilSolP[i][k] <= 1.e-6) m_soilSolP[i][k] = 1.e-6;
if (m_soilActvMinP[i][k] <= 1.e-6) m_soilActvMinP[i][k] = 1.e-6;
if (m_soilStabMinP[i][k] <= 1.e-6) m_soilStabMinP[i][k] = 1.e-6;

FLTPT solp = 0.;
FLTPT actp = 0.;
FLTPT stap = 0.;
solp = m_soilSolP[i][k] * m_conv_wt_reverse[i][k] * 1000000.;
actp = m_soilActvMinP[i][k] * m_conv_wt_reverse[i][k] * 1000000.;
stap = m_soilStabMinP[i][k] * m_conv_wt_reverse[i][k] * 1000000.;

FLTPT psp = 0.;
if (m_soilClay[i][k] > 0.) {
psp = -0.045 * CalLn(m_soilClay[i][k]) + 0.001 * solp;
psp = psp - 0.035 * m_soilCbn[i][k] + 0.43;
} else {
psp = 0.4;
}
if (psp < 0.1) psp = 0.1;
if (psp > 0.7) psp = 0.7;

if (m_phpSorpIdxBsn > 0.) psp = (m_phpSorpIdxBsn * 29. + psp * 1.) * 0.03333333333333333; 
m_phpSorpIdxBsn = psp;

/
cdg = CalPow(m_soilTemp[i] + 5., _8div3) * (50. - m_soilTemp[i]) * 3.562449888909787e-06;
if (cdg < 0.) cdg = 0.;

FLTPT ox = 0.;
ox = 1.f - 0.8 * ((m_soilDepth[i][kk] + m_soilDepth[i][kk - 1]) * 0.5) /
((m_soilDepth[i][kk] + m_soilDepth[i][kk - 1]) * 0.5 +
CalExp(18.40961 - 0.023683632 * ((m_soilDepth[i][kk] + m_soilDepth[i][kk - 1]) * 0.5)));
FLTPT cs = 0.;
cs = Min(10., CalSqrt(cdg * sut) * 0.9 * ox * x1);
if (cdg > 0. && voidfr <= 0.1f) {
FLTPT vof = 1. / (1. + CalPow(voidfr * 25., 5.));
wdn = m_soilNO3[i][k] * (1. - CalExp(-m_denitCoef * cdg * vof * m_soilCbn[i][k]));
m_soilNO3[i][k] -= wdn;
}
m_wshd_dnit += wdn * m_cellAreaFr;
m_wdntl[i] += wdn;

FLTPT sol_min_n = m_soilNO3[i][k] + m_soilNH4[i][k];
RLR = Min(0.8, m_sol_LSL[i][k] / (m_sol_LS[i][k] + 1.e-5f));
HSR = 5.4799998e-4;
HPR = 1.2e-5;
APCO2 = .55;
ASCO2 = .60;
PRMT_51 = 0.; 
PRMT_51 = 1.;
if (k == 0) {
cs = cs * PRMT_51;
ABCO2 = .55;
A1CO2 = .55;
BMR = .0164;
LMR = .0405;
LSR = .0107;
NCHP = .1;
XBM = 1.;
x1 = 0.1 * (m_sol_LSN[i][k] + m_sol_LMN[i][k]) / (m_soilRsd[i][k] / 1000. + 1.e-5);
if (x1 > 2.) {
NCBM = .1;
} else if (x1 > .01) {
NCBM = 1. / (20.05 - 5.0251 * x1);
} else {
NCBM = .05;
}
NCHS = NCBM / (5. * NCBM + 1.);
} else {
ABCO2 = 0.17 + 0.0068 * m_soilSand[i][k];
A1CO2 = .55;
BMR = .02;
LMR = .0507;
LSR = .0132;
XBM = .25 + .0075 * m_soilSand[i][k];

x1 = 1000. * sol_min_n / (m_soilMass[i][k] * 0.001);
if (x1 > 7.15) {
NCBM = .33;
NCHS = .083;
NCHP = .143;
} else {
NCBM = 1. / (15. - 1.678 * x1);
NCHS = 1. / (20. - 1.119 * x1);
NCHP = 1. / (10. - .42 * x1);
}
}
ABP = .003 + .00032 * m_soilClay[i][k];

PRMT_45 = 0.; 
PRMT_45 = 5.0000001e-2;
ASP = Max(.001, PRMT_45 - .00009 * m_soilClay[i][k]);
x1 = LSR * cs * CalExp(-3. * RLR);
LSCTP = x1 * m_sol_LSC[i][k];
LSLCTP = LSCTP * RLR;
LSLNCTP = LSCTP * (1. - RLR);
LSNTP = x1 * m_sol_LSN[i][k];
x1 = LMR * cs;
LMCTP = m_sol_LMC[i][k] * x1;
LMNTP = m_sol_LMN[i][k] * x1;
x1 = BMR * cs * XBM;
BMCTP = m_sol_BMC[i][k] * x1;
BMNTP = m_sol_BMN[i][k] * x1;
x1 = HSR * cs;
HSCTP = m_sol_HSC[i][k] * x1;
HSNTP = m_sol_HSN[i][k] * x1;
x1 = cs * HPR;
HPCTP = m_sol_HPC[i][k] * x1;
HPNTP = m_sol_HPN[i][k] * x1;
A1 = 1. - A1CO2;
ASX = 1. - ASCO2 - ASP;
APX = 1. - APCO2;

PN1 = LSLNCTP * A1 * NCBM; 
PN2 = .7 * LSLCTP * NCHS; 
PN3 = LMCTP * A1 * NCBM;   
PN5 = BMCTP * ABP * NCHP;                 
PN6 = BMCTP * (1. - ABP - ABCO2) * NCHS; 
PN7 = HSCTP * ASX * NCBM;                 
PN8 = HSCTP * ASP * NCHP;                 
PN9 = HPCTP * APX * NCBM;                 

SUM = 0.;
x1 = PN1 + PN2;
if (LSNTP < x1) { CPN1 = x1 - LSNTP; } else { SUM = SUM + LSNTP - x1; }
if (LMNTP < PN3) { CPN2 = PN3 - LMNTP; } else { SUM = SUM + LMNTP - PN3; }
x1 = PN5 + PN6;
if (BMNTP < x1) { CPN3 = x1 - BMNTP; } else { SUM = SUM + BMNTP - x1; }
x1 = PN7 + PN8;
if (HSNTP < x1) { CPN4 = x1 - HSNTP; } else { SUM = SUM + HSNTP - x1; }
if (HPNTP < PN9) { CPN5 = PN9 - HPNTP; } else { SUM = SUM + HPNTP - PN9; }

FLTPT Wmin = Max(1.e-5, m_soilNO3[i][k] + m_soilNH4[i][k] + SUM);
FLTPT DMDN = CPN1 + CPN2 + CPN3 + CPN4 + CPN5;
FLTPT x3 = 1.;
if (Wmin < DMDN) x3 = Wmin / DMDN;


if (CPN1 > 0.) {
LSCTA = LSCTP * x3;
LSNTA = LSNTP * x3;
LSLCTA = LSLCTP * x3;
LSLNCTA = LSLNCTP * x3;
} else {
LSCTA = LSCTP;
LSNTA = LSNTP;
LSLCTA = LSLCTP;
LSLNCTA = LSLNCTP;
}
if (CPN2 > 0.) {
LMCTA = LMCTP * x3;
LMNTA = LMNTP * x3;
} else {
LMCTA = LMCTP;
LMNTA = LMNTP;
}
if (CPN3 > 0.) {
BMCTA = BMCTP * x3;
BMNTA = BMNTP * x3;
} else {
BMCTA = BMCTP;
BMNTA = BMNTP;
}
if (CPN4 > 0.) {
HSCTA = HSCTP * x3;
HSNTA = HSNTP * x3;
} else {
HSCTA = HSCTP;
HSNTA = HSNTP;
}
if (CPN5 > 0.) {
HPCTA = HPCTP * x3;
HPNTA = HPNTP * x3;
} else {
HPCTA = HPCTP;
HPNTA = HPNTP;
}

PN1 = LSLNCTA * A1 * NCBM; 
PN2 = .7 * LSLCTA * NCHS; 
PN3 = LMCTA * A1 * NCBM;   
PN5 = BMCTA * ABP * NCHP;                 
PN6 = BMCTA * (1. - ABP - ABCO2) * NCHS;  
PN7 = HSCTA * ASX * NCBM;                 
PN8 = HSCTA * ASP * NCHP;                 
PN9 = HPCTA * APX * NCBM;                 
SUM = 0.;
CPN1 = 0.;
CPN2 = 0.;
CPN3 = 0.;
CPN4 = 0.;
CPN5 = 0.;
x1 = PN1 + PN2;
if (LSNTA < x1) { CPN1 = x1 - LSNTA; } else { SUM = SUM + LSNTA - x1; }
if (LMNTA < PN3) { CPN2 = PN3 - LMNTA; } else { SUM = SUM + LMNTA - PN3; }
x1 = PN5 + PN6;
if (BMNTA < x1) { CPN3 = x1 - BMNTA; } else { SUM = SUM + BMNTA - x1; }
x1 = PN7 + PN8;
if (HSNTA < x1) { CPN4 = x1 - HSNTA; } else { SUM = SUM + HSNTA - x1; }
if (HPNTA < PN9) { CPN5 = PN9 - HPNTA; } else { SUM = SUM + HPNTA - PN9; }

Wmin = Max(1.e-5, m_soilNO3[i][k] + m_soilNH4[i][k] + SUM);
DMDN = CPN1 + CPN2 + CPN3 + CPN4 + CPN5;

m_sol_RNMN[i][k] = SUM - DMDN;
if (m_sol_RNMN[i][k] > 0.) {
m_soilNH4[i][k] += m_sol_RNMN[i][k];
} else {
x1 = m_soilNO3[i][k] + m_sol_RNMN[i][k];
if (x1 < 0.) {
m_sol_RNMN[i][k] = -m_soilNO3[i][k];
m_soilNO3[i][k] = 1.e-6;
} else {
m_soilNO3[i][k] = x1;
}
}
DF1 = LSNTA;
DF2 = LMNTA;

FLTPT hmp = 0.;
FLTPT hmp_rate = 0.;
hmp_rate = 1.4 * (HSNTA + HPNTA) / (m_sol_HSN[i][k] + m_sol_HPN[i][k] + 1.e-6);
hmp = hmp_rate * m_soilHumOrgP[i][k];
hmp = Min(hmp, m_soilHumOrgP[i][k]);
m_soilHumOrgP[i][k] = m_soilHumOrgP[i][k] - hmp;
m_soilSolP[i][k] = m_soilSolP[i][k] + hmp;

FLTPT rmp = 0.;
FLTPT decr = 0.;
decr = (LSCTA + LMCTA) / (m_sol_LSC[i][k] + m_sol_LMC[i][k] + 1.e-6);
decr = Min(1., decr);
rmp = decr * m_soilFrshOrgP[i][k];

m_soilFrshOrgP[i][k] = m_soilFrshOrgP[i][k] - rmp;
m_soilSolP[i][k] = m_soilSolP[i][k] + .8 * rmp;
m_soilHumOrgP[i][k] = m_soilHumOrgP[i][k] + .2 * rmp;

LSCTA = Min(m_sol_LSC[i][k], LSCTA);
m_sol_LSC[i][k] = Max(1.e-10, m_sol_LSC[i][k] - LSCTA);
LSLCTA = Min(m_sol_LSLC[i][k], LSLCTA);
m_sol_LSLC[i][k] = Max(1.e-10, m_sol_LSLC[i][k] - LSLCTA);
m_sol_LSLNC[i][k] = Max(1.e-10, m_sol_LSLNC[i][k] - LSLNCTA);
LMCTA = Min(m_sol_LMC[i][k], LMCTA);
if (m_sol_LM[i][k] > 0.) {
RTO = Max(0.42, m_sol_LMC[i][k] / m_sol_LM[i][k]);
m_sol_LM[i][k] = m_sol_LM[i][k] - LMCTA / RTO;
m_sol_LMC[i][k] = m_sol_LMC[i][k] - LMCTA;
}
m_sol_LSL[i][k] = Max(1.e-10, m_sol_LSL[i][k] - LSLCTA * 2.380952380952381); 
m_sol_LS[i][k] = Max(1.e-10, m_sol_LS[i][k] - LSCTA * 2.380952380952381);

x3 = APX * HPCTA + ASX * HSCTA + A1 * (LMCTA + LSLNCTA);
m_sol_BMC[i][k] = m_sol_BMC[i][k] - BMCTA + x3;
DF3 = BMNTA - NCBM * x3;
x1 = .7 * LSLCTA + BMCTA * (1. - ABP - ABCO2);
m_sol_HSC[i][k] = m_sol_HSC[i][k] - HSCTA + x1;
DF4 = HSNTA - NCHS * x1;
x1 = HSCTA * ASP + BMCTA * ABP;
m_sol_HPC[i][k] = m_sol_HPC[i][k] - HPCTA + x1;
DF5 = HPNTA - NCHP * x1;
DF6 = sol_min_n - m_soilNO3[i][k] - m_soilNH4[i][k];

ADD = DF1 + DF2 + DF3 + DF4 + DF5 + DF6;
ADF1 = Abs(DF1);
ADF2 = Abs(DF2);
ADF3 = Abs(DF3);
ADF4 = Abs(DF4);
ADF5 = Abs(DF5);
TOT = ADF1 + ADF2 + ADF3 + ADF4 + ADF5;
xx = ADD / (TOT + 1.e-10);
m_sol_LSN[i][k] = Max(.001, m_sol_LSN[i][k] - DF1 + xx * ADF1);
m_sol_LMN[i][k] = Max(.001, m_sol_LMN[i][k] - DF2 + xx * ADF2);
m_sol_BMN[i][k] = m_sol_BMN[i][k] - DF3 + xx * ADF3;
m_sol_HSN[i][k] = m_sol_HSN[i][k] - DF4 + xx * ADF4;
m_sol_HPN[i][k] = m_sol_HPN[i][k] - DF5 + xx * ADF5;
m_sol_RSPC[i][k] = .3 * LSLCTA + A1CO2 * (LSLNCTA + LMCTA) +
ABCO2 * BMCTA + ASCO2 * HSCTA + APCO2 * HPCTA;

m_soilRsd[i][k] = m_sol_LS[i][k] + m_sol_LM[i][k];
m_soilStabOrgN[i][k] = m_sol_HPN[i][k];
m_soilActvOrgN[i][k] = m_sol_HSN[i][k];
m_soilFrshOrgN[i][k] = m_sol_LMN[i][k] + m_sol_LSN[i][k];
m_soilCbn[i][k] = 100. * (m_sol_LSC[i][k] + m_sol_LMC[i][k] + m_sol_HSC[i][k] +
m_sol_HPC[i][k] + m_sol_BMC[i][k]) / m_soilMass[i][k];

FLTPT hmn = 0.;
hmn = m_sol_RNMN[i][k];
m_wshd_hmn = m_wshd_hmn + hmn * m_cellAreaFr;
FLTPT rwn = 0.;
rwn = HSNTA;
m_wshd_rwn = m_wshd_rwn + rwn * m_cellAreaFr;

m_wshd_hmp = m_wshd_hmp + hmp * m_cellAreaFr;
FLTPT rmn1 = 0.;
rmn1 = LSNTA + LMNTA;
m_wshd_rmn = m_wshd_rmn + rmn1 * m_cellAreaFr;
m_wshd_rmp = m_wshd_rmp + rmp * m_cellAreaFr;
m_wshd_dnit = m_wshd_dnit + wdn * m_cellAreaFr;
m_hmntl[i] = m_hmntl[i] + hmn;
m_rwntl[i] = m_rwntl[i] + rwn;
m_hmptl[i] = m_hmptl[i] + hmp;
m_rmn2tl[i] = m_rmn2tl[i] + rmn1;
m_rmptl[i] = m_rmptl[i] + rmp;
m_wdntl[i] = m_wdntl[i] + wdn;

#ifdef _DEBUG
if (isnan(wdn) || isinf(wdn) ||
isnan(m_soilNO3[i][k]) || isinf(m_soilNO3[i][k]) ||
isnan(m_sol_HPN[i][k]) || isinf(m_sol_HPN[i][k]) ||
isnan(m_sol_LSN[i][k]) || isinf(m_sol_LSN[i][k]) ||
isnan(m_sol_LMN[i][k]) || isinf(m_sol_LMN[i][k]) ||
isnan(m_sol_BMN[i][k]) || isinf(m_sol_BMN[i][k]) ||
isnan(m_sol_HSN[i][k]) || isinf(m_sol_HSN[i][k]) ||
isnan(m_soilRsd[i][k]) || isinf(m_soilRsd[i][k])) {
cout << "NUTRTF: m_soilNO3[i][k] " << m_soilNO3[i][k] << endl;
}
if (m_sol_BMC[i][k] < 0.f )
{
cout << "NUTRTF: m_sol_BMC: " << m_sol_BMC[i][k]
<< " sut: " << sut << " x1: " << x1 << " x3: " << x3 << " Wmin: " << Wmin << " DMDN: " << DMDN
<< " APX: " << APX << " HPCTA: " << HPCTA << " ASX: " << ASX
<< " HSCTA: " << HSCTA << " A1: " << A1 << " LMCTA: " << LMCTA << " LSLNCTA: " << LSLNCTA
<< " BMCTA: " << BMCTA << endl;
}
#endif
}
}
}

void Nutrient_Transformation::GetValue(const char* key, FLTPT* value) {
string sk(key);
if (StringMatch(sk, VAR_WSHD_DNIT[0])) {
*value = m_wshd_dnit;
} else if (StringMatch(sk, VAR_WSHD_HMN[0])) {
*value = m_wshd_hmn;
} else if (StringMatch(sk, VAR_WSHD_HMP[0])) {
*value = m_wshd_hmp;
} else if (StringMatch(sk, VAR_WSHD_RMN[0])) {
*value = m_wshd_rmn;
} else if (StringMatch(sk, VAR_WSHD_RMP[0])) {
*value = m_wshd_rmp;
} else if (StringMatch(sk, VAR_WSHD_RWN[0])) {
*value = m_wshd_rwn;
} else if (StringMatch(sk, VAR_WSHD_NITN[0])) {
*value = m_wshd_nitn;
} else if (StringMatch(sk, VAR_WSHD_VOLN[0])) {
*value = m_wshd_voln;
} else if (StringMatch(sk, VAR_WSHD_PAL[0])) {
*value = m_wshd_pal;
} else if (StringMatch(sk, VAR_WSHD_PAS[0])) {
*value = m_wshd_pas;
} else {
throw ModelException(M_NUTR_TF[0], "GetValue",
"Parameter " + sk + " does not exist.");
}
}

void Nutrient_Transformation::Get1DData(const char* key, int* n, FLTPT** data) {
InitialOutputs();
string sk(key);
if (StringMatch(sk, VAR_HMNTL[0])) {
*data = m_hmntl;
} else if (StringMatch(sk, VAR_HMPTL[0])) {
*data = m_hmptl;
} else if (StringMatch(sk, VAR_RMN2TL[0])) {
*data = m_rmn2tl;
} else if (StringMatch(sk, VAR_RMPTL[0])) {
*data = m_rmptl;
} else if (StringMatch(sk, VAR_RWNTL[0])) {
*data = m_rwntl;
} else if (StringMatch(sk, VAR_WDNTL[0])) {
*data = m_wdntl;
} else if (StringMatch(sk, VAR_RMP1TL[0])) {
*data = m_rmp1tl;
} else if (StringMatch(sk, VAR_ROCTL[0])) {
*data = m_roctl;
} else if (StringMatch(sk, VAR_SOL_COV[0])) {
*data = m_rsdCovSoil;
} else if (StringMatch(sk, VAR_A_DAYS[0])) {
*data = m_phpApldDays;
} else if (StringMatch(sk, VAR_B_DAYS[0])) {
*data = m_phpDefDays;
} else {
throw ModelException(M_NUTR_TF[0], "Get1DData",
"Parameter " + sk + " does not exist.");
}
*n = m_nCells;
}

void Nutrient_Transformation::Get2DData(const char* key, int* nRows, int* nCols, FLTPT*** data) {
InitialOutputs();
string sk(key);
*nRows = m_nCells;
*nCols = m_maxSoilLyrs;
if (StringMatch(sk, VAR_SOL_AORGN[0])) {
*data = m_soilActvOrgN;
} else if (StringMatch(sk, VAR_SOL_FORGN[0])) {
*data = m_soilFrshOrgN;
} else if (StringMatch(sk, VAR_SOL_FORGP[0])) {
*data = m_soilFrshOrgP;
} else if (StringMatch(sk, VAR_SOL_NO3[0])) {
*data = m_soilNO3;
} else if (StringMatch(sk, VAR_SOL_NH4[0])) {
*data = m_soilNH4;
} else if (StringMatch(sk, VAR_SOL_SORGN[0])) {
*data = m_soilStabOrgN;
} else if (StringMatch(sk, VAR_SOL_HORGP[0])) {
*data = m_soilHumOrgP;
} else if (StringMatch(sk, VAR_SOL_SOLP[0])) {
*data = m_soilSolP;
} else if (StringMatch(sk, VAR_SOL_ACTP[0])) {
*data = m_soilActvMinP;
} else if (StringMatch(sk, VAR_SOL_STAP[0])) {
*data = m_soilStabMinP;
} else if (StringMatch(sk, VAR_SOL_RSD[0])) {
*data = m_soilRsd;
}
else if (StringMatch(sk, VAR_SOL_WOC[0])) {
*data = m_sol_WOC;
} else if (StringMatch(sk, VAR_SOL_WON[0])) {
*data = m_sol_WON;
} else if (StringMatch(sk, VAR_SOL_BM[0])) {
*data = m_sol_BM;
} else if (StringMatch(sk, VAR_SOL_BMC[0])) {
*data = m_sol_BMC;
} else if (StringMatch(sk, VAR_SOL_BMN[0])) {
*data = m_sol_BMN;
} else if (StringMatch(sk, VAR_SOL_HP[0])) {
*data = m_sol_HP;
} else if (StringMatch(sk, VAR_SOL_HS[0])) {
*data = m_sol_HS;
} else if (StringMatch(sk, VAR_SOL_HSC[0])) {
*data = m_sol_HSC;
} else if (StringMatch(sk, VAR_SOL_HSN[0])) {
*data = m_sol_HSN;
} else if (StringMatch(sk, VAR_SOL_HPC[0])) {
*data = m_sol_HPC;
} else if (StringMatch(sk, VAR_SOL_HPN[0])) {
*data = m_sol_HPN;
} else if (StringMatch(sk, VAR_SOL_LM[0])) {
*data = m_sol_LM;
} else if (StringMatch(sk, VAR_SOL_LMC[0])) {
*data = m_sol_LMC;
} else if (StringMatch(sk, VAR_SOL_LMN[0])) {
*data = m_sol_LMN;
} else if (StringMatch(sk, VAR_SOL_LSC[0])) {
*data = m_sol_LSC;
} else if (StringMatch(sk, VAR_SOL_LSN[0])) {
*data = m_sol_LSN;
} else if (StringMatch(sk, VAR_SOL_LS[0])) {
*data = m_sol_LS;
} else if (StringMatch(sk, VAR_SOL_LSL[0])) {
*data = m_sol_LSL;
} else if (StringMatch(sk, VAR_SOL_LSLC[0])) {
*data = m_sol_LSLC;
} else if (StringMatch(sk, VAR_SOL_LSLNC[0])) {
*data = m_sol_LSLNC;
} else if (StringMatch(sk, VAR_SOL_RNMN[0])) {
*data = m_sol_RNMN;
} else if (StringMatch(sk, VAR_SOL_RSPC[0])) {
*data = m_sol_RSPC;
} else if (StringMatch(sk, VAR_CONV_WT[0])) {
*data = m_conv_wt;
} else {
throw ModelException(M_NUTR_TF[0], "Get2DData",
"Parameter " + sk + " does not exist.");
}
}
