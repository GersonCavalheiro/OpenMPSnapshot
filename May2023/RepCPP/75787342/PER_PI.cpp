#include "PER_PI.h"

#include "text.h"

PER_PI::PER_PI() :
m_maxSoilLyrs(-1), m_nSoilLyrs(nullptr), m_soilThk(nullptr),
m_dt(-1), m_nCells(-1), m_soilFrozenTemp(NODATA_VALUE),
m_ks(nullptr), m_soilSat(nullptr), m_soilFC(nullptr), m_soilWP(nullptr),
m_poreIdx(nullptr), m_soilWtrSto(nullptr), m_soilWtrStoPrfl(nullptr),
m_soilTemp(nullptr), m_infil(nullptr), m_impndTrig(nullptr),
m_soilPerco(nullptr) {
}

PER_PI::~PER_PI() {
if (m_soilPerco != nullptr) Release2DArray(m_soilPerco);
}

void PER_PI::InitialOutputs() {
CHECK_POSITIVE(M_PER_PI[0], m_nCells);
if (nullptr == m_soilPerco) Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilPerco, NODATA_VALUE);
}

int PER_PI::Execute() {
CheckInputData();
InitialOutputs();

#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
for (int j = 0; j < CVT_INT(m_nSoilLyrs[i]); j++) {
float k = 0.f, swater = 0.f, maxSoilWater = 0.f, fcSoilWater = 0.f;
if (j == 0 && m_soilTemp[i] <= m_soilFrozenTemp) {
continue;
}
swater = m_soilWtrSto[i][j];
maxSoilWater = m_soilSat[i][j];
fcSoilWater = m_soilFC[i][j];


if (swater > fcSoilWater) {

if (swater > maxSoilWater) {
k = m_ks[i][j];
} else {
float dcIndex = 2.f * m_poreIdx[i][j] + 3.f;          
k = m_ks[i][j] * CalPow(swater / maxSoilWater, dcIndex); 
}

m_soilPerco[i][j] = k * m_dt / 3600.f; 

if (swater - m_soilPerco[i][j] > maxSoilWater) {
m_soilPerco[i][j] = swater - maxSoilWater;
} else if (swater - m_soilPerco[i][j] < fcSoilWater) {
m_soilPerco[i][j] = swater - fcSoilWater;
}

if (m_soilPerco[i][j] < 0.f) {
m_soilPerco[i][j] = 0.f;
}
m_soilWtrSto[i][j] -= m_soilPerco[i][j];
if (j < m_nSoilLyrs[i] - 1) {
m_soilWtrSto[i][j + 1] += m_soilPerco[i][j];
}

} else {
m_soilPerco[i][j] = 0.f;
}
}
m_soilWtrStoPrfl[i] = 0.f;
for (int ly = 0; ly < CVT_INT(m_nSoilLyrs[i]); ly++) {
m_soilWtrStoPrfl[i] += m_soilWtrSto[i][ly];
}
}
return 0;
}

void PER_PI::Get2DData(const char* key, int* nRows, int* nCols, float*** data) {
InitialOutputs();
string sk(key);
*nRows = m_nCells;
*nCols = m_maxSoilLyrs;
if (StringMatch(sk, VAR_PERCO[0])) *data = m_soilPerco;
else {
throw ModelException(M_PER_PI[0], "Get2DData", "Output " + sk + " does not exist.");
}
}

void PER_PI::Set1DData(const char* key, const int nrows, float* data) {
CheckInputSize(M_PER_PI[0], key, nrows, m_nCells);
string sk(key);
if (StringMatch(sk, VAR_SOTE[0])) m_soilTemp = data;
else if (StringMatch(sk, VAR_INFIL[0])) m_infil = data;
else if (StringMatch(sk, VAR_SOILLAYERS[0])) m_nSoilLyrs = data;
else if (StringMatch(sk, VAR_SOL_SW[0])) m_soilWtrStoPrfl = data;
else if (StringMatch(sk, VAR_IMPOUND_TRIG[0])) m_impndTrig = data;
else {
throw ModelException(M_PER_PI[0], "Set1DData", "Parameter " + sk + " does not exist.");
}
}

void PER_PI::Set2DData(const char* key, const int nrows, const int ncols, float** data) {
CheckInputSize2D(M_PER_PI[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
string sk(key);
if (StringMatch(sk, VAR_CONDUCT[0])) m_ks = data;
else if (StringMatch(sk, VAR_SOILTHICK[0])) m_soilThk = data;
else if (StringMatch(sk, VAR_SOL_AWC[0])) m_soilFC = data;
else if (StringMatch(sk, VAR_SOL_WPMM[0])) m_soilWP = data;
else if (StringMatch(sk, VAR_POREIDX[0])) m_poreIdx = data;
else if (StringMatch(sk, VAR_SOL_UL[0])) m_soilSat = data;
else if (StringMatch(sk, VAR_SOL_ST[0])) m_soilWtrSto = data;
else {
throw ModelException(M_PER_PI[0], "Set2DData", "Parameter " + sk + " does not exist.");
}
}

void PER_PI::SetValue(const char* key, const float value) {
string s(key);
if (StringMatch(s, Tag_TimeStep[0])) m_dt = CVT_INT(value);
else if (StringMatch(s, VAR_T_SOIL[0])) m_soilFrozenTemp = value;
else {
throw ModelException(M_PER_PI[0], "SetValue", "Parameter " + s + " does not exist.");
}
}

bool PER_PI::CheckInputData() {
CHECK_POSITIVE(M_PER_PI[0], m_date);
CHECK_POSITIVE(M_PER_PI[0], m_nCells);
CHECK_POSITIVE(M_PER_PI[0], m_dt);
CHECK_POINTER(M_PER_PI[0], m_ks);
CHECK_POINTER(M_PER_PI[0], m_soilSat);
CHECK_POINTER(M_PER_PI[0], m_poreIdx);
CHECK_POINTER(M_PER_PI[0], m_soilFC);
CHECK_POINTER(M_PER_PI[0], m_soilWP);
CHECK_POINTER(M_PER_PI[0], m_soilThk);
CHECK_POINTER(M_PER_PI[0], m_soilTemp);
CHECK_POINTER(M_PER_PI[0], m_infil);
CHECK_NODATA(M_PER_PI[0], m_soilFrozenTemp);
CHECK_POINTER(M_PER_PI[0], m_soilWtrSto);
CHECK_POINTER(M_PER_PI[0], m_soilWtrStoPrfl);
return true;
}
