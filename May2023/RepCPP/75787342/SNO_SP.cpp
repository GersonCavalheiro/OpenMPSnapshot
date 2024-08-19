#include "SNO_SP.h"

#include "text.h"
#include "PlantGrowthCommon.h"
#include "utils_time.h"


SNO_SP::SNO_SP() :
m_nCells(-1), m_t0(NODATA_VALUE), m_kblow(NODATA_VALUE), m_snowTemp(NODATA_VALUE),
m_lagSnow(NODATA_VALUE), m_csnow6(NODATA_VALUE), m_csnow12(NODATA_VALUE),
m_snowCoverMax(NODATA_VALUE), m_snowCover50(NODATA_VALUE),
m_snowCoverCoef1(NODATA_VALUE), m_snowCoverCoef2(NODATA_VALUE),
m_meanTemp(nullptr), m_maxTemp(nullptr), m_netPcp(nullptr),
m_snowAccum(nullptr), m_SE(nullptr), m_packT(nullptr), m_snowMelt(nullptr), m_SA(nullptr) {
}

SNO_SP::~SNO_SP() {
if (m_snowMelt != nullptr) Release1DArray(m_snowMelt);
if (m_SA != nullptr) Release1DArray(m_SA);
if (m_packT != nullptr) Release1DArray(m_packT);
}

bool SNO_SP::CheckInputData() {
CHECK_POSITIVE(M_SNO_SP[0], m_nCells);
CHECK_NODATA(M_SNO_SP[0], m_t0);
CHECK_NODATA(M_SNO_SP[0], m_kblow);
CHECK_NODATA(M_SNO_SP[0], m_snowTemp);
CHECK_NODATA(M_SNO_SP[0], m_lagSnow);
CHECK_NODATA(M_SNO_SP[0], m_csnow6);
CHECK_NODATA(M_SNO_SP[0], m_csnow12);
CHECK_NODATA(M_SNO_SP[0], m_snowCoverMax);
CHECK_NODATA(M_SNO_SP[0], m_snowCover50);
CHECK_POINTER(M_SNO_SP[0], m_meanTemp);
CHECK_POINTER(M_SNO_SP[0], m_maxTemp);
CHECK_POINTER(M_SNO_SP[0], m_netPcp);
return true;
}

void SNO_SP::InitialOutputs() {
CHECK_POSITIVE(M_SNO_SP[0], m_nCells);
if (nullptr == m_snowMelt) Initialize1DArray(m_nCells, m_snowMelt, 0.);
if (nullptr == m_SA) Initialize1DArray(m_nCells, m_SA, 0.);
if (nullptr == m_packT) Initialize1DArray(m_nCells, m_packT, 0.);
if (nullptr == m_snowAccum) {
Initialize1DArray(m_nCells, m_snowAccum, 0.);
}
if (nullptr == m_SE) {
Initialize1DArray(m_nCells, m_SE, 0.);
}
}

int SNO_SP::Execute() {
CheckInputData();
InitialOutputs();
if (FloatEqual(m_snowCoverCoef1, NODATA_VALUE) || FloatEqual(m_snowCoverCoef2, NODATA_VALUE)) {
GetScurveShapeParameter(0.5, 0.95, m_snowCover50, 0.95,
&m_snowCoverCoef1, &m_snowCoverCoef2);
}
FLTPT sinv = CVT_FLT(sin(2. * PI / 365. * (m_dayOfYear - 81.)));
FLTPT cmelt = (m_csnow6 + m_csnow12) * 0.5 + (m_csnow6 - m_csnow12) * 0.5 * sinv;
#pragma omp parallel for
for (int rw = 0; rw < m_nCells; rw++) {
m_packT[rw] = m_packT[rw] * (1. - m_lagSnow) + m_meanTemp[rw] * m_lagSnow;
m_SA[rw] += m_snowAccum[rw] - m_SE[rw];
if (m_meanTemp[rw] < m_snowTemp) {
m_SA[rw] += m_kblow * m_netPcp[rw];
m_netPcp[rw] *= (1. - m_kblow);
}

if (m_SA[rw] < 0.01) {
m_snowMelt[rw] = 0.;
} else {
FLTPT dt = m_maxTemp[rw] - m_t0;
if (dt < 0) {
m_snowMelt[rw] = 0.; 
} else {
m_snowMelt[rw] = cmelt * ((m_packT[rw] + m_maxTemp[rw]) * 0.5 - m_t0);
FLTPT snowCoverFrac = 0.; 
if (m_SA[rw] < m_snowCoverMax) {
FLTPT xx = m_SA[rw] / m_snowCoverMax;
snowCoverFrac = xx / (xx + CalExp(m_snowCoverCoef1 = m_snowCoverCoef2 * xx));
} else {
snowCoverFrac = 1.;
}
m_snowMelt[rw] *= snowCoverFrac;
if (m_snowMelt[rw] < 0.) m_snowMelt[rw] = 0.;
if (m_snowMelt[rw] > m_SA[rw]) m_snowMelt[rw] = m_SA[rw];
m_SA[rw] -= m_snowMelt[rw];
m_netPcp[rw] += m_snowMelt[rw];
if (m_netPcp[rw] < 0.) m_netPcp[rw] = 0.;
}
}
}
return 0;
}

void SNO_SP::SetValue(const char* key, const FLTPT value) {
string s(key);
if (StringMatch(s, VAR_K_BLOW[0])) m_kblow = value;
else if (StringMatch(s, VAR_T0[0])) m_t0 = value;
else if (StringMatch(s, VAR_T_SNOW[0])) m_snowTemp = value;
else if (StringMatch(s, VAR_LAG_SNOW[0])) m_lagSnow = value;
else if (StringMatch(s, VAR_C_SNOW6[0])) m_csnow6 = value;
else if (StringMatch(s, VAR_C_SNOW12[0])) m_csnow12 = value;
else if (StringMatch(s, VAR_SNOCOVMX[0])) m_snowCoverMax = value;
else if (StringMatch(s, VAR_SNO50COV[0])) m_snowCover50 = value;
else {
throw ModelException(M_SNO_SP[0], "SetValue",
"Parameter " + s + " does not exist.");
}
}

void SNO_SP::Set1DData(const char* key, const int n, FLTPT* data) {
CheckInputSize(M_SNO_SP[0], key, n, m_nCells);
string s(key);
if (StringMatch(s, VAR_TMEAN[0])) m_meanTemp = data;
else if (StringMatch(s, VAR_TMAX[0])) m_maxTemp = data;
else if (StringMatch(s, VAR_NEPR[0])) m_netPcp = data;
else {
throw ModelException(M_SNO_SP[0], "Set1DData",
"Parameter " + s + " does not exist.");
}
}

void SNO_SP::Get1DData(const char* key, int* n, FLTPT** data) {
InitialOutputs();
string s(key);
if (StringMatch(s, VAR_SNME[0])) *data = m_snowMelt;
else if (StringMatch(s, VAR_SNAC[0])) *data = m_snowAccum;
else {
throw ModelException(M_SNO_SP[0], "Get1DData",
"Result " + s + " does not exist.");
}
*n = m_nCells;
}
