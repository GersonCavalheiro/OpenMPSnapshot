#include "SSR_DA.h"

#include "text.h"

SSR_DA::SSR_DA() :
m_inputSubbsnID(-1), m_nCells(-1), m_CellWth(-1.), m_maxSoilLyrs(-1), m_nSoilLyrs(nullptr),
m_soilThk(nullptr),
m_dt(-1), m_ki(NODATA_VALUE),
m_soilFrozenTemp(NODATA_VALUE), m_slope(nullptr), m_ks(nullptr), m_soilSat(nullptr),
m_poreIdx(nullptr),
m_soilFC(nullptr), m_soilWP(nullptr),
m_soilWtrSto(nullptr), m_soilWtrStoPrfl(nullptr), m_soilTemp(nullptr), m_chWidth(nullptr),
m_rchID(nullptr), m_flowInIdx(nullptr), m_flowInFrac(nullptr), m_rteLyrs(nullptr),
m_nRteLyrs(-1), m_nSubbsns(-1), m_subbsnID(nullptr),
m_subSurfRf(nullptr), m_subSurfRfVol(nullptr), m_ifluQ2Rch(nullptr) {
}

SSR_DA::~SSR_DA() {
if (m_subSurfRf != nullptr) Release2DArray(m_subSurfRf);
if (m_subSurfRfVol != nullptr) Release2DArray(m_subSurfRfVol);
if (m_ifluQ2Rch != nullptr) Release1DArray(m_ifluQ2Rch);
}

FLTPT SSR_DA::GetFlowInFraction(const int id, const int up_idx) {
if (nullptr == m_flowInFrac) return 1.;
return m_flowInFrac[id][up_idx];
}

bool SSR_DA::FlowInSoil(const int id) {
FLTPT s0 = Max(m_slope[id], 0.01);
FLTPT flowWidth = m_CellWth;
if (m_rchID[id] > 0) {
flowWidth -= m_chWidth[id];
}
for (int j = 0; j < CVT_INT(m_nSoilLyrs[id]); j++) {
m_subSurfRf[id][j] = 0.;
m_subSurfRfVol[id][j] = 0.;
}




int nUpstream = CVT_INT(m_flowInIdx[id][0]);
m_soilWtrStoPrfl[id] = 0.; 
for (int j = 0; j < CVT_INT(m_nSoilLyrs[id]); j++) {
FLTPT smOld = m_soilWtrSto[id][j]; 
FLTPT qUp = 0.;    
FLTPT qUpVol = 0.; 
for (int upIndex = 1; upIndex <= nUpstream; upIndex++) {
int flowInID = CVT_INT(m_flowInIdx[id][upIndex]);
if (CVT_INT(m_subbsnID[flowInID]) != CVT_INT(m_subbsnID[id])) { continue; }
if (m_subSurfRf[flowInID][j] < 0.) { continue; }
qUp += m_subSurfRf[flowInID][j] * GetFlowInFraction(id, upIndex);
qUpVol += m_subSurfRfVol[flowInID][j] * GetFlowInFraction(id, upIndex);
}
if (qUp <= 0. || qUpVol <= 0.) {
qUp = 0.;
qUpVol = 0.;
}
if (flowWidth <= 0.) {
m_subSurfRf[id][j] = qUp;
m_subSurfRfVol[id][j] = qUpVol;
continue;
}
if (m_soilWtrSto[id][j] != m_soilWtrSto[id][j] || m_soilWtrSto[id][j] < 0.) {
cout << "cell id: " << id << ", layer: " << j << ", moisture is less than zero: "
<< m_soilWtrSto[id][j] << ", previous: " << smOld << ", qUp: " << qUp << ", depth:"
<< m_soilThk[id][j] << endl;
return false;
}

m_soilWtrSto[id][j] += qUp; 

if (m_soilWtrSto[id][j] <= m_soilFC[id][j]) continue;
if (j == 0 && m_soilTemp[id] <= m_soilFrozenTemp && qUp <= 0.) {
continue;
}

FLTPT k = 0.;
FLTPT maxSoilWater = 0.;
FLTPT soilWater = 0.;
FLTPT fcSoilWater = 0.;
soilWater = m_soilWtrSto[id][j];
maxSoilWater = m_soilSat[id][j];
fcSoilWater = m_soilFC[id][j];
if (m_soilWtrSto[id][j] > maxSoilWater) {
k = m_ks[id][j];
} else {
FLTPT dcIndex = 2. * m_poreIdx[id][j] + 3.; 
k = m_ks[id][j] * CalPow(m_soilWtrSto[id][j] / maxSoilWater, dcIndex);
if (k <= UTIL_ZERO) k = 0.;
}
m_subSurfRf[id][j] = m_ki * s0 * k * m_dt * 0.0002777777777777778 * m_soilThk[id][j] * 0.001 / flowWidth;

if (soilWater - m_subSurfRf[id][j] > maxSoilWater) {
m_subSurfRf[id][j] = soilWater - maxSoilWater;
} else if (soilWater - m_subSurfRf[id][j] < fcSoilWater) {
m_subSurfRf[id][j] = soilWater - fcSoilWater;
}
m_subSurfRf[id][j] = Max(0., m_subSurfRf[id][j]);

m_subSurfRfVol[id][j] = m_subSurfRf[id][j] * 0.001 * m_CellWth * flowWidth; 
m_subSurfRfVol[id][j] = Max(UTIL_ZERO, m_subSurfRfVol[id][j]);
m_soilWtrSto[id][j] -= m_subSurfRf[id][j];
m_soilWtrSto[id][j] = Max(UTIL_ZERO, m_soilWtrSto[id][j]);
m_soilWtrStoPrfl[id] += m_soilWtrSto[id][j];

if (isinf(m_soilWtrSto[id][j]) ||isnan(m_soilWtrSto[id][j]) || m_soilWtrSto[id][j] < 0.) {
cout << "cell id: " << id << ", layer: " << j << ", moisture is less than zero: "
<< m_soilWtrSto[id][j] << ", subsurface runoff: " << m_subSurfRf[id][j] << ", depth:"
<< m_soilThk[id][j] << endl;
return false;
}
}
return true;
}

int SSR_DA::Execute() {
CheckInputData();
InitialOutputs();

for (int ilyr = 0; ilyr < m_nRteLyrs; ilyr++) {
int ncells = CVT_INT(m_rteLyrs[ilyr][0]);
int errCount = 0;
#pragma omp parallel for reduction(+: errCount)
for (int icell = 1; icell <= ncells; icell++) {
int id = m_rteLyrs[ilyr][icell];
if (!FlowInSoil(id)) errCount++;
}
if (errCount > 0) {
throw ModelException(M_SSR_DA[0], "Execute:FlowInSoil",
"Please check the error message for more information");
}
}


for (int i = 0; i <= m_nSubbsns; i++) {
m_ifluQ2Rch[i] = 0.;
}
#pragma omp parallel
{
FLTPT* tmp_qiSubbsn = new(nothrow) FLTPT[m_nSubbsns + 1];
for (int i = 0; i <= m_nSubbsns; i++) {
tmp_qiSubbsn[i] = 0.;
}
#pragma omp for
for (int i = 0; i < m_nCells; i++) {
if (m_rchID[i] <= 0) continue;
FLTPT qiAllLayers = 0.;
for (int j = 0; j < CVT_INT(m_nSoilLyrs[i]); j++) {
if (m_subSurfRfVol[i][j] > UTIL_ZERO) {
qiAllLayers += m_subSurfRfVol[i][j] / m_dt; 
}
}
tmp_qiSubbsn[CVT_INT(m_rchID[i])] += qiAllLayers;
}
#pragma omp critical
{
for (int i = 1; i <= m_nSubbsns; i++) {
m_ifluQ2Rch[i] += tmp_qiSubbsn[i];
}
}
delete[] tmp_qiSubbsn;
tmp_qiSubbsn = nullptr;
} 

for (int i = 1; i <= m_nSubbsns; i++) {
m_ifluQ2Rch[0] += m_ifluQ2Rch[i];
}
return 0;
}

void SSR_DA::SetValue(const char* key, const FLTPT value) {
string s(key);
if (StringMatch(s, VAR_T_SOIL[0])) {
m_soilFrozenTemp = value;
} else if (StringMatch(s, VAR_KI[0])) {
m_ki = value;
} else if (StringMatch(s, Tag_CellWidth[0])) {
m_CellWth = value;
} else {
throw ModelException(M_SSR_DA[0], "SetValue",
"Parameter " + s + " does not exist.");
}
}

void SSR_DA::SetValue(const char* key, const int value) {
string s(key);
if (StringMatch(s, VAR_SUBBSNID_NUM[0])) {
m_nSubbsns = value;
} else if (StringMatch(s, Tag_SubbasinId)) {
m_inputSubbsnID = value;
} else if (StringMatch(s, Tag_TimeStep[0])) {
m_dt = value;
} else {
throw ModelException(M_SSR_DA[0], "SetValue",
"Integer Parameter " + s + " does not exist.");
}
}

void SSR_DA::Set1DData(const char* key, const int nrows, FLTPT* data) {
string s(key);
CheckInputSize(M_SSR_DA[0], key, nrows, m_nCells);
if (StringMatch(s, VAR_SLOPE[0])) {
m_slope = data;
} else if (StringMatch(s, VAR_CHWIDTH[0])) {
m_chWidth = data;
} else if (StringMatch(s, VAR_SOTE[0])) {
m_soilTemp = data;
} else if (StringMatch(s, VAR_SOL_SW[0])) {
m_soilWtrStoPrfl = data;
} else {
throw ModelException(M_SSR_DA[0], "Set1DData",
"Parameter " + s + " does not exist.");
}
}

void SSR_DA::Set1DData(const char* key, const int nrows, int* data) {
string s(key);
CheckInputSize(M_SSR_DA[0], key, nrows, m_nCells);
if (StringMatch(s, VAR_STREAM_LINK[0])) {
m_rchID = data;
} else if (StringMatch(s, VAR_SUBBSN[0])) {
m_subbsnID = data;
} else if (StringMatch(s, VAR_SOILLAYERS[0])) {
m_nSoilLyrs = data;
} else {
throw ModelException(M_SSR_DA[0], "Set1DData",
"Integer Parameter " + s + " does not exist.");
}
}

void SSR_DA::Set2DData(const char* key, const int nrows, const int ncols, FLTPT** data) {
string sk(key);
if (StringMatch(sk, VAR_SOILTHICK[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_soilThk = data;
} else if (StringMatch(sk, VAR_CONDUCT[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_ks = data;
} else if (StringMatch(sk, VAR_SOL_UL[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_soilSat = data;
} else if (StringMatch(sk, VAR_SOL_AWC[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_soilFC = data;
} else if (StringMatch(sk, VAR_SOL_WPMM[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_soilWP = data;
} else if (StringMatch(sk, VAR_POREIDX[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_poreIdx = data;
} else if (StringMatch(sk, VAR_SOL_ST[0])) {
CheckInputSize2D(M_SSR_DA[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
m_soilWtrSto = data;
} else if (StringMatch(sk, Tag_FLOWIN_FRACTION[0])) {
CheckInputSize(M_SSR_DA[0], key, nrows, m_nCells);
m_flowInFrac = data;
} else {
throw ModelException(M_SSR_DA[0], "Set2DData",
"Parameter " + sk + " does not exist.");
}
}

void SSR_DA::Set2DData(const char* key, const int nrows, const int ncols, int** data) {
string sk(key);
if (StringMatch(sk, Tag_ROUTING_LAYERS[0])) {
CheckInputSize(M_SSR_DA[0], key, nrows, m_nRteLyrs);
m_rteLyrs = data;
} else if (StringMatch(sk, Tag_FLOWIN_INDEX[0])) {
CheckInputSize(M_SSR_DA[0], key, nrows, m_nCells);
m_flowInIdx = data;
} else {
throw ModelException(M_SSR_DA[0], "Set2DData",
"Integer Parameter " + sk + " does not exist.");
}
}

void SSR_DA::Get1DData(const char* key, int* n, FLTPT** data) {
InitialOutputs();
string sk(key);
if (StringMatch(sk, VAR_SBIF[0])) *data = m_ifluQ2Rch;
else {
throw ModelException(M_SSR_DA[0], "Get1DData",
"Result " + sk + " does not exist.");
}
*n = m_nSubbsns + 1;
}

void SSR_DA::Get2DData(const char* key, int* nrows, int* ncols, FLTPT*** data) {
InitialOutputs();
string sk(key);
*nrows = m_nCells;
*ncols = m_maxSoilLyrs;

if (StringMatch(sk, VAR_SSRU[0])) {
*data = m_subSurfRf;
} else if (StringMatch(sk, VAR_SSRUVOL[0])) {
*data = m_subSurfRfVol;
} else {
throw ModelException(M_SSR_DA[0], "Get2DData",
"Output " + sk + " does not exist.");
}
}

bool SSR_DA::CheckInputData() {
CHECK_NONNEGATIVE(M_SSR_DA[0], m_inputSubbsnID);
CHECK_POSITIVE(M_SSR_DA[0], m_nCells);
CHECK_POSITIVE(M_SSR_DA[0], m_ki);
CHECK_NODATA(M_SSR_DA[0], m_soilFrozenTemp);
CHECK_POSITIVE(M_SSR_DA[0], m_dt);
CHECK_POSITIVE(M_SSR_DA[0], m_CellWth);
CHECK_POSITIVE(M_SSR_DA[0], m_nSubbsns);
CHECK_POSITIVE(M_SSR_DA[0], m_nRteLyrs);
CHECK_POINTER(M_SSR_DA[0], m_subbsnID);
CHECK_POINTER(M_SSR_DA[0], m_nSoilLyrs);
CHECK_POINTER(M_SSR_DA[0], m_soilThk);
CHECK_POINTER(M_SSR_DA[0], m_slope);
CHECK_POINTER(M_SSR_DA[0], m_poreIdx);
CHECK_POINTER(M_SSR_DA[0], m_ks);
CHECK_POINTER(M_SSR_DA[0], m_soilSat);
CHECK_POINTER(M_SSR_DA[0], m_soilFC);
CHECK_POINTER(M_SSR_DA[0], m_soilWP);
CHECK_POINTER(M_SSR_DA[0], m_soilWtrSto);
CHECK_POINTER(M_SSR_DA[0], m_soilWtrStoPrfl);
CHECK_POINTER(M_SSR_DA[0], m_soilTemp);
CHECK_POINTER(M_SSR_DA[0], m_chWidth);
CHECK_POINTER(M_SSR_DA[0], m_rchID);
CHECK_POINTER(M_SSR_DA[0], m_flowInIdx);
CHECK_POINTER(M_SSR_DA[0], m_rteLyrs);



return true;
}

void SSR_DA::InitialOutputs() {
CHECK_POSITIVE(M_SSR_DA[0], m_nCells);
CHECK_POSITIVE(M_SSR_DA[0], m_nSubbsns);
if (nullptr == m_ifluQ2Rch) Initialize1DArray(m_nSubbsns + 1, m_ifluQ2Rch, 0.);
if (nullptr == m_subSurfRf) Initialize2DArray(m_nCells, m_maxSoilLyrs, m_subSurfRf, 0.);
if (nullptr == m_subSurfRfVol) Initialize2DArray(m_nCells, m_maxSoilLyrs, m_subSurfRfVol, 0.);
}