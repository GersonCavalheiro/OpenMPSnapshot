#include "IKW_CH.h"
#include "text.h"


ImplicitKinematicWave_CH::ImplicitKinematicWave_CH() :
m_nCells(-1), m_chNumber(-1), m_dt(-1.0f),
m_CellWidth(-1.0f), 
m_sRadian(nullptr), m_direction(nullptr), m_reachDownStream(nullptr),
m_chWidth(nullptr),
m_qs(nullptr), m_hCh(nullptr), m_qCh(nullptr), m_prec(nullptr),
m_qSubbasin(nullptr), m_qg(nullptr),
m_flowLen(nullptr), m_qi(nullptr), m_streamLink(nullptr),
m_sourceCellIds(nullptr),
m_idUpReach(-1), m_qUpReach(0.f),
m_qgDeep(100.f),
m_idOutlet(-1)
{
}

ImplicitKinematicWave_CH::~ImplicitKinematicWave_CH(void) {
Release2DArray(m_hCh);
Release2DArray(m_qCh);
Release2DArray(m_flowLen);

Release1DArray(m_sourceCellIds);
Release1DArray(m_qSubbasin);
}


float ImplicitKinematicWave_CH::GetNewQ(float qIn, float qLast, float surplus, float alpha, float dt, float dx) {

float ab_pQ, dtX, C;  
int count;
float Qkx; 
float fQkx; 
float dfQkx;  
const float _epsilon = 1e-12f;
const float beta = 0.6f;


if ((qIn + qLast) <= -surplus * dx)
{
return (0);
}


ab_pQ = alpha * beta * CalPow(((qLast + qIn) / 2), beta - 1);

dtX = dt / dx;
C = dtX * qIn + alpha * CalPow(qLast, beta) + dt * surplus;
Qkx = (dtX * qIn + qLast * ab_pQ + dt * surplus) / (dtX + ab_pQ);

if (Qkx < MIN_FLUX) {
return (0);
}

Qkx = Max(Qkx, MIN_FLUX);

count = 0;
do {
fQkx = dtX * Qkx + alpha * Power(Qkx, beta) - C;   
dfQkx = dtX + alpha * beta * Power(Qkx, beta - 1);  
Qkx -= fQkx / dfQkx;                                
Qkx = Max(Qkx, MIN_FLUX);
count++;
} while (Abs(fQkx) > _epsilon && count < MAX_ITERS_KW);

if (Qkx != Qkx) {
throw ModelException(M_IKW_CH[0], "GetNewQ", "Error in iteration!");
}

return Qkx;
}


bool ImplicitKinematicWave_CH::CheckInputData(void) {
if (m_date <= 0) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "You have not set the Date variable.");
}

if (m_nCells <= 0) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "The cell number of the input can not be less than zero.");
}

if (m_dt <= 0) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "You have not set the TimeStep variable.");
}

if (m_CellWidth <= 0) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "You have not set the CellWidth variable.");
}

if (m_sRadian == nullptr) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "The parameter: RadianSlope has not been set.");
}
if (m_direction == nullptr) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "The parameter: flow direction has not been set.");
}

if (m_chWidth == nullptr) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "The parameter: CHWIDTH has not been set.");
}
if (m_streamLink == nullptr) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "The parameter: STREAM_LINK has not been set.");
}

if (m_prec == nullptr) {
throw ModelException(M_IKW_CH[0], "CheckInputData", "The parameter: D_P(precipitation) has not been set.");
}

return true;
}

void ImplicitKinematicWave_CH:: InitialOutputs() {
if (m_nCells <= 0) {
throw ModelException(M_IKW_CH[0], "InitialOutputs", "The cell number of the input can not be less than zero.");
}

if (m_hCh == nullptr) {
m_sourceCellIds = new int[m_chNumber];

for (int i = 0; i < m_chNumber; ++i) {
m_sourceCellIds[i] = -1;
}

for (int i = 0; i < m_nCells; i++) {
if (FloatEqual(m_streamLink[i], NODATA_VALUE)) {
continue;
}
int reachId = (int) m_streamLink[i];
bool isSource = true;
for (int k = 1; k <= (int) m_flowInIdx[i][0]; ++k) {
int flowInId = (int) m_flowInIdx[i][k];
int flowInReachId = (int) m_streamLink[flowInId];
if (flowInReachId == reachId) {
isSource = false;
break;
}
}

if ((int) m_flowInIdx[i][0] == 0) {
isSource = true;
}

if (isSource) {
int reachIndex = m_idToIndex[reachId];
m_sourceCellIds[reachIndex] = i;
}
}


for (int iCh = 0; iCh < m_chNumber; iCh++) {
int iCell = m_sourceCellIds[iCh];
int reachId = (int) m_streamLink[iCell];
while ((int) m_streamLink[iCell] == reachId) {
m_reachs[iCh].push_back(iCell);
iCell = (int) m_flowOutIdx[iCell];
}
}

m_hCh = new float *[m_chNumber];
m_qCh = new float *[m_chNumber];


m_qSubbasin = new float[m_chNumber];
for (int i = 0; i < m_chNumber; ++i) {
int n = CVT_INT(m_reachs[i].size());
m_hCh[i] = new float[n];
m_qCh[i] = new float[n];


m_qSubbasin[i] = 0.f;

for (int j = 0; j < n; ++j) {
m_hCh[i][j] = 0.f;
m_qCh[i][j] = 0.f;
}
}

}

}

void ImplicitKinematicWave_CH::initialOutputs2() {
if (m_flowLen != nullptr) {
return;
}

m_flowLen = new float *[m_chNumber];

for (int i = 0; i < m_chNumber; ++i) {
int n = m_reachs[i].size();
m_flowLen[i] = new float[n];

int id;
float dx;
for (int j = 0; j < n; ++j) {
id = m_reachs[i][j];
dx = m_CellWidth / cos(m_sRadian[id]);
int dir = (int) m_direction[id];
if (DiagonalCCW[dir] == 1) {
dx = SQ2 * dx;
}
m_flowLen[i][j] = dx;
}
}
}

void ImplicitKinematicWave_CH::ChannelFlow(int iReach, int iCell, int id, float qgEachCell) {
float qUp = 0.f;

if (iReach == 0 && iCell == 0) {
qUp = m_qUpReach;
}

if (iCell == 0)
{
for (size_t i = 0; i < m_reachUpStream[iReach].size(); ++i) {
int upReachId = m_reachUpStream[iReach][i];
if (upReachId >= 0) {
int upCellsNum = CVT_INT(m_reachs[upReachId].size());
int upCellId = m_reachs[upReachId][upCellsNum - 1];
qUp += m_qCh[upReachId][upCellsNum - 1];
}
}
} else {
qUp = m_qCh[iReach][iCell - 1];
}

float dx = m_flowLen[iReach][iCell];

float qLat = m_prec[id] / 1000.f * m_chWidth[id] * dx / m_dt;
qLat += qgEachCell;

qLat += m_qs[id];
if (m_qi != nullptr) {
qLat += m_qi[id];
}

if (qLat < MIN_FLUX && qUp < MIN_FLUX) {
m_hCh[iReach][iCell] = 0.f;
m_qCh[iReach][iCell] = 0.f;
return;
}

qUp += qLat;

float Perim = 2.f * m_hCh[iReach][iCell] + m_chWidth[id];

float sSin = CalSqrt(sin(m_sRadian[id]));
float alpha = CalPow(m_reachN[iReach] / sSin * CalPow(Perim, _2div3), 0.6f);

float qIn = m_qCh[iReach][iCell];

m_qCh[iReach][iCell] = GetNewQ(qUp, qIn, 0.f, alpha, m_dt, dx);

float hTest = m_hCh[iReach][iCell] + (qUp - m_qCh[iReach][iCell]) * m_dt / m_chWidth[id] / dx;
float hNew = (alpha * CalPow(m_qCh[iReach][iCell], 0.6f)) / m_chWidth[id]; 
m_hCh[iReach][iCell] = (alpha * CalPow(m_qCh[iReach][iCell], 0.6f)) / m_chWidth[id]; 
}

int ImplicitKinematicWave_CH::Execute() {
CheckInputData();

InitialOutputs();
initialOutputs2();

for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
int nReaches = it->second.size();
#pragma omp parallel for
for (int i = 0; i < nReaches; ++i) {
int reachIndex = it->second[i]; 

vector<int> &vecCells = m_reachs[reachIndex];
int n = vecCells.size();
float qgEachCell = 0.f;
if (m_qg != nullptr) {
qgEachCell = m_qg[i + 1] / n;
}
for (int iCell = 0; iCell < n; ++iCell) {
int idCell = vecCells[iCell];
ChannelFlow(reachIndex, iCell, idCell, qgEachCell);
}
m_qSubbasin[reachIndex] = m_qCh[reachIndex][n - 1];
}
}

return 0;
}

bool ImplicitKinematicWave_CH::CheckInputSizeChannel(const char *key, int n) {
if (n <= 0) {
return false;
}
if (m_chNumber != n) {
if (m_chNumber <= 0) { m_chNumber = n; }
else {
return false;
}
}

return true;
}

void ImplicitKinematicWave_CH::GetValue(const char *key, float *value) {
string sk(key);
if (StringMatch(sk, VAR_QTOTAL[0])) {
auto it = m_reachLayers.end();
--it;
int reachId = it->second[0];
int iLastCell = CVT_INT(m_reachs[reachId].size()) - 1;
*value = m_qCh[reachId][iLastCell] + m_qgDeep;
}
else {
throw ModelException(M_IKW_CH[0], "GetValue",
"Output " + sk + " does not exist.");
}
}

void ImplicitKinematicWave_CH::SetValue(const char *key, float value) {
string sk(key);
if (StringMatch(sk, Tag_HillSlopeTimeStep[0])) {
m_dt = value;
} else if (StringMatch(sk, Tag_CellWidth[0])) {
m_CellWidth = value;
} else {
throw ModelException(M_IKW_CH[0], "SetValue",
"Parameter " + sk + " does not exist.");
}

}

void ImplicitKinematicWave_CH::Set1DData(const char *key, int n, float *data) {
string sk(key);

if (StringMatch(sk, VAR_SBQG[0])) {
CheckInputSize(M_IKW_CH[0], key, n, m_chNumber);
m_qg = data;
return;
}

CheckInputSize(M_IKW_CH[0], key, n, m_nCells);

if (StringMatch(sk, VAR_RadianSlope[0])) {
m_sRadian = data;
} else if (StringMatch(sk, VAR_FLOWDIR[0])) {
m_direction = data;
} else if (StringMatch(sk, VAR_PCP[0])) {
m_prec = data;
} else if (StringMatch(sk, VAR_QSOIL[0])) {
m_qi = data;
} else if (StringMatch(sk, VAR_QOVERLAND[0])) {
m_qs = data;
} else if (StringMatch(sk, VAR_CHWIDTH[0])) {
m_chWidth = data;
} else if (StringMatch(sk, VAR_STREAM_LINK[0])) {
m_streamLink = data;
} else if (StringMatch(sk, Tag_FLOWOUT_INDEX[0])) {
m_flowOutIdx = data;
for (int i = 0; i < m_nCells; i++) {
if (m_flowOutIdx[i] < 0) {
m_idOutlet = i;
break;
}
}
} else {
throw ModelException(M_IKW_CH[0], "Set1DData",
"Parameter " + sk + " does not exist.");
}
}

void ImplicitKinematicWave_CH::Get1DData(const char *key, int *n, float **data) {
string sk(key);
*n = m_chNumber;
if (StringMatch(sk, VAR_QRECH[0])) {
*data = m_qSubbasin;
}
else if (StringMatch(sk, VAR_QRECH[0])) {
auto it = m_reachLayers.end();
--it;
int reachId = it->second[0];
*data = m_qCh[reachId];
} else {
throw ModelException(M_IKW_CH[0], "Get1DData",
"Output " + sk + " does not exist.");
}
}

void ImplicitKinematicWave_CH::Get2DData(const char *key, int *nrows, int *ncols, float ***data) {
if (m_hCh == nullptr) { 
InitialOutputs();
}
string sk(key);
*nrows = m_chNumber;
if (StringMatch(sk, VAR_HCH[0])) {
*data = m_hCh;
} else {
throw ModelException(M_IKW_CH[0], "Get2DData",
"Output " + sk + " does not exist.");
}
}

void ImplicitKinematicWave_CH::Set2DData(const char *key, int nrows, int ncols, float **data) {
string sk(key);
if (StringMatch(sk, Tag_FLOWIN_INDEX[0])) {
m_flowInIdx = data;
} else {
throw ModelException(M_IKW_CH[0], "Set1DData",
"Parameter " + sk + " does not exist.");
}
}

void ImplicitKinematicWave_CH::SetReaches(clsReaches *reaches) {
if (nullptr == reaches) {
throw ModelException(M_IKW_CH[0], "SetReaches",
"The reaches input can not to be nullptr.");
}
m_chNumber = reaches->GetReachNumber();

if (nullptr == m_reachDownStream) reaches->GetReachesSingleProperty(REACH_DOWNSTREAM, &m_reachDownStream);
if (nullptr == m_chWidth) reaches->GetReachesSingleProperty(REACH_WIDTH, &m_chWidth);
if (nullptr == m_reachN) reaches->GetReachesSingleProperty(REACH_MANNING, &m_reachN);

m_reachUpStream = reaches->GetUpStreamIDs();
m_reachLayers = reaches->GetReachLayers();
}
