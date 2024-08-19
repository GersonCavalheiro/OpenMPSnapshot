#include "IKW_REACH.h"
#include "text.h"


IKW_REACH::IKW_REACH() : m_dt(-1), m_nreach(-1), m_Kchb(nullptr),
m_Kbank(nullptr), m_Epch(NODATA_VALUE), m_Bnk0(NODATA_VALUE), m_Chs0(NODATA_VALUE),
m_aBank(NODATA_VALUE),
m_bBank(NODATA_VALUE), m_subbasin(nullptr), m_qsSub(nullptr),
m_qiSub(nullptr), m_qgSub(nullptr), m_petCh(nullptr), m_gwStorage(nullptr), m_area(nullptr),
m_Vseep0(0.f), m_chManning(nullptr), m_chSlope(nullptr),m_chWTdepth(nullptr),
m_bankStorage(nullptr), m_seepage(nullptr),
m_qsCh(nullptr), m_qiCh(nullptr), m_qgCh(nullptr),
m_x(0.2f), m_co1(0.7f), m_qIn(nullptr), m_chStorage(nullptr),
m_qUpReach(0.f), m_deepGroudwater(0.f) {
}

IKW_REACH::~IKW_REACH() {

if (nullptr != m_chStorage) Release1DArray(m_chStorage);
if (nullptr != m_qOut) Release1DArray(m_qOut);
if (nullptr != m_bankStorage) Release1DArray(m_bankStorage);
if (nullptr != m_seepage) Release1DArray(m_seepage);
if (nullptr != m_chStorage) Release1DArray(m_chStorage);
if (nullptr != m_qsCh) Release1DArray(m_qsCh);
if (nullptr != m_qiCh) Release1DArray(m_qiCh);
if (nullptr != m_qgCh) Release1DArray(m_qgCh);
if (nullptr != m_chWTdepth) Release1DArray(m_chWTdepth);
}

bool IKW_REACH::CheckInputData() {
if (m_dt < 0) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: m_dt has not been set.");
}

if (m_nreach < 0) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: m_nreach has not been set.");
}

if (nullptr == m_Kchb) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: K_chb has not been set.");
}
if (nullptr == m_Kbank) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: K_bank has not been set.");
}
if (FloatEqual(m_Epch, NODATA_VALUE)) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: Ep_ch has not been set.");
}
if (FloatEqual(m_Bnk0, NODATA_VALUE)) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: Bnk0 has not been set.");
}
if (FloatEqual(m_Chs0, NODATA_VALUE)) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: Chs0 has not been set.");
}
if (FloatEqual(m_aBank, NODATA_VALUE)) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: A_bnk has not been set.");
}
if (FloatEqual(m_bBank, NODATA_VALUE)) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: B_bnk has not been set.");
}
if (FloatEqual(m_Vseep0, NODATA_VALUE)) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: m_Vseep0 has not been set.");
}
if (nullptr == m_subbasin) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: m_subbasin has not been set.");
}
if (nullptr == m_qsSub) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: Q_SBOF has not been set.");
}
if (nullptr == m_chWidth) {
throw ModelException("IKW_REACH", "CheckInputData", "The parameter: RchParam has not been set.");
}
return true;
}

void IKW_REACH:: InitialOutputs() {
if (m_nreach <= 0) {
throw ModelException("IKW_REACH", "initialOutputs", "The cell number of the input can not be less than zero.");
}

if (nullptr == m_chStorage) {
m_chStorage = new float[m_nreach + 1];
m_qIn = new float[m_nreach + 1];
m_qOut = new float[m_nreach + 1];
m_bankStorage = new float[m_nreach + 1];
m_seepage = new float[m_nreach + 1];
m_qsCh = new float[m_nreach + 1];
m_qiCh = new float[m_nreach + 1];
m_qgCh = new float[m_nreach + 1];
m_chWTdepth = new float[m_nreach + 1];

#pragma omp parallel for
for (int i = 1; i <= m_nreach; i++) {
float qiSub = 0.f;
float qgSub = 0.f;
if (nullptr != m_qiSub) {
qiSub = m_qiSub[i];
}
if (nullptr != m_qgSub) {
qgSub = m_qgSub[i];
}
m_seepage[i] = 0.f;
m_bankStorage[i] = m_Bnk0 * m_chLen[i];
m_chStorage[i] = m_Chs0 * m_chLen[i];
m_qIn[i] = 0.f;
m_qOut[i] = m_qsSub[i] + qiSub + qgSub;
m_qsCh[i] = m_qsSub[i];
m_qiCh[i] = qiSub;
m_qgCh[i] = qgSub;
m_chWTdepth[i] = 0.f;

}
}
}

int IKW_REACH::Execute() {
InitialOutputs();

for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
int nReaches = it->second.size();
#pragma omp parallel for
for (int i = 0; i < nReaches; ++i) {
int reachIndex = it->second[i]; 
ChannelFlow(reachIndex);
}
}

return 0;
}

bool IKW_REACH::CheckInputSize(const char *key, int n) {
if (n <= 0) {
return false;
}
#ifdef STORM_MODE
if(m_nreach != n-1)
{
if(m_nreach <=0)
m_nreach = n-1;
else
{
ostringstream oss;
oss << "Input data for "+string(key) << " is invalid with size: " << n << ". The origin size is " << m_nreach << ".\n";
throw ModelException("IKW_REACH","CheckInputSize",oss.str());
}
}
#else
if (m_nreach != n - 1) {
if (m_nreach <= 0) {
m_nreach = n - 1;
} else {
std::ostringstream oss;
oss << "Input data for " + string(key) << " is invalid with size: "
<< n << ". The origin size is " << m_nreach << ".\n";
throw ModelException("IKW_REACH", "CheckInputSize", oss.str());
}
}
#endif 
return true;
}

void IKW_REACH::SetValue(const char *key, float value) {
string sk(key);

if (StringMatch(sk, Tag_ChannelTimeStep[0])) {
m_dt = int(value);
} else if (StringMatch(sk, VAR_EP_CH[0])) {
m_Epch = value;
} else if (StringMatch(sk, VAR_BNK0[0])) {
m_Bnk0 = value;
} else if (StringMatch(sk, VAR_CHS0[0])) {
m_Chs0 = value;
} else if (StringMatch(sk, VAR_VSEEP0[0])) {
m_Vseep0 = value;
} else if (StringMatch(sk, VAR_A_BNK[0])) {
m_aBank = value;
} else if (StringMatch(sk, VAR_B_BNK[0])) {
m_bBank = value;
} else if (StringMatch(sk, VAR_MSK_X[0])) {
m_x = value;
} else if (StringMatch(sk, VAR_MSK_CO1[0])) {
m_co1 = value;
} else {
throw ModelException("IKW_REACH", "SetSingleData",
"Parameter " + sk + " does not exist. Please contact the module developer.");
}

}

void IKW_REACH::Set1DData(const char *key, int n, float *value) {
string sk(key);
if (StringMatch(sk, VAR_SUBBSN[0])) {
m_subbasin = value;   
} else if (StringMatch(sk, VAR_SBOF[0])) {
CheckInputSize(key, n);
m_qsSub = value;
} else if (StringMatch(sk, VAR_SBIF[0])) {
CheckInputSize(key, n);
m_qiSub = value;
} else if (StringMatch(sk, VAR_SBQG[0])) {
m_qgSub = value;
} else if (StringMatch(sk, VAR_SBPET[0])) {
m_petCh = value;
} else if (StringMatch(sk, VAR_SBGS[0])) {
m_gwStorage = value;
} else {
throw ModelException("IKW_REACH", "Set1DData", "Parameter " + sk
+ " does not exist. Please contact the module developer.");
}

}

float IKW_REACH::GetNewQ(float qIn, float qLast, float surplus, float alpha, float dt, float dx) {

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
fQkx = dtX * Qkx + alpha * CalPow(Qkx, beta) - C;   
dfQkx = dtX + alpha * beta * CalPow(Qkx, beta - 1);  
Qkx -= fQkx / dfQkx;                                
Qkx = Max(Qkx, MIN_FLUX);
count++;
} while (Abs(fQkx) > _epsilon && count < MAX_ITERS_KW);

if (Qkx != Qkx) {
throw ModelException("IKW_OL", "GetNewQ", "Error in iteration!");
}

return Qkx;
}

void IKW_REACH::ChannelFlow(int i) {
float st0 = m_chStorage[i];

float qiSub = 0.f;
if (nullptr != m_qiSub) {
qiSub = m_qiSub[i];
}
float qgSub = 0.f;
if (nullptr != m_qgSub) {
qgSub = m_qgSub[i];
}

float qIn = m_qsSub[i] + qiSub + qgSub;

float qsUp = 0.f;
float qiUp = 0.f;
float qgUp = 0.f;
for (size_t j = 0; j < m_reachUpStream[i].size(); ++j) {
int upReachId = m_reachUpStream[i][j];
qsUp += m_qsCh[upReachId];
qiUp += m_qiCh[upReachId];
qgUp += m_qgCh[upReachId];
}
qIn += qsUp + qiUp + qgUp;
qIn += m_qUpReach;

float bankOut = m_bankStorage[i] * (1 - CalExp(-m_aBank));

m_bankStorage[i] -= bankOut;
qIn += bankOut / m_dt;

m_chStorage[i] += qIn * m_dt;

float seepage = m_Kchb[i] / 1000.f / 3600.f * m_chWidth[i] * m_chLen[i] * m_dt;
if (qgSub < 0.001f) {
if (m_chStorage[i] > seepage) {
m_seepage[i] = seepage;
m_chStorage[i] -= seepage;
} else {
m_seepage[i] = m_chStorage[i];
m_chStorage[i] = 0.f;
m_qOut[i] = 0.f;
m_qsCh[i] = 0.f;
m_qiCh[i] = 0.f;
m_qgCh[i] = 0.f;
return;
}
} else {
m_seepage[i] = 0.f;
}

float dch = m_chStorage[i] / (m_chWidth[i] * m_chLen[i]);
float bankInLoss = 2 * m_Kbank[i] / 1000.f / 3600.f * dch * m_chLen[i] * m_dt;   
bankInLoss = 0.f;
if (m_chStorage[i] > bankInLoss) {
m_chStorage[i] -= bankInLoss;
} else {
bankInLoss = m_chStorage[i];
m_chStorage[i] = 0.f;
}
float bankOutGw = m_bankStorage[i] * (1 - CalExp(-m_bBank));
bankOutGw = 0.f;
m_bankStorage[i] = m_bankStorage[i] + bankInLoss - bankOutGw;
if (nullptr != m_gwStorage) {
m_gwStorage[i] += bankOutGw / m_area[i] * 1000.f;
}   

if (m_chStorage[i] <= 0.f) {
m_qOut[i] = 0.f;
m_qsCh[i] = 0.f;
m_qiCh[i] = 0.f;
m_qgCh[i] = 0.f;
return;
}

float et = 0.f;
if (nullptr != m_petCh) {
et = m_Epch * m_petCh[i] / 1000.0f * m_chWidth[i] * m_chLen[i];    
if (m_chStorage[i] > et) {
m_chStorage[i] -= et;
} else {
et = m_chStorage[i];
m_chStorage[i] = 0.f;
m_qOut[i] = 0.f;
m_qsCh[i] = 0.f;
m_qiCh[i] = 0.f;
m_qgCh[i] = 0.f;
return;
}
}

float totalLoss = m_seepage[i] + bankInLoss + et;

if (m_chStorage[i] >= 0.f) {
m_chStorage[i] = st0;

float h = m_chStorage[i] / m_chWidth[i] / m_chLen[i];
float Perim = 2.f * h + m_chWidth[i];
float sSin = CalSqrt(sin(m_chSlope[i]));
float alpha = CalPow(m_chManning[i] / sSin * CalPow(Perim, _2div3), 0.6f);

float lossRate = -totalLoss / m_dt / m_chWidth[i];
m_qOut[i] = GetNewQ(qIn, m_qOut[i], lossRate, alpha, m_dt, m_chLen[i]);

float hNew = (alpha * CalPow(m_qOut[i], 0.6f)) / m_chWidth[i]; 

m_chStorage[i] += (qIn - m_qOut[i]) * m_dt;

if (m_chStorage[i] < 0.f) {
m_qOut[i] = qIn;
m_chStorage[i] = 0.f;
}
} else {
m_qOut[i] = 0.f;
m_chStorage[i] = 0.f;
qIn = 0.f;
}

float qInSum = m_qsSub[i] + qiSub + qgSub + qsUp + qiUp + qgUp;
m_qsCh[i] = m_qOut[i] * (m_qsSub[i] + qsUp) / qInSum;
m_qiCh[i] = m_qOut[i] * (qiSub + qiUp) / qInSum;
m_qgCh[i] = m_qOut[i] * (qgSub + qgUp) / qInSum;

m_qIn[i] = qIn;

m_chWTdepth[i] = dch;
}
