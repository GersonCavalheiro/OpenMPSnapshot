#include "SUR_GreenAmpt.h"
#include "text.h"

SUR_GreenAmpt::SUR_GreenAmpt(void) : m_TimeStep(NODATA_VALUE), m_Conductivity(NULL), m_porosity(NULL), m_clay(NULL), m_sand(NULL),
m_rootDepth(NULL),
m_cn2(NULL), m_P_NET(NULL), m_cellSize(-1), m_fieldCap(NULL), m_wiltingPoint(NULL),
m_soilMoisture(NULL), m_INFRate(NULL),
m_SD(NULL), m_tMin(NULL), m_tMax(NULL), m_Tsnow(NODATA_VALUE), m_Tsoil(NODATA_VALUE),
m_Sfrozen(NODATA_VALUE),
m_T0(NODATA_VALUE), m_SM(NULL),
m_SA(NULL), m_TS(NULL), m_mask(NULL), m_INFIL(NULL), m_PE(NULL), m_date(-1), m_w1(NULL),
m_w2(NULL), m_sMax(NULL), m_wfmp(NULL) {
}

SUR_GreenAmpt::~SUR_GreenAmpt(void) {
Release1DArray(m_PE);
Release1DArray(m_INFIL);
Release1DArray(m_w1);
Release1DArray(m_w2);
Release1DArray(m_sMax);
Release1DArray(m_wfmp);
Release1DArray(m_INFRate);
}

bool SUR_GreenAmpt::CheckInputData(void) {
if (this->m_date < 0) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "You have not set the time.");
return false;
}

if (this->m_cellSize <= 0) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The cell number of the input can not be less than zero.");
return false;
}

if (this->m_Conductivity == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The saturated hydraulic conductivity of the input data can not be NULL.");
return false;
}

if (this->m_porosity == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The soil porosity of the input data can not be NULL.");
return false;
}

if (this->m_clay == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The percent of clay content of the input data can not be NULL.");
return false;
}

if (this->m_sand == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The percent of sand content of the input data can not be NULL.");
return false;
}

if (this->m_rootDepth == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The root depth of the input data can not be NULL.");
return false;
}

if (this->m_cn2 == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The CN under moisture condition II of the input data can not be NULL.");
return false;
}

if (this->m_P_NET == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The net precipitation of the input data can not be NULL.");
return false;
}

if (this->m_fieldCap == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The water content of soil at field capacity of the input data can not be NULL.");
return false;
}

if (this->m_wiltingPoint == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The plant wilting point moisture of the input data can not be NULL.");
return false;
}

if (this->m_soilMoisture == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The soil moisture of the input data can not be NULL.");
return false;
}


if (FloatEqual(this->m_Sfrozen, NODATA_VALUE)) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The frozen soil moisture of the input data can not be NULL.");
return false;
}

if (this->m_SD == NULL) {
throw ModelException("SUR_GreenAmpt",
"CheckInputData",
"The depression storage or the depression storage capacity and depression storage coefficient of the input data can not be NULL.");
return false;
}

if (this->m_tMax == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The maximum temperature of the input data can not be NULL.");
return false;
}

if (this->m_tMin == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The minimum temperature of the input data can not be NULL.");
return false;
}

if (FloatEqual(this->m_Tsnow, NODATA_VALUE)) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The snowfall temperature of the input data can not be NULL.");
return false;
}

if (FloatEqual(this->m_Tsoil, NODATA_VALUE)) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The soil freezing temperature of the input data can not be NULL.");
return false;
}

if (FloatEqual(this->m_T0, NODATA_VALUE)) {
throw ModelException("SUR_GreenAmpt", "CheckInputData",
"The snowmelt threshold temperature of the input data can not be NULL.");
return false;
}

if (this->m_SM == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The snowmelt of the input data can not be NULL.");
return false;
}

if (this->m_SA == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The snow accumulation of the input data can not be NULL.");
return false;
}

if (this->m_TS == NULL) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The soil temperature of the input data can not be NULL.");
return false;
}

if (this->m_mask == NULL) {
throw ModelException("SUR_CN", "CheckInputData", "The mask of the input data can not be NULL.");
return false;
}

return true;
}

void SUR_GreenAmpt::InitalOutputs() {
if (m_cellSize <= 0) {
throw ModelException("SUR_GreenAmpt", "CheckInputData", "The dimension of the input data can not be less than zero.");
}
if (this->m_PE == NULL) {
this->m_PE = new float[this->m_cellSize];
}
if (this->m_INFIL == NULL) {
this->m_INFIL = new float[this->m_cellSize];
}
for (int i = 0; i < m_cellSize; i++) {
m_PE[i] = 0.0f;
m_INFIL[i] = 0.0f;
}

if (this->m_w1 == NULL) initalW1W2();
if (this->m_wfmp == NULL) this->initialWFMP();
if (this->m_INFRate == NULL) {
this->m_INFRate = new float[this->m_cellSize];
for (int iCell = 0; iCell < m_cellSize; iCell++) {
this->m_INFRate[iCell] = 0.0f;
}
}
}

int SUR_GreenAmpt::Execute() {
this->CheckInputData();

this->InitalOutputs();

float sol_por, dthet, SW;
float adj_hc, sol_k, cnday, psidt, tst, f1, pNet;
float rateinf0, rateinf, cuminf, rintns, wfmp;

#pragma omp parallel for
for (int iCell = 0; iCell < m_cellSize; iCell++) {
rateinf = 0.0f;
cuminf = 0.0f;
pNet = 0.0f;
float pcp = 0.0f;  
float dep = 0.0f;  
float sna = 0.0f;   
float snm = 0.0f;  
float Tsoil = 0.0f;  
float t = 0.0f;  
float sm_frozen = 0.0f; 
float soilmoist = 0.0f; 
pcp = m_P_NET[iCell];
dep = m_SD[iCell];
sna = m_SA[iCell];
snm = m_SM[iCell];
t = (m_tMin[iCell] + m_tMax[iCell]) / 2;
Tsoil = m_TS[iCell];
sm_frozen = m_Sfrozen * m_porosity[iCell];
soilmoist = m_soilMoisture[iCell];
if (t <= m_Tsnow) {
pNet = 0.0f;
}
else if (t > m_Tsnow && t <= m_T0 && m_SA[iCell] > pNet) {
pNet = 0.0f;
} else {
pNet = pcp + dep + snm;    
}

if (pNet > 0.0f) {
if (Tsoil <= m_Tsoil && soilmoist >= sm_frozen) {
m_PE[iCell] = pNet;
m_INFIL[iCell] = 0.0f;
}
else {
cnday = 0.0f;
cnday = Calculate_CN(iCell);
sol_k = m_Conductivity[iCell];
adj_hc = (56.82f * CalPow(sol_k, 0.286f)) / (1.f + 0.051f * CalExp(0.062f * cnday)) - 2.0f;
if (adj_hc <= 0.0) {
adj_hc = 0.001f;
}
dthet = 0.0f;
float soilw = 0.0f;
rateinf0 = m_INFRate[iCell];

SW = m_soilMoisture[iCell] * m_rootDepth[iCell];
float fieldcap = m_fieldCap[iCell] * m_rootDepth[iCell];
if (SW >= fieldcap) {
soilw = 0.999f * fieldcap;
} else {
soilw = SW;
}
sol_por = m_porosity[iCell];
dthet = (1.0f - soilw / fieldcap) * sol_por * 0.95f;

psidt = 0.0;
wfmp = this->m_wfmp[iCell];
psidt = dthet * wfmp;

rintns = 60.0f * pNet / m_TimeStep;
cuminf = 0.0f;
if (rateinf0 >= rintns) {
cuminf = rintns * m_TimeStep / 60.0f;
} else {
tst = 0.0f;
tst = adj_hc * m_TimeStep / 60.0f;
do {
f1 = 0.0f;
f1 = cuminf + adj_hc * m_TimeStep / 60.0f + psidt * CalLn((tst + psidt) / (cuminf + psidt));
if (abs(f1 - tst) <= 0.001) {
cuminf = f1;
break;
} else {
tst = 0.0f;
tst = f1;
}
} while (1);
}
if (cuminf != 0) {
rateinf = adj_hc * (psidt / (cuminf + 0.000001f) + 1.0f);
}
m_INFRate[iCell] = rateinf;   
m_INFIL[iCell] = cuminf;
m_PE[iCell] = pcp + snm - cuminf;
}
} else {
m_PE[iCell] = 0.0f;
m_INFIL[iCell] = 0.0f;
}
if (m_INFIL[iCell] > 10000.0f || m_INFIL[iCell] < 0.0f) {
std::ostringstream oss;
oss << "Cell(Row:" << m_mask[iCell][0] << ", Col:" << m_mask[iCell][1] << "\n Infiltration =" <<
m_INFIL[iCell]
<< "\n Precipitation(mm) = " << pNet << "\n InfiltrationRate(m/h) = " << rateinf << "\n";
throw ModelException("SUR_GreenAmpt", "Execute",
"Output data error: infiltration is out of reasonable range. Where:\n"
+ oss.str() + "Please contact the module developer. ");
}
}
return 0;
}

bool SUR_GreenAmpt::CheckInputSize(const char *key, int n) {
if (n <= 0) {
throw ModelException("SUR_GreenAmpt", "CheckInputSize",
"Input data for " + string(key) + " is invalid. The size could not be less than zero.");
return false;
}
if (this->m_cellSize != n) {
if (this->m_cellSize <= 0) { this->m_cellSize = n; }
else {
throw ModelException("SUR_GreenAmpt", "CheckInputSize", "Input data for " + string(key) +
" is invalid. All the input data should have same size.");
return false;
}
}

return true;
}

void SUR_GreenAmpt::SetValue(const char *key, float value) {
string sk(key);

if (StringMatch(sk, "ThreadNum")) {
SetOpenMPThread((int) value);
} else if (StringMatch(sk, "TimeStep")) {
m_TimeStep = value * 60; 
} else if (StringMatch(sk, "T_snow")) {
m_Tsnow = value;
} else if (StringMatch(sk, "t_soil")) {
m_Tsoil = value;
} else if (StringMatch(sk, "T0")) {
m_T0 = value;
} else if (StringMatch(sk, "s_frozen")) {
this->m_Sfrozen = value;
} else {
throw ModelException("SUR_GreenAmpt", "SetValue", "Parameter " + sk
+
" does not exist in SUR_GreenAmpt method. Please contact the module developer.");
}

}

void SUR_GreenAmpt::Set1DData(const char *key, int n, float *data) {

this->CheckInputSize(key, n);

string sk(key);

if (StringMatch(sk, "Conductivity")) {
m_Conductivity = data;
} else if (StringMatch(sk, "porosity")) {
m_porosity = data;
} else if (StringMatch(sk, "clay")) {
m_clay = data;
} else if (StringMatch(sk, "sand")) {
m_sand = data;
} else if (StringMatch(sk, "rootDepth")) {
m_rootDepth = data;
} else if (StringMatch(sk, "cn2")) {
m_cn2 = data;
} else if (StringMatch(sk, "D_NEPR")) {
m_P_NET = data;
} else if (StringMatch(sk, "fieldCap")) {
m_fieldCap = data;
} else if (StringMatch(sk, "wiltingPoint")) {
m_wiltingPoint = data;
} else if (StringMatch(sk, "D_SOMO")) {
m_soilMoisture = data;
}


else if (StringMatch(sk, "D_DPST")) {
m_SD = data;
} else if (StringMatch(sk, "D_TMin")) {
m_tMin = data;
} else if (StringMatch(sk, "D_TMax")) {
m_tMax = data;
} else if (StringMatch(sk, "D_SNME")) {
m_SM = data;
} else if (StringMatch(sk, "D_SNAC")) {
m_SA = data;
} else if (StringMatch(sk, "D_SOTE")) {
m_TS = data;
} else {
throw ModelException("SUR_GreenAmpt", "SetValue", "Parameter " + sk +
" does not exist in SUR_GreenAmpt method. Please contact the module developer.");
}

}

void SUR_GreenAmpt::Get1DData(const char *key, int *n, float **data) {
string sk(key);
if (StringMatch(sk, "INFIL")) {
*data = m_INFIL;
} else if (StringMatch(sk, "EXCP")) {
*data = m_PE;
} else {
throw ModelException("SUR_GreenAmpt", "getResult", "Result " + sk +
" does not exist in SUR_GreenAmpt method. Please contact the module developer.");
}

*n = this->m_cellSize;
}

float SUR_GreenAmpt::Calculate_CN(int cell) {
float sw, s, CNday, xx;

s = 0.;
sw = this->m_soilMoisture[cell] * this->m_rootDepth[cell];
xx = this->m_w1[cell] - this->m_w2[cell] * sw;
if (xx < -20.f) {
xx = -20.;
}
if (xx > 20.f) {
xx = 20.f;
}
if ((sw + CalExp(xx)) > 0.001f) {
s = this->m_sMax[cell] * (1.f - sw / (sw + CalExp(xx)));  
}

CNday = 25400.f / (s + 254.f);  
return CNday;
}

void SUR_GreenAmpt::initalW1W2() {
this->m_w1 = new float[this->m_cellSize];
this->m_w2 = new float[this->m_cellSize];
this->m_sMax = new float[this->m_cellSize];

for (int i = 0; i < this->m_cellSize; i++) {
float cnn = this->m_cn2[i];
float fieldcap = this->m_fieldCap[i] * this->m_rootDepth[i];
float wsat = this->m_porosity[i] * this->m_rootDepth[i];
float c1, c3, c2, smx, s3, rto3, rtos, xx, wrt1, wrt2;
c2 = 100.0f - cnn;
c1 = cnn - 20.f * c2 / (c2 + CalExp(2.533f - 0.0636f * c2));    
c1 = Max(c1, 0.4f * cnn);
c3 = cnn * CalExp(0.006729f * c2);                                

smx = 254.f * (100.f / c1 - 1.f);                            

s3 = 254.f * (100.f / c3 - 1.f);

rto3 = 1.f - s3 / smx;
rtos = 1.f - 2.54f / smx;

xx = CalLn(fieldcap / rto3 - fieldcap);
wrt2 = (xx - CalLn(wsat / rtos - wsat)) /
(wsat - fieldcap);    
wrt1 = xx + (fieldcap * wrt2); 

this->m_w1[i] = wrt1;
this->m_w2[i] = wrt2;
this->m_sMax[i] = smx;
}
}

void SUR_GreenAmpt::initialWFMP() {
m_wfmp = new float[m_cellSize];
for (int i = 0; i < this->m_cellSize; i++) {
float sol_por = m_porosity[i];
float sol_clay = m_clay[i];
float sand = m_sand[i];
float wfmp = 0.0f;
wfmp = Calculate_WFMP(sol_por, sol_clay, sand);
m_wfmp[i] = wfmp;
}
}

float SUR_GreenAmpt::Calculate_WFMP(float sol_por, float sol_clay, float sand) {
float wfmp = 10.0f * CalExp(6.5309f - 7.32561f * sol_por + 3.809479f * CalPow(sol_por, 2) + 0.001583f *
CalPow(sol_clay, 2) +
0.000344f * sand * sol_clay - 0.049837f * sol_por * sand + 0.001608f *
CalPow(sol_por, 2) * CalPow(sand, 2) +
0.001602f * CalPow(sol_por, 2) * CalPow(sol_clay, 2) -
0.0000136f * CalPow(sand, 2) * sol_clay - 0.003479f * CalPow(sol_clay, 2) *
sol_por - 0.000799f * CalPow(sand, 2) * sol_por);

return wfmp;
}
