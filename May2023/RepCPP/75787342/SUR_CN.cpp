#include "SUR_CN.h"
#include "text.h"

SUR_CN::SUR_CN(void) : m_nCells(-1), m_Tsnow(NODATA_VALUE), m_Tsoil(NODATA_VALUE), m_T0(NODATA_VALUE),
m_Sfrozen(NODATA_VALUE),
m_cn2(NULL), m_initSoilMoisture(NULL), m_rootDepth(NULL),
m_soilDepth(NULL), m_porosity(NULL), m_fieldCap(NULL), m_wiltingPoint(NULL),
m_P_NET(NULL), m_SD(NULL), m_tMean(NULL), m_TS(NULL), m_SM(NULL), m_SA(NULL),
m_PE(NULL), m_INFIL(NULL), m_soilMoisture(NULL),
m_w1(NULL), m_w2(NULL), m_sMax(NULL) {
}

SUR_CN::~SUR_CN(void) {
Release1DArray(m_PE);
Release1DArray(m_INFIL);
Release2DArray(m_soilMoisture);
Release1DArray(m_w1);
Release1DArray(m_w2);
Release1DArray(m_sMax);
}

bool SUR_CN::CheckInputData(void) {
if (m_date < 0) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "You have not set the time.");
return false;
}
if (m_nCells <= 0) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The cell number of the input can not be less than zero.");
return false;
}
if (FloatEqual(m_Sfrozen, NODATA_VALUE)) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The frozen soil moisture of the input data can not be NULL.");
return false;
}
if (FloatEqual(m_Tsnow, NODATA_VALUE)) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The snowfall temperature of the input data can not be NULL.");
return false;
}
if (FloatEqual(m_Tsoil, NODATA_VALUE)) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The soil freezing temperature of the input data can not be NULL.");
return false;
}
if (FloatEqual(m_T0, NODATA_VALUE)) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The snowmelt threshold temperature of the input data can not be NULL.");
return false;
}
if (m_cn2 == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The CN under moisture condition II of the input data can not be NULL.");
return false;
}
if (m_initSoilMoisture == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The initial soil moisture or soil moisture of the input data can not be NULL.");
return false;
}
if (m_rootDepth == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The root depth of the input data can not be NULL.");
return false;
}
if (m_soilDepth == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The soil depth of the input data can not be NULL.");
return false;
}
if (m_porosity == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The soil porosity of the input data can not be NULL.");
return false;
}
if (m_fieldCap == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The water content of soil at field capacity of the input data can not be NULL.");
return false;
}
if (m_wiltingPoint == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The plant wilting point moisture of the input data can not be NULL.");
return false;
}
if (m_P_NET == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The net precipitation of the input data can not be NULL.");
return false;
}
if (m_tMean == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The mean air temperature of the input data can not be NULL.");
return false;
}
if (m_TS == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The soil temperature of the input data can not be NULL.");
return false;
}
if (m_SD == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The depression storage of the input data can not be NULL.");
return false;
}
if (m_SM == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The snow melt of the input data can not be NULL.");
return false;
}
if (m_SA == NULL) {
throw ModelException(M_SUR_CN[0], "CheckInputData", "The snow accumulation of the input data can not be NULL.");
return false;
}
return true;
}

void SUR_CN:: InitialOutputs() {
if (m_nCells <= 0) {
throw ModelException(M_SUR_CN[0], "CheckInputData",
"The dimension of the input data can not be less than zero.");
}
if (m_PE == NULL) {
m_PE = new float[m_nCells];
m_INFIL = new float[m_nCells];

m_soilMoisture = new float *[m_nCells];
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
m_PE[i] = 0.0f;
m_INFIL[i] = 0.0f;

m_soilMoisture[i] = new float[m_nSoilLayers];
for (int j = 0; j < m_nSoilLayers; j++) {
m_soilMoisture[i][j] = m_initSoilMoisture[i] * m_fieldCap[i][j];
}
}
initalW1W2();
}
}

int SUR_CN::Execute() {
CheckInputData();
InitialOutputs();

float cnday;
float pNet, surfq, infil;

#pragma omp parallel for
for (int iCell = 0; iCell < m_nCells; iCell++) {
surfq = 0.0f;
infil = 0.0f;
pNet = 0.0f;
float pcp = m_P_NET[iCell];  
float dep = 0.0f;  
float snm = 0.0f;  
float t = 0.0f;  

if (m_SD == NULL)        
{
dep = 0.f;
} else {
dep = m_SD[iCell];
}

if (m_SM == NULL) {
snm = 0.f;
} else {
snm = m_SM[iCell];
}

t = m_tMean[iCell];
if (t <= m_Tsnow) {
pNet = 0.0f;
}
else if (t > m_Tsnow && t <= m_T0 && m_SA != NULL && m_SA[iCell] > pcp) {
pNet = 0.0f;
} else {
pNet = pcp + dep + snm;    
}

if (pNet > 0.0f) {
float sm = 0.f;
float por = 0.f;

int curSoilLayers = -1, j;
m_upSoilDepth[0] = m_soilDepth[iCell][0];
for (j = 1; j < m_nSoilLayers; j++) {
if (!FloatEqual(m_soilDepth[iCell][j], NODATA_VALUE)) {
m_upSoilDepth[j] = m_soilDepth[iCell][j] - m_soilDepth[iCell][j - 1];
} else {
break;
}
}
curSoilLayers = j;

for (j = 0; j < curSoilLayers; j++) {
sm += m_soilMoisture[iCell][j] * m_upSoilDepth[j];
por += m_porosity[iCell][j] * m_upSoilDepth[j];
}

sm /= por;
sm = Min(sm, 1.0f);

if (m_TS[iCell] <= m_Tsoil && sm >= m_Sfrozen * por) {
m_PE[iCell] = pcp + snm;
m_INFIL[iCell] = 0.0f;
}
else if (sm > por) {
m_PE[iCell] = pcp + snm;
m_INFIL[iCell] = 0.0f;
}
else {
cnday = Calculate_CN(sm, iCell);
float bb, pb;
float s = 0.0f;
bb = 0.0f;
pb = 0.0f;
s = 25400.0f / cnday - 254.0f;
bb = 0.2f * s;
pb = pNet - bb;
if (pb > 0.0f) {
surfq = pb * pb / (pNet + 0.8f * s);
}

if (surfq < 0.0f) {
surfq = 0.0f;
}

infil = pNet - surfq;

m_INFIL[iCell] = infil;
m_PE[iCell] = pcp + snm - infil;    
}

if (m_INFIL[iCell] < 0) {
m_INFIL[iCell] = 0.f;
m_PE[iCell] = pcp + snm - m_INFIL[iCell];
}

if (m_INFIL[iCell] != m_INFIL[iCell] || m_INFIL[iCell] < 0.0f) {
cout << m_INFIL[iCell] << endl;
throw ModelException(M_SUR_CN[0],
"Execute",
"Output data error: infiltration is less than zero. :\n Please contact the module developer. ");
}

} else {
m_PE[iCell] = 0.0f;
m_INFIL[iCell] = 0.0f;
}
}
return 0;
}

bool SUR_CN::CheckInputSize(const char *key, int n) {
if (n <= 0) {
throw ModelException(M_SUR_CN[0], "CheckInputSize",
"Input data for " + string(key) + " is invalid. The size could not be less than zero.");
return false;
}
if (m_nCells != n) {
if (m_nCells <= 0) { m_nCells = n; }
else {
throw ModelException(M_SUR_CN[0], "CheckInputSize", "Input data for " + string(key) +
" is invalid. All the input data should have same size.");
return false;
}
}
return true;
}

void SUR_CN::SetValue(const char *key, float value) {
string sk(key);
if (StringMatch(sk, VAR_T_SNOW[0])) { m_Tsnow = value; }
else if (StringMatch(sk, VAR_T_SOIL[0])) { m_Tsoil = value; }
else if (StringMatch(sk, VAR_T0[0])) { m_T0 = value; }
else if (StringMatch(sk, VAR_S_FROZEN[0])) { m_Sfrozen = value; }
else {
throw ModelException(M_SUR_CN[0], "SetValue", "Parameter " + sk
+
" does not exist in current module. Please contact the module developer.");
}
}

void SUR_CN::Set1DData(const char *key, int n, float *data) {
CheckInputSize(key, n);
string sk(key);

if (StringMatch(sk, VAR_CN2[0])) { m_cn2 = data; }
else if (StringMatch(sk, VAR_MOIST_IN[0])) { m_initSoilMoisture = data; }
else if (StringMatch(sk, VAR_ROOTDEPTH[0])) { m_rootDepth = data; }
else if (StringMatch(sk, VAR_NEPR[0])) { m_P_NET = data; }
else if (StringMatch(sk, VAR_DPST[0])) {
m_SD = data; 
} else if (StringMatch(sk, VAR_TMEAN[0])) { m_tMean = data; }
else if (StringMatch(sk, VAR_SNME[0])) { m_SM = data; }
else if (StringMatch(sk, VAR_SNAC[0])) { m_SA = data; }
else if (StringMatch(sk, VAR_SOTE[0])) { m_TS = data; }
else {
throw ModelException(M_SUR_CN[0], "Set1DData", "Parameter " + sk +
" does not exist in current module. Please contact the module developer.");
}
}

void SUR_CN::Set2DData(const char *key, int nrows, int ncols, float **data) {
string sk(key);
CheckInputSize(key, nrows);
m_nSoilLayers = ncols;
if (StringMatch(sk, VAR_SOILDEPTH[0])) { m_soilDepth = data; }
else if (StringMatch(sk, VAR_POROST[0])) { m_porosity = data; }
else if (StringMatch(sk, VAR_FIELDCAP[0])) { m_fieldCap = data; }
else if (StringMatch(sk, VAR_WILTPOINT[0])) { m_wiltingPoint = data; }
else {
throw ModelException(M_SUR_CN[0], "Set2DData", "Parameter " + sk
+ " does not exist. Please contact the module developer.");
}
}

void SUR_CN::Get1DData(const char *key, int *n, float **data) {
string sk(key);

if (StringMatch(sk, VAR_INFIL[0])) { *data = m_INFIL; }
else if (StringMatch(sk, VAR_EXCP[0])) { *data = m_PE; }
else {
throw ModelException(M_SUR_CN[0], "Get1DData",
"Result " + sk +
" does not exist in current module. Please contact the module developer.");
}
*n = m_nCells;
}

void SUR_CN::Get2DData(const char *key, int *nRows, int *nCols, float ***data) {
string sk(key);
*nRows = m_nCells;
*nCols = m_nSoilLayers;
if (StringMatch(sk, VAR_SOL_ST[0])) { *data = m_soilMoisture; }
else {
throw ModelException(M_SUR_CN[0], "Get2DData", "Output " + sk
+ " does not exist. Please contact the module developer.");
}
}

float SUR_CN::Calculate_CN(float sm, int cell) {
float sw, s, CNday, xx;

s = 0.;
sw = sm * m_rootDepth[cell];
xx = m_w1[cell] - m_w2[cell] * sw;
if (xx < -20.f) {
xx = -20.f;
}
if (xx > 20.f) {
xx = 20.f;
}
if ((sw + CalExp(xx)) > 0.001f) {
s = m_sMax[cell] * (1.f - sw / (sw + CalExp(xx)));  
}
CNday = 25400.f / (s + 254.f);  
return CNday;
}

void SUR_CN::initalW1W2() {
m_w1 = new float[m_nCells];
m_w2 = new float[m_nCells];
m_sMax = new float[m_nCells];
if (m_upSoilDepth == NULL) {
m_upSoilDepth = new float[m_nSoilLayers];
}

for (int i = 0; i < m_nCells; i++) {
float fieldcap = 0.f;
float wsat = 0.f;
int curSoilLayers = -1, j;
m_upSoilDepth[0] = m_soilDepth[i][0];
for (j = 1; j < m_nSoilLayers; j++) {
if (!FloatEqual(m_soilDepth[i][j], NODATA_VALUE)) {
m_upSoilDepth[j] = m_soilDepth[i][j] - m_soilDepth[i][j - 1];
} else {
break;
}
}
curSoilLayers = j;
for (j = 0; j < curSoilLayers; j++) {
fieldcap += m_fieldCap[i][j] * m_upSoilDepth[j];
wsat += m_porosity[i][j] * m_upSoilDepth[j];
}


float cnn = m_cn2[i];
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

m_w1[i] = wrt1;
m_w2[i] = wrt2;
m_sMax[i] = smx;
}
}




