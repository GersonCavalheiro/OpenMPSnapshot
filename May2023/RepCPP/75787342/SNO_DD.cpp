#include "SNO_DD.h"
#include "text.h"

SNO_DD::SNO_DD(void) : m_nCells(-1), m_t0(NODATA_VALUE), m_kblow(NODATA_VALUE),
m_tsnow(NODATA_VALUE), m_crain(NODATA_VALUE), m_csnow(NODATA_VALUE),
m_Pnet(NULL), m_tMean(NULL),
m_SE(NULL), m_SR(NULL),
m_SM(NULL), m_SA(NULL) {
}

SNO_DD::~SNO_DD(void) {
if (this->m_SM != NULL) Release1DArray(this->m_SM);
if (this->m_SA != NULL) Release1DArray(this->m_SA);
}

bool SNO_DD::CheckInputData(void) {
if (this->m_date <= 0) throw ModelException(M_SNO_DD[0], "CheckInputData", "You have not set the time.");
if (this->m_nCells <= 0) {
throw ModelException(M_SNO_DD[0], "CheckInputData",
"The dimension of the input data can not be less than zero.");
}
if (this->m_Pnet == NULL) {
throw ModelException(M_SNO_DD[0], "CheckInputData", "The net precipitation data can not be NULL.");
}
if (this->m_tMean == NULL) {
throw ModelException(M_SNO_DD[0], "CheckInputData", "The mean air temperature data can not be NULL.");
}
if (this->m_kblow == NODATA_VALUE) {
throw ModelException(M_SNO_SP[0], "CheckInputData",
"The fraction coefficient of precipitation as snow can not be NODATA");
}
if (this->m_csnow == NODATA_VALUE) {
throw ModelException(M_SNO_DD[0], "CheckInputData", "The temperature impact factor can not be NODATA.");
}
if (this->m_crain == NODATA_VALUE) {
throw ModelException(M_SNO_DD[0], "CheckInputData", "The rainfall impact factor can not be NODATA.");
}
if (this->m_t0 == NODATA_VALUE) {
throw ModelException(M_SNO_DD[0], "CheckInputData", "The Snow melt temperature can not be NODATA.");
}
if (this->m_tsnow == NODATA_VALUE) {
throw ModelException(M_SNO_DD[0], "CheckInputData", "The snow fall temperature can not be NODATA.");
}
return true;
}

void SNO_DD:: InitialOutputs() {
if (m_nCells <= 0) {
throw ModelException(M_SNO_DD[0], "CheckInputData",
"The dimension of the input data can not be less than zero.");
}
if (m_SM == NULL) Initialize1DArray(m_nCells, m_SM, 0.f);
if (m_SA == NULL) Initialize1DArray(m_nCells, m_SA, 0.f);
if (m_SR == NULL) {  
Initialize1DArray(m_nCells, m_SR, 0.f);
}
if (m_SE == NULL) { 
Initialize1DArray(m_nCells, m_SE, 0.f);
}
}

int SNO_DD::Execute() {
this->CheckInputData();

this-> InitialOutputs();

#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {

m_SA[i] = m_SA[i] + m_SR[i] - m_SE[i];

if (FloatEqual(m_SA[i], 0.f) && FloatEqual(m_Pnet[i], 0.f)) {
m_SM[i] = 0.f;
continue;
}

if (m_tMean[i] < m_tsnow) 
{
m_SM[i] = 0.f;
m_SA[i] += m_kblow * m_Pnet[i];
m_Pnet[i] *= (1.f - m_kblow);
}

else {
float dt = (m_tMean[i] - m_t0);
if (dt < 0.f) {
m_SM[i] = 0.f;  
} else {
m_SM[i] = m_csnow * dt + m_crain * m_Pnet[i] * dt;
if (m_SM[i] < 0.f) m_SM[i] = 0.f;
if (m_SM[i] > m_SA[i]) m_SM[i] = m_SA[i];
m_SA[i] -= m_SM[i];
m_Pnet[i] += m_SM[i];
if (m_Pnet[i] < 0.f) m_Pnet[i] = 0.f;
}
}
}
return 0;
}

bool SNO_DD::CheckInputSize(const char *key, int n) {
if (n <= 0) {
throw ModelException(M_SNO_DD[0], "CheckInputSize",
"Input data for " + string(key) + " is invalid. The size could not be less than zero.");
return false;
}
if (this->m_nCells != n) {
if (this->m_nCells <= 0) { this->m_nCells = n; }
else {
throw ModelException(M_SNO_DD[0], "CheckInputSize", "Input data for " + string(key) +
" is invalid. All the input data should have same size.");
return false;
}
}
return true;
}

void SNO_DD::SetValue(const char *key, float data) {
string s(key);
if (StringMatch(s, VAR_K_BLOW[0])) { this->m_kblow = data; }
else if (StringMatch(s, VAR_C_SNOW[0])) { this->m_csnow = data; }
else if (StringMatch(s, VAR_C_RAIN[0])) { this->m_crain = data; }
else if (StringMatch(s, VAR_T0[0])) { this->m_t0 = data; }
else if (StringMatch(s, VAR_T_SNOW[0])) { this->m_tsnow = data; }
else {
throw ModelException(M_SNO_DD[0], "SetValue", "Parameter " + s
+ " does not exist in current module. Please contact the module developer.");
}
}

void SNO_DD::Set1DData(const char *key, int n, float *data) {
string s(key);
this->CheckInputSize(key, n);
if (StringMatch(s, VAR_TMEAN[0])) { this->m_tMean = data; }
else if (StringMatch(s, VAR_NEPR[0])) {
this->m_Pnet = data;
} else {
throw ModelException(M_SNO_DD[0], "Set1DData", "Parameter " + s +
" does not exist in current module. Please contact the module developer.");
}
}

void SNO_DD::Get1DData(const char *key, int *n, float **data) {
InitialOutputs();
string s(key);
if (StringMatch(s, VAR_SNME[0])) { *data = this->m_SM; }
else if (StringMatch(s, VAR_SNAC[0])) { *data = this->m_SA; }
else if (StringMatch(s, VAR_NEPR[0])) { *data = this->m_Pnet; }
else {
throw ModelException(M_SNO_DD[0], "Get1DData",
"Result " + s + " does not exist in current module. Please contact the module developer.");
}
*n = this->m_nCells;
}

