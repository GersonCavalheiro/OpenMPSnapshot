#include "NutrCH_QUAL2E.h"

#include "text.h"

NutrCH_QUAL2E::NutrCH_QUAL2E() :
m_inputSubbsnID(-1), m_nCells(-1), m_dt(-1), m_reachDownStream(nullptr), m_nReaches(-1),
m_ai0(-1.), m_ai1(-1.),
m_ai2(-1.), m_ai3(-1.), m_ai4(-1.), m_ai5(-1.), m_ai6(-1.), m_lambda0(-1.),
m_lambda1(-1.), m_lambda2(-1.), m_k_l(-1.), m_k_n(-1.), m_k_p(-1.), m_p_n(-1.),
tfact(-1.), m_rnum1(0.), igropt(-1), m_mumax(-1.), m_rhoq(-1.), m_cod_n(-1), m_cod_k(-1),
m_rchID(nullptr), m_soilTemp(nullptr), m_dayLen(nullptr), m_sr(nullptr),
m_qRchOut(nullptr), m_chStorage(nullptr), m_rteWtrIn(nullptr), m_rteWtrOut(nullptr), m_chWtrDepth(nullptr),
m_chTemp(nullptr), m_bc1(nullptr), m_bc2(nullptr), m_bc3(nullptr),
m_bc4(nullptr), m_rs1(nullptr), m_rs2(nullptr), m_rs3(nullptr), m_rs4(nullptr),
m_rs5(nullptr), m_rk1(nullptr), m_rk2(nullptr), m_rk3(nullptr),
m_rk4(nullptr), m_chOrgNCo(NODATA_VALUE), m_chOrgPCo(NODATA_VALUE), m_latNO3ToCh(nullptr), m_surfRfNO3ToCh(nullptr),
m_surfRfNH4ToCh(nullptr), m_surfRfSolPToCh(nullptr), m_surfRfCodToCh(nullptr), m_gwNO3ToCh(nullptr),
m_gwSolPToCh(nullptr), m_surfRfSedOrgNToCh(nullptr), m_surfRfSedOrgPToCh(nullptr),
m_surfRfSedAbsorbMinPToCh(nullptr), m_surfRfSedSorbMinPToCh(nullptr), m_no2ToCh(nullptr),
m_ptNO3ToCh(nullptr), m_ptNH4ToCh(nullptr), m_ptOrgNToCh(nullptr),
m_ptTNToCh(nullptr), m_ptSolPToCh(nullptr), m_ptOrgPToCh(nullptr), m_ptTPToCh(nullptr),
m_ptCODToCh(nullptr), m_rchDeg(nullptr), m_chAlgae(nullptr), m_chOrgN(nullptr),
m_chNH4(nullptr),
m_chNO2(nullptr), m_chNO3(nullptr), m_chTN(nullptr), m_chOrgP(nullptr), m_chSolP(nullptr), m_chTP(nullptr),
m_chCOD(nullptr), m_chDOx(nullptr), m_chChlora(nullptr), m_chSatDOx(NODATA_VALUE), m_chOutAlgae(nullptr),
m_chOutAlgaeConc(nullptr),
m_chOutChlora(nullptr),
m_chOutChloraConc(nullptr), m_chOutOrgN(nullptr), m_chOutOrgNConc(nullptr), m_chOutOrgP(nullptr),
m_chOutOrgPConc(nullptr),
m_chOutNH4(nullptr), m_chOutNH4Conc(nullptr), m_chOutNO2(nullptr), m_chOutNO2Conc(nullptr), m_chOutNO3(nullptr),
m_chOutNO3Conc(nullptr), m_chOutSolP(nullptr),
m_chOutSolPConc(nullptr), m_chOutCOD(nullptr), m_chOutCODConc(nullptr), m_chOutDOx(nullptr),
m_chOutDOxConc(nullptr), m_chOutTN(nullptr), m_chOutTNConc(nullptr),
m_chOutTP(nullptr), m_chOutTPConc(nullptr),
m_chDaylen(nullptr), m_chSr(nullptr), m_chCellCount(nullptr) {
}

NutrCH_QUAL2E::~NutrCH_QUAL2E() {
if (nullptr != m_ptNO3ToCh) Release1DArray(m_ptNO3ToCh);
if (nullptr != m_ptNH4ToCh) Release1DArray(m_ptNH4ToCh);
if (nullptr != m_ptOrgNToCh) Release1DArray(m_ptOrgNToCh);
if (nullptr != m_ptTNToCh) Release1DArray(m_ptTNToCh);
if (nullptr != m_ptSolPToCh) Release1DArray(m_ptSolPToCh);
if (nullptr != m_ptOrgPToCh) Release1DArray(m_ptOrgPToCh);
if (nullptr != m_ptTPToCh) Release1DArray(m_ptTPToCh);
if (nullptr != m_ptCODToCh) Release1DArray(m_ptCODToCh);
if (nullptr != m_chTN) Release1DArray(m_chTN);
if (nullptr != m_chTP) Release1DArray(m_chTP);
if (nullptr != m_chChlora) Release1DArray(m_chChlora);
if (nullptr != m_chOutChlora) Release1DArray(m_chOutChlora);
if (nullptr != m_chOutAlgae) Release1DArray(m_chOutAlgae);
if (nullptr != m_chOutOrgN) Release1DArray(m_chOutOrgN);
if (nullptr != m_chOutOrgP) Release1DArray(m_chOutOrgP);
if (nullptr != m_chOutNH4) Release1DArray(m_chOutNH4);
if (nullptr != m_chOutNO2) Release1DArray(m_chOutNO2);
if (nullptr != m_chOutNO3) Release1DArray(m_chOutNO3);
if (nullptr != m_chOutSolP) Release1DArray(m_chOutSolP);
if (nullptr != m_chOutDOx) Release1DArray(m_chOutDOx);
if (nullptr != m_chOutCOD) Release1DArray(m_chOutCOD);
if (nullptr != m_chOutTN) Release1DArray(m_chOutTN);
if (nullptr != m_chOutTP) Release1DArray(m_chOutTP);
if (nullptr != m_chOutChloraConc) Release1DArray(m_chOutChloraConc);
if (nullptr != m_chOutAlgaeConc) Release1DArray(m_chOutAlgaeConc);
if (nullptr != m_chOutOrgNConc) Release1DArray(m_chOutOrgNConc);
if (nullptr != m_chOutOrgPConc) Release1DArray(m_chOutOrgPConc);
if (nullptr != m_chOutNH4Conc) Release1DArray(m_chOutNH4Conc);
if (nullptr != m_chOutNO2Conc) Release1DArray(m_chOutNO2Conc);
if (nullptr != m_chOutNO3Conc) Release1DArray(m_chOutNO3Conc);
if (nullptr != m_chOutSolPConc) Release1DArray(m_chOutSolPConc);
if (nullptr != m_chOutDOxConc) Release1DArray(m_chOutDOxConc);
if (nullptr != m_chOutCODConc) Release1DArray(m_chOutCODConc);
if (nullptr != m_chOutTNConc) Release1DArray(m_chOutTNConc);
if (nullptr != m_chOutTPConc) Release1DArray(m_chOutTPConc);
if (nullptr != m_chCellCount) Release1DArray(m_chCellCount);
if (nullptr != m_chDaylen) Release1DArray(m_chDaylen);
if (nullptr != m_chSr) Release1DArray(m_chSr);
if (nullptr != m_chTemp) Release1DArray(m_chTemp);
}

void NutrCH_QUAL2E::ParametersSubbasinForChannel() {
if (nullptr == m_chCellCount) {
Initialize1DArray(m_nReaches + 1, m_chCellCount, 0);
}
if (nullptr == m_chDaylen) {
Initialize1DArray(m_nReaches + 1, m_chDaylen, 0.);
Initialize1DArray(m_nReaches + 1, m_chSr, 0.);
Initialize1DArray(m_nReaches + 1, m_chTemp, 0.);
} else {
return; 
}
#pragma omp parallel
{
FLTPT* tmp_chDaylen = new(nothrow) FLTPT[m_nReaches + 1];
FLTPT* tmp_chSr = new(nothrow) FLTPT[m_nReaches + 1];
FLTPT* tmp_chTemp = new(nothrow) FLTPT[m_nReaches + 1];
int* tmp_chCellCount = new(nothrow) int[m_nReaches + 1];
for (int irch = 0; irch <= m_nReaches; irch++) {
tmp_chDaylen[irch] = 0.;
tmp_chSr[irch] = 0.;
tmp_chTemp[irch] = 0.;
tmp_chCellCount[irch] = 0;
}
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
if (m_rchID[i] <= 0) {
continue;
}
int irch = m_rchID[i];
if (irch >= m_nReaches + 1) {
throw ModelException(M_NUTRCH_QUAL2E[0], "Execute",
"The subbasin " + ValueToString(irch) + " is invalid.");
}
tmp_chDaylen[irch] += m_dayLen[i];
tmp_chSr[irch] += m_sr[i];
tmp_chTemp[irch] += m_soilTemp[i];
tmp_chCellCount[irch] += 1;
}
#pragma omp critical
{
for (int irch = 0; irch <= m_nReaches; irch++) {
m_chDaylen[irch] += tmp_chDaylen[irch];
m_chSr[irch] += tmp_chSr[irch];
m_chTemp[irch] += tmp_chTemp[irch];
m_chCellCount[irch] += tmp_chCellCount[irch];
}
}
delete[] tmp_chDaylen;
delete[] tmp_chSr;
delete[] tmp_chTemp;
delete[] tmp_chCellCount;
}

for (int irch = 1; irch <= m_nReaches; irch++) {
m_chDaylen[irch] /= m_chCellCount[irch];
m_chSr[irch] /= m_chCellCount[irch];
m_chTemp[irch] /= m_chCellCount[irch];

m_chDaylen[0] += m_chDaylen[irch];
m_chSr[0] += m_chSr[irch];
m_chTemp[0] += m_chTemp[irch];
}
m_chDaylen[0] /= m_nReaches;
m_chSr[0] /= m_nReaches;
m_chTemp[0] /= m_nReaches;
}

bool NutrCH_QUAL2E::CheckInputCellSize(const char* key, const int n) {
if (n <= 0) {
throw ModelException(M_NUTRCH_QUAL2E[0], "CheckInputSize",
"Input data for " + string(key) +
" is invalid. The size could not be less than zero.");
}
if (m_nCells != n) {
if (m_nCells <= 0) {
m_nCells = n;
} else {
throw ModelException(M_NUTRCH_QUAL2E[0], "CheckInputCellSize",
"Input data for " + string(key) +
" is invalid with size: " + ValueToString(n) + ". The origin size is " +
ValueToString(m_nCells));
}
}
return true;
}

bool NutrCH_QUAL2E::CheckInputData() {
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_dt);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_nReaches);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_rnum1);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], igropt);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_cod_n);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_cod_k);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai0);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai1);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai2);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai3);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai4);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai5);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_ai6);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_lambda0);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_lambda1);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_lambda2);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_k_l);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_k_n);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_k_p);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_p_n);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], tfact);
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_mumax);

CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_dayLen);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_sr);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_qRchOut);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_chStorage);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_rteWtrIn);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_rteWtrOut);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_chWtrDepth);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_latNO3ToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfNO3ToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfSolPToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfCodToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_gwNO3ToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_gwSolPToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfSedOrgNToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfSedOrgPToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfSedAbsorbMinPToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_surfRfSedSorbMinPToCh);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_rchID);
CHECK_POINTER(M_NUTRCH_QUAL2E[0], m_soilTemp);
return true;
}

void NutrCH_QUAL2E::SetValue(const char* key, const FLTPT value) {
string sk(key);
if (StringMatch(sk, VAR_RNUM1[0])) m_rnum1 = value;
else if (StringMatch(sk, VAR_COD_N[0])) m_cod_n = value;
else if (StringMatch(sk, VAR_COD_K[0])) m_cod_k = value;
else if (StringMatch(sk, VAR_AI0[0])) m_ai0 = value;
else if (StringMatch(sk, VAR_AI1[0])) m_ai1 = value;
else if (StringMatch(sk, VAR_AI2[0])) m_ai2 = value;
else if (StringMatch(sk, VAR_AI3[0])) m_ai3 = value;
else if (StringMatch(sk, VAR_AI4[0])) m_ai4 = value;
else if (StringMatch(sk, VAR_AI5[0])) m_ai5 = value;
else if (StringMatch(sk, VAR_AI6[0])) m_ai6 = value;
else if (StringMatch(sk, VAR_LAMBDA0[0])) m_lambda0 = value;
else if (StringMatch(sk, VAR_LAMBDA1[0])) m_lambda1 = value;
else if (StringMatch(sk, VAR_LAMBDA2[0])) m_lambda2 = value;
else if (StringMatch(sk, VAR_K_L[0])) {
m_k_l = value * 1.e-3 * 60.;
} else if (StringMatch(sk, VAR_K_N[0])) m_k_n = value;
else if (StringMatch(sk, VAR_K_P[0])) m_k_p = value;
else if (StringMatch(sk, VAR_P_N[0])) m_p_n = value;
else if (StringMatch(sk, VAR_TFACT[0])) tfact = value;
else if (StringMatch(sk, VAR_MUMAX[0])) m_mumax = value;
else if (StringMatch(sk, VAR_RHOQ[0])) m_rhoq = value;
else if (StringMatch(sk, VAR_CH_ONCO[0])) m_chOrgNCo = value;
else if (StringMatch(sk, VAR_CH_OPCO[0])) m_chOrgPCo = value;
else {
throw ModelException(M_NUTRCH_QUAL2E[0], "SetValue",
"Parameter " + sk + " does not exist.");
}
}

void NutrCH_QUAL2E::SetValue(const char* key, const int value) {
string sk(key);
if (StringMatch(sk, Tag_SubbasinId)) m_inputSubbsnID = value;
else if (StringMatch(sk, Tag_ChannelTimeStep[0])) m_dt = value;
else if (StringMatch(sk, VAR_IGROPT[0])) igropt = value;
else {
throw ModelException(M_NUTRCH_QUAL2E[0], "SetValue",
"Integer Parameter " + sk + " does not exist.");
}
}

void NutrCH_QUAL2E::SetValueByIndex(const char* key, const int index, const FLTPT data) {
if (m_inputSubbsnID == 0) return;             
if (index <= 0 || index > m_nReaches) return; 
if (nullptr == m_chOutAlgae) InitialOutputs();
string sk(key);
if (StringMatch(sk, VAR_CH_ALGAE[0])) m_chOutAlgae[index] = data;
else if (StringMatch(sk, VAR_CH_ALGAEConc[0])) m_chOutAlgaeConc[index] = data;
else if (StringMatch(sk, VAR_CH_NO2[0])) m_chOutNO2[index] = data;
else if (StringMatch(sk, VAR_CH_NO2Conc[0])) m_chOutNO2Conc[index] = data;
else if (StringMatch(sk, VAR_CH_COD[0])) m_chOutCOD[index] = data;
else if (StringMatch(sk, VAR_CH_CODConc[0])) m_chOutCODConc[index] = data;
else if (StringMatch(sk, VAR_CH_CHLORA[0])) m_chOutChlora[index] = data;
else if (StringMatch(sk, VAR_CH_CHLORAConc[0])) m_chOutChloraConc[index] = data;
else if (StringMatch(sk, VAR_CH_NO3[0])) m_chOutNO3[index] = data;
else if (StringMatch(sk, VAR_CH_NO3Conc[0])) m_chOutNO3Conc[index] = data;
else if (StringMatch(sk, VAR_CH_SOLP[0])) m_chOutSolP[index] = data;
else if (StringMatch(sk, VAR_CH_SOLPConc[0])) m_chOutSolPConc[index] = data;
else if (StringMatch(sk, VAR_CH_ORGN[0])) m_chOutOrgN[index] = data;
else if (StringMatch(sk, VAR_CH_ORGNConc[0])) m_chOutOrgNConc[index] = data;
else if (StringMatch(sk, VAR_CH_ORGP[0])) m_chOutOrgP[index] = data;
else if (StringMatch(sk, VAR_CH_ORGPConc[0])) m_chOutOrgPConc[index] = data;
else if (StringMatch(sk, VAR_CH_NH4[0])) m_chOutNH4[index] = data;
else if (StringMatch(sk, VAR_CH_NH4Conc[0])) m_chOutNH4Conc[index] = data;
else if (StringMatch(sk, VAR_CH_DOX[0])) m_chOutDOx[index] = data;
else if (StringMatch(sk, VAR_CH_DOXConc[0])) m_chOutDOxConc[index] = data;
else if (StringMatch(sk, VAR_CH_TN[0])) m_chOutTN[index] = data;
else if (StringMatch(sk, VAR_CH_TNConc[0])) m_chOutTNConc[index] = data;
else if (StringMatch(sk, VAR_CH_TP[0])) m_chOutTP[index] = data;
else if (StringMatch(sk, VAR_CH_TPConc[0])) m_chOutTPConc[index] = data;
else if (StringMatch(sk, VAR_CHSTR_NO3[0])) m_chNO3[index] = data;
else if (StringMatch(sk, VAR_CHSTR_NH4[0])) m_chNH4[index] = data;
else if (StringMatch(sk, VAR_CHSTR_TN[0])) m_chTN[index] = data;
else if (StringMatch(sk, VAR_CHSTR_TP[0])) m_chTP[index] = data;
else {
throw ModelException(M_NUTRCH_QUAL2E[0], "SetValueByIndex",
"Parameter " + sk + " does not exist.");
}
}

void NutrCH_QUAL2E::Set1DData(const char* key, const int n, FLTPT* data) {
string sk(key);
if (StringMatch(sk, VAR_DAYLEN[0])) {
CheckInputCellSize(key, n);
m_dayLen = data;
return;
}
if (StringMatch(sk, DataType_SolarRadiation)) {
CheckInputCellSize(key, n);
m_sr = data;
return;
}
if (StringMatch(sk, VAR_SOTE[0])) {
CheckInputCellSize(key, n);
m_soilTemp = data;
return;
}

CheckInputSize(M_NUTRCH_QUAL2E[0], key, n - 1, m_nReaches);
if (StringMatch(sk, VAR_QRECH[0])) m_qRchOut = data;
else if (StringMatch(sk, VAR_CHST[0])) {
m_chStorage = data;
for (int i = 0; i <= m_nReaches; i++) {
FLTPT cvt_conc2amount = m_chStorage[i] * 0.001;
m_chAlgae[i] *= cvt_conc2amount;
m_chOrgN[i] *= cvt_conc2amount;
m_chOrgP[i] *= cvt_conc2amount;
m_chNH4[i] *= cvt_conc2amount;
m_chNO2[i] *= cvt_conc2amount;
m_chNO3[i] *= cvt_conc2amount;
m_chSolP[i] *= cvt_conc2amount;
m_chDOx[i] *= cvt_conc2amount;
m_chCOD[i] *= cvt_conc2amount;
}
} else if (StringMatch(sk, VAR_RTE_WTRIN[0])) m_rteWtrIn = data;
else if (StringMatch(sk, VAR_RTE_WTROUT[0])) m_rteWtrOut = data;
else if (StringMatch(sk, VAR_CHWTRDEPTH[0])) m_chWtrDepth = data;
else if (StringMatch(sk, VAR_WATTEMP[0])) m_chTemp = data;

else if (StringMatch(sk, VAR_LATNO3_TOCH[0])) m_latNO3ToCh = data;
else if (StringMatch(sk, VAR_SUR_NO3_TOCH[0])) m_surfRfNO3ToCh = data;
else if (StringMatch(sk, VAR_SUR_NH4_TOCH[0])) m_surfRfNH4ToCh = data;
else if (StringMatch(sk, VAR_SUR_SOLP_TOCH[0])) m_surfRfSolPToCh = data;
else if (StringMatch(sk, VAR_SUR_COD_TOCH[0])) m_surfRfCodToCh = data;
else if (StringMatch(sk, VAR_NO3GW_TOCH[0])) m_gwNO3ToCh = data;
else if (StringMatch(sk, VAR_MINPGW_TOCH[0])) m_gwSolPToCh = data;
else if (StringMatch(sk, VAR_SEDORGN_TOCH[0])) m_surfRfSedOrgNToCh = data;
else if (StringMatch(sk, VAR_SEDORGP_TOCH[0])) m_surfRfSedOrgPToCh = data;
else if (StringMatch(sk, VAR_SEDMINPA_TOCH[0])) m_surfRfSedAbsorbMinPToCh = data;
else if (StringMatch(sk, VAR_SEDMINPS_TOCH[0])) m_surfRfSedSorbMinPToCh = data;
else if (StringMatch(sk, VAR_NO2_TOCH[0])) m_no2ToCh = data;
else if (StringMatch(sk, VAR_RCH_DEG[0])) m_rchDeg = data;
else {
throw ModelException(M_NUTRCH_QUAL2E[0], "Set1DData",
"Parameter " + sk + " does not exist.");
}
}

void NutrCH_QUAL2E::Set1DData(const char* key, const int n, int* data) {
string sk(key);
if (StringMatch(sk, VAR_STREAM_LINK[0])) {
CheckInputCellSize(key, n);
m_rchID = data;
return;
}
throw ModelException(M_NUTRCH_QUAL2E[0], "Set1DData",
"Integer Parameter " + sk + " does not exist.");
}

void NutrCH_QUAL2E::SetReaches(clsReaches* reaches) {
if (nullptr == reaches) {
throw ModelException(M_NUTRCH_QUAL2E[0], "SetReaches",
"The reaches input can not to be NULL.");
}
m_nReaches = reaches->GetReachNumber();

if (nullptr == m_reachDownStream) {
FLTPT* tmp = nullptr;
reaches->GetReachesSingleProperty(REACH_DOWNSTREAM, &tmp);
Initialize1DArray(m_nReaches + 1, m_reachDownStream, tmp);
Release1DArray(tmp);
}
if (nullptr == m_bc1) reaches->GetReachesSingleProperty(REACH_BC1, &m_bc1);
if (nullptr == m_bc2) reaches->GetReachesSingleProperty(REACH_BC2, &m_bc2);
if (nullptr == m_bc3) reaches->GetReachesSingleProperty(REACH_BC3, &m_bc3);
if (nullptr == m_bc4) reaches->GetReachesSingleProperty(REACH_BC4, &m_bc4);
if (nullptr == m_rk1) reaches->GetReachesSingleProperty(REACH_RK1, &m_rk1);
if (nullptr == m_rk2) reaches->GetReachesSingleProperty(REACH_RK2, &m_rk2);
if (nullptr == m_rk3) reaches->GetReachesSingleProperty(REACH_RK3, &m_rk3);
if (nullptr == m_rk4) reaches->GetReachesSingleProperty(REACH_RK4, &m_rk4);
if (nullptr == m_rs1) reaches->GetReachesSingleProperty(REACH_RS1, &m_rs1);
if (nullptr == m_rs2) reaches->GetReachesSingleProperty(REACH_RS2, &m_rs2);
if (nullptr == m_rs3) reaches->GetReachesSingleProperty(REACH_RS3, &m_rs3);
if (nullptr == m_rs4) reaches->GetReachesSingleProperty(REACH_RS4, &m_rs4);
if (nullptr == m_rs5) reaches->GetReachesSingleProperty(REACH_RS5, &m_rs5);
if (nullptr == m_chAlgae) reaches->GetReachesSingleProperty(REACH_ALGAE, &m_chAlgae);
if (nullptr == m_chOrgN) reaches->GetReachesSingleProperty(REACH_ORGN, &m_chOrgN);
if (nullptr == m_chOrgP) reaches->GetReachesSingleProperty(REACH_ORGP, &m_chOrgP);
if (nullptr == m_chNH4) reaches->GetReachesSingleProperty(REACH_NH4, &m_chNH4);
if (nullptr == m_chNO2) reaches->GetReachesSingleProperty(REACH_NO2, &m_chNO2);
if (nullptr == m_chNO3) reaches->GetReachesSingleProperty(REACH_NO3, &m_chNO3);
if (nullptr == m_chSolP) reaches->GetReachesSingleProperty(REACH_SOLP, &m_chSolP);
if (nullptr == m_chDOx) reaches->GetReachesSingleProperty(REACH_DISOX, &m_chDOx);
if (nullptr == m_chCOD) reaches->GetReachesSingleProperty(REACH_BOD, &m_chCOD);

if (nullptr == m_chChlora) Initialize1DArray(m_nReaches + 1, m_chChlora, 0.);
if (nullptr == m_chTP) Initialize1DArray(m_nReaches + 1, m_chTP, 0.);
if (nullptr == m_chTN) Initialize1DArray(m_nReaches + 1, m_chTN, 0.);

m_reachUpStream = reaches->GetUpStreamIDs();
m_reachLayers = reaches->GetReachLayers();
}

void NutrCH_QUAL2E::SetScenario(Scenario* sce) {
if (nullptr == sce) {
throw ModelException(M_NUTRCH_QUAL2E[0], "SetScenario",
"The scenario can not to be NULL.");
}
map<int, BMPFactory *>& tmpBMPFactories = sce->GetBMPFactories();
for (auto it = tmpBMPFactories.begin(); it != tmpBMPFactories.end(); ++it) {
if (it->first / 100000 != BMP_TYPE_POINTSOURCE) continue;
#ifdef HAS_VARIADIC_TEMPLATES
m_ptSrcFactory.emplace(it->first, static_cast<BMPPointSrcFactory *>(it->second));
#else
m_ptSrcFactory.insert(make_pair(it->first, static_cast<BMPPointSrcFactory *>(it->second)));
#endif
}
}

void NutrCH_QUAL2E::InitialOutputs() {
CHECK_POSITIVE(M_NUTRCH_QUAL2E[0], m_nReaches);
if (nullptr == m_chOutAlgae) {
m_chSatDOx = 0.;
Initialize1DArray(m_nReaches + 1, m_chOutChlora, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutAlgae, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutOrgN, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutOrgP, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutNH4, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutNO2, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutNO3, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutSolP, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutDOx, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutCOD, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutTN, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutTP, 0.);

Initialize1DArray(m_nReaches + 1, m_chOutChloraConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutAlgaeConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutOrgNConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutOrgPConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutNH4Conc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutNO2Conc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutNO3Conc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutSolPConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutDOxConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutCODConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutTNConc, 0.);
Initialize1DArray(m_nReaches + 1, m_chOutTPConc, 0.);
}
}

void NutrCH_QUAL2E::PointSourceLoading() {
if (nullptr == m_ptNO3ToCh) {
Initialize1DArray(m_nReaches + 1, m_ptNO3ToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptNH4ToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptOrgNToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptTNToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptSolPToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptOrgPToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptTPToCh, 0.);
Initialize1DArray(m_nReaches + 1, m_ptCODToCh, 0.);
} else {
#pragma omp parallel for
for (int i = 0; i <= m_nReaches; i++) {
m_ptNO3ToCh[i] = 0.;
m_ptNH4ToCh[i] = 0.;
m_ptOrgNToCh[i] = 0.;
m_ptTNToCh[i] = 0.;
m_ptSolPToCh[i] = 0.;
m_ptOrgPToCh[i] = 0.;
m_ptTPToCh[i] = 0.;
m_ptCODToCh[i] = 0.;
}
}
for (auto it = m_ptSrcFactory.begin(); it != m_ptSrcFactory.end(); ++it) {
vector<int>& ptSrcMgtSeqs = it->second->GetPointSrcMgtSeqs();
map<int, PointSourceMgtParams *>& pointSrcMgtMap = it->second->GetPointSrcMgtMap();
vector<int>& ptSrcIDs = it->second->GetPointSrcIDs();
map<int, PointSourceLocations *>& pointSrcLocsMap = it->second->GetPointSrcLocsMap();
for (auto seqIter = ptSrcMgtSeqs.begin(); seqIter != ptSrcMgtSeqs.end(); ++seqIter) {
PointSourceMgtParams* curPtMgt = pointSrcMgtMap.at(*seqIter);
if (curPtMgt->GetStartDate() != 0 && curPtMgt->GetEndDate() != 0) {
if (m_date < curPtMgt->GetStartDate() || m_date > curPtMgt->GetEndDate()) {
continue;
}
}
FLTPT per_wtr = curPtMgt->GetWaterVolume();
FLTPT per_no3 = curPtMgt->GetNO3();
FLTPT per_nh4 = curPtMgt->GetNH4();
FLTPT per_orgn = curPtMgt->GetOrgN();
FLTPT per_solp = curPtMgt->GetSolP();
FLTPT per_orgp = curPtMgt->GetOrgP();
FLTPT per_cod = curPtMgt->GetCOD();
for (auto locIter = ptSrcIDs.begin(); locIter != ptSrcIDs.end(); ++locIter) {
if (pointSrcLocsMap.find(*locIter) != pointSrcLocsMap.end()) {
PointSourceLocations* curPtLoc = pointSrcLocsMap.at(*locIter);
int curSubID = curPtLoc->GetSubbasinID();
FLTPT cvt = per_wtr * curPtLoc->GetSize() * 0.001 * m_dt * 1.1574074074074073e-05;
m_ptNO3ToCh[curSubID] += per_no3 * cvt;
m_ptNH4ToCh[curSubID] += per_nh4 * cvt;
m_ptOrgNToCh[curSubID] += per_orgn * cvt;
m_ptOrgPToCh[curSubID] += per_orgp * cvt;
m_ptSolPToCh[curSubID] += per_solp * cvt;
m_ptCODToCh[curSubID] += per_cod * cvt;
m_ptTNToCh[curSubID] += (per_no3 + per_nh4 + per_orgn) * cvt;
m_ptTPToCh[curSubID] += (per_solp + per_orgp) * cvt;
}
}
}
}
for (int i = 1; i <= m_nReaches; i++) {
m_ptTNToCh[0] += m_ptTNToCh[i];
m_ptTPToCh[0] += m_ptTPToCh[i];
m_ptCODToCh[0] += m_ptCODToCh[i];
}
}

int NutrCH_QUAL2E::Execute() {
CheckInputData();
InitialOutputs();
PointSourceLoading();
ParametersSubbasinForChannel();

for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); ++it) {
int reachNum = CVT_INT(it->second.size());
#pragma omp parallel for
for (int i = 0; i < reachNum; i++) {
int reachIndex = it->second[i];
if (m_inputSubbsnID == 0 || m_inputSubbsnID == reachIndex) {
NutrientTransform(reachIndex);
AddInputNutrient(reachIndex);
RouteOut(reachIndex);
}
}
}
return 0;
}

void NutrCH_QUAL2E::AddInputNutrient(const int i) {
for (auto upRchID = m_reachUpStream.at(i).begin(); upRchID != m_reachUpStream.at(i).end(); ++upRchID) {
int upReachId = *upRchID;
m_chOrgN[i] += m_chOutOrgN[upReachId];
m_chNO3[i] += m_chOutNO3[upReachId];
m_chNO2[i] += m_chOutNO2[upReachId];
m_chNH4[i] += m_chOutNH4[upReachId];
m_chOrgP[i] += m_chOutOrgP[upReachId];
m_chSolP[i] += m_chOutSolP[upReachId];
m_chCOD[i] += m_chOutCOD[upReachId];
m_chDOx[i] += m_chOutDOx[upReachId];
m_chChlora[i] += m_chOutChlora[upReachId];
m_chAlgae[i] += m_chOutAlgae[upReachId];
}
m_chOrgN[i] += m_surfRfSedOrgNToCh[i];
m_chOrgP[i] += m_surfRfSedOrgPToCh[i];
if (nullptr != m_rchDeg && FloatEqual(m_chOrgPCo, NODATA_VALUE) && FloatEqual(m_chOrgNCo, NODATA_VALUE)) {
m_chOrgN[i] += m_rchDeg[i] * m_chOrgNCo * 0.001;
m_chOrgP[i] += m_rchDeg[i] * m_chOrgPCo * 0.001;
}
m_chNO3[i] += m_surfRfNO3ToCh[i] + m_latNO3ToCh[i] + m_gwNO3ToCh[i];
if (nullptr != m_surfRfNH4ToCh && m_surfRfNH4ToCh[i] > 0.) m_chNH4[i] += m_surfRfNH4ToCh[i];
m_chSolP[i] += m_surfRfSolPToCh[i] + m_gwSolPToCh[i];

if (nullptr != m_no2ToCh && m_no2ToCh[i] > 0.) m_chNO2[i] += m_no2ToCh[i];
if (nullptr != m_surfRfCodToCh && m_surfRfCodToCh[i] > 0.) {
m_chCOD[i] += m_surfRfCodToCh[i];
}
if (nullptr != m_ptNO3ToCh && m_ptNO3ToCh[i] > 0.) m_chNO3[i] += m_ptNO3ToCh[i];
if (nullptr != m_ptNH4ToCh && m_ptNH4ToCh[i] > 0.) m_chNH4[i] += m_ptNH4ToCh[i];
if (nullptr != m_ptOrgNToCh && m_ptOrgNToCh[i] > 0.) m_chOrgN[i] += m_ptOrgNToCh[i];
if (nullptr != m_ptSolPToCh && m_ptSolPToCh[i] > 0.) m_chSolP[i] += m_ptSolPToCh[i];
if (nullptr != m_ptOrgPToCh && m_ptOrgPToCh[i] > 0.) m_chOrgP[i] += m_ptOrgPToCh[i];
if (nullptr != m_ptCODToCh && m_ptCODToCh[i] > 0.) m_chCOD[i] += m_ptCODToCh[i];
}

void NutrCH_QUAL2E::RouteOut(const int i) {
m_chOutAlgae[i] = 0.;
m_chOutAlgaeConc[i] = 0.;
m_chOutChlora[i] = 0.;
m_chOutChloraConc[i] = 0.;
m_chOutOrgN[i] = 0.;
m_chOutOrgNConc[i] = 0.;
m_chOutNH4[i] = 0.;
m_chOutNH4Conc[i] = 0.;
m_chOutNO2[i] = 0.;
m_chOutNO2Conc[i] = 0.;
m_chOutNO3[i] = 0.;
m_chOutNO3Conc[i] = 0.;
m_chOutOrgP[i] = 0.;
m_chOutOrgPConc[i] = 0.;
m_chOutSolP[i] = 0.;
m_chOutSolPConc[i] = 0.;
m_chOutCOD[i] = 0.;
m_chOutCODConc[i] = 0.;
m_chOutDOx[i] = 0.;
m_chOutDOxConc[i] = 0.;
m_chOutTN[i] = 0.;
m_chOutTNConc[i] = 0.;
m_chOutTP[i] = 0.;
m_chOutTPConc[i] = 0.;
FLTPT wtrTotal = m_chStorage[i] + m_rteWtrOut[i]; 
if (wtrTotal <= UTIL_ZERO || m_rteWtrOut[i] <= UTIL_ZERO || m_chWtrDepth[i] <= UTIL_ZERO) {
return;
}
FLTPT outFraction = m_rteWtrOut[i] / wtrTotal;
if (outFraction >= 1.) outFraction = 1.;
if (outFraction <= UTIL_ZERO) outFraction = UTIL_ZERO;
m_chOutOrgN[i] = m_chOrgN[i] * outFraction;
m_chOutNO3[i] = m_chNO3[i] * outFraction;
m_chOutNO2[i] = m_chNO2[i] * outFraction;
m_chOutNH4[i] = m_chNH4[i] * outFraction;
m_chOutOrgP[i] = m_chOrgP[i] * outFraction;
m_chOutSolP[i] = m_chSolP[i] * outFraction;
m_chOutCOD[i] = m_chCOD[i] * outFraction;
m_chOutDOx[i] = m_chDOx[i] * outFraction;
m_chOutAlgae[i] = m_chAlgae[i] * outFraction;
m_chOutChlora[i] = m_chChlora[i] * outFraction;
m_chOutTN[i] = m_chOutOrgN[i] + m_chOutNH4[i] + m_chOutNO2[i] + m_chOutNO3[i];
m_chOutTP[i] = m_chOutOrgP[i] + m_chOutSolP[i];
FLTPT cvt = 1000. / m_rteWtrOut[i];
m_chOutOrgNConc[i] = m_chOutOrgN[i] * cvt;
m_chOutNO3Conc[i] = m_chOutNO3[i] * cvt;
m_chOutNO2Conc[i] = m_chOutNO2[i] * cvt;
m_chOutNH4Conc[i] = m_chOutNH4[i] * cvt;
m_chOutOrgPConc[i] = m_chOutOrgP[i] * cvt;
m_chOutSolPConc[i] = m_chOutSolP[i] * cvt;
m_chOutCODConc[i] = m_chOutCOD[i] * cvt;
m_chOutDOxConc[i] = m_chOutDOx[i] * cvt;
m_chOutAlgaeConc[i] = m_chOutAlgae[i] * cvt;
m_chOutChloraConc[i] = m_chOutChlora[i] * cvt;
m_chOutTNConc[i] = m_chOutTN[i] * cvt;
m_chOutTPConc[i] = m_chOutTP[i] * cvt;

m_chNO3[i] -= m_chOutNO3[i];
m_chNO2[i] -= m_chOutNO2[i];
m_chNH4[i] -= m_chOutNH4[i];
m_chOrgN[i] -= m_chOutOrgN[i];
m_chOrgP[i] -= m_chOutOrgP[i];
m_chSolP[i] -= m_chOutSolP[i];
m_chCOD[i] -= m_chOutCOD[i];
m_chDOx[i] -= m_chOutDOx[i];
m_chAlgae[i] -= m_chOutAlgae[i];
m_chChlora[i] -= m_chOutChlora[i];
m_chTN[i] = m_chOrgN[i] + m_chNH4[i] + m_chNO2[i] + m_chNO3[i];
m_chTP[i] = m_chOrgP[i] + m_chSolP[i];

if (m_chNO3[i] < UTIL_ZERO) m_chNO3[i] = UTIL_ZERO;
if (m_chNO2[i] < UTIL_ZERO) m_chNO2[i] = UTIL_ZERO;
if (m_chNH4[i] < UTIL_ZERO) m_chNH4[i] = UTIL_ZERO;
if (m_chOrgN[i] < UTIL_ZERO) m_chOrgN[i] = UTIL_ZERO;
if (m_chOrgP[i] < UTIL_ZERO) m_chOrgP[i] = UTIL_ZERO;
if (m_chSolP[i] < UTIL_ZERO) m_chSolP[i] = UTIL_ZERO;
if (m_chCOD[i] < UTIL_ZERO) m_chCOD[i] = UTIL_ZERO;
if (m_chDOx[i] < UTIL_ZERO) m_chDOx[i] = UTIL_ZERO;
if (m_chAlgae[i] < UTIL_ZERO) m_chAlgae[i] = UTIL_ZERO;
if (m_chChlora[i] < UTIL_ZERO) m_chChlora[i] = UTIL_ZERO;
}

void NutrCH_QUAL2E::NutrientTransform(const int i) {
FLTPT thbc1 = 1.083; 
FLTPT thbc2 = 1.047; 
FLTPT thbc3 = 1.04;  
FLTPT thbc4 = 1.047; 

FLTPT thgra = 1.047; 
FLTPT thrho = 1.047; 

FLTPT thm_rk1 = 1.047; 
FLTPT thm_rk2 = 1.024; 
FLTPT thm_rk3 = 1.024; 
FLTPT thm_rk4 = 1.060; 

FLTPT thrs1 = 1.024; 
FLTPT thrs2 = 1.074; 
FLTPT thrs3 = 1.074; 
FLTPT thrs4 = 1.024; 
FLTPT thrs5 = 1.024; 

FLTPT wtrTotal = m_rteWtrOut[i] + m_chStorage[i]; 
if (m_chWtrDepth[i] <= 0.01) {
m_chWtrDepth[i] = 0.01;
}
if (wtrTotal <= 0.) {
m_chAlgae[i] = 0.;
m_chChlora[i] = 0.;
m_chOrgN[i] = 0.;
m_chNH4[i] = 0.;
m_chNO2[i] = 0.;
m_chNO3[i] = 0.;
m_chTN[i] = 0.;
m_chOrgP[i] = 0.;
m_chSolP[i] = 0.;
m_chTP[i] = 0.;
m_chDOx[i] = 0.;
m_chCOD[i] = 0.;
m_chSatDOx = 0.;
return; 
}
FLTPT cvt_amout2conc = 1000. / wtrTotal;
FLTPT algcon = cvt_amout2conc * m_chAlgae[i];
FLTPT orgncon = cvt_amout2conc * m_chOrgN[i];
FLTPT nh4con = cvt_amout2conc * m_chNH4[i];
FLTPT no2con = cvt_amout2conc * m_chNO2[i];
FLTPT no3con = cvt_amout2conc * m_chNO3[i];
FLTPT orgpcon = cvt_amout2conc * m_chOrgP[i];
FLTPT solpcon = cvt_amout2conc * m_chSolP[i];
FLTPT cbodcon = cvt_amout2conc * m_chCOD[i];
FLTPT o2con = cvt_amout2conc * m_chDOx[i];

FLTPT wtmp = Max(m_chTemp[i], 0.1);
FLTPT cinn = nh4con + no3con;

FLTPT ww = -139.34410 + 1.575701e+05 / (wtmp + 273.15);
FLTPT xx = 6.642308e+07 / CalPow(wtmp + 273.15, 2.);
FLTPT yy = 1.243800e+10 / CalPow(wtmp + 273.15, 3.);
FLTPT zz = 8.621949e+11 / CalPow(wtmp + 273.15, 4.);
m_chSatDOx = 0.;
m_chSatDOx = CalExp(ww - xx + yy - zz);
if (m_chSatDOx < 1.e-6) {
m_chSatDOx = 0.;
}

FLTPT cordo = 0.;
FLTPT o2con2 = o2con;
if (o2con2 <= 0.1) {
o2con2 = 0.1;
}
if (o2con2 > 30.) {
o2con2 = 30.;
}
cordo = 1. - CalExp(-0.6 * o2con2);
if (o2con <= 0.001) {
o2con = 0.001;
}
if (o2con > 30.) {
o2con = 30.;
}
cordo = 1. - CalExp(-0.6 * o2con);


FLTPT bc1mod = 0.;
FLTPT bc2mod = 0.;
bc1mod = m_bc1[i] * cordo;
bc2mod = m_bc2[i] * cordo;

FLTPT tday = 1.;

FLTPT lambda = 0.;
if (m_ai0 * algcon > 1.e-6) {
lambda = m_lambda0 + m_lambda1 * m_ai0 * algcon + m_lambda2 * CalPow(m_ai0 * algcon, 0.66667);
} else {
lambda = m_lambda0;
}
if (lambda > m_lambda0) lambda = m_lambda0;
FLTPT fnn = 0.;
FLTPT fpp = 0.;
fnn = cinn / (cinn + m_k_n);
fpp = solpcon / (solpcon + m_k_p);

FLTPT algi = 0.;
if (m_chDaylen[i] > 0.) {
algi = m_chSr[i] * tfact / m_chDaylen[i];
} else {
algi = 0.00001;
}

FLTPT fl_1 = 0.;
FLTPT fll = 0.;
fl_1 = 1. / (lambda * m_chWtrDepth[i]) * CalLn((m_k_l + algi)
/ (m_k_l + algi * CalExp(-lambda * m_chWtrDepth[i])));

fll = 0.92 * (m_chDaylen[i] / 24.) * fl_1;

FLTPT gra = 0.;
FLTPT dbod = 0.;
FLTPT ddisox = 0.;
FLTPT dorgn = 0.;
FLTPT dnh4 = 0.;
FLTPT dno2 = 0.;
FLTPT dno3 = 0.;
FLTPT dorgp = 0.;
FLTPT dsolp = 0.;
switch (igropt) {
case 1:
gra = m_mumax * fll * fnn * fpp;
case 2:
gra = m_mumax * fll * Min(fnn, fpp);
case 3:
if (fnn > 1.e-6 && fpp > 1.e-6f) {
gra = m_mumax * fll * 2. / (1. / fnn + 1. / fpp);
} else {
gra = 0.;
}
default: break;
}

FLTPT dalgae = 0.;
FLTPT setl = Min(1., corTempc(m_rs1[i], thrs1, wtmp) / m_chWtrDepth[i]);
dalgae = algcon + (corTempc(gra, thgra, wtmp) * algcon -
corTempc(m_rhoq, thrho, wtmp) * algcon - setl * algcon) * tday;
if (dalgae < 1.e-6) {
dalgae = 1.e-6;
}
FLTPT dcoef = 3.;
if (dalgae > 5000.) dalgae = 5000.;
if (dalgae > dcoef * algcon) dalgae = dcoef * algcon;

FLTPT yyy = 0.;
FLTPT zzz = 0.;
cbodcon /= m_cod_n * (1. - CalExp(-5. * m_cod_k));
yyy = corTempc(m_rk1[i], thm_rk1, wtmp) * cbodcon;
zzz = corTempc(m_rk3[i], thm_rk3, wtmp) * cbodcon;
dbod = 0.;
dbod = cbodcon - (yyy + zzz) * tday;


if (dbod < 1.e-6) dbod = 1.e-6;
dbod *= m_cod_n * (1. - CalExp(-5. * m_cod_k));

FLTPT uu = 0.; 
FLTPT vv = 0.; 
ww = 0.;       
xx = 0.;       
yy = 0.;       
zz = 0.;       


uu = corTempc(m_rk2[i], thm_rk2, wtmp) * (m_chSatDOx - o2con);
if (algcon > 0.001) {
vv = (m_ai3 * corTempc(gra, thgra, wtmp) - m_ai4 * corTempc(m_rhoq, thrho, wtmp)) * algcon;
} else {
algcon = 0.001;
}

ww = corTempc(m_rk1[i], thm_rk1, wtmp) * cbodcon;
if (m_chWtrDepth[i] > 0.001) {
xx = corTempc(m_rk4[i], thm_rk4, wtmp) / (m_chWtrDepth[i] * 1000.);
}
if (nh4con > 0.001) {
yy = m_ai5 * corTempc(bc1mod, thbc1, wtmp) * nh4con;
} else {
nh4con = 0.001;
}
if (no2con > 0.001) {
zz = m_ai6 * corTempc(bc2mod, thbc2, wtmp) * no2con;
} else {
no2con = 0.001;
}
ddisox = 0.;
ddisox = o2con + (uu + vv - ww - xx - yy - zz) * tday;
if (ddisox < 0.1 || ddisox != ddisox) {
ddisox = 0.1;
}
xx = 0.;
yy = 0.;
zz = 0.;
xx = m_ai1 * corTempc(m_rhoq, thrho, wtmp) * algcon;
yy = corTempc(m_bc3[i], thbc3, wtmp) * orgncon;
zz = corTempc(m_rs4[i], thrs4, wtmp) * orgncon;
dorgn = 0.;
dorgn = orgncon + (xx - yy - zz) * tday;
if (dorgn < 1.e-6) dorgn = 0.;
if (dorgn > dcoef * orgncon) dorgn = dcoef * orgncon;
FLTPT f1 = 0.;
f1 = m_p_n * nh4con / (m_p_n * nh4con + (1. - m_p_n) * no3con + 1.e-6);

ww = 0.;
xx = 0.;
yy = 0.;
zz = 0.;
ww = corTempc(m_bc3[i], thbc3, wtmp) * orgncon;
xx = corTempc(bc1mod, thbc1, wtmp) * nh4con;
yy = corTempc(m_rs3[i], thrs3, wtmp) / (m_chWtrDepth[i] * 1000.);
zz = f1 * m_ai1 * algcon * corTempc(gra, thgra, wtmp);
dnh4 = 0.;
dnh4 = nh4con + (ww - xx + yy - zz) * tday;
if (dnh4 < 1.e-6) dnh4 = 0.;
if (dnh4 > dcoef * nh4con && nh4con > 0.) {
dnh4 = dcoef * nh4con;
}
yy = 0.;
zz = 0.;
yy = corTempc(bc1mod, thbc1, wtmp) * nh4con;
zz = corTempc(bc2mod, thbc2, wtmp) * no2con;
dno2 = 0.;
dno2 = no2con + (yy - zz) * tday;
if (dno2 < 1.e-6) dno2 = 0.;
if (dno2 > dcoef * no2con && no2con > 0.) {
dno2 = dcoef * no2con;
}
yy = 0.;
zz = 0.;
yy = corTempc(bc2mod, thbc2, wtmp) * no2con;
zz = (1. - f1) * m_ai1 * algcon * corTempc(gra, thgra, wtmp);
dno3 = 0.;
dno3 = no3con + (yy - zz) * tday;
if (dno3 < 1.e-6) dno3 = 0.;
xx = 0.;
yy = 0.;
zz = 0.;
xx = m_ai2 * corTempc(m_rhoq, thrho, wtmp) * algcon;
yy = corTempc(m_bc4[i], thbc4, wtmp) * orgpcon;
zz = corTempc(m_rs5[i], thrs5, wtmp) * orgpcon;
dorgp = 0.;
dorgp = orgpcon + (xx - yy - zz) * tday;
if (dorgp < 1.e-6) dorgp = 0.;
if (dorgp > dcoef * orgpcon) {
dorgp = dcoef * orgpcon;
}

xx = 0.;
yy = 0.;
zz = 0.;
xx = corTempc(m_bc4[i], thbc4, wtmp) * orgpcon;
yy = corTempc(m_rs2[i], thrs2, wtmp) / (m_chWtrDepth[i] * 1000.);
zz = m_ai2 * corTempc(gra, thgra, wtmp) * algcon;
dsolp = 0.;
dsolp = solpcon + (xx + yy - zz) * tday;
if (dsolp < 1.e-6) dsolp = 0.;
if (dsolp > dcoef * solpcon) {
dsolp = dcoef * solpcon;
}
m_chAlgae[i] = dalgae * wtrTotal * 0.001;
m_chChlora[i] = m_chAlgae[i] * m_ai0;
m_chOrgN[i] = dorgn * wtrTotal * 0.001;
m_chNH4[i] = dnh4 * wtrTotal * 0.001;
m_chNO2[i] = dno2 * wtrTotal * 0.001;
m_chNO3[i] = dno3 * wtrTotal * 0.001;
m_chOrgP[i] = dorgp * wtrTotal * 0.001;
m_chSolP[i] = dsolp * wtrTotal * 0.001;
m_chCOD[i] = dbod * wtrTotal * 0.001;
m_chDOx[i] = ddisox * wtrTotal / 1000.;
}

FLTPT NutrCH_QUAL2E::corTempc(const FLTPT r20, const FLTPT thk, const FLTPT tmp) {
return r20 * CalPow(thk, tmp - 20.);
}

void NutrCH_QUAL2E::GetValue(const char* key, FLTPT* value) {
string sk(key);
if (StringMatch(sk, VAR_SOXY[0])) *value = m_chSatDOx;
else if (StringMatch(sk, VAR_CH_ALGAE[0])) *value = m_chOutAlgae[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_ALGAEConc[0])) *value = m_chOutAlgaeConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_NO2[0])) *value = m_chOutNO2[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_NO2Conc[0])) *value = m_chOutNO2Conc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_COD[0])) *value = m_chOutCOD[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_CODConc[0])) *value = m_chOutCODConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_CHLORA[0])) *value = m_chOutChlora[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_CHLORAConc[0])) *value = m_chOutChloraConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_NO3[0])) *value = m_chOutNO3[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_NO3Conc[0])) *value = m_chOutNO3Conc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_SOLP[0])) *value = m_chOutSolP[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_SOLPConc[0])) *value = m_chOutSolPConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_ORGN[0])) *value = m_chOutOrgN[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_ORGNConc[0])) *value = m_chOutOrgNConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_ORGP[0])) *value = m_chOutOrgP[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_ORGPConc[0])) *value = m_chOutOrgPConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_NH4[0])) *value = m_chOutNH4[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_NH4Conc[0])) *value = m_chOutNH4Conc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_DOX[0])) *value = m_chOutDOx[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_DOXConc[0])) *value = m_chOutDOxConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_TN[0])) *value = m_chOutTN[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_TNConc[0])) *value = m_chOutTNConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_TP[0])) *value = m_chOutTP[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CH_TPConc[0])) *value = m_chOutTPConc[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CHSTR_NO3[0])) *value = m_chNO3[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CHSTR_NH4[0])) *value = m_chNH4[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CHSTR_TN[0])) *value = m_chTN[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CHSTR_TP[0])) *value = m_chTP[m_inputSubbsnID];
else {
throw ModelException(M_NUTRCH_QUAL2E[0], "GetValue",
"Parameter " + sk + " does not exist.");
}
}

void NutrCH_QUAL2E::Get1DData(const char* key, int* n, FLTPT** data) {
InitialOutputs();
string sk(key);
*n = m_nReaches + 1;
if (StringMatch(sk, VAR_CH_ALGAE[0])) *data = m_chOutAlgae;
else if (StringMatch(sk, VAR_CH_ALGAEConc[0])) *data = m_chOutAlgaeConc;
else if (StringMatch(sk, VAR_CH_NO2[0])) *data = m_chOutNO2;
else if (StringMatch(sk, VAR_CH_NO2Conc[0])) *data = m_chOutNO2Conc;
else if (StringMatch(sk, VAR_CH_COD[0])) *data = m_chOutCOD;
else if (StringMatch(sk, VAR_CH_CODConc[0])) *data = m_chOutCODConc;
else if (StringMatch(sk, VAR_CH_CHLORA[0])) *data = m_chOutChlora;
else if (StringMatch(sk, VAR_CH_CHLORAConc[0])) *data = m_chOutChloraConc;
else if (StringMatch(sk, VAR_CH_NO3[0])) *data = m_chOutNO3;
else if (StringMatch(sk, VAR_CH_NO3Conc[0])) *data = m_chOutNO3Conc;
else if (StringMatch(sk, VAR_CH_SOLP[0])) *data = m_chOutSolP;
else if (StringMatch(sk, VAR_CH_SOLPConc[0])) *data = m_chOutSolPConc;
else if (StringMatch(sk, VAR_CH_ORGN[0])) *data = m_chOutOrgN;
else if (StringMatch(sk, VAR_CH_ORGNConc[0])) *data = m_chOutOrgNConc;
else if (StringMatch(sk, VAR_CH_ORGP[0])) *data = m_chOutOrgP;
else if (StringMatch(sk, VAR_CH_ORGPConc[0])) *data = m_chOutOrgPConc;
else if (StringMatch(sk, VAR_CH_NH4[0])) *data = m_chOutNH4;
else if (StringMatch(sk, VAR_CH_NH4Conc[0])) *data = m_chOutNH4Conc;
else if (StringMatch(sk, VAR_CH_DOX[0])) *data = m_chOutDOx;
else if (StringMatch(sk, VAR_CH_DOXConc[0])) *data = m_chOutDOxConc;
else if (StringMatch(sk, VAR_CH_TN[0])) *data = m_chOutTN;
else if (StringMatch(sk, VAR_CH_TNConc[0])) *data = m_chOutTNConc;
else if (StringMatch(sk, VAR_CH_TP[0])) *data = m_chOutTP;
else if (StringMatch(sk, VAR_CH_TPConc[0])) *data = m_chOutTPConc;
else if (StringMatch(sk, VAR_PTTN2CH[0])) *data = m_ptTNToCh;
else if (StringMatch(sk, VAR_PTTP2CH[0])) *data = m_ptTPToCh;
else if (StringMatch(sk, VAR_PTCOD2CH[0])) *data = m_ptCODToCh;
else if (StringMatch(sk, VAR_CHSTR_NO3[0])) *data = m_chNO3;
else if (StringMatch(sk, VAR_CHSTR_NH4[0])) *data = m_chNH4;
else if (StringMatch(sk, VAR_CHSTR_TN[0])) *data = m_chTN;
else if (StringMatch(sk, VAR_CHSTR_TP[0])) *data = m_chTP;
else {
throw ModelException(M_NUTRCH_QUAL2E[0], "Get1DData",
"Parameter " + sk + " does not exist.");
}
}
