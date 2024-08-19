#include "NutrientTransportSediment.h"

#include "text.h"
#include "NutrientCommon.h"

NutrientTransportSediment::NutrientTransportSediment() :
m_nSubbsns(-1), m_inputSubbsnID(-1), m_cellWth(-1.), m_cellArea(-1.), m_nCells(-1),
m_nSoilLyrs(nullptr), m_maxSoilLyrs(-1),
m_soilRock(nullptr), m_soilSat(nullptr), m_cbnModel(0), m_enratio(nullptr),
m_olWtrEroSed(nullptr), m_surfRf(nullptr), m_soilBD(nullptr), m_soilThk(nullptr), m_soilMass(nullptr),
m_subbsnID(nullptr), m_subbasinsInfo(nullptr), m_surfRfSedOrgN(nullptr),
m_surfRfSedOrgP(nullptr), m_surfRfSedAbsorbMinP(nullptr), m_surfRfSedSorbMinP(nullptr),
m_surfRfSedOrgNToCh(nullptr), m_surfRfSedOrgPToCh(nullptr),
m_surfRfSedAbsorbMinPToCh(nullptr), m_surfRfSedSorbMinPToCh(nullptr), m_soilActvOrgN(nullptr),
m_soilFrshOrgN(nullptr),
m_soilStabOrgN(nullptr),
m_soilHumOrgP(nullptr), m_soilFrshOrgP(nullptr), m_soilStabMinP(nullptr), m_soilActvMinP(nullptr),
m_soilManP(nullptr),
m_sol_LSN(nullptr), m_sol_LMN(nullptr), m_sol_HPN(nullptr), m_sol_HSN(nullptr), m_sol_HPC(nullptr),
m_sol_HSC(nullptr), m_sol_LMC(nullptr), m_sol_LSC(nullptr),
m_sol_LS(nullptr),
m_sol_LM(nullptr), m_sol_LSL(nullptr), m_sol_LSLC(nullptr), m_sol_LSLNC(nullptr), m_sol_BMC(nullptr),
m_sol_WOC(nullptr), m_soilPerco(nullptr), m_subSurfRf(nullptr), m_soilIfluCbn(nullptr),
m_soilPercoCbn(nullptr), m_soilIfluCbnPrfl(nullptr), m_soilPercoCbnPrfl(nullptr), m_sedLossCbn(nullptr) {
}

NutrientTransportSediment::~NutrientTransportSediment() {
if (m_soilMass != nullptr) Release2DArray(m_soilMass);
if (m_enratio != nullptr) Release1DArray(m_enratio);

if (m_surfRfSedOrgP != nullptr) Release1DArray(m_surfRfSedOrgP);
if (m_surfRfSedOrgN != nullptr) Release1DArray(m_surfRfSedOrgN);
if (m_surfRfSedAbsorbMinP != nullptr) Release1DArray(m_surfRfSedAbsorbMinP);
if (m_surfRfSedSorbMinP != nullptr) Release1DArray(m_surfRfSedSorbMinP);

if (m_surfRfSedOrgNToCh != nullptr) Release1DArray(m_surfRfSedOrgNToCh);
if (m_surfRfSedOrgPToCh != nullptr) Release1DArray(m_surfRfSedOrgPToCh);
if (m_surfRfSedAbsorbMinPToCh != nullptr) Release1DArray(m_surfRfSedAbsorbMinPToCh);
if (m_surfRfSedSorbMinPToCh != nullptr) Release1DArray(m_surfRfSedSorbMinPToCh);

if (m_soilIfluCbn != nullptr) Release2DArray(m_soilPercoCbn);
if (m_soilPercoCbn != nullptr) Release2DArray(m_soilPercoCbn);
if (m_soilIfluCbnPrfl != nullptr) Release1DArray(m_soilIfluCbnPrfl);
if (m_soilPercoCbnPrfl != nullptr) Release1DArray(m_soilPercoCbnPrfl);
if (m_sedLossCbn != nullptr) Release1DArray(m_sedLossCbn);
}

bool NutrientTransportSediment::CheckInputData() {
CHECK_POSITIVE(M_NUTRSED[0], m_nCells);
CHECK_POSITIVE(M_NUTRSED[0], m_nSubbsns);
CHECK_POSITIVE(M_NUTRSED[0], m_cellWth);
CHECK_POSITIVE(M_NUTRSED[0], m_maxSoilLyrs);
CHECK_POINTER(M_NUTRSED[0], m_nSoilLyrs);
CHECK_POINTER(M_NUTRSED[0], m_olWtrEroSed);
CHECK_POINTER(M_NUTRSED[0], m_surfRf);
CHECK_POINTER(M_NUTRSED[0], m_soilBD);
CHECK_POINTER(M_NUTRSED[0], m_soilActvMinP);
CHECK_POINTER(M_NUTRSED[0], m_soilStabOrgN);
CHECK_POINTER(M_NUTRSED[0], m_soilHumOrgP);
CHECK_POINTER(M_NUTRSED[0], m_soilStabMinP);
CHECK_POINTER(M_NUTRSED[0], m_soilActvOrgN);
CHECK_POINTER(M_NUTRSED[0], m_soilFrshOrgN);
CHECK_POINTER(M_NUTRSED[0], m_soilFrshOrgP);
CHECK_POINTER(M_NUTRSED[0], m_subbsnID);
CHECK_POINTER(M_NUTRSED[0], m_subbasinsInfo);
if (!(m_cbnModel == 0 || m_cbnModel == 1 || m_cbnModel == 2)) {
throw ModelException(M_NUTRSED[0], "CheckInputData",
"Carbon modeling method must be 0, 1, or 2.");
}
return true;
}

bool NutrientTransportSediment::CheckInputDataCenturyModel() {
CHECK_POINTER(M_NUTRSED[0], m_sol_LSN);
CHECK_POINTER(M_NUTRSED[0], m_sol_LMN);
CHECK_POINTER(M_NUTRSED[0], m_sol_HPN);
CHECK_POINTER(M_NUTRSED[0], m_sol_HSN);
CHECK_POINTER(M_NUTRSED[0], m_sol_HPC);
CHECK_POINTER(M_NUTRSED[0], m_sol_HSC);
CHECK_POINTER(M_NUTRSED[0], m_sol_LMC);
CHECK_POINTER(M_NUTRSED[0], m_sol_LSC);
CHECK_POINTER(M_NUTRSED[0], m_sol_LS);
CHECK_POINTER(M_NUTRSED[0], m_sol_LM);
CHECK_POINTER(M_NUTRSED[0], m_sol_LSL);
CHECK_POINTER(M_NUTRSED[0], m_sol_LSLC);
CHECK_POINTER(M_NUTRSED[0], m_sol_LSLNC);
CHECK_POINTER(M_NUTRSED[0], m_sol_BMC);
CHECK_POINTER(M_NUTRSED[0], m_sol_WOC);
CHECK_POINTER(M_NUTRSED[0], m_soilPerco);
CHECK_POINTER(M_NUTRSED[0], m_subSurfRf);
return true;
}

bool NutrientTransportSediment::CheckInputDataCFarmModel() {
CHECK_POINTER(M_NUTRSED[0], m_soilManP);
return true;
}

void NutrientTransportSediment::SetValue(const char* key, const FLTPT value) {
string sk(key);
if (StringMatch(sk, Tag_CellWidth[0])) m_cellWth = value;
else {
throw ModelException(M_NUTRSED[0], "SetValue",
"Parameter " + sk + " does not exist.");
}
}

void NutrientTransportSediment::SetValue(const char* key, const int value) {
string sk(key);
if (StringMatch(sk, VAR_SUBBSNID_NUM[0])) m_nSubbsns = value;
else if (StringMatch(sk, Tag_SubbasinId)) m_inputSubbsnID = value;
else if (StringMatch(sk, VAR_CSWAT[0])) m_cbnModel = value;
else {
throw ModelException(M_NUTRSED[0], "SetValue",
"Integer Parameter " + sk + " does not exist.");
}
}

void NutrientTransportSediment::Set1DData(const char* key, const int n, FLTPT* data) {
CheckInputSize(M_NUTRSED[0], key, n, m_nCells);
string sk(key);
if (StringMatch(sk, VAR_SEDYLD[0])) {
m_olWtrEroSed = data;
} else if (StringMatch(sk, VAR_OLFLOW[0])) {
m_surfRf = data;
} else {
throw ModelException(M_NUTRSED[0], "Set1DData",
"Parameter " + sk + " does not exist.");
}
}

void NutrientTransportSediment::Set1DData(const char* key, const int n, int* data) {
CheckInputSize(M_NUTRSED[0], key, n, m_nCells);
string sk(key);
if (StringMatch(sk, VAR_SUBBSN[0])) {
m_subbsnID = data;
} else if (StringMatch(sk, VAR_SOILLAYERS[0])) {
m_nSoilLyrs = data;
} else {
throw ModelException(M_NUTRSED[0], "Set1DData",
"Integer Parameter " + sk + " does not exist.");
}
}

void NutrientTransportSediment::Set2DData(const char* key, const int nrows, const int ncols, FLTPT** data) {
CheckInputSize2D(M_NUTRSED[0], key, nrows, ncols, m_nCells, m_maxSoilLyrs);
string sk(key);
if (StringMatch(sk, VAR_SOILTHICK[0])) m_soilThk = data;
else if (StringMatch(sk, VAR_SOL_BD[0])) m_soilBD = data;
else if (StringMatch(sk, VAR_SOL_AORGN[0])) m_soilActvOrgN = data;
else if (StringMatch(sk, VAR_SOL_SORGN[0])) m_soilStabOrgN = data;
else if (StringMatch(sk, VAR_SOL_HORGP[0])) m_soilHumOrgP = data;
else if (StringMatch(sk, VAR_SOL_FORGP[0])) m_soilFrshOrgP = data;
else if (StringMatch(sk, VAR_SOL_FORGN[0])) m_soilFrshOrgN = data;
else if (StringMatch(sk, VAR_SOL_ACTP[0])) m_soilActvMinP = data;
else if (StringMatch(sk, VAR_SOL_STAP[0])) m_soilStabMinP = data;
else if (StringMatch(sk, VAR_ROCK[0])) m_soilRock = data;
else if (StringMatch(sk, VAR_SOL_UL[0])) m_soilSat = data;
else if (StringMatch(sk, VAR_SOL_LSN[0])) m_sol_LSN = data;
else if (StringMatch(sk, VAR_SOL_LMN[0])) m_sol_LMN = data;
else if (StringMatch(sk, VAR_SOL_HPN[0])) m_sol_HPN = data;
else if (StringMatch(sk, VAR_SOL_HSN[0])) m_sol_HSN = data;
else if (StringMatch(sk, VAR_SOL_HPC[0])) m_sol_HPC = data;
else if (StringMatch(sk, VAR_SOL_HSC[0])) m_sol_HSC = data;
else if (StringMatch(sk, VAR_SOL_LMC[0])) m_sol_LMC = data;
else if (StringMatch(sk, VAR_SOL_LSC[0])) m_sol_LSC = data;
else if (StringMatch(sk, VAR_SOL_LS[0])) m_sol_LS = data;
else if (StringMatch(sk, VAR_SOL_LM[0])) m_sol_LM = data;
else if (StringMatch(sk, VAR_SOL_LSL[0])) m_sol_LSL = data;
else if (StringMatch(sk, VAR_SOL_LSLC[0])) m_sol_LSLC = data;
else if (StringMatch(sk, VAR_SOL_LSLNC[0])) m_sol_LSLNC = data;
else if (StringMatch(sk, VAR_SOL_BMC[0])) m_sol_BMC = data;
else if (StringMatch(sk, VAR_SOL_WOC[0])) m_sol_WOC = data;
else if (StringMatch(sk, VAR_PERCO[0])) m_soilPerco = data;
else if (StringMatch(sk, VAR_SSRU[0])) m_subSurfRf = data;
else if (StringMatch(sk, VAR_SOL_MP[0])) m_soilManP = data;
else {
throw ModelException(M_NUTRSED[0], "Set2DData",
"Parameter " + sk + " does not exist.");
}
}

void NutrientTransportSediment::InitialOutputs() {
CHECK_POSITIVE(M_NUTRSED[0], m_nCells);
if (nullptr == m_enratio) {
Initialize1DArray(m_nCells, m_enratio, 0.);
}
if (m_cellArea < 0) {
m_cellArea = m_cellWth * m_cellWth * 0.0001; 
}
if (nullptr == m_surfRfSedOrgN) {
Initialize1DArray(m_nCells, m_surfRfSedOrgN, 0.);
Initialize1DArray(m_nCells, m_surfRfSedOrgP, 0.);
Initialize1DArray(m_nCells, m_surfRfSedAbsorbMinP, 0.);
Initialize1DArray(m_nCells, m_surfRfSedSorbMinP, 0.);

Initialize1DArray(m_nSubbsns + 1, m_surfRfSedOrgNToCh, 0.);
Initialize1DArray(m_nSubbsns + 1, m_surfRfSedOrgPToCh, 0.);
Initialize1DArray(m_nSubbsns + 1, m_surfRfSedAbsorbMinPToCh, 0.);
Initialize1DArray(m_nSubbsns + 1, m_surfRfSedSorbMinPToCh, 0.);
}
if (m_cbnModel == 2 && nullptr == m_soilIfluCbn) {
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilIfluCbn, 0.);
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilPercoCbn, 0.);
Initialize1DArray(m_nCells, m_soilIfluCbnPrfl, 0.);
Initialize1DArray(m_nCells, m_soilPercoCbnPrfl, 0.);
Initialize1DArray(m_nCells, m_sedLossCbn, 0.);
}
}

void NutrientTransportSediment::InitialIntermediates() {
if (!m_reCalIntermediates) return;

if (nullptr == m_soilMass) {
Initialize2DArray(m_nCells, m_maxSoilLyrs, m_soilMass, 0.);
}
#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
for (int k = 0; k < CVT_INT(m_nSoilLyrs[i]); k++) {
m_soilMass[i][k] = 10000. * m_soilThk[i][k] * m_soilBD[i][k] * (1. - m_soilRock[i][k] * 0.01);
}
}
m_reCalIntermediates = false;
}

void NutrientTransportSediment::SetSubbasins(clsSubbasins* subbasins) {
if (nullptr == m_subbasinsInfo) {
m_subbasinsInfo = subbasins;
m_subbasinIDs = m_subbasinsInfo->GetSubbasinIDs();
}
}

int NutrientTransportSediment::Execute() {
CheckInputData();
if (m_cbnModel == 1) {
if (!CheckInputDataCFarmModel()) return false;
}
if (m_cbnModel == 2) {
if (!CheckInputDataCenturyModel()) return false;
}
InitialIntermediates();
InitialOutputs();
for (int i = 0; i < m_nSubbsns + 1; i++) {
m_surfRfSedOrgNToCh[i] = 0.;
m_surfRfSedOrgPToCh[i] = 0.;
m_surfRfSedAbsorbMinPToCh[i] = 0.;
m_surfRfSedSorbMinPToCh[i] = 0.;
}

#pragma omp parallel for
for (int i = 0; i < m_nCells; i++) {
if (m_olWtrEroSed[i] < 1.e-4) m_olWtrEroSed[i] = 0.;
m_enratio[i] = CalEnrichmentRatio(m_olWtrEroSed[i], m_surfRf[i], m_cellArea);

if (m_cbnModel == 0) {
OrgNRemovedInRunoffStaticMethod(i);
} else if (m_cbnModel == 1) {
OrgNRemovedInRunoffCFarmOneCarbonModel(i);
} else if (m_cbnModel == 2) {
OrgNRemovedInRunoffCenturyModel(i);
}
OrgPAttachedtoSed(i);
}
#pragma omp parallel
{
FLTPT* tmp_orgn2ch = new(nothrow) FLTPT[m_nSubbsns + 1];
FLTPT* tmp_orgp2ch = new(nothrow) FLTPT[m_nSubbsns + 1];
FLTPT* tmp_minpa2ch = new(nothrow) FLTPT[m_nSubbsns + 1];
FLTPT* tmp_minps2ch = new(nothrow) FLTPT[m_nSubbsns + 1];
for (int i = 0; i <= m_nSubbsns; i++) {
tmp_orgn2ch[i] = 0.;
tmp_orgp2ch[i] = 0.;
tmp_minpa2ch[i] = 0.;
tmp_minps2ch[i] = 0.;
}
#pragma omp for
for (int i = 0; i < m_nCells; i++) {
tmp_orgn2ch[CVT_INT(m_subbsnID[i])] += m_surfRfSedOrgN[i];
tmp_orgp2ch[CVT_INT(m_subbsnID[i])] += m_surfRfSedOrgP[i];
tmp_minpa2ch[CVT_INT(m_subbsnID[i])] += m_surfRfSedAbsorbMinP[i];
tmp_minps2ch[CVT_INT(m_subbsnID[i])] += m_surfRfSedSorbMinP[i];
}
#pragma omp critical
{
for (int i = 1; i <= m_nSubbsns; i++) {
m_surfRfSedOrgNToCh[i] += tmp_orgn2ch[i];
m_surfRfSedOrgPToCh[i] += tmp_orgp2ch[i];
m_surfRfSedAbsorbMinPToCh[i] += tmp_minpa2ch[i];
m_surfRfSedSorbMinPToCh[i] += tmp_minps2ch[i];
}
}
delete[] tmp_orgn2ch;
delete[] tmp_orgp2ch;
delete[] tmp_minpa2ch;
delete[] tmp_minps2ch;
tmp_orgn2ch = nullptr;
tmp_orgp2ch = nullptr;
tmp_minpa2ch = nullptr;
tmp_minps2ch = nullptr;
} 
for (int i = 1; i < m_nSubbsns + 1; i++) {
m_surfRfSedOrgNToCh[i] *= m_cellArea;
m_surfRfSedOrgPToCh[i] *= m_cellArea;
m_surfRfSedAbsorbMinPToCh[i] *= m_cellArea;
m_surfRfSedSorbMinPToCh[i] *= m_cellArea;
m_surfRfSedOrgNToCh[0] += m_surfRfSedOrgNToCh[i];
m_surfRfSedOrgPToCh[0] += m_surfRfSedOrgPToCh[i];
m_surfRfSedAbsorbMinPToCh[0] += m_surfRfSedAbsorbMinPToCh[i];
m_surfRfSedSorbMinPToCh[0] += m_surfRfSedSorbMinPToCh[i];
}
return 0;
}

void NutrientTransportSediment::OrgNRemovedInRunoffStaticMethod(const int i) {
FLTPT orgninfl = 0.;
FLTPT wt = 0.;
orgninfl = m_soilStabOrgN[i][0] + m_soilActvOrgN[i][0] + m_soilFrshOrgN[i][0];
wt = m_soilBD[i][0] * m_soilThk[i][0] * 0.01;
FLTPT concn = orgninfl * m_enratio[i] / wt;
m_surfRfSedOrgN[i] = 0.001 * concn * m_olWtrEroSed[i] * 0.001 / m_cellArea; 
if (orgninfl > 1.e-6) {
m_soilActvOrgN[i][0] = m_soilActvOrgN[i][0] - m_surfRfSedOrgN[i] * (m_soilActvOrgN[i][0] / orgninfl);
m_soilStabOrgN[i][0] = m_soilStabOrgN[i][0] - m_surfRfSedOrgN[i] * (m_soilStabOrgN[i][0] / orgninfl);
m_soilFrshOrgN[i][0] = m_soilFrshOrgN[i][0] - m_surfRfSedOrgN[i] * (m_soilFrshOrgN[i][0] / orgninfl);
if (m_soilActvOrgN[i][0] < 0.) {
m_surfRfSedOrgN[i] = m_surfRfSedOrgN[i] + m_soilActvOrgN[i][0];
m_soilActvOrgN[i][0] = 0.;
}
if (m_soilStabOrgN[i][0] < 0.) {
m_surfRfSedOrgN[i] = m_surfRfSedOrgN[i] + m_soilStabOrgN[i][0];
m_soilStabOrgN[i][0] = 0.;
}
if (m_soilFrshOrgN[i][0] < 0.) {
m_surfRfSedOrgN[i] = m_surfRfSedOrgN[i] + m_soilFrshOrgN[i][0];
m_soilFrshOrgN[i][0] = 0.;
}
}
}

void NutrientTransportSediment::OrgNRemovedInRunoffCFarmOneCarbonModel(const int i) {
}

void NutrientTransportSediment::OrgNRemovedInRunoffCenturyModel(const int i) {
FLTPT totOrgN_lyr0 = 0.; 
FLTPT wt1 = 0.;          
FLTPT er = 0.;           
FLTPT conc = 0.;         
FLTPT QBC = 0.;          
FLTPT VBC = 0.;          
FLTPT YBC = 0.;          
FLTPT YOC = 0.;          
FLTPT YW = 0.;           
FLTPT TOT = 0.;          
FLTPT YEW = 0.;          
FLTPT X1 = 0.;
FLTPT PRMT_21 = 0.;
FLTPT PRMT_44 = 0.; 
FLTPT XX = 0.;
FLTPT DK = 0.;
FLTPT V = 0.;
FLTPT X3 = 0.;
FLTPT CO = 0.; 
FLTPT CS = 0.; 
FLTPT perc_clyr = 0.;
FLTPT latc_clyr = 0.;

totOrgN_lyr0 = m_sol_LSN[i][0] + m_sol_LMN[i][0] + m_sol_HPN[i][0] + m_sol_HSN[i][0];
wt1 = m_soilBD[i][0] * m_soilThk[i][0] * 0.01;
er = m_enratio[i];
conc = totOrgN_lyr0 * er / wt1;
m_surfRfSedOrgN[i] = 0.001 * conc * m_olWtrEroSed[i] * 0.001 / m_cellArea;
if (totOrgN_lyr0 > UTIL_ZERO) {
FLTPT xx1 = 1. - m_surfRfSedOrgN[i] / totOrgN_lyr0;
m_sol_LSN[i][0] *= xx1;
m_sol_LMN[i][0] *= xx1;
m_sol_HPN[i][0] *= xx1;
m_sol_HSN[i][0] *= xx1;
}
TOT = m_sol_HPC[i][0] + m_sol_HSC[i][0] + m_sol_LMC[i][0] + m_sol_LSC[i][0];
YEW = Min((m_olWtrEroSed[i] / m_cellArea + YW / m_cellArea) / m_soilMass[i][0], 0.9);

X1 = 1. - YEW;
YOC = YEW * TOT;
m_sol_HSC[i][0] *= X1;
m_sol_HPC[i][0] *= X1;
m_sol_LS[i][0] *= X1;
m_sol_LM[i][0] *= X1;
m_sol_LSL[i][0] *= X1;
m_sol_LSC[i][0] *= X1;
m_sol_LMC[i][0] *= X1;
m_sol_LSLC[i][0] *= X1;
m_sol_LSLNC[i][0] = m_sol_LSC[i][0] - m_sol_LSLC[i][0];

if (m_sol_BMC[i][0] > 0.01) {
PRMT_21 = 1000.;
m_sol_WOC[i][0] = m_sol_LSC[i][0] + m_sol_LMC[i][0] + m_sol_HPC[i][0] + m_sol_HSC[i][0] + m_sol_BMC[i][0];
DK = 0.0001 * PRMT_21 * m_sol_WOC[i][0];
X1 = m_soilSat[i][0];
if (X1 <= 0.) X1 = 0.01;
XX = X1 + DK;
V = m_surfRf[i] + m_soilPerco[i][0] + m_subSurfRf[i][0];
if (V > 1.e-10) {
X3 = m_sol_BMC[i][0] * (1. - CalExp(-V / XX)); 
PRMT_44 = 0.5;
CO = X3 / (m_soilPerco[i][0] + PRMT_44 * (m_surfRf[i] + m_subSurfRf[i][0]));
CS = PRMT_44 * CO;
VBC = CO * m_soilPerco[i][0];
m_sol_BMC[i][0] -= X3;
QBC = CS * (m_surfRf[i] + m_subSurfRf[i][0]);
if (YEW > 0.) {
CS = DK * m_sol_BMC[i][0] / XX;
YBC = YEW * CS;
}
}
}
m_sol_BMC[i][0] -= YBC;
m_soilIfluCbn[i][0] = QBC * (m_subSurfRf[i][0] / (m_surfRf[i] + m_subSurfRf[i][0] + UTIL_ZERO));
m_soilPercoCbn[i][0] = VBC;
m_sedLossCbn[i] = YOC + YBC;

latc_clyr += m_soilIfluCbn[i][0];
for (int k = 1; k < CVT_INT(m_nSoilLyrs[i]); k++) {
m_sol_WOC[i][k] = m_sol_LSC[i][k] + m_sol_LMC[i][k] + m_sol_HPC[i][k] + m_sol_HSC[i][k];
FLTPT Y1 = m_sol_BMC[i][k] + VBC;
VBC = 0.;
if (Y1 > 0.01) {
V = m_soilPerco[i][k] + m_subSurfRf[i][k];
if (V > 0.) {
VBC = Y1 * (1. - CalExp(-V / (m_soilSat[i][k] + 0.0001 * PRMT_21 * m_sol_WOC[i][k])));
}
}
m_soilIfluCbn[i][k] = VBC * (m_subSurfRf[i][k] / (m_subSurfRf[i][k] + m_soilPerco[i][k] + UTIL_ZERO));
m_soilPercoCbn[i][k] = VBC - m_soilIfluCbn[i][k];
m_sol_BMC[i][k] = Y1 - VBC;

perc_clyr += m_soilPercoCbn[i][k];
latc_clyr += m_soilIfluCbn[i][k];
}
m_soilIfluCbnPrfl[i] = latc_clyr;
m_soilPercoCbnPrfl[i] = perc_clyr;
}

void NutrientTransportSediment::OrgPAttachedtoSed(const int i) {
FLTPT sol_attp = 0.;
FLTPT sol_attp_o = 0.;
FLTPT sol_attp_a = 0.;
FLTPT sol_attp_s = 0.;
sol_attp = m_soilHumOrgP[i][0] + m_soilFrshOrgP[i][0] + m_soilActvMinP[i][0] + m_soilStabMinP[i][0];
if (m_soilManP != nullptr) {
sol_attp += m_soilManP[i][0];
}
if (sol_attp > 1.e-3) {
sol_attp_o = (m_soilHumOrgP[i][0] + m_soilFrshOrgP[i][0]) / sol_attp;
if (m_soilManP != nullptr) {
sol_attp_o += m_soilManP[i][0] / sol_attp;
}
sol_attp_a = m_soilActvMinP[i][0] / sol_attp;
sol_attp_s = m_soilStabMinP[i][0] / sol_attp;
}
FLTPT wt = m_soilBD[i][0] * m_soilThk[i][0] * 0.01;
FLTPT concp = 0.;
concp = sol_attp * m_enratio[i] / wt; 
FLTPT sedp = 1.e-6 * concp * m_olWtrEroSed[i] / m_cellArea; 
m_surfRfSedOrgP[i] = sedp * sol_attp_o;
m_surfRfSedAbsorbMinP[i] = sedp * sol_attp_a;
m_surfRfSedSorbMinP[i] = sedp * sol_attp_s;



FLTPT porgg = 0.;
porgg = m_soilHumOrgP[i][0] + m_soilFrshOrgP[i][0];
if (porgg > 1.e-3) {
m_soilHumOrgP[i][0] = m_soilHumOrgP[i][0] - m_surfRfSedOrgP[i] * (m_soilHumOrgP[i][0] / porgg);
m_soilFrshOrgP[i][0] = m_soilFrshOrgP[i][0] - m_surfRfSedOrgP[i] * (m_soilFrshOrgP[i][0] / porgg);
}
m_soilActvMinP[i][0] = m_soilActvMinP[i][0] - m_surfRfSedAbsorbMinP[i];
m_soilStabMinP[i][0] = m_soilStabMinP[i][0] - m_surfRfSedSorbMinP[i];
if (m_soilHumOrgP[i][0] < 0.) {
m_surfRfSedOrgP[i] = m_surfRfSedOrgP[i] + m_soilHumOrgP[i][0];
m_soilHumOrgP[i][0] = 0.;
}
if (m_soilFrshOrgP[i][0] < 0.) {
m_surfRfSedOrgP[i] = m_surfRfSedOrgP[i] + m_soilFrshOrgP[i][0];
m_soilFrshOrgP[i][0] = 0.;
}
if (m_soilActvMinP[i][0] < 0.) {
m_surfRfSedAbsorbMinP[i] = m_surfRfSedAbsorbMinP[i] + m_soilActvMinP[i][0];
m_soilActvMinP[i][0] = 0.;
}
if (m_soilStabMinP[i][0] < 0.) {
m_surfRfSedSorbMinP[i] = m_surfRfSedSorbMinP[i] + m_soilStabMinP[i][0];
m_soilStabMinP[i][0] = 0.;
}
}

void NutrientTransportSediment::Get1DData(const char* key, int* n, FLTPT** data) {
InitialOutputs();
string sk(key);
if (StringMatch(sk, VAR_SEDORGN[0])) {
*data = m_surfRfSedOrgN;
*n = m_nCells;
} else if (StringMatch(sk, VAR_SEDORGP[0])) {
*data = m_surfRfSedOrgP;
*n = m_nCells;
} else if (StringMatch(sk, VAR_SEDMINPA[0])) {
*data = m_surfRfSedAbsorbMinP;
*n = m_nCells;
} else if (StringMatch(sk, VAR_SEDMINPS[0])) {
*data = m_surfRfSedSorbMinP;
*n = m_nCells;
} else if (StringMatch(sk, VAR_SEDORGN_TOCH[0])) {
*data = m_surfRfSedOrgNToCh;
*n = m_nSubbsns + 1;
} else if (StringMatch(sk, VAR_SEDORGP_TOCH[0])) {
*data = m_surfRfSedOrgPToCh;
*n = m_nSubbsns + 1;
} else if (StringMatch(sk, VAR_SEDMINPA_TOCH[0])) {
*data = m_surfRfSedAbsorbMinPToCh;
*n = m_nSubbsns + 1;
} else if (StringMatch(sk, VAR_SEDMINPS_TOCH[0])) {
*data = m_surfRfSedSorbMinPToCh;
*n = m_nSubbsns + 1;
}
else if (StringMatch(sk, VAR_LATERAL_C[0])) {
*data = m_soilIfluCbnPrfl;
*n = m_nCells;
} else if (StringMatch(sk, VAR_PERCO_C[0])) {
*data = m_soilPercoCbnPrfl;
*n = m_nCells;
} else if (StringMatch(sk, VAR_SEDLOSS_C[0])) {
*data = m_sedLossCbn;
*n = m_nCells;
} else {
throw ModelException(M_NUTRSED[0], "Get1DData",
"Parameter " + sk + " does not exist");
}
}

void NutrientTransportSediment::Get2DData(const char* key, int* nrows, int* ncols, FLTPT*** data) {
InitialOutputs();
string sk(key);
*nrows = m_nCells;
*ncols = m_maxSoilLyrs;
if (StringMatch(sk, VAR_SOL_AORGN[0])) *data = m_soilActvOrgN;
else if (StringMatch(sk, VAR_SOL_FORGN[0])) *data = m_soilFrshOrgN;
else if (StringMatch(sk, VAR_SOL_SORGN[0])) *data = m_soilStabOrgN;
else if (StringMatch(sk, VAR_SOL_HORGP[0])) *data = m_soilHumOrgP;
else if (StringMatch(sk, VAR_SOL_FORGP[0])) *data = m_soilFrshOrgP;
else if (StringMatch(sk, VAR_SOL_STAP[0])) *data = m_soilStabMinP;
else if (StringMatch(sk, VAR_SOL_ACTP[0])) *data = m_soilActvMinP;
else if (StringMatch(sk, VAR_SOL_LATERAL_C[0])) *data = m_soilIfluCbn;
else if (StringMatch(sk, VAR_SOL_PERCO_C[0])) *data = m_soilPercoCbn;
else {
throw ModelException(M_NUTRSED[0], "Get2DData",
"Output " + sk + " does not exist.");
}
}
