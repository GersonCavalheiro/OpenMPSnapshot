#include "SEDR_SBAGNOLD.h"

#include "text.h"
#include "ChannelRoutingCommon.h"


SEDR_SBAGNOLD::SEDR_SBAGNOLD() :
m_dt(-1), m_inputSubbsnID(-1),
m_nreach(-1), m_vcd(false), m_peakRateAdj(NODATA_VALUE),
m_sedTransEqCoef(NODATA_VALUE), m_sedTransEqExp(NODATA_VALUE), m_initChSto(NODATA_VALUE),
m_reachDownStream(nullptr), m_chOrder(nullptr), m_chWth(nullptr),
m_chDepth(nullptr), m_chWthDepthRt(nullptr), m_chLen(nullptr),
m_chSlope(nullptr), m_chSideSlope(nullptr),
m_chBnkBD(nullptr), m_chBedBD(nullptr), m_chBnkCov(nullptr),
m_chBnkErod(nullptr), m_chBedErod(nullptr), m_chBnkTc(nullptr), m_chBedTc(nullptr),
m_chBnkSand(nullptr), m_chBnkSilt(nullptr), m_chBnkClay(nullptr), m_chBnkGravel(nullptr),
m_chBedSand(nullptr), m_chBedSilt(nullptr), m_chBedClay(nullptr), m_chBedGravel(nullptr),
m_ptSub(nullptr),
m_initChDepth(nullptr), m_initChLen(nullptr), m_initChSlope(nullptr),
m_preRchDep(nullptr), m_preFldplnDep(nullptr),
m_sedtoCh(nullptr), m_sandtoCh(nullptr), m_silttoCh(nullptr), m_claytoCh(nullptr),
m_sagtoCh(nullptr), m_lagtoCh(nullptr), m_graveltoCh(nullptr),
m_qRchOut(nullptr), m_chSto(nullptr), m_rteWtrOut(nullptr),
m_chBtmWth(nullptr), m_chWtrDepth(nullptr), m_chWtrWth(nullptr),
m_sedRchOut(nullptr), m_sedConcRchOut(nullptr), m_sandRchOut(nullptr), m_siltRchOut(nullptr),
m_clayRchOut(nullptr), m_sagRchOut(nullptr), m_lagRchOut(nullptr), m_gravelRchOut(nullptr),
m_rchBnkEro(nullptr), m_rchDeg(nullptr),
m_rchDep(nullptr), m_dltRchDep(nullptr),
m_rchDepSand(nullptr), m_rchDepSilt(nullptr), m_rchDepClay(nullptr),
m_rchDepSag(nullptr), m_rchDepLag(nullptr), m_rchDepGravel(nullptr),
m_fldplnDep(nullptr), m_dltFldplnDep(nullptr), m_fldplnDepSilt(nullptr), m_fldplnDepClay(nullptr),
m_sedSto(nullptr), m_sandSto(nullptr), m_siltSto(nullptr), m_claySto(nullptr),
m_sagSto(nullptr), m_lagSto(nullptr), m_gravelSto(nullptr) {
}

SEDR_SBAGNOLD::~SEDR_SBAGNOLD() {

if (nullptr != m_ptSub) Release1DArray(m_ptSub);

if (nullptr != m_initChDepth) Release1DArray(m_initChDepth);
if (nullptr != m_initChLen) Release1DArray(m_initChLen);
if (nullptr != m_initChSlope) Release1DArray(m_initChSlope);

if (nullptr != m_sedRchOut) Release1DArray(m_sedRchOut);
if (nullptr != m_sedConcRchOut) Release1DArray(m_sedConcRchOut);
if (nullptr != m_sandRchOut) Release1DArray(m_sandRchOut);
if (nullptr != m_siltRchOut) Release1DArray(m_siltRchOut);
if (nullptr != m_clayRchOut) Release1DArray(m_clayRchOut);
if (nullptr != m_sagRchOut) Release1DArray(m_sagRchOut);
if (nullptr != m_lagRchOut) Release1DArray(m_lagRchOut);
if (nullptr != m_gravelRchOut) Release1DArray(m_gravelRchOut);

if (nullptr != m_rchBnkEro) Release1DArray(m_rchBnkEro);
if (nullptr != m_rchDeg) Release1DArray(m_rchDeg);

if (nullptr != m_rchDep) Release1DArray(m_rchDep);
if (nullptr != m_preRchDep) Release1DArray(m_preRchDep);
if (nullptr != m_dltRchDep) Release1DArray(m_dltRchDep);
if (nullptr != m_rchDepSand) Release1DArray(m_rchDepSand);
if (nullptr != m_rchDepSilt) Release1DArray(m_rchDepSilt);
if (nullptr != m_rchDepClay) Release1DArray(m_rchDepClay);
if (nullptr != m_rchDepSag) Release1DArray(m_rchDepSag);
if (nullptr != m_rchDepLag) Release1DArray(m_rchDepLag);
if (nullptr != m_rchDepGravel) Release1DArray(m_rchDepGravel);

if (nullptr != m_fldplnDep) Release1DArray(m_fldplnDep);
if (nullptr != m_preFldplnDep) Release1DArray(m_preFldplnDep);
if (nullptr != m_dltFldplnDep) Release1DArray(m_dltFldplnDep);
if (nullptr != m_fldplnDepSilt) Release1DArray(m_fldplnDepSilt);
if (nullptr != m_fldplnDepClay) Release1DArray(m_fldplnDepClay);

if (nullptr != m_sedSto) Release1DArray(m_sedSto);
if (nullptr != m_sandSto) Release1DArray(m_sandSto);
if (nullptr != m_siltSto) Release1DArray(m_siltSto);
if (nullptr != m_claySto) Release1DArray(m_claySto);
if (nullptr != m_sagSto) Release1DArray(m_sagSto);
if (nullptr != m_lagSto) Release1DArray(m_lagSto);
if (nullptr != m_gravelSto) Release1DArray(m_gravelSto);
}

bool SEDR_SBAGNOLD::CheckInputData() {
CHECK_POSITIVE(M_SEDR_SBAGNOLD[0], m_dt);
CHECK_POSITIVE(M_SEDR_SBAGNOLD[0], m_nreach);
CHECK_NONNEGATIVE(M_SEDR_SBAGNOLD[0], m_inputSubbsnID);

CHECK_NODATA(M_SEDR_SBAGNOLD[0], m_peakRateAdj);
CHECK_NODATA(M_SEDR_SBAGNOLD[0], m_sedTransEqCoef);
CHECK_NODATA(M_SEDR_SBAGNOLD[0], m_sedTransEqExp);
CHECK_NODATA(M_SEDR_SBAGNOLD[0], m_initChSto);

CHECK_POINTER(M_SEDR_SBAGNOLD[0], m_sedtoCh);
CHECK_POINTER(M_SEDR_SBAGNOLD[0], m_qRchOut);
CHECK_POINTER(M_SEDR_SBAGNOLD[0], m_chSto);
CHECK_POINTER(M_SEDR_SBAGNOLD[0], m_rteWtrOut);
CHECK_POINTER(M_SEDR_SBAGNOLD[0], m_chWtrDepth);
CHECK_POINTER(M_SEDR_SBAGNOLD[0], m_chWtrWth);
return true;
}

void SEDR_SBAGNOLD::InitialOutputs() {
if (m_sedRchOut != nullptr) return;
CHECK_POSITIVE(M_SEDR_SBAGNOLD[0], m_nreach);
Initialize1DArray(m_nreach + 1, m_sedRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_sedConcRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_sandRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_siltRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_clayRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_sagRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_lagRchOut, 0.);
Initialize1DArray(m_nreach + 1, m_gravelRchOut, 0.);

Initialize1DArray(m_nreach + 1, m_rchBnkEro, 0.);
Initialize1DArray(m_nreach + 1, m_rchDeg, 0.);

Initialize1DArray(m_nreach + 1, m_rchDep, 0.);
Initialize1DArray(m_nreach + 1, m_preRchDep, 0.);
Initialize1DArray(m_nreach + 1, m_dltRchDep, 0.);
Initialize1DArray(m_nreach + 1, m_rchDepSand, 0.);
Initialize1DArray(m_nreach + 1, m_rchDepSilt, 0.);
Initialize1DArray(m_nreach + 1, m_rchDepClay, 0.);
Initialize1DArray(m_nreach + 1, m_rchDepSag, 0.);
Initialize1DArray(m_nreach + 1, m_rchDepLag, 0.);
Initialize1DArray(m_nreach + 1, m_rchDepGravel, 0.);

Initialize1DArray(m_nreach + 1, m_fldplnDep, 0.);
Initialize1DArray(m_nreach + 1, m_preFldplnDep, 0.);
Initialize1DArray(m_nreach + 1, m_dltFldplnDep, 0.);
Initialize1DArray(m_nreach + 1, m_fldplnDepSilt, 0.);
Initialize1DArray(m_nreach + 1, m_fldplnDepClay, 0.);

Initialize1DArray(m_nreach + 1, m_sedSto, 0.);
Initialize1DArray(m_nreach + 1, m_sandSto, 0.);
Initialize1DArray(m_nreach + 1, m_siltSto, 0.);
Initialize1DArray(m_nreach + 1, m_claySto, 0.);
Initialize1DArray(m_nreach + 1, m_sagSto, 0.);
Initialize1DArray(m_nreach + 1, m_lagSto, 0.);
Initialize1DArray(m_nreach + 1, m_gravelSto, 0.);
if (nullptr == m_ptSub) Initialize1DArray(m_nreach + 1, m_ptSub, 0.);
}

void SEDR_SBAGNOLD::PointSourceLoading() {
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
FLTPT per_sed = curPtMgt->GetSedment(); 
for (auto locIter = ptSrcIDs.begin(); locIter != ptSrcIDs.end(); ++locIter) {
if (pointSrcLocsMap.find(*locIter) != pointSrcLocsMap.end()) {
PointSourceLocations* curPtLoc = pointSrcLocsMap.at(*locIter);
int curSubID = curPtLoc->GetSubbasinID();
m_ptSub[curSubID] += per_sed * curPtLoc->GetSize() * 1000. * m_dt * 1.1574074074074073e-05;
}
}
}
}
}

int SEDR_SBAGNOLD::Execute() {
CheckInputData();
InitialOutputs();
PointSourceLoading();
for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); ++it) {
int nReaches = CVT_INT(it->second.size());
#pragma omp parallel for
for (int i = 0; i < nReaches; i++) {
int reachIndex = it->second[i]; 
if (m_inputSubbsnID == 0 || m_inputSubbsnID == reachIndex) {
SedChannelRouting(reachIndex);
ChannelDowncuttingWidening(reachIndex);
}
}
}
return 0;
}

void SEDR_SBAGNOLD::GetValue(const char* key, FLTPT* value) {
InitialOutputs();
string sk(key);
if (StringMatch(sk, VAR_SED_RECH[0])) *value = m_sedRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_SED_RECHConc[0])) *value = m_sedConcRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_SAND_RECH[0])) *value = m_sandRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_SILT_RECH[0])) *value = m_siltRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_CLAY_RECH[0])) *value = m_clayRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_SAG_RECH[0])) *value = m_sagRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_LAG_RECH[0])) *value = m_lagRchOut[m_inputSubbsnID];
else if (StringMatch(sk, VAR_GRAVEL_RECH[0])) *value = m_gravelRchOut[m_inputSubbsnID];
else {
throw ModelException(M_SEDR_SBAGNOLD[0], "GetValue",
"Parameter " + sk + " does not exist.");
}
}

void SEDR_SBAGNOLD::SetValue(const char* key, const FLTPT value) {
string sk(key);
if (StringMatch(sk, VAR_P_RF[0])) m_peakRateAdj = value;
else if (StringMatch(sk, VAR_SPCON[0])) m_sedTransEqCoef = value;
else if (StringMatch(sk, VAR_SPEXP[0])) m_sedTransEqExp = value;
else if (StringMatch(sk, VAR_CHS0[0])) m_initChSto = value;
else {
throw ModelException(M_SEDR_SBAGNOLD[0], "SetValue",
"Parameter " + sk + " does not exist.");
}
}

void SEDR_SBAGNOLD::SetValue(const char* key, const int value) {
string sk(key);
if (StringMatch(sk, Tag_SubbasinId)) m_inputSubbsnID = value;
#ifdef STORM_MODE
else if (StringMatch(sk, Tag_ChannelTimeStep)) m_dt = value;
#else
else if (StringMatch(sk, Tag_TimeStep[0])) m_dt = value;
#endif 
else if (StringMatch(sk, VAR_VCD[0])) m_vcd = value == 1;
else {
throw ModelException(M_SEDR_SBAGNOLD[0], "SetValue",
"Integer Parameter " + sk + " does not exist.");
}
}

void SEDR_SBAGNOLD::SetValueByIndex(const char* key, const int index, const FLTPT value) {
if (m_inputSubbsnID == 0) return;           
if (index <= 0 || index > m_nreach) return; 
if (nullptr == m_sedRchOut) InitialOutputs();
string sk(key);
if (StringMatch(sk, VAR_SED_RECH[0])) m_sedRchOut[index] = value;
else if (StringMatch(sk, VAR_SED_RECHConc[0])) m_sedConcRchOut[index] = value;
else if (StringMatch(sk, VAR_SAND_RECH[0])) m_sandRchOut[index] = value;
else if (StringMatch(sk, VAR_SILT_RECH[0])) m_siltRchOut[index] = value;
else if (StringMatch(sk, VAR_CLAY_RECH[0])) m_clayRchOut[index] = value;
else if (StringMatch(sk, VAR_SAG_RECH[0])) m_sagRchOut[index] = value;
else if (StringMatch(sk, VAR_LAG_RECH[0])) m_lagRchOut[index] = value;
else if (StringMatch(sk, VAR_GRAVEL_RECH[0])) m_gravelRchOut[index] = value;
else {
throw ModelException(M_SEDR_SBAGNOLD[0], "SetValueByIndex",
"Parameter " + sk + " does not exist");
}
}

void SEDR_SBAGNOLD::Set1DData(const char* key, const int n, FLTPT* data) {
string sk(key);
CheckInputSize(M_SEDR_SBAGNOLD[0], key, n - 1, m_nreach);
if (StringMatch(sk, VAR_SED_TO_CH[0])) m_sedtoCh = data; 
else if (StringMatch(sk, VAR_SAND_TO_CH[0])) m_sandtoCh = data;
else if (StringMatch(sk, VAR_SILT_TO_CH[0])) m_silttoCh = data;
else if (StringMatch(sk, VAR_CLAY_TO_CH[0])) m_claytoCh = data;
else if (StringMatch(sk, VAR_SAG_TO_CH[0])) m_sagtoCh = data;
else if (StringMatch(sk, VAR_LAG_TO_CH[0])) m_lagtoCh = data;
else if (StringMatch(sk, VAR_GRAVEL_TO_CH[0])) m_graveltoCh = data;

else if (StringMatch(sk, VAR_QRECH[0])) m_qRchOut = data;
else if (StringMatch(sk, VAR_CHST[0])) m_chSto = data;
else if (StringMatch(sk, VAR_RTE_WTROUT[0])) m_rteWtrOut = data;
else if (StringMatch(sk, VAR_CHBTMWIDTH[0])) m_chBtmWth = data;
else if (StringMatch(sk, VAR_CHWTRDEPTH[0])) m_chWtrDepth = data;
else if (StringMatch(sk, VAR_CHWTRWIDTH[0])) m_chWtrWth = data;
else {
throw ModelException(M_SEDR_SBAGNOLD[0], "Set1DData",
"Parameter " + sk + " does not exist");
}
}

void SEDR_SBAGNOLD::Get1DData(const char* key, int* n, FLTPT** data) {
InitialOutputs();
string sk(key);
*n = m_nreach + 1;
if (StringMatch(sk, VAR_SED_RECH[0])) *data = m_sedRchOut;
else if (StringMatch(sk, VAR_SED_RECHConc[0])) *data = m_sedConcRchOut;
else if (StringMatch(sk, VAR_SAND_RECH[0])) *data = m_sandRchOut;
else if (StringMatch(sk, VAR_SILT_RECH[0])) *data = m_siltRchOut;
else if (StringMatch(sk, VAR_CLAY_RECH[0])) *data = m_clayRchOut;
else if (StringMatch(sk, VAR_SAG_RECH[0])) *data = m_sagRchOut;
else if (StringMatch(sk, VAR_LAG_RECH[0])) *data = m_lagRchOut;
else if (StringMatch(sk, VAR_GRAVEL_RECH[0])) *data = m_gravelRchOut;

else if (StringMatch(sk, VAR_RCH_BANKERO[0])) *data = m_rchBnkEro;
else if (StringMatch(sk, VAR_RCH_DEG[0])) *data = m_rchDeg;

else if (StringMatch(sk, VAR_RCH_DEP[0])) *data = m_rchDep;
else if (StringMatch(sk, VAR_RCH_DEPNEW[0])) *data = m_dltRchDep;
else if (StringMatch(sk, VAR_RCH_DEPSAND[0])) *data = m_rchDepSand;
else if (StringMatch(sk, VAR_RCH_DEPSILT[0])) *data = m_rchDepSilt;
else if (StringMatch(sk, VAR_RCH_DEPCLAY[0])) *data = m_rchDepClay;
else if (StringMatch(sk, VAR_RCH_DEPSAG[0])) *data = m_rchDepSag;
else if (StringMatch(sk, VAR_RCH_DEPLAG[0])) *data = m_rchDepLag;
else if (StringMatch(sk, VAR_RCH_DEPGRAVEL[0])) *data = m_rchDepGravel;

else if (StringMatch(sk, VAR_FLDPLN_DEP[0])) *data = m_fldplnDep;
else if (StringMatch(sk, VAR_FLDPLN_DEPNEW[0])) *data = m_dltFldplnDep;
else if (StringMatch(sk, VAR_FLDPLN_DEPSILT[0])) *data = m_fldplnDepSilt;
else if (StringMatch(sk, VAR_FLDPLN_DEPCLAY[0])) *data = m_fldplnDepClay;

else if (StringMatch(sk, VAR_SEDSTO_CH[0])) *data = m_sedSto;
else if (StringMatch(sk, VAR_SANDSTO_CH[0])) *data = m_sandSto;
else if (StringMatch(sk, VAR_SILTSTO_CH[0])) *data = m_siltSto;
else if (StringMatch(sk, VAR_CLAYSTO_CH[0])) *data = m_claySto;
else if (StringMatch(sk, VAR_SAGSTO_CH[0])) *data = m_sagSto;
else if (StringMatch(sk, VAR_LAGSTO_CH[0])) *data = m_lagSto;
else if (StringMatch(sk, VAR_GRAVELSTO_CH[0])) *data = m_gravelSto;
else {
throw ModelException(M_SEDR_SBAGNOLD[0], "Get1DData",
"Output " + sk + " does not exist.");
}
}

void SEDR_SBAGNOLD::SetScenario(Scenario* sce) {
if (nullptr == sce) {
throw ModelException(M_SEDR_SBAGNOLD[0], "SetScenario",
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

void SEDR_SBAGNOLD::SetReaches(clsReaches* reaches) {
if (nullptr == reaches) {
throw ModelException(M_SEDR_SBAGNOLD[0], "SetReaches",
"The reaches input can not to be NULL.");
}
m_nreach = reaches->GetReachNumber();

if (nullptr == m_reachDownStream) {
FLTPT* tmp = nullptr;
reaches->GetReachesSingleProperty(REACH_DOWNSTREAM, &tmp);
Initialize1DArray(m_nreach + 1, m_reachDownStream, tmp);
Release1DArray(tmp);
}
if (nullptr == m_chOrder) {
FLTPT* tmp = nullptr;
reaches->GetReachesSingleProperty(REACH_UPDOWN_ORDER, &tmp);
Initialize1DArray(m_nreach + 1, m_chOrder, tmp);
Release1DArray(tmp);
}
if (nullptr == m_chWth) reaches->GetReachesSingleProperty(REACH_WIDTH, &m_chWth);
if (nullptr == m_chDepth) reaches->GetReachesSingleProperty(REACH_DEPTH, &m_chDepth);
if (nullptr == m_chWthDepthRt) reaches->GetReachesSingleProperty(REACH_WDRATIO, &m_chWthDepthRt);
if (nullptr == m_chLen) reaches->GetReachesSingleProperty(REACH_LENGTH, &m_chLen);
if (nullptr == m_chSlope) reaches->GetReachesSingleProperty(REACH_SLOPE, &m_chSlope);
if (nullptr == m_chSideSlope) reaches->GetReachesSingleProperty(REACH_SIDESLP, &m_chSideSlope);
if (nullptr == m_chBnkBD) reaches->GetReachesSingleProperty(REACH_BNKBD, &m_chBnkBD);
if (nullptr == m_chBedBD) reaches->GetReachesSingleProperty(REACH_BEDBD, &m_chBedBD);
if (nullptr == m_chBnkCov) reaches->GetReachesSingleProperty(REACH_BNKCOV, &m_chBnkCov);
if (nullptr == m_chBnkErod) reaches->GetReachesSingleProperty(REACH_BNKEROD, &m_chBnkErod);
if (nullptr == m_chBedErod) reaches->GetReachesSingleProperty(REACH_BEDEROD, &m_chBedErod);
if (nullptr == m_chBnkTc) reaches->GetReachesSingleProperty(REACH_BNKTC, &m_chBnkTc);
if (nullptr == m_chBedTc) reaches->GetReachesSingleProperty(REACH_BEDTC, &m_chBedTc);
if (nullptr == m_chBnkSand) reaches->GetReachesSingleProperty(REACH_BNKSAND, &m_chBnkSand);
if (nullptr == m_chBnkSilt) reaches->GetReachesSingleProperty(REACH_BNKSILT, &m_chBnkSilt);
if (nullptr == m_chBnkClay) reaches->GetReachesSingleProperty(REACH_BNKCLAY, &m_chBnkClay);
if (nullptr == m_chBnkGravel) reaches->GetReachesSingleProperty(REACH_BNKGRAVEL, &m_chBnkGravel);
if (nullptr == m_chBedSand) reaches->GetReachesSingleProperty(REACH_BEDSAND, &m_chBedSand);
if (nullptr == m_chBedSilt) reaches->GetReachesSingleProperty(REACH_BEDSILT, &m_chBedSilt);
if (nullptr == m_chBedClay) reaches->GetReachesSingleProperty(REACH_BEDCLAY, &m_chBedClay);
if (nullptr == m_chBedGravel) reaches->GetReachesSingleProperty(REACH_BEDGRAVEL, &m_chBedGravel);

m_reachUpStream = reaches->GetUpStreamIDs();
m_reachLayers = reaches->GetReachLayers();
}

void SEDR_SBAGNOLD::SedChannelRouting(const int i) {
m_sedRchOut[i] = 0.;
m_sedConcRchOut[i] = 0.;
m_sandRchOut[i] = 0.;
m_siltRchOut[i] = 0.;
m_clayRchOut[i] = 0.;
m_sagRchOut[i] = 0.;
m_lagRchOut[i] = 0.;
m_gravelRchOut[i] = 0.;

m_rchBnkEro[i] = 0.;
m_rchDeg[i] = 0.;

m_rchDep[i] = 0.;
m_dltRchDep[i] = 0.;
m_rchDepSand[i] = 0.;
m_rchDepSilt[i] = 0.;
m_rchDepClay[i] = 0.;
m_rchDepSag[i] = 0.;
m_rchDepLag[i] = 0.;
m_rchDepGravel[i] = 0.;

m_fldplnDep[i] = 0.;
m_dltFldplnDep[i] = 0.;
m_fldplnDepSilt[i] = 0.;
m_fldplnDepClay[i] = 0.;

if (m_rteWtrOut[i] <= UTIL_ZERO || m_chWtrDepth[i] <= UTIL_ZERO) {
return;
}

FLTPT qdin = m_chSto[i] + m_rteWtrOut[i]; 

FLTPT sedin = 0.;    
FLTPT sandin = 0.;   
FLTPT siltin = 0.;   
FLTPT clayin = 0.;   
FLTPT sagin = 0.;    
FLTPT lagin = 0.;    
FLTPT gravelin = 0.; 
for (auto upRchID = m_reachUpStream.at(i).begin(); upRchID != m_reachUpStream.at(i).end(); ++upRchID) {
sedin += m_sedRchOut[*upRchID];
sandin += m_sandRchOut[*upRchID];
siltin += m_siltRchOut[*upRchID];
clayin += m_clayRchOut[*upRchID];
sagin += m_sagRchOut[*upRchID];
lagin += m_lagRchOut[*upRchID];
gravelin += m_gravelRchOut[*upRchID];
}
sedin += m_sedtoCh[i] + m_sedSto[i];
if (m_sandtoCh != nullptr) sandin += m_sandtoCh[i] + m_sandSto[i];
if (m_silttoCh != nullptr) siltin += m_silttoCh[i] + m_siltSto[i];
if (m_claytoCh != nullptr) clayin += m_claytoCh[i] + m_claySto[i];
if (m_sagtoCh != nullptr) sagin += m_sagtoCh[i] + m_sagSto[i];
if (m_lagtoCh != nullptr) lagin += m_lagtoCh[i] + m_lagSto[i];
if (m_graveltoCh != nullptr) gravelin += m_graveltoCh[i] + m_gravelSto[i];
if (nullptr != m_ptSub && m_ptSub[i] > 0.) {
sedin += m_ptSub[i];
}

if (qdin <= 0.01) {
m_sedSto[i] = sedin;
m_sandSto[i] = sandin;
m_siltSto[i] = siltin;
m_claySto[i] = clayin;
m_sagSto[i] = sagin;
m_lagSto[i] = lagin;
m_gravelSto[i] = gravelin;
return;
}

FLTPT cross_area = ChannelCrossSectionalArea(m_chBtmWth[i], m_chDepth[i], m_chWtrDepth[i],
m_chSideSlope[i], m_chWth[i], 4.);
FLTPT peakRfRate = m_qRchOut[i] * m_peakRateAdj; 
FLTPT peakVel = 0.;                             
if (cross_area < 0.01) {
peakVel = 0.01;
} else {
peakVel = peakRfRate / cross_area;
}
if (peakVel > 5.) {
peakVel = 5.;
}
FLTPT tbase = m_chLen[i] / (m_dt * peakVel); 
if (tbase > 1.) tbase = 1.;
#ifdef PRINT_DEBUG
cout << "ID: " << i << ", qchOut: " << m_qRchOut[i] << ", qdin: " << qdin <<
", chStorage: " << m_chSto[i] <<
", sedtoCh: " << m_sedtoCh[i] <<
", sedStorage: " << m_sedSto[i] << ", allSediment: " << sedin <<
", chLen: " << m_chLen[i] << ", peakVelocity: " << peakVel << ", tbase: " << tbase;
#endif

FLTPT cyin = 0.;      
FLTPT cych = 0.;      
FLTPT sedDepNet = 0.; 

FLTPT sedDeg1 = 0.;       
FLTPT sedDeg1Sand = 0.;   
FLTPT sedDeg1Silt = 0.;   
FLTPT sedDeg1Clay = 0.;   
FLTPT sedDeg1Sag = 0.;    
FLTPT sedDeg1Lag = 0.;    
FLTPT sedDeg1Gravel = 0.; 

FLTPT degrte = 0.;       
FLTPT sedDegRemain = 0.; 
FLTPT sedDegSand = 0.;   
FLTPT sedDegSilt = 0.;   
FLTPT sedDegClay = 0.;   
FLTPT sedDegGravel = 0.; 

FLTPT bnkrte = 0.;    
FLTPT bnkSand = 0.;   
FLTPT bnkSilt = 0.;   
FLTPT bnkClay = 0.;   
FLTPT bnkGravel = 0.; 

FLTPT sedDep = 0.;       
FLTPT sedDepSand = 0.;   
FLTPT sedDepSilt = 0.;   
FLTPT sedDepClay = 0.;   
FLTPT sedDepSag = 0.;    
FLTPT sedDepLag = 0.;    
FLTPT sedDepGravel = 0.; 

FLTPT wet_peri = ChannelWettingPerimeter(m_chBtmWth[i], m_chDepth[i], m_chWtrDepth[i],
m_chSideSlope[i], m_chWth[i], 4.);
FLTPT pbank = wet_peri - m_chBtmWth[i];

FLTPT fpratio = 0.;
if (m_chWtrDepth[i] > m_chDepth[i]) {
FLTPT adddep = m_chWtrDepth[i] - m_chDepth[i];
FLTPT cross_area_bankful = ChannelCrossSectionalArea(m_chBtmWth[i], m_chDepth[i], m_chSideSlope[i]);
fpratio = 1. - (cross_area_bankful + adddep * m_chWth[i]) / cross_area;
fpratio = Max(0., fpratio);
}

FLTPT sfbank = CalPow(10., -1.4026 * log10(m_chBtmWth[i] / pbank + 1.5) + 2.247);
FLTPT tou = 9800. * m_chWtrDepth[i] * m_chSlope[i];
FLTPT asinea = 1. / CalSqrt(1. + m_chSideSlope[i] * m_chSideSlope[i]);
FLTPT tbank = tou * sfbank / 100. * (m_chWtrWth[i] + m_chBtmWth[i]) * asinea / (4. * m_chWtrDepth[i]);
FLTPT tbed = tou * (1. - sfbank / 100.) * (m_chWtrWth[i] / 2. / m_chBtmWth[i] + 0.5);

bnkrte = m_chBnkErod[i] * (tbank - m_chBnkTc[i]) * 1.e-6; 
if (bnkrte < 0.) bnkrte = 0.;
bnkrte *= m_chLen[i] * (m_chWtrDepth[i] * CalSqrt(1. + m_chSideSlope[i] * m_chSideSlope[i])) *
m_chBnkBD[i] * m_dt * 1000.; 
degrte = m_chBedErod[i] * (tbed - m_chBedTc[i]) * 1.e-6;
if (degrte < 0.) degrte = 0.;
degrte *= m_chLen[i] * m_chBtmWth[i] * m_chBedBD[i] * m_dt * 1000.;

FLTPT bnkrt = 1.;
if (bnkrte + degrte > 1.e-6) {
bnkrt = bnkrte / (bnkrte + degrte);
}
bnkrt = Min(1., bnkrt);
FLTPT bedrt = 1. - bnkrt; 

cyin = sedin / qdin; 

cych = m_sedTransEqCoef * CalPow(peakVel, m_sedTransEqExp) * 1000.; 
#ifdef PRINT_DEBUG
cout << ", cyin: " << cyin << ", cych: " << cych << endl;
#endif
sedDepNet = qdin * (cych - cyin);
if (sedDepNet <= UTIL_ZERO) {
sedDepNet = 0.;
bnkrte = 0.;
degrte = 0.;
} else {
if (sedDepNet >= m_rchDep[i]) {
FLTPT effbnkbed = sedDepNet - m_rchDep[i];
bnkrte = Min(effbnkbed * bnkrt, bnkrte);
bnkSand = bnkrte * m_chBnkSand[i];
bnkSilt = bnkrte * m_chBnkSilt[i];
bnkClay = bnkrte * m_chBnkClay[i];
bnkGravel = bnkrte * m_chBnkGravel[i];

degrte = Min(effbnkbed * bedrt, degrte);
sedDegSand = degrte * m_chBedSand[i];
sedDegSilt = degrte * m_chBedSilt[i];
sedDegClay = degrte * m_chBedClay[i];
sedDegGravel = degrte * m_chBedGravel[i];

sedDeg1 = m_rchDep[i];
sedDeg1Sand = m_rchDepSand[i];
sedDeg1Silt = m_rchDepSilt[i];
sedDeg1Clay = m_rchDepClay[i];
sedDeg1Sag = m_rchDepSag[i];
sedDeg1Lag = m_rchDepLag[i];
sedDeg1Gravel = m_rchDepGravel[i];

m_rchDep[i] = 0.;
m_rchDepSand[i] = 0.;
m_rchDepSilt[i] = 0.;
m_rchDepClay[i] = 0.;
m_rchDepSag[i] = 0.;
m_rchDepLag[i] = 0.;
m_rchDepGravel[i] = 0.;
} else {
bnkrte = 0.;
degrte = 0.;
sedDegSand = 0.;
sedDegSilt = 0.;
sedDegClay = 0.;
sedDegGravel = 0.;
bnkSand = 0.;
bnkSilt = 0.;
bnkClay = 0.;
bnkGravel = 0.;
m_rchDep[i] -= sedDepNet;
sedDeg1 = sedDepNet;
if (m_rchDepClay[i] >= sedDepNet) {
m_rchDepClay[i] -= sedDepNet;
sedDeg1Clay = sedDepNet;
sedDegRemain = 0.;
} else {
sedDegRemain = sedDepNet - m_rchDepClay[i];
sedDeg1Clay = m_rchDepClay[i];
m_rchDepClay[i] = 0.;
if (m_rchDepSilt[i] >= sedDegRemain) {
m_rchDepSilt[i] -= sedDegRemain;
sedDeg1Silt = sedDegRemain;
sedDegRemain = 0.;
} else {
sedDegRemain -= m_rchDepSilt[i];
sedDeg1Silt = m_rchDepSilt[i];
m_rchDepSilt[i] = 0.;
if (m_rchDepSag[i] >= sedDegRemain) {
m_rchDepSag[i] -= sedDegRemain;
sedDeg1Sag = sedDegRemain;
sedDegRemain = 0.;
} else {
sedDegRemain -= m_rchDepSag[i];
sedDeg1Sag = m_rchDepSag[i];
m_rchDepSag[i] = 0.;
if (m_rchDepSand[i] >= sedDegRemain) {
m_rchDepSand[i] -= sedDegRemain;
sedDeg1Sand = sedDegRemain;
sedDegRemain = 0.;
} else {
sedDegRemain -= m_rchDepSand[i];
sedDeg1Sand = m_rchDepSand[i];
m_rchDepSand[i] = 0.;
if (m_rchDepLag[i] >= sedDegRemain) {
m_rchDepLag[i] -= sedDegRemain;
sedDeg1Lag = sedDegRemain;
sedDegRemain = 0.;
} else {
sedDegRemain -= m_rchDepLag[i];
sedDeg1Lag = m_rchDepLag[i];
m_rchDepLag[i] = 0.;
if (m_rchDepGravel[i] >= sedDegRemain) {
m_rchDepGravel[i] -= sedDegRemain;
sedDeg1Gravel = sedDegRemain;
sedDegRemain = 0.;
} else {
sedDegRemain -= m_rchDepGravel[i];
sedDeg1Gravel = m_rchDepGravel[i];
m_rchDepGravel[i] = 0.;
} 
}     
}         
}             
}                 
}                     
}                         
}                             

if (m_rchDep[i] < UTIL_ZERO) {
m_rchDep[i] = 0.;
m_rchDepSand[i] = 0.;
m_rchDepSilt[i] = 0.;
m_rchDepClay[i] = 0.;
m_rchDepSag[i] = 0.;
m_rchDepLag[i] = 0.;
m_rchDepGravel[i] = 0.;
}


FLTPT x = 0.;

x = 1.055 * m_chLen[i] * 0.45666667 / (peakVel * m_chWtrDepth[i]);
if (x > 20.) x = 20.;
sedDepGravel = gravelin * Min(1. - CalExp(-x), 1.);

x = 1.055 * m_chLen[i] * 0.00456667 / (peakVel * m_chWtrDepth[i]);
if (x > 20.) x = 20.;
sedDepSand = sandin * Min(1. - CalExp(-x), 1.);

x = 1.055 * m_chLen[i] * 1.14166667e-05 / (peakVel * m_chWtrDepth[i]);
if (x > 20.) x = 20.;
sedDepSilt = siltin * Min(1. - CalExp(-x), 1.);

x = 1.055 * m_chLen[i] * 4.56666667e-07 / (peakVel * m_chWtrDepth[i]);
if (x > 20.) x = 20.;
sedDepClay = clayin * Min(1. - CalExp(-x), 1.);

x = 1.055 * m_chLen[i] * 0.00010275 / (peakVel * m_chWtrDepth[i]);
if (x > 20.) x = 20.;
sedDepSag = sagin * Min(1. - CalExp(-x), 1.);

x = 1.055 * m_chLen[i] * 0.02854167 / (peakVel * m_chWtrDepth[i]);
if (x > 20.) x = 20.;
sedDepLag = lagin * Min(1. - CalExp(-x), 1.);

sedDep = sedDepSand + sedDepSilt + sedDepClay + sedDepSag + sedDepLag + sedDepGravel;

m_fldplnDep[i] += (sedDepSilt + sedDepClay) * fpratio;
m_fldplnDepSilt[i] += sedDepSilt * fpratio;
m_fldplnDepClay[i] += sedDepClay * fpratio;

m_rchDep[i] += sedDep - (sedDepSilt + sedDepClay) * fpratio;
m_rchDepSilt[i] += sedDepSilt * (1. - fpratio);
m_rchDepClay[i] += sedDepClay * (1. - fpratio);
m_rchDepSand[i] += sedDepSand;
m_rchDepSag[i] += sedDepSag;
m_rchDepLag[i] += sedDepLag;
m_rchDepGravel[i] += sedDepGravel;

sedin += degrte + bnkrte + sedDeg1 - sedDep;
sandin += sedDegSand + bnkSand + sedDeg1Sand - sedDepSand;
siltin += sedDegSilt + bnkSilt + sedDeg1Silt - sedDepSilt;
clayin += sedDegClay + bnkClay + sedDeg1Clay - sedDepClay;
sagin += sedDeg1Sag - sedDepSag;
lagin += sedDeg1Lag - sedDepLag;
gravelin += sedDegGravel + bnkGravel + sedDeg1Gravel - sedDepGravel;

if (sedin < UTIL_ZERO) {
sedin = 0.;
sandin = 0.;
siltin = 0.;
clayin = 0.;
sagin = 0.;
lagin = 0.;
gravelin = 0.;
}

FLTPT outfract = m_rteWtrOut[i] / qdin;
m_sedRchOut[i] = sedin * outfract;                    
m_sedConcRchOut[i] = m_sedRchOut[i] / m_rteWtrOut[i]; 
m_sandRchOut[i] = sandin * outfract;                  
m_siltRchOut[i] = siltin * outfract;                  
m_clayRchOut[i] = clayin * outfract;                  
m_sagRchOut[i] = sagin * outfract;                    
m_lagRchOut[i] = lagin * outfract;                    
m_gravelRchOut[i] = gravelin * outfract;              

if (m_sedRchOut[i] < UTIL_ZERO) {
m_sedRchOut[i] = 0.;
m_sedConcRchOut[i] = 0.;
m_sandRchOut[i] = 0.;
m_siltRchOut[i] = 0.;
m_clayRchOut[i] = 0.;
m_sagRchOut[i] = 0.;
m_lagRchOut[i] = 0.;
m_gravelRchOut[i] = 0.;
}

m_sedSto[i] = sedin - m_sedRchOut[i];
m_sandSto[i] = sandin - m_sandRchOut[i];
m_siltSto[i] = siltin - m_siltRchOut[i];
m_claySto[i] = clayin - m_clayRchOut[i];
m_sagSto[i] = sagin - m_sagRchOut[i];
m_lagSto[i] = lagin - m_lagRchOut[i];
m_gravelSto[i] = gravelin - m_gravelRchOut[i];
if (m_sedSto[i] < UTIL_ZERO) {
m_sedSto[i] = 0.;
m_sandSto[i] = 0.;
m_siltSto[i] = 0.;
m_claySto[i] = 0.;
m_sagSto[i] = 0.;
m_lagSto[i] = 0.;
m_gravelSto[i] = 0.;
}

m_rchBnkEro[i] = bnkrte;
m_rchDeg[i] = degrte;
if (m_rchDep[i] >= m_preRchDep[i]) {
m_dltRchDep[i] = m_rchDep[i] - m_preRchDep[i];
} else {
m_dltRchDep[i] = 0.;
}
m_preRchDep[i] = m_rchDep[i];
if (m_fldplnDep[i] >= m_preFldplnDep[i]) {
m_dltFldplnDep[i] = m_fldplnDep[i] - m_preFldplnDep[i];
} else {
m_dltFldplnDep[i] = 0.;
}
m_preFldplnDep[i] = m_fldplnDep[i];

#ifdef PRINT_DEBUG
cout << ", sedRchOut: " << m_sedRchOut[i] << endl;
#endif
}

void SEDR_SBAGNOLD::ChannelDowncuttingWidening(const int i) {
if (!m_vcd) return;
FLTPT depdeg = m_chDepth[i] - m_initChDepth[i];
if (depdeg < m_initChSlope[i] * m_initChLen[i]) {
if (m_chSto[i] + m_rteWtrOut[i] > 1.4e6) {
FLTPT cutdepth = 358.6 * m_chWtrDepth[i] * m_chSlope[i] * m_chBnkCov[i];
m_chDepth[i] += cutdepth;
m_chWth[i] = m_chDepth[i] * m_chWthDepthRt[i];
m_chSlope[i] -= cutdepth / m_chLen[i];
m_chSlope[i] = Max(0.0001, m_chSlope[i]);
m_chBtmWth[i] = ChannleBottomWidth(m_chWth[i], m_chSideSlope[i], m_chDepth[i]);
}
}
}
