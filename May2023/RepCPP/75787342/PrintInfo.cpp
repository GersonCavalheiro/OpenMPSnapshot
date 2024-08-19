#include "PrintInfo.h"

#include "utils_time.h"
#include "text.h"
#include "BMPText.h"
#include "Logging.h"

using namespace utils_time;
using namespace data_raster;


PrintInfoItem::PrintInfoItem(const int scenario_id , const int calibration_id )
: m_1DDataWithRowCol(nullptr), m_nRows(-1), m_1DData(nullptr),
m_nLayers(-1), m_2DData(nullptr), TimeSeriesDataForSubbasinCount(-1), SiteID(-1),
SiteIndex(-1), SubbasinID(-1), SubbasinIndex(-1),
m_startTime(0),  m_endTime(0),
m_scenarioID(scenario_id), m_calibrationID(calibration_id),
m_Counter(-1), m_AggregationType(AT_Unknown) {
TimeSeriesData.clear();
TimeSeriesDataForSubbasin.clear();
}

PrintInfoItem::~PrintInfoItem() {
CLOG(TRACE, LOG_RELEASE) << "Start to release PrintInfoItem for " << Filename << " ...";
Release2DArray(m_1DDataWithRowCol);
Release1DArray(m_1DData);
Release2DArray(m_2DData);

for (auto it = TimeSeriesDataForSubbasin.begin(); it != TimeSeriesDataForSubbasin.end(); ++it) {
if (it->second != nullptr) {
Release1DArray(it->second);
}
}
TimeSeriesDataForSubbasin.clear();

CLOG(TRACE, LOG_RELEASE) << "End to release PrintInfoItem for " << Filename << " .";
}

bool PrintInfoItem::IsDateInRange(time_t dt) {
bool bStatus = false;
if (dt >= m_startTime && dt <= m_endTime) {
bStatus = true;
}
return bStatus;
}

void PrintInfoItem::add1DTimeSeriesResult(time_t t, int n, const FLTPT* data) {
FLTPT* temp = new FLTPT[n];
for (int i = 0; i < n; i++) {
temp[i] = data[i];
}
TimeSeriesDataForSubbasin[t] = temp;
TimeSeriesDataForSubbasinCount = n;
}

void PrintInfoItem::add1DRasterTimeSeriesResult(time_t t, int n, const float* data) {
float* temp = new float[n];
for (int i = 0; i < n; i++) {
temp[i] = data[i];
}
TimeSeriesDataForRaster[t] = temp;
TimeSeriesDataForRasterCount = n;
}
void PrintInfoItem::Flush(const string& projectPath, MongoGridFs* gfs, IntRaster* templateRaster, const string& header) {


bool outToMongoDB = false; 
map<string, string> opts;
if (SubbasinID != 0 && SubbasinID != 9999 && 
(m_1DData != nullptr || m_2DData != nullptr)) {
outToMongoDB = true;
Filename = itoa(SubbasinID) + "_" + Corename;
#ifdef HAS_VARIADIC_TEMPLATES
opts.emplace("SCENARIO_ID", itoa(m_scenarioID));
opts.emplace("CALIBRATION_ID", itoa(m_calibrationID));
opts.emplace("DATATYPE_OUT", "FLOAT");
#else
opts.insert(make_pair("SCENARIO_ID", itoa(m_scenarioID)));
opts.insert(make_pair("CALIBRATION_ID", itoa(m_calibrationID)));
opts.insert(make_pair("DATATYPE_OUT", "FLOAT")); 
#endif
}
if (!StringMatch(AggType, "")) {
Filename += "_" + AggType;
Corename += "_" + AggType;
}
string gfs_name = Filename + "_";
if (m_scenarioID >= 0) gfs_name += itoa(m_scenarioID);
gfs_name += "_";
if (m_calibrationID >= 0) gfs_name += itoa(m_calibrationID);
if (outToMongoDB) {
CLOG(TRACE, LOG_OUTPUT) << "Creating output file " << Filename << "(GridFS file: " << gfs_name << ")...";
} else {
CLOG(TRACE, LOG_OUTPUT) << "Creating output file " << Filename << "...";
}
if (m_AggregationType == AT_SpecificCells) {

return;
}
if (!TimeSeriesData.empty() && (SiteID != -1 || SubbasinID != -1)) {
std::ofstream fs;
string filename = projectPath + Filename + "." + TextExtension;
fs.open(filename.c_str(), std::ios::out | std::ios::app); 
if (fs.is_open()) {
if (SubbasinID == 0) {
fs << "Watershed: " << endl;
} else {
fs << "Subbasin: " << SubbasinID << endl;
}
fs << header << endl;
for (auto it = TimeSeriesData.begin(); it != TimeSeriesData.end(); ++it) {
fs << ConvertToString2(it->first) << " " << std::right << std::fixed
<< std::setw(15) << std::setfill(' ') << setprecision(8) << it->second << endl;
}
fs.close();
CLOG(TRACE, LOG_OUTPUT) << "Create " << filename + " successfully!";
}
return;
}
if (!TimeSeriesDataForRaster.empty() && SubbasinID != -1) {
for (auto it = TimeSeriesDataForRaster.begin(); it != TimeSeriesDataForRaster.end(); ++it) {
string filename = Filename + "_" + ConvertToString3(it->first);
}
}
if (!TimeSeriesDataForSubbasin.empty() && SubbasinID != -1) {
std::ofstream fs;
string filename = projectPath + Filename + "." + TextExtension;
fs.open(filename.c_str(), std::ios::out | std::ios::app);
if (fs.is_open()) {
fs << endl;
if (SubbasinID == 0 || SubbasinID == 9999) {
fs << "Watershed: " << endl;
} else {
fs << "Subbasin: " << SubbasinID << endl;
}
fs << header << endl;
for (auto it = TimeSeriesDataForSubbasin.begin(); it != TimeSeriesDataForSubbasin.end(); ++it) {
fs << ConvertToString2(it->first);
for (int i = 0; i < TimeSeriesDataForSubbasinCount; i++) {
fs << " " << std::right << std::fixed << std::setw(15) << std::setfill(' ')
<< setprecision(8) << it->second[i];
}
fs << endl;
}
fs.close();
CLOG(TRACE, LOG_OUTPUT) << "Create " << filename << " successfully!";
}
return;
}
if (m_nRows > -1 &&
(nullptr != m_1DData && m_nLayers == 1) || 
(nullptr != m_2DData && m_nLayers > 1)) {  
if (nullptr == templateRaster) {
throw ModelException("PrintInfoItem", "Flush", "The templateRaster is NULL.");
}
bool is1d = nullptr != m_1DData && m_nLayers == 1 ? true : false;

if (Suffix == GTiffExtension || Suffix == ASCIIExtension) {
FloatRaster* rs_data = nullptr;
if (is1d) { 
rs_data = new FloatRaster(templateRaster, m_1DData, m_nRows);
} else { 
rs_data = new FloatRaster(templateRaster, m_2DData, m_nRows, m_nLayers);
}
if (outToMongoDB) {
gfs->RemoveFile(gfs_name);
int try_times = 1;
while (try_times <= 3) { 
if (!rs_data->OutputToMongoDB(gfs, gfs_name, opts, false)) {
CLOG(WARNING, LOG_OUTPUT) << "-- The raster data " << gfs_name << " output to MongoDB FAILED!"
<< " Try time: " << try_times;
} else {
CLOG(TRACE, LOG_OUTPUT) << "-- The raster data " << gfs_name << "-- saved to MongoDB SUCCEED!";
break;
}
SleepMs(2); 
try_times++;
}
} else {
rs_data->OutputToFile(projectPath + Filename + "." + Suffix);
}
delete rs_data; 
} else if (Suffix == TextExtension) { 
std::ofstream fs;
string filename = projectPath + Filename + "." + TextExtension;
DeleteExistedFile(filename);
fs.open(filename.c_str(), std::ios::out);
if (fs.is_open()) {
int valid_num = templateRaster->GetValidNumber();
int nlyrs = is1d == true ? 1 : templateRaster->GetLayers();
for (int idx = 0; idx < valid_num; idx++) {
if (is1d) {
fs << idx << ", " << setprecision(8) << m_1DData[idx] << endl;
} else {
fs << idx;
for (int ilyr = 0; ilyr < nlyrs; ilyr++) {
fs << ", " << setprecision(8) << m_2DData[idx][ilyr];
}
fs << endl;
}
}
fs.close();
CLOG(TRACE, LOG_OUTPUT) << "-- Create " << filename << " successfully!";
}
}
return;
}
if (!TimeSeriesData.empty()) {
std::ofstream fs;
string filename = projectPath + Filename + "." + TextExtension;
DeleteExistedFile(filename);
fs.open(filename.c_str(), std::ios::out);
if (fs.is_open()) {
for (auto it = TimeSeriesData.begin(); it != TimeSeriesData.end(); ++it) {
fs << ConvertToString2(it->first) << " " << std::right << std::fixed
<< std::setw(15) << std::setfill(' ') << setprecision(8) << it->second << endl;
}
fs.close();
CLOG(TRACE, LOG_OUTPUT) << "Create " << filename << " successfully!";
}
return;
}
LOG(WARNING) << "PrintInfoItem\n Flush\n Creating " << Filename <<
" is failed. There is no result data for this file. Please check output variables of modules.";
}

void PrintInfoItem::AggregateData2D(time_t time, int nRows, int nCols, FLTPT** data) {
if (m_AggregationType == AT_SpecificCells) {
return; 
}
if (nullptr == m_2DData) {
m_nRows = nRows;
m_nLayers = nCols;
Initialize2DArray(m_nRows, m_nLayers, m_2DData, NODATA_VALUE);
m_Counter = 0;
}

switch (m_AggregationType) {
case AT_Average:
#pragma omp parallel for
for (int i = 0; i < m_nRows; i++) {
for (int j = 0; j < m_nLayers; j++) {
if (!FloatEqual(data[i][j], NODATA_VALUE)) {
if (FloatEqual(m_2DData[i][j], NODATA_VALUE)) {
m_2DData[i][j] = 0.f;
}
m_2DData[i][j] = (m_2DData[i][j] * m_Counter + data[i][j]) / (m_Counter + 1.f);
}
}
}
break;
case AT_Sum:
#pragma omp parallel for
for (int i = 0; i < m_nRows; i++) {
for (int j = 0; j < m_nLayers; j++) {
if (!FloatEqual(data[i][j], NODATA_VALUE)) {
if (FloatEqual(m_2DData[i][j], NODATA_VALUE)) {
m_2DData[i][j] = 0.;
}
m_2DData[i][j] += data[i][j];
}
}
}
break;
case AT_Minimum:
#pragma omp parallel for
for (int i = 0; i < m_nRows; i++) {
for (int j = 0; j < m_nLayers; j++) {
if (!FloatEqual(data[i][j], NODATA_VALUE)) {
if (FloatEqual(m_2DData[i][j], NODATA_VALUE)) {
m_2DData[i][j] = MAXIMUMFLOAT;
}
if (data[i][j] <= m_2DData[i][j]) {
m_2DData[i][j] = data[i][j];
}
}
}
}
break;
case AT_Maximum:
#pragma omp parallel for
for (int i = 0; i < m_nRows; i++) {
for (int j = 0; j < m_nLayers; j++) {
if (!FloatEqual(data[i][j], NODATA_VALUE)) {
if (FloatEqual(m_2DData[i][j], NODATA_VALUE)) {
m_2DData[i][j] = MISSINGFLOAT;
}
if (data[i][j] >= m_2DData[i][j]) {
m_2DData[i][j] = data[i][j];
}
}
}
}
break;
default: break;
}
m_Counter++;
}


void PrintInfoItem::AggregateData(time_t time, int numrows, FLTPT* data) {
if (m_AggregationType == AT_SpecificCells) {

} else {
if (m_1DData == nullptr) {
m_nRows = numrows;
m_nLayers = 1;
Initialize1DArray(m_nRows, m_1DData, NODATA_VALUE);
m_Counter = 0;
}

#pragma omp parallel for
for (int rw = 0; rw < m_nRows; rw++) {
switch (m_AggregationType) {
case AT_Average:
if (!FloatEqual(data[rw], NODATA_VALUE)) {
if (FloatEqual(m_1DData[rw], NODATA_VALUE)) {
m_1DData[rw] = 0.;
}
m_1DData[rw] = (m_1DData[rw] * m_Counter + data[rw]) / (m_Counter + 1.);
}
break;
case AT_Sum:
if (!FloatEqual(data[rw], NODATA_VALUE)) {
if (FloatEqual(m_1DData[rw], NODATA_VALUE)) {
m_1DData[rw] = 0.;
}
m_1DData[rw] += data[rw];
}
break;
case AT_Minimum:
if (!FloatEqual(data[rw], NODATA_VALUE)) {
if (FloatEqual(m_1DData[rw], NODATA_VALUE)) {
m_1DData[rw] = MAXIMUMFLOAT;
}
if (m_1DData[rw] >= data[rw]) {
m_1DData[rw] = data[rw];
}
}
break;
case AT_Maximum:
if (!FloatEqual(data[rw], NODATA_VALUE)) {
if (FloatEqual(m_1DData[rw], NODATA_VALUE)) {
m_1DData[rw] = MISSINGFLOAT;
}
if (m_1DData[rw] <= data[rw]) {
m_1DData[rw] = data[rw];
}
}
break;
default: break;
}
}
m_Counter++;
}
}

void PrintInfoItem::AggregateData(int numrows, FLTPT** data, AggregationType type, FLTPT NoDataValue) {
if (m_1DDataWithRowCol == nullptr) {
m_nRows = numrows;
m_1DDataWithRowCol = new FLTPT *[m_nRows];
for (int i = 0; i < m_nRows; i++) {
m_1DDataWithRowCol[i] = new FLTPT[3];
m_1DDataWithRowCol[i][0] = NoDataValue;
m_1DDataWithRowCol[i][1] = NoDataValue;
m_1DDataWithRowCol[i][2] = NoDataValue;
}
m_Counter = 0;
}

switch (type) {
case AT_Average:
#pragma omp parallel for
for (int rw = 0; rw < m_nRows; rw++) {
if (!FloatEqual(data[rw][2], NoDataValue)) {
if (FloatEqual(m_1DDataWithRowCol[rw][2], NoDataValue)) m_1DDataWithRowCol[rw][2] = 0.0f;
m_1DDataWithRowCol[rw][0] = data[rw][0]; 
m_1DDataWithRowCol[rw][1] = data[rw][1]; 
if (m_Counter == 0) {
m_1DDataWithRowCol[rw][2] = data[rw][2]; 
} else {
m_1DDataWithRowCol[rw][2] =
(m_1DDataWithRowCol[rw][2] * m_Counter + data[rw][2]) / (m_Counter + 1);
}
}
}
m_Counter++;
break;
case AT_Sum:
#pragma omp parallel for
for (int rw = 0; rw < m_nRows; rw++) {
if (!FloatEqual(data[rw][2], NoDataValue)) {
if (FloatEqual(m_1DDataWithRowCol[rw][2], NoDataValue)) m_1DDataWithRowCol[rw][2] = 0.;
m_1DDataWithRowCol[rw][0] = data[rw][0];
m_1DDataWithRowCol[rw][1] = data[rw][1];
m_1DDataWithRowCol[rw][2] += data[rw][2];
}
}
break;
case AT_Minimum:
#pragma omp parallel for
for (int rw = 0; rw < m_nRows; rw++) {
if (!FloatEqual(data[rw][2], NoDataValue)) {
if (FloatEqual(m_1DDataWithRowCol[rw][2], NoDataValue)) m_1DDataWithRowCol[rw][2] = 0.;
m_1DDataWithRowCol[rw][0] = data[rw][0];
m_1DDataWithRowCol[rw][1] = data[rw][1];
if (data[rw][2] <= m_1DDataWithRowCol[rw][2]) {
m_1DDataWithRowCol[rw][2] = data[rw][2];
}
}
}
break;
case AT_Maximum:
#pragma omp parallel for
for (int rw = 0; rw < m_nRows; rw++) {
if (!FloatEqual(data[rw][2], NoDataValue)) {
if (FloatEqual(m_1DDataWithRowCol[rw][2], NoDataValue)) m_1DDataWithRowCol[rw][2] = 0.;
m_1DDataWithRowCol[rw][0] = data[rw][0];
m_1DDataWithRowCol[rw][1] = data[rw][1];
if (data[rw][2] >= m_1DDataWithRowCol[rw][2]) {
m_1DDataWithRowCol[rw][2] = data[rw][2];
}
}
}
break;
default: break;
}
}

AggregationType PrintInfoItem::MatchAggregationType(const string& type) {
AggregationType res = AT_Unknown;
if (StringMatch(type, Tag_Unknown)) {
res = AT_Unknown;
}
if (StringMatch(type, Tag_Sum)) {
res = AT_Sum;
}
if (StringMatch(type, Tag_Average) || StringMatch(type, "AVERAGE") || StringMatch(type, "MEAN")) {
res = AT_Average;
}
if (StringMatch(type, Tag_Minimum)) {
res = AT_Minimum;
}
if (StringMatch(type, Tag_Maximum)) {
res = AT_Maximum;
}
if (StringMatch(type, Tag_SpecificCells)) {
res = AT_SpecificCells;
}
if (StringMatch(type, Tag_TimeSeries)) {
res = AT_TimeSeries;
}
return res;
}



PrintInfo::PrintInfo(const int scenario_id , const int calibration_id )
: m_scenarioID(scenario_id), m_calibrationID(calibration_id),
m_Interval(0), m_param(nullptr), m_subbasinSelectedArray(nullptr) {
m_PrintItems.clear();
}

PrintInfo::~PrintInfo() {
for (auto it = m_PrintItems.begin(); it != m_PrintItems.end(); ++it) {
if (nullptr != *it) {
delete *it;
*it = nullptr;
}
}
m_PrintItems.clear();

m_param = nullptr; 
if (nullptr != m_subbasinSelectedArray) {
Release1DArray(m_subbasinSelectedArray);
}
}

string PrintInfo::getOutputTimeSeriesHeader() {
vector<string> headers;
if (StringMatch(m_OutputID, VAR_SNWB[0])) {
headers.emplace_back("Time");
headers.emplace_back("P");
headers.emplace_back("P_net");
headers.emplace_back("P_blow");
headers.emplace_back("T");
headers.emplace_back("Wind");
headers.emplace_back("SR");
headers.emplace_back("SE");
headers.emplace_back("SM");
headers.emplace_back("SA");
} else if (StringMatch(m_OutputID, VAR_SOWB[0])) {
headers.emplace_back("Time");
headers.emplace_back("PCP (mm)");
headers.emplace_back("meanTmp (deg C)");
headers.emplace_back("soilTmp (deg C)");
headers.emplace_back("netPcp (mm)");
headers.emplace_back("InterceptionET (mm)");
headers.emplace_back("DepressionET (mm)");
headers.emplace_back("Infiltration (mm)");
headers.emplace_back("Total_ET (mm)");
headers.emplace_back("Soil_ET (mm)");
headers.emplace_back("NetPercolation (mm)");
headers.emplace_back("Revaporization (mm)");
headers.emplace_back("SurfaceRunoff (mm)");
headers.emplace_back("SubsurfaceRunoff (mm)");
headers.emplace_back("Baseflow (mm)");
headers.emplace_back("AllRunoff (mm)");
headers.emplace_back("SoilMoisture (mm)");
} else if (StringMatch(m_OutputID, VAR_GWWB[0])) {
headers.emplace_back("Time");
headers.emplace_back("Percolation (mm)");
headers.emplace_back("Revaporization (mm)");
headers.emplace_back("Deep Percolation (mm)");
headers.emplace_back("Baseflow (mm)");
headers.emplace_back("Groundwater storage (mm)");
headers.emplace_back("Baseflow discharge (m3/s)");
} else if (StringMatch(m_OutputID, "T_RECH")) {
headers.emplace_back("Time");
headers.emplace_back("QS");
headers.emplace_back("QI");
headers.emplace_back("QG");
headers.emplace_back("Q(m^3/s)");
headers.emplace_back("Q(mm)");
} else if (StringMatch(m_OutputID, "T_WABA")) {
headers.emplace_back("Time");
headers.emplace_back("Vin");
headers.emplace_back("Vside");
headers.emplace_back("Vdiv");
headers.emplace_back("Vpoint");
headers.emplace_back("Vseep");
headers.emplace_back("Vnb");
headers.emplace_back("Ech");
headers.emplace_back("Lbank");
headers.emplace_back("Vbank");
headers.emplace_back("Vout");
headers.emplace_back("Vch");
headers.emplace_back("Depth");
headers.emplace_back("Velocity");
} else if (StringMatch(m_OutputID, "T_RSWB")) {
headers.emplace_back("Time");
headers.emplace_back("Qin(m^3/s)");
headers.emplace_back("Area(ha)");
headers.emplace_back("Storage(10^4*m^3)");
headers.emplace_back("QSout(m^3/s)");
headers.emplace_back("QIout(m^3/s)");
headers.emplace_back("QGout(m^3/s)");
headers.emplace_back("Qout(m^3/s)");
headers.emplace_back("Qout(mm)");
} else if (StringMatch(m_OutputID, "T_CHSB")) {
headers.emplace_back("Time");
headers.emplace_back("SedInUpStream");
headers.emplace_back("SedInSubbasin");
headers.emplace_back("SedDeposition");
headers.emplace_back("SedDegradation");
headers.emplace_back("SedStorage");
headers.emplace_back("SedOut");
} else if (StringMatch(m_OutputID, "T_RESB")) {
headers.emplace_back("Time");
headers.emplace_back("ResSedIn");
headers.emplace_back("ResSedSettling");
headers.emplace_back("ResSedStorage");
headers.emplace_back("ResSedOut");
}
std::ostringstream oss;
for (auto it = headers.begin(); it < headers.end(); ++it) {
if (StringMatch(*it, "Time")) {
oss << *it;
} else {
oss << " " << std::setw(15) << std::right << std::setfill(' ') << *it;
}
}
return oss.str();
}

void PrintInfo::AddPrintItem(const time_t start, const time_t end, const string& file, const string& sufi) {
PrintInfoItem* itm = new PrintInfoItem(m_scenarioID, m_calibrationID);

itm->SiteID = -1;
itm->Corename = file;
itm->Filename = file;
itm->Suffix = sufi;
itm->m_startTime = start;
itm->m_endTime = end;
m_PrintItems.emplace_back(itm);
}

void PrintInfo::AddPrintItem(string& type, const time_t start, const time_t end,
const string& file, const string& sufi, const int subbasinID ) {
PrintInfoItem* itm = new PrintInfoItem(m_scenarioID, m_calibrationID);

itm->SiteID = -1;
itm->SubbasinID = subbasinID;
itm->Corename = file;
itm->Filename = subbasinID == 0 || subbasinID == 9999 ? file : file + "_" + ValueToString(subbasinID);
itm->Suffix = sufi;

itm->m_startTime = start;
itm->m_endTime = end;

type = Trim(type);
itm->AggType = type;
AggregationType enumType = PrintInfoItem::MatchAggregationType(type);
if (enumType == AT_Unknown) {
throw ModelException("PrintInfo", "AddPrintItem", "The type of output " + m_OutputID +
" can't be unknown. Please check file.out. The type should be MIN, MAX, SUM or AVERAGE (AVE or MEAN).");
}
itm->setAggregationType(enumType);
m_PrintItems.emplace_back(itm);
}

void PrintInfo::AddPrintItem(const time_t start, const time_t end, const string& file, const string sitename,
const string& sufi, const bool isSubbasin) {
PrintInfoItem* itm = new PrintInfoItem(m_scenarioID, m_calibrationID);
char* strend = nullptr;
errno = 0;
if (!isSubbasin) {
itm->SiteID = strtol(sitename.c_str(), &strend, 10);
if (errno != 0) {
throw ModelException("PrintInfo", "AddPrintItem", "SiteID converted to integer failed!");
}
} else {
itm->SubbasinID = strtol(sitename.c_str(), &strend, 10); 
if (errno != 0) {
throw ModelException("PrintInfo", "AddPrintItem", "SubbasinID converted to integer failed!");
}
itm->SubbasinIndex = CVT_INT(m_subbasinSeleted.size());
m_subbasinSeleted.emplace_back(itm->SubbasinID);
}
itm->Corename = file;
itm->Filename = file;
itm->Suffix = sufi;
itm->m_startTime = start;
itm->m_endTime = end;

m_PrintItems.emplace_back(itm);
}

void PrintInfo::getSubbasinSelected(int* count, int** subbasins) {
*count = CVT_INT(m_subbasinSeleted.size());
if (m_subbasinSelectedArray == nullptr && !m_subbasinSeleted.empty()) {
m_subbasinSelectedArray = new int[m_subbasinSeleted.size()];
int index = 0;
for (auto it = m_subbasinSeleted.begin(); it < m_subbasinSeleted.end(); ++it) {
m_subbasinSelectedArray[index] = *it;
index++;
}
}
*subbasins = m_subbasinSelectedArray;
}

PrintInfoItem* PrintInfo::getPrintInfoItem(const int index) {
PrintInfoItem* res = nullptr;

if (index >= 0 && index < CVT_INT(m_PrintItems.size())) {
res = m_PrintItems.at(index);
}
return res;
}
