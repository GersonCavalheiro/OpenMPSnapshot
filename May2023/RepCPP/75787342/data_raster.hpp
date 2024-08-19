
#ifndef CCGL_DATA_RASTER_H
#define CCGL_DATA_RASTER_H

#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cassert>
#ifdef SUPPORT_OMP
#include <omp.h>
#endif 

#ifdef USE_GDAL
#include "gdal.h"
#include "gdal_priv.h"
#include "cpl_string.h"
#include "ogr_spatialref.h"
#endif 

#include "basic.h"
#include "utils_filesystem.h"
#include "utils_string.h"
#include "utils_array.h"
#include "utils_math.h"
#ifdef USE_MONGODB
#include "db_mongoc.h"
#endif 

using std::map;
using std::set;
#ifndef HAS_VARIADIC_TEMPLATES 
using std::make_pair;
#endif
using std::setprecision;

#ifndef CONST_CHARS
#define CONST_CHARS static const char* 
#endif

namespace ccgl {
using namespace utils_string;
using namespace utils_filesystem;
using namespace utils_array;
using namespace utils_math;
#ifdef USE_MONGODB
using namespace db_mongoc;
#endif 


namespace data_raster {
CONST_CHARS HEADER_RS_NODATA = "NODATA_VALUE"; 
CONST_CHARS HEADER_RS_XLL = "XLLCENTER";  
CONST_CHARS HEADER_RS_YLL = "YLLCENTER";  
CONST_CHARS HEADER_RS_XLLCOR = "XLLCORNER";  
CONST_CHARS HEADER_RS_YLLCOR = "YLLCORNER"; 
CONST_CHARS HEADER_RS_NROWS = "NROWS"; 
CONST_CHARS HEADER_RS_NCOLS = "NCOLS"; 
CONST_CHARS HEADER_RS_CELLSIZE = "CELLSIZE"; 
CONST_CHARS HEADER_RS_LAYERS = "LAYERS"; 
CONST_CHARS HEADER_RS_CELLSNUM = "CELLSNUM"; 
CONST_CHARS HEADER_RS_SRS = "SRS"; 
CONST_CHARS HEADER_RS_DATATYPE = "DATATYPE"; 
CONST_CHARS HEADER_RSOUT_DATATYPE = "DATATYPE_OUT"; 
CONST_CHARS HEADER_INC_NODATA = "INCLUDE_NODATA"; 
CONST_CHARS HEADER_MASK_NAME = "MASK_NAME"; 
CONST_CHARS STATS_RS_VALIDNUM = "VALID_CELLNUMBER"; 
CONST_CHARS STATS_RS_MEAN = "MEAN"; 
CONST_CHARS STATS_RS_MIN = "MIN"; 
CONST_CHARS STATS_RS_MAX = "MAX"; 
CONST_CHARS STATS_RS_STD = "STD"; 
CONST_CHARS STATS_RS_RANGE = "RANGE"; 
CONST_CHARS ASCIIExtension = "asc"; 
CONST_CHARS GTiffExtension = "tif"; 

typedef std::pair<int, int> ROW_COL; 
typedef std::pair<double, double> XY_COOR; 


typedef enum {
RDT_Unknown,
RDT_UInt8,
RDT_Int8,
RDT_UInt16,
RDT_Int16,
RDT_UInt32,
RDT_Int32,
RDT_Float,
RDT_Double
} RasterDataType;




string RasterDataTypeToString(int type);


RasterDataType StringToRasterDataType(const string& stype);


RasterDataType TypeToRasterDataType(const std::type_info& t);


double DefaultNoDataByType(RasterDataType type);

#ifdef USE_GDAL
GDALDataType CvtToGDALDataType(RasterDataType type);
#endif


STRDBL_MAP InitialHeader();


void CopyHeader(const STRDBL_MAP& refers, STRDBL_MAP& dst);


template <typename T>
void UpdateHeader(STRDBL_MAP& header, const string& key, T val) {
if (header.find(key) == header.end()) {
#ifdef HAS_VARIADIC_TEMPLATES
header.emplace(key, CVT_DBL(val));
#else
header.insert(make_pair(key, CVT_DBL(val)));
#endif
}
else { header.at(key) = CVT_DBL(val); }
}


STRING_MAP InitialStrHeader();


void UpdateStrHeader(STRING_MAP& strheader, const string& key, const string& val);

RasterDataType RasterDataTypeInOptionals(const STRING_MAP& opts);


void InitialStatsMap(STRDBL_MAP& stats, map<string, double*>& stats2d);


template <typename T>
bool ReadAscFile(const string& filename, STRDBL_MAP& header, T*& values) {
StatusMessage(("Read " + filename + "...").c_str());
vector<string> content_strs;
if (!LoadPlainTextFile(filename, content_strs)) { return false; }
if (content_strs.size() < 7) {
StatusMessage("Error: ASCII raster data requires at least 7 lines!");
return false;
}
double cellsize = -1.;
double xll = MISSINGFLOAT;
double yll = MISSINGFLOAT;
double nodata = MISSINGFLOAT;
bool xllcorner = false;
bool yllcorner = false;
int rows = -1;
int cols = -1;
int vidx = 0;
for (auto iter = content_strs.begin(); iter != content_strs.end(); ++iter) {
vector<string> tokens = SplitString(*iter);
if (tokens.empty()) { continue; }
bool str2value = false;
if (StringMatch(tokens[0], HEADER_RS_NCOLS)) {
if (tokens.size() < 2) { return false; }
cols = CVT_INT(IsInt(tokens[1], str2value));
if (!str2value) { return false; }
}
else if (StringMatch(tokens[0], HEADER_RS_NROWS)) {
if (tokens.size() < 2) { return false; }
rows = CVT_INT(IsInt(tokens[1], str2value));
if (!str2value) { return false; }
}
else if (StringMatch(tokens[0], HEADER_RS_XLL) ||
StringMatch(tokens[0], HEADER_RS_XLLCOR)) {
if (tokens.size() < 2) { return false; }
xll = IsDouble(tokens[1], str2value);
if (StringMatch(tokens[0], HEADER_RS_XLLCOR)) { xllcorner = true; }
if (!str2value) { return false; }
}
else if (StringMatch(tokens[0], HEADER_RS_YLL) ||
StringMatch(tokens[0], HEADER_RS_YLLCOR)) {
if (tokens.size() < 2) { return false; }
yll = IsDouble(tokens[1], str2value);
if (StringMatch(tokens[0], HEADER_RS_YLLCOR)) { yllcorner = true; }
if (!str2value) { return false; }
}
else if (StringMatch(tokens[0], HEADER_RS_CELLSIZE)) {
if (tokens.size() < 2) { return false; }
cellsize = IsDouble(tokens[1], str2value);
if (!str2value) { return false; }
}
else if (StringMatch(tokens[0], HEADER_RS_NODATA)) {
if (tokens.size() < 2) { return false; }
nodata = IsDouble(tokens[1], str2value);
if (!str2value) { return false; }
}
else {
if (rows < 0 || cols < 0) {
StatusMessage("Warning: NCOLS and NROWS should be defined first!");
}
if (nullptr == values) { Initialize1DArray(rows * cols, values, nodata); }
for (auto it_value = tokens.begin(); it_value != tokens.end(); ++it_value) {
double tmpv = IsDouble(*it_value, str2value);
if (!str2value) {
StatusMessage("Error: No value occurred in Raster data matrix!");
if (nullptr != values) { Release1DArray(values); }
return false;
}
if (vidx == rows * cols) {
StatusMessage("Error: Count of values MUST equal to rows * cols!");
if (nullptr != values) { Release1DArray(values); }
return false;
}
values[vidx++] = static_cast<T>(tmpv);
}
}
}
if (rows < 0 || cols < 0 || cellsize < 0 ||
FloatEqual(xll, MISSINGFLOAT) || FloatEqual(yll, MISSINGFLOAT)) {
StatusMessage("Error: Header information incomplete!");
return false;
}
if (xllcorner) { xll += 0.5 * cellsize; }
if (yllcorner) { yll += 0.5 * cellsize; }
UpdateHeader(header, HEADER_RS_NCOLS, cols);
UpdateHeader(header, HEADER_RS_NROWS, rows);
UpdateHeader(header, HEADER_RS_XLL, xll);
UpdateHeader(header, HEADER_RS_YLL, yll);
UpdateHeader(header, HEADER_RS_CELLSIZE, cellsize);
UpdateHeader(header, HEADER_RS_NODATA, nodata);
UpdateHeader(header, HEADER_RS_LAYERS, 1);
UpdateHeader(header, HEADER_RS_CELLSNUM, cols * rows);

return true;
}


bool WriteAscHeaders(const string& filename, const STRDBL_MAP& header);


template <typename T>
bool WriteSingleAsc(const string& filename, const STRDBL_MAP& header, T* values) {
string abs_filename = GetAbsolutePath(filename);
string dirname = GetPathFromFullName(abs_filename);
if (!PathExists(dirname)) { MakeDirectory(dirname); }
if (!WriteAscHeaders(abs_filename, header)) { return false; }
std::ofstream raster_file(abs_filename.c_str(), std::ios::app | std::ios::out);
if (!raster_file.is_open()) {
StatusMessage("Error opening file: " + abs_filename);
return false;
}
int rows = CVT_INT(header.at(HEADER_RS_NROWS));
int cols = CVT_INT(header.at(HEADER_RS_NCOLS));
for (int i = 0; i < rows; i++) {
for (int j = 0; j < cols; j++) {
raster_file << setprecision(6) << values[i * cols + j] << " ";
}
raster_file << endl;
}
raster_file << endl;
raster_file.close();
return true;
}

#ifdef USE_GDAL

template <typename T>
bool ReadRasterFileByGdal(const string& filename, STRDBL_MAP& header, T*& values,
RasterDataType& in_type, string& srs) {
StatusMessage(("Read " + filename + "...").c_str());
GDALDataset* po_dataset = static_cast<GDALDataset*>(GDALOpen(filename.c_str(),
GA_ReadOnly));
if (nullptr == po_dataset) {
StatusMessage("Open file " + filename + " failed.");
return false;
}
GDALRasterBand* po_band = po_dataset->GetRasterBand(1);
int n_rows = po_band->GetYSize();
int n_cols = po_band->GetXSize();
int get_value_flag = false;
double nodata = po_band->GetNoDataValue(&get_value_flag);
int approx_minmax = false;
double minmax[2];
T* tmprasterdata = nullptr;
bool read_as_signedbyte = false;
signed char* char_data = nullptr; 
unsigned char* uchar_data = nullptr;
vuint16_t* uint16_data = nullptr; 
vint16_t* int16_data = nullptr;   
vuint32_t* uint32_data = nullptr; 
vint32_t* int32_data = nullptr;   
float* float_data = nullptr;
double* double_data = nullptr;
CPLErr result;
switch (po_band->GetRasterDataType()) {
case GDT_Byte:
po_band->ComputeRasterMinMax(approx_minmax, minmax);
if ((minmax[1] <= 127 && minmax[0] < 0)
|| (minmax[1] <= 127 && minmax[0] >= 0 && (!get_value_flag || get_value_flag && nodata < 0))) {
read_as_signedbyte = true;
}
uchar_data = static_cast<unsigned char*>(CPLMalloc(sizeof(unsigned char) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, uchar_data,
n_cols, n_rows, GDT_Byte, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
if (read_as_signedbyte) {
StatusMessage("Read GDT_Byte raster as signed char!");
Initialize1DArray(n_rows * n_cols, char_data, uchar_data);
Initialize1DArray(n_rows * n_cols, tmprasterdata, char_data);
Release1DArray(char_data);
in_type = RDT_Int8;
} else {
StatusMessage("Read GDT_Byte raster as unsigned char!");
Initialize1DArray(n_rows * n_cols, tmprasterdata, uchar_data);
in_type = RDT_UInt8;
}
CPLFree(uchar_data);
break;
case GDT_UInt16:
uint16_data = static_cast<vuint16_t*>(CPLMalloc(sizeof(vuint16_t) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, uint16_data,
n_cols, n_rows, GDT_UInt16, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
Initialize1DArray(n_rows * n_cols, tmprasterdata, uint16_data);
CPLFree(uint16_data);
in_type = RDT_UInt16;
break;
case GDT_Int16:
int16_data = static_cast<vint16_t*>(CPLMalloc(sizeof(vint16_t) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, int16_data,
n_cols, n_rows, GDT_Int16, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
Initialize1DArray(n_rows * n_cols, tmprasterdata, int16_data);
CPLFree(int16_data);
in_type = RDT_Int16;
break;
case GDT_UInt32:
uint32_data = static_cast<vuint32_t*>(CPLMalloc(sizeof(vuint32_t) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, uint32_data,
n_cols, n_rows, GDT_UInt32, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
Initialize1DArray(n_rows * n_cols, tmprasterdata, uint32_data);
CPLFree(uint32_data);
in_type = RDT_UInt32;
break;
case GDT_Int32:
int32_data = static_cast<vint32_t*>(CPLMalloc(sizeof(vint32_t) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, int32_data,
n_cols, n_rows, GDT_Int32, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
Initialize1DArray(n_rows * n_cols, tmprasterdata, int32_data);
CPLFree(int32_data);
in_type = RDT_Int32;
break;
case GDT_Float32:
float_data = static_cast<float*>(CPLMalloc(sizeof(float) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, float_data,
n_cols, n_rows, GDT_Float32, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
Initialize1DArray(n_rows * n_cols, tmprasterdata, float_data);
CPLFree(float_data);
in_type = RDT_Float;
break;
case GDT_Float64:
double_data = static_cast<double*>(CPLMalloc(sizeof(double) * n_cols * n_rows));
result = po_band->RasterIO(GF_Read, 0, 0, n_cols, n_rows, double_data,
n_cols, n_rows, GDT_Float64, 0, 0);
if (result != CE_None) {
StatusMessage("RaterIO trouble: " + string(CPLGetLastErrorMsg()));
GDALClose(po_dataset);
return false;
}
Initialize1DArray(n_rows * n_cols, tmprasterdata, double_data);
CPLFree(double_data);
in_type = RDT_Double;
break;
default:
StatusMessage("Unexpected GDALDataType: " + string(RasterDataTypeToString(in_type)));
GDALClose(po_dataset);
return false;
}

if (!get_value_flag) { 
nodata = DefaultNoDataByType(in_type);
}

double geo_trans[6];
po_dataset->GetGeoTransform(geo_trans);
UpdateHeader(header, HEADER_RS_NCOLS, n_cols);
UpdateHeader(header, HEADER_RS_NROWS, n_rows);
UpdateHeader(header, HEADER_RS_NODATA, nodata);
UpdateHeader(header, HEADER_RS_CELLSIZE, geo_trans[1]);
UpdateHeader(header, HEADER_RS_XLL, geo_trans[0] + 0.5 * geo_trans[1]);
UpdateHeader(header, HEADER_RS_YLL, geo_trans[3] + (n_rows - 0.5) * geo_trans[5]);
UpdateHeader(header, HEADER_RS_LAYERS, 1.);
UpdateHeader(header, HEADER_RS_CELLSNUM, n_cols * n_rows);
srs = string(po_dataset->GetProjectionRef());

GDALClose(po_dataset);

values = tmprasterdata;
return true;
}


template <typename T>
bool WriteSingleGeotiff(const string& filename, const STRDBL_MAP& header,
const STRING_MAP& opts, T* values) {
int n_rows = CVT_INT(header.at(HEADER_RS_NROWS));
int n_cols = CVT_INT(header.at(HEADER_RS_NCOLS));
char** papsz_options = nullptr;
void* new_values = nullptr;
bool recreate_flag = true; 
double old_nodata = header.at(HEADER_RS_NODATA);
bool change_nodata = false;
double new_nodata = old_nodata; 
bool convert_permit = true; 
RasterDataType tmptype = TypeToRasterDataType(typeid(T));
RasterDataType outtype = RasterDataTypeInOptionals(opts);
if (outtype == RDT_Unknown) {
outtype = tmptype;
recreate_flag = false;
}
else if (outtype == tmptype && outtype != RDT_Int8) { 
recreate_flag = false;
}
else {
if (outtype == RDT_UInt8) { 
new_values = static_cast<unsigned char*>(CPLMalloc(sizeof(unsigned char) * n_cols * n_rows));
unsigned char* values_uchar = static_cast<unsigned char*>(new_values);
if (old_nodata < 0 || old_nodata > UINT8_MAX) {
new_nodata = UINT8_MAX;
change_nodata = true;
}
int illegal_count = 0;
#pragma omp parallel for reduction(+:illegal_count)
for (int i = 0; i < n_cols * n_rows; i++) {
if (FloatEqual(values[i], old_nodata) && change_nodata) {
values_uchar[i] = UINT8_MAX;
continue;
}
if (values[i] < 0 || values[i] > UINT8_MAX) illegal_count += 1;
values_uchar[i] = static_cast<unsigned char>(values[i]);
}
if (illegal_count > 0) convert_permit = false;
}
else if (outtype == RDT_Int8) { 
papsz_options = CSLSetNameValue(papsz_options, "PIXELTYPE", "SIGNEDBYTE");
new_values = static_cast<signed char*>(CPLMalloc(sizeof(signed char) * n_cols * n_rows));
signed char* values_char = static_cast<signed char*>(new_values);
if (old_nodata < INT8_MIN || old_nodata > INT8_MAX) {
new_nodata = INT8_MIN;
change_nodata = true;
}
int illegal_count = 0;
#pragma omp parallel for reduction(+:illegal_count)
for (int i = 0; i < n_cols * n_rows; i++) {
if (FloatEqual(values[i], old_nodata) && change_nodata) {
values_char[i] = INT8_MIN;
continue;
}
if (values[i] < INT8_MIN || values[i] > INT8_MAX) { illegal_count += 1; }
values_char[i] = static_cast<signed char>(values[i]);
}
if (illegal_count > 0) convert_permit = false;
}
else if (outtype == RDT_UInt16) { 
new_values = static_cast<vuint16_t*>(CPLMalloc(sizeof(vuint16_t) * n_cols * n_rows));
vuint16_t* values_uint16 = static_cast<vuint16_t*>(new_values);
if (old_nodata < 0 || old_nodata > UINT16_MAX) {
new_nodata = UINT16_MAX;
change_nodata = true;
}
int illegal_count = 0;
#pragma omp parallel for reduction(+:illegal_count)
for (int i = 0; i < n_cols * n_rows; i++) {
if (FloatEqual(values[i], old_nodata) && change_nodata) {
values_uint16[i] = UINT16_MAX;
continue;
}
if (values[i] < 0 || values[i] > UINT16_MAX) illegal_count += 1;
values_uint16[i] = static_cast<vuint16_t>(values[i]);
}
if (illegal_count > 0) convert_permit = false;
}
else if (outtype == RDT_Int16) { 
new_values = static_cast<vint16_t*>(CPLMalloc(sizeof(vint16_t) * n_cols * n_rows));
vint16_t* values_int16 = static_cast<vint16_t*>(new_values);
if (old_nodata < INT16_MIN || old_nodata > INT16_MAX) {
new_nodata = INT16_MIN;
change_nodata = true;
}
int illegal_count = 0;
#pragma omp parallel for reduction(+:illegal_count)
for (int i = 0; i < n_cols * n_rows; i++) {
if (FloatEqual(values[i], old_nodata) && change_nodata) {
values_int16[i] = INT16_MIN;
continue;
}
if (values[i] < INT16_MIN || values[i] > INT16_MAX) illegal_count += 1;
values_int16[i] = static_cast<vint16_t>(values[i]);
}
if (illegal_count > 0) convert_permit = false;
}
else if (outtype == RDT_UInt32) { 
new_values = static_cast<vuint32_t*>(CPLMalloc(sizeof(vuint32_t) * n_cols * n_rows));
vuint32_t* values_uint32 = static_cast<vuint32_t*>(new_values);
if (old_nodata < 0 || old_nodata > UINT32_MAX) {
new_nodata = UINT32_MAX;
change_nodata = true;
}
int illegal_count = 0;
#pragma omp parallel for reduction(+:illegal_count)
for (int i = 0; i < n_cols * n_rows; i++) {
if (FloatEqual(values[i], old_nodata) && change_nodata) {
values_uint32[i] = UINT32_MAX;
continue;
}
if (values[i] < 0 || values[i] > UINT32_MAX) illegal_count += 1;
values_uint32[i] = static_cast<vuint32_t>(values[i]);
}
if (illegal_count > 0) convert_permit = false;
}
else if (outtype == RDT_Int32) { 
new_values = static_cast<vint32_t*>(CPLMalloc(sizeof(vint32_t) * n_cols * n_rows));
vint32_t* values_int32 = static_cast<vint32_t*>(new_values);
if (old_nodata < INT32_MIN || old_nodata > INT32_MAX) {
new_nodata = INT32_MIN;
change_nodata = true;
}
int illegal_count = 0;
#pragma omp parallel for reduction(+:illegal_count)
for (int i = 0; i < n_cols * n_rows; i++) {
if (FloatEqual(values[i], old_nodata) && change_nodata) {
values_int32[i] = INT32_MIN;
continue;
}
if (values[i] < INT32_MIN || values[i] > INT32_MAX) illegal_count += 1;
values_int32[i] = static_cast<vint32_t>(values[i]);
}
if (illegal_count > 0) convert_permit = false;
}
else if (outtype == RDT_Float) {
new_values = static_cast<float*>(CPLMalloc(sizeof(float) * n_cols * n_rows));
float* values_float = static_cast<float*>(new_values);
#pragma omp parallel for
for (int i = 0; i < n_cols * n_rows; i++) values_float[i] = static_cast<float>(values[i]);
}
else if (outtype == RDT_Double) {
new_values = static_cast<double*>(CPLMalloc(sizeof(double) * n_cols * n_rows));
double* values_double = static_cast<double*>(new_values);
#pragma omp parallel for
for (int i = 0; i < n_cols * n_rows; i++) values_double[i] = static_cast<double>(values[i]);
}
}
if ((outtype == RDT_Unknown || (recreate_flag && nullptr == new_values))
|| (!convert_permit)) {
cout << "Error: The specific raster output data type is not allowed!\n";
if (papsz_options != nullptr) { CSLDestroy(papsz_options); }
if (new_values != nullptr) { CPLFree(new_values); }
return false;
}
GDALDriver* po_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
if (nullptr == po_driver) { return false; }
string dirname = GetPathFromFullName(filename);
if (!PathExists(dirname)) { MakeDirectory(dirname); }
GDALDataset* po_dst_ds = po_driver->Create(filename.c_str(), n_cols, n_rows, 1,
CvtToGDALDataType(outtype), papsz_options);
if (nullptr == po_dst_ds) { return false; }
GDALRasterBand* po_dst_band = po_dst_ds->GetRasterBand(1);
CPLErr result = CE_None;
if (!recreate_flag) {
result = po_dst_band->RasterIO(GF_Write, 0, 0, n_cols, n_rows, values,
n_cols, n_rows, CvtToGDALDataType(outtype), 0, 0);
}
else {
result = po_dst_band->RasterIO(GF_Write, 0, 0, n_cols, n_rows, new_values,
n_cols, n_rows, CvtToGDALDataType(outtype), 0, 0);
}
if (result != CE_None || nullptr == po_dst_band) {
StatusMessage("RaterIO Error: " + string(CPLGetLastErrorMsg()));
if (papsz_options != nullptr) { CSLDestroy(papsz_options); }
if (new_values != nullptr) { CPLFree(new_values); }
GDALClose(po_dst_ds);
return false;
}
po_dst_band->SetNoDataValue(new_nodata);

double geo_trans[6]; 
geo_trans[0] = header.at(HEADER_RS_XLL) - 0.5 * header.at(HEADER_RS_CELLSIZE);
geo_trans[1] = header.at(HEADER_RS_CELLSIZE);
geo_trans[2] = 0.;
geo_trans[3] = header.at(HEADER_RS_YLL) + (n_rows - 0.5) * header.at(HEADER_RS_CELLSIZE);
geo_trans[4] = 0.;
geo_trans[5] = -header.at(HEADER_RS_CELLSIZE);
po_dst_ds->SetGeoTransform(geo_trans);
if (opts.find(HEADER_RS_SRS) == opts.end()) {
po_dst_ds->SetProjection("");
} else {
po_dst_ds->SetProjection(opts.at(HEADER_RS_SRS).c_str());
}
if (papsz_options != nullptr) { CSLDestroy(papsz_options); }
if (new_values != nullptr) { CPLFree(new_values); }
GDALClose(po_dst_ds);

return true;
}
#endif 


template <typename T>
bool WriteRasterToFile(const string& filename, const STRDBL_MAP& header,
const STRING_MAP& opts, T* values) {
if (GetPathFromFullName(filename).empty()) { return false; }
string abs_filename = GetAbsolutePath(filename);
string filetype = GetUpper(GetSuffix(abs_filename));
if (StringMatch(filetype, ASCIIExtension)) {
return WriteSingleAsc(abs_filename, header, values);
}
#ifdef USE_GDAL
if (StringMatch(filetype, GTiffExtension)) {
return WriteSingleGeotiff(abs_filename, header, opts, values);
}
return WriteSingleGeotiff(ReplaceSuffix(abs_filename, GTiffExtension),
header, opts, values);
#else
StatusMessage("Warning: Without GDAL, ASC file will be exported as default!");
return WriteSingleAsc(ReplaceSuffix(abs_filename, ASCIIExtension), header, values);
#endif 
}

#ifdef USE_MONGODB

template <typename T>
bool ReadGridFsFile(MongoGridFs* gfs, const string& filename,
T*& data, STRDBL_MAP& header,
STRING_MAP& header_str,
const STRING_MAP& opts ) {
vint length;
char* buf = nullptr;
if (!gfs->GetStreamData(filename, buf, length, nullptr, opts) ||
nullptr == buf) {
return false;
}
bson_t* bmeta = gfs->GetFileMetadata(filename, nullptr, opts);

bson_iter_t iter; 
if (nullptr != bmeta && bson_iter_init(&iter, bmeta)) {
while (bson_iter_next(&iter)) {
const char* key = bson_iter_key(&iter);
if (header.find(key) != header.end()) {
GetNumericFromBsonIterator(&iter, header[key]);
}
else {
header_str[key] = GetStringFromBsonIterator(&iter);
}
}
}
bson_destroy(bmeta); 

int n_rows = CVT_INT(header.at(HEADER_RS_NROWS));
int n_cols = CVT_INT(header.at(HEADER_RS_NCOLS));
int n_lyrs = CVT_INT(header.at(HEADER_RS_LAYERS));
int n_cells = CVT_INT(header.at(HEADER_RS_CELLSNUM));
if (n_rows < 0 || n_cols < 0 || n_lyrs < 0) { 
delete[] buf;
return false;
}
int value_count = n_cells * n_lyrs;
size_t size_dtype = length / value_count;

RasterDataType rstype = RDT_Unknown;
if (header_str.find(HEADER_RSOUT_DATATYPE) != header_str.end()) {
rstype = StringToRasterDataType(header_str.at(HEADER_RSOUT_DATATYPE));
}
if (rstype == RDT_Unknown) {
StatusMessage("Unknown data type in MongoDB GridFS!");
delete[] buf;
return false;
}

if (rstype == RDT_Double && size_dtype == sizeof(double)) {
double* data_dbl = reinterpret_cast<double*>(buf);
Initialize1DArray(value_count, data, data_dbl);
Release1DArray(data_dbl);
}
else if (rstype == RDT_Float && size_dtype == sizeof(float)) {
float* data_flt = reinterpret_cast<float*>(buf);
Initialize1DArray(value_count, data, data_flt);
Release1DArray(data_flt);
}
else if (rstype == RDT_Int32 && size_dtype == sizeof(vint32_t)) {
vint32_t* data_int32 = reinterpret_cast<vint32_t*>(buf);
Initialize1DArray(value_count, data, data_int32);
Release1DArray(data_int32);
}
else if (rstype == RDT_UInt32 && size_dtype == sizeof(vuint32_t)) {
vuint32_t* data_uint32 = reinterpret_cast<vuint32_t*>(buf);
Initialize1DArray(value_count, data, data_uint32);
Release1DArray(data_uint32);
}
else if (rstype == RDT_Int16 && size_dtype == sizeof(vint16_t)) {
vint16_t* data_int16 = reinterpret_cast<vint16_t*>(buf);
Initialize1DArray(value_count, data, data_int16);
Release1DArray(data_int16);
}
else if (rstype == RDT_UInt16 && size_dtype == sizeof(vuint16_t)) {
vuint16_t* data_uint16 = reinterpret_cast<vuint16_t*>(buf);
Initialize1DArray(value_count, data, data_uint16);
Release1DArray(data_uint16);
}
else if (rstype == RDT_Int8 && size_dtype == sizeof(vint8_t)) {
vint8_t* data_int8 = reinterpret_cast<vint8_t*>(buf);
Initialize1DArray(value_count, data, data_int8);
Release1DArray(data_int8);
}
else if (rstype == RDT_UInt8 && size_dtype == sizeof(vuint8_t)) {
vuint8_t* data_uint8 = reinterpret_cast<vuint8_t*>(buf);
Initialize1DArray(value_count, data, data_uint8);
Release1DArray(data_uint8);
}
else {
StatusMessage("Unconsistent of data type and size!");
delete[] buf;
return false;
}
return true;
}


template <typename T>
bool WriteStreamDataAsGridfs(MongoGridFs* gfs, const string& filename,
STRDBL_MAP& header, T* values, const int datalength,
const STRING_MAP& opts = STRING_MAP()) {
STRING_MAP curopts;
CopyStringMap(opts, curopts);
RasterDataType temp_type = TypeToRasterDataType(typeid(T));
if (curopts.find(HEADER_RSOUT_DATATYPE) == curopts.end()) {
UpdateStringMap(curopts, HEADER_RSOUT_DATATYPE, RasterDataTypeToString(temp_type));
}
gfs->RemoveFile(filename, nullptr, curopts);
bson_t p = BSON_INITIALIZER;
double intpart; 
for (auto iter = header.begin(); iter != header.end(); ++iter) {
if (!StringMatch(HEADER_RS_NODATA, iter->first)
&& std::modf(iter->second, &intpart) == 0.0) {
BSON_APPEND_INT32(&p, iter->first.c_str(), CVT_INT(iter->second));
}
else {
BSON_APPEND_DOUBLE(&p, iter->first.c_str(), iter->second);
}
}
char* buf = nullptr;
vint buflength = -1;
bool transformed = true;
RasterDataType opt_type = StringToRasterDataType(curopts.at(HEADER_RSOUT_DATATYPE));
if (opt_type == temp_type) {
buf = reinterpret_cast<char*>(values);
buflength = datalength * sizeof(T);
transformed = false;
} else {
if (opt_type == RDT_UInt8) {
vuint8_t* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(vuint8_t);
} else if (opt_type == RDT_Int8) {
vint8_t* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(vint8_t);
} else if (opt_type == RDT_UInt16) {
vuint16_t* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(vuint16_t);
} else if (opt_type == RDT_Int16) {
vint16_t* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(vint16_t);
} else if (opt_type == RDT_UInt32) {
vuint32_t* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(vuint32_t);
} else if (opt_type == RDT_Int32) {
vint32_t* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(vint32_t);
} else if (opt_type == RDT_Float) {
float* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(float);
} else if (opt_type == RDT_Double) {
double* newdata = nullptr;
Initialize1DArray(datalength, newdata, values);
buf = reinterpret_cast<char*>(newdata);
buflength = datalength * sizeof(double);
}
}
AppendStringOptionsToBson(&p, curopts);

int try_times = 0;
bool gstatus = false;
while (try_times <= 3) { 
gstatus = gfs->WriteStreamData(filename, buf, buflength, &p);
if (gstatus) { break; }
SleepMs(2); 
try_times++;
}
bson_destroy(&p);
if (transformed) { delete[] buf; }
return gstatus;
}

#endif 


class SubsetPositions: NotCopyable {
public:
SubsetPositions();

SubsetPositions(int srow, int erow, int scol, int ecol);

explicit SubsetPositions(SubsetPositions*& src, bool deep_copy = false);

~SubsetPositions();

bool Initialization();

template <typename T>
bool SetData(const int n, T* data) {
if (n != n_cells) { return false; }
if (nullptr == data) { return false; }
if (1 != n_lyrs) { n_lyrs = 1; }
if (nullptr != data_) {
for (int i = 0; i < n_cells; i++) { data_[i] = CVT_DBL(data[i]); }
} else {
Initialize1DArray(n_cells, data_, data);
}
usable = true;
return true;
}

template <typename T>
bool Set2DData(const int n, const int lyr, T** data2d) {
if (n != n_cells) { return false; }
if (nullptr == data2d) { return false; }
if (lyr != n_lyrs) { n_lyrs = lyr; }
if (nullptr != data2d_) {
for (int i = 0; i < n_cells; i++) {
for (int j = 0; j < n_lyrs; j++) {
data2d_[i][j] = CVT_DBL(data2d[i][j]);
}
}
} else {
Initialize2DArray(n_cells, n_lyrs, data2d_, data2d);
}
usable = true;
return true;
}
#ifdef USE_MONGODB
bool ReadFromMongoDB(MongoGridFs* gfs, const string& fname,
const STRING_MAP& opts = STRING_MAP());
#endif
void GetHeader(double gxll, double gyll, int gnrows, double cellsize,
double nodata, STRDBL_MAP& subheader);
template <typename T>
void Output(T nodata, vector<T*>& fulldata) {
if (nullptr == data_ && nullptr == data2d_) { return; }
int nrows = g_erow - g_srow + 1;
int ncols = g_ecol - g_scol + 1;
int fullsize = nrows * ncols;
for (int ilyr = 0; ilyr < n_lyrs; ilyr++) {
T* tmpdata = nullptr;
Initialize1DArray(fullsize, tmpdata, nodata);
for (int vi = 0; vi < n_cells; vi++) {
int j = local_pos_[vi][0] * ncols + local_pos_[vi][1];
if (n_lyrs > 1 && nullptr != data2d_) {
tmpdata[j] = static_cast<T>(data2d_[vi][ilyr]);
}
else if (n_lyrs == 1 && nullptr != data_) {
tmpdata[j] = static_cast<T>(data_[vi]);
}
}
fulldata.emplace_back(tmpdata);
}
}

bool usable; 
int n_cells; 
int n_lyrs; 
int g_srow; 
int g_erow; 
int g_scol; 
int g_ecol; 
bool alloc_; 
int** local_pos_; 
int* global_; 
double* data_; 
double** data2d_; 
};


template <typename T, typename MASK_T = T>
class clsRasterData {
public:



explicit clsRasterData(bool is_2d = false);


clsRasterData(T* data, int cols, int rows, T nodata, double dx, double xll,
double yll, const STRING_MAP& opts = STRING_MAP());


clsRasterData(T* data, int cols, int rows, T nodata, double dx, double xll,
double yll, const string& srs);


clsRasterData(T** data2d, int cols, int rows, int nlayers, T nodata,
double dx, double xll, double yll, const STRING_MAP& opts = STRING_MAP());


clsRasterData(T** data2d, int cols, int rows, int nlayers, T nodata,
double dx, double xll, double yll, const string& srs);


explicit clsRasterData(const string& filename, bool calc_pos = false,
clsRasterData<MASK_T>* mask = nullptr, bool use_mask_ext = true,
double default_value = NODATA_VALUE, const STRING_MAP& opts = STRING_MAP());


static clsRasterData<T, MASK_T>* Init(const string& filename, bool calc_pos = false,
clsRasterData<MASK_T>* mask = nullptr, bool use_mask_ext = true,
double default_value = NODATA_VALUE, const STRING_MAP& opts = STRING_MAP());


explicit clsRasterData(vector<string>& filenames, bool calc_pos = false,
clsRasterData<MASK_T>* mask = nullptr, bool use_mask_ext = true,
double default_value = NODATA_VALUE, const STRING_MAP& opts = STRING_MAP());


static clsRasterData<T, MASK_T>* Init(vector<string>& filenames, bool calc_pos = false,
clsRasterData<MASK_T>* mask = nullptr, bool use_mask_ext = true,
double default_value = NODATA_VALUE, const STRING_MAP& opts = STRING_MAP());


clsRasterData(clsRasterData<MASK_T>* mask, T* values, int len,
const STRING_MAP& opts = STRING_MAP());


clsRasterData(clsRasterData<MASK_T>* mask, T** values, int len, int lyrs,
const STRING_MAP& opts = STRING_MAP());

#ifdef USE_MONGODB


clsRasterData(MongoGridFs* gfs, const char* remote_filename, bool calc_pos = false,
clsRasterData<MASK_T>* mask = nullptr, bool use_mask_ext = true,
double default_value = NODATA_VALUE, const STRING_MAP& opts = STRING_MAP());


static clsRasterData<T, MASK_T>* Init(MongoGridFs* gfs, const char* remote_filename,
bool calc_pos = false, clsRasterData<MASK_T>* mask = nullptr,
bool use_mask_ext = true, double default_value = NODATA_VALUE,
const STRING_MAP& opts = STRING_MAP());
#endif


explicit clsRasterData(clsRasterData<T, MASK_T>* another);

~clsRasterData();



bool ReadFromFile(const string& filename, bool calc_pos = false, clsRasterData<MASK_T>* mask = nullptr,
bool use_mask_ext = true, double default_value = NODATA_VALUE,
const STRING_MAP& opts = STRING_MAP());

bool ReadFromFiles(vector<string>& filenames, bool calc_pos = false, clsRasterData<MASK_T>* mask = nullptr,
bool use_mask_ext = true, double default_value = NODATA_VALUE,
const STRING_MAP& opts = STRING_MAP());

#ifdef USE_MONGODB

bool ReadFromMongoDB(MongoGridFs* gfs, const string& filename, bool calc_pos = false,
clsRasterData<MASK_T>* mask = nullptr, bool use_mask_ext = true,
double default_value = NODATA_VALUE,
const STRING_MAP& opts = STRING_MAP());

#endif 



bool OutputToFile(const string& filename, bool out_origin = true);


bool OutputSubsetToFile(bool out_origin = false, bool out_combined = true,
const string& outname = string(),
const map<vint, vector<double> >& recls = map<vint, vector<double> >(),
double default_value = NODATA_VALUE);


bool OutputAscFile(const string& filename);

#ifdef USE_GDAL

bool OutputFileByGdal(const string& filename);
#endif 

#ifdef USE_MONGODB


bool OutputToMongoDB(MongoGridFs* gfs, const string& filename = string(),
const STRING_MAP& opts = STRING_MAP(),
bool include_nodata = true,
bool out_origin = true);


bool OutputSubsetToMongoDB(MongoGridFs* gfs, const string& filename = string(),
const STRING_MAP& opts = STRING_MAP(),
bool include_nodata = true,
bool out_origin = false, bool out_combined = true,
const map<vint, vector<double> >& recls = map<vint, vector<double> >(),
double default_value = NODATA_VALUE);

#endif 





void SetHeader(const STRDBL_MAP& refers);

void SetCoreName(const string& name) { core_name_ = name; }


void SetValue(int row, int col, T value, int lyr = 1);


bool SetCalcPositions();


bool SetPositions(int len, int** pdata);


bool SetUseMaskExt();

void SetDataType(RasterDataType const type) { rs_type_ = type; }

void SetDataType(const string& strtype) { rs_type_ = StringToRasterDataType(strtype); }

void SetOutDataType(RasterDataType type);

void SetOutDataType(const string& strtype);

void SetDefaultValue(const double defaultv) { default_value_ = defaultv; }


bool BuildSubSet(map<int, int> groups = map<int, int>());


bool ReleaseSubset();


bool RebuildSubSet(map<int, int> groups = map<int, int>());






void CalculateStatistics();


void UpdateStatistics();


void ReleaseStatsMap2D();


double GetStatistics(string sindex, int lyr = 1);


void GetStatistics(string sindex, int* lyrnum, double** values);


double GetAverage(const int lyr = 1) { return GetStatistics(STATS_RS_MEAN, lyr); }


double GetMaximum(const int lyr = 1) { return GetStatistics(STATS_RS_MAX, lyr); }


double GetMinimum(const int lyr = 1) { return GetStatistics(STATS_RS_MIN, lyr); }


double GetStd(const int lyr = 1) { return GetStatistics(STATS_RS_STD, lyr); }


double GetRange(const int lyr = 1) { return GetStatistics(STATS_RS_RANGE, lyr); }


void GetAverage(int* lyrnum, double** values) { GetStatistics(STATS_RS_MEAN, lyrnum, values); }


void GetMaximum(int* lyrnum, double** values) { GetStatistics(STATS_RS_MAX, lyrnum, values); }


void GetMinimum(int* lyrnum, double** values) { GetStatistics(STATS_RS_MIN, lyrnum, values); }


void GetStd(int* lyrnum, double** values) { GetStatistics(STATS_RS_STD, lyrnum, values); }


void GetRange(int* lyrnum, double** values) { GetStatistics(STATS_RS_RANGE, lyrnum, values); }


void GetValidNumber(int* lyrnum, double** values) { GetStatistics(STATS_RS_VALIDNUM, lyrnum, values); }


int GetValidNumber(const int lyr = 1) { return CVT_INT(GetStatistics(STATS_RS_VALIDNUM, lyr)); }

int GetCellNumber() const { return n_cells_; } 
int GetDataLength() const { return n_cells_ < 0 || n_lyrs_ < 0 ? -1 : n_cells_ * n_lyrs_; }
int GetCols() const { return CVT_INT(headers_.at(HEADER_RS_NCOLS)); } 
int GetRows() const { return CVT_INT(headers_.at(HEADER_RS_NROWS)); } 
double GetCellWidth() const { return headers_.at(HEADER_RS_CELLSIZE); } 

double GetXllCenter() const {
if (headers_.find(HEADER_RS_XLL) != headers_.end()) {
return headers_.at(HEADER_RS_XLL);
}
if (headers_.find(HEADER_RS_XLLCOR) != headers_.end()) {
return headers_.at(HEADER_RS_XLLCOR) + 0.5 * GetCellWidth();
}
return NODATA_VALUE;
}

double GetYllCenter() const {
if (headers_.find(HEADER_RS_YLL) != headers_.end()) {
return headers_.at(HEADER_RS_YLL);
}
if (headers_.find(HEADER_RS_YLLCOR) != headers_.end()) {
return headers_.at(HEADER_RS_YLLCOR) + 0.5 * GetCellWidth();
}
return NODATA_VALUE;
}

int GetLayers() const { return n_lyrs_; } 
RasterDataType GetDataType() const { return rs_type_; } 
RasterDataType GetOutDataType() const { return rs_type_out_; } 
T GetNoDataValue() const { return no_data_value_; } 
double GetDefaultValue() const { return default_value_; } 


int GetPosition(int row, int col);

int GetPosition(float x, float y);

int GetPosition(double x, double y);

map<int, SubsetPositions*>& GetSubset() { return subset_; }


bool GetRasterData(int* n_cells, T** data);


bool Get2DRasterData(int* n_cells, int* n_lyrs, T*** data);

const STRDBL_MAP& GetRasterHeader() const { return headers_; }

const STRDBL_MAP& GetStatistics() const { return stats_; }

const map<string, double *>& GetStatistics2D() const { return stats_2d_; };

string GetFilePath() const { return full_path_; }

string GetCoreName() const { return core_name_; }


void GetRasterPositionData(int* datalength, int*** positiondata);

T* GetRasterDataPointer() const { return raster_; } 
int** GetRasterPositionDataPointer() const { return pos_data_; } 
T** Get2DRasterDataPointer() const { return raster_2d_; } 
const char* GetSrs(); 
string GetSrsString(); 
string GetOption(const char* key); 
const STRING_MAP& GetOptions() const { return options_; } 


T GetValueByIndex(int cell_index, int lyr = 1);


void GetValueByIndex(int cell_index, T*& values);


T GetValue(int row, int col, int lyr = 1);


void GetValue(int row, int col, T*& values);


bool IsNoData(const int row, const int col, const int lyr = 1) {
return FloatEqual(GetValue(row, col, lyr), no_data_value_);
}

bool Is2DRaster() const { return is_2draster; } 
bool PositionsCalculated() const { return calc_pos_; } 
bool PositionsAllocated() const { return store_pos_; } 
bool MaskExtented() const { return use_mask_ext_; } 
bool StatisticsCalculated() const { return stats_calculated_; } 
bool Initialized() const { return initialized_; } 


bool ValidateRasterData() {
if ((!is_2draster && nullptr != raster_) || 
(is_2draster && nullptr != raster_2d_)) 
{ return true; }
StatusMessage("Error: Please initialize the raster object first.");
return false;
}


bool ValidateRowCol(const int row, const int col) {
if ((row < 0 || row >= GetRows()) || (col < 0 || col >= GetCols())) {
StatusMessage("The row must between 0 and " + ValueToString(GetRows() - 1) +
", and the col must between 0 and " + ValueToString(GetCols() - 1));
return false;
}
return true;
}


bool ValidateLayer(const int lyr) {
if (lyr <= 0 || lyr > n_lyrs_) {
StatusMessage("The layer must be 1 ");
if (n_lyrs_ > 1) StatusMessage(" or between 1 and " +
utils_string::ValueToString(n_lyrs_));
return false;
}
return true;
}


bool ValidateIndex(const int idx) {
if (idx < 0 || idx >= n_cells_) {
StatusMessage("The index must between 0 and " + utils_string::ValueToString(n_cells_ - 1));
return false;
}
return true;
}

string GetFullFileName() const { return full_path_; }

clsRasterData<MASK_T>* GetMask() const { return mask_; }


void Copy(clsRasterData<T, MASK_T>* orgraster);


void ReplaceNoData(T replacedv);


void Reclassify(map<int, T> reclass_map);




XY_COOR GetCoordinateByRowCol(int row, int col);


ROW_COL GetPositionByCoordinate(double x, double y, STRDBL_MAP* header = nullptr);

private:

void InitializeRasterClass(bool is_2d = false);


void InitializeReadFunction(const string& filename, bool calc_pos = false, clsRasterData<MASK_T>* mask = nullptr,
bool use_mask_ext = true, double default_value = NODATA_VALUE,
const STRING_MAP& opts = STRING_MAP());


bool ConstructFromSingleFile(const string& filename, bool calc_pos = false, clsRasterData<MASK_T>* mask = nullptr,
bool use_mask_ext = true, double default_value = NODATA_VALUE,
const STRING_MAP& opts = STRING_MAP());


int MaskAndCalculateValidPosition();


void CalculateValidPositionsFromGridData();


bool PrepareCombSubsetData(T**values, int* datalen, int* datalyrs,
bool out_origin = false, bool include_nodata = true,
const map<vint, vector<double> >&recls = map<vint, vector<double> >(),
double default_value = NODATA_VALUE);


bool PrepareSubsetData(int sub_id, SubsetPositions* sub,
T** values, int* datalen, int* datalyrs,
bool out_origin = false, bool include_nodata = true,
const map<vint, vector<double> >& recls = map<vint, vector<double> >(),
double default_value = NODATA_VALUE);


bool OutputFullsizeToFiles(T* fullsizedata, int fsize, int datalyrs,
const string& fullfilename, const STRDBL_MAP& header,
const STRING_MAP& opts);


void AddOtherLayerRasterData(int row, int col, int cellidx, int lyr,
STRDBL_MAP lyrheader, T* lyrdata);


void CheckDefaultValue() {
if (FloatEqual(default_value_, NODATA_VALUE) && !FloatEqual(no_data_value_, NODATA_VALUE)) {
default_value_ = CVT_DBL(no_data_value_);
}
}


clsRasterData& operator=(const clsRasterData&) { }


int n_cells_;
int n_lyrs_;
RasterDataType rs_type_;
RasterDataType rs_type_out_;
T no_data_value_;
double default_value_;
string full_path_;
string core_name_;
T* raster_;
T** raster_2d_;
int** pos_data_;
STRING_MAP options_;
STRDBL_MAP headers_;
STRDBL_MAP stats_;
map<string, double *> stats_2d_;
clsRasterData<MASK_T>* mask_;
map<int, SubsetPositions*> subset_;
bool initialized_;
bool is_2draster;
bool calc_pos_;
bool store_pos_;
bool use_mask_ext_;
bool stats_calculated_;
};


#ifndef IntRaster
#define IntRaster          clsRasterData<int>
#endif
#ifndef FltRaster
#define FltRaster          clsRasterData<float>
#endif
#ifndef DblRaster
#define DblRaster          clsRasterData<double>
#endif
#ifndef IntIntRaster
#define IntIntRaster       clsRasterData<int, int>
#endif
#ifndef FltIntRaster
#define FltIntRaster       clsRasterData<float, int>
#endif
#ifndef DblIntRaster
#define DblIntRaster       clsRasterData<double, int>
#endif







template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::InitializeRasterClass(bool is_2d ) {
full_path_ = "";
core_name_ = "";
n_cells_ = -1;
rs_type_ = RDT_Unknown;
rs_type_out_ = RDT_Unknown;
no_data_value_ = static_cast<T>(NODATA_VALUE); 
default_value_ = NODATA_VALUE;
raster_ = nullptr;
pos_data_ = nullptr;
mask_ = nullptr;
subset_ = map<int, SubsetPositions*>();
n_lyrs_ = -1;
is_2draster = is_2d;
raster_2d_ = nullptr;
calc_pos_ = false;
store_pos_ = false;
use_mask_ext_ = false;
stats_calculated_ = false;
headers_ = InitialHeader();
options_ = InitialStrHeader();
InitialStatsMap(stats_, stats_2d_);
initialized_ = true;
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::InitializeReadFunction(const string& filename, const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
if (!initialized_) { InitializeRasterClass(); }
full_path_ = filename;
mask_ = mask;
calc_pos_ = calc_pos;
use_mask_ext_ = use_mask_ext;
if (nullptr == mask_) { use_mask_ext_ = false; }
default_value_ = default_value;
rs_type_out_ = RasterDataTypeInOptionals(opts);
core_name_ = GetCoreFileName(full_path_);
CopyStringMap(opts, options_);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(bool is_2d ) {
InitializeRasterClass(is_2d);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(T* data, int cols, int rows, T nodata,
double dx, double xll, double yll,
const string& srs) {
STRING_MAP opts = InitialStrHeader();
UpdateStrHeader(opts, HEADER_RS_SRS, srs);
clsRasterData(data, cols, rows, nodata, dx, xll, yll, opts);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(T* data, const int cols, const int rows, T nodata,
const double dx, const double xll, const double yll,
const STRING_MAP& opts ) {
InitializeRasterClass(false);
rs_type_out_ = RasterDataTypeInOptionals(opts);
raster_ = data;
no_data_value_ = nodata;
CopyStringMap(opts, options_);
n_cells_ = cols * rows;
n_lyrs_ = 1;
UpdateHeader(headers_, HEADER_RS_NCOLS, cols);
UpdateHeader(headers_, HEADER_RS_NROWS, rows);
UpdateHeader(headers_, HEADER_RS_XLL, xll);
UpdateHeader(headers_, HEADER_RS_YLL, yll);
UpdateHeader(headers_, HEADER_RS_CELLSIZE, dx);
UpdateHeader(headers_, HEADER_RS_NODATA, nodata);
UpdateHeader(headers_, HEADER_RS_LAYERS, 1);
UpdateHeader(headers_, HEADER_RS_CELLSNUM, n_cells_);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(T** data2d, const int cols, const int rows, const int nlayers,
T nodata, const double dx, const double xll, const double yll,
const string& srs) {
STRING_MAP opts = InitialStrHeader();
UpdateStrHeader(opts, HEADER_RS_SRS, srs);
clsRasterData(data2d, cols, rows, nlayers, nodata, dx, xll, yll, opts);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(T** data2d, const int cols, const int rows, const int nlayers,
T nodata, const double dx, const double xll, const double yll,
const STRING_MAP& opts ) {
InitializeRasterClass(true);
rs_type_out_ = RasterDataTypeInOptionals(opts);
raster_2d_ = data2d;
no_data_value_ = nodata;
CopyStringMap(opts, options_);
n_cells_ = cols * rows;
n_lyrs_ = nlayers;
headers_[HEADER_RS_NCOLS] = cols;
headers_[HEADER_RS_NROWS] = rows;
headers_[HEADER_RS_XLL] = xll;
headers_[HEADER_RS_YLL] = yll;
headers_[HEADER_RS_CELLSIZE] = dx;
headers_[HEADER_RS_NODATA] = no_data_value_;
headers_[HEADER_RS_LAYERS] = n_lyrs_;
headers_[HEADER_RS_CELLSNUM] = n_cells_;
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(const string& filename, const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
InitializeRasterClass();
rs_type_out_ = RasterDataTypeInOptionals(opts);
ReadFromFile(filename, calc_pos, mask, use_mask_ext, default_value, opts);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>* clsRasterData<T, MASK_T>::Init(const string& filename,
const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
if (!FileExists(filename)) { return nullptr; }
clsRasterData<T, MASK_T>* rs = new clsRasterData<T, MASK_T>();
rs->SetOutDataType(RasterDataTypeInOptionals(opts));
if (!rs->ReadFromFile(filename, calc_pos, mask, use_mask_ext, default_value, opts)) {
delete rs;
return nullptr;
}
return rs;
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(vector<string>& filenames,
const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
if (!FilesExist(filenames)) {
StatusMessage("Please make sure all file path existed!");
return;
}
if (filenames.size() == 1) {
ReadFromFile(filenames[0], calc_pos, mask,
use_mask_ext, default_value, opts);
}
ReadFromFiles(filenames, calc_pos, mask,
use_mask_ext, default_value, opts);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>* clsRasterData<T, MASK_T>::Init(vector<string>& filenames,
bool calc_pos ,
clsRasterData<MASK_T>* mask ,
bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
if (!FilesExist(filenames)) {
StatusMessage("Please make sure all file path existed!");
return nullptr;
}
if (filenames.size() == 1) {
return clsRasterData<T, MASK_T>::Init(filenames[0], calc_pos, mask, use_mask_ext,
default_value, opts);
}
clsRasterData<T, MASK_T>* rs2d = new clsRasterData<T, MASK_T>();
rs2d->SetOutDataType(RasterDataTypeInOptionals(opts));
if (!rs2d->ReadFromFiles(filenames, calc_pos, mask, use_mask_ext, default_value, opts)) {
delete rs2d;
return nullptr;
}
return rs2d;
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(clsRasterData<MASK_T>* mask, T* const values, const int len,
const STRING_MAP& opts ) {
InitializeRasterClass(false);
rs_type_out_ = RasterDataTypeInOptionals(opts);
calc_pos_ = false;
mask_ = mask;
use_mask_ext_ = true;
n_lyrs_ = 1;
mask->GetRasterPositionData(&n_cells_, &pos_data_);
if (n_cells_ != len) {
StatusMessage("Input data length MUST EQUALS TO valid cell's number of mask!");
initialized_ = false;
return;
}
Initialize1DArray(n_cells_, raster_, values); 
default_value_ = mask_->GetDefaultValue();
CopyHeader(mask_->GetRasterHeader(), headers_);
UpdateStrHeader(options_, HEADER_RS_SRS, mask_->GetSrsString());
CopyStringMap(opts, options_);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(clsRasterData<MASK_T>* mask, T** const values, const int len,
const int lyrs, const STRING_MAP& opts ) {
InitializeRasterClass(true);
calc_pos_ = false;
mask_ = mask;
use_mask_ext_ = true;
n_lyrs_ = lyrs;
mask->GetRasterPositionData(&n_cells_, &pos_data_);
if (n_cells_ != len) {
StatusMessage("Input data length MUST EQUALS TO valid cell's number of mask!");
initialized_ = false;
return;
}
Initialize2DArray(n_cells_, n_lyrs_, raster_2d_, values); 
CopyHeader(mask_->GetRasterHeader(), headers_);
UpdateHeader(headers_, HEADER_RS_LAYERS, n_lyrs_);
CopyStringMap(opts, options_);
UpdateStrHeader(options_, HEADER_RS_SRS, mask_->GetSrsString());
}

#ifdef USE_MONGODB

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(MongoGridFs* gfs, const char* remote_filename,
const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
InitializeRasterClass();
rs_type_out_ = RasterDataTypeInOptionals(opts);
ReadFromMongoDB(gfs, remote_filename, calc_pos, mask, use_mask_ext, default_value, opts);
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>* clsRasterData<T, MASK_T>::Init(MongoGridFs* gfs, const char* remote_filename,
const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
clsRasterData<T, MASK_T>* rs_mongo = new clsRasterData<T, MASK_T>();
rs_mongo->SetOutDataType(RasterDataTypeInOptionals(opts));
if (!rs_mongo->ReadFromMongoDB(gfs, remote_filename, calc_pos, mask, use_mask_ext, default_value, opts)) {
delete rs_mongo;
return nullptr;
}
return rs_mongo;
}

#endif 

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::ConstructFromSingleFile(const string& filename,
const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
InitializeReadFunction(filename, calc_pos, mask, use_mask_ext, default_value, opts);

bool readflag = false;
string srs = string();
if (StringMatch(GetUpper(GetSuffix(filename)), ASCIIExtension)) {
readflag = ReadAscFile(full_path_, headers_, raster_);
} else {
#ifdef USE_GDAL
readflag = ReadRasterFileByGdal(full_path_, headers_, raster_, rs_type_, srs);
#else
StatusMessage("Warning: Only ASC format is supported without GDAL!");
return false;
#endif 
}
no_data_value_ = static_cast<T>(headers_.at(HEADER_RS_NODATA));
UpdateStrHeader(options_, HEADER_RS_SRS, srs);
if (rs_type_out_ == 0) {
rs_type_out_ = rs_type_;
UpdateStrHeader(options_, HEADER_RSOUT_DATATYPE, RasterDataTypeToString(rs_type_out_));
}
CheckDefaultValue();
if (readflag) {
if (n_lyrs_ < 0) { n_lyrs_ = 1; }
if (is_2draster) { is_2draster = false; }
return MaskAndCalculateValidPosition() >= 0;
}
return false;
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::~clsRasterData() {
if (!core_name_.empty()) { StatusMessage(("Release raster: " + core_name_).c_str()); }
if (nullptr != raster_) { Release1DArray(raster_); }
if (nullptr != pos_data_ && store_pos_) { Release2DArray(pos_data_); }
if (nullptr != raster_2d_ && is_2draster) { Release2DArray(raster_2d_); }
if (is_2draster && stats_calculated_) { ReleaseStatsMap2D(); }
ReleaseSubset();
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::BuildSubSet(map<int, int> groups ) {
if (!ValidateRasterData()) { return false; }
if (nullptr == pos_data_) {
if (!SetCalcPositions()) { return false; }
}
if (!subset_.empty()) { return true; }
map<int, vector<int> > global_idx;
for (int vi = 0; vi < n_cells_; vi++) {
T curv = GetValueByIndex(vi); 
if (FloatEqual(curv, no_data_value_)) { continue; }
int groupv = CVT_INT(curv); 
if (!groups.empty() && groups.find(groupv) != groups.end()) {
groupv = groups.at(groupv); 
}
int currow = pos_data_[vi][0];
int curcol = pos_data_[vi][1];
if (subset_.find(groupv) == subset_.end()) {
#ifdef HAS_VARIADIC_TEMPLATES
subset_.emplace(groupv, new SubsetPositions(currow, currow, curcol, curcol));
#else
subset_.insert(make_pair(groupv, new SubsetPositions(currow, currow, curcol, curcol)));
#endif
}
SubsetPositions*& cursubset = subset_.at(groupv);
if (currow > cursubset->g_erow) { cursubset->g_erow = currow; }
if (currow < cursubset->g_srow) { cursubset->g_srow = currow; }
if (curcol > cursubset->g_ecol) { cursubset->g_ecol = curcol; }
if (curcol < cursubset->g_scol) { cursubset->g_scol = curcol; }
if (global_idx.find(groupv) == global_idx.end()) {
#ifdef HAS_VARIADIC_TEMPLATES
global_idx.emplace(groupv, vector<int>());
#else
global_idx.insert(make_pair(groupv, vector<int>()));
#endif
}
global_idx[groupv].emplace_back(vi);
cursubset->n_cells += 1;
}
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
Initialize1DArray(it->second->n_cells, it->second->global_, 0);
for (auto it2 = global_idx[it->first].begin(); it2 != global_idx[it->first].end(); ++it2) {
it->second->global_[it2 - global_idx[it->first].begin()] = *it2;
}
}
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
Initialize2DArray(it->second->n_cells, 2, it->second->local_pos_, -1);
for (int gidx = 0; gidx < it->second->n_cells; gidx++) {
it->second->local_pos_[gidx][0] = pos_data_[it->second->global_[gidx]][0] - it->second->g_srow;
it->second->local_pos_[gidx][1] = pos_data_[it->second->global_[gidx]][1] - it->second->g_scol;
}
it->second->alloc_ = true;
}
return true;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::ReleaseSubset() {
if (!subset_.empty()) {
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
delete it->second;
it->second = nullptr;
}
subset_.clear();
}
return true;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::RebuildSubSet(const map<int, int> groups ) {
return ReleaseSubset() && BuildSubSet(groups);
}



template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::SetOutDataType(RasterDataType const type) {
rs_type_out_ = type;
UpdateStringMap(options_, HEADER_RSOUT_DATATYPE, RasterDataTypeToString(rs_type_out_));
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::SetOutDataType(const string& strtype) {
rs_type_out_ = StringToRasterDataType(strtype);
UpdateStringMap(options_, HEADER_RSOUT_DATATYPE, RasterDataTypeToString(rs_type_out_));
}


template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::CalculateStatistics() {
if (stats_calculated_) { return; }
if (stats_.empty() || stats_2d_.empty()) { InitialStatsMap(stats_, stats_2d_); }
if (is_2draster && nullptr != raster_2d_) {
double** derivedvs = nullptr;
BasicStatistics(raster_2d_, n_cells_, n_lyrs_, &derivedvs, no_data_value_);
stats_2d_.at(STATS_RS_VALIDNUM) = derivedvs[0];
stats_2d_.at(STATS_RS_MEAN) = derivedvs[1];
stats_2d_.at(STATS_RS_MAX) = derivedvs[2];
stats_2d_.at(STATS_RS_MIN) = derivedvs[3];
stats_2d_.at(STATS_RS_STD) = derivedvs[4];
stats_2d_.at(STATS_RS_RANGE) = derivedvs[5];
Release1DArray(derivedvs);
} else {
double* derivedv = nullptr;
BasicStatistics(raster_, n_cells_, &derivedv, no_data_value_);
stats_.at(STATS_RS_VALIDNUM) = derivedv[0];
stats_.at(STATS_RS_MEAN) = derivedv[1];
stats_.at(STATS_RS_MAX) = derivedv[2];
stats_.at(STATS_RS_MIN) = derivedv[3];
stats_.at(STATS_RS_STD) = derivedv[4];
stats_.at(STATS_RS_RANGE) = derivedv[5];
Release1DArray(derivedv);
}
stats_calculated_ = true;
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::ReleaseStatsMap2D() {
for (auto it = stats_2d_.begin(); it != stats_2d_.end(); ++it) {
if (nullptr != it->second) {
Release1DArray(it->second);
}
}
stats_2d_.clear();
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::UpdateStatistics() {
if (is_2draster && stats_calculated_) ReleaseStatsMap2D();
stats_calculated_ = false;
CalculateStatistics();
}

template <typename T, typename MASK_T>
double clsRasterData<T, MASK_T>::GetStatistics(string sindex, const int lyr ) {
sindex = GetUpper(sindex);
if (!ValidateRasterData() || !ValidateLayer(lyr)) {
StatusMessage("No available raster statistics!");
return CVT_DBL(no_data_value_);
}
if (is_2draster && nullptr != raster_2d_) {
auto it = stats_2d_.find(sindex);
if (it != stats_2d_.end()) {
if (nullptr == it->second) {
stats_calculated_ = false;
CalculateStatistics();
}
return stats_2d_.at(sindex)[lyr - 1];
}
StatusMessage("WARNING: " + ValueToString(sindex) + " is not supported currently.");
return CVT_DBL(no_data_value_);
}
auto it = stats_.find(sindex);
if (it != stats_.end()) {
if (FloatEqual(it->second, CVT_DBL(NODATA_VALUE)) || !stats_calculated_) {
CalculateStatistics();
}
return stats_.at(sindex);
}
StatusMessage("WARNING: " + ValueToString(sindex) + " is not supported currently.");
return CVT_DBL(no_data_value_);
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::GetStatistics(string sindex, int* lyrnum, double** values) {
if (!is_2draster || nullptr != raster_2d_) {
StatusMessage("Please initialize the raster object first.");
*values = nullptr;
return;
}
sindex = GetUpper(sindex);
*lyrnum = n_lyrs_;
auto it = stats_2d_.find(sindex);
if (!Is2DRaster()) {
*values = nullptr;
return;
}
if (it == stats_2d_.end()) {
*values = nullptr;
StatusMessage("WARNING: " + ValueToString(sindex) + " is not supported currently.");
return;
}
if (nullptr == it->second || !stats_calculated_) {
CalculateStatistics();
}
*values = it->second;
}

template <typename T, typename MASK_T>
int clsRasterData<T, MASK_T>::GetPosition(const int row, const int col) {
if (!ValidateRasterData() || !ValidateRowCol(row, col)) {
return -2; 
}
if (!calc_pos_ || nullptr == pos_data_) {
return GetCols() * row + col;
}
for (int i = 0; i < n_cells_; i++) {
if (row == pos_data_[i][0] && col == pos_data_[i][1]) {
return i;
}
}
return -1; 
}

template <typename T, typename MASK_T>
int clsRasterData<T, MASK_T>::GetPosition(const float x, const float y) {
return GetPosition(CVT_DBL(x), CVT_DBL(y));
}

template <typename T, typename MASK_T>
int clsRasterData<T, MASK_T>::GetPosition(const double x, const double y) {
if (!initialized_) return -2;
double xll_center = GetXllCenter();
double yll_center = GetYllCenter();
double dx = GetCellWidth();
double dy = GetCellWidth();
int n_rows = GetRows();
int n_cols = GetCols();

if (FloatEqual(xll_center, CVT_DBL(NODATA_VALUE)) || FloatEqual(yll_center, CVT_DBL(NODATA_VALUE)) ||
FloatEqual(dx, CVT_DBL(NODATA_VALUE)) || n_rows < 0 || n_cols < 0) {
StatusMessage("No available header information!");
return -2;
}

double x_min = xll_center - dx / 2.;
double x_max = x_min + dx * n_cols;
if (x > x_max || x < x_min) {
return -2;
}

double y_min = yll_center - dy / 2.;
double y_max = y_min + dy * n_rows;
if (y > y_max || y < y_min) {
return -2;
}

int n_row = CVT_INT((y_max - y) / dy); 
int n_col = CVT_INT((x - x_min) / dx); 

return GetPosition(n_row, n_col);
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::GetRasterData(int* n_cells, T** data) {
if (ValidateRasterData() && !is_2draster) {
*n_cells = n_cells_;
*data = raster_;
return true;
}
*n_cells = -1;
*data = nullptr;
return false;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::Get2DRasterData(int* n_cells, int* n_lyrs, T*** data) {
if (ValidateRasterData() && is_2draster) {
*n_cells = n_cells_;
*n_lyrs = n_lyrs_;
*data = raster_2d_;
return true;
}
*n_cells = -1;
*n_lyrs = -1;
*data = nullptr;
return false;
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::GetRasterPositionData(int* datalength, int*** positiondata) {
if (nullptr != pos_data_) {
*datalength = n_cells_;
*positiondata = pos_data_;
} else {
if (!ValidateRasterData()) {
*datalength = -1;
*positiondata = nullptr;
return;
}
CalculateValidPositionsFromGridData();
*datalength = n_cells_;
*positiondata = pos_data_;
}
}

template <typename T, typename MASK_T>
const char* clsRasterData<T, MASK_T>::GetSrs() {
return GetSrsString().c_str();
}

template <typename T, typename MASK_T>
string clsRasterData<T, MASK_T>::GetSrsString() {
return GetOption(HEADER_RS_SRS);
}

template <typename T, typename MASK_T>
string clsRasterData<T, MASK_T>::GetOption(const char* key) {
if (options_.find(key) == options_.end()) {
StatusMessage((string(key) + " is not existed in the options of the raster data!").c_str());
return "";
}
return options_.at(key);
}

template <typename T, typename MASK_T>
T clsRasterData<T, MASK_T>::GetValueByIndex(const int cell_index, const int lyr ) {
if (!ValidateRasterData() || !ValidateIndex(cell_index) || !ValidateLayer(lyr)) {
return no_data_value_;
}
if (is_2draster) {
return raster_2d_[cell_index][lyr - 1];
}
return raster_[cell_index];
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::GetValueByIndex(const int cell_index, T*& values) {
if (!ValidateRasterData() || !ValidateIndex(cell_index)) {
if (nullptr != values) { Release1DArray(values); }
values = nullptr;
return;
}
if (nullptr == values) {
Initialize1DArray(n_lyrs_, values, no_data_value_);
}
if (is_2draster) {
for (int i = 0; i < n_lyrs_; i++) {
values[i] = raster_2d_[cell_index][i];
}
} else {
values[0] = raster_[cell_index];
}
}

template <typename T, typename MASK_T>
T clsRasterData<T, MASK_T>::GetValue(const int row, const int col, const int lyr ) {
if (!ValidateRasterData() || !ValidateRowCol(row, col) || !ValidateLayer(lyr)) {
return no_data_value_;
}
if (calc_pos_ && nullptr != pos_data_) {
int valid_cell_index = GetPosition(row, col);
if (valid_cell_index < 0) { return no_data_value_; }
return GetValueByIndex(valid_cell_index, lyr);
}
if (is_2draster) { return raster_2d_[row * GetCols() + col][lyr - 1]; }
return raster_[row * GetCols() + col];
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::GetValue(const int row, const int col, T*& values) {
if (!ValidateRasterData() || !ValidateRowCol(row, col)) {
if (nullptr != values) { Release1DArray(values); }
values = nullptr;
return;
}
if (nullptr == values) {
Initialize1DArray(n_lyrs_, values, no_data_value_);
}
if (calc_pos_ && nullptr != pos_data_) {
int valid_cell_index = GetPosition(row, col);
if (valid_cell_index == -1) {
for (int i = 0; i < n_lyrs_; i++) {
values[i] = no_data_value_; 
}
} else {
GetValueByIndex(valid_cell_index, values);
}
} else {
if (is_2draster) {
for (int i = 0; i < n_lyrs_; i++) {
values[i] = raster_2d_[row * GetCols() + col][i];
}
} else {
values[0] = raster_[row * GetCols() + col];
}
}
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::SetHeader(const STRDBL_MAP& refers) {
CopyHeader(refers, headers_);
auto it = headers_.find(HEADER_RS_CELLSNUM);
if (it != headers_.end()) { n_cells_ = CVT_INT(it->second); }
it = headers_.find(HEADER_RS_LAYERS);
if (it != headers_.end()) { n_lyrs_ = CVT_INT(it->second); }
it = headers_.find(HEADER_RS_NODATA);
if (it != headers_.end()) { no_data_value_ = static_cast<T>(it->second); }
}


template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::SetValue(const int row, const int col, T value, const int lyr ) {
if (!ValidateRasterData() || !ValidateRowCol(row, col) || !ValidateLayer(lyr)) {
StatusMessage("Set value failed!");
return;
}
int idx = GetPosition(row, col);
if (idx == -1) {
StatusMessage("Current version do not support to setting value to NoDATA location!");
} else {
if (is_2draster) {
raster_2d_[idx][lyr - 1] = value;
} else {
raster_[idx] = value;
}
}
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::SetCalcPositions() {
if (!ValidateRasterData()) { return false; }
if (calc_pos_ && nullptr != pos_data_) {
return false;
}
calc_pos_ = true;
return MaskAndCalculateValidPosition() >= 0;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::SetPositions(int len, int** pdata) {
if (nullptr != pos_data_) {
if (len != n_cells_) { return false; } 
Release2DArray(pos_data_);
}
pos_data_ = pdata;
calc_pos_ = true;
store_pos_ = false;
return true;
}


template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::SetUseMaskExt() {
if (!ValidateRasterData()) return false;
if (nullptr == mask_) return false;
if (use_mask_ext_) return false; 
use_mask_ext_ = true;
return MaskAndCalculateValidPosition() >= 0;
}



template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputToFile(const string& filename, const bool out_origin ) {
if (GetPathFromFullName(filename).empty()) { return false; }
string abs_filename = GetAbsolutePath(filename);
if (!ValidateRasterData()) { return false; }
if (!out_origin) { return OutputSubsetToFile(false, true, filename); }
string filetype = GetUpper(GetSuffix(abs_filename));
if (StringMatch(filetype, ASCIIExtension)) {
return OutputAscFile(abs_filename);
}
#ifdef USE_GDAL
if (StringMatch(filetype, GTiffExtension)) {
return OutputFileByGdal(abs_filename);
}
return OutputFileByGdal(ReplaceSuffix(abs_filename, GTiffExtension));
#else
StatusMessage("Warning: Without GDAL, ASC file will be exported as default!");
return OutputAscFile(ReplaceSuffix(abs_filename, ASCIIExtension));
#endif 
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputSubsetToFile(const bool out_origin ,
const bool out_combined ,
const string& outname ,
const map<vint, vector<double> >& recls ,
const double default_value ) {
if (!ValidateRasterData()) { return false; }
if (subset_.empty()) { return false; }
string outpathact = outname.empty() ? full_path_ : outname;
bool out_comb = out_combined;
if (out_comb) {
T* data1d = nullptr;
int sublyrs;
int sublen;
if (!PrepareCombSubsetData(&data1d, &sublen, &sublyrs,
out_origin, true, recls, default_value)) {
return false;
}
STRDBL_MAP tmpheader;
CopyHeader(headers_, tmpheader);
UpdateHeader(tmpheader, HEADER_RS_LAYERS, sublyrs);
UpdateHeader(tmpheader, HEADER_RS_CELLSNUM, sublen / sublyrs);
bool combflag = OutputFullsizeToFiles(data1d, sublen / sublyrs, sublyrs,
PrefixCoreFileName(outpathact, 0),
tmpheader, options_);
Release1DArray(data1d);
return combflag;
}
STRDBL_MAP subheader;
bool flag = true;
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
if (!it->second->usable) { continue; }
it->second->GetHeader(GetXllCenter(), GetYllCenter(), GetRows(),
GetCellWidth(), CVT_DBL(no_data_value_), subheader);
T* tmpdata1d = nullptr;
int tmpdatalen;
int tmplyrs;
if (!PrepareSubsetData(it->first, it->second, &tmpdata1d, &tmpdatalen, &tmplyrs,
out_origin, true, recls, default_value)) {
continue;
}
UpdateHeader(subheader, HEADER_RS_LAYERS, tmplyrs);
UpdateHeader(subheader, HEADER_RS_CELLSNUM, tmpdatalen / tmplyrs);
flag = flag && OutputFullsizeToFiles(tmpdata1d, tmpdatalen / tmplyrs, tmplyrs,
PrefixCoreFileName(outpathact, it->first),
subheader, options_);
Release1DArray(tmpdata1d);
}
return flag;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::PrepareCombSubsetData(T** values, int* datalen, int* datalyrs,
bool out_origin , bool include_nodata ,
const map<vint, vector<double> >& recls ,
double default_value ) {
if (subset_.empty()) { return false; }
T* data1d = nullptr; 
int lyr_recls = -1;
if (!recls.empty()) {
for (auto it = recls.begin(); it != recls.end(); ++it) {
if (lyr_recls < 0) { lyr_recls = CVT_INT(it->second.size()); }
else if (lyr_recls != it->second.size()) {
StatusMessage("Error: Reclassification layer count MUST be consistent!");
return false;
}
}
}
int lyrs_subset = -1;
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
if (!it->second->usable) { continue; }
if (!(nullptr != it->second->data_    
|| nullptr != it->second->data2d_ 
|| !recls.empty())) {             
return false;
}
if (lyrs_subset < 0) { lyrs_subset = it->second->n_lyrs; }
else if (it->second->n_lyrs != lyrs_subset) {
StatusMessage("Error: Subset's layer count MUST be consistent!");
return false;
}
}
int lyrs = -1;
if (out_origin) {
lyrs = n_lyrs_;
if (lyr_recls > 0) { lyrs = lyr_recls; }
} else {
lyrs = lyrs_subset;
if (lyrs < 0 && lyr_recls > 0) {
lyrs = lyr_recls;
}
}
if (lyrs < 0) {
StatusMessage("Error: Cannot determine valid layer count!");
return false;
}

int gnrows = GetRows();
int gncols = GetCols();
if (FloatEqual(default_value, NODATA_VALUE)) {
default_value = default_value_;
}
int gncells = include_nodata ? gnrows * gncols : n_cells_;
int data_length = gncells * lyrs;
Initialize1DArray(data_length, data1d, no_data_value_);
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
bool use_defaultv_directly = false;
if (!it->second->usable) {
if (!FloatEqual(no_data_value_, default_value)) {
use_defaultv_directly = true;
} else {
continue;
}
}
for (int vi = 0; vi < it->second->n_cells; vi++) {
for (int ilyr = 0; ilyr < lyrs; ilyr++) {
int gidx = it->second->global_[vi];
int tmpr = pos_data_[gidx][0];
int tmpc = pos_data_[gidx][1];
int tmprc = tmpr * gncols + tmpc;
if (!include_nodata) { tmprc = gidx; }
if (!recls.empty()) { 
double uniqe_value = default_value;
int recls_key = it->first; 
if (out_origin) {
if (nullptr != raster_) { recls_key = CVT_INT(raster_[gidx]); }
else if (nullptr != raster_2d_) {
recls_key = CVT_INT(raster_2d_[gidx][ilyr]);
}
}
if (recls.count(recls_key) > 0 && recls.at(recls_key).size() > ilyr) {
uniqe_value = recls.at(recls_key).at(ilyr);
}
data1d[tmprc * lyrs + ilyr] = static_cast<T>(uniqe_value);
}
else if (use_defaultv_directly) {
data1d[tmprc * lyrs + ilyr] = static_cast<T>(default_value);
}
else if (!out_origin && lyrs > 1 && nullptr != it->second->data2d_) { 
data1d[tmprc * lyrs + ilyr] = static_cast<T>(it->second->data2d_[vi][ilyr]);
}
else if (!out_origin && lyrs == 1 && nullptr != it->second->data_) { 
data1d[tmprc * lyrs + ilyr] = static_cast<T>(it->second->data_[vi]);
}
else { 
StatusMessage("Error: No subset or reclassification map can be output!");
if (nullptr != data1d) { Release1DArray(data1d); }
return false;
}
}
}
}
*values = data1d;
*datalen = data_length;
*datalyrs = lyrs;
return true;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::PrepareSubsetData(const int sub_id, SubsetPositions* sub,
T** values, int* datalen, int* datalyrs,
bool out_origin , bool include_nodata ,
const map<vint, vector<double> >& recls ,
double default_value ) {
if (nullptr == sub) { return false; }
T* data1d = nullptr; 
int lyr_recls = -1;
if (!recls.empty()) {
for (auto it = recls.begin(); it != recls.end(); ++it) {
if (lyr_recls < 0) { lyr_recls = CVT_INT(it->second.size()); }
else if (lyr_recls != it->second.size()) {
StatusMessage("Error: Reclassification layer count MUST be consistent!");
return false;
}
}
}
int lyrs = - 1;
if (out_origin) {
lyrs = n_lyrs_;
if (lyr_recls > 0) { lyrs = lyr_recls; }
} else {
lyrs = sub->n_lyrs;
}
if (lyrs < 0) {
StatusMessage("Error: Cannot determine valid layer count!");
return false;
}
int nrows = sub->g_erow - sub->g_srow + 1;
int ncols = sub->g_ecol - sub->g_scol + 1;
if (FloatEqual(default_value, NODATA_VALUE)) {
default_value = default_value_;
}
int ncells = include_nodata ? nrows * ncols : sub->n_cells;
int data_length = ncells * lyrs;
Initialize1DArray(data_length, data1d, no_data_value_);
for (int vi = 0; vi < sub->n_cells; vi++) {
for (int ilyr = 0; ilyr < lyrs; ilyr++) {
int j = sub->local_pos_[vi][0] * ncols + sub->local_pos_[vi][1];
int gidx = sub->global_[vi];
if (!include_nodata) { j = vi; }
if (!recls.empty()) { 
double uniqe_value = default_value;
int recls_key = sub_id;
if (out_origin) {
if (nullptr != raster_) { recls_key = CVT_INT(raster_[gidx]); }
else if (nullptr != raster_2d_) {
recls_key = CVT_INT(raster_2d_[gidx][ilyr]);
}
}
if (recls.count(recls_key) > 0 && recls.at(recls_key).size() > ilyr) {
uniqe_value = recls.at(recls_key).at(ilyr);
}
data1d[j * lyrs + ilyr] = static_cast<T>(uniqe_value);
}
else if (out_origin && lyrs > 1 && nullptr != raster_2d_) {
data1d[j * lyrs + ilyr] = raster_2d_[gidx][ilyr];
}
else if (out_origin && lyrs == 1 && nullptr != raster_) {
data1d[j * lyrs + ilyr] = raster_[gidx];
}
else if (!out_origin && lyrs > 1 && nullptr != sub->data2d_) { 
data1d[j * lyrs + ilyr] = static_cast<T>(sub->data2d_[vi][ilyr]);
}
else if (!out_origin && lyrs == 1 && nullptr != sub->data_) { 
data1d[j * lyrs + ilyr] = static_cast<T>(sub->data_[vi]);
}
else { 
StatusMessage("Error: No subset or reclassification map can be output!");
if (nullptr != data1d) { Release1DArray(data1d); }
return false;
}
}
}
*values = data1d;
*datalen = data_length;
*datalyrs = lyrs;
return true;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputFullsizeToFiles(T* fullsizedata, const int fsize, const int datalyrs,
const string& fullfilename,
const STRDBL_MAP& header, const STRING_MAP& opts) {
if (nullptr == fullsizedata) { return false; }
if (datalyrs < 1) { return false; }
if (datalyrs == 1) {
return WriteRasterToFile(fullfilename, header, opts, fullsizedata);
}
T* tmpdata1d = nullptr;
Initialize1DArray(fsize, tmpdata1d, NODATA_VALUE);
bool flag = true;
for (int ilyr = 0; ilyr < datalyrs; ilyr++) {
for (int gi = 0; gi < fsize; gi++) {
tmpdata1d[gi] = fullsizedata[gi * datalyrs + ilyr];
}
flag = flag && WriteRasterToFile(AppendCoreFileName(fullfilename, ilyr + 1),
header, opts, tmpdata1d);
}
if (nullptr != tmpdata1d) { Release1DArray(tmpdata1d); }
return flag;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputAscFile(const string& filename) {
string abs_filename = GetAbsolutePath(filename);
int count;
int** position = nullptr;
bool outputdirectly = true;
if (nullptr != pos_data_) {
GetRasterPositionData(&count, &position);
outputdirectly = false;
assert(nullptr != position);
}
int rows = CVT_INT(headers_.at(HEADER_RS_NROWS));
int cols = CVT_INT(headers_.at(HEADER_RS_NCOLS));
if (is_2draster) { 
string pre_path = GetPathFromFullName(abs_filename);
if (StringMatch(pre_path, "")) { return false; }
for (int lyr = 0; lyr < n_lyrs_; lyr++) {
string tmpfilename = AppendCoreFileName(abs_filename, itoa(CVT_VINT(lyr)));
if (!WriteAscHeaders(tmpfilename, headers_)) { return false; }
std::ofstream raster_file(tmpfilename.c_str(), std::ios::app | std::ios::out);
if (!raster_file.is_open()) {
StatusMessage("Error opening file: " + tmpfilename);
return false;
}
int index = 0;
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < cols; ++j) {
if (outputdirectly) {
index = i * cols + j;
raster_file << setprecision(6) << raster_2d_[index][lyr] << " ";
continue;
}
if (index < n_cells_ && (position[index][0] == i && position[index][1] == j)) {
raster_file << setprecision(6) << raster_2d_[index][lyr] << " ";
index++;
} else { raster_file << setprecision(6) << NODATA_VALUE << " "; }
}
raster_file << endl;
}
raster_file.close();
}
} else { 
if (!WriteAscHeaders(abs_filename, headers_)) { return false; }
std::ofstream raster_file(filename.c_str(), std::ios::app | std::ios::out);
if (!raster_file.is_open()) {
StatusMessage("Error opening file: " + abs_filename);
return false;
}
int index = 0;
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < cols; ++j) {
if (outputdirectly) {
index = i * cols + j;
raster_file << setprecision(6) << raster_[index] << " ";
continue;
}
if (index < n_cells_) {
if (position[index][0] == i && position[index][1] == j) {
raster_file << setprecision(6) << raster_[index] << " ";
index++;
} else { raster_file << setprecision(6) << no_data_value_ << " "; }
} else { raster_file << setprecision(6) << no_data_value_ << " "; }
}
raster_file << endl;
}
raster_file.close();
}
position = nullptr;
return true;
}

#ifdef USE_GDAL
template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputFileByGdal(const string& filename) {
string abs_filename = GetAbsolutePath(filename);
bool outputdirectly = nullptr == pos_data_;
int n_rows = CVT_INT(headers_.at(HEADER_RS_NROWS));
int n_cols = CVT_INT(headers_.at(HEADER_RS_NCOLS));
bool outflag = false;
T* data_1d = nullptr;
if (is_2draster) {
string pre_path = GetPathFromFullName(abs_filename);
if (StringMatch(pre_path, "")) { return false; }
for (int lyr = 0; lyr < n_lyrs_; lyr++) {
string tmpfilename = AppendCoreFileName(abs_filename, lyr + 1);
if (outputdirectly) {
if (nullptr == data_1d) {
Initialize1DArray(n_rows * n_cols, data_1d, no_data_value_);
}
for (int gi = 0; gi < n_rows * n_cols; gi++) {
data_1d[gi] = raster_2d_[gi][lyr];
}
} else {
if (nullptr == data_1d) {
Initialize1DArray(n_rows * n_cols, data_1d, no_data_value_);
}
for (int vi = 0; vi < n_cells_; vi++) {
data_1d[pos_data_[vi][0] * n_cols + pos_data_[vi][1]] = raster_2d_[vi][lyr];
}
}
outflag = WriteSingleGeotiff(tmpfilename, headers_, options_, data_1d);
if (!outflag) {
Release1DArray(data_1d);;
return false;
}
}
Release1DArray(data_1d);
} else {
if (outputdirectly) {
outflag = WriteSingleGeotiff(abs_filename, headers_, options_, raster_);
} else {
Initialize1DArray(n_rows * n_cols, data_1d, no_data_value_);
for (int vi = 0; vi < n_cells_; vi++) {
data_1d[pos_data_[vi][0] * n_cols + pos_data_[vi][1]] = raster_[vi];
}
outflag = WriteSingleGeotiff(abs_filename, headers_, options_, data_1d);
Release1DArray(data_1d);
}
if (!outflag) { return false; }
}
return true;
}
#endif 

#ifdef USE_MONGODB
template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputToMongoDB(MongoGridFs* gfs, const string& filename ,
const STRING_MAP& opts ,
bool include_nodata ,
bool out_origin ) {
if (nullptr == gfs) { return false; }
if (!out_origin) { 
return OutputSubsetToMongoDB(gfs, filename, opts, include_nodata, false, true);
}
CopyStringMap(opts, options_); 
if (options_.find(HEADER_RSOUT_DATATYPE) == options_.end()
|| StringMatch("Unknown", options_.at(HEADER_RSOUT_DATATYPE))) {
UpdateStrHeader(options_, HEADER_RSOUT_DATATYPE,
RasterDataTypeToString(TypeToRasterDataType(typeid(T))));
}
bool outputdirectly = true; 
int cnt;
int** pos = nullptr;
if (nullptr != pos_data_ && include_nodata) {
outputdirectly = false;
GetRasterPositionData(&cnt, &pos);
}
if (nullptr == pos_data_ && !include_nodata) {
SetCalcPositions();
GetRasterPositionData(&cnt, &pos);
}
int n_rows = GetRows();
int n_cols = GetCols();
int n_fullsize = n_rows * n_cols;
if (include_nodata) {
UpdateStringMap(options_, HEADER_INC_NODATA, "TRUE");
} else {
UpdateStringMap(options_, HEADER_INC_NODATA, "FALSE");
}
T* data_1d = nullptr;
T no_data_value = GetNoDataValue();
int datalength;
string core_name = filename.empty() ? core_name_ : filename;
if (is_2draster) { 
if (outputdirectly) {
data_1d = raster_2d_[0]; 
datalength = n_lyrs_ * n_cells_; 
} else {
datalength = n_lyrs_ * n_fullsize;
Initialize1DArray(datalength, data_1d, no_data_value);
for (int idx = 0; idx < n_cells_; idx++) {
int rowcol_index = pos[idx][0] * n_cols + pos[idx][1];
for (int k = 0; k < n_lyrs_; k++) {
data_1d[n_lyrs_ * rowcol_index + k] = raster_2d_[idx][k];
}
}
}
} else { 
if (outputdirectly) {
data_1d = raster_;
datalength = n_cells_; 
} else {
datalength = n_fullsize;
Initialize1DArray(datalength, data_1d, no_data_value);
for (int idx = 0; idx < n_cells_; idx++) {
data_1d[pos[idx][0] * n_cols + pos[idx][1]] = raster_[idx];
}
}
}
STRDBL_MAP tmpheader;
CopyHeader(headers_, tmpheader);
if (include_nodata) { UpdateHeader(tmpheader, HEADER_RS_CELLSNUM, n_fullsize); }
bool saved = WriteStreamDataAsGridfs(gfs, core_name, tmpheader,
data_1d, datalength, options_);
if (outputdirectly) {
data_1d = nullptr;
} else {
Release1DArray(data_1d);
}
return saved;
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::OutputSubsetToMongoDB(MongoGridFs* gfs,
const string& filename ,
const STRING_MAP& opts ,
bool include_nodata ,
bool out_origin ,
bool out_combined ,
const map<vint, vector<double> >& recls ,
double default_value ) {
if (!ValidateRasterData()) { return false; }
if (nullptr == gfs) { return false; }
if (subset_.empty()) { return false; }
CopyStringMap(opts, options_); 
if (include_nodata) {
UpdateStringMap(options_, HEADER_INC_NODATA, "TRUE");
}
else {
UpdateStringMap(options_, HEADER_INC_NODATA, "FALSE");
}
if (options_.find(HEADER_RSOUT_DATATYPE) == options_.end()
|| StringMatch("Unknown", options_.at(HEADER_RSOUT_DATATYPE))) {
UpdateStrHeader(options_, HEADER_RSOUT_DATATYPE,
RasterDataTypeToString(TypeToRasterDataType(typeid(T))));
}
int grows = GetRows();
string outnameact = filename.empty() ? core_name_ : filename;
bool out_comb = out_combined;
if (out_comb) {
T* data1d = nullptr;
int sublyrs;
int sublen;
bool flag = PrepareCombSubsetData(&data1d, &sublen, &sublyrs,
out_origin, include_nodata, recls, default_value);
STRDBL_MAP tmpheader;
CopyHeader(headers_, tmpheader);
UpdateHeader(tmpheader, HEADER_RS_LAYERS, sublyrs);
UpdateHeader(tmpheader, HEADER_RS_CELLSNUM, sublen / sublyrs);
flag = flag && WriteStreamDataAsGridfs(gfs, "0_" + outnameact,
tmpheader, data1d, sublen, options_);
Release1DArray(data1d);
return flag;
}
STRDBL_MAP subheader;
bool flag = true;
for (auto it = subset_.begin(); it != subset_.end(); ++it) {
if (!it->second->usable) { continue; }
it->second->GetHeader(GetXllCenter(), GetYllCenter(), grows,
GetCellWidth(), CVT_DBL(no_data_value_), subheader);
T* tmpdata1d = nullptr;
int tmpdatalen;
int tmplyrs;
if (!PrepareSubsetData(it->first, it->second, &tmpdata1d, &tmpdatalen, &tmplyrs,
out_origin, include_nodata, recls, default_value)) {
continue;
}
UpdateHeader(subheader, HEADER_RS_LAYERS, tmplyrs);
UpdateHeader(subheader, HEADER_RS_CELLSNUM, tmpdatalen / tmplyrs);
string tmpfname = itoa(CVT_VINT(it->first)) + "_" + outnameact;
flag = flag && WriteStreamDataAsGridfs(gfs, tmpfname, subheader, tmpdata1d, tmpdatalen, options_);
if (nullptr != tmpdata1d) { Release1DArray(tmpdata1d); }
}
return true;
}

#endif 


template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::ReadFromFile(const string& filename, const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
if (!FileExists(filename)) { return false; }
if (!initialized_) { InitializeRasterClass(); }
rs_type_out_ = RasterDataTypeInOptionals(opts);
return ConstructFromSingleFile(filename, calc_pos, mask, use_mask_ext, default_value, opts);
}

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::ReadFromFiles(vector<string>& filenames, const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
if (!FilesExist(filenames)) { return false; }
n_lyrs_ = CVT_INT(filenames.size());
if (n_lyrs_ == 1) { return ReadFromFile(filenames[0], calc_pos, mask, use_mask_ext, default_value, opts); }
if (!initialized_) { InitializeRasterClass(true); }
rs_type_out_ = RasterDataTypeInOptionals(opts);
if (!ConstructFromSingleFile(filenames[0], calc_pos, mask, use_mask_ext, default_value, opts)) {
return false;
}
string::size_type last_underline = core_name_.find_last_of('_');
if (last_underline == string::npos) {
last_underline = core_name_.length();
}
core_name_ = core_name_.substr(0, last_underline);
full_path_ = GetPathFromFullName(filenames[0]) + core_name_ + "_%d." + GetSuffix(filenames[0]);
Initialize2DArray(n_cells_, n_lyrs_, raster_2d_, no_data_value_);
#pragma omp parallel for
for (int i = 0; i < n_cells_; i++) {
raster_2d_[i][0] = raster_[i];
}
Release1DArray(raster_);
int rows = GetRows();
int cols = GetCols();
for (int fileidx = 1; fileidx < CVT_INT(filenames.size()); fileidx++) {
STRDBL_MAP tmpheader;
T* tmplyrdata = nullptr;
string curfilename = filenames[fileidx];
if (StringMatch(GetUpper(GetSuffix(curfilename)), string(ASCIIExtension))) {
ReadAscFile(curfilename, tmpheader, tmplyrdata);
}
else {
#ifdef USE_GDAL
RasterDataType tmpintype;
string tmpsrs;
ReadRasterFileByGdal(curfilename, tmpheader, tmplyrdata, tmpintype, tmpsrs);
#else
StatusMessage("Warning: Only ASC format is supported without GDAL!");
return false;
#endif
}
if (nullptr != pos_data_) {
#pragma omp parallel for
for (int i = 0; i < n_cells_; ++i) {
int tmp_row = pos_data_[i][0];
int tmp_col = pos_data_[i][1];
AddOtherLayerRasterData(tmp_row, tmp_col, i, fileidx, tmpheader, tmplyrdata);
}
}
else {
#pragma omp parallel for
for (int i = 0; i < rows; i++) {
for (int j = 0; j < cols; j++) {
int cellidx = i * cols + j;
if (FloatEqual(no_data_value_, raster_2d_[cellidx][0])) {
raster_2d_[cellidx][fileidx] = no_data_value_;
continue;
}
AddOtherLayerRasterData(i, j, cellidx, fileidx, tmpheader, tmplyrdata);
}
}
}
Release1DArray(tmplyrdata);
}
if(!is_2draster) is_2draster = true;
UpdateHeader(headers_, HEADER_RS_LAYERS, n_lyrs_); 
return true;
}

#ifdef USE_MONGODB

template <typename T, typename MASK_T>
bool clsRasterData<T, MASK_T>::ReadFromMongoDB(MongoGridFs* gfs,
const string& filename,
const bool calc_pos ,
clsRasterData<MASK_T>* mask ,
const bool use_mask_ext ,
double default_value ,
const STRING_MAP& opts ) {
InitializeReadFunction(filename, calc_pos, mask, use_mask_ext, default_value, opts);
T* dbdata = nullptr;
STRDBL_MAP header_dbl = InitialHeader();
STRING_MAP header_str = InitialStrHeader();
STRING_MAP opts_upd;
CopyStringMap(opts, opts_upd);
if (opts.empty() || opts.count(HEADER_INC_NODATA) < 1) { 
UpdateStringMap(opts_upd, HEADER_INC_NODATA, "TRUE");
}
if (!ReadGridFsFile(gfs, filename, dbdata, header_dbl, header_str, opts_upd)) { return false; }

if (headers_.at(HEADER_RS_NROWS) > 0 && headers_.at(HEADER_RS_NCOLS) > 0
&& headers_.at(HEADER_RS_LAYERS) > 0 && headers_.at(HEADER_RS_CELLSNUM) > 0) {
if (!FloatEqual(headers_.at(HEADER_RS_NROWS), header_dbl.at(HEADER_RS_NROWS))
|| !FloatEqual(headers_.at(HEADER_RS_NCOLS), header_dbl.at(HEADER_RS_NCOLS))
|| !FloatEqual(headers_.at(HEADER_RS_LAYERS), header_dbl.at(HEADER_RS_LAYERS))
|| !FloatEqual(headers_.at(HEADER_RS_CELLSNUM), header_dbl.at(HEADER_RS_CELLSNUM))) {
Release1DArray(dbdata);
return false;
}
} else {
CopyHeader(header_dbl, headers_);
}
CopyStringMap(header_str, options_);

bool include_nodata = !StringMatch(options_.at(HEADER_INC_NODATA), "FALSE");

int n_rows = CVT_INT(headers_.at(HEADER_RS_NROWS));
int n_cols = CVT_INT(headers_.at(HEADER_RS_NCOLS));
n_lyrs_ = CVT_INT(headers_.at(HEADER_RS_LAYERS));
int fullsize = n_rows * n_cols;
no_data_value_ = static_cast<T>(headers_.at(HEADER_RS_NODATA));
n_cells_ = CVT_INT(headers_.at(HEADER_RS_CELLSNUM));

if (include_nodata && n_cells_ != fullsize) { return false; }
if (n_cells_ != fullsize) { calc_pos_ = true; }

bool mask_pos_subset = true;
if (nullptr != mask_ && calc_pos_ && use_mask_ext_ && n_cells_ == mask_->GetValidNumber()) {
store_pos_ = false;
mask_->GetRasterPositionData(&n_cells_, &pos_data_);
if (!mask->GetSubset().empty()) {
map<int, SubsetPositions*>& mask_subset = mask_->GetSubset();
for (auto it = mask_subset.begin(); it != mask_subset.end(); ++it) {
SubsetPositions* tmp = new SubsetPositions(it->second, true);
tmp->n_lyrs = n_lyrs_;
#ifdef HAS_VARIADIC_TEMPLATES
subset_.emplace(it->first, tmp);
#else
subset_.insert(make_pair(it->first, tmp));
#endif
}
}
mask_pos_subset = false;
}

if (!include_nodata && nullptr == pos_data_) { return false; }

if (n_lyrs_ == 1) {
is_2draster = false;
if (include_nodata && !mask_pos_subset) {
Initialize1DArray(n_cells_, raster_, no_data_value_);
#pragma omp parallel for
for (int i = 0; i < n_cells_; i++) {
int tmpidx = pos_data_[i][0] * n_cols + pos_data_[i][1];
raster_[i] = static_cast<T>(dbdata[tmpidx]);
}
Release1DArray(dbdata);
} else {
raster_ = dbdata;
}
} else {
Initialize2DArray(n_cells_, n_lyrs_, raster_2d_, no_data_value_);
#pragma omp parallel for
for (int i = 0; i < n_cells_; i++) {
int tmpidx = i;
if (include_nodata && !mask_pos_subset) {
tmpidx = pos_data_[i][0] * n_cols + pos_data_[i][1];
}
for (int j = 0; j < n_lyrs_; j++) {
int idx = tmpidx * n_lyrs_ + j;
raster_2d_[i][j] = dbdata[idx];
}
}
is_2draster = true;
Release1DArray(dbdata);
}
CheckDefaultValue();
if (include_nodata && mask_pos_subset) {
return MaskAndCalculateValidPosition() >= 0;
}
return true;
}

#endif 

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::AddOtherLayerRasterData(const int row, const int col,
const int cellidx, const int lyr,
STRDBL_MAP lyrheader, T* lyrdata) {
int tmpcols = CVT_INT(lyrheader.at(HEADER_RS_NCOLS));
double tmpnodata = lyrheader.at(HEADER_RS_NODATA);
XY_COOR tmp_xy = GetCoordinateByRowCol(row, col);
ROW_COL tmp_pos = GetPositionByCoordinate(tmp_xy.first, tmp_xy.second, &lyrheader);
if (tmp_pos.first == -1 || tmp_pos.second == -1) {
raster_2d_[cellidx][lyr] = no_data_value_;
} else {
T tmpvalue = lyrdata[tmp_pos.first * tmpcols + tmp_pos.second];
raster_2d_[cellidx][lyr] = FloatEqual(tmpvalue, tmpnodata) ? no_data_value_ : tmpvalue;
}
}

template <typename T, typename MASK_T>
clsRasterData<T, MASK_T>::clsRasterData(clsRasterData<T, MASK_T>* another) {
InitializeRasterClass();
Copy(another);
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::Copy(clsRasterData<T, MASK_T>* orgraster) {
if (is_2draster && nullptr != raster_2d_ && n_cells_ > 0) {
Release2DArray(raster_2d_);
}
if (!is_2draster && nullptr != raster_) {
Release1DArray(raster_);
}
if (nullptr != pos_data_) {
Release2DArray(pos_data_);
}
if (stats_calculated_) {
ReleaseStatsMap2D();
stats_calculated_ = false;
}
if (!subset_.empty()) {
ReleaseSubset();
}
InitializeReadFunction(orgraster->GetFilePath(), orgraster->PositionsCalculated(),
orgraster->GetMask(), orgraster->MaskExtented(),
orgraster->GetDefaultValue(), orgraster->GetOptions());
n_cells_ = orgraster->GetCellNumber();
n_lyrs_ = orgraster->GetLayers();
rs_type_ = orgraster->GetDataType();
rs_type_out_ = orgraster->GetOutDataType();
no_data_value_ = orgraster->GetNoDataValue();
if (orgraster->Is2DRaster()) {
is_2draster = true;
Initialize2DArray(n_cells_, n_lyrs_, raster_2d_,
orgraster->Get2DRasterDataPointer());
} else {
Initialize1DArray(n_cells_, raster_, orgraster->GetRasterDataPointer());
}
if (calc_pos_) {
store_pos_ = true;
Initialize2DArray(n_cells_, 2, pos_data_,
orgraster->GetRasterPositionDataPointer());
}
stats_calculated_ = orgraster->StatisticsCalculated();
if (stats_calculated_) {
if (is_2draster) {
map<string, double *> stats2D = orgraster->GetStatistics2D();
for (auto iter = stats2D.begin(); iter != stats2D.end(); ++iter) {
double* tmpstatvalues = nullptr;
Initialize1DArray(n_lyrs_, tmpstatvalues, iter->second);
stats_2d_.at(iter->first) = tmpstatvalues;
}
} else {
STRDBL_MAP stats = orgraster->GetStatistics();
for (auto iter = stats.begin(); iter != stats.end(); ++iter) {
stats_[iter->first] = iter->second;
}
}
}
CopyHeader(orgraster->GetRasterHeader(), headers_);
if (!orgraster->GetSubset().empty()) {
for (auto it = orgraster->GetSubset().begin(); it != orgraster->GetSubset().end(); ++it) {
SubsetPositions* tmp = new SubsetPositions(it->second, true);
#ifdef HAS_VARIADIC_TEMPLATES
subset_.emplace(it->first, tmp);
#else
subset_.insert(make_pair(it->first, tmp));
#endif
}
}
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::ReplaceNoData(T replacedv) {
#pragma omp parallel for
for (int i = 0; i < n_cells_; i++) {
for (int lyr = 0; lyr < n_lyrs_; lyr++) {
bool flag = is_2draster && nullptr != raster_2d_
? FloatEqual(raster_2d_[i][lyr], no_data_value_)
: FloatEqual(raster_[i], no_data_value_);
if (!flag) { continue; }
if (is_2draster && nullptr != raster_2d_) {
raster_2d_[i][lyr] = replacedv;
} else if (nullptr != raster_) {
raster_[i] = replacedv;
}
}
}
no_data_value_ = replacedv;
default_value_ = CVT_DBL(replacedv);
UpdateHeader(headers_, HEADER_RS_NODATA, replacedv);
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::Reclassify(const map<int, T> reclass_map) {
map<int, T> recls;
for(auto it = reclass_map.begin(); it != reclass_map.end(); ++it) {
#ifdef HAS_VARIADIC_TEMPLATES
recls.emplace(it->first, it->second);
#else
recls.insert(make_pair(it->first, it->second));
#endif
}
for (int i = 0; i < n_cells_; i++) {
for (int lyr = 0; lyr < n_lyrs_; lyr++) {
T curv = is_2draster && nullptr != raster_2d_
? CVT_INT(raster_2d_[i][lyr])
: CVT_INT(raster_[i]);
if (recls.count(curv) < 0) {
#ifdef HAS_VARIADIC_TEMPLATES
recls.emplace(curv, no_data_value_);
#else
recls.insert(make_pair(curv, no_data_value_));
#endif
}
if (is_2draster && nullptr != raster_2d_) {
raster_2d_[i][lyr] = recls.at(curv);
} else if (nullptr != raster_) {
raster_[i] = recls.at(curv);
}
}
}
}



template <typename T, typename MASK_T>
XY_COOR clsRasterData<T, MASK_T>::GetCoordinateByRowCol(const int row, const int col) {
return XY_COOR(GetXllCenter() + col * GetCellWidth(),
GetYllCenter() + (GetRows() - row - 1) * GetCellWidth());
}

template <typename T, typename MASK_T>
ROW_COL clsRasterData<T, MASK_T>::GetPositionByCoordinate(const double x, const double y,
STRDBL_MAP* header ) {
if (nullptr == header) { header = &headers_; }
double xll_center = (*header).at(HEADER_RS_XLL);
double yll_center = (*header).at(HEADER_RS_YLL);
double dx = (*header).at(HEADER_RS_CELLSIZE);
double dy = dx;
int n_rows = CVT_INT((*header).at(HEADER_RS_NROWS));
int n_cols = CVT_INT((*header).at(HEADER_RS_NCOLS));

double x_min = xll_center - dx / 2.;
double x_max = x_min + dx * n_cols;

double y_min = yll_center - dy / 2.;
double y_max = y_min + dy * n_rows;
if ((x > x_max || x < xll_center) || (y > y_max || y < yll_center)) {
return ROW_COL(-1, -1);
}
return ROW_COL(CVT_INT((y_max - y) / dy), CVT_INT((x - x_min) / dx));
}

template <typename T, typename MASK_T>
void clsRasterData<T, MASK_T>::CalculateValidPositionsFromGridData() {
vector<T> values; 
vector<vector<T> > values_2d; 
vector<int> pos_rows;
vector<int> pos_cols;
int nrows = CVT_INT(headers_.at(HEADER_RS_NROWS));
int ncols = CVT_INT(headers_.at(HEADER_RS_NCOLS));
for (int i = 0; i < nrows; ++i) {
for (int j = 0; j < ncols; ++j) {
int idx = i * ncols + j;
T tmp_value;
if (is_2draster) {
tmp_value = raster_2d_[idx][0];
} else {
tmp_value = raster_[idx];
}
if (FloatEqual(tmp_value, static_cast<T>(no_data_value_))) continue;
values.emplace_back(tmp_value);
if (is_2draster && n_lyrs_ > 1) {
vector<T> tmpv(n_lyrs_ - 1);
for (int lyr = 1; lyr < n_lyrs_; lyr++) {
tmpv[lyr - 1] = raster_2d_[idx][lyr];
}
values_2d.emplace_back(tmpv);
}
pos_rows.emplace_back(i);
pos_cols.emplace_back(j);
}
}
vector<T>(values).swap(values);
if (is_2draster && n_lyrs_ > 1) {
vector<vector<T> >(values_2d).swap(values_2d);
}
vector<int>(pos_rows).swap(pos_rows);
vector<int>(pos_cols).swap(pos_cols);
n_cells_ = CVT_INT(values.size());
UpdateHeader(headers_, HEADER_RS_CELLSNUM, n_cells_);
if (is_2draster) {
Release2DArray(raster_2d_);
Initialize2DArray(n_cells_, n_lyrs_, raster_2d_, no_data_value_);
} else {
Release1DArray(raster_);
Initialize1DArray(n_cells_, raster_, no_data_value_);
}
Initialize2DArray(n_cells_, 2, pos_data_, 0);
store_pos_ = true;
#pragma omp parallel for
for (int i = 0; i < n_cells_; ++i) {
if (is_2draster) {
raster_2d_[i][0] = values.at(i);
if (n_lyrs_ > 1) {
for (int lyr = 1; lyr < n_lyrs_; lyr++) {
raster_2d_[i][lyr] = values_2d[i][lyr - 1];
}
}
} else {
raster_[i] = values.at(i);
}
pos_data_[i][0] = pos_rows.at(i);
pos_data_[i][1] = pos_cols.at(i);
}
calc_pos_ = true;
}

template <typename T, typename MASK_T>
int clsRasterData<T, MASK_T>::MaskAndCalculateValidPosition() {
int old_fullsize = GetRows() * GetCols();
if (nullptr == mask_) {
if (calc_pos_) {
if (nullptr == pos_data_) {
CalculateValidPositionsFromGridData();
return 1;
}
return 0;
}
n_cells_ = old_fullsize;
UpdateHeader(headers_, HEADER_RS_CELLSNUM, n_cells_);
return 0;
}
int mask_ncells;
int** valid_pos = nullptr;
int mask_rows = mask_->GetRows();
int mask_cols = mask_->GetCols();
if (!mask_->PositionsCalculated()) { mask_->SetCalcPositions(); }
mask_->GetRasterPositionData(&mask_ncells, &valid_pos);
vector<T> values(mask_ncells);             
vector<vector<T> > values_2d(mask_ncells); 
vector<int> pos_rows(mask_ncells); 
vector<int> pos_cols(mask_ncells); 
int max_row = -1;
int min_row = mask_rows;
int max_col = -1;
int min_col = mask_cols;
int masked_count = 0; 
int matched_count = 0; 
for (int i = 0; i < mask_ncells; i++) {
int tmp_row = valid_pos[i][0];
int tmp_col = valid_pos[i][1];
XY_COOR tmp_xy = mask_->GetCoordinateByRowCol(tmp_row, tmp_col);
ROW_COL tmp_pos = GetPositionByCoordinate(tmp_xy.first, tmp_xy.second);
T tmp_value;
if (tmp_pos.first == -1 || tmp_pos.second == -1) {
tmp_value = no_data_value_; 
if (is_2draster && n_lyrs_ > 1) {
vector<T> tmp_values(n_lyrs_ - 1);
for (int lyr = 1; lyr < n_lyrs_; lyr++) { tmp_values[lyr - 1] = no_data_value_; }
values_2d[i] = tmp_values;
}
values[i] = tmp_value;
pos_rows[i] = tmp_row;
pos_cols[i] = tmp_col;
continue;
}
tmp_value = GetValue(tmp_pos.first, tmp_pos.second, 1);
if (FloatEqual(tmp_value, no_data_value_)) {
if (!FloatEqual(default_value_, no_data_value_)) {
tmp_value = static_cast<T>(default_value_);
SetValue(tmp_pos.first, tmp_pos.second, tmp_value);
}
} else { 
matched_count++;
if (max_row < tmp_row) max_row = tmp_row;
if (min_row > tmp_row) min_row = tmp_row;
if (max_col < tmp_col) max_col = tmp_col;
if (min_col > tmp_col) min_col = tmp_col;
}
if (is_2draster && n_lyrs_ > 1) {
vector<T> tmp_values(n_lyrs_ - 1);
for (int lyr = 1; lyr < n_lyrs_; lyr++) {
tmp_values[lyr - 1] = GetValue(tmp_pos.first, tmp_pos.second, lyr + 1);
if (FloatEqual(tmp_values[lyr - 1], no_data_value_)
&& !FloatEqual(default_value_, no_data_value_)) {
tmp_values[lyr - 1] = static_cast<T>(default_value_);
SetValue(tmp_pos.first, tmp_pos.second, tmp_values[lyr - 1], lyr - 1);
}
}
values_2d[i] = tmp_values;
}
values[i] = tmp_value;
pos_rows[i] = tmp_row;
pos_cols[i] = tmp_col;
masked_count++;
}
if (masked_count == 0) { return -1; }
n_cells_ = masked_count;

CopyHeader(mask_->GetRasterHeader(), headers_);
UpdateHeader(headers_, HEADER_RS_NODATA, no_data_value_);
UpdateStrHeader(options_, HEADER_RS_SRS, mask_->GetSrsString());
UpdateHeader(headers_, HEADER_RS_LAYERS, n_lyrs_);

map<int, SubsetPositions*>& mask_subset = mask_->GetSubset();
bool mask_has_subset = !mask_subset.empty(); 
if (mask_has_subset) {
ReleaseSubset();
for (auto it = mask_subset.begin(); it != mask_subset.end(); ++it) {
SubsetPositions* tmp = new SubsetPositions(it->second, true);
tmp->n_lyrs = n_lyrs_;
#ifdef HAS_VARIADIC_TEMPLATES
subset_.emplace(it->first, tmp);
#else
subset_.insert(make_pair(it->first, tmp));
#endif
}
}

bool match_exactly = matched_count == mask_ncells;
bool within_ext = true;
int new_rows = max_row - min_row + 1;
int new_cols = max_col - min_col + 1;
if (new_rows != mask_rows || new_cols != mask_cols) {
within_ext = false;
} else { 
assert(masked_count == mask_ncells);
}
bool upd_header_rowcol = false; 
bool recalc_pos = false; 
bool upd_header_valid_num = false; 
bool store_fullsize = false; 
bool recalc_subset = false; 

if (within_ext) {
upd_header_rowcol = false;
if (use_mask_ext_) { 
if (calc_pos_) {
recalc_pos = !match_exactly;
upd_header_valid_num = recalc_pos;
store_fullsize = false;
} else { 
recalc_pos = false;
upd_header_valid_num = false;
store_fullsize = false;
calc_pos_ = true; 
}
} else { 
if (calc_pos_) {
recalc_pos = !match_exactly;
upd_header_valid_num = recalc_pos;
store_fullsize = false;
} else { 
recalc_pos = false;
upd_header_valid_num = true;
store_fullsize = false;
calc_pos_ = true; 
}
}
} else { 
if (use_mask_ext_) { 
upd_header_rowcol = false;
if (calc_pos_) {
recalc_pos = !match_exactly;
upd_header_valid_num = recalc_pos;
store_fullsize = false;
} else { 
recalc_pos = false;
upd_header_valid_num = false;
store_fullsize = false;
calc_pos_ = true; 
}
} else { 
upd_header_rowcol = true;
upd_header_valid_num = true;
if (calc_pos_) {
recalc_pos = true;
store_fullsize = false;
} else { 
recalc_pos = false;
store_fullsize = true;
}
}
}
if (mask_has_subset && !match_exactly && upd_header_rowcol) {
recalc_subset = true;
}
if (upd_header_rowcol) {
UpdateHeader(headers_, HEADER_RS_NROWS, new_rows);
UpdateHeader(headers_, HEADER_RS_NCOLS, new_cols);
headers_.at(HEADER_RS_XLL) += min_col * mask_->GetCellWidth();
headers_.at(HEADER_RS_YLL) += (mask_rows - max_row - 1) * mask_->GetCellWidth();
headers_.at(HEADER_RS_CELLSIZE) = mask_->GetCellWidth();
}

if (recalc_pos) {
auto rit = pos_rows.begin();
auto cit = pos_cols.begin();
auto vit = values.begin();
auto data2dit = values_2d.begin();
for (auto it = values.begin(); it != values.end();) {
size_t idx = distance(vit, it);
int tmpr = pos_rows.at(idx);
int tmpc = pos_cols.at(idx);
if (tmpr > max_row || tmpr < min_row || tmpc > max_col || tmpc < min_col
|| FloatEqual(static_cast<T>(*it), no_data_value_)) {
it = values.erase(it);
if (is_2draster && n_lyrs_ > 1) {
values_2d.erase(data2dit + idx);
data2dit = values_2d.begin();
}
pos_cols.erase(cit + idx);
pos_rows.erase(rit + idx);
rit = pos_rows.begin(); 
cit = pos_cols.begin();
vit = values.begin();
} else {
pos_rows[idx] -= min_row; 
pos_cols[idx] -= min_col;
++it;
}
}
vector<T>(values).swap(values);
if (is_2draster && n_lyrs_ > 1) { vector<vector<T> >(values_2d).swap(values_2d); }
vector<int>(pos_rows).swap(pos_rows);
vector<int>(pos_cols).swap(pos_cols);

store_pos_ = true;
n_cells_ = CVT_INT(values.size());
if (nullptr != pos_data_) { Release1DArray(pos_data_); }
Initialize2DArray(n_cells_, 2, pos_data_, 0);
for (size_t k = 0; k < pos_rows.size(); ++k) {
pos_data_[k][0] = pos_rows.at(k);
pos_data_[k][1] = pos_cols.at(k);
}
} else {
if (nullptr != pos_data_) { Release1DArray(pos_data_); }
if (calc_pos_) mask_->GetRasterPositionData(&n_cells_, &pos_data_);
store_pos_ = false;
}

assert(ValidateRasterData());
int ncols = CVT_INT(headers_.at(HEADER_RS_NCOLS));
int nrows = CVT_INT(headers_.at(HEADER_RS_NROWS));
if (store_fullsize) {
n_cells_ = ncols * nrows;
}
bool release_origin = true;
if (store_fullsize && old_fullsize == n_cells_) {
release_origin = false;
}
if (release_origin) {
if (is_2draster && nullptr != raster_2d_) { 
Release2DArray(raster_2d_);
Initialize2DArray(n_cells_, n_lyrs_, raster_2d_, no_data_value_);
} else { 
Release1DArray(raster_);
Initialize1DArray(n_cells_, raster_, no_data_value_);
}
int synthesis_idx = -1;
for (size_t k = 0; k < pos_rows.size(); ++k) {
synthesis_idx = k;
int tmpr = pos_rows.at(k);
int tmpc = pos_cols.at(k);
if (store_fullsize) {
if (tmpr > max_row || tmpr < min_row || tmpc > max_col || tmpc < min_col) {
continue;
}
synthesis_idx = (tmpr - min_row) * ncols + tmpc - min_col;
if (synthesis_idx > n_cells_ - 1) {
continue; 
}
}
if (is_2draster) { 
raster_2d_[synthesis_idx][0] = values.at(k);
if (n_lyrs_ > 1) {
for (int lyr = 1; lyr < n_lyrs_; lyr++) {
raster_2d_[synthesis_idx][lyr] = values_2d[k][lyr - 1];
}
}
} else { 
raster_[synthesis_idx] = values.at(k);
}
}
}

if (upd_header_valid_num) {
UpdateHeader(headers_, HEADER_RS_CELLSNUM, n_cells_);
}

if (mask_has_subset) { 
for (auto it = subset_.begin(); it != subset_.end();) {
int count = 0;
int srow = nrows;
int erow = -1;
int scol = ncols;
int ecol = -1;
vector<int> globalpos;
for (int i = 0; i < it->second->n_cells; i++) {
int gi = it->second->global_[i];
int tmprow = valid_pos[gi][0];
int tmpcol = valid_pos[gi][1];
XY_COOR tmpxy = mask_->GetCoordinateByRowCol(tmprow, tmpcol);
if (GetPosition(tmpxy.first, tmpxy.second) < 0) { continue; }
ROW_COL tmppos = GetPositionByCoordinate(tmpxy.first, tmpxy.second);
if (IsNoData(tmppos.first, tmppos.second)) { continue; }
if (tmppos.first < srow) { srow = tmppos.first; }
if (tmppos.first > erow) { erow = tmppos.first; }
if (tmppos.second < scol) { scol = tmppos.second; }
if (tmppos.second > ecol) { ecol = tmppos.second; }
globalpos.emplace_back(GetPosition(tmppos.first, tmppos.second));
count++;
}
if (count == 0) {
delete it->second;
subset_.erase(it++);
continue;
}
if (recalc_subset) {
it->second->g_srow = srow;
it->second->g_erow = erow;
it->second->g_scol = scol;
it->second->g_ecol = ecol;
it->second->n_cells = count;
it->second->n_lyrs = n_lyrs_;
if (it->second->alloc_) {
Release1DArray(it->second->global_);
Release2DArray(it->second->local_pos_);
}
it->second->global_ = nullptr; 
it->second->local_pos_ = nullptr;
it->second->alloc_ = true;
Initialize1DArray(count, it->second->global_, -1);
Initialize2DArray(count, 2, it->second->local_pos_, -1);
for (int ii = 0; ii < count; ii++) {
it->second->global_[ii] = globalpos[ii];
if (nullptr == pos_data_) {
it->second->local_pos_[ii][0] = globalpos[ii] / ncols - it->second->g_srow;
it->second->local_pos_[ii][1] = globalpos[ii] % ncols - it->second->g_scol;
} else {
it->second->local_pos_[ii][0] = pos_data_[globalpos[ii]][0] - it->second->g_srow;
it->second->local_pos_[ii][1] = pos_data_[globalpos[ii]][1] - it->second->g_scol;
}
}
it->second->data_ = nullptr;
it->second->data2d_ = nullptr;
it->second->alloc_ = true;
}
++it;
}
}
if (store_fullsize || recalc_pos) {
mask_ = nullptr;
}
return 2; 
}

} 
} 
#endif 
