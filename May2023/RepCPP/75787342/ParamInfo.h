
#ifndef SEIMS_PARAMETER_INFO_H
#define SEIMS_PARAMETER_INFO_H
#include <vector>
#include <algorithm>

#include "utils_math.h"
#include "utils_string.h"

#include "MetadataInfoConst.h"
#include "basic.h"
#include "text.h"

using namespace ccgl;
using namespace utils_math;
using namespace utils_string;
using std::vector;
using std::map;


template <typename T>
class ParamInfo {
public:
ParamInfo();

ParamInfo(string& name, string& desc, string& unit, string& mid, T value, string& change, T impact,
T maximum, T minimum, bool isint);

ParamInfo(string& name, string& basicname, string& desc, string& unit, string& source, string& mid,
dimensionTypes dim, string& climtype, T value = 0);

ParamInfo(string& name, string& basicname, string& desc, string& unit, string& source, string& mid,
dimensionTypes dim, transferTypes tftype, string& climtype,
bool isconst, bool isoutput);

ParamInfo(const ParamInfo& another);

~ParamInfo();


T GetAdjustedValue(T pre_value = NODATA_VALUE);

T GetAdjustedValueWithImpactIndexes(T pre_value, int curImpactIndex);

void Adjust1DArray(int n, T* data);

void Adjust1DRaster(int n, T* data);

int Adjust1DRaster(int n, T* data, const int* units, const vector<int>& selunits,
const int* lu, const vector<int>& sellu);

int Adjust1DRasterWithImpactIndexes(int n, T* data, const int* units,
const vector<int>& selunits, const map<int, int>& impactIndexes,
const int* lu, const vector<int>& sellu);

void Adjust2DArray(int n, T** data);

void Adjust2DRaster(int n, int lyrs, T** data);

int Adjust2DRaster(int n, int lyrs, T** data, const int* units, const vector<int>& selunits,
const int* lu, const vector<int>& sellu);

int Adjust2DRasterWithImpactIndexes(int n, int lyrs, T** data, const int* units,
const vector<int>& selunits, const map<int, int>& impactIndexes,
const int* lu, const vector<int>& sellu);

string Name;
string BasicName;
string Description;
string Units;
string Source;
string ModuleID;
dimensionTypes Dimension;
transferTypes Transfer;
T Value;
string Change;
T Impact;
T Maximum;
T Minimum;
bool IsInteger;
ParamInfo* DependPara;
string ClimateType;
bool IsConstant;
bool IsOutput;
bool OutputToOthers;
bool initialized;
vector<T> ImpactSeries;
};






template <typename T>
ParamInfo<T>::ParamInfo() :
Name(""), BasicName(""), Description(""), Units(""), Source(""), ModuleID(""), 
Dimension(DT_Unknown), Transfer(TF_None),
Value(0), Change(""), Impact(0), Maximum(0), Minimum(0), IsInteger(false),
DependPara(nullptr), ClimateType(""),
IsConstant(false), IsOutput(false), OutputToOthers(false),
initialized(false), ImpactSeries() {
}

template <typename T>
ParamInfo<T>::ParamInfo(string& name, string& desc, string& unit, string& mid, T value, string& change,
T impact, T maximum, T minimum, bool isint):
Name(name), BasicName(""), Description(desc), Units(unit), Source(""), ModuleID(mid),
Dimension(DT_Unknown), Transfer(TF_None),
Value(value), Change(change), Impact(impact), Maximum(maximum), Minimum(minimum), IsInteger(isint),
DependPara(nullptr), ClimateType(""),
IsConstant(false), IsOutput(false), OutputToOthers(false),
initialized(false), ImpactSeries() {

}

template <typename T>
ParamInfo<T>::ParamInfo(string& name, string& basicname, string& desc, string& unit, string& source, string& mid,
dimensionTypes dim, string& climtype, T value ):
Name(name), BasicName(basicname), Description(desc), Units(unit), Source(source), ModuleID(mid), 
Dimension(dim), Transfer(TF_None),
Value(value), Change(""), Impact(0), Maximum(0), Minimum(0), IsInteger(false),
DependPara(nullptr), ClimateType(climtype),
IsConstant(false), IsOutput(false), OutputToOthers(false),
initialized(false), ImpactSeries() {

}

template <typename T>
ParamInfo<T>::ParamInfo(string& name, string& basicname, string& desc, string& unit, string& source, string& mid,
dimensionTypes dim, transferTypes tftype, string& climtype,
bool isconst, bool isoutput) :
Name(name), BasicName(basicname), Description(desc), Units(unit), Source(source), ModuleID(mid),
Dimension(dim), Transfer(tftype),
Value(0), Change(""), Impact(0), Maximum(0), Minimum(0), IsInteger(false),
DependPara(nullptr), ClimateType(climtype),
IsConstant(isconst), IsOutput(isoutput), OutputToOthers(false),
initialized(false), ImpactSeries() {

}

template <typename T>
ParamInfo<T>::ParamInfo(const ParamInfo<T>& another) {
Name = another.Name;
BasicName = another.BasicName;
Description = another.Description;
Units = another.Units;
Source = another.Source;
ModuleID = another.ModuleID;
Dimension = another.Dimension;
Transfer = another.Transfer;
Value = another.Value;
Change = another.Change;
Impact = another.Impact;
Maximum = another.Maximum;
Minimum = another.Minimum;
IsInteger = another.IsInteger;
DependPara = another.DependPara;
ClimateType = another.ClimateType;
IsConstant = another.IsConstant;
IsOutput = another.IsOutput;
OutputToOthers = another.OutputToOthers;
initialized = another.initialized;
ImpactSeries = another.ImpactSeries;
}

template <typename T>
ParamInfo<T>::~ParamInfo() {
if (DependPara != nullptr) {
DependPara = nullptr;
}
}

template <typename T>
T ParamInfo<T>::GetAdjustedValue(const T pre_value ) {
T res = pre_value;
if (FloatEqual(pre_value, NODATA_VALUE)) {
res = Value;
}
if (FloatEqual(res, NODATA_VALUE)) {
return res;
}

if (StringMatch(Change, PARAM_CHANGE_RC) && !FloatEqual(Impact, 1.)) {
res *= Impact;
}
else if (StringMatch(Change, PARAM_CHANGE_AC) && !FloatEqual(Impact, 0.)) {
res += Impact;
}
else if (StringMatch(Change, PARAM_CHANGE_VC) && !FloatEqual(Impact, NODATA_VALUE)) {
res = Impact;
}
else if (StringMatch(Change, PARAM_CHANGE_NC)) {
return res;
}

if (!FloatEqual(Maximum, NODATA_VALUE) && res > Maximum) res = Maximum;
if (!FloatEqual(Minimum, NODATA_VALUE) && res < Minimum) res = Minimum;
return res;
}

template <typename T>
T ParamInfo<T>::GetAdjustedValueWithImpactIndexes(const T pre_value, const int curImpactIndex) {
T res = pre_value;
if (FloatEqual(pre_value, NODATA_VALUE)) {
res = Value;
}
if (FloatEqual(res, NODATA_VALUE)) {
return res;
}

T tmpImpact = ImpactSeries[curImpactIndex];
if (StringMatch(Change, PARAM_CHANGE_RC) && !FloatEqual(tmpImpact, 1)) {
res *= tmpImpact;
}
else if (StringMatch(Change, PARAM_CHANGE_AC) && !FloatEqual(tmpImpact, 0)) {
res += tmpImpact;
}
else if (StringMatch(Change, PARAM_CHANGE_VC) && !FloatEqual(tmpImpact, NODATA_VALUE)) {
res = tmpImpact;
}
else if (StringMatch(Change, PARAM_CHANGE_NC)) { 
return res;
}

if (!FloatEqual(Maximum, NODATA_VALUE) && res > Maximum) res = Maximum;
if (!FloatEqual(Minimum, NODATA_VALUE) && res < Minimum) res = Minimum;
return res;
}

template <typename T>
void ParamInfo<T>::Adjust1DArray(const int n, T* data) {
#pragma omp parallel for
for (int i = 0; i < n; i++) {
if (!FloatEqual(data[i], NODATA_VALUE)) {
data[i] = GetAdjustedValue(data[i]);
}
}
}

template <typename T>
void ParamInfo<T>::Adjust1DRaster(const int n, T* data) {
Adjust1DArray(n, data);
}

template <typename T>
int ParamInfo<T>::Adjust1DRaster(const int n, T* data, const int* units,
const vector<int>& selunits,
const int* lu, const vector<int>& sellu) {
int count = 0;
for (int i = 0; i < n; i++) {
if (FloatEqual(data[i], NODATA_VALUE)) {
continue;
}
int curunit = units[i];
int curlu = lu[i];
if (find(selunits.begin(), selunits.end(), curunit) == selunits.end()) {
continue;
}
if (find(sellu.begin(), sellu.end(), curlu) == sellu.end()) {
continue;
}
data[i] = GetAdjustedValue(data[i]);
count += 1;
}
return count;
}

template <typename T>
int ParamInfo<T>::Adjust1DRasterWithImpactIndexes(const int n, T* data, const int* units,
const vector<int>& selunits, const map<int, int>& impactIndexes,
const int* lu, const vector<int>& sellu) {
int count = 0;
for (int i = 0; i < n; i++) {
if (FloatEqual(data[i], NODATA_VALUE)) {
continue;
}
int curunit = units[i];
int curlu = lu[i];
if (find(selunits.begin(), selunits.end(), curunit) == selunits.end()) {
continue;
}
if (find(sellu.begin(), sellu.end(), curlu) == sellu.end()) {
continue;
}
map<int, int>::const_iterator it = impactIndexes.find(curunit);
if (it == impactIndexes.end()) {
continue;
}
data[i] = GetAdjustedValueWithImpactIndexes(data[i], it->second);
count += 1;
}
return count;
}

template <typename T>
void ParamInfo<T>::Adjust2DArray(const int n, T** data) {
#pragma omp parallel for
for (int i = 0; i < n; i++) {
Adjust1DArray(CVT_INT(data[i][0]), data[i] + 1);
}
}

template <typename T>
void ParamInfo<T>::Adjust2DRaster(const int n, const int lyrs, T** data) {
#pragma omp parallel for
for (int i = 0; i < n; i++) {
Adjust1DArray(lyrs, data[i]);
}
}

template <typename T>
int ParamInfo<T>::Adjust2DRaster(const int n, const int lyrs, T** data, 
const int* units, const vector<int>& selunits,
const int* lu, const vector<int>& sellu) {
int count = 0;
for (int i = 0; i < n; i++) {
int curunit = units[i];
int curlu = lu[i];
if (find(selunits.begin(), selunits.end(), curunit) == selunits.end()) {
continue;
}
if (find(sellu.begin(), sellu.end(), curlu) == sellu.end()) {
continue;
}
for (int j = 0; j < lyrs; j++) {
data[i][j] = GetAdjustedValue(data[i][j]);
}
count += 1;
}
return count;
}

template <typename T>
int ParamInfo<T>::Adjust2DRasterWithImpactIndexes(const int n, const int lyrs, T** data,
const int* units, const vector<int>& selunits,
const map<int, int>& impactIndexes,
const int* lu, const vector<int>& sellu) {
int count = 0;
for (int i = 0; i < n; i++) {
int curunit = units[i];
int curlu = lu[i];
if (find(selunits.begin(), selunits.end(), curunit) == selunits.end()) {
continue;
}
if (find(sellu.begin(), sellu.end(), curlu) == sellu.end()) {
continue;
}
map<int, int>::const_iterator it = impactIndexes.find(curunit);
if (it == impactIndexes.end()) {
continue;
}
for (int j = 0; j < lyrs; j++) {
data[i][j] = GetAdjustedValueWithImpactIndexes(data[i][j], it->second);
}
count += 1;
}
return count;
}

#endif 
