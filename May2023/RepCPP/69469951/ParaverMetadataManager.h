




#pragma once


#include <string>
using std::string;
#include <vector>
using std::vector;
#include <ostream>
using std::ostream;

#include "paraverkerneltypes.h"


class Metadata
{
public:
static int FIELD_COUNT;

protected:
bool   Error;
string ErrorMessage;

string Date;
string Action;
string Application;
string OriginalTrace;

public:
Metadata (void) { Error = false; };

Metadata (string Date,
string Action,
string Application,
string OriginalTrace);

string GetDate(void)          { return Date; };
string GetAction(void)        { return Action; };
string GetApplication(void)   { return Application; };
string GetOriginalTrace(void) { return OriginalTrace; };

bool   GetError(void)         { return Error; };
string GetErrorMessage(void)  { return ErrorMessage; };

void Write(ostream& os) const;

private:

virtual void FlushSpecificFields(ostream& os) const = 0;
};

ostream& operator<< (ostream& os, const Metadata& MetadataRecord);

class CutterMetadata: public Metadata
{
public:
static int    FIELD_COUNT;
static string ACTION_ID;
static string RUNAPP_APPLICATION_ID;
static string ORIGINAL_APPLICATION_ID;

private:
PRV_UINT64 Offset;
PRV_UINT64 BeginTime;
PRV_UINT64 EndTime;

public:
CutterMetadata (vector<string>& CutterMetadataFields);


CutterMetadata (string Date,
string Application,
string OriginalTrace,
PRV_UINT64 Offset,
PRV_UINT64 BeginTime,
PRV_UINT64 EndTime);


PRV_UINT64 GetOffset(void)    { return Offset; };
PRV_UINT64 GetBeginTime(void) { return BeginTime; };
PRV_UINT64 GetEndTime(void)   { return EndTime; };

private:

void FlushSpecificFields(ostream& os) const;
};

class MetadataManager
{
private:

bool   Error;
string ErrorMessage;

vector<Metadata*>       TraceMetadataStorage;
vector<CutterMetadata*> CutterMetadataStorage;
PRV_UINT64 lastOffset;
PRV_UINT64 lastBeginTime;
PRV_UINT64 lastEndTime;
PRV_UINT64 totalOffset;

public:
MetadataManager(void) : Error( false ), ErrorMessage( "" ), totalOffset(0) {};

bool NewMetadata(string MetadataStr);

vector<Metadata*>& GetMetadata(void) { return TraceMetadataStorage; };

vector<CutterMetadata*>& GetCutterMetadata(void) { return CutterMetadataStorage; };
PRV_UINT64 GetCutterLastOffset(void) const { return lastOffset; };
PRV_UINT64 GetCutterLastBeginTime(void) const { return lastBeginTime; };
PRV_UINT64 GetCutterLastEndTime(void) const { return lastEndTime; };
PRV_UINT64 GetCutterTotalOffset(void) const { return totalOffset; };

bool GetError         (void) { return Error; };
string GetErrorMessage(void) { return ErrorMessage; };

static string GetCurrentDate();

private:

void PopulateRecord (vector<string> &Record,
const string   &Line,
char            Delimiter);
};


