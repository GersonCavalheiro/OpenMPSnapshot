


#pragma once


#include <string>
#include "output.h"
#include "selectionmanagement.h"
#include "histogram.h"
#include "progresscontroller.h"

class Timeline;
class Histogram;

class TextOutput:public Output
{
public:
TextOutput();
virtual ~TextOutput();

void dumpWindow( Timeline *whichWindow,
std::string& strOutputFile,
ProgressController *progress = nullptr );
void dumpHistogram( Histogram *whichHisto,
std::string& strOutputFile,
bool onlySelectedPlane = false,
bool hideEmptyColumns = false,
bool withLabels = true,
bool withPreferencesPrecision = true,
bool recalcHisto = true,
ProgressController *progress = nullptr );

bool getMultipleFiles() const;
void setMultipleFiles( bool newValue );

bool getObjectHierarchy() const;
void setObjectHierarchy( bool newValue );

bool getWindowTimeUnits() const;
void setWindowTimeUnits( bool newValue );

bool getTextualSemantic() const;
void setTextualSemantic( bool newValue );

TTime getMinTime() const;
TTime getMaxTime() const;

protected:

private:
typedef TSemanticValue (HistogramTotals::*THistogramTotalsMethod)( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane ) const;
bool multipleFiles;
bool objectHierarchy;
bool windowTimeUnits;
bool textualSemantic;

TTime minTime;
TTime maxTime;

void dumpMatrixHorizontal( Histogram *whichHisto,
TObjectOrder numRows,
THistogramColumn numColumns,
PRV_UINT16 currentStat,
std::vector<THistogramColumn> printedColumns,
THistogramColumn iPlane,
std::ofstream &outputfile,
bool withLabels,
ProgressController *progress = nullptr );

void dumpMatrixVertical( Histogram *whichHisto,
TObjectOrder numRows,
THistogramColumn numColumns,
PRV_UINT16 currentStat,
std::vector<THistogramColumn> printedColumns,
THistogramColumn iPlane,
std::ofstream &outputfile,
bool withLabels,
ProgressController *progress = nullptr );

void dumpTotalColumns( Histogram *whichHisto,
HistogramTotals *totals,
std::string totalName,
THistogramTotalsMethod totalFunction,
PRV_UINT16 currentStat,
std::vector<THistogramColumn> printedColumns,
THistogramColumn iPlane,
std::ofstream &outputFile,
ProgressController *progress = nullptr );

void dumpTotalRows( HistogramTotals *totals,
std::string totalName,
THistogramTotalsMethod totalFunction,
PRV_UINT16 currentStat,
TObjectOrder numRows,
THistogramColumn iPlane,
std::ofstream &outputFile,
ProgressController *progress = nullptr );

void dumpMatrixCommHorizontal( Histogram *whichHisto,
TObjectOrder numRows,
THistogramColumn numColumns,
PRV_UINT16 currentStat,
std::vector<THistogramColumn> printedColumns,
THistogramColumn iPlane,
std::ofstream &outputfile,
bool withLabels,
ProgressController *progress = nullptr );

void dumpMatrixCommVertical( Histogram *whichHisto,
TObjectOrder numRows,
THistogramColumn numColumns,
PRV_UINT16 currentStat,
std::vector<THistogramColumn> printedColumns,
THistogramColumn iPlane,
std::ofstream &outputfile,
bool withLabels,
ProgressController *progress = nullptr );

};


