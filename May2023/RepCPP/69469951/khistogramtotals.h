


#pragma once


#include <vector>
#include "paraverkerneltypes.h"
#include "utils/include/vectorutils.h"
#include "histogramtotals.h"

class KHistogramTotals: public HistogramTotals
{
public:
KHistogramTotals( KHistogramTotals *& source );

KHistogramTotals( PRV_UINT16 numStat, THistogramColumn numColumns,
THistogramColumn numPlanes );
~KHistogramTotals();

void newValue( TSemanticValue whichValue,
PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 );
void finish();

TSemanticValue getTotal( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;
TSemanticValue getAverage( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;
TSemanticValue getMaximum( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;
TSemanticValue getMinimum( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;
TSemanticValue getStdev( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;
TSemanticValue getAvgDivMax( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;
void getAll( std::vector<TSemanticValue>& where,
PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const;

std::vector<int>& sortByTotal( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 );
std::vector<int>& sortByAverage( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 );
std::vector<int>& sortByMaximum( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 );
std::vector<int>& sortByMinimum( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 );
std::vector<int>& sortByStdev( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 );
std::vector<int>& sortByAvgDivMax( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 );

protected:

private:
THistogramColumn columns;
PRV_UINT16 stats;
std::vector<std::vector<std::vector<TSemanticValue> > > total;
std::vector<std::vector<std::vector<TSemanticValue> > > average;
std::vector<std::vector<std::vector<TSemanticValue> > > maximum;
std::vector<std::vector<std::vector<TSemanticValue> > > minimum;
std::vector<std::vector<std::vector<TSemanticValue> > > stdev;

SortIndex<TSemanticValue> *sort;
std::vector<int> nullSort;
};



