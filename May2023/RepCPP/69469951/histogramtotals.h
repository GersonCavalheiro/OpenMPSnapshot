


#pragma once


#include "paraverkerneltypes.h"

class HistogramTotals
{
public:
static HistogramTotals *create( HistogramTotals *whichTotals );

virtual ~HistogramTotals() {}

virtual TSemanticValue getTotal( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;
virtual TSemanticValue getAverage( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;
virtual TSemanticValue getMaximum( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;
virtual TSemanticValue getMinimum( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;
virtual TSemanticValue getStdev( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;
virtual TSemanticValue getAvgDivMax( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;
virtual void getAll( std::vector<TSemanticValue>& where,
PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const = 0;

virtual std::vector<int>& sortByTotal( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) = 0;
virtual std::vector<int>& sortByAverage( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) = 0;
virtual std::vector<int>& sortByMaximum( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) = 0;
virtual std::vector<int>& sortByMinimum( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) = 0;
virtual std::vector<int>& sortByStdev( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) = 0;
virtual std::vector<int>& sortByAvgDivMax( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) = 0;

protected:

private:

};

class HistogramTotalsProxy: public HistogramTotals
{
public:
virtual ~HistogramTotalsProxy() {}

virtual TSemanticValue getTotal( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;
virtual TSemanticValue getAverage( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;
virtual TSemanticValue getMaximum( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;
virtual TSemanticValue getMinimum( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;
virtual TSemanticValue getStdev( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;
virtual TSemanticValue getAvgDivMax( PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;
virtual void getAll( std::vector<TSemanticValue>& where,
PRV_UINT16 idStat,
THistogramColumn whichColumn,
THistogramColumn whichPlane = 0 ) const override;

virtual std::vector<int>& sortByTotal( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) override;
virtual std::vector<int>& sortByAverage( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) override;
virtual std::vector<int>& sortByMaximum( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) override;
virtual std::vector<int>& sortByMinimum( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) override;
virtual std::vector<int>& sortByStdev( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) override;
virtual std::vector<int>& sortByAvgDivMax( PRV_UINT16 idStat, THistogramColumn whichPlane = 0 ) override;

protected:

private:
HistogramTotalsProxy( HistogramTotals *whichTotals );

HistogramTotals *myTotals;

friend HistogramTotals *HistogramTotals::create( HistogramTotals * );
};


