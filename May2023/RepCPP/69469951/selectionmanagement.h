


#pragma once


#include <vector>
#include "paraverkerneltypes.h"

class Trace;
class HistogramTotals;

template <typename SelType, typename LevelType>
class SelectionManagement
{
public:
SelectionManagement();
SelectionManagement( const SelectionManagement& whichSelection );
~SelectionManagement();


void init( Trace *trace );
void init( HistogramTotals *totals,
PRV_UINT16 idStat,
THistogramColumn numColumns,
THistogramColumn whichPlane );

void copy( const SelectionManagement &selection );
bool operator== ( const SelectionManagement<SelType, LevelType> &selection ) const;

void setSelected( std::vector< bool > &selection, LevelType level = (LevelType)0 );
void setSelected( std::vector< SelType > &selection, SelType maxElems, LevelType level = (LevelType)0 );

void getSelected( std::vector< bool > &selection, LevelType level = (LevelType)0 ) const;
void getSelected( std::vector< bool > &selection, SelType first, SelType last, LevelType level = (LevelType)0 ) const;
void getSelected( std::vector< SelType > &selection, LevelType level = (LevelType)0 ) const;
void getSelected( std::vector< SelType > &selection, SelType first, SelType last, LevelType level = (LevelType)0 ) const;

bool isSelectedPosition( SelType whichSelected, LevelType level = (LevelType)0 ) const;

SelType shiftFirst( SelType whichFirst, PRV_INT64 shiftAmount, PRV_INT64& appliedAmount, LevelType level = (LevelType)0 ) const;
SelType shiftLast( SelType whichLast, PRV_INT64 shiftAmount, PRV_INT64& appliedAmount, LevelType level = (LevelType)0 ) const;

private:
std::vector< std::vector< bool > > selected;
std::vector< std::vector< SelType > > selectedSet;
};

#include "selectionmanagement_impl.h"
