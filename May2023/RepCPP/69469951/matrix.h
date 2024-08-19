


#pragma once

#include <vector>
#include "column.h"
#include "paraverkerneltypes.h"

template <typename ValueType, size_t NStats>
class Matrix
{
public:
Matrix( PRV_UINT32 numCols );
Matrix( TObjectOrder currentRow, PRV_UINT32 numCols );
Matrix( Matrix<ValueType, NStats>& source );

void init( PRV_UINT16 idStat );
void init( );
void setValue( PRV_UINT32 col, PRV_UINT16 idStat, ValueType semVal );
void setValue( PRV_UINT32 col, const std::array<ValueType, NStats>& semVal, bool isNotZeroValue = true );
void addValue( PRV_UINT32 col, PRV_UINT16 idStat, ValueType semVal );
void addValue( PRV_UINT32 col, const std::array<ValueType, NStats>& semVal );
ValueType getCurrentValue( PRV_UINT32 col, PRV_UINT16 idStat ) const;
std::array<ValueType, NStats> getCurrentValue( PRV_UINT32 col ) const;
TObjectOrder getCurrentRow( PRV_UINT32 col ) const;
bool currentCellModified( PRV_UINT32 col ) const;
void newRow( );
void newRow( PRV_UINT32 col, TObjectOrder row );
void finish( );
void setNextCell( PRV_UINT32 col );
void setFirstCell( PRV_UINT32 col );
bool endCell( PRV_UINT32 col );
void eraseColumns( PRV_UINT32 ini_col, PRV_UINT32 fin_col );

bool getCellValue( ValueType& semVal, int whichRow, PRV_UINT32 whichCol, PRV_UINT16 idStat ) const;
bool getNotZeroValue( int whichRow, PRV_UINT32 whichCol, PRV_UINT16 idStat ) const;
bool getCellValue( std::array<ValueType, NStats>& semVal, int whichRow, PRV_UINT32 whichCol ) const;

private:
std::vector<Column<ValueType, NStats> > cols;
bool finished;
};

#include "matrix_impl.h"
