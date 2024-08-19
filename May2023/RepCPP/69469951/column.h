


#pragma once

#include <vector>
#include "cell.h"


template <typename ValueType, size_t NStats>
class Column
{
public:
Column( bool *mat_finished );
Column( int currentRow, bool *mat_finished );
Column( const Column<ValueType, NStats>& source );

void init( short idStat );
void init();
void setValue( short idStat, ValueType semVal );
void setValue( const std::array<ValueType, NStats>& semVal, bool isNotZeroValue = true );
void addValue( short idStat, ValueType semVal );
void addValue( const std::array<ValueType, NStats>& semVal );
ValueType getCurrentValue( short idStat ) const;
std::array<ValueType, NStats> getCurrentValue() const;
int getCurrentRow( ) const;
bool currentCellModified( ) const;
void newRow( );
void newRow( int row );
void setNextCell( );
void setFirstCell( );
bool endCell( );

bool getCellValue( ValueType& semVal, int whichRow, short idStat ) const;
bool getNotZeroValue( int whichRow, short idStat ) const;
bool getCellValue( std::array<ValueType, NStats>& semVal, int whichRow ) const;

private:
std::vector<Cell<ValueType, NStats> > cells;
typename std::vector<Cell<ValueType, NStats> >::iterator it_cell;

Cell<ValueType, NStats> current_cell;
bool modified;
bool *finished;
};

#include "column_impl.h"
