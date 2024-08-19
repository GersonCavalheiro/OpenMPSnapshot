

#pragma once

#include "paraverkerneltypes.h"

template <typename ValueType, size_t NStats>
class Cell
{
public:
Cell();
Cell( TObjectOrder idRow );
Cell( const Cell< ValueType, NStats >& source );

void init( PRV_UINT16 idStat );
void init( );
void setValue( PRV_UINT16 idStat, ValueType semVal );
void setValue( const std::array<ValueType, NStats>& semVal, bool isNotZeroValue = true );
void addValue( PRV_UINT16 idStat, ValueType semVal );
void addValue( const std::array<ValueType, NStats>& semVal );
ValueType getValue( PRV_UINT16 idStat ) const;
bool getNotZeroValue( PRV_UINT16 idStat ) const;
std::array<ValueType, NStats> getValue() const;
TObjectOrder getRow( ) const;
void setRow( TObjectOrder row );

bool operator==( const ValueType& anotherCell ) const;
bool operator<( const ValueType& anotherCell ) const;

private:
TObjectOrder row;
std::array<ValueType, NStats> values;
bool isNotZeroValue;
};

#include "cell_impl.h"


