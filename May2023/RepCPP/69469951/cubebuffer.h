

#pragma once

#include <array>
#include <unordered_map>
#include <vector>

#include "paraverkerneltypes.h" 

template< size_t NStats >
class CubeBuffer
{
public:
CubeBuffer( PRV_UINT32 numPlanes, PRV_UINT32 numRows );

void addValue( PRV_UINT32 plane, PRV_UINT32 row, THistogramColumn col, const std::array< TSemanticValue, NStats >& semVal, bool isNotZeroValue = true );
void setValue( PRV_UINT32 plane, PRV_UINT32 row, THistogramColumn col, const std::array< TSemanticValue, NStats >& semVal );
bool getCellValue( std::array< TSemanticValue, NStats >& semVal, PRV_UINT32 plane, PRV_UINT32 row, PRV_UINT32 col ) const;

const std::unordered_map< THistogramColumn, std::array< TSemanticValue, NStats > >& getRowValues( PRV_UINT32 plane, PRV_UINT32 row ) const;
const std::unordered_map< THistogramColumn, bool >& getNotZeroValue( PRV_UINT32 plane, PRV_UINT32 row ) const;

private:
std::vector< std::vector< std::unordered_map< THistogramColumn, std::array< TSemanticValue, NStats > > > > buffer;
std::vector< std::vector< std::unordered_map< THistogramColumn, bool > > > bufferNotZeroValue;
};

#include "cubebuffer_impl.h"
