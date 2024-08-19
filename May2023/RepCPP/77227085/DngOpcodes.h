

#pragma once

#include <cstdint>  
#include <map>      
#include <memory>   
#include <optional> 
#include <utility>  
#include <vector>   

namespace rawspeed {

class ByteStream;
class RawImage;
class iRectangle2D;

class DngOpcodes {
public:
DngOpcodes(const RawImage& ri, ByteStream bs);
~DngOpcodes();
void applyOpCodes(const RawImage& ri) const;

private:
class DngOpcode;

std::vector<std::unique_ptr<DngOpcode>> opcodes;

protected:
class DeltaRowOrColBase;
class DummyROIOpcode;
class FixBadPixelsConstant;
class FixBadPixelsList;
class LookupOpcode;
class PixelOpcode;
class PolynomialMap;
class ROIOpcode;
class TableMap;
class TrimBounds;
template <typename S> class DeltaRowOrCol;
template <typename S> class OffsetPerRowOrCol;
template <typename S> class ScalePerRowOrCol;

template <class Opcode>
static std::unique_ptr<DngOpcode>
constructor(const RawImage& ri, ByteStream& bs,
iRectangle2D& integrated_subimg);

using constructor_t = std::unique_ptr<DngOpcode> (*)(
const RawImage& ri, ByteStream& bs, iRectangle2D& integrated_subimg);
static std::optional<std::pair<const char*, DngOpcodes::constructor_t>>
Map(uint32_t code);
};

} 
