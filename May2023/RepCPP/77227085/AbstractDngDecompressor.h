

#pragma once

#include "adt/Invariant.h"                      
#include "adt/Point.h"                          
#include "common/Common.h"                      
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <cstdint>                              
#include <utility>                              
#include <vector>                               

namespace rawspeed {

struct DngTilingDescription final {
const iPoint2D& dim;

const uint32_t tileW;

const uint32_t tileH;

const uint32_t tilesX;

const uint32_t tilesY;

const unsigned numTiles;

DngTilingDescription(const iPoint2D& dim_, uint32_t tileW_, uint32_t tileH_)
: dim(dim_), tileW(tileW_), tileH(tileH_),
tilesX(roundUpDivision(dim.x, tileW)),
tilesY(roundUpDivision(dim.y, tileH)), numTiles(tilesX * tilesY) {
invariant(dim.area() > 0);
invariant(tileW > 0);
invariant(tileH > 0);
invariant(tilesX > 0);
invariant(tilesY > 0);
invariant(tileW * tilesX >= static_cast<unsigned>(dim.x));
invariant(tileH * tilesY >= static_cast<unsigned>(dim.y));
invariant(tileW * (tilesX - 1) < static_cast<unsigned>(dim.x));
invariant(tileH * (tilesY - 1) < static_cast<unsigned>(dim.y));
invariant(numTiles > 0);
}
};

struct DngSliceElement final {
const DngTilingDescription& dsc;

const unsigned n;

const ByteStream bs;

const unsigned column;
const unsigned row;

const bool lastColumn;
const bool lastRow;

const unsigned offX;
const unsigned offY;

const unsigned width;
const unsigned height;

DngSliceElement() = delete;
DngSliceElement(const DngSliceElement&) = default;
DngSliceElement(DngSliceElement&&) noexcept = default;
DngSliceElement& operator=(const DngSliceElement&) noexcept = delete;
DngSliceElement& operator=(DngSliceElement&&) noexcept = delete;

DngSliceElement(const DngTilingDescription& dsc_, unsigned n_, ByteStream bs_)
: dsc(dsc_), n(n_), bs(bs_), column(n % dsc.tilesX), row(n / dsc.tilesX),
lastColumn((column + 1) == dsc.tilesX),
lastRow((row + 1) == dsc.tilesY), offX(dsc.tileW * column),
offY(dsc.tileH * row),
width(!lastColumn ? dsc.tileW : dsc.dim.x - offX),
height(!lastRow ? dsc.tileH : dsc.dim.y - offY) {
invariant(n < dsc.numTiles);
invariant(bs.getRemainSize() > 0);
invariant(column < dsc.tilesX);
invariant(row < dsc.tilesY);
invariant(offX < static_cast<unsigned>(dsc.dim.x));
invariant(offY < static_cast<unsigned>(dsc.dim.y));
invariant(width > 0);
invariant(height > 0);
invariant(offX + width <= static_cast<unsigned>(dsc.dim.x));
invariant(offY + height <= static_cast<unsigned>(dsc.dim.y));
invariant(!lastColumn ||
(offX + width == static_cast<unsigned>(dsc.dim.x)));
invariant(!lastRow || (offY + height == static_cast<unsigned>(dsc.dim.y)));
}
};

class AbstractDngDecompressor final : public AbstractDecompressor {
RawImage mRaw;

template <int compression> void decompressThread() const noexcept;

void decompressThread() const noexcept;

public:
AbstractDngDecompressor(const RawImage& img, const DngTilingDescription& dsc_,
int compression_, bool mFixLjpeg_, uint32_t mBps_,
uint32_t mPredictor_)
: mRaw(img), dsc(dsc_), compression(compression_), mFixLjpeg(mFixLjpeg_),
mBps(mBps_), mPredictor(mPredictor_) {}

void decompress() const;

const DngTilingDescription dsc;

std::vector<DngSliceElement> slices;

const int compression;
const bool mFixLjpeg = false;
const uint32_t mBps;
const uint32_t mPredictor;
};

} 
