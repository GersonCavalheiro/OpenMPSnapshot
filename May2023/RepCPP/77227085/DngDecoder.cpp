

#include "rawspeedconfig.h" 
#include "decoders/DngDecoder.h"
#include "adt/NORangesSet.h"                       
#include "adt/NotARational.h"                      
#include "adt/Point.h"                             
#include "common/Common.h"                         
#include "common/DngOpcodes.h"                     
#include "decoders/RawDecoderException.h"          
#include "decompressors/AbstractDngDecompressor.h" 
#include "io/Buffer.h"                             
#include "io/ByteStream.h"                         
#include "metadata/BlackArea.h"                    
#include "metadata/Camera.h"                       
#include "metadata/CameraMetaData.h"               
#include "metadata/ColorFilterArray.h"             
#include "parsers/TiffParserException.h"           
#include "tiff/TiffEntry.h"                        
#include "tiff/TiffIFD.h"                          
#include "tiff/TiffTag.h"                          
#include <algorithm>                               
#include <array>                                   
#include <cassert>                                 
#include <limits>                                  
#include <map>                                     
#include <memory>                                  
#include <string>                                  
#include <utility>                                 
#include <vector>                                  

using std::map;
using std::vector;

namespace rawspeed {

bool RAWSPEED_READONLY DngDecoder::isAppropriateDecoder(
const TiffRootIFD* rootIFD, [[maybe_unused]] Buffer file) {
return rootIFD->hasEntryRecursive(TiffTag::DNGVERSION);
}

DngDecoder::DngDecoder(TiffRootIFDOwner&& rootIFD, Buffer file)
: AbstractTiffDecoder(std::move(rootIFD), file) {
if (!mRootIFD->hasEntryRecursive(TiffTag::DNGVERSION))
ThrowRDE("DNG, but version tag is missing. Will not guess.");

const uint8_t* v =
mRootIFD->getEntryRecursive(TiffTag::DNGVERSION)->getData().getData(4);

if (v[0] != 1)
ThrowRDE("Not a supported DNG image format: v%u.%u.%u.%u", (int)v[0],
(int)v[1], (int)v[2], (int)v[3]);

mFixLjpeg = (v[0] <= 1) && (v[1] < 1);
}

void DngDecoder::dropUnsuportedChunks(std::vector<const TiffIFD*>* data) {
for (auto i = data->begin(); i != data->end();) {
const auto& ifd = *i;

int comp = ifd->getEntry(TiffTag::COMPRESSION)->getU16();
bool isSubsampled = false;
bool isAlpha = false;

if (ifd->hasEntry(TiffTag::NEWSUBFILETYPE) &&
ifd->getEntry(TiffTag::NEWSUBFILETYPE)->isInt()) {
const uint32_t NewSubFileType =
(*i)->getEntry(TiffTag::NEWSUBFILETYPE)->getU32();

isSubsampled = NewSubFileType & (1 << 0);

isAlpha = NewSubFileType & (1 << 2);
}

bool supported = !isSubsampled && !isAlpha;

switch (comp) {
case 1: 
case 7: 
#ifdef HAVE_ZLIB
case 8: 
#endif
case 9: 
#ifdef HAVE_JPEG
case 0x884c: 
#endif
break;

#ifndef HAVE_ZLIB
case 8: 
#pragma message                                                                \
"ZLIB is not present! Deflate compression will not be supported!"
writeLog(DEBUG_PRIO::WARNING, "DNG Decoder: found Deflate-encoded chunk, "
"but the deflate support was disabled at "
"build!");
[[clang::fallthrough]];
#endif
#ifndef HAVE_JPEG
case 0x884c: 
#pragma message                                                                \
"JPEG is not present! Lossy JPEG compression will not be supported!"
writeLog(DEBUG_PRIO::WARNING, "DNG Decoder: found lossy JPEG-encoded "
"chunk, but the jpeg support was "
"disabled at build!");
[[clang::fallthrough]];
#endif
default:
supported = false;
break;
}

if (supported)
++i;
else
i = data->erase(i);
}
}

std::optional<iRectangle2D>
DngDecoder::parseACTIVEAREA(const TiffIFD* raw) const {
if (!raw->hasEntry(TiffTag::ACTIVEAREA))
return {};

const TiffEntry* active_area = raw->getEntry(TiffTag::ACTIVEAREA);
if (active_area->count != 4)
ThrowRDE("active area has %d values instead of 4", active_area->count);

const iRectangle2D fullImage(0, 0, mRaw->dim.x, mRaw->dim.y);

const auto corners = active_area->getU32Array(4);
const iPoint2D topLeft(static_cast<int>(corners[1]),
static_cast<int>(corners[0]));
const iPoint2D bottomRight(static_cast<int>(corners[3]),
static_cast<int>(corners[2]));

if (!(fullImage.isPointInsideInclusive(topLeft) &&
fullImage.isPointInsideInclusive(bottomRight) &&
bottomRight >= topLeft)) {
ThrowRDE("Rectangle (%u, %u, %u, %u) not inside image (%u, %u, %u, %u).",
topLeft.x, topLeft.y, bottomRight.x, bottomRight.y,
fullImage.getTopLeft().x, fullImage.getTopLeft().y,
fullImage.getBottomRight().x, fullImage.getBottomRight().y);
}

iRectangle2D crop;
crop.setTopLeft(topLeft);
crop.setBottomRightAbsolute(bottomRight);
assert(fullImage.isThisInside(fullImage));

return crop;
}

void DngDecoder::parseCFA(const TiffIFD* raw) const {

if (raw->hasEntry(TiffTag::CFALAYOUT) &&
raw->getEntry(TiffTag::CFALAYOUT)->getU16() != 1)
ThrowRDE("Unsupported CFA Layout.");

const TiffEntry* cfadim = raw->getEntry(TiffTag::CFAREPEATPATTERNDIM);
if (cfadim->count != 2)
ThrowRDE("Couldn't read CFA pattern dimension");

const TiffEntry* cPat = raw->getEntry(TiffTag::CFAPATTERN);
if (!cPat->count)
ThrowRDE("CFA pattern is empty!");

iPoint2D cfaSize(cfadim->getU32(1), cfadim->getU32(0));
if (!cfaSize.hasPositiveArea() || cfaSize.area() != cPat->count) {
ThrowRDE("CFA pattern dimension and pattern count does not "
"match: %d.",
cPat->count);
}

mRaw->cfa.setSize(cfaSize);

auto getAsCFAColor = [](uint32_t c) -> std::optional<CFAColor> {
switch (c) {
case 0:
return CFAColor::RED;
case 1:
return CFAColor::GREEN;
case 2:
return CFAColor::BLUE;
case 3:
return CFAColor::CYAN;
case 4:
return CFAColor::MAGENTA;
case 5:
return CFAColor::YELLOW;
case 6:
return CFAColor::WHITE;
default:
return std::nullopt;
}
};

for (int y = 0; y < cfaSize.y; y++) {
for (int x = 0; x < cfaSize.x; x++) {
uint32_t c1 = cPat->getByte(x + y * cfaSize.x);

auto c2 = getAsCFAColor(c1);
if (!c2)
ThrowRDE("Unsupported CFA Color: %u", c1);

mRaw->cfa.setColorAt(iPoint2D(x, y), *c2);
}
}

const std::optional<iRectangle2D> aa = parseACTIVEAREA(raw);
if (!aa)
return;

mRaw->cfa.shiftRight(-int(aa->pos.x));
mRaw->cfa.shiftDown(-int(aa->pos.y));
}

void DngDecoder::parseColorMatrix() const {

auto impl = [this](TiffTag I, TiffTag M) -> TiffEntry* {
if (!mRootIFD->hasEntryRecursive(I))
return nullptr;
if (const TiffEntry* illuminant = mRootIFD->getEntryRecursive(I);
illuminant->getU16() != 21 || 
!mRootIFD->hasEntryRecursive(M))
return nullptr;
return mRootIFD->getEntryRecursive(M);
};

const TiffEntry* mat;
mat = impl(TiffTag::CALIBRATIONILLUMINANT1, TiffTag::COLORMATRIX1);
if (!mat)
mat = impl(TiffTag::CALIBRATIONILLUMINANT2, TiffTag::COLORMATRIX2);
if (!mat)
return;

const auto srat_vals = mat->getSRationalArray(mat->count);
bool Success = true;
mRaw->metadata.colorMatrix.reserve(mat->count);
for (const auto& val : srat_vals) {
Success &= val.den != 0;
if (!Success)
break;
mRaw->metadata.colorMatrix.emplace_back(val);
}
if (!Success)
mRaw->metadata.colorMatrix.clear();
}

DngTilingDescription
DngDecoder::getTilingDescription(const TiffIFD* raw) const {
if (raw->hasEntry(TiffTag::TILEOFFSETS)) {
const uint32_t tilew = raw->getEntry(TiffTag::TILEWIDTH)->getU32();
const uint32_t tileh = raw->getEntry(TiffTag::TILELENGTH)->getU32();

if (tilew <= 0 || tileh <= 0)
ThrowRDE("Invalid tile size: (%u, %u)", tilew, tileh);

assert(tilew > 0);
const uint32_t tilesX = roundUpDivision(mRaw->dim.x, tilew);
if (!tilesX)
ThrowRDE("Zero tiles horizontally");

assert(tileh > 0);
const uint32_t tilesY = roundUpDivision(mRaw->dim.y, tileh);
if (!tilesY)
ThrowRDE("Zero tiles vertically");

const TiffEntry* offsets = raw->getEntry(TiffTag::TILEOFFSETS);
if (const TiffEntry* counts = raw->getEntry(TiffTag::TILEBYTECOUNTS);
offsets->count != counts->count) {
ThrowRDE("Tile count mismatch: offsets:%u count:%u", offsets->count,
counts->count);
}

if ((offsets->count / tilesX != tilesY || (offsets->count % tilesX != 0)) ||
(offsets->count / tilesY != tilesX || (offsets->count % tilesY != 0))) {
ThrowRDE("Tile X/Y count mismatch: total:%u X:%u, Y:%u", offsets->count,
tilesX, tilesY);
}

return {mRaw->dim, tilew, tileh};
}

const TiffEntry* offsets = raw->getEntry(TiffTag::STRIPOFFSETS);
const TiffEntry* counts = raw->getEntry(TiffTag::STRIPBYTECOUNTS);

if (counts->count != offsets->count) {
ThrowRDE("Byte count number does not match strip size: "
"count:%u, stips:%u ",
counts->count, offsets->count);
}

uint32_t yPerSlice = raw->hasEntry(TiffTag::ROWSPERSTRIP)
? raw->getEntry(TiffTag::ROWSPERSTRIP)->getU32()
: mRaw->dim.y;

if (yPerSlice == 0 ||
roundUpDivision(mRaw->dim.y, yPerSlice) != counts->count) {
ThrowRDE("Invalid y per slice %u or strip count %u (height = %u)",
yPerSlice, counts->count, mRaw->dim.y);
}

return {mRaw->dim, static_cast<uint32_t>(mRaw->dim.x), yPerSlice};
}

void DngDecoder::decodeData(const TiffIFD* raw, uint32_t sample_format) const {
if (compression == 8 && sample_format != 3) {
ThrowRDE("Only float format is supported for "
"deflate-compressed data.");
} else if ((compression == 7 || compression == 0x884c) &&
sample_format != 1) {
ThrowRDE("Only 16 bit unsigned data supported for "
"JPEG-compressed data.");
}

uint32_t predictor = ~0U;
if (raw->hasEntry(TiffTag::PREDICTOR))
predictor = raw->getEntry(TiffTag::PREDICTOR)->getU32();

if (raw->hasEntry(TiffTag::WHITELEVEL)) {
const TiffEntry* whitelevel = raw->getEntry(TiffTag::WHITELEVEL);
if (whitelevel->isInt())
mRaw->whitePoint = whitelevel->getU32();
}

AbstractDngDecompressor slices(mRaw, getTilingDescription(raw), compression,
mFixLjpeg, bps, predictor);

slices.slices.reserve(slices.dsc.numTiles);

const TiffEntry* offsets = nullptr;
const TiffEntry* counts = nullptr;
if (raw->hasEntry(TiffTag::TILEOFFSETS)) {
offsets = raw->getEntry(TiffTag::TILEOFFSETS);
counts = raw->getEntry(TiffTag::TILEBYTECOUNTS);
} else { 
offsets = raw->getEntry(TiffTag::STRIPOFFSETS);
counts = raw->getEntry(TiffTag::STRIPBYTECOUNTS);
}
assert(slices.dsc.numTiles == offsets->count);
assert(slices.dsc.numTiles == counts->count);

NORangesSet<Buffer> tilesLegality;
for (auto n = 0U; n < slices.dsc.numTiles; n++) {
const auto offset = offsets->getU32(n);
const auto count = counts->getU32(n);

if (count < 1)
ThrowRDE("Tile %u is empty", n);

ByteStream bs(DataBuffer(mFile.getSubView(offset, count),
mRootIFD->rootBuffer.getByteOrder()));

if (!tilesLegality.insert(bs))
ThrowTPE("Two tiles overlap. Raw corrupt!");

slices.slices.emplace_back(slices.dsc, n, bs);
}

assert(slices.slices.size() == slices.dsc.numTiles);
if (slices.slices.empty())
ThrowRDE("No valid slices found.");


mRaw->createData();

slices.decompress();
}

RawImage DngDecoder::decodeRawInternal() {
vector<const TiffIFD*> data = mRootIFD->getIFDsWithTag(TiffTag::COMPRESSION);

if (data.empty())
ThrowRDE("No image data found");

dropUnsuportedChunks(&data);

if (data.empty())
ThrowRDE("No RAW chunks found");

if (data.size() > 1) {
writeLog(DEBUG_PRIO::EXTRA,
"Multiple RAW chunks found - using first only!");
}

const TiffIFD* raw = data[0];

bps = raw->getEntry(TiffTag::BITSPERSAMPLE)->getU32();
if (bps < 1 || bps > 32)
ThrowRDE("Unsupported bit per sample count: %u.", bps);

uint32_t sample_format = 1;
if (raw->hasEntry(TiffTag::SAMPLEFORMAT))
sample_format = raw->getEntry(TiffTag::SAMPLEFORMAT)->getU32();

compression = raw->getEntry(TiffTag::COMPRESSION)->getU16();

switch (sample_format) {
case 1:
mRaw = RawImage::create(RawImageType::UINT16);
break;
case 3:
mRaw = RawImage::create(RawImageType::F32);
break;
default:
ThrowRDE("Only 16 bit unsigned or float point data supported. Sample "
"format %u is not supported.",
sample_format);
}

mRaw->isCFA =
(raw->getEntry(TiffTag::PHOTOMETRICINTERPRETATION)->getU16() == 32803);

if (mRaw->isCFA)
writeLog(DEBUG_PRIO::EXTRA, "This is a CFA image");
else {
writeLog(DEBUG_PRIO::EXTRA, "This is NOT a CFA image");
}

if (sample_format == 1 && bps > 16)
ThrowRDE("Integer precision larger than 16 bits currently not supported.");

if (sample_format == 3 && bps != 16 && bps != 24 && bps != 32)
ThrowRDE("Floating point must be 16/24/32 bits per sample.");

mRaw->dim.x = raw->getEntry(TiffTag::IMAGEWIDTH)->getU32();
mRaw->dim.y = raw->getEntry(TiffTag::IMAGELENGTH)->getU32();

if (!mRaw->dim.hasPositiveArea())
ThrowRDE("Image has zero size");

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
if (mRaw->dim.x > 9280 || mRaw->dim.y > 7680) {
ThrowRDE("Unexpected image dimensions found: (%u; %u)", mRaw->dim.x,
mRaw->dim.y);
}
#endif

if (mRaw->isCFA)
parseCFA(raw);

uint32_t cpp = raw->getEntry(TiffTag::SAMPLESPERPIXEL)->getU32();

if (cpp < 1 || cpp > 4)
ThrowRDE("Unsupported samples per pixel count: %u.", cpp);

mRaw->setCpp(cpp);

decodeData(raw, sample_format);

handleMetadata(raw);

return mRaw;
}

void DngDecoder::handleMetadata(const TiffIFD* raw) {
if (const std::optional<iRectangle2D> aa = parseACTIVEAREA(raw))
mRaw->subFrame(*aa);

if (raw->hasEntry(TiffTag::DEFAULTCROPORIGIN) &&
raw->hasEntry(TiffTag::DEFAULTCROPSIZE)) {
iRectangle2D cropped(0, 0, mRaw->dim.x, mRaw->dim.y);
const TiffEntry* origin_entry = raw->getEntry(TiffTag::DEFAULTCROPORIGIN);
const TiffEntry* size_entry = raw->getEntry(TiffTag::DEFAULTCROPSIZE);

const auto tl_r = origin_entry->getRationalArray(2);
std::array<unsigned, 2> tl;
std::transform(tl_r.begin(), tl_r.end(), tl.begin(),
[](const NotARational<unsigned>& r) {
if (r.den == 0 || r.num % r.den != 0)
ThrowRDE("Error decoding default crop origin");
return r.num / r.den;
});

if (iPoint2D cropOrigin(tl[0], tl[1]);
cropped.isPointInsideInclusive(cropOrigin))
cropped = iRectangle2D(cropOrigin, {0, 0});

cropped.dim = mRaw->dim - cropped.pos;

const auto sz_r = size_entry->getRationalArray(2);
std::array<unsigned, 2> sz;
std::transform(sz_r.begin(), sz_r.end(), sz.begin(),
[](const NotARational<unsigned>& r) {
if (r.den == 0 || r.num % r.den != 0)
ThrowRDE("Error decoding default crop size");
return r.num / r.den;
});

if (iPoint2D size(sz[0], sz[1]);
size.isThisInside(mRaw->dim) &&
(size + cropped.pos).isThisInside(mRaw->dim))
cropped.dim = size;

if (!cropped.hasPositiveArea())
ThrowRDE("No positive crop area");

mRaw->subFrame(cropped);
}
if (mRaw->dim.area() <= 0)
ThrowRDE("No image left after crop");

if (raw->hasEntry(TiffTag::DEFAULTSCALE)) {
const TiffEntry* default_scale = raw->getEntry(TiffTag::DEFAULTSCALE);
const auto scales = default_scale->getRationalArray(2);
for (const auto& scale : scales) {
if (scale.num == 0 || scale.den == 0)
ThrowRDE("Error decoding default pixel scale");
}
mRaw->metadata.pixelAspectRatio =
static_cast<double>(scales[0]) / static_cast<double>(scales[1]);
}

if (applyStage1DngOpcodes && raw->hasEntry(TiffTag::OPCODELIST1)) {
try {
const TiffEntry* opcodes = raw->getEntry(TiffTag::OPCODELIST1);
if (opcodes->count > 0) {
DngOpcodes codes(mRaw, opcodes->getData());
codes.applyOpCodes(mRaw);
}
} catch (const RawDecoderException& e) {
mRaw->setError(e.what());
}
}

if (raw->hasEntry(TiffTag::LINEARIZATIONTABLE) &&
raw->getEntry(TiffTag::LINEARIZATIONTABLE)->count > 0) {
const TiffEntry* lintable = raw->getEntry(TiffTag::LINEARIZATIONTABLE);
auto table = lintable->getU16Array(lintable->count);
RawImageCurveGuard curveHandler(&mRaw, table, uncorrectedRawValues);
if (!uncorrectedRawValues)
mRaw->sixteenBitLookup();
}

if (mRaw->getDataType() == RawImageType::UINT16) {
mRaw->whitePoint = (1UL << bps) - 1UL;
} else if (mRaw->getDataType() == RawImageType::F32) {
mRaw->whitePoint = 65535;
}

if (raw->hasEntry(TiffTag::WHITELEVEL)) {
const TiffEntry* whitelevel = raw->getEntry(TiffTag::WHITELEVEL);
if (whitelevel->isInt())
mRaw->whitePoint = whitelevel->getU32();
}
setBlack(raw);

if (compression == 0x884c && !uncorrectedRawValues &&
raw->hasEntry(TiffTag::OPCODELIST2)) {
mRaw->scaleBlackWhite();

try {
DngOpcodes codes(mRaw, raw->getEntry(TiffTag::OPCODELIST2)->getData());
codes.applyOpCodes(mRaw);
} catch (const RawDecoderException& e) {
mRaw->setError(e.what());
}
mRaw->blackAreas.clear();
mRaw->blackLevel = 0;
mRaw->blackLevelSeparate[0] = mRaw->blackLevelSeparate[1] =
mRaw->blackLevelSeparate[2] = mRaw->blackLevelSeparate[3] = 0;
mRaw->whitePoint = 65535;
}
}

void DngDecoder::decodeMetaDataInternal(const CameraMetaData* meta) {
if (mRootIFD->hasEntryRecursive(TiffTag::ISOSPEEDRATINGS))
mRaw->metadata.isoSpeed =
mRootIFD->getEntryRecursive(TiffTag::ISOSPEEDRATINGS)->getU32();

TiffID id;

try {
id = mRootIFD->getID();
} catch (const RawspeedException& e) {
mRaw->setError(e.what());
}

mRaw->metadata.make = id.make;
mRaw->metadata.model = id.model;

const Camera* cam = meta->getCamera(id.make, id.model, "dng");
if (!cam) 
cam = meta->getCamera(id.make, id.model, "");
if (!cam) 
cam = meta->getCamera(id.make, id.model);
if (cam) {
mRaw->metadata.canonical_make = cam->canonical_make;
mRaw->metadata.canonical_model = cam->canonical_model;
mRaw->metadata.canonical_alias = cam->canonical_alias;
mRaw->metadata.canonical_id = cam->canonical_id;
} else {
mRaw->metadata.canonical_make = id.make;
mRaw->metadata.canonical_model = mRaw->metadata.canonical_alias = id.model;
if (mRootIFD->hasEntryRecursive(TiffTag::UNIQUECAMERAMODEL)) {
mRaw->metadata.canonical_id =
mRootIFD->getEntryRecursive(TiffTag::UNIQUECAMERAMODEL)->getString();
} else {
mRaw->metadata.canonical_id = id.make + " " + id.model;
}
}

if (mRootIFD->hasEntryRecursive(TiffTag::ASSHOTNEUTRAL)) {
const TiffEntry* as_shot_neutral =
mRootIFD->getEntryRecursive(TiffTag::ASSHOTNEUTRAL);
if (as_shot_neutral->count == 3) {
for (uint32_t i = 0; i < 3; i++) {
float c = as_shot_neutral->getFloat(i);
mRaw->metadata.wbCoeffs[i] = (c > 0.0F) ? (1.0F / c) : 0.0F;
}
}
} else if (mRootIFD->hasEntryRecursive(TiffTag::ASSHOTWHITEXY)) {
const TiffEntry* as_shot_white_xy =
mRootIFD->getEntryRecursive(TiffTag::ASSHOTWHITEXY);
if (as_shot_white_xy->count == 2) {
mRaw->metadata.wbCoeffs[0] = as_shot_white_xy->getFloat(0);
mRaw->metadata.wbCoeffs[1] = as_shot_white_xy->getFloat(1);
mRaw->metadata.wbCoeffs[2] =
1 - mRaw->metadata.wbCoeffs[0] - mRaw->metadata.wbCoeffs[1];

const std::array<float, 3> d65_white = {{0.950456, 1, 1.088754}};
for (uint32_t i = 0; i < 3; i++)
mRaw->metadata.wbCoeffs[i] /= d65_white[i];
}
}

parseColorMatrix();
}


void DngDecoder::checkSupportInternal(const CameraMetaData* meta) {
failOnUnknown = false;

if (!(mRootIFD->hasEntryRecursive(TiffTag::MAKE) &&
mRootIFD->hasEntryRecursive(TiffTag::MODEL))) {
if (mRootIFD->hasEntryRecursive(TiffTag::UNIQUECAMERAMODEL)) {
std::string unique =
mRootIFD->getEntryRecursive(TiffTag::UNIQUECAMERAMODEL)->getString();
checkCameraSupported(meta, {unique, unique}, "dng");
return;
}
return;
}

checkCameraSupported(meta, mRootIFD->getID(), "dng");
}


bool DngDecoder::decodeMaskedAreas(const TiffIFD* raw) const {
const TiffEntry* masked = raw->getEntry(TiffTag::MASKEDAREAS);

if (masked->type != TiffDataType::SHORT && masked->type != TiffDataType::LONG)
return false;

uint32_t nrects = masked->count / 4;
if (0 == nrects)
return false;


auto rects = masked->getU32Array(nrects * 4);

const iRectangle2D fullImage(0, 0, mRaw->getUncroppedDim().x,
mRaw->getUncroppedDim().y);
const iPoint2D top = mRaw->getCropOffset();

for (uint32_t i = 0; i < nrects; i++) {
iPoint2D topleft(rects[i * 4UL + 1UL], rects[i * 4UL]);
iPoint2D bottomright(rects[i * 4UL + 3UL], rects[i * 4UL + 2UL]);

if (!(fullImage.isPointInsideInclusive(topleft) &&
fullImage.isPointInsideInclusive(bottomright) &&
(topleft < bottomright)))
ThrowRDE("Bad masked area.");

if (topleft.x <= top.x && bottomright.x >= (mRaw->dim.x + top.x)) {
mRaw->blackAreas.emplace_back(topleft.y, bottomright.y - topleft.y,
false);
}
else if (topleft.y <= top.y && bottomright.y >= (mRaw->dim.y + top.y)) {
mRaw->blackAreas.emplace_back(topleft.x, bottomright.x - topleft.x, true);
}
}
return !mRaw->blackAreas.empty();
}

bool DngDecoder::decodeBlackLevels(const TiffIFD* raw) const {
iPoint2D blackdim(1, 1);
if (raw->hasEntry(TiffTag::BLACKLEVELREPEATDIM)) {
const TiffEntry* bleveldim = raw->getEntry(TiffTag::BLACKLEVELREPEATDIM);
if (bleveldim->count != 2)
return false;
blackdim = iPoint2D(bleveldim->getU32(0), bleveldim->getU32(1));
}

if (!blackdim.hasPositiveArea())
return false;

if (!raw->hasEntry(TiffTag::BLACKLEVEL))
return true;

if (mRaw->getCpp() != 1)
return false;

const TiffEntry* black_entry = raw->getEntry(TiffTag::BLACKLEVEL);
if (black_entry->count < blackdim.area())
ThrowRDE("BLACKLEVEL entry is too small");

using BlackType = decltype(mRaw->blackLevelSeparate)::value_type;

if (blackdim.x < 2 || blackdim.y < 2) {
float value = black_entry->getFloat();

if (value < std::numeric_limits<BlackType>::min() ||
value > std::numeric_limits<BlackType>::max())
ThrowRDE("Error decoding black level");

for (int y = 0; y < 2; y++) {
for (int x = 0; x < 2; x++)
mRaw->blackLevelSeparate[y * 2 + x] = value;
}
} else {
for (int y = 0; y < 2; y++) {
for (int x = 0; x < 2; x++) {
float value = black_entry->getFloat(y * blackdim.x + x);

if (value < std::numeric_limits<BlackType>::min() ||
value > std::numeric_limits<BlackType>::max())
ThrowRDE("Error decoding black level");

mRaw->blackLevelSeparate[y * 2 + x] = value;
}
}
}

if (raw->hasEntry(TiffTag::BLACKLEVELDELTAV)) {
const TiffEntry* blackleveldeltav =
raw->getEntry(TiffTag::BLACKLEVELDELTAV);
if (static_cast<int>(blackleveldeltav->count) < mRaw->dim.y)
ThrowRDE("BLACKLEVELDELTAV array is too small");
std::array<float, 2> black_sum = {{}};
for (int i = 0; i < mRaw->dim.y; i++)
black_sum[i & 1] += blackleveldeltav->getFloat(i);

for (int i = 0; i < 4; i++) {
const float value =
black_sum[i >> 1] / static_cast<float>(mRaw->dim.y) * 2.0F;
if (value < std::numeric_limits<BlackType>::min() ||
value > std::numeric_limits<BlackType>::max())
ThrowRDE("Error decoding black level");

if (__builtin_sadd_overflow(mRaw->blackLevelSeparate[i], value,
&mRaw->blackLevelSeparate[i]))
ThrowRDE("Integer overflow when calculating black level");
}
}

if (raw->hasEntry(TiffTag::BLACKLEVELDELTAH)) {
const TiffEntry* blackleveldeltah =
raw->getEntry(TiffTag::BLACKLEVELDELTAH);
if (static_cast<int>(blackleveldeltah->count) < mRaw->dim.x)
ThrowRDE("BLACKLEVELDELTAH array is too small");
std::array<float, 2> black_sum = {{}};
for (int i = 0; i < mRaw->dim.x; i++)
black_sum[i & 1] += blackleveldeltah->getFloat(i);

for (int i = 0; i < 4; i++) {
const float value =
black_sum[i & 1] / static_cast<float>(mRaw->dim.x) * 2.0F;
if (value < std::numeric_limits<BlackType>::min() ||
value > std::numeric_limits<BlackType>::max())
ThrowRDE("Error decoding black level");

if (__builtin_sadd_overflow(mRaw->blackLevelSeparate[i], value,
&mRaw->blackLevelSeparate[i]))
ThrowRDE("Integer overflow when calculating black level");
}
}
return true;
}

void DngDecoder::setBlack(const TiffIFD* raw) const {

if (raw->hasEntry(TiffTag::MASKEDAREAS) && decodeMaskedAreas(raw))
return;

mRaw->blackLevelSeparate.fill(0);

if (raw->hasEntry(TiffTag::BLACKLEVEL))
decodeBlackLevels(raw);
}
} 
