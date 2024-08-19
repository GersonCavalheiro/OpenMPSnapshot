

#include "RawSpeed-API.h"   
#include "adt/Array2DRef.h" 
#include <array>            
#include <cstddef>          
#include <cstdint>          
#include <cstdio>           
#include <memory>           
#include <string>           
#include <sys/stat.h>       
#include <vector>           

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 
#endif

#include <Windows.h>
#endif

namespace rawspeed::identify {

std::string find_cameras_xml(const char* argv0);

std::string find_cameras_xml(const char* argv0) {
struct stat statbuf;

#ifdef RS_CAMERAS_XML_PATH
if (static const char* set_camfile = RS_CAMERAS_XML_PATH;
stat(set_camfile, &statbuf)) {
fprintf(stderr, "WARNING: Couldn't find cameras.xml in '%s'\n",
set_camfile);
} else {
return set_camfile;
}
#endif

const std::string self(argv0);

const std::size_t lastslash = self.find_last_of(R"(/\)");
const std::string bindir(self.substr(0, lastslash));

std::string found_camfile(bindir +
"/../share/darktable/rawspeed/cameras.xml");

if (stat(found_camfile.c_str(), &statbuf)) {
#ifndef __APPLE__
fprintf(stderr, "WARNING: Couldn't find cameras.xml in '%s'\n",
found_camfile.c_str());
#else
fprintf(stderr, "WARNING: Couldn't find cameras.xml in '%s'\n",
found_camfile.c_str());
found_camfile =
bindir + "/../Resources/share/darktable/rawspeed/cameras.xml";
if (stat(found_camfile.c_str(), &statbuf)) {
fprintf(stderr, "WARNING: Couldn't find cameras.xml in '%s'\n",
found_camfile.c_str());
}
#endif
}

#ifdef RAWSPEED_STANDALONE_BUILD
found_camfile = std::string(RAWSPEED_SOURCE_DIR "/data/cameras.xml");
#endif

if (stat(found_camfile.c_str(), &statbuf)) {
#ifndef __APPLE__
fprintf(stderr, "ERROR: Couldn't find cameras.xml in '%s'\n",
found_camfile.c_str());
return {};
#else
fprintf(stderr, "WARNING: Couldn't find cameras.xml in '%s'\n",
found_camfile.c_str());
found_camfile =
bindir + "/../Resources/share/darktable/rawspeed/cameras.xml";
if (stat(found_camfile.c_str(), &statbuf)) {
fprintf(stderr, "ERROR: Couldn't find cameras.xml in '%s'\n",
found_camfile.c_str());
return {};
}
#endif
}

return found_camfile;
}

} 

using rawspeed::Buffer;
using rawspeed::CameraMetaData;
using rawspeed::FileReader;
using rawspeed::iPoint2D;
using rawspeed::RawImage;
using rawspeed::RawParser;
using rawspeed::RawspeedException;
using rawspeed::identify::find_cameras_xml;

int main(int argc, char* argv[]) { 

if (argc != 2) {
fprintf(stderr, "Usage: darktable-rs-identify <file>\n");
return 0;
}

const std::string camfile = find_cameras_xml(argv[0]);
if (camfile.empty()) {
return 2;
}

try {
std::unique_ptr<const CameraMetaData> meta;

#ifdef HAVE_PUGIXML
meta = std::make_unique<CameraMetaData>(camfile.c_str());
#else
meta = std::make_unique<CameraMetaData>();
#endif

if (!meta) {
fprintf(stderr, "ERROR: Couldn't get a CameraMetaData instance\n");
return 2;
}

#ifndef _WIN32
char* imageFileName = argv[1];
#else
int size = MultiByteToWideChar(CP_ACP, 0, argv[1], -1, nullptr, 0);
std::wstring wImageFileName;
wImageFileName.resize(size);
MultiByteToWideChar(CP_ACP, 0, argv[1], -1, &wImageFileName[0], size);
size = WideCharToMultiByte(CP_UTF8, 0, &wImageFileName[0], -1, nullptr, 0,
nullptr, nullptr);
std::string _imageFileName;
_imageFileName.resize(size);
char* imageFileName = &_imageFileName[0];
WideCharToMultiByte(CP_UTF8, 0, &wImageFileName[0], -1, imageFileName, size,
nullptr, nullptr);
#endif

fprintf(stderr, "Loading file: \"%s\"\n", imageFileName);

FileReader f(imageFileName);

auto [storage, buf] = f.readFile();

RawParser t(buf);

auto d(t.getDecoder(meta.get()));

if (!d) {
fprintf(stderr, "ERROR: Couldn't get a RawDecoder instance\n");
return 2;
}

d->applyCrop = false;
d->failOnUnknown = true;
RawImage r = d->mRaw;
const RawImage* const raw = &r;

d->decodeMetaData(meta.get());

fprintf(stdout, "make: %s\n", r->metadata.make.c_str());
fprintf(stdout, "model: %s\n", r->metadata.model.c_str());

fprintf(stdout, "canonical_make: %s\n", r->metadata.canonical_make.c_str());
fprintf(stdout, "canonical_model: %s\n",
r->metadata.canonical_model.c_str());
fprintf(stdout, "canonical_alias: %s\n",
r->metadata.canonical_alias.c_str());

d->checkSupport(meta.get());
d->decodeRaw();
d->decodeMetaData(meta.get());
r = d->mRaw;

const auto errors = r->getErrors();
for (const auto& error : errors)
fprintf(stderr, "WARNING: [rawspeed] %s\n", error.c_str());

fprintf(stdout, "blackLevel: %d\n", r->blackLevel);
fprintf(stdout, "whitePoint: %d\n", r->whitePoint);

fprintf(stdout, "blackLevelSeparate: %d %d %d %d\n",
r->blackLevelSeparate[0], r->blackLevelSeparate[1],
r->blackLevelSeparate[2], r->blackLevelSeparate[3]);

fprintf(stdout, "wbCoeffs: %f %f %f %f\n", r->metadata.wbCoeffs[0],
r->metadata.wbCoeffs[1], r->metadata.wbCoeffs[2],
r->metadata.wbCoeffs[3]);

fprintf(stdout, "isCFA: %d\n", r->isCFA);
uint32_t filters = r->cfa.getDcrawFilter();
fprintf(stdout, "filters: %u (0x%x)\n", filters, filters);
const uint32_t bpp = r->getBpp();
fprintf(stdout, "bpp: %u\n", bpp);
const uint32_t cpp = r->getCpp();
fprintf(stdout, "cpp: %u\n", cpp);
fprintf(stdout, "dataType: %u\n", static_cast<unsigned>(r->getDataType()));

const iPoint2D dimUncropped = r->getUncroppedDim();
fprintf(stdout, "dimUncropped: %dx%d\n", dimUncropped.x, dimUncropped.y);

iPoint2D dimCropped = r->dim;
fprintf(stdout, "dimCropped: %dx%d\n", dimCropped.x, dimCropped.y);

iPoint2D cropTL = r->getCropOffset();
fprintf(stdout, "cropOffset: %dx%d\n", cropTL.x, cropTL.y);

fprintf(stdout, "fuji_rotation_pos: %d\n", r->metadata.fujiRotationPos);
fprintf(stdout, "pixel_aspect_ratio: %f\n", r->metadata.pixelAspectRatio);

double sum = 0.0F;
#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) firstprivate(dimUncropped, raw, bpp) schedule(static) reduction(+ : sum)
#endif
for (int y = 0; y < dimUncropped.y; ++y) {
const rawspeed::Array2DRef<std::byte> img =
(*raw)->getByteDataAsUncroppedArray2DRef();
for (unsigned x = 0; x < bpp * dimUncropped.x; ++x)
sum += static_cast<double>(img(y, x));
}
fprintf(stdout, "Image byte sum: %lf\n", sum);
fprintf(stdout, "Image byte avg: %lf\n",
sum / static_cast<double>(dimUncropped.y * dimUncropped.x * bpp));

if (r->getDataType() == rawspeed::RawImageType::F32) {
sum = 0.0F;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) firstprivate(dimUncropped, raw, cpp) schedule(static) reduction(+ : sum)
#endif
for (int y = 0; y < dimUncropped.y; ++y) {
const rawspeed::Array2DRef<float> img =
(*raw)->getF32DataAsUncroppedArray2DRef();
for (unsigned x = 0; x < cpp * dimUncropped.x; ++x)
sum += static_cast<double>(img(y, x));
}

fprintf(stdout, "Image float sum: %lf\n", sum);
fprintf(stdout, "Image float avg: %lf\n",
sum / static_cast<double>(dimUncropped.y * dimUncropped.x));
} else if (r->getDataType() == rawspeed::RawImageType::UINT16) {
sum = 0.0F;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) firstprivate(dimUncropped, raw, cpp) schedule(static) reduction(+ : sum)
#endif
for (int y = 0; y < dimUncropped.y; ++y) {
const rawspeed::Array2DRef<uint16_t> img =
(*raw)->getU16DataAsUncroppedArray2DRef();
for (unsigned x = 0; x < cpp * dimUncropped.x; ++x)
sum += static_cast<double>(img(y, x));
}

fprintf(stdout, "Image uint16_t sum: %lf\n", sum);
fprintf(stdout, "Image uint16_t avg: %lf\n",
sum / static_cast<double>(dimUncropped.y * dimUncropped.x));
}
} catch (const RawspeedException& e) {
fprintf(stderr, "ERROR: [rawspeed] %s\n", e.what());


return 2;
}

return 0;
}

