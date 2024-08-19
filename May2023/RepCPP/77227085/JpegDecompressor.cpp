

#include "rawspeedconfig.h"       
#include "adt/AlignedAllocator.h" 

#ifdef HAVE_JPEG

#include "adt/Array2DRef.h"               
#include "adt/Point.h"                    
#include "decoders/RawDecoderException.h" 
#include "decompressors/JpegDecompressor.h"
#include <algorithm> 
#include <array>     
#include <jpeglib.h> 
#include <memory>    
#include <vector>    

#ifndef HAVE_JPEG_MEM_SRC
#include "io/IOException.h" 
#endif

using std::min;
using std::unique_ptr;

namespace rawspeed {

#ifdef HAVE_JPEG_MEM_SRC

#define JPEG_MEMSRC(A, B, C)                                                   \
jpeg_mem_src(A, const_cast<unsigned char*>(B), C) 

#else

#define JPEG_MEMSRC(A, B, C) jpeg_mem_src_int(A, B, C)


static void init_source(j_decompress_ptr cinfo) {}
static boolean fill_input_buffer(j_decompress_ptr cinfo) {
auto* src = (struct jpeg_source_mgr*)cinfo->src;
return (boolean) !!src->bytes_in_buffer;
}
static void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
auto* src = (struct jpeg_source_mgr*)cinfo->src;

if (num_bytes > (int)src->bytes_in_buffer)
ThrowIOE("read out of buffer");
if (num_bytes > 0) {
src->next_input_byte += (size_t)num_bytes;
src->bytes_in_buffer -= (size_t)num_bytes;
}
}
static void term_source(j_decompress_ptr cinfo) {}
static void jpeg_mem_src_int(j_decompress_ptr cinfo,
const unsigned char* buffer, long nbytes) {
struct jpeg_source_mgr* src;

if (cinfo->src == nullptr) { 
cinfo->src = (struct jpeg_source_mgr*)(*cinfo->mem->alloc_small)(
(j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(struct jpeg_source_mgr));
}

src = (struct jpeg_source_mgr*)cinfo->src;
src->init_source = init_source;
src->fill_input_buffer = fill_input_buffer;
src->skip_input_data = skip_input_data;
src->resync_to_restart = jpeg_resync_to_restart; 
src->term_source = term_source;
src->bytes_in_buffer = nbytes;
src->next_input_byte = (const JOCTET*)buffer;
}

#endif

[[noreturn]] METHODDEF(void) my_error_throw(j_common_ptr cinfo) {
std::array<char, JMSG_LENGTH_MAX> buf;
buf.fill(0);
(*cinfo->err->format_message)(cinfo, buf.data());
ThrowRDE("JPEG decoder error: %s", buf.data());
}

struct JpegDecompressor::JpegDecompressStruct : jpeg_decompress_struct {
struct jpeg_error_mgr jerr;

JpegDecompressStruct(const JpegDecompressStruct&) = delete;
JpegDecompressStruct(JpegDecompressStruct&&) noexcept = delete;
JpegDecompressStruct&
operator=(const JpegDecompressStruct&) noexcept = delete;
JpegDecompressStruct& operator=(JpegDecompressStruct&&) noexcept = delete;

JpegDecompressStruct() {
jpeg_create_decompress(this);

err = jpeg_std_error(&jerr);
jerr.error_exit = &my_error_throw;
}
~JpegDecompressStruct() { jpeg_destroy_decompress(this); }
};

void JpegDecompressor::decode(uint32_t offX,
uint32_t offY) { 
struct JpegDecompressStruct dinfo;

JPEG_MEMSRC(&dinfo, input.begin(), input.getSize());

if (JPEG_HEADER_OK != jpeg_read_header(&dinfo, static_cast<boolean>(true)))
ThrowRDE("Unable to read JPEG header");

jpeg_start_decompress(&dinfo);
if (dinfo.output_components != static_cast<int>(mRaw->getCpp()))
ThrowRDE("Component count doesn't match");
int row_stride = dinfo.output_width * dinfo.output_components;

std::vector<uint8_t, AlignedAllocator<uint8_t, 16>> complete_buffer;
complete_buffer.resize(dinfo.output_height * row_stride);

const Array2DRef<uint8_t> tmp(&complete_buffer[0],
dinfo.output_components * dinfo.output_width,
dinfo.output_height, row_stride);

while (dinfo.output_scanline < dinfo.output_height) {
JSAMPROW rowOut = &tmp(dinfo.output_scanline, 0);
if (0 == jpeg_read_scanlines(&dinfo, &rowOut, 1))
ThrowRDE("JPEG Error while decompressing image.");
}
jpeg_finish_decompress(&dinfo);

int copy_w = min(mRaw->dim.x - offX, dinfo.output_width);
int copy_h = min(mRaw->dim.y - offY, dinfo.output_height);

const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());
for (int row = 0; row < copy_h; row++) {
for (int col = 0; col < dinfo.output_components * copy_w; col++)
out(row + offY, dinfo.output_components * offX + col) = tmp(row, col);
}
}

} 

#else

#pragma message                                                                \
"JPEG is not present! Lossy JPEG compression will not be supported!"

#endif
