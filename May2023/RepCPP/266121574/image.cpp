
#include "image.h"

template<typename P>
ImageClass<P>::ImageClass(int const _width, int const _height)
: width(_width), height(_height) {
pixel = (P*)_mm_malloc(sizeof(P)*width*height, 64);
#pragma omp parallel for
for (int i = 0; i < height; i++)
for (int j = 0; j < width; j++)
pixel[i*width + j] = (P)0;
}


template<typename P>
ImageClass<P>::ImageClass(char const * file_name) {
FILE *fp = fopen(file_name, "rb");
if (!fp) {
printf("Could not open %s\n", file_name);
exit(1);
}

png_byte header[8];
fread(header, 1, 8, fp);
if (png_sig_cmp(header, 0, 8)){
printf("File %s is not a proper PNG file\n", file_name);
fclose(fp);
exit(1);
}

png_structp ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
png_infop info = png_create_info_struct(ptr);
setjmp(png_jmpbuf(ptr));
png_init_io(ptr, fp);
png_set_sig_bytes(ptr, 8);
png_read_info(ptr, info);

png_byte color_type = png_get_color_type(ptr, info);
png_byte bit_depth = png_get_bit_depth(ptr, info);

width = png_get_image_width(ptr, info);
int width_png = width;
if (width%(64/sizeof(P)) != 0) width += (64/sizeof(P)) - width%(64/sizeof(P));
height = png_get_image_height(ptr, info);

int number_of_passes = png_set_interlace_handling(ptr);
png_read_update_info(ptr, info);

setjmp(png_jmpbuf(ptr));
png_bytep* row = (png_bytep*) malloc(sizeof(png_bytep) * height);
for (int i = 0; i < height; i++)
row[i] = (png_byte*) malloc(png_get_rowbytes(ptr, info));
png_read_image(ptr, row);

if(png_get_rowbytes(ptr, info) != width_png) {
printf("Error: the image is not in grayscale\n");
}

fclose(fp);
pixel = (P*)_mm_malloc(sizeof(P)*width*height, 64);

#pragma omp parallel for
for(int i = 0; i < height; i++)
for(int j = 0; j < width; j++)
pixel[i*width+j] = (P) row[i][j];

for (int i = 0; i < height; i++)
free(row[i]);
free(row);
}

template<typename P>
ImageClass<P>::~ImageClass() {
_mm_free(pixel);
}



template<typename P>
void ImageClass<P>::WriteToFile(char const * file_name) {
FILE *fp = fopen(file_name, "wb");
if (!fp) {
printf("Could not open %s for writing\n", file_name);
exit(1);
}
png_structp ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);;
png_infop info = png_create_info_struct(ptr);
setjmp(png_jmpbuf(ptr));

png_init_io(ptr, fp);
png_set_IHDR(ptr, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
png_write_info(ptr, info);

png_bytep row = (png_bytep) malloc(sizeof(png_byte)*width);
for(int i = 0; i < height; i++) {
for(int j = 0; j < width; j++) {
png_byte t = (png_byte)pixel[i*width+j];
if (t < 0) t = 0;
if (t > 255) t = 255;
row[j] = (png_byte) t;
}
png_write_row(ptr, row);
}
png_write_end(ptr, NULL);
png_free_data(ptr, info, PNG_FREE_ALL, -1);
png_destroy_write_struct(&ptr, (png_infopp)NULL);
fclose(fp);
free(row);
}


template class ImageClass<float>;