#ifndef __IMAGE_H__
#define __IMAGE_H__
#include <inttypes.h>
#pragma pack(1)
struct bmp_header {             
uint16_t type;              
uint32_t size;              
uint16_t reserved1;         
uint16_t reserved2;         
uint32_t offset;            
uint32_t header_size;       
uint32_t width;             
uint32_t height;            
uint16_t planes;            
uint16_t bits;              
uint32_t compression;       
uint32_t imagesize;         
uint32_t xresolution;       
uint32_t yresolution;       
uint32_t num_colors;          
uint32_t important_colors;  
};
#pragma pack(1)
struct bmp_color_table {      
uint8_t red;
uint8_t green;
uint8_t blue;
uint8_t reserved;
};
struct bmp_image {
struct bmp_header header;
unsigned char table[1024];
unsigned char* data;
};
struct raw_image {
int width;
int height;
int nchannels;
unsigned char *data;
};
void bmp_describe(const struct bmp_image *img);
struct bmp_image *bmp_load(const char *filepath);
int bmp_save(const struct bmp_image *img, const char *filepath);
void bmp_destroy(struct bmp_image *img);
void raw_destroy(struct raw_image *raw);
struct bmp_image *raw_to_bmp(struct raw_image *raw);
struct raw_image *bmp_to_raw(struct bmp_image *bmp, int padding);
struct raw_image *raw_create(int width, int height, int nchannels);
struct raw_image *raw_create_empty(int width, int height, int nchannels);
#endif
