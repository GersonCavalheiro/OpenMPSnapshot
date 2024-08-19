#include "stdio.h"
#include "image.h"

#define DEFAULT_ALPHA 255

std::shared_ptr<ImageRgb> convertBytesToImage(std::vector<unsigned char> bytes, unsigned int width, unsigned int height, int start, int end) {
if (end == -1) {
end = bytes.size();
}
std::shared_ptr<ImageRgb> image(new ImageRgb());
image->numPixels = (end - start + 1) / 4;
std::vector<std::shared_ptr<PixelRgba>> pixels(image->numPixels);
image->width = width;
image->height = height;
#pragma omp parallel for
for (int i = start; i < end; i += 4) {
std::shared_ptr<PixelRgba> pixel(new PixelRgba());
pixel->r = bytes[i];
pixel->g = bytes[i + 1];
pixel->b = bytes[i + 2];
pixel->a = bytes[i + 3];
pixels[(i - start) / 4] = pixel;
}
image->pixels = pixels;
return image;
}

std::vector<unsigned char> convertImageToBytes(std::shared_ptr<ImageRgb> image) {
std::vector<unsigned char> bytes;
for (auto pixel : image->pixels) {
bytes.push_back(pixel->r);
bytes.push_back(pixel->g);
bytes.push_back(pixel->b);
bytes.push_back(DEFAULT_ALPHA);
}
return bytes;
}

std::shared_ptr<ImageYcbcr> convertRgbToYcbcr(std::shared_ptr<ImageRgb> input) {
std::shared_ptr<ImageYcbcr> result(new ImageYcbcr());
result->numPixels = input->numPixels;
result->width = input->width;
result->height = input->height;
std::vector<std::shared_ptr<PixelYcbcr>> new_pixels(input->pixels.size());
#pragma omp parallel for
for (unsigned int i = 0; i < input->pixels.size(); i++) {
auto pixel = input->pixels[i];
std::shared_ptr<PixelYcbcr> new_pixel(new PixelYcbcr());
new_pixel->y = 16 + (65.738 * pixel->r + 129.057 * pixel->g + 25.064 * pixel->b) / 256;
new_pixel->cb = 128 - (37.945 * pixel->r + 74.494 * pixel->g - 112.439 * pixel->b) / 256;
new_pixel->cr = 128 + (112.439 * pixel->r - 94.154 * pixel->g - 18.285 * pixel->b) / 256;
new_pixels[i] = new_pixel;
}
result->pixels = new_pixels;
return result;
}

std::shared_ptr<ImageRgb> convertYcbcrToRgb(std::shared_ptr<ImageYcbcr> input) {
std::shared_ptr<ImageRgb> result(new ImageRgb());
result->numPixels = input->numPixels;
result->width = input->width;
result->height = input->height;
std::vector<std::shared_ptr<PixelRgba>> new_pixels;
for (auto pixel : input->pixels) {
std::shared_ptr<PixelRgba> new_pixel(new PixelRgba());
new_pixel->r = (298.082 * pixel->y + 408.583 * pixel->cr) / 256 - 222.921;
new_pixel->g = (298.082 * pixel->y - 100.291 * pixel->cb - 208.120 * pixel->cr) / 256 + 135.576;
new_pixel->b = (298.082 * pixel->y + 516.412 * pixel->cb) / 256 - 276.836;
new_pixels.push_back(new_pixel);
}
result->pixels = new_pixels;
return result;
}

std::shared_ptr<ImageBlocks> convertYcbcrToBlocks(std::shared_ptr<ImageYcbcr> input, int block_size) {
std::shared_ptr<ImageBlocks> result(new ImageBlocks());
std::vector<std::vector<std::shared_ptr<PixelYcbcr>>> blocks;
result->width = input->width;
result->height = input->height;
int blocks_width = (input->width + block_size - 1) / block_size;
int blocks_height = (input->height + block_size - 1) / block_size;
result->numBlocks = blocks_width * blocks_height;
int offset = 0;
for (int i = 0; i < blocks_height; i++) {
int start_index = i * block_size * block_size * blocks_width;
std::vector<std::vector<std::shared_ptr<PixelYcbcr>>> block_row(blocks_width, std::vector<std::shared_ptr<PixelYcbcr>>(block_size * block_size));
for (int j = 0; j < block_size; j++) {
int row_index = start_index + j * block_size * blocks_width;
#pragma omp parallel for
for (int k = 0; k < block_size * blocks_width; k++) {
int pixel_index = row_index + k - offset;
std::shared_ptr<PixelYcbcr> pixel(new PixelYcbcr());
if (pixel_in_bounds(i * block_size + j, k, input->width, input->height)) {
pixel->y = input->pixels[pixel_index]->y;
pixel->cb = input->pixels[pixel_index]->cb;
pixel->cr = input->pixels[pixel_index]->cr;
} else {
pixel->y = 0;
pixel->cb = 0;
pixel->cr = 0;
offset++;
}
block_row[k / block_size][sub2ind(block_size, k % block_size, j)] = pixel;
}
}
blocks.insert(blocks.end(), block_row.begin(), block_row.end());
}
result->blocks = blocks;
if (result->width % block_size != 0) {
result->width += block_size - (result->width % block_size);
}
if (result->height % block_size != 0) {
result->height += block_size - (result->height % block_size);
}
downsampleCbcr(result, block_size);
return result;
}

std::shared_ptr<ImageYcbcr> convertBlocksToYcbcr(std::shared_ptr<ImageBlocks> input, int block_size) {
upsampleCbcr(input, block_size);
std::shared_ptr<ImageYcbcr> result(new ImageYcbcr());
std::vector<std::shared_ptr<PixelYcbcr>> pixels;
result->numPixels = input->width * input->height;
result->width = input->width;
result->height = input->height;
int blocks_width = (input->width + block_size - 1) / block_size;
int blocks_height = (input->height + block_size - 1) / block_size;
for (int i = 0; i < blocks_height; i++) {
std::vector<std::vector<std::shared_ptr<PixelYcbcr>>> pixel_rows(block_size);
for (int j = 0; j < blocks_width; j++) {
auto block = input->blocks[sub2ind(blocks_width, j, i)];
for (unsigned int k = 0; k < block.size(); k++) {
Coord block_coord = ind2sub(block_size, k);
if (pixel_in_bounds(i * block_size + block_coord.row, j * block_size + block_coord.col, input->width, input->height)) {
std::shared_ptr<PixelYcbcr> pixel(new PixelYcbcr());
pixel->y = block[k]->y;
pixel->cb = block[k]->cb;
pixel->cr = block[k]->cr;
pixel_rows[block_coord.row].push_back(pixel);
}
}
}
for (int j = 0; j < block_size; j++) {
pixels.insert(pixels.end(), pixel_rows[j].begin(), pixel_rows[j].end());
}
}
result->pixels = pixels;
return result;
}

void downsampleCbcr(std::shared_ptr<ImageBlocks> image, int block_size) {
#pragma omp parallel for
for (unsigned int i = 0; i < image->blocks.size(); i++) {
auto block = image->blocks[i];
for (unsigned int j = 0; j < block.size(); j++) {
Coord coord = ind2sub(block_size, j);
if ((coord.row < block_size / 2) && (coord.col < block_size / 2)) {
Coord sample_coord;
sample_coord.col = coord.col * 2;
sample_coord.row = coord.row * 2;
int sample_index = sub2ind(block_size, sample_coord);
block[j]->cb = block[sample_index]->cb;
block[j]->cr = block[sample_index]->cr;
} else {
block[j]->cb = 0;
block[j]->cr = 0;
}
}
}
}

void upsampleCbcr(std::shared_ptr<ImageBlocks> image, int block_size) {
for (auto block : image->blocks) {
for (unsigned int i = block.size() - 1; i > 0; i--) {
Coord coord = ind2sub(block_size, i);
Coord sample_coord;
sample_coord.row = coord.row / 2;
sample_coord.col = coord.col / 2;
int sample_index = sub2ind(block_size, sample_coord);
block[i]->cb = block[sample_index]->cb;
block[i]->cr = block[sample_index]->cr;
}
for (int i = 1; i < block_size - 1; i += 2) {
for (int j = 1; j < block_size - 1; j += 2) {
int lower_index = sub2ind(block_size, j-1, i-1);
int index = sub2ind(block_size, j, i);
int upper_index = sub2ind(block_size, j+1, i+1);
double cb = (block[lower_index]->cb + block[upper_index]->cb) / 2;
double cr = (block[lower_index]->cr + block[upper_index]->cr) / 2;
block[index]->cb = cb;
block[index]->cr = cr;
}
}
}
}


bool pixel_in_bounds(int row, int col, int width, int height) {
return (row < height && col < width);
}

int sub2ind(int width, int col, int row) {
return row * width + col;
}

int sub2ind(int width, Coord coord) {
return coord.row * width + coord.col;
}

Coord ind2sub(int width, int idx) {
Coord result;
result.col = idx % width;
result.row = idx / width;
return result;
}
