#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include "myMalloc.h"
#include "bitmap.h"
struct Bitmap *newBitmap( uint32_t size)
{
struct Bitmap *bitmap = (struct Bitmap *) my_malloc( sizeof(struct Bitmap));
bitmap->bitarray = (uint32_t *) my_malloc(sizeof(uint32_t) * ((size + kBitsPerWord - 1) / kBitsPerWord));
bitmap->real_size = ((size + kBitsPerWord - 1) / kBitsPerWord);
memset(bitmap->bitarray, 0, (sizeof(uint32_t) * ((size + kBitsPerWord - 1) / kBitsPerWord)));
bitmap->size =  size;
bitmap->numSetBits =  0;
return bitmap;
}
struct Bitmap *newBitmapSet( uint32_t size)
{
struct Bitmap *bitmap = (struct Bitmap *) my_malloc( sizeof(struct Bitmap));
bitmap->bitarray = (uint32_t *) my_malloc(sizeof(uint32_t) * ((size + kBitsPerWord - 1) / kBitsPerWord));
bitmap->real_size = ((size + kBitsPerWord - 1) / kBitsPerWord);
memset(bitmap->bitarray, 1, (sizeof(uint32_t) * ((size + kBitsPerWord - 1) / kBitsPerWord)));
bitmap->size =  size;
bitmap->numSetBits =  size;
return bitmap;
}
void freeBitmap( struct Bitmap *bitmap)
{
if(bitmap)
{
if(bitmap->bitarray)
free(bitmap->bitarray);
free(bitmap);
}
}
void clearBitmap(struct Bitmap *bitmap)
{
memset(bitmap->bitarray, 0, (sizeof(uint32_t) * ((bitmap->size + kBitsPerWord - 1) / kBitsPerWord)));
bitmap->numSetBits =  0;
}
void setBit(struct Bitmap *bitmap, uint32_t pos)
{
bitmap->bitarray[word_offset(pos)] |= (uint32_t) (1 << bit_offset(pos));
}
void setBitXOR(struct Bitmap *bitmap, uint32_t pos)
{
bitmap->bitarray[word_offset(pos)] ^= (uint32_t) (1 << bit_offset(pos));
}
void setBitRange(struct Bitmap *bitmap, uint32_t start, uint32_t end)
{
uint32_t pos;
for (pos = start; pos < end; ++pos)
{
setBit(bitmap, pos);
}
}
void setBitAtomic(struct Bitmap *bitmap, uint32_t pos)
{
__sync_fetch_and_or(bitmap->bitarray + word_offset(pos), 1ul << bit_offset(pos));
}
uint32_t getBit(struct Bitmap *bitmap, uint32_t pos)
{
return (bitmap->bitarray[word_offset(pos)] >> bit_offset(pos)) & 1l;;
}
void clearBit(struct Bitmap *bitmap, uint32_t pos)
{
bitmap->bitarray[word_offset(pos)] &= ((uint32_t) (~(1l << bit_offset(pos))));
}
struct Bitmap  *orBitmap(struct Bitmap *bitmap1, struct Bitmap *bitmap2)
{
uint32_t i;
uint32_t *word1 = bitmap1->bitarray;
uint32_t *word2 = bitmap2->bitarray;
bitmap1->numSetBits = 0;
for(i = 0 ; i < ((bitmap1->size + kBitsPerWord - 1) / kBitsPerWord); i++)
{
word1[i] = word1[i] | word2[i];
}
bitmap1->numSetBits = getNumOfSetBits(bitmap1);
return bitmap1;
}
struct Bitmap  *andBitmap(struct Bitmap *bitmap1, struct Bitmap *bitmap2)
{
uint32_t i;
uint32_t *byte1 = bitmap1->bitarray;
uint32_t *byte2 = bitmap2->bitarray;
bitmap1->numSetBits = 0;
for(i = 0 ; i < ((bitmap1->size + kBitsPerWord - 1) / kBitsPerWord); i++)
{
byte1[i] = byte1[i] & byte2[i];
}
bitmap1->numSetBits = getNumOfSetBits(bitmap1);
return bitmap1;
}
void swapBitmaps (struct Bitmap **bitmap1, struct Bitmap **bitmap2)
{
struct Bitmap *temp_bitmap = *bitmap1;
*bitmap1 = *bitmap2;
*bitmap2 = temp_bitmap;
}
uint32_t getNumOfSetBits (struct Bitmap *bitmap)
{
uint32_t i;
uint32_t numSetBits = 0;
#pragma omp parallel for reduction(+:numSetBits) schedule(dynamic,256)
for(i = 0 ; i < (bitmap->size); i++)
{
if(getBit(bitmap, i))
numSetBits++;
}
return numSetBits;
}
void printSetBits (struct Bitmap *bitmap)
{
uint32_t i;
for(i = 0 ; i < (bitmap->size); i++)
{
if(getBit(bitmap, i))
{
printf("**%u \n", i);
}
}
}