#include "Game.h"
static inline uint8_t get_neighbors(unsigned int x, unsigned int y) {
uint8_t alive = 0;
{
if (x > 0) {
if (world_get(x - 1, y) == ALIVE) {
alive++;
}
}
if (x + 1 < WORLD_WIDTH) {
if (world_get(x + 1, y) == ALIVE) {
alive++;
}
}
}
if (y > 0) {
if (x > 0) {
if (world_get(x - 1, y - 1) == ALIVE) {
alive++;
}
}
if (world_get(x, y - 1) == ALIVE) {
alive++;
}
if (x + 1 < WORLD_WIDTH) {
if (world_get(x + 1, y - 1) == ALIVE) {
alive++;
}
}
}
if (y + 1 < WORLD_HEIGHT) {
if (x > 0) {
if (world_get(x - 1, y + 1) == ALIVE) {
alive++;
}
}
if (world_get(x, y + 1) == ALIVE) {
alive++;
}
if (x + 1 < WORLD_WIDTH) {
if (world_get(x + 1, y + 1) == ALIVE) {
alive++;
}
}
}
return alive;
}
void step() {
unsigned long capacity = WORLD_WIDTH * WORLD_HEIGHT;
unsigned char *new_world = BitVector_init(capacity);
unsigned char byte = 0x00;
#pragma omp parallel for private(byte)
for (unsigned int i = 0; i < capacity; i += 8) {
byte = 0x00;
for (int k = 0; k < 8; k++) {
unsigned int x = (i + k) % WORLD_WIDTH;
unsigned int y = (i + k) / WORLD_WIDTH;
int8_t alive = get_neighbors(x, y);
if (alive > 3) {
} else if (alive == 3) {
byte |= (ALIVE << k);
} else if (alive == 2) {
byte |= (world_get(x, y) << k);
} else if (alive < 2) {
}
}
BitVector_set8(new_world, i / 8, byte);
}
WORLD = new_world;
}
