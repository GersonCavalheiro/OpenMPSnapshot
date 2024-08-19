#pragma once

struct VirtualPageDescriptor {
bool R;
bool M;
int indexOfFrameInPhysicalMemory;
char place;
};
