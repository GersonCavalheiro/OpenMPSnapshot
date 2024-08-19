#pragma GCC target ("crc32")
unsigned int
crc32b (unsigned int x, unsigned char y)
{
return __builtin_ia32_crc32qi (x, y);
}
unsigned int
crc32w (unsigned int x, unsigned short y)
{
return __builtin_ia32_crc32hi (x, y);
}
unsigned int
crc32d (unsigned int x, unsigned int y)
{
return __builtin_ia32_crc32si (x, y);
}
