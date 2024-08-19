#include <fstream>


#pragma pack(push, 1)  
typedef struct {
unsigned short    bfType;
unsigned long    bfSize;
unsigned short    bfReserved1;
unsigned short    bfReserved2;
unsigned long    bfOffBits;
} BITMAPFILEHEADER;
#pragma pack(pop)


typedef struct {
unsigned long       biSize;
long        biWidth;
long        biHeight;
unsigned short        biPlanes;
unsigned short       biBitCount;
unsigned long       biCompression;
unsigned long       biSizeImage;
long        biXPelsPerMeter;
long        biYPelsPerMeter;
unsigned long       biClrUsed;
unsigned long       biClrImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBTRIPLE {
unsigned char rgbtBlue;
unsigned char     rgbtGreen;
unsigned char     rgbtRed;
} RGBTRIPLE;



void BMPRead(RGBTRIPLE **&, BITMAPFILEHEADER&, BITMAPINFOHEADER&, const char *);
void BMPWrite(RGBTRIPLE**& rgb, int imWidth, int imHeight, const char* fout);

unsigned char get_row_data_padding(unsigned int width);
unsigned int bmp24b_file_size_calc(unsigned int width, unsigned int height);

unsigned char get_row_data_padding(unsigned int width) {
return (width % 4 == 0) ? 0 : (4 - (width * sizeof(RGBTRIPLE)) % 4);
}

unsigned int bmp24b_file_size_calc(unsigned int width, unsigned int height) {
return sizeof(BITMAPFILEHEADER)+ sizeof(BITMAPINFOHEADER) +  height * (width* sizeof(RGBTRIPLE) + get_row_data_padding(width));
}


void BMPRead(RGBTRIPLE** &rgb, BITMAPFILEHEADER &header, \
BITMAPINFOHEADER &bmiHeader, const char* fin)
{
std::ifstream InFile(fin, std::ios::binary);
InFile.read((char*)(&header), sizeof(BITMAPFILEHEADER));
InFile.read((char*)(&bmiHeader), sizeof(BITMAPINFOHEADER));
rgb = new RGBTRIPLE*[bmiHeader.biHeight];
rgb[0] = new RGBTRIPLE[bmiHeader.biWidth*bmiHeader.biHeight];
for (int i = 1; i < bmiHeader.biHeight; i++)
{   
rgb[i] = &rgb[0][bmiHeader.biWidth*i];
}
int padding = get_row_data_padding(bmiHeader.biWidth);
char tmp[3] = { 0,0,0 };
for (int i = 0; i < bmiHeader.biHeight; i++)
{
InFile.read((char*)(&rgb[bmiHeader.biHeight-1-i][0]), bmiHeader.biWidth*sizeof(RGBTRIPLE)); 
if (padding > 0)
InFile.read((char*)(&tmp[0]), padding);
}
InFile.close();
}

void BMPWrite(RGBTRIPLE**& rgb, int imWidth , int imHeight, const char* fout)
{
std::ofstream OutFile(fout, std::ios::binary);
BITMAPFILEHEADER header = { 0 };
header.bfType = ('M' << 8) + 'B';
header.bfSize = bmp24b_file_size_calc(imWidth, imHeight);;
header.bfOffBits = 54;
BITMAPINFOHEADER bmiHeader = { 0 };
bmiHeader.biSize = 40;
bmiHeader.biWidth = imWidth;
bmiHeader.biHeight = imHeight;
bmiHeader.biPlanes = 1;
bmiHeader.biBitCount = 24;
bmiHeader.biSizeImage = header.bfSize - sizeof(BITMAPINFOHEADER)- sizeof(BITMAPFILEHEADER);
OutFile.write((char*)(&header), sizeof(BITMAPFILEHEADER));
OutFile.write((char*)(&bmiHeader), sizeof(BITMAPINFOHEADER));
int padding = get_row_data_padding(bmiHeader.biWidth);
char tmp[3] = { 0,0,0 };
for (int i = 0; i < bmiHeader.biHeight; i++)
{
OutFile.write((char*)&(rgb[bmiHeader.biHeight - i - 1][0]), bmiHeader.biWidth * sizeof(RGBTRIPLE));
if (padding > 0)
OutFile.write((char*)(&tmp[0]), padding);
}
OutFile.close();
}

