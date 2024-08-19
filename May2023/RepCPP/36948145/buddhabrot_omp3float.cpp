

#if _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h> 
#include <string.h> 
#include <omp.h>
#include "util_threads.h"

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#define VERBOSE if(gbVerbose)


float     gnWorldMinX        = -2.102613; 
float     gnWorldMaxX        =  1.200613;
float     gnWorldMinY        = -1.237710; 
float     gnWorldMaxY        =  1.239710;

int       gnMaxDepth         = 1000; 
int       gnWidth            = 1024; 
int       gnHeight           =  768; 
int       gnScale            =   10; 

bool      gbAutoBrightness   = false;
int       gnGreyscaleBias    = -230; 

float     gnScaleR           = 0.09f; 
float     gnScaleG           = 0.11f; 
float     gnScaleB           = 0.18f; 

bool      gbVerbose          = false;
bool      gbSaveRawGreyscale = true ;
bool      gbRotateOutput     = true ;
bool      gbSaveBMP          = true ;

uint32_t  gnImageArea        =    0; 

uint16_t *gpGreyscaleTexels  = NULL; 
uint8_t  *gpChromaticTexels  = NULL; 

const int BUFFER_BACKSPACE   = 64;
char      gaBackspace[ BUFFER_BACKSPACE ];

char     *gpFileNameBMP      = 0; 
char     *gpFileNameRAW      = 0; 



#ifdef _WIN32 
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h> 



typedef struct timeval {
long tv_sec;
long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL); 

SYSTEMTIME  nSystemTime;
FILETIME    nFileTime;
uint64_t    nTime;

GetSystemTime( &nSystemTime );
SystemTimeToFileTime( &nSystemTime, &nFileTime );
nTime =  ((uint64_t)nFileTime.dwLowDateTime )      ;
nTime += ((uint64_t)nFileTime.dwHighDateTime) << 32;

tp->tv_sec  = (long) ((nTime - EPOCH) / 10000000L);
tp->tv_usec = (long) (nSystemTime.wMilliseconds * 1000);
return 0;
}
#else
#include <sys/time.h>
#endif 

struct DataRate
{
char     prefix ;
uint64_t samples;
uint64_t per_sec;
};

class Timer
{
timeval start, end; 
public:
double   elapsed; 
uint8_t  secs;
uint8_t  mins;
uint8_t  hour;
uint32_t days;

DataRate throughput;
char     day[ 16 ]; 
char     hms[ 12 ]; 

void Start()
{
gettimeofday( &start, NULL );
}

void Stop()
{
gettimeofday( &end, NULL );
elapsed = (end.tv_sec - start.tv_sec);

size_t s = elapsed;
secs = s % 60; s /= 60;
mins = s % 60; s /= 60;
hour = s % 24; s /= 24;
days = s;

day[0] = 0;
if( days > 0 )
snprintf( day, 15, "%d day%s, ", days, (days == 1) ? "" : "s" );

sprintf( hms, "%02d:%02d:%02d", hour, mins, secs );
}

void Throughput( uint64_t size )
{
const int MAX_PREFIX = 4;
DataRate datarate[ MAX_PREFIX ] = {
{' ',0,0}, {'K',0,0}, {'M',0,0}, {'G',0,0} 
};

if( !elapsed )
return;

int best = 0;
for( int units = 0; units < MAX_PREFIX; units++ )
{
datarate[ units ].samples = size >> (10*units);
datarate[ units ].per_sec = (uint64_t) (datarate[units].samples / elapsed);
if (datarate[ units ].per_sec > 0)
best = units;
}
throughput = datarate[ best ];
}
};



void AllocImageMemory( const int width, const int height )
{
gnImageArea = width * height;

const size_t nGreyscaleBytes = gnImageArea  * sizeof( uint16_t );
gpGreyscaleTexels = (uint16_t*) malloc( nGreyscaleBytes );          
memset( gpGreyscaleTexels, 0, nGreyscaleBytes );

const size_t chromaticBytes  = gnImageArea * 3 * sizeof( uint8_t ); 
gpChromaticTexels = (uint8_t*) malloc( chromaticBytes );
memset( gpChromaticTexels, 0, chromaticBytes );

for( int i = 0; i < (BUFFER_BACKSPACE-1); i++ )
gaBackspace[ i ] = 8; 

gaBackspace[ BUFFER_BACKSPACE-1 ] = 0;

if(!gnThreadsActive) 
gnThreadsActive = gnThreadsMaximum;
else
omp_set_num_threads( gnThreadsActive );

for( int iThread = 0; iThread < gnThreadsActive; iThread++ )
{
gaThreadsTexels[ iThread ] = (uint16_t*) malloc( nGreyscaleBytes );
memset( gaThreadsTexels[ iThread ], 0,                   nGreyscaleBytes );
}
}


void BMP_WriteColor24bit( const char * filename, const uint8_t *texelsRGB, const int width, const int height )
{
uint32_t headers[13]; 
FILE   * pFileSave;
int x, y, i;

int      nExtraBytes = (width * 3) % 4;
int      nPaddedSize = (width * 3 + nExtraBytes) * height;
uint32_t nPlanes     =  1      ; 
uint32_t nBitcount   = 24 << 16; 

headers[ 0] = nPaddedSize + 54;    
headers[ 1] = 0;                   
headers[ 2] = 54;                  
headers[ 3] = 40;                  
headers[ 4] = width;               
headers[ 5] = height;              
headers[ 6] = nBitcount | nPlanes; 
headers[ 7] = 0;                   
headers[ 8] = nPaddedSize;         
headers[ 9] = 0;                   
headers[10] = 0;                   
headers[11] = 0;                   
headers[12] = 0;                   

pFileSave = fopen(filename, "wb");
if( pFileSave )
{
fprintf(pFileSave, "BM");
for( i = 0; i < 13; i++ )
{
fprintf( pFileSave, "%c", ((headers[i]) >>  0) & 0xFF );
fprintf( pFileSave, "%c", ((headers[i]) >>  8) & 0xFF );
fprintf( pFileSave, "%c", ((headers[i]) >> 16) & 0xFF );
fprintf( pFileSave, "%c", ((headers[i]) >> 24) & 0xFF );
}

for( y = height - 1; y >= 0; y-- )
{
const uint8_t* scanline = &texelsRGB[ y*width*3 ];
for( x = 0; x < width; x++ )
{
uint8_t r = *scanline++;
uint8_t g = *scanline++;
uint8_t b = *scanline++;

fprintf( pFileSave, "%c", b );
fprintf( pFileSave, "%c", g );
fprintf( pFileSave, "%c", r );
}

if( nExtraBytes ) 
for( i = 0; i < nExtraBytes; i++ )
fprintf( pFileSave, "%c", 0 );
}

fclose( pFileSave );
}
}


uint16_t
Image_Greyscale16bitMaxValue( const uint16_t *texels, const int width, const int height )
{
const uint16_t *pSrc = texels;
const int       nLen = width * height;
int       nMax = *pSrc;

for( int iPix = 0; iPix < nLen; iPix++ )
{
if( nMax < *pSrc )
nMax = *pSrc;
pSrc++;
}

return nMax;
}


void
Image_Greyscale16bitRotateRight( const uint16_t *input, const int width, const int height, uint16_t *output_ )
{


for( int y = 0; y < height; y++ )
{
const uint16_t *pSrc = input   + ((width   ) * y);
uint16_t *pDst = output_ + ((height-1) - y);

for( int x = 0; x < width; x++ )
{
*pDst = *pSrc;

pSrc++;
pDst += height;
}
}
}


uint16_t
Image_Greyscale16bitToBrightnessBias( int* bias_, float* scaleR_, float* scaleG_, float* scaleB_ )
{
uint16_t nMaxBrightness = Image_Greyscale16bitMaxValue( gpGreyscaleTexels, gnWidth, gnHeight );

if( gbAutoBrightness )
{
if( nMaxBrightness < 256)
*bias_ = 0;

*bias_ = (int)(-0.045 * nMaxBrightness); 

*scaleR_ = 430.f / (float)nMaxBrightness;
*scaleG_ = 525.f / (float)nMaxBrightness;
*scaleB_ = 860.f / (float)nMaxBrightness;
}

return nMaxBrightness;
}


void
Image_Greyscale16bitToColor24bit(
const uint16_t* greyscale, const int width, const int height,
uint8_t * chromatic_,
const int bias, const double scaleR, const double scaleG, const double scaleB )
{
const int       nLen = width * height;
const uint16_t *pSrc = greyscale;
uint8_t  *pDst = chromatic_;

for( int iPix = 0; iPix < nLen; iPix++ )
{
int i = *pSrc++ + bias  ; 
int r = (int)(i * scaleR);
int g = (int)(i * scaleG);
int b = (int)(i * scaleB);

if (r > 255) r = 255; if (r < 0) r = 0;
if (g > 255) g = 255; if (g < 0) g = 0;
if (b > 255) b = 255; if (b < 0) b = 0;

*pDst++ = r;
*pDst++ = g;
*pDst++ = b;
}
}


char* itoaComma( size_t n, char *output_ = NULL )
{
const  size_t SIZE = 32;
static char   buffer[ SIZE ];
char  *p = buffer + SIZE-1;
*p-- = 0;

while( n >= 1000 )
{
*p-- = '0' + (n % 10); n /= 10;
*p-- = '0' + (n % 10); n /= 10;
*p-- = '0' + (n % 10); n /= 10;
*p-- = ','                    ;
}

{ *p-- = '0' + (n % 10); n /= 10; }
if( n > 0) { *p-- = '0' + (n % 10); n /= 10; }
if( n > 0) { *p-- = '0' + (n % 10); n /= 10; }

if( output_ )
{
char   *pEnd = buffer + SIZE - 1;
size_t  nLen = pEnd - p; 
memcpy( output_, p+1, nLen );
}

return ++p;
}


void
RAW_WriteGreyscale16bit( const char *filename, const uint16_t *texels, const int width, const int height )
{
FILE *file = fopen( filename, "wb" );
if( file )
{
const size_t area = width * height;
fwrite( texels, sizeof( uint16_t ), area, file );
fclose( file );
}
}


inline
void plot( double wx, double wy, double sx, double sy, uint16_t *texels, const int width, const int height, const int maxdepth )
{
float   r = 0.f, i = 0.f; 
float   s      , j      ; 
int     u      , v      ; 

for( int depth = 0; depth < maxdepth; depth++ )
{
s = (r*r - i*i) + wx;
j = (2.0f*r*i)  + wy;

r = s;
i = j;

if ((r*r + i*i) > 4.0f ) 
return;

u = (int) ((r - gnWorldMinX) * sx); 
v = (int) ((i - gnWorldMinY) * sy); 

if( (u < width) && (v < height) && (u >= 0) && (v >= 0) )
texels[ (v * width) + u ]++;
}
}


int Buddhabrot()
{
if( gnScale < 0)
gnScale = 1;

const size_t nCol = gnWidth  * gnScale ; 
const size_t nRow = gnHeight * gnScale ; 

size_t iCel = 0                  ; 
const size_t nCel = nCol     * nRow    ; 

const float  nWorldW = gnWorldMaxX - gnWorldMinX;
const float  nWorldH = gnWorldMaxY - gnWorldMinY;

const float  nWorld2ImageX = (float)(gnWidth  - 1.0f) / nWorldW;
const float  nWorld2ImageY = (float)(gnHeight - 1.0f) / nWorldH;

const float  dx = nWorldW / (nCol - 1.0f);
const float  dy = nWorldH / (nRow - 1.0f);

char sDenominator[ 32 ];
itoaComma( nCel, sDenominator );


#pragma omp parallel for
for( size_t iPix = 0; iPix < nCel; iPix++ )
{
#pragma omp atomic
iCel++;

const int       iTid = omp_get_thread_num(); 
uint16_t* pTex = gaThreadsTexels[ iTid ];

const size_t    iCol = iCel % nCol;
const size_t    iRow = iCel / nCol;

const float     x = gnWorldMinX + (iCol * dx);
const float     y = gnWorldMinY + (iRow * dy);

float     r = 0.f, i = 0.f, s, j;

for (int depth = 0; depth < gnMaxDepth; depth++)
{
s = (r*r - i*i) + x; 
j = (2.0f*r*i)   + y;

r = s;
i = j;

if ((r*r + i*i) > 4.0f) 
{
plot( x, y, nWorld2ImageX, nWorld2ImageY, pTex, gnWidth, gnHeight, gnMaxDepth );
break;
}
}

VERBOSE
if( (iTid == 0) && ((iCel & 0xFFFF) == 0) )
{
{
const size_t n = iCel;
const double percent = (100.0 * n) / nCel;
static char  sNumerator[ 32 ];
itoaComma( n, sNumerator );

printf( "%6.2f%% = %s / %s%s", percent, sNumerator, sDenominator, gaBackspace );
fflush( stdout );
}
}
}

const int nPix = gnWidth  * gnHeight; 
for( int iThread = 0; iThread < gnThreadsActive; iThread++ )
{
const uint16_t *pSrc = gaThreadsTexels[ iThread ];
uint16_t *pDst = gpGreyscaleTexels;

for( int iPix = 0; iPix < nPix; iPix++ )
*pDst++ += *pSrc++;
}

return nCel;
}


int Usage()
{
const char *aOffOn[2] =
{
"OFF"
,"ON "
};

const char *aSaved[2] =
{
"SKIP"
,"SAVE"
};

printf(
"Buddhabrot (OMP) by Michael Pohoreski\n"
"https:
"Usage: [width [height [depth [scale]]]]\n"
"\n"
"-?       Display usage help\n"
"-b       Use auto brightness\n"
"-bmp foo Save .BMP as filename foo\n"
"-j#      Use this # of threads. (Default: %d)\n"
"--no-bmp Don't save .BMP  (Default: %s)\n"
"--no-raw Don't save .data (Default: %s)\n"
"--no-rot Don't rotate BMP (Default: %s)\n"
"-r       Rotation output bitmap 90 degrees right\n"
"-raw foo Save raw greyscale as foo\n"
"-v       Verbose.  Display %% complete\n"
, gnThreadsMaximum
, aSaved[ (int) gbSaveBMP          ]
, aOffOn[ (int) gbRotateOutput     ]
, aOffOn[ (int) gbSaveRawGreyscale ]
);

return 0;
}


void Text_CopyFileName( char *buffer, const char *source, const size_t maxlen )
{
size_t  nLen = strlen( source );

if( nLen >  maxlen )
nLen =  maxlen ;

strncpy( buffer, source, nLen );
buffer[ nLen ] = 0;
}


int main( int nArg, char * aArg[] )
{
gnThreadsMaximum = omp_get_num_procs();
if( gnThreadsMaximum > MAX_THREADS )
gnThreadsMaximum = MAX_THREADS;

int   iArg = 0;

if( nArg > 1 )
{
while( iArg < nArg )
{
char *pArg = aArg[ iArg + 1 ];
if(  !pArg )
break;

if( pArg[0] == '-' )
{
iArg++;
pArg++; 

if( strcmp( pArg, "--no-bmp" ) == 0 )
gbSaveBMP = false;
else 
if( strcmp( pArg, "--no-raw" ) == 0 )
gbSaveRawGreyscale = false;
else 
if( strcmp( pArg, "--no-rot" ) == 0 )
gbRotateOutput = false;
else 
if( *pArg == '?' || (strcmp( pArg, "-help" ) == 0) )
return Usage();
else
if( *pArg == 'b' && (strcmp( pArg, "bmp") != 0) ) 
gbAutoBrightness = true;
else
if( strcmp( pArg, "bmp" ) == 0 )
{
int n = iArg+1; 
if( n < nArg )
{
iArg++;
pArg = aArg[ n ];
gpFileNameBMP = pArg;

n = iArg + 1;
if( n < nArg )
{
pArg = aArg[ n ] - 1; 
*pArg = 0; 
}
}
}
else
if( *pArg == 'j' )
{
int i = atoi( pArg+1 ); 
if( i > 0 )
gnThreadsActive = i;
if( gnThreadsActive > MAX_THREADS )
gnThreadsActive = MAX_THREADS;
}
else
if( *pArg == 'r' && (strcmp( pArg, "raw") != 0) ) 
gbRotateOutput = true;
else
if( *pArg == 'v' )
gbVerbose = true;
else
if( strcmp( pArg, "raw" ) == 0 )
{
int n = iArg+1; 
if( n < nArg )
{
iArg++;
pArg = aArg[ n ];
gpFileNameRAW = pArg;

n = iArg + 1;
if( n < nArg )
{
pArg = aArg[ n ] - 1; 
*pArg = 0; 
}
}
}
else
printf( "Unrecognized option: %c\n", *pArg ); 
}
else
break;
}
}

if ((iArg+1) < nArg) gnWidth    = atoi( aArg[iArg+1] );
if ((iArg+2) < nArg) gnHeight   = atoi( aArg[iArg+2] );
if ((iArg+3) < nArg) gnMaxDepth = atoi( aArg[iArg+3] );
if ((iArg+4) < nArg) gnScale    = atoi( aArg[iArg+4] );

printf( "Width: %d  Height: %d  Depth: %d  Scale: %d  RotateBMP: %d  SaveRaw: %d\n", gnWidth, gnHeight, gnMaxDepth, gnScale, gbRotateOutput, gbSaveRawGreyscale );

AllocImageMemory( gnWidth, gnHeight );

printf( "Using: %u / %u threads\n", gnThreadsActive, gnThreadsMaximum );

Timer stopwatch;
stopwatch.Start();
int nCells = Buddhabrot();
stopwatch.Stop();

VERBOSE printf( "100.00%%\n" );
stopwatch.Throughput( nCells ); 
printf( "%d %cpix/s (%d pixels, %.f seconds = %s%s)\n"
, (int)stopwatch.throughput.per_sec, stopwatch.throughput.prefix
, nCells
, stopwatch.elapsed
, stopwatch.day
, stopwatch.hms
);

int nMaxBrightness = Image_Greyscale16bitToBrightnessBias( &gnGreyscaleBias, &gnScaleR, &gnScaleG, &gnScaleB ); 
printf( "Max brightness: %d\n", nMaxBrightness );

const int PATH_SIZE = 256;
const char *pBaseName = "omp3float_buddhabrot";
char filenameRAW[ PATH_SIZE ];
char filenameBMP[ PATH_SIZE ];

if( gbSaveRawGreyscale )
{
if( gpFileNameRAW )
Text_CopyFileName( filenameRAW, gpFileNameRAW, PATH_SIZE-1 ); 
else
sprintf( filenameRAW, "raw_%s_%dx%d_d%d_s%d_j%d.u16.data"
, pBaseName, gnWidth, gnHeight, gnMaxDepth, gnScale, gnThreadsActive );

RAW_WriteGreyscale16bit( filenameRAW, gpGreyscaleTexels, gnWidth, gnHeight );
printf( "Saved: %s\n", filenameRAW );
}

uint16_t *pRotatedTexels = gpGreyscaleTexels; 
if( gbRotateOutput )
{
const int nBytes =  gnImageArea * sizeof( uint16_t );
pRotatedTexels = (uint16_t*) malloc( nBytes ); 
Image_Greyscale16bitRotateRight( gpGreyscaleTexels, gnWidth, gnHeight, pRotatedTexels );

int t = gnWidth;
gnWidth = gnHeight;
gnHeight = t;
}

if( gbSaveBMP )
{
if( gpFileNameBMP )
Text_CopyFileName( filenameBMP, gpFileNameBMP, PATH_SIZE-1 ); 
else
sprintf( filenameBMP, "%s_%dx%d_%d.bmp", pBaseName, gnWidth, gnHeight, gnMaxDepth );

Image_Greyscale16bitToColor24bit( pRotatedTexels, gnWidth, gnHeight, gpChromaticTexels, gnGreyscaleBias, gnScaleR, gnScaleG, gnScaleB );
BMP_WriteColor24bit( filenameBMP, gpChromaticTexels, gnWidth, gnHeight );
printf( "Saved: %s\n", filenameBMP );
}

return 0;
}
