

#ifndef UFML_VR_CINE_H_
#define UFML_VR_CINE_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef int bool32_t;


#if !defined(_TIMEDEFINED_)
#define _TIMEDEFINED_
typedef uint32_t FRACTIONS, *PFRACTIONS;

typedef struct tagTIME64
{
FRACTIONS fractions;     

uint32_t seconds;     

} TIME64, *PTIME64;


typedef struct tagTC
{
uint8_t framesU:4; 
uint8_t framesT:2; 
uint8_t dropFrameFlag:1; 
uint8_t colorFrameFlag:1; 
uint8_t secondsU:4; 
uint8_t secondsT:3; 
uint8_t flag1:1; 
uint8_t minutesU:4; 
uint8_t minutesT:3; 
uint8_t flag2:1; 
uint8_t hoursU:4; 
uint8_t hoursT:2; 
uint8_t flag3:1; 
uint8_t flag4:1; 
uint32_t userBitData; 
}TC, *PTC;

typedef struct tagTCU
{
uint32_t frames;
uint32_t seconds;
uint32_t minutes;
uint32_t hours;
bool32_t dropFrameFlag;
bool32_t colorFrameFlag;
bool32_t flag1;
bool32_t flag2;
bool32_t flag3;
bool32_t flag4;
uint32_t userBitData;
}TCU, *PTCU;

#endif  


#if !defined(_WBGAIN_)
#define _WBGAIN_
typedef struct tagWBGAIN
{
float R; 
float B; 
} WBGAIN, *PWBGAIN;
#endif


#if !defined(_WINDOWS)
typedef struct tagRECT
{
int32_t left;
int32_t top;
int32_t right;
int32_t bottom;
} RECT, *PRECT;
#endif 

#define OLDMAXFILENAME 65 

#define MAXLENDESCRIPTION_OLD 121 

#define MAXLENDESCRIPTION 4096 


typedef struct tagIMFILTER
{
int32_t dim; 
int32_t shifts; 
int32_t bias; 
int32_t Coef[5*5]; 
}
IMFILTER, *PIMFILTER;

typedef struct tagCINEFILEHEADER
{
uint16_t Type;
uint16_t Headersize;
uint16_t Compression;
uint16_t Version;
int32_t FirstMovieImage;
uint32_t TotalImageCount;
int32_t FirstImageNo;
uint32_t ImageCount;
uint32_t OffImageHeader; 
uint32_t OffSetup; 
uint32_t OffImageOffsets; 
TIME64 TriggerTime;
} CINEFILEHEADER;

typedef struct tagBITMAPINFOHEADER
{
uint32_t biSize;
int32_t biWidth;
int32_t biHeight;
uint16_t biPlanes;
uint16_t biBitCount;
uint32_t biCompression;
uint32_t biSizeImage;
int32_t biXPelsPerMeter;
int32_t biYPelsPerMeter;
uint32_t biClrUsed;
uint32_t biClrImportant;
} BITMAPINFOHEADER;



#pragma pack(1)
typedef struct tagSETUP
{
uint16_t FrameRate16; 
uint16_t Shutter16; 
uint16_t PostTrigger16; 
uint16_t FrameDelay16; 
uint16_t AspectRatio; 
uint16_t Res7; 
uint16_t Res8; 
uint8_t Res9;  
uint8_t Res10; 
uint8_t Res11; 
uint8_t TrigFrame; 
uint8_t Res12; 
char DescriptionOld[MAXLENDESCRIPTION_OLD]; 
uint16_t Mark; 
uint16_t Length; 
uint16_t Res13; 
uint16_t SigOption; 
int16_t BinChannels; 
uint8_t SamplesPerImage; 
char BinName[8][11]; 
uint16_t AnaOption; 
int16_t AnaChannels; 
uint8_t Res6;        
uint8_t AnaBoard;    
int16_t ChOption[8]; 
float AnaGain[8];    
char AnaUnit[8][6];  
char AnaName[8][11]; 
int32_t lFirstImage; 
uint32_t dwImageCount; 
int16_t nQFactor;   


uint16_t wCineFileType; 
char szCinePath[4][OLDMAXFILENAME]; 

uint16_t Res14; 

uint8_t Res15;  

uint8_t Res16;  

uint16_t Res17;  

double Res18; 
double Res19; 

uint16_t Res20; 
int32_t Res1; 
int32_t Res2; 
int32_t Res3; 

uint16_t ImWidth; 
uint16_t ImHeight; 
uint16_t EDRShutter16; 
uint32_t Serial; 

int32_t Saturation; 
uint8_t Res5;  

uint32_t AutoExposure; 

bool32_t bFlipH; 
bool32_t bFlipV; 
uint32_t Grid; 
uint32_t FrameRate; 
uint32_t Shutter; 

uint32_t EDRShutter; 

uint32_t PostTrigger; 
uint32_t FrameDelay; 
bool32_t bEnableColor; 

uint32_t CameraVersion; 

uint32_t FirmwareVersion;
uint32_t SoftwareVersion;

int32_t RecordingTimeZone;

uint32_t CFA; 

int32_t Bright; 
int32_t Contrast; 
int32_t Gamma; 

uint32_t Res21; 

uint32_t AutoExpLevel; 
uint32_t AutoExpSpeed; 
RECT AutoExpRect; 
WBGAIN WBGain[4]; 
int32_t Rotate;  


WBGAIN WBView;  
uint32_t RealBPP; 

uint32_t Conv8Min;
uint32_t Conv8Max; 

int32_t FilterCode; 
int32_t FilterParam; 
IMFILTER UF; 
uint32_t BlackCalSVer; 
uint32_t WhiteCalSVer; 
uint32_t GrayCalSVer; 
bool32_t bStampTime;

uint32_t SoundDest; 

uint32_t FRPSteps; 

int32_t FRPImgNr[16]; 

uint32_t FRPRate[16]; 
uint32_t FRPExp[16]; 

int32_t MCCnt; 

float MCPercent[64]; 

uint32_t CICalib; 

uint32_t CalibWidth; 
uint32_t CalibHeight;
uint32_t CalibRate; 
uint32_t CalibExp; 
uint32_t CalibEDR; 
uint32_t CalibTemp; 
uint32_t HeadSerial[4]; 
uint32_t RangeCode; 
uint32_t RangeSize; 
uint32_t Decimation; 

uint32_t MasterSerial; 

uint32_t Sensor; 

uint32_t ShutterNs; 
uint32_t EDRShutterNs; 
uint32_t FrameDelayNs; 

uint32_t ImPosXAcq; 

uint32_t ImPosYAcq;

uint32_t ImWidthAcq; 

uint32_t ImHeightAcq;

char Description[MAXLENDESCRIPTION];
bool32_t RisingEdge; 
uint32_t FilterTime; 
bool32_t LongReady; 

bool32_t ShutterOff;

uint8_t Res4[16]; 

bool32_t bMetaWB; 

int32_t Hue; 

int32_t BlackLevel; 
int32_t WhiteLevel; 

char LensDescription[256];
float LensAperture; 
float LensFocusDistance;
float LensFocalLength; 

float fOffset;
float fGain; 
float fSaturation; 
float fHue; 

float fGamma;
float fGammaR; 

float fGammaB;
float fFlare; 

float fPedestalR; 
float fPedestalG; 
float fPedestalB;

float fChroma; 

char  ToneLabel[256];
int32_t   TonePoints;
float fTone[32*2]; 

char UserMatrixLabel[256];
bool32_t EnableMatrices;
float fUserMatrix[9]; 

bool32_t EnableCrop; 
RECT CropRect;
bool32_t EnableResample;
uint32_t ResampleWidth;
uint32_t ResampleHeight;

float fGain16_8; 

uint32_t FRPShape[16];
TC TrigTC; 
float fPbRate; 
float fTcRate; 

char CineName[256]; 


} SETUP, *PSETUP;


typedef struct tagINFORMATIONBLOCK
{
uint32_t BlockSize;
uint16_t Type;
uint16_t Reserved;
uint8_t *Data; 
} INFORMATIONBLOCK;

typedef struct tagANNOTATIONBLOCK
{
uint32_t AnnotationSize;
uint8_t *Annotation;  
uint32_t ImageSize;
} ANNOTATIONBLOCK;

#pragma pack()
#ifdef __cplusplus
}
#endif

#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

BOOST_FUSION_ADAPT_STRUCT(
tagCINEFILEHEADER,
(    uint16_t, Type)
(    uint16_t, Headersize)
(    uint16_t, Compression)
(    uint16_t, Version)
(    int32_t, FirstMovieImage)
(    uint32_t, TotalImageCount)
(    int32_t, FirstImageNo)
(    uint32_t, ImageCount)
(    uint32_t, OffImageHeader)
(    uint32_t, OffSetup)
(    uint32_t, OffImageOffsets)
)

#endif 
