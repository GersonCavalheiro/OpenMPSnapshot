#ifndef FREEIMAGE_H
#define FREEIMAGE_H
#define FREEIMAGE_MAJOR_VERSION   3
#define FREEIMAGE_MINOR_VERSION   18
#define FREEIMAGE_RELEASE_SERIAL  0
#include <wchar.h>	
#if defined(FREEIMAGE_LIB)
#define DLL_API
#define DLL_CALLCONV
#else
#if defined(_WIN32) || defined(__WIN32__)
#define DLL_CALLCONV __stdcall
#ifdef FREEIMAGE_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif 
#else 
#if defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#ifndef GCC_HASCLASSVISIBILITY
#define GCC_HASCLASSVISIBILITY
#endif
#endif 
#define DLL_CALLCONV
#if defined(GCC_HASCLASSVISIBILITY)
#define DLL_API __attribute__ ((visibility("default")))
#else
#define DLL_API
#endif		
#endif 
#endif 
#if (!defined(FREEIMAGE_BIGENDIAN) && !defined(FREEIMAGE_LITTLEENDIAN))
#if (defined(BYTE_ORDER) && BYTE_ORDER==BIG_ENDIAN) || (defined(__BYTE_ORDER) && __BYTE_ORDER==__BIG_ENDIAN) || (defined(__BYTE_ORDER) && __BYTE_ORDER==__ORDER_BIG_ENDIAN__) || defined(__BIG_ENDIAN__)
#define FREEIMAGE_BIGENDIAN
#endif 
#endif 
#define FREEIMAGE_COLORORDER_BGR    0
#define FREEIMAGE_COLORORDER_RGB    1
#if (!defined(FREEIMAGE_COLORORDER)) || ((FREEIMAGE_COLORORDER != FREEIMAGE_COLORORDER_BGR) && (FREEIMAGE_COLORORDER != FREEIMAGE_COLORORDER_RGB))
#if defined(FREEIMAGE_BIGENDIAN)
#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_RGB
#else
#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_BGR
#endif 
#endif 
#if defined(__BORLANDC__)
#pragma option push -b
#endif
#ifdef __cplusplus
#define FI_DEFAULT(x)	= x
#define FI_ENUM(x)      enum x
#define FI_STRUCT(x)	struct x
#else
#define FI_DEFAULT(x)
#define FI_ENUM(x)      typedef int x; enum x
#define FI_STRUCT(x)	typedef struct x x; struct x
#endif
FI_STRUCT (FIBITMAP) { void *data; };
FI_STRUCT (FIMULTIBITMAP) { void *data; };
#if defined(__MINGW32__) && defined(_WINDOWS_H)
#define _WINDOWS_	
#endif 
#ifndef _WINDOWS_
#define _WINDOWS_
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef NULL
#define NULL 0
#endif
#ifndef SEEK_SET
#define SEEK_SET  0
#define SEEK_CUR  1
#define SEEK_END  2
#endif
#ifndef _MSC_VER
#include <inttypes.h>
typedef int32_t BOOL;
typedef uint8_t BYTE;
typedef uint16_t WORD;
typedef uint32_t DWORD;
typedef int32_t LONG;
typedef int64_t INT64;
typedef uint64_t UINT64;
#else
typedef long BOOL;
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;
typedef long LONG;
typedef signed __int64 INT64;
typedef unsigned __int64 UINT64;
#endif 
#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif 
typedef struct tagRGBQUAD {
#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
BYTE rgbBlue;
BYTE rgbGreen;
BYTE rgbRed;
#else
BYTE rgbRed;
BYTE rgbGreen;
BYTE rgbBlue;
#endif 
BYTE rgbReserved;
} RGBQUAD;
typedef struct tagRGBTRIPLE {
#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
BYTE rgbtBlue;
BYTE rgbtGreen;
BYTE rgbtRed;
#else
BYTE rgbtRed;
BYTE rgbtGreen;
BYTE rgbtBlue;
#endif 
} RGBTRIPLE;
#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(pop)
#else
#pragma pack()
#endif 
typedef struct tagBITMAPINFOHEADER{
DWORD biSize;
LONG  biWidth; 
LONG  biHeight; 
WORD  biPlanes; 
WORD  biBitCount;
DWORD biCompression; 
DWORD biSizeImage; 
LONG  biXPelsPerMeter; 
LONG  biYPelsPerMeter; 
DWORD biClrUsed; 
DWORD biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER; 
typedef struct tagBITMAPINFO { 
BITMAPINFOHEADER bmiHeader; 
RGBQUAD          bmiColors[1];
} BITMAPINFO, *PBITMAPINFO;
#endif 
#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif 
typedef struct tagFIRGB16 {
WORD red;
WORD green;
WORD blue;
} FIRGB16;
typedef struct tagFIRGBA16 {
WORD red;
WORD green;
WORD blue;
WORD alpha;
} FIRGBA16;
typedef struct tagFIRGBF {
float red;
float green;
float blue;
} FIRGBF;
typedef struct tagFIRGBAF {
float red;
float green;
float blue;
float alpha;
} FIRGBAF;
typedef struct tagFICOMPLEX {
double r;
double i;
} FICOMPLEX;
#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(pop)
#else
#pragma pack()
#endif 
#ifndef FREEIMAGE_BIGENDIAN
#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
#define FI_RGBA_RED				2
#define FI_RGBA_GREEN			1
#define FI_RGBA_BLUE			0
#define FI_RGBA_ALPHA			3
#define FI_RGBA_RED_MASK		0x00FF0000
#define FI_RGBA_GREEN_MASK		0x0000FF00
#define FI_RGBA_BLUE_MASK		0x000000FF
#define FI_RGBA_ALPHA_MASK		0xFF000000
#define FI_RGBA_RED_SHIFT		16
#define FI_RGBA_GREEN_SHIFT		8
#define FI_RGBA_BLUE_SHIFT		0
#define FI_RGBA_ALPHA_SHIFT		24
#else
#define FI_RGBA_RED				0
#define FI_RGBA_GREEN			1
#define FI_RGBA_BLUE			2
#define FI_RGBA_ALPHA			3
#define FI_RGBA_RED_MASK		0x000000FF
#define FI_RGBA_GREEN_MASK		0x0000FF00
#define FI_RGBA_BLUE_MASK		0x00FF0000
#define FI_RGBA_ALPHA_MASK		0xFF000000
#define FI_RGBA_RED_SHIFT		0
#define FI_RGBA_GREEN_SHIFT		8
#define FI_RGBA_BLUE_SHIFT		16
#define FI_RGBA_ALPHA_SHIFT		24
#endif 
#else
#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
#define FI_RGBA_RED				2
#define FI_RGBA_GREEN			1
#define FI_RGBA_BLUE			0
#define FI_RGBA_ALPHA			3
#define FI_RGBA_RED_MASK		0x0000FF00
#define FI_RGBA_GREEN_MASK		0x00FF0000
#define FI_RGBA_BLUE_MASK		0xFF000000
#define FI_RGBA_ALPHA_MASK		0x000000FF
#define FI_RGBA_RED_SHIFT		8
#define FI_RGBA_GREEN_SHIFT		16
#define FI_RGBA_BLUE_SHIFT		24
#define FI_RGBA_ALPHA_SHIFT		0
#else
#define FI_RGBA_RED				0
#define FI_RGBA_GREEN			1
#define FI_RGBA_BLUE			2
#define FI_RGBA_ALPHA			3
#define FI_RGBA_RED_MASK		0xFF000000
#define FI_RGBA_GREEN_MASK		0x00FF0000
#define FI_RGBA_BLUE_MASK		0x0000FF00
#define FI_RGBA_ALPHA_MASK		0x000000FF
#define FI_RGBA_RED_SHIFT		24
#define FI_RGBA_GREEN_SHIFT		16
#define FI_RGBA_BLUE_SHIFT		8
#define FI_RGBA_ALPHA_SHIFT		0
#endif 
#endif 
#define FI_RGBA_RGB_MASK		(FI_RGBA_RED_MASK|FI_RGBA_GREEN_MASK|FI_RGBA_BLUE_MASK)
#define FI16_555_RED_MASK		0x7C00
#define FI16_555_GREEN_MASK		0x03E0
#define FI16_555_BLUE_MASK		0x001F
#define FI16_555_RED_SHIFT		10
#define FI16_555_GREEN_SHIFT	5
#define FI16_555_BLUE_SHIFT		0
#define FI16_565_RED_MASK		0xF800
#define FI16_565_GREEN_MASK		0x07E0
#define FI16_565_BLUE_MASK		0x001F
#define FI16_565_RED_SHIFT		11
#define FI16_565_GREEN_SHIFT	5
#define FI16_565_BLUE_SHIFT		0
#define FIICC_DEFAULT			0x00
#define FIICC_COLOR_IS_CMYK		0x01
FI_STRUCT (FIICCPROFILE) { 
WORD    flags;	
DWORD	size;	
void   *data;	
};
FI_ENUM(FREE_IMAGE_FORMAT) {
FIF_UNKNOWN = -1,
FIF_BMP		= 0,
FIF_ICO		= 1,
FIF_JPEG	= 2,
FIF_JNG		= 3,
FIF_KOALA	= 4,
FIF_LBM		= 5,
FIF_IFF = FIF_LBM,
FIF_MNG		= 6,
FIF_PBM		= 7,
FIF_PBMRAW	= 8,
FIF_PCD		= 9,
FIF_PCX		= 10,
FIF_PGM		= 11,
FIF_PGMRAW	= 12,
FIF_PNG		= 13,
FIF_PPM		= 14,
FIF_PPMRAW	= 15,
FIF_RAS		= 16,
FIF_TARGA	= 17,
FIF_TIFF	= 18,
FIF_WBMP	= 19,
FIF_PSD		= 20,
FIF_CUT		= 21,
FIF_XBM		= 22,
FIF_XPM		= 23,
FIF_DDS		= 24,
FIF_GIF     = 25,
FIF_HDR		= 26,
FIF_FAXG3	= 27,
FIF_SGI		= 28,
FIF_EXR		= 29,
FIF_J2K		= 30,
FIF_JP2		= 31,
FIF_PFM		= 32,
FIF_PICT	= 33,
FIF_RAW		= 34,
FIF_WEBP	= 35,
FIF_JXR		= 36
};
FI_ENUM(FREE_IMAGE_TYPE) {
FIT_UNKNOWN = 0,	
FIT_BITMAP  = 1,	
FIT_UINT16	= 2,	
FIT_INT16	= 3,	
FIT_UINT32	= 4,	
FIT_INT32	= 5,	
FIT_FLOAT	= 6,	
FIT_DOUBLE	= 7,	
FIT_COMPLEX	= 8,	
FIT_RGB16	= 9,	
FIT_RGBA16	= 10,	
FIT_RGBF	= 11,	
FIT_RGBAF	= 12	
};
FI_ENUM(FREE_IMAGE_COLOR_TYPE) {
FIC_MINISWHITE = 0,		
FIC_MINISBLACK = 1,		
FIC_RGB        = 2,		
FIC_PALETTE    = 3,		
FIC_RGBALPHA   = 4,		
FIC_CMYK       = 5		
};
FI_ENUM(FREE_IMAGE_QUANTIZE) {
FIQ_WUQUANT = 0,		
FIQ_NNQUANT = 1,		
FIQ_LFPQUANT = 2		
};
FI_ENUM(FREE_IMAGE_DITHER) {
FID_FS			= 0,	
FID_BAYER4x4	= 1,	
FID_BAYER8x8	= 2,	
FID_CLUSTER6x6	= 3,	
FID_CLUSTER8x8	= 4,	
FID_CLUSTER16x16= 5,	
FID_BAYER16x16	= 6		
};
FI_ENUM(FREE_IMAGE_JPEG_OPERATION) {
FIJPEG_OP_NONE			= 0,	
FIJPEG_OP_FLIP_H		= 1,	
FIJPEG_OP_FLIP_V		= 2,	
FIJPEG_OP_TRANSPOSE		= 3,	
FIJPEG_OP_TRANSVERSE	= 4,	
FIJPEG_OP_ROTATE_90		= 5,	
FIJPEG_OP_ROTATE_180	= 6,	
FIJPEG_OP_ROTATE_270	= 7		
};
FI_ENUM(FREE_IMAGE_TMO) {
FITMO_DRAGO03	 = 0,	
FITMO_REINHARD05 = 1,	
FITMO_FATTAL02	 = 2	
};
FI_ENUM(FREE_IMAGE_FILTER) {
FILTER_BOX		  = 0,	
FILTER_BICUBIC	  = 1,	
FILTER_BILINEAR   = 2,	
FILTER_BSPLINE	  = 3,	
FILTER_CATMULLROM = 4,	
FILTER_LANCZOS3	  = 5	
};
FI_ENUM(FREE_IMAGE_COLOR_CHANNEL) {
FICC_RGB	= 0,	
FICC_RED	= 1,	
FICC_GREEN	= 2,	
FICC_BLUE	= 3,	
FICC_ALPHA	= 4,	
FICC_BLACK	= 5,	
FICC_REAL	= 6,	
FICC_IMAG	= 7,	
FICC_MAG	= 8,	
FICC_PHASE	= 9		
};
FI_ENUM(FREE_IMAGE_MDTYPE) {
FIDT_NOTYPE		= 0,	
FIDT_BYTE		= 1,	
FIDT_ASCII		= 2,	
FIDT_SHORT		= 3,	
FIDT_LONG		= 4,	
FIDT_RATIONAL	= 5,	
FIDT_SBYTE		= 6,	
FIDT_UNDEFINED	= 7,	
FIDT_SSHORT		= 8,	
FIDT_SLONG		= 9,	
FIDT_SRATIONAL	= 10,	
FIDT_FLOAT		= 11,	
FIDT_DOUBLE		= 12,	
FIDT_IFD		= 13,	
FIDT_PALETTE	= 14,	
FIDT_LONG8		= 16,	
FIDT_SLONG8		= 17,	
FIDT_IFD8		= 18	
};
FI_ENUM(FREE_IMAGE_MDMODEL) {
FIMD_NODATA			= -1,
FIMD_COMMENTS		= 0,	
FIMD_EXIF_MAIN		= 1,	
FIMD_EXIF_EXIF		= 2,	
FIMD_EXIF_GPS		= 3,	
FIMD_EXIF_MAKERNOTE = 4,	
FIMD_EXIF_INTEROP	= 5,	
FIMD_IPTC			= 6,	
FIMD_XMP			= 7,	
FIMD_GEOTIFF		= 8,	
FIMD_ANIMATION		= 9,	
FIMD_CUSTOM			= 10,	
FIMD_EXIF_RAW		= 11	
};
FI_STRUCT (FIMETADATA) { void *data; };
FI_STRUCT (FITAG) { void *data; };
#ifndef FREEIMAGE_IO
#define FREEIMAGE_IO
typedef void* fi_handle;
typedef unsigned (DLL_CALLCONV *FI_ReadProc) (void *buffer, unsigned size, unsigned count, fi_handle handle);
typedef unsigned (DLL_CALLCONV *FI_WriteProc) (void *buffer, unsigned size, unsigned count, fi_handle handle);
typedef int (DLL_CALLCONV *FI_SeekProc) (fi_handle handle, long offset, int origin);
typedef long (DLL_CALLCONV *FI_TellProc) (fi_handle handle);
#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif 
FI_STRUCT(FreeImageIO) {
FI_ReadProc  read_proc;     
FI_WriteProc write_proc;    
FI_SeekProc  seek_proc;     
FI_TellProc  tell_proc;     
};
#if (defined(_WIN32) || defined(__WIN32__))
#pragma pack(pop)
#else
#pragma pack()
#endif 
FI_STRUCT (FIMEMORY) { void *data; };
#endif 
#ifndef PLUGINS
#define PLUGINS
typedef const char *(DLL_CALLCONV *FI_FormatProc)(void);
typedef const char *(DLL_CALLCONV *FI_DescriptionProc)(void);
typedef const char *(DLL_CALLCONV *FI_ExtensionListProc)(void);
typedef const char *(DLL_CALLCONV *FI_RegExprProc)(void);
typedef void *(DLL_CALLCONV *FI_OpenProc)(FreeImageIO *io, fi_handle handle, BOOL read);
typedef void (DLL_CALLCONV *FI_CloseProc)(FreeImageIO *io, fi_handle handle, void *data);
typedef int (DLL_CALLCONV *FI_PageCountProc)(FreeImageIO *io, fi_handle handle, void *data);
typedef int (DLL_CALLCONV *FI_PageCapabilityProc)(FreeImageIO *io, fi_handle handle, void *data);
typedef FIBITMAP *(DLL_CALLCONV *FI_LoadProc)(FreeImageIO *io, fi_handle handle, int page, int flags, void *data);
typedef BOOL (DLL_CALLCONV *FI_SaveProc)(FreeImageIO *io, FIBITMAP *dib, fi_handle handle, int page, int flags, void *data);
typedef BOOL (DLL_CALLCONV *FI_ValidateProc)(FreeImageIO *io, fi_handle handle);
typedef const char *(DLL_CALLCONV *FI_MimeProc)(void);
typedef BOOL (DLL_CALLCONV *FI_SupportsExportBPPProc)(int bpp);
typedef BOOL (DLL_CALLCONV *FI_SupportsExportTypeProc)(FREE_IMAGE_TYPE type);
typedef BOOL (DLL_CALLCONV *FI_SupportsICCProfilesProc)(void);
typedef BOOL (DLL_CALLCONV *FI_SupportsNoPixelsProc)(void);
FI_STRUCT (Plugin) {
FI_FormatProc format_proc;
FI_DescriptionProc description_proc;
FI_ExtensionListProc extension_proc;
FI_RegExprProc regexpr_proc;
FI_OpenProc open_proc;
FI_CloseProc close_proc;
FI_PageCountProc pagecount_proc;
FI_PageCapabilityProc pagecapability_proc;
FI_LoadProc load_proc;
FI_SaveProc save_proc;
FI_ValidateProc validate_proc;
FI_MimeProc mime_proc;
FI_SupportsExportBPPProc supports_export_bpp_proc;
FI_SupportsExportTypeProc supports_export_type_proc;
FI_SupportsICCProfilesProc supports_icc_profiles_proc;
FI_SupportsNoPixelsProc supports_no_pixels_proc;
};
typedef void (DLL_CALLCONV *FI_InitProc)(Plugin *plugin, int format_id);
#endif 
#define FIF_LOAD_NOPIXELS 0x8000	
#define BMP_DEFAULT         0
#define BMP_SAVE_RLE        1
#define CUT_DEFAULT         0
#define DDS_DEFAULT			0
#define EXR_DEFAULT			0		
#define EXR_FLOAT			0x0001	
#define EXR_NONE			0x0002	
#define EXR_ZIP				0x0004	
#define EXR_PIZ				0x0008	
#define EXR_PXR24			0x0010	
#define EXR_B44				0x0020	
#define EXR_LC				0x0040	
#define FAXG3_DEFAULT		0
#define GIF_DEFAULT			0
#define GIF_LOAD256			1		
#define GIF_PLAYBACK		2		
#define HDR_DEFAULT			0
#define ICO_DEFAULT         0
#define ICO_MAKEALPHA		1		
#define IFF_DEFAULT         0
#define J2K_DEFAULT			0		
#define JP2_DEFAULT			0		
#define JPEG_DEFAULT        0		
#define JPEG_FAST           0x0001	
#define JPEG_ACCURATE       0x0002	
#define JPEG_CMYK			0x0004	
#define JPEG_EXIFROTATE		0x0008	
#define JPEG_GREYSCALE		0x0010	
#define JPEG_QUALITYSUPERB  0x80	
#define JPEG_QUALITYGOOD    0x0100	
#define JPEG_QUALITYNORMAL  0x0200	
#define JPEG_QUALITYAVERAGE 0x0400	
#define JPEG_QUALITYBAD     0x0800	
#define JPEG_PROGRESSIVE	0x2000	
#define JPEG_SUBSAMPLING_411 0x1000		
#define JPEG_SUBSAMPLING_420 0x4000		
#define JPEG_SUBSAMPLING_422 0x8000		
#define JPEG_SUBSAMPLING_444 0x10000	
#define JPEG_OPTIMIZE		0x20000		
#define JPEG_BASELINE		0x40000		
#define KOALA_DEFAULT       0
#define LBM_DEFAULT         0
#define MNG_DEFAULT         0
#define PCD_DEFAULT         0
#define PCD_BASE            1		
#define PCD_BASEDIV4        2		
#define PCD_BASEDIV16       3		
#define PCX_DEFAULT         0
#define PFM_DEFAULT         0
#define PICT_DEFAULT        0
#define PNG_DEFAULT         0
#define PNG_IGNOREGAMMA		1		
#define PNG_Z_BEST_SPEED			0x0001	
#define PNG_Z_DEFAULT_COMPRESSION	0x0006	
#define PNG_Z_BEST_COMPRESSION		0x0009	
#define PNG_Z_NO_COMPRESSION		0x0100	
#define PNG_INTERLACED				0x0200	
#define PNM_DEFAULT         0
#define PNM_SAVE_RAW        0       
#define PNM_SAVE_ASCII      1       
#define PSD_DEFAULT         0
#define PSD_CMYK			1		
#define PSD_LAB				2		
#define PSD_NONE			0x0100	
#define PSD_RLE				0x0200	
#define PSD_PSB             0x2000  
#define RAS_DEFAULT         0
#define RAW_DEFAULT         0		
#define RAW_PREVIEW			1		
#define RAW_DISPLAY			2		
#define RAW_HALFSIZE		4		
#define RAW_UNPROCESSED		8		
#define SGI_DEFAULT			0
#define TARGA_DEFAULT       0
#define TARGA_LOAD_RGB888   1       
#define TARGA_SAVE_RLE		2		
#define TIFF_DEFAULT        0
#define TIFF_CMYK			0x0001	
#define TIFF_PACKBITS       0x0100  
#define TIFF_DEFLATE        0x0200  
#define TIFF_ADOBE_DEFLATE  0x0400  
#define TIFF_NONE           0x0800  
#define TIFF_CCITTFAX3		0x1000  
#define TIFF_CCITTFAX4		0x2000  
#define TIFF_LZW			0x4000	
#define TIFF_JPEG			0x8000	
#define TIFF_LOGLUV			0x10000	
#define WBMP_DEFAULT        0
#define XBM_DEFAULT			0
#define XPM_DEFAULT			0
#define WEBP_DEFAULT		0		
#define WEBP_LOSSLESS		0x100	
#define JXR_DEFAULT			0		
#define JXR_LOSSLESS		0x0064	
#define JXR_PROGRESSIVE		0x2000	
#define FI_COLOR_IS_RGB_COLOR			0x00	
#define FI_COLOR_IS_RGBA_COLOR			0x01	
#define FI_COLOR_FIND_EQUAL_COLOR		0x02	
#define FI_COLOR_ALPHA_IS_INDEX			0x04	
#define FI_COLOR_PALETTE_SEARCH_MASK	(FI_COLOR_FIND_EQUAL_COLOR | FI_COLOR_ALPHA_IS_INDEX)	
#define FI_RESCALE_DEFAULT			0x00    
#define FI_RESCALE_TRUE_COLOR		0x01	
#define FI_RESCALE_OMIT_METADATA	0x02	
#ifdef __cplusplus
extern "C" {
#endif
DLL_API void DLL_CALLCONV FreeImage_Initialise(BOOL load_local_plugins_only FI_DEFAULT(FALSE));
DLL_API void DLL_CALLCONV FreeImage_DeInitialise(void);
DLL_API const char *DLL_CALLCONV FreeImage_GetVersion(void);
DLL_API const char *DLL_CALLCONV FreeImage_GetCopyrightMessage(void);
typedef void (*FreeImage_OutputMessageFunction)(FREE_IMAGE_FORMAT fif, const char *msg);
typedef void (DLL_CALLCONV *FreeImage_OutputMessageFunctionStdCall)(FREE_IMAGE_FORMAT fif, const char *msg); 
DLL_API void DLL_CALLCONV FreeImage_SetOutputMessageStdCall(FreeImage_OutputMessageFunctionStdCall omf); 
DLL_API void DLL_CALLCONV FreeImage_SetOutputMessage(FreeImage_OutputMessageFunction omf);
DLL_API void DLL_CALLCONV FreeImage_OutputMessageProc(int fif, const char *fmt, ...);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Allocate(int width, int height, int bpp, unsigned red_mask FI_DEFAULT(0), unsigned green_mask FI_DEFAULT(0), unsigned blue_mask FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_AllocateT(FREE_IMAGE_TYPE type, int width, int height, int bpp FI_DEFAULT(8), unsigned red_mask FI_DEFAULT(0), unsigned green_mask FI_DEFAULT(0), unsigned blue_mask FI_DEFAULT(0));
DLL_API FIBITMAP * DLL_CALLCONV FreeImage_Clone(FIBITMAP *dib);
DLL_API void DLL_CALLCONV FreeImage_Unload(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_HasPixels(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Load(FREE_IMAGE_FORMAT fif, const char *filename, int flags FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_LoadU(FREE_IMAGE_FORMAT fif, const wchar_t *filename, int flags FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_LoadFromHandle(FREE_IMAGE_FORMAT fif, FreeImageIO *io, fi_handle handle, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_Save(FREE_IMAGE_FORMAT fif, FIBITMAP *dib, const char *filename, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_SaveU(FREE_IMAGE_FORMAT fif, FIBITMAP *dib, const wchar_t *filename, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_SaveToHandle(FREE_IMAGE_FORMAT fif, FIBITMAP *dib, FreeImageIO *io, fi_handle handle, int flags FI_DEFAULT(0));
DLL_API FIMEMORY *DLL_CALLCONV FreeImage_OpenMemory(BYTE *data FI_DEFAULT(0), DWORD size_in_bytes FI_DEFAULT(0));
DLL_API void DLL_CALLCONV FreeImage_CloseMemory(FIMEMORY *stream);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_LoadFromMemory(FREE_IMAGE_FORMAT fif, FIMEMORY *stream, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_SaveToMemory(FREE_IMAGE_FORMAT fif, FIBITMAP *dib, FIMEMORY *stream, int flags FI_DEFAULT(0));
DLL_API long DLL_CALLCONV FreeImage_TellMemory(FIMEMORY *stream);
DLL_API BOOL DLL_CALLCONV FreeImage_SeekMemory(FIMEMORY *stream, long offset, int origin);
DLL_API BOOL DLL_CALLCONV FreeImage_AcquireMemory(FIMEMORY *stream, BYTE **data, DWORD *size_in_bytes);
DLL_API unsigned DLL_CALLCONV FreeImage_ReadMemory(void *buffer, unsigned size, unsigned count, FIMEMORY *stream);
DLL_API unsigned DLL_CALLCONV FreeImage_WriteMemory(const void *buffer, unsigned size, unsigned count, FIMEMORY *stream);
DLL_API FIMULTIBITMAP *DLL_CALLCONV FreeImage_LoadMultiBitmapFromMemory(FREE_IMAGE_FORMAT fif, FIMEMORY *stream, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_SaveMultiBitmapToMemory(FREE_IMAGE_FORMAT fif, FIMULTIBITMAP *bitmap, FIMEMORY *stream, int flags);
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_RegisterLocalPlugin(FI_InitProc proc_address, const char *format FI_DEFAULT(0), const char *description FI_DEFAULT(0), const char *extension FI_DEFAULT(0), const char *regexpr FI_DEFAULT(0));
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_RegisterExternalPlugin(const char *path, const char *format FI_DEFAULT(0), const char *description FI_DEFAULT(0), const char *extension FI_DEFAULT(0), const char *regexpr FI_DEFAULT(0));
DLL_API int DLL_CALLCONV FreeImage_GetFIFCount(void);
DLL_API int DLL_CALLCONV FreeImage_SetPluginEnabled(FREE_IMAGE_FORMAT fif, BOOL enable);
DLL_API int DLL_CALLCONV FreeImage_IsPluginEnabled(FREE_IMAGE_FORMAT fif);
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFIFFromFormat(const char *format);
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFIFFromMime(const char *mime);
DLL_API const char *DLL_CALLCONV FreeImage_GetFormatFromFIF(FREE_IMAGE_FORMAT fif);
DLL_API const char *DLL_CALLCONV FreeImage_GetFIFExtensionList(FREE_IMAGE_FORMAT fif);
DLL_API const char *DLL_CALLCONV FreeImage_GetFIFDescription(FREE_IMAGE_FORMAT fif);
DLL_API const char *DLL_CALLCONV FreeImage_GetFIFRegExpr(FREE_IMAGE_FORMAT fif);
DLL_API const char *DLL_CALLCONV FreeImage_GetFIFMimeType(FREE_IMAGE_FORMAT fif);
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFIFFromFilename(const char *filename);
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFIFFromFilenameU(const wchar_t *filename);
DLL_API BOOL DLL_CALLCONV FreeImage_FIFSupportsReading(FREE_IMAGE_FORMAT fif);
DLL_API BOOL DLL_CALLCONV FreeImage_FIFSupportsWriting(FREE_IMAGE_FORMAT fif);
DLL_API BOOL DLL_CALLCONV FreeImage_FIFSupportsExportBPP(FREE_IMAGE_FORMAT fif, int bpp);
DLL_API BOOL DLL_CALLCONV FreeImage_FIFSupportsExportType(FREE_IMAGE_FORMAT fif, FREE_IMAGE_TYPE type);
DLL_API BOOL DLL_CALLCONV FreeImage_FIFSupportsICCProfiles(FREE_IMAGE_FORMAT fif);
DLL_API BOOL DLL_CALLCONV FreeImage_FIFSupportsNoPixels(FREE_IMAGE_FORMAT fif);
DLL_API FIMULTIBITMAP * DLL_CALLCONV FreeImage_OpenMultiBitmap(FREE_IMAGE_FORMAT fif, const char *filename, BOOL create_new, BOOL read_only, BOOL keep_cache_in_memory FI_DEFAULT(FALSE), int flags FI_DEFAULT(0));
DLL_API FIMULTIBITMAP * DLL_CALLCONV FreeImage_OpenMultiBitmapFromHandle(FREE_IMAGE_FORMAT fif, FreeImageIO *io, fi_handle handle, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_SaveMultiBitmapToHandle(FREE_IMAGE_FORMAT fif, FIMULTIBITMAP *bitmap, FreeImageIO *io, fi_handle handle, int flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_CloseMultiBitmap(FIMULTIBITMAP *bitmap, int flags FI_DEFAULT(0));
DLL_API int DLL_CALLCONV FreeImage_GetPageCount(FIMULTIBITMAP *bitmap);
DLL_API void DLL_CALLCONV FreeImage_AppendPage(FIMULTIBITMAP *bitmap, FIBITMAP *data);
DLL_API void DLL_CALLCONV FreeImage_InsertPage(FIMULTIBITMAP *bitmap, int page, FIBITMAP *data);
DLL_API void DLL_CALLCONV FreeImage_DeletePage(FIMULTIBITMAP *bitmap, int page);
DLL_API FIBITMAP * DLL_CALLCONV FreeImage_LockPage(FIMULTIBITMAP *bitmap, int page);
DLL_API void DLL_CALLCONV FreeImage_UnlockPage(FIMULTIBITMAP *bitmap, FIBITMAP *data, BOOL changed);
DLL_API BOOL DLL_CALLCONV FreeImage_MovePage(FIMULTIBITMAP *bitmap, int target, int source);
DLL_API BOOL DLL_CALLCONV FreeImage_GetLockedPageNumbers(FIMULTIBITMAP *bitmap, int *pages, int *count);
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFileType(const char *filename, int size FI_DEFAULT(0));
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFileTypeU(const wchar_t *filename, int size FI_DEFAULT(0));
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFileTypeFromHandle(FreeImageIO *io, fi_handle handle, int size FI_DEFAULT(0));
DLL_API FREE_IMAGE_FORMAT DLL_CALLCONV FreeImage_GetFileTypeFromMemory(FIMEMORY *stream, int size FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_Validate(FREE_IMAGE_FORMAT fif, const char *filename);
DLL_API BOOL DLL_CALLCONV FreeImage_ValidateU(FREE_IMAGE_FORMAT fif, const wchar_t *filename);
DLL_API BOOL DLL_CALLCONV FreeImage_ValidateFromHandle(FREE_IMAGE_FORMAT fif, FreeImageIO *io, fi_handle handle);
DLL_API BOOL DLL_CALLCONV FreeImage_ValidateFromMemory(FREE_IMAGE_FORMAT fif, FIMEMORY *stream);
DLL_API FREE_IMAGE_TYPE DLL_CALLCONV FreeImage_GetImageType(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_IsLittleEndian(void);
DLL_API BOOL DLL_CALLCONV FreeImage_LookupX11Color(const char *szColor, BYTE *nRed, BYTE *nGreen, BYTE *nBlue);
DLL_API BOOL DLL_CALLCONV FreeImage_LookupSVGColor(const char *szColor, BYTE *nRed, BYTE *nGreen, BYTE *nBlue);
DLL_API BYTE *DLL_CALLCONV FreeImage_GetBits(FIBITMAP *dib);
DLL_API BYTE *DLL_CALLCONV FreeImage_GetScanLine(FIBITMAP *dib, int scanline);
DLL_API BOOL DLL_CALLCONV FreeImage_GetPixelIndex(FIBITMAP *dib, unsigned x, unsigned y, BYTE *value);
DLL_API BOOL DLL_CALLCONV FreeImage_GetPixelColor(FIBITMAP *dib, unsigned x, unsigned y, RGBQUAD *value);
DLL_API BOOL DLL_CALLCONV FreeImage_SetPixelIndex(FIBITMAP *dib, unsigned x, unsigned y, BYTE *value);
DLL_API BOOL DLL_CALLCONV FreeImage_SetPixelColor(FIBITMAP *dib, unsigned x, unsigned y, RGBQUAD *value);
DLL_API unsigned DLL_CALLCONV FreeImage_GetColorsUsed(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetBPP(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetWidth(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetHeight(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetLine(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetPitch(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetDIBSize(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetMemorySize(FIBITMAP *dib);
DLL_API RGBQUAD *DLL_CALLCONV FreeImage_GetPalette(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetDotsPerMeterX(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetDotsPerMeterY(FIBITMAP *dib);
DLL_API void DLL_CALLCONV FreeImage_SetDotsPerMeterX(FIBITMAP *dib, unsigned res);
DLL_API void DLL_CALLCONV FreeImage_SetDotsPerMeterY(FIBITMAP *dib, unsigned res);
DLL_API BITMAPINFOHEADER *DLL_CALLCONV FreeImage_GetInfoHeader(FIBITMAP *dib);
DLL_API BITMAPINFO *DLL_CALLCONV FreeImage_GetInfo(FIBITMAP *dib);
DLL_API FREE_IMAGE_COLOR_TYPE DLL_CALLCONV FreeImage_GetColorType(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetRedMask(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetGreenMask(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetBlueMask(FIBITMAP *dib);
DLL_API unsigned DLL_CALLCONV FreeImage_GetTransparencyCount(FIBITMAP *dib);
DLL_API BYTE * DLL_CALLCONV FreeImage_GetTransparencyTable(FIBITMAP *dib);
DLL_API void DLL_CALLCONV FreeImage_SetTransparent(FIBITMAP *dib, BOOL enabled);
DLL_API void DLL_CALLCONV FreeImage_SetTransparencyTable(FIBITMAP *dib, BYTE *table, int count);
DLL_API BOOL DLL_CALLCONV FreeImage_IsTransparent(FIBITMAP *dib);
DLL_API void DLL_CALLCONV FreeImage_SetTransparentIndex(FIBITMAP *dib, int index);
DLL_API int DLL_CALLCONV FreeImage_GetTransparentIndex(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_HasBackgroundColor(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_GetBackgroundColor(FIBITMAP *dib, RGBQUAD *bkcolor);
DLL_API BOOL DLL_CALLCONV FreeImage_SetBackgroundColor(FIBITMAP *dib, RGBQUAD *bkcolor);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_GetThumbnail(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_SetThumbnail(FIBITMAP *dib, FIBITMAP *thumbnail);
DLL_API FIICCPROFILE *DLL_CALLCONV FreeImage_GetICCProfile(FIBITMAP *dib);
DLL_API FIICCPROFILE *DLL_CALLCONV FreeImage_CreateICCProfile(FIBITMAP *dib, void *data, long size);
DLL_API void DLL_CALLCONV FreeImage_DestroyICCProfile(FIBITMAP *dib);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To4(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine8To4(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To4_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To4_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine24To4(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine32To4(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To8(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine4To8(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To8_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To8_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine24To8(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine32To8(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To16_555(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine4To16_555(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine8To16_555(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16_565_To16_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine24To16_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine32To16_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To16_565(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine4To16_565(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine8To16_565(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16_555_To16_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine24To16_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine32To16_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To24(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine4To24(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine8To24(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To24_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To24_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine32To24(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To32(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine1To32MapTransparency(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette, BYTE *table, int transparent_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine4To32(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine4To32MapTransparency(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette, BYTE *table, int transparent_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine8To32(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine8To32MapTransparency(BYTE *target, BYTE *source, int width_in_pixels, RGBQUAD *palette, BYTE *table, int transparent_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To32_555(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine16To32_565(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API void DLL_CALLCONV FreeImage_ConvertLine24To32(BYTE *target, BYTE *source, int width_in_pixels);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo4Bits(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo8Bits(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToGreyscale(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo16Bits555(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo16Bits565(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo24Bits(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertTo32Bits(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ColorQuantize(FIBITMAP *dib, FREE_IMAGE_QUANTIZE quantize);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ColorQuantizeEx(FIBITMAP *dib, FREE_IMAGE_QUANTIZE quantize FI_DEFAULT(FIQ_WUQUANT), int PaletteSize FI_DEFAULT(256), int ReserveSize FI_DEFAULT(0), RGBQUAD *ReservePalette FI_DEFAULT(NULL));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Threshold(FIBITMAP *dib, BYTE T);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Dither(FIBITMAP *dib, FREE_IMAGE_DITHER algorithm);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertFromRawBits(BYTE *bits, int width, int height, int pitch, unsigned bpp, unsigned red_mask, unsigned green_mask, unsigned blue_mask, BOOL topdown FI_DEFAULT(FALSE));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertFromRawBitsEx(BOOL copySource, BYTE *bits, FREE_IMAGE_TYPE type, int width, int height, int pitch, unsigned bpp, unsigned red_mask, unsigned green_mask, unsigned blue_mask, BOOL topdown FI_DEFAULT(FALSE));
DLL_API void DLL_CALLCONV FreeImage_ConvertToRawBits(BYTE *bits, FIBITMAP *dib, int pitch, unsigned bpp, unsigned red_mask, unsigned green_mask, unsigned blue_mask, BOOL topdown FI_DEFAULT(FALSE));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToFloat(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToRGBF(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToRGBAF(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToUINT16(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToRGB16(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToRGBA16(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToStandardType(FIBITMAP *src, BOOL scale_linear FI_DEFAULT(TRUE));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ConvertToType(FIBITMAP *src, FREE_IMAGE_TYPE dst_type, BOOL scale_linear FI_DEFAULT(TRUE));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_ToneMapping(FIBITMAP *dib, FREE_IMAGE_TMO tmo, double first_param FI_DEFAULT(0), double second_param FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_TmoDrago03(FIBITMAP *src, double gamma FI_DEFAULT(2.2), double exposure FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_TmoReinhard05(FIBITMAP *src, double intensity FI_DEFAULT(0), double contrast FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_TmoReinhard05Ex(FIBITMAP *src, double intensity FI_DEFAULT(0), double contrast FI_DEFAULT(0), double adaptation FI_DEFAULT(1), double color_correction FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_TmoFattal02(FIBITMAP *src, double color_saturation FI_DEFAULT(0.5), double attenuation FI_DEFAULT(0.85));
DLL_API DWORD DLL_CALLCONV FreeImage_ZLibCompress(BYTE *target, DWORD target_size, BYTE *source, DWORD source_size);
DLL_API DWORD DLL_CALLCONV FreeImage_ZLibUncompress(BYTE *target, DWORD target_size, BYTE *source, DWORD source_size);
DLL_API DWORD DLL_CALLCONV FreeImage_ZLibGZip(BYTE *target, DWORD target_size, BYTE *source, DWORD source_size);
DLL_API DWORD DLL_CALLCONV FreeImage_ZLibGUnzip(BYTE *target, DWORD target_size, BYTE *source, DWORD source_size);
DLL_API DWORD DLL_CALLCONV FreeImage_ZLibCRC32(DWORD crc, BYTE *source, DWORD source_size);
DLL_API FITAG *DLL_CALLCONV FreeImage_CreateTag(void);
DLL_API void DLL_CALLCONV FreeImage_DeleteTag(FITAG *tag);
DLL_API FITAG *DLL_CALLCONV FreeImage_CloneTag(FITAG *tag);
DLL_API const char *DLL_CALLCONV FreeImage_GetTagKey(FITAG *tag);
DLL_API const char *DLL_CALLCONV FreeImage_GetTagDescription(FITAG *tag);
DLL_API WORD DLL_CALLCONV FreeImage_GetTagID(FITAG *tag);
DLL_API FREE_IMAGE_MDTYPE DLL_CALLCONV FreeImage_GetTagType(FITAG *tag);
DLL_API DWORD DLL_CALLCONV FreeImage_GetTagCount(FITAG *tag);
DLL_API DWORD DLL_CALLCONV FreeImage_GetTagLength(FITAG *tag);
DLL_API const void *DLL_CALLCONV FreeImage_GetTagValue(FITAG *tag);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagKey(FITAG *tag, const char *key);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagDescription(FITAG *tag, const char *description);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagID(FITAG *tag, WORD id);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagType(FITAG *tag, FREE_IMAGE_MDTYPE type);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagCount(FITAG *tag, DWORD count);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagLength(FITAG *tag, DWORD length);
DLL_API BOOL DLL_CALLCONV FreeImage_SetTagValue(FITAG *tag, const void *value);
DLL_API FIMETADATA *DLL_CALLCONV FreeImage_FindFirstMetadata(FREE_IMAGE_MDMODEL model, FIBITMAP *dib, FITAG **tag);
DLL_API BOOL DLL_CALLCONV FreeImage_FindNextMetadata(FIMETADATA *mdhandle, FITAG **tag);
DLL_API void DLL_CALLCONV FreeImage_FindCloseMetadata(FIMETADATA *mdhandle);
DLL_API BOOL DLL_CALLCONV FreeImage_SetMetadata(FREE_IMAGE_MDMODEL model, FIBITMAP *dib, const char *key, FITAG *tag);
DLL_API BOOL DLL_CALLCONV FreeImage_GetMetadata(FREE_IMAGE_MDMODEL model, FIBITMAP *dib, const char *key, FITAG **tag);
DLL_API BOOL DLL_CALLCONV FreeImage_SetMetadataKeyValue(FREE_IMAGE_MDMODEL model, FIBITMAP *dib, const char *key, const char *value);
DLL_API unsigned DLL_CALLCONV FreeImage_GetMetadataCount(FREE_IMAGE_MDMODEL model, FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_CloneMetadata(FIBITMAP *dst, FIBITMAP *src);
DLL_API const char* DLL_CALLCONV FreeImage_TagToString(FREE_IMAGE_MDMODEL model, FITAG *tag, char *Make FI_DEFAULT(NULL));
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGTransform(const char *src_file, const char *dst_file, FREE_IMAGE_JPEG_OPERATION operation, BOOL perfect FI_DEFAULT(TRUE));
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGTransformU(const wchar_t *src_file, const wchar_t *dst_file, FREE_IMAGE_JPEG_OPERATION operation, BOOL perfect FI_DEFAULT(TRUE));
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGCrop(const char *src_file, const char *dst_file, int left, int top, int right, int bottom);
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGCropU(const wchar_t *src_file, const wchar_t *dst_file, int left, int top, int right, int bottom);
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGTransformFromHandle(FreeImageIO* src_io, fi_handle src_handle, FreeImageIO* dst_io, fi_handle dst_handle, FREE_IMAGE_JPEG_OPERATION operation, int* left, int* top, int* right, int* bottom, BOOL perfect FI_DEFAULT(TRUE));
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGTransformCombined(const char *src_file, const char *dst_file, FREE_IMAGE_JPEG_OPERATION operation, int* left, int* top, int* right, int* bottom, BOOL perfect FI_DEFAULT(TRUE));
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGTransformCombinedU(const wchar_t *src_file, const wchar_t *dst_file, FREE_IMAGE_JPEG_OPERATION operation, int* left, int* top, int* right, int* bottom, BOOL perfect FI_DEFAULT(TRUE));
DLL_API BOOL DLL_CALLCONV FreeImage_JPEGTransformCombinedFromMemory(FIMEMORY* src_stream, FIMEMORY* dst_stream, FREE_IMAGE_JPEG_OPERATION operation, int* left, int* top, int* right, int* bottom, BOOL perfect FI_DEFAULT(TRUE));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Rotate(FIBITMAP *dib, double angle, const void *bkcolor FI_DEFAULT(NULL));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_RotateEx(FIBITMAP *dib, double angle, double x_shift, double y_shift, double x_origin, double y_origin, BOOL use_mask);
DLL_API BOOL DLL_CALLCONV FreeImage_FlipHorizontal(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_FlipVertical(FIBITMAP *dib);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Rescale(FIBITMAP *dib, int dst_width, int dst_height, FREE_IMAGE_FILTER filter FI_DEFAULT(FILTER_CATMULLROM));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_MakeThumbnail(FIBITMAP *dib, int max_pixel_size, BOOL convert FI_DEFAULT(TRUE));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_RescaleRect(FIBITMAP *dib, int dst_width, int dst_height, int left, int top, int right, int bottom, FREE_IMAGE_FILTER filter FI_DEFAULT(FILTER_CATMULLROM), unsigned flags FI_DEFAULT(0));
DLL_API BOOL DLL_CALLCONV FreeImage_AdjustCurve(FIBITMAP *dib, BYTE *LUT, FREE_IMAGE_COLOR_CHANNEL channel);
DLL_API BOOL DLL_CALLCONV FreeImage_AdjustGamma(FIBITMAP *dib, double gamma);
DLL_API BOOL DLL_CALLCONV FreeImage_AdjustBrightness(FIBITMAP *dib, double percentage);
DLL_API BOOL DLL_CALLCONV FreeImage_AdjustContrast(FIBITMAP *dib, double percentage);
DLL_API BOOL DLL_CALLCONV FreeImage_Invert(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_GetHistogram(FIBITMAP *dib, DWORD *histo, FREE_IMAGE_COLOR_CHANNEL channel FI_DEFAULT(FICC_BLACK));
DLL_API int DLL_CALLCONV FreeImage_GetAdjustColorsLookupTable(BYTE *LUT, double brightness, double contrast, double gamma, BOOL invert);
DLL_API BOOL DLL_CALLCONV FreeImage_AdjustColors(FIBITMAP *dib, double brightness, double contrast, double gamma, BOOL invert FI_DEFAULT(FALSE));
DLL_API unsigned DLL_CALLCONV FreeImage_ApplyColorMapping(FIBITMAP *dib, RGBQUAD *srccolors, RGBQUAD *dstcolors, unsigned count, BOOL ignore_alpha, BOOL swap);
DLL_API unsigned DLL_CALLCONV FreeImage_SwapColors(FIBITMAP *dib, RGBQUAD *color_a, RGBQUAD *color_b, BOOL ignore_alpha);
DLL_API unsigned DLL_CALLCONV FreeImage_ApplyPaletteIndexMapping(FIBITMAP *dib, BYTE *srcindices,	BYTE *dstindices, unsigned count, BOOL swap);
DLL_API unsigned DLL_CALLCONV FreeImage_SwapPaletteIndices(FIBITMAP *dib, BYTE *index_a, BYTE *index_b);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_GetChannel(FIBITMAP *dib, FREE_IMAGE_COLOR_CHANNEL channel);
DLL_API BOOL DLL_CALLCONV FreeImage_SetChannel(FIBITMAP *dst, FIBITMAP *src, FREE_IMAGE_COLOR_CHANNEL channel);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_GetComplexChannel(FIBITMAP *src, FREE_IMAGE_COLOR_CHANNEL channel);
DLL_API BOOL DLL_CALLCONV FreeImage_SetComplexChannel(FIBITMAP *dst, FIBITMAP *src, FREE_IMAGE_COLOR_CHANNEL channel);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Copy(FIBITMAP *dib, int left, int top, int right, int bottom);
DLL_API BOOL DLL_CALLCONV FreeImage_Paste(FIBITMAP *dst, FIBITMAP *src, int left, int top, int alpha);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_CreateView(FIBITMAP *dib, unsigned left, unsigned top, unsigned right, unsigned bottom);
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_Composite(FIBITMAP *fg, BOOL useFileBkg FI_DEFAULT(FALSE), RGBQUAD *appBkColor FI_DEFAULT(NULL), FIBITMAP *bg FI_DEFAULT(NULL));
DLL_API BOOL DLL_CALLCONV FreeImage_PreMultiplyWithAlpha(FIBITMAP *dib);
DLL_API BOOL DLL_CALLCONV FreeImage_FillBackground(FIBITMAP *dib, const void *color, int options FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_EnlargeCanvas(FIBITMAP *src, int left, int top, int right, int bottom, const void *color, int options FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_AllocateEx(int width, int height, int bpp, const RGBQUAD *color, int options FI_DEFAULT(0), const RGBQUAD *palette FI_DEFAULT(NULL), unsigned red_mask FI_DEFAULT(0), unsigned green_mask FI_DEFAULT(0), unsigned blue_mask FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_AllocateExT(FREE_IMAGE_TYPE type, int width, int height, int bpp, const void *color, int options FI_DEFAULT(0), const RGBQUAD *palette FI_DEFAULT(NULL), unsigned red_mask FI_DEFAULT(0), unsigned green_mask FI_DEFAULT(0), unsigned blue_mask FI_DEFAULT(0));
DLL_API FIBITMAP *DLL_CALLCONV FreeImage_MultigridPoissonSolver(FIBITMAP *Laplacian, int ncycle FI_DEFAULT(3));
#if defined(__BORLANDC__)
#pragma option pop
#endif
#ifdef __cplusplus
}
#endif
#endif 
