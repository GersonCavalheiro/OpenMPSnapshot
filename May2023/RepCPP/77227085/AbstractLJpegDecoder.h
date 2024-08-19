

#pragma once

#include "codes/HuffmanCode.h"                  
#include "codes/PrefixCodeDecoder.h"            
#include "common/RawImage.h"                    
#include "common/RawspeedException.h"           
#include "decoders/RawDecoderException.h"       
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <algorithm>                            
#include <array>                                
#include <cstdint>                              
#include <memory>                               
#include <vector>                               



namespace rawspeed {

enum class JpegMarker { 
STUFF = 0x00,
SOF0 = 0xc0, 
SOF1 = 0xc1, 
SOF2 = 0xc2, 
SOF3 = 0xc3, 

SOF5 = 0xc5, 
SOF6 = 0xc6, 
SOF7 = 0xc7, 

JPG = 0xc8,   
SOF9 = 0xc9,  
SOF10 = 0xca, 
SOF11 = 0xcb, 

SOF13 = 0xcd, 
SOF14 = 0xce, 
SOF15 = 0xcf, 

DHT = 0xc4, 

DAC = 0xcc, 

RST0 = 0xd0, 
RST1 = 0xd1, 
RST2 = 0xd2, 
RST3 = 0xd3, 
RST4 = 0xd4, 
RST5 = 0xd5, 
RST6 = 0xd6, 
RST7 = 0xd7, 

SOI = 0xd8, 
EOI = 0xd9, 
SOS = 0xda, 
DQT =
0xdb,   
DNL = 0xdc, 
DRI = 0xdd, 
DHP = 0xde, 
EXP =
0xdf, 

APP0 =
0xe0,     
APP1 = 0xe1,  
APP2 = 0xe2,  
APP3 = 0xe3,  
APP4 = 0xe4,  
APP5 = 0xe5,  
APP6 = 0xe6,  
APP7 = 0xe7,  
APP8 = 0xe8,  
APP9 = 0xe9,  
APP10 = 0xea, 
APP11 = 0xeb, 
APP12 = 0xec, 
APP13 = 0xed, 
APP14 =
0xee,     
APP15 = 0xef, 

JPG0 = 0xf0,  
JPG13 = 0xfd, 
COM = 0xfe,   

TEM = 0x01, 
FILL = 0xFF

};


struct JpegComponentInfo {

uint32_t componentId = ~0U; 


uint32_t dcTblNo = ~0U;
uint32_t superH = ~0U; 
uint32_t superV = ~0U; 
};

class SOFInfo {
public:
std::array<JpegComponentInfo, 4> compInfo;
uint32_t w = 0;    
uint32_t h = 0;    
uint32_t cps = 0;  
uint32_t prec = 0; 
bool initialized = false;
};

class AbstractLJpegDecoder : public AbstractDecompressor {
std::vector<std::unique_ptr<const HuffmanCode<BaselineCodeTag>>>
huffmanCodeStore;
std::vector<std::unique_ptr<const PrefixCodeDecoder<>>>
PrefixCodeDecoderStore;

uint32_t Pt = 0;
std::array<const PrefixCodeDecoder<>*, 4> huff{
{}}; 

public:
AbstractLJpegDecoder(ByteStream bs, const RawImage& img);

virtual ~AbstractLJpegDecoder() = default;

protected:
bool fixDng16Bug = false; 
bool fullDecodeHT = true; 

void decodeSOI();
void parseSOF(ByteStream data, SOFInfo* i);
void parseSOS(ByteStream data);
void parseDHT(ByteStream data);
static void parseDRI(ByteStream dri);
JpegMarker getNextMarker(bool allowskip);

[[nodiscard]] std::vector<const PrefixCodeDecoder<>*>
getPrefixCodeDecoders(int N_COMP) const {
std::vector<const PrefixCodeDecoder<>*> ht(N_COMP);
for (int i = 0; i < N_COMP; ++i) {
const unsigned dcTblNo = frame.compInfo[i].dcTblNo;
if (const unsigned dcTbls = huff.size(); dcTblNo >= dcTbls) {
ThrowRDE("Decoding table %u for comp %i does not exist (tables = %u)",
dcTblNo, i, dcTbls);
}
ht[i] = huff[dcTblNo];
}

return ht;
}

[[nodiscard]] std::vector<uint16_t> getInitialPredictors(int N_COMP) const {
std::vector<uint16_t> pred(N_COMP);
if (frame.prec < (Pt + 1)) {
ThrowRDE("Invalid precision (%u) and point transform (%u) combination!",
frame.prec, Pt);
}
std::fill(pred.begin(), pred.end(), 1 << (frame.prec - Pt - 1));
return pred;
}

virtual void decodeScan() = 0;

ByteStream input;
RawImage mRaw;

SOFInfo frame;
uint32_t predictorMode = 0;
};

} 
