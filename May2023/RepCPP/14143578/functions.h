
#pragma once


#define DG_DEVICE
#ifdef __CUDACC__
#undef DG_DEVICE
#define DG_DEVICE __host__ __device__
#endif

namespace dg{

DG_DEVICE static inline double one( double x) {return 1;}

DG_DEVICE static inline double one( double x, double y) {return 1;}

DG_DEVICE static inline double one( double x, double y, double z) {return 1;}

DG_DEVICE static inline double zero( double x) {return 0.;}

DG_DEVICE static inline double zero( double x, double y) {return 0.;}

DG_DEVICE static inline double zero( double x, double y, double z) {return 0.;}

DG_DEVICE static inline double cooX1d( double x) {return x;}
DG_DEVICE static inline double cooX2d( double x, double y) {return x;}
DG_DEVICE static inline double cooX3d( double x, double y, double z) {return x;}

DG_DEVICE static inline double cooY2d( double x, double y) {return y;}
DG_DEVICE static inline double cooY3d( double x, double y, double z) {return y;}
DG_DEVICE static inline double cooZ3d( double x, double y, double z) {return z;}


DG_DEVICE static inline double cooRZP2X( double R, double Z, double P){ return R*sin(P);}
DG_DEVICE static inline double cooRZP2Y( double R, double Z, double P){ return R*cos(P);}
DG_DEVICE static inline double cooRZP2Z( double R, double Z, double P){ return Z;}

DG_DEVICE static inline float one( float x) {return 1;}

DG_DEVICE static inline float one( float x, float y) {return 1;}

DG_DEVICE static inline float one( float x, float y, float z) {return 1;}

DG_DEVICE static inline float zero( float x) {return 0.;}

DG_DEVICE static inline float zero( float x, float y) {return 0.;}

DG_DEVICE static inline float zero( float x, float y, float z) {return 0.;}

} 
