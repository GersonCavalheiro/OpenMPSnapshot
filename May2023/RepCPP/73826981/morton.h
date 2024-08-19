#pragma once


#include "morton2D.h"
#include "morton3D.h"



#define morton2D_32_encode m2D_e_sLUT<uint_fast32_t, uint_fast16_t>
#define morton2D_64_encode m2D_e_sLUT<uint_fast64_t, uint_fast32_t>
#define morton2D_32_decode m2D_d_sLUT<uint_fast32_t, uint_fast16_t>
#define morton2D_64_decode m2D_d_sLUT<uint_fast64_t, uint_fast32_t>

#define morton3D_32_encode m3D_e_sLUT<uint_fast32_t, uint_fast16_t>
#define morton3D_64_encode m3D_e_sLUT<uint_fast64_t, uint_fast32_t>
#define morton3D_32_decode m3D_d_sLUT<uint_fast32_t, uint_fast16_t>
#define morton3D_64_decode m3D_d_sLUT<uint_fast64_t, uint_fast32_t>