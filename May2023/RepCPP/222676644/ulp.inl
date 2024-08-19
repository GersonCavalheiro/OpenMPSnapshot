
#include <cmath>
#include <cfloat>

#if(GLM_COMPILER & GLM_COMPILER_VC)
#	pragma warning(push)
#	pragma warning(disable : 4127)
#endif

typedef union
{
float value;

unsigned int word;
} ieee_float_shape_type;

typedef union
{
double value;
struct
{
glm::detail::int32 lsw;
glm::detail::int32 msw;
} parts;
} ieee_double_shape_type;

#define GLM_EXTRACT_WORDS(ix0,ix1,d)                                \
do {                                                            \
ieee_double_shape_type ew_u;                                  \
ew_u.value = (d);                                             \
(ix0) = ew_u.parts.msw;                                       \
(ix1) = ew_u.parts.lsw;                                       \
} while (0)

#define GLM_GET_FLOAT_WORD(i,d)                                     \
do {                                                            \
ieee_float_shape_type gf_u;                                   \
gf_u.value = (d);                                             \
(i) = gf_u.word;                                              \
} while (0)

#define GLM_SET_FLOAT_WORD(d,i)                                     \
do {                                                            \
ieee_float_shape_type sf_u;                                   \
sf_u.word = (i);                                              \
(d) = sf_u.value;                                             \
} while (0)

#define GLM_INSERT_WORDS(d,ix0,ix1)                                 \
do {                                                            \
ieee_double_shape_type iw_u;                                  \
iw_u.parts.msw = (ix0);                                       \
iw_u.parts.lsw = (ix1);                                       \
(d) = iw_u.value;                                             \
} while (0)

namespace glm{
namespace detail
{
GLM_FUNC_QUALIFIER float nextafterf(float x, float y)
{
volatile float t;
glm::detail::int32 hx, hy, ix, iy;

GLM_GET_FLOAT_WORD(hx, x);
GLM_GET_FLOAT_WORD(hy, y);
ix = hx&0x7fffffff;             
iy = hy&0x7fffffff;             

if((ix>0x7f800000) ||   
(iy>0x7f800000))     
return x+y;
if(x==y) return y;              
if(ix==0) {                             
GLM_SET_FLOAT_WORD(x,(hy&0x80000000)|1);
t = x*x;
if(t==x) return t; else return x;   
}
if(hx>=0) {                             
if(hx>hy) {                         
hx -= 1;
} else {                            
hx += 1;
}
} else {                                
if(hy>=0||hx>hy){                   
hx -= 1;
} else {                            
hx += 1;
}
}
hy = hx&0x7f800000;
if(hy>=0x7f800000) return x+x;  
if(hy<0x00800000) {             
t = x*x;
if(t!=x) {          
GLM_SET_FLOAT_WORD(y,hx);
return y;
}
}
GLM_SET_FLOAT_WORD(x,hx);
return x;
}

GLM_FUNC_QUALIFIER double nextafter(double x, double y)
{
volatile double t;
glm::detail::int32 hx, hy, ix, iy;
glm::detail::uint32 lx, ly;

GLM_EXTRACT_WORDS(hx, lx, x);
GLM_EXTRACT_WORDS(hy, ly, y);
ix = hx & 0x7fffffff;             
iy = hy & 0x7fffffff;             

if(((ix>=0x7ff00000)&&((ix-0x7ff00000)|lx)!=0) ||   
((iy>=0x7ff00000)&&((iy-0x7ff00000)|ly)!=0))     
return x+y;
if(x==y) return y;              
if((ix|lx)==0) {                        
GLM_INSERT_WORDS(x, hy & 0x80000000, 1);    
t = x*x;
if(t==x) return t; else return x;   
}
if(hx>=0) {                             
if(hx>hy||((hx==hy)&&(lx>ly))) {    
if(lx==0) hx -= 1;
lx -= 1;
} else {                            
lx += 1;
if(lx==0) hx += 1;
}
} else {                                
if(hy>=0||hx>hy||((hx==hy)&&(lx>ly))){
if(lx==0) hx -= 1;
lx -= 1;
} else {                            
lx += 1;
if(lx==0) hx += 1;
}
}
hy = hx&0x7ff00000;
if(hy>=0x7ff00000) return x+x;  
if(hy<0x00100000) {             
t = x*x;
if(t!=x) {          
GLM_INSERT_WORDS(y,hx,lx);
return y;
}
}
GLM_INSERT_WORDS(x,hx,lx);
return x;
}
}
}

#if(GLM_COMPILER & GLM_COMPILER_VC)
#	pragma warning(pop)
#endif

#if((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
#	define GLM_NEXT_AFTER_FLT(x, toward) glm::detail::nextafterf((x), (toward))
#	define GLM_NEXT_AFTER_DBL(x, toward) _nextafter((x), (toward))
#else
#	define GLM_NEXT_AFTER_FLT(x, toward) nextafterf((x), (toward))
#	define GLM_NEXT_AFTER_DBL(x, toward) nextafter((x), (toward))
#endif

namespace glm
{
GLM_FUNC_QUALIFIER float next_float(float const & x)
{
return GLM_NEXT_AFTER_FLT(x, std::numeric_limits<float>::max());
}

GLM_FUNC_QUALIFIER double next_float(double const & x)
{
return GLM_NEXT_AFTER_DBL(x, std::numeric_limits<double>::max());
}

template<typename T, template<typename> class vecType>
GLM_FUNC_QUALIFIER vecType<T> next_float(vecType<T> const & x)
{
vecType<T> Result;
for(std::size_t i = 0; i < Result.length(); ++i)
Result[i] = next_float(x[i]);
return Result;
}

GLM_FUNC_QUALIFIER float prev_float(float const & x)
{
return GLM_NEXT_AFTER_FLT(x, std::numeric_limits<float>::min());
}

GLM_FUNC_QUALIFIER double prev_float(double const & x)
{
return GLM_NEXT_AFTER_DBL(x, std::numeric_limits<double>::min());
}

template<typename T, template<typename> class vecType>
GLM_FUNC_QUALIFIER vecType<T> prev_float(vecType<T> const & x)
{
vecType<T> Result;
for(std::size_t i = 0; i < Result.length(); ++i)
Result[i] = prev_float(x[i]);
return Result;
}

template <typename T>
GLM_FUNC_QUALIFIER T next_float(T const & x, uint const & ulps)
{
T temp = x;
for(std::size_t i = 0; i < ulps; ++i)
temp = next_float(temp);
return temp;
}

template<typename T, template<typename> class vecType>
GLM_FUNC_QUALIFIER vecType<T> next_float(vecType<T> const & x, vecType<uint> const & ulps)
{
vecType<T> Result;
for(std::size_t i = 0; i < Result.length(); ++i)
Result[i] = next_float(x[i], ulps[i]);
return Result;
}

template <typename T>
GLM_FUNC_QUALIFIER T prev_float(T const & x, uint const & ulps)
{
T temp = x;
for(std::size_t i = 0; i < ulps; ++i)
temp = prev_float(temp);
return temp;
}

template<typename T, template<typename> class vecType>
GLM_FUNC_QUALIFIER vecType<T> prev_float(vecType<T> const & x, vecType<uint> const & ulps)
{
vecType<T> Result;
for(std::size_t i = 0; i < Result.length(); ++i)
Result[i] = prev_float(x[i], ulps[i]);
return Result;
}

template <typename T>
GLM_FUNC_QUALIFIER uint float_distance(T const & x, T const & y)
{
uint ulp = 0;

if(x < y)
{
T temp = x;
while(temp != y && ulp < std::numeric_limits<std::size_t>::max())
{
++ulp;
temp = next_float(temp);
}
}
else if(y < x)
{
T temp = y;
while(temp != x && ulp < std::numeric_limits<std::size_t>::max())
{
++ulp;
temp = next_float(temp);
}
}
else 
{

}

return ulp;
}

template<typename T, template<typename> class vecType>
GLM_FUNC_QUALIFIER vecType<uint> float_distance(vecType<T> const & x, vecType<T> const & y)
{
vecType<uint> Result;
for(std::size_t i = 0; i < Result.length(); ++i)
Result[i] = float_distance(x[i], y[i]);
return Result;
}
}
