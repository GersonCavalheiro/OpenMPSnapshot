

#define SWIGPYTHON
#define SWIG_PYTHON_DIRECTOR_NO_VTABLE


#ifdef __cplusplus

template<typename T> class SwigValueWrapper {
struct SwigMovePointer {
T *ptr;
SwigMovePointer(T *p) : ptr(p) { }
~SwigMovePointer() { delete ptr; }
SwigMovePointer& operator=(SwigMovePointer& rhs) { T* oldptr = ptr; ptr = 0; delete oldptr; ptr = rhs.ptr; rhs.ptr = 0; return *this; }
} pointer;
SwigValueWrapper& operator=(const SwigValueWrapper<T>& rhs);
SwigValueWrapper(const SwigValueWrapper<T>& rhs);
public:
SwigValueWrapper() : pointer(0) { }
SwigValueWrapper& operator=(const T& t) { SwigMovePointer tmp(new T(t)); pointer = tmp; return *this; }
operator T&() const { return *pointer.ptr; }
T *operator&() { return pointer.ptr; }
};

template <typename T> T SwigValueInit() {
return T();
}
#endif




#ifndef SWIGTEMPLATEDISAMBIGUATOR
# if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x560)
#  define SWIGTEMPLATEDISAMBIGUATOR template
# elif defined(__HP_aCC)


#  define SWIGTEMPLATEDISAMBIGUATOR template
# else
#  define SWIGTEMPLATEDISAMBIGUATOR
# endif
#endif


#ifndef SWIGINLINE
# if defined(__cplusplus) || (defined(__GNUC__) && !defined(__STRICT_ANSI__))
#   define SWIGINLINE inline
# else
#   define SWIGINLINE
# endif
#endif


#ifndef SWIGUNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define SWIGUNUSED __attribute__ ((__unused__))
#   else
#     define SWIGUNUSED
#   endif
# elif defined(__ICC)
#   define SWIGUNUSED __attribute__ ((__unused__))
# else
#   define SWIGUNUSED
# endif
#endif

#ifndef SWIG_MSC_UNSUPPRESS_4505
# if defined(_MSC_VER)
#   pragma warning(disable : 4505) 
# endif
#endif

#ifndef SWIGUNUSEDPARM
# ifdef __cplusplus
#   define SWIGUNUSEDPARM(p)
# else
#   define SWIGUNUSEDPARM(p) p SWIGUNUSED
# endif
#endif


#ifndef SWIGINTERN
# define SWIGINTERN static SWIGUNUSED
#endif


#ifndef SWIGINTERNINLINE
# define SWIGINTERNINLINE SWIGINTERN SWIGINLINE
#endif


#if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#  ifndef GCC_HASCLASSVISIBILITY
#    define GCC_HASCLASSVISIBILITY
#  endif
#endif

#ifndef SWIGEXPORT
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   if defined(STATIC_LINKED)
#     define SWIGEXPORT
#   else
#     define SWIGEXPORT __declspec(dllexport)
#   endif
# else
#   if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#     define SWIGEXPORT __attribute__ ((visibility("default")))
#   else
#     define SWIGEXPORT
#   endif
# endif
#endif


#ifndef SWIGSTDCALL
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   define SWIGSTDCALL __stdcall
# else
#   define SWIGSTDCALL
# endif
#endif


#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE
#endif


#if !defined(SWIG_NO_SCL_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_SCL_SECURE_NO_DEPRECATE)
# define _SCL_SECURE_NO_DEPRECATE
#endif



#if defined(_DEBUG) && defined(SWIG_PYTHON_INTERPRETER_NO_DEBUG)

# undef _DEBUG
# include <Python.h>
# define _DEBUG
#else
# include <Python.h>
#endif




#define SWIG_RUNTIME_VERSION "4"


#ifdef SWIG_TYPE_TABLE
# define SWIG_QUOTE_STRING(x) #x
# define SWIG_EXPAND_AND_QUOTE_STRING(x) SWIG_QUOTE_STRING(x)
# define SWIG_TYPE_TABLE_NAME SWIG_EXPAND_AND_QUOTE_STRING(SWIG_TYPE_TABLE)
#else
# define SWIG_TYPE_TABLE_NAME
#endif



#ifndef SWIGRUNTIME
# define SWIGRUNTIME SWIGINTERN
#endif

#ifndef SWIGRUNTIMEINLINE
# define SWIGRUNTIMEINLINE SWIGRUNTIME SWIGINLINE
#endif


#ifndef SWIG_BUFFER_SIZE
# define SWIG_BUFFER_SIZE 1024
#endif


#define SWIG_POINTER_DISOWN        0x1
#define SWIG_CAST_NEW_MEMORY       0x2


#define SWIG_POINTER_OWN           0x1




#define SWIG_OK                    (0)
#define SWIG_ERROR                 (-1)
#define SWIG_IsOK(r)               (r >= 0)
#define SWIG_ArgError(r)           ((r != SWIG_ERROR) ? r : SWIG_TypeError)


#define SWIG_CASTRANKLIMIT         (1 << 8)

#define SWIG_NEWOBJMASK            (SWIG_CASTRANKLIMIT  << 1)

#define SWIG_TMPOBJMASK            (SWIG_NEWOBJMASK << 1)

#define SWIG_BADOBJ                (SWIG_ERROR)
#define SWIG_OLDOBJ                (SWIG_OK)
#define SWIG_NEWOBJ                (SWIG_OK | SWIG_NEWOBJMASK)
#define SWIG_TMPOBJ                (SWIG_OK | SWIG_TMPOBJMASK)

#define SWIG_AddNewMask(r)         (SWIG_IsOK(r) ? (r | SWIG_NEWOBJMASK) : r)
#define SWIG_DelNewMask(r)         (SWIG_IsOK(r) ? (r & ~SWIG_NEWOBJMASK) : r)
#define SWIG_IsNewObj(r)           (SWIG_IsOK(r) && (r & SWIG_NEWOBJMASK))
#define SWIG_AddTmpMask(r)         (SWIG_IsOK(r) ? (r | SWIG_TMPOBJMASK) : r)
#define SWIG_DelTmpMask(r)         (SWIG_IsOK(r) ? (r & ~SWIG_TMPOBJMASK) : r)
#define SWIG_IsTmpObj(r)           (SWIG_IsOK(r) && (r & SWIG_TMPOBJMASK))


#if defined(SWIG_CASTRANK_MODE)
#  ifndef SWIG_TypeRank
#    define SWIG_TypeRank             unsigned long
#  endif
#  ifndef SWIG_MAXCASTRANK            
#    define SWIG_MAXCASTRANK          (2)
#  endif
#  define SWIG_CASTRANKMASK          ((SWIG_CASTRANKLIMIT) -1)
#  define SWIG_CastRank(r)           (r & SWIG_CASTRANKMASK)
SWIGINTERNINLINE int SWIG_AddCast(int r) {
return SWIG_IsOK(r) ? ((SWIG_CastRank(r) < SWIG_MAXCASTRANK) ? (r + 1) : SWIG_ERROR) : r;
}
SWIGINTERNINLINE int SWIG_CheckState(int r) {
return SWIG_IsOK(r) ? SWIG_CastRank(r) + 1 : 0;
}
#else 
#  define SWIG_AddCast(r) (r)
#  define SWIG_CheckState(r) (SWIG_IsOK(r) ? 1 : 0)
#endif


#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*swig_converter_func)(void *, int *);
typedef struct swig_type_info *(*swig_dycast_func)(void **);


typedef struct swig_type_info {
const char             *name;			
const char             *str;			
swig_dycast_func        dcast;		
struct swig_cast_info  *cast;			
void                   *clientdata;		
int                    owndata;		
} swig_type_info;


typedef struct swig_cast_info {
swig_type_info         *type;			
swig_converter_func     converter;		
struct swig_cast_info  *next;			
struct swig_cast_info  *prev;			
} swig_cast_info;


typedef struct swig_module_info {
swig_type_info         **types;		
size_t                 size;		        
struct swig_module_info *next;		
swig_type_info         **type_initial;	
swig_cast_info         **cast_initial;	
void                    *clientdata;		
} swig_module_info;


SWIGRUNTIME int
SWIG_TypeNameComp(const char *f1, const char *l1,
const char *f2, const char *l2) {
for (;(f1 != l1) && (f2 != l2); ++f1, ++f2) {
while ((*f1 == ' ') && (f1 != l1)) ++f1;
while ((*f2 == ' ') && (f2 != l2)) ++f2;
if (*f1 != *f2) return (*f1 > *f2) ? 1 : -1;
}
return (int)((l1 - f1) - (l2 - f2));
}


SWIGRUNTIME int
SWIG_TypeCmp(const char *nb, const char *tb) {
int equiv = 1;
const char* te = tb + strlen(tb);
const char* ne = nb;
while (equiv != 0 && *ne) {
for (nb = ne; *ne; ++ne) {
if (*ne == '|') break;
}
equiv = SWIG_TypeNameComp(nb, ne, tb, te);
if (*ne) ++ne;
}
return equiv;
}


SWIGRUNTIME int
SWIG_TypeEquiv(const char *nb, const char *tb) {
return SWIG_TypeCmp(nb, tb) == 0 ? 1 : 0;
}


SWIGRUNTIME swig_cast_info *
SWIG_TypeCheck(const char *c, swig_type_info *ty) {
if (ty) {
swig_cast_info *iter = ty->cast;
while (iter) {
if (strcmp(iter->type->name, c) == 0) {
if (iter == ty->cast)
return iter;

iter->prev->next = iter->next;
if (iter->next)
iter->next->prev = iter->prev;
iter->next = ty->cast;
iter->prev = 0;
if (ty->cast) ty->cast->prev = iter;
ty->cast = iter;
return iter;
}
iter = iter->next;
}
}
return 0;
}


SWIGRUNTIME swig_cast_info *
SWIG_TypeCheckStruct(swig_type_info *from, swig_type_info *ty) {
if (ty) {
swig_cast_info *iter = ty->cast;
while (iter) {
if (iter->type == from) {
if (iter == ty->cast)
return iter;

iter->prev->next = iter->next;
if (iter->next)
iter->next->prev = iter->prev;
iter->next = ty->cast;
iter->prev = 0;
if (ty->cast) ty->cast->prev = iter;
ty->cast = iter;
return iter;
}
iter = iter->next;
}
}
return 0;
}


SWIGRUNTIMEINLINE void *
SWIG_TypeCast(swig_cast_info *ty, void *ptr, int *newmemory) {
return ((!ty) || (!ty->converter)) ? ptr : (*ty->converter)(ptr, newmemory);
}


SWIGRUNTIME swig_type_info *
SWIG_TypeDynamicCast(swig_type_info *ty, void **ptr) {
swig_type_info *lastty = ty;
if (!ty || !ty->dcast) return ty;
while (ty && (ty->dcast)) {
ty = (*ty->dcast)(ptr);
if (ty) lastty = ty;
}
return lastty;
}


SWIGRUNTIMEINLINE const char *
SWIG_TypeName(const swig_type_info *ty) {
return ty->name;
}


SWIGRUNTIME const char *
SWIG_TypePrettyName(const swig_type_info *type) {

if (!type) return NULL;
if (type->str != NULL) {
const char *last_name = type->str;
const char *s;
for (s = type->str; *s; s++)
if (*s == '|') last_name = s+1;
return last_name;
}
else
return type->name;
}


SWIGRUNTIME void
SWIG_TypeClientData(swig_type_info *ti, void *clientdata) {
swig_cast_info *cast = ti->cast;

ti->clientdata = clientdata;

while (cast) {
if (!cast->converter) {
swig_type_info *tc = cast->type;
if (!tc->clientdata) {
SWIG_TypeClientData(tc, clientdata);
}
}
cast = cast->next;
}
}
SWIGRUNTIME void
SWIG_TypeNewClientData(swig_type_info *ti, void *clientdata) {
SWIG_TypeClientData(ti, clientdata);
ti->owndata = 1;
}


SWIGRUNTIME swig_type_info *
SWIG_MangledTypeQueryModule(swig_module_info *start,
swig_module_info *end,
const char *name) {
swig_module_info *iter = start;
do {
if (iter->size) {
register size_t l = 0;
register size_t r = iter->size - 1;
do {

register size_t i = (l + r) >> 1;
const char *iname = iter->types[i]->name;
if (iname) {
register int compare = strcmp(name, iname);
if (compare == 0) {
return iter->types[i];
} else if (compare < 0) {
if (i) {
r = i - 1;
} else {
break;
}
} else if (compare > 0) {
l = i + 1;
}
} else {
break; 
}
} while (l <= r);
}
iter = iter->next;
} while (iter != end);
return 0;
}


SWIGRUNTIME swig_type_info *
SWIG_TypeQueryModule(swig_module_info *start,
swig_module_info *end,
const char *name) {

swig_type_info *ret = SWIG_MangledTypeQueryModule(start, end, name);
if (ret) {
return ret;
} else {

swig_module_info *iter = start;
do {
register size_t i = 0;
for (; i < iter->size; ++i) {
if (iter->types[i]->str && (SWIG_TypeEquiv(iter->types[i]->str, name)))
return iter->types[i];
}
iter = iter->next;
} while (iter != end);
}


return 0;
}


SWIGRUNTIME char *
SWIG_PackData(char *c, void *ptr, size_t sz) {
static const char hex[17] = "0123456789abcdef";
register const unsigned char *u = (unsigned char *) ptr;
register const unsigned char *eu =  u + sz;
for (; u != eu; ++u) {
register unsigned char uu = *u;
*(c++) = hex[(uu & 0xf0) >> 4];
*(c++) = hex[uu & 0xf];
}
return c;
}


SWIGRUNTIME const char *
SWIG_UnpackData(const char *c, void *ptr, size_t sz) {
register unsigned char *u = (unsigned char *) ptr;
register const unsigned char *eu = u + sz;
for (; u != eu; ++u) {
register char d = *(c++);
register unsigned char uu;
if ((d >= '0') && (d <= '9'))
uu = ((d - '0') << 4);
else if ((d >= 'a') && (d <= 'f'))
uu = ((d - ('a'-10)) << 4);
else
return (char *) 0;
d = *(c++);
if ((d >= '0') && (d <= '9'))
uu |= (d - '0');
else if ((d >= 'a') && (d <= 'f'))
uu |= (d - ('a'-10));
else
return (char *) 0;
*u = uu;
}
return c;
}


SWIGRUNTIME char *
SWIG_PackVoidPtr(char *buff, void *ptr, const char *name, size_t bsz) {
char *r = buff;
if ((2*sizeof(void *) + 2) > bsz) return 0;
*(r++) = '_';
r = SWIG_PackData(r,&ptr,sizeof(void *));
if (strlen(name) + 1 > (bsz - (r - buff))) return 0;
strcpy(r,name);
return buff;
}

SWIGRUNTIME const char *
SWIG_UnpackVoidPtr(const char *c, void **ptr, const char *name) {
if (*c != '_') {
if (strcmp(c,"NULL") == 0) {
*ptr = (void *) 0;
return name;
} else {
return 0;
}
}
return SWIG_UnpackData(++c,ptr,sizeof(void *));
}

SWIGRUNTIME char *
SWIG_PackDataName(char *buff, void *ptr, size_t sz, const char *name, size_t bsz) {
char *r = buff;
size_t lname = (name ? strlen(name) : 0);
if ((2*sz + 2 + lname) > bsz) return 0;
*(r++) = '_';
r = SWIG_PackData(r,ptr,sz);
if (lname) {
strncpy(r,name,lname+1);
} else {
*r = 0;
}
return buff;
}

SWIGRUNTIME const char *
SWIG_UnpackDataName(const char *c, void *ptr, size_t sz, const char *name) {
if (*c != '_') {
if (strcmp(c,"NULL") == 0) {
memset(ptr,0,sz);
return name;
} else {
return 0;
}
}
return SWIG_UnpackData(++c,ptr,sz);
}

#ifdef __cplusplus
}
#endif


#define  SWIG_UnknownError    	   -1
#define  SWIG_IOError        	   -2
#define  SWIG_RuntimeError   	   -3
#define  SWIG_IndexError     	   -4
#define  SWIG_TypeError      	   -5
#define  SWIG_DivisionByZero 	   -6
#define  SWIG_OverflowError  	   -7
#define  SWIG_SyntaxError    	   -8
#define  SWIG_ValueError     	   -9
#define  SWIG_SystemError    	   -10
#define  SWIG_AttributeError 	   -11
#define  SWIG_MemoryError    	   -12
#define  SWIG_NullReferenceError   -13




#if PY_VERSION_HEX >= 0x03000000

#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
#define PyString_AsString(str) PyBytes_AsString(str)
#define PyString_Size(str) PyBytes_Size(str)	
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define PyString_AS_STRING(x) PyUnicode_AS_STRING(x)
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)

#endif

#ifndef Py_TYPE
#  define Py_TYPE(op) ((op)->ob_type)
#endif



#if PY_VERSION_HEX >= 0x03000000
#  define SWIG_Python_str_FromFormat PyUnicode_FromFormat
#else
#  define SWIG_Python_str_FromFormat PyString_FromFormat
#endif



SWIGINTERN char*
SWIG_Python_str_AsChar(PyObject *str)
{
#if PY_VERSION_HEX >= 0x03000000
char *cstr;
char *newstr;
Py_ssize_t len;
str = PyUnicode_AsUTF8String(str);
PyBytes_AsStringAndSize(str, &cstr, &len);
newstr = (char *) malloc(len+1);
memcpy(newstr, cstr, len+1);
Py_XDECREF(str);
return newstr;
#else
return PyString_AsString(str);
#endif
}

#if PY_VERSION_HEX >= 0x03000000
#  define SWIG_Python_str_DelForPy3(x) free( (void*) (x) )
#else
#  define SWIG_Python_str_DelForPy3(x) 
#endif


SWIGINTERN PyObject*
SWIG_Python_str_FromChar(const char *c)
{
#if PY_VERSION_HEX >= 0x03000000
return PyUnicode_FromString(c); 
#else
return PyString_FromString(c);
#endif
}


#if PY_VERSION_HEX < 0x02020000
# if defined(_MSC_VER) || defined(__BORLANDC__) || defined(_WATCOM)
#  define PyOS_snprintf _snprintf
# else
#  define PyOS_snprintf snprintf
# endif
#endif


#if PY_VERSION_HEX < 0x02020000

#ifndef SWIG_PYBUFFER_SIZE
# define SWIG_PYBUFFER_SIZE 1024
#endif

static PyObject *
PyString_FromFormat(const char *fmt, ...) {
va_list ap;
char buf[SWIG_PYBUFFER_SIZE * 2];
int res;
va_start(ap, fmt);
res = vsnprintf(buf, sizeof(buf), fmt, ap);
va_end(ap);
return (res < 0 || res >= (int)sizeof(buf)) ? 0 : PyString_FromString(buf);
}
#endif


#if PY_VERSION_HEX < 0x01060000
# define PyObject_Del(op) PyMem_DEL((op))
#endif
#ifndef PyObject_DEL
# define PyObject_DEL PyObject_Del
#endif


#if PY_VERSION_HEX < 0x02020000
# ifndef PyExc_StopIteration
#  define PyExc_StopIteration PyExc_RuntimeError
# endif
# ifndef PyObject_GenericGetAttr
#  define PyObject_GenericGetAttr 0
# endif
#endif


#if PY_VERSION_HEX < 0x02010000
# ifndef Py_NotImplemented
#  define Py_NotImplemented PyExc_RuntimeError
# endif
#endif


#if PY_VERSION_HEX < 0x02010000
# ifndef PyString_AsStringAndSize
#  define PyString_AsStringAndSize(obj, s, len) {*s = PyString_AsString(obj); *len = *s ? strlen(*s) : 0;}
# endif
#endif


#if PY_VERSION_HEX < 0x02000000
# ifndef PySequence_Size
#  define PySequence_Size PySequence_Length
# endif
#endif


#if PY_VERSION_HEX < 0x02030000
static
PyObject *PyBool_FromLong(long ok)
{
PyObject *result = ok ? Py_True : Py_False;
Py_INCREF(result);
return result;
}
#endif




#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
# define PY_SSIZE_T_MAX INT_MAX
# define PY_SSIZE_T_MIN INT_MIN
typedef inquiry lenfunc;
typedef intargfunc ssizeargfunc;
typedef intintargfunc ssizessizeargfunc;
typedef intobjargproc ssizeobjargproc;
typedef intintobjargproc ssizessizeobjargproc;
typedef getreadbufferproc readbufferproc;
typedef getwritebufferproc writebufferproc;
typedef getsegcountproc segcountproc;
typedef getcharbufferproc charbufferproc;
static long PyNumber_AsSsize_t (PyObject *x, void *SWIGUNUSEDPARM(exc))
{
long result = 0;
PyObject *i = PyNumber_Int(x);
if (i) {
result = PyInt_AsLong(i);
Py_DECREF(i);
}
return result;
}
#endif

#if PY_VERSION_HEX < 0x02050000
#define PyInt_FromSize_t(x) PyInt_FromLong((long)x)
#endif

#if PY_VERSION_HEX < 0x02040000
#define Py_VISIT(op)				\
do { 						\
if (op) {					\
int vret = visit((op), arg);		\
if (vret)					\
return vret;				\
}						\
} while (0)
#endif

#if PY_VERSION_HEX < 0x02030000
typedef struct {
PyTypeObject type;
PyNumberMethods as_number;
PyMappingMethods as_mapping;
PySequenceMethods as_sequence;
PyBufferProcs as_buffer;
PyObject *name, *slots;
} PyHeapTypeObject;
#endif

#if PY_VERSION_HEX < 0x02030000
typedef destructor freefunc;
#endif

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION > 6) || \
(PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 0) || \
(PY_MAJOR_VERSION > 3))
# define SWIGPY_USE_CAPSULE
# define SWIGPY_CAPSULE_NAME ((char*)"swig_runtime_data" SWIG_RUNTIME_VERSION ".type_pointer_capsule" SWIG_TYPE_TABLE_NAME)
#endif

#if PY_VERSION_HEX < 0x03020000
#define PyDescr_TYPE(x) (((PyDescrObject *)(x))->d_type)
#define PyDescr_NAME(x) (((PyDescrObject *)(x))->d_name)
#endif



SWIGRUNTIME PyObject*
SWIG_Python_ErrorType(int code) {
PyObject* type = 0;
switch(code) {
case SWIG_MemoryError:
type = PyExc_MemoryError;
break;
case SWIG_IOError:
type = PyExc_IOError;
break;
case SWIG_RuntimeError:
type = PyExc_RuntimeError;
break;
case SWIG_IndexError:
type = PyExc_IndexError;
break;
case SWIG_TypeError:
type = PyExc_TypeError;
break;
case SWIG_DivisionByZero:
type = PyExc_ZeroDivisionError;
break;
case SWIG_OverflowError:
type = PyExc_OverflowError;
break;
case SWIG_SyntaxError:
type = PyExc_SyntaxError;
break;
case SWIG_ValueError:
type = PyExc_ValueError;
break;
case SWIG_SystemError:
type = PyExc_SystemError;
break;
case SWIG_AttributeError:
type = PyExc_AttributeError;
break;
default:
type = PyExc_RuntimeError;
}
return type;
}


SWIGRUNTIME void
SWIG_Python_AddErrorMsg(const char* mesg)
{
PyObject *type = 0;
PyObject *value = 0;
PyObject *traceback = 0;

if (PyErr_Occurred()) PyErr_Fetch(&type, &value, &traceback);
if (value) {
char *tmp;
PyObject *old_str = PyObject_Str(value);
PyErr_Clear();
Py_XINCREF(type);

PyErr_Format(type, "%s %s", tmp = SWIG_Python_str_AsChar(old_str), mesg);
SWIG_Python_str_DelForPy3(tmp);
Py_DECREF(old_str);
Py_DECREF(value);
} else {
PyErr_SetString(PyExc_RuntimeError, mesg);
}
}

#if defined(SWIG_PYTHON_NO_THREADS)
#  if defined(SWIG_PYTHON_THREADS)
#    undef SWIG_PYTHON_THREADS
#  endif
#endif
#if defined(SWIG_PYTHON_THREADS) 
#  if !defined(SWIG_PYTHON_USE_GIL) && !defined(SWIG_PYTHON_NO_USE_GIL)
#    if (PY_VERSION_HEX >= 0x02030000) 
#      define SWIG_PYTHON_USE_GIL
#    endif
#  endif
#  if defined(SWIG_PYTHON_USE_GIL) 
#    ifndef SWIG_PYTHON_INITIALIZE_THREADS
#     define SWIG_PYTHON_INITIALIZE_THREADS  PyEval_InitThreads() 
#    endif
#    ifdef __cplusplus 
class SWIG_Python_Thread_Block {
bool status;
PyGILState_STATE state;
public:
void end() { if (status) { PyGILState_Release(state); status = false;} }
SWIG_Python_Thread_Block() : status(true), state(PyGILState_Ensure()) {}
~SWIG_Python_Thread_Block() { end(); }
};
class SWIG_Python_Thread_Allow {
bool status;
PyThreadState *save;
public:
void end() { if (status) { PyEval_RestoreThread(save); status = false; }}
SWIG_Python_Thread_Allow() : status(true), save(PyEval_SaveThread()) {}
~SWIG_Python_Thread_Allow() { end(); }
};
#      define SWIG_PYTHON_THREAD_BEGIN_BLOCK   SWIG_Python_Thread_Block _swig_thread_block
#      define SWIG_PYTHON_THREAD_END_BLOCK     _swig_thread_block.end()
#      define SWIG_PYTHON_THREAD_BEGIN_ALLOW   SWIG_Python_Thread_Allow _swig_thread_allow
#      define SWIG_PYTHON_THREAD_END_ALLOW     _swig_thread_allow.end()
#    else 
#      define SWIG_PYTHON_THREAD_BEGIN_BLOCK   PyGILState_STATE _swig_thread_block = PyGILState_Ensure()
#      define SWIG_PYTHON_THREAD_END_BLOCK     PyGILState_Release(_swig_thread_block)
#      define SWIG_PYTHON_THREAD_BEGIN_ALLOW   PyThreadState *_swig_thread_allow = PyEval_SaveThread()
#      define SWIG_PYTHON_THREAD_END_ALLOW     PyEval_RestoreThread(_swig_thread_allow)
#    endif
#  else 
#    if !defined(SWIG_PYTHON_INITIALIZE_THREADS)
#      define SWIG_PYTHON_INITIALIZE_THREADS
#    endif
#    if !defined(SWIG_PYTHON_THREAD_BEGIN_BLOCK)
#      define SWIG_PYTHON_THREAD_BEGIN_BLOCK
#    endif
#    if !defined(SWIG_PYTHON_THREAD_END_BLOCK)
#      define SWIG_PYTHON_THREAD_END_BLOCK
#    endif
#    if !defined(SWIG_PYTHON_THREAD_BEGIN_ALLOW)
#      define SWIG_PYTHON_THREAD_BEGIN_ALLOW
#    endif
#    if !defined(SWIG_PYTHON_THREAD_END_ALLOW)
#      define SWIG_PYTHON_THREAD_END_ALLOW
#    endif
#  endif
#else 
#  define SWIG_PYTHON_INITIALIZE_THREADS
#  define SWIG_PYTHON_THREAD_BEGIN_BLOCK
#  define SWIG_PYTHON_THREAD_END_BLOCK
#  define SWIG_PYTHON_THREAD_BEGIN_ALLOW
#  define SWIG_PYTHON_THREAD_END_ALLOW
#endif



#ifdef __cplusplus
extern "C" {
#endif




#define SWIG_PY_POINTER 4
#define SWIG_PY_BINARY  5


typedef struct swig_const_info {
int type;
char *name;
long lvalue;
double dvalue;
void   *pvalue;
swig_type_info **ptype;
} swig_const_info;



#if PY_VERSION_HEX >= 0x03000000
SWIGRUNTIME PyObject* SWIG_PyInstanceMethod_New(PyObject *SWIGUNUSEDPARM(self), PyObject *func)
{
return PyInstanceMethod_New(func);
}
#else
SWIGRUNTIME PyObject* SWIG_PyInstanceMethod_New(PyObject *SWIGUNUSEDPARM(self), PyObject *SWIGUNUSEDPARM(func))
{
return NULL;
}
#endif

#ifdef __cplusplus
}
#endif







#define SWIG_Python_ConvertPtr(obj, pptr, type, flags)  SWIG_Python_ConvertPtrAndOwn(obj, pptr, type, flags, 0)
#define SWIG_ConvertPtr(obj, pptr, type, flags)         SWIG_Python_ConvertPtr(obj, pptr, type, flags)
#define SWIG_ConvertPtrAndOwn(obj,pptr,type,flags,own)  SWIG_Python_ConvertPtrAndOwn(obj, pptr, type, flags, own)

#ifdef SWIGPYTHON_BUILTIN
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_Python_NewPointerObj(self, ptr, type, flags)
#else
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_Python_NewPointerObj(NULL, ptr, type, flags)
#endif

#define SWIG_InternalNewPointerObj(ptr, type, flags)	SWIG_Python_NewPointerObj(NULL, ptr, type, flags)

#define SWIG_CheckImplicit(ty)                          SWIG_Python_CheckImplicit(ty) 
#define SWIG_AcquirePtr(ptr, src)                       SWIG_Python_AcquirePtr(ptr, src)
#define swig_owntype                                    int


#define SWIG_ConvertPacked(obj, ptr, sz, ty)            SWIG_Python_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewPackedObj(ptr, sz, type)                SWIG_Python_NewPackedObj(ptr, sz, type)


#define SWIG_ConvertInstance(obj, pptr, type, flags)    SWIG_ConvertPtr(obj, pptr, type, flags)
#define SWIG_NewInstanceObj(ptr, type, flags)           SWIG_NewPointerObj(ptr, type, flags)


#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_Python_ConvertFunctionPtr(obj, pptr, type)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_Python_NewPointerObj(NULL, ptr, type, 0)


#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_Python_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIG_Python_NewPackedObj(ptr, sz, type)




#define SWIG_GetModule(clientdata)                      SWIG_Python_GetModule(clientdata)
#define SWIG_SetModule(clientdata, pointer)             SWIG_Python_SetModule(pointer)
#define SWIG_NewClientData(obj)                         SwigPyClientData_New(obj)

#define SWIG_SetErrorObj                                SWIG_Python_SetErrorObj                            
#define SWIG_SetErrorMsg                        	SWIG_Python_SetErrorMsg				   
#define SWIG_ErrorType(code)                    	SWIG_Python_ErrorType(code)                        
#define SWIG_Error(code, msg)            		SWIG_Python_SetErrorMsg(SWIG_ErrorType(code), msg) 
#define SWIG_fail                        		goto fail					   






SWIGINTERN void 
SWIG_Python_SetErrorObj(PyObject *errtype, PyObject *obj) {
SWIG_PYTHON_THREAD_BEGIN_BLOCK; 
PyErr_SetObject(errtype, obj);
Py_DECREF(obj);
SWIG_PYTHON_THREAD_END_BLOCK;
}

SWIGINTERN void 
SWIG_Python_SetErrorMsg(PyObject *errtype, const char *msg) {
SWIG_PYTHON_THREAD_BEGIN_BLOCK;
PyErr_SetString(errtype, msg);
SWIG_PYTHON_THREAD_END_BLOCK;
}

#define SWIG_Python_Raise(obj, type, desc)  SWIG_Python_SetErrorObj(SWIG_Python_ExceptionType(desc), obj)



#if defined(SWIGPYTHON_BUILTIN)

SWIGINTERN void
SwigPyBuiltin_AddPublicSymbol(PyObject *seq, const char *key) {
PyObject *s = PyString_InternFromString(key);
PyList_Append(seq, s);
Py_DECREF(s);
}

SWIGINTERN void
SWIG_Python_SetConstant(PyObject *d, PyObject *public_interface, const char *name, PyObject *obj) {   
#if PY_VERSION_HEX < 0x02030000
PyDict_SetItemString(d, (char *)name, obj);
#else
PyDict_SetItemString(d, name, obj);
#endif
Py_DECREF(obj);
if (public_interface)
SwigPyBuiltin_AddPublicSymbol(public_interface, name);
}

#else

SWIGINTERN void
SWIG_Python_SetConstant(PyObject *d, const char *name, PyObject *obj) {   
#if PY_VERSION_HEX < 0x02030000
PyDict_SetItemString(d, (char *)name, obj);
#else
PyDict_SetItemString(d, name, obj);
#endif
Py_DECREF(obj);                            
}

#endif



SWIGINTERN PyObject*
SWIG_Python_AppendOutput(PyObject* result, PyObject* obj) {
#if !defined(SWIG_PYTHON_OUTPUT_TUPLE)
if (!result) {
result = obj;
} else if (result == Py_None) {
Py_DECREF(result);
result = obj;
} else {
if (!PyList_Check(result)) {
PyObject *o2 = result;
result = PyList_New(1);
PyList_SetItem(result, 0, o2);
}
PyList_Append(result,obj);
Py_DECREF(obj);
}
return result;
#else
PyObject*   o2;
PyObject*   o3;
if (!result) {
result = obj;
} else if (result == Py_None) {
Py_DECREF(result);
result = obj;
} else {
if (!PyTuple_Check(result)) {
o2 = result;
result = PyTuple_New(1);
PyTuple_SET_ITEM(result, 0, o2);
}
o3 = PyTuple_New(1);
PyTuple_SET_ITEM(o3, 0, obj);
o2 = result;
result = PySequence_Concat(o2, o3);
Py_DECREF(o2);
Py_DECREF(o3);
}
return result;
#endif
}



SWIGINTERN int
SWIG_Python_UnpackTuple(PyObject *args, const char *name, Py_ssize_t min, Py_ssize_t max, PyObject **objs)
{
if (!args) {
if (!min && !max) {
return 1;
} else {
PyErr_Format(PyExc_TypeError, "%s expected %s%d arguments, got none", 
name, (min == max ? "" : "at least "), (int)min);
return 0;
}
}  
if (!PyTuple_Check(args)) {
if (min <= 1 && max >= 1) {
register int i;
objs[0] = args;
for (i = 1; i < max; ++i) {
objs[i] = 0;
}
return 2;
}
PyErr_SetString(PyExc_SystemError, "UnpackTuple() argument list is not a tuple");
return 0;
} else {
register Py_ssize_t l = PyTuple_GET_SIZE(args);
if (l < min) {
PyErr_Format(PyExc_TypeError, "%s expected %s%d arguments, got %d", 
name, (min == max ? "" : "at least "), (int)min, (int)l);
return 0;
} else if (l > max) {
PyErr_Format(PyExc_TypeError, "%s expected %s%d arguments, got %d", 
name, (min == max ? "" : "at most "), (int)max, (int)l);
return 0;
} else {
register int i;
for (i = 0; i < l; ++i) {
objs[i] = PyTuple_GET_ITEM(args, i);
}
for (; l < max; ++l) {
objs[l] = 0;
}
return i + 1;
}    
}
}


#if PY_VERSION_HEX >= 0x02020000
#define SWIG_Python_CallFunctor(functor, obj)	        PyObject_CallFunctionObjArgs(functor, obj, NULL);
#else
#define SWIG_Python_CallFunctor(functor, obj)	        PyObject_CallFunction(functor, "O", obj);
#endif


#ifdef __cplusplus
#define SWIG_STATIC_POINTER(var)  var
#else
#define SWIG_STATIC_POINTER(var)  var = 0; if (!var) var
#endif




#define SWIG_POINTER_NOSHADOW       (SWIG_POINTER_OWN      << 1)
#define SWIG_POINTER_NEW            (SWIG_POINTER_NOSHADOW | SWIG_POINTER_OWN)

#define SWIG_POINTER_IMPLICIT_CONV  (SWIG_POINTER_DISOWN   << 1)

#define SWIG_BUILTIN_TP_INIT	    (SWIG_POINTER_OWN << 2)
#define SWIG_BUILTIN_INIT	    (SWIG_BUILTIN_TP_INIT | SWIG_POINTER_OWN)

#ifdef __cplusplus
extern "C" {
#endif


#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#  ifndef SWIG_PYTHON_NO_BUILD_NONE
#    ifndef SWIG_PYTHON_BUILD_NONE
#      define SWIG_PYTHON_BUILD_NONE
#    endif
#  endif
#endif

#ifdef SWIG_PYTHON_BUILD_NONE
#  ifdef Py_None
#   undef Py_None
#   define Py_None SWIG_Py_None()
#  endif
SWIGRUNTIMEINLINE PyObject * 
_SWIG_Py_None(void)
{
PyObject *none = Py_BuildValue((char*)"");
Py_DECREF(none);
return none;
}
SWIGRUNTIME PyObject * 
SWIG_Py_None(void)
{
static PyObject *SWIG_STATIC_POINTER(none) = _SWIG_Py_None();
return none;
}
#endif



SWIGRUNTIMEINLINE PyObject * 
SWIG_Py_Void(void)
{
PyObject *none = Py_None;
Py_INCREF(none);
return none;
}



typedef struct {
PyObject *klass;
PyObject *newraw;
PyObject *newargs;
PyObject *destroy;
int delargs;
int implicitconv;
PyTypeObject *pytype;
} SwigPyClientData;

SWIGRUNTIMEINLINE int 
SWIG_Python_CheckImplicit(swig_type_info *ty)
{
SwigPyClientData *data = (SwigPyClientData *)ty->clientdata;
return data ? data->implicitconv : 0;
}

SWIGRUNTIMEINLINE PyObject *
SWIG_Python_ExceptionType(swig_type_info *desc) {
SwigPyClientData *data = desc ? (SwigPyClientData *) desc->clientdata : 0;
PyObject *klass = data ? data->klass : 0;
return (klass ? klass : PyExc_RuntimeError);
}


SWIGRUNTIME SwigPyClientData * 
SwigPyClientData_New(PyObject* obj)
{
if (!obj) {
return 0;
} else {
SwigPyClientData *data = (SwigPyClientData *)malloc(sizeof(SwigPyClientData));

data->klass = obj;
Py_INCREF(data->klass);

if (PyClass_Check(obj)) {
data->newraw = 0;
data->newargs = obj;
Py_INCREF(obj);
} else {
#if (PY_VERSION_HEX < 0x02020000)
data->newraw = 0;
#else
data->newraw = PyObject_GetAttrString(data->klass, (char *)"__new__");
#endif
if (data->newraw) {
Py_INCREF(data->newraw);
data->newargs = PyTuple_New(1);
PyTuple_SetItem(data->newargs, 0, obj);
} else {
data->newargs = obj;
}
Py_INCREF(data->newargs);
}

data->destroy = PyObject_GetAttrString(data->klass, (char *)"__swig_destroy__");
if (PyErr_Occurred()) {
PyErr_Clear();
data->destroy = 0;
}
if (data->destroy) {
int flags;
Py_INCREF(data->destroy);
flags = PyCFunction_GET_FLAGS(data->destroy);
#ifdef METH_O
data->delargs = !(flags & (METH_O));
#else
data->delargs = 0;
#endif
} else {
data->delargs = 0;
}
data->implicitconv = 0;
data->pytype = 0;
return data;
}
}

SWIGRUNTIME void 
SwigPyClientData_Del(SwigPyClientData *data) {
Py_XDECREF(data->newraw);
Py_XDECREF(data->newargs);
Py_XDECREF(data->destroy);
}



typedef struct {
PyObject_HEAD
void *ptr;
swig_type_info *ty;
int own;
PyObject *next;
#ifdef SWIGPYTHON_BUILTIN
PyObject *dict;
#endif
} SwigPyObject;

SWIGRUNTIME PyObject *
SwigPyObject_long(SwigPyObject *v)
{
return PyLong_FromVoidPtr(v->ptr);
}

SWIGRUNTIME PyObject *
SwigPyObject_format(const char* fmt, SwigPyObject *v)
{
PyObject *res = NULL;
PyObject *args = PyTuple_New(1);
if (args) {
if (PyTuple_SetItem(args, 0, SwigPyObject_long(v)) == 0) {
PyObject *ofmt = SWIG_Python_str_FromChar(fmt);
if (ofmt) {
#if PY_VERSION_HEX >= 0x03000000
res = PyUnicode_Format(ofmt,args);
#else
res = PyString_Format(ofmt,args);
#endif
Py_DECREF(ofmt);
}
Py_DECREF(args);
}
}
return res;
}

SWIGRUNTIME PyObject *
SwigPyObject_oct(SwigPyObject *v)
{
return SwigPyObject_format("%o",v);
}

SWIGRUNTIME PyObject *
SwigPyObject_hex(SwigPyObject *v)
{
return SwigPyObject_format("%x",v);
}

SWIGRUNTIME PyObject *
#ifdef METH_NOARGS
SwigPyObject_repr(SwigPyObject *v)
#else
SwigPyObject_repr(SwigPyObject *v, PyObject *args)
#endif
{
const char *name = SWIG_TypePrettyName(v->ty);
PyObject *repr = SWIG_Python_str_FromFormat("<Swig Object of type '%s' at %p>", (name ? name : "unknown"), (void *)v);
if (v->next) {
# ifdef METH_NOARGS
PyObject *nrep = SwigPyObject_repr((SwigPyObject *)v->next);
# else
PyObject *nrep = SwigPyObject_repr((SwigPyObject *)v->next, args);
# endif
# if PY_VERSION_HEX >= 0x03000000
PyObject *joined = PyUnicode_Concat(repr, nrep);
Py_DecRef(repr);
Py_DecRef(nrep);
repr = joined;
# else
PyString_ConcatAndDel(&repr,nrep);
# endif
}
return repr;  
}

SWIGRUNTIME int
SwigPyObject_compare(SwigPyObject *v, SwigPyObject *w)
{
void *i = v->ptr;
void *j = w->ptr;
return (i < j) ? -1 : ((i > j) ? 1 : 0);
}


SWIGRUNTIME PyObject*
SwigPyObject_richcompare(SwigPyObject *v, SwigPyObject *w, int op)
{
PyObject* res;
if( op != Py_EQ && op != Py_NE ) {
Py_INCREF(Py_NotImplemented);
return Py_NotImplemented;
}
res = PyBool_FromLong( (SwigPyObject_compare(v, w)==0) == (op == Py_EQ) ? 1 : 0);
return res;  
}


SWIGRUNTIME PyTypeObject* SwigPyObject_TypeOnce(void);

#ifdef SWIGPYTHON_BUILTIN
static swig_type_info *SwigPyObject_stype = 0;
SWIGRUNTIME PyTypeObject*
SwigPyObject_type(void) {
SwigPyClientData *cd;
assert(SwigPyObject_stype);
cd = (SwigPyClientData*) SwigPyObject_stype->clientdata;
assert(cd);
assert(cd->pytype);
return cd->pytype;
}
#else
SWIGRUNTIME PyTypeObject*
SwigPyObject_type(void) {
static PyTypeObject *SWIG_STATIC_POINTER(type) = SwigPyObject_TypeOnce();
return type;
}
#endif

SWIGRUNTIMEINLINE int
SwigPyObject_Check(PyObject *op) {
#ifdef SWIGPYTHON_BUILTIN
PyTypeObject *target_tp = SwigPyObject_type();
if (PyType_IsSubtype(op->ob_type, target_tp))
return 1;
return (strcmp(op->ob_type->tp_name, "SwigPyObject") == 0);
#else
return (Py_TYPE(op) == SwigPyObject_type())
|| (strcmp(Py_TYPE(op)->tp_name,"SwigPyObject") == 0);
#endif
}

SWIGRUNTIME PyObject *
SwigPyObject_New(void *ptr, swig_type_info *ty, int own);

SWIGRUNTIME void
SwigPyObject_dealloc(PyObject *v)
{
SwigPyObject *sobj = (SwigPyObject *) v;
PyObject *next = sobj->next;
if (sobj->own == SWIG_POINTER_OWN) {
swig_type_info *ty = sobj->ty;
SwigPyClientData *data = ty ? (SwigPyClientData *) ty->clientdata : 0;
PyObject *destroy = data ? data->destroy : 0;
if (destroy) {

PyObject *res;
if (data->delargs) {

PyObject *tmp = SwigPyObject_New(sobj->ptr, ty, 0);
res = SWIG_Python_CallFunctor(destroy, tmp);
Py_DECREF(tmp);
} else {
PyCFunction meth = PyCFunction_GET_FUNCTION(destroy);
PyObject *mself = PyCFunction_GET_SELF(destroy);
res = ((*meth)(mself, v));
}
Py_XDECREF(res);
} 
#if !defined(SWIG_PYTHON_SILENT_MEMLEAK)
else {
const char *name = SWIG_TypePrettyName(ty);
printf("swig/python detected a memory leak of type '%s', no destructor found.\n", (name ? name : "unknown"));
}
#endif
} 
Py_XDECREF(next);
PyObject_DEL(v);
}

SWIGRUNTIME PyObject* 
SwigPyObject_append(PyObject* v, PyObject* next)
{
SwigPyObject *sobj = (SwigPyObject *) v;
#ifndef METH_O
PyObject *tmp = 0;
if (!PyArg_ParseTuple(next,(char *)"O:append", &tmp)) return NULL;
next = tmp;
#endif
if (!SwigPyObject_Check(next)) {
return NULL;
}
sobj->next = next;
Py_INCREF(next);
return SWIG_Py_Void();
}

SWIGRUNTIME PyObject* 
#ifdef METH_NOARGS
SwigPyObject_next(PyObject* v)
#else
SwigPyObject_next(PyObject* v, PyObject *SWIGUNUSEDPARM(args))
#endif
{
SwigPyObject *sobj = (SwigPyObject *) v;
if (sobj->next) {    
Py_INCREF(sobj->next);
return sobj->next;
} else {
return SWIG_Py_Void();
}
}

SWIGINTERN PyObject*
#ifdef METH_NOARGS
SwigPyObject_disown(PyObject *v)
#else
SwigPyObject_disown(PyObject* v, PyObject *SWIGUNUSEDPARM(args))
#endif
{
SwigPyObject *sobj = (SwigPyObject *)v;
sobj->own = 0;
return SWIG_Py_Void();
}

SWIGINTERN PyObject*
#ifdef METH_NOARGS
SwigPyObject_acquire(PyObject *v)
#else
SwigPyObject_acquire(PyObject* v, PyObject *SWIGUNUSEDPARM(args))
#endif
{
SwigPyObject *sobj = (SwigPyObject *)v;
sobj->own = SWIG_POINTER_OWN;
return SWIG_Py_Void();
}

SWIGINTERN PyObject*
SwigPyObject_own(PyObject *v, PyObject *args)
{
PyObject *val = 0;
#if (PY_VERSION_HEX < 0x02020000)
if (!PyArg_ParseTuple(args,(char *)"|O:own",&val))
#elif (PY_VERSION_HEX < 0x02050000)
if (!PyArg_UnpackTuple(args, (char *)"own", 0, 1, &val)) 
#else
if (!PyArg_UnpackTuple(args, "own", 0, 1, &val)) 
#endif
{
return NULL;
} 
else
{
SwigPyObject *sobj = (SwigPyObject *)v;
PyObject *obj = PyBool_FromLong(sobj->own);
if (val) {
#ifdef METH_NOARGS
if (PyObject_IsTrue(val)) {
SwigPyObject_acquire(v);
} else {
SwigPyObject_disown(v);
}
#else
if (PyObject_IsTrue(val)) {
SwigPyObject_acquire(v,args);
} else {
SwigPyObject_disown(v,args);
}
#endif
} 
return obj;
}
}

#ifdef METH_O
static PyMethodDef
swigobject_methods[] = {
{(char *)"disown",  (PyCFunction)SwigPyObject_disown,  METH_NOARGS,  (char *)"releases ownership of the pointer"},
{(char *)"acquire", (PyCFunction)SwigPyObject_acquire, METH_NOARGS,  (char *)"acquires ownership of the pointer"},
{(char *)"own",     (PyCFunction)SwigPyObject_own,     METH_VARARGS, (char *)"returns/sets ownership of the pointer"},
{(char *)"append",  (PyCFunction)SwigPyObject_append,  METH_O,       (char *)"appends another 'this' object"},
{(char *)"next",    (PyCFunction)SwigPyObject_next,    METH_NOARGS,  (char *)"returns the next 'this' object"},
{(char *)"__repr__",(PyCFunction)SwigPyObject_repr,    METH_NOARGS,  (char *)"returns object representation"},
{0, 0, 0, 0}  
};
#else
static PyMethodDef
swigobject_methods[] = {
{(char *)"disown",  (PyCFunction)SwigPyObject_disown,  METH_VARARGS,  (char *)"releases ownership of the pointer"},
{(char *)"acquire", (PyCFunction)SwigPyObject_acquire, METH_VARARGS,  (char *)"aquires ownership of the pointer"},
{(char *)"own",     (PyCFunction)SwigPyObject_own,     METH_VARARGS,  (char *)"returns/sets ownership of the pointer"},
{(char *)"append",  (PyCFunction)SwigPyObject_append,  METH_VARARGS,  (char *)"appends another 'this' object"},
{(char *)"next",    (PyCFunction)SwigPyObject_next,    METH_VARARGS,  (char *)"returns the next 'this' object"},
{(char *)"__repr__",(PyCFunction)SwigPyObject_repr,   METH_VARARGS,  (char *)"returns object representation"},
{0, 0, 0, 0}  
};
#endif

#if PY_VERSION_HEX < 0x02020000
SWIGINTERN PyObject *
SwigPyObject_getattr(SwigPyObject *sobj,char *name)
{
return Py_FindMethod(swigobject_methods, (PyObject *)sobj, name);
}
#endif

SWIGRUNTIME PyTypeObject*
SwigPyObject_TypeOnce(void) {
static char swigobject_doc[] = "Swig object carries a C/C++ instance pointer";

static PyNumberMethods SwigPyObject_as_number = {
(binaryfunc)0, 
(binaryfunc)0, 
(binaryfunc)0, 

#if PY_VERSION_HEX < 0x03000000
(binaryfunc)0, 
#endif
(binaryfunc)0, 
(binaryfunc)0, 
(ternaryfunc)0,
(unaryfunc)0,  
(unaryfunc)0,  
(unaryfunc)0,  
(inquiry)0,    
0,		   
0,		   
0,		   
0,		   
0,		   
0,		   
#if PY_VERSION_HEX < 0x03000000
0,   
#endif
(unaryfunc)SwigPyObject_long, 
#if PY_VERSION_HEX < 0x03000000
(unaryfunc)SwigPyObject_long, 
#else
0, 
#endif
(unaryfunc)0,                 
#if PY_VERSION_HEX < 0x03000000
(unaryfunc)SwigPyObject_oct,  
(unaryfunc)SwigPyObject_hex,  
#endif
#if PY_VERSION_HEX >= 0x03000000 
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
#elif PY_VERSION_HEX >= 0x02050000 
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
#elif PY_VERSION_HEX >= 0x02020000 
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
#elif PY_VERSION_HEX >= 0x02000000 
0,0,0,0,0,0,0,0,0,0,0 
#endif
};

static PyTypeObject swigpyobject_type;
static int type_init = 0;
if (!type_init) {
const PyTypeObject tmp = {

#if PY_VERSION_HEX >= 0x03000000
PyVarObject_HEAD_INIT(NULL, 0)
#else
PyObject_HEAD_INIT(NULL)
0,                                    
#endif
(char *)"SwigPyObject",               
sizeof(SwigPyObject),                 
0,                                    
(destructor)SwigPyObject_dealloc,     
0,				    
#if PY_VERSION_HEX < 0x02020000
(getattrfunc)SwigPyObject_getattr,    
#else
(getattrfunc)0,                       
#endif
(setattrfunc)0,                       
#if PY_VERSION_HEX >= 0x03000000
0, 
#else
(cmpfunc)SwigPyObject_compare,        
#endif
(reprfunc)SwigPyObject_repr,          
&SwigPyObject_as_number,              
0,                                    
0,                                    
(hashfunc)0,                          
(ternaryfunc)0,                       
0,				    
PyObject_GenericGetAttr,              
0,                                    
0,                                    
Py_TPFLAGS_DEFAULT,                   
swigobject_doc,                       
0,                                    
0,                                    
(richcmpfunc)SwigPyObject_richcompare,
0,                                    
#if PY_VERSION_HEX >= 0x02020000
0,                                    
0,                                    
swigobject_methods,                   
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
#endif
#if PY_VERSION_HEX >= 0x02030000
0,                                    
#endif
#if PY_VERSION_HEX >= 0x02060000
0,                                    
#endif
#ifdef COUNT_ALLOCS
0,0,0,0                               
#endif
};
swigpyobject_type = tmp;
type_init = 1;
#if PY_VERSION_HEX < 0x02020000
swigpyobject_type.ob_type = &PyType_Type;
#else
if (PyType_Ready(&swigpyobject_type) < 0)
return NULL;
#endif
}
return &swigpyobject_type;
}

SWIGRUNTIME PyObject *
SwigPyObject_New(void *ptr, swig_type_info *ty, int own)
{
SwigPyObject *sobj = PyObject_NEW(SwigPyObject, SwigPyObject_type());
if (sobj) {
sobj->ptr  = ptr;
sobj->ty   = ty;
sobj->own  = own;
sobj->next = 0;
}
return (PyObject *)sobj;
}



typedef struct {
PyObject_HEAD
void *pack;
swig_type_info *ty;
size_t size;
} SwigPyPacked;

SWIGRUNTIME int
SwigPyPacked_print(SwigPyPacked *v, FILE *fp, int SWIGUNUSEDPARM(flags))
{
char result[SWIG_BUFFER_SIZE];
fputs("<Swig Packed ", fp); 
if (SWIG_PackDataName(result, v->pack, v->size, 0, sizeof(result))) {
fputs("at ", fp); 
fputs(result, fp); 
}
fputs(v->ty->name,fp); 
fputs(">", fp);
return 0; 
}

SWIGRUNTIME PyObject *
SwigPyPacked_repr(SwigPyPacked *v)
{
char result[SWIG_BUFFER_SIZE];
if (SWIG_PackDataName(result, v->pack, v->size, 0, sizeof(result))) {
return SWIG_Python_str_FromFormat("<Swig Packed at %s%s>", result, v->ty->name);
} else {
return SWIG_Python_str_FromFormat("<Swig Packed %s>", v->ty->name);
}  
}

SWIGRUNTIME PyObject *
SwigPyPacked_str(SwigPyPacked *v)
{
char result[SWIG_BUFFER_SIZE];
if (SWIG_PackDataName(result, v->pack, v->size, 0, sizeof(result))){
return SWIG_Python_str_FromFormat("%s%s", result, v->ty->name);
} else {
return SWIG_Python_str_FromChar(v->ty->name);
}  
}

SWIGRUNTIME int
SwigPyPacked_compare(SwigPyPacked *v, SwigPyPacked *w)
{
size_t i = v->size;
size_t j = w->size;
int s = (i < j) ? -1 : ((i > j) ? 1 : 0);
return s ? s : strncmp((char *)v->pack, (char *)w->pack, 2*v->size);
}

SWIGRUNTIME PyTypeObject* SwigPyPacked_TypeOnce(void);

SWIGRUNTIME PyTypeObject*
SwigPyPacked_type(void) {
static PyTypeObject *SWIG_STATIC_POINTER(type) = SwigPyPacked_TypeOnce();
return type;
}

SWIGRUNTIMEINLINE int
SwigPyPacked_Check(PyObject *op) {
return ((op)->ob_type == SwigPyPacked_TypeOnce()) 
|| (strcmp((op)->ob_type->tp_name,"SwigPyPacked") == 0);
}

SWIGRUNTIME void
SwigPyPacked_dealloc(PyObject *v)
{
if (SwigPyPacked_Check(v)) {
SwigPyPacked *sobj = (SwigPyPacked *) v;
free(sobj->pack);
}
PyObject_DEL(v);
}

SWIGRUNTIME PyTypeObject*
SwigPyPacked_TypeOnce(void) {
static char swigpacked_doc[] = "Swig object carries a C/C++ instance pointer";
static PyTypeObject swigpypacked_type;
static int type_init = 0;
if (!type_init) {
const PyTypeObject tmp = {

#if PY_VERSION_HEX>=0x03000000
PyVarObject_HEAD_INIT(NULL, 0)
#else
PyObject_HEAD_INIT(NULL)
0,                                    
#endif
(char *)"SwigPyPacked",               
sizeof(SwigPyPacked),                 
0,                                    
(destructor)SwigPyPacked_dealloc,     
(printfunc)SwigPyPacked_print,        
(getattrfunc)0,                       
(setattrfunc)0,                       
#if PY_VERSION_HEX>=0x03000000
0, 
#else
(cmpfunc)SwigPyPacked_compare,        
#endif
(reprfunc)SwigPyPacked_repr,          
0,                                    
0,                                    
0,                                    
(hashfunc)0,                          
(ternaryfunc)0,                       
(reprfunc)SwigPyPacked_str,           
PyObject_GenericGetAttr,              
0,                                    
0,                                    
Py_TPFLAGS_DEFAULT,                   
swigpacked_doc,                       
0,                                    
0,                                    
0,                                    
0,                                    
#if PY_VERSION_HEX >= 0x02020000
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
0,                                    
#endif
#if PY_VERSION_HEX >= 0x02030000
0,                                    
#endif
#if PY_VERSION_HEX >= 0x02060000
0,                                    
#endif
#ifdef COUNT_ALLOCS
0,0,0,0                               
#endif
};
swigpypacked_type = tmp;
type_init = 1;
#if PY_VERSION_HEX < 0x02020000
swigpypacked_type.ob_type = &PyType_Type;
#else
if (PyType_Ready(&swigpypacked_type) < 0)
return NULL;
#endif
}
return &swigpypacked_type;
}

SWIGRUNTIME PyObject *
SwigPyPacked_New(void *ptr, size_t size, swig_type_info *ty)
{
SwigPyPacked *sobj = PyObject_NEW(SwigPyPacked, SwigPyPacked_type());
if (sobj) {
void *pack = malloc(size);
if (pack) {
memcpy(pack, ptr, size);
sobj->pack = pack;
sobj->ty   = ty;
sobj->size = size;
} else {
PyObject_DEL((PyObject *) sobj);
sobj = 0;
}
}
return (PyObject *) sobj;
}

SWIGRUNTIME swig_type_info *
SwigPyPacked_UnpackData(PyObject *obj, void *ptr, size_t size)
{
if (SwigPyPacked_Check(obj)) {
SwigPyPacked *sobj = (SwigPyPacked *)obj;
if (sobj->size != size) return 0;
memcpy(ptr, sobj->pack, size);
return sobj->ty;
} else {
return 0;
}
}



SWIGRUNTIMEINLINE PyObject *
_SWIG_This(void)
{
return SWIG_Python_str_FromChar("this");
}

static PyObject *swig_this = NULL;

SWIGRUNTIME PyObject *
SWIG_This(void)
{
if (swig_this == NULL)
swig_this = _SWIG_This();
return swig_this;
}




#if PY_VERSION_HEX>=0x03000000
#define SWIG_PYTHON_SLOW_GETSET_THIS 
#endif

SWIGRUNTIME SwigPyObject *
SWIG_Python_GetSwigThis(PyObject *pyobj) 
{
PyObject *obj;

if (SwigPyObject_Check(pyobj))
return (SwigPyObject *) pyobj;

#ifdef SWIGPYTHON_BUILTIN
(void)obj;
# ifdef PyWeakref_CheckProxy
if (PyWeakref_CheckProxy(pyobj)) {
pyobj = PyWeakref_GET_OBJECT(pyobj);
if (pyobj && SwigPyObject_Check(pyobj))
return (SwigPyObject*) pyobj;
}
# endif
return NULL;
#else

obj = 0;

#if (!defined(SWIG_PYTHON_SLOW_GETSET_THIS) && (PY_VERSION_HEX >= 0x02030000))
if (PyInstance_Check(pyobj)) {
obj = _PyInstance_Lookup(pyobj, SWIG_This());      
} else {
PyObject **dictptr = _PyObject_GetDictPtr(pyobj);
if (dictptr != NULL) {
PyObject *dict = *dictptr;
obj = dict ? PyDict_GetItem(dict, SWIG_This()) : 0;
} else {
#ifdef PyWeakref_CheckProxy
if (PyWeakref_CheckProxy(pyobj)) {
PyObject *wobj = PyWeakref_GET_OBJECT(pyobj);
return wobj ? SWIG_Python_GetSwigThis(wobj) : 0;
}
#endif
obj = PyObject_GetAttr(pyobj,SWIG_This());
if (obj) {
Py_DECREF(obj);
} else {
if (PyErr_Occurred()) PyErr_Clear();
return 0;
}
}
}
#else
obj = PyObject_GetAttr(pyobj,SWIG_This());
if (obj) {
Py_DECREF(obj);
} else {
if (PyErr_Occurred()) PyErr_Clear();
return 0;
}
#endif
if (obj && !SwigPyObject_Check(obj)) {

return SWIG_Python_GetSwigThis(obj);
}
return (SwigPyObject *)obj;
#endif
}



SWIGRUNTIME int
SWIG_Python_AcquirePtr(PyObject *obj, int own) {
if (own == SWIG_POINTER_OWN) {
SwigPyObject *sobj = SWIG_Python_GetSwigThis(obj);
if (sobj) {
int oldown = sobj->own;
sobj->own = own;
return oldown;
}
}
return 0;
}



SWIGRUNTIME int
SWIG_Python_ConvertPtrAndOwn(PyObject *obj, void **ptr, swig_type_info *ty, int flags, int *own) {
int res;
SwigPyObject *sobj;
int implicit_conv = (flags & SWIG_POINTER_IMPLICIT_CONV) != 0;

if (!obj)
return SWIG_ERROR;
if (obj == Py_None && !implicit_conv) {
if (ptr)
*ptr = 0;
return SWIG_OK;
}

res = SWIG_ERROR;

sobj = SWIG_Python_GetSwigThis(obj);
if (own)
*own = 0;
while (sobj) {
void *vptr = sobj->ptr;
if (ty) {
swig_type_info *to = sobj->ty;
if (to == ty) {

if (ptr) *ptr = vptr;
break;
} else {
swig_cast_info *tc = SWIG_TypeCheck(to->name,ty);
if (!tc) {
sobj = (SwigPyObject *)sobj->next;
} else {
if (ptr) {
int newmemory = 0;
*ptr = SWIG_TypeCast(tc,vptr,&newmemory);
if (newmemory == SWIG_CAST_NEW_MEMORY) {
assert(own); 
if (own)
*own = *own | SWIG_CAST_NEW_MEMORY;
}
}
break;
}
}
} else {
if (ptr) *ptr = vptr;
break;
}
}
if (sobj) {
if (own)
*own = *own | sobj->own;
if (flags & SWIG_POINTER_DISOWN) {
sobj->own = 0;
}
res = SWIG_OK;
} else {
if (implicit_conv) {
SwigPyClientData *data = ty ? (SwigPyClientData *) ty->clientdata : 0;
if (data && !data->implicitconv) {
PyObject *klass = data->klass;
if (klass) {
PyObject *impconv;
data->implicitconv = 1; 
impconv = SWIG_Python_CallFunctor(klass, obj);
data->implicitconv = 0;
if (PyErr_Occurred()) {
PyErr_Clear();
impconv = 0;
}
if (impconv) {
SwigPyObject *iobj = SWIG_Python_GetSwigThis(impconv);
if (iobj) {
void *vptr;
res = SWIG_Python_ConvertPtrAndOwn((PyObject*)iobj, &vptr, ty, 0, 0);
if (SWIG_IsOK(res)) {
if (ptr) {
*ptr = vptr;

iobj->own = 0;
res = SWIG_AddCast(res);
res = SWIG_AddNewMask(res);
} else {
res = SWIG_AddCast(res);		    
}
}
}
Py_DECREF(impconv);
}
}
}
}
if (!SWIG_IsOK(res) && obj == Py_None) {
if (ptr)
*ptr = 0;
if (PyErr_Occurred())
PyErr_Clear();
res = SWIG_OK;
}
}
return res;
}



SWIGRUNTIME int
SWIG_Python_ConvertFunctionPtr(PyObject *obj, void **ptr, swig_type_info *ty) {
if (!PyCFunction_Check(obj)) {
return SWIG_ConvertPtr(obj, ptr, ty, 0);
} else {
void *vptr = 0;


const char *doc = (((PyCFunctionObject *)obj) -> m_ml -> ml_doc);
const char *desc = doc ? strstr(doc, "swig_ptr: ") : 0;
if (desc)
desc = ty ? SWIG_UnpackVoidPtr(desc + 10, &vptr, ty->name) : 0;
if (!desc) 
return SWIG_ERROR;
if (ty) {
swig_cast_info *tc = SWIG_TypeCheck(desc,ty);
if (tc) {
int newmemory = 0;
*ptr = SWIG_TypeCast(tc,vptr,&newmemory);
assert(!newmemory); 
} else {
return SWIG_ERROR;
}
} else {
*ptr = vptr;
}
return SWIG_OK;
}
}



SWIGRUNTIME int
SWIG_Python_ConvertPacked(PyObject *obj, void *ptr, size_t sz, swig_type_info *ty) {
swig_type_info *to = SwigPyPacked_UnpackData(obj, ptr, sz);
if (!to) return SWIG_ERROR;
if (ty) {
if (to != ty) {

swig_cast_info *tc = SWIG_TypeCheck(to->name,ty);
if (!tc) return SWIG_ERROR;
}
}
return SWIG_OK;
}  





SWIGRUNTIME PyObject* 
SWIG_Python_NewShadowInstance(SwigPyClientData *data, PyObject *swig_this)
{
#if (PY_VERSION_HEX >= 0x02020000)
PyObject *inst = 0;
PyObject *newraw = data->newraw;
if (newraw) {
inst = PyObject_Call(newraw, data->newargs, NULL);
if (inst) {
#if !defined(SWIG_PYTHON_SLOW_GETSET_THIS)
PyObject **dictptr = _PyObject_GetDictPtr(inst);
if (dictptr != NULL) {
PyObject *dict = *dictptr;
if (dict == NULL) {
dict = PyDict_New();
*dictptr = dict;
PyDict_SetItem(dict, SWIG_This(), swig_this);
}
}
#else
PyObject *key = SWIG_This();
PyObject_SetAttr(inst, key, swig_this);
#endif
}
} else {
#if PY_VERSION_HEX >= 0x03000000
inst = PyBaseObject_Type.tp_new((PyTypeObject*) data->newargs, Py_None, Py_None);
if (inst) {
PyObject_SetAttr(inst, SWIG_This(), swig_this);
Py_TYPE(inst)->tp_flags &= ~Py_TPFLAGS_VALID_VERSION_TAG;
}
#else
PyObject *dict = PyDict_New();
if (dict) {
PyDict_SetItem(dict, SWIG_This(), swig_this);
inst = PyInstance_NewRaw(data->newargs, dict);
Py_DECREF(dict);
}
#endif
}
return inst;
#else
#if (PY_VERSION_HEX >= 0x02010000)
PyObject *inst = 0;
PyObject *dict = PyDict_New();
if (dict) {
PyDict_SetItem(dict, SWIG_This(), swig_this);
inst = PyInstance_NewRaw(data->newargs, dict);
Py_DECREF(dict);
}
return (PyObject *) inst;
#else
PyInstanceObject *inst = PyObject_NEW(PyInstanceObject, &PyInstance_Type);
if (inst == NULL) {
return NULL;
}
inst->in_class = (PyClassObject *)data->newargs;
Py_INCREF(inst->in_class);
inst->in_dict = PyDict_New();
if (inst->in_dict == NULL) {
Py_DECREF(inst);
return NULL;
}
#ifdef Py_TPFLAGS_HAVE_WEAKREFS
inst->in_weakreflist = NULL;
#endif
#ifdef Py_TPFLAGS_GC
PyObject_GC_Init(inst);
#endif
PyDict_SetItem(inst->in_dict, SWIG_This(), swig_this);
return (PyObject *) inst;
#endif
#endif
}

SWIGRUNTIME void
SWIG_Python_SetSwigThis(PyObject *inst, PyObject *swig_this)
{
PyObject *dict;
#if (PY_VERSION_HEX >= 0x02020000) && !defined(SWIG_PYTHON_SLOW_GETSET_THIS)
PyObject **dictptr = _PyObject_GetDictPtr(inst);
if (dictptr != NULL) {
dict = *dictptr;
if (dict == NULL) {
dict = PyDict_New();
*dictptr = dict;
}
PyDict_SetItem(dict, SWIG_This(), swig_this);
return;
}
#endif
dict = PyObject_GetAttrString(inst, (char*)"__dict__");
PyDict_SetItem(dict, SWIG_This(), swig_this);
Py_DECREF(dict);
} 


SWIGINTERN PyObject *
SWIG_Python_InitShadowInstance(PyObject *args) {
PyObject *obj[2];
if (!SWIG_Python_UnpackTuple(args, "swiginit", 2, 2, obj)) {
return NULL;
} else {
SwigPyObject *sthis = SWIG_Python_GetSwigThis(obj[0]);
if (sthis) {
SwigPyObject_append((PyObject*) sthis, obj[1]);
} else {
SWIG_Python_SetSwigThis(obj[0], obj[1]);
}
return SWIG_Py_Void();
}
}



SWIGRUNTIME PyObject *
SWIG_Python_NewPointerObj(PyObject *self, void *ptr, swig_type_info *type, int flags) {
SwigPyClientData *clientdata;
PyObject * robj;
int own;

if (!ptr)
return SWIG_Py_Void();

clientdata = type ? (SwigPyClientData *)(type->clientdata) : 0;
own = (flags & SWIG_POINTER_OWN) ? SWIG_POINTER_OWN : 0;
if (clientdata && clientdata->pytype) {
SwigPyObject *newobj;
if (flags & SWIG_BUILTIN_TP_INIT) {
newobj = (SwigPyObject*) self;
if (newobj->ptr) {
PyObject *next_self = clientdata->pytype->tp_alloc(clientdata->pytype, 0);
while (newobj->next)
newobj = (SwigPyObject *) newobj->next;
newobj->next = next_self;
newobj = (SwigPyObject *)next_self;
}
} else {
newobj = PyObject_New(SwigPyObject, clientdata->pytype);
}
if (newobj) {
newobj->ptr = ptr;
newobj->ty = type;
newobj->own = own;
newobj->next = 0;
#ifdef SWIGPYTHON_BUILTIN
newobj->dict = 0;
#endif
return (PyObject*) newobj;
}
return SWIG_Py_Void();
}

assert(!(flags & SWIG_BUILTIN_TP_INIT));

robj = SwigPyObject_New(ptr, type, own);
if (robj && clientdata && !(flags & SWIG_POINTER_NOSHADOW)) {
PyObject *inst = SWIG_Python_NewShadowInstance(clientdata, robj);
Py_DECREF(robj);
robj = inst;
}
return robj;
}



SWIGRUNTIMEINLINE PyObject *
SWIG_Python_NewPackedObj(void *ptr, size_t sz, swig_type_info *type) {
return ptr ? SwigPyPacked_New((void *) ptr, sz, type) : SWIG_Py_Void();
}



#ifdef SWIG_LINK_RUNTIME
void *SWIG_ReturnGlobalTypeList(void *);
#endif

SWIGRUNTIME swig_module_info *
SWIG_Python_GetModule(void *SWIGUNUSEDPARM(clientdata)) {
static void *type_pointer = (void *)0;

if (!type_pointer) {
#ifdef SWIG_LINK_RUNTIME
type_pointer = SWIG_ReturnGlobalTypeList((void *)0);
#else
# ifdef SWIGPY_USE_CAPSULE
type_pointer = PyCapsule_Import(SWIGPY_CAPSULE_NAME, 0);
# else
type_pointer = PyCObject_Import((char*)"swig_runtime_data" SWIG_RUNTIME_VERSION,
(char*)"type_pointer" SWIG_TYPE_TABLE_NAME);
# endif
if (PyErr_Occurred()) {
PyErr_Clear();
type_pointer = (void *)0;
}
#endif
}
return (swig_module_info *) type_pointer;
}

#if PY_MAJOR_VERSION < 2

SWIGINTERN int
PyModule_AddObject(PyObject *m, char *name, PyObject *o)
{
PyObject *dict;
if (!PyModule_Check(m)) {
PyErr_SetString(PyExc_TypeError,
"PyModule_AddObject() needs module as first arg");
return SWIG_ERROR;
}
if (!o) {
PyErr_SetString(PyExc_TypeError,
"PyModule_AddObject() needs non-NULL value");
return SWIG_ERROR;
}

dict = PyModule_GetDict(m);
if (dict == NULL) {

PyErr_Format(PyExc_SystemError, "module '%s' has no __dict__",
PyModule_GetName(m));
return SWIG_ERROR;
}
if (PyDict_SetItemString(dict, name, o))
return SWIG_ERROR;
Py_DECREF(o);
return SWIG_OK;
}
#endif

SWIGRUNTIME void
#ifdef SWIGPY_USE_CAPSULE
SWIG_Python_DestroyModule(PyObject *obj)
#else
SWIG_Python_DestroyModule(void *vptr)
#endif
{
#ifdef SWIGPY_USE_CAPSULE
swig_module_info *swig_module = (swig_module_info *) PyCapsule_GetPointer(obj, SWIGPY_CAPSULE_NAME);
#else
swig_module_info *swig_module = (swig_module_info *) vptr;
#endif
swig_type_info **types = swig_module->types;
size_t i;
for (i =0; i < swig_module->size; ++i) {
swig_type_info *ty = types[i];
if (ty->owndata) {
SwigPyClientData *data = (SwigPyClientData *) ty->clientdata;
if (data) SwigPyClientData_Del(data);
}
}
Py_DECREF(SWIG_This());
swig_this = NULL;
}

SWIGRUNTIME void
SWIG_Python_SetModule(swig_module_info *swig_module) {
#if PY_VERSION_HEX >= 0x03000000

PyObject *module = PyImport_AddModule((char*)"swig_runtime_data" SWIG_RUNTIME_VERSION);
#else
static PyMethodDef swig_empty_runtime_method_table[] = { {NULL, NULL, 0, NULL} }; 
PyObject *module = Py_InitModule((char*)"swig_runtime_data" SWIG_RUNTIME_VERSION, swig_empty_runtime_method_table);
#endif
#ifdef SWIGPY_USE_CAPSULE
PyObject *pointer = PyCapsule_New((void *) swig_module, SWIGPY_CAPSULE_NAME, SWIG_Python_DestroyModule);
if (pointer && module) {
PyModule_AddObject(module, (char*)"type_pointer_capsule" SWIG_TYPE_TABLE_NAME, pointer);
} else {
Py_XDECREF(pointer);
}
#else
PyObject *pointer = PyCObject_FromVoidPtr((void *) swig_module, SWIG_Python_DestroyModule);
if (pointer && module) {
PyModule_AddObject(module, (char*)"type_pointer" SWIG_TYPE_TABLE_NAME, pointer);
} else {
Py_XDECREF(pointer);
}
#endif
}


SWIGRUNTIME PyObject *
SWIG_Python_TypeCache(void) {
static PyObject *SWIG_STATIC_POINTER(cache) = PyDict_New();
return cache;
}

SWIGRUNTIME swig_type_info *
SWIG_Python_TypeQuery(const char *type)
{
PyObject *cache = SWIG_Python_TypeCache();
PyObject *key = SWIG_Python_str_FromChar(type); 
PyObject *obj = PyDict_GetItem(cache, key);
swig_type_info *descriptor;
if (obj) {
#ifdef SWIGPY_USE_CAPSULE
descriptor = (swig_type_info *) PyCapsule_GetPointer(obj, NULL);
#else
descriptor = (swig_type_info *) PyCObject_AsVoidPtr(obj);
#endif
} else {
swig_module_info *swig_module = SWIG_GetModule(0);
descriptor = SWIG_TypeQueryModule(swig_module, swig_module, type);
if (descriptor) {
#ifdef SWIGPY_USE_CAPSULE
obj = PyCapsule_New((void*) descriptor, NULL, NULL);
#else
obj = PyCObject_FromVoidPtr(descriptor, NULL);
#endif
PyDict_SetItem(cache, key, obj);
Py_DECREF(obj);
}
}
Py_DECREF(key);
return descriptor;
}


#define SWIG_POINTER_EXCEPTION  0
#define SWIG_arg_fail(arg)      SWIG_Python_ArgFail(arg)
#define SWIG_MustGetPtr(p, type, argnum, flags)  SWIG_Python_MustGetPtr(p, type, argnum, flags)

SWIGRUNTIME int
SWIG_Python_AddErrMesg(const char* mesg, int infront)
{  
if (PyErr_Occurred()) {
PyObject *type = 0;
PyObject *value = 0;
PyObject *traceback = 0;
PyErr_Fetch(&type, &value, &traceback);
if (value) {
char *tmp;
PyObject *old_str = PyObject_Str(value);
Py_XINCREF(type);
PyErr_Clear();
if (infront) {
PyErr_Format(type, "%s %s", mesg, tmp = SWIG_Python_str_AsChar(old_str));
} else {
PyErr_Format(type, "%s %s", tmp = SWIG_Python_str_AsChar(old_str), mesg);
}
SWIG_Python_str_DelForPy3(tmp);
Py_DECREF(old_str);
}
return 1;
} else {
return 0;
}
}

SWIGRUNTIME int
SWIG_Python_ArgFail(int argnum)
{
if (PyErr_Occurred()) {

char mesg[256];
PyOS_snprintf(mesg, sizeof(mesg), "argument number %d:", argnum);
return SWIG_Python_AddErrMesg(mesg, 1);
} else {
return 0;
}
}

SWIGRUNTIMEINLINE const char *
SwigPyObject_GetDesc(PyObject *self)
{
SwigPyObject *v = (SwigPyObject *)self;
swig_type_info *ty = v ? v->ty : 0;
return ty ? ty->str : "";
}

SWIGRUNTIME void
SWIG_Python_TypeError(const char *type, PyObject *obj)
{
if (type) {
#if defined(SWIG_COBJECT_TYPES)
if (obj && SwigPyObject_Check(obj)) {
const char *otype = (const char *) SwigPyObject_GetDesc(obj);
if (otype) {
PyErr_Format(PyExc_TypeError, "a '%s' is expected, 'SwigPyObject(%s)' is received",
type, otype);
return;
}
} else 
#endif      
{
const char *otype = (obj ? obj->ob_type->tp_name : 0); 
if (otype) {
PyObject *str = PyObject_Str(obj);
const char *cstr = str ? SWIG_Python_str_AsChar(str) : 0;
if (cstr) {
PyErr_Format(PyExc_TypeError, "a '%s' is expected, '%s(%s)' is received",
type, otype, cstr);
SWIG_Python_str_DelForPy3(cstr);
} else {
PyErr_Format(PyExc_TypeError, "a '%s' is expected, '%s' is received",
type, otype);
}
Py_XDECREF(str);
return;
}
}   
PyErr_Format(PyExc_TypeError, "a '%s' is expected", type);
} else {
PyErr_Format(PyExc_TypeError, "unexpected type is received");
}
}



SWIGRUNTIME void *
SWIG_Python_MustGetPtr(PyObject *obj, swig_type_info *ty, int SWIGUNUSEDPARM(argnum), int flags) {
void *result;
if (SWIG_Python_ConvertPtr(obj, &result, ty, flags) == -1) {
PyErr_Clear();
#if SWIG_POINTER_EXCEPTION
if (flags) {
SWIG_Python_TypeError(SWIG_TypePrettyName(ty), obj);
SWIG_Python_ArgFail(argnum);
}
#endif
}
return result;
}

#ifdef SWIGPYTHON_BUILTIN
SWIGRUNTIME int
SWIG_Python_NonDynamicSetAttr(PyObject *obj, PyObject *name, PyObject *value) {
PyTypeObject *tp = obj->ob_type;
PyObject *descr;
PyObject *encoded_name;
descrsetfunc f;
int res = -1;

# ifdef Py_USING_UNICODE
if (PyString_Check(name)) {
name = PyUnicode_Decode(PyString_AsString(name), PyString_Size(name), NULL, NULL);
if (!name)
return -1;
} else if (!PyUnicode_Check(name))
# else
if (!PyString_Check(name))
# endif
{
PyErr_Format(PyExc_TypeError, "attribute name must be string, not '%.200s'", name->ob_type->tp_name);
return -1;
} else {
Py_INCREF(name);
}

if (!tp->tp_dict) {
if (PyType_Ready(tp) < 0)
goto done;
}

descr = _PyType_Lookup(tp, name);
f = NULL;
if (descr != NULL)
f = descr->ob_type->tp_descr_set;
if (!f) {
if (PyString_Check(name)) {
encoded_name = name;
Py_INCREF(name);
} else {
encoded_name = PyUnicode_AsUTF8String(name);
}
PyErr_Format(PyExc_AttributeError, "'%.100s' object has no attribute '%.200s'", tp->tp_name, PyString_AsString(encoded_name));
Py_DECREF(encoded_name);
} else {
res = f(descr, obj, value);
}

done:
Py_DECREF(name);
return res;
}
#endif


#ifdef __cplusplus
}
#endif



#define SWIG_exception_fail(code, msg) do { SWIG_Error(code, msg); SWIG_fail; } while(0) 

#define SWIG_contract_assert(expr, msg) if (!(expr)) { SWIG_Error(SWIG_RuntimeError, msg); SWIG_fail; } else 





#define SWIGTYPE_p_char swig_types[0]
#define SWIGTYPE_p_kitty swig_types[1]
static swig_type_info *swig_types[3];
static swig_module_info swig_module = {swig_types, 2, 0, 0, 0, 0};
#define SWIG_TypeQuery(name) SWIG_TypeQueryModule(&swig_module, &swig_module, name)
#define SWIG_MangledTypeQuery(name) SWIG_MangledTypeQueryModule(&swig_module, &swig_module, name)



#if (PY_VERSION_HEX <= 0x02000000)
# if !defined(SWIG_PYTHON_CLASSIC)
#  error "This python version requires swig to be run with the '-classic' option"
# endif
#endif


#if PY_VERSION_HEX >= 0x03000000
#  define SWIG_init    PyInit__kitty

#else
#  define SWIG_init    init_kitty

#endif
#define SWIG_name    "_kitty"

#define SWIGVERSION 0x020011 
#define SWIG_VERSION SWIGVERSION


#define SWIG_as_voidptr(a) const_cast< void * >(static_cast< const void * >(a)) 
#define SWIG_as_voidptrptr(a) ((void)SWIG_as_voidptr(*a),reinterpret_cast< void** >(a)) 


#include <stdexcept>


namespace swig {
class SwigPtr_PyObject {
protected:
PyObject *_obj;

public:
SwigPtr_PyObject() :_obj(0)
{
}

SwigPtr_PyObject(const SwigPtr_PyObject& item) : _obj(item._obj)
{
Py_XINCREF(_obj);      
}

SwigPtr_PyObject(PyObject *obj, bool initial_ref = true) :_obj(obj)
{
if (initial_ref) {
Py_XINCREF(_obj);
}
}

SwigPtr_PyObject & operator=(const SwigPtr_PyObject& item) 
{
Py_XINCREF(item._obj);
Py_XDECREF(_obj);
_obj = item._obj;
return *this;      
}

~SwigPtr_PyObject() 
{
Py_XDECREF(_obj);
}

operator PyObject *() const
{
return _obj;
}

PyObject *operator->() const
{
return _obj;
}
};
}


namespace swig {
struct SwigVar_PyObject : SwigPtr_PyObject {
SwigVar_PyObject(PyObject* obj = 0) : SwigPtr_PyObject(obj, false) { }

SwigVar_PyObject & operator = (PyObject* obj)
{
Py_XDECREF(_obj);
_obj = obj;
return *this;      
}
};
}


#include "kitty.hpp"

#ifdef __cplusplus
extern "C" {
#endif
SWIGINTERN PyObject *_wrap_new_kitty(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {
PyObject *resultobj = 0;
kitty *result = 0 ;

if (!PyArg_ParseTuple(args,(char *)":new_kitty")) SWIG_fail;
result = (kitty *)new kitty();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_kitty, SWIG_POINTER_NEW |  0 );
return resultobj;
fail:
return NULL;
}


SWIGINTERN PyObject *_wrap_delete_kitty(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {
PyObject *resultobj = 0;
kitty *arg1 = (kitty *) 0 ;
void *argp1 = 0 ;
int res1 = 0 ;
PyObject * obj0 = 0 ;

if (!PyArg_ParseTuple(args,(char *)"O:delete_kitty",&obj0)) SWIG_fail;
res1 = SWIG_ConvertPtr(obj0, &argp1,SWIGTYPE_p_kitty, SWIG_POINTER_DISOWN |  0 );
if (!SWIG_IsOK(res1)) {
SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "delete_kitty" "', argument " "1"" of type '" "kitty *""'"); 
}
arg1 = reinterpret_cast< kitty * >(argp1);
delete arg1;
resultobj = SWIG_Py_Void();
return resultobj;
fail:
return NULL;
}


SWIGINTERN PyObject *_wrap_kitty_speak(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {
PyObject *resultobj = 0;
kitty *arg1 = (kitty *) 0 ;
void *argp1 = 0 ;
int res1 = 0 ;
PyObject * obj0 = 0 ;

if (!PyArg_ParseTuple(args,(char *)"O:kitty_speak",&obj0)) SWIG_fail;
res1 = SWIG_ConvertPtr(obj0, &argp1,SWIGTYPE_p_kitty, 0 |  0 );
if (!SWIG_IsOK(res1)) {
SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "kitty_speak" "', argument " "1"" of type '" "kitty *""'"); 
}
arg1 = reinterpret_cast< kitty * >(argp1);
(arg1)->speak();
resultobj = SWIG_Py_Void();
return resultobj;
fail:
return NULL;
}


SWIGINTERN PyObject *_wrap_kitty_speak2(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {
PyObject *resultobj = 0;
kitty *arg1 = (kitty *) 0 ;
void *argp1 = 0 ;
int res1 = 0 ;
PyObject * obj0 = 0 ;

if (!PyArg_ParseTuple(args,(char *)"O:kitty_speak2",&obj0)) SWIG_fail;
res1 = SWIG_ConvertPtr(obj0, &argp1,SWIGTYPE_p_kitty, 0 |  0 );
if (!SWIG_IsOK(res1)) {
SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "kitty_speak2" "', argument " "1"" of type '" "kitty *""'"); 
}
arg1 = reinterpret_cast< kitty * >(argp1);
(arg1)->speak2();
resultobj = SWIG_Py_Void();
return resultobj;
fail:
return NULL;
}


SWIGINTERN PyObject *kitty_swigregister(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {
PyObject *obj;
if (!PyArg_ParseTuple(args,(char*)"O:swigregister", &obj)) return NULL;
SWIG_TypeNewClientData(SWIGTYPE_p_kitty, SWIG_NewClientData(obj));
return SWIG_Py_Void();
}

static PyMethodDef SwigMethods[] = {
{ (char *)"SWIG_PyInstanceMethod_New", (PyCFunction)SWIG_PyInstanceMethod_New, METH_O, NULL},
{ (char *)"new_kitty", _wrap_new_kitty, METH_VARARGS, NULL},
{ (char *)"delete_kitty", _wrap_delete_kitty, METH_VARARGS, NULL},
{ (char *)"kitty_speak", _wrap_kitty_speak, METH_VARARGS, NULL},
{ (char *)"kitty_speak2", _wrap_kitty_speak2, METH_VARARGS, NULL},
{ (char *)"kitty_swigregister", kitty_swigregister, METH_VARARGS, NULL},
{ NULL, NULL, 0, NULL }
};




static swig_type_info _swigt__p_char = {"_p_char", "char *", 0, 0, (void*)0, 0};
static swig_type_info _swigt__p_kitty = {"_p_kitty", "kitty *", 0, 0, (void*)0, 0};

static swig_type_info *swig_type_initial[] = {
&_swigt__p_char,
&_swigt__p_kitty,
};

static swig_cast_info _swigc__p_char[] = {  {&_swigt__p_char, 0, 0, 0},{0, 0, 0, 0}};
static swig_cast_info _swigc__p_kitty[] = {  {&_swigt__p_kitty, 0, 0, 0},{0, 0, 0, 0}};

static swig_cast_info *swig_cast_initial[] = {
_swigc__p_char,
_swigc__p_kitty,
};




static swig_const_info swig_const_table[] = {
{0, 0, 0, 0.0, 0, 0}};

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#if 0
} 
#endif
#endif

#if 0
#define SWIGRUNTIME_DEBUG
#endif


SWIGRUNTIME void
SWIG_InitializeModule(void *clientdata) {
size_t i;
swig_module_info *module_head, *iter;
int found, init;


if (swig_module.next==0) {

swig_module.type_initial = swig_type_initial;
swig_module.cast_initial = swig_cast_initial;
swig_module.next = &swig_module;
init = 1;
} else {
init = 0;
}


module_head = SWIG_GetModule(clientdata);
if (!module_head) {


SWIG_SetModule(clientdata, &swig_module);
module_head = &swig_module;
} else {

found=0;
iter=module_head;
do {
if (iter==&swig_module) {
found=1;
break;
}
iter=iter->next;
} while (iter!= module_head);


if (found) return;

swig_module.next = module_head->next;
module_head->next = &swig_module;
}


if (init == 0) return;


#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: size %d\n", swig_module.size);
#endif
for (i = 0; i < swig_module.size; ++i) {
swig_type_info *type = 0;
swig_type_info *ret;
swig_cast_info *cast;

#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: type %d %s\n", i, swig_module.type_initial[i]->name);
#endif


if (swig_module.next != &swig_module) {
type = SWIG_MangledTypeQueryModule(swig_module.next, &swig_module, swig_module.type_initial[i]->name);
}
if (type) {

#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: found type %s\n", type->name);
#endif
if (swig_module.type_initial[i]->clientdata) {
type->clientdata = swig_module.type_initial[i]->clientdata;
#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: found and overwrite type %s \n", type->name);
#endif
}
} else {
type = swig_module.type_initial[i];
}


cast = swig_module.cast_initial[i];
while (cast->type) {

ret = 0;
#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: look cast %s\n", cast->type->name);
#endif
if (swig_module.next != &swig_module) {
ret = SWIG_MangledTypeQueryModule(swig_module.next, &swig_module, cast->type->name);
#ifdef SWIGRUNTIME_DEBUG
if (ret) printf("SWIG_InitializeModule: found cast %s\n", ret->name);
#endif
}
if (ret) {
if (type == swig_module.type_initial[i]) {
#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: skip old type %s\n", ret->name);
#endif
cast->type = ret;
ret = 0;
} else {

swig_cast_info *ocast = SWIG_TypeCheck(ret->name, type);
#ifdef SWIGRUNTIME_DEBUG
if (ocast) printf("SWIG_InitializeModule: skip old cast %s\n", ret->name);
#endif
if (!ocast) ret = 0;
}
}

if (!ret) {
#ifdef SWIGRUNTIME_DEBUG
printf("SWIG_InitializeModule: adding cast %s\n", cast->type->name);
#endif
if (type->cast) {
type->cast->prev = cast;
cast->next = type->cast;
}
type->cast = cast;
}
cast++;
}

swig_module.types[i] = type;
}
swig_module.types[i] = 0;

#ifdef SWIGRUNTIME_DEBUG
printf("**** SWIG_InitializeModule: Cast List ******\n");
for (i = 0; i < swig_module.size; ++i) {
int j = 0;
swig_cast_info *cast = swig_module.cast_initial[i];
printf("SWIG_InitializeModule: type %d %s\n", i, swig_module.type_initial[i]->name);
while (cast->type) {
printf("SWIG_InitializeModule: cast type %s\n", cast->type->name);
cast++;
++j;
}
printf("---- Total casts: %d\n",j);
}
printf("**** SWIG_InitializeModule: Cast List ******\n");
#endif
}


SWIGRUNTIME void
SWIG_PropagateClientData(void) {
size_t i;
swig_cast_info *equiv;
static int init_run = 0;

if (init_run) return;
init_run = 1;

for (i = 0; i < swig_module.size; i++) {
if (swig_module.types[i]->clientdata) {
equiv = swig_module.types[i]->cast;
while (equiv) {
if (!equiv->converter) {
if (equiv->type && !equiv->type->clientdata)
SWIG_TypeClientData(equiv->type, swig_module.types[i]->clientdata);
}
equiv = equiv->next;
}
}
}
}

#ifdef __cplusplus
#if 0
{

#endif
}
#endif



#ifdef __cplusplus
extern "C" {
#endif


#define SWIG_newvarlink()                             SWIG_Python_newvarlink()
#define SWIG_addvarlink(p, name, get_attr, set_attr)  SWIG_Python_addvarlink(p, name, get_attr, set_attr)
#define SWIG_InstallConstants(d, constants)           SWIG_Python_InstallConstants(d, constants)



typedef struct swig_globalvar {
char       *name;                  
PyObject *(*get_attr)(void);       
int       (*set_attr)(PyObject *); 
struct swig_globalvar *next;
} swig_globalvar;

typedef struct swig_varlinkobject {
PyObject_HEAD
swig_globalvar *vars;
} swig_varlinkobject;

SWIGINTERN PyObject *
swig_varlink_repr(swig_varlinkobject *SWIGUNUSEDPARM(v)) {
#if PY_VERSION_HEX >= 0x03000000
return PyUnicode_InternFromString("<Swig global variables>");
#else
return PyString_FromString("<Swig global variables>");
#endif
}

SWIGINTERN PyObject *
swig_varlink_str(swig_varlinkobject *v) {
#if PY_VERSION_HEX >= 0x03000000
PyObject *str = PyUnicode_InternFromString("(");
PyObject *tail;
PyObject *joined;
swig_globalvar *var;
for (var = v->vars; var; var=var->next) {
tail = PyUnicode_FromString(var->name);
joined = PyUnicode_Concat(str, tail);
Py_DecRef(str);
Py_DecRef(tail);
str = joined;
if (var->next) {
tail = PyUnicode_InternFromString(", ");
joined = PyUnicode_Concat(str, tail);
Py_DecRef(str);
Py_DecRef(tail);
str = joined;
}
}
tail = PyUnicode_InternFromString(")");
joined = PyUnicode_Concat(str, tail);
Py_DecRef(str);
Py_DecRef(tail);
str = joined;
#else
PyObject *str = PyString_FromString("(");
swig_globalvar *var;
for (var = v->vars; var; var=var->next) {
PyString_ConcatAndDel(&str,PyString_FromString(var->name));
if (var->next) PyString_ConcatAndDel(&str,PyString_FromString(", "));
}
PyString_ConcatAndDel(&str,PyString_FromString(")"));
#endif
return str;
}

SWIGINTERN int
swig_varlink_print(swig_varlinkobject *v, FILE *fp, int SWIGUNUSEDPARM(flags)) {
char *tmp;
PyObject *str = swig_varlink_str(v);
fprintf(fp,"Swig global variables ");
fprintf(fp,"%s\n", tmp = SWIG_Python_str_AsChar(str));
SWIG_Python_str_DelForPy3(tmp);
Py_DECREF(str);
return 0;
}

SWIGINTERN void
swig_varlink_dealloc(swig_varlinkobject *v) {
swig_globalvar *var = v->vars;
while (var) {
swig_globalvar *n = var->next;
free(var->name);
free(var);
var = n;
}
}

SWIGINTERN PyObject *
swig_varlink_getattr(swig_varlinkobject *v, char *n) {
PyObject *res = NULL;
swig_globalvar *var = v->vars;
while (var) {
if (strcmp(var->name,n) == 0) {
res = (*var->get_attr)();
break;
}
var = var->next;
}
if (res == NULL && !PyErr_Occurred()) {
PyErr_SetString(PyExc_NameError,"Unknown C global variable");
}
return res;
}

SWIGINTERN int
swig_varlink_setattr(swig_varlinkobject *v, char *n, PyObject *p) {
int res = 1;
swig_globalvar *var = v->vars;
while (var) {
if (strcmp(var->name,n) == 0) {
res = (*var->set_attr)(p);
break;
}
var = var->next;
}
if (res == 1 && !PyErr_Occurred()) {
PyErr_SetString(PyExc_NameError,"Unknown C global variable");
}
return res;
}

SWIGINTERN PyTypeObject*
swig_varlink_type(void) {
static char varlink__doc__[] = "Swig var link object";
static PyTypeObject varlink_type;
static int type_init = 0;
if (!type_init) {
const PyTypeObject tmp = {

#if PY_VERSION_HEX >= 0x03000000
PyVarObject_HEAD_INIT(NULL, 0)
#else
PyObject_HEAD_INIT(NULL)
0,                                  
#endif
(char *)"swigvarlink",              
sizeof(swig_varlinkobject),         
0,                                  
(destructor) swig_varlink_dealloc,  
(printfunc) swig_varlink_print,     
(getattrfunc) swig_varlink_getattr, 
(setattrfunc) swig_varlink_setattr, 
0,                                  
(reprfunc) swig_varlink_repr,       
0,                                  
0,                                  
0,                                  
0,                                  
0,                                  
(reprfunc) swig_varlink_str,        
0,                                  
0,                                  
0,                                  
0,                                  
varlink__doc__,                     
0,                                  
0,                                  
0,                                  
0,                                  
#if PY_VERSION_HEX >= 0x02020000
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
#endif
#if PY_VERSION_HEX >= 0x02030000
0,                                  
#endif
#if PY_VERSION_HEX >= 0x02060000
0,                                  
#endif
#ifdef COUNT_ALLOCS
0,0,0,0                             
#endif
};
varlink_type = tmp;
type_init = 1;
#if PY_VERSION_HEX < 0x02020000
varlink_type.ob_type = &PyType_Type;
#else
if (PyType_Ready(&varlink_type) < 0)
return NULL;
#endif
}
return &varlink_type;
}


SWIGINTERN PyObject *
SWIG_Python_newvarlink(void) {
swig_varlinkobject *result = PyObject_NEW(swig_varlinkobject, swig_varlink_type());
if (result) {
result->vars = 0;
}
return ((PyObject*) result);
}

SWIGINTERN void 
SWIG_Python_addvarlink(PyObject *p, char *name, PyObject *(*get_attr)(void), int (*set_attr)(PyObject *p)) {
swig_varlinkobject *v = (swig_varlinkobject *) p;
swig_globalvar *gv = (swig_globalvar *) malloc(sizeof(swig_globalvar));
if (gv) {
size_t size = strlen(name)+1;
gv->name = (char *)malloc(size);
if (gv->name) {
strncpy(gv->name,name,size);
gv->get_attr = get_attr;
gv->set_attr = set_attr;
gv->next = v->vars;
}
}
v->vars = gv;
}

SWIGINTERN PyObject *
SWIG_globals(void) {
static PyObject *_SWIG_globals = 0; 
if (!_SWIG_globals) _SWIG_globals = SWIG_newvarlink();  
return _SWIG_globals;
}




SWIGINTERN void
SWIG_Python_InstallConstants(PyObject *d, swig_const_info constants[]) {
PyObject *obj = 0;
size_t i;
for (i = 0; constants[i].type; ++i) {
switch(constants[i].type) {
case SWIG_PY_POINTER:
obj = SWIG_InternalNewPointerObj(constants[i].pvalue, *(constants[i]).ptype,0);
break;
case SWIG_PY_BINARY:
obj = SWIG_NewPackedObj(constants[i].pvalue, constants[i].lvalue, *(constants[i].ptype));
break;
default:
obj = 0;
break;
}
if (obj) {
PyDict_SetItemString(d, constants[i].name, obj);
Py_DECREF(obj);
}
}
}





SWIGINTERN void
SWIG_Python_FixMethods(PyMethodDef *methods,
swig_const_info *const_table,
swig_type_info **types,
swig_type_info **types_initial) {
size_t i;
for (i = 0; methods[i].ml_name; ++i) {
const char *c = methods[i].ml_doc;
if (c && (c = strstr(c, "swig_ptr: "))) {
int j;
swig_const_info *ci = 0;
const char *name = c + 10;
for (j = 0; const_table[j].type; ++j) {
if (strncmp(const_table[j].name, name, 
strlen(const_table[j].name)) == 0) {
ci = &(const_table[j]);
break;
}
}
if (ci) {
void *ptr = (ci->type == SWIG_PY_POINTER) ? ci->pvalue : 0;
if (ptr) {
size_t shift = (ci->ptype) - types;
swig_type_info *ty = types_initial[shift];
size_t ldoc = (c - methods[i].ml_doc);
size_t lptr = strlen(ty->name)+2*sizeof(void*)+2;
char *ndoc = (char*)malloc(ldoc + lptr + 10);
if (ndoc) {
char *buff = ndoc;
strncpy(buff, methods[i].ml_doc, ldoc);
buff += ldoc;
strncpy(buff, "swig_ptr: ", 10);
buff += 10;
SWIG_PackVoidPtr(buff, ptr, ty->name, lptr);
methods[i].ml_doc = ndoc;
}
}
}
}
}
} 

#ifdef __cplusplus
}
#endif



#ifdef __cplusplus
extern "C"
#endif

SWIGEXPORT 
#if PY_VERSION_HEX >= 0x03000000
PyObject*
#else
void
#endif
SWIG_init(void) {
PyObject *m, *d, *md;
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef SWIG_module = {
# if PY_VERSION_HEX >= 0x03020000
PyModuleDef_HEAD_INIT,
# else
{
PyObject_HEAD_INIT(NULL)
NULL, 
0,    
NULL, 
},
# endif
(char *) SWIG_name,
NULL,
-1,
SwigMethods,
NULL,
NULL,
NULL,
NULL
};
#endif

#if defined(SWIGPYTHON_BUILTIN)
static SwigPyClientData SwigPyObject_clientdata = {
0, 0, 0, 0, 0, 0, 0
};
static PyGetSetDef this_getset_def = {
(char *)"this", &SwigPyBuiltin_ThisClosure, NULL, NULL, NULL
};
static SwigPyGetSet thisown_getset_closure = {
(PyCFunction) SwigPyObject_own,
(PyCFunction) SwigPyObject_own
};
static PyGetSetDef thisown_getset_def = {
(char *)"thisown", SwigPyBuiltin_GetterClosure, SwigPyBuiltin_SetterClosure, NULL, &thisown_getset_closure
};
PyObject *metatype_args;
PyTypeObject *builtin_pytype;
int builtin_base_count;
swig_type_info *builtin_basetype;
PyObject *tuple;
PyGetSetDescrObject *static_getset;
PyTypeObject *metatype;
SwigPyClientData *cd;
PyObject *public_interface, *public_symbol;
PyObject *this_descr;
PyObject *thisown_descr;
int i;

(void)builtin_pytype;
(void)builtin_base_count;
(void)builtin_basetype;
(void)tuple;
(void)static_getset;


metatype_args = Py_BuildValue("(s(O){})", "SwigPyObjectType", &PyType_Type);
assert(metatype_args);
metatype = (PyTypeObject *) PyType_Type.tp_call((PyObject *) &PyType_Type, metatype_args, NULL);
assert(metatype);
Py_DECREF(metatype_args);
metatype->tp_setattro = (setattrofunc) &SwigPyObjectType_setattro;
assert(PyType_Ready(metatype) >= 0);
#endif


SWIG_Python_FixMethods(SwigMethods, swig_const_table, swig_types, swig_type_initial);

#if PY_VERSION_HEX >= 0x03000000
m = PyModule_Create(&SWIG_module);
#else
m = Py_InitModule((char *) SWIG_name, SwigMethods);
#endif
md = d = PyModule_GetDict(m);
(void)md;

SWIG_InitializeModule(0);

#ifdef SWIGPYTHON_BUILTIN
SwigPyObject_stype = SWIG_MangledTypeQuery("_p_SwigPyObject");
assert(SwigPyObject_stype);
cd = (SwigPyClientData*) SwigPyObject_stype->clientdata;
if (!cd) {
SwigPyObject_stype->clientdata = &SwigPyObject_clientdata;
SwigPyObject_clientdata.pytype = SwigPyObject_TypeOnce();
} else if (SwigPyObject_TypeOnce()->tp_basicsize != cd->pytype->tp_basicsize) {
PyErr_SetString(PyExc_RuntimeError, "Import error: attempted to load two incompatible swig-generated modules.");
# if PY_VERSION_HEX >= 0x03000000
return NULL;
# else
return;
# endif
}


this_descr = PyDescr_NewGetSet(SwigPyObject_type(), &this_getset_def);
(void)this_descr;


thisown_descr = PyDescr_NewGetSet(SwigPyObject_type(), &thisown_getset_def);
(void)thisown_descr;

public_interface = PyList_New(0);
public_symbol = 0;
(void)public_symbol;

PyDict_SetItemString(md, "__all__", public_interface);
Py_DECREF(public_interface);
for (i = 0; SwigMethods[i].ml_name != NULL; ++i)
SwigPyBuiltin_AddPublicSymbol(public_interface, SwigMethods[i].ml_name);
for (i = 0; swig_const_table[i].name != 0; ++i)
SwigPyBuiltin_AddPublicSymbol(public_interface, swig_const_table[i].name);
#endif

SWIG_InstallConstants(d,swig_const_table);

#if PY_VERSION_HEX >= 0x03000000
return m;
#else
return;
#endif
}

