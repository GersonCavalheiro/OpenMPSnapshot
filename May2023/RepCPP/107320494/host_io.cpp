

#include "host_io.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "rocalution/version.hpp"

#include <cinttypes>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace rocalution
{

struct mm_banner
{
char array_type[64];
char matrix_type[64];
char storage_type[64];
};

bool mm_read_banner(FILE* fin, mm_banner& b)
{
char line[1025];

if(!fgets(line, 1025, fin))
{
return false;
}

char banner[64];
char mtx[64];

if(sscanf(line, "%s %s %s %s %s", banner, mtx, b.array_type, b.matrix_type, b.storage_type)
!= 5)
{
return false;
}

for(char *p = mtx;            *p != '\0'; *p = tolower(*p), ++p);
for(char *p = b.array_type;   *p != '\0'; *p = tolower(*p), ++p);
for(char *p = b.matrix_type;  *p != '\0'; *p = tolower(*p), ++p);
for(char *p = b.storage_type; *p != '\0'; *p = tolower(*p), ++p);

if(strncmp(banner, "%%MatrixMarket", 14))
{
return false;
}

if(strncmp(mtx, "matrix", 6))
{
return false;
}

if(strncmp(b.array_type, "coordinate", 10))
{
return false;
}

if(strncmp(b.matrix_type, "real", 4) && strncmp(b.matrix_type, "complex", 7)
&& strncmp(b.matrix_type, "integer", 7) && strncmp(b.matrix_type, "pattern", 7))
{
return false;
}

if(strncmp(b.storage_type, "general", 7) && strncmp(b.storage_type, "symmetric", 9)
&& strncmp(b.storage_type, "hermitian", 9))
{
return false;
}

return true;
}

template <typename ValueType,
typename std::enable_if<std::is_same<ValueType, float>::value
|| std::is_same<ValueType, double>::value,
int>::type
= 0>
static ValueType read_complex(double real, double imag)
{
return static_cast<ValueType>(real);
}

template <typename ValueType,
typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value
|| std::is_same<ValueType, std::complex<double>>::value,
int>::type
= 0>
static ValueType read_complex(double real, double imag)
{
return ValueType(real, imag);
}

template <typename ValueType>
bool mm_read_coordinate(FILE*       fin,
mm_banner&  b,
int&        nrow,
int&        ncol,
int64_t&    nnz,
int**       row,
int**       col,
ValueType** val)
{
char line[1025];

do
{
if(!fgets(line, 1025, fin))
{
return false;
}
} while(line[0] == '%');

while(sscanf(line, "%d %d %" SCNd64, &nrow, &ncol, &nnz) != 3)
{
if(!fgets(line, 1025, fin))
{
return false;
}
}

allocate_host(nnz, row);
allocate_host(nnz, col);
allocate_host(nnz, val);

if(!strncmp(b.matrix_type, "complex", 7))
{
double real, imag;
for(int64_t i = 0; i < nnz; ++i)
{
if(fscanf(fin, "%d %d %lg %lg", (*row) + i, (*col) + i, &real, &imag) != 4)
{
return false;
}
--(*row)[i];
--(*col)[i];
(*val)[i] = read_complex<ValueType>(real, imag);
}
}
else if(!strncmp(b.matrix_type, "real", 4) || !strncmp(b.matrix_type, "integer", 7))
{
double tmp;
for(int64_t i = 0; i < nnz; ++i)
{
if(fscanf(fin, "%d %d %lg\n", (*row) + i, (*col) + i, &tmp) != 3)
{
return false;
}
--(*row)[i];
--(*col)[i];
(*val)[i] = read_complex<ValueType>(tmp, tmp);
}
}
else if(!strncmp(b.matrix_type, "pattern", 7))
{
for(int64_t i = 0; i < nnz; ++i)
{
if(fscanf(fin, "%d %d\n", (*row) + i, (*col) + i) != 2)
{
return false;
}
--(*row)[i];
--(*col)[i];
(*val)[i] = static_cast<ValueType>(1);
}
}
else
{
return false;
}

if(strncmp(b.storage_type, "general", 7))
{
int ndiag = 0;
for(int64_t i = 0; i < nnz; ++i)
{
if((*row)[i] == (*col)[i])
{
++ndiag;
}
}

int64_t tot_nnz = (nnz - ndiag) * 2 + ndiag;

int*       sym_row = *row;
int*       sym_col = *col;
ValueType* sym_val = *val;

*row = NULL;
*col = NULL;
*val = NULL;

allocate_host(tot_nnz, row);
allocate_host(tot_nnz, col);
allocate_host(tot_nnz, val);

int64_t idx = 0;
for(int64_t i = 0; i < nnz; ++i)
{
(*row)[idx] = sym_row[i];
(*col)[idx] = sym_col[i];
(*val)[idx] = sym_val[i];
++idx;

if(sym_row[i] != sym_col[i])
{
(*row)[idx] = sym_col[i];
(*col)[idx] = sym_row[i];
(*val)[idx] = sym_val[i];
++idx;
}
}

if(idx != tot_nnz)
{
return false;
}

nnz = tot_nnz;

free_host(&sym_row);
free_host(&sym_col);
free_host(&sym_val);
}

return true;
}

template <typename ValueType>
bool read_matrix_mtx(int&        nrow,
int&        ncol,
int64_t&    nnz,
int**       row,
int**       col,
ValueType** val,
const char* filename)
{
FILE* file = fopen(filename, "r");

if(!file)
{
LOG_INFO("ReadFileMTX: cannot open file " << filename);
return false;
}

mm_banner banner;
if(mm_read_banner(file, banner) != true)
{
LOG_INFO("ReadFileMTX: invalid matrix market banner");
return false;
}

if(strncmp(banner.array_type, "coordinate", 10))
{
return false;
}
else
{
if(mm_read_coordinate(file, banner, nrow, ncol, nnz, row, col, val) != true)
{
LOG_INFO("ReadFileMTX: invalid matrix data");
return false;
}
}

fclose(file);

return true;
}

template <typename ValueType,
typename std::enable_if<std::is_same<ValueType, float>::value
|| std::is_same<ValueType, double>::value,
int>::type
= 0>
void write_banner(FILE* file)
{
char sign[3];
strcpy(sign, "%%");

fprintf(file, "%sMatrixMarket matrix coordinate real general\n", sign);
}

template <typename ValueType,
typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value
|| std::is_same<ValueType, std::complex<double>>::value,
int>::type
= 0>
void write_banner(FILE* file)
{
char sign[3];
strcpy(sign, "%%");

fprintf(file, "%sMatrixMarket matrix coordinate complex general\n", sign);
}

template <typename ValueType,
typename std::enable_if<std::is_same<ValueType, float>::value, int>::type = 0>
void write_value(FILE* file, ValueType val)
{
fprintf(file, "%0.12g\n", val);
}

template <typename ValueType,
typename std::enable_if<std::is_same<ValueType, double>::value, int>::type = 0>
void write_value(FILE* file, ValueType val)
{
fprintf(file, "%0.12lg\n", val);
}

template <
typename ValueType,
typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value, int>::type = 0>
void write_value(FILE* file, ValueType val)
{
fprintf(file, "%0.12g %0.12g\n", val.real(), val.imag());
}

template <
typename ValueType,
typename std::enable_if<std::is_same<ValueType, std::complex<double>>::value, int>::type
= 0>
void write_value(FILE* file, ValueType val)
{
fprintf(file, "%0.12lg %0.12lg\n", val.real(), val.imag());
}

template <typename ValueType>
bool write_matrix_mtx(int              nrow,
int              ncol,
int64_t          nnz,
const int*       row,
const int*       col,
const ValueType* val,
const char*      filename)
{
FILE* file = fopen(filename, "w");

if(!file)
{
LOG_INFO("WriteFileMTX: cannot open file " << filename);
return false;
}

write_banner<ValueType>(file);

fprintf(file, "%d %d %" PRId64 "\n", nrow, ncol, nnz);

for(int64_t i = 0; i < nnz; ++i)
{
fprintf(file, "%d %d ", row[i] + 1, col[i] + 1);
write_value(file, val[i]);
}

fclose(file);

return true;
}

static inline void read_csr_values(std::ifstream& in, int64_t nnz, float* val)
{
std::vector<double> tmp(nnz);

in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < nnz; ++i)
{
val[i] = static_cast<float>(tmp[i]);
}
}

static inline void read_csr_values(std::ifstream& in, int64_t nnz, double* val)
{
in.read((char*)val, sizeof(double) * nnz);
}

static inline void read_csr_values(std::ifstream& in, int64_t nnz, std::complex<float>* val)
{
std::vector<std::complex<double>> tmp(nnz);

in.read((char*)tmp.data(), sizeof(std::complex<double>) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < nnz; ++i)
{
val[i] = std::complex<float>(static_cast<float>(tmp[i].real()),
static_cast<float>(tmp[i].imag()));
}
}

static inline void read_csr_values(std::ifstream& in, int64_t nnz, std::complex<double>* val)
{
in.read((char*)val, sizeof(std::complex<double>) * nnz);
}

static inline void read_csr_row_ptr_32(std::ifstream& in, int64_t nrow, int* ptr)
{
in.read((char*)ptr, sizeof(int) * (nrow + 1));
}

static inline void read_csr_row_ptr_32(std::ifstream& in, int64_t nrow, int64_t* ptr)
{
std::vector<int> tmp(nrow + 1);

in.read((char*)tmp.data(), sizeof(int) * (nrow + 1));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < nrow + 1; ++i)
{
ptr[i] = static_cast<int64_t>(tmp[i]);
}
}

static inline void read_csr_row_ptr_64(std::ifstream& in, int64_t nrow, int* ptr)
{
LOG_INFO("ReadFileCSR: cannot read 64 bit sparsity pattern into 32 bit structure");
FATAL_ERROR(__FILE__, __LINE__);
}

static inline void read_csr_row_ptr_64(std::ifstream& in, int64_t nrow, int64_t* ptr)
{
in.read((char*)ptr, sizeof(int64_t) * (nrow + 1));
}

template <typename ValueType, typename IndexType, typename PointerType>
bool read_matrix_csr(int64_t&      nrow,
int64_t&      ncol,
int64_t&      nnz,
PointerType** ptr,
IndexType**   col,
ValueType**   val,
const char*   filename)
{
std::ifstream in(filename, std::ios::in | std::ios::binary);

if(!in.is_open())
{
LOG_INFO("ReadFileCSR: cannot open file " << filename);
return false;
}

std::string header;
std::getline(in, header);

if(header != "#rocALUTION binary csr file")
{
LOG_INFO("ReadFileCSR: invalid rocALUTION matrix header");
return false;
}

int version;
in.read((char*)&version, sizeof(int));


if(version < 30000)
{
int nrow32;
int ncol32;
int nnz32;

in.read((char*)&nrow32, sizeof(int));
in.read((char*)&ncol32, sizeof(int));
in.read((char*)&nnz32, sizeof(int));

nrow = static_cast<int64_t>(nrow32);
ncol = static_cast<int64_t>(ncol32);
nnz  = static_cast<int64_t>(nnz32);

int* ptr32 = NULL;

allocate_host(nrow32 + 1, &ptr32);
allocate_host(nrow + 1, ptr);

in.read((char*)ptr32, (nrow32 + 1) * sizeof(int));

for(int i = 0; i < nrow32 + 1; ++i)
{
(*ptr)[i] = ptr32[i];
}

free_host(&ptr32);
}
else
{
in.read((char*)&nrow, sizeof(int64_t));
in.read((char*)&ncol, sizeof(int64_t));
in.read((char*)&nnz, sizeof(int64_t));

allocate_host(nrow + 1, ptr);

if(nnz < std::numeric_limits<int>::max())
{
read_csr_row_ptr_32(in, nrow, *ptr);
}
else
{
read_csr_row_ptr_64(in, nrow, *ptr);
}
}


allocate_host(nnz, col);
allocate_host(nnz, val);

in.read((char*)*col, nnz * sizeof(int));
read_csr_values(in, nnz, *val);

if(!in)
{
LOG_INFO("ReadFileCSR: invalid matrix data");
return false;
}

in.close();

return true;
}

static inline void write_csr_values(std::ofstream& out, int64_t nnz, const float* val)
{
std::vector<double> tmp(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < nnz; ++i)
{
tmp[i] = static_cast<double>(val[i]);
}

out.write((char*)tmp.data(), sizeof(double) * nnz);
}

static inline void write_csr_values(std::ofstream& out, int64_t nnz, const double* val)
{
out.write((char*)val, sizeof(double) * nnz);
}

static inline void
write_csr_values(std::ofstream& out, int64_t nnz, const std::complex<float>* val)
{
std::vector<std::complex<double>> tmp(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < nnz; ++i)
{
tmp[i] = std::complex<double>(static_cast<double>(val[i].real()),
static_cast<double>(val[i].imag()));
}

out.write((char*)tmp.data(), sizeof(std::complex<double>) * nnz);
}

static inline void
write_csr_values(std::ofstream& out, int64_t nnz, const std::complex<double>* val)
{
out.write((char*)val, sizeof(std::complex<double>) * nnz);
}

static inline void write_csr_row_ptr_32(std::ofstream& out, int64_t nrow, const int* ptr)
{
out.write((char*)ptr, (nrow + 1) * sizeof(int));
}

static inline void write_csr_row_ptr_32(std::ofstream& out, int64_t nrow, const int64_t* ptr)
{
std::vector<int> tmp(nrow + 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < nrow + 1; ++i)
{
tmp[i] = static_cast<int>(ptr[i]);
}

out.write((char*)tmp.data(), sizeof(int) * (nrow + 1));
}

static inline void write_csr_row_ptr_64(std::ofstream& out, int64_t nrow, const int* ptr)
{
LOG_INFO("This function should never be called");
FATAL_ERROR(__FILE__, __LINE__);
}

static inline void write_csr_row_ptr_64(std::ofstream& out, int64_t nrow, const int64_t* ptr)
{
out.write((char*)ptr, (nrow + 1) * sizeof(int64_t));
}

template <typename ValueType, typename IndexType, typename PointerType>
bool write_matrix_csr(int64_t            nrow,
int64_t            ncol,
int64_t            nnz,
const PointerType* ptr,
const IndexType*   col,
const ValueType*   val,
const char*        filename)
{
std::ofstream out(filename, std::ios::out | std::ios::binary);

if(!out.is_open())
{
LOG_INFO("WriteFileCSR: cannot open file " << filename);
return false;
}

out << "#rocALUTION binary csr file" << std::endl;

int version = __ROCALUTION_VER;
out.write((char*)&version, sizeof(int));

out.write((char*)&nrow, sizeof(int64_t));
out.write((char*)&ncol, sizeof(int64_t));
out.write((char*)&nnz, sizeof(int64_t));

if(nnz <= std::numeric_limits<int>::max())
{
write_csr_row_ptr_32(out, nrow, ptr);
}
else
{
write_csr_row_ptr_64(out, nrow, ptr);
}

out.write((char*)col, nnz * sizeof(int));

write_csr_values(out, nnz, val);

if(!out)
{
LOG_INFO("WriteFileCSR: filename=" << filename << "; could not write to file");
return false;
}

out.close();

return true;
}

template bool read_matrix_mtx(int&        nrow,
int&        ncol,
int64_t&    nnz,
int**       row,
int**       col,
float**     val,
const char* filename);
template bool read_matrix_mtx(int&        nrow,
int&        ncol,
int64_t&    nnz,
int**       row,
int**       col,
double**    val,
const char* filename);
#ifdef SUPPORT_COMPLEX
template bool read_matrix_mtx(int&                  nrow,
int&                  ncol,
int64_t&              nnz,
int**                 row,
int**                 col,
std::complex<float>** val,
const char*           filename);
template bool read_matrix_mtx(int&                   nrow,
int&                   ncol,
int64_t&               nnz,
int**                  row,
int**                  col,
std::complex<double>** val,
const char*            filename);
#endif

template bool write_matrix_mtx(int          nrow,
int          ncol,
int64_t      nnz,
const int*   row,
const int*   col,
const float* val,
const char*  filename);
template bool write_matrix_mtx(int           nrow,
int           ncol,
int64_t       nnz,
const int*    row,
const int*    col,
const double* val,
const char*   filename);
#ifdef SUPPORT_COMPLEX
template bool write_matrix_mtx(int                        nrow,
int                        ncol,
int64_t                    nnz,
const int*                 row,
const int*                 col,
const std::complex<float>* val,
const char*                filename);
template bool write_matrix_mtx(int                         nrow,
int                         ncol,
int64_t                     nnz,
const int*                  row,
const int*                  col,
const std::complex<double>* val,
const char*                 filename);
#endif

template bool read_matrix_csr(int64_t&    nrow,
int64_t&    ncol,
int64_t&    nnz,
int**       ptr,
int**       col,
float**     val,
const char* filename);
template bool read_matrix_csr(int64_t&    nrow,
int64_t&    ncol,
int64_t&    nnz,
int**       ptr,
int**       col,
double**    val,
const char* filename);
#ifdef SUPPORT_COMPLEX
template bool read_matrix_csr(int64_t&              nrow,
int64_t&              ncol,
int64_t&              nnz,
int**                 ptr,
int**                 col,
std::complex<float>** val,
const char*           filename);
template bool read_matrix_csr(int64_t&               nrow,
int64_t&               ncol,
int64_t&               nnz,
int**                  ptr,
int**                  col,
std::complex<double>** val,
const char*            filename);
#endif

template bool read_matrix_csr(int64_t&    nrow,
int64_t&    ncol,
int64_t&    nnz,
int64_t**   ptr,
int**       col,
float**     val,
const char* filename);
template bool read_matrix_csr(int64_t&    nrow,
int64_t&    ncol,
int64_t&    nnz,
int64_t**   ptr,
int**       col,
double**    val,
const char* filename);
#ifdef SUPPORT_COMPLEX
template bool read_matrix_csr(int64_t&              nrow,
int64_t&              ncol,
int64_t&              nnz,
int64_t**             ptr,
int**                 col,
std::complex<float>** val,
const char*           filename);
template bool read_matrix_csr(int64_t&               nrow,
int64_t&               ncol,
int64_t&               nnz,
int64_t**              ptr,
int**                  col,
std::complex<double>** val,
const char*            filename);
#endif

template bool write_matrix_csr(int64_t      nrow,
int64_t      ncol,
int64_t      nnz,
const int*   ptr,
const int*   col,
const float* val,
const char*  filename);
template bool write_matrix_csr(int64_t       nrow,
int64_t       ncol,
int64_t       nnz,
const int*    ptr,
const int*    col,
const double* val,
const char*   filename);
#ifdef SUPPORT_COMPLEX
template bool write_matrix_csr(int64_t                    nrow,
int64_t                    ncol,
int64_t                    nnz,
const int*                 ptr,
const int*                 col,
const std::complex<float>* val,
const char*                filename);
template bool write_matrix_csr(int64_t                     nrow,
int64_t                     ncol,
int64_t                     nnz,
const int*                  ptr,
const int*                  col,
const std::complex<double>* val,
const char*                 filename);
#endif

template bool write_matrix_csr(int64_t        nrow,
int64_t        ncol,
int64_t        nnz,
const int64_t* ptr,
const int*     col,
const float*   val,
const char*    filename);
template bool write_matrix_csr(int64_t        nrow,
int64_t        ncol,
int64_t        nnz,
const int64_t* ptr,
const int*     col,
const double*  val,
const char*    filename);
#ifdef SUPPORT_COMPLEX
template bool write_matrix_csr(int64_t                    nrow,
int64_t                    ncol,
int64_t                    nnz,
const int64_t*             ptr,
const int*                 col,
const std::complex<float>* val,
const char*                filename);
template bool write_matrix_csr(int64_t                     nrow,
int64_t                     ncol,
int64_t                     nnz,
const int64_t*              ptr,
const int*                  col,
const std::complex<double>* val,
const char*                 filename);
#endif

} 
