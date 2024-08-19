#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"
#include "read_mtx.hpp"
#include "read_dmat.hpp"
#include "read_dvec.hpp"
#include "read_svec.hpp"
#include "write_mtx.hpp"
#include "write_dvec.hpp"
#include "write_svec.hpp"


enum class IO_MODE
{
ANSI_BASE,
UNSAFE,
SAFE
};

class IO_DATA
{
public:
IO_DATA(IO_MODE io_method) : m_io_method{io_method}
{

}

~IO_DATA()
{

}

bool load_mat(std::string file_name, COO &output_mat)
{
bool load_result = true;
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
r_mtx.load_matrix_mtx_ANSI_based(file_name, output_mat);
break;
case IO_MODE::SAFE:
r_mtx.load_matrix_mtx_SAFE(file_name, output_mat);
break;
case IO_MODE::UNSAFE:
r_mtx.load_matrix_mtx_UNSAFE(file_name, output_mat);
break;
}
return load_result;
}

bool load_mat(std::string file_name, D_MATRIX &output_mat)
{
bool load_result = true;
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
load_result = r_dmat.load_dmat_ANSI_BASE(file_name, output_mat);
break;
case IO_MODE::SAFE:
load_result = r_dmat.load_dmat_SAFE(file_name, output_mat);
break;
case IO_MODE::UNSAFE:
load_result = r_dmat.load_dmat_UNSAFE(file_name, output_mat);
break;
}
return load_result;
}

bool load_vec(std::string file_name, D_VECTOR &output_vec)
{
bool load_result = true;
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
load_result = r_dvec.load_dvec_ANSI_BASE(file_name, output_vec);
break;
case IO_MODE::SAFE:
load_result = r_dvec.load_dvec_SAFE(file_name, output_vec);
break;
case IO_MODE::UNSAFE:
load_result = r_dvec.load_dvec_UNSAFE(file_name, output_vec);
break;
}
return load_result;
}

bool load_vec(std::string file_name, S_VECTOR &output_vec)
{
bool load_result = true;
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
load_result = r_svec.load_svec_ANSI_BASE(file_name, output_vec);
break;
case IO_MODE::SAFE:
load_result = r_svec.load_svec_SAFE(file_name, output_vec);
break;
case IO_MODE::UNSAFE:
load_result = r_svec.load_svec_UNSAFE(file_name, output_vec);
break;
}
return load_result;
}

bool write_mat(std::string file_name, COO &output_mat)
{
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
w_mtx.write_mtx_ANSI_based(file_name, output_mat);
break;
case IO_MODE::SAFE:
w_mtx.write_mtx_SAFE(file_name, output_mat);
break;
case IO_MODE::UNSAFE:
w_mtx.write_mtx_UNSAFE(file_name, output_mat);
break;
}
return true;
}

bool write_vec(std::string file_name, D_VECTOR &output_vec)
{
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
w_dvec.write_dvec_ANSI_based(file_name, output_vec);
break;
case IO_MODE::SAFE:
w_dvec.write_dvec_SAFE(file_name, output_vec);
break;
case IO_MODE::UNSAFE:
w_dvec.write_dvec_UNSAFE(file_name, output_vec);
break;
}
return true;
}

bool write_vec(std::string file_name, S_VECTOR &output_vec)
{
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
w_svec.write_svec_ANSI_based(file_name, output_vec);
break;
case IO_MODE::SAFE:
w_svec.write_svec_SAFE(file_name, output_vec);
break;
case IO_MODE::UNSAFE:
w_svec.write_svec_UNSAFE(file_name, output_vec);
break;
}
return true;
}

void check_io_method() const
{
std::cout << "Current loading method is ";
print_mode();
}

void set_io_method(IO_MODE new_mode)
{
check_io_method();
this->m_io_method = new_mode;
print_mode();
}

private:
IO_MODE m_io_method;
READ_MTX r_mtx;
READ_DMAT r_dmat;
READ_DVEC r_dvec;
READ_SVEC r_svec;
WRITE_MTX w_mtx;
WRITE_DVEC w_dvec;
WRITE_SVEC w_svec;

void print_mode() const
{
switch (m_io_method)
{
case IO_MODE::ANSI_BASE:
std::cout << "ANSI based mode.\n";
break;
case IO_MODE::SAFE:
std::cout << "Safe load mode.\n";
break;
case IO_MODE::UNSAFE:
std::cout << "Unsafe load mode.\n";
break;
}
}
};