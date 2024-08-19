#pragma once
#include <vector>

#include "monolish/common/monolish_common.hpp"
#include <functional>

namespace monolish {

namespace solver {


enum class initvec_scheme {
RANDOM,
USER,
};

template <typename MATRIX, typename Float> class precondition;



template <typename MATRIX, typename Float> class solver {
private:
protected:
int lib = 0;
double tol = 1.0e-8;
size_t miniter = 0;
size_t maxiter = SIZE_MAX;
size_t resid_method = 0;
bool print_rhistory = false;
std::string rhistory_file;
std::ostream *rhistory_stream;
initvec_scheme initvecscheme = initvec_scheme::RANDOM;

double final_resid = 0;
size_t final_iter = 0;

Float omega = 1.9; 
int singularity;   
int reorder = 3;   

Float get_residual(vector<Float> &x);
precondition<MATRIX, Float> precond;

public:

solver(){};


~solver() {
if (rhistory_stream != &std::cout && rhistory_file.empty() != true) {
delete rhistory_stream;
}
}


template <class PRECOND> void set_create_precond(PRECOND &p);


template <class PRECOND> void set_apply_precond(PRECOND &p);


void set_lib(int l) { lib = l; }


void set_tol(double t) { tol = t; }


void set_maxiter(size_t max) { maxiter = max; }


void set_miniter(size_t min) { miniter = min; }


void set_residual_method(size_t r) { resid_method = r; }


void set_print_rhistory(bool flag) {
print_rhistory = flag;
rhistory_stream = &std::cout;
}


void set_rhistory_filename(std::string file) {
rhistory_file = file;

rhistory_stream = new std::ofstream(rhistory_file);
if (rhistory_stream->fail()) {
throw std::runtime_error("error bad filename");
}
}


void set_initvec_scheme(initvec_scheme scheme) { initvecscheme = scheme; }


[[nodiscard]] int get_lib() const { return lib; }


[[nodiscard]] double get_tol() const { return tol; }


[[nodiscard]] size_t get_maxiter() const { return maxiter; }


[[nodiscard]] size_t get_miniter() const { return miniter; }


[[nodiscard]] size_t get_residual_method() const { return resid_method; }


bool get_print_rhistory() const { return print_rhistory; }


initvec_scheme get_initvec_scheme() const { return initvecscheme; }


void set_omega(Float w) { omega = w; };


Float get_omega() { return omega; };


void set_reorder(int r) { reorder = r; }


int get_reorder() { return reorder; }


int get_singularity() { return singularity; }

double get_final_residual() { return final_resid; }
size_t get_final_iter() { return final_iter; }
};


template <typename MATRIX, typename Float> class precondition {
private:
public:
vector<Float> M;
MATRIX *A;

std::function<void(MATRIX &)> create_precond;
std::function<void(const vector<Float> &r, vector<Float> &z)> apply_precond;

std::function<void(void)> get_precond();

void set_precond_data(vector<Float> &m) { M = m; };
vector<Float> get_precond_data() { return M; };

precondition() {
auto create = [](MATRIX &) {};
auto apply = [](const vector<Float> &r, vector<Float> &z) { z = r; };
create_precond = create;
apply_precond = apply;
};
};

} 
} 
