#pragma once

#include <functional>
#include <memory>

namespace bnmf_algs {
namespace util {


template <typename T, typename Computer>
class ComputationIterator : public std::iterator<std::forward_iterator_tag, T> {
public:

ComputationIterator(T* init_val_ptr, size_t* step_count_ptr,
Computer* computer_ptr)
: curr_val_ptr(init_val_ptr), step_count_ptr(step_count_ptr),
computer_ptr(computer_ptr){};


explicit ComputationIterator(size_t* step_count_ptr)
: curr_val_ptr(nullptr), step_count_ptr(step_count_ptr),
computer_ptr(nullptr) {}


ComputationIterator& operator++() {
(*computer_ptr)(*step_count_ptr, *curr_val_ptr);
++(*step_count_ptr);
return *this;
}


const T& operator*() const { return *curr_val_ptr; }


const T* operator->() const { return curr_val_ptr; }


bool operator==(const ComputationIterator& other) const {
return *(this->step_count_ptr) == *(other.step_count_ptr);
}


bool operator!=(const ComputationIterator& other) const {
return !(*this == other);
}

private:
T* curr_val_ptr;
size_t* step_count_ptr;
Computer* computer_ptr;
};


template <typename T, typename Computer> class Generator {
public:

using iter_type = ComputationIterator<T, Computer>;

public:

Generator(const T& init_val, size_t iter_count, Computer&& computer)
: init_val(init_val), curr_step_count(0), total_iter_count(iter_count),
computer(std::move(computer)),
begin_it(&(this->init_val), &(this->curr_step_count),
&(this->computer)),
end_it(&(this->total_iter_count)){};


Generator(const Generator& other)
: init_val(other.init_val), curr_step_count(other.curr_step_count),
total_iter_count(other.total_iter_count), computer(other.computer),
begin_it(iter_type(&(this->init_val), &(this->curr_step_count),
&(this->computer))),
end_it(iter_type(&(this->total_iter_count))) {}


Generator& operator=(const Generator& other) {
this->init_val = other.init_val;
this->curr_step_count = other.curr_step_count;
this->total_iter_count = other.total_iter_count;
this->computer = other.computer;

this->begin_it = iter_type(&(this->init_val), &(this->curr_step_count),
&(this->total_iter_count));
this->end_it = iter_type(&(this->total_iter_count));

return *this;
}


Generator(Generator&& other)
: init_val(std::move(other.init_val)),
curr_step_count(std::move(other.curr_step_count)),
total_iter_count(std::move(other.total_iter_count)),
computer(std::move(other.computer)),
begin_it(iter_type(&(this->init_val), &(this->curr_step_count),
&(this->computer))),
end_it(iter_type(&(this->total_iter_count))) {}


Generator& operator=(Generator&& other) {
this->init_val = std::move(other.init_val);
this->curr_step_count = std::move(other.curr_step_count);
this->total_iter_count = std::move(other.total_iter_count);
this->computer = std::move(other.computer);

this->begin_it = iter_type(&(this->init_val), &(this->curr_step_count),
&(this->computer));
this->end_it = iter_type(&(this->total_iter_count));

return *this;
}

iter_type begin() { return begin_it; }


iter_type end() { return end_it; }

private:
T init_val;
size_t curr_step_count;
size_t total_iter_count;
Computer computer;
iter_type begin_it;
iter_type end_it;
};
} 
} 
