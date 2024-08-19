#pragma once
#include <vector>
#include "IndividualParameters.h"

class Individual {
public:
Individual() : infected_(false), hit_(false), recovered_(), epochs_infected_(0), location_(0) { } 
Individual(bool infected, bool hit, bool recovered, std::uint8_t days_infected, int location) 
: infected_(infected), hit_(hit), recovered_(recovered), epochs_infected_(days_infected), location_(location) { }
void infect();
void recover();
void advance_epoch();
void try_infect();
void move(std::vector<int>& new_locations);
void set_location(int location);
int get_location() const;
bool is_infected() const;
bool is_hit() const;
bool is_recovered() const;
private:
bool infected_;
bool hit_; 
bool recovered_;
std::uint8_t epochs_infected_;
int location_; 
IndividualParameters parameters_;
static float get_random_infect_chance();
static int get_random_location(size_t neighbours_size);
};

inline void Individual::infect() {
infected_ = true;
hit_ = true;
}

inline void Individual::recover() {
if (infected_) {
infected_ = false;
recovered_ = true;
}
}

inline void Individual::advance_epoch() {
if (infected_) {
if (epochs_infected_ >= parameters_.DiseaseDuration)
recover();
else
++epochs_infected_;
}
}

inline void Individual::set_location(int location) {
location_ = location;
}

inline int Individual::get_location() const {
return location_;
}

inline bool Individual::is_infected() const {
return infected_;
}

inline bool Individual::is_hit() const {
return hit_;
}

inline bool Individual::is_recovered() const {
return recovered_;
}