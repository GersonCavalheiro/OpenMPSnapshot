
#pragma once

#include <string>
#include <iostream>


#ifdef KRATOS_INDEPENDENT
#else
#include "includes/define.h"
#endif

namespace Kratos {




template<class TConfiguration>
class OctreeBinaryCell {
public:


typedef typename TConfiguration::data_type data_type;

typedef TConfiguration configuration_type;

typedef typename TConfiguration::pointer_type pointer_type;

typedef std::vector<pointer_type> object_container_type;

typedef std::size_t key_type;

static constexpr std::size_t CHILDREN_NUMBER = 8;
static constexpr std::size_t DIMENSION = TConfiguration::DIMENSION;
static constexpr std::size_t MAX_LEVEL = TConfiguration::MAX_LEVEL;
static constexpr std::size_t ROOT_LEVEL = MAX_LEVEL - 1;
static constexpr std::size_t MIN_LEVEL = TConfiguration::MIN_LEVEL;

enum {
LEFT = 0,
RIGHT = 1,
BACK = 2,
FRONT = 3,
BOTTOM = 4,
TOP = 6
};



explicit OctreeBinaryCell(char Level = ROOT_LEVEL) : level_(Level), children_(NULL), data_(NULL) {
for (std::size_t i = 0; i < DIMENSION; i++)
min_key_[i] = 0;
}


virtual ~OctreeBinaryCell() {

if (data_) configuration_type::DeleteData(data_);
delete [] children_;
}

void DeleteChildren() {
delete [] children_;
children_=NULL;
}
void DeleteData() {
if (data_){
configuration_type::DeleteData(data_);
data_=NULL;
}
}



std::size_t GetChildIndex(key_type x_key, key_type y_key, key_type z_key) const {
char next_level = ( char)( level_ - 1);
key_type level_bit = 1 << next_level;
return (((x_key & level_bit) >> next_level) + (((y_key & level_bit) >> next_level) << 1) + (((z_key & level_bit) >> next_level) << 2));
}

int SubdivideCell() {
if (level_ == 0)
return 1;
if(children_)
return 1;

children_ = new OctreeBinaryCell[CHILDREN_NUMBER];

char next_level = ( char)( level_ - 1);

for (std::size_t i = 0; i < CHILDREN_NUMBER; i++) {
children_[i].SetMinKey(min_key_[0] | ((i & 1) << next_level), min_key_[1] | (((i & 2) >> 1)) << next_level, min_key_[2] | (((i & 4) >> 2)) << next_level);
children_[i].SetLevel(next_level);
}

return 0; 
}

void GetMinPointNormalized(double* min_point) const {
for (std::size_t i = 0; i < DIMENSION; i++) {
min_point[i] = GetCoordinateNormalized(min_key_[i]);
}
}

void GetMaxPointNormalized(double* max_point) const {
double size = CalcSizeNormalized();
for (std::size_t i = 0; i < DIMENSION; i++) {
max_point[i] = GetCoordinateNormalized(min_key_[i]) + size;
}
}

int GetLeftKey(key_type* keys) const {
if (min_key_[0] >= 1) {
keys[0] = min_key_[0] - 1;
keys[1] = min_key_[1];
keys[2] = min_key_[2];
return 1; 
}
return 0; 
}

int GetRightKey(key_type* keys) const {
if (min_key_[0] < static_cast<key_type> ((1 << ROOT_LEVEL) - (1 << level_))) {
keys[0] = min_key_[0] + (static_cast<key_type>(1) << level_);
keys[1] = min_key_[1];
keys[2] = min_key_[2];
return 1; 
}
return 0; 
}

int GetBackKey(key_type* keys) const {
if (min_key_[1] >= 1) {
keys[0] = min_key_[0];
keys[1] = min_key_[1] - 1;
keys[2] = min_key_[2];
return 1; 
}
return 0; 
}

int GetFrontKey(key_type* keys) const {
if (min_key_[1] < static_cast<key_type> ((1 << ROOT_LEVEL) - (1 << level_))) {
keys[0] = min_key_[0];
keys[1] = min_key_[1] + (static_cast<key_type>(1) << level_);
keys[2] = min_key_[2];
return 1; 
}
return 0; 
}

int GetBottomKey(key_type* keys) const {
if (min_key_[2] >= 1) {
keys[0] = min_key_[0];
keys[1] = min_key_[1];
keys[2] = min_key_[2] - 1;
return 1; 
}
return 0; 
}

int GetTopKey(key_type* keys) const {
if (min_key_[2] < static_cast<key_type> ((1 << ROOT_LEVEL) - (1 << level_))) {
keys[0] = min_key_[0];
keys[1] = min_key_[1];
keys[2] = min_key_[2] + (static_cast<key_type>(1) << level_);
return 1; 
}
return 0; 
}

int GetKey(std::size_t position, key_type* keys) const {
const std::size_t x_position[] = {0, 2, 2, 0, 0, 2, 2, 0, 1, 1, 2, 1, 0, 0, 2, 2, 0, 1, 2, 1, 0, 1, 1, 2, 1, 0, 1};
const std::size_t y_position[] = {0, 0, 2, 2, 0, 0, 2, 2, 1, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 2, 1, 1, 0, 1, 2, 1, 1};
const std::size_t z_position[] = {0, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2};

keys[0] = min_key_[0] + ((x_position[position]) << (level_ - 1));
keys[1] = min_key_[1] + ((y_position[position]) << (level_ - 1));
keys[2] = min_key_[2] + ((z_position[position]) << (level_ - 1));

return 1;
}

int GetNeighbourKey(std::size_t direction, key_type* keys) const {


assert(direction<18);
const std::size_t x_offset[]={0,2,1,1,1,1,0,2,0,2,0,2,0,2,1,1,1,1};
const std::size_t y_offset[]={1,1,0,2,1,1,0,0,2,2,1,1,1,1,0,2,0,2};
const std::size_t z_offset[]={1,1,1,1,0,2,1,1,1,1,0,0,2,2,0,0,2,2};
const std::size_t x_coef[]  ={0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0};
const std::size_t y_coef[]  ={0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,1,0,1};
const std::size_t z_coef[]  ={0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,1};

std::size_t size = (1<<level_);

keys[0] = min_key_[0] + x_offset[direction] + x_coef[direction] * size;
keys[1] = min_key_[1] + y_offset[direction] + y_coef[direction] * size;
keys[2] = min_key_[2] + z_offset[direction] + z_coef[direction] * size;

for(unsigned int i = 0 ; i < DIMENSION ; i++){
if(keys[i] == 0)
return 0; 
else
(keys[i])--;

if(keys[i] > static_cast<key_type>(1 << ROOT_LEVEL))
return 0; 
}
return 1; 
}

int GetNeighbourKey(std::size_t position, std::size_t direction, key_type* keys) const {

GetKey(position, keys);

keys[0] += (direction & 1) << 1;
keys[1] += (direction & 2);
keys[2] += (direction & 4) >> 1;

for(unsigned int i = 0 ; i < DIMENSION ; i++){
if(keys[i] == 0)
return 0; 
else
(keys[i])--;

if(keys[i] > static_cast<key_type>(1 << ROOT_LEVEL))
return 0; 
}
return 1; 
}

std::size_t GetLocalPosition(key_type* keys){
key_type position[3];
const std::size_t local_index[]={0, 9, 1,12,21,10, 3,11, 2,13,22,14,25, 8,23,16,24,15, 4,17, 5,20,26,18, 7,19,6};

for(std::size_t i = 0 ; i < DIMENSION ; i++)
{
position[i] = (keys[i] - min_key_[i]) >> (level_-1);
}
std::size_t index = position[0] + position[1] * 3 + position[2] * 9;
assert(index <= 26);
return local_index[index];

}


void Insert(pointer_type object){
objects_.push_back(object);
}

void TransferObjectsToSonsNormalized(){      

if (!objects_.size()) return;
assert(this->HasChildren());

const double tolerance = 0.001 * double(1 << MIN_LEVEL) / double(1 << ROOT_LEVEL) ; 
double min_coord[3]={0.00, 0.00, 0.00};
double max_coord[3]={0.00, 0.00, 0.00};

for (std::size_t i = 0; i < CHILDREN_NUMBER; i++){            
OctreeBinaryCell* son = pGetChild(i);
if (son->HasChildren()){
son->TransferObjectsToSonsNormalized();
continue;
}
son->GetMinPointNormalized(min_coord);
son->GetMaxPointNormalized(max_coord);
pointer_type object;
for (int j=0;j<(int)objects_.size();j++){
object=objects_[j];
const int is_intersected = configuration_type::IsIntersected(object,tolerance, min_coord, max_coord);
if(is_intersected)
son->Insert(object);
}
}        

object_container_type temp;
objects_.swap(temp);                   
}


unsigned char GetLevel() const {
return level_;
}

char SetLevel(char level) {
level_ = level;
return level_;
}

void GetMinKey(key_type& min_key_x, key_type& min_key_y, key_type& min_key_z) const {
min_key_x = min_key_[0];
min_key_y = min_key_[1];
min_key_z = min_key_[2];
}

void SetMinKey(key_type min_key_x, key_type min_key_y, key_type min_key_z) {
min_key_[0] = min_key_x;
min_key_[1] = min_key_y;
min_key_[2] = min_key_z;
}

OctreeBinaryCell& rGetChild(std::size_t pos) const {
return children_[pos];
}

OctreeBinaryCell* pGetChild(std::size_t pos) const {
return children_ + pos;
}

OctreeBinaryCell* pGetChild(key_type x_key, key_type y_key, key_type z_key) const {
return pGetChild(GetChildIndex(x_key, y_key, z_key));
}

OctreeBinaryCell* GetChildren() {
return children_;
}

OctreeBinaryCell const* GetChildren() const {
return children_;
}

data_type* pGetData() const
{
return data_;
}

data_type** pGetDataPointer()
{
return &data_;
}

const std::vector<pointer_type>* pGetObjects() const
{
return &objects_;
}

std::vector<pointer_type>* pGetObjects()
{
return &objects_;
}

void EmptyObjects()
{
object_container_type tmp;     
tmp.swap(objects_);
}


bool IsLeaf() const {
return (children_ == NULL);
}

bool HasChildren() const {
return (children_ != NULL);
}



virtual std::string Info() const {
return "OctreeBinaryCell";
}


virtual void PrintInfo(std::ostream& rOStream) const {
for (char i = ROOT_LEVEL; i > level_; i--) {
rOStream << "  ";
}
rOStream << Info() << " at level " << static_cast<int> (level_);
}


virtual void PrintData(std::ostream& rOStream) const {
rOStream << "(" << GetCoordinateNormalized(min_key_[0]) << "," << GetCoordinateNormalized(min_key_[1]) << "," << GetCoordinateNormalized(min_key_[2]) << "),";
rOStream << "(" << GetCoordinateNormalized(min_key_[0]) + CalcSizeNormalized() << "," << GetCoordinateNormalized(min_key_[1]) + CalcSizeNormalized() << "," << GetCoordinateNormalized(min_key_[2]) + CalcSizeNormalized() << ")" << std::endl;

for (std::size_t i = 0; i < CHILDREN_NUMBER; i++) {
if (children_) {
for (char j = ROOT_LEVEL + 1; j > level_; j--) {
rOStream << "  ";
}

rOStream << "child #" << i;

children_[i].PrintData(rOStream);
}
}
}




protected:















private:



char level_;
key_type min_key_[DIMENSION];
OctreeBinaryCell* children_;
data_type* data_;
object_container_type objects_;


double CalcSizeNormalized() const {
const double scale = 1.00 / (1 << ROOT_LEVEL);

return (1 << level_) * scale; 
}

double GetCoordinateNormalized(key_type key) const {
const double scale = 1.00 / (1 << ROOT_LEVEL);

return static_cast<double>(key * scale);
}











OctreeBinaryCell & operator=(OctreeBinaryCell const& rOther) {
return *this;
}


OctreeBinaryCell(OctreeBinaryCell const& rOther) {
}



}; 







template<class TConfiguration>
inline std::ostream & operator <<(std::ostream& rOStream,
const OctreeBinaryCell<TConfiguration>& rThis) {
rThis.PrintInfo(rOStream);
rThis.PrintData(rOStream);

return rOStream;
}


} 


