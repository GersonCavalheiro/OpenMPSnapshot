
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4244 4267)
#endif

#define BOOST_NO_AUTO_PTR

#include <sstream>
#include <iostream>
#include <chrono>
#include <random>

#include <boost/format.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/base_object.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>


template <class T>
std::chrono::nanoseconds
saveData( T const & data, std::function<void(std::ostringstream &, T const&)> saveFunction, std::ostringstream & os )
{
auto start = std::chrono::high_resolution_clock::now();
saveFunction( os, data );
return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::high_resolution_clock::now() - start );
}


template <class T>
std::pair<T, std::chrono::nanoseconds>
loadData( std::ostringstream const & dataStream, std::function<void(std::istringstream &, T &)> loadFunction )
{
T data;
std::istringstream os( dataStream.str() );

auto start = std::chrono::high_resolution_clock::now();
loadFunction( os, data );

return {data, std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::high_resolution_clock::now() - start )};
}

struct cerealBinary
{
template <class T>
static void save( std::ostringstream & os, T const & data )
{
cereal::BinaryOutputArchive oar(os);
oar(data);
}

template <class T>
static void load( std::istringstream & is, T & data )
{
cereal::BinaryInputArchive iar(is);
iar(data);
}
};

struct boostBinary
{
template <class T>
static void save( std::ostringstream & os, T const & data )
{
boost::archive::binary_oarchive oar(os);
oar & data;
}

template <class T>
static void load( std::istringstream & is, T & data )
{
boost::archive::binary_iarchive iar(is);
iar & data;
}
};

struct binary
{
typedef boostBinary  boost;
typedef cerealBinary cereal;
};


template <class SerializationT, class DataTCereal, class DataTBoost>
void test( std::string const & name,
DataTCereal const & dataC,
DataTBoost const & dataB,
size_t numAverages = 100,
bool validateData = false );

template <class SerializationT, class DataTCereal, class DataTBoost>
void test( std::string const & name,
DataTCereal const & dataC,
DataTBoost const & dataB,
size_t numAverages,
bool  )
{
std::cout << "-----------------------------------" << std::endl;
std::cout << "Running test: " << name << std::endl;

std::chrono::nanoseconds totalBoostSave{0};
std::chrono::nanoseconds totalBoostLoad{0};

std::chrono::nanoseconds totalCerealSave{0};
std::chrono::nanoseconds totalCerealLoad{0};

size_t boostSize = 0;
size_t cerealSize = 0;

for(size_t i = 0; i < numAverages; ++i)
{
{
std::ostringstream os;
auto saveResult = saveData<DataTBoost>( dataB, {SerializationT::boost::template save<DataTBoost>}, os );
totalBoostSave += saveResult;
if(!boostSize)
boostSize = os.tellp();

auto loadResult = loadData<DataTBoost>( os, {SerializationT::boost::template load<DataTBoost>} );
totalBoostLoad += loadResult.second;
}

{
std::ostringstream os;
auto saveResult = saveData<DataTCereal>( dataC, {SerializationT::cereal::template save<DataTCereal>}, os );
totalCerealSave += saveResult;
if(!cerealSize)
cerealSize = os.tellp();

auto loadResult = loadData<DataTCereal>( os, {SerializationT::cereal::template load<DataTCereal>} );
totalCerealLoad += loadResult.second;
}
}

double averageBoostSave = std::chrono::duration_cast<std::chrono::milliseconds>(totalBoostSave).count() / static_cast<double>( numAverages );
double averageBoostLoad = std::chrono::duration_cast<std::chrono::milliseconds>(totalBoostLoad).count() / static_cast<double>( numAverages );

double averageCerealSave = std::chrono::duration_cast<std::chrono::milliseconds>(totalCerealSave).count() / static_cast<double>( numAverages );
double averageCerealLoad = std::chrono::duration_cast<std::chrono::milliseconds>(totalCerealLoad).count() / static_cast<double>( numAverages );

double cerealSaveP = averageCerealSave / averageBoostSave;
double cerealLoadP = averageCerealLoad / averageBoostLoad;
double cerealSizeP = cerealSize / static_cast<double>( boostSize );

std::cout << "  Boost results:" << std::endl;
std::cout << boost::format("\tsave | time: %06.4fms (%1.2f) size: %20.8fkb (%1.8f) total: %6.1fms")
% averageBoostSave % 1.0 % (boostSize / 1024.0) % 1.0 % static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds>(totalBoostSave).count() );
std::cout << std::endl;
std::cout << boost::format("\tload | time: %06.4fms (%1.2f) total: %6.1fms")
% averageBoostLoad % 1.0 % static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds>(totalBoostLoad).count() );
std::cout << std::endl;

std::cout << "  Cereal results:" << std::endl;
std::cout << boost::format("\tsave | time: %06.4fms (%1.2f) size: %20.8fkb (%1.8f) total: %6.1fms")
% averageCerealSave % cerealSaveP % (cerealSize / 1024.0) % cerealSizeP % static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds>(totalCerealSave).count() );
std::cout << std::endl;
std::cout << boost::format("\tload | time: %06.4fms (%1.2f) total: %6.1fms")
% averageCerealLoad % cerealLoadP % static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds>(totalCerealLoad).count() );
std::cout << std::endl;
}

template <class SerializationT, class DataT>
void test( std::string const & name,
DataT const & data,
size_t numAverages = 100,
bool validateData = false )
{
return test<SerializationT, DataT, DataT>( name, data, data, numAverages, validateData );
}

template<class T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
random_value(std::mt19937 & gen)
{ return std::uniform_real_distribution<T>(-10000.0, 10000.0)(gen); }

template<class T>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) != sizeof(char), T>::type
random_value(std::mt19937 & gen)
{ return std::uniform_int_distribution<T>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max())(gen); }

template<class T>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) == sizeof(char), T>::type
random_value(std::mt19937 & gen)
{ return static_cast<T>( std::uniform_int_distribution<int64_t>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max())(gen) ); }

template<class T>
typename std::enable_if<std::is_same<T, std::string>::value, std::string>::type
random_value(std::mt19937 & gen)
{
std::string s(std::uniform_int_distribution<int>(3, 30)(gen), ' ');
for(char & c : s)
c = std::uniform_int_distribution<char>(' ', '~')(gen);
return s;
}

template<class C>
std::basic_string<C> random_basic_string(std::mt19937 & gen, size_t maxSize = 30)
{
std::basic_string<C> s(std::uniform_int_distribution<int>(3, maxSize)(gen), ' ');
for(C & c : s)
c = static_cast<C>( std::uniform_int_distribution<int>( '~', '~' )(gen) );
return s;
return s;
}

template <size_t N>
std::string random_binary_string(std::mt19937 & gen)
{
std::string s(N, ' ');
for(auto & c : s )
c = std::uniform_int_distribution<char>('0', '1')(gen);
return s;
}

struct PoDStructCereal
{
int32_t a;
int64_t b;
float c;
double d;

template <class Archive>
void serialize( Archive & ar )
{
ar(a, b, c, d);
}
};

struct PoDStructBoost
{
int32_t a;
int64_t b;
float c;
double d;

template <class Archive>
void serialize( Archive & ar, const unsigned int  )
{
ar & a & b & c & d;
}
};

struct PoDChildCereal : virtual PoDStructCereal
{
PoDChildCereal() : v(1024)
{ }

std::vector<float> v;

template <class Archive>
void serialize( Archive & ar )
{
ar( cereal::virtual_base_class<PoDStructCereal>(this), v );
}
};

struct PoDChildBoost : virtual PoDStructBoost
{
PoDChildBoost() : v(1024)
{ }

std::vector<float> v;

template <class Archive>
void serialize( Archive & ar, const unsigned int  )
{
ar & boost::serialization::base_object<PoDStructBoost>(*this);
ar & v;
}
};

int main()
{
std::random_device rd;
std::mt19937 gen(rd());
auto rngC = [&](){ return random_value<uint8_t>(gen); };
auto rngD = [&](){ return random_value<double>(gen); };
const bool randomize = false;

auto vectorDoubleTest = [&](size_t s, bool randomize_)
{
std::ostringstream name;
name << "Vector(double) size " << s;

std::vector<double> data(s);
if(randomize_)
for( auto & d : data )
d = rngD();

test<binary>( name.str(), data );
};

vectorDoubleTest(1, randomize); 
vectorDoubleTest(16, randomize); 
vectorDoubleTest(1024, randomize); 
vectorDoubleTest(1024*1024, randomize); 

auto vectorCharTest = [&](size_t s, bool randomize_)
{
std::ostringstream name;
name << "Vector(uint8_t) size " << s;

std::vector<uint8_t> data(s);
if(randomize_)
for( auto & d : data )
d = rngC();

test<binary>( name.str(), data );
};

vectorCharTest(1024*1024*64, randomize);

auto vectorPoDStructTest = [&](size_t s)
{
std::ostringstream name;
name << "Vector(PoDStruct) size " << s;

std::vector<PoDStructCereal> dataC(s);
std::vector<PoDStructBoost> dataB(s);
test<binary>( name.str(), dataC, dataB );
};

vectorPoDStructTest(1);
vectorPoDStructTest(64);
vectorPoDStructTest(1024);
vectorPoDStructTest(1024*1024);
vectorPoDStructTest(1024*1024*2);

auto vectorPoDChildTest = [&](size_t s)
{
std::ostringstream name;
name << "Vector(PoDChild) size " << s;

std::vector<PoDChildCereal> dataC(s);
std::vector<PoDChildBoost> dataB(s);
test<binary>( name.str(), dataC, dataB );
};

vectorPoDChildTest(1024);
vectorPoDChildTest(1024*32);

auto stringTest = [&](size_t s)
{
std::ostringstream name;
name << "String size " << s;

std::string data = random_basic_string<char>(gen, s);
std::cout << "data.size " << data.size() << std::endl;
test<binary>( name.str(), data );
};

stringTest(200000);
stringTest(2000000);
stringTest(20000000);

auto vectorStringTest = [&](size_t s)
{
std::ostringstream name;
name << "Vector(String) size " << s;

std::vector<std::string> data(s);
for(size_t i=0; i<data.size(); ++i)
data[i] = random_basic_string<char>(gen);

test<binary>( name.str(), data );
};

vectorStringTest(512);
vectorStringTest(1024);
vectorStringTest(1024*64);
vectorStringTest(1024*128);

auto mapPoDStructTest = [&](size_t s)
{
std::ostringstream name;
name << "Map(PoDStruct) size " <<s;

std::map<std::string, PoDStructCereal> mC;
std::map<std::string, PoDStructBoost> mB;
for(size_t i=0; i<s; ++i)
{
mC[std::to_string( i )] = PoDStructCereal();
mB[std::to_string( i )] = PoDStructBoost();
}
test<binary>(name.str(), mC, mB);
};

mapPoDStructTest(1024);
mapPoDStructTest(1024*64);

return 0;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
