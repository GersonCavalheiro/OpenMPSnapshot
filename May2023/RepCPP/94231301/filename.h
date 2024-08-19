
#pragma once

#include "platform.h"

namespace embree
{

class FileName
{
public:


FileName ();


FileName (const char* filename);


FileName (const std::string& filename);


static FileName homeFolder();


static FileName executableFolder();


operator std::string() const { return filename; }


const std::string str() const { return filename; }


const char* c_str() const { return filename.c_str(); }


FileName path() const;


std::string base() const;


std::string name() const;


std::string ext() const;


FileName dropExt() const;


FileName setExt(const std::string& ext = "") const;


FileName addExt(const std::string& ext = "") const;


FileName operator +( const FileName& other ) const;


FileName operator +( const std::string& other ) const;


FileName operator -( const FileName& base ) const;


friend bool operator==(const FileName& a, const FileName& b);


friend bool operator!=(const FileName& a, const FileName& b);


friend std::ostream& operator<<(std::ostream& cout, const FileName& filename);

private:
std::string filename;
};
}
