


#pragma once

#include <map>
#include <vector>
#include <string>

#include "paraverkerneltypes.h"

template <class T>
class FunctionManagement
{

private:
static FunctionManagement<T> *inst;

std::vector<std::string> nameGroups;
std::map<std::string, T *> hash;
std::vector<std::vector<T *> > groups;

FunctionManagement( std::vector<std::string>&,
std::vector<std::string>&,
std::vector<std::vector<T *> >& );


public:
static FunctionManagement<T> *getInstance();
static FunctionManagement<T> *getInstance( std::vector<std::string>&,
std::vector<std::string>&,
std::vector<std::vector<T *> >& );

~FunctionManagement();

T *getFunction( const std::string& ) const;
const T * getFunctionNoClone( const std::string& ) const;
PRV_UINT32 numGroups() const;
void getNameGroups( std::vector<std::string>& );
void getAll( std::vector<T *>& ) const;
void getAll( std::vector<T *>&, PRV_UINT32 ) const;
void getAll( std::vector<std::string>& ) const;
void getAll( std::vector<std::string>&, PRV_UINT32 ) const;
};

#include "functionmanagement_impl.h"
