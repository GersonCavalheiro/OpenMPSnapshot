

#pragma once


#include <sys/stat.h>
#ifdef _WIN32
#include <shlobj.h>
#include <Shlwapi.h>
#else
#include <sys/types.h>
#endif

#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/binary_object.hpp>

#include "paraverkerneltypes.h"

constexpr char cfgSeparator[] = "=============";

enum class WorkspaceType { STATE = 1, EVENT };

struct WorkspaceValue
{
WorkspaceType myType;

union
{
TState state;
TEventType eventType;
} UInfo;

bool operator<( const WorkspaceValue& anotherValue ) const
{
if( myType == WorkspaceType::STATE )
return UInfo.state < anotherValue.UInfo.state;
else
return UInfo.eventType < anotherValue.UInfo.eventType;
}

bool operator==( const WorkspaceValue& anotherValue ) const
{
if( myType != anotherValue.myType )
return false;

if( myType == WorkspaceType::STATE )
return UInfo.state == anotherValue.UInfo.state;
else
return UInfo.eventType == anotherValue.UInfo.eventType;
}

template< class Archive >
void serialize( Archive & ar, const unsigned int version )
{
ar & boost::serialization::make_nvp( "type", myType );
ar & boost::serialization::make_nvp( "WorkspaceValue", boost::serialization::make_binary_object( this, sizeof( *this ) ) );
}
};

BOOST_CLASS_VERSION( WorkspaceValue, 1 )

class Workspace
{
public:
Workspace();
Workspace( std::string whichName, WorkspaceType whichType ) : name( whichName ),
myType( whichType )
{}
~Workspace();

virtual std::string getName() const;
virtual WorkspaceType getType() const;
virtual std::vector<WorkspaceValue> getAutoTypes() const;
virtual std::vector<std::pair<std::string,std::string> > getHintCFGs() const;
virtual std::pair<std::string,std::string> getHintCFG( size_t whichHint ) const;

virtual void setName( std::string& whichName );
virtual void setType( WorkspaceType whichType );
virtual void setAutoTypes( std::vector<WorkspaceValue>& whichAutoTypes );
virtual void addHintCFG( std::pair<std::string,std::string>& whichCFG );
virtual void addHintCFG( size_t position, std::pair<std::string,std::string>& whichCFG );
virtual void removeHintCFG( size_t whichHint );
virtual void modifyHintCFG( size_t position, std::pair<std::string,std::string>& whichCFG );
virtual void clearHintCFGs();

virtual void loadXML( const std::string &wsFile );
virtual void saveXML( std::string &wsFile );

virtual void importWSXML( std::string &wsFile, const std::string& paraverUserDir );
virtual void importWSCFGs( std::string& wsFile, const std::string& paraverUserDir );

template< class TFuncMakeAbs, class TFuncGetFullName >
void exportWS( std::string &wsFile, TFuncMakeAbs makeAbs, TFuncGetFullName getFullName )
{
auto tmpHints = hintCFGs;
for_each( hintCFGs.begin(), hintCFGs.end(), 
[&]( std::pair< std::string, std::string >& elem )
{
elem.first = getFullName( elem.first );
} );

saveXML( wsFile );
hintCFGs = tmpHints;

std::ofstream ofs( wsFile.c_str(), std::ios::app );
if( ofs.good() )
{
for( auto hint : hintCFGs )
{
std::string cfgFilename = getFullName( hint.first );
ofs << cfgSeparator << cfgFilename << std::endl;
std::ifstream cfgFile( makeAbs( hint.first ) );
ofs << cfgFile.rdbuf();
cfgFile.close();
}
}

ofs.close();
}

template< class Archive >
void serialize( Archive & ar, const unsigned int version )
{
ar & boost::serialization::make_nvp( "name", name );
if( version == 1 )
myType = WorkspaceType::EVENT;
else if( version >= 2 )
ar & boost::serialization::make_nvp( "type", myType );

if( version == 1 )
{
std::vector<TEventType> tmpEventTypes;
ar & boost::serialization::make_nvp( "autoTypes", tmpEventTypes );
for( std::vector<TEventType>::iterator it = tmpEventTypes.begin(); it != tmpEventTypes.end(); ++it )
{
WorkspaceValue tmpWorkspaceValue;
tmpWorkspaceValue.myType = WorkspaceType::EVENT;
tmpWorkspaceValue.UInfo.eventType = *it;
autoTypes.push_back( tmpWorkspaceValue );
}
}
else if( version >= 2 )
ar & boost::serialization::make_nvp( "autoTypes", autoTypes );

ar & boost::serialization::make_nvp( "hintCFGs", hintCFGs );
}

protected:
std::string name;
WorkspaceType myType;
std::vector< WorkspaceValue > autoTypes;
std::vector< std::pair< std::string, std::string > > hintCFGs; 

private:
std::streampos wsFileXMLPos;
std::string firstCFGName;

std::string readToCFGSeparator( std::ifstream& ifs, std::ofstream& ofs );

void createDir( const std::string& whichDir );

};

BOOST_CLASS_VERSION( Workspace, 2 )



