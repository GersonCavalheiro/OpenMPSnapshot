


#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include "tracetypes.h"

template< typename dummy = std::nullptr_t >
class PCFFileParser
{
public:
using rgb = std::tuple< ParaverColor, ParaverColor, ParaverColor >;

template< typename dummyParser = std::nullptr_t >
class SectionParser
{
public:
SectionParser() = delete;
SectionParser( PCFFileParser<> *whichMainParser ) : mainParser( whichMainParser )
{}

virtual ~SectionParser() = default;

virtual void parseLine( const std::string& line ) = 0;

protected:
PCFFileParser<> *mainParser;
};

static bool openPCFFileParser( const std::string& filename, PCFFileParser<dummy>& outPCFFile );

PCFFileParser() = default;
PCFFileParser( const std::string& filename );
~PCFFileParser() = default;

void dumpToFile( const std::string& filename ) const;

std::string getLevel() const;
void setLevel( const std::string& newValue );

std::string getUnits() const;
void setUnits( const std::string& newValue );

std::string getLookBack() const;
void setLookBack( const std::string& newValue );

std::string getSpeed() const;
void setSpeed( const std::string& newValue );

std::string getFlagIcons() const;
void setFlagIcons( const std::string& newValue );

std::string getYmaxScale() const;
void setYmaxScale( const  std::string& newValue );

std::string getThreadFunc() const;
void setThreadFunc( const std::string& newValue );

const std::map< TState, std::string >& getStates() const;
void setState( TState state, const std::string& label );

const std::map< uint32_t, PCFFileParser::rgb >& getSemanticColors() const;
void setSemanticColor( uint32_t semanticValue, PCFFileParser::rgb color );

void                                        getEventTypes( std::vector< TEventType >& onVector ) const;
unsigned int                                getEventPrecision( TEventType eventType ) const;
const std::string&                          getEventLabel( TEventType eventType ) const;
const std::map< TEventValue, std::string >& getEventValues( TEventType eventType ) const;
void setEventType( TEventType eventType,
unsigned int precision,
const std::string& label,
const std::map< TEventValue, std::string >& values );
void setEventPrecision( TEventType eventType, unsigned int precision );
void setEventLabel( TEventType eventType, const std::string& label );
void setEventValues( TEventType eventType, const std::map< TEventValue, std::string >& values );
void setEventValueLabel( TEventType eventType, TEventValue eventValue, const std::string& label );

private:
struct EventTypeData
{
unsigned int precision;
std::string label;
std::map< TEventValue, std::string > values;
};

std::string level;
std::string units;
std::string lookBack;
std::string speed;
std::string flagIcons;
std::string ymaxScale;
std::string threadFunc;
std::map< TState, std::string > states;
std::map< uint32_t, PCFFileParser::rgb > semanticColors;
std::map< TEventType, PCFFileParser::EventTypeData > events;

std::map< std::string, std::function<PCFFileParser::SectionParser<>*(void)> > sectionParserFactory;
SectionParser<> *currentParser;

void initSectionParserFactory();
void trimAndCleanTabs( std::string& strLine );

template< typename dummyParser > friend class DefaultOptionsParser;
template< typename dummyParser > friend class DefaultSemanticParser;
template< typename dummyParser > friend class StatesParser;
template< typename dummyParser > friend class StatesColorParser;
template< typename dummyParser > friend class EventParser;
template< typename dummyParser > friend class GradientColorParser;
template< typename dummyParser > friend class GradientNamesParser;
};

#include "pcffileparser.cpp"
