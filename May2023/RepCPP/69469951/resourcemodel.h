


#pragma once


#include <vector>
#include <string>
#include "resourcemodelnode.h"

template< typename NodeOrderT = TNodeOrder,
typename CPUOrderT = TCPUOrder >
class ResourceModel
{
public:
ResourceModel()
{
ready = false;
}

ResourceModel( std::istringstream& headerInfo );

~ResourceModel()
{}

bool operator<( const ResourceModel< NodeOrderT, CPUOrderT >& other ) const;
bool operator==( const ResourceModel< NodeOrderT, CPUOrderT >& other ) const;

bool isReady() const
{
return ready;
}

void setReady( bool newValue )
{
ready = newValue;
}

size_t size() const { return nodes.size(); }
typename std::vector< ResourceModelNode< NodeOrderT, CPUOrderT > >::const_iterator cbegin() const { return nodes.cbegin(); }
typename std::vector< ResourceModelNode< NodeOrderT, CPUOrderT > >::const_iterator cend() const { return nodes.cend(); }

NodeOrderT totalNodes() const;
CPUOrderT totalCPUs() const;

CPUOrderT getGlobalCPU( const NodeOrderT& inNode,
const CPUOrderT& inCPU ) const;
void getCPULocation( CPUOrderT globalCPU,
NodeOrderT& inNode,
CPUOrderT& inCPU ) const;
CPUOrderT getFirstCPU( NodeOrderT inNode ) const;
CPUOrderT getLastCPU( NodeOrderT inNode ) const;

void addNode();
void addCPU( NodeOrderT whichNode );

bool isValidNode( NodeOrderT whichNode ) const;
bool isValidCPU( NodeOrderT whichNode, CPUOrderT whichCPU ) const;
bool isValidGlobalCPU( CPUOrderT whichCPU ) const;

protected:
struct CPULocation
{
NodeOrderT node;
CPUOrderT CPU;

bool operator==( const CPULocation& other ) const
{
return node == other.node &&
CPU  == other.CPU;
}
};

std::vector< CPULocation > CPUs;
std::vector< ResourceModelNode< NodeOrderT, CPUOrderT > > nodes;
bool ready;

private:

};

template< typename ResourceModelT >
void dumpResourceModelToFile( const ResourceModelT& resourceModel, std::fstream& file );


#include "resourcemodel.cpp"
