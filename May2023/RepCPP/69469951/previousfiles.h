


#pragma once


#include <string>
#include <fstream>
#include <vector>

#include "paraverkerneltypes.h"

class PreviousFiles
{
public:
static PreviousFiles *createPreviousTraces();
static PreviousFiles *createPreviousCFGs();
static PreviousFiles *createPreviousSessions();
static PreviousFiles *createPreviousTreatedTraces();

~PreviousFiles();

bool add( const std::string &newFile );
const std::vector<std::string>& getFiles() const;

static const std::string previousTracesFile;
static const std::string previousCFGsFile;
static const std::string previousSessionsFile;
static const std::string previousTreatedTracesFile;

static const PRV_UINT16 SIZE = 20;

private:
PreviousFiles( const std::string &filename, bool purge = false );

std::fstream myFile;
std::string  myFileName;
std::vector< std::string > listFiles;

void create();
bool update();
void read( std::fstream &myFile );
};



