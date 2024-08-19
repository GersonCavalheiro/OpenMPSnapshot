


#pragma once


#include <fstream>
#include <string>
#include <zlib.h>
#include "paraverkerneltypes.h"

class TraceStream
{
public:
TraceStream()
{}

TraceStream( const std::string& filename )
{
this->filename = filename;
}

virtual ~TraceStream()
{}

virtual void open( const std::string& filename ) = 0;
virtual void close() = 0;
virtual void getline( std::string& strLine ) = 0;
virtual bool eof() = 0;
virtual void seekbegin() = 0;
virtual void seekend() = 0;
virtual void seekg( std::streampos pos ) = 0;
virtual std::streampos tellg() = 0;
virtual bool canseekend() = 0;
virtual bool good() const = 0;
virtual void clear() = 0;
virtual int peek() = 0;

static TraceStream *openFile( const std::string& filename );

static TTraceSize getTraceFileSize( const std::string& filename );

static const double GZIP_COMPRESSION_RATIO;

virtual std::string getFilename() const;
virtual void setFilename( const std::string &newFile );
protected:
std::string filename;
};


class NotCompressed: public TraceStream
{
public:
NotCompressed()
{}

NotCompressed( const std::string& filename );

virtual ~NotCompressed()
{}

virtual void open( const std::string& filename ) override;
virtual void close() override;
virtual void getline( std::string& strLine ) override;
virtual bool eof() override;
virtual void seekbegin() override;
virtual void seekend() override;
virtual void seekg( std::streampos pos ) override;
virtual std::streampos tellg() override;
virtual bool canseekend() override;
virtual bool good() const override;
virtual void clear() override;
virtual int peek() override;

static TTraceSize getTraceFileSize( const std::string& filename );

private:
std::ifstream file;

};


class Compressed: public TraceStream
{
public:
Compressed()
{}

Compressed( const std::string& filename );

virtual ~Compressed()
{}

virtual void open( const std::string& filename ) override;
virtual void close() override;
virtual void getline( std::string& strLine ) override;
virtual bool eof() override;
virtual void seekbegin() override;
virtual void seekend() override;
virtual void seekg( std::streampos pos ) override;
virtual std::streampos tellg() override;
virtual bool canseekend() override;
virtual bool good() const override;
virtual void clear() override;
virtual int peek() override;

static TTraceSize getTraceFileSize( const std::string& filename );

private:
static const PRV_UINT32 LINESIZE = 1000 * 1024;
gzFile file;
char tmpLine[ LINESIZE ];

};


