
#ifndef BOOST_DLL_DETAIL_WINDOWS_PE_INFO_HPP
#define BOOST_DLL_DETAIL_WINDOWS_PE_INFO_HPP

#include <boost/dll/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

#include <cstring>
#include <fstream>
#include <string> 

#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

namespace boost { namespace dll { namespace detail {


typedef unsigned char BYTE_;
typedef unsigned short WORD_;
typedef boost::uint32_t DWORD_;
typedef boost::int32_t LONG_;
typedef boost::uint32_t ULONG_;
typedef boost::int64_t LONGLONG_;
typedef boost::uint64_t ULONGLONG_;

struct IMAGE_DOS_HEADER_ { 
boost::dll::detail::WORD_   e_magic;        
boost::dll::detail::WORD_   e_cblp;         
boost::dll::detail::WORD_   e_cp;           
boost::dll::detail::WORD_   e_crlc;         
boost::dll::detail::WORD_   e_cparhdr;      
boost::dll::detail::WORD_   e_minalloc;     
boost::dll::detail::WORD_   e_maxalloc;     
boost::dll::detail::WORD_   e_ss;           
boost::dll::detail::WORD_   e_sp;           
boost::dll::detail::WORD_   e_csum;         
boost::dll::detail::WORD_   e_ip;           
boost::dll::detail::WORD_   e_cs;           
boost::dll::detail::WORD_   e_lfarlc;       
boost::dll::detail::WORD_   e_ovno;         
boost::dll::detail::WORD_   e_res[4];       
boost::dll::detail::WORD_   e_oemid;        
boost::dll::detail::WORD_   e_oeminfo;      
boost::dll::detail::WORD_   e_res2[10];     
boost::dll::detail::LONG_   e_lfanew;       
};

struct IMAGE_FILE_HEADER_ { 
boost::dll::detail::WORD_   Machine;
boost::dll::detail::WORD_   NumberOfSections;
boost::dll::detail::DWORD_  TimeDateStamp;
boost::dll::detail::DWORD_  PointerToSymbolTable;
boost::dll::detail::DWORD_  NumberOfSymbols;
boost::dll::detail::WORD_   SizeOfOptionalHeader;
boost::dll::detail::WORD_   Characteristics;
};

struct IMAGE_DATA_DIRECTORY_ { 
boost::dll::detail::DWORD_  VirtualAddress;
boost::dll::detail::DWORD_  Size;
};

struct IMAGE_EXPORT_DIRECTORY_ { 
boost::dll::detail::DWORD_  Characteristics;
boost::dll::detail::DWORD_  TimeDateStamp;
boost::dll::detail::WORD_   MajorVersion;
boost::dll::detail::WORD_   MinorVersion;
boost::dll::detail::DWORD_  Name;
boost::dll::detail::DWORD_  Base;
boost::dll::detail::DWORD_  NumberOfFunctions;
boost::dll::detail::DWORD_  NumberOfNames;
boost::dll::detail::DWORD_  AddressOfFunctions;
boost::dll::detail::DWORD_  AddressOfNames;
boost::dll::detail::DWORD_  AddressOfNameOrdinals;
};

struct IMAGE_SECTION_HEADER_ { 
static const std::size_t    IMAGE_SIZEOF_SHORT_NAME_ = 8;

boost::dll::detail::BYTE_   Name[IMAGE_SIZEOF_SHORT_NAME_];
union {
boost::dll::detail::DWORD_   PhysicalAddress;
boost::dll::detail::DWORD_   VirtualSize;
} Misc;
boost::dll::detail::DWORD_  VirtualAddress;
boost::dll::detail::DWORD_  SizeOfRawData;
boost::dll::detail::DWORD_  PointerToRawData;
boost::dll::detail::DWORD_  PointerToRelocations;
boost::dll::detail::DWORD_  PointerToLinenumbers;
boost::dll::detail::WORD_   NumberOfRelocations;
boost::dll::detail::WORD_   NumberOfLinenumbers;
boost::dll::detail::DWORD_  Characteristics;
};


template <class AddressOffsetT>
struct IMAGE_OPTIONAL_HEADER_template {
static const std::size_t IMAGE_NUMBEROF_DIRECTORY_ENTRIES_ = 16;

boost::dll::detail::WORD_   Magic;
boost::dll::detail::BYTE_   MajorLinkerVersion;
boost::dll::detail::BYTE_   MinorLinkerVersion;
boost::dll::detail::DWORD_  SizeOfCode;
boost::dll::detail::DWORD_  SizeOfInitializedData;
boost::dll::detail::DWORD_  SizeOfUninitializedData;
boost::dll::detail::DWORD_  AddressOfEntryPoint;
union {
boost::dll::detail::DWORD_   BaseOfCode;
unsigned char padding_[sizeof(AddressOffsetT) == 8 ? 4 : 8]; 
} BaseOfCode_and_BaseOfData;

AddressOffsetT              ImageBase;
boost::dll::detail::DWORD_  SectionAlignment;
boost::dll::detail::DWORD_  FileAlignment;
boost::dll::detail::WORD_   MajorOperatingSystemVersion;
boost::dll::detail::WORD_   MinorOperatingSystemVersion;
boost::dll::detail::WORD_   MajorImageVersion;
boost::dll::detail::WORD_   MinorImageVersion;
boost::dll::detail::WORD_   MajorSubsystemVersion;
boost::dll::detail::WORD_   MinorSubsystemVersion;
boost::dll::detail::DWORD_  Win32VersionValue;
boost::dll::detail::DWORD_  SizeOfImage;
boost::dll::detail::DWORD_  SizeOfHeaders;
boost::dll::detail::DWORD_  CheckSum;
boost::dll::detail::WORD_   Subsystem;
boost::dll::detail::WORD_   DllCharacteristics;
AddressOffsetT              SizeOfStackReserve;
AddressOffsetT              SizeOfStackCommit;
AddressOffsetT              SizeOfHeapReserve;
AddressOffsetT              SizeOfHeapCommit;
boost::dll::detail::DWORD_  LoaderFlags;
boost::dll::detail::DWORD_  NumberOfRvaAndSizes;
IMAGE_DATA_DIRECTORY_       DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES_];
};

typedef IMAGE_OPTIONAL_HEADER_template<boost::dll::detail::DWORD_>      IMAGE_OPTIONAL_HEADER32_;
typedef IMAGE_OPTIONAL_HEADER_template<boost::dll::detail::ULONGLONG_>  IMAGE_OPTIONAL_HEADER64_;

template <class AddressOffsetT>
struct IMAGE_NT_HEADERS_template {
boost::dll::detail::DWORD_                      Signature;
IMAGE_FILE_HEADER_                              FileHeader;
IMAGE_OPTIONAL_HEADER_template<AddressOffsetT>  OptionalHeader;
};

typedef IMAGE_NT_HEADERS_template<boost::dll::detail::DWORD_>      IMAGE_NT_HEADERS32_;
typedef IMAGE_NT_HEADERS_template<boost::dll::detail::ULONGLONG_>  IMAGE_NT_HEADERS64_;


template <class AddressOffsetT>
class pe_info {
typedef IMAGE_NT_HEADERS_template<AddressOffsetT>   header_t;
typedef IMAGE_EXPORT_DIRECTORY_                     exports_t;
typedef IMAGE_SECTION_HEADER_                       section_t;
typedef IMAGE_DOS_HEADER_                           dos_t;

template <class T>
static void read_raw(std::ifstream& fs, T& value, std::size_t size = sizeof(T)) {
fs.read(reinterpret_cast<char*>(&value), size);
}

public:
static bool parsing_supported(std::ifstream& fs) {
dos_t dos;
fs.seekg(0);
fs.read(reinterpret_cast<char*>(&dos), sizeof(dos));

if (dos.e_magic != 0x4D5A && dos.e_magic != 0x5A4D) {
return false;
}

header_t h;
fs.seekg(dos.e_lfanew);
fs.read(reinterpret_cast<char*>(&h), sizeof(h));

return h.Signature == 0x00004550 
&& h.OptionalHeader.Magic == (sizeof(boost::uint32_t) == sizeof(AddressOffsetT) ? 0x10B : 0x20B);
}

private:
static header_t header(std::ifstream& fs) {
header_t h;

dos_t dos;
fs.seekg(0);
read_raw(fs, dos);

fs.seekg(dos.e_lfanew);
read_raw(fs, h);

return h;
}

static exports_t exports(std::ifstream& fs, const header_t& h) {
static const unsigned int IMAGE_DIRECTORY_ENTRY_EXPORT_ = 0;
const std::size_t exp_virtual_address = h.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT_].VirtualAddress;
exports_t exports;

if (exp_virtual_address == 0) {
std::memset(&exports, 0, sizeof(exports));
return exports;
}

const std::size_t real_offset = get_file_offset(fs, exp_virtual_address, h);
BOOST_ASSERT(real_offset);

fs.seekg(real_offset);
read_raw(fs, exports);

return exports;
}

static std::size_t get_file_offset(std::ifstream& fs, std::size_t virtual_address, const header_t& h) {
BOOST_ASSERT(virtual_address);

section_t image_section_header;

{   
dos_t dos;
fs.seekg(0);
read_raw(fs, dos);
fs.seekg(dos.e_lfanew + sizeof(header_t));
}

for (std::size_t i = 0;i < h.FileHeader.NumberOfSections;++i) {
read_raw(fs, image_section_header);
if (virtual_address >= image_section_header.VirtualAddress 
&& virtual_address < image_section_header.VirtualAddress + image_section_header.SizeOfRawData) 
{
return image_section_header.PointerToRawData + virtual_address - image_section_header.VirtualAddress;
}
}

return 0;
}

public:
static std::vector<std::string> sections(std::ifstream& fs) {
std::vector<std::string> ret;

const header_t h = header(fs);
ret.reserve(h.FileHeader.NumberOfSections);

section_t image_section_header;
char name_helper[section_t::IMAGE_SIZEOF_SHORT_NAME_ + 1];
std::memset(name_helper, 0, sizeof(name_helper));
for (std::size_t i = 0;i < h.FileHeader.NumberOfSections;++i) {
read_raw(fs, image_section_header);
std::memcpy(name_helper, image_section_header.Name, section_t::IMAGE_SIZEOF_SHORT_NAME_);

if (name_helper[0] != '/') {
ret.push_back(name_helper);
} else {
ret.push_back(name_helper);
}
}

return ret;
}

static std::vector<std::string> symbols(std::ifstream& fs) {
std::vector<std::string> ret;

const header_t h = header(fs);
const exports_t exprt = exports(fs, h);
const std::size_t exported_symbols = exprt.NumberOfNames;

if (exported_symbols == 0) {
return ret;
}

const std::size_t fixed_names_addr = get_file_offset(fs, exprt.AddressOfNames, h);

ret.reserve(exported_symbols);
boost::dll::detail::DWORD_ name_offset;
std::string symbol_name;
for (std::size_t i = 0;i < exported_symbols;++i) {
fs.seekg(fixed_names_addr + i * sizeof(name_offset));
read_raw(fs, name_offset);
fs.seekg(get_file_offset(fs, name_offset, h));
std::getline(fs, symbol_name, '\0');
ret.push_back(symbol_name);
}

return ret;
}

static std::vector<std::string> symbols(std::ifstream& fs, const char* section_name) {
std::vector<std::string> ret;

const header_t h = header(fs);

std::size_t section_begin_addr = 0;
std::size_t section_end_addr = 0;

{   
section_t image_section_header;
char name_helper[section_t::IMAGE_SIZEOF_SHORT_NAME_ + 1];
std::memset(name_helper, 0, sizeof(name_helper));
for (std::size_t i = 0;i < h.FileHeader.NumberOfSections;++i) {
read_raw(fs, image_section_header);
std::memcpy(name_helper, image_section_header.Name, section_t::IMAGE_SIZEOF_SHORT_NAME_);
if (!std::strcmp(section_name, name_helper)) {
section_begin_addr = image_section_header.PointerToRawData;
section_end_addr = section_begin_addr + image_section_header.SizeOfRawData;
}
}

if(section_begin_addr == 0 || section_end_addr == 0)
return ret;
}

const exports_t exprt = exports(fs, h);
const std::size_t exported_symbols = exprt.NumberOfFunctions;
const std::size_t fixed_names_addr = get_file_offset(fs, exprt.AddressOfNames, h);
const std::size_t fixed_ordinals_addr = get_file_offset(fs, exprt.AddressOfNameOrdinals, h);
const std::size_t fixed_functions_addr = get_file_offset(fs, exprt.AddressOfFunctions, h);

ret.reserve(exported_symbols);
boost::dll::detail::DWORD_ ptr;
boost::dll::detail::WORD_ ordinal;
std::string symbol_name;
for (std::size_t i = 0;i < exported_symbols;++i) {
fs.seekg(fixed_ordinals_addr + i * sizeof(ordinal));
read_raw(fs, ordinal);

fs.seekg(fixed_functions_addr + ordinal * sizeof(ptr));
read_raw(fs, ptr);
ptr = static_cast<boost::dll::detail::DWORD_>( get_file_offset(fs, ptr, h) );

if (ptr >= section_end_addr || ptr < section_begin_addr) {
continue;
}

fs.seekg(fixed_names_addr + i * sizeof(ptr));
read_raw(fs, ptr);
fs.seekg(get_file_offset(fs, ptr, h));
std::getline(fs, symbol_name, '\0');
ret.push_back(symbol_name);
}

return ret;
}



};

typedef pe_info<boost::dll::detail::DWORD_>      pe_info32;
typedef pe_info<boost::dll::detail::ULONGLONG_>  pe_info64;

}}} 

#endif 
