
#ifndef BOOST_DLL_DETAIL_POSIX_ELF_INFO_HPP
#define BOOST_DLL_DETAIL_POSIX_ELF_INFO_HPP

#include <boost/dll/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

#include <cstring>
#include <fstream>
#include <limits>

#include <boost/cstdint.hpp>
#include <boost/throw_exception.hpp>

namespace boost { namespace dll { namespace detail {

template <class AddressOffsetT>
struct Elf_Ehdr_template {
unsigned char     e_ident[16];    
boost::uint16_t   e_type;         
boost::uint16_t   e_machine;      
boost::uint32_t   e_version;      
AddressOffsetT    e_entry;        
AddressOffsetT    e_phoff;        
AddressOffsetT    e_shoff;        
boost::uint32_t   e_flags;        
boost::uint16_t   e_ehsize;       
boost::uint16_t   e_phentsize;    
boost::uint16_t   e_phnum;        
boost::uint16_t   e_shentsize;    
boost::uint16_t   e_shnum;        
boost::uint16_t   e_shstrndx;     
};

typedef Elf_Ehdr_template<boost::uint32_t> Elf32_Ehdr_;
typedef Elf_Ehdr_template<boost::uint64_t> Elf64_Ehdr_;

template <class AddressOffsetT>
struct Elf_Shdr_template {
boost::uint32_t   sh_name;        
boost::uint32_t   sh_type;        
AddressOffsetT    sh_flags;       
AddressOffsetT    sh_addr;        
AddressOffsetT    sh_offset;      
AddressOffsetT    sh_size;        
boost::uint32_t   sh_link;        
boost::uint32_t   sh_info;        
AddressOffsetT    sh_addralign;   
AddressOffsetT    sh_entsize;     
};

typedef Elf_Shdr_template<boost::uint32_t> Elf32_Shdr_;
typedef Elf_Shdr_template<boost::uint64_t> Elf64_Shdr_;

template <class AddressOffsetT>
struct Elf_Sym_template;

template <>
struct Elf_Sym_template<boost::uint32_t> {
typedef boost::uint32_t AddressOffsetT;

boost::uint32_t   st_name;    
AddressOffsetT    st_value;   
AddressOffsetT    st_size;    
unsigned char     st_info;    
unsigned char     st_other;   
boost::uint16_t   st_shndx;   
};

template <>
struct Elf_Sym_template<boost::uint64_t> {
typedef boost::uint64_t AddressOffsetT;

boost::uint32_t   st_name;    
unsigned char     st_info;    
unsigned char     st_other;   
boost::uint16_t   st_shndx;   
AddressOffsetT    st_value;   
AddressOffsetT    st_size;    
};


typedef Elf_Sym_template<boost::uint32_t> Elf32_Sym_;
typedef Elf_Sym_template<boost::uint64_t> Elf64_Sym_;

template <class AddressOffsetT>
class elf_info {
typedef boost::dll::detail::Elf_Ehdr_template<AddressOffsetT>  header_t;
typedef boost::dll::detail::Elf_Shdr_template<AddressOffsetT>  section_t;
typedef boost::dll::detail::Elf_Sym_template<AddressOffsetT>   symbol_t;

BOOST_STATIC_CONSTANT(boost::uint32_t, SHT_SYMTAB_ = 2);
BOOST_STATIC_CONSTANT(boost::uint32_t, SHT_STRTAB_ = 3);

BOOST_STATIC_CONSTANT(unsigned char, STB_LOCAL_ = 0);   
BOOST_STATIC_CONSTANT(unsigned char, STB_GLOBAL_ = 1);  
BOOST_STATIC_CONSTANT(unsigned char, STB_WEAK_ = 2);    


BOOST_STATIC_CONSTANT(unsigned char, STV_DEFAULT_ = 0);      
BOOST_STATIC_CONSTANT(unsigned char, STV_INTERNAL_ = 1);     
BOOST_STATIC_CONSTANT(unsigned char, STV_HIDDEN_ = 2);       
BOOST_STATIC_CONSTANT(unsigned char, STV_PROTECTED_ = 3);    

public:
static bool parsing_supported(std::ifstream& fs) {
const unsigned char magic_bytes[5] = { 
0x7f, 'E', 'L', 'F', sizeof(boost::uint32_t) == sizeof(AddressOffsetT) ? 1 : 2
};

unsigned char ch;
fs.seekg(0);
for (std::size_t i = 0; i < sizeof(magic_bytes); ++i) {
fs >> ch;
if (ch != magic_bytes[i]) {
return false;
}
}

return true;
}

static std::vector<std::string> sections(std::ifstream& fs) {
std::vector<std::string> ret;
std::vector<char> names;
sections_names_raw(fs, names);

const char* name_begin = &names[0];
const char* const name_end = name_begin + names.size();
ret.reserve(header(fs).e_shnum);
do {
ret.push_back(name_begin);
name_begin += ret.back().size() + 1;
} while (name_begin != name_end);

return ret;
}

private:
template <class Integer>
static void checked_seekg(std::ifstream& fs, Integer pos) {

fs.seekg(static_cast<std::streamoff>(pos));
}

template <class T>
static void read_raw(std::ifstream& fs, T& value, std::size_t size = sizeof(T)) {
fs.read(reinterpret_cast<char*>(&value), size);
}

static header_t header(std::ifstream& fs) {
header_t elf;

fs.seekg(0);
read_raw(fs, elf);

return elf;
}

static void sections_names_raw(std::ifstream& fs, std::vector<char>& sections) {
const header_t elf = header(fs);

section_t section_names_section;
checked_seekg(fs, elf.e_shoff + elf.e_shstrndx * sizeof(section_t));
read_raw(fs, section_names_section);

sections.resize(static_cast<std::size_t>(section_names_section.sh_size));
checked_seekg(fs, section_names_section.sh_offset);
read_raw(fs, sections[0], static_cast<std::size_t>(section_names_section.sh_size));
}

static void symbols_text(std::ifstream& fs, std::vector<symbol_t>& symbols, std::vector<char>& text) {
const header_t elf = header(fs);
checked_seekg(fs, elf.e_shoff);

for (std::size_t i = 0; i < elf.e_shnum; ++i) {
section_t section;
read_raw(fs, section);

if (section.sh_type == SHT_SYMTAB_) {
symbols.resize(static_cast<std::size_t>(section.sh_size / sizeof(symbol_t)));

const std::ifstream::pos_type pos = fs.tellg();
checked_seekg(fs, section.sh_offset);
read_raw(fs, symbols[0], static_cast<std::size_t>(section.sh_size - (section.sh_size % sizeof(symbol_t))) );
checked_seekg(fs, pos);
} else if (section.sh_type == SHT_STRTAB_) {
text.resize(static_cast<std::size_t>(section.sh_size));

const std::ifstream::pos_type pos = fs.tellg();
checked_seekg(fs, section.sh_offset);
read_raw(fs, text[0], static_cast<std::size_t>(section.sh_size));
checked_seekg(fs, pos);
}
}
}

static bool is_visible(const symbol_t& sym) BOOST_NOEXCEPT {
return (sym.st_other & 0x03) == STV_DEFAULT_ && (sym.st_info >> 4) != STB_LOCAL_ && !!sym.st_size;
}

public:
static std::vector<std::string> symbols(std::ifstream& fs) {
std::vector<std::string> ret;

std::vector<symbol_t> symbols;
std::vector<char>   text;
symbols_text(fs, symbols, text);

ret.reserve(symbols.size());
for (std::size_t i = 0; i < symbols.size(); ++i) {
if (is_visible(symbols[i])) {
ret.push_back(&text[0] + symbols[i].st_name);
if (ret.back().empty()) {
ret.pop_back(); 
}
}
}

return ret;
}

static std::vector<std::string> symbols(std::ifstream& fs, const char* section_name) {
std::vector<std::string> ret;

std::size_t index = 0;
std::size_t ptrs_in_section_count = 0;
{
std::vector<char> names;
sections_names_raw(fs, names);

const header_t elf = header(fs);

for (; index < elf.e_shnum; ++index) {
section_t section;
checked_seekg(fs, elf.e_shoff + index * sizeof(section_t));
read_raw(fs, section);

if (!std::strcmp(&names[0] + section.sh_name, section_name)) {
if (!section.sh_entsize) {
section.sh_entsize = 1;
}
ptrs_in_section_count = static_cast<std::size_t>(section.sh_size / section.sh_entsize);
break;
}
}                        
}

std::vector<symbol_t> symbols;
std::vector<char>   text;
symbols_text(fs, symbols, text);

if (ptrs_in_section_count < symbols.size()) {
ret.reserve(ptrs_in_section_count);
} else {
ret.reserve(symbols.size());
}

for (std::size_t i = 0; i < symbols.size(); ++i) {
if (symbols[i].st_shndx == index && is_visible(symbols[i])) {
ret.push_back(&text[0] + symbols[i].st_name);
if (ret.back().empty()) {
ret.pop_back(); 
}
}
}

return ret;
}
};

typedef elf_info<boost::uint32_t> elf_info32;
typedef elf_info<boost::uint64_t> elf_info64;

}}} 

#endif 
