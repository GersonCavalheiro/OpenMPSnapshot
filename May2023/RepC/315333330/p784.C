typedef void* Pix;
extern "C" {
typedef long int ptrdiff_t;
typedef unsigned int size_t;
}
class ostream; class streambuf;
typedef long streamoff, streampos;
struct _ios_fields { 
streambuf *_strbuf;
ostream* _tie;
long _width;
unsigned long _flags;
char _fill;
unsigned char _state;
unsigned short _precision;
};
enum state_value { _good = 0, _eof = 1,  _fail = 2, _bad  = 4 };
class ios : public _ios_fields {
public:
enum io_state { goodbit=0, eofbit=1, failbit=2, badbit=4 };
enum open_mode {
in=1,
out=2,
ate=4,
app=8,
trunc=16,
nocreate=32,
noreplace=64 };
enum seek_dir { beg, cur, end};
enum { skipws=01, left=02, right=04, internal=010,
dec=020, oct=040, hex=0100,
showbase=0200, showpoint=0400, uppercase=01000, showpos=02000,
scientific=04000, fixed=0100000, unitbuf=020000, stdio=040000,
dont_close=0x80000000 
};
ostream* tie() { return _tie; }
ostream* tie(ostream* val) { ostream* save=_tie; _tie=val; return save; }
char fill() { return _fill; }
char fill(char newf) { char oldf = _fill; _fill = newf; return oldf; }
unsigned long flags() { return _flags; }
unsigned long flags(unsigned long new_val) {
unsigned long old_val = _flags; _flags = new_val; return old_val; }
unsigned short precision() { return _precision; }
unsigned short precision(int newp) {
unsigned short oldp = _precision; _precision = (unsigned short)newp;
return oldp; }
unsigned long setf(unsigned long val) {
unsigned long oldbits = _flags;
_flags |= val; return oldbits; }
unsigned long setf(unsigned long val, unsigned long mask) {
unsigned long oldbits = _flags;
_flags = (_flags & ~mask) | (val & mask); return oldbits; }
unsigned long unsetf(unsigned long mask) {
unsigned long oldbits = _flags & mask;
_flags &= ~mask; return oldbits; }
long width() { return _width; }
long width(long val) { long save = _width; _width = val; return save; }
static const unsigned long basefield;
static const unsigned long adjustfield;
static const unsigned long floatfield;
streambuf* rdbuf() { return _strbuf; }
void clear(int state = 0) { _state = state; }
int good() { return _state == 0; }
int eof() { return _state & ios::eofbit; }
int fail() { return _state & (ios::badbit|ios::failbit); }
int bad() { return _state & ios::badbit; }
int rdstate() { return _state; }
void set(int flag) { _state |= flag; }
operator void*() { return fail() ? (void*)0 : (void*)this; }
int operator!() { return fail(); }
void unset(state_value flag) { _state &= ~flag; }
void close();
int is_open();
int readable();
int writable();
protected:
ios(streambuf*sb) { _strbuf=sb; _state=0; _width=0; _fill=' ';
_flags=ios::skipws; _precision=6; }
};
typedef ios::seek_dir _seek_dir;
struct __streambuf {
int _flags;		
char* _gptr;	
char* _egptr;	
char* _eback;	
char* _pbase;	
char* _pptr;	
char* _epptr;	
char* _base;	
char* _ebuf;	
struct streambuf *_chain;
};
struct streambuf : private __streambuf {
friend class ios;
friend class istream;
friend class ostream;
protected:
static streambuf* _list_all; 
streambuf*& xchain() { return _chain; }
void _un_link();
void _link_in();
char* gptr() const { return _gptr; }
char* pptr() const { return _pptr; }
char* egptr() const { return _egptr; }
char* epptr() const { return _epptr; }
char* pbase() const { return _pbase; }
char* eback() const { return _eback; }
char* ebuf() const { return _ebuf; }
char* base() const { return _base; }
void xput_char(char c) { *_pptr++ = c; }
int xflags() { return _flags; }
int xflags(int f) { int fl = _flags; _flags = f; return fl; }
void xsetflags(int f) { _flags |= f; }
void gbump(int n) { _gptr += n; }
void pbump(int n) { _pptr += n; }
void setb(char* b, char* eb, int a=0);
void setp(char* p, char* ep) { _pbase=_pptr=p; _epptr=ep; }
void setg(char* eb, char* g, char *eg) { _eback=eb; _gptr=g; _egptr=eg; }
public:
static int flush_all();
static void flush_all_linebuffered(); 
virtual int underflow(); 
virtual int overflow(int c = (-1) ); 
virtual int doallocate();
virtual streampos seekoff(streamoff, _seek_dir, int mode=ios::in|ios::out);
virtual streampos seekpos(streampos pos, int mode = ios::in|ios::out);
int sputbackc(char c);
int sungetc();
streambuf();
virtual ~streambuf();
int unbuffered() { return _flags & 2  ? 1 : 0; }
int linebuffered() { return _flags & 0x4000  ? 1 : 0; }
void unbuffered(int i)
{ if (i) _flags |= 2 ; else _flags &= ~2 ; }
void linebuffered(int i)
{ if (i) _flags |= 0x4000 ; else _flags &= ~0x4000 ; }
int allocate() {
if (base() || unbuffered()) return 0;
else return doallocate(); }
virtual int sync();
virtual int pbackfail(int c);
virtual int ungetfail();
virtual streambuf* setbuf(char* p, int len);
int in_avail() { return _egptr - _gptr; }
int out_waiting() { return _pptr - _pbase; }
virtual int sputn(const char* s, int n);
virtual int sgetn(char* s, int n);
long sgetline(char* buf, size_t n, char delim, int putback_delim);
int sbumpc() {
if (_gptr >= _egptr && underflow() == (-1) ) return (-1) ;
else return *(unsigned char*)_gptr++; }
int sgetc() {
if (_gptr >= _egptr && underflow() == (-1) ) return (-1) ;
else return *(unsigned char*)_gptr; }
int snextc() {
if (++_gptr >= _egptr && underflow() == (-1) ) return (-1) ;
else return *(unsigned char*)_gptr; }
int sputc(int c) {
if (_pptr >= _epptr) return overflow(c);
return *_pptr++ = c, (unsigned char)c; }
int vscan(char const *fmt0, char*  ap);
int vform(char const *fmt0, char*  ap);
};
struct __file_fields {
char _fake;
char _shortbuf[1];
short _fileno;
int _blksize;
char* _save_gptr;
char* _save_egptr;
long  _offset;
};
class filebuf : public streambuf {
struct __file_fields _fb;
void init();
public:
filebuf();
filebuf(int fd);
filebuf(int fd, char* p, int len);
~filebuf();
filebuf* attach(int fd);
filebuf* open(const char *filename, const char *mode);
filebuf* open(const char *filename, int mode, int prot = 0664);
virtual int underflow();
virtual int overflow(int c = (-1) );
int is_open() { return _fb._fileno >= 0; }
int fd() { return is_open() ? _fb._fileno : (-1) ; }
filebuf* close();
virtual int doallocate();
virtual streampos seekoff(streamoff, _seek_dir, int mode=ios::in|ios::out);
int sputn(const char* s, int n);
int sgetn(char* s, int n);
protected: 
virtual int pbackfail(int c);
virtual int sync();
int is_reading() { return eback() != egptr(); }
char* cur_ptr() { return is_reading() ?  gptr() : pptr(); }
char* file_ptr() { return _fb._save_gptr ? _fb._save_egptr : egptr(); }
int do_flush();
virtual int sys_read(char* buf, size_t size);
virtual long  sys_seek(long , _seek_dir);
virtual long sys_write(const void*, long);
virtual int sys_stat(void*); 
virtual int sys_close();
};
inline int ios::readable() { return rdbuf()->_flags & 4 ; }
inline int ios::writable() { return rdbuf()->_flags & 8 ; }
inline int ios::is_open() {return rdbuf()->_flags & 4 +8 ;}
class istream; class ostream;
typedef istream& (*__imanip)(istream&);
typedef ostream& (*__omanip)(ostream&);
extern istream& ws(istream& ins);
extern ostream& flush(ostream& outs);
extern ostream& endl(ostream& outs);
extern ostream& ends(ostream& outs);
class ostream : public ios
{
void do_osfx();
public:
ostream();
ostream(streambuf* sb, ostream* tied=(__null) );
~ostream();
int opfx() { if (!good()) return 0; if (_tie) _tie->flush(); return 1; }
void osfx() { if (flags() & (ios::unitbuf|ios::stdio))
do_osfx(); }
streambuf* ostreambuf() const { return _strbuf; }
ostream& flush();
ostream& put(char c);
ostream& write(const char *s, int n);
ostream& write(const unsigned char *s, int n) { return write((char*)s, n);}
ostream& write(const void *s, int n) { return write((char*)s, n);}
ostream& seekp(streampos);
ostream& seekp(streamoff, _seek_dir);
streampos tellp();
ostream& form(const char *format ...);
ostream& vform(const char *format, char*  args);
};
ostream& operator<<(ostream&, char c);
ostream& operator<<(ostream& os, unsigned char c) { return os << (char)c; }
extern ostream& operator<<(ostream&, const char *s);
inline ostream& operator<<(ostream& os, const unsigned char *s)
{ return os << (const char*)s; }
ostream& operator<<(ostream&, void *p);
ostream& operator<<(ostream&, int n);
ostream& operator<<(ostream&, long n);
ostream& operator<<(ostream&, unsigned int n);
ostream& operator<<(ostream&, unsigned long n);
ostream& operator<<(ostream& os, short n) {return os << (int)n;}
ostream& operator<<(ostream& os, unsigned short n)
{return os << (unsigned int)n;}
ostream& operator<<(ostream&, float n);
ostream& operator<<(ostream&, double n);
ostream& operator<<(ostream& os, __omanip func) { return (*func)(os); }
ostream& operator<<(ostream&, streambuf*);
class istream : public ios
{
size_t _gcount;
public:
istream();
istream(streambuf* sb, ostream*tied=(__null) );
~istream();
streambuf* istreambuf() const { return _strbuf; }
istream& get(char& c);
istream& get(unsigned char& c);
istream& read(char *ptr, int n);
istream& read(unsigned char *ptr, int n) { return read((char*)ptr, n); }
istream& read(void *ptr, int n) { return read((char*)ptr, n); }
int get() { return _strbuf->sbumpc(); }
istream& getline(char* ptr, int len, char delim = '\n');
istream& get(char* ptr, int len, char delim = '\n');
istream& gets(char **s, char delim = '\n');
int ipfx(int need) {
if (!good()) { set(ios::failbit); return 0; }
if (_tie && (need == 0 || rdbuf()->in_avail())) ;  
if (!need && (flags() & ios::skipws) && !ws(*this)) return 0;
return 1;
}
int ipfx0() { 
if (!good()) { set(ios::failbit); return 0; }
if (_tie) _tie->flush();
if ((flags() & ios::skipws) && !ws(*this)) return 0;
return 1;
}
int ipfx1() { 
if (!good()) { set(ios::failbit); return 0; }
if (_tie && rdbuf()->in_avail() == 0) _tie->flush();
return 1;
}
size_t gcount() { return _gcount; }
istream& seekg(streampos);
istream& seekg(streamoff, _seek_dir);
streampos tellg();
istream& putback(char ch) {
if (good() && _strbuf->sputbackc(ch) == (-1) ) clear(ios::badbit);
return *this;}
istream& unget() {
if (good() && _strbuf->sungetc() == (-1) ) clear(ios::badbit);
return *this;}
istream& unget(char ch) { return putback(ch); }
int skip(int i);
};
istream& operator>>(istream&, char*);
istream& operator>>(istream& is, unsigned char* p) { return is >> (char*)p; }
istream& operator>>(istream&, char& c);
istream& operator>>(istream&, unsigned char& c);
istream& operator>>(istream&, int&);
istream& operator>>(istream&, long&);
istream& operator>>(istream&, short&);
istream& operator>>(istream&, unsigned int&);
istream& operator>>(istream&, unsigned long&);
istream& operator>>(istream&, unsigned short&);
istream& operator>>(istream&, float&);
istream& operator>>(istream&, double&);
istream& operator>>(istream& is, __imanip func) { return (*func)(is); }
class iostream : public ios {
size_t _gcount;
public:
iostream();
operator istream&() { return *(istream*)this; }
operator ostream&() { return *(ostream*)this; }
~iostream();
istream& get(char& c) { return ((istream*)this)->get(c); }
istream& get(unsigned char& c) { return ((istream*)this)->get(c); }
istream& read(char *ptr, int n) { return ((istream*)this)->read(ptr, n); }
istream& read(unsigned char *ptr, int n)
{ return ((istream*)this)->read((char*)ptr, n); }
istream& read(void *ptr, int n)
{ return ((istream*)this)->read((char*)ptr, n); }
int get() { return _strbuf->sbumpc(); }
istream& getline(char* ptr, int len, char delim = '\n')
{ return ((istream*)this)->getline(ptr, len, delim); }
istream& get(char* ptr, int len, char delim = '\n')
{ return ((istream*)this)->get(ptr, len, delim); }
istream& gets(char **s, char delim = '\n')
{ return ((istream*)this)->gets(s, delim); }
int ipfx(int need) { return ((istream*)this)->ipfx(need); }
int ipfx0()  { return ((istream*)this)->ipfx0(); }
int ipfx1()  { return ((istream*)this)->ipfx1(); }
size_t gcount() { return _gcount; }
istream& putback(char ch) { return ((istream*)this)->putback(ch); }
istream& unget() { return ((istream*)this)->unget(); }
istream& seekg(streampos pos) { return ((istream*)this)->seekg(pos); }
istream& seekg(streamoff off, _seek_dir dir)
{ return ((istream*)this)->seekg(off, dir); }
streampos tellg() { return ((istream*)this)->tellg(); }
istream& unget(char ch) { return putback(ch); }
int opfx() { return ((ostream*)this)->opfx(); }
void osfx() { ((ostream*)this)->osfx(); }
ostream& flush() { return ((ostream*)this)->flush(); }
ostream& put(char c) { return ((ostream*)this)->put(c); }
ostream& write(const char *s, int n)
{ return ((ostream*)this)->write(s, n); }
ostream& write(const unsigned char *s, int n)
{ return ((ostream*)this)->write((char*)s, n); }
ostream& write(const void *s, int n)
{ return ((ostream*)this)->write((char*)s, n); }
ostream& form(const char *format ...);
ostream& vform(const char *format, char*  args)
{ return ((ostream*)this)->vform(format, args); }
ostream& seekp(streampos pos) { return ((ostream*)this)->seekp(pos); }
ostream& seekp(streamoff off, _seek_dir dir)
{ return ((ostream*)this)->seekp(off, dir); }
streampos tellp() { return ((ostream*)this)->tellp(); }
};
extern istream cin;
extern ostream cout, cerr, clog; 
inline ostream& ostream::put(char c) { _strbuf->sputc(c); return *this; }
struct Iostream_init { } ;  
extern char* form(char*, ...);
extern char* dec(long, int=0);
extern char* dec(int, int=0);
extern char* dec(unsigned long, int=0);
extern char* dec(unsigned int, int=0);
extern char* hex(long, int=0);
extern char* hex(int, int=0);
extern char* hex(unsigned long, int=0);
extern char* hex(unsigned int, int=0);
extern char* oct(long, int=0);
extern char* oct(int, int=0);
extern char* oct(unsigned long, int=0);
extern char* oct(unsigned int, int=0);
inline istream& WS(istream& str) { return ws(str); }
struct re_pattern_buffer;       
struct re_registers;
class Regex
{
private:
Regex(const Regex&) {}  
void               operator = (const Regex&) {} 
protected:
re_pattern_buffer* buf;
re_registers*      reg;
public:
Regex(const char* t,
int fast = 0,
int bufsize = 40,
const char* transtable = 0);
~Regex();
int                match(const char* s, int len, int pos = 0) const;
int                search(const char* s, int len,
int& matchlen, int startpos = 0) const;
int                match_info(int& start, int& length, int nth = 0) const;
int                OK() const;  
};
extern const Regex RXwhite;          
extern const Regex RXint;            
extern const Regex RXdouble;         
extern const Regex RXalpha;          
extern const Regex RXlowercase;      
extern const Regex RXuppercase;      
extern const Regex RXalphanum;       
extern const Regex RXidentifier;     
struct StrRep                     
{
unsigned short    len;         
unsigned short    sz;          
char              s[1];        
};
StrRep*     Salloc(StrRep*, const char*, int, int);
StrRep*     Scopy(StrRep*, StrRep*);
StrRep*     Sresize(StrRep*, int);
StrRep*     Scat(StrRep*, const char*, int, const char*, int);
StrRep*     Scat(StrRep*, const char*, int,const char*,int, const char*,int);
StrRep*     Sprepend(StrRep*, const char*, int);
StrRep*     Sreverse(StrRep*, StrRep*);
StrRep*     Supcase(StrRep*, StrRep*);
StrRep*     Sdowncase(StrRep*, StrRep*);
StrRep*     Scapitalize(StrRep*, StrRep*);
class String;
class SubString;
class SubString
{
friend class      String;
protected:
String&           S;        
unsigned short    pos;      
unsigned short    len;      
void              assign(StrRep*, const char*, int = -1);
SubString(String& x, int p, int l);
SubString(const SubString& x);
public:
~SubString();
void              operator =  (const String&     y);
void              operator =  (const SubString&  y);
void              operator =  (const char* t);
void              operator =  (char        c);
int               contains(char        c) const;
int               contains(const String&     y) const;
int               contains(const SubString&  y) const;
int               contains(const char* t) const;
int               contains(const Regex&       r) const;
int               matches(const Regex&  r) const;
friend ostream&   operator<<(ostream& s, const SubString& x);
unsigned int      length() const;
int               empty() const;
const char*       chars() const;
int               OK() const;
};
class String
{
friend class      SubString;
protected:
StrRep*           rep;   
int               search(int, int, const char*, int = -1) const;
int               search(int, int, char) const;
int               match(int, int, int, const char*, int = -1) const;
int               _gsub(const char*, int, const char* ,int);
int               _gsub(const Regex&, const char*, int);
SubString         _substr(int, int);
public:
String();
String(const String& x);
String(const SubString&  x);
String(const char* t);
String(const char* t, int len);
String(char c);
~String();
void              operator =  (const String&     y);
void              operator =  (const char* y);
void              operator =  (char        c);
void              operator =  (const SubString&  y);
void              operator += (const String&     y);
void              operator += (const SubString&  y);
void              operator += (const char* t);
void              operator += (char        c);
void              prepend(const String&     y);
void              prepend(const SubString&  y);
void              prepend(const char* t);
void              prepend(char        c);
friend void     cat(const String&, const String&, String&);
friend void     cat(const String&, const SubString&, String&);
friend void     cat(const String&, const char*, String&);
friend void     cat(const String&, char, String&);
friend void     cat(const SubString&, const String&, String&);
friend void     cat(const SubString&, const SubString&, String&);
friend void     cat(const SubString&, const char*, String&);
friend void     cat(const SubString&, char, String&);
friend void     cat(const char*, const String&, String&);
friend void     cat(const char*, const SubString&, String&);
friend void     cat(const char*, const char*, String&);
friend void     cat(const char*, char, String&);
friend void     cat(const String&,const String&, const String&,String&);
friend void     cat(const String&,const String&,const SubString&,String&);
friend void     cat(const String&,const String&, const char*, String&);
friend void     cat(const String&,const String&, char, String&);
friend void     cat(const String&,const SubString&,const String&,String&);
friend void     cat(const String&,const SubString&,const SubString&,String&);
friend void     cat(const String&,const SubString&, const char*, String&);
friend void     cat(const String&,const SubString&, char, String&);
friend void     cat(const String&,const char*, const String&,    String&);
friend void     cat(const String&,const char*, const SubString&, String&);
friend void     cat(const String&,const char*, const char*, String&);
friend void     cat(const String&,const char*, char, String&);
friend void     cat(const char*, const String&, const String&,String&);
friend void     cat(const char*,const String&,const SubString&,String&);
friend void     cat(const char*,const String&, const char*, String&);
friend void     cat(const char*,const String&, char, String&);
friend void     cat(const char*,const SubString&,const String&,String&);
friend void     cat(const char*,const SubString&,const SubString&,String&);
friend void     cat(const char*,const SubString&, const char*, String&);
friend void     cat(const char*,const SubString&, char, String&);
friend void     cat(const char*,const char*, const String&,    String&);
friend void     cat(const char*,const char*, const SubString&, String&);
friend void     cat(const char*,const char*, const char*, String&);
friend void     cat(const char*,const char*, char, String&);
int               index(char        c, int startpos = 0) const;
int               index(const String&     y, int startpos = 0) const;
int               index(const SubString&  y, int startpos = 0) const;
int               index(const char* t, int startpos = 0) const;
int               index(const Regex&      r, int startpos = 0) const;
int               contains(char        c) const;
int               contains(const String&     y) const;
int               contains(const SubString&  y) const;
int               contains(const char* t) const;
int               contains(const Regex&      r) const;
int               contains(char        c, int pos) const;
int               contains(const String&     y, int pos) const;
int               contains(const SubString&  y, int pos) const;
int               contains(const char* t, int pos) const;
int               contains(const Regex&      r, int pos) const;
int               matches(char        c, int pos = 0) const;
int               matches(const String&     y, int pos = 0) const;
int               matches(const SubString&  y, int pos = 0) const;
int               matches(const char* t, int pos = 0) const;
int               matches(const Regex&      r, int pos = 0) const;
int               freq(char        c) const;
int               freq(const String&     y) const;
int               freq(const SubString&  y) const;
int               freq(const char* t) const;
SubString         at(int         pos, int len);
SubString         operator () (int         pos, int len); 
SubString         at(const String&     x, int startpos = 0);
SubString         at(const SubString&  x, int startpos = 0);
SubString         at(const char* t, int startpos = 0);
SubString         at(char        c, int startpos = 0);
SubString         at(const Regex&      r, int startpos = 0);
SubString         before(int          pos);
SubString         before(const String&      x, int startpos = 0);
SubString         before(const SubString&   x, int startpos = 0);
SubString         before(const char*  t, int startpos = 0);
SubString         before(char         c, int startpos = 0);
SubString         before(const Regex&       r, int startpos = 0);
SubString         through(int          pos);
SubString         through(const String&      x, int startpos = 0);
SubString         through(const SubString&   x, int startpos = 0);
SubString         through(const char*  t, int startpos = 0);
SubString         through(char         c, int startpos = 0);
SubString         through(const Regex&       r, int startpos = 0);
SubString         from(int          pos);
SubString         from(const String&      x, int startpos = 0);
SubString         from(const SubString&   x, int startpos = 0);
SubString         from(const char*  t, int startpos = 0);
SubString         from(char         c, int startpos = 0);
SubString         from(const Regex&       r, int startpos = 0);
SubString         after(int         pos);
SubString         after(const String&     x, int startpos = 0);
SubString         after(const SubString&  x, int startpos = 0);
SubString         after(const char* t, int startpos = 0);
SubString         after(char        c, int startpos = 0);
SubString         after(const Regex&      r, int startpos = 0);
void              del(int         pos, int len);
void              del(const String&     y, int startpos = 0);
void              del(const SubString&  y, int startpos = 0);
void              del(const char* t, int startpos = 0);
void              del(char        c, int startpos = 0);
void              del(const Regex&      r, int startpos = 0);
int               gsub(const String&     pat, const String&     repl);
int               gsub(const SubString&  pat, const String&     repl);
int               gsub(const char* pat, const String&     repl);
int               gsub(const char* pat, const char* repl);
int               gsub(const Regex&      pat, const String&     repl);
friend int        split(const String& x, String res[], int maxn,
const String& sep);
friend int        split(const String& x, String res[], int maxn,
const Regex&  sep);
friend String     common_prefix(const String& x, const String& y,
int startpos = 0);
friend String     common_suffix(const String& x, const String& y,
int startpos = -1);
friend String     replicate(char        c, int n);
friend String     replicate(const String&     y, int n);
friend String     join(String src[], int n, const String& sep);
friend String     reverse(const String& x);
friend String     upcase(const String& x);
friend String     downcase(const String& x);
friend String     capitalize(const String& x);
void              reverse();
void              upcase();
void              downcase();
void              capitalize();
char&             operator [] (int i);
char              elem(int i) const;
char              firstchar() const;
char              lastchar() const;
operator const char*() const;
const char*       chars() const;
friend ostream&   operator<<(ostream& s, const String& x);
friend ostream&   operator<<(ostream& s, const SubString& x);
friend istream&   operator>>(istream& s, String& x);
friend int        readline(istream& s, String& x,
char terminator = '\n',
int discard_terminator = 1);
unsigned int      length() const;
int               empty() const;
void              alloc(int newsize);
int               allocation() const;
volatile void     error(const char* msg) const;
int               OK() const;
};
typedef String StrTmp; 
int        compare(const String&    x, const String&     y);
int        compare(const String&    x, const SubString&  y);
int        compare(const String&    x, const char* y);
int        compare(const SubString& x, const String&     y);
int        compare(const SubString& x, const SubString&  y);
int        compare(const SubString& x, const char* y);
int        fcompare(const String&   x, const String&     y); 
extern StrRep  _nilStrRep;
extern String _nilString;
String operator + (const String& x, const String& y);
String operator + (const String& x, const SubString& y);
String operator + (const String& x, const char* y);
String operator + (const String& x, char y);
String operator + (const SubString& x, const String& y);
String operator + (const SubString& x, const SubString& y);
String operator + (const SubString& x, const char* y);
String operator + (const SubString& x, char y);
String operator + (const char* x, const String& y);
String operator + (const char* x, const SubString& y);
int operator==(const String& x, const String& y);
int operator!=(const String& x, const String& y);
int operator> (const String& x, const String& y);
int operator>=(const String& x, const String& y);
int operator< (const String& x, const String& y);
int operator<=(const String& x, const String& y);
int operator==(const String& x, const SubString&  y);
int operator!=(const String& x, const SubString&  y);
int operator> (const String& x, const SubString&  y);
int operator>=(const String& x, const SubString&  y);
int operator< (const String& x, const SubString&  y);
int operator<=(const String& x, const SubString&  y);
int operator==(const String& x, const char* t);
int operator!=(const String& x, const char* t);
int operator> (const String& x, const char* t);
int operator>=(const String& x, const char* t);
int operator< (const String& x, const char* t);
int operator<=(const String& x, const char* t);
int operator==(const SubString& x, const String& y);
int operator!=(const SubString& x, const String& y);
int operator> (const SubString& x, const String& y);
int operator>=(const SubString& x, const String& y);
int operator< (const SubString& x, const String& y);
int operator<=(const SubString& x, const String& y);
int operator==(const SubString& x, const SubString&  y);
int operator!=(const SubString& x, const SubString&  y);
int operator> (const SubString& x, const SubString&  y);
int operator>=(const SubString& x, const SubString&  y);
int operator< (const SubString& x, const SubString&  y);
int operator<=(const SubString& x, const SubString&  y);
int operator==(const SubString& x, const char* t);
int operator!=(const SubString& x, const char* t);
int operator> (const SubString& x, const char* t);
int operator>=(const SubString& x, const char* t);
int operator< (const SubString& x, const char* t);
int operator<=(const SubString& x, const char* t);
inline unsigned int String::length() const {  return rep->len; }
inline int         String::empty() const { return rep->len == 0; }
inline const char* String::chars() const { return &(rep->s[0]); }
inline int         String::allocation() const { return rep->sz; }
inline void        String::alloc(int newsize) { rep = Sresize(rep, newsize); }
inline unsigned int SubString::length() const { return len; }
inline int         SubString::empty() const { return len == 0; }
inline const char* SubString::chars() const { return &(S.rep->s[pos]); }
inline String::String()
: rep(&_nilStrRep) {}
inline String::String(const String& x)
: rep(Scopy(0, x.rep)) {}
inline String::String(const char* t)
: rep(Salloc(0, t, -1, -1)) {}
inline String::String(const char* t, int tlen)
: rep(Salloc(0, t, tlen, tlen)) {}
inline String::String(const SubString& y)
: rep(Salloc(0, y.chars(), y.length(), y.length())) {}
inline String::String(char c)
: rep(Salloc(0, &c, 1, 1)) {}
inline String::~String() { if (rep != &_nilStrRep) delete rep; }
inline SubString::SubString(const SubString& x)
:S(x.S), pos(x.pos), len(x.len) {}
inline SubString::SubString(String& x, int first, int l)
:S(x), pos(first), len(l) {}
inline SubString::~SubString() {}
inline void String::operator =  (const String& y)
{
rep = Scopy(rep, y.rep);
}
inline void String::operator=(const char* t)
{
rep = Salloc(rep, t, -1, -1);
}
inline void String::operator=(const SubString&  y)
{
rep = Salloc(rep, y.chars(), y.length(), y.length());
}
inline void String::operator=(char c)
{
rep = Salloc(rep, &c, 1, 1);
}
inline void SubString::operator = (const char* ys)
{
assign(0, ys);
}
inline void SubString::operator = (char ch)
{
assign(0, &ch, 1);
}
inline void SubString::operator = (const String& y)
{
assign(y.rep, y.chars(), y.length());
}
inline void SubString::operator = (const SubString& y)
{
assign(y.S.rep, y.chars(), y.length());
}
inline void cat(const String& x, const String& y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const String& x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const String& x, const char* y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), y, -1);
}
inline void cat(const String& x, char y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), &y, 1);
}
inline void cat(const SubString& x, const String& y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const SubString& x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const SubString& x, const char* y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), y, -1);
}
inline void cat(const SubString& x, char y, String& r)
{
r.rep = Scat(r.rep, x.chars(), x.length(), &y, 1);
}
inline void cat(const char* x, const String& y, String& r)
{
r.rep = Scat(r.rep, x, -1, y.chars(), y.length());
}
inline void cat(const char* x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, x, -1, y.chars(), y.length());
}
inline void cat(const char* x, const char* y, String& r)
{
r.rep = Scat(r.rep, x, -1, y, -1);
}
inline void cat(const char* x, char y, String& r)
{
r.rep = Scat(r.rep, x, -1, &y, 1);
}
inline void cat(const String& a, const String& x, const String& y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const String& a, const String& x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const String& a, const String& x, const char* y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), y, -1);
}
inline void cat(const String& a, const String& x, char y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), &y, 1);
}
inline void cat(const String& a, const SubString& x, const String& y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const String& a, const SubString& x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const String& a, const SubString& x, const char* y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), y, -1);
}
inline void cat(const String& a, const SubString& x, char y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x.chars(), x.length(), &y, 1);
}
inline void cat(const String& a, const char* x, const String& y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x, -1, y.chars(), y.length());
}
inline void cat(const String& a, const char* x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x, -1, y.chars(), y.length());
}
inline void cat(const String& a, const char* x, const char* y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x, -1, y, -1);
}
inline void cat(const String& a, const char* x, char y, String& r)
{
r.rep = Scat(r.rep, a.chars(), a.length(), x, -1, &y, 1);
}
inline void cat(const char* a, const String& x, const String& y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const char* a, const String& x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const char* a, const String& x, const char* y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), y, -1);
}
inline void cat(const char* a, const String& x, char y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), &y, 1);
}
inline void cat(const char* a, const SubString& x, const String& y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const char* a, const SubString& x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), y.chars(), y.length());
}
inline void cat(const char* a, const SubString& x, const char* y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), y, -1);
}
inline void cat(const char* a, const SubString& x, char y, String& r)
{
r.rep = Scat(r.rep, a, -1, x.chars(), x.length(), &y, 1);
}
inline void cat(const char* a, const char* x, const String& y, String& r)
{
r.rep = Scat(r.rep, a, -1, x, -1, y.chars(), y.length());
}
inline void cat(const char* a, const char* x, const SubString& y, String& r)
{
r.rep = Scat(r.rep, a, -1, x, -1, y.chars(), y.length());
}
inline void cat(const char* a, const char* x, const char* y, String& r)
{
r.rep = Scat(r.rep, a, -1, x, -1, y, -1);
}
inline void cat(const char* a, const char* x, char y, String& r)
{
r.rep = Scat(r.rep, a, -1, x, -1, &y, 1);
}
inline void String::operator +=(const String& y)
{
cat(*this, y, *this);
}
inline void String::operator +=(const SubString& y)
{
cat(*this, y, *this);
}
inline void String::operator += (const char* y)
{
cat(*this, y, *this);
}
inline void String:: operator +=(char y)
{
cat(*this, y, *this);
}
inline String operator + (const String& x, const String& y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const String& x, const SubString& y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const String& x, const char* y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const String& x, char y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const SubString& x, const String& y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const SubString& x, const SubString& y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const SubString& x, const char* y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const SubString& x, char y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const char* x, const String& y) return r; 
{
cat(x, y, r); 
}
inline String operator + (const char* x, const SubString& y) return r; 
{
cat(x, y, r); 
}
inline String reverse(const String& x) return r; 
{
r.rep = Sreverse(x.rep, r.rep); 
}
inline String upcase(const String& x) return r; 
{
r.rep = Supcase(x.rep, r.rep); 
}
inline String downcase(const String& x) return r; 
{
r.rep = Sdowncase(x.rep, r.rep); 
}
inline String capitalize(const String& x) return r; 
{
r.rep = Scapitalize(x.rep, r.rep); 
}
inline void String::prepend(const String& y)
{
rep = Sprepend(rep, y.chars(), y.length());
}
inline void String::prepend(const char* y)
{
rep = Sprepend(rep, y, -1);
}
inline void String::prepend(char y)
{
rep = Sprepend(rep, &y, 1);
}
inline void String::prepend(const SubString& y)
{
rep = Sprepend(rep, y.chars(), y.length());
}
inline void String::reverse()
{
rep = Sreverse(rep, rep);
}
inline void String::upcase()
{
rep = Supcase(rep, rep);
}
inline void String::downcase()
{
rep = Sdowncase(rep, rep);
}
inline void String::capitalize()
{
rep = Scapitalize(rep, rep);
}
inline char&  String::operator [] (int i)
{
if (((unsigned)i) >= length()) error("invalid index");
return rep->s[i];
}
inline char  String::elem (int i) const
{
if (((unsigned)i) >= length()) error("invalid index");
return rep->s[i];
}
inline char  String::firstchar() const
{
return elem(0);
}
inline char  String::lastchar() const
{
return elem(length() - 1);
}
inline int String::index(char c, int startpos) const
{
return search(startpos, length(), c);
}
inline int String::index(const char* t, int startpos) const
{
return search(startpos, length(), t);
}
inline int String::index(const String& y, int startpos) const
{
return search(startpos, length(), y.chars(), y.length());
}
inline int String::index(const SubString& y, int startpos) const
{
return search(startpos, length(), y.chars(), y.length());
}
inline int String::index(const Regex& r, int startpos) const
{
int unused;  return r.search(chars(), length(), unused, startpos);
}
inline int String::contains(char c) const
{
return search(0, length(), c) >= 0;
}
inline int String::contains(const char* t) const
{
return search(0, length(), t) >= 0;
}
inline int String::contains(const String& y) const
{
return search(0, length(), y.chars(), y.length()) >= 0;
}
inline int String::contains(const SubString& y) const
{
return search(0, length(), y.chars(), y.length()) >= 0;
}
inline int String::contains(char c, int p) const
{
return match(p, length(), 0, &c, 1) >= 0;
}
inline int String::contains(const char* t, int p) const
{
return match(p, length(), 0, t) >= 0;
}
inline int String::contains(const String& y, int p) const
{
return match(p, length(), 0, y.chars(), y.length()) >= 0;
}
inline int String::contains(const SubString& y, int p) const
{
return match(p, length(), 0, y.chars(), y.length()) >= 0;
}
inline int String::contains(const Regex& r) const
{
int unused;  return r.search(chars(), length(), unused, 0) >= 0;
}
inline int String::contains(const Regex& r, int p) const
{
return r.match(chars(), length(), p) >= 0;
}
inline int String::matches(const SubString& y, int p) const
{
return match(p, length(), 1, y.chars(), y.length()) >= 0;
}
inline int String::matches(const String& y, int p) const
{
return match(p, length(), 1, y.chars(), y.length()) >= 0;
}
inline int String::matches(const char* t, int p) const
{
return match(p, length(), 1, t) >= 0;
}
inline int String::matches(char c, int p) const
{
return match(p, length(), 1, &c, 1) >= 0;
}
inline int String::matches(const Regex& r, int p) const
{
int l = (p < 0)? -p : length() - p;
return r.match(chars(), length(), p) == l;
}
inline int SubString::contains(const char* t) const
{
return S.search(pos, pos+len, t) >= 0;
}
inline int SubString::contains(const String& y) const
{
return S.search(pos, pos+len, y.chars(), y.length()) >= 0;
}
inline int SubString::contains(const SubString&  y) const
{
return S.search(pos, pos+len, y.chars(), y.length()) >= 0;
}
inline int SubString::contains(char c) const
{
return S.search(pos, pos+len, 0, c) >= 0;
}
inline int SubString::contains(const Regex& r) const
{
int unused;  return r.search(chars(), len, unused, 0) >= 0;
}
inline int SubString::matches(const Regex& r) const
{
return r.match(chars(), len, 0) == len;
}
inline int String::gsub(const String& pat, const String& r)
{
return _gsub(pat.chars(), pat.length(), r.chars(), r.length());
}
inline int String::gsub(const SubString&  pat, const String& r)
{
return _gsub(pat.chars(), pat.length(), r.chars(), r.length());
}
inline int String::gsub(const Regex& pat, const String& r)
{
return _gsub(pat, r.chars(), r.length());
}
inline int String::gsub(const char* pat, const String& r)
{
return _gsub(pat, -1, r.chars(), r.length());
}
inline int String::gsub(const char* pat, const char* r)
{
return _gsub(pat, -1, r, -1);
}
inline  ostream& operator<<(ostream& s, const String& x)
{
s << x.chars(); return s;
}
inline int operator==(const String& x, const String& y)
{
return compare(x, y) == 0;
}
inline int operator!=(const String& x, const String& y)
{
return compare(x, y) != 0;
}
inline int operator>(const String& x, const String& y)
{
return compare(x, y) > 0;
}
inline int operator>=(const String& x, const String& y)
{
return compare(x, y) >= 0;
}
inline int operator<(const String& x, const String& y)
{
return compare(x, y) < 0;
}
inline int operator<=(const String& x, const String& y)
{
return compare(x, y) <= 0;
}
inline int operator==(const String& x, const SubString&  y)
{
return compare(x, y) == 0;
}
inline int operator!=(const String& x, const SubString&  y)
{
return compare(x, y) != 0;
}
inline int operator>(const String& x, const SubString&  y)
{
return compare(x, y) > 0;
}
inline int operator>=(const String& x, const SubString&  y)
{
return compare(x, y) >= 0;
}
inline int operator<(const String& x, const SubString&  y)
{
return compare(x, y) < 0;
}
inline int operator<=(const String& x, const SubString&  y)
{
return compare(x, y) <= 0;
}
inline int operator==(const String& x, const char* t)
{
return compare(x, t) == 0;
}
inline int operator!=(const String& x, const char* t)
{
return compare(x, t) != 0;
}
inline int operator>(const String& x, const char* t)
{
return compare(x, t) > 0;
}
inline int operator>=(const String& x, const char* t)
{
return compare(x, t) >= 0;
}
inline int operator<(const String& x, const char* t)
{
return compare(x, t) < 0;
}
inline int operator<=(const String& x, const char* t)
{
return compare(x, t) <= 0;
}
inline int operator==(const SubString& x, const String& y)
{
return compare(y, x) == 0;
}
inline int operator!=(const SubString& x, const String& y)
{
return compare(y, x) != 0;
}
inline int operator>(const SubString& x, const String& y)
{
return compare(y, x) < 0;
}
inline int operator>=(const SubString& x, const String& y)
{
return compare(y, x) <= 0;
}
inline int operator<(const SubString& x, const String& y)
{
return compare(y, x) > 0;
}
inline int operator<=(const SubString& x, const String& y)
{
return compare(y, x) >= 0;
}
inline int operator==(const SubString& x, const SubString&  y)
{
return compare(x, y) == 0;
}
inline int operator!=(const SubString& x, const SubString&  y)
{
return compare(x, y) != 0;
}
inline int operator>(const SubString& x, const SubString&  y)
{
return compare(x, y) > 0;
}
inline int operator>=(const SubString& x, const SubString&  y)
{
return compare(x, y) >= 0;
}
inline int operator<(const SubString& x, const SubString&  y)
{
return compare(x, y) < 0;
}
inline int operator<=(const SubString& x, const SubString&  y)
{
return compare(x, y) <= 0;
}
inline int operator==(const SubString& x, const char* t)
{
return compare(x, t) == 0;
}
inline int operator!=(const SubString& x, const char* t)
{
return compare(x, t) != 0;
}
inline int operator>(const SubString& x, const char* t)
{
return compare(x, t) > 0;
}
inline int operator>=(const SubString& x, const char* t)
{
return compare(x, t) >= 0;
}
inline int operator<(const SubString& x, const char* t)
{
return compare(x, t) < 0;
}
inline int operator<=(const SubString& x, const char* t)
{
return compare(x, t) <= 0;
}
inline SubString String::_substr(int first, int l)
{
if (first >= length() )  
return SubString(_nilString, 0, 0) ;
else
return SubString(*this, first, l);
}
class strstreambuf : public streambuf {
size_t *lenp; 
size_t *sizep; 
char **bufp;
size_t _len;
size_t _size;
char *buf;
int _frozen;
protected:
virtual int overflow(int = (-1) );
public:
strstreambuf();
strstreambuf(int initial);
strstreambuf(char *ptr, int size, char *pstart = (__null) );
~strstreambuf();
int frozen() { return _frozen; }
void freeze(int n=1) { _frozen = n != 0; }
size_t pcount();
char *str();
};
class istrstream : public istream {
public:
istrstream(char*);
istrstream(char*, int);
strstreambuf* rdbuf() { return (strstreambuf*)_strbuf; }
};
class ostrstream : public ostream {
public:
ostrstream();
ostrstream(char *cp, int n, int mode=ios::out);
size_t pcount() { return ((strstreambuf*)_strbuf)->pcount(); }
char *str() { return ((strstreambuf*)_strbuf)->str(); }
void freeze(int n = 1) { ((strstreambuf*)_strbuf)->freeze(n); }
int frozen() { return ((strstreambuf*)_strbuf)->frozen(); }
strstreambuf* rdbuf() { return (strstreambuf*)_strbuf; }
};
class tostrstream: public ostrstream {
public:
tostrstream(): ostrstream()
{ }
tostrstream(char *cp, int n, int mode=ios::out): ostrtream(cp, n, mode)	
{ }
char *str()
{
char *s = ostrstream::str();
s[ostrstream::pcount()] = '\0';
return s;
}
};
extern "C" {
typedef long FitAny;		
extern int nocase_strcmp (char *, char *)		;
extern int nocase_strncmp (char *, char *, int)		;
extern bool	 nocase_strequal (char *, char *)		;
extern bool	 nocase_strnequal (char *, char *, int)		;
extern bool	 lead_strequal (char *, char *)		;
extern bool	 nocase_lead_strequal (char *, char *)		;
extern int strhash (char *, int)		;
extern int nocase_strhash (char *, int)		;
extern int sign (int)		;
}
extern const char *stringify(bool b);
extern ostream& operator<<(ostream&, bool);
enum unit {
UNIT = 1,
};
extern const char *stringify(unit u);
extern ostream& operator<<(ostream&, unit);
typedef void (*zero_arg_error_handler_t)();
extern void default_zero_arg_error_handler();
extern void exit_zero_arg_error_handler();
extern void exit_one_arg_error_handler(const char *message);
extern void exit_two_arg_error_handler(const char *kind, const char *message);
extern void abort_zero_arg_error_handler();
extern void abort_one_arg_error_handler(const char *message);
extern void abort_two_arg_error_handler(const char *kind, const char *message);
extern void preserve_File_error_handler(const char *message);
class GttErrorHandler {
public:
GttErrorHandler();
GttErrorHandler(const char *program);
virtual ~GttErrorHandler();
static void error(const char *message);
static void error(tostrstream& message);
static void error(const char *function, const char *message);
static void error(const char *function, tostrstream& message);
static void error(const char *class_name, const char *method, const char *message);
static void error(const char *class_name, const char *method, tostrstream& message);
static void fatal(const char *message);
static void fatal(tostrstream& message);
static void fatal(const char *function, const char *message);
static void fatal(const char *function, tostrstream& message);
static void fatal(const char *class_name, const char *method, const char *message);
static void fatal(const char *class_name, const char *method, tostrstream& message);
private:
static bool __partial_init;
static void __partial_initialize();
static bool __full_init;
static void __full_initialize(const char *program);
static char *__program;
static void __handle_error();
static void __handle_fatal();
static void __add_newline(const char *message);
static bool __output_valid();
static ostream *__output;
};
class GttObject: virtual public GttErrorHandler {
protected:
GttObject();
GttObject(const GttObject&);
virtual ~GttObject();	
public:
virtual const char *stringify();
protected:
tostrstream *stringbuf;
void clear_stringbuf();
public:
virtual void OK() const;
protected:
void ok() const;
protected:
virtual const char *class_name() const = 0;
};
extern ostream& operator<<(ostream&, GttObject&);
class GctErrorHandler: virtual public GttObject {
public:
GctErrorHandler();
GctErrorHandler(const String& program);
virtual ~GctErrorHandler();
static void debug(const char *message);
static void debug(tostrstream& message);
static void debug(const char *function, const char *message);
static void debug(const char *function, tostrstream& message);
static void debug(const char *class_name, const char *method, const char *message);
static void debug(const char *class_name, const char *method, tostrstream& message);
static bool debug();		
static void debug(bool value);	
static void note(const char *message);
static void note(tostrstream& message);
static void note(const char *function, const char *message);
static void note(const char *function, tostrstream& message);
static void note(const char *class_name, const char *method, const char *message);
static void note(const char *class_name, const char *method, tostrstream& message);
static bool note();			
static void note(bool value);	
static void warning(const char *message);
static void warning(tostrstream& message);
static void warning(const char *function, const char *message);
static void warning(const char *function, tostrstream& message);
static void warning(const char *class_name, const char *method, const char *message);
static void warning(const char *class_name, const char *method, tostrstream& message);
static bool warning();		
static void warning(bool value);	
static void error(const char *message);
static void error(tostrstream& message);
static void error(const char *function, const char *message);
static void error(const char *function, tostrstream& message);
static void error(const char *class_name, const char *method, const char *message);
static void error(const char *class_name, const char *method, tostrstream& message);
static zero_arg_error_handler_t error();		
static void error(zero_arg_error_handler_t handler);
static void error_is_lib_error_handler();		
static void error_is_exit();			
static const char *error_handler_description();
static void fatal(const char *message);
static void fatal(tostrstream& message);
static void fatal(const char *function, const char *message);
static void fatal(const char *function, tostrstream& message);
static void fatal(const char *class_name, const char *method, const char *message);
static void fatal(const char *class_name, const char *method, tostrstream& message);
static zero_arg_error_handler_t fatal();			
static void fatal(zero_arg_error_handler_t handler);	
static void fatal_is_exit();	
static void fatal_is_abort();	
static const char *fatal_handler_description();
private:
static bool __debug;
static bool __note;
static bool __warning;
static void (*__error_handler)();	
static void (*__fatal_handler)();	
static bool __partial_init;
static void __partial_initialize();
static bool __full_init;
static void __full_initialize(const char *program);
static char *__program;
static void __handle_error();
static void __handle_fatal();
static void __add_newline(const char *message);
static void __message_switch(bool value, bool& flag, const char *description);
static void __message_switch(bool value, bool& flag);
static const char *__describe_handler(zero_arg_error_handler_t handler);
static bool __output_valid();
static ostream *__output;
const char *class_name() const;
};
class GctObject: virtual public GctErrorHandler  {
protected:
GctObject();
GctObject(const GctObject&);
virtual ~GctObject();	
public:
virtual const char *stringify();
protected:
tostrstream *stringbuf;
void clear_stringbuf();
public:
virtual void OK() const;
protected:
void ok() const;
protected:
virtual const char *class_name() const = 0;
public:
unsigned objectId() const;
private:
unsigned __object_id;
static unsigned __next_id;
};
extern ostream& operator<<(ostream&, GctObject&);
class GctHashObject: virtual public GctObject {
protected:
GctHashObject();
GctHashObject(const GctHashObject&);
public:
virtual unsigned hash() const;
};
class GctSymbol: virtual public GctHashObject, String {
public:
GctSymbol();		
GctSymbol(const char*);
GctSymbol(const String&);
GctSymbol(const GctSymbol&);
operator const char *() const;
bool operator==(const GctSymbol&) const;
bool operator!=(const GctSymbol&) const;
bool operator<=(const GctSymbol&) const;
bool operator<(const GctSymbol&) const;
bool operator>=(const GctSymbol&) const;
bool operator>(const GctSymbol&) const;
unsigned hash() const;
const char *stringify();
void OK() const;
private:
const char *class_name() const;
};
extern unsigned hash(GctSymbol&);	
GctSymbol::operator const char *() const
{
return String::operator const char *();
}
bool
GctSymbol::operator==(const GctSymbol& other) const
{
return (bool)::operator==(*this, other);
}
bool
GctSymbol::operator!=(const GctSymbol& other) const
{
return (bool)::operator!=(*this, other);
}
bool
GctSymbol::operator<=(const GctSymbol& other) const
{
return (bool)::operator<=(*this, other);
}
bool
GctSymbol::operator<(const GctSymbol& other) const
{
return (bool)::operator<(*this, other);
}
bool
GctSymbol::operator>=(const GctSymbol& other) const
{
return (bool)::operator>=(*this, other);
}
bool
GctSymbol::operator>(const GctSymbol& other) const
{
return (bool)::operator>(*this, other);
}
extern unsigned int hash(GctSymbol&);
class GctSymbolGctSymbolMap
{
protected:
int                   count;
GctSymbol                   def;
public:
GctSymbolGctSymbolMap(GctSymbol& dflt);
virtual              ~GctSymbolGctSymbolMap();
int                   length();                
int                   empty();
virtual int           contains(GctSymbol& key);      
virtual void          clear();                 
virtual GctSymbol&          operator [] (GctSymbol& key) = 0; 
virtual void          del(GctSymbol& key) = 0;       
virtual Pix           first() = 0;             
virtual void          next(Pix& i) = 0;        
virtual GctSymbol&          key(Pix i) = 0;          
virtual GctSymbol&          contents(Pix i) = 0;     
virtual int           owns(Pix i);             
virtual Pix           seek(GctSymbol& key);          
GctSymbol&                  dflt();                  
void                  error(const char* msg);
virtual int           OK() = 0;                
};
inline GctSymbolGctSymbolMap::~GctSymbolGctSymbolMap() {}
inline int GctSymbolGctSymbolMap::length()
{
return count;
}
inline int GctSymbolGctSymbolMap::empty()
{
return count == 0;
}
inline GctSymbol& GctSymbolGctSymbolMap::dflt()
{
return def;
}
inline GctSymbolGctSymbolMap::GctSymbolGctSymbolMap(GctSymbol& dflt) :def(dflt)
{
count = 0;
}
struct GctSymbolGctSymbolCHNode
{
GctSymbolGctSymbolCHNode*      tl;
GctSymbol                hd;
GctSymbol                cont;
GctSymbolGctSymbolCHNode();
GctSymbolGctSymbolCHNode(GctSymbol& h, GctSymbol& c, GctSymbolGctSymbolCHNode* t = 0);
~GctSymbolGctSymbolCHNode();
};
inline GctSymbolGctSymbolCHNode::GctSymbolGctSymbolCHNode() {}
inline GctSymbolGctSymbolCHNode::GctSymbolGctSymbolCHNode(GctSymbol& h, GctSymbol& c, GctSymbolGctSymbolCHNode* t)
: hd(h), cont(c), tl(t) {}
inline GctSymbolGctSymbolCHNode::~GctSymbolGctSymbolCHNode() {}
typedef GctSymbolGctSymbolCHNode* GctSymbolGctSymbolCHNodePtr;
class GctSymbolGctSymbolCHMap : public GctSymbolGctSymbolMap
{
protected:
GctSymbolGctSymbolCHNode** tab;
unsigned int   size;
public:
GctSymbolGctSymbolCHMap(GctSymbol& dflt,unsigned int sz=100 );
GctSymbolGctSymbolCHMap(GctSymbolGctSymbolCHMap& a);
~GctSymbolGctSymbolCHMap();
GctSymbol&          operator [] (GctSymbol& key);
void          del(GctSymbol& key);
Pix           first();
void          next(Pix& i);
GctSymbol&          key(Pix i);
GctSymbol&          contents(Pix i);
Pix           seek(GctSymbol& key);
int           contains(GctSymbol& key);
void          clear();
int           OK();
};
inline GctSymbolGctSymbolCHMap::~GctSymbolGctSymbolCHMap()
{
clear();
delete tab;
}
inline int GctSymbolGctSymbolCHMap::contains(GctSymbol& key)
{
return seek(key) != 0;
}
inline GctSymbol& GctSymbolGctSymbolCHMap::key(Pix p)
{
if (p == 0) error("null Pix");
return ((GctSymbolGctSymbolCHNode*)p)->hd;
}
inline GctSymbol& GctSymbolGctSymbolCHMap::contents(Pix p)
{
if (p == 0) error("null Pix");
return ((GctSymbolGctSymbolCHNode*)p)->cont;
}
static inline int goodCHptr(GctSymbolGctSymbolCHNode* t)
{
return ((((unsigned)t) & 1) == 0);
}
static inline GctSymbolGctSymbolCHNode* index_to_CHptr(int i)
{
return (GctSymbolGctSymbolCHNode*)((i << 1) + 1);
}
static inline int CHptr_to_index(GctSymbolGctSymbolCHNode* t)
{
return ( ((unsigned) t) >> 1);
}
GctSymbolGctSymbolCHMap::GctSymbolGctSymbolCHMap(GctSymbol& dflt, unsigned int sz)
:GctSymbolGctSymbolMap(dflt)
{
tab = (GctSymbolGctSymbolCHNode**)(new GctSymbolGctSymbolCHNodePtr[size = sz]);
for (unsigned int i = 0; i < size; ++i) tab[i] = index_to_CHptr(i+1);
count = 0;
}
GctSymbolGctSymbolCHMap::GctSymbolGctSymbolCHMap(GctSymbolGctSymbolCHMap& a) :GctSymbolGctSymbolMap(a.def)
{
tab = (GctSymbolGctSymbolCHNode**)(new GctSymbolGctSymbolCHNodePtr[size = a.size]);
for (unsigned int i = 0; i < size; ++i) tab[i] = index_to_CHptr(i+1);
count = 0;
for (Pix p = a.first(); p; a.next(p)) (*this)[a.key(p)] = a.contents(p); 
}
