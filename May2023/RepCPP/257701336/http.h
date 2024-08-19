#ifndef HEADER_CURL_HTTP_H
#define HEADER_CURL_HTTP_H

#include "curl_setup.h"

#ifndef CURL_DISABLE_HTTP

#ifdef USE_NGHTTP2
#include <nghttp2/nghttp2.h>
#endif

extern const struct Curl_handler Curl_handler_http;

#ifdef USE_SSL
extern const struct Curl_handler Curl_handler_https;
#endif


bool Curl_compareheader(const char *headerline,  
const char *header,   
const char *content); 

char *Curl_copy_header_value(const char *header);

char *Curl_checkProxyheaders(const struct connectdata *conn,
const char *thisheader);


struct Curl_send_buffer {
char *buffer;
size_t size_max;
size_t size_used;
};
typedef struct Curl_send_buffer Curl_send_buffer;

Curl_send_buffer *Curl_add_buffer_init(void);
void Curl_add_buffer_free(Curl_send_buffer *buff);
CURLcode Curl_add_bufferf(Curl_send_buffer *in, const char *fmt, ...);
CURLcode Curl_add_buffer(Curl_send_buffer *in, const void *inptr, size_t size);
CURLcode Curl_add_buffer_send(Curl_send_buffer *in,
struct connectdata *conn,
long *bytes_written,
size_t included_body_bytes,
int socketindex);

CURLcode Curl_add_timecondition(struct Curl_easy *data,
Curl_send_buffer *buf);
CURLcode Curl_add_custom_headers(struct connectdata *conn,
bool is_connect,
Curl_send_buffer *req_buffer);


CURLcode Curl_http(struct connectdata *conn, bool *done);
CURLcode Curl_http_done(struct connectdata *, CURLcode, bool premature);
CURLcode Curl_http_connect(struct connectdata *conn, bool *done);
CURLcode Curl_http_setup_conn(struct connectdata *conn);


void Curl_httpchunk_init(struct connectdata *conn);
CHUNKcode Curl_httpchunk_read(struct connectdata *conn, char *datap,
ssize_t length, ssize_t *wrote);


void Curl_http_auth_stage(struct Curl_easy *data, int stage);
CURLcode Curl_http_input_auth(struct connectdata *conn, bool proxy,
const char *auth);
CURLcode Curl_http_auth_act(struct connectdata *conn);
CURLcode Curl_http_perhapsrewind(struct connectdata *conn);


#define CURLAUTH_PICKNONE (1<<30) 


#ifndef MAX_INITIAL_POST_SIZE
#define MAX_INITIAL_POST_SIZE (64*1024)
#endif


#ifndef EXPECT_100_THRESHOLD
#define EXPECT_100_THRESHOLD 1024
#endif

#endif 


struct HTTP {
curl_mimepart *sendit;
curl_off_t postsize; 
const char *postdata;

const char *p_pragma;      
const char *p_accept;      
curl_off_t readbytecount;
curl_off_t writebytecount;


curl_mimepart form;

struct back {
curl_read_callback fread_func; 
void *fread_in;           
const char *postdata;
curl_off_t postsize;
} backup;

enum {
HTTPSEND_NADA,    
HTTPSEND_REQUEST, 
HTTPSEND_BODY,    
HTTPSEND_LAST     
} sending;

void *send_buffer; 

#ifdef USE_NGHTTP2

int32_t stream_id; 

bool bodystarted;

Curl_send_buffer *header_recvbuf;
size_t nread_header_recvbuf; 
Curl_send_buffer *trailer_recvbuf;
int status_code; 
const uint8_t *pausedata; 
size_t pauselen; 
bool closed; 
bool close_handled; 
char *mem;     
size_t len;    
size_t memlen; 

const uint8_t *upload_mem; 
size_t upload_len; 
curl_off_t upload_left; 

char **push_headers;       
size_t push_headers_used;  
size_t push_headers_alloc; 
#endif
};

#ifdef USE_NGHTTP2

struct h2settings {
uint32_t max_concurrent_streams;
bool enable_push;
};
#endif

struct http_conn {
#ifdef USE_NGHTTP2
#define H2_BINSETTINGS_LEN 80
nghttp2_session *h2;
uint8_t binsettings[H2_BINSETTINGS_LEN];
size_t  binlen; 
Curl_send *send_underlying; 
Curl_recv *recv_underlying; 
char *inbuf; 
size_t inbuflen; 
size_t nread_inbuf; 

int32_t pause_stream_id; 
size_t drain_total; 


struct h2settings settings;


nghttp2_settings_entry local_settings[3];
size_t local_settings_num;
uint32_t error_code; 
#else
int unused; 
#endif
};

CURLcode Curl_http_readwrite_headers(struct Curl_easy *data,
struct connectdata *conn,
ssize_t *nread,
bool *stop_reading);


CURLcode
Curl_http_output_auth(struct connectdata *conn,
const char *request,
const char *path,
bool proxytunnel); 

#endif 

