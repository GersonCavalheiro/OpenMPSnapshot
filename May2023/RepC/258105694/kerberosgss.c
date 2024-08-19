#include "kerberosgss.h"
#include "base64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdarg.h>
#include <unistd.h>
#include <krb5.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
void die1(const char *message) {
if(errno) {
perror(message);
} else {
printf("ERROR: %s\n", message);
}
exit(1);
}
static gss_client_response *gss_error(const char *func, const char *op, OM_uint32 err_maj, OM_uint32 err_min);
static gss_client_response *other_error(const char *fmt, ...);
static gss_client_response *krb5_ctx_error(krb5_context context, krb5_error_code problem);
static gss_client_response *store_gss_creds(gss_server_state *state);
static gss_client_response *create_krb5_ccache(gss_server_state *state, krb5_context context, krb5_principal princ, krb5_ccache *ccache);
static gss_client_response *verify_krb5_kdc(krb5_context context,
krb5_creds *creds,
const char *service);
static gss_client_response *init_gss_creds(const char *credential_cache, gss_cred_id_t *cred);
OM_uint32 KRB5_CALLCONV
gss_acquire_cred_impersonate_name(
OM_uint32 *,	    
const gss_cred_id_t,    
const gss_name_t,	    
OM_uint32,		    
const gss_OID_set,	    
gss_cred_usage_t,	    
gss_cred_id_t *,	    
gss_OID_set *,	    
OM_uint32 *);	    
gss_client_response *authenticate_gss_client_init(const char* service, long int gss_flags, const char* credentials_cache, gss_client_state* state) {
OM_uint32 maj_stat;
OM_uint32 min_stat;
gss_buffer_desc name_token = GSS_C_EMPTY_BUFFER;
gss_client_response *response = NULL;
int ret = AUTH_GSS_COMPLETE;
state->server_name = GSS_C_NO_NAME;
state->context = GSS_C_NO_CONTEXT;
state->gss_flags = gss_flags;
state->username = NULL;
state->response = NULL;
state->credentials_cache = NULL;
name_token.length = strlen(service);
name_token.value = (char *)service;
maj_stat = gss_import_name(&min_stat, &name_token, gss_krb5_nt_service_name, &state->server_name);
if (GSS_ERROR(maj_stat)) {
response = gss_error(__func__, "gss_import_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
if (credentials_cache && strlen(credentials_cache) > 0) {
state->credentials_cache = strdup(credentials_cache);
if (state->credentials_cache == NULL) die1("Memory allocation failed");
}
end:
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_gss_client_clean(gss_client_state *state) {
OM_uint32 min_stat;
int ret = AUTH_GSS_COMPLETE;
gss_client_response *response = NULL;
if(state->context != GSS_C_NO_CONTEXT)
gss_delete_sec_context(&min_stat, &state->context, GSS_C_NO_BUFFER);
if(state->server_name != GSS_C_NO_NAME)
gss_release_name(&min_stat, &state->server_name);
if(state->username != NULL) {
free(state->username);
state->username = NULL;
}
if (state->response != NULL) {
free(state->response);
state->response = NULL;
}
if (state->credentials_cache != NULL) {
free(state->credentials_cache);
state->credentials_cache = NULL;
}
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_gss_client_step(gss_client_state* state, const char* challenge) {
OM_uint32 maj_stat;
OM_uint32 min_stat;
gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;
int ret = AUTH_GSS_CONTINUE;
gss_client_response *response = NULL;
gss_cred_id_t gss_cred = GSS_C_NO_CREDENTIAL;
if (state->response != NULL) {
free(state->response);
state->response = NULL;
}
if (challenge && *challenge) {
int len;
input_token.value = base64_decode(challenge, &len);
input_token.length = len;
}
if (state->credentials_cache) {
response = init_gss_creds(state->credentials_cache, &gss_cred);
if (response) {
goto end;
}
}
maj_stat = gss_init_sec_context(&min_stat,
gss_cred,
&state->context,
state->server_name,
GSS_C_NO_OID,
(OM_uint32)state->gss_flags,
0,
GSS_C_NO_CHANNEL_BINDINGS,
&input_token,
NULL,
&output_token,
NULL,
NULL);
if ((maj_stat != GSS_S_COMPLETE) && (maj_stat != GSS_S_CONTINUE_NEEDED)) {
response = gss_error(__func__, "gss_init_sec_context", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
ret = (maj_stat == GSS_S_COMPLETE) ? AUTH_GSS_COMPLETE : AUTH_GSS_CONTINUE;
if(output_token.length) {
state->response = base64_encode((const unsigned char *)output_token.value, output_token.length);
maj_stat = gss_release_buffer(&min_stat, &output_token);
}
if (ret == AUTH_GSS_COMPLETE) {
gss_name_t gssuser = GSS_C_NO_NAME;
maj_stat = gss_inquire_context(&min_stat, state->context, &gssuser, NULL, NULL, NULL,  NULL, NULL, NULL);
if(GSS_ERROR(maj_stat)) {
response = gss_error(__func__, "gss_inquire_context", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
gss_buffer_desc name_token;
name_token.length = 0;
maj_stat = gss_display_name(&min_stat, gssuser, &name_token, NULL);
if(GSS_ERROR(maj_stat)) {
if(name_token.value)
gss_release_buffer(&min_stat, &name_token);
gss_release_name(&min_stat, &gssuser);
response = gss_error(__func__, "gss_display_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
} else {
state->username = (char *)malloc(name_token.length + 1);
if(state->username == NULL) die1("Memory allocation failed");
strncpy(state->username, (char*) name_token.value, name_token.length);
state->username[name_token.length] = 0;
gss_release_buffer(&min_stat, &name_token);
gss_release_name(&min_stat, &gssuser);
}
}
end:
if (gss_cred != GSS_C_NO_CREDENTIAL)
gss_release_cred(&min_stat, &gss_cred);
if(output_token.value)
gss_release_buffer(&min_stat, &output_token);
if(input_token.value)
free(input_token.value);
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
static gss_client_response *init_gss_creds(const char *credential_cache, gss_cred_id_t *cred) {
OM_uint32 maj_stat;
OM_uint32 min_stat;
krb5_context context;
krb5_error_code problem;
gss_client_response *response = NULL;
krb5_ccache ccache = NULL;
*cred = GSS_C_NO_CREDENTIAL;
if (credential_cache == NULL || strlen(credential_cache) == 0) {
return NULL;
}
problem = krb5_init_context(&context);
if (problem) {
return other_error("unable to initialize krb5 context (%d)", (int)problem);
}
problem = krb5_cc_resolve(context, credential_cache, &ccache);
if (problem) {
response = krb5_ctx_error(context, problem);
goto done;
}
maj_stat = gss_krb5_import_cred(&min_stat, ccache, NULL, NULL, cred);
if (GSS_ERROR(maj_stat)) {
response = gss_error(__func__, "gss_krb5_import_cred", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
}
done:
if (response && ccache) {
krb5_cc_close(context, ccache);
}
krb5_free_context(context);
return response;
}
gss_client_response *authenticate_gss_client_unwrap(gss_client_state *state, const char *challenge) {
OM_uint32 maj_stat;
OM_uint32 min_stat;
gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;
gss_client_response *response = NULL;
int ret = AUTH_GSS_CONTINUE;
if(state->response != NULL) {
free(state->response);
state->response = NULL;
}
if(challenge && *challenge) {
int len;
input_token.value = base64_decode(challenge, &len);
input_token.length = len;
}
maj_stat = gss_unwrap(&min_stat,
state->context,
&input_token,
&output_token,
NULL,
NULL);
if(maj_stat != GSS_S_COMPLETE) {
response = gss_error(__func__, "gss_unwrap", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
} else {
ret = AUTH_GSS_COMPLETE;
}
if(output_token.length) {
state->response = base64_encode((const unsigned char *)output_token.value, output_token.length);
gss_release_buffer(&min_stat, &output_token);
}
end:
if(output_token.value)
gss_release_buffer(&min_stat, &output_token);
if(input_token.value)
free(input_token.value);
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_gss_client_wrap(gss_client_state* state, const char* challenge, const char* user) {
OM_uint32 maj_stat;
OM_uint32 min_stat;
gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;
int ret = AUTH_GSS_CONTINUE;
gss_client_response *response = NULL;
char buf[4096], server_conf_flags;
unsigned long buf_size;
if(state->response != NULL) {
free(state->response);
state->response = NULL;
}
if(challenge && *challenge) {
int len;
input_token.value = base64_decode(challenge, &len);
input_token.length = len;
}
if(user) {
server_conf_flags = ((char*) input_token.value)[0];
((char*) input_token.value)[0] = 0;
buf_size = ntohl(*((long *) input_token.value));
free(input_token.value);
#ifdef PRINTFS
printf("User: %s, %c%c%c\n", user,
server_conf_flags & GSS_AUTH_P_NONE      ? 'N' : '-',
server_conf_flags & GSS_AUTH_P_INTEGRITY ? 'I' : '-',
server_conf_flags & GSS_AUTH_P_PRIVACY   ? 'P' : '-');
printf("Maximum GSS token size is %ld\n", buf_size);
#endif
buf_size = htonl(buf_size); 
memcpy(buf, &buf_size, 4);
buf[0] = GSS_AUTH_P_NONE;
strncpy(buf + 4, user, sizeof(buf) - 4);
input_token.value = buf;
input_token.length = 4 + strlen(user);
}
maj_stat = gss_wrap(&min_stat,
state->context,
0,
GSS_C_QOP_DEFAULT,
&input_token,
NULL,
&output_token);
if (maj_stat != GSS_S_COMPLETE) {
response = gss_error(__func__, "gss_wrap", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
} else
ret = AUTH_GSS_COMPLETE;
if (output_token.length) {
state->response = base64_encode((const unsigned char *)output_token.value, output_token.length);;
gss_release_buffer(&min_stat, &output_token);
}
end:
if (output_token.value)
gss_release_buffer(&min_stat, &output_token);
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_gss_server_init(const char *service, bool constrained_delegation, const char *username, gss_server_state *state)
{
OM_uint32 maj_stat;
OM_uint32 min_stat;
gss_buffer_desc name_token = GSS_C_EMPTY_BUFFER;
int ret = AUTH_GSS_COMPLETE;
gss_client_response *response = NULL;
gss_cred_usage_t usage = GSS_C_ACCEPT;
state->context = GSS_C_NO_CONTEXT;
state->server_name = GSS_C_NO_NAME;
state->client_name = GSS_C_NO_NAME;
state->server_creds = GSS_C_NO_CREDENTIAL;
state->client_creds = GSS_C_NO_CREDENTIAL;
state->username = NULL;
state->targetname = NULL;
state->response = NULL;
state->constrained_delegation = constrained_delegation;
state->delegated_credentials_cache = NULL;
size_t service_len = strlen(service);
if (service_len != 0)
{
name_token.length = strlen(service);
name_token.value = (char *)service;
maj_stat = gss_import_name(&min_stat, &name_token, GSS_C_NT_HOSTBASED_SERVICE, &state->server_name);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_import_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
if (state->constrained_delegation)
{
usage = GSS_C_BOTH;
}
maj_stat = gss_acquire_cred(&min_stat, state->server_name, GSS_C_INDEFINITE,
GSS_C_NO_OID_SET, usage, &state->server_creds, NULL, NULL);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_acquire_cred", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
}
if (username != NULL)
{
gss_name_t gss_username;
name_token.length = strlen(username);
name_token.value = (char *)username;
maj_stat = gss_import_name(&min_stat, &name_token, GSS_C_NT_USER_NAME, &gss_username);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_import_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
maj_stat = gss_acquire_cred_impersonate_name(&min_stat,
state->server_creds,
gss_username,
GSS_C_INDEFINITE,
GSS_C_NO_OID_SET,
GSS_C_INITIATE,
&state->client_creds,
NULL,
NULL);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_acquire_cred_impersonate_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
}
gss_release_name(&min_stat, &gss_username);
if (response != NULL)
{
goto end;
}
maj_stat = gss_inquire_cred(&min_stat, state->client_creds, &state->client_name, NULL, NULL, NULL);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_inquire_cred", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
}
end:
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_gss_server_clean(gss_server_state *state)
{
OM_uint32 min_stat;
int ret = AUTH_GSS_COMPLETE;
gss_client_response *response = NULL;
if (state->context != GSS_C_NO_CONTEXT)
gss_delete_sec_context(&min_stat, &state->context, GSS_C_NO_BUFFER);
if (state->server_name != GSS_C_NO_NAME)
gss_release_name(&min_stat, &state->server_name);
if (state->client_name != GSS_C_NO_NAME)
gss_release_name(&min_stat, &state->client_name);
if (state->server_creds != GSS_C_NO_CREDENTIAL)
gss_release_cred(&min_stat, &state->server_creds);
if (state->client_creds != GSS_C_NO_CREDENTIAL)
gss_release_cred(&min_stat, &state->client_creds);
if (state->username != NULL)
{
free(state->username);
state->username = NULL;
}
if (state->targetname != NULL)
{
free(state->targetname);
state->targetname = NULL;
}
if (state->response != NULL)
{
free(state->response);
state->response = NULL;
}
if (state->delegated_credentials_cache)
{
free(state->delegated_credentials_cache);
}
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_gss_server_step(gss_server_state *state, const char *auth_data)
{
OM_uint32 maj_stat;
OM_uint32 min_stat;
gss_buffer_desc input_token = GSS_C_EMPTY_BUFFER;
gss_buffer_desc output_token = GSS_C_EMPTY_BUFFER;
int ret = AUTH_GSS_CONTINUE;
gss_client_response *response = NULL;
if (state->response != NULL)
{
free(state->response);
state->response = NULL;
}
if (state->client_creds == GSS_C_NO_CREDENTIAL)
{
if (auth_data && *auth_data)
{
int len;
input_token.value = base64_decode(auth_data, &len);
input_token.length = len;
}
else
{
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->message = strdup("No auth_data value in request from client");
response->return_code = AUTH_GSS_ERROR;
goto end;
}
maj_stat = gss_accept_sec_context(&min_stat,
&state->context,
state->server_creds,
&input_token,
GSS_C_NO_CHANNEL_BINDINGS,
&state->client_name,
NULL,
&output_token,
NULL,
NULL,
&state->client_creds);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_accept_sec_context", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
if (output_token.length)
{
state->response = base64_encode((const unsigned char *)output_token.value, output_token.length);
maj_stat = gss_release_buffer(&min_stat, &output_token);
}
}
maj_stat = gss_display_name(&min_stat, state->client_name, &output_token, NULL);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_display_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
state->username = (char *)malloc(output_token.length + 1);
strncpy(state->username, (char*) output_token.value, output_token.length);
state->username[output_token.length] = 0;
if (state->server_creds == GSS_C_NO_CREDENTIAL)
{
gss_name_t target_name = GSS_C_NO_NAME;
maj_stat = gss_inquire_context(&min_stat, state->context, NULL, &target_name, NULL, NULL, NULL, NULL, NULL);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_inquire_context", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
maj_stat = gss_display_name(&min_stat, target_name, &output_token, NULL);
if (GSS_ERROR(maj_stat))
{
response = gss_error(__func__, "gss_display_name", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
state->targetname = (char *)malloc(output_token.length + 1);
strncpy(state->targetname, (char*) output_token.value, output_token.length);
state->targetname[output_token.length] = 0;
}
if (state->constrained_delegation && state->client_creds != GSS_C_NO_CREDENTIAL)
{
if ((response = store_gss_creds(state)) != NULL)
{
goto end;
}
}
ret = AUTH_GSS_COMPLETE;
end:
if (output_token.length)
gss_release_buffer(&min_stat, &output_token);
if (input_token.value)
free(input_token.value);
if(response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = ret;
}
return response;
}
gss_client_response *authenticate_user_krb5_password(const char *username,
const char *password,
const char *service)
{
krb5_context context = NULL;
krb5_error_code problem;
krb5_principal user_principal = NULL;
krb5_get_init_creds_opt *opt = NULL;
krb5_creds creds;
bool auth_ok = false;
gss_client_response *response = NULL;
if (username == NULL || password == NULL || service == NULL) {
return other_error("username, password and service must all be non-null");
}
memset(&creds, 0, sizeof(creds));
problem = krb5_init_context(&context);
if (problem) {
response = other_error("unable to initialize krb5 context (%d)", (int)problem);
goto out;
}
problem = krb5_parse_name(context, username, &user_principal);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
problem = krb5_get_init_creds_opt_alloc(context, &opt);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
problem = krb5_get_init_creds_password(context, &creds, user_principal,
(char *)password, NULL,
NULL, 0, NULL, opt);
switch (problem) {
case 0:
auth_ok = true;
break;
case KRB5KDC_ERR_PREAUTH_FAILED:
case KRB5KRB_AP_ERR_BAD_INTEGRITY:
case KRB5KDC_ERR_C_PRINCIPAL_UNKNOWN:
auth_ok = false;
break;
default:
response = krb5_ctx_error(context, problem);
break;
}
if (auth_ok && strlen(service) > 0) {
response = verify_krb5_kdc(context, &creds, service);
}
out:
krb5_free_cred_contents(context, &creds);
if (opt != NULL) {
krb5_get_init_creds_opt_free(context, opt);
}
if (user_principal != NULL) {
krb5_free_principal(context, user_principal);
}
if (context != NULL) {
krb5_free_context(context);
}
if (response == NULL) {
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = auth_ok ? 1 : 0;
}
return response;
}
static gss_client_response *verify_krb5_kdc(krb5_context context,
krb5_creds *creds,
const char *service)
{
krb5_error_code problem;
krb5_keytab keytab = NULL;
krb5_ccache tmp_ccache = NULL;
krb5_principal server_princ = NULL;
krb5_creds *new_creds = NULL;
krb5_auth_context auth_context = NULL;
krb5_data req;
gss_client_response *response = NULL;
memset(&req, 0, sizeof(req));
problem = krb5_kt_default (context, &keytab);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
problem = krb5_cc_new_unique(context, "MEMORY", NULL, &tmp_ccache);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
problem = krb5_cc_initialize(context, tmp_ccache, creds->client);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
problem = krb5_cc_store_cred(context, tmp_ccache, creds);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
problem = krb5_parse_name(context, service, &server_princ);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
if (!krb5_principal_compare(context, server_princ, creds->server)) {
krb5_creds match_cred;
memset (&match_cred, 0, sizeof(match_cred));
match_cred.client = creds->client;
match_cred.server = server_princ;
problem = krb5_get_credentials(context, 0, tmp_ccache, &match_cred, &new_creds);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
creds = new_creds;
}
problem = krb5_mk_req_extended(context, &auth_context, 0, NULL, creds, &req);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
krb5_auth_con_free(context, auth_context);
auth_context = NULL;
problem = krb5_auth_con_init(context, &auth_context);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
krb5_auth_con_setflags(context, auth_context, KRB5_AUTH_CONTEXT_DO_SEQUENCE);
problem = krb5_rd_req(context, &auth_context, &req,
server_princ, keytab, 0, NULL);
if (problem) {
response = krb5_ctx_error(context, problem);
goto out;
}
out:
krb5_free_data_contents(context, &req);
if (auth_context) {
krb5_auth_con_free(context, auth_context);
}
if (new_creds) {
krb5_free_creds(context, new_creds);
}
if (server_princ) {
krb5_free_principal(context, server_princ);
}
if (tmp_ccache) {
krb5_cc_destroy (context, tmp_ccache);
}
if (keytab) {
krb5_kt_close (context, keytab);
}
return response;
}
static gss_client_response *store_gss_creds(gss_server_state *state)
{
OM_uint32 maj_stat, min_stat;
krb5_principal princ = NULL;
krb5_ccache ccache = NULL;
krb5_error_code problem;
krb5_context context;
gss_client_response *response = NULL;
problem = krb5_init_context(&context);
if (problem) {
response = other_error("No auth_data value in request from client");
return response;
}
problem = krb5_parse_name(context, state->username, &princ);
if (problem) {
response = krb5_ctx_error(context, problem);
goto end;
}
if ((response = create_krb5_ccache(state, context, princ, &ccache)))
{
goto end;
}
maj_stat = gss_krb5_copy_ccache(&min_stat, state->client_creds, ccache);
if (GSS_ERROR(maj_stat)) {
response = gss_error(__func__, "gss_krb5_copy_ccache", maj_stat, min_stat);
response->return_code = AUTH_GSS_ERROR;
goto end;
}
krb5_cc_close(context, ccache);
ccache = NULL;
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->return_code = AUTH_GSS_COMPLETE;
end:
if (princ)
krb5_free_principal(context, princ);
if (ccache)
krb5_cc_destroy(context, ccache);
krb5_free_context(context);
return response;
}
static gss_client_response *create_krb5_ccache(gss_server_state *state, krb5_context kcontext, krb5_principal princ, krb5_ccache *ccache)
{
char *ccname = NULL;
int fd;
krb5_error_code problem;
krb5_ccache tmp_ccache = NULL;
gss_client_response *error = NULL;
ccname = strdup("FILE:/tmp/krb5cc_nodekerberos_XXXXXX");
if (!ccname) die1("Memory allocation failed");
fd = mkstemp(ccname + strlen("FILE:"));
if (fd < 0) {
error = other_error("mkstemp() failed: %s", strerror(errno));
goto end;
}
close(fd);
problem = krb5_cc_resolve(kcontext, ccname, &tmp_ccache);
if (problem) {
error = krb5_ctx_error(kcontext, problem);
goto end;
}
problem = krb5_cc_initialize(kcontext, tmp_ccache, princ);
if (problem) {
error = krb5_ctx_error(kcontext, problem);
goto end;
}
state->delegated_credentials_cache = strdup(ccname);
*ccache = tmp_ccache;
tmp_ccache = NULL;
end:
if (tmp_ccache)
krb5_cc_destroy(kcontext, tmp_ccache);
if (ccname && error)
unlink(ccname);
if (ccname)
free(ccname);
return error;
}
gss_client_response *gss_error(const char *func, const char *op, OM_uint32 err_maj, OM_uint32 err_min) {
OM_uint32 maj_stat, min_stat;
OM_uint32 msg_ctx = 0;
gss_buffer_desc status_string;
gss_client_response *response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
char *message = NULL;
message = calloc(1024, 1);
if(message == NULL) die1("Memory allocation failed");
response->message = message;
int nleft = 1024;
int n;
n = snprintf(message, nleft, "%s(%s)", func, op);
message += n;
nleft -= n;
do {
maj_stat = gss_display_status (&min_stat,
err_maj,
GSS_C_GSS_CODE,
GSS_C_NO_OID,
&msg_ctx,
&status_string);
if(GSS_ERROR(maj_stat))
break;
n = snprintf(message, nleft, ": %.*s",
(int)status_string.length, (char*)status_string.value);
message += n;
nleft -= n;
gss_release_buffer(&min_stat, &status_string);
maj_stat = gss_display_status (&min_stat,
err_min,
GSS_C_MECH_CODE,
GSS_C_NULL_OID,
&msg_ctx,
&status_string);
if(!GSS_ERROR(maj_stat)) {
n = snprintf(message, nleft, ": %.*s",
(int)status_string.length, (char*)status_string.value);
message += n;
nleft -= n;
gss_release_buffer(&min_stat, &status_string);
}
} while (!GSS_ERROR(maj_stat) && msg_ctx != 0);
return response;
}
static gss_client_response *krb5_ctx_error(krb5_context context, krb5_error_code problem)
{
gss_client_response *response = NULL;
const char *error_text = krb5_get_error_message(context, problem);
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->message = strdup(error_text);
response->return_code = AUTH_GSS_ERROR;
krb5_free_error_message(context, error_text);
return response;
}
static gss_client_response *other_error(const char *fmt, ...)
{
size_t needed;
char *msg;
gss_client_response *response = NULL;
va_list ap, aps;
va_start(ap, fmt);
va_copy(aps, ap);
needed = vsnprintf(NULL, 0, fmt, aps) + 1;
va_end(aps);
msg = malloc(needed);
if (!msg) die1("Memory allocation failed");
vsnprintf(msg, needed, fmt, ap);
va_end(ap);
response = calloc(1, sizeof(gss_client_response));
if(response == NULL) die1("Memory allocation failed");
response->message = msg;
response->return_code = AUTH_GSS_ERROR;
return response;
}
#pragma clang diagnostic pop
