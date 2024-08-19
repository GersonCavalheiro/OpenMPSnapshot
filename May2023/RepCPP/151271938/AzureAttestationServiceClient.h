#pragma once

#include <enclave/outside/attestation/EnclaveInfo.h>

#include <jwt-cpp/jwt.h>
#include <string>

namespace sgx {

typedef jwt::decoded_jwt<jwt::traits::kazuho_picojson> DecodedJwt;
typedef jwt::jwks<jwt::traits::kazuho_picojson> JwksSet;


class AzureAttestationServiceClient
{
private:
std::string attestationServiceUrl;
std::string certificateEndpoint;
std::string tenantName;

JwksSet cachedJwks;

JwksSet fetchJwks();

void validateJkuUri(const DecodedJwt& decodedJwt);

void validateJwtSignature(const DecodedJwt& decodedJwt);

public:
static std::string requestBodyFromEnclaveInfo(
const EnclaveInfo& enclaveInfo);

AzureAttestationServiceClient(const std::string& attestationServiceUrlIn);

std::string attestEnclave(const EnclaveInfo& enclaveInfo);

void validateJwtToken(const std::string& jwtToken);
};
}
