#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rb {

static unsigned char const kCRC8Table[] = {
    0x00, 0x31, 0x62, 0x53, 0xc4, 0xf5, 0xa6, 0x97, 0xb9, 0x88, 0xdb, 0xea, 0x7d, 0x4c, 0x1f, 0x2e, 0x43, 0x72, 0x21,
    0x10, 0x87, 0xb6, 0xe5, 0xd4, 0xfa, 0xcb, 0x98, 0xa9, 0x3e, 0x0f, 0x5c, 0x6d, 0x86, 0xb7, 0xe4, 0xd5, 0x42, 0x73,
    0x20, 0x11, 0x3f, 0x0e, 0x5d, 0x6c, 0xfb, 0xca, 0x99, 0xa8, 0xc5, 0xf4, 0xa7, 0x96, 0x01, 0x30, 0x63, 0x52, 0x7c,
    0x4d, 0x1e, 0x2f, 0xb8, 0x89, 0xda, 0xeb, 0x3d, 0x0c, 0x5f, 0x6e, 0xf9, 0xc8, 0x9b, 0xaa, 0x84, 0xb5, 0xe6, 0xd7,
    0x40, 0x71, 0x22, 0x13, 0x7e, 0x4f, 0x1c, 0x2d, 0xba, 0x8b, 0xd8, 0xe9, 0xc7, 0xf6, 0xa5, 0x94, 0x03, 0x32, 0x61,
    0x50, 0xbb, 0x8a, 0xd9, 0xe8, 0x7f, 0x4e, 0x1d, 0x2c, 0x02, 0x33, 0x60, 0x51, 0xc6, 0xf7, 0xa4, 0x95, 0xf8, 0xc9,
    0x9a, 0xab, 0x3c, 0x0d, 0x5e, 0x6f, 0x41, 0x70, 0x23, 0x12, 0x85, 0xb4, 0xe7, 0xd6, 0x7a, 0x4b, 0x18, 0x29, 0xbe,
    0x8f, 0xdc, 0xed, 0xc3, 0xf2, 0xa1, 0x90, 0x07, 0x36, 0x65, 0x54, 0x39, 0x08, 0x5b, 0x6a, 0xfd, 0xcc, 0x9f, 0xae,
    0x80, 0xb1, 0xe2, 0xd3, 0x44, 0x75, 0x26, 0x17, 0xfc, 0xcd, 0x9e, 0xaf, 0x38, 0x09, 0x5a, 0x6b, 0x45, 0x74, 0x27,
    0x16, 0x81, 0xb0, 0xe3, 0xd2, 0xbf, 0x8e, 0xdd, 0xec, 0x7b, 0x4a, 0x19, 0x28, 0x06, 0x37, 0x64, 0x55, 0xc2, 0xf3,
    0xa0, 0x91, 0x47, 0x76, 0x25, 0x14, 0x83, 0xb2, 0xe1, 0xd0, 0xfe, 0xcf, 0x9c, 0xad, 0x3a, 0x0b, 0x58, 0x69, 0x04,
    0x35, 0x66, 0x57, 0xc0, 0xf1, 0xa2, 0x93, 0xbd, 0x8c, 0xdf, 0xee, 0x79, 0x48, 0x1b, 0x2a, 0xc1, 0xf0, 0xa3, 0x92,
    0x05, 0x34, 0x67, 0x56, 0x78, 0x49, 0x1a, 0x2b, 0xbc, 0x8d, 0xde, 0xef, 0x82, 0xb3, 0xe0, 0xd1, 0x46, 0x77, 0x24,
    0x15, 0x3b, 0x0a, 0x59, 0x68, 0xff, 0xce, 0x9d, 0xac};

inline unsigned char CalculateCRC8(unsigned char* mem, size_t len) {
  unsigned char crc = 0xff;
  unsigned char const* data = mem;
  if (data == nullptr)
    return 0xff;
  crc &= 0xff;
  while (len--)
    crc = kCRC8Table[crc ^ *data++];
  return crc;
}

// RPC -> UPC
// 0x24 0x24 len:ushort N:uchar t:double R:uchar[0...N-1] P:double[0...N-1] V:double[0...N-1] C:double[0...N-1] T:double[0...N-1] CRC 0x25 0x25

// UPC -> RPC
// 0x24 0x24 len:ushort UID:uint64 N:uchar Mode:uchar[0...N-1] T:double[0...N-1] FBG:uint[0...N-1] FFT:double[0...N-1] Finished:uchar CRC 0x25 0x25

// mode - if 0, position, if 1, velocity
int BuildRobotCommandRTPacket(unsigned char* packet, uint64_t uid, size_t N, const bool* mode, const double* target,
                              const unsigned int* feedback_gain, const double* feedforward_torque, bool finished) {
  unsigned short len = 8 +                  // UID:uint64
                       1 +                  // N:uchar
                       N +                  // Mode:uchar[0...N-1]
                       8 * N * 2 + 4 * N +  // T,FBG,FFT:double[0...N-1]
                       1 +                  // Finished:uchar
                       1 +                  // CRC
                       2                    // footer
      ;

  int idx = 0;
  packet[idx++] = 0x24;
  packet[idx++] = 0x24;
  packet[idx++] = len & 0xff;
  packet[idx++] = ((len >> 8) & 0xff);
  for (int i = 0; i < 8; i++) {
    packet[idx++] = (uid >> (8 * i)) & 0xff;
  }
  packet[idx++] = (N & 0xff);
  for (int i = 0; i < N; i++) {
    packet[idx++] = (mode[i] ? 1 : 0);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&target[i]), sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&feedback_gain[i]), sizeof(unsigned int));
    idx += sizeof(unsigned int);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&feedforward_torque[i]), sizeof(double));
    idx += sizeof(double);
  }
  packet[idx++] = (finished ? 1 : 0);
  packet[idx] = CalculateCRC8(packet, idx);
  idx++;
  packet[idx++] = 0x25;
  packet[idx++] = 0x25;

  return idx;
}

int BuildRobotStateRTPacket(unsigned char* packet, size_t N, double t, const bool* is_ready, const double* position,
                            const double* velocity, const double* current, const double* torque) {
  unsigned short len = 1 +          // N:uchar // degree of freedom
                       8 +          // t:double // time
                       N +          // R:unchar[0...N-1] // is_ready
                       8 * N * 4 +  // P,V,C,T:double[0...N-1] // position, velocity, current, torque
                       1 +          // CRC
                       2            // footer
      ;

  int idx = 0;
  packet[idx++] = 0x24;
  packet[idx++] = 0x24;
  packet[idx++] = len & 0xff;
  packet[idx++] = ((len >> 8) & 0xff);
  packet[idx++] = (N & 0xff);
  memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&t), sizeof(double));
  idx += sizeof(double);
  for (int i = 0; i < N; i++) {
    packet[idx++] = (is_ready[i] ? 1 : 0);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&position[i]), sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&velocity[i]), sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&current[i]), sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < N; i++) {
    memcpy(&packet[idx], reinterpret_cast<const unsigned char*>(&torque[i]), sizeof(double));
    idx += sizeof(double);
  }
  packet[idx] = CalculateCRC8(packet, idx);
  idx++;
  packet[idx++] = 0x25;
  packet[idx++] = 0x25;

  return idx;
}

std::pair<bool, int> ValidateRTProtocol(const unsigned char* packet, int packet_size) {
  if (packet_size < 2) {
    return {false, 0};
  }

  if (packet[0] == 0x24 && packet[1] == 0x24)
    ;
  else {
    return {false, 2};
  }

  if (packet_size < 4) {
    return {false, 0};
  }
  unsigned int len = (unsigned int)(packet[2]) | (unsigned int)(packet[3] << 8);

  if (packet_size < 2 + 2 + len) {
    return {false, 0};
  }

  if (packet[2 + 2 + len - 1] == 0x25 && packet[2 + 2 + len - 2] == 0x25)
    ;
  else {
    return {false, 2 + 2 + len};
  }

  if (packet[2 + 2 + len - 3] == CalculateCRC8((unsigned char*)packet, 2 + 2 + len - 3))
    ;
  else {
    return {false, 2 + 2 + len};
  }

  return {true, 2 + 2 + len};
}

// N
size_t GetInfoRobotStateRTProtocol(const unsigned char* packet) {
  return (size_t)(packet[4]);
}

// UID, N
std::pair<uint64_t, size_t> GetInfoRobotCommandRTProtocol(const unsigned char* packet) {
  uint64_t uid = 0;
  for (int i = 0; i < 8; i++) {
    uid |= ((uint64_t)packet[4 + i] << (8 * i));
  }
  return {uid, (size_t)(packet[4])};
}

void ParseRobotStateRTProtocol(const unsigned char* packet, size_t* N, double* t, bool* is_ready, double* position,
                               double* velocity, double* current, double* torque) {
  int idx = 4;

  *N = packet[idx++];

  memcpy(t, &packet[idx], sizeof(double));
  idx += sizeof(double);

  for (int i = 0; i < *N; i++) {
    is_ready[i] = packet[idx++] == 1;
  }
  for (int i = 0; i < *N; i++) {
    memcpy(&position[i], &packet[idx], sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < *N; i++) {
    memcpy(&velocity[i], &packet[idx], sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < *N; i++) {
    memcpy(&current[i], &packet[idx], sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < *N; i++) {
    memcpy(&torque[i], &packet[idx], sizeof(double));
    idx += sizeof(double);
  }
}

void ParseRobotCommandRTProtocol(const unsigned char* packet, uint64_t* uid, size_t* N, bool* mode, double* target,
                                 unsigned int* feedback_gain, double* feedforward_torque, bool* finished) {
  int idx = 4;

  if (uid != nullptr) {
    *uid = 0;
  }
  for (int i = 0; i < 8; i++) {
    if (uid != nullptr) {
      *uid |= (uint64_t)(packet[idx] << (i * 8));
    }
    idx++;
  }

  size_t dof = packet[idx];
  if (N != nullptr) {
    *N = dof;
  }
  idx++;

  for (int i = 0; i < dof; i++) {
    mode[i] = packet[idx++] == 1;
  }
  for (int i = 0; i < dof; i++) {
    memcpy(&target[i], &packet[idx], sizeof(double));
    idx += sizeof(double);
  }
  for (int i = 0; i < dof; i++) {
    memcpy(&feedback_gain[i], &packet[idx], sizeof(unsigned int));
    idx += sizeof(unsigned int);
  }
  for (int i = 0; i < dof; i++) {
    memcpy(&feedforward_torque[i], &packet[idx], sizeof(double));
    idx += sizeof(double);
  }
  *finished = packet[idx++] == 1;
}

}  // namespace rb