#pragma once

constexpr int N = 1024;
constexpr int d = 64;
constexpr float scale = 1.0f / 8.0f;

constexpr int SOFTMAX_DIM = 256;
constexpr int B_r = 16;
constexpr int B_c = 16;
constexpr int T_r = (N + B_r - 1) / B_r;
constexpr int T_c = (N + B_c - 1) / B_c;

void launch_naive(const float* q,
                  const float* k,
                  const float* v,
                  float* score_workspace,
                  float* o);

void launch_flash(const float* q,
                  const float* k,
                  const float* v,
                  float* o);
