// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIMULATOR_BASIC_H_
#define SIMULATOR_BASIC_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>
#include <stdio.h>

#include "simulator.h"
#include "statespace_basic.h"

static int test_counter = 0;
namespace qsim {

/**
 * Quantum circuit simulator without vectorization.
 */
template <typename For, typename FP = float>
class SimulatorBasic final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceBasic<For, FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorBasic(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      ApplyGateH<1>(qs, matrix, state);
      break;
    case 2:
      ApplyGateH<2>(qs, matrix, state);
      break;
    case 3:
      ApplyGateH<3>(qs, matrix, state);
      break;
    case 4:
      ApplyGateH<4>(qs, matrix, state);
      break;
    case 5:
      ApplyGateH<5>(qs, matrix, state);
      break;
    case 6:
      ApplyGateH<6>(qs, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      ApplyControlledGateH<1>(qs, cqs, cvals, matrix, state);
      break;
    case 2:
      ApplyControlledGateH<2>(qs, cqs, cvals, matrix, state);
      break;
    case 3:
      ApplyControlledGateH<3>(qs, cqs, cvals, matrix, state);
      break;
    case 4:
      ApplyControlledGateH<4>(qs, cqs, cvals, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Computes the expectation value of an operator using non-vectorized
   * instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      return ExpectationValueH<1>(qs, matrix, state);
      break;
    case 2:
      return ExpectationValueH<2>(qs, matrix, state);
      break;
    case 3:
      return ExpectationValueH<3>(qs, matrix, state);
      break;
    case 4:
      return ExpectationValueH<4>(qs, matrix, state);
      break;
    case 5:
      return ExpectationValueH<5>(qs, matrix, state);
      break;
    case 6:
      return ExpectationValueH<6>(qs, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }

    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 1;
  }

 private:
  template <unsigned H>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    // n = num_threads
    // m = current_thread
    // i = current_iteration
    // v = gate_matrix
    // ms = table of masks
    // xss = table of offset indices
    // rstate = state_vector
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      fp_type rs[hsize], is[hsize]; // rs = reals, is = imaginaries

      uint64_t ii = i & ms[0]; // ii = start offset in state vector
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii; // pointer to start index in state vector

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = *(p0 + xss[k]);
        is[k] = *(p0 + xss[k] + 1);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn += rs[l] * v[j] - is[l] * v[j + 1];
          in += rs[l] * v[j + 1] + is[l] * v[j];

          j += 2;
        }

        *(p0 + xss[k]) = rn;
        *(p0 + xss[k] + 1) = in;
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];


    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    if (test_counter < 6) {
        printf("test #%d\n", test_counter);
        printf("writing pre buffers to file...\n");
        char file_name[20];
        sprintf(file_name, "test%d_pre.bin", test_counter);
        FILE *test_file = fopen(file_name, "wb");
        unsigned local_h = H;
        uint64_t local_num_qubits = state.num_qubits();
        if (test_file) {
            fwrite(&local_h, sizeof(unsigned), 1, test_file);
            fwrite(&local_num_qubits, sizeof(uint64_t), 1, test_file);
            fwrite(&size, sizeof(uint64_t), 1, test_file);
            fwrite(ms, sizeof(uint64_t), H + 1, test_file);
            fwrite(xss, sizeof(uint64_t), 1 << H, test_file);
            fwrite(matrix, sizeof(fp_type), (1 << H) * (1 << H), test_file);
            fwrite(state.get(), sizeof(fp_type), 2 * (1 << state.num_qubits()), test_file);
            printf("finished writing buffers...\n");
        }
        fclose(test_file);
        unsigned h_read;
        uint64_t size_read;
        uint64_t q_read;
        uint64_t ms_read[H + 1];
        uint64_t xss_read[1 << H];
        fp_type matrix_read[(1 << H) * (1 << H)];
        fp_type state_read[2 * 1 << state.num_qubits()];
        FILE *read_test_file = fopen(file_name, "rb");
        if (read_test_file) {
            fread(&h_read, sizeof(unsigned), 1, read_test_file);
            fread(&q_read, sizeof(uint64_t), 1, read_test_file);
            fread(&size_read, sizeof(uint64_t), 1, read_test_file);
            fread(ms_read, sizeof(uint64_t), H + 1, read_test_file);
            fread(xss_read, sizeof(uint64_t), 1 << H, read_test_file);
            fread(matrix_read, sizeof(fp_type), (1 << H) * (1 << H), read_test_file);
            fread(state_read, sizeof(fp_type), 2 * (1 << state.num_qubits()), read_test_file);
            fclose(read_test_file);
        }
        printf("wrote h: %u\n", local_h);
        printf("read  h: %u\n", h_read);
        printf("wrote size: %u\n", size);
        printf("read  size: %u\n", size_read);
        printf("wrote qubits: %u\n", state.num_qubits());
        printf("read  qubits: %u\n", q_read);
        printf("wrote ms[0]: %u\n", ms[0]);
        printf("read  ms[0]: %u\n", ms_read[0]);
        printf("wrote state[0]: %f\n", state.get()[0]);
        printf("read  state[0]: %f\n", state_read[0]);
        printf("wrote xss[0]: %u\n", xss[0]);
        printf("read  xss[0]: %u\n", xss_read[0]);
    }

    for_.Run(size, f, matrix, ms, xss, state.get());
    if (test_counter < 6) {
        printf("writing post buffers to file...\n");
        char file_name[20];
        sprintf(file_name, "test%d_post.bin", test_counter);
        FILE *test_file = fopen(file_name, "wb");
        unsigned local_h = H;
        if (test_file) {
            fwrite(state.get(), sizeof(fp_type), 2 * (1 << state.num_qubits()), test_file);
            printf("finished writing buffers...\n");
        }

        fclose(test_file);
    }
    test_counter++;
  }

  template <unsigned H>
  void ApplyControlledGateH(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs,
                            uint64_t cvals, const fp_type* matrix,
                            State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t cvalsh, uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      fp_type rs[hsize], is[hsize];

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) == cvalsh) {
        auto p0 = rstate + 2 * ii;

        for (unsigned k = 0; k < hsize; ++k) {
          rs[k] = *(p0 + xss[k]);
          is[k] = *(p0 + xss[k] + 1);
        }

        uint64_t j = 0;

        for (unsigned k = 0; k < hsize; ++k) {
          rn = rs[0] * v[j] - is[0] * v[j + 1];
          in = rs[0] * v[j + 1] + is[0] * v[j];

          j += 2;

          for (unsigned l = 1; l < hsize; ++l) {
            rn += rs[l] * v[j] - is[l] * v[j + 1];
            in += rs[l] * v[j + 1] + is[l] * v[j];

            j += 2;
          }

          *(p0 + xss[k]) = rn;
          *(p0 + xss[k] + 1) = in;
        }
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    auto m = GetMasks7(state.num_qubits(), qs, cqs, cvals);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      fp_type rs[hsize], is[hsize];

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = *(p0 + xss[k]);
        is[k] = *(p0 + xss[k] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn += rs[l] * v[j] - is[l] * v[j + 1];
          in += rs[l] * v[j + 1] + is[l] * v[j];

          j += 2;
        }

        re += rs[k] * rn + is[k] * in;
        im += rs[k] * in - is[k] * rn;
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, state.get());
  }

  For for_;
};

}  // namespace qsim

#endif  // SIMULATOR_BASIC_H_
