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

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_cirq.h"
#include "../lib/io.h"
#include "../lib/qtrajectory.h"
#include "../lib/simmux.h"

namespace qsim {

namespace {

namespace types {

using StateSpace = Simulator<For>::StateSpace;
using State = StateSpace::State;
using fp_type = StateSpace::fp_type;
using Gate = Cirq::GateCirq<fp_type>;
using QTSimulator = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                               Simulator<For>>;
using NoisyCircuit = NoisyCircuit<Gate>;

}  // namespace types

void AddBitFlipNoise1(
    unsigned time, unsigned q, double p, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - p;
  double p2 = p;

  auto normal = KrausOperator<types::Gate>::kNormal;

  ncircuit.push_back({{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, q)}},
                      {normal, 1, p2, {Cirq::X<fp_type>::Create(time, q)}}});
}

void AddBitFlipNoise2(unsigned time, double p, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - p;
  double p2 = p;

  auto normal = KrausOperator<types::Gate>::kNormal;

  ncircuit.push_back({
    {normal, 1, p1 * p1, {Cirq::I1<fp_type>::Create(time, 0),
                          Cirq::I1<fp_type>::Create(time, 1)}},
    {normal, 1, p1 * p2, {Cirq::I1<fp_type>::Create(time, 0),
                          Cirq::X<fp_type>::Create(time, 1)}},
    {normal, 1, p2 * p1, {Cirq::X<fp_type>::Create(time, 0),
                          Cirq::I1<fp_type>::Create(time, 1)}},
    {normal, 1, p2 * p2, {Cirq::X<fp_type>::Create(time, 0),
                          Cirq::X<fp_type>::Create(time, 1)}},
  });

//  This can also be imnplemented as the following.
//
//  ncircuit.push_back({{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, 0)}},
//                      {normal, 1, p2, {Cirq::X<fp_type>::Create(time, 0)}}});
//  ncircuit.push_back({{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, 1)}},
//                      {normal, 1, p2, {Cirq::X<fp_type>::Create(time, 1)}}});
}

void AddPhaseDumpNoise1(
    unsigned time, unsigned q, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, q, {0, 0, 0, 0, 0, 0, s, 0})}}});
}

void AddPhaseDumpNoise2(
    unsigned time, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {0, 0, 0, 0, 0, 0, s, 0})}}});
  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {0, 0, 0, 0, 0, 0, s, 0})}}});
}

void AddAmplDumpNoise1(
    unsigned time, unsigned q, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, q, {0, 0, s, 0, 0, 0, 0, 0})}}});
}

void AddAmplDumpNoise2(unsigned time, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {0, 0, s, 0, 0, 0, 0, 0})}}});
  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {0, 0, s, 0, 0, 0, 0, 0})}}});
}

void AddGenAmplDumpNoise1(
    unsigned time, unsigned q, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  // Probability of exchanging energy with the environment.
  double p = 0.5;

  double p1 = p * (1 - g);
  double p2 = (1 - p) * (1 - g);
  double p3 = 0;

  fp_type t1 = std::sqrt(p);
  fp_type r1 = std::sqrt(p * (1 - g));
  fp_type s1 = std::sqrt(p * g);
  fp_type t2 = std::sqrt(1 - p);
  fp_type r2 = std::sqrt((1 - p) * (1 - g));
  fp_type s2 = std::sqrt((1 - p) * g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, q, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, q, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, q, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, q, {0, 0, 0, 0, s2, 0, 0, 0})}}});
}

void AddGenAmplDumpNoise2(
    unsigned time, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  // Probability of exchanging energy with the environment.
  double p = 0.5;

  double p1 = p * (1 - g);
  double p2 = (1 - p) * (1 - g);
  double p3 = 0;

  fp_type t1 = std::sqrt(p);
  fp_type r1 = std::sqrt(p * (1 - g));
  fp_type s1 = std::sqrt(p * g);
  fp_type t2 = std::sqrt(1 - p);
  fp_type r2 = std::sqrt((1 - p) * (1 - g));
  fp_type s2 = std::sqrt((1 - p) * g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0})}}});
  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}}});
}

template <typename AddNoise1, typename AddNoise2>
types::NoisyCircuit GenerateNoisyCircuit(
    double p, AddNoise1&& add_noise1, AddNoise2&& add_noise2) {
  using fp_type = types::Gate::fp_type;

  types::NoisyCircuit ncircuit;
  ncircuit.reserve(24);

  using Hd = Cirq::H<fp_type>;
  using IS = Cirq::ISWAP<fp_type>;
  using Rx = Cirq::rx<fp_type>;
  using Ry = Cirq::ry<fp_type>;

  auto normal = KrausOperator<types::Gate>::kNormal;

  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 0)}}});
  add_noise1(1, 0, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 1)}}});
  add_noise1(1, 1, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {IS::Create(2, 0, 1)}}});
  add_noise2(3, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Rx::Create(4, 0, 0.7)}}});
  add_noise1(5, 0, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Ry::Create(4, 1, 0.1)}}});
  add_noise1(5, 1, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {IS::Create(6, 0, 1)}}});
  add_noise2(7, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Ry::Create(8, 0, 0.4)}}});
  add_noise1(9, 0, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Rx::Create(8, 1, 0.7)}}});
  add_noise1(9, 1, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {IS::Create(10, 0, 1)}}});
  add_noise2(11, p, ncircuit);
  ncircuit.push_back({{KrausOperator<types::Gate>::kMeasurement, 1, 1.0,
                       {gate::Measurement<types::Gate>::Create(12, {0, 1})}}});
  add_noise2(13, p, ncircuit);

  return ncircuit;
}

void RunBatch(const types::NoisyCircuit& ncircuit,
          const std::vector<double>& expected_results) {
  unsigned num_qubits = 2;
  unsigned num_reps = 25000;

  auto measure = [](uint64_t r, const types::State& state,
                    const std::vector<uint64_t>& stat,
                    std::vector<unsigned>& histogram) {
    ASSERT_EQ(stat.size(), 1);
    ++histogram[stat[0]];
  };

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  types::QTSimulator::Parameter param;
  param.collect_mea_stat = true;

  EXPECT_TRUE(types::QTSimulator::Run(param, num_qubits, ncircuit,
                                      0, num_reps, measure, histogram));

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

void RunOnceRepeatedly(const types::NoisyCircuit& ncircuit,
          const std::vector<double>& expected_results) {
  unsigned num_qubits = 2;
  unsigned num_reps = 25000;

  types::StateSpace state_space(1);

  types::State scratch = state_space.Null();
  types::State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  auto state_pointer = state.get();

  std::vector<uint64_t> stat;

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  types::QTSimulator::Parameter param;
  param.collect_mea_stat = true;

  for (unsigned i = 0; i < num_reps; ++i) {
    state_space.SetStateZero(state);

    EXPECT_TRUE(types::QTSimulator::Run(param, num_qubits, ncircuit, i,
                                        scratch, state, stat));

    EXPECT_EQ(state_pointer, state.get());

    ASSERT_EQ(stat.size(), 1);
    ++histogram[stat[0]];
  }

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

}  // namespace

TEST(QTrajectoryTest, BitFlip) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

circuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
)

ncircuit = circuit.with_noise(cirq.bit_flip(0.01))

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.389352, 0.242790, 0.081009, 0.286850,
  };

  auto ncircuit1 = GenerateNoisyCircuit(0.01, AddBitFlipNoise1,
                                        AddBitFlipNoise2);
  RunBatch(ncircuit1, expected_results);
}

TEST(QTrajectoryTest, PhaseDump) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.phase_damp(0.02)

ncircuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
  channel.on(qs[0]),
  channel.on(qs[1]),
)

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.412300, 0.230500, 0.057219, 0.299982,
  };

  auto ncircuit = GenerateNoisyCircuit(0.02, AddPhaseDumpNoise1,
                                       AddPhaseDumpNoise2);
  RunOnceRepeatedly(ncircuit, expected_results);
}

TEST(QTrajectoryTest, AmplDump) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.amplitude_damp(0.05)

ncircuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
  channel.on(qs[0]),
  channel.on(qs[1]),
)

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.500494, 0.235273, 0.090879, 0.173354,
  };

  auto ncircuit = GenerateNoisyCircuit(0.05, AddAmplDumpNoise1,
                                       AddAmplDumpNoise2);
  RunBatch(ncircuit, expected_results);
}

TEST(QTrajectoryTest, GenDump) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.generalized_amplitude_damp(0.5, 0.1)

ncircuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
  channel.on(qs[0]),
  channel.on(qs[1]),
)

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.318501, 0.260538, 0.164616, 0.256345,
  };

  auto ncircuit = GenerateNoisyCircuit(0.1, AddGenAmplDumpNoise1,
                                       AddGenAmplDumpNoise2);
  RunOnceRepeatedly(ncircuit, expected_results);
}

TEST(QTrajectoryTest, CollectKopStat) {
  unsigned num_qubits = 4;
  unsigned num_reps = 20000;
  double p = 0.1;

  double p1 = 1 - p;
  double p2 = p;

  using fp_type = types::Gate::fp_type;
  using Hd = Cirq::H<fp_type>;
  using I = Cirq::I1<fp_type>;
  using X = Cirq::X<fp_type>;

  auto normal = KrausOperator<types::Gate>::kNormal;

  types::NoisyCircuit ncircuit;
  ncircuit.reserve(8);

  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 0)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 1)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 2)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 3)}}});

  // Add bit flip noise.
  ncircuit.push_back({{normal, 1, p1, {I::Create(1, 0)}},
                      {normal, 1, p2, {X::Create(1, 0)}}});
  ncircuit.push_back({{normal, 1, p1, {I::Create(1, 1)}},
                      {normal, 1, p2, {X::Create(1, 1)}}});
  ncircuit.push_back({{normal, 1, p1, {I::Create(1, 2)}},
                      {normal, 1, p2, {X::Create(1, 2)}}});
  ncircuit.push_back({{normal, 1, p1, {I::Create(1, 3)}},
                      {normal, 1, p2, {X::Create(1, 3)}}});

  auto measure = [](uint64_t r, const types::State& state,
                    const std::vector<uint64_t>& stat,
                    std::vector<std::vector<unsigned>>& histogram) {
    ASSERT_EQ(stat.size(), histogram.size());
    for (std::size_t i = 0; i < histogram.size(); ++i) {
      ++histogram[i][stat[i]];
    }
  };

  std::vector<std::vector<unsigned>> histogram(8, std::vector<unsigned>(2, 0));

  types::QTSimulator::Parameter param;
  param.collect_kop_stat = true;

  EXPECT_TRUE(types::QTSimulator::Run(param, num_qubits, ncircuit,
                                      0, num_reps, measure, histogram));

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(histogram[i][0], num_reps);
    EXPECT_EQ(histogram[i][1], 0);
  }

  for (std::size_t i = 4; i < 8; ++i) {
    EXPECT_NEAR(double(histogram[i][0]) / num_reps, p1, 0.005);
    EXPECT_NEAR(double(histogram[i][1]) / num_reps, p2, 0.005);
  }
}

TEST(QTrajectoryTest, CleanCircuit) {
  unsigned num_qubits = 4;
  auto size = uint64_t{1} << num_qubits;

  std::vector<types::Gate> circuit;
  circuit.reserve(16);

  using fp_type = types::Gate::fp_type;

  circuit.push_back(Cirq::H<fp_type>::Create(0, 0));
  circuit.push_back(Cirq::H<fp_type>::Create(0, 1));
  circuit.push_back(Cirq::H<fp_type>::Create(0, 2));
  circuit.push_back(Cirq::H<fp_type>::Create(0, 3));

  circuit.push_back(Cirq::T<fp_type>::Create(1, 0));
  circuit.push_back(Cirq::T<fp_type>::Create(1, 1));
  circuit.push_back(Cirq::T<fp_type>::Create(1, 2));
  circuit.push_back(Cirq::T<fp_type>::Create(1, 3));

  circuit.push_back(Cirq::CX<fp_type>::Create(2, 0, 2));
  circuit.push_back(Cirq::CZ<fp_type>::Create(2, 1, 3));

  circuit.push_back(Cirq::XPowGate<fp_type>::Create(3, 0, 0.3, 1.1));
  circuit.push_back(Cirq::YPowGate<fp_type>::Create(3, 1, 0.4, 1.0));
  circuit.push_back(Cirq::ZPowGate<fp_type>::Create(3, 2, 0.5, 0.9));
  circuit.push_back(Cirq::HPowGate<fp_type>::Create(3, 3, 0.6, 0.8));

  circuit.push_back(Cirq::CZPowGate<fp_type>::Create(4, 0, 1, 0.7, 0.2));
  circuit.push_back(Cirq::CXPowGate<fp_type>::Create(4, 2, 3, 1.2, 0.4));

  circuit.push_back(Cirq::HPowGate<fp_type>::Create(5, 0, 0.7, 0.2));
  circuit.push_back(Cirq::XPowGate<fp_type>::Create(5, 1, 0.8, 0.3));
  circuit.push_back(Cirq::YPowGate<fp_type>::Create(5, 2, 0.9, 0.4));
  circuit.push_back(Cirq::ZPowGate<fp_type>::Create(5, 3, 1.0, 0.5));

  types::NoisyCircuit ncircuit;
  ncircuit.reserve(16);

  auto normal = KrausOperator<types::Gate>::kNormal;

  for (std::size_t i = 0; i < circuit.size(); ++i) {
    ncircuit.push_back({{normal, 1, 1.0, {circuit[i]}}});
  }

  Simulator<For> simulator(1);
  types::StateSpace state_space(1);

  types::State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  // Run clean-circuit simulator.
  for (const auto& gate : circuit) {
    ApplyGate(simulator, gate, state);
  }

  types::State scratch = state_space.Null();
  types::State nstate = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(nstate));

  std::vector<uint64_t> stat;

  types::QTSimulator::Parameter param;

  state_space.SetStateZero(nstate);

  // Run quantum trajectory simulator.
  EXPECT_TRUE(types::QTSimulator::Run(param, num_qubits, ncircuit, 0,
                                      scratch, nstate, stat));

  EXPECT_EQ(stat.size(), 0);

  for (uint64_t i = 0; i < size; ++i) {
    auto a1 = state_space.GetAmpl(state, i);
    auto a2 = state_space.GetAmpl(nstate, i);
    EXPECT_NEAR(std::real(a1), std::real(a2), 1e-6);
    EXPECT_NEAR(std::imag(a1), std::imag(a2), 1e-6);
  }
}

// Test that QTSimulator::Run does not overwrite initial states.
TEST(QTrajectoryTest, InitialState) {
  unsigned num_qubits = 3;

  types::NoisyCircuit ncircuit;
  ncircuit.reserve(2);

  using fp_type = types::Gate::fp_type;
  auto normal = KrausOperator<types::Gate>::kNormal;

  ncircuit.push_back({{normal, 1, 1.0, {Cirq::X<fp_type>::Create(0, 0)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Cirq::X<fp_type>::Create(0, 1)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Cirq::X<fp_type>::Create(0, 2)}}});

  types::StateSpace state_space(1);

  types::State scratch = state_space.Null();
  types::State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  types::QTSimulator::Parameter param;
  std::vector<uint64_t> stat;

  for (unsigned i = 0; i < 8; ++i) {
    state_space.SetAmpl(state, i, 1 + i, 0);
  }

  EXPECT_TRUE(types::QTSimulator::Run(param, num_qubits, ncircuit, 0,
                                      scratch, state, stat));

  // Expect reversed order of amplitudes.
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(std::real(state_space.GetAmpl(state, i)), 8 - i);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
