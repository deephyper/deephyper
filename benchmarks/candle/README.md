# Benchmarks
ECP-CANDLE Benchmarks


This repository contains the CANDLE benchmark codes. These codes implement deep learning architectures that are relevant to problems in cancer. These architectures address problems at different biological scales, specifically problems at the molecular, cellular and population scales.

The naming conventions adopted reflect the different biological scales.

Pilot1 (P1) benchmarks are formed out of problems and data at the cellular level. The high level goal of the problem behind the P1 benchmarks is to predict drug response based on molecular features of tumor cells and drug descriptors.

Pilot2 (P2) benchmarks are formed out of problems and data at the molecular level. The high level goal of the problem behind the P2 benchmarks is molecular dynamic simulations of proteins involved in cancer, specifically the RAS protein.

Pilot3 (P3) benchmarks are formed out of problems and data at the population level. The high level goal of the problem behind the P3 benchmarks is to predict cancer recurrence in patients based on patient related data.

Each of the problems (P1,P2,P3) informed the implementation of specific benchmarks, so P1B3 would be benchmark three of problem 1.
At this point, we will refer to a benchmark by it's problem area and benchmark number. So it's natural to talk of the P1B1 benchmark. Inside each benchmark directory, there exists a readme file that contains an overview of the benchmark, a description of the data and expected outcomes along with instructions for running the benchmark code.

Over time, we will be adding implementations that make use of different tensor frameworks. The primary (baseline) benchmarks are implemented using keras, and are named with '_baseline' in the name, for example p3b1_baseline_keras2.py. 

Implementations that use alternative tensor frameworks, such as mxnet or neon, will have the name of the framework in the name. Examples can be seen in the P1B3 benchmark contribs/ directory, for example:
        p1b3_mxnet.py
        p1b3_neon.py
