#include <iostream>
#include <chrono>
#include <math.h>
#include "TVflow/tvflow.hpp"

int main()
{
  int W = 291;
  int H = 305;
  Eigen::MatrixXd g = Eigen::MatrixXd::Random(W, H);
  g = (g + Eigen::MatrixXd::Constant(W, H, 1.)) * 5 / 2.;

  //Eigen::MatrixXd f = openData("D:\\UWaterloo\\Tizhoosh\\TV\\f.csv");
  Eigen::MatrixXd f = openData("D:\\UWaterloo\\Tizhoosh\\TV\\PythonExperiments\\f.csv");

  float T = 10.0;
  float dt = 1.0 / 50.0;

  int NOB = int(ceil(T / dt));

  int NIT = 100;
  float tol = 1e-5;

  float lami = 2 * dt;

  auto start = std::chrono::steady_clock::now();

  Eigen::VectorXd S = run_TV_flow(f, W, H, NOB, lami, dt, tol, NIT);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << S << std::endl;
  saveData("S_test.csv", S);
  
  std::cout << "Time taken by the program: " << diff.count() << " seconds\n";

}
