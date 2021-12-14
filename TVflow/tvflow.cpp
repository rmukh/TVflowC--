#include "tvflow.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include "math.h"

void derivative_index_2D(const Eigen::Ref<Eigen::MatrixXd> &image, Eigen::Ref<Eigen::MatrixX4d> &derivative_index)
{
    unsigned long *I, nm, i, j, k, l, N, vi, vj;

    nm = image.rows() * image.cols();

    I = new unsigned long[nm];

    k = 0;
    for (i = 0; i < nm; i++)
    {
        I[k] = i;
        k++;
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < 4; j++)
        {
            derivative_index(i, j) = (double)i;
        }
    }

    i = 0;
    for (k = 0; k < nm; k++)
    {
        vi = k % image.rows();
        vj = (k - vi) / image.rows();

        if (vi != image.rows() - 1)
        {
            derivative_index(i, 0) = (double)(i + 1);
        }

        if (vi != 0)
        {
            derivative_index(i, 1) = (double)(i - 1);
        }

        if (vj != image.cols() - 1)
        {
            l = k + image.rows();
            j = 0;
            while (I[i + j] < l)
            {
                j++;
            }
            derivative_index(i, 2) = (double)(i + j);
        }

        if (vj != 0)
        {
            l = k - image.rows();
            j = 0;
            while (I[i - j] > l)
            {
                j++;
            }
            derivative_index(i, 3) = (double)(i - j);
        }

        i++;
    }

    delete[] I;
}

void grad(const Eigen::Ref<Eigen::MatrixXd> &img, Eigen::Ref<Eigen::MatrixXd> out1, Eigen::Ref<Eigen::MatrixXd> out2, int n, int m)
{
    out1 << img(Eigen::seq(1, n - 1), Eigen::all) - img(Eigen::seq(0, n - 2), Eigen::all), Eigen::MatrixXd::Zero(1, m);
    out2 << img(Eigen::all, Eigen::seq(1, m - 1)) - img(Eigen::all, Eigen::seq(0, m - 2)), Eigen::MatrixXd::Zero(n, 1);
}

void div(const Eigen::Ref<Eigen::MatrixXd> &g1, const Eigen::Ref<Eigen::MatrixXd> &g2, Eigen::Ref<Eigen::MatrixXd> out1, Eigen::Ref<Eigen::MatrixXd> out2, int n, int m)
{
    out1 << -g1(0, Eigen::all), g1(Eigen::seq(0, n - 3), Eigen::all) - g1(Eigen::seq(1, n - 2), Eigen::all), g1(n - 1, Eigen::all);
    out2 << -g2(Eigen::all, 0), g2(Eigen::all, Eigen::seq(0, m - 3)) - g2(Eigen::all, Eigen::seq(1, m - 2)), g2(Eigen::all, m - 1);
}

void tvdff(const Eigen::Ref<Eigen::MatrixXd> &f, Eigen::Ref<Eigen::MatrixXd> out, int n, int m, double lmd, double tol, int iters)
{
    Eigen::MatrixXd v_old_1 = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_old_2 = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_hat_1 = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_hat_2 = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_1 = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_2 = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd d_1 = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd d_2 = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd div_out = Eigen::MatrixXd::Zero(n, m);
    Eigen::ArrayXXd norm_denom = Eigen::MatrixXd::Zero(n, m);

    double told = 1.0;
    double t = 1.0;
    double dt = 1.0;

    for (int itr{0}; itr < iters; ++itr)
    {
        div(v_hat_1, v_hat_2, v_1, v_2, n, m);
        div_out = v_1 + v_2;
        div_out.noalias() -= f;
        grad(div_out, v_1, v_2, n, m);

        v_1.noalias() = v_hat_1 - (v_1 / 6.0);
        v_2.noalias() = v_hat_2 - (v_2 / 6.0);

        norm_denom = ((v_1.array().square() + v_2.array().square()).sqrt()).cwiseMax(lmd);

        v_1 = (lmd * v_1.array()) / norm_denom;
        v_2 = (lmd * v_2.array()) / norm_denom;

        d_1.noalias() = v_1 - v_old_1;
        d_2.noalias() = v_2 - v_old_2;

        if (itr % 5 == 0)
        {
            if ((d_1.array().abs()).maxCoeff() < tol && (d_2.array().abs()).maxCoeff() < tol)
            {
                break;
            }
        }

        t = (1.0 + std::sqrt(1.0 + 4.0 * told * told)) / 2.0;
        dt = (told - 1.0) / t;

        v_hat_1.noalias() = v_1 + dt * d_1;
        v_hat_2.noalias() = v_2 + dt * d_2;

        v_old_1.noalias() = v_1;
        v_old_2.noalias() = v_2;

        told = t;
    }

    // use v_old as the output since it is the last iteration and we don't need them anymore
    div(v_1, v_2, v_old_1, v_old_2, n, m);
    div_out = v_old_1 + v_old_2;
    Eigen::MatrixXd u = f - div_out;

    out = u.array() - u.mean() + f.mean();
}

void tvdff_color(const Eigen::Ref<Eigen::MatrixXd> &f_r, const Eigen::Ref<Eigen::MatrixXd> &f_g, const Eigen::Ref<Eigen::MatrixXd> &f_b, Eigen::Ref<Eigen::MatrixXd> out_r, Eigen::Ref<Eigen::MatrixXd> out_g, Eigen::Ref<Eigen::MatrixXd> out_b, int n, int m, double lmd, double tol, int iters)
{
    Eigen::MatrixXd v_old_1_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_old_2_r = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_old_1_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_old_2_g = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_old_1_b = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_old_2_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_hat_1_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_hat_2_r = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_hat_1_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_hat_2_g = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_hat_1_b = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_hat_2_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_1_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_2_r = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_1_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_2_g = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd v_1_b = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd v_2_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd d_1_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd d_2_r = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd d_1_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd d_2_g = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd d_1_b = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd d_2_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd div_out_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd div_out_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd div_out_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::ArrayXXd norm_denom = Eigen::MatrixXd::Zero(n, m);

    double told = 1.0;
    double t = 1.0;
    double dt = 1.0;

    for (int itr{0}; itr < iters; ++itr)
    {
        div(v_hat_1_r, v_hat_2_r, v_1_r, v_2_r, n, m);
        div_out_r = v_1_r + v_2_r;
        div_out_r.noalias() -= f_r;
        grad(div_out_r, v_1_r, v_2_r, n, m);

        div(v_hat_1_g, v_hat_2_g, v_1_g, v_2_g, n, m);
        div_out_g = v_1_g + v_2_g;
        div_out_g.noalias() -= f_g;
        grad(div_out_g, v_1_g, v_2_g, n, m);

        div(v_hat_1_b, v_hat_2_b, v_1_b, v_2_b, n, m);
        div_out_b = v_1_b + v_2_b;
        div_out_b.noalias() -= f_b;
        grad(div_out_b, v_1_b, v_2_b, n, m);

        v_1_r.noalias() = v_hat_1_r - (v_1_r / 6.0);
        v_2_r.noalias() = v_hat_2_r - (v_2_r / 6.0);

        v_1_g.noalias() = v_hat_1_g - (v_1_g / 6.0);
        v_2_g.noalias() = v_hat_2_g - (v_2_g / 6.0);

        v_1_b.noalias() = v_hat_1_b - (v_1_b / 6.0);
        v_2_b.noalias() = v_hat_2_b - (v_2_b / 6.0);

        norm_denom = ((
                          v_1_r.array().square() + v_2_r.array().square() +
                          v_1_g.array().square() + v_2_g.array().square() +
                          v_1_b.array().square() + v_2_b.array().square())
                          .sqrt())
                         .cwiseMax(lmd);

        v_1_r = (lmd * v_1_r.array()) / norm_denom;
        v_2_r = (lmd * v_2_r.array()) / norm_denom;

        v_1_g = (lmd * v_1_g.array()) / norm_denom;
        v_2_g = (lmd * v_2_g.array()) / norm_denom;

        v_1_b = (lmd * v_1_b.array()) / norm_denom;
        v_2_b = (lmd * v_2_b.array()) / norm_denom;

        d_1_r.noalias() = v_1_r - v_old_1_r;
        d_2_r.noalias() = v_2_r - v_old_2_r;

        d_1_g.noalias() = v_1_g - v_old_1_g;
        d_2_g.noalias() = v_2_g - v_old_2_g;

        d_1_b.noalias() = v_1_b - v_old_1_b;
        d_2_b.noalias() = v_2_b - v_old_2_b;

        if (itr % 5 == 0)
        {
            if ((d_1_r.array().abs()).maxCoeff() < tol && (d_2_r.array().abs()).maxCoeff() < tol &&
                (d_1_g.array().abs()).maxCoeff() < tol && (d_2_g.array().abs()).maxCoeff() < tol &&
                (d_1_b.array().abs()).maxCoeff() < tol && (d_2_b.array().abs()).maxCoeff() < tol)
            {
                break;
            }
        }

        t = (1.0 + std::sqrt(1.0 + 4.0 * told * told)) / 2.0;
        dt = (told - 1.0) / t;

        v_hat_1_r.noalias() = v_1_r + dt * d_1_r;
        v_hat_2_r.noalias() = v_2_r + dt * d_2_r;

        v_hat_1_g.noalias() = v_1_g + dt * d_1_g;
        v_hat_2_g.noalias() = v_2_g + dt * d_2_g;

        v_hat_1_b.noalias() = v_1_b + dt * d_1_b;
        v_hat_2_b.noalias() = v_2_b + dt * d_2_b;

        v_old_1_r.noalias() = v_1_r;
        v_old_2_r.noalias() = v_2_r;

        v_old_1_g.noalias() = v_1_g;
        v_old_2_g.noalias() = v_2_g;

        v_old_1_b.noalias() = v_1_b;
        v_old_2_b.noalias() = v_2_b;

        told = t;
    }

    // use v_old as the output since it is the last iteration and we don't need them anymore
    div(v_1_r, v_2_r, v_old_1_r, v_old_2_r, n, m);
    div_out_r = v_old_1_r + v_old_2_r;
    Eigen::MatrixXd u_r = f_r - div_out_r;

    div(v_1_g, v_2_g, v_old_1_g, v_old_2_g, n, m);
    div_out_g = v_old_1_g + v_old_2_g;
    Eigen::MatrixXd u_g = f_g - div_out_g;

    div(v_1_b, v_2_b, v_old_1_b, v_old_2_b, n, m);
    div_out_b = v_old_1_b + v_old_2_b;
    Eigen::MatrixXd u_b = f_b - div_out_b;

    out_r = u_r.array() - u_r.mean() + f_r.mean();
    out_g = u_g.array() - u_g.mean() + f_g.mean();
    out_b = u_b.array() - u_b.mean() + f_b.mean();
}

Eigen::VectorXd run_TV_flow(const Eigen::Ref<Eigen::MatrixXd> &f, int n, int m, int NOB, double lami, double dt, double tol, int NIT)
{
    Eigen::VectorXd S(NOB);
    S.setZero();

    Eigen::MatrixXd u0 = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u1 = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u2 = Eigen::MatrixXd::Zero(n, m);

    u0 = f;
    tvdff(u0, u1, n, m, lami, tol, NIT);
    tvdff(u1, u2, n, m, lami, tol, NIT);

    Eigen::MatrixXd phi = (1.0 / dt) * (u0 - 2 * u1 + u2);
    S(0) = (phi.cwiseAbs()).sum();

    for (int i{1}; i < NOB; ++i)
    {
        u0.noalias() = u1;
        u1.noalias() = u2;
        tvdff(u1, u2, n, m, lami, tol, NIT);

        phi.noalias() = ((i + 1) / dt) * (u0 - 2 * u1 + u2);
        S(i) = (phi.cwiseAbs()).sum();
    }
    return S;
}

Eigen::MatrixX3d run_TV_flow_RGB(const Eigen::Ref<Eigen::MatrixXd> &r, const Eigen::Ref<Eigen::MatrixXd> &g, const Eigen::Ref<Eigen::MatrixXd> &b, int n, int m, int NOB, double lami, double dt, double tol, int NIT)
{
    Eigen::MatrixX3d S(NOB, 3);
    S.setZero();

    Eigen::MatrixXd u0_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u0_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u0_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd u1_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u1_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u1_b = Eigen::MatrixXd::Zero(n, m);

    Eigen::MatrixXd u2_r = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u2_g = Eigen::MatrixXd::Zero(n, m);
    Eigen::MatrixXd u2_b = Eigen::MatrixXd::Zero(n, m);

    u0_r = r;
    u0_g = g;
    u0_b = b;

    tvdff_color(u0_r, u0_g, u0_b, u1_r, u1_g, u1_b, n, m, lami, tol, NIT);
    tvdff_color(u1_r, u1_g, u1_b, u2_r, u2_g, u2_b, n, m, lami, tol, NIT);

    Eigen::MatrixXd phi_r = (1.0 / dt) * (u0_r - 2 * u1_r + u2_r);
    Eigen::MatrixXd phi_g = (1.0 / dt) * (u0_g - 2 * u1_g + u2_g);
    Eigen::MatrixXd phi_b = (1.0 / dt) * (u0_b - 2 * u1_b + u2_b);

    S(0, 0) = (phi_r.cwiseAbs()).sum();
    S(0, 1) = (phi_g.cwiseAbs()).sum();
    S(0, 2) = (phi_b.cwiseAbs()).sum();

    for (int i{1}; i < NOB; ++i)
    {
        u0_r.noalias() = u1_r;
        u1_r.noalias() = u2_r;

        u0_g.noalias() = u1_g;
        u1_g.noalias() = u2_g;

        u0_b.noalias() = u1_b;
        u1_b.noalias() = u2_b;

        tvdff_color(u1_r, u1_g, u1_b, u2_r, u2_g, u2_b, n, m, lami, tol, NIT);

        phi_r.noalias() = ((i + 1) / dt) * (u0_r - 2 * u1_r + u2_r);
        phi_g.noalias() = ((i + 1) / dt) * (u0_g - 2 * u1_g + u2_g);
        phi_b.noalias() = ((i + 1) / dt) * (u0_b - 2 * u1_b + u2_b);

        S(i, 0) = (phi_r.cwiseAbs()).sum();
        S(i, 1) = (phi_g.cwiseAbs()).sum();
        S(i, 2) = (phi_b.cwiseAbs()).sum();
    }
    return S;
}

template <typename T>
void saveData(std::string fileName, const T &v)
{
    // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << v.format(CSVFormat);
        file.close();
    }
}

Eigen::MatrixXd openData(std::string fileToOpen)
{
    std::vector<double> matrixEntries;

    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);

    // this variable is used to store the row of the matrix that contains commas
    std::string matrixRowString;

    // this variable is used to store the matrix entry;
    std::string matrixEntry;

    // this variable is used to track the number of rows
    int matrixRowNumber = 0;

    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString);     // convert matrixRowString that is a string to a stream variable.
        while (std::getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry)); // here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; // update the column numbers
    }

    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

template void saveData<Eigen::VectorXd>(std::string fileName, const Eigen::VectorXd &v);
template void saveData<Eigen::MatrixX3d>(std::string fileName, const Eigen::MatrixX3d &v);
