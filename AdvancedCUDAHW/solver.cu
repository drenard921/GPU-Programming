// solver.cu
//
// CUDA Dividend Portfolio Allocation Optimizer backend
//
// Usage:
//   ./solver_app input.csv results.csv
//
// Reads the input file written by gui.py, computes recommended
// allocation weights, simulates monthly portfolio growth with
// dividend reinvestment, and writes a time-series CSV.
//
// Libraries used:
//   - cuSOLVER: solve a dense linear system for portfolio weights
//   - cuBLAS: vector summaries and contribution-vector setup
//

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Stock {
    std::string ticker;
    float price_now = 0.0f;
    float cagr_3y = 0.0f;
    float cagr_5y = 0.0f;
    float dividend_yield = 0.0f;
    float current_holding = 0.0f;
};

struct InputData {
    float monthly_investment = 500.0f;
    int years = 10;
    std::string goal = "balanced";
    std::string growth_basis = "3y";
    std::vector<Stock> stocks;
};

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::ostringstream oss__;                                         \
            oss__ << "CUDA error: " << cudaGetErrorString(err__)              \
                  << " at " << __FILE__ << ":" << __LINE__;                   \
            throw std::runtime_error(oss__.str());                            \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status__ = (call);                                     \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                              \
            std::ostringstream oss__;                                         \
            oss__ << "cuBLAS error at " << __FILE__ << ":" << __LINE__        \
                  << " status=" << static_cast<int>(status__);                \
            throw std::runtime_error(oss__.str());                            \
        }                                                                     \
    } while (0)

#define CUSOLVER_CHECK(call)                                                  \
    do {                                                                      \
        cusolverStatus_t status__ = (call);                                   \
        if (status__ != CUSOLVER_STATUS_SUCCESS) {                            \
            std::ostringstream oss__;                                         \
            oss__ << "cuSOLVER error at " << __FILE__ << ":" << __LINE__      \
                  << " status=" << static_cast<int>(status__);                \
            throw std::runtime_error(oss__.str());                            \
        }                                                                     \
    } while (0)

static std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(trim(item));
    }
    return out;
}

static float active_growth_rate(const Stock& stock, const std::string& basis) {
    if (basis == "5y") {
        return stock.cagr_5y;
    }
    return stock.cagr_3y;
}

static InputData read_input_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + path);
    }

    InputData data;
    std::string line;
    bool in_stock_table = false;

    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        auto cols = split_csv_line(line);
        if (cols.empty()) {
            continue;
        }

        if (!in_stock_table) {
            if (cols.size() >= 2 && cols[0] == "monthly_investment") {
                data.monthly_investment = std::stof(cols[1]);
            } else if (cols.size() >= 2 && cols[0] == "years") {
                data.years = std::stoi(cols[1]);
            } else if (cols.size() >= 2 && cols[0] == "goal") {
                data.goal = cols[1];
            } else if (cols.size() >= 2 && cols[0] == "growth_basis") {
                data.growth_basis = cols[1];
            } else if (cols.size() >= 6 &&
                       cols[0] == "ticker" &&
                       cols[1] == "price_now" &&
                       cols[2] == "cagr_3y" &&
                       cols[3] == "cagr_5y" &&
                       cols[4] == "dividend_yield" &&
                       cols[5] == "current_holding") {
                in_stock_table = true;
            }
        } else {
            if (cols.size() < 6) {
                continue;
            }

            Stock s;
            s.ticker = cols[0];
            s.price_now = std::stof(cols[1]);
            s.cagr_3y = std::stof(cols[2]);
            s.cagr_5y = std::stof(cols[3]);
            s.dividend_yield = std::stof(cols[4]);
            s.current_holding = std::stof(cols[5]);
            data.stocks.push_back(s);
        }
    }

    if (data.stocks.size() < 2) {
        throw std::runtime_error("Need at least 2 stocks in input.csv");
    }
    if (data.years <= 0) {
        throw std::runtime_error("Years must be positive");
    }
    if (data.monthly_investment <= 0.0f) {
        throw std::runtime_error("Monthly investment must be positive");
    }
    if (data.growth_basis != "3y" && data.growth_basis != "5y") {
        throw std::runtime_error("growth_basis must be '3y' or '5y'");
    }

    return data;
}

static std::vector<float> build_score_vector(
    const std::vector<Stock>& stocks,
    const std::string& goal,
    const std::string& growth_basis
) {
    std::vector<float> scores(stocks.size(), 0.0f);

    for (size_t i = 0; i < stocks.size(); ++i) {
        const float g = active_growth_rate(stocks[i], growth_basis);
        const float d = stocks[i].dividend_yield;

        if (goal == "growth") {
            scores[i] = g;
        } else if (goal == "income") {
            scores[i] = d;
        } else {
            scores[i] = 0.5f * g + 0.5f * d;
        }
    }

    float min_score = *std::min_element(scores.begin(), scores.end());
    if (min_score < 0.0f) {
        for (float& x : scores) {
            x -= min_score;
        }
    }

    for (float& x : scores) {
        x += 1.0e-3f;
    }

    return scores;
}

static std::vector<float> solve_weights_with_cusolver(
    const std::vector<float>& scores
) {
    const int n = static_cast<int>(scores.size());
    const float alpha = 0.25f;

    std::vector<float> h_A(n * n, 0.0f);
    std::vector<float> h_b = scores;

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            float val = alpha;
            if (row == col) {
                val += 1.0f;
            }
            h_A[col * n + row] = val;
        }
    }

    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    float* d_A = nullptr;
    float* d_b = nullptr;
    int* d_ipiv = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * n * n));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_b, h_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice
    ));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(solver, n, n, d_A, n, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * lwork));

    CUSOLVER_CHECK(cusolverDnSgetrf(
        solver, n, n, d_A, n, d_work, d_ipiv, d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        throw std::runtime_error(
            "LU factorization failed, info=" + std::to_string(info_h)
        );
    }

    CUSOLVER_CHECK(cusolverDnSgetrs(
        solver, CUBLAS_OP_N, n, 1, d_A, n, d_ipiv, d_b, n, d_info
    ));

    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        throw std::runtime_error(
            "Linear solve failed, info=" + std::to_string(info_h)
        );
    }

    std::vector<float> weights(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(
        weights.data(), d_b, sizeof(float) * n, cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));

    for (float& w : weights) {
        if (w < 0.0f) {
            w = 0.0f;
        }
    }

    float sum_w = 0.0f;
    for (float w : weights) {
        sum_w += w;
    }

    if (sum_w <= 0.0f) {
        float fallback = 1.0f / static_cast<float>(n);
        for (float& w : weights) {
            w = fallback;
        }
    } else {
        for (float& w : weights) {
            w /= sum_w;
        }
    }

    return weights;
}

__global__ void monthly_update_kernel(
    float* values,
    const float* monthly_return_rates,
    const float* contribution_vector,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        values[i] = values[i] * (1.0f + monthly_return_rates[i]) +
                    contribution_vector[i];
    }
}

static void write_results_csv(
    const std::string& path,
    const std::vector<float>& totals,
    const std::vector<float>& dividends
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    out << "month,total_value,total_dividend_income,monthly_dividend_income\n";
    for (size_t i = 0; i < totals.size(); ++i) {
        out << i << ","
            << std::fixed << std::setprecision(2)
            << totals[i] << ","
            << dividends[i] << ","
            << (dividends[i] / 12.0f) << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0]
                      << " input.csv results.csv\n";
            return 1;
        }

        const std::string input_path = argv[1];
        const std::string output_path = argv[2];

        InputData input = read_input_csv(input_path);
        const int n = static_cast<int>(input.stocks.size());
        const int months = input.years * 12;

        std::vector<float> scores = build_score_vector(
            input.stocks, input.goal, input.growth_basis
        );
        std::vector<float> weights = solve_weights_with_cusolver(scores);

        std::vector<float> h_initial_values(n, 0.0f);
        std::vector<float> h_monthly_return(n, 0.0f);
        std::vector<float> h_annual_dividend(n, 0.0f);
        std::vector<float> h_ones(n, 1.0f);
        std::vector<float> h_current_weights(n, 0.0f);

        float current_total = 0.0f;
        for (int i = 0; i < n; ++i) {
            h_initial_values[i] = input.stocks[i].current_holding;
            current_total += h_initial_values[i];
        }

        if (current_total > 0.0f) {
            for (int i = 0; i < n; ++i) {
                h_current_weights[i] = h_initial_values[i] / current_total;
            }
        }

        for (int i = 0; i < n; ++i) {
            const float annual_growth =
                active_growth_rate(input.stocks[i], input.growth_basis);
            const float annual_div = input.stocks[i].dividend_yield;

            h_monthly_return[i] = (annual_growth + annual_div) / 12.0f;
            h_annual_dividend[i] = annual_div;
        }

        cublasHandle_t blas = nullptr;
        CUBLAS_CHECK(cublasCreate(&blas));

        float* d_values = nullptr;
        float* d_monthly_return = nullptr;
        float* d_contrib = nullptr;
        float* d_weights = nullptr;
        float* d_ones = nullptr;
        float* d_annual_dividend = nullptr;

        CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&d_monthly_return, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&d_contrib, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&d_weights, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&d_annual_dividend, sizeof(float) * n));

        CUDA_CHECK(cudaMemcpy(
            d_values, h_initial_values.data(), sizeof(float) * n,
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_monthly_return, h_monthly_return.data(), sizeof(float) * n,
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_weights, weights.data(), sizeof(float) * n,
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_ones, h_ones.data(), sizeof(float) * n,
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_annual_dividend, h_annual_dividend.data(), sizeof(float) * n,
            cudaMemcpyHostToDevice
        ));

        CUBLAS_CHECK(cublasScopy(blas, n, d_weights, 1, d_contrib, 1));
        CUBLAS_CHECK(cublasSscal(
            blas, n, &input.monthly_investment, d_contrib, 1
        ));

        std::vector<float> total_value_series(months + 1, 0.0f);
        std::vector<float> annual_div_income_series(months + 1, 0.0f);

        total_value_series[0] = current_total;

        float initial_div_income = 0.0f;
        for (int i = 0; i < n; ++i) {
            initial_div_income += h_initial_values[i] * h_annual_dividend[i];
        }
        annual_div_income_series[0] = initial_div_income;
        float initial_monthly_div_income = initial_div_income / 12.0f;

        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;

        for (int month = 1; month <= months; ++month) {
            monthly_update_kernel<<<grid_size, block_size>>>(
                d_values, d_monthly_return, d_contrib, n
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            float total_value = 0.0f;
            float annual_div_income = 0.0f;

            CUBLAS_CHECK(cublasSdot(
                blas, n, d_values, 1, d_ones, 1, &total_value
            ));
            CUBLAS_CHECK(cublasSdot(
                blas, n, d_values, 1, d_annual_dividend, 1, &annual_div_income
            ));

            total_value_series[month] = total_value;
            annual_div_income_series[month] = annual_div_income;
        }

        write_results_csv(
            output_path,
            total_value_series,
            annual_div_income_series
        );

        float final_annual_div_income = annual_div_income_series.back();
        float final_monthly_div_income = final_annual_div_income / 12.0f;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Goal: " << input.goal << "\n";
        std::cout << "Growth basis: " << input.growth_basis << "\n";
        std::cout << "Monthly investment: $" << input.monthly_investment << "\n";
        std::cout << "Duration: " << input.years << " years\n";
        std::cout << "Current portfolio value: $" << current_total << "\n\n";

        std::cout << "Current vs Recommended allocations:\n";
        for (int i = 0; i < n; ++i) {
            const float active_growth =
                active_growth_rate(input.stocks[i], input.growth_basis);

            std::cout << "  " << input.stocks[i].ticker
                      << " | current: " << (h_current_weights[i] * 100.0f)
                      << "% | recommended: " << (weights[i] * 100.0f)
                      << "% | growth: " << (active_growth * 100.0f)
                      << "% | dividend: "
                      << (input.stocks[i].dividend_yield * 100.0f)
                      << "%\n";
        }

        std::cout << "\nIncome comparison:\n";
        std::cout << "  Current monthly dividend income: $"
                  << initial_monthly_div_income << "\n";
        std::cout << "  Current annual dividend income:  $"
                  << initial_div_income << "\n";
        std::cout << "  Projected monthly dividend income at end: $"
                  << final_monthly_div_income << "\n";
        std::cout << "  Projected annual dividend income at end:  $"
                  << final_annual_div_income << "\n";

        std::cout << "\nProjected final value: $"
                  << total_value_series.back() << "\n";
        std::cout << "Wrote results to " << output_path << "\n";

        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_monthly_return));
        CUDA_CHECK(cudaFree(d_contrib));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_ones));
        CUDA_CHECK(cudaFree(d_annual_dividend));
        CUBLAS_CHECK(cublasDestroy(blas));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "solver_app failed: " << e.what() << "\n";
        return 1;
    }
}