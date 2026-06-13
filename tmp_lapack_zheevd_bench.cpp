#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

extern "C" {
void zheevd_(
    const char *jobz,
    const char *uplo,
    const int *n,
    std::complex<double> *a,
    const int *lda,
    double *w,
    std::complex<double> *work,
    const int *lwork,
    double *rwork,
    const int *lrwork,
    int *iwork,
    const int *liwork,
    int *info
);
}

namespace {

using clock_type = std::chrono::steady_clock;

double elapsed_seconds(clock_type::time_point start, clock_type::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::vector<std::complex<double>> random_hermitian(int n, std::mt19937_64 &rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<std::complex<double>> matrix(static_cast<size_t>(n) * n);
    for (int col = 0; col < n; ++col) {
        for (int row = col; row < n; ++row) {
            if (row == col) {
                matrix[static_cast<size_t>(col) * n + row] = normal(rng);
            } else {
                const std::complex<double> value(normal(rng), normal(rng));
                matrix[static_cast<size_t>(col) * n + row] = value;
                matrix[static_cast<size_t>(row) * n + col] = std::conj(value);
            }
        }
    }
    return matrix;
}

struct Workspace {
    std::vector<std::complex<double>> work;
    std::vector<double> rwork;
    std::vector<int> iwork;
};

Workspace query_workspace(int n) {
    const char jobz = 'V';
    const char uplo = 'L';
    const int lda = std::max(1, n);
    int info = 0;
    int lwork = -1;
    int lrwork = -1;
    int liwork = -1;
    std::vector<std::complex<double>> matrix(static_cast<size_t>(n) * n);
    std::vector<double> eigenvalues(n);
    std::complex<double> work_query = 0.0;
    double rwork_query = 0.0;
    int iwork_query = 0;

    zheevd_(
        &jobz,
        &uplo,
        &n,
        matrix.data(),
        &lda,
        eigenvalues.data(),
        &work_query,
        &lwork,
        &rwork_query,
        &lrwork,
        &iwork_query,
        &liwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error("zheevd workspace query failed");
    }

    lwork = std::max(1, static_cast<int>(std::real(work_query)));
    lrwork = std::max(1, static_cast<int>(rwork_query));
    liwork = std::max(1, iwork_query);
    return Workspace{
        .work = std::vector<std::complex<double>>(static_cast<size_t>(lwork)),
        .rwork = std::vector<double>(static_cast<size_t>(lrwork)),
        .iwork = std::vector<int>(static_cast<size_t>(liwork)),
    };
}

void diagonalize(
    std::vector<std::complex<double>> &matrix,
    std::vector<double> &eigenvalues,
    Workspace &workspace,
    int n
) {
    const char jobz = 'V';
    const char uplo = 'L';
    const int lda = std::max(1, n);
    int info = 0;
    int lwork = static_cast<int>(workspace.work.size());
    int lrwork = static_cast<int>(workspace.rwork.size());
    int liwork = static_cast<int>(workspace.iwork.size());
    zheevd_(
        &jobz,
        &uplo,
        &n,
        matrix.data(),
        &lda,
        eigenvalues.data(),
        workspace.work.data(),
        &lwork,
        workspace.rwork.data(),
        &lrwork,
        workspace.iwork.data(),
        &liwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error("zheevd failed");
    }
}

}  // namespace

int main(int argc, char **argv) {
    setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
    setenv("OMP_NUM_THREADS", "1", 0);

    const int n = argc > 1 ? std::atoi(argv[1]) : 288;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 100;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 5;

    std::mt19937_64 rng(12345);
    std::vector<std::vector<std::complex<double>>> inputs;
    inputs.reserve(static_cast<size_t>(repeats + warmup));
    for (int index = 0; index < repeats + warmup; ++index) {
        inputs.push_back(random_hermitian(n, rng));
    }

    auto workspace = query_workspace(n);
    std::vector<std::complex<double>> matrix(static_cast<size_t>(n) * n);
    std::vector<double> eigenvalues(n);

    for (int index = 0; index < warmup; ++index) {
        matrix = inputs[static_cast<size_t>(index)];
        diagonalize(matrix, eigenvalues, workspace, n);
    }

    double copy_seconds = 0.0;
    auto start = clock_type::now();
    for (int index = 0; index < repeats; ++index) {
        matrix = inputs[static_cast<size_t>(warmup + index)];
    }
    copy_seconds = elapsed_seconds(start, clock_type::now());

    start = clock_type::now();
    for (int index = 0; index < repeats; ++index) {
        matrix = inputs[static_cast<size_t>(warmup + index)];
        diagonalize(matrix, eigenvalues, workspace, n);
    }
    const double total_seconds = elapsed_seconds(start, clock_type::now());
    const double diagonalize_seconds = total_seconds - copy_seconds;

    start = clock_type::now();
    for (int index = 0; index < repeats; ++index) {
        auto local_workspace = query_workspace(n);
        matrix = inputs[static_cast<size_t>(warmup + index)];
        diagonalize(matrix, eigenvalues, local_workspace, n);
    }
    const double production_style_seconds = elapsed_seconds(start, clock_type::now());
    const double production_style_diagonalize_seconds =
        production_style_seconds - copy_seconds;

    std::cout << std::setprecision(9);
    std::cout << "n " << n << "\n";
    std::cout << "repeats " << repeats << "\n";
    std::cout << "warmup " << warmup << "\n";
    std::cout << "copy_total_seconds " << copy_seconds << "\n";
    std::cout << "copy_per_call_ms " << 1000.0 * copy_seconds / repeats << "\n";
    std::cout << "zheevd_total_seconds_including_copy " << total_seconds << "\n";
    std::cout << "zheevd_per_call_ms_including_copy " << 1000.0 * total_seconds / repeats << "\n";
    std::cout << "zheevd_per_call_ms_minus_copy " << 1000.0 * diagonalize_seconds / repeats << "\n";
    std::cout << "production_style_total_seconds_including_query_and_copy " << production_style_seconds << "\n";
    std::cout << "production_style_per_call_ms_including_query_and_copy " << 1000.0 * production_style_seconds / repeats << "\n";
    std::cout << "production_style_per_call_ms_minus_copy " << 1000.0 * production_style_diagonalize_seconds / repeats << "\n";
    std::cout << "workspace_lwork " << workspace.work.size() << "\n";
    std::cout << "workspace_lrwork " << workspace.rwork.size() << "\n";
    std::cout << "workspace_liwork " << workspace.iwork.size() << "\n";
}
