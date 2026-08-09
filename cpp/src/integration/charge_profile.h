#pragma once

#include <fermisimplex/integration.h>

#include <adaptivesimplex/adaptive/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fermisimplex::integration_detail {

// Benchmark-only timing data. The top-level stages are mutually exclusive;
// linear_charge_seconds and error_estimation_seconds are nested details of the
// charge-and-error stage and must not be added to the top-level total.
struct ChargeProfile {
    double total_seconds = 0.0;
    double vertex_cache_seconds = 0.0;
    double root_certification_seconds = 0.0;
    double charge_and_error_seconds = 0.0;
    double linear_charge_seconds = 0.0;
    double error_estimation_seconds = 0.0;

    // Mutually exclusive diagnostic details nested inside
    // error_estimation_seconds.
    double error_hamiltonian_seconds = 0.0;
    double error_full_eigensystem_seconds = 0.0;
    double error_reduced_eigensystem_seconds = 0.0;
    double error_norm_eigensystem_seconds = 0.0;
    double error_schur_setup_seconds = 0.0;
    double error_schur_application_seconds = 0.0;
    double error_certification_seconds = 0.0;
    double error_subdivision_seconds = 0.0;
    double error_defect_assembly_seconds = 0.0;
    double error_defect_norm_overhead_seconds = 0.0;
    double error_band_comparison_seconds = 0.0;
    double error_occupied_volume_seconds = 0.0;

    // Subregions and calls below explain the exclusive regions above; they
    // overlap their parent and must not be added to error_estimation_seconds.
    double error_hamiltonian_scalar_seconds = 0.0;
    double error_full_vector_seconds = 0.0;
    double error_full_values_seconds = 0.0;
    double error_certification_prepare_seconds = 0.0;
    double error_certification_apply_seconds = 0.0;
    double error_projected_schur_setup_seconds = 0.0;
    double error_general_schur_setup_seconds = 0.0;
    double error_projected_schur_application_seconds = 0.0;
    double error_general_schur_application_seconds = 0.0;
    std::uint64_t error_projected_schur_setup_calls = 0;
    std::uint64_t error_general_schur_setup_calls = 0;
    std::uint64_t error_projected_schur_evaluations = 0;
    std::uint64_t error_cache_live_records = 0;
    std::uint64_t error_cache_peak_records = 0;
    std::uint64_t error_cache_live_payload_bytes = 0;
    std::uint64_t error_cache_peak_payload_bytes = 0;
    std::uint64_t error_root_caches_created = 0;
    std::uint64_t error_root_caches_destroyed = 0;
    std::uint64_t error_values_to_vectors_upgrades = 0;
    std::uint64_t error_certification_calls = 0;
    std::uint64_t error_subdivision_calls = 0;
    std::uint64_t error_hamiltonian_scalar_calls = 0;
    std::uint64_t error_full_vector_calls = 0;
    std::uint64_t error_full_values_calls = 0;
    std::uint64_t error_certification_direct_calls = 0;
    std::uint64_t error_certification_prepared_calls = 0;
    std::uint64_t error_certification_reused_calls = 0;
    std::uint64_t error_defect_samples = 0;
    std::uint64_t error_band_comparisons = 0;
    std::uint64_t error_occupied_volume_calls = 0;
    std::uint64_t error_matrix_requests = 0;
    std::uint64_t error_matrix_hits = 0;
    std::uint64_t error_effective_matrix_requests = 0;
    std::uint64_t error_effective_matrix_hits = 0;
    std::uint64_t error_spectrum_requests = 0;
    std::uint64_t error_spectrum_hits = 0;
    std::uint64_t error_effective_spectrum_requests = 0;
    std::uint64_t error_effective_spectrum_hits = 0;
    std::uint64_t error_eigenvalue_requests = 0;
    std::uint64_t error_eigenvalue_hits = 0;
    std::vector<std::uint64_t> error_reduced_dimension_counts;
    std::vector<std::uint64_t> error_certification_dimension_counts;
    std::vector<double> error_certification_dimension_seconds;
    std::vector<double> error_certification_radius_sums;
    std::vector<double> error_certification_radius_maxima;
    std::vector<std::uint64_t> error_certification_result_active_counts;
    std::vector<std::uint64_t> error_norm_dimension_counts;
    std::vector<double> error_norm_dimension_seconds;
    std::vector<std::uint64_t> error_terminal_dimension_counts;
    std::vector<std::uint64_t> error_visit_depth_counts;
    std::vector<std::uint64_t> error_terminal_depth_counts;
    std::vector<std::uint64_t> error_subdivision_depth_counts;

    double auxiliary_seconds() const noexcept {
        return std::max(
            0.0,
            total_seconds - vertex_cache_seconds -
                root_certification_seconds - charge_and_error_seconds
        );
    }

    double charge_overhead_seconds() const noexcept {
        return std::max(
            0.0,
            charge_and_error_seconds - linear_charge_seconds -
                error_estimation_seconds
        );
    }

    double error_other_seconds() const noexcept {
        return std::max(
            0.0,
            error_estimation_seconds - error_hamiltonian_seconds -
                error_full_eigensystem_seconds -
                error_reduced_eigensystem_seconds -
                error_norm_eigensystem_seconds -
                error_schur_setup_seconds -
                error_schur_application_seconds -
                error_certification_seconds -
                error_subdivision_seconds -
                error_defect_assembly_seconds -
                error_defect_norm_overhead_seconds -
                error_band_comparison_seconds -
                error_occupied_volume_seconds
        );
    }

    void record_reduced_dimension(std::size_t dimension) {
        if (error_reduced_dimension_counts.size() <= dimension) {
            error_reduced_dimension_counts.resize(dimension + 1, 0);
        }
        ++error_reduced_dimension_counts[dimension];
    }

    static void increment_at(
        std::vector<std::uint64_t> &values,
        std::size_t index
    ) {
        if (values.size() <= index) {
            values.resize(index + 1, 0);
        }
        ++values[index];
    }

    static void add_at(
        std::vector<double> &values,
        std::size_t index,
        double seconds
    ) {
        if (values.size() <= index) {
            values.resize(index + 1, 0.0);
        }
        values[index] += seconds;
    }

    void record_certification(
        std::size_t dimension,
        std::size_t active,
        double radius,
        double seconds
    ) {
        increment_at(error_certification_dimension_counts, dimension);
        add_at(error_certification_dimension_seconds, dimension, seconds);
        add_at(error_certification_radius_sums, dimension, radius);
        if (error_certification_radius_maxima.size() <= dimension) {
            error_certification_radius_maxima.resize(dimension + 1, 0.0);
        }
        error_certification_radius_maxima[dimension] = std::max(
            error_certification_radius_maxima[dimension], radius
        );
        increment_at(error_certification_result_active_counts, active);
    }

    void record_norm(std::size_t dimension, double seconds) {
        increment_at(error_norm_dimension_counts, dimension);
        add_at(error_norm_dimension_seconds, dimension, seconds);
    }
};

// Runs the same calculation as integrate_charge while collecting stage times.
// Kept internal so instrumentation does not become part of the public API.
ChargeResult integrate_charge_profiled(
    SpectralMesh &mesh,
    double mu,
    const adaptivesimplex::adaptive::Options &options,
    std::uint32_t error_depth,
    ChargeProfile &profile
);

}  // namespace fermisimplex::integration_detail
