#include "certification/simplex/simplex_blocks.h"

namespace fermisimplex::certification::detail {

SimplexBlocks build_simplex_blocks(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const Complex>> eigenvectors,
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t nocc,
    size_t anchor_vertex_index
) {
    const auto nunocc = ndof - nocc;
    SimplexBlocks blocks;
    blocks.vertices.reserve(eigenvalues.size());
    blocks.average_coupling.assign(nunocc * nocc, Complex{0.0, 0.0});
    for (size_t vertex = 0; vertex < eigenvalues.size(); ++vertex) {
        if (vertex == anchor_vertex_index) {
            blocks.vertices.push_back(
                build_anchor_vertex_blocks(ndof, nocc, eigenvalues[vertex], mu)
            );
        } else {
            blocks.vertices.push_back(build_vertex_blocks(
                anchor_vectors,
                ndof,
                nocc,
                eigenvalues[vertex],
                eigenvectors[vertex],
                mu
            ));
        }
        for (size_t index = 0; index < blocks.average_coupling.size(); ++index) {
            blocks.average_coupling[index] += blocks.vertices.back().coupling_block[index];
        }
    }

    const auto average_scale = 1.0 / static_cast<double>(eigenvalues.size());
    for (auto &value : blocks.average_coupling) {
        value *= average_scale;
    }
    return blocks;
}

std::vector<Complex> perturbative_rotation(
    std::span<const Complex> average_coupling,
    std::span<const double> unoccupied_gaps,
    std::span<const double> occupied_gaps
) {
    const auto nunocc = unoccupied_gaps.size();
    const auto nocc = occupied_gaps.size();
    std::vector<Complex> rotation(nunocc * nocc, Complex{0.0, 0.0});
    for (size_t occ = 0; occ < nocc; ++occ) {
        for (size_t unocc = 0; unocc < nunocc; ++unocc) {
            rotation[column_major_index(unocc, occ, nunocc)] =
                average_coupling[column_major_index(unocc, occ, nunocc)] /
                (unoccupied_gaps[unocc] + occupied_gaps[occ]);
        }
    }
    return rotation;
}

}  // namespace fermisimplex::certification::detail
