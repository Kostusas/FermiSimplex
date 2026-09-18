#include "evaluated_snapshot.h"
#include "arrays.h"

namespace fermisimplex::bindings {

nb::dict evaluated_snapshot_arrays(const SpectralMesh &mesh, bool include_eigenvectors) {
    auto snapshot = mesh.evaluated_snapshot(include_eigenvectors);
    const auto vertices = snapshot.vertex_ids.size();
    const auto cells = snapshot.simplex_ids.size();
    const auto dimension = snapshot.ndim;
    const auto bands = snapshot.ndof;
    auto result = nb::dict{};
    result["ndim"] = dimension;
    result["ndof"] = bands;
    result["active_vertices"] = snapshot.active_vertices;
    result["active_simplices"] = snapshot.active_simplices;
    result["vertex_ids"] = make_array(std::move(snapshot.vertex_ids), {vertices});
    result["points"] = make_array(std::move(snapshot.points), {vertices, dimension});
    result["dyadic_numerators"] =
        make_array(std::move(snapshot.dyadic_numerators), {vertices, dimension});
    result["dyadic_levels"] = make_array(std::move(snapshot.dyadic_levels), {vertices});
    result["eigenvalues"] = make_array(std::move(snapshot.eigenvalues), {vertices, bands});
    if (snapshot.eigenvectors) {
        result["eigenvectors"] =
            make_array(std::move(*snapshot.eigenvectors), {vertices, bands, bands});
    } else {
        result["eigenvectors"] = nb::none();
    }
    result["vertex_is_active"] = make_array(std::move(snapshot.vertex_is_active), {vertices});
    result["simplex_ids"] = make_array(std::move(snapshot.simplex_ids), {cells});
    result["active_ancestor_ids"] = make_array(std::move(snapshot.active_ancestor_ids), {cells});
    result["simplices"] = make_array(std::move(snapshot.simplices), {cells, dimension + 1});
    result["volumes"] = make_array(std::move(snapshot.volumes), {cells});
    return result;
}

}  // namespace fermisimplex::bindings
