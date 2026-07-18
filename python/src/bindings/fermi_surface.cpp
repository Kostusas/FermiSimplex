#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/fermi_surface.h>

#include <cstdint>
#include <vector>

namespace fermisimplex::bindings {

void bind_fermi_surface(nb::module_ &module) {
    nb::class_<FermiSurfaceStats>(module, "FermiSurfaceStats")
        .def_ro("evaluations", &FermiSurfaceStats::evaluations)
        .def_ro(
            "terminal_visible_simplices",
            &FermiSurfaceStats::terminal_visible_simplices
        )
        .def_ro(
            "terminal_inconclusive_simplices",
            &FermiSurfaceStats::terminal_inconclusive_simplices
        );

    nb::class_<FermiSurfaceResult>(module, "FermiSurfaceResult")
        .def_prop_ro(
            "points",
            [](const FermiSurfaceResult &result) {
                return make_array(
                    std::vector<double>(result.points),
                    {result.points.size() / result.ndim, result.ndim}
                );
            },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "cells",
            [](const FermiSurfaceResult &result) {
                return make_array(
                    std::vector<std::int64_t>(result.cells),
                    {result.cells.size() / result.ndim, result.ndim}
                );
            },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "cell_bands",
            [](const FermiSurfaceResult &result) {
                return make_array(
                    std::vector<std::int64_t>(result.cell_bands),
                    {result.cell_bands.size()}
                );
            },
            nb::rv_policy::move
        )
        .def_ro("completed", &FermiSurfaceResult::completed)
        .def_ro(
            "coverage_certified",
            &FermiSurfaceResult::coverage_certified
        )
        .def_prop_ro(
            "stats",
            [](const FermiSurfaceResult &result) { return result.stats; }
        );
}

}  // namespace fermisimplex::bindings
