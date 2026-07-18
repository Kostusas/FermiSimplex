#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/fermi_surface.h>
#include <fermisimplex/hamiltonian.h>
#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fermisimplex::bindings {
namespace adaptive = adaptivesimplex::adaptive;

namespace {

class PythonHamiltonian final : public HamiltonianModel {
public:
    PythonHamiltonian(
        nb::object callable,
        std::size_t ndim,
        std::size_t ndof
    ) : callable_(std::move(callable)),
        ndim_(ndim),
        ndof_(ndof) {
        if (ndim_ == 0 || ndof_ == 0) {
            throw std::runtime_error("Hamiltonian dimensions must be positive");
        }
    }

    std::size_t ndim() const noexcept override { return ndim_; }
    std::size_t ndof() const noexcept override { return ndof_; }

    std::vector<std::complex<double>> evaluate(
        std::span<const double> reduced_point
    ) const override {
        if (reduced_point.size() != ndim_) {
            throw std::runtime_error("Hamiltonian point dimension mismatch");
        }

        nb::gil_scoped_acquire gil;
        if (!callable_) {
            throw std::runtime_error("Hamiltonian callable is no longer available");
        }
        auto point = make_array(
            std::vector<double>(reduced_point.begin(), reduced_point.end()),
            {ndim_}
        );
        const auto matrix = nb::cast<CallbackMatrixArray>(callable_(point));
        if (matrix.shape(0) != ndof_ || matrix.shape(1) != ndof_) {
            throw std::runtime_error(
                "Hamiltonian callable returned a matrix with inconsistent shape"
            );
        }

        std::vector<std::complex<double>> result(ndof_ * ndof_);
        for (std::size_t row = 0; row < ndof_; ++row) {
            for (std::size_t column = 0; column < ndof_; ++column) {
                result[column * ndof_ + row] = matrix.data()[row * ndof_ + column];
            }
        }
        return result;
    }

    nb::handle callable() const noexcept { return callable_; }
    void clear_callable() const { callable_.reset(); }

private:
    mutable nb::object callable_;
    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
};

std::shared_ptr<const HamiltonianModel> python_hamiltonian(
    nb::object callable,
    std::size_t ndim,
    std::size_t ndof
) {
    return std::make_shared<PythonHamiltonian>(
        std::move(callable),
        ndim,
        ndof
    );
}

int spectral_mesh_tp_traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self));

    if (!nb::inst_ready(self)) {
        return 0;
    }

    const auto *mesh = nb::inst_ptr<SpectralMesh>(self);
    const auto *model = dynamic_cast<const PythonHamiltonian *>(&mesh->model());
    if (model != nullptr) {
        Py_VISIT(model->callable().ptr());
    }
    return 0;
}

int spectral_mesh_tp_clear(PyObject *self) {
    if (!nb::inst_ready(self)) {
        return 0;
    }

    const auto *mesh = nb::inst_ptr<SpectralMesh>(self);
    const auto *model = dynamic_cast<const PythonHamiltonian *>(&mesh->model());
    if (model != nullptr) {
        model->clear_callable();
    }
    return 0;
}

PyType_Slot spectral_mesh_slots[] = {
    {Py_tp_traverse, reinterpret_cast<void *>(spectral_mesh_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(spectral_mesh_tp_clear)},
    {0, nullptr},
};

}  // namespace

void bind_spectral_mesh(nb::module_ &module) {
    using namespace nb::literals;

    nb::class_<SpectralMesh>(
        module,
        "SpectralMesh",
        nb::type_slots(spectral_mesh_slots)
    )
        .def(
            "__init__",
            [](SpectralMesh *self,
               std::shared_ptr<TightBindingModel> model,
               double tolerance,
               std::uint32_t root_level) {
                new (self) SpectralMesh(
                    std::static_pointer_cast<const HamiltonianModel>(std::move(model)),
                    tolerance,
                    root_level
                );
            },
            "model"_a,
            "tolerance"_a = 1e-14,
            "root_level"_a = 1
        )
        .def(
            "__init__",
            [](SpectralMesh *self,
               nb::object callable,
               std::size_t ndim,
               std::size_t ndof,
               double tolerance,
               std::uint32_t root_level) {
                new (self) SpectralMesh(
                    python_hamiltonian(
                        std::move(callable),
                        ndim,
                        ndof
                    ),
                    tolerance,
                    root_level
                );
            },
            "callable"_a,
            "ndim"_a,
            "ndof"_a,
            "tolerance"_a = 1e-14,
            "root_level"_a = 1
        )
        .def_prop_ro("ndim", &SpectralMesh::ndim)
        .def_prop_ro("ndof", &SpectralMesh::ndof)
        .def_prop_ro("tolerance", &SpectralMesh::tolerance)
        .def_prop_ro("cached_vertices", &SpectralMesh::cached_vertices)
        .def_prop_ro("active_simplices", &SpectralMesh::active_simplices)
        .def_prop_ro("active_vertices", &SpectralMesh::active_vertices)
        .def(
            "integrate_charge",
            [](SpectralMesh &mesh,
               double mu,
               const adaptive::Options &options,
               double curvature_bound) {
                return fermisimplex::integrate_charge(
                    mesh,
                    mu,
                    options,
                    curvature_bound
                );
            },
            "mu"_a,
            "options"_a,
            "curvature_bound"_a = 0.0,
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def(
            "estimate_charge_on_current_mesh",
            [](SpectralMesh &mesh,
               double mu,
               double target_error,
               std::uint32_t preview_depth,
               double curvature_bound) {
                return fermisimplex::estimate_charge_on_current_mesh(
                    mesh,
                    mu,
                    target_error,
                    preview_depth,
                    curvature_bound
                );
            },
            "mu"_a,
            "target_error"_a,
            "preview_depth"_a = 1,
            "curvature_bound"_a = 0.0,
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def(
            "integrate_density_matrix",
            [](SpectralMesh &mesh,
               double mu,
               LatticeVectorArray lattice_vectors,
               const adaptive::Options &options) {
                return fermisimplex::integrate_density_matrix(
                    mesh,
                    mu,
                    copy_lattice_vectors(lattice_vectors),
                    options
                );
            },
            "mu"_a,
            "lattice_vectors"_a,
            "options"_a,
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def(
            "fermi_surface",
            [](SpectralMesh &mesh,
               double mu,
               double min_feature_size,
               std::optional<std::int64_t> max_evaluations,
               double curvature_bound) {
                return fermisimplex::fermi_surface(
                    mesh,
                    mu,
                    min_feature_size,
                    max_evaluations.value_or(-1),
                    curvature_bound
                );
            },
            "mu"_a,
            "min_feature_size"_a,
            "max_evaluations"_a = nb::none(),
            "curvature_bound"_a = 0.0,
            nb::call_guard<nb::gil_scoped_release>()
        );
}

}  // namespace fermisimplex::bindings
