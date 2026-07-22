#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/fermi_surface.h>
#include <fermisimplex/hamiltonian.h>
#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include <nanobind/stl/optional.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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
        auto arguments = nb::steal<nb::tuple>(
            PyTuple_New(static_cast<Py_ssize_t>(ndim_))
        );
        if (!arguments) {
            throw nb::python_error();
        }
        for (std::size_t axis = 0; axis < ndim_; ++axis) {
            auto *coordinate = PyFloat_FromDouble(reduced_point[axis]);
            if (coordinate == nullptr) {
                throw nb::python_error();
            }
            PyTuple_SET_ITEM(arguments.ptr(), static_cast<Py_ssize_t>(axis), coordinate);
        }
        auto returned = nb::steal<nb::object>(
            PyObject_CallObject(callable_.ptr(), arguments.ptr())
        );
        if (!returned) {
            throw nb::python_error();
        }
        const auto matrix = nb::cast<CallbackMatrixArray>(returned);
        if (matrix.shape(0) != ndof_ || matrix.shape(1) != ndof_) {
            throw nb::value_error(
                ("Hamiltonian must return a matrix with shape (" +
                 std::to_string(ndof_) + ", " + std::to_string(ndof_) + ")").c_str()
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

std::vector<HoppingTerm> copy_hoppings(
    LatticeVectorArray lattice_vectors,
    HoppingMatrixArray matrices
) {
    const auto term_count = matrices.shape(0);
    const auto ndim = lattice_vectors.shape(1);
    const auto ndof = matrices.shape(1);
    std::vector<HoppingTerm> result;
    result.reserve(term_count);
    for (std::size_t term = 0; term < term_count; ++term) {
        auto hopping = HoppingTerm{};
        hopping.lattice_vector.assign(
            lattice_vectors.data() + term * ndim,
            lattice_vectors.data() + (term + 1) * ndim
        );
        hopping.matrix.resize(ndof * ndof);
        for (std::size_t row = 0; row < ndof; ++row) {
            for (std::size_t column = 0; column < ndof; ++column) {
                hopping.matrix[column * ndof + row] =
                    matrices.data()[(term * ndof + row) * ndof + column];
            }
        }
        result.push_back(std::move(hopping));
    }
    return result;
}

std::shared_ptr<const HamiltonianModel> tight_binding_hamiltonian(
    LatticeVectorArray lattice_vectors,
    HoppingMatrixArray matrices
) {
    if (lattice_vectors.shape(0) == 0 || lattice_vectors.shape(1) == 0) {
        throw std::runtime_error("tight-binding lattice vectors must be non-empty");
    }
    if (
        matrices.shape(0) != lattice_vectors.shape(0) ||
        matrices.shape(1) == 0 ||
        matrices.shape(1) != matrices.shape(2)
    ) {
        throw std::runtime_error("invalid tight-binding hopping-matrix shape");
    }
    return std::make_shared<TightBindingModel>(
        copy_hoppings(lattice_vectors, matrices)
    );
}

adaptive::Options adaptive_options(
    double target_error,
    std::int64_t max_refinements,
    std::uint32_t preview_depth,
    std::size_t min_refinement_batch_size,
    std::size_t max_refinement_batch_size
) {
    return adaptive::Options{
        .target_error = target_error,
        .max_refinements = max_refinements,
        .preview_depth = preview_depth,
        .min_refinement_batch_size = min_refinement_batch_size,
        .max_refinement_batch_size = max_refinement_batch_size,
    };
}

auto evaluate_hamiltonian(const SpectralMesh &mesh, PointArray point) {
    const auto matrix = mesh.hamiltonian(
        std::span<const double>(point.data(), point.shape(0))
    );
    std::vector<std::complex<double>> row_major(mesh.ndof() * mesh.ndof());
    for (std::size_t row = 0; row < mesh.ndof(); ++row) {
        for (std::size_t column = 0; column < mesh.ndof(); ++column) {
            row_major[row * mesh.ndof() + column] =
                matrix[column * mesh.ndof() + row];
        }
    }
    return make_array(std::move(row_major), {mesh.ndof(), mesh.ndof()});
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
               LatticeVectorArray lattice_vectors,
               HoppingMatrixArray matrices,
               double tolerance,
               std::uint32_t root_level) {
                new (self) SpectralMesh(
                    tight_binding_hamiltonian(lattice_vectors, matrices),
                    tolerance,
                    root_level
                );
            },
            "lattice_vectors"_a,
            "matrices"_a,
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
            "evaluate",
            &evaluate_hamiltonian,
            "point"_a
        )
        .def(
            "integrate_charge",
            [](SpectralMesh &mesh,
               double mu,
               double target_error,
               std::int64_t max_refinements,
               std::uint32_t preview_depth,
               std::size_t min_refinement_batch_size,
               std::size_t max_refinement_batch_size,
               double curvature_bound) {
                return fermisimplex::integrate_charge(
                    mesh,
                    mu,
                    adaptive_options(
                        target_error,
                        max_refinements,
                        preview_depth,
                        min_refinement_batch_size,
                        max_refinement_batch_size
                    ),
                    curvature_bound
                );
            },
            "mu"_a,
            "target_error"_a,
            "max_refinements"_a,
            "preview_depth"_a,
            "min_refinement_batch_size"_a,
            "max_refinement_batch_size"_a,
            "curvature_bound"_a,
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
               double target_error,
               std::int64_t max_refinements,
               std::uint32_t preview_depth,
               std::size_t min_refinement_batch_size,
               std::size_t max_refinement_batch_size) {
                return fermisimplex::integrate_density_matrix(
                    mesh,
                    mu,
                    copy_lattice_vectors(lattice_vectors),
                    adaptive_options(
                        target_error,
                        max_refinements,
                        preview_depth,
                        min_refinement_batch_size,
                        max_refinement_batch_size
                    )
                );
            },
            "mu"_a,
            "lattice_vectors"_a,
            "target_error"_a,
            "max_refinements"_a,
            "preview_depth"_a,
            "min_refinement_batch_size"_a,
            "max_refinement_batch_size"_a,
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
