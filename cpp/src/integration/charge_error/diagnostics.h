#pragma once

#include "integration/charge_profile.h"

#include <array>
#include <chrono>
#include <cstddef>

namespace fermisimplex::integration_detail::charge_error_detail {

// Records one exclusive wall-time region. A null profile makes the timer a
// cheap no-op, keeping profiling branches out of the numerical algorithm.
class ProfileTimer {
public:
    using Field = double ChargeProfile::*;

    ProfileTimer(ChargeProfile *profile, Field first, Field second = nullptr)
        : profile_(profile),
        fields_{first, second},
        started_(profile == nullptr ? Clock::time_point{} : Clock::now()) {}

    ~ProfileTimer() {
        if (profile_ == nullptr) {
            return;
        }
        const auto seconds = std::chrono::duration<double>(
            Clock::now() - started_
        ).count();
        for (const auto field : fields_) {
            if (field != nullptr) {
                profile_->*field += seconds;
            }
        }
    }

    ProfileTimer(const ProfileTimer &) = delete;
    ProfileTimer &operator=(const ProfileTimer &) = delete;
    ProfileTimer(ProfileTimer &&) = delete;
    ProfileTimer &operator=(ProfileTimer &&) = delete;

private:
    using Clock = std::chrono::steady_clock;

    ChargeProfile *profile_ = nullptr;
    std::array<Field, 2> fields_{};
    Clock::time_point started_;
};

}  // namespace fermisimplex::integration_detail::charge_error_detail
