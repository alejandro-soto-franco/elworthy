//! Runtime CPU-feature detection.
//!
//! Used by the SIMD backend to decide which vector width to dispatch.
//! The scalar and F64X2 paths are always available on any x86-64 or
//! aarch64 target; F64X4 requires AVX2 on x86-64.

/// Return `true` if the running CPU supports AVX2.
///
/// On non-x86 targets this always returns `false` so the F64X4 backend
/// stays inactive. On x86-64, consults `std::arch::is_x86_feature_detected`.
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Return the preferred SIMD lane count for `f64` on this host.
///
/// Logic: AVX2 available -> 4 lanes; otherwise 2 lanes (portable SSE2).
pub fn preferred_f64_lanes() -> usize {
    if has_avx2() {
        4
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_count_matches_flag() {
        let lanes = preferred_f64_lanes();
        if has_avx2() {
            assert_eq!(lanes, 4);
        } else {
            assert_eq!(lanes, 2);
        }
    }
}
