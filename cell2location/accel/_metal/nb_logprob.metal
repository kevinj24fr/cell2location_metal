// Fused negative-binomial log-likelihood for the Metal backend.
//
// Why this exists
// ---------------
// The eager expression
//
//     L   = log(theta + mu + eps)
//     res = theta * (log(theta + eps) - L)
//         + value * (log(mu + eps)    - L)
//         + lgamma(value + theta) - lgamma(theta) - lgamma(value + 1)
//
// materialises roughly a dozen (n_obs x n_genes) intermediates. At a realistic
// Visium shape -- 4992 locations x 12000 genes -- each one is 240 MB, so a single
// evaluation moves several gigabytes through memory to produce 240 MB of answer.
// The arithmetic is trivial; the cost is entirely bandwidth.
//
// Fusing the whole expression into one pass reads value and mu once, reads the
// (usually tiny) theta vector, and writes the result. That is ~4 full-size accesses
// instead of ~24. On a unified-memory Mac, where GPU bandwidth is shared with
// everything else on the machine, this is the single biggest lever available.
//
// Why lgamma and digamma are reimplemented here
// ---------------------------------------------
// Two reasons. Metal Shading Library's lgamma is not the code path used by the
// PyTorch MPS backend, so relying on it would trade one unverified implementation
// for another. And digamma -- needed for the backward pass -- has no MSL equivalent
// at all. Both are built from the same shifted-asymptotic-series approach used by
// ``cell2location.accel._ops.lgamma_stirling``, so the Python reference
// implementation and this kernel agree by construction rather than by coincidence.

#include <metal_stdlib>
using namespace metal;

// Number of recurrence steps before the asymptotic expansion. The series converges
// rapidly for z >= 8; a compile-time constant keeps the loop unrolled and the
// instruction count fixed.
constant int SHIFT = 8;
constant float LOG_SQRT_2PI = 0.91893853320467274178f;

// log |Gamma(x)| for x > 0.
//
//   log G(x) = log G(x + n) - sum_{k<n} log(x + k)
//   log G(z) ~ (z - 1/2) log z - z + log(sqrt(2 pi))
//              + 1/(12z) - 1/(360 z^3) + 1/(1260 z^5) - 1/(1680 z^7)
//
// Truncation error below 1e-11 at z >= 8, far under float32 resolution. The error
// that survives is absolute (~1e-6) rather than relative, because the recurrence
// subtracts two similar magnitudes -- see the note in _ops.lgamma_stirling.
inline float lgamma_shifted(float x) {
    float shift = 0.0f;
    for (int k = 0; k < SHIFT; ++k) {
        shift += log(x + float(k));
    }

    float z    = x + float(SHIFT);
    float inv  = 1.0f / z;
    float inv2 = inv * inv;

    float series = inv * (1.0f / 12.0f
                 + inv2 * (-1.0f / 360.0f
                 + inv2 * (1.0f / 1260.0f
                 + inv2 * (-1.0f / 1680.0f))));

    return (z - 0.5f) * log(z) - z + LOG_SQRT_2PI + series - shift;
}

// psi(x), the derivative of log Gamma, for x > 0.
//
//   psi(x) = psi(x + n) - sum_{k<n} 1/(x + k)
//   psi(z) ~ log z - 1/(2z) - 1/(12 z^2) + 1/(120 z^4) - 1/(252 z^6) + 1/(240 z^8)
//
// Same shift, same accuracy regime. Differentiating lgamma_shifted analytically
// gives exactly this, which is what keeps forward and backward consistent.
inline float digamma_shifted(float x) {
    float shift = 0.0f;
    for (int k = 0; k < SHIFT; ++k) {
        shift += 1.0f / (x + float(k));
    }

    float z    = x + float(SHIFT);
    float inv  = 1.0f / z;
    float inv2 = inv * inv;

    float series = log(z) - 0.5f * inv
                 + inv2 * (-1.0f / 12.0f
                 + inv2 * (1.0f / 120.0f
                 + inv2 * (-1.0f / 252.0f
                 + inv2 * (1.0f / 240.0f))));

    return series - shift;
}

// theta is commonly a (1, n_genes) row broadcast against an (n_obs, n_genes) batch.
// Rather than materialising the broadcast -- which is both the memory cost we are
// trying to avoid and the exact shape that trips the stock MPS lgamma -- the index
// is computed inline.
inline uint theta_index(uint idx, uint n_cols, uint broadcast) {
    return broadcast != 0u ? (idx % n_cols) : idx;
}

kernel void nb_logprob_forward(
    device float*       out            [[buffer(0)]],
    device const float* value          [[buffer(1)]],
    device const float* mu             [[buffer(2)]],
    device const float* theta          [[buffer(3)]],
    constant uint&      n_total        [[buffer(4)]],
    constant uint&      n_cols         [[buffer(5)]],
    constant uint&      theta_bcast    [[buffer(6)]],
    constant float&     eps            [[buffer(7)]],
    uint                idx            [[thread_position_in_grid]])
{
    if (idx >= n_total) {
        return;
    }

    float v = value[idx];
    float m = mu[idx];
    float t = theta[theta_index(idx, n_cols, theta_bcast)];

    float log_theta_mu_eps = log(t + m + eps);

    out[idx] = t * (log(t + eps) - log_theta_mu_eps)
             + v * (log(m + eps) - log_theta_mu_eps)
             + lgamma_shifted(v + t)
             - lgamma_shifted(t)
             - lgamma_shifted(v + 1.0f);
}

// Gradients, derived from the forward expression:
//
//   d/dmu    = value/(mu + eps) - (theta + value)/(theta + mu + eps)
//   d/dtheta = log(theta + eps) + theta/(theta + eps) - log(theta + mu + eps)
//              - (theta + value)/(theta + mu + eps)
//              + psi(value + theta) - psi(theta)
//
// ``value`` is observed count data and never requires grad, so no term is emitted
// for it.
//
// grad_theta is written per element. When theta is broadcast, the caller reduces
// over rows with a single torch.sum -- cheaper than a threadgroup reduction here
// and it keeps this kernel free of synchronisation.
kernel void nb_logprob_backward(
    device float*       grad_mu        [[buffer(0)]],
    device float*       grad_theta     [[buffer(1)]],
    device const float* grad_out       [[buffer(2)]],
    device const float* value          [[buffer(3)]],
    device const float* mu             [[buffer(4)]],
    device const float* theta          [[buffer(5)]],
    constant uint&      n_total        [[buffer(6)]],
    constant uint&      n_cols         [[buffer(7)]],
    constant uint&      theta_bcast    [[buffer(8)]],
    constant float&     eps            [[buffer(9)]],
    uint                idx            [[thread_position_in_grid]])
{
    if (idx >= n_total) {
        return;
    }

    float g = grad_out[idx];
    float v = value[idx];
    float m = mu[idx];
    float t = theta[theta_index(idx, n_cols, theta_bcast)];

    float denom = t + m + eps;
    float ratio = (t + v) / denom;

    grad_mu[idx] = g * (v / (m + eps) - ratio);

    grad_theta[idx] = g * (log(t + eps)
                         + t / (t + eps)
                         - log(denom)
                         - ratio
                         + digamma_shifted(v + t)
                         - digamma_shifted(t));
}
