"""
Cross-modal support: structural representations and similarity measures that do
not assume the two images share an intensity relationship.

Matching an optical frame against an infrared one breaks the assumption every
default matcher relies on — that corresponding points have comparable
brightness. A crater rim bright at 0.6 um can be dark at 4 um. CLAHE does not
help; it is a local contrast operator, not a bridge between modalities.

Two tools here address that:

`phase_congruency` reduces an image to a structural map built from Fourier
phase rather than amplitude, which is invariant to brightness and contrast. Both
images are converted to this common representation and the ordinary matching
stack then runs on top unchanged.

`mutual_information` scores alignment by how much knowing one patch tells you
about the other, which holds up when the intensity relationship is statistical
rather than functional. It replaces NCC during sub-pixel refinement.

References
----------
Kovesi, "Image Features from Phase Congruency", Videre 1(3), 1999.
Li et al., "RIFT: Multi-modal Image Matching Based on Radiation-variation
Insensitive Feature Transform", IEEE TIP 29, 2020.
"""

import numpy as np


def _lowpass_filter(shape, cutoff, order):
    """Butterworth low-pass in the frequency domain, origin at the corners."""
    rows, cols = shape
    y, x = np.meshgrid(
        (np.arange(rows) - rows // 2) / rows,
        (np.arange(cols) - cols // 2) / cols,
        indexing="ij",
    )
    radius = np.sqrt(x**2 + y**2)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * order)))


def phase_congruency(
    image,
    nscale=4,
    norient=6,
    min_wavelength=3.0,
    mult=2.1,
    sigma_onf=0.55,
    k=2.0,
    cut_off=0.5,
    g=10.0,
):
    """
    Kovesi phase congruency: a dimensionless 0-1 measure of feature significance.

    Points where the Fourier components of the image are maximally in phase are
    perceived as features — edges, lines, corners. Because the measure depends
    on phase alignment and not on how large the components are, it survives the
    contrast and brightness differences that separate two imaging modalities.

    Returns a float array in [0, 1] the same size as the input.
    """
    image = image.astype(np.float64)
    rows, cols = image.shape
    image_fft = np.fft.fft2(image)

    # Frequency-domain coordinates, with the origin moved to the corners so the
    # filters line up with the FFT layout.
    y, x = np.meshgrid(
        (np.arange(rows) - rows // 2) / rows,
        (np.arange(cols) - cols // 2) / cols,
        indexing="ij",
    )
    radius = np.fft.ifftshift(np.sqrt(x**2 + y**2))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    radius[0, 0] = 1.0  # keep log(radius) finite at DC

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lowpass = _lowpass_filter((rows, cols), 0.45, 15)

    # Radial log-Gabor bank: one filter per scale, geometrically spaced.
    log_gabor = []
    for scale in range(nscale):
        wavelength = min_wavelength * (mult**scale)
        f0 = 1.0 / wavelength
        radial = np.exp(-((np.log(radius / f0)) ** 2) / (2 * np.log(sigma_onf) ** 2))
        radial = radial * lowpass
        radial[0, 0] = 0.0
        log_gabor.append(radial)

    total_energy = np.zeros((rows, cols))
    total_sum_an = np.zeros((rows, cols))
    epsilon = 1e-4

    for orientation in range(norient):
        angle = orientation * np.pi / norient

        # Angular spread, cosine-tapered and wrapped so opposite directions of
        # the same orientation share a filter.
        delta_sin = sin_theta * np.cos(angle) - cos_theta * np.sin(angle)
        delta_cos = cos_theta * np.cos(angle) + sin_theta * np.sin(angle)
        d_theta = np.minimum(np.abs(np.arctan2(delta_sin, delta_cos)) * norient / 2.0, np.pi)
        spread = (np.cos(d_theta) + 1.0) / 2.0

        responses = []
        sum_an = np.zeros((rows, cols))
        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        max_an = np.zeros((rows, cols))

        filters = []
        for scale in range(nscale):
            filt = log_gabor[scale] * spread
            filters.append(filt)

            response = np.fft.ifft2(image_fft * filt)
            responses.append(response)

            amplitude = np.abs(response)
            sum_an += amplitude
            sum_e += response.real
            sum_o += response.imag
            max_an = amplitude if scale == 0 else np.maximum(max_an, amplitude)

        # Weighted mean phase direction across scales.
        norm = np.sqrt(sum_e**2 + sum_o**2) + epsilon
        mean_e = sum_e / norm
        mean_o = sum_o / norm

        energy = np.zeros((rows, cols))
        for response in responses:
            e, o = response.real, response.imag
            # Projection onto the mean phase, penalised by deviation from it.
            energy += e * mean_e + o * mean_o - np.abs(e * mean_o - o * mean_e)

        # Noise threshold. The smallest scale is dominated by noise, so its
        # median amplitude gives a Rayleigh parameter for the whole bank.
        filter_energy = np.sum(filters[0] ** 2)
        median_e2 = np.median(np.abs(responses[0]) ** 2)
        noise_power = (-median_e2 / np.log(0.5)) / max(filter_energy, epsilon)

        sum_an2 = sum(np.sum(f**2) for f in filters)
        sum_aiaj = 0.0
        for i in range(nscale - 1):
            for j in range(i + 1, nscale):
                sum_aiaj += np.sum(filters[i] * filters[j])

        noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_aiaj
        tau = np.sqrt(noise_energy2 / 2.0)
        noise_mean = tau * np.sqrt(np.pi / 2.0)
        noise_sigma = np.sqrt(max((2.0 - np.pi / 2.0), 0.0) * tau**2)
        threshold = max(noise_mean + k * noise_sigma, epsilon)

        energy = np.maximum(energy - threshold, 0.0)

        # Penalise points where only a narrow band of frequencies is active:
        # a single dominant scale is more likely noise than a real feature.
        width = (sum_an / (max_an + epsilon) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))

        total_energy += weight * energy
        total_sum_an += sum_an

    return np.clip(total_energy / (total_sum_an + epsilon), 0.0, 1.0)


def structural_map(image, **kwargs):
    """
    Phase congruency rendered as an 8-bit image the existing matchers can consume.

    Contrast-stretched between percentiles rather than min/max so one bright
    outlier cannot flatten the rest of the map.
    """
    pc = phase_congruency(image, **kwargs)

    lo, hi = np.percentile(pc, (1.0, 99.0))
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)

    stretched = np.clip((pc - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def mutual_information(patch_a, patch_b, bins=32):
    """
    Mutual information between two patches, in bits.

    Unlike NCC this makes no assumption that the two images are linearly
    related — only that the joint intensity distribution is concentrated when
    they are aligned. That is what makes it usable across modalities.
    """
    a = np.asarray(patch_a, dtype=np.float64).ravel()
    b = np.asarray(patch_b, dtype=np.float64).ravel()
    if a.size == 0 or a.size != b.size:
        return 0.0

    joint, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 256], [0, 256]])
    total = joint.sum()
    if total <= 0:
        return 0.0

    joint /= total
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    nz = joint > 0
    outer = p_a[:, None] * p_b[None, :]
    return float(np.sum(joint[nz] * np.log2(joint[nz] / outer[nz])))
