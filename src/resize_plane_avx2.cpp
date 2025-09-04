#include <immintrin.h>

#include "JincResize.h"

#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif

template <typename T, int thr, int subsampled>
void JincResize::resize_plane_avx2(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi)
{
    const int planes_y[4] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V, AVS_PLANAR_A };
    const int planes_r[4] = { AVS_PLANAR_G, AVS_PLANAR_B, AVS_PLANAR_R, AVS_PLANAR_A };
    const int* current_planes = (avs_is_rgb(vi)) ? planes_r : planes_y;
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = current_planes[i];

        const int src_stride = avs_get_pitch_p(src, plane) / sizeof(T);
        const int dst_stride = avs_get_pitch_p(dst, plane) / sizeof(T);
        const int dst_width = avs_get_row_size_p(dst, plane) / sizeof(T);
        const int dst_height = avs_get_height_p(dst, plane);
        const T* srcp = reinterpret_cast<const T*>(avs_get_read_ptr_p(src, plane));
        const __m256 min_val = (i && !avs_is_rgb(vi)) ? _mm256_set1_ps(-0.5f) : _mm256_setzero_ps();

        EWAPixelCoeff* out = [&]()
        {
            if constexpr (subsampled)
                return (i) ? (i == 3) ? JincResize::out[0] : JincResize::out[1] : JincResize::out[0];
            else
                return JincResize::out[0];
        }();

        const int filter_size_mod2 = (out->filter_size / 2) * 2; // for pairs of rows for better efficiency
        const bool notMod2 = filter_size_mod2 < out->filter_size;

        auto loop = [&](int y)
        {
            T* __restrict dstp = reinterpret_cast<T*>(avs_get_write_ptr_p(dst, plane)) + static_cast<int64_t>(y) * dst_stride;

            for (int x = 0; x < dst_width; ++x)
            {
                EWAPixelCoeffMeta* meta = out->meta + static_cast<int64_t>(y) * dst_width + x;
                const T* src_ptr = srcp + (meta->start_y * static_cast<int64_t>(src_stride)) + meta->start_x;
                const float* coeff_ptr = out->factor + meta->coeff_meta;
                __m256 result = _mm256_setzero_ps();
                __m256 result2 = _mm256_setzero_ps();

                if constexpr (std::is_same_v<T, uint8_t>)
                {
                    for (int ly = 0; ly < out->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + lx)))));
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += out->coeff_stride;
                        src_ptr += src_stride;
                    }

                /* double rows version - at i5-9600K looks like same performance or lower. need test at different hosts
                    for (int ly = 0; ly < filter_size_mod2; ly+=2)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + lx)))));
                            const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + src_stride + lx)))));
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            const __m256 coeff2 = _mm256_load_ps(coeff_ptr + out->coeff_stride + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                            result2 = _mm256_fmadd_ps(src_ps2, coeff2, result2);
                        }

                        coeff_ptr += out->coeff_stride * 2;
                        src_ptr += src_stride * 2;
                    }

                    result = _mm256_add_ps(result, result2);

                   if (notMod2) {
                        for (int lx = 0; lx < out->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + lx)))));
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }
                    }
                    */

                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1));
                    hsum = _mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum));
                    dstp[x] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()), _mm_setzero_si128()));
                }
                else if constexpr (std::is_same_v<T, uint16_t>)
                {
                    for (int ly = 0; ly < out->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + lx)))));
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += out->coeff_stride;
                        src_ptr += src_stride;
                    }

                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1));
                    hsum = _mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum));
                    dstp[x] = _mm_cvtsi128_si32(_mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()));
                }
                else
                {
                    for (int ly = 0; ly < out->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_max_ps(_mm256_loadu_ps(src_ptr + lx), min_val);
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += out->coeff_stride;
                        src_ptr += src_stride;
                    }

                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1));
                    dstp[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum)));
                }
            }
        };

        if constexpr (thr)
        {
            for (intptr_t i = 0; i < dst_height; ++i)
                loop(i);
        }
        else
        {
            std::vector<int> l(dst_height);
            std::iota(std::begin(l), std::end(l), 0);
            std::for_each(std::execution::par, std::begin(l), std::end(l), loop);
        }
    }
}

template void JincResize::resize_plane_avx2<uint8_t, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<uint16_t, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<float, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_plane_avx2<uint8_t, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<uint16_t, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<float, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_plane_avx2<uint8_t, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<uint16_t, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<float, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_plane_avx2<uint8_t, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<uint16_t, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx2<float, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template <typename T, int thr, int subsampled>
void JincResize::resize_3planes_avx2(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi)
{
    const int planes_y[4] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V, AVS_PLANAR_A };
    const int planes_r[4] = { AVS_PLANAR_G, AVS_PLANAR_B, AVS_PLANAR_R, AVS_PLANAR_A };
    const int* current_planes = (avs_is_rgb(vi)) ? planes_r : planes_y;

    const int plane0 = current_planes[0];
    const int plane1 = current_planes[1];
    const int plane2 = current_planes[2];

    const int src_stride0 = avs_get_pitch_p(src, plane0) / sizeof(T);
    const int dst_stride0 = avs_get_pitch_p(dst, plane0) / sizeof(T);
    const int dst_width0 = avs_get_row_size_p(dst, plane0) / sizeof(T);
    const int dst_height0 = avs_get_height_p(dst, plane0);

    const int src_stride1 = avs_get_pitch_p(src, plane1) / sizeof(T);
    const int dst_stride1 = avs_get_pitch_p(dst, plane1) / sizeof(T);
    const int dst_width1 = avs_get_row_size_p(dst, plane1) / sizeof(T);

    const int src_stride2 = avs_get_pitch_p(src, plane2) / sizeof(T);
    const int dst_stride2 = avs_get_pitch_p(dst, plane2) / sizeof(T);
    const int dst_width2 = avs_get_row_size_p(dst, plane2) / sizeof(T);

    const T* srcp0 = reinterpret_cast<const T*>(avs_get_read_ptr_p(src, plane0));
    const T* srcp1 = reinterpret_cast<const T*>(avs_get_read_ptr_p(src, plane1));
    const T* srcp2 = reinterpret_cast<const T*>(avs_get_read_ptr_p(src, plane2));

    const __m256 min_val = (0 && !avs_is_rgb(vi)) ? _mm256_set1_ps(-0.5f) : _mm256_setzero_ps();

    EWAPixelCoeff* out = [&]()
    {
            return JincResize::out[0];
    }();

    const int filter_size_mod2 = (out->filter_size / 2) * 2; // for pairs of rows
    const bool notMod2 = filter_size_mod2 < out->filter_size;

    const int filter_size_mod16 = (out->filter_size / 16) * 16; // for pairs of 8-columns
    const bool notMod16 = filter_size_mod16 < out->filter_size;

    auto loop = [&](int y)
    {
        T* __restrict dstp0 = reinterpret_cast<T*>(avs_get_write_ptr_p(dst, plane0)) + static_cast<int64_t>(y) * dst_stride0;
        T* __restrict dstp1 = reinterpret_cast<T*>(avs_get_write_ptr_p(dst, plane1)) + static_cast<int64_t>(y) * dst_stride1;
        T* __restrict dstp2 = reinterpret_cast<T*>(avs_get_write_ptr_p(dst, plane2)) + static_cast<int64_t>(y) * dst_stride2;

        for (int x = 0; x < dst_width0; ++x)
        {
            EWAPixelCoeffMeta* meta = out->meta + static_cast<int64_t>(y) * dst_width0 + x;
            const T* src_ptr0 = srcp0 + (meta->start_y * static_cast<int64_t>(src_stride0)) + meta->start_x;
            const T* src_ptr1 = srcp1 + (meta->start_y * static_cast<int64_t>(src_stride1)) + meta->start_x;
            const T* src_ptr2 = srcp2 + (meta->start_y * static_cast<int64_t>(src_stride2)) + meta->start_x;
            const float* coeff_ptr = out->factor + meta->coeff_meta;
            __m256 result0 = _mm256_setzero_ps();
            __m256 result1 = _mm256_setzero_ps();
            __m256 result2 = _mm256_setzero_ps();

            __m256 result0_2 = _mm256_setzero_ps();
            __m256 result1_2 = _mm256_setzero_ps();
            __m256 result2_2 = _mm256_setzero_ps();


            if constexpr (std::is_same_v<T, uint8_t>)
            {
                for (int ly = 0; ly < out->filter_size; ++ly)
                {
                    for (int lx = 0; lx < out->filter_size; lx += 8)
                    {
                        const __m256 src_ps0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx)))));
                        const __m256 src_ps1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx)))));
                        const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx)))));
                        const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                        result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                        result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                        result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);
                    }

                    coeff_ptr += out->coeff_stride;
                    src_ptr0 += src_stride0;
                    src_ptr1 += src_stride1;
                    src_ptr2 += src_stride2;
                }
                
                /*  double rows version - at i5-9600K looks like same performance or lower. need test at different hosts
                for (int ly = 0; ly < filter_size_mod2; ly+=2)
                {
                    for (int lx = 0; lx < out->filter_size; lx += 8)
                    {
                        const __m256 src_ps0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx)))));
                        const __m256 src_ps1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx)))));
                        const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx)))));

                        const __m256 src_ps0_2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + src_stride0 + lx)))));
                        const __m256 src_ps1_2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + src_stride1 + lx)))));
                        const __m256 src_ps2_2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + src_stride2 + lx)))));


                        const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                        const __m256 coeff_2 = _mm256_load_ps(coeff_ptr + out->coeff_stride + lx);

                        result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                        result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                        result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);

                        result0_2 = _mm256_fmadd_ps(src_ps0_2, coeff_2, result0_2);
                        result1_2 = _mm256_fmadd_ps(src_ps1_2, coeff_2, result1_2);
                        result2_2 = _mm256_fmadd_ps(src_ps2_2, coeff_2, result2_2);

                    }

                    coeff_ptr += out->coeff_stride * 2;
                    src_ptr0 += src_stride0 * 2;
                    src_ptr1 += src_stride1 * 2;
                    src_ptr2 += src_stride2 * 2;
                }

                result0 = _mm256_add_ps(result0, result0_2);
                result1 = _mm256_add_ps(result1, result1_2);
                result2 = _mm256_add_ps(result2, result2_2);

                if (notMod2) {
                    for (int lx = 0; lx < out->filter_size; lx += 8)
                    {
                        const __m256 src_ps0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx)))));
                        const __m256 src_ps1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx)))));
                        const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx)))));

                        const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);

                        result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                        result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                        result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);
                    }
                }
                */ 

                /*  double 8 - columns version
                for (int ly = 0; ly < out->filter_size; ++ly)
                {
                    for (int lx = 0; lx < filter_size_mod16; lx += 16)
                    {
                        const __m256 src_ps0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx)))));
                        const __m256 src_ps1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx)))));
                        const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx)))));

                        const __m256 src_ps0_2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx + 8)))));
                        const __m256 src_ps1_2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx + 8)))));
                        const __m256 src_ps2_2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx + 8)))));

                        const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                        const __m256 coeff_2 = _mm256_load_ps(coeff_ptr + lx + 8);

                        result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                        result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                        result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);

                        result0 = _mm256_fmadd_ps(src_ps0_2, coeff_2, result0);
                        result1 = _mm256_fmadd_ps(src_ps1_2, coeff_2, result1);
                        result2 = _mm256_fmadd_ps(src_ps2_2, coeff_2, result2);

                    }

                    if (notMod16) {
                        for (int lx = filter_size_mod16; lx < out->filter_size; lx += 8)
                        {
                            const __m256 src_ps0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx)))));
                            const __m256 src_ps1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx)))));
                            const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx)))));

                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);

                            result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                            result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                            result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);
                        }
                    }

                    coeff_ptr += out->coeff_stride;
                    src_ptr0 += src_stride0;
                    src_ptr1 += src_stride1;
                    src_ptr2 += src_stride2;
                }
                */

                __m128 hsum0 = _mm_add_ps(_mm256_castps256_ps128(result0), _mm256_extractf128_ps(result0, 1));
                hsum0 = _mm_hadd_ps(_mm_hadd_ps(hsum0, hsum0), _mm_hadd_ps(hsum0, hsum0));
                dstp0[x] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(hsum0), _mm_setzero_si128()), _mm_setzero_si128()));

                __m128 hsum1 = _mm_add_ps(_mm256_castps256_ps128(result1), _mm256_extractf128_ps(result1, 1));
                hsum1 = _mm_hadd_ps(_mm_hadd_ps(hsum1, hsum1), _mm_hadd_ps(hsum1, hsum1));
                dstp1[x] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(hsum1), _mm_setzero_si128()), _mm_setzero_si128()));

                __m128 hsum2 = _mm_add_ps(_mm256_castps256_ps128(result2), _mm256_extractf128_ps(result2, 1));
                hsum2 = _mm_hadd_ps(_mm_hadd_ps(hsum2, hsum2), _mm_hadd_ps(hsum2, hsum2));
                dstp2[x] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(hsum2), _mm_setzero_si128()), _mm_setzero_si128()));

            }
            else if constexpr (std::is_same_v<T, uint16_t>)
            {
                for (int ly = 0; ly < out->filter_size; ++ly)
                {
                    for (int lx = 0; lx < out->filter_size; lx += 8)
                    {
                        const __m256 src_ps0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr0 + lx)))));
                        const __m256 src_ps1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr1 + lx)))));
                        const __m256 src_ps2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr2 + lx)))));
                        const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                        result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                        result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                        result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);
                    }

                    coeff_ptr += out->coeff_stride;
                    src_ptr0 += src_stride0;
                    src_ptr1 += src_stride1;
                    src_ptr2 += src_stride2;
                }

                __m128 hsum0 = _mm_add_ps(_mm256_castps256_ps128(result0), _mm256_extractf128_ps(result0, 1));
                hsum0 = _mm_hadd_ps(_mm_hadd_ps(hsum0, hsum0), _mm_hadd_ps(hsum0, hsum0));
                dstp0[x] = _mm_cvtsi128_si32(_mm_packus_epi32(_mm_cvtps_epi32(hsum0), _mm_setzero_si128()));

                __m128 hsum1 = _mm_add_ps(_mm256_castps256_ps128(result1), _mm256_extractf128_ps(result1, 1));
                hsum1 = _mm_hadd_ps(_mm_hadd_ps(hsum1, hsum1), _mm_hadd_ps(hsum1, hsum1));
                dstp1[x] = _mm_cvtsi128_si32(_mm_packus_epi32(_mm_cvtps_epi32(hsum1), _mm_setzero_si128()));

                __m128 hsum2 = _mm_add_ps(_mm256_castps256_ps128(result2), _mm256_extractf128_ps(result2, 1));
                hsum2 = _mm_hadd_ps(_mm_hadd_ps(hsum2, hsum2), _mm_hadd_ps(hsum2, hsum2));
                dstp2[x] = _mm_cvtsi128_si32(_mm_packus_epi32(_mm_cvtps_epi32(hsum2), _mm_setzero_si128()));

            }
            else
            {
                for (int ly = 0; ly < out->filter_size; ++ly)
                {
                    for (int lx = 0; lx < out->filter_size; lx += 8)
                    {
                        const __m256 src_ps0 = _mm256_max_ps(_mm256_loadu_ps(src_ptr0 + lx), min_val);
                        const __m256 src_ps1 = _mm256_max_ps(_mm256_loadu_ps(src_ptr1 + lx), min_val);
                        const __m256 src_ps2 = _mm256_max_ps(_mm256_loadu_ps(src_ptr2 + lx), min_val);
                        const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                        result0 = _mm256_fmadd_ps(src_ps0, coeff, result0);
                        result1 = _mm256_fmadd_ps(src_ps1, coeff, result1);
                        result2 = _mm256_fmadd_ps(src_ps2, coeff, result2);
                    }

                    coeff_ptr += out->coeff_stride;
                    src_ptr0 += src_stride0;
                    src_ptr1 += src_stride1;
                    src_ptr2 += src_stride2;
                }

                __m128 hsum0 = _mm_add_ps(_mm256_castps256_ps128(result0), _mm256_extractf128_ps(result0, 1));
                dstp0[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(hsum0, hsum0), _mm_hadd_ps(hsum0, hsum0)));

                __m128 hsum1 = _mm_add_ps(_mm256_castps256_ps128(result1), _mm256_extractf128_ps(result1, 1));
                dstp1[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(hsum1, hsum1), _mm_hadd_ps(hsum1, hsum1)));

                __m128 hsum2 = _mm_add_ps(_mm256_castps256_ps128(result2), _mm256_extractf128_ps(result2, 1));
                dstp2[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(hsum2, hsum2), _mm_hadd_ps(hsum2, hsum2)));

            }
        }
    };

    if constexpr (thr)
    {
        for (intptr_t i = 0; i < dst_height0; ++i)
            loop(i);
    }
    else
    {
        std::vector<int> l(dst_height0);
        std::iota(std::begin(l), std::end(l), 0);
        std::for_each(std::execution::par, std::begin(l), std::end(l), loop);
    }

}

template void JincResize::resize_3planes_avx2<uint8_t, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<uint16_t, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<float, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_3planes_avx2<uint8_t, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<uint16_t, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<float, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_3planes_avx2<uint8_t, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<uint16_t, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<float, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_3planes_avx2<uint8_t, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<uint16_t, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_3planes_avx2<float, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
