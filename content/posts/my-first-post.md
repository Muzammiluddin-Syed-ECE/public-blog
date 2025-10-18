+++
title = 'An attempt to document my work'
date = 2024-01-14T07:07:07+01:00
draft = false
+++
## Introduction

# Project 1 - Writing RoiAlign as a LinAlg generic Op

## The problem

When trying to lower the Mark-CNN on IREE we encounter problems with the Roi Align pooling function that [this paper](https://arxiv.org/pdf/1703.06870v3) implements. It currently is not supported in the Torch to Linalg Conversion pipeline.

## How to create a linalg generic op

Looking at existing examples in the IREE codebase we notice that other similar pooling ops follow a simple recipe in implementation.

Provide inputs, outputs, attribtues and a linalg-compatible description of what the op should do. 

In this way linalg generic ops can act as compiler-readable kernels. 

## RoiAlign

While I do not have a fully fleshed out mental map of the operations done in RoiAlign, at a high level I understand that it is a way to perform a pooling operation across "bins". These bins are subdivisions of a tensor that have been aligned using bounds (or something to that effect). I'll look into this deeper should my unawareness become an obstacle. There's also some nuances with clamping and sampling, but once again I feel I have an intuition of what these things mean. 

But for the purpose of writing this linalg generic I will be relying on the CUDA implementation of the kernel that is available online [at this link](https://github.com/pytorch/vision/blob/9b7c7d395d44832104b583e82dfc5c5272535ebb/torchvision/csrc/ops/cuda/roi_align_kernel.cu#L14-L143).

## Translation into linalg

Let us break down the CUDA kernel that we're trying to write as linalg:

    template <typename T>
    __device__ T bilinear_interpolate(
        const T* input,
        int height,
        int width,
        T y,
        T x,
        int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0;
    }

    if (y <= 0)
        y = 0;
    if (x <= 0)
        x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;

    // do bilinear interpolation
    T v1 = input[y_low * width + x_low];
    T v2 = input[y_low * width + x_high];
    T v3 = input[y_high * width + x_low];
    T v4 = input[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
    }

    template <typename T>
    __global__ void roi_align_forward_kernel_impl(
        int nthreads,
        const T* input,
        const T spatial_scale,
        int channels,
        int height,
        int width,
        int pooled_height,
        int pooled_width,
        int sampling_ratio,
        bool aligned,
        const T* rois,
        T* output) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        // Do not using rounding; this implementation detail is critical
        T offset = aligned ? (T)0.5 : (T)0.0;
        T roi_start_w = offset_rois[1] * spatial_scale - offset;
        T roi_start_h = offset_rois[2] * spatial_scale - offset;
        T roi_end_w = offset_rois[3] * spatial_scale - offset;
        T roi_end_h = offset_rois[4] * spatial_scale - offset;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
        if (!aligned) {
        // Force malformed ROIs to be 1x1
        roi_width = max(roi_width, (T)1.);
        roi_height = max(roi_height, (T)1.);
        }

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        const T* offset_input =
            input + (roi_batch_ind * channels + c) * height * width;

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / pooled_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        // When the grid is empty, output zeros.
        const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

        T output_val = 0.;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
        {
        const T y = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const T x = roi_start_w + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);

            T val = bilinear_interpolate(offset_input, height, width, y, x, index);
            output_val += val;
        }
        }
        output_val /= count;

        output[index] = output_val;
    }
    }

Ok so I've realized that I'm supposed to be using the Python implementation as reference not the cuda implementation. 

Let's first compare the AtenAvgPool2D to see if i can understand how the translation works



For each ROI we are given 5 values in a list (index, $x_1$, $x_2$, $y_1$, $y_2$)

where x represents start and y represents end

roi_width = $ y_1 - x_1$
roi_height = $ y_2 - x_2$

if we do not want to align we can clamp the width to min 1

height of bin = roi_eight / pooled_height
width of bin = roi_width / pooled_width

if we do not want exact sampling

roi_bin_grid_h = torch.ceil(height of bin)
roi_bin_grid_w = torch.ceil(wifth of bin)

else set both to sampling ratio


_Pseudocode_:

    roi
    this is code