#!/usr/bin/env python3

##################### inference_backend.py #####################################
#
# Copyright 2021 The IBM Research Authors.
#
################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import warnings
import json
import base64
import numpy as np
import subprocess
from onnx.backend.base import Device, DeviceType, Backend
from onnx import numpy_helper
import variables
from variables import *
from common import compile_model


def get_test_models():
    # Test directories:
    # https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node
    # In our directories, the python files that generate the tests are found here
    # onnx-mlir/third_party/onnx/onnx/backend/test/case/node

    # Each benchmark is defined by a dictionary element: `key:value`, where
    # - key: the ONNX testname
    # - value: a dictionary accepting only three key "static", "dynamic", and
    #          "constant" to enable testings for static, dynamic, and constant
    #          inputs, respectively.
    # When a dynamic or constant testing is enabled, we must enter the indices of
    # the tensor expected to be dynamic or constant.
    # Indices start from 0. -1 means all inputs or all dimensions.
    #
    # Value for "static" key is not taken into account. So empty set {} is O.K.
    #
    # Value for "dynamic" key is a dict to define which inputs/dimensions are changed
    # to unknown, where its key is an input index and its value is a set of
    # dimension indices, e.g. {0:{0,1}, 1:{-1}, 2:{0}}
    #
    # Value for "constant" key is set of indices, e.g. {0, 2, 3}


    # ADDING NEW TESTS / OPS
    #
    # * Please add new ops in the order they are found in
    #   onnx-mlir/third_party/onnx/onnx/backend/test/case/node
    # * Please add individual tests in the order they are found in the file.
    #   Most have been properly ordered, some are still not. Please fix as you
    #   make changes
    #
    #
    # SEMANTIC for LABELING (one line per directive)
    # see utils/genSupportedOps.py
    # command processed by makefile. hi alex

    variables.test_to_enable_dict = {
        ############################################################
        # Elementary ops, ordered in the order they are found in
        # onnx-mlir/third_party/onnx/onnx/backend/test/case/node.

        # ==ARCH== cpu

        # ==OP== Abs current
        "test_abs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Acos current
        "test_acos_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_acos_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Acosh current
        "test_acosh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_acosh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Adagrad

        # ==OP== Adam

        # ==OP== Add current
        # ==LIM== No support for short integers.
        "test_add_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_add_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_add_uint8_cpu" : {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== And current
        "test_and2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and_bcast3v1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and_bcast3v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and_bcast4v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and_bcast4v3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_and_bcast4v4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Argmax current
        "test_argmax_no_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_argmax_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_argmax_default_axis_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_argmax_no_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_argmax_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_argmax_default_axis_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    
        # ==OP== Argmin

        # ==OP== Asin current
        "test_asin_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_asin_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Asinh current
        "test_asinh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_asinh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Atan current
        "test_atan_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_atan_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP==  Atanh current
        "test_atanh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_atanh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== AveragePool current
        # TODO: original comment stated "same_upper/lower with dynamic padding-shapes not supported."
        # However, I see the dyn shape test being done on all tests, including same_upper. So I am 
        # assuming that this comment is outdated.
        "test_averagepool_1d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_ceil_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_pads_count_include_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_precomputed_pads_count_include_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_precomputed_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_precomputed_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_precomputed_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_same_lower_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_2d_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_averagepool_3d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== BatchNormalization current
        # ==LIM== Training not supported.
        "test_batchnorm_epsilon_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_batchnorm_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Bernoulli

        # ==OP== Bitshift

        # ==OP== Cast current
        # ==LIM== Cast only between float and double types
        "test_cast_FLOAT_to_DOUBLE_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cast_DOUBLE_to_FLOAT_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cast_FLOAT_to_FLOAT16_cpu": {}, # appears unsupported at this time
        "test_cast_FLOAT16_to_FLOAT_cpu": {}, # appears unsupported at this time
        "test_cast_FLOAT16_to_DOUBLE_cpu": {}, # appears unsupported at this time
        "test_cast_DOUBLE_to_FLOAT16_cpu": {}, # appears unsupported at this time
        "test_cast_FLOAT_to_STRING_cpu": {}, # appears unsupported at this time
        "test_cast_STRING_to_FLOAT_cpu": {}, # appears unsupported at this time

        # ==OP== CastLike

        # ==OP== Ceil current
        "test_ceil_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_ceil_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Celu

        # ==OP== Clip current
        # ==LIM== No support for short integers
        "test_clip_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_inbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_outbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_splitbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_default_min_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_default_max_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_clip_default_inbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_clip_default_int8_min_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        #"test_clip_default_int8_max_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        #"test_clip_default_int8_inbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

        # ==OP== Compress current
        "test_compress_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_compress_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_compress_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_compress_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Concat current
        "test_concat_1d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
        "test_concat_2d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
        "test_concat_2d_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
        "test_concat_3d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
        "test_concat_3d_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
        "test_concat_3d_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{2}}, CONSTANT_INPUT:{-1}},
        "test_concat_1d_axis_negative_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
        "test_concat_2d_axis_negative_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
        "test_concat_2d_axis_negative_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
        "test_concat_3d_axis_negative_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{2}}, CONSTANT_INPUT:{-1}},
        "test_concat_3d_axis_negative_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
        "test_concat_3d_axis_negative_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},

        # ==OP== Constant current
        # By def, no dynamic shapes.
        "test_constant_cpu": {STATIC_SHAPE:{}},

        # ==OP== ConstantOfShape current
        # By def, no dynamic shapes.
        "test_constantofshape_float_ones_cpu": {STATIC_SHAPE:{}},
        "test_constantofshape_int_zeros_cpu": {STATIC_SHAPE:{}},
        "test_constantofshape_int_shape_zero_cpu": {STATIC_SHAPE:{}},

        # ==OP== Conv current
        # CONSTANT_INPUT for weight only. No need to make a restriction.
        "test_basic_conv_with_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
        "test_basic_conv_without_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
        "test_conv_with_autopad_same_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
        "test_conv_with_strides_no_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
        "test_conv_with_strides_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
        "test_conv_with_strides_and_asymmetric_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},

        # ==OP== ConvInteger

        # ==OP== ConvTranspose

        # ==OP== Cos current
        "test_cos_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cos_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Cosh current
        "test_cosh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cosh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== CumSum current
        "test_cumsum_1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cumsum_1d_exclusive_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cumsum_1d_reverse_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cumsum_1d_reverse_exclusive_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cumsum_2d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cumsum_2d_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_cumsum_2d_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== DepthOfSpace current
        "test_depthtospace_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_depthtospace_crd_mode_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== DequatizeLinear

        # ==OP== Det

        # ==OP== Div current
        # ==LIM== No support for short integers.
        "test_div_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_div_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_div_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_div_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Dropout current
        # ==LIM== Does not support masked and training.
        "test_dropout_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_dropout_default_ratio_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # Other dropout test case failed: implementation is missing
        # mask is not supported for inference
        #"test_dropout_default_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        #"test_dropout_default_mask_ratio_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        # Error: input arrays contain a mixture of endianness configuration
        #"test_training_dropout_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        #"test_training_dropout_default_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        # Error: input arrays contain a mixture of endianness configuration
        #"test_training_dropout_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        #"test_training_dropout_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        # Error: input arrays contain a mixture of endianness configuration
        #"test_training_dropout_zero_ratio_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        #"test_training_dropout_zero_ratio_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

        # ==OP== DynamicQuantizeLinear

        # ==OP== EinSum

        # ==OP== Elu current
        "test_elu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_elu_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_elu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Equal current
        "test_equal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_equal_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Erf current
        "test_erf_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Exp current
        "test_exp_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_exp_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Expand current
        "test_expand_dim_changed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_expand_dim_unchanged_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Eyelike

        # ==OP== Flatten current
        "test_flatten_axis0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_axis1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_axis2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_axis3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_negative_axis1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_negative_axis2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_negative_axis3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_flatten_negative_axis4_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Floor current
        "test_floor_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_floor_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Gather current
        "test_gather_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gather_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gather_2d_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gather_negative_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== GatherElements current
        "test_gather_elements_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gather_elements_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gather_elements_negative_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== GatherND current
        "test_gathernd_example_int32_cpu": {STATIC_SHAPE:{}, CONSTANT_INPUT:{-1}},
        "test_gathernd_example_float32_cpu": {STATIC_SHAPE:{}, CONSTANT_INPUT:{-1}},
        "test_gathernd_example_int32_batch_dim1_cpu": {STATIC_SHAPE:{}, CONSTANT_INPUT:{-1}},

        # ==OP== Gemm current
        "test_gemm_all_attributes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_alpha_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_beta_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_default_matrix_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_default_no_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_default_scalar_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_default_single_elem_vector_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_default_vector_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_default_zero_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_transposeA_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_gemm_transposeB_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== GlobalAveragePool current
        "test_globalaveragepool_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_globalaveragepool_precomputed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== GlobalMaxPool current
        "test_globalmaxpool_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_globalmaxpool_precomputed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== GreaterOrEqual current
        "test_greater_equal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_greater_equal_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # Could not find code for the next two, no idea where they are coming from, but they work.
        "test_greater_equal_bcast_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_greater_equal_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Greater current
        "test_greater_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_greater_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== GridSample

        # ==OP== GRU current
        # ==LIM== Batchwise test is not supported.
        # CONSTANT_INPUT for W and R.
        "test_gru_defaults_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        "test_gru_with_initial_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        "test_gru_seq_length_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        #"test_gru_batchwise_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},

        # ==OP== HardMax current
        "test_hardmax_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardmax_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardmax_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardmax_one_hot_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardmax_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardmax_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardmax_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== HardSigmoid current
        "test_hardsigmoid_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardsigmoid_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_hardsigmoid_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== HardSwish

        # ==OP== Identity current
        # ==LIM== Sequence identity not supported.
        "test_identity_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_identity_sequence_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_identity_opt_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== If

        # ==OP== InstanceNorm current
        "test_instancenorm_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_instancenorm_epsilon_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== IsInf
        
        # ==OP== IsNan

        # ==OP== LeakyRelu current
        "test_leakyrelu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_leakyrelu_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_leakyrelu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== LessOrEqual current
        "test_less_equal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_less_equal_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # No idea where the code is for the expanded version, but it works.
        "test_less_equal_bcast_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_less_equal_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Less current
        "test_less_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_less_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Log current
        "test_log_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_log_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== LogSoftmax current
        # ==LIM== Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13
        # ==TODO== Temporally removed due to changes in onnx 1.8.1
        # "test_logsoftmax_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_logsoftmax_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_logsoftmax_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_logsoftmax_example_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_logsoftmax_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_logsoftmax_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_logsoftmax_large_number_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Loop Opset 11
        # ==LIM== No support for opset 13 and 16 at this time.
        "test_loop11_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_loop13_seq_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_loop16_seq_none_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== LRN current
        "test_lrn_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_lrn_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== LSTM current
        # ==LIM== No support for batchwise examples
        # CONSTANT_INPUT for W and R.
        "test_lstm_defaults_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        "test_lstm_with_initial_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        "test_lstm_with_peepholes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        #"test_lstm_batchwise_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},

        # ==OP== Matmul current
        "test_matmul_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_matmul_3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_matmul_4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== MatmulInteger

        # ==OP== Max current
        # ==LIM== No support for short floats and unsigned int.
        "test_max_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # float16 failed on Z. It seems LLVM on Z does not have fp16 simulation.
        # "test_max_float16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_float32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_float64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_int8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_int16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_int32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_max_int64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # loc("onnx.Max"): error: 'std.cmpi' op operand #0 must be signless-integer-like, but got 'ui8'
        # MLIR integers are currently signless.
        # "test_max_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_max_uint16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_max_uint32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_max_uint64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== MaxPoolSingleOut current
        # ==LIM== Does not support argmax and short ints
        # TODO: this comment does not appear to be true: same_upper/lower dyn padding-shapes not supported.
        #"test_maxpool_2d_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_precomputed_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_maxpool_with_argmax_2d_precomputed_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_precomputed_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_maxpool_with_argmax_2d_precomputed_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_precomputed_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_1d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_3d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_same_lower_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_ceil_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_maxpool_2d_dilations_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== MaxUnpool

        # ==OP== Mean current
        "test_mean_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_mean_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_mean_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        
        # ==OP== MeanVarianceNormalization

        # ==OP== Min current
        # ==LIM== Does not support short floats and unsigned numbers.
        "test_min_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # float16 failed on Z. It seems LLVM on Z does not have fp16 simulation.
        # "test_min_float16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_float32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_float64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_int8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_int16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_int32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_min_int64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # loc("onnx.Min"): error: 'std.cmpi' op operand #0 must be signless-integer-like, but got 'ui8'
        # MLIR integers are curretnly signless.
        # "test_min_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_min_uint16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_min_uint32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_min_uint64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Mod current
        # ==LIM==  Support float and double only.
        "test_mod_mixed_sign_float32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_mod_mixed_sign_float64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # float16 failed on Z. It seems LLVM on Z does not have fp16 simulation.
        # "test_mod_mixed_sign_float16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # Not yet support integers since MLIR integers are signless.
        # "test_mod_broadcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_int64_fmod_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_mixed_sign_int16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_mixed_sign_int32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_mixed_sign_int64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_mixed_sign_int8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_uint16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_uint32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_uint64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_mod_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Momentum

        # ==OP== Mul current
        # ==LIM== Does not support short integers.
        "test_mul_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_mul_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_mul_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_mul_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Neg current
        "test_neg_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_neg_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== NegativeLogLikelihoodLoss

        # ==OP== NonMaxSuppression current
        "test_nonmaxsuppression_center_point_box_format_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_flipped_coordinates_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_identical_boxes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_limit_output_size_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_single_box_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_suppress_by_IOU_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_two_batches_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_nonmaxsuppression_two_classes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== NonZero current
        "test_nonzero_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Not current
        "test_not_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_not_3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_not_4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== OneHot current
        "test_onehot_without_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_onehot_with_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_onehot_negative_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_onehot_with_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== OptionalGetElement

        # ==OP== OptionalHasElement
        
        # ==OP== Or current
        "test_or2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or_bcast3v1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or_bcast3v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or_bcast4v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or_bcast4v3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_or_bcast4v4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Pad current
        "test_constant_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_edge_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reflect_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Pow current
        # ==LIM== No support for power with integer types
        "test_pow_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_pow_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_pow_bcast_scalar_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_pow_bcast_array_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== PRelu current
        "test_prelu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_prelu_broadcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== QLinearConv

        # ==OP== QLinearMatmul

        # ==OP== QuantizeLinear

        # ==OP== Range current
        "test_range_float_type_positive_delta_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_range_int32_type_negative_delta_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReciprocalOp current
        "test_reciprocal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reciprocal_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceLogSumExp current
        "test_reduce_log_sum_exp_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceLogSum current
        "test_reduce_log_sum_desc_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_asc_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_negative_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_log_sum_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceL1 current
        "test_reduce_l1_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l1_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l1_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l1_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l1_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l1_keep_dims_random_cpu": {STATIC_SHAPE:{}},
        "test_reduce_l1_negative_axes_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l1_negative_axes_keep_dims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceL2 current
        "test_reduce_l2_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_keep_dims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_negative_axes_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_l2_negative_axes_keep_dims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceMax current
        "test_reduce_max_default_axes_keepdim_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_max_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceMean current
        "test_reduce_mean_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_mean_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceMin current
        "test_reduce_min_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_min_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceProd current
        "test_reduce_prod_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_prod_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ReduceSum current
        # ==LIM== Default axis and do_not_keep_dim not supported.
        # ==TODO== Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1
        #"test_reduce_sum_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_reduce_sum_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_reduce_sum_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_reduce_sum_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{0}},
        "test_reduce_sum_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{0}},
        "test_reduce_sum_empty_axes_input_noop_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{0}},

        # ==OP== ReduceSumSquare current
        "test_reduce_sum_square_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reduce_sum_square_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Relu current
        "test_relu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Reshape current
        "test_reshape_extended_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_negative_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_negative_extended_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_one_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_reduced_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_reordered_all_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_reordered_last_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_zero_and_negative_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reshape_zero_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Resize current
        # ==LIM== Missing support for linear, cubic, crop, pytorch_half_pixel, and floor.
        "test_resize_upsample_scales_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_resize_downsample_scales_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_resize_upsample_sizes_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_resize_downsample_sizes_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
        # missing test_resize_upsample_scales_linear
        # test_resize_downsample_scales_linear
        # test_resize_downsample_scales_linear_align_corners
        # test_resize_upsample_scales_cubic
        # test_resize_upsample_scales_cubic_align_corners
        # test_resize_downsample_scales_cubic
        # test_resize_downsample_scales_cubic_align_corners
        # test_resize_upsample_sizes_cubic
        # test_resize_downsample_sizes_cubic
        # test_resize_upsample_scales_cubic_A_n0p5_exclude_outside
        # test_resize_downsample_scales_cubic_A_n0p5_exclude_outside
        # test_resize_upsample_scales_cubic_asymmetric
        # test_resize_tf_crop_and_resize
        # test_resize_tf_crop_and_resize
        # test_resize_downsample_sizes_linear_pytorch_half_pixel
        # test_resize_upsample_sizes_nearest_floor_align_corners
        "test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
        "test_resize_upsample_sizes_nearest_ceil_half_pixel_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Reverse Sequence current
        "test_reversesequence_time_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_reversesequence_batch_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== RNN current
        # ==LIM== Batchwise not supported.
        "test_simple_rnn_defaults_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        "test_simple_rnn_with_initial_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        "test_rnn_seq_length_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
        #  "test_simple_rnn_batchwise_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},

        # ==OP== RoiAlign

        # ==OP== Round current
        "test_round_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Scan Opset 9
        # ==LIM== Does not support dynamic shapes.
        # ==TODO== Precision issue with newer opset, maybe just unsupported. Dynamic shape?
        #  "test_scan_sum_cpu": {STATIC_SHAPE:{}},
        "test_scan9_sum_cpu": {STATIC_SHAPE:{}},

        # ==OP== ScatterElements current
        # ==LIM== Does not support duplicate indices.
        "test_scatter_elements_without_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_scatter_elements_with_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_scatter_elements_with_negative_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_scatter_elements_with_duplicate_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== ScatterND current
        # ==LIM== Does not support scatternd add/multiply.
        "test_scatternd_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_scatternd_add_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_scatternd_multiply_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Selu current
        "test_selu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_selu_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_selu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== SequenceInsert

        # ==OP== Shape current
        "test_shape_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_shape_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Shrink

        # ==OP== Sigmoid current
        "test_sigmoid_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sigmoid_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Sign current
        "test_sign_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Sin current
        "test_sin_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sin_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Sinh current
        "test_sinh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sinh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Size current
        "test_size_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_size_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Slice current
        # ==LIM== Axis must be a constant argument.
        # ==TODO== Add tests to slices, currently have none.
        # (makes Axis a runtime argument, which is not supported).

        # ==OP== Softmax current
        "test_softmax_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softmax_large_number_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softmax_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softmax_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softmax_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softmax_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softmax_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== SoftmaxCrossEntropyLoss

        # ==OP== Softplus current
        "test_softplus_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softplus_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Softsign current
        "test_softsign_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_softsign_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== SpaceToDepth
        # ==TODO== Example works, the other is imprecise. To investigate.
        #"test_spacetodepth_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_spacetodepth_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Split current
        # ==LIM== Does not support static and dynamic shape, zero size splits.
        # ==TODO== Temporally removed due to changes in onnx 1.8.1
        # "test_split_equal_parts_1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_split_variable_parts_1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_split_equal_parts_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_split_variable_parts_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_split_equal_parts_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_split_variable_parts_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # Enabled to test for constant splits
        "test_split_equal_parts_1d_cpu": {CONSTANT_INPUT:{-1}},
        "test_split_variable_parts_1d_cpu": {CONSTANT_INPUT:{1}},
        "test_split_equal_parts_2d_cpu": {CONSTANT_INPUT:{-1}},
        "test_split_variable_parts_2d_cpu": {CONSTANT_INPUT:{1}},
        "test_split_equal_parts_default_axis_cpu": {CONSTANT_INPUT:{-1}},
        "test_split_variable_parts_default_axis_cpu": {CONSTANT_INPUT:{1}},
        #"test_split_zero_size_splits_cpu": {CONSTANT_INPUT:{1}},

        # ==OP== Sqrt current
        "test_sqrt_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sqrt_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Squeeze current
        # ==LIM== Does not support static and dynamic shape.
        # ==TODO== Temporally removed due to changes in onnx 1.8.1
        #"test_squeeze_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_squeeze_negative_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # Enabled to test for constant axes
        "test_squeeze_cpu": {CONSTANT_INPUT:{1}},
        "test_squeeze_negative_axes_cpu": {CONSTANT_INPUT:{1}},

        # ==OP== StrNormalizer

        # ==OP== Sub current
        # ==LIM== Does not support short integers.
        "test_sub_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sub_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        #"test_sub_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sub_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Sum current
        "test_sum_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sum_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_sum_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Tan current
        "test_tan_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_tan_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Tanh current
        "test_tanh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_tanh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== TfdfVectorizer

        # ==OP== ThresholdRelu

        # ==OP== Tile current
        "test_tile_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_tile_precomputed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== TopK current
        "test_top_k_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_top_k_smallest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_top_k_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Transpose current
        "test_transpose_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_transpose_all_permutations_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_transpose_all_permutations_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_transpose_all_permutations_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_transpose_all_permutations_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_transpose_all_permutations_4_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_transpose_all_permutations_5_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Trilu

        # ==OP== Unique

        # ==OP== Unsqueeze current
        # ==LIM== Does not support static and dynamic shape.
        # ==TODO== Temporally removed due to changes in onnx 1.8.1
        # "test_unsqueeze_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_axis_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_negative_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_three_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_two_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # "test_unsqueeze_unsorted_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        # Enabled to test for constant axes
        "test_unsqueeze_axis_0_cpu": {CONSTANT_INPUT:{1}},
        "test_unsqueeze_axis_1_cpu": {CONSTANT_INPUT:{1}},
        "test_unsqueeze_axis_2_cpu": {CONSTANT_INPUT:{1}},
        # Using Opset v11 still
        "test_unsqueeze_axis_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
        "test_unsqueeze_negative_axes_cpu": {CONSTANT_INPUT:{1}},
        "test_unsqueeze_three_axes_cpu": {CONSTANT_INPUT:{1}},
        "test_unsqueeze_two_axes_cpu": {CONSTANT_INPUT:{1}},
        "test_unsqueeze_unsorted_axes_cpu": {CONSTANT_INPUT:{1}},

        # ==OP== Upsample current
        "test_upsample_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Where current
        "test_where_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_where_long_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        # ==OP== Xor current
        "test_xor2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor_bcast3v1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor_bcast3v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor_bcast4v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor_bcast4v3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
        "test_xor_bcast4v4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

        ############################################################
        # Model (alphabetical order)

        "test_densenet121_cpu": {STATIC_SHAPE:{}},
        "test_inception_v1_cpu": {STATIC_SHAPE:{}},
        "test_resnet50_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}},
        "test_shufflenet_cpu": {STATIC_SHAPE:{}},
        "test_squeezenet_cpu": {STATIC_SHAPE:{}},
        "test_vgg19_cpu": {STATIC_SHAPE:{}},
    }

    # Test for static inputs.
    test_to_enable = [
        key for (key, value) in variables.test_to_enable_dict.items() if STATIC_SHAPE in value
    ]

    # Test for dynamic inputs.
    # Specify the test cases which currently can not pass for dynamic shape
    # Presumably, this list should be empty
    # Except for some operation too difficult to handle for dynamic shape
    # or big models
    variables.test_for_dynamic = [
        key for (key, value) in variables.test_to_enable_dict.items() if DYNAMIC_SHAPE in value
    ]

    # Test for constant inputs.
    variables.test_for_constant = [
        key for (key, value) in variables.test_to_enable_dict.items() if CONSTANT_INPUT in value
    ]

    # Specify the test cases which need version converter
    variables.test_need_converter = []

    if args.dynamic:
        print("dynamic shape is enabled", file=sys.stderr)
        test_to_enable = variables.test_for_dynamic

    if args.constant:
        print("constant input is enabled", file=sys.stderr)
        test_to_enable = variables.test_for_constant

    # User case specify one test case with BCKEND_TEST env
    TEST_CASE_BY_USER = os.getenv("TEST_CASE_BY_USER")
    if TEST_CASE_BY_USER is not None and TEST_CASE_BY_USER != "":
        test_to_enable = TEST_CASE_BY_USER.split()

    return test_to_enable


def JniExecutionSession(jar_name, inputs):
    procStdin = json.dumps(
        list(
            map(
                lambda tensor: {
                    "buffer": base64.b64encode(tensor.flatten().tobytes()).decode(
                        "utf-8"
                    ),
                    "dtype": tensor.dtype.str,
                    "shape": np.shape(tensor),
                },
                inputs,
            )
        )
    )
    # print(str(inputs), file=sys.stderr)
    # print('stdin=' + str(procStdin), file=sys.stderr)
    cmd = [
        "java",
        "-cp",
        jar_name + ":" + os.getenv("JSONITER_JAR"),
        "com.ibm.onnxmlir.OMRunner",
    ]
    print(cmd, file=sys.stderr)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    procStdout = json.loads(
        proc.communicate(input=procStdin.encode("utf-8"))[0].decode("utf-8").strip()
    )

    dtype = {
        "b1": np.bool,
        "i1": np.int8,
        "u1": np.uint8,
        "i2": np.int16,
        "u2": np.uint16,
        "i4": np.int32,
        "u4": np.uint32,
        "i8": np.int64,
        "u8": np.uint64,
        "f4": np.float32,
        "f8": np.float64,
    }

    # print('stdout=' + str(procStdout), file=sys.stderr)
    outputs = list(
        map(
            lambda tensor: np.frombuffer(
                base64.b64decode(tensor["buffer"]), dtype[tensor["dtype"][1:]]
            ).reshape(tensor["shape"]),
            procStdout,
        )
    )
    # print('outputs=' + str(outputs), file=sys.stderr)
    return outputs


# There are two issues, which necessitates the adoption of this endianness
# aware wrapper around Execution Session:
# 1. Input arrays are given sometimes in native byte order, sometime in
#    LE byte order, and as soon as the python array enters into py::array
#    C++ objects through pybind, we will no longer be able to query their
#    endianness. So we must intercept the inputs and convert them into
#    native endianness.
# 2. Output arrays are compared with reference outputs, the comparison
#    unfortunately includes checking that our outputs and reference outputs
#    share the same endianness. So we try to figure out what is the desired
#    reference output endianness, and convert our outputs to this desired
#    endianness.
class EndiannessAwareExecutionSession(object):
    def __init__(self, model):
        self.model = model
        self.exec_name = None
        # Compiling the model in advance if not testing constants, so that
        # the model is compiled once and used multiple times.
        if not args.constant:
            self.exec_name = compile_model(self.model, args.emit)

    def is_input_le(self, inputs):
        inputs_endianness = list(map(lambda x: x.dtype.byteorder, inputs))
        endianness_is_consistent = len(set(inputs_endianness)) <= 1
        assert (
            endianness_is_consistent
        ), "Input arrays contain a mixture of endianness configuration."

        sys_is_le = sys.byteorder == "little"
        # To interpret character symbols indicating endianness:
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
        explicitly_le = inputs_endianness[0] == "<"
        implicitly_le = inputs_endianness[0] == "=" and sys_is_le
        return explicitly_le or implicitly_le

    def is_not_relevant_endian(self, inputs):
        inputs_endianness = list(map(lambda x: x.dtype.byteorder, inputs))
        endianness_is_consistent = len(set(inputs_endianness)) <= 1
        assert (
            endianness_is_consistent
        ), "Input arrays contain a mixture of endianness configuration."

        sys_is_le = sys.byteorder == "little"
        # To interpret character symbols indicating endianness:
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
        i1_not_relevant_endian = inputs_endianness[0] == "|"
        return i1_not_relevant_endian

    def turn_model_input_to_constant(self, inputs):
        # If IMPORTER_FORCE_CONSTANT is set, get input indices from it.
        # Otherwise, get from test_to_enable_dict.
        IMPORTER_FORCE_CONSTANT = os.getenv("IMPORTER_FORCE_CONSTANT")
        input_indices = {}

        if IMPORTER_FORCE_CONSTANT:
            input_indices = set(
                map(lambda x: int(x.strip()), IMPORTER_FORCE_CONSTANT.split(","))
            )
        else:
            test_name_cpu = self.model.graph.name + "_cpu"
            if test_name_cpu in variables.test_for_constant:
                test_info = variables.test_to_enable_dict[test_name_cpu]
                input_indices = test_info.get(CONSTANT_INPUT)

        # Change the model by turning input tensors to initializers with the
        # same name, so that the inputs will be constants at compile time.
        # This is for testing a model when its inputs are constants.
        num_of_inputs = len(inputs)
        if -1 in input_indices:
            input_indices = range(num_of_inputs)
        # Create initializers that have the same name as inputs.
        for idx in input_indices:
            tensor = inputs[idx]
            tensor = numpy_helper.from_array(tensor, self.model.graph.input[idx].name)
            self.model.graph.initializer.extend([tensor])
        # Remove inputs that were turned to constants.
        new_inputs = []
        for idx in range(num_of_inputs):
            if idx not in input_indices:
                new_inputs.append(inputs[idx])
        return new_inputs

    def run(self, inputs, **kwargs):
        sys.path.append(RUNTIME_DIR)
        from PyRuntime import ExecutionSession

        if len(inputs):
            inputs_endianness = list(map(lambda x: x.dtype.byteorder, inputs))
            endianness_is_consistent = len(set(inputs_endianness)) <= 1
            # Deduce desired endianness of output from inputs.
            # Only possible if all inputs are consistent in endiannness.
            if endianness_is_consistent:
                sys_is_le = sys.byteorder == "little"
                inp_is_le = self.is_input_le(inputs)
                inp_is_not_relevant_endian = self.is_not_relevant_endian(inputs)
                if not inp_is_not_relevant_endian and sys_is_le != inp_is_le:
                    inputs = list(map(lambda x: x.byteswap().newbyteorder(), inputs))
            # If constant test, change the model inputs to constants.
            if args.constant:
                inputs = self.turn_model_input_to_constant(inputs)
                self.exec_name = compile_model(self.model, args.emit)
            if args.emit == "lib":
                session = ExecutionSession(self.exec_name)
                outputs = session.run(inputs)
                # print('input='+str(inputs), file=sys.stderr)
                # print('output='+str(outputs), file=sys.stderr)
            elif args.emit == "jni":
                outputs = JniExecutionSession(self.exec_name, inputs)
            if (
                endianness_is_consistent
                and not inp_is_not_relevant_endian
                and sys_is_le != inp_is_le
            ):
                outputs = list(map(lambda x: x.byteswap().newbyteorder(), outputs))
            return outputs
        else:
            # Can't deduce desired output endianess, fingers crossed.
            warnings.warn(
                "Cannot deduce desired output endianness, using native endianness by default."
            )
            if args.emit == "lib":
                session = ExecutionSession(self.exec_name)
                outputs = session.run(inputs)
            elif args.emit == "jni":
                outputs = JniExecutionSession(self.exec_name, inputs)
            return outputs


class InferenceBackend(Backend):
    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        super(InferenceBackend, cls).prepare(model, device, **kwargs)
        return EndiannessAwareExecutionSession(model)

    @classmethod
    def supports_device(cls, device):
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False
