module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @main_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256, 256], strides: [256, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [256, 256], strides: [256, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [256, 256], strides: [256, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
    scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %alloc_2 = memref.alloc() : memref<128x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
          %alloc_3 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %4 = arith.muli %arg12, %c128_i32 : i32
          %5 = arith.index_cast %4 : i32 to index
          %6 = arith.muli %arg14, %c64_i32 : i32
          %7 = arith.index_cast %6 : i32 to index
          %subview_4 = memref.subview %reinterpret_cast[%5, %7] [128, 64] [1, 1] : memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<128x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_4 : memref<128x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_2 : memref<128x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          %8 = arith.muli %arg13, %c128_i32 : i32
          %9 = arith.index_cast %8 : i32 to index
          %subview_5 = memref.subview %reinterpret_cast_1[%7, %9] [64, 128] [1, 1] : memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<64x128xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_3 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          %10 = arith.cmpi eq, %arg14, %c0_i32 : i32
          hivm.hir.mmadL1 ins(%alloc_2, %alloc_3, %10, %c128, %c64, %c128 : memref<128x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
        }
        %0 = arith.muli %arg12, %c128_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg13, %c128_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %subview = memref.subview %reinterpret_cast_0[%1, %3] [128, 128] [1, 1] : memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%alloc : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%subview : memref<128x128xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
      }
    }
    return
  }
  func.func @main_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    return
  }
}