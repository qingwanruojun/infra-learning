module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @main_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1], strides: [%1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %2 = arith.index_cast %arg8 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %2], strides: [%2, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%1, %2], strides: [%2, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
    %5 = arith.addi %arg7, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    scf.for %arg15 = %c0_i32 to %6 step %c1_i32  : i32 {
      %alloc_3 = memref.alloc() : memref<128x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %21 = arith.addi %arg8, %c127_i32 : i32
      %22 = arith.divsi %21, %c128_i32 : i32
      %23 = arith.divsi %4, %22 : i32
      %24 = arith.muli %23, %c128_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.subi %arg6, %24 : i32
      %27 = arith.minsi %26, %c128_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.muli %arg15, %c64_i32 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.subi %arg7, %29 : i32
      %32 = arith.minsi %31, %c64_i32 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_5 = memref.subview %reinterpret_cast[%25, %30] [%28, %33] [1, 1] : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %alloc_3[0, 0] [%28, %33] [1, 1] : memref<128x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %34 = arith.remsi %4, %22 : i32
      %35 = arith.muli %34, %c128_i32 : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.subi %arg8, %35 : i32
      %38 = arith.minsi %37, %c128_i32 : i32
      %39 = arith.index_cast %38 : i32 to index
      %subview_7 = memref.subview %reinterpret_cast_1[%30, %36] [%33, %39] [1, 1] : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %alloc_4[0, 0] [%33, %39] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %40 = arith.cmpi eq, %arg15, %c0_i32 : i32
      hivm.hir.mmadL1 ins(%alloc_3, %alloc_4, %40, %28, %33, %39 : memref<128x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
    }
    %7 = arith.addi %arg8, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.divsi %4, %8 : i32
    %10 = arith.muli %9, %c128_i32 : i32
    %11 = arith.subi %arg6, %10 : i32
    %12 = arith.minsi %11, %c128_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.remsi %4, %8 : i32
    %15 = arith.muli %14, %c128_i32 : i32
    %16 = arith.subi %arg8, %15 : i32
    %17 = arith.minsi %16, %c128_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %subview = memref.subview %alloc[0, 0] [%13, %18] [1, 1] : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<cc>>
    %19 = arith.index_cast %10 : i32 to index
    %20 = arith.index_cast %15 : i32 to index
    %subview_2 = memref.subview %reinterpret_cast_0[%19, %20] [%13, %18] [1, 1] : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    return
  }
  func.func @main_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    return
  }
}