module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @main_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<64xf16, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<64xf16, strided<[1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<cbuf>>
      %2 = arith.muli %1, %c8_i32 : i32
      %3 = arith.muli %arg11, %c8_i32 : i32
      %4 = arith.addi %2, %3 : i32
      %5 = arith.index_cast %4 : i32 to index
      %subview = memref.subview %reinterpret_cast[%5] [16] [1] : memref<64xf16, strided<[1]>, #hivm.address_space<gm>> to memref<16xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %subview_1 = memref.subview %reinterpret_cast_0[%5] [16] [1] : memref<64xf16, strided<[1]>, #hivm.address_space<gm>> to memref<16xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.fixpipe ins(%alloc : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<cbuf>>) outs(%subview_1 : memref<16xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    }
    return
  }
  func.func @main_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    return
  }
}