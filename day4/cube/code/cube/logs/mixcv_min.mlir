module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.part_of_mix, mix_mode = "mix"} {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024, 512], strides: [%c512, %c1] : memref<?xf16> to memref<1024x512xf16, strided<[512, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1024, 1024], strides: [%c1024, %c1] : memref<?xf32> to memref<1024x1024xf32, strided<[1024, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 1024], strides: [%c1024, %c1] : memref<?xf16> to memref<512x1024xf16, strided<[1024, 1]>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1024, 1024], strides: [%c1024, %c1] : memref<?xf32> to memref<1024x1024xf32, strided<[1024, 1]>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = tensor.empty() : tensor<128x256xf32>
    %3 = tensor.empty() : tensor<128x256xf32>
    %4 = arith.divsi %1, %c4_i32 : i32
    %5 = arith.muli %4, %c128_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %subview = memref.subview %reinterpret_cast[%6, 0] [128, 512] [1, 1] : memref<1024x512xf16, strided<[512, 1]>> to memref<128x512xf16, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x512xf16>
    memref.copy %subview, %alloc : memref<128x512xf16, strided<[512, 1], offset: ?>> to memref<128x512xf16>
    %7 = bufferization.to_tensor %alloc restrict : memref<128x512xf16>
    %8 = arith.remsi %1, %c4_i32 : i32
    %9 = arith.muli %8, %c256_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %subview_3 = memref.subview %reinterpret_cast_1[0, %10] [512, 256] [1, 1] : memref<512x1024xf16, strided<[1024, 1]>> to memref<512x256xf16, strided<[1024, 1], offset: ?>>
    %alloc_4 = memref.alloc() : memref<512x256xf16>
    memref.copy %subview_3, %alloc_4 : memref<512x256xf16, strided<[1024, 1], offset: ?>> to memref<512x256xf16>
    %11 = bufferization.to_tensor %alloc_4 restrict : memref<512x256xf16>
    %12 = hivm.hir.mmadL1 ins(%7, %11, %true, %c128, %c512, %c256 : tensor<128x512xf16>, tensor<512x256xf16>, i1, index, index, index) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %13 = hivm.hir.vexp ins(%12 : tensor<128x256xf32>) outs(%3 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %subview_5 = memref.subview %reinterpret_cast_0[%6, %10] [128, 256] [1, 1] : memref<1024x1024xf32, strided<[1024, 1]>> to memref<128x256xf32, strided<[1024, 1], offset: ?>>
    bufferization.materialize_in_destination %12 in writable %subview_5 : (tensor<128x256xf32>, memref<128x256xf32, strided<[1024, 1], offset: ?>>) -> ()
    %subview_6 = memref.subview %reinterpret_cast_2[%6, %10] [128, 256] [1, 1] : memref<1024x1024xf32, strided<[1024, 1]>> to memref<128x256xf32, strided<[1024, 1], offset: ?>>
    bufferization.materialize_in_destination %13 in writable %subview_6 : (tensor<128x256xf32>, memref<128x256xf32, strided<[1024, 1], offset: ?>>) -> ()
    return
  }
}