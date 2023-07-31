// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-optimize-bitwidths %s --split-input-file | FileCheck %s

handshake.func @forkBW(%arg0: i32, %start: none) -> (i16, i8) {
  %results:2 = fork [2] %arg0 : i32
  %trunc0 = arith.trunci %results#0 : i32 to i16
  %trunc1 = arith.trunci %results#1 : i32 to i8
  %returnVals:2 = d_return %trunc0, %trunc1 : i16, i8
  end %returnVals#0, %returnVals#1 : i16, i8
}

// -----

handshake.func @lazyForkBW(%arg0: i32, %start: none) -> (i16, i8) {
  %results:2 = lazy_fork [2] %arg0 : i32
  %trunc0 = arith.trunci %results#0 : i32 to i16
  %trunc1 = arith.trunci %results#1 : i32 to i8
  %returnVals:2 = d_return %trunc0, %trunc1 : i16, i8
  end %returnVals#0, %returnVals#1 : i16, i8
}

// -----

handshake.func @mergeBW(%arg0: i32, %arg1: i32, %start: none) -> i16 {
  %merge = merge %arg0, %arg1 : i32
  %trunc = arith.trunci %merge : i32 to i16
  %returnVal = d_return %trunc : i16
  end %returnVal : i16
}

// -----

handshake.func @branchBW(%arg0: i32, %start: none) -> i16 {
  %branch = br %arg0 : i32
  %trunc = arith.trunci %branch : i32 to i16
  %returnVal = d_return %trunc : i16
  end %returnVal : i16
}

// -----

handshake.func @cmergeBW(%arg0: i32, %arg1: i32, %start: none) -> (i16, i16) {
  %merge, %index = control_merge %arg0, %arg1 : i32, i32
  %truncMerge = arith.trunci %merge : i32 to i16
  %truncIndex = arith.trunci %index : i32 to i16
  %returnVals:2 = d_return %truncMerge, %truncIndex : i16, i16
  end %returnVals#0, %returnVals#1 : i16, i16
}

// -----

handshake.func @muxBW(%arg0: i32, %arg1: i32, %index: i32, %start: none) -> i16 {
  %mux = mux %index [%arg0, %arg1] : i32, i32
  %trunc = arith.trunci %mux : i32 to i16
  %returnVal = d_return %trunc : i16
  end %returnVal : i16
}

// -----

handshake.func @condBrBW(%arg0: i32, %cond: i1, %start: none) -> (i16, i8) {
  %true, %false = cond_br %cond, %arg0 : i32
  %truncTrue = arith.trunci %true : i32 to i16
  %truncFalse = arith.trunci %false : i32 to i8
  %returnVals:2 = d_return %truncTrue, %truncFalse : i16, i8
  end %returnVals#0, %returnVals#1 : i16, i8
}

// -----

handshake.func @bufferBW(%arg0: i32, %start: none) -> i16 {
  %buf = buffer [2] seq %arg0 : i32
  %trunc = arith.trunci %buf : i32 to i16
  %returnVal = d_return %trunc : i16
  end %returnVal : i16
}
