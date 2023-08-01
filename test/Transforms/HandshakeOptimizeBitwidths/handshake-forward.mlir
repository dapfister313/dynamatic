// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-optimize-bitwidths %s --split-input-file | FileCheck %s

handshake.func @forkFW(%arg0: i16, %start: none) -> (i32, i32) {
  %ext0 = arith.extsi %arg0 : i16 to i32
  %results:2 = fork [2] %ext0 : i32
  %returnVals:2 = d_return %results#0, %results#1 : i32, i32
  end %returnVals#0, %returnVals#1 : i32, i32
}

// -----

handshake.func @lazyForkFW(%arg0: i16, %start: none) -> (i32, i32) {
  %ext0 = arith.extsi %arg0 : i16 to i32
  %results:2 = lazy_fork [2] %ext0 : i32
  %returnVals:2 = d_return %results#0, %results#1 : i32, i32
  end %returnVals#0, %returnVals#1 : i32, i32
}

// -----

handshake.func @mergeFW(%arg0: i8, %arg1: i16, %start: none) -> i32 {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %ext1 = arith.extsi %arg1 : i16 to i32
  %merge = merge %ext0, %ext1 : i32
  %returnVal = d_return %merge : i32
  end %returnVal : i32
}

// -----

handshake.func @branchFW(%arg0: i16, %start: none) -> i32 {
  %ext0 = arith.extsi %arg0 : i16 to i32
  %branch = br %ext0 : i32
  %returnVal = d_return %branch : i32
  end %returnVal : i32
}

// -----

handshake.func @cmergeFW(%arg0: i8, %arg1: i16, %start: none) -> (i32, i8) {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %ext1 = arith.extsi %arg1 : i16 to i32
  %merge, %index = control_merge %ext0, %ext1 : i32, i8
  %returnVals:2 = d_return %merge, %index : i32, i8
  end %returnVals#0, %returnVals#1 : i32, i8
}

// -----

handshake.func @muxFW(%arg0: i8, %arg1: i16, %index: i8, %start: none) -> i32 {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %ext1 = arith.extsi %arg1 : i16 to i32
  %mux = mux %index [%ext0, %ext1] : i8, i32
  %returnVal = d_return %mux : i32
  end %returnVal : i32
}

// -----

handshake.func @condBrFw(%arg0: i16, %cond: i1, %start: none) -> (i32, i32) {
  %ext0 = arith.extsi %arg0 : i16 to i32
  %true, %false = cond_br %cond, %ext0 : i32
  %returnVals:2 = d_return %true, %false : i32, i32
  end %returnVals#0, %returnVals#1 : i32, i32
}

// -----

handshake.func @bufferFW(%arg0: i16, %start: none) -> i32 {
  %ext0 = arith.extsi %arg0 : i16 to i32
  %buf = buffer [2] seq %ext0 : i32
  %returnVal = d_return %buf : i32
  end %returnVal : i32
}

// -----

handshake.func @cmergeToMuxFW(%arg0: i32, %arg1: i32, %start: none) -> i32 {
  %result, %index = control_merge %arg0, %arg1 : i32, i32
  %mux = mux %index [%arg0, %arg1] : i32, i32
  %returnVal = d_return %mux : i32
  end %returnVal : i32
}