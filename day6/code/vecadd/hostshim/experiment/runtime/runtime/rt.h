// Local shim: CANN 9.0.0 install exposes rt.h under pkg_inc/runtime/runtime/,
// while the VecAdd test includes the source-tree path experiment/runtime/runtime/rt.h.
#include "runtime/runtime/rt.h"
