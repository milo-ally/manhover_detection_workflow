#include <stdio.h>
#include "ax_base_type.h"

// 偽造的 Start 函式，直接回傳 0 (成功)
AX_S32 COMMON_TCM_StartOutsideDev(void) {
    // printf("[Warning] COMMON_TCM_StartOutsideDev is mocked (doing nothing).\n");
    return 0;
}

// 偽造的 Stop 函式，直接回傳 0 (成功)
AX_S32 COMMON_TCM_StopOutsideDev(void) {
    return 0;
}
