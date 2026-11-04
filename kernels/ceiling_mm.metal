// 第 0 階段：neural accelerator 矩陣運算量測
// 主程式會在這段原始碼前面加上 #define：
//   TA / TB / TC：左、右、輸出的元素型別；PA / PB：指標型別（4-bit、8-bit 格式型別用 uchar）
//   TM / TN：每個 threadgroup 負責的輸出區塊大小
//   SG：每個 threadgroup 的 SIMD group 數量
// 資料排列（row-major）：A 是 M×K，B 是 K×N，C 是 M×N。
// tensor 的 extent(0) 是最內層（連續）維度，所以 A 的 extents 是 (K, M)。
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mm(device PA *A [[buffer(0)]],
               device PB *B [[buffer(1)]],
               device TC *C [[buffer(2)]],
               constant uint3 &mnk [[buffer(3)]],
               uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = mnk.x, N = mnk.y, K = mnk.z;
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
    tensor<device TB, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(N, K));
    tensor<device TC, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto mA = tA.slice(0, tgid.y * TM);
    auto mB = tB.slice(tgid.x * TN, 0);
    auto mC = tC.slice(tgid.x * TN, tgid.y * TM);
    op.run(mA, mB, mC);
}
