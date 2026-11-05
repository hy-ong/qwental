CC      := clang
CFLAGS  := -std=c23 -O2 -Wall -Wextra -Isrc
OBJC    := -O2 -Wall -Wextra -fobjc-arc -Isrc
LDLIBS  := -framework Metal -framework Foundation -framework IOKit

BUILD := build

.PHONY: all clean ceiling
all: $(BUILD)/ceiling $(BUILD)/mslc $(BUILD)/qmv $(BUILD)/gdn $(BUILD)/stinspect $(BUILD)/ceiling2 $(BUILD)/qmm $(BUILD)/moe_prefill $(BUILD)/qsa $(BUILD)/moe_decode $(BUILD)/gdn_decode $(BUILD)/modelcheck $(BUILD)/run $(BUILD)/tokcheck $(BUILD)/chatcheck $(BUILD)/qwental-serve

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/gpu_metal.o: src/gpu_metal.m src/gpu.h | $(BUILD)
	$(CC) $(OBJC) -c $< -o $@

$(BUILD)/ceiling: bench/ceiling.c $(BUILD)/gpu_metal.o $(wildcard kernels/ceiling_*.metal) src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/ceiling.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

ceiling: $(BUILD)/ceiling
	./$(BUILD)/ceiling

clean:
	rm -rf $(BUILD)

$(BUILD)/mslc: tools/mslc.c $(BUILD)/gpu_metal.o src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) tools/mslc.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/qmv: bench/qmv.c $(BUILD)/gpu_metal.o kernels/qmv.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/qmv.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/gdn: bench/gdn.c $(BUILD)/gpu_metal.o kernels/gdn.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/gdn.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/stinspect: tools/stinspect.c src/st.c src/st.h | $(BUILD)
	$(CC) $(CFLAGS) tools/stinspect.c src/st.c -o $@

$(BUILD)/ceiling2: bench/ceiling2.c $(BUILD)/gpu_metal.o kernels/ceiling_mm2.metal kernels/ceiling_mem.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/ceiling2.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/qmm: bench/qmm.c $(BUILD)/gpu_metal.o kernels/qmm.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/qmm.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/moe_prefill: bench/moe_prefill.c $(BUILD)/gpu_metal.o kernels/qmm.metal kernels/moe_prefill.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/moe_prefill.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/qsa: bench/qsa.c $(BUILD)/gpu_metal.o kernels/qsa.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/qsa.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/moe_decode: bench/moe_decode.c $(BUILD)/gpu_metal.o kernels/moe_decode.metal kernels/ceiling_mem.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/moe_decode.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/gdn_decode: bench/gdn_decode.c $(BUILD)/gpu_metal.o kernels/gdn_decode.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/gdn_decode.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/modelcheck: tools/modelcheck.c src/model.c src/model.h src/st.c src/st.h src/gguf.c src/gguf.h | $(BUILD)
	$(CC) $(CFLAGS) tools/modelcheck.c src/model.c src/st.c src/gguf.c -o $@

ENGINE_SRC := src/engine.c src/model.c src/st.c src/tok.c src/json.c src/requant.c src/dflash.c src/gguf.c src/ggml_dq.c
ENGINE_DEP := $(ENGINE_SRC) src/engine.h src/model.h src/st.h src/gpu.h kernels/engine.metal kernels/qmv.metal kernels/moe_decode.metal kernels/gdn_decode.metal kernels/qsa.metal kernels/qsa_pf.metal kernels/qmm_a8.metal kernels/qmm_i8.metal kernels/moe_iq.metal kernels/iq_tables.metal kernels/prefill.metal kernels/qmm.metal kernels/gdn.metal kernels/moe_prefill.metal kernels/qmvb.metal kernels/qmvb_tc.metal kernels/moe_tc.metal kernels/qmv_gg.metal kernels/gg_ple.metal kernels/hyper_gg.metal src/tok.c src/tok.h src/json.c src/json.h src/unicode_tab.h src/requant.h src/dflash.h src/gguf.h src/ggml_dq.h kernels/dflash.metal kernels/df_q4.metal kernels/dn_proj.metal kernels/qmv_ggm.metal kernels/moe_gg.metal

$(BUILD)/run: tools/run.c $(ENGINE_DEP) $(BUILD)/gpu_metal.o | $(BUILD)
	$(CC) $(CFLAGS) tools/run.c $(ENGINE_SRC) $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/tokcheck: tools/tokcheck.c src/tok.c src/tok.h src/unicode_tab.h $(BUILD)/gpu_metal.o | $(BUILD)
	$(CC) $(CFLAGS) tools/tokcheck.c src/tok.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/chatcheck: tools/chatcheck.c src/chat.c src/chat.h src/json.c src/json.h | $(BUILD)
	$(CC) $(CFLAGS) tools/chatcheck.c src/chat.c src/json.c -o $@

$(BUILD)/qwental-serve: tools/serve.c src/chat.c src/chat.h $(ENGINE_DEP) $(BUILD)/gpu_metal.o | $(BUILD)
	$(CC) $(CFLAGS) tools/serve.c src/chat.c $(ENGINE_SRC) $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/dispatch: bench/dispatch.c $(BUILD)/gpu_metal.o src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/dispatch.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/dqtest: tools/dqtest.c $(BUILD)/gpu_metal.o kernels/moe_iq.metal kernels/iq_tables.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) tools/dqtest.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/ggmtest: tools/ggmtest.c $(BUILD)/gpu_metal.o kernels/qmv_ggm.metal kernels/moe_iq.metal kernels/iq_tables.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) tools/ggmtest.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/moe_bw: bench/moe_bw.c $(BUILD)/gpu_metal.o kernels/moe_bw.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/moe_bw.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)

$(BUILD)/dqcpu: tools/dqcpu.c src/ggml_dq.c src/ggml_dq.h | $(BUILD)
	$(CC) $(CFLAGS) tools/dqcpu.c src/ggml_dq.c -o $@

$(BUILD)/qmm_i8: bench/qmm_i8.c $(BUILD)/gpu_metal.o kernels/qmm_i8.metal src/gpu.h | $(BUILD)
	$(CC) $(CFLAGS) bench/qmm_i8.c $(BUILD)/gpu_metal.o -o $@ $(LDLIBS)
