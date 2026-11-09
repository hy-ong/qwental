class Qwental < Formula
  desc "Metal inference engine and API server for Qwen3.8-Flash-Next on Apple M5 Max"
  homepage "https://github.com/hy-ong/qwental"
  url "https://github.com/hy-ong/qwental/archive/refs/tags/v1.5.0.tar.gz"
  sha256 "5ff3ef081a0cf7153679887d66014c9263f81cb7b41e43b79dec562a5757a561"
  license "MIT"
  head "https://github.com/hy-ong/qwental.git", branch: "main"

  depends_on xcode: ["27.0", :build]
  depends_on arch: :arm64

  def install
    system "make", "build/qwental-serve"
    bin.install "build/qwental-serve"
  end

  def caveats
    <<~EOS
      Model weights are not included. Put the IQ3_S GGUF (two shards of
      ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) in a directory, the DFlash2
      drafter pack in dflash2/ next to it, then run:
        qwental-serve --gguf /path/to/iq3s-gguf --port 8080
      The SSD prefix cache goes to ./cache unless you pass --cache-dir.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/qwental-serve --version")
  end
end
