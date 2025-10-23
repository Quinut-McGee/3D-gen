module.exports = {
  apps : [{
    name: 'generation-competitive',
    script: 'serve_competitive.py',
    interpreter: '/home/kobe/miniconda3/envs/three-gen-mining/bin/python',
    args: '--port 8093 --config configs/text_mv_fast.yaml --flux-steps 20 --validation-threshold 0.6 --enable-validation',
    env: {
      CUDA_VISIBLE_DEVICES: '0',           // Use only GPU 0 (RTX 4090)
      PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8',  // Optimize VRAM usage, reduce fragmentation, aggressive GC
      OMP_NUM_THREADS: '8',                // Optimize CPU threads
      MKL_NUM_THREADS: '8',
      PYTORCH_NO_CUDA_MEMORY_CACHING: '0',  // Keep caching enabled but with better management
      // CUDA compilation environment - using system CUDA 12.1
      CUDA_HOME: '/usr/local/cuda',
      CUDA_PATH: '/usr/local/cuda',        // Some tools check CUDA_PATH instead of CUDA_HOME
      PATH: '/usr/local/cuda/bin:' + process.env.PATH,
      LD_LIBRARY_PATH: '/usr/local/cuda/lib64:/usr/local/cuda/lib:' + (process.env.LD_LIBRARY_PATH || ''),
      C_INCLUDE_PATH: '/usr/local/cuda/include',       // Prioritize system CUDA headers
      CPLUS_INCLUDE_PATH: '/usr/local/cuda/include',   // Prioritize system CUDA headers for C++
      TORCH_CUDA_ARCH_LIST: '8.9',         // RTX 4090 compute capability
      FORCE_CUDA: '1',                      // Force CUDA compilation for extensions
      MAX_JOBS: '4',                        // Limit parallel compilation jobs
      TORCH_NVCC_FLAGS: '-Xfatbin -compress-all',  // Optimize CUDA binary compression
      NVCC_APPEND_FLAGS: '-Xcudafe --diag_suppress=20012',  // Suppress warning #20012 (ignored annotations)
      CXXFLAGS: '-I/usr/local/cuda/include',
      CUDAFLAGS: '-I/usr/local/cuda/include'
    }
  }]
};
